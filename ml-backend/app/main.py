import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import Dinov2ForImageClassification
import traceback
import requests

# ==========================================================
# 1️⃣ FASTAPI SETUP
# ==========================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# 2️⃣ IMAGE CLASSIFIER (LOCAL)
# ==========================================================
class DinoV2Classifier(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", num_classes=23):
        super(DinoV2Classifier, self).__init__()
        self.dinov2 = Dinov2ForImageClassification.from_pretrained(model_name)
        classifier_in_features = self.dinov2.classifier.in_features
        self.dinov2.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_in_features, num_classes)
        )

    def forward(self, x):
        outputs = self.dinov2(x)
        return outputs.logits

# ==========================================================
# 3️⃣ LOAD IMAGE MODEL
# ==========================================================
IMAGE_MODEL_PATH = "app/models/image_classifier.pth"
NUM_CLASSES = 23
CLASS_NAMES = [
    'Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell ...', 'Atopic Dermatitis Photos',
    'Bullous Disease Photos', 'Cellulitis Impetigo and other ...', 'Eczema Photos',
    'Exanthems and Drug Eruption ...', 'Hair Loss Photos Alopecia a...', 'Herpes HPV and other STDs',
    'Light Diseases and Disorder...', 'Lupus and other Connective ...', 'Melanoma Skin Cancer Nevi',
    'Nail Fungus and other Nail Di...', 'Poison Ivy Photos and other ...', 'Psoriasis pictures Lichen Pla...',
    'Scabies Lyme Disease and ot...', 'Seborrheic Keratoses and ot...', 'Systemic Disease',
    'Tinea Ringworm Candidiasis', 'Urticaria Hives', 'Vascular Tumors',
    'Vasculitis Photos', 'Warts Molluscum and other ...'
]

image_model = None
try:
    print("🔁 Loading DINOv2 image classifier...")
    image_model = DinoV2Classifier(model_name="facebook/dinov2-base", num_classes=NUM_CLASSES)
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=torch.device('cpu')))
    image_model.eval()
    print("✅ DINOv2 Image Classifier loaded successfully!")
except Exception as e:
    print(f"❌ Could not load image classifier: {e}")
    print("ℹ️ Note: This is expected if you don't have the 'models/image_classifier.pth' file locally.")


# ==========================================================
# 4️⃣ HF ONLINE CHAT MODEL
# ==========================================================
HF_TOKEN = "xxxxxxxxx"  # 🔑 IMPORTANT: Make sure this is the token from the account that was granted access!
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf" # <-- Switching back now that you have access
# HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1" # <-- This was the previous model

# ==========================================================
# 5️⃣ IMAGE PREPROCESSING
# ==========================================================
image_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================================
# 6️⃣ ROUTES
# ==========================================================
@app.get("/")
def root():
    return {"message": "✅ SkinAid ML Backend (DINOv2 + Hugging Face Llama 2) running successfully!"}

@app.post("/classify-image")
async def classify_image_endpoint(file: UploadFile = File(...)):
    try:
        print(f"🖼️ Classifying image: {file.filename}")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        if not image_model:
            print("⚠️ Image model not loaded, providing demo predictions")
            # Demo predictions when model isn't available
            demo_predictions = [
                {"label": "Acne and Rosacea Photos", "confidence": 0.85},
                {"label": "Eczema Photos", "confidence": 0.10},
                {"label": "Atopic Dermatitis Photos", "confidence": 0.03},
                {"label": "Psoriasis pictures Lichen Plan...", "confidence": 0.01},
                {"label": "Other Skin Condition", "confidence": 0.01}
            ]
            return {"filename": file.filename, "predictions": demo_predictions, "note": "Demo predictions - model loading in progress"}
        
        input_tensor = image_preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = image_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top_prob, top_indices = torch.topk(probabilities, 5)
        predictions = [
            {"label": CLASS_NAMES[top_indices[i]], "confidence": round(top_prob[i].item(), 4)}
            for i in range(top_prob.size(0))
        ]
        print("✅ Image classification completed successfully.")
        return {"filename": file.filename, "predictions": predictions}

    except Exception as e:
        print(f"❌ Classification error: {str(e)}")
        print(traceback.format_exc())
        return {"error": f"Error during image classification: {str(e)}"}

@app.post("/chat")
async def chat_endpoint(prompt: str = Form(...)):
    try:
        print(f"🧠 User prompt: {prompt}")
        print(f"🔁 Sending request to Hugging Face Inference API for model: {HF_MODEL}")

        # Llama 2 Chat models work best with a specific prompt structure.
        # We add the [INST] and <<SYS>> tags to guide the model.
        llama_prompt = f"""[INST] <<SYS>>
You are a helpful, respectful, and honest assistant providing information about skin conditions.
Answer concisely and in a friendly manner. Do not provide medical advice.
<</SYS>>

{prompt} [/INST]"""

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": llama_prompt, # Use the formatted Llama 2 prompt
                # It's good practice to add parameters to control the output
                "parameters": {
                    "max_new_tokens": 250,
                    "return_full_text": False, # We just want the model's response
                }
            }
        )

        print(f"🔁 HF API Response Code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ HF API Error: {response.text}")
            # Check for model loading error (503)
            if response.status_code == 503:
                return {"error": f"HF API Error: Model '{HF_MODEL}' is currently loading. Please try again in a few moments."}
            return {"error": f"HF API Error: {response.text}"}

        data = response.json()
        print("✅ HF API Response received.")

        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            answer = data[0]["generated_text"]
        else:
            # Handle other possible valid responses or errors
            answer = str(data)

        print("🤖 Final response:", answer)
        return {"response": answer.strip()}

    except Exception as e:
        print("❌ Exception:", e)
        print(traceback.format_exc())
        return {"error": str(e)}

# ==========================================================
# 7️⃣ RUN SERVER
# ==========================================================
# Run using: uvicorn main:app --reload

