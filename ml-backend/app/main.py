import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import traceback
import requests
import json

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
# 2️⃣ LOAD LABEL MAP
# ==========================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_MAP_PATH = os.path.join(BASE_DIR, "models", "label_map.json")


with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

idx_to_label = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)

print("✅ Classes:", idx_to_label)

# ==========================================================
# 3️⃣ LOAD MOBILE NET MODEL
# ==========================================================
MODEL_PATH = os.path.join(BASE_DIR, "models", "skin_model.pth")

image_model = None

try:
    print("🔁 Loading MobileNet model...")

    image_model = models.mobilenet_v2(weights=None)
    image_model.classifier[1] = nn.Linear(
        image_model.classifier[1].in_features,
        NUM_CLASSES
    )

    image_model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    )

    image_model.eval()
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Model loading failed: {e}")

# ==========================================================
# 4️⃣ IMAGE PREPROCESSING
# ==========================================================
image_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================================================
# 5️⃣ ROOT
# ==========================================================
@app.get("/")
def root():
    return {"message": "✅ Skin Disease Detection API running!"}

# ==========================================================
# 6️⃣ IMAGE CLASSIFICATION
# ==========================================================
@app.post("/classify-image")
async def classify_image_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        if not image_model:
            return {"error": "Model not loaded"}

        input_tensor = image_preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = image_model(input_tensor)
            probs = F.softmax(output[0], dim=0)

        top_prob, top_idx = torch.topk(probs, 3)

        predictions = [
            {
                "label": idx_to_label[top_idx[i].item()],
                "confidence": round(top_prob[i].item(), 4)
            }
            for i in range(len(top_prob))
        ]

        return {
            "filename": file.filename,
            "predictions": predictions
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

# ==========================================================
# 7️⃣ CHAT (UNCHANGED)
# ==========================================================
HF_TOKEN = "xxxxxxxxx"
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"

@app.post("/chat")
async def chat_endpoint(prompt: str = Form(...)):
    try:
        llama_prompt = f"""[INST] <<SYS>>
You are a helpful assistant providing information about skin conditions.
Do not provide medical advice.
<</SYS>>

{prompt} [/INST]"""

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "inputs": llama_prompt,
                "parameters": {
                    "max_new_tokens": 250,
                    "return_full_text": False,
                }
            }
        )

        data = response.json()

        if isinstance(data, list) and "generated_text" in data[0]:
            answer = data[0]["generated_text"]
        else:
            answer = str(data)

        return {"response": answer.strip()}

    except Exception as e:
        return {"error": str(e)}