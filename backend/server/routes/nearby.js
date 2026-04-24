const express = require("express");
const axios = require("axios");
const router = express.Router();

// FREE Overpass API
router.get("/dermatologists", async (req, res) => {
  try {
    const { lat, lng } = req.query;

    if (!lat || !lng) {
      return res.status(400).json({ message: "lat & lng required" });
    }

    // Overpass query (5km radius)
    const query = `
      [out:json];
      (
        node["healthcare"="doctor"](around:5000,${lat},${lng});
        node["amenity"="hospital"](around:5000,${lat},${lng});
      );
      out;
    `;

    const response = await axios.post(
      "https://overpass-api.de/api/interpreter",
      query,
      { headers: { "Content-Type": "text/plain" } }
    );

    const places = response.data.elements.map((place, index) => ({
      _id: place.id || index,
      name: place.tags?.name || "Dermatology Clinic",
      address: place.tags?.["addr:full"] || "Nearby location",
      rating: "N/A",
      review_count: 0,
      distance: "Nearby",
      map_url: `https://www.openstreetmap.org/?mlat=${place.lat}&mlon=${place.lon}`
    }));

    res.json(places);

  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error fetching places" });
  }
});

module.exports = router;