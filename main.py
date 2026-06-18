from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import math

app = FastAPI(title="HWC Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ──────────────────────────────────────────────────
model    = joblib.load("P5model.pkl")
scaler   = joblib.load("P5scaler.pkl")
FEATURES = joblib.load("P5feature_columns.pkl")

# ── Real HWC feature data ───────────────────────────────────────
HWC_REAL_FEATURES = {
    (11.93, 76.13): {"name": "Nagarhole",       "f": [0.82,0.45,900, 12,0.1,0.3,0.5]},
    (11.67, 76.63): {"name": "Bandipur",        "f": [0.79,0.40,850, 14,0.2,0.4,0.6]},
    (11.61, 76.13): {"name": "Wayanad",         "f": [0.85,0.50,780, 18,0.1,0.2,0.4]},
    (11.56, 76.52): {"name": "Mudumalai",       "f": [0.80,0.42,920, 16,0.2,0.3,0.5]},
    (11.98, 77.05): {"name": "BRT Hills",       "f": [0.76,0.38,1100,22,0.3,0.5,0.7]},
    (11.50, 77.23): {"name": "Sathyamangalam",  "f": [0.74,0.36,800, 15,0.3,0.6,0.8]},
    (12.94, 75.78): {"name": "Sakleshpur",      "f": [0.83,0.47,960, 20,0.2,0.3,0.4]},
    (12.42, 75.74): {"name": "Kodagu",          "f": [0.86,0.52,1050,25,0.1,0.2,0.3]},
    (10.57, 76.93): {"name": "Anamalai",        "f": [0.81,0.44,870, 17,0.2,0.4,0.6]},
    (11.41, 76.73): {"name": "Nilgiris",        "f": [0.78,0.41,1200,28,0.2,0.3,0.5]},
    (13.65, 75.62): {"name": "Bhadra WLS",      "f": [0.77,0.39,820, 14,0.2,0.4,0.6]},
    (12.10, 77.20): {"name": "Cauvery WLS",     "f": [0.75,0.37,750, 12,0.3,0.5,0.7]},
    (12.63, 75.72): {"name": "Pushpagiri",      "f": [0.84,0.48,1100,24,0.1,0.3,0.4]},
    (13.15, 75.25): {"name": "Kudremukh",       "f": [0.88,0.55,1200,30,0.1,0.2,0.3]},
    (14.00, 74.80): {"name": "Sharavathi",      "f": [0.80,0.50,670, 16,0.2,0.2,0.5]},
    (12.97, 77.59): {"name": "Bengaluru",       "f": [0.18,0.10,920,  2,8.0,5.0,0.5]},
    (12.30, 76.64): {"name": "Mysuru",          "f": [0.22,0.12,770,  3,5.0,3.0,0.4]},
    (12.87, 74.88): {"name": "Mangalore",       "f": [0.30,0.35, 10,  1,6.0,0.5,0.3]},
    (13.00, 76.10): {"name": "Hassan",          "f": [0.35,0.20,980,  5,4.0,2.0,0.6]},
}

def get_nearest(lat, lon, threshold=0.3):
    best_dist = float('inf')
    best      = None
    for (klat, klon), val in HWC_REAL_FEATURES.items():
        dist = math.sqrt((lat - klat)**2 + (lon - klon)**2)
        if dist < best_dist:
            best_dist = dist
            best      = val
    if best_dist <= threshold:
        return best
    return None

def build_features(lat, lon, NDVI, NDWI, elevation, slope,
                   dist_forest, dist_water, dist_road):
    VWR                   = NDVI / (NDWI + 0.01)
    TRI                   = slope * 0.5
    NLD                   = min(1, (lon - 75) / 5)
    HAS                   = min(1, (lat - 10) / 5)
    ESI                   = (NDVI + NDWI) / 2
    NDVI_NDWI_interaction = NDVI * NDWI
    veg_water_risk        = NDVI / (dist_water + 0.1)
    isolation_index       = (dist_forest + dist_road) / 2
    terrain_ratio         = slope / (elevation + 1)
    human_pressure        = (HAS + NLD) / 2
    eco_stress            = (NDVI + NDWI + HAS) / 3
    slope_elev_risk       = (slope * elevation) / 1000
    return [lat, lon, NDVI, NDWI, elevation, slope, dist_forest, dist_water,
            dist_road, VWR, TRI, NLD, HAS, ESI, NDVI_NDWI_interaction,
            veg_water_risk, isolation_index, terrain_ratio, human_pressure,
            eco_stress, slope_elev_risk]

def run_prediction(lat, lon):
    nearest = get_nearest(lat, lon)
    if nearest:
        NDVI, NDWI, elevation, slope, df, dw, dr = nearest["f"]
        location = nearest["name"]
    else:
        NDVI      = max(0.05, min(0.95, 0.3 + ((lat - 10) / 5) * 0.6))
        NDWI      = max(0.05, min(0.65, 0.2 + ((lon - 75) / 5) * 0.6))
        elevation = 100 + (lat - 10) * 200
        slope     = (lon - 75) * 8
        df        = abs(12.5 - lat)
        dw        = abs(77.5 - lon)
        dr        = abs(lat - lon / 10)
        location  = f"({lat:.2f}, {lon:.2f})"

    feats = build_features(lat, lon, NDVI, NDWI, elevation, slope, df, dw, dr)
    dfx   = pd.DataFrame([feats], columns=FEATURES)
    prob  = float(model.predict_proba(scaler.transform(dfx))[0][1] * 100)
    risk  = "HIGH" if prob >= 70 else "MEDIUM" if prob >= 40 else "LOW"

    return {
        "risk":        risk,
        "probability": round(prob, 2),
        "location":    location,
        "lat":         lat,
        "lon":         lon
    }

# ── Routes ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

# GET route — works directly in browser
@app.get("/predict")
def predict_get(lat: float, lon: float):
    return run_prediction(lat, lon)

# POST route — used by Flutter app
class PredictRequest(BaseModel):
    lat: float
    lon: float

@app.post("/predict")
def predict_post(req: PredictRequest):
    return run_prediction(req.lat, req.lon)

# Root — shows usage instructions
@app.get("/")
def root():
    return {
        "name": "HWC Prediction API",
        "usage": {
            "health_check": "/health",
            "predict_browser": "/predict?lat=11.93&lon=76.13",
            "predict_api": "POST /predict with body {lat, lon}"
        },
        "sample_high_risk": [
            {"name": "Nagarhole",  "lat": 11.93, "lon": 76.13},
            {"name": "Bandipur",   "lat": 11.67, "lon": 76.63},
            {"name": "Wayanad",    "lat": 11.61, "lon": 76.13},
            {"name": "Kudremukh", "lat": 13.15, "lon": 75.25},
        ]
    }
