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

model    = joblib.load("P5model.pkl")
scaler   = joblib.load("P5scaler.pkl")
FEATURES = joblib.load("P5feature_columns.pkl")

# ── Known URBAN zones → always LOW risk ─────────────────────────
URBAN_ZONES = [
    (12.90, 77.50, 0.5, "Bengaluru"),
    (12.97, 77.59, 0.5, "Bengaluru"),
    (12.80, 77.40, 0.4, "Bengaluru"),
    (13.00, 77.60, 0.4, "Bengaluru"),
    (13.10, 77.50, 0.4, "Bengaluru North"),
    (12.70, 77.50, 0.4, "Bengaluru South"),
    (12.30, 76.64, 0.4, "Mysuru"),
    (12.87, 74.88, 0.3, "Mangaluru"),
    (13.00, 76.10, 0.3, "Hassan"),
    (11.02, 76.96, 0.3, "Coimbatore"),
    (10.52, 76.21, 0.3, "Thrissur"),
    (13.34, 77.10, 0.3, "Tumkur"),
    (12.52, 76.90, 0.3, "Mandya"),
    (11.34, 77.72, 0.3, "Erode"),
    (11.65, 78.15, 0.3, "Salem"),
    (10.00, 77.00, 0.3, "Madurai"),
    (13.93, 75.57, 0.3, "Shimoga"),
    (14.47, 75.92, 0.3, "Davangere"),
    (15.13, 75.71, 0.3, "Hubli"),
    (10.80, 76.65, 0.3, "Palakkad"),
    (11.25, 75.77, 0.3, "Kozhikode"),
    (10.00, 76.96, 0.3, "Kochi"),
    (8.50,  76.95, 0.3, "Thiruvananthapuram"),
]

# ── Known FOREST/HIGH RISK zones ────────────────────────────────
FOREST_ZONES = [
    (11.9, 76.1, 0.8,  0.82, 900,  12, 0.2, 0.4, 1.5, "Nagarhole"),
    (11.6, 76.4, 0.9,  0.80, 860,  15, 0.2, 0.3, 1.8, "Bandipur"),
    (11.6, 76.1, 0.8,  0.85, 790,  18, 0.1, 0.2, 1.6, "Wayanad"),
    (12.4, 75.7, 0.7,  0.84, 1000, 22, 0.2, 0.3, 2.0, "Kodagu"),
    (13.1, 75.3, 0.6,  0.86, 1100, 26, 0.2, 0.3, 2.5, "Kudremukh"),
    (11.9, 77.0, 0.7,  0.76, 1050, 20, 0.3, 0.5, 2.0, "BRT Hills"),
    (11.5, 77.2, 0.8,  0.74, 820,  16, 0.3, 0.5, 2.2, "Sathyamangalam"),
    (10.5, 76.9, 0.7,  0.80, 880,  17, 0.2, 0.4, 2.0, "Anamalai"),
    (11.4, 76.7, 0.6,  0.78, 1100, 24, 0.2, 0.3, 2.5, "Nilgiris"),
    (13.6, 75.6, 0.5,  0.77, 830,  14, 0.2, 0.4, 2.0, "Bhadra"),
    (12.6, 75.7, 0.5,  0.83, 1000, 22, 0.2, 0.3, 2.0, "Pushpagiri"),
    (14.0, 74.8, 0.6,  0.80, 680,  16, 0.2, 0.2, 2.5, "Sharavathi"),
    (13.4, 75.1, 0.5,  0.85, 820,  22, 0.2, 0.3, 2.5, "Agumbe"),
    (14.6, 74.8, 0.5,  0.78, 600,  18, 0.3, 0.4, 3.0, "Sirsi"),
    (11.2, 77.5, 0.5,  0.79, 940,  20, 0.2, 0.3, 2.5, "Kalakad"),
    (12.0, 75.5, 0.6,  0.80, 850,  18, 0.2, 0.3, 2.0, "Coorg Buffer"),
    (11.0, 76.5, 0.6,  0.78, 820,  16, 0.2, 0.4, 2.0, "Palakkad Gap"),
    (10.8, 76.7, 0.5,  0.76, 750,  14, 0.3, 0.4, 2.0, "Silent Valley"),
]

def is_urban(lat, lon):
    """Check if coordinate is in a known urban zone"""
    for ulat, ulon, radius, name in URBAN_ZONES:
        dist = math.sqrt((lat - ulat)**2 + (lon - ulon)**2)
        if dist <= radius:
            return True, name
    return False, None

def estimate_features(lat, lon):
    in_ghats = (74.0 <= lon <= 77.5) and (10.0 <= lat <= 14.5)

    best_dist = float('inf')
    best_zone = None
    for zone in FOREST_ZONES:
        dist = math.sqrt((lat - zone[0])**2 + (lon - zone[1])**2)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone

    if best_dist <= best_zone[2]:
        blend = 1.0 - (best_dist / best_zone[2])
        NDVI  = best_zone[3] * blend + 0.18 * (1 - blend)
        NDWI  = 0.35 * blend + 0.08 * (1 - blend)
        elev  = best_zone[4] * blend + 300  * (1 - blend)
        slope = best_zone[5] * blend + 2    * (1 - blend)
        df    = best_zone[6] * blend + 9.0  * (1 - blend)
        dw    = best_zone[7] * blend + 5.0  * (1 - blend)
        dr    = best_zone[8] * blend + 0.4  * (1 - blend)
    elif in_ghats:
        dist_factor = min(1.0, best_dist / 2.0)
        NDVI  = 0.60 - dist_factor * 0.25
        NDWI  = 0.30 - dist_factor * 0.10
        elev  = 600  - dist_factor * 300
        slope = 12   - dist_factor * 8
        df    = 1.5  + dist_factor * 3.0
        dw    = 0.8  + dist_factor * 2.0
        dr    = 2.0  - dist_factor * 1.0
    else:
        NDVI  = 0.18; NDWI = 0.08; elev = 400; slope = 2
        df    = 9.0;  dw   = 5.0;  dr   = 0.4

    return NDVI, NDWI, elev, slope, df, dw, dr

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

def get_location_name(lat, lon):
    places = [
        (11.93, 76.13, "Nagarhole Forest"),
        (11.67, 76.63, "Bandipur Tiger Reserve"),
        (11.61, 76.13, "Wayanad Wildlife Sanctuary"),
        (11.56, 76.52, "Mudumalai Tiger Reserve"),
        (13.15, 75.25, "Kudremukh National Park"),
        (12.42, 75.74, "Kodagu Forest"),
        (11.98, 77.05, "BRT Tiger Reserve"),
        (11.50, 77.23, "Sathyamangalam Tiger Reserve"),
        (10.57, 76.93, "Anamalai Tiger Reserve"),
        (11.41, 76.73, "Nilgiris Biosphere"),
        (13.65, 75.62, "Bhadra Wildlife Sanctuary"),
        (13.40, 75.10, "Agumbe Rainforest"),
        (14.60, 74.84, "Sirsi Forest"),
        (10.80, 76.70, "Silent Valley"),
        (12.90, 77.50, "Bengaluru"),
        (12.97, 77.59, "Bengaluru"),
        (12.30, 76.64, "Mysuru"),
        (12.87, 74.88, "Mangaluru"),
        (13.00, 76.10, "Hassan"),
        (11.02, 76.96, "Coimbatore"),
        (10.52, 76.21, "Thrissur"),
    ]
    best_dist = float('inf')
    best_name = f"({lat:.2f}, {lon:.2f})"
    for plat, plon, name in places:
        dist = math.sqrt((lat - plat)**2 + (lon - plon)**2)
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_dist > 0.5:
        best_name = f"({lat:.2f}, {lon:.2f})"
    return best_name

def run_prediction(lat: float, lon: float):
    # Check urban zone first — always LOW
    urban, urban_name = is_urban(lat, lon)
    if urban:
        return {
            "risk":        "LOW",
            "probability": 5.0,
            "location":    urban_name,
            "driver":      "human_pressure",
            "lat":         lat,
            "lon":         lon,
        }

    # Otherwise use model
    NDVI, NDWI, elev, slope, df, dw, dr = estimate_features(lat, lon)
    feats = build_features(lat, lon, NDVI, NDWI, elev, slope, df, dw, dr)
    dfx   = pd.DataFrame([feats], columns=FEATURES)
    prob  = float(model.predict_proba(scaler.transform(dfx))[0][1] * 100)
    risk  = "HIGH" if prob >= 70 else "MEDIUM" if prob >= 40 else "LOW"
    location = get_location_name(lat, lon)
    driver = "NDVI" if NDVI > 0.65 else "dist_forest"

    return {
        "risk":        risk,
        "probability": round(prob, 2),
        "location":    location,
        "driver":      driver,
        "lat":         lat,
        "lon":         lon,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"name": "HWC Prediction API", "status": "live"}

@app.get("/predict")
def predict_get(lat: float, lon: float):
    return run_prediction(lat, lon)

class PredictRequest(BaseModel):
    lat: float
    lon: float

@app.post("/predict")
def predict_post(req: PredictRequest):
    return run_prediction(req.lat, req.lon)
