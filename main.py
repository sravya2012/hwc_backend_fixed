from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

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

# ── Feature computation from lat/lon ────────────────────────────
# Model was trained on ecological patterns:
# HIGH risk  → high NDVI, high elevation, steep slope, close to forest, far from road
# LOW risk   → low NDVI, flat, far from forest, near road

def estimate_features(lat: float, lon: float):
    """
    Estimate real-world ecological features from lat/lon.
    Based on known South India geography patterns.
    """
    # Western Ghats forest zone detection
    # High forest density between lat 10-14, lon 74-77
    in_ghats = (74.0 <= lon <= 77.5) and (10.0 <= lat <= 14.5)

    # Coastal zone
    near_coast = lon < 75.2

    # Known forest corridors (broader zones, not just points)
    forest_zones = [
        # (center_lat, center_lon, radius_deg, ndvi, elev, slope, df, dw, dr)
        (11.9, 76.1, 0.8,  0.82, 900,  12, 0.2, 0.4, 1.5),  # Nagarhole region
        (11.6, 76.4, 0.9,  0.80, 860,  15, 0.2, 0.3, 1.8),  # Bandipur/Mudumalai
        (11.6, 76.1, 0.8,  0.85, 790,  18, 0.1, 0.2, 1.6),  # Wayanad region
        (12.4, 75.7, 0.7,  0.84, 1000, 22, 0.2, 0.3, 2.0),  # Kodagu region
        (13.1, 75.3, 0.6,  0.86, 1100, 26, 0.2, 0.3, 2.5),  # Kudremukh region
        (11.9, 77.0, 0.7,  0.76, 1050, 20, 0.3, 0.5, 2.0),  # BRT Hills region
        (11.5, 77.2, 0.8,  0.74, 820,  16, 0.3, 0.5, 2.2),  # Sathyamangalam
        (10.5, 76.9, 0.7,  0.80, 880,  17, 0.2, 0.4, 2.0),  # Anamalai region
        (11.4, 76.7, 0.6,  0.78, 1100, 24, 0.2, 0.3, 2.5),  # Nilgiris region
        (13.6, 75.6, 0.5,  0.77, 830,  14, 0.2, 0.4, 2.0),  # Bhadra region
        (12.6, 75.7, 0.5,  0.83, 1000, 22, 0.2, 0.3, 2.0),  # Pushpagiri region
        (14.0, 74.8, 0.6,  0.80, 680,  16, 0.2, 0.2, 2.5),  # Sharavathi region
        (13.4, 75.1, 0.5,  0.85, 820,  22, 0.2, 0.3, 2.5),  # Agumbe region
        (14.6, 74.8, 0.5,  0.78, 600,  18, 0.3, 0.4, 3.0),  # Sirsi region
        (11.2, 77.5, 0.5,  0.79, 940,  20, 0.2, 0.3, 2.5),  # Kalakad region
        (12.0, 75.5, 0.6,  0.80, 850,  18, 0.2, 0.3, 2.0),  # Coorg buffer
        (11.0, 76.5, 0.6,  0.78, 820,  16, 0.2, 0.4, 2.0),  # Palakkad gap
        (10.8, 76.7, 0.5,  0.76, 750,  14, 0.3, 0.4, 2.0),  # Silent Valley
    ]

    # Find closest forest zone
    best_dist = float('inf')
    best_zone = None
    for zone in forest_zones:
        clat, clon, radius = zone[0], zone[1], zone[2]
        dist = ((lat - clat)**2 + (lon - clon)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_zone = zone

    if best_dist <= best_zone[2]:
        # Inside a known forest zone — use zone features with distance-based blending
        blend = 1.0 - (best_dist / best_zone[2])  # 1.0 at center, 0.0 at edge
        _, _, _, NDVI, elev, slope, df, dw, dr = best_zone[3], best_zone[4], best_zone[5], best_zone[3], best_zone[4], best_zone[5], best_zone[6], best_zone[7], best_zone[8]

        # Blend with urban values at edges
        urban_ndvi = 0.20
        urban_elev = 300
        urban_slope = 3
        urban_df = 8.0
        urban_dw = 4.0
        urban_dr = 0.8

        NDVI  = NDVI  * blend + urban_ndvi  * (1 - blend)
        elev  = elev  * blend + urban_elev  * (1 - blend)
        slope = slope * blend + urban_slope * (1 - blend)
        df    = df    * blend + urban_df    * (1 - blend)
        dw    = dw    * blend + urban_dw    * (1 - blend)
        dr    = dr    * blend + urban_dr    * (1 - blend)
        NDWI  = 0.35 * blend + 0.10 * (1 - blend)
    elif in_ghats:
        # In Western Ghats but outside known zone — moderate forest values
        dist_factor = min(1.0, best_dist / 2.0)
        NDVI  = 0.60 - dist_factor * 0.25
        NDWI  = 0.30 - dist_factor * 0.10
        elev  = 600  - dist_factor * 300
        slope = 12   - dist_factor * 8
        df    = 1.5  + dist_factor * 3.0
        dw    = 0.8  + dist_factor * 2.0
        dr    = 2.0  - dist_factor * 1.0
    elif near_coast:
        # Coastal zone
        NDVI  = 0.35
        NDWI  = 0.50
        elev  = 15
        slope = 2
        df    = 5.0
        dw    = 0.3
        dr    = 0.8
    else:
        # Urban/agricultural plains
        dist_factor = min(1.0, best_dist / 3.0)
        NDVI  = 0.20 + (1 - dist_factor) * 0.10
        NDWI  = 0.10
        elev  = 400
        slope = 2
        df    = 6.0 + dist_factor * 4.0
        dw    = 3.0 + dist_factor * 2.0
        dr    = 0.5

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
    """Return nearest known place name"""
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
        (12.63, 75.72, "Pushpagiri Wildlife Sanctuary"),
        (14.00, 74.80, "Sharavathi Wildlife Sanctuary"),
        (13.40, 75.10, "Agumbe Rainforest"),
        (14.60, 74.84, "Sirsi Forest"),
        (11.25, 77.55, "Kalakad Mundanthurai"),
        (10.78, 76.65, "Silent Valley"),
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
        dist = ((lat - plat)**2 + (lon - plon)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_dist > 0.5:
        best_name = f"({lat:.2f}, {lon:.2f})"
    return best_name


def run_prediction(lat: float, lon: float):
    NDVI, NDWI, elev, slope, df, dw, dr = estimate_features(lat, lon)
    feats = build_features(lat, lon, NDVI, NDWI, elev, slope, df, dw, dr)
    dfx   = pd.DataFrame([feats], columns=FEATURES)
    prob  = float(model.predict_proba(scaler.transform(dfx))[0][1] * 100)
    risk  = "HIGH" if prob >= 70 else "MEDIUM" if prob >= 40 else "LOW"
    location = get_location_name(lat, lon)

    return {
        "risk":        risk,
        "probability": round(prob, 2),
        "location":    location,
        "driver":      "NDVI" if NDVI > 0.65 else "dist_forest",
        "lat":         lat,
        "lon":         lon,
    }


# ── Routes ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {
        "name": "HWC Prediction API — Pattern Based",
        "info": "Works for ANY coordinate in South India (lat 10-15, lon 74-80)",
        "usage": {
            "browser_test": "/predict?lat=11.93&lon=76.13",
            "api":          "POST /predict with {lat, lon}"
        }
    }

@app.get("/predict")
def predict_get(lat: float, lon: float):
    return run_prediction(lat, lon)

class PredictRequest(BaseModel):
    lat: float
    lon: float

@app.post("/predict")
def predict_post(req: PredictRequest):
    return run_prediction(req.lat, req.lon)

