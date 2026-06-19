from fastapi import FastAPI
from math import radians, sin, cos, sqrt, atan2

app = FastAPI()

# -------------------------------
# FOREST / WILDLIFE HOTSPOTS
# -------------------------------
FORESTS = [
    {"name": "Bandipur", "lat": 11.65, "lon": 76.63},
    {"name": "Nagarhole", "lat": 12.05, "lon": 76.15},
    {"name": "BRT Hills", "lat": 11.93, "lon": 77.15},
    {"name": "Mudumalai", "lat": 11.56, "lon": 76.53},
    {"name": "Wayanad", "lat": 11.72, "lon": 76.13},
]
# -------------------------------
# MAJOR URBAN AREAS
# -------------------------------
def is_urban_area(lat, lon):

    # Bengaluru
    if 12.80 <= lat <= 13.20 and 77.40 <= lon <= 77.85:
        return True

    # Mysuru
    if 12.20 <= lat <= 12.40 and 76.55 <= lon <= 76.75:
        return True

    # Chennai
    if 12.90 <= lat <= 13.20 and 80.15 <= lon <= 80.35:
        return True

    # Hyderabad
    if 17.25 <= lat <= 17.60 and 78.20 <= lon <= 78.65:
        return True

    return False


# -------------------------------
# DISTANCE CALCULATION
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# -------------------------------
# FIND NEAREST FOREST
# -------------------------------
def nearest_forest(lat, lon):

    nearest_name = None
    min_distance = float("inf")

    for forest in FORESTS:
        dist = haversine(
            lat,
            lon,
            forest["lat"],
            forest["lon"]
        )

        if dist < min_distance:
            min_distance = dist
            nearest_name = forest["name"]

    return nearest_name, round(min_distance, 2)


# -------------------------------
# API
# -------------------------------
@app.get("/")
def home():
    return {"message": "Human Wildlife Conflict Risk API Running"}


@app.get("/predict")
def predict(lat: float, lon: float):

    # Urban override
    if is_urban_area(lat, lon):
        return {
            "risk_level": "LOW",
            "confidence": 95,
            "nearest_forest": "Urban Area",
            "distance_km": 0,
            "message": "Location is inside a major urban area."
        }

    forest_name, distance = nearest_forest(lat, lon)

    if distance <= 15:
        risk = "HIGH"
        confidence = 90

    elif distance <= 40:
        risk = "MEDIUM"
        confidence = 75

    else:
        risk = "LOW"
        confidence = 85

    return {
        "risk_level": risk,
        "confidence": confidence,
        "nearest_forest": forest_name,
        "distance_km": distance,
        "message": f"Nearest wildlife hotspot: {forest_name}"
    }
