from pyexpat import model
from flask import Blueprint, request, jsonify
import numpy as np
import joblib
import os

predict_bp = Blueprint("predict", __name__)

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

PIPELINE_PATH = os.path.join(ROOT, "models", "habitability_pipeline.pkl")

pipeline = joblib.load(PIPELINE_PATH)


STAR_MAP = {
    "O": 6,
    "B": 5,
    "A": 4,
    "F": 3,
    "G": 2,
    "K": 1,
    "M": 0
}

@predict_bp.route("/predict", methods=["POST"])
def predict():

    data = request.json

    radius = float(data["radius"])
    mass = float(data["mass"])
    period = float(data["period"])
    temp = float(data["temp"])
    star = data["star"]

    star_map = {
        "O":0,
        "B":1,
        "A":2,
        "F":3,
        "G":4,
        "K":5,
        "M":6
    }

    star_encoded = star_map.get(star,4)

    X = [[radius, mass, period, temp, star_encoded]]

    # 🔥 USE PIPELINE DIRECTLY
    prob = model.predict_proba(X)[0][1]

    return {
        "habitable": bool(prob > 0.5),
        "score": float(prob)
    }

