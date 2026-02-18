from flask import Blueprint, request, jsonify
import joblib
import os
from config import Config
from utils import prepare_features, generate_response

api = Blueprint("api", __name__)

# ===============================
# LOAD MODEL SAFELY
# ===============================
if not os.path.exists(Config.MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {Config.MODEL_PATH}")

model = joblib.load(Config.MODEL_PATH)


# ===============================
# HOME ROUTE
# ===============================
@api.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ExoHabitAI API Running 🚀",
        "endpoints": [
            "/predict",
            "/feature-importance",
            "/model-metrics",
            "/dashboard-data",
            "/rankings"
        ]
    })


# ===============================
# PREDICT
# ===============================
@api.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        features = prepare_features(data)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        response = generate_response(prediction, probability)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ===============================
# FEATURE IMPORTANCE
# ===============================
@api.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        # If pipeline
        if hasattr(model, "named_steps"):
            model_step = model.named_steps.get("model", None)
        else:
            model_step = model

        if not hasattr(model_step, "feature_importances_"):
            return jsonify({"error": "Feature importance not available"}), 400

        importances = model_step.feature_importances_

        features = [
            "Planet Radius",
            "Equilibrium Temp",
            "Orbital Period",
            "Stellar Temp",
            "Stellar Compatibility"
        ]

        result = [
            {"feature": f, "importance": float(i)}
            for f, i in zip(features, importances)
        ]

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ===============================
# MODEL METRICS (STATIC)
# ===============================
@api.route("/model-metrics", methods=["GET"])
def model_metrics():
    return jsonify({
        "accuracy": 0.91,
        "precision": 0.89,
        "recall": 0.88,
        "f1_score": 0.885,
        "roc_auc": 0.93
    })


# ===============================
# DASHBOARD DATA
# ===============================
@api.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    try:
        return jsonify({
            "habitability_distribution": [0.1, 0.25, 0.4, 0.6, 0.85],
            "temperature_vs_habitability": [
                {"temp": 200, "score": 0.2},
                {"temp": 250, "score": 0.5},
                {"temp": 300, "score": 0.8}
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ===============================
# RANKINGS (STATIC DATA)
# ===============================
@api.route("/rankings", methods=["GET"])
def rankings():
    data = [
        {"rank": 1, "planet": "Kepler-452b", "score": 0.95, "probability": 0.92, "star_type": "G"},
        {"rank": 2, "planet": "TRAPPIST-1e", "score": 0.91, "probability": 0.89, "star_type": "M"},
        {"rank": 3, "planet": "Proxima Centauri b", "score": 0.88, "probability": 0.85, "star_type": "M"}
    ]

    return jsonify(data)
