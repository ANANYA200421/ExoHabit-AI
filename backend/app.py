from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# ------------------------------------------------
# Setup paths
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "final_habitability_model.pkl"
)

# ------------------------------------------------
# Initialize Flask app
# ------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------
# Load ML Model
# ------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Load Failed:", e)
    model = None

# ------------------------------------------------
# Health Check
# ------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "ExoHabit-AI Backend Running 🚀",
        "endpoints": {
            "/predict": "POST – Predict habitability for one planet",
            "/rank": "POST – Rank multiple planets by habitability"
        }
    })

# ------------------------------------------------
# Prediction Endpoint (Single Planet)
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        required_fields = [
            "pl_rade",
            "pl_eqt",
            "pl_orbper",
            "st_teff",
            "stellar_compatibility"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = np.array([[
            float(data["pl_rade"]),
            float(data["pl_eqt"]),
            float(data["pl_orbper"]),
            float(data["st_teff"]),
            float(data["stellar_compatibility"])
        ]])

        prediction = int(model.predict(features)[0])

        proba = model.predict_proba(features)[0]
        probability = float(proba[1]) if len(proba) > 1 else 0.0

        return jsonify({
            "habitable": prediction,
            "habitability_probability": round(probability, 4),
            "interpretation": (
                "Habitable Candidate" if prediction == 1
                else "Non-Habitable"
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------
# Ranking Endpoint (Multiple Planets)
# ------------------------------------------------
@app.route("/rank", methods=["POST"])
def rank():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data or "planets" not in data:
            return jsonify({"error": "Missing 'planets' array"}), 400

        ranked_results = []

        for planet in data["planets"]:
            required_fields = [
                "pl_rade",
                "pl_eqt",
                "pl_orbper",
                "st_teff",
                "stellar_compatibility"
            ]

            for field in required_fields:
                if field not in planet:
                    return jsonify({"error": f"Missing field: {field}"}), 400

            features = np.array([[
                float(planet["pl_rade"]),
                float(planet["pl_eqt"]),
                float(planet["pl_orbper"]),
                float(planet["st_teff"]),
                float(planet["stellar_compatibility"])
            ]])

            proba = model.predict_proba(features)[0]
            score = float(proba[1]) if len(proba) > 1 else 0.0

            planet_out = planet.copy()
            planet_out["habitability_score"] = round(score, 4)

            ranked_results.append(planet_out)

        ranked_results.sort(
            key=lambda x: x["habitability_score"],
            reverse=True
        )

        return jsonify({
            "ranked_planets": ranked_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------
# Run Server
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
