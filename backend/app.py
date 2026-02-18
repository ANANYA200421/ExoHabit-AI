from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "frontend", "templates")
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "final_habitability_model.pkl")

model = joblib.load(MODEL_PATH)
print("MODEL LOADED SUCCESSFULLY")


# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        pl_rade = float(data["pl_rade"])
        pl_mass = float(data["pl_bmasse"])
        pl_orbper = float(data["pl_orbper"])
        pl_eqt = float(data["pl_eqt"])

        # ⭐ HANDLE "G (Sun-like)" → G
        spectral = data["spectral_type"][0].upper()

        # ⭐ AUTO STELLAR COMPATIBILITY
        spectral_map = {
            "O":0.1,
            "B":0.2,
            "A":0.3,
            "F":0.6,
            "G":0.9,
            "K":0.8,
            "M":0.5
        }

        stellar_compatibility = spectral_map.get(spectral, 0.5)

        # ⭐ MODEL INPUT (5 FEATURES ONLY)
        features = np.array([[ 
            pl_rade,
            pl_mass,
            pl_orbper,
            pl_eqt,
            stellar_compatibility
        ]])

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        return jsonify({
            "success": True,
            "habitable": int(prediction),
            "habitability_probability": round(float(prob),3)
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "success": False,
            "message": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)
