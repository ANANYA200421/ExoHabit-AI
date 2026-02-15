from flask import Flask, request, jsonify, render_template


import joblib
import pandas as pd

app = Flask(__name__,
            template_folder="../frontend/templates",
            static_folder="../frontend/static")

# Load model and preprocessing pipeline
model = joblib.load("../models/best_model_random_forest.pkl")
pipeline = joblib.load("../models/preprocessing_pipeline.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ExoHabitAI API Running",
        "message": "Welcome to Habitability Prediction API"
    })
@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = [
    "planet_radius",
    "planet_mass",
    "orbital_period",
    "equilibrium_temperature",
    "st_spectype_simple",
    "stellar_compatibility"
]



        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

       
        input_df = pd.DataFrame([{
    "pl_rade": data["planet_radius"],
    "pl_orbper": data["orbital_period"],
    "pl_eqt": data["equilibrium_temperature"],
    "st_spectype_simple": data["st_spectype_simple"],
    "stellar_compatibility": data["stellar_compatibility"]
}])



        input_processed = pipeline.transform(input_df)

        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]

        return jsonify({
            "habitable": bool(prediction),
            "habitability_score": round(float(probability), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
