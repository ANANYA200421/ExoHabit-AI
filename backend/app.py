# ============================================================
# ExoHabit-AI Backend (Flask Integration)
# Week 5 + Week 6
# ============================================================

from flask import Flask, request, jsonify, render_template
import os
import joblib
import pandas as pd

# ------------------------------------------------------------
# Define Base Directory
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute paths (SAFE & PRODUCTION READY)
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "..", "frontend", "static")
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "best_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "preprocessing_pipeline.pkl")
RANKING_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "exoplanet_ranking.csv")

# ------------------------------------------------------------
# Initialize Flask App
# ------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# ------------------------------------------------------------
# Load Model & Preprocessor
# ------------------------------------------------------------
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PIPELINE_PATH)

# Expected input columns
num_cols = ['pl_rade', 'pl_eqt', 'pl_orbper', 'stellar_compatibility']
cat_cols = ['st_spectype_simple']

# ============================================================
# ROUTES
# ============================================================

# ------------------------------------------------------------
# Health Check
# ------------------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "ExoHabit-AI Backend Running Successfully 🚀"
    })


# ------------------------------------------------------------
# Frontend UI Route
# ------------------------------------------------------------
@app.route("/ui")
def ui():
    return render_template("index.html")


# ------------------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Validate required fields
        required_fields = num_cols + cat_cols
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing field: {field}"
                }), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Preprocess input
        input_processed = preprocessor.transform(input_df)

        # Predict probability
        probability = model.predict_proba(input_processed)[0][1]
        prediction = int(probability > 0.5)

        return jsonify({
            "habitability_probability": round(float(probability), 4),
            "prediction": prediction,
            "interpretation": "Habitable Candidate"
            if prediction == 1 else "Non-Habitable"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------
# Ranking Endpoint
# ------------------------------------------------------------
@app.route("/rank", methods=["GET"])
def rank():
    try:
        df = pd.read_csv(RANKING_PATH)

        top_n = int(request.args.get("top", 10))

        top_planets = df.head(top_n).to_dict(orient="records")

        return jsonify({
            "top_planets": top_planets
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/dashboard")
def dashboard():

    import pandas as pd

    X = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "processed", "X_train.csv"))
    ranking_df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "processed", "exoplanet_ranking.csv"))

    # Feature importance
    if hasattr(model, "feature_importances_"):
        feature_importance = dict(zip(X.columns, model.feature_importances_))
    else:
        feature_importance = {}

    return render_template(
        "dashboard_plotly.html",
        feature_importance=feature_importance,
        probabilities=ranking_df["Habitability_Probability"].tolist(),
        correlation_data=X.select_dtypes(include=["number"]).corr().values.tolist(),
        correlation_columns=X.select_dtypes(include=["number"]).columns.tolist()
    )

# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
