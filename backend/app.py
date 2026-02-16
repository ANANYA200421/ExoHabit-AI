"""
ExoHabitAI - Full Stack Flask Backend
Supports:
- REST API
- UI Rendering
- Dashboard
- Model Evaluation
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================================
# PATH CONFIGURATION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "..", "frontend", "static")

MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "best_model.pkl")
RANKING_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "exoplanet_ranking.csv")
TEST_X_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "X_test.csv")
TEST_Y_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "y_test.csv")

OPTIMAL_THRESHOLD = 0.0763





# ==========================================================
# FLASK INITIALIZATION
# ==========================================================

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

CORS(app)

# ==========================================================
# LOAD MODEL
# ==========================================================

try:
    model = joblib.load(MODEL_PATH)
    print("✓ Model Loaded Successfully")
except Exception as e:
    print("✗ Model Load Failed:", e)
    model = None

# ==========================================================
# ROUTES
# ==========================================================

# -------------------------------
# Health Check
# -------------------------------
@app.route("/status")
def status():
    return jsonify({
        "status": "operational",
        "model_loaded": model is not None,
        "threshold": OPTIMAL_THRESHOLD,
        "timestamp": datetime.now().isoformat()
    })


# -------------------------------
# UI Route
# -------------------------------
@app.route("/ui")
def ui():
    return render_template("index.html")


# -------------------------------
# Dashboard Route
# -------------------------------
@app.route("/dashboard")
def dashboard():

    if not os.path.exists(RANKING_PATH):
        return "Ranking file not found."

    ranking_df = pd.read_csv(RANKING_PATH)

    # Feature Importance (if available)
    feature_importance = {}
    if hasattr(model, "estimators_") and len(model.estimators_) > 1:
        rf_model = model.estimators_[1]
        if hasattr(rf_model, "feature_importances_"):
            feature_importance = dict(
                zip(ranking_df.columns[:len(rf_model.feature_importances_)],
                    rf_model.feature_importances_)
            )

    return render_template(
        "dashboard_plotly.html",
        feature_importance=feature_importance,
        probabilities=ranking_df['Habitability_Probability'].tolist(),
        correlation_data=ranking_df.select_dtypes(include=["number"]).corr().values.tolist(),
        correlation_columns=ranking_df.select_dtypes(include=["number"]).columns.tolist()
    )


# -------------------------------
# Predict API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        probability = model.predict_proba(input_df)[:, 1][0]
        prediction = int(probability >= OPTIMAL_THRESHOLD)

        return jsonify({
            "prediction": "Habitable" if prediction else "Non-Habitable",
            "confidence_score": round(float(probability), 4),
            "threshold_used": OPTIMAL_THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Ranking API
# -------------------------------
@app.route("/rank", methods=["GET"])
def rank():

    if not os.path.exists(RANKING_PATH):
        return jsonify({"error": "Ranking file not found"}), 404

    df = pd.read_csv(RANKING_PATH)

    top_n = request.args.get("top", default=10, type=int)
    min_score = request.args.get("min_score", default=0.0, type=float)

    filtered = df[df["habitability_score"] >= min_score].head(top_n)

    return jsonify({
        "total_planets": len(df),
        "returned_count": len(filtered),
        "planets": filtered.to_dict(orient="records")
    })


# -------------------------------
# Model Evaluation (Accuracy Check)
# -------------------------------
@app.route("/model/evaluate")
def evaluate():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    if not os.path.exists(TEST_X_PATH) or not os.path.exists(TEST_Y_PATH):
        return jsonify({"error": "Test dataset not found"}), 404

    X_test = pd.read_csv(TEST_X_PATH)
    y_test = pd.read_csv(TEST_Y_PATH).values.ravel()

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= OPTIMAL_THRESHOLD).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1_score": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "threshold_used": OPTIMAL_THRESHOLD
    }

    cm = confusion_matrix(y_test, predictions).tolist()

    return jsonify({
        "metrics": metrics,
        "confusion_matrix": cm
    })


# -------------------------------
# Feature Importance API
# -------------------------------
@app.route("/dashboard/feature-importance")
def feature_importance():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        if hasattr(model, "estimators_") and len(model.estimators_) > 1:
            rf_model = model.estimators_[1]
            if hasattr(rf_model, "feature_importances_"):
                return jsonify({
                    "importance_scores": rf_model.feature_importances_.tolist()
                })

        return jsonify({"importance_scores": []})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# ERROR HANDLERS
# ==========================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == "__main__":
    print("🚀 ExoHabitAI Backend Starting...")
    app.run(debug=True, host="0.0.0.0", port=5000)
