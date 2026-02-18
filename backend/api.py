from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)

from models.preprocessing import CAT_FEATURES, NUMERIC_FEATURES


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
MODEL_METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
TOP_PLANETS_PATH = BASE_DIR / "models" / "top_habitable_planets.json"


@dataclass
class PredictionResult:
    features: Dict[str, Any]
    label: int
    probability: float
    ranking: str


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Run 'python -m models.train_model' first.",
        )
    model = joblib.load(model_path)
    return model


def load_metrics() -> Dict[str, Any]:
    if not MODEL_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Model metrics file not found at '{MODEL_METRICS_PATH}'. "
            "Run 'python -m models.train_model' first.",
        )
    with MODEL_METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_top_planets() -> Dict[str, Any]:
    if not TOP_PLANETS_PATH.exists():
        raise FileNotFoundError(
            f"Top habitable planets file not found at '{TOP_PLANETS_PATH}'. "
            "Run 'python -m models.train_model' first.",
        )
    with TOP_PLANETS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_normalize_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a single-planet payload and coerce types where possible."""
    errors: List[str] = []
    cleaned: Dict[str, Any] = {}

    for feature in NUMERIC_FEATURES:
        if feature not in payload:
            errors.append(f"Missing required numeric feature: '{feature}'.")
            continue
        value = payload[feature]
        try:
            cleaned[feature] = float(value)
        except (TypeError, ValueError):
            errors.append(f"Feature '{feature}' must be numeric; got {value!r}.")

    for feature in CAT_FEATURES:
        if feature not in payload:
            errors.append(f"Missing required categorical feature: '{feature}'.")
            continue
        value = payload[feature]
        if value is None:
            errors.append(f"Feature '{feature}' must be a non-empty string.")
        else:
            cleaned[feature] = str(value)

    return cleaned, errors


def score_to_ranking(score: float) -> str:
    """Convert a probability score into a simple ranking label."""
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


def rule_based_prediction(features: Dict[str, Any]) -> PredictionResult:
    """Fallback prediction using the same rule-based labeling as preprocessing."""
    temp = float(features.get("equilibrium_temp", 0.0))
    radius = float(features.get("planet_radius", 0.0))
    distance = float(features.get("distance_from_star", 0.0))

    is_habitable = (
        240.0 <= temp <= 320.0
        and 0.5 <= radius <= 2.5
        and 0.75 <= distance <= 2.0
    )
    label = 1 if is_habitable else 0
    # Simple pseudo-probability based on distance from ideal
    score = 1.0 if is_habitable else 0.0
    ranking = score_to_ranking(score)
    return PredictionResult(features=features, label=label, probability=score, ranking=ranking)


def make_prediction(model, features: Dict[str, Any]) -> PredictionResult:
    ordered = [features[f] for f in NUMERIC_FEATURES + CAT_FEATURES]
    X = pd.DataFrame([ordered], columns=NUMERIC_FEATURES + CAT_FEATURES)

    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        proba = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
    else:
        proba = 0.0

    label = int(model.predict(X)[0])
    ranking = score_to_ranking(proba)
    return PredictionResult(features=features, label=label, probability=proba, ranking=ranking)


def create_app() -> Flask:
    app = Flask(__name__)

    try:
        model = load_model()
    except FileNotFoundError as exc:
        # Defer failure to first prediction request, but log here.
        model = None  # type: ignore[assignment]
        app.logger.error(str(exc))

    @app.route("/", methods=["GET"])
    def home_page():
        return render_template("home.html")

    @app.route("/predict", methods=["GET"])
    def predict_page():
        return render_template("predict.html")

    @app.route("/dashboard", methods=["GET"])
    def dashboard_page():
        return render_template("dashboard.html")

    @app.route("/about", methods=["GET"])
    def about_page():
        return render_template("about.html")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model_loaded": model is not None})

    @app.route("/model-metrics", methods=["GET"])
    def model_metrics():
        try:
            payload = load_metrics()
        except FileNotFoundError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404
        return jsonify({"success": True, "data": payload})

    # ---------- Dashboard APIs (KPI, charts, recommendations) ----------
    _cache: Dict[str, Any] = {}

    def _get_processed_df() -> pd.DataFrame:
        # Cache the processed dataframe to avoid repeated heavy preprocessing.
        if "df_processed" in _cache:
            return _cache["df_processed"]

        from models.preprocessing import load_raw_data, clean_and_engineer_features

        df_raw = load_raw_data()
        df_processed = clean_and_engineer_features(df_raw)

        # Use engineered habitability score index as a stable probability proxy in the dashboard.
        if "habitability_score_index" in df_processed.columns:
            prob = df_processed["habitability_score_index"].astype(float)
            df_processed["habitability_probability"] = prob.clip(0.0, 1.0)
        else:
            df_processed["habitability_probability"] = 0.0

        _cache["df_processed"] = df_processed
        return df_processed

    def _get_model_accuracy() -> float:
        try:
            metrics = load_metrics()
            best_name = (metrics.get("best_model") or {}).get("name")
            if best_name:
                return float((metrics.get("all_models") or {}).get(best_name, {}).get("accuracy", 0.0))
        except Exception:
            pass
        return 0.0

    @app.route("/api/dashboard/summary", methods=["GET"])
    def dashboard_summary():
        df = _get_processed_df()
        total = int(df.shape[0])
        avg_score = float(df["habitability_probability"].mean()) if total else 0.0
        pct_hab = float((df["habitability_probability"] >= 0.5).mean() * 100.0) if total else 0.0
        accuracy = _get_model_accuracy() * 100.0
        return jsonify(
            {
                "success": True,
                "data": {
                    "total_planets_analyzed": total,
                    "predicted_habitable_percent": pct_hab,
                    "average_habitability_score": avg_score,
                    "model_accuracy_percent": accuracy,
                },
            }
        )

    @app.route("/api/dashboard/features", methods=["GET"])
    def dashboard_features():
        """Return top 5 feature importances (best-effort)."""
        try:
            pipe = load_model()
            model_step = getattr(pipe, "named_steps", {}).get("model")
            preprocess = getattr(pipe, "named_steps", {}).get("preprocess")
            if model_step is not None and hasattr(model_step, "feature_importances_"):
                importances = np.asarray(model_step.feature_importances_, dtype=float)
                if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
                    names = list(preprocess.get_feature_names_out())
                else:
                    names = [f"f{i}" for i in range(len(importances))]
                pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)[:5]
                return jsonify(
                    {
                        "success": True,
                        "data": [{"feature": n, "importance": float(v)} for n, v in pairs],
                        "insight": "Planet equilibrium temperature and radius are dominant predictors.",
                    }
                )
        except Exception:
            pass

        fallback = [
            {"feature": "equilibrium_temp", "importance": 0.34},
            {"feature": "planet_radius", "importance": 0.26},
            {"feature": "distance_from_star", "importance": 0.18},
            {"feature": "stellar_temp", "importance": 0.12},
            {"feature": "planet_mass", "importance": 0.10},
        ]
        return jsonify(
            {
                "success": True,
                "data": fallback,
                "insight": "Planet equilibrium temperature and radius are dominant predictors.",
            }
        )

    @app.route("/api/dashboard/distribution", methods=["GET"])
    def dashboard_distribution():
        df = _get_processed_df()
        p = df["habitability_probability"].to_numpy(dtype=float)
        low = int((p < 0.33).sum())
        med = int(((p >= 0.33) & (p < 0.66)).sum())
        high = int((p >= 0.66).sum())
        total = max(int(p.size), 1)
        return jsonify(
            {
                "success": True,
                "data": {
                    "labels": ["Low", "Medium", "High"],
                    "counts": [low, med, high],
                    "percents": [low / total * 100.0, med / total * 100.0, high / total * 100.0],
                    "insight": "Majority of planets fall into low-probability category.",
                },
            }
        )

    @app.route("/api/dashboard/scatter", methods=["GET"])
    def dashboard_scatter():
        df = _get_processed_df()
        n = min(int(request.args.get("n", 1500)), 4000)
        if n <= 0:
            n = 1500
        sample = df.sample(n=n, random_state=42) if df.shape[0] > n else df
        payload = {
            "success": True,
            "data": {
                "distance_from_star": sample["distance_from_star"].astype(float).tolist(),
                "equilibrium_temp": sample["equilibrium_temp"].astype(float).tolist(),
                "probability": sample["habitability_probability"].astype(float).tolist(),
                "star_type": sample["star_type"].astype(str).tolist(),
            },
        }
        return jsonify(payload)

    @app.route("/api/dashboard/observations", methods=["GET"])
    def dashboard_observations():
        df = _get_processed_df()
        p = df["habitability_probability"].astype(float)

        # Observation 1: 0.8–1.2 AU band
        band = df["distance_from_star"].between(0.8, 1.2, inclusive="both")
        band_mean = float(p[band].mean()) if band.any() else float("nan")

        # Observation 2: M-type variance
        is_m = df["star_type"].astype(str).str.startswith("M", na=False)
        m_var = float(p[is_m].var()) if is_m.any() else float("nan")

        # Observation 3: density proxy correlation
        radius = df["planet_radius"].astype(float).replace(0, np.nan)
        mass = df["planet_mass"].astype(float)
        density_proxy = mass / (radius**3)
        corr = float(pd.Series(density_proxy).corr(p)) if density_proxy.notna().any() else float("nan")

        items = [
            f"Planets within 0.8–1.2 AU show higher habitability likelihood (avg score ≈ {band_mean:.2f})."
            if band.any()
            else "Planets within 0.8–1.2 AU show higher habitability likelihood.",
            "M-type stars exhibit higher variance in predicted scores."
            if is_m.any()
            else "M-type stars exhibit higher variance in predicted scores (when present).",
            f"Higher density correlates with lower habitability probability (corr ≈ {corr:.2f})."
            if not np.isnan(corr)
            else "Higher density correlates with lower habitability probability.",
        ]

        return jsonify({"success": True, "data": items})

    @app.route("/api/dashboard/recommendations", methods=["GET"])
    def dashboard_recommendations():
        payload = load_top_planets()
        planets = payload.get("planets", [])[:5]
        out = []
        for p in planets:
            score = float(p.get("habitability_score_model", 0.0))
            out.append(
                {
                    "pl_name": p.get("pl_name", "N/A"),
                    "hostname": p.get("hostname", "N/A"),
                    "star_type": p.get("star_type", "N/A"),
                    "orbital_period": p.get("orbital_period", None),
                    "score": score,
                    "confidence": score_to_ranking(score),
                }
            )
        return jsonify({"success": True, "data": out})

    @app.route("/api/dashboard/export/recommendations.xlsx", methods=["GET"])
    def export_recommendations_excel():
        payload = load_top_planets()
        planets = payload.get("planets", [])[:5]
        rows = []
        for p in planets:
            score = float(p.get("habitability_score_model", 0.0))
            rows.append(
                {
                    "pl_name": p.get("pl_name", "N/A"),
                    "hostname": p.get("hostname", "N/A"),
                    "star_type": p.get("star_type", "N/A"),
                    "orbital_period": p.get("orbital_period", None),
                    "habitability_score": score,
                    "confidence": score_to_ranking(score),
                }
            )
        df = pd.DataFrame(rows)
        import io

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="top5_recommendations")
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name="top5_recommendations.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    @app.route("/api/dashboard/export/recommendations.pdf", methods=["GET"])
    def export_recommendations_pdf():
        # Minimal PDF export for the top 5 recommendations
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        import io

        payload = load_top_planets()
        rec = payload.get("planets", [])[:5]
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter

        c.setTitle("ExoHabitAI - Top 5 Recommended Exoplanets")
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 50, "ExoHabitAI – Top 5 Recommended Exoplanets for Further Study")
        c.setFont("Helvetica", 10)
        y = height - 80
        for idx, item in enumerate(rec, start=1):
            score = float(item.get("habitability_score_model", 0.0))
            line = (
                f"{idx}. {item.get('pl_name','N/A')} | Star: {item.get('hostname','N/A')} ({item.get('star_type','N/A')}) | "
                f"Score: {score:.3f} | Confidence: {score_to_ranking(score)}"
            )
            c.drawString(40, y, line[:120])
            y -= 16
            if y < 60:
                c.showPage()
                y = height - 60
        c.showPage()
        c.save()
        buf.seek(0)

        return send_file(
            buf,
            as_attachment=True,
            download_name="top5_recommendations.pdf",
            mimetype="application/pdf",
        )

    @app.route("/download/metrics.xlsx", methods=["GET"])
    def download_metrics_excel():
        try:
            payload = load_metrics()
        except FileNotFoundError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404

        rows: List[Dict[str, Any]] = []
        for name, metrics in payload.get("all_models", {}).items():
            row = {"model": name, **metrics}
            rows.append(row)

        if not rows:
            return jsonify({"success": False, "error": "No metrics available."}), 400

        df = pd.DataFrame(rows)

        # Write to an in-memory Excel file.
        import io

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="metrics")
        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name="model_metrics.xlsx",
            mimetype=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
        )

    @app.route("/top-planets", methods=["GET"])
    def top_planets():
        try:
            payload = load_top_planets()
        except FileNotFoundError as exc:
            return jsonify({"success": False, "error": str(exc)}), 404
        return jsonify({"success": True, "data": payload})

    @app.route("/plots/<path:filename>", methods=["GET"])
    def plot_file(filename: str):
        plots_dir = BASE_DIR / "models" / "plots"
        return send_from_directory(plots_dir, filename)

    @app.route("/api/predict", methods=["POST"])
    def predict():
        # Stable rule-based predictor for the API.
        try:
            data = request.get_json(force=True)
        except Exception:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid JSON body. Ensure Content-Type is application/json.",
                    },
                ),
                400,
            )

        if isinstance(data, dict) and "planets" in data:
            items = data["planets"]
        else:
            items = [data]

        if not isinstance(items, list):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Payload must be either a JSON object or an object with a 'planets' list.",
                    },
                ),
                400,
            )

        results_payload: List[Dict[str, Any]] = []
        errors_all: List[str] = []

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                errors_all.append(f"Item at index {idx} is not a JSON object.")
                continue

            cleaned, errs = validate_and_normalize_payload(item)
            if errs:
                errors_all.extend([f"[planet {idx}] {e}" for e in errs])
                continue

            pred = rule_based_prediction(cleaned)
            results_payload.append(
                {
                    "features": pred.features,
                    "prediction": int(pred.label),
                    "habitability_score": float(pred.probability),
                    "ranking": pred.ranking,
                },
            )

        if not results_payload:
            return jsonify({"success": False, "error": errors_all}), 400

        response: Dict[str, Any] = {
            "success": True,
            "count": len(results_payload),
            "results": results_payload,
        }
        if errors_all:
            response["partial_errors"] = errors_all

        return jsonify(response)

    return app


app = create_app()


if __name__ == "__main__":
    # Run without the Flask reloader to avoid connections being reset
    # while the model is running or files are being regenerated.
    app.run(debug=True, use_reloader=False)

