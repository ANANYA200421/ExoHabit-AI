from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from .evaluate import evaluate_and_report
from .preprocessing import (
    DATASET_DEFAULT_PATH,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CAT_FEATURES,
    build_model_pipeline,
    load_and_prepare_data,
    load_raw_data,
    clean_and_engineer_features,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore[misc,assignment]

try:
    import joblib
except Exception as exc:  # pragma: no cover - joblib is a required dependency
    raise ImportError(
        "joblib is required to save trained models. "
        "Please install it via 'pip install joblib'.",
    ) from exc


def train_test_split_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset into train and test sets with stratification if possible."""
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test


def train_random_forest_with_grid_search(
    X_train,
    y_train,
) -> Tuple[object, Dict]:
    """Train a Random Forest classifier with hyperparameter tuning via GridSearchCV."""
    base_rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_pipeline = build_model_pipeline(base_rf)

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 8, 16],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }

    # Use accuracy for CV scoring to avoid issues when some folds
    # contain only a single class (which breaks ROC-AUC scoring).
    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print("\n=== RandomForest best params from GridSearchCV ===")
    print(grid_search.best_params_)
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    best_pipeline = grid_search.best_estimator_
    meta = {
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
    }
    return best_pipeline, meta


def train_logistic_regression(
    X_train,
    y_train,
):
    """Train a Logistic Regression classifier within the preprocessing pipeline.

    If the training data contains only a single target class, the model is not
    trained and ``None`` is returned so that callers can skip it gracefully.
    """
    if len(np.unique(y_train)) < 2:
        print(
            "Skipping Logistic Regression: training data contains only a single "
            "target class.",
        )
        return None

    log_reg = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        solver="saga",
        penalty="l2",
    )
    pipeline = build_model_pipeline(log_reg)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost(
    X_train,
    y_train,
):
    """Train an XGBoost classifier within the preprocessing pipeline, if available."""
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not installed. "
            "Install it with 'pip install xgboost' to use the XGBoost model.",
        )

    if len(np.unique(y_train)) < 2:
        print(
            "Skipping XGBoost: training data contains only a single target class.",
        )
        return None

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    pipeline = build_model_pipeline(xgb)
    pipeline.fit(X_train, y_train)
    return pipeline


def select_best_model(
    metrics_by_model: Dict[str, Dict[str, float]],
) -> str:
    """Select the best-performing model based on ROC-AUC."""
    best_name = None
    best_score = float("-inf")
    for name, metrics in metrics_by_model.items():
        roc_auc = metrics.get("roc_auc", float("nan"))
        if np.isnan(roc_auc):
            continue
        if roc_auc > best_score:
            best_score = roc_auc
            best_name = name

    if best_name is None:
        # Fallback to accuracy if ROC-AUC is unavailable everywhere
        for name, metrics in metrics_by_model.items():
            acc = metrics.get("accuracy", float("nan"))
            if np.isnan(acc):
                continue
            if acc > best_score:
                best_score = acc
                best_name = name

    if best_name is None:
        raise RuntimeError("Could not determine a best model from the metrics.")

    return best_name


def save_model_and_metrics(
    best_model_name: str,
    pipelines: Dict[str, object],
    metrics_by_model: Dict[str, Dict[str, float]],
    output_dir: Path | str,
) -> None:
    """Persist the best model (pipeline) and the evaluation metrics to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_pipeline = pipelines[best_model_name]
    best_model_path = output_dir / "best_model.pkl"
    metrics_path = output_dir / "model_metrics.json"

    joblib.dump(best_pipeline, best_model_path)

    payload = {
        "target_column": TARGET_COLUMN,
        "dataset_path": str(DATASET_DEFAULT_PATH),
        "best_model": {
            "name": best_model_name,
            "roc_auc": metrics_by_model.get(best_model_name, {}).get("roc_auc"),
        },
        "all_models": metrics_by_model,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved best model pipeline to: {best_model_path}")
    print(f"Saved evaluation metrics to:   {metrics_path}")


def compute_and_save_top_habitable_planets(
    best_pipeline,
    output_dir: Path | str,
    top_k: int = 10,
) -> None:
    """Compute and persist the top-K most habitable planets by model score.

    Uses the fully preprocessed dataset (all rows) and the best model pipeline
    to:
    - predict habitability probabilities for every planet
    - rank planets by score
    - save the top-K entries as JSON
    - generate a bar chart for the dashboard
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(DATASET_DEFAULT_PATH)
    df_processed = clean_and_engineer_features(df_raw)

    feature_cols = NUMERIC_FEATURES + CAT_FEATURES
    X_all = df_processed[feature_cols]

    if hasattr(best_pipeline, "predict_proba"):
        proba = best_pipeline.predict_proba(X_all)
        scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    else:
        # Fallback to binary predictions if probabilities are unavailable.
        scores = best_pipeline.predict(X_all)

    df_processed = df_processed.copy()
    df_processed["habitability_score_model"] = scores

    id_cols = []
    for col in ("pl_name", "hostname"):
        if col in df_processed.columns:
            id_cols.append(col)

    cols_to_keep = id_cols + feature_cols + [
        "habitability_score_index",
        "stellar_compatibility_index",
        "habitability_score_model",
    ]
    cols_to_keep = [c for c in cols_to_keep if c in df_processed.columns]

    df_sorted = df_processed.sort_values(
        "habitability_score_model",
        ascending=False,
    ).reset_index(drop=True)
    df_top = df_sorted.head(top_k)[cols_to_keep]

    # Save JSON payload
    top_path = output_dir / "top_habitable_planets.json"
    payload = {
        "top_k": top_k,
        "count": int(df_top.shape[0]),
        "planets": df_top.to_dict(orient="records"),
    }
    with top_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Bar chart for dashboard
    names = df_top[id_cols[0]] if id_cols else df_top.index.astype(str)
    scores_top = df_top["habitability_score_model"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(scores_top)), scores_top[::-1], color="#6366f1")
    ax.set_yticks(range(len(scores_top)))
    ax.set_yticklabels(list(names[::-1]))
    ax.set_xlabel("Predicted habitability score")
    ax.set_title("Top Habitable Candidates")
    fig.tight_layout()

    plot_path = plots_dir / "top10_habitable_random_forest.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """End-to-end training entrypoint for the ExoHabitAI project."""
    print("=== Loading and preprocessing data ===")
    X, y = load_and_prepare_data(DATASET_DEFAULT_PATH)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    models: Dict[str, object] = {}
    metrics_by_model: Dict[str, Dict[str, float]] = {}

    # Train Random Forest with hyperparameter tuning
    print("\n=== Training RandomForest (with GridSearchCV) ===")
    rf_pipeline, rf_meta = train_random_forest_with_grid_search(X_train, y_train)
    models["random_forest"] = rf_pipeline
    print(f"RandomForest grid-search metadata: {rf_meta}")

    # Train Logistic Regression
    print("\n=== Training Logistic Regression ===")
    logreg_pipeline = train_logistic_regression(X_train, y_train)
    if logreg_pipeline is not None:
        models["logistic_regression"] = logreg_pipeline

    # Train XGBoost (optional if xgboost is installed)
    if XGBClassifier is not None:
        print("\n=== Training XGBoost ===")
        try:
            xgb_pipeline = train_xgboost(X_train, y_train)
            if xgb_pipeline is not None:
                models["xgboost"] = xgb_pipeline
        except Exception as exc:
            print(f"Warning: XGBoost training failed: {exc}")
    else:
        print("\nXGBoost is not installed; skipping XGBoost model.")

    # Evaluation
    scripts_dir = Path(__file__).resolve().parent
    plots_dir = scripts_dir / "plots"

    for name, pipeline in models.items():
        metrics = evaluate_and_report(name, pipeline, X_test, y_test, plots_dir)
        metrics_by_model[name] = metrics

    # Model selection and persistence
    best_model_name = select_best_model(metrics_by_model)
    print(f"\n=== Best model based on ROC-AUC: {best_model_name} ===")

    save_model_and_metrics(
        best_model_name=best_model_name,
        pipelines=models,
        metrics_by_model=metrics_by_model,
        output_dir=scripts_dir,
    )

    # Compute and persist top candidate planets using the best pipeline.
    best_pipeline = models[best_model_name]
    compute_and_save_top_habitable_planets(best_pipeline, scripts_dir)


if __name__ == "__main__":
    main()

