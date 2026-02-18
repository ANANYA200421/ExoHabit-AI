from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore[misc,assignment]


def _ensure_output_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute and return standard binary classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    metrics: Dict[str, float] = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            metrics["roc_auc"] = float(roc_auc)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


def print_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Print a detailed sklearn classification report."""
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, digits=4))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path | str,
) -> Path:
    """Generate and save a confusion matrix plot."""
    output_dir = _ensure_output_dir(output_dir)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()

    out_path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    model_name: str,
    output_dir: Path | str,
) -> Optional[Path]:
    """Generate and save ROC curve plot if possible.

    When ROC cannot be computed (e.g. only one class present), generate a
    placeholder plot instead so the dashboard always has an image to show.
    """
    output_dir = _ensure_output_dir(output_dir)
    out_path = output_dir / f"roc_curve_{model_name}.png"

    if y_proba is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(
            0.5,
            0.5,
            "ROC curve not available\n(single-class predictions)",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(
            0.5,
            0.5,
            "ROC curve not available\n(invalid probability data)",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_importance(
    pipeline,
    model_name: str,
    output_dir: Path | str,
) -> Optional[Path]:
    """Plot feature importance for tree-based models within a pipeline."""
    model = pipeline.named_steps.get("model")
    if model is None:
        return None

    is_rf = isinstance(model, RandomForestClassifier)
    is_xgb = XGBClassifier is not None and isinstance(model, XGBClassifier)
    if not (is_rf or is_xgb):
        # Feature importance requested only for tree-based models.
        return None

    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.asarray(model.feature_importances_)

    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
        feature_names = preprocess.get_feature_names_out()
    else:
        feature_names = np.array([f"f{i}" for i in range(len(importances))])

    # Sort by importance
    indices = np.argsort(importances)[::-1]
    feature_names = feature_names[indices]
    importances = importances[indices]

    output_dir = _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(importances)), importances[::-1], align="center")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(feature_names[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Feature Importance - {model_name}")
    fig.tight_layout()

    out_path = output_dir / f"feature_importance_{model_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_habitability_distribution(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    model_name: str,
    output_dir: Path | str,
) -> Optional[Path]:
    """Plot distribution of true labels and (optionally) predicted probabilities."""
    output_dir = _ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Bar chart for class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    ax.bar(
        [str(u) for u in unique],
        counts,
        alpha=0.7,
        label="Class counts",
        color="#3b82f6",
    )
    ax.set_xlabel("Habitability label")
    ax.set_ylabel("Count")

    # Add a twin axis for probability histogram if available
    if y_proba is not None:
        ax2 = ax.twinx()
        ax2.hist(
            y_proba,
            bins=20,
            alpha=0.3,
            color="#f97316",
            label="Predicted probability",
        )
        ax2.set_ylabel("Frequency (probability histogram)")
        ax2.set_ylim(0, max(ax2.get_ylim()[1], 1))
        lines_labels = [ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        ax.legend(lines, labels, loc="upper right")
    else:
        ax.legend(loc="upper right")

    ax.set_title(f"Habitability Distribution - {model_name}")
    fig.tight_layout()

    out_path = output_dir / f"habitability_distribution_{model_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_star_planet_scatter(
    X,
    y_true: np.ndarray,
    model_name: str,
    output_dir: Path | str,
) -> Optional[Path]:
    """Plot a simple star–planet relationship scatter (distance vs temperature)."""
    # Expect dataframe-like input with named columns
    if not hasattr(X, "columns"):
        return None

    cols = getattr(X, "columns", [])
    required = {"distance_from_star", "equilibrium_temp"}
    if not required.issubset(set(cols)):
        return None

    output_dir = _ensure_output_dir(output_dir)

    distance = X["distance_from_star"].to_numpy()
    temp = X["equilibrium_temp"].to_numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(
        distance,
        temp,
        c=y_true,
        cmap="viridis",
        alpha=0.6,
        edgecolors="none",
    )
    ax.set_xlabel("Distance from star (AU-like scale)")
    ax.set_ylabel("Equilibrium temperature (K)")
    ax.set_title(f"Star–Planet Relationship - {model_name}")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Habitability label")

    fig.tight_layout()

    out_path = output_dir / f"star_planet_scatter_{model_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def evaluate_and_report(
    model_name: str,
    pipeline,
    X_test,
    y_test,
    plots_dir: Path | str,
) -> Dict[str, float]:
    """Run full evaluation, plotting, and return metrics dict."""
    y_pred = pipeline.predict(X_test)

    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        if proba.shape[1] == 2:
            y_proba = proba[:, 1]

    metrics = compute_classification_metrics(y_test, y_pred, y_proba)

    # Console reporting
    print(f"\n=== Evaluation for {model_name} ===")
    print(f"Metrics: {metrics}")
    print_classification_summary(y_test, y_pred)

    # Plots
    plots_dir = _ensure_output_dir(plots_dir)
    plot_confusion_matrix(y_test, y_pred, model_name, plots_dir)
    plot_roc_curve(y_test, y_proba, model_name, plots_dir)
    plot_feature_importance(pipeline, model_name, plots_dir)
    plot_habitability_distribution(y_test, y_proba, model_name, plots_dir)
    plot_star_planet_scatter(X_test, y_test, model_name, plots_dir)

    return metrics


__all__ = [
    "compute_classification_metrics",
    "print_classification_summary",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_feature_importance",
    "plot_habitability_distribution",
    "plot_star_planet_scatter",
    "evaluate_and_report",
]

