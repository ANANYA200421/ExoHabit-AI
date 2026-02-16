# ============================================================
# ExoHabit-AI Dashboard Visualization Script
# WEEK 7 — Visualization & Dashboard
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set_style("whitegrid")

# ------------------------------------------------------------
# Define Base Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "X_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "best_model.pkl")
RANKING_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "exoplanet_ranking.csv")
STATIC_PATH = os.path.join(BASE_DIR, "..", "frontend", "static")

os.makedirs(STATIC_PATH, exist_ok=True)

# ------------------------------------------------------------
# Load Data & Model
# ------------------------------------------------------------
X = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# ============================================================
# 1️⃣ FEATURE IMPORTANCE PLOT
# ============================================================

if hasattr(model, "feature_importances_"):

    importances = model.feature_importances_
    feature_names = X.columns

    # Safety check if lengths mismatch
    if len(feature_names) != len(importances):
        min_len = min(len(feature_names), len(importances))
        feature_names = feature_names[:min_len]
        importances = importances[:min_len]

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(data=fi_df.head(10), x="Importance", y="Feature")
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_PATH, "feature_importance.png"))
    plt.close()

else:
    print("Model does not support feature_importances_ (e.g., Logistic Regression).")

# ============================================================
# 2️⃣ HABITABILITY PROBABILITY DISTRIBUTION
# ============================================================

ranking_df = pd.read_csv(RANKING_PATH)

plt.figure(figsize=(6,4))
sns.histplot(ranking_df["Habitability_Probability"], kde=True)
plt.title("Habitability Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, "distribution.png"))
plt.close()

# ============================================================
# 3️⃣ NUMERIC CORRELATION HEATMAP
# ============================================================

# Select only numeric columns to avoid string conversion errors
numeric_X = X.select_dtypes(include=["number"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_X.corr(), cmap="coolwarm", annot=True)
plt.title("Star–Planet Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, "correlation.png"))
plt.close()

# ============================================================
# 4️⃣ EXPORT TOP CANDIDATE REPORT (EXCEL)
# ============================================================

ranking_df.head(20).to_excel(
    os.path.join(STATIC_PATH, "top_exoplanets.xlsx"),
    index=False
)

print("Dashboard charts & report generated successfully.")
