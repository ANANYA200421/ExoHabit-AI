import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and pipeline
model = joblib.load("../models/best_model_random_forest.pkl")
pipeline = joblib.load("../models/preprocessing_pipeline.pkl")

importances = model.feature_importances_

# Try extracting names safely
try:
    feature_names = pipeline.get_feature_names_out()
except:
    feature_names = [f"Feature_{i}" for i in range(len(importances))]

# Align lengths safely
feature_names = feature_names[:len(importances)]

df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
plt.barh(df["Feature"], df["Importance"])
plt.xlabel("Importance")
plt.title("Feature Importance - Habitability Model")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("../frontend/static/feature_importance.png")
plt.show()
