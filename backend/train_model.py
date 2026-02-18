import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("../data/processed/exoplanet_cleaned.csv")

# ---------- TARGET ----------
df["habitable"] = (df["habitability_score"] >= 0.45).astype(int)

# ---------- FEATURES (UPDATED) ----------
features = [
    "pl_rade",
    "pl_bmasse",      # planet mass
    "pl_orbper",
    "pl_eqt",
    "st_teff",
    "stellar_compatibility"
]

X = df[features]
y = df["habitable"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("model", RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "../models/habitability_model.pkl")
print("Model trained and saved.")
