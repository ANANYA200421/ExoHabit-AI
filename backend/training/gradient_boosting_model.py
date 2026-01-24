import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GRADIENT BOOSTING - EXOPLANET HABITABILITY PREDICTION")
print("=" * 80)

# LOADING DATA
print("\n Loading data...")
df = pd.read_csv('../d/More data added.csv')
print(f"✓ Loaded {len(df)} exoplanets")

# CREATE HABITABILITY LABELS
print("\n Creating habitability labels...")

def assign_habitability(row):
    score = 0
    if pd.notna(row['pl_rade']) and 0.5 <= row['pl_rade'] <= 2.0:
        score += 2
    elif pd.notna(row['pl_rade']) and 2.0 < row['pl_rade'] <= 4.0:
        score += 1
    
    if pd.notna(row['pl_insol']) and 0.25 <= row['pl_insol'] <= 4.0:
        score += 3
    elif pd.notna(row['pl_insol']) and 0.1 <= row['pl_insol'] <= 10:
        score += 1
    
    if pd.notna(row['pl_orbeccen']) and row['pl_orbeccen'] <= 0.3:
        score += 1
    if pd.notna(row['st_teff']) and 2700 <= row['st_teff'] <= 7200:
        score += 1
    if pd.notna(row['pl_bmasse']) and 0.1 <= row['pl_bmasse'] <= 10:
        score += 1
    if pd.notna(row['st_mass']) and 0.08 <= row['st_mass'] <= 1.5:
        score += 1
    
    return 1 if score >= 6 else 0

df['habitable'] = df.apply(assign_habitability, axis=1)
print(f"✓ Habitable: {df['habitable'].sum()} ({df['habitable'].mean()*100:.1f}%)")

# PREPARING FEATURES
print("\n Preparing features...")
feature_cols = [
    'pl_orbper_scaled', 'pl_orbsmax_scaled', 'pl_rade_scaled',
    'pl_bmasse_scaled', 'pl_orbeccen_scaled', 'pl_insol_scaled',
    'pl_eqt_scaled', 'st_teff_scaled', 'st_rad_scaled', 
    'st_mass_scaled', 'st_met_scaled'
]

df_clean = df.dropna(subset=feature_cols + ['habitable'])
print(f"✓ Clean dataset: {len(df_clean)} exoplanets")

X = df_clean[feature_cols]
y = df_clean['habitable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TRAIN GRADIENT BOOSTING MODEL
print("\n Training Gradient Boosting Model...")
print("-" * 80)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# EVALUATE MODEL
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"  → Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  → AUC-ROC: {auc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Habitable', 'Habitable']))

# FEATURE IMPORTANCE
print("\n Analyzing feature importance...")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance Rankings:")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Gradient Boosting - Feature Importance')
plt.tight_layout()
plt.savefig('gb_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: gb_feature_importance.png")

# SAVE MODEL
joblib.dump(model, 'gb_exoplanet_model.pkl')
print("✓ Saved: gb_exoplanet_model.pkl")

# PREDICTIONS
df_clean['gb_predicted'] = model.predict(X)
df_clean['gb_probability'] = model.predict_proba(X)[:, 1]
df_clean.to_csv('gb_predictions.csv', index=False)
print("✓ Saved: gb_predictions.csv")

print("\n" + "=" * 80)
print("GRADIENT BOOSTING MODEL COMPLETE")
print("=" * 80)