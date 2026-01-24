import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RANDOM FOREST - EXOPLANET HABITABILITY PREDICTION")
print("=" * 80)

# LOADING DATA
print("\n Loading data...")
df = pd.read_csv('../data/More data added.csv')
print(f"✓ Loaded {len(df)} exoplanets")

# CREATE HABITABILITY LABELS
print("\n Creating habitability labels...")

def assign_habitability(row):
    score = 0
    
    # 1. Planet size (0.5-2.0 Earth radii = rocky planet)
    if pd.notna(row['pl_rade']) and 0.5 <= row['pl_rade'] <= 2.0:
        score += 2
    elif pd.notna(row['pl_rade']) and 2.0 < row['pl_rade'] <= 4.0:
        score += 1
    
    # 2. Insolation flux (0.25-4.0 Earth flux for habitable zone)
    if pd.notna(row['pl_insol']) and 0.25 <= row['pl_insol'] <= 4.0:
        score += 3
    elif pd.notna(row['pl_insol']) and 0.1 <= row['pl_insol'] <= 10:
        score += 1
    
    # 3. Equilibrium temperature (200-320 K for liquid water)
    if pd.notna(row['pl_eqt']) and 200 <= row['pl_eqt'] <= 320:
        score += 2
    elif pd.notna(row['pl_eqt']) and 150 <= row['pl_eqt'] <= 400:
        score += 1
    
    # 4. Orbital eccentricity (< 0.3 for stable orbit)
    if pd.notna(row['pl_orbeccen']) and row['pl_orbeccen'] <= 0.3:
        score += 1
    
    # 5. Stellar temperature (2700-7200 K)
    if pd.notna(row['st_teff']) and 2700 <= row['st_teff'] <= 7200:
        score += 1
    
    # 6. Planet mass (0.1-10 Earth masses for atmosphere retention)
    if pd.notna(row['pl_bmasse']) and 0.1 <= row['pl_bmasse'] <= 10:
        score += 1
    
    # 7. Stellar mass (0.08-1.5 solar masses)
    if pd.notna(row['st_mass']) and 0.08 <= row['st_mass'] <= 1.5:
        score += 1
    
    # Need score >= 6 to be considered habitable
    return 1 if score >= 6 else 0

df['habitable'] = df.apply(assign_habitability, axis=1)
print(f"✓ Habitable: {df['habitable'].sum()} ({df['habitable'].mean()*100:.1f}%)")

# PREPARING FEATURES - UPDATED WITH pl_eqt_scaled
print("\n Preparing features...")
feature_cols = [
    'pl_orbper_scaled', 
    'pl_orbsmax_scaled', 
    'pl_rade_scaled',
    'pl_bmasse_scaled', 
    'pl_orbeccen_scaled', 
    'pl_insol_scaled',
    'pl_eqt_scaled',      # NEW FEATURE ADDED
    'st_teff_scaled', 
    'st_rad_scaled', 
    'st_mass_scaled', 
    'st_met_scaled'
]

df_clean = df.dropna(subset=feature_cols + ['habitable'])
print(f"✓ Clean dataset: {len(df_clean)} exoplanets")
print(f"✓ Total features: {len(feature_cols)}")

X = df_clean[feature_cols]
y = df_clean['habitable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TRAIN RANDOM FOREST MODEL
print("\n Training Random Forest Model...")
print("-" * 80)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
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
plt.title('Random Forest - Feature Importance (with Equilibrium Temperature)')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: rf_feature_importance.png")

# SAVE MODEL
joblib.dump(model, '../models/rf_exoplanet_model.pkl')
print("✓ Saved: ../models/rf_exoplanet_model.pkl")

# PREDICTIONS - Save with standard column names for backend
df_clean['predicted_habitable'] = model.predict(X)
df_clean['habitability_probability'] = model.predict_proba(X)[:, 1]

# Save to data folder
df_clean.to_csv('../data/predictions.csv', index=False)
print("✓ Saved: ../data/predictions.csv")

# Also save with rf prefix for reference
df_clean.to_csv('rf_predictions.csv', index=False)
print("✓ Saved: rf_predictions.csv (backup)")

print("\n" + "=" * 80)
print("RANDOM FOREST MODEL COMPLETE")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"  • Features used: {len(feature_cols)}")
print(f"  • Including: pl_eqt_scaled (Equilibrium Temperature)")
print(f"  • Model accuracy: {accuracy*100:.2f}%")
print(f"  • Habitable planets found: {df_clean['predicted_habitable'].sum()}")
print("=" * 80)