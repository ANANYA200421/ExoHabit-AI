
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EXOPLANET HABITABILITY PREDICTION MODEL")
print("=" * 80)


# LOADING DATA

print("\n Loading data...")


df = pd.read_excel('exoplanets_with_predictions.xlsx')

print(f"✓ Loaded {len(df)} exoplanets")
print(f"✓ Features: {df.shape[1]} columns")


# CREATE HABITABILITY LABELS

print("\n Creating habitability labels...")

def assign_habitability(row):
    """
    Score planets based on habitability criteria
    Returns 1 if potentially habitable, 0 otherwise
    """
    score = 0
    
    # 1. Planet size (0.5-2.0 Earth radii = rocky planet)
    if pd.notna(row['pl_rade']):
        if 0.5 <= row['pl_rade'] <= 2.0:
            score += 2  # Strong indicator
        elif 2.0 < row['pl_rade'] <= 4.0:
            score += 1  # Possible mini-Neptune
    
    # 2. Habitable zone (0.25-4.0 Earth flux) 
    if pd.notna(row['pl_insol']):
        if 0.25 <= row['pl_insol'] <= 4.0:
            score += 3  # Most important factor
        elif 0.1 <= row['pl_insol'] <= 10:
            score += 1
    
    # 3. Orbital stability (eccentricity < 0.3)
    if pd.notna(row['pl_orbeccen']):
        if row['pl_orbeccen'] <= 0.3:
            score += 1
    
    # 4. Stellar temperature (2700-7200 K)
    if pd.notna(row['st_teff']):
        if 2700 <= row['st_teff'] <= 7200:
            score += 1
    
    # 5. Planet mass (0.1-10 Earth masses for atmosphere)
    if pd.notna(row['pl_bmasse']):
        if 0.1 <= row['pl_bmasse'] <= 10:
            score += 1
    
    # 6. Stellar mass (0.08-1.5 solar masses)
    if pd.notna(row['st_mass']):
        if 0.08 <= row['st_mass'] <= 1.5:
            score += 1
    
    # Need score >= 6 to be considered habitable
    return 1 if score >= 6 else 0

# Apply labeling
df['habitable'] = df.apply(assign_habitability, axis=1)

print(f"✓ Habitable planets: {df['habitable'].sum()} ({df['habitable'].mean()*100:.1f}%)")
print(f"✓ Non-habitable planets: {(df['habitable'] == 0).sum()} ({(1-df['habitable'].mean())*100:.1f}%)")


#  PREPARING FEATURES

print("\n Preparing features...")

# Using SCALED columns (since you already scaled them)
feature_cols = [
    'pl_orbper_scaled',
    'pl_orbsmax_scaled',
    'pl_rade_scaled',
    'pl_bmasse_scaled',
    'pl_orbeccen_scaled',
    'pl_insol_scaled',
    'st_teff_scaled',
    'st_rad_scaled',
    'st_mass_scaled',
    'st_met_scaled'
]

# Remove rows with missing scaled features
df_clean = df.dropna(subset=feature_cols + ['habitable'])

print(f"✓ Clean dataset: {len(df_clean)} exoplanets")
print(f"✓ Dropped {len(df) - len(df_clean)} rows with missing values")

# Prepare X (features) and y (target)
X = df_clean[feature_cols]
y = df_clean['habitable']

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# TRAIN MULTIPLE MODELS

print("\n[Step 4] Training models...")
print("-" * 80)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
}

results = {}
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
    
    # Print results
    print(f"  → Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  → AUC-ROC: {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Habitable', 'Habitable']))

print("-" * 80)
print(f"\n🏆 BEST MODEL: {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)")


#  FEATURE IMPORTANCE ANALYSIS

print("\n Analyzing feature importance...")

best_model = results[best_model_name]['model']

# Get feature importance (works for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance Rankings:")
    print(importance_df.to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved feature importance plot: feature_importance.png")


#  SAVE MODEL AND MAKE PREDICTIONS

print("\n[Step 6] Saving model and generating predictions...")

# Save the best model
joblib.dump(best_model, 'exoplanet_habitability_model.pkl')
print("✓ Saved model: exoplanet_habitability_model.pkl")

# Add predictions to the original dataframe
df_clean['predicted_habitable'] = best_model.predict(X)
df_clean['habitability_probability'] = best_model.predict_proba(X)[:, 1]

# Save results
output_file = 'exoplanets_with_predictions.csv'
df_clean.to_csv(output_file, index=False)
print(f"✓ Saved predictions: {output_file}")

#  SUMMARY AND EXAMPLES

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n📊 Dataset Statistics:")
print(f"  • Total exoplanets analyzed: {len(df_clean)}")
print(f"  • Labeled as habitable: {df_clean['habitable'].sum()}")
print(f"  • Predicted as habitable: {df_clean['predicted_habitable'].sum()}")

print(f"\n🎯 Model Performance:")
print(f"  • Best Model: {best_model_name}")
print(f"  • Accuracy: {best_accuracy*100:.2f}%")
print(f"  • AUC-ROC: {results[best_model_name]['auc']:.4f}")

print("\n🌍 Sample Predictions (Top 5 Most Likely Habitable):")
top_habitable = df_clean.nlargest(5, 'habitability_probability')[
    ['pl_name', 'hostname', 'predicted_habitable', 'habitability_probability']
]
print(top_habitable.to_string(index=False))

print("\n" + "=" * 80)
print("✅ COMPLETE! Model is ready to use.")
print("=" * 80)

# ============================================================================
# BONUS: Function to predict new planets
# ============================================================================

# def predict_new_planet(planet_features_scaled):
#     """
#     Use this function to predict habitability for new exoplanets
    
#     Args:
#         planet_features_scaled: dict or DataFrame with scaled features
        
#     Example:
#         new_planet = {
#             'pl_orbper_scaled': -0.39,
#             'pl_orbsmax_scaled': 2.51,
#             'pl_rade_scaled': -0.06,
#             'pl_bmasse_scaled': -0.18,
#             'pl_orbeccen_scaled': 6.47,
#             'pl_insol_scaled': -0.25,
#             'st_teff_scaled': -6.98,
#             'st_rad_scaled': -1.33,
#             'st_mass_scaled': -3.18,
#             'st_met_scaled': 1.34
#         }
#         predict_new_planet(new_planet)
#     """
#     # Convert dict to DataFrame if needed
#     if isinstance(planet_features_scaled, dict):
#         planet_features_scaled = pd.DataFrame([planet_features_scaled])
    
#     # Make prediction
#     prediction = best_model.predict(planet_features_scaled)[0]
#     probability = best_model.predict_proba(planet_features_scaled)[0, 1]
    
#     print(f"\n🔮 Prediction: {'✅ HABITABLE' if prediction == 1 else '❌ NOT HABITABLE'}")
#     print(f"   Confidence: {probability*100:.1f}%")
    
#     return prediction, probability

print("\n💡 TIP: Use predict_new_planet() function to classify new exoplanets!")
print("   Example: predict_new_planet({'pl_rade_scaled': -0.06, ...})")