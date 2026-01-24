import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data paths - SINGLE FILES FOR ALL MODELS
DATA_DIR = BASE_DIR / 'data'
DATASET_PATH = DATA_DIR / 'Planets data after sorting.xlsx'
PREDICTIONS_PATH = DATA_DIR / 'predictions.csv'  # Single predictions file

# Model configurations
MODELS = {
    'random_forest': {
        'name': 'Random Forest',
        'path': BASE_DIR / 'models' / 'rf_exoplanet_model.pkl',
        'active': True,  # Set this to True for the model you want to use
        'pred_column': 'predicted_habitable',  # Column name in predictions.csv
        'prob_column': 'habitability_probability'
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'path': BASE_DIR / 'models' / 'gb_exoplanet_model.pkl',
        'active': False,
        'pred_column': 'predicted_habitable',
        'prob_column': 'habitability_probability'
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'path': BASE_DIR / 'models' / 'lr_exoplanet_model.pkl',
        'active': False,
        'pred_column': 'predicted_habitable',
        'prob_column': 'habitability_probability'
    },
    'neural_network': {
        'name': 'Neural Network',
        'path': BASE_DIR / 'models' / 'nn_exoplanet_model.pkl',
        'active': False,
        'pred_column': 'predicted_habitable',
        'prob_column': 'habitability_probability'
    }
}

# Get active model
def get_active_model():
    for key, config in MODELS.items():
        if config['active']:
            return key, config
    # Default to random_forest if none active
    return 'random_forest', MODELS['random_forest']

# Feature columns (must match training data)
FEATURE_COLS = [
    'pl_orbper_scaled',
    'pl_orbsmax_scaled',
    'pl_rade_scaled',
    'pl_bmasse_scaled',
    'pl_orbeccen_scaled',
    'pl_insol_scaled',
    'pl_eqt_scaled',      # Equilibrium Temperature (NEW)
    'st_teff_scaled',
    'st_rad_scaled',
    'st_mass_scaled',
    'st_met_scaled'
]

# Flask configuration
class Config:
    DEBUG = os.getenv('FLASK_DEBUG', True)
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    # Model settings
    MODEL_NAME, MODEL_CONFIG = get_active_model()
    MODEL_PATH = MODEL_CONFIG['path']