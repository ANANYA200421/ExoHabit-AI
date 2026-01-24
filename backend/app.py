from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import configuration and utilities
from config import Config, MODELS, FEATURE_COLS, PREDICTIONS_PATH
from utils.model_loader import ModelLoader

app = Flask(__name__)
CORS(app)

# Initialize model loader with SINGLE predictions file
model_loader = ModelLoader(MODELS, PREDICTIONS_PATH)

# Load active model
model, predictions_df, model_name = model_loader.get_active_model()

print("=" * 80)
print("🚀 EXOHABITAI BACKEND API")
print("=" * 80)
print(f"Active Model: {model_name}")
print(f"Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
print(f"Dataset Status: {'✓ Loaded' if predictions_df is not None else '✗ Not Loaded'}")
if predictions_df is not None:
    print(f"Total Planets: {len(predictions_df)}")
print("=" * 80 + "\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_input(data):
    """Validate input data structure"""
    missing_features = [f for f in FEATURE_COLS if f not in data]
    if missing_features:
        return False, f"Missing required features: {missing_features}"
    
    for feature in FEATURE_COLS:
        try:
            float(data[feature])
        except (ValueError, TypeError):
            return False, f"Invalid value for feature '{feature}': must be numeric"
    
    return True, "Valid"


def format_prediction_response(prediction, probability, input_data):
    """Format prediction into a structured response"""
    return {
        'prediction': {
            'habitable': bool(prediction),
            'label': 'Habitable' if prediction == 1 else 'Not Habitable',
            'confidence': float(probability),
            'confidence_percent': f"{float(probability) * 100:.2f}%"
        },
        'model_used': model_name,
        'input_features': input_data,
        'timestamp': datetime.now().isoformat()
    }


def get_prediction_columns():
    """Get the prediction column names for active model"""
    for key, config in MODELS.items():
        if config['active']:
            return config['pred_column'], config['prob_column']
    # Default columns
    return 'predicted_habitable', 'habitability_probability'


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """API Home - Health Check"""
    return jsonify({
        'status': 'online',
        'message': 'ExoHabitAI Backend API',
        'version': '1.0.0',
        'model_loaded': model is not None,
        'active_model': model_name,
        'available_models': model_loader.list_available_models(),
        'endpoints': {
            'POST /predict': 'Predict habitability for a single exoplanet',
            'POST /predict_batch': 'Predict habitability for multiple exoplanets',
            'GET /top_habitable': 'Get top N most habitable planets',
            'GET /model_info': 'Get model performance information',
            'GET /stats': 'Get dataset statistics',
            'GET /models': 'List all available models',
            'POST /switch_model': 'Switch active model'
        }
    })


@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict habitability for a single exoplanet"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        features = pd.DataFrame([data])[FEATURE_COLS]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]
        
        response = format_prediction_response(prediction, probability, data)
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict habitability for multiple exoplanets"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        if 'planets' not in data or not isinstance(data['planets'], list):
            return jsonify({'error': 'Expected "planets" array in request body'}), 400
        
        planets = data['planets']
        if len(planets) == 0:
            return jsonify({'error': 'No planets provided'}), 400
        
        results = []
        for idx, planet_data in enumerate(planets):
            is_valid, message = validate_input(planet_data)
            if not is_valid:
                results.append({
                    'index': idx,
                    'error': message,
                    'status': 'failed'
                })
                continue
            
            features = pd.DataFrame([planet_data])[FEATURE_COLS]
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0, 1]
            
            results.append({
                'index': idx,
                'prediction': {
                    'habitable': bool(prediction),
                    'label': 'Habitable' if prediction == 1 else 'Not Habitable',
                    'confidence': float(probability),
                    'confidence_percent': f"{float(probability) * 100:.2f}%"
                },
                'status': 'success'
            })
        
        return jsonify({
            'total_planets': len(planets),
            'successful_predictions': sum(1 for r in results if r.get('status') == 'success'),
            'model_used': model_name,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@app.route('/top_habitable', methods=['GET'])
def get_top_habitable():
    """Get top N most habitable planets from the predictions dataset"""
    try:
        if predictions_df is None:
            return jsonify({'error': 'Predictions dataset not available'}), 404
        
        limit = request.args.get('limit', default=10, type=int)
        min_confidence = request.args.get('min_confidence', default=0.5, type=float)
        
        # Get column names for active model
        pred_col, prob_col = get_prediction_columns()
        
        # Check if columns exist
        if pred_col not in predictions_df.columns or prob_col not in predictions_df.columns:
            return jsonify({'error': f'Prediction columns not found in dataset'}), 404
        
        habitable_planets = predictions_df[
            (predictions_df[pred_col] == 1) & 
            (predictions_df[prob_col] >= min_confidence)
        ].nlargest(limit, prob_col)
        
        results = []
        for _, planet in habitable_planets.iterrows():
            results.append({
                'name': planet.get('pl_name', 'Unknown'),
                'host_star': planet.get('hostname', 'Unknown'),
                'confidence': float(planet[prob_col]),
                'confidence_percent': f"{float(planet[prob_col]) * 100:.2f}%",
                'features': {
                    'radius': float(planet.get('pl_rade', 0)) if pd.notna(planet.get('pl_rade')) else None,
                    'mass': float(planet.get('pl_bmasse', 0)) if pd.notna(planet.get('pl_bmasse')) else None,
                    'orbital_period': float(planet.get('pl_orbper', 0)) if pd.notna(planet.get('pl_orbper')) else None,
                    'insolation': float(planet.get('pl_insol', 0)) if pd.notna(planet.get('pl_insol')) else None
                }
            })
        
        return jsonify({
            'count': len(results),
            'limit': limit,
            'min_confidence': min_confidence,
            'model_used': model_name,
            'planets': results,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve top planets: {str(e)}'}), 500


@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get model information and performance metrics"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        info = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'features': FEATURE_COLS,
            'feature_count': len(FEATURE_COLS),
            'model_loaded': True
        }
        
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            info['feature_importance'] = importance
        
        if hasattr(model, 'get_params'):
            info['parameters'] = model.get_params()
        
        return jsonify(info), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve model info: {str(e)}'}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get dataset statistics"""
    try:
        if predictions_df is None:
            return jsonify({'error': 'Predictions dataset not available'}), 404
        
        # Get column names
        pred_col, prob_col = get_prediction_columns()
        
        if pred_col not in predictions_df.columns or prob_col not in predictions_df.columns:
            return jsonify({'error': f'Prediction columns not found in dataset'}), 404
        
        total_planets = len(predictions_df)
        habitable_count = int(predictions_df[pred_col].sum())
        non_habitable_count = total_planets - habitable_count
        
        avg_confidence = float(predictions_df[prob_col].mean())
        max_confidence = float(predictions_df[prob_col].max())
        min_confidence = float(predictions_df[prob_col].min())
        
        stats = {
            'total_planets': total_planets,
            'habitable_planets': habitable_count,
            'non_habitable_planets': non_habitable_count,
            'habitable_percentage': f"{(habitable_count / total_planets * 100):.2f}%",
            'model_used': model_name,
            'confidence_stats': {
                'average': f"{avg_confidence * 100:.2f}%",
                'maximum': f"{max_confidence * 100:.2f}%",
                'minimum': f"{min_confidence * 100:.2f}%"
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve stats: {str(e)}'}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List all available models"""
    models_list = model_loader.list_available_models()
    return jsonify({
        'models': models_list,
        'active_model': model_name
    }), 200


@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    global model, predictions_df, model_name
    
    try:
        data = request.json
        model_key = data.get('model_key')
        
        if not model_key:
            return jsonify({'error': 'model_key is required'}), 400
        
        new_model, new_predictions, new_name = model_loader.switch_model(model_key)
        
        if new_model is None:
            return jsonify({'error': f'Failed to load model: {model_key}'}), 500
        
        model = new_model
        predictions_df = new_predictions
        model_name = new_name
        
        return jsonify({
            'message': f'Successfully switched to {new_name}',
            'active_model': new_name,
            'model_key': model_key
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Failed to switch model: {str(e)}'}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    print("\n📡 Starting Flask server...")
    print(f"Access API at: http://{Config.HOST}:{Config.PORT}")
    print("=" * 80 + "\n")
    
    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT
    )