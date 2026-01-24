"""
Model Loader Utility
Handles loading and managing multiple ML models with SINGLE shared dataset
"""

import joblib
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelLoader:
    """Load and manage multiple ML models using single shared dataset"""
    
    def __init__(self, models_config, predictions_path):
        """
        Initialize model loader
        
        Args:
            models_config: Dictionary with model configurations
            predictions_path: Path to single predictions CSV file
        """
        self.models_config = models_config
        self.predictions_path = predictions_path
        self.loaded_models = {}
        self.predictions_df = None
        
    def load_model(self, model_key):
        """
        Load a specific model
        
        Args:
            model_key: Key from models_config
            
        Returns:
            Loaded model object or None
        """
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        config = self.models_config.get(model_key)
        if not config:
            print(f"❌ Model '{model_key}' not found in configuration")
            return None
        
        model_path = config['path']
        
        try:
            if not model_path.exists():
                print(f"⚠️  Model file not found: {model_path}")
                return None
            
            model = joblib.load(model_path)
            self.loaded_models[model_key] = model
            print(f"✓ Loaded model: {config['name']} from {model_path.name}")
            return model
        
        except Exception as e:
            print(f"❌ Error loading model '{model_key}': {e}")
            return None
    
    def load_predictions(self):
        """
        Load predictions CSV (single file for all models)
        
        Returns:
            DataFrame or None
        """
        if self.predictions_df is not None:
            return self.predictions_df
        
        try:
            if not self.predictions_path.exists():
                print(f"⚠️  Predictions file not found: {self.predictions_path}")
                return None
            
            df = pd.read_csv(self.predictions_path)
            self.predictions_df = df
            print(f"✓ Loaded predictions: {len(df)} records from {self.predictions_path.name}")
            return df
        
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return None
    
    def load_all_models(self):
        """Load all models from configuration"""
        print("\n" + "=" * 80)
        print("LOADING ML MODELS")
        print("=" * 80)
        
        for key in self.models_config.keys():
            self.load_model(key)
        
        self.load_predictions()
        
        print("=" * 80)
        print(f"✓ Loaded {len(self.loaded_models)} model(s)")
        print("=" * 80 + "\n")
    
    def get_active_model(self):
        """Get the active model based on configuration"""
        for key, config in self.models_config.items():
            if config.get('active', False):
                model = self.load_model(key)
                predictions = self.load_predictions()
                return model, predictions, config['name']
        
        # Return first model as default
        first_key = list(self.models_config.keys())[0]
        model = self.load_model(first_key)
        predictions = self.load_predictions()
        return model, predictions, self.models_config[first_key]['name']
    
    def switch_model(self, model_key):
        """
        Switch to a different model
        
        Args:
            model_key: Key of model to activate
            
        Returns:
            (model, predictions, name) tuple or (None, None, None)
        """
        if model_key not in self.models_config:
            print(f"❌ Model '{model_key}' not found")
            return None, None, None
        
        # Deactivate all models
        for config in self.models_config.values():
            config['active'] = False
        
        # Activate selected model
        self.models_config[model_key]['active'] = True
        
        model = self.load_model(model_key)
        predictions = self.load_predictions()
        name = self.models_config[model_key]['name']
        
        print(f"✓ Switched to model: {name}")
        return model, predictions, name
    
    def get_model_info(self, model_key):
        """Get information about a specific model"""
        if model_key not in self.models_config:
            return None
        
        config = self.models_config[model_key]
        model = self.loaded_models.get(model_key)
        
        info = {
            'name': config['name'],
            'active': config.get('active', False),
            'loaded': model is not None,
            'path': str(config['path']),
        }
        
        if model:
            info['type'] = type(model).__name__
            if hasattr(model, 'feature_importances_'):
                info['has_feature_importance'] = True
        
        return info
    
    def list_available_models(self):
        """List all available models"""
        return [
            {
                'key': key,
                'name': config['name'],
                'active': config.get('active', False),
                'loaded': key in self.loaded_models
            }
            for key, config in self.models_config.items()
        ]