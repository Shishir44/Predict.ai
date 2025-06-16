"""
Battery Prediction Module

This module provides functionality for making predictions using trained models.
"""

import numpy as np
import tensorflow as tf
from typing import Dict
from config.model_config import ProjectConfig

class BatteryPredictor:
    """
    Class for making SOH and SOC predictions.
    
    Args:
        model_path: Path to the trained model
    """
    def __init__(self, model_path: str):
        """Initialize the predictor."""
        self.model = tf.keras.models.load_model(model_path)
        self.feature_scaler = None
        self.config = ProjectConfig()
        
    def predict_soh_soc(self, battery_data: np.ndarray) -> dict:
        """
        Predict SOH and SOC from battery data.
        
        Args:
            battery_data: Input data array
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        # Preprocess data
        processed_data = self._preprocess_input(battery_data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        return {
            'soh': float(predictions['soh_output'][0][0]),
            'soc': float(predictions['soc_output'][0][0]),
            'confidence_soh': self._calculate_confidence(predictions['soh_output']),
            'confidence_soc': self._calculate_confidence(predictions['soc_output'])
        }
    
    def _preprocess_input(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Raw input data
            
        Returns:
            Processed data ready for prediction
        """
        # Apply same preprocessing as training
        if self.feature_scaler:
            data = self.feature_scaler.transform(data)
        
        # Reshape for LSTM input
        return data.reshape(1, -1, data.shape[-1])
    
    def _calculate_confidence(self, prediction: np.ndarray) -> float:
        """
        Calculate prediction confidence score.
        
        Args:
            prediction: Model prediction output
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence based on prediction probability
        pred_val = prediction[0][0]
        confidence = 1 - 2 * abs(pred_val - 0.5)  # Higher confidence for values away from 0.5
        return max(0.0, confidence)
    
    def evaluate_model(self, test_data: np.ndarray, true_values: Dict[str, np.ndarray]) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test input data
            true_values: Dictionary of true SOH and SOC values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict_soh_soc(test_data)
        
        metrics = {
            'mae_soh': np.abs(predictions['soh'] - true_values['soh']).mean(),
            'mae_soc': np.abs(predictions['soc'] - true_values['soc']).mean(),
            'rmse_soh': np.sqrt(np.mean((predictions['soh'] - true_values['soh'])**2)),
            'rmse_soc': np.sqrt(np.mean((predictions['soc'] - true_values['soc'])**2))
        }
        
        return metrics
