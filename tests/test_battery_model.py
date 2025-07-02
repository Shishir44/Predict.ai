"""
Test Module for Battery Prediction System

This module contains unit tests for the battery prediction system.
"""

import unittest
import numpy as np
import tensorflow as tf
from src.model_training.battery_model import BatteryLSTMModel
from src.model_training.trainer import BatteryModelTrainer
from src.inference.battery_predictor import BatteryPredictor
from config.model_config import ProjectConfig

class TestBatteryModel(unittest.TestCase):
    """
    Test class for battery prediction models.
    """
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProjectConfig()
        self.model = BatteryLSTMModel(
            sequence_length=self.config.ml_model_config.sequence_length,
            num_features=self.config.ml_model_config.num_features
        )
        
    def test_model_structure(self):
        """Test model architecture."""
        self.assertIsNotNone(self.model.lstm1)
        self.assertIsNotNone(self.model.lstm2)
        self.assertIsNotNone(self.model.soh_output)
        self.assertIsNotNone(self.model.soc_output)
        
    def test_model_prediction(self):
        """Test model prediction."""
        # Create dummy input data
        batch_size = 2
        input_shape = (batch_size, self.config.ml_model_config.sequence_length,
                      self.config.ml_model_config.num_features)
        
        inputs = np.random.random(input_shape).astype(np.float32)
        
        # Make predictions
        predictions = self.model(inputs)
        
        # Check output shapes
        self.assertIn('soh_output', predictions)
        self.assertIn('soc_output', predictions)
        self.assertEqual(predictions['soh_output'].shape, (batch_size, 1))
        self.assertEqual(predictions['soc_output'].shape, (batch_size, 1))
        
    def test_trainer(self):
        """Test model training pipeline."""
        trainer = BatteryModelTrainer(self.config)
        
        # Create dummy data
        X = np.random.random((100, self.config.ml_model_config.sequence_length,
                            self.config.ml_model_config.num_features))
        y_soh = np.random.random((100, 1))
        y_soc = np.random.random((100, 1))
        
        # Prepare data
        data_dict = trainer.prepare_data(X, y_soh, y_soc)
        
        # Train model with reduced epochs for testing
        trainer.config.ml_model_config.epochs = 1  # Override for testing
        history = trainer.train_model(data_dict)
        
        # Check history
        self.assertIn('loss', history.history)
        self.assertIn('soh_output_mae', history.history)
        self.assertIn('soc_output_mae', history.history)
        
    def test_inference(self):
        """Test model inference."""
        from src.inference.production_predictor import ProductionBatteryPredictor
        
        # Create predictor (will use available models)
        predictor = ProductionBatteryPredictor()
        
        # Test with feature dictionary
        test_features = {
            'cycle_number': 100,
            'capacity_ah': 1.8,
            'ambient_temperature': 24.0,
            'capacity_normalized': 0.8,
            'cycle_progress': 0.5,
            'temp_deviation': 0.0
        }
        
        # Make prediction
        result = predictor.predict_battery_health(test_features)
        
        # Check results
        self.assertIsInstance(result.soh_prediction, float)
        self.assertIsInstance(result.soc_prediction, float)
        self.assertIsInstance(result.soh_confidence, float)
        self.assertIsInstance(result.soc_confidence, float)
        self.assertGreater(result.prediction_time_ms, 0)
        self.assertIsNotNone(result.model_version)
        
        # Test health status
        health = predictor.get_health_status()
        self.assertIn('status', health)
        self.assertIn('models_loaded', health)
        
if __name__ == '__main__':
    unittest.main()
