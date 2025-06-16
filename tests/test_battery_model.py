"""
Test Module for Battery Prediction System

This module contains unit tests for the battery prediction system.
"""

import unittest
import numpy as np
import tensorflow as tf
from src.model_training.battery_model import BatteryLSTMModel
from src.model_training.trainer import BatteryModelTrainer
from config.model_config import ProjectConfig

class TestBatteryModel(unittest.TestCase):
    """
    Test class for battery prediction models.
    """
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProjectConfig()
        self.model = BatteryLSTMModel(
            sequence_length=self.config.model_config.sequence_length,
            num_features=self.config.model_config.num_features
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
        input_shape = (batch_size, self.config.model_config.sequence_length,
                      self.config.model_config.num_features)
        
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
        X = np.random.random((100, self.config.model_config.sequence_length,
                            self.config.model_config.num_features))
        y_soh = np.random.random((100, 1))
        y_soc = np.random.random((100, 1))
        
        # Prepare data
        data_dict = trainer.prepare_data(X, y_soh, y_soc)
        
        # Train model
        history = trainer.train_model(data_dict)
        
        # Check history
        self.assertIn('loss', history.history)
        self.assertIn('soh_output_mae', history.history)
        self.assertIn('soc_output_mae', history.history)
        
    def test_inference(self):
        """Test model inference."""
        # Create dummy data for inference
        test_data = np.random.random((1, self.config.model_config.sequence_length,
                                    self.config.model_config.num_features))
        
        # Create predictor
        predictor = BatteryPredictor(self.config.model_config.model_save_path)
        
        # Make predictions
        results = predictor.predict_soh_soc(test_data)
        
        # Check results
        self.assertIn('soh', results)
        self.assertIn('soc', results)
        self.assertIn('confidence_soh', results)
        self.assertIn('confidence_soc', results)
        
if __name__ == '__main__':
    unittest.main()
