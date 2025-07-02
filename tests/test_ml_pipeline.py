import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import logging
from pathlib import Path
import tempfile
import shutil
import sys

from src.data_collection.battery_data_collector import BatteryDataCollector
from src.scripts.preprocess_data import BatteryDataPreprocessor
from src.scripts.train_lstm_model import LSTMModelTrainer
from src.scripts.train_transformer_model import TransformerTrainer
from src.scripts.compare_models import ModelComparator
from src.scripts.deploy_model import ModelDeployer
from src.scripts.monitor_model import ModelMonitor
from src.scripts.retrain_model import ModelRetrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / 'data'
        self.data_dir.mkdir()

        # Create sample data
        self.create_sample_data()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)

    def create_sample_data(self):
        """Create sample battery data for testing."""
        # Generate sample data
        np.random.seed(42)
        num_samples = 1000

        data = {
            'battery_id': np.random.randint(1, 5, num_samples),
            'cycle': np.random.randint(1, 100, num_samples),
            'capacity': np.random.uniform(0.5, 1.0, num_samples),
            'voltage': np.random.uniform(3.0, 4.5, num_samples),
            'current': np.random.uniform(-5.0, 5.0, num_samples),
            'temperature': np.random.uniform(20.0, 40.0, num_samples),
            'SOH': np.random.uniform(0.7, 1.0, num_samples),
            'SOC': np.random.uniform(0.0, 1.0, num_samples)
        }

        df = pd.DataFrame(data)
        df.to_csv(self.data_dir / 'battery_data.csv', index=False)

    def test_data_collection(self):
        """Test data collection functionality."""
        collector = BatteryDataCollector(data_dir=str(self.data_dir))
        collector.collect_synthetic_data()

        # Verify data collection
        self.assertTrue((self.data_dir / 'synthetic_battery_data.csv').exists())
        df = pd.read_csv(self.data_dir / 'synthetic_battery_data.csv')
        self.assertGreater(len(df), 0)

    def test_data_preprocessing(self):
        """Test data preprocessing pipeline."""
        preprocessor = BatteryDataPreprocessor(data_dir=str(self.data_dir))
        result = preprocessor.prepare_data_for_training()
        
        if result is None:
            # Skip test if no data available
            self.skipTest("No data available for preprocessing test")
        
        (X_train, X_val, X_test), (y_soh_train, y_soh_val, y_soh_test), \
        (y_soc_train, y_soc_val, y_soc_test) = result

        # Verify data shapes
        self.assertEqual(len(X_train), len(y_soh_train))
        self.assertEqual(len(X_train), len(y_soc_train))
        self.assertEqual(len(X_val), len(y_soh_val))
        self.assertEqual(len(X_val), len(y_soc_val))
        self.assertEqual(len(X_test), len(y_soh_test))
        self.assertEqual(len(X_test), len(y_soc_test))

    def test_lstm_model_training(self):
        """Test LSTM model training."""
        preprocessor = BatteryDataPreprocessor(data_dir=str(self.data_dir))
        result = preprocessor.prepare_data_for_training()
        
        if result is None:
            self.skipTest("No data available for LSTM training test")
        
        (X_train, _, _), (y_soh_train, _, _), (y_soc_train, _, _) = result

        trainer = LSTMModelTrainer()
        history = trainer.train_model(
            X_train=X_train,
            y_soh_train=y_soh_train,
            y_soc_train=y_soc_train,
            epochs=1  # Reduced epochs for testing
        )

        # Verify training completed
        self.assertTrue('loss' in history.history)
        self.assertTrue('soh_output_mse' in history.history)
        self.assertTrue('soc_output_mse' in history.history)

    def test_transformer_model_training(self):
        """Test Transformer model training."""
        preprocessor = BatteryDataPreprocessor(data_dir=str(self.data_dir))
        result = preprocessor.prepare_data_for_training()
        
        if result is None:
            self.skipTest("No data available for Transformer training test")
        
        (X_train, _, _), (y_soh_train, _, _), (y_soc_train, _, _) = result

        trainer = TransformerTrainer()
        trainer.train_model(
            X_train=X_train,
            y_soh_train=y_soh_train,
            y_soc_train=y_soc_train,
            epochs=2  # Reduced epochs for testing
        )

        # Verify model exists
        model_path = Path("models/transformer/transformer_model.pth")
        self.assertTrue(model_path.exists())

    def test_model_comparison(self):
        """Test model comparison functionality."""
        from src.inference.production_predictor import ProductionBatteryPredictor
        
        # Create predictor
        predictor = ProductionBatteryPredictor()
        
        # Test features
        test_features = {
            'cycle_number': 100,
            'capacity_ah': 1.8,
            'ambient_temperature': 24.0,
            'capacity_normalized': 0.8,
            'cycle_progress': 0.5,
            'temp_deviation': 0.0
        }
        
        # Test Random Forest prediction
        rf_result = predictor.predict_battery_health(test_features, model_type="random_forest")
        self.assertIsInstance(rf_result.soh_prediction, float)

        # Test model info
        model_info = predictor.get_model_info()
        self.assertIn('available_models', model_info)
        self.assertIn('total_predictions', model_info)
        
        # Test health status
        health = predictor.get_health_status()
        self.assertIn('status', health)
        self.assertIn('models_loaded', health)

    def test_model_deployment(self):
        """Test model deployment functionality."""
        from src.deployment.tensorflow_serving import TensorFlowServingDeployer
        
        # Create deployer
        deployer = TensorFlowServingDeployer(serving_dir=str(Path(self.test_dir) / 'serving'))

        # Check if LSTM model exists
        lstm_path = Path("models/lstm_soh_model.h5")
        if lstm_path.exists():
            # Test model preparation
            result = deployer.prepare_model_for_serving(str(lstm_path))
            self.assertEqual(result['status'], 'success')
            
            # Test model status
            status = deployer.get_model_status()
            self.assertIn('model_ready', status)
            
            # Test prediction (if model is prepared)
            if status['model_ready']:
                test_input = np.random.random((1, 6))  # 6 features
                pred_result = deployer.make_prediction(test_input)
                self.assertIn('status', pred_result)
        else:
            self.skipTest("LSTM model file not found for deployment test")

    def test_model_monitoring(self):
        """Test model monitoring functionality."""
        from src.monitoring.monitoring_service import MonitoringService
        from src.inference.production_predictor import ProductionBatteryPredictor
        
        # Create predictor and monitoring service
        predictor = ProductionBatteryPredictor()
        monitor = MonitoringService(predictor, db_path=str(Path(self.test_dir) / 'test_monitoring.db'))
        
        # Test features
        test_features = {
            'cycle_number': 100,
            'capacity_ah': 1.8,
            'ambient_temperature': 24.0,
            'capacity_normalized': 0.8,
            'cycle_progress': 0.5,
            'temp_deviation': 0.0
        }
        
        # Make a prediction
        result = predictor.predict_battery_health(test_features)
        
        # Record prediction for monitoring
        monitor.record_prediction(test_features, result)
        
        # Test drift detection
        drift_result = monitor.detect_drift(test_features)
        self.assertIn('drift_detected', drift_result)
        self.assertIn('drift_score', drift_result)

        # Test metrics
        metrics = monitor.get_metrics(hours=1)
        self.assertIn('prediction_count', metrics)
        self.assertIn('performance', metrics)

    def test_model_retraining(self):
        """Test model retraining functionality."""
        retrainer = ModelRetrainer()

        results = retrainer.retrain()

        # Verify result structure (should skip without metrics file)
        self.assertIn('status', results)
        self.assertEqual(results['status'], 'skipped')

if __name__ == '__main__':
    unittest.main()
