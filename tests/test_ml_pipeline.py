import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import logging
from pathlib import Path
import tempfile
import shutil

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
        collector.collect_data()
        
        # Verify data collection
        self.assertTrue((self.data_dir / 'battery_data.csv').exists())
        df = pd.read_csv(self.data_dir / 'battery_data.csv')
        self.assertGreater(len(df), 0)
        
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline."""
        preprocessor = BatteryDataPreprocessor(data_dir=str(self.data_dir))
        X_train, X_val, X_test, y_soh_train, y_soh_val, y_soh_test, \
        y_soc_train, y_soc_val, y_soc_test = preprocessor.prepare_data_for_training()
        
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
        X_train, _, _, y_soh_train, _, _, y_soc_train, _, _ = \
            preprocessor.prepare_data_for_training()
        
        trainer = LSTMModelTrainer()
        history = trainer.train_model(
            X_train=X_train,
            y_soh_train=y_soh_train,
            y_soc_train=y_soc_train,
            epochs=2  # Reduced epochs for testing
        )
        
        # Verify training completed
        self.assertTrue('loss' in history.history)
        self.assertTrue('soh_output_mse' in history.history)
        self.assertTrue('soc_output_mse' in history.history)
        
    def test_transformer_model_training(self):
        """Test Transformer model training."""
        preprocessor = BatteryDataPreprocessor(data_dir=str(self.data_dir))
        X_train, _, _, y_soh_train, _, _, y_soc_train, _, _ = \
            preprocessor.prepare_data_for_training()
        
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
        preprocessor = BatteryDataPreprocessor(data_dir=str(self.data_dir))
        _, _, X_test, _, y_soh_test, _, y_soc_test = \
            preprocessor.prepare_data_for_training()
        
        comparator = ModelComparator()
        metrics = comparator.evaluate_models(
            X_test=X_test,
            y_soh_test=y_soh_test,
            y_soc_test=y_soc_test
        )
        
        # Verify metrics
        self.assertIn('lstm', metrics)
        self.assertIn('transformer', metrics)
        self.assertIn('soh_mse', metrics['lstm'])
        self.assertIn('soc_mse', metrics['lstm'])
        
    def test_model_deployment(self):
        """Test model deployment functionality."""
        deployer = ModelDeployer()
        deployer.prepare_model_for_serving()
        
        # Verify model files
        serving_dir = Path("models/ensemble/serving")
        self.assertTrue(serving_dir.exists())
        self.assertTrue((serving_dir / '1/saved_model.pb').exists())
        
    def test_model_monitoring(self):
        """Test model monitoring functionality."""
        preprocessor = BatteryDataPreprocessor(data_dir=str(self.data_dir))
        _, _, X_test, _, y_soh_test, _, y_soc_test = \
            preprocessor.prepare_data_for_training()
        
        monitor = ModelMonitor()
        metrics = monitor.get_model_metrics(
            X=X_test[:10],
            y_true={'soh': y_soh_test[:10], 'soc': y_soc_test[:10]}
        )
        
        # Verify metrics
        self.assertIn('latency', metrics)
        self.assertIn('soh_mae', metrics)
        self.assertIn('soc_mae', metrics)
        
    def test_model_retraining(self):
        """Test model retraining functionality."""
        retrainer = ModelRetrainer()
        
        # Create sample monitoring metrics
        metrics = {
            'soh_mae': 0.15,  # Above threshold
            'soc_mae': 0.15,  # Above threshold
            'drift': 0.06,    # Above threshold
        }
        
        results = retrainer.retrain()
        
        # Verify retraining
        self.assertTrue(results['retrained'])
        self.assertIn('metrics', results)
        
if __name__ == '__main__':
    unittest.main()
