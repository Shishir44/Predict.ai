import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import tensorflow as tf
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .preprocess_data import BatteryDataPreprocessor
from .train_transformer_model import TransformerModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelMaintainer:
    def __init__(self,
                model_dir: str = "models/ensemble",
                maintenance_dir: str = "maintenance"):
        self.model_dir = Path(model_dir)
        self.maintenance_dir = Path(maintenance_dir)
        self.maintenance_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, model_type: str = "lstm") -> Any:
        """Load model for maintenance."""
        if model_type == "lstm":
            model_path = self.model_dir / 'lstm_model.h5'
            if not model_path.exists():
                raise FileNotFoundError(f"LSTM model not found at {model_path}")

            return tf.keras.models.load_model(str(model_path))

        elif model_type == "transformer":
            model_path = self.model_dir / 'transformer_model.pth'
            if not model_path.exists():
                raise FileNotFoundError(f"Transformer model not found at {model_path}")

            model = TransformerModel(input_dim=10)  # Adjust input_dim as needed
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def optimize_hyperparameters(self,
                               model: Any,
                               X_train: np.ndarray,
                               y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize model hyperparameters."""
        if isinstance(model, tf.keras.Model):
            # Define parameter grid
            param_grid = {
                'learning_rate': [0.001, 0.0001, 0.00001],
                'batch_size': [32, 64, 128],
                'dropout_rate': [0.1, 0.2, 0.3],
                'num_layers': [1, 2, 3]
            }

            # Create wrapper for Keras model
            from scikeras.wrappers import KerasRegressor

            def create_model(learning_rate=0.001, dropout_rate=0.1, num_layers=2):
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))

                for _ in range(num_layers - 1):
                    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
                    model.add(tf.keras.layers.Dropout(dropout_rate))

                model.add(tf.keras.layers.LSTM(64))
                model.add(tf.keras.layers.Dropout(dropout_rate))
                model.add(tf.keras.layers.Dense(1))

                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mse')
                return model

            # Create model wrapper
            model_wrapper = KerasRegressor(build_fn=create_model)

            # Perform grid search
            grid = GridSearchCV(
                estimator=model_wrapper,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_squared_error'
            )

            grid_result = grid.fit(X_train, y_train)

            return {
                'best_params': grid_result.best_params_,
                'best_score': grid_result.best_score_,
                'all_results': grid_result.cv_results_
            }

        else:
            # Implement optimization for other model types
            raise NotImplementedError("Optimization not implemented for this model type")

    def prune_model(self, model: Any, threshold: float = 0.1) -> Any:
        """Prune model to reduce complexity."""
        if isinstance(model, tf.keras.Model):
            # Create pruned model
            pruned_model = tf.keras.Sequential()

            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    weights = layer.get_weights()
                    # Prune weights below threshold
                    weights[0][np.abs(weights[0]) < threshold] = 0
                    pruned_layer = tf.keras.layers.Dense(
                        layer.output_shape[1],
                        weights=weights,
                        activation=layer.activation
                    )
                    pruned_model.add(pruned_layer)
                else:
                    pruned_model.add(layer)

            return pruned_model

        else:
            raise NotImplementedError("Pruning not implemented for this model type")

    def quantize_model(self, model: Any) -> Any:
        """Quantize model to reduce size and improve inference speed."""
        if isinstance(model, tf.keras.Model):
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_model = converter.convert()

            # Save quantized model
            with open(self.maintenance_dir / 'quantized_model.tflite', 'wb') as f:
                f.write(quantized_model)

            return quantized_model

        else:
            raise NotImplementedError("Quantization not implemented for this model type")

    def evaluate_model_size(self, model: Any) -> Dict[str, float]:
        """Evaluate model size and complexity."""
        if isinstance(model, tf.keras.Model):
            # Get model size
            model_size = sum(
                tf.size(var).numpy() for var in model.trainable_variables
            )

            # Get number of parameters
            num_params = sum(
                np.prod(var.shape) for var in model.trainable_variables
            )

            return {
                'size_bytes': model_size * 4,  # Assuming float32
                'num_parameters': num_params,
                'size_mb': model_size * 4 / 1e6
            }

        else:
            raise NotImplementedError("Size evaluation not implemented for this model type")

    def generate_maintenance_report(self,
                                  original_model: Any,
                                  optimized_model: Any,
                                  maintenance_dir: str = "maintenance/reports") -> None:
        """Generate maintenance report."""
        # Create output directory
        maintenance_dir = Path(maintenance_dir)
        maintenance_dir.mkdir(parents=True, exist_ok=True)

        # Evaluate both models
        original_metrics = self.evaluate_model_size(original_model)
        optimized_metrics = self.evaluate_model_size(optimized_model)

        # Create report content
        report = {
            'timestamp': datetime.now().isoformat(),
            'original_model': original_metrics,
            'optimized_model': optimized_metrics,
            'optimization_metrics': {
                'size_reduction': (
                    original_metrics['size_mb'] - optimized_metrics['size_mb']
                ) / original_metrics['size_mb'] * 100,
                'param_reduction': (
                    original_metrics['num_parameters'] - optimized_metrics['num_parameters']
                ) / original_metrics['num_parameters'] * 100
            }
        }

        # Save report
        with open(maintenance_dir / 'maintenance_report.json', 'w') as f:
            json.dump(report, f, indent=4)

    def maintain_model(self) -> None:
        """Maintain and optimize model."""
        try:
            # Load model
            model = self.load_model()

            # Load data
            preprocessor = BatteryDataPreprocessor()
            X, y_soh, y_soc = preprocessor.prepare_data_for_training()[0:3]

            # Optimize hyperparameters
            optimization_results = self.optimize_hyperparameters(model, X, y_soh)
            logger.info(f"Best hyperparameters found: {optimization_results['best_params']}")

            # Prune model
            pruned_model = self.prune_model(model)
            logger.info("Model pruning completed")

            # Quantize model
            self.quantize_model(model)
            logger.info("Model quantization completed")

            # Generate maintenance report
            self.generate_maintenance_report(model, pruned_model)
            logger.info("Maintenance report generated")

        except Exception as e:
            logger.error(f"Error during model maintenance: {str(e)}")
            raise

def main():
    """Main function to run model maintenance."""
    maintainer = ModelMaintainer()
    maintainer.maintain_model()

if __name__ == "__main__":
    main()
