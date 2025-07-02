import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, 
                model_dir: str = "models/ensemble",
                validation_dir: str = "validation"):
        self.model_dir = Path(model_dir)
        self.validation_dir = Path(validation_dir)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, model_type: str = "lstm") -> Any:
        """Load model for validation."""
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
            
    def validate_input_shapes(self, model: Any, X: np.ndarray) -> bool:
        """Validate input shapes."""
        if isinstance(model, tf.keras.Model):
            expected_shape = model.input_shape[1:]
            actual_shape = X.shape[1:]
            
            if expected_shape != actual_shape:
                logger.warning(f"Input shape mismatch: expected {expected_shape}, got {actual_shape}")
                return False
                
        elif isinstance(model, torch.nn.Module):
            expected_shape = model.input_dim
            actual_shape = X.shape[-1]
            
            if expected_shape != actual_shape:
                logger.warning(f"Input shape mismatch: expected {expected_shape}, got {actual_shape}")
                return False
                
        return True
        
    def validate_output_shapes(self, model: Any, predictions: np.ndarray) -> bool:
        """Validate output shapes."""
        if isinstance(model, tf.keras.Model):
            expected_shapes = {
                'soh_output': model.output_shape[0][1:],
                'soc_output': model.output_shape[1][1:]
            }
            
            if predictions[0].shape[1:] != expected_shapes['soh_output']:
                logger.warning(f"SOH output shape mismatch: expected {expected_shapes['soh_output']}, got {predictions[0].shape[1:]}")
                return False
                
            if predictions[1].shape[1:] != expected_shapes['soc_output']:
                logger.warning(f"SOC output shape mismatch: expected {expected_shapes['soc_output']}, got {predictions[1].shape[1:]}")
                return False
                
        elif isinstance(model, torch.nn.Module):
            expected_shape = (1,)
            
            if predictions[0].shape != expected_shape:
                logger.warning(f"SOH output shape mismatch: expected {expected_shape}, got {predictions[0].shape}")
                return False
                
            if predictions[1].shape != expected_shape:
                logger.warning(f"SOC output shape mismatch: expected {expected_shape}, got {predictions[1].shape}")
                return False
                
        return True
        
    def validate_predictions(self, predictions: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Validate predictions against ground truth."""
        metrics = {
            'soh_mse': np.mean((predictions[0].flatten() - y_true['soh']) ** 2),
            'soh_mae': np.mean(np.abs(predictions[0].flatten() - y_true['soh'])),
            'soc_mse': np.mean((predictions[1].flatten() - y_true['soc']) ** 2),
            'soc_mae': np.mean(np.abs(predictions[1].flatten() - y_true['soc']))
        }
        
        # Check thresholds
        thresholds = {
            'soh_mse': 0.1,
            'soh_mae': 0.1,
            'soc_mse': 0.1,
            'soc_mae': 0.1
        }
        
        validation_passed = True
        for metric, value in metrics.items():
            if value > thresholds[metric]:
                logger.warning(f"Validation failed for {metric}: {value} > {thresholds[metric]}")
                validation_passed = False
                
        return {
            'metrics': metrics,
            'validation_passed': validation_passed
        }
        
    def plot_validation_results(self, predictions: np.ndarray, y_true: np.ndarray) -> None:
        """Plot validation results."""
        # Plot SOH predictions
        plt.figure(figsize=(12, 6))
        plt.scatter(y_true['soh'], predictions[0].flatten(), alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('True SOH')
        plt.ylabel('Predicted SOH')
        plt.title('SOH Prediction Validation')
        plt.savefig(self.validation_dir / 'soh_validation.png')
        plt.close()
        
        # Plot SOC predictions
        plt.figure(figsize=(12, 6))
        plt.scatter(y_true['soc'], predictions[1].flatten(), alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('True SOC')
        plt.ylabel('Predicted SOC')
        plt.title('SOC Prediction Validation')
        plt.savefig(self.validation_dir / 'soc_validation.png')
        plt.close()
        
        # Plot error distributions
        soh_errors = predictions[0].flatten() - y_true['soh']
        soc_errors = predictions[1].flatten() - y_true['soc']
        
        plt.figure(figsize=(12, 6))
        sns.histplot(soh_errors, kde=True, label='SOH Error')
        sns.histplot(soc_errors, kde=True, label='SOC Error')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.legend()
        plt.savefig(self.validation_dir / 'error_distribution.png')
        plt.close()
        
    def validate_model(self, model_type: str = "lstm") -> Dict[str, Any]:
        """Validate model performance."""
        # Load model
        model = self.load_model(model_type)
        
        # Load validation data
        preprocessor = BatteryDataPreprocessor()
        _, _, X_val, _, y_soh_val, _, y_soc_val = preprocessor.prepare_data_for_training()
        y_val = {'soh': y_soh_val, 'soc': y_soc_val}
        
        # Validate input shapes
        if not self.validate_input_shapes(model, X_val):
            return {'validation_passed': False, 'error': 'Input shape validation failed'}
            
        # Make predictions
        if isinstance(model, tf.keras.Model):
            predictions = model.predict(X_val)
        else:
            X_torch = torch.FloatTensor(X_val)
            predictions = model(X_torch)
            predictions = (
                predictions[0].cpu().detach().numpy(),
                predictions[1].cpu().detach().numpy()
            )
            
        # Validate output shapes
        if not self.validate_output_shapes(model, predictions):
            return {'validation_passed': False, 'error': 'Output shape validation failed'}
            
        # Validate predictions
        validation_results = self.validate_predictions(predictions, y_val)
        
        # Plot results
        self.plot_validation_results(predictions, y_val)
        
        return {
            'validation_passed': validation_results['validation_passed'],
            'metrics': validation_results['metrics'],
            'plots': {
                'soh_validation': str(self.validation_dir / 'soh_validation.png'),
                'soc_validation': str(self.validation_dir / 'soc_validation.png'),
                'error_distribution': str(self.validation_dir / 'error_distribution.png')
            }
        }
        
    def cross_validate_model(self, model_type: str = "lstm", n_splits: int = 5) -> Dict[str, Any]:
        """Perform cross-validation."""
        # Load data
        preprocessor = BatteryDataPreprocessor()
        X, y_soh, y_soc = preprocessor.prepare_data_for_training()[0:3]
        y = {'soh': y_soh, 'soc': y_soc}
        
        # Split data
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y['soh'][train_idx], y['soh'][val_idx]
            
            # Train model
            if model_type == "lstm":
                trainer = LSTMModelTrainer()
                trainer.train_model(
                    X_train=X_train,
                    y_soh_train=y_train,
                    y_soc_train=y_train,  # Using same labels for simplicity
                    epochs=2  # Reduced epochs for validation
                )
                model = trainer.model
                
            elif model_type == "transformer":
                trainer = TransformerTrainer()
                trainer.train_model(
                    X_train=X_train,
                    y_soh_train=y_train,
                    y_soc_train=y_train,  # Using same labels for simplicity
                    epochs=2  # Reduced epochs for validation
                )
                model = trainer.model
            
            # Validate fold
            fold_results = self.validate_model(model_type)
            metrics.append(fold_results['metrics'])
            
        # Calculate average metrics
        avg_metrics = {}
        for metric in metrics[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in metrics])
            
        return {
            'cross_validation_results': metrics,
            'average_metrics': avg_metrics
        }

def main():
    """Main function to run model validation."""
    validator = ModelValidator()
    
    # Validate current model
    results = validator.validate_model()
    if results['validation_passed']:
        logger.info("Model validation passed")
    else:
        logger.warning("Model validation failed")
    
    # Cross-validate model
    cross_val_results = validator.cross_validate_model()
    logger.info(f"Cross-validation average metrics: {cross_val_results['average_metrics']}")

if __name__ == "__main__":
    main()
