import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnsembleModel:
    def __init__(self, 
                lstm_model_path: str = "models/lstm/lstm_model.h5",
                transformer_model_path: str = "models/transformer/transformer_model.pth"):
        self.lstm_model_path = Path(lstm_model_path)
        self.transformer_model_path = Path(transformer_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self) -> Tuple[tf.keras.Model, nn.Module]:
        """Load both models."""
        # Load LSTM model
        if not self.lstm_model_path.exists():
            raise FileNotFoundError(f"LSTM model not found at {self.lstm_model_path}")
        lstm_model = tf.keras.models.load_model(str(self.lstm_model_path))
        
        # Load Transformer model
        if not self.transformer_model_path.exists():
            raise FileNotFoundError(f"Transformer model not found at {self.transformer_model_path}")
        transformer_model = TransformerModel(input_dim=10)  # Adjust input_dim as needed
        transformer_model.load_state_dict(torch.load(self.transformer_model_path))
        transformer_model.to(self.device)
        transformer_model.eval()
        
        return lstm_model, transformer_model
        
    def predict(self, X: np.ndarray, method: str = "weighted_average") -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using ensemble method."""
        lstm_model, transformer_model = self.load_models()
        
        # Get predictions from both models
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        X_torch = torch.FloatTensor(X).to(self.device)
        
        lstm_predictions = lstm_model.predict(X_tf)
        transformer_predictions = transformer_model(X_torch)
        
        # Convert transformer predictions to numpy
        transformer_predictions = (
            transformer_predictions[0].cpu().detach().numpy(),
            transformer_predictions[1].cpu().detach().numpy()
        )
        
        if method == "weighted_average":
            # Use model performance metrics as weights
            lstm_weights = np.array([0.6, 0.4])  # Adjust weights based on model performance
            transformer_weights = np.array([0.4, 0.6])
            
            # Calculate weighted average
            soh_pred = lstm_weights[0] * lstm_predictions[0] + transformer_weights[0] * transformer_predictions[0]
            soc_pred = lstm_weights[1] * lstm_predictions[1] + transformer_weights[1] * transformer_predictions[1]
            
        elif method == "stacking":
            # Create meta-model input
            meta_input = np.concatenate([
                lstm_predictions[0],
                transformer_predictions[0],
                lstm_predictions[1],
                transformer_predictions[1]
            ], axis=1)
            
            # TODO: Implement meta-model training
            # For now, use simple average
            soh_pred = (lstm_predictions[0] + transformer_predictions[0]) / 2
            soc_pred = (lstm_predictions[1] + transformer_predictions[1]) / 2
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
            
        return soh_pred, soc_pred
        
    def evaluate(self, X_test: np.ndarray, y_soh_test: np.ndarray, y_soc_test: np.ndarray,
                method: str = "weighted_average") -> Dict[str, float]:
        """Evaluate ensemble model performance."""
        soh_pred, soc_pred = self.predict(X_test, method)
        
        metrics = {
            'soh_mse': np.mean((soh_pred.flatten() - y_soh_test) ** 2),
            'soh_mae': np.mean(np.abs(soh_pred.flatten() - y_soh_test)),
            'soc_mse': np.mean((soc_pred.flatten() - y_soc_test) ** 2),
            'soc_mae': np.mean(np.abs(soc_pred.flatten() - y_soc_test))
        }
        
        logger.info(f"Ensemble model metrics ({method}): {metrics}")
        return metrics
        
    def plot_ensemble_results(self, X_test: np.ndarray, y_soh_test: np.ndarray, y_soc_test: np.ndarray,
                            num_samples: int = 100, output_dir: str = "models/ensemble") -> None:
        """Plot ensemble model predictions vs true values."""
        import matplotlib.pyplot as plt
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Select random samples
        sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_sample = X_test[sample_indices]
        y_soh_sample = y_soh_test[sample_indices]
        y_soc_sample = y_soc_test[sample_indices]
        
        # Get predictions
        soh_pred, soc_pred = self.predict(X_sample)
        
        # Plot SOH predictions
        plt.figure(figsize=(12, 6))
        plt.scatter(y_soh_sample, soh_pred.flatten(), alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('True SOH')
        plt.ylabel('Predicted SOH')
        plt.title('Ensemble Model SOH Predictions')
        plt.savefig(output_dir / 'ensemble_soh_predictions.png')
        plt.close()
        
        # Plot SOC predictions
        plt.figure(figsize=(12, 6))
        plt.scatter(y_soc_sample, soc_pred.flatten(), alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('True SOC')
        plt.ylabel('Predicted SOC')
        plt.title('Ensemble Model SOC Predictions')
        plt.savefig(output_dir / 'ensemble_soc_predictions.png')
        plt.close()

def main():
    """Main function to run ensemble model."""
    # Load preprocessed data
    preprocessor = BatteryDataPreprocessor()
    _, _, X_test, _, y_soh_test, _, y_soc_test = preprocessor.prepare_data_for_training()
    
    # Initialize ensemble model
    ensemble = EnsembleModel()
    
    # Evaluate different ensemble methods
    methods = ['weighted_average', 'stacking']
    results = {}
    
    for method in methods:
        logger.info(f"Evaluating ensemble method: {method}")
        metrics = ensemble.evaluate(X_test, y_soh_test, y_soc_test, method)
        results[method] = metrics
        
        # Plot results
        ensemble.plot_ensemble_results(X_test, y_soh_test, y_soc_test, method=method)
    
    # Save results
    import json
    with open('models/ensemble/results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Ensemble model evaluation completed successfully")

if __name__ == "__main__":
    main()
