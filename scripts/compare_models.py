import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelComparator:
    def __init__(self, lstm_model_path: str = "models/lstm/lstm_model.h5",
                transformer_model_path: str = "models/transformer/transformer_model.pth"):
        self.lstm_model_path = Path(lstm_model_path)
        self.transformer_model_path = Path(transformer_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_lstm_model(self) -> tf.keras.Model:
        """Load trained LSTM model."""
        if not self.lstm_model_path.exists():
            raise FileNotFoundError(f"LSTM model not found at {self.lstm_model_path}")
            
        return tf.keras.models.load_model(str(self.lstm_model_path))
        
    def load_transformer_model(self, input_dim: int) -> nn.Module:
        """Load trained Transformer model."""
        if not self.transformer_model_path.exists():
            raise FileNotFoundError(f"Transformer model not found at {self.transformer_model_path}")
            
        model = TransformerModel(input_dim=input_dim)
        model.load_state_dict(torch.load(self.transformer_model_path))
        model.to(self.device)
        model.eval()
        return model
        
    def evaluate_models(self, X_test: np.ndarray, y_soh_test: np.ndarray, y_soc_test: np.ndarray) -> Dict:
        """Evaluate both models on test data."""
        # Load models
        lstm_model = self.load_lstm_model()
        transformer_model = self.load_transformer_model(X_test.shape[-1])
        
        # Prepare test data
        X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
        X_test_torch = torch.FloatTensor(X_test).to(self.device)
        
        # Get predictions
        lstm_predictions = lstm_model.predict(X_test_tf)
        transformer_predictions = transformer_model(X_test_torch)
        
        # Calculate metrics
        lstm_metrics = {
            'soh_mse': np.mean((lstm_predictions[0].flatten() - y_soh_test) ** 2),
            'soh_mae': np.mean(np.abs(lstm_predictions[0].flatten() - y_soh_test)),
            'soc_mse': np.mean((lstm_predictions[1].flatten() - y_soc_test) ** 2),
            'soc_mae': np.mean(np.abs(lstm_predictions[1].flatten() - y_soc_test))
        }
        
        transformer_metrics = {
            'soh_mse': np.mean((transformer_predictions[0].cpu().detach().numpy().flatten() - y_soh_test) ** 2),
            'soh_mae': np.mean(np.abs(transformer_predictions[0].cpu().detach().numpy().flatten() - y_soh_test)),
            'soc_mse': np.mean((transformer_predictions[1].cpu().detach().numpy().flatten() - y_soc_test) ** 2),
            'soc_mae': np.mean(np.abs(transformer_predictions[1].cpu().detach().numpy().flatten() - y_soc_test))
        }
        
        return {
            'lstm': lstm_metrics,
            'transformer': transformer_metrics
        }
        
    def plot_comparison(self, metrics: Dict, output_dir: str = "models/comparison") -> None:
        """Plot model comparison results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Metric': ['SOH MSE', 'SOH MAE', 'SOC MSE', 'SOC MAE'] * 2,
            'Value': list(metrics['lstm'].values()) + list(metrics['transformer'].values()),
            'Model': ['LSTM'] * 4 + ['Transformer'] * 4
        })
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Metric', y='Value', hue='Model')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png')
        plt.close()
        
        # Save metrics to CSV
        pd.DataFrame({
            'Metric': ['SOH MSE', 'SOH MAE', 'SOC MSE', 'SOC MAE'],
            'LSTM': list(metrics['lstm'].values()),
            'Transformer': list(metrics['transformer'].values())
        }).to_csv(output_dir / 'model_comparison_metrics.csv', index=False)
        
    def plot_predictions(self, X_test: np.ndarray, y_soh_test: np.ndarray, y_soc_test: np.ndarray,
                        num_samples: int = 100, output_dir: str = "models/comparison") -> None:
        """Plot predictions from both models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        lstm_model = self.load_lstm_model()
        transformer_model = self.load_transformer_model(X_test.shape[-1])
        
        # Select random samples
        sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        X_sample = X_test[sample_indices]
        y_soh_sample = y_soh_test[sample_indices]
        y_soc_sample = y_soc_test[sample_indices]
        
        # Get predictions
        lstm_predictions = lstm_model.predict(X_sample)
        transformer_predictions = transformer_model(torch.FloatTensor(X_sample).to(self.device))
        
        # Plot SOH predictions
        plt.figure(figsize=(12, 6))
        plt.scatter(y_soh_sample, lstm_predictions[0].flatten(), label='LSTM', alpha=0.5)
        plt.scatter(y_soh_sample, transformer_predictions[0].cpu().detach().numpy().flatten(), 
                   label='Transformer', alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('True SOH')
        plt.ylabel('Predicted SOH')
        plt.title('SOH Prediction Comparison')
        plt.legend()
        plt.savefig(output_dir / 'soh_prediction_comparison.png')
        plt.close()
        
        # Plot SOC predictions
        plt.figure(figsize=(12, 6))
        plt.scatter(y_soc_sample, lstm_predictions[1].flatten(), label='LSTM', alpha=0.5)
        plt.scatter(y_soc_sample, transformer_predictions[1].cpu().detach().numpy().flatten(), 
                   label='Transformer', alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('True SOC')
        plt.ylabel('Predicted SOC')
        plt.title('SOC Prediction Comparison')
        plt.legend()
        plt.savefig(output_dir / 'soc_prediction_comparison.png')
        plt.close()

def main():
    """Main function to compare models."""
    # Load preprocessed data
    preprocessor = BatteryDataPreprocessor()
    _, _, X_test, _, y_soh_test, _, y_soc_test = preprocessor.prepare_data_for_training()
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Evaluate models
    metrics = comparator.evaluate_models(X_test, y_soh_test, y_soc_test)
    
    # Plot comparison
    comparator.plot_comparison(metrics)
    
    # Plot predictions
    comparator.plot_predictions(X_test, y_soh_test, y_soc_test)
    
    logger.info("Model comparison completed successfully")

if __name__ == "__main__":
    main()
