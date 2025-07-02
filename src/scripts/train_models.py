import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatteryModelTrainer:
    def __init__(self,
                data_dir: str = "data/processed",
                model_dir: str = "models",
                config: Dict[str, Any] = None):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {
            'lstm': {
                'sequence_length': 10,
                'hidden_units': 64,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'validation_split': 0.2
            },
            'transformer': {
                'sequence_length': 10,
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 2,
                'dropout_rate': 0.1,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'validation_split': 0.2
            }
        }

    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess battery data."""
        # Load data
        data_path = self.data_dir / 'nasa_comprehensive_dataset.csv'
        df = pd.read_csv(data_path)

        # Select features (adjust to match our NASA dataset)
        self.features = ['cycle_number', 'capacity_ah', 'ambient_temperature']
        X = df[self.features].values

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Create sequences
        seq_length = self.config['lstm']['sequence_length']
        X_seq = []
        y_soh = []
        y_soc = []

        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            y_soh.append(df['soh_relative'].iloc[i + seq_length])
            # For SOC, we'll use capacity as a proxy (normalized)
            y_soc.append(df['capacity_ah'].iloc[i + seq_length] / df['capacity_ah'].max())

        X_seq = np.array(X_seq)
        y_soh = np.array(y_soh)
        y_soc = np.array(y_soc)

        # Split data
        X_train, X_val, y_soh_train, y_soh_val, y_soc_train, y_soc_val = train_test_split(
            X_seq, y_soh, y_soc,
            test_size=self.config['lstm']['validation_split'],
            random_state=42
        )

        # Save scaler
        import joblib
        scaler_path = self.model_dir / 'scaler.joblib'
        joblib.dump(scaler, scaler_path)

        return X_train, X_val, y_soh_train, y_soh_val, y_soc_train, y_soc_val

    def build_lstm_model(self) -> tf.keras.Model:
        """Build LSTM model."""
        from tensorflow import keras

        model = keras.models.Sequential([
            keras.layers.LSTM(
                self.config['lstm']['hidden_units'],
                return_sequences=True,
                input_shape=(self.config['lstm']['sequence_length'], len(self.features))
            ),
            keras.layers.Dropout(self.config['lstm']['dropout_rate']),
            keras.layers.LSTM(self.config['lstm']['hidden_units']),
            keras.layers.Dropout(self.config['lstm']['dropout_rate']),
            keras.layers.Dense(2)  # Predict both SOH and SOC
        ])

        optimizer = keras.optimizers.Adam(learning_rate=self.config['lstm']['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        return model

    class TransformerModel(nn.Module):
        def __init__(self,
                    d_model: int,
                    nhead: int,
                    num_layers: int,
                    dropout: float,
                    input_dim: int):
            super().__init__()

            self.embedding = nn.Linear(input_dim, d_model)
            self.positional_encoding = nn.Parameter(torch.zeros(1, self.config['transformer']['sequence_length'], d_model))

            encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

            self.output_layer = nn.Linear(d_model, 2)  # Predict both SOH and SOC

        def forward(self, x):
            x = self.embedding(x)
            x = x + self.positional_encoding
            x = self.transformer_encoder(x)
            x = x.mean(dim=1)  # Take mean across sequence
            return self.output_layer(x)

    def build_transformer_model(self) -> nn.Module:
        """Build Transformer model."""
        return self.TransformerModel(
            d_model=self.config['transformer']['d_model'],
            nhead=self.config['transformer']['num_heads'],
            num_layers=self.config['transformer']['num_layers'],
            dropout=self.config['transformer']['dropout_rate'],
            input_dim=len(self.features)
        )

    def train_lstm_model(self) -> tf.keras.Model:
        """Train LSTM model."""
        # Load and preprocess data
        X_train, X_val, y_soh_train, y_soh_val, y_soc_train, y_soc_val = self.load_and_preprocess_data()

        # Build model
        model = self.build_lstm_model()

        # Train model
        history = model.fit(
            X_train,
            np.stack([y_soh_train, y_soc_train], axis=-1),
            validation_data=(
                X_val,
                np.stack([y_soh_val, y_soc_val], axis=-1)
            ),
            batch_size=self.config['lstm']['batch_size'],
            epochs=self.config['lstm']['epochs'],
            verbose=1
        )

        # Save model
        model_path = self.model_dir / 'lstm_model.h5'
        model.save(model_path)

        # Save training history
        history_path = self.model_dir / 'lstm_training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'history': history.history,
                'config': self.config['lstm']
            }, f, indent=4)

        return model

    def train_transformer_model(self) -> nn.Module:
        """Train Transformer model."""
        # Load and preprocess data
        X_train, X_val, y_soh_train, y_soh_val, y_soc_train, y_soc_val = self.load_and_preprocess_data()

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.FloatTensor(np.stack([y_soh_train, y_soc_train], axis=-1))
        y_val = torch.FloatTensor(np.stack([y_soh_val, y_soc_val], axis=-1))

        # Build model
        model = self.build_transformer_model()

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['transformer']['learning_rate']
        )

        # Training loop
        batch_size = self.config['transformer']['batch_size']
        history = {'loss': [], 'val_loss': []}

        for epoch in range(self.config['transformer']['epochs']):
            model.train()

            # Shuffle training data
            permutation = torch.randperm(X_train.size()[0])
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            # Train in batches
            for i in range(0, X_train.size()[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)

            history['loss'].append(loss.item())
            history['val_loss'].append(val_loss.item())

            logger.info(f"Epoch {epoch + 1}/{self.config['transformer']['epochs']}")
            logger.info(f"Training loss: {loss.item():.4f}")
            logger.info(f"Validation loss: {val_loss.item():.4f}")

        # Save model
        model_path = self.model_dir / 'transformer_model.pth'
        torch.save(model.state_dict(), model_path)

        # Save training history
        history_path = self.model_dir / 'transformer_training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'history': history,
                'config': self.config['transformer']
            }, f, indent=4)

        return model

    def evaluate_models(self, X_val: np.ndarray, y_soh_val: np.ndarray, y_soc_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate both models on validation data."""
        # Load models
        lstm_model = tf.keras.models.load_model(self.model_dir / 'lstm_model.h5')
        transformer_model = self.TransformerModel(
            d_model=self.config['transformer']['d_model'],
            nhead=self.config['transformer']['num_heads'],
            num_layers=self.config['transformer']['num_layers'],
            dropout=self.config['transformer']['dropout_rate'],
            input_dim=len(features)
        )
        transformer_model.load_state_dict(torch.load(self.model_dir / 'transformer_model.pth'))
        transformer_model.eval()

        # Convert validation data for transformer
        X_val_torch = torch.FloatTensor(X_val)
        y_val_torch = torch.FloatTensor(np.stack([y_soh_val, y_soc_val], axis=-1))

        # Evaluate LSTM model
        lstm_predictions = lstm_model.predict(X_val)
        lstm_metrics = {
            'soh_mse': mean_squared_error(y_soh_val, lstm_predictions[:, 0]),
            'soh_mae': mean_absolute_error(y_soh_val, lstm_predictions[:, 0]),
            'soc_mse': mean_squared_error(y_soc_val, lstm_predictions[:, 1]),
            'soc_mae': mean_absolute_error(y_soc_val, lstm_predictions[:, 1])
        }

        # Evaluate Transformer model
        with torch.no_grad():
            transformer_predictions = transformer_model(X_val_torch).numpy()
            transformer_metrics = {
                'soh_mse': mean_squared_error(y_soh_val, transformer_predictions[:, 0]),
                'soh_mae': mean_absolute_error(y_soh_val, transformer_predictions[:, 0]),
                'soc_mse': mean_squared_error(y_soc_val, transformer_predictions[:, 1]),
                'soc_mae': mean_absolute_error(y_soc_val, transformer_predictions[:, 1])
            }

        # Save evaluation results
        eval_path = self.model_dir / 'model_evaluation.json'
        with open(eval_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'lstm_metrics': lstm_metrics,
                'transformer_metrics': transformer_metrics
            }, f, indent=4)

        return {
            'lstm': lstm_metrics,
            'transformer': transformer_metrics
        }

    def train_and_evaluate(self) -> Dict[str, Dict[str, float]]:
        """Train both models and evaluate their performance."""
        try:
            # Train LSTM model
            logger.info("Training LSTM model...")
            lstm_model = self.train_lstm_model()

            # Train Transformer model
            logger.info("Training Transformer model...")
            transformer_model = self.train_transformer_model()

            # Load validation data
            _, X_val, _, y_soh_val, _, y_soc_val = self.load_and_preprocess_data()

            # Evaluate models
            logger.info("Evaluating models...")
            metrics = self.evaluate_models(X_val, y_soh_val, y_soc_val)

            # Log results
            logger.info("Model training and evaluation completed")
            logger.info(f"LSTM SOH MAE: {metrics['lstm']['soh_mae']:.4f}")
            logger.info(f"LSTM SOC MAE: {metrics['lstm']['soc_mae']:.4f}")
            logger.info(f"Transformer SOH MAE: {metrics['transformer']['soh_mae']:.4f}")
            logger.info(f"Transformer SOC MAE: {metrics['transformer']['soc_mae']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

def main():
    """Main function to train and evaluate models."""
    trainer = BatteryModelTrainer()
    metrics = trainer.train_and_evaluate()

    # Print final metrics
    print("\nFinal Model Metrics:")
    print("LSTM Model:")
    print(f"SOH MAE: {metrics['lstm']['soh_mae']:.4f}")
    print(f"SOC MAE: {metrics['lstm']['soc_mae']:.4f}")
    print("\nTransformer Model:")
    print(f"SOH MAE: {metrics['transformer']['soh_mae']:.4f}")
    print(f"SOC MAE: {metrics['transformer']['soc_mae']:.4f}")

if __name__ == "__main__":
    main()
