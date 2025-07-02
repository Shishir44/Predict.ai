import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import logging
from pathlib import Path
import json
import requests
from datetime import datetime, timedelta
import sys
from typing import Any, Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers

from .train_transformer_model import BatteryDataset, TransformerModel
from .deploy_model import ModelDeployer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self,
                model_dir: str = "models/ensemble",
                monitoring_dir: str = "monitoring",
                retrain_thresholds: Optional[Dict[str, float]] = None):
        self.model_dir = Path(model_dir)
        self.monitoring_dir = Path(monitoring_dir)
        self.retrain_thresholds = retrain_thresholds or {
            'performance_degradation': 0.1,  # 10% performance degradation
            'drift_threshold': 0.05,  # 5% drift threshold
            'data_volume': 1000,  # Minimum new data points required
            'retrain_period': 7  # Days between retraining attempts
        }
        self.last_retrain_date = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_retrain_conditions(self, metrics: Dict[str, float]) -> bool:
        """Check if retraining conditions are met."""
        # Check if enough time has passed since last retrain
        if self.last_retrain_date is not None:
            days_since_last_retrain = (datetime.now() - self.last_retrain_date).days
            if days_since_last_retrain < self.retrain_thresholds['retrain_period']:
                return False

        # Check performance degradation
        if metrics['soh_mae'] > self.retrain_thresholds['performance_degradation']:
            return True

        # Check drift
        if metrics['drift'] > self.retrain_thresholds['drift_threshold']:
            return True

        # Check data volume
        new_data_path = self.monitoring_dir / 'new_data.csv'
        if new_data_path.exists():
            new_data = pd.read_csv(new_data_path)
            if len(new_data) >= self.retrain_thresholds['data_volume']:
                return True

        return False

    def load_new_data(self) -> pd.DataFrame:
        """Load new data for retraining."""
        new_data_path = self.monitoring_dir / 'new_data.csv'
        if not new_data_path.exists():
            raise FileNotFoundError("No new data available for retraining")

        return pd.read_csv(new_data_path)

    def update_training_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Combine new data with existing training data."""
        # Load existing training data
        train_data_path = self.model_dir / 'training_data.csv'
        if train_data_path.exists():
            existing_data = pd.read_csv(train_data_path)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_data = new_data

        # Save updated training data
        combined_data.to_csv(train_data_path, index=False)
        return combined_data

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data for training."""
        # Split features and labels
        X = df.drop(['battery_id', 'SOH', 'SOC'], axis=1).values
        y_soh = df['SOH'].values
        y_soc = df['SOC'].values

        # Normalize features
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        return X_normalized, y_soh, y_soc

    def train_new_model(self,
                       X_train: np.ndarray,
                       y_soh_train: np.ndarray,
                       y_soc_train: np.ndarray,
                       model_type: str = "lstm") -> Any:
        """Train new model."""
        if model_type == "lstm":
            # Build LSTM model
            model = tf.keras.Sequential([
                layers.LSTM(128, return_sequences=True, input_shape=X_train.shape[1:]),
                layers.Dropout(0.2),
                layers.LSTM(64),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid', name='soh_output'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid', name='soc_output')
            ])

            # Compile model
            model.compile(
                optimizer='adam',
                loss={
                    'soh_output': 'mse',
                    'soc_output': 'mse'
                },
                metrics={
                    'soh_output': ['mse', 'mae'],
                    'soc_output': ['mse', 'mae']
                },
                loss_weights={
                    'soh_output': 0.5,
                    'soc_output': 0.5
                }
            )

            # Train model
            history = model.fit(
                X_train,
                {'soh_output': y_soh_train, 'soc_output': y_soc_train},
                validation_split=0.1,
                epochs=50,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )

            return model, history

        elif model_type == "transformer":
            # Build Transformer model
            model = TransformerModel(input_dim=X_train.shape[-1])

            # Train model
            train_dataset = BatteryDataset(X_train, y_soh_train, y_soc_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True
            )

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            best_loss = float('inf')
            patience = 10
            epochs_without_improvement = 0

            for epoch in range(50):
                model.train()
                train_loss = 0.0

                for batch in train_loader:
                    x, y_soh, y_soc = [t.to(self.device) for t in batch]

                    optimizer.zero_grad()
                    soh_pred, soc_pred = model(x)

                    loss = 0.5 * criterion(soh_pred, y_soh.unsqueeze(1)) + \
                          0.5 * criterion(soc_pred, y_soc.unsqueeze(1))

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                if train_loss < best_loss:
                    best_loss = train_loss
                    torch.save(model.state_dict(), self.model_dir / 'transformer_model.pth')
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    break

            return model, None

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def evaluate_new_model(self,
                         X_test: np.ndarray,
                         y_soh_test: np.ndarray,
                         y_soc_test: np.ndarray) -> Dict[str, float]:
        """Evaluate new model performance."""
        # Load existing model
        lstm_model = tf.keras.models.load_model(str(self.model_dir / 'lstm_model.h5'))

        # Get predictions
        soh_pred, soc_pred = lstm_model.predict(X_test)

        # Calculate metrics
        metrics = {
            'soh_mse': np.mean((soh_pred.flatten() - y_soh_test) ** 2),
            'soh_mae': np.mean(np.abs(soh_pred.flatten() - y_soh_test)),
            'soc_mse': np.mean((soc_pred.flatten() - y_soc_test) ** 2),
            'soc_mae': np.mean(np.abs(soc_pred.flatten() - y_soc_test))
        }

        return metrics

    def deploy_new_model(self, model: Any, model_type: str = "lstm") -> None:
        """Deploy new model."""
        if model_type == "lstm":
            # Save LSTM model
            model.save(str(self.model_dir / 'lstm_model.h5'))

        elif model_type == "transformer":
            # Save Transformer model
            torch.save(model.state_dict(), self.model_dir / 'transformer_model.pth')

        # Update deployment
        deployer = ModelDeployer()
        deployer.prepare_model_for_serving()
        deployer.start_serving()

        # Update last retrain date
        self.last_retrain_date = datetime.now()

    def retrain(self) -> Dict[str, Any]:
        """Retrain model if conditions are met."""
        # Check monitoring results
        latest_metrics_path = self.monitoring_dir / 'latest_metrics.json'
        if not latest_metrics_path.exists():
            logger.warning("No monitoring metrics found, skipping retrain check.")
            return {"status": "skipped", "reason": "No metrics found"}

        with open(latest_metrics_path, 'r') as f:
            metrics = json.load(f)

        if self.check_retrain_conditions(metrics):
            logger.info("Retraining conditions met, starting retraining process.")

            try:
                # Load new data
                logger.info("Loading new data")
                new_data = self.load_new_data()

                logger.info("Updating training data")
                combined_data = self.update_training_data(new_data)

                logger.info("Preprocessing data")
                X, y_soh, y_soc = self.preprocess_data(combined_data)

                # Train new model
                logger.info("Training new model")
                model, history = self.train_new_model(X, y_soh, y_soc)

                # Evaluate new model
                logger.info("Evaluating new model")
                metrics = self.evaluate_new_model(X, y_soh, y_soc)

                # Deploy new model if performance is better
                if metrics['soh_mae'] < self.retrain_thresholds['performance_degradation']:
                    logger.info("Deploying new model")
                    self.deploy_new_model(model)
                    return {
                        'retrained': True,
                        'metrics': metrics,
                        'history': history.history if history else None
                    }
                else:
                    logger.info("New model performance not better than existing")
                    return {
                        'retrained': False,
                        'metrics': metrics
                    }

            except Exception as e:
                logger.error(f"Retraining failed: {str(e)}")
                raise
        else:
            logger.info("Retraining conditions not met")
            return {'retrained': False}

def main():
    """Main function to run retraining process."""
    retrainer = ModelRetrainer()
    results = retrainer.retrain()

    if results['retrained']:
        logger.info("Retraining completed successfully")
    else:
        logger.info("Retraining not performed")

if __name__ == "__main__":
    main()
