#!/usr/bin/env python3
"""
Train Battery Health Prediction Models
=====================================

Train SOH/SOC prediction models using the synthetic battery dataset.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatteryModelTrainer:
    """Train battery health prediction models"""

    def __init__(self, data_file: str = "data/raw/synthetic/battery_dataset.csv"):
        self.data_file = Path(data_file)
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        # Load and prepare data
        self.df = pd.read_csv(self.data_file)
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for training"""
        logger.info("Preparing data for training...")

        # Feature columns (exclude target and identifiers)
        feature_cols = ['cycle_number', 'capacity_ah', 'voltage', 'current',
                       'temperature', 'internal_resistance', 'energy_wh',
                       'efficiency', 'days_elapsed']

        self.features = self.df[feature_cols].copy()
        self.target = self.df['soh'].copy()

        # Split by battery ID to ensure proper evaluation
        battery_ids = self.df['battery_id'].unique()
        train_batteries = battery_ids[:4]  # First 4 batteries for training
        test_batteries = battery_ids[4:]   # Last 2 batteries for testing

        train_mask = self.df['battery_id'].isin(train_batteries)
        test_mask = self.df['battery_id'].isin(test_batteries)

        self.X_train = self.features[train_mask]
        self.X_test = self.features[test_mask]
        self.y_train = self.target[train_mask]
        self.y_test = self.target[test_mask]

        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        logger.info(f"Training set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")

    def train_random_forest(self):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")

        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(self.X_train, self.y_train)

        # Evaluate
        train_pred = rf_model.predict(self.X_train)
        test_pred = rf_model.predict(self.X_test)

        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        test_mae = mean_absolute_error(self.y_test, test_pred)
        test_r2 = r2_score(self.y_test, test_pred)

        logger.info(f"Random Forest Results:")
        logger.info(f"  Train RMSE: {train_rmse:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.4f}")
        logger.info(f"  Test MAE: {test_mae:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}")

        # Save model
        rf_model_file = self.models_dir / 'random_forest_soh_model.joblib'
        joblib.dump(rf_model, rf_model_file)

        # Save scaler
        scaler_file = self.models_dir / 'feature_scaler.joblib'
        joblib.dump(self.scaler, scaler_file)

        logger.info(f"Random Forest model saved to: {rf_model_file}")

        return rf_model, {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}

    def create_lstm_sequences(self, X, y, sequence_length=10):
        """Create sequences for LSTM training"""
        sequences_X, sequences_y = [], []

        # Get the corresponding battery IDs for the data split
        if len(X) == len(self.X_train):
            # Training data
            battery_ids = self.df[self.df['battery_id'].isin(self.df['battery_id'].unique()[:4])]['battery_id'].unique()
            df_subset = self.df[self.df['battery_id'].isin(battery_ids)]
        else:
            # Test data
            battery_ids = self.df[self.df['battery_id'].isin(self.df['battery_id'].unique()[4:])]['battery_id'].unique()
            df_subset = self.df[self.df['battery_id'].isin(battery_ids)]

        start_idx = 0
        for battery_id in battery_ids:
            battery_mask = df_subset['battery_id'] == battery_id
            battery_length = battery_mask.sum()

            battery_data = X[start_idx:start_idx + battery_length]
            battery_target = y[start_idx:start_idx + battery_length]

            for i in range(len(battery_data) - sequence_length + 1):
                sequences_X.append(battery_data[i:i+sequence_length])
                sequences_y.append(battery_target[i+sequence_length-1])

            start_idx += battery_length

        return np.array(sequences_X), np.array(sequences_y)

    def train_lstm(self):
        """Train LSTM model"""
        logger.info("Training LSTM model...")

        sequence_length = 10

        # Create sequences
        X_train_seq, y_train_seq = self.create_lstm_sequences(
            self.X_train_scaled, self.y_train.values, sequence_length
        )
        X_test_seq, y_test_seq = self.create_lstm_sequences(
            self.X_test_scaled, self.y_test.values, sequence_length
        )

        logger.info(f"LSTM sequences - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

        # Build LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train_seq.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            verbose=1
        )

        # Evaluate
        test_pred = model.predict(X_test_seq)
        test_rmse = np.sqrt(mean_squared_error(y_test_seq, test_pred))
        test_mae = mean_absolute_error(y_test_seq, test_pred)
        test_r2 = r2_score(y_test_seq, test_pred)

        logger.info(f"LSTM Results:")
        logger.info(f"  Test RMSE: {test_rmse:.4f}")
        logger.info(f"  Test MAE: {test_mae:.4f}")
        logger.info(f"  Test R²: {test_r2:.4f}")

        # Save model
        lstm_model_file = self.models_dir / 'lstm_soh_model.h5'
        model.save(lstm_model_file)

        logger.info(f"LSTM model saved to: {lstm_model_file}")

        return model, {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}

    def create_model_info(self, rf_results, lstm_results):
        """Create model information file"""

        info_file = self.models_dir / 'MODEL_INFO.md'
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("# Battery Health Prediction Models\n\n")
            f.write("## Trained Models\n\n")

            f.write("### Random Forest Model\n")
            f.write(f"- **File:** random_forest_soh_model.joblib\n")
            f.write(f"- **RMSE:** {rf_results['rmse']:.4f}\n")
            f.write(f"- **MAE:** {rf_results['mae']:.4f}\n")
            f.write(f"- **R²:** {rf_results['r2']:.4f}\n\n")

            f.write("### LSTM Model\n")
            f.write(f"- **File:** lstm_soh_model.h5\n")
            f.write(f"- **RMSE:** {lstm_results['rmse']:.4f}\n")
            f.write(f"- **MAE:** {lstm_results['mae']:.4f}\n")
            f.write(f"- **R²:** {lstm_results['r2']:.4f}\n\n")

            f.write("## Additional Files\n\n")
            f.write("- **feature_scaler.joblib** - StandardScaler for preprocessing\n\n")

            f.write("## Usage\n\n")
            f.write("1. Load the model files using joblib (RF) or tf.keras (LSTM)\n")
            f.write("2. Scale input features using the saved scaler\n")
            f.write("3. Make predictions using model.predict()\n")
            f.write("4. Models predict SOH (State of Health) values\n\n")

            f.write("## Input Features\n\n")
            f.write("- cycle_number: Battery cycle count\n")
            f.write("- capacity_ah: Discharge capacity (Ah)\n")
            f.write("- voltage: Average voltage (V)\n")
            f.write("- current: Average current (A)\n")
            f.write("- temperature: Temperature (°C)\n")
            f.write("- internal_resistance: Internal resistance (Ohm)\n")
            f.write("- energy_wh: Energy (Wh)\n")
            f.write("- efficiency: Efficiency (%)\n")
            f.write("- days_elapsed: Days since start\n")


def main():
    """Train battery health prediction models"""
    print("Battery Health Prediction Model Training")
    print("=" * 45)

    # Initialize trainer
    trainer = BatteryModelTrainer()

    # Train Random Forest
    print("\n1. Training Random Forest...")
    rf_model, rf_results = trainer.train_random_forest()

    # Train LSTM
    print("\n2. Training LSTM...")
    lstm_model, lstm_results = trainer.train_lstm()

    # Create model info
    trainer.create_model_info(rf_results, lstm_results)

    print(f"\n✅ Model Training Completed!")
    print(f"📁 Models saved to: {trainer.models_dir}")
    print(f"🎯 Random Forest R²: {rf_results['r2']:.4f}")
    print(f"🎯 LSTM R²: {lstm_results['r2']:.4f}")
    print(f"🚀 Models ready for deployment!")


if __name__ == "__main__":
    main()
