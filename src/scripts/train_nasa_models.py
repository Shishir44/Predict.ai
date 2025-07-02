#!/usr/bin/env python3
"""
Train machine learning models on NASA comprehensive battery dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_models():
    """Train Random Forest model on NASA comprehensive dataset."""

    # Create directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)

    logger.info("🔋 Training models on NASA Comprehensive Dataset")
    logger.info("=" * 60)

    # Load data
    data_path = processed_dir / "nasa_comprehensive_dataset.csv"
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"📊 Loaded dataset: {len(df)} records from {df['battery_id'].nunique()} batteries")

    # Prepare features and targets
    feature_cols = ['cycle_number', 'capacity_ah', 'ambient_temperature']
    X = df[feature_cols].copy()
    y_soh = df['soh_relative'].copy()

    # Add engineered features
    X['capacity_normalized'] = X['capacity_ah'] / X.groupby(X.index // 100)['capacity_ah'].transform('max')
    X['cycle_progress'] = X['cycle_number'] / X.groupby(X.index // 100)['cycle_number'].transform('max')
    X['temp_deviation'] = X['ambient_temperature'] - 24  # Deviation from room temp

    logger.info(f"📈 Features: {list(X.columns)}")
    logger.info(f"🎯 Target range: {y_soh.min():.3f} - {y_soh.max():.3f}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_soh, test_size=0.2, random_state=42, stratify=df['battery_id']
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    logger.info("🌲 Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = rf_model.predict(X_train_scaled)
    y_pred_test = rf_model.predict(X_test_scaled)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Log results
    logger.info("📊 Model Performance:")
    logger.info(f"   Training R²: {train_r2:.4f}")
    logger.info(f"   Test R²: {test_r2:.4f}")
    logger.info(f"   Training MAE: {train_mae:.4f}")
    logger.info(f"   Test MAE: {test_mae:.4f}")
    logger.info(f"   Training RMSE: {np.sqrt(train_mse):.4f}")
    logger.info(f"   Test RMSE: {np.sqrt(test_mse):.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("🔍 Feature Importance:")
    for _, row in feature_importance.iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.4f}")

    # Save model and scaler
    model_path = models_dir / "random_forest_soh_model.joblib"
    scaler_path = models_dir / "feature_scaler.joblib"

    joblib.dump(rf_model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save training metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'RandomForestRegressor',
        'dataset': 'nasa_comprehensive_dataset.csv',
        'features': list(X.columns),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_rmse': float(np.sqrt(train_mse)),
            'test_rmse': float(np.sqrt(test_mse))
        },
        'feature_importance': feature_importance.to_dict('records'),
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"💾 Model saved: {model_path}")
    logger.info(f"📏 Scaler saved: {scaler_path}")
    logger.info(f"📝 Metadata saved: {metadata_path}")
    logger.info("✅ Training completed successfully!")

    return rf_model, scaler, metadata

if __name__ == "__main__":
    train_models()
