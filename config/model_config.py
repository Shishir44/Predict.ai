"""
Configuration Module

This module contains configuration classes for the battery prediction system.
"""

from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Optional

class MLModelConfig(BaseModel):
    """
    Configuration for the ML model.
    """
    model_config = ConfigDict(protected_namespaces=())

    # Model architecture
    sequence_length: int = 50
    num_features: int = 10
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    dropout_rate: float = 0.2

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2

    # Data processing
    feature_columns: List[str] = [
        'voltage', 'current', 'temperature', 'cycle_count',
        'charge_time', 'discharge_time', 'capacity_fade_rate'
    ]

    # Model paths
    model_save_path: str = "models/battery_model.h5"
    scaler_save_path: str = "models/feature_scaler.pkl"

class DeploymentConfig(BaseModel):
    """
    Configuration for deployment.
    """
    model_config = ConfigDict(validate_assignment=True)

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # UI settings
    ui_port: int = 8501

    # Monitoring
    log_level: str = "INFO"
    metrics_enabled: bool = True

class ProjectConfig(BaseModel):
    """
    Root configuration class.
    """
    model_config = ConfigDict(validate_assignment=True)

    ml_model_config: MLModelConfig = MLModelConfig()
    deployment_config: DeploymentConfig = DeploymentConfig()

# For backward compatibility
ModelConfig = MLModelConfig
