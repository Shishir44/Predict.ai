"""
Model Training Module

This module provides functionality for training battery prediction models.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.model_config import ProjectConfig

class BatteryModelTrainer:
    """
    Class for training battery prediction models.
    
    Args:
        config: Project configuration
    """
    def __init__(self, config: ProjectConfig):
        """Initialize the trainer."""
        self.config = config
        self.model = None
        self.history = None
        self.feature_scaler = StandardScaler()
        
    def prepare_data(self, X: np.ndarray, y_soh: np.ndarray, y_soc: np.ndarray) -> Dict:
        """
        Prepare data for training.
        
        Args:
            X: Input features
            y_soh: True SOH values
            y_soc: True SOC values
            
        Returns:
            Dictionary containing split datasets
        """
        # Scale features
        X = self.feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_soh_train, y_soh_test, y_soc_train, y_soc_test = train_test_split(
            X, y_soh, y_soc,
            test_size=self.config.model_config.validation_split,
            random_state=42
        )
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_soh_train': y_soh_train, 'y_soh_test': y_soh_test,
            'y_soc_train': y_soc_train, 'y_soc_test': y_soc_test
        }
    
    def train_model(self, data_dict: Dict[str, np.ndarray]) -> tf.keras.callbacks.History:
        """
        Train the multi-output model.
        
        Args:
            data_dict: Dictionary containing split datasets
            
        Returns:
            Training history
        """
        # Build model
        self.model = BatteryLSTMModel(
            sequence_length=self.config.model_config.sequence_length,
            num_features=self.config.model_config.num_features
        )
        
        # Compile with multiple outputs
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.model_config.learning_rate
            ),
            loss={
                'soh_output': 'mse',
                'soc_output': 'mse'
            },
            loss_weights={'soh_output': 1.0, 'soc_output': 1.0},
            metrics={'soh_output': 'mae', 'soc_output': 'mae'}
        )
        
        # Training
        self.history = self.model.fit(
            data_dict['X_train'],
            {
                'soh_output': data_dict['y_soh_train'],
                'soc_output': data_dict['y_soc_train']
            },
            validation_data=(
                data_dict['X_test'],
                {
                    'soh_output': data_dict['y_soh_test'],
                    'soc_output': data_dict['y_soc_test']
                }
            ),
            epochs=self.config.model_config.epochs,
            batch_size=self.config.model_config.batch_size,
            callbacks=self._get_callbacks(),
            verbose=1
        )
        
        return self.history
    
    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Get training callbacks.
        
        Returns:
            List of callback objects
        """
        return [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.config.model_config.model_save_path,
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    
    def save_model(self):
        """Save the trained model and scaler."""
        self.model.save(self.config.model_config.model_save_path)
        
        import joblib
        scaler_path = self.config.model_config.scaler_save_path
        joblib.dump(self.feature_scaler, scaler_path)
    
    def plot_training_history(self) -> None:
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        history = self.history.history
        
        # Plot loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot metrics
        plt.subplot(1, 2, 2)
        plt.plot(history['soh_output_mae'], label='SOH MAE')
        plt.plot(history['soc_output_mae'], label='SOC MAE')
        plt.plot(history['val_soh_output_mae'], label='Val SOH MAE')
        plt.plot(history['val_soc_output_mae'], label='Val SOC MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.title('MAE Metrics')
        
        plt.tight_layout()
        plt.show()
