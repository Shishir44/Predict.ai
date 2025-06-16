import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import logging
from pathlib import Path
import joblib
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMModelTrainer:
    def __init__(self, model_dir: str = "models/lstm"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.history = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture."""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.Dropout(0.2),
            
            # SOH prediction branch
            layers.Dense(32, activation='relu', name='soh_dense1'),
            layers.Dense(1, activation='sigmoid', name='soh_output'),
            
            # SOC prediction branch
            layers.Dense(32, activation='relu', name='soc_dense1'),
            layers.Dense(1, activation='sigmoid', name='soc_output')
        ])
        
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
        
        return model
        
    def train_model(self, 
                  X_train: np.ndarray, 
                  y_soh_train: np.ndarray, 
                  y_soc_train: np.ndarray,
                  X_val: np.ndarray = None,
                  y_soh_val: np.ndarray = None,
                  y_soc_val: np.ndarray = None,
                  epochs: int = 50,
                  batch_size: int = 32) -> tf.keras.callbacks.History:
        """Train the LSTM model."""
        logger.info("Starting LSTM model training")
        
        # Build model
        self.model = self.build_model(X_train.shape[1:])
        
        # Prepare callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / 'lstm_model_{epoch:02d}_{val_loss:.4f}.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                min_lr=1e-6
            ),
            callbacks.TensorBoard(
                log_dir=str(self.model_dir / 'logs'),
                histogram_freq=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train,
            {'soh_output': y_soh_train, 'soc_output': y_soc_train},
            validation_data=(
                X_val,
                {'soh_output': y_soh_val, 'soc_output': y_soc_val}
            ) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        self.model.save(str(self.model_dir / 'lstm_model.h5'))
        
        logger.info("Model training completed successfully")
        return history
        
    def evaluate_model(self, 
                     X_test: np.ndarray, 
                     y_soh_test: np.ndarray, 
                     y_soc_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model on test data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        results = self.model.evaluate(
            X_test,
            {'soh_output': y_soh_test, 'soc_output': y_soc_test},
            verbose=1
        )
        
        metrics = {
            'soh_mse': results[1],
            'soh_mae': results[2],
            'soc_mse': results[3],
            'soc_mae': results[4]
        }
        
        logger.info(f"Test metrics: {metrics}")
        return metrics
        
    def save_training_history(self, history: tf.keras.callbacks.History) -> None:
        """Save training history."""
        history_path = self.model_dir / 'training_history.pkl'
        joblib.dump(history.history, history_path)
        
    def plot_training_history(self, history: tf.keras.callbacks.History) -> None:
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.model_dir / 'training_loss.png')
        plt.close()

def main():
    """Main function to train LSTM model."""
    # Load preprocessed data
    preprocessor = BatteryDataPreprocessor()
    (X_train, X_val, X_test), (y_soh_train, y_soh_val, y_soh_test), \
    (y_soc_train, y_soc_val, y_soc_test) = preprocessor.prepare_data_for_training()
    
    # Initialize trainer
    trainer = LSTMModelTrainer()
    
    # Train model
    history = trainer.train_model(
        X_train=X_train,
        y_soh_train=y_soh_train,
        y_soc_train=y_soc_train,
        X_val=X_val,
        y_soh_val=y_soh_val,
        y_soc_val=y_soc_val
    )
    
    # Evaluate model
    metrics = trainer.evaluate_model(
        X_test=X_test,
        y_soh_test=y_soh_test,
        y_soc_test=y_soc_test
    )
    
    # Save and plot training history
    trainer.save_training_history(history)
    trainer.plot_training_history(history)

if __name__ == "__main__":
    main()
