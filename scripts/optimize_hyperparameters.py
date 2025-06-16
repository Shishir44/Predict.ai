import optuna
import numpy as np
import tensorflow as tf
import torch
from pathlib import Path
import logging
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    def __init__(self, model_type: str = "lstm", 
                data_dir: str = "data/processed", 
                study_name: str = "battery_prediction_study"):
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.study_name = study_name
        self.preprocessor = BatteryDataPreprocessor(data_dir)
        
    def objective_lstm(self, trial: optuna.Trial) -> float:
        """Objective function for LSTM hyperparameter optimization."""
        # Define hyperparameters
        params = {
            'units_lstm1': trial.suggest_int('units_lstm1', 64, 256, step=32),
            'units_lstm2': trial.suggest_int('units_lstm2', 32, 128, step=32),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'dense_units': trial.suggest_int('dense_units', 16, 64, step=16)
        }
        
        # Load data
        (X_train, _, _), (y_soh_train, _, _), (y_soc_train, _, _) = \
            self.preprocessor.prepare_data_for_training()
        
        # Build model
        model = tf.keras.Sequential([
            layers.LSTM(params['units_lstm1'], return_sequences=True, 
                      input_shape=X_train.shape[1:]),
            layers.Dropout(params['dropout_rate']),
            layers.LSTM(params['units_lstm2']),
            layers.Dropout(params['dropout_rate']),
            
            # SOH prediction branch
            layers.Dense(params['dense_units'], activation='relu'),
            layers.Dense(1, activation='sigmoid', name='soh_output'),
            
            # SOC prediction branch
            layers.Dense(params['dense_units'], activation='relu'),
            layers.Dense(1, activation='sigmoid', name='soc_output')
        ])
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
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
            epochs=10,
            batch_size=params['batch_size'],
            verbose=0
        )
        
        # Return validation loss
        return history.history['val_loss'][-1]
        
    def objective_transformer(self, trial: optuna.Trial) -> float:
        """Objective function for Transformer hyperparameter optimization."""
        # Define hyperparameters
        params = {
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
            'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        }
        
        # Load data
        (X_train, _, _), (y_soh_train, _, _), (y_soc_train, _, _) = \
            self.preprocessor.prepare_data_for_training()
        
        # Create dataset
        train_dataset = BatteryDataset(X_train, y_soh_train, y_soc_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True
        )
        
        # Initialize model
        model = TransformerModel(
            input_dim=X_train.shape[-1],
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers']
        )
        
        # Loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate']
        )
        
        # Training loop
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
        return train_loss
        
    def optimize(self, n_trials: int = 50) -> Dict[str, Any]:
        """Perform hyperparameter optimization."""
        logger.info(f"Starting hyperparameter optimization for {self.model_type} model")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            storage=f'sqlite:///{self.study_name}.db',
            load_if_exists=True
        )
        
        # Run optimization
        objective = self.objective_lstm if self.model_type == 'lstm' else self.objective_transformer
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation loss: {best_value}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }
        
    def plot_optimization_results(self, study: optuna.Study, output_dir: str = "models/optimization") -> None:
        """Plot optimization results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot optimization history
        optuna.visualization.plot_optimization_history(study).write_image(
            str(output_dir / 'optimization_history.png')
        )
        
        # Plot parameter importance
        optuna.visualization.plot_param_importances(study).write_image(
            str(output_dir / 'parameter_importances.png')
        )
        
        # Plot parallel coordinates
        optuna.visualization.plot_parallel_coordinate(study).write_image(
            str(output_dir / 'parallel_coordinates.png')
        )

def main():
    """Main function to perform hyperparameter optimization."""
    # LSTM optimization
    lstm_optimizer = HyperparameterOptimizer(model_type='lstm')
    lstm_results = lstm_optimizer.optimize()
    lstm_optimizer.plot_optimization_results(lstm_results['study'], 
                                           "models/optimization/lstm")
    
    # Transformer optimization
    transformer_optimizer = HyperparameterOptimizer(model_type='transformer')
    transformer_results = transformer_optimizer.optimize()
    transformer_optimizer.plot_optimization_results(transformer_results['study'], 
                                                 "models/optimization/transformer")
    
    logger.info("Hyperparameter optimization completed successfully")

if __name__ == "__main__":
    main()
