import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetrainingMonitor:
    def __init__(self, 
                monitoring_dir: str = "monitoring/retraining",
                retraining_config: Dict[str, Any] = None):
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        self.retraining_config = retraining_config or {
            'retraining_interval': 7,  # days
            'performance_threshold': 0.1,  # 10% performance degradation
            'drift_threshold': 0.05,  # 5% drift threshold
            'min_samples': 1000,  # minimum samples required for retraining
            'metrics': {
                'soh_mae': 0.1,  # SOH MAE threshold
                'soc_mae': 0.1,  # SOC MAE threshold
                'soh_mse': 0.01,  # SOH MSE threshold
                'soc_mse': 0.01  # SOC MSE threshold
            }
        }
        
    def load_retraining_history(self) -> List[Dict]:
        """Load retraining history from file."""
        history_path = self.monitoring_dir / 'retraining_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        return []
        
    def save_retraining_history(self, history: List[Dict]) -> None:
        """Save retraining history to file."""
        history_path = self.monitoring_dir / 'retraining_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
            
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current model performance metrics."""
        # Load latest monitoring metrics
        metrics_path = self.monitoring_dir / '../monitoring_metrics.csv'
        if not metrics_path.exists():
            raise FileNotFoundError("Monitoring metrics not found")
            
        metrics_df = pd.read_csv(metrics_path)
        latest_metrics = metrics_df.iloc[-1]
        
        return {
            'soh_mae': latest_metrics['soh_mae'],
            'soc_mae': latest_metrics['soc_mae'],
            'soh_mse': latest_metrics['soh_mse'],
            'soc_mse': latest_metrics['soc_mse']
        }
        
    def check_retraining_conditions(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if retraining conditions are met."""
        conditions = {}
        
        # Check performance degradation
        if metrics['soh_mae'] > self.retraining_config['metrics']['soh_mae']:
            conditions['soh_performance'] = True
        
        if metrics['soc_mae'] > self.retraining_config['metrics']['soc_mae']:
            conditions['soc_performance'] = True
            
        # Check drift
        if metrics['soh_mse'] > self.retraining_config['metrics']['soh_mse']:
            conditions['soh_drift'] = True
        
        if metrics['soc_mse'] > self.retraining_config['metrics']['soc_mse']:
            conditions['soc_drift'] = True
            
        return conditions
        
    def collect_new_data(self) -> pd.DataFrame:
        """Collect new data for retraining."""
        # Load existing data
        data_path = Path("data/processed/battery_data.csv")
        if not data_path.exists():
            raise FileNotFoundError("Processed data not found")
            
        df = pd.read_csv(data_path)
        
        # Filter for recent data
        recent_df = df[df['timestamp'] >= (datetime.now() - timedelta(days=self.retraining_config['retraining_interval'])).isoformat()]
        
        return recent_df
        
    def preprocess_new_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess new data for retraining."""
        # Load existing scaler
        from sklearn.preprocessing import StandardScaler
        scaler_path = Path("models/scaler.joblib")
        if not scaler_path.exists():
            raise FileNotFoundError("Scaler not found")
            
        scaler = joblib.load(scaler_path)
        
        # Select features
        features = ['capacity', 'voltage', 'current', 'temperature']
        
        # Scale features
        X = scaler.transform(df[features])
        
        # Create sequences
        sequence_length = 10  # configurable
        X_seq = []
        y_soh = []
        y_soc = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_soh.append(df['SOH'].iloc[i + sequence_length])
            y_soc.append(df['SOC'].iloc[i + sequence_length])
            
        return np.array(X_seq), np.array(y_soh), np.array(y_soc)
        
    def evaluate_retrained_model(self, 
                               model: Any, 
                               X_val: np.ndarray, 
                               y_soh_val: np.ndarray, 
                               y_soc_val: np.ndarray) -> Dict[str, float]:
        """Evaluate retrained model performance."""
        # Make predictions
        if isinstance(model, tf.keras.Model):
            predictions = model.predict(X_val)
        else:
            predictions = model.predict(X_val)
            
        # Calculate metrics
        metrics = {
            'soh_mae': mean_absolute_error(y_soh_val, predictions[0]),
            'soc_mae': mean_absolute_error(y_soc_val, predictions[1]),
            'soh_mse': mean_squared_error(y_soh_val, predictions[0]),
            'soc_mse': mean_squared_error(y_soc_val, predictions[1])
        }
        
        return metrics
        
    def plot_retraining_metrics(self, history: List[Dict], output_dir: str = "monitoring/reports") -> None:
        """Plot retraining metrics."""
        import matplotlib.pyplot as plt
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract metrics
        dates = [entry['timestamp'] for entry in history]
        soh_mae = [entry['metrics']['soh_mae'] for entry in history]
        soc_mae = [entry['metrics']['soc_mae'] for entry in history]
        
        # Plot SOH MAE
        plt.figure(figsize=(12, 6))
        plt.plot(dates, soh_mae, marker='o')
        plt.axhline(y=self.retraining_config['metrics']['soh_mae'], color='r', linestyle='--')
        plt.title('SOH MAE Over Time')
        plt.xlabel('Retraining Date')
        plt.ylabel('SOH MAE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'soh_mae_history.png')
        plt.close()
        
        # Plot SOC MAE
        plt.figure(figsize=(12, 6))
        plt.plot(dates, soc_mae, marker='o')
        plt.axhline(y=self.retraining_config['metrics']['soc_mae'], color='r', linestyle='--')
        plt.title('SOC MAE Over Time')
        plt.xlabel('Retraining Date')
        plt.ylabel('SOC MAE')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'soc_mae_history.png')
        plt.close()
        
    def monitor_retraining(self) -> None:
        """Monitor and manage model retraining."""
        # Load retraining history
        history = self.load_retraining_history()
        
        try:
            # Get current metrics
            current_metrics = self.get_current_metrics()
            
            # Check retraining conditions
            conditions = self.check_retraining_conditions(current_metrics)
            
            if any(conditions.values()):
                logger.info("Retraining conditions met")
                
                # Collect new data
                df = self.collect_new_data()
                
                if len(df) < self.retraining_config['min_samples']:
                    logger.warning("Insufficient data for retraining")
                    return
                    
                # Preprocess data
                X, y_soh, y_soc = self.preprocess_new_data(df)
                
                # Train new model
                model = ModelRetrainer().retrain()
                
                # Evaluate new model
                metrics = self.evaluate_retrained_model(model, X, y_soh, y_soc)
                
                # Compare with current model
                if metrics['soh_mae'] < current_metrics['soh_mae'] and \
                   metrics['soc_mae'] < current_metrics['soc_mae']:
                    
                    # Update model
                    ModelDeployer().deploy_model(model)
                    
                    # Update history
                    history.append({
                        'timestamp': datetime.now().isoformat(),
                        'metrics': metrics,
                        'conditions': conditions
                    })
                    
                    self.save_retraining_history(history)
                    
                    # Generate plots
                    self.plot_retraining_metrics(history)
                    
                    logger.info("Model retraining successful")
                else:
                    logger.info("New model performance not better than current model")
                    
        except Exception as e:
            logger.error(f"Retraining error: {str(e)}")
            
    def run_continuous_monitoring(self, interval: int = 3600) -> None:
        """Run continuous retraining monitoring."""
        while True:
            try:
                self.monitor_retraining()
                import time
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Retraining monitoring stopped")
                break
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                continue

def main():
    """Main function to run retraining monitoring."""
    monitor = RetrainingMonitor()
    monitor.run_continuous_monitoring()

if __name__ == "__main__":
    main()
