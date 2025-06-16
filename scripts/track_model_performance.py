import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    def __init__(self, 
                tracking_dir: str = "tracking",
                metrics_config: Dict[str, Any] = None):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_config = metrics_config or {
            'window_size': 30,  # days
            'metrics': {
                'soh': {
                    'mae': {'threshold': 0.1, 'window': 7},
                    'mse': {'threshold': 0.01, 'window': 7},
                    'drift': {'threshold': 0.05, 'window': 7}
                },
                'soc': {
                    'mae': {'threshold': 0.1, 'window': 7},
                    'mse': {'threshold': 0.01, 'window': 7},
                    'drift': {'threshold': 0.05, 'window': 7}
                }
            }
        }
        
    def load_tracking_data(self) -> pd.DataFrame:
        """Load tracking data from file."""
        tracking_path = self.tracking_dir / 'performance_tracking.csv'
        if tracking_path.exists():
            return pd.read_csv(tracking_path)
        return pd.DataFrame(columns=['timestamp', 'soh_mae', 'soc_mae', 'soh_mse', 'soc_mse'])
        
    def save_tracking_data(self, df: pd.DataFrame) -> None:
        """Save tracking data to file."""
        tracking_path = self.tracking_dir / 'performance_tracking.csv'
        df.to_csv(tracking_path, index=False)
        
    def calculate_metrics(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
        
    def detect_drift(self, 
                    metric_history: List[float], 
                    threshold: float = 0.05) -> bool:
        """Detect concept drift using statistical tests."""
        if len(metric_history) < 2:
            return False
            
        # Calculate rolling mean and standard deviation
        rolling_mean = np.mean(metric_history)
        rolling_std = np.std(metric_history)
        
        # Calculate drift score
        drift_score = abs(metric_history[-1] - rolling_mean) / rolling_std
        
        return drift_score > threshold
        
    def track_performance(self, 
                        y_true: Dict[str, np.ndarray],
                        y_pred: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Track model performance."""
        # Calculate metrics
        metrics = {
            'soh': self.calculate_metrics(y_true['soh'], y_pred['soh']),
            'soc': self.calculate_metrics(y_true['soc'], y_pred['soc'])
        }
        
        # Load tracking data
        tracking_df = self.load_tracking_data()
        
        # Add new metrics
        new_row = {
            'timestamp': datetime.now().isoformat(),
            'soh_mae': metrics['soh']['mae'],
            'soc_mae': metrics['soc']['mae'],
            'soh_mse': metrics['soh']['mse'],
            'soc_mse': metrics['soc']['mse']
        }
        
        tracking_df = pd.concat([tracking_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save tracking data
        self.save_tracking_data(tracking_df)
        
        # Detect drift
        drift_detected = False
        
        for metric_type in ['soh', 'soc']:
            for metric in ['mae', 'mse']:
                metric_history = tracking_df[f"{metric_type}_{metric}"].tail(
                    self.metrics_config['metrics'][metric_type][metric]['window']
                ).tolist()
                
                if self.detect_drift(
                    metric_history,
                    self.metrics_config['metrics'][metric_type][metric]['threshold']
                ):
                    drift_detected = True
                    logger.warning(f"Drift detected in {metric_type.upper()} {metric.upper()}")
                    
        return {
            'metrics': metrics,
            'drift_detected': drift_detected
        }
        
    def generate_performance_report(self, 
                                  output_dir: str = "tracking/reports") -> None:
        """Generate performance report."""
        # Load tracking data
        tracking_df = self.load_tracking_data()
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot SOH metrics
        plt.figure(figsize=(12, 6))
        plt.plot(tracking_df['timestamp'], tracking_df['soh_mae'], label='SOH MAE')
        plt.plot(tracking_df['timestamp'], tracking_df['soh_mse'], label='SOH MSE')
        plt.title('SOH Performance Metrics Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'soh_metrics.png')
        plt.close()
        
        # Plot SOC metrics
        plt.figure(figsize=(12, 6))
        plt.plot(tracking_df['timestamp'], tracking_df['soc_mae'], label='SOC MAE')
        plt.plot(tracking_df['timestamp'], tracking_df['soc_mse'], label='SOC MSE')
        plt.title('SOC Performance Metrics Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'soc_metrics.png')
        plt.close()
        
        # Calculate summary statistics
        summary = {
            'soh': {
                'mae': {
                    'mean': tracking_df['soh_mae'].mean(),
                    'std': tracking_df['soh_mae'].std(),
                    'min': tracking_df['soh_mae'].min(),
                    'max': tracking_df['soh_mae'].max()
                },
                'mse': {
                    'mean': tracking_df['soh_mse'].mean(),
                    'std': tracking_df['soh_mse'].std(),
                    'min': tracking_df['soh_mse'].min(),
                    'max': tracking_df['soh_mse'].max()
                }
            },
            'soc': {
                'mae': {
                    'mean': tracking_df['soc_mae'].mean(),
                    'std': tracking_df['soc_mae'].std(),
                    'min': tracking_df['soc_mae'].min(),
                    'max': tracking_df['soc_mae'].max()
                },
                'mse': {
                    'mean': tracking_df['soc_mse'].mean(),
                    'std': tracking_df['soc_mse'].std(),
                    'min': tracking_df['soc_mse'].min(),
                    'max': tracking_df['soc_mse'].max()
                }
            }
        }
        
        # Save summary
        with open(output_dir / 'performance_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
    def run_continuous_tracking(self, interval: int = 3600) -> None:
        """Run continuous performance tracking."""
        while True:
            try:
                # Get predictions from deployed model
                validator = ModelValidator()
                predictions = validator.validate_model()
                
                # Track performance
                results = self.track_performance(
                    predictions['y_true'],
                    predictions['y_pred']
                )
                
                # Generate report if drift detected
                if results['drift_detected']:
                    self.generate_performance_report()
                    
                # Wait for next interval
                import time
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Performance tracking stopped")
                break
                
            except Exception as e:
                logger.error(f"Tracking error: {str(e)}")
                continue

def main():
    """Main function to run performance tracking."""
    tracker = PerformanceTracker()
    tracker.run_continuous_tracking()

if __name__ == "__main__":
    main()
