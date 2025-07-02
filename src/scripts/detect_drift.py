import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ks_2samp, ttest_ind

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, 
                drift_dir: str = "drift_detection",
                config: Dict[str, Any] = None):
        self.drift_dir = Path(drift_dir)
        self.drift_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {
            'window_size': 30,  # days
            'thresholds': {
                'ks_test': 0.05,  # KS test p-value threshold
                't_test': 0.05,  # T-test p-value threshold
                'mse_threshold': 0.1,  # MSE threshold
                'mae_threshold': 0.1,  # MAE threshold
                'drift_window': 7,  # days for drift detection
                'metrics': {
                    'soh': {
                        'mse': {'threshold': 0.01, 'window': 7},
                        'mae': {'threshold': 0.1, 'window': 7}
                    },
                    'soc': {
                        'mse': {'threshold': 0.01, 'window': 7},
                        'mae': {'threshold': 0.1, 'window': 7}
                    }
                }
            }
        }
        
    def load_data(self, data_path: str = "data/processed/battery_data.csv") -> pd.DataFrame:
        """Load battery data."""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        return pd.read_csv(data_path)
        
    def split_data_by_time(self, df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into reference and current periods."""
        ref_df = df[df['timestamp'] < split_date]
        curr_df = df[df['timestamp'] >= split_date]
        
        return ref_df, curr_df
        
    def perform_ks_test(self, ref_data: np.ndarray, curr_data: np.ndarray) -> Dict[str, float]:
        """Perform Kolmogorov-Smirnov test for distribution comparison."""
        stat, p_value = ks_2samp(ref_data, curr_data)
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'drift_detected': p_value < self.config['thresholds']['ks_test']
        }
        
    def perform_t_test(self, ref_data: np.ndarray, curr_data: np.ndarray) -> Dict[str, float]:
        """Perform T-test for mean comparison."""
        stat, p_value = ttest_ind(ref_data, curr_data)
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'drift_detected': p_value < self.config['thresholds']['t_test']
        }
        
    def detect_feature_drift(self, 
                           ref_df: pd.DataFrame, 
                           curr_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Detect drift in input features."""
        features = ['capacity', 'voltage', 'current', 'temperature']
        drift_results = {}
        
        for feature in features:
            ref_data = ref_df[feature].values
            curr_data = curr_df[feature].values
            
            # Perform tests
            ks_results = self.perform_ks_test(ref_data, curr_data)
            t_results = self.perform_t_test(ref_data, curr_data)
            
            drift_results[feature] = {
                'ks_test': ks_results,
                't_test': t_results,
                'drift_detected': ks_results['drift_detected'] or t_results['drift_detected']
            }
            
        return drift_results
        
    def detect_performance_drift(self, 
                               y_true: Dict[str, np.ndarray],
                               y_pred: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Detect drift in model performance."""
        drift_results = {}
        
        for metric_type in ['soh', 'soc']:
            # Calculate metrics
            mse = mean_squared_error(y_true[metric_type], y_pred[metric_type])
            mae = mean_absolute_error(y_true[metric_type], y_pred[metric_type])
            
            # Check thresholds
            mse_threshold = self.config['thresholds']['metrics'][metric_type]['mse']['threshold']
            mae_threshold = self.config['thresholds']['metrics'][metric_type]['mae']['threshold']
            
            drift_results[metric_type] = {
                'mse': {
                    'value': mse,
                    'threshold': mse_threshold,
                    'drift_detected': mse > mse_threshold
                },
                'mae': {
                    'value': mae,
                    'threshold': mae_threshold,
                    'drift_detected': mae > mae_threshold
                }
            }
            
        return drift_results
        
    def plot_drift_detection(self, 
                           drift_results: Dict[str, Dict[str, Any]],
                           output_dir: str = "drift_detection/reports") -> None:
        """Plot drift detection results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot feature drift
        plt.figure(figsize=(15, 8))
        for i, (feature, results) in enumerate(drift_results.items()):
            plt.subplot(2, 2, i + 1)
            
            # Plot KS test statistic
            plt.bar(['KS Test'], [results['ks_test']['statistic']], color='blue')
            plt.axhline(y=self.config['thresholds']['ks_test'], color='r', linestyle='--')
            plt.title(f'{feature.upper()} KS Test')
            plt.ylabel('Statistic')
            
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_drift.png')
        plt.close()
        
        # Plot performance drift
        plt.figure(figsize=(15, 6))
        for i, (metric_type, results) in enumerate(drift_results.items()):
            plt.subplot(1, 2, i + 1)
            
            # Plot MSE and MAE
            plt.bar(['MSE', 'MAE'],
                   [results['mse']['value'], results['mae']['value']],
                   color=['blue', 'green'])
            
            # Add thresholds
            plt.axhline(y=results['mse']['threshold'], color='r', linestyle='--', label='Threshold')
            plt.axhline(y=results['mae']['threshold'], color='r', linestyle='--')
            
            plt.title(f'{metric_type.upper()} Performance')
            plt.ylabel('Error')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_drift.png')
        plt.close()
        
    def generate_drift_report(self, 
                            feature_drift: Dict[str, Dict[str, Any]],
                            performance_drift: Dict[str, Dict[str, Any]],
                            output_dir: str = "drift_detection/reports") -> None:
        """Generate drift detection report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'feature_drift': feature_drift,
            'performance_drift': performance_drift,
            'summary': {
                'total_features': len(feature_drift),
                'drifted_features': sum(1 for results in feature_drift.values() 
                                     if results['drift_detected']),
                'performance_degradation': any(
                    results['drift_detected']
                    for metric_type in performance_drift.values()
                    for results in metric_type.values()
                )
            }
        }
        
        with open(output_dir / 'drift_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
    def adapt_model(self, drift_results: Dict[str, Dict[str, Any]]) -> None:
        """Adapt model based on drift detection results."""
        # Check if adaptation is needed
        if not any(results['drift_detected']
                  for feature_results in drift_results.values()
                  for results in feature_results.values()):
            logger.info("No significant drift detected, no adaptation needed")
            return
            
        # Load model
        model = ModelLoader().load_model()
        
        # Get drifted features
        drifted_features = [
            feature for feature, results in drift_results.items()
            if results['drift_detected']
        ]
        
        # Adapt model
        if drifted_features:
            logger.info(f"Adapting model for drifted features: {drifted_features}")
            
            # Implement adaptation strategy
            # This could involve:
            # 1. Retraining on drifted features
            # 2. Updating feature weights
            # 3. Adding new features
            # 4. Updating normalization parameters
            
            # For now, we'll just retrain the model
            model_trainer = ModelTrainer()
            model_trainer.retrain()
            
            logger.info("Model adaptation completed")
            
    def detect_and_adapt(self) -> None:
        """Detect drift and adapt model if needed."""
        try:
            # Load data
            df = self.load_data()
            
            # Split data into reference and current periods
            split_date = (datetime.now() - timedelta(days=self.config['window_size'])).isoformat()
            ref_df, curr_df = self.split_data_by_time(df, split_date)
            
            # Detect feature drift
            feature_drift = self.detect_feature_drift(ref_df, curr_df)
            
            # Get model predictions
            validator = ModelValidator()
            predictions = validator.validate_model()
            
            # Detect performance drift
            performance_drift = self.detect_performance_drift(
                predictions['y_true'],
                predictions['y_pred']
            )
            
            # Generate plots and report
            self.plot_drift_detection(feature_drift)
            self.generate_drift_report(feature_drift, performance_drift)
            
            # Adapt model if needed
            self.adapt_model(feature_drift)
            
        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")
            raise

def main():
    """Main function to run drift detection and adaptation."""
    detector = DriftDetector()
    detector.detect_and_adapt()

if __name__ == "__main__":
    main()
