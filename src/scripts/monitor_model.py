import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import logging
from pathlib import Path
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import sys

from .preprocess_data import BatteryDataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self,
                model_url: str = "http://localhost:8501/v1/models/battery_prediction:predict",
                monitoring_window: int = 7,  # days
                alert_thresholds: Optional[Dict[str, float]] = None):
        self.model_url = model_url
        self.monitoring_window = monitoring_window
        self.alert_thresholds = alert_thresholds or {
            'soh_error': 0.1,  # 10% error threshold
            'soc_error': 0.1,  # 10% error threshold
            'drift_threshold': 0.05,  # 5% drift threshold
            'latency_threshold': 0.5  # 500ms latency threshold
        }
        self.metrics = []

    def get_model_metrics(self, X: np.ndarray, y_true: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Get model metrics for a batch of predictions."""
        # Prepare request
        data = json.dumps({
            "signature_name": "serving_default",
            "instances": X.tolist()
        })

        # Measure latency
        start_time = datetime.now()
        response = requests.post(
            self.model_url,
            data=data,
            headers={"content-type": "application/json"}
        )
        latency = (datetime.now() - start_time).total_seconds()

        if response.status_code != 200:
            raise Exception(f"Prediction failed: {response.text}")

        # Parse predictions
        predictions = response.json()['predictions'][0]

        # Calculate metrics
        metrics = {
            'latency': latency,
            'soh_mse': np.mean((predictions['soh_output'] - y_true['soh']) ** 2),
            'soh_mae': np.mean(np.abs(predictions['soh_output'] - y_true['soh'])),
            'soc_mse': np.mean((predictions['soc_output'] - y_true['soc']) ** 2),
            'soc_mae': np.mean(np.abs(predictions['soc_output'] - y_true['soc'])),
            'timestamp': datetime.now().isoformat()
        }

        return metrics

    def detect_drift(self, metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Detect concept drift in model performance."""
        if len(metrics) < 2:
            return {'drift_detected': False, 'drift_metrics': {}}

        # Calculate metrics changes
        last_metrics = metrics[-1]
        prev_metrics = metrics[-2]

        drift_metrics = {
            'soh_drift': abs(last_metrics['soh_mae'] - prev_metrics['soh_mae']) / prev_metrics['soh_mae'],
            'soc_drift': abs(last_metrics['soc_mae'] - prev_metrics['soc_mae']) / prev_metrics['soc_mae']
        }

        # Check thresholds
        drift_detected = any(
            drift > self.alert_thresholds['drift_threshold']
            for drift in drift_metrics.values()
        )

        return {
            'drift_detected': drift_detected,
            'drift_metrics': drift_metrics
        }

    def detect_anomalies(self, metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Detect anomalies in model performance."""
        if len(metrics) < 10:  # Need enough data points for anomaly detection
            return {'anomalies_detected': False, 'metric_stats': {}}

        # Calculate mean and std for each metric
        metric_stats = {}
        for key in ['soh_mae', 'soc_mae', 'latency']:
            values = [m[key] for m in metrics]
            mean = np.mean(values)
            std = np.std(values)

            metric_stats[key] = {
                'mean': mean,
                'std': std,
                'current': metrics[-1][key]
            }

        # Detect anomalies (3 sigma rule)
        anomalies_detected = any(
            abs(stats['current'] - stats['mean']) > 3 * stats['std']
            for stats in metric_stats.values()
        )

        return {
            'anomalies_detected': anomalies_detected,
            'metric_stats': metric_stats
        }

    def generate_alerts(self, metrics: List[Dict[str, float]]) -> List[Dict[str, str]]:
        """Generate alerts based on monitoring metrics."""
        alerts = []

        # Check drift
        drift = self.detect_drift(metrics)
        if drift['drift_detected']:
            alerts.append({
                'type': 'drift',
                'message': f"Model drift detected: {drift['drift_metrics']}",
                'timestamp': datetime.now().isoformat()
            })

        # Check anomalies
        anomalies = self.detect_anomalies(metrics)
        if anomalies['anomalies_detected']:
            alerts.append({
                'type': 'anomaly',
                'message': f"Model anomalies detected: {anomalies['metric_stats']}",
                'timestamp': datetime.now().isoformat()
            })

        # Check thresholds
        last_metrics = metrics[-1]
        for metric, threshold in self.alert_thresholds.items():
            if metric in last_metrics and last_metrics[metric] > threshold:
                alerts.append({
                    'type': 'threshold',
                    'message': f"{metric} exceeded threshold: {last_metrics[metric]} > {threshold}",
                    'timestamp': datetime.now().isoformat()
                })

        return alerts

    def save_metrics(self, metrics: List[Dict[str, float]], output_dir: str = "monitoring") -> None:
        """Save monitoring metrics to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics as CSV
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(output_path / 'monitoring_metrics.csv', index=False)

        # Save latest metrics as JSON
        with open(output_path / 'latest_metrics.json', 'w') as f:
            json.dump(metrics[-1], f, indent=4)

    def plot_metrics(self, metrics: List[Dict[str, float]], output_dir: str = "monitoring") -> None:
        """Plot monitoring metrics."""
        import matplotlib.pyplot as plt

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(metrics)
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])

        # Plot SOH metrics
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['timestamp'], metrics_df['soh_mae'], label='SOH MAE')
        plt.axhline(y=self.alert_thresholds['soh_error'], color='r', linestyle='--', label='Threshold')
        plt.title('SOH Prediction Error Over Time')
        plt.xlabel('Time')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(output_path / 'soh_metrics.png')
        plt.close()

        # Plot SOC metrics
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['timestamp'], metrics_df['soc_mae'], label='SOC MAE')
        plt.axhline(y=self.alert_thresholds['soc_error'], color='r', linestyle='--', label='Threshold')
        plt.title('SOC Prediction Error Over Time')
        plt.xlabel('Time')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig(output_path / 'soc_metrics.png')
        plt.close()

        # Plot latency
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['timestamp'], metrics_df['latency'], label='Latency')
        plt.axhline(y=self.alert_thresholds['latency_threshold'], color='r', linestyle='--', label='Threshold')
        plt.title('Model Latency Over Time')
        plt.xlabel('Time')
        plt.ylabel('Latency (s)')
        plt.legend()
        plt.savefig(output_path / 'latency_metrics.png')
        plt.close()

    def monitor(self, X: np.ndarray, y_true: Dict[str, np.ndarray],
               monitoring_window: Optional[int] = None) -> Dict[str, Any]:
        """Monitor model performance."""
        if monitoring_window is not None:
            self.monitoring_window = monitoring_window

        # Get current metrics
        current_metrics = self.get_model_metrics(X, y_true)
        self.metrics.append(current_metrics)

        # Remove old metrics
        if len(self.metrics) > self.monitoring_window:
            self.metrics = self.metrics[-self.monitoring_window:]

        # Generate alerts
        alerts = self.generate_alerts(self.metrics)

        # Save and plot metrics
        self.save_metrics(self.metrics)
        self.plot_metrics(self.metrics)

        return {
            'metrics': current_metrics,
            'alerts': alerts,
            'drift': self.detect_drift(self.metrics),
            'anomalies': self.detect_anomalies(self.metrics)
        }

def main():
    """Main function to monitor model performance."""
    # Initialize monitor
    monitor = ModelMonitor()

    # Load preprocessed data
    preprocessor = BatteryDataPreprocessor()
    _, _, X_test, _, y_soh_test, _, y_soc_test = preprocessor.prepare_data_for_training()

    # Select random samples for monitoring
    sample_indices = np.random.choice(len(X_test), 100, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = {
        'soh': y_soh_test[sample_indices],
        'soc': y_soc_test[sample_indices]
    }

    # Start monitoring
    while True:
        try:
            results = monitor.monitor(X_sample, y_sample)

            if results['alerts']:
                logger.warning(f"Alerts detected: {results['alerts']}")

            if results['drift']['drift_detected']:
                logger.warning(f"Model drift detected: {results['drift']}")

            if results['anomalies']['anomalies_detected']:
                logger.warning(f"Model anomalies detected: {results['anomalies']}")

            # Wait for next monitoring interval
            import time
            time.sleep(3600)  # Monitor every hour

        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
            break
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")
            continue

if __name__ == "__main__":
    main()
