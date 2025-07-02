import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentMonitor:
    def __init__(self, 
                monitoring_dir: str = "monitoring",
                alert_thresholds: Dict[str, float] = None,
                email_config: Dict[str, str] = None):
        self.monitoring_dir = Path(monitoring_dir)
        self.alert_thresholds = alert_thresholds or {
            'latency': 0.5,  # 500ms latency threshold
            'error_rate': 0.01,  # 1% error rate threshold
            'drift_threshold': 0.05,  # 5% drift threshold
            'anomaly_threshold': 3.0  # 3 sigma anomaly threshold
        }
        self.email_config = email_config or {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': 'your-email@gmail.com',
            'receiver_email': 'receiver-email@gmail.com',
            'password': 'your-app-password'  # Use app-specific password
        }
        self.metrics = []
        
    def get_deployment_metrics(self) -> Dict[str, float]:
        """Get deployment metrics from monitoring system."""
        # Load monitoring metrics
        metrics_path = self.monitoring_dir / 'monitoring_metrics.csv'
        if not metrics_path.exists():
            raise FileNotFoundError("Monitoring metrics not found")
            
        metrics_df = pd.read_csv(metrics_path)
        
        # Calculate metrics
        latest_metrics = metrics_df.iloc[-1]
        
        return {
            'latency': latest_metrics['latency'],
            'soh_mae': latest_metrics['soh_mae'],
            'soc_mae': latest_metrics['soc_mae'],
            'error_rate': self.calculate_error_rate(metrics_df),
            'drift': self.calculate_drift(metrics_df)
        }
        
    def calculate_error_rate(self, metrics_df: pd.DataFrame) -> float:
        """Calculate error rate from monitoring metrics."""
        # Count errors in last hour
        recent_df = metrics_df[
            metrics_df['timestamp'] >= (datetime.now() - timedelta(hours=1)).isoformat()
        ]
        
        return len(recent_df[recent_df['error'] == 1]) / len(recent_df)
        
    def calculate_drift(self, metrics_df: pd.DataFrame) -> float:
        """Calculate concept drift."""
        if len(metrics_df) < 2:
            return 0.0
            
        # Calculate drift for SOH and SOC metrics
        soh_drift = abs(metrics_df['soh_mae'].iloc[-1] - metrics_df['soh_mae'].mean()) / metrics_df['soh_mae'].std()
        soc_drift = abs(metrics_df['soc_mae'].iloc[-1] - metrics_df['soc_mae'].mean()) / metrics_df['soc_mae'].std()
        
        return max(soh_drift, soc_drift)
        
    def detect_anomalies(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Detect anomalies in deployment metrics."""
        anomalies = {}
        
        # Check latency
        if metrics['latency'] > self.alert_thresholds['latency']:
            anomalies['latency'] = True
        
        # Check error rate
        if metrics['error_rate'] > self.alert_thresholds['error_rate']:
            anomalies['error_rate'] = True
        
        # Check drift
        if metrics['drift'] > self.alert_thresholds['drift_threshold']:
            anomalies['drift'] = True
        
        return anomalies
        
    def send_alert_email(self, subject: str, message: str) -> None:
        """Send alert email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['receiver_email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], 
                            self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], 
                           self.email_config['password'])
                server.send_message(msg)
                
            logger.info("Alert email sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {str(e)}")
            
    def create_alert_message(self, anomalies: Dict[str, bool], metrics: Dict[str, float]) -> str:
        """Create alert message."""
        message = """Deployment Alert Report\n\n"""
        
        # Add anomaly information
        message += "Anomalies Detected:\n"
        for anomaly, detected in anomalies.items():
            if detected:
                message += f"- {anomaly.upper()} anomaly detected\n"
                
        # Add metrics
        message += "\nCurrent Metrics:\n"
        for metric, value in metrics.items():
            threshold = self.alert_thresholds.get(metric, 0)
            message += f"- {metric.upper()}: {value:.4f} (Threshold: {threshold:.4f})\n"
            
        return message
        
    def monitor(self, monitoring_interval: int = 300) -> None:
        """Monitor deployment continuously."""
        while True:
            try:
                # Get current metrics
                metrics = self.get_deployment_metrics()
                self.metrics.append(metrics)
                
                # Detect anomalies
                anomalies = self.detect_anomalies(metrics)
                
                # Send alert if anomalies detected
                if any(anomalies.values()):
                    message = self.create_alert_message(anomalies, metrics)
                    self.send_alert_email(
                        "Deployment Alert: Battery Prediction Model",
                        message
                    )
                    
                # Save metrics
                self.save_metrics(metrics)
                
                # Wait for next monitoring interval
                import time
                time.sleep(monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                continue
                
    def save_metrics(self, metrics: Dict[str, float]) -> None:
        """Save monitoring metrics to disk."""
        # Create DataFrame
        metrics_df = pd.DataFrame([metrics])
        metrics_df['timestamp'] = datetime.now().isoformat()
        
        # Save to CSV
        metrics_path = self.monitoring_dir / 'deployment_metrics.csv'
        if not metrics_path.exists():
            metrics_df.to_csv(metrics_path, index=False)
        else:
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
            
    def plot_metrics(self, output_dir: str = "monitoring/reports") -> None:
        """Plot deployment metrics."""
        import matplotlib.pyplot as plt
        
        # Load metrics
        metrics_path = self.monitoring_dir / 'deployment_metrics.csv'
        metrics_df = pd.read_csv(metrics_path)
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot latency
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['timestamp'], metrics_df['latency'])
        plt.axhline(y=self.alert_thresholds['latency'], color='r', linestyle='--')
        plt.title('Model Latency Over Time')
        plt.xlabel('Time')
        plt.ylabel('Latency (s)')
        plt.savefig(output_dir / 'latency_metrics.png')
        plt.close()
        
        # Plot error rate
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['timestamp'], metrics_df['error_rate'])
        plt.axhline(y=self.alert_thresholds['error_rate'], color='r', linestyle='--')
        plt.title('Error Rate Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error Rate')
        plt.savefig(output_dir / 'error_rate_metrics.png')
        plt.close()
        
        # Plot drift
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_df['timestamp'], metrics_df['drift'])
        plt.axhline(y=self.alert_thresholds['drift_threshold'], color='r', linestyle='--')
        plt.title('Concept Drift Over Time')
        plt.xlabel('Time')
        plt.ylabel('Drift Score')
        plt.savefig(output_dir / 'drift_metrics.png')
        plt.close()

def main():
    """Main function to run deployment monitoring."""
    # Initialize monitor
    monitor = DeploymentMonitor()
    
    # Start monitoring
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped")
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")

if __name__ == "__main__":
    main()
