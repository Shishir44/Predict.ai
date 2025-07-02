import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Dashboard:
    def __init__(self, 
                data_dir: str = "data",
                monitoring_dir: str = "monitoring",
                reports_dir: str = "reports"):
        self.data_dir = Path(data_dir)
        self.monitoring_dir = Path(monitoring_dir)
        self.reports_dir = Path(reports_dir)
        
    def load_model_metrics(self) -> pd.DataFrame:
        """Load model metrics from monitoring data."""
        metrics_path = self.monitoring_dir / 'monitoring_metrics.csv'
        if not metrics_path.exists():
            raise FileNotFoundError("Monitoring metrics not found")
            
        return pd.read_csv(metrics_path)
        
    def load_performance_metrics(self) -> pd.DataFrame:
        """Load performance metrics from tracking data."""
        tracking_path = self.monitoring_dir / 'performance_tracking.csv'
        if not tracking_path.exists():
            raise FileNotFoundError("Performance tracking data not found")
            
        return pd.read_csv(tracking_path)
        
    def load_drift_report(self) -> Dict:
        """Load latest drift report."""
        report_path = self.reports_dir / 'drift_detection' / 'drift_report.json'
        if not report_path.exists():
            return {}
            
        with open(report_path, 'r') as f:
            return json.load(f)
            
    def create_metrics_dashboard(self, metrics_df: pd.DataFrame) -> None:
        """Create metrics dashboard section."""
        st.header("Model Metrics Dashboard")
        
        # Create metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        latest_metrics = metrics_df.iloc[-1]
        
        col1.metric(
            "SOH MAE",
            f"{latest_metrics['soh_mae']:.4f}",
            delta=f"{latest_metrics['soh_mae'] - metrics_df['soh_mae'].mean():+.4f}"
        )
        
        col2.metric(
            "SOC MAE",
            f"{latest_metrics['soc_mae']:.4f}",
            delta=f"{latest_metrics['soc_mae'] - metrics_df['soc_mae'].mean():+.4f}"
        )
        
        col3.metric(
            "SOH MSE",
            f"{latest_metrics['soh_mse']:.4f}",
            delta=f"{latest_metrics['soh_mse'] - metrics_df['soh_mse'].mean():+.4f}"
        )
        
        col4.metric(
            "SOC MSE",
            f"{latest_metrics['soc_mse']:.4f}",
            delta=f"{latest_metrics['soc_mse'] - metrics_df['soc_mse'].mean():+.4f}"
        )
        
        # Create metrics trends plot
        st.subheader("Metrics Trends")
        fig = px.line(
            metrics_df,
            x='timestamp',
            y=['soh_mae', 'soc_mae', 'soh_mse', 'soc_mse'],
            title='Performance Metrics Over Time'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    def create_performance_dashboard(self, perf_df: pd.DataFrame) -> None:
        """Create performance dashboard section."""
        st.header("Model Performance Dashboard")
        
        # Create performance summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Average SOH MAE",
                f"{perf_df['soh_mae'].mean():.4f}",
                delta=f"±{perf_df['soh_mae'].std():.4f}"
            )
            
            st.metric(
                "Average SOC MAE",
                f"{perf_df['soc_mae'].mean():.4f}",
                delta=f"±{perf_df['soc_mae'].std():.4f}"
            )
            
        with col2:
            st.metric(
                "Average SOH MSE",
                f"{perf_df['soh_mse'].mean():.4f}",
                delta=f"±{perf_df['soh_mse'].std():.4f}"
            )
            
            st.metric(
                "Average SOC MSE",
                f"{perf_df['soc_mse'].mean():.4f}",
                delta=f"±{perf_df['soc_mse'].std():.4f}"
            )
            
        # Create performance distribution plots
        st.subheader("Performance Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                perf_df,
                x='soh_mae',
                title='SOH MAE Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.histogram(
                perf_df,
                x='soc_mae',
                title='SOC MAE Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
    def create_drift_dashboard(self, drift_report: Dict) -> None:
        """Create drift detection dashboard section."""
        st.header("Data Drift Dashboard")
        
        if not drift_report:
            st.info("No drift detection data available")
            return
            
        # Create drift summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Total Features",
                drift_report['summary']['total_features']
            )
            
        with col2:
            st.metric(
                "Drifted Features",
                drift_report['summary']['drifted_features']
            )
            
        # Create feature drift details
        st.subheader("Feature Drift Details")
        
        # Load drift plots
        drift_dir = self.reports_dir / 'drift_detection'
        feature_drift_path = drift_dir / 'feature_drift.png'
        performance_drift_path = drift_dir / 'performance_drift.png'
        
        if feature_drift_path.exists():
            st.image(str(feature_drift_path), caption='Feature Drift Detection')
        
        if performance_drift_path.exists():
            st.image(str(performance_drift_path), caption='Performance Drift Detection')
            
    def create_alerts_dashboard(self, metrics_df: pd.DataFrame) -> None:
        """Create alerts dashboard section."""
        st.header("Alerts Dashboard")
        
        # Define thresholds
        thresholds = {
            'soh_mae': 0.1,
            'soc_mae': 0.1,
            'soh_mse': 0.01,
            'soc_mse': 0.01
        }
        
        # Check for alerts
        alerts = []
        latest_metrics = metrics_df.iloc[-1]
        
        for metric, threshold in thresholds.items():
            if latest_metrics[metric] > threshold:
                alerts.append({
                    'metric': metric.upper(),
                    'value': latest_metrics[metric],
                    'threshold': threshold
                })
                
        if alerts:
            st.warning("Active Alerts")
            
            for alert in alerts:
                st.metric(
                    alert['metric'],
                    f"{alert['value']:.4f}",
                    delta=f"> {alert['threshold']:.4f}"
                )
        else:
            st.success("No active alerts")
            
    def run_dashboard(self) -> None:
        """Run the dashboard application."""
        st.title("Battery Prediction Model Monitoring Dashboard")
        
        try:
            # Load data
            metrics_df = self.load_model_metrics()
            perf_df = self.load_performance_metrics()
            drift_report = self.load_drift_report()
            
            # Create dashboard sections
            self.create_metrics_dashboard(metrics_df)
            self.create_performance_dashboard(perf_df)
            self.create_drift_dashboard(drift_report)
            self.create_alerts_dashboard(metrics_df)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Dashboard error: {str(e)}")

def main():
    """Main function to run the dashboard."""
    dashboard = Dashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
