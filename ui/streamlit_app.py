"""
Battery Health Monitor UI

This module provides a Streamlit-based web interface for monitoring battery SOH and SOC.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.inference.battery_predictor import BatteryPredictor
from config.model_config import ProjectConfig

class BatteryMonitorUI:
    """
    Streamlit-based UI for battery health monitoring.
    """
    def __init__(self):
        """Initialize the UI."""
        self.predictor = BatteryPredictor(ProjectConfig().model_config.model_save_path)
        
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="Battery Health Monitor",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔋 Battery Health & Charge Monitor")
        st.sidebar.title("Controls")
        
        # File upload or manual input
        input_method = st.sidebar.selectbox(
            "Input Method",
            ["Upload File", "Manual Input", "Live Data"]
        )
        
        if input_method == "Upload File":
            self._file_upload_interface()
        elif input_method == "Manual Input":
            self._manual_input_interface()
        else:
            self._live_data_interface()
            
    def _file_upload_interface(self):
        """File upload interface."""
        uploaded_file = st.file_uploader(
            "Upload Battery Data",
            type=['csv', 'xlsx'],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            if st.button("Analyze Battery"):
                results = self._analyze_battery_data(data)
                self._display_results(results)
                
    def _manual_input_interface(self):
        """Manual input interface."""
        col1, col2 = st.columns(2)
        
        with col1:
            voltage = st.number_input(
                "Voltage (V)",
                min_value=0.0,
                max_value=5.0,
                value=3.7,
                step=0.1
            )
            current = st.number_input(
                "Current (A)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1
            )
            
        with col2:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-20.0,
                max_value=80.0,
                value=25.0,
                step=1.0
            )
            cycle_count = st.number_input(
                "Cycle Count",
                min_value=0,
                max_value=5000,
                value=100,
                step=1
            )
            
        if st.button("Predict SOH & SOC"):
            # Create dummy data for prediction
            data = np.array([
                [voltage, current, temperature, cycle_count]
            ])
            results = self.predictor.predict_soh_soc(data)
            self._display_results(results)
            
    def _live_data_interface(self):
        """Live data interface."""
        st.warning("Live data collection not yet implemented")
        
    def _analyze_battery_data(self, data: pd.DataFrame) -> dict:
        """Analyze uploaded battery data."""
        # Placeholder for data analysis
        return self.predictor.predict_soh_soc(data.values)
        
    def _display_results(self, results: dict):
        """Display prediction results."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "State of Health (SOH)",
                f"{results['soh']:.1%}",
                delta=f"{results['confidence_soh']:.1%} confidence"
            )
            
        with col2:
            st.metric(
                "State of Charge (SOC)",
                f"{results['soc']:.1%}",
                delta=f"{results['confidence_soc']:.1%} confidence"
            )
            
        with col3:
            st.metric(
                "SOH Confidence",
                f"{results['confidence_soh']:.1%}"
            )
            
        with col4:
            st.metric(
                "SOC Confidence",
                f"{results['confidence_soc']:.1%}"
            )
            
        # Visual indicators
        self._create_gauge_charts(results)
        
    def _create_gauge_charts(self, results: dict):
        """Create gauge charts for SOH and SOC."""
        fig = go.Figure()
        
        # SOH gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=results['soh'] * 100,
            domain={'x': [0, 0.5], 'y': [0, 1]},
            title={'text': "State of Health (%)"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        # SOC gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=results['soc'] * 100,
            domain={'x': [0.5, 1], 'y': [0, 1]},
            title={'text': "State of Charge (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 25], 'color': "red"},
                    {'range': [25, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ]
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    app = BatteryMonitorUI()
    app.run()
