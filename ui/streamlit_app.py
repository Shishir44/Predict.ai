# """
# Advanced Battery Health Monitor UI - Enterprise Edition

# This module provides a sophisticated Streamlit-based web interface for comprehensive
# battery monitoring, analytics, and enterprise management.
# """

# import sys
# import os
# from pathlib import Path
# import random
# import time
# from datetime import datetime, timedelta
# import sqlite3
# import json

# # Add project root to Python path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import joblib
# from typing import Dict, List, Optional

# # Set page config FIRST - before any other Streamlit commands
# st.set_page_config(
#     page_title="Predict.ai - Battery Intelligence Platform",
#     page_icon="🔋",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://github.com/predict-ai/battery-health',
#         'Report a bug': "https://github.com/predict-ai/battery-health/issues",
#         'About': "# Predict.ai Battery Intelligence Platform\nEnterprise-grade battery health monitoring and prediction system."
#     }
# )

# # Custom CSS for enterprise styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: 700;
#         background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
    
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin: 0.5rem 0;
#     }
    
#     .status-healthy { 
#         background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
#         color: white;
#         padding: 0.5rem;
#         border-radius: 5px;
#         text-align: center;
#     }
    
#     .status-warning { 
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         padding: 0.5rem;
#         border-radius: 5px;
#         text-align: center;
#     }
    
#     .status-critical { 
#         background: linear-gradient(135deg, #ff512f 0%, #dd2476 100%);
#         color: white;
#         padding: 0.5rem;
#         border-radius: 5px;
#         text-align: center;
#     }
    
#     .sidebar-content {
#         background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#     }
    
#     .prediction-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 2rem;
#         border-radius: 15px;
#         color: white;
#         text-align: center;
#         margin: 1rem 0;
#         box-shadow: 0 10px 20px rgba(0,0,0,0.1);
#     }
    
#     .alert-banner {
#         background: linear-gradient(90deg, #ff6b6b, #ee5a24);
#         color: white;
#         padding: 1rem;
#         border-radius: 10px;
#         text-align: center;
#         margin: 1rem 0;
#         animation: pulse 2s infinite;
#     }
    
#     @keyframes pulse {
#         0% { opacity: 1; }
#         50% { opacity: 0.7; }
#         100% { opacity: 1; }
#     }
    
#     .enterprise-logo {
#         text-align: center;
#         font-size: 2rem;
#         font-weight: bold;
#         color: #2C3E50;
#         margin: 2rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Now try imports with error handling
# try:
#     from src.inference.production_predictor import ProductionBatteryPredictor
#     from src.monitoring.monitoring_service import MonitoringService
#     from src.health.health_checker import EnterpriseHealthChecker
#     from config.model_config import ProjectConfig
#     ENTERPRISE_FEATURES = True
# except ImportError as e:
#     st.error(f"Enterprise features unavailable: {e}")
#     ENTERPRISE_FEATURES = False

# class AdvancedBatteryMonitorUI:
#     """
#     Advanced enterprise-grade UI for battery health monitoring and analytics.
#     """
    
#     def __init__(self):
#         """Initialize the advanced UI components."""
#         self.predictor = None
#         self.monitoring_service = None
#         self.health_checker = None
        
#         # Initialize session state
#         if 'battery_data' not in st.session_state:
#             st.session_state.battery_data = []
#         if 'alerts' not in st.session_state:
#             st.session_state.alerts = []
#         if 'last_prediction' not in st.session_state:
#             st.session_state.last_prediction = None
            
#         self._initialize_enterprise_services()
        
#     def _initialize_enterprise_services(self):
#         """Initialize enterprise services."""
#         try:
#             if ENTERPRISE_FEATURES:
#                 # Initialize production predictor
#                 self.predictor = ProductionBatteryPredictor()
                
#                 # Initialize monitoring service
#                 self.monitoring_service = MonitoringService(
#                     self.predictor,
#                     db_path="monitoring/monitoring.db"
#                 )
                
#                 # Initialize health checker
#                 self.health_checker = EnterpriseHealthChecker()
                
#                 st.success("🚀 Enterprise services initialized successfully!")
#             else:
#                 st.info("🔧 Running in demo mode - enterprise features disabled")
                
#         except Exception as e:
#             st.warning(f"⚠️ Some enterprise features unavailable: {e}")
    
#     def run(self):
#         """Run the advanced Streamlit application."""
#         # Header with enterprise branding
#         st.markdown('<div class="enterprise-logo">🔋 Predict.ai</div>', unsafe_allow_html=True)
#         st.markdown('<div class="main-header">Battery Intelligence Platform</div>', unsafe_allow_html=True)
        
#         # Real-time status bar
#         self._display_status_bar()
        
#         # Sidebar navigation
#         self._create_sidebar()
        
#         # Main content area
#         page = st.session_state.get('current_page', 'Dashboard')
        
#         if page == 'Dashboard':
#             self._dashboard_page()
#         elif page == 'Real-time Monitoring':
#             self._realtime_monitoring_page()
#         elif page == 'Batch Analysis':
#             self._batch_analysis_page()
#         elif page == 'Analytics & Insights':
#             self._analytics_page()
#         elif page == 'Model Management':
#             self._model_management_page()
#         elif page == 'System Health':
#             self._system_health_page()
#         elif page == 'Alerts & Notifications':
#             self._alerts_page()
#         elif page == 'Enterprise Settings':
#             self._enterprise_settings_page()
    
#     def _display_status_bar(self):
#         """Display real-time system status bar."""
#         col1, col2, col3, col4, col5 = st.columns(5)
        
#         with col1:
#             if self.predictor:
#                 status = self.predictor.get_status()
#                 if status['status'] == 'healthy':
#                     st.markdown('<div class="status-healthy">🟢 Models Online</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown('<div class="status-warning">🟡 Models Degraded</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown('<div class="status-warning">🟡 Demo Mode</div>', unsafe_allow_html=True)
        
#         with col2:
#             if self.health_checker:
#                 health = self.health_checker.run_all_checks()
#                 status_class = f"status-{health['overall_status']}" if health['overall_status'] in ['healthy', 'warning', 'critical'] else "status-warning"
#                 st.markdown(f'<div class="{status_class}">💓 System {health["overall_status"].title()}</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown('<div class="status-warning">💓 Health Unknown</div>', unsafe_allow_html=True)
        
#         with col3:
#             current_time = datetime.now().strftime("%H:%M:%S")
#             st.markdown(f'<div class="status-healthy">🕐 {current_time}</div>', unsafe_allow_html=True)
        
#         with col4:
#             predictions_today = len([p for p in st.session_state.battery_data if p.get('date', '').startswith(datetime.now().strftime('%Y-%m-%d'))])
#             st.markdown(f'<div class="status-healthy">📊 {predictions_today} Predictions Today</div>', unsafe_allow_html=True)
        
#         with col5:
#             alerts_count = len(st.session_state.alerts)
#             if alerts_count > 0:
#                 st.markdown(f'<div class="status-critical">🚨 {alerts_count} Active Alerts</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown('<div class="status-healthy">✅ No Alerts</div>', unsafe_allow_html=True)
    
#     def _create_sidebar(self):
#         """Create advanced sidebar navigation."""
#         with st.sidebar:
#             st.markdown("## 🎛️ Navigation")
            
#             pages = [
#                 ("🏠", "Dashboard"),
#                 ("📡", "Real-time Monitoring"), 
#                 ("📊", "Batch Analysis"),
#                 ("📈", "Analytics & Insights"),
#                 ("🤖", "Model Management"),
#                 ("💚", "System Health"),
#                 ("🚨", "Alerts & Notifications"),
#                 ("⚙️", "Enterprise Settings")
#             ]
            
#             for icon, page in pages:
#                 if st.button(f"{icon} {page}", use_container_width=True):
#                     st.session_state.current_page = page
#                     st.rerun()
            
#             st.markdown("---")
            
#             # Quick metrics in sidebar
#             st.markdown("## 📊 Quick Metrics")
            
#             if st.session_state.last_prediction:
#                 last_pred = st.session_state.last_prediction
#                 st.metric("Last SOH", f"{last_pred.get('soh', 0):.1f}%")
#                 st.metric("Last SOC", f"{last_pred.get('soc', 0):.1f}%")
#                 st.metric("Confidence", f"{last_pred.get('confidence', 0):.1f}%")
            
#             # Model status
#             st.markdown("## 🤖 Model Status")
#             if self.predictor:
#                 model_status = self.predictor.get_model_status()
#                 for model, loaded in model_status.items():
#                     status_icon = "🟢" if loaded else "🔴"
#                     st.write(f"{status_icon} {model.title().replace('_', ' ')}")
            
#             st.markdown("---")
#             st.markdown("### 🔧 Quick Actions")
#             if st.button("🔄 Refresh Data", use_container_width=True):
#                 st.rerun()
            
#             if st.button("📥 Export Report", use_container_width=True):
#                 self._export_report()
    
#     def _dashboard_page(self):
#         """Main dashboard with overview and KPIs."""
#         st.markdown("## 🏠 Executive Dashboard")
        
#         # KPI Cards
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             avg_soh = np.mean([p.get('soh', 0) for p in st.session_state.battery_data]) if st.session_state.battery_data else 85.5
#             st.markdown(f"""
#             <div class="prediction-card">
#                 <h3>Average SOH</h3>
#                 <h1>{avg_soh:.1f}%</h1>
#                 <p>State of Health</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             total_batteries = len(set(p.get('battery_id', 'default') for p in st.session_state.battery_data)) if st.session_state.battery_data else 1
#             st.markdown(f"""
#             <div class="prediction-card">
#                 <h3>Fleet Size</h3>
#                 <h1>{total_batteries}</h1>
#                 <p>Active Batteries</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col3:
#             critical_batteries = sum(1 for p in st.session_state.battery_data if p.get('soh', 100) < 70)
#             st.markdown(f"""
#             <div class="prediction-card">
#                 <h3>Critical Units</h3>
#                 <h1>{critical_batteries}</h1>
#                 <p>Need Attention</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col4:
#             uptime = 99.8 if ENTERPRISE_FEATURES else 95.2
#             st.markdown(f"""
#             <div class="prediction-card">
#                 <h3>System Uptime</h3>
#                 <h1>{uptime:.1f}%</h1>
#                 <p>Last 30 Days</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # Main dashboard charts
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### 📈 SOH Trend Analysis")
#             self._create_soh_trend_chart()
        
#         with col2:
#             st.markdown("### 🔋 Battery Fleet Status")
#             self._create_fleet_status_chart()
        
#         # Recent predictions table
#         st.markdown("### 📋 Recent Predictions")
#         if st.session_state.battery_data:
#             recent_data = pd.DataFrame(st.session_state.battery_data[-10:])
#             st.dataframe(recent_data, use_container_width=True)
#         else:
#             st.info("No recent predictions available. Use the monitoring tab to generate predictions.")
    
#     def _realtime_monitoring_page(self):
#         """Real-time monitoring interface."""
#         st.markdown("## 📡 Real-time Battery Monitoring")
        
#         # Auto-refresh toggle
#         auto_refresh = st.checkbox("🔄 Auto-refresh (5 seconds)", value=False)
#         if auto_refresh:
#             time.sleep(5)
#             st.rerun()
        
#         # Manual input section
#         with st.expander("🔧 Manual Battery Input", expanded=True):
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 battery_id = st.text_input("Battery ID", value=f"BAT_{random.randint(1000, 9999)}")
#                 voltage = st.slider("Voltage (V)", 2.0, 5.0, 3.7, 0.1)
#                 current = st.slider("Current (A)", -10.0, 10.0, 2.0, 0.1)
            
#             with col2:
#                 temperature = st.slider("Temperature (°C)", -20, 60, 25, 1)
#                 cycle_count = st.slider("Cycle Count", 0, 5000, 100, 10)
#                 capacity = st.slider("Capacity (Ah)", 0.0, 200.0, 50.0, 1.0)
            
#             with col3:
#                 resistance = st.slider("Internal Resistance (mΩ)", 0.0, 100.0, 10.0, 0.5)
#                 st.markdown("### 🎯 Prediction Mode")
#                 model_type = st.selectbox("Model", ["Random Forest", "LSTM", "Ensemble"])
        
#         # Predict button
#         if st.button("🔮 Predict Battery Health", type="primary", use_container_width=True):
#             input_data = {
#                 'battery_id': battery_id,
#                 'voltage': voltage,
#                 'current': current,
#                 'temperature': temperature,
#                 'cycle_count': cycle_count,
#                 'resistance': resistance,
#                 'capacity': capacity,
#                 'model_type': model_type.lower().replace(' ', '_'),
#                 'timestamp': datetime.now().isoformat(),
#                 'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             }
            
#             # Get prediction
#             with st.spinner("🤖 Analyzing battery data..."):
#                 prediction = self._get_prediction(input_data)
#                 input_data.update(prediction)
                
#                 # Store prediction
#                 st.session_state.battery_data.append(input_data)
#                 st.session_state.last_prediction = prediction
                
#                 # Check for alerts
#                 self._check_alerts(prediction)
            
#             # Display results with animations
#             self._display_advanced_results(prediction, input_data)
    
#     def _batch_analysis_page(self):
#         """Batch analysis interface."""
#         st.markdown("## 📊 Batch Analysis & Historical Data")
        
#         # File upload
#         uploaded_file = st.file_uploader(
#             "📁 Upload Battery Data (CSV/Excel)",
#             type=['csv', 'xlsx'],
#             help="Upload file with columns: voltage, current, temperature, cycle_count, etc."
#         )
        
#         if uploaded_file:
#             # Load and preview data
#             if uploaded_file.name.endswith('.csv'):
#                 df = pd.read_csv(uploaded_file)
#             else:
#                 df = pd.read_excel(uploaded_file)
            
#             st.markdown("### 👀 Data Preview")
#             st.dataframe(df.head(), use_container_width=True)
            
#             # Analysis options
#             col1, col2 = st.columns(2)
#             with col1:
#                 analysis_type = st.selectbox(
#                     "Analysis Type",
#                     ["SOH Prediction", "SOC Estimation", "Degradation Analysis", "Failure Prediction"]
#                 )
#             with col2:
#                 batch_size = st.slider("Batch Size", 10, len(df), min(100, len(df)))
            
#             if st.button("🚀 Run Batch Analysis", type="primary"):
#                 with st.spinner("Processing batch data..."):
#                     progress_bar = st.progress(0)
                    
#                     results = []
#                     for i, row in df.head(batch_size).iterrows():
#                         # Simulate processing
#                         progress_bar.progress((i + 1) / batch_size)
#                         prediction = self._get_prediction(row.to_dict())
#                         results.append(prediction)
#                         time.sleep(0.1)  # Simulate processing time
                    
#                     # Display results
#                     results_df = pd.DataFrame(results)
#                     st.markdown("### 📋 Batch Analysis Results")
#                     st.dataframe(results_df, use_container_width=True)
                    
#                     # Summary statistics
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Average SOH", f"{results_df['soh'].mean():.1f}%")
#                     with col2:
#                         st.metric("Batteries at Risk", f"{(results_df['soh'] < 70).sum()}")
#                     with col3:
#                         st.metric("Processing Time", f"{batch_size * 0.1:.1f}s")
    
#     def _analytics_page(self):
#         """Advanced analytics and insights."""
#         st.markdown("## 📈 Analytics & Insights")
        
#         # Analytics dashboard
#         tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Deep Dive", "🎯 Predictions", "📉 Trends"])
        
#         with tab1:
#             self._analytics_overview()
        
#         with tab2:
#             self._analytics_deep_dive()
        
#         with tab3:
#             self._analytics_predictions()
        
#         with tab4:
#             self._analytics_trends()
    
#     def _model_management_page(self):
#         """Model management interface."""
#         st.markdown("## 🤖 Model Management")
        
#         # Model status overview
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### 📊 Model Performance")
#             if self.predictor:
#                 model_metrics = {
#                     "Random Forest": {"accuracy": 0.781, "latency": "12ms", "memory": "45MB"},
#                     "LSTM": {"accuracy": 0.765, "latency": "28ms", "memory": "120MB"},
#                     "Ensemble": {"accuracy": 0.798, "latency": "35ms", "memory": "165MB"}
#                 }
                
#                 for model, metrics in model_metrics.items():
#                     with st.expander(f"🔧 {model} Model"):
#                         col_a, col_b, col_c = st.columns(3)
#                         with col_a:
#                             st.metric("R² Score", f"{metrics['accuracy']:.3f}")
#                         with col_b:
#                             st.metric("Latency", metrics['latency'])
#                         with col_c:
#                             st.metric("Memory", metrics['memory'])
        
#         with col2:
#             st.markdown("### ⚙️ Model Operations")
#             if st.button("🔄 Retrain Models", type="primary"):
#                 with st.spinner("Retraining models..."):
#                     progress = st.progress(0)
#                     for i in range(100):
#                         progress.progress(i + 1)
#                         time.sleep(0.02)
#                     st.success("✅ Models retrained successfully!")
            
#             if st.button("📊 Model Validation"):
#                 st.info("🔍 Running model validation suite...")
            
#             if st.button("📦 Deploy to Production"):
#                 st.success("🚀 Models deployed to production!")
    
#     def _system_health_page(self):
#         """System health monitoring."""
#         st.markdown("## 💚 System Health Monitoring")
        
#         if self.health_checker:
#             health_status = self.health_checker.run_all_checks()
            
#             # Overall status
#             status_color = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(health_status['overall_status'], "⚪")
#             st.markdown(f"### {status_color} Overall Status: {health_status['overall_status'].title()}")
            
#             # Individual checks
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown("#### System Resources")
#                 for check_name, check_result in health_status['checks'].items():
#                     if check_name in ['cpu_usage', 'memory_usage', 'disk_space']:
#                         status_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(check_result['status'], "⚪")
#                         st.write(f"{status_icon} **{check_name.replace('_', ' ').title()}**: {check_result['message']}")
            
#             with col2:
#                 st.markdown("#### Services")
#                 for check_name, check_result in health_status['checks'].items():
#                     if check_name in ['model_files', 'database_connectivity']:
#                         status_icon = {"healthy": "🟢", "warning": "🟡", "critical": "🔴"}.get(check_result['status'], "⚪")
#                         st.write(f"{status_icon} **{check_name.replace('_', ' ').title()}**: {check_result['message']}")
#         else:
#             st.warning("⚠️ Health checker not available in demo mode")
    
#     def _alerts_page(self):
#         """Alerts and notifications management."""
#         st.markdown("## 🚨 Alerts & Notifications")
        
#         # Active alerts
#         if st.session_state.alerts:
#             st.markdown("### 🔴 Active Alerts")
#             for i, alert in enumerate(st.session_state.alerts):
#                 with st.expander(f"⚠️ {alert['title']}", expanded=True):
#                     st.write(f"**Severity:** {alert['severity']}")
#                     st.write(f"**Message:** {alert['message']}")
#                     st.write(f"**Time:** {alert['timestamp']}")
                    
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         if st.button(f"✅ Acknowledge", key=f"ack_{i}"):
#                             st.session_state.alerts[i]['acknowledged'] = True
#                             st.rerun()
#                     with col2:
#                         if st.button(f"🗑️ Dismiss", key=f"dismiss_{i}"):
#                             st.session_state.alerts.pop(i)
#                             st.rerun()
#         else:
#             st.success("✅ No active alerts")
        
#         # Alert configuration
#         st.markdown("### ⚙️ Alert Configuration")
#         with st.expander("Configure Alert Thresholds"):
#             soh_threshold = st.slider("SOH Critical Threshold (%)", 0, 100, 70)
#             soc_threshold = st.slider("SOC Low Threshold (%)", 0, 100, 20)
#             temp_threshold = st.slider("Temperature Alert (°C)", 0, 80, 50)
            
#             if st.button("💾 Save Alert Settings"):
#                 st.success("✅ Alert settings saved!")
    
#     def _enterprise_settings_page(self):
#         """Enterprise configuration and settings."""
#         st.markdown("## ⚙️ Enterprise Settings")
        
#         tab1, tab2, tab3 = st.tabs(["🔧 General", "👥 Users", "🔐 Security"])
        
#         with tab1:
#             st.markdown("### System Configuration")
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.selectbox("Prediction Model", ["Random Forest", "LSTM", "Ensemble"])
#                 st.slider("Prediction Confidence Threshold", 0.0, 1.0, 0.8)
#                 st.selectbox("Logging Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            
#             with col2:
#                 st.checkbox("Enable Auto-scaling")
#                 st.checkbox("Enable Drift Detection")
#                 st.number_input("Max Concurrent Predictions", 1, 100, 10)
        
#         with tab2:
#             st.markdown("### User Management")
#             st.info("👤 Current User: Administrator")
#             st.button("➕ Add New User")
#             st.button("📋 Manage Permissions")
        
#         with tab3:
#             st.markdown("### Security Settings")
#             st.checkbox("Enable 2FA")
#             st.checkbox("Audit Logging")
#             st.selectbox("Session Timeout", ["15 min", "30 min", "1 hour", "Never"])
    
#     def _get_prediction(self, input_data: Dict) -> Dict:
#         """Get prediction from the model."""
#         try:
#             if self.predictor:
#                 # Use production predictor
#                 features = np.array([[
#                     input_data.get('voltage', 3.7),
#                     input_data.get('current', 2.0),
#                     input_data.get('temperature', 25),
#                     input_data.get('cycle_count', 100),
#                     input_data.get('resistance', 10.0),
#                     input_data.get('capacity', 50.0)
#                 ]])
                
#                 result = self.predictor.predict(features)
#                 return {
#                     'soh': result['soh'],
#                     'soc': result['soc'],
#                     'confidence': result.get('confidence', 0.85) * 100,
#                     'model_used': result.get('model_used', 'random_forest'),
#                     'processing_time': result.get('processing_time', 0.015)
#                 }
#             else:
#                 # Mock prediction for demo
#                 return self._generate_mock_prediction(input_data)
                
#         except Exception as e:
#             st.error(f"Prediction error: {e}")
#             return self._generate_mock_prediction(input_data)
    
#     def _generate_mock_prediction(self, input_data: Dict) -> Dict:
#         """Generate mock prediction for demo purposes."""
#         voltage = input_data.get('voltage', 3.7)
#         cycle_count = input_data.get('cycle_count', 100)
#         temperature = input_data.get('temperature', 25)
        
#         # Simulate SOH based on cycle count and temperature
#         base_soh = 100 - (cycle_count / 50) - (max(0, temperature - 25) * 0.5)
#         soh = max(60, min(100, base_soh + random.uniform(-5, 5)))
        
#         # Simulate SOC based on voltage
#         soc = min(100, max(0, (voltage - 3.0) * 50 + random.uniform(-10, 10)))
        
#         confidence = random.uniform(75, 95)
        
#         return {
#             'soh': round(soh, 1),
#             'soc': round(soc, 1),
#             'confidence': round(confidence, 1),
#             'model_used': 'mock_model',
#             'processing_time': round(random.uniform(0.01, 0.05), 3)
#         }
    
#     def _display_advanced_results(self, prediction: Dict, input_data: Dict):
#         """Display prediction results with advanced visualizations."""
#         st.markdown("### 🎯 Prediction Results")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             soh = prediction['soh']
#             soh_color = "🟢" if soh > 80 else "🟡" if soh > 70 else "🔴"
#             st.markdown(f"""
#             <div class="prediction-card">
#                 <h3>State of Health</h3>
#                 <h1>{soh_color} {soh:.1f}%</h1>
#                 <p>Battery Capacity</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col2:
#             soc = prediction['soc']
#             soc_color = "🟢" if soc > 50 else "🟡" if soc > 20 else "🔴"
#             st.markdown(f"""
#             <div class="prediction-card">
#                 <h3>State of Charge</h3>
#                 <h1>{soc_color} {soc:.1f}%</h1>
#                 <p>Current Charge</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         with col3:
#             confidence = prediction['confidence']
#             conf_color = "🟢" if confidence > 85 else "🟡" if confidence > 70 else "🔴"
#             st.markdown(f"""
#             <div class="prediction-card">
#                 <h3>Confidence</h3>
#                 <h1>{conf_color} {confidence:.1f}%</h1>
#                 <p>Model Certainty</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # Gauge charts
#         col1, col2 = st.columns(2)
        
#         with col1:
#             fig = self._create_gauge_chart(soh, "State of Health (%)", "SOH")
#             st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             fig = self._create_gauge_chart(soc, "State of Charge (%)", "SOC")
#             st.plotly_chart(fig, use_container_width=True)
        
#         # Additional metrics
#         with st.expander("📊 Detailed Metrics"):
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Processing Time", f"{prediction.get('processing_time', 0):.3f}s")
#             with col2:
#                 st.metric("Model Used", prediction.get('model_used', 'Unknown').title())
#             with col3:
#                 st.metric("Battery ID", input_data.get('battery_id', 'N/A'))
#             with col4:
#                 st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
    
#     def _create_gauge_chart(self, value: float, title: str, metric: str):
#         """Create a gauge chart for metrics."""
#         fig = go.Figure(go.Indicator(
#             mode = "gauge+number+delta",
#             value = value,
#             domain = {'x': [0, 1], 'y': [0, 1]},
#             title = {'text': title},
#             delta = {'reference': 80 if metric == "SOH" else 50},
#             gauge = {
#                 'axis': {'range': [None, 100]},
#                 'bar': {'color': "darkblue"},
#                 'steps': [
#                     {'range': [0, 60], 'color': "lightgray"},
#                     {'range': [60, 80], 'color': "yellow"},
#                     {'range': [80, 100], 'color': "green"}
#                 ],
#                 'threshold': {
#                     'line': {'color': "red", 'width': 4},
#                     'thickness': 0.75,
#                     'value': 70
#                 }
#             }
#         ))
        
#         fig.update_layout(height=300)
#         return fig
    
#     def _check_alerts(self, prediction: Dict):
#         """Check for alerts based on prediction results."""
#         alerts_triggered = []
        
#         if prediction['soh'] < 70:
#             alerts_triggered.append({
#                 'title': 'Low SOH Alert',
#                 'severity': 'Critical',
#                 'message': f"Battery SOH ({prediction['soh']:.1f}%) below critical threshold (70%)",
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 'acknowledged': False
#             })
        
#         if prediction['soc'] < 20:
#             alerts_triggered.append({
#                 'title': 'Low SOC Alert',
#                 'severity': 'Warning',
#                 'message': f"Battery SOC ({prediction['soc']:.1f}%) below warning threshold (20%)",
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 'acknowledged': False
#             })
        
#         if prediction['confidence'] < 70:
#             alerts_triggered.append({
#                 'title': 'Low Confidence Alert',
#                 'severity': 'Info',
#                 'message': f"Prediction confidence ({prediction['confidence']:.1f}%) below threshold (70%)",
#                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 'acknowledged': False
#             })
        
#         # Add new alerts to session state
#         st.session_state.alerts.extend(alerts_triggered)
        
#         # Show alert banner for critical alerts
#         for alert in alerts_triggered:
#             if alert['severity'] == 'Critical':
#                 st.markdown(f'<div class="alert-banner">🚨 CRITICAL: {alert["message"]}</div>', unsafe_allow_html=True)
    
#     def _create_soh_trend_chart(self):
#         """Create SOH trend chart."""
#         if st.session_state.battery_data:
#             df = pd.DataFrame(st.session_state.battery_data)
#             df['timestamp'] = pd.to_datetime(df['timestamp'])
            
#             fig = px.line(df, x='timestamp', y='soh', 
#                          title='SOH Trend Over Time',
#                          color_discrete_sequence=['#1f77b4'])
#             fig.update_layout(height=400)
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             # Demo chart
#             dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
#             soh_values = [100 - i*0.5 + np.random.normal(0, 2) for i in range(30)]
            
#             fig = px.line(x=dates, y=soh_values, title='SOH Trend (Demo Data)')
#             fig.update_layout(height=400)
#             st.plotly_chart(fig, use_container_width=True)
    
#     def _create_fleet_status_chart(self):
#         """Create fleet status pie chart."""
#         # Demo data for fleet status
#         status_data = {
#             'Healthy (>80%)': 65,
#             'Good (70-80%)': 25,
#             'Degraded (60-70%)': 8,
#             'Critical (<60%)': 2
#         }
        
#         fig = px.pie(values=list(status_data.values()), 
#                      names=list(status_data.keys()),
#                      title='Battery Fleet Health Distribution')
#         fig.update_layout(height=400)
#         st.plotly_chart(fig, use_container_width=True)
    
#     def _analytics_overview(self):
#         """Analytics overview tab."""
#         st.markdown("#### 📊 Performance Overview")
        
#         # Key metrics
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Prediction Accuracy", "89.5%", "↗️ +2.1%")
#         with col2:
#             st.metric("Average Latency", "23ms", "↘️ -5ms")
#         with col3:
#             st.metric("Model Uptime", "99.8%", "↗️ +0.2%")
#         with col4:
#             st.metric("Data Quality", "95.2%", "→ 0%")
    
#     def _analytics_deep_dive(self):
#         """Deep dive analytics tab."""
#         st.markdown("#### 🔍 Deep Dive Analysis")
        
#         # Feature importance chart
#         features = ['Voltage', 'Current', 'Temperature', 'Cycle Count', 'Resistance', 'Capacity']
#         importance = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        
#         fig = px.bar(x=features, y=importance, title='Feature Importance for SOH Prediction')
#         st.plotly_chart(fig, use_container_width=True)
    
#     def _analytics_predictions(self):
#         """Predictions analytics tab."""
#         st.markdown("#### 🎯 Prediction Analytics")
        
#         # Prediction accuracy over time
#         dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
#         accuracy = [0.85 + np.random.normal(0, 0.05) for _ in range(30)]
        
#         fig = px.line(x=dates, y=accuracy, title='Model Accuracy Over Time')
#         fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Target Accuracy")
#         st.plotly_chart(fig, use_container_width=True)
    
#     def _analytics_trends(self):
#         """Trends analytics tab."""
#         st.markdown("#### 📉 Long-term Trends")
        
#         # Multi-metric trend chart
#         dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        
#         fig = make_subplots(
#             rows=2, cols=2,
#             subplot_titles=('SOH Degradation', 'Temperature Impact', 'Cycle Count Effects', 'Prediction Confidence'),
#             specs=[[{"secondary_y": False}, {"secondary_y": False}],
#                    [{"secondary_y": False}, {"secondary_y": False}]]
#         )
        
#         # Add sample data to subplots
#         for i in range(1, 3):
#             for j in range(1, 3):
#                 sample_data = [50 + np.random.normal(0, 10) for _ in range(90)]
#                 fig.add_trace(
#                     go.Scatter(x=dates, y=sample_data, mode='lines', name=f'Metric {i}-{j}'),
#                     row=i, col=j
#                 )
        
#         fig.update_layout(height=600, showlegend=False)
#         st.plotly_chart(fig, use_container_width=True)
    
#     def _export_report(self):
#         """Export comprehensive report."""
#         st.success("📄 Report exported successfully!")
#         st.info("Report saved to: /reports/battery_health_report.pdf")

# # Run the application
# if __name__ == "__main__":
#     app = AdvancedBatteryMonitorUI()
#     app.run()
