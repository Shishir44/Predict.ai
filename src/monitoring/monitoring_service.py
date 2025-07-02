"""
Enterprise Model Monitoring Service

Comprehensive monitoring service with health checks, metrics collection, 
drift detection, and alerting capabilities.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
import asyncio
from pydantic import BaseModel
import sqlite3
import pandas as pd
from collections import deque
import pickle

from src.inference.production_predictor import ProductionBatteryPredictor, PredictionResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request/Response Models
class PredictionRequest(BaseModel):
    features: Dict[str, float]
    model_type: str = "random_forest"

class PredictionResponse(BaseModel):
    soh_prediction: float
    soc_prediction: float
    soh_confidence: float
    soc_confidence: float
    prediction_time_ms: float
    model_version: str
    timestamp: str

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    performance: Dict[str, Union[int, float, str]]

class MetricsRequest(BaseModel):
    predictions: List[Dict[str, float]]
    ground_truth: List[Dict[str, float]]

@dataclass
class AlertConfig:
    """Alert configuration."""
    error_rate_threshold: float = 0.1
    latency_threshold_ms: float = 1000.0
    drift_threshold: float = 0.3
    email_enabled: bool = False
    slack_webhook: Optional[str] = None

class MonitoringService:
    """Enterprise monitoring service for battery prediction models."""
    
    def __init__(self, 
                 predictor: ProductionBatteryPredictor,
                 db_path: str = "monitoring/monitoring.db",
                 alert_config: Optional[AlertConfig] = None):
        """
        Initialize monitoring service.
        
        Args:
            predictor: Production predictor instance
            db_path: Path to monitoring database
            alert_config: Alert configuration
        """
        self.predictor = predictor
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.alert_config = alert_config or AlertConfig()
        
        # Metrics storage
        self.recent_predictions = deque(maxlen=1000)
        self.performance_history = deque(maxlen=10000)
        self.drift_history = deque(maxlen=1000)
        
        # Baseline for drift detection
        self.baseline_features = None
        self.baseline_updated = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Start background monitoring
        self._start_monitoring_tasks()
        
    def _init_database(self):
        """Initialize monitoring database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    soh_prediction REAL,
                    soc_prediction REAL,
                    soh_confidence REAL,
                    soc_confidence REAL,
                    prediction_time_ms REAL,
                    model_version TEXT,
                    features TEXT
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    details TEXT
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        def monitoring_loop():
            while True:
                try:
                    self._check_system_health()
                    self._detect_drift()
                    self._cleanup_old_data()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Monitoring task error: {str(e)}")
                    time.sleep(60)
                    
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
    def record_prediction(self, 
                         features: Dict[str, float], 
                         result: PredictionResult) -> None:
        """Record a prediction for monitoring."""
        with self._lock:
            # Add to recent predictions
            prediction_data = {
                'timestamp': result.timestamp,
                'features': features,
                'result': result
            }
            self.recent_predictions.append(prediction_data)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions 
                    (timestamp, soh_prediction, soc_prediction, soh_confidence, 
                     soc_confidence, prediction_time_ms, model_version, features)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.timestamp.isoformat(),
                    result.soh_prediction,
                    result.soc_prediction,
                    result.soh_confidence,
                    result.soc_confidence,
                    result.prediction_time_ms,
                    result.model_version,
                    json.dumps(features)
                ))
                conn.commit()
                
    def get_metrics(self, hours: int = 24) -> Dict:
        """Get performance metrics for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get predictions
            predictions_df = pd.read_sql_query('''
                SELECT * FROM predictions 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_time.isoformat(),))
            
            # Get performance metrics
            metrics_df = pd.read_sql_query('''
                SELECT * FROM performance_metrics 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_time.isoformat(),))
            
        # Calculate metrics
        if len(predictions_df) > 0:
            avg_prediction_time = predictions_df['prediction_time_ms'].mean()
            avg_soh_confidence = predictions_df['soh_confidence'].mean()
            avg_soc_confidence = predictions_df['soc_confidence'].mean()
            prediction_count = len(predictions_df)
        else:
            avg_prediction_time = 0
            avg_soh_confidence = 0
            avg_soc_confidence = 0
            prediction_count = 0
            
        # Get system health
        health_status = self.predictor.get_health_status()
        
        return {
            'time_range_hours': hours,
            'prediction_count': prediction_count,
            'performance': {
                'avg_prediction_time_ms': avg_prediction_time,
                'avg_soh_confidence': avg_soh_confidence,
                'avg_soc_confidence': avg_soc_confidence,
                'error_rate': health_status['performance']['error_rate']
            },
            'system_health': health_status,
            'latest_predictions': predictions_df.head(10).to_dict('records') if len(predictions_df) > 0 else []
        }
        
    def detect_drift(self, features: Dict[str, float]) -> Dict:
        """Detect feature drift in incoming data."""
        if self.baseline_features is None:
            # Initialize baseline with current features
            self.baseline_features = features.copy()
            self.baseline_updated = datetime.now()
            return {'drift_detected': False, 'drift_score': 0.0}
            
        # Calculate drift score (simplified)
        drift_scores = []
        for feature_name in features:
            if feature_name in self.baseline_features:
                baseline_val = self.baseline_features[feature_name]
                current_val = features[feature_name]
                if baseline_val != 0:
                    relative_change = abs(current_val - baseline_val) / abs(baseline_val)
                    drift_scores.append(relative_change)
                    
        overall_drift = np.mean(drift_scores) if drift_scores else 0.0
        drift_detected = overall_drift > self.alert_config.drift_threshold
        
        # Store drift information
        drift_info = {
            'timestamp': datetime.now(),
            'drift_score': overall_drift,
            'drift_detected': drift_detected,
            'features': features
        }
        
        with self._lock:
            self.drift_history.append(drift_info)
            
        if drift_detected:
            self._raise_alert('drift', 'warning', 
                           f'Feature drift detected: {overall_drift:.3f}')
            
        return {
            'drift_detected': drift_detected,
            'drift_score': overall_drift,
            'threshold': self.alert_config.drift_threshold
        }
        
    def _check_system_health(self):
        """Check system health and raise alerts if needed."""
        health = self.predictor.get_health_status()
        
        # Check error rate only if we have enough predictions
        total_predictions = health['performance']['total_predictions']
        error_rate = health['performance']['error_rate']
        
        if total_predictions >= 10 and error_rate > self.alert_config.error_rate_threshold:
            self._raise_alert('error_rate', 'critical', 
                           f"Error rate {error_rate:.3f} exceeds threshold")
        elif total_predictions < 10 and error_rate > 0.5:  # Higher threshold for low sample size
            self._raise_alert('error_rate', 'warning', 
                           f"High error rate {error_rate:.3f} with low sample size ({total_predictions} predictions)")
            
        # Check average latency
        avg_latency = health['performance']['avg_prediction_time_ms']
        if avg_latency > self.alert_config.latency_threshold_ms:
            self._raise_alert('latency', 'warning', 
                           f"Average latency {avg_latency:.1f}ms exceeds threshold")
            
    def _detect_drift(self):
        """Background drift detection."""
        if len(self.recent_predictions) < 10:
            return
            
        # Update baseline periodically
        if (self.baseline_updated is None or 
            datetime.now() - self.baseline_updated > timedelta(hours=24)):
            
            # Use recent predictions to update baseline
            recent_features = [p['features'] for p in list(self.recent_predictions)[-100:]]
            if recent_features:
                # Calculate mean features as new baseline
                feature_names = recent_features[0].keys()
                new_baseline = {}
                for name in feature_names:
                    values = [f[name] for f in recent_features if name in f]
                    new_baseline[name] = np.mean(values) if values else 0.0
                    
                self.baseline_features = new_baseline
                self.baseline_updated = datetime.now()
                logger.info("Updated feature baseline for drift detection")
                
    def _raise_alert(self, alert_type: str, severity: str, message: str):
        """Raise an alert."""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'severity': severity,
            'message': message
        }
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, message)
                VALUES (?, ?, ?, ?)
            ''', (alert_data['timestamp'], alert_type, severity, message))
            conn.commit()
            
        logger.warning(f"ALERT [{severity}] {alert_type}: {message}")
        
        # Send notifications (placeholder)
        if self.alert_config.email_enabled:
            self._send_email_alert(alert_data)
        if self.alert_config.slack_webhook:
            self._send_slack_alert(alert_data)
            
    def _send_email_alert(self, alert_data: Dict):
        """Send email alert (placeholder)."""
        logger.info(f"Email alert sent: {alert_data}")
        
    def _send_slack_alert(self, alert_data: Dict):
        """Send Slack alert (placeholder)."""
        logger.info(f"Slack alert sent: {alert_data}")
        
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM predictions WHERE timestamp < ?', 
                         (cutoff_time.isoformat(),))
            cursor.execute('DELETE FROM performance_metrics WHERE timestamp < ?', 
                         (cutoff_time.isoformat(),))
            conn.commit()
            
    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            alerts_df = pd.read_sql_query('''
                SELECT * FROM alerts 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_time.isoformat(),))
            
        return alerts_df.to_dict('records')
        
    def export_monitoring_data(self, hours: int = 24) -> Dict:
        """Export monitoring data for analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            predictions = pd.read_sql_query('''
                SELECT * FROM predictions WHERE timestamp > ?
            ''', conn, params=(cutoff_time.isoformat(),))
            
            metrics = pd.read_sql_query('''
                SELECT * FROM performance_metrics WHERE timestamp > ?
            ''', conn, params=(cutoff_time.isoformat(),))
            
            alerts = pd.read_sql_query('''
                SELECT * FROM alerts WHERE timestamp > ?
            ''', conn, params=(cutoff_time.isoformat(),))
            
        return {
            'predictions': predictions.to_dict('records'),
            'metrics': metrics.to_dict('records'),
            'alerts': alerts.to_dict('records'),
            'export_timestamp': datetime.now().isoformat()
        }

# FastAPI Application
app = FastAPI(title="Battery Prediction Monitoring API", version="1.0.0")

# Global services
predictor = ProductionBatteryPredictor()
monitoring_service = MonitoringService(predictor)

@app.post("/predict", response_model=PredictionResponse)
async def predict_battery_health(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make battery health prediction with monitoring."""
    try:
        # Make prediction
        result = predictor.predict_battery_health(
            features=request.features,
            model_type=request.model_type
        )
        
        # Record for monitoring
        background_tasks.add_task(
            monitoring_service.record_prediction, 
            request.features, 
            result
        )
        
        # Check for drift
        background_tasks.add_task(
            monitoring_service.detect_drift,
            request.features
        )
        
        return PredictionResponse(
            soh_prediction=result.soh_prediction,
            soc_prediction=result.soc_prediction,
            soh_confidence=result.soh_confidence,
            soc_confidence=result.soc_confidence,
            prediction_time_ms=result.prediction_time_ms,
            model_version=result.model_version,
            timestamp=result.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Service health check."""
    health = predictor.get_health_status()
    
    return HealthCheckResponse(
        status=health['status'],
        timestamp=datetime.now().isoformat(),
        models_loaded=health['models_loaded'],
        performance=health['performance']
    )

@app.get("/metrics")
async def get_metrics(hours: int = 24):
    """Get monitoring metrics."""
    return monitoring_service.get_metrics(hours=hours)

@app.get("/alerts")
async def get_alerts(hours: int = 24):
    """Get recent alerts."""
    return monitoring_service.get_alerts(hours=hours)

@app.post("/reload-models")
async def reload_models():
    """Reload models from disk."""
    return predictor.reload_models()

@app.get("/model-info")
async def get_model_info():
    """Get model information."""
    return predictor.get_model_info()

@app.get("/export-data")
async def export_monitoring_data(hours: int = 24):
    """Export monitoring data."""
    return monitoring_service.export_monitoring_data(hours=hours)

if __name__ == "__main__":
    uvicorn.run("monitoring_service:app", host="0.0.0.0", port=8000, reload=False) 