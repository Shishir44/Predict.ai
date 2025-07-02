"""
Enterprise Health Checking System

Comprehensive health checking with service discovery, dependency monitoring,
and automated recovery capabilities.
"""

import asyncio
import time
import requests
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable
    interval_seconds: int = 30
    timeout_seconds: int = 5
    retry_count: int = 3
    enabled: bool = True
    dependencies: List[str] = None

@dataclass
class HealthResult:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class EnterpriseHealthChecker:
    """
    Enterprise health checking system.
    
    Features:
    - Dependency monitoring
    - Automated recovery
    - Performance tracking
    - Alert integration
    - Service discovery
    """
    
    def __init__(self,
                 db_path: str = "health/health_checks.db",
                 alert_callback: Optional[Callable] = None):
        """
        Initialize health checker.
        
        Args:
            db_path: Path to health check database
            alert_callback: Function to call for alerts
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.alert_callback = alert_callback
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_results: Dict[str, HealthResult] = {}
        self.health_history: Dict[str, List[HealthResult]] = {}
        
        # Control flags
        self.running = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Initialize database
        self._init_database()
        
        # Register default health checks
        self._register_default_checks()
        
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks (simplified version)."""
        results = {}
        
        for name, health_check in self.health_checks.items():
            try:
                start_time = time.time()
                result = health_check.check_function()
                response_time = (time.time() - start_time) * 1000
                
                results[name] = {
                    **result,
                    'response_time_ms': response_time,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                results[name] = {
                    'status': HealthStatus.CRITICAL.value,
                    'message': f'Check failed: {str(e)}',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
        return self._get_overall_health_simple(results)
        
    def _get_overall_health_simple(self, results: Dict) -> Dict[str, Any]:
        """Calculate overall health status (simplified)."""
        statuses = [result.get('status') for result in results.values()]
        
        if any(status == HealthStatus.CRITICAL.value for status in statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.WARNING.value for status in statuses):
            overall_status = HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY.value for status in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
            
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'checks': results,
            'summary': {
                'total_checks': len(results),
                'healthy': sum(1 for s in statuses if s == HealthStatus.HEALTHY.value),
                'warning': sum(1 for s in statuses if s == HealthStatus.WARNING.value),
                'critical': sum(1 for s in statuses if s == HealthStatus.CRITICAL.value)
            }
        }
        
    def _init_database(self):
        """Initialize health check database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    status TEXT,
                    message TEXT,
                    timestamp TEXT,
                    response_time_ms REAL,
                    details TEXT,
                    error TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    status TEXT,
                    message TEXT,
                    timestamp TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        with self._lock:
            self.health_checks[health_check.name] = health_check
            logger.info(f"Registered health check: {health_check.name}")
            
    def _register_default_checks(self):
        """Register default system health checks."""
        # System resource checks
        self.register_health_check(HealthCheck(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            interval_seconds=30
        ))
        
        self.register_health_check(HealthCheck(
            name="memory_usage",
            check_function=self._check_memory_usage,
            interval_seconds=30
        ))
        
        self.register_health_check(HealthCheck(
            name="disk_space",
            check_function=self._check_disk_space,
            interval_seconds=60
        ))
        
        # Model file checks
        self.register_health_check(HealthCheck(
            name="model_files",
            check_function=self._check_model_files,
            interval_seconds=300  # 5 minutes
        ))
        
        # Database connectivity
        self.register_health_check(HealthCheck(
            name="database_connectivity",
            check_function=self._check_database_connectivity,
            interval_seconds=60
        ))
        
    async def run_health_check(self, name: str) -> HealthResult:
        """Run a single health check."""
        if name not in self.health_checks:
            return HealthResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Health check not found",
                timestamp=datetime.now(),
                response_time_ms=0,
                error="Health check not registered"
            )
            
        health_check = self.health_checks[name]
        if not health_check.enabled:
            return HealthResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Health check disabled",
                timestamp=datetime.now(),
                response_time_ms=0
            )
            
        start_time = time.time()
        
        try:
            # Run with timeout
            if asyncio.iscoroutinefunction(health_check.check_function):
                result = await asyncio.wait_for(
                    health_check.check_function(),
                    timeout=health_check.timeout_seconds
                )
            else:
                result = health_check.check_function()
                
            response_time = (time.time() - start_time) * 1000
            
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'unknown'))
                message = result.get('message', 'OK')
                details = result.get('details')
                error = result.get('error')
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                message = "OK" if result else "Check failed"
                details = None
                error = None
                
            return HealthResult(
                name=name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details=details,
                error=error
            )
            
        except asyncio.TimeoutError:
            return HealthResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message="Health check timed out",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error="Timeout"
            )
            
        except Exception as e:
            return HealthResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
            
    async def run_all_health_checks(self) -> Dict[str, HealthResult]:
        """Run all enabled health checks."""
        tasks = []
        for name, health_check in self.health_checks.items():
            if health_check.enabled:
                tasks.append(self.run_health_check(name))
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_results = {}
        for result in results:
            if isinstance(result, HealthResult):
                health_results[result.name] = result
                self._store_health_result(result)
                
        with self._lock:
            self.health_results.update(health_results)
            
        return health_results
        
    def _store_health_result(self, result: HealthResult):
        """Store health result in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO health_results 
                    (name, status, message, timestamp, response_time_ms, details, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.name,
                    result.status.value,
                    result.message,
                    result.timestamp.isoformat(),
                    result.response_time_ms,
                    json.dumps(result.details) if result.details else None,
                    result.error
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing health result: {str(e)}")
            
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        with self._lock:
            if not self.health_results:
                return {
                    'status': HealthStatus.UNKNOWN.value,
                    'message': 'No health checks run yet',
                    'timestamp': datetime.now().isoformat()
                }
                
            statuses = [result.status for result in self.health_results.values()]
            
            # Determine overall status
            if any(status == HealthStatus.CRITICAL for status in statuses):
                overall_status = HealthStatus.CRITICAL
                message = "Critical issues detected"
            elif any(status == HealthStatus.WARNING for status in statuses):
                overall_status = HealthStatus.WARNING
                message = "Warning conditions detected"
            elif all(status == HealthStatus.HEALTHY for status in statuses):
                overall_status = HealthStatus.HEALTHY
                message = "All systems healthy"
            else:
                overall_status = HealthStatus.UNKNOWN
                message = "Unknown system state"
                
            # Calculate performance metrics
            response_times = [r.response_time_ms for r in self.health_results.values()]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                'status': overall_status.value,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'checks_count': len(self.health_results),
                'healthy_count': sum(1 for s in statuses if s == HealthStatus.HEALTHY),
                'warning_count': sum(1 for s in statuses if s == HealthStatus.WARNING),
                'critical_count': sum(1 for s in statuses if s == HealthStatus.CRITICAL),
                'avg_response_time_ms': avg_response_time,
                'details': {name: asdict(result) for name, result in self.health_results.items()}
            }
            
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.running:
            logger.warning("Health monitoring already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self.running:
                try:
                    # Run health checks
                    results = loop.run_until_complete(self.run_all_health_checks())
                    
                    # Check for alerts
                    self._check_alerts(results)
                    
                    time.sleep(10)  # Wait before next check
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(30)  # Wait longer on error
                    
        finally:
            loop.close()
            
    def _check_alerts(self, results: Dict[str, HealthResult]):
        """Check for alert conditions."""
        for name, result in results.items():
            if result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                self._raise_alert(result)
                
    def _raise_alert(self, result: HealthResult):
        """Raise an alert for a health check result."""
        alert_data = {
            'name': result.name,
            'status': result.status.value,
            'message': result.message,
            'timestamp': result.timestamp.isoformat(),
            'error': result.error
        }
        
        # Store alert
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO health_alerts (name, status, message, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (result.name, result.status.value, result.message, result.timestamp.isoformat()))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing alert: {str(e)}")
            
        # Call alert callback
        if self.alert_callback:
            try:
                self.alert_callback(alert_data)
            except Exception as e:
                logger.error(f"Error calling alert callback: {str(e)}")
                
        logger.warning(f"HEALTH ALERT: {result.name} - {result.status.value} - {result.message}")
        
    # Default health check implementations
    def _check_cpu_usage(self) -> Dict:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
        elif cpu_percent > 70:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
            
        return {
            'status': status.value,
            'message': f'CPU usage: {cpu_percent:.1f}%',
            'value': cpu_percent
        }
        
    def _check_memory_usage(self) -> Dict:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            status = HealthStatus.CRITICAL
        elif memory.percent > 80:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
            
        return {
            'status': status.value,
            'message': f'Memory usage: {memory.percent:.1f}%',
            'value': memory.percent
        }
        
    def _check_disk_space(self) -> Dict:
        """Check disk space."""
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100
        
        if disk_percent > 95:
            status = HealthStatus.CRITICAL
        elif disk_percent > 85:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
            
        return {
            'status': status.value,
            'message': f'Disk usage: {disk_percent:.1f}%',
            'value': disk_percent
        }
        
    def _check_model_files(self) -> Dict:
        """Check model files existence."""
        model_files = [
            'models/random_forest_soh_model.joblib',
            'models/feature_scaler.joblib',
            'models/lstm_soh_model.h5'
        ]
        
        existing = sum(1 for f in model_files if Path(f).exists())
        total = len(model_files)
        
        if existing == 0:
            status = HealthStatus.CRITICAL
        elif existing < total:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
            
        return {
            'status': status.value,
            'message': f'Model files: {existing}/{total} available',
            'value': existing / total
        }
        
    def _check_database_connectivity(self) -> Dict:
        """Check database connectivity."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                cursor.fetchone()
                
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'Database connectivity OK',
                'details': {'database_path': str(self.db_path)}
            }
            
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f'Database connectivity failed: {str(e)}',
                'error': str(e)
            } 