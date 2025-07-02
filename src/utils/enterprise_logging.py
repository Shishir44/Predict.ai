"""
Enterprise Logging System

Comprehensive logging system with structured logging, metrics collection,
audit trails, and enterprise-grade features.
"""

import logging
import logging.handlers
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback
import os
import sys
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class LogLevel(Enum):
    """Enhanced log levels."""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    AUDIT = "AUDIT"
    METRICS = "METRICS"
    SECURITY = "SECURITY"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: str = "battery_prediction"
    environment: str = "production"
    metrics: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None
    audit_trail: Optional[Dict[str, Any]] = None

class EnterpriseLogger:
    """
    Enterprise-grade logger with advanced features.
    
    Features:
    - Structured JSON logging
    - Audit trails
    - Performance metrics
    - Security event logging
    - Log aggregation
    - Compliance logging
    """
    
    def __init__(self,
                 name: str = "battery_prediction",
                 log_dir: str = "logs",
                 max_file_size: int = 50 * 1024 * 1024,  # 50MB
                 backup_count: int = 10,
                 enable_console: bool = True,
                 enable_json: bool = True,
                 enable_audit: bool = True,
                 enable_metrics: bool = True):
        """
        Initialize enterprise logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            max_file_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
            enable_console: Enable console output
            enable_json: Enable JSON structured logging
            enable_audit: Enable audit logging
            enable_metrics: Enable metrics logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.enable_json = enable_json
        self.enable_audit = enable_audit
        self.enable_metrics = enable_metrics
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.request_counter = 0
        self._lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {}
        self.error_counts = {}
        
        # Initialize loggers
        self._setup_main_logger(max_file_size, backup_count, enable_console)
        
        if enable_audit:
            self._setup_audit_logger(max_file_size, backup_count)
            
        if enable_metrics:
            self._setup_metrics_logger(max_file_size, backup_count)
            
        # Setup security logger
        self._setup_security_logger(max_file_size, backup_count)
        
    def _setup_main_logger(self, max_file_size: int, backup_count: int, enable_console: bool):
        """Setup main application logger."""
        self.logger = logging.getLogger(f"{self.name}.main")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "application.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # JSON formatter for structured logging
        if self.enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(StandardFormatter())
            
        self.logger.addHandler(file_handler)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(StandardFormatter())
            self.logger.addHandler(console_handler)
            
    def _setup_audit_logger(self, max_file_size: int, backup_count: int):
        """Setup audit logger."""
        self.audit_logger = logging.getLogger(f"{self.name}.audit")
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.handlers.clear()
        
        audit_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "audit.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        audit_handler.setFormatter(JSONFormatter())
        self.audit_logger.addHandler(audit_handler)
        
    def _setup_metrics_logger(self, max_file_size: int, backup_count: int):
        """Setup metrics logger."""
        self.metrics_logger = logging.getLogger(f"{self.name}.metrics")
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.handlers.clear()
        
        metrics_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "metrics.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        metrics_handler.setFormatter(JSONFormatter())
        self.metrics_logger.addHandler(metrics_handler)
        
    def _setup_security_logger(self, max_file_size: int, backup_count: int):
        """Setup security event logger."""
        self.security_logger = logging.getLogger(f"{self.name}.security")
        self.security_logger.setLevel(logging.WARNING)
        self.security_logger.handlers.clear()
        
        security_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "security.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        security_handler.setFormatter(JSONFormatter())
        self.security_logger.addHandler(security_handler)
        
    def get_request_id(self) -> str:
        """Generate unique request ID."""
        with self._lock:
            self.request_counter += 1
            return f"{self.session_id}-{self.request_counter:06d}"
            
    def log(self, 
            level: LogLevel, 
            message: str,
            user_id: Optional[str] = None,
            request_id: Optional[str] = None,
            metrics: Optional[Dict[str, Any]] = None,
            error_details: Optional[Dict[str, Any]] = None,
            audit_data: Optional[Dict[str, Any]] = None):
        """
        Log message with structured data.
        
        Args:
            level: Log level
            message: Log message
            user_id: User identifier
            request_id: Request identifier
            metrics: Performance metrics
            error_details: Error information
            audit_data: Audit trail data
        """
        # Get caller information
        frame = sys._getframe(1)
        module = frame.f_globals.get('__name__', 'unknown')
        function = frame.f_code.co_name
        
        # Create log entry
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            message=message,
            module=module,
            function=function,
            user_id=user_id,
            session_id=self.session_id,
            request_id=request_id or self.get_request_id(),
            metrics=metrics,
            error_details=error_details,
            audit_trail=audit_data
        )
        
        # Route to appropriate logger
        if level == LogLevel.AUDIT and self.enable_audit:
            self.audit_logger.info(json.dumps(asdict(log_entry)))
        elif level == LogLevel.METRICS and self.enable_metrics:
            self.metrics_logger.info(json.dumps(asdict(log_entry)))
        elif level == LogLevel.SECURITY:
            self.security_logger.warning(json.dumps(asdict(log_entry)))
        else:
            # Main logger
            log_method = getattr(self.logger, level.value.lower())
            log_method(json.dumps(asdict(log_entry)) if self.enable_json else message)
            
        # Update performance tracking
        if metrics:
            self._update_performance_metrics(metrics)
            
        # Update error tracking
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._update_error_counts(module, function)
            
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **kwargs)
        
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details."""
        error_details = None
        if exception:
            error_details = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
        self.log(LogLevel.ERROR, message, error_details=error_details, **kwargs)
        
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message."""
        error_details = None
        if exception:
            error_details = {
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            }
        self.log(LogLevel.CRITICAL, message, error_details=error_details, **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
        
    def audit(self, action: str, resource: str, user_id: str, result: str, **kwargs):
        """Log audit event."""
        audit_data = {
            'action': action,
            'resource': resource,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        audit_data.update(kwargs)
        
        self.log(LogLevel.AUDIT, f"Audit: {action} on {resource} by {user_id}: {result}",
                user_id=user_id, audit_data=audit_data)
                
    def security(self, event: str, severity: str, details: Dict[str, Any], **kwargs):
        """Log security event."""
        security_data = {
            'event': event,
            'severity': severity,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        self.log(LogLevel.SECURITY, f"Security Event: {event} (Severity: {severity})",
                audit_data=security_data, **kwargs)
                
    def metrics(self, metric_name: str, value: float, unit: str = "", tags: Optional[Dict] = None, **kwargs):
        """Log performance metric."""
        metrics_data = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'tags': tags or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.log(LogLevel.METRICS, f"Metric: {metric_name} = {value} {unit}",
                metrics=metrics_data, **kwargs)
                
    def _update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics tracking."""
        with self._lock:
            for key, value in metrics.items():
                if key not in self.performance_metrics:
                    self.performance_metrics[key] = []
                self.performance_metrics[key].append(value)
                
                # Keep only last 1000 entries
                if len(self.performance_metrics[key]) > 1000:
                    self.performance_metrics[key] = self.performance_metrics[key][-1000:]
                    
    def _update_error_counts(self, module: str, function: str):
        """Update error counts."""
        with self._lock:
            key = f"{module}.{function}"
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        with self._lock:
            summary = {}
            for metric, values in self.performance_metrics.items():
                if values:
                    summary[metric] = {
                        'count': len(values),
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'recent': values[-10:]  # Last 10 values
                    }
            return summary
            
    def get_error_summary(self) -> Dict[str, int]:
        """Get error counts summary."""
        with self._lock:
            return self.error_counts.copy()

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        # If record.msg is already JSON, return it as is
        if hasattr(record, 'msg') and record.msg.startswith('{'):
            return record.msg
            
        # Otherwise create JSON structure
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

class StandardFormatter(logging.Formatter):
    """Standard formatter for human-readable logs."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

# Global logger instance
enterprise_logger = EnterpriseLogger()

def get_logger() -> EnterpriseLogger:
    """Get global enterprise logger instance."""
    return enterprise_logger

# Convenience functions
def log_info(message: str, **kwargs):
    """Log info message."""
    enterprise_logger.info(message, **kwargs)

def log_warning(message: str, **kwargs):
    """Log warning message."""
    enterprise_logger.warning(message, **kwargs)

def log_error(message: str, exception: Optional[Exception] = None, **kwargs):
    """Log error message."""
    enterprise_logger.error(message, exception=exception, **kwargs)

def log_audit(action: str, resource: str, user_id: str, result: str, **kwargs):
    """Log audit event."""
    enterprise_logger.audit(action, resource, user_id, result, **kwargs)

def log_security(event: str, severity: str, details: Dict[str, Any], **kwargs):
    """Log security event."""
    enterprise_logger.security(event, severity, details, **kwargs)

def log_metric(metric_name: str, value: float, unit: str = "", tags: Optional[Dict] = None, **kwargs):
    """Log performance metric."""
    enterprise_logger.metrics(metric_name, value, unit, tags, **kwargs) 