"""
Professional Logging System for the Counsel Of Agents Orchestration Platform.

Provides structured, configurable logging with support for:
- Multiple output targets (console, file, JSON)
- Log levels and filtering
- Contextual information (agent ID, task ID, etc.)
- Performance metrics
- Business analytics events
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from functools import wraps
import time


class LogLevel(Enum):
    """Log levels for the Counsel platform."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    # Custom levels for business events
    METRIC = "METRIC"
    AUDIT = "AUDIT"
    TELEMETRY = "TELEMETRY"


@dataclass
class LogEvent:
    """A structured log event."""
    timestamp: str
    level: str
    message: str
    logger_name: str
    
    # Context
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    job_id: Optional[str] = None
    
    # Additional data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    duration_ms: Optional[float] = None
    
    # Error info
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in result.items() if v is not None}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        event = LogEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            agent_id=getattr(record, 'agent_id', None),
            task_id=getattr(record, 'task_id', None),
            job_id=getattr(record, 'job_id', None),
            data=getattr(record, 'data', {}),
            duration_ms=getattr(record, 'duration_ms', None),
        )
        
        if record.exc_info:
            event.error_type = record.exc_info[0].__name__ if record.exc_info[0] else None
            event.error_message = str(record.exc_info[1]) if record.exc_info[1] else None
            event.stack_trace = self.formatException(record.exc_info)
        
        return event.to_json()


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'METRIC': '\033[34m',     # Blue
        'AUDIT': '\033[37m',      # White
        'TELEMETRY': '\033[90m',  # Gray
    }
    RESET = '\033[0m'
    DIM = '\033[2m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET
        dim = self.DIM
        
        # Build context string
        context_parts = []
        if hasattr(record, 'agent_id') and record.agent_id:
            context_parts.append(f"agent={record.agent_id}")
        if hasattr(record, 'task_id') and record.task_id:
            context_parts.append(f"task={record.task_id}")
        if hasattr(record, 'job_id') and record.job_id:
            context_parts.append(f"job={record.job_id}")
        
        context_str = f" {dim}[{', '.join(context_parts)}]{reset}" if context_parts else ""
        
        # Format timestamp
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        # Duration if present
        duration_str = ""
        if hasattr(record, 'duration_ms') and record.duration_ms is not None:
            duration_str = f" {dim}({record.duration_ms:.1f}ms){reset}"
        
        return f"{dim}{timestamp}{reset} {color}{record.levelname:8}{reset}{context_str} {record.getMessage()}{duration_str}"


class CounselLogger:
    """
    Professional logger for the Counsel platform.
    
    Provides:
    - Structured logging with context
    - Multiple output formats
    - Performance timing
    - Business event tracking
    """
    
    def __init__(
        self,
        name: str = "counsel",
        level: Union[str, LogLevel] = LogLevel.INFO,
        log_file: Optional[str] = None,
        json_output: bool = False,
        enable_console: bool = True
    ):
        self.name = name
        self._logger = logging.getLogger(name)
        
        # Convert level
        if isinstance(level, LogLevel):
            level = level.value
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Remove existing handlers
        self._logger.handlers = []
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if json_output:
                console_handler.setFormatter(JsonFormatter())
            else:
                console_handler.setFormatter(ColoredConsoleFormatter())
            self._logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JsonFormatter())  # Always JSON for files
            self._logger.addHandler(file_handler)
        
        # Context storage (thread-local)
        self._context = threading.local()
    
    def set_context(
        self,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        job_id: Optional[str] = None
    ):
        """Set contextual information for subsequent log calls."""
        if agent_id is not None:
            self._context.agent_id = agent_id
        if task_id is not None:
            self._context.task_id = task_id
        if job_id is not None:
            self._context.job_id = job_id
    
    def clear_context(self):
        """Clear all context."""
        self._context.agent_id = None
        self._context.task_id = None
        self._context.job_id = None
    
    def _log(
        self,
        level: int,
        message: str,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        job_id: Optional[str] = None,
        data: Optional[Dict] = None,
        duration_ms: Optional[float] = None,
        exc_info: bool = False
    ):
        """Internal log method with context support."""
        extra = {
            'agent_id': agent_id or getattr(self._context, 'agent_id', None),
            'task_id': task_id or getattr(self._context, 'task_id', None),
            'job_id': job_id or getattr(self._context, 'job_id', None),
            'data': data or {},
            'duration_ms': duration_ms,
        }
        self._logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log an info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log an error message."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log a critical message."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)
    
    def metric(
        self,
        name: str,
        value: Union[int, float],
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """Log a metric for analytics."""
        data = {
            'metric_name': name,
            'metric_value': value,
            'metric_unit': unit,
            'tags': tags or {}
        }
        self._log(logging.INFO, f"METRIC: {name}={value}{unit}", data=data, **kwargs)
    
    def audit(
        self,
        action: str,
        resource: str,
        outcome: str,
        details: Optional[Dict] = None,
        **kwargs
    ):
        """Log an audit event for compliance tracking."""
        data = {
            'audit_action': action,
            'audit_resource': resource,
            'audit_outcome': outcome,
            'audit_details': details or {}
        }
        self._log(logging.INFO, f"AUDIT: {action} on {resource} -> {outcome}", data=data, **kwargs)
    
    def telemetry(
        self,
        event_name: str,
        properties: Optional[Dict] = None,
        **kwargs
    ):
        """Log a telemetry event for business analytics."""
        data = {
            'event_name': event_name,
            'event_properties': properties or {}
        }
        self._log(logging.INFO, f"TELEMETRY: {event_name}", data=data, **kwargs)
    
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        return _LogTimer(self, operation_name)
    
    def timed(self, operation_name: Optional[str] = None):
        """Decorator for timing function calls."""
        def decorator(func):
            name = operation_name or func.__name__
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.timer(name):
                    return func(*args, **kwargs)
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = (time.perf_counter() - start) * 1000
                    self.info(f"Completed {name}", duration_ms=duration)
            
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper
        return decorator


class _LogTimer:
    """Context manager for timing operations."""
    
    def __init__(self, logger: CounselLogger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.perf_counter() - self.start_time) * 1000
        if exc_type:
            self.logger.error(
                f"Failed {self.operation_name}",
                duration_ms=duration,
                exc_info=True
            )
        else:
            self.logger.info(f"Completed {self.operation_name}", duration_ms=duration)
        return False


# Global logger instance
_logger: Optional[CounselLogger] = None
_logger_lock = threading.Lock()


def get_logger(
    name: str = "counsel",
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[str] = None
) -> CounselLogger:
    """Get or create the global Counsel logger."""
    global _logger
    
    with _logger_lock:
        if _logger is None:
            _logger = CounselLogger(
                name=name,
                level=level,
                log_file=log_file
            )
        return _logger


def configure_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[str] = None,
    json_output: bool = False,
    enable_console: bool = True
):
    """Configure the global logger."""
    global _logger
    
    with _logger_lock:
        _logger = CounselLogger(
            name="counsel",
            level=level,
            log_file=log_file,
            json_output=json_output,
            enable_console=enable_console
        )


# Convenience exports
def debug(message: str, **kwargs):
    get_logger().debug(message, **kwargs)

def info(message: str, **kwargs):
    get_logger().info(message, **kwargs)

def warning(message: str, **kwargs):
    get_logger().warning(message, **kwargs)

def error(message: str, **kwargs):
    get_logger().error(message, **kwargs)

def critical(message: str, **kwargs):
    get_logger().critical(message, **kwargs)

def metric(name: str, value: Union[int, float], **kwargs):
    get_logger().metric(name, value, **kwargs)

def audit(action: str, resource: str, outcome: str, **kwargs):
    get_logger().audit(action, resource, outcome, **kwargs)

def telemetry(event_name: str, **kwargs):
    get_logger().telemetry(event_name, **kwargs)
