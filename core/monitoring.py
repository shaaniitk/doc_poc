"""Comprehensive monitoring, logging, and observability system for LangGraph document processing."""

import asyncio
import json
import logging
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import threading
from contextlib import asynccontextmanager, contextmanager


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class LogLevel(Enum):
    """Enhanced log levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    throughput_docs_per_second: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 0.0
    queue_size: int = 0
    active_tasks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "processing_time": self.processing_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "throughput_docs_per_second": self.throughput_docs_per_second,
            "error_rate": self.error_rate,
            "success_rate": self.success_rate,
            "queue_size": self.queue_size,
            "active_tasks": self.active_tasks
        }


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += value
            metric = Metric(
                name=name,
                value=self.counters[name],
                metric_type=MetricType.COUNTER,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        with self._lock:
            self.histograms[name].append(value)
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer duration."""
        with self._lock:
            self.timers[name].append(duration)
            metric = Metric(
                name=name,
                value=duration,
                metric_type=MetricType.TIMER,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self._lock:
            if name not in self.metrics:
                return {}
            
            values = [m.value for m in self.metrics[name]]
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else 0,
                "total": sum(values) if name in self.counters else None
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {k: list(v) for k, v in self.histograms.items()},
                "timers": {k: list(v) for k, v in self.timers.items()}
            }


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.metrics_collector = MetricsCollector()
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self):
        """Start system monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = self.process.cpu_percent()
                self.metrics_collector.set_gauge("system.cpu_percent", cpu_percent)
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.metrics_collector.set_gauge("system.memory_mb", memory_mb)
                
                # System-wide metrics
                system_cpu = psutil.cpu_percent()
                system_memory = psutil.virtual_memory().percent
                
                self.metrics_collector.set_gauge("system.total_cpu_percent", system_cpu)
                self.metrics_collector.set_gauge("system.total_memory_percent", system_memory)
                
                # Disk usage for current directory
                disk_usage = psutil.disk_usage('.')
                disk_percent = (disk_usage.used / disk_usage.total) * 100
                self.metrics_collector.set_gauge("system.disk_percent", disk_percent)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        metrics = self.metrics_collector.get_all_metrics()
        
        return PerformanceMetrics(
            memory_usage_mb=metrics["gauges"].get("system.memory_mb", 0.0),
            cpu_usage_percent=metrics["gauges"].get("system.cpu_percent", 0.0)
        )


class StructuredLogger:
    """Enhanced structured logging with context and correlation."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None, log_level: LogLevel = LogLevel.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.value.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        self.context_stack: List[Dict[str, Any]] = []
        self.correlation_id: Optional[str] = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking."""
        self.correlation_id = correlation_id
    
    @contextmanager
    def context(self, **kwargs):
        """Add context to all log messages within this block."""
        self.context_stack.append(kwargs)
        try:
            yield
        finally:
            self.context_stack.pop()
    
    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format message with context and correlation ID."""
        context = {}
        for ctx in self.context_stack:
            context.update(ctx)
        
        if extra:
            context.update(extra)
        
        if self.correlation_id:
            context["correlation_id"] = self.correlation_id
        
        if context:
            context_str = json.dumps(context, default=str)
            return f"{message} | Context: {context_str}"
        
        return message
    
    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self.logger.debug(self._format_message(f"[TRACE] {message}", kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(self._format_message(message, kwargs))


class PerformanceTracker:
    """Tracks performance metrics and timing information."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_timers: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics_collector.record_timer(name, duration, tags)
    
    @asynccontextmanager
    async def async_timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Async context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics_collector.record_timer(name, duration, tags)
    
    def start_timer(self, name: str):
        """Start a named timer."""
        with self._lock:
            self.active_timers[name] = time.time()
    
    def stop_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Stop a named timer and record the duration."""
        with self._lock:
            if name not in self.active_timers:
                raise ValueError(f"Timer '{name}' was not started")
            
            duration = time.time() - self.active_timers[name]
            del self.active_timers[name]
            
            self.metrics_collector.record_timer(name, duration, tags)
            return duration


class HealthChecker:
    """Monitors system health and provides health checks."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_func):
                    is_healthy = await check_func()
                else:
                    is_healthy = check_func()
                
                duration = time.time() - start_time
                
                results[name] = {
                    "healthy": is_healthy,
                    "duration_ms": duration * 1000,
                    "timestamp": datetime.now().isoformat()
                }
                
                if not is_healthy:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "checks": results,
            "timestamp": datetime.now().isoformat()
        }


class MonitoringSystem:
    """Comprehensive monitoring system integrating all components."""
    
    def __init__(self, log_dir: Optional[Path] = None, enable_system_monitoring: bool = True):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor() if enable_system_monitoring else None
        self.performance_tracker = PerformanceTracker(self.metrics_collector)
        self.health_checker = HealthChecker()
        
        # Initialize logger
        self.logger = StructuredLogger(
            "monitoring_system",
            log_file=self.log_dir / "system.log"
        )
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        def memory_check() -> bool:
            """Check if memory usage is within acceptable limits."""
            memory_percent = psutil.virtual_memory().percent
            return memory_percent < 90
        
        def disk_check() -> bool:
            """Check if disk usage is within acceptable limits."""
            disk_percent = psutil.disk_usage('.').used / psutil.disk_usage('.').total * 100
            return disk_percent < 95
        
        self.health_checker.register_health_check("memory", memory_check)
        self.health_checker.register_health_check("disk", disk_check)
    
    async def start(self):
        """Start the monitoring system."""
        self.logger.info("Starting monitoring system")
        
        if self.system_monitor:
            await self.system_monitor.start_monitoring()
        
        self.logger.info("Monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system."""
        self.logger.info("Stopping monitoring system")
        
        if self.system_monitor:
            await self.system_monitor.stop_monitoring()
        
        self.logger.info("Monitoring system stopped")
    
    def get_logger(self, name: str) -> StructuredLogger:
        """Get a structured logger for a component."""
        return StructuredLogger(
            name,
            log_file=self.log_dir / f"{name}.log"
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        metrics = self.metrics_collector.get_all_metrics()
        
        # Get system metrics if available
        system_metrics = {}
        if self.system_monitor:
            current_metrics = self.system_monitor.get_current_metrics()
            system_metrics = current_metrics.to_dict()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "system_metrics": system_metrics,
            "health_status": asyncio.run(self.health_checker.run_health_checks())
        }
    
    def export_metrics(self, output_path: Path, format: str = "json"):
        """Export metrics to file."""
        data = self.get_dashboard_data()
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Metrics exported to {output_path}")


# Decorators for automatic monitoring
def monitor_performance(name: str, monitoring_system: Optional[MonitoringSystem] = None):
    """Decorator to automatically monitor function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            system = monitoring_system or MonitoringSystem()
            
            with system.performance_tracker.timer(f"{name}.duration"):
                system.metrics_collector.increment_counter(f"{name}.calls")
                
                try:
                    result = await func(*args, **kwargs)
                    system.metrics_collector.increment_counter(f"{name}.success")
                    return result
                except Exception as e:
                    system.metrics_collector.increment_counter(f"{name}.errors")
                    raise
        
        def sync_wrapper(*args, **kwargs):
            system = monitoring_system or MonitoringSystem()
            
            with system.performance_tracker.timer(f"{name}.duration"):
                system.metrics_collector.increment_counter(f"{name}.calls")
                
                try:
                    result = func(*args, **kwargs)
                    system.metrics_collector.increment_counter(f"{name}.success")
                    return result
                except Exception as e:
                    system.metrics_collector.increment_counter(f"{name}.errors")
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator