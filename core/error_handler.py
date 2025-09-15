"""Enhanced error handling and recovery mechanisms for LangGraph document processing."""

import asyncio
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path
import json


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    IO_ERROR = "io_error"
    NETWORK = "network"
    LLM_API = "llm_api"
    PROCESSING = "processing"
    MEMORY = "memory"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    DEGRADE = "degrade"
    RESTART = "restart"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception: Optional[Exception] = None
    traceback_str: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "traceback": self.traceback_str,
            "component": self.component,
            "operation": self.operation,
            "metadata": self.metadata,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts
        }


@dataclass
class RecoveryAction:
    """Recovery action configuration."""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay: float = 60.0
    fallback_function: Optional[Callable] = None
    condition_check: Optional[Callable[[ErrorContext], bool]] = None
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff."""
        delay = self.delay_seconds * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay)


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery strategies."""
    
    def __init__(self):
        self.classification_rules = {
            # Network and API errors
            (ConnectionError, TimeoutError): (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            (OSError,): (ErrorCategory.IO_ERROR, ErrorSeverity.MEDIUM),
            
            # LLM API specific errors
            "rate_limit": (ErrorCategory.LLM_API, ErrorSeverity.MEDIUM),
            "api_key": (ErrorCategory.LLM_API, ErrorSeverity.HIGH),
            "quota_exceeded": (ErrorCategory.LLM_API, ErrorSeverity.HIGH),
            
            # Processing errors
            (ValueError, TypeError): (ErrorCategory.VALIDATION, ErrorSeverity.LOW),
            (MemoryError,): (ErrorCategory.MEMORY, ErrorSeverity.CRITICAL),
            
            # Configuration errors
            (FileNotFoundError,): (ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH),
        }
        
        self.recovery_strategies = {
            ErrorCategory.NETWORK: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=3,
                delay_seconds=2.0,
                backoff_multiplier=2.0
            ),
            ErrorCategory.LLM_API: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=5,
                delay_seconds=5.0,
                backoff_multiplier=1.5
            ),
            ErrorCategory.MEMORY: RecoveryAction(
                strategy=RecoveryStrategy.DEGRADE,
                max_attempts=1
            ),
            ErrorCategory.VALIDATION: RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                max_attempts=1
            ),
            ErrorCategory.MEMORY: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                max_attempts=0
            )
        }
    
    def classify_error(self, exception: Exception, message: str = "") -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify an error based on exception type and message."""
        # Check exception type
        for exc_types, (category, severity) in self.classification_rules.items():
            if isinstance(exc_types, tuple) and isinstance(exception, exc_types):
                return category, severity
        
        # Check message content
        message_lower = message.lower()
        for keyword, (category, severity) in self.classification_rules.items():
            if isinstance(keyword, str) and keyword in message_lower:
                return category, severity
        
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def get_recovery_strategy(self, category: ErrorCategory) -> RecoveryAction:
        """Get recovery strategy for error category."""
        return self.recovery_strategies.get(category, RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            max_attempts=1
        ))


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.classifier = ErrorClassifier()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.logger = self._setup_logger(log_file)
        self.recovery_callbacks: Dict[str, Callable] = {}
        
    def _setup_logger(self, log_file: Optional[Path]) -> logging.Logger:
        """Setup error logging."""
        logger = logging.getLogger("error_handler")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
        
        logger.addHandler(console_handler)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        for handler in logger.handlers:
            handler.setFormatter(formatter)
        
        return logger
    
    def get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for component."""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
        return self.circuit_breakers[component]
    
    def register_recovery_callback(self, error_type: str, callback: Callable):
        """Register custom recovery callback."""
        self.recovery_callbacks[error_type] = callback
    
    async def handle_error(
        self,
        exception: Exception,
        component: str = "unknown",
        operation: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Handle an error with classification and recovery."""
        # Create error context
        error_id = f"{component}_{operation}_{datetime.now().timestamp()}"
        category, severity = self.classifier.classify_error(exception, str(exception))
        
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=str(exception),
            exception=exception,
            traceback_str=traceback.format_exc(),
            component=component,
            operation=operation,
            metadata=metadata or {}
        )
        
        # Log error
        self.logger.error(
            f"Error in {component}.{operation}: {exception}",
            extra={"error_context": error_context.to_dict()}
        )
        
        # Add to history
        self.error_history.append(error_context)
        
        # Update circuit breaker
        circuit_breaker = self.get_circuit_breaker(component)
        circuit_breaker.record_failure()
        
        return error_context
    
    async def attempt_recovery(
        self,
        error_context: ErrorContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt recovery using appropriate strategy."""
        recovery_action = self.classifier.get_recovery_strategy(error_context.category)
        
        if error_context.recovery_attempts >= recovery_action.max_attempts:
            self.logger.error(f"Max recovery attempts exceeded for {error_context.error_id}")
            raise error_context.exception
        
        error_context.recovery_attempts += 1
        
        if recovery_action.strategy == RecoveryStrategy.RETRY:
            return await self._retry_operation(
                error_context, recovery_action, operation_func, *args, **kwargs
            )
        elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
            return await self._fallback_operation(
                error_context, recovery_action, *args, **kwargs
            )
        elif recovery_action.strategy == RecoveryStrategy.SKIP:
            self.logger.warning(f"Skipping operation due to error: {error_context.error_id}")
            return None
        elif recovery_action.strategy == RecoveryStrategy.DEGRADE:
            return await self._degrade_operation(
                error_context, operation_func, *args, **kwargs
            )
        else:  # ABORT
            self.logger.critical(f"Aborting due to critical error: {error_context.error_id}")
            raise error_context.exception
    
    async def _retry_operation(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Retry operation with exponential backoff."""
        delay = recovery_action.calculate_delay(error_context.recovery_attempts - 1)
        
        self.logger.info(
            f"Retrying operation {error_context.operation} "
            f"(attempt {error_context.recovery_attempts}) after {delay}s delay"
        )
        
        await asyncio.sleep(delay)
        
        try:
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            # Record success
            circuit_breaker = self.get_circuit_breaker(error_context.component)
            circuit_breaker.record_success()
            
            return result
        except Exception as e:
            # Handle retry failure
            new_error_context = await self.handle_error(
                e, error_context.component, error_context.operation
            )
            return await self.attempt_recovery(
                new_error_context, operation_func, *args, **kwargs
            )
    
    async def _fallback_operation(
        self,
        error_context: ErrorContext,
        recovery_action: RecoveryAction,
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback operation."""
        if recovery_action.fallback_function:
            self.logger.info(f"Executing fallback for {error_context.operation}")
            if asyncio.iscoroutinefunction(recovery_action.fallback_function):
                return await recovery_action.fallback_function(*args, **kwargs)
            else:
                return recovery_action.fallback_function(*args, **kwargs)
        else:
            self.logger.warning(f"No fallback available for {error_context.operation}")
            return None
    
    async def _degrade_operation(
        self,
        error_context: ErrorContext,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation in degraded mode."""
        self.logger.info(f"Executing {error_context.operation} in degraded mode")
        
        # Modify kwargs for degraded performance
        degraded_kwargs = kwargs.copy()
        if 'chunk_size' in degraded_kwargs:
            degraded_kwargs['chunk_size'] = min(degraded_kwargs['chunk_size'], 500)
        if 'max_concurrent' in degraded_kwargs:
            degraded_kwargs['max_concurrent'] = 1
        
        try:
            if asyncio.iscoroutinefunction(operation_func):
                return await operation_func(*args, **degraded_kwargs)
            else:
                return operation_func(*args, **degraded_kwargs)
        except Exception as e:
            # If degraded mode also fails, abort
            self.logger.critical(f"Degraded mode failed for {error_context.operation}: {e}")
            raise e
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and health metrics."""
        if not self.error_history:
            return {"total_errors": 0, "health_status": "healthy"}
        
        recent_errors = [
            err for err in self.error_history
            if datetime.now() - err.timestamp < timedelta(hours=1)
        ]
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        critical_errors = severity_counts.get("critical", 0)
        high_errors = severity_counts.get("high", 0)
        
        if critical_errors > 0:
            health_status = "critical"
        elif high_errors > 3:
            health_status = "degraded"
        elif len(recent_errors) > 10:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "health_status": health_status,
            "circuit_breaker_states": {
                name: cb.state for name, cb in self.circuit_breakers.items()
            }
        }
    
    def export_error_report(self, output_path: Path):
        """Export detailed error report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_error_statistics(),
            "error_history": [err.to_dict() for err in self.error_history[-100:]],  # Last 100 errors
            "circuit_breaker_status": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Error report exported to {output_path}")


# Decorator for automatic error handling
def handle_errors(
    component: str = "unknown",
    operation: str = "unknown",
    error_handler: Optional[ErrorHandler] = None
):
    """Decorator for automatic error handling and recovery."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = await handler.handle_error(e, component, operation)
                return await handler.attempt_recovery(error_context, func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we can't use async recovery
                error_context = asyncio.run(handler.handle_error(e, component, operation))
                raise e
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator