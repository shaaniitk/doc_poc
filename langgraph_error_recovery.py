"""Error handling and recovery state management for LangGraph workflow"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import traceback
import json
import logging
from langgraph_state import ErrorInfo, ErrorSeverity, ProcessingStage


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"                    # Retry the failed operation
    SKIP = "skip"                      # Skip the failed operation
    FALLBACK = "fallback"              # Use fallback implementation
    ROLLBACK = "rollback"              # Rollback to previous state
    MANUAL = "manual"                  # Require manual intervention
    ABORT = "abort"                    # Abort the entire workflow


class RecoveryAction(Enum):
    """Recovery action types"""
    IMMEDIATE = "immediate"            # Execute recovery immediately
    DELAYED = "delayed"                # Execute recovery after delay
    CONDITIONAL = "conditional"        # Execute recovery based on condition
    ESCALATED = "escalated"            # Escalate to higher level handler


class ErrorCategory(Enum):
    """Error categorization for recovery decisions"""
    TRANSIENT = "transient"            # Temporary errors (network, timeout)
    CONFIGURATION = "configuration"    # Configuration-related errors
    DATA = "data"                      # Data format or content errors
    RESOURCE = "resource"              # Resource availability errors
    LOGIC = "logic"                    # Business logic errors
    SYSTEM = "system"                  # System-level errors
    EXTERNAL = "external"              # External service errors


@dataclass
class RecoveryRule:
    """Rule for error recovery"""
    
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    processing_stage: Optional[ProcessingStage] = None
    error_pattern: Optional[str] = None  # Regex pattern for error message
    
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    action: RecoveryAction = RecoveryAction.IMMEDIATE
    
    max_attempts: int = 3
    delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 60.0
    
    condition_check: Optional[Callable[['ErrorContext'], bool]] = None
    custom_handler: Optional[Callable[['ErrorContext'], Any]] = None
    
    priority: int = 0  # Higher priority rules are checked first
    
    def matches(self, error_context: 'ErrorContext') -> bool:
        """Check if this rule matches the error context"""
        # Check category
        if self.error_category != error_context.category:
            return False
        
        # Check severity
        if self.error_severity != error_context.error_info.severity:
            return False
        
        # Check processing stage
        if self.processing_stage and self.processing_stage != error_context.processing_stage:
            return False
        
        # Check error pattern
        if self.error_pattern:
            import re
            if not re.search(self.error_pattern, error_context.error_info.message):
                return False
        
        # Check custom condition
        if self.condition_check and not self.condition_check(error_context):
            return False
        
        return True


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    
    attempt_id: str
    timestamp: datetime
    strategy: RecoveryStrategy
    action: RecoveryAction
    
    success: bool = False
    error_message: Optional[str] = None
    execution_time: float = 0.0
    
    # Recovery-specific data
    retry_count: int = 0
    delay_applied: float = 0.0
    fallback_used: Optional[str] = None
    rollback_point: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'attempt_id': self.attempt_id,
            'timestamp': self.timestamp.isoformat(),
            'strategy': self.strategy.value,
            'action': self.action.value,
            'success': self.success,
            'error_message': self.error_message,
            'execution_time': self.execution_time,
            'retry_count': self.retry_count,
            'delay_applied': self.delay_applied,
            'fallback_used': self.fallback_used,
            'rollback_point': self.rollback_point
        }


@dataclass
class ErrorContext:
    """Context information for error handling"""
    
    error_info: ErrorInfo
    processing_stage: ProcessingStage
    category: ErrorCategory
    
    # State information
    current_state: Dict[str, Any] = field(default_factory=dict)
    previous_states: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recovery history
    recovery_attempts: List[RecoveryAttempt] = field(default_factory=list)
    
    # Additional context
    document_info: Optional[Dict[str, Any]] = None
    chunk_info: Optional[Dict[str, Any]] = None
    session_info: Optional[Dict[str, Any]] = None
    
    def add_recovery_attempt(self, attempt: RecoveryAttempt):
        """Add a recovery attempt to history"""
        self.recovery_attempts.append(attempt)
    
    def get_retry_count(self, strategy: RecoveryStrategy) -> int:
        """Get retry count for a specific strategy"""
        return sum(1 for attempt in self.recovery_attempts 
                  if attempt.strategy == strategy)
    
    def get_last_attempt(self, strategy: Optional[RecoveryStrategy] = None) -> Optional[RecoveryAttempt]:
        """Get the last recovery attempt"""
        attempts = self.recovery_attempts
        if strategy:
            attempts = [a for a in attempts if a.strategy == strategy]
        
        return attempts[-1] if attempts else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'error_info': self.error_info.to_dict(),
            'processing_stage': self.processing_stage.value,
            'category': self.category.value,
            'current_state': self.current_state,
            'previous_states': self.previous_states,
            'recovery_attempts': [attempt.to_dict() for attempt in self.recovery_attempts],
            'document_info': self.document_info,
            'chunk_info': self.chunk_info,
            'session_info': self.session_info
        }


class ErrorRecoveryManager:
    """Manages error recovery strategies and execution"""
    
    def __init__(self):
        self.recovery_rules: List[RecoveryRule] = []
        self.fallback_handlers: Dict[ProcessingStage, Callable] = {}
        self.rollback_points: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize default recovery rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default recovery rules"""
        # Transient errors - retry with backoff
        self.add_recovery_rule(RecoveryRule(
            error_category=ErrorCategory.TRANSIENT,
            error_severity=ErrorSeverity.WARNING,
            strategy=RecoveryStrategy.RETRY,
            max_attempts=3,
            delay_seconds=1.0,
            backoff_multiplier=2.0
        ))
        
        # Configuration errors - manual intervention
        self.add_recovery_rule(RecoveryRule(
            error_category=ErrorCategory.CONFIGURATION,
            error_severity=ErrorSeverity.ERROR,
            strategy=RecoveryStrategy.MANUAL,
            action=RecoveryAction.ESCALATED
        ))
        
        # Data errors - skip or fallback
        self.add_recovery_rule(RecoveryRule(
            error_category=ErrorCategory.DATA,
            error_severity=ErrorSeverity.WARNING,
            strategy=RecoveryStrategy.SKIP,
            action=RecoveryAction.IMMEDIATE
        ))
        
        self.add_recovery_rule(RecoveryRule(
            error_category=ErrorCategory.DATA,
            error_severity=ErrorSeverity.ERROR,
            strategy=RecoveryStrategy.FALLBACK,
            action=RecoveryAction.IMMEDIATE
        ))
        
        # Resource errors - retry with longer delay
        self.add_recovery_rule(RecoveryRule(
            error_category=ErrorCategory.RESOURCE,
            error_severity=ErrorSeverity.ERROR,
            strategy=RecoveryStrategy.RETRY,
            max_attempts=5,
            delay_seconds=5.0,
            backoff_multiplier=1.5
        ))
        
        # System errors - abort
        self.add_recovery_rule(RecoveryRule(
            error_category=ErrorCategory.SYSTEM,
            error_severity=ErrorSeverity.CRITICAL,
            strategy=RecoveryStrategy.ABORT,
            action=RecoveryAction.IMMEDIATE
        ))
        
        # External service errors - fallback
        self.add_recovery_rule(RecoveryRule(
            error_category=ErrorCategory.EXTERNAL,
            error_severity=ErrorSeverity.ERROR,
            strategy=RecoveryStrategy.FALLBACK,
            action=RecoveryAction.IMMEDIATE
        ))
    
    def add_recovery_rule(self, rule: RecoveryRule):
        """Add a recovery rule"""
        self.recovery_rules.append(rule)
        # Sort by priority (higher first)
        self.recovery_rules.sort(key=lambda r: r.priority, reverse=True)
    
    def register_fallback_handler(self, stage: ProcessingStage, handler: Callable):
        """Register a fallback handler for a processing stage"""
        self.fallback_handlers[stage] = handler
    
    def create_rollback_point(self, point_id: str, state: Dict[str, Any]):
        """Create a rollback point"""
        self.rollback_points[point_id] = state.copy()
    
    def categorize_error(self, error_info: ErrorInfo, 
                        processing_stage: ProcessingStage,
                        context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """Categorize an error for recovery decision"""
        error_message = error_info.message.lower()
        
        # Check for transient errors
        transient_keywords = ['timeout', 'connection', 'network', 'temporary', 'retry']
        if any(keyword in error_message for keyword in transient_keywords):
            return ErrorCategory.TRANSIENT
        
        # Check for configuration errors
        config_keywords = ['config', 'setting', 'parameter', 'invalid', 'missing']
        if any(keyword in error_message for keyword in config_keywords):
            return ErrorCategory.CONFIGURATION
        
        # Check for data errors
        data_keywords = ['format', 'parse', 'decode', 'invalid data', 'corrupt']
        if any(keyword in error_message for keyword in data_keywords):
            return ErrorCategory.DATA
        
        # Check for resource errors
        resource_keywords = ['memory', 'disk', 'space', 'limit', 'quota', 'resource']
        if any(keyword in error_message for keyword in resource_keywords):
            return ErrorCategory.RESOURCE
        
        # Check for external service errors
        external_keywords = ['api', 'service', 'endpoint', 'http', 'request']
        if any(keyword in error_message for keyword in external_keywords):
            return ErrorCategory.EXTERNAL
        
        # Default to logic error
        return ErrorCategory.LOGIC
    
    def find_recovery_rule(self, error_context: ErrorContext) -> Optional[RecoveryRule]:
        """Find the best matching recovery rule"""
        for rule in self.recovery_rules:
            if rule.matches(error_context):
                return rule
        return None
    
    def execute_recovery(self, error_context: ErrorContext) -> tuple[bool, Optional[Any]]:
        """Execute recovery strategy for an error"""
        rule = self.find_recovery_rule(error_context)
        if not rule:
            self.logger.warning(f"No recovery rule found for error: {error_context.error_info.message}")
            return False, None
        
        attempt_id = f"recovery_{len(error_context.recovery_attempts) + 1}"
        attempt = RecoveryAttempt(
            attempt_id=attempt_id,
            timestamp=datetime.now(),
            strategy=rule.strategy,
            action=rule.action
        )
        
        start_time = datetime.now()
        
        try:
            success, result = self._execute_strategy(rule, error_context, attempt)
            attempt.success = success
            
            if not success and attempt.error_message is None:
                attempt.error_message = "Recovery strategy failed"
            
            return success, result
            
        except Exception as e:
            attempt.success = False
            attempt.error_message = str(e)
            self.logger.error(f"Recovery execution failed: {e}")
            return False, None
            
        finally:
            attempt.execution_time = (datetime.now() - start_time).total_seconds()
            error_context.add_recovery_attempt(attempt)
    
    def _execute_strategy(self, rule: RecoveryRule, 
                         error_context: ErrorContext, 
                         attempt: RecoveryAttempt) -> tuple[bool, Optional[Any]]:
        """Execute a specific recovery strategy"""
        
        if rule.strategy == RecoveryStrategy.RETRY:
            return self._execute_retry(rule, error_context, attempt)
        
        elif rule.strategy == RecoveryStrategy.SKIP:
            return self._execute_skip(rule, error_context, attempt)
        
        elif rule.strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback(rule, error_context, attempt)
        
        elif rule.strategy == RecoveryStrategy.ROLLBACK:
            return self._execute_rollback(rule, error_context, attempt)
        
        elif rule.strategy == RecoveryStrategy.MANUAL:
            return self._execute_manual(rule, error_context, attempt)
        
        elif rule.strategy == RecoveryStrategy.ABORT:
            return self._execute_abort(rule, error_context, attempt)
        
        else:
            attempt.error_message = f"Unknown recovery strategy: {rule.strategy}"
            return False, None
    
    def _execute_retry(self, rule: RecoveryRule, 
                      error_context: ErrorContext, 
                      attempt: RecoveryAttempt) -> tuple[bool, Optional[Any]]:
        """Execute retry strategy"""
        retry_count = error_context.get_retry_count(RecoveryStrategy.RETRY)
        
        if retry_count >= rule.max_attempts:
            attempt.error_message = f"Max retry attempts ({rule.max_attempts}) exceeded"
            return False, None
        
        # Calculate delay
        delay = min(
            rule.delay_seconds * (rule.backoff_multiplier ** retry_count),
            rule.max_delay_seconds
        )
        
        attempt.retry_count = retry_count + 1
        attempt.delay_applied = delay
        
        if rule.action == RecoveryAction.DELAYED and delay > 0:
            import time
            time.sleep(delay)
        
        # For retry, we return success=True to indicate the retry should be attempted
        # The actual retry logic is handled by the calling workflow
        return True, {'retry_count': attempt.retry_count, 'delay': delay}
    
    def _execute_skip(self, rule: RecoveryRule, 
                     error_context: ErrorContext, 
                     attempt: RecoveryAttempt) -> tuple[bool, Optional[Any]]:
        """Execute skip strategy"""
        self.logger.info(f"Skipping failed operation: {error_context.error_info.message}")
        return True, {'skipped': True}
    
    def _execute_fallback(self, rule: RecoveryRule, 
                         error_context: ErrorContext, 
                         attempt: RecoveryAttempt) -> tuple[bool, Optional[Any]]:
        """Execute fallback strategy"""
        fallback_handler = self.fallback_handlers.get(error_context.processing_stage)
        
        if not fallback_handler:
            attempt.error_message = f"No fallback handler for stage: {error_context.processing_stage}"
            return False, None
        
        try:
            result = fallback_handler(error_context)
            attempt.fallback_used = fallback_handler.__name__
            return True, result
            
        except Exception as e:
            attempt.error_message = f"Fallback handler failed: {str(e)}"
            return False, None
    
    def _execute_rollback(self, rule: RecoveryRule, 
                         error_context: ErrorContext, 
                         attempt: RecoveryAttempt) -> tuple[bool, Optional[Any]]:
        """Execute rollback strategy"""
        # Find the most recent rollback point
        if not self.rollback_points:
            attempt.error_message = "No rollback points available"
            return False, None
        
        # Get the latest rollback point
        latest_point = max(self.rollback_points.keys())
        rollback_state = self.rollback_points[latest_point]
        
        attempt.rollback_point = latest_point
        
        return True, {'rollback_state': rollback_state, 'rollback_point': latest_point}
    
    def _execute_manual(self, rule: RecoveryRule, 
                       error_context: ErrorContext, 
                       attempt: RecoveryAttempt) -> tuple[bool, Optional[Any]]:
        """Execute manual intervention strategy"""
        self.logger.critical(f"Manual intervention required: {error_context.error_info.message}")
        
        # In a real implementation, this might trigger notifications, 
        # create tickets, or pause the workflow
        return False, {
            'manual_intervention_required': True,
            'error_context': error_context.to_dict()
        }
    
    def _execute_abort(self, rule: RecoveryRule, 
                      error_context: ErrorContext, 
                      attempt: RecoveryAttempt) -> tuple[bool, Optional[Any]]:
        """Execute abort strategy"""
        self.logger.critical(f"Aborting workflow due to critical error: {error_context.error_info.message}")
        
        return False, {
            'workflow_aborted': True,
            'abort_reason': error_context.error_info.message
        }
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        # This would typically aggregate statistics from all error contexts
        # For now, return basic structure
        return {
            'total_rules': len(self.recovery_rules),
            'fallback_handlers': len(self.fallback_handlers),
            'rollback_points': len(self.rollback_points),
            'rules_by_category': {
                category.value: len([r for r in self.recovery_rules if r.error_category == category])
                for category in ErrorCategory
            },
            'rules_by_strategy': {
                strategy.value: len([r for r in self.recovery_rules if r.strategy == strategy])
                for strategy in RecoveryStrategy
            }
        }


# Utility functions for common error scenarios

def create_error_context(error: Exception, 
                        processing_stage: ProcessingStage,
                        current_state: Optional[Dict[str, Any]] = None) -> ErrorContext:
    """Create error context from exception"""
    error_info = ErrorInfo(
        error_type=type(error).__name__,
        message=str(error),
        traceback=traceback.format_exc(),
        timestamp=datetime.now(),
        severity=ErrorSeverity.ERROR  # Default severity
    )
    
    recovery_manager = ErrorRecoveryManager()
    category = recovery_manager.categorize_error(error_info, processing_stage)
    
    return ErrorContext(
        error_info=error_info,
        processing_stage=processing_stage,
        category=category,
        current_state=current_state or {}
    )


def handle_error_with_recovery(error: Exception,
                              processing_stage: ProcessingStage,
                              current_state: Optional[Dict[str, Any]] = None,
                              recovery_manager: Optional[ErrorRecoveryManager] = None) -> tuple[bool, Optional[Any]]:
    """Handle error with automatic recovery"""
    error_context = create_error_context(error, processing_stage, current_state)
    
    if recovery_manager is None:
        recovery_manager = ErrorRecoveryManager()
    
    return recovery_manager.execute_recovery(error_context)


# Global recovery manager instance
_global_recovery_manager = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get global recovery manager instance"""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager()
    return _global_recovery_manager