"""Modern Centralized State Management System for LangGraph Document Processing

This module provides a comprehensive, type-safe state management system that replaces
the fragmented state handling across multiple modules with a unified, observable,
and performance-optimized approach.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from uuid import uuid4, UUID
from contextlib import asynccontextmanager
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor
from threading import RLock

# Type definitions
T = TypeVar('T')
StateObserver = Callable[[str, Any, Any], None]  # (key, old_value, new_value)
StateValidator = Callable[[Any], bool]

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Enhanced processing stages with better granularity."""
    INITIALIZATION = auto()
    DOCUMENT_PARSING = auto()
    CHUNKING = auto()
    KNOWLEDGE_GRAPH_BUILDING = auto()
    SEMANTIC_MAPPING = auto()
    LLM_PROCESSING = auto()
    OUTPUT_GENERATION = auto()
    QUALITY_VALIDATION = auto()
    ANALYTICS_COLLECTION = auto()
    COMPLETED = auto()
    ERROR = auto()
    CANCELLED = auto()


class StateChangeType(Enum):
    """Types of state changes for observability."""
    CREATE = auto()
    UPDATE = auto()
    DELETE = auto()
    BATCH_UPDATE = auto()


class ErrorSeverity(Enum):
    """Error severity levels with clear escalation paths."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    FATAL = auto()


@dataclass(frozen=True)
class StateChange:
    """Immutable record of a state change for audit and rollback."""
    timestamp: datetime
    change_type: StateChangeType
    key: str
    old_value: Any
    new_value: Any
    stage: ProcessingStage
    session_id: str
    change_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ProcessingMetrics:
    """Comprehensive metrics for performance monitoring."""
    stage_start_times: Dict[ProcessingStage, datetime] = field(default_factory=dict)
    stage_end_times: Dict[ProcessingStage, datetime] = field(default_factory=dict)
    stage_durations: Dict[ProcessingStage, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    api_calls: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    quality_scores: Dict[str, float] = field(default_factory=dict)
    throughput_metrics: Dict[str, float] = field(default_factory=dict)

    def record_stage_start(self, stage: ProcessingStage) -> None:
        """Record the start time of a processing stage."""
        self.stage_start_times[stage] = datetime.now(timezone.utc)

    def record_stage_end(self, stage: ProcessingStage) -> None:
        """Record the end time and calculate duration of a processing stage."""
        end_time = datetime.now(timezone.utc)
        self.stage_end_times[stage] = end_time
        
        if stage in self.stage_start_times:
            duration = (end_time - self.stage_start_times[stage]).total_seconds()
            self.stage_durations[stage] = duration

    def get_total_processing_time(self) -> float:
        """Calculate total processing time across all stages."""
        return sum(self.stage_durations.values())


@dataclass
class ErrorContext:
    """Rich error context for better debugging and recovery."""
    error_id: str = field(default_factory=lambda: str(uuid4()))
    stage: ProcessingStage = ProcessingStage.ERROR
    severity: ErrorSeverity = ErrorSeverity.ERROR
    message: str = ""
    exception_type: str = ""
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int = 0
    max_retries: int = 3
    recoverable: bool = True
    recovery_strategy: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    affected_chunks: List[str] = field(default_factory=list)


class ObservableDict(Generic[T]):
    """Thread-safe observable dictionary with validation and change tracking."""
    
    def __init__(self, initial_data: Optional[Dict[str, T]] = None):
        self._data: Dict[str, T] = initial_data or {}
        self._observers: List[StateObserver] = []
        self._validators: Dict[str, StateValidator] = {}
        self._lock = RLock()
        self._change_history: List[StateChange] = []
        self._session_id = str(uuid4())
        self._current_stage = ProcessingStage.INITIALIZATION

    def add_observer(self, observer: StateObserver) -> None:
        """Add a state change observer."""
        with self._lock:
            self._observers.append(observer)

    def add_validator(self, key: str, validator: StateValidator) -> None:
        """Add a validator for a specific key."""
        with self._lock:
            self._validators[key] = validator

    def set_stage(self, stage: ProcessingStage) -> None:
        """Set the current processing stage."""
        with self._lock:
            self._current_stage = stage

    def __getitem__(self, key: str) -> T:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: T) -> None:
        with self._lock:
            # Validate if validator exists
            if key in self._validators and not self._validators[key](value):
                raise ValueError(f"Validation failed for key '{key}' with value: {value}")
            
            old_value = self._data.get(key)
            self._data[key] = value
            
            # Record change
            change = StateChange(
                timestamp=datetime.now(timezone.utc),
                change_type=StateChangeType.UPDATE if old_value is not None else StateChangeType.CREATE,
                key=key,
                old_value=old_value,
                new_value=value,
                stage=self._current_stage,
                session_id=self._session_id
            )
            self._change_history.append(change)
            
            # Notify observers
            for observer in self._observers:
                try:
                    observer(key, old_value, value)
                except Exception as e:
                    logger.warning(f"Observer failed for key '{key}': {e}")

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        with self._lock:
            return self._data.get(key, default)

    def update(self, data: Dict[str, T]) -> None:
        """Batch update with single change record."""
        with self._lock:
            old_data = self._data.copy()
            self._data.update(data)
            
            change = StateChange(
                timestamp=datetime.now(timezone.utc),
                change_type=StateChangeType.BATCH_UPDATE,
                key="batch_update",
                old_value=old_data,
                new_value=self._data.copy(),
                stage=self._current_stage,
                session_id=self._session_id
            )
            self._change_history.append(change)

    def get_change_history(self) -> List[StateChange]:
        """Get the complete change history."""
        with self._lock:
            return self._change_history.copy()

    def to_dict(self) -> Dict[str, T]:
        """Get a copy of the internal data."""
        with self._lock:
            return self._data.copy()


class CentralizedStateManager:
    """Modern centralized state manager with observability and performance optimization."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid4())
        self.created_at = datetime.now(timezone.utc)
        
        # Core state containers
        self.document_state = ObservableDict[Any]()
        self.processing_state = ObservableDict[Any]()
        self.chunk_state = ObservableDict[Any]()
        self.knowledge_graph_state = ObservableDict[Any]()
        self.llm_state = ObservableDict[Any]()
        self.output_state = ObservableDict[Any]()
        
        # System state
        self.current_stage = ProcessingStage.INITIALIZATION
        self.progress = 0.0
        self.is_cancelled = False
        self.errors: List[ErrorContext] = []
        self.metrics = ProcessingMetrics()
        
        # Configuration and metadata
        self.config: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="StateManager")
        
        # Setup observers
        self._setup_observers()
        
        logger.info(f"Initialized CentralizedStateManager with session_id: {self.session_id}")

    def _setup_observers(self) -> None:
        """Setup default observers for state changes."""
        def log_state_change(key: str, old_value: Any, new_value: Any) -> None:
            logger.debug(f"State change - {key}: {type(old_value).__name__} -> {type(new_value).__name__}")
        
        def update_progress(key: str, old_value: Any, new_value: Any) -> None:
            if key == "current_stage" and isinstance(new_value, ProcessingStage):
                # Auto-update progress based on stage
                stage_progress = {
                    ProcessingStage.INITIALIZATION: 0.1,
                    ProcessingStage.DOCUMENT_PARSING: 0.2,
                    ProcessingStage.CHUNKING: 0.3,
                    ProcessingStage.KNOWLEDGE_GRAPH_BUILDING: 0.4,
                    ProcessingStage.SEMANTIC_MAPPING: 0.5,
                    ProcessingStage.LLM_PROCESSING: 0.7,
                    ProcessingStage.OUTPUT_GENERATION: 0.9,
                    ProcessingStage.COMPLETED: 1.0
                }
                self.progress = stage_progress.get(new_value, self.progress)
        
        # Add observers to all state containers
        for state_container in [self.document_state, self.processing_state, 
                               self.chunk_state, self.knowledge_graph_state,
                               self.llm_state, self.output_state]:
            state_container.add_observer(log_state_change)
            state_container.add_observer(update_progress)

    @asynccontextmanager
    async def stage_context(self, stage: ProcessingStage):
        """Context manager for stage transitions with automatic metrics."""
        logger.info(f"Entering stage: {stage.name}")
        self.current_stage = stage
        self.metrics.record_stage_start(stage)
        
        # Update all state containers with current stage
        for state_container in [self.document_state, self.processing_state,
                               self.chunk_state, self.knowledge_graph_state,
                               self.llm_state, self.output_state]:
            state_container.set_stage(stage)
        
        try:
            yield self
        except Exception as e:
            self.add_error(ErrorContext(
                stage=stage,
                severity=ErrorSeverity.ERROR,
                message=str(e),
                exception_type=type(e).__name__
            ))
            raise
        finally:
            self.metrics.record_stage_end(stage)
            logger.info(f"Completed stage: {stage.name} in {self.metrics.stage_durations.get(stage, 0):.2f}s")

    def add_error(self, error: ErrorContext) -> None:
        """Add an error to the state with automatic severity handling."""
        self.errors.append(error)
        self.metrics.error_counts[error.severity] += 1
        
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            logger.error(f"Critical error in {error.stage.name}: {error.message}")
        
        # Auto-transition to error state for fatal errors
        if error.severity == ErrorSeverity.FATAL:
            self.current_stage = ProcessingStage.ERROR

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a complete snapshot of the current state."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "current_stage": self.current_stage.name,
            "progress": self.progress,
            "is_cancelled": self.is_cancelled,
            "document_state": self.document_state.to_dict(),
            "processing_state": self.processing_state.to_dict(),
            "chunk_state": self.chunk_state.to_dict(),
            "knowledge_graph_state": self.knowledge_graph_state.to_dict(),
            "llm_state": self.llm_state.to_dict(),
            "output_state": self.output_state.to_dict(),
            "config": self.config,
            "metadata": self.metadata,
            "metrics": asdict(self.metrics),
            "error_count": len(self.errors),
            "has_critical_errors": any(e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL] for e in self.errors)
        }

    async def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Save state checkpoint asynchronously."""
        def _save():
            snapshot = self.get_state_snapshot()
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, default=str)
        
        await asyncio.get_event_loop().run_in_executor(self._executor, _save)
        logger.info(f"State checkpoint saved to {checkpoint_path}")

    @classmethod
    async def load_checkpoint(cls, checkpoint_path: Path) -> 'CentralizedStateManager':
        """Load state from checkpoint asynchronously."""
        def _load():
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        loop = asyncio.get_event_loop()
        snapshot = await loop.run_in_executor(None, _load)
        
        # Reconstruct state manager
        state_manager = cls(session_id=snapshot["session_id"])
        state_manager.created_at = datetime.fromisoformat(snapshot["created_at"])
        state_manager.current_stage = ProcessingStage[snapshot["current_stage"]]
        state_manager.progress = snapshot["progress"]
        state_manager.is_cancelled = snapshot["is_cancelled"]
        state_manager.config = snapshot["config"]
        state_manager.metadata = snapshot["metadata"]
        
        # Restore state containers
        state_manager.document_state.update(snapshot["document_state"])
        state_manager.processing_state.update(snapshot["processing_state"])
        state_manager.chunk_state.update(snapshot["chunk_state"])
        state_manager.knowledge_graph_state.update(snapshot["knowledge_graph_state"])
        state_manager.llm_state.update(snapshot["llm_state"])
        state_manager.output_state.update(snapshot["output_state"])
        
        logger.info(f"State checkpoint loaded from {checkpoint_path}")
        return state_manager

    def cancel_processing(self) -> None:
        """Cancel the current processing session."""
        self.is_cancelled = True
        self.current_stage = ProcessingStage.CANCELLED
        logger.info(f"Processing cancelled for session {self.session_id}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._executor.shutdown(wait=True)
        logger.info(f"StateManager cleanup completed for session {self.session_id}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()