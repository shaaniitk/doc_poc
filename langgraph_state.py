"""LangGraph State Schema for Document Processing Pipeline

This module defines the comprehensive state schema for the LangGraph-based
document processing pipeline, including all stages from input to output.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class ProcessingStage(Enum):
    """Enumeration of processing stages in the pipeline."""
    INITIALIZATION = "initialization"
    DOCUMENT_PARSING = "document_parsing"
    CHUNKING = "chunking"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    SEMANTIC_MAPPING = "semantic_mapping"
    LLM_PROCESSING = "llm_processing"
    OUTPUT_GENERATION = "output_generation"
    ANALYTICS = "analytics"
    COMPLETED = "completed"
    ERROR = "error"


class ErrorSeverity(Enum):
    """Error severity levels for error handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Information about errors that occur during processing."""
    stage: ProcessingStage
    severity: ErrorSeverity
    message: str
    exception_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    traceback: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    recoverable: bool = True


@dataclass
class DocumentInfo:
    """Information about the input document."""
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    encoding: str = "utf-8"
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkInfo:
    """Information about a document chunk."""
    chunk_id: str
    content: str
    start_position: int
    end_position: int
    chunk_size: int
    overlap_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    semantic_tags: List[str] = field(default_factory=list)


@dataclass
class KnowledgeGraphNode:
    """Represents a node in the knowledge graph."""
    node_id: str
    content: str
    node_type: str
    confidence: float
    relationships: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraphEdge:
    """Represents an edge in the knowledge graph."""
    edge_id: str
    source_node: str
    target_node: str
    relationship_type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticMapping:
    """Semantic mapping information for chunks."""
    chunk_id: str
    semantic_category: str
    confidence: float
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)


@dataclass
class LLMProcessingResult:
    """Result from LLM processing of a chunk."""
    chunk_id: str
    processed_content: str
    analysis: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    processing_time: float = 0.0
    model_used: str = ""
    tokens_used: int = 0


@dataclass
class QualityMetrics:
    """Quality metrics for processed content."""
    coherence_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    relevance_score: float = 0.0
    overall_quality: float = 0.0
    quality_gates_passed: List[str] = field(default_factory=list)
    quality_gates_failed: List[str] = field(default_factory=list)


@dataclass
class AnalyticsData:
    """Analytics and performance data."""
    processing_start_time: datetime = field(default_factory=datetime.now)
    processing_end_time: Optional[datetime] = None
    total_processing_time: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    performance_bottlenecks: List[str] = field(default_factory=list)


@dataclass
class OutputData:
    """Final output data structure."""
    output_format: str
    content: Union[str, Dict[str, Any], List[Any]]
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.0


@dataclass
class SessionInfo:
    """Session management information."""
    session_id: str
    user_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    status: str = "active"


@dataclass
class PipelineState:
    """Comprehensive state for the LangGraph document processing pipeline."""
    
    # Core identification and session management
    session_info: SessionInfo
    
    # Current processing stage and progress
    current_stage: ProcessingStage = ProcessingStage.INITIALIZATION
    progress: float = 0.0
    
    # Document and input information
    document_info: Optional[DocumentInfo] = None
    raw_content: str = ""
    
    # Processing configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Chunking stage data
    chunks: List[ChunkInfo] = field(default_factory=list)
    chunk_strategy: str = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Knowledge graph data
    knowledge_graph_nodes: List[KnowledgeGraphNode] = field(default_factory=list)
    knowledge_graph_edges: List[KnowledgeGraphEdge] = field(default_factory=list)
    
    # Semantic mapping data
    semantic_mappings: List[SemanticMapping] = field(default_factory=list)
    
    # LLM processing results
    llm_results: List[LLMProcessingResult] = field(default_factory=list)
    
    # Output data
    outputs: List[OutputData] = field(default_factory=list)
    
    # Error handling
    errors: List[ErrorInfo] = field(default_factory=list)
    has_errors: bool = False
    is_recoverable: bool = True
    
    # Analytics and performance
    analytics: AnalyticsData = field(default_factory=AnalyticsData)
    
    # Human-in-the-loop data
    human_feedback: Dict[str, Any] = field(default_factory=dict)
    requires_human_input: bool = False
    human_input_prompt: str = ""
    
    # Conditional routing data
    routing_decisions: Dict[str, str] = field(default_factory=dict)
    next_stage_override: Optional[ProcessingStage] = None
    
    # Quality gates
    quality_gates: Dict[str, bool] = field(default_factory=dict)
    quality_threshold: float = 0.7
    
    # Retry and recovery
    retry_count: int = 0
    max_retries: int = 3
    recovery_actions: List[str] = field(default_factory=list)
    
    # Intermediate results for debugging
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    debug_mode: bool = False
    
    def add_error(self, error: ErrorInfo) -> None:
        """Add an error to the state."""
        self.errors.append(error)
        self.has_errors = True
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.is_recoverable = error.recoverable
    
    def clear_errors(self) -> None:
        """Clear all errors from the state."""
        self.errors.clear()
        self.has_errors = False
        self.is_recoverable = True
    
    def update_progress(self, stage: ProcessingStage, progress: float) -> None:
        """Update the current stage and progress."""
        self.current_stage = stage
        self.progress = progress
    
    def add_chunk(self, chunk: ChunkInfo) -> None:
        """Add a chunk to the state."""
        self.chunks.append(chunk)
    
    def add_llm_result(self, result: LLMProcessingResult) -> None:
        """Add an LLM processing result."""
        self.llm_results.append(result)
    
    def add_output(self, output: OutputData) -> None:
        """Add output data."""
        self.outputs.append(output)
    
    def get_chunks_by_stage(self, stage: str) -> List[ChunkInfo]:
        """Get chunks filtered by processing stage."""
        return [chunk for chunk in self.chunks if chunk.metadata.get('stage') == stage]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorInfo]:
        """Get errors filtered by severity."""
        return [error for error in self.errors if error.severity == severity]
    
    def should_retry(self) -> bool:
        """Check if the pipeline should retry after an error."""
        return self.retry_count < self.max_retries and self.is_recoverable
    
    def increment_retry(self) -> None:
        """Increment the retry counter."""
        self.retry_count += 1
    
    def set_quality_gate(self, gate_name: str, passed: bool) -> None:
        """Set the status of a quality gate."""
        self.quality_gates[gate_name] = passed
    
    def all_quality_gates_passed(self) -> bool:
        """Check if all quality gates have passed."""
        return all(self.quality_gates.values()) if self.quality_gates else True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'session_info': self.session_info.__dict__,
            'current_stage': self.current_stage.value,
            'progress': self.progress,
            'document_info': self.document_info.__dict__ if self.document_info else None,
            'config': self.config,
            'chunks_count': len(self.chunks),
            'errors_count': len(self.errors),
            'has_errors': self.has_errors,
            'analytics': self.analytics.__dict__,
            'quality_gates': self.quality_gates,
            'retry_count': self.retry_count
        }


# Type aliases for LangGraph integration
StateDict = Dict[str, Any]
NodeFunction = callable
ConditionalFunction = callable


# Helper functions for state management
def create_initial_state(session_id: str, document_path: str, config: Dict[str, Any]) -> PipelineState:
    """Create an initial pipeline state."""
    session_info = SessionInfo(
        session_id=session_id,
        configuration=config
    )
    
    document_info = DocumentInfo(
        file_path=document_path,
        file_name=Path(document_path).name,
        file_size=Path(document_path).stat().st_size if Path(document_path).exists() else 0,
        file_type=Path(document_path).suffix
    )
    
    return PipelineState(
        session_info=session_info,
        document_info=document_info,
        config=config
    )


def serialize_state(state: PipelineState) -> Dict[str, Any]:
    """Serialize state for persistence."""
    return state.to_dict()


def deserialize_state(state_dict: Dict[str, Any]) -> PipelineState:
    """Deserialize state from dictionary."""
    # This would need proper implementation based on the state_dict structure
    # For now, return a basic state
    session_info = SessionInfo(session_id=state_dict.get('session_id', 'unknown'))
    return PipelineState(session_info=session_info)