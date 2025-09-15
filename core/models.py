from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class DocumentFormat(Enum):
    """Supported document formats."""
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    PDF = "pdf"

class ProcessingStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    title: str
    format: DocumentFormat
    creation_time: datetime
    modification_time: datetime
    author: Optional[str] = None
    file_size: Optional[int] = None
    language: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content."""
    content: str
    chunk_id: str
    start_position: int
    end_position: int
    metadata: Optional[Dict[str, Any]] = None
    semantic_tags: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    parent_section: Optional[str] = None

@dataclass
class ProcessedSection:
    """Represents a processed document section with template-based processing."""
    section_name: str
    content: str
    confidence_score: float
    source_chunks: List[DocumentChunk]
    subsections: Optional[List['ProcessedSection']] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    template_applied: Optional[str] = None
    prompt_used: Optional[str] = None

@dataclass
class SemanticMapping:
    """Represents semantic mapping between content and concepts."""
    source_content: str
    target_concept: str
    similarity_score: float
    mapping_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkingResult:
    """Result of document chunking operation."""
    chunks: List[DocumentChunk]
    total_chunks: int
    chunking_strategy: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingResult:
    """Result of document processing operation."""
    status: ProcessingStatus
    processed_content: Optional[str] = None
    processed_sections: List[ProcessedSection] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of content validation."""
    is_valid: bool
    validation_score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validator_used: Optional[str] = None

@dataclass
class OutputSection:
    """Represents a section in the final output document."""
    title: str
    content: str
    section_type: str
    order: int
    subsections: List['OutputSection'] = field(default_factory=list)
    formatting_hints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentTemplate:
    """Template configuration for document processing."""
    name: str
    description: str
    sections: Dict[str, Dict[str, Any]]
    output_format: DocumentFormat
    processing_rules: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowState:
    """State object for LangGraph workflow."""
    document_id: str
    document_path: str
    original_content: str
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = field(default_factory=list)
    processed_sections: List[ProcessedSection] = field(default_factory=list)
    document_content: Optional[Any] = None
    processing_config: Dict[str, Any] = field(default_factory=dict)
    llm_client: Optional[Any] = None
    llm_handler: Optional[Any] = None
    final_output: Optional[str] = None
    template_name: Optional[str] = None
    template_applied: Optional[str] = None
    processing_errors: List[str] = field(default_factory=list)
    processing_warnings: List[str] = field(default_factory=list)
    current_stage: str = "initialization"
    stage_results: Dict[str, Any] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class PromptApplication:
    """Tracks the application of prompts during processing."""
    prompt_name: str
    prompt_content: str
    applied_to: str  # section name or content identifier
    application_time: datetime
    result_quality: Optional[float] = None
    processing_notes: List[str] = field(default_factory=list)

@dataclass
class QualityMetrics:
    """Quality metrics for processed content."""
    coherence_score: float
    completeness_score: float
    accuracy_score: float
    readability_score: float
    overall_score: float
    detailed_metrics: Dict[str, float] = field(default_factory=dict)
    assessment_notes: List[str] = field(default_factory=list)