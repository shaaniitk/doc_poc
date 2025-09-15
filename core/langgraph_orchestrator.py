"""LangGraph Workflow Orchestrator for Document Processing

This module implements a modern, graph-based workflow orchestrator that replaces
the linear pipeline approach with a flexible, robust, and observable system.
It leverages LangGraph's state management and node orchestration capabilities
to create a production-ready document processing platform.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, TypedDict, Annotated
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum, auto

# LangGraph imports with correct paths
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Internal imports
from .state_manager import (
    CentralizedStateManager, ProcessingStage, ErrorContext, ErrorSeverity,
    ProcessingMetrics
)

logger = logging.getLogger(__name__)


class NodeResult(Enum):
    """Standardized node execution results."""
    SUCCESS = "success"
    RETRY = "retry"
    SKIP = "skip"
    ERROR = "error"
    FATAL = "fatal"


class WorkflowState(TypedDict):
    """Enhanced workflow state schema for LangGraph."""
    # Core processing data
    document_content: str
    document_path: str
    chunks: List[Dict[str, Any]]
    knowledge_graph: Dict[str, Any]
    semantic_mappings: List[Dict[str, Any]]
    llm_outputs: List[Dict[str, Any]]
    final_output: Dict[str, Any]
    
    # Processing metadata
    current_stage: str
    progress: float
    session_id: str
    processing_config: Dict[str, Any]
    
    # Error handling
    errors: List[Dict[str, Any]]
    retry_counts: Dict[str, int]
    max_retries: Dict[str, int]
    
    # Performance metrics
    stage_timings: Dict[str, float]
    memory_usage: Dict[str, float]
    token_usage: Dict[str, int]
    
    # Quality metrics
    quality_scores: Dict[str, float]
    validation_results: Dict[str, bool]
    
    # Control flow
    next_node: Optional[str]
    should_continue: bool
    is_cancelled: bool


@dataclass
class NodeConfig:
    """Configuration for workflow nodes."""
    name: str
    max_retries: int = 3
    timeout_seconds: float = 300.0
    memory_limit_mb: Optional[int] = None
    required_inputs: List[str] = field(default_factory=list)
    optional_inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    parallel_execution: bool = False
    error_recovery_strategy: str = "retry"  # retry, skip, fail


class BaseWorkflowNode:
    """Base class for all workflow nodes with standardized error handling and metrics."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        self.config = config
        self.state_manager = state_manager
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the node with comprehensive error handling and metrics."""
        start_time = datetime.now(timezone.utc)
        node_name = self.config.name
        
        try:
            # Validate inputs
            self._validate_inputs(state)
            
            # Check cancellation
            if state.get("is_cancelled", False):
                self.logger.info(f"Node {node_name} skipped due to cancellation")
                return state
            
            # Execute the actual node logic
            self.logger.info(f"Executing node: {node_name}")
            result_state = await self._execute_impl(state)
            
            # Record success metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            result_state["stage_timings"][node_name] = duration
            
            self.logger.info(f"Node {node_name} completed successfully in {duration:.2f}s")
            return result_state
            
        except Exception as e:
            # Handle errors with retry logic
            return await self._handle_error(state, e, start_time)
    
    def _validate_inputs(self, state: WorkflowState) -> None:
        """Validate required inputs are present."""
        for required_input in self.config.required_inputs:
            if required_input not in state or state[required_input] is None:
                raise ValueError(f"Required input '{required_input}' missing for node {self.config.name}")
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Override this method in subclasses to implement node logic."""
        raise NotImplementedError(f"Node {self.config.name} must implement _execute_impl")
    
    async def _handle_error(self, state: WorkflowState, error: Exception, start_time: datetime) -> WorkflowState:
        """Handle errors with configurable retry and recovery strategies."""
        node_name = self.config.name
        retry_count = state.get("retry_counts", {}).get(node_name, 0)
        max_retries = self.config.max_retries
        
        # Record error
        error_context = ErrorContext(
            stage=ProcessingStage[state["current_stage"]],
            severity=ErrorSeverity.ERROR,
            message=str(error),
            exception_type=type(error).__name__,
            retry_count=retry_count,
            max_retries=max_retries,
            recoverable=retry_count < max_retries
        )
        
        self.state_manager.add_error(error_context)
        
        # Update state with error info
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({
            "node": node_name,
            "error": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retry_count": retry_count
        })
        
        # Apply recovery strategy
        if retry_count < max_retries and self.config.error_recovery_strategy == "retry":
            # Increment retry count and retry
            if "retry_counts" not in state:
                state["retry_counts"] = {}
            state["retry_counts"][node_name] = retry_count + 1
            
            self.logger.warning(f"Node {node_name} failed, retrying ({retry_count + 1}/{max_retries}): {error}")
            await asyncio.sleep(2 ** retry_count)  # Exponential backoff
            return await self.execute(state)
            
        elif self.config.error_recovery_strategy == "skip":
            # Skip this node and continue
            self.logger.warning(f"Node {node_name} failed, skipping: {error}")
            return state
            
        else:
            # Fail the entire workflow
            self.logger.error(f"Node {node_name} failed fatally: {error}")
            state["should_continue"] = False
            raise error


class DocumentParsingNode(BaseWorkflowNode):
    """Enhanced document parsing with multiple format support."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Parse document with enhanced error handling and format detection."""
        document_path = Path(state["document_path"])
        
        # Import parsing modules dynamically to avoid circular imports
        from ..modules.document_parser import DocumentParser
        
        parser = DocumentParser()
        
        try:
            # Parse document with format auto-detection
            content = await asyncio.get_event_loop().run_in_executor(
                None, parser.parse_document, document_path
            )
            
            state["document_content"] = content
            state["current_stage"] = ProcessingStage.DOCUMENT_PARSING.name
            
            # Store parsing metadata
            self.state_manager.document_state["original_path"] = str(document_path)
            self.state_manager.document_state["file_size"] = document_path.stat().st_size
            self.state_manager.document_state["content_length"] = len(content)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Document parsing failed for {document_path}: {e}")
            raise


class IntelligentChunkingNode(BaseWorkflowNode):
    """Advanced chunking with semantic awareness and optimization."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Perform intelligent chunking with quality validation."""
        content = state["document_content"]
        config = state["processing_config"]
        
        # Import chunking module
        from ..modules.chunker import Chunker
        
        chunker = Chunker(
            chunk_size=config.get("chunk_size", 1000),
            overlap=config.get("chunk_overlap", 200),
            quality_threshold=config.get("chunk_quality_threshold", 0.7)
        )
        
        try:
            # Perform chunking with quality assessment
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, chunker.chunk_document, content
            )
            
            # Validate chunk quality
            quality_scores = [chunk.get("quality_score", 0.0) for chunk in chunks]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            state["chunks"] = chunks
            state["current_stage"] = ProcessingStage.CHUNKING.name
            state["quality_scores"]["chunking"] = avg_quality
            
            # Store chunking metrics
            self.state_manager.chunk_state["total_chunks"] = len(chunks)
            self.state_manager.chunk_state["average_quality"] = avg_quality
            self.state_manager.chunk_state["quality_distribution"] = quality_scores
            
            self.logger.info(f"Created {len(chunks)} chunks with average quality {avg_quality:.3f}")
            return state
            
        except Exception as e:
            self.logger.error(f"Chunking failed: {e}")
            raise


class KnowledgeGraphNode(BaseWorkflowNode):
    """Knowledge graph construction with relationship extraction."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Build knowledge graph from chunks."""
        chunks = state["chunks"]
        
        # Import knowledge graph module
        from ..modules.knowledge_graph_processor import KnowledgeGraphProcessor
        
        kg_processor = KnowledgeGraphProcessor()
        
        try:
            # Build knowledge graph
            knowledge_graph = await asyncio.get_event_loop().run_in_executor(
                None, kg_processor.build_graph, chunks
            )
            
            state["knowledge_graph"] = knowledge_graph
            state["current_stage"] = ProcessingStage.KNOWLEDGE_GRAPH_BUILDING.name
            
            # Store KG metrics
            node_count = len(knowledge_graph.get("nodes", []))
            edge_count = len(knowledge_graph.get("edges", []))
            
            self.state_manager.knowledge_graph_state["node_count"] = node_count
            self.state_manager.knowledge_graph_state["edge_count"] = edge_count
            self.state_manager.knowledge_graph_state["density"] = edge_count / max(node_count, 1)
            
            self.logger.info(f"Built knowledge graph with {node_count} nodes and {edge_count} edges")
            return state
            
        except Exception as e:
            self.logger.error(f"Knowledge graph construction failed: {e}")
            raise


class LLMProcessingNode(BaseWorkflowNode):
    """LLM processing with intelligent batching and error recovery."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Process chunks through LLM with batching and rate limiting."""
        chunks = state["chunks"]
        config = state["processing_config"]
        
        # Import LLM handler
        from ..modules.llm_handler import LLMHandler
        
        llm_handler = LLMHandler(
            batch_size=config.get("llm_batch_size", 5),
            rate_limit=config.get("llm_rate_limit", 10)
        )
        
        try:
            # Process chunks in batches
            llm_outputs = []
            total_tokens = 0
            
            for i in range(0, len(chunks), llm_handler.batch_size):
                batch = chunks[i:i + llm_handler.batch_size]
                
                # Check for cancellation
                if state.get("is_cancelled", False):
                    break
                
                batch_results = await llm_handler.process_batch(batch)
                llm_outputs.extend(batch_results)
                
                # Track token usage
                batch_tokens = sum(result.get("token_count", 0) for result in batch_results)
                total_tokens += batch_tokens
                
                # Update progress
                progress = min(0.7 + 0.2 * (i + len(batch)) / len(chunks), 0.9)
                state["progress"] = progress
                
                self.logger.info(f"Processed batch {i//llm_handler.batch_size + 1}, tokens: {batch_tokens}")
            
            state["llm_outputs"] = llm_outputs
            state["current_stage"] = ProcessingStage.LLM_PROCESSING.name
            state["token_usage"]["total"] = total_tokens
            
            # Store LLM metrics
            self.state_manager.llm_state["total_tokens"] = total_tokens
            self.state_manager.llm_state["processed_chunks"] = len(llm_outputs)
            self.state_manager.llm_state["average_tokens_per_chunk"] = total_tokens / max(len(llm_outputs), 1)
            
            return state
            
        except Exception as e:
            self.logger.error(f"LLM processing failed: {e}")
            raise


class OutputGenerationNode(BaseWorkflowNode):
    """Final output generation with quality validation."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Generate final output with comprehensive validation."""
        llm_outputs = state["llm_outputs"]
        knowledge_graph = state["knowledge_graph"]
        config = state["processing_config"]
        
        try:
            # Generate final output
            final_output = {
                "document_path": state["document_path"],
                "session_id": state["session_id"],
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "chunks_processed": len(llm_outputs),
                "knowledge_graph_summary": {
                    "nodes": len(knowledge_graph.get("nodes", [])),
                    "edges": len(knowledge_graph.get("edges", []))
                },
                "llm_outputs": llm_outputs,
                "quality_metrics": state["quality_scores"],
                "processing_metrics": {
                    "total_time": sum(state["stage_timings"].values()),
                    "token_usage": state["token_usage"],
                    "memory_usage": state["memory_usage"]
                }
            }
            
            # Validate output quality
            quality_score = self._calculate_output_quality(final_output)
            final_output["overall_quality_score"] = quality_score
            
            state["final_output"] = final_output
            state["current_stage"] = ProcessingStage.OUTPUT_GENERATION.name
            state["progress"] = 0.95
            
            # Store output metrics
            self.state_manager.output_state["quality_score"] = quality_score
            self.state_manager.output_state["output_size"] = len(str(final_output))
            
            return state
            
        except Exception as e:
            self.logger.error(f"Output generation failed: {e}")
            raise
    
    def _calculate_output_quality(self, output: Dict[str, Any]) -> float:
        """Calculate overall output quality score."""
        # Simple quality calculation - can be enhanced
        scores = []
        
        # Check completeness
        required_fields = ["llm_outputs", "knowledge_graph_summary", "processing_metrics"]
        completeness = sum(1 for field in required_fields if field in output) / len(required_fields)
        scores.append(completeness)
        
        # Check processing success rate
        if "quality_metrics" in output:
            avg_quality = sum(output["quality_metrics"].values()) / max(len(output["quality_metrics"]), 1)
            scores.append(avg_quality)
        
        return sum(scores) / len(scores) if scores else 0.0


class LangGraphOrchestrator:
    """Main orchestrator class that builds and manages the LangGraph workflow."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.state_manager = CentralizedStateManager()
        self.workflow_graph = None
        self.checkpointer = MemorySaver()
        
        # Initialize nodes
        self.nodes = self._create_nodes()
        
        # Build the workflow graph
        self._build_workflow()
        
        logger.info("LangGraphOrchestrator initialized successfully")
    
    def _create_nodes(self) -> Dict[str, BaseWorkflowNode]:
        """Create and configure all workflow nodes."""
        nodes = {}
        
        # Document parsing node
        nodes["parse_document"] = DocumentParsingNode(
            NodeConfig(
                name="parse_document",
                max_retries=2,
                required_inputs=["document_path"],
                outputs=["document_content"]
            ),
            self.state_manager
        )
        
        # Chunking node
        nodes["chunk_document"] = IntelligentChunkingNode(
            NodeConfig(
                name="chunk_document",
                max_retries=3,
                required_inputs=["document_content"],
                outputs=["chunks"]
            ),
            self.state_manager
        )
        
        # Knowledge graph node
        nodes["build_knowledge_graph"] = KnowledgeGraphNode(
            NodeConfig(
                name="build_knowledge_graph",
                max_retries=2,
                required_inputs=["chunks"],
                outputs=["knowledge_graph"]
            ),
            self.state_manager
        )
        
        # LLM processing node
        nodes["process_llm"] = LLMProcessingNode(
            NodeConfig(
                name="process_llm",
                max_retries=3,
                timeout_seconds=600.0,
                required_inputs=["chunks"],
                outputs=["llm_outputs"]
            ),
            self.state_manager
        )
        
        # Output generation node
        nodes["generate_output"] = OutputGenerationNode(
            NodeConfig(
                name="generate_output",
                max_retries=2,
                required_inputs=["llm_outputs", "knowledge_graph"],
                outputs=["final_output"]
            ),
            self.state_manager
        )
        
        return nodes
    
    def _build_workflow(self) -> None:
        """Build the LangGraph workflow with proper node connections."""
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes to the graph
        for node_name, node_instance in self.nodes.items():
            workflow.add_node(node_name, node_instance.execute)
        
        # Define the workflow edges (execution order)
        workflow.set_entry_point("parse_document")
        workflow.add_edge("parse_document", "chunk_document")
        workflow.add_edge("chunk_document", "build_knowledge_graph")
        workflow.add_edge("chunk_document", "process_llm")  # Parallel execution
        workflow.add_edge("build_knowledge_graph", "generate_output")
        workflow.add_edge("process_llm", "generate_output")
        workflow.add_edge("generate_output", END)
        
        # Compile the workflow with checkpointing
        self.workflow_graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("Workflow graph compiled successfully")
    
    async def process_document(self, document_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a document through the complete workflow."""
        session_id = self.state_manager.session_id
        processing_config = {**self.config, **(config or {})}
        
        # Initialize workflow state
        initial_state: WorkflowState = {
            "document_content": "",
            "document_path": document_path,
            "chunks": [],
            "knowledge_graph": {},
            "semantic_mappings": [],
            "llm_outputs": [],
            "final_output": {},
            "current_stage": ProcessingStage.INITIALIZATION.name,
            "progress": 0.0,
            "session_id": session_id,
            "processing_config": processing_config,
            "errors": [],
            "retry_counts": {},
            "max_retries": {},
            "stage_timings": {},
            "memory_usage": {},
            "token_usage": {},
            "quality_scores": {},
            "validation_results": {},
            "next_node": None,
            "should_continue": True,
            "is_cancelled": False
        }
        
        try:
            logger.info(f"Starting document processing for: {document_path}")
            
            # Execute the workflow
            async with self.state_manager.stage_context(ProcessingStage.INITIALIZATION):
                final_state = await self.workflow_graph.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": session_id}}
                )
            
            # Mark as completed
            async with self.state_manager.stage_context(ProcessingStage.COMPLETED):
                final_state["current_stage"] = ProcessingStage.COMPLETED.name
                final_state["progress"] = 1.0
            
            logger.info(f"Document processing completed successfully for session: {session_id}")
            return final_state["final_output"]
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_path}: {e}")
            
            # Record fatal error
            self.state_manager.add_error(ErrorContext(
                stage=ProcessingStage.ERROR,
                severity=ErrorSeverity.FATAL,
                message=str(e),
                exception_type=type(e).__name__
            ))
            
            raise
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and metrics."""
        return self.state_manager.get_state_snapshot()
    
    async def cancel_processing(self) -> None:
        """Cancel the current processing session."""
        self.state_manager.cancel_processing()
        logger.info("Processing cancellation requested")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.state_manager.cleanup()
        logger.info("LangGraphOrchestrator cleanup completed")


# Factory function for easy instantiation
def create_orchestrator(config: Optional[Dict[str, Any]] = None) -> LangGraphOrchestrator:
    """Create a new LangGraph orchestrator instance."""
    return LangGraphOrchestrator(config)