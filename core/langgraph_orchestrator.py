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
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

# LangGraph imports with correct paths
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .models import WorkflowState
from .chunking_processor import DocumentChunk, ChunkMetadata
from langgraph.prebuilt import ToolNode
from .llm_handler import ProcessingRequest

class LangGraphState(TypedDict, total=False):
    """TypedDict version of WorkflowState for LangGraph compatibility."""
    document_id: str
    document_path: str
    original_content: str
    document_content: Any
    processing_config: Dict[str, Any]
    metadata: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    processed_sections: List[Dict[str, Any]]
    template_applied: str
    validation_results: Dict[str, Any]
    combined_content: str
    combination_metadata: Dict[str, Any]
    formatted_content: str
    formatting_metadata: Dict[str, Any]
    knowledge_graph: Dict[str, Any]
    llm_results: Dict[str, Any]
    final_output: str
    errors: List[str]
    processing_time: float
    llm_handler: Any
    llm_client: Any
    target_format: str
    formatting_options: Dict[str, Any]

# Internal imports
from .state_manager import (
    CentralizedStateManager, ProcessingStage, ErrorContext, ErrorSeverity,
    ProcessingMetrics
)
from .workflow_base import BaseWorkflowNode, NodeConfig
from .workflow_node_wrappers import (
    TemplateProcessorWorkflowNode,
    ValidationWorkflowNode,
    CombinationWorkflowNode,
    OutputFormatterWorkflowNode
)

logger = logging.getLogger(__name__)


class NodeResult(Enum):
    """Standardized node execution results."""
    SUCCESS = "success"
    RETRY = "retry"
    SKIP = "skip"
    ERROR = "error"
    FATAL = "fatal"


class LegacyWorkflowState(TypedDict):
    """Legacy workflow state schema - kept for backward compatibility."""
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





class DocumentParsingNode(BaseWorkflowNode):
    """Enhanced document parsing with multiple format support."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Parse document with enhanced error handling and format detection."""
        document_path = Path(state.document_path or "")
        
        # Import parsing modules dynamically to avoid circular imports
        from core.document_parser import DocumentParser
        
        parser = DocumentParser()
        
        try:
            # Parse document with format auto-detection
            content = await asyncio.get_event_loop().run_in_executor(
                None, parser.parse_document, document_path
            )
            
            # Extract text content from ParsedDocument object
            if hasattr(content, 'content'):
                state.document_content = content.content
            else:
                state.document_content = str(content)
            
            state.current_stage = ProcessingStage.DOCUMENT_PARSING.name
            
            # Store parsing metadata
            self.state_manager.document_state["original_path"] = str(document_path)
            self.state_manager.document_state["file_size"] = document_path.stat().st_size
            self.state_manager.document_state["content_length"] = len(state.document_content)
            
            # Debug: Confirm document_content was set
            print(f"DEBUG: DocumentParsingNode set document_content: {state.document_content is not None}")
            print(f"DEBUG: DocumentParsingNode current_stage: {state.current_stage}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Document parsing failed for {document_path}: {e}")
            raise


class IntelligentChunkingNode(BaseWorkflowNode):
    """Advanced chunking with semantic awareness and optimization."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Perform intelligent chunking with quality validation."""
        content = state.document_content or ""
        config = state.processing_config or {}
        
        # Import chunking module
        from modules.chunker import AdaptiveChunker
        
        chunker = AdaptiveChunker(
            max_tokens_fine=config.get("chunk_size", 1000),
            overlap_tokens=config.get("chunk_overlap", 200)
        )
        
        try:
            # Perform chunking with quality assessment
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, chunker.process_chunks, content
            )
            
            # Validate chunk quality
            quality_scores = [chunk.get("quality_score", 0.0) for chunk in chunks]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            state.chunks = chunks
            state.current_stage = ProcessingStage.CHUNKING.name
            if not state.quality_scores:
                state.quality_scores = {}
            state.quality_scores["chunking"] = avg_quality
            
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
        chunks = state.chunks or []
        
        # Import knowledge graph module
        from modules.knowledge_graph_processor import KnowledgeGraphProcessor
        from sentence_transformers import SentenceTransformer
        
        # Get chunks from state
        chunks = state.chunks or []
        if not chunks:
            logger.warning("No chunks available for knowledge graph processing")
            return state
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        kg_processor = KnowledgeGraphProcessor(chunks, embedding_model)
        
        try:
            # Build knowledge graph
            knowledge_graph = await asyncio.get_event_loop().run_in_executor(
                None, kg_processor.build_graphs
            )
            
            # Handle None result from build_graphs
            if knowledge_graph is None:
                knowledge_graph = {"nodes": [], "edges": []}
                logger.warning("Knowledge graph processor returned None, using empty graph")
            
            state.knowledge_graph = knowledge_graph
            state.current_stage = ProcessingStage.KNOWLEDGE_GRAPH_BUILDING.name
            
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
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager, llm_handler):
        super().__init__(config, state_manager)
        self.llm_handler = llm_handler
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Process chunks through LLM with batching and rate limiting."""
        chunks = state.chunks or []
        config = state.processing_config or {}
        
        # Use LLM handler from orchestrator (not from state to avoid serialization issues)
        llm_handler = self.llm_handler
        logger.info(f"LLMProcessingNode: llm_handler is None: {llm_handler is None}")
        if llm_handler is None:
            logger.error("LLM handler is None in processing node!")
            raise ValueError("LLM handler is None - cannot process chunks")
        
        try:
            # Process chunks in batches
            llm_outputs = []
            total_tokens = 0
            
            for i in range(0, len(chunks), llm_handler.config.batch_size):
                chunk_dicts = chunks[i:i + llm_handler.config.batch_size]
                
                # Convert chunk dicts to ProcessingRequest objects
                batch = []
                for chunk_dict in chunk_dicts:
                    # Create ChunkMetadata from dict
                    metadata_dict = chunk_dict.get('metadata', {})
                    chunk_metadata = ChunkMetadata(
                        chunk_id=chunk_dict.get('chunk_id', f'chunk_{i}'),
                        index=metadata_dict.get('index', 0),
                        start_position=chunk_dict.get('start_position', 0),
                        end_position=chunk_dict.get('end_position', 0),
                        character_count=len(chunk_dict.get('content', '')),
                        word_count=len(chunk_dict.get('content', '').split()),
                        sentence_count=chunk_dict.get('content', '').count('.') + 1,
                        paragraph_count=chunk_dict.get('content', '').count('\n\n') + 1
                    )
                    
                    # Create DocumentChunk from dict
                    doc_chunk = DocumentChunk(
                        content=chunk_dict.get('content', ''),
                        metadata=chunk_metadata
                    )
                    
                    # Create ProcessingRequest
                    processing_request = ProcessingRequest(
                        chunk=doc_chunk,
                        prompt_template=config.get('prompt_template', 'Analyze this content: {content}'),
                        system_prompt=config.get('system_prompt'),
                        context=config.get('context', {})
                    )
                    batch.append(processing_request)
                
                # Check for cancellation
                if getattr(state, 'is_cancelled', False):
                    break
                
                batch_results = await llm_handler._process_batch(batch)
                llm_outputs.extend(batch_results)
                
                # Track token usage
                batch_tokens = sum(result.token_usage.get("total", 0) for result in batch_results)
                total_tokens += batch_tokens
                
                # Update progress
                progress = min(0.7 + 0.2 * (i + len(batch)) / len(chunks), 0.9)
                state.progress = progress
                
                self.logger.info(f"Processed batch {i//llm_handler.config.batch_size + 1}, tokens: {batch_tokens}")
            
            state.llm_outputs = llm_outputs
            state.current_stage = ProcessingStage.LLM_PROCESSING.name
            if "token_usage" not in state.stage_results:
                state.stage_results["token_usage"] = {}
            state.stage_results["token_usage"]["total"] = total_tokens
            
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
        llm_outputs = state.llm_outputs or []
        knowledge_graph = state.knowledge_graph or {}
        config = state.processing_config or {}
        
        try:
            # Generate final output
            final_output = {
                "document_path": state.document_path or "",
                "session_id": state.session_id or "",
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "chunks_processed": len(llm_outputs),
                "knowledge_graph_summary": {
                    "nodes": len(knowledge_graph.get("nodes", [])),
                    "edges": len(knowledge_graph.get("edges", []))
                },
                "llm_outputs": llm_outputs,
                "quality_metrics": state.quality_scores or {},
                "processing_metrics": {
                    "total_time": sum((state.stage_timings or {}).values()),
                    "token_usage": state.token_usage or {},
                    "memory_usage": getattr(state, 'memory_usage', {})
                }
            }
            
            # Validate output quality
            quality_score = self._calculate_output_quality(final_output)
            final_output["overall_quality_score"] = quality_score
            
            state.final_output = final_output
            state.current_stage = ProcessingStage.OUTPUT_GENERATION.name
            state.progress = 0.95
            
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_handler: Optional[Any] = None):
        self.config = config or {}
        self.llm_handler = llm_handler
        self.state_manager = CentralizedStateManager()
        self.workflow_graph = None
        self.checkpointer = MemorySaver()
        
        # Debug logging
        logger.info(f"LangGraphOrchestrator initializing with llm_handler: {self.llm_handler is not None}")
        if self.llm_handler:
            logger.info(f"LLM Handler provider: {self.llm_handler.config.provider}")
            logger.info(f"LLM Handler batch_size: {self.llm_handler.config.batch_size}")
        
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
        
        # Template processor node
        nodes["process_template"] = TemplateProcessorWorkflowNode(
            NodeConfig(
                name="process_template",
                max_retries=2,
                required_inputs=["chunks"],
                outputs=["processed_sections"]
            ),
            self.state_manager
        )
        
        # Validation node
        nodes["validate_sections"] = ValidationWorkflowNode(
            NodeConfig(
                name="validate_sections",
                max_retries=2,
                required_inputs=["processed_sections"],
                outputs=["validation_results"]
            ),
            self.state_manager
        )
        
        # Combination node
        nodes["combine_document"] = CombinationWorkflowNode(
            NodeConfig(
                name="combine_document",
                max_retries=2,
                required_inputs=["processed_sections", "validation_results"],
                outputs=["combined_content"]
            ),
            self.state_manager
        )
        
        # Output formatter node
        nodes["format_output"] = OutputFormatterWorkflowNode(
            NodeConfig(
                name="format_output",
                max_retries=2,
                required_inputs=["combined_content"],
                outputs=["formatted_content"]
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
            self.state_manager,
            self.llm_handler
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
        workflow = StateGraph(LangGraphState)
        
        # Add nodes to the graph with state conversion wrappers
        for node_name, node_instance in self.nodes.items():
            workflow.add_node(node_name, self._create_node_wrapper(node_instance))
        
        # Define the workflow edges (execution order)
        workflow.set_entry_point("parse_document")
        workflow.add_edge("parse_document", "chunk_document")
        
        # Sequential processing pipeline to avoid concurrent state updates
        workflow.add_edge("chunk_document", "process_template")
        workflow.add_edge("process_template", "validate_sections")
        workflow.add_edge("validate_sections", "combine_document")
        workflow.add_edge("combine_document", "format_output")
        
        # Sequential knowledge graph and LLM processing
        workflow.add_edge("format_output", "build_knowledge_graph")
        workflow.add_edge("build_knowledge_graph", "process_llm")
        
        # Final output generation
        workflow.add_edge("process_llm", "generate_output")
        workflow.add_edge("generate_output", END)
        
        # Compile the workflow with checkpointing
        self.workflow_graph = workflow.compile(checkpointer=self.checkpointer)
        
        logger.info("Workflow graph compiled successfully")
    
    def _create_node_wrapper(self, node_instance: BaseWorkflowNode):
        """Create a wrapper function that converts between LangGraphState and WorkflowState."""
        async def wrapper(state: LangGraphState) -> LangGraphState:
            try:
                logger.info(f"Executing node: {node_instance.config.name}")
                
                # Convert LangGraphState dict to WorkflowState object
                workflow_state = self._dict_to_workflow_state(state)
                
                # Execute the node with the WorkflowState object
                result_workflow_state = await node_instance.execute(workflow_state)
                
                # Convert WorkflowState back to LangGraphState dict
                result_state = self._workflow_state_to_dict(result_workflow_state)
                
                logger.info(f"Node {node_instance.config.name} completed successfully")
                return result_state
                
            except Exception as e:
                logger.error(f"Error in node {node_instance.config.name}: {e}")
                logger.exception(f"Full traceback for node {node_instance.config.name}:")
                # Return the original state with error information
                if 'errors' not in state:
                    state['errors'] = []
                state['errors'].append(f"Node {node_instance.config.name} failed: {str(e)}")
                return state
        
        return wrapper
    
    def _dict_to_workflow_state(self, state_dict: LangGraphState) -> WorkflowState:
        """Convert LangGraphState dict to WorkflowState dataclass."""
        from .models import DocumentMetadata
        
        print(f"DEBUG: Converting dict to WorkflowState. document_content in dict: {'document_content' in state_dict}")
        if 'document_content' in state_dict:
            print(f"DEBUG: document_content value: {type(state_dict['document_content'])}")
        
        # Convert metadata dict back to DocumentMetadata if needed
        metadata = state_dict.get("metadata", {})
        if isinstance(metadata, dict) and not isinstance(metadata, DocumentMetadata):
            metadata = DocumentMetadata(**metadata)
        
        return WorkflowState(
            document_id=state_dict.get("document_id", ""),
            document_path=state_dict.get("document_path", ""),
            original_content=state_dict.get("original_content", ""),
            document_content=state_dict.get("document_content"),
            processing_config=state_dict.get("processing_config", {}),
            metadata=metadata,
            chunks=state_dict.get("chunks", []),
            processed_sections=state_dict.get("processed_sections", []),
            llm_client=state_dict.get("llm_client"),
            llm_handler=state_dict.get("llm_handler"),
            final_output=state_dict.get("final_output", ""),
            template_name=state_dict.get("template_name"),
            template_applied=state_dict.get("template_applied"),
            processing_errors=state_dict.get("errors", []),
            processing_warnings=state_dict.get("warnings", []),
            current_stage=state_dict.get("current_stage", "initialization"),
            stage_results=state_dict.get("stage_results", {}),
            quality_scores=state_dict.get("quality_scores", {})
        )
    
    def _workflow_state_to_dict(self, workflow_state: WorkflowState) -> LangGraphState:
        """Convert WorkflowState dataclass to LangGraphState dict."""
        state_dict = asdict(workflow_state)
        
        print(f"DEBUG: Converting WorkflowState to dict. document_content in state_dict: {'document_content' in state_dict}")
        if 'document_content' in state_dict:
            print(f"DEBUG: document_content value: {type(state_dict['document_content'])}")
        
        # Ensure all required fields are present
        result: LangGraphState = {
            "document_id": state_dict.get("document_id", ""),
            "document_path": state_dict.get("document_path", ""),
            "original_content": state_dict.get("original_content", ""),
            "document_content": state_dict.get("document_content"),
            "processing_config": state_dict.get("processing_config", {}),
            "metadata": state_dict.get("metadata", {}),
            "chunks": state_dict.get("chunks", []),
            "processed_sections": state_dict.get("processed_sections", []),
            "template_applied": state_dict.get("template_applied", ""),
            "validation_results": state_dict.get("validation_results", {}),
            "combined_content": state_dict.get("combined_content", ""),
            "combination_metadata": state_dict.get("combination_metadata", {}),
            "formatted_content": state_dict.get("formatted_content", ""),
            "formatting_metadata": state_dict.get("formatting_metadata", {}),
            "knowledge_graph": state_dict.get("knowledge_graph", {}),
            "llm_results": state_dict.get("llm_results", {}),
            "final_output": state_dict.get("final_output", ""),
            "errors": state_dict.get("errors", []),
            "processing_time": state_dict.get("processing_time", 0.0),
            "llm_handler": state_dict.get("llm_handler"),
            "llm_client": state_dict.get("llm_client"),
            "target_format": state_dict.get("target_format", "markdown"),
            "formatting_options": state_dict.get("formatting_options", {})
        }
        
        return result
    
    async def process_document(self, document_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a document through the complete workflow."""
        session_id = self.state_manager.session_id
        processing_config = {**self.config, **(config or {})}
        
        # Initialize workflow state
        from pathlib import Path
        from .models import DocumentMetadata, DocumentFormat
        from datetime import datetime
        from dataclasses import asdict
        import uuid
        
        # Create required metadata
        doc_path = Path(document_path)
        metadata = DocumentMetadata(
            title=doc_path.stem,
            format=DocumentFormat.PLAIN_TEXT,
            creation_time=datetime.now(),
            modification_time=datetime.now()
        )
        
        initial_state: LangGraphState = {
            "document_id": str(uuid.uuid4()),
            "document_path": document_path,
            "original_content": "",
            "metadata": asdict(metadata),
            "chunks": [],
            "processed_sections": [],
            "template_applied": "",
            "validation_results": {},
            "combined_content": "",
            "combination_metadata": {},
            "formatted_content": "",
            "formatting_metadata": {},
            "knowledge_graph": {},
            "llm_results": {},
            "final_output": "",
            "errors": [],
            "processing_time": 0.0,
            "processing_config": processing_config,
            # Note: LLM handler is available through self.llm_handler, not stored in state
            # to avoid serialization issues with LangGraph checkpointer
            "target_format": "markdown",
            "formatting_options": {}
        }
        
        try:
            logger.info(f"Starting document processing for: {document_path}")
            
            # Execute the workflow
            async with self.state_manager.stage_context(ProcessingStage.INITIALIZATION):
                # Debug: Log initial state
                print(f"DEBUG: Initial state keys: {list(initial_state.keys())}")
                print(f"DEBUG: document_path in initial_state: {'document_path' in initial_state}")
                if 'document_path' in initial_state:
                    print(f"DEBUG: document_path value: {initial_state['document_path']}")
                
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