"""LangGraph Workflow Orchestrator for Document Processing Pipeline

This module defines the LangGraph workflow structure, routing logic,
and conditional flows for the document processing pipeline.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Fallback for when LangGraph is not available
    logging.warning("LangGraph not available, using fallback implementation")
    StateGraph = None
    END = "END"
    MemorySaver = None

from langgraph_state import (
    PipelineState, ProcessingStage, ErrorInfo, ErrorSeverity,
    create_initial_state, serialize_state
)
from langgraph_nodes import DocumentProcessingNodes


logger = logging.getLogger(__name__)


class DocumentProcessingWorkflow:
    """LangGraph workflow orchestrator for document processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes = DocumentProcessingNodes(config)
        self.graph = None
        self.checkpointer = MemorySaver() if MemorySaver else None
        self._build_workflow()
    
    def _build_workflow(self) -> None:
        """Build the LangGraph workflow structure."""
        if not StateGraph:
            logger.warning("LangGraph not available, workflow will use fallback mode")
            return
        
        try:
            # Create the state graph
            workflow = StateGraph(PipelineState)
            
            # Add nodes for each processing stage
            workflow.add_node("initialization", self.nodes.initialization_node)
            workflow.add_node("document_parsing", self.nodes.document_parsing_node)
            workflow.add_node("chunking", self.nodes.chunking_node)
            workflow.add_node("knowledge_graph", self.nodes.knowledge_graph_node)
            workflow.add_node("semantic_mapping", self.nodes.semantic_mapping_node)
            workflow.add_node("llm_processing", self.nodes.llm_processing_node)
            workflow.add_node("output_generation", self.nodes.output_generation_node)
            workflow.add_node("analytics", self.nodes.analytics_node)
            workflow.add_node("completion", self.nodes.completion_node)
            
            # Add error handling and recovery nodes
            workflow.add_node("error_handler", self._error_handler_node)
            workflow.add_node("retry_handler", self._retry_handler_node)
            workflow.add_node("human_input", self._human_input_node)
            workflow.add_node("quality_gate", self._quality_gate_node)
            
            # Set entry point
            workflow.set_entry_point("initialization")
            
            # Add conditional edges with routing logic
            workflow.add_conditional_edges(
                "initialization",
                self._route_after_initialization,
                {
                    "continue": "document_parsing",
                    "error": "error_handler",
                    "retry": "retry_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "document_parsing",
                self._route_after_parsing,
                {
                    "continue": "chunking",
                    "error": "error_handler",
                    "retry": "retry_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "chunking",
                self._route_after_chunking,
                {
                    "continue": "quality_gate",
                    "error": "error_handler",
                    "retry": "retry_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "quality_gate",
                self._route_after_quality_gate,
                {
                    "knowledge_graph": "knowledge_graph",
                    "semantic_mapping": "semantic_mapping",
                    "llm_processing": "llm_processing",
                    "human_input": "human_input",
                    "error": "error_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "knowledge_graph",
                self._route_after_knowledge_graph,
                {
                    "continue": "semantic_mapping",
                    "skip": "llm_processing",
                    "error": "error_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "semantic_mapping",
                self._route_after_semantic_mapping,
                {
                    "continue": "llm_processing",
                    "error": "error_handler",
                    "retry": "retry_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "llm_processing",
                self._route_after_llm_processing,
                {
                    "continue": "output_generation",
                    "error": "error_handler",
                    "retry": "retry_handler",
                    "human_input": "human_input"
                }
            )
            
            workflow.add_conditional_edges(
                "output_generation",
                self._route_after_output_generation,
                {
                    "continue": "analytics",
                    "error": "error_handler",
                    "retry": "retry_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "analytics",
                self._route_after_analytics,
                {
                    "continue": "completion",
                    "error": "error_handler"
                }
            )
            
            workflow.add_conditional_edges(
                "completion",
                self._route_after_completion,
                {
                    "end": END,
                    "human_input": "human_input"
                }
            )
            
            # Error handling routes
            workflow.add_conditional_edges(
                "error_handler",
                self._route_after_error_handler,
                {
                    "retry": "retry_handler",
                    "human_input": "human_input",
                    "end": END
                }
            )
            
            workflow.add_conditional_edges(
                "retry_handler",
                self._route_after_retry_handler,
                {
                    "initialization": "initialization",
                    "document_parsing": "document_parsing",
                    "chunking": "chunking",
                    "knowledge_graph": "knowledge_graph",
                    "semantic_mapping": "semantic_mapping",
                    "llm_processing": "llm_processing",
                    "output_generation": "output_generation",
                    "error": "error_handler",
                    "end": END
                }
            )
            
            workflow.add_conditional_edges(
                "human_input",
                self._route_after_human_input,
                {
                    "continue": "completion",
                    "retry": "retry_handler",
                    "end": END
                }
            )
            
            # Compile the workflow
            self.graph = workflow.compile(checkpointer=self.checkpointer)
            logger.info("LangGraph workflow compiled successfully")
            
        except Exception as e:
            logger.error(f"Failed to build workflow: {e}")
            self.graph = None
    
    async def process_document(self, document_path: str, session_id: str = None) -> PipelineState:
        """Process a document through the LangGraph workflow."""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create initial state
        initial_state = create_initial_state(session_id, document_path, self.config)
        
        if self.graph:
            # Use LangGraph workflow
            try:
                config = {"configurable": {"thread_id": session_id}}
                result = await self.graph.ainvoke(initial_state, config=config)
                return result
            except Exception as e:
                logger.error(f"LangGraph workflow failed: {e}")
                return await self._fallback_processing(initial_state)
        else:
            # Use fallback linear processing
            return await self._fallback_processing(initial_state)
    
    async def _fallback_processing(self, state: PipelineState) -> PipelineState:
        """Fallback linear processing when LangGraph is not available."""
        logger.info("Using fallback linear processing")
        
        try:
            # Process through each stage sequentially
            state = await self.nodes.initialization_node(state)
            if state.has_errors and not state.is_recoverable:
                return state
            
            state = await self.nodes.document_parsing_node(state)
            if state.has_errors and not state.is_recoverable:
                return state
            
            state = await self.nodes.chunking_node(state)
            if state.has_errors and not state.is_recoverable:
                return state
            
            state = await self.nodes.knowledge_graph_node(state)
            # Continue even if knowledge graph fails (non-critical)
            
            state = await self.nodes.semantic_mapping_node(state)
            # Continue even if semantic mapping fails (non-critical)
            
            state = await self.nodes.llm_processing_node(state)
            if state.has_errors and not state.is_recoverable:
                return state
            
            state = await self.nodes.output_generation_node(state)
            if state.has_errors and not state.is_recoverable:
                return state
            
            state = await self.nodes.analytics_node(state)
            # Continue even if analytics fails (non-critical)
            
            state = await self.nodes.completion_node(state)
            
        except Exception as e:
            error = ErrorInfo(
                stage=state.current_stage,
                severity=ErrorSeverity.CRITICAL,
                message=f"Fallback processing failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
        
        return state
    
    # Routing functions for conditional edges
    def _route_after_initialization(self, state: PipelineState) -> str:
        """Route after initialization stage."""
        if state.has_errors:
            if state.should_retry():
                return "retry"
            return "error"
        return "continue"
    
    def _route_after_parsing(self, state: PipelineState) -> str:
        """Route after document parsing stage."""
        if state.has_errors:
            if state.should_retry():
                return "retry"
            return "error"
        return "continue"
    
    def _route_after_chunking(self, state: PipelineState) -> str:
        """Route after chunking stage."""
        if state.has_errors:
            if state.should_retry():
                return "retry"
            return "error"
        return "continue"
    
    def _route_after_quality_gate(self, state: PipelineState) -> str:
        """Route after quality gate check."""
        if state.has_errors:
            return "error"
        
        # Check configuration for which stages to run
        if self.config.get('enable_knowledge_graph', True):
            return "knowledge_graph"
        elif self.config.get('enable_semantic_mapping', True):
            return "semantic_mapping"
        else:
            return "llm_processing"
    
    def _route_after_knowledge_graph(self, state: PipelineState) -> str:
        """Route after knowledge graph creation."""
        if state.has_errors:
            # Knowledge graph errors are not critical
            if self.config.get('enable_semantic_mapping', True):
                return "continue"
            else:
                return "skip"
        return "continue"
    
    def _route_after_semantic_mapping(self, state: PipelineState) -> str:
        """Route after semantic mapping."""
        if state.has_errors:
            if state.should_retry():
                return "retry"
            return "error"
        return "continue"
    
    def _route_after_llm_processing(self, state: PipelineState) -> str:
        """Route after LLM processing."""
        if state.has_errors:
            if state.should_retry():
                return "retry"
            return "error"
        
        # Check if human input is required
        if state.requires_human_input:
            return "human_input"
        
        return "continue"
    
    def _route_after_output_generation(self, state: PipelineState) -> str:
        """Route after output generation."""
        if state.has_errors:
            if state.should_retry():
                return "retry"
            return "error"
        return "continue"
    
    def _route_after_analytics(self, state: PipelineState) -> str:
        """Route after analytics generation."""
        if state.has_errors:
            return "error"
        return "continue"
    
    def _route_after_completion(self, state: PipelineState) -> str:
        """Route after completion."""
        if state.requires_human_input:
            return "human_input"
        return "end"
    
    def _route_after_error_handler(self, state: PipelineState) -> str:
        """Route after error handling."""
        if state.should_retry():
            return "retry"
        elif state.requires_human_input:
            return "human_input"
        return "end"
    
    def _route_after_retry_handler(self, state: PipelineState) -> str:
        """Route after retry handling."""
        if not state.should_retry():
            return "end"
        
        # Route back to the failed stage
        if state.current_stage == ProcessingStage.INITIALIZATION:
            return "initialization"
        elif state.current_stage == ProcessingStage.DOCUMENT_PARSING:
            return "document_parsing"
        elif state.current_stage == ProcessingStage.CHUNKING:
            return "chunking"
        elif state.current_stage == ProcessingStage.KNOWLEDGE_GRAPH:
            return "knowledge_graph"
        elif state.current_stage == ProcessingStage.SEMANTIC_MAPPING:
            return "semantic_mapping"
        elif state.current_stage == ProcessingStage.LLM_PROCESSING:
            return "llm_processing"
        elif state.current_stage == ProcessingStage.OUTPUT_GENERATION:
            return "output_generation"
        else:
            return "error"
    
    def _route_after_human_input(self, state: PipelineState) -> str:
        """Route after human input."""
        # Check human feedback for next action
        action = state.human_feedback.get('action', 'continue')
        
        if action == 'retry':
            return "retry"
        elif action == 'continue':
            return "continue"
        else:
            return "end"
    
    # Special node implementations
    async def _error_handler_node(self, state: PipelineState) -> PipelineState:
        """Handle errors in the pipeline."""
        logger.info(f"Handling errors in stage: {state.current_stage}")
        
        # Log all errors
        for error in state.errors:
            logger.error(f"Error in {error.stage}: {error.message}")
        
        # Determine if recovery is possible
        critical_errors = state.get_errors_by_severity(ErrorSeverity.CRITICAL)
        if critical_errors:
            state.is_recoverable = False
            logger.error("Critical errors detected, pipeline cannot recover")
        
        # Check if human input is needed
        high_errors = state.get_errors_by_severity(ErrorSeverity.HIGH)
        if high_errors and not state.should_retry():
            state.requires_human_input = True
            state.human_input_prompt = f"High severity errors in {state.current_stage}. Please review."
        
        return state
    
    async def _retry_handler_node(self, state: PipelineState) -> PipelineState:
        """Handle retries in the pipeline."""
        logger.info(f"Handling retry for stage: {state.current_stage}")
        
        if state.should_retry():
            state.increment_retry()
            
            # Clear recoverable errors
            recoverable_errors = [e for e in state.errors if e.recoverable]
            if recoverable_errors:
                logger.info(f"Clearing {len(recoverable_errors)} recoverable errors")
                state.errors = [e for e in state.errors if not e.recoverable]
                state.has_errors = len(state.errors) > 0
        
        return state
    
    async def _human_input_node(self, state: PipelineState) -> PipelineState:
        """Handle human input requirements."""
        logger.info("Waiting for human input")
        
        # In a real implementation, this would wait for actual human input
        # For now, we'll simulate automatic approval
        state.human_feedback = {
            'action': 'continue',
            'timestamp': datetime.now().isoformat(),
            'message': 'Automatically approved for demo'
        }
        
        state.requires_human_input = False
        
        return state
    
    async def _quality_gate_node(self, state: PipelineState) -> PipelineState:
        """Check quality gates before proceeding."""
        logger.info("Checking quality gates")
        
        # Check minimum chunk count
        if len(state.chunks) < self.config.get('min_chunks', 1):
            state.set_quality_gate("min_chunks", False)
            error = ErrorInfo(
                stage=ProcessingStage.CHUNKING,
                severity=ErrorSeverity.MEDIUM,
                message=f"Insufficient chunks: {len(state.chunks)}",
                exception_type="QualityGateError"
            )
            state.add_error(error)
        else:
            state.set_quality_gate("min_chunks", True)
        
        # Check content quality
        avg_chunk_size = sum(len(c.content) for c in state.chunks) / len(state.chunks) if state.chunks else 0
        if avg_chunk_size < self.config.get('min_chunk_size', 50):
            state.set_quality_gate("chunk_quality", False)
        else:
            state.set_quality_gate("chunk_quality", True)
        
        return state
    
    def get_workflow_status(self, session_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow session."""
        if not self.checkpointer:
            return {"status": "unknown", "message": "No checkpointer available"}
        
        try:
            # In a real implementation, this would retrieve the actual state
            return {
                "status": "active",
                "session_id": session_id,
                "message": "Workflow status retrieved"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def list_active_sessions(self) -> List[str]:
        """List all active workflow sessions."""
        if not self.checkpointer:
            return []
        
        # In a real implementation, this would list actual sessions
        return []