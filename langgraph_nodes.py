"""LangGraph Workflow Nodes for Document Processing Pipeline

This module implements the workflow nodes for each stage of the document
processing pipeline using LangGraph architecture.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib
import json

from langgraph_state import (
    PipelineState, ProcessingStage, ErrorInfo, ErrorSeverity,
    DocumentInfo, ChunkInfo, KnowledgeGraphNode, KnowledgeGraphEdge,
    SemanticMapping, LLMProcessingResult, OutputData, QualityMetrics
)

# Import existing modules
try:
    from modules.chunking_handler import ChunkingHandler
    from modules.llm_handler import LLMHandler
    from modules.intelligent_mapper import IntelligentMapper
    from modules.output_manager import OutputManager
    from modules.error_handler import ErrorHandler
    from config import Config
except ImportError as e:
    logging.warning(f"Could not import modules: {e}")
    # Fallback implementations will be used


logger = logging.getLogger(__name__)


class DocumentProcessingNodes:
    """Collection of LangGraph nodes for document processing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunking_handler = None
        self.llm_handler = None
        self.intelligent_mapper = None
        self.output_manager = None
        self.error_handler = None
        
        # Initialize handlers if available
        try:
            from modules.chunker import AdaptiveChunker
            from modules.llm_client import UnifiedLLMClient
            from modules.intelligent_mapper import IntelligentMapper
            from modules.output_manager import OutputManager
            from modules.error_handler import ErrorHandler
            
            self.chunking_handler = AdaptiveChunker()
            self.llm_handler = UnifiedLLMClient()
            self.intelligent_mapper = IntelligentMapper(config)
            self.output_manager = OutputManager(config)
            self.error_handler = ErrorHandler(config)
        except Exception as e:
            logger.warning(f"Could not initialize handlers: {e}")
    
    async def initialization_node(self, state: PipelineState) -> PipelineState:
        """Initialize the document processing pipeline."""
        try:
            logger.info(f"Starting initialization for session {state.session_info.session_id}")
            
            # Update stage and progress
            state.update_progress(ProcessingStage.INITIALIZATION, 0.1)
            
            # Validate document exists
            if not state.document_info or not Path(state.document_info.file_path).exists():
                error = ErrorInfo(
                    stage=ProcessingStage.INITIALIZATION,
                    severity=ErrorSeverity.CRITICAL,
                    message="Document file not found or not specified",
                    exception_type="FileNotFoundError"
                )
                state.add_error(error)
                return state
            
            # Calculate file hash for integrity
            with open(state.document_info.file_path, 'rb') as f:
                content = f.read()
                state.document_info.content_hash = hashlib.sha256(content).hexdigest()
            
            # Read document content
            with open(state.document_info.file_path, 'r', encoding='utf-8') as f:
                state.raw_content = f.read()
            
            # Initialize analytics
            state.analytics.processing_start_time = datetime.now()
            
            # Set quality gates
            state.set_quality_gate("document_loaded", True)
            state.set_quality_gate("content_valid", len(state.raw_content) > 0)
            
            state.update_progress(ProcessingStage.INITIALIZATION, 1.0)
            logger.info("Initialization completed successfully")
            
        except Exception as e:
            error = ErrorInfo(
                stage=ProcessingStage.INITIALIZATION,
                severity=ErrorSeverity.CRITICAL,
                message=f"Initialization failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
            logger.error(f"Initialization failed: {e}")
        
        return state
    
    async def document_parsing_node(self, state: PipelineState) -> PipelineState:
        """Parse and preprocess the document."""
        try:
            logger.info("Starting document parsing")
            state.update_progress(ProcessingStage.DOCUMENT_PARSING, 0.2)
            
            # Basic text preprocessing
            content = state.raw_content
            
            # Remove excessive whitespace
            content = ' '.join(content.split())
            
            # Store preprocessed content
            state.intermediate_results['preprocessed_content'] = content
            
            # Update document metadata
            state.document_info.metadata.update({
                'original_length': len(state.raw_content),
                'processed_length': len(content),
                'preprocessing_time': datetime.now().isoformat()
            })
            
            # Quality gate for parsing
            parsing_quality = len(content) > 100  # Minimum content length
            state.set_quality_gate("parsing_quality", parsing_quality)
            
            state.update_progress(ProcessingStage.DOCUMENT_PARSING, 1.0)
            logger.info("Document parsing completed")
            
        except Exception as e:
            error = ErrorInfo(
                stage=ProcessingStage.DOCUMENT_PARSING,
                severity=ErrorSeverity.HIGH,
                message=f"Document parsing failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
            logger.error(f"Document parsing failed: {e}")
        
        return state
    
    async def chunking_node(self, state: PipelineState) -> PipelineState:
        """Chunk the document into manageable pieces."""
        try:
            logger.info("Starting document chunking")
            state.update_progress(ProcessingStage.CHUNKING, 0.3)
            
            content = state.intermediate_results.get('preprocessed_content', state.raw_content)
            
            if self.chunking_handler:
                # Use existing chunking handler
                chunks_data = self.chunking_handler.process_chunks(content, state.document_info.file_path if state.document_info else "")
            else:
                # Fallback chunking implementation
                chunks_data = self._fallback_chunking(content, state.chunk_size, state.chunk_overlap)
            
            # Convert to ChunkInfo objects
            for i, chunk_data in enumerate(chunks_data):
                chunk = ChunkInfo(
                    chunk_id=f"chunk_{i:04d}",
                    content=chunk_data.get('content', ''),
                    start_position=chunk_data.get('start', 0),
                    end_position=chunk_data.get('end', 0),
                    chunk_size=len(chunk_data.get('content', '')),
                    overlap_size=state.chunk_overlap,
                    metadata={
                        'chunk_index': i,
                        'strategy': state.chunk_strategy,
                        'created_at': datetime.now().isoformat()
                    }
                )
                state.add_chunk(chunk)
            
            # Quality gate for chunking
            chunking_quality = len(state.chunks) > 0 and all(len(c.content) > 50 for c in state.chunks)
            state.set_quality_gate("chunking_quality", chunking_quality)
            
            # Update analytics
            state.analytics.stage_timings['chunking'] = 0.5  # Placeholder timing
            
            state.update_progress(ProcessingStage.CHUNKING, 1.0)
            logger.info(f"Chunking completed: {len(state.chunks)} chunks created")
            
        except Exception as e:
            error = ErrorInfo(
                stage=ProcessingStage.CHUNKING,
                severity=ErrorSeverity.HIGH,
                message=f"Chunking failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
            logger.error(f"Chunking failed: {e}")
        
        return state
    
    async def knowledge_graph_node(self, state: PipelineState) -> PipelineState:
        """Create knowledge graph from chunks."""
        try:
            logger.info("Starting knowledge graph creation")
            state.update_progress(ProcessingStage.KNOWLEDGE_GRAPH, 0.4)
            
            # Process each chunk for knowledge extraction
            for chunk in state.chunks:
                # Extract entities and relationships (simplified)
                entities = self._extract_entities(chunk.content)
                relationships = self._extract_relationships(chunk.content)
                
                # Create nodes
                for entity in entities:
                    node = KnowledgeGraphNode(
                        node_id=f"node_{entity['id']}",
                        content=entity['text'],
                        node_type=entity['type'],
                        confidence=entity['confidence'],
                        metadata={
                            'source_chunk': chunk.chunk_id,
                            'created_at': datetime.now().isoformat()
                        }
                    )
                    state.knowledge_graph_nodes.append(node)
                
                # Create edges
                for rel in relationships:
                    edge = KnowledgeGraphEdge(
                        edge_id=f"edge_{rel['id']}",
                        source_node=rel['source'],
                        target_node=rel['target'],
                        relationship_type=rel['type'],
                        confidence=rel['confidence'],
                        metadata={
                            'source_chunk': chunk.chunk_id,
                            'created_at': datetime.now().isoformat()
                        }
                    )
                    state.knowledge_graph_edges.append(edge)
            
            # Quality gate for knowledge graph
            kg_quality = len(state.knowledge_graph_nodes) > 0
            state.set_quality_gate("knowledge_graph_quality", kg_quality)
            
            state.update_progress(ProcessingStage.KNOWLEDGE_GRAPH, 1.0)
            logger.info(f"Knowledge graph created: {len(state.knowledge_graph_nodes)} nodes, {len(state.knowledge_graph_edges)} edges")
            
        except Exception as e:
            error = ErrorInfo(
                stage=ProcessingStage.KNOWLEDGE_GRAPH,
                severity=ErrorSeverity.MEDIUM,
                message=f"Knowledge graph creation failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
            logger.error(f"Knowledge graph creation failed: {e}")
        
        return state
    
    async def semantic_mapping_node(self, state: PipelineState) -> PipelineState:
        """Perform semantic mapping of chunks."""
        try:
            logger.info("Starting semantic mapping")
            state.update_progress(ProcessingStage.SEMANTIC_MAPPING, 0.5)
            
            if self.intelligent_mapper:
                # Use existing intelligent mapper
                for chunk in state.chunks:
                    mapping_result = await self.intelligent_mapper.map_chunk(chunk.content)
                    
                    semantic_mapping = SemanticMapping(
                        chunk_id=chunk.chunk_id,
                        semantic_category=mapping_result.get('category', 'unknown'),
                        confidence=mapping_result.get('confidence', 0.0),
                        keywords=mapping_result.get('keywords', []),
                        entities=mapping_result.get('entities', []),
                        relationships=mapping_result.get('relationships', [])
                    )
                    state.semantic_mappings.append(semantic_mapping)
            else:
                # Fallback semantic mapping
                for chunk in state.chunks:
                    mapping = self._fallback_semantic_mapping(chunk)
                    state.semantic_mappings.append(mapping)
            
            # Quality gate for semantic mapping
            semantic_quality = len(state.semantic_mappings) == len(state.chunks)
            state.set_quality_gate("semantic_mapping_quality", semantic_quality)
            
            state.update_progress(ProcessingStage.SEMANTIC_MAPPING, 1.0)
            logger.info(f"Semantic mapping completed: {len(state.semantic_mappings)} mappings created")
            
        except Exception as e:
            error = ErrorInfo(
                stage=ProcessingStage.SEMANTIC_MAPPING,
                severity=ErrorSeverity.MEDIUM,
                message=f"Semantic mapping failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
            logger.error(f"Semantic mapping failed: {e}")
        
        return state
    
    async def llm_processing_node(self, state: PipelineState) -> PipelineState:
        """Process chunks using LLM."""
        try:
            logger.info("Starting LLM processing")
            state.update_progress(ProcessingStage.LLM_PROCESSING, 0.6)
            
            if self.llm_handler:
                # Use existing LLM handler
                for chunk in state.chunks:
                    start_time = datetime.now()
                    
                    llm_result = await self.llm_handler.process_chunk(
                        chunk.content, 
                        state.config
                    )
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    result = LLMProcessingResult(
                        chunk_id=chunk.chunk_id,
                        processed_content=llm_result.get('content', ''),
                        analysis=llm_result.get('analysis', {}),
                        confidence=llm_result.get('confidence', 0.0),
                        processing_time=processing_time,
                        model_used=llm_result.get('model', ''),
                        tokens_used=llm_result.get('tokens', 0)
                    )
                    state.add_llm_result(result)
            else:
                # Fallback LLM processing
                for chunk in state.chunks:
                    result = self._fallback_llm_processing(chunk)
                    state.add_llm_result(result)
            
            # Quality gate for LLM processing
            llm_quality = len(state.llm_results) == len(state.chunks)
            state.set_quality_gate("llm_processing_quality", llm_quality)
            
            # Update token usage analytics
            total_tokens = sum(result.tokens_used for result in state.llm_results)
            state.analytics.token_usage['total'] = total_tokens
            
            state.update_progress(ProcessingStage.LLM_PROCESSING, 1.0)
            logger.info(f"LLM processing completed: {len(state.llm_results)} results")
            
        except Exception as e:
            error = ErrorInfo(
                stage=ProcessingStage.LLM_PROCESSING,
                severity=ErrorSeverity.HIGH,
                message=f"LLM processing failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
            logger.error(f"LLM processing failed: {e}")
        
        return state
    
    async def output_generation_node(self, state: PipelineState) -> PipelineState:
        """Generate final output."""
        try:
            logger.info("Starting output generation")
            state.update_progress(ProcessingStage.OUTPUT_GENERATION, 0.8)
            
            if self.output_manager:
                # Use existing output manager
                output_data = await self.output_manager.generate_output(
                    state.llm_results,
                    state.config
                )
            else:
                # Fallback output generation
                output_data = self._fallback_output_generation(state)
            
            # Create output object
            output = OutputData(
                output_format=state.config.get('output_format', 'json'),
                content=output_data,
                metadata={
                    'generation_time': datetime.now().isoformat(),
                    'total_chunks': len(state.chunks),
                    'total_results': len(state.llm_results)
                }
            )
            state.add_output(output)
            
            # Quality gate for output
            output_quality = len(state.outputs) > 0
            state.set_quality_gate("output_generation_quality", output_quality)
            
            state.update_progress(ProcessingStage.OUTPUT_GENERATION, 1.0)
            logger.info("Output generation completed")
            
        except Exception as e:
            error = ErrorInfo(
                stage=ProcessingStage.OUTPUT_GENERATION,
                severity=ErrorSeverity.HIGH,
                message=f"Output generation failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
            logger.error(f"Output generation failed: {e}")
        
        return state
    
    async def analytics_node(self, state: PipelineState) -> PipelineState:
        """Generate analytics and performance metrics."""
        try:
            logger.info("Starting analytics generation")
            state.update_progress(ProcessingStage.ANALYTICS, 0.9)
            
            # Calculate final metrics
            state.analytics.processing_end_time = datetime.now()
            state.analytics.total_processing_time = (
                state.analytics.processing_end_time - state.analytics.processing_start_time
            ).total_seconds()
            
            # Calculate quality metrics
            quality_scores = []
            for result in state.llm_results:
                quality_scores.append(result.confidence)
            
            if quality_scores:
                state.analytics.quality_metrics.overall_quality = sum(quality_scores) / len(quality_scores)
            
            # Identify performance bottlenecks
            if state.analytics.stage_timings:
                slowest_stage = max(state.analytics.stage_timings.items(), key=lambda x: x[1])
                if slowest_stage[1] > 5.0:  # More than 5 seconds
                    state.analytics.performance_bottlenecks.append(f"Slow stage: {slowest_stage[0]}")
            
            # Quality gates summary
            passed_gates = [gate for gate, passed in state.quality_gates.items() if passed]
            failed_gates = [gate for gate, passed in state.quality_gates.items() if not passed]
            
            state.analytics.quality_metrics.quality_gates_passed = passed_gates
            state.analytics.quality_metrics.quality_gates_failed = failed_gates
            
            state.update_progress(ProcessingStage.ANALYTICS, 1.0)
            logger.info("Analytics generation completed")
            
        except Exception as e:
            error = ErrorInfo(
                stage=ProcessingStage.ANALYTICS,
                severity=ErrorSeverity.LOW,
                message=f"Analytics generation failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
            logger.error(f"Analytics generation failed: {e}")
        
        return state
    
    async def completion_node(self, state: PipelineState) -> PipelineState:
        """Mark pipeline as completed."""
        try:
            logger.info("Completing pipeline")
            state.update_progress(ProcessingStage.COMPLETED, 1.0)
            
            # Final validation
            if not state.all_quality_gates_passed():
                logger.warning("Some quality gates failed")
                state.requires_human_input = True
                state.human_input_prompt = "Quality gates failed. Please review results."
            
            # Update session info
            state.session_info.end_time = datetime.now()
            state.session_info.status = "completed"
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            error = ErrorInfo(
                stage=ProcessingStage.COMPLETED,
                severity=ErrorSeverity.MEDIUM,
                message=f"Completion failed: {str(e)}",
                exception_type=type(e).__name__
            )
            state.add_error(error)
            logger.error(f"Completion failed: {e}")
        
        return state
    
    # Fallback implementations
    def _fallback_chunking(self, content: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Fallback chunking implementation."""
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_content = content[start:end]
            
            chunks.append({
                'content': chunk_content,
                'start': start,
                'end': end
            })
            
            start = end - overlap if end < len(content) else end
        
        return chunks
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Simple entity extraction."""
        # Simplified entity extraction
        words = content.split()
        entities = []
        
        for i, word in enumerate(words[:10]):  # Limit to first 10 words
            if word.istitle():  # Simple heuristic for entities
                entities.append({
                    'id': f"entity_{i}",
                    'text': word,
                    'type': 'PERSON' if len(word) > 3 else 'OTHER',
                    'confidence': 0.7
                })
        
        return entities
    
    def _extract_relationships(self, content: str) -> List[Dict[str, Any]]:
        """Simple relationship extraction."""
        # Simplified relationship extraction
        return [{
            'id': 'rel_1',
            'source': 'entity_0',
            'target': 'entity_1',
            'type': 'RELATED_TO',
            'confidence': 0.6
        }]
    
    def _fallback_semantic_mapping(self, chunk: ChunkInfo) -> SemanticMapping:
        """Fallback semantic mapping."""
        return SemanticMapping(
            chunk_id=chunk.chunk_id,
            semantic_category='general',
            confidence=0.5,
            keywords=chunk.content.split()[:5],  # First 5 words as keywords
            entities=[],
            relationships=[]
        )
    
    def _fallback_llm_processing(self, chunk: ChunkInfo) -> LLMProcessingResult:
        """Fallback LLM processing."""
        return LLMProcessingResult(
            chunk_id=chunk.chunk_id,
            processed_content=f"Processed: {chunk.content[:100]}...",
            analysis={'summary': 'Basic processing completed'},
            confidence=0.6,
            processing_time=0.1,
            model_used='fallback',
            tokens_used=len(chunk.content.split())
        )
    
    def _fallback_output_generation(self, state: PipelineState) -> Dict[str, Any]:
        """Fallback output generation."""
        return {
            'summary': f"Processed {len(state.chunks)} chunks",
            'results': [result.processed_content for result in state.llm_results],
            'metadata': {
                'total_chunks': len(state.chunks),
                'processing_time': state.analytics.total_processing_time
            }
        }