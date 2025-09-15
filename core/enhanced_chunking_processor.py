"""Enhanced Chunking Processor with LangGraph Integration

This module extends the base chunking processor with LangGraph workflow nodes
for intelligent orchestration, adaptive strategy selection, and semantic-aware
processing with comprehensive error handling and performance optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone
import json
import hashlib

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Internal imports
from .chunking_processor import (
    ChunkingProcessor, ChunkingConfig, ChunkingStrategy, OverlapStrategy,
    DocumentChunk, ChunkingResult, ChunkMetadata, ChunkingError
)
from .document_parser import ParsedDocument
from .langgraph_orchestrator import BaseWorkflowNode, NodeConfig, WorkflowState, NodeResult
from .state_manager import CentralizedStateManager, ProcessingStage, ErrorSeverity
from .template_processor_node import TemplateProcessorNode
from .config import CHUNKING_STRATEGIES, QUALITY_THRESHOLDS
from .models import DocumentChunk, ChunkingResult, WorkflowState
from .semantic_mapper import SemanticMapper, BasicSemanticMapper

logger = logging.getLogger(__name__)


class ChunkingNodeType(Enum):
    """Types of chunking workflow nodes."""
    STRATEGY_SELECTOR = "strategy_selector"
    CONTENT_ANALYZER = "content_analyzer"
    ADAPTIVE_CHUNKER = "adaptive_chunker"
    SEMANTIC_PROCESSOR = "semantic_processor"
    QUALITY_VALIDATOR = "quality_validator"
    OVERLAP_MANAGER = "overlap_manager"
    METADATA_ENRICHER = "metadata_enricher"
    ORPHAN_HANDLER = "orphan_handler"


@dataclass
class EnhancedChunkingConfig(ChunkingConfig):
    """Enhanced configuration with LangGraph workflow settings."""
    enable_workflow: bool = True
    enable_adaptive_strategy: bool = True
    enable_semantic_analysis: bool = True
    enable_quality_validation: bool = True
    enable_orphan_handling: bool = True
    workflow_timeout: float = 300.0
    max_workflow_retries: int = 3
    parallel_processing: bool = True
    cache_intermediate_results: bool = True
    
    # Advanced chunking features
    enable_cross_chunk_relationships: bool = True
    enable_hierarchical_chunking: bool = False
    enable_dynamic_sizing: bool = True
    content_type_detection: bool = True
    
    # Performance optimization
    batch_processing_size: int = 10
    memory_optimization: bool = True
    streaming_mode: bool = False


class StrategySelectionNode(BaseWorkflowNode):
    """Node for intelligent chunking strategy selection."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Analyze document and select optimal chunking strategy."""
        document_content = state["document_content"]
        processing_config = state["processing_config"]
        
        # Analyze document characteristics
        analysis = await self._analyze_document_characteristics(document_content)
        
        # Select optimal strategy
        selected_strategy = self._select_optimal_strategy(analysis, processing_config)
        
        # Update state with strategy and analysis
        state["chunking_strategy"] = selected_strategy.value
        state["document_analysis"] = analysis
        state["strategy_confidence"] = analysis.get("confidence", 0.8)
        
        self.logger.info(f"Selected chunking strategy: {selected_strategy.name} (confidence: {analysis.get('confidence', 0.8):.2f})")
        
        return state
    
    async def _analyze_document_characteristics(self, content: str) -> Dict[str, Any]:
        import re
        """Analyze document to determine optimal chunking approach."""
        analysis = {
            "content_length": len(content),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            "sentence_count": len([s for s in content.split('.') if s.strip()]),
            "has_latex": bool(re.search(r'\\[a-zA-Z]+', content)),
            "has_code": bool(re.search(r'```|`[^`]+`', content)),
            "has_tables": '|' in content or 'table' in content.lower(),
            "has_lists": bool(re.search(r'^\s*[-*+]\s+', content, re.MULTILINE)),
            "avg_paragraph_length": 0,
            "structural_complexity": 0,
            "semantic_density": 0,
            "confidence": 0.8
        }
        
        # Calculate average paragraph length
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if paragraphs:
            analysis["avg_paragraph_length"] = sum(len(p) for p in paragraphs) / len(paragraphs)
        
        # Assess structural complexity
        complexity_score = 0
        if analysis["has_latex"]: complexity_score += 0.3
        if analysis["has_code"]: complexity_score += 0.2
        if analysis["has_tables"]: complexity_score += 0.2
        if analysis["paragraph_count"] > 10: complexity_score += 0.2
        if analysis["avg_paragraph_length"] > 500: complexity_score += 0.1
        
        analysis["structural_complexity"] = min(complexity_score, 1.0)
        
        return analysis
    
    def _select_optimal_strategy(self, analysis: Dict[str, Any], config: Dict[str, Any]) -> ChunkingStrategy:
        """Select optimal chunking strategy based on analysis."""
        # LaTeX documents
        if analysis["has_latex"]:
            return ChunkingStrategy.LATEX_AWARE
        
        # Code-heavy documents
        if analysis["has_code"] and analysis["structural_complexity"] > 0.6:
            return ChunkingStrategy.ADAPTIVE
        
        # Long documents with clear structure
        if analysis["content_length"] > 5000 and analysis["paragraph_count"] > 10:
            return ChunkingStrategy.HIERARCHICAL
        
        # Documents with high semantic density
        if analysis["structural_complexity"] > 0.4:
            return ChunkingStrategy.SEMANTIC
        
        # Well-structured documents
        if analysis["avg_paragraph_length"] > 200 and analysis["paragraph_count"] > 3:
            return ChunkingStrategy.PARAGRAPH_AWARE
        
        # Default to sentence-aware for most cases
        return ChunkingStrategy.SENTENCE_AWARE


class ContentAnalyzerNode(BaseWorkflowNode):
    """Node for deep content analysis and preprocessing."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Perform comprehensive content analysis."""
        document_content = state["document_content"]
        
        # Perform content analysis
        content_analysis = await self._analyze_content_structure(document_content)
        
        # Detect content boundaries
        boundaries = await self._detect_content_boundaries(document_content, content_analysis)
        
        # Identify orphan content
        orphan_analysis = await self._identify_orphan_content(document_content, boundaries)
        
        # Update state
        state["content_analysis"] = content_analysis
        state["content_boundaries"] = boundaries
        state["orphan_analysis"] = orphan_analysis
        
        self.logger.info(f"Content analysis complete: {len(boundaries)} boundaries detected, {len(orphan_analysis.get('orphans', []))} orphan segments")
        
        return state
    
    async def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structural elements of the content."""
        import re
        
        structure = {
            "sections": [],
            "paragraphs": [],
            "sentences": [],
            "special_elements": [],
            "metadata": {}
        }
        
        # Detect sections (headers, LaTeX sections, etc.)
        header_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^\s*\d+\.\s+(.+)$',  # Numbered sections
            r'\\section\{([^}]*)\}',  # LaTeX sections
            r'\\subsection\{([^}]*)\}',  # LaTeX subsections
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for pattern in header_patterns:
                match = re.search(pattern, line, re.MULTILINE)
                if match:
                    structure["sections"].append({
                        "line_number": i + 1,
                        "title": match.group(1) if match.groups() else line.strip(),
                        "type": "header",
                        "level": self._determine_header_level(line)
                    })
        
        # Analyze paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for i, para in enumerate(paragraphs):
            structure["paragraphs"].append({
                "index": i,
                "length": len(para),
                "word_count": len(para.split()),
                "has_special_content": self._has_special_content(para)
            })
        
        return structure
    
    def _determine_header_level(self, line: str) -> int:
        """Determine the hierarchical level of a header."""
        if line.startswith('#'):
            return len(line) - len(line.lstrip('#'))
        elif re.match(r'^\s*\d+\.', line):
            return 1
        elif '\\section{' in line:
            return 1
        elif '\\subsection{' in line:
            return 2
        return 0
    
    def _has_special_content(self, text: str) -> bool:
        """Check if text contains special content like code, tables, etc."""
        special_patterns = [
            r'```',  # Code blocks
            r'\|.*\|',  # Tables
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'def |class |import ',  # Code keywords
        ]
        
        return any(re.search(pattern, text) for pattern in special_patterns)
    
    async def _detect_content_boundaries(self, content: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect natural content boundaries for chunking."""
        boundaries = []
        
        # Section boundaries
        for section in analysis["sections"]:
            boundaries.append({
                "type": "section",
                "position": section["line_number"],
                "priority": "high",
                "metadata": section
            })
        
        # Paragraph boundaries
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if not line.strip() and i > 0 and i < len(lines) - 1:
                if lines[i-1].strip() and lines[i+1].strip():
                    boundaries.append({
                        "type": "paragraph",
                        "position": i + 1,
                        "priority": "medium",
                        "metadata": {}
                    })
        
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x["position"])
        
        return boundaries
    
    async def _identify_orphan_content(self, content: str, boundaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify content that might become orphaned during chunking."""
        lines = content.split('\n')
        orphan_analysis = {
            "orphans": [],
            "potential_orphans": [],
            "recommendations": []
        }
        
        # Look for very short segments between boundaries
        for i in range(len(boundaries) - 1):
            start_pos = boundaries[i]["position"]
            end_pos = boundaries[i + 1]["position"]
            
            segment_lines = lines[start_pos:end_pos]
            segment_content = '\n'.join(segment_lines).strip()
            
            if segment_content and len(segment_content) < 50:  # Very short segments
                orphan_analysis["orphans"].append({
                    "content": segment_content,
                    "start_line": start_pos,
                    "end_line": end_pos,
                    "length": len(segment_content),
                    "type": "short_segment"
                })
        
        return orphan_analysis


class AdaptiveChunkerNode(BaseWorkflowNode):
    """Node for adaptive chunking with dynamic strategy adjustment."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        super().__init__(config, state_manager)
        self.chunking_processor = ChunkingProcessor()
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Perform adaptive chunking based on analysis results."""
        document_content = state["document_content"]
        document_path = state["document_path"]
        chunking_strategy = ChunkingStrategy(state["chunking_strategy"])
        content_analysis = state.get("content_analysis", {})
        
        # Create parsed document
        from .document_parser import DocumentMetadata
        metadata = DocumentMetadata(
            file_path=document_path,
            file_size=len(document_content),
            content_type="text",
            encoding="utf-8"
        )
        
        parsed_doc = ParsedDocument(
            content=document_content,
            metadata=metadata,
            sections=content_analysis.get("sections", [])
        )
        
        # Configure chunking based on analysis
        chunking_config = self._create_adaptive_config(chunking_strategy, content_analysis, state)
        
        # Perform chunking
        chunking_result = await self.chunking_processor.process_document_async(parsed_doc, chunking_config)
        
        # Update state with results
        state["chunks"] = [self._chunk_to_dict(chunk) for chunk in chunking_result.chunks]
        state["chunking_metrics"] = {
            "total_chunks": chunking_result.total_chunks,
            "total_characters": chunking_result.total_characters,
            "average_chunk_size": chunking_result.average_chunk_size,
            "processing_time": chunking_result.processing_time,
            "quality_metrics": chunking_result.quality_metrics
        }
        
        self.logger.info(f"Adaptive chunking complete: {chunking_result.total_chunks} chunks created in {chunking_result.processing_time:.2f}s")
        
        return state
    
    def _create_adaptive_config(self, strategy: ChunkingStrategy, analysis: Dict[str, Any], state: WorkflowState) -> ChunkingConfig:
        """Create adaptive chunking configuration."""
        base_config = state.get("processing_config", {})
        
        # Adjust chunk size based on content characteristics
        content_length = analysis.get("content_length", 1000)
        avg_paragraph_length = analysis.get("avg_paragraph_length", 200)
        
        if content_length > 10000:  # Large documents
            max_chunk_size = min(1500, int(avg_paragraph_length * 1.5))
        elif content_length < 2000:  # Small documents
            max_chunk_size = max(500, int(avg_paragraph_length * 0.8))
        else:
            max_chunk_size = 1000
        
        # Adjust overlap based on content type
        if analysis.get("has_code", False):
            overlap_size = 1  # Minimal overlap for code
            overlap_strategy = OverlapStrategy.FIXED_SENTENCES
        elif analysis.get("structural_complexity", 0) > 0.5:
            overlap_size = 3  # More overlap for complex content
            overlap_strategy = OverlapStrategy.FIXED_SENTENCES
        else:
            overlap_size = 2  # Standard overlap
            overlap_strategy = OverlapStrategy.FIXED_SENTENCES
        
        return ChunkingConfig(
            strategy=strategy,
            max_chunk_size=max_chunk_size,
            min_chunk_size=max(100, max_chunk_size // 10),
            overlap_strategy=overlap_strategy,
            overlap_size=overlap_size,
            preserve_formatting=True,
            respect_boundaries=True,
            quality_threshold=0.6,
            enable_metadata=True
        )
    
    def _chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert DocumentChunk to dictionary for state storage."""
        return {
            "content": chunk.content,
            "metadata": {
                "chunk_id": chunk.metadata.chunk_id,
                "index": chunk.metadata.index,
                "start_position": chunk.metadata.start_position,
                "end_position": chunk.metadata.end_position,
                "character_count": chunk.metadata.character_count,
                "word_count": chunk.metadata.word_count,
                "sentence_count": chunk.metadata.sentence_count,
                "quality_score": chunk.metadata.quality_score,
                "contains_tables": chunk.metadata.contains_tables,
                "contains_lists": chunk.metadata.contains_lists,
                "contains_code": chunk.metadata.contains_code
            },
            "source_document": chunk.source_document,
            "hash": chunk.hash
        }


class EnhancedChunkingProcessor:
    """Enhanced chunking processor with LangGraph workflow integration."""
    
    def __init__(self, config: Optional[EnhancedChunkingConfig] = None, 
                 state_manager: Optional[CentralizedStateManager] = None,
                 llm=None, semantic_mapper=None):
        self.config = config or EnhancedChunkingConfig()
        self.state_manager = state_manager or CentralizedStateManager()
        self.workflow_graph = None
        self.memory_saver = MemorySaver()
        self.llm = llm
        self.semantic_mapper = semantic_mapper or BasicSemanticMapper()
        
        if self.config.enable_workflow:
            self._build_workflow_graph()
        
        logger.info(f"EnhancedChunkingProcessor initialized with workflow: {self.config.enable_workflow}")
    
    def _build_workflow_graph(self) -> None:
        """Build the LangGraph workflow for enhanced chunking with template processing."""
        workflow = StateGraph(WorkflowState)
        
        # Create workflow nodes
        strategy_node = StrategySelectionNode(
            NodeConfig(name="strategy_selector", max_retries=2),
            self.state_manager
        )
        
        content_node = ContentAnalyzerNode(
            NodeConfig(name="content_analyzer", max_retries=2),
            self.state_manager
        )
        
        chunker_node = AdaptiveChunkerNode(
            NodeConfig(name="adaptive_chunker", max_retries=3),
            self.state_manager
        )
        
        # Create template processor node if components are available
        template_processor = None
        if self.llm and self.semantic_mapper:
            template_processor = TemplateProcessorNode(
                llm=self.llm,
                semantic_mapper=self.semantic_mapper
            )
        
        # Add nodes to workflow
        workflow.add_node("strategy_selector", strategy_node.execute)
        workflow.add_node("content_analyzer", content_node.execute)
        workflow.add_node("adaptive_chunker", chunker_node.execute)
        
        if template_processor:
            workflow.add_node("template_processor", template_processor.process_document_with_template)
        
        # Define workflow edges
        workflow.add_edge("strategy_selector", "content_analyzer")
        workflow.add_edge("content_analyzer", "adaptive_chunker")
        
        if template_processor:
            workflow.add_edge("adaptive_chunker", "template_processor")
            workflow.add_edge("template_processor", END)
        else:
            workflow.add_edge("adaptive_chunker", END)
        
        # Set entry point
        workflow.set_entry_point("strategy_selector")
        
        # Compile workflow
        self.workflow_graph = workflow.compile(checkpointer=self.memory_saver)
        
        logger.info("LangGraph workflow with template processing compiled successfully")
    
    async def process_document_async(self, document: ParsedDocument, 
                                   config: Optional[EnhancedChunkingConfig] = None) -> ChunkingResult:
        """Process document using enhanced workflow."""
        config = config or self.config
        
        if not config.enable_workflow or not self.workflow_graph:
            # Fallback to basic processing
            base_processor = ChunkingProcessor(config)
            return await base_processor.process_document_async(document, config)
        
        # Prepare workflow state
        initial_state = {
            "document_content": document.content,
            "document_path": str(document.metadata.file_path),
            "chunks": [],
            "knowledge_graph": {},
            "semantic_mappings": [],
            "llm_outputs": [],
            "final_output": {},
            "current_stage": "chunking",
            "progress": 0.0,
            "session_id": hashlib.md5(f"{document.metadata.file_path}_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "processing_config": config,
            "errors": [],
            "retry_counts": {},
            "max_retries": {"default": 3},
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
            # Execute workflow
            result_state = await self.workflow_graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": initial_state["session_id"]}}
            )
            
            # Convert result to ChunkingResult
            chunks = [self._dict_to_chunk(chunk_dict) for chunk_dict in result_state.get("chunks", [])]
            metrics = result_state.get("chunking_metrics", {})
            
            return ChunkingResult(
                chunks=chunks,
                total_chunks=metrics.get("total_chunks", len(chunks)),
                total_characters=metrics.get("total_characters", sum(len(c.content) for c in chunks)),
                total_words=metrics.get("total_words", sum(c.metadata.word_count for c in chunks)),
                average_chunk_size=metrics.get("average_chunk_size", 0),
                chunking_strategy=ChunkingStrategy(result_state.get("chunking_strategy", "sentence_aware")),
                overlap_strategy=config.overlap_strategy,
                processing_time=metrics.get("processing_time", 0),
                quality_metrics=metrics.get("quality_metrics", {}),
                warnings=result_state.get("errors", []),
                metadata={
                    "workflow_used": True,
                    "session_id": result_state["session_id"],
                    "stage_timings": result_state.get("stage_timings", {})
                }
            )
            
        except Exception as e:
            logger.error(f"Enhanced chunking workflow failed: {e}")
            # Fallback to basic processing
            base_processor = ChunkingProcessor(config)
            return await base_processor.process_document_async(document, config)
    
    def _dict_to_chunk(self, chunk_dict: Dict[str, Any]) -> DocumentChunk:
        """Convert dictionary back to DocumentChunk."""
        metadata_dict = chunk_dict["metadata"]
        metadata = ChunkMetadata(
            chunk_id=metadata_dict["chunk_id"],
            index=metadata_dict["index"],
            start_position=metadata_dict["start_position"],
            end_position=metadata_dict["end_position"],
            character_count=metadata_dict["character_count"],
            word_count=metadata_dict["word_count"],
            sentence_count=metadata_dict["sentence_count"],
            paragraph_count=metadata_dict.get("paragraph_count", 1),
            quality_score=metadata_dict["quality_score"],
            contains_tables=metadata_dict["contains_tables"],
            contains_lists=metadata_dict["contains_lists"],
            contains_code=metadata_dict["contains_code"]
        )
        
        return DocumentChunk(
            content=chunk_dict["content"],
            metadata=metadata,
            source_document=chunk_dict["source_document"],
            hash=chunk_dict["hash"]
        )


# Factory functions
def create_enhanced_chunking_processor(enable_workflow: bool = True,
                                     enable_adaptive: bool = True,
                                     llm=None,
                                     semantic_mapper=None,
                                     **kwargs) -> EnhancedChunkingProcessor:
    """Create enhanced chunking processor with specified configuration."""
    config = EnhancedChunkingConfig(
        enable_workflow=enable_workflow,
        enable_adaptive_strategy=enable_adaptive,
        **kwargs
    )
    if semantic_mapper is None:
        semantic_mapper = BasicSemanticMapper()
    return EnhancedChunkingProcessor(config, llm=llm, semantic_mapper=semantic_mapper)


async def enhanced_chunk_document_async(document: ParsedDocument,
                                      strategy: Optional[ChunkingStrategy] = None,
                                      enable_workflow: bool = True) -> ChunkingResult:
    """Convenience function for enhanced async document chunking."""
    config = EnhancedChunkingConfig(
        strategy=strategy or ChunkingStrategy.ADAPTIVE,
        enable_workflow=enable_workflow
    )
    processor = EnhancedChunkingProcessor(config)
    return await processor.process_document_async(document, config)