"""Enhanced Aggregation Engine with LangGraph Integration

This module provides intelligent content aggregation capabilities with LangGraph workflow
orchestration for sophisticated synthesis, hierarchical aggregation, context-aware merging,
and adaptive aggregation strategies based on content analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone
import json
import hashlib
import numpy as np
from collections import defaultdict, Counter

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Internal imports
from .chunking_processor import DocumentChunk, ChunkMetadata
from .enhanced_semantic_mapper import SemanticMapping, SemanticCluster
from .langgraph_orchestrator import BaseWorkflowNode, NodeConfig, WorkflowState, NodeResult
from .state_manager import CentralizedStateManager, ProcessingStage, ErrorSeverity
from .llm_handler import LLMProvider, LLMConfig, ProcessingResult

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategies for content aggregation."""
    HIERARCHICAL = "hierarchical"
    THEMATIC = "thematic"
    SEQUENTIAL = "sequential"
    IMPORTANCE_BASED = "importance_based"
    SIMILARITY_BASED = "similarity_based"
    CONTEXT_AWARE = "context_aware"
    ADAPTIVE = "adaptive"
    CUSTOM = "custom"


class AggregationNodeType(Enum):
    """Types of aggregation workflow nodes."""
    STRATEGY_SELECTOR = "strategy_selector"
    CONTENT_ANALYZER = "content_analyzer"
    IMPORTANCE_RANKER = "importance_ranker"
    CONTEXT_BUILDER = "context_builder"
    CONTENT_SYNTHESIZER = "content_synthesizer"
    HIERARCHY_ORGANIZER = "hierarchy_organizer"
    QUALITY_VALIDATOR = "quality_validator"
    OUTPUT_FORMATTER = "output_formatter"


@dataclass
class AggregatedContent:
    """Represents aggregated content with metadata."""
    content: str
    source_chunk_ids: List[str]
    aggregation_strategy: AggregationStrategy
    confidence_score: float
    coherence_score: float
    completeness_score: float
    importance_score: float
    synthesis_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "source_chunk_ids": self.source_chunk_ids,
            "aggregation_strategy": self.aggregation_strategy.value,
            "confidence_score": self.confidence_score,
            "coherence_score": self.coherence_score,
            "completeness_score": self.completeness_score,
            "importance_score": self.importance_score,
            "synthesis_method": self.synthesis_method,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class AggregationGroup:
    """Represents a group of content for aggregation."""
    group_id: str
    chunk_ids: List[str]
    theme: str
    priority: float
    aggregation_strategy: AggregationStrategy
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "group_id": self.group_id,
            "chunk_ids": self.chunk_ids,
            "theme": self.theme,
            "priority": self.priority,
            "aggregation_strategy": self.aggregation_strategy.value,
            "context": self.context
        }


@dataclass
class EnhancedAggregationConfig:
    """Configuration for enhanced aggregation engine."""
    enable_workflow: bool = True
    enable_llm_synthesis: bool = True
    enable_hierarchical_organization: bool = True
    enable_context_awareness: bool = True
    enable_quality_validation: bool = True
    
    # Aggregation strategies
    primary_strategy: AggregationStrategy = AggregationStrategy.ADAPTIVE
    fallback_strategy: AggregationStrategy = AggregationStrategy.SIMILARITY_BASED
    
    # Content thresholds
    min_aggregation_size: int = 2
    max_aggregation_size: int = 10
    similarity_threshold: float = 0.6
    importance_threshold: float = 0.5
    coherence_threshold: float = 0.7
    
    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 2000
    temperature: float = 0.3
    
    # Synthesis parameters
    preserve_original_structure: bool = True
    maintain_source_references: bool = True
    enable_cross_references: bool = True
    synthesis_style: str = "comprehensive"  # comprehensive, concise, detailed
    
    # Performance optimization
    parallel_processing: bool = True
    batch_processing_size: int = 5
    cache_synthesis_results: bool = True
    max_synthesis_retries: int = 3


class AggregationStrategySelector(BaseWorkflowNode):
    """Node for selecting optimal aggregation strategy."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Analyze content and select optimal aggregation strategy."""
        chunks = state.get("chunks", [])
        semantic_relationships = state.get("semantic_relationships", [])
        semantic_clusters = state.get("semantic_clusters", [])
        aggregation_config = state.get("aggregation_config", {})
        
        # Analyze content characteristics
        content_analysis = await self._analyze_content_for_aggregation(chunks, semantic_relationships, semantic_clusters)
        
        # Select optimal strategy
        selected_strategy = self._select_optimal_strategy(content_analysis, aggregation_config)
        
        # Create aggregation groups
        aggregation_groups = await self._create_aggregation_groups(chunks, semantic_relationships, semantic_clusters, selected_strategy)
        
        # Update state
        state["aggregation_strategy"] = selected_strategy.value
        state["content_analysis"] = content_analysis
        state["aggregation_groups"] = [group.to_dict() for group in aggregation_groups]
        state["strategy_confidence"] = content_analysis.get("strategy_confidence", 0.8)
        
        self.logger.info(f"Selected aggregation strategy: {selected_strategy.name}, created {len(aggregation_groups)} groups")
        
        return state
    
    async def _analyze_content_for_aggregation(self, chunks: List[Dict[str, Any]], 
                                             relationships: List[Dict[str, Any]], 
                                             clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content characteristics for aggregation strategy selection."""
        analysis = {
            "total_chunks": len(chunks),
            "total_relationships": len(relationships),
            "total_clusters": len(clusters),
            "avg_chunk_size": 0,
            "content_diversity": 0,
            "relationship_density": 0,
            "cluster_coherence": 0,
            "has_hierarchical_structure": False,
            "has_sequential_flow": False,
            "has_thematic_groups": False,
            "complexity_score": 0,
            "strategy_confidence": 0.8
        }
        
        if chunks:
            # Calculate average chunk size
            chunk_sizes = [chunk["metadata"]["character_count"] for chunk in chunks]
            analysis["avg_chunk_size"] = np.mean(chunk_sizes)
            
            # Assess content diversity
            unique_words = set()
            total_words = 0
            for chunk in chunks:
                words = chunk["content"].lower().split()
                unique_words.update(words)
                total_words += len(words)
            
            analysis["content_diversity"] = len(unique_words) / total_words if total_words > 0 else 0
        
        if relationships:
            # Calculate relationship density
            max_possible_relationships = len(chunks) * (len(chunks) - 1) / 2
            analysis["relationship_density"] = len(relationships) / max_possible_relationships if max_possible_relationships > 0 else 0
            
            # Check for hierarchical patterns
            hierarchical_types = ["hierarchical", "definitional"]
            hierarchical_count = sum(1 for rel in relationships if rel.get("mapping_type") in hierarchical_types)
            analysis["has_hierarchical_structure"] = hierarchical_count / len(relationships) > 0.3
            
            # Check for sequential patterns
            sequential_types = ["sequential", "causal"]
            sequential_count = sum(1 for rel in relationships if rel.get("mapping_type") in sequential_types)
            analysis["has_sequential_flow"] = sequential_count / len(relationships) > 0.2
        
        if clusters:
            # Calculate average cluster coherence
            coherence_scores = [cluster.get("coherence_score", 0) for cluster in clusters]
            analysis["cluster_coherence"] = np.mean(coherence_scores) if coherence_scores else 0
            analysis["has_thematic_groups"] = len(clusters) > 1 and analysis["cluster_coherence"] > 0.6
        
        # Calculate overall complexity
        complexity_factors = [
            analysis["content_diversity"],
            analysis["relationship_density"],
            min(analysis["total_clusters"] / max(len(chunks) / 5, 1), 1.0),  # Normalized cluster count
            0.5 if analysis["has_hierarchical_structure"] else 0,
            0.3 if analysis["has_sequential_flow"] else 0
        ]
        analysis["complexity_score"] = np.mean(complexity_factors)
        
        return analysis
    
    def _select_optimal_strategy(self, analysis: Dict[str, Any], config: Dict[str, Any]) -> AggregationStrategy:
        """Select optimal aggregation strategy based on content analysis."""
        # Check for hierarchical structure
        if analysis["has_hierarchical_structure"] and analysis["cluster_coherence"] > 0.7:
            return AggregationStrategy.HIERARCHICAL
        
        # Check for strong thematic grouping
        if analysis["has_thematic_groups"] and analysis["total_clusters"] >= 3:
            return AggregationStrategy.THEMATIC
        
        # Check for sequential flow
        if analysis["has_sequential_flow"] and analysis["relationship_density"] > 0.4:
            return AggregationStrategy.SEQUENTIAL
        
        # High complexity content
        if analysis["complexity_score"] > 0.6:
            return AggregationStrategy.CONTEXT_AWARE
        
        # High relationship density
        if analysis["relationship_density"] > 0.5:
            return AggregationStrategy.SIMILARITY_BASED
        
        # Default to adaptive for most cases
        return AggregationStrategy.ADAPTIVE
    
    async def _create_aggregation_groups(self, chunks: List[Dict[str, Any]], 
                                       relationships: List[Dict[str, Any]], 
                                       clusters: List[Dict[str, Any]], 
                                       strategy: AggregationStrategy) -> List[AggregationGroup]:
        """Create aggregation groups based on selected strategy."""
        groups = []
        
        if strategy == AggregationStrategy.THEMATIC and clusters:
            # Group by semantic clusters
            for i, cluster in enumerate(clusters):
                if cluster["size"] >= 2:  # Only include clusters with multiple chunks
                    groups.append(AggregationGroup(
                        group_id=f"thematic_{i}",
                        chunk_ids=cluster["chunk_ids"],
                        theme=cluster.get("cluster_theme", f"Theme {i+1}"),
                        priority=cluster.get("coherence_score", 0.5),
                        aggregation_strategy=strategy,
                        context={"cluster_info": cluster}
                    ))
        
        elif strategy == AggregationStrategy.SIMILARITY_BASED and relationships:
            # Group by strong relationships
            relationship_graph = defaultdict(set)
            for rel in relationships:
                if rel["confidence_score"] > 0.7:
                    relationship_graph[rel["source_chunk_id"]].add(rel["target_chunk_id"])
                    relationship_graph[rel["target_chunk_id"]].add(rel["source_chunk_id"])
            
            # Find connected components
            visited = set()
            group_id = 0
            
            for chunk_id in relationship_graph:
                if chunk_id not in visited:
                    # BFS to find connected component
                    component = set()
                    queue = [chunk_id]
                    
                    while queue:
                        current = queue.pop(0)
                        if current not in visited:
                            visited.add(current)
                            component.add(current)
                            queue.extend(relationship_graph[current] - visited)
                    
                    if len(component) >= 2:
                        groups.append(AggregationGroup(
                            group_id=f"similarity_{group_id}",
                            chunk_ids=list(component),
                            theme=f"Related Content {group_id+1}",
                            priority=0.7,
                            aggregation_strategy=strategy
                        ))
                        group_id += 1
        
        else:
            # Default grouping: sequential pairs or small groups
            chunk_ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
            
            for i in range(0, len(chunk_ids), 3):  # Group in sets of 3
                group_chunk_ids = chunk_ids[i:i+3]
                if len(group_chunk_ids) >= 2:
                    groups.append(AggregationGroup(
                        group_id=f"default_{i//3}",
                        chunk_ids=group_chunk_ids,
                        theme=f"Content Group {i//3 + 1}",
                        priority=0.5,
                        aggregation_strategy=strategy
                    ))
        
        return groups


class ContentSynthesizerNode(BaseWorkflowNode):
    """Node for synthesizing content using LLM."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        super().__init__(config, state_manager)
        self.synthesis_cache = {}
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Synthesize content for each aggregation group."""
        chunks = state.get("chunks", [])
        aggregation_groups = state.get("aggregation_groups", [])
        aggregation_config = state.get("aggregation_config", {})
        
        # Create chunk lookup
        chunk_lookup = {chunk["metadata"]["chunk_id"]: chunk for chunk in chunks}
        
        # Synthesize content for each group
        synthesized_contents = []
        
        for group_dict in aggregation_groups:
            group = AggregationGroup(
                group_id=group_dict["group_id"],
                chunk_ids=group_dict["chunk_ids"],
                theme=group_dict["theme"],
                priority=group_dict["priority"],
                aggregation_strategy=AggregationStrategy(group_dict["aggregation_strategy"]),
                context=group_dict.get("context", {})
            )
            
            synthesized = await self._synthesize_group_content(group, chunk_lookup, aggregation_config)
            if synthesized:
                synthesized_contents.append(synthesized.to_dict())
        
        # Update state
        state["synthesized_contents"] = synthesized_contents
        state["synthesis_stats"] = {
            "total_groups_processed": len(aggregation_groups),
            "successful_syntheses": len(synthesized_contents),
            "avg_confidence": np.mean([content["confidence_score"] for content in synthesized_contents]) if synthesized_contents else 0,
            "avg_coherence": np.mean([content["coherence_score"] for content in synthesized_contents]) if synthesized_contents else 0
        }
        
        self.logger.info(f"Synthesized content for {len(synthesized_contents)} groups")
        
        return state
    
    async def _synthesize_group_content(self, group: AggregationGroup, 
                                      chunk_lookup: Dict[str, Dict[str, Any]], 
                                      config: Dict[str, Any]) -> Optional[AggregatedContent]:
        """Synthesize content for a single aggregation group."""
        try:
            # Gather source content
            source_contents = []
            for chunk_id in group.chunk_ids:
                if chunk_id in chunk_lookup:
                    source_contents.append(chunk_lookup[chunk_id]["content"])
            
            if not source_contents:
                return None
            
            # Check cache
            cache_key = hashlib.md5("|".join(sorted(source_contents)).encode()).hexdigest()
            if cache_key in self.synthesis_cache and config.get("cache_synthesis_results", True):
                cached_result = self.synthesis_cache[cache_key]
                return AggregatedContent(**cached_result)
            
            # Perform synthesis
            synthesized_content = await self._perform_llm_synthesis(source_contents, group, config)
            
            if not synthesized_content:
                return None
            
            # Calculate quality scores
            quality_scores = await self._calculate_quality_scores(synthesized_content, source_contents)
            
            # Create aggregated content object
            aggregated = AggregatedContent(
                content=synthesized_content,
                source_chunk_ids=group.chunk_ids,
                aggregation_strategy=group.aggregation_strategy,
                confidence_score=quality_scores["confidence"],
                coherence_score=quality_scores["coherence"],
                completeness_score=quality_scores["completeness"],
                importance_score=group.priority,
                synthesis_method="llm",
                metadata={
                    "group_theme": group.theme,
                    "source_count": len(source_contents),
                    "synthesis_timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Cache the result
            if config.get("cache_synthesis_results", True):
                self.synthesis_cache[cache_key] = {
                    "content": synthesized_content,
                    "source_chunk_ids": group.chunk_ids,
                    "aggregation_strategy": group.aggregation_strategy,
                    "confidence_score": quality_scores["confidence"],
                    "coherence_score": quality_scores["coherence"],
                    "completeness_score": quality_scores["completeness"],
                    "importance_score": group.priority,
                    "synthesis_method": "llm"
                }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error synthesizing content for group {group.group_id}: {e}")
            return None
    
    async def _perform_llm_synthesis(self, source_contents: List[str], 
                                   group: AggregationGroup, 
                                   config: Dict[str, Any]) -> Optional[str]:
        """Perform LLM-based content synthesis."""
        # Mock LLM synthesis (replace with actual LLM call)
        try:
            # Prepare synthesis prompt
            synthesis_style = config.get("synthesis_style", "comprehensive")
            
            if synthesis_style == "comprehensive":
                instruction = "Synthesize the following content pieces into a comprehensive, well-structured summary that captures all key information:"
            elif synthesis_style == "concise":
                instruction = "Create a concise synthesis of the following content, focusing on the most important points:"
            else:  # detailed
                instruction = "Provide a detailed synthesis that elaborates on the relationships and connections between these content pieces:"
            
            # Combine source contents
            combined_content = "\n\n---\n\n".join(source_contents)
            
            # Mock synthesis (replace with actual LLM call)
            synthesized = await self._mock_llm_synthesis(instruction, combined_content, group.theme)
            
            return synthesized
            
        except Exception as e:
            self.logger.error(f"LLM synthesis failed: {e}")
            return None
    
    async def _mock_llm_synthesis(self, instruction: str, content: str, theme: str) -> str:
        """Mock LLM synthesis (replace with actual implementation)."""
        # Simple extractive synthesis
        sentences = []
        for part in content.split("\n\n---\n\n"):
            part_sentences = [s.strip() for s in part.split('.') if s.strip()]
            # Take first 2 sentences from each part
            sentences.extend(part_sentences[:2])
        
        # Create synthesis
        synthesis_parts = [
            f"## {theme}\n",
            "This section synthesizes the following key information:\n",
            "\n".join(f"• {sentence}." for sentence in sentences[:5]),
            "\n\nThe content demonstrates interconnected concepts that build upon each other to provide a comprehensive understanding of the topic."
        ]
        
        return "".join(synthesis_parts)
    
    async def _calculate_quality_scores(self, synthesized_content: str, source_contents: List[str]) -> Dict[str, float]:
        """Calculate quality scores for synthesized content."""
        # Mock quality calculation (replace with actual metrics)
        
        # Confidence: based on content length and structure
        confidence = min(1.0, len(synthesized_content) / 500)  # Normalize by expected length
        
        # Coherence: based on sentence structure and flow
        sentences = [s.strip() for s in synthesized_content.split('.') if s.strip()]
        coherence = min(1.0, len(sentences) / 5) * 0.8  # Basic coherence estimate
        
        # Completeness: based on coverage of source content
        source_words = set()
        for content in source_contents:
            source_words.update(content.lower().split())
        
        synth_words = set(synthesized_content.lower().split())
        completeness = len(synth_words.intersection(source_words)) / len(source_words) if source_words else 0
        
        return {
            "confidence": confidence,
            "coherence": coherence,
            "completeness": completeness
        }


class HierarchyOrganizerNode(BaseWorkflowNode):
    """Node for organizing synthesized content into hierarchical structure."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Organize synthesized content into hierarchical structure."""
        synthesized_contents = state.get("synthesized_contents", [])
        aggregation_strategy = state.get("aggregation_strategy", "adaptive")
        
        # Organize content hierarchically
        organized_structure = await self._organize_hierarchically(synthesized_contents, aggregation_strategy)
        
        # Update state
        state["organized_structure"] = organized_structure
        state["hierarchy_stats"] = {
            "total_sections": len(organized_structure.get("sections", [])),
            "max_depth": self._calculate_max_depth(organized_structure),
            "organization_method": aggregation_strategy
        }
        
        self.logger.info(f"Organized content into {len(organized_structure.get('sections', []))} hierarchical sections")
        
        return state
    
    async def _organize_hierarchically(self, contents: List[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
        """Organize content into hierarchical structure."""
        structure = {
            "title": "Aggregated Document",
            "sections": [],
            "metadata": {
                "organization_strategy": strategy,
                "total_content_pieces": len(contents),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        }
        
        # Sort contents by importance score
        sorted_contents = sorted(contents, key=lambda x: x["importance_score"], reverse=True)
        
        for i, content in enumerate(sorted_contents):
            section = {
                "id": f"section_{i+1}",
                "title": self._extract_title_from_content(content["content"]),
                "content": content["content"],
                "level": 1,  # All at top level for now
                "source_chunks": content["source_chunk_ids"],
                "quality_scores": {
                    "confidence": content["confidence_score"],
                    "coherence": content["coherence_score"],
                    "completeness": content["completeness_score"]
                },
                "metadata": content.get("metadata", {})
            }
            
            structure["sections"].append(section)
        
        return structure
    
    def _extract_title_from_content(self, content: str) -> str:
        """Extract or generate title from content."""
        lines = content.strip().split('\n')
        
        # Look for markdown headers
        for line in lines:
            if line.startswith('#'):
                return line.lstrip('#').strip()
        
        # Use first sentence as title
        first_sentence = content.split('.')[0].strip()
        if len(first_sentence) < 100:
            return first_sentence
        
        # Generate generic title
        return "Content Section"
    
    def _calculate_max_depth(self, structure: Dict[str, Any]) -> int:
        """Calculate maximum depth of hierarchical structure."""
        max_depth = 0
        
        def traverse(node, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "sections" and isinstance(value, list):
                        for section in value:
                            traverse(section, depth + 1)
                    elif isinstance(value, (dict, list)):
                        traverse(value, depth)
        
        traverse(structure, 0)
        return max_depth


class EnhancedAggregationEngine:
    """Enhanced aggregation engine with LangGraph workflow integration."""
    
    def __init__(self, config: Optional[EnhancedAggregationConfig] = None,
                 state_manager: Optional[CentralizedStateManager] = None):
        self.config = config or EnhancedAggregationConfig()
        self.state_manager = state_manager or CentralizedStateManager()
        self.workflow_graph = None
        self.memory_saver = MemorySaver()
        
        if self.config.enable_workflow:
            self._build_workflow_graph()
        
        logger.info(f"EnhancedAggregationEngine initialized with workflow: {self.config.enable_workflow}")
    
    def _build_workflow_graph(self) -> None:
        """Build the LangGraph workflow for content aggregation."""
        workflow = StateGraph(WorkflowState)
        
        # Create workflow nodes
        strategy_node = AggregationStrategySelector(
            NodeConfig(name="strategy_selector", max_retries=2),
            self.state_manager
        )
        
        synthesizer_node = ContentSynthesizerNode(
            NodeConfig(name="content_synthesizer", max_retries=3),
            self.state_manager
        )
        
        organizer_node = HierarchyOrganizerNode(
            NodeConfig(name="hierarchy_organizer", max_retries=2),
            self.state_manager
        )
        
        # Add nodes to workflow
        workflow.add_node("strategy_selector", strategy_node.execute)
        workflow.add_node("content_synthesizer", synthesizer_node.execute)
        workflow.add_node("hierarchy_organizer", organizer_node.execute)
        
        # Define workflow edges
        workflow.add_edge("strategy_selector", "content_synthesizer")
        workflow.add_edge("content_synthesizer", "hierarchy_organizer")
        workflow.add_edge("hierarchy_organizer", END)
        
        # Set entry point
        workflow.set_entry_point("strategy_selector")
        
        # Compile workflow
        self.workflow_graph = workflow.compile(checkpointer=self.memory_saver)
        
        logger.info("Aggregation workflow compiled successfully")
    
    async def aggregate_content_async(self, chunks: List[DocumentChunk], 
                                    semantic_mappings: Optional[Dict[str, Any]] = None,
                                    config: Optional[EnhancedAggregationConfig] = None) -> Dict[str, Any]:
        """Aggregate content using enhanced workflow."""
        config = config or self.config
        
        if not config.enable_workflow or not self.workflow_graph:
            # Fallback to basic aggregation
            return await self._basic_content_aggregation(chunks)
        
        # Convert chunks to dictionaries
        chunk_dicts = [self._chunk_to_dict(chunk) for chunk in chunks]
        
        # Prepare workflow state
        initial_state = {
            "chunks": chunk_dicts,
            "semantic_relationships": semantic_mappings.get("semantic_relationships", []) if semantic_mappings else [],
            "semantic_clusters": semantic_mappings.get("semantic_clusters", []) if semantic_mappings else [],
            "knowledge_graph": {},
            "semantic_mappings": [],
            "llm_outputs": [],
            "final_output": {},
            "current_stage": "aggregation",
            "progress": 0.0,
            "session_id": hashlib.md5(f"aggregation_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "aggregation_config": config.__dict__,
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
            
            # Extract results
            return {
                "aggregated_content": result_state.get("synthesized_contents", []),
                "organized_structure": result_state.get("organized_structure", {}),
                "aggregation_groups": result_state.get("aggregation_groups", []),
                "statistics": {
                    "synthesis_stats": result_state.get("synthesis_stats", {}),
                    "hierarchy_stats": result_state.get("hierarchy_stats", {}),
                    "content_analysis": result_state.get("content_analysis", {})
                },
                "metadata": {
                    "workflow_used": True,
                    "session_id": result_state["session_id"],
                    "aggregation_strategy": result_state.get("aggregation_strategy", "adaptive"),
                    "processing_time": result_state.get("stage_timings", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced aggregation workflow failed: {e}")
            # Fallback to basic aggregation
            return await self._basic_content_aggregation(chunks)
    
    def _chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert DocumentChunk to dictionary."""
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
                "quality_score": chunk.metadata.quality_score
            },
            "source_document": chunk.source_document,
            "hash": chunk.hash
        }
    
    async def _basic_content_aggregation(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Basic content aggregation fallback."""
        # Simple concatenation-based aggregation
        aggregated_sections = []
        
        # Group chunks in pairs
        for i in range(0, len(chunks), 2):
            chunk_group = chunks[i:i+2]
            
            combined_content = "\n\n".join([chunk.content for chunk in chunk_group])
            
            aggregated_sections.append({
                "content": combined_content,
                "source_chunk_ids": [chunk.metadata.chunk_id for chunk in chunk_group],
                "aggregation_strategy": "basic",
                "confidence_score": 0.6,
                "coherence_score": 0.5,
                "completeness_score": 0.7,
                "importance_score": 0.5,
                "synthesis_method": "concatenation"
            })
        
        return {
            "aggregated_content": aggregated_sections,
            "organized_structure": {
                "title": "Basic Aggregated Document",
                "sections": [{
                    "id": f"section_{i+1}",
                    "title": f"Section {i+1}",
                    "content": section["content"],
                    "level": 1
                } for i, section in enumerate(aggregated_sections)]
            },
            "statistics": {"total_sections": len(aggregated_sections)},
            "metadata": {"workflow_used": False}
        }


# Factory functions
def create_enhanced_aggregation_engine(enable_workflow: bool = True,
                                     enable_llm_synthesis: bool = True,
                                     **kwargs) -> EnhancedAggregationEngine:
    """Create enhanced aggregation engine with specified configuration."""
    config = EnhancedAggregationConfig(
        enable_workflow=enable_workflow,
        enable_llm_synthesis=enable_llm_synthesis,
        **kwargs
    )
    return EnhancedAggregationEngine(config)


async def aggregate_document_content_async(chunks: List[DocumentChunk],
                                         semantic_mappings: Optional[Dict[str, Any]] = None,
                                         strategy: Optional[AggregationStrategy] = None) -> Dict[str, Any]:
    """Convenience function for enhanced async content aggregation."""
    config = EnhancedAggregationConfig(
        primary_strategy=strategy or AggregationStrategy.ADAPTIVE
    )
    engine = EnhancedAggregationEngine(config)
    return await engine.aggregate_content_async(chunks, semantic_mappings, config)