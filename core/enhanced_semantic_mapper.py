"""Enhanced Semantic Mapping System with LangGraph Integration

This module provides advanced semantic mapping capabilities with LangGraph workflow
orchestration for intelligent content relationship analysis, embedding-guided mapping,
and hierarchical semantic structure discovery.
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
from .langgraph_orchestrator import BaseWorkflowNode, NodeConfig, WorkflowState, NodeResult
from .state_manager import CentralizedStateManager, ProcessingStage, ErrorSeverity
from .llm_handler import LLMProvider, LLMConfig, ProcessingResult

logger = logging.getLogger(__name__)


class SemanticMappingType(Enum):
    """Types of semantic mappings."""
    CONCEPTUAL = "conceptual"
    HIERARCHICAL = "hierarchical"
    SEQUENTIAL = "sequential"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    DEFINITIONAL = "definitional"
    EXEMPLIFICATION = "exemplification"
    ELABORATION = "elaboration"


class MappingStrategy(Enum):
    """Semantic mapping strategies."""
    EMBEDDING_BASED = "embedding_based"
    RELATIONSHIP_FOCUSED = "relationship_focused"
    CLUSTER_BASED = "cluster_based"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


class MappingNodeType(Enum):
    """Types of semantic mapping workflow nodes."""
    CONTENT_EMBEDDER = "content_embedder"
    SIMILARITY_ANALYZER = "similarity_analyzer"
    RELATIONSHIP_DETECTOR = "relationship_detector"
    HIERARCHY_BUILDER = "hierarchy_builder"
    SEMANTIC_CLUSTERER = "semantic_clusterer"
    CONTEXT_ENRICHER = "context_enricher"
    MAPPING_VALIDATOR = "mapping_validator"
    ORPHAN_MAPPER = "orphan_mapper"


@dataclass
class SemanticMapping:
    """Represents a semantic relationship between content chunks."""
    source_chunk_id: str
    target_chunk_id: str
    mapping_type: SemanticMappingType
    confidence_score: float
    relationship_strength: float
    semantic_distance: float
    context_overlap: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_chunk_id": self.source_chunk_id,
            "target_chunk_id": self.target_chunk_id,
            "mapping_type": self.mapping_type.value,
            "confidence_score": self.confidence_score,
            "relationship_strength": self.relationship_strength,
            "semantic_distance": self.semantic_distance,
            "context_overlap": self.context_overlap,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class SemanticCluster:
    """Represents a cluster of semantically related chunks."""
    cluster_id: str
    chunk_ids: List[str]
    centroid_embedding: Optional[np.ndarray]
    cluster_theme: str
    coherence_score: float
    size: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cluster_id": self.cluster_id,
            "chunk_ids": self.chunk_ids,
            "centroid_embedding": self.centroid_embedding.tolist() if self.centroid_embedding is not None else None,
            "cluster_theme": self.cluster_theme,
            "coherence_score": self.coherence_score,
            "size": self.size,
            "metadata": self.metadata
        }


@dataclass
class EnhancedSemanticMappingConfig:
    """Configuration for enhanced semantic mapping."""
    enable_workflow: bool = True
    enable_embeddings: bool = True
    enable_llm_analysis: bool = True
    enable_clustering: bool = True
    enable_hierarchy_detection: bool = True
    enable_relationship_detection: bool = True
    
    # Strategy configuration
    primary_strategy: MappingStrategy = MappingStrategy.HYBRID
    
    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    batch_size: int = 32
    
    # Similarity thresholds
    similarity_threshold: float = 0.7
    relationship_threshold: float = 0.6
    clustering_threshold: float = 0.75
    
    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.3
    
    # Processing limits
    max_mappings_per_chunk: int = 10
    max_cluster_size: int = 20
    min_cluster_size: int = 2
    
    # Performance optimization
    parallel_processing: bool = True
    cache_embeddings: bool = True
    use_approximate_search: bool = True


class ContentEmbedderNode(BaseWorkflowNode):
    """Node for generating content embeddings."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        super().__init__(config, state_manager)
        self.embedding_cache = {}
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Generate embeddings for all chunks."""
        chunks = state.get("chunks", [])
        mapping_config = state.get("mapping_config", {})
        
        # Generate embeddings
        embeddings = await self._generate_embeddings(chunks, mapping_config)
        
        # Update state
        state["chunk_embeddings"] = embeddings
        state["embedding_metadata"] = {
            "model": mapping_config.get("embedding_model", "default"),
            "dimension": mapping_config.get("embedding_dimension", 384),
            "total_chunks": len(chunks)
        }
        
        self.logger.info(f"Generated embeddings for {len(chunks)} chunks")
        
        return state
    
    async def _generate_embeddings(self, chunks: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate embeddings for chunk contents."""
        try:
            # Import sentence transformers (mock implementation)
            # In real implementation, use actual sentence-transformers library
            embeddings = {}
            
            for chunk in chunks:
                chunk_id = chunk["metadata"]["chunk_id"]
                content = chunk["content"]
                
                # Check cache first
                cache_key = hashlib.md5(content.encode()).hexdigest()
                if cache_key in self.embedding_cache:
                    embeddings[chunk_id] = self.embedding_cache[cache_key]
                    continue
                
                # Generate embedding (mock implementation)
                # In real implementation, use actual embedding model
                embedding = await self._mock_generate_embedding(content, config)
                embeddings[chunk_id] = embedding
                
                # Cache the result
                if config.get("cache_embeddings", True):
                    self.embedding_cache[cache_key] = embedding
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return {}
    
    async def _mock_generate_embedding(self, content: str, config: Dict[str, Any]) -> List[float]:
        """Mock embedding generation (replace with actual implementation)."""
        # Simple mock: use content hash to generate consistent "embedding"
        import hashlib
        hash_obj = hashlib.md5(content.encode())
        hash_bytes = hash_obj.digest()
        
        dimension = config.get("embedding_dimension", 384)
        embedding = []
        
        for i in range(dimension):
            byte_val = hash_bytes[i % len(hash_bytes)]
            normalized_val = (byte_val - 127.5) / 127.5  # Normalize to [-1, 1]
            embedding.append(normalized_val)
        
        return embedding


class SimilarityAnalyzerNode(BaseWorkflowNode):
    """Node for analyzing semantic similarities between chunks."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Analyze similarities between chunk embeddings."""
        embeddings = state.get("chunk_embeddings", {})
        chunks = state.get("chunks", [])
        mapping_config = state.get("mapping_config", {})
        
        # Calculate similarity matrix
        similarity_matrix = await self._calculate_similarity_matrix(embeddings)
        
        # Find similar pairs
        similar_pairs = await self._find_similar_pairs(similarity_matrix, mapping_config)
        
        # Update state
        state["similarity_matrix"] = similarity_matrix
        state["similar_pairs"] = similar_pairs
        state["similarity_stats"] = {
            "total_pairs": len(similar_pairs),
            "avg_similarity": np.mean([pair["similarity"] for pair in similar_pairs]) if similar_pairs else 0,
            "max_similarity": max([pair["similarity"] for pair in similar_pairs]) if similar_pairs else 0
        }
        
        self.logger.info(f"Found {len(similar_pairs)} similar chunk pairs")
        
        return state
    
    async def _calculate_similarity_matrix(self, embeddings: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate cosine similarity matrix between all embeddings."""
        chunk_ids = list(embeddings.keys())
        similarity_matrix = {}
        
        for i, chunk_id1 in enumerate(chunk_ids):
            similarity_matrix[chunk_id1] = {}
            embedding1 = np.array(embeddings[chunk_id1])
            
            for j, chunk_id2 in enumerate(chunk_ids):
                if i == j:
                    similarity_matrix[chunk_id1][chunk_id2] = 1.0
                elif chunk_id2 in similarity_matrix and chunk_id1 in similarity_matrix[chunk_id2]:
                    # Use already calculated similarity (symmetric)
                    similarity_matrix[chunk_id1][chunk_id2] = similarity_matrix[chunk_id2][chunk_id1]
                else:
                    embedding2 = np.array(embeddings[chunk_id2])
                    similarity = self._cosine_similarity(embedding1, embedding2)
                    similarity_matrix[chunk_id1][chunk_id2] = similarity
        
        return similarity_matrix
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _find_similar_pairs(self, similarity_matrix: Dict[str, Dict[str, float]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find pairs of chunks above similarity threshold."""
        threshold = config.get("similarity_threshold", 0.7)
        similar_pairs = []
        processed_pairs = set()
        
        for chunk_id1, similarities in similarity_matrix.items():
            for chunk_id2, similarity in similarities.items():
                if chunk_id1 != chunk_id2 and similarity >= threshold:
                    # Avoid duplicate pairs
                    pair_key = tuple(sorted([chunk_id1, chunk_id2]))
                    if pair_key not in processed_pairs:
                        similar_pairs.append({
                            "chunk_id1": chunk_id1,
                            "chunk_id2": chunk_id2,
                            "similarity": similarity,
                            "pair_key": pair_key
                        })
                        processed_pairs.add(pair_key)
        
        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar_pairs


class RelationshipDetectorNode(BaseWorkflowNode):
    """Node for detecting semantic relationships using LLM analysis."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        super().__init__(config, state_manager)
        self.llm_provider = None  # Initialize with actual LLM provider
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Detect semantic relationships between similar chunks."""
        similar_pairs = state.get("similar_pairs", [])
        chunks = state.get("chunks", [])
        mapping_config = state.get("mapping_config", {})
        
        # Create chunk lookup
        chunk_lookup = {chunk["metadata"]["chunk_id"]: chunk for chunk in chunks}
        
        # Analyze relationships
        relationships = await self._analyze_relationships(similar_pairs, chunk_lookup, mapping_config)
        
        # Update state
        state["semantic_relationships"] = relationships
        state["relationship_stats"] = {
            "total_relationships": len(relationships),
            "relationship_types": Counter([rel["mapping_type"] for rel in relationships]),
            "avg_confidence": np.mean([rel["confidence_score"] for rel in relationships]) if relationships else 0
        }
        
        self.logger.info(f"Detected {len(relationships)} semantic relationships")
        
        return state
    
    async def _analyze_relationships(self, similar_pairs: List[Dict[str, Any]], 
                                   chunk_lookup: Dict[str, Dict[str, Any]], 
                                   config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze semantic relationships using LLM."""
        relationships = []
        
        for pair in similar_pairs[:config.get("max_pairs_to_analyze", 50)]:
            chunk1 = chunk_lookup.get(pair["chunk_id1"])
            chunk2 = chunk_lookup.get(pair["chunk_id2"])
            
            if not chunk1 or not chunk2:
                continue
            
            # Analyze relationship
            relationship = await self._analyze_pair_relationship(chunk1, chunk2, pair["similarity"], config)
            if relationship:
                relationships.append(relationship)
        
        return relationships
    
    async def _analyze_pair_relationship(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any], 
                                       similarity: float, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze relationship between a pair of chunks."""
        try:
            # Mock LLM analysis (replace with actual LLM call)
            relationship_type, confidence = await self._mock_llm_relationship_analysis(
                chunk1["content"], chunk2["content"], similarity
            )
            
            if confidence >= config.get("relationship_threshold", 0.6):
                return {
                    "source_chunk_id": chunk1["metadata"]["chunk_id"],
                    "target_chunk_id": chunk2["metadata"]["chunk_id"],
                    "mapping_type": relationship_type,
                    "confidence_score": confidence,
                    "relationship_strength": similarity,
                    "semantic_distance": 1.0 - similarity,
                    "context_overlap": self._calculate_context_overlap(chunk1["content"], chunk2["content"]),
                    "metadata": {
                        "analysis_method": "llm",
                        "similarity_score": similarity
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing relationship: {e}")
            return None
    
    async def _mock_llm_relationship_analysis(self, content1: str, content2: str, similarity: float) -> Tuple[str, float]:
        """Mock LLM relationship analysis (replace with actual implementation)."""
        # Simple heuristic-based analysis
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        # Check for definitional relationships
        if any(word in content1_lower for word in ["define", "definition", "means", "refers to"]):
            return "definitional", 0.8
        
        # Check for sequential relationships
        if any(word in content1_lower for word in ["first", "then", "next", "finally", "step"]):
            return "sequential", 0.7
        
        # Check for comparative relationships
        if any(word in content1_lower for word in ["compare", "contrast", "versus", "different", "similar"]):
            return "comparative", 0.75
        
        # Check for causal relationships
        if any(word in content1_lower for word in ["because", "therefore", "result", "cause", "effect"]):
            return "causal", 0.8
        
        # Default to conceptual with confidence based on similarity
        confidence = min(0.9, similarity + 0.1)
        return "conceptual", confidence
    
    def _calculate_context_overlap(self, content1: str, content2: str) -> float:
        """Calculate context overlap between two content pieces."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class SemanticClustererNode(BaseWorkflowNode):
    """Node for clustering semantically related chunks."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Cluster chunks based on semantic similarity."""
        embeddings = state.get("chunk_embeddings", {})
        similarity_matrix = state.get("similarity_matrix", {})
        mapping_config = state.get("mapping_config", {})
        
        # Perform clustering
        clusters = await self._perform_clustering(embeddings, similarity_matrix, mapping_config)
        
        # Generate cluster themes
        enriched_clusters = await self._enrich_clusters_with_themes(clusters, state)
        
        # Update state
        state["semantic_clusters"] = [cluster.to_dict() for cluster in enriched_clusters]
        state["clustering_stats"] = {
            "total_clusters": len(enriched_clusters),
            "avg_cluster_size": np.mean([cluster.size for cluster in enriched_clusters]) if enriched_clusters else 0,
            "largest_cluster_size": max([cluster.size for cluster in enriched_clusters]) if enriched_clusters else 0,
            "avg_coherence": np.mean([cluster.coherence_score for cluster in enriched_clusters]) if enriched_clusters else 0
        }
        
        self.logger.info(f"Created {len(enriched_clusters)} semantic clusters")
        
        return state
    
    async def _perform_clustering(self, embeddings: Dict[str, List[float]], 
                                similarity_matrix: Dict[str, Dict[str, float]], 
                                config: Dict[str, Any]) -> List[SemanticCluster]:
        """Perform hierarchical clustering on embeddings."""
        threshold = config.get("clustering_threshold", 0.75)
        min_size = config.get("min_cluster_size", 2)
        max_size = config.get("max_cluster_size", 20)
        
        # Simple agglomerative clustering implementation
        chunk_ids = list(embeddings.keys())
        clusters = [[chunk_id] for chunk_id in chunk_ids]  # Start with individual clusters
        
        while True:
            # Find the most similar pair of clusters
            best_similarity = -1
            best_pair = None
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    similarity = self._calculate_cluster_similarity(clusters[i], clusters[j], similarity_matrix)
                    if similarity > best_similarity and len(clusters[i]) + len(clusters[j]) <= max_size:
                        best_similarity = similarity
                        best_pair = (i, j)
            
            # Stop if no good merge found
            if best_similarity < threshold or best_pair is None:
                break
            
            # Merge the best pair
            i, j = best_pair
            merged_cluster = clusters[i] + clusters[j]
            clusters = [cluster for k, cluster in enumerate(clusters) if k not in (i, j)] + [merged_cluster]
        
        # Filter clusters by minimum size and convert to SemanticCluster objects
        semantic_clusters = []
        for i, cluster_chunk_ids in enumerate(clusters):
            if len(cluster_chunk_ids) >= min_size:
                cluster_id = f"cluster_{i}_{hashlib.md5(''.join(sorted(cluster_chunk_ids)).encode()).hexdigest()[:8]}"
                
                # Calculate centroid embedding
                cluster_embeddings = [embeddings[chunk_id] for chunk_id in cluster_chunk_ids]
                centroid = np.mean(cluster_embeddings, axis=0) if cluster_embeddings else None
                
                # Calculate coherence score
                coherence = self._calculate_cluster_coherence(cluster_chunk_ids, similarity_matrix)
                
                semantic_clusters.append(SemanticCluster(
                    cluster_id=cluster_id,
                    chunk_ids=cluster_chunk_ids,
                    centroid_embedding=centroid,
                    cluster_theme="",  # Will be filled by theme enrichment
                    coherence_score=coherence,
                    size=len(cluster_chunk_ids)
                ))
        
        return semantic_clusters
    
    def _calculate_cluster_similarity(self, cluster1: List[str], cluster2: List[str], 
                                    similarity_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate average similarity between two clusters."""
        similarities = []
        
        for chunk1 in cluster1:
            for chunk2 in cluster2:
                if chunk1 in similarity_matrix and chunk2 in similarity_matrix[chunk1]:
                    similarities.append(similarity_matrix[chunk1][chunk2])
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_cluster_coherence(self, chunk_ids: List[str], 
                                   similarity_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate internal coherence of a cluster."""
        if len(chunk_ids) < 2:
            return 1.0
        
        similarities = []
        for i, chunk1 in enumerate(chunk_ids):
            for j, chunk2 in enumerate(chunk_ids[i+1:], i+1):
                if chunk1 in similarity_matrix and chunk2 in similarity_matrix[chunk1]:
                    similarities.append(similarity_matrix[chunk1][chunk2])
        
        return np.mean(similarities) if similarities else 0.0
    
    async def _enrich_clusters_with_themes(self, clusters: List[SemanticCluster], 
                                         state: WorkflowState) -> List[SemanticCluster]:
        """Enrich clusters with thematic descriptions."""
        chunks = state.get("chunks", [])
        chunk_lookup = {chunk["metadata"]["chunk_id"]: chunk for chunk in chunks}
        
        for cluster in clusters:
            # Extract content from cluster chunks
            cluster_contents = []
            for chunk_id in cluster.chunk_ids:
                if chunk_id in chunk_lookup:
                    cluster_contents.append(chunk_lookup[chunk_id]["content"])
            
            # Generate theme (mock implementation)
            theme = await self._generate_cluster_theme(cluster_contents)
            cluster.cluster_theme = theme
        
        return clusters
    
    async def _generate_cluster_theme(self, contents: List[str]) -> str:
        """Generate a thematic description for cluster contents."""
        # Mock theme generation (replace with actual LLM call)
        combined_content = " ".join(contents)[:500]  # Limit for analysis
        
        # Simple keyword extraction
        words = combined_content.lower().split()
        word_freq = Counter(words)
        
        # Filter common words and get top keywords
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        keywords = [word for word, freq in word_freq.most_common(5) if word not in common_words and len(word) > 2]
        
        if keywords:
            return f"Content related to: {', '.join(keywords[:3])}"
        else:
            return "General content cluster"


class EnhancedSemanticMapper:
    """Enhanced semantic mapping system with LangGraph workflow integration."""
    
    def __init__(self, config: Optional[EnhancedSemanticMappingConfig] = None,
                 state_manager: Optional[CentralizedStateManager] = None):
        self.config = config or EnhancedSemanticMappingConfig()
        self.state_manager = state_manager or CentralizedStateManager()
        self.workflow_graph = None
        self.memory_saver = MemorySaver()
        
        if self.config.enable_workflow:
            self._build_workflow_graph()
        
        logger.info(f"EnhancedSemanticMapper initialized with workflow: {self.config.enable_workflow}")
    
    def _build_workflow_graph(self) -> None:
        """Build the LangGraph workflow for semantic mapping."""
        workflow = StateGraph(WorkflowState)
        
        # Create workflow nodes
        embedder_node = ContentEmbedderNode(
            NodeConfig(name="content_embedder", max_retries=2),
            self.state_manager
        )
        
        similarity_node = SimilarityAnalyzerNode(
            NodeConfig(name="similarity_analyzer", max_retries=2),
            self.state_manager
        )
        
        relationship_node = RelationshipDetectorNode(
            NodeConfig(name="relationship_detector", max_retries=3),
            self.state_manager
        )
        
        clusterer_node = SemanticClustererNode(
            NodeConfig(name="semantic_clusterer", max_retries=2),
            self.state_manager
        )
        
        # Add nodes to workflow
        workflow.add_node("content_embedder", embedder_node.execute)
        workflow.add_node("similarity_analyzer", similarity_node.execute)
        workflow.add_node("relationship_detector", relationship_node.execute)
        workflow.add_node("semantic_clusterer", clusterer_node.execute)
        
        # Define workflow edges
        workflow.add_edge("content_embedder", "similarity_analyzer")
        workflow.add_edge("similarity_analyzer", "relationship_detector")
        workflow.add_edge("relationship_detector", "semantic_clusterer")
        workflow.add_edge("semantic_clusterer", END)
        
        # Set entry point
        workflow.set_entry_point("content_embedder")
        
        # Compile workflow
        self.workflow_graph = workflow.compile(checkpointer=self.memory_saver)
        
        logger.info("Semantic mapping workflow compiled successfully")
    
    async def create_semantic_mappings_async(self, chunks: List[DocumentChunk], 
                                           config: Optional[EnhancedSemanticMappingConfig] = None) -> Dict[str, Any]:
        """Create semantic mappings between chunks (alias for map_chunks_async)."""
        return await self.map_chunks_async(chunks, config)
    
    async def map_chunks_async(self, chunks: List[DocumentChunk], 
                             config: Optional[EnhancedSemanticMappingConfig] = None) -> Dict[str, Any]:
        """Map semantic relationships between chunks using enhanced workflow."""
        config = config or self.config
        
        if not config.enable_workflow or not self.workflow_graph:
            # Fallback to basic mapping
            return await self._basic_semantic_mapping(chunks)
        
        # Convert chunks to dictionaries
        chunk_dicts = [self._chunk_to_dict(chunk) for chunk in chunks]
        
        # Prepare workflow state
        initial_state = {
            "chunks": chunk_dicts,
            "knowledge_graph": {},
            "semantic_mappings": [],
            "llm_outputs": [],
            "final_output": {},
            "current_stage": "semantic_mapping",
            "progress": 0.0,
            "session_id": hashlib.md5(f"semantic_mapping_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "mapping_config": config.__dict__,
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
                "semantic_relationships": result_state.get("semantic_relationships", []),
                "semantic_clusters": result_state.get("semantic_clusters", []),
                "similarity_matrix": result_state.get("similarity_matrix", {}),
                "chunk_embeddings": result_state.get("chunk_embeddings", {}),
                "statistics": {
                    "relationship_stats": result_state.get("relationship_stats", {}),
                    "clustering_stats": result_state.get("clustering_stats", {}),
                    "similarity_stats": result_state.get("similarity_stats", {})
                },
                "metadata": {
                    "workflow_used": True,
                    "session_id": result_state["session_id"],
                    "processing_time": result_state.get("stage_timings", {})
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced semantic mapping workflow failed: {e}")
            # Fallback to basic mapping
            return await self._basic_semantic_mapping(chunks)
    
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
    
    async def _basic_semantic_mapping(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Basic semantic mapping fallback."""
        # Simple similarity-based mapping
        mappings = []
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                # Simple content overlap calculation
                words1 = set(chunk1.content.lower().split())
                words2 = set(chunk2.content.lower().split())
                
                if words1 and words2:
                    overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                    
                    if overlap > 0.3:  # Basic threshold
                        mappings.append({
                            "source_chunk_id": chunk1.metadata.chunk_id,
                            "target_chunk_id": chunk2.metadata.chunk_id,
                            "mapping_type": "conceptual",
                            "confidence_score": overlap,
                            "relationship_strength": overlap,
                            "semantic_distance": 1.0 - overlap,
                            "context_overlap": overlap,
                            "metadata": {"analysis_method": "basic"}
                        })
        
        return {
            "semantic_relationships": mappings,
            "semantic_clusters": [],
            "similarity_matrix": {},
            "chunk_embeddings": {},
            "statistics": {"total_relationships": len(mappings)},
            "metadata": {"workflow_used": False}
        }


# Factory functions
def create_enhanced_semantic_mapper(enable_workflow: bool = True,
                                  enable_embeddings: bool = True,
                                  **kwargs) -> EnhancedSemanticMapper:
    """Create enhanced semantic mapper with specified configuration."""
    config = EnhancedSemanticMappingConfig(
        enable_workflow=enable_workflow,
        enable_embeddings=enable_embeddings,
        **kwargs
    )
    return EnhancedSemanticMapper(config)


async def map_document_semantics_async(chunks: List[DocumentChunk],
                                     enable_workflow: bool = True,
                                     enable_clustering: bool = True) -> Dict[str, Any]:
    """Convenience function for enhanced async semantic mapping."""
    config = EnhancedSemanticMappingConfig(
        enable_workflow=enable_workflow,
        enable_clustering=enable_clustering
    )
    mapper = EnhancedSemanticMapper(config)
    return await mapper.map_chunks_async(chunks, config)