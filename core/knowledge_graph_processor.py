"""Enhanced Knowledge Graph Processor for LangGraph Integration

This module provides advanced knowledge graph construction, analysis, and
visualization capabilities that integrate seamlessly with the LangGraph
orchestrator and centralized state management system.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Set, Tuple, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import time
from collections import defaultdict, Counter

# Graph libraries
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logging.warning("NetworkX not available. Install networkx for advanced graph features.")

# Numerical libraries
try:
    import numpy as np
except ImportError:
    np = None
    logging.warning("NumPy not available. Some cohesion computation features may be limited.")

# NLP libraries
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logging.warning("spaCy not available. Install spacy for advanced NLP features.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    logging.warning("NLTK not available. Install nltk for basic NLP features.")

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    logging.warning("Visualization libraries not available. Install matplotlib and plotly.")

from .chunking_processor import DocumentChunk, ChunkingResult
from .llm_handler import ProcessingResult, BatchProcessingResult

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities that can be extracted."""
    PERSON = auto()
    ORGANIZATION = auto()
    LOCATION = auto()
    DATE = auto()
    MONEY = auto()
    CONCEPT = auto()
    TECHNOLOGY = auto()
    PROCESS = auto()
    PRODUCT = auto()
    EVENT = auto()
    CUSTOM = auto()


class RelationType(Enum):
    """Types of relationships between entities."""
    MENTIONS = auto()
    RELATED_TO = auto()
    PART_OF = auto()
    CAUSES = auto()
    ENABLES = auto()
    REQUIRES = auto()
    SIMILAR_TO = auto()
    OPPOSITE_TO = auto()
    TEMPORAL = auto()
    SPATIAL = auto()
    HIERARCHICAL = auto()
    FUNCTIONAL = auto()
    CUSTOM = auto()


class ExtractionMethod(Enum):
    """Methods for entity and relationship extraction."""
    RULE_BASED = auto()
    NLP_BASED = auto()
    LLM_BASED = auto()
    HYBRID = auto()
    STATISTICAL = auto()


class GraphAnalysisType(Enum):
    """Types of graph analysis to perform."""
    CENTRALITY = auto()
    CLUSTERING = auto()
    COMMUNITY = auto()
    PATHS = auto()
    CONNECTIVITY = auto()
    STRUCTURE = auto()
    SIMILARITY = auto()
    EVOLUTION = auto()


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    entity_type: EntityType
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    mentions: List[Dict[str, Any]] = field(default_factory=list)
    aliases: Set[str] = field(default_factory=set)
    source_chunks: Set[str] = field(default_factory=set)
    extraction_method: ExtractionMethod = ExtractionMethod.RULE_BASED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    confidence: float = 1.0
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    source_chunks: Set[str] = field(default_factory=set)
    extraction_method: ExtractionMethod = ExtractionMethod.RULE_BASED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Represents a complete knowledge graph."""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    graph: Optional[Any] = None  # NetworkX graph if available
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExtractionConfig:
    """Configuration for entity and relationship extraction."""
    extraction_method: ExtractionMethod = ExtractionMethod.HYBRID
    entity_types: Set[EntityType] = field(default_factory=lambda: set(EntityType))
    relation_types: Set[RelationType] = field(default_factory=lambda: set(RelationType))
    min_confidence: float = 0.5
    max_entities_per_chunk: int = 50
    max_relationships_per_chunk: int = 100
    enable_coreference_resolution: bool = True
    enable_entity_linking: bool = True
    enable_relationship_inference: bool = True
    custom_patterns: Dict[str, List[str]] = field(default_factory=dict)
    custom_rules: Dict[str, Any] = field(default_factory=dict)
    llm_prompts: Dict[str, str] = field(default_factory=dict)
    nlp_model: str = "en_core_web_sm"
    similarity_threshold: float = 0.8
    merge_similar_entities: bool = True
    deduplicate_relationships: bool = True


@dataclass
class GraphAnalysisConfig:
    """Configuration for graph analysis."""
    analysis_types: Set[GraphAnalysisType] = field(default_factory=lambda: {GraphAnalysisType.CENTRALITY, GraphAnalysisType.CLUSTERING})
    centrality_measures: List[str] = field(default_factory=lambda: ["degree", "betweenness", "closeness", "pagerank"])
    clustering_algorithm: str = "louvain"
    community_resolution: float = 1.0
    path_analysis_depth: int = 3
    similarity_metrics: List[str] = field(default_factory=lambda: ["jaccard", "cosine"])
    include_subgraphs: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "gexf", "graphml"])


class KnowledgeGraphProcessor:
    """Enhanced knowledge graph processor with advanced analysis capabilities."""
    
    def __init__(self, 
                 extraction_config: Optional[ExtractionConfig] = None,
                 analysis_config: Optional[GraphAnalysisConfig] = None):
        self.extraction_config = extraction_config or ExtractionConfig()
        self.analysis_config = analysis_config or GraphAnalysisConfig()
        
        # Initialize NLP components
        self._nlp = None
        self._initialize_nlp()
        
        # Cache for processed entities and relationships
        self._entity_cache = {}
        self._relationship_cache = {}
        
        # Statistics tracking
        self._stats = {
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "chunks_processed": 0,
            "processing_time": 0.0
        }
        
        # Initialize embedding model for cohesion computation
        self._embedding_model = None
        
        logger.info("KnowledgeGraphProcessor initialized")
    
    def _initialize_nlp(self) -> None:
        """Initialize NLP components."""
        try:
            if HAS_SPACY:
                try:
                    self._nlp = spacy.load(self.extraction_config.nlp_model)
                    logger.debug(f"Loaded spaCy model: {self.extraction_config.nlp_model}")
                except OSError:
                    logger.warning(f"spaCy model {self.extraction_config.nlp_model} not found, using basic features")
                    self._nlp = None
            
            if HAS_NLTK:
                # Download required NLTK data if not present
                try:
                    nltk.data.find('tokenizers/punkt')
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                    nltk.data.find('chunkers/maxent_ne_chunker')
                    nltk.data.find('corpora/words')
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    logger.info("Downloading required NLTK data...")
                    nltk.download('punkt', quiet=True)
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                    nltk.download('maxent_ne_chunker', quiet=True)
                    nltk.download('words', quiet=True)
                    nltk.download('stopwords', quiet=True)
                
                logger.debug("NLTK components initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize some NLP components: {e}")
    
    async def build_knowledge_graph_async(self, 
                                        chunks: List[DocumentChunk],
                                        llm_results: Optional[BatchProcessingResult] = None) -> KnowledgeGraph:
        """Build knowledge graph from document chunks asynchronously."""
        start_time = time.time()
        
        # Initialize knowledge graph
        kg = KnowledgeGraph()
        
        # Extract entities and relationships from chunks
        extraction_tasks = []
        for chunk in chunks:
            llm_result = None
            if llm_results:
                # Find corresponding LLM result
                for result in llm_results.results:
                    if result.chunk_id == chunk.metadata.chunk_id:
                        llm_result = result
                        break
            
            task = self._extract_from_chunk_async(chunk, llm_result)
            extraction_tasks.append(task)
        
        # Process chunks in parallel
        extraction_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        
        # Merge results into knowledge graph
        for result in extraction_results:
            if isinstance(result, Exception):
                logger.error(f"Extraction failed: {result}")
                continue
            
            entities, relationships = result
            
            # Add entities
            for entity in entities:
                if entity.id in kg.entities:
                    # Merge with existing entity
                    kg.entities[entity.id] = self._merge_entities(kg.entities[entity.id], entity)
                else:
                    kg.entities[entity.id] = entity
            
            # Add relationships
            for relationship in relationships:
                if relationship.id in kg.relationships:
                    # Merge with existing relationship
                    kg.relationships[relationship.id] = self._merge_relationships(
                        kg.relationships[relationship.id], relationship
                    )
                else:
                    kg.relationships[relationship.id] = relationship
        
        # Post-processing
        if self.extraction_config.merge_similar_entities:
            kg = await self._merge_similar_entities_async(kg)
        
        if self.extraction_config.deduplicate_relationships:
            kg = self._deduplicate_relationships(kg)
        
        if self.extraction_config.enable_relationship_inference:
            kg = await self._infer_relationships_async(kg)
        
        # Build NetworkX graph if available
        if HAS_NETWORKX:
            kg.graph = self._build_networkx_graph(kg)
            # Build unified graph with advanced features
            kg.metadata["unified_graph"] = self._build_unified_graph(chunks)
        
        # Calculate statistics
        kg.statistics = self._calculate_graph_statistics(kg)
        
        # Update processing stats
        processing_time = time.time() - start_time
        self._stats["chunks_processed"] += len(chunks)
        self._stats["entities_extracted"] += len(kg.entities)
        self._stats["relationships_extracted"] += len(kg.relationships)
        self._stats["processing_time"] += processing_time
        
        kg.metadata["processing_time"] = processing_time
        kg.metadata["extraction_config"] = self.extraction_config.__dict__
        
        logger.info(f"Knowledge graph built: {len(kg.entities)} entities, {len(kg.relationships)} relationships")
        
        return kg
    
    async def _extract_from_chunk_async(self, 
                                      chunk: DocumentChunk,
                                      llm_result: Optional[ProcessingResult] = None) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a single chunk."""
        entities = []
        relationships = []
        
        # Use different extraction methods based on configuration
        if self.extraction_config.extraction_method == ExtractionMethod.RULE_BASED:
            chunk_entities, chunk_relationships = await self._extract_rule_based_async(chunk)
        elif self.extraction_config.extraction_method == ExtractionMethod.NLP_BASED:
            chunk_entities, chunk_relationships = await self._extract_nlp_based_async(chunk)
        elif self.extraction_config.extraction_method == ExtractionMethod.LLM_BASED:
            chunk_entities, chunk_relationships = await self._extract_llm_based_async(chunk, llm_result)
        elif self.extraction_config.extraction_method == ExtractionMethod.HYBRID:
            # Combine multiple methods
            rule_entities, rule_relationships = await self._extract_rule_based_async(chunk)
            nlp_entities, nlp_relationships = await self._extract_nlp_based_async(chunk)
            
            chunk_entities = rule_entities + nlp_entities
            chunk_relationships = rule_relationships + nlp_relationships
            
            if llm_result:
                llm_entities, llm_relationships = await self._extract_llm_based_async(chunk, llm_result)
                chunk_entities.extend(llm_entities)
                chunk_relationships.extend(llm_relationships)
        else:
            chunk_entities, chunk_relationships = await self._extract_statistical_async(chunk)
        
        # Filter by confidence
        entities = [e for e in chunk_entities if e.confidence >= self.extraction_config.min_confidence]
        relationships = [r for r in chunk_relationships if r.confidence >= self.extraction_config.min_confidence]
        
        # Limit number of entities and relationships
        entities = entities[:self.extraction_config.max_entities_per_chunk]
        relationships = relationships[:self.extraction_config.max_relationships_per_chunk]
        
        return entities, relationships
    
    async def _extract_rule_based_async(self, chunk: DocumentChunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships using rule-based methods."""
        entities = []
        relationships = []
        
        content = chunk.content
        
        # Extract entities using regex patterns
        entity_patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\. [A-Z][a-z]+ [A-Z][a-z]+\b'  # Title First Last
            ],
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                r'\b(?:University of|Institute of) [A-Z][a-z]+\b'
            ],
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+, [A-Z][A-Z]\b',  # City, State
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'   # Generic location
            ],
            EntityType.DATE: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'
            ],
            EntityType.MONEY: [
                r'\$[\d,]+(?:\.\d{2})?\b',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})? (?:dollars|USD|euros|EUR)\b'
            ]
        }
        
        # Custom patterns from config
        for entity_type_name, patterns in self.extraction_config.custom_patterns.items():
            try:
                entity_type = EntityType[entity_type_name.upper()]
                if entity_type not in entity_patterns:
                    entity_patterns[entity_type] = []
                entity_patterns[entity_type].extend(patterns)
            except KeyError:
                logger.warning(f"Unknown entity type in custom patterns: {entity_type_name}")
        
        # Extract entities
        for entity_type, patterns in entity_patterns.items():
            if entity_type not in self.extraction_config.entity_types:
                continue
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    entity_id = self._generate_entity_id(entity_text, entity_type)
                    
                    entity = Entity(
                        id=entity_id,
                        name=entity_text,
                        entity_type=entity_type,
                        confidence=0.8,  # Rule-based confidence
                        mentions=[{
                            "text": entity_text,
                            "start": match.start(),
                            "end": match.end(),
                            "chunk_id": chunk.metadata.chunk_id
                        }],
                        source_chunks={chunk.metadata.chunk_id},
                        extraction_method=ExtractionMethod.RULE_BASED
                    )
                    
                    entities.append(entity)
        
        # Extract relationships using simple co-occurrence
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Simple co-occurrence relationship
                relationship_id = self._generate_relationship_id(
                    entity1.id, entity2.id, RelationType.MENTIONS
                )
                
                relationship = Relationship(
                    id=relationship_id,
                    source_entity_id=entity1.id,
                    target_entity_id=entity2.id,
                    relation_type=RelationType.MENTIONS,
                    confidence=0.6,
                    weight=1.0,
                    evidence=[{
                        "type": "co_occurrence",
                        "chunk_id": chunk.metadata.chunk_id,
                        "context": content[:200] + "..." if len(content) > 200 else content
                    }],
                    source_chunks={chunk.metadata.chunk_id},
                    extraction_method=ExtractionMethod.RULE_BASED
                )
                
                relationships.append(relationship)
        
        return entities, relationships
    
    async def _extract_nlp_based_async(self, chunk: DocumentChunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships using NLP methods."""
        entities = []
        relationships = []
        
        content = chunk.content
        
        # Use spaCy if available
        if self._nlp:
            doc = self._nlp(content)
            
            # Extract named entities
            for ent in doc.ents:
                entity_type = self._map_spacy_label_to_entity_type(ent.label_)
                if entity_type and entity_type in self.extraction_config.entity_types:
                    entity_id = self._generate_entity_id(ent.text, entity_type)
                    
                    entity = Entity(
                        id=entity_id,
                        name=ent.text,
                        entity_type=entity_type,
                        confidence=0.9,  # NLP-based confidence
                        attributes={
                            "spacy_label": ent.label_,
                            "start_char": ent.start_char,
                            "end_char": ent.end_char
                        },
                        mentions=[{
                            "text": ent.text,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "chunk_id": chunk.metadata.chunk_id
                        }],
                        source_chunks={chunk.metadata.chunk_id},
                        extraction_method=ExtractionMethod.NLP_BASED
                    )
                    
                    entities.append(entity)
            
            # Extract relationships using dependency parsing
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in ["nsubj", "dobj", "pobj"]:
                        # Find related entities
                        head_ents = [ent for ent in sent.ents if ent.start <= token.head.i < ent.end]
                        token_ents = [ent for ent in sent.ents if ent.start <= token.i < ent.end]
                        
                        for head_ent in head_ents:
                            for token_ent in token_ents:
                                if head_ent != token_ent:
                                    relation_type = self._map_dependency_to_relation_type(token.dep_)
                                    if relation_type and relation_type in self.extraction_config.relation_types:
                                        relationship_id = self._generate_relationship_id(
                                            self._generate_entity_id(head_ent.text, self._map_spacy_label_to_entity_type(head_ent.label_)),
                                            self._generate_entity_id(token_ent.text, self._map_spacy_label_to_entity_type(token_ent.label_)),
                                            relation_type
                                        )
                                        
                                        relationship = Relationship(
                                            id=relationship_id,
                                            source_entity_id=self._generate_entity_id(head_ent.text, self._map_spacy_label_to_entity_type(head_ent.label_)),
                                            target_entity_id=self._generate_entity_id(token_ent.text, self._map_spacy_label_to_entity_type(token_ent.label_)),
                                            relation_type=relation_type,
                                            confidence=0.8,
                                            weight=1.0,
                                            attributes={
                                                "dependency": token.dep_,
                                                "sentence": sent.text
                                            },
                                            evidence=[{
                                                "type": "dependency_parsing",
                                                "chunk_id": chunk.metadata.chunk_id,
                                                "sentence": sent.text,
                                                "dependency": token.dep_
                                            }],
                                            source_chunks={chunk.metadata.chunk_id},
                                            extraction_method=ExtractionMethod.NLP_BASED
                                        )
                                        
                                        relationships.append(relationship)
        
        # Fallback to NLTK if spaCy not available
        elif HAS_NLTK:
            sentences = sent_tokenize(content)
            
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                
                # Simple named entity recognition
                try:
                    chunks = ne_chunk(pos_tags)
                    
                    for chunk_item in chunks:
                        if hasattr(chunk_item, 'label'):
                            entity_text = ' '.join([token for token, pos in chunk_item.leaves()])
                            entity_type = self._map_nltk_label_to_entity_type(chunk_item.label())
                            
                            if entity_type and entity_type in self.extraction_config.entity_types:
                                entity_id = self._generate_entity_id(entity_text, entity_type)
                                
                                entity = Entity(
                                    id=entity_id,
                                    name=entity_text,
                                    entity_type=entity_type,
                                    confidence=0.7,  # NLTK-based confidence
                                    attributes={
                                        "nltk_label": chunk_item.label(),
                                        "sentence": sentence
                                    },
                                    mentions=[{
                                        "text": entity_text,
                                        "sentence": sentence,
                                        "chunk_id": chunk.metadata.chunk_id
                                    }],
                                    source_chunks={chunk.metadata.chunk_id},
                                    extraction_method=ExtractionMethod.NLP_BASED
                                )
                                
                                entities.append(entity)
                
                except Exception as e:
                    logger.warning(f"NLTK NER failed for sentence: {e}")
        
        return entities, relationships
    
    async def _extract_llm_based_async(self, 
                                     chunk: DocumentChunk,
                                     llm_result: Optional[ProcessingResult] = None) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships using LLM-based methods."""
        entities = []
        relationships = []
        
        if not llm_result or not llm_result.processed_content:
            return entities, relationships
        
        # Parse LLM output for entities and relationships
        try:
            # Assume LLM output is structured JSON or can be parsed
            llm_content = llm_result.processed_content
            
            # Try to parse as JSON first
            try:
                parsed_data = json.loads(llm_content)
                
                # Extract entities from parsed data
                if "entities" in parsed_data:
                    for entity_data in parsed_data["entities"]:
                        entity_type = EntityType[entity_data.get("type", "CONCEPT").upper()]
                        if entity_type in self.extraction_config.entity_types:
                            entity_id = self._generate_entity_id(entity_data["name"], entity_type)
                            
                            entity = Entity(
                                id=entity_id,
                                name=entity_data["name"],
                                entity_type=entity_type,
                                confidence=entity_data.get("confidence", 0.9),
                                attributes=entity_data.get("attributes", {}),
                                source_chunks={chunk.metadata.chunk_id},
                                extraction_method=ExtractionMethod.LLM_BASED
                            )
                            
                            entities.append(entity)
                
                # Extract relationships from parsed data
                if "relationships" in parsed_data:
                    for rel_data in parsed_data["relationships"]:
                        relation_type = RelationType[rel_data.get("type", "RELATED_TO").upper()]
                        if relation_type in self.extraction_config.relation_types:
                            relationship_id = self._generate_relationship_id(
                                rel_data["source"], rel_data["target"], relation_type
                            )
                            
                            relationship = Relationship(
                                id=relationship_id,
                                source_entity_id=rel_data["source"],
                                target_entity_id=rel_data["target"],
                                relation_type=relation_type,
                                confidence=rel_data.get("confidence", 0.9),
                                weight=rel_data.get("weight", 1.0),
                                attributes=rel_data.get("attributes", {}),
                                source_chunks={chunk.metadata.chunk_id},
                                extraction_method=ExtractionMethod.LLM_BASED
                            )
                            
                            relationships.append(relationship)
            
            except json.JSONDecodeError:
                # Parse as text if JSON parsing fails
                entities, relationships = self._parse_llm_text_output(llm_content, chunk)
        
        except Exception as e:
            logger.error(f"LLM-based extraction failed: {e}")
        
        return entities, relationships
    
    async def _extract_statistical_async(self, chunk: DocumentChunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships using statistical methods."""
        entities = []
        relationships = []
        
        content = chunk.content
        
        # Simple statistical extraction based on word frequency and patterns
        words = re.findall(r'\b[A-Za-z]+\b', content.lower())
        word_freq = Counter(words)
        
        # Extract high-frequency terms as potential concepts
        for word, freq in word_freq.most_common(20):
            if len(word) > 3 and freq > 2:  # Simple heuristics
                entity_id = self._generate_entity_id(word, EntityType.CONCEPT)
                
                entity = Entity(
                    id=entity_id,
                    name=word,
                    entity_type=EntityType.CONCEPT,
                    confidence=min(0.9, freq / 10),  # Frequency-based confidence
                    attributes={"frequency": freq},
                    source_chunks={chunk.metadata.chunk_id},
                    extraction_method=ExtractionMethod.STATISTICAL
                )
                
                entities.append(entity)
        
        # Create relationships based on co-occurrence
        for i, entity1 in enumerate(entities[:10]):  # Limit to avoid too many relationships
            for entity2 in entities[i+1:10]:
                relationship_id = self._generate_relationship_id(
                    entity1.id, entity2.id, RelationType.RELATED_TO
                )
                
                relationship = Relationship(
                    id=relationship_id,
                    source_entity_id=entity1.id,
                    target_entity_id=entity2.id,
                    relation_type=RelationType.RELATED_TO,
                    confidence=0.5,
                    weight=1.0,
                    source_chunks={chunk.metadata.chunk_id},
                    extraction_method=ExtractionMethod.STATISTICAL
                )
                
                relationships.append(relationship)
        
        return entities, relationships
    
    def _generate_entity_id(self, name: str, entity_type: EntityType) -> str:
        """Generate unique ID for entity."""
        normalized_name = name.lower().strip()
        type_prefix = entity_type.name.lower()[:3]
        hash_suffix = hashlib.sha256(f"{normalized_name}_{entity_type.name}".encode()).hexdigest()[:8]
        return f"{type_prefix}_{hash_suffix}"
    
    def _generate_relationship_id(self, source_id: str, target_id: str, relation_type: RelationType) -> str:
        """Generate unique ID for relationship."""
        # Ensure consistent ordering for bidirectional relationships
        if source_id > target_id:
            source_id, target_id = target_id, source_id
        
        type_prefix = relation_type.name.lower()[:3]
        hash_suffix = hashlib.sha256(f"{source_id}_{target_id}_{relation_type.name}".encode()).hexdigest()[:8]
        return f"{type_prefix}_{hash_suffix}"
    
    def _map_spacy_label_to_entity_type(self, label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our entity types."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "MONEY": EntityType.MONEY,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.CONCEPT,
            "LAW": EntityType.CONCEPT,
            "LANGUAGE": EntityType.CONCEPT
        }
        return mapping.get(label)
    
    def _map_nltk_label_to_entity_type(self, label: str) -> Optional[EntityType]:
        """Map NLTK entity labels to our entity types."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOCATION": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "MONEY": EntityType.MONEY
        }
        return mapping.get(label)
    
    def _map_dependency_to_relation_type(self, dep: str) -> Optional[RelationType]:
        """Map dependency relations to our relation types."""
        mapping = {
            "nsubj": RelationType.RELATED_TO,
            "dobj": RelationType.RELATED_TO,
            "pobj": RelationType.RELATED_TO,
            "compound": RelationType.PART_OF,
            "amod": RelationType.RELATED_TO,
            "prep": RelationType.RELATED_TO
        }
        return mapping.get(dep)
    
    def _parse_llm_text_output(self, content: str, chunk: DocumentChunk) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM text output for entities and relationships."""
        entities = []
        relationships = []
        
        # Simple text parsing - can be enhanced with more sophisticated methods
        lines = content.split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.lower().startswith('entities:'):
                current_section = 'entities'
                continue
            elif line.lower().startswith('relationships:'):
                current_section = 'relationships'
                continue
            
            if current_section == 'entities':
                # Parse entity line: "- EntityName (Type)"
                match = re.match(r'-\s*(.+?)\s*\((.+?)\)', line)
                if match:
                    name, type_str = match.groups()
                    try:
                        entity_type = EntityType[type_str.upper()]
                        entity_id = self._generate_entity_id(name, entity_type)
                        
                        entity = Entity(
                            id=entity_id,
                            name=name,
                            entity_type=entity_type,
                            confidence=0.8,
                            source_chunks={chunk.metadata.chunk_id},
                            extraction_method=ExtractionMethod.LLM_BASED
                        )
                        
                        entities.append(entity)
                    except KeyError:
                        logger.warning(f"Unknown entity type: {type_str}")
            
            elif current_section == 'relationships':
                # Parse relationship line: "- Entity1 -> Entity2 (RelationType)"
                match = re.match(r'-\s*(.+?)\s*->\s*(.+?)\s*\((.+?)\)', line)
                if match:
                    source, target, type_str = match.groups()
                    try:
                        relation_type = RelationType[type_str.upper()]
                        relationship_id = self._generate_relationship_id(source, target, relation_type)
                        
                        relationship = Relationship(
                            id=relationship_id,
                            source_entity_id=source,
                            target_entity_id=target,
                            relation_type=relation_type,
                            confidence=0.8,
                            weight=1.0,
                            source_chunks={chunk.metadata.chunk_id},
                            extraction_method=ExtractionMethod.LLM_BASED
                        )
                        
                        relationships.append(relationship)
                    except KeyError:
                        logger.warning(f"Unknown relation type: {type_str}")
        
        return entities, relationships
    
    def _merge_entities(self, existing: Entity, new: Entity) -> Entity:
        """Merge two entities with the same ID."""
        # Combine mentions
        existing.mentions.extend(new.mentions)
        
        # Combine source chunks
        existing.source_chunks.update(new.source_chunks)
        
        # Combine aliases
        existing.aliases.update(new.aliases)
        existing.aliases.add(new.name)
        
        # Update confidence (take maximum)
        existing.confidence = max(existing.confidence, new.confidence)
        
        # Merge attributes
        existing.attributes.update(new.attributes)
        
        # Update timestamp
        existing.updated_at = datetime.now(timezone.utc)
        
        return existing
    
    def _merge_relationships(self, existing: Relationship, new: Relationship) -> Relationship:
        """Merge two relationships with the same ID."""
        # Combine evidence
        existing.evidence.extend(new.evidence)
        
        # Combine source chunks
        existing.source_chunks.update(new.source_chunks)
        
        # Update confidence (take maximum)
        existing.confidence = max(existing.confidence, new.confidence)
        
        # Update weight (take average)
        existing.weight = (existing.weight + new.weight) / 2
        
        # Merge attributes
        existing.attributes.update(new.attributes)
        
        # Update timestamp
        existing.updated_at = datetime.now(timezone.utc)
        
        return existing
    
    async def _merge_similar_entities_async(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """Merge similar entities based on similarity threshold."""
        # Simple similarity based on name similarity
        entities_to_merge = []
        processed_entities = set()
        
        entity_list = list(kg.entities.values())
        
        for i, entity1 in enumerate(entity_list):
            if entity1.id in processed_entities:
                continue
            
            similar_entities = [entity1]
            
            for entity2 in entity_list[i+1:]:
                if entity2.id in processed_entities:
                    continue
                
                # Calculate similarity (simple string similarity)
                similarity = self._calculate_string_similarity(entity1.name, entity2.name)
                
                if (similarity >= self.extraction_config.similarity_threshold and 
                    entity1.entity_type == entity2.entity_type):
                    similar_entities.append(entity2)
                    processed_entities.add(entity2.id)
            
            if len(similar_entities) > 1:
                entities_to_merge.append(similar_entities)
            
            processed_entities.add(entity1.id)
        
        # Merge similar entities
        for entity_group in entities_to_merge:
            primary_entity = entity_group[0]
            
            for entity in entity_group[1:]:
                primary_entity = self._merge_entities(primary_entity, entity)
                # Remove merged entity
                if entity.id in kg.entities:
                    del kg.entities[entity.id]
                
                # Update relationships that reference the merged entity
                for rel_id, relationship in kg.relationships.items():
                    if relationship.source_entity_id == entity.id:
                        relationship.source_entity_id = primary_entity.id
                    if relationship.target_entity_id == entity.id:
                        relationship.target_entity_id = primary_entity.id
            
            # Update the primary entity in the graph
            kg.entities[primary_entity.id] = primary_entity
        
        return kg
    
    def _deduplicate_relationships(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """Remove duplicate relationships."""
        unique_relationships = {}
        
        for relationship in kg.relationships.values():
            # Create a key for deduplication
            key = (relationship.source_entity_id, relationship.target_entity_id, relationship.relation_type)
            
            if key not in unique_relationships:
                unique_relationships[key] = relationship
            else:
                # Merge with existing relationship
                unique_relationships[key] = self._merge_relationships(unique_relationships[key], relationship)
        
        # Update relationships in knowledge graph
        kg.relationships = {rel.id: rel for rel in unique_relationships.values()}
        
        return kg
    
    async def _infer_relationships_async(self, kg: KnowledgeGraph) -> KnowledgeGraph:
        """Infer additional relationships based on existing ones."""
        # Simple transitive relationship inference
        new_relationships = []
        
        for rel1 in kg.relationships.values():
            for rel2 in kg.relationships.values():
                if (rel1.target_entity_id == rel2.source_entity_id and 
                    rel1.source_entity_id != rel2.target_entity_id):
                    
                    # Infer transitive relationship
                    inferred_type = self._infer_transitive_relation_type(rel1.relation_type, rel2.relation_type)
                    
                    if inferred_type:
                        relationship_id = self._generate_relationship_id(
                            rel1.source_entity_id, rel2.target_entity_id, inferred_type
                        )
                        
                        # Check if relationship already exists
                        if relationship_id not in kg.relationships:
                            inferred_relationship = Relationship(
                                id=relationship_id,
                                source_entity_id=rel1.source_entity_id,
                                target_entity_id=rel2.target_entity_id,
                                relation_type=inferred_type,
                                confidence=min(rel1.confidence, rel2.confidence) * 0.8,  # Lower confidence for inferred
                                weight=min(rel1.weight, rel2.weight),
                                attributes={"inferred": True, "source_relations": [rel1.id, rel2.id]},
                                evidence=[{
                                    "type": "transitive_inference",
                                    "source_relations": [rel1.id, rel2.id]
                                }],
                                extraction_method=ExtractionMethod.HYBRID
                            )
                            
                            new_relationships.append(inferred_relationship)
        
        # Add inferred relationships to knowledge graph
        for relationship in new_relationships:
            kg.relationships[relationship.id] = relationship
        
        return kg
    
    def _infer_transitive_relation_type(self, rel1_type: RelationType, rel2_type: RelationType) -> Optional[RelationType]:
        """Infer transitive relationship type."""
        # Simple transitive rules
        transitive_rules = {
            (RelationType.PART_OF, RelationType.PART_OF): RelationType.PART_OF,
            (RelationType.CAUSES, RelationType.CAUSES): RelationType.CAUSES,
            (RelationType.ENABLES, RelationType.ENABLES): RelationType.ENABLES,
            (RelationType.REQUIRES, RelationType.REQUIRES): RelationType.REQUIRES,
        }
        
        return transitive_rules.get((rel1_type, rel2_type))
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        # Simple Jaccard similarity
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _build_networkx_graph(self, kg: KnowledgeGraph) -> Any:
        """Build NetworkX graph from knowledge graph."""
        if not HAS_NETWORKX:
            return None
        
        G = nx.DiGraph()
        
        # Add nodes (entities)
        for entity in kg.entities.values():
            G.add_node(
                entity.id,
                name=entity.name,
                entity_type=entity.entity_type.name,
                confidence=entity.confidence,
                **entity.attributes
            )
        
        # Add edges (relationships)
        for relationship in kg.relationships.values():
            if (relationship.source_entity_id in kg.entities and 
                relationship.target_entity_id in kg.entities):
                G.add_edge(
                    relationship.source_entity_id,
                    relationship.target_entity_id,
                    relation_type=relationship.relation_type.name,
                    confidence=relationship.confidence,
                    weight=relationship.weight,
                    **relationship.attributes
                )
        
        return G
    
    def _calculate_graph_statistics(self, kg: KnowledgeGraph) -> Dict[str, Any]:
        """Calculate statistics for the knowledge graph."""
        stats = {
            "num_entities": len(kg.entities),
            "num_relationships": len(kg.relationships),
            "entity_types": {},
            "relation_types": {},
            "avg_confidence": 0.0,
            "extraction_methods": {}
        }
        
        # Entity type distribution
        for entity in kg.entities.values():
            entity_type = entity.entity_type.name
            stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1
            
            # Extraction method distribution
            method = entity.extraction_method.name
            stats["extraction_methods"][method] = stats["extraction_methods"].get(method, 0) + 1
        
        # Relationship type distribution
        total_confidence = 0.0
        for relationship in kg.relationships.values():
            relation_type = relationship.relation_type.name
            stats["relation_types"][relation_type] = stats["relation_types"].get(relation_type, 0) + 1
            total_confidence += relationship.confidence
        
        # Average confidence
        if kg.relationships:
            stats["avg_confidence"] = total_confidence / len(kg.relationships)
        
        # NetworkX statistics if available
        if kg.graph and HAS_NETWORKX:
            stats["networkx_stats"] = {
                "num_nodes": kg.graph.number_of_nodes(),
                "num_edges": kg.graph.number_of_edges(),
                "density": nx.density(kg.graph),
                "is_connected": nx.is_weakly_connected(kg.graph),
                "num_components": nx.number_weakly_connected_components(kg.graph)
            }
        
        return stats
    
    async def analyze_graph_async(self, kg: KnowledgeGraph) -> Dict[str, Any]:
        """Perform comprehensive graph analysis."""
        analysis_results = {}
        
        if not kg.graph or not HAS_NETWORKX:
            logger.warning("NetworkX graph not available for analysis")
            return analysis_results
        
        G = kg.graph
        
        # Centrality analysis
        if GraphAnalysisType.CENTRALITY in self.analysis_config.analysis_types:
            centrality_results = {}
            
            for measure in self.analysis_config.centrality_measures:
                try:
                    if measure == "degree":
                        centrality_results["degree"] = dict(G.degree())
                    elif measure == "betweenness":
                        centrality_results["betweenness"] = nx.betweenness_centrality(G)
                    elif measure == "closeness":
                        centrality_results["closeness"] = nx.closeness_centrality(G)
                    elif measure == "pagerank":
                        centrality_results["pagerank"] = nx.pagerank(G)
                    elif measure == "eigenvector":
                        try:
                            centrality_results["eigenvector"] = nx.eigenvector_centrality(G)
                        except nx.PowerIterationFailedConvergence:
                            logger.warning("Eigenvector centrality failed to converge")
                except Exception as e:
                    logger.error(f"Failed to calculate {measure} centrality: {e}")
            
            analysis_results["centrality"] = centrality_results
        
        # Community detection
        if GraphAnalysisType.COMMUNITY in self.analysis_config.analysis_types:
            try:
                # Convert to undirected for community detection
                G_undirected = G.to_undirected()
                
                if self.analysis_config.clustering_algorithm == "louvain":
                    try:
                        import community as community_louvain
                        communities = community_louvain.best_partition(
                            G_undirected, 
                            resolution=self.analysis_config.community_resolution
                        )
                        analysis_results["communities"] = communities
                    except ImportError:
                        logger.warning("python-louvain not available for community detection")
                
                # Alternative: use NetworkX's built-in community detection
                try:
                    communities = nx.community.greedy_modularity_communities(G_undirected)
                    community_dict = {}
                    for i, community in enumerate(communities):
                        for node in community:
                            community_dict[node] = i
                    analysis_results["greedy_communities"] = community_dict
                except Exception as e:
                    logger.error(f"Community detection failed: {e}")
            
            except Exception as e:
                logger.error(f"Community analysis failed: {e}")
        
        # Path analysis
        if GraphAnalysisType.PATHS in self.analysis_config.analysis_types:
            try:
                # Find shortest paths between high-centrality nodes
                if "pagerank" in analysis_results.get("centrality", {}):
                    pagerank_scores = analysis_results["centrality"]["pagerank"]
                    top_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    path_analysis = {}
                    for i, (node1, _) in enumerate(top_nodes[:5]):
                        for node2, _ in top_nodes[i+1:5]:
                            try:
                                if nx.has_path(G, node1, node2):
                                    path = nx.shortest_path(G, node1, node2)
                                    path_analysis[f"{node1}_to_{node2}"] = {
                                        "path": path,
                                        "length": len(path) - 1
                                    }
                            except nx.NetworkXNoPath:
                                continue
                    
                    analysis_results["shortest_paths"] = path_analysis
            
            except Exception as e:
                logger.error(f"Path analysis failed: {e}")
        
        # Structural analysis
        if GraphAnalysisType.STRUCTURE in self.analysis_config.analysis_types:
            try:
                structural_analysis = {
                    "diameter": nx.diameter(G) if nx.is_strongly_connected(G) else None,
                    "radius": nx.radius(G) if nx.is_strongly_connected(G) else None,
                    "average_clustering": nx.average_clustering(G.to_undirected()),
                    "transitivity": nx.transitivity(G.to_undirected()),
                    "assortativity": nx.degree_assortativity_coefficient(G),
                }
                
                analysis_results["structural"] = structural_analysis
            
            except Exception as e:
                logger.error(f"Structural analysis failed: {e}")
        
        return analysis_results
    
    def export_knowledge_graph(self, kg: KnowledgeGraph, output_path: Path, format: str = "json") -> bool:
        """Export knowledge graph to various formats."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                # Export as JSON
                export_data = {
                    "entities": {
                        entity_id: {
                            "id": entity.id,
                            "name": entity.name,
                            "type": entity.entity_type.name,
                            "confidence": entity.confidence,
                            "attributes": entity.attributes,
                            "mentions": entity.mentions,
                            "aliases": list(entity.aliases),
                            "source_chunks": list(entity.source_chunks),
                            "extraction_method": entity.extraction_method.name,
                            "created_at": entity.created_at.isoformat(),
                            "updated_at": entity.updated_at.isoformat(),
                            "metadata": entity.metadata
                        }
                        for entity_id, entity in kg.entities.items()
                    },
                    "relationships": {
                        rel_id: {
                            "id": rel.id,
                            "source_entity_id": rel.source_entity_id,
                            "target_entity_id": rel.target_entity_id,
                            "relation_type": rel.relation_type.name,
                            "confidence": rel.confidence,
                            "weight": rel.weight,
                            "attributes": rel.attributes,
                            "evidence": rel.evidence,
                            "source_chunks": list(rel.source_chunks),
                            "extraction_method": rel.extraction_method.name,
                            "created_at": rel.created_at.isoformat(),
                            "updated_at": rel.updated_at.isoformat(),
                            "metadata": rel.metadata
                        }
                        for rel_id, rel in kg.relationships.items()
                    },
                    "statistics": kg.statistics,
                    "metadata": kg.metadata,
                    "created_at": kg.created_at.isoformat(),
                    "updated_at": kg.updated_at.isoformat()
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() in ["gexf", "graphml"] and kg.graph and HAS_NETWORKX:
                # Export NetworkX graph
                if format.lower() == "gexf":
                    nx.write_gexf(kg.graph, output_path)
                elif format.lower() == "graphml":
                    nx.write_graphml(kg.graph, output_path)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Knowledge graph exported to {output_path} in {format} format")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export knowledge graph: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "chunks_processed": 0,
            "processing_time": 0.0
        }
    
    def _build_unified_graph(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Build unified graph with advanced features from chunks."""
        try:
            import networkx as nx
            
            # Create unified graph
            G = nx.DiGraph()
            
            # Add chunk nodes with metadata
            for chunk in chunks:
                chunk_id = chunk.metadata.chunk_id
                G.add_node(chunk_id, 
                          content=chunk.content[:200],  # Truncated content
                          section=chunk.metadata.section_title,
                          page=chunk.metadata.page_number,
                          score=self._calculate_chunk_score(chunk))
            
            # Add section affinity edges
            section_chunks = {}
            for chunk in chunks:
                section = chunk.metadata.section_title or "unknown"
                if section not in section_chunks:
                    section_chunks[section] = []
                section_chunks[section].append(chunk.metadata.chunk_id)
            
            # Connect chunks within same section
            for section, chunk_ids in section_chunks.items():
                for i, chunk_id1 in enumerate(chunk_ids):
                    for chunk_id2 in chunk_ids[i+1:]:
                        affinity = self._calculate_section_affinity(chunk_id1, chunk_id2, chunks)
                        if affinity > 0.3:  # Threshold for connection
                            G.add_edge(chunk_id1, chunk_id2, weight=affinity, type="section_affinity")
            
            # Calculate graph metrics
            metrics = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G) if G.number_of_nodes() > 1 else 0,
                "components": nx.number_weakly_connected_components(G)
            }
            
            return {
                "graph_data": nx.node_link_data(G),
                "metrics": metrics,
                "section_distribution": {section: len(chunks) for section, chunks in section_chunks.items()}
            }
            
        except Exception as e:
            logger.error(f"Failed to build unified graph: {e}")
            return {"error": str(e)}
    
    def _calculate_chunk_score(self, chunk: DocumentChunk) -> float:
        """Calculate quality score for a chunk."""
        score = 0.0
        
        # Content length factor (normalized)
        content_length = len(chunk.content)
        if content_length > 100:
            score += min(content_length / 1000, 1.0) * 0.3
        
        # Section title presence
        if chunk.metadata.section_title:
            score += 0.2
        
        # Page number presence (indicates structured document)
        if chunk.metadata.page_number is not None:
            score += 0.1
        
        # Mathematical content bonus (for LaTeX)
        if any(marker in chunk.content for marker in ['$', '\\begin{', '\\equation']):
            score += 0.2
        
        # Code content bonus
        if any(marker in chunk.content for marker in ['```', 'def ', 'class ', 'import ']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_section_affinity(self, chunk_id1: str, chunk_id2: str, chunks: List[DocumentChunk]) -> float:
        """Calculate affinity between two chunks in the same section."""
        chunk1 = next((c for c in chunks if c.metadata.chunk_id == chunk_id1), None)
        chunk2 = next((c for c in chunks if c.metadata.chunk_id == chunk_id2), None)
        
        if not chunk1 or not chunk2:
            return 0.0
        
        affinity = 0.0
        
        # Same section bonus
        if chunk1.metadata.section_title == chunk2.metadata.section_title:
            affinity += 0.4
        
        # Sequential pages bonus
        if (chunk1.metadata.page_number is not None and 
            chunk2.metadata.page_number is not None):
            page_diff = abs(chunk1.metadata.page_number - chunk2.metadata.page_number)
            if page_diff <= 1:
                affinity += 0.3
            elif page_diff <= 3:
                affinity += 0.1
        
        # Content similarity (basic keyword overlap)
        words1 = set(chunk1.content.lower().split())
        words2 = set(chunk2.content.lower().split())
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard = overlap / union if union > 0 else 0
            affinity += jaccard * 0.3
        
        return min(affinity, 1.0)
    
    async def compute_cohesion_async(self, chunks: List[DocumentChunk]) -> Dict[str, float]:
        """Compute cohesion metrics for document chunks."""
        if not chunks:
            return {}
        
        try:
            # Initialize embedding model if needed
            if self._embedding_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                except ImportError:
                    logger.warning("SentenceTransformers not available, using basic cohesion")
                    return self._compute_basic_cohesion(chunks)
            
            # Extract content for embedding
            contents = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
            
            # Compute embeddings
            embeddings = self._embedding_model.encode(contents)
            
            # Calculate pairwise similarities
            cohesion_scores = {}
            for i, chunk_id in enumerate(chunk_ids):
                similarities = []
                for j, other_embedding in enumerate(embeddings):
                    if i != j:
                        # Cosine similarity
                        similarity = np.dot(embeddings[i], other_embedding) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(other_embedding)
                        )
                        similarities.append(similarity)
                
                # Average similarity as cohesion score
                cohesion_scores[chunk_id] = np.mean(similarities) if similarities else 0.0
            
            return cohesion_scores
            
        except Exception as e:
            logger.error(f"Failed to compute cohesion: {e}")
            return self._compute_basic_cohesion(chunks)
    
    def _compute_basic_cohesion(self, chunks: List[DocumentChunk]) -> Dict[str, float]:
        """Compute basic cohesion without embeddings."""
        cohesion_scores = {}
        
        for chunk in chunks:
            score = 0.0
            
            # Section consistency
            if chunk.metadata.section_title:
                score += 0.3
            
            # Content length consistency
            content_length = len(chunk.content)
            if 200 <= content_length <= 2000:  # Optimal range
                score += 0.4
            
            # Structural elements
            if any(marker in chunk.content for marker in ['\n\n', '- ', '1. ', '* ']):
                score += 0.3
            
            cohesion_scores[chunk.metadata.chunk_id] = min(score, 1.0)
        
        return cohesion_scores


# Factory functions
def create_knowledge_graph_processor(extraction_method: ExtractionMethod = ExtractionMethod.HYBRID,
                                    entity_types: Optional[Set[EntityType]] = None,
                                    relation_types: Optional[Set[RelationType]] = None,
                                    **kwargs) -> KnowledgeGraphProcessor:
    """Create a knowledge graph processor with specified configuration."""
    extraction_config = ExtractionConfig(
        extraction_method=extraction_method,
        entity_types=entity_types or set(EntityType),
        relation_types=relation_types or set(RelationType),
        **kwargs
    )
    
    return KnowledgeGraphProcessor(extraction_config)


async def build_knowledge_graph_from_chunks(chunks: List[DocumentChunk],
                                          llm_results: Optional[BatchProcessingResult] = None,
                                          extraction_method: ExtractionMethod = ExtractionMethod.HYBRID,
                                          **kwargs) -> KnowledgeGraph:
    """Convenience function for building knowledge graph from chunks."""
    processor = create_knowledge_graph_processor(extraction_method, **kwargs)
    return await processor.build_knowledge_graph_async(chunks, llm_results)