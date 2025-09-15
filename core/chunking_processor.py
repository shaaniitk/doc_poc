"""Enhanced Chunking Processor for LangGraph Integration

This module provides intelligent document chunking with semantic awareness,
overlap management, and integration with the LangGraph orchestrator.
It supports multiple chunking strategies optimized for different use cases.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import hashlib
from datetime import datetime, timezone
import math

# NLP imports for semantic chunking
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    logging.warning("NLTK not available. Install nltk for advanced text processing.")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logging.warning("spaCy not available. Install spacy for advanced NLP features.")

from .document_parser import ParsedDocument, DocumentMetadata

# LaTeX-specific patterns for chunking
LATEX_SECTION_PATTERNS = [
    r'\\section\{[^}]*\}',
    r'\\subsection\{[^}]*\}',
    r'\\subsubsection\{[^}]*\}',
    r'\\chapter\{[^}]*\}',
    r'\\part\{[^}]*\}'
]

LATEX_ENVIRONMENT_PATTERNS = [
    r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}',
    r'\\\[.*?\\\]',  # Display math
    r'\$\$.*?\$\$',    # Display math alternative
]

LATEX_COMMAND_PATTERNS = [
    r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*'
]

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Different chunking strategies for various use cases."""
    FIXED_SIZE = auto()        # Fixed character/token count
    SENTENCE_AWARE = auto()    # Respect sentence boundaries
    PARAGRAPH_AWARE = auto()   # Respect paragraph boundaries
    SEMANTIC = auto()          # Semantic similarity-based chunking
    SLIDING_WINDOW = auto()    # Overlapping sliding windows
    HIERARCHICAL = auto()      # Multi-level hierarchical chunks
    ADAPTIVE = auto()          # Adaptive based on content type
    CUSTOM = auto()            # Custom chunking logic
    LATEX_AWARE = auto()       # LaTeX-aware chunking


class OverlapStrategy(Enum):
    """Strategies for handling chunk overlaps."""
    NONE = auto()              # No overlap
    FIXED_TOKENS = auto()      # Fixed number of tokens
    FIXED_SENTENCES = auto()   # Fixed number of sentences
    PERCENTAGE = auto()        # Percentage of chunk size
    SEMANTIC_BOUNDARY = auto() # Overlap at semantic boundaries


@dataclass
class ChunkingConfig:
    """Configuration for chunking operations."""
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_AWARE
    max_chunk_size: int = 1000  # Maximum characters per chunk
    min_chunk_size: int = 100   # Minimum characters per chunk
    overlap_strategy: OverlapStrategy = OverlapStrategy.FIXED_SENTENCES
    overlap_size: int = 2       # Overlap amount (depends on strategy)
    preserve_formatting: bool = True
    respect_boundaries: bool = True  # Respect natural text boundaries
    target_chunks: Optional[int] = None  # Target number of chunks
    quality_threshold: float = 0.7  # Minimum quality score for chunks
    enable_metadata: bool = True
    custom_separators: List[str] = field(default_factory=lambda: ['\n\n', '\n', '. ', '! ', '? '])
    language: str = "en"


@dataclass
class ChunkMetadata:
    """Metadata for individual chunks."""
    chunk_id: str
    index: int
    start_position: int
    end_position: int
    character_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    quality_score: float = 0.0
    semantic_density: float = 0.0
    readability_score: float = 0.0
    contains_tables: bool = False
    contains_lists: bool = False
    contains_code: bool = False
    language: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    creation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class DocumentChunk:
    """Container for a document chunk with metadata and relationships."""
    content: str
    metadata: ChunkMetadata
    source_document: Optional[str] = None
    parent_chunk: Optional[str] = None
    child_chunks: List[str] = field(default_factory=list)
    related_chunks: List[str] = field(default_factory=list)
    embeddings: Optional[List[float]] = None
    hash: Optional[str] = None
    
    def __post_init__(self):
        """Generate hash for content integrity."""
        if not self.hash:
            self.hash = hashlib.sha256(self.content.encode('utf-8')).hexdigest()[:16]


@dataclass
class ChunkingResult:
    """Result of chunking operation with comprehensive metadata."""
    chunks: List[DocumentChunk]
    total_chunks: int
    total_characters: int
    total_words: int
    average_chunk_size: float
    chunking_strategy: ChunkingStrategy
    overlap_strategy: OverlapStrategy
    processing_time: float
    quality_metrics: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkingProcessor:
    """Enhanced chunking processor with multiple strategies and semantic awareness."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self._nlp_model = None
        self._sentence_tokenizer = None
        
        # Initialize NLP components if available
        self._initialize_nlp()
        
        logger.info(f"ChunkingProcessor initialized with strategy: {self.config.strategy.name}")
    
    def _initialize_nlp(self) -> None:
        """Initialize NLP components for advanced processing."""
        if HAS_NLTK:
            try:
                # Download required NLTK data
                import nltk
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self._sentence_tokenizer = sent_tokenize
                logger.debug("NLTK initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NLTK: {e}")
        
        if HAS_SPACY:
            try:
                # Try to load English model
                import spacy
                self._nlp_model = spacy.load("en_core_web_sm")
                logger.debug("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            except Exception as e:
                logger.warning(f"Failed to initialize spaCy: {e}")
    
    async def process_document_async(self, document: ParsedDocument, 
                                   config: Optional[ChunkingConfig] = None) -> ChunkingResult:
        """Asynchronously process document into chunks."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.process_document, document, config
        )
    
    def process_document(self, document: ParsedDocument, 
                        config: Optional[ChunkingConfig] = None) -> ChunkingResult:
        """Process document into chunks using specified strategy."""
        start_time = datetime.now(timezone.utc)
        config = config or self.config
        
        try:
            # Validate input
            if not document.content.strip():
                raise ValueError("Document content is empty")
            
            # Choose chunking strategy
            chunks = self._chunk_by_strategy(document, config)
            
            # Post-process chunks
            chunks = self._post_process_chunks(chunks, config)
            
            # Calculate metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            quality_metrics = self._calculate_quality_metrics(chunks)
            
            result = ChunkingResult(
                chunks=chunks,
                total_chunks=len(chunks),
                total_characters=sum(len(chunk.content) for chunk in chunks),
                total_words=sum(chunk.metadata.word_count for chunk in chunks),
                average_chunk_size=sum(len(chunk.content) for chunk in chunks) / len(chunks) if chunks else 0,
                chunking_strategy=config.strategy,
                overlap_strategy=config.overlap_strategy,
                processing_time=processing_time,
                quality_metrics=quality_metrics
            )
            
            logger.info(f"Successfully chunked document into {len(chunks)} chunks in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to chunk document: {e}")
            raise ChunkingError(f"Chunking failed: {e}") from e
    
    def _chunk_by_strategy(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Chunk document using specified strategy."""
        strategies = {
            ChunkingStrategy.FIXED_SIZE: self._chunk_fixed_size,
            ChunkingStrategy.SENTENCE_AWARE: self._chunk_sentence_aware,
            ChunkingStrategy.PARAGRAPH_AWARE: self._chunk_paragraph_aware,
            ChunkingStrategy.SEMANTIC: self._chunk_semantic,
            ChunkingStrategy.SLIDING_WINDOW: self._chunk_sliding_window,
            ChunkingStrategy.HIERARCHICAL: self._chunk_hierarchical,
            ChunkingStrategy.ADAPTIVE: self._chunk_adaptive,
            ChunkingStrategy.CUSTOM: self._chunk_custom,
            ChunkingStrategy.LATEX_AWARE: self._chunk_latex_aware
        }
        
        strategy_func = strategies.get(config.strategy, self._chunk_sentence_aware)
        return strategy_func(document, config)
    
    def _chunk_fixed_size(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Chunk document into fixed-size pieces."""
        content = document.content
        chunks = []
        
        for i in range(0, len(content), config.max_chunk_size):
            chunk_content = content[i:i + config.max_chunk_size]
            
            # Skip chunks that are too small
            if len(chunk_content.strip()) < config.min_chunk_size:
                continue
            
            chunk_id = f"chunk_{len(chunks):04d}"
            metadata = self._create_chunk_metadata(
                chunk_id, len(chunks), i, i + len(chunk_content), chunk_content
            )
            
            chunks.append(DocumentChunk(
                content=chunk_content,
                metadata=metadata,
                source_document=str(document.metadata.file_path)
            ))
        
        return chunks
    
    def _chunk_sentence_aware(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Chunk document respecting sentence boundaries."""
        content = document.content
        
        # Get sentences
        sentences = self._get_sentences(content)
        if not sentences:
            return self._chunk_fixed_size(document, config)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed max size
            if current_size + sentence_size > config.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_content = ' '.join(current_chunk)
                if len(chunk_content.strip()) >= config.min_chunk_size:
                    chunk_id = f"chunk_{len(chunks):04d}"
                    start_pos = content.find(current_chunk[0])
                    end_pos = start_pos + len(chunk_content)
                    
                    metadata = self._create_chunk_metadata(
                        chunk_id, len(chunks), start_pos, end_pos, chunk_content
                    )
                    metadata.sentence_count = len(current_chunk)
                    
                    chunks.append(DocumentChunk(
                        content=chunk_content,
                        metadata=metadata,
                        source_document=str(document.metadata.file_path)
                    ))
                
                # Start new chunk
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            if len(chunk_content.strip()) >= config.min_chunk_size:
                chunk_id = f"chunk_{len(chunks):04d}"
                start_pos = content.find(current_chunk[0])
                end_pos = start_pos + len(chunk_content)
                
                metadata = self._create_chunk_metadata(
                    chunk_id, len(chunks), start_pos, end_pos, chunk_content
                )
                metadata.sentence_count = len(current_chunk)
                
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    metadata=metadata,
                    source_document=str(document.metadata.file_path)
                ))
        
        return chunks
    
    def _chunk_paragraph_aware(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Chunk document respecting paragraph boundaries."""
        content = document.content
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return self._chunk_sentence_aware(document, config)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            # If single paragraph is too large, split it
            if para_size > config.max_chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    if len(chunk_content.strip()) >= config.min_chunk_size:
                        chunks.append(self._create_chunk_from_content(
                            chunk_content, len(chunks), document, content
                        ))
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph using sentence-aware method
                para_doc = ParsedDocument(content=paragraph, metadata=document.metadata)
                para_chunks = self._chunk_sentence_aware(para_doc, config)
                chunks.extend(para_chunks)
                
            elif current_size + para_size > config.max_chunk_size and current_chunk:
                # Create chunk from current paragraphs
                chunk_content = '\n\n'.join(current_chunk)
                if len(chunk_content.strip()) >= config.min_chunk_size:
                    chunks.append(self._create_chunk_from_content(
                        chunk_content, len(chunks), document, content
                    ))
                
                # Start new chunk
                current_chunk = [paragraph]
                current_size = para_size
            else:
                current_chunk.append(paragraph)
                current_size += para_size + 2  # Account for \n\n
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            if len(chunk_content.strip()) >= config.min_chunk_size:
                chunks.append(self._create_chunk_from_content(
                    chunk_content, len(chunks), document, content
                ))
        
        return chunks
    
    def _chunk_semantic(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Chunk document based on semantic similarity (requires NLP models)."""
        if not self._nlp_model:
            logger.warning("Semantic chunking requires spaCy. Falling back to sentence-aware chunking.")
            return self._chunk_sentence_aware(document, config)
        
        content = document.content
        sentences = self._get_sentences(content)
        
        if len(sentences) < 2:
            return self._chunk_sentence_aware(document, config)
        
        try:
            # Process sentences with spaCy
            sentence_docs = [self._nlp_model(sent) for sent in sentences]
            
            # Calculate semantic similarities
            similarities = []
            for i in range(len(sentence_docs) - 1):
                sim = sentence_docs[i].similarity(sentence_docs[i + 1])
                similarities.append(sim)
            
            # Find semantic boundaries (low similarity points)
            threshold = sum(similarities) / len(similarities) - 0.1  # Below average
            boundaries = [0]
            
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    boundaries.append(i + 1)
            
            boundaries.append(len(sentences))
            
            # Create chunks from semantic segments
            chunks = []
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]
                
                segment_sentences = sentences[start_idx:end_idx]
                chunk_content = ' '.join(segment_sentences)
                
                # Ensure chunk size constraints
                if len(chunk_content) > config.max_chunk_size:
                    # Split large semantic chunk
                    segment_doc = ParsedDocument(content=chunk_content, metadata=document.metadata)
                    sub_chunks = self._chunk_sentence_aware(segment_doc, config)
                    chunks.extend(sub_chunks)
                elif len(chunk_content.strip()) >= config.min_chunk_size:
                    chunks.append(self._create_chunk_from_content(
                        chunk_content, len(chunks), document, content
                    ))
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}. Falling back to sentence-aware.")
            return self._chunk_sentence_aware(document, config)
    
    def _chunk_sliding_window(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Create overlapping chunks using sliding window approach."""
        content = document.content
        sentences = self._get_sentences(content)
        
        if not sentences:
            return self._chunk_fixed_size(document, config)
        
        chunks = []
        window_size = self._calculate_window_size(sentences, config)
        step_size = max(1, window_size - config.overlap_size)
        
        for i in range(0, len(sentences), step_size):
            window_sentences = sentences[i:i + window_size]
            chunk_content = ' '.join(window_sentences)
            
            if len(chunk_content.strip()) >= config.min_chunk_size:
                chunk_id = f"chunk_{len(chunks):04d}"
                start_pos = content.find(window_sentences[0]) if window_sentences else 0
                end_pos = start_pos + len(chunk_content)
                
                metadata = self._create_chunk_metadata(
                    chunk_id, len(chunks), start_pos, end_pos, chunk_content
                )
                metadata.sentence_count = len(window_sentences)
                
                # Calculate overlap
                if chunks:
                    prev_sentences = chunks[-1].content.split('. ')
                    overlap_count = len(set(window_sentences) & set(prev_sentences))
                    metadata.overlap_with_previous = overlap_count
                
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    metadata=metadata,
                    source_document=str(document.metadata.file_path)
                ))
            
            # Break if we've covered all sentences
            if i + window_size >= len(sentences):
                break
        
        return chunks
    
    def _chunk_hierarchical(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Create hierarchical chunks with parent-child relationships."""
        # First create large parent chunks
        parent_config = ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH_AWARE,
            max_chunk_size=config.max_chunk_size * 3,
            min_chunk_size=config.min_chunk_size * 2
        )
        
        parent_chunks = self._chunk_paragraph_aware(document, parent_config)
        
        # Then create child chunks from each parent
        all_chunks = []
        
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            # Add parent chunk
            parent_chunk.metadata.chunk_id = f"parent_{parent_idx:04d}"
            all_chunks.append(parent_chunk)
            
            # Create child chunks if parent is large enough
            if len(parent_chunk.content) > config.max_chunk_size:
                child_doc = ParsedDocument(content=parent_chunk.content, metadata=document.metadata)
                child_chunks = self._chunk_sentence_aware(child_doc, config)
                
                for child_idx, child_chunk in enumerate(child_chunks):
                    child_chunk.metadata.chunk_id = f"child_{parent_idx:04d}_{child_idx:04d}"
                    child_chunk.parent_chunk = parent_chunk.metadata.chunk_id
                    parent_chunk.child_chunks.append(child_chunk.metadata.chunk_id)
                    all_chunks.append(child_chunk)
        
        return all_chunks
    
    def _chunk_adaptive(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Adaptively choose chunking strategy based on content characteristics."""
        content = document.content
        
        # Check for LaTeX content first
        has_latex_sections = any(section.get('type') == 'latex_section' for section in document.sections)
        has_latex_commands = bool(re.search(r'\\[a-zA-Z]+', content))
        
        # Analyze content characteristics
        has_clear_paragraphs = '\n\n' in content and len(content.split('\n\n')) > 2
        has_structured_sections = any(section.get('type') == 'header' for section in document.sections)
        avg_sentence_length = len(content.split('. ')) / max(1, len(content.split('\n')))
        
        # Choose strategy based on analysis
        if has_latex_sections or has_latex_commands:
            chosen_strategy = ChunkingStrategy.LATEX_AWARE
        elif has_structured_sections and self._nlp_model:
            chosen_strategy = ChunkingStrategy.SEMANTIC
        elif has_clear_paragraphs:
            chosen_strategy = ChunkingStrategy.PARAGRAPH_AWARE
        elif avg_sentence_length > 20:  # Long sentences
            chosen_strategy = ChunkingStrategy.SENTENCE_AWARE
        else:
            chosen_strategy = ChunkingStrategy.SLIDING_WINDOW
        
        # Update config and chunk
        adaptive_config = ChunkingConfig(
            strategy=chosen_strategy,
            max_chunk_size=config.max_chunk_size,
            min_chunk_size=config.min_chunk_size,
            overlap_strategy=config.overlap_strategy,
            overlap_size=config.overlap_size
        )
        
        logger.info(f"Adaptive chunking chose strategy: {chosen_strategy.name}")
        return self._chunk_by_strategy(document, adaptive_config)
    
    def _chunk_latex_aware(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Chunk LaTeX documents respecting LaTeX structure."""
        content = document.content
        chunks = []
        
        # Find LaTeX sections from document metadata
        latex_sections = [s for s in document.sections if s.get('type') == 'latex_section']
        
        if latex_sections:
            # Chunk by LaTeX sections
            lines = content.split('\n')
            current_chunk_lines = []
            current_section = None
            
            for i, line in enumerate(lines, 1):
                # Check if this line starts a new section
                section_at_line = next((s for s in latex_sections if s.get('line_number') == i), None)
                
                if section_at_line and current_chunk_lines:
                    # Finalize current chunk
                    chunk_content = '\n'.join(current_chunk_lines)
                    if len(chunk_content.strip()) >= config.min_chunk_size:
                        chunks.append(self._create_chunk_from_content(
                            chunk_content, len(chunks), document, content
                        ))
                    current_chunk_lines = []
                
                if section_at_line:
                    current_section = section_at_line
                
                current_chunk_lines.append(line)
                
                # Check if chunk is getting too large
                current_content = '\n'.join(current_chunk_lines)
                if len(current_content) > config.max_chunk_size:
                    # Split at natural boundaries within the section
                    sub_chunks = self._split_large_latex_chunk(current_content, config, current_section, document, content)
                    chunks.extend(sub_chunks)
                    current_chunk_lines = []
            
            # Handle remaining content
            if current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines)
                if len(chunk_content.strip()) >= config.min_chunk_size:
                    chunks.append(self._create_chunk_from_content(
                        chunk_content, len(chunks), document, content
                    ))
        else:
            # Fallback to environment-aware chunking
            chunks = self._chunk_by_latex_environments(document, config)
        
        return chunks
    
    def _split_large_latex_chunk(self, content: str, config: ChunkingConfig, section: Dict, 
                                document: ParsedDocument, full_content: str) -> List[DocumentChunk]:
        """Split large LaTeX chunks at natural boundaries."""
        chunks = []
        
        # Try to split at paragraph boundaries first
        paragraphs = content.split('\n\n')
        current_chunk = ''
        
        for para in paragraphs:
            if len(current_chunk + para) > config.max_chunk_size and current_chunk:
                chunks.append(self._create_chunk_from_content(
                    current_chunk.strip(), len(chunks), document, full_content
                ))
                current_chunk = para + '\n\n'
            else:
                current_chunk += para + '\n\n'
        
        if current_chunk.strip():
            chunks.append(self._create_chunk_from_content(
                current_chunk.strip(), len(chunks), document, full_content
            ))
        
        return chunks
    
    def _chunk_by_latex_environments(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Chunk by LaTeX environments when no clear sections are found."""
        content = document.content
        chunks = []
        
        # Find LaTeX environments
        env_pattern = r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}'
        matches = list(re.finditer(env_pattern, content, re.DOTALL))
        
        last_end = 0
        for match in matches:
            # Add content before environment
            before_content = content[last_end:match.start()].strip()
            if before_content and len(before_content) >= config.min_chunk_size:
                chunks.append(self._create_chunk_from_content(
                    before_content, len(chunks), document, content
                ))
            
            # Add environment as separate chunk
            env_content = match.group(0)
            if len(env_content) >= config.min_chunk_size:
                chunks.append(self._create_chunk_from_content(
                    env_content, len(chunks), document, content
                ))
            
            last_end = match.end()
        
        # Add remaining content
        remaining = content[last_end:].strip()
        if remaining and len(remaining) >= config.min_chunk_size:
            chunks.append(self._create_chunk_from_content(
                remaining, len(chunks), document, content
            ))
        
        return chunks
    
    def _chunk_custom(self, document: ParsedDocument, config: ChunkingConfig) -> List[DocumentChunk]:
        """Custom chunking logic - can be overridden by subclasses."""
        logger.warning("Custom chunking not implemented. Using sentence-aware chunking.")
        return self._chunk_sentence_aware(document, config)
    
    def _get_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using available tokenizers."""
        if self._sentence_tokenizer:
            try:
                return [s.strip() for s in self._sentence_tokenizer(text) if s.strip()]
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}")
        
        # Fallback to simple regex-based sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_window_size(self, sentences: List[str], config: ChunkingConfig) -> int:
        """Calculate optimal window size for sliding window chunking."""
        if not sentences:
            return 1
        
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
        target_sentences = max(1, int(config.max_chunk_size / avg_sentence_length))
        
        return min(target_sentences, len(sentences))
    
    def _create_chunk_metadata(self, chunk_id: str, index: int, start_pos: int, 
                              end_pos: int, content: str) -> ChunkMetadata:
        """Create comprehensive metadata for a chunk."""
        words = content.split()
        sentences = self._get_sentences(content)
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            index=index,
            start_position=start_pos,
            end_position=end_pos,
            character_count=len(content),
            word_count=len(words),
            sentence_count=len(sentences),
            paragraph_count=len(paragraphs)
        )
        
        # Detect content types
        metadata.contains_tables = 'table' in content.lower() or '|' in content
        metadata.contains_lists = bool(re.search(r'^\s*[-*+]\s+', content, re.MULTILINE))
        metadata.contains_code = bool(re.search(r'```|`[^`]+`|def |class |import ', content))
        
        # Calculate quality score
        metadata.quality_score = self._calculate_chunk_quality(content)
        
        return metadata
    
    def _create_chunk_from_content(self, content: str, index: int, 
                                  document: ParsedDocument, full_content: str) -> DocumentChunk:
        """Helper to create chunk from content string."""
        chunk_id = f"chunk_{index:04d}"
        start_pos = full_content.find(content)
        end_pos = start_pos + len(content)
        
        metadata = self._create_chunk_metadata(chunk_id, index, start_pos, end_pos, content)
        
        return DocumentChunk(
            content=content,
            metadata=metadata,
            source_document=str(document.metadata.file_path)
        )
    
    def _calculate_chunk_quality(self, content: str) -> float:
        """Calculate quality score for a chunk."""
        scores = []
        
        # Length score
        length = len(content.strip())
        if 200 <= length <= 1500:
            scores.append(1.0)
        elif 100 <= length < 200 or 1500 < length <= 2000:
            scores.append(0.8)
        elif 50 <= length < 100 or 2000 < length <= 3000:
            scores.append(0.6)
        else:
            scores.append(0.3)
        
        # Completeness score (ends with sentence)
        if content.strip().endswith(('.', '!', '?', ':', ';')):
            scores.append(1.0)
        else:
            scores.append(0.7)
        
        # Coherence score (simple heuristic)
        sentences = self._get_sentences(content)
        if len(sentences) >= 2:
            scores.append(0.9)
        elif len(sentences) == 1:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _post_process_chunks(self, chunks: List[DocumentChunk], config: ChunkingConfig) -> List[DocumentChunk]:
        """Post-process chunks to add overlaps and relationships."""
        if not chunks:
            return chunks
        
        # Add overlaps based on strategy
        if config.overlap_strategy != OverlapStrategy.NONE:
            chunks = self._add_overlaps(chunks, config)
        
        # Filter by quality if threshold is set
        if config.quality_threshold > 0:
            chunks = [chunk for chunk in chunks 
                     if chunk.metadata.quality_score >= config.quality_threshold]
        
        # Update indices after filtering
        for i, chunk in enumerate(chunks):
            chunk.metadata.index = i
        
        return chunks
    
    def _add_overlaps(self, chunks: List[DocumentChunk], config: ChunkingConfig) -> List[DocumentChunk]:
        """Add overlaps between chunks based on strategy."""
        if len(chunks) < 2:
            return chunks
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            if config.overlap_strategy == OverlapStrategy.FIXED_SENTENCES:
                # Add last N sentences from previous chunk
                prev_sentences = self._get_sentences(prev_chunk.content)
                if len(prev_sentences) > config.overlap_size:
                    overlap_sentences = prev_sentences[-config.overlap_size:]
                    overlap_text = ' '.join(overlap_sentences)
                    curr_chunk.content = overlap_text + ' ' + curr_chunk.content
                    curr_chunk.metadata.overlap_with_previous = len(overlap_sentences)
            
            elif config.overlap_strategy == OverlapStrategy.FIXED_TOKENS:
                # Add last N words from previous chunk
                prev_words = prev_chunk.content.split()
                if len(prev_words) > config.overlap_size:
                    overlap_words = prev_words[-config.overlap_size:]
                    overlap_text = ' '.join(overlap_words)
                    curr_chunk.content = overlap_text + ' ' + curr_chunk.content
                    curr_chunk.metadata.overlap_with_previous = len(overlap_words)
            
            elif config.overlap_strategy == OverlapStrategy.PERCENTAGE:
                # Add percentage of previous chunk
                prev_content = prev_chunk.content
                overlap_size = int(len(prev_content) * (config.overlap_size / 100))
                if overlap_size > 0:
                    overlap_text = prev_content[-overlap_size:]
                    curr_chunk.content = overlap_text + ' ' + curr_chunk.content
                    curr_chunk.metadata.overlap_with_previous = overlap_size
        
        return chunks
    
    def _calculate_quality_metrics(self, chunks: List[DocumentChunk]) -> Dict[str, float]:
        """Calculate overall quality metrics for chunking result."""
        if not chunks:
            return {}
        
        quality_scores = [chunk.metadata.quality_score for chunk in chunks]
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        
        return {
            "average_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "size_variance": self._calculate_variance(chunk_sizes),
            "completeness_ratio": sum(1 for chunk in chunks 
                                    if chunk.content.strip().endswith(('.', '!', '?'))) / len(chunks)
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)


class ChunkingError(Exception):
    """Custom exception for chunking errors."""
    pass


# Factory functions
def create_chunking_processor(strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_AWARE,
                            max_chunk_size: int = 1000,
                            **kwargs) -> ChunkingProcessor:
    """Create a chunking processor with specified configuration."""
    config = ChunkingConfig(
        strategy=strategy,
        max_chunk_size=max_chunk_size,
        **kwargs
    )
    return ChunkingProcessor(config)


async def chunk_document_async(document: ParsedDocument,
                             strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_AWARE,
                             max_chunk_size: int = 1000) -> ChunkingResult:
    """Convenience function for async document chunking."""
    processor = create_chunking_processor(strategy, max_chunk_size)
    return await processor.process_document_async(document)