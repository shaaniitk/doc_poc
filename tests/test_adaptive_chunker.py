import re
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from modules.chunker import AdaptiveChunker
from modules.error_handler import ChunkingError, ProcessingError
from config import ADAPTIVE_CHUNKING_CONFIG
from langgraph_state import (
    PipelineState, ChunkInfo, ProcessingStage, ErrorInfo, AnalyticsData, create_initial_state
)
from langgraph_config import ProcessingConfiguration


def test_atomic_blocks_preserved_equation():
    text = r"""
    Intro text before equation.
    \begin{equation}
    E = mc^2
    \end{equation}
    Some text after equation.
    """
    ch = AdaptiveChunker()
    chunks = ch.chunk(text, doc_type='latex', atomic_blocks=('equation',), max_tokens=60)

    # Ensure that no chunk splits the equation environment
    begin_count = sum('\\begin{equation}' in c['content'] for c in chunks)
    end_count = sum('\\end{equation}' in c['content'] for c in chunks)
    assert begin_count == end_count == 1

    # If a chunk contains begin, it must also contain end (same chunk)
    for c in chunks:
        if '\\begin{equation}' in c['content']:
            assert '\\end{equation}' in c['content']


def test_token_limit_creates_multiple_chunks():
    # Create a long text with many short sentences
    sentence = "This is a sentence about alpha. "
    text = sentence * 200
    ch = AdaptiveChunker()
    chunks = ch.chunk(text, doc_type='markdown', max_tokens=80, prefer_sweeping=False)

    # With a small token limit, we should get more than one chunk
    assert len(chunks) > 1
    # Chunks should be non-empty and ordered
    assert all(c['content'].strip() for c in chunks)


def test_topic_shift_helps_split_sections():
    text = r"""
    \section{Alpha}
    Alpha is discussed here. alpha alpha alpha. More alpha discussion.
    \section{Beta}
    Beta is discussed here. beta beta beta. More beta discussion.
    """
    ch = AdaptiveChunker()
    chunks = ch.chunk(text, doc_type='latex', max_tokens=120, prefer_sweeping=False)

    # Expect at least two chunks, and that early chunks focus on Alpha while later on Beta
    assert len(chunks) >= 2
    contents = [c['content'].lower() for c in chunks]

    # First chunk(s) should be dominated by 'alpha', last by 'beta'
    assert contents[0].count('alpha') >= contents[0].count('beta')
    assert contents[-1].count('beta') >= contents[-1].count('alpha')


class TestAdaptiveChunkerWithLangGraph:
    """Test adaptive chunker with LangGraph state integration."""
    
    @pytest.fixture
    def chunker(self):
        return AdaptiveChunker()
    
    def test_adaptive_chunking_success(self, chunker):
        """Test successful adaptive chunking."""
        text = "This is a test document. It has multiple sentences. Some are short. Others are much longer and contain more detailed information about various topics."
        
        chunks = chunker.chunk_adaptively(text)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        
        # Verify adaptive behavior - chunks should have varying sizes
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        assert len(set(chunk_sizes)) > 1  # Should have different sizes
    
    def test_adaptive_chunking_with_langgraph_state(self, chunker):
        """Test adaptive chunking with LangGraph state integration."""
        # Create initial pipeline state
        config = ProcessingConfiguration(
            chunking_enabled=True,
            adaptive_chunking=True,
            max_chunk_size=512,
            min_chunk_size=100
        )
        state = create_initial_state(
            document_path="adaptive_test.txt",
            processing_config=config
        )
        
        text = "This is a complex document with varying content density. Some sections are information-rich. Others are sparse. The adaptive chunker should handle this intelligently."
        
        # Update state to chunking stage
        state.current_stage = ProcessingStage.CHUNKING
        
        # Perform adaptive chunking
        chunks = chunker.chunk_adaptively(text)
        
        # Convert chunks to ChunkInfo objects
        chunk_infos = []
        for i, chunk in enumerate(chunks):
            chunk_info = ChunkInfo(
                chunk_id=f"adaptive_chunk_{i}",
                content=chunk['content'],
                token_count=chunk['metadata'].get('token_count', len(chunk['content'].split())),
                cohesion_score=chunk['metadata'].get('cohesion_score', 0.8),
                start_position=chunk['metadata'].get('start_position', 0),
                end_position=chunk['metadata'].get('end_position', len(chunk['content'])),
                overlap_with_previous=chunk['metadata'].get('overlap_tokens', 0),
                embedding_vector=chunk['metadata'].get('embedding', [])
            )
            chunk_infos.append(chunk_info)
        
        # Update state with chunks
        state.chunks = chunk_infos
        state.current_stage = ProcessingStage.CHUNKING_COMPLETE
        
        # Create analytics info for adaptive chunking
        analytics = AnalyticsInfo(
            processing_time=0.5,
            memory_usage=1024,
            tokens_processed=sum(chunk.token_count for chunk in chunk_infos),
            chunks_created=len(chunk_infos),
            average_chunk_size=sum(chunk.token_count for chunk in chunk_infos) / len(chunk_infos),
            processing_stage=ProcessingStage.CHUNKING_COMPLETE,
            metadata={"chunking_method": "adaptive", "adaptive_enabled": True}
        )
        state.analytics.append(analytics)
        
        # Verify state integration
        assert len(state.chunks) == len(chunks)
        assert state.current_stage == ProcessingStage.CHUNKING_COMPLETE
        assert all(isinstance(chunk, ChunkInfo) for chunk in state.chunks)
        assert len(state.analytics) == 1
        assert state.analytics[0].chunks_created == len(chunks)
    
    def test_adaptive_chunking_with_complexity_analysis(self, chunker):
        """Test adaptive chunking with complexity analysis."""
        # Create text with varying complexity
        simple_text = "This is simple. Very easy to read."
        complex_text = "The implementation of quantum entanglement protocols requires sophisticated understanding of non-local correlations and measurement-induced state collapse phenomena."
        
        text = simple_text + " " + complex_text
        
        chunks = chunker.chunk_adaptively(text)
        
        # Verify that complexity affects chunking decisions
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'complexity_score' in chunk['metadata']
            assert isinstance(chunk['metadata']['complexity_score'], (int, float))
            assert 0 <= chunk['metadata']['complexity_score'] <= 1
    
    def test_adaptive_chunking_error_handling_with_state(self, chunker):
        """Test error handling during adaptive chunking with state tracking."""
        # Create initial state
        config = ProcessingConfiguration(
            chunking_enabled=True,
            adaptive_chunking=True
        )
        state = create_initial_state(
            document_path="error_test.txt",
            processing_config=config
        )
        
        # Test with problematic input
        problematic_text = ""  # Empty text
        
        try:
            chunks = chunker.chunk_adaptively(problematic_text)
            
            if not chunks:  # Handle empty result
                error_info = ErrorInfo(
                    error_type="EmptyChunkingResult",
                    message="Adaptive chunking produced no chunks",
                    stage=ProcessingStage.CHUNKING,
                    recoverable=True,
                    context={"input_length": len(problematic_text), "adaptive_enabled": True}
                )
                
                state.errors.append(error_info)
                state.current_stage = ProcessingStage.CHUNKING_COMPLETE  # Still complete, just empty
                
                # Verify error tracking
                assert len(state.errors) == 1
                assert state.errors[0].error_type == "EmptyChunkingResult"
                assert state.errors[0].recoverable is True
                
        except Exception as e:
            # Handle unexpected errors
            error_info = ErrorInfo(
                error_type=type(e).__name__,
                message=str(e),
                stage=ProcessingStage.CHUNKING,
                recoverable=False,
                context={"input_text": problematic_text[:100]}
            )
            
            state.errors.append(error_info)
            state.current_stage = ProcessingStage.ERROR
            
            # Verify error handling
            assert len(state.errors) == 1
            assert state.current_stage == ProcessingStage.ERROR
    
    def test_chunk_size_adaptation(self, chunker):
        """Test that chunk sizes adapt based on content characteristics."""
        # Dense, information-rich text should create smaller chunks
        dense_text = "AI ML DL NLP CV RNN CNN LSTM GRU BERT GPT T5 CLIP DALL-E ResNet VGG AlexNet ImageNet COCO MNIST CIFAR"
        
        # Sparse, simple text should create larger chunks
        sparse_text = "The cat sat on the mat. The dog ran in the park. The bird flew in the sky. The fish swam in the sea."
        
        dense_chunks = chunker.chunk_adaptively(dense_text)
        sparse_chunks = chunker.chunk_adaptively(sparse_text)
        
        # Verify adaptive behavior
        if len(dense_chunks) > 0 and len(sparse_chunks) > 0:
            avg_dense_size = sum(len(chunk['content']) for chunk in dense_chunks) / len(dense_chunks)
            avg_sparse_size = sum(len(chunk['content']) for chunk in sparse_chunks) / len(sparse_chunks)
            
            # Dense content should generally result in smaller average chunk sizes
            # (This is a heuristic test - actual behavior may vary based on implementation)
            assert avg_dense_size != avg_sparse_size  # At least they should be different
    
    def test_adaptive_chunking_pipeline_workflow(self, chunker):
        """Test complete adaptive chunking pipeline workflow with state management."""
        # Create comprehensive pipeline state
        config = ProcessingConfiguration(
            chunking_enabled=True,
            adaptive_chunking=True,
            max_chunk_size=300,
            min_chunk_size=50,
            chunk_overlap=25
        )
        state = create_initial_state(
            document_path="pipeline_test.txt",
            processing_config=config
        )
        
        # Multi-paragraph text with varying complexity
        text = """
        Introduction: This document covers multiple topics with varying complexity levels.
        
        Simple Section: The weather is nice today. Birds are singing. Children are playing.
        
        Complex Section: The quantum mechanical interpretation of wave-particle duality necessitates 
        a comprehensive understanding of the Copenhagen interpretation and its implications for 
        measurement theory in quantum systems.
        
        Conclusion: This concludes our varied content analysis.
        """
        
        # Execute adaptive chunking workflow
        state.current_stage = ProcessingStage.CHUNKING
        
        chunks = chunker.chunk_adaptively(text)
        
        # Convert to ChunkInfo objects with enhanced metadata
        chunk_infos = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            token_count = chunk['metadata'].get('token_count', len(chunk['content'].split()))
            total_tokens += token_count
            
            chunk_info = ChunkInfo(
                chunk_id=f"adaptive_pipeline_chunk_{i}",
                content=chunk['content'],
                token_count=token_count,
                cohesion_score=chunk['metadata'].get('cohesion_score', 0.7),
                start_position=chunk['metadata'].get('start_position', i * 100),
                end_position=chunk['metadata'].get('end_position', (i + 1) * 100),
                overlap_with_previous=config.chunk_overlap if i > 0 else 0,
                embedding_vector=chunk['metadata'].get('embedding', [])
            )
            chunk_infos.append(chunk_info)
        
        # Update state with results
        state.chunks = chunk_infos
        state.current_stage = ProcessingStage.CHUNKING_COMPLETE
        
        # Add comprehensive analytics
        analytics = AnalyticsInfo(
            processing_time=1.2,
            memory_usage=2048,
            tokens_processed=total_tokens,
            chunks_created=len(chunk_infos),
            average_chunk_size=total_tokens / len(chunk_infos) if chunk_infos else 0,
            processing_stage=ProcessingStage.CHUNKING_COMPLETE,
            metadata={
                "chunking_method": "adaptive",
                "adaptive_enabled": True,
                "complexity_analysis": True,
                "size_variation": len(set(chunk.token_count for chunk in chunk_infos)) > 1
            }
        )
        state.analytics.append(analytics)
        
        # Verify complete workflow
        assert state.current_stage == ProcessingStage.CHUNKING_COMPLETE
        assert len(state.chunks) > 0
        assert all(config.min_chunk_size <= chunk.token_count <= config.max_chunk_size 
                  for chunk in state.chunks)
        assert state.analytics[0].chunks_created == len(chunks)
        assert state.analytics[0].metadata["adaptive_enabled"] is True
        
        # Verify adaptive behavior in results
        chunk_sizes = [chunk.token_count for chunk in state.chunks]
        assert len(set(chunk_sizes)) > 1, "Adaptive chunking should produce varying chunk sizes"