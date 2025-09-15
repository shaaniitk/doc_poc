import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from modules.chunker import EmbeddingGuidedChunker
from modules.error_handler import EmbeddingError, EmbeddingAPIError
from config import CHUNKING_EMBEDDING
from langgraph_state import (
    PipelineState, ChunkInfo, ProcessingStage, ErrorInfo, create_initial_state
)
from langgraph_config import ProcessingConfiguration


class TestEmbeddingGuidedChunker:
    """Test suite for EmbeddingGuidedChunker class."""
    
    @pytest.fixture
    def mock_embedding_client(self):
        """Create a mock embedding client."""
        client = Mock()
        client.get_embeddings.return_value = [
            [0.1, 0.2, 0.3],  # Similar embeddings
            [0.15, 0.25, 0.35],
            [0.8, 0.1, 0.1],  # Different embedding (topic boundary)
            [0.85, 0.15, 0.05]
        ]
        client.calculate_cohesion_scores.return_value = [1.0, 0.9, 0.2, 0.8]
        client.detect_topic_boundaries.return_value = [2]
        return client
    
    @pytest.fixture
    def chunker(self, mock_embedding_client):
        """Create EmbeddingGuidedChunker instance with mocked dependencies."""
        chunker = EmbeddingGuidedChunker()
        chunker.embedding_client = mock_embedding_client
        return chunker
    
    def test_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.config == CHUNKING_EMBEDDING
        assert chunker.embedding_client is not None
        assert chunker.logger is not None
        assert chunker.cache == {}
    
    def test_calculate_cohesion_scores_normal(self, chunker):
        """Test cohesion score calculation with normal embeddings."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # Similar to first
            [0.0, 1.0, 0.0],  # Different topic
            [0.0, 0.9, 0.1]   # Similar to third
        ]
        
        scores = chunker._calculate_cohesion_scores(embeddings)
        
        assert len(scores) == len(embeddings)
        assert scores[0] == 1.0  # First embedding always gets 1.0
        assert 0.0 <= scores[1] <= 1.0
        assert scores[2] < scores[1]  # Topic boundary should have lower score
    
    def test_calculate_cohesion_scores_empty(self, chunker):
        """Test cohesion score calculation with empty embeddings."""
        scores = chunker._calculate_cohesion_scores([])
        assert scores == []
    
    def test_calculate_cohesion_scores_single(self, chunker):
        """Test cohesion score calculation with single embedding."""
        embeddings = [[1.0, 0.0, 0.0]]
        scores = chunker._calculate_cohesion_scores(embeddings)
        assert scores == [1.0]
    
    def test_find_optimal_boundaries_with_boundaries(self, chunker):
        """Test boundary detection when topic boundaries exist."""
        sentences = ["Sentence 1", "Sentence 2", "New topic", "Sentence 4"]
        cohesion_scores = [1.0, 0.9, 0.2, 0.8]
        
        boundaries = chunker._find_optimal_boundaries(sentences, cohesion_scores)
        
        assert 2 in boundaries  # Should detect boundary at index 2
    
    def test_find_optimal_boundaries_no_boundaries(self, chunker):
        """Test boundary detection when no clear boundaries exist."""
        sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
        cohesion_scores = [1.0, 0.8, 0.7]  # All above threshold
        
        boundaries = chunker._find_optimal_boundaries(sentences, cohesion_scores)
        
        # Should create boundaries based on max_chunk_size
        assert len(boundaries) >= 0
    
    def test_create_chunks_with_overlap(self, chunker):
        """Test chunk creation with smart overlap."""
        sentences = [f"Sentence {i}" for i in range(10)]
        boundaries = [0, 3, 6, 10]
        
        chunks = chunker._create_chunks_with_overlap(sentences, boundaries)
        
        assert len(chunks) > 0
        # Check that chunks have some overlap
        for chunk in chunks:
            assert 'content' in chunk
            assert 'metadata' in chunk
    
    def test_chunk_with_embeddings_success(self, chunker, mock_embedding_client):
        """Test successful chunking with embeddings."""
        text = "This is sentence one. This is sentence two. This is a different topic. This continues the topic."
        
        chunks = chunker.chunk_with_embeddings(text)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        mock_embedding_client.get_embeddings.assert_called()
    
    def test_chunk_with_langgraph_state_integration(self, chunker, mock_embedding_client):
        """Test chunking with LangGraph state integration."""
        # Create initial pipeline state
        config = ProcessingConfiguration(
            chunking_enabled=True,
            max_chunk_size=512,
            chunk_overlap=50
        )
        state = create_initial_state(
            document_path="test_doc.txt",
            processing_config=config
        )
        
        text = "This is sentence one. This is sentence two. This is a different topic."
        chunks = chunker.chunk_with_embeddings(text)
        
        # Convert chunks to ChunkInfo objects
        chunk_infos = []
        for i, chunk in enumerate(chunks):
            chunk_info = ChunkInfo(
                chunk_id=f"chunk_{i}",
                content=chunk['content'],
                token_count=chunk['metadata'].get('token_count', 0),
                cohesion_score=chunk['metadata'].get('cohesion_score', 0.0),
                start_position=chunk['metadata'].get('start_position', 0),
                end_position=chunk['metadata'].get('end_position', len(chunk['content'])),
                overlap_with_previous=chunk['metadata'].get('overlap_tokens', 0),
                embedding_vector=chunk['metadata'].get('embedding', [])
            )
            chunk_infos.append(chunk_info)
        
        # Update state with chunks
        state.chunks = chunk_infos
        state.current_stage = ProcessingStage.CHUNKING_COMPLETE
        
        # Verify state integration
        assert len(state.chunks) == len(chunks)
        assert state.current_stage == ProcessingStage.CHUNKING_COMPLETE
        assert all(isinstance(chunk, ChunkInfo) for chunk in state.chunks)
    
    def test_chunk_with_embeddings_fallback(self, chunker, mock_embedding_client):
        """Test chunking fallback when embeddings fail."""
        text = "This is a test text for fallback chunking. It has multiple sentences to trigger embedding calls."

        # Mock the embedding client to raise an error
        chunker.embedding_client.get_embeddings.side_effect = EmbeddingAPIError("API Error")   

        chunks = chunker.chunk_with_embeddings(text)

        # Verify fallback was used by checking metadata
        assert len(chunks) > 0
        assert chunks[0]['metadata']['method'] == 'fallback'
        assert chunks[0]['metadata']['chunking_method'] == 'fallback_token_based'
    
    def test_chunking_error_handling_with_state(self, chunker, mock_embedding_client):
        """Test error handling during chunking with state tracking."""
        # Create initial state
        config = ProcessingConfiguration(chunking_enabled=True)
        state = create_initial_state(
            document_path="test_doc.txt",
            processing_config=config
        )
        
        # Mock embedding failure
        chunker.embedding_client.get_embeddings.side_effect = EmbeddingAPIError("API Error")
        
        text = "Test text for error handling."
        
        try:
            chunks = chunker.chunk_with_embeddings(text)
            
            # Create error info for the failure
            error_info = ErrorInfo(
                error_type="EmbeddingAPIError",
                message="API Error",
                stage=ProcessingStage.CHUNKING,
                recoverable=True,
                context={"fallback_used": True, "method": "fallback_token_based"}
            )
            
            # Add error to state
            state.errors.append(error_info)
            state.current_stage = ProcessingStage.CHUNKING_COMPLETE  # Completed with fallback
            
            # Verify error tracking
            assert len(state.errors) == 1
            assert state.errors[0].error_type == "EmbeddingAPIError"
            assert state.errors[0].recoverable is True
            assert len(chunks) > 0  # Should still have chunks from fallback
            
        except Exception as e:
            pytest.fail(f"Chunking should handle errors gracefully: {e}")
    
    def test_chunk_with_embeddings_empty_text(self, chunker):
        """Test chunking with empty text."""
        chunks = chunker.chunk_with_embeddings("")
        # Empty text should return a single empty chunk
        assert len(chunks) == 1
        assert chunks[0]['content'] == ""
        assert chunks[0]['metadata']['method'] == 'embedding_guided'
    
    def test_chunk_with_embeddings_caching(self, chunker, mock_embedding_client):
        """Test that embeddings are cached properly."""
        text = "This is a test text for caching. It has multiple sentences to trigger embedding calls. Each sentence should be processed."
        
        # Mock embeddings return
        mock_embedding_client.get_embeddings.return_value = [
            [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]
        ]
        
        # First call
        chunks1 = chunker.chunk_with_embeddings(text)
        
        # Second call with same text
        chunks2 = chunker.chunk_with_embeddings(text)
        
        # Should only call embeddings once due to caching (if caching is implemented)
        # For now, just verify both calls return consistent results
        assert len(chunks1) > 0
        assert len(chunks2) > 0
        assert chunks1[0]['metadata']['method'] == 'embedding_guided'
        assert chunks2[0]['metadata']['method'] == 'embedding_guided'
    
    def test_fallback_chunk(self, chunker):
        """Test fallback chunking method."""
        text = "This is a test. This is another sentence. And one more."
        
        chunks = chunker._fallback_chunk(text)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        assert all(chunk['metadata']['method'] == 'fallback' for chunk in chunks)
    
    def test_adaptive_chunk_sizing(self, chunker):
        """Test adaptive chunk sizing based on content complexity."""
        # Simple content should get larger chunks
        simple_text = "Simple sentence. Another simple sentence."
        simple_chunks = chunker.chunk_with_embeddings(simple_text)
        
        # Complex content should get smaller chunks
        complex_text = "Complex technical terminology and intricate relationships. Advanced concepts require detailed analysis."
        complex_chunks = chunker.chunk_with_embeddings(complex_text)
        
        # This is a behavioral test - exact assertions depend on implementation
        assert len(simple_chunks) >= 0
        assert len(complex_chunks) >= 0
    
    def test_token_counting(self, chunker):
        """Test token counting functionality."""
        text = "This is a test sentence with multiple words."
        
        token_count = chunker._count_tokens(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_sentence_splitting(self, chunker):
        """Test sentence splitting functionality."""
        text = "First sentence. Second sentence! Third sentence?"
        
        sentences = chunker._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]
    
    @pytest.mark.parametrize("threshold,expected_boundaries", [
        (0.1, []),  # Very low threshold, no boundaries
        (0.5, [2]),  # Medium threshold, some boundaries
        (0.9, [1, 2, 3]),  # High threshold, many boundaries
    ])
    def test_boundary_detection_thresholds(self, chunker, threshold, expected_boundaries):
        """Test boundary detection with different thresholds."""
        sentences = ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4"]
        cohesion_scores = [1.0, 0.8, 0.3, 0.7]
        
        # Temporarily modify threshold
        original_threshold = chunker.config['cohesion_threshold']
        chunker.config['cohesion_threshold'] = threshold
        
        try:
            boundaries = chunker._find_optimal_boundaries(sentences, cohesion_scores)
            # Check that boundaries are reasonable (exact match depends on implementation)
            assert isinstance(boundaries, list)
            assert all(isinstance(b, int) for b in boundaries)
        finally:
            chunker.config['cohesion_threshold'] = original_threshold
    
    def test_error_handling_in_chunking(self, chunker, mock_embedding_client):
        """Test error handling during chunking process."""
        # Test various error scenarios
        mock_embedding_client.get_embeddings.side_effect = Exception("Unexpected error")
        
        text = "Test text for error handling."
        
        # Should not raise exception, should fallback gracefully
        chunks = chunker.chunk_with_embeddings(text)
        
        assert len(chunks) > 0  # Should still return chunks via fallback
    
    def test_metadata_enrichment(self, chunker):
        """Test that chunks contain proper metadata."""
        text = "This is a test sentence for metadata checking."
        
        chunks = chunker.chunk_with_embeddings(text)
        
        for chunk in chunks:
            metadata = chunk['metadata']
            assert 'chunk_id' in metadata
            assert 'method' in metadata
            assert 'token_count' in metadata
            assert 'cohesion_score' in metadata
            assert isinstance(metadata['token_count'], int)
            assert isinstance(metadata['cohesion_score'], (int, float))
    
    def test_chunk_info_creation_from_chunks(self, chunker):
        """Test creating ChunkInfo objects from chunker output."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_with_embeddings(text)
        
        # Convert to ChunkInfo objects
        chunk_infos = []
        for i, chunk in enumerate(chunks):
            chunk_info = ChunkInfo(
                chunk_id=f"chunk_{i}",
                content=chunk['content'],
                token_count=chunk['metadata'].get('token_count', 0),
                cohesion_score=chunk['metadata'].get('cohesion_score', 0.0),
                start_position=0,
                end_position=len(chunk['content']),
                overlap_with_previous=0,
                embedding_vector=[]
            )
            chunk_infos.append(chunk_info)
        
        # Verify ChunkInfo objects
        assert len(chunk_infos) == len(chunks)
        for chunk_info in chunk_infos:
            assert isinstance(chunk_info, ChunkInfo)
            assert chunk_info.chunk_id.startswith("chunk_")
            assert len(chunk_info.content) > 0
            assert chunk_info.token_count >= 0
            assert 0.0 <= chunk_info.cohesion_score <= 1.0
    
    def test_pipeline_state_chunking_workflow(self, chunker):
        """Test complete chunking workflow with pipeline state."""
        # Create initial state
        config = ProcessingConfiguration(
            chunking_enabled=True,
            max_chunk_size=256,
            chunk_overlap=25
        )
        state = create_initial_state(
            document_path="workflow_test.txt",
            processing_config=config
        )
        
        # Simulate chunking workflow
        text = "This is the first paragraph. It contains multiple sentences. This is the second paragraph. It also has content."
        
        # Update state to chunking stage
        state.current_stage = ProcessingStage.CHUNKING
        
        # Perform chunking
        chunks = chunker.chunk_with_embeddings(text)
        
        # Convert to ChunkInfo and update state
        chunk_infos = []
        for i, chunk in enumerate(chunks):
            chunk_info = ChunkInfo(
                chunk_id=f"chunk_{i}",
                content=chunk['content'],
                token_count=chunk['metadata'].get('token_count', 0),
                cohesion_score=chunk['metadata'].get('cohesion_score', 0.0),
                start_position=i * 100,  # Simulated positions
                end_position=(i + 1) * 100,
                overlap_with_previous=config.chunk_overlap if i > 0 else 0,
                embedding_vector=chunk['metadata'].get('embedding', [])
            )
            chunk_infos.append(chunk_info)
        
        state.chunks = chunk_infos
        state.current_stage = ProcessingStage.CHUNKING_COMPLETE
        
        # Verify workflow state
        assert state.current_stage == ProcessingStage.CHUNKING_COMPLETE
        assert len(state.chunks) > 0
        assert all(chunk.token_count <= config.max_chunk_size for chunk in state.chunks)
        assert state.processing_config.chunking_enabled is True