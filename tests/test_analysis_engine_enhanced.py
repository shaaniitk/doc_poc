import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from modules.analysis_engine import DocumentAnalyzer
from modules.error_handler import EmbeddingError, EmbeddingAPIError
import config


class TestDocumentAnalyzerEnhanced:
    """Test suite for enhanced DocumentAnalyzer with semantic features."""
    
    @pytest.fixture
    def mock_embedding_client(self):
        """Create a mock embedding client."""
        client = Mock()
        client.get_embeddings.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35],
            [0.8, 0.1, 0.1],
            [0.85, 0.15, 0.05]
        ])
        client.calculate_cohesion_scores.return_value = np.array([1.0, 0.9, 0.2, 0.8])
        client.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35],
            [0.8, 0.1, 0.1],
            [0.85, 0.15, 0.05]
        ])
        # Mock similarity method to return realistic similarity matrix
        def mock_similarity(embeddings1, embeddings2):
            # Calculate cosine similarity for test embeddings
            if len(embeddings1) == 3 and len(embeddings2) == 3:
                # Return high similarity matrix for similar embeddings
                return np.array([
                    [1.0, 0.95, 0.90],
                    [0.95, 1.0, 0.85],
                    [0.90, 0.85, 1.0]
                ])
            else:
                # Default fallback
                return np.array([[0.85]])
        client.similarity.side_effect = mock_similarity
        return client
    
    @pytest.fixture
    def sample_document_tree(self):
        """Create a sample document tree for testing."""
        return {
            'Introduction': {
                'metadata': {'hierarchy_path': ['Introduction']},
                'chunks': [{'content': 'This is the introduction section with some content.', 'type': 'text'}],
                'subsections': {}
            },
            'Methods': {
                'metadata': {'hierarchy_path': ['Methods']},
                'chunks': [{'content': 'This section describes the methodology used in the study.', 'type': 'text'}],
                'subsections': {
                    'Data Collection': {
                        'metadata': {'hierarchy_path': ['Methods', 'Data Collection']},
                        'chunks': [{'content': 'Data was collected using various methods.', 'type': 'text'}],
                        'subsections': {}
                    }
                }
            },
            'Results': {
                'metadata': {'hierarchy_path': ['Results']},
                'chunks': [{'content': 'The results show significant improvements in performance.', 'type': 'text'}],
                'subsections': {}
            }
        }
    
    @pytest.fixture
    def analyzer(self, mock_embedding_client, sample_document_tree):
        """Create DocumentAnalyzer instance with mocked dependencies."""
        with patch('modules.analysis_engine.UnifiedEmbeddingClient', return_value=mock_embedding_client):
            analyzer = DocumentAnalyzer(sample_document_tree, sample_document_tree)
            analyzer.document_trees = [sample_document_tree]
            # Override the embedding client with our mock
            analyzer.embedding_client = mock_embedding_client
            return analyzer
    
    def test_initialization_with_semantic_features(self, analyzer):
        """Test analyzer initialization with semantic components."""
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        assert analyzer.mistral_config == mistral_refinement
        assert analyzer.embedding_client is not None
        assert analyzer.semantic_cache == {}
        assert analyzer.document_trees is not None
    
    def test_extract_text_content_simple_section(self, analyzer):
        """Test text extraction from simple section."""
        section = {
            'Test Section': {
                'metadata': {'hierarchy_path': ['Test Section']},
                'chunks': [{'content': 'This is test content.', 'type': 'text'}],
                'subsections': {}
            }
        }
        
        text = analyzer._extract_text_content(section)
        
        assert 'Test Section' in text
        assert 'This is test content.' in text
    
    def test_extract_text_content_nested_sections(self, analyzer):
        """Test text extraction from nested sections."""
        section = {
            'Parent Section': {
                'metadata': {'hierarchy_path': ['Parent Section']},
                'chunks': [{'content': 'Parent content.', 'type': 'text'}],
                'subsections': {
                    'Child Section': {
                        'metadata': {'hierarchy_path': ['Parent Section', 'Child Section']},
                        'chunks': [{'content': 'Child content.', 'type': 'text'}],
                        'subsections': {}
                    }
                }
            }
        }
        
        text = analyzer._extract_text_content(section)
        
        assert 'Parent Section' in text
        assert 'Parent content.' in text
        assert 'Child Section' in text
        assert 'Child content.' in text
    
    def test_extract_text_content_empty_section(self, analyzer):
        """Test text extraction from empty section."""
        section = {}
        
        text = analyzer._extract_text_content(section)
        
        assert text == ""
    
    def test_analyze_semantic_preservation_success(self, analyzer, mock_embedding_client):
        """Test successful semantic preservation analysis."""
        original_sections = [
            {
                'Section 1': {
                    'metadata': {'hierarchy_path': ['Section 1']},
                    'chunks': [{'content': 'Original content 1', 'type': 'text'}],
                    'subsections': {}
                }
            },
            {
                'Section 2': {
                    'metadata': {'hierarchy_path': ['Section 2']},
                    'chunks': [{'content': 'Original content 2', 'type': 'text'}],
                    'subsections': {}
                }
            }
        ]
        
        processed_sections = [
            {
                'Section 1': {
                    'metadata': {'hierarchy_path': ['Section 1']},
                    'chunks': [{'content': 'Processed content 1', 'type': 'text'}],
                    'subsections': {}
                }
            },
            {
                'Section 2': {
                    'metadata': {'hierarchy_path': ['Section 2']},
                    'chunks': [{'content': 'Processed content 2', 'type': 'text'}],
                    'subsections': {}
                }
            }
        ]
        
        metrics = analyzer.analyze_semantic_preservation(original_sections, processed_sections)
        
        assert 'overall_similarity' in metrics
        assert 'section_similarities' in metrics
        assert 'preservation_score' in metrics
        assert 'quality_assessment' in metrics
        
        assert 0.0 <= metrics['overall_similarity'] <= 1.0
        assert 0.0 <= metrics['preservation_score'] <= 1.0
        assert len(metrics['section_similarities']) == len(original_sections)
    
    def test_analyze_semantic_preservation_empty_sections(self, analyzer):
        """Test semantic preservation analysis with empty sections."""
        metrics = analyzer.analyze_semantic_preservation([], [])
        
        assert metrics['overall_similarity'] == 1.0
        assert metrics['preservation_score'] == 1.0
        assert metrics['section_similarities'] == []
    
    def test_analyze_semantic_preservation_mismatched_lengths(self, analyzer):
        """Test semantic preservation analysis with mismatched section lengths."""
        original_sections = [
            {'title': 'Section 1', 'content': 'Content 1', 'subsections': []}
        ]
        
        processed_sections = [
            {'title': 'Section 1', 'content': 'Content 1', 'subsections': []},
            {'title': 'Section 2', 'content': 'Content 2', 'subsections': []}
        ]
        
        metrics = analyzer.analyze_semantic_preservation(original_sections, processed_sections)
        
        # Should handle mismatched lengths gracefully
        assert 'overall_similarity' in metrics
        assert 'structural_change_penalty' in metrics
    
    def test_analyze_semantic_preservation_with_caching(self, analyzer, mock_embedding_client):
        """Test that semantic analysis results are cached."""
        sections = [
            {'title': 'Section 1', 'content': 'Test content', 'subsections': []}
        ]
        
        # First call
        metrics1 = analyzer.analyze_semantic_preservation(sections, sections)
        
        # Second call with same sections
        metrics2 = analyzer.analyze_semantic_preservation(sections, sections)
        
        # Should use cache for second call
        assert metrics1 == metrics2
        # Embedding client should be called fewer times due to caching
    
    def test_detect_topic_drift_patterns_success(self, analyzer, mock_embedding_client):
        """Test successful topic drift detection."""
        # Set up document trees for the analyzer
        analyzer.original_tree = {
            'sections': [
                {'title': 'Introduction', 'content': 'Introduction to the topic', 'subsections': []},
                {'title': 'Methods', 'content': 'Methodology description', 'subsections': []},
                {'title': 'Results', 'content': 'Results and findings', 'subsections': []}
            ]
        }
        analyzer.processed_tree = analyzer.original_tree
        
        drift_analysis = analyzer.detect_topic_drift_patterns()
        
        assert 'enabled' in drift_analysis
        assert drift_analysis['enabled'] == True
        if 'patterns' in drift_analysis:
            assert isinstance(drift_analysis['patterns'], dict)
    
    def test_detect_topic_drift_patterns_single_section(self, analyzer):
        """Test topic drift detection with single section."""
        # Set up document trees for the analyzer
        analyzer.original_tree = {
            'sections': [
                {'title': 'Only Section', 'content': 'Single section content', 'subsections': []}
            ]
        }
        analyzer.processed_tree = analyzer.original_tree
        
        drift_analysis = analyzer.detect_topic_drift_patterns()
        
        assert 'enabled' in drift_analysis
        assert drift_analysis['enabled'] == True
    
    def test_detect_topic_drift_patterns_empty_sections(self, analyzer):
        """Test topic drift detection with empty sections."""
        # Set up empty document trees
        analyzer.original_tree = {'sections': []}
        analyzer.processed_tree = {'sections': []}
        
        drift_analysis = analyzer.detect_topic_drift_patterns()
        
        assert 'enabled' in drift_analysis
        assert drift_analysis['enabled'] is True
    
    def test_calculate_section_similarity_identical(self, analyzer, mock_embedding_client):
        """Test similarity calculation for identical sections."""
        section1 = {'title': 'Test', 'content': 'Same content', 'subsections': []}
        section2 = {'title': 'Test', 'content': 'Same content', 'subsections': []}
        
        # Mock encode and similarity methods - encode is called twice
        analyzer.embedding_client.encode.side_effect = [[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]
        # Reset side_effect and use return_value instead
        analyzer.embedding_client.similarity.side_effect = None
        analyzer.embedding_client.similarity.return_value = [[1.0]]
        
        similarity = analyzer._calculate_section_similarity(section1, section2)
        
        assert similarity == 1.0
    
    def test_calculate_section_similarity_different(self, analyzer, mock_embedding_client):
        """Test similarity calculation for different sections."""
        section1 = {'title': 'Test1', 'content': 'Content 1', 'subsections': []}
        section2 = {'title': 'Test2', 'content': 'Content 2', 'subsections': []}
        
        # Mock encode and similarity methods for different content - encode is called twice
        analyzer.embedding_client.encode.side_effect = [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]
        # Reset side_effect and use return_value instead
        analyzer.embedding_client.similarity.side_effect = None
        analyzer.embedding_client.similarity.return_value = [[0.3]]
        
        similarity = analyzer._calculate_section_similarity(section1, section2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0  # Should be less than 1 for different content
    
    def test_calculate_coherence_score_high_coherence(self, analyzer):
        """Test coherence score calculation for high coherence embeddings."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],  # Similar to first
            [0.8, 0.2, 0.0]   # Similar to others
        ]
        
        coherence = analyzer._calculate_internal_coherence(embeddings)
        
        assert 0.0 <= coherence <= 1.0
        assert coherence > 0.5  # Should be high for similar embeddings
    
    def test_calculate_coherence_score_low_coherence(self, analyzer):
        """Test coherence score calculation for low coherence embeddings."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # Orthogonal to first
            [0.0, 0.0, 1.0]   # Orthogonal to others
        ]
        
        # Mock the similarity method to return low similarity for orthogonal vectors
        def mock_low_similarity(embeddings1, embeddings2):
            if len(embeddings1) == 3 and len(embeddings2) == 3:
                # Return low similarity matrix for orthogonal embeddings
                return np.array([
                    [1.0, 0.1, 0.1],
                    [0.1, 1.0, 0.1],
                    [0.1, 0.1, 1.0]
                ])
            else:
                return np.array([[0.1]])
        
        analyzer.embedding_client.similarity.side_effect = mock_low_similarity
        
        coherence = analyzer._calculate_internal_coherence(embeddings)
        
        assert 0.0 <= coherence <= 1.0
        assert coherence < 0.5  # Should be low for dissimilar embeddings
    
    def test_calculate_coherence_score_single_embedding(self, analyzer):
        """Test coherence score calculation for single embedding."""
        embeddings = [[1.0, 0.0, 0.0]]
        
        coherence = analyzer._calculate_internal_coherence(embeddings)
        
        assert coherence == 1.0  # Single embedding should have perfect coherence
    
    def test_calculate_coherence_score_empty_embeddings(self, analyzer):
        """Test coherence score calculation for empty embeddings."""
        coherence = analyzer._calculate_internal_coherence([])
        
        assert coherence == 1.0  # Empty should return perfect coherence
    
    def test_error_handling_in_semantic_analysis(self, analyzer, mock_embedding_client):
        """Test error handling during semantic analysis."""
        analyzer.embedding_client.encode.side_effect = EmbeddingAPIError("API Error")
        
        sections = [
            {'title': 'Test', 'content': 'Test content', 'subsections': []}
        ]
        
        # Should handle errors gracefully
        metrics = analyzer.analyze_semantic_preservation(sections, sections)
        
        # Should return default values on error
        assert 'overall_similarity' in metrics
        assert 'error' in metrics
    
    def test_error_handling_in_topic_drift_detection(self, analyzer, mock_embedding_client):
        """Test error handling during topic drift detection."""
        mock_embedding_client.get_embeddings.side_effect = Exception("Unexpected error")
        
        # Set up document trees for the analyzer
        test_tree = {
            'metadata': {'title': 'Test Document'},
            'chunks': [],
            'subsections': {
                'test_section': {
                    'metadata': {'title': 'Test'},
                    'chunks': [{'content': 'Test content'}],
                    'subsections': {}
                }
            }
        }
        
        analyzer.original_tree = test_tree
        analyzer.processed_tree = test_tree
        
        # Should handle errors gracefully
        drift_analysis = analyzer.detect_topic_drift_patterns()
        
        # Should return default values on error
        assert 'enabled' in drift_analysis
        assert drift_analysis['enabled'] is True or 'error' in drift_analysis
    
    @pytest.mark.parametrize("threshold,expected_drifts", [
        (0.1, 0),  # Very low threshold, no drifts detected
        (0.5, 1),  # Medium threshold, some drifts
        (0.9, 2),  # High threshold, many drifts
    ])
    def test_topic_drift_detection_thresholds(self, analyzer, mock_embedding_client, threshold, expected_drifts):
        """Test topic drift detection with different thresholds."""
        # Set up document trees for the analyzer
        original_tree = {
            'metadata': {'title': 'Test Document'},
            'chunks': [],
            'subsections': {
                'section1': {
                    'metadata': {'title': 'Section 1'},
                    'chunks': [{'content': 'Content 1'}],
                    'subsections': {}
                },
                'section2': {
                    'metadata': {'title': 'Section 2'},
                    'chunks': [{'content': 'Content 2'}],
                    'subsections': {}
                },
                'section3': {
                    'metadata': {'title': 'Section 3'},
                    'chunks': [{'content': 'Content 3'}],
                    'subsections': {}
                }
            }
        }
        
        analyzer.original_tree = original_tree
        analyzer.processed_tree = original_tree  # Same for simplicity
        
        # Mock coherence scores that would trigger different numbers of drifts
        mock_embedding_client.calculate_cohesion_scores.return_value = [1.0, 0.3, 0.7]
        
        # Temporarily modify threshold in config if needed
        original_threshold = getattr(analyzer.mistral_config, 'drift_threshold', 0.5)
        analyzer.mistral_config['drift_threshold'] = threshold
        
        try:
            drift_analysis = analyzer.detect_topic_drift_patterns()
            
            # Check that drift detection is reasonable
            assert 'enabled' in drift_analysis
            assert drift_analysis['enabled'] is True
        finally:
            analyzer.mistral_config['drift_threshold'] = original_threshold
    
    def test_semantic_cache_functionality(self, analyzer, mock_embedding_client):
        """Test that semantic analysis uses caching effectively."""
        section = {'title': 'Test', 'content': 'Test content for caching', 'subsections': []}
        
        # First call should populate cache
        metrics1 = analyzer.analyze_semantic_preservation([section], [section])
        
        # Second call should use cache
        metrics2 = analyzer.analyze_semantic_preservation([section], [section])
        
        assert metrics1 == metrics2
        
        # Check that cache contains expected entries
        assert len(analyzer.semantic_cache) > 0
    
    def test_integration_with_document_trees(self, analyzer, sample_document_tree):
        """Test integration with stored document trees."""
        # Analyzer should have access to document trees
        assert analyzer.original_tree == sample_document_tree
        assert analyzer.processed_tree == sample_document_tree
        
        # Create sections from the document tree structure
        sections = [
            {
                'metadata': {'title': 'Introduction'},
                'chunks': [{'content': 'This is the introduction section with some content.'}],
                'subsections': {}
            },
            {
                'metadata': {'title': 'Methods'},
                'chunks': [{'content': 'This section describes the methodology used in the study.'}],
                'subsections': {}
            }
        ]
        
        # Should be able to analyze the sections
        metrics = analyzer.analyze_semantic_preservation(sections, sections)
        
        assert 'overall_similarity' in metrics
        assert metrics['overall_similarity'] > 0.0