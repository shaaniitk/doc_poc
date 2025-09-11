import pytest
from unittest.mock import patch, Mock
from config import CHUNKING_EMBEDDING, MISTRAL_REFINEMENT


class TestChunkingEmbeddingConfig:
    """Test suite for CHUNKING_EMBEDDING configuration."""
    
    def test_chunking_embedding_config_exists(self):
        """Test that CHUNKING_EMBEDDING configuration is properly defined."""
        assert CHUNKING_EMBEDDING is not None
        assert isinstance(CHUNKING_EMBEDDING, dict)
    
    def test_chunking_embedding_enabled_flag(self):
        """Test that enabled flag exists and is boolean."""
        assert 'enable' in CHUNKING_EMBEDDING
        assert isinstance(CHUNKING_EMBEDDING['enable'], bool)
        # Should be True by default based on config
        assert CHUNKING_EMBEDDING['enable'] is True
    
    def test_chunking_embedding_required_fields(self):
        """Test that all required configuration fields are present."""
        required_fields = [
            'enable',
            'cohesion_threshold',
            'max_tokens_per_chunk',
            'overlap_tokens',
            'fallback_provider',
            'boundary_detection_method',
            'adaptive_sizing',
            'smart_overlap'
        ]
        
        for field in required_fields:
            assert field in CHUNKING_EMBEDDING, f"Missing required field: {field}"
    
    def test_chunking_embedding_field_types(self):
        """Test that configuration fields have correct types."""
        assert isinstance(CHUNKING_EMBEDDING['cohesion_threshold'], (int, float))
        assert isinstance(CHUNKING_EMBEDDING['max_tokens_per_chunk'], int)
        assert isinstance(CHUNKING_EMBEDDING['overlap_tokens'], int)
        assert isinstance(CHUNKING_EMBEDDING['fallback_provider'], str)
        assert isinstance(CHUNKING_EMBEDDING['boundary_detection_method'], str)
        assert isinstance(CHUNKING_EMBEDDING['adaptive_sizing'], bool)
        assert isinstance(CHUNKING_EMBEDDING['smart_overlap'], bool)
    
    def test_chunking_embedding_value_ranges(self):
        """Test that configuration values are within reasonable ranges."""
        # Cohesion threshold should be between 0 and 1
        assert 0.0 <= CHUNKING_EMBEDDING['cohesion_threshold'] <= 1.0
        
        # Chunk sizes should be positive
        assert CHUNKING_EMBEDDING['max_tokens_per_chunk'] > 0
        
        # Overlap should be non-negative
        assert CHUNKING_EMBEDDING['overlap_tokens'] >= 0
    
    def test_chunking_embedding_fallback_specification(self):
        """Test that fallback provider is properly specified."""
        provider = CHUNKING_EMBEDDING['fallback_provider']
        assert len(provider) > 0
        # Should be a valid provider identifier
        assert isinstance(provider, str)
    
    @pytest.mark.parametrize("field,expected_type", [
        ('cohesion_threshold', (int, float)),
        ('max_tokens_per_chunk', int),
        ('overlap_tokens', int),
        ('fallback_provider', str),
        ('boundary_detection_method', str),
        ('adaptive_sizing', bool),
        ('smart_overlap', bool),
        ('enable', bool)
    ])
    def test_chunking_embedding_field_type_validation(self, field, expected_type):
        """Parametrized test for field type validation."""
        assert isinstance(CHUNKING_EMBEDDING[field], expected_type)


class TestMistralRefinementConfig:
    """Test suite for MISTRAL_REFINEMENT configuration."""
    
    def test_mistral_refinement_config_exists(self):
        """Test that MISTRAL_REFINEMENT configuration is properly defined."""
        assert MISTRAL_REFINEMENT is not None
        assert isinstance(MISTRAL_REFINEMENT, dict)
    
    def test_mistral_refinement_enabled_flag(self):
        """Test that enabled flag exists and is boolean."""
        assert 'enable' in MISTRAL_REFINEMENT
        assert isinstance(MISTRAL_REFINEMENT['enable'], bool)
        # Should be True by default based on config
        assert MISTRAL_REFINEMENT['enable'] is True
    
    def test_mistral_refinement_required_fields(self):
        """Test that all required configuration fields are present."""
        required_fields = [
            'enable',
            'only_for_low_confidence',
            'confidence_threshold',
            'max_cases_per_doc',
            'refinement_model',
            'temperature',
            'max_tokens'
        ]
        
        for field in required_fields:
            assert field in MISTRAL_REFINEMENT, f"Missing required field: {field}"
    
    def test_mistral_refinement_field_types(self):
        """Test that configuration fields have correct types."""
        assert isinstance(MISTRAL_REFINEMENT['only_for_low_confidence'], bool)
        assert isinstance(MISTRAL_REFINEMENT['confidence_threshold'], (int, float))
        assert isinstance(MISTRAL_REFINEMENT['max_cases_per_doc'], int)
        assert isinstance(MISTRAL_REFINEMENT['refinement_model'], str)
        assert isinstance(MISTRAL_REFINEMENT['temperature'], (int, float))
        assert isinstance(MISTRAL_REFINEMENT['max_tokens'], int)
    
    def test_mistral_refinement_value_ranges(self):
        """Test that configuration values are within reasonable ranges."""
        # Confidence threshold should be between 0 and 1
        assert 0.0 <= MISTRAL_REFINEMENT['confidence_threshold'] <= 1.0
        
        # Temperature should be reasonable for LLM
        assert 0.0 <= MISTRAL_REFINEMENT['temperature'] <= 2.0
        
        # Max cases should be positive
        assert MISTRAL_REFINEMENT['max_cases_per_doc'] > 0
        
        # Max tokens should be positive
        assert MISTRAL_REFINEMENT['max_tokens'] > 0
    
    @pytest.mark.parametrize("field,expected_type", [
        ('only_for_low_confidence', bool),
        ('confidence_threshold', (int, float)),
        ('max_cases_per_doc', int),
        ('refinement_model', str),
        ('temperature', (int, float)),
        ('max_tokens', int),
        ('enable', bool)
    ])
    def test_mistral_refinement_field_type_validation(self, field, expected_type):
        """Parametrized test for field type validation."""
        assert isinstance(MISTRAL_REFINEMENT[field], expected_type)


class TestConfigurationIntegration:
    """Test suite for configuration integration scenarios."""
    
    def test_both_configs_enabled_by_default(self):
        """Test that both new configurations are enabled by default."""
        assert CHUNKING_EMBEDDING['enable'] is True
        assert MISTRAL_REFINEMENT['enable'] is True
    
    def test_config_compatibility(self):
        """Test that configurations are compatible with each other."""
        # Both configs should be able to be enabled simultaneously
        # This tests that there are no conflicting field names or values
        
        chunking_fields = set(CHUNKING_EMBEDDING.keys())
        mistral_fields = set(MISTRAL_REFINEMENT.keys())
        
        # Should not have conflicting field names (except 'enable' which is expected)
        common_fields = chunking_fields.intersection(mistral_fields)
        expected_common = {'enable'}  # Only 'enable' should be common
        
        assert common_fields == expected_common, f"Unexpected common fields: {common_fields - expected_common}"
    
    def test_config_import_success(self):
        """Test that configurations can be imported successfully."""
        try:
            from config import CHUNKING_EMBEDDING, MISTRAL_REFINEMENT
            assert True  # Import successful
        except ImportError as e:
            pytest.fail(f"Failed to import configurations: {e}")
    
    def test_config_modification_safety(self):
        """Test that configurations can be safely modified at runtime."""
        # Store original values
        original_chunking_enabled = CHUNKING_EMBEDDING['enable']
        original_mistral_enabled = MISTRAL_REFINEMENT['enable']
        
        try:
            # Modify configurations
            CHUNKING_EMBEDDING['enable'] = False
            MISTRAL_REFINEMENT['enable'] = False
            
            # Verify modifications
            assert CHUNKING_EMBEDDING['enable'] is False
            assert MISTRAL_REFINEMENT['enable'] is False
            
        finally:
            # Restore original values
            CHUNKING_EMBEDDING['enable'] = original_chunking_enabled
            MISTRAL_REFINEMENT['enable'] = original_mistral_enabled
    
    def test_config_validation_with_modules(self):
        """Test that configurations work with the modules that use them."""
        # Test that chunker can import and use CHUNKING_EMBEDDING
        try:
            from modules.chunker import EmbeddingGuidedChunker
            # Should not raise import error
            assert True
        except ImportError as e:
            pytest.fail(f"EmbeddingGuidedChunker cannot import CHUNKING_EMBEDDING: {e}")
        
        # Test that analysis_engine can import and use MISTRAL_REFINEMENT
        try:
            from modules.analysis_engine import DocumentAnalyzer
            # Should not raise import error
            assert True
        except ImportError as e:
            pytest.fail(f"DocumentAnalyzer cannot import MISTRAL_REFINEMENT: {e}")


class TestConfigurationUsageScenarios:
    """Test suite for realistic configuration usage scenarios."""
    
    def test_enable_chunking_embedding_scenario(self):
        """Test scenario where chunking embedding is enabled."""
        # Store original value
        original_enabled = CHUNKING_EMBEDDING['enable']
        
        try:
            # Enable chunking embedding
            CHUNKING_EMBEDDING['enable'] = True
            
            # Test that dependent modules can handle enabled state
            with patch('modules.chunker.UnifiedEmbeddingClient') as mock_client:
                from modules.chunker import EmbeddingGuidedChunker
                
                chunker = EmbeddingGuidedChunker()
                assert chunker.config.get('enable', False) is True
                
        finally:
            # Restore original value
            CHUNKING_EMBEDDING['enable'] = original_enabled
    
    def test_enable_mistral_refinement_scenario(self):
        """Test scenario where Mistral refinement is enabled."""
        # Store original value
        original_enabled = MISTRAL_REFINEMENT['enable']
        
        try:
            # Enable Mistral refinement
            MISTRAL_REFINEMENT['enable'] = True
            
            # Test that dependent modules can handle enabled state
            with patch('modules.analysis_engine.UnifiedEmbeddingClient') as mock_client:
                from modules.analysis_engine import DocumentAnalyzer
                
                analyzer = DocumentAnalyzer({}, {})  # Need original and processed trees
                assert analyzer.semantic_config.get('enable', False) is True
                
        finally:
            # Restore original value
            MISTRAL_REFINEMENT['enable'] = original_enabled
    
    def test_both_features_enabled_scenario(self):
        """Test scenario where both features are enabled simultaneously."""
        # Store original values
        original_chunking = CHUNKING_EMBEDDING['enable']
        original_mistral = MISTRAL_REFINEMENT['enable']
        
        try:
            # Enable both features
            CHUNKING_EMBEDDING['enable'] = True
            MISTRAL_REFINEMENT['enable'] = True
            
            # Test that both can work together
            with patch('modules.chunker.UnifiedEmbeddingClient'), \
                 patch('modules.analysis_engine.UnifiedEmbeddingClient'):
                
                from modules.chunker import EmbeddingGuidedChunker
                from modules.analysis_engine import DocumentAnalyzer
                
                chunker = EmbeddingGuidedChunker()
                analyzer = DocumentAnalyzer({}, {})  # Need original and processed trees
                
                assert chunker.config.get('enable', False) is True
                assert analyzer.semantic_config.get('enable', False) is True
                
        finally:
            # Restore original values
            CHUNKING_EMBEDDING['enable'] = original_chunking
            MISTRAL_REFINEMENT['enable'] = original_mistral
    
    def test_configuration_parameter_adjustment(self):
        """Test adjusting configuration parameters for different use cases."""
        # Store original values
        original_threshold = CHUNKING_EMBEDDING['cohesion_threshold']
        original_chunk_size = CHUNKING_EMBEDDING['max_tokens_per_chunk']
        
        try:
            # Adjust for high-precision chunking
            CHUNKING_EMBEDDING['cohesion_threshold'] = 0.8  # Higher threshold
            CHUNKING_EMBEDDING['max_tokens_per_chunk'] = 200  # Smaller chunks
            
            # Verify adjustments
            assert CHUNKING_EMBEDDING['cohesion_threshold'] == 0.8
            assert CHUNKING_EMBEDDING['max_tokens_per_chunk'] == 200
            
            # Test that modules can handle adjusted parameters
            with patch('modules.chunker.UnifiedEmbeddingClient'):
                from modules.chunker import EmbeddingGuidedChunker
                
                chunker = EmbeddingGuidedChunker()
                assert chunker.config.get('cohesion_threshold') == 0.8
                assert chunker.config.get('max_tokens_per_chunk') == 200
                
        finally:
            # Restore original values
            CHUNKING_EMBEDDING['cohesion_threshold'] = original_threshold
            CHUNKING_EMBEDDING['max_tokens_per_chunk'] = original_chunk_size
    
    @pytest.mark.parametrize("chunking_enabled,mistral_enabled", [
        (True, False),
        (False, True),
        (True, True),
        (False, False)
    ])
    def test_various_enable_combinations(self, chunking_enabled, mistral_enabled):
        """Test various combinations of feature enablement."""
        # Store original values
        original_chunking = CHUNKING_EMBEDDING['enable']
        original_mistral = MISTRAL_REFINEMENT['enable']
        
        try:
            # Set test configuration
            CHUNKING_EMBEDDING['enable'] = chunking_enabled
            MISTRAL_REFINEMENT['enable'] = mistral_enabled
            
            # Verify configuration is set correctly
            assert CHUNKING_EMBEDDING['enable'] == chunking_enabled
            assert MISTRAL_REFINEMENT['enable'] == mistral_enabled
            
            # Test that modules can handle this configuration
            with patch('modules.chunker.UnifiedEmbeddingClient'), \
                 patch('modules.analysis_engine.UnifiedEmbeddingClient'):
                
                # Should not raise errors regardless of configuration
                from modules.chunker import EmbeddingGuidedChunker
                from modules.analysis_engine import DocumentAnalyzer
                
                if chunking_enabled:
                    chunker = EmbeddingGuidedChunker()
                    assert chunker.config.get('enable', False) == chunking_enabled
                
                if mistral_enabled:
                    analyzer = DocumentAnalyzer({}, {})  # Need original and processed trees
                    assert analyzer.semantic_config.get('enable', False) == mistral_enabled
                
        finally:
            # Restore original values
            CHUNKING_EMBEDDING['enable'] = original_chunking
            MISTRAL_REFINEMENT['enable'] = original_mistral