import pytest
from unittest.mock import patch, Mock
import config


class TestChunkingEmbeddingConfig:
    """Test suite for CHUNKING_EMBEDDING configuration."""
    
    def test_chunking_embedding_config_exists(self):
        """Test that CHUNKING_EMBEDDING configuration is properly defined."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        assert chunking_embedding is not None
        assert isinstance(chunking_embedding, dict)
    
    def test_chunking_embedding_enabled_flag(self):
        """Test that enabled flag exists and is boolean."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        assert 'enable' in chunking_embedding
        assert isinstance(chunking_embedding['enable'], bool)
        # Should be True by default based on config
        assert chunking_embedding['enable'] is True
    
    def test_chunking_embedding_required_fields(self):
        """Test that all required configuration fields are present."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
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
            assert field in chunking_embedding, f"Missing required field: {field}"
    
    def test_chunking_embedding_field_types(self):
        """Test that configuration fields have correct types."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        assert isinstance(chunking_embedding['cohesion_threshold'], (int, float))
        assert isinstance(chunking_embedding['max_tokens_per_chunk'], int)
        assert isinstance(chunking_embedding['overlap_tokens'], int)
        assert isinstance(chunking_embedding['fallback_provider'], str)
        assert isinstance(chunking_embedding['boundary_detection_method'], str)
        assert isinstance(chunking_embedding['adaptive_sizing'], bool)
        assert isinstance(chunking_embedding['smart_overlap'], bool)
    
    def test_chunking_embedding_value_ranges(self):
        """Test that configuration values are within reasonable ranges."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        # Cohesion threshold should be between 0 and 1
        assert 0.0 <= chunking_embedding['cohesion_threshold'] <= 1.0
        
        # Chunk sizes should be positive
        assert chunking_embedding['max_tokens_per_chunk'] > 0
        
        # Overlap should be non-negative
        assert chunking_embedding['overlap_tokens'] >= 0
    
    def test_chunking_embedding_fallback_specification(self):
        """Test that fallback provider is properly specified."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        provider = chunking_embedding['fallback_provider']
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
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        assert isinstance(chunking_embedding[field], expected_type)


class TestMistralRefinementConfig:
    """Test suite for MISTRAL_REFINEMENT configuration."""
    
    def test_mistral_refinement_config_exists(self):
        """Test that MISTRAL_REFINEMENT configuration is properly defined."""
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        assert mistral_refinement is not None
        assert isinstance(mistral_refinement, dict)
    
    def test_mistral_refinement_enabled_flag(self):
        """Test that enabled flag exists and is boolean."""
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        assert 'enable' in mistral_refinement
        assert isinstance(mistral_refinement['enable'], bool)
        # Should be True by default based on config
        assert mistral_refinement['enable'] is True
    
    def test_mistral_refinement_required_fields(self):
        """Test that all required configuration fields are present."""
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
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
            assert field in mistral_refinement, f"Missing required field: {field}"
    
    def test_mistral_refinement_field_types(self):
        """Test that configuration fields have correct types."""
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        assert isinstance(mistral_refinement['only_for_low_confidence'], bool)
        assert isinstance(mistral_refinement['confidence_threshold'], (int, float))
        assert isinstance(mistral_refinement['max_cases_per_doc'], int)
        assert isinstance(mistral_refinement['refinement_model'], str)
        assert isinstance(mistral_refinement['temperature'], (int, float))
        assert isinstance(mistral_refinement['max_tokens'], int)
    
    def test_mistral_refinement_value_ranges(self):
        """Test that configuration values are within reasonable ranges."""
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        # Confidence threshold should be between 0 and 1
        assert 0.0 <= mistral_refinement['confidence_threshold'] <= 1.0
        
        # Temperature should be reasonable for LLM
        assert 0.0 <= mistral_refinement['temperature'] <= 2.0
        
        # Max cases should be positive
        assert mistral_refinement['max_cases_per_doc'] > 0
        
        # Max tokens should be positive
        assert mistral_refinement['max_tokens'] > 0
    
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
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        assert isinstance(mistral_refinement[field], expected_type)


class TestConfigurationIntegration:
    """Test suite for configuration integration scenarios."""
    
    def test_both_configs_enabled_by_default(self):
        """Test that both new configurations are enabled by default."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        assert chunking_embedding['enable'] is True
        assert mistral_refinement['enable'] is True
    
    def test_config_compatibility(self):
        """Test that configurations are compatible with each other."""
        # Both configs should be able to be enabled simultaneously
        # This tests that there are no conflicting field names or values
        
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        chunking_fields = set(chunking_embedding.keys())
        mistral_fields = set(mistral_refinement.keys())
        
        # Should not have conflicting field names (except 'enable' and 'device' which are expected)
        common_fields = chunking_fields.intersection(mistral_fields)
        expected_common = {'enable', 'device'}  # Only 'enable' and 'device' should be common
        
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
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        # Store original values
        original_chunking_enabled = chunking_embedding['enable']
        original_mistral_enabled = mistral_refinement['enable']
        
        try:
            # Modify configurations
            chunking_embedding['enable'] = False
            mistral_refinement['enable'] = False
            
            # Verify modifications
            assert chunking_embedding['enable'] is False
            assert mistral_refinement['enable'] is False
            
        finally:
            # Restore original values
            chunking_embedding['enable'] = original_chunking_enabled
            mistral_refinement['enable'] = original_mistral_enabled
    
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
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        # Store original value
        original_enabled = chunking_embedding['enable']
        
        try:
            # Enable chunking embedding
            chunking_embedding['enable'] = True
            
            # Test that dependent modules can handle enabled state
            with patch('modules.chunker.UnifiedEmbeddingClient') as mock_client:
                from modules.chunker import EmbeddingGuidedChunker
                
                chunker = EmbeddingGuidedChunker()
                assert chunker.config.get('enable', False) is True
                
        finally:
            # Restore original value
            chunking_embedding['enable'] = original_enabled
    
    def test_enable_mistral_refinement_scenario(self):
        """Test scenario where Mistral refinement is enabled."""
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        # Store original value
        original_enabled = mistral_refinement['enable']
        
        try:
            # Enable Mistral refinement
            mistral_refinement['enable'] = True
            
            # Test that dependent modules can handle enabled state
            with patch('modules.analysis_engine.UnifiedEmbeddingClient') as mock_client:
                from modules.analysis_engine import DocumentAnalyzer
                
                analyzer = DocumentAnalyzer({}, {})  # Need original and processed trees
                assert analyzer.semantic_config.get('enable', False) is True
                
        finally:
            # Restore original value
            mistral_refinement['enable'] = original_enabled
    
    def test_both_features_enabled_scenario(self):
        """Test scenario where both features are enabled simultaneously."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        # Store original values
        original_chunking = chunking_embedding['enable']
        original_mistral = mistral_refinement['enable']
        
        try:
            # Enable both features
            chunking_embedding['enable'] = True
            mistral_refinement['enable'] = True
            
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
            chunking_embedding['enable'] = original_chunking
            mistral_refinement['enable'] = original_mistral
    
    def test_configuration_parameter_adjustment(self):
        """Test adjusting configuration parameters for different use cases."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        # Store original values
        original_threshold = chunking_embedding['cohesion_threshold']
        original_chunk_size = chunking_embedding['max_tokens_per_chunk']
        
        try:
            # Adjust for high-precision chunking
            chunking_embedding['cohesion_threshold'] = 0.8  # Higher threshold
            chunking_embedding['max_tokens_per_chunk'] = 200  # Smaller chunks
            
            # Verify adjustments
            assert chunking_embedding['cohesion_threshold'] == 0.8
            assert chunking_embedding['max_tokens_per_chunk'] == 200
            
            # Test that modules can handle adjusted parameters
            with patch('modules.chunker.UnifiedEmbeddingClient'):
                from modules.chunker import EmbeddingGuidedChunker
                
                chunker = EmbeddingGuidedChunker()
                assert chunker.config.get('cohesion_threshold') == 0.8
                assert chunker.config.get('max_tokens_per_chunk') == 200
                
        finally:
            # Restore original values
            chunking_embedding['cohesion_threshold'] = original_threshold
            chunking_embedding['max_tokens_per_chunk'] = original_chunk_size
    
    @pytest.mark.parametrize("chunking_enabled,mistral_enabled", [
        (True, False),
        (False, True),
        (True, True),
        (False, False)
    ])
    def test_various_enable_combinations(self, chunking_enabled, mistral_enabled):
        """Test various combinations of feature enablement."""
        chunking_embedding = getattr(config, 'CHUNKING_EMBEDDING', {})
        mistral_refinement = getattr(config, 'MISTRAL_REFINEMENT', {})
        # Store original values
        original_chunking = chunking_embedding['enable']
        original_mistral = mistral_refinement['enable']
        
        try:
            # Set test configuration
            chunking_embedding['enable'] = chunking_enabled
            mistral_refinement['enable'] = mistral_enabled
            
            # Verify configuration is set correctly
            assert chunking_embedding['enable'] == chunking_enabled
            assert mistral_refinement['enable'] == mistral_enabled
            
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
            chunking_embedding['enable'] = original_chunking
            mistral_refinement['enable'] = original_mistral