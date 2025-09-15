"""Comprehensive unit tests for core components."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json
from typing import Dict, Any, List

# Import core components
from core.state_manager import CentralizedStateManager, ProcessingStage
from core.chunking_processor import ChunkingProcessor, ChunkingConfig, ChunkingStrategy
from core.llm_handler import LLMHandler, LLMConfig, ProcessingRequest, ProcessingResult
from core.knowledge_graph_processor import KnowledgeGraphProcessor, ExtractionConfig
from core.output_generator import OutputGenerator, OutputFormat, OutputTemplate
from core.error_handler import ErrorHandler, ErrorCategory, RecoveryStrategy
from core.monitoring import MetricsCollector, SystemMonitor, PerformanceTracker
from core.config import ConfigManager, ProcessingConfig


class TestStateManager:
    """Test cases for CentralizedStateManager."""
    
    @pytest.fixture
    def state_manager(self):
        return CentralizedStateManager()
    
    def test_initialization(self, state_manager):
        """Test state manager initialization."""
        assert state_manager.current_stage == ProcessingStage.INITIALIZATION
        assert state_manager.progress == 0.0
        assert len(state_manager.errors) == 0
        assert state_manager.chunk_state.to_dict() == {}
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, state_manager):
        """Test valid state transitions."""
        # Test valid transition using stage_context
        async with state_manager.stage_context(ProcessingStage.DOCUMENT_PARSING):
            assert state_manager.current_stage == ProcessingStage.DOCUMENT_PARSING
        
        # Test that stage is properly managed
        assert state_manager.current_stage == ProcessingStage.DOCUMENT_PARSING
    
    def test_chunk_management(self, state_manager):
        """Test chunk addition and retrieval."""
        chunk_data = {
            'id': 'chunk_1',
            'content': 'Test content',
            'metadata': {'page': 1}
        }
        
        state_manager.chunk_state['chunk_1'] = chunk_data
        assert len(state_manager.chunk_state.to_dict()) == 1
        assert state_manager.chunk_state.get('chunk_1') == chunk_data
    
    def test_error_tracking(self, state_manager):
        """Test error tracking functionality."""
        from core.state_manager import ErrorContext, ErrorSeverity
        
        error_context = ErrorContext(
            message="Test error",
            severity=ErrorSeverity.ERROR,
            stage=ProcessingStage.INITIALIZATION
        )
        state_manager.add_error(error_context)
        
        assert len(state_manager.errors) == 1
        assert state_manager.errors[0].message == "Test error"
        assert state_manager.errors[0].severity == ErrorSeverity.ERROR
    
    def test_serialization(self, state_manager):
        """Test state serialization."""
        state_manager.document_state['document_id'] = "test_doc"
        state_manager.chunk_state['chunk_1'] = {'id': 'chunk_1', 'content': 'test'}
        
        snapshot = state_manager.get_state_snapshot()
        assert snapshot['document_state']['document_id'] == "test_doc"
        assert len(snapshot['chunk_state']) == 1
        assert snapshot['session_id'] == state_manager.session_id


class TestChunkingProcessor:
    """Test cases for ChunkingProcessor."""
    
    @pytest.fixture
    def chunking_config(self):
        return ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            max_chunk_size=512,
            overlap_size=50,
            min_chunk_size=100
        )
    
    @pytest.fixture
    def chunking_processor(self, chunking_config):
        return ChunkingProcessor(chunking_config)
    
    def test_initialization(self, chunking_processor, chunking_config):
        """Test chunking processor initialization."""
        assert chunking_processor.config == chunking_config
        assert chunking_processor.strategy == ChunkingStrategy.SEMANTIC
    
    @pytest.mark.asyncio
    async def test_simple_chunking(self, chunking_processor):
        """Test basic text chunking."""
        text = "This is a test document. " * 100  # Create long text
        
        chunks = await chunking_processor.process_text(text)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_semantic_chunking(self, chunking_processor):
        """Test semantic-based chunking."""
        text = """
        Introduction to Machine Learning.
        Machine learning is a subset of artificial intelligence.
        
        Types of Machine Learning.
        There are three main types: supervised, unsupervised, and reinforcement learning.
        
        Applications of Machine Learning.
        Machine learning is used in many fields including healthcare and finance.
        """
        
        chunks = await chunking_processor.process_text(text)
        
        assert len(chunks) >= 2  # Should create multiple semantic chunks
        assert any('Introduction' in chunk['content'] for chunk in chunks)
        assert any('Applications' in chunk['content'] for chunk in chunks)
    
    def test_chunk_validation(self, chunking_processor):
        """Test chunk validation logic."""
        valid_chunk = {
            'content': 'Valid content with sufficient length',
            'metadata': {'source': 'test'}
        }
        
        invalid_chunk = {
            'content': 'Short',  # Too short
            'metadata': {}
        }
        
        assert chunking_processor._validate_chunk(valid_chunk)
        assert not chunking_processor._validate_chunk(invalid_chunk)


class TestLLMHandler:
    """Test cases for LLMHandler."""
    
    @pytest.fixture
    def llm_config(self):
        from core.llm_handler import LLMProvider
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-3.5-turbo",
            api_key="test_key",
            max_tokens=1000,
            temperature=0.7
        )
    
    @pytest.fixture
    def llm_handler(self, llm_config):
        return LLMHandler(llm_config)
    
    def test_initialization(self, llm_handler, llm_config):
        """Test LLM handler initialization."""
        assert llm_handler.config == llm_config
        assert llm_handler.config.provider.name == "OPENAI"
    
    @pytest.mark.asyncio
    async def test_process_chunk_mock(self, llm_handler):
        """Test chunk processing with mocked LLM."""
        from core.chunking_processor import DocumentChunk, ChunkMetadata
        from datetime import datetime, timezone
        
        # Create test chunk metadata
        metadata = ChunkMetadata(
            chunk_id="test_chunk",
            index=0,
            start_position=0,
            end_position=25,
            character_count=25,
            word_count=4,
            sentence_count=1,
            paragraph_count=1
        )
        
        # Create test document chunk
        chunk = DocumentChunk(
            content="Test content for processing",
            metadata=metadata,
            source_document="test.txt"
        )
        
        request = ProcessingRequest(
            chunk=chunk,
            prompt_template="Process this content: {content}",
            context={},
            metadata={"source": "test"}
        )
        
        with patch.object(llm_handler, '_make_api_call', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"content": "Processed content", "usage": {"total_tokens": 50}}
            
            result = await llm_handler._process_single_request(request)
            
            assert isinstance(result, ProcessingResult)
            assert result.error is None
            assert result.processed_content == "Processed content"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, llm_handler):
        """Test batch processing of multiple chunks."""
        from core.chunking_processor import DocumentChunk, ChunkMetadata
        
        chunks = []
        for i in range(3):
            metadata = ChunkMetadata(
                chunk_id=f"chunk_{i}",
                index=i,
                start_position=i*10,
                end_position=(i+1)*10,
                character_count=10,
                word_count=2,
                sentence_count=1,
                paragraph_count=1
            )
            
            chunk = DocumentChunk(
                content=f"Content {i}",
                metadata=metadata,
                source_document="test.txt"
            )
            
            chunks.append(chunk)
        
        with patch.object(llm_handler, '_make_api_call', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {"content": "Processed", "usage": {"total_tokens": 30}}
            
            batch_result = await llm_handler.process_chunks_async(chunks, "Process: {content}", context={})
            
            assert batch_result.total_chunks == 3
            assert batch_result.successful_chunks == 3
            assert len(batch_result.results) == 3
    
    def test_retry_mechanism(self, llm_handler):
        """Test retry mechanism for failed requests."""
        # This would test the retry logic in real implementation
        assert llm_handler.config.max_retries >= 0
        assert llm_handler.config.retry_delay >= 0


class TestKnowledgeGraphProcessor:
    """Test cases for KnowledgeGraphProcessor."""
    
    @pytest.fixture
    def kg_config(self):
        return ExtractionConfig(
            extraction_method="hybrid",
            min_confidence=0.7,
            max_entities_per_chunk=10
        )
    
    @pytest.fixture
    def kg_processor(self, kg_config):
        return KnowledgeGraphProcessor(kg_config)
    
    def test_initialization(self, kg_processor, kg_config):
        """Test knowledge graph processor initialization."""
        assert kg_processor.config == kg_config
        assert kg_processor.extraction_method == "hybrid"
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, kg_processor):
        """Test entity extraction from text."""
        text = "Apple Inc. is a technology company founded by Steve Jobs in California."
        
        with patch.object(kg_processor, '_extract_entities_nlp') as mock_extract:
            mock_extract.return_value = [
                {'text': 'Apple Inc.', 'label': 'ORG', 'confidence': 0.9},
                {'text': 'Steve Jobs', 'label': 'PERSON', 'confidence': 0.95},
                {'text': 'California', 'label': 'GPE', 'confidence': 0.8}
            ]
            
            entities = await kg_processor.extract_entities(text)
            
            assert len(entities) == 3
            assert any(entity['text'] == 'Apple Inc.' for entity in entities)
    
    @pytest.mark.asyncio
    async def test_relationship_extraction(self, kg_processor):
        """Test relationship extraction between entities."""
        entities = [
            {'text': 'Apple Inc.', 'label': 'ORG'},
            {'text': 'Steve Jobs', 'label': 'PERSON'}
        ]
        text = "Steve Jobs founded Apple Inc."
        
        with patch.object(kg_processor, '_extract_relationships_llm') as mock_extract:
            mock_extract.return_value = [
                {
                    'subject': 'Steve Jobs',
                    'predicate': 'founded',
                    'object': 'Apple Inc.',
                    'confidence': 0.9
                }
            ]
            
            relationships = await kg_processor.extract_relationships(entities, text)
            
            assert len(relationships) == 1
            assert relationships[0]['predicate'] == 'founded'
    
    def test_graph_construction(self, kg_processor):
        """Test knowledge graph construction."""
        entities = [{'text': 'Entity1', 'label': 'ORG'}]
        relationships = [{
            'subject': 'Entity1',
            'predicate': 'relates_to',
            'object': 'Entity2'
        }]
        
        graph = kg_processor.build_graph(entities, relationships)
        
        assert graph is not None
        assert len(graph.nodes()) >= 1


class TestOutputGenerator:
    """Test cases for OutputGenerator."""
    
    @pytest.fixture
    def output_generator(self):
        return OutputGenerator()
    
    def test_initialization(self, output_generator):
        """Test output generator initialization."""
        assert output_generator.supported_formats
        assert OutputFormat.MARKDOWN in output_generator.supported_formats
    
    @pytest.mark.asyncio
    async def test_markdown_generation(self, output_generator):
        """Test markdown output generation."""
        content = {
            'title': 'Test Document',
            'sections': [
                {'title': 'Introduction', 'content': 'This is the introduction.'},
                {'title': 'Conclusion', 'content': 'This is the conclusion.'}
            ]
        }
        
        result = await output_generator.generate(
            content=content,
            format_type=OutputFormat.MARKDOWN
        )
        
        assert result.success
        assert '# Test Document' in result.content
        assert '## Introduction' in result.content
    
    @pytest.mark.asyncio
    async def test_template_rendering(self, output_generator):
        """Test template-based rendering."""
        template = OutputTemplate(
            name="test_template",
            content="# {{title}}\n\n{{content}}",
            format_type=OutputFormat.MARKDOWN
        )
        
        data = {'title': 'Test Title', 'content': 'Test content'}
        
        result = await output_generator.render_template(template, data)
        
        assert '# Test Title' in result
        assert 'Test content' in result


class TestErrorHandler:
    """Test cases for ErrorHandler."""
    
    @pytest.fixture
    def error_handler(self):
        return ErrorHandler()
    
    def test_error_classification(self, error_handler):
        """Test error type classification."""
        # Test different error types
        connection_error = ConnectionError("Connection failed")
        value_error = ValueError("Invalid value")
        
        assert error_handler.classifier.classify_error(connection_error)[0] == ErrorCategory.NETWORK
        assert error_handler.classifier.classify_error(value_error)[0] == ErrorCategory.VALIDATION
        
        # Test critical error handling
        critical_error = MemoryError("Out of memory")
        recovery_strategy = error_handler.classifier.get_recovery_strategy(ErrorCategory.MEMORY)
        assert recovery_strategy.strategy == RecoveryStrategy.ABORT
    
    @pytest.mark.asyncio
    async def test_recovery_strategies(self, error_handler):
        """Test error recovery strategies."""
        error = Exception("Test error")
        
        # Test error handling
        result = await error_handler.handle_error(
            error, 
            component="test_component",
            operation="test_operation"
        )
        
        assert result.category == ErrorCategory.UNKNOWN
        assert result.exception == error
    
    def test_circuit_breaker(self, error_handler):
        """Test circuit breaker functionality."""
        # Get circuit breaker for component
        circuit_breaker = error_handler.get_circuit_breaker("test_component")
        
        # Simulate multiple failures
        for _ in range(5):
            circuit_breaker.record_failure()
        
        assert not circuit_breaker.can_execute()


class TestMonitoring:
    """Test cases for monitoring components."""
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector()
    
    @pytest.fixture
    def system_monitor(self):
        return SystemMonitor()
    
    def test_metrics_collection(self, metrics_collector):
        """Test metrics collection functionality."""
        metrics_collector.increment_counter("test_counter")
        metrics_collector.record_histogram("test_histogram", 1.5)
        metrics_collector.set_gauge("test_gauge", 100)
        
        metrics = metrics_collector.get_metrics()
        
        assert "test_counter" in metrics
        assert "test_histogram" in metrics
        assert "test_gauge" in metrics
    
    def test_system_monitoring(self, system_monitor):
        """Test system resource monitoring."""
        stats = system_monitor.get_system_stats()
        
        assert "cpu_percent" in stats
        assert "memory_percent" in stats
        assert "disk_usage" in stats
        assert all(isinstance(v, (int, float)) for v in stats.values())
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test performance tracking context manager."""
        tracker = PerformanceTracker()
        
        async with tracker.track("test_operation"):
            await asyncio.sleep(0.1)  # Simulate work
        
        metrics = tracker.get_metrics()
        assert "test_operation" in metrics
        assert metrics["test_operation"]["duration"] >= 0.1


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_config_loading(self):
        """Test configuration loading from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
            application:
              name: test_app
              version: 1.0.0
            
            llm:
              provider: openai
              model: gpt-3.5-turbo
            """)
            f.flush()
            
            config = ConfigManager.load_config(f.name)
            
            assert config.application.name == "test_app"
            assert config.llm.provider == "openai"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ProcessingConfig()
        
        # Test valid configuration
        assert ConfigManager.validate_config(config)
        
        # Test invalid configuration
        config.chunking.max_chunk_size = -1  # Invalid size
        assert not ConfigManager.validate_config(config)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ProcessingConfig()
        
        # Test to dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "application" in config_dict
        
        # Test from dict
        new_config = ProcessingConfig.from_dict(config_dict)
        assert new_config.application.name == config.application.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])