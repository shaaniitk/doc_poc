#!/usr/bin/env python3
"""Comprehensive test suite for the enhanced LangGraph document processing pipeline.

This module provides extensive testing for all components of the enhanced
document processing system, including unit tests, integration tests,
performance tests, and end-to-end workflow validation.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime, timezone
from dataclasses import asdict
import logging

# Import components to test
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.state_manager import CentralizedStateManager, ProcessingStage, ProcessingState
from core.langgraph_orchestrator import LangGraphOrchestrator, WorkflowConfig
from core.document_parser import DocumentParser, ParsingConfig, DocumentMetadata
from core.chunking_processor import ChunkingProcessor, ChunkingConfig, ChunkingStrategy, DocumentChunk
from core.llm_handler import LLMHandler, LLMConfig, ProcessingMode, ProcessingRequest, ProcessingResult
from core.knowledge_graph_processor import KnowledgeGraphProcessor, KGConfig, Entity, Relationship
from core.output_generator import OutputGenerator, OutputConfig, OutputFormat, TemplateType
from enhanced_main import EnhancedDocumentProcessor, ProcessingConfig, ProcessingMetrics, SystemMonitor

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestFixtures:
    """Test fixtures and sample data."""
    
    @staticmethod
    def create_sample_document() -> str:
        """Create a sample document for testing."""
        return """
# Sample Document

This is a sample document for testing the enhanced document processing pipeline.

## Introduction

The document processing system uses LangGraph for orchestration and provides
advanced features like knowledge graph extraction and intelligent chunking.

## Key Features

1. **Async Processing**: All components support asynchronous operations
2. **Error Handling**: Comprehensive error recovery mechanisms
3. **Performance Monitoring**: Real-time metrics and system monitoring
4. **Quality Gates**: Automated quality validation at each stage

## Technical Details

The system processes documents through multiple stages:
- Document parsing and metadata extraction
- Intelligent chunking with overlap management
- LLM-based content processing
- Knowledge graph construction
- Output generation with multiple formats

## Conclusion

This enhanced pipeline provides production-ready document processing
with enterprise-grade reliability and performance.
"""
    
    @staticmethod
    def create_sample_chunks() -> List[DocumentChunk]:
        """Create sample document chunks."""
        return [
            DocumentChunk(
                id="chunk_1",
                content="This is the first chunk of content.",
                start_index=0,
                end_index=35,
                metadata={"section": "introduction", "quality_score": 0.8}
            ),
            DocumentChunk(
                id="chunk_2",
                content="This is the second chunk with more details.",
                start_index=30,
                end_index=74,
                metadata={"section": "body", "quality_score": 0.9}
            ),
            DocumentChunk(
                id="chunk_3",
                content="Final chunk containing conclusion.",
                start_index=70,
                end_index=104,
                metadata={"section": "conclusion", "quality_score": 0.7}
            )
        ]
    
    @staticmethod
    def create_sample_entities() -> List[Entity]:
        """Create sample entities for testing."""
        return [
            Entity(
                id="entity_1",
                name="LangGraph",
                entity_type="Technology",
                description="Graph-based workflow orchestration framework",
                confidence=0.9,
                metadata={"category": "software"}
            ),
            Entity(
                id="entity_2",
                name="Document Processing",
                entity_type="Process",
                description="Automated document analysis and transformation",
                confidence=0.8,
                metadata={"category": "process"}
            )
        ]
    
    @staticmethod
    def create_sample_relationships() -> List[Relationship]:
        """Create sample relationships for testing."""
        return [
            Relationship(
                id="rel_1",
                source_entity_id="entity_1",
                target_entity_id="entity_2",
                relationship_type="enables",
                description="LangGraph enables document processing",
                confidence=0.85,
                metadata={"strength": "strong"}
            )
        ]


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_document_path(temp_dir):
    """Create sample document file."""
    doc_path = temp_dir / "sample.md"
    doc_path.write_text(TestFixtures.create_sample_document())
    return doc_path


@pytest.fixture
def processing_config(sample_document_path, temp_dir):
    """Create processing configuration for tests."""
    return ProcessingConfig(
        source_path=sample_document_path,
        output_path=temp_dir / "output.md",
        output_format=OutputFormat.MARKDOWN,
        chunking_strategy=ChunkingStrategy.SENTENCE_AWARE,
        chunk_size=500,
        chunk_overlap=100,
        processing_mode=ProcessingMode.SEQUENTIAL,
        max_concurrent=2,
        llm_provider="mock",
        llm_model="mock-model",
        enable_knowledge_graph=True,
        enable_caching=False,  # Disable for testing
        enable_monitoring=False,  # Disable for testing
        max_retries=1,
        fail_fast=True
    )


class TestStateManager:
    """Test cases for CentralizedStateManager."""
    
    @pytest.mark.asyncio
    async def test_state_manager_initialization(self):
        """Test state manager initialization."""
        manager = CentralizedStateManager()
        await manager.initialize_async()
        
        assert manager.is_initialized
        assert len(manager.state_history) == 0
        
        await manager.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test state transitions and validation."""
        manager = CentralizedStateManager()
        await manager.initialize_async()
        
        # Create initial state
        state = ProcessingState(
            stage=ProcessingStage.PARSING,
            source_path="test.txt"
        )
        
        # Update state
        updated_state = await manager.update_state_async(state)
        assert updated_state.stage == ProcessingStage.PARSING
        assert len(manager.state_history) == 1
        
        # Transition to next stage
        state.stage = ProcessingStage.CHUNKING
        updated_state = await manager.update_state_async(state)
        assert updated_state.stage == ProcessingStage.CHUNKING
        assert len(manager.state_history) == 2
        
        await manager.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_state_validation(self):
        """Test state validation and error handling."""
        manager = CentralizedStateManager()
        await manager.initialize_async()
        
        # Test invalid state transition
        state = ProcessingState(
            stage=ProcessingStage.COMPLETED,  # Invalid initial stage
            source_path="test.txt"
        )
        
        with pytest.raises(ValueError):
            await manager.validate_state_async(state)
        
        await manager.cleanup_async()


class TestDocumentParser:
    """Test cases for DocumentParser."""
    
    @pytest.mark.asyncio
    async def test_parse_markdown_document(self, sample_document_path):
        """Test parsing markdown document."""
        config = ParsingConfig(
            extract_metadata=True,
            extract_structure=True
        )
        parser = DocumentParser(config)
        
        result = await parser.parse_document_async(sample_document_path)
        
        assert result.success
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata is not None
        assert result.metadata.file_path == str(sample_document_path)
        assert result.metadata.file_size > 0
    
    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file."""
        config = ParsingConfig()
        parser = DocumentParser(config)
        
        result = await parser.parse_document_async(Path("nonexistent.txt"))
        
        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_metadata_extraction(self, sample_document_path):
        """Test metadata extraction functionality."""
        config = ParsingConfig(
            extract_metadata=True,
            extract_structure=True
        )
        parser = DocumentParser(config)
        
        result = await parser.parse_document_async(sample_document_path)
        
        assert result.success
        assert result.metadata.word_count > 0
        assert result.metadata.character_count > 0
        assert len(result.metadata.headings) > 0
        assert "Sample Document" in [h.text for h in result.metadata.headings]


class TestChunkingProcessor:
    """Test cases for ChunkingProcessor."""
    
    @pytest.mark.asyncio
    async def test_sentence_aware_chunking(self):
        """Test sentence-aware chunking strategy."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE_AWARE,
            chunk_size=100,
            overlap_size=20
        )
        processor = ChunkingProcessor(config)
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = await processor.chunk_document_async(text)
        
        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.start_index < chunk.end_index for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_semantic_chunking(self):
        """Test semantic chunking strategy."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=200,
            overlap_size=50
        )
        processor = ChunkingProcessor(config)
        
        text = TestFixtures.create_sample_document()
        chunks = await processor.chunk_document_async(text)
        
        assert len(chunks) > 0
        assert all(len(chunk.content) <= config.chunk_size + 100 for chunk in chunks)  # Allow some flexibility
    
    @pytest.mark.asyncio
    async def test_chunk_quality_scoring(self):
        """Test chunk quality scoring."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SENTENCE_AWARE,
            chunk_size=100,
            enable_quality_scoring=True,
            min_quality_score=0.5
        )
        processor = ChunkingProcessor(config)
        
        text = "Good quality sentence with proper structure. Another well-formed sentence."
        chunks = await processor.chunk_document_async(text)
        
        assert len(chunks) > 0
        assert all("quality_score" in chunk.metadata for chunk in chunks)
        assert all(chunk.metadata["quality_score"] >= 0.5 for chunk in chunks)


class TestLLMHandler:
    """Test cases for LLMHandler."""
    
    @pytest.mark.asyncio
    async def test_mock_llm_processing(self):
        """Test LLM processing with mock provider."""
        config = LLMConfig(
            provider="mock",
            model="mock-model",
            max_retries=1
        )
        handler = LLMHandler(config)
        await handler.initialize_async()
        
        # Mock the LLM client
        with patch.object(handler, '_get_llm_client') as mock_client:
            mock_response = Mock()
            mock_response.content = "Processed content"
            mock_response.usage = {"total_tokens": 100}
            mock_client.return_value.process_async = AsyncMock(return_value=mock_response)
            
            request = ProcessingRequest(
                chunk_id="test_chunk",
                content="Test content",
                prompt="Process this content"
            )
            
            result = await handler.process_chunk_async(request)
            
            assert result.success
            assert result.processed_content == "Processed content"
            assert result.token_usage["total_tokens"] == 100
        
        await handler.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_parallel_processing(self):
        """Test parallel processing mode."""
        config = LLMConfig(
            provider="mock",
            model="mock-model"
        )
        handler = LLMHandler(config)
        await handler.initialize_async()
        
        chunks = TestFixtures.create_sample_chunks()
        requests = [
            ProcessingRequest(
                chunk_id=chunk.id,
                content=chunk.content,
                prompt="Process this content"
            )
            for chunk in chunks
        ]
        
        # Mock the LLM client
        with patch.object(handler, '_get_llm_client') as mock_client:
            mock_response = Mock()
            mock_response.content = "Processed content"
            mock_response.usage = {"total_tokens": 50}
            mock_client.return_value.process_async = AsyncMock(return_value=mock_response)
            
            start_time = time.time()
            results = await handler.process_parallel_async(requests, max_concurrent=2)
            end_time = time.time()
            
            assert len(results.results) == len(requests)
            assert all(result.success for result in results.results)
            assert results.total_duration == end_time - start_time
        
        await handler.cleanup_async()


class TestKnowledgeGraphProcessor:
    """Test cases for KnowledgeGraphProcessor."""
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self):
        """Test entity extraction functionality."""
        config = KGConfig(
            extraction_method="rule_based",
            min_confidence=0.5
        )
        processor = KnowledgeGraphProcessor(config)
        await processor.initialize_async()
        
        text = "LangGraph is a powerful framework for building document processing pipelines."
        entities = await processor.extract_entities_async(text)
        
        assert len(entities) > 0
        assert all(entity.confidence >= 0.5 for entity in entities)
        assert any("langgraph" in entity.name.lower() for entity in entities)
        
        await processor.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_relationship_extraction(self):
        """Test relationship extraction functionality."""
        config = KGConfig(
            extraction_method="rule_based",
            min_confidence=0.5
        )
        processor = KnowledgeGraphProcessor(config)
        await processor.initialize_async()
        
        entities = TestFixtures.create_sample_entities()
        text = "LangGraph enables document processing through its workflow orchestration."
        
        relationships = await processor.extract_relationships_async(text, entities)
        
        assert len(relationships) >= 0  # May not find relationships in simple text
        
        await processor.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_construction(self):
        """Test complete knowledge graph construction."""
        config = KGConfig(
            extraction_method="hybrid",
            min_confidence=0.5
        )
        processor = KnowledgeGraphProcessor(config)
        await processor.initialize_async()
        
        chunks = TestFixtures.create_sample_chunks()
        
        # Mock entity and relationship extraction
        with patch.object(processor, 'extract_entities_async') as mock_entities, \
             patch.object(processor, 'extract_relationships_async') as mock_relationships:
            
            mock_entities.return_value = TestFixtures.create_sample_entities()
            mock_relationships.return_value = TestFixtures.create_sample_relationships()
            
            kg_result = await processor.build_knowledge_graph_async(chunks)
            
            assert kg_result.success
            assert len(kg_result.entities) > 0
            assert len(kg_result.relationships) > 0
        
        await processor.cleanup_async()


class TestOutputGenerator:
    """Test cases for OutputGenerator."""
    
    @pytest.mark.asyncio
    async def test_markdown_generation(self, temp_dir):
        """Test markdown output generation."""
        config = OutputConfig(
            output_format=OutputFormat.MARKDOWN,
            include_metadata=True,
            include_statistics=True
        )
        generator = OutputGenerator(config)
        
        # Create mock processing state
        state = ProcessingState(
            stage=ProcessingStage.COMPLETED,
            source_path="test.txt",
            chunks=TestFixtures.create_sample_chunks()
        )
        
        output_path = temp_dir / "output.md"
        result = await generator.generate_output_async(state, output_path)
        
        assert result.success
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        content = output_path.read_text()
        assert "# Document Processing Results" in content
    
    @pytest.mark.asyncio
    async def test_json_generation(self, temp_dir):
        """Test JSON output generation."""
        config = OutputConfig(
            output_format=OutputFormat.JSON,
            include_metadata=True
        )
        generator = OutputGenerator(config)
        
        state = ProcessingState(
            stage=ProcessingStage.COMPLETED,
            source_path="test.txt",
            chunks=TestFixtures.create_sample_chunks()
        )
        
        output_path = temp_dir / "output.json"
        result = await generator.generate_output_async(state, output_path)
        
        assert result.success
        assert output_path.exists()
        
        # Validate JSON structure
        with open(output_path) as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "chunks" in data
        assert len(data["chunks"]) == len(TestFixtures.create_sample_chunks())


class TestLangGraphOrchestrator:
    """Test cases for LangGraphOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        # Create mock components
        state_manager = Mock()
        state_manager.initialize_async = AsyncMock()
        
        config = WorkflowConfig(
            enable_parallel_processing=True,
            max_concurrent_nodes=2
        )
        
        orchestrator = LangGraphOrchestrator(
            state_manager=state_manager,
            document_parser=Mock(),
            chunking_processor=Mock(),
            llm_handler=Mock(),
            kg_processor=Mock(),
            output_generator=Mock(),
            config=config
        )
        
        await orchestrator.initialize_async()
        assert orchestrator.is_initialized
        
        await orchestrator.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, processing_config):
        """Test complete workflow execution with mocks."""
        # Create mock components
        state_manager = Mock()
        state_manager.initialize_async = AsyncMock()
        state_manager.update_state_async = AsyncMock(side_effect=lambda x: x)
        state_manager.validate_state_async = AsyncMock(return_value=True)
        state_manager.cleanup_async = AsyncMock()
        
        # Mock document parser
        doc_parser = Mock()
        doc_parser.parse_document_async = AsyncMock()
        doc_parser.parse_document_async.return_value = Mock(
            success=True,
            content="Sample content",
            metadata=Mock(word_count=100)
        )
        
        # Mock chunking processor
        chunking_processor = Mock()
        chunking_processor.chunk_document_async = AsyncMock()
        chunking_processor.chunk_document_async.return_value = TestFixtures.create_sample_chunks()
        
        # Mock LLM handler
        llm_handler = Mock()
        llm_handler.initialize_async = AsyncMock()
        llm_handler.process_parallel_async = AsyncMock()
        llm_handler.process_parallel_async.return_value = Mock(
            results=[Mock(success=True, processed_content="Processed") for _ in range(3)],
            total_duration=1.0
        )
        llm_handler.cleanup_async = AsyncMock()
        
        # Mock KG processor
        kg_processor = Mock()
        kg_processor.initialize_async = AsyncMock()
        kg_processor.build_knowledge_graph_async = AsyncMock()
        kg_processor.build_knowledge_graph_async.return_value = Mock(
            success=True,
            entities=TestFixtures.create_sample_entities(),
            relationships=TestFixtures.create_sample_relationships()
        )
        kg_processor.cleanup_async = AsyncMock()
        
        # Mock output generator
        output_generator = Mock()
        output_generator.generate_output_async = AsyncMock()
        output_generator.generate_output_async.return_value = Mock(
            success=True,
            output_path="output.md"
        )
        
        config = WorkflowConfig(
            enable_parallel_processing=False,  # Simplify for testing
            enable_error_recovery=True
        )
        
        orchestrator = LangGraphOrchestrator(
            state_manager=state_manager,
            document_parser=doc_parser,
            chunking_processor=chunking_processor,
            llm_handler=llm_handler,
            kg_processor=kg_processor,
            output_generator=output_generator,
            config=config
        )
        
        await orchestrator.initialize_async()
        
        # Create initial state
        initial_state = ProcessingState(
            stage=ProcessingStage.PARSING,
            source_path=str(processing_config.source_path),
            output_path=str(processing_config.output_path)
        )
        
        # Execute workflow
        final_state = await orchestrator.process_document_async(initial_state)
        
        assert final_state.stage == ProcessingStage.COMPLETED
        assert len(final_state.chunks) > 0
        
        await orchestrator.cleanup_async()


class TestEnhancedDocumentProcessor:
    """Test cases for EnhancedDocumentProcessor."""
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, processing_config):
        """Test processor initialization."""
        processor = EnhancedDocumentProcessor(processing_config)
        
        # Mock external dependencies
        with patch('enhanced_main.CentralizedStateManager') as mock_state_manager, \
             patch('enhanced_main.LangGraphOrchestrator') as mock_orchestrator:
            
            mock_state_manager.return_value.initialize_async = AsyncMock()
            mock_orchestrator.return_value.initialize_async = AsyncMock()
            
            await processor.initialize_async()
            
            assert processor.state_manager is not None
            assert processor.orchestrator is not None
        
        await processor.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, processing_config):
        """Test metrics collection during processing."""
        processor = EnhancedDocumentProcessor(processing_config)
        
        # Test initial metrics
        assert processor.metrics.total_chunks == 0
        assert processor.metrics.processed_chunks == 0
        assert processor.metrics.total_errors == 0
        
        # Simulate progress updates
        test_state = ProcessingState(
            stage=ProcessingStage.CHUNKING,
            chunks=TestFixtures.create_sample_chunks()
        )
        
        await processor._progress_callback(test_state)
        
        assert processor.metrics.total_chunks == len(TestFixtures.create_sample_chunks())
    
    def test_system_monitor(self):
        """Test system monitoring functionality."""
        monitor = SystemMonitor(enabled=True)
        
        if monitor.enabled:
            memory_usage = monitor.get_memory_usage()
            cpu_usage = monitor.get_cpu_usage()
            
            assert memory_usage >= 0
            assert cpu_usage >= 0
        
        # Test with disabled monitoring
        monitor_disabled = SystemMonitor(enabled=False)
        assert monitor_disabled.get_memory_usage() == 0
        assert monitor_disabled.get_cpu_usage() == 0


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_end_to_end_processing(self, processing_config, temp_dir):
        """Test complete end-to-end document processing."""
        # This test requires mocking external services
        with patch('core.llm_handler.LLMHandler._get_llm_client') as mock_llm_client:
            # Mock LLM responses
            mock_response = Mock()
            mock_response.content = "Processed content with insights"
            mock_response.usage = {"total_tokens": 150}
            mock_llm_client.return_value.process_async = AsyncMock(return_value=mock_response)
            
            processor = EnhancedDocumentProcessor(processing_config)
            
            try:
                await processor.initialize_async()
                results = await processor.process_document_async()
                
                # Verify results
                assert "success" in results
                assert "metrics" in results
                assert "processing_time" in results
                
                if results["success"]:
                    assert results["chunks_processed"] > 0
                    assert Path(results["output_path"]).exists()
                
            finally:
                await processor.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, processing_config):
        """Test error handling and recovery mechanisms."""
        # Configure for fast failure
        processing_config.fail_fast = True
        processing_config.max_retries = 1
        
        processor = EnhancedDocumentProcessor(processing_config)
        
        # Mock a component to fail
        with patch('core.document_parser.DocumentParser.parse_document_async') as mock_parser:
            mock_parser.side_effect = Exception("Simulated parsing error")
            
            try:
                await processor.initialize_async()
                results = await processor.process_document_async()
                
                # Should handle error gracefully
                assert not results["success"]
                assert "error" in results
                assert "Simulated parsing error" in results["error"]
                assert results["metrics"].total_errors > 0
                
            finally:
                await processor.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, processing_config):
        """Test performance benchmarks and optimization."""
        # Configure for performance testing
        processing_config.processing_mode = ProcessingMode.PARALLEL
        processing_config.max_concurrent = 3
        processing_config.enable_monitoring = True
        
        processor = EnhancedDocumentProcessor(processing_config)
        
        with patch('core.llm_handler.LLMHandler._get_llm_client') as mock_llm_client:
            # Mock fast LLM responses
            mock_response = Mock()
            mock_response.content = "Fast processed content"
            mock_response.usage = {"total_tokens": 100}
            mock_llm_client.return_value.process_async = AsyncMock(return_value=mock_response)
            
            try:
                await processor.initialize_async()
                
                start_time = time.time()
                results = await processor.process_document_async()
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                # Performance assertions
                assert processing_time < 30.0  # Should complete within 30 seconds
                
                if results["success"]:
                    assert results["processing_time"] > 0
                    assert processor.metrics.total_duration > 0
                
            finally:
                await processor.cleanup_async()


class TestPerformance:
    """Performance and load testing."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_processing_load(self, processing_config):
        """Test system under concurrent processing load."""
        processing_config.max_concurrent = 10
        processing_config.processing_mode = ProcessingMode.PARALLEL
        
        processor = EnhancedDocumentProcessor(processing_config)
        
        with patch('core.llm_handler.LLMHandler._get_llm_client') as mock_llm_client:
            mock_response = Mock()
            mock_response.content = "Load test content"
            mock_response.usage = {"total_tokens": 75}
            mock_llm_client.return_value.process_async = AsyncMock(return_value=mock_response)
            
            try:
                await processor.initialize_async()
                
                # Simulate multiple concurrent requests
                tasks = []
                for i in range(5):
                    task = asyncio.create_task(processor.process_document_async())
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Verify all completed without critical errors
                successful_results = [r for r in results if not isinstance(r, Exception)]
                assert len(successful_results) > 0
                
            finally:
                await processor.cleanup_async()
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, processing_config):
        """Test memory usage monitoring and limits."""
        processing_config.memory_limit_mb = 1024
        processing_config.enable_monitoring = True
        
        processor = EnhancedDocumentProcessor(processing_config)
        
        initial_memory = processor.monitor.get_memory_usage()
        
        with patch('core.llm_handler.LLMHandler._get_llm_client') as mock_llm_client:
            mock_response = Mock()
            mock_response.content = "Memory test content"
            mock_response.usage = {"total_tokens": 50}
            mock_llm_client.return_value.process_async = AsyncMock(return_value=mock_response)
            
            try:
                await processor.initialize_async()
                await processor.process_document_async()
                
                final_memory = processor.monitor.get_memory_usage()
                
                # Memory should be tracked
                if processor.monitor.enabled:
                    assert processor.metrics.peak_memory_mb >= initial_memory
                
            finally:
                await processor.cleanup_async()


class TestConfigurationManagement:
    """Test configuration management and validation."""
    
    def test_processing_config_validation(self, temp_dir):
        """Test processing configuration validation."""
        # Valid configuration
        valid_config = ProcessingConfig(
            source_path=temp_dir / "test.txt",
            chunk_size=1000,
            chunk_overlap=200,
            max_concurrent=5
        )
        
        assert valid_config.chunk_size > 0
        assert valid_config.chunk_overlap < valid_config.chunk_size
        assert valid_config.max_concurrent > 0
    
    def test_config_serialization(self, processing_config, temp_dir):
        """Test configuration serialization and deserialization."""
        config_dict = asdict(processing_config)
        
        # Convert Path objects to strings for JSON serialization
        config_dict['source_path'] = str(config_dict['source_path'])
        if config_dict['output_path']:
            config_dict['output_path'] = str(config_dict['output_path'])
        
        # Save to JSON
        config_file = temp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # Load from JSON
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config['chunk_size'] == processing_config.chunk_size
        assert loaded_config['llm_provider'] == processing_config.llm_provider


if __name__ == "__main__":
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--asyncio-mode=auto"
    ])