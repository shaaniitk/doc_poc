"""Integration tests for the LangGraph document processing pipeline."""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Import core components
from core.state_manager import CentralizedStateManager, ProcessingStage
from core.langgraph_orchestrator import LangGraphOrchestrator
from core.document_parser import DocumentParser
from core.chunking_processor import ChunkingProcessor
from core.llm_handler import LLMHandler, LLMConfig
from core.knowledge_graph_processor import KnowledgeGraphProcessor
from core.output_generator import OutputGenerator
from core.config import ApplicationConfig, get_config
from enhanced_main import EnhancedDocumentProcessor


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    async def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        (temp_dir / "input").mkdir()
        (temp_dir / "output").mkdir()
        (temp_dir / "cache").mkdir()
        (temp_dir / "temp").mkdir()
        (temp_dir / "logs").mkdir()
        
        # Create test document
        test_doc = temp_dir / "input" / "test.txt"
        test_doc.write_text(
            "This is a test document for integration testing. "
            "It contains multiple sentences and paragraphs. "
            "The document discusses artificial intelligence and machine learning. "
            "These technologies are transforming various industries."
        )
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_workspace):
        """Create mock configuration for testing."""
        config = ApplicationConfig()
        config.storage.input_dir = temp_workspace / "input"
        config.storage.output_dir = temp_workspace / "output"
        config.storage.cache_dir = temp_workspace / "cache"
        config.storage.temp_dir = temp_workspace / "temp"
        config.storage.log_dir = temp_workspace / "logs"
        
        # Configure for testing
        config.processing.max_concurrent = 2
        config.processing.timeout_seconds = 30
        config.chunking.chunk_size = 100
        config.chunking.chunk_overlap = 20
        
        return config
    
    @pytest.fixture
    def mock_llm_handler(self):
        """Create mock LLM handler."""
        handler = AsyncMock(spec=LLMHandler)
        handler.process_chunk.return_value = {
            "processed_text": "Processed content",
            "summary": "Test summary",
            "key_points": ["Point 1", "Point 2"],
            "quality_score": 0.9
        }
        return handler
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, temp_workspace, mock_config, mock_llm_handler):
        """Test complete document processing pipeline."""
        # Initialize components
        state_manager = CentralizedStateManager()
        
        with patch('core.llm_handler.LLMHandler', return_value=mock_llm_handler):
            orchestrator = LangGraphOrchestrator(state_manager, mock_config)
            
            # Process document
            input_file = temp_workspace / "input" / "test.txt"
            result = await orchestrator.process_document(str(input_file))
            
            # Verify results
            assert result is not None
            assert "document_id" in result
            assert "processing_time" in result
            assert "status" in result
            
            # Verify state progression
            state = state_manager.get_state(result["document_id"])
            assert state["stage"] == ProcessingStage.COMPLETED
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, temp_workspace, mock_config):
        """Test error handling and recovery across components."""
        state_manager = CentralizedStateManager()
        
        # Mock LLM handler that fails initially
        failing_handler = AsyncMock(spec=LLMHandler)
        failing_handler.process_chunk.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"),
            {"processed_text": "Success after retries", "quality_score": 0.8}
        ]
        
        with patch('core.llm_handler.LLMHandler', return_value=failing_handler):
            orchestrator = LangGraphOrchestrator(state_manager, mock_config)
            
            input_file = temp_workspace / "input" / "test.txt"
            result = await orchestrator.process_document(str(input_file))
            
            # Should eventually succeed after retries
            assert result is not None
            assert failing_handler.process_chunk.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, temp_workspace, mock_config, mock_llm_handler):
        """Test concurrent processing of multiple documents."""
        # Create multiple test documents
        for i in range(3):
            doc_path = temp_workspace / "input" / f"test_{i}.txt"
            doc_path.write_text(f"Test document {i} content for concurrent processing.")
        
        state_manager = CentralizedStateManager()
        
        with patch('core.llm_handler.LLMHandler', return_value=mock_llm_handler):
            orchestrator = LangGraphOrchestrator(state_manager, mock_config)
            
            # Process documents concurrently
            tasks = []
            for i in range(3):
                input_file = temp_workspace / "input" / f"test_{i}.txt"
                task = orchestrator.process_document(str(input_file))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Verify all documents processed
            assert len(results) == 3
            for result in results:
                assert result is not None
                assert "document_id" in result
    
    @pytest.mark.asyncio
    async def test_state_consistency_across_components(self, temp_workspace, mock_config, mock_llm_handler):
        """Test state consistency across all pipeline components."""
        state_manager = CentralizedStateManager()
        
        with patch('core.llm_handler.LLMHandler', return_value=mock_llm_handler):
            orchestrator = LangGraphOrchestrator(state_manager, mock_config)
            
            input_file = temp_workspace / "input" / "test.txt"
            result = await orchestrator.process_document(str(input_file))
            
            doc_id = result["document_id"]
            final_state = state_manager.get_state(doc_id)
            
            # Verify state contains all expected components
            assert "document" in final_state
            assert "chunks" in final_state
            assert "processed_chunks" in final_state
            assert "knowledge_graph" in final_state
            assert "output" in final_state
            
            # Verify stage progression was tracked
            history = state_manager.get_processing_history(doc_id)
            expected_stages = [
                ProcessingStage.PARSING,
                ProcessingStage.CHUNKING,
                ProcessingStage.LLM_PROCESSING,
                ProcessingStage.KNOWLEDGE_GRAPH,
                ProcessingStage.OUTPUT_GENERATION,
                ProcessingStage.COMPLETED
            ]
            
            recorded_stages = [entry["stage"] for entry in history]
            for stage in expected_stages:
                assert stage in recorded_stages


class TestComponentIntegration:
    """Integration tests for individual component interactions."""
    
    @pytest.fixture
    def sample_document(self):
        """Sample document for testing."""
        return {
            "content": "Artificial intelligence is transforming healthcare. Machine learning algorithms can analyze medical data to improve diagnosis and treatment.",
            "metadata": {
                "title": "AI in Healthcare",
                "author": "Test Author",
                "format": "text"
            }
        }
    
    @pytest.mark.asyncio
    async def test_parser_chunker_integration(self, sample_document):
        """Test integration between document parser and chunking processor."""
        # Mock document parser
        parser = AsyncMock(spec=DocumentParser)
        parser.parse_document.return_value = sample_document
        
        # Real chunking processor
        chunker = ChunkingProcessor()
        
        # Test integration
        parsed_doc = await parser.parse_document("test.txt")
        chunks = await chunker.process_document(parsed_doc)
        
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_chunker_llm_integration(self, sample_document):
        """Test integration between chunking processor and LLM handler."""
        chunker = ChunkingProcessor()
        
        # Mock LLM handler
        llm_handler = AsyncMock(spec=LLMHandler)
        llm_handler.process_chunk.return_value = {
            "processed_text": "Enhanced content",
            "summary": "AI healthcare summary",
            "quality_score": 0.9
        }
        
        # Test integration
        chunks = await chunker.process_document(sample_document)
        processed_chunks = []
        
        for chunk in chunks:
            processed = await llm_handler.process_chunk(chunk)
            processed_chunks.append(processed)
        
        assert len(processed_chunks) == len(chunks)
        assert all("processed_text" in chunk for chunk in processed_chunks)
    
    @pytest.mark.asyncio
    async def test_llm_knowledge_graph_integration(self):
        """Test integration between LLM handler and knowledge graph processor."""
        # Mock processed chunks from LLM
        processed_chunks = [
            {
                "processed_text": "Artificial intelligence improves healthcare diagnosis.",
                "entities": ["artificial intelligence", "healthcare", "diagnosis"],
                "relationships": [("artificial intelligence", "improves", "diagnosis")]
            },
            {
                "processed_text": "Machine learning analyzes medical data effectively.",
                "entities": ["machine learning", "medical data"],
                "relationships": [("machine learning", "analyzes", "medical data")]
            }
        ]
        
        kg_processor = KnowledgeGraphProcessor()
        
        # Test integration
        knowledge_graph = await kg_processor.build_knowledge_graph(processed_chunks)
        
        assert "entities" in knowledge_graph
        assert "relationships" in knowledge_graph
        assert len(knowledge_graph["entities"]) > 0
        assert len(knowledge_graph["relationships"]) > 0
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_output_integration(self):
        """Test integration between knowledge graph processor and output generator."""
        # Mock knowledge graph
        knowledge_graph = {
            "entities": [
                {"id": "ai", "label": "Artificial Intelligence", "type": "technology"},
                {"id": "healthcare", "label": "Healthcare", "type": "domain"}
            ],
            "relationships": [
                {"source": "ai", "target": "healthcare", "type": "transforms"}
            ],
            "metrics": {"node_count": 2, "edge_count": 1}
        }
        
        output_generator = OutputGenerator()
        
        # Test integration
        output = await output_generator.generate_output(
            processed_chunks=[{"processed_text": "Test content"}],
            knowledge_graph=knowledge_graph,
            output_format="markdown"
        )
        
        assert output is not None
        assert "content" in output
        assert len(output["content"]) > 0


class TestPerformanceIntegration:
    """Performance-focused integration tests."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_processing(self, temp_workspace, mock_config):
        """Test memory usage remains within acceptable limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger test document
        large_doc = temp_workspace / "input" / "large_test.txt"
        content = "This is a test sentence. " * 1000  # ~25KB
        large_doc.write_text(content)
        
        state_manager = CentralizedStateManager()
        
        # Mock LLM handler
        mock_llm = AsyncMock(spec=LLMHandler)
        mock_llm.process_chunk.return_value = {
            "processed_text": "Processed",
            "quality_score": 0.8
        }
        
        with patch('core.llm_handler.LLMHandler', return_value=mock_llm):
            orchestrator = LangGraphOrchestrator(state_manager, mock_config)
            
            # Process document
            await orchestrator.process_document(str(large_doc))
            
            # Check memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB for this test)
            assert memory_increase < 100
    
    @pytest.mark.asyncio
    async def test_processing_time_scalability(self, temp_workspace, mock_config):
        """Test processing time scales reasonably with document size."""
        import time
        
        state_manager = CentralizedStateManager()
        
        # Mock fast LLM handler
        mock_llm = AsyncMock(spec=LLMHandler)
        mock_llm.process_chunk.return_value = {
            "processed_text": "Fast processing",
            "quality_score": 0.8
        }
        
        processing_times = []
        
        with patch('core.llm_handler.LLMHandler', return_value=mock_llm):
            orchestrator = LangGraphOrchestrator(state_manager, mock_config)
            
            # Test with different document sizes
            for size_multiplier in [1, 2, 4]:
                doc_path = temp_workspace / "input" / f"size_test_{size_multiplier}.txt"
                content = "Test sentence. " * (100 * size_multiplier)
                doc_path.write_text(content)
                
                start_time = time.time()
                await orchestrator.process_document(str(doc_path))
                end_time = time.time()
                
                processing_times.append(end_time - start_time)
        
        # Processing time should scale sub-linearly (due to parallelization)
        # Time for 4x document should be less than 4x the time for 1x document
        assert processing_times[2] < processing_times[0] * 3.5


class TestConfigurationIntegration:
    """Integration tests for configuration management."""
    
    @pytest.mark.asyncio
    async def test_config_driven_pipeline_behavior(self, temp_workspace):
        """Test that configuration properly drives pipeline behavior."""
        # Create test document
        test_doc = temp_workspace / "input" / "config_test.txt"
        test_doc.write_text("Configuration test document content.")
        
        # Test with different configurations
        configs = [
            {"processing": {"mode": "sequential", "max_concurrent": 1}},
            {"processing": {"mode": "parallel", "max_concurrent": 3}},
            {"chunking": {"chunk_size": 50, "chunk_overlap": 10}},
            {"chunking": {"chunk_size": 200, "chunk_overlap": 40}}
        ]
        
        for config_override in configs:
            # Create config with override
            config = ApplicationConfig()
            config.storage.input_dir = temp_workspace / "input"
            config.storage.output_dir = temp_workspace / "output"
            
            # Apply overrides
            for section, values in config_override.items():
                section_config = getattr(config, section)
                for key, value in values.items():
                    setattr(section_config, key, value)
            
            # Test processing with this configuration
            state_manager = CentralizedStateManager()
            
            mock_llm = AsyncMock(spec=LLMHandler)
            mock_llm.process_chunk.return_value = {
                "processed_text": "Config test result",
                "quality_score": 0.8
            }
            
            with patch('core.llm_handler.LLMHandler', return_value=mock_llm):
                orchestrator = LangGraphOrchestrator(state_manager, config)
                result = await orchestrator.process_document(str(test_doc))
                
                # Verify processing completed successfully
                assert result is not None
                assert "document_id" in result
    
    def test_environment_variable_integration(self):
        """Test configuration loading from environment variables."""
        import os
        from core.config import ConfigManager
        
        # Set test environment variables
        test_env_vars = {
            "DOC_PROCESSING_ENV": "testing",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4",
            "MAX_CONCURRENT": "3",
            "LOG_LEVEL": "DEBUG"
        }
        
        # Temporarily set environment variables
        original_values = {}
        for key, value in test_env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Load configuration
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Verify environment variables were applied
            assert config.environment.value == "testing"
            assert config.llm.provider.value == "openai"
            assert config.llm.model == "gpt-4"
            assert config.processing.max_concurrent == 3
            assert config.monitoring.log_level == "DEBUG"
            
        finally:
            # Restore original environment
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])