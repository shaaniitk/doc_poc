"""Performance and stress tests for the document processing pipeline."""

import pytest
import asyncio
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import statistics
import gc
import sys

# Import components for testing
from core.state_manager import StateManager
from core.chunking_processor import ChunkingProcessor, ChunkingConfig, ChunkingStrategy
from core.llm_handler import LLMHandler, LLMConfig, ProcessingRequest
from core.knowledge_graph_processor import KnowledgeGraphProcessor, KGConfig
from core.output_generator import OutputGenerator, OutputFormat
from core.monitoring import MetricsCollector, PerformanceTracker
from enhanced_main import EnhancedDocumentProcessor, ProcessingConfig


class PerformanceTestBase:
    """Base class for performance tests with common utilities."""
    
    @staticmethod
    def measure_memory_usage():
        """Measure current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    @staticmethod
    def measure_cpu_usage():
        """Measure current CPU usage."""
        return psutil.cpu_percent(interval=1)
    
    @staticmethod
    async def time_async_operation(coro):
        """Time an async operation and return duration and result."""
        start_time = time.perf_counter()
        result = await coro
        end_time = time.perf_counter()
        return end_time - start_time, result
    
    @staticmethod
    def generate_test_document(size_kb: int = 100) -> str:
        """Generate a test document of specified size."""
        # Create realistic document content
        base_content = """
        Introduction to Advanced Computing Systems
        
        Modern computing systems have evolved significantly over the past decades.
        The integration of artificial intelligence, machine learning, and distributed
        computing has revolutionized how we process and analyze information.
        
        Key Components of Modern Systems:
        1. Processing Units: CPUs, GPUs, and specialized accelerators
        2. Memory Hierarchies: Cache systems, RAM, and storage solutions
        3. Network Infrastructure: High-speed interconnects and protocols
        4. Software Frameworks: Operating systems, middleware, and applications
        
        Performance Optimization Strategies:
        - Parallel processing and concurrent execution
        - Memory management and caching techniques
        - Load balancing and resource allocation
        - Algorithm optimization and data structure selection
        
        Future Trends:
        The future of computing lies in quantum computing, neuromorphic processors,
        and edge computing architectures that bring computation closer to data sources.
        """
        
        # Repeat content to reach desired size
        target_chars = size_kb * 1024
        current_chars = len(base_content)
        repetitions = max(1, target_chars // current_chars)
        
        return (base_content * repetitions)[:target_chars]


class TestChunkingPerformance(PerformanceTestBase):
    """Performance tests for chunking processor."""
    
    @pytest.fixture
    def chunking_processor(self):
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=512,
            overlap_size=50
        )
        return ChunkingProcessor(config)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_chunking_speed_small_document(self, chunking_processor):
        """Test chunking speed for small documents (1KB)."""
        document = self.generate_test_document(1)  # 1KB
        
        duration, chunks = await self.time_async_operation(
            chunking_processor.process_text(document)
        )
        
        # Performance assertions
        assert duration < 1.0  # Should complete within 1 second
        assert len(chunks) > 0
        
        print(f"Small document chunking: {duration:.3f}s, {len(chunks)} chunks")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_chunking_speed_medium_document(self, chunking_processor):
        """Test chunking speed for medium documents (100KB)."""
        document = self.generate_test_document(100)  # 100KB
        
        duration, chunks = await self.time_async_operation(
            chunking_processor.process_text(document)
        )
        
        # Performance assertions
        assert duration < 10.0  # Should complete within 10 seconds
        assert len(chunks) > 0
        
        print(f"Medium document chunking: {duration:.3f}s, {len(chunks)} chunks")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_chunking_speed_large_document(self, chunking_processor):
        """Test chunking speed for large documents (1MB)."""
        document = self.generate_test_document(1000)  # 1MB
        
        memory_before = self.measure_memory_usage()
        
        duration, chunks = await self.time_async_operation(
            chunking_processor.process_text(document)
        )
        
        memory_after = self.measure_memory_usage()
        memory_used = memory_after - memory_before
        
        # Performance assertions
        assert duration < 30.0  # Should complete within 30 seconds
        assert memory_used < 500  # Should use less than 500MB
        assert len(chunks) > 0
        
        print(f"Large document chunking: {duration:.3f}s, {len(chunks)} chunks, {memory_used:.1f}MB")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_chunking(self, chunking_processor):
        """Test concurrent chunking of multiple documents."""
        documents = [self.generate_test_document(10) for _ in range(5)]  # 5 x 10KB
        
        start_time = time.perf_counter()
        
        # Process documents concurrently
        tasks = [chunking_processor.process_text(doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Performance assertions
        assert duration < 15.0  # Should complete within 15 seconds
        assert len(results) == 5
        assert all(len(chunks) > 0 for chunks in results)
        
        total_chunks = sum(len(chunks) for chunks in results)
        print(f"Concurrent chunking: {duration:.3f}s, {total_chunks} total chunks")


class TestLLMHandlerPerformance(PerformanceTestBase):
    """Performance tests for LLM handler."""
    
    @pytest.fixture
    def llm_handler(self):
        config = LLMConfig(
            provider="mock",
            model="test-model",
            max_tokens=1000,
            temperature=0.7
        )
        handler = LLMHandler(config)
        
        # Mock the LLM call for performance testing
        async def mock_llm_call(prompt, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return f"Processed: {prompt[:50]}..."
        
        handler._call_llm = mock_llm_call
        return handler
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_single_chunk_processing_speed(self, llm_handler):
        """Test processing speed for a single chunk."""
        request = ProcessingRequest(
            chunk_id="test_chunk",
            content=self.generate_test_document(1),
            task_type="summarization"
        )
        
        duration, result = await self.time_async_operation(
            llm_handler.process_chunk(request)
        )
        
        assert duration < 2.0  # Should complete within 2 seconds
        assert result.success
        
        print(f"Single chunk processing: {duration:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_batch_processing_speed(self, llm_handler):
        """Test batch processing speed."""
        requests = [
            ProcessingRequest(
                chunk_id=f"chunk_{i}",
                content=self.generate_test_document(1),
                task_type="summarization"
            )
            for i in range(10)
        ]
        
        duration, results = await self.time_async_operation(
            llm_handler.process_batch(requests)
        )
        
        assert duration < 5.0  # Should complete within 5 seconds
        assert len(results) == 10
        assert all(result.success for result in results)
        
        print(f"Batch processing (10 chunks): {duration:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_processing_load(self, llm_handler):
        """Test system under high concurrent load."""
        num_concurrent = 20
        requests = [
            ProcessingRequest(
                chunk_id=f"chunk_{i}",
                content=self.generate_test_document(2),
                task_type="summarization"
            )
            for i in range(num_concurrent)
        ]
        
        memory_before = self.measure_memory_usage()
        
        # Process all requests concurrently
        start_time = time.perf_counter()
        tasks = [llm_handler.process_chunk(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.perf_counter()
        
        memory_after = self.measure_memory_usage()
        duration = end_time - start_time
        memory_used = memory_after - memory_before
        
        # Count successful results
        successful = sum(1 for r in results if hasattr(r, 'success') and r.success)
        
        assert duration < 10.0  # Should complete within 10 seconds
        assert memory_used < 200  # Should use less than 200MB
        assert successful >= num_concurrent * 0.8  # At least 80% success rate
        
        print(f"Concurrent load test: {duration:.3f}s, {successful}/{num_concurrent} successful, {memory_used:.1f}MB")


class TestKnowledgeGraphPerformance(PerformanceTestBase):
    """Performance tests for knowledge graph processor."""
    
    @pytest.fixture
    def kg_processor(self):
        config = KGConfig(
            extraction_method="rule_based",  # Faster for testing
            min_entity_confidence=0.5,
            max_entities_per_chunk=20
        )
        return KnowledgeGraphProcessor(config)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_entity_extraction_speed(self, kg_processor):
        """Test entity extraction speed."""
        text = self.generate_test_document(10)  # 10KB
        
        with patch.object(kg_processor, '_extract_entities_rule_based') as mock_extract:
            # Mock fast entity extraction
            mock_extract.return_value = [
                {'text': f'Entity_{i}', 'label': 'ORG', 'confidence': 0.8}
                for i in range(10)
            ]
            
            duration, entities = await self.time_async_operation(
                kg_processor.extract_entities(text)
            )
            
            assert duration < 2.0  # Should complete within 2 seconds
            assert len(entities) > 0
            
            print(f"Entity extraction: {duration:.3f}s, {len(entities)} entities")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_graph_construction_speed(self, kg_processor):
        """Test knowledge graph construction speed."""
        # Generate test entities and relationships
        entities = [
            {'text': f'Entity_{i}', 'label': 'ORG'}
            for i in range(100)
        ]
        
        relationships = [
            {
                'subject': f'Entity_{i}',
                'predicate': 'relates_to',
                'object': f'Entity_{i+1}'
            }
            for i in range(99)
        ]
        
        start_time = time.perf_counter()
        graph = kg_processor.build_graph(entities, relationships)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        
        assert duration < 1.0  # Should complete within 1 second
        assert graph is not None
        assert len(graph.nodes()) == 100
        
        print(f"Graph construction: {duration:.3f}s, {len(graph.nodes())} nodes")


class TestOutputGenerationPerformance(PerformanceTestBase):
    """Performance tests for output generation."""
    
    @pytest.fixture
    def output_generator(self):
        return OutputGenerator()
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_markdown_generation_speed(self, output_generator):
        """Test markdown generation speed."""
        # Generate large content structure
        content = {
            'title': 'Performance Test Document',
            'sections': [
                {
                    'title': f'Section {i}',
                    'content': self.generate_test_document(5)  # 5KB per section
                }
                for i in range(20)  # 20 sections = ~100KB total
            ]
        }
        
        duration, result = await self.time_async_operation(
            output_generator.generate(
                content=content,
                format_type=OutputFormat.MARKDOWN
            )
        )
        
        assert duration < 5.0  # Should complete within 5 seconds
        assert result.success
        assert len(result.content) > 1000  # Should generate substantial content
        
        print(f"Markdown generation: {duration:.3f}s, {len(result.content)} chars")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_multiple_format_generation(self, output_generator):
        """Test generating multiple output formats concurrently."""
        content = {
            'title': 'Multi-format Test',
            'sections': [
                {'title': 'Section 1', 'content': self.generate_test_document(2)}
            ]
        }
        
        formats = [OutputFormat.MARKDOWN, OutputFormat.JSON, OutputFormat.TEXT]
        
        start_time = time.perf_counter()
        tasks = [
            output_generator.generate(content=content, format_type=fmt)
            for fmt in formats
        ]
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        
        assert duration < 3.0  # Should complete within 3 seconds
        assert len(results) == 3
        assert all(result.success for result in results)
        
        print(f"Multi-format generation: {duration:.3f}s")


class TestSystemIntegrationPerformance(PerformanceTestBase):
    """End-to-end performance tests for the complete system."""
    
    @pytest.fixture
    def processing_config(self):
        return ProcessingConfig()
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_end_to_end_processing_speed(self, processing_config):
        """Test complete document processing pipeline speed."""
        # Create a realistic test document
        test_document = self.generate_test_document(50)  # 50KB
        
        processor = EnhancedDocumentProcessor(processing_config)
        
        # Mock external dependencies for performance testing
        with patch.object(processor.llm_handler, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Processed content"
            
            memory_before = self.measure_memory_usage()
            
            duration, result = await self.time_async_operation(
                processor.process_document(
                    content=test_document,
                    document_id="perf_test_doc"
                )
            )
            
            memory_after = self.measure_memory_usage()
            memory_used = memory_after - memory_before
            
            # Performance assertions
            assert duration < 30.0  # Should complete within 30 seconds
            assert memory_used < 300  # Should use less than 300MB
            assert result['success']
            
            print(f"End-to-end processing: {duration:.3f}s, {memory_used:.1f}MB")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_concurrent_document_processing(self, processing_config):
        """Test processing multiple documents concurrently."""
        num_documents = 3
        documents = [
            (self.generate_test_document(20), f"doc_{i}")
            for i in range(num_documents)
        ]
        
        processor = EnhancedDocumentProcessor(processing_config)
        
        with patch.object(processor.llm_handler, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Processed content"
            
            start_time = time.perf_counter()
            
            tasks = [
                processor.process_document(content=content, document_id=doc_id)
                for content, doc_id in documents
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
            
            assert duration < 60.0  # Should complete within 60 seconds
            assert successful >= num_documents * 0.8  # At least 80% success rate
            
            print(f"Concurrent processing: {duration:.3f}s, {successful}/{num_documents} successful")


class TestMemoryLeakDetection(PerformanceTestBase):
    """Tests to detect memory leaks and resource management issues."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_stability_chunking(self):
        """Test memory stability during repeated chunking operations."""
        config = ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=512)
        processor = ChunkingProcessor(config)
        
        memory_readings = []
        document = self.generate_test_document(10)  # 10KB
        
        # Perform multiple chunking operations
        for i in range(10):
            asyncio.run(processor.process_text(document))
            
            # Force garbage collection
            gc.collect()
            
            # Record memory usage
            memory_readings.append(self.measure_memory_usage())
            
            if i > 0:
                # Memory should not continuously increase
                memory_growth = memory_readings[-1] - memory_readings[0]
                assert memory_growth < 50  # Less than 50MB growth
        
        print(f"Memory stability test - Growth: {memory_readings[-1] - memory_readings[0]:.1f}MB")
    
    @pytest.mark.performance
    def test_resource_cleanup(self):
        """Test proper resource cleanup after processing."""
        initial_memory = self.measure_memory_usage()
        
        # Create and destroy multiple processors
        for _ in range(5):
            config = ProcessingConfig()
            processor = EnhancedDocumentProcessor(config)
            
            # Simulate some work
            state_manager = StateManager()
            state_manager.add_chunk({'id': 'test', 'content': 'test content'})
            
            # Explicit cleanup
            del processor
            del state_manager
            gc.collect()
        
        final_memory = self.measure_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal
        assert memory_growth < 30  # Less than 30MB growth
        
        print(f"Resource cleanup test - Growth: {memory_growth:.1f}MB")


class TestPerformanceMetrics:
    """Tests for performance monitoring and metrics collection."""
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector()
    
    @pytest.fixture
    def performance_tracker(self):
        return PerformanceTracker()
    
    def test_metrics_collection_overhead(self, metrics_collector):
        """Test that metrics collection has minimal overhead."""
        # Measure time without metrics
        start_time = time.perf_counter()
        for i in range(1000):
            pass  # Dummy operation
        baseline_time = time.perf_counter() - start_time
        
        # Measure time with metrics
        start_time = time.perf_counter()
        for i in range(1000):
            metrics_collector.increment_counter("test_counter")
            metrics_collector.record_histogram("test_histogram", i)
        metrics_time = time.perf_counter() - start_time
        
        # Overhead should be minimal
        overhead = metrics_time - baseline_time
        overhead_percentage = (overhead / baseline_time) * 100
        
        assert overhead_percentage < 50  # Less than 50% overhead
        
        print(f"Metrics overhead: {overhead_percentage:.1f}%")
    
    @pytest.mark.asyncio
    async def test_performance_tracking_accuracy(self, performance_tracker):
        """Test accuracy of performance tracking."""
        expected_duration = 0.1  # 100ms
        
        async with performance_tracker.track("test_operation"):
            await asyncio.sleep(expected_duration)
        
        metrics = performance_tracker.get_metrics()
        actual_duration = metrics["test_operation"]["duration"]
        
        # Should be within 10% of expected duration
        assert abs(actual_duration - expected_duration) < expected_duration * 0.1
        
        print(f"Tracking accuracy: {actual_duration:.3f}s (expected {expected_duration:.3f}s)")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([
        __file__,
        "-v",
        "-m", "performance",
        "--tb=short"
    ])