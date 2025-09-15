"""End-to-end tests for the complete document processing pipeline."""

import pytest
import asyncio
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

# Import main components
from enhanced_main import EnhancedDocumentProcessor, ProcessingConfig
from core.state_manager import StateManager, ProcessingStage
from core.chunking_processor import ChunkingStrategy
from core.output_generator import OutputFormat
from core.config import ConfigManager


class TestEndToEndWorkflow:
    """Complete end-to-end workflow tests."""
    
    @pytest.fixture
    def sample_document(self):
        """Sample document for testing."""
        return """
        # Advanced Machine Learning Systems
        
        ## Introduction
        
        Machine learning has revolutionized how we approach complex problems in computer science.
        This document explores advanced concepts in machine learning systems, including deep learning,
        reinforcement learning, and distributed training architectures.
        
        ## Deep Learning Fundamentals
        
        Deep learning models, particularly neural networks, have shown remarkable success in various
        domains including computer vision, natural language processing, and speech recognition.
        
        ### Neural Network Architectures
        
        1. **Convolutional Neural Networks (CNNs)**: Primarily used for image processing tasks
        2. **Recurrent Neural Networks (RNNs)**: Designed for sequential data processing
        3. **Transformer Networks**: State-of-the-art architecture for language understanding
        
        ## Reinforcement Learning
        
        Reinforcement learning enables agents to learn optimal behaviors through interaction with
        their environment. Key concepts include:
        
        - **Q-Learning**: Value-based learning algorithm
        - **Policy Gradients**: Direct policy optimization methods
        - **Actor-Critic Methods**: Combining value and policy-based approaches
        
        ## Distributed Training
        
        Modern machine learning systems require distributed training to handle large datasets
        and complex models. This involves:
        
        ### Data Parallelism
        
        Distributing training data across multiple devices while replicating the model.
        
        ### Model Parallelism
        
        Splitting the model itself across multiple devices when it's too large for a single device.
        
        ## Applications
        
        Machine learning systems are applied in numerous fields:
        
        - **Healthcare**: Medical image analysis, drug discovery
        - **Finance**: Fraud detection, algorithmic trading
        - **Transportation**: Autonomous vehicles, route optimization
        - **Technology**: Recommendation systems, search engines
        
        ## Conclusion
        
        The field of machine learning continues to evolve rapidly, with new architectures
        and training methodologies emerging regularly. Understanding these advanced concepts
        is crucial for developing effective AI systems.
        """
    
    @pytest.fixture
    def processing_config(self):
        """Standard processing configuration for tests."""
        config = ProcessingConfig()
        # Configure for testing
        config.chunking.strategy = ChunkingStrategy.SEMANTIC
        config.chunking.chunk_size = 512
        config.llm.provider = "mock"
        config.llm.model = "test-model"
        return config
    
    @pytest.fixture
    def mock_llm_responses(self):
        """Mock LLM responses for different processing tasks."""
        return {
            "summarization": "This section discusses {topic} with key points about {concepts}.",
            "analysis": "The analysis reveals important insights about {domain} and {methodology}.",
            "extraction": "Key entities: {entities}. Main concepts: {concepts}.",
            "enhancement": "Enhanced content with improved clarity and structure."
        }
    
    @pytest.mark.asyncio
    async def test_complete_document_processing_workflow(self, sample_document, processing_config, mock_llm_responses):
        """Test the complete document processing workflow from start to finish."""
        processor = EnhancedDocumentProcessor(processing_config)
        
        # Mock LLM calls
        async def mock_llm_call(prompt, **kwargs):
            # Determine task type from prompt
            if "summarize" in prompt.lower():
                return mock_llm_responses["summarization"].format(
                    topic="machine learning",
                    concepts="neural networks, deep learning"
                )
            elif "analyze" in prompt.lower():
                return mock_llm_responses["analysis"].format(
                    domain="artificial intelligence",
                    methodology="supervised learning"
                )
            else:
                return mock_llm_responses["enhancement"]
        
        with patch.object(processor.llm_handler, '_call_llm', side_effect=mock_llm_call):
            # Process the document
            result = await processor.process_document(
                content=sample_document,
                document_id="test_ml_document",
                output_format=OutputFormat.MARKDOWN
            )
            
            # Verify successful processing
            assert result['success'] is True
            assert 'document_id' in result
            assert result['document_id'] == "test_ml_document"
            
            # Verify processing stages completed
            assert 'processing_stages' in result
            stages = result['processing_stages']
            expected_stages = [
                ProcessingStage.DOCUMENT_PARSING,
                ProcessingStage.CHUNKING,
                ProcessingStage.LLM_PROCESSING,
                ProcessingStage.OUTPUT_GENERATION
            ]
            
            for stage in expected_stages:
                assert stage.value in [s['stage'] for s in stages]
            
            # Verify output content
            assert 'output' in result
            assert len(result['output']) > 0
            
            # Verify chunks were created
            assert 'chunks_processed' in result
            assert result['chunks_processed'] > 0
            
            # Verify metrics were collected
            assert 'metrics' in result
            metrics = result['metrics']
            assert 'processing_time' in metrics
            assert 'memory_usage' in metrics
    
    @pytest.mark.asyncio
    async def test_document_processing_with_knowledge_graph(self, sample_document, processing_config):
        """Test document processing with knowledge graph generation."""
        # Enable knowledge graph processing
        processing_config.knowledge_graph.enabled = True
        processing_config.knowledge_graph.extraction_method = "rule_based"
        
        processor = EnhancedDocumentProcessor(processing_config)
        
        # Mock knowledge graph extraction
        mock_entities = [
            {'text': 'Machine Learning', 'label': 'CONCEPT', 'confidence': 0.9},
            {'text': 'Neural Networks', 'label': 'CONCEPT', 'confidence': 0.85},
            {'text': 'Deep Learning', 'label': 'CONCEPT', 'confidence': 0.8}
        ]
        
        mock_relationships = [
            {
                'subject': 'Deep Learning',
                'predicate': 'is_part_of',
                'object': 'Machine Learning',
                'confidence': 0.9
            }
        ]
        
        with patch.object(processor.kg_processor, 'extract_entities', new_callable=AsyncMock) as mock_extract_entities, \
             patch.object(processor.kg_processor, 'extract_relationships', new_callable=AsyncMock) as mock_extract_rels, \
             patch.object(processor.llm_handler, '_call_llm', new_callable=AsyncMock) as mock_llm:
            
            mock_extract_entities.return_value = mock_entities
            mock_extract_rels.return_value = mock_relationships
            mock_llm.return_value = "Processed content"
            
            result = await processor.process_document(
                content=sample_document,
                document_id="test_kg_document"
            )
            
            # Verify knowledge graph was created
            assert result['success'] is True
            assert 'knowledge_graph' in result
            
            kg_data = result['knowledge_graph']
            assert 'entities' in kg_data
            assert 'relationships' in kg_data
            assert len(kg_data['entities']) > 0
    
    @pytest.mark.asyncio
    async def test_multi_format_output_generation(self, sample_document, processing_config):
        """Test generating multiple output formats from the same document."""
        processor = EnhancedDocumentProcessor(processing_config)
        
        formats_to_test = [
            OutputFormat.MARKDOWN,
            OutputFormat.JSON,
            OutputFormat.TEXT
        ]
        
        with patch.object(processor.llm_handler, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Processed content"
            
            results = {}
            
            for output_format in formats_to_test:
                result = await processor.process_document(
                    content=sample_document,
                    document_id=f"test_doc_{output_format.value}",
                    output_format=output_format
                )
                
                results[output_format] = result
                
                # Verify each format was generated successfully
                assert result['success'] is True
                assert 'output' in result
                assert len(result['output']) > 0
            
            # Verify different formats produce different outputs
            markdown_output = results[OutputFormat.MARKDOWN]['output']
            json_output = results[OutputFormat.JSON]['output']
            text_output = results[OutputFormat.TEXT]['output']
            
            # Markdown should contain formatting
            assert '#' in markdown_output or '##' in markdown_output
            
            # JSON should be parseable
            try:
                json.loads(json_output)
                json_valid = True
            except:
                json_valid = False
            assert json_valid
            
            # Text should be plain
            assert '#' not in text_output  # No markdown formatting
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, sample_document, processing_config):
        """Test error recovery during document processing."""
        processor = EnhancedDocumentProcessor(processing_config)
        
        # Configure to fail initially then succeed
        call_count = 0
        
        async def failing_llm_call(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two calls
                raise Exception("Simulated LLM failure")
            return "Recovered and processed successfully"
        
        with patch.object(processor.llm_handler, '_call_llm', side_effect=failing_llm_call):
            result = await processor.process_document(
                content=sample_document,
                document_id="test_recovery_document"
            )
            
            # Should eventually succeed due to retry mechanism
            assert result['success'] is True
            assert 'errors' in result
            assert len(result['errors']) > 0  # Should record the failures
            
            # Verify recovery was attempted
            assert call_count > 2  # Should have retried
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self, processing_config):
        """Test processing of large documents with many sections."""
        # Generate a large document
        large_document = """
        # Large Document Processing Test
        
        ## Executive Summary
        This is a comprehensive test document designed to validate the system's ability
        to handle large documents with multiple sections and complex content.
        """
        
        # Add many sections
        for i in range(20):
            large_document += f"""
            
            ## Section {i+1}: Advanced Topic {i+1}
            
            This section covers advanced topic {i+1} in detail. It includes multiple
            paragraphs of content, technical details, and comprehensive explanations
            that demonstrate the system's ability to process complex information.
            
            ### Subsection {i+1}.1: Technical Details
            
            Technical implementation details for topic {i+1} including algorithms,
            data structures, and performance considerations.
            
            ### Subsection {i+1}.2: Practical Applications
            
            Real-world applications and use cases for the concepts discussed in
            this section, with examples and case studies.
            """
        
        processor = EnhancedDocumentProcessor(processing_config)
        
        with patch.object(processor.llm_handler, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Processed large section content"
            
            result = await processor.process_document(
                content=large_document,
                document_id="test_large_document"
            )
            
            # Verify successful processing of large document
            assert result['success'] is True
            assert result['chunks_processed'] > 10  # Should create many chunks
            
            # Verify reasonable processing time (should complete)
            assert 'metrics' in result
            assert result['metrics']['processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, processing_config):
        """Test processing multiple documents concurrently."""
        documents = {
            "doc1": "# Document 1\n\nThis is the first test document with content about AI.",
            "doc2": "# Document 2\n\nThis is the second test document with content about ML.",
            "doc3": "# Document 3\n\nThis is the third test document with content about DL."
        }
        
        processor = EnhancedDocumentProcessor(processing_config)
        
        with patch.object(processor.llm_handler, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Processed concurrent document"
            
            # Process all documents concurrently
            tasks = [
                processor.process_document(
                    content=content,
                    document_id=doc_id
                )
                for doc_id, content in documents.items()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all documents were processed successfully
            assert len(results) == 3
            
            successful_results = [
                r for r in results 
                if isinstance(r, dict) and r.get('success')
            ]
            
            assert len(successful_results) == 3
            
            # Verify each document has unique ID
            doc_ids = [r['document_id'] for r in successful_results]
            assert len(set(doc_ids)) == 3  # All unique
    
    def test_configuration_validation_workflow(self):
        """Test configuration validation in the complete workflow."""
        # Test with invalid configuration
        invalid_config = ProcessingConfig()
        invalid_config.chunking.chunk_size = -1  # Invalid
        
        with pytest.raises((ValueError, AssertionError)):
            processor = EnhancedDocumentProcessor(invalid_config)
    
    @pytest.mark.asyncio
    async def test_state_persistence_workflow(self, sample_document, processing_config):
        """Test state persistence during document processing."""
        processor = EnhancedDocumentProcessor(processing_config)
        
        # Mock state saving
        saved_states = []
        
        def mock_save_state(state_dict):
            saved_states.append(state_dict.copy())
        
        with patch.object(processor.llm_handler, '_call_llm', new_callable=AsyncMock) as mock_llm, \
             patch.object(processor.state_manager, 'save_state', side_effect=mock_save_state):
            
            mock_llm.return_value = "Processed with state persistence"
            
            result = await processor.process_document(
                content=sample_document,
                document_id="test_state_persistence"
            )
            
            # Verify processing succeeded
            assert result['success'] is True
            
            # Verify state was saved at various points
            assert len(saved_states) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_collection_workflow(self, sample_document, processing_config):
        """Test comprehensive metrics collection during processing."""
        processor = EnhancedDocumentProcessor(processing_config)
        
        with patch.object(processor.llm_handler, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Processed with metrics collection"
            
            result = await processor.process_document(
                content=sample_document,
                document_id="test_metrics_collection"
            )
            
            # Verify comprehensive metrics were collected
            assert result['success'] is True
            assert 'metrics' in result
            
            metrics = result['metrics']
            expected_metrics = [
                'processing_time',
                'memory_usage',
                'chunks_created',
                'chunks_processed',
                'llm_calls_made',
                'errors_encountered'
            ]
            
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], (int, float))


class TestRealWorldScenarios:
    """Tests simulating real-world usage scenarios."""
    
    @pytest.fixture
    def research_paper_content(self):
        """Simulate a research paper document."""
        return """
        # Quantum Computing Applications in Machine Learning
        
        ## Abstract
        
        This paper explores the intersection of quantum computing and machine learning,
        investigating how quantum algorithms can enhance traditional ML approaches.
        We present novel quantum-enhanced algorithms and demonstrate their effectiveness
        on benchmark datasets.
        
        ## 1. Introduction
        
        Quantum computing represents a paradigm shift in computational capabilities,
        offering exponential speedups for certain classes of problems. Recent advances
        in quantum hardware and algorithms have opened new possibilities for machine
        learning applications.
        
        ## 2. Background
        
        ### 2.1 Quantum Computing Fundamentals
        
        Quantum computers leverage quantum mechanical phenomena such as superposition
        and entanglement to process information in ways impossible for classical computers.
        
        ### 2.2 Machine Learning Challenges
        
        Traditional machine learning faces scalability challenges with large datasets
        and high-dimensional feature spaces. Quantum algorithms may offer solutions
        to these computational bottlenecks.
        
        ## 3. Methodology
        
        We developed quantum-enhanced versions of popular ML algorithms:
        
        1. **Quantum Support Vector Machines**: Utilizing quantum feature maps
        2. **Quantum Neural Networks**: Implementing quantum gates as neurons
        3. **Quantum Clustering**: Leveraging quantum interference patterns
        
        ## 4. Experimental Results
        
        Our experiments on benchmark datasets show promising results:
        
        - 40% improvement in classification accuracy on MNIST dataset
        - 60% reduction in training time for quantum SVMs
        - Novel clustering patterns discovered through quantum interference
        
        ## 5. Discussion
        
        The results demonstrate the potential of quantum-enhanced machine learning.
        However, current quantum hardware limitations restrict practical applications
        to small-scale problems.
        
        ## 6. Conclusion
        
        Quantum computing offers exciting possibilities for machine learning advancement.
        As quantum hardware continues to improve, we expect broader adoption of these
        hybrid quantum-classical approaches.
        
        ## References
        
        [1] Nielsen, M. A., & Chuang, I. L. (2010). Quantum computation and quantum information.
        [2] Biamonte, J., et al. (2017). Quantum machine learning. Nature, 549(7671), 195-202.
        [3] Schuld, M., & Petruccione, F. (2018). Supervised learning with quantum computers.
        """
    
    @pytest.fixture
    def technical_manual_content(self):
        """Simulate a technical manual document."""
        return """
        # Advanced Database Management System - User Manual
        
        ## Table of Contents
        
        1. Installation and Setup
        2. Basic Operations
        3. Advanced Queries
        4. Performance Optimization
        5. Troubleshooting
        
        ## 1. Installation and Setup
        
        ### 1.1 System Requirements
        
        - Operating System: Linux, Windows, or macOS
        - Memory: Minimum 8GB RAM, Recommended 16GB+
        - Storage: 100GB available space
        - Network: High-speed internet connection
        
        ### 1.2 Installation Steps
        
        1. Download the installer from the official website
        2. Run the installer with administrator privileges
        3. Follow the setup wizard instructions
        4. Configure initial database settings
        5. Verify installation with test queries
        
        ## 2. Basic Operations
        
        ### 2.1 Creating Databases
        
        ```sql
        CREATE DATABASE company_db;
        USE company_db;
        ```
        
        ### 2.2 Table Management
        
        Create tables with appropriate data types and constraints:
        
        ```sql
        CREATE TABLE employees (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(100) NOT NULL,
            department VARCHAR(50),
            salary DECIMAL(10,2),
            hire_date DATE
        );
        ```
        
        ## 3. Advanced Queries
        
        ### 3.1 Complex Joins
        
        Perform multi-table joins for comprehensive data analysis:
        
        ```sql
        SELECT e.name, d.department_name, p.project_name
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        JOIN project_assignments pa ON e.id = pa.employee_id
        JOIN projects p ON pa.project_id = p.id;
        ```
        
        ### 3.2 Window Functions
        
        Use window functions for advanced analytics:
        
        ```sql
        SELECT name, salary,
               RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank
        FROM employees;
        ```
        
        ## 4. Performance Optimization
        
        ### 4.1 Index Management
        
        Create appropriate indexes to improve query performance:
        
        - Primary indexes on frequently queried columns
        - Composite indexes for multi-column queries
        - Partial indexes for filtered queries
        
        ### 4.2 Query Optimization
        
        Best practices for writing efficient queries:
        
        - Use EXPLAIN to analyze query execution plans
        - Avoid SELECT * in production queries
        - Use appropriate WHERE clause filtering
        - Consider query result caching
        
        ## 5. Troubleshooting
        
        ### 5.1 Common Issues
        
        **Connection Timeouts**
        - Check network connectivity
        - Verify firewall settings
        - Increase connection timeout values
        
        **Performance Issues**
        - Monitor system resources
        - Analyze slow query logs
        - Review index usage statistics
        
        **Data Corruption**
        - Run database integrity checks
        - Restore from recent backups
        - Contact technical support if needed
        """
    
    @pytest.mark.asyncio
    async def test_research_paper_processing(self, research_paper_content):
        """Test processing a research paper document."""
        config = ProcessingConfig()
        config.chunking.strategy = ChunkingStrategy.SEMANTIC
        config.knowledge_graph.enabled = True
        
        processor = EnhancedDocumentProcessor(config)
        
        # Mock academic-focused processing
        async def academic_llm_call(prompt, **kwargs):
            if "abstract" in prompt.lower():
                return "This research explores quantum-enhanced machine learning algorithms."
            elif "methodology" in prompt.lower():
                return "The methodology involves quantum algorithm development and benchmarking."
            elif "results" in prompt.lower():
                return "Results show significant improvements in accuracy and training time."
            else:
                return "Academic content processed with domain expertise."
        
        with patch.object(processor.llm_handler, '_call_llm', side_effect=academic_llm_call), \
             patch.object(processor.kg_processor, 'extract_entities', new_callable=AsyncMock) as mock_entities:
            
            # Mock academic entities
            mock_entities.return_value = [
                {'text': 'Quantum Computing', 'label': 'CONCEPT', 'confidence': 0.95},
                {'text': 'Machine Learning', 'label': 'CONCEPT', 'confidence': 0.9},
                {'text': 'MNIST dataset', 'label': 'DATASET', 'confidence': 0.85}
            ]
            
            result = await processor.process_document(
                content=research_paper_content,
                document_id="quantum_ml_paper",
                output_format=OutputFormat.MARKDOWN
            )
            
            # Verify academic document processing
            assert result['success'] is True
            assert 'knowledge_graph' in result
            
            # Should identify academic sections
            output = result['output']
            assert 'Abstract' in output or 'abstract' in output.lower()
            assert 'Methodology' in output or 'methodology' in output.lower()
    
    @pytest.mark.asyncio
    async def test_technical_manual_processing(self, technical_manual_content):
        """Test processing a technical manual document."""
        config = ProcessingConfig()
        config.chunking.strategy = ChunkingStrategy.HIERARCHICAL
        
        processor = EnhancedDocumentProcessor(config)
        
        # Mock technical documentation processing
        async def technical_llm_call(prompt, **kwargs):
            if "installation" in prompt.lower():
                return "Installation section with system requirements and setup steps."
            elif "sql" in prompt.lower() or "query" in prompt.lower():
                return "Database query examples and SQL code snippets."
            elif "troubleshooting" in prompt.lower():
                return "Troubleshooting guide with common issues and solutions."
            else:
                return "Technical documentation content processed."
        
        with patch.object(processor.llm_handler, '_call_llm', side_effect=technical_llm_call):
            result = await processor.process_document(
                content=technical_manual_content,
                document_id="database_manual",
                output_format=OutputFormat.MARKDOWN
            )
            
            # Verify technical manual processing
            assert result['success'] is True
            
            # Should preserve code blocks and technical structure
            output = result['output']
            assert 'Installation' in output
            assert 'SQL' in output or 'sql' in output
            assert 'Troubleshooting' in output
    
    @pytest.mark.asyncio
    async def test_mixed_content_processing(self):
        """Test processing document with mixed content types."""
        mixed_content = """
        # Comprehensive AI System Documentation
        
        ## Executive Summary
        This document provides a comprehensive overview of our AI system implementation.
        
        ## Technical Architecture
        
        ### System Components
        - Data Processing Pipeline
        - Machine Learning Models
        - API Gateway
        - Database Layer
        
        ### Code Example
        ```python
        class AIProcessor:
            def __init__(self, config):
                self.config = config
                self.model = load_model(config.model_path)
            
            def process(self, data):
                return self.model.predict(data)
        ```
        
        ## Performance Metrics
        
        | Metric | Value | Target |
        |--------|-------|--------|
        | Accuracy | 94.5% | >90% |
        | Latency | 150ms | <200ms |
        | Throughput | 1000 req/s | >500 req/s |
        
        ## Research Background
        
        Our approach is based on recent advances in transformer architectures
        and attention mechanisms. Key references include:
        
        1. Vaswani et al. (2017) - Attention Is All You Need
        2. Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers
        
        ## Deployment Guide
        
        ### Prerequisites
        - Docker 20.10+
        - Kubernetes 1.20+
        - GPU support (CUDA 11.0+)
        
        ### Deployment Steps
        1. Build Docker image
        2. Deploy to Kubernetes cluster
        3. Configure load balancer
        4. Set up monitoring
        """
        
        config = ProcessingConfig()
        processor = EnhancedDocumentProcessor(config)
        
        with patch.object(processor.llm_handler, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Mixed content processed successfully"
            
            result = await processor.process_document(
                content=mixed_content,
                document_id="mixed_content_doc",
                output_format=OutputFormat.MARKDOWN
            )
            
            # Verify mixed content processing
            assert result['success'] is True
            
            # Should handle different content types
            output = result['output']
            assert len(output) > 0
            assert result['chunks_processed'] > 5  # Multiple sections


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])