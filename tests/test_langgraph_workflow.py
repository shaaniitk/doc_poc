"""Integration tests for LangGraph workflow orchestration"""
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from datetime import datetime
import json
from typing import Dict, List, Any

# Import LangGraph components
from langgraph_state import (
    ProcessingStage, ErrorSeverity, ErrorInfo, DocumentInfo,
    PipelineState, create_initial_state
)
from langgraph_workflow import DocumentProcessingWorkflow
from langgraph_nodes import DocumentProcessingNodes


class TestDocumentProcessingWorkflow(unittest.TestCase):
    """Test LangGraph workflow orchestrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DocumentProcessingWorkflow({})
        self.test_config = {
            'session_id': 'test-session-123',
            'file_path': '/test/document.pdf',
            'output_dir': '/test/output',
            'use_enhanced_processing': True
        }
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsNotNone(self.orchestrator)
        # Test that LangGraph is available (mocked in conftest.py if needed)
        self.assertTrue(hasattr(self.orchestrator, 'create_workflow'))
    
    @patch('langgraph_workflow.StateGraph')
    def test_create_workflow(self, mock_state_graph):
        """Test workflow creation with mocked LangGraph"""
        mock_graph = Mock()
        mock_state_graph.return_value = mock_graph
        
        workflow = self.orchestrator.create_workflow()
        
        # Verify graph was created and configured
        mock_state_graph.assert_called_once()
        self.assertIsNotNone(workflow)
    
    @patch('langgraph_workflow.StateGraph')
    def test_workflow_node_registration(self, mock_state_graph):
        """Test that all workflow nodes are registered"""
        mock_graph = Mock()
        mock_state_graph.return_value = mock_graph
        
        self.orchestrator.create_workflow()
        
        # Verify add_node was called for each processing stage
        expected_nodes = [
            'initialize', 'parse_document', 'chunk_document',
            'create_knowledge_graph', 'semantic_mapping', 'llm_processing',
            'generate_output', 'collect_analytics', 'complete_pipeline'
        ]
        
        self.assertEqual(mock_graph.add_node.call_count, len(expected_nodes))
    
    @patch('langgraph_workflow.StateGraph')
    def test_workflow_edge_configuration(self, mock_state_graph):
        """Test workflow edge configuration"""
        mock_graph = Mock()
        mock_state_graph.return_value = mock_graph
        
        self.orchestrator.create_workflow()
        
        # Verify edges were added (at least some basic ones)
        self.assertGreater(mock_graph.add_edge.call_count, 0)
    
    def test_fallback_linear_processing(self):
        """Test fallback to linear processing when LangGraph unavailable"""
        # Mock LangGraph as unavailable
        with patch('langgraph_workflow.StateGraph', side_effect=ImportError("LangGraph not available")):
            result = self.orchestrator.run_pipeline(self.test_config)
            
            # Should fall back to linear processing
            self.assertIsInstance(result, dict)
            self.assertIn('status', result)
            self.assertEqual(result['status'], 'completed_linear')
    
    @patch('langgraph_workflow.StateGraph')
    def test_run_pipeline_success(self, mock_state_graph):
        """Test successful pipeline execution"""
        # Mock the compiled workflow
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = {
            'session_id': 'test-session-123',
            'current_stage': ProcessingStage.COMPLETED,
            'errors': [],
            'output_info': {'status': 'success'}
        }
        
        mock_graph = Mock()
        mock_graph.compile.return_value = mock_workflow
        mock_state_graph.return_value = mock_graph
        
        result = self.orchestrator.run_pipeline(self.test_config)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        mock_workflow.invoke.assert_called_once()
    
    @patch('langgraph_workflow.StateGraph')
    def test_run_pipeline_with_errors(self, mock_state_graph):
        """Test pipeline execution with errors"""
        # Mock workflow that returns errors
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = {
            'session_id': 'test-session-123',
            'current_stage': ProcessingStage.ERROR,
            'errors': [
                {
                    'message': 'Test error',
                    'severity': ErrorSeverity.HIGH,
                    'stage': ProcessingStage.CHUNKING
                }
            ]
        }
        
        mock_graph = Mock()
        mock_graph.compile.return_value = mock_workflow
        mock_state_graph.return_value = mock_graph
        
        result = self.orchestrator.run_pipeline(self.test_config)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)


class TestLangGraphNodes(unittest.TestCase):
    """Test individual LangGraph workflow nodes"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.initial_state = create_initial_state(
            session_id='test-session',
            file_path='/test/document.pdf'
        )
        self.nodes = DocumentProcessingNodes({})
    
    async def test_initialize_pipeline_node(self):
        """Test pipeline initialization node"""
        result = await self.nodes.initialization_node(self.initial_state)
        
        self.assertIsInstance(result, PipelineState)
        self.assertEqual(result.current_stage, ProcessingStage.INITIALIZATION)
        self.assertIsNotNone(result.document_info)
    
    async def test_parse_document_node(self):
        """Test document parsing node"""
        # Set up state for document parsing
        self.initial_state.current_stage = ProcessingStage.DOCUMENT_PARSING
        self.initial_state.raw_content = "Test document content for parsing"
        
        result = await self.nodes.document_parsing_node(self.initial_state)
        
        self.assertIsInstance(result, PipelineState)
        self.assertEqual(result.current_stage, ProcessingStage.DOCUMENT_PARSING)
        self.assertIn('preprocessed_content', result.intermediate_results)
    
    async def test_chunk_document_node(self):
        """Test document chunking node"""
        # Set up state for chunking
        self.initial_state.current_stage = ProcessingStage.CHUNKING
        self.initial_state.raw_content = "This is a long document content that needs to be chunked into smaller pieces for processing."
        self.initial_state.intermediate_results['preprocessed_content'] = self.initial_state.raw_content
        
        result = await self.nodes.chunking_node(self.initial_state)
        
        self.assertIsInstance(result, PipelineState)
        self.assertEqual(result.current_stage, ProcessingStage.CHUNKING)
        self.assertGreater(len(result.chunks), 0)
    
    async def test_create_knowledge_graph_node(self):
        """Test knowledge graph creation node"""
        # Set up state for knowledge graph creation
        self.initial_state.current_stage = ProcessingStage.KNOWLEDGE_GRAPH
        # Add some chunks to work with
        from langgraph_state import ChunkInfo
        chunk1 = ChunkInfo(
            chunk_id="chunk_0001",
            content="This is the first chunk with entities like John and Mary.",
            start_position=0,
            end_position=50,
            chunk_size=50
        )
        self.initial_state.add_chunk(chunk1)
        
        result = await self.nodes.knowledge_graph_node(self.initial_state)
        
        self.assertIsInstance(result, PipelineState)
        self.assertEqual(result.current_stage, ProcessingStage.KNOWLEDGE_GRAPH)
    
    async def test_semantic_mapping_node(self):
        """Test semantic mapping node"""
        # Set up state for semantic mapping
        self.initial_state.current_stage = ProcessingStage.SEMANTIC_MAPPING
        # Add some knowledge graph data
        from langgraph_state import KnowledgeGraphNode
        node1 = KnowledgeGraphNode(
            node_id="node_001",
            entity_type="PERSON",
            entity_name="John",
            properties={"age": "30"}
        )
        self.initial_state.add_kg_node(node1)
        
        result = await self.nodes.semantic_mapping_node(self.initial_state)
        
        self.assertIsInstance(result, PipelineState)
        self.assertEqual(result.current_stage, ProcessingStage.SEMANTIC_MAPPING)
    
    async def test_llm_processing_node(self):
        """Test LLM processing node"""
        # Set up state for LLM processing
        self.initial_state.current_stage = ProcessingStage.LLM_PROCESSING
        # Add some semantic mapping data
        self.initial_state.intermediate_results['semantic_vectors'] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        result = await self.nodes.llm_processing_node(self.initial_state)
        
        self.assertIsInstance(result, PipelineState)
        self.assertEqual(result.current_stage, ProcessingStage.LLM_PROCESSING)
    
    async def test_generate_output_node(self):
        """Test output generation node"""
        # Set up state for output generation
        self.initial_state.current_stage = ProcessingStage.OUTPUT_GENERATION
        # Add some LLM processing results
        self.initial_state.intermediate_results['llm_analysis'] = {
            'summary': 'Document analysis complete',
            'key_points': ['Point 1', 'Point 2']
        }
        
        result = await self.nodes.output_generation_node(self.initial_state)
        
        self.assertIsInstance(result, PipelineState)
        self.assertEqual(result.current_stage, ProcessingStage.OUTPUT_GENERATION)
    
    async def test_collect_analytics_node(self):
        """Test analytics collection node"""
        # Set up state for analytics
        self.initial_state.current_stage = ProcessingStage.ANALYTICS
        # Add some output generation results
        self.initial_state.intermediate_results['final_output'] = {
            'content': 'Generated final output',
            'format': 'markdown'
        }
        
        result = await self.nodes.analytics_node(self.initial_state)
        
        self.assertIsInstance(result, PipelineState)
        self.assertEqual(result.current_stage, ProcessingStage.ANALYTICS)
    
    def test_complete_pipeline_node(self):
        """Test pipeline completion node"""
        state = self.initial_state.to_dict()
        state['current_stage'] = ProcessingStage.COMPLETED
        state['analytics_info'] = {'processing_metrics': {}}
        
        result = complete_pipeline(state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        self.assertIn('completion_time', result)
        self.assertIn('final_status', result)
        self.assertEqual(result['final_status'], 'success')


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for complete workflow scenarios"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.orchestrator = DocumentProcessingWorkflow({})
        self.config = {
            'session_id': 'integration-test-session',
            'file_path': '/test/integration-doc.pdf',
            'output_dir': '/test/integration-output',
            'use_enhanced_processing': True,
            'enable_analytics': True
        }
    
    @patch('langgraph_workflow.StateGraph')
    def test_complete_workflow_integration(self, mock_state_graph):
        """Test complete workflow from start to finish"""
        # Mock a complete successful workflow execution
        final_state = {
            'session_id': 'integration-test-session',
            'current_stage': ProcessingStage.COMPLETED,
            'document_info': {
                'file_path': '/test/integration-doc.pdf',
                'file_type': 'pdf',
                'size_bytes': 2048
            },
            'chunk_info': {
                'total_chunks': 5,
                'processed_chunks': 5
            },
            'knowledge_graph_info': {
                'node_count': 15,
                'edge_count': 20
            },
            'semantic_mapping_info': {
                'mapping_count': 12,
                'average_similarity': 0.78
            },
            'llm_processing_info': {
                'tokens_processed': 1500,
                'processing_time': 8.5
            },
            'output_info': {
                'output_files': ['/test/integration-output/result.json'],
                'formats': ['json']
            },
            'analytics_info': {
                'processing_metrics': {
                    'total_time': 25.3,
                    'memory_used': '512MB'
                },
                'quality_metrics': {
                    'coherence_score': 0.89,
                    'completeness_score': 0.95
                }
            },
            'errors': [],
            'completion_time': datetime.now().isoformat(),
            'final_status': 'success'
        }
        
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = final_state
        
        mock_graph = Mock()
        mock_graph.compile.return_value = mock_workflow
        mock_state_graph.return_value = mock_graph
        
        result = self.orchestrator.run_pipeline(self.config)
        
        # Verify complete workflow execution
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        self.assertEqual(result['final_status'], 'success')
        self.assertEqual(len(result['errors']), 0)
        
        # Verify all processing stages completed
        self.assertIn('chunk_info', result)
        self.assertIn('knowledge_graph_info', result)
        self.assertIn('semantic_mapping_info', result)
        self.assertIn('llm_processing_info', result)
        self.assertIn('output_info', result)
        self.assertIn('analytics_info', result)
    
    @patch('langgraph_workflow.StateGraph')
    def test_workflow_error_recovery(self, mock_state_graph):
        """Test workflow error handling and recovery"""
        # Mock workflow with recoverable error
        error_state = {
            'session_id': 'integration-test-session',
            'current_stage': ProcessingStage.COMPLETED,  # Recovered
            'errors': [
                {
                    'message': 'Temporary LLM timeout',
                    'severity': ErrorSeverity.MEDIUM,
                    'stage': ProcessingStage.LLM_PROCESSING,
                    'timestamp': datetime.now().isoformat()
                }
            ],
            'final_status': 'success_with_warnings'
        }
        
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = error_state
        
        mock_graph = Mock()
        mock_graph.compile.return_value = mock_workflow
        mock_state_graph.return_value = mock_graph
        
        result = self.orchestrator.run_pipeline(self.config)
        
        # Verify error was handled and workflow completed
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        self.assertEqual(result['final_status'], 'success_with_warnings')
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(result['errors'][0]['severity'], ErrorSeverity.MEDIUM)
    
    def test_workflow_performance_metrics(self):
        """Test workflow performance tracking"""
        # This test would measure actual performance in a real scenario
        # For now, we'll test the structure of performance data
        
        with patch('langgraph_workflow.StateGraph') as mock_state_graph:
            # Mock workflow with performance metrics
            perf_state = {
                'session_id': 'perf-test-session',
                'current_stage': ProcessingStage.COMPLETED,
                'analytics_info': {
                    'processing_metrics': {
                        'total_time': 18.7,
                        'memory_used': '384MB',
                        'cpu_usage': '45%',
                        'stage_timings': {
                            'document_parsing': 2.1,
                            'chunking': 1.5,
                            'knowledge_graph': 4.2,
                            'semantic_mapping': 3.8,
                            'llm_processing': 5.9,
                            'output_generation': 1.2
                        }
                    }
                },
                'errors': []
            }
            
            mock_workflow = Mock()
            mock_workflow.invoke.return_value = perf_state
            
            mock_graph = Mock()
            mock_graph.compile.return_value = mock_workflow
            mock_state_graph.return_value = mock_graph
            
            result = self.orchestrator.run_pipeline(self.config)
            
            # Verify performance metrics are captured
            self.assertIn('analytics_info', result)
            self.assertIn('processing_metrics', result['analytics_info'])
            
            metrics = result['analytics_info']['processing_metrics']
            self.assertIn('total_time', metrics)
            self.assertIn('memory_used', metrics)
            self.assertIn('stage_timings', metrics)
            
            # Verify stage timings are reasonable
            stage_timings = metrics['stage_timings']
            self.assertGreater(len(stage_timings), 0)
            for stage, timing in stage_timings.items():
                self.assertGreater(timing, 0.0)


if __name__ == '__main__':
    unittest.main()