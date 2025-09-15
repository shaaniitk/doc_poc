"""Integration tests for LangGraph workflow orchestration"""
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# Import LangGraph components
from langgraph_state import (
    ProcessingStage, ErrorSeverity, ErrorInfo, DocumentInfo,
    ChunkInfo, KnowledgeGraphNode, KnowledgeGraphEdge, SemanticMapping,
    LLMProcessingResult, OutputData, AnalyticsData,
    PipelineState, create_initial_state
)
from langgraph_workflow import DocumentProcessingWorkflow
from langgraph_nodes import DocumentProcessingNodes


class TestLangGraphWorkflowIntegration(unittest.TestCase):
    """Integration tests for complete LangGraph workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LangGraphWorkflowOrchestrator()
        self.test_file_path = '/test/sample_document.pdf'
        self.session_id = 'test-integration-session'
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock configuration
        self.config = {
            'output_dir': self.temp_dir,
            'output_formats': ['json', 'markdown'],
            'use_enhanced_processing': True,
            'chunk_size': 1000,
            'overlap_size': 200,
            'llm_model': 'gpt-4',
            'max_tokens': 2000
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('langgraph_nodes.DocumentParser')
    @patch('langgraph_nodes.DocumentChunker')
    @patch('langgraph_nodes.KnowledgeGraphBuilder')
    @patch('langgraph_nodes.SemanticMapper')
    @patch('langgraph_nodes.LLMProcessor')
    @patch('langgraph_nodes.OutputGenerator')
    @patch('langgraph_nodes.AnalyticsCollector')
    def test_complete_workflow_success(self, mock_analytics, mock_output, mock_llm, 
                                     mock_semantic, mock_kg, mock_chunker, mock_parser):
        """Test complete successful workflow execution"""
        # Mock all components
        self._setup_successful_mocks(
            mock_parser, mock_chunker, mock_kg, mock_semantic, 
            mock_llm, mock_output, mock_analytics
        )
        
        # Create initial state
        initial_state = create_initial_state(
            session_id=self.session_id,
            file_path=self.test_file_path
        )
        initial_state.config = self.config
        
        # Run workflow
        result = self.orchestrator.run_workflow(initial_state.to_dict())
        
        # Verify final state
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        self.assertEqual(result['final_status'], 'success')
        self.assertIn('completion_time', result)
        
        # Verify all stages were processed
        self.assertIn('document_info', result)
        self.assertIn('chunk_info', result)
        self.assertIn('knowledge_graph_info', result)
        self.assertIn('semantic_mapping_info', result)
        self.assertIn('llm_processing_info', result)
        self.assertIn('output_info', result)
        self.assertIn('analytics_info', result)
        
        # Verify processing metrics
        self.assertGreater(result['analytics_info']['processing_metrics']['total_time'], 0)
        self.assertIn('stage_timings', result['analytics_info']['processing_metrics'])
    
    @patch('langgraph_nodes.DocumentParser')
    def test_workflow_early_failure(self, mock_parser):
        """Test workflow handling of early stage failure"""
        # Mock parser to fail
        mock_parser_instance = Mock()
        mock_parser_instance.parse.side_effect = Exception("Document parsing failed")
        mock_parser.return_value = mock_parser_instance
        
        # Create initial state
        initial_state = create_initial_state(
            session_id=self.session_id,
            file_path=self.test_file_path
        )
        
        # Run workflow
        result = self.orchestrator.run_workflow(initial_state.to_dict())
        
        # Verify error handling
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('Document parsing failed', result['errors'][0]['message'])
        self.assertEqual(result['errors'][0]['severity'], ErrorSeverity.CRITICAL)
    
    @patch('langgraph_nodes.DocumentParser')
    @patch('langgraph_nodes.DocumentChunker')
    @patch('langgraph_nodes.KnowledgeGraphBuilder')
    def test_workflow_mid_stage_failure(self, mock_kg, mock_chunker, mock_parser):
        """Test workflow handling of mid-stage failure"""
        # Mock successful early stages
        self._setup_parser_mock(mock_parser)
        self._setup_chunker_mock(mock_chunker)
        
        # Mock KG builder to fail
        mock_kg_instance = Mock()
        mock_kg_instance.build.side_effect = Exception("Knowledge graph creation failed")
        mock_kg.return_value = mock_kg_instance
        
        # Create initial state
        initial_state = create_initial_state(
            session_id=self.session_id,
            file_path=self.test_file_path
        )
        
        # Run workflow
        result = self.orchestrator.run_workflow(initial_state.to_dict())
        
        # Verify error handling
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('Knowledge graph creation failed', result['errors'][0]['message'])
        
        # Verify partial processing occurred
        self.assertIn('document_info', result)
        self.assertIn('chunk_info', result)
        # But not later stages
        self.assertNotIn('knowledge_graph_info', result)
    
    @patch('langgraph_nodes.DocumentParser')
    @patch('langgraph_nodes.DocumentChunker')
    @patch('langgraph_nodes.KnowledgeGraphBuilder')
    @patch('langgraph_nodes.SemanticMapper')
    @patch('langgraph_nodes.LLMProcessor')
    @patch('langgraph_nodes.OutputGenerator')
    @patch('langgraph_nodes.AnalyticsCollector')
    def test_workflow_with_warnings(self, mock_analytics, mock_output, mock_llm, 
                                  mock_semantic, mock_kg, mock_chunker, mock_parser):
        """Test workflow execution with non-critical warnings"""
        # Mock all components with some warnings
        self._setup_successful_mocks(
            mock_parser, mock_chunker, mock_kg, mock_semantic, 
            mock_llm, mock_output, mock_analytics
        )
        
        # Create initial state with a warning
        initial_state = create_initial_state(
            session_id=self.session_id,
            file_path=self.test_file_path
        )
        initial_state.add_error(
            "Minor processing warning",
            ErrorSeverity.LOW,
            ProcessingStage.CHUNKING
        )
        
        # Run workflow
        result = self.orchestrator.run_workflow(initial_state.to_dict())
        
        # Verify successful completion with warnings
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        self.assertEqual(result['final_status'], 'success_with_warnings')
        self.assertEqual(result['warning_count'], 1)
        self.assertGreater(len(result['errors']), 0)
        self.assertEqual(result['errors'][0]['severity'], ErrorSeverity.LOW)
    
    def test_workflow_state_transitions(self):
        """Test proper state transitions throughout workflow"""
        # Create initial state
        initial_state = create_initial_state(
            session_id=self.session_id,
            file_path=self.test_file_path
        )
        
        # Test state progression
        expected_stages = [
            ProcessingStage.INITIALIZATION,
            ProcessingStage.DOCUMENT_PARSING,
            ProcessingStage.CHUNKING,
            ProcessingStage.KNOWLEDGE_GRAPH,
            ProcessingStage.SEMANTIC_MAPPING,
            ProcessingStage.LLM_PROCESSING,
            ProcessingStage.OUTPUT_GENERATION,
            ProcessingStage.ANALYTICS,
            ProcessingStage.COMPLETED
        ]
        
        # Verify initial state
        self.assertEqual(initial_state.current_stage, ProcessingStage.INITIALIZATION)
        
        # Test stage progression logic
        current_stage = ProcessingStage.INITIALIZATION
        for expected_next in expected_stages[1:]:
            from langgraph_nodes import route_to_next_stage
            next_stage = route_to_next_stage(current_stage)
            self.assertEqual(next_stage, expected_next)
            current_stage = next_stage
    
    @patch('langgraph_nodes.DocumentParser')
    @patch('langgraph_nodes.DocumentChunker')
    @patch('langgraph_nodes.KnowledgeGraphBuilder')
    @patch('langgraph_nodes.SemanticMapper')
    @patch('langgraph_nodes.LLMProcessor')
    @patch('langgraph_nodes.OutputGenerator')
    @patch('langgraph_nodes.AnalyticsCollector')
    def test_workflow_performance_metrics(self, mock_analytics, mock_output, mock_llm, 
                                        mock_semantic, mock_kg, mock_chunker, mock_parser):
        """Test workflow performance metrics collection"""
        # Mock all components
        self._setup_successful_mocks(
            mock_parser, mock_chunker, mock_kg, mock_semantic, 
            mock_llm, mock_output, mock_analytics
        )
        
        # Create initial state
        initial_state = create_initial_state(
            session_id=self.session_id,
            file_path=self.test_file_path
        )
        
        # Run workflow
        result = self.orchestrator.run_workflow(initial_state.to_dict())
        
        # Verify performance metrics
        analytics = result['analytics_info']
        self.assertIn('processing_metrics', analytics)
        
        processing_metrics = analytics['processing_metrics']
        self.assertIn('total_time', processing_metrics)
        self.assertIn('memory_used', processing_metrics)
        self.assertIn('cpu_usage', processing_metrics)
        self.assertIn('stage_timings', processing_metrics)
        
        # Verify stage timings
        stage_timings = processing_metrics['stage_timings']
        expected_stages = [
            'document_parsing', 'chunking', 'knowledge_graph',
            'semantic_mapping', 'llm_processing', 'output_generation'
        ]
        for stage in expected_stages:
            self.assertIn(stage, stage_timings)
            self.assertGreater(stage_timings[stage], 0)
    
    def test_workflow_configuration_handling(self):
        """Test workflow configuration parameter handling"""
        # Create initial state with custom config
        initial_state = create_initial_state(
            session_id=self.session_id,
            file_path=self.test_file_path
        )
        
        custom_config = {
            'chunk_size': 2000,
            'overlap_size': 400,
            'llm_model': 'gpt-3.5-turbo',
            'max_tokens': 1500,
            'use_enhanced_processing': False,
            'output_formats': ['json']
        }
        initial_state.config = custom_config
        
        # Verify configuration is preserved
        state_dict = initial_state.to_dict()
        self.assertEqual(state_dict['config']['chunk_size'], 2000)
        self.assertEqual(state_dict['config']['llm_model'], 'gpt-3.5-turbo')
        self.assertFalse(state_dict['config']['use_enhanced_processing'])
    
    def test_workflow_session_management(self):
        """Test workflow session management"""
        # Create multiple sessions
        session1 = create_initial_state(
            session_id='session-1',
            file_path='/test/doc1.pdf'
        )
        session2 = create_initial_state(
            session_id='session-2',
            file_path='/test/doc2.pdf'
        )
        
        # Verify session isolation
        self.assertNotEqual(session1.session_id, session2.session_id)
        self.assertNotEqual(session1.document_info.file_path, session2.document_info.file_path)
        
        # Verify session state independence
        session1.add_error("Error in session 1", ErrorSeverity.MEDIUM, ProcessingStage.CHUNKING)
        self.assertEqual(len(session1.errors), 1)
        self.assertEqual(len(session2.errors), 0)
    
    def test_workflow_error_recovery(self):
        """Test workflow error recovery mechanisms"""
        # Create state with recoverable error
        initial_state = create_initial_state(
            session_id=self.session_id,
            file_path=self.test_file_path
        )
        
        # Add recoverable error
        initial_state.add_error(
            "Recoverable processing warning",
            ErrorSeverity.LOW,
            ProcessingStage.SEMANTIC_MAPPING
        )
        
        # Verify error doesn't stop processing
        from langgraph_nodes import should_continue_processing
        should_continue = should_continue_processing(initial_state.to_dict())
        self.assertTrue(should_continue)
        
        # Add critical error
        initial_state.add_error(
            "Critical system failure",
            ErrorSeverity.CRITICAL,
            ProcessingStage.LLM_PROCESSING
        )
        
        # Verify critical error stops processing
        should_continue = should_continue_processing(initial_state.to_dict())
        self.assertFalse(should_continue)
    
    def test_workflow_fallback_mode(self):
        """Test workflow fallback to linear processing"""
        # Test when LangGraph is not available
        with patch('langgraph_workflow.LANGGRAPH_AVAILABLE', False):
            orchestrator = LangGraphWorkflowOrchestrator()
            
            # Create initial state
            initial_state = create_initial_state(
                session_id=self.session_id,
                file_path=self.test_file_path
            )
            
            # Verify fallback mode is used
            with patch.object(orchestrator, '_run_linear_fallback') as mock_fallback:
                mock_fallback.return_value = {'current_stage': ProcessingStage.COMPLETED}
                
                result = orchestrator.run_workflow(initial_state.to_dict())
                
                mock_fallback.assert_called_once()
                self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
    
    def _setup_successful_mocks(self, mock_parser, mock_chunker, mock_kg, 
                              mock_semantic, mock_llm, mock_output, mock_analytics):
        """Set up all mocks for successful workflow execution"""
        self._setup_parser_mock(mock_parser)
        self._setup_chunker_mock(mock_chunker)
        self._setup_kg_mock(mock_kg)
        self._setup_semantic_mock(mock_semantic)
        self._setup_llm_mock(mock_llm)
        self._setup_output_mock(mock_output)
        self._setup_analytics_mock(mock_analytics)
    
    def _setup_parser_mock(self, mock_parser):
        """Set up document parser mock"""
        mock_parser_instance = Mock()
        mock_parser_instance.parse.return_value = {
            'content': 'Sample document content for testing workflow integration.',
            'metadata': {
                'pages': 3,
                'word_count': 500,
                'sections': ['Introduction', 'Content', 'Conclusion']
            }
        }
        mock_parser.return_value = mock_parser_instance
    
    def _setup_chunker_mock(self, mock_chunker):
        """Set up document chunker mock"""
        mock_chunker_instance = Mock()
        mock_chunker_instance.chunk.return_value = {
            'chunks': [
                {'content': 'Sample document content', 'metadata': {'chunk_id': 0}},
                {'content': 'for testing workflow', 'metadata': {'chunk_id': 1}},
                {'content': 'integration.', 'metadata': {'chunk_id': 2}}
            ],
            'metadata': {
                'total_chunks': 3,
                'average_chunk_size': 20,
                'chunking_strategy': 'semantic'
            }
        }
        mock_chunker.return_value = mock_chunker_instance
    
    def _setup_kg_mock(self, mock_kg):
        """Set up knowledge graph builder mock"""
        mock_kg_instance = Mock()
        mock_kg_instance.build.return_value = {
            'nodes': [
                {'id': 'document', 'type': 'concept', 'properties': {}},
                {'id': 'workflow', 'type': 'concept', 'properties': {}},
                {'id': 'testing', 'type': 'concept', 'properties': {}}
            ],
            'edges': [
                {'source': 'document', 'target': 'workflow', 'relation': 'part_of'},
                {'source': 'workflow', 'target': 'testing', 'relation': 'enables'}
            ],
            'metadata': {
                'node_count': 3,
                'edge_count': 2,
                'extraction_confidence': 0.88
            }
        }
        mock_kg.return_value = mock_kg_instance
    
    def _setup_semantic_mock(self, mock_semantic):
        """Set up semantic mapper mock"""
        mock_semantic_instance = Mock()
        mock_semantic_instance.map.return_value = {
            'mappings': [
                {'chunk_id': 0, 'concepts': ['document'], 'similarity_scores': [0.91], 'confidence': 0.89},
                {'chunk_id': 1, 'concepts': ['workflow'], 'similarity_scores': [0.87], 'confidence': 0.85},
                {'chunk_id': 2, 'concepts': ['testing'], 'similarity_scores': [0.93], 'confidence': 0.91}
            ],
            'metadata': {
                'mapping_count': 3,
                'average_similarity': 0.90,
                'mapping_strategy': 'embedding_similarity'
            }
        }
        mock_semantic.return_value = mock_semantic_instance
    
    def _setup_llm_mock(self, mock_llm):
        """Set up LLM processor mock"""
        mock_llm_instance = Mock()
        mock_llm_instance.process.return_value = {
            'enhanced_content': [
                {
                    'chunk_id': 0,
                    'original': 'Sample document content',
                    'enhanced': 'Enhanced sample document content with additional context.',
                    'insights': ['Document processing', 'Content enhancement']
                },
                {
                    'chunk_id': 1,
                    'original': 'for testing workflow',
                    'enhanced': 'Designed for comprehensive testing of workflow systems.',
                    'insights': ['Testing methodology', 'Workflow validation']
                },
                {
                    'chunk_id': 2,
                    'original': 'integration.',
                    'enhanced': 'System integration and component interaction.',
                    'insights': ['Integration patterns', 'Component coupling']
                }
            ],
            'metadata': {
                'tokens_processed': 200,
                'processing_time': 3.2,
                'model_used': 'gpt-4',
                'enhancement_quality': 0.92
            }
        }
        mock_llm.return_value = mock_llm_instance
    
    def _setup_output_mock(self, mock_output):
        """Set up output generator mock"""
        mock_output_instance = Mock()
        mock_output_instance.generate.return_value = {
            'output_files': [
                os.path.join(self.temp_dir, 'result.json'),
                os.path.join(self.temp_dir, 'summary.md')
            ],
            'formats': ['json', 'markdown'],
            'metadata': {
                'generation_time': 1.5,
                'file_sizes': {'json': 3072, 'markdown': 1536},
                'quality_score': 0.95
            }
        }
        mock_output.return_value = mock_output_instance
    
    def _setup_analytics_mock(self, mock_analytics):
        """Set up analytics collector mock"""
        mock_analytics_instance = Mock()
        mock_analytics_instance.collect.return_value = {
            'processing_metrics': {
                'total_time': 18.3,
                'memory_used': '312MB',
                'cpu_usage': '38%',
                'stage_timings': {
                    'document_parsing': 2.5,
                    'chunking': 1.9,
                    'knowledge_graph': 4.8,
                    'semantic_mapping': 3.4,
                    'llm_processing': 4.2,
                    'output_generation': 1.5
                }
            },
            'quality_metrics': {
                'coherence_score': 0.91,
                'completeness_score': 0.94,
                'accuracy_score': 0.89
            },
            'metadata': {
                'collection_time': datetime.now().isoformat(),
                'metrics_version': '1.0'
            }
        }
        mock_analytics.return_value = mock_analytics_instance


class TestLangGraphWorkflowEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions for LangGraph workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = LangGraphWorkflowOrchestrator()
    
    def test_empty_document_handling(self):
        """Test workflow handling of empty documents"""
        initial_state = create_initial_state(
            session_id='empty-doc-test',
            file_path='/test/empty.pdf'
        )
        
        with patch('langgraph_nodes.DocumentParser') as mock_parser:
            mock_parser_instance = Mock()
            mock_parser_instance.parse.return_value = {
                'content': '',
                'metadata': {'pages': 0, 'word_count': 0}
            }
            mock_parser.return_value = mock_parser_instance
            
            result = self.orchestrator.run_workflow(initial_state.to_dict())
            
            # Should handle empty content gracefully
            self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
            self.assertGreater(len(result['errors']), 0)
    
    def test_large_document_handling(self):
        """Test workflow handling of very large documents"""
        initial_state = create_initial_state(
            session_id='large-doc-test',
            file_path='/test/large_document.pdf'
        )
        
        # Simulate large document
        large_content = 'Large document content. ' * 10000  # ~250KB of text
        
        with patch('langgraph_nodes.DocumentParser') as mock_parser:
            mock_parser_instance = Mock()
            mock_parser_instance.parse.return_value = {
                'content': large_content,
                'metadata': {'pages': 100, 'word_count': 20000}
            }
            mock_parser.return_value = mock_parser_instance
            
            with patch('langgraph_nodes.DocumentChunker') as mock_chunker:
                # Simulate many chunks
                chunks = [{
                    'content': f'Chunk {i} content',
                    'metadata': {'chunk_id': i}
                } for i in range(50)]
                
                mock_chunker_instance = Mock()
                mock_chunker_instance.chunk.return_value = {
                    'chunks': chunks,
                    'metadata': {
                        'total_chunks': 50,
                        'average_chunk_size': 5000,
                        'chunking_strategy': 'fixed_size'
                    }
                }
                mock_chunker.return_value = mock_chunker_instance
                
                result = self.orchestrator.run_workflow(initial_state.to_dict())
                
                # Should handle large documents
                if result['current_stage'] != ProcessingStage.ERROR:
                    self.assertIn('chunk_info', result)
                    self.assertEqual(result['chunk_info']['total_chunks'], 50)
    
    def test_malformed_state_handling(self):
        """Test workflow handling of malformed state data"""
        # Test with missing required fields
        malformed_state = {
            'session_id': 'malformed-test'
            # Missing other required fields
        }
        
        result = self.orchestrator.run_workflow(malformed_state)
        
        # Should handle malformed state gracefully
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
    
    def test_concurrent_workflow_execution(self):
        """Test concurrent execution of multiple workflows"""
        import threading
        import time
        
        results = {}
        
        def run_workflow(session_id):
            initial_state = create_initial_state(
                session_id=session_id,
                file_path=f'/test/doc_{session_id}.pdf'
            )
            
            with patch('langgraph_nodes.DocumentParser') as mock_parser:
                mock_parser_instance = Mock()
                mock_parser_instance.parse.return_value = {
                    'content': f'Content for {session_id}',
                    'metadata': {'pages': 1, 'word_count': 10}
                }
                mock_parser.return_value = mock_parser_instance
                
                # Add small delay to simulate processing
                time.sleep(0.1)
                
                orchestrator = LangGraphWorkflowOrchestrator()
                results[session_id] = orchestrator.run_workflow(initial_state.to_dict())
        
        # Run multiple workflows concurrently
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_workflow, args=(f'concurrent-{i}',))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all workflows completed
        self.assertEqual(len(results), 3)
        for session_id, result in results.items():
            self.assertIn('session_id', result)
            self.assertEqual(result['session_id'], session_id)


if __name__ == '__main__':
    unittest.main()