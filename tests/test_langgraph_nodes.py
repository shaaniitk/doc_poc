"""Tests for LangGraph workflow nodes"""
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
from typing import Dict, List, Any

# Import LangGraph components
from langgraph_state import (
    ProcessingStage, ErrorSeverity, ErrorInfo, DocumentInfo,
    ChunkInfo, KnowledgeGraphNode, KnowledgeGraphEdge, SemanticMapping,
    LLMProcessingResult, OutputData, AnalyticsData,
    PipelineState, create_initial_state
)
from langgraph_nodes import DocumentProcessingNodes


class TestInitializePipeline(unittest.TestCase):
    """Test pipeline initialization node"""
    
    def test_initialize_pipeline_success(self):
        """Test successful pipeline initialization"""
        initial_state = create_initial_state(
            session_id='test-session',
            file_path='/test/document.pdf'
        )
        
        result = initialize_pipeline(initial_state.to_dict())
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['current_stage'], ProcessingStage.DOCUMENT_PARSING)
        self.assertIn('document_info', result)
        self.assertEqual(result['document_info']['file_path'], '/test/document.pdf')
    
    def test_initialize_pipeline_with_config(self):
        """Test pipeline initialization with additional config"""
        initial_state = create_initial_state(
            session_id='test-session',
            file_path='/test/document.pdf'
        )
        state_dict = initial_state.to_dict()
        state_dict['config'] = {
            'use_enhanced_processing': True,
            'output_format': 'json'
        }
        
        result = initialize_pipeline(state_dict)
        
        self.assertEqual(result['current_stage'], ProcessingStage.DOCUMENT_PARSING)
        self.assertIn('config', result)
        self.assertTrue(result['config']['use_enhanced_processing'])
    
    def test_initialize_pipeline_missing_file(self):
        """Test pipeline initialization with missing file path"""
        state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.INITIALIZATION,
            'document_info': {},
            'errors': []
        }
        
        result = initialize_pipeline(state)
        
        # Should add error and move to error stage
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('file_path', result['errors'][0]['message'])


class TestParseDocument(unittest.TestCase):
    """Test document parsing node"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.DOCUMENT_PARSING,
            'document_info': {
                'file_path': '/test/document.pdf',
                'file_type': 'pdf',
                'size_bytes': 2048
            },
            'errors': []
        }
    
    @patch('langgraph_nodes.DocumentParser')
    def test_parse_document_success(self, mock_parser_class):
        """Test successful document parsing"""
        # Mock parser
        mock_parser = Mock()
        mock_parser.parse.return_value = {
            'content': 'Parsed document content with multiple sections.',
            'metadata': {
                'pages': 5,
                'word_count': 1200,
                'sections': ['Introduction', 'Methods', 'Results']
            }
        }
        mock_parser_class.return_value = mock_parser
        
        result = parse_document(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.CHUNKING)
        self.assertIn('parsed_content', result)
        self.assertEqual(result['parsed_content']['content'], 'Parsed document content with multiple sections.')
        self.assertEqual(result['parsed_content']['metadata']['pages'], 5)
        mock_parser.parse.assert_called_once_with('/test/document.pdf')
    
    @patch('langgraph_nodes.DocumentParser')
    def test_parse_document_failure(self, mock_parser_class):
        """Test document parsing failure"""
        # Mock parser that raises exception
        mock_parser = Mock()
        mock_parser.parse.side_effect = Exception("Failed to parse document")
        mock_parser_class.return_value = mock_parser
        
        result = parse_document(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('Failed to parse document', result['errors'][0]['message'])
    
    def test_parse_document_missing_file_path(self):
        """Test parsing with missing file path"""
        state = self.state.copy()
        state['document_info'] = {}
        
        result = parse_document(state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)


class TestChunkDocument(unittest.TestCase):
    """Test document chunking node"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.CHUNKING,
            'parsed_content': {
                'content': 'This is a long document that needs to be chunked into smaller pieces for processing.',
                'metadata': {'pages': 3}
            },
            'errors': []
        }
    
    @patch('langgraph_nodes.DocumentChunker')
    def test_chunk_document_success(self, mock_chunker_class):
        """Test successful document chunking"""
        # Mock chunker
        mock_chunker = Mock()
        mock_chunker.chunk.return_value = {
            'chunks': [
                {'content': 'This is a long document', 'metadata': {'chunk_id': 0}},
                {'content': 'that needs to be chunked', 'metadata': {'chunk_id': 1}},
                {'content': 'into smaller pieces for processing.', 'metadata': {'chunk_id': 2}}
            ],
            'metadata': {
                'total_chunks': 3,
                'average_chunk_size': 25,
                'chunking_strategy': 'semantic'
            }
        }
        mock_chunker_class.return_value = mock_chunker
        
        result = chunk_document(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.KNOWLEDGE_GRAPH)
        self.assertIn('chunk_info', result)
        self.assertEqual(result['chunk_info']['total_chunks'], 3)
        self.assertEqual(result['chunk_info']['processed_chunks'], 3)
        self.assertEqual(len(result['chunks']), 3)
    
    @patch('langgraph_nodes.DocumentChunker')
    def test_chunk_document_empty_content(self, mock_chunker_class):
        """Test chunking with empty content"""
        state = self.state.copy()
        state['parsed_content']['content'] = ''
        
        result = chunk_document(state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
    
    @patch('langgraph_nodes.DocumentChunker')
    def test_chunk_document_chunker_failure(self, mock_chunker_class):
        """Test chunking failure"""
        mock_chunker = Mock()
        mock_chunker.chunk.side_effect = Exception("Chunking failed")
        mock_chunker_class.return_value = mock_chunker
        
        result = chunk_document(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('Chunking failed', result['errors'][0]['message'])


class TestCreateKnowledgeGraph(unittest.TestCase):
    """Test knowledge graph creation node"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.KNOWLEDGE_GRAPH,
            'chunks': [
                {'content': 'Machine learning is a subset of AI.', 'metadata': {'chunk_id': 0}},
                {'content': 'Neural networks are used in deep learning.', 'metadata': {'chunk_id': 1}}
            ],
            'errors': []
        }
    
    @patch('langgraph_nodes.KnowledgeGraphBuilder')
    def test_create_knowledge_graph_success(self, mock_kg_class):
        """Test successful knowledge graph creation"""
        # Mock knowledge graph builder
        mock_kg = Mock()
        mock_kg.build.return_value = {
            'nodes': [
                {'id': 'machine_learning', 'type': 'concept', 'properties': {}},
                {'id': 'ai', 'type': 'concept', 'properties': {}},
                {'id': 'neural_networks', 'type': 'concept', 'properties': {}}
            ],
            'edges': [
                {'source': 'machine_learning', 'target': 'ai', 'relation': 'subset_of'},
                {'source': 'neural_networks', 'target': 'machine_learning', 'relation': 'used_in'}
            ],
            'metadata': {
                'node_count': 3,
                'edge_count': 2,
                'extraction_confidence': 0.85
            }
        }
        mock_kg_class.return_value = mock_kg
        
        result = create_knowledge_graph(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.SEMANTIC_MAPPING)
        self.assertIn('knowledge_graph_info', result)
        self.assertEqual(result['knowledge_graph_info']['node_count'], 3)
        self.assertEqual(result['knowledge_graph_info']['edge_count'], 2)
        self.assertIn('knowledge_graph', result)
    
    @patch('langgraph_nodes.KnowledgeGraphBuilder')
    def test_create_knowledge_graph_no_chunks(self, mock_kg_class):
        """Test knowledge graph creation with no chunks"""
        state = self.state.copy()
        state['chunks'] = []
        
        result = create_knowledge_graph(state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
    
    @patch('langgraph_nodes.KnowledgeGraphBuilder')
    def test_create_knowledge_graph_builder_failure(self, mock_kg_class):
        """Test knowledge graph creation failure"""
        mock_kg = Mock()
        mock_kg.build.side_effect = Exception("KG building failed")
        mock_kg_class.return_value = mock_kg
        
        result = create_knowledge_graph(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('KG building failed', result['errors'][0]['message'])


class TestPerformSemanticMapping(unittest.TestCase):
    """Test semantic mapping node"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.SEMANTIC_MAPPING,
            'chunks': [
                {'content': 'Machine learning algorithms', 'metadata': {'chunk_id': 0}}
            ],
            'knowledge_graph': {
                'nodes': [{'id': 'machine_learning', 'type': 'concept'}],
                'edges': []
            },
            'errors': []
        }
    
    @patch('langgraph_nodes.SemanticMapper')
    def test_perform_semantic_mapping_success(self, mock_mapper_class):
        """Test successful semantic mapping"""
        # Mock semantic mapper
        mock_mapper = Mock()
        mock_mapper.map.return_value = {
            'mappings': [
                {
                    'chunk_id': 0,
                    'concepts': ['machine_learning'],
                    'similarity_scores': [0.92],
                    'confidence': 0.88
                }
            ],
            'metadata': {
                'mapping_count': 1,
                'average_similarity': 0.92,
                'mapping_strategy': 'embedding_similarity'
            }
        }
        mock_mapper_class.return_value = mock_mapper
        
        result = perform_semantic_mapping(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.LLM_PROCESSING)
        self.assertIn('semantic_mapping_info', result)
        self.assertEqual(result['semantic_mapping_info']['mapping_count'], 1)
        self.assertIn('semantic_mappings', result)
    
    @patch('langgraph_nodes.SemanticMapper')
    def test_perform_semantic_mapping_no_knowledge_graph(self, mock_mapper_class):
        """Test semantic mapping without knowledge graph"""
        state = self.state.copy()
        del state['knowledge_graph']
        
        result = perform_semantic_mapping(state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
    
    @patch('langgraph_nodes.SemanticMapper')
    def test_perform_semantic_mapping_failure(self, mock_mapper_class):
        """Test semantic mapping failure"""
        mock_mapper = Mock()
        mock_mapper.map.side_effect = Exception("Semantic mapping failed")
        mock_mapper_class.return_value = mock_mapper
        
        result = perform_semantic_mapping(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('Semantic mapping failed', result['errors'][0]['message'])


class TestProcessWithLLM(unittest.TestCase):
    """Test LLM processing node"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.LLM_PROCESSING,
            'semantic_mappings': [
                {'chunk_id': 0, 'concepts': ['machine_learning'], 'confidence': 0.88}
            ],
            'chunks': [
                {'content': 'Machine learning algorithms', 'metadata': {'chunk_id': 0}}
            ],
            'errors': []
        }
    
    @patch('langgraph_nodes.LLMProcessor')
    def test_process_with_llm_success(self, mock_llm_class):
        """Test successful LLM processing"""
        # Mock LLM processor
        mock_llm = Mock()
        mock_llm.process.return_value = {
            'enhanced_content': [
                {
                    'chunk_id': 0,
                    'original': 'Machine learning algorithms',
                    'enhanced': 'Machine learning algorithms are computational methods that enable systems to learn from data.',
                    'insights': ['Focuses on computational methods', 'Emphasizes learning from data']
                }
            ],
            'metadata': {
                'tokens_processed': 150,
                'processing_time': 2.5,
                'model_used': 'gpt-4',
                'enhancement_quality': 0.91
            }
        }
        mock_llm_class.return_value = mock_llm
        
        result = process_with_llm(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.OUTPUT_GENERATION)
        self.assertIn('llm_processing_info', result)
        self.assertEqual(result['llm_processing_info']['tokens_processed'], 150)
        self.assertIn('llm_results', result)
    
    @patch('langgraph_nodes.LLMProcessor')
    def test_process_with_llm_no_mappings(self, mock_llm_class):
        """Test LLM processing without semantic mappings"""
        state = self.state.copy()
        state['semantic_mappings'] = []
        
        result = process_with_llm(state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
    
    @patch('langgraph_nodes.LLMProcessor')
    def test_process_with_llm_failure(self, mock_llm_class):
        """Test LLM processing failure"""
        mock_llm = Mock()
        mock_llm.process.side_effect = Exception("LLM processing failed")
        mock_llm_class.return_value = mock_llm
        
        result = process_with_llm(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('LLM processing failed', result['errors'][0]['message'])


class TestGenerateOutput(unittest.TestCase):
    """Test output generation node"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.OUTPUT_GENERATION,
            'llm_results': {
                'enhanced_content': [
                    {'chunk_id': 0, 'enhanced': 'Enhanced content'}
                ]
            },
            'config': {
                'output_dir': '/test/output',
                'output_formats': ['json', 'markdown']
            },
            'errors': []
        }
    
    @patch('langgraph_nodes.OutputGenerator')
    def test_generate_output_success(self, mock_output_class):
        """Test successful output generation"""
        # Mock output generator
        mock_generator = Mock()
        mock_generator.generate.return_value = {
            'output_files': [
                '/test/output/result.json',
                '/test/output/summary.md'
            ],
            'formats': ['json', 'markdown'],
            'metadata': {
                'generation_time': 1.8,
                'file_sizes': {'json': 2048, 'markdown': 1024},
                'quality_score': 0.94
            }
        }
        mock_output_class.return_value = mock_generator
        
        result = generate_output(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ANALYTICS)
        self.assertIn('output_info', result)
        self.assertEqual(len(result['output_info']['output_files']), 2)
        self.assertIn('json', result['output_info']['formats'])
    
    @patch('langgraph_nodes.OutputGenerator')
    def test_generate_output_no_llm_results(self, mock_output_class):
        """Test output generation without LLM results"""
        state = self.state.copy()
        del state['llm_results']
        
        result = generate_output(state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
    
    @patch('langgraph_nodes.OutputGenerator')
    def test_generate_output_failure(self, mock_output_class):
        """Test output generation failure"""
        mock_generator = Mock()
        mock_generator.generate.side_effect = Exception("Output generation failed")
        mock_output_class.return_value = mock_generator
        
        result = generate_output(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('Output generation failed', result['errors'][0]['message'])


class TestCollectAnalytics(unittest.TestCase):
    """Test analytics collection node"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.ANALYTICS,
            'output_info': {
                'output_files': ['/test/output/result.json'],
                'formats': ['json']
            },
            'start_time': datetime.now().isoformat(),
            'errors': []
        }
    
    @patch('langgraph_nodes.AnalyticsCollector')
    def test_collect_analytics_success(self, mock_analytics_class):
        """Test successful analytics collection"""
        # Mock analytics collector
        mock_analytics = Mock()
        mock_analytics.collect.return_value = {
            'processing_metrics': {
                'total_time': 15.7,
                'memory_used': '256MB',
                'cpu_usage': '42%',
                'stage_timings': {
                    'document_parsing': 2.1,
                    'chunking': 1.8,
                    'knowledge_graph': 4.2,
                    'semantic_mapping': 3.1,
                    'llm_processing': 3.5,
                    'output_generation': 1.0
                }
            },
            'quality_metrics': {
                'coherence_score': 0.89,
                'completeness_score': 0.92,
                'accuracy_score': 0.87
            },
            'metadata': {
                'collection_time': datetime.now().isoformat(),
                'metrics_version': '1.0'
            }
        }
        mock_analytics_class.return_value = mock_analytics
        
        result = collect_analytics(self.state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        self.assertIn('analytics_info', result)
        self.assertIn('processing_metrics', result['analytics_info'])
        self.assertIn('quality_metrics', result['analytics_info'])
    
    @patch('langgraph_nodes.AnalyticsCollector')
    def test_collect_analytics_failure(self, mock_analytics_class):
        """Test analytics collection failure"""
        mock_analytics = Mock()
        mock_analytics.collect.side_effect = Exception("Analytics collection failed")
        mock_analytics_class.return_value = mock_analytics
        
        result = collect_analytics(self.state)
        
        # Analytics failure shouldn't stop the pipeline
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        self.assertGreater(len(result['errors']), 0)
        self.assertIn('Analytics collection failed', result['errors'][0]['message'])
        # Should have basic analytics info even on failure
        self.assertIn('analytics_info', result)


class TestCompletePipeline(unittest.TestCase):
    """Test pipeline completion node"""
    
    def test_complete_pipeline_success(self):
        """Test successful pipeline completion"""
        state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.COMPLETED,
            'analytics_info': {
                'processing_metrics': {'total_time': 20.5}
            },
            'errors': []
        }
        
        result = complete_pipeline(state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        self.assertIn('completion_time', result)
        self.assertIn('final_status', result)
        self.assertEqual(result['final_status'], 'success')
    
    def test_complete_pipeline_with_warnings(self):
        """Test pipeline completion with warnings"""
        state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.COMPLETED,
            'analytics_info': {
                'processing_metrics': {'total_time': 20.5}
            },
            'errors': [
                {
                    'message': 'Minor warning',
                    'severity': ErrorSeverity.LOW,
                    'stage': ProcessingStage.CHUNKING
                }
            ]
        }
        
        result = complete_pipeline(state)
        
        self.assertEqual(result['current_stage'], ProcessingStage.COMPLETED)
        self.assertEqual(result['final_status'], 'success_with_warnings')
        self.assertIn('warning_count', result)
        self.assertEqual(result['warning_count'], 1)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for workflow nodes"""
    
    def test_handle_error(self):
        """Test error handling utility"""
        state = {
            'session_id': 'test-session',
            'current_stage': ProcessingStage.CHUNKING,
            'errors': []
        }
        
        error_msg = "Test error occurred"
        severity = ErrorSeverity.HIGH
        
        result = handle_error(state, error_msg, severity)
        
        self.assertEqual(result['current_stage'], ProcessingStage.ERROR)
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(result['errors'][0]['message'], error_msg)
        self.assertEqual(result['errors'][0]['severity'], severity)
    
    def test_should_continue_processing_no_critical_errors(self):
        """Test processing continuation with no critical errors"""
        state = {
            'errors': [
                {
                    'message': 'Warning',
                    'severity': ErrorSeverity.LOW,
                    'stage': ProcessingStage.CHUNKING
                }
            ]
        }
        
        should_continue = should_continue_processing(state)
        self.assertTrue(should_continue)
    
    def test_should_continue_processing_with_critical_errors(self):
        """Test processing continuation with critical errors"""
        state = {
            'errors': [
                {
                    'message': 'Critical error',
                    'severity': ErrorSeverity.CRITICAL,
                    'stage': ProcessingStage.LLM_PROCESSING
                }
            ]
        }
        
        should_continue = should_continue_processing(state)
        self.assertFalse(should_continue)
    
    def test_route_to_next_stage(self):
        """Test routing to next processing stage"""
        # Test normal progression
        self.assertEqual(
            route_to_next_stage(ProcessingStage.INITIALIZATION),
            ProcessingStage.DOCUMENT_PARSING
        )
        self.assertEqual(
            route_to_next_stage(ProcessingStage.CHUNKING),
            ProcessingStage.KNOWLEDGE_GRAPH
        )
        
        # Test final stages
        self.assertEqual(
            route_to_next_stage(ProcessingStage.COMPLETED),
            ProcessingStage.COMPLETED
        )
        self.assertEqual(
            route_to_next_stage(ProcessingStage.ERROR),
            ProcessingStage.ERROR
        )


if __name__ == '__main__':
    unittest.main()