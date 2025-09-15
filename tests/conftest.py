"""Pytest configuration and shared fixtures for the document processing pipeline tests."""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import types

# Import actual classes from core modules
from core.state_manager import CentralizedStateManager, ProcessingStage, ProcessingMetrics, ErrorContext
from enhanced_main import ProcessingConfig, EnhancedDocumentProcessor, SystemMonitor
from core.chunking_processor import ChunkingProcessor, ChunkingConfig, ChunkingStrategy
from core.llm_handler import LLMHandler, LLMConfig, ProcessingMode
from core.knowledge_graph_processor import KnowledgeGraphProcessor, ExtractionConfig
from core.output_generator import OutputGenerator, OutputConfig, OutputFormat
from core.monitoring import MonitoringSystem
from core.error_handler import ErrorHandler, ErrorSeverity
from core.config import ConfigManager

# Stub heavy external dependencies to keep test environment lean.
# sentence-transformers
st = types.ModuleType('sentence_transformers')
class _DummySTModel:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, texts, show_progress_bar=True, **kwargs):
        """Generate embeddings with realistic semantic similarity behavior."""
        if isinstance(texts, str):
            texts = [texts]
        
        import numpy as np
        import hashlib
        embeddings = []
        
        for text in texts:
            text_lower = str(text).lower()
            
            # Create deterministic but varied embeddings based on content
            # Use hash to ensure consistent embeddings for same text
            text_hash = hashlib.md5(text_lower.encode()).hexdigest()
            
            # Convert hash to numeric seed
            seed = int(text_hash[:8], 16) % 1000000
            np.random.seed(seed)
            
            # Generate base embedding with positive bias to avoid negative similarities
            base_embedding = np.random.normal(0.1, 0.05, 384)  # Positive mean, smaller variance
            
            # Add semantic features based on content type
            # Machine learning / AI content
            if any(term in text_lower for term in ['machine learning', 'artificial intelligence', 'algorithm', 'data analysis']):
                base_embedding[:50] += 0.4  # Strong signal in first dimensions
                base_embedding[200:250] += 0.2  # Secondary signal
            
            # Commerce / financial / introduction content
            if any(term in text_lower for term in ['commerce', 'financial', 'institutions', 'trust', 'electronic payments', 'internet']):
                base_embedding[50:100] += 0.5  # Strong signal for introduction-related content
                base_embedding[300:350] += 0.3  # Secondary signal
            
            # Cooking / recipe content  
            elif any(term in text_lower for term in ['cooking', 'recipe', 'measurement', 'timing', 'ingredient']):
                base_embedding[50:100] += 0.4  # Different signal region
                base_embedding[250:300] += 0.2
            
            # Bitcoin / crypto content
            elif any(term in text_lower for term in ['bitcoin', 'crypto', 'blockchain', 'transaction', 'payment']):
                base_embedding[100:150] += 0.4
                base_embedding[300:350] += 0.2
            
            # Introduction content
            elif any(term in text_lower for term in ['introduction', 'commerce', 'internet', 'financial institutions']):
                base_embedding[150:200] += 0.4
                base_embedding[350:384] += 0.2
            
            # Normalize to unit vector for cosine similarity
            norm = np.linalg.norm(base_embedding)
            if norm > 0:
                base_embedding = base_embedding / norm
            
            embeddings.append(base_embedding)
        
        return np.array(embeddings, dtype=np.float32)

def SentenceTransformer(*args, **kwargs):
    return _DummySTModel()

st.SentenceTransformer = SentenceTransformer
sys.modules['sentence_transformers'] = st

# langchain stubs
langchain_mod = types.ModuleType('langchain')
text_splitter_mod = types.ModuleType('langchain.text_splitter')
class _DummySplitter:
    def __init__(self, *args, **kwargs):
        pass
    def split_text(self, text):
        return [text]
text_splitter_mod.MarkdownHeaderTextSplitter = _DummySplitter
sys.modules['langchain'] = langchain_mod
sys.modules['langchain.text_splitter'] = text_splitter_mod

# Add langchain.prompts and langchain.chains stubs
prompts_mod = types.ModuleType('langchain.prompts')
class PromptTemplate:
    def __init__(self, input_variables=None, template=""):
        self.input_variables = input_variables or []
        self.template = template
    def format(self, **kwargs):
        return self.template.format(**kwargs)
prompts_mod.PromptTemplate = PromptTemplate
chains_mod = types.ModuleType('langchain.chains')
class LLMChain:
    def __init__(self, llm=None, prompt=None):
        self.llm = llm
        self.prompt = prompt
    def invoke(self, inputs):
        # Return empty mapping to avoid affecting logic
        return {"text": ""}
chains_mod.LLMChain = LLMChain
sys.modules['langchain.prompts'] = prompts_mod
sys.modules['langchain.chains'] = chains_mod

# docx stub
sys.modules.setdefault('docx', types.ModuleType('docx'))

# sklearn stubs if needed
sk_mod = types.ModuleType('sklearn')
cluster_mod = types.ModuleType('sklearn.cluster')
class _DummyKMeans:
    def __init__(self, *args, **kwargs):
        self.labels_ = []
    def fit(self, X):
        import numpy as np
        n = len(X)
        self.labels_ = [0] * n
        return self
cluster_mod.KMeans = _DummyKMeans
# Add sklearn.metrics.pairwise.cosine_similarity stub
metrics_mod = types.ModuleType('sklearn.metrics')
pairwise_mod = types.ModuleType('sklearn.metrics.pairwise')

def _cosine_similarity(X, Y):
    import numpy as np
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9
    Xn = X / X_norm
    Yn = Y / Y_norm
    return Xn @ Yn.T

pairwise_mod.cosine_similarity = _cosine_similarity
sys.modules['sklearn'] = sk_mod
sys.modules['sklearn.cluster'] = cluster_mod
sys.modules['sklearn.metrics'] = metrics_mod
sys.modules['sklearn.metrics.pairwise'] = pairwise_mod

# modules.llm_client stub
llm_client_mod = types.ModuleType('modules.llm_client')
class _DummyLLMClient:
    def __init__(self, *args, **kwargs):
        pass
    def complete(self, *args, **kwargs):
        return ""
# Also provide UnifiedLLMClient and LangChainLLM names used in code
class UnifiedLLMClient(_DummyLLMClient):
    pass
class LangChainLLM:
    def __init__(self, client=None):
        self.client = client
    def __call__(self, prompt):
        return ""
llm_client_mod.LLMClient = _DummyLLMClient
llm_client_mod.UnifiedLLMClient = UnifiedLLMClient
llm_client_mod.LangChainLLM = LangChainLLM
sys.modules['modules.llm_client'] = llm_client_mod

# jinja2 stub (minimal) to satisfy imports if needed
jinja2_mod = types.ModuleType('jinja2')
class _DummyTemplate:
    def __init__(self, name):
        self.name = name
    def render(self, **kwargs):
        # Very naive rendering used only if code insists on Jinja2
        processed_tree = kwargs.get('processed_tree', {})
        orphaned_content = kwargs.get('orphaned_content', [])
        # If latex template
        if self.name.endswith('.tex.j2'):
            body = []
            for k, v in processed_tree.items():
                body.append(f"\\section{{{k}}}\n{v if isinstance(v, str) else ''}")
            return "\\begin{document}\n" + "\n".join(body) + "\n\\end{document}"
        # If markdown template
        if self.name.endswith('.md.j2'):
            body = []
            for k, v in processed_tree.items():
                body.append(f"# {k}\n{v if isinstance(v, str) else ''}")
            return "\n\n".join(body)
        # If JSON template
        if self.name.endswith('.json.j2'):
            import json
            result = {
                "processed_tree": processed_tree,
                "orphaned_content": orphaned_content
            }
            return json.dumps(result, indent=2)
        return ""
class Environment:
    def __init__(self, loader=None):
        self.loader = loader
        # Allow attributes assignment used by code
        self.block_start_string = '{%'
        self.block_end_string = '%}'
        self.variable_start_string = '{{'
        self.variable_end_string = '}}'
        self.comment_start_string = '{#'
        self.comment_end_string = '#}'
    def get_template(self, name):
        return _DummyTemplate(name)
class FileSystemLoader:
    def __init__(self, *args, **kwargs):
        pass
jinja2_mod.Environment = Environment
jinja2_mod.FileSystemLoader = FileSystemLoader
sys.modules['jinja2'] = jinja2_mod

# networkx stub (minimal Graph API)
nx_mod = types.ModuleType('networkx')
class _DummyGraph:
    def __init__(self):
        self._adj = {}
    def add_edge(self, u, v):
        self._adj.setdefault(u, set()).add(v)
        self._adj.setdefault(v, set()).add(u)
    def neighbors(self, u):
        return list(self._adj.get(u, []))
class Graph(_DummyGraph):
    pass
class DiGraph(_DummyGraph):
    pass
nx_mod.Graph = Graph
nx_mod.DiGraph = DiGraph
sys.modules['networkx'] = nx_mod


# Configure asyncio for testing
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config = ProcessingConfig()
    
    # Configure for testing environment
    config.chunking.strategy = ChunkingStrategy.SEMANTIC
    config.chunking.chunk_size = 512
    config.chunking.overlap = 50
    
    config.llm.provider = "mock"
    config.llm.model = "test-model"
    config.llm.temperature = 0.1
    config.llm.max_tokens = 1000
    
    config.knowledge_graph.enabled = True
    config.knowledge_graph.extraction_method = "rule_based"
    
    config.monitoring.enabled = True
    config.monitoring.log_level = "INFO"
    
    config.error_handling.max_retries = 3
    config.error_handling.retry_delay = 1.0
    
    return config


@pytest.fixture
def mock_application_config():
    """Create a processing config for testing."""
    return ProcessingConfig(
        source_path=Path("test_document.pdf"),
        output_path=Path("test_output"),
        chunking_strategy=ChunkingStrategy.SENTENCE_AWARE,
        chunk_size=1000,
        chunk_overlap=200,
        processing_mode=ProcessingMode.PARALLEL
    )


@pytest.fixture
def mock_chunking_processor():
    """Create a chunking processor for testing."""
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SENTENCE_AWARE,
        chunk_size=1000,
        overlap_size=200
    )
    return ChunkingProcessor(config)


@pytest.fixture
def mock_llm_handler():
    """Create an LLM handler for testing."""
    config = LLMConfig(
        provider="mock",
        model="test-model",
        temperature=0.1,
        max_tokens=1000
    )
    return LLMHandler(config)


@pytest.fixture
def mock_kg_processor():
    """Create a knowledge graph processor for testing."""
    config = ExtractionConfig(
        enabled=True,
        extraction_method="rule_based"
    )
    return KnowledgeGraphProcessor(config)


@pytest.fixture
def mock_output_generator():
    """Create a mock output generator for testing."""
    generator = Mock(spec=OutputGenerator)
    
    def mock_generate_output(processed_chunks, output_format=OutputFormat.MARKDOWN, **kwargs):
        if output_format == OutputFormat.JSON:
            import json
            return json.dumps({
                'chunks': [{'content': chunk} for chunk in processed_chunks],
                'metadata': {'format': 'json', 'chunk_count': len(processed_chunks)}
            })
        elif output_format == OutputFormat.MARKDOWN:
            content = "# Processed Document\n\n"
            for i, chunk in enumerate(processed_chunks):
                content += f"## Section {i+1}\n\n{chunk}\n\n"
            return content
        else:  # TEXT
            return "\n\n".join(processed_chunks)
    
    generator.generate_output.side_effect = mock_generate_output
    return generator


@pytest.fixture
def mock_error_handler():
    """Create a mock error handler for testing."""
    handler = Mock(spec=ErrorHandler)
    
    async def mock_handle_error(error, context=None, **kwargs):
        return {
            'handled': True,
            'recovery_attempted': True,
            'error_type': type(error).__name__,
            'severity': ErrorSeverity.MEDIUM.value
        }
    
    handler.handle_error = AsyncMock(side_effect=mock_handle_error)
    handler.classify_error = Mock(return_value=ErrorSeverity.MEDIUM)
    handler.should_retry = Mock(return_value=True)
    
    return handler


@pytest.fixture
def mock_monitoring_system():
    """Create a monitoring system for testing."""
    return MonitoringSystem()


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return {
        'simple': """
        # Simple Document
        
        This is a simple test document with basic content.
        It contains a few paragraphs for testing purposes.
        
        ## Section 1
        
        This is the first section with some content.
        
        ## Section 2
        
        This is the second section with more content.
        """,
        
        'complex': """
        # Complex Technical Document
        
        ## Abstract
        
        This document presents a comprehensive analysis of advanced machine learning
        techniques and their applications in modern AI systems.
        
        ## Introduction
        
        Machine learning has revolutionized numerous fields, from computer vision
        to natural language processing. This document explores cutting-edge approaches
        and their practical implementations.
        
        ### Background
        
        The field of artificial intelligence has seen unprecedented growth in recent years.
        Deep learning models, particularly transformer architectures, have achieved
        remarkable success across various domains.
        
        ## Methodology
        
        Our approach combines several state-of-the-art techniques:
        
        1. **Attention Mechanisms**: Self-attention and cross-attention layers
        2. **Transfer Learning**: Pre-trained models fine-tuned for specific tasks
        3. **Ensemble Methods**: Combining multiple models for improved performance
        
        ### Data Processing Pipeline
        
        The data processing pipeline consists of several stages:
        
        - Data collection and cleaning
        - Feature extraction and engineering
        - Model training and validation
        - Performance evaluation and optimization
        
        ## Results
        
        Our experiments demonstrate significant improvements over baseline methods:
        
        - 15% increase in accuracy on benchmark datasets
        - 30% reduction in training time
        - Improved generalization to unseen data
        
        ## Conclusion
        
        The proposed methodology shows promising results and opens new avenues
        for future research in machine learning and artificial intelligence.
        """,
        
        'code_heavy': """
        # Software Development Guide
        
        ## Python Best Practices
        
        ### Code Structure
        
        ```python
        class DocumentProcessor:
            def __init__(self, config):
                self.config = config
                self.logger = setup_logger()
            
            def process(self, document):
                try:
                    chunks = self.chunk_document(document)
                    results = []
                    for chunk in chunks:
                        result = self.process_chunk(chunk)
                        results.append(result)
                    return self.combine_results(results)
                except Exception as e:
                    self.logger.error(f"Processing failed: {e}")
                    raise
        ```
        
        ### Error Handling
        
        ```python
        def safe_process(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {e}")
                    return None
            return wrapper
        ```
        
        ## Database Operations
        
        ### SQL Queries
        
        ```sql
        SELECT u.name, p.title, COUNT(c.id) as comment_count
        FROM users u
        JOIN posts p ON u.id = p.user_id
        LEFT JOIN comments c ON p.id = c.post_id
        WHERE u.active = 1
        GROUP BY u.id, p.id
        ORDER BY comment_count DESC;
        ```
        
        ### Performance Optimization
        
        - Use appropriate indexes
        - Optimize query structure
        - Consider caching strategies
        - Monitor query performance
        """
    }


@pytest.fixture
def performance_test_config():
    """Configuration optimized for performance testing."""
    config = ProcessingConfig()
    
    # Optimize for speed in tests
    config.chunking.chunk_size = 256  # Smaller chunks for faster processing
    config.llm.max_tokens = 500
    config.llm.temperature = 0.0  # Deterministic for testing
    
    config.monitoring.enabled = True
    config.monitoring.collect_system_metrics = True
    
    return config


@pytest.fixture
def integration_test_processor(sample_config):
    """Create a processor instance for integration testing."""
    # Create actual processor instance for integration testing
    state_manager = CentralizedStateManager()
    processor = EnhancedDocumentProcessor(config=sample_config, state_manager=state_manager)
    return processor


# Utility functions for tests
def assert_valid_chunks(chunks: List[Dict]):
    """Assert that chunks are valid."""
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert 'id' in chunk
        assert 'content' in chunk
        assert 'metadata' in chunk
        assert len(chunk['content']) > 0


def assert_valid_processing_result(result: Dict):
    """Assert that processing result is valid."""
    assert isinstance(result, dict)
    assert 'success' in result
    assert 'document_id' in result
    
    if result['success']:
        assert 'output' in result
        assert 'chunks_processed' in result
        assert 'processing_stages' in result
        assert 'metrics' in result
        
        # Verify metrics structure
        metrics = result['metrics']
        assert 'processing_time' in metrics
        assert isinstance(metrics['processing_time'], (int, float))
        assert metrics['processing_time'] >= 0


def assert_valid_knowledge_graph(kg_data: Dict):
    """Assert that knowledge graph data is valid."""
    assert isinstance(kg_data, dict)
    assert 'entities' in kg_data
    assert 'relationships' in kg_data
    
    entities = kg_data['entities']
    assert isinstance(entities, list)
    
    for entity in entities:
        assert 'text' in entity
        assert 'label' in entity
        assert 'confidence' in entity
        assert 0 <= entity['confidence'] <= 1
    
    relationships = kg_data['relationships']
    assert isinstance(relationships, list)
    
    for rel in relationships:
        assert 'subject' in rel
        assert 'predicate' in rel
        assert 'object' in rel


# Markers for different test categories
pytestmark = [
    pytest.mark.asyncio,  # Default async support
]


# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring LLM access"
    )