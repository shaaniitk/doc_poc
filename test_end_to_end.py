"""End-to-end integration test for the complete document processing workflow."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

from core.template_processor_node import TemplateProcessorNode
from core.validation_node import ValidationNode, ValidationLevel
from core.combination_node import DocumentCombinationNode, CombinationStrategy
from core.output_formatter_node import OutputFormatterNode
from core.semantic_mapper import BasicSemanticMapper
from core.llm_handler import LLMHandler, LLMHandlerError
from core.models import (
    WorkflowState, DocumentMetadata, DocumentFormat, DocumentChunk,
    ProcessedSection, ValidationResult
)

# Mock LLM for testing
try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    class ChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass
        
        async def ainvoke(self, messages):
            # Mock response for testing
            return type('MockResponse', (), {'content': 'Mock LLM response for testing'})()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockLLM:
    """Mock LLM for testing purposes."""
    
    def __init__(self):
        self.model = "mock-gpt-3.5-turbo"
        self.temperature = 0.1
    
    async def agenerate(self, messages):
        # Return a mock response
        return "Mock LLM response for testing"
    
    async def ainvoke(self, messages):
        # Mock response object with content attribute
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse("Mock validation response: The content appears valid with good structure and completeness.")

class MockLLMHandler:
    """Mock LLM handler for testing purposes."""
    
    def __init__(self):
        self.model_name = "mock-gpt-3.5-turbo"
        self.temperature = 0.1
        self.max_tokens = 1000
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        return f"Mock response for prompt: {prompt[:50]}..."
    
    async def generate_structured_response(self, prompt: str, schema: dict, **kwargs) -> dict:
        """Generate a mock structured response."""
        return {"result": "mock structured response", "confidence": 0.8}
    
    def is_available(self) -> bool:
        """Always available for testing."""
        return True

def create_test_document() -> WorkflowState:
    """Create a test document for end-to-end processing."""
    # Create document metadata
    metadata = DocumentMetadata(
        title="System Architecture Analysis",
        format=DocumentFormat.MARKDOWN,
        creation_time=datetime.now(),
        modification_time=datetime.now()
    )
    
    # Create test content
    original_content = """
This document analyzes the system architecture for our new platform.

The system is designed with microservices architecture in mind, ensuring scalability and maintainability.
Each service is containerized using Docker and orchestrated with Kubernetes.

Our methodology involves iterative development with continuous integration and deployment.
We use automated testing at multiple levels including unit tests, integration tests, and end-to-end tests.
The development team follows agile practices with two-week sprints.

The results show significant improvements in system performance and reliability.
Response times have decreased by 40% compared to the legacy system.
System uptime has increased to 99.9% with the new architecture.

In conclusion, the new architecture provides a solid foundation for future growth.
The microservices approach allows for independent scaling and deployment of components.
This will enable faster feature development and better resource utilization.
"""
    
    # Create document chunks
    chunks = [
        DocumentChunk(
            content="This document analyzes the system architecture for our new platform. The system is designed with microservices architecture in mind, ensuring scalability and maintainability.",
            chunk_id="chunk_1",
            start_position=0,
            end_position=150
        ),
        DocumentChunk(
            content="Our methodology involves iterative development with continuous integration and deployment. We use automated testing at multiple levels including unit tests, integration tests, and end-to-end tests.",
            chunk_id="chunk_2",
            start_position=151,
            end_position=350
        ),
        DocumentChunk(
            content="The results show significant improvements in system performance and reliability. Response times have decreased by 40% compared to the legacy system.",
            chunk_id="chunk_3",
            start_position=351,
            end_position=500
        ),
        DocumentChunk(
            content="In conclusion, the new architecture provides a solid foundation for future growth. The microservices approach allows for independent scaling and deployment of components.",
            chunk_id="chunk_4",
            start_position=501,
            end_position=650
        )
    ]
    
    # Create workflow state
    workflow_state = WorkflowState(
        document_id="test_doc_e2e_001",
        original_content=original_content,
        metadata=metadata,
        chunks=chunks
    )
    
    return workflow_state

async def test_template_processing_step(workflow_state: WorkflowState) -> WorkflowState:
    """Test the template processing step."""
    print("\n📝 Step 1: Template Processing")
    print("-" * 40)
    
    # Initialize dependencies with mocks for testing
    llm = MockLLM()
    semantic_mapper = BasicSemanticMapper()
    
    processor = TemplateProcessorNode(llm=llm, semantic_mapper=semantic_mapper)
    
    # Convert WorkflowState to dict format expected by processor
    state_dict = {
        'document_id': workflow_state.document_id,
        'original_content': workflow_state.original_content,
        'metadata': workflow_state.metadata,
        'chunks': workflow_state.chunks
    }
    
    result = await processor.process(state_dict)
    
    # Extract processed sections from result
    if isinstance(result, dict) and 'processed_sections' in result:
        processed_sections = result['processed_sections']
    else:
        processed_sections = getattr(result, 'processed_sections', [])
    
    # Update workflow state with processed sections
    workflow_state.processed_sections = processed_sections
    
    logger.info(f"   ✅ Processed {len(processed_sections)} sections")
    for section in processed_sections:
        logger.info(f"   ✅ Section: {section.section_name} ({len(section.content)} chars)")
    
    return workflow_state

async def test_validation_step(workflow_state: WorkflowState) -> WorkflowState:
    """Test the validation step."""
    print("\n🔍 Step 2: Validation")
    print("-" * 40)
    
    # Initialize dependencies with mocks for testing
    llm = MockLLM()
    
    validator = ValidationNode(llm=llm, validation_level=ValidationLevel.STANDARD)
    result = await validator.validate_workflow_state(workflow_state)
    
    # Extract validation results (result is already a dict of ValidationResult objects)
    validation_results = result
    
    # Update workflow state with validation results
    workflow_state.validation_results = validation_results
    
    logger.info(f"   ✅ Validated {len(validation_results)} sections")
    for section_name, validation_result in validation_results.items():
        if hasattr(validation_result, 'is_valid'):
            logger.info(f"   ✅ Section '{section_name}': Valid={validation_result.is_valid}, Confidence={validation_result.confidence_score:.2f}")
        else:
            logger.info(f"   ✅ Section '{section_name}': validation completed")
    
    return workflow_state

async def test_combination_step(workflow_state: WorkflowState) -> WorkflowState:
    """Test the document combination step."""
    print("\n🔗 Step 3: Document Combination")
    print("-" * 40)
    
    combiner = DocumentCombinationNode(strategy=CombinationStrategy.WEAVE)
    result = await combiner.combine_sections(workflow_state)
    
    # Extract combined content (result is a CombinationResult object)
    if hasattr(result, 'combined_content'):
        combined_content = result.combined_content
        combination_metadata = {
            'strategy': result.strategy_used,
            'confidence': result.confidence_score,
            'sections_combined': result.sections_combined,
            'metadata': result.metadata
        }
    else:
        # Fallback for dict format
        combined_content = result.get('combined_content', '')
        combination_metadata = result.get('combination_metadata', {})
    
    # Update workflow state with combined content
    workflow_state.combined_content = combined_content
    workflow_state.combination_metadata = combination_metadata
    
    logger.info(f"   ✅ Combined content: {len(combined_content)} characters")
    logger.info(f"   ✅ Strategy: {combination_metadata.get('strategy', 'unknown')}")
    logger.info(f"   ✅ Confidence: {combination_metadata.get('confidence', 0.0):.2f}")
    
    return workflow_state

async def test_formatting_step(workflow_state: WorkflowState, target_format: DocumentFormat) -> Dict[str, Any]:
    """Test the output formatting step."""
    print(f"\n🎨 Step 4: Output Formatting ({target_format.value})")
    print("-" * 40)
    
    # Initialize dependencies
    llm_handler = MockLLMHandler()
    
    formatter = OutputFormatterNode(llm_handler=llm_handler)
    
    # Update metadata format for testing
    workflow_state.metadata.format = target_format
    
    result = await formatter.process(workflow_state)
    
    formatted_content = result['formatted_content']
    formatting_metadata = result['formatting_metadata']
    
    logger.info(f"   ✅ Formatted content: {len(formatted_content)} characters")
    logger.info(f"   ✅ Target format: {formatting_metadata['target_format']}")
    logger.info(f"   ✅ Transformations: {len(formatting_metadata['transformations_applied'])}")
    
    return result

async def test_complete_workflow():
    """Test the complete end-to-end workflow."""
    print("\n🚀 Testing Complete Document Processing Workflow")
    print("=" * 60)
    
    # Create test document
    workflow_state = create_test_document()
    logger.info(f"📄 Created test document: {workflow_state.metadata.title}")
    logger.info(f"📄 Original content: {len(workflow_state.original_content)} characters")
    logger.info(f"📄 Document chunks: {len(workflow_state.chunks)}")
    
    try:
        # Step 1: Template Processing
        workflow_state = await test_template_processing_step(workflow_state)
        
        # Step 2: Validation
        workflow_state = await test_validation_step(workflow_state)
        
        # Step 3: Document Combination
        workflow_state = await test_combination_step(workflow_state)
        
        # Step 4: Output Formatting (test multiple formats)
        formats_to_test = [DocumentFormat.MARKDOWN, DocumentFormat.LATEX]
        
        formatting_results = {}
        for target_format in formats_to_test:
            result = await test_formatting_step(workflow_state, target_format)
            formatting_results[target_format.value] = result
        
        return workflow_state, formatting_results
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        raise

async def test_workflow_state_integrity(workflow_state: WorkflowState, formatting_results: Dict[str, Any]):
    """Test the integrity of the workflow state throughout the process."""
    print("\n🔍 Testing Workflow State Integrity")
    print("-" * 40)
    
    # Check that all required attributes are present
    required_attributes = [
        'document_id', 'original_content', 'metadata', 'chunks',
        'processed_sections', 'validation_results', 'combined_content'
    ]
    
    for attr in required_attributes:
        assert hasattr(workflow_state, attr), f"Missing attribute: {attr}"
        value = getattr(workflow_state, attr)
        assert value is not None, f"Attribute {attr} is None"
        logger.info(f"   ✅ {attr}: {type(value).__name__}")
    
    # Check processed sections
    assert len(workflow_state.processed_sections) > 0, "No processed sections found"
    for section in workflow_state.processed_sections:
        assert hasattr(section, 'section_name'), "Section missing section_name"
        assert hasattr(section, 'content'), "Section missing content"
        assert len(section.content) > 0, f"Empty content in section {section.section_name}"
    
    # Check validation results
    assert len(workflow_state.validation_results) > 0, "No validation results found"
    
    # Check combined content
    assert len(workflow_state.combined_content) > 0, "No combined content found"
    
    # Check formatting results
    for format_name, result in formatting_results.items():
        assert 'formatted_content' in result, f"Missing formatted_content for {format_name}"
        assert 'formatting_metadata' in result, f"Missing formatting_metadata for {format_name}"
        assert len(result['formatted_content']) > 0, f"Empty formatted content for {format_name}"
    
    logger.info("   ✅ All workflow state integrity checks passed")

async def test_content_quality_metrics(workflow_state: WorkflowState, formatting_results: Dict[str, Any]):
    """Test content quality metrics throughout the workflow."""
    print("\n📊 Testing Content Quality Metrics")
    print("-" * 40)
    
    original_length = len(workflow_state.original_content)
    combined_length = len(workflow_state.combined_content)
    
    # Check that content is preserved and enhanced
    assert combined_length > 0, "Combined content is empty"
    
    # Calculate content expansion ratio
    expansion_ratio = combined_length / original_length if original_length > 0 else 0
    logger.info(f"   ✅ Content expansion ratio: {expansion_ratio:.2f}")
    
    # Check section coverage
    section_names = [section.section_name for section in workflow_state.processed_sections]
    expected_sections = ['introduction', 'methodology', 'results', 'conclusion']
    
    for expected in expected_sections:
        found = any(expected.lower() in name.lower() for name in section_names)
        if found:
            logger.info(f"   ✅ Section found: {expected}")
        else:
            logger.info(f"   ⚠️  Section not found: {expected}")
    
    # Check formatting quality
    for format_name, result in formatting_results.items():
        metadata = result['formatting_metadata']
        confidence = metadata.get('formatting_confidence', 0.0)
        transformations = len(metadata.get('transformations_applied', []))
        
        logger.info(f"   ✅ {format_name} confidence: {confidence:.2f}")
        logger.info(f"   ✅ {format_name} transformations: {transformations}")
        
        # Basic quality checks
        formatted_content = result['formatted_content']
        if format_name == 'markdown':
            assert '##' in formatted_content, "Missing markdown headers"
        elif format_name == 'latex':
            print(f"LaTeX content preview: {formatted_content[:200]}...")  # Debug output
            assert '\\documentclass' in formatted_content, "Missing LaTeX document class"
            assert '\\section{' in formatted_content, "Missing LaTeX sections"
    
    logger.info("   ✅ All content quality checks passed")

async def main():
    """Run the complete end-to-end test suite."""
    print("🎯 Starting End-to-End Document Processing Test")
    print("=" * 60)
    
    try:
        # Run complete workflow
        workflow_state, formatting_results = await test_complete_workflow()
        
        # Test workflow integrity
        await test_workflow_state_integrity(workflow_state, formatting_results)
        
        # Test content quality
        await test_content_quality_metrics(workflow_state, formatting_results)
        
        print("\n🎉 End-to-End Test Completed Successfully!")
        print("=" * 60)
        
        logger.info("")
        logger.info("📊 Final Summary:")
        logger.info(f"   ✅ Document ID: {workflow_state.document_id}")
        logger.info(f"   ✅ Original content: {len(workflow_state.original_content)} chars")
        logger.info(f"   ✅ Processed sections: {len(workflow_state.processed_sections)}")
        logger.info(f"   ✅ Validation results: {len(workflow_state.validation_results)}")
        logger.info(f"   ✅ Combined content: {len(workflow_state.combined_content)} chars")
        logger.info(f"   ✅ Output formats: {len(formatting_results)}")
        
        logger.info("")
        logger.info("🚀 Workflow Components Tested:")
        logger.info("   ✅ Template Processing: PASSED")
        logger.info("   ✅ Validation: PASSED")
        logger.info("   ✅ Document Combination: PASSED")
        logger.info("   ✅ Output Formatting: PASSED")
        logger.info("   ✅ State Integrity: PASSED")
        logger.info("   ✅ Content Quality: PASSED")
        
        logger.info("")
        logger.info("🎯 Next Steps:")
        logger.info("   1. Integrate into main workflow orchestrator")
        logger.info("   2. Add performance benchmarking")
        logger.info("   3. Implement LLM-based enhancements")
        logger.info("   4. Add more output formats and validation rules")
        
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())