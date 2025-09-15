"""Test validation and quality control functionality."""

import asyncio
import logging
from datetime import datetime
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our components
from core.models import DocumentChunk, DocumentMetadata, DocumentFormat
from core.semantic_mapper import BasicSemanticMapper
from core.template_processor_node import TemplateProcessorNode
from core.validation_node import ValidationNode, QualityControlNode, ValidationLevel

# Mock LLM for testing
class MockLLM:
    """Mock LLM that simulates processing and validation."""

    def __init__(self):
        self.call_count = 0

    async def ainvoke(self, messages):
        """Mock LLM response that processes content."""
        self.call_count += 1
        content = messages[0].content if messages else ""
        
        # Simulate different responses based on content type
        if "validate" in content.lower() or "validation" in content.lower():
            # This is a validation request
            return self._create_validation_response(content)
        else:
            # This is a content generation request
            return self._create_content_response(content)
    
    def _create_validation_response(self, content: str):
        """Create a mock validation response."""
        
        # Analyze the content being validated
        if "introduction" in content.lower():
            if len(content) > 200:  # Good content
                response = """
VALID: true
CONFIDENCE: 0.85
ISSUES: None
SUGGESTIONS: Consider adding more specific examples
IMPROVED_CONTENT: None
"""
            else:  # Poor content
                response = """
VALID: false
CONFIDENCE: 0.45
ISSUES: Too brief, lacks context, missing key concepts
SUGGESTIONS: Expand introduction, add background context, include purpose statement
IMPROVED_CONTENT: ## Introduction

This comprehensive introduction provides essential background and context for understanding the key concepts and methodologies presented in this document. The purpose is to establish a foundation for the detailed analysis that follows.
"""
        elif "methodology" in content.lower():
            response = """
VALID: true
CONFIDENCE: 0.78
ISSUES: Could use more technical detail
SUGGESTIONS: Add specific implementation steps, include technical specifications
IMPROVED_CONTENT: None
"""
        elif "results" in content.lower():
            response = """
VALID: false
CONFIDENCE: 0.60
ISSUES: Missing quantitative data, lacks analysis depth
SUGGESTIONS: Include specific metrics, add comparative analysis, provide data visualization
IMPROVED_CONTENT: ## Results

Key findings and comprehensive analysis:

The evaluation demonstrates significant improvements across multiple metrics. Performance increased by 40% compared to baseline measurements, with consistency maintained across different test scenarios.
"""
        else:
            response = """
VALID: true
CONFIDENCE: 0.70
ISSUES: None
SUGGESTIONS: Consider adding more detail
IMPROVED_CONTENT: None
"""
        
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(response)
    
    def _create_content_response(self, content: str):
        """Create a mock content generation response."""
        
        # Extract the actual content from the prompt
        actual_content = content.split("Create a")[-1] if "Create a" in content else content
        actual_content = actual_content[:300]  # Limit length
        
        # Generate appropriate content based on section type
        if "introduction" in content.lower():
            processed = f"## Introduction\n\nThis section introduces the key concepts:\n\n{actual_content[:150]}..."
        elif "methodology" in content.lower():
            processed = f"## Methodology\n\nThe technical approach described here:\n\n{actual_content[:150]}..."
        elif "results" in content.lower():
            processed = f"## Results\n\nKey findings and analysis:\n\n{actual_content[:150]}..."
        elif "conclusion" in content.lower():
            processed = f"## Conclusion\n\nSummary and implications:\n\n{actual_content[:150]}..."
        else:
            processed = f"## Processed Content\n\n{actual_content[:150]}..."

        class MockResponse:
            def __init__(self, content):
                self.content = content

        return MockResponse(processed)

def create_test_chunks() -> List[DocumentChunk]:
    """Create test chunks for validation testing."""
    chunks = [
        DocumentChunk(
            chunk_id="test_1",
            content="Bitcoin is a peer-to-peer electronic cash system that allows online payments to be sent directly from one party to another without going through a financial institution.",
            start_position=0,
            end_position=150,
            metadata={"section": "introduction", "importance": 0.9}
        ),
        DocumentChunk(
            chunk_id="test_2",
            content="The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work.",
            start_position=151,
            end_position=300,
            metadata={"section": "methodology", "importance": 0.8}
        ),
        DocumentChunk(
            chunk_id="test_3",
            content="The longest chain not only serves as proof of the sequence of events witnessed, but proof that it came from the largest pool of CPU power.",
            start_position=301,
            end_position=450,
            metadata={"section": "results", "importance": 0.7}
        )
    ]
    return chunks

async def test_validation_workflow():
    """Test the complete validation workflow."""
    logger.info("🔍 Testing validation and quality control workflow...")
    
    # Setup components
    mock_llm = MockLLM()
    semantic_mapper = BasicSemanticMapper()
    processor = TemplateProcessorNode(mock_llm, semantic_mapper)
    validation_node = ValidationNode(mock_llm, ValidationLevel.STANDARD)
    quality_control = QualityControlNode(mock_llm, validation_node)
    
    # Create test data
    chunks = create_test_chunks()
    metadata = DocumentMetadata(
        title="Test Document for Validation",
        format=DocumentFormat.PDF,
        creation_time=datetime.now(),
        modification_time=datetime.now(),
        author="Test Author"
    )
    
    state = {
        'document_id': 'validation_test',
        'original_content': 'Test content for validation...',
        'metadata': metadata,
        'chunks': chunks,
        'processed_sections': [],
    }
    
    logger.info(f"📄 Processing document with {len(chunks)} chunks")
    
    # Step 1: Process document with template
    processed_state = await processor.process(state)
    logger.info(f"✅ Template processing completed: {len(processed_state.get('processed_sections', []))} sections")
    
    # Step 2: Validate processed content
    logger.info("\n🔍 Running validation on processed sections...")
    logger.info(f"Debug: processed_state type: {type(processed_state)}")
    logger.info(f"Debug: processed_sections type: {type(processed_state.get('processed_sections', []))}")
    logger.info(f"Debug: processed_sections length: {len(processed_state.get('processed_sections', []))}")
    if processed_state.get('processed_sections'):
        logger.info(f"Debug: first section type: {type(processed_state['processed_sections'][0])}")
    validation_results = await validation_node.validate_workflow_state(processed_state)
    
    logger.info(f"📊 Validation completed for {len(validation_results)} sections:")
    for section_name, result in validation_results.items():
        status = "✅ VALID" if result.is_valid else "❌ INVALID"
        logger.info(f"   {section_name}: {status} (confidence: {result.confidence_score:.2f})")
        if result.issues:
            logger.info(f"     Issues: {', '.join(result.issues)}")
        if result.suggestions:
            logger.info(f"     Suggestions: {', '.join(result.suggestions)}")
    
    # Step 3: Apply quality control improvements
    logger.info("\n🛠️ Applying quality control improvements...")
    improved_state = await quality_control.improve_document_quality(processed_state, validation_results)
    
    # Step 4: Verify improvements
    logger.info("\n📈 Quality control results:")
    
    # Count improvements
    improvements_applied = 0
    validation_passed = 0
    
    if hasattr(improved_state, 'processed_sections'):
        sections = improved_state.processed_sections
        if isinstance(sections, list):
            for section in sections:
                if isinstance(section, dict):
                    if section.get('validation_applied'):
                        improvements_applied += 1
                    if section.get('validation_passed'):
                        validation_passed += 1
    
    logger.info(f"   Sections improved: {improvements_applied}")
    logger.info(f"   Sections passed validation: {validation_passed}")
    logger.info(f"   Total LLM calls: {mock_llm.call_count}")
    
    # Verify validation results are stored
    if isinstance(improved_state, dict):
        assert 'validation_results' in improved_state, "Validation results should be stored in state"
    else:
        assert hasattr(improved_state, 'validation_results'), "Validation results should be stored in state"
    assert len(validation_results) > 0, "Should have validation results"
    
    logger.info("\n🎉 Validation workflow test completed successfully!")
    
    return {
        'validation_results': validation_results,
        'improvements_applied': improvements_applied,
        'llm_calls': mock_llm.call_count,
        'sections_processed': len(validation_results)
    }

async def test_validation_levels():
    """Test different validation levels."""
    logger.info("\n🎚️ Testing different validation levels...")
    
    mock_llm = MockLLM()
    
    # Test each validation level
    levels = [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.STRICT]
    
    for level in levels:
        logger.info(f"\n   Testing {level.value} validation...")
        validation_node = ValidationNode(mock_llm, level)
        
        # Create a test section
        test_section = type('TestSection', (), {
            'section_name': 'introduction',
            'content': 'Short intro.',  # Intentionally brief
            'confidence': 0.5,
            'source_chunks': 1
        })()
        
        result = await validation_node.validate_section(
            test_section, [], {'min_length': 100}
        )
        
        logger.info(f"     Result: valid={result.is_valid}, confidence={result.confidence_score:.2f}")
        logger.info(f"     Issues: {len(result.issues)}, Suggestions: {len(result.suggestions)}")
    
    logger.info("\n✅ Validation levels test completed")

async def main():
    """Run all validation tests."""
    try:
        # Test main validation workflow
        workflow_results = await test_validation_workflow()
        
        # Test validation levels
        await test_validation_levels()
        
        logger.info("\n🎊 All validation tests completed successfully!")
        logger.info("\n📊 Summary:")
        logger.info(f"   Sections validated: {workflow_results['sections_processed']}")
        logger.info(f"   Improvements applied: {workflow_results['improvements_applied']}")
        logger.info(f"   Total LLM calls: {workflow_results['llm_calls']}")
        
        logger.info("\n🚀 Next steps:")
        logger.info("   1. Integrate validation into main workflow")
        logger.info("   2. Add more sophisticated validation criteria")
        logger.info("   3. Implement document combination logic")
        logger.info("   4. Create output formatting system")
        
    except Exception as e:
        logger.error(f"❌ Validation test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())