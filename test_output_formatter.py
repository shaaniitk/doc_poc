"""Test suite for OutputFormatterNode functionality."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from core.output_formatter_node import OutputFormatterNode, FormattingStrategy
from core.models import WorkflowState, DocumentMetadata, DocumentFormat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_combined_content() -> str:
    """Create test combined content for formatting."""
    return """
Introduction
This document provides a comprehensive overview of the system architecture and implementation details. The system is designed to handle complex document processing workflows with high efficiency and reliability.

Methodology
The methodology section describes the approach taken to solve the problem, including data collection and analysis techniques. We employed a multi-stage processing pipeline that ensures quality and consistency throughout the document transformation process.

Results
The results demonstrate significant improvements in performance metrics, with a 25% increase in efficiency compared to previous implementations. The system successfully processed over 1000 documents with 99.5% accuracy.

Conclusion
In conclusion, the proposed solution effectively addresses the identified challenges and provides a solid foundation for future work. The implementation demonstrates both technical excellence and practical utility.
"""

async def test_basic_formatting():
    """Test basic formatting functionality."""
    print("\n🎨 Testing basic formatting functionality...")
    
    formatter = OutputFormatterNode()
    combined_content = create_test_combined_content()
    
    # Test different formats
    formats_to_test = [
        (DocumentFormat.MARKDOWN, "markdown"),
        (DocumentFormat.LATEX, "latex"),
        # Add more formats as needed
    ]
    
    for doc_format, format_name in formats_to_test:
        print(f"\n   Testing {format_name} formatting...")
        
        input_data = {
            'combined_content': combined_content,
            'target_format': doc_format,
            'formatting_options': {}
        }
        
        result = await formatter.process(input_data)
        
        assert 'formatted_content' in result, f"Missing formatted_content for {format_name}"
        assert 'formatting_metadata' in result, f"Missing formatting_metadata for {format_name}"
        assert len(result['formatted_content']) > 0, f"Empty formatted content for {format_name}"
        
        metadata = result['formatting_metadata']
        assert metadata['target_format'] == format_name, f"Wrong target format for {format_name}"
        assert metadata['original_length'] > 0, f"Invalid original length for {format_name}"
        assert metadata['formatted_length'] > 0, f"Invalid formatted length for {format_name}"
        
        logger.info(f"     ✅ Format: {format_name}")
        logger.info(f"     ✅ Original length: {metadata['original_length']} characters")
        logger.info(f"     ✅ Formatted length: {metadata['formatted_length']} characters")
        logger.info(f"     ✅ Transformations: {len(metadata['transformations_applied'])}")

async def test_markdown_formatting():
    """Test specific markdown formatting features."""
    print("\n📝 Testing markdown-specific formatting...")
    
    formatter = OutputFormatterNode()
    combined_content = create_test_combined_content()
    
    input_data = {
        'combined_content': combined_content,
        'target_format': DocumentFormat.MARKDOWN,
        'formatting_options': {}
    }
    
    result = await formatter.process(input_data)
    formatted_content = result['formatted_content']
    
    # Check for markdown-specific formatting
    assert '## Introduction' in formatted_content, "Missing markdown header for Introduction"
    assert '## Methodology' in formatted_content, "Missing markdown header for Methodology"
    assert '## Results' in formatted_content, "Missing markdown header for Results"
    assert '## Conclusion' in formatted_content, "Missing markdown header for Conclusion"
    
    logger.info("   ✅ Markdown headers properly formatted")
    logger.info("   ✅ Section structure maintained")
    
    # Print a sample of the formatted content
    lines = formatted_content.split('\n')[:10]
    logger.info(f"   ✅ Sample output: {lines[0][:50]}...")

async def test_latex_formatting():
    """Test specific LaTeX formatting features."""
    print("\n📄 Testing LaTeX-specific formatting...")
    
    formatter = OutputFormatterNode()
    combined_content = create_test_combined_content()
    
    input_data = {
        'combined_content': combined_content,
        'target_format': DocumentFormat.LATEX,
        'formatting_options': {}
    }
    
    result = await formatter.process(input_data)
    formatted_content = result['formatted_content']
    
    # Check for LaTeX-specific formatting
    assert '\\documentclass{article}' in formatted_content, "Missing LaTeX document class"
    assert '\\begin{document}' in formatted_content, "Missing LaTeX begin document"
    assert '\\end{document}' in formatted_content, "Missing LaTeX end document"
    assert '\\section{' in formatted_content, "Missing LaTeX section commands"
    
    logger.info("   ✅ LaTeX document structure created")
    logger.info("   ✅ Section commands properly formatted")
    logger.info("   ✅ Document class and packages included")

async def test_workflowstate_compatibility():
    """Test compatibility with WorkflowState objects."""
    print("\n🔄 Testing WorkflowState compatibility...")
    
    formatter = OutputFormatterNode()
    
    # Create WorkflowState object
    metadata = DocumentMetadata(
        title="Test Document",
        format=DocumentFormat.MARKDOWN,
        creation_time=datetime.now(),
        modification_time=datetime.now()
    )
    
    workflow_state = WorkflowState(
        document_id="test_doc_456",
        original_content="Original content",
        metadata=metadata
    )
    
    # Add combined_content attribute
    workflow_state.combined_content = create_test_combined_content()
    
    result = await formatter.process(workflow_state)
    
    assert 'formatted_content' in result, "Missing formatted_content for WorkflowState"
    assert 'formatting_metadata' in result, "Missing formatting_metadata for WorkflowState"
    assert len(result['formatted_content']) > 0, "Empty formatted content for WorkflowState"
    
    metadata_result = result['formatting_metadata']
    assert metadata_result['target_format'] == 'markdown', "Wrong target format from WorkflowState"
    
    logger.info("   ✅ WorkflowState compatibility verified")
    logger.info(f"   ✅ Format detected: {metadata_result['target_format']}")
    logger.info(f"   ✅ Content formatted: {metadata_result['formatted_length']} characters")

async def test_empty_content_handling():
    """Test handling of empty or missing content."""
    print("\n🚫 Testing empty content handling...")
    
    formatter = OutputFormatterNode()
    
    # Test with empty content
    input_data = {
        'combined_content': '',
        'target_format': DocumentFormat.MARKDOWN,
        'formatting_options': {}
    }
    
    result = await formatter.process(input_data)
    
    assert result['formatted_content'] == '', "Should return empty string for empty content"
    assert 'error' in result['formatting_metadata'], "Should include error in metadata"
    assert result['formatting_metadata']['error'] == 'No content to format', "Wrong error message"
    
    logger.info("   ✅ Empty content handled correctly")
    logger.info(f"   ✅ Error message: {result['formatting_metadata']['error']}")
    
    # Test with missing content key
    input_data_missing = {
        'target_format': DocumentFormat.MARKDOWN,
        'formatting_options': {}
    }
    
    result_missing = await formatter.process(input_data_missing)
    assert result_missing['formatted_content'] == '', "Should return empty string for missing content"
    
    logger.info("   ✅ Missing content key handled correctly")

async def test_formatting_strategies():
    """Test different formatting strategies."""
    print("\n🎯 Testing formatting strategies...")
    
    formatter = OutputFormatterNode()
    combined_content = create_test_combined_content()
    
    strategies = [
        (FormattingStrategy.MARKDOWN, DocumentFormat.MARKDOWN),
        (FormattingStrategy.LATEX, DocumentFormat.LATEX),
        (FormattingStrategy.HTML, 'html'),
        (FormattingStrategy.PLAIN_TEXT, 'plain_text')
    ]
    
    for strategy, doc_format in strategies:
        print(f"\n   Testing {strategy.value} strategy...")
        
        input_data = {
            'combined_content': combined_content,
            'target_format': doc_format,
            'formatting_options': {}
        }
        
        result = await formatter.process(input_data)
        
        assert result['formatting_metadata']['target_format'] == strategy.value
        assert len(result['formatted_content']) > 0
        
        transformations = result['formatting_metadata']['transformations_applied']
        assert len(transformations) > 0, f"No transformations listed for {strategy.value}"
        
        logger.info(f"     ✅ Strategy: {strategy.value}")
        logger.info(f"     ✅ Transformations: {len(transformations)}")
        logger.info(f"     ✅ Content length: {len(result['formatted_content'])} characters")

async def main():
    """Run all output formatter tests."""
    print("🚀 Starting Output Formatter Tests")
    print("=" * 50)
    
    try:
        await test_basic_formatting()
        await test_markdown_formatting()
        await test_latex_formatting()
        await test_workflowstate_compatibility()
        await test_empty_content_handling()
        await test_formatting_strategies()
        
        print("\n🎉 All output formatter tests completed successfully!")
        
        logger.info("")
        logger.info("📊 Summary:")
        logger.info("   ✅ Basic formatting: PASSED")
        logger.info("   ✅ Markdown formatting: PASSED")
        logger.info("   ✅ LaTeX formatting: PASSED")
        logger.info("   ✅ WorkflowState compatibility: PASSED")
        logger.info("   ✅ Empty content handling: PASSED")
        logger.info("   ✅ Formatting strategies: PASSED")
        
        logger.info("")
        logger.info("🚀 Next steps:")
        logger.info("   1. Integrate formatter into main workflow")
        logger.info("   2. Add LLM-based formatting enhancements")
        logger.info("   3. Test end-to-end document processing")
        logger.info("   4. Add more output formats (PDF, DOCX, etc.)")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())