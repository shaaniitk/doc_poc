#!/usr/bin/env python3
"""
Test script for document combination functionality.
Tests the DocumentCombinationNode with processed sections and validation results.
"""

import asyncio
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules
from core.models import ProcessedSection
from core.combination_node import DocumentCombinationNode, CombinationStrategy
from core.validation_node import ValidationResult

def create_test_processed_sections() -> List[ProcessedSection]:
    """Create test processed sections for combination testing."""
    from core.models import DocumentChunk
    
    # Create dummy chunks for testing
    def create_dummy_chunks(count: int) -> List[DocumentChunk]:
        return [
            DocumentChunk(
                content=f"Chunk {i+1} content",
                chunk_id=f"chunk_{i+1}",
                start_position=i*100,
                end_position=(i+1)*100
            ) for i in range(count)
        ]
    
    sections = [
        ProcessedSection(
            section_name="introduction",
            content="This document provides a comprehensive overview of the system architecture and implementation details.",
            confidence_score=0.85,
            source_chunks=create_dummy_chunks(3)
        ),
        ProcessedSection(
            section_name="methodology",
            content="The methodology section describes the approach taken to solve the problem, including data collection and analysis techniques.",
            confidence_score=0.90,
            source_chunks=create_dummy_chunks(4)
        ),
        ProcessedSection(
            section_name="results",
            content="The results demonstrate significant improvements in performance metrics, with a 25% increase in efficiency.",
            confidence_score=0.75,
            source_chunks=create_dummy_chunks(2)
        ),
        ProcessedSection(
            section_name="conclusion",
            content="In conclusion, the proposed solution effectively addresses the identified challenges and provides a solid foundation for future work.",
            confidence_score=0.80,
            source_chunks=create_dummy_chunks(2)
        )
    ]
    return sections

def create_test_validation_results() -> Dict[str, ValidationResult]:
    """Create test validation results."""
    return {
        "introduction": ValidationResult(
            is_valid=True,
            confidence_score=0.85,
            issues=[],
            suggestions=["Consider adding more context"]
        ),
        "methodology": ValidationResult(
            is_valid=True,
            confidence_score=0.90,
            issues=[],
            suggestions=[]
        ),
        "results": ValidationResult(
            is_valid=False,
            confidence_score=0.60,
            issues=["Missing statistical significance"],
            suggestions=["Add error bars and p-values"]
        ),
        "conclusion": ValidationResult(
            is_valid=True,
            confidence_score=0.80,
            issues=[],
            suggestions=["Consider adding future work section"]
        )
    }

async def test_basic_combination():
    """Test basic document combination without validation."""
    logger.info("\n🔄 Testing basic document combination...")
    
    # Create combination node
    combination_node = DocumentCombinationNode(CombinationStrategy.WEAVE)
    
    # Create test state
    test_state = {
        'processed_sections': create_test_processed_sections()
    }
    
    # Perform combination
    result = await combination_node.combine_sections(test_state)
    
    # Verify results
    assert result.sections_combined == 4, f"Expected 4 sections, got {result.sections_combined}"
    assert result.strategy_used == "weave", f"Expected 'weave' strategy, got {result.strategy_used}"
    assert len(result.combined_content) > 0, "Combined content should not be empty"
    assert result.confidence_score > 0, "Confidence score should be positive"
    
    logger.info(f"   ✅ Combined {result.sections_combined} sections")
    logger.info(f"   ✅ Strategy: {result.strategy_used}")
    logger.info(f"   ✅ Confidence: {result.confidence_score:.2f}")
    logger.info(f"   ✅ Content length: {len(result.combined_content)} characters")
    
    return result

async def test_combination_with_validation():
    """Test document combination with validation results."""
    logger.info("\n🔍 Testing combination with validation results...")
    
    # Create combination node
    combination_node = DocumentCombinationNode(CombinationStrategy.WEAVE)
    
    # Create test state and validation results
    test_state = {
        'processed_sections': create_test_processed_sections()
    }
    validation_results = create_test_validation_results()
    
    # Perform combination with validation
    result = await combination_node.combine_sections(test_state, validation_results)
    
    # Verify results
    assert result.sections_combined == 4, f"Expected 4 sections, got {result.sections_combined}"
    assert result.metadata['validation_applied'] == True, "Validation should be applied"
    assert len(result.combined_content) > 0, "Combined content should not be empty"
    
    logger.info(f"   ✅ Combined {result.sections_combined} sections with validation")
    logger.info(f"   ✅ Confidence: {result.confidence_score:.2f}")
    logger.info(f"   ✅ Validation applied: {result.metadata['validation_applied']}")
    
    # Check if low-confidence sections are handled properly
    valid_sections = sum(1 for vr in validation_results.values() if vr.is_valid)
    logger.info(f"   ✅ Valid sections: {valid_sections}/4")
    
    return result

async def test_different_strategies():
    """Test different combination strategies."""
    logger.info("\n🎯 Testing different combination strategies...")
    
    test_state = {
        'processed_sections': create_test_processed_sections()
    }
    
    strategies = [CombinationStrategy.WEAVE, CombinationStrategy.CONTRAST, CombinationStrategy.HIERARCHICAL]
    
    for strategy in strategies:
        logger.info(f"\n   Testing {strategy.value} strategy...")
        combination_node = DocumentCombinationNode(strategy)
        result = await combination_node.combine_sections(test_state)
        
        assert result.strategy_used == strategy.value, f"Strategy mismatch: expected {strategy.value}, got {result.strategy_used}"
        assert result.sections_combined == 4, f"Expected 4 sections, got {result.sections_combined}"
        
        logger.info(f"     ✅ Strategy: {result.strategy_used}")
        logger.info(f"     ✅ Confidence: {result.confidence_score:.2f}")
        logger.info(f"     ✅ Content length: {len(result.combined_content)} characters")

async def test_empty_sections():
    """Test combination with empty sections."""
    logger.info("\n🚫 Testing combination with empty sections...")
    
    combination_node = DocumentCombinationNode()
    
    # Test with empty state
    empty_state = {'processed_sections': []}
    result = await combination_node.combine_sections(empty_state)
    
    assert result.sections_combined == 0, f"Expected 0 sections, got {result.sections_combined}"
    assert result.combined_content == "", "Combined content should be empty"
    assert result.confidence_score == 0.0, "Confidence should be 0 for empty sections"
    assert "error" in result.metadata, "Should have error in metadata"
    
    logger.info("   ✅ Empty sections handled correctly")
    logger.info(f"   ✅ Error message: {result.metadata['error']}")

async def test_workflowstate_compatibility():
    """Test combination with WorkflowState objects."""
    logger.info("\n🔄 Testing WorkflowState compatibility...")
    
    from core.models import WorkflowState
    
    combination_node = DocumentCombinationNode()
    
    # Create WorkflowState object with required fields
    from core.models import DocumentMetadata, DocumentFormat
    from datetime import datetime
    
    metadata = DocumentMetadata(
        title="Test Document",
        format=DocumentFormat.MARKDOWN,
        creation_time=datetime.now(),
        modification_time=datetime.now()
    )
    
    workflow_state = WorkflowState(
        document_id="test_doc_123",
        original_content="Test content",
        metadata=metadata,
        processed_sections=create_test_processed_sections()
    )
    
    result = await combination_node.combine_sections(workflow_state)
    
    assert result.sections_combined == 4, f"Expected 4 sections, got {result.sections_combined}"
    assert len(result.combined_content) > 0, "Combined content should not be empty"
    
    logger.info("   ✅ WorkflowState compatibility verified")
    logger.info(f"   ✅ Sections combined: {result.sections_combined}")

async def main():
    """Run all combination tests."""
    logger.info("🚀 Starting Document Combination Tests")
    logger.info("=" * 50)
    
    try:
        # Run all tests
        await test_basic_combination()
        await test_combination_with_validation()
        await test_different_strategies()
        await test_empty_sections()
        await test_workflowstate_compatibility()
        
        logger.info("\n🎉 All combination tests completed successfully!")
        logger.info("\n📊 Summary:")
        logger.info("   ✅ Basic combination: PASSED")
        logger.info("   ✅ Validation integration: PASSED")
        logger.info("   ✅ Multiple strategies: PASSED")
        logger.info("   ✅ Empty sections handling: PASSED")
        logger.info("   ✅ WorkflowState compatibility: PASSED")
        
        logger.info("\n🚀 Next steps:")
        logger.info("   1. Integrate combination into main workflow")
        logger.info("   2. Add output formatting system")
        logger.info("   3. Test end-to-end document processing")
        logger.info("   4. Add more sophisticated combination strategies")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())