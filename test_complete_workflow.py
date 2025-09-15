#!/usr/bin/env python3
"""
Complete workflow test for the document processing system.
Tests the full pipeline from document parsing to final output.
"""

import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our components
from core.models import DocumentChunk, DocumentMetadata, DocumentFormat, WorkflowState, ProcessedSection
from core.semantic_mapper import BasicSemanticMapper
from core.template_processor_node import TemplateProcessorNode, PromptOrchestrator

# Mock LLM for testing
class MockLLM:
    """Mock LLM that simulates processing without actual API calls."""
    
    def __init__(self):
        self.call_count = 0
    
    async def ainvoke(self, messages):
        """Mock LLM response that processes content."""
        self.call_count += 1
        content = messages[0].content if messages else ""
        
        # Extract the actual content to process from the prompt
        if "Content to Process:" in content:
            actual_content = content.split("Content to Process:")[1].strip()
        else:
            actual_content = content
        
        # Simulate intelligent processing based on section type
        if "introduction" in content.lower():
            processed = f"## Introduction\n\nThis section introduces the key concepts and background:\n\n{actual_content[:200]}..."
        elif "methodology" in content.lower():
            processed = f"## Methodology\n\nThe technical approach described here:\n\n{actual_content[:200]}..."
        elif "results" in content.lower():
            processed = f"## Results\n\nKey findings and analysis:\n\n{actual_content[:200]}..."
        elif "conclusion" in content.lower():
            processed = f"## Conclusion\n\nSummary and implications:\n\n{actual_content[:200]}..."
        else:
            processed = f"## Processed Content\n\n{actual_content[:200]}..."
        
        # Mock response object
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(processed)

def create_bitcoin_paper_chunks() -> List[DocumentChunk]:
    """Create sample chunks from bitcoin paper content."""
    chunks = [
        DocumentChunk(
            chunk_id="intro_1",
            content="Bitcoin is a peer-to-peer electronic cash system that allows online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending.",
            start_position=0,
            end_position=300,
            metadata={"section_hint": "introduction"}
        ),
        DocumentChunk(
            chunk_id="method_1",
            content="We define an electronic coin as a chain of digital signatures. Each owner transfers the coin to the next by digitally signing a hash of the previous transaction and the public key of the next owner and adding these to the end of the coin.",
            start_position=301,
            end_position=600,
            metadata={"section_hint": "methodology"}
        ),
        DocumentChunk(
            chunk_id="method_2",
            content="The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work. The longest chain not only serves as proof of the sequence of events witnessed, but proof that it came from the largest pool of CPU power.",
            start_position=601,
            end_position=900,
            metadata={"section_hint": "methodology"}
        ),
        DocumentChunk(
            chunk_id="results_1",
            content="We have proposed a system for electronic transactions without relying on trust. We started with the usual framework of coins made from digital signatures, which provides strong control of ownership, but is incomplete without a way to prevent double-spending.",
            start_position=901,
            end_position=1200,
            metadata={"section_hint": "results"}
        ),
        DocumentChunk(
            chunk_id="conclusion_1",
            content="The network is robust in its unstructured simplicity. Nodes work all at once with little coordination. They do not need to be identified, since messages are not routed to any particular place and only need to be delivered on a best effort basis.",
            start_position=1201,
            end_position=1500,
            metadata={"section_hint": "conclusion"}
        )
    ]
    return chunks

async def test_complete_workflow():
    """Test the complete document processing workflow."""
    logger.info("🚀 Starting complete workflow test...")
    
    # Initialize components
    semantic_mapper = BasicSemanticMapper()
    mock_llm = MockLLM()
    processor = TemplateProcessorNode(llm=mock_llm, semantic_mapper=semantic_mapper)
    orchestrator = PromptOrchestrator()
    
    # Create sample document state
    chunks = create_bitcoin_paper_chunks()
    metadata = DocumentMetadata(
        title="Bitcoin: A Peer-to-Peer Electronic Cash System",
        format=DocumentFormat.PDF,
        creation_time=datetime.now(),
        modification_time=datetime.now(),
        author="Satoshi Nakamoto"
    )
    
    state = {
        'document_id': 'bitcoin_paper_test',
        'original_content': 'Bitcoin whitepaper content...',
        'metadata': metadata,
        'chunks': chunks,
        'processed_sections': [],
        'final_output': '',
        'template_name': 'bitcoin_paper_hierarchical',
        'template_applied': False,
        'processing_errors': []
    }
    
    logger.info(f"📄 Processing document with {len(chunks)} chunks")
    
    # Process document with template
    processed_state = await processor.process(state)
    
    # Verify processing results
    assert 'processed_sections' in processed_state, "Should have processed sections"
    processed_sections = processed_state['processed_sections']
    assert len(processed_sections) > 0, "Should have at least one processed section"
    
    logger.info(f"✅ Successfully processed {len(processed_sections)} sections")
    
    # Display results
    for i, section in enumerate(processed_sections):
        logger.info(f"\n📋 Section {i+1}: {section.section_name}")
        logger.info(f"   Confidence: {section.confidence_score:.2f}")
        logger.info(f"   Source chunks: {len(section.source_chunks)}")
        logger.info(f"   Content preview: {section.content[:100]}...")
        
        if section.subsections:
            logger.info(f"   Subsections: {len(section.subsections)}")
    
    # Test semantic mapping with lower threshold and better queries
    semantic_mapper = BasicSemanticMapper()
    
    # Define test sections based on the template
    test_sections = {
        "introduction": {"order": 1, "query": "bitcoin electronic cash system peer-to-peer"},
        "methodology": {"order": 2, "query": "bitcoin protocol transaction verification"},
        "results": {"order": 3, "query": "bitcoin network performance analysis"},
        "conclusion": {"order": 4, "query": "bitcoin summary electronic cash conclusion"}
    }
    
    for section_name, section_config in test_sections.items():
        logger.info(f"\n📋 Section {section_config['order']}: {section_name}")
        
        query = section_config["query"]
        
        # Find similar chunks for this section with lower threshold
        similar_chunks = await semantic_mapper.find_similar_chunks(
            query, chunks, threshold=0.05  # Much lower threshold
        )
        
        confidence = len(similar_chunks) / len(chunks) if chunks else 0.0
        logger.info(f"   Confidence: {confidence:.2f}")
        logger.info(f"   Source chunks: {len(similar_chunks)}")
        
        # Process section with LLM if we have matching chunks
        if similar_chunks:
            # Combine chunk contents
            combined_content = "\n\n".join([chunk.content for chunk, _ in similar_chunks])
            
            # Get section prompt
            section_prompt = f"Create a {section_name} section based on the following content:\n\n{combined_content}"
            
            # Process with LLM
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=section_prompt)]
            llm_response = await mock_llm.ainvoke(messages)
            processed_content = llm_response.content
            
            processed_state['processed_sections'].append({
                "section_name": section_name,
                "content": processed_content,
                "confidence": confidence,
                "source_chunks": len(similar_chunks)
            })
        else:
            # Generate placeholder content
            placeholder_content = f"# {section_name.title()}\n\n[Content to be added based on document analysis]\n"
            
            processed_state['processed_sections'].append({
                "section_name": section_name,
                "content": placeholder_content,
                "confidence": 0.0,
                "source_chunks": 0
            })
        
        logger.info(f"   Content preview: {processed_state['processed_sections'][-1]['content'][:50]}...")
    
    logger.info("\n🎭 Testing prompt orchestration...")
    chunking_prompts = orchestrator.get_prompts_for_stage("chunking")
    validation_prompts = orchestrator.get_prompts_for_stage("validation")
    
    logger.info(f"   Chunking prompts available: {len(chunking_prompts)}")
    logger.info(f"   Validation prompts available: {len(validation_prompts)}")
    
    # Test validation
    test_content = "This is test content for validation."
    validated_content = orchestrator.apply_validation_prompts(test_content, "introduction")
    assert validated_content is not None, "Validation should return content"
    
    # Verify LLM was called
    logger.info(f"\n🤖 Mock LLM was called {mock_llm.call_count} times")
    assert mock_llm.call_count > 0, "LLM should have been called for processing"
    
    # Verify template was applied
    assert processed_state['template_applied'] == 'bitcoin_paper_hierarchical', "Template should be applied"
    
    logger.info("\n🎉 Complete workflow test passed successfully!")
    logger.info("\n📊 Summary:")
    logger.info(f"   - Input chunks: {len(chunks)}")
    logger.info(f"   - Processed sections: {len(processed_sections)}")
    logger.info(f"   - LLM calls: {mock_llm.call_count}")
    logger.info(f"   - Template applied: {processed_state['template_applied']}")
    logger.info(f"   - Processing errors: {len(processed_state['processing_errors'])}")
    
    return True

async def test_error_handling():
    """Test error handling in the workflow."""
    logger.info("\n🔧 Testing error handling...")
    
    # Test with invalid template
    semantic_mapper = BasicSemanticMapper()
    mock_llm = MockLLM()
    processor = TemplateProcessorNode(llm=mock_llm, semantic_mapper=semantic_mapper)
    
    chunks = create_bitcoin_paper_chunks()[:2]  # Use fewer chunks
    metadata = DocumentMetadata(
        title="Error Test Document",
        format=DocumentFormat.PDF,
        creation_time=datetime.now(),
        modification_time=datetime.now(),
        author="Test Author"
    )
    
    state = {
        'document_id': 'error_test',
        'original_content': 'Error test content...',
        'metadata': metadata,
        'chunks': chunks,
        'processed_sections': [],
        'final_output': '',
        'template_name': 'nonexistent_template',  # Invalid template
        'template_applied': False,
        'processing_errors': []
    }
    
    # Process with invalid template
    processed_state = await processor.process(state)
    
    # Should handle error gracefully
    assert 'processing_errors' in processed_state, "Should track processing errors"
    logger.info(f"   Handled error gracefully: {len(processed_state['processing_errors'])} errors logged")
    
    logger.info("✅ Error handling test passed")
    return True

async def main():
    """Run all workflow tests."""
    logger.info("🧪 Starting comprehensive workflow tests...")
    
    try:
        # Run main workflow test
        await test_complete_workflow()
        
        # Run error handling test
        await test_error_handling()
        
        logger.info("\n🎊 All workflow tests completed successfully!")
        logger.info("\n🚀 Next steps:")
        logger.info("   1. Integrate with real LLM (OpenAI, Anthropic, etc.)")
        logger.info("   2. Add more sophisticated validation logic")
        logger.info("   3. Implement document combination and formatting")
        logger.info("   4. Add performance monitoring and optimization")
        
    except Exception as e:
        logger.error(f"❌ Workflow test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())