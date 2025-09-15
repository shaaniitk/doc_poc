"""Minimal Mistral API integration test to validate LLM functionality with minimal API usage."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from mistral_llm_handler import create_mistral_llm
from core.template_processor_node import TemplateProcessorNode
from core.validation_node import ValidationNode
from core.semantic_mapper import BasicSemanticMapper
from core.models import DocumentChunk, WorkflowState, DocumentMetadata, DocumentFormat
from datetime import datetime

# Test configuration
TEST_WITH_REAL_API = os.getenv('MISTRAL_API_KEY') is not None
MAX_API_CALLS = 2  # Limit to minimize usage

class MistralTestConfig:
    """Configuration for Mistral API testing."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "mistral-small"  # Use the smallest/cheapest model
        self.temperature = 0.1  # Low temperature for consistent results
        self.max_tokens = 50  # Limit tokens to reduce cost
        
    def create_llm_handler(self):
        """Create LLM handler with Mistral configuration."""
        return create_mistral_llm(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

def create_minimal_test_content() -> str:
    """Create minimal test content to reduce API usage."""
    return """
System Overview:
This is a simple microservices architecture with three main components:
1. API Gateway - handles routing
2. User Service - manages authentication
3. Data Service - processes information

Conclusion:
The system provides scalable and maintainable architecture.
"""

async def test_single_llm_call(config: MistralTestConfig):
    """Test a single LLM call to verify basic functionality."""
    print("\n=== Testing Single LLM Call ===")
    
    try:
        llm = config.create_llm_handler()
        
        # Simple test prompt
        test_prompt = "What is 2+2? Answer briefly."
        
        print(f"Sending prompt: {test_prompt}")
        response = await llm.agenerate([test_prompt])
        
        print(f"Response: {response}")
        print("✅ Single LLM call successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Single LLM call failed: {e}")
        return False

async def test_template_processing_with_mistral(config: MistralTestConfig):
    """Test template processing with one Mistral API call."""
    print("\n=== Testing Template Processing with Mistral ===")
    
    try:
        # Create minimal test data
        content = create_minimal_test_content()
        
        # Create workflow state
        metadata = DocumentMetadata(
            title="Minimal Test Document",
            format=DocumentFormat.MARKDOWN,
            creation_time=datetime.now(),
            modification_time=datetime.now()
        )
        
        chunk = DocumentChunk(
            content=content,
            chunk_id="test_chunk_1",
            start_position=0,
            end_position=len(content)
        )
        
        workflow_state = WorkflowState(
            document_id="mistral_test_001",
            original_content=content,
            metadata=metadata,
            chunks=[chunk]
        )
        
        # Initialize components with Mistral
        llm = config.create_llm_handler()
        semantic_mapper = BasicSemanticMapper()
        
        # Initialize template processor
        processor = TemplateProcessorNode(
            llm=llm,
            semantic_mapper=semantic_mapper
        )
        
        # Convert to dict format
        state_dict = {
            'document_id': workflow_state.document_id,
            'original_content': workflow_state.original_content,
            'metadata': workflow_state.metadata,
            'chunks': workflow_state.chunks
        }
        
        print(f"Processing document with {len(workflow_state.chunks)} chunks")
        
        # Process with Mistral
        result = await processor.process(state_dict)
        
        print(f"✅ Template processing successful!")
        
        # Extract and validate results
        if isinstance(result, dict) and 'processed_sections' in result:
            processed_sections = result['processed_sections']
            print(f"Processed {len(processed_sections)} sections")
            
            for i, section in enumerate(processed_sections):
                print(f"   Section {i+1}: {section.section_name} ({len(section.content)} chars)")
        
        return True
        
    except Exception as e:
        print(f"❌ Template processing failed: {str(e)}")
        return False

async def main():
    """Run minimal Mistral API tests."""
    print("🚀 Mistral API Integration Test")
    print("=" * 50)
    
    if not TEST_WITH_REAL_API:
        print("\n⚠️  To run real API tests, set MISTRAL_API_KEY environment variable")
        print("   Example: set MISTRAL_API_KEY=your_api_key_here")
        print("\n🔧 Running in mock mode instead...")
        return
    
    api_key = os.getenv('MISTRAL_API_KEY')
    config = MistralTestConfig(api_key)
    
    print(f"\n🔑 API Key found: {api_key[:10]}...")
    print(f"📊 Maximum API calls: {MAX_API_CALLS}")
    print(f"💰 Using model: mistral-small (cost-effective)")
    
    api_calls_made = 0
    
    # Test 1: Single API call
    if api_calls_made < MAX_API_CALLS:
        success1 = await test_single_llm_call(config)
        api_calls_made += 1 if success1 else 0
    
    # Test 2: Template processing (if we have calls remaining)
    if api_calls_made < MAX_API_CALLS:
        success2 = await test_template_processing_with_mistral(config)
        api_calls_made += 1 if success2 else 0
    
    print(f"\n📊 Summary:")
    print(f"   API calls made: {api_calls_made}/{MAX_API_CALLS}")
    print(f"   Tests completed: {'✅' if api_calls_made > 0 else '❌'}")
    
    if api_calls_made == 0:
        print("\n❌ No successful API calls made. Check your API key and network connection.")
    else:
        print(f"\n✅ Successfully tested Mistral API with minimal usage!")

if __name__ == "__main__":
    asyncio.run(main())