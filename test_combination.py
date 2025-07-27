"""
Test script for document combination functionality
"""

from document_processor_v2 import DocumentProcessor

def test_document_combination():
    """Test the document combination feature"""
    
    # Initialize processor
    processor = DocumentProcessor('config_v2.yaml')
    
    # Test different combination strategies
    strategies = ['merge', 'interleave', 'append', 'smart_merge', 'bitcoin_merge']
    
    for strategy in strategies:
        print(f"\n🔄 Testing {strategy} strategy...")
        
        try:
            result = processor.combine_documents(
                source1='bitcoin_whitepaper.tex',
                source2='test_document2.tex',
                strategy=strategy,
                template='bitcoin'
            )
            
            print(f"✅ {strategy} completed successfully")
            print(f"📄 Final document: {result['final_document']}")
            print(f"📊 Sections: {result['processed_sections']}")
            print(f"📁 Session: {result['session_path']}")
            
        except Exception as e:
            print(f"❌ {strategy} failed: {e}")

def test_single_document():
    """Test single document processing"""
    
    processor = DocumentProcessor('config_v2.yaml')
    
    print("\n📄 Testing single document processing...")
    
    try:
        result = processor.process_document(
            source='bitcoin_whitepaper.tex',
            template='bitcoin'
        )
        
        print("✅ Single document processing completed")
        print(f"📄 Final document: {result['final_document']}")
        print(f"📊 Sections: {result['processed_sections']}")
        
    except Exception as e:
        print(f"❌ Single document processing failed: {e}")

if __name__ == "__main__":
    print("🧪 Document Processing System Test")
    print("=" * 50)
    
    # Test single document first
    test_single_document()
    
    # Test document combination
    test_document_combination()
    
    print("\n🎯 Testing completed!")