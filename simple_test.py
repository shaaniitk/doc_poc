"""
Simple test without LLM processing to check basic functionality
"""

import os
from document_processor_v2 import DocumentProcessor

def test_without_llm():
    """Test document processing without LLM enhancement"""
    
    # Temporarily disable LLM processing
    processor = DocumentProcessor('config_v2.yaml')
    
    # Override config to disable enhancement
    processor.config_manager.config['processing']['enable_enhancement'] = False
    
    print("📊 Original File Sizes:")
    files = ['bitcoin_whitepaper.tex', 'blockchain_security.tex']
    total_original = 0
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            total_original += size
            print(f"  {file}: {size:,} bytes")
    
    print(f"  Total Original: {total_original:,} bytes")
    
    print("\n🔄 Testing Document Combination (No LLM)...")
    
    try:
        result = processor.combine_documents(
            source1='bitcoin_whitepaper.tex',
            source2='blockchain_security.tex',
            strategy='smart_merge',
            template='bitcoin'
        )
        
        print("✅ Combination completed successfully!")
        print(f"📄 Final document: {result['final_document']}")
        print(f"📊 Sections: {result['processed_sections']}")
        
        # Analyze output
        if os.path.exists(result['final_document']):
            size = os.path.getsize(result['final_document'])
            with open(result['final_document'], 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.split('\n'))
                chars = len(content)
            
            print(f"\n📈 Combined Output Analysis:")
            print(f"  Size: {size:,} bytes")
            print(f"  Lines: {lines:,}")
            print(f"  Characters: {chars:,}")
            print(f"  Size vs Original: {((size - total_original) / total_original * 100):+.1f}%")
            
            # Quick content check
            equations = len([line for line in content.split('\n') if 'equation' in line])
            sections = len([line for line in content.split('\n') if line.startswith('\\section')])
            
            print(f"  Equations found: {equations}")
            print(f"  Sections found: {sections}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_different_strategies():
    """Test different combination strategies without LLM"""
    
    processor = DocumentProcessor('config_v2.yaml')
    processor.config_manager.config['processing']['enable_enhancement'] = False
    
    strategies = ['merge', 'append', 'smart_merge']
    
    print("\n📊 Strategy Comparison (No LLM):")
    print(f"{'Strategy':<12} {'Size (KB)':<10} {'Lines':<8} {'Sections':<10}")
    print("-" * 45)
    
    for strategy in strategies:
        try:
            result = processor.combine_documents(
                source1='bitcoin_whitepaper.tex',
                source2='blockchain_security.tex',
                strategy=strategy,
                template='bitcoin'
            )
            
            if os.path.exists(result['final_document']):
                size = os.path.getsize(result['final_document'])
                with open(result['final_document'], 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                
                size_kb = size / 1024
                print(f"{strategy:<12} {size_kb:<10.1f} {lines:<8} {result['processed_sections']:<10}")
            
        except Exception as e:
            print(f"{strategy:<12} {'ERROR':<10} {'ERROR':<8} {'ERROR':<10}")

if __name__ == "__main__":
    print("🧪 Simple Document Processing Test")
    print("=" * 40)
    
    # Test basic combination
    result = test_without_llm()
    
    # Test different strategies
    test_different_strategies()
    
    print("\n🎯 Simple test completed!")