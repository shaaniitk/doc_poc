"""
Comprehensive Analysis of Document Processing and Combination
"""

import os
import re
from document_processor_v2 import DocumentProcessor

def analyze_file_sizes():
    """Analyze original file sizes"""
    print("📊 Original File Analysis")
    print("=" * 50)
    
    files = ['bitcoin_whitepaper.tex', 'test_document2.tex', 'blockchain_security.tex']
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.split('\n'))
                chars = len(content)
                words = len(content.split())
            
            print(f"📄 {file}:")
            print(f"  Size: {size:,} bytes")
            print(f"  Lines: {lines:,}")
            print(f"  Characters: {chars:,}")
            print(f"  Words: {words:,}")
            print()

def test_single_document_processing():
    """Test single document processing"""
    print("🔄 Single Document Processing Test")
    print("=" * 50)
    
    processor = DocumentProcessor('config_v2.yaml')
    
    try:
        result = processor.process_document(
            source='bitcoin_whitepaper.tex',
            template='bitcoin'
        )
        
        print("✅ Single document processing completed")
        print(f"📄 Final document: {result['final_document']}")
        print(f"📊 Sections processed: {result['processed_sections']}")
        print(f"📁 Session path: {result['session_path']}")
        
        # Analyze output
        if os.path.exists(result['final_document']):
            size = os.path.getsize(result['final_document'])
            with open(result['final_document'], 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.split('\n'))
                chars = len(content)
            
            print(f"📈 Output Analysis:")
            print(f"  Size: {size:,} bytes")
            print(f"  Lines: {lines:,}")
            print(f"  Characters: {chars:,}")
        
        return result
        
    except Exception as e:
        print(f"❌ Single document processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_document_combination():
    """Test document combination with different strategies"""
    print("\n🔄 Document Combination Tests")
    print("=" * 50)
    
    processor = DocumentProcessor('config_v2.yaml')
    
    # Test combinations
    test_cases = [
        ('bitcoin_whitepaper.tex', 'blockchain_security.tex', 'smart_merge'),
        ('bitcoin_whitepaper.tex', 'blockchain_security.tex', 'merge'),
        ('bitcoin_whitepaper.tex', 'blockchain_security.tex', 'append'),
    ]
    
    results = []
    
    for source1, source2, strategy in test_cases:
        print(f"\n🧪 Testing: {strategy} strategy")
        print(f"📄 Documents: {source1} + {source2}")
        
        try:
            result = processor.combine_documents(
                source1=source1,
                source2=source2,
                strategy=strategy,
                template='bitcoin'
            )
            
            print(f"✅ {strategy} completed successfully")
            print(f"📄 Final document: {result['final_document']}")
            print(f"📊 Sections: {result['processed_sections']}")
            
            # Analyze combined output
            if os.path.exists(result['final_document']):
                size = os.path.getsize(result['final_document'])
                with open(result['final_document'], 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                    chars = len(content)
                
                print(f"📈 Combined Output:")
                print(f"  Size: {size:,} bytes")
                print(f"  Lines: {lines:,}")
                print(f"  Characters: {chars:,}")
                
                result['output_stats'] = {
                    'size': size,
                    'lines': lines,
                    'characters': chars
                }
            
            results.append((strategy, result))
            
        except Exception as e:
            print(f"❌ {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def analyze_content_preservation(original_files, processed_file):
    """Analyze content preservation in detail"""
    print(f"\n🔍 Content Preservation Analysis")
    print("=" * 50)
    
    # Read original files
    original_content = ""
    original_stats = {'equations': 0, 'code_blocks': 0, 'lists': 0, 'figures': 0}
    
    for file in original_files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                original_content += content + "\n"
    
    # Read processed file
    if not os.path.exists(processed_file):
        print(f"❌ Processed file not found: {processed_file}")
        return
    
    with open(processed_file, 'r', encoding='utf-8') as f:
        processed_content = f.read()
    
    # Clean content for comparison
    def clean_content(text):
        # Remove LaTeX structure
        text = re.sub(r'\\documentclass.*?\\begin\{document\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\end\{document\}', '', text)
        text = re.sub(r'\\title\{.*?\}', '', text)
        text = re.sub(r'\\author\{.*?\}', '', text)
        text = re.sub(r'\\date\{.*?\}', '', text)
        text = re.sub(r'\\maketitle', '', text)
        text = re.sub(r'% --- .+? ---', '', text)
        text = re.sub(r'% .+', '', text)
        return re.sub(r'\n\s*\n', '\n', text).strip()
    
    original_clean = clean_content(original_content)
    processed_clean = clean_content(processed_content)
    
    # Calculate preservation metrics
    original_chars = len(original_clean)
    processed_chars = len(processed_clean)
    change_percent = ((processed_chars - original_chars) / original_chars) * 100 if original_chars > 0 else 0
    
    print(f"📊 Character Analysis:")
    print(f"  Original: {original_chars:,} characters")
    print(f"  Processed: {processed_chars:,} characters")
    print(f"  Change: {change_percent:+.1f}%")
    
    # Count LaTeX environments
    patterns = {
        'equations': r'\\begin\{equation\}.*?\\end\{equation\}',
        'code_blocks': r'\\begin\{verbatim\}.*?\\end\{verbatim\}',
        'lists': r'\\begin\{enumerate\}.*?\\end\{enumerate\}',
        'figures': r'\\begin\{figure\}.*?\\end\{figure\}'
    }
    
    print(f"\n📋 LaTeX Environment Preservation:")
    for env_name, pattern in patterns.items():
        original_count = len(re.findall(pattern, original_content, re.DOTALL))
        processed_count = len(re.findall(pattern, processed_content, re.DOTALL))
        
        if original_count > 0:
            preservation_rate = (processed_count / original_count) * 100
            print(f"  {env_name.title()}: {original_count} → {processed_count} ({preservation_rate:.1f}% preserved)")
        else:
            print(f"  {env_name.title()}: {original_count} → {processed_count} (N/A)")
    
    # Check for specific content
    print(f"\n🔍 Specific Content Check:")
    
    key_content = [
        ('Bitcoin equations', r'q_z = \\begin\{cases\}'),
        ('C code implementation', r'AttackerSuccessProbability'),
        ('Lambda equation', r'\\lambda = z \\frac\{q\}\{p\}'),
        ('Poisson distribution', r'Poisson'),
        ('Hash functions', r'SHA-256'),
        ('Cryptographic', r'cryptographic')
    ]
    
    for content_name, pattern in key_content:
        original_found = bool(re.search(pattern, original_content, re.IGNORECASE))
        processed_found = bool(re.search(pattern, processed_content, re.IGNORECASE))
        
        status = "✅" if (original_found and processed_found) or (not original_found and not processed_found) else "❌"
        print(f"  {status} {content_name}: {'Found' if processed_found else 'Missing'}")

def compare_combination_strategies(results):
    """Compare different combination strategies"""
    print(f"\n📊 Combination Strategy Comparison")
    print("=" * 50)
    
    if not results:
        print("❌ No results to compare")
        return
    
    print(f"{'Strategy':<15} {'Size (KB)':<12} {'Lines':<8} {'Characters':<12} {'Sections':<10}")
    print("-" * 65)
    
    for strategy, result in results:
        if 'output_stats' in result:
            stats = result['output_stats']
            size_kb = stats['size'] / 1024
            print(f"{strategy:<15} {size_kb:<12.1f} {stats['lines']:<8} {stats['characters']:<12,} {result['processed_sections']:<10}")
        else:
            print(f"{strategy:<15} {'N/A':<12} {'N/A':<8} {'N/A':<12} {'N/A':<10}")

def main():
    """Run comprehensive analysis"""
    print("🧪 Comprehensive Document Processing Analysis")
    print("=" * 60)
    
    # 1. Analyze original files
    analyze_file_sizes()
    
    # 2. Test single document processing
    single_result = test_single_document_processing()
    
    # 3. Test document combination
    combination_results = test_document_combination()
    
    # 4. Compare strategies
    compare_combination_strategies(combination_results)
    
    # 5. Detailed content preservation analysis
    if single_result:
        print(f"\n🔍 Single Document Content Preservation:")
        analyze_content_preservation(['bitcoin_whitepaper.tex'], single_result['final_document'])
    
    if combination_results:
        for strategy, result in combination_results:
            if 'final_document' in result:
                print(f"\n🔍 {strategy.title()} Strategy Content Preservation:")
                analyze_content_preservation(
                    ['bitcoin_whitepaper.tex', 'blockchain_security.tex'], 
                    result['final_document']
                )
    
    print(f"\n🎯 Analysis Complete!")
    print("Check the outputs/ directory for detailed results and individual section files.")

if __name__ == "__main__":
    main()