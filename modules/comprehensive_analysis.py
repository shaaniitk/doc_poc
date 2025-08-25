"""
Comprehensive Analysis of Document Processing and Combination
"""

import os
import re
import logging

log = logging.getLogger(__name__)

def analyze_file_sizes():
    """Analyze original file sizes"""
    log.info("Original File Analysis")
    log.info("=" * 50)
    
    files = ['bitcoin_whitepaper.tex', 'test_document2.tex', 'blockchain_security.tex']
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = len(content.split('\n'))
                chars = len(content)
                words = len(content.split())
            
            log.info(f"{file}:")
            log.info(f"  Size: {size:,} bytes")
            log.info(f"  Lines: {lines:,}")
            log.info(f"  Characters: {chars:,}")
            log.info(f"  Words: {words:,}")
            log.info("")

def test_single_document_processing():
    """Test single document processing using modular framework"""
    log.info("Single Document Processing Test")
    log.info("=" * 50)
    
    try:
        # Import modular components
        from .file_loader import load_latex_file
        from .chunker import extract_latex_sections, group_chunks_by_section
        from .section_mapper import assign_chunks_to_skeleton
        from .llm_handler import ContextualLLMHandler
        from .output_manager import OutputManager
        
        source_file = 'bitcoin_whitepaper.tex'
        if not os.path.exists(source_file):
            log.error(f"[ERROR] Source file not found: {source_file}")
            return None
            
        # Process using modular framework
        content = load_latex_file(source_file)
        chunks = extract_latex_sections(content)
        grouped_chunks = group_chunks_by_section(chunks)
        assignments = assign_chunks_to_skeleton(grouped_chunks)
        
        # Initialize handlers
        llm_handler = ContextualLLMHandler()
        output_manager = OutputManager()
        
        # Process sections
        processed_sections = {}
        for section_name, section_chunks in assignments.items():
            if section_chunks:
                combined_content = '\n\n'.join([chunk['content'] for chunk in section_chunks])
                result = llm_handler.process_section(section_name, combined_content, "Process this section.")
                processed_sections[section_name] = result
        
        # Save final document
        final_path = output_manager.aggregate_document(processed_sections)
        
        log.info("[OK] Single document processing completed")
        return {
            'final_document': final_path,
            'processed_sections': len(processed_sections),
            'session_path': output_manager.session_path
        }
        
    except Exception as e:
        log.error(f"[ERROR] Single document processing failed: {e}")
        return None

def test_document_combination():
    """Test document combination using modular framework"""
    log.info("Document Combination Tests")
    log.info("=" * 50)
    
    try:
        # Import modular components
        from .document_combiner import DocumentCombiner
        from .output_manager import OutputManager
        
        source1 = 'bitcoin_whitepaper.tex'
        source2 = 'blockchain_security.tex'
        
        if not os.path.exists(source1) or not os.path.exists(source2):
            log.error(f"[ERROR] Source files not found")
            return []
        
        results = []
        strategies = ['smart_merge', 'append']
        
        for strategy in strategies:
            log.info(f"Testing: {strategy} strategy")
            
            combiner = DocumentCombiner('latex')
            output_manager = OutputManager()
            
            combined_content, format_issues = combiner.combine_documents(
                source1, source2, strategy, 'latex'
            )
            
            final_path = output_manager.save_section_output(f"combined_{strategy}", combined_content)
            
            # Analyze output
            if os.path.exists(final_path):
                size = os.path.getsize(final_path)
                with open(final_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                    chars = len(content)
                
                result = {
                    'final_document': final_path,
                    'processed_sections': len(re.findall(r'\\section\{.*?\}', content)),
                    'output_stats': {
                        'size': size,
                        'lines': lines,
                        'characters': chars
                    }
                }
                results.append((strategy, result))
                log.info(f"[OK] {strategy} completed successfully")
            
        return results
        
    except Exception as e:
        log.error(f"[ERROR] Document combination failed: {e}")
        return []

def analyze_content_preservation(original_files, processed_file):
    """Analyze content preservation in detail"""
    log.info(f"Content Preservation Analysis")
    log.info("=" * 50)
    
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
        log.error(f"❌ Processed file not found: {processed_file}")
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
    
    log.info(f"Character Analysis:")
    log.info(f"  Original: {original_chars:,} characters")
    log.info(f"  Processed: {processed_chars:,} characters")
    log.info(f"  Change: {change_percent:+.1f}%")
    
    # Count LaTeX environments
    patterns = {
        'equations': r'\\begin\{equation\}.*?\\end\{equation\}',
        'code_blocks': r'\\begin\{verbatim\}.*?\\end\{verbatim\}',
        'lists': r'\\begin\{enumerate\}.*?\\end\{enumerate\}',
        'figures': r'\\begin\{figure\}.*?\\end\{figure\}'
    }
    
    log.info(f"LaTeX Environment Preservation:")
    for env_name, pattern in patterns.items():
        original_count = len(re.findall(pattern, original_content, re.DOTALL))
        processed_count = len(re.findall(pattern, processed_content, re.DOTALL))
        
        if original_count > 0:
            preservation_rate = (processed_count / original_count) * 100
            log.info(f"  {env_name.title()}: {original_count} -> {processed_count} ({preservation_rate:.1f}% preserved)")
        else:
            log.info(f"  {env_name.title()}: {original_count} -> {processed_count} (N/A)")
    
    # Check for specific content
    log.info(f"Specific Content Check:")
    
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
        
        status = "[OK]" if (original_found and processed_found) or (not original_found and not processed_found) else "[ERROR]"
        log.info(f"  {status} {content_name}: {'Found' if processed_found else 'Missing'}")

def compare_combination_strategies(results):
    """Compare different combination strategies"""
    log.info(f"Combination Strategy Comparison")
    log.info("=" * 50)
    
    if not results:
        log.error("[ERROR] No results to compare")
        return
    
    log.info(f"{'Strategy':<15} {'Size (KB)':<12} {'Lines':<8} {'Characters':<12} {'Sections':<10}")
    log.info("-" * 65)
    
    for strategy, result in results:
        if 'output_stats' in result:
            stats = result['output_stats']
            size_kb = stats['size'] / 1024
            log.info(f"{strategy:<15} {size_kb:<12.1f} {stats['lines']:<8} {stats['characters']:<12,} {result['processed_sections']:<10}")
        else:
            log.info(f"{strategy:<15} {'N/A':<12} {'N/A':<8} {'N/A':<12} {'N/A':<10}")

def main():
    """Run comprehensive analysis"""
    log.info("Comprehensive Document Processing Analysis")
    log.info("=" * 60)
    
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
        log.info(f"Single Document Content Preservation:")
        analyze_content_preservation(['bitcoin_whitepaper.tex'], single_result['final_document'])
    
    if combination_results:
        for strategy, result in combination_results:
            if 'final_document' in result:
                log.info(f"{strategy.title()} Strategy Content Preservation:")
                analyze_content_preservation(
                    ['bitcoin_whitepaper.tex', 'blockchain_security.tex'], 
                    result['final_document']
                )
    
    log.info(f"Analysis Complete!")
    log.info("Check the outputs/ directory for detailed results and individual section files.")

if __name__ == "__main__":
    main()