"""
Compare single document vs multi-document processing results
"""

import os
import re

def test_single_document():
    """Test single document processing using modular framework"""
    print("SINGLE DOCUMENT PROCESSING TEST")
    print("=" * 50)
    
    try:
        # Import modular components
        from .file_loader import load_latex_file
        from .chunker import extract_latex_sections, group_chunks_by_section
        from .section_mapper import assign_chunks_to_skeleton
        from .output_manager import OutputManager
        
        source_file = 'bitcoin_whitepaper.tex'
        if not os.path.exists(source_file):
            print(f"[ERROR] Source file not found: {source_file}")
            return None
            
        # Process document
        content = load_latex_file(source_file)
        chunks = extract_latex_sections(content)
        grouped_chunks = group_chunks_by_section(chunks)
        assignments = assign_chunks_to_skeleton(grouped_chunks)
        
        # Create simple processed content
        processed_content = ""
        for section_name, section_chunks in assignments.items():
            if section_chunks:
                processed_content += f"\\section{{{section_name}}}\n"
                for chunk in section_chunks:
                    processed_content += chunk['content'] + "\n\n"
        
        print("[OK] Single document processing completed")
        return analyze_document_quality('bitcoin_whitepaper.tex', processed_content, "Single Document")
        
    except Exception as e:
        print(f"[ERROR] Single document processing failed: {e}")
        return None

def test_combination():
    """Test document combination using modular framework"""
    print("\nDOCUMENT COMBINATION TEST")
    print("=" * 50)
    
    try:
        # Import modular components
        from .document_combiner import DocumentCombiner
        
        source1 = 'bitcoin_whitepaper.tex'
        source2 = 'blockchain_security.tex'
        
        if not os.path.exists(source1) or not os.path.exists(source2):
            print(f"[ERROR] Source files not found")
            return None
        
        combiner = DocumentCombiner('latex')
        combined_content, format_issues = combiner.combine_documents(
            source1, source2, 'smart_merge', 'latex'
        )
        
        print("[OK] Document combination completed")
        return analyze_document_quality([source1, source2], combined_content, "Combined Documents")
        
    except Exception as e:
        print(f"[ERROR] Document combination failed: {e}")
        return None

def analyze_document_quality(original_files, processed_content, test_name):
    """Analyze document quality in detail"""
    
    print(f"\n📊 {test_name.upper()} ANALYSIS:")
    
    # Handle single file vs multiple files
    if isinstance(original_files, str):
        original_files = [original_files]
    
    # Read original content
    original_content = ""
    original_stats = {'size': 0, 'equations': 0, 'code_blocks': 0, 'lists': 0}
    
    for file in original_files:
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                original_content += file_content + "\n"
                
                # Count elements in this file
                original_stats['size'] += len(file_content)
                original_stats['equations'] += len(re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', file_content, re.DOTALL))
                original_stats['code_blocks'] += len(re.findall(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', file_content, re.DOTALL))
                original_stats['lists'] += len(re.findall(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', file_content, re.DOTALL))
    
    # Analyze processed content
    processed_stats = {
        'size': len(processed_content),
        'equations': len(re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', processed_content, re.DOTALL)),
        'code_blocks': len(re.findall(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', processed_content, re.DOTALL)),
        'lists': len(re.findall(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', processed_content, re.DOTALL))
    }
    
    # Calculate preservation rates
    preservation = {}
    for key in ['equations', 'code_blocks', 'lists']:
        if original_stats[key] > 0:
            preservation[key] = (processed_stats[key] / original_stats[key]) * 100
        else:
            preservation[key] = 100 if processed_stats[key] == 0 else 0
    
    size_change = ((processed_stats['size'] - original_stats['size']) / original_stats['size']) * 100
    
    print(f"  Original size: {original_stats['size']:,} characters")
    print(f"  Processed size: {processed_stats['size']:,} characters")
    print(f"  Size change: {size_change:+.1f}%")
    print(f"  Equations: {original_stats['equations']} -> {processed_stats['equations']} ({preservation['equations']:.1f}% preserved)")
    print(f"  Code blocks: {original_stats['code_blocks']} -> {processed_stats['code_blocks']} ({preservation['code_blocks']:.1f}% preserved)")
    print(f"  Lists: {original_stats['lists']} -> {processed_stats['lists']} ({preservation['lists']:.1f}% preserved)")
    
    # Check for structural issues
    structural_issues = []
    
    # Multiple \end{document}
    end_docs = len(re.findall(r'\\end\{document\}', processed_content))
    if end_docs > 1:
        structural_issues.append(f"Multiple \\end{{document}} ({end_docs})")
    
    # Orphaned commands
    orphaned = len(re.findall(r'^(equation|enumerate|verbatim)$', processed_content, re.MULTILINE))
    if orphaned > 0:
        structural_issues.append(f"Orphaned commands ({orphaned})")
    
    # Missing sections
    sections = len(re.findall(r'\\section\{.*?\}', processed_content))
    if sections == 0:
        structural_issues.append("No proper sections")
    
    print(f"  Structural issues: {len(structural_issues)}")
    for issue in structural_issues:
        print(f"    - {issue}")
    
    # Calculate quality score
    quality_score = 100
    
    # Deduct for content loss
    if preservation['equations'] < 90:
        quality_score -= 20
    if preservation['code_blocks'] < 90:
        quality_score -= 15
    if preservation['lists'] < 90:
        quality_score -= 10
    
    # Deduct for size changes
    if abs(size_change) > 20:
        quality_score -= 15
    elif abs(size_change) > 10:
        quality_score -= 10
    
    # Deduct for structural issues
    quality_score -= len(structural_issues) * 5
    
    print(f"  Quality Score: {quality_score}/100")
    
    return {
        'test_name': test_name,
        'original_stats': original_stats,
        'processed_stats': processed_stats,
        'preservation': preservation,
        'size_change': size_change,
        'structural_issues': len(structural_issues),
        'quality_score': quality_score
    }

def identify_combination_issues():
    """Identify specific issues with document combination using modular framework"""
    
    print("\nCOMBINATION ISSUE ANALYSIS:")
    print("=" * 50)
    
    try:
        # Import modular components
        from .file_loader import load_latex_file
        from .chunker import extract_latex_sections, group_chunks_by_section
        
        # Load both documents
        bitcoin_content = load_latex_file('bitcoin_whitepaper.tex')
        security_content = load_latex_file('blockchain_security.tex')
        
        # Extract chunks from both
        bitcoin_chunks = extract_latex_sections(bitcoin_content)
        security_chunks = extract_latex_sections(security_content)
        
        print(f"Chunk Analysis:")
        print(f"  Bitcoin chunks: {len(bitcoin_chunks)}")
        print(f"  Security chunks: {len(security_chunks)}")
        
        # Group by sections
        bitcoin_sections = group_chunks_by_section(bitcoin_chunks)
        security_sections = group_chunks_by_section(security_chunks)
        
        print(f"  Bitcoin sections: {len(bitcoin_sections)}")
        print(f"  Security sections: {len(security_sections)}")
        
        # Section overlap analysis
        bitcoin_section_names = set(bitcoin_sections.keys())
        security_section_names = set(security_sections.keys())
        
        common_sections = bitcoin_section_names & security_section_names
        bitcoin_only = bitcoin_section_names - security_section_names
        security_only = security_section_names - bitcoin_section_names
        
        print(f"\nSection Overlap Analysis:")
        print(f"  Common sections: {len(common_sections)}")
        print(f"  Bitcoin-only sections: {len(bitcoin_only)}")
        print(f"  Security-only sections: {len(security_only)}")
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")

def main():
    """Run comparison analysis"""
    
    print("SINGLE vs MULTI-DOCUMENT PROCESSING COMPARISON")
    print("=" * 60)
    
    # Analyze existing files
    files_to_check = [
        'bitcoin_whitepaper.tex',
        'blockchain_security.tex',
        'export/final_document.pdf',
        'export/combined_document.pdf'
    ]
    
    print(f"\nFILE ANALYSIS:")
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  [OK] {file}: {size:,} bytes")
        else:
            print(f"  [ERROR] {file}: Not found")
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()