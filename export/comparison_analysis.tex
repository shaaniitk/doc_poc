"""
Compare single document vs multi-document processing results
"""

import os
import re
from document_processor_v2 import DocumentProcessor

def test_single_document():
    """Test single document processing"""
    print("🔄 SINGLE DOCUMENT PROCESSING TEST")
    print("=" * 50)
    
    processor = DocumentProcessor('config_v2.yaml')
    processor.config_manager.config['processing']['enable_enhancement'] = False
    
    try:
        result = processor.process_document(
            source='bitcoin_whitepaper.tex',
            template='bitcoin'
        )
        
        print("✅ Single document processing completed")
        
        # Analyze output
        if os.path.exists(result['final_document']):
            with open(result['final_document'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            return analyze_document_quality('bitcoin_whitepaper.tex', content, "Single Document")
        
    except Exception as e:
        print(f"❌ Single document processing failed: {e}")
        return None

def test_combination():
    """Test document combination"""
    print("\n🔄 DOCUMENT COMBINATION TEST")
    print("=" * 50)
    
    processor = DocumentProcessor('config_v2.yaml')
    processor.config_manager.config['processing']['enable_enhancement'] = False
    
    try:
        result = processor.combine_documents(
            source1='bitcoin_whitepaper.tex',
            source2='blockchain_security.tex',
            strategy='smart_merge',
            template='bitcoin'
        )
        
        print("✅ Document combination completed")
        
        # Analyze output
        if os.path.exists(result['final_document']):
            with open(result['final_document'], 'r', encoding='utf-8') as f:
                content = f.read()
            
            return analyze_document_quality(['bitcoin_whitepaper.tex', 'blockchain_security.tex'], content, "Combined Documents")
        
    except Exception as e:
        print(f"❌ Document combination failed: {e}")
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
    print(f"  Equations: {original_stats['equations']} → {processed_stats['equations']} ({preservation['equations']:.1f}% preserved)")
    print(f"  Code blocks: {original_stats['code_blocks']} → {processed_stats['code_blocks']} ({preservation['code_blocks']:.1f}% preserved)")
    print(f"  Lists: {original_stats['lists']} → {processed_stats['lists']} ({preservation['lists']:.1f}% preserved)")
    
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
    """Identify specific issues with document combination"""
    
    print("\n🔍 COMBINATION ISSUE ANALYSIS:")
    print("=" * 50)
    
    # Check the combination logic
    processor = DocumentProcessor('config_v2.yaml')
    
    # Load both documents
    with open('bitcoin_whitepaper.tex', 'r') as f:
        bitcoin_content = f.read()
    with open('blockchain_security.tex', 'r') as f:
        security_content = f.read()
    
    # Extract chunks from both
    bitcoin_chunks = processor.chunker.extract_chunks(bitcoin_content)
    security_chunks = processor.chunker.extract_chunks(security_content)
    
    print(f"📊 Chunk Analysis:")
    print(f"  Bitcoin chunks: {len(bitcoin_chunks)}")
    print(f"  Security chunks: {len(security_chunks)}")
    
    # Group by sections
    bitcoin_sections = processor._group_chunks_by_section(bitcoin_chunks)
    security_sections = processor._group_chunks_by_section(security_chunks)
    
    print(f"  Bitcoin sections: {len(bitcoin_sections)}")
    print(f"  Security sections: {len(security_sections)}")
    
    print(f"\n📋 Section Overlap Analysis:")
    bitcoin_section_names = set(bitcoin_sections.keys())
    security_section_names = set(security_sections.keys())
    
    common_sections = bitcoin_section_names & security_section_names
    bitcoin_only = bitcoin_section_names - security_section_names
    security_only = security_section_names - bitcoin_section_names
    
    print(f"  Common sections: {len(common_sections)}")
    for section in sorted(common_sections):
        print(f"    - {section}")
    
    print(f"  Bitcoin-only sections: {len(bitcoin_only)}")
    for section in sorted(bitcoin_only):
        print(f"    - {section}")
    
    print(f"  Security-only sections: {len(security_only)}")
    for section in sorted(security_only):
        print(f"    - {section}")
    
    # Test combination strategy
    combined_sections = processor.combiner.combine_documents(
        bitcoin_sections, security_sections, 'smart_merge'
    )
    
    print(f"\n🔄 Combination Results:")
    print(f"  Combined sections: {len(combined_sections)}")
    
    # Check for content loss during combination
    total_original_chars = len(bitcoin_content) + len(security_content)
    total_combined_chars = sum(len(content) for content in combined_sections.values())
    
    combination_loss = ((total_original_chars - total_combined_chars) / total_original_chars) * 100
    print(f"  Content loss during combination: {combination_loss:.1f}%")
    
    # Check specific sections for issues
    print(f"\n🔍 Section-by-Section Analysis:")
    for section_name, content in list(combined_sections.items())[:5]:  # First 5 sections
        char_count = len(content)
        equations = len(re.findall(r'\\begin\{equation\}', content))
        print(f"  {section_name}: {char_count} chars, {equations} equations")

def main():
    """Run comparison analysis"""
    
    print("🧪 SINGLE vs MULTI-DOCUMENT PROCESSING COMPARISON")
    print("=" * 60)
    
    # Test single document
    single_result = test_single_document()
    
    # Test combination
    combination_result = test_combination()
    
    # Compare results
    if single_result and combination_result:
        print(f"\n📊 COMPARISON SUMMARY:")
        print("=" * 50)
        
        print(f"{'Metric':<20} {'Single':<15} {'Combined':<15} {'Difference':<15}")
        print("-" * 65)
        
        # Quality scores
        single_quality = single_result['quality_score']
        combined_quality = combination_result['quality_score']
        quality_diff = single_quality - combined_quality
        
        print(f"{'Quality Score':<20} {single_quality:<15} {combined_quality:<15} {quality_diff:+.0f}")
        
        # Preservation rates
        for metric in ['equations', 'code_blocks', 'lists']:
            single_pres = single_result['preservation'][metric]
            combined_pres = combination_result['preservation'][metric]
            pres_diff = single_pres - combined_pres
            
            print(f"{f'{metric.title()} %':<20} {single_pres:<15.1f} {combined_pres:<15.1f} {pres_diff:+.1f}")
        
        # Size changes
        single_size = single_result['size_change']
        combined_size = combination_result['size_change']
        size_diff = abs(single_size) - abs(combined_size)
        
        print(f"{'Size Change %':<20} {single_size:<15.1f} {combined_size:<15.1f} {size_diff:+.1f}")
        
        print(f"\n🎯 CONCLUSION:")
        if quality_diff > 10:
            print("  ❌ SIGNIFICANT DEGRADATION in multi-document processing")
        elif quality_diff > 5:
            print("  ⚠️  MODERATE DEGRADATION in multi-document processing")
        else:
            print("  ✅ COMPARABLE QUALITY between single and multi-document processing")
    
    # Identify specific combination issues
    identify_combination_issues()

if __name__ == "__main__":
    main()