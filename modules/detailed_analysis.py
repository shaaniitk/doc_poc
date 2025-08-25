"""
Detailed Analysis of Document Combination Results
"""

import os
import re
import logging

# Setup logging
log = logging.getLogger(__name__)

def analyze_combination_results():
    """Analyze the combination results in detail"""
    
    log.info("DETAILED DOCUMENT COMBINATION ANALYSIS")
    log.info("=" * 60)
    
    # Original files analysis
    original_files = {
        'bitcoin_whitepaper.tex': 0,
        'blockchain_security.tex': 0
    }
    
    total_original_size = 0
    total_original_chars = 0
    
    log.info("ORIGINAL FILES ANALYSIS:")
    for file in original_files.keys():
        if os.path.exists(file):
            size = os.path.getsize(file)
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                chars = len(content)
                lines = len(content.split('\n'))
                words = len(content.split())
                
                # Count LaTeX elements
                equations = len(re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', content, re.DOTALL))
                code_blocks = len(re.findall(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', content, re.DOTALL))
                enumerates = len(re.findall(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', content, re.DOTALL))
                
            original_files[file] = {
                'size': size,
                'chars': chars,
                'lines': lines,
                'words': words,
                'equations': equations,
                'code_blocks': code_blocks,
                'enumerates': enumerates
            }
            
            total_original_size += size
            total_original_chars += chars
            
            log.info(f"  {file}:")
            log.info(f"    Size: {size:,} bytes")
            log.info(f"    Characters: {chars:,}")
            log.info(f"    Lines: {lines:,}")
            log.info(f"    Words: {words:,}")
            log.info(f"    Equations: {equations}")
            log.info(f"    Code blocks: {code_blocks}")
            log.info(f"    Lists: {enumerates}")
            log.info("")
    
    log.info(f"COMBINED ORIGINAL TOTALS:")
    log.info(f"  Total Size: {total_original_size:,} bytes")
    log.info(f"  Total Characters: {total_original_chars:,}")
    
    # Analyze combined output - check multiple possible locations (text files only)
    possible_combined_files = [
        'outputs/20250727_140400/final_document.tex',
        'export/augmented_output.tex',
        'augmented_output.tex',
        'final_document.tex',
        'combined_document.tex'
    ]
    
    combined_file = None
    for file_path in possible_combined_files:
        if os.path.exists(file_path):
            combined_file = file_path
            break
    
    if combined_file and os.path.exists(combined_file):
        log.info(f"COMBINED OUTPUT ANALYSIS ({combined_file}):")
        
        size = os.path.getsize(combined_file)
        with open(combined_file, 'r', encoding='utf-8') as f:
            content = f.read()
            chars = len(content)
            lines = len(content.split('\n'))
            words = len(content.split())
            
            # Count LaTeX elements
            equations = len(re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', content, re.DOTALL))
            code_blocks = len(re.findall(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', content, re.DOTALL))
            enumerates = len(re.findall(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', content, re.DOTALL))
            
            # Count sections
            sections = len(re.findall(r'\\section\{.*?\}', content))
            
        log.info(f"  Size: {size:,} bytes")
        log.info(f"  Characters: {chars:,}")
        log.info(f"  Lines: {lines:,}")
        log.info(f"  Words: {words:,}")
        log.info(f"  Equations: {equations}")
        log.info(f"  Code blocks: {code_blocks}")
        log.info(f"  Lists: {enumerates}")
        log.info(f"  Sections: {sections}")
        
        # Calculate preservation metrics
        log.info(f"PRESERVATION ANALYSIS:")
        
        size_change = ((size - total_original_size) / total_original_size) * 100
        char_change = ((chars - total_original_chars) / total_original_chars) * 100
        
        log.info(f"  Size change: {size_change:+.1f}%")
        log.info(f"  Character change: {char_change:+.1f}%")
        
        # Calculate total original LaTeX elements
        total_orig_equations = sum(f['equations'] for f in original_files.values())
        total_orig_code = sum(f['code_blocks'] for f in original_files.values())
        total_orig_lists = sum(f['enumerates'] for f in original_files.values())
        
        log.info(f"  Equations: {total_orig_equations} -> {equations} ({((equations/total_orig_equations)*100 if total_orig_equations > 0 else 0):.1f}% preserved)")
        log.info(f"  Code blocks: {total_orig_code} -> {code_blocks} ({((code_blocks/total_orig_code)*100 if total_orig_code > 0 else 0):.1f}% preserved)")
        log.info(f"  Lists: {total_orig_lists} -> {enumerates} ({((enumerates/total_orig_lists)*100 if total_orig_lists > 0 else 0):.1f}% preserved)")
        
        # Content quality analysis
        log.info(f"CONTENT QUALITY ANALYSIS:")
        
        # Check for key Bitcoin content
        bitcoin_content = [
            ('Double-spending', r'double.?spend'),
            ('Proof-of-work', r'proof.?of.?work'),
            ('Merkle tree', r'merkle'),
            ('Hash functions', r'hash'),
            ('Digital signatures', r'digital.?signature'),
            ('Consensus', r'consensus'),
            ('Blockchain', r'blockchain'),
            ('Cryptographic', r'cryptographic')
        ]
        
        for content_name, pattern in bitcoin_content:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            log.info(f"  {content_name}: {matches} mentions")
        
        # Check for structural issues
        log.info(f"STRUCTURAL ISSUES:")
        
        # Check for duplicate \end{document}
        end_docs = len(re.findall(r'\\end\{document\}', content))
        if end_docs > 1:
            log.error(f"  [ERROR] Multiple \\end{{document}} found: {end_docs}")
        else:
            log.info(f"  [OK] Single \\end{{document}} found")
        
        # Check for orphaned LaTeX commands
        orphaned_commands = re.findall(r'^(equation|enumerate|verbatim)$', content, re.MULTILINE)
        if orphaned_commands:
            log.info(f"  [WARNING] Orphaned LaTeX commands: {len(orphaned_commands)}")
        else:
            log.info(f"  [OK] No orphaned LaTeX commands")
        
        # Check for proper section structure
        if sections > 0:
            log.info(f"  [OK] Document has proper section structure: {sections} sections")
        else:
            log.error(f"  [ERROR] No proper sections found")
    
    else:
        log.error(f"[ERROR] No combined files found in any of the expected locations")
    
    # Overall quality assessment
    log.info(f"OVERALL QUALITY ASSESSMENT:")
    
    if os.path.exists(combined_file):
        quality_score = 100
        
        # Deduct points for size changes
        if abs(size_change) > 20:
            quality_score -= 20
            log.info(f"  -20 points: Significant size change ({size_change:+.1f}%)")
        elif abs(size_change) > 10:
            quality_score -= 10
            log.info(f"  -10 points: Moderate size change ({size_change:+.1f}%)")
        
        # Deduct points for missing equations
        if total_orig_equations > 0 and equations < total_orig_equations:
            quality_score -= 15
            log.info(f"  -15 points: Missing equations ({total_orig_equations - equations} lost)")
        
        # Deduct points for missing code blocks
        if total_orig_code > 0 and code_blocks < total_orig_code:
            quality_score -= 15
            log.info(f"  -15 points: Missing code blocks ({total_orig_code - code_blocks} lost)")
        
        # Deduct points for structural issues
        if end_docs > 1:
            quality_score -= 10
            log.info(f"  -10 points: Structural issues (multiple \\end{{document}})")
        
        log.info(f"FINAL QUALITY SCORE: {quality_score}/100")
        
        if quality_score >= 90:
            log.info("  [EXCELLENT] High quality combination")
        elif quality_score >= 75:
            log.info("  [GOOD] Acceptable quality with minor issues")
        elif quality_score >= 60:
            log.info("  [FAIR] Usable but needs improvement")
        else:
            log.info("  [POOR] Significant quality issues")

def main():
    """Main function for detailed analysis"""
    analyze_combination_results()

if __name__ == "__main__":
    main()