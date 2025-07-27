"""
Detailed Analysis of Document Combination Results
"""

import os
import re

def analyze_combination_results():
    """Analyze the combination results in detail"""
    
    print("📊 DETAILED DOCUMENT COMBINATION ANALYSIS")
    print("=" * 60)
    
    # Original files analysis
    original_files = {
        'bitcoin_whitepaper.tex': 0,
        'blockchain_security.tex': 0
    }
    
    total_original_size = 0
    total_original_chars = 0
    
    print("📄 ORIGINAL FILES ANALYSIS:")
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
            
            print(f"  {file}:")
            print(f"    Size: {size:,} bytes")
            print(f"    Characters: {chars:,}")
            print(f"    Lines: {lines:,}")
            print(f"    Words: {words:,}")
            print(f"    Equations: {equations}")
            print(f"    Code blocks: {code_blocks}")
            print(f"    Lists: {enumerates}")
            print()
    
    print(f"📊 COMBINED ORIGINAL TOTALS:")
    print(f"  Total Size: {total_original_size:,} bytes")
    print(f"  Total Characters: {total_original_chars:,}")
    
    # Analyze combined output
    combined_file = 'outputs/20250727_140400/final_document.tex'
    
    if os.path.exists(combined_file):
        print(f"\n📄 COMBINED OUTPUT ANALYSIS:")
        
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
            
        print(f"  Size: {size:,} bytes")
        print(f"  Characters: {chars:,}")
        print(f"  Lines: {lines:,}")
        print(f"  Words: {words:,}")
        print(f"  Equations: {equations}")
        print(f"  Code blocks: {code_blocks}")
        print(f"  Lists: {enumerates}")
        print(f"  Sections: {sections}")
        
        # Calculate preservation metrics
        print(f"\n📈 PRESERVATION ANALYSIS:")
        
        size_change = ((size - total_original_size) / total_original_size) * 100
        char_change = ((chars - total_original_chars) / total_original_chars) * 100
        
        print(f"  Size change: {size_change:+.1f}%")
        print(f"  Character change: {char_change:+.1f}%")
        
        # Calculate total original LaTeX elements
        total_orig_equations = sum(f['equations'] for f in original_files.values())
        total_orig_code = sum(f['code_blocks'] for f in original_files.values())
        total_orig_lists = sum(f['enumerates'] for f in original_files.values())
        
        print(f"  Equations: {total_orig_equations} → {equations} ({((equations/total_orig_equations)*100 if total_orig_equations > 0 else 0):.1f}% preserved)")
        print(f"  Code blocks: {total_orig_code} → {code_blocks} ({((code_blocks/total_orig_code)*100 if total_orig_code > 0 else 0):.1f}% preserved)")
        print(f"  Lists: {total_orig_lists} → {enumerates} ({((enumerates/total_orig_lists)*100 if total_orig_lists > 0 else 0):.1f}% preserved)")
        
        # Content quality analysis
        print(f"\n🔍 CONTENT QUALITY ANALYSIS:")
        
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
            print(f"  {content_name}: {matches} mentions")
        
        # Check for structural issues
        print(f"\n⚠️  STRUCTURAL ISSUES:")
        
        # Check for duplicate \end{document}
        end_docs = len(re.findall(r'\\end\{document\}', content))
        if end_docs > 1:
            print(f"  ❌ Multiple \\end{{document}} found: {end_docs}")
        else:
            print(f"  ✅ Single \\end{{document}} found")
        
        # Check for orphaned LaTeX commands
        orphaned_commands = re.findall(r'^(equation|enumerate|verbatim)$', content, re.MULTILINE)
        if orphaned_commands:
            print(f"  ⚠️  Orphaned LaTeX commands: {len(orphaned_commands)}")
        else:
            print(f"  ✅ No orphaned LaTeX commands")
        
        # Check for proper section structure
        if sections > 0:
            print(f"  ✅ Document has proper section structure: {sections} sections")
        else:
            print(f"  ❌ No proper sections found")
    
    else:
        print(f"\n❌ Combined file not found: {combined_file}")
    
    # Overall quality assessment
    print(f"\n🏆 OVERALL QUALITY ASSESSMENT:")
    
    if os.path.exists(combined_file):
        quality_score = 100
        
        # Deduct points for size changes
        if abs(size_change) > 20:
            quality_score -= 20
            print(f"  -20 points: Significant size change ({size_change:+.1f}%)")
        elif abs(size_change) > 10:
            quality_score -= 10
            print(f"  -10 points: Moderate size change ({size_change:+.1f}%)")
        
        # Deduct points for missing equations
        if total_orig_equations > 0 and equations < total_orig_equations:
            quality_score -= 15
            print(f"  -15 points: Missing equations ({total_orig_equations - equations} lost)")
        
        # Deduct points for missing code blocks
        if total_orig_code > 0 and code_blocks < total_orig_code:
            quality_score -= 15
            print(f"  -15 points: Missing code blocks ({total_orig_code - code_blocks} lost)")
        
        # Deduct points for structural issues
        if end_docs > 1:
            quality_score -= 10
            print(f"  -10 points: Structural issues (multiple \\end{{document}})")
        
        print(f"\n🎯 FINAL QUALITY SCORE: {quality_score}/100")
        
        if quality_score >= 90:
            print("  ✅ EXCELLENT - High quality combination")
        elif quality_score >= 75:
            print("  ⚠️  GOOD - Acceptable quality with minor issues")
        elif quality_score >= 60:
            print("  ⚠️  FAIR - Usable but needs improvement")
        else:
            print("  ❌ POOR - Significant quality issues")

if __name__ == "__main__":
    analyze_combination_results()