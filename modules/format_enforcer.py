"""Format enforcement for LLM outputs"""

class FormatEnforcer:
    def __init__(self, output_format="latex"):
        self.output_format = output_format
        self.format_rules = {
            "latex": {
                "system_prompt": """You are a LaTeX expert. Output ONLY valid LaTeX content.
STRICT RULES:
- Use \\section{Title} for sections, NOT ### Title
- Use \\subsection{Title} for subsections, NOT #### Title  
- Use \\textbf{text} for bold, NOT **text**
- Use \\textit{text} for italics, NOT *text*
- Use \\begin{itemize} \\item ... \\end{itemize} for lists, NOT - item
- Use \\begin{enumerate} \\item ... \\end{enumerate} for numbered lists
- Preserve all equations exactly as \\begin{equation} ... \\end{equation}
- NO Markdown syntax allowed
- NO document structure (\\documentclass, \\begin{document}, \\end{document})""",
                
                "validation_patterns": [
                    (r"^#{1,6}\s", "INVALID: Use \\section{} not ###"),
                    (r"\*\*([^*]+)\*\*", "INVALID: Use \\textbf{} not **text**"),
                    (r"\*([^*]+)\*", "INVALID: Use \\textit{} not *text*"),
                    (r"^-\s", "INVALID: Use \\begin{itemize} not - lists"),
                    (r"```", "INVALID: Use \\begin{verbatim} not ```")
                ],
                
                "post_process": [
                    (r"^#{3}\s*(.+)$", r"\\section{\1}"),
                    (r"^#{4}\s*(.+)$", r"\\subsection{\1}"),
                    (r"\*\*([^*]+)\*\*", r"\\textbf{\1}"),
                    (r"\*([^*]+)\*", r"\\textit{\1}")
                ]
            },
            
            "markdown": {
                "system_prompt": """You are a Markdown expert. Output ONLY valid Markdown content.
STRICT RULES:
- Use ### for sections, NOT \\section{}
- Use #### for subsections, NOT \\subsection{}
- Use **text** for bold, NOT \\textbf{}
- Use *text* for italics, NOT \\textit{}
- Use - item for lists, NOT \\begin{itemize}
- Use ```language for code blocks
- Convert LaTeX equations to $...$ or $$...$$""",
                
                "validation_patterns": [
                    (r"\\section\{", "INVALID: Use ### not \\section{}"),
                    (r"\\textbf\{", "INVALID: Use **text** not \\textbf{}"),
                    (r"\\begin\{itemize\}", "INVALID: Use - lists not \\begin{itemize}")
                ],
                
                "post_process": [
                    (r"\\section\{([^}]+)\}", r"### \1"),
                    (r"\\subsection\{([^}]+)\}", r"#### \1"),
                    (r"\\textbf\{([^}]+)\}", r"**\1**"),
                    (r"\\textit\{([^}]+)\}", r"*\1*")
                ]
            }
        }
    
    def get_system_prompt(self):
        """Get format-specific system prompt"""
        return self.format_rules[self.output_format]["system_prompt"]
    
    def validate_output(self, content):
        """Validate output against format rules"""
        import re
        issues = []
        
        patterns = self.format_rules[self.output_format]["validation_patterns"]
        for pattern, message in patterns:
            if re.search(pattern, content, re.MULTILINE):
                issues.append(message)
        
        return issues
    
    def post_process_output(self, content):
        """Fix common format issues"""
        import re
        
        post_rules = self.format_rules[self.output_format]["post_process"]
        for pattern, replacement in post_rules:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        return content
    
    def enforce_format(self, content):
        """Complete format enforcement pipeline"""
        # Post-process to fix issues
        fixed_content = self.post_process_output(content)
        
        # Validate and report remaining issues
        issues = self.validate_output(fixed_content)
        
        return fixed_content, issues