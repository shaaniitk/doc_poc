"""Format enforcement module - output quality guardian

This module ensures LLM outputs conform to strict format requirements,
preventing compilation errors and maintaining professional document quality.

KEY RESPONSIBILITIES:
- Validate LLM outputs against format-specific rules
- Automatically fix common formatting errors
- Provide clear error messages for manual correction
- Support multiple output formats (LaTeX, Markdown, JSON)

QUALITY ASSURANCE:
- Pattern-based validation for format compliance
- Automatic correction of common mistakes
- Prevention of compilation-breaking errors
- Consistent formatting across all outputs

SUPPORTED FORMATS:
- LaTeX: Academic papers, technical documents
- Markdown: Documentation, web content
- JSON: Structured data output

This is the quality control checkpoint for all generated content.
"""

class FormatEnforcer:
    """Format enforcement engine
    
    Enforces strict format compliance for LLM outputs to ensure
    professional, compilation-ready documents.
    
    ENFORCEMENT STRATEGIES:
    - System prompts for LLM behavior control
    - Validation patterns for error detection
    - Post-processing rules for automatic fixes
    - Multi-format support with format-specific rules
    
    QUALITY FOCUS:
    Prevents common LLM formatting mistakes that break compilation.
    """
    def __init__(self, output_format="latex"):
        # 🏷️ Set target output format
        self.output_format = output_format
        
        # 📜 Format-specific rules and patterns
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
                    (r"^\-\s", "INVALID: Use \\begin{itemize} not - lists"),
                    (r"```", "INVALID: Use \\begin{verbatim} not ```")
                ],
                
                "post_process": [
                    (r"^#{3}\s*(.+)$", r"\\section{\1}"),
                    (r"^#{4}\s*(.+)$", r"\\subsection{\1}"),
                    (r"\*\*([^*]+)\*\*", r"\\textbf{\1}"),
                    (r"\*([^*]+)\*", r"\\textit{\1}"),
                    # Convert Markdown list items to LaTeX item entries
                    (r"(?m)^-\s+(.+)$", r"\\item \1")
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
        """🎯 GET FORMAT-SPECIFIC SYSTEM PROMPT
        
        Returns the system prompt that instructs LLMs on proper
        formatting for the target output format.
        
        Returns:
            str: System prompt with format instructions
            
        🧠 LLM GUIDANCE:
        These prompts are crucial for preventing format errors
        at the source rather than fixing them later.
        """
        return self.format_rules[self.output_format]["system_prompt"]
    
    def validate_output(self, content):
        """🔍 OUTPUT VALIDATION ENGINE
        
        Validates content against format-specific rules to identify
        formatting issues that could break compilation or presentation.
        
        Args:
            content: Generated content to validate
            
        Returns:
            list: List of validation issues found
            
        🔧 VALIDATION PROCESS:
        Uses regex patterns to detect common formatting mistakes
        and provides specific error messages for each issue.
        """
        import re
        issues = []  # 📋 Collection of found issues
        
        # 🔍 Check content against validation patterns
        patterns = self.format_rules[self.output_format]["validation_patterns"]
        for pattern, message in patterns:
            if re.search(pattern, content, re.MULTILINE):
                issues.append(message)  # ⚠️ Add issue to list
        
        return issues
    
    def post_process_output(self, content):
        """🔧 AUTOMATIC FORMAT CORRECTION ENGINE
        
        Automatically fixes common formatting issues using
        pattern-based replacements.
        
        Args:
            content: Content with potential format issues
            
        Returns:
            str: Content with format issues corrected
            
        🧹 CORRECTION PROCESS:
        Applies format-specific correction rules to fix
        common LLM formatting mistakes automatically.
        """
        import re
        
        # 🔧 Apply post-processing correction rules
        post_rules = self.format_rules[self.output_format]["post_process"]
        for pattern, replacement in post_rules:
            # 🧹 Fix formatting issues with regex replacement
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        return content
    
    def enforce_format(self, content):
        """🎨 COMPLETE FORMAT ENFORCEMENT PIPELINE
        
        Runs the full format enforcement process: automatic fixes
        followed by validation to catch remaining issues.
        
        Args:
            content: Raw LLM output to enforce
            
        Returns:
            tuple: (corrected_content, remaining_issues)
            
        🔄 ENFORCEMENT PROCESS:
        1. Apply automatic corrections
        2. Validate corrected content
        3. Report any remaining issues
        
        This ensures maximum format compliance with minimal manual intervention.
        """
        # 🔧 Step 1: Apply automatic corrections
        fixed_content = self.post_process_output(content)
        
        # 🔍 Step 2: Validate corrected content for remaining issues
        issues = self.validate_output(fixed_content)
        
        # 🏆 Return corrected content and any remaining issues
        return fixed_content, issues