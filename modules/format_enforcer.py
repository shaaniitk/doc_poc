"""
Format Enforcement Utility - Output Quality Guardian.

This module acts as a self-contained utility for ensuring LLM outputs conform to
strict format requirements (e.g., valid LaTeX). It loads its instructional system
prompts from the central config file, separating configuration from logic.
"""
import re
from config import FORMAT_ENFORCER_PROMPTS

class FormatEnforcer:
    """
    Enforces strict format compliance for LLM outputs. Prompts are loaded dynamically.
    """
    def __init__(self, output_format="latex"):
        self.output_format = output_format
        
        # --- Architectural Decision: Configuration vs. Logic ---
        # The regex patterns and replacement rules below are treated as 'logic',
        # not 'configuration'. While they could be moved to the config file,
        # keeping them here offers significant advantages:
        #
        # 1.  **Maintainability:** The regex pattern and its corresponding action
        #     (a replacement string or validation message) are a tightly coupled
        #     pair. Keeping them together in code makes the transformation logic
        #     self-contained and easier to understand and debug.
        # 2.  **Separation of Concerns:** Prompts (in config.py) are user-facing
        #     'configuration' that defines *what* the LLM should do. These patterns
        #     are 'logic' that defines *how* our program sanitizes output. This
        #     separation respects the distinct roles of a prompt engineer (editing
        #     config) and a developer (editing code).
        # 3.  **Robustness:** Regex patterns are a form of code. Modifying them
        #     can have significant behavioral impacts. Keeping them within the
        #     Python file ensures they are version-controlled and reviewed as
        #     part of the core application logic.
        # ---------------------------------------------------------------------
        self.format_rules = {
            "latex": {
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
                    (r"\*([^*]+)\*", r"\\textit{\1}"),
                    (r"(?m)^-\s+(.+)$", r"\\item \1")
                ]
            },
            "markdown": {
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

        # Dynamically inject the system prompts from the config file.
        # This makes the class much cleaner and more maintainable.
        for format_name, prompt in FORMAT_ENFORCER_PROMPTS.items():
            if format_name in self.format_rules:
                self.format_rules[format_name]['system_prompt'] = prompt
    
    def get_system_prompt(self):
        """Returns the format-specific system prompt loaded from the config."""
        return self.format_rules[self.output_format].get("system_prompt", "")

    def enforce_format(self, content):
        """
        Runs the full format enforcement process: automatic fixes followed by validation.
        """
        fixed_content = self._post_process_output(content)
        issues = self._validate_output(fixed_content)
        return fixed_content, issues

    def _validate_output(self, content):
        """Validates content against format-specific rules."""
        issues = []
        patterns = self.format_rules[self.output_format].get("validation_patterns", [])
        for pattern, message in patterns:
            if re.search(pattern, content, re.MULTILINE):
                issues.append(message)
        return issues
    
    def _post_process_output(self, content):
        """Automatically fixes common formatting issues."""
        post_rules = self.format_rules[self.output_format].get("post_process", [])
        for pattern, replacement in post_rules:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        return content