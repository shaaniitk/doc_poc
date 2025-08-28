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
                        (r"(?i)^\s*Here is the (standardized text|transition sentence|refactored content).*?:?\s*$", "", re.MULTILINE),
                        # Removes "Key Improvements:" headers and similar lines.
                        (r"(?i)^\s*###?\s*Key Improvements:?\s*$", "", re.MULTILINE),
                        (r"\\textbf\{([\s\S]*?)\}", r"\1", re.DOTALL),
                        (r"```(.*?)```", r"\\begin{verbatim}\1\\end{verbatim}", re.DOTALL),
                        (r"^#{3}\s*(.+)$", r"\\section{\1}", 0),
                        (r"^#{4}\s*(.+)$", r"\\subsection{\1}", 0),
                        (r"\*\*([^*]+)\*\*", r"\\textbf{\1}", 0),
                        (r"\*([^*]+)\*", r"\\textit{\1}", 0),
                        (r"(?m)^-\s+(.+)$", r"\\item \1", 0) # Note: (m) is a flag, so we'll use re.MULTILINE
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
        """Automatically fixes common formatting issues, now with flag support."""
        # The rules are now tuples of (pattern, replacement, flags)
        # where flags can be 0 if not needed.
        post_rules = self.format_rules[self.output_format].get("post_process", [])
        
        # We need to add re.MULTILINE to the last rule to be fully explicit
        # This is just a cleanup step to match the new structure
        final_rules = []
        for rule in post_rules:
            pattern, replacement, flags = rule
            if "(?m)" in pattern:
                final_rules.append((pattern.replace("(?m)", ""), replacement, re.MULTILINE))
            else:
                final_rules.append(rule)

        for pattern, replacement, flags in final_rules:
            try:
                content = re.sub(pattern, replacement, content, flags=flags)
            except re.error as e:
                # Log an error if a regex is invalid, but don't crash
                log.error(f"Invalid regex in FormatEnforcer: {pattern}. Error: {e}")
        return content