"""Output formatting module for different formats"""
import json
from config import OUTPUT_FORMATS

class OutputFormatter:
    def __init__(self, format_type="latex"):
        self.format_type = format_type
        self.config = OUTPUT_FORMATS.get(format_type, OUTPUT_FORMATS["latex"])
    
    def format_document(self, processed_sections):
        if self.format_type == "latex":
            return self._format_latex(processed_sections)
        elif self.format_type == "markdown":
            return self._format_markdown(processed_sections)
        elif self.format_type == "json":
            return self._format_json(processed_sections)
    
    def _format_latex(self, sections):
        lines = [
            "\\documentclass{article}",
            "\\usepackage[pdftex]{graphicx}",
            "\\usepackage{amsmath}",
            "\\begin{document}",
            ""
        ]
        
        for section, content in sections.items():
            if content.strip():
                lines.extend([f"% {section}", content, ""])
        
        lines.append("\\end{document}")
        return "\n".join(lines)
    
    def _format_markdown(self, sections):
        lines = []
        for section, content in sections.items():
            if content.strip():
                lines.extend([f"# {section}", "", content, ""])
        return "\n".join(lines)
    
    def _format_json(self, sections):
        return json.dumps(sections, indent=2)