"""Output management and aggregation module"""
import os
from datetime import datetime

class OutputManager:
    def __init__(self, base_path="outputs"):
        self.base_path = base_path
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = os.path.join(base_path, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        
    def save_section_output(self, section_name, content):
        """Save individual section output"""
        safe_name = section_name.replace(" ", "_").replace(".", "")
        file_path = os.path.join(self.session_path, f"{safe_name}.tex")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path
    
    def aggregate_document(self, processed_sections):
        """Aggregate all sections into final document"""
        latex_doc = []
        
        # Document header
        latex_doc.extend([
            "\\documentclass{article}",
            "\\usepackage[pdftex]{graphicx}",
            "\\usepackage{amsmath}",
            "\\usepackage{pgfplots}",
            "\\usetikzlibrary{shapes.geometric, arrows}",
            "\\pgfplotsset{width=10cm, compat=1.18}",
            "",
            "\\begin{document}",
            ""
        ])
        
        # Add sections in order
        section_order = [
            "Title", "Author", "Abstract", "1. Introduction", 
            "2. Theoretical Foundations", "3. The Proposed Framework: DAWF",
            "4. Input Data and Database", "5. Implementation Details",
            "6. Testing and Verification", "7. Experimental Results and Discussion",
            "8. Conclusion", "References"
        ]
        
        for section_name in section_order:
            if section_name in processed_sections:
                content = processed_sections[section_name]
                if content.strip():
                    latex_doc.append(f"% {section_name}")
                    latex_doc.append(content)
                    latex_doc.append("")
        
        latex_doc.append("\\end{document}")
        
        # Save final document
        final_content = "\n".join(latex_doc)
        final_path = os.path.join(self.session_path, "final_document.tex")
        
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        return final_path
    
    def save_processing_log(self, log_entries):
        """Save processing log"""
        log_path = os.path.join(self.session_path, "processing_log.txt")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(f"{entry}\n")
        
        return log_path