"""Document combination module with intelligent merging"""
from .file_loader import load_latex_file
from .chunker import extract_latex_sections, group_chunks_by_section
from .format_enforcer import FormatEnforcer

class DocumentCombiner:
    def __init__(self, output_format="latex"):
        self.format_enforcer = FormatEnforcer(output_format)
        self.combination_strategies = {
            "merge": self._merge_documents,
            "interleave": self._interleave_documents,
            "append": self._append_documents,
            "smart_merge": self._smart_merge_documents
        }
    
    def combine_documents(self, doc1_path, doc2_path, strategy="smart_merge", output_format="latex"):
        """Combine two documents using specified strategy"""
        
        # Load documents
        doc1_content = load_latex_file(doc1_path)
        doc2_content = load_latex_file(doc2_path)
        
        # Extract and chunk both documents
        doc1_chunks = extract_latex_sections(doc1_content)
        doc2_chunks = extract_latex_sections(doc2_content)
        
        doc1_grouped = group_chunks_by_section(doc1_chunks)
        doc2_grouped = group_chunks_by_section(doc2_chunks)
        
        # Apply combination strategy
        combined_content = self.combination_strategies[strategy](doc1_grouped, doc2_grouped)
        
        # Enforce format consistency
        formatted_content, issues = self.format_enforcer.enforce_format(combined_content)
        
        return formatted_content, issues
    
    def _merge_documents(self, doc1_sections, doc2_sections):
        """Merge sections by combining content from both documents"""
        combined = {}
        all_sections = set(doc1_sections.keys()) | set(doc2_sections.keys())
        
        for section in all_sections:
            content_parts = []
            
            if section in doc1_sections:
                for chunk in doc1_sections[section]:
                    content_parts.append(f"% From Document 1\n{chunk['content']}")
            
            if section in doc2_sections:
                for chunk in doc2_sections[section]:
                    content_parts.append(f"% From Document 2\n{chunk['content']}")
            
            combined[section] = "\n\n".join(content_parts)
        
        return self._format_combined_document(combined)
    
    def _interleave_documents(self, doc1_sections, doc2_sections):
        """Interleave sections from both documents"""
        combined = {}
        all_sections = sorted(set(doc1_sections.keys()) | set(doc2_sections.keys()))
        
        for section in all_sections:
            content_parts = []
            
            # Interleave chunks from both documents
            doc1_chunks = doc1_sections.get(section, [])
            doc2_chunks = doc2_sections.get(section, [])
            
            max_chunks = max(len(doc1_chunks), len(doc2_chunks))
            
            for i in range(max_chunks):
                if i < len(doc1_chunks):
                    content_parts.append(f"% Document 1 - Chunk {i+1}\n{doc1_chunks[i]['content']}")
                if i < len(doc2_chunks):
                    content_parts.append(f"% Document 2 - Chunk {i+1}\n{doc2_chunks[i]['content']}")
            
            combined[section] = "\n\n".join(content_parts)
        
        return self._format_combined_document(combined)
    
    def _append_documents(self, doc1_sections, doc2_sections):
        """Append second document after first document"""
        combined = {}
        
        # Add all sections from doc1
        for section, chunks in doc1_sections.items():
            combined[f"Part I - {section}"] = "\n\n".join([chunk['content'] for chunk in chunks])
        
        # Add all sections from doc2
        for section, chunks in doc2_sections.items():
            combined[f"Part II - {section}"] = "\n\n".join([chunk['content'] for chunk in chunks])
        
        return self._format_combined_document(combined)
    
    def _smart_merge_documents(self, doc1_sections, doc2_sections):
        """Intelligently merge documents based on content similarity"""
        combined = {}
        
        # Priority order for sections
        section_priority = [
            'Abstract', 'Introduction', 'Background', 'Methodology', 'Methods',
            'Results', 'Discussion', 'Conclusion', 'References'
        ]
        
        # Handle priority sections first
        for priority_section in section_priority:
            matching_sections = []
            
            # Find matching sections in both documents
            for section in doc1_sections.keys():
                if priority_section.lower() in section.lower():
                    matching_sections.append(('doc1', section))
            
            for section in doc2_sections.keys():
                if priority_section.lower() in section.lower():
                    matching_sections.append(('doc2', section))
            
            if matching_sections:
                content_parts = []
                for doc_id, section in matching_sections:
                    source_sections = doc1_sections if doc_id == 'doc1' else doc2_sections
                    for chunk in source_sections[section]:
                        content_parts.append(chunk['content'])
                
                combined[priority_section] = "\n\n".join(content_parts)
        
        # Handle remaining sections
        remaining_sections = set(doc1_sections.keys()) | set(doc2_sections.keys())
        handled_sections = set()
        
        for priority_section in section_priority:
            for section in list(remaining_sections):
                if priority_section.lower() in section.lower():
                    handled_sections.add(section)
        
        remaining_sections -= handled_sections
        
        for section in remaining_sections:
            content_parts = []
            
            if section in doc1_sections:
                for chunk in doc1_sections[section]:
                    content_parts.append(chunk['content'])
            
            if section in doc2_sections:
                for chunk in doc2_sections[section]:
                    content_parts.append(chunk['content'])
            
            combined[section] = "\n\n".join(content_parts)
        
        return self._format_combined_document(combined)
    
    def _format_combined_document(self, combined_sections):
        """Format combined sections into a complete document"""
        document_parts = []
        
        # Document header
        document_parts.extend([
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{amsmath}",
            "\\usepackage{graphicx}",
            "",
            "\\title{Combined Document}",
            "\\author{Document Combination System}",
            "\\date{\\today}",
            "",
            "\\begin{document}",
            "\\maketitle",
            ""
        ])
        
        # Add sections
        for section_name, content in combined_sections.items():
            if content.strip():
                document_parts.append(f"\\section{{{section_name}}}")
                document_parts.append(content)
                document_parts.append("")
        
        document_parts.append("\\end{document}")
        
        return "\n".join(document_parts)