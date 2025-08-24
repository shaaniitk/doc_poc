"""Simplified, robust document combiner with fallback strategies"""
from .file_loader import load_latex_file
from .chunker import extract_latex_sections, group_chunks_by_section
from .format_enforcer import FormatEnforcer
from .llm_client import UnifiedLLMClient
from .error_handler import robust_llm_call, LLMError, ProcessingError
import re

class RobustDocumentCombiner:
    """Simplified document combiner with robust error handling"""
    
    def __init__(self, output_format="latex"):
        self.format_enforcer = FormatEnforcer(output_format)
        self.llm_client = UnifiedLLMClient()
    
    def combine_documents(self, doc1_path, doc2_path, strategy="smart_merge", output_format="latex"):
        """Combine documents with robust error handling"""
        try:
            # Load documents with validation
            doc1_content = self._safe_load_document(doc1_path)
            doc2_content = self._safe_load_document(doc2_path)
            
            # Extract chunks with error handling
            doc1_chunks = extract_latex_sections(doc1_content)
            doc2_chunks = extract_latex_sections(doc2_content)
            
            doc1_grouped = group_chunks_by_section(doc1_chunks)
            doc2_grouped = group_chunks_by_section(doc2_chunks)
            
            # Apply combination strategy with fallbacks
            if strategy == "smart_merge":
                combined_content = self._robust_smart_merge(doc1_grouped, doc2_grouped)
            else:
                combined_content = self._simple_append(doc1_grouped, doc2_grouped)
            
            # Format with validation
            formatted_content, issues = self.format_enforcer.enforce_format(combined_content)
            
            return formatted_content, issues
            
        except Exception as e:
            raise ProcessingError(f"Document combination failed: {e}")
    
    def _safe_load_document(self, doc_path):
        """Safely load document with validation"""
        try:
            content = load_latex_file(doc_path)
            if not content or len(content.strip()) < 50:
                raise ProcessingError(f"Document {doc_path} is empty or too short")
            return content
        except Exception as e:
            raise ProcessingError(f"Failed to load {doc_path}: {e}")
    
    def _robust_smart_merge(self, doc1_sections, doc2_sections):
        """Smart merge with fallback to simple append"""
        try:
            return self._llm_enhanced_merge(doc1_sections, doc2_sections)
        except LLMError:
            print("LLM merge failed, falling back to simple append")
            return self._simple_append(doc1_sections, doc2_sections)
    
    @robust_llm_call(max_retries=1)
    def _llm_enhanced_merge(self, doc1_sections, doc2_sections):
        """LLM-enhanced merge with strict error handling"""
        combined = {}
        
        # Start with original document structure
        for section_name, chunks in doc1_sections.items():
            original_content = '\n\n'.join([chunk['content'] for chunk in chunks])
            
            # Check if doc2 has relevant content for this section
            if section_name in doc2_sections:
                aug_content = '\n\n'.join([chunk['content'] for chunk in doc2_sections[section_name]])
                combined[section_name] = self._merge_section_content(section_name, original_content, aug_content)
            else:
                combined[section_name] = original_content
        
        # Add new sections from doc2
        for section_name, chunks in doc2_sections.items():
            if section_name not in combined:
                combined[f"Additional: {section_name}"] = '\n\n'.join([chunk['content'] for chunk in chunks])
        
        return self._format_combined_document(combined)
    
    @robust_llm_call(max_retries=1)
    def _merge_section_content(self, section_name, original_content, aug_content):
        """Merge section content with LLM assistance"""
        prompt = f"""Merge this additional content into the original section while preserving all original content:

ORIGINAL {section_name}:
{original_content[:1000]}

ADDITIONAL CONTENT:
{aug_content[:1000]}

Requirements:
1. Keep ALL original content intact
2. Add relevant additional content where appropriate
3. Maintain coherent flow
4. Preserve all LaTeX formatting

Enhanced section:"""
        
        result = self.llm_client.call_llm(prompt, max_tokens=1500)
        return result.strip()
    
    def _simple_append(self, doc1_sections, doc2_sections):
        """Simple append strategy as fallback"""
        combined = {}
        
        # Add all sections from doc1
        for section, chunks in doc1_sections.items():
            combined[section] = '\n\n'.join([chunk['content'] for chunk in chunks])
        
        # Add all sections from doc2 with prefix
        for section, chunks in doc2_sections.items():
            combined[f"Additional: {section}"] = '\n\n'.join([chunk['content'] for chunk in chunks])
        
        return self._format_combined_document(combined)
    
    def _format_combined_document(self, combined_sections):
        """Format sections into complete document"""
        document_parts = [
            "\\documentclass{article}",
            "\\usepackage{amsmath}",
            "\\usepackage{graphicx}",
            "",
            "\\begin{document}",
            ""
        ]
        
        for section_name, content in combined_sections.items():
            if content.strip():
                document_parts.append(f"\\section{{{section_name}}}")
                document_parts.append(content)
                document_parts.append("")
        
        document_parts.append("\\end{document}")
        return "\n".join(document_parts)