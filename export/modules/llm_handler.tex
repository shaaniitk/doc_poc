"""LLM interaction module with context management"""
import os
from .llm_client import UnifiedLLMClient
from .format_enforcer import FormatEnforcer

class ContextualLLMHandler:
    def __init__(self, provider=None, model=None, output_format="latex"):
        self.document_context = ""
        self.section_contexts = {}
        self.llm_client = UnifiedLLMClient(provider, model)
        self.format_enforcer = FormatEnforcer(output_format)
    
    def process_section(self, section_name, content, prompt):
        """Process a section with contextual awareness"""
        # Build context
        context_parts = []
        if self.document_context:
            context_parts.append(f"DOCUMENT CONTEXT:\n{self.document_context}")
        if section_name in self.section_contexts:
            context_parts.append(f"SECTION CONTEXT:\n{self.section_contexts[section_name]}")
        
        context = "\n\n".join(context_parts)
        
        # Build full prompt
        full_prompt = f"{context}\n\nTASK: {prompt}\n\nIMPORTANT: Output ONLY section content, no document structure. Preserve all LaTeX environments exactly.\n\nCONTENT:\n{content}"
        
        system_prompt = self.format_enforcer.get_system_prompt()
        
        # Get LLM response
        raw_result = self.llm_client.call_llm(full_prompt, system_prompt)
        
        # Enforce format
        result, format_issues = self.format_enforcer.enforce_format(raw_result)
        
        # Log format issues if any
        if format_issues:
            print(f"Format issues in {section_name}: {format_issues}")
        
        # Update contexts
        self.update_context(section_name, result)
        
        return result
    
    def update_context(self, section_name, new_content):
        """Update document and section contexts"""
        # Update section context (keep last 500 chars)
        summary = new_content[:500] + "..." if len(new_content) > 500 else new_content
        self.section_contexts[section_name] = summary
        
        # Update document context (accumulate key points)
        if len(self.document_context) > 1000:
            self.document_context = self.document_context[-500:] + f"\n{section_name}: {summary[:200]}"
        else:
            self.document_context += f"\n{section_name}: {summary[:200]}"
        
        # Special handling for Summary section - use full document context
        if section_name == "Summary":
            self.document_context = f"FULL DOCUMENT SUMMARY: {summary[:800]}"