"""LLM interaction module with context management"""
import os
from .llm_client import UnifiedLLMClient

class ContextualLLMHandler:
    def __init__(self):
        self.document_context = ""
        self.section_contexts = {}
        
    def __init__(self, provider=None, model=None):
        self.document_context = ""
        self.section_contexts = {}
        self.llm_client = UnifiedLLMClient(provider, model)
    
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
        full_prompt = f"{context}\n\nTASK: {prompt}\n\nPreserve all LaTeX environments exactly.\n\nCONTENT:\n{content}"
        
        from config import CHUNKING_PROMPTS
        system_prompt = CHUNKING_PROMPTS['system_prompt']
        
        # Get LLM response
        result = self.llm_client.call_llm(full_prompt, system_prompt)
        
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