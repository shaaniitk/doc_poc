"""Intelligent document aggregation with coherence optimization"""
from .llm_client import UnifiedLLMClient

class IntelligentAggregator:
    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        
    def coherence_optimization(self, processed_sections):
        """Optimize document coherence through intelligent transitions"""
        optimized_sections = {}
        section_order = list(processed_sections.keys())
        
        for i, section_name in enumerate(section_order):
            content = processed_sections[section_name]
            
            # Add intelligent transitions
            if i > 0:
                prev_section = section_order[i-1]
                transition = self._generate_transition(
                    processed_sections[prev_section], 
                    content, 
                    prev_section, 
                    section_name
                )
                content = transition + "\n\n" + content
            
            # Add forward references if needed
            if i < len(section_order) - 1:
                next_section = section_order[i+1]
                content = self._add_forward_references(
                    content, 
                    processed_sections[next_section], 
                    next_section
                )
            
            optimized_sections[section_name] = content
        
        return optimized_sections
    
    def _generate_transition(self, prev_content, current_content, prev_section, current_section):
        """Generate smooth transitions between sections"""
        transition_prompt = f"""Create a 1-2 sentence transition from '{prev_section}' to '{current_section}'.

Previous section ending: {prev_content[-300:]}
Current section beginning: {current_content[:300]}

The transition should:
1. Connect the ideas smoothly
2. Maintain academic tone
3. Be concise and natural"""
        
        return self.llm_client.call_llm(transition_prompt, max_tokens=100)
    
    def _add_forward_references(self, content, next_content, next_section):
        """Add forward references where appropriate"""
        reference_prompt = f"""Should this section reference the upcoming '{next_section}'? 
If yes, suggest a brief forward reference (1 sentence). If no, respond 'NONE'.

Current content: {content[-400:]}
Next section preview: {next_content[:200]}"""
        
        reference = self.llm_client.call_llm(reference_prompt, max_tokens=50)
        
        if reference.strip() != 'NONE' and len(reference.strip()) > 10:
            return content + "\n\n" + reference.strip()
        
        return content
    
    def terminology_consistency(self, processed_sections):
        """Ensure consistent terminology throughout document"""
        # Extract key terms
        key_terms = self._extract_key_terms(processed_sections)
        
        # Standardize usage
        standardized_sections = {}
        for section_name, content in processed_sections.items():
            standardized_sections[section_name] = self._standardize_terminology(content, key_terms)
        
        return standardized_sections
    
    def _extract_key_terms(self, sections):
        """Extract key technical terms from all sections"""
        all_content = '\n'.join(sections.values())
        
        extraction_prompt = f"""Extract 10-15 key technical terms from this document. 
Return as comma-separated list.

Content: {all_content[:2000]}"""
        
        terms_text = self.llm_client.call_llm(extraction_prompt, max_tokens=200)
        return [term.strip() for term in terms_text.split(',') if term.strip()]
    
    def _standardize_terminology(self, content, key_terms):
        """Standardize terminology usage in content"""
        standardization_prompt = f"""Ensure consistent usage of these key terms: {', '.join(key_terms)}

Review and correct any inconsistent terminology in:

{content}

Maintain the same meaning while ensuring consistency."""
        
        return self.llm_client.call_llm(standardization_prompt)
    
    def document_flow_optimization(self, sections):
        """Optimize overall document flow and structure"""
        flow_analysis = self._analyze_document_flow(sections)
        
        optimization_prompt = f"""Based on this flow analysis, suggest structural improvements:

{flow_analysis}

Current sections: {list(sections.keys())}

Suggest:
1. Section reordering (if needed)
2. Content redistribution
3. Missing connections"""
        
        suggestions = self.llm_client.call_llm(optimization_prompt, max_tokens=300)
        
        return self._apply_flow_optimizations(sections, suggestions)
    
    def _analyze_document_flow(self, sections):
        """Analyze logical flow of document"""
        section_summaries = {}
        for name, content in sections.items():
            summary_prompt = f"Summarize the main purpose and key points of this section in 2 sentences:\n\n{content[:800]}"
            section_summaries[name] = self.llm_client.call_llm(summary_prompt, max_tokens=100)
        
        flow_prompt = f"""Analyze the logical flow of these sections:

{section_summaries}

Rate the flow quality and identify any issues."""
        
        return self.llm_client.call_llm(flow_prompt, max_tokens=200)
    
    def _apply_flow_optimizations(self, sections, suggestions):
        """Apply flow optimization suggestions"""
        # For now, return original sections
        # In practice, this would parse suggestions and apply changes
        return sections
    
    def quality_assurance_pass(self, final_document):
        """Final quality assurance pass"""
        qa_checks = {
            'completeness': self._check_completeness(final_document),
            'consistency': self._check_consistency(final_document),
            'clarity': self._check_clarity(final_document),
            'technical_accuracy': self._check_technical_accuracy(final_document)
        }
        
        return qa_checks
    
    def _check_completeness(self, document):
        """Check document completeness"""
        completeness_prompt = f"""Check if this document is complete and well-structured:

{document[:2000]}

Rate completeness (0.0-1.0) and list any missing elements."""
        
        return self.llm_client.call_llm(completeness_prompt, max_tokens=150)
    
    def _check_consistency(self, document):
        """Check internal consistency"""
        consistency_prompt = f"""Check for internal consistency issues:

{document[:2000]}

Identify any contradictions or inconsistencies."""
        
        return self.llm_client.call_llm(consistency_prompt, max_tokens=150)
    
    def _check_clarity(self, document):
        """Check clarity and readability"""
        clarity_prompt = f"""Assess clarity and readability:

{document[:2000]}

Rate clarity (0.0-1.0) and suggest improvements."""
        
        return self.llm_client.call_llm(clarity_prompt, max_tokens=150)
    
    def _check_technical_accuracy(self, document):
        """Check technical accuracy"""
        accuracy_prompt = f"""Review technical accuracy:

{document[:2000]}

Identify any technical errors or unclear explanations."""
        
        return self.llm_client.call_llm(accuracy_prompt, max_tokens=150)