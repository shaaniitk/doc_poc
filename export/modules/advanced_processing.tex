"""Advanced LLM processing strategies for exceptional document refactoring"""
import json
from .llm_client import UnifiedLLMClient

class AdvancedProcessor:
    def __init__(self):
        self.llm_client = UnifiedLLMClient()
        self.global_context = {}
        self.section_dependencies = {}
        
    def multi_pass_processing(self, chunks, section_name, prompt):
        """Multi-pass processing: analyze, refine, validate"""
        # Pass 1: Analysis
        analysis = self._analyze_content(chunks, section_name)
        
        # Pass 2: Initial processing with analysis context
        initial_result = self._process_with_context(chunks, prompt, analysis)
        
        # Pass 3: Self-critique and refinement
        final_result = self._refine_output(initial_result, prompt, analysis)
        
        return final_result
    
    def _analyze_content(self, chunks, section_name):
        """Analyze content structure and requirements"""
        combined_content = '\n'.join([c['content'] for c in chunks])
        
        analysis_prompt = f"""Analyze this content for section '{section_name}':
1. Key concepts and themes
2. Technical complexity level
3. Required writing style
4. Critical elements to preserve
5. Logical flow requirements

Content: {combined_content[:1500]}"""
        
        return self.llm_client.call_llm(analysis_prompt, max_tokens=300)
    
    def _process_with_context(self, chunks, prompt, analysis):
        """Process with enhanced context"""
        combined_content = '\n'.join([c['content'] for c in chunks])
        
        enhanced_prompt = f"""CONTENT ANALYSIS: {analysis}

GLOBAL CONTEXT: {self._get_global_context()}

TASK: {prompt}

CONTENT: {combined_content}"""
        
        return self.llm_client.call_llm(enhanced_prompt)
    
    def _refine_output(self, initial_result, original_prompt, analysis):
        """Self-critique and refinement"""
        critique_prompt = f"""Review and improve this output:

ORIGINAL TASK: {original_prompt}
CONTENT ANALYSIS: {analysis}

OUTPUT TO REVIEW: {initial_result}

Provide an improved version that:
1. Better addresses the original task
2. Improves clarity and flow
3. Maintains technical accuracy
4. Preserves all LaTeX elements"""
        
        return self.llm_client.call_llm(critique_prompt)
    
    def cross_section_validation(self, processed_sections):
        """Validate consistency across sections"""
        validation_results = {}
        
        for section_name, content in processed_sections.items():
            consistency_check = self._check_consistency(section_name, content, processed_sections)
            validation_results[section_name] = consistency_check
        
        return validation_results
    
    def _check_consistency(self, section_name, content, all_sections):
        """Check consistency with other sections"""
        other_sections = {k: v for k, v in all_sections.items() if k != section_name}
        
        consistency_prompt = f"""Check consistency of '{section_name}' with other sections:

CURRENT SECTION: {content[:800]}

OTHER SECTIONS SUMMARY: {str(other_sections)[:1000]}

Report any inconsistencies in:
1. Terminology usage
2. Technical claims
3. Reference consistency
4. Writing style"""
        
        return self.llm_client.call_llm(consistency_prompt, max_tokens=200)
    
    def adaptive_prompting(self, chunk_content, section_name, base_prompt):
        """Adapt prompts based on content characteristics"""
        content_analysis = self._analyze_chunk_characteristics(chunk_content)
        
        if "mathematical" in content_analysis.lower():
            adapted_prompt = f"{base_prompt}\n\nEMPHASIS: This content is mathematical. Ensure precise notation and clear explanations of formulas."
        elif "experimental" in content_analysis.lower():
            adapted_prompt = f"{base_prompt}\n\nEMPHASIS: This content is experimental. Focus on methodology clarity and result interpretation."
        elif "implementation" in content_analysis.lower():
            adapted_prompt = f"{base_prompt}\n\nEMPHASIS: This content is implementation-focused. Ensure code clarity and technical accuracy."
        else:
            adapted_prompt = base_prompt
        
        return adapted_prompt
    
    def _analyze_chunk_characteristics(self, content):
        """Analyze chunk characteristics for adaptive prompting"""
        analysis_prompt = f"""Classify this content's primary characteristic. Respond with one word: 'mathematical', 'experimental', 'implementation', 'theoretical', or 'general'.

Content: {content[:500]}"""
        
        return self.llm_client.call_llm(analysis_prompt, max_tokens=10)
    
    def iterative_refinement(self, content, target_quality_metrics):
        """Iteratively refine until quality metrics are met"""
        current_content = content
        iteration = 0
        max_iterations = 3
        
        while iteration < max_iterations:
            quality_score = self._assess_quality(current_content, target_quality_metrics)
            
            if quality_score >= target_quality_metrics.get('threshold', 0.8):
                break
            
            current_content = self._improve_content(current_content, quality_score)
            iteration += 1
        
        return current_content
    
    def _assess_quality(self, content, metrics):
        """Assess content quality against metrics"""
        assessment_prompt = f"""Rate this content quality (0.0-1.0) for:
1. Clarity
2. Technical accuracy
3. Professional tone
4. Logical flow

Content: {content[:1000]}

Respond with only a number between 0.0 and 1.0."""
        
        try:
            score = float(self.llm_client.call_llm(assessment_prompt, max_tokens=10))
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5
    
    def _improve_content(self, content, current_score):
        """Improve content based on quality assessment"""
        improvement_prompt = f"""Improve this content (current quality: {current_score:.2f}):

{content}

Focus on:
1. Enhancing clarity
2. Improving technical precision
3. Better logical flow
4. Professional academic tone"""
        
        return self.llm_client.call_llm(improvement_prompt)
    
    def _get_global_context(self):
        """Get accumulated global context"""
        return json.dumps(self.global_context, indent=2)[:500]
    
    def update_global_context(self, section_name, processed_content):
        """Update global context with new section"""
        summary_prompt = f"""Summarize key points from this section in 2-3 sentences:

{processed_content[:800]}"""
        
        summary = self.llm_client.call_llm(summary_prompt, max_tokens=100)
        self.global_context[section_name] = summary