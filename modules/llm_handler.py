"""LLM interaction module with semantic validation and smart context management"""
import os
from .llm_client import UnifiedLLMClient
from .format_enforcer import FormatEnforcer
from .error_handler import robust_llm_call, LLMError
from .semantic_validator import SemanticValidator, SmartContextManager

class ContextualLLMHandler:
    def __init__(self, provider=None, model=None, output_format="latex"):
        self.document_context = ""
        self.section_contexts = {}
        self.llm_client = UnifiedLLMClient(provider, model)
        self.format_enforcer = FormatEnforcer(output_format)
        
        # Semantic validation and smart context management
        self.semantic_validator = SemanticValidator()
        self.context_manager = SmartContextManager()
        
        # Track processing quality metrics
        self.quality_metrics = {
            'sections_processed': 0,
            'validation_failures': 0,
            'coherence_scores': [],
            'quality_scores': []
        }
    
    @robust_llm_call(max_retries=2)
    def process_section(self, section_name, content, prompt):
        """Process a section with semantic validation and smart context management"""
        self.quality_metrics['sections_processed'] += 1
        
        # Pre-processing validation
        is_valid_input, input_quality = self.semantic_validator.validate_content_quality(content)
        if not is_valid_input:
            print(f"Warning: Input content for {section_name} has low quality (score: {input_quality:.2f})")
            self.quality_metrics['validation_failures'] += 1
        
        # Build smart context using context manager
        smart_context = self.context_manager.build_context(
            [section_name], 
            [content] + list(self.section_contexts.values())
        )
        
        # Build context with semantic awareness
        context_parts = []
        if smart_context:
            context_parts.append(f"SEMANTIC CONTEXT:\n{smart_context}")
        if self.document_context:
            context_parts.append(f"DOCUMENT CONTEXT:\n{self.document_context}")
        if section_name in self.section_contexts:
            # Check coherence with previous section content
            coherence_score = self.semantic_validator.validate_content_coherence(
                content, self.section_contexts[section_name]
            )
            self.quality_metrics['coherence_scores'].append(coherence_score)
            context_parts.append(f"SECTION CONTEXT (coherence: {coherence_score:.2f}):\n{self.section_contexts[section_name]}")
        
        context = "\n\n".join(context_parts)
        
        # Build enhanced prompt with semantic requirements
        full_prompt = f"{context}\n\nTASK: {prompt}\n\nSEMANTIC REQUIREMENTS:\n- Maintain semantic consistency with existing content\n- Ensure logical flow and coherence\n- Preserve technical accuracy and detail\n- Use appropriate academic/technical tone\n\nIMPORTANT: Output ONLY section content, no document structure. Preserve all LaTeX environments exactly.\n\nCONTENT:\n{content}"
        
        system_prompt = self.format_enforcer.get_system_prompt()
        
        # Get LLM response
        raw_result = self.llm_client.call_llm(full_prompt, system_prompt)
        
        # Enforce format
        result, format_issues = self.format_enforcer.enforce_format(raw_result)
        
        # Post-processing semantic validation
        is_valid_output, output_quality = self.semantic_validator.validate_content_quality(result)
        self.quality_metrics['quality_scores'].append(output_quality)
        
        if not is_valid_output:
            print(f"Warning: Output for {section_name} has low quality (score: {output_quality:.2f})")
            self.quality_metrics['validation_failures'] += 1
        else:
            print(f"Processed {section_name} with quality score: {output_quality:.2f}")
        
        # Validate coherence with input if both are valid
        if is_valid_input and is_valid_output:
            coherence = self.semantic_validator.validate_content_coherence(content, result)
            if coherence < 0.3:
                print(f"Warning: Low coherence between input and output for {section_name} (score: {coherence:.2f})")
        
        # Log format issues if any
        if format_issues:
            print(f"Format issues in {section_name}: {format_issues}")
        
        # Update contexts with semantic awareness
        self.update_context(section_name, result)
        
        return result
    
    def update_context(self, section_name, new_content):
        """Update document and section contexts with semantic awareness"""
        # Validate new content before updating context
        is_valid, quality_score = self.semantic_validator.validate_content_quality(new_content)
        
        if not is_valid:
            print(f"Warning: Not updating context with low-quality content from {section_name}")
            return
        
        # Create semantic summary using context manager
        summary = self.context_manager.create_section_summary(section_name, new_content)
        
        # Update section context with validated content
        self.section_contexts[section_name] = summary
        
        # Smart context management with semantic relevance
        if len(self.document_context) > 2000:
            # Use context manager to prioritize most relevant content
            all_sections = list(self.section_contexts.keys())
            relevant_context = self.context_manager.build_context(
                [section_name], 
                list(self.section_contexts.values())
            )
            self.document_context = relevant_context[:1000]
        else:
            self.document_context += f"\n{section_name}: {summary[:150]}"
        
        # Special handling for Summary section - use full document context
        if section_name == "Summary":
            self.document_context = f"FULL DOCUMENT SUMMARY: {summary[:800]}"
    
    def get_quality_report(self):
        """Get processing quality metrics report"""
        if not self.quality_metrics['sections_processed']:
            return "No sections processed yet."
        
        avg_quality = sum(self.quality_metrics['quality_scores']) / len(self.quality_metrics['quality_scores']) if self.quality_metrics['quality_scores'] else 0
        avg_coherence = sum(self.quality_metrics['coherence_scores']) / len(self.quality_metrics['coherence_scores']) if self.quality_metrics['coherence_scores'] else 0
        failure_rate = self.quality_metrics['validation_failures'] / self.quality_metrics['sections_processed']
        
        return f"""Quality Report:
- Sections processed: {self.quality_metrics['sections_processed']}
- Average quality score: {avg_quality:.2f}
- Average coherence score: {avg_coherence:.2f}
- Validation failure rate: {failure_rate:.1%}
- Total validation failures: {self.quality_metrics['validation_failures']}"""