"""Validation and quality control nodes for document processing pipeline."""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .models import WorkflowState, ProcessedSection
from .template_processor_node import PromptOrchestrator

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"

@dataclass
class ValidationResult:
    """Result of content validation."""
    is_valid: bool
    confidence_score: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]
    validated_content: Optional[str] = None

class ValidationNode:
    """Node for validating and improving processed content quality."""
    
    def __init__(self, llm_handler, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.llm_handler = llm_handler
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        self.use_mock = llm_handler is None
        self.prompt_orchestrator = PromptOrchestrator()
        
    async def validate_section(self, section: ProcessedSection, 
                             original_chunks: List[Any],
                             section_requirements: Dict[str, Any]) -> ValidationResult:
        """Validate a single processed section."""
        try:
            # Get validation prompt based on section type and requirements
            validation_prompt = self._get_validation_prompt(
                section, original_chunks, section_requirements
            )
            
            # Run validation through LLM
            validation_response = await self._run_validation(validation_prompt)
            
            # Parse validation response
            result = self._parse_validation_response(validation_response, section)
            
            logger.info(f"Validated section '{section.section_name}': "
                       f"valid={result.is_valid}, confidence={result.confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for section '{section.section_name}': {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Manual review required"]
            )
    
    async def validate_workflow_state(self, state) -> Dict[str, ValidationResult]:
        """Validate all processed sections in the workflow state."""
        validation_results = {}
        
        # Handle both dict and WorkflowState objects
        if isinstance(state, dict):
            processed_sections = state.get('processed_sections', [])
        else:
            processed_sections = getattr(state, 'processed_sections', [])
            
        if not processed_sections:
            logger.warning("No processed sections found for validation")
            return validation_results
        
        # Handle both dict and list formats for processed_sections
        sections_to_validate = []
        if isinstance(processed_sections, dict):
            for section_name, section_data in processed_sections.items():
                if isinstance(section_data, dict):
                    # Convert dict to ProcessedSection-like object
                    section = type('ProcessedSection', (), {
                        'section_name': section_name,
                        'content': section_data.get('content', ''),
                        'confidence': section_data.get('confidence', 0.0),
                        'source_chunks': section_data.get('source_chunks', 0)
                    })()
                    sections_to_validate.append(section)
        elif isinstance(processed_sections, list):
            for section_data in processed_sections:
                if isinstance(section_data, dict):
                    section = type('ProcessedSection', (), {
                        'section_name': section_data.get('section_name', 'unknown'),
                        'content': section_data.get('content', ''),
                        'confidence': section_data.get('confidence', 0.0),
                        'source_chunks': section_data.get('source_chunks', 0)
                    })()
                    sections_to_validate.append(section)
                else:
                    # Assume it's already a ProcessedSection object
                    sections_to_validate.append(section_data)
        
        # Validate each section
        for section in sections_to_validate:
            section_requirements = self._get_section_requirements(section.section_name)
            original_chunks = getattr(state, 'chunks', [])
            
            validation_result = await self.validate_section(
                section, original_chunks, section_requirements
            )
            validation_results[section.section_name] = validation_result
        
        return validation_results
    
    def _get_validation_prompt(self, section: Any, original_chunks: List[Any], 
                              requirements: Dict[str, Any]) -> str:
        """Generate validation prompt for a section."""
        
        # Get base validation prompt from orchestrator
        section_type = getattr(section, 'section_name', 'general')
        base_prompts = self.prompt_orchestrator.get_validation_prompts(section_type)
        
        if not base_prompts:
            # Fallback validation prompt
            return self._create_fallback_validation_prompt(section, requirements)
        
        # Use the first available validation prompt as base
        base_prompt = base_prompts[0]
        
        # Customize prompt for this specific section
        section_context = f"""
Section Name: {section.section_name}
Section Content:
{section.content}

Original Source Chunks: {len(original_chunks)} chunks available
Section Confidence: {getattr(section, 'confidence', 0.0):.2f}
Source Chunks Used: {getattr(section, 'source_chunks', 0)}

Section Requirements:
{self._format_requirements(requirements)}
"""
        
        return f"{base_prompt}\n\n{section_context}"
    
    def _create_fallback_validation_prompt(self, section: Any, requirements: Dict[str, Any]) -> str:
        """Create a fallback validation prompt when none are available."""
        return f"""
Please validate the following section content for quality and accuracy:

Section: {section.section_name}
Content:
{section.content}

Validation Criteria:
1. Content relevance and accuracy
2. Proper structure and formatting
3. Completeness based on requirements
4. Clarity and readability
5. Consistency with source material

Requirements:
{self._format_requirements(requirements)}

Please provide:
1. VALID: true/false
2. CONFIDENCE: 0.0-1.0 score
3. ISSUES: List any problems found
4. SUGGESTIONS: Recommendations for improvement
5. IMPROVED_CONTENT: If needed, provide corrected version

Format your response as:
VALID: [true/false]
CONFIDENCE: [0.0-1.0]
ISSUES: [comma-separated list]
SUGGESTIONS: [comma-separated list]
IMPROVED_CONTENT: [improved content if needed]
"""
    
    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format section requirements for the prompt."""
        if not requirements:
            return "No specific requirements provided"
        
        formatted = []
        for key, value in requirements.items():
            formatted.append(f"- {key}: {value}")
        
        return "\n".join(formatted)
    
    async def _run_validation(self, prompt: str) -> str:
        """Run validation prompt through LLM."""
        try:
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            raise
    
    def _parse_validation_response(self, response: str, section: Any) -> ValidationResult:
        """Parse LLM validation response into ValidationResult."""
        try:
            # Parse structured response
            lines = response.strip().split('\n')
            
            is_valid = False
            confidence_score = 0.0
            issues = []
            suggestions = []
            improved_content = None
            
            current_section = None
            content_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('VALID:'):
                    is_valid = 'true' in line.lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence_score = float(line.split(':', 1)[1].strip())
                    except (ValueError, IndexError):
                        confidence_score = 0.5  # Default
                elif line.startswith('ISSUES:'):
                    issues_text = line.split(':', 1)[1].strip()
                    if issues_text and issues_text != 'None':
                        issues = [i.strip() for i in issues_text.split(',') if i.strip()]
                elif line.startswith('SUGGESTIONS:'):
                    suggestions_text = line.split(':', 1)[1].strip()
                    if suggestions_text and suggestions_text != 'None':
                        suggestions = [s.strip() for s in suggestions_text.split(',') if s.strip()]
                elif line.startswith('IMPROVED_CONTENT:'):
                    current_section = 'content'
                    content_text = line.split(':', 1)[1].strip()
                    if content_text:
                        content_lines.append(content_text)
                elif current_section == 'content' and line:
                    content_lines.append(line)
            
            if content_lines:
                improved_content = '\n'.join(content_lines)
            
            return ValidationResult(
                is_valid=is_valid,
                confidence_score=max(0.0, min(1.0, confidence_score)),
                issues=issues,
                suggestions=suggestions,
                validated_content=improved_content
            )
            
        except Exception as e:
            logger.error(f"Failed to parse validation response: {e}")
            # Fallback: assume content needs review
            return ValidationResult(
                is_valid=False,
                confidence_score=0.3,
                issues=["Could not parse validation response"],
                suggestions=["Manual review recommended"]
            )
    
    def _get_section_requirements(self, section_name: str) -> Dict[str, Any]:
        """Get validation requirements for a specific section type."""
        
        # Default requirements by section type
        requirements_map = {
            'introduction': {
                'min_length': 100,
                'should_contain': ['overview', 'purpose', 'context'],
                'structure': 'Clear introduction with context and purpose'
            },
            'methodology': {
                'min_length': 150,
                'should_contain': ['approach', 'method', 'process'],
                'structure': 'Detailed explanation of methods and approaches'
            },
            'results': {
                'min_length': 100,
                'should_contain': ['findings', 'data', 'analysis'],
                'structure': 'Clear presentation of findings and results'
            },
            'conclusion': {
                'min_length': 80,
                'should_contain': ['summary', 'implications'],
                'structure': 'Concise summary with key takeaways'
            }
        }
        
        return requirements_map.get(section_name.lower(), {
            'min_length': 50,
            'structure': 'Well-structured content appropriate for section type'
        })

class QualityControlNode:
    """Node for overall document quality control and improvement."""
    
    def __init__(self, llm, validation_node: ValidationNode):
        self.llm = llm
        self.validation_node = validation_node
    
    async def improve_document_quality(self, state, 
                                     validation_results: Dict[str, ValidationResult]):
        """Improve document quality based on validation results."""
        
        improved_sections = []
        
        # Process validation results and improve content
        for section_name, validation_result in validation_results.items():
            if not validation_result.is_valid and validation_result.validated_content:
                # Use improved content from validation
                improved_sections.append({
                    'section_name': section_name,
                    'content': validation_result.validated_content,
                    'confidence': min(1.0, validation_result.confidence_score + 0.2),
                    'validation_applied': True,
                    'original_issues': validation_result.issues
                })
                logger.info(f"Applied validation improvements to section '{section_name}'")
            else:
                # Keep original content but mark as validated
                original_section = self._find_section_in_state(state, section_name)
                if original_section:
                    improved_sections.append({
                        'section_name': section_name,
                        'content': original_section.get('content', ''),
                        'confidence': original_section.get('confidence', 0.0),
                        'validation_applied': False,
                        'validation_passed': validation_result.is_valid
                    })
        
        # Update state with improved sections
        if isinstance(state, dict):
            # Handle dict state
            if 'processed_sections' in state:
                if isinstance(state['processed_sections'], list):
                    state['processed_sections'] = improved_sections
                elif isinstance(state['processed_sections'], dict):
                    # Convert back to dict format if needed
                    improved_dict = {}
                    for section in improved_sections:
                        improved_dict[section['section_name']] = section
                    state['processed_sections'] = improved_dict
            # Add validation metadata
            state['validation_results'] = validation_results
        else:
            # Handle WorkflowState object
            if hasattr(state, 'processed_sections'):
                if isinstance(state.processed_sections, list):
                    state.processed_sections = improved_sections
                elif isinstance(state.processed_sections, dict):
                    # Convert back to dict format if needed
                    improved_dict = {}
                    for section in improved_sections:
                        improved_dict[section['section_name']] = section
                    state.processed_sections = improved_dict
            # Add validation metadata
            if not hasattr(state, 'validation_results'):
                state.validation_results = validation_results
        
        logger.info(f"Quality control completed: {len(improved_sections)} sections processed")
        return state
    
    def _find_section_in_state(self, state, section_name: str) -> Optional[Dict[str, Any]]:
        """Find a section in the workflow state."""
        processed_sections = None
        
        if isinstance(state, dict):
            processed_sections = state.get('processed_sections')
        elif hasattr(state, 'processed_sections'):
            processed_sections = state.processed_sections
        
        if processed_sections is None:
            return None
        
        if isinstance(processed_sections, dict):
            return processed_sections.get(section_name)
        elif isinstance(processed_sections, list):
            for section in processed_sections:
                if isinstance(section, dict) and section.get('section_name') == section_name:
                    return section
        
        return None