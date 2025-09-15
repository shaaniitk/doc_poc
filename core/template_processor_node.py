from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Optional langchain imports
try:
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Fallback classes
    class HumanMessage:
        def __init__(self, content):
            self.content = content
    
    class AIMessage:
        def __init__(self, content):
            self.content = content
    
    class ChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

# Import from root config.py since templates are defined there
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import PROMPTS

# Define DOCUMENT_TEMPLATES locally since it's not in the config
DOCUMENT_TEMPLATES = {
    "bitcoin_paper_hierarchical": {
        "sections": {
            "introduction": {
                "description": "Introduction and background",
                "keywords": ["bitcoin", "introduction", "background", "overview"]
            },
            "methodology": {
                "description": "Technical methodology and approach", 
                "keywords": ["method", "approach", "technical", "implementation"]
            },
            "results": {
                "description": "Results and findings",
                "keywords": ["results", "findings", "analysis", "data"]
            },
            "conclusion": {
                "description": "Conclusions and future work",
                "keywords": ["conclusion", "future", "summary", "implications"]
            }
        }
    }
}
from .models import DocumentChunk, ProcessedSection
from .semantic_mapper import SemanticMapper

logger = logging.getLogger(__name__)

@dataclass
class SectionMapping:
    """Represents a mapping between document chunks and template sections."""
    section_name: str
    section_prompt: str
    chunks: List[DocumentChunk]
    confidence_score: float
    subsections: List['SectionMapping'] = None

class TemplateProcessorNode:
    """Node for processing documents using hierarchical templates and section-specific prompts."""
    
    def __init__(self, llm: ChatOpenAI, semantic_mapper: SemanticMapper):
        self.llm = llm
        self.semantic_mapper = semantic_mapper
        self.logger = logging.getLogger(__name__)
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method that orchestrates document processing with templates."""
        return await self.process_document_with_template(state)
    
    async def process_document_with_template(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing function that applies template-based section processing."""
        try:
            chunks = state.get('chunks', [])
            template_name = state.get('template_name', 'bitcoin_paper_hierarchical')
            
            # Get template configuration
            template_config = DOCUMENT_TEMPLATES.get(template_name)
            if not template_config:
                self.logger.error(f"Template {template_name} not found")
                return state
            
            # Map chunks to template sections
            section_mappings = await self._map_chunks_to_sections(chunks, template_config)
            
            # Process each section with appropriate prompts
            processed_sections = await self._process_sections(section_mappings, template_config)
            
            # Update state with processed sections
            state['processed_sections'] = processed_sections
            state['template_applied'] = template_name
            
            self.logger.info(f"Successfully processed {len(processed_sections)} sections using template {template_name}")
            return state
            
        except Exception as e:
            self.logger.error(f"Error in template processing: {str(e)}")
            state['processing_errors'] = state.get('processing_errors', []) + [str(e)]
            return state
    
    async def _map_chunks_to_sections(self, chunks: List[DocumentChunk], template_config: Dict) -> List[SectionMapping]:
        """Map document chunks to template sections using semantic similarity."""
        section_mappings = []
        sections = template_config.get('sections', {})
        
        for section_name, section_data in sections.items():
            # Get section prompt and description for semantic matching
            section_prompt = section_data.get('prompt', '')
            section_description = section_data.get('description', section_name)
            
            # Find chunks that best match this section
            matched_chunks = await self._find_matching_chunks(
                chunks, section_description, section_prompt
            )
            
            if matched_chunks:
                mapping = SectionMapping(
                    section_name=section_name,
                    section_prompt=section_prompt,
                    chunks=matched_chunks['chunks'],
                    confidence_score=matched_chunks['confidence']
                )
                
                # Handle subsections if they exist
                subsections = section_data.get('subsections', {})
                if subsections:
                    mapping.subsections = await self._map_subsections(
                        matched_chunks['chunks'], subsections
                    )
                
                section_mappings.append(mapping)
        
        return section_mappings
    
    async def _find_matching_chunks(self, chunks: List[DocumentChunk], 
                                   section_description: str, section_prompt: str) -> Dict:
        """Find chunks that semantically match a section description."""
        # Use semantic mapper to find relevant chunks
        query = f"{section_description} {section_prompt}"
        
        # Score each chunk against the section
        chunk_scores = []
        for chunk in chunks:
            # Simple semantic similarity (can be enhanced with embeddings)
            similarity_score = await self.semantic_mapper.calculate_similarity(
                chunk.content, query
            )
            chunk_scores.append((chunk, similarity_score))
        
        # Sort by similarity and take top matches
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take chunks with similarity above threshold
        threshold = 0.3  # Configurable threshold
        matched_chunks = [chunk for chunk, score in chunk_scores if score > threshold]
        avg_confidence = sum(score for _, score in chunk_scores[:len(matched_chunks)]) / len(matched_chunks) if matched_chunks else 0
        
        return {
            'chunks': matched_chunks,
            'confidence': avg_confidence
        }
    
    async def _map_subsections(self, chunks: List[DocumentChunk], 
                              subsections: Dict) -> List[SectionMapping]:
        """Map chunks to subsections within a main section."""
        subsection_mappings = []
        
        for subsection_name, subsection_data in subsections.items():
            subsection_prompt = subsection_data.get('prompt', '')
            subsection_description = subsection_data.get('description', subsection_name)
            
            matched_chunks = await self._find_matching_chunks(
                chunks, subsection_description, subsection_prompt
            )
            
            if matched_chunks['chunks']:
                mapping = SectionMapping(
                    section_name=subsection_name,
                    section_prompt=subsection_prompt,
                    chunks=matched_chunks['chunks'],
                    confidence_score=matched_chunks['confidence']
                )
                subsection_mappings.append(mapping)
        
        return subsection_mappings
    
    async def _process_sections(self, section_mappings: List[SectionMapping], 
                               template_config: Dict) -> List[ProcessedSection]:
        """Process each section using appropriate prompts and LLM."""
        processed_sections = []
        
        for mapping in section_mappings:
            try:
                # Combine chunk content for this section
                section_content = "\n\n".join([chunk.content for chunk in mapping.chunks])
                
                # Apply section-specific prompt
                processed_content = await self._apply_section_prompt(
                    section_content, mapping.section_prompt, mapping.section_name
                )
                
                # Process subsections if they exist
                processed_subsections = []
                if mapping.subsections:
                    for subsection in mapping.subsections:
                        subsection_content = "\n\n".join([chunk.content for chunk in subsection.chunks])
                        processed_subsection_content = await self._apply_section_prompt(
                            subsection_content, subsection.section_prompt, subsection.section_name
                        )
                        processed_subsections.append(ProcessedSection(
                            section_name=subsection.section_name,
                            content=processed_subsection_content,
                            confidence_score=subsection.confidence_score,
                            source_chunks=subsection.chunks
                        ))
                
                processed_section = ProcessedSection(
                    section_name=mapping.section_name,
                    content=processed_content,
                    confidence_score=mapping.confidence_score,
                    source_chunks=mapping.chunks,
                    subsections=processed_subsections
                )
                
                processed_sections.append(processed_section)
                
            except Exception as e:
                self.logger.error(f"Error processing section {mapping.section_name}: {str(e)}")
                continue
        
        return processed_sections
    
    async def _apply_section_prompt(self, content: str, section_prompt: str, 
                                   section_name: str) -> str:
        """Apply section-specific prompt to content using LLM."""
        try:
            # Get appropriate processing prompt based on section type
            processing_prompt = self._get_processing_prompt(section_name)
            
            # Combine section prompt with processing instructions
            full_prompt = f"""{processing_prompt}

Section Guidelines:
{section_prompt}

Content to Process:
{content}

Please process this content according to the section guidelines above."""
            
            # Process with LLM if available
            if self.llm and LANGCHAIN_AVAILABLE:
                # Create message for LLM
                messages = [HumanMessage(content=full_prompt)]
                
                # Get LLM response
                response = await self.llm.ainvoke(messages)
                
                return response.content
            else:
                # Fallback: return formatted content without LLM processing
                if not LANGCHAIN_AVAILABLE:
                    self.logger.warning("LangChain not available, using basic formatting")
                formatted_content = f"# {section_name.title()}\n\n{content}"
                return formatted_content
            
        except Exception as e:
            self.logger.error(f"Error applying prompt to section {section_name}: {str(e)}")
            return content  # Return original content if processing fails
    
    def _get_processing_prompt(self, section_name: str) -> str:
        """Get appropriate processing prompt based on section type."""
        # Map section types to appropriate prompts from config
        section_prompt_mapping = {
            'abstract': PROMPTS.get('ABSTRACT_PROCESSING', ''),
            'introduction': PROMPTS.get('INTRODUCTION_PROCESSING', ''),
            'methodology': PROMPTS.get('METHODOLOGY_PROCESSING', ''),
            'results': PROMPTS.get('RESULTS_PROCESSING', ''),
            'conclusion': PROMPTS.get('CONCLUSION_PROCESSING', ''),
            'references': PROMPTS.get('REFERENCES_PROCESSING', '')
        }
        
        # Default to general processing prompt
        return section_prompt_mapping.get(
            section_name.lower(), 
            PROMPTS.get('GENERAL_SECTION_PROCESSING', 'Process this section maintaining academic rigor and clarity.')
        )

class PromptOrchestrator:
    """Orchestrates the application of different prompts based on processing stage."""
    
    def __init__(self):
        self.prompt_categories = {
            'chunking': ['CHUNKING_STRATEGY', 'SEMANTIC_CHUNKING'],
            'mapping': ['SEMANTIC_MAPPING', 'SECTION_MAPPING'],
            'validation': ['OUTPUT_VALIDATION', 'CONSISTENCY_CHECK'],
            'formatting': ['LATEX_FORMATTING', 'MARKDOWN_FORMATTING'],
            'combination': ['DOCUMENT_COMBINATION', 'SECTION_INTEGRATION']
        }
    
    def get_prompts_for_stage(self, stage: str) -> List[str]:
        """Get appropriate prompts for a processing stage."""
        return [PROMPTS.get(prompt_name, '') for prompt_name in self.prompt_categories.get(stage, [])]
    
    def apply_validation_prompts(self, content: str, section_type: str) -> str:
        """Apply validation prompts to check content quality."""
        validation_prompts = self.get_prompts_for_stage('validation')
        # Implementation for applying validation prompts
        return content
    
    def apply_formatting_prompts(self, content: str, output_format: str) -> str:
        """Apply formatting prompts based on desired output format."""
        formatting_prompts = self.get_prompts_for_stage('formatting')
        # Implementation for applying formatting prompts
        return content
    
    def get_validation_prompts(self, section_type: str) -> List[str]:
        """Get validation prompts for a specific section type."""
        validation_prompts = {
            'introduction': [
                "Verify the introduction clearly states the problem and objectives",
                "Check that background information is accurate and relevant",
                "Ensure the introduction flows logically to the main content"
            ],
            'methodology': [
                "Validate that the methodology is clearly described",
                "Check for completeness of technical details",
                "Ensure reproducibility of the described methods"
            ],
            'results': [
                "Verify that results are clearly presented",
                "Check for proper data analysis and interpretation",
                "Ensure results support the stated conclusions"
            ],
            'conclusion': [
                "Validate that conclusions follow from the presented evidence",
                "Check for discussion of limitations and future work",
                "Ensure the conclusion ties back to the original objectives"
            ]
        }
        return validation_prompts.get(section_type, ["Validate content quality and accuracy"])