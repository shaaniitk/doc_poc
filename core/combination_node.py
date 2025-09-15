import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .models import WorkflowState, ProcessedSection
from modules.document_combiner import HierarchicalDocumentCombiner
from .validation_node import ValidationResult

logger = logging.getLogger(__name__)

class CombinationStrategy(Enum):
    """Different strategies for combining document sections."""
    WEAVE = "weave"  # Default weaving strategy
    CONTRAST = "contrast"  # Contrast-based combination
    HIERARCHICAL = "hierarchical"  # Hierarchical merging

@dataclass
class CombinationResult:
    """Result of document combination operation."""
    combined_content: str
    sections_combined: int
    strategy_used: str
    confidence_score: float  # 0.0 to 1.0
    metadata: Dict[str, Any]

class DocumentCombinationNode:
    """Node responsible for combining processed document sections into a coherent document."""
    
    def __init__(self, strategy: CombinationStrategy = CombinationStrategy.WEAVE):
        self.strategy = strategy
        self.combiner = HierarchicalDocumentCombiner()
        logger.info(f"DocumentCombinationNode initialized with strategy: {strategy.value}")
    
    async def combine_sections(self, state, validation_results: Optional[Dict[str, ValidationResult]] = None) -> CombinationResult:
        """Combine processed sections into a final document."""
        logger.info("Starting document combination process")
        
        # Handle both dict and WorkflowState inputs
        if isinstance(state, dict):
            processed_sections = state.get('processed_sections', [])
        else:
            processed_sections = getattr(state, 'processed_sections', [])
        
        if not processed_sections:
            logger.warning("No processed sections found for combination")
            return CombinationResult(
                combined_content="",
                sections_combined=0,
                strategy_used=self.strategy.value,
                confidence_score=0.0,
                metadata={"error": "No sections to combine"}
            )
        
        # Convert processed sections to hierarchical document tree
        base_doc_tree = self._convert_to_document_tree(processed_sections)
        
        # If we have validation results, filter or prioritize sections
        if validation_results:
            base_doc_tree = self._apply_validation_filtering(base_doc_tree, validation_results)
        
        # For now, we'll use the base document as is since we don't have multiple documents to combine
        # In a real scenario, you might have multiple document versions or sources
        combined_tree = base_doc_tree
        
        # Convert back to combined content
        combined_content = self._tree_to_content(combined_tree)
        
        # Calculate confidence based on validation results
        confidence = self._calculate_combination_confidence(validation_results) if validation_results else 0.8
        
        result = CombinationResult(
            combined_content=combined_content,
            sections_combined=len(processed_sections),
            strategy_used=self.strategy.value,
            confidence_score=confidence,
            metadata={
                "sections_processed": len(processed_sections),
                "validation_applied": validation_results is not None,
                "tree_structure": list(combined_tree.keys())
            }
        )
        
        logger.info(f"Document combination completed: {len(processed_sections)} sections combined")
        return result
    
    def _convert_to_document_tree(self, processed_sections: List[Any]) -> Dict[str, Any]:
        """Convert processed sections to hierarchical document tree format."""
        tree = {}
        
        for section in processed_sections:
            if hasattr(section, 'section_name') and hasattr(section, 'content'):
                section_name = section.section_name
                tree[section_name] = {
                    'processed_content': section.content,
                    'confidence': getattr(section, 'confidence_score', 0.0),
                    'source_chunks': len(getattr(section, 'source_chunks', [])),
                    'subsections': {}  # Could be expanded for hierarchical sections
                }
            elif isinstance(section, dict):
                section_name = section.get('section_name', f'section_{len(tree)}')
                tree[section_name] = {
                    'processed_content': section.get('content', ''),
                    'confidence': section.get('confidence', 0.0),
                    'source_chunks': section.get('source_chunks', 0),
                    'subsections': {}
                }
        
        return tree
    
    def _apply_validation_filtering(self, doc_tree: Dict[str, Any], 
                                  validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Apply validation results to filter or modify sections."""
        filtered_tree = {}
        
        for section_name, section_data in doc_tree.items():
            validation_result = validation_results.get(section_name)
            
            if validation_result:
                # Only include sections that passed validation or have high confidence
                if validation_result.is_valid or validation_result.confidence_score > 0.5:
                    # Use validated content if available
                    if validation_result.validated_content:
                        section_data = section_data.copy()
                        section_data['processed_content'] = validation_result.validated_content
                    filtered_tree[section_name] = section_data
                else:
                    logger.warning(f"Excluding section '{section_name}' due to low validation score")
            else:
                # Include sections without validation results
                filtered_tree[section_name] = section_data
        
        return filtered_tree
    
    def _tree_to_content(self, doc_tree: Dict[str, Any]) -> str:
        """Convert document tree back to combined content string."""
        content_parts = []
        
        for section_name, section_data in doc_tree.items():
            # Add section header
            content_parts.append(f"\n## {section_name.title()}\n")
            
            # Add section content
            section_content = section_data.get('processed_content', '')
            if section_content:
                content_parts.append(section_content)
            
            # Add subsections if any
            subsections = section_data.get('subsections', {})
            if subsections:
                for subsection_name, subsection_data in subsections.items():
                    content_parts.append(f"\n### {subsection_name.title()}\n")
                    subsection_content = subsection_data.get('processed_content', '')
                    if subsection_content:
                        content_parts.append(subsection_content)
        
        return "\n".join(content_parts)
    
    def _calculate_combination_confidence(self, validation_results: Dict[str, ValidationResult]) -> float:
        """Calculate overall confidence score based on validation results."""
        if not validation_results:
            return 0.8  # Default confidence
        
        valid_sections = sum(1 for result in validation_results.values() if result.is_valid)
        total_sections = len(validation_results)
        
        if total_sections == 0:
            return 0.8
        
        # Base confidence on validation success rate
        validation_rate = valid_sections / total_sections
        
        # Average confidence scores
        avg_confidence = sum(result.confidence_score for result in validation_results.values()) / total_sections
        
        # Combine both metrics
        overall_confidence = (validation_rate * 0.6) + (avg_confidence * 0.4)
        
        return min(overall_confidence, 1.0)
    
    async def combine_multiple_documents(self, base_state, augmentation_state, 
                                       strategy: Optional[CombinationStrategy] = None) -> CombinationResult:
        """Combine two different document states using the hierarchical combiner."""
        combination_strategy = strategy or self.strategy
        
        # Convert both states to document trees
        base_tree = self._convert_to_document_tree(
            base_state.get('processed_sections', []) if isinstance(base_state, dict) 
            else getattr(base_state, 'processed_sections', [])
        )
        
        aug_tree = self._convert_to_document_tree(
            augmentation_state.get('processed_sections', []) if isinstance(augmentation_state, dict)
            else getattr(augmentation_state, 'processed_sections', [])
        )
        
        # Use the hierarchical combiner
        combined_tree = self.combiner.combine_documents(base_tree, aug_tree, combination_strategy.value)
        
        # Convert to final content
        combined_content = self._tree_to_content(combined_tree)
        
        return CombinationResult(
            combined_content=combined_content,
            sections_combined=len(base_tree) + len(aug_tree),
            strategy_used=combination_strategy.value,
            confidence_score=0.85,  # Higher confidence for multi-document combination
            metadata={
                "base_sections": len(base_tree),
                "augmentation_sections": len(aug_tree),
                "combined_sections": len(combined_tree)
            }
        )