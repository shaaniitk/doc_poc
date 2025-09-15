"""Workflow node wrappers for integrating processing nodes with LangGraph orchestrator."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from .workflow_base import BaseWorkflowNode, NodeConfig
from .state_manager import CentralizedStateManager
from .models import WorkflowState
from .template_processor_node import TemplateProcessorNode
from .validation_node import ValidationNode, ValidationLevel
from .combination_node import DocumentCombinationNode, CombinationStrategy
from .output_formatter_node import OutputFormatterNode
from .semantic_mapper import SemanticMapper
from .llm_handler import LLMHandler

logger = logging.getLogger(__name__)

class TemplateProcessorWorkflowNode(BaseWorkflowNode):
    """Workflow node wrapper for TemplateProcessorNode."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        super().__init__(config, state_manager)
        self.template_processor = None
        
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Execute template processing."""
        try:
            # Initialize template processor if not already done
            if not self.template_processor:
                # For now, use None LLM handler to avoid serialization issues
                # The template processor should handle None gracefully
                llm_handler = None
                
                # Create semantic mapper
                semantic_mapper = SemanticMapper()
                
                # Initialize template processor
                self.template_processor = TemplateProcessorNode(llm_handler, semantic_mapper)
            
            # Convert WorkflowState to dict for processing
            state_dict = {
                'chunks': getattr(state, 'chunks', []),
                'template_name': getattr(state, 'template_name', 'bitcoin_paper_hierarchical')
            }
            
            # Process with template processor
            result_dict = await self.template_processor.process(state_dict)
            
            # Update state with results
            state.processed_sections = result_dict.get('processed_sections', [])
            state.template_applied = result_dict.get('template_applied', '')
            
            # Add any processing errors to state
            if 'processing_errors' in result_dict:
                if not hasattr(state, 'errors'):
                    state.errors = []
                state.errors.extend(result_dict['processing_errors'])
            
            self.logger.info(f"Template processing completed: {len(state.processed_sections)} sections")
            return state
            
        except Exception as e:
            self.logger.error(f"Template processing failed: {e}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"Template processing error: {str(e)}")
            return state

class ValidationWorkflowNode(BaseWorkflowNode):
    """Workflow node wrapper for ValidationNode."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        super().__init__(config, state_manager)
        self.validation_node = None
        
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Execute validation processing."""
        try:
            self.logger.info("Starting validation processing")
            
            # Initialize validation node if not already done
            if not self.validation_node:
                # Create a simple mock validation for now to avoid LLM dependency
                from .validation_node import ValidationNode, ValidationLevel
                
                # Initialize validation node with None LLM (will use mock validation)
                self.validation_node = ValidationNode(None, ValidationLevel.STANDARD)
                self.logger.info("Validation node initialized")
            
            # Validate workflow state
            self.logger.info("Calling validate_workflow_state")
            validation_results = await self.validation_node.validate_workflow_state(state)
            self.logger.info(f"Validation results received: {len(validation_results) if validation_results else 0} results")
            
            # Update state with validation results
            state.validation_results = validation_results
            self.logger.info("State updated with validation results")
            
            # Log validation summary
            valid_sections = sum(1 for result in validation_results.values() if result.is_valid)
            total_sections = len(validation_results)
            
            self.logger.info(f"Validation completed: {valid_sections}/{total_sections} sections valid")
            
            # Add validation issues to errors if any critical failures
            critical_issues = []
            for section_name, result in validation_results.items():
                if not result.is_valid and result.confidence_score < 0.3:
                    critical_issues.append(f"Critical validation failure in {section_name}: {', '.join(result.issues)}")
            
            if critical_issues:
                if not hasattr(state, 'errors'):
                    state.errors = []
                state.errors.extend(critical_issues)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"Validation error: {str(e)}")
            return state

class CombinationWorkflowNode(BaseWorkflowNode):
    """Workflow node wrapper for DocumentCombinationNode."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        super().__init__(config, state_manager)
        self.combination_node = DocumentCombinationNode(CombinationStrategy.WEAVE)
        
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Execute document combination."""
        try:
            # Get validation results if available
            validation_results = getattr(state, 'validation_results', None)
            
            # Combine sections
            combination_result = await self.combination_node.combine_sections(state, validation_results)
            
            # Update state with combined content
            state.combined_content = combination_result.combined_content
            state.combination_metadata = {
                'sections_combined': combination_result.sections_combined,
                'strategy_used': combination_result.strategy_used,
                'confidence_score': combination_result.confidence_score,
                'metadata': combination_result.metadata
            }
            
            self.logger.info(f"Document combination completed: {combination_result.sections_combined} sections, "
                           f"confidence: {combination_result.confidence_score:.2f}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Document combination failed: {e}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"Combination error: {str(e)}")
            return state

class OutputFormatterWorkflowNode(BaseWorkflowNode):
    """Workflow node wrapper for OutputFormatterNode."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        super().__init__(config, state_manager)
        self.formatter_node = None
        
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Execute output formatting."""
        try:
            # Initialize formatter node if not already done
            if not self.formatter_node:
                # Use None LLM handler to avoid serialization issues
                llm_handler = None
                self.formatter_node = OutputFormatterNode(llm_handler)
            
            # Prepare input data for formatter
            input_data = {
                'combined_content': getattr(state, 'combined_content', ''),
                'target_format': getattr(state, 'target_format', 'markdown'),
                'formatting_options': getattr(state, 'formatting_options', {})
            }
            
            # Process with output formatter
            result = await self.formatter_node.process(input_data)
            
            # Update state with formatted content
            state.formatted_content = result.get('formatted_content', '')
            state.formatting_metadata = result.get('formatting_metadata', {})
            
            self.logger.info(f"Output formatting completed: {state.formatting_metadata.get('target_format', 'unknown')} format, "
                           f"{len(state.formatted_content)} characters")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Output formatting failed: {e}")
            if not hasattr(state, 'errors'):
                state.errors = []
            state.errors.append(f"Formatting error: {str(e)}")
            return state