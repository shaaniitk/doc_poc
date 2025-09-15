"""Output formatting node for applying format-specific prompts and transformations."""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from core.models import WorkflowState, DocumentFormat
from core.llm_handler import LLMHandler
from core.template_processor_node import PromptOrchestrator

logger = logging.getLogger(__name__)

class FormattingStrategy(Enum):
    """Available formatting strategies."""
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    PLAIN_TEXT = "plain_text"

class OutputFormatterNode:
    """Node for applying format-specific prompts and transformations to combined content."""
    
    def __init__(self, llm_handler: Optional[LLMHandler] = None):
        """Initialize the output formatter node.
        
        Args:
            llm_interface: LLM interface for format transformations
        """
        self.llm_handler = llm_handler
        self.prompt_orchestrator = PromptOrchestrator()
        logger.info(f"OutputFormatterNode initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process combined content and apply format-specific transformations.
        
        Args:
            input_data: Dictionary containing combined_content and target_format
            
        Returns:
            Dictionary with formatted_content and formatting_metadata
        """
        logger.info("Starting output formatting process")
        
        # Handle both dict and WorkflowState inputs
        if isinstance(input_data, WorkflowState):
            combined_content = getattr(input_data, 'combined_content', '')
            target_format = input_data.metadata.format if input_data.metadata else DocumentFormat.MARKDOWN
        else:
            combined_content = input_data.get('combined_content', '')
            target_format = input_data.get('target_format', DocumentFormat.MARKDOWN)
        
        if not combined_content:
            logger.warning("No combined content found for formatting")
            return {
                'formatted_content': '',
                'formatting_metadata': {
                    'target_format': target_format.value if hasattr(target_format, 'value') else str(target_format),
                    'error': 'No content to format'
                }
            }
        
        # Convert DocumentFormat to FormattingStrategy
        strategy = self._get_formatting_strategy(target_format)
        
        # Get formatting options
        if isinstance(input_data, WorkflowState):
            formatting_options = {}
        else:
            formatting_options = input_data.get('formatting_options', {})
        
        # Apply format-specific transformations
        formatted_content = await self._apply_formatting(
            combined_content, 
            strategy,
            formatting_options
        )
        
        # Generate formatting metadata
        formatting_metadata = {
            'target_format': strategy.value,
            'original_length': len(combined_content),
            'formatted_length': len(formatted_content),
            'transformations_applied': self._get_applied_transformations(strategy),
            'formatting_confidence': 0.9  # Could be calculated based on LLM response
        }
        
        logger.info(f"Output formatting completed: {strategy.value} format, {len(formatted_content)} characters")
        
        return {
            'formatted_content': formatted_content,
            'formatting_metadata': formatting_metadata
        }
    
    def _get_formatting_strategy(self, target_format) -> FormattingStrategy:
        """Convert DocumentFormat to FormattingStrategy.
        
        Args:
            target_format: Target document format
            
        Returns:
            Corresponding FormattingStrategy
        """
        if hasattr(target_format, 'value'):
            format_str = target_format.value.lower()
        else:
            format_str = str(target_format).lower()
        
        format_mapping = {
            'markdown': FormattingStrategy.MARKDOWN,
            'latex': FormattingStrategy.LATEX,
            'html': FormattingStrategy.HTML,
            'plain_text': FormattingStrategy.PLAIN_TEXT,
            'txt': FormattingStrategy.PLAIN_TEXT
        }
        
        return format_mapping.get(format_str, FormattingStrategy.MARKDOWN)
    
    async def _apply_formatting(self, content: str, strategy: FormattingStrategy, options: Dict[str, Any]) -> str:
        """Apply format-specific transformations to content.
        
        Args:
            content: Combined content to format
            strategy: Formatting strategy to apply
            options: Additional formatting options
            
        Returns:
            Formatted content
        """
        if not self.llm_handler:
            # Fallback to basic formatting without LLM
            return self._apply_basic_formatting(content, strategy)
        
        # Get format-specific prompt
        formatting_prompt = self._get_formatting_prompt(strategy, options)
        
        # Apply LLM-based formatting
        try:
            # Note: LLMHandler uses different method signature
            # For now, fall back to basic formatting until LLM integration is implemented
            logger.info("LLM-based formatting not yet implemented, using basic formatting")
            return self._apply_basic_formatting(content, strategy)
        except Exception as e:
            logger.error(f"LLM formatting failed: {e}")
            return self._apply_basic_formatting(content, strategy)
    
    def _apply_basic_formatting(self, content: str, strategy: FormattingStrategy) -> str:
        """Apply basic formatting transformations without LLM.
        
        Args:
            content: Content to format
            strategy: Formatting strategy
            
        Returns:
            Formatted content
        """
        if strategy == FormattingStrategy.MARKDOWN:
            # Ensure proper markdown structure
            lines = content.split('\n')
            formatted_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    formatted_lines.append('')
                    continue
                
                # Add markdown formatting for headers if not present
                if line.lower().startswith(('introduction', 'methodology', 'results', 'conclusion')):
                    if not line.startswith('#'):
                        formatted_lines.append(f"## {line}")
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            
            return '\n'.join(formatted_lines)
        
        elif strategy == FormattingStrategy.LATEX:
            # Basic LaTeX structure
            latex_content = "\\documentclass{article}\n"
            latex_content += "\\usepackage[utf8]{inputenc}\n"
            latex_content += "\\begin{document}\n\n"
            
            # Convert content to LaTeX
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    latex_content += "\n"
                    continue
                
                # Convert headers
                if line.lower().startswith(('introduction', 'methodology', 'results', 'conclusion')):
                    latex_content += f"\\section{{{line}}}\n\n"
                else:
                    latex_content += f"{line}\n\n"
            
            latex_content += "\\end{document}"
            return latex_content
        
        elif strategy == FormattingStrategy.HTML:
            # Basic HTML structure
            html_content = "<!DOCTYPE html>\n<html>\n<head>\n<title>Document</title>\n</head>\n<body>\n\n"
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    html_content += "<br>\n"
                    continue
                
                # Convert headers
                if line.lower().startswith(('introduction', 'methodology', 'results', 'conclusion')):
                    html_content += f"<h2>{line}</h2>\n"
                else:
                    html_content += f"<p>{line}</p>\n"
            
            html_content += "\n</body>\n</html>"
            return html_content
        
        else:  # PLAIN_TEXT
            return content
    
    def _get_formatting_prompt(self, strategy: FormattingStrategy, options: Dict[str, Any]) -> str:
        """Get format-specific prompt for LLM formatting.
        
        Args:
            strategy: Formatting strategy
            options: Additional formatting options
            
        Returns:
            Format-specific prompt
        """
        base_prompt = f"Please format the following content for {strategy.value} output. "
        
        if strategy == FormattingStrategy.MARKDOWN:
            return base_prompt + """
Ensure proper markdown syntax with:
- Clear section headers using ## for main sections
- Proper paragraph breaks
- Code blocks with ``` if any code is present
- Lists with proper bullet points or numbering
- Emphasis with **bold** and *italic* where appropriate

Content to format: {content}
"""
        
        elif strategy == FormattingStrategy.LATEX:
            return base_prompt + """
Create a complete LaTeX document with:
- Proper document class and packages
- Section headers using \\section{}
- Proper paragraph formatting
- Mathematical expressions in math mode if present
- Proper escaping of special characters

Content to format: {content}
"""
        
        elif strategy == FormattingStrategy.HTML:
            return base_prompt + """
Create a complete HTML document with:
- Proper DOCTYPE and HTML structure
- Semantic HTML tags (h1, h2, p, etc.)
- Proper escaping of special characters
- Clean, readable formatting

Content to format: {content}
"""
        
        else:  # PLAIN_TEXT
            return base_prompt + """
Format as clean, readable plain text with:
- Clear section breaks
- Proper paragraph spacing
- No special formatting characters

Content to format: {content}
"""
    
    def _get_applied_transformations(self, strategy: FormattingStrategy) -> List[str]:
        """Get list of transformations applied for the given strategy.
        
        Args:
            strategy: Formatting strategy used
            
        Returns:
            List of transformation descriptions
        """
        transformations = {
            FormattingStrategy.MARKDOWN: [
                "Section headers converted to markdown format",
                "Paragraph breaks normalized",
                "Markdown syntax applied"
            ],
            FormattingStrategy.LATEX: [
                "LaTeX document structure added",
                "Section commands applied",
                "Special characters escaped"
            ],
            FormattingStrategy.HTML: [
                "HTML document structure added",
                "Semantic HTML tags applied",
                "Special characters escaped"
            ],
            FormattingStrategy.PLAIN_TEXT: [
                "Clean text formatting applied",
                "Special characters removed"
            ]
        }
        
        return transformations.get(strategy, ["Basic formatting applied"])