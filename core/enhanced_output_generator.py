"""Enhanced Output Generator with Jinja Templating and LaTeX Support

This module provides sophisticated output generation capabilities with LangGraph workflow
orchestration, Jinja2 templating engine, LaTeX document generation, multi-format support,
and advanced formatting strategies for professional document output.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import tempfile
import subprocess
from collections import defaultdict

# Template engine imports
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from jinja2.exceptions import TemplateError, TemplateNotFound

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Internal imports
from .chunking_processor import DocumentChunk, ChunkMetadata
from .enhanced_aggregation_engine import AggregatedContent, AggregationGroup
from .langgraph_orchestrator import BaseWorkflowNode, NodeConfig, WorkflowState, NodeResult
from .state_manager import CentralizedStateManager, ProcessingStage, ErrorSeverity
from .llm_handler import LLMProvider, LLMConfig, ProcessingResult

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""
    LATEX = "latex"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    DOCX = "docx"
    TXT = "txt"
    JSON = "json"
    XML = "xml"


class TemplateType(Enum):
    """Types of document templates."""
    ACADEMIC_PAPER = "academic_paper"
    TECHNICAL_REPORT = "technical_report"
    RESEARCH_SUMMARY = "research_summary"
    DOCUMENTATION = "documentation"
    PRESENTATION = "presentation"
    BOOK_CHAPTER = "book_chapter"
    ARTICLE = "article"
    CUSTOM = "custom"


class FormattingStrategy(Enum):
    """Document formatting strategies."""
    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    HIERARCHICAL = "hierarchical"
    MODULAR = "modular"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    CREATIVE = "creative"


@dataclass
class OutputConfiguration:
    """Configuration for output generation."""
    format: OutputFormat = OutputFormat.LATEX
    template_type: TemplateType = TemplateType.TECHNICAL_REPORT
    formatting_strategy: FormattingStrategy = FormattingStrategy.STRUCTURED
    
    # Template settings
    template_path: Optional[str] = None
    custom_template: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)
    
    # LaTeX specific settings
    latex_engine: str = "pdflatex"  # pdflatex, xelatex, lualatex
    latex_packages: List[str] = field(default_factory=lambda: [
        "geometry", "amsmath", "amsfonts", "amssymb", "graphicx", 
        "hyperref", "booktabs", "longtable", "fancyhdr", "titlesec"
    ])
    bibliography_style: str = "plain"
    
    # Document metadata
    title: str = "Generated Document"
    author: str = "Document Generator"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Formatting options
    include_toc: bool = True
    include_bibliography: bool = False
    include_appendix: bool = False
    page_numbering: bool = True
    two_column: bool = False
    font_size: str = "11pt"
    paper_size: str = "a4paper"
    
    # Content organization
    max_section_depth: int = 4
    auto_section_numbering: bool = True
    include_source_references: bool = True
    include_quality_metrics: bool = False
    
    # Output settings
    output_directory: str = "./output"
    filename_prefix: str = "generated_doc"
    include_timestamp: bool = True
    cleanup_temp_files: bool = True


@dataclass
class GeneratedDocument:
    """Represents a generated document with metadata."""
    content: str
    format: OutputFormat
    template_type: TemplateType
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_stats: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "format": self.format.value,
            "template_type": self.template_type.value,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "generation_stats": self.generation_stats,
            "created_at": self.created_at.isoformat()
        }


class TemplateManagerNode(BaseWorkflowNode):
    """Node for managing document templates."""
    
    def __init__(self, config: NodeConfig, state_manager: CentralizedStateManager):
        super().__init__(config, state_manager)
        self.template_cache = {}
        self.jinja_env = None
        self._setup_jinja_environment()
    
    def _setup_jinja_environment(self):
        """Setup Jinja2 environment with custom filters and functions."""
        # Setup template directories
        template_dirs = [
            "./templates",
            "./core/templates",
            os.path.join(os.path.dirname(__file__), "templates")
        ]
        
        # Create directories if they don't exist
        for template_dir in template_dirs:
            os.makedirs(template_dir, exist_ok=True)
        
        # Initialize Jinja environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dirs),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.jinja_env.filters['latex_escape'] = self._latex_escape
        self.jinja_env.filters['format_date'] = self._format_date
        self.jinja_env.filters['word_count'] = self._word_count
        self.jinja_env.filters['truncate_smart'] = self._truncate_smart
        
        # Add custom functions
        self.jinja_env.globals['now'] = datetime.now
        self.jinja_env.globals['generate_id'] = self._generate_id
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Load and prepare document template."""
        output_config = state.get("output_config", {})
        organized_structure = state.get("organized_structure", {})
        
        # Load template
        template = await self._load_template(output_config)
        
        # Prepare template variables
        template_vars = await self._prepare_template_variables(organized_structure, output_config)
        
        # Update state
        state["document_template"] = template
        state["template_variables"] = template_vars
        state["template_stats"] = {
            "template_type": output_config.get("template_type", "technical_report"),
            "variables_count": len(template_vars),
            "template_loaded": template is not None
        }
        
        self.logger.info(f"Template loaded: {output_config.get('template_type', 'default')}")
        
        return state
    
    async def _load_template(self, config: Dict[str, Any]) -> Optional[Template]:
        """Load document template based on configuration."""
        template_type = config.get("template_type", "technical_report")
        custom_template = config.get("custom_template")
        template_path = config.get("template_path")
        
        try:
            # Use custom template if provided
            if custom_template:
                return self.jinja_env.from_string(custom_template)
            
            # Load from file if path provided
            if template_path and os.path.exists(template_path):
                return self.jinja_env.get_template(os.path.basename(template_path))
            
            # Load built-in template
            template_filename = f"{template_type}.tex"
            
            # Check cache first
            if template_filename in self.template_cache:
                return self.template_cache[template_filename]
            
            # Try to load from template directories
            try:
                template = self.jinja_env.get_template(template_filename)
                self.template_cache[template_filename] = template
                return template
            except TemplateNotFound:
                # Create default template if not found
                default_template = await self._create_default_template(template_type)
                self.template_cache[template_filename] = default_template
                return default_template
                
        except Exception as e:
            self.logger.error(f"Error loading template: {e}")
            return await self._create_default_template("basic")
    
    async def _create_default_template(self, template_type: str) -> Template:
        """Create default template for specified type."""
        if template_type == "academic_paper":
            template_content = self._get_academic_paper_template()
        elif template_type == "technical_report":
            template_content = self._get_technical_report_template()
        else:
            template_content = self._get_basic_template()
        
        return self.jinja_env.from_string(template_content)
    
    def _get_technical_report_template(self) -> str:
        """Get technical report LaTeX template."""
        return r"""
\documentclass[{{ font_size }},{{ paper_size }}]{article}

% Packages
{% for package in latex_packages %}
\usepackage{ {{- package -}} }
{% endfor %}

% Document metadata
\title{ {{- title | latex_escape -}} }
\author{ {{- author | latex_escape -}} }
\date{ {{- date -}} }

% Page setup
\geometry{margin=1in}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{ {{- title | latex_escape -}} }
\fancyhead[R]{\thepage}

\begin{document}

\maketitle

{% if abstract %}
\begin{abstract}
{{ abstract | latex_escape }}
\end{abstract}
{% endif %}

{% if include_toc %}
\tableofcontents
\newpage
{% endif %}

{% for section in sections %}
\section{ {{- section.title | latex_escape -}} }
{% if section.content %}
{{ section.content | latex_escape }}
{% endif %}

{% if section.subsections %}
{% for subsection in section.subsections %}
\subsection{ {{- subsection.title | latex_escape -}} }
{{ subsection.content | latex_escape }}
{% endfor %}
{% endif %}

{% endfor %}

{% if include_source_references and source_references %}
\section{Source References}
\begin{itemize}
{% for ref in source_references %}
\item {{ ref | latex_escape }}
{% endfor %}
\end{itemize}
{% endif %}

\end{document}
"""
    
    def _get_academic_paper_template(self) -> str:
        """Get academic paper LaTeX template."""
        return r"""
\documentclass[{{ font_size }},{{ paper_size }}{% if two_column %},twocolumn{% endif %}]{article}

% Essential packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{booktabs}
\usepackage{fancyhdr}

% Document setup
\geometry{margin=1in}
\title{ {{- title | latex_escape -}} }
\author{ {{- author | latex_escape -}} }
\date{ {{- date -}} }

\begin{document}

\maketitle

{% if abstract %}
\begin{abstract}
{{ abstract | latex_escape }}
{% if keywords %}
\\[1em]
\textbf{Keywords:} {{ keywords | join(', ') | latex_escape }}
{% endif %}
\end{abstract}
{% endif %}

{% if include_toc %}
\tableofcontents
\newpage
{% endif %}

{% for section in sections %}
\section{ {{- section.title | latex_escape -}} }
{{ section.content | latex_escape }}

{% for subsection in section.get('subsections', []) %}
\subsection{ {{- subsection.title | latex_escape -}} }
{{ subsection.content | latex_escape }}
{% endfor %}

{% endfor %}

{% if include_bibliography %}
\bibliographystyle{ {{- bibliography_style -}} }
\bibliography{references}
{% endif %}

\end{document}
"""
    
    def _get_basic_template(self) -> str:
        """Get basic LaTeX template."""
        return r"""
\documentclass[{{ font_size }},{{ paper_size }}]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{hyperref}

\geometry{margin=1in}
\title{ {{- title | latex_escape -}} }
\author{ {{- author | latex_escape -}} }
\date{ {{- date -}} }

\begin{document}

\maketitle

{% for section in sections %}
\section{ {{- section.title | latex_escape -}} }
{{ section.content | latex_escape }}
{% endfor %}

\end{document}
"""
    
    async def _prepare_template_variables(self, structure: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variables for template rendering."""
        variables = {
            # Document metadata
            "title": config.get("title", "Generated Document"),
            "author": config.get("author", "Document Generator"),
            "date": config.get("date", datetime.now().strftime("%Y-%m-%d")),
            "abstract": config.get("abstract", ""),
            "keywords": config.get("keywords", []),
            
            # LaTeX settings
            "font_size": config.get("font_size", "11pt"),
            "paper_size": config.get("paper_size", "a4paper"),
            "latex_packages": config.get("latex_packages", []),
            "bibliography_style": config.get("bibliography_style", "plain"),
            
            # Document structure
            "sections": structure.get("sections", []),
            "include_toc": config.get("include_toc", True),
            "include_bibliography": config.get("include_bibliography", False),
            "include_source_references": config.get("include_source_references", True),
            "two_column": config.get("two_column", False),
            
            # Generation metadata
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_sections": len(structure.get("sections", [])),
            
            # Custom variables
            **config.get("template_variables", {})
        }
        
        # Add source references if requested
        if config.get("include_source_references", True):
            source_refs = []
            for section in structure.get("sections", []):
                if "source_chunks" in section:
                    for chunk_id in section["source_chunks"]:
                        source_refs.append(f"Chunk ID: {chunk_id}")
            variables["source_references"] = list(set(source_refs))
        
        return variables
    
    def _latex_escape(self, text: str) -> str:
        """Escape special LaTeX characters."""
        if not isinstance(text, str):
            text = str(text)
        
        # LaTeX special characters
        replacements = {
            '\\': r'\textbackslash{}',
            '{': r'\{',
            '}': r'\}',
            '$': r'\$',
            '&': r'\&',
            '%': r'\%',
            '#': r'\#',
            '^': r'\textasciicircum{}',
            '_': r'\_',
            '~': r'\textasciitilde{}'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def _format_date(self, date_obj: datetime, format_str: str = "%Y-%m-%d") -> str:
        """Format datetime object."""
        if isinstance(date_obj, str):
            return date_obj
        return date_obj.strftime(format_str)
    
    def _word_count(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _truncate_smart(self, text: str, length: int = 100) -> str:
        """Smart truncation that preserves word boundaries."""
        if len(text) <= length:
            return text
        
        truncated = text[:length]
        last_space = truncated.rfind(' ')
        
        if last_space > length * 0.8:  # If we can find a space near the end
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."
    
    def _generate_id(self, prefix: str = "id") -> str:
        """Generate unique ID."""
        return f"{prefix}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"


class DocumentRendererNode(BaseWorkflowNode):
    """Node for rendering documents using templates."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Render document using template and variables."""
        template = state.get("document_template")
        template_vars = state.get("template_variables", {})
        output_config = state.get("output_config", {})
        
        if not template:
            raise ValueError("No template available for rendering")
        
        # Render document
        rendered_content = await self._render_document(template, template_vars)
        
        # Post-process content
        processed_content = await self._post_process_content(rendered_content, output_config)
        
        # Update state
        state["rendered_content"] = processed_content
        state["rendering_stats"] = {
            "content_length": len(processed_content),
            "template_variables_used": len(template_vars),
            "rendering_successful": True
        }
        
        self.logger.info(f"Document rendered successfully, length: {len(processed_content)} characters")
        
        return state
    
    async def _render_document(self, template: Template, variables: Dict[str, Any]) -> str:
        """Render document using Jinja2 template."""
        try:
            return template.render(**variables)
        except TemplateError as e:
            self.logger.error(f"Template rendering error: {e}")
            raise
    
    async def _post_process_content(self, content: str, config: Dict[str, Any]) -> str:
        """Post-process rendered content."""
        # Clean up extra whitespace
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            cleaned_lines.append(line)
        
        # Remove excessive blank lines
        final_lines = []
        blank_count = 0
        
        for line in cleaned_lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:  # Allow max 2 consecutive blank lines
                    final_lines.append(line)
            else:
                blank_count = 0
                final_lines.append(line)
        
        return '\n'.join(final_lines)


class OutputWriterNode(BaseWorkflowNode):
    """Node for writing rendered content to files."""
    
    async def _execute_impl(self, state: WorkflowState) -> WorkflowState:
        """Write rendered content to output files."""
        rendered_content = state.get("rendered_content", "")
        output_config = state.get("output_config", {})
        
        # Generate output files
        output_files = await self._write_output_files(rendered_content, output_config)
        
        # Update state
        state["output_files"] = output_files
        state["output_stats"] = {
            "files_generated": len(output_files),
            "formats": [file_info["format"] for file_info in output_files],
            "total_size": sum(file_info.get("size", 0) for file_info in output_files)
        }
        
        self.logger.info(f"Generated {len(output_files)} output files")
        
        return state
    
    async def _write_output_files(self, content: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write content to various output formats."""
        output_files = []
        output_dir = config.get("output_directory", "./output")
        filename_prefix = config.get("filename_prefix", "generated_doc")
        include_timestamp = config.get("include_timestamp", True)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp suffix
        timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S") if include_timestamp else ""
        
        # Write LaTeX file
        latex_filename = f"{filename_prefix}{timestamp}.tex"
        latex_path = os.path.join(output_dir, latex_filename)
        
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        output_files.append({
            "format": "latex",
            "filename": latex_filename,
            "path": latex_path,
            "size": len(content.encode('utf-8'))
        })
        
        # Generate PDF if requested
        output_format = config.get("format", "latex")
        if output_format == "pdf" or config.get("generate_pdf", False):
            pdf_path = await self._compile_latex_to_pdf(latex_path, config)
            if pdf_path:
                output_files.append({
                    "format": "pdf",
                    "filename": os.path.basename(pdf_path),
                    "path": pdf_path,
                    "size": os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
                })
        
        # Generate other formats if requested
        additional_formats = config.get("additional_formats", [])
        for fmt in additional_formats:
            if fmt == "markdown":
                md_content = await self._convert_to_markdown(content)
                md_filename = f"{filename_prefix}{timestamp}.md"
                md_path = os.path.join(output_dir, md_filename)
                
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                output_files.append({
                    "format": "markdown",
                    "filename": md_filename,
                    "path": md_path,
                    "size": len(md_content.encode('utf-8'))
                })
        
        return output_files
    
    async def _compile_latex_to_pdf(self, latex_path: str, config: Dict[str, Any]) -> Optional[str]:
        """Compile LaTeX file to PDF."""
        try:
            latex_engine = config.get("latex_engine", "pdflatex")
            output_dir = os.path.dirname(latex_path)
            
            # Run LaTeX compilation
            cmd = [
                latex_engine,
                "-interaction=nonstopmode",
                f"-output-directory={output_dir}",
                latex_path
            ]
            
            # Run compilation (may need multiple passes)
            for i in range(2):  # Run twice for cross-references
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=output_dir
                )
                
                if result.returncode != 0:
                    self.logger.warning(f"LaTeX compilation warning (pass {i+1}): {result.stderr}")
            
            # Check if PDF was generated
            pdf_path = latex_path.replace('.tex', '.pdf')
            if os.path.exists(pdf_path):
                return pdf_path
            else:
                self.logger.error("PDF compilation failed - no output file generated")
                return None
                
        except Exception as e:
            self.logger.error(f"Error compiling LaTeX to PDF: {e}")
            return None
    
    async def _convert_to_markdown(self, latex_content: str) -> str:
        """Convert LaTeX content to Markdown (basic conversion)."""
        # Basic LaTeX to Markdown conversion
        md_content = latex_content
        
        # Convert sections
        md_content = md_content.replace(r'\section{', '\n# ')
        md_content = md_content.replace(r'\subsection{', '\n## ')
        md_content = md_content.replace(r'\subsubsection{', '\n### ')
        
        # Remove LaTeX commands
        import re
        md_content = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', md_content)
        md_content = re.sub(r'\\[a-zA-Z]+', '', md_content)
        
        # Clean up
        md_content = re.sub(r'\n\s*\n\s*\n', '\n\n', md_content)
        
        return md_content.strip()


class EnhancedOutputGenerator:
    """Enhanced output generator with LangGraph workflow integration."""
    
    def __init__(self, config: Optional[OutputConfiguration] = None,
                 state_manager: Optional[CentralizedStateManager] = None):
        self.config = config or OutputConfiguration()
        self.state_manager = state_manager or CentralizedStateManager()
        self.workflow_graph = None
        self.memory_saver = MemorySaver()
        
        self._build_workflow_graph()
        
        logger.info(f"EnhancedOutputGenerator initialized with format: {self.config.format.value}")
    
    def _build_workflow_graph(self) -> None:
        """Build the LangGraph workflow for output generation."""
        workflow = StateGraph(WorkflowState)
        
        # Create workflow nodes
        template_node = TemplateManagerNode(
            NodeConfig(name="template_manager", max_retries=2),
            self.state_manager
        )
        
        renderer_node = DocumentRendererNode(
            NodeConfig(name="document_renderer", max_retries=3),
            self.state_manager
        )
        
        writer_node = OutputWriterNode(
            NodeConfig(name="output_writer", max_retries=2),
            self.state_manager
        )
        
        # Add nodes to workflow
        workflow.add_node("template_manager", template_node.execute)
        workflow.add_node("document_renderer", renderer_node.execute)
        workflow.add_node("output_writer", writer_node.execute)
        
        # Define workflow edges
        workflow.add_edge("template_manager", "document_renderer")
        workflow.add_edge("document_renderer", "output_writer")
        workflow.add_edge("output_writer", END)
        
        # Set entry point
        workflow.set_entry_point("template_manager")
        
        # Compile workflow
        self.workflow_graph = workflow.compile(checkpointer=self.memory_saver)
        
        logger.info("Output generation workflow compiled successfully")
    
    async def generate_document_async(self, aggregated_content: Dict[str, Any],
                                    config: Optional[OutputConfiguration] = None) -> GeneratedDocument:
        """Generate document using enhanced workflow."""
        config = config or self.config
        
        # Prepare workflow state
        initial_state = {
            "organized_structure": aggregated_content.get("organized_structure", {}),
            "aggregated_content": aggregated_content.get("aggregated_content", []),
            "output_config": config.__dict__,
            "chunks": [],
            "semantic_relationships": [],
            "semantic_clusters": [],
            "knowledge_graph": {},
            "semantic_mappings": [],
            "llm_outputs": [],
            "final_output": {},
            "current_stage": "output_generation",
            "progress": 0.0,
            "session_id": hashlib.md5(f"output_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "errors": [],
            "retry_counts": {},
            "max_retries": {"default": 3},
            "stage_timings": {},
            "memory_usage": {},
            "token_usage": {},
            "quality_scores": {},
            "validation_results": {},
            "next_node": None,
            "should_continue": True,
            "is_cancelled": False
        }
        
        try:
            # Execute workflow
            result_state = await self.workflow_graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": initial_state["session_id"]}}
            )
            
            # Extract results
            rendered_content = result_state.get("rendered_content", "")
            output_files = result_state.get("output_files", [])
            
            # Create generated document
            document = GeneratedDocument(
                content=rendered_content,
                format=config.format,
                template_type=config.template_type,
                file_path=output_files[0]["path"] if output_files else None,
                metadata={
                    "template_stats": result_state.get("template_stats", {}),
                    "rendering_stats": result_state.get("rendering_stats", {}),
                    "output_stats": result_state.get("output_stats", {}),
                    "session_id": result_state["session_id"]
                },
                generation_stats={
                    "workflow_used": True,
                    "files_generated": len(output_files),
                    "processing_time": result_state.get("stage_timings", {})
                }
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Enhanced output generation workflow failed: {e}")
            # Create fallback document
            return GeneratedDocument(
                content=f"Error generating document: {e}",
                format=config.format,
                template_type=config.template_type,
                metadata={"error": str(e)},
                generation_stats={"workflow_used": False}
            )


# Factory functions
def create_enhanced_output_generator(format: OutputFormat = OutputFormat.LATEX,
                                   template_type: TemplateType = TemplateType.TECHNICAL_REPORT,
                                   **kwargs) -> EnhancedOutputGenerator:
    """Create enhanced output generator with specified configuration."""
    config = OutputConfiguration(
        format=format,
        template_type=template_type,
        **kwargs
    )
    return EnhancedOutputGenerator(config)


async def generate_document_async(aggregated_content: Dict[str, Any],
                                format: OutputFormat = OutputFormat.LATEX,
                                template_type: TemplateType = TemplateType.TECHNICAL_REPORT,
                                **kwargs) -> GeneratedDocument:
    """Convenience function for enhanced async document generation."""
    generator = create_enhanced_output_generator(format, template_type, **kwargs)
    return await generator.generate_document_async(aggregated_content)