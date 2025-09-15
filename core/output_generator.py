"""Enhanced Output Generator for LangGraph Integration

This module provides comprehensive output generation capabilities that integrate
seamlessly with the LangGraph orchestrator, supporting multiple formats,
templates, and advanced document generation features.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Union, AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import time
from collections import defaultdict
import tempfile
import shutil

# Template engines
try:
    from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    logging.warning("Jinja2 not available. Install jinja2 for advanced templating.")

# Document generation libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logging.warning("ReportLab not available. Install reportlab for PDF generation.")

try:
    from docx import Document
    from docx.shared import Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_PYTHON_DOCX = True
except ImportError:
    HAS_PYTHON_DOCX = False
    logging.warning("python-docx not available. Install python-docx for DOCX generation.")

# Markdown and HTML libraries
try:
    import markdown
    from markdown.extensions import codehilite, toc, tables
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False
    logging.warning("Markdown not available. Install markdown for Markdown processing.")

try:
    from weasyprint import HTML, CSS
    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False
    logging.warning("WeasyPrint not available. Install weasyprint for HTML to PDF conversion.")

# Data visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    logging.warning("Visualization libraries not available.")

from .chunking_processor import DocumentChunk, ChunkingResult
from .llm_handler import ProcessingResult, BatchProcessingResult
from .knowledge_graph_processor import KnowledgeGraph, Entity, Relationship

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""
    TEXT = auto()
    MARKDOWN = auto()
    HTML = auto()
    PDF = auto()
    DOCX = auto()
    JSON = auto()
    XML = auto()
    LATEX = auto()
    EPUB = auto()
    RTF = auto()
    CSV = auto()
    EXCEL = auto()


class TemplateType(Enum):
    """Types of output templates."""
    REPORT = auto()
    SUMMARY = auto()
    ANALYSIS = auto()
    PRESENTATION = auto()
    ARTICLE = auto()
    BOOK = auto()
    RESEARCH_PAPER = auto()
    TECHNICAL_DOC = auto()
    EXECUTIVE_SUMMARY = auto()
    CUSTOM = auto()


class ContentSection(Enum):
    """Standard content sections."""
    TITLE = auto()
    ABSTRACT = auto()
    INTRODUCTION = auto()
    METHODOLOGY = auto()
    RESULTS = auto()
    DISCUSSION = auto()
    CONCLUSION = auto()
    REFERENCES = auto()
    APPENDIX = auto()
    EXECUTIVE_SUMMARY = auto()
    TABLE_OF_CONTENTS = auto()
    GLOSSARY = auto()
    INDEX = auto()
    BIBLIOGRAPHY = auto()
    CUSTOM = auto()


class VisualizationType(Enum):
    """Types of visualizations to include."""
    KNOWLEDGE_GRAPH = auto()
    ENTITY_DISTRIBUTION = auto()
    RELATIONSHIP_NETWORK = auto()
    PROCESSING_METRICS = auto()
    CONTENT_STATISTICS = auto()
    TIMELINE = auto()
    WORD_CLOUD = auto()
    SENTIMENT_ANALYSIS = auto()
    TOPIC_MODELING = auto()
    CUSTOM = auto()


@dataclass
class OutputSection:
    """Represents a section in the output document."""
    section_type: ContentSection
    title: str
    content: str
    level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    subsections: List['OutputSection'] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    order: int = 0


@dataclass
class OutputTemplate:
    """Template configuration for output generation."""
    template_type: TemplateType
    name: str
    description: str
    sections: List[ContentSection] = field(default_factory=list)
    format_settings: Dict[str, Any] = field(default_factory=dict)
    style_settings: Dict[str, Any] = field(default_factory=dict)
    template_content: Optional[str] = None
    template_path: Optional[Path] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    filters: List[Callable] = field(default_factory=list)
    custom_functions: Dict[str, Callable] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    output_format: OutputFormat = OutputFormat.MARKDOWN
    template: Optional[OutputTemplate] = None
    include_metadata: bool = True
    include_statistics: bool = True
    include_visualizations: bool = True
    visualization_types: List[VisualizationType] = field(default_factory=list)
    custom_css: Optional[str] = None
    custom_styles: Dict[str, Any] = field(default_factory=dict)
    page_settings: Dict[str, Any] = field(default_factory=dict)
    export_settings: Dict[str, Any] = field(default_factory=dict)
    quality_settings: Dict[str, Any] = field(default_factory=dict)
    compression_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of output generation."""
    success: bool
    output_path: Optional[Path] = None
    output_content: Optional[str] = None
    output_format: Optional[OutputFormat] = None
    file_size: Optional[int] = None
    generation_time: float = 0.0
    sections_generated: int = 0
    visualizations_generated: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class OutputGenerator:
    """Enhanced output generator with multiple format support."""
    
    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()
        
        # Initialize template environment
        self._template_env = None
        self._initialize_template_engine()
        
        # Built-in templates
        self._builtin_templates = self._load_builtin_templates()
        
        # Statistics tracking
        self._stats = {
            "documents_generated": 0,
            "total_generation_time": 0.0,
            "formats_used": defaultdict(int),
            "templates_used": defaultdict(int)
        }
        
        logger.info("OutputGenerator initialized")
    
    def _initialize_template_engine(self) -> None:
        """Initialize Jinja2 template engine."""
        if HAS_JINJA2:
            # Create template environment with custom filters and functions
            self._template_env = Environment(
                loader=FileSystemLoader(searchpath="./"),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Add custom filters
            self._template_env.filters['format_date'] = self._format_date_filter
            self._template_env.filters['format_number'] = self._format_number_filter
            self._template_env.filters['truncate_text'] = self._truncate_text_filter
            self._template_env.filters['highlight_keywords'] = self._highlight_keywords_filter
            
            # Add custom functions
            self._template_env.globals['generate_toc'] = self._generate_table_of_contents
            self._template_env.globals['format_references'] = self._format_references
            self._template_env.globals['create_visualization'] = self._create_visualization_placeholder
            
            logger.debug("Jinja2 template engine initialized")
    
    def _load_builtin_templates(self) -> Dict[TemplateType, OutputTemplate]:
        """Load built-in templates."""
        templates = {}
        
        # Research Paper Template
        templates[TemplateType.RESEARCH_PAPER] = OutputTemplate(
            template_type=TemplateType.RESEARCH_PAPER,
            name="Research Paper",
            description="Academic research paper format",
            sections=[
                ContentSection.TITLE,
                ContentSection.ABSTRACT,
                ContentSection.INTRODUCTION,
                ContentSection.METHODOLOGY,
                ContentSection.RESULTS,
                ContentSection.DISCUSSION,
                ContentSection.CONCLUSION,
                ContentSection.REFERENCES
            ],
            format_settings={
                "font_family": "Times New Roman",
                "font_size": 12,
                "line_spacing": 1.5,
                "margins": {"top": 1, "bottom": 1, "left": 1, "right": 1}
            }
        )
        
        # Executive Summary Template
        templates[TemplateType.EXECUTIVE_SUMMARY] = OutputTemplate(
            template_type=TemplateType.EXECUTIVE_SUMMARY,
            name="Executive Summary",
            description="Business executive summary format",
            sections=[
                ContentSection.TITLE,
                ContentSection.EXECUTIVE_SUMMARY,
                ContentSection.RESULTS,
                ContentSection.CONCLUSION
            ],
            format_settings={
                "font_family": "Arial",
                "font_size": 11,
                "line_spacing": 1.2,
                "margins": {"top": 0.75, "bottom": 0.75, "left": 0.75, "right": 0.75}
            }
        )
        
        # Technical Documentation Template
        templates[TemplateType.TECHNICAL_DOC] = OutputTemplate(
            template_type=TemplateType.TECHNICAL_DOC,
            name="Technical Documentation",
            description="Technical documentation format",
            sections=[
                ContentSection.TITLE,
                ContentSection.TABLE_OF_CONTENTS,
                ContentSection.INTRODUCTION,
                ContentSection.METHODOLOGY,
                ContentSection.RESULTS,
                ContentSection.APPENDIX,
                ContentSection.GLOSSARY
            ],
            format_settings={
                "font_family": "Calibri",
                "font_size": 11,
                "line_spacing": 1.15,
                "code_font": "Consolas",
                "margins": {"top": 1, "bottom": 1, "left": 1, "right": 1}
            }
        )
        
        return templates
    
    async def generate_output_async(self,
                                  chunks: List[DocumentChunk],
                                  llm_results: Optional[BatchProcessingResult] = None,
                                  knowledge_graph: Optional[KnowledgeGraph] = None,
                                  output_path: Optional[Path] = None,
                                  template_override: Optional[OutputTemplate] = None) -> GenerationResult:
        """Generate output document asynchronously."""
        start_time = time.time()
        
        try:
            # Use template override or default template
            template = template_override or self.config.template or self._builtin_templates.get(
                TemplateType.REPORT, self._create_default_template()
            )
            
            # Generate content sections
            sections = await self._generate_sections_async(
                chunks, llm_results, knowledge_graph, template
            )
            
            # Generate visualizations if requested
            visualizations = []
            if self.config.include_visualizations:
                visualizations = await self._generate_visualizations_async(
                    chunks, llm_results, knowledge_graph
                )
            
            # Compile final document
            document_content = await self._compile_document_async(
                sections, visualizations, template
            )
            
            # Generate output in specified format
            result = await self._generate_format_output_async(
                document_content, output_path, template
            )
            
            # Update statistics
            generation_time = time.time() - start_time
            result.generation_time = generation_time
            result.sections_generated = len(sections)
            result.visualizations_generated = len(visualizations)
            
            self._stats["documents_generated"] += 1
            self._stats["total_generation_time"] += generation_time
            self._stats["formats_used"][self.config.output_format.name] += 1
            self._stats["templates_used"][template.name] += 1
            
            logger.info(f"Output generated successfully in {generation_time:.2f}s")
            
            return result
        
        except Exception as e:
            logger.error(f"Output generation failed: {e}")
            return GenerationResult(
                success=False,
                generation_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _generate_sections_async(self,
                                     chunks: List[DocumentChunk],
                                     llm_results: Optional[BatchProcessingResult],
                                     knowledge_graph: Optional[KnowledgeGraph],
                                     template: OutputTemplate) -> List[OutputSection]:
        """Generate content sections based on template."""
        sections = []
        
        for i, section_type in enumerate(template.sections):
            section_content = await self._generate_section_content_async(
                section_type, chunks, llm_results, knowledge_graph
            )
            
            section = OutputSection(
                section_type=section_type,
                title=self._get_section_title(section_type),
                content=section_content,
                level=1,
                order=i
            )
            
            sections.append(section)
        
        return sections
    
    async def _generate_section_content_async(self,
                                            section_type: ContentSection,
                                            chunks: List[DocumentChunk],
                                            llm_results: Optional[BatchProcessingResult],
                                            knowledge_graph: Optional[KnowledgeGraph]) -> str:
        """Generate content for a specific section type."""
        if section_type == ContentSection.TITLE:
            return self._generate_title_section(chunks)
        
        elif section_type == ContentSection.ABSTRACT:
            return await self._generate_abstract_section_async(chunks, llm_results)
        
        elif section_type == ContentSection.INTRODUCTION:
            return await self._generate_introduction_section_async(chunks, llm_results)
        
        elif section_type == ContentSection.METHODOLOGY:
            return self._generate_methodology_section()
        
        elif section_type == ContentSection.RESULTS:
            return await self._generate_results_section_async(chunks, llm_results, knowledge_graph)
        
        elif section_type == ContentSection.DISCUSSION:
            return await self._generate_discussion_section_async(chunks, llm_results, knowledge_graph)
        
        elif section_type == ContentSection.CONCLUSION:
            return await self._generate_conclusion_section_async(chunks, llm_results)
        
        elif section_type == ContentSection.REFERENCES:
            return self._generate_references_section(chunks)
        
        elif section_type == ContentSection.EXECUTIVE_SUMMARY:
            return await self._generate_executive_summary_section_async(chunks, llm_results, knowledge_graph)
        
        elif section_type == ContentSection.TABLE_OF_CONTENTS:
            return "[Table of Contents will be generated automatically]"
        
        elif section_type == ContentSection.GLOSSARY:
            return self._generate_glossary_section(knowledge_graph)
        
        elif section_type == ContentSection.APPENDIX:
            return self._generate_appendix_section(chunks, llm_results)
        
        else:
            return f"Content for {section_type.name} section"
    
    def _generate_title_section(self, chunks: List[DocumentChunk]) -> str:
        """Generate title section."""
        # Extract title from first chunk or metadata
        if chunks:
            first_chunk = chunks[0]
            if hasattr(first_chunk.metadata, 'title') and first_chunk.metadata.title:
                return first_chunk.metadata.title
            
            # Try to extract title from content
            content_lines = first_chunk.content.split('\n')
            for line in content_lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line and len(line) < 100:  # Reasonable title length
                    return line
        
        return "Document Analysis Report"
    
    async def _generate_abstract_section_async(self,
                                             chunks: List[DocumentChunk],
                                             llm_results: Optional[BatchProcessingResult]) -> str:
        """Generate abstract section."""
        if not llm_results or not llm_results.results:
            return "This document presents an analysis of the provided content."
        
        # Combine key insights from LLM results
        key_insights = []
        for result in llm_results.results[:5]:  # Use first 5 results
            if result.processed_content:
                # Extract first sentence or paragraph
                sentences = result.processed_content.split('. ')
                if sentences:
                    key_insights.append(sentences[0] + '.')
        
        if key_insights:
            return ' '.join(key_insights)
        
        return "This document presents a comprehensive analysis of the provided content using advanced natural language processing techniques."
    
    async def _generate_introduction_section_async(self,
                                                 chunks: List[DocumentChunk],
                                                 llm_results: Optional[BatchProcessingResult]) -> str:
        """Generate introduction section."""
        intro_parts = [
            "This report presents a comprehensive analysis of the document content.",
            f"The analysis covers {len(chunks)} content segments extracted from the source material."
        ]
        
        if llm_results:
            intro_parts.append(
                f"Advanced language model processing was applied to {len(llm_results.results)} segments, "
                "providing enhanced insights and understanding."
            )
        
        intro_parts.append(
            "The following sections detail the methodology, findings, and conclusions drawn from this analysis."
        )
        
        return ' '.join(intro_parts)
    
    def _generate_methodology_section(self) -> str:
        """Generate methodology section."""
        methodology_parts = [
            "The analysis employed a multi-stage document processing pipeline:",
            "\n\n1. **Document Parsing**: The source document was parsed and segmented into manageable chunks.",
            "\n2. **Content Analysis**: Each chunk was analyzed using advanced natural language processing techniques.",
            "\n3. **Knowledge Extraction**: Key entities, relationships, and concepts were identified and extracted.",
            "\n4. **Synthesis**: The extracted information was synthesized into coherent insights and findings."
        ]
        
        return ''.join(methodology_parts)
    
    async def _generate_results_section_async(self,
                                            chunks: List[DocumentChunk],
                                            llm_results: Optional[BatchProcessingResult],
                                            knowledge_graph: Optional[KnowledgeGraph]) -> str:
        """Generate results section."""
        results_parts = ["## Key Findings\n\n"]
        
        # Document statistics
        results_parts.append(f"The analysis processed {len(chunks)} content segments.\n\n")
        
        # LLM processing results
        if llm_results:
            successful_results = [r for r in llm_results.results if r.success]
            results_parts.append(
                f"Language model processing achieved a {len(successful_results)/len(llm_results.results)*100:.1f}% "
                f"success rate across {len(llm_results.results)} segments.\n\n"
            )
        
        # Knowledge graph statistics
        if knowledge_graph:
            results_parts.append(
                f"Knowledge extraction identified {len(knowledge_graph.entities)} entities and "
                f"{len(knowledge_graph.relationships)} relationships.\n\n"
            )
            
            # Entity type distribution
            if knowledge_graph.statistics and 'entity_types' in knowledge_graph.statistics:
                results_parts.append("### Entity Distribution\n\n")
                for entity_type, count in knowledge_graph.statistics['entity_types'].items():
                    results_parts.append(f"- {entity_type}: {count}\n")
                results_parts.append("\n")
        
        # Content insights
        if llm_results:
            results_parts.append("### Content Insights\n\n")
            for i, result in enumerate(llm_results.results[:3], 1):
                if result.success and result.processed_content:
                    # Extract key insight
                    content_preview = result.processed_content[:200] + "..." if len(result.processed_content) > 200 else result.processed_content
                    results_parts.append(f"{i}. {content_preview}\n\n")
        
        return ''.join(results_parts)
    
    async def _generate_discussion_section_async(self,
                                               chunks: List[DocumentChunk],
                                               llm_results: Optional[BatchProcessingResult],
                                               knowledge_graph: Optional[KnowledgeGraph]) -> str:
        """Generate discussion section."""
        discussion_parts = []
        
        discussion_parts.append(
            "The analysis reveals several important patterns and insights within the document content.\n\n"
        )
        
        # Discuss processing quality
        if llm_results:
            avg_confidence = sum(r.confidence for r in llm_results.results if r.confidence) / len(llm_results.results)
            discussion_parts.append(
                f"The language model processing achieved an average confidence score of {avg_confidence:.2f}, "
                "indicating high-quality content analysis.\n\n"
            )
        
        # Discuss knowledge graph insights
        if knowledge_graph and knowledge_graph.statistics:
            discussion_parts.append(
                "The knowledge graph analysis provides valuable insights into the document's conceptual structure. "
            )
            
            if 'avg_confidence' in knowledge_graph.statistics:
                discussion_parts.append(
                    f"Entity extraction achieved an average confidence of {knowledge_graph.statistics['avg_confidence']:.2f}.\n\n"
                )
        
        # Discuss limitations and considerations
        discussion_parts.append(
            "### Limitations and Considerations\n\n"
            "While this analysis provides comprehensive insights, several factors should be considered:\n\n"
            "- The analysis is based on automated processing and may not capture all nuances of human interpretation.\n"
            "- Entity and relationship extraction accuracy depends on the clarity and structure of the source content.\n"
            "- Results should be validated against domain expertise where applicable.\n\n"
        )
        
        return ''.join(discussion_parts)
    
    async def _generate_conclusion_section_async(self,
                                               chunks: List[DocumentChunk],
                                               llm_results: Optional[BatchProcessingResult]) -> str:
        """Generate conclusion section."""
        conclusion_parts = []
        
        conclusion_parts.append(
            "This comprehensive analysis has successfully processed and analyzed the document content, "
            "providing valuable insights and structured information extraction.\n\n"
        )
        
        # Summarize key achievements
        achievements = []
        achievements.append(f"Successfully processed {len(chunks)} content segments")
        
        if llm_results:
            successful_results = [r for r in llm_results.results if r.success]
            achievements.append(f"Achieved {len(successful_results)/len(llm_results.results)*100:.1f}% processing success rate")
        
        if achievements:
            conclusion_parts.append("Key achievements include:\n\n")
            for achievement in achievements:
                conclusion_parts.append(f"- {achievement}\n")
            conclusion_parts.append("\n")
        
        conclusion_parts.append(
            "The structured approach and advanced processing techniques employed in this analysis "
            "provide a solid foundation for further research and application of the extracted insights."
        )
        
        return ''.join(conclusion_parts)
    
    def _generate_references_section(self, chunks: List[DocumentChunk]) -> str:
        """Generate references section."""
        references = []
        
        # Extract references from chunk metadata
        for chunk in chunks:
            if hasattr(chunk.metadata, 'source_file') and chunk.metadata.source_file:
                source_ref = f"Source Document: {chunk.metadata.source_file}"
                if source_ref not in references:
                    references.append(source_ref)
        
        if not references:
            references.append("Source: Provided document content")
        
        references_text = "\n".join(f"{i+1}. {ref}" for i, ref in enumerate(references))
        return references_text
    
    async def _generate_executive_summary_section_async(self,
                                                      chunks: List[DocumentChunk],
                                                      llm_results: Optional[BatchProcessingResult],
                                                      knowledge_graph: Optional[KnowledgeGraph]) -> str:
        """Generate executive summary section."""
        summary_parts = []
        
        # High-level overview
        summary_parts.append(
            f"This executive summary presents key findings from the analysis of {len(chunks)} content segments.\n\n"
        )
        
        # Key metrics
        summary_parts.append("**Key Metrics:**\n\n")
        
        if llm_results:
            successful_results = [r for r in llm_results.results if r.success]
            summary_parts.append(f"- Processing Success Rate: {len(successful_results)/len(llm_results.results)*100:.1f}%\n")
        
        if knowledge_graph:
            summary_parts.append(f"- Entities Identified: {len(knowledge_graph.entities)}\n")
            summary_parts.append(f"- Relationships Mapped: {len(knowledge_graph.relationships)}\n")
        
        summary_parts.append("\n")
        
        # Key insights (top 3)
        if llm_results:
            summary_parts.append("**Key Insights:**\n\n")
            for i, result in enumerate(llm_results.results[:3], 1):
                if result.success and result.processed_content:
                    insight = result.processed_content.split('.')[0] + '.'
                    summary_parts.append(f"{i}. {insight}\n")
            summary_parts.append("\n")
        
        # Recommendations
        summary_parts.append(
            "**Recommendations:**\n\n"
            "- Leverage the extracted knowledge graph for further analysis and insights\n"
            "- Consider the identified entities and relationships for strategic planning\n"
            "- Validate key findings with domain experts where applicable\n"
        )
        
        return ''.join(summary_parts)
    
    def _generate_glossary_section(self, knowledge_graph: Optional[KnowledgeGraph]) -> str:
        """Generate glossary section."""
        if not knowledge_graph:
            return "No entities available for glossary generation."
        
        # Create glossary from entities
        glossary_entries = []
        
        # Sort entities by name
        sorted_entities = sorted(knowledge_graph.entities.values(), key=lambda e: e.name.lower())
        
        for entity in sorted_entities[:20]:  # Limit to top 20 entities
            entry = f"**{entity.name}**: {entity.entity_type.name.title()}"
            
            # Add description if available in attributes
            if 'description' in entity.attributes:
                entry += f" - {entity.attributes['description']}"
            
            glossary_entries.append(entry)
        
        if glossary_entries:
            return "\n\n".join(glossary_entries)
        
        return "No glossary entries available."
    
    def _generate_appendix_section(self,
                                 chunks: List[DocumentChunk],
                                 llm_results: Optional[BatchProcessingResult]) -> str:
        """Generate appendix section."""
        appendix_parts = []
        
        # Processing statistics
        appendix_parts.append("## Processing Statistics\n\n")
        appendix_parts.append(f"- Total chunks processed: {len(chunks)}\n")
        
        if llm_results:
            appendix_parts.append(f"- LLM processing results: {len(llm_results.results)}\n")
            successful_results = [r for r in llm_results.results if r.success]
            appendix_parts.append(f"- Successful processing: {len(successful_results)}\n")
            
            if llm_results.total_processing_time:
                appendix_parts.append(f"- Total processing time: {llm_results.total_processing_time:.2f} seconds\n")
        
        appendix_parts.append("\n")
        
        # Technical details
        appendix_parts.append(
            "## Technical Details\n\n"
            "This analysis was performed using advanced natural language processing techniques "
            "including document chunking, language model processing, and knowledge graph extraction.\n\n"
        )
        
        return ''.join(appendix_parts)
    
    async def _generate_visualizations_async(self,
                                           chunks: List[DocumentChunk],
                                           llm_results: Optional[BatchProcessingResult],
                                           knowledge_graph: Optional[KnowledgeGraph]) -> List[Dict[str, Any]]:
        """Generate visualizations for the document."""
        visualizations = []
        
        if not HAS_VISUALIZATION:
            return visualizations
        
        # Generate requested visualization types
        for viz_type in self.config.visualization_types:
            try:
                if viz_type == VisualizationType.ENTITY_DISTRIBUTION and knowledge_graph:
                    viz = await self._create_entity_distribution_chart_async(knowledge_graph)
                    if viz:
                        visualizations.append(viz)
                
                elif viz_type == VisualizationType.PROCESSING_METRICS and llm_results:
                    viz = await self._create_processing_metrics_chart_async(llm_results)
                    if viz:
                        visualizations.append(viz)
                
                elif viz_type == VisualizationType.CONTENT_STATISTICS:
                    viz = await self._create_content_statistics_chart_async(chunks)
                    if viz:
                        visualizations.append(viz)
            
            except Exception as e:
                logger.warning(f"Failed to generate {viz_type.name} visualization: {e}")
        
        return visualizations
    
    async def _create_entity_distribution_chart_async(self, knowledge_graph: KnowledgeGraph) -> Optional[Dict[str, Any]]:
        """Create entity distribution chart."""
        if not knowledge_graph.statistics or 'entity_types' not in knowledge_graph.statistics:
            return None
        
        entity_types = knowledge_graph.statistics['entity_types']
        
        # Create plotly pie chart
        fig = px.pie(
            values=list(entity_types.values()),
            names=list(entity_types.keys()),
            title="Entity Type Distribution"
        )
        
        # Save as HTML
        chart_path = Path(tempfile.mktemp(suffix=".html"))
        fig.write_html(str(chart_path))
        
        return {
            "type": "entity_distribution",
            "title": "Entity Type Distribution",
            "path": chart_path,
            "format": "html",
            "description": "Distribution of entity types found in the document"
        }
    
    async def _create_processing_metrics_chart_async(self, llm_results: BatchProcessingResult) -> Optional[Dict[str, Any]]:
        """Create processing metrics chart."""
        # Prepare data
        success_count = sum(1 for r in llm_results.results if r.success)
        failure_count = len(llm_results.results) - success_count
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(name='Successful', x=['Processing Results'], y=[success_count]),
            go.Bar(name='Failed', x=['Processing Results'], y=[failure_count])
        ])
        
        fig.update_layout(
            title="Processing Success Rate",
            barmode='stack',
            yaxis_title="Number of Chunks"
        )
        
        # Save as HTML
        chart_path = Path(tempfile.mktemp(suffix=".html"))
        fig.write_html(str(chart_path))
        
        return {
            "type": "processing_metrics",
            "title": "Processing Success Rate",
            "path": chart_path,
            "format": "html",
            "description": "Success rate of language model processing"
        }
    
    async def _create_content_statistics_chart_async(self, chunks: List[DocumentChunk]) -> Optional[Dict[str, Any]]:
        """Create content statistics chart."""
        # Calculate chunk size distribution
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        
        # Create histogram
        fig = px.histogram(
            x=chunk_sizes,
            nbins=20,
            title="Chunk Size Distribution",
            labels={'x': 'Chunk Size (characters)', 'y': 'Frequency'}
        )
        
        # Save as HTML
        chart_path = Path(tempfile.mktemp(suffix=".html"))
        fig.write_html(str(chart_path))
        
        return {
            "type": "content_statistics",
            "title": "Chunk Size Distribution",
            "path": chart_path,
            "format": "html",
            "description": "Distribution of content chunk sizes"
        }
    
    async def _compile_document_async(self,
                                    sections: List[OutputSection],
                                    visualizations: List[Dict[str, Any]],
                                    template: OutputTemplate) -> str:
        """Compile final document content."""
        if self._template_env and template.template_content:
            # Use Jinja2 template
            template_obj = self._template_env.from_string(template.template_content)
            
            return template_obj.render(
                sections=sections,
                visualizations=visualizations,
                metadata=self.config.__dict__,
                timestamp=datetime.now(timezone.utc).isoformat(),
                **template.variables
            )
        
        else:
            # Simple compilation
            document_parts = []
            
            # Add title
            if sections:
                title_section = next((s for s in sections if s.section_type == ContentSection.TITLE), None)
                if title_section:
                    document_parts.append(f"# {title_section.content}\n\n")
            
            # Add sections in order
            sorted_sections = sorted(sections, key=lambda s: s.order)
            for section in sorted_sections:
                if section.section_type != ContentSection.TITLE:
                    document_parts.append(f"## {section.title}\n\n")
                    document_parts.append(f"{section.content}\n\n")
            
            # Add visualizations
            if visualizations:
                document_parts.append("## Visualizations\n\n")
                for viz in visualizations:
                    document_parts.append(f"### {viz['title']}\n\n")
                    document_parts.append(f"{viz['description']}\n\n")
                    if viz['format'] == 'html':
                        document_parts.append(f"[Visualization: {viz['path']}]\n\n")
            
            return ''.join(document_parts)
    
    async def _generate_format_output_async(self,
                                          content: str,
                                          output_path: Optional[Path],
                                          template: OutputTemplate) -> GenerationResult:
        """Generate output in the specified format."""
        try:
            if not output_path:
                output_path = Path(f"output.{self.config.output_format.name.lower()}")
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.output_format == OutputFormat.TEXT:
                # Plain text output
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            elif self.config.output_format == OutputFormat.MARKDOWN:
                # Markdown output
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            elif self.config.output_format == OutputFormat.HTML:
                # HTML output
                html_content = await self._convert_to_html_async(content, template)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            elif self.config.output_format == OutputFormat.PDF:
                # PDF output
                success = await self._generate_pdf_async(content, output_path, template)
                if not success:
                    raise Exception("PDF generation failed")
            
            elif self.config.output_format == OutputFormat.DOCX:
                # DOCX output
                success = await self._generate_docx_async(content, output_path, template)
                if not success:
                    raise Exception("DOCX generation failed")
            
            elif self.config.output_format == OutputFormat.JSON:
                # JSON output
                json_content = await self._convert_to_json_async(content)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, indent=2, ensure_ascii=False)
            
            else:
                raise Exception(f"Unsupported output format: {self.config.output_format}")
            
            # Get file size
            file_size = output_path.stat().st_size if output_path.exists() else 0
            
            return GenerationResult(
                success=True,
                output_path=output_path,
                output_content=content if len(content) < 10000 else None,  # Don't store large content
                output_format=self.config.output_format,
                file_size=file_size
            )
        
        except Exception as e:
            logger.error(f"Format generation failed: {e}")
            return GenerationResult(
                success=False,
                errors=[str(e)]
            )
    
    async def _convert_to_html_async(self, content: str, template: OutputTemplate) -> str:
        """Convert content to HTML."""
        if HAS_MARKDOWN:
            # Convert Markdown to HTML
            md = markdown.Markdown(extensions=['codehilite', 'toc', 'tables'])
            html_body = md.convert(content)
        else:
            # Simple HTML conversion
            html_body = content.replace('\n', '<br>\n')
        
        # Wrap in HTML document
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Analysis Report</title>
    <style>
        body {{ font-family: {template.format_settings.get('font_family', 'Arial')}, sans-serif; }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        {self.config.custom_css or ''}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>
        """
        
        return html_template
    
    async def _generate_pdf_async(self, content: str, output_path: Path, template: OutputTemplate) -> bool:
        """Generate PDF output."""
        try:
            if HAS_WEASYPRINT:
                # Use WeasyPrint for HTML to PDF conversion
                html_content = await self._convert_to_html_async(content, template)
                HTML(string=html_content).write_pdf(str(output_path))
                return True
            
            elif HAS_REPORTLAB:
                # Use ReportLab for direct PDF generation
                doc = SimpleDocTemplate(str(output_path), pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Convert content to ReportLab elements
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        if para.startswith('#'):
                            # Header
                            level = len(para) - len(para.lstrip('#'))
                            text = para.lstrip('# ').strip()
                            if level == 1:
                                story.append(Paragraph(text, styles['Title']))
                            elif level == 2:
                                story.append(Paragraph(text, styles['Heading1']))
                            else:
                                story.append(Paragraph(text, styles['Heading2']))
                        else:
                            # Regular paragraph
                            story.append(Paragraph(para, styles['Normal']))
                        
                        story.append(Spacer(1, 12))
                
                doc.build(story)
                return True
            
            else:
                logger.error("No PDF generation library available")
                return False
        
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return False
    
    async def _generate_docx_async(self, content: str, output_path: Path, template: OutputTemplate) -> bool:
        """Generate DOCX output."""
        try:
            if not HAS_PYTHON_DOCX:
                logger.error("python-docx not available for DOCX generation")
                return False
            
            doc = Document()
            
            # Set document styles
            font_name = template.format_settings.get('font_family', 'Calibri')
            
            # Process content
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    if para.startswith('#'):
                        # Header
                        level = len(para) - len(para.lstrip('#'))
                        text = para.lstrip('# ').strip()
                        
                        if level == 1:
                            heading = doc.add_heading(text, level=1)
                        elif level == 2:
                            heading = doc.add_heading(text, level=2)
                        else:
                            heading = doc.add_heading(text, level=3)
                    else:
                        # Regular paragraph
                        p = doc.add_paragraph(para)
                        p.style.font.name = font_name
            
            doc.save(str(output_path))
            return True
        
        except Exception as e:
            logger.error(f"DOCX generation failed: {e}")
            return False
    
    async def _convert_to_json_async(self, content: str) -> Dict[str, Any]:
        """Convert content to JSON format."""
        return {
            "content": content,
            "format": "markdown",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "OutputGenerator",
            "metadata": self.config.__dict__
        }
    
    def _get_section_title(self, section_type: ContentSection) -> str:
        """Get display title for section type."""
        title_mapping = {
            ContentSection.TITLE: "Title",
            ContentSection.ABSTRACT: "Abstract",
            ContentSection.INTRODUCTION: "Introduction",
            ContentSection.METHODOLOGY: "Methodology",
            ContentSection.RESULTS: "Results",
            ContentSection.DISCUSSION: "Discussion",
            ContentSection.CONCLUSION: "Conclusion",
            ContentSection.REFERENCES: "References",
            ContentSection.EXECUTIVE_SUMMARY: "Executive Summary",
            ContentSection.TABLE_OF_CONTENTS: "Table of Contents",
            ContentSection.GLOSSARY: "Glossary",
            ContentSection.APPENDIX: "Appendix"
        }
        
        return title_mapping.get(section_type, section_type.name.title())
    
    def _create_default_template(self) -> OutputTemplate:
        """Create a default template."""
        return OutputTemplate(
            template_type=TemplateType.REPORT,
            name="Default Report",
            description="Default report template",
            sections=[
                ContentSection.TITLE,
                ContentSection.INTRODUCTION,
                ContentSection.RESULTS,
                ContentSection.CONCLUSION
            ]
        )
    
    # Template filters and functions
    def _format_date_filter(self, date_obj: datetime) -> str:
        """Format date for templates."""
        return date_obj.strftime("%Y-%m-%d %H:%M:%S")
    
    def _format_number_filter(self, number: Union[int, float], decimals: int = 2) -> str:
        """Format number for templates."""
        return f"{number:.{decimals}f}"
    
    def _truncate_text_filter(self, text: str, length: int = 100) -> str:
        """Truncate text for templates."""
        return text[:length] + "..." if len(text) > length else text
    
    def _highlight_keywords_filter(self, text: str, keywords: List[str]) -> str:
        """Highlight keywords in text."""
        for keyword in keywords:
            text = re.sub(f'\\b{re.escape(keyword)}\\b', f'**{keyword}**', text, flags=re.IGNORECASE)
        return text
    
    def _generate_table_of_contents(self, sections: List[OutputSection]) -> str:
        """Generate table of contents."""
        toc_lines = []
        for section in sections:
            if section.section_type != ContentSection.TABLE_OF_CONTENTS:
                indent = "  " * (section.level - 1)
                toc_lines.append(f"{indent}- {section.title}")
        
        return "\n".join(toc_lines)
    
    def _format_references(self, references: List[str]) -> str:
        """Format references list."""
        return "\n".join(f"{i+1}. {ref}" for i, ref in enumerate(references))
    
    def _create_visualization_placeholder(self, viz_type: str, title: str) -> str:
        """Create visualization placeholder."""
        return f"[{viz_type.upper()}: {title}]"
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return dict(self._stats)
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            "documents_generated": 0,
            "total_generation_time": 0.0,
            "formats_used": defaultdict(int),
            "templates_used": defaultdict(int)
        }


# Factory functions
def create_output_generator(output_format: OutputFormat = OutputFormat.MARKDOWN,
                          template_type: Optional[TemplateType] = None,
                          **kwargs) -> OutputGenerator:
    """Create an output generator with specified configuration."""
    config = OutputConfig(
        output_format=output_format,
        **kwargs
    )
    
    generator = OutputGenerator(config)
    
    if template_type and template_type in generator._builtin_templates:
        config.template = generator._builtin_templates[template_type]
    
    return generator


async def generate_document_output(chunks: List[DocumentChunk],
                                 llm_results: Optional[BatchProcessingResult] = None,
                                 knowledge_graph: Optional[KnowledgeGraph] = None,
                                 output_format: OutputFormat = OutputFormat.MARKDOWN,
                                 output_path: Optional[Path] = None,
                                 **kwargs) -> GenerationResult:
    """Convenience function for generating document output."""
    generator = create_output_generator(output_format, **kwargs)
    return await generator.generate_output_async(
        chunks, llm_results, knowledge_graph, output_path
    )