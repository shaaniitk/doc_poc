# Advanced Document Refactoring System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Processing Pipeline](#processing-pipeline)
3. [Component Functions](#component-functions)
4. [Configuration Options](#configuration-options)
5. [Enhancement Scope](#enhancement-scope)
6. [Debugging Options](#debugging-options)

## System Overview

The Advanced Document Refactoring System is a modular pipeline that transforms unstructured LaTeX documents into well-organized, professionally formatted documents using Large Language Models (LLMs). The system provides intelligent chunking, content analysis, section mapping, and comprehensive output generation.

### Key Features
- **Multi-provider LLM support** (Mistral, OpenAI, Hugging Face)
- **Advanced chunking strategies** with semantic analysis
- **Intelligent section mapping** and content assignment
- **Contribution tracking** for full transparency
- **Multiple output formats** (LaTeX, Markdown, JSON)
- **Comprehensive quality assurance**

## Processing Pipeline

### Phase 1: Input Processing
```
Input Document → File Loading → Content Extraction
```

### Phase 2: Content Analysis
```
Raw Content → LaTeX Parsing → Chunk Extraction → Content Enhancement
```

### Phase 3: Structure Mapping
```
Enhanced Chunks → Section Assignment → Template Mapping → Content Organization
```

### Phase 4: LLM Processing
```
Organized Content → Multi-pass Processing → Quality Enhancement → Section Generation
```

### Phase 5: Output Generation
```
Processed Sections → Document Aggregation → Format Conversion → Final Output
```

### Phase 6: Analysis & Reporting
```
Processing Data → Contribution Analysis → Quality Reports → Debug Information
```

## Component Functions

### 1. File Loader (`modules/file_loader.py`)
**Purpose**: Load and preprocess LaTeX documents

**Functions**:
- `load_latex_file(file_path)`: Load LaTeX file with encoding detection
- `validate_latex_structure(content)`: Validate LaTeX syntax
- `preprocess_content(content)`: Clean and normalize content

**Options**:
- Encoding detection (UTF-8, Latin-1, ASCII)
- Structure validation
- Content normalization

### 2. Chunker (`modules/chunker.py`)
**Purpose**: Intelligent document chunking and content segmentation

**Functions**:
- `extract_latex_sections(content)`: Parse LaTeX and extract sections
- `group_chunks_by_section(chunks)`: Group related chunks
- `llm_enhance_chunking(chunks)`: LLM-based chunk optimization
- `dependency_aware_chunking(chunks)`: Analyze content dependencies
- `content_type_classification(chunks)`: Classify content types

**Options**:
- **Chunking Strategies**:
  - `semantic`: LLM-based intelligent chunking
  - `regex_only`: Pattern-based chunking
- **Enhancement Features**:
  - Dependency analysis
  - Content type classification
  - Boundary optimization

### 3. Section Mapper (`modules/section_mapper.py`)
**Purpose**: Map chunks to document skeleton sections

**Functions**:
- `assign_chunks_to_skeleton(grouped_chunks)`: Assign chunks to sections
- `get_section_prompt(section_name)`: Retrieve section-specific prompts
- `get_document_skeleton(template_name)`: Load document templates

**Options**:
- **Document Templates**:
  - `bitcoin_paper`: Bitcoin whitepaper structure
  - `academic_paper`: Standard academic format
- **Mapping Strategies**:
  - Direct mapping
  - Semantic similarity
  - Content-based assignment

### 4. LLM Handler (`modules/llm_handler.py`)
**Purpose**: Contextual LLM processing with memory management

**Functions**:
- `process_section(section_name, content, prompt)`: Process individual sections
- `update_context(section_name, content)`: Update processing context
- `build_contextual_prompt(content, context)`: Create enhanced prompts

**Options**:
- **Context Management**:
  - Document-level context
  - Section-level context
  - Cross-reference tracking
- **Processing Modes**:
  - Single-pass processing
  - Multi-pass refinement
  - Context-aware generation

### 5. LLM Client (`modules/llm_client.py`)
**Purpose**: Unified interface for multiple LLM providers

**Functions**:
- `call_llm(prompt, system_prompt, max_tokens, temperature)`: Universal LLM calling
- `_call_mistral()`: Mistral API integration
- `_call_openai()`: OpenAI API integration
- `_call_huggingface()`: Hugging Face API integration

**Options**:
- **Providers**: Mistral, OpenAI, Hugging Face
- **Models**: Provider-specific model selection
- **Parameters**: Temperature, max tokens, timeout
- **Error Handling**: Retry logic, fallback options

### 6. Advanced Processing (`modules/advanced_processing.py`)
**Purpose**: Multi-pass processing with quality optimization

**Functions**:
- `multi_pass_processing(chunks, section_name, prompt)`: Multi-stage processing
- `cross_section_validation(processed_sections)`: Consistency checking
- `adaptive_prompting(content, section_name)`: Dynamic prompt adaptation
- `iterative_refinement(content, quality_metrics)`: Quality-based refinement

**Options**:
- **Processing Passes**:
  - Analysis pass
  - Initial processing
  - Refinement pass
- **Quality Metrics**:
  - Clarity assessment
  - Technical accuracy
  - Consistency checking

### 7. Intelligent Aggregation (`modules/intelligent_aggregation.py`)
**Purpose**: Smart document assembly with coherence optimization

**Functions**:
- `coherence_optimization(processed_sections)`: Improve document flow
- `terminology_consistency(sections)`: Standardize terminology
- `document_flow_optimization(sections)`: Optimize structure
- `quality_assurance_pass(document)`: Final QA checks

**Options**:
- **Optimization Features**:
  - Transition generation
  - Forward references
  - Terminology standardization
- **QA Dimensions**:
  - Completeness
  - Consistency
  - Clarity
  - Technical accuracy

### 8. Contribution Tracker (`modules/contribution_tracker.py`)
**Purpose**: Track content transformation and chunk distribution

**Functions**:
- `track_chunk_assignment(chunk_id, original_section, target_section)`: Track assignments
- `generate_contribution_report()`: Create analysis report
- `save_contribution_report(output_path)`: Export tracking data

**Options**:
- **Tracking Granularity**:
  - Chunk-level tracking
  - Section-level mapping
  - Content flow analysis
- **Report Formats**:
  - Markdown summary
  - JSON data export
  - Visual flow diagrams

### 9. Output Manager (`modules/output_manager.py`)
**Purpose**: Document assembly and output generation

**Functions**:
- `save_section_output(section_name, content)`: Save individual sections
- `aggregate_document(processed_sections)`: Assemble final document
- `save_processing_log(log_entries)`: Generate processing logs

**Options**:
- **Output Organization**:
  - Session-based folders
  - Timestamped outputs
  - Individual section files
- **Document Assembly**:
  - Template-based structure
  - Custom section ordering
  - Format-specific optimization

### 10. Output Formatter (`modules/output_formatter.py`)
**Purpose**: Multi-format document generation

**Functions**:
- `format_document(processed_sections)`: Apply format-specific styling
- `_format_latex(sections)`: LaTeX document generation
- `_format_markdown(sections)`: Markdown conversion
- `_format_json(sections)`: JSON export

**Options**:
- **Output Formats**:
  - LaTeX (publication-ready)
  - Markdown (web-friendly)
  - JSON (data exchange)
- **Formatting Features**:
  - Template customization
  - Style preservation
  - Cross-format compatibility

## Configuration Options

### LLM Configuration (`config.py`)
```python
LLM_CONFIG = {
    "provider": "mistral",           # LLM provider selection
    "model": "mistral-small-latest", # Model specification
    "max_tokens": 2048,              # Response length limit
    "temperature": 0.2,              # Creativity control
    "timeout": 30                    # Request timeout
}
```

### Document Templates
```python
DOCUMENT_TEMPLATES = {
    "bitcoin_paper": [...],          # Bitcoin whitepaper structure
    "academic_paper": [...],         # Academic paper format
    "technical_report": [...]        # Technical report layout
}
```

### Chunking Strategies
```python
CHUNKING_STRATEGIES = {
    "semantic": {
        "use_llm": True,             # Enable LLM enhancement
        "merge_threshold": 200,      # Minimum chunk size
        "split_threshold": 2000      # Maximum chunk size
    }
}
```

### Output Formats
```python
OUTPUT_FORMATS = {
    "latex": {
        "extension": ".tex",         # File extension
        "preserve_environments": True # LaTeX environment handling
    }
}
```

## Enhancement Scope

### Current Capabilities
1. **Multi-provider LLM Support**: Mistral, OpenAI, Hugging Face
2. **Advanced Chunking**: Semantic analysis, dependency tracking
3. **Intelligent Mapping**: Content-aware section assignment
4. **Quality Optimization**: Multi-pass processing, consistency checking
5. **Comprehensive Tracking**: Full content transformation audit
6. **Multiple Outputs**: LaTeX, Markdown, JSON formats

### Planned Enhancements

#### Phase 1: Advanced Analytics
- **Content Similarity Analysis**: Detect duplicate or similar content
- **Citation Extraction**: Automatic reference identification
- **Figure/Table Analysis**: Enhanced multimedia content handling
- **Language Detection**: Multi-language document support

#### Phase 2: Interactive Features
- **Web Interface**: Browser-based document processing
- **Real-time Preview**: Live document generation
- **Manual Override**: User-guided section assignment
- **Batch Processing**: Multiple document handling

#### Phase 3: AI Enhancements
- **Custom Model Training**: Domain-specific model fine-tuning
- **Automated Quality Scoring**: ML-based quality assessment
- **Content Recommendation**: Suggest improvements and additions
- **Style Transfer**: Adapt writing style to target audience

#### Phase 4: Integration Features
- **API Endpoints**: RESTful service interface
- **Database Integration**: Persistent storage and retrieval
- **Version Control**: Document change tracking
- **Collaboration Tools**: Multi-user editing support

## Debugging Options

### Logging Levels
```python
# Set in environment or config
DEBUG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
```

### Debug Modes

#### 1. Verbose Processing
```bash
python modular_refactor.py --debug --verbose
```
- Detailed step-by-step logging
- Intermediate file outputs
- Processing time measurements
- Memory usage tracking

#### 2. Chunk Analysis
```bash
python modular_refactor.py --debug-chunks
```
- Individual chunk inspection
- Content type classification results
- Dependency analysis output
- Chunk assignment decisions

#### 3. LLM Interaction Debugging
```bash
python modular_refactor.py --debug-llm
```
- Full prompt/response logging
- API call timing and status
- Token usage statistics
- Error response analysis

#### 4. Section Mapping Debug
```bash
python modular_refactor.py --debug-mapping
```
- Section assignment logic
- Template matching results
- Content distribution analysis
- Mapping confidence scores

### Debug Output Files

#### Processing Logs
- `processing_log.txt`: Complete processing timeline
- `error_log.txt`: Error messages and stack traces
- `performance_log.txt`: Timing and resource usage

#### Analysis Reports
- `chunk_contributions.md`: Content transformation tracking
- `quality_report.json`: Quality assessment results
- `section_analysis.txt`: Section-by-section breakdown

#### Intermediate Files
- `raw_chunks.json`: Initial chunk extraction
- `enhanced_chunks.json`: Post-LLM chunk enhancement
- `section_assignments.json`: Chunk-to-section mapping

### Troubleshooting Guide

#### Common Issues

1. **API Key Errors**
   - Verify environment variables
   - Check API key validity
   - Confirm provider selection

2. **Memory Issues**
   - Reduce chunk size limits
   - Enable chunk streaming
   - Use lighter LLM models

3. **Quality Issues**
   - Adjust temperature settings
   - Enable multi-pass processing
   - Review prompt templates

4. **Performance Issues**
   - Enable parallel processing
   - Optimize chunk sizes
   - Use local models for speed

#### Debug Commands
```bash
# Full debug mode
python modular_refactor.py --source document.tex --debug-all

# Specific component debugging
python modular_refactor.py --debug-component chunker

# Performance profiling
python modular_refactor.py --profile --source document.tex

# Test mode (no LLM calls)
python modular_refactor.py --test-mode --source document.tex
```

### Monitoring and Metrics

#### Real-time Monitoring
- Processing progress indicators
- Resource usage dashboards
- Error rate tracking
- Quality score trends

#### Performance Metrics
- Processing time per section
- Token usage efficiency
- Memory consumption patterns
- API response times

#### Quality Metrics
- Content preservation rates
- Structural improvement scores
- Consistency measurements
- User satisfaction ratings

---

## Getting Started

### Basic Usage
```bash
# Process Bitcoin whitepaper
python modular_refactor.py --source bitcoin_whitepaper.tex

# Use different template
python modular_refactor.py --source document.tex --template academic_paper

# Enable debugging
python modular_refactor.py --source document.tex --debug
```

### Advanced Usage
```bash
# Custom configuration
python modular_refactor.py --source document.tex --provider openai --model gpt-4

# Multiple outputs
python modular_refactor.py --source document.tex --format latex,markdown,json

# Batch processing
python modular_refactor.py --batch documents/ --template academic_paper
```

This documentation provides a comprehensive overview of the system's capabilities, configuration options, and debugging features. The modular design allows for easy extension and customization based on specific requirements.