# Complete Document Processing System - Technical Documentation

## Overview
This system provides sophisticated document processing with LLM-enhanced reformatting and intelligent augmentation. It follows the exact workflow from `modular_refactor.py` with all modules consolidated into a single file for ease of use.

## System Architecture

### Core Components
1. **Document Chunker** - Extracts and processes document sections
2. **LLM Handler** - Multi-pass processing with context management
3. **Section Mapper** - Maps content to target document structure
4. **Document Combiner** - Intelligent document augmentation
5. **Output Manager** - Handles file generation and PDF compilation

## Detailed Function Documentation

### 1. Document Loading & Parsing

#### `load_latex_file(file_path: str) -> str`
**Purpose**: Safely loads LaTeX files with automatic encoding detection
**Process**:
- Attempts multiple encodings (utf-8, latin-1, ascii)
- Returns decoded file content
- Raises ValueError if file cannot be decoded

#### `extract_latex_sections(content: str) -> List[Dict[str, Any]]`
**Purpose**: Extracts document sections using regex patterns
**Process**:
1. Searches for comment-based sections (`% --- Section Name ---`)
2. Extracts content between section markers
3. Processes each section through `_extract_content_parts()`
4. Handles bibliography separately
5. Returns list of content chunks with metadata

#### `_extract_content_parts(content: str, section_name: str) -> List[Dict[str, Any]]`
**Purpose**: Breaks down section content into manageable parts
**Process**:
1. **Table Extraction**: Finds LaTeX tables with labels
2. **Placeholder Replacement**: Temporarily replaces tables with placeholders
3. **Environment Detection**: Identifies LaTeX environments (equations, figures, etc.)
4. **Content Segmentation**: Splits content while preserving structure
5. **Metadata Assignment**: Tags each part with type and parent section

### 2. Smart Chunking System

#### `llm_enhance_chunking(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
**Purpose**: Uses LLM to intelligently optimize chunk boundaries
**Process**:
1. **Content Classification**: Categorizes chunks (equation, table, text, etc.)
2. **Dependency Analysis**: Identifies content relationships
3. **Smart Merging**: Combines related small chunks
4. **Quality Enhancement**: Improves chunk coherence

#### `_content_type_classification(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
**Purpose**: LLM-based content type identification
**Process**:
- Sends content samples to LLM
- Classifies as: equation, table, figure, code, or text
- Updates chunk metadata with content type
- Handles API failures gracefully

#### `_dependency_aware_chunking(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
**Purpose**: Groups chunks based on content dependencies
**Process**:
- Analyzes consecutive chunks for relationships
- Uses LLM to detect dependencies
- Merges dependent chunks automatically
- Preserves content integrity

#### `_should_merge_chunks(content1: str, content2: str) -> bool`
**Purpose**: LLM decision-making for chunk merging
**Process**:
- Sends both chunks to LLM for analysis
- Receives YES/NO decision on merging
- Considers content coherence and flow
- Returns boolean merge recommendation

### 3. Section Mapping & Assignment

#### `assign_chunks_to_skeleton(grouped_chunks: Dict, template: str) -> Dict[str, List]`
**Purpose**: Maps extracted content to target document structure
**Process**:
1. **Template Loading**: Gets target document skeleton
2. **Section Mapping**: Maps source sections to target sections
3. **Content Assignment**: Assigns chunks to appropriate skeleton sections
4. **Structure Validation**: Ensures all target sections are addressed

#### `get_section_prompt(section_name: str, template: str) -> str`
**Purpose**: Retrieves section-specific processing prompts
**Process**:
- Looks up section in document template
- Returns customized prompt for section type
- Handles equation preservation instructions
- Provides fallback for unknown sections

### 4. Advanced LLM Processing

#### `ContextualLLMHandler` Class
**Purpose**: Manages sophisticated multi-pass LLM processing with context

##### `process_section(section_name: str, content: str, prompt: str) -> str`
**Multi-Pass Processing**:
1. **Analysis Pass**: Analyzes content structure and requirements
2. **Processing Pass**: Processes with enhanced context
3. **Refinement Pass**: Self-critique and improvement
4. **Context Update**: Updates document and section contexts

##### `_analyze_content(content: str, section_name: str) -> str`
**Purpose**: Pre-processing content analysis
**Analysis Points**:
- Key concepts and themes
- Technical complexity level
- Required writing style
- Critical elements to preserve
- Logical flow requirements

##### `_process_with_context(content: str, prompt: str, analysis: str, section_name: str) -> str`
**Purpose**: Main processing with full context awareness
**Context Integration**:
- Document-level context from previous sections
- Section-specific context
- Content analysis results
- Equation preservation instructions

##### `_refine_output(initial_result: str, original_prompt: str, analysis: str) -> str`
**Purpose**: Self-critique and improvement of initial output
**Refinement Criteria**:
- Better task adherence
- Improved clarity and flow
- Technical accuracy maintenance
- LaTeX element preservation

##### `_update_context(section_name: str, new_content: str)`
**Purpose**: Maintains document-wide context memory
**Context Management**:
- Section context (last 500 characters)
- Document context (accumulated key points)
- Global context (section summaries)
- Special Summary section handling

### 5. Multi-Provider LLM Support

#### `_call_mistral(prompt: str, system_prompt: str) -> str`
**Purpose**: Mistral AI API integration
**Features**:
- Configurable model selection
- Temperature and token control
- Error handling and retries
- Response parsing

#### `_call_openai(prompt: str, system_prompt: str) -> str`
**Purpose**: OpenAI API integration
**Features**:
- GPT model support
- Chat completion format
- Configurable parameters
- Error handling

#### `_call_huggingface(prompt: str, system_prompt: str) -> str`
**Purpose**: Hugging Face API integration
**Features**:
- Instruction-following model support
- Custom prompt formatting
- Parameter configuration
- Response extraction

### 6. Document Augmentation System

#### `DocumentCombiner` Class
**Purpose**: Intelligent document combination and augmentation

##### `combine_documents(source1: str, source2: str, strategy: str) -> str`
**Purpose**: Main document combination orchestrator
**Process**:
1. Load both source documents
2. Extract sections from each
3. Apply combination strategy
4. Generate final combined document

##### `_smart_merge(grouped1: Dict, grouped2: Dict) -> str`
**Purpose**: LLM-enhanced intelligent merging
**Process**:
1. **Base Integration**: Start with first document structure
2. **Conflict Analysis**: Identify overlapping sections
3. **LLM Merging**: Use AI to resolve conflicts intelligently
4. **Content Enhancement**: Improve transitions and flow

##### `_llm_merge_sections(base_content: str, additional_content: str, section_name: str) -> str`
**Purpose**: AI-powered section merging
**Capabilities**:
- Contradiction resolution
- Redundancy elimination
- Smooth transition creation
- Technical content preservation
- Coherence improvement

##### `_simple_append(grouped1: Dict, grouped2: Dict) -> str`
**Purpose**: Fallback combination strategy
**Process**:
- Preserves all content from both documents
- Creates separate sections for additional content
- No content loss but less integration

### 7. Output Management

#### `OutputManager` Class
**Purpose**: Handles all file generation and compilation

##### `save_section_output(section_name: str, content: str) -> str`
**Purpose**: Saves individual processed sections
**Features**:
- Safe filename generation
- UTF-8 encoding
- Timestamped filenames
- Path management

##### `aggregate_document(processed_sections: Dict[str, str], title: str) -> Tuple[str, str]`
**Purpose**: Assembles final LaTeX document
**Process**:
1. **Document Structure**: Creates LaTeX preamble
2. **Section Integration**: Adds processed sections in order
3. **Special Formatting**: Handles Summary (unnumbered) and Abstract
4. **PDF Compilation**: Automatically compiles to PDF
5. **File Management**: Returns both .tex and .pdf paths

##### `compile_latex_to_pdf(tex_file: str) -> str`
**Purpose**: Automatic PDF generation
**Process**:
- Runs pdflatex with error suppression
- Handles compilation errors gracefully
- Returns PDF path on success
- Provides error feedback on failure

### 8. Main Processing Workflow

#### `main(source, source2, combine_strategy, template) -> Dict`
**Purpose**: Orchestrates the complete processing pipeline

**Single Document Processing**:
1. **File Loading**: Load and validate source document
2. **Content Extraction**: Extract sections and create chunks
3. **Smart Chunking**: Apply LLM-enhanced chunking
4. **Section Assignment**: Map to target document structure
5. **LLM Processing**: Multi-pass processing with context
6. **Document Assembly**: Create final formatted document
7. **PDF Generation**: Compile to PDF automatically
8. **Logging**: Generate processing logs and reports

**Document Augmentation**:
1. **Dual Loading**: Load both source documents
2. **Section Extraction**: Extract content from both
3. **Intelligent Merging**: Apply LLM-enhanced combination
4. **Conflict Resolution**: Resolve contradictions automatically
5. **Final Assembly**: Generate combined document
6. **Quality Assurance**: Validate output quality

## Configuration System

### Document Templates
- **bitcoin_paper**: 14-section Bitcoin whitepaper structure
- **academic_paper**: 6-section academic format
- Extensible for custom document types

### LLM Configuration
- **Provider Selection**: mistral, openai, huggingface
- **Model Configuration**: Customizable per provider
- **Parameter Control**: Temperature, tokens, etc.
- **Fallback Handling**: Graceful degradation

### Processing Options
- **Chunking Strategies**: semantic vs regex-only
- **Enhancement Levels**: Full LLM vs basic processing
- **Output Formats**: LaTeX with PDF compilation
- **Quality Thresholds**: Configurable quality metrics

## Error Handling & Resilience

### API Failure Management
- Graceful fallbacks for LLM failures
- Timeout handling for API calls
- Rate limit awareness
- Error logging and reporting

### Content Preservation
- Original content fallback on processing failure
- Equation and table preservation priority
- Format integrity maintenance
- Loss prevention mechanisms

### File System Safety
- Encoding detection and handling
- Path sanitization
- Directory creation
- Permission handling

## Performance Optimization

### LLM Call Efficiency
- Context reuse across sections
- Batch processing where possible
- Smart caching of analysis results
- Minimal redundant API calls

### Memory Management
- Streaming file processing
- Context size management
- Garbage collection awareness
- Large document handling

### Processing Speed
- Parallel section processing capability
- Efficient regex operations
- Optimized chunking algorithms
- Fast PDF compilation

## Quality Assurance

### Content Quality Metrics
- Technical accuracy preservation
- Equation integrity validation
- Format consistency checking
- Flow and coherence analysis

### Processing Validation
- Section completeness verification
- Template compliance checking
- Output format validation
- Error detection and reporting

### User Feedback
- Progress reporting
- Quality scoring
- Processing statistics
- Detailed logging

## Usage Patterns

### Single Document Reformatting
```python
result = main(
    source='document.tex',
    template='bitcoin_paper'
)
```

### Document Augmentation
```python
result = main(
    source='base_document.tex',
    source2='additional_content.tex',
    combine_strategy='smart_merge'
)
```

### Custom Configuration
```python
# Edit config.yaml for custom settings
result = main(
    source='document.tex',
    template='custom_template'
)
```

## Extension Points

### Custom Templates
- Add new document structures to DOCUMENT_TEMPLATES
- Define section-specific prompts
- Configure content type handling

### New LLM Providers
- Implement new `_call_provider()` methods
- Add provider configuration
- Handle provider-specific formatting

### Processing Enhancements
- Add new chunking strategies
- Implement custom quality metrics
- Create specialized processors

## Troubleshooting Guide

### Common Issues
1. **API Key Missing**: Set environment variables
2. **PDF Compilation Failed**: Install pdflatex
3. **Encoding Errors**: Check file encoding
4. **Memory Issues**: Process smaller documents

### Debug Information
- Processing logs with timestamps
- Section-by-section progress
- LLM call tracking
- Error stack traces

### Performance Tuning
- Adjust chunk sizes for memory
- Configure LLM parameters
- Optimize regex patterns
- Enable/disable features as needed

This system represents a sophisticated document processing pipeline that combines traditional text processing with modern LLM capabilities to produce high-quality, professionally formatted documents while preserving technical accuracy and mathematical content.