# Quick Reference Guide

## Command Line Usage

### Basic Commands
```bash
# Process local file
python modular_refactor.py --source document.tex

# Download and process from URL
python modular_refactor.py --source https://example.com/paper.tex

# Process arXiv paper
python modular_refactor.py --source arxiv:2301.12345

# Use specific template
python modular_refactor.py --source document.tex --template academic_paper
```

### Debug Commands
```bash
# Enable debugging
python modular_refactor.py --source document.tex --debug

# Verbose output
python modular_refactor.py --source document.tex --verbose

# Component-specific debugging
python modular_refactor.py --source document.tex --debug-chunks
python modular_refactor.py --source document.tex --debug-llm
python modular_refactor.py --source document.tex --debug-mapping
```

## Configuration Quick Setup

### Environment Variables
```bash
export MISTRAL_API_KEY="your_mistral_key"
export OPENAI_API_KEY="your_openai_key"
export HUGGINGFACE_TOKEN="your_hf_token"
```

### Config File Modifications
```python
# Change LLM provider
LLM_CONFIG["provider"] = "openai"  # mistral, openai, huggingface

# Adjust processing parameters
LLM_CONFIG["temperature"] = 0.7    # 0.0-1.0 (creativity)
LLM_CONFIG["max_tokens"] = 4096    # Response length

# Enable advanced chunking
CHUNKING_STRATEGIES["semantic"]["use_llm"] = True
```

## Output Files Structure
```
outputs/YYYYMMDD_HHMMSS/
├── final_document.tex           # Main output
├── processing_log.txt           # Processing timeline
├── chunk_contributions.md       # Content tracking
├── Abstract.tex                 # Individual sections
├── 1_Introduction.tex
├── 2_Transactions.tex
└── ...
```

## Key Functions by Module

### File Operations
- `load_latex_file()` - Load document
- `save_section_output()` - Save sections
- `aggregate_document()` - Combine sections

### Content Processing
- `extract_latex_sections()` - Parse LaTeX
- `group_chunks_by_section()` - Organize chunks
- `assign_chunks_to_skeleton()` - Map to template

### LLM Operations
- `call_llm()` - Universal LLM interface
- `process_section()` - Section processing
- `multi_pass_processing()` - Advanced processing

### Analysis & Tracking
- `track_chunk_assignment()` - Content tracking
- `generate_contribution_report()` - Analysis report
- `quality_assurance_pass()` - QA checks

## Common Issues & Solutions

### API Errors
- **Issue**: "MISTRAL_API_KEY not set"
- **Solution**: Set environment variable or check .env file

### Memory Issues
- **Issue**: Out of memory during processing
- **Solution**: Reduce chunk sizes in config

### Quality Issues
- **Issue**: Poor output quality
- **Solution**: Lower temperature, enable multi-pass processing

### Performance Issues
- **Issue**: Slow processing
- **Solution**: Use smaller models, optimize chunk sizes

## Templates Available

### Bitcoin Paper Template
- Summary, Abstract, Introduction
- Transactions, Timestamp Server, Proof-of-Work
- Network, Incentive, Disk Space
- SPV, Value Splitting, Privacy
- Assumptions, Calculations, Conclusion

### Academic Paper Template
- Abstract, Introduction, Literature Review
- Methodology, Results, Discussion
- Conclusion, References

## Enhancement Checklist

### Before Processing
- [ ] API keys configured
- [ ] Template selected
- [ ] Input file accessible
- [ ] Output directory writable

### During Processing
- [ ] Monitor progress logs
- [ ] Check for errors
- [ ] Verify section generation
- [ ] Review intermediate outputs

### After Processing
- [ ] Compile LaTeX output
- [ ] Review contribution report
- [ ] Check quality metrics
- [ ] Validate content preservation

## Performance Optimization

### Speed Improvements
- Use local models for development
- Reduce max_tokens for faster responses
- Enable parallel processing
- Optimize chunk sizes

### Quality Improvements
- Enable multi-pass processing
- Use higher-quality models
- Adjust temperature settings
- Review and refine prompts

### Resource Management
- Monitor memory usage
- Set appropriate timeouts
- Use streaming for large files
- Implement caching strategies