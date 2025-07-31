# Complete Document Processing System

Self-contained document processing with LLM enhancement and PDF generation.

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**:
   ```bash
   cp .env.template .env
   # Edit .env and add your MISTRAL_API_KEY
   ```

3. **Run Processing**:
   ```bash
   jupyter notebook notebook.ipynb
   ```

## 📁 Files Included

- `modular_document_processor.py` - Complete processing system
- `notebook.ipynb` - Jupyter interface
- `config.yaml` - Configuration settings
- `bitcoin_whitepaper.tex` - Sample input document
- `blockchain_security.tex` - Sample augmentation document
- `requirements.txt` - Python dependencies
- `.env.template` - API key template

## 🎯 Features

- **LLM Processing**: Real API calls to Mistral/OpenAI/HuggingFace
- **PDF Generation**: Automatic LaTeX to PDF compilation
- **Document Augmentation**: Smart merging of multiple documents
- **All-in-one**: Everything generated in current directory
- **Self-contained**: No external dependencies except API keys

## 📋 Usage

### Single Document Processing
```python
result = main(
    source='bitcoin_whitepaper.tex',
    template='bitcoin_paper'
)
```

### Document Augmentation
```python
result = main(
    source='document1.tex',
    source2='document2.tex',
    combine_strategy='smart_merge'
)
```

## 🔧 Configuration

Edit `config.yaml` to change:
- LLM provider (mistral/openai/huggingface)
- Model settings
- Processing parameters

## 📄 Output Files

All files are generated in the current directory:
- `final_document_TIMESTAMP.tex` - Processed LaTeX
- `final_document_TIMESTAMP.pdf` - Compiled PDF
- `processing_log_TIMESTAMP.txt` - Processing log
- Individual section files

## 🛠️ Requirements

- Python 3.8+
- pdflatex (for PDF generation)
- API key for chosen LLM provider

## ✅ Ready to Use

This folder contains everything needed for document processing. Just add your API key and run!