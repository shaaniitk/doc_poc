"""Configuration file for document processing"""

# LLM Configuration
LLM_CONFIG = {
    "provider": "gemini",  # Options: "mistral", "openai", "huggingface", "gemini", "vertexai"
    "model": "gemini-1.5-flash",
    "api_key_env": "GEMINI_API_KEY",
    "max_tokens": 2048,
    "temperature": 0.2,
    "timeout": 30
}

# Alternative LLM providers
LLM_PROVIDERS = {
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "models": ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    },
    "huggingface": {
        "url": "https://api-inference.huggingface.co/models/",
        "models": ["mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b-it"]
    },
    "gemini": {
        "models": ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro-latest"]
    },
    "vertexai": {
        "models": ["gemini-1.0-pro", "gemini-1.5-pro-latest", "gemini-2.5-pro"]
    }
}

# Chunking Strategies
CHUNKING_STRATEGIES = {
    "semantic": "Semantic-based chunking",
    "section": "Section-based chunking",
    "llm_enhanced": "LLM-enhanced chunking"
}

# Document Templates
DOCUMENT_TEMPLATES = {
    "bitcoin_paper": [
        {"section": "Summary", "prompt": "Format as LaTeX with section title 'Summary'. Preserve all technical content exactly."},
        {"section": "Abstract", "prompt": "Format as LaTeX with section title 'Abstract'. Preserve all technical content exactly."},
        {"section": "1. Introduction", "prompt": "Format as LaTeX with section title 'Introduction'. Preserve all technical content exactly."},
        {"section": "2. Transactions", "prompt": "Format as LaTeX with section title 'Transactions'. Preserve all technical content exactly."},
        {"section": "3. Timestamp Server", "prompt": "Format as LaTeX with section title 'Timestamp Server'. Preserve all technical content exactly."},
        {"section": "4. Proof-of-Work", "prompt": "Format as LaTeX with section title 'Proof-of-Work'. Preserve all technical content exactly."},
        {"section": "5. Network", "prompt": "Format as LaTeX with section title 'Network'. Preserve all technical content exactly."},
        {"section": "6. Incentive", "prompt": "Format as LaTeX with section title 'Incentive'. Preserve all technical content exactly."},
        {"section": "7. Reclaiming Disk Space", "prompt": "Format as LaTeX with section title 'Reclaiming Disk Space'. Preserve all technical content exactly."},
        {"section": "8. Simplified Payment Verification", "prompt": "Format as LaTeX with section title 'Simplified Payment Verification'. Preserve all technical content exactly."},
        {"section": "9. Combining and Splitting Value", "prompt": "Format as LaTeX with section title 'Combining and Splitting Value'. Preserve all technical content exactly."},
        {"section": "10. Privacy", "prompt": "Format as LaTeX with section title 'Privacy'. Preserve all technical content exactly."},
        {"section": "11. Major and Minor Assumptions", "prompt": "Format as LaTeX with section title 'Major and Minor Assumptions'. Preserve all technical content exactly."},
        {"section": "12. Calculations", "prompt": "Format as LaTeX with section title 'Calculations'. Preserve all technical content exactly."},
        {"section": "13. Conclusion", "prompt": "Format as LaTeX with section title 'Conclusion'. Preserve all technical content exactly."}
    ]
}

# Output Formats
OUTPUT_FORMATS = {
    "latex": {
        "extension": ".tex",
        "header": "\\documentclass{article}\n\\usepackage[pdftex]{graphicx}\n\\usepackage{amsmath}\n\\begin{document}",
        "footer": "\\end{document}"
    },
    "markdown": {
        "extension": ".md",
        "header": "",
        "footer": ""
    },
    "json": {
        "extension": ".json",
        "header": "",
        "footer": ""
    }
}
