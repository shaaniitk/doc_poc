"""Configuration file for document processing"""

# LLM Configuration
LLM_CONFIG = {
    "provider": "mistral",  # Options: "mistral", "openai", "huggingface", "local"
    "model": "mistral-small-latest",
    "api_key_env": "MISTRAL_API_KEY",
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
        "models": ["mistralai/Mistral-7B-Instruct-v0.3", "microsoft/DialoGPT-medium"]
    }
}

# Output Format Options
OUTPUT_FORMATS = {
    "latex": {
        "extension": ".tex",
        "template": "latex_template",
        "preserve_environments": True
    },
    "markdown": {
        "extension": ".md", 
        "template": "markdown_template",
        "preserve_environments": False
    },
    "json": {
        "extension": ".json",
        "template": "json_template", 
        "preserve_environments": True
    }
}

# Chunking Strategies
CHUNKING_STRATEGIES = {
    "semantic": {
        "use_llm": True,
        "merge_threshold": 200,
        "split_threshold": 2000
    },
    "regex_only": {
        "use_llm": False,
        "merge_threshold": 0,
        "split_threshold": 0
    },
    "hybrid": {
        "use_llm": True,
        "merge_threshold": 150,
        "split_threshold": 1500
    }
}

# Document Structure Templates
DOCUMENT_TEMPLATES = {
    "academic_paper": [
        {"section": "Title", "prompt": "Create a professional academic title."},
        {"section": "Abstract", "prompt": "Write a concise abstract (150 words)."},
        {"section": "1. Introduction", "prompt": "Write a compelling introduction."},
        {"section": "2. Literature Review", "prompt": "Structure the literature review."},
        {"section": "3. Methodology", "prompt": "Detail the methodology."},
        {"section": "4. Results", "prompt": "Present results clearly."},
        {"section": "5. Discussion", "prompt": "Discuss implications."},
        {"section": "6. Conclusion", "prompt": "Summarize contributions."},
        {"section": "References", "prompt": "Format references properly."}
    ],
    "technical_report": [
        {"section": "Executive Summary", "prompt": "Write executive summary."},
        {"section": "1. Introduction", "prompt": "Introduce the problem."},
        {"section": "2. Technical Approach", "prompt": "Detail technical approach."},
        {"section": "3. Implementation", "prompt": "Describe implementation."},
        {"section": "4. Results", "prompt": "Present results."},
        {"section": "5. Recommendations", "prompt": "Provide recommendations."}
    ]
}