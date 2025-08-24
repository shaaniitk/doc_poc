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
        {"section": "Summary", "prompt": "Create a comprehensive summary covering key findings and results."},
        {"section": "Abstract", "prompt": "Rewrite as a concise abstract for the peer-to-peer electronic cash system."},
        {"section": "1. Introduction", "prompt": "Explain the problem with current payment systems."},
        {"section": "2. Transactions", "prompt": "Explain how electronic coins work as chains of digital signatures."},
        {"section": "3. Timestamp Server", "prompt": "Describe the timestamp server solution."},
        {"section": "4. Proof-of-Work", "prompt": "Explain the proof-of-work system and implementation."},
        {"section": "5. Network", "prompt": "Detail the network protocol and steps."},
        {"section": "6. Incentive", "prompt": "Explain the incentive mechanism for nodes."},
        {"section": "7. Reclaiming Disk Space", "prompt": "Explain how to reclaim disk space by pruning old transactions."},
        {"section": "8. Simplified Payment Verification", "prompt": "Describe simplified payment verification for lightweight clients."},
        {"section": "9. Combining and Splitting Value", "prompt": "Explain how to handle multiple inputs and outputs in transactions."},
        {"section": "10. Privacy", "prompt": "Discuss privacy considerations and limitations."},
        {"section": "11. Major and Minor Assumptions", "prompt": "Analyze the key assumptions underlying the system."},
        {"section": "12. Calculations", "prompt": "Present the mathematical analysis of attack probabilities."},
        {"section": "13. Conclusion", "prompt": "Summarize the proposed system and its benefits."}
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
