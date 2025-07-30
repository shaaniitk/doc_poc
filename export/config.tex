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

# Document Structure Templates
DOCUMENT_TEMPLATES = {
    "bitcoin_paper": [
        {"section": "Summary", "prompt": "Create a comprehensive summary of the entire document covering key findings, major results, and main contributions. Do not include equations or tables, focus on high-level insights and conclusions."},
        {"section": "Abstract", "prompt": "Rewrite as a concise abstract summarizing the peer-to-peer electronic cash system."},
        {"section": "1. Introduction", "prompt": "Rewrite as an introduction explaining the problem with current payment systems."},
        {"section": "2. Transactions", "prompt": "Explain how electronic coins work as chains of digital signatures."},
        {"section": "3. Timestamp Server", "prompt": "Describe the timestamp server solution."},
        {"section": "4. Proof-of-Work", "prompt": "Explain the proof-of-work system and its implementation."},
        {"section": "5. Network", "prompt": "Detail the network protocol and steps."},
        {"section": "6. Incentive", "prompt": "Explain the incentive mechanism for nodes."},
        {"section": "7. Reclaiming Disk Space", "prompt": "Describe disk space optimization using Merkle trees."},
        {"section": "8. Simplified Payment Verification", "prompt": "Explain SPV for lightweight clients."},
        {"section": "9. Combining and Splitting Value", "prompt": "Describe transaction inputs and outputs."},
        {"section": "10. Privacy", "prompt": "Explain the privacy model and key anonymity."},
        {"section": "11. Major and Minor Assumptions", "prompt": "Identify and explain the key assumptions underlying the Bitcoin system, categorizing them as major (critical) or minor (less critical) assumptions."},
        {"section": "12. Calculations", "prompt": "Present the mathematical analysis of attack probabilities."},
        {"section": "13. Conclusion", "prompt": "Summarize the proposed system and its benefits."}
    ],
    "academic_paper": [
        {"section": "Abstract", "prompt": "Write a concise abstract (150 words)."},
        {"section": "1. Introduction", "prompt": "Write a compelling introduction."},
        {"section": "2. Literature Review", "prompt": "Structure the literature review."},
        {"section": "3. Methodology", "prompt": "Detail the methodology."},
        {"section": "4. Results", "prompt": "Present results clearly."},
        {"section": "5. Discussion", "prompt": "Discuss implications."},
        {"section": "6. Conclusion", "prompt": "Summarize contributions."},
        {"section": "References", "prompt": "Format references properly."}
    ]
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
    }
}

# LLM Prompts for Chunking
CHUNKING_PROMPTS = {
    "merge_decision": "Should these two text segments be merged into one coherent paragraph? Respond with only 'YES' or 'NO'.",
    "boundary_detection": "Identify natural break points in this text where it should be split into coherent chunks. Return only the first few words of each new chunk, separated by '|||'.",
    "system_prompt": "You are a technical writer. Output only raw LaTeX content without explanations. Preserve all equations, tables, figures exactly.",
    "content_classification": "Classify this content type. Respond with only: 'equation', 'table', 'figure', 'code', 'paragraph', or 'mixed'.",
    "dependency_analysis": "Does this content reference or depend on the previous content? Respond with only 'YES' or 'NO'.",
    "section_assignment": "Which document section does this content best fit? Choose from: Abstract, Introduction, Theory, Methods, Results, Discussion, Conclusion."
}