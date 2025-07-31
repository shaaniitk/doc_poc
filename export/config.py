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
