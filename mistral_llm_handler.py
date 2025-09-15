"""Simple Mistral API integration for testing purposes."""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MistralLLMHandler:
    """Simple Mistral API handler for testing."""
    
    def __init__(self, api_key: str, model: str = "mistral-small", temperature: float = 0.1, max_tokens: int = 100):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Mistral API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Mistral API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "choices" not in result or len(result["choices"]) == 0:
                        raise Exception("No response from Mistral API")
                    
                    return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"Mistral API call failed: {e}")
            raise
    
    async def generate_structured_response(self, prompt: str, schema: dict, **kwargs) -> dict:
        """Generate structured response (simplified for testing)."""
        response = await self.generate_response(prompt, **kwargs)
        
        # Try to parse as JSON, fallback to simple structure
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"result": response, "confidence": 0.8}
    
    def is_available(self) -> bool:
        """Check if handler is available."""
        return bool(self.api_key)

class MistralLLMWrapper:
    """Wrapper to make MistralLLMHandler compatible with existing code."""
    
    def __init__(self, api_key: str, model: str = "mistral-small", temperature: float = 0.1, max_tokens: int = 100):
        self.handler = MistralLLMHandler(api_key, model, temperature, max_tokens)
        self.model = model
        self.temperature = temperature
    
    async def agenerate(self, messages):
        """Generate response compatible with existing LLM interface."""
        # Convert messages to simple prompt
        if isinstance(messages, list) and len(messages) > 0:
            if hasattr(messages[0], 'content'):
                prompt = messages[0].content
            elif isinstance(messages[0], dict) and 'content' in messages[0]:
                prompt = messages[0]['content']
            else:
                prompt = str(messages[0])
        else:
            prompt = str(messages)
        
        return await self.handler.generate_response(prompt)
    
    async def ainvoke(self, messages):
        """Invoke method compatible with LangChain interface."""
        response_text = await self.agenerate(messages)
        
        # Return object with content attribute
        class MistralResponse:
            def __init__(self, content):
                self.content = content
        
        return MistralResponse(response_text)

# Convenience function for easy usage
def create_mistral_llm(api_key: str, model: str = "mistral-small", temperature: float = 0.1, max_tokens: int = 100):
    """Create a Mistral LLM wrapper ready for use."""
    return MistralLLMWrapper(api_key, model, temperature, max_tokens)