"""🧠 UNIFIED LLM CLIENT - MULTI-PROVIDER INTELLIGENCE HUB

This module provides a unified interface to multiple LLM providers,
enabling seamless switching between different AI models and services.
It's the intelligence backbone of the entire document processing system.

🌍 SUPPORTED PROVIDERS:
- Mistral AI: High-quality, efficient models (mistral-small/medium/large)
- OpenAI: Industry-leading GPT models (gpt-3.5-turbo, gpt-4, gpt-4-turbo)
- Hugging Face: Open-source models and inference API
- Gemini: Google's next-generation multimodal models
- Vertex AI: Google Cloud's unified AI platform

🔧 KEY FEATURES:
- Provider abstraction - switch models without code changes
- Consistent API across all providers
- Robust error handling and graceful failures
- Environment-based configuration
- Timeout and rate limiting support

🎯 INTELLIGENCE INTEGRATION:
- Powers semantic chunk analysis
- Drives content synthesis and enhancement
- Enables intelligent decision making
- Provides natural language understanding

This is the brain that makes everything else intelligent!
"""
import os
import requests
from dotenv import load_dotenv
from config import LLM_CONFIG, LLM_PROVIDERS

# Import Google Cloud libraries
import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel

# 🔑 Load environment variables for API keys
load_dotenv()

class UnifiedLLMClient:
    def __init__(self, provider=None, model=None):
        self.provider = provider or LLM_CONFIG["provider"]
        self.model = model or LLM_CONFIG["model"]
        self.config = LLM_PROVIDERS.get(self.provider, {})

        if self.provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
        elif self.provider == "vertexai":
            # Vertex AI authentication is handled by the environment
            # (e.g., gcloud auth application-default login)
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            if project_id:
                vertexai.init(project=project_id, location=location)

    def call_llm(self, prompt, system_prompt="", max_tokens=None, temperature=None):
        """Unified LLM calling interface"""
        max_tokens = max_tokens or LLM_CONFIG["max_tokens"]
        temperature = temperature or LLM_CONFIG["temperature"]

        if self.provider == "mistral":
            return self._call_mistral(prompt, system_prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return self._call_openai(prompt, system_prompt, max_tokens, temperature)
        elif self.provider == "huggingface":
            return self._call_huggingface(prompt, system_prompt, max_tokens, temperature)
        elif self.provider == "gemini":
            return self._call_gemini(prompt, system_prompt, max_tokens, temperature)
        elif self.provider == "vertexai":
            return self._call_vertexai(prompt, system_prompt, max_tokens, temperature)
        else:
            return f"% Error: Unsupported provider {self.provider}"

    def _call_mistral(self, prompt, system_prompt, max_tokens, temperature):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return "% Error: MISTRAL_API_KEY not set"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                self.config["url"],
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=LLM_CONFIG["timeout"]
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"% API Error: {response.status_code}"
        except Exception as e:
            return f"% Error: {str(e)}"

    def _call_openai(self, prompt, system_prompt, max_tokens, temperature):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "% Error: OPENAI_API_KEY not set"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                self.config["url"],
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=LLM_CONFIG["timeout"]
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"% API Error: {response.status_code}"
        except Exception as e:
            return f"% Error: {str(e)}"

    def _call_huggingface(self, prompt, system_prompt, max_tokens, temperature):
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            return "% Error: HUGGINGFACE_API_KEY not set"

        full_prompt = f"{system_prompt}

{prompt}" if system_prompt else prompt

        try:
            response = requests.post(
                f"{self.config['url']}{self.model}",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "inputs": full_prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=LLM_CONFIG["timeout"]
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "% No response")
                return "% Invalid response format"
            else:
                return f"% API Error: {response.status_code}"
        except Exception as e:
            return f"% Error: {str(e)}"

    def _call_gemini(self, prompt, system_prompt, max_tokens, temperature):
        try:
            model = genai.GenerativeModel(self.model)
            full_prompt = f"{system_prompt}

{prompt}" if system_prompt else prompt
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            return f"% Error: {str(e)}"

    def _call_vertexai(self, prompt, system_prompt, max_tokens, temperature):
        try:
            model = GenerativeModel(self.model)
            full_prompt = f"{system_prompt}

{prompt}" if system_prompt else prompt
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return response.text
        except Exception as e:
            return f"% Error: {str(e)}"
import os
import requests
from dotenv import load_dotenv
from config import LLM_CONFIG, LLM_PROVIDERS

# 🔑 Load environment variables for API keys
load_dotenv()

class UnifiedLLMClient:
    def __init__(self, provider=None, model=None):
        self.provider = provider or LLM_CONFIG["provider"]
        self.model = model or LLM_CONFIG["model"]
        self.config = LLM_PROVIDERS.get(self.provider, {})
        
    def call_llm(self, prompt, system_prompt="", max_tokens=None, temperature=None):
        """Unified LLM calling interface"""
        max_tokens = max_tokens or LLM_CONFIG["max_tokens"]
        temperature = temperature or LLM_CONFIG["temperature"]
        
        if self.provider == "mistral":
            return self._call_mistral(prompt, system_prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return self._call_openai(prompt, system_prompt, max_tokens, temperature)
        elif self.provider == "huggingface":
            return self._call_huggingface(prompt, system_prompt, max_tokens, temperature)
        else:
            return f"% Error: Unsupported provider {self.provider}"
    
    def _call_mistral(self, prompt, system_prompt, max_tokens, temperature):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return "% Error: MISTRAL_API_KEY not set"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                self.config["url"],
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=LLM_CONFIG["timeout"]
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"% API Error: {response.status_code}"
        except Exception as e:
            return f"% Error: {str(e)}"
    
    def _call_openai(self, prompt, system_prompt, max_tokens, temperature):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "% Error: OPENAI_API_KEY not set"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                self.config["url"],
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=LLM_CONFIG["timeout"]
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"% API Error: {response.status_code}"
        except Exception as e:
            return f"% Error: {str(e)}"
    
    def _call_huggingface(self, prompt, system_prompt, max_tokens, temperature):
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            return "% Error: HUGGINGFACE_API_KEY not set"
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        try:
            response = requests.post(
                f"{self.config['url']}{self.model}",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "inputs": full_prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=LLM_CONFIG["timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "% No response")
                return "% Invalid response format"
            else:
                return f"% API Error: {response.status_code}"
        except Exception as e:
            return f"% Error: {str(e)}"