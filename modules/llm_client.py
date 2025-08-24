"""UNIFIED LLM CLIENT - MULTI-PROVIDER INTELLIGENCE HUB"""
import os
import requests
from dotenv import load_dotenv
from config import LLM_CONFIG, LLM_PROVIDERS
from .error_handler import LLMError, ResponseValidator, CircuitBreaker

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
except ImportError:
    vertexai = None
    GenerativeModel = None

load_dotenv()

class UnifiedLLMClient:
    def __init__(self, provider=None, model=None):
        self.provider = provider or LLM_CONFIG["provider"]
        self.model = model or LLM_CONFIG["model"]
        self.config = LLM_PROVIDERS.get(self.provider, {})
        # Basic circuit breaker to prevent cascading failures
        self.circuit_breaker = CircuitBreaker()

    def call_llm(self, prompt, system_prompt="", max_tokens=None, temperature=None):
        max_tokens = max_tokens or LLM_CONFIG["max_tokens"]
        temperature = temperature or LLM_CONFIG["temperature"]

        if not self.circuit_breaker.allow_request():
            raise LLMError("Circuit breaker is OPEN; skipping LLM call")

        try:
            if self.provider == "mistral":
                raw = self._call_mistral(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == "openai":
                raw = self._call_openai(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == "huggingface":
                raw = self._call_huggingface(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == "gemini":
                raw = self._call_gemini(prompt, system_prompt, max_tokens, temperature)
            elif self.provider == "vertexai":
                raw = self._call_vertexai(prompt, system_prompt, max_tokens, temperature)
            else:
                raise LLMError(f"Unsupported provider {self.provider}")

            # Validate response before returning
            validated = ResponseValidator.validate_llm_response(raw)
            self.circuit_breaker.record_success()
            return validated
        except Exception as e:
            # Convert to LLMError and record failure
            if not isinstance(e, LLMError):
                e = LLMError(str(e))
            self.circuit_breaker.record_failure()
            raise e

    def _call_mistral(self, prompt, system_prompt, max_tokens, temperature):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise LLMError("MISTRAL_API_KEY not set")

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
                raise LLMError(f"API Error: {response.status_code}")
        except Exception as e:
            raise LLMError(str(e))

    def _call_openai(self, prompt, system_prompt, max_tokens, temperature):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OPENAI_API_KEY not set")

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
                raise LLMError(f"API Error: {response.status_code}")
        except Exception as e:
            raise LLMError(str(e))

    def _call_huggingface(self, prompt, system_prompt, max_tokens, temperature):
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise LLMError("HUGGINGFACE_API_KEY not set")

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
                if isinstance(result, list) and len(result) > 0 and result[0].get("generated_text"):
                    return result[0]["generated_text"]
                raise LLMError("Invalid response format")
            else:
                raise LLMError(f"API Error: {response.status_code}")
        except Exception as e:
            raise LLMError(str(e))

    def _call_gemini(self, prompt, system_prompt, max_tokens, temperature):
        if not genai:
            raise LLMError("google-generativeai not installed")
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model)
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            raise LLMError(str(e))

    def _call_vertexai(self, prompt, system_prompt, max_tokens, temperature):
        if not vertexai or not GenerativeModel:
            raise LLMError("vertexai not installed")
        try:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            if project_id:
                vertexai.init(project=project_id, location=location)
            model = GenerativeModel(self.model)
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return response.text
        except Exception as e:
            raise LLMError(str(e))