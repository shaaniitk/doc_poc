"""UNIFIED LLM CLIENT - MULTI-PROVIDER INTELLIGENCE HUB"""
import os
import requests
from dotenv import load_dotenv
from config import LLM_CONFIG, LLM_PROVIDERS

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

    def call_llm(self, prompt, system_prompt="", max_tokens=None, temperature=None):
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

    def _call_gemini(self, prompt, system_prompt, max_tokens, temperature):
        if not genai:
            return "% Error: google-generativeai not installed"
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
            return f"% Error: {str(e)}"

    def _call_vertexai(self, prompt, system_prompt, max_tokens, temperature):
        if not vertexai or not GenerativeModel:
            return "% Error: vertexai not installed"
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
            return f"% Error: {str(e)}"