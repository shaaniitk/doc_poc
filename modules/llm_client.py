
import os
import requests
import logging
from dotenv import load_dotenv
import config
from .error_handler import LLMError, ResponseValidator, CircuitBreaker
from langchain.llms.base import LLM # <-- LANGCHAIN IMPORT
from typing import Any, List, Mapping, Optional # <-- LANGCHAIN IMPORT

# Dynamically import optional Google libraries
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None

load_dotenv()

class UnifiedLLMClient:
    def __init__(self, provider=None):
        llm_config = getattr(config, 'LLM_CONFIG', {})
        llm_providers = getattr(config, 'LLM_PROVIDERS', {})
        self.provider = provider or llm_config.get("provider", "gemini")
        self.config = llm_providers.get(self.provider, {})
        self.circuit_breaker = CircuitBreaker()
        self.logger = logging.getLogger(self.__class__.__name__)

    def call_llm(self, messages, model=None, max_tokens=None, temperature=None, stream=False):
        if not self.circuit_breaker.allow_request():
            raise LLMError("Circuit breaker is OPEN; skipping LLM call")

        llm_config = getattr(config, 'LLM_CONFIG', {})
        model = model or llm_config.get("model", "gemini-1.5-flash")
        max_tokens = max_tokens or llm_config.get("max_tokens", 4000)
        temperature = temperature or llm_config.get("temperature", 0.7)
        
        try:
            provider_method_map = {
                "gemini": self._call_gemini, 
                "openai": self._call_openai, 
                "mistral": self._call_mistral,
                "huggingface_local": self._call_huggingface_local
            }
            provider_method = provider_method_map.get(self.provider)
            
            if not provider_method:
                raise LLMError(f"Unsupported provider: {self.provider}")

            raw_response = provider_method(messages, model, max_tokens, temperature, stream)

            if not stream:
                validated = ResponseValidator.validate_llm_response(raw_response)
                self.circuit_breaker.record_success()
                return validated
            else:
                self.circuit_breaker.record_success()
                return raw_response

        except Exception as e:
            if not isinstance(e, LLMError):
                e = LLMError(f"LLM call failed for provider {self.provider}: {str(e)}")
            self.circuit_breaker.record_failure()
            raise e
    
    # --- Provider-Specific Implementations ---
    def _call_gemini(self, messages, model_name, max_tokens, temperature, stream):
        if not genai: raise LLMError("google-generativeai library not installed.")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise LLMError("GEMINI_API_KEY not set in environment.")
        genai.configure(api_key=api_key)

        system_prompt, gemini_messages = None, []
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
                continue
            role = 'model' if msg['role'] == 'assistant' else 'user'
            content = msg['content']
            if not content or not content.strip(): raise LLMError(f"Empty content detected for role {role}")
            gemini_messages.append({'role': role, 'parts': [content]})

        model_kwargs = {'system_instruction': system_prompt} if system_prompt else {}
        model = genai.GenerativeModel(model_name, **model_kwargs)
        generation_config = genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)

        if stream: raise NotImplementedError("Streaming is not yet implemented for the Gemini client.")
        response = model.generate_content(
            gemini_messages, generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        return response.text

    def _call_openai(self, messages, model_name, max_tokens, temperature, stream):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise LLMError("OPENAI_API_KEY not set.")
        response = requests.post(
            self.config["url"], headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": stream},
            timeout=getattr(config, 'LLM_CONFIG', {}).get("timeout", 30)
        )
        if response.status_code == 200:
            if stream: raise NotImplementedError("Streaming is not yet implemented for the OpenAI client.")
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise LLMError(f"OpenAI API Error: {response.status_code} - {response.text}")

    def _call_mistral(self, messages, model_name, max_tokens, temperature, stream):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key: raise LLMError("MISTRAL_API_KEY not set.")
        response = requests.post(
            self.config["url"], headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": stream},
            timeout=getattr(config, 'LLM_CONFIG', {}).get("timeout", 30)
        )
        if response.status_code == 200:
            if stream: raise NotImplementedError("Streaming is not yet implemented for the Mistral client.")
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise LLMError(f"Mistral API Error: {response.status_code} - {response.text}")
    
    def _call_huggingface_local(self, messages, model_name, max_tokens, temperature, stream):
        """Handle local Hugging Face model calls using transformers library"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except ImportError:
            raise LLMError("transformers library not installed. Install with: pip install transformers torch")
        
        try:
            # Get device configuration
            llm_config = getattr(config, 'LLM_CONFIG', {})
            device_config = llm_config.get('device', 'auto')
            
            # Determine device
            if device_config == 'auto':
                device = 0 if torch.cuda.is_available() else -1
            elif device_config == 'cpu':
                device = -1
            else:
                device = device_config
            
            # Convert messages to a single prompt string
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nassistant:"
            
            # Create text generation pipeline with cache folder
            cache_dir = llm_config.get('cache_folder', './models')
            generator = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device=device,
                torch_dtype=torch.float16 if device >= 0 else torch.float32,
                trust_remote_code=True,
                model_kwargs={'cache_dir': cache_dir}
            )
            
            # GPT-2 specific: Ensure prompt + max_tokens doesn't exceed 1024 token limit
            tokenizer = generator.tokenizer
            prompt_tokens = tokenizer.encode(prompt)
            
            # GPT-2 has a hard limit of 1024 tokens total (input + output)
            max_context_length = 1024
            available_tokens_for_generation = max_context_length - len(prompt_tokens) - 10  # 10 token safety buffer
            
            if available_tokens_for_generation <= 0:
                # Truncate prompt if it's too long
                max_prompt_tokens = max_context_length - max_tokens - 10
                if max_prompt_tokens <= 0:
                    max_prompt_tokens = max_context_length // 2  # Use half for prompt, half for generation
                    max_tokens = max_context_length - max_prompt_tokens - 10
                
                # Truncate from the beginning to keep the most recent context
                truncated_tokens = prompt_tokens[-max_prompt_tokens:]
                prompt = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                available_tokens_for_generation = max_tokens
            
            # Use the smaller of requested max_tokens or available tokens
            actual_max_tokens = min(max_tokens, available_tokens_for_generation)
            
            if actual_max_tokens <= 0:
                raise LLMError("Prompt too long for GPT-2 model. Cannot generate any tokens.")
            
            # Generate response
            result = generator(
                prompt,
                max_new_tokens=actual_max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            # Extract generated text (remove the input prompt)
            generated_text = result[0]['generated_text']
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            raise LLMError(f"Local Hugging Face model call failed: {str(e)}")


class LangChainLLM(LLM):
    """
    A LangChain-compatible wrapper for the UnifiedLLMClient.
    This allows the custom client to be used seamlessly within LangChain chains and agents.
    """
    client: UnifiedLLMClient

    @property
    def _llm_type(self) -> str:
        return "custom_unified_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        The core method that LangChain calls. It translates the single string prompt
        into the message format expected by UnifiedLLMClient.
        """
     
        # We pass other kwargs (like model, temperature) to the client.
        messages = [{"role": "user", "content": prompt}]
        return self.client.call_llm(messages, **kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Returns a dictionary of identifying parameters for logging and caching.
        """
        llm_config = getattr(config, 'LLM_CONFIG', {})
        return {"provider": self.client.provider, "model": llm_config.get("model", "gemini-1.5-flash")}