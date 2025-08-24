"""Robust error handling for document processing"""
import functools
import time
from typing import Any, Callable, Optional

class ProcessingError(Exception):
    """Base exception for processing errors"""
    pass

class LLMError(ProcessingError):
    """LLM-specific errors that should halt processing"""
    pass

class ChunkingError(ProcessingError):
    """Chunking-specific errors"""
    pass

def robust_llm_call(max_retries: int = 2, backoff_delay: float = 1.0):
    """Decorator for robust LLM calls with retries and error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if not result or len(str(result).strip()) < 10:
                        raise LLMError(f"LLM returned empty/invalid response in {func.__name__}")
                    return result
                    
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        time.sleep(backoff_delay * (2 ** attempt))
                        continue
                    break
            
            # All retries failed - raise LLMError to halt processing
            raise LLMError(f"LLM call failed after {max_retries + 1} attempts in {func.__name__}: {last_error}")
        
        return wrapper
    return decorator

def validate_chunk(chunk: dict) -> bool:
    """Validate chunk structure and content"""
    required_fields = ['type', 'content', 'parent_section']
    
    if not isinstance(chunk, dict):
        return False
    
    for field in required_fields:
        if field not in chunk:
            return False
    
    # Accept any non-empty content; do not enforce a minimum length here to keep validation lightweight
    if not chunk['content'] or len(str(chunk['content']).strip()) == 0:
        return False
    
    return True

def safe_regex_extract(pattern: str, content: str, flags: int = 0) -> list:
    """Safe regex extraction with error handling"""
    try:
        import re
        return re.findall(pattern, content, flags)
    except Exception as e:
        raise ChunkingError(f"Regex extraction failed: {e}")


class ResponseValidator:
    """Validates LLM responses to prevent silent failures."""
    @staticmethod
    def validate_llm_response(response: str, min_length: int = 10) -> str:
        if response is None:
            raise LLMError("LLM returned None response")
        text = str(response).strip()
        if len(text) < min_length:
            raise LLMError("Response too short or empty")
        # Guard against error-prefixed strings returned by providers
        lowered = text.lower()
        if lowered.startswith("% error") or lowered.startswith("% api error") or lowered.startswith("error:"):
            raise LLMError(f"LLM returned error: {text}")
        return text


class CircuitBreaker:
    """Simple circuit breaker to protect LLM providers from cascading failures.
    States: CLOSED -> OPEN -> HALF_OPEN (implicit via allow_request check).
    """
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN
        self.last_failure_time: Optional[float] = None

    def allow_request(self) -> bool:
        if self.state == "OPEN":
            now = time.time()
            if self.last_failure_time is None:
                # Safety: treat as open until timeout elapses once we set it
                self.last_failure_time = now
                return False
            if (now - self.last_failure_time) >= self.timeout:
                # Transition to HALF_OPEN implicitly by allowing a trial request
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        if self.state != "CLOSED":
            self.state = "CLOSED"
            self.last_failure_time = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_failure_time = time.time()