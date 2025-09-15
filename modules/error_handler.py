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

class EmbeddingError(ProcessingError):
    """Embedding-specific errors"""
    pass

class EmbeddingAPIError(EmbeddingError):
    """API-related embedding errors (rate limits, timeouts, etc.)"""
    def __init__(self, message="Embedding API error", **kwargs):
        super().__init__(message)
        self.__dict__.update(kwargs)

class EmbeddingModelError(EmbeddingError):
    """Model loading or inference errors"""
    def __init__(self, message="Embedding model error", **kwargs):
        super().__init__(message)
        self.__dict__.update(kwargs)

class EmbeddingFallbackError(EmbeddingError):
    """All embedding providers failed"""
    def __init__(self, message="All embedding fallback strategies failed", **kwargs):
        super().__init__(message)
        self.__dict__.update(kwargs)

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


def robust_embedding_call(max_retries: int = 3, backoff_delay: float = 1.0, 
                         fallback_providers: list = None):
    """Decorator for robust embedding calls with retries, rate limiting, and fallback providers"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            providers_to_try = [kwargs.get('provider')] if 'provider' in kwargs else ['sentence_transformer']
            
            # Add fallback providers if specified
            if fallback_providers:
                providers_to_try.extend([p for p in fallback_providers if p not in providers_to_try])
            
            for provider in providers_to_try:
                # Only set provider if the function accepts it
                import inspect
                sig = inspect.signature(func)
                if provider and 'provider' in sig.parameters:
                    kwargs['provider'] = provider
                
                for attempt in range(max_retries + 1):
                    try:
                        result = func(*args, **kwargs)
                        if result is None or (hasattr(result, '__len__') and len(result) == 0):
                            raise EmbeddingError(f"Embedding function returned empty result in {func.__name__}")
                        return result
                        
                    except (EmbeddingAPIError, EmbeddingModelError, EmbeddingFallbackError) as e:
                        last_error = e
                        if attempt < max_retries:
                            # Exponential backoff with jitter
                            import random
                            jitter = random.uniform(0.1, 0.5)
                            sleep_time = (backoff_delay * (2 ** attempt)) + jitter
                            time.sleep(sleep_time)
                            continue
                        break
                    except (ConnectionError, TimeoutError) as e:
                        # Network errors - don't retry, re-raise immediately
                        raise e
                    except Exception as e:
                        # Re-raise non-embedding errors immediately without retry
                        if not isinstance(e, (EmbeddingError, EmbeddingAPIError, EmbeddingModelError, EmbeddingFallbackError)):
                            raise e
                        
                        last_error = e
                        if attempt < max_retries:
                            time.sleep(backoff_delay * (2 ** attempt))
                            continue
                        break
            
            # All providers and retries failed
            if last_error:
                # Preserve the original error type
                raise last_error
            else:
                raise EmbeddingFallbackError(
                    f"All embedding providers failed after {max_retries + 1} attempts in {func.__name__}"
                )
        
        return wrapper
    return decorator


class ErrorHandler:
    """Main error handler class for the document processing pipeline"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.circuit_breaker = CircuitBreaker()
        self.validator = ResponseValidator()
    
    def handle_processing_error(self, error: Exception, context: str = "") -> None:
        """Handle processing errors with appropriate logging and recovery"""
        if isinstance(error, LLMError):
            self.circuit_breaker.record_failure()
            raise error  # LLM errors should halt processing
        elif isinstance(error, (EmbeddingError, ChunkingError)):
            # Log but allow processing to continue with fallbacks
            print(f"Warning: {context} - {str(error)}")
        else:
            # Unknown error - log and re-raise
            print(f"Error in {context}: {str(error)}")
            raise error
    
    def validate_response(self, response: str, min_length: int = 10) -> str:
        """Validate LLM response using the response validator"""
        return self.validator.validate_llm_response(response, min_length)
    
    def check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests"""
        return self.circuit_breaker.allow_request()
    
    def record_success(self) -> None:
        """Record successful operation"""
        self.circuit_breaker.record_success()


class EmbeddingRateLimiter:
    """Rate limiter for embedding API calls to prevent hitting provider limits"""
    
    def __init__(self, calls_per_minute: int = 60, calls_per_second: int = 10):
        self.calls_per_minute = calls_per_minute
        self.calls_per_second = calls_per_second
        self.minute_calls = []
        self.second_calls = []
        self.call_history = []  # For test compatibility
        self.last_cleanup = time.time()
    
    def can_make_call(self) -> bool:
        """Check if a call can be made without exceeding rate limits"""
        now = time.time()
        self._cleanup_old_calls(now)
        
        # Check per-second limit
        recent_second_calls = [t for t in self.second_calls if now - t < 1.0]
        if len(recent_second_calls) >= self.calls_per_second:
            return False
        
        # Check per-minute limit
        recent_minute_calls = [t for t in self.minute_calls if now - t < 60.0]
        if len(recent_minute_calls) >= self.calls_per_minute:
            return False
        
        return True
    
    def record_call(self) -> None:
        """Record that a call was made"""
        now = time.time()
        self.second_calls.append(now)
        self.minute_calls.append(now)
        self.call_history.append(now)  # For test compatibility
        self._cleanup_old_calls(now)
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits"""
        now = time.time()
        self._cleanup_old_calls(now)
        
        # Check per-second limit and wait if needed
        recent_second_calls = [t for t in self.second_calls if now - t < 1.0]
        if len(recent_second_calls) >= self.calls_per_second:
            wait_time = 1.0 - (now - min(recent_second_calls))
            if wait_time > 0:
                time.sleep(wait_time)
        
        # Check per-minute limit and wait if needed
        recent_minute_calls = [t for t in self.minute_calls if now - t < 60.0]
        if len(recent_minute_calls) >= self.calls_per_minute:
            wait_time = 60.0 - (now - min(recent_minute_calls))
            if wait_time > 0:
                time.sleep(wait_time)
        
        # Record the call
        self.record_call()
    
    def _cleanup_old_calls(self, now: float) -> None:
         """Remove old calls from tracking lists"""
         # Clean up calls older than 1 second
         self.second_calls = [t for t in self.second_calls if now - t < 1.0]
         # Clean up calls older than 1 minute
         self.minute_calls = [t for t in self.minute_calls if now - t < 60.0]
         # Clean up call history older than 1 minute
         self.call_history = [t for t in self.call_history if now - t < 60.0]
         self.last_cleanup = now


class EmbeddingFallbackManager:
    """Manages fallback strategies for embedding operations"""
    
    def __init__(self, primary_provider: str = 'sentence_transformer', 
                 fallback_providers: list = None, max_fallback_attempts: int = 3):
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or ['sentence_transformer']
        self.max_fallback_attempts = max_fallback_attempts
        self.provider_health = {}
        self.circuit_breakers = {}
        self.fallback_strategies = {
            'default': [lambda **kwargs: None]  # Default fallback strategy
        }
        
        # Initialize circuit breakers for each provider
        for provider in [primary_provider] + self.fallback_providers:
            if provider not in self.circuit_breakers:
                self.circuit_breakers[provider] = CircuitBreaker(
                    failure_threshold=3, timeout=300.0  # 5 minutes
                )
    
    def get_available_provider(self) -> str:
        """Get the next available provider based on circuit breaker states"""
        # Try primary provider first
        if self.circuit_breakers[self.primary_provider].allow_request():
            return self.primary_provider
        
        # Try fallback providers
        for provider in self.fallback_providers:
            if provider != self.primary_provider:
                if provider not in self.circuit_breakers:
                    self.circuit_breakers[provider] = CircuitBreaker()
                
                if self.circuit_breakers[provider].allow_request():
                    return provider
        
        # If all providers are down, return primary (let it fail gracefully)
        return self.primary_provider
    
    def record_success(self, provider: str) -> None:
        """Record successful operation for a provider"""
        if provider in self.circuit_breakers:
            self.circuit_breakers[provider].record_success()
    
    def record_failure(self, provider: str) -> None:
        """Record failed operation for a provider"""
        if provider in self.circuit_breakers:
            self.circuit_breakers[provider].record_failure()
    
    def get_provider_status(self) -> dict:
        """Get status of all providers"""
        status = {}
        for provider, breaker in self.circuit_breakers.items():
            status[provider] = {
                'state': breaker.state,
                'failure_count': breaker.failure_count,
                'available': breaker.allow_request()
            }
        return status
    
    def register_fallback(self, provider: str, strategy: Callable = None) -> None:
        """Register a new fallback provider with optional strategy"""
        if provider not in self.fallback_providers:
            self.fallback_providers.append(provider)
        
        # Store the strategy if provided
        if strategy is not None:
            if provider not in self.fallback_strategies:
                self.fallback_strategies[provider] = []
            self.fallback_strategies[provider].append(strategy)
        
        # Initialize circuit breaker for new provider
        if provider not in self.circuit_breakers:
            self.circuit_breakers[provider] = CircuitBreaker(
                failure_threshold=3, timeout=300.0
            )
    
    def try_fallback(self, operation: str, **kwargs) -> Any:
        """Try fallback strategies for the given operation"""
        if operation not in self.fallback_strategies:
            return None
        
        strategies = self.fallback_strategies[operation]
        attempts = 0
        
        for strategy in strategies:
            if attempts >= self.max_fallback_attempts:
                break
            
            try:
                attempts += 1
                return strategy(**kwargs)
            except Exception:
                # Strategy failed, try next one
                continue
        
        return None