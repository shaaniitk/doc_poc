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
    
    if not chunk['content'] or len(chunk['content'].strip()) < 5:
        return False
    
    return True

def safe_regex_extract(pattern: str, content: str, flags: int = 0) -> list:
    """Safe regex extraction with error handling"""
    try:
        import re
        return re.findall(pattern, content, flags)
    except Exception as e:
        raise ChunkingError(f"Regex extraction failed: {e}")