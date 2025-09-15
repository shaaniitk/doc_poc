"""Enhanced LLM Handler for LangGraph Integration

This module provides a modern, async-capable LLM handler that integrates
seamlessly with the LangGraph orchestrator. It supports multiple LLM providers,
intelligent retry mechanisms, and comprehensive error handling.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timezone
import time
import hashlib
from pathlib import Path

# HTTP and API imports
try:
    import aiohttp
    import httpx
    HAS_HTTP_CLIENTS = True
except ImportError:
    HAS_HTTP_CLIENTS = False
    logging.warning("HTTP clients not available. Install aiohttp and httpx for API support.")

# OpenAI imports
try:
    import openai
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("OpenAI not available. Install openai for GPT support.")

# Anthropic imports
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logging.warning("Anthropic not available. Install anthropic for Claude support.")

from .chunking_processor import DocumentChunk, ChunkingResult

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = auto()
    ANTHROPIC = auto()
    AZURE_OPENAI = auto()
    HUGGING_FACE = auto()
    OLLAMA = auto()
    CUSTOM = auto()


class ProcessingMode(Enum):
    """Different processing modes for LLM operations."""
    SEQUENTIAL = auto()    # Process chunks one by one
    PARALLEL = auto()      # Process chunks in parallel
    BATCH = auto()         # Process chunks in batches
    STREAMING = auto()     # Stream processing results
    ADAPTIVE = auto()      # Adapt based on content and load


class RetryStrategy(Enum):
    """Retry strategies for failed requests."""
    EXPONENTIAL_BACKOFF = auto()
    LINEAR_BACKOFF = auto()
    FIXED_DELAY = auto()
    ADAPTIVE = auto()
    NONE = auto()


@dataclass
class LLMConfig:
    """Configuration for LLM operations."""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retry_delay: float = 1.0
    processing_mode: ProcessingMode = ProcessingMode.PARALLEL
    batch_size: int = 5
    max_concurrent: int = 10
    enable_caching: bool = True
    cache_ttl: int = 3600  # Cache TTL in seconds
    custom_headers: Dict[str, str] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingRequest:
    """Request for LLM processing."""
    chunk: DocumentChunk
    prompt_template: str
    system_prompt: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher numbers = higher priority
    timeout_override: Optional[int] = None
    retry_override: Optional[int] = None


@dataclass
class ProcessingResult:
    """Result of LLM processing."""
    chunk_id: str
    original_content: str
    processed_content: str
    prompt_used: str
    model_used: str
    provider: LLMProvider
    processing_time: float
    token_usage: Dict[str, int]
    confidence_score: Optional[float] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    retry_count: int = 0
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class BatchProcessingResult:
    """Result of batch processing operation."""
    results: List[ProcessingResult]
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    total_processing_time: float
    average_processing_time: float
    total_tokens_used: Dict[str, int]
    error_summary: Dict[str, int]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMHandler:
    """Enhanced LLM handler with multi-provider support and advanced features."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._clients = {}
        self._cache = {}
        self._rate_limiter = None
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0
        }
        
        # Initialize clients
        self._initialize_clients()
        
        logger.info(f"LLMHandler initialized with provider: {self.config.provider.name}")
    
    def _initialize_clients(self) -> None:
        """Initialize LLM clients based on configuration."""
        try:
            if self.config.provider == LLMProvider.OPENAI and HAS_OPENAI:
                self._clients[LLMProvider.OPENAI] = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    timeout=self.config.timeout
                )
                logger.debug("OpenAI client initialized")
            
            if self.config.provider == LLMProvider.ANTHROPIC and HAS_ANTHROPIC:
                self._clients[LLMProvider.ANTHROPIC] = anthropic.AsyncAnthropic(
                    api_key=self.config.api_key
                )
                logger.debug("Anthropic client initialized")
            
            # Add other providers as needed
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {e}")
            raise LLMHandlerError(f"Client initialization failed: {e}") from e
    
    async def process_chunks_async(self, 
                                 chunks: List[DocumentChunk],
                                 prompt_template: str,
                                 system_prompt: Optional[str] = None,
                                 context: Optional[Dict[str, Any]] = None) -> BatchProcessingResult:
        """Process multiple chunks asynchronously."""
        start_time = time.time()
        
        # Create processing requests
        requests = [
            ProcessingRequest(
                chunk=chunk,
                prompt_template=prompt_template,
                system_prompt=system_prompt,
                context=context or {}
            )
            for chunk in chunks
        ]
        
        # Process based on mode
        if self.config.processing_mode == ProcessingMode.SEQUENTIAL:
            results = await self._process_sequential(requests)
        elif self.config.processing_mode == ProcessingMode.PARALLEL:
            results = await self._process_parallel(requests)
        elif self.config.processing_mode == ProcessingMode.BATCH:
            results = await self._process_batch(requests)
        elif self.config.processing_mode == ProcessingMode.STREAMING:
            results = await self._process_streaming(requests)
        else:  # ADAPTIVE
            results = await self._process_adaptive(requests)
        
        # Calculate metrics
        total_time = time.time() - start_time
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        
        # Aggregate token usage
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for result in successful:
            for key in total_tokens:
                total_tokens[key] += result.token_usage.get(key, 0)
        
        # Error summary
        error_summary = {}
        for result in failed:
            error_type = type(result.error).__name__ if result.error else "Unknown"
            error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        # Quality metrics
        quality_metrics = self._calculate_batch_quality_metrics(successful)
        
        return BatchProcessingResult(
            results=results,
            total_chunks=len(chunks),
            successful_chunks=len(successful),
            failed_chunks=len(failed),
            total_processing_time=total_time,
            average_processing_time=total_time / len(chunks) if chunks else 0,
            total_tokens_used=total_tokens,
            error_summary=error_summary,
            quality_metrics=quality_metrics
        )
    
    async def process_single_chunk_async(self, 
                                       chunk: DocumentChunk,
                                       prompt_template: str,
                                       system_prompt: Optional[str] = None,
                                       context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process a single chunk asynchronously."""
        request = ProcessingRequest(
            chunk=chunk,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            context=context or {}
        )
        
        return await self._process_single_request(request)
    
    async def _process_sequential(self, requests: List[ProcessingRequest]) -> List[ProcessingResult]:
        """Process requests sequentially."""
        results = []
        
        for request in requests:
            try:
                result = await self._process_single_request(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Sequential processing failed for chunk {request.chunk.metadata.chunk_id}: {e}")
                results.append(self._create_error_result(request, str(e)))
        
        return results
    
    async def _process_parallel(self, requests: List[ProcessingRequest]) -> List[ProcessingResult]:
        """Process requests in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(request: ProcessingRequest) -> ProcessingResult:
            async with semaphore:
                try:
                    return await self._process_single_request(request)
                except Exception as e:
                    logger.error(f"Parallel processing failed for chunk {request.chunk.metadata.chunk_id}: {e}")
                    return self._create_error_result(request, str(e))
        
        tasks = [process_with_semaphore(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def _process_batch(self, requests: List[ProcessingRequest]) -> List[ProcessingResult]:
        """Process requests in batches."""
        results = []
        
        for i in range(0, len(requests), self.config.batch_size):
            batch = requests[i:i + self.config.batch_size]
            batch_results = await self._process_parallel(batch)
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limiting
            if i + self.config.batch_size < len(requests):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _process_streaming(self, requests: List[ProcessingRequest]) -> List[ProcessingResult]:
        """Process requests with streaming results."""
        # For now, fall back to parallel processing
        # Could be enhanced to support actual streaming APIs
        logger.info("Streaming mode not fully implemented, using parallel processing")
        return await self._process_parallel(requests)
    
    async def _process_adaptive(self, requests: List[ProcessingRequest]) -> List[ProcessingResult]:
        """Adaptively choose processing strategy based on load and content."""
        # Simple adaptive logic - can be enhanced
        if len(requests) <= 5:
            return await self._process_sequential(requests)
        elif len(requests) <= 20:
            return await self._process_parallel(requests)
        else:
            return await self._process_batch(requests)
    
    async def _process_single_request(self, request: ProcessingRequest) -> ProcessingResult:
        """Process a single request with retry logic."""
        start_time = time.time()
        
        # Check cache first
        if self.config.enable_caching:
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                cached_result.cache_hit = True
                self._metrics["cache_hits"] += 1
                return cached_result
        
        # Prepare prompt
        formatted_prompt = self._format_prompt(request)
        
        # Process with retries
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Make API call
                response = await self._make_api_call(
                    formatted_prompt,
                    request.system_prompt,
                    request.timeout_override or self.config.timeout
                )
                
                # Create result
                processing_time = time.time() - start_time
                result = ProcessingResult(
                    chunk_id=request.chunk.metadata.chunk_id,
                    original_content=request.chunk.content,
                    processed_content=response["content"],
                    prompt_used=formatted_prompt,
                    model_used=self.config.model,
                    provider=self.config.provider,
                    processing_time=processing_time,
                    token_usage=response.get("usage", {}),
                    retry_count=attempt,
                    metadata=request.metadata
                )
                
                # Calculate quality metrics
                result.quality_metrics = self._calculate_quality_metrics(result)
                
                # Cache result
                if self.config.enable_caching:
                    self._cache_result(cache_key, result)
                
                # Update metrics
                self._metrics["successful_requests"] += 1
                self._metrics["total_tokens"] += response.get("usage", {}).get("total_tokens", 0)
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for chunk {request.chunk.metadata.chunk_id}: {e}")
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    await asyncio.sleep(delay)
        
        # All retries failed
        self._metrics["failed_requests"] += 1
        return self._create_error_result(request, str(last_error))
    
    async def _make_api_call(self, prompt: str, system_prompt: Optional[str], timeout: int) -> Dict[str, Any]:
        """Make API call to the configured LLM provider."""
        self._metrics["total_requests"] += 1
        
        if self.config.provider == LLMProvider.OPENAI:
            return await self._call_openai(prompt, system_prompt, timeout)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(prompt, system_prompt, timeout)
        else:
            raise LLMHandlerError(f"Provider {self.config.provider.name} not implemented")
    
    async def _call_openai(self, prompt: str, system_prompt: Optional[str], timeout: int) -> Dict[str, Any]:
        """Make API call to OpenAI."""
        if not HAS_OPENAI or LLMProvider.OPENAI not in self._clients:
            raise LLMHandlerError("OpenAI client not available")
        
        client = self._clients[LLMProvider.OPENAI]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                timeout=timeout
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else {}
            }
            
        except Exception as e:
            raise LLMHandlerError(f"OpenAI API call failed: {e}") from e
    
    async def _call_anthropic(self, prompt: str, system_prompt: Optional[str], timeout: int) -> Dict[str, Any]:
        """Make API call to Anthropic."""
        if not HAS_ANTHROPIC or LLMProvider.ANTHROPIC not in self._clients:
            raise LLMHandlerError("Anthropic client not available")
        
        client = self._clients[LLMProvider.ANTHROPIC]
        
        try:
            # Anthropic uses a different message format
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
            
            response = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens or 1000,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": full_prompt}],
                timeout=timeout
            )
            
            return {
                "content": response.content[0].text,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                } if hasattr(response, 'usage') else {}
            }
            
        except Exception as e:
            raise LLMHandlerError(f"Anthropic API call failed: {e}") from e
    
    def _format_prompt(self, request: ProcessingRequest) -> str:
        """Format prompt template with chunk content and context."""
        template_vars = {
            "content": request.chunk.content,
            "chunk_id": request.chunk.metadata.chunk_id,
            "chunk_index": request.chunk.metadata.index,
            "word_count": request.chunk.metadata.word_count,
            "character_count": request.chunk.metadata.character_count,
            **request.context
        }
        
        try:
            return request.prompt_template.format(**template_vars)
        except KeyError as e:
            raise LLMHandlerError(f"Missing template variable: {e}") from e
    
    def _generate_cache_key(self, request: ProcessingRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            "content_hash": hashlib.sha256(request.chunk.content.encode()).hexdigest()[:16],
            "prompt_hash": hashlib.sha256(request.prompt_template.encode()).hexdigest()[:16],
            "system_prompt": request.system_prompt,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "context": json.dumps(request.context, sort_keys=True)
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[ProcessingResult]:
        """Get result from cache if not expired."""
        if cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if time.time() - cached_item["timestamp"] < self.config.cache_ttl:
                return cached_item["result"]
            else:
                # Remove expired item
                del self._cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: ProcessingResult) -> None:
        """Cache processing result."""
        self._cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # Simple cache cleanup - remove oldest items if cache is too large
        if len(self._cache) > 1000:
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return self.config.retry_delay * (2 ** attempt)
        elif self.config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return self.config.retry_delay * (attempt + 1)
        elif self.config.retry_strategy == RetryStrategy.FIXED_DELAY:
            return self.config.retry_delay
        elif self.config.retry_strategy == RetryStrategy.ADAPTIVE:
            # Simple adaptive strategy
            base_delay = self.config.retry_delay
            if attempt == 0:
                return base_delay
            elif attempt == 1:
                return base_delay * 2
            else:
                return base_delay * 4
        else:
            return 0
    
    def _create_error_result(self, request: ProcessingRequest, error_message: str) -> ProcessingResult:
        """Create error result for failed processing."""
        return ProcessingResult(
            chunk_id=request.chunk.metadata.chunk_id,
            original_content=request.chunk.content,
            processed_content="",
            prompt_used=request.prompt_template,
            model_used=self.config.model,
            provider=self.config.provider,
            processing_time=0.0,
            token_usage={},
            error=error_message,
            metadata=request.metadata
        )
    
    def _calculate_quality_metrics(self, result: ProcessingResult) -> Dict[str, float]:
        """Calculate quality metrics for processing result."""
        metrics = {}
        
        # Length ratio
        original_len = len(result.original_content)
        processed_len = len(result.processed_content)
        if original_len > 0:
            metrics["length_ratio"] = processed_len / original_len
        
        # Processing efficiency (tokens per second)
        if result.processing_time > 0:
            total_tokens = result.token_usage.get("total_tokens", 0)
            metrics["tokens_per_second"] = total_tokens / result.processing_time
        
        # Content similarity (simple heuristic)
        common_words = set(result.original_content.lower().split()) & set(result.processed_content.lower().split())
        total_words = set(result.original_content.lower().split()) | set(result.processed_content.lower().split())
        if total_words:
            metrics["content_similarity"] = len(common_words) / len(total_words)
        
        return metrics
    
    def _calculate_batch_quality_metrics(self, results: List[ProcessingResult]) -> Dict[str, float]:
        """Calculate quality metrics for batch of results."""
        if not results:
            return {}
        
        # Aggregate individual metrics
        aggregated = {}
        for result in results:
            for metric, value in result.quality_metrics.items():
                if metric not in aggregated:
                    aggregated[metric] = []
                aggregated[metric].append(value)
        
        # Calculate averages
        batch_metrics = {}
        for metric, values in aggregated.items():
            batch_metrics[f"avg_{metric}"] = sum(values) / len(values)
            batch_metrics[f"min_{metric}"] = min(values)
            batch_metrics[f"max_{metric}"] = max(values)
        
        # Overall success rate
        batch_metrics["success_rate"] = len(results) / len(results) if results else 0
        
        return batch_metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current handler metrics."""
        return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset handler metrics."""
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0
        }
    
    def clear_cache(self) -> None:
        """Clear the processing cache."""
        self._cache.clear()
        logger.info("LLM handler cache cleared")


class LLMHandlerError(Exception):
    """Custom exception for LLM handler errors."""
    pass


# Factory functions
def create_llm_handler(provider: LLMProvider = LLMProvider.OPENAI,
                      model: str = "gpt-3.5-turbo",
                      api_key: Optional[str] = None,
                      **kwargs) -> LLMHandler:
    """Create an LLM handler with specified configuration."""
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )
    return LLMHandler(config)


async def process_chunks_with_llm(chunks: List[DocumentChunk],
                                prompt_template: str,
                                provider: LLMProvider = LLMProvider.OPENAI,
                                model: str = "gpt-3.5-turbo",
                                **kwargs) -> BatchProcessingResult:
    """Convenience function for processing chunks with LLM."""
    handler = create_llm_handler(provider, model, **kwargs)
    return await handler.process_chunks_async(chunks, prompt_template)