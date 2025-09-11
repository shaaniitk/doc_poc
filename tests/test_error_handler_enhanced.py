import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from modules.error_handler import (
    EmbeddingError, EmbeddingAPIError, EmbeddingModelError, EmbeddingFallbackError,
    robust_embedding_call, EmbeddingRateLimiter, EmbeddingFallbackManager
)


class TestEmbeddingErrorClasses:
    """Test suite for embedding-specific error classes."""
    
    def test_embedding_error_basic(self):
        """Test basic EmbeddingError functionality."""
        error = EmbeddingError("Test embedding error")
        assert str(error) == "Test embedding error"
        assert isinstance(error, Exception)
    
    def test_embedding_api_error(self):
        """Test EmbeddingAPIError with provider info."""
        error = EmbeddingAPIError("API failed", provider="openai", status_code=429)
        assert "API failed" in str(error)
        assert error.provider == "openai"
        assert error.status_code == 429
    
    def test_embedding_model_error(self):
        """Test EmbeddingModelError with model info."""
        error = EmbeddingModelError("Model loading failed", model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert "Model loading failed" in str(error)
        assert error.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    def test_embedding_fallback_error(self):
        """Test EmbeddingFallbackError with fallback info."""
        error = EmbeddingFallbackError("All fallbacks failed", attempted_providers=["openai", "cohere"])
        assert "All fallbacks failed" in str(error)
        assert error.attempted_providers == ["openai", "cohere"]


class TestRobustEmbeddingCallDecorator:
    """Test suite for robust_embedding_call decorator."""
    
    def test_successful_call_no_retries(self):
        """Test successful function call without retries."""
        @robust_embedding_call(max_retries=3)
        def successful_function(x):
            return x * 2
        
        result = successful_function(5)
        assert result == 10
    
    def test_retry_on_embedding_error(self):
        """Test retry mechanism on EmbeddingError."""
        call_count = 0
        
        @robust_embedding_call(max_retries=3, backoff_delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise EmbeddingAPIError("Temporary API error")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        @robust_embedding_call(max_retries=2, backoff_delay=0.1)
        def always_failing_function():
            raise EmbeddingAPIError("Persistent API error")
        
        with pytest.raises(EmbeddingAPIError):
            always_failing_function()
    
    def test_non_embedding_error_no_retry(self):
        """Test that non-embedding errors are not retried."""
        call_count = 0
        
        @robust_embedding_call(max_retries=3)
        def function_with_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not an embedding error")
        
        with pytest.raises(ValueError):
            function_with_value_error()
        
        assert call_count == 1  # Should not retry
    
    def test_backoff_timing(self):
        """Test exponential backoff timing."""
        call_times = []
        
        @robust_embedding_call(max_retries=3, backoff_delay=0.1)
        def timing_test_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise EmbeddingAPIError("Retry test")
            return "success"
        
        start_time = time.time()
        result = timing_test_function()
        
        assert result == "success"
        assert len(call_times) == 3
        
        # Check that delays increase exponentially
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            assert delay1 >= 0.1  # First backoff
        
        if len(call_times) >= 3:
            delay2 = call_times[2] - call_times[1]
            assert delay2 >= 0.2  # Second backoff (doubled)
    
    def test_decorator_with_arguments(self):
        """Test decorator works with function arguments."""
        @robust_embedding_call(max_retries=2, backoff_delay=0.1)
        def function_with_args(a, b, c=None):
            if c is None:
                raise EmbeddingAPIError("Missing argument")
            return a + b + c
        
        result = function_with_args(1, 2, c=3)
        assert result == 6
    
    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""
        @robust_embedding_call(max_retries=1)
        def documented_function():
            """This function has documentation."""
            return "result"
        
        assert documented_function.__doc__ == "This function has documentation."
        assert documented_function.__name__ == "documented_function"


class TestEmbeddingRateLimiter:
    """Test suite for EmbeddingRateLimiter class."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = EmbeddingRateLimiter(calls_per_minute=60, calls_per_second=2)
        assert limiter.calls_per_minute == 60
        assert limiter.calls_per_second == 2
        assert len(limiter.call_history) == 0
    
    def test_no_wait_when_under_limit(self):
        """Test no waiting when under rate limits."""
        limiter = EmbeddingRateLimiter(calls_per_minute=60, calls_per_second=10)
        
        start_time = time.time()
        limiter.wait_if_needed()
        end_time = time.time()
        
        # Should not wait significantly
        assert end_time - start_time < 0.1
    
    def test_wait_when_over_second_limit(self):
        """Test waiting when over per-second limit."""
        limiter = EmbeddingRateLimiter(calls_per_minute=60, calls_per_second=1)
        
        # Make first call
        limiter.wait_if_needed()
        
        # Second call should wait
        start_time = time.time()
        limiter.wait_if_needed()
        end_time = time.time()
        
        # Should wait at least 1 second
        assert end_time - start_time >= 0.9
    
    def test_call_history_cleanup(self):
        """Test that old calls are removed from history."""
        limiter = EmbeddingRateLimiter(calls_per_minute=60, calls_per_second=10)
        
        # Add old call to history
        old_time = time.time() - 70  # 70 seconds ago
        limiter.call_history.append(old_time)
        
        # Make new call
        limiter.wait_if_needed()
        
        # Old call should be removed
        assert all(call_time > time.time() - 60 for call_time in limiter.call_history)
    
    def test_multiple_rapid_calls(self):
        """Test behavior with multiple rapid calls."""
        limiter = EmbeddingRateLimiter(calls_per_minute=60, calls_per_second=2)
        
        start_time = time.time()
        
        # Make multiple calls
        for _ in range(3):
            limiter.wait_if_needed()
        
        end_time = time.time()
        
        # Should have waited to respect rate limits
        assert end_time - start_time >= 1.0  # At least 1 second for 3 calls at 2/sec


class TestEmbeddingFallbackManager:
    """Test suite for EmbeddingFallbackManager class."""
    
    def test_initialization(self):
        """Test fallback manager initialization."""
        manager = EmbeddingFallbackManager()
        assert len(manager.fallback_strategies) > 0
        assert manager.max_fallback_attempts > 0
    
    def test_try_fallback_with_strategy(self):
        """Test fallback attempt with available strategy."""
        manager = EmbeddingFallbackManager()
        
        # Mock a fallback strategy
        mock_strategy = Mock(return_value="fallback_result")
        manager.fallback_strategies['test_operation'] = [mock_strategy]
        
        result = manager.try_fallback('test_operation', arg1="value1", arg2="value2")
        
        assert result == "fallback_result"
        mock_strategy.assert_called_once_with(arg1="value1", arg2="value2")
    
    def test_try_fallback_no_strategy(self):
        """Test fallback attempt with no available strategy."""
        manager = EmbeddingFallbackManager()
        
        result = manager.try_fallback('unknown_operation', arg1="value1")
        
        assert result is None
    
    def test_try_fallback_strategy_fails(self):
        """Test fallback when strategy itself fails."""
        manager = EmbeddingFallbackManager()
        
        # Mock a failing fallback strategy
        mock_strategy = Mock(side_effect=Exception("Fallback failed"))
        manager.fallback_strategies['test_operation'] = [mock_strategy]
        
        result = manager.try_fallback('test_operation', arg1="value1")
        
        assert result is None
    
    def test_multiple_fallback_strategies(self):
        """Test multiple fallback strategies in sequence."""
        manager = EmbeddingFallbackManager()
        
        # First strategy fails, second succeeds
        failing_strategy = Mock(side_effect=Exception("First failed"))
        success_strategy = Mock(return_value="second_success")
        
        manager.fallback_strategies['test_operation'] = [failing_strategy, success_strategy]
        
        result = manager.try_fallback('test_operation', arg1="value1")
        
        assert result == "second_success"
        failing_strategy.assert_called_once()
        success_strategy.assert_called_once()
    
    def test_register_fallback_strategy(self):
        """Test registering new fallback strategy."""
        manager = EmbeddingFallbackManager()
        
        def custom_fallback(**kwargs):
            return "custom_result"
        
        manager.register_fallback('custom_operation', custom_fallback)
        
        assert 'custom_operation' in manager.fallback_strategies
        assert custom_fallback in manager.fallback_strategies['custom_operation']
        
        # Test the registered fallback
        result = manager.try_fallback('custom_operation', test_arg="value")
        assert result == "custom_result"
    
    def test_max_fallback_attempts(self):
        """Test that fallback attempts are limited."""
        manager = EmbeddingFallbackManager(max_fallback_attempts=2)
        
        # Create more strategies than max attempts
        strategies = [Mock(side_effect=Exception(f"Fail {i}")) for i in range(5)]
        manager.fallback_strategies['test_operation'] = strategies
        
        result = manager.try_fallback('test_operation')
        
        assert result is None
        # Should only try up to max_fallback_attempts
        assert sum(strategy.called for strategy in strategies) <= 2


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    def test_robust_call_with_rate_limiting(self):
        """Test robust call decorator with rate limiting."""
        rate_limiter = EmbeddingRateLimiter(calls_per_second=1)
        
        @robust_embedding_call(max_retries=2, backoff_delay=0.1)
        def rate_limited_function():
            rate_limiter.wait_if_needed()
            return "success"
        
        start_time = time.time()
        result = rate_limited_function()
        end_time = time.time()
        
        assert result == "success"
        # Should complete without excessive delay on first call
        assert end_time - start_time < 1.0
    
    def test_robust_call_with_fallback_manager(self):
        """Test robust call decorator with fallback manager."""
        fallback_manager = EmbeddingFallbackManager()
        
        # Register a fallback strategy
        def simple_fallback(**kwargs):
            return "fallback_success"
        
        fallback_manager.register_fallback('test_function', simple_fallback)
        
        call_count = 0
        
        @robust_embedding_call(max_retries=1, backoff_delay=0.1)
        def function_with_fallback():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two attempts
                raise EmbeddingAPIError("API Error")
            return "direct_success"
        
        # This would normally fail, but we can manually test fallback
        try:
            result = function_with_fallback()
        except EmbeddingAPIError:
            # Use fallback manager
            result = fallback_manager.try_fallback('test_function')
        
        assert result in ["direct_success", "fallback_success"]
    
    def test_complete_error_handling_workflow(self):
        """Test complete error handling workflow."""
        rate_limiter = EmbeddingRateLimiter(calls_per_second=10)
        fallback_manager = EmbeddingFallbackManager()
        
        # Register fallback
        fallback_manager.register_fallback('embedding_operation', 
                                         lambda texts: [[0.0] * 384] * len(texts))
        
        @robust_embedding_call(max_retries=2, backoff_delay=0.1)
        def embedding_operation(texts):
            rate_limiter.wait_if_needed()
            # Simulate API failure
            raise EmbeddingAPIError("Service unavailable")
        
        texts = ["test text 1", "test text 2"]
        
        try:
            result = embedding_operation(texts)
        except EmbeddingAPIError:
            # Use fallback
            result = fallback_manager.try_fallback('embedding_operation', texts=texts)
        
        assert result is not None
        assert len(result) == len(texts)
        assert all(isinstance(embedding, list) for embedding in result)
    
    @pytest.mark.parametrize("error_type,should_retry", [
        (EmbeddingAPIError("API Error"), True),
        (EmbeddingModelError("Model Error"), True),
        (EmbeddingFallbackError("Fallback Error"), True),
        (ValueError("Value Error"), False),
        (ConnectionError("Connection Error"), False),
    ])
    def test_error_type_retry_behavior(self, error_type, should_retry):
        """Test that different error types are handled appropriately."""
        call_count = 0
        
        @robust_embedding_call(max_retries=2, backoff_delay=0.1)
        def error_test_function():
            nonlocal call_count
            call_count += 1
            raise error_type
        
        with pytest.raises(type(error_type)):
            error_test_function()
        
        if should_retry:
            assert call_count > 1  # Should have retried
        else:
            assert call_count == 1  # Should not have retried