import pytest
from modules.error_handler import validate_chunk, safe_regex_extract, ResponseValidator, LLMError, CircuitBreaker
from langgraph_state import ErrorInfo, ErrorSeverity, PipelineState, ProcessingStage
from langgraph_error_recovery import ErrorRecoveryManager, RecoveryStrategy, ErrorCategory


def test_validate_chunk_missing_fields():
    assert validate_chunk({}) is False
    assert validate_chunk({"type": "text", "content": "abc"}) is False
    assert validate_chunk({"type": "text", "content": "abc", "parent_section": "Intro"}) is True


def test_safe_regex_extract_basic():
    content = "abc 123 def 456"
    matches = safe_regex_extract(r"\d+", content)
    assert matches == ["123", "456"]


def test_response_validator_rejects_error_prefix_and_short():
    with pytest.raises(LLMError):
        ResponseValidator.validate_llm_response("% Error: something went wrong")
    with pytest.raises(LLMError):
        ResponseValidator.validate_llm_response("  ")


def test_circuit_breaker_state_transitions():
    cb = CircuitBreaker(failure_threshold=2, timeout=5)

    assert cb.allow_request() is True
    cb.record_failure()
    assert cb.allow_request() is True  # Below threshold
    cb.record_failure()
    assert cb.allow_request() is False  # Now OPEN

    # Success should close it when allowed (simulate timeout by forcing success path)
    # We can't sleep in unit tests; directly call record_success to close
    cb.record_success()
    assert cb.allow_request() is True


def test_error_info_integration():
    """Test integration with LangGraph ErrorInfo state."""
    error_info = ErrorInfo(
        error_type="ValidationError",
        message="Chunk validation failed",
        severity=ErrorSeverity.HIGH,
        stage=ProcessingStage.CHUNKING,
        recoverable=True
    )
    
    assert error_info.error_type == "ValidationError"
    assert error_info.severity == ErrorSeverity.HIGH
    assert error_info.recoverable is True


def test_error_recovery_manager_integration():
    """Test integration with ErrorRecoveryManager."""
    recovery_manager = ErrorRecoveryManager()
    
    # Test error categorization
    error_info = ErrorInfo(
        error_type="LLMError",
        message="API timeout",
        severity=ErrorSeverity.MEDIUM,
        stage=ProcessingStage.LLM_PROCESSING
    )
    
    category = recovery_manager.categorize_error(error_info)
    assert category in [ErrorCategory.TRANSIENT, ErrorCategory.CONFIGURATION, ErrorCategory.RESOURCE]
    
    # Test recovery strategy selection
    strategy = recovery_manager.get_recovery_strategy(error_info)
    assert strategy in [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK, RecoveryStrategy.SKIP]


def test_pipeline_state_error_tracking():
    """Test error tracking in PipelineState."""
    state = PipelineState(
        session_id="test-session",
        current_stage=ProcessingStage.CHUNKING
    )
    
    # Add error to state
    error_info = ErrorInfo(
        error_type="ChunkingError",
        message="Failed to chunk document",
        severity=ErrorSeverity.HIGH,
        stage=ProcessingStage.CHUNKING
    )
    
    state.errors.append(error_info)
    
    assert len(state.errors) == 1
    assert state.errors[0].error_type == "ChunkingError"
    assert state.has_critical_errors() is True