import pytest
from modules.error_handler import validate_chunk, safe_regex_extract, ResponseValidator, LLMError, CircuitBreaker


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