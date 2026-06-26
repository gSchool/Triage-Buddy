import pytest

from triage_buddy.adapters.llm._retry import call_with_retries
from triage_buddy.ports.llm import LLMError


def test_returns_immediately_on_success():
    sleeps = []
    calls = []

    def op():
        calls.append(1)
        return "ok"

    assert call_with_retries(op, sleep=sleeps.append) == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_retries_then_succeeds():
    sleeps = []
    state = {"n": 0}

    def op():
        state["n"] += 1
        if state["n"] < 3:
            raise LLMError("transient")
        return "ok"

    result = call_with_retries(op, attempts=3, base_delay=0.5, sleep=sleeps.append)
    assert result == "ok"
    assert state["n"] == 3
    assert sleeps == [0.5, 1.0]  # exponential backoff between the 3 attempts


def test_exhausts_and_reraises_last_error():
    sleeps = []
    state = {"n": 0}

    def op():
        state["n"] += 1
        raise LLMError(f"fail {state['n']}")

    with pytest.raises(LLMError, match="fail 3"):
        call_with_retries(op, attempts=3, base_delay=0.1, sleep=sleeps.append)
    assert state["n"] == 3
    assert len(sleeps) == 2  # slept between attempts, not after the last


def test_backoff_is_capped():
    sleeps = []

    def op():
        raise LLMError("x")

    with pytest.raises(LLMError):
        call_with_retries(op, attempts=5, base_delay=1.0, max_delay=2.0, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0, 2.0, 2.0]


def test_non_llm_error_is_not_retried():
    state = {"n": 0}

    def op():
        state["n"] += 1
        raise ValueError("not an LLMError")

    with pytest.raises(ValueError):
        call_with_retries(op, attempts=3, sleep=lambda _d: None)
    assert state["n"] == 1  # propagated on the first attempt


def test_attempts_must_be_positive():
    with pytest.raises(ValueError):
        call_with_retries(lambda: "x", attempts=0)
