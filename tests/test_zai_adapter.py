"""Unit tests for the Z.ai adapter, using a fake client (no network, no key)."""

import json
from types import SimpleNamespace

import pytest

from triage_buddy.adapters.llm.zai import DEFAULT_MODEL, ZaiProvider
from triage_buddy.ports.llm import LLMError, LLMRequest, RateLimitError


class FakeCompletions:
    def __init__(self, *, content=None, error=None, fail_times=0):
        self._content = content
        self._error = error
        self._fail_times = fail_times
        self.last_kwargs = None
        self.calls = 0

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self.calls <= self._fail_times:
            raise RuntimeError("transient boom")
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=FakeCompletions(**kwargs))


def _request():
    return LLMRequest(system="sys", user="usr")


def test_generate_sends_messages_and_returns_text(monkeypatch):
    monkeypatch.delenv("ZAI_MODEL", raising=False)
    payload = json.dumps({"urgency": "medium", "recommendation": "a", "disclaimer": "d"})
    provider = ZaiProvider(client=FakeClient(content=payload))
    response = provider.generate(_request())
    assert response.text == payload
    kwargs = provider._client.chat.completions.last_kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert kwargs["temperature"] == 0.0
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["thinking"] == {"type": "disabled"}


def test_default_model_used_when_no_env(monkeypatch):
    # No explicit arg and no ZAI_MODEL -> the hardcoded DEFAULT_MODEL fallback.
    monkeypatch.delenv("ZAI_MODEL", raising=False)
    provider = ZaiProvider(client=FakeClient(content="{}"))
    assert provider._model == DEFAULT_MODEL


def test_env_var_overrides_default_model(monkeypatch):
    monkeypatch.setenv("ZAI_MODEL", "glm-4.7-air")
    provider = ZaiProvider(client=FakeClient(content="{}"))
    assert provider._model == "glm-4.7-air"


def test_explicit_model_arg_beats_env_var(monkeypatch):
    monkeypatch.setenv("ZAI_MODEL", "glm-4.7-air")
    provider = ZaiProvider(model="glm-9", client=FakeClient(content="{}"))
    assert provider._model == "glm-9"


def test_none_content_becomes_empty_string():
    provider = ZaiProvider(client=FakeClient(content=None))
    assert provider.generate(_request()).text == ""


def test_api_errors_become_llm_error():
    provider = ZaiProvider(client=FakeClient(error=RuntimeError("rate limited")))
    with pytest.raises(LLMError):
        provider.generate(_request())


def test_missing_key_raises_llm_error(monkeypatch):
    # No injected client and no key -> clear config error (not a triage fallback).
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        ZaiProvider()


def test_balance_error_raises_plain_llm_error_not_rate_limit():
    # Z.ai signals insufficient balance as a 429 (code 1113) — same status as a
    # transient throttle, but it won't clear on retry. It must NOT be a
    # RateLimitError, or the core tells the user to retry a billing problem.
    msg = (
        'Error code: 429, {"error":{"code":"1113","message":'
        '"Insufficient balance or no resource package. Please recharge."}}'
    )
    provider = ZaiProvider(client=FakeClient(error=RuntimeError(msg)), max_attempts=1)
    with pytest.raises(LLMError) as ei:
        provider.generate(_request())
    assert not isinstance(ei.value, RateLimitError)


def test_transient_rate_limit_still_raises_rate_limit_error():
    # A genuine 429 throttle (not a balance error) still surfaces as
    # RateLimitError so the core can tell the user the service is busy.
    provider = ZaiProvider(
        client=FakeClient(error=RuntimeError("429 Too Many Requests")), max_attempts=1
    )
    with pytest.raises(RateLimitError):
        provider.generate(_request())


def test_retries_transient_failures_then_succeeds():
    payload = json.dumps({"urgency": "medium", "recommendation": "a", "disclaimer": "d"})
    client = FakeClient(content=payload, fail_times=2)
    provider = ZaiProvider(client=client)
    assert provider.generate(_request()).text == payload
    assert client.chat.completions.calls == 3


def test_gives_up_after_max_attempts():
    client = FakeClient(error=RuntimeError("always down"))
    provider = ZaiProvider(client=client, max_attempts=2)
    with pytest.raises(LLMError):
        provider.generate(_request())
    assert client.chat.completions.calls == 2


def test_check_health_ok():
    ZaiProvider(client=FakeClient(content="{}")).check_health()  # no exception


def test_check_health_failure_raises_llm_error():
    provider = ZaiProvider(client=FakeClient(error=RuntimeError("no network")))
    with pytest.raises(LLMError):
        provider.check_health()


def test_client_configured_with_timeout_and_no_sdk_retries(monkeypatch):
    import zai

    captured = {}

    def fake_ctor(**kwargs):
        captured.update(kwargs)
        return FakeClient(content="{}")

    monkeypatch.setattr(zai, "ZaiClient", fake_ctor)
    ZaiProvider(api_key="x", timeout=12.5)
    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 0
    assert captured["api_key"] == "x"
