"""Unit tests for the Groq adapter, using a fake client (no network, no key)."""

import json
from types import SimpleNamespace

import pytest

from triage_buddy.adapters.llm.groq import DEFAULT_MODEL, GroqProvider
from triage_buddy.ports.llm import LLMError, LLMRequest


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


class FakeModelsList:
    def __init__(self, error=None):
        self._error = error
        self.calls = 0

    def list(self):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return ["llama-x"]


class FakeClient:
    def __init__(self, *, health_error=None, **kwargs):
        self.chat = SimpleNamespace(completions=FakeCompletions(**kwargs))
        self.models = FakeModelsList(error=health_error)


def _request():
    return LLMRequest(system="sys", user="usr")


def test_generate_sends_messages_and_returns_text():
    payload = json.dumps({"urgency": "medium", "recommendation": "a", "disclaimer": "d"})
    client = FakeClient(content=payload)
    provider = GroqProvider(client=client, model="custom-model")

    response = provider.generate(_request())

    assert response.text == payload
    kwargs = client.chat.completions.last_kwargs
    assert kwargs["model"] == "custom-model"
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["messages"][1] == {"role": "user", "content": "usr"}
    assert kwargs["response_format"] == {"type": "json_object"}


def test_default_model_is_llama_versatile():
    assert DEFAULT_MODEL == "llama-3.3-70b-versatile"


def test_none_content_becomes_empty_string():
    provider = GroqProvider(client=FakeClient(content=None))
    assert provider.generate(_request()).text == ""


def test_api_errors_become_llm_error():
    provider = GroqProvider(client=FakeClient(error=RuntimeError("rate limited")))
    with pytest.raises(LLMError):
        provider.generate(_request())


def test_missing_key_raises_llm_error(monkeypatch):
    # No injected client and no key -> clear config error (not a triage fallback).
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError):
        GroqProvider()


def test_retries_transient_failures_then_succeeds():
    payload = json.dumps({"urgency": "medium", "recommendation": "a", "disclaimer": "d"})
    client = FakeClient(content=payload, fail_times=2)
    provider = GroqProvider(client=client, max_attempts=3, retry_base_delay=0)

    assert provider.generate(_request()).text == payload
    assert client.chat.completions.calls == 3


def test_gives_up_after_max_attempts():
    client = FakeClient(error=RuntimeError("always down"))
    provider = GroqProvider(client=client, max_attempts=2, retry_base_delay=0)

    with pytest.raises(LLMError):
        provider.generate(_request())
    assert client.chat.completions.calls == 2


def test_check_health_ok():
    GroqProvider(client=FakeClient(content="{}")).check_health()  # no exception


def test_check_health_failure_raises_llm_error():
    provider = GroqProvider(client=FakeClient(health_error=RuntimeError("no network")))
    with pytest.raises(LLMError):
        provider.check_health()


def test_client_configured_with_timeout_and_no_sdk_retries(monkeypatch):
    import groq

    captured = {}

    def fake_ctor(**kwargs):
        captured.update(kwargs)
        return FakeClient(content="{}")

    monkeypatch.setattr(groq, "Groq", fake_ctor)
    GroqProvider(api_key="x", timeout=12.5)
    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 0
    assert captured["api_key"] == "x"
