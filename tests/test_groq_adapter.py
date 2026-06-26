"""Unit tests for the Groq adapter, using a fake client (no network, no key)."""

import json
from types import SimpleNamespace

import pytest

from triage_buddy.adapters.llm.groq import DEFAULT_MODEL, GroqProvider
from triage_buddy.ports.llm import LLMError, LLMRequest


class FakeCompletions:
    def __init__(self, *, content=None, error=None):
        self._content = content
        self._error = error
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=FakeCompletions(**kwargs))


def _request():
    return LLMRequest(system="sys", user="usr")


def test_generate_sends_messages_and_returns_text():
    payload = json.dumps({"level": "PROMPT", "rationale": "r", "advice": "a"})
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
