"""Unit tests for the Gemini adapter, using a fake client (no network, no key)."""

import json
from types import SimpleNamespace

import pytest

from triage_buddy.adapters.llm.gemini import DEFAULT_MODEL, GeminiProvider
from triage_buddy.ports.llm import LLMError, LLMRequest


class FakeModels:
    def __init__(self, *, text=None, error=None, fail_times=0, health_error=None):
        self._text = text
        self._error = error
        self._fail_times = fail_times
        self._health_error = health_error
        self.last_kwargs = None
        self.calls = 0

    def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self.calls <= self._fail_times:
            raise RuntimeError("transient boom")
        return SimpleNamespace(text=self._text)

    def list(self):
        if self._health_error is not None:
            raise self._health_error
        return ["gemini-x"]


class FakeClient:
    def __init__(self, **kwargs):
        self.models = FakeModels(**kwargs)


def _request():
    return LLMRequest(system="sys", user="usr")


def test_generate_sends_prompt_and_returns_text():
    payload = json.dumps({"urgency": "medium", "recommendation": "a", "disclaimer": "d"})
    client = FakeClient(text=payload)
    provider = GeminiProvider(client=client, model="custom-gemini")

    response = provider.generate(_request())

    assert response.text == payload
    kwargs = client.models.last_kwargs
    assert kwargs["model"] == "custom-gemini"
    assert kwargs["contents"] == "usr"
    assert kwargs["config"]["system_instruction"] == "sys"
    assert kwargs["config"]["response_mime_type"] == "application/json"
    assert kwargs["config"]["temperature"] == 0.0


def test_default_model_is_gemini_flash():
    assert DEFAULT_MODEL == "gemini-2.5-flash"


def test_default_model_used_when_no_env(monkeypatch):
    monkeypatch.delenv("TRIAGE_MODEL", raising=False)
    provider = GeminiProvider(client=FakeClient(text="{}"))
    assert provider._model == DEFAULT_MODEL


def test_env_var_overrides_default_model(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "gemini-pro")
    provider = GeminiProvider(client=FakeClient(text="{}"))
    assert provider._model == "gemini-pro"


def test_explicit_model_arg_beats_env_var(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "gemini-pro")
    provider = GeminiProvider(model="custom-gemini", client=FakeClient(text="{}"))
    assert provider._model == "custom-gemini"


def test_none_text_becomes_empty_string():
    provider = GeminiProvider(client=FakeClient(text=None))
    assert provider.generate(_request()).text == ""


def test_api_errors_become_llm_error():
    provider = GeminiProvider(client=FakeClient(error=RuntimeError("quota exceeded")))
    with pytest.raises(LLMError):
        provider.generate(_request())


def test_missing_key_raises_llm_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        GeminiProvider()


def test_retries_transient_failures_then_succeeds():
    payload = json.dumps({"urgency": "medium", "recommendation": "a", "disclaimer": "d"})
    client = FakeClient(text=payload, fail_times=2)
    provider = GeminiProvider(client=client, max_attempts=3, retry_base_delay=0)

    assert provider.generate(_request()).text == payload
    assert client.models.calls == 3


def test_gives_up_after_max_attempts():
    client = FakeClient(error=RuntimeError("always down"))
    provider = GeminiProvider(client=client, max_attempts=2, retry_base_delay=0)

    with pytest.raises(LLMError):
        provider.generate(_request())
    assert client.models.calls == 2


def test_check_health_ok():
    GeminiProvider(client=FakeClient(text="{}")).check_health()  # no exception


def test_check_health_failure_raises_llm_error():
    provider = GeminiProvider(client=FakeClient(health_error=RuntimeError("no network")))
    with pytest.raises(LLMError):
        provider.check_health()


def test_client_configured_with_timeout_in_milliseconds(monkeypatch):
    import google.genai as genai_mod

    captured = {}

    def fake_ctor(**kwargs):
        captured.update(kwargs)
        return FakeClient(text="{}")

    monkeypatch.setattr(genai_mod, "Client", fake_ctor)
    GeminiProvider(api_key="x", timeout=15)
    assert captured["http_options"].timeout == 15000  # seconds -> milliseconds
