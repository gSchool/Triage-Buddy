"""Unit tests for the Ollama adapter, using a fake opener (no network)."""

import io
import json
import urllib.error

import pytest

from triage_buddy.adapters.llm.ollama import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaProvider
from triage_buddy.ports.llm import LLMError, LLMRequest, RateLimitError


class FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, *, payload=None, error=None, fail_times=0):
        self._payload = payload
        self._error = error
        self._fail_times = fail_times
        self.calls = 0
        self.last_request = None

    def __call__(self, request, timeout=None):
        self.last_request = request
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self.calls <= self._fail_times:
            raise RuntimeError("transient boom")
        return FakeResponse(self._payload)


def _request():
    return LLMRequest(system="sys", user="usr")


def _message_payload(text="hi"):
    return {"message": {"role": "assistant", "content": text}}


def test_generate_sends_chat_request_and_returns_text(monkeypatch):
    monkeypatch.delenv("TRIAGE_MODEL", raising=False)
    opener = FakeOpener(payload=_message_payload("hello"))
    provider = OllamaProvider(
        base_url="http://localhost:11434", model="gemma4", opener=opener
    )

    response = provider.generate(_request())

    assert response.text == "hello"
    req = opener.last_request
    assert req.full_url == "http://localhost:11434/api/chat"
    body = json.loads(req.data)
    assert body["model"] == "gemma4"
    assert body["stream"] is False
    assert body["format"] == "json"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert body["options"]["temperature"] == 0.0


def test_base_url_trailing_slash_is_stripped():
    opener = FakeOpener(payload=_message_payload())
    provider = OllamaProvider(base_url="http://localhost:11434/", opener=opener)
    provider.generate(_request())
    assert opener.last_request.full_url == "http://localhost:11434/api/chat"


def test_default_base_url_used_when_no_env_or_arg(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    provider = OllamaProvider(opener=FakeOpener())
    assert provider._base_url == DEFAULT_BASE_URL


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env.example:11434")
    opener = FakeOpener(payload=_message_payload())
    provider = OllamaProvider(opener=opener)
    provider.generate(_request())
    assert opener.last_request.full_url == "http://from-env.example:11434/api/chat"


def test_default_model_used_when_no_env(monkeypatch):
    monkeypatch.delenv("TRIAGE_MODEL", raising=False)
    provider = OllamaProvider(opener=FakeOpener())
    assert provider._model == DEFAULT_MODEL


def test_env_var_overrides_default_model(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "some-model")
    provider = OllamaProvider(opener=FakeOpener())
    assert provider._model == "some-model"


def test_explicit_model_arg_beats_env_var(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "some-model")
    provider = OllamaProvider(model="custom-model", opener=FakeOpener())
    assert provider._model == "custom-model"


def test_no_api_key_required():
    # Ollama is an unauthenticated local server; construction never raises for
    # missing credentials, unlike the hosted providers.
    OllamaProvider(opener=FakeOpener())


def test_http_error_becomes_llm_error():
    error = urllib.error.HTTPError("url", 500, "boom", {}, io.BytesIO(b""))
    provider = OllamaProvider(opener=FakeOpener(error=error))
    with pytest.raises(LLMError):
        provider.generate(_request())


def test_http_429_becomes_rate_limit_error():
    error = urllib.error.HTTPError("url", 429, "too many", {}, io.BytesIO(b""))
    provider = OllamaProvider(opener=FakeOpener(error=error))
    with pytest.raises(RateLimitError):
        provider.generate(_request())


def test_connection_failure_becomes_llm_error():
    # Ollama not running locally: urllib raises URLError, not HTTPError.
    error = urllib.error.URLError("Connection refused")
    provider = OllamaProvider(opener=FakeOpener(error=error))
    with pytest.raises(LLMError):
        provider.generate(_request())


def test_retries_transient_failures_then_succeeds():
    opener = FakeOpener(payload=_message_payload("ok"), fail_times=2)
    provider = OllamaProvider(opener=opener, max_attempts=3, retry_base_delay=0)
    assert provider.generate(_request()).text == "ok"
    assert opener.calls == 3


def test_gives_up_after_max_attempts():
    opener = FakeOpener(error=RuntimeError("always down"))
    provider = OllamaProvider(opener=opener, max_attempts=2, retry_base_delay=0)
    with pytest.raises(LLMError):
        provider.generate(_request())
    assert opener.calls == 2


def test_check_health_ok():
    provider = OllamaProvider(opener=FakeOpener(payload=_message_payload()))
    provider.check_health()  # no exception


def test_check_health_failure_raises_llm_error():
    provider = OllamaProvider(opener=FakeOpener(error=RuntimeError("no network")))
    with pytest.raises(LLMError):
        provider.check_health()
