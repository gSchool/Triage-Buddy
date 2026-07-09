"""Unit tests for the OpenRouter adapter, using a fake opener (no network)."""

import io
import json
import urllib.error

import pytest

from triage_buddy.adapters.llm.openrouter import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OpenRouterProvider,
)
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


def _completion_payload(text="hi"):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    # A key is required at construction; supply one so tests don't depend on env.
    monkeypatch.setenv("OPEN_ROUTER_KEY", "sk-or-test")


def test_generate_sends_chat_request_and_returns_text(monkeypatch):
    monkeypatch.delenv("TRIAGE_MODEL", raising=False)
    opener = FakeOpener(payload=_completion_payload("hello"))
    provider = OpenRouterProvider(model="openai/gpt-4o-mini", opener=opener)

    response = provider.generate(_request())

    assert response.text == "hello"
    req = opener.last_request
    assert req.full_url == "https://openrouter.ai/api/v1/chat/completions"
    body = json.loads(req.data)
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_authorization_header_sent():
    opener = FakeOpener(payload=_completion_payload())
    provider = OpenRouterProvider(api_key="sk-or-explicit", opener=opener)
    provider.generate(_request())
    # urllib normalizes header names to title-case.
    assert opener.last_request.get_header("Authorization") == "Bearer sk-or-explicit"


def test_base_url_trailing_slash_is_stripped():
    opener = FakeOpener(payload=_completion_payload())
    provider = OpenRouterProvider(
        base_url="https://openrouter.ai/api/v1/", opener=opener
    )
    provider.generate(_request())
    assert (
        opener.last_request.full_url
        == "https://openrouter.ai/api/v1/chat/completions"
    )


def test_default_base_url_used_when_no_env_or_arg(monkeypatch):
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    provider = OpenRouterProvider(opener=FakeOpener())
    assert provider._base_url == DEFAULT_BASE_URL


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://proxy.example/api/v1")
    opener = FakeOpener(payload=_completion_payload())
    provider = OpenRouterProvider(opener=opener)
    provider.generate(_request())
    assert (
        opener.last_request.full_url
        == "https://proxy.example/api/v1/chat/completions"
    )


def test_default_model_used_when_no_env(monkeypatch):
    monkeypatch.delenv("TRIAGE_MODEL", raising=False)
    provider = OpenRouterProvider(opener=FakeOpener())
    assert provider._model == DEFAULT_MODEL


def test_env_var_overrides_default_model(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "some/model")
    provider = OpenRouterProvider(opener=FakeOpener())
    assert provider._model == "some/model"


def test_explicit_model_arg_beats_env_var(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "some/model")
    provider = OpenRouterProvider(model="custom/model", opener=FakeOpener())
    assert provider._model == "custom/model"


def test_missing_key_raises_at_construction(monkeypatch):
    monkeypatch.delenv("OPEN_ROUTER_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError):
        OpenRouterProvider(opener=FakeOpener())


def test_openrouter_api_key_env_spelling_accepted(monkeypatch):
    monkeypatch.delenv("OPEN_ROUTER_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-alt")
    opener = FakeOpener(payload=_completion_payload())
    provider = OpenRouterProvider(opener=opener)
    provider.generate(_request())
    assert opener.last_request.get_header("Authorization") == "Bearer sk-or-alt"


def test_http_error_becomes_llm_error():
    error = urllib.error.HTTPError("url", 500, "boom", {}, io.BytesIO(b""))
    provider = OpenRouterProvider(opener=FakeOpener(error=error))
    with pytest.raises(LLMError):
        provider.generate(_request())


def test_http_429_becomes_rate_limit_error():
    error = urllib.error.HTTPError("url", 429, "too many", {}, io.BytesIO(b""))
    provider = OpenRouterProvider(opener=FakeOpener(error=error))
    with pytest.raises(RateLimitError):
        provider.generate(_request())


def test_connection_failure_becomes_llm_error():
    error = urllib.error.URLError("Connection refused")
    provider = OpenRouterProvider(opener=FakeOpener(error=error))
    with pytest.raises(LLMError):
        provider.generate(_request())


def test_retries_transient_failures_then_succeeds():
    opener = FakeOpener(payload=_completion_payload("ok"), fail_times=2)
    provider = OpenRouterProvider(opener=opener, max_attempts=3, retry_base_delay=0)
    assert provider.generate(_request()).text == "ok"
    assert opener.calls == 3


def test_gives_up_after_max_attempts():
    opener = FakeOpener(error=RuntimeError("always down"))
    provider = OpenRouterProvider(opener=opener, max_attempts=2, retry_base_delay=0)
    with pytest.raises(LLMError):
        provider.generate(_request())
    assert opener.calls == 2


def test_check_health_ok():
    provider = OpenRouterProvider(opener=FakeOpener(payload={"data": []}))
    provider.check_health()  # no exception


def test_check_health_failure_raises_llm_error():
    provider = OpenRouterProvider(opener=FakeOpener(error=RuntimeError("no network")))
    with pytest.raises(LLMError):
        provider.check_health()
