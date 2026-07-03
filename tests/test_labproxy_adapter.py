"""Unit tests for the lab-proxy adapter, using a fake opener (no network)."""

import io
import json
import urllib.error

import pytest

from triage_buddy.adapters.llm.labproxy import DEFAULT_MODEL, LabProxyProvider
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


def _content_payload(text="hi"):
    return {"content": [{"type": "text", "text": text}]}


def test_generate_sends_anthropic_shaped_request_and_returns_text():
    opener = FakeOpener(payload=_content_payload("hello"))
    provider = LabProxyProvider(
        base_url="https://proxy.example",
        student_id="jb",
        api_key="k",
        model="claude-sonnet-5",
        opener=opener,
    )

    response = provider.generate(_request())

    assert response.text == "hello"
    req = opener.last_request
    assert req.full_url == "https://proxy.example/v1/messages"
    assert req.get_header("X-student-id") == "jb"
    assert req.get_header("X-api-key") == "k"
    assert req.get_header("Anthropic-version") == "2023-06-01"
    body = json.loads(req.data)
    assert body["model"] == "claude-sonnet-5"
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "usr"}]
    assert body["temperature"] == 0.0


def test_base_url_trailing_slash_is_stripped():
    opener = FakeOpener(payload=_content_payload())
    provider = LabProxyProvider(base_url="https://proxy.example/", opener=opener)
    provider.generate(_request())
    assert opener.last_request.full_url == "https://proxy.example/v1/messages"


def test_missing_base_url_raises_llm_error(monkeypatch):
    monkeypatch.delenv("LABPROXY_BASE_URL", raising=False)
    with pytest.raises(LLMError):
        LabProxyProvider()


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("LABPROXY_BASE_URL", "https://from-env.example")
    opener = FakeOpener(payload=_content_payload())
    provider = LabProxyProvider(opener=opener)
    provider.generate(_request())
    assert opener.last_request.full_url == "https://from-env.example/v1/messages"


def test_api_key_defaults_to_placeholder_when_unset(monkeypatch):
    monkeypatch.delenv("LABPROXY_API_KEY", raising=False)
    opener = FakeOpener(payload=_content_payload())
    provider = LabProxyProvider(base_url="https://proxy.example", opener=opener)
    provider.generate(_request())
    assert opener.last_request.get_header("X-api-key")


def test_student_id_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("LABPROXY_STUDENT_ID", raising=False)
    opener = FakeOpener(payload=_content_payload())
    provider = LabProxyProvider(base_url="https://proxy.example", opener=opener)
    provider.generate(_request())
    assert opener.last_request.get_header("X-student-id")


def test_default_model_used_when_no_env(monkeypatch):
    monkeypatch.delenv("TRIAGE_MODEL", raising=False)
    provider = LabProxyProvider(base_url="https://proxy.example", opener=FakeOpener())
    assert provider._model == DEFAULT_MODEL


def test_env_var_overrides_default_model(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "some-model")
    provider = LabProxyProvider(base_url="https://proxy.example", opener=FakeOpener())
    assert provider._model == "some-model"


def test_explicit_model_arg_beats_env_var(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL", "some-model")
    provider = LabProxyProvider(
        base_url="https://proxy.example", model="custom-model", opener=FakeOpener()
    )
    assert provider._model == "custom-model"


def test_non_text_content_blocks_are_ignored():
    payload = {"content": [{"type": "tool_use", "id": "x"}, {"type": "text", "text": "ok"}]}
    provider = LabProxyProvider(
        base_url="https://proxy.example", opener=FakeOpener(payload=payload)
    )
    assert provider.generate(_request()).text == "ok"


def test_http_error_becomes_llm_error():
    error = urllib.error.HTTPError("url", 500, "boom", {}, io.BytesIO(b""))
    provider = LabProxyProvider(
        base_url="https://proxy.example", opener=FakeOpener(error=error)
    )
    with pytest.raises(LLMError):
        provider.generate(_request())


def test_http_429_becomes_rate_limit_error():
    error = urllib.error.HTTPError("url", 429, "too many", {}, io.BytesIO(b""))
    provider = LabProxyProvider(
        base_url="https://proxy.example", opener=FakeOpener(error=error)
    )
    with pytest.raises(RateLimitError):
        provider.generate(_request())


def test_retries_transient_failures_then_succeeds():
    opener = FakeOpener(payload=_content_payload("ok"), fail_times=2)
    provider = LabProxyProvider(
        base_url="https://proxy.example",
        opener=opener,
        max_attempts=3,
        retry_base_delay=0,
    )
    assert provider.generate(_request()).text == "ok"
    assert opener.calls == 3


def test_gives_up_after_max_attempts():
    opener = FakeOpener(error=RuntimeError("always down"))
    provider = LabProxyProvider(
        base_url="https://proxy.example",
        opener=opener,
        max_attempts=2,
        retry_base_delay=0,
    )
    with pytest.raises(LLMError):
        provider.generate(_request())
    assert opener.calls == 2


def test_check_health_ok():
    provider = LabProxyProvider(
        base_url="https://proxy.example", opener=FakeOpener(payload=_content_payload())
    )
    provider.check_health()  # no exception


def test_check_health_failure_raises_llm_error():
    provider = LabProxyProvider(
        base_url="https://proxy.example",
        opener=FakeOpener(error=RuntimeError("no network")),
    )
    with pytest.raises(LLMError):
        provider.check_health()
