"""Ollama adapter for the ``LLMProvider`` port.

Targets a local Ollama server (``ollama serve``, default port 11434) via its
native ``/api/chat`` endpoint. Ollama has no auth and no SDK dependency worth
adding — it just speaks plain HTTP + JSON — so this adapter uses the stdlib
``urllib``, same as the lab-proxy adapter, keeping the core + mock slice free
of third-party deps.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from triage_buddy.adapters.llm._retry import (
    DEFAULT_ATTEMPTS,
    DEFAULT_BASE_DELAY,
    DEFAULT_TIMEOUT,
    call_with_retries,
    is_rate_limit,
)
from triage_buddy.ports.llm import LLMError, LLMRequest, LLMResponse, RateLimitError

DEFAULT_MODEL = "gemma4"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """``LLMProvider`` backed by a local Ollama server.

    Args:
        base_url: Ollama server base URL. Falls back to the
            ``OLLAMA_BASE_URL`` environment variable, then to
            ``http://localhost:11434`` (Ollama's default).
        model: Model id to request. If omitted, falls back to the
            ``TRIAGE_MODEL`` environment variable, then to ``DEFAULT_MODEL``.
        temperature: Sampling temperature. Defaults to ``0`` for repeatable
            triage output.
        opener: Injected for testing — a callable taking a
            ``urllib.request.Request`` and returning a file-like object with
            ``.read()``. Defaults to ``urllib.request.urlopen``.

    No API key: Ollama is an unauthenticated local server, unlike the hosted
    providers. Connection failures (e.g. Ollama not running) surface as a
    plain ``LLMError`` from ``generate``/``check_health``, same as any other
    provider outage — there is no separate config-time check since there is
    no key to validate up front.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_ATTEMPTS,
        retry_base_delay: float = DEFAULT_BASE_DELAY,
        opener=None,
    ) -> None:
        url = base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL
        self._base_url = url.rstrip("/")
        self._model = model or os.environ.get("TRIAGE_MODEL") or DEFAULT_MODEL
        self._temperature = temperature
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay
        self._opener = opener or urllib.request.urlopen

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimitError(f"Ollama rate limit: {exc}") from exc
            raise LLMError(f"Ollama request failed: {exc}") from exc
        except Exception as exc:  # network, timeout, malformed JSON
            if is_rate_limit(exc):
                raise RateLimitError(f"Ollama rate limit: {exc}") from exc
            raise LLMError(f"Ollama request failed: {exc}") from exc

    def generate(self, request: LLMRequest) -> LLMResponse:
        def attempt() -> LLMResponse:
            data = self._post(
                {
                    "model": self._model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": self._temperature},
                    "messages": [
                        {"role": "system", "content": request.system},
                        {"role": "user", "content": request.user},
                    ],
                }
            )
            text = data.get("message", {}).get("content", "")
            return LLMResponse(text=text)

        return call_with_retries(
            attempt, attempts=self._max_attempts, base_delay=self._retry_base_delay
        )

    def check_health(self) -> None:
        """Cheap reachability probe: a 1-token completion.

        Ollama's ``/api/tags`` lists installed models but doesn't confirm the
        chat endpoint works for the configured model, so an actual (minimal)
        chat call is used instead, matching the other adapters' approach.
        """
        self._post(
            {
                "model": self._model,
                "stream": False,
                "options": {"num_predict": 1},
                "messages": [{"role": "user", "content": "ping"}],
            }
        )
