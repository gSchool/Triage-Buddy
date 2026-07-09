"""OpenRouter adapter for the ``LLMProvider`` port.

Targets OpenRouter's chat-completions API (OpenAI-compatible) at
``https://openrouter.ai/api/v1/chat/completions``. OpenRouter is a hosted,
authenticated router in front of many models; the model is chosen by a slug like
``openai/gpt-4o-mini`` or ``meta-llama/llama-3.3-70b-instruct``. Like every
adapter, it knows nothing about triage — it just turns an ``LLMRequest`` into a
chat call and returns the reply text. All prompt and parsing logic stays in the
core.

OpenRouter just speaks plain HTTP + JSON, so this adapter uses the stdlib
``urllib`` (like the ollama and lab-proxy adapters) rather than the OpenAI SDK —
no extra install needed, keeping the core + mock slice free of third-party deps.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from triage_buddy.adapters.llm._http import default_opener
from triage_buddy.adapters.llm._retry import (
    DEFAULT_ATTEMPTS,
    DEFAULT_BASE_DELAY,
    DEFAULT_TIMEOUT,
    call_with_retries,
    is_rate_limit,
)
from triage_buddy.ports.llm import LLMError, LLMRequest, LLMResponse, RateLimitError

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Optional attribution headers OpenRouter uses for its rankings; harmless to send.
DEFAULT_REFERER = "https://github.com/triage-buddy"
DEFAULT_TITLE = "Triage Buddy"


class OpenRouterProvider:
    """``LLMProvider`` backed by OpenRouter's OpenAI-compatible chat API.

    Args:
        base_url: OpenRouter API base URL. Falls back to the
            ``OPENROUTER_BASE_URL`` environment variable, then to
            ``https://openrouter.ai/api/v1``.
        api_key: Sent as ``Authorization: Bearer``. Falls back to the
            ``OPEN_ROUTER_KEY`` environment variable (also accepts the
            ``OPENROUTER_API_KEY`` spelling). Required.
        model: OpenRouter model slug (e.g. ``deepseek/deepseek-v4-flash``). If omitted,
            falls back to the ``TRIAGE_MODEL`` environment variable, then to
            ``DEFAULT_MODEL``.
        temperature: Sampling temperature. Defaults to ``0`` for repeatable
            triage output.
        opener: Injected for testing — a callable taking a
            ``urllib.request.Request`` and returning a file-like object with
            ``.read()``. Defaults to a certifi-backed opener when ``certifi``
            is installed, else ``urllib.request.urlopen``.

    Config problems (missing key) raise ``LLMError`` *here*, at construction, so
    they surface as a clear setup error rather than masquerading as a triage
    fallback.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_ATTEMPTS,
        retry_base_delay: float = DEFAULT_BASE_DELAY,
        opener=None,
    ) -> None:
        key = (
            api_key
            or os.environ.get("OPEN_ROUTER_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if not key:
            raise LLMError(
                "OPEN_ROUTER_KEY is not set; export it or pass api_key= to use "
                "the OpenRouter provider"
            )
        self._api_key = key
        url = base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL
        self._base_url = url.rstrip("/")
        self._model = model or os.environ.get("TRIAGE_MODEL") or DEFAULT_MODEL
        self._temperature = temperature
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay
        self._opener = opener or default_opener()

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                # Optional attribution headers used for OpenRouter's rankings.
                "HTTP-Referer": DEFAULT_REFERER,
                "X-Title": DEFAULT_TITLE,
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimitError(f"OpenRouter rate limit: {exc}") from exc
            raise LLMError(f"OpenRouter request failed: {exc}") from exc
        except Exception as exc:  # network, timeout, malformed JSON
            if is_rate_limit(exc):
                raise RateLimitError(f"OpenRouter rate limit: {exc}") from exc
            raise LLMError(f"OpenRouter request failed: {exc}") from exc

    def generate(self, request: LLMRequest) -> LLMResponse:
        def attempt() -> LLMResponse:
            data = self._post(
                {
                    "model": self._model,
                    "temperature": self._temperature,
                    # The core's system prompt already demands a JSON object;
                    # JSON mode makes the model hold to it more reliably.
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": request.system},
                        {"role": "user", "content": request.user},
                    ],
                }
            )
            choices = data.get("choices") or []
            text = choices[0].get("message", {}).get("content", "") if choices else ""
            return LLMResponse(text=text)

        return call_with_retries(
            attempt, attempts=self._max_attempts, base_delay=self._retry_base_delay
        )

    def check_health(self) -> None:
        """Cheap reachability probe: list models (no token cost).

        Raises ``LLMError`` if the provider is unreachable or unauthorized.
        """
        request = urllib.request.Request(
            f"{self._base_url}/models",
            headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                response.read()
        except Exception as exc:
            raise LLMError(f"OpenRouter health check failed: {exc}") from exc
