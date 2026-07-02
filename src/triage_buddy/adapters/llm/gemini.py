"""Google Gemini adapter for the ``LLMProvider`` port.

Uses the ``google-genai`` SDK (``from google import genai``) with a Gemini model
by default. Like every adapter, it knows nothing about triage — it turns an
``LLMRequest`` into a ``generate_content`` call and returns the reply text. All
prompt and parsing logic stays in the core.

The ``google-genai`` SDK is an optional extra (``pip install
'triage-buddy[gemini]'``), so it is imported lazily: the core + mock slice still
has zero third-party deps.
"""

from __future__ import annotations

import os

from triage_buddy.adapters.llm._retry import (
    DEFAULT_ATTEMPTS,
    DEFAULT_BASE_DELAY,
    DEFAULT_TIMEOUT,
    call_with_retries,
    is_rate_limit,
)
from triage_buddy.ports.llm import LLMError, LLMRequest, LLMResponse, RateLimitError

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiProvider:
    """``LLMProvider`` backed by Google Gemini models.

    Args:
        model: Gemini model id. If omitted, falls back to the ``TRIAGE_MODEL``
            environment variable, then to the hardcoded ``DEFAULT_MODEL``.
        api_key: Overrides the ``GEMINI_API_KEY`` environment variable.
        temperature: Sampling temperature. Defaults to ``0`` for repeatable
            triage output.
        client: A pre-built genai client (or any object exposing the same
            ``models.generate_content`` interface). Mainly for testing — when
            omitted, a real client is constructed from the API key.

    Config problems (missing SDK, missing key) raise ``LLMError`` *here*, at
    construction, so they surface as a clear setup error rather than masquerading
    as a triage fallback.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_ATTEMPTS,
        retry_base_delay: float = DEFAULT_BASE_DELAY,
        client: object | None = None,
    ) -> None:
        self._model = model or os.environ.get("TRIAGE_MODEL") or DEFAULT_MODEL
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay

        if client is not None:
            self._client = client
            return

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise LLMError(
                "the 'google-genai' package is required for the Gemini provider; "
                "install it with: pip install 'triage-buddy[gemini]'"
            ) from exc

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError(
                "GEMINI_API_KEY is not set; export it or pass api_key= to use the Gemini provider"
            )
        # HttpOptions.timeout is in milliseconds. This adapter owns retries (via
        # call_with_retries), so we configure only the per-attempt deadline here.
        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        def attempt() -> LLMResponse:
            try:
                completion = self._client.models.generate_content(
                    model=self._model,
                    contents=request.user,
                    # Passed as a dict (the SDK coerces it to
                    # GenerateContentConfig), so we don't import the SDK here.
                    config={
                        "system_instruction": request.system,
                        "temperature": self._temperature,
                        # The core's system prompt already demands a JSON object;
                        # JSON mode makes Gemini hold to it more reliably.
                        "response_mime_type": "application/json",
                    },
                )
                text = completion.text or ""
            except Exception as exc:  # network, auth, rate limit, timeout
                if is_rate_limit(exc):
                    raise RateLimitError(f"Gemini rate limit: {exc}") from exc
                raise LLMError(f"Gemini request failed: {exc}") from exc
            return LLMResponse(text=text)

        return call_with_retries(
            attempt, attempts=self._max_attempts, base_delay=self._retry_base_delay
        )

    def check_health(self) -> None:
        """Cheap reachability probe: list models (no token cost).

        Raises ``LLMError`` if the provider is unreachable or unauthorized.
        """
        try:
            next(iter(self._client.models.list()), None)
        except Exception as exc:
            raise LLMError(f"Gemini health check failed: {exc}") from exc
