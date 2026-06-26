"""Groq adapter for the ``LLMProvider`` port.

Targets Groq's chat-completions API (OpenAI-compatible) with a Llama model by
default. Like every adapter, it knows nothing about triage — it just turns an
``LLMRequest`` into a chat call and returns the reply text. All prompt and
parsing logic stays in the core.

The ``groq`` SDK is an optional extra (``pip install 'triage-buddy[groq]'``), so
it is imported lazily: the core + mock slice still has zero third-party deps.
"""

from __future__ import annotations

import os

from triage_buddy.ports.llm import LLMError, LLMRequest, LLMResponse

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqProvider:
    """``LLMProvider`` backed by Groq-hosted Llama models.

    Args:
        model: Groq model id. Defaults to ``llama-3.3-70b-versatile``.
        api_key: Overrides the ``GROQ_API_KEY`` environment variable.
        temperature: Sampling temperature. Defaults to ``0`` for repeatable
            triage output.
        client: A pre-built Groq client (or any object exposing the same
            ``chat.completions.create`` interface). Mainly for testing — when
            omitted, a real client is constructed from the API key.

    Config problems (missing SDK, missing key) raise ``LLMError`` *here*, at
    construction, so they surface as a clear setup error rather than masquerading
    as a triage fallback.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        client: object | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature

        if client is not None:
            self._client = client
            return

        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise LLMError(
                "the 'groq' package is required for the Groq provider; "
                "install it with: pip install 'triage-buddy[groq]'"
            ) from exc

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMError(
                "GROQ_API_KEY is not set; export it or pass api_key= to use the Groq provider"
            )
        self._client = Groq(api_key=key)

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                # The core's system prompt already demands a JSON object; asking
                # for JSON mode makes Llama hold to it more reliably.
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.user},
                ],
            )
            text = completion.choices[0].message.content or ""
        except Exception as exc:  # network, auth, rate limit, malformed response
            raise LLMError(f"Groq request failed: {exc}") from exc

        return LLMResponse(text=text)
