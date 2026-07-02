"""Z.ai adapter for the ``LLMProvider`` port.

Targets Z.ai's chat-completions API (OpenAI-compatible) with a GLM model by
default. Like every adapter, it knows nothing about triage — it just turns an
``LLMRequest`` into a chat call and returns the reply text. All prompt and
parsing logic stays in the core.

The ``zai-sdk`` package is an optional extra (``pip install 'triage-buddy[zai]'``), so
it is imported lazily: the core + mock slice still has zero third-party deps.
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

DEFAULT_MODEL = "GLM-4.7-FlashX"


def _is_quota_exhausted(exc: BaseException) -> bool:
    """True if Z.ai reports the account is out of balance/quota.

    Z.ai signals "insufficient balance / no resource package" as HTTP 429 with
    error code 1113 — the same status as a transient rate limit, but it will NOT
    clear on retry (the account needs a recharge). Surfaced as a plain
    ``LLMError`` rather than ``RateLimitError`` so the core's fail-safe message
    doesn't tell the user to retry a billing problem. Detected from the message
    to stay free of SDK exception imports, matching ``is_rate_limit``'s style.
    """
    text = str(exc).lower()
    return (
        "insufficient balance" in text
        or "no resource package" in text
        or "please recharge" in text
        or '"1113"' in text
    )


class ZaiProvider:
    """``LLMProvider`` backed by Z.ai (Zhipu AI) GLM models.

    Args:
        model: Z.ai model id. If omitted, falls back to the ``TRIAGE_MODEL``
            environment variable, then to the hardcoded ``DEFAULT_MODEL``.
        api_key: Overrides the ``ZAI_API_KEY`` environment variable.
        temperature: Sampling temperature. Defaults to ``0`` for repeatable
            triage output.
        client: A pre-built ZaiClient (or any object exposing the same
            ``chat.completions.create`` interface). Mainly for testing — when
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
            from zai import ZaiClient
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise LLMError(
                "the 'zai-sdk' package is required for the Z.ai provider; "
                "install it with: pip install 'triage-buddy[zai]'"
            ) from exc

        key = api_key or os.environ.get("ZAI_API_KEY")
        if not key:
            raise LLMError(
                "ZAI_API_KEY is not set; export it or pass api_key= to use the Z.ai provider"
            )
        # max_retries=0: this adapter owns retries (via call_with_retries) so
        # behavior is uniform across providers and not double-counted.
        self._client = ZaiClient(api_key=key, timeout=timeout, max_retries=0)

    def generate(self, request: LLMRequest) -> LLMResponse:
        def attempt() -> LLMResponse:
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    # The core's system prompt already demands a JSON object;
                    # JSON mode makes GLM hold to it more reliably.
                    response_format={"type": "json_object"},
                    # The GLM-4.5/4.6 family runs "thinking mode" by default — a
                    # long internal chain-of-thought (~60s) for no benefit on a
                    # temperature-0 JSON reply. Disable it for fast, deterministic triage.
                    thinking={"type": "disabled"},
                    messages=[
                        {"role": "system", "content": request.system},
                        {"role": "user", "content": request.user},
                    ],
                )
                text = completion.choices[0].message.content or ""
            except Exception as exc:  # network, auth, rate limit, timeout
                if _is_quota_exhausted(exc):
                    raise LLMError(
                        f"Z.ai account has insufficient balance/quota: {exc}"
                    ) from exc
                if is_rate_limit(exc):
                    raise RateLimitError(f"Z.ai rate limit: {exc}") from exc
                raise LLMError(f"Z.ai request failed: {exc}") from exc
            return LLMResponse(text=text)

        return call_with_retries(
            attempt, attempts=self._max_attempts, base_delay=self._retry_base_delay
        )

    def check_health(self) -> None:
        """Cheap reachability probe: a 1-token completion.

        The zai-sdk client has no model-listing endpoint (unlike Groq/Gemini),
        so an actual completion call is the only reachability check available.
        Kept as cheap as possible: ``max_tokens=1`` and thinking disabled.

        Raises ``LLMError`` if the provider is unreachable or unauthorized.
        """
        try:
            self._client.chat.completions.create(
                model=self._model,
                max_tokens=1,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": "ping"}],
            )
        except Exception as exc:
            raise LLMError(f"Z.ai health check failed: {exc}") from exc
