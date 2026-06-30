"""Shared retry-with-backoff helper for LLM adapters.

Network-backed providers fail transiently (timeouts, rate limits, 5xx). Rather
than each adapter (and each provider SDK) handling that differently, every real
adapter runs its request through ``call_with_retries`` so retry behavior is
uniform and testable. The per-attempt deadline (timeout) is configured on the
SDK client; this helper governs *how many* attempts and the delay between them.

All adapter failures surface as ``LLMError``, so that is what we retry on. If
every attempt fails, the last ``LLMError`` is re-raised and the core handles it
by failing safe.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from triage_buddy.ports.llm import LLMError

T = TypeVar("T")


def is_rate_limit(exc: BaseException) -> bool:
    """True if ``exc`` looks like an HTTP 429 / quota-exhaustion error.

    Avoids importing SDK exception types (adapters load SDKs lazily): both the
    Groq and Gemini SDKs put 429 on a ``status_code``/``code`` attribute, and
    failing that the message names the condition. Best-effort and conservative —
    a missed 429 just yields the generic failure message, never a wrong level.
    """
    for attr in ("status_code", "code"):
        if getattr(exc, attr, None) == 429:
            return True
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or "rate_limit" in text

DEFAULT_TIMEOUT = 30.0  # seconds, per attempt
DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 0.5  # seconds; doubled each retry


def call_with_retries(
    operation: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``operation``, retrying on ``LLMError`` with exponential backoff.

    Args:
        operation: a zero-arg callable that performs one attempt.
        attempts: total attempts (must be >= 1). ``1`` disables retrying.
        base_delay: delay before the first retry; doubled each subsequent retry.
        max_delay: cap on any single backoff delay.
        sleep: injected for testing; defaults to ``time.sleep``.

    Returns the operation's result, or re-raises the last ``LLMError`` once all
    attempts are exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_error: LLMError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except LLMError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            sleep(min(max_delay, base_delay * (2 ** (attempt - 1))))

    assert last_error is not None  # loop runs at least once with attempts >= 1
    raise last_error
