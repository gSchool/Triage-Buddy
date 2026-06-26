"""The LLM provider port.

This is deliberately a *generic text-completion* contract, not a triage-shaped
one. A provider's only job is: given a system + user prompt, return text. All
triage knowledge (how to build the prompt, how to interpret the reply) lives in
the core, so swapping Anthropic for OpenAI for a local model touches only an
adapter — never the domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMRequest:
    """A prompt to send to a provider."""

    system: str
    user: str


@dataclass(frozen=True)
class LLMResponse:
    """The raw text a provider returns."""

    text: str


class LLMError(Exception):
    """Raised by an adapter when generation fails (network, auth, rate limit).

    The core catches this and falls back to a conservative assessment rather
    than crashing — failing safe matters in a triage context.
    """


@runtime_checkable
class LLMProvider(Protocol):
    """A swappable text-completion provider."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return the model's text completion, or raise ``LLMError``."""
        ...
