"""Composition root: where adapters are chosen and wired to the core.

This is the only place that knows about concrete providers. Adding a real
provider later means adding a branch here (and an adapter module) — the core and
the CLI stay untouched.
"""

from __future__ import annotations

from triage_buddy.adapters.llm.gemini import GeminiProvider
from triage_buddy.adapters.llm.groq import GroqProvider
from triage_buddy.adapters.llm.mock import MockLLMProvider
from triage_buddy.domain.triage import TriageService
from triage_buddy.ports.llm import LLMProvider


def build_provider(name: str = "mock") -> LLMProvider:
    """Return an ``LLMProvider`` by name."""
    if name == "mock":
        return MockLLMProvider()
    if name == "groq":
        return GroqProvider()
    if name == "gemini":
        return GeminiProvider()
    # Future: "anthropic", "openai", ... resolved here.
    raise ValueError(f"unknown LLM provider: {name!r}")


def build_service(provider: str = "mock") -> TriageService:
    """Wire a ``TriageService`` with the chosen provider adapter."""
    return TriageService(llm=build_provider(provider))
