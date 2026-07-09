"""Composition root: where adapters are chosen and wired to the core.

This is the only place that knows about concrete providers. Adding a real
provider later means adding a branch here (and an adapter module) — the core and
the CLI stay untouched.
"""

from __future__ import annotations

from triage_buddy.adapters.llm.gemini import GeminiProvider
from triage_buddy.adapters.llm.groq import GroqProvider
from triage_buddy.adapters.llm.labproxy import LabProxyProvider
from triage_buddy.adapters.llm.ollama import OllamaProvider
from triage_buddy.adapters.llm.openrouter import OpenRouterProvider
from triage_buddy.adapters.llm.zai import ZaiProvider
from triage_buddy.adapters.llm.mock import MockLLMProvider
from triage_buddy.domain.triage import TriageService
from triage_buddy.ports.llm import LLMProvider


def build_provider(name: str = "mock", *, model: str | None = None) -> LLMProvider:
    """Return an ``LLMProvider`` by name.

    ``model`` overrides the chosen provider's default model id when given; when
    omitted (None) the provider keeps its own default / env-var resolution, so
    existing callers are unaffected.
    """
    kwargs = {"model": model} if model else {}
    if name == "mock":
        return MockLLMProvider()
    if name == "groq":
        return GroqProvider(**kwargs)
    if name == "gemini":
        return GeminiProvider(**kwargs)
    if name == "zai":
        return ZaiProvider(**kwargs)
    if name == "labproxy":
        return LabProxyProvider(**kwargs)
    if name == "ollama":
        return OllamaProvider(**kwargs)
    if name == "openrouter":
        return OpenRouterProvider(**kwargs)
    # Future: "anthropic", "openai", ... resolved here.
    raise ValueError(f"unknown LLM provider: {name!r}")


def build_service(provider: str = "mock", *, model: str | None = None) -> TriageService:
    """Wire a ``TriageService`` with the chosen provider adapter."""
    return TriageService(llm=build_provider(provider, model=model))
