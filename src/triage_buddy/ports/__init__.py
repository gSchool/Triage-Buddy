"""Ports: the interfaces the core depends on.

The core imports from here; adapters implement these. This is the seam that
keeps provider and transport details out of the domain.
"""

from triage_buddy.ports.llm import LLMError, LLMProvider, LLMRequest, LLMResponse

__all__ = ["LLMError", "LLMProvider", "LLMRequest", "LLMResponse"]
