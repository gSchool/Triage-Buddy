"""A deterministic, offline mock LLM provider.

This is the first adapter behind the ``LLMProvider`` port. It needs no network,
no API key, and no third-party SDK, so the whole slice runs and tests in
isolation. It deliberately speaks the *same wire shape* a real provider would
(JSON text matching the system prompt), so the core's prompt-building and
reply-parsing paths are exercised exactly as they would be in production.

It is NOT a clinical model — it keyword-matches to pick a plausible level.
"""

from __future__ import annotations

import json
import re

from triage_buddy.domain.models import EscalationLevel
from triage_buddy.ports.llm import LLMRequest, LLMResponse

# (regex over the user prompt, level, advice). First match wins; order matters,
# most-urgent first. The red-flag emergencies are handled by the core's safety
# layer before we're ever called, so we focus on the milder gradient here.
_RULES: tuple[tuple[str, EscalationLevel, str], ...] = (
    (
        r"high fever|fever .*(several days|days)|severe pain|dehydrat",
        EscalationLevel.URGENT,
        "Seek care today; your symptoms may need prompt evaluation.",
    ),
    (
        r"fever|persistent|worsening|rash|infection|won'?t go away",
        EscalationLevel.PROMPT,
        "Book a visit with your provider in the next day or two.",
    ),
    (
        r"mild|minor|slight|runny nose|sore throat|cough|headache",
        EscalationLevel.SELF_CARE,
        "Rest, fluids, and over-the-counter remedies are reasonable; monitor for changes.",
    ),
)

_DEFAULT = (
    EscalationLevel.ROUTINE,
    "Schedule a routine appointment to have these symptoms looked at.",
)


class MockLLMProvider:
    """Deterministic stand-in for a real LLM provider."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        text = request.user.lower()
        level, advice = _DEFAULT
        for pattern, rule_level, rule_advice in _RULES:
            if re.search(pattern, text):
                level, advice = rule_level, rule_advice
                break

        payload = {
            "level": level.name,
            "rationale": (
                f"Based on the described symptoms, a {level.label.lower()} "
                "level of care appears appropriate."
            ),
            "advice": advice,
        }
        return LLMResponse(text=json.dumps(payload))

    def check_health(self) -> None:
        """Always healthy — the mock has no external dependency."""
        return None
