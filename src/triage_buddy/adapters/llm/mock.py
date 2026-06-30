"""A deterministic, offline mock LLM provider.

This is the first adapter behind the ``LLMProvider`` port. It needs no network,
no API key, and no third-party SDK, so the whole slice runs and tests in
isolation. It deliberately speaks the *same wire shape* a real provider would
(JSON text matching the system prompt: ``{urgency, recommendation,
disclaimer}``), so the core's prompt-building and reply-parsing paths are
exercised exactly as they would be in production.

It is NOT a clinical model — it keyword-matches to pick a plausible urgency.
"""

from __future__ import annotations

import json
import re

from triage_buddy.ports.llm import LLMRequest, LLMResponse

# Phrase used to satisfy the prompt's disclaimer rule. Parsed but discarded by
# the core (the domain appends its own standing disclaimer), so its exact
# wording never reaches the user — it just keeps the wire shape honest.
_DISCLAIMER = (
    "This is general guidance and is not a substitute for professional medical advice."
)

# (regex over the user prompt, urgency bucket, recommendation). First match
# wins; order matters, most-urgent first. The red-flag emergencies are handled
# by the core's safety layer before we're ever called, so we focus on the milder
# gradient here and never emit the "emergency" bucket. Recommendations follow
# the prompt rules: no "wait", and no "emergency" wording below emergency.
_RULES: tuple[tuple[str, str, str], ...] = (
    (
        r"high fever|fever .*(several days|days)|severe pain|dehydrat",
        "high",
        "Seek prompt medical attention today; your symptoms may need urgent evaluation.",
    ),
    (
        r"fever|persistent|worsening|rash|infection|won'?t go away",
        "medium",
        "See a doctor in the next day or two to have this evaluated.",
    ),
    (
        r"mild|minor|slight|runny nose|sore throat|cough|headache",
        "low",
        "Rest and drink plenty of fluids; monitor for any changes.",
    ),
)

_DEFAULT = (
    "low",
    "See a doctor for a routine check of these symptoms.",
)


class MockLLMProvider:
    """Deterministic stand-in for a real LLM provider."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        text = request.user.lower()
        urgency, recommendation = _DEFAULT
        for pattern, rule_urgency, rule_recommendation in _RULES:
            if re.search(pattern, text):
                urgency, recommendation = rule_urgency, rule_recommendation
                break

        payload = {
            "urgency": urgency,
            "recommendation": recommendation,
            "disclaimer": _DISCLAIMER,
        }
        return LLMResponse(text=json.dumps(payload))

    def check_health(self) -> None:
        """Always healthy — the mock has no external dependency."""
        return None
