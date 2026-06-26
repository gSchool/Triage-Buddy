"""The triage service: the application's core use case.

It orchestrates three things, in safety-first order:

1. Deterministic red-flag rules (cannot be overruled by a model).
2. The LLM provider's suggestion (via the port).
3. A conservative fallback when the provider fails or returns garbage.

The final level is always the *most severe* of the deterministic floor and the
model's suggestion, so the LLM can only ever escalate, never de-escalate, past a
recognized red flag.
"""

from __future__ import annotations

from triage_buddy.domain.models import (
    EscalationLevel,
    SymptomReport,
    TriageAssessment,
)
from triage_buddy.domain.safety import DISCLAIMER, detect_red_flags
from triage_buddy.prompts import (
    DraftParseError,
    build_request,
    parse_draft,
)
from triage_buddy.ports.llm import LLMError, LLMProvider


class TriageService:
    """Produces a ``TriageAssessment`` from a ``SymptomReport``."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def assess(self, report: SymptomReport) -> TriageAssessment:
        red_flags = detect_red_flags(report.description)
        if red_flags:
            # Hard override: a recognized emergency cue short-circuits the LLM.
            return TriageAssessment(
                level=EscalationLevel.EMERGENCY,
                rationale=(
                    "Your description contains signs that can indicate a "
                    "medical emergency: " + ", ".join(red_flags) + "."
                ),
                advice=EscalationLevel.EMERGENCY.action,
                red_flags=red_flags,
                source="safety-override",
                disclaimer=DISCLAIMER,
            )

        try:
            response = self._llm.generate(build_request(report))
            draft = parse_draft(response.text)
        except (LLMError, DraftParseError):
            return self._fallback()

        return TriageAssessment(
            level=draft.level,
            rationale=draft.rationale,
            advice=draft.advice,
            source="llm",
            disclaimer=DISCLAIMER,
        )

    @staticmethod
    def _fallback() -> TriageAssessment:
        """Conservative result used when the provider can't be trusted."""
        return TriageAssessment(
            level=EscalationLevel.URGENT,
            rationale=(
                "We couldn't complete an automated assessment of your symptoms."
            ),
            advice=(
                "To be safe, contact a healthcare provider or urgent care to "
                "describe your symptoms in person."
            ),
            source="fallback",
            disclaimer=DISCLAIMER,
        )
