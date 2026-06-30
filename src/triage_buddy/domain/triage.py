"""The triage service: the application's core use case.

It combines two signals and returns the **more severe** of the two:

1. A deterministic *severity floor* (``safety.severity_floor``) — rules that
   cannot be lowered by a model (today: red flags → ``EMERGENCY``).
2. The LLM provider's suggestion (via the port).

Taking the max means the LLM can only ever *escalate* relative to the floor,
never pull the result below it. A conservative fallback covers provider failure.

Optimization & safety: when the floor is already at the maximum severity
(``EMERGENCY``), no LLM answer could raise it, so we skip the call entirely —
keeping true emergencies instant and independent of provider availability.
"""

from __future__ import annotations

from collections.abc import Callable

from triage_buddy.domain.models import (
    EscalationLevel,
    SymptomReport,
    TriageAssessment,
)
from triage_buddy.domain.safety import DISCLAIMER, severity_floor
from triage_buddy.prompts import (
    DraftParseError,
    build_request,
    parse_draft,
)
from triage_buddy.ports.llm import LLMError, LLMProvider

# Level used when the LLM can't be trusted (error or unparseable reply).
# HIGH, not EMERGENCY: conservative (don't sit on it) without auto-dialing 911.
_FALLBACK_LEVEL = EscalationLevel.HIGH

FloorFn = Callable[[str], "tuple[EscalationLevel, tuple[str, ...]]"]


class TriageService:
    """Produces a ``TriageAssessment`` from a ``SymptomReport``."""

    def __init__(self, llm: LLMProvider, *, floor: FloorFn = severity_floor) -> None:
        self._llm = llm
        self._floor = floor  # injectable for testing and future floor rules

    def assess(self, report: SymptomReport) -> TriageAssessment:
        floor_level, reasons = self._floor(report.description)

        # Floor already at the ceiling: the LLM cannot raise it. Skip the call.
        if floor_level is EscalationLevel.EMERGENCY:
            return self._raised_to_floor(floor_level, reasons)

        try:
            draft = parse_draft(self._llm.generate(build_request(report)).text)
        except (LLMError, DraftParseError):
            return self._fallback(max(floor_level, _FALLBACK_LEVEL))

        # max-of-both: keep the LLM's assessment when it meets or exceeds the
        # floor; otherwise the deterministic floor wins.
        if draft.level >= floor_level:
            return TriageAssessment(
                level=draft.level,
                rationale=draft.rationale,
                advice=draft.advice,
                red_flags=reasons,
                source="llm",
                disclaimer=DISCLAIMER,
            )
        return self._raised_to_floor(floor_level, reasons)

    @staticmethod
    def _raised_to_floor(
        level: EscalationLevel, reasons: tuple[str, ...]
    ) -> TriageAssessment:
        """Result when a deterministic floor sets (or raises to) the level."""
        if level is EscalationLevel.EMERGENCY:
            if reasons:
                rationale = (
                    "Your description contains signs that can indicate a medical "
                    "emergency: " + ", ".join(reasons) + "."
                )
            else:
                rationale = "This has been escalated to an emergency on safety grounds."
        elif reasons:
            rationale = "Safety rules raised the urgency based on: " + ", ".join(reasons) + "."
        else:
            rationale = "Safety rules set a minimum urgency above the automated assessment."

        return TriageAssessment(
            level=level,
            rationale=rationale,
            advice=level.action,
            red_flags=reasons,
            source="safety-override",
            disclaimer=DISCLAIMER,
        )

    @staticmethod
    def _fallback(level: EscalationLevel) -> TriageAssessment:
        """Conservative result used when the provider can't be trusted."""
        return TriageAssessment(
            level=level,
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
