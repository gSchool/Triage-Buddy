"""Domain models for triage.

These are plain, framework-free value objects. They carry no knowledge of how
a report arrives (CLI, web) or how an assessment is produced (which LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class EscalationLevel(IntEnum):
    """How urgently the patient should seek care.

    Ordered by severity so levels compare directly (``EMERGENCY > ROUTINE``),
    which lets the core take the *more severe* of two signals when combining
    deterministic safety rules with an LLM's suggestion.
    """

    SELF_CARE = 1
    ROUTINE = 2
    PROMPT = 3
    URGENT = 4
    EMERGENCY = 5

    @property
    def label(self) -> str:
        return _LEVEL_LABELS[self]

    @property
    def action(self) -> str:
        """A short, imperative description of the recommended action."""
        return _LEVEL_ACTIONS[self]

    @classmethod
    def from_name(cls, name: str) -> "EscalationLevel":
        """Parse a level from a (case-insensitive) name, e.g. ``"urgent"``.

        Raises ``ValueError`` for anything unrecognized so callers can decide
        on a conservative fallback rather than silently guessing.
        """
        try:
            return cls[name.strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unknown escalation level: {name!r}") from exc


_LEVEL_LABELS = {
    EscalationLevel.SELF_CARE: "Self-care",
    EscalationLevel.ROUTINE: "Routine",
    EscalationLevel.PROMPT: "Prompt",
    EscalationLevel.URGENT: "Urgent",
    EscalationLevel.EMERGENCY: "Emergency",
}

_LEVEL_ACTIONS = {
    EscalationLevel.SELF_CARE: "Manage at home and monitor your symptoms.",
    EscalationLevel.ROUTINE: "Schedule a routine appointment with your provider.",
    EscalationLevel.PROMPT: "See a healthcare provider within the next 24–48 hours.",
    EscalationLevel.URGENT: "Seek care today — visit urgent care or call your provider now.",
    EscalationLevel.EMERGENCY: "Call emergency services (911) or go to the ER immediately.",
}


@dataclass(frozen=True)
class SymptomReport:
    """A patient's described symptoms plus optional structured context."""

    description: str
    age: int | None = None
    sex: str | None = None
    duration: str | None = None

    def __post_init__(self) -> None:
        if not self.description or not self.description.strip():
            raise ValueError("symptom description must not be empty")
        if self.age is not None and self.age < 0:
            raise ValueError("age must not be negative")


@dataclass(frozen=True)
class TriageAssessment:
    """The result returned to the caller.

    ``source`` records how the level was decided, which matters in a medical
    context for auditing and for explaining safety overrides to the user.
    """

    level: EscalationLevel
    rationale: str
    advice: str
    red_flags: tuple[str, ...] = ()
    source: str = "llm"  # "llm" | "safety-override" | "fallback"
    disclaimer: str = field(default="")
