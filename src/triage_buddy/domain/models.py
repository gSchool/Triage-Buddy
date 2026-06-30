"""Domain models for triage.

These are plain, framework-free value objects. They carry no knowledge of how
a report arrives (CLI, web) or how an assessment is produced (which LLM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class EscalationLevel(IntEnum):
    """How urgently the patient should seek care.

    Ordered by severity so levels compare directly (``EMERGENCY > LOW``), which
    lets the core take the *more severe* of two signals when combining
    deterministic safety rules with an LLM's suggestion.

    The names match the wire vocabulary the model speaks (``low``/``medium``/
    ``high``/``emergency``), so a parsed urgency maps straight to a member via
    ``from_name`` with no intermediate bucketing.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EMERGENCY = 4

    @property
    def label(self) -> str:
        return _LEVEL_LABELS[self]

    @property
    def action(self) -> str:
        """A short, imperative description of the recommended action."""
        return _LEVEL_ACTIONS[self]

    @classmethod
    def from_name(cls, name: str) -> "EscalationLevel":
        """Parse a level from a (case-insensitive) name, e.g. ``"high"``.

        Raises ``ValueError`` for anything unrecognized so callers can decide
        on a conservative fallback rather than silently guessing.
        """
        try:
            return cls[name.strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unknown escalation level: {name!r}") from exc


_LEVEL_LABELS = {
    EscalationLevel.LOW: "Low",
    EscalationLevel.MEDIUM: "Medium",
    EscalationLevel.HIGH: "High",
    EscalationLevel.EMERGENCY: "Emergency",
}

_LEVEL_ACTIONS = {
    EscalationLevel.LOW: "Manage at home with rest and monitor your symptoms.",
    EscalationLevel.MEDIUM: "See a doctor for evaluation in the next day or two.",
    EscalationLevel.HIGH: "Seek prompt medical attention today — visit urgent care or call your provider now.",
    EscalationLevel.EMERGENCY: "Call emergency services (911) or go to the ER immediately.",
}


@dataclass(frozen=True)
class SymptomReport:
    """A patient's described symptoms."""

    description: str

    def __post_init__(self) -> None:
        if not self.description or not self.description.strip():
            raise ValueError("symptom description must not be empty")


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
