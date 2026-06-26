"""Prompt construction and reply parsing.

This bridges the domain's ``SymptomReport`` to the generic LLM port and back. It
lives just outside ``domain`` because it knows the wire shape we ask providers
for (a small JSON object), but it imports no provider SDK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from triage_buddy.domain.models import EscalationLevel, SymptomReport
from triage_buddy.ports.llm import LLMRequest

_LEVEL_NAMES = ", ".join(level.name for level in EscalationLevel)

SYSTEM_PROMPT = f"""\
You are a cautious medical triage assistant. Given a patient's described \
symptoms, judge how urgently they should seek care. You do not diagnose.

When unsure, escalate: prefer the more urgent level. Account for any age, sex, \
and duration provided.

Respond with ONLY a JSON object, no prose, in exactly this shape:
{{
  "level": "<one of: {_LEVEL_NAMES}>",
  "rationale": "<one or two sentences explaining the level>",
  "advice": "<concrete next step for the patient>"
}}"""


@dataclass(frozen=True)
class TriageDraft:
    """The parsed, still-untrusted suggestion from a provider."""

    level: EscalationLevel
    rationale: str
    advice: str


class DraftParseError(Exception):
    """Raised when a provider reply can't be parsed into a ``TriageDraft``."""


def build_request(report: SymptomReport) -> LLMRequest:
    """Turn a ``SymptomReport`` into a provider-agnostic ``LLMRequest``."""
    parts = [f"Symptoms: {report.description.strip()}"]
    if report.age is not None:
        parts.append(f"Age: {report.age}")
    if report.sex:
        parts.append(f"Sex: {report.sex}")
    if report.duration:
        parts.append(f"Duration: {report.duration}")
    return LLMRequest(system=SYSTEM_PROMPT, user="\n".join(parts))


def parse_draft(text: str) -> TriageDraft:
    """Parse a provider reply into a ``TriageDraft``.

    Tolerates a leading/trailing code fence and surrounding whitespace, since
    models often wrap JSON in ```json fences. Raises ``DraftParseError`` on any
    malformed or incomplete reply so the core can fail safe.
    """
    payload = _extract_json(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DraftParseError(f"reply was not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise DraftParseError("reply JSON was not an object")

    try:
        level = EscalationLevel.from_name(str(data["level"]))
        rationale = str(data["rationale"]).strip()
        advice = str(data["advice"]).strip()
    except (KeyError, ValueError) as exc:
        raise DraftParseError(str(exc)) from exc

    if not rationale or not advice:
        raise DraftParseError("rationale and advice must be non-empty")

    return TriageDraft(level=level, rationale=rationale, advice=advice)


def _extract_json(text: str) -> str:
    """Strip optional markdown code fences and return the JSON substring."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence (optionally ```json) and the closing fence.
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
        stripped = stripped.strip()
    # Fall back to the outermost braces if there's stray text around the object.
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped
