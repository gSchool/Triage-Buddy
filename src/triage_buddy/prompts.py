"""Prompt construction and reply parsing.

This bridges the domain's ``SymptomReport`` to the generic LLM port and back. It
lives just outside ``domain`` because it knows the wire shape we ask providers
for (a small JSON object), but it imports no provider SDK.

The wire's 4-bucket urgency (``low``/``medium``/``high``/``emergency``) matches
the ``EscalationLevel`` member names one-to-one, so a parsed urgency maps
straight to a level via ``from_name`` — no intermediate bucketing. The model is
also *asked* to flag emergencies and emit a disclaimer, but neither is trusted
here: the domain's deterministic safety floor remains the source of truth for
red flags, and the domain appends the standing disclaimer — so the model's
``disclaimer`` field is parsed-and-discarded.

The model returns both a ``rationale`` (why this urgency) and a
``recommendation`` (what to do). Splitting them gives the model a dedicated place
for its reasoning so symptom-specific justification doesn't get crowded out of,
or omitted from, the recommendation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from triage_buddy.domain.models import EscalationLevel, SymptomReport
from triage_buddy.ports.llm import LLMRequest

SYSTEM_PROMPT = """\
You are a medical triage assistant. A patient will describe their symptoms, \
and you assess how urgently they should seek care and what they should do next.

Always respond with valid JSON matching exactly this schema:
{
  "urgency": one of "low", "medium", "high", or "emergency",
  "rationale": a short string explaining WHY this urgency level fits the \
described symptoms, referencing the specific symptoms given,
  "recommendation": a short string describing what the patient should do next,
  "disclaimer": a string that explicitly states this is not a substitute for \
professional medical advice
}

Rules you must always follow:
- Set "urgency" to exactly one of: "low", "medium", "high", "emergency".
- The "rationale" must explain why the urgency fits the specific symptoms \
described (e.g. the fever's duration, that it isn't responding to medication), \
not a generic restatement of the urgency level.
- Never diagnose a condition. Only suggest appropriate next steps (e.g. rest \
and monitor, see a doctor, go to urgent care, call emergency services).
- The "disclaimer" field must always include the phrase "not a substitute for \
professional medical advice".
- For any symptoms that suggest a medical emergency — including chest pain, \
difficulty breathing, or stroke symptoms (such as face drooping, arm weakness, \
or slurred speech) — always set "urgency" to "emergency" and recommend calling \
emergency services immediately.
- For mild, self-limiting illness (e.g. common cold, minor sore throat), the \
recommendation should advise the patient to rest and drink plenty of fluids.
- Regardless of urgency level, the recommendation must reference the specific \
symptoms described (e.g. mention the fever's duration or that medication isn't \
holding it down) rather than a generic "see a doctor for evaluation".
- If a patient reports feeling unwell or "not right" but gives no specific or \
severe symptoms, set "urgency" to "medium" and recommend they see a doctor for \
evaluation. Do not dismiss vague but real concerns as "low".
- This vague-symptom rule does not apply when the patient names a specific, \
benign, self-explained cause (e.g. tiredness attributed to poor sleep, soreness \
from exercise) and reports no other symptoms — treat that as "low", explicitly \
acknowledge the cause the patient named, and give cause-appropriate self-care \
advice (e.g. better sleep habits for poor sleep) plus monitoring for new or \
worsening symptoms, rather than recommending a doctor visit.
- Never use the word "wait" in the recommendation. For emergencies, instruct \
the patient to call 911 or emergency services immediately.
- Only use the word "emergency" in the recommendation when "urgency" is \
"emergency". For lower urgencies, phrase escalation as "seek prompt medical \
attention" or "see a doctor" instead.
- When the patient is an infant or child, first assess the urgency as you would \
for an adult with the same symptoms, then raise it by one level (low -> medium, \
medium -> high, high -> emergency; "emergency" is already the maximum). This \
applies only when the child is the patient: if a child is merely mentioned but \
the symptoms described are an adult's own, do not escalate."""

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
    return LLMRequest(system=SYSTEM_PROMPT, user=f"Symptoms: {report.description.strip()}")


def parse_draft(text: str) -> TriageDraft:
    """Parse a provider reply into a ``TriageDraft``.

    Expects the wire schema ``{"urgency", "rationale", "recommendation",
    "disclaimer"}``. The ``urgency`` maps directly onto an ``EscalationLevel`` by
    name; ``rationale`` becomes the draft's rationale and ``recommendation`` its
    advice. ``rationale`` is optional — when the model omits it, a generic
    level-derived sentence is synthesized so the draft is always complete. The
    model's ``disclaimer`` is intentionally ignored — the domain appends its own
    standing disclaimer.

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
        urgency = str(data["urgency"]).strip()
        advice = str(data["recommendation"]).strip()
    except KeyError as exc:
        raise DraftParseError(f"missing field: {exc}") from exc

    try:
        level = EscalationLevel.from_name(urgency)
    except ValueError as exc:
        raise DraftParseError(str(exc)) from exc

    if not advice:
        raise DraftParseError("recommendation must be non-empty")

    # rationale is optional: when the model omits it (or sends it empty), fall
    # back to a generic level-derived sentence so the draft is always complete.
    rationale = str(data.get("rationale", "")).strip() or (
        f"The described symptoms suggest a {level.label.lower()} urgency level of care."
    )
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
