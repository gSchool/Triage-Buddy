"""Transport-agnostic request handling for the web adapter.

These pure functions take already-parsed inputs and return a status code plus a
plain dict. Both the JSON API and the HTML form go through ``run_triage`` so the
two surfaces can never drift apart, and the logic is trivially unit-testable
without spinning up a server.
"""

from __future__ import annotations

from typing import Any

from triage_buddy.composition import build_service
from triage_buddy.domain.models import SymptomReport, TriageAssessment
from triage_buddy.ports.llm import LLMError


def assessment_to_dict(assessment: TriageAssessment) -> dict[str, Any]:
    """Serialize an assessment to a JSON-friendly dict."""
    return {
        "level": assessment.level.name,
        "label": assessment.level.label,
        "action": assessment.level.action,
        "rationale": assessment.rationale,
        "advice": assessment.advice,
        "red_flags": list(assessment.red_flags),
        "source": assessment.source,
        "disclaimer": assessment.disclaimer,
    }


def run_triage(
    *,
    description: str | None,
    age: Any = None,
    sex: str | None = None,
    duration: str | None = None,
    provider: str = "mock",
) -> tuple[int, dict[str, Any]]:
    """Validate inputs, run triage, and return ``(http_status, body)``.

    Returns 400 for bad input, 503 for a provider/config problem, and 200 with
    the serialized assessment on success.
    """
    if not description or not description.strip():
        return 400, {"error": "description is required"}

    age_value, age_error = _coerce_age(age)
    if age_error is not None:
        return 400, {"error": age_error}

    try:
        report = SymptomReport(
            description=description,
            age=age_value,
            sex=(sex or None),
            duration=(duration or None),
        )
    except ValueError as exc:
        return 400, {"error": str(exc)}

    try:
        service = build_service(provider=provider)
    except (ValueError, LLMError) as exc:
        # Unknown provider name, or a misconfigured provider (e.g. missing key).
        return 503, {"error": f"provider unavailable: {exc}"}

    assessment = service.assess(report)
    return 200, assessment_to_dict(assessment)


def _coerce_age(age: Any) -> tuple[int | None, str | None]:
    """Normalize an age that may arrive as int, numeric string, or empty."""
    if age is None or age == "":
        return None, None
    try:
        return int(age), None
    except (TypeError, ValueError):
        return None, "age must be a whole number"
