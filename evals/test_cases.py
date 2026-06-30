"""End-to-end eval suite for the triage pipeline.

These are *evals*, not unit tests: they score a provider's judgment rather than
verify code correctness. Each case in ``cases.json`` runs through the full
``TriageService`` and is checked for the expected urgency bucket plus
required/forbidden phrases.

They live in a top-level ``evals/`` directory (sibling of ``tests/``) and are
deliberately excluded from the default pytest run (``testpaths = ["tests"]``),
because:

* against the offline ``mock`` provider they won't satisfy the prose-level
  checks — the keyword-driven mock isn't meant to, so failures here are
  informative, not regressions; and
* against a real provider they make non-deterministic, billable network calls.

Run them explicitly, picking a provider via the ``EVAL_PROVIDER`` env var
(default ``mock``)::

    .venv/bin/python -m pytest evals/                       # mock
    EVAL_PROVIDER=groq .venv/bin/python -m pytest evals/ -v # real LLM

``cases.json`` is loaded relative to this file, so the suite runs from any cwd.
Imports resolve via ``pythonpath = ["src"]`` in ``pyproject.toml`` — the same
mechanism the unit tests use, so nothing needs to be installed into the wheel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from triage_buddy.composition import build_service
from triage_buddy.config import load_dotenv
from triage_buddy.domain.models import EscalationLevel, SymptomReport, TriageAssessment

# Collapse the 5-level domain enum onto the 4 buckets the eval cases use.
_BUCKET_OF_LEVEL: dict[EscalationLevel, str] = {
    EscalationLevel.SELF_CARE: "low",
    EscalationLevel.ROUTINE: "low",
    EscalationLevel.PROMPT: "medium",
    EscalationLevel.URGENT: "high",
    EscalationLevel.EMERGENCY: "emergency",
}

_BUCKETS = ("low", "medium", "high", "emergency")

_CASES_PATH = Path(__file__).with_name("cases.json")


def _load_cases() -> list[dict]:
    return json.loads(_CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def service():
    """One service for the whole module, provider chosen via ``EVAL_PROVIDER``."""
    load_dotenv()  # pick up GROQ_API_KEY (and friends) from a local .env, if present
    provider = os.environ.get("EVAL_PROVIDER", "mock")
    return build_service(provider=provider)


def _searchable_text(assessment: TriageAssessment) -> str:
    """User-visible text from an assessment, lower-cased, for substring checks.

    The standing ``disclaimer`` is deliberately excluded: it's boilerplate
    present on every assessment and itself contains the word "emergency", which
    would spuriously trip every ``must_not_contain: "emergency"`` check. The
    cases mean to test the model's *advice*, not the fixed legal text.
    """
    parts = [
        assessment.level.label,
        assessment.level.action,
        assessment.rationale,
        assessment.advice,
        *assessment.red_flags,
    ]
    return "\n".join(parts).lower()


# Parametrize over the cases, labelling each subtest by its case id.
_CASES = _load_cases()
_IDS = [c.get("id", str(i)) for i, c in enumerate(_CASES)]


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_eval_case(case: dict, service) -> None:
    report = SymptomReport(
        description=case["symptoms"],
        age=case.get("age"),
        sex=case.get("sex"),
        duration=case.get("duration"),
    )
    assessment = service.assess(report)
    text = _searchable_text(assessment)

    failures: list[str] = []

    # 1. Urgency bucket.
    expected = str(case.get("expected_urgency", "")).strip().lower()
    actual_bucket = _BUCKET_OF_LEVEL[assessment.level]
    if expected not in _BUCKETS:
        failures.append(f"case declares unknown expected_urgency {expected!r}")
    elif actual_bucket != expected:
        failures.append(
            f"urgency: expected {expected!r}, got {actual_bucket!r} "
            f"(level {assessment.level.name})"
        )

    # 2. must_contain.
    for needle in case.get("must_contain", []):
        if needle.lower() not in text:
            failures.append(f"must_contain {needle!r} MISSING")

    # 3. must_not_contain.
    for needle in case.get("must_not_contain", []):
        if needle.lower() in text:
            failures.append(f"must_not_contain {needle!r} PRESENT (should not be)")

    if failures:
        pytest.fail(
            f"{case.get('id', '?')} (level {assessment.level.name}):\n  "
            + "\n  ".join(failures)
        )
