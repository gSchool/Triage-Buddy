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

from _match import Judge, build_judge, contains, excludes

# The domain enum names match the cases' urgency buckets one-to-one, so the
# bucket is just the level name lower-cased — no mapping table needed.
_BUCKETS = tuple(level.name.lower() for level in EscalationLevel)

_CASES_PATH = Path(__file__).with_name("cases.json")


def _load_cases() -> list[dict]:
    return json.loads(_CASES_PATH.read_text(encoding="utf-8"))


def _provider_name(request) -> str:
    """Provider for the system under test: --provider, EVAL_PROVIDER, then mock."""
    return (
        request.config.getoption("--provider")
        or os.environ.get("EVAL_PROVIDER")
        or "mock"
    )


def _judge_provider_name(request) -> str:
    """Provider for the judge: --judge-provider, EVAL_JUDGE_PROVIDER, then the SUT."""
    return (
        request.config.getoption("--judge-provider")
        or os.environ.get("EVAL_JUDGE_PROVIDER")
        or _provider_name(request)
    )


@pytest.fixture(scope="module")
def service(request):
    """One service for the whole module (provider per ``_provider_name``)."""
    load_dotenv()  # pick up GROQ_API_KEY (and friends) from a local .env, if present
    return build_service(provider=_provider_name(request))


class _LazyJudge:
    """A ``Judge``-shaped wrapper that builds the real judge on first use, so
    runs where every needle is satisfied literally/by synonym need no judge key.

    Exposes ``satisfies`` so it drops straight into ``_match.contains/excludes``;
    building the underlying judge may raise ``JudgeUnavailable``."""

    def __init__(self, provider_name: str) -> None:
        self._provider_name = provider_name
        self._judge: Judge | None = None

    def satisfies(self, text: str, requirement: str) -> bool:
        if self._judge is None:
            self._judge = build_judge(self._provider_name)  # raises JudgeUnavailable
        return self._judge.satisfies(text, requirement)


@pytest.fixture(scope="module")
def judge(request):
    """A lazily-built judge, used only when literal/synonym tiers don't resolve."""
    load_dotenv()
    return _LazyJudge(_judge_provider_name(request))


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
def test_eval_case(case: dict, service, judge) -> None:
    report = SymptomReport(description=case["symptoms"])
    assessment = service.assess(report)
    text = _searchable_text(assessment)

    failures: list[str] = []

    # 1. Urgency bucket.
    expected = str(case.get("expected_urgency", "")).strip().lower()
    actual_bucket = assessment.level.name.lower()
    if expected not in _BUCKETS:
        failures.append(f"case declares unknown expected_urgency {expected!r}")
    elif actual_bucket != expected:
        failures.append(
            f"urgency: expected {expected!r}, got {actual_bucket!r} "
            f"(level {assessment.level.name})"
        )

    # 2. must_contain — literal, then synonym, then LLM judge (semantic).
    for needle in case.get("must_contain", []):
        if not contains(needle, text, judge):
            failures.append(f"must_contain {needle!r} MISSING (no literal/synonym/semantic match)")

    # 3. must_not_contain — same tiers, inverted.
    for needle in case.get("must_not_contain", []):
        if not excludes(needle, text, judge):
            failures.append(f"must_not_contain {needle!r} PRESENT (should not be)")

    if failures:
        pytest.fail(
            f"{case.get('id', '?')} (level {assessment.level.name}):\n  "
            + "\n  ".join(failures)
        )
