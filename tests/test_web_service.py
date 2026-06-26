"""Tests for the transport-agnostic web request handling."""

from triage_buddy.adapters.web.service import (
    assessment_to_dict,
    provider_health,
    run_triage,
)
from triage_buddy.domain.models import EscalationLevel, TriageAssessment


def test_run_triage_success_with_mock():
    status, body = run_triage(description="mild runny nose", provider="mock")
    assert status == 200
    assert body["level"] == "SELF_CARE"
    assert body["disclaimer"]
    assert body["source"] == "llm"


def test_run_triage_red_flag_overrides():
    status, body = run_triage(description="severe chest pain", provider="mock")
    assert status == 200
    assert body["level"] == "EMERGENCY"
    assert body["source"] == "safety-override"
    assert body["red_flags"]


def test_run_triage_missing_description():
    status, body = run_triage(description="  ")
    assert status == 400
    assert "description" in body["error"]


def test_run_triage_bad_age():
    status, body = run_triage(description="cough", age="old")
    assert status == 400
    assert "age" in body["error"]


def test_run_triage_age_coerced_from_string():
    status, _ = run_triage(description="cough", age="40", provider="mock")
    assert status == 200


def test_run_triage_unknown_provider_is_503():
    status, body = run_triage(description="cough", provider="nope")
    assert status == 503
    assert "provider" in body["error"]


def test_provider_health_ok_for_mock():
    status, body = provider_health("mock")
    assert status == 200
    assert body == {"status": "ok", "provider": "mock"}


def test_provider_health_unknown_provider_is_503():
    status, body = provider_health("nope")
    assert status == 503
    assert body["status"] == "unavailable"
    assert body["provider"] == "nope"


def test_assessment_to_dict_shape():
    a = TriageAssessment(
        level=EscalationLevel.URGENT,
        rationale="r",
        advice="a",
        red_flags=("x",),
        source="llm",
        disclaimer="d",
    )
    d = assessment_to_dict(a)
    assert d == {
        "level": "URGENT",
        "label": "Urgent",
        "action": EscalationLevel.URGENT.action,
        "rationale": "r",
        "advice": "a",
        "red_flags": ["x"],
        "source": "llm",
        "disclaimer": "d",
    }
