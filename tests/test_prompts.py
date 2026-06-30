import json

import pytest

from triage_buddy.domain.models import EscalationLevel, SymptomReport
from triage_buddy.prompts import DraftParseError, build_request, parse_draft


def test_build_request_carries_description_and_system_prompt():
    req = build_request(SymptomReport(description="  cough  "))
    assert "Symptoms: cough" in req.user  # description is stripped
    assert "JSON" in req.system


def test_parse_draft_plain_json():
    text = json.dumps(
        {"urgency": "medium", "recommendation": "see a doctor", "disclaimer": "d"}
    )
    draft = parse_draft(text)
    assert draft.level is EscalationLevel.MEDIUM
    assert draft.advice == "see a doctor"


@pytest.mark.parametrize(
    "bucket,level",
    [
        ("low", EscalationLevel.LOW),
        ("medium", EscalationLevel.MEDIUM),
        ("high", EscalationLevel.HIGH),
        ("emergency", EscalationLevel.EMERGENCY),
    ],
)
def test_parse_draft_maps_each_urgency_bucket(bucket, level):
    text = json.dumps({"urgency": bucket, "recommendation": "a", "disclaimer": "d"})
    assert parse_draft(text).level is level


def test_parse_draft_ignores_model_disclaimer():
    # The model's disclaimer is parsed-and-discarded; the domain appends its own.
    text = json.dumps(
        {"urgency": "low", "recommendation": "rest", "disclaimer": "anything"}
    )
    draft = parse_draft(text)
    assert not hasattr(draft, "disclaimer")
    assert draft.advice == "rest"


def test_parse_draft_tolerates_code_fence_and_prose():
    text = '```json\n{"urgency": "high", "recommendation": "a", "disclaimer": "d"}\n```'
    assert parse_draft(text).level is EscalationLevel.HIGH


def test_parse_draft_disclaimer_optional():
    # Only urgency + recommendation are required by the parser.
    text = json.dumps({"urgency": "low", "recommendation": "rest"})
    assert parse_draft(text).level is EscalationLevel.LOW


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        json.dumps({"urgency": "high"}),  # missing recommendation
        json.dumps({"recommendation": "a"}),  # missing urgency
        json.dumps({"urgency": "whenever", "recommendation": "a"}),  # bad bucket
        json.dumps({"urgency": "high", "recommendation": ""}),  # empty recommendation
        json.dumps(["not", "an", "object"]),
    ],
)
def test_parse_draft_rejects_malformed(text):
    with pytest.raises(DraftParseError):
        parse_draft(text)
