import json

import pytest

from triage_buddy.domain.models import EscalationLevel, SymptomReport
from triage_buddy.prompts import DraftParseError, build_request, parse_draft


def test_build_request_includes_optional_context():
    report = SymptomReport(description="cough", age=40, sex="F", duration="3 days")
    req = build_request(report)
    assert "cough" in req.user
    assert "Age: 40" in req.user
    assert "Sex: F" in req.user
    assert "Duration: 3 days" in req.user
    assert "JSON" in req.system


def test_build_request_omits_absent_context():
    req = build_request(SymptomReport(description="cough"))
    assert "Age:" not in req.user


def test_parse_draft_plain_json():
    text = json.dumps({"level": "PROMPT", "rationale": "r", "advice": "a"})
    draft = parse_draft(text)
    assert draft.level is EscalationLevel.PROMPT
    assert draft.rationale == "r"


def test_parse_draft_tolerates_code_fence_and_prose():
    text = '```json\n{"level": "URGENT", "rationale": "r", "advice": "a"}\n```'
    assert parse_draft(text).level is EscalationLevel.URGENT


@pytest.mark.parametrize(
    "text",
    [
        "not json at all",
        json.dumps({"level": "URGENT", "rationale": "r"}),  # missing advice
        json.dumps({"level": "WHENEVER", "rationale": "r", "advice": "a"}),  # bad level
        json.dumps({"level": "URGENT", "rationale": "", "advice": "a"}),  # empty
        json.dumps(["not", "an", "object"]),
    ],
)
def test_parse_draft_rejects_malformed(text):
    with pytest.raises(DraftParseError):
        parse_draft(text)
