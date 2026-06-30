import pytest

from triage_buddy.domain.models import EscalationLevel, SymptomReport


def test_levels_are_ordered_by_severity():
    assert EscalationLevel.EMERGENCY > EscalationLevel.HIGH
    assert EscalationLevel.LOW < EscalationLevel.MEDIUM
    assert max(EscalationLevel) is EscalationLevel.EMERGENCY


def test_level_has_label_and_action():
    assert EscalationLevel.HIGH.label == "High"
    assert "today" in EscalationLevel.HIGH.action.lower()


@pytest.mark.parametrize(
    "name,expected",
    [("high", EscalationLevel.HIGH), ("EMERGENCY", EscalationLevel.EMERGENCY), (" low ", EscalationLevel.LOW)],
)
def test_from_name_is_case_and_space_insensitive(name, expected):
    assert EscalationLevel.from_name(name) is expected


def test_from_name_rejects_unknown():
    with pytest.raises(ValueError):
        EscalationLevel.from_name("kinda urgent")


def test_symptom_report_rejects_empty_description():
    with pytest.raises(ValueError):
        SymptomReport(description="   ")


def test_symptom_report_rejects_negative_age():
    with pytest.raises(ValueError):
        SymptomReport(description="headache", age=-1)
