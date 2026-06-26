import pytest

from triage_buddy.domain.models import EscalationLevel, SymptomReport


def test_levels_are_ordered_by_severity():
    assert EscalationLevel.EMERGENCY > EscalationLevel.URGENT
    assert EscalationLevel.SELF_CARE < EscalationLevel.ROUTINE
    assert max(EscalationLevel) is EscalationLevel.EMERGENCY


def test_level_has_label_and_action():
    assert EscalationLevel.URGENT.label == "Urgent"
    assert "today" in EscalationLevel.URGENT.action.lower()


@pytest.mark.parametrize(
    "name,expected",
    [("urgent", EscalationLevel.URGENT), ("EMERGENCY", EscalationLevel.EMERGENCY), (" self_care ", EscalationLevel.SELF_CARE)],
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
