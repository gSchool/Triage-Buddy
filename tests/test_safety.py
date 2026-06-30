import pytest

from triage_buddy.domain.safety import detect_red_flags


@pytest.mark.parametrize(
    "text",
    [
        "I have severe chest pain radiating to my arm",
        "I can't breathe and feel dizzy",
        "my face is drooping and my speech is slurred",
        "I think I want to die",
        "she passed out and is unresponsive",
    ],
)
def test_detects_emergencies(text):
    assert detect_red_flags(text)


@pytest.mark.parametrize(
    "text",
    ["mild runny nose for two days", "slight headache after work", "sore throat"],
)
def test_no_false_positives_on_mild_symptoms(text):
    assert detect_red_flags(text) == ()


def test_returns_deduplicated_descriptions():
    flags = detect_red_flags("chest pain and chest tightness")
    assert flags.count("chest pain or pressure") == 1


@pytest.mark.parametrize(
    "text",
    [
        "no chest pain, no shortness of breath",
        "no fever that I've measured, no chest pain, no shortness of breath",
        "denies chest pain or difficulty breathing",
        "I do not have any chest pain",
    ],
)
def test_explicitly_denied_symptoms_do_not_floor(text):
    # A negated red flag ("no chest pain") must not force EMERGENCY.
    assert detect_red_flags(text) == ()


@pytest.mark.parametrize(
    "text",
    [
        # A clause reset (comma / "but") after the negation re-arms the flag.
        "no fever, but chest pain that started an hour ago",
        "I'm not sure if it's nothing but I have crushing chest pain",
        # Plain affirmative red flags are unaffected by the negation handling.
        "sudden weakness on one side and slurred speech",
    ],
)
def test_negation_handling_never_suppresses_a_real_red_flag(text):
    # Fail-safe asymmetry: when negation is ambiguous, the flag must stand.
    assert detect_red_flags(text)
