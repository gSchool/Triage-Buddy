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
