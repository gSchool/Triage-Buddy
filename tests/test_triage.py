import json

from triage_buddy.adapters.llm.mock import MockLLMProvider
from triage_buddy.domain.models import EscalationLevel, SymptomReport
from triage_buddy.domain.triage import TriageService
from triage_buddy.ports.llm import LLMError, LLMRequest, LLMResponse


class StubProvider:
    """A provider returning a fixed reply, to test the core in isolation."""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text=self.text)


class BoomProvider:
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMError("provider down")


def _report(desc="I have a sore throat"):
    return SymptomReport(description=desc)


def test_red_flag_forces_emergency_and_skips_llm():
    stub = StubProvider(json.dumps({"urgency": "low", "recommendation": "a", "disclaimer": "d"}))
    service = TriageService(llm=stub)

    assessment = service.assess(_report("I have crushing chest pain"))

    assert assessment.level is EscalationLevel.EMERGENCY
    assert assessment.source == "safety-override"
    assert assessment.red_flags
    assert stub.calls == 0  # the LLM is never consulted on a red flag


def test_uses_llm_suggestion_when_no_red_flag():
    stub = StubProvider(json.dumps({"urgency": "medium", "recommendation": "a", "disclaimer": "d"}))
    assessment = TriageService(llm=stub).assess(_report())
    assert assessment.level is EscalationLevel.MEDIUM
    assert assessment.source == "llm"


def test_falls_back_when_provider_errors():
    assessment = TriageService(llm=BoomProvider()).assess(_report())
    assert assessment.level is EscalationLevel.HIGH  # conservative, not EMERGENCY
    assert assessment.source == "fallback"


def test_falls_back_when_reply_is_garbage():
    assessment = TriageService(llm=StubProvider("¯\\_(ツ)_/¯")).assess(_report())
    assert assessment.source == "fallback"


def test_every_assessment_carries_a_disclaimer():
    assessment = TriageService(llm=MockLLMProvider()).assess(_report())
    assert assessment.disclaimer


def test_end_to_end_with_mock_provider():
    service = TriageService(llm=MockLLMProvider())
    mild = service.assess(_report("mild runny nose"))
    assert mild.level is EscalationLevel.LOW


# --- max-of-both-signals: an injected intermediate floor proves the mechanism --

def _floor_high(_description):
    return EscalationLevel.HIGH, ("test floor",)


def test_floor_raises_under_calling_llm():
    # LLM says MEDIUM, floor says HIGH -> the more severe (HIGH) wins.
    stub = StubProvider(json.dumps({"urgency": "medium", "recommendation": "a", "disclaimer": "d"}))
    assessment = TriageService(llm=stub, floor=_floor_high).assess(_report())
    assert assessment.level is EscalationLevel.HIGH
    assert assessment.source == "safety-override"
    assert assessment.red_flags == ("test floor",)
    assert stub.calls == 1  # the LLM was still consulted (floor below ceiling)


def test_llm_above_floor_is_kept():
    # LLM says EMERGENCY, floor says HIGH -> LLM (more severe) is used.
    stub = StubProvider(json.dumps({"urgency": "emergency", "recommendation": "a", "disclaimer": "d"}))
    assessment = TriageService(llm=stub, floor=_floor_high).assess(_report())
    assert assessment.level is EscalationLevel.EMERGENCY
    assert assessment.source == "llm"


def test_llm_equal_to_floor_is_kept_as_llm():
    stub = StubProvider(json.dumps({"urgency": "high", "recommendation": "a", "disclaimer": "d"}))
    assessment = TriageService(llm=stub, floor=_floor_high).assess(_report())
    assert assessment.level is EscalationLevel.HIGH
    assert assessment.source == "llm"


def test_floor_applies_even_when_llm_fails():
    # Provider fails; floor (HIGH) and fallback (HIGH) agree -> HIGH.
    assessment = TriageService(llm=BoomProvider(), floor=_floor_high).assess(_report())
    assert assessment.level is EscalationLevel.HIGH
