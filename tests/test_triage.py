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
    stub = StubProvider(json.dumps({"level": "SELF_CARE", "rationale": "r", "advice": "a"}))
    service = TriageService(llm=stub)

    assessment = service.assess(_report("I have crushing chest pain"))

    assert assessment.level is EscalationLevel.EMERGENCY
    assert assessment.source == "safety-override"
    assert assessment.red_flags
    assert stub.calls == 0  # the LLM is never consulted on a red flag


def test_uses_llm_suggestion_when_no_red_flag():
    stub = StubProvider(json.dumps({"level": "PROMPT", "rationale": "r", "advice": "a"}))
    assessment = TriageService(llm=stub).assess(_report())
    assert assessment.level is EscalationLevel.PROMPT
    assert assessment.source == "llm"


def test_falls_back_when_provider_errors():
    assessment = TriageService(llm=BoomProvider()).assess(_report())
    assert assessment.level is EscalationLevel.URGENT
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
    assert mild.level is EscalationLevel.SELF_CARE


# --- max-of-both-signals: an injected intermediate floor proves the mechanism --

def _floor_urgent(_description):
    return EscalationLevel.URGENT, ("test floor",)


def test_floor_raises_under_calling_llm():
    # LLM says PROMPT, floor says URGENT -> the more severe (URGENT) wins.
    stub = StubProvider(json.dumps({"level": "PROMPT", "rationale": "r", "advice": "a"}))
    assessment = TriageService(llm=stub, floor=_floor_urgent).assess(_report())
    assert assessment.level is EscalationLevel.URGENT
    assert assessment.source == "safety-override"
    assert assessment.red_flags == ("test floor",)
    assert stub.calls == 1  # the LLM was still consulted (floor below ceiling)


def test_llm_above_floor_is_kept():
    # LLM says EMERGENCY, floor says URGENT -> LLM (more severe) is used.
    stub = StubProvider(json.dumps({"level": "EMERGENCY", "rationale": "r", "advice": "a"}))
    assessment = TriageService(llm=stub, floor=_floor_urgent).assess(_report())
    assert assessment.level is EscalationLevel.EMERGENCY
    assert assessment.source == "llm"


def test_llm_equal_to_floor_is_kept_as_llm():
    stub = StubProvider(json.dumps({"level": "URGENT", "rationale": "r", "advice": "a"}))
    assessment = TriageService(llm=stub, floor=_floor_urgent).assess(_report())
    assert assessment.level is EscalationLevel.URGENT
    assert assessment.source == "llm"


def test_floor_applies_even_when_llm_fails():
    # Provider fails; floor (URGENT) and fallback (URGENT) agree -> URGENT.
    assessment = TriageService(llm=BoomProvider(), floor=_floor_urgent).assess(_report())
    assert assessment.level is EscalationLevel.URGENT
