"""LLM-judge grading for the eval suite's ``should`` / ``should_not`` rubrics.

Each case describes, in plain language, the *spirit* of a good answer rather
than required substrings:

* ``should`` — what good advice does (e.g. "directs the patient to urgent care").
* ``should_not`` — what it must avoid (e.g. "does not suggest waiting").

A judge model reads the assessment and answers whether each rubric holds. This
replaces literal/synonym phrase matching: the model phrases advice freely, so we
grade intent, not vocabulary. The deterministic ``expected_urgency`` check (in
``test_cases.py``) remains the hard, judge-free guarantee that emergencies
escalate; the judge only grades advice quality on top of that.

The ``mock`` provider is rejected as a judge — it keyword-matches and cannot
reason — so grading without a real provider is a hard ``JudgeUnavailable`` error
rather than a silent garbage verdict.
"""

from __future__ import annotations

import json

from triage_buddy.adapters.llm._retry import call_with_retries
from triage_buddy.composition import build_provider
from triage_buddy.ports.llm import LLMError, LLMRequest, LLMProvider


class JudgeUnavailable(Exception):
    """Raised when grading needs the LLM judge but none is configured/working."""


def _parse_verdict(reply: str) -> bool:
    """Extract the boolean ``satisfied`` from a judge reply.

    Tolerates surrounding prose / code fences by scanning for the outermost JSON
    object. Falls back to a yes/no word scan if no usable JSON is found, so a
    judge that ignores the format instruction still grades rather than erroring.
    """
    stripped = reply.strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(stripped[start : end + 1])
            if isinstance(data, dict) and "satisfied" in data:
                return bool(data["satisfied"])
        except json.JSONDecodeError:
            pass
    lowered = stripped.lower()
    if "true" in lowered or lowered.startswith("y") or "yes" in lowered:
        return True
    return False


class Judge:
    """Wraps an ``LLMProvider`` to grade rubrics about triage advice."""

    def __init__(self, provider: LLMProvider, name: str) -> None:
        self._provider = provider
        self.name = name

    def holds(self, advice: str, rubric: str) -> bool:
        """True if ``advice`` satisfies the natural-language ``rubric``.

        Asks for a small JSON verdict (the triage adapters run in JSON-object
        mode, so a JSON ask both fits that mode and keeps parsing unambiguous)
        and retries transient API errors with backoff. A persistent failure
        becomes ``JudgeUnavailable``.
        """
        system = (
            "You grade whether a piece of medical-triage advice satisfies a "
            "stated criterion. Judge the meaning, not the exact words — "
            "synonyms and paraphrases count. "
            'Respond with ONLY a JSON object: {"satisfied": true} or '
            '{"satisfied": false}.'
        )
        user = (
            f"ADVICE:\n{advice}\n\n"
            f"CRITERION: {rubric}\n\n"
            'Does the advice satisfy the criterion? Respond with JSON: '
            '{"satisfied": true|false}.'
        )
        request = LLMRequest(system=system, user=user)
        try:
            reply = call_with_retries(
                lambda: self._provider.generate(request).text,
                attempts=4,
                base_delay=1.0,
            )
        except LLMError as exc:
            raise JudgeUnavailable(f"judge {self.name!r} failed: {exc}") from exc
        return _parse_verdict(reply)


def build_judge(provider_name: str) -> Judge:
    """Build a ``Judge`` from a provider name, or raise ``JudgeUnavailable``.

    Raises rather than returning ``None`` so a missing/misconfigured judge
    surfaces immediately. The ``mock`` provider is explicitly rejected: it
    keyword-matches and cannot reason semantically, so using it as a judge would
    silently produce garbage verdicts. Run evals under a real ``--judge-provider``
    instead.
    """
    if provider_name == "mock":
        raise JudgeUnavailable(
            "the 'mock' provider cannot judge rubrics "
            "(it has no language understanding); pass --judge-provider groq|gemini"
        )
    try:
        provider = build_provider(provider_name)
    except (ValueError, LLMError) as exc:
        raise JudgeUnavailable(
            f"judge provider {provider_name!r} unavailable: {exc}"
        ) from exc
    return Judge(provider, provider_name)
