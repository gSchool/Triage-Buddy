"""Semantic matching for the eval suite's must_contain / must_not_contain checks.

The cases assert that the advice mentions a concept (e.g. ``"see a doctor"``),
but a real model phrases it freely (``"seek prompt medical attention"``,
``"consult a physician"``). Literal substring matching rejects those valid
paraphrases, so we match in three tiers, cheapest first:

1. **Literal** — exact (lower-cased) substring. Free, deterministic.
2. **Synonym group** — for a few common concepts (doctor, hospital, 911, …),
   any known equivalent phrase counts. Free, deterministic, curated here.
3. **LLM judge** — only if tiers 1–2 don't resolve it, ask a judge model whether
   the text satisfies the requirement. Handles open-ended paraphrase (e.g.
   ``"fluids"`` ~ ``"drink tea or water"``) that no synonym list could enumerate.

The judge is a ``Judge`` object (see ``build_judge``). If a needle reaches the
judge tier and no judge is configured, ``contains``/``excludes`` raise
``JudgeUnavailable`` — the suite is meant to score a real provider, so a missing
judge is a hard error, not a silent pass.
"""

from __future__ import annotations

import json

from triage_buddy.adapters.llm._retry import call_with_retries
from triage_buddy.composition import build_provider
from triage_buddy.ports.llm import LLMError, LLMRequest, LLMProvider

# Curated synonym groups for the common, closed-vocabulary concepts. Each maps a
# canonical needle to phrases that should also satisfy it. Matching is by
# substring (lower-cased), so short stems like "physician" also catch
# "physician's". Open-ended concepts (fluids, rest, …) are intentionally left to
# the judge tier — enumerating their paraphrases is a losing game.
_DOCTOR = (
    "doctor",
    "physician",
    "medical provider",
    "healthcare provider",
    "health care provider",
    "medical professional",
    "see a clinician",
    "seek medical attention",
    "seek medical care",
    "medical attention",
    "consult a",
    "get evaluated",
    "be evaluated",
    "be seen",
)

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "doctor": _DOCTOR,
    # Needle phrasings that mean the same concept as "doctor" map to the same
    # group, so e.g. must_contain "see a doctor" is satisfied by "seek medical
    # attention" without a judge call.
    "see a doctor": _DOCTOR,
    "see a physician": _DOCTOR,
    "medical": ("medical", "clinical", "healthcare", "health care"),
    "urgent": ("urgent", "promptly", "right away", "as soon as possible", "immediately"),
    "emergency": ("emergency", "emergency services", "emergency room", "er ", " ed "),
    "911": ("911", "emergency services", "emergency number", "ambulance"),
    "hospital": ("hospital", "emergency room", "er ", "urgent care"),
    "stroke": ("stroke", "fast test", "f.a.s.t"),
}


class JudgeUnavailable(Exception):
    """Raised when a check needs the LLM judge but none is configured/working."""


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
    """Wraps an ``LLMProvider`` to answer yes/no semantic questions about text."""

    def __init__(self, provider: LLMProvider, name: str) -> None:
        self._provider = provider
        self.name = name

    def satisfies(self, text: str, requirement: str) -> bool:
        """True if ``text`` semantically satisfies ``requirement`` (mentions it).

        Asks for a small JSON verdict rather than free text: the triage provider
        adapters run in JSON-object mode, and a JSON ask both fits that mode and
        keeps parsing unambiguous.
        """
        system = (
            "You grade whether a piece of medical-triage advice satisfies a "
            "requirement. The requirement is satisfied if the advice expresses "
            "the concept in ANY wording (synonyms and paraphrases count). "
            'Respond with ONLY a JSON object: {"satisfied": true} or '
            '{"satisfied": false}.'
        )
        user = (
            f"ADVICE:\n{text}\n\n"
            f'REQUIREMENT: the advice mentions or recommends "{requirement}".\n\n'
            'Does the advice satisfy the requirement? Respond with JSON: '
            '{"satisfied": true|false}.'
        )
        request = LLMRequest(system=system, user=user)
        try:
            # Retry transient failures (rate limits/timeouts under a full-run
            # burst) with backoff before giving up; persistent failure is fatal.
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
    surfaces immediately (the suite scores a real provider; a missing judge is an
    error). The ``mock`` provider is explicitly rejected: it keyword-matches and
    cannot reason semantically, so using it as a judge would silently produce
    garbage verdicts. Run evals under a real ``--judge-provider`` instead.
    """
    if provider_name == "mock":
        raise JudgeUnavailable(
            "the 'mock' provider cannot judge semantic matches "
            "(it has no language understanding); pass --judge-provider groq|gemini"
        )
    try:
        provider = build_provider(provider_name)
    except (ValueError, LLMError) as exc:
        raise JudgeUnavailable(
            f"judge provider {provider_name!r} unavailable: {exc}"
        ) from exc
    return Judge(provider, provider_name)


def _literal_or_synonym(needle: str, text: str) -> bool:
    """Tier 1+2: exact substring, or any synonym of a known concept."""
    needle_l = needle.lower()
    if needle_l in text:
        return True
    for variant in _SYNONYMS.get(needle_l, ()):
        if variant in text:
            return True
    return False


def contains(needle: str, text: str, judge: Judge) -> bool:
    """True if ``text`` contains ``needle`` literally, by synonym, or by judge.

    ``text`` is expected already lower-cased. Falls through to the judge only
    when the cheap tiers don't already confirm a match; building/using the judge
    may raise ``JudgeUnavailable`` if it isn't configured/working.
    """
    if _literal_or_synonym(needle, text):
        return True
    return judge.satisfies(text, needle)


def excludes(needle: str, text: str, judge: Judge) -> bool:
    """True if ``text`` does NOT express ``needle`` (the must_not_contain check).

    Symmetric with ``contains``: a literal or synonym hit means the forbidden
    concept is present (so excludes -> False) without a judge call; only an
    ambiguous case (no cheap hit) consults the judge.
    """
    if _literal_or_synonym(needle, text):
        return False
    return not judge.satisfies(text, needle)
