"""Deterministic safety rules that sit *in front of* the LLM.

In a medical setting we never want a probabilistic model to be the only thing
standing between a patient and an emergency. These rules run independently of
the LLM: if a recognized red-flag phrase appears, the core forces an
``EMERGENCY`` assessment regardless of what any provider says.

The list is intentionally conservative and non-exhaustive — it is a safety net,
not a diagnostic engine.
"""

from __future__ import annotations

import re

from triage_buddy.domain.models import EscalationLevel

# A constant disclaimer attached to every assessment. Triage Buddy is decision
# support, not a diagnosis or a substitute for professional care.
DISCLAIMER = (
    "Triage Buddy provides general guidance only and is not a medical diagnosis "
    "or a substitute for professional care. If you think this may be an "
    "emergency, call your local emergency number immediately."
)

# (human-readable description, regex pattern). Patterns use word boundaries to
# avoid matching substrings inside unrelated words.
_RED_FLAGS: tuple[tuple[str, str], ...] = (
    ("chest pain or pressure", r"chest (pain|pressure|tightness)"),
    ("difficulty breathing", r"(can'?t breathe|can'?t catch (my )?breath|trouble breathing|short(ness)? of breath|difficulty breathing|wheezing|gasping)"),
    ("signs of stroke", r"(droop|slurred|slur(red)? speech|sudden numbness|sudden weakness|can'?t move (one|an? )?(side|arm|leg))"),
    ("severe bleeding", r"(severe|heavy|uncontrolled) bleeding|won'?t stop bleeding"),
    ("loss of consciousness", r"(passed out|fainted|unconscious|unresponsive|loss of consciousness)"),
    ("suicidal ideation", r"(suicidal|kill myself|end my life|want to die)"),
    ("anaphylaxis", r"(throat (closing|swelling)|anaphylaxis|severe allergic reaction)"),
    ("seizure", r"(seizure|convulsions?)"),
    ("coughing or vomiting blood", r"(coughing up|vomiting) blood|blood in (my )?vomit"),
)

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (desc, re.compile(pattern, re.IGNORECASE)) for desc, pattern in _RED_FLAGS
)

# Negation cues and clause-resetting boundaries, for ``_is_negated``. A red-flag
# match is treated as *denied* only when one of these cues sits just before it
# with no boundary in between, so "no chest pain" is dropped but "no fever, but
# chest pain" is kept.
_NEGATION_CUE = re.compile(r"\b(no|not|n't|without|deny|denies|denied|negative for)\b", re.IGNORECASE)
_CLAUSE_RESET = re.compile(r"[.;,]|\b(but|however|although|though)\b", re.IGNORECASE)

# How far back (characters) to look for a negating cue before a match.
_NEGATION_LOOKBACK = 40


def _is_negated(text: str, match: re.Match[str]) -> bool:
    """True if the red-flag ``match`` is denied by a preceding negation.

    Conservative by design: a match counts as negated only when a negation cue
    appears within ``_NEGATION_LOOKBACK`` characters *immediately* before it and
    no clause boundary (``,`` ``;`` ``.`` ``but`` …) intervenes to reset it. So
    "no chest pain" is negated, but "no fever, but chest pain" is not — the comma
    and "but" cancel the "no" before "chest pain". When in doubt this returns
    ``False`` (not negated → the flag stands), preserving the fail-safe rule that
    the floor may over-triage but must never miss a real red flag.
    """
    window = text[max(0, match.start() - _NEGATION_LOOKBACK) : match.start()]
    last_cue: re.Match[str] | None = None
    for cue in _NEGATION_CUE.finditer(window):
        last_cue = cue
    if last_cue is None:
        return False
    # A clause reset between the cue and the match cancels the negation.
    return not _CLAUSE_RESET.search(window[last_cue.end() :])


def detect_red_flags(text: str) -> tuple[str, ...]:
    """Return descriptions of any emergency red flags found in ``text``.

    A pattern counts only when it appears *un-negated*: an explicitly denied
    symptom ("no chest pain, no shortness of breath") does not floor the result.
    See ``_is_negated`` for the deliberately conservative negation rule. Returns
    an empty tuple when none match; ordered as declared, de-duplicated.
    """
    found: list[str] = []
    for description, pattern in _COMPILED:
        if description in found:
            continue
        if any(not _is_negated(text, m) for m in pattern.finditer(text)):
            found.append(description)
    return tuple(found)


def severity_floor(text: str) -> tuple[EscalationLevel, tuple[str, ...]]:
    """The deterministic *minimum* severity for a description, with reasons.

    The core takes the more severe of this floor and the LLM's suggestion, so a
    floor can only ever raise the result, never lower it.

    Today this encodes a single rule: a recognized red flag floors the result at
    ``EMERGENCY``. Intermediate floors (e.g. ``HIGH``/``MEDIUM`` for specific
    concerning-but-not-emergency phrases) are a deliberate extension point —
    pending clinical review, add them here and the max-of-both logic honors them
    automatically. Absent any rule, the floor is ``LOW`` (the minimum), which
    imposes no constraint on the LLM.
    """
    red = detect_red_flags(text)
    if red:
        return EscalationLevel.EMERGENCY, red
    return EscalationLevel.LOW, ()
