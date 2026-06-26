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
    ("difficulty breathing", r"(can'?t breathe|trouble breathing|short(ness)? of breath|difficulty breathing)"),
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


def detect_red_flags(text: str) -> tuple[str, ...]:
    """Return descriptions of any emergency red flags found in ``text``.

    Returns an empty tuple when none match. The result is ordered as the rules
    are declared, and de-duplicated.
    """
    found: list[str] = []
    for description, pattern in _COMPILED:
        if pattern.search(text) and description not in found:
            found.append(description)
    return tuple(found)
