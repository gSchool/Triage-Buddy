"""CLI adapter: a thin driving adapter over the triage core.

It owns only presentation and input concerns — parsing args, reading the
description, and formatting the assessment. All triage logic lives in the core.
"""

from __future__ import annotations

import argparse
import sys

from triage_buddy.composition import build_service
from triage_buddy.config import load_dotenv
from triage_buddy.domain.models import EscalationLevel, SymptomReport, TriageAssessment
from triage_buddy.ports.llm import LLMError

# Map each level to a glyph so urgency reads at a glance in a terminal.
_GLYPHS = {
    EscalationLevel.LOW: "🟢",
    EscalationLevel.MEDIUM: "🟡",
    EscalationLevel.HIGH: "🟠",
    EscalationLevel.EMERGENCY: "🔴",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage-buddy",
        description="Get escalation advice for described symptoms.",
    )
    parser.add_argument(
        "symptoms",
        nargs="*",
        help="Symptom description. If omitted, you'll be prompted to type it.",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        help="LLM provider adapter to use (default: mock).",
    )
    return parser


def _format(assessment: TriageAssessment) -> str:
    glyph = _GLYPHS[assessment.level]
    lines = [
        f"{glyph}  {assessment.level.label.upper()} — {assessment.level.action}",
        "",
        f"Why: {assessment.rationale}",
        f"What to do: {assessment.advice}",
    ]
    if assessment.red_flags:
        lines.append("Flags: " + ", ".join(assessment.red_flags))
    lines += ["", assessment.disclaimer]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # pick up GROQ_API_KEY (and friends) from a local .env, if present
    args = _build_parser().parse_args(argv)

    description = " ".join(args.symptoms).strip()
    if not description:
        if sys.stdin.isatty():
            description = input("Describe your symptoms: ").strip()
        else:
            description = sys.stdin.read().strip()
    if not description:
        print("No symptoms provided.", file=sys.stderr)
        return 2

    try:
        report = SymptomReport(description=description)
    except ValueError as exc:
        print(f"Invalid input: {exc}", file=sys.stderr)
        return 2

    try:
        service = build_service(provider=args.provider)
    except (ValueError, LLMError) as exc:
        # ValueError: unknown provider name. LLMError: provider misconfigured
        # (e.g. missing API key). Both are setup problems, not triage outcomes.
        print(f"Provider error: {exc}", file=sys.stderr)
        return 3

    assessment = service.assess(report)
    print(_format(assessment))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
