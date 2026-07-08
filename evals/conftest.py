"""Pytest config for the eval suite: ``--provider``, ``--judge-provider``,
``--samples`` options.

    pytest evals/ --provider groq --judge-provider gemini --samples 5

Each option falls back to an env var then a default (see the fixtures in
``test_cases.py``): ``--provider``→``EVAL_PROVIDER``→``mock``,
``--judge-provider``→``EVAL_JUDGE_PROVIDER``→``--provider``,
``--samples``→``EVAL_SAMPLES``→``1``. This conftest is scoped to ``evals/`` so
the options don't appear on the main ``tests/`` run.
"""

from __future__ import annotations


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--provider",
        action="store",
        default=None,
        help="LLM provider for the eval suite (overrides EVAL_PROVIDER; default mock).",
    )
    parser.addoption(
        "--judge-provider",
        action="store",
        default=None,
        help=(
            "LLM provider used to semantically grade should/should_not rubrics "
            "(overrides EVAL_JUDGE_PROVIDER; defaults to --provider)."
        ),
    )
    parser.addoption(
        "--samples",
        action="store",
        type=int,
        default=None,
        help=(
            "How many times to run each case and majority-vote the urgency "
            "(overrides EVAL_SAMPLES; default 1, i.e. voting disabled). The model "
            "is non-deterministic, so raising N gives a stabler score at the cost "
            "of N full assessments per case."
        ),
    )
