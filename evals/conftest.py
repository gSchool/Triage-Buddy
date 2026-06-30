"""Pytest config for the eval suite: a ``--provider`` option.

Lets you pick the LLM provider on the command line, e.g.::

    pytest evals/ --provider groq

Precedence (see the ``service`` fixture in ``test_cases.py``): the ``--provider``
flag wins, then the ``EVAL_PROVIDER`` env var, then ``mock``. This conftest is
scoped to ``evals/`` so the option doesn't appear on the main ``tests/`` run.
"""

from __future__ import annotations


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--provider",
        action="store",
        default=None,
        help="LLM provider for the eval suite (overrides EVAL_PROVIDER; default mock).",
    )
