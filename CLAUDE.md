# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Triage Buddy is a web and CLI app that provides escalation advice for medical symptoms (i.e. how urgently a user should seek care).

## Tech stack

- **Language:** Python (>=3.10), `src/` layout, packaged via `pyproject.toml` (setuptools).
- **Core + first adapters:** standard library only — no third-party runtime dependency for the mock slice.
- **Tests:** `pytest`.
- **Virtual environment:** `.venv` in the repo root. Always use it (`.venv/bin/python`).
- Real LLM SDKs are **optional extras** (e.g. `pip install -e ".[anthropic]"`), kept out of the core.

## Commands

```bash
# One-time setup
python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"

# Run the test suite (pythonpath=src is configured in pyproject)
.venv/bin/python -m pytest -q

# Run a single test
.venv/bin/python -m pytest tests/test_triage.py::test_red_flag_forces_emergency_and_skips_llm

# Run the CLI (mock provider, default — offline, no key)
.venv/bin/triage-buddy "mild sore throat for two days"
.venv/bin/triage-buddy --age 34 --duration "3 days" "high fever that won't go away"
# or without install: PYTHONPATH=src .venv/bin/python -m triage_buddy.adapters.cli.app "..."

# Run against Groq (Llama). Needs the [groq] extra and a key:
.venv/bin/python -m pip install -e ".[groq]"
cp .env.example .env   # then put your GROQ_API_KEY in .env (git-ignored)
.venv/bin/triage-buddy --provider groq "persistent cough and mild fever for three days"

# Run the web server (browser form + JSON API)
.venv/bin/triage-buddy-web --port 8000 --provider groq   # default host 127.0.0.1, provider mock
#   GET  /         browser form
#   POST /triage   JSON API: {"description": "...", "age": 40, "sex": "...", "duration": "..."}
#   GET  /healthz  liveness check
```

### Configuration / secrets

Secrets load from a git-ignored `.env` in the repo root via `triage_buddy.config.load_dotenv`,
called at CLI startup. A real exported env var always overrides `.env`. Template: `.env.example`.
- `GROQ_API_KEY` — required for `--provider groq`.

There is no lint config yet.

## Architecture (hexagonal — as implemented)

Keep domain/core logic free of framework, transport, and provider concerns; push those to the edges (adapters).

- **`src/triage_buddy/domain/`** — the core. No I/O, no SDKs.
  - `models.py`: `EscalationLevel` (ordered IntEnum: SELF_CARE→EMERGENCY), `SymptomReport`, `TriageAssessment`.
  - `safety.py`: deterministic red-flag detection + the standing `DISCLAIMER`. Runs *in front of* the LLM.
  - `triage.py`: `TriageService.assess()` — the use case. Safety-first order: red-flag override → LLM suggestion → conservative fallback. The LLM can only escalate, never overrule a recognized red flag.
- **`src/triage_buddy/ports/llm.py`** — `LLMProvider` port: a generic `generate(LLMRequest) -> LLMResponse` text contract. No triage knowledge lives here. `LLMError` signals provider failure.
- **`src/triage_buddy/prompts.py`** — builds the request from a `SymptomReport` and parses the JSON reply into a `TriageDraft`. Knows the wire shape; imports no SDK.
- **`src/triage_buddy/adapters/`**
  - `llm/mock.py`: `MockLLMProvider` — deterministic, offline, keyword-driven. First adapter behind the port; speaks the same JSON shape a real provider would.
  - `llm/groq.py`: `GroqProvider` — Groq-hosted Llama (`llama-3.3-70b-versatile`), JSON mode, `temperature=0`. Optional `[groq]` extra; SDK imported lazily. API failures → `LLMError` (core fails safe).
  - `cli/app.py`: CLI driving adapter (argparse). Presentation only.
  - `web/`: web driving adapter on the stdlib `http.server` (no framework dep).
    - `service.py`: transport-agnostic request handling — `run_triage(...)` (validate → assess → `(status, dict)`) and `assessment_to_dict`. Shared by both web surfaces.
    - `app.py`: HTTP handler + HTML rendering + server entry (`triage-buddy-web`). Routes: `GET /` form, `POST /` form result, `POST /triage` JSON API, `GET /healthz`. User input is HTML-escaped; request body capped at 64 KiB.
- **`src/triage_buddy/config.py`** — `load_dotenv()` (stdlib, no dep), called by adapter entry points.
- **`src/triage_buddy/composition.py`** — composition root. The only place that picks concrete adapters (`build_service`/`build_provider`). Providers: `mock`, `groq`.

### Extending

- **New LLM provider:** add an adapter under `adapters/llm/` implementing `LLMProvider.generate`, then add a branch in `composition.build_provider`. Core, CLI, and web stay untouched.
- **New presentation surface:** add a driving adapter that calls `TriageService` (CLI and web are the existing examples). For HTTP surfaces, reuse `adapters/web/service.run_triage` so validation/serialization stay shared.

## Safety notes

This is medical-adjacent software. Two invariants must hold:
1. Recognized red flags force `EMERGENCY` deterministically, independent of any model.
2. Every `TriageAssessment` carries the disclaimer, and provider failures fail *safe* (conservative `URGENT` fallback, never a crash or silent downgrade).

## Build plan (from README)

1. Settle on a plan. ✅
2. Start with the core logic. ✅ (`domain/`)
3. Plumbing: tech stack + testing tools. ✅ (Python, pytest, `.venv`)
4. Determine the ports and adapters. ✅ LLM port; driven adapters: `mock`, `groq` (Llama). Driving adapters: CLI, web (form + JSON API).
