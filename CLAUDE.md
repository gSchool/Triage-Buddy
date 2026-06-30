# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Triage Buddy is a web and CLI app that provides escalation advice for medical symptoms (i.e. how urgently a user should seek care).

## Tech stack

- **Language:** Python (>=3.10), `src/` layout, packaged via `pyproject.toml` (setuptools).
- **Dependencies:** prefer the standard library for the core and the default (mock) slice, so a fresh clone runs offline with no installs. Third-party runtime deps are allowed when they earn their place (e.g. a real web framework, a parser with edge cases worth not owning) — they're a deliberate choice, not a banned one. Today the core + mock slice happen to be pure stdlib.
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

# Run the end-to-end eval suite (top-level evals/, NOT collected by the default run).
# Provider via --provider / EVAL_PROVIDER (default mock); these score a model, not the code.
# Advice is graded against natural-language should/should_not rubrics by an LLM judge,
# so they need a real provider — mock cannot judge, so grading under mock is a hard error.
.venv/bin/python -m pytest evals/ --provider groq -v
.venv/bin/python -m pytest evals/ --provider groq --judge-provider gemini  # judge with a different model

# Run the CLI (mock provider, default — offline, no key)
.venv/bin/triage-buddy "mild sore throat for two days"
.venv/bin/triage-buddy "high fever that won't go away"
# or without install: PYTHONPATH=src .venv/bin/python -m triage_buddy.adapters.cli.app "..."

# Run against Groq (Llama). Needs the [groq] extra and a key:
.venv/bin/python -m pip install -e ".[groq]"
cp .env.example .env   # then put your GROQ_API_KEY in .env (git-ignored)
.venv/bin/triage-buddy --provider groq "persistent cough and mild fever for three days"

# Or Google Gemini (needs the [gemini] extra and GEMINI_API_KEY in .env):
.venv/bin/python -m pip install -e ".[gemini]"
.venv/bin/triage-buddy --provider gemini "persistent cough and mild fever for three days"

# Run the web server (FastAPI: browser form + JSON API; needs the [web] extra, included in [dev])
.venv/bin/python -m pip install -e ".[web]"              # fastapi + uvicorn (skip if you installed [dev])
.venv/bin/triage-buddy-web --port 8000 --provider groq   # default host 127.0.0.1, provider mock
#   GET  /         browser form
#   POST /triage   JSON API: {"description": "..."}
#   GET  /healthz  liveness check
```

### Configuration / secrets

Secrets load from a git-ignored `.env` in the repo root via `triage_buddy.config.load_dotenv`,
called at entry-point startup (CLI and web; the eval suite calls it too). A real exported env var always overrides `.env`. Template: `.env.example`.
- `GROQ_API_KEY` — required for `--provider groq`.

There is no lint config yet.

## Architecture (hexagonal — as implemented)

Keep domain/core logic free of framework, transport, and provider concerns; push those to the edges (adapters).

- **`src/triage_buddy/domain/`** — the core. No I/O, no SDKs.
  - `models.py`: `EscalationLevel` (ordered IntEnum: `LOW`→`MEDIUM`→`HIGH`→`EMERGENCY`, matching the wire vocabulary 1:1), `SymptomReport`, `TriageAssessment`.
  - `safety.py`: red-flag detection, the standing `DISCLAIMER`, and `severity_floor()` (deterministic minimum severity + reasons; today red flags → `EMERGENCY`, else `LOW`). The injectable seam for future intermediate floors. Detection is negation-aware (`_is_negated`): an explicitly denied symptom ("no chest pain") doesn't floor, but a clause reset re-arms it ("no fever, but chest pain") and ambiguous negation keeps the flag — the rule is deliberately conservative so the floor may over-triage but never misses a real red flag.
  - `triage.py`: `TriageService.assess()` — the use case. **Max-of-both-signals**: returns the more severe of `severity_floor` and the LLM suggestion (`floor` is injectable via the constructor). When the floor is already `EMERGENCY` the LLM is skipped (can't raise it; keeps emergencies instant). Provider failure → conservative `HIGH` fallback (not `EMERGENCY` — don't auto-dial 911 on an outage), taken as max with the floor.
- **`src/triage_buddy/ports/llm.py`** — `LLMProvider` port: a generic `generate(LLMRequest) -> LLMResponse` text contract. No triage knowledge lives here. `LLMError` signals provider failure.
- **`src/triage_buddy/prompts.py`** — builds the request from a `SymptomReport` and parses the JSON reply into a `TriageDraft`. Knows the wire shape; imports no SDK. The wire schema is `{urgency, recommendation, disclaimer}`; `urgency` (`low`/`medium`/`high`/`emergency`) maps 1:1 onto `EscalationLevel` by name via `from_name` (no bucketing table). `recommendation` becomes the draft's advice, a rationale is synthesized (the model no longer returns one), and the model's `disclaimer` is parsed-and-discarded — the domain appends its own standing `DISCLAIMER`, and `severity_floor` (not the model) remains the source of truth for red flags. The system prompt also instructs the model on emergency flagging, pediatric escalation, and recommendation wording, but the domain invariants hold regardless of whether the model obeys.
- **`src/triage_buddy/adapters/`**
  - `llm/mock.py`: `MockLLMProvider` — deterministic, offline, keyword-driven. First adapter behind the port; speaks the same JSON shape a real provider would.
  - `llm/_retry.py`: `call_with_retries()` — shared retry-with-exponential-backoff wrapper (injectable `sleep`). Real adapters run each request through it; defaults: 30s timeout, 3 attempts, 0.5s base delay.
  - `llm/groq.py`: `GroqProvider` — Groq-hosted Llama (`llama-3.3-70b-versatile`), JSON mode, `temperature=0`. Per-request `timeout` on the client (SDK's own `max_retries=0` — this adapter owns retries). Optional `[groq]` extra; SDK imported lazily. API failures → `LLMError` (core fails safe).
  - `llm/gemini.py`: `GeminiProvider` — Google Gemini (`gemini-2.5-flash`) via `google-genai`, JSON mode, `temperature=0`. Per-request timeout via `HttpOptions(timeout=ms)`; retries via `_retry`. Optional `[gemini]` extra; SDK imported lazily. `GEMINI_API_KEY`. API failures → `LLMError`.
  - `cli/app.py`: CLI driving adapter (argparse). Presentation only.
  - `web/`: web driving adapter built on **FastAPI** with **Jinja2** templates (the `[web]` extra: `fastapi` + `uvicorn` + `jinja2`, included in `[dev]`).
    - `service.py`: transport-agnostic request handling — `run_triage(...)` (validate → assess → `(status, dict)`), `assessment_to_dict`, `provider_health(name)` (build + cheap probe → `(200|503, dict)`), and `ProviderHealthCache` (per-provider TTL cache, thread-safe, injectable clock). Pure stdlib; knows nothing about FastAPI. Shared by both web surfaces.
    - `app.py`: `create_app(provider, *, health_ttl)` FastAPI app factory + `render_page(...)` (renders `templates/page.html` via a module-level autoescaping Jinja2 env) + server entry (`triage-buddy-web`, `--health-ttl`; serves via uvicorn, imported lazily in `main`). Routes: `GET /` form, `POST /` form result (raw urlencoded body, no `python-multipart` dep), `POST /triage` JSON API (raw body parsed by hand so a non-object/invalid body → 400, not FastAPI's 422), `GET /healthz` (cached provider health: 200 reachable / 503 misconfigured or unreachable). One health cache is shared across requests. User input is HTML-escaped (Jinja2 autoescape); request body capped at 64 KiB (→ 413).
    - `templates/page.html`: the single Jinja2 page template (form + error card + result card + styles). Packaged via `[tool.setuptools.package-data]`.
- **`src/triage_buddy/config.py`** — `load_dotenv()` (stdlib, no dep), called by adapter entry points.
- **`src/triage_buddy/composition.py`** — composition root. The only place that picks concrete adapters (`build_service`/`build_provider`). Providers: `mock`, `groq`, `gemini`.
- **`evals/`** (top-level, outside the package — like `tests/`) — end-to-end eval suite. `test_cases.py` is a pytest module parametrized over `cases.json`. Each case carries an `expected_urgency` plus `should`/`should_not` rubrics describing the *spirit* of good advice in plain language (not required substrings), and is checked two ways: (1) **`expected_urgency`** — exact, deterministic level match (`EscalationLevel` name == bucket, 1:1), judge-free; this is the hard guard, so a flaky judge can never pass a non-escalated emergency. (2) **`should`/`should_not`** — graded by an **LLM judge** (`_match.py`) that reads the model's rationale+advice and answers whether each rubric holds, judging meaning not wording. The judge asks for a JSON `{"satisfied": bool}` verdict (adapters are in JSON mode) and retries transient errors; `should_not` is phrased as the *bad behavior to detect* (judge YES → violated). Judge provider is `--judge-provider` (or `EVAL_JUDGE_PROVIDER`), defaulting to `--provider`; `mock` is rejected as a judge, so grading under `--provider mock` is a hard error. **Evals are not tests** — they score a model's judgment, not code correctness, so they're excluded from the default run (`testpaths = ["tests"]`) and run on demand via `pytest evals/ --provider groq`. Scores vary run to run (model is non-deterministic even at temperature 0, mostly on the urgency level). Imports resolve via `pythonpath = ["src"]`; nothing is shipped in the wheel.

### Extending

- **New LLM provider:** add an adapter under `adapters/llm/` implementing `LLMProvider.generate`, then add a branch in `composition.build_provider`. Core, CLI, and web stay untouched.
- **New presentation surface:** add a driving adapter that calls `TriageService` (CLI and web are the existing examples). For HTTP surfaces, reuse `adapters/web/service.run_triage` so validation/serialization stay shared.

## Safety notes

This is medical-adjacent software. Two invariants must hold:
1. The final level is the *max* of the deterministic `severity_floor` and the LLM's suggestion — the model can escalate but never lower the result below the floor. Recognized red flags floor at `EMERGENCY` independent of any model.
2. Every `TriageAssessment` carries the disclaimer, and provider failures fail *safe* (conservative `HIGH` fallback, never a crash or silent downgrade).

## Build plan (from README)

1. Settle on a plan. ✅
2. Start with the core logic. ✅ (`domain/`)
3. Plumbing: tech stack + testing tools. ✅ (Python, pytest, `.venv`)
4. Determine the ports and adapters. ✅ LLM port; driven adapters: `mock`, `groq` (Llama), `gemini`. Driving adapters: CLI, web (form + JSON API).
