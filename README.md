# Triage Buddy

Escalation advice for medical symptoms — *how urgently* you should seek care, with
a rationale and a recommended next step. Available as a CLI and a web app (browser
form + JSON API).

It is **decision support, not diagnosis.** Every assessment carries a disclaimer,
and the design fails safe: a recognized emergency phrase always returns `EMERGENCY`
regardless of what any model says, and a provider outage degrades to a conservative
`HIGH` recommendation rather than a crash or a silent downgrade.

## How it works

Given a symptom description, Triage Buddy returns
an assessment with an **escalation level**, in increasing severity:

```
LOW → MEDIUM → HIGH → EMERGENCY
```

The final level is the **more severe of two signals**:

1. A deterministic **safety floor**. Recognized red flags (chest pain, difficulty
   breathing, stroke signs, suicidal ideation, …) floor at `EMERGENCY` and
   short-circuit — no model is called, so true emergencies are instant and
   independent of provider availability. Detection skips *explicitly denied*
   symptoms ("no chest pain, no shortness of breath"), but conservatively: when a
   negation is ambiguous the flag stands, so the floor may over-triage but never
   misses a real red flag.
2. An **LLM suggestion** for everything else. The model can *escalate* above the
   floor but can never lower the result below it.

If the provider errors or returns unparseable output, the result falls back to a
conservative `HIGH` (taken as the max with the floor). See
[docs/requirements.md](docs/requirements.md) for the full behavior spec in EARS
format, cross-referenced to the tests that verify each requirement.

## Quick start

The core and the default (mock) provider are pure Python standard library, so a
fresh clone runs offline with no third-party installs and no API key.

```bash
# One-time setup
python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"

# CLI (mock provider — offline, deterministic, no key)
.venv/bin/triage-buddy "mild sore throat for two days"
.venv/bin/triage-buddy "high fever that won't go away"

# Web server (FastAPI: browser form at / + JSON API at /triage)
# The [dev] install above already includes the [web] extra (FastAPI + uvicorn).
.venv/bin/triage-buddy-web --port 8000        # default host 127.0.0.1, provider mock
```

### Web endpoints

| Method | Path       | Description                                                        |
| ------ | ---------- | ------------------------------------------------------------------ |
| `GET`  | `/`        | Browser symptom form                                               |
| `POST` | `/`        | Form-encoded submit → form re-rendered with the assessment         |
| `POST` | `/triage`  | JSON API: `{"description": "..."}` |
| `GET`  | `/healthz` | Cached provider health: `200` reachable / `503` misconfigured or unreachable |

## Using a real LLM provider

Real LLM SDKs are **optional extras** kept out of the core. Install the one you want
and supply its key:

```bash
# Groq (Llama 3.3 70B)
.venv/bin/python -m pip install -e ".[groq]"
cp .env.example .env          # then put GROQ_API_KEY in .env (git-ignored)
.venv/bin/triage-buddy --provider groq "persistent cough and mild fever for three days"

# Google Gemini (gemini-2.5-flash)
.venv/bin/python -m pip install -e ".[gemini]"
# put GEMINI_API_KEY in .env
.venv/bin/triage-buddy --provider gemini "persistent cough and mild fever for three days"

# Z.ai (glm-4.6)
.venv/bin/python -m pip install -e ".[zai]"
# put ZAI_API_KEY in .env
.venv/bin/triage-buddy --provider zai "persistent cough and mild fever for three days"
```

Secrets load from a git-ignored `.env` in the repo root at startup (a real exported
env var always wins over `.env`). Template: [.env.example](.env.example).

If a real provider is selected but its SDK isn't installed or its key is missing, it
fails with a clear configuration error at startup — never as a silent triage
fallback. Providers also get bounded per-request timeouts and retry-with-backoff.

## Tests

```bash
.venv/bin/python -m pytest -q                              # full suite
.venv/bin/python -m pytest tests/test_triage.py::test_red_flag_forces_emergency_and_skips_llm
```

## Evals

An end-to-end eval suite runs a set of symptom scenarios through the full triage
pipeline and checks each one against the expected urgency plus a natural-language
rubric for the advice. It lives in a top-level `evals/` directory (sibling of
`tests/`) and is written as a pytest module, but is **excluded from the default
`pytest` run** — run it explicitly. Pick the provider with `--provider` (or the
`EVAL_PROVIDER` env var); default is `mock`:

```bash
.venv/bin/python -m pytest evals/ --provider groq -v             # evaluate groq
EVAL_PROVIDER=groq .venv/bin/python -m pytest evals/ -v          # same, via env var
.venv/bin/python -m pytest evals/ --provider groq --judge-provider gemini  # judge with a different model
```

Evals are *not* tests: they score a provider's judgment rather than verify code
correctness, so failures are informative rather than regressions, and they're kept
out of the CI gate.

Each case in `cases.json` carries an `expected_urgency` plus `should` / `should_not`
rubrics that describe, in plain language, the *spirit* of a good answer (e.g.
`should: "directs the patient to urgent, same-day care"`, `should_not: "downplays it
as self-care"`) — not required substrings. Two kinds of check run per case:

- **`expected_urgency`** — an exact, deterministic level match (no model involved).
  This is the hard guard, so a flaky judge can never let a non-escalated emergency pass.
- **`should` / `should_not`** — graded by an **LLM judge** that reads the model's advice
  and answers whether each rubric holds, judging meaning rather than wording.

The judge defaults to `--provider` but can be set with `--judge-provider` (or
`EVAL_JUDGE_PROVIDER`). Because grading needs real language understanding, the `mock`
provider **cannot** judge — under `--provider mock`, grading is a hard error. In short:
these cases score a *real* provider (`groq` / `gemini` / `zai`, with a key set), not the
offline mock. Note that scores vary run to run — the model is non-deterministic even
at temperature 0, mostly on the urgency level.

## Architecture

Hexagonal (ports & adapters). The domain core holds no framework, transport, or SDK
imports; everything outward-facing is an adapter behind a port. Adding a provider or
a presentation surface does not touch the core.

```
domain/        core: escalation levels, red-flag safety, max-of-both triage rule
ports/llm.py   the LLMProvider port (a generic text generate() contract)
prompts.py     builds the request from a SymptomReport, parses the JSON reply
adapters/
  llm/         mock (default, offline), groq (Llama), gemini, zai (GLM) — behind the port
  cli/         CLI driving adapter
  web/         FastAPI app + Jinja2 template: form, JSON API, /healthz
composition.py composition root — the only place concrete adapters are chosen
```

The `tests/` and `evals/` directories sit outside the package (not shipped in the
wheel): `tests/` is the pytest suite that gates correctness; `evals/` scores model
judgment and runs on demand (see [Evals](#evals)).

- **Add an LLM provider:** implement `LLMProvider.generate` under `adapters/llm/`,
  then add a branch in `composition.build_provider`. Core, CLI, and web are untouched.
- **Add a presentation surface:** add a driving adapter that calls `TriageService`
  (CLI and web are the examples); HTTP surfaces reuse `adapters/web/service.run_triage`.

See [CLAUDE.md](CLAUDE.md) for a fuller architecture walkthrough and the safety
invariants.

## Safety invariants

This is medical-adjacent software. Two invariants must always hold:

1. The final level is the **max** of the deterministic safety floor and the LLM
   suggestion — the model may escalate, never lower below the floor. Recognized red
   flags floor at `EMERGENCY` independent of any model.
2. Every assessment carries the disclaimer, and provider failures fail *safe*
   (conservative `HIGH` fallback, never a crash or silent downgrade).

The red-flag list and escalation mapping are intentionally conservative and **have
not undergone clinical validation** — see *Out of Scope* and *Open Questions* in
[docs/requirements.md](docs/requirements.md).
```

