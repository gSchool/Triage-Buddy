# Requirements Document: Triage Buddy

## Overview

Triage Buddy accepts a patient's described symptoms and returns a triage
**assessment**: how urgently the patient should seek care, with a rationale and a
recommended next step. It is decision support, not diagnosis.

These requirements describe the system **as built** (core + LLM port with `mock`
and `groq` adapters + CLI and web presentation adapters), written in EARS format
so each is testable. Existing automated tests are cross-referenced where they
verify a requirement.

## User Roles

- **Patient / end user** — describes symptoms and receives escalation advice, via
  either the CLI or the web form.
- **API client** — a program that POSTs symptoms to the JSON API and consumes a
  structured assessment.
- **Operator** — runs the CLI or web server and selects/configures the LLM
  provider (including secrets).

## Definitions

- **Escalation level** — one of, in increasing severity: `LOW`, `MEDIUM`,
  `HIGH`, `EMERGENCY` (matching the model's wire vocabulary).
- **Red flag** — a recognized phrase indicating a potential emergency (e.g. chest
  pain, difficulty breathing, stroke signs, suicidal ideation).
- **Assessment** — `{ level, rationale, advice, red_flags, source, disclaimer }`,
  where `source ∈ { llm, safety-override, fallback }`.

---

## Requirements

### Requirement 1: Symptom intake & validation

**User Story:** As a patient, I want to submit a description of my symptoms, so
that I can receive escalation advice.

**Acceptance Criteria:**
1. WHEN a request provides a non-empty symptom description THEN the system SHALL
   produce an assessment for it.
2. IF the symptom description is empty or whitespace-only THEN the system SHALL
   reject the request with a validation error and SHALL NOT produce an assessment.

**Edge Cases:**
- Empty description on the CLI with no piped input → prompt or error (see Req 6).

*Verified by:* `test_domain.py`, `test_web_service.py::test_run_triage_missing_description`.

---

### Requirement 2: Deterministic emergency safety override

**User Story:** As a patient describing a life-threatening symptom, I want an
immediate emergency recommendation, so that a model error or outage can never
downgrade a true emergency.

**Acceptance Criteria:**
1. WHEN a symptom description contains a recognized red flag THEN the system SHALL
   return level `EMERGENCY` with `source = "safety-override"`.
2. WHEN a red flag is detected THEN the system SHALL NOT call the LLM provider.
3. WHEN a red flag is detected THEN the system SHALL include the matched red-flag
   description(s) in the assessment.
4. WHEN multiple distinct red flags match THEN the system SHALL list each at most
   once (de-duplicated).

**Edge Cases:**
- Red-flag detection is case-insensitive and tolerant of word order
  (e.g. "face is drooping", "speech is slurred").
- The red-flag list is intentionally conservative and non-exhaustive — a safety
  net, not a diagnostic engine.

*Verified by:* `test_safety.py`, `test_triage.py::test_red_flag_forces_emergency_and_skips_llm`,
`test_web_http.py::test_json_api_triage`.

---

### Requirement 3: LLM-backed assessment (non-emergency path)

**User Story:** As a patient with non-emergency symptoms, I want a considered
urgency level and advice, so that I know how soon to seek care.

**Acceptance Criteria:**
1. WHEN the deterministic severity floor is below `EMERGENCY` THEN the system
   SHALL request an assessment from the configured LLM provider through the port.
2. WHEN the provider returns a well-formed reply whose level meets or exceeds the
   floor THEN the system SHALL return that level, rationale, and advice, with
   `source = "llm"`.
3. WHEN building the request THEN the system SHALL instruct the provider to choose
   exactly one of the defined escalation levels and SHALL prefer the more urgent
   level when uncertain.

**Edge Cases:**
- The LLM may only ever *escalate* relative to the floor — it can never lower the
  result below it (see Req 10).

*Verified by:* `test_triage.py::test_uses_llm_suggestion_when_no_red_flag`,
`::test_llm_above_floor_is_kept`, `::test_llm_equal_to_floor_is_kept_as_llm`,
`test_prompts.py`.

---

### Requirement 4: Fail-safe behavior on provider failure or bad output

**User Story:** As a patient, I want a safe, sensible response even when the AI
service fails, so that I am never left with a crash or silently wrong advice.

**Acceptance Criteria:**
1. IF the LLM provider raises an error (network, auth, rate limit) THEN the system
   SHALL return a conservative assessment with level `HIGH` and
   `source = "fallback"`.
2. IF the provider reply cannot be parsed into a valid urgency + recommendation
   THEN the system SHALL return the same conservative fallback.
3. WHEN a fallback occurs THEN the system SHALL advise contacting a healthcare
   provider and SHALL NOT crash.

**Edge Cases:**
- Malformed JSON, a missing `urgency`/`recommendation` field, an empty
  recommendation, an unknown urgency bucket, or a non-object reply all trigger
  the fallback. (The wire reply is `{urgency, recommendation, disclaimer}`; the
  `urgency` bucket — `low`/`medium`/`high`/`emergency` — is mapped onto the
  `EscalationLevel` enum, and the model's `disclaimer` is discarded in favor of
  the standing one.)
- A provider reply wrapped in a ```` ```json ```` code fence is still parsed
  successfully (not a failure).

*Verified by:* `test_triage.py::test_falls_back_when_provider_errors`,
`::test_falls_back_when_reply_is_garbage`, `test_prompts.py::test_parse_draft_rejects_malformed`.

---

### Requirement 5: Disclaimer on every assessment

**User Story:** As a patient, I want to be reminded this is not a diagnosis, so
that I seek professional care appropriately.

**Acceptance Criteria:**
1. WHEN the system returns any assessment (override, LLM, or fallback) THEN it
   SHALL include the standing medical disclaimer.

*Verified by:* `test_triage.py::test_every_assessment_carries_a_disclaimer`,
`test_web_service.py`.

---

### Requirement 6: CLI presentation adapter

**User Story:** As a patient at a terminal, I want to get advice from the command
line, so that I can use Triage Buddy without a browser.

**Acceptance Criteria:**
1. WHEN symptoms are passed as command-line arguments THEN the CLI SHALL assess
   them and print a formatted result, exiting `0`.
2. WHEN no symptoms are passed AND input is interactive THEN the CLI SHALL prompt
   for a description; WHEN input is piped THEN the CLI SHALL read it from stdin.
3. IF no symptom text is ultimately provided THEN the CLI SHALL print an error to
   stderr and exit `2`.
4. IF the selected provider is unknown or misconfigured THEN the CLI SHALL print a
   provider error to stderr and exit `3` (distinct from a triage result).
5. WHEN printing a result THEN the CLI SHALL show the level, recommended action,
   rationale, advice, any red flags, and the disclaimer.

*Verified by:* `test_cli.py`.

---

### Requirement 7: Web presentation adapter (form + JSON API)

**User Story:** As a patient or API client, I want a web interface and a JSON
endpoint, so that I can use Triage Buddy from a browser or another program.

**Acceptance Criteria:**
1. WHEN a client issues `GET /` THEN the server SHALL return an HTML symptom form
   (HTTP 200).
2. WHEN a client issues `POST /` with form-encoded fields THEN the server SHALL
   return the form re-rendered with the assessment.
3. WHEN a client issues `POST /triage` with a JSON object THEN the server SHALL
   return the assessment as JSON.
4. IF a `POST /triage` body is not a JSON object THEN the server SHALL respond
   `400` with an error message.
5. IF input validation fails THEN the server SHALL respond `400`; IF the provider
   is unavailable/misconfigured THEN the server SHALL respond `503`.
6. WHEN a client issues `GET /healthz` THEN the server SHALL report the configured
   provider's health: `200` `{"status":"ok","provider":...}` when reachable, or
   `503` `{"status":"unavailable",...}` when misconfigured (missing key/SDK) or
   unreachable. The probe SHALL be cheap (no generation / token cost) AND its
   result SHALL be cached per provider for a short TTL (default 10s, configurable
   via `--health-ttl`) so frequent polling does not probe the provider each time.
7. WHEN a request targets an unknown route THEN the server SHALL respond `404`.
8. WHEN rendering any user-supplied value into HTML THEN the server SHALL escape
   it (no reflected script execution).
9. IF a request body exceeds 64 KiB THEN the server SHALL respond `413` and SHALL
   NOT process it.

*Verified by:* `test_web_http.py`, `test_web_render.py`, `test_web_service.py`,
`test_health_cache.py`.

---

### Requirement 8: Swappable LLM providers

**User Story:** As an operator, I want to choose the LLM provider without changing
core logic, so that providers stay interchangeable.

**Acceptance Criteria:**
1. WHEN the operator selects provider `mock` THEN the system SHALL produce
   deterministic, offline assessments with no network call or API key.
2. WHEN the operator selects provider `groq` THEN the system SHALL use the
   Groq-hosted model `llama-3.3-70b-versatile` through the LLM port; WHEN the
   operator selects provider `gemini` THEN the system SHALL use the Google model
   `gemini-2.5-flash` through the LLM port.
3. IF an unknown provider name is selected THEN the system SHALL raise/return a
   provider error (surfaced per the active adapter: CLI exit `3`, web `503`).
4. IF a real provider (`groq`, `gemini`) is selected but its SDK is not installed
   OR its API key is missing THEN the system SHALL fail with a clear configuration
   error at startup, NOT as a triage fallback.

**Edge Cases:**
- Each real provider's SDK is an optional extra; the core + `mock` slice has no
  third-party runtime dependency.

*Verified by:* `test_groq_adapter.py`, `test_gemini_adapter.py`,
`test_triage.py::test_end_to_end_with_mock_provider`,
`test_web_service.py::test_run_triage_unknown_provider_is_503`.

---

### Requirement 9: Configuration & secrets

**User Story:** As an operator, I want to keep API keys out of source control and
shell history, so that secrets are handled safely.

**Acceptance Criteria:**
1. WHEN an adapter entry point starts THEN it SHALL load key/value pairs from a
   local `.env` file if present.
2. IF an environment variable is already set THEN the value in `.env` SHALL NOT
   override it.
3. WHEN no `.env` file exists THEN startup SHALL proceed without error.
4. WHEN `.env` is used THEN it SHALL be excluded from version control.

*Verified by:* `test_config.py`.

---

### Requirement 10: Max-of-both-signals escalation

**User Story:** As a patient, I want the more cautious of the deterministic safety
rules and the AI's judgment, so that neither signal can quietly under-call my
urgency.

**Acceptance Criteria:**
1. WHEN producing a final level THEN the system SHALL take the **more severe** of
   the deterministic severity floor and the LLM's suggested level.
2. IF the LLM suggests a level **below** the floor THEN the system SHALL return the
   floor level with `source = "safety-override"` and the floor's reasons.
3. IF the LLM suggests a level **at or above** the floor THEN the system SHALL
   return the LLM's level, rationale, and advice with `source = "llm"`.
4. WHEN the floor is already at the maximum severity (`EMERGENCY`) THEN the system
   SHALL NOT call the LLM (no answer could raise it), keeping emergencies instant
   and independent of provider availability.
5. IF the LLM fails while the floor is below `EMERGENCY` THEN the system SHALL
   return the more severe of the floor and the conservative fallback level.

**Edge Cases:**
- The severity floor is computed by an injectable function. Currently it encodes
  one rule (red flags → `EMERGENCY`); intermediate floors are a tested extension
  point (`test_triage.py::test_floor_raises_under_calling_llm`) pending clinical
  review (see Open Questions).

*Verified by:* `test_triage.py::test_floor_raises_under_calling_llm`,
`::test_llm_above_floor_is_kept`, `::test_llm_equal_to_floor_is_kept_as_llm`,
`::test_floor_applies_even_when_llm_fails`,
`::test_red_flag_forces_emergency_and_skips_llm`.

---

### Requirement 11: Provider timeouts & retries

**User Story:** As a patient (or operator), I want transient provider hiccups to
be retried and slow calls bounded, so that a brief network blip doesn't degrade
the result and a hung provider doesn't hang the request.

**Acceptance Criteria:**
1. WHEN a real provider request exceeds the configured per-attempt timeout
   (default 30s) THEN the SDK client SHALL abort that attempt.
2. WHEN a provider request fails transiently THEN the adapter SHALL retry it up to
   the configured number of attempts (default 3) with exponential backoff
   (default base 0.5s, doubling, capped).
3. WHEN all attempts are exhausted THEN the adapter SHALL raise `LLMError` (which
   the core converts to the conservative fallback, Req 4).
4. WHEN a request succeeds on a retry THEN the system SHALL return that result
   normally.

**Edge Cases:**
- Retry behavior is uniform across providers (a shared helper); each provider
  SDK's own retry counter is disabled to avoid double-counting.
- Backoff delays are bounded by a max-delay cap.

*Verified by:* `test_retry.py`, `test_groq_adapter.py::test_retries_transient_failures_then_succeeds`,
`::test_gives_up_after_max_attempts`, `::test_client_configured_with_timeout_and_no_sdk_retries`,
and the matching cases in `test_gemini_adapter.py`.

---

## Non-Functional Requirements

- **Safety:** A recognized red flag SHALL always yield `EMERGENCY` independent of
  any model or network state (Req 2). Every assessment SHALL carry the disclaimer
  (Req 5). Provider failures SHALL fail safe (Req 4).
- **Architecture:** Core/domain logic SHALL contain no framework, transport, or
  provider imports; providers and presentation surfaces SHALL be adapters behind
  ports (hexagonal). Adding a provider or surface SHALL not modify the core.
- **Dependencies:** The core, CLI, web adapter, and `mock` provider SHALL run with
  the Python standard library only; real LLM SDKs SHALL be optional extras.
- **Security:** User input rendered to HTML SHALL be escaped; request bodies SHALL
  be size-capped (64 KiB).

---

## Out of Scope (current build)

- Intermediate deterministic floors (e.g. `HIGH`/`MEDIUM` for specific
  concerning-but-not-emergency phrases). The max-of-both mechanism supports them
  (Req 10), but the rules themselves await clinical review.
- Multi-turn / conversational triage and follow-up questions.
- Persistence, user accounts, audit logging, and rate limiting.
- Authentication/authorization on the web API.
- Internationalization / non-English symptom descriptions.
- Clinical validation of the red-flag list or escalation mapping.

---

## Open Questions

1. ~~Should the final level be the **max** of (deterministic baseline, LLM
   suggestion)?~~ **Resolved (implemented, Req 10):** the final level is the more
   severe of the floor and the LLM suggestion. Red flags still short-circuit
   because their floor (`EMERGENCY`) is already the ceiling.
2. Should non-emergency concerning phrases (e.g. mild allergic reaction) have an
   intermediate deterministic floor below `EMERGENCY`? (The mechanism is ready;
   the clinical rules are not yet defined.)
3. What are the production latency/availability targets for the web API? Provider
   timeouts/retries (Req 11) and a cached provider-aware `/healthz` (Req 7.6) are
   implemented. Still open: do we also want a separate pure-liveness endpoint
   (process-up, no provider probe) distinct from this readiness-style check?
4. Does the disclaimer wording need legal/clinical review before any real use?
