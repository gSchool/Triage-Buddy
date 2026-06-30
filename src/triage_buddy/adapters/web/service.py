"""Transport-agnostic request handling for the web adapter.

These pure functions take already-parsed inputs and return a status code plus a
plain dict. Both the JSON API and the HTML form go through ``run_triage`` so the
two surfaces can never drift apart, and the logic is trivially unit-testable
without spinning up a server.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from triage_buddy.composition import build_provider, build_service
from triage_buddy.domain.models import SymptomReport, TriageAssessment
from triage_buddy.ports.llm import LLMError

DEFAULT_HEALTH_TTL = 10.0  # seconds; reuse a probe result for this long


def assessment_to_dict(assessment: TriageAssessment) -> dict[str, Any]:
    """Serialize an assessment to a JSON-friendly dict."""
    return {
        "level": assessment.level.name,
        "label": assessment.level.label,
        "action": assessment.level.action,
        "rationale": assessment.rationale,
        "advice": assessment.advice,
        "red_flags": list(assessment.red_flags),
        "source": assessment.source,
        "disclaimer": assessment.disclaimer,
    }


def run_triage(
    *,
    description: str | None,
    provider: str = "mock",
) -> tuple[int, dict[str, Any]]:
    """Validate inputs, run triage, and return ``(http_status, body)``.

    Returns 400 for bad input, 503 for a provider/config problem, and 200 with
    the serialized assessment on success.
    """
    if not description or not description.strip():
        return 400, {"error": "description is required"}

    try:
        report = SymptomReport(description=description)
    except ValueError as exc:
        return 400, {"error": str(exc)}

    try:
        service = build_service(provider=provider)
    except (ValueError, LLMError) as exc:
        # Unknown provider name, or a misconfigured provider (e.g. missing key).
        return 503, {"error": f"provider unavailable: {exc}"}

    assessment = service.assess(report)
    return 200, assessment_to_dict(assessment)


def provider_health(provider: str = "mock") -> tuple[int, dict[str, Any]]:
    """Report whether the configured provider is reachable.

    Returns ``(200, {...ok})`` when the provider can be built and its cheap
    reachability probe succeeds; ``(503, {...})`` when it is misconfigured
    (missing key/SDK, unknown name) or unreachable. Providers without a
    ``check_health`` method are treated as live (config check only).
    """
    try:
        prov = build_provider(provider)
    except (ValueError, LLMError) as exc:
        return 503, {"status": "unavailable", "provider": provider, "error": str(exc)}

    check = getattr(prov, "check_health", None)
    if callable(check):
        try:
            check()
        except LLMError as exc:
            return 503, {"status": "unavailable", "provider": provider, "error": str(exc)}

    return 200, {"status": "ok", "provider": provider}


class ProviderHealthCache:
    """Caches ``provider_health`` results per provider for a short TTL.

    A health endpoint may be polled frequently (load balancers, uptime checks);
    without caching each poll would make a network probe. This reuses a recent
    result for ``ttl`` seconds. Both healthy and unhealthy results are cached so
    a failing provider isn't hammered; with a short TTL, recovery is still
    reflected promptly.

    Thread-safe (the server is threaded). The probe runs outside the lock, so a
    burst on a cold/expired entry may briefly cause a few concurrent probes
    rather than blocking all health checks behind one.
    """

    def __init__(
        self,
        *,
        ttl: float = DEFAULT_HEALTH_TTL,
        probe: Callable[[str], tuple[int, dict[str, Any]]] = None,  # type: ignore[assignment]
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl
        self._probe = probe or provider_health
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, tuple[int, dict[str, Any]]]] = {}

    def get(self, provider: str = "mock") -> tuple[int, dict[str, Any]]:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(provider)
            if entry is not None and now < entry[0]:
                return entry[1]

        result = self._probe(provider)  # outside the lock (may do network I/O)

        with self._lock:
            self._entries[provider] = (now + self._ttl, result)
        return result
