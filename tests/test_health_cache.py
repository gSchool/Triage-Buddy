"""Tests for the provider-health TTL cache (fake probe + fake clock)."""

from triage_buddy.adapters.web.service import ProviderHealthCache


class CountingProbe:
    def __init__(self, result=(200, {"status": "ok"})):
        self.result = result
        self.calls = []

    def __call__(self, provider):
        self.calls.append(provider)
        return self.result


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_probes_once_within_ttl():
    probe, clock = CountingProbe(), FakeClock()
    cache = ProviderHealthCache(ttl=10, probe=probe, clock=clock)

    assert cache.get("groq") == (200, {"status": "ok"})
    clock.now += 5  # still within TTL
    assert cache.get("groq") == (200, {"status": "ok"})

    assert probe.calls == ["groq"]  # only probed once


def test_reprobes_after_ttl_expires():
    probe, clock = CountingProbe(), FakeClock()
    cache = ProviderHealthCache(ttl=10, probe=probe, clock=clock)

    cache.get("groq")
    clock.now += 11  # TTL elapsed
    cache.get("groq")

    assert probe.calls == ["groq", "groq"]


def test_caches_per_provider_separately():
    probe, clock = CountingProbe(), FakeClock()
    cache = ProviderHealthCache(ttl=10, probe=probe, clock=clock)

    cache.get("groq")
    cache.get("gemini")
    cache.get("groq")

    assert probe.calls == ["groq", "gemini"]  # each provider probed once


def test_caches_failures_too():
    probe = CountingProbe(result=(503, {"status": "unavailable"}))
    clock = FakeClock()
    cache = ProviderHealthCache(ttl=10, probe=probe, clock=clock)

    assert cache.get("groq")[0] == 503
    clock.now += 1
    assert cache.get("groq")[0] == 503
    assert probe.calls == ["groq"]  # failure cached, not re-probed
