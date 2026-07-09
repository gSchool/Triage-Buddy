"""Tests for the composition root: provider selection and model forwarding.

Providers are stubbed so no API key is needed — these cover the wiring in
``build_provider``/``build_service``, not provider behavior.
"""

from triage_buddy import composition


def test_build_provider_forwards_model_when_given(monkeypatch):
    captured = {}

    def fake_zai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(composition, "ZaiProvider", fake_zai)
    composition.build_provider("zai", model="glm-9")
    assert captured == {"model": "glm-9"}


def test_build_provider_omits_model_when_none(monkeypatch):
    # No --model: the provider keeps its own default / env-var resolution, so no
    # model kwarg is forwarded.
    captured = {}

    def fake_zai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(composition, "ZaiProvider", fake_zai)
    composition.build_provider("zai")
    assert captured == {}


def test_build_provider_resolves_labproxy(monkeypatch):
    captured = {}

    def fake_labproxy(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(composition, "LabProxyProvider", fake_labproxy)
    composition.build_provider("labproxy", model="claude-sonnet-5")
    assert captured == {"model": "claude-sonnet-5"}


def test_build_provider_resolves_ollama(monkeypatch):
    captured = {}

    def fake_ollama(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(composition, "OllamaProvider", fake_ollama)
    composition.build_provider("ollama", model="gemma4")
    assert captured == {"model": "gemma4"}


def test_build_provider_resolves_openrouter(monkeypatch):
    captured = {}

    def fake_openrouter(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(composition, "OpenRouterProvider", fake_openrouter)
    composition.build_provider("openrouter", model="deepseek/deepseek-v4-flash")
    assert captured == {"model": "deepseek/deepseek-v4-flash"}
