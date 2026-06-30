"""End-to-end HTTP tests: drive the FastAPI app through Starlette's TestClient."""

import pytest
from fastapi.testclient import TestClient

from triage_buddy.adapters.web.app import create_app


@pytest.fixture()
def client():
    with TestClient(create_app("mock")) as client:
        yield client


def test_get_root_serves_form(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<form" in resp.text


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["provider"] == "mock"


def test_json_api_triage(client):
    resp = client.post("/triage", json={"description": "severe chest pain"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["level"] == "EMERGENCY"
    assert data["source"] == "safety-override"


def test_json_api_bad_body(client):
    resp = client.post(
        "/triage",
        content="not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_form_post_renders_result(client):
    resp = client.post(
        "/",
        content="description=mild+runny+nose",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200
    assert "LOW" in resp.text


def test_unknown_path_404(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
