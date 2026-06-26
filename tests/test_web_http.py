"""End-to-end HTTP tests: start the real server on an ephemeral port."""

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from triage_buddy.adapters.web.app import make_handler


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("mock"))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield host, port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def _request(server, method, path, body=None, headers=None):
    host, port = server
    conn = HTTPConnection(host, port, timeout=5)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")
    finally:
        conn.close()


def test_get_root_serves_form(server):
    status, body = _request(server, "GET", "/")
    assert status == 200
    assert "<form" in body


def test_healthz(server):
    status, body = _request(server, "GET", "/healthz")
    assert status == 200
    assert json.loads(body) == {"status": "ok"}


def test_json_api_triage(server):
    status, body = _request(
        server,
        "POST",
        "/triage",
        body=json.dumps({"description": "severe chest pain"}),
        headers={"Content-Type": "application/json"},
    )
    assert status == 200
    data = json.loads(body)
    assert data["level"] == "EMERGENCY"
    assert data["source"] == "safety-override"


def test_json_api_bad_body(server):
    status, body = _request(server, "POST", "/triage", body="not json",
                            headers={"Content-Type": "application/json"})
    assert status == 400


def test_form_post_renders_result(server):
    status, body = _request(
        server,
        "POST",
        "/",
        body="description=mild+runny+nose",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert status == 200
    assert "SELF-CARE" in body


def test_unknown_path_404(server):
    status, _ = _request(server, "GET", "/nope")
    assert status == 404
