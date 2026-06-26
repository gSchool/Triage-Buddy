"""Web adapter: a stdlib HTTP server exposing a browser form and a JSON API.

Routes:
- ``GET  /``        -> HTML form
- ``POST /``        -> HTML form re-rendered with the assessment (urlencoded)
- ``POST /triage``  -> JSON API (``{"description": ..., "age": ...}``)
- ``GET  /healthz`` -> ``{"status": "ok"}``

This is presentation only. All triage logic lives in the core; all request
validation/serialization lives in ``service.py``. The server just maps HTTP to
those calls.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from triage_buddy.adapters.web.service import run_triage
from triage_buddy.config import load_dotenv

# Level name -> (badge background, text) for a quick visual read of urgency.
_LEVEL_COLORS = {
    "EMERGENCY": "#c0392b",
    "URGENT": "#e67e22",
    "PROMPT": "#f1c40f",
    "ROUTINE": "#27ae60",
    "SELF_CARE": "#2ecc71",
}

_MAX_BODY_BYTES = 64 * 1024  # generous for a symptom description; rejects abuse


# --------------------------------------------------------------------------- #
# HTML rendering (presentation)
# --------------------------------------------------------------------------- #

def render_page(
    *,
    form: dict[str, str] | None = None,
    result: dict | None = None,
    error: str | None = None,
    provider: str = "mock",
) -> str:
    """Render the full HTML page. All interpolated values are escaped."""
    form = form or {}

    def val(name: str) -> str:
        return html.escape(form.get(name, ""))

    blocks = ""
    if error:
        blocks += f'<div class="card error"><strong>Error:</strong> {html.escape(error)}</div>'
    if result:
        blocks += _render_result(result)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Triage Buddy</title>
<style>
  :root {{ font-family: system-ui, sans-serif; line-height: 1.5; }}
  body {{ max-width: 640px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1 {{ margin-bottom: .25rem; }}
  .sub {{ color: #666; margin-top: 0; }}
  form {{ display: grid; gap: .75rem; margin: 1.5rem 0; }}
  label {{ font-weight: 600; font-size: .9rem; }}
  textarea, input {{ width: 100%; padding: .5rem; font: inherit; box-sizing: border-box;
                     border: 1px solid #ccc; border-radius: 6px; }}
  textarea {{ min-height: 5rem; resize: vertical; }}
  .row {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .75rem; }}
  button {{ padding: .6rem 1rem; font: inherit; font-weight: 600; cursor: pointer;
            background: #2c3e50; color: #fff; border: 0; border-radius: 6px; }}
  .card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin: 1rem 0; }}
  .card.error {{ border-color: #c0392b; background: #fdecea; }}
  .badge {{ display: inline-block; padding: .2rem .6rem; border-radius: 999px;
            color: #fff; font-weight: 700; font-size: .85rem; }}
  .badge.PROMPT {{ color: #222; }}
  .flags {{ color: #c0392b; font-size: .9rem; }}
  .disclaimer {{ color: #666; font-size: .85rem; margin-top: 1rem; }}
  .meta {{ color: #999; font-size: .75rem; }}
</style>
</head>
<body>
  <h1>Triage Buddy</h1>
  <p class="sub">Describe your symptoms for escalation advice. Provider: <code>{html.escape(provider)}</code></p>
  <form method="post" action="/">
    <div>
      <label for="description">Symptoms</label>
      <textarea id="description" name="description" required
                placeholder="e.g. sore throat and mild fever for two days">{val("description")}</textarea>
    </div>
    <div class="row">
      <div><label for="age">Age</label><input id="age" name="age" inputmode="numeric" value="{val("age")}"></div>
      <div><label for="sex">Sex</label><input id="sex" name="sex" value="{val("sex")}"></div>
      <div><label for="duration">Duration</label><input id="duration" name="duration" value="{val("duration")}"></div>
    </div>
    <button type="submit">Get advice</button>
  </form>
  {blocks}
</body>
</html>"""


def _render_result(result: dict) -> str:
    level = result["level"]
    color = _LEVEL_COLORS.get(level, "#2c3e50")
    flags = ""
    if result.get("red_flags"):
        flags = '<p class="flags">Flags: ' + html.escape(", ".join(result["red_flags"])) + "</p>"
    return f"""<div class="card">
  <span class="badge {html.escape(level)}" style="background:{color}">{html.escape(result["label"]).upper()}</span>
  <p><strong>{html.escape(result["action"])}</strong></p>
  <p><em>Why:</em> {html.escape(result["rationale"])}</p>
  <p><em>What to do:</em> {html.escape(result["advice"])}</p>
  {flags}
  <p class="disclaimer">{html.escape(result["disclaimer"])}</p>
  <p class="meta">source: {html.escape(result["source"])}</p>
</div>"""


# --------------------------------------------------------------------------- #
# HTTP handler (transport)
# --------------------------------------------------------------------------- #

def make_handler(provider: str = "mock") -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to a chosen LLM provider."""

    class TriageHandler(BaseHTTPRequestHandler):
        server_version = "TriageBuddy/0.1"

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok"})
            elif self.path in ("/", ""):
                self._send_html(200, render_page(provider=provider))
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            body = self._read_body()
            if body is None:
                return  # error already sent

            if self.path == "/triage":
                self._handle_json_api(body)
            elif self.path in ("/", ""):
                self._handle_form(body)
            else:
                self._send_json(404, {"error": "not found"})

        # -- route bodies ---------------------------------------------------- #

        def _handle_json_api(self, body: bytes) -> None:
            try:
                payload = json.loads(body or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                self._send_json(400, {"error": "request body must be a JSON object"})
                return
            status, data = run_triage(
                description=payload.get("description"),
                age=payload.get("age"),
                sex=payload.get("sex"),
                duration=payload.get("duration"),
                provider=provider,
            )
            self._send_json(status, data)

        def _handle_form(self, body: bytes) -> None:
            fields = parse_qs(body.decode("utf-8", "replace"))

            def first(name: str) -> str:
                return fields.get(name, [""])[0]

            form = {k: first(k) for k in ("description", "age", "sex", "duration")}
            status, data = run_triage(
                description=form["description"],
                age=form["age"],
                sex=form["sex"],
                duration=form["duration"],
                provider=provider,
            )
            if status == 200:
                page = render_page(form=form, result=data, provider=provider)
            else:
                page = render_page(form=form, error=data.get("error"), provider=provider)
            self._send_html(status, page)

        # -- helpers --------------------------------------------------------- #

        def _read_body(self) -> bytes | None:
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return None
            if length > _MAX_BODY_BYTES:
                self._send_json(413, {"error": "request too large"})
                return None
            return self.rfile.read(length) if length > 0 else b""

        def _send_json(self, status: int, data: dict) -> None:
            self._send(status, "application/json", json.dumps(data).encode("utf-8"))

        def _send_html(self, status: int, page: str) -> None:
            self._send(status, "text/html; charset=utf-8", page.encode("utf-8"))

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # silence default stderr logging
            pass

    return TriageHandler


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="triage-buddy-web",
        description="Run the Triage Buddy web server (browser form + JSON API).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument("--provider", default="mock", help="LLM provider adapter (default: mock).")
    args = parser.parse_args(argv)

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(args.provider))
    host, port = httpd.server_address[0], httpd.server_address[1]
    print(f"Triage Buddy serving on http://{host}:{port}  (provider: {args.provider}, Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
