"""Web adapter: a FastAPI app exposing a browser form and a JSON API.

Routes:
- ``GET  /``        -> HTML form
- ``POST /``        -> HTML form re-rendered with the assessment (urlencoded)
- ``POST /triage``  -> JSON API (``{"description": ...}``)
- ``GET  /healthz`` -> provider health: 200 when reachable, 503 otherwise

This is presentation only. All triage logic lives in the core; all request
validation/serialization lives in ``service.py``. The FastAPI app just maps HTTP
to those calls. FastAPI/uvicorn live behind the ``[web]`` extra and are imported
lazily, so the core + mock slice still installs and runs with no third-party deps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from triage_buddy.adapters.web.service import (
    DEFAULT_HEALTH_TTL,
    ProviderHealthCache,
    run_triage,
)
from triage_buddy.config import load_dotenv

# Escalation level name -> badge background color, for a quick visual read of urgency.
_LEVEL_COLORS = {
    "EMERGENCY": "#c0392b",
    "HIGH": "#e67e22",
    "MEDIUM": "#f1c40f",
    "LOW": "#27ae60",
}

_MAX_BODY_BYTES = 64 * 1024  # generous for a symptom description; rejects abuse

# Jinja2 environment over the adapter's templates/ dir. Autoescape is on, so every
# interpolated value (user input included) is HTML-escaped — the XSS guard.
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


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
    """Render the full HTML page from ``templates/page.html`` (autoescaped)."""
    return _jinja_env.get_template("page.html").render(
        form=form or {},
        result=result,
        error=error,
        provider=provider,
        colors=_LEVEL_COLORS,
    )


# --------------------------------------------------------------------------- #
# FastAPI app (transport)
# --------------------------------------------------------------------------- #

def create_app(provider: str = "mock", *, health_ttl: float = DEFAULT_HEALTH_TTL):
    """Build a FastAPI app bound to a chosen LLM provider.

    A single ``ProviderHealthCache`` is shared across all requests (captured in
    the closure) so ``/healthz`` polls reuse a recent probe for ``health_ttl``s.
    """
    app = FastAPI(title="Triage Buddy", version="0.1.0")
    health_cache = ProviderHealthCache(ttl=health_ttl)

    async def _read_capped_body(request: Request) -> bytes | None:
        """Read the request body, or return ``None`` if it exceeds the cap.

        Mirrors the old stdlib handler: a generous limit for a symptom
        description that still rejects abusive payloads.
        """
        try:
            length = int(request.headers.get("content-length", 0))
        except ValueError:
            length = 0
        if length > _MAX_BODY_BYTES:
            return None
        body = await request.body()
        if len(body) > _MAX_BODY_BYTES:
            return None
        return body

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return render_page(provider=provider)

    @app.post("/", response_class=HTMLResponse)
    async def submit_form(request: Request) -> HTMLResponse:
        body = await _read_capped_body(request)
        if body is None:
            return HTMLResponse(
                render_page(error="request too large", provider=provider),
                status_code=413,
            )
        fields = parse_qs(body.decode("utf-8", "replace"))

        def first(name: str) -> str:
            return fields.get(name, [""])[0]

        form = {"description": first("description")}
        status, data = run_triage(
            description=form["description"],
            provider=provider,
        )
        if status == 200:
            page = render_page(form=form, result=data, provider=provider)
        else:
            page = render_page(form=form, error=data.get("error"), provider=provider)
        return HTMLResponse(page, status_code=status)

    @app.post("/triage")
    async def triage_api(request: Request) -> JSONResponse:
        body = await _read_capped_body(request)
        if body is None:
            return JSONResponse({"error": "request too large"}, status_code=413)
        try:
            payload = json.loads(body or b"{}")
            if not isinstance(payload, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            return JSONResponse(
                {"error": "request body must be a JSON object"}, status_code=400
            )
        status, data = run_triage(
            description=payload.get("description"),
            provider=provider,
        )
        return JSONResponse(data, status_code=status)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        status, data = health_cache.get(provider)
        return JSONResponse(data, status_code=status)

    return app


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
    parser.add_argument(
        "--health-ttl",
        type=float,
        default=DEFAULT_HEALTH_TTL,
        help=f"Seconds to cache /healthz probe results (default: {DEFAULT_HEALTH_TTL}).",
    )
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(args.provider, health_ttl=args.health_ttl)
    print(
        f"Triage Buddy serving on http://{args.host}:{args.port}  "
        f"(provider: {args.provider}, Ctrl-C to stop)"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
