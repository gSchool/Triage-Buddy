"""Lab-proxy adapter for the ``LLMProvider`` port.

Targets a classroom/lab proxy speaking the Anthropic ``/v1/messages`` wire
shape. This is **not** a general Anthropic API integration: the specific
instance this was built against (an ngrok tunnel shared by a coworker) has no
real authentication (any ``x-api-key`` value is accepted) and, when probed,
served a local ``llama3.1:8b`` rather than a real Claude model. Treat it as an
unauthenticated, unknown-provenance backend suitable for lab/classroom
experiments only — never for real patient input or as a production provider.

Because the proxy just speaks plain HTTP + JSON, this adapter uses the stdlib
``urllib`` rather than the ``anthropic`` SDK (reserved as a separate, still
unused, optional extra for a real Anthropic integration) — no extra install
needed.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request

from triage_buddy.adapters.llm._retry import (
    DEFAULT_ATTEMPTS,
    DEFAULT_BASE_DELAY,
    DEFAULT_TIMEOUT,
    call_with_retries,
    is_rate_limit,
)
from triage_buddy.ports.llm import LLMError, LLMRequest, LLMResponse, RateLimitError

DEFAULT_MODEL = "claude-sonnet-4-6"  # ignored server-side by the observed proxy; kept for portability
DEFAULT_API_KEY = "lab"
ANTHROPIC_VERSION = "2023-06-01"


def _default_opener():
    """Build a urlopen-like callable, using certifi's CA bundle if available.

    Some Python installs (notably python.org's on macOS) ship without a wired
    system CA bundle, so stdlib ``ssl``/``urllib`` fail to verify otherwise-valid
    certificates (e.g. an ngrok tunnel's). ``certifi`` is not a declared
    dependency of this project — it's only used here, opportunistically, if
    something else already pulled it into the environment. Falls back to
    ``urllib.request.urlopen``'s own default verification otherwise.
    """
    try:
        import certifi
    except ImportError:
        return urllib.request.urlopen

    context = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    return opener.open


class LabProxyProvider:
    """``LLMProvider`` backed by a lab proxy speaking the Anthropic wire shape.

    Args:
        base_url: Proxy base URL. Falls back to the ``LABPROXY_BASE_URL``
            environment variable. Required (no hardcoded default — a shared
            tunnel URL is temporary and shouldn't be baked into the codebase).
        student_id: Sent as ``X-Student-Id`` for the proxy operator's metering.
            Falls back to ``LABPROXY_STUDENT_ID``, then to ``"triage-buddy"``.
        api_key: Sent as ``x-api-key``. Falls back to ``LABPROXY_API_KEY``,
            then to a placeholder — the observed proxy does not actually
            check this value.
        model: Model id to request. If omitted, falls back to the
            ``TRIAGE_MODEL`` environment variable, then to ``DEFAULT_MODEL``.
            The proxy may silently serve a different model regardless of what
            is requested here.
        temperature: Sampling temperature. Defaults to ``0`` for repeatable
            triage output.
        opener: Injected for testing — a callable taking a
            ``urllib.request.Request`` and returning a file-like object with
            ``.read()``. Defaults to a certifi-backed opener when ``certifi``
            is installed, else ``urllib.request.urlopen``.

    Config problems (missing base URL) raise ``LLMError`` *here*, at
    construction, so they surface as a clear setup error rather than
    masquerading as a triage fallback.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        student_id: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_ATTEMPTS,
        retry_base_delay: float = DEFAULT_BASE_DELAY,
        opener=None,
    ) -> None:
        url = base_url or os.environ.get("LABPROXY_BASE_URL")
        if not url:
            raise LLMError(
                "LABPROXY_BASE_URL is not set; export it or pass base_url= to "
                "use the lab-proxy provider"
            )
        self._base_url = url.rstrip("/")
        self._student_id = (
            student_id or os.environ.get("LABPROXY_STUDENT_ID") or "triage-buddy"
        )
        self._api_key = api_key or os.environ.get("LABPROXY_API_KEY") or DEFAULT_API_KEY
        self._model = model or os.environ.get("TRIAGE_MODEL") or DEFAULT_MODEL
        self._temperature = temperature
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._retry_base_delay = retry_base_delay
        self._opener = opener or _default_opener()

    def _post(self, payload: dict, *, max_tokens: int) -> dict:
        body = json.dumps({**payload, "max_tokens": max_tokens}).encode()
        request = urllib.request.Request(
            f"{self._base_url}/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "X-Student-Id": self._student_id,
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimitError(f"lab-proxy rate limit: {exc}") from exc
            raise LLMError(f"lab-proxy request failed: {exc}") from exc
        except Exception as exc:  # network, timeout, malformed JSON
            if is_rate_limit(exc):
                raise RateLimitError(f"lab-proxy rate limit: {exc}") from exc
            raise LLMError(f"lab-proxy request failed: {exc}") from exc

    def generate(self, request: LLMRequest) -> LLMResponse:
        def attempt() -> LLMResponse:
            data = self._post(
                {
                    "model": self._model,
                    "temperature": self._temperature,
                    "system": request.system,
                    "messages": [{"role": "user", "content": request.user}],
                },
                max_tokens=1024,
            )
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            return LLMResponse(text=text)

        return call_with_retries(
            attempt, attempts=self._max_attempts, base_delay=self._retry_base_delay
        )

    def check_health(self) -> None:
        """Cheap reachability probe: a 1-token completion.

        The proxy exposes no model-listing endpoint, so an actual (minimal)
        completion call is the only reachability check available.
        """
        self._post(
            {"model": self._model, "messages": [{"role": "user", "content": "ping"}]},
            max_tokens=1,
        )
