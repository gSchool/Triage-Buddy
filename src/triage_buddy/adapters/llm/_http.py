"""Shared HTTP helpers for the stdlib-``urllib`` LLM adapters.

The ``urllib``-based adapters (lab-proxy, OpenRouter, ...) all need to make
HTTPS calls, and some Python installs (notably python.org's on macOS) ship
without a wired system CA bundle, so stdlib ``ssl``/``urllib`` fail to verify
otherwise-valid certificates. ``default_opener`` centralizes the certifi-backed
fallback so each adapter doesn't reimplement it.
"""

from __future__ import annotations

import ssl
import urllib.request


def default_opener():
    """Build a urlopen-like callable, using certifi's CA bundle if available.

    ``certifi`` is not a declared dependency of this project — it's used here
    opportunistically, only if something else already pulled it into the
    environment. Falls back to ``urllib.request.urlopen``'s own default
    verification otherwise.
    """
    try:
        import certifi
    except ImportError:
        return urllib.request.urlopen

    context = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    return opener.open
