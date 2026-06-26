"""Minimal ``.env`` loading, dependency-free.

Edge adapters (CLI, web) call ``load_dotenv()`` at startup so secrets like
``GROQ_API_KEY`` can live in a local, git-ignored ``.env`` file instead of the
shell history. The core never touches this — config loading is an edge concern.

Real environment variables always win over the file, so an explicit
``export GROQ_API_KEY=...`` overrides whatever ``.env`` says.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike[str] = ".env") -> dict[str, str]:
    """Load ``KEY=value`` pairs from ``path`` into ``os.environ``.

    Returns the parsed pairs. Missing file is not an error (returns ``{}``).
    Supports ``# comments``, blank lines, an optional ``export`` prefix, and
    single/double-quoted values. Existing environment variables are left intact.
    """
    file = Path(path)
    if not file.is_file():
        return {}

    parsed: dict[str, str] = {}
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (len(value) >= 2) and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        parsed[key] = value
        os.environ.setdefault(key, value)
    return parsed
