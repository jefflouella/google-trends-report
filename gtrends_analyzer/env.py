from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_if_present(path: str | Path = ".env") -> bool:
    """
    Minimal .env loader (no dependency on python-dotenv).

    - Loads KEY=VALUE lines into os.environ if the key is not already set.
    - Ignores blank lines and comments starting with '#'.
    - Strips surrounding single/double quotes from values.
    """
    p = Path(path)
    if not p.is_file():
        return False

    for raw_line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")  # drop surrounding quotes
        if not key:
            continue
        os.environ.setdefault(key, value)

    return True


