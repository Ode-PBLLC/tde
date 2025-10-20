"""URL helper utilities for MCP servers and orchestrators."""

from __future__ import annotations

import os
from typing import Optional

_DEFAULT_BASE_URL = "https://api.transitiondigital.org"


def ensure_absolute_url(path: Optional[str]) -> Optional[str]:
    """Return an absolute URL for static assets served by the API.

    Falls back to ``https://api.transitiondigital.org`` when
    ``API_BASE_URL`` is not set so that production links remain stable.
    Relative paths are expected to start with ``/static`` but the helper
    defensively normalises any leading slashes.
    """

    if path is None:
        return None
    if not isinstance(path, str):
        return path

    trimmed = path.strip()
    if not trimmed:
        return trimmed
    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        return trimmed

    base_url = os.getenv("API_BASE_URL", _DEFAULT_BASE_URL).strip()
    if not base_url:
        base_url = _DEFAULT_BASE_URL

    base_url = base_url.rstrip("/")
    relative = "/" + trimmed.lstrip("/")
    return f"{base_url}{relative}"


__all__ = ["ensure_absolute_url"]
