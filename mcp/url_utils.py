"""URL helper utilities for MCP servers and orchestrators."""

from __future__ import annotations

import os
import re
from typing import Optional

_DEFAULT_BASE_URL = "https://api.transitiondigital.org"


def ensure_absolute_url(path: Optional[str]) -> Optional[str]:
    """Return an absolute URL for static assets served by the API.

    Falls back to ``https://api.transitiondigital.org`` when
    ``API_BASE_URL`` is not set so that production links remain stable.
    Relative paths are expected to start with ``/static`` but the helper
    defensively normalises any leading slashes.

    IMPORTANT: This function also enforces HTTPS for production domains to prevent
    mixed content errors when the frontend is loaded over HTTPS.
    """

    if path is None:
        return None
    if not isinstance(path, str):
        return path

    trimmed = path.strip()
    if not trimmed:
        return trimmed

    # If URL is already absolute, keep it but may need to enforce HTTPS below
    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        result_url = trimmed
    else:
        # Build absolute URL from relative path
        base_url = os.getenv("API_BASE_URL", _DEFAULT_BASE_URL).strip()
        if not base_url:
            base_url = _DEFAULT_BASE_URL

        base_url = base_url.rstrip("/")
        relative = "/" + trimmed.lstrip("/")
        result_url = f"{base_url}{relative}"

    # Defensive HTTPS enforcement for production domains
    # This prevents mixed content errors where HTTPS frontend loads HTTP resources
    production_patterns = [
        r'http://([\w\-\.]*\.)?sunship\.one',
        r'http://([\w\-\.]*\.)?transitiondigital\.org',
    ]

    for pattern in production_patterns:
        if re.match(pattern, result_url):
            result_url = result_url.replace('http://', 'https://', 1)
            break

    return result_url


__all__ = ["ensure_absolute_url"]
