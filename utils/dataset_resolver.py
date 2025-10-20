"""
Lightweight dataset resolver for attaching SourceURL to dataset citations.

Resolution strategy (in order):
1) Explicit tool metadata (dataset_id/source_url) if provided by tools in future.
2) Naming convention by tool prefix (e.g., GetSolar* -> solar_facilities).
3) Fuzzy match by source_name/provider against static/meta/datasets.json.
4) Fallback to (None, None).

This keeps maintenance low and centralizes logic in one place.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple


DATASETS_JSON = Path("static/meta/datasets.json")


# Simple provider normalization to improve fuzzy matching
PROVIDER_SYNONYMS: Dict[str, str] = {}


def _normalize(text: Optional[str]) -> str:
    if not text:
        return ""
    s = text.strip().lower()
    # Collapse punctuation/whitespace
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


@lru_cache(maxsize=1)
def _datasets_index() -> Dict[str, Dict[str, str]]:
    """Load and index datasets metadata by id.

    Returns a dict: id -> {title, provider?, source_url}
    """
    if not DATASETS_JSON.exists():
        return {}
    try:
        data = json.loads(DATASETS_JSON.read_text())
    except Exception:
        return {}

    idx: Dict[str, Dict[str, str]] = {}
    for item in data.get("items", []):
        ds_id = item.get("id")
        if not ds_id:
            continue
        title = item.get("title", "")
        provider = _normalize(item.get("provider"))
        provider = PROVIDER_SYNONYMS.get(provider, provider)
        idx[ds_id] = {
            "title": _normalize(title),
            "provider": provider,
            # JSON uses "source" for URL; surface as source_url here
            "source_url": (item.get("source") or "").strip(),
        }
    return idx


def _dataset_id_by_server(server_name: Optional[str]) -> Optional[str]:
    """Module-level default mapping: server -> dataset id.

    Keep this list tiny and stable. Adjust when adding new servers/datasets.
    """
    server = (server_name or "").lower()
    # Known MCP servers -> dataset_id used for SourceURL lookup in static/meta/datasets.json
    # ONLY the exact server names you connect with in mcp_chat.py
    mapping = {
        "cpr": "climate_policy_radar",                     # CPR KG (documents/passages)
        "solar": "solar_facilities",    # TZ-SAM dataset id
        "gist": "gist_multi_dataset",   # GIST (likely no public URL)
        "lse": "ndc_align",                     # Set when you add an LSE dataset id
        "viz": None,                     # Visualization helpers
        "deforestation": "brazil_deforestation",
        "geospatial": None,              # Analysis utilities
        "admin": "brazilian_admin_boundaries",
        "heat": None,                    # Set when heat dataset id is added
        "meta": "meta",                    # Exposes static/meta itself
        "spa": "spa",                      # SPA document set
        "wmo_cli": "wmo_ipcc_climate_assessments",
        "mb_deforest": "mapbiomas_rad_2024",
    }
    return mapping.get(server)


def _fuzzy_dataset_id(source_name: str, provider: str) -> Optional[str]:
    """Disabled: fuzzy matching not needed for stable, module-level mapping."""
    return None


def resolve_dataset_url(
    tool_name: str,
    tool_metadata: Optional[dict] = None,
    citation_source: Optional[dict] = None,
    server_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a (dataset_id, source_url) for a dataset citation.

    - tool_metadata: optional block that tools may emit with dataset_id/source_url
    - citation_source: minimal info like title/source_name and provider
    """
    # 1) Trust explicit tool metadata when present
    if tool_metadata:
        ds_id = tool_metadata.get("dataset_id")
        url = tool_metadata.get("source_url")
        if ds_id and not url:
            url = _datasets_index().get(ds_id, {}).get("source_url")
        if url and not ds_id:
            # Attempt reverse lookup by URL
            for k, v in _datasets_index().items():
                if v.get("source_url") == url:
                    ds_id = k
                    break
        if ds_id or url:
            return ds_id, url

    # 2) Module-level default by server name
    ds_id = _dataset_id_by_server(server_name)
    if ds_id:
        return ds_id, _datasets_index().get(ds_id, {}).get("source_url")

    # 3) No match — give up gracefully
    return None, None
