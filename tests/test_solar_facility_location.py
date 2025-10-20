"""Tests for solar facility location helper/tool."""

from __future__ import annotations

import types
import sys
from pathlib import Path


def _ensure_support_stub() -> None:
    if "mcp.support_intent" in sys.modules:
        return
    stub = types.ModuleType("mcp.support_intent")

    class SupportIntent:  # pragma: no cover - simple stub for tests
        def __init__(self, supported: bool = True, score: float = 1.0, reasons: list[str] | None = None) -> None:
            self.supported = supported
            self.score = score
            self.reasons = reasons or []

    stub.SupportIntent = SupportIntent
    sys.modules["mcp.support_intent"] = stub


def test_facility_location_payload_creates_geojson() -> None:
    _ensure_support_stub()
    from mcp.servers_v2.solar_server_v2 import SolarServerV2

    server = SolarServerV2()
    payload = server._facility_location_payload("14079")

    assert payload.get("cluster_id") == "14079"
    coords = payload.get("coordinates") or {}
    assert coords.get("latitude") is not None
    assert coords.get("longitude") is not None

    geojson_url = payload.get("geojson_url")
    assert isinstance(geojson_url, str) and geojson_url.endswith(".geojson")

    geojson_path = Path("static/maps") / Path(geojson_url).name
    assert geojson_path.exists(), "Expected facility GeoJSON to be written to static/maps"

    metadata = payload.get("geojson_metadata") or {}
    layer = metadata.get("layers", {}).get("solar_facilities", {})
    assert layer.get("plotted_count") == 1
