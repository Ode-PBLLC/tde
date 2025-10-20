#!/usr/bin/env python3
"""Quick regression test for Brazilian state-level solar queries."""

from __future__ import annotations

import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "mcp.support_intent" not in sys.modules:
    support_module = types.ModuleType("mcp.support_intent")

    class SupportIntentStub:  # pragma: no cover - simple stub for tests
        def __init__(self, supported: bool = True, score: float = 1.0, reasons: list[str] | None = None) -> None:
            self.supported = supported
            self.score = score
            self.reasons = reasons or []

    support_module.SupportIntent = SupportIntentStub
    sys.modules["mcp.support_intent"] = support_module

from mcp.contracts_v2 import RunQueryResponse
from mcp.servers_v2.solar_server_v2 import SolarServerV2


def test_mato_grosso_summary() -> None:
    server = SolarServerV2()
    response = server.handle_run_query(
        query="List solar facilities near Mato Grosso",
        context={},
    )

    assert isinstance(response, RunQueryResponse)
    assert any(fact.id == "state_facility_count" for fact in response.facts), "state count fact missing"

    artifact = response.artifacts[0]
    metadata = artifact.metadata or {}

    assert metadata.get("scope", {}).get("state") == "Mato Grosso"
    assert metadata.get("total_facilities_state") == 198
    assert round(metadata.get("total_capacity_state_mw", 0.0), 1) == 591.4

    largest = next((fact for fact in response.facts if fact.id == "state_largest_facility"), None)
    assert largest is not None and "89.2" in largest.text, "largest facility fact missing expected capacity"

    print("✓ Mato Grosso state query returns 198 facilities and ~591.4 MW")


if __name__ == "__main__":
    test_mato_grosso_summary()
