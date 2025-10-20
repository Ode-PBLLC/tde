"""Lightweight checks for the LSE MCP server tools.

Run with `python test_scripts/test_lse_server_tools.py` to validate that the
normalized dataset is loaded, the tool surface is intact, and key tool
responses include expected content.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Dict

# Ensure repository root is importable when the script is executed from its own directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastmcp.tools.tool import FunctionTool

import mcp.lse_server as lse_server


def _collect_tools() -> Dict[str, FunctionTool]:
    """Return all FastMCP FunctionTool instances exposed by the server."""

    return {
        name: value
        for name, value in vars(lse_server).items()
        if isinstance(value, FunctionTool)
    }


def main() -> None:
    tools = _collect_tools()
    expected_tool_count = 23
    assert len(tools) == expected_tool_count, (
        f"Expected {expected_tool_count} tools, found {len(tools)}"
    )

    missing_docstrings = [
        name for name, tool in tools.items() if not inspect.getdoc(tool.fn)
    ]
    assert not missing_docstrings, f"Missing docstrings for: {missing_docstrings}"

    overview = tools["GetLSEDatasetOverview"].fn()
    assert overview.get("records", 0) > 0, "Dataset overview missing record count"

    ndc_slug = lse_server.catalog.module_index["ndc_overview"][0]
    ndc_tab = tools["GetLSETab"].fn(ndc_slug, include_records=False)
    assert ndc_tab["title"], "NDC tab missing title"

    ndc_targets = tools["GetNDCTargets"].fn()
    assert ndc_targets["long_term"], "NDC targets missing long-term entry"

    search = tools["SearchLSEContent"].fn("adaptation")
    assert search["count"] > 0, "Search should surface adaptation references"

    state_overview = tools["GetBrazilianStatesOverview"].fn()
    assert state_overview["total_states"] >= 20, "Unexpected state coverage"

    comparison = tools["CompareBrazilianStates"].fn(
        states=["Amazonas", "Rio de Janeiro"]
    )
    assert set(comparison["states"]) == {"Amazonas", "Rio de Janeiro"}, (
        "State comparison missing expected entries"
    )

    print(f"✓ tool count: {len(tools)}")
    print("✓ docstrings present for all tools")
    print(
        "✓ key tool calls (overview, NDC targets, search, states) returned expected data"
    )


if __name__ == "__main__":
    main()
