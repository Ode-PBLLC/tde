"""Exercise GetHeatQuintilesMap and print key fields."""

from __future__ import annotations

import asyncio
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _main() -> None:
    module_globals = runpy.run_path("mcp/servers_v2/extreme_heat_server_v2.py")
    server_cls = module_globals["ExtremeHeatServerV2"]
    server = server_cls()
    tools = await server.mcp.get_tools()
    tool = tools["GetHeatQuintilesMap"]
    result = tool.fn()

    print("Summary:", result.get("summary"))
    facts = result.get("facts") or []
    print("Facts count:", len(facts))
    artifacts = result.get("artifacts") or []
    if artifacts:
        artifact = artifacts[0]
        print("Artifact type:", artifact.get("type"))
        print("GeoJSON URL:", artifact.get("geojson_url"))
        print("Metadata keys:", sorted((artifact.get("metadata") or {}).keys()))
    else:
        print("Artifacts: <none>")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
