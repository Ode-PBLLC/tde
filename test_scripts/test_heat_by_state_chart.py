"""Exercise GetHeatByStateChart and print key fields."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _main() -> None:
    module_globals = runpy.run_path("mcp/servers_v2/extreme_heat_server_v2.py")
    server_cls = module_globals["ExtremeHeatServerV2"]
    server = server_cls()
    tools = await server.mcp.get_tools()
    tool = tools["GetHeatByStateChart"]
    result = tool.fn(top_n=5)

    print("Summary:", result.get("summary"))
    facts = result.get("facts") or []
    print("First fact:", facts[0] if facts else "<none>")
    artifacts = result.get("artifacts") or []
    if artifacts:
        artifact = artifacts[0]
        print("Artifact type:", artifact.get("type"))
        data = artifact.get("data") or {}
        print("Data keys:", sorted(data.keys()))
        print("Labels sample:", data.get("labels", [])[:3])
    else:
        print("Artifacts: <none>")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
