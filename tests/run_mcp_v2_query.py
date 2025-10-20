"""Small helper script to exercise the v2 MCP orchestrator."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def pretty_print(response: dict) -> None:
    metadata = response.get("metadata", {})
    modules = response.get("modules", [])

    print("== Metadata ==")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    summary_module = next(
        (module for module in modules if module.get("type") == "text"), None
    )
    if summary_module:
        print("\n== Synthesized Response ==")
        texts = summary_module.get("texts")
        if isinstance(texts, str):
            texts = [texts]
        for text in texts or []:
            print(text.strip())

    print("\n== Modules ==")
    for index, module in enumerate(modules, start=1):
        heading = module.get("heading") or ""
        print(f"[{index}] {module.get('type')} :: {heading}")
        if module.get("type") == "numbered_citation_table":
            print(json.dumps(module.get("rows", []), indent=2, ensure_ascii=False))


async def main(query: str) -> None:
    from mcp.mcp_chat_v2 import process_query, cleanup_v2_client  # lazy import

    try:
        response = await process_query(query)
        pretty_print(response)
    finally:
        try:
            await cleanup_v2_client()
        except Exception as exc:
            print(f"Warning: failed to cleanup MCP client: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a query against mcp_chat_v2")
    parser.add_argument("query", help="User query to send to the orchestrator")
    args = parser.parse_args()

    # Ensure current repo is on PYTHONPATH for child processes (servers)
    current = os.environ.get("PYTHONPATH")
    if current:
        if str(REPO_ROOT) not in current.split(":"):
            os.environ["PYTHONPATH"] = f"{REPO_ROOT}:{current}"
    else:
        os.environ["PYTHONPATH"] = str(REPO_ROOT)

    asyncio.run(main(args.query))
