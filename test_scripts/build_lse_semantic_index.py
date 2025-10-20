"""Build or refresh the LSE semantic index using OpenAI embeddings."""

from __future__ import annotations

import asyncio
import os
import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(ENV_PATH)
    except Exception:
        pass


async def main() -> None:
    _load_env()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot generate embeddings")

    module_globals = runpy.run_path(str(PROJECT_ROOT / "mcp" / "servers_v2" / "lse_server_v2.py"))
    server_cls = module_globals["LSEServerV2"]
    server = server_cls()

    print(f"Semantic records available: {len(server._semantic_records)}")
    if server._semantic_records:
        print("Semantic index already built or loaded.")
    else:
        if server._semantic_matrix is None:
            raise RuntimeError("Failed to build semantic index; check OpenAI connectivity.")


if __name__ == "__main__":
    asyncio.run(main())
