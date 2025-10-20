#!/usr/bin/env python3
"""Offline smoke tests for mcp/mb_deforest_server.py."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace


def setup_paths_and_env():
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "mcp"))
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    import types

    if "chromadb" not in sys.modules:
        chroma_mod = types.ModuleType("chromadb")

        class _DummyCollection:
            def query(self, *args, **kwargs):
                return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        class _DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            def get_or_create_collection(self, *args, **kwargs):
                return _DummyCollection()

        class _DummySettings:
            def __init__(self, *args, **kwargs):
                pass

        chroma_mod.Client = _DummyClient
        config_mod = types.ModuleType("chromadb.config")
        config_mod.Settings = _DummySettings
        sys.modules["chromadb"] = chroma_mod
        sys.modules["chromadb.config"] = config_mod

    if "fastmcp" not in sys.modules:
        fm_mod = types.ModuleType("fastmcp")

        class _DummyFastMCP:
            def __init__(self, *args, **kwargs):
                pass

            def tool(self):
                def deco(fn):
                    return fn

                return deco

            def run(self):  # pragma: no cover
                pass

        fm_mod.FastMCP = _DummyFastMCP
        sys.modules["fastmcp"] = fm_mod


def _call_tool(tool, *args, **kwargs):
    if callable(tool):
        return tool(*args, **kwargs)
    for attr in ("fn", "func", "callback", "wrapped", "wrapped_fn"):
        if hasattr(tool, attr):
            return getattr(tool, attr)(*args, **kwargs)
    raise TypeError("Tool is not directly callable")


def test_mb_search_normalization():
    import mb_deforest_server as mb

    snippets = [
        {
            "doc_id": "rad2024",
            "file": "rad2024.pdf",
            "page": 18,
            "text": "Deforestation increased 12% year over year.",
            "preview": "Deforestation increased 12% year over year.",
            "similarity": 0.9,
            "chunk_id": "rad2024::chunk_0",
        }
    ]

    mb._search = lambda q, k: snippets  # type: ignore
    res = _call_tool(mb.MBReportSearch, "deforestation trend", k=1)
    assert res == snippets
    print("✓ MBReportSearch returns normalized snippets")


def test_mb_ask_passages():
    import mb_deforest_server as mb

    snippets = [
        {
            "doc_id": "rad2024",
            "file": "rad2024.pdf",
            "page": 18,
            "text": "Deforestation increased 12% year over year.",
            "preview": "Deforestation increased 12% year over year.",
            "similarity": 0.9,
            "chunk_id": "rad2024::chunk_0",
        }
    ]

    mb._search = lambda q, k: snippets  # type: ignore

    fake_chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Mock MB answer."))]
            )
        )
    )
    mb.oai = SimpleNamespace(chat=fake_chat)  # type: ignore

    out = _call_tool(mb.MBReportAsk, "What changed in 2024?", k=1)
    assert out["passages"] == snippets
    assert out["metadata"]["top_k"] == 1
    assert out["answer"].startswith("Mock MB answer")
    print("✓ MBReportAsk returns normalized passages")


def main():
    setup_paths_and_env()
    test_mb_search_normalization()
    test_mb_ask_passages()
    print("OK: mb_deforest_server offline tests passed")


if __name__ == "__main__":
    main()
