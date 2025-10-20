#!/usr/bin/env python3
"""Offline smoke tests for mcp/wmo_cli_server.py."""

import os
import sys
import json
from pathlib import Path
from types import SimpleNamespace


def setup_paths_and_env():
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "mcp"))
    os.environ.setdefault("OPENAI_API_KEY", "test-key")

    # Stub chromadb to avoid optional dependency
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


def test_search_normalization():
    import wmo_cli_server as wmo

    snippets = [
        {
            "doc_id": "wmo",
            "file": "report.pdf",
            "page": 3,
            "text": "Heatwaves increased in frequency.",
            "preview": "Heatwaves increased in frequency.",
            "similarity": 0.88,
            "chunk_id": "wmo::chunk_0",
        }
    ]

    wmo._search = lambda q, k: snippets  # type: ignore

    res = _call_tool(wmo.WMOIPCCReportSearch, "heatwaves", k=1)
    assert res == snippets
    print("✓ WMOIPCCReportSearch returns normalized snippets")


def test_ask_returns_passages():
    import wmo_cli_server as wmo

    snippets = [
        {
            "doc_id": "wmo",
            "file": "wmo_report.pdf",
            "page": 7,
            "text": "Regional drought events intensified.",
            "preview": "Regional drought events intensified.",
            "similarity": 0.93,
            "chunk_id": "wmo::chunk_0",
        }
    ]

    wmo._search = lambda q, k: snippets  # type: ignore

    fake_chat = SimpleNamespace(
        completions=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Mock WMO answer."))]
            )
        )
    )
    wmo.oai = SimpleNamespace(chat=fake_chat)  # type: ignore

    out = _call_tool(wmo.WMOReportAsk, "What about droughts?", k=1)
    assert out["passages"] == snippets
    assert out["metadata"]["top_k"] == 1
    assert out["answer"].startswith("Mock WMO answer")
    print("✓ WMOReportAsk returns normalized passages")


def main():
    setup_paths_and_env()
    test_search_normalization()
    test_ask_returns_passages()
    print("OK: wmo_cli_server offline tests passed")


if __name__ == "__main__":
    main()
