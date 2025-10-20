#!/usr/bin/env python3
"""
Network-independent tests for mcp/spa_server.py tools.

This script:
- Ensures import works by setting a dummy OPENAI_API_KEY.
- Tests AmazonAssessmentListDocs using a small local manifest.
- Mocks search and OpenAI chat to test AmazonAssessmentSearch and AmazonAssessmentAsk.

Run: python test_scripts/test_spa_server.py
"""

import os
import sys
import json
from pathlib import Path


def setup_paths_and_env():
    # Allow importing from project root and mcp/
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "mcp"))
    # Provide a dummy key to pass import-time check
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    
    # Stub heavy/optional deps if missing to keep tests offline
    import types
    
    # chromadb stub (module + submodule chromadb.config.Settings)
    try:
        import chromadb  # type: ignore  # noqa: F401
    except Exception:
        chroma_mod = types.ModuleType("chromadb")
        config_mod = types.ModuleType("chromadb.config")

        class _DummyClient:
            def __init__(self, *args, **kwargs):
                pass

            def get_or_create_collection(self, name):  # pragma: no cover - not used in mocked path
                class _Col:
                    def query(self, *a, **k):
                        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

                return _Col()

        class _DummySettings:
            def __init__(self, *args, **kwargs):
                pass

        chroma_mod.Client = _DummyClient
        config_mod.Settings = _DummySettings
        sys.modules["chromadb"] = chroma_mod
        sys.modules["chromadb.config"] = config_mod

    # fastmcp stub to satisfy decorator at import time
    try:
        from fastmcp import FastMCP  # type: ignore  # noqa: F401
    except Exception:
        fm_mod = types.ModuleType("fastmcp")

        class _FM:
            def __init__(self, *args, **kwargs):
                pass

            def tool(self):
                def deco(fn):
                    return fn
                return deco

            def run(self):  # pragma: no cover
                pass

        fm_mod.FastMCP = _FM
        sys.modules["fastmcp"] = fm_mod

    # openai stub with minimal OpenAI class used at import time
    try:
        from openai import OpenAI  # type: ignore  # noqa: F401
    except Exception:
        oa_mod = types.ModuleType("openai")

        class _OpenAI:
            def __init__(self, *args, **kwargs):
                # real methods are mocked later per test
                self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: None))
                self.embeddings = types.SimpleNamespace(create=lambda **k: None)

        oa_mod.OpenAI = _OpenAI
        sys.modules["openai"] = oa_mod


def make_manifest(items):
    """Create a minimal data/spa_index/manifest.json for listing test."""
    index_dir = Path("data/spa_index")
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = index_dir / "manifest.json"
    manifest.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2))
    return manifest


def _call_tool(maybe_tool, *args, **kwargs):
    """Call a FastMCP tool whether it's a plain function or a wrapped object."""
    if callable(maybe_tool):
        return maybe_tool(*args, **kwargs)
    for attr in ("fn", "func", "callback", "wrapped", "wrapped_fn"):
        if hasattr(maybe_tool, attr):
            return getattr(maybe_tool, attr)(*args, **kwargs)
    raise TypeError(f"Tool object is not callable and no known function attribute found: {type(maybe_tool)}")


def test_list_docs():
    # Create a small manifest
    items = [
        {"file": "doc1.pdf", "page": 1},
        {"file": "doc2.pdf", "page": 2},
    ]
    make_manifest(items)

    import spa_server as spa

    out_all = _call_tool(spa.AmazonAssessmentListDocs, limit=10)
    assert isinstance(out_all, list)
    assert len(out_all) == 2
    assert set(out_all[0].keys()) == {"file", "page"}

    out_one = _call_tool(spa.AmazonAssessmentListDocs, limit=1)
    assert len(out_one) == 1
    print("✓ AmazonAssessmentListDocs returns manifest entries (limit respected)")


def test_search_and_ask_with_mocks():
    import types
    import spa_server as spa

    # Mock _search to avoid Chroma/embeddings
    fake_hits = [
        {
            "doc_id": "reportA",
            "file": "reportA.pdf",
            "page": 10,
            "text": "Forest conservation reduces emissions.",
            "preview": "Forest conservation reduces emissions.",
            "similarity": 0.92,
            "chunk_id": "reportA::chunk_0",
        },
        {
            "doc_id": "reportB",
            "file": "reportB.pdf",
            "page": 5,
            "text": "Indigenous stewardship improves biodiversity outcomes.",
            "preview": "Indigenous stewardship improves biodiversity outcomes.",
            "similarity": 0.89,
            "chunk_id": "reportB::chunk_1",
        },
    ]

    def fake_search(q: str, k: int):
        return fake_hits[:k]

    spa._search = fake_search  # type: ignore

    # Verify AmazonAssessmentSearch delegates to _search
    res = _call_tool(spa.AmazonAssessmentSearch, "amazon conservation", k=2)
    assert isinstance(res, list) and len(res) == 2
    first = res[0]
    assert first["file"] == "reportA.pdf"
    assert first["doc_id"] == "reportA"
    assert first["chunk_id"].startswith("reportA::")
    print("✓ AmazonAssessmentSearch returns normalized snippets")

    # Mock OpenAI chat completion for AmazonAssessmentAsk
    class _Msg:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    class _Chat:
        class completions:
            @staticmethod
            def create(model, messages, temperature, max_tokens):  # noqa: D401
                return _Resp("Expert answer about SPA evidence.")

    class _FakeOAI:
        chat = _Chat()

    # Swap global OpenAI client used by spa_server
    spa.oai = _FakeOAI()  # type: ignore

    ask_out = _call_tool(spa.AmazonAssessmentAsk, "What are key SPA findings?", k=2, max_tokens=128)
    assert "answer" in ask_out and "passages" in ask_out
    assert "Expert answer" in ask_out["answer"]
    assert len(ask_out["passages"]) == 2
    assert {"doc_id", "file", "page", "preview", "similarity", "chunk_id"}.issubset(ask_out["passages"][0].keys())
    assert ask_out.get("metadata", {}).get("top_k") == 2
    print("✓ AmazonAssessmentAsk returns normalized passages")


def main():
    setup_paths_and_env()
    test_list_docs()
    test_search_and_ask_with_mocks()
    print("OK: spa_server tools behave under mocked conditions")


if __name__ == "__main__":
    main()
