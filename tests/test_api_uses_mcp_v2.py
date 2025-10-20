"""Tests ensuring the FastAPI layer can consume `mcp_chat_v2` style responses."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_query_endpoint_accepts_mcp_v2_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `/query` can serve a response shaped like `mcp_chat_v2` output."""

    import api_server  # Local import so monkeypatching affects the loaded app
    from mcp.mcp_chat_v2 import validate_final_response

    query = "Which climate policies reference solar energy?"

    modules: List[Dict[str, Any]] = [
        {
            "type": "text",
            "heading": "Summary",
            "texts": [
                "- Brazil's NDC documentation highlights solar energy investments ^1^",
            ],
        },
        {
            "type": "table",
            "heading": "Relevant CPR Passages",
            "columns": ["Concept", "Document", "Passage"],
            "rows": [
                [
                    "Solar energy",
                    "UNFCCC.party.166.0",
                    "Brazil's action plans include solar energy expansion commitments.",
                ]
            ],
        },
        {
            "type": "numbered_citation_table",
            "heading": "References",
            "columns": ["#", "Source", "ID/Tool", "Type", "Description", "SourceURL"],
            "rows": [
                [
                    "1",
                    "CPR passage UNFCCC.party.166.0",
                    "run_query",
                    "Policy Document",
                    "Passage 11285 in document UNFCCC.party.166.0 mentioning solar energy.",
                    "",
                ]
            ],
        },
    ]

    sample_response: Dict[str, Any] = {
        "query": query,
        "modules": modules,
        "metadata": {
            "modules_count": len(modules),
            "has_maps": False,
            "has_charts": False,
            "has_tables": True,
            "module_types": [module["type"] for module in modules],
            "kg_visualization_url": "/kg-viz",
            "kg_query_url": "/kg-viz?query=Which%20climate%20policies%20reference%20solar%20energy?",
        },
        "kg_context": {
            "nodes": [
                {"id": "solar_energy", "label": "Solar energy", "type": "concept"},
                {"id": "energy_policy", "label": "Energy policy", "type": "concept"},
            ],
            "edges": [
                {"source": "solar_energy", "target": "energy_policy", "type": "RELATED_TO"},
            ],
        },
        "citation_registry": {
            "citations": {
                1: {
                    "server": "cpr",
                    "tool": "run_query",
                    "title": "CPR passage UNFCCC.party.166.0",
                    "source_type": "Policy Document",
                    "description": "Passage 11285 in document UNFCCC.party.166.0 mentioning solar energy.",
                    "url": "",
                }
            }
        },
    }

    # Sanity check: make sure the mocked response matches the orchestrator contract.
    validate_final_response(sample_response)

    async def fake_process_chat_query(
        incoming_query: str,
        *,
        conversation_history: List[Dict[str, str]] | None = None,
        correlation_session_id: str | None = None,
        target_language: str | None = None,
    ) -> Dict[str, Any]:
        assert incoming_query == query
        # Return a deep copy to mirror the behaviour of real orchestrator output.
        return {
            key: (value.copy() if isinstance(value, dict) else value[:] if isinstance(value, list) else value)
            for key, value in sample_response.items()
        }

    async def fake_generate_embed(
        _: str,
        __: Dict[str, Any] | None = None,
        ___: Dict[str, Any] | None = None,
    ) -> Dict[str, str]:
        return {
            "relative_path": "kg/sample.html",
            "absolute_path": "/tmp/kg/sample.html",
            "url_path": "/static/kg/sample.html",
            "filename": "sample.html",
        }

    async def fake_fetch_kg_data(_: str) -> Dict[str, Any]:
        return {
            "concepts": [{"id": "solar_energy", "label": "Solar energy"}],
            "relationships": [
                {
                    "source_id": "solar_energy",
                    "target_id": "energy_policy",
                    "source_label": "Solar energy",
                    "target_label": "Energy policy",
                    "relationship_type": "RELATED_TO",
                    "formatted": "Solar energy -> Energy policy (RELATED_TO)",
                }
            ],
            "query_concepts": [],
            "query_concept_labels": [],
            "kg_extraction_method": "stub",
        }

    # Patch the API to consume the mocked orchestrator + supporting utilities.
    monkeypatch.setattr(api_server, "process_chat_query", fake_process_chat_query)
    monkeypatch.setattr(api_server.kg_generator, "generate_embed", fake_generate_embed)
    monkeypatch.setattr(api_server, "_fetch_kg_data", fake_fetch_kg_data)
    monkeypatch.setattr(api_server.session_store, "log_conversation", lambda *_, **__: None)

    async with AsyncClient(app=api_server.app, base_url="http://testserver") as client:
        response = await client.post(
            "/query",
            json={
                "query": query,
                "conversation_id": None,
                "reset": False,
                "language": None,
                "include_thinking": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()

    assert payload["query"] == query
    assert payload["modules"] == modules

    metadata = payload["metadata"]
    assert metadata["modules_count"] == len(modules)
    assert set(metadata["module_types"]) == {"text", "table", "numbered_citation_table"}
    assert metadata["kg_query_url"].endswith("query=Which%20climate%20policies%20reference%20solar%20energy?")

    # Ensure KG embed paths are exposed both at the top level and inside metadata.
    expected_embed_url = "http://testserver/static/kg/sample.html"
    assert payload["kg_embed_url"] == expected_embed_url
    assert metadata["kg_embed_url"] == expected_embed_url

    # Confirm KG context propagated through the API helpers.
    assert payload["concepts"] == [{"id": "solar_energy", "label": "Solar energy"}]
    assert payload["relationships"][0]["formatted"].startswith("Solar energy -> Energy policy")

    summary_module = next(module for module in modules if module["type"] == "text")
    assert isinstance(summary_module["texts"], list) and summary_module["texts"]


@pytest.mark.asyncio
async def test_stream_endpoint_emits_thinking_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure `/query/stream` emits thinking events and augments final payload."""

    import api_server
    from mcp.mcp_chat_v2 import validate_final_response

    query = "Which climate policies reference solar energy?"

    modules: List[Dict[str, Any]] = [
        {
            "type": "text",
            "heading": "Summary",
            "texts": ["- Solar policy documents cite expansion targets ^1^"]
        },
        {
            "type": "numbered_citation_table",
            "heading": "References",
            "columns": ["#", "Source", "ID/Tool", "Type", "Description", "SourceURL"],
            "rows": [
                [
                    "1",
                    "CPR passage UNFCCC.party.166.0",
                    "run_query",
                    "Policy Document",
                    "Passage 11285 in document UNFCCC.party.166.0 mentioning solar energy.",
                    "",
                ]
            ],
        },
    ]

    final_payload: Dict[str, Any] = {
        "query": query,
        "modules": modules,
        "metadata": {
            "modules_count": len(modules),
            "has_maps": False,
            "has_charts": False,
            "has_tables": True,
            "module_types": [module["type"] for module in modules],
            "kg_visualization_url": "/kg-viz",
            "kg_query_url": "/kg-viz?query=Which%20climate%20policies%20reference%20solar%20energy?",
        },
        "kg_context": {
            "nodes": [
                {"id": "solar_energy", "label": "Solar energy", "type": "concept"},
                {"id": "energy_policy", "label": "Energy policy", "type": "concept"},
            ],
            "edges": [
                {"source": "solar_energy", "target": "energy_policy", "type": "RELATED_TO"},
            ],
        },
        "citation_registry": {
            "citations": {
                1: {
                    "server": "cpr",
                    "tool": "run_query",
                    "title": "CPR passage UNFCCC.party.166.0",
                    "source_type": "Policy Document",
                    "description": "Passage 11285 in document UNFCCC.party.166.0 mentioning solar energy.",
                    "url": "",
                }
            }
        },
    }

    validate_final_response(final_payload)

    async def fake_stream_chat_query(*_: Any, **__: Any):
        yield {
            "type": "thinking",
            "data": {
                "message": "🔍 Checking concept database for query-related terms",
                "category": "search",
            },
        }
        yield {
            "type": "thinking",
            "data": {
                "message": '🔍 Relevant Fact Found: "Solar expansion commitments hit a record in 2024"',
                "category": "fact",
            },
        }
        yield {
            "type": "thinking_complete",
            "data": {
                "message": "✅ Found relevant policy concepts",
                "category": "search",
            },
        }
        yield {
            "type": "complete",
            "data": {
                key: (value.copy() if isinstance(value, dict) else value[:] if isinstance(value, list) else value)
                for key, value in final_payload.items()
            },
        }

    async def fake_generate_embed(*_: Any, **__: Any) -> Dict[str, str]:
        return {
            "relative_path": "kg/sample.html",
            "absolute_path": "/tmp/kg/sample.html",
            "url_path": "/static/kg/sample.html",
            "filename": "sample.html",
        }

    async def fake_fetch_kg_data(_: str) -> Dict[str, Any]:
        return {
            "concepts": [{"id": "solar_energy", "label": "Solar energy"}],
            "relationships": [
                {
                    "source_id": "solar_energy",
                    "target_id": "energy_policy",
                    "source_label": "Solar energy",
                    "target_label": "Energy policy",
                    "relationship_type": "RELATED_TO",
                    "formatted": "Solar energy -> Energy policy (RELATED_TO)",
                }
            ],
            "query_concepts": [],
            "query_concept_labels": [],
            "kg_extraction_method": "stub",
        }

    monkeypatch.setattr(api_server, "stream_chat_query", fake_stream_chat_query)
    monkeypatch.setattr(api_server.kg_generator, "generate_embed", fake_generate_embed)
    monkeypatch.setattr(api_server, "_fetch_kg_data", fake_fetch_kg_data)
    monkeypatch.setattr(api_server.session_store, "log_conversation", lambda *_, **__: None)

    events: List[Dict[str, Any]] = []
    async with AsyncClient(app=api_server.app, base_url="http://testserver") as client:
        async with client.stream(
            "POST",
            "/query/stream",
            json={
                "query": query,
                "conversation_id": None,
                "reset": False,
                "language": None,
            },
        ) as response:
            assert response.status_code == 200

            async for line in response.aiter_lines():
                if not line:
                    continue
                assert line.startswith("data: ")
                payload = json.loads(line[len("data: "):])
                events.append(payload)
                if payload.get("type") == "done":
                    break

    assert events[0]["type"] == "conversation_id"
    thinking_event = next(
        evt for evt in events if evt.get("type") == "thinking" and evt["data"].get("category") == "search"
    )
    assert thinking_event["data"]["category"] == "search"
    fact_event = next(
        evt for evt in events if evt.get("type") == "thinking" and evt["data"].get("category") == "fact"
    )
    assert fact_event["data"]["message"].startswith("🔍 Relevant Fact Found: \"")

    complete_event = next(evt for evt in events if evt.get("type") == "complete")
    complete_data = complete_event["data"]
    expected_embed_url = "http://testserver/static/kg/sample.html"
    assert complete_data["metadata"]["kg_embed_url"] == expected_embed_url
    assert complete_data["kg_embed_url"] == expected_embed_url
    assert complete_data["concepts"] == [{"id": "solar_energy", "label": "Solar energy"}]
    assert complete_data["relationships"][0]["relationship_type"] == "RELATED_TO"

    assert events[-1] == {"type": "done"}
