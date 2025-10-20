"""Tests for the v2 meta MCP server."""

from mcp.servers_v2.meta_server_v2 import MetaServerV2


def test_meta_server_v2_returns_dataset_fact() -> None:
    server = MetaServerV2()
    response = server.handle_run_query(
        query="What can you tell me about the TransitionZero solar asset mapper?",
        context={},
    )

    assert response.facts, "Expected at least one fact in the response"
    assert any(
        citation.metadata.get("dataset_id") == "solar_facilities"
        for citation in response.citations
    ), "Expected solar dataset citation to be present"
    assert any(
        fact.text.lower().startswith("transitionzero solar") or "solar" in fact.text.lower()
        for fact in response.facts
    ), "Expected a fact describing the solar dataset"
    assert any(
        action.startswith("Call GetSolarDatasetMeta")
        for action in response.next_actions
    ), "Expected next action to reference the solar helper tool"


def test_meta_server_v2_handles_collaborator_queries() -> None:
    server = MetaServerV2()
    response = server.handle_run_query(
        query="Who are the collaborators behind this project?",
        context={},
    )

    assert response.facts, "Expected collaborator fact for project question"
    assert any(
        citation.id == "orgs::collaborative" for citation in response.citations
    ), "Expected collaborator citation to be emitted"
