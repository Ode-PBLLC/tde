"""Regression tests for citation URL hydration in mcp_chat_v2."""

from __future__ import annotations

import pytest

from mcp.contracts_v2 import CitationPayload, FactPayload, RunQueryResponse
from mcp.mcp_chat_v2 import MultiServerClient, SimpleOrchestrator


@pytest.mark.asyncio
async def test_populate_citation_urls_hydrates_cpr_sources() -> None:
    client = MultiServerClient()
    orchestrator = SimpleOrchestrator(client)

    citation = CitationPayload(
        id="passage_UNFCCC.party.166.0",
        server="cpr",
        tool="run_query",
        title="CPR passage UNFCCC.party.166.0",
        source_type="Policy Document",
        description="Passage highlighting solar commitments.",
        url=None,
    )
    fact = FactPayload(
        id="fact-1",
        text="Brazil's NDC discusses solar energy expansion.",
        citation_id=citation.id,
    )
    response = RunQueryResponse(
        server="cpr",
        query="Which policies cite solar energy?",
        facts=[fact],
        citations=[citation],
    )

    orchestrator._populate_citation_urls([response])

    assert response.citations[0].url == "https://www.climatepolicyradar.org/"
