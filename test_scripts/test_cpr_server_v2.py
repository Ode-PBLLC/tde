#!/usr/bin/env python3
"""
Comprehensive tests for mcp/servers_v2/cpr_server_v2.py

Tests the CPR Knowledge Graph v2 MCP server including:
- Tool registration and initialization
- Query support detection
- Concept operations
- Relationship traversal
- Passage retrieval
- run_query response format
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _install_fastmcp_stub():
    """Install a minimal FastMCP stub for testing."""
    if 'fastmcp' in sys.modules:
        return
    m = types.ModuleType('fastmcp')

    class FastMCP:
        def __init__(self, name: str):
            self.name = name
            self.tools = {}

        def tool(self, *args, **kwargs):
            def _decorator(func):
                # Store the tool function for testing
                self.tools[func.__name__] = func
                return func
            return _decorator

        def run(self):
            return None

    m.FastMCP = FastMCP
    sys.modules['fastmcp'] = m


def _install_support_intent_stub():
    """Install minimal SupportIntent stubs for testing."""

    @dataclass
    class SupportIntent:
        supported: bool
        score: float
        reasons: list[str]

    for module_name in (
        'mcp.support_intent',
        'mcp.servers_v2.support_intent',
    ):
        if module_name in sys.modules:
            continue
        support_module = types.ModuleType(module_name)
        support_module.SupportIntent = SupportIntent
        sys.modules[module_name] = support_module


def _create_mock_cpr_tools():
    """Create mock implementations for cpr_tools functions."""
    mock_tools = MagicMock()

    # Mock basic functions
    mock_tools.get_concepts.return_value = [
        "climate change",
        "renewable energy",
        "carbon emissions",
        "climate policy"
    ]

    mock_tools.check_concept_exists.return_value = True

    mock_tools.get_semantically_similar_concepts.return_value = [
        "global warming",
        "climate crisis"
    ]

    mock_tools.search_concepts_by_text.return_value = [
        {"label": "climate change", "wikibase_id": "Q1"},
        {"label": "climate policy", "wikibase_id": "Q2"}
    ]

    mock_tools.search_concepts_fuzzy.return_value = [
        {"label": "climate change", "score": 0.95, "wikibase_id": "Q1"}
    ]

    mock_tools.find_concept_matches_by_ngrams.return_value = [
        {"label": "climate change", "matched_ngrams": ["climate"], "wikibase_id": "Q1"}
    ]

    mock_tools.get_top_concepts_by_query.return_value = [
        {"label": "renewable energy", "similarity": 0.89, "wikibase_id": "Q3"}
    ]

    mock_tools.get_top_concepts_by_query_local.return_value = [
        {"label": "carbon emissions", "overlap_score": 0.75, "wikibase_id": "Q4"}
    ]

    mock_tools.get_alternative_labels.return_value = ["GHG emissions", "greenhouse gas"]

    mock_tools.get_description.return_value = "Emissions of carbon dioxide and other greenhouse gases"

    mock_tools.get_related_concepts.return_value = [
        {"label": "mitigation", "edge_type": "RELATED_TO", "wikibase_id": "Q5"}
    ]

    mock_tools.get_subconcepts.return_value = [
        {"label": "solar energy", "edge_type": "HAS_SUBCONCEPT", "wikibase_id": "Q6"}
    ]

    mock_tools.get_parent_concepts.return_value = [
        {"label": "energy policy", "edge_type": "HAS_SUBCONCEPT", "wikibase_id": "Q7"}
    ]

    mock_tools.get_concept_graph_neighbors.return_value = [
        {"neighbor": "adaptation", "edge_type": "RELATED_TO", "wikibase_id": "Q8"}
    ]

    mock_tools.find_concept_path_with_edges.return_value = [
        {
            "path": ["climate change", "mitigation", "renewable energy"],
            "edges": [("climate change", "mitigation"), ("mitigation", "renewable energy")]
        }
    ]

    mock_tools.find_concept_path_rich.return_value = [
        {"source": "climate change", "target": "renewable energy", "path_length": 2}
    ]

    mock_tools.explain_concept_relationship.return_value = {
        "source": "climate change",
        "target": "renewable energy",
        "relationship": "connected through mitigation strategies"
    }

    mock_tools.get_passages_mentioning_concept.return_value = [
        {
            "passage_id": "P1",
            "text": "Brazil's climate policy focuses on renewable energy...",
            "doc_id": "D1",
            "concept": "renewable energy"
        }
    ]

    mock_tools.passages_mentioning_both_concepts.return_value = [
        {
            "passage_id": "P2",
            "text": "Climate change mitigation through renewable sources...",
            "doc_id": "D2",
            "concepts": ["climate change", "renewable energy"]
        }
    ]

    mock_tools.get_dataset_metadata.return_value = {
        "total_concepts": 500,
        "total_passages": 10000,
        "total_documents": 200
    }

    mock_tools.get_available_datasets.return_value = [
        {"id": "cpr_brazil", "name": "CPR Brazil Policy Corpus"}
    ]

    mock_tools.get_dataset_content.return_value = {
        "dataset_id": "cpr_brazil",
        "content_summary": "Brazilian climate policy documents"
    }

    mock_tools.describe_server.return_value = {
        "server": "cpr",
        "status": "active",
        "tools_available": 25
    }

    mock_tools.debug_embedding_status.return_value = {
        "embeddings_available": True,
        "model": "text-embedding-ada-002"
    }

    mock_tools.get_semantic_debug_log.return_value = {
        "recent_queries": ["climate change", "renewable energy"],
        "log_count": 2
    }

    mock_tools.discover_policy_context_for_query.return_value = {
        "concepts": [
            {"label": "climate policy", "wikibase_id": "Q2"},
            {"label": "renewable energy", "wikibase_id": "Q3"}
        ],
        "neighbors": [
            {
                "source_label": "climate policy",
                "target_label": "mitigation",
                "edge_type": "RELATED_TO"
            }
        ],
        "passages": [
            {
                "passage_id": "P1",
                "text": "Brazil's NDC commits to reducing emissions by 43% by 2030...",
                "doc_id": "D1",
                "concept": "climate policy",
                "metadata": {"document_id": "NDC_2020"}
            }
        ]
    }

    return mock_tools


def test_server_initialization():
    """Test that the CPR server initializes correctly."""
    _install_fastmcp_stub()

    with patch('mcp.servers_v2.cpr_server_v2.tools') as mock_cpr_tools:
        mock_cpr_tools.configure_mock(**_create_mock_cpr_tools().__dict__)

        from mcp.servers_v2.cpr_server_v2 import CPRServerV2

        server = CPRServerV2()
        assert server.mcp.name == "cpr-server-v2"

        # Check that all expected tools are registered
        expected_tools = [
            'describe_capabilities',
            'query_support',
            'GetConcepts',
            'CheckConceptExists',
            'GetSemanticallySimilarConcepts',
            'SearchConceptsByText',
            'SearchConceptsFuzzy',
            'FindConceptMatchesByNgrams',
            'GetTopConceptsByQuery',
            'GetTopConceptsByQueryLocal',
            'GetAlternativeLabels',
            'GetDescription',
            'GetRelatedConcepts',
            'GetSubconcepts',
            'GetParentConcepts',
            'GetConceptGraphNeighbors',
            'FindConceptPathWithEdges',
            'FindConceptPathRich',
            'ExplainConceptRelationship',
            'GetPassagesMentioningConcept',
            'PassagesMentioningBothConcepts',
            'GetKGDatasetMetadata',
            'GetAvailableDatasets',
            'GetDatasetContent',
            'DescribeServer',
            'DebugEmbeddingStatus',
            'GetSemanticDebugLog',
            'run_query'
        ]

        for tool_name in expected_tools:
            assert tool_name in server.mcp.tools, f"Tool {tool_name} not registered"

        print(f"✓ Server initialized with {len(server.mcp.tools)} tools")
        return server


def test_capabilities_tool(server):
    """Test the describe_capabilities tool."""
    tool = server.mcp.tools['describe_capabilities']
    result = tool(format='json')
    data = json.loads(result)

    assert data['name'] == 'cpr'
    assert 'knowledge graph' in data['description'].lower()
    assert data['version'] == '2.0.0'
    assert 'policy' in data['tags']
    print("✓ describe_capabilities tool works")


def test_query_support_tool(server):
    """Test query support detection."""
    tool = server.mcp.tools['query_support']

    # Test policy-related query
    result = tool(query="What are Brazil's climate policies?", context={})
    data = json.loads(result)
    assert data['server'] == 'cpr'
    assert data['supported'] == True
    assert data['score'] > 0.5
    print("✓ query_support detects policy queries")

    # Test non-policy query
    result = tool(query="What is the weather today?", context={})
    data = json.loads(result)
    assert data['supported'] == False
    assert data['score'] < 0.5
    print("✓ query_support rejects non-policy queries")


def test_concept_tools(server):
    """Test concept lookup and search tools."""
    # Test GetConcepts
    concepts = server.mcp.tools['GetConcepts']()
    assert isinstance(concepts, list)
    assert len(concepts) > 0
    print(f"✓ GetConcepts returns {len(concepts)} concepts")

    # Test CheckConceptExists
    exists = server.mcp.tools['CheckConceptExists'](concept="climate change")
    assert exists == True
    print("✓ CheckConceptExists works")

    # Test SearchConceptsByText
    results = server.mcp.tools['SearchConceptsByText'](text="climate", limit=5)
    assert isinstance(results, list)
    assert len(results) > 0
    print(f"✓ SearchConceptsByText found {len(results)} results")

    # Test GetSemanticallySimilarConcepts
    similar = server.mcp.tools['GetSemanticallySimilarConcepts'](concept="climate change")
    assert isinstance(similar, list)
    print(f"✓ GetSemanticallySimilarConcepts found {len(similar)} similar concepts")


def test_relationship_tools(server):
    """Test relationship and graph traversal tools."""
    # Test GetRelatedConcepts
    related = server.mcp.tools['GetRelatedConcepts'](concept="climate change")
    assert isinstance(related, list)
    assert len(related) > 0
    print(f"✓ GetRelatedConcepts found {len(related)} related concepts")

    # Test GetSubconcepts
    subconcepts = server.mcp.tools['GetSubconcepts'](concept="renewable energy")
    assert isinstance(subconcepts, list)
    print(f"✓ GetSubconcepts found {len(subconcepts)} subconcepts")

    # Test GetParentConcepts
    parents = server.mcp.tools['GetParentConcepts'](concept="solar energy")
    assert isinstance(parents, list)
    print(f"✓ GetParentConcepts found {len(parents)} parent concepts")

    # Test path finding
    paths = server.mcp.tools['FindConceptPathWithEdges'](
        source_concept="climate change",
        target_concept="renewable energy",
        max_len=5
    )
    assert isinstance(paths, list)
    print(f"✓ FindConceptPathWithEdges found {len(paths)} paths")


def test_passage_tools(server):
    """Test passage retrieval tools."""
    # Test GetPassagesMentioningConcept
    passages = server.mcp.tools['GetPassagesMentioningConcept'](
        concept="renewable energy",
        limit=3
    )
    assert isinstance(passages, list)
    assert len(passages) > 0
    print(f"✓ GetPassagesMentioningConcept found {len(passages)} passages")

    # Test PassagesMentioningBothConcepts
    co_passages = server.mcp.tools['PassagesMentioningBothConcepts'](
        concept_a="climate change",
        concept_b="renewable energy",
        limit=2
    )
    assert isinstance(co_passages, list)
    print(f"✓ PassagesMentioningBothConcepts found {len(co_passages)} co-occurrences")


def test_metadata_tools(server):
    """Test metadata and debug tools."""
    # Test GetKGDatasetMetadata
    metadata = server.mcp.tools['GetKGDatasetMetadata']()
    assert isinstance(metadata, dict)
    assert 'total_concepts' in metadata
    print(f"✓ GetKGDatasetMetadata reports {metadata.get('total_concepts', 0)} concepts")

    # Test DescribeServer
    server_info = server.mcp.tools['DescribeServer']()
    assert isinstance(server_info, dict)
    assert server_info['server'] == 'cpr'
    print("✓ DescribeServer works")

    # Test DebugEmbeddingStatus
    embed_status = server.mcp.tools['DebugEmbeddingStatus']()
    assert isinstance(embed_status, dict)
    print(f"✓ DebugEmbeddingStatus: embeddings={embed_status.get('embeddings_available')}")


def test_run_query_tool(server):
    """Test the main run_query tool with full response validation."""
    tool = server.mcp.tools['run_query']

    # Execute run_query
    result = tool(
        query="What are Brazil's renewable energy policies?",
        context={"test": True}
    )

    # Parse and validate response
    data = json.loads(result)

    # Check required fields
    assert data['server'] == 'cpr'
    assert data['query'] == "What are Brazil's renewable energy policies?"
    assert 'facts' in data
    assert 'citations' in data
    assert 'artifacts' in data
    assert 'messages' in data
    assert 'kg' in data

    # Validate facts structure
    facts = data['facts']
    assert isinstance(facts, list)
    if facts:
        fact = facts[0]
        assert 'id' in fact
        assert 'text' in fact
        assert 'citation_id' in fact
        print(f"✓ run_query returned {len(facts)} facts")

    # Validate citations structure
    citations = data['citations']
    assert isinstance(citations, list)
    assert len(citations) > 0
    citation = citations[0]
    assert 'id' in citation
    assert citation.get('server') == 'cpr'
    assert citation.get('tool') == 'run_query'
    assert citation.get('title')
    print(f"✓ run_query returned {len(citations)} citations")

    # Validate artifacts if present
    artifacts = data['artifacts']
    assert isinstance(artifacts, list)
    if artifacts:
        artifact = artifacts[0]
        assert 'id' in artifact
        assert 'type' in artifact
        assert 'title' in artifact
        assert 'data' in artifact
        print(f"✓ run_query returned {len(artifacts)} artifacts")

    # Validate knowledge graph
    kg = data['kg']
    assert 'nodes' in kg
    assert 'edges' in kg
    assert isinstance(kg['nodes'], list)
    assert isinstance(kg['edges'], list)

    if kg['nodes']:
        node = kg['nodes'][0]
        assert 'id' in node
        assert 'label' in node
        assert 'type' in node
        print(f"✓ Knowledge graph has {len(kg['nodes'])} nodes and {len(kg['edges'])} edges")

    print("✓ run_query response structure is valid")


def test_run_query_edge_cases(server):
    """Test run_query with edge cases."""
    tool = server.mcp.tools['run_query']

    # Test with empty query
    result = tool(query="", context={})
    data = json.loads(result)
    assert data['server'] == 'cpr'
    assert isinstance(data.get('facts'), list)
    print(f"✓ run_query handles empty query (facts={len(data['facts'])})")

    # Test with very specific query
    result = tool(
        query="Show me passages about solar energy regulation in São Paulo",
        context={"detailed": True}
    )
    data = json.loads(result)
    assert data['server'] == 'cpr'
    assert 'facts' in data
    print("✓ run_query handles specific queries")


def test_create_server_function():
    """Test the create_server entry point."""
    with patch('mcp.servers_v2.cpr_server_v2.tools') as mock_cpr_tools:
        mock_cpr_tools.configure_mock(**_create_mock_cpr_tools().__dict__)

        from mcp.servers_v2.cpr_server_v2 import create_server

        mcp = create_server()
        assert mcp.name == "cpr-server-v2"
        print("✓ create_server() returns FastMCP instance")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing CPR Server V2")
    print("=" * 60)

    _install_fastmcp_stub()
    _install_support_intent_stub()

    mock_tools = _create_mock_cpr_tools()

    # First import the module to make it available for patching
    import mcp.servers_v2.cpr_server_v2

    from mcp.servers_v2.cpr_server_v2 import CPRServerV2

    SupportIntent = sys.modules['mcp.servers_v2.support_intent'].SupportIntent

    def _mock_classify_support(self, query: str) -> SupportIntent:
        lowered = (query or "").lower()
        is_policy = any(keyword in lowered for keyword in ["policy", "policies", "ndc", "climate"])
        score = 0.9 if is_policy else 0.1
        reasons = ["mocked"]
        return SupportIntent(supported=is_policy, score=score, reasons=reasons)

    with patch.object(mcp.servers_v2.cpr_server_v2, 'tools', mock_tools), \
        patch.object(CPRServerV2, '_classify_support', _mock_classify_support):

        # Initialize server once for all tests
        server = test_server_initialization()

        # Run all test suites
        test_capabilities_tool(server)
        test_query_support_tool(server)
        test_concept_tools(server)
        test_relationship_tools(server)
        test_passage_tools(server)
        test_metadata_tools(server)
        test_run_query_tool(server)
        test_run_query_edge_cases(server)
        test_create_server_function()

    print("=" * 60)
    print("✅ All CPR Server V2 tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
