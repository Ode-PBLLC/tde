"""Tests for the deforestation v2 MCP server."""

from mcp.servers_v2.deforestation_server_v2 import DeforestationServerV2


def test_deforestation_run_query_returns_map() -> None:
    server = DeforestationServerV2()
    response = server.handle_run_query(
        query="Where are deforestation hotspots?",
        context={"min_area_hectares": 3000},
    )

    assert response.server == "deforestation"
    assert response.facts, "Expected at least one fact from the deforestation server"

    map_artifacts = [artifact for artifact in response.artifacts if artifact.type == "map"]
    assert map_artifacts, "Expected a map artifact with GeoJSON output"
    assert any(
        artifact.geojson_url for artifact in map_artifacts
    ), "Expected at least one map artifact to expose a geojson_url"


def test_deforestation_run_query_infers_min_area_from_query() -> None:
    server = DeforestationServerV2()
    response = server.handle_run_query(
        query="Show me recent deforestation polygons larger than 3,000 hectares.",
        context={},
    )

    table_artifact = next(
        artifact for artifact in response.artifacts if artifact.type == "table"
    )
    rows = table_artifact.data.get("rows", [])
    assert rows, "Expected table rows for deforestation results"
    assert min(row[1] for row in rows) >= 3000, "Rows should respect inferred minimum area"
    assert any(
        "minimum area filter of 3000" in fact.text for fact in response.facts
    ), "Expected facts to note the inferred minimum area filter"
