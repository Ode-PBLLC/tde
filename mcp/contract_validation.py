"""Validation helpers for the MCP response contracts.

These utilities provide a single place to assert that both the per-server
`run_query` payloads and the orchestrator's final response honour the
strict contract we agreed on. They raise
:class:`ContractValidationError` with actionable error messages so the
callers (tests, CI harnesses, or debug scripts) can fail early when a
payload drifts away from the schema.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from pydantic import ValidationError

from .contracts_v2 import (
    QuerySupportPayload,
    RunQueryResponse,
    parse_run_query_response,
)


class ContractValidationError(ValueError):
    """Raised when a payload violates the response contract."""


def _fail(message: str) -> None:
    raise ContractValidationError(message)


def validate_run_query_response(payload: Any) -> RunQueryResponse:
    """Validate and return a :class:`RunQueryResponse` instance.

    Besides the structural checks handled by :mod:`pydantic`, ensure that
    payloads include at least one fact and citation so the orchestrator
    can build a numbered citation table without guessing.
    """

    try:
        response = parse_run_query_response(payload)
    except ValidationError as exc:  # pragma: no cover - exercised in tests
        raise ContractValidationError(str(exc)) from exc

    if not response.facts:
        _fail("run_query response must include at least one fact")
    if not response.citations:
        _fail("run_query response must include at least one citation entry")

    cited_ids = {fact.citation_id for fact in response.facts}
    available_ids = {citation.id for citation in response.citations}
    missing = cited_ids - available_ids
    if missing:
        _fail(
            "run_query facts reference missing citation ids: "
            + ", ".join(sorted(missing))
        )

    return response


def validate_query_support_response(payload: Any) -> QuerySupportPayload:
    """Ensure query_support payloads respect the schema and basic rules."""

    from .contracts_v2 import parse_query_support_response

    response = parse_query_support_response(payload)
    if response.supported and response.score <= 0:
        _fail("supported query_support responses must provide score > 0")
    return response


def _find_modules_by_type(modules: Sequence[Mapping[str, Any]], module_type: str) -> list:
    return [m for m in modules if str(m.get("type")) == module_type]


def validate_final_response(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate an orchestrator response destined for the API layer.

    The function focuses on contract-critical fields: top-level keys,
    module structure (especially the numbered citation table), metadata,
    and knowledge-graph context. Optional embellishments (e.g., thinking
    traces) are ignored.
    """

    if not isinstance(payload, Mapping):
        _fail("final response must be a mapping/dict")

    for key in ("query", "modules", "metadata", "kg_context"):
        if key not in payload:
            _fail(f"final response missing required key '{key}'")

    query = payload["query"]
    if not isinstance(query, str) or not query.strip():
        _fail("query must be a non-empty string")

    modules = payload["modules"]
    if not isinstance(modules, Sequence) or not modules:
        _fail("modules must be a non-empty list")

    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        _fail("metadata must be a dictionary")
    required_metadata_keys = {"modules_count", "has_maps", "has_charts", "has_tables"}
    missing_meta = required_metadata_keys - metadata.keys()
    if missing_meta:
        _fail(
            "metadata missing required keys: " + ", ".join(sorted(missing_meta))
        )

    kg_context = payload["kg_context"]
    if not isinstance(kg_context, Mapping):
        _fail("kg_context must be a dictionary")
    for key in ("nodes", "edges"):
        if key not in kg_context:
            _fail(f"kg_context missing '{key}' list")
        if not isinstance(kg_context[key], Sequence):
            _fail(f"kg_context['{key}'] must be a list")

    # Module-specific checks -------------------------------------------------
    citation_tables = _find_modules_by_type(modules, "numbered_citation_table")
    if not citation_tables:
        _fail("modules must include a numbered_citation_table module")
    citation_table = citation_tables[-1]
    _validate_citation_table(citation_table)

    for map_module in _find_modules_by_type(modules, "map"):
        _validate_map_module(map_module)

    # Table detection considers the citation table as well
    if metadata.get("modules_count") != len(modules):
        _fail("metadata.modules_count does not match number of modules")

    return payload


def _validate_citation_table(module: Mapping[str, Any]) -> None:
    if not isinstance(module, Mapping):
        _fail("citation module must be a dict")
    columns = module.get("columns")
    rows = module.get("rows")
    allow_empty = bool(module.get("allow_empty"))
    expected_columns = ["#", "Source", "ID/Tool", "Type", "Description", "SourceURL"]
    if columns != expected_columns:
        _fail(
            "numbered_citation_table columns must be "
            + str(expected_columns)
        )
    if not isinstance(rows, Sequence):
        _fail("numbered_citation_table rows must be a list")
    if not rows and not allow_empty:
        _fail("numbered_citation_table must include at least one row")
    for row in rows:
        if not isinstance(row, Sequence) or len(row) != len(expected_columns):
            _fail("every citation row must match the column structure")
        number = row[0]
        if not isinstance(number, str) or not number.strip():
            _fail("citation row numbers must be non-empty strings")


def _validate_map_module(module: Mapping[str, Any]) -> None:
    if not isinstance(module, Mapping):
        _fail("map module must be a dict")
    map_type = module.get("mapType")
    if map_type == "geojson_url":
        geojson_url = module.get("geojson_url")
        if not isinstance(geojson_url, str) or not geojson_url.strip():
            _fail("map module (geojson_url) requires a non-empty geojson_url field")
    elif map_type == "geojson":
        geojson = module.get("geojson")
        if not isinstance(geojson, Mapping):
            _fail("inline geojson map modules must include a geojson object")
    else:
        _fail("map module must specify mapType 'geojson_url' or 'geojson'")


__all__ = [
    "ContractValidationError",
    "validate_final_response",
    "validate_query_support_response",
    "validate_run_query_response",
]
