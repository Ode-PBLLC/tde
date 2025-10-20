"""Shared MCP response contracts for the simplified orchestrator.

The goal of this module is to codify the payload shape every MCP server
has to honour when implementing the new `run_query` flow.  Keeping the
schema centralised allows the orchestrator to:

* validate server responses aggressively,
* provide meaningful error messages when a server violates the contract,
* assign citation numbers deterministically, and
* keep the orchestration layer extremely small.

All models are Pydantic based for strong runtime validation while still
offering `.model_dump()` helpers for downstream consumers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


def _default_dict() -> Dict[str, Any]:
    """Factory for mutable default dictionaries."""

    return {}


def _default_list() -> List[Any]:
    """Factory for mutable default lists."""

    return []


class CitationPayload(BaseModel):
    """Metadata describing one citation emitted by a server."""

    id: str = Field(..., description="Server-scoped citation identifier")
    server: str = Field(..., description="Name of the emitting MCP server")
    tool: str = Field(..., description="Tool that produced the evidence")
    title: str = Field(..., description="Human readable source title")
    source_type: str = Field(..., description="Source category (Dataset, Report, etc.)")
    description: Optional[str] = Field(
        default=None, description="Short note that will appear in the table"
    )
    url: Optional[str] = Field(default=None, description="Public source URL")
    metadata: Dict[str, Any] = Field(default_factory=_default_dict)


class FactPayload(BaseModel):
    """Structured fact returned by a server."""

    id: str = Field(..., description="Server-scoped fact identifier")
    text: str = Field(..., description="Narrative sentence or summary of the fact")
    citation_id: str = Field(
        ..., description="Reference to `CitationPayload.id` within the same response"
    )
    kind: Literal["text", "table", "chart", "map", "metric"] = Field(
        default="text", description="Fact category for downstream formatting"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional structured payload for tables/charts"
    )
    metadata: Dict[str, Any] = Field(default_factory=_default_dict)


class ArtifactPayload(BaseModel):
    """Rich artifact (map, chart, attachment) produced by a server."""

    id: str = Field(..., description="Server-scoped artifact identifier")
    type: Literal["map", "chart", "table", "image", "attachment"]
    title: str = Field(..., description="Display title for the artifact")
    url: Optional[str] = Field(default=None, description="Static asset URL")
    geojson_url: Optional[str] = Field(
        default=None,
        description="Convenience field for GeoJSON exports; mirrors `url` when applicable.",
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None, description="Inline JSON payload for the artifact"
    )
    description: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=_default_dict)


class KnowledgeGraphPayload(BaseModel):
    """Minimal representation of the nodes/edges used by the server."""

    nodes: List[Dict[str, Any]] = Field(default_factory=_default_list)
    edges: List[Dict[str, Any]] = Field(default_factory=_default_list)
    url: Optional[str] = Field(
        default=None,
        description="Optional URL pointing to a serialized graph artefact",
    )


class MessagePayload(BaseModel):
    """Auxiliary message for the UI (warnings, limitations, etc.)."""

    level: Literal["info", "warning", "error"] = "info"
    text: str


class RunQueryResponse(BaseModel):
    """Canonical payload returned by every `run_query` implementation."""

    server: str
    tool: Literal["run_query"] = "run_query"
    query: str = Field(..., description="Original user query")
    facts: List[FactPayload] = Field(default_factory=_default_list)
    citations: List[CitationPayload] = Field(default_factory=_default_list)
    artifacts: List[ArtifactPayload] = Field(default_factory=_default_list)
    messages: List[MessagePayload] = Field(default_factory=_default_list)
    kg: KnowledgeGraphPayload = Field(default_factory=KnowledgeGraphPayload)
    next_actions: List[str] = Field(default_factory=_default_list)
    duration_ms: Optional[int] = Field(
        default=None, description="Optional execution time reported by the server"
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_citation_links(self) -> "RunQueryResponse":
        citation_ids = {citation.id for citation in self.citations}
        missing = {fact.citation_id for fact in self.facts} - citation_ids
        if missing:
            raise ValueError(
                f"Facts reference missing citation ids: {', '.join(sorted(missing))}"
            )
        return self


class QuerySupportPayload(BaseModel):
    """Response from the lightweight relevance probe (`query_support`)."""

    server: str
    query: str
    supported: bool
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 (no) and 1 (definitely)",
    )
    reasons: List[str] = Field(default_factory=_default_list)


class QueryContext(BaseModel):
    """Metadata shared with servers when invoking `query_support` or `run_query`."""

    query: str
    conversation: List[Dict[str, Any]] = Field(default_factory=_default_list)
    language: Optional[str] = None
    session_id: Optional[str] = None
    previous_user_message: Optional[str] = None
    previous_assistant_message: Optional[str] = None
    previous_response_modules: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


def parse_run_query_response(payload: Any) -> RunQueryResponse:
    """Validate and convert a raw tool payload into `RunQueryResponse`."""

    if isinstance(payload, RunQueryResponse):
        return payload
    if isinstance(payload, dict):
        return RunQueryResponse.model_validate(payload)
    raise TypeError(
        "run_query response must be a dict or RunQueryResponse model; "
        f"received type {type(payload)!r}"
    )


def parse_query_support_response(payload: Any) -> QuerySupportPayload:
    """Validate and convert a raw tool payload into `QuerySupportPayload`."""

    if isinstance(payload, QuerySupportPayload):
        return payload
    if isinstance(payload, dict):
        return QuerySupportPayload.model_validate(payload)
    raise TypeError(
        "query_support response must be a dict or QuerySupportPayload model; "
        f"received type {type(payload)!r}"
    )


__all__ = [
    "ArtifactPayload",
    "CitationPayload",
    "FactPayload",
    "KnowledgeGraphPayload",
    "MessagePayload",
    "QueryContext",
    "QuerySupportPayload",
    "RunQueryResponse",
    "parse_query_support_response",
    "parse_run_query_response",
    "ValidationError",
]
