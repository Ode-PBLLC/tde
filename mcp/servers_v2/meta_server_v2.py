"""Meta MCP server implementing the v2 `run_query` contract.

This server serves project-level metadata drawn from ``static/meta``.  It
focuses on lightweight retrieval rather than generation: user queries are
matched to relevant dataset or collaborator entries and returned as
structured facts with citations so the orchestrator can surface them to the
user without additional processing.

The server intentionally keeps the tool surface small.  Three helper tools
expose the primary datasets (solar facilities, deforestation polygons, and
GIST impact data) while ``run_query`` stitches together responses depending
on the incoming question.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

if load_dotenv:
    try:  # pragma: no cover - best effort
        load_dotenv()
    except Exception as exc:  # pragma: no cover
        print(f"[meta-server] Warning: load_dotenv failed: {exc}")

if __package__ in {None, ""}:  # pragma: no cover - direct execution helper
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from mcp.contracts_v2 import (  # type: ignore
        CitationPayload,
        FactPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from mcp.servers_v2.base import RunQueryMixin  # type: ignore
else:  # pragma: no cover - package execution
    from ..contracts_v2 import (
        CitationPayload,
        FactPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from ..servers_v2.base import RunQueryMixin


PRIMARY_DATASET_IDS = (
    "solar_facilities",
    "brazil_deforestation",
    "gist_multi_dataset",
)

INTENT_CONTEXT_KEY = "meta::intent"

DATASET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "solar_facilities": ("tz-sam", "transition zero", "solar asset mapper", "solar dataset"),
    "brazil_deforestation": (
        "deforestation",
        "prodes",
        "amazon deforestation",
        "forest loss",
    ),
    "gist_multi_dataset": ("gist", "biodiversity", "company impacts", "scope 3"),
    "climate_policy_radar": ("cpr", "policy radar", "policy dataset", "ndc"),
    "climate_policy_radar_concept_store": ("concept store", "knowledge graph", "cpr concepts"),
    "admin": ("administrative", "ibge", "municipal", "boundaries"),
    "lse": ("ndc align", "lse", "alignment"),
    "meta": ("meta dataset", "metadata"),
    "spa": ("science panel", "amazon assessment", "spa report"),
}


def _now_iso() -> str:
    """Return a truncated UTC timestamp for payload metadata."""

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class DatasetRecord:
    """Typed view of a dataset entry from ``datasets.json``."""

    id: str
    title: str
    description: str
    source: Optional[str]
    last_updated: Optional[str]
    citation: Optional[str]
    raw: Dict[str, object]
    aliases: Tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> Optional["DatasetRecord"]:
        """Convert a raw JSON payload into a :class:`DatasetRecord`."""

        dataset_id = str(payload.get("id", "")).strip()
        title = str(payload.get("title", "")).strip()
        description = str(payload.get("description", payload.get("long_description", ""))).strip()
        if not dataset_id or not title:
            return None

        return cls(
            id=dataset_id,
            title=title,
            description=description,
            source=str(payload.get("source")) if payload.get("source") else None,
            last_updated=str(payload.get("last_updated")) if payload.get("last_updated") else None,
            citation=str(payload.get("citation")) if payload.get("citation") else None,
            raw=payload,
            aliases=DATASET_ALIASES.get(dataset_id, tuple()),
        )


@dataclass
class MetaIntent:
    """Structured interpretation of a user query produced by an LLM."""

    meta_relevant: bool
    dataset_ids: Tuple[str, ...]
    include_orgs: bool
    needs_methodology: bool
    reason: Optional[str]

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        allowed_dataset_ids: Tuple[str, ...],
    ) -> "MetaIntent":
        meta_relevant = bool(data.get("meta_relevant", False))
        raw_dataset_ids = data.get("dataset_ids", [])
        dataset_ids: Tuple[str, ...]
        if isinstance(raw_dataset_ids, list):
            dataset_ids = tuple(
                dataset_id
                for dataset_id in raw_dataset_ids
                if isinstance(dataset_id, str) and dataset_id in allowed_dataset_ids
            )
        else:
            dataset_ids = tuple()

        include_orgs = bool(data.get("include_orgs", False))
        needs_methodology = bool(data.get("needs_methodology", False))
        reason = str(data.get("reason")) if data.get("reason") else None

        return cls(
            meta_relevant=meta_relevant,
            dataset_ids=dataset_ids,
            include_orgs=include_orgs,
            needs_methodology=needs_methodology,
            reason=reason,
        )


class MetaServerV2(RunQueryMixin):
    """FastMCP server exposing project metadata under the v2 contract."""

    def __init__(self) -> None:
        self.mcp = FastMCP("meta-server-v2")
        self.root = Path(__file__).resolve().parents[2]
        self.static_dir = self.root / "static" / "meta"

        self.datasets: Dict[str, DatasetRecord] = self._load_datasets()
        self.org_payload = self._load_json(self.static_dir / "orgs.json")
        self.methodology_summary = self._load_methodology_summary()

        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential issues
                print(f"[meta-server] Warning: Anthropic client unavailable: {exc}")

        self._register_capabilities_tool()
        self._register_dataset_tools()
        self._register_query_support_tool()
        self._register_run_query_tool()

    # ------------------------------------------------------------------ loaders
    def _load_json(self, path: Path) -> Dict[str, object]:
        """Best-effort JSON loader returning a dictionary."""

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _load_datasets(self) -> Dict[str, DatasetRecord]:
        """Load dataset metadata from ``datasets.json`` into a lookup table."""

        datasets_path = self.static_dir / "datasets.json"
        payload = self._load_json(datasets_path)
        items = payload.get("items") if isinstance(payload.get("items"), list) else []

        records: Dict[str, DatasetRecord] = {}
        for raw in items:  # type: ignore[assignment]
            if not isinstance(raw, dict):
                continue
            record = DatasetRecord.from_payload(raw)
            if record:
                records[record.id] = record
        return records

    def _load_methodology_summary(self) -> Optional[str]:
        """Return the first descriptive paragraph from the methodology markdown."""

        path = self.static_dir / "methodology_and_datasets.md"
        if not path.exists():
            return None

        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None

        paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip() and not segment.strip().startswith("#")]
        return paragraphs[0] if paragraphs else None

    # ------------------------------------------------------------------ tool registration
    def _register_capabilities_tool(self) -> None:
        @self.mcp.tool()
        def describe_capabilities(format: str = "json") -> str:  # type: ignore[misc]
            """Return human-readable metadata describing this server."""

            payload = {
                "name": "meta",
                "version": "2.0.0",
                "description": "Static project metadata aggregator with dataset summaries.",
                "static_dir": str(self.static_dir),
                "primary_tools": [
                    "GetSolarDatasetMeta",
                    "GetDeforestationDatasetMeta",
                    "GetGistDatasetMeta",
                    "run_query",
                ],
                "primary_dataset_ids": list(PRIMARY_DATASET_IDS),
                "generated_at": _now_iso(),
            }

            return json.dumps(payload) if format == "json" else str(payload)

    def _register_dataset_tools(self) -> None:
        @self.mcp.tool()
        def GetSolarDatasetMeta() -> Dict[str, object]:  # type: ignore[misc]
            """Return metadata for the TransitionZero Solar Asset Mapper dataset."""

            return self._dataset_payload("solar_facilities")

        @self.mcp.tool()
        def GetDeforestationDatasetMeta() -> Dict[str, object]:  # type: ignore[misc]
            """Return metadata for the PRODES deforestation polygons dataset."""

            return self._dataset_payload("brazil_deforestation")

        @self.mcp.tool()
        def GetGistDatasetMeta() -> Dict[str, object]:  # type: ignore[misc]
            """Return metadata for the GIST environmental impact dataset."""

            return self._dataset_payload("gist_multi_dataset")

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> Dict[str, object]:  # type: ignore[misc]
            """Estimate whether this server can address the incoming question."""

            intent = self._classify_query(query, context)

            if intent:
                supported = intent.meta_relevant
                score = 0.9 if supported else 0.0
                reasons = [intent.reason] if intent.reason else []
                if not reasons:
                    if supported:
                        reasons.append("LLM classified as project/tool metadata question")
                    else:
                        reasons.append("LLM classified as non-meta question")
            else:
                supported = True
                score = 0.2
                reasons = ["LLM unavailable; returning default metadata summary"]

            return {
                "server": "meta",
                "query": query,
                "supported": supported,
                "score": round(score, 3),
                "reasons": reasons,
            }

    # ------------------------------------------------------------------ helpers
    def _dataset_payload(self, dataset_id: str) -> Dict[str, object]:
        """Return a standard payload for dataset helper tools."""

        record = self.datasets.get(dataset_id)
        return {
            "schema_version": "1.0",
            "generated_at": _now_iso(),
            "dataset_id": dataset_id,
            "dataset": record.raw if record else None,
            "available_ids": sorted(self.datasets.keys()),
        }

    def _format_dataset_fact(self, record: DatasetRecord) -> str:
        """Compose a readable fact summarising a dataset entry."""

        parts: List[str] = [f"{record.title} covers {record.description}" if record.description else record.title]
        if record.last_updated:
            parts.append(f"Last updated {record.last_updated}.")
        if record.source:
            parts.append(f"Source: {record.source}.")
        return " ".join(parts)

    def _llm_query_intent(
        self,
        query: str,
        *,
        allowed_dataset_ids: Tuple[str, ...],
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to describe the query intent using structured JSON."""

        if not self._anthropic_client:
            return None

        dataset_catalog_lines: List[str] = []
        for dataset_id in allowed_dataset_ids:
            record = self.datasets.get(dataset_id)
            if not record:
                continue
            alias_text = f" Aliases: {', '.join(record.aliases)}." if record.aliases else ""
            description = record.description.replace("\n", " ") if record.description else ""
            truncated_description = (description[:220] + "...") if len(description) > 220 else description
            dataset_catalog_lines.append(
                f"- {dataset_id}: {record.title}. {truncated_description}{alias_text}"
            )

        dataset_catalog = "\n".join(dataset_catalog_lines)

        prompt = (
            "You classify questions about the Transition Digital project and its datasets."
            " Decide if the query is about the project/tool itself (meta) and identify"
            " which datasets, if any, are explicitly or implicitly referenced."
            " Return strict JSON with the following shape:\n"
            "{\n"
            "  \"meta_relevant\": true|false,\n"
            "  \"dataset_ids\": [<zero or more dataset ids from the allowed list>],\n"
            "  \"include_orgs\": true|false,\n"
            "  \"needs_methodology\": true|false,\n"
            "  \"reason\": \"short justification\"\n"
            "}\n"
            "If no dataset is referenced, return an empty list. Only choose ids from this list:\n"
            f"{', '.join(allowed_dataset_ids)}\n"
            "Dataset catalog:\n"
            f"{dataset_catalog}\n\n"
            f"Query: {query}\n"
            "Respond with JSON only."
        )

        try:
            response = self._anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=512,
                temperature=0,
                system="Respond with valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        except Exception as exc:  # pragma: no cover - network failures
            print(f"[meta-server] LLM intent classification failed: {exc}")
            return None

        def _parse_json(blob: str) -> Optional[Dict[str, Any]]:
            try:
                parsed = json.loads(blob)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None

        data = _parse_json(text)
        if not data:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                data = _parse_json(text[start : end + 1])

        if not data:
            print(f"[meta-server] LLM intent classification returned non-JSON: {text}")
            return None

        if "reason" not in data and "explanation" in data:
            data["reason"] = data.get("explanation")

        return data

    def _classify_query(self, query: str, context: dict) -> Optional[MetaIntent]:
        """Return a cached or freshly-built intent description for the query."""

        allowed_dataset_ids = tuple(sorted(self.datasets.keys()))

        cached_payload: Optional[Dict[str, Any]] = None
        if isinstance(context, dict):
            raw_cached = context.get(INTENT_CONTEXT_KEY)
            if isinstance(raw_cached, dict):
                cached_payload = raw_cached

        payload = cached_payload or self._llm_query_intent(query, allowed_dataset_ids=allowed_dataset_ids)
        if not payload:
            return None

        intent = MetaIntent.from_dict(payload, allowed_dataset_ids=allowed_dataset_ids)

        if isinstance(context, dict) and not cached_payload:
            context[INTENT_CONTEXT_KEY] = payload

        return intent

    def _org_fact(self) -> Optional[Tuple[FactPayload, CitationPayload]]:
        """Build a fact/citation pair describing Transition Digital collaborators."""

        items = self.org_payload.get("items")
        if not isinstance(items, list) or not items:
            return None

        primary = items[0]
        if not isinstance(primary, dict):
            return None

        collab = primary.get("name_of_collaborative", {})
        name = collab.get("name") if isinstance(collab, dict) else None
        tagline = collab.get("tagline") if isinstance(collab, dict) else None
        founders = primary.get("founding_collaborators") if isinstance(primary.get("founding_collaborators"), list) else []

        founder_names = [entry.get("name") for entry in founders if isinstance(entry, dict) and entry.get("name")]
        summary_parts = []
        if name:
            summary_parts.append(f"{name} brings together {', '.join(founder_names)}." if founder_names else f"{name} coordinates the project.")
        if tagline:
            summary_parts.append(tagline)

        if not summary_parts:
            return None

        citation = CitationPayload(
            id="orgs::collaborative",
            server="meta",
            tool="run_query",
            title=name or "Transition Digital collaborators",
            source_type="Metadata",
            description="Information about Transition Digital and project collaborators",
            url=collab.get("url") if isinstance(collab, dict) else None,
            metadata={"path": "static/meta/orgs.json"},
        )
        fact = FactPayload(
            id="org_fact_1",
            text=" ".join(summary_parts),
            citation_id=citation.id,
        )
        return fact, citation

    # ------------------------------------------------------------------ run_query implementation
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        start = time.time()
        intent = self._classify_query(query, context)
        facts: List[FactPayload] = []
        citations: List[CitationPayload] = []
        next_actions: List[str] = []
        dataset_ids = intent.dataset_ids if intent else tuple()

        for idx, dataset_id in enumerate(dataset_ids, start=1):
            record = self.datasets.get(dataset_id)
            if not record:
                continue

            citation_id = f"dataset::{record.id}"
            citation = CitationPayload(
                id=citation_id,
                server="meta",
                tool="run_query",
                title=record.title,
                source_type="Dataset",
                description=(record.description[:160] + "...") if len(record.description) > 160 else record.description,
                url=record.source,
                metadata={"dataset_id": record.id},
            )
            fact = FactPayload(
                id=f"dataset_fact_{idx}",
                text=self._format_dataset_fact(record),
                citation_id=citation_id,
            )
            citations.append(citation)
            facts.append(fact)
            helper_tool = self._helper_tool_for_dataset(record.id)
            if helper_tool:
                next_actions.append(f"Call {helper_tool} for the raw metadata entry.")

        include_orgs = bool(intent.include_orgs) if intent else False
        if include_orgs:
            org_fact = self._org_fact()
            if org_fact:
                fact, citation = org_fact
                facts.append(fact)
                citations.append(citation)

        add_methodology = False
        if intent:
            add_methodology = intent.needs_methodology or not dataset_ids
        else:
            add_methodology = True

        if add_methodology or not facts:
            summary_text = self.methodology_summary or (
                "Transition Digital aggregates climate datasets and exposes them via MCP servers with curated metadata."
            )
            citation = CitationPayload(
                id="meta::methodology",
                server="meta",
                tool="run_query",
                title="Methodology and datasets overview",
                source_type="Documentation",
                description="Project methodology and data sources documentation",
                url=None,
                metadata={"path": "static/meta/methodology_and_datasets.md"},
            )
            fact = FactPayload(
                id="meta_fact_methodology",
                text=summary_text,
                citation_id=citation.id,
            )
            citations.append(citation)
            facts.append(fact)

        duration_ms = int((time.time() - start) * 1000)

        messages: List[MessagePayload] = []
        if not intent:
            messages.append(
                MessagePayload(
                    level="info",
                    text="Returned default metadata summary because LLM intent classification was unavailable.",
                )
            )
        elif not intent.meta_relevant:
            messages.append(
                MessagePayload(
                    level="info",
                    text="LLM classified the query as non-meta; provided fallback project overview.",
                )
            )
        elif not dataset_ids:
            messages.append(
                MessagePayload(
                    level="info",
                    text="LLM intent did not reference specific datasets; included project overview instead.",
                )
            )

        return RunQueryResponse(
            server="meta",
            query=query,
            facts=facts,
            citations=citations,
            messages=messages,
            next_actions=next_actions,
            duration_ms=duration_ms,
        )

    def _helper_tool_for_dataset(self, dataset_id: str) -> Optional[str]:
        """Return the helper tool name for a dataset if one exists."""

        mapping = {
            "solar_facilities": "GetSolarDatasetMeta",
            "brazil_deforestation": "GetDeforestationDatasetMeta",
            "gist_multi_dataset": "GetGistDatasetMeta",
        }
        return mapping.get(dataset_id)


def main() -> None:
    """Entrypoint used by FastMCP when the script is executed directly."""

    server = MetaServerV2()
    server.mcp.run()


if __name__ == "__main__":  # pragma: no cover - executable script helper
    main()
