"""CPR Knowledge Graph server for the v2 MCP contract."""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import Annotated, get_type_hints

from pydantic import Field

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

from fastmcp import FastMCP

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from mcp.contracts_v2 import (  # type: ignore
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from mcp.servers_v2.base import RunQueryMixin  # type: ignore
    from mcp.servers_v2 import cpr_tools as tools  # type: ignore
    from mcp.servers_v2.support_intent import SupportIntent  # type: ignore
else:  # pragma: no cover
    from ..contracts_v2 import (
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from ..servers_v2.base import RunQueryMixin
    from . import cpr_tools as tools
    from .support_intent import SupportIntent  # TODO: RE ADD .


DATASET_TITLE = "CPR Knowledge Graph (Brazilian policy corpus)"
DATASET_URL = "https://www.climatepolicyradar.org/"
DATASET_ID = "climate_policy_radar"

# Fixed: Added missing value for the constant
FACT_SNIPPET_MAX_CHARS = 500


def _load_dataset_citations() -> Dict[str, str]:
    path = Path(__file__).resolve().parents[2] / "static" / "meta" / "datasets.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    mapping: Dict[str, str] = {}
    for item in payload.get("items", []):
        dataset_id = item.get("id")
        citation = item.get("citation")
        if dataset_id and citation:
            mapping[str(dataset_id)] = str(citation)
    return mapping


DATASET_CITATIONS = _load_dataset_citations()


def _dataset_citation(dataset_id: str) -> Optional[str]:
    return DATASET_CITATIONS.get(dataset_id)


import logging
import sys
from typing import get_type_hints
from fastmcp import FastMCP

print("[cpr-server] Initializing CPRServerV2", file=sys.stderr, flush=True)

# Create the FastMCP instance at module level
mcp = FastMCP("cpr-server-v2")

class CPRServerV2(RunQueryMixin):
    """Expose CPR knowledge graph tools and run_query via FastMCP."""

    def __init__(self) -> None:
        # Configure logging once, at server init
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stderr,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        self.log = logging.getLogger("cpr-server")

        # Use the module-level mcp instance
        self.mcp = mcp
        self.log.info("Initializing CPRServerV2")

        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:
                self.log.warning("Anthropic client unavailable: %s", exc)

        # Register tools using proper method registration
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all tools with the MCP server."""
        # Instead of trying to register instance methods directly,
        # we create wrapper functions that call the instance methods
        
        # Create a reference to self for the closures
        server_instance = self
        
        @self.mcp.tool
        def describe_capabilities(
            format: Annotated[
                str,
                Field(description="Output format: 'json' or 'text'"),
            ] = "json",
        ) -> str:
            """Describe the CPR KG dataset, provenance, and key tools."""
            return server_instance.describe_capabilities(format)

        @self.mcp.tool
        def query_support(
            query: Annotated[
                str,
                Field(description="User question to evaluate for CPR KG suitability"),
            ],
            context: Annotated[
                Optional[Dict[str, Any]],
                Field(description="Optional orchestrator context"),
            ] = None,
        ) -> str:
            """Decide if the CPR knowledge graph should handle this query."""
            return server_instance.query_support(query, context)

        @self.mcp.tool
        def GetConcepts() -> str:
            """Get all available concepts in the knowledge graph."""
            return server_instance.GetConcepts()

        @self.mcp.tool
        def CheckConceptExists(
            concept: Annotated[
                str,
                Field(description="Preferred concept label to verify"),
            ],
        ) -> str:
            """Check if a concept exists in the knowledge graph."""
            return server_instance.CheckConceptExists(concept)

        @self.mcp.tool
        def GetSemanticallySimilarConcepts(
            concept: Annotated[str, Field(description="Seed concept label")],
        ) -> str:
            """Find semantically similar concepts."""
            return server_instance.GetSemanticallySimilarConcepts(concept)

        @self.mcp.tool
        def SearchConceptsByText(
            text: Annotated[str, Field(description="Substring to search for")],
            limit: Annotated[
                int,
                Field(description="Maximum results to return"),
            ] = 10,
        ) -> str:
            """Search concepts by text substring."""
            return server_instance.SearchConceptsByText(text, limit)

        @self.mcp.tool
        def SearchConceptsFuzzy(
            text: Annotated[str, Field(description="Query text allowing minor typos")],
            limit: Annotated[
                int,
                Field(description="Maximum results to return"),
            ] = 10,
        ) -> str:
            """Search concepts with fuzzy matching."""
            return server_instance.SearchConceptsFuzzy(text, limit)

        @self.mcp.tool
        def FindConceptMatchesByNgrams(
            text: Annotated[str, Field(description="Token or phrase to match")],
            top_k: Annotated[
                int,
                Field(description="Maximum matches to return"),
            ] = 5,
        ) -> str:
            """Find concept matches using n-gram analysis."""
            return server_instance.FindConceptMatchesByNgrams(text, top_k)

        @self.mcp.tool
        def GetTopConceptsByQuery(
            query: Annotated[str, Field(description="Natural language query")],
            top_k: Annotated[
                int,
                Field(description="Maximum concepts to return"),
            ] = 5,
        ) -> str:
            """Get top concepts matching a natural language query."""
            return server_instance.GetTopConceptsByQuery(query, top_k)

        @self.mcp.tool
        def GetTopConceptsByQueryLocal(
            query: Annotated[str, Field(description="Natural language query")],
            top_k: Annotated[
                int,
                Field(description="Maximum concepts to return"),
            ] = 5,
        ) -> str:
            """Get top concepts using local query processing."""
            return server_instance.GetTopConceptsByQueryLocal(query, top_k)

        @self.mcp.tool
        def GetAlternativeLabels(
            concept: Annotated[str, Field(description="Preferred concept label")],
        ) -> str:
            """Get alternative labels for a concept."""
            return server_instance.GetAlternativeLabels(concept)

        @self.mcp.tool
        def GetDescription(
            concept: Annotated[str, Field(description="Preferred concept label")],
        ) -> str:
            """Get description of a concept."""
            return server_instance.GetDescription(concept)

        @self.mcp.tool
        def GetRelatedConcepts(
            concept: Annotated[str, Field(description="Preferred concept label")],
        ) -> str:
            """Get concepts related to the given concept."""
            return server_instance.GetRelatedConcepts(concept)

        @self.mcp.tool
        def GetSubconcepts(
            concept: Annotated[str, Field(description="Preferred concept label")],
        ) -> str:
            """Get subconcepts of the given concept."""
            return server_instance.GetSubconcepts(concept)

        @self.mcp.tool
        def GetParentConcepts(
            concept: Annotated[str, Field(description="Preferred concept label")],
        ) -> str:
            """Get parent concepts of the given concept."""
            return server_instance.GetParentConcepts(concept)

        @self.mcp.tool
        def GetConceptGraphNeighbors(
            concept: Annotated[str, Field(description="Preferred concept label")],
            limit: Annotated[
                int,
                Field(description="Maximum neighbors to return"),
            ] = 15,
        ) -> str:
            """Get neighboring concepts in the knowledge graph."""
            return server_instance.GetConceptGraphNeighbors(concept, limit)

        @self.mcp.tool
        def FindConceptPathWithEdges(
            source_concept: Annotated[str, Field(description="Source concept label")],
            target_concept: Annotated[str, Field(description="Target concept label")],
            max_len: Annotated[
                int,
                Field(description="Maximum hops"),
            ] = 5,
        ) -> str:
            """Find paths between two concepts with edge information."""
            return server_instance.FindConceptPathWithEdges(source_concept, target_concept, max_len)

        @self.mcp.tool
        def FindConceptPathRich(
            source_concept: Annotated[str, Field(description="Source concept label")],
            target_concept: Annotated[str, Field(description="Target concept label")],
        ) -> str:
            """Find rich paths between two concepts."""
            return server_instance.FindConceptPathRich(source_concept, target_concept)

        @self.mcp.tool
        def ExplainConceptRelationship(
            source_concept: Annotated[str, Field(description="Source concept label")],
            target_concept: Annotated[str, Field(description="Target concept label")],
        ) -> str:
            """Explain the relationship between two concepts."""
            return server_instance.ExplainConceptRelationship(source_concept, target_concept)

        @self.mcp.tool
        def GetPassagesMentioningConcept(
            concept: Annotated[str, Field(description="Concept label to search")],
            limit: Annotated[
                int,
                Field(description="Maximum passages to return"),
            ] = 5,
        ) -> str:
            """Get passages that mention a specific concept."""
            return server_instance.GetPassagesMentioningConcept(concept, limit)

        @self.mcp.tool
        def PassagesMentioningBothConcepts(
            concept_a: Annotated[str, Field(description="First concept label")],
            concept_b: Annotated[str, Field(description="Second concept label")],
            limit: Annotated[
                int,
                Field(description="Maximum passages to return"),
            ] = 5,
        ) -> str:
            """Get passages mentioning both concepts."""
            return server_instance.PassagesMentioningBothConcepts(concept_a, concept_b, limit)

        @self.mcp.tool
        def GetKGDatasetMetadata() -> str:
            """Get metadata about the knowledge graph dataset."""
            return server_instance.GetKGDatasetMetadata()

        @self.mcp.tool
        def GetAvailableDatasets() -> str:
            """Get list of available datasets."""
            return server_instance.GetAvailableDatasets()

        @self.mcp.tool
        def GetDatasetContent(
            dataset_id: Annotated[str, Field(description="Dataset identifier")],
        ) -> str:
            """Get content of a specific dataset."""
            return server_instance.GetDatasetContent(dataset_id)

        @self.mcp.tool
        def DescribeServer() -> str:
            """Get server description and status."""
            return server_instance.DescribeServer()

        @self.mcp.tool
        def DebugEmbeddingStatus() -> str:
            """Get embedding system debug information."""
            return server_instance.DebugEmbeddingStatus()

        @self.mcp.tool
        def GetSemanticDebugLog(
            limit: Annotated[
                int,
                Field(description="Maximum log entries to return"),
            ] = 50,
        ) -> str:
            """Get semantic processing debug log."""
            return server_instance.GetSemanticDebugLog(limit)

        @self.mcp.tool
        def run_query(
            query: Annotated[str, Field(description="User question")],
            context: Annotated[
                Optional[Dict[str, Any]],
                Field(description="Optional orchestrator context"),
            ] = None,
        ) -> str:
            """Execute the primary run_query workflow."""
            return server_instance.run_query(query, context)

    # ------------------------------------------------------------------ helpers
    def _capabilities_metadata(self) -> Dict[str, Any]:
        return {
            "name": "cpr",
            "description": "Brazil-focused policy knowledge graph with concepts, relationships, and annotated passages.",
            "version": "2.0.0",
            "tags": ["policy", "knowledge graph", "passages"],
            "dataset": DATASET_TITLE,
            "url": DATASET_URL,
            "tools": [
                "describe_capabilities",
                "query_support",
                "GetConcepts",
                "CheckConceptExists",
                "GetSemanticallySimilarConcepts",
                "SearchConceptsByText",
                "SearchConceptsFuzzy",
                "FindConceptMatchesByNgrams",
                "GetTopConceptsByQuery",
                "GetDocumentNeighbors",
                "GetConceptNeighbors",
                "GetDocumentPassages",
                "GetDocumentMetadata",
                "DebugConcept",
                "run_query",
            ],
        }

    def _capability_summary(self) -> str:
        metadata = self._capabilities_metadata()
        return (
            f"Dataset: {metadata['dataset']} ({metadata['description']}). "
            "Covers Brazilian climate policy documents, concept nodes, relationships, and annotated evidence passages."
        )

    def _classify_support(self, query: str) -> SupportIntent:
        if not self._anthropic_client:
            return SupportIntent(
                supported=True,
                score=0.3,
                reasons=["LLM unavailable; defaulting to dataset summary"],
            )

        prompt = (
            "Decide whether the CPR climate policy knowledge graph should be used to answer the question."
            " The graph spans Brazilian laws, regulations, strategies, and related passages."
            " Reply with JSON {\"supported\": true|false, \"reason\": \"short explanation\"}.\n"
            f"Dataset capabilities: {self._capability_summary()}\n"
            f"Question: {query}"
        )

        try:
            response = self._anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=128,
                temperature=0,
                system="Respond with valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        except Exception as exc:  # pragma: no cover - network failures
            return SupportIntent(
                supported=True,
                score=0.3,
                reasons=[f"LLM intent unavailable: {exc}"],
            )

        def _parse(blob: str) -> Optional[Dict[str, Any]]:
            try:
                parsed = json.loads(blob)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None

        data = _parse(text)
        if not data:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                data = _parse(text[start : end + 1])

        if not data:
            return SupportIntent(
                supported=True,
                score=0.3,
                reasons=["LLM returned non-JSON response"],
            )

        supported = bool(data.get("supported", False))
        reason = str(data.get("reason")) if data.get("reason") else None
        score = 0.9 if supported else 0.1

        reasons = [reason] if reason else []
        if not reasons:
            reasons.append("LLM classification")

        return SupportIntent(supported=supported, score=score, reasons=reasons)

    @staticmethod
    def _as_str(value: Any) -> str:
        if value is None:
            return ""
        try:
            if isinstance(value, (int, float)):
                return str(value)
            if hasattr(value, "item"):
                try:
                    return str(value.item())
                except Exception:
                    pass
            return str(value)
        except Exception:
            return ""

    def _candidate_concepts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        ranked: Dict[str, Dict[str, Any]] = {}
        order_counter = 0

        def _register(items: Optional[List[Dict[str, Any]]], priority: int) -> None:
            nonlocal order_counter
            if not items:
                return
            for item in items:
                label_raw = item.get("label")
                label = str(label_raw or "").strip()
                if not label:
                    continue
                key = label.lower()
                candidate = {
                    "label": label,
                    "wikibase_id": self._as_str(item.get("wikibase_id")) or None,
                }
                existing = ranked.get(key)
                if existing:
                    if priority < existing["priority"]:
                        existing.update({"priority": priority, "candidate": candidate})
                    continue
                ranked[key] = {
                    "priority": priority,
                    "order": order_counter,
                    "candidate": candidate,
                }
                order_counter += 1

        # Direct heuristics on the full query first.
        _register(tools.search_concepts_by_text(cleaned_query, limit=limit * 2), priority=1)
        _register(tools.search_concepts_fuzzy(cleaned_query, limit=limit * 2), priority=1)
        try:
            _register(tools.get_top_concepts_by_query(cleaned_query, top_k=limit * 2), priority=2)
        except Exception:
            pass
        _register(tools.find_concept_matches_by_ngrams(cleaned_query, top_k=limit * 2), priority=3)
        _register(tools.get_top_concepts_by_query_local(cleaned_query, top_k=limit * 2), priority=3)

        # Break the query into candidate keywords and short phrases to broaden matches.
        token_pattern = re.compile(r"[\w'-]+", re.UNICODE)
        raw_tokens = [tok.strip("-'_") for tok in token_pattern.findall(cleaned_query)]
        tokens = [tok for tok in raw_tokens if tok and len(tok) >= 4]
        stopwords = {
            "about",
            "across",
            "after",
            "between",
            "brazil",
            "brazilian",
            "climate",
            "including",
            "policy",
            "policies",
            "regarding",
            "toward",
            "towards",
            "using",
            "which",
            "where",
            "would",
        }

        keywords: List[str] = []
        seen_kw: set[str] = set()
        for token in tokens:
            token_lower = token.lower()
            if token_lower in stopwords or token_lower in seen_kw:
                continue
            seen_kw.add(token_lower)
            keywords.append(token)
            if len(keywords) >= 12:
                break

        # Also capture a handful of two/three-word phrases for additional context.
        phrases: List[str] = []
        for window in (3, 2):
            if len(tokens) < window:
                continue
            for idx in range(len(tokens) - window + 1):
                phrase = " ".join(tokens[idx : idx + window])
                phrase_lower = phrase.lower()
                if phrase_lower in stopwords or phrase_lower in seen_kw:
                    continue
                seen_kw.add(phrase_lower)
                phrases.append(phrase)
                if len(phrases) >= 8:
                    break
            if len(phrases) >= 8:
                break

        def _search_with_phrase(phrase: str, *, priority: int) -> None:
            _register(tools.search_concepts_by_text(phrase, limit=4), priority=priority)
            _register(tools.search_concepts_fuzzy(phrase, limit=4), priority=priority)
            _register(tools.find_concept_matches_by_ngrams(phrase, top_k=4), priority=priority + 1)
            try:
                _register(tools.get_top_concepts_by_query(phrase, top_k=3), priority=priority + 1)
            except Exception:
                pass

        for keyword in keywords:
            _search_with_phrase(keyword, priority=0)

        for phrase in phrases:
            _search_with_phrase(phrase, priority=1)

        if not ranked:
            return []

        ordered = sorted(
            ranked.values(), key=lambda payload: (payload["priority"], payload["order"])
        )
        return [entry["candidate"] for entry in ordered[:limit]]

    @staticmethod
    def _fact_snippet(text: str) -> str:
        """Return a longer snippet for fact rendering without overwhelming output."""

        text = (text or "").strip()
        if len(text) <= FACT_SNIPPET_MAX_CHARS:
            return text

        candidate = text[:FACT_SNIPPET_MAX_CHARS]
        cutoff = candidate.rfind(" ")
        if cutoff == -1:
            cutoff = FACT_SNIPPET_MAX_CHARS
        return candidate[:cutoff].rstrip() + "…"

    def _passage_citation(
        self,
        concept_label: str,
        passage: Dict[str, Any],
    ) -> CitationPayload:
        metadata = passage.get("metadata", {}) or {}
        passage_id = str(metadata.get("passage_id") or passage.get("id") or "unknown")
        document_id = str(metadata.get("document_id") or "unknown")
        citation_id = f"passage_{document_id}_{passage_id}"
        description = f"Passage {passage_id} in document {document_id} mentioning {concept_label}."
        return CitationPayload(
            id=citation_id,
            server="cpr",
            tool="run_query",
            title=f"CPR passage {document_id}",
            source_type="Policy Document",
            description=_dataset_citation(DATASET_ID) or description,
            metadata={
                "document_id": document_id,
                "passage_id": passage_id,
                "concept": concept_label,
            },
        )

    def _build_kg(
        self,
        concepts: List[Dict[str, Any]],
    ) -> KnowledgeGraphPayload:
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        nodes["cpr_dataset"] = {
            "id": "cpr_dataset",
            "type": "dataset",
            "label": DATASET_TITLE,
            "url": DATASET_URL,
        }

        for concept in concepts:
            label = concept.get("label")
            if not label:
                continue
            cid = self._as_str(concept.get("wikibase_id") or label) or label
            nodes[cid] = {
                "id": cid,
                "type": "concept",
                "label": label,
            }
            edges.append({"source": "cpr_dataset", "target": cid, "type": "dataset_contains"})

            neighbors = tools.get_concept_graph_neighbors(label, limit=10)
            for neighbor in neighbors:
                target_label = neighbor.get("target_label") or neighbor.get("label")
                if not target_label:
                    continue
                target_id = self._as_str(neighbor.get("target_id")) or target_label
                if target_id not in nodes:
                    nodes[target_id] = {
                        "id": target_id,
                        "type": "concept",
                        "label": target_label,
                    }
                edges.append(
                    {
                        "source": cid,
                        "target": target_id,
                        "type": neighbor.get("edge_type", "related"),
                    }
                )

        return KnowledgeGraphPayload(nodes=list(nodes.values()), edges=edges)

    # ------------------------------------------------------------------ tools
    def describe_capabilities(
        self,
        format: Annotated[
            str,
            Field(description="Output format: 'json' or 'text'"),
        ] = "json",
    ) -> str:
        """Describe the CPR KG dataset, provenance, and key tools."""

        payload = self._capabilities_metadata()
        return json.dumps(payload, ensure_ascii=False) if format.lower() == "json" else str(payload)

    def query_support(
        self,
        query: Annotated[
            str,
            Field(description="User question to evaluate for CPR KG suitability"),
        ],
        context: Annotated[
            Optional[Dict[str, Any]],
            Field(description="Optional orchestrator context"),
        ] = None,
    ) -> str:
        """Decide if the CPR knowledge graph should handle this query."""

        intent = self._classify_support(query)
        payload = {
            "server": "cpr",
            "query": query,
            "supported": intent.supported,
            "score": intent.score,
            "reasons": intent.reasons,
        }
        return json.dumps(payload, ensure_ascii=False)

    def GetConcepts(self) -> str:
        concepts = tools.get_concepts()
        return json.dumps({"concepts": concepts}, ensure_ascii=False)

    def CheckConceptExists(
        self,
        concept: Annotated[
            str,
            Field(description="Preferred concept label to verify"),
        ],
    ) -> str:
        exists = tools.check_concept_exists(concept)
        return json.dumps({"concept": concept, "exists": exists}, ensure_ascii=False)

    def GetSemanticallySimilarConcepts(
        self,
        concept: Annotated[str, Field(description="Seed concept label")],
    ) -> str:
        similar = tools.get_semantically_similar_concepts(concept)
        return json.dumps({"concept": concept, "similar_concepts": similar}, ensure_ascii=False)

    def SearchConceptsByText(
        self,
        text: Annotated[str, Field(description="Substring to search for")],
        limit: Annotated[
            int,
            Field(description="Maximum results to return"),
        ] = 10,
    ) -> str:
        results = tools.search_concepts_by_text(text, limit)
        return json.dumps({"query": text, "limit": limit, "results": results}, ensure_ascii=False)

    def SearchConceptsFuzzy(
        self,
        text: Annotated[str, Field(description="Query text allowing minor typos")],
        limit: Annotated[
            int,
            Field(description="Maximum results to return"),
        ] = 10,
    ) -> str:
        results = tools.search_concepts_fuzzy(text, limit)
        return json.dumps({"query": text, "limit": limit, "results": results}, ensure_ascii=False)

    def FindConceptMatchesByNgrams(
        self,
        text: Annotated[str, Field(description="Token or phrase to match")],
        top_k: Annotated[
            int,
            Field(description="Maximum matches to return"),
        ] = 5,
    ) -> str:
        results = tools.find_concept_matches_by_ngrams(text, top_k)
        return json.dumps({"query": text, "limit": top_k, "results": results}, ensure_ascii=False)

    def GetTopConceptsByQuery(
        self,
        query: Annotated[str, Field(description="Natural language query")],
        top_k: Annotated[
            int,
            Field(description="Maximum concepts to return"),
        ] = 5,
    ) -> str:
        results = tools.get_top_concepts_by_query(query, top_k)
        return json.dumps({"query": query, "limit": top_k, "results": results}, ensure_ascii=False)

    def GetTopConceptsByQueryLocal(
        self,
        query: Annotated[str, Field(description="Natural language query")],
        top_k: Annotated[
            int,
            Field(description="Maximum concepts to return"),
        ] = 5,
    ) -> str:
        results = tools.get_top_concepts_by_query_local(query, top_k)
        return json.dumps({"query": query, "limit": top_k, "results": results}, ensure_ascii=False)

    def GetAlternativeLabels(
        self,
        concept: Annotated[str, Field(description="Preferred concept label")],
    ) -> str:
        labels = tools.get_alternative_labels(concept)
        return json.dumps({"concept": concept, "labels": labels}, ensure_ascii=False)

    def GetDescription(
        self,
        concept: Annotated[str, Field(description="Preferred concept label")],
    ) -> str:
        description = tools.get_description(concept)
        return json.dumps({"concept": concept, "description": description}, ensure_ascii=False)

    def GetRelatedConcepts(
        self,
        concept: Annotated[str, Field(description="Preferred concept label")],
    ) -> str:
        related = tools.get_related_concepts(concept)
        return json.dumps({"concept": concept, "related": related}, ensure_ascii=False)

    def GetSubconcepts(
        self,
        concept: Annotated[str, Field(description="Preferred concept label")],
    ) -> str:
        subconcepts = tools.get_subconcepts(concept)
        return json.dumps({"concept": concept, "subconcepts": subconcepts}, ensure_ascii=False)

    def GetParentConcepts(
        self,
        concept: Annotated[str, Field(description="Preferred concept label")],
    ) -> str:
        parents = tools.get_parent_concepts(concept)
        return json.dumps({"concept": concept, "parents": parents}, ensure_ascii=False)

    def GetConceptGraphNeighbors(
        self,
        concept: Annotated[str, Field(description="Preferred concept label")],
        limit: Annotated[
            int,
            Field(description="Maximum neighbors to return"),
        ] = 15,
    ) -> str:
        neighbors = tools.get_concept_graph_neighbors(concept, limit)
        return json.dumps({"concept": concept, "limit": limit, "neighbors": neighbors}, ensure_ascii=False)

    def FindConceptPathWithEdges(
        self,
        source_concept: Annotated[str, Field(description="Source concept label")],
        target_concept: Annotated[str, Field(description="Target concept label")],
        max_len: Annotated[
            int,
            Field(description="Maximum hops"),
        ] = 5,
    ) -> str:
        paths = tools.find_concept_path_with_edges(source_concept, target_concept, max_len)
        return json.dumps(
            {
                "source_concept": source_concept,
                "target_concept": target_concept,
                "max_len": max_len,
                "paths": paths,
            },
            ensure_ascii=False,
        )

    def FindConceptPathRich(
        self,
        source_concept: Annotated[str, Field(description="Source concept label")],
        target_concept: Annotated[str, Field(description="Target concept label")],
    ) -> str:
        path = tools.find_concept_path_rich(source_concept, target_concept)
        return json.dumps(
            {
                "source_concept": source_concept,
                "target_concept": target_concept,
                "path": path,
            },
            ensure_ascii=False,
        )

    def ExplainConceptRelationship(
        self,
        source_concept: Annotated[str, Field(description="Source concept label")],
        target_concept: Annotated[str, Field(description="Target concept label")],
    ) -> str:
        explanation = tools.explain_concept_relationship(source_concept, target_concept)
        return json.dumps(
            {
                "source_concept": source_concept,
                "target_concept": target_concept,
                "explanation": explanation,
            },
            ensure_ascii=False,
        )

    def GetPassagesMentioningConcept(
        self,
        concept: Annotated[str, Field(description="Concept label to search")],
        limit: Annotated[
            int,
            Field(description="Maximum passages to return"),
        ] = 5,
    ) -> str:
        passages = tools.get_passages_mentioning_concept(concept, limit)
        return json.dumps({"concept": concept, "limit": limit, "passages": passages}, ensure_ascii=False)

    def PassagesMentioningBothConcepts(
        self,
        concept_a: Annotated[str, Field(description="First concept label")],
        concept_b: Annotated[str, Field(description="Second concept label")],
        limit: Annotated[
            int,
            Field(description="Maximum passages to return"),
        ] = 5,
    ) -> str:
        passages = tools.passages_mentioning_both_concepts(concept_a, concept_b, limit)
        return json.dumps(
            {
                "concept_a": concept_a,
                "concept_b": concept_b,
                "limit": limit,
                "passages": passages,
            },
            ensure_ascii=False,
        )

    def GetKGDatasetMetadata(self) -> str:
        metadata = tools.get_dataset_metadata()
        return json.dumps({"metadata": metadata}, ensure_ascii=False)

    def GetAvailableDatasets(self) -> str:
        datasets = tools.get_available_datasets()
        return json.dumps({"datasets": datasets}, ensure_ascii=False)

    def GetDatasetContent(
        self,
        dataset_id: Annotated[str, Field(description="Dataset identifier")],
    ) -> str:
        content = tools.get_dataset_content(dataset_id)
        return json.dumps({"dataset_id": dataset_id, "content": content}, ensure_ascii=False)

    def DescribeServer(self) -> str:
        payload = tools.describe_server()
        return json.dumps(payload, ensure_ascii=False)

    def DebugEmbeddingStatus(self) -> str:
        payload = tools.debug_embedding_status()
        return json.dumps(payload, ensure_ascii=False)

    def GetSemanticDebugLog(
        self,
        limit: Annotated[
            int,
            Field(description="Maximum log entries to return"),
        ] = 50,
    ) -> str:
        payload = tools.get_semantic_debug_log(limit)
        return json.dumps({"limit": limit, "log": payload}, ensure_ascii=False)

    def run_query(
        self,
        query: Annotated[str, Field(description="User question")],
        context: Annotated[
            Optional[Dict[str, Any]],
            Field(description="Optional orchestrator context"),
        ] = None,
    ) -> str:
        """Execute the primary run_query workflow."""

        response = self.handle_run_query(query=query, context=context or {})
        return response.model_dump_json()

    # ------------------------------------------------------------------ run_query
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        start = time.perf_counter()
        concepts = self._candidate_concepts(query, limit=5)

        citations: Dict[str, CitationPayload] = {}
        facts: List[FactPayload] = []
        table_rows: List[List[Any]] = []
        processed_labels: set[str] = set()

        def _consume(batch: List[Dict[str, Any]]) -> None:
            for concept in batch:
                label = concept.get("label")
                if not label or label in processed_labels:
                    continue
                processed_labels.add(label)
                passages = tools.get_passages_mentioning_concept(label, limit=3)
                if not passages:
                    continue
                for passage in passages:
                    citation = self._passage_citation(label, passage)
                    citations[citation.id] = citation

                    passage_text = str(passage.get("text", ""))
                    metadata = passage.get("metadata", {}) or {}
                    document_id = self._as_str(metadata.get("document_id") or "unknown") or "unknown"
                    passage_id = self._as_str(metadata.get("passage_id") or "unknown") or "unknown"

                    print(
                        "[cpr] context snippet:",
                        f"concept={label!r}",
                        f"document={document_id}",
                        f"passage={passage_id}",
                        passage_text,
                        flush=True,
                    )

                    snippet = self._fact_snippet(passage_text)
                    fact_id = f"{citation.id}_fact"
                    facts.append(
                        FactPayload(
                            id=fact_id,
                            text=f"Document {document_id} references {label}: {snippet}",
                            citation_id=citation.id,
                            metadata={
                                "concept": label,
                                "document_id": document_id,
                                "passage_id": passage_id,
                            },
                        )
                    )
                    table_rows.append([
                        label,
                        document_id,
                        snippet,
                    ])

        _consume(concepts)

        if not facts:
            expanded = self._candidate_concepts(query, limit=12)
            _consume(expanded)
            concepts = expanded

        messages: List[MessagePayload] = []
        if not facts:
            messages.append(
                MessagePayload(
                    level="warning",
                    text="No CPR passages matched the query using concept discovery heuristics.",
                )
            )

        kg_payload = self._build_kg(concepts)

        artifacts: List[ArtifactPayload] = []

        duration_ms = int((time.perf_counter() - start) * 1000)

        next_actions = []
        if concepts:
            next_actions.append("Use cpr:GetPassagesMentioningConcept for full context")
            next_actions.append("Call cpr:GetConceptGraphNeighbors on top concepts for deeper KG analysis")

        return RunQueryResponse(
            server="cpr",
            query=query,
            facts=facts,
            citations=list(citations.values()),
            artifacts=artifacts,
            messages=messages,
            kg=kg_payload,
            next_actions=next_actions,
            duration_ms=duration_ms,
        )


def create_server() -> FastMCP:
    """Entry point used by ``python -m mcp.servers_v2.cpr_server_v2``."""

    server = CPRServerV2()
    return server.mcp


if __name__ == "__main__":  # pragma: no cover - manual execution
    create_server().run()
