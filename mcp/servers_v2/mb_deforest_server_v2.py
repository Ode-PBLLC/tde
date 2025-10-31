"""MapBiomas Deforestation MCP server (v2 contract).

This server reimplements the legacy MapBiomas RAG endpoints using the v2
`run_query` schema so orchestrators receive structured facts, citations, and
artifacts while retaining the original tool surface.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastmcp import FastMCP

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

try:  # Optional dotenv support to mirror legacy behaviour
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - best effort only
    pass

import chromadb
from chromadb.config import Settings
from openai import OpenAI

if __package__ in {None, ""}:  # pragma: no cover - direct execution helper
    import sys

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
    from mcp.servers_v2.support_intent import SupportIntent  # type: ignore
else:
    from ..contracts_v2 import (
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from ..servers_v2.base import RunQueryMixin
    from ..support_intent import SupportIntent


INDEX_DIR = Path("data/mb-deforest")
COLLECTION_NAME = "mb-deforest"
TOP_K = 6

DATASET_NAME = "MapBiomas Annual Deforestation Report (RAD) 2024"
DATASET_URL = "https://mapbiomas.org/en/deforestation"

DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


@dataclass
class RetrievalSnippet:
    doc_id: str
    file: Optional[str]
    page: Optional[int]
    text: str
    preview: str
    similarity: float
    chunk_id: str


class MBDeforestServerV2(RunQueryMixin):
    """FastMCP server exposing MapBiomas deforestation tools."""

    def __init__(self) -> None:
        self.mcp = FastMCP("mb-deforest-server-v2")

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("Set OPENAI_API_KEY before launching mb_deforest_server_v2")
        self.openai = OpenAI(api_key=openai_key)

        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential issues
                print(f"[mb-deforest-server] Warning: Anthropic client unavailable: {exc}")

        self._register_capabilities_tool()
        self._register_query_support_tool()
        self._register_search_tool()
        self._register_mb_report_ask_tool()
        self._register_ipcc_report_ask_tool()
        self._register_run_query_tool()

    # ------------------------------------------------------------------ helpers
    def _get_collection(self) -> Any:
        if not INDEX_DIR.exists():
            raise RuntimeError("MapBiomas index missing. Run rag_embed_generator.py first.")
        client = chromadb.Client(Settings(persist_directory=str(INDEX_DIR), is_persistent=True))
        return client.get_or_create_collection(COLLECTION_NAME)

    def _capabilities_metadata(self) -> Dict[str, Any]:
        return {
            "name": "mb_deforest",
            "description": "MapBiomas Annual Deforestation (RAD) report passages with retrieval and QA tools.",
            "version": "2.0.0",
            "tags": ["mapbiomas", "deforestation", "amazon", "land-use"],
            "dataset": DATASET_NAME,
            "url": DATASET_URL,
            "tools": [
                "describe_capabilities",
                "query_support",
                "MBReportSearch",
                "MBReportAsk",
                "IPCCReportAsk",
                "run_query",
            ],
        }

    def _capability_summary(self) -> str:
        metadata = self._capabilities_metadata()
        return (
            f"Dataset: {metadata['dataset']} ({metadata['description']}). "
            "Offers retrieval and Q&A over RAD report passages, plus IPCC context for deforestation analysis."
        )

    def _classify_support(self, query: str) -> SupportIntent:
        if self._anthropic_client:
            try:
                prompt = (
                    "Determine whether the MapBiomas Annual Deforestation (RAD) dataset should handle the question."
                    " Respond strictly with JSON containing keys 'supported' (true/false) and 'reason' (short explanation).\n"
                    f"Dataset capabilities: {self._capability_summary()}\n"
                    f"Question: {query}"
                )
                response = self._anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=128,
                    temperature=0,
                    system="Respond with valid JSON only.",
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                intent = self._parse_support_intent(text)
                if intent:
                    return intent
            except Exception as exc:  # pragma: no cover
                return SupportIntent(
                    supported=True,
                    score=0.3,
                    reasons=[f"Anthropic intent unavailable: {exc}"],
                )

        try:
            prompt = (
                "Determine whether the MapBiomas Annual Deforestation (RAD) dataset should handle the question."
                " Respond strictly with JSON containing keys 'supported' (true/false) and 'reason' (short explanation).\n"
                f"Dataset capabilities: {self._capability_summary()}\n"
                f"Question: {query}"
            )
            completion = self.openai.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Respond with JSON containing keys supported (true/false) and reason (string).",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=80,
                temperature=0,
            )
            text = completion.choices[0].message.content.strip()
            intent = self._parse_support_intent(text)
            if intent:
                return intent
        except Exception as exc:  # pragma: no cover
            return SupportIntent(
                supported=True,
                score=0.3,
                reasons=[f"OpenAI intent unavailable: {exc}"],
            )

        return SupportIntent(
            supported=True,
            score=0.3,
            reasons=["LLM returned non-JSON response"],
        )

    @staticmethod
    def _parse_support_intent(text: str) -> Optional[SupportIntent]:
        def _parse(blob: str) -> Optional[Dict[str, Any]]:
            try:
                payload = json.loads(blob)
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return None

        data = _parse(text)
        if not data:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                data = _parse(text[start : end + 1])
        if not data:
            return None

        supported = bool(data.get("supported", False))
        reason = str(data.get("reason")) if data.get("reason") else None
        score = 0.9 if supported else 0.1
        reasons = [reason] if reason else ["LLM classification"]
        return SupportIntent(supported=supported, score=score, reasons=reasons)


    def _embed_query(self, text: str) -> List[float]:
        embedding = self.openai.embeddings.create(model="text-embedding-3-small", input=[text])
        return embedding.data[0].embedding

    def _search(self, query: str, k: int) -> List[RetrievalSnippet]:
        collection = self._get_collection()
        query_vector = self._embed_query(query)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        snippets: List[RetrievalSnippet] = []
        for idx, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            similarity = 1.0 - float(distance) if distance is not None else 0.0
            file_path: Optional[str] = metadata.get("file") if isinstance(metadata, dict) else None
            doc_id = Path(file_path).stem if file_path else metadata.get("doc_id") if isinstance(metadata, dict) else f"doc_{idx}"
            snippets.append(
                RetrievalSnippet(
                    doc_id=doc_id or f"doc_{idx}",
                    file=file_path,
                    page=metadata.get("page") if isinstance(metadata, dict) else None,
                    text=doc,
                    preview=(doc or "")[:240],
                    similarity=round(similarity, 4),
                    chunk_id=f"{doc_id or 'doc'}::chunk_{idx}",
                )
            )
        return snippets

    def _format_snippet_for_prompt(self, snippet: RetrievalSnippet) -> str:
        file_name = Path(snippet.file).name if snippet.file else snippet.doc_id
        page = f" p.{snippet.page}" if snippet.page is not None else ""
        return f"[{file_name}{page}] {snippet.text}"

    def _call_chat_model(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        response = self.openai.chat.completions.create(
            model=DEFAULT_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _split_answer_into_facts(self, answer: str, citation_id: str) -> List[FactPayload]:
        cleaned = answer.strip()
        if not cleaned:
            return []

        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", cleaned) if segment.strip()]
        facts: List[FactPayload] = []
        for idx, sentence in enumerate(sentences[:4]):
            facts.append(
                FactPayload(
                    id=f"mb_fact_{idx+1}",
                    text=sentence,
                    citation_id=citation_id,
                )
            )
        if not facts:
            facts.append(
                FactPayload(id="mb_fact", text=cleaned, citation_id=citation_id)
            )
        return facts

    def _build_snippet_citations(self, snippets: Iterable[RetrievalSnippet]) -> List[CitationPayload]:
        citations: List[CitationPayload] = []
        for snippet in snippets:
            file_name = Path(snippet.file).name if snippet.file else snippet.doc_id
            description_parts = [f"Similarity {snippet.similarity}"]
            if snippet.page is not None:
                description_parts.append(f"page {snippet.page}")
            citations.append(
                CitationPayload(
                    id=snippet.chunk_id,
                    server="mb_deforest",
                    tool="MBReportSearch",
                    title=file_name,
                    source_type="Report",
                    description=", ".join(description_parts),
                    url=None,
                    metadata={
                        "doc_id": snippet.doc_id,
                        "file": snippet.file,
                        "page": snippet.page,
                        "similarity": snippet.similarity,
                    },
                )
            )
        return citations

    # ------------------------------------------------------------------ tools
    def _register_capabilities_tool(self) -> None:
        @self.mcp.tool()
        def describe_capabilities(format: str = "json") -> str:  # type: ignore[misc]
            """Describe the MapBiomas dataset and exposed tools.

            Example:
                >>> describe_capabilities()
            """

            payload = self._capabilities_metadata()
            return json.dumps(payload) if format == "json" else str(payload)

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> str:  # type: ignore[misc]
            """Estimate whether the MapBiomas dataset can answer the query.

            Example:
                >>> query_support("MapBiomas deforestation trends in Pará", {})
            """

            intent = self._classify_support(query)
            payload = {
                "server": "mb_deforest",
                "query": query,
                "supported": intent.supported,
                "score": intent.score,
                "reasons": intent.reasons,
            }
            return json.dumps(payload)

    def _register_search_tool(self) -> None:
        @self.mcp.tool()
        def MBReportSearch(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:  # noqa: N802 - legacy casing
            """
            Vector search over the MapBiomas index. Returns top-k chunks with file/page/similarity.

            Example:
                >>> MBReportSearch("Amazon biome enforcement", k=3)
            """

            snippets = self._search(query, k)
            return [snippet.__dict__ for snippet in snippets]

    def _register_mb_report_ask_tool(self) -> None:
        @self.mcp.tool()
        def MBReportAsk(query: str, k: int = TOP_K, max_tokens: int = 800) -> Dict[str, Any]:  # noqa: N802 - legacy casing
            """
            Gets context from the MapBiomas Deforestation report focused,
            using retrieved snippets. Returns model answer + citations and snippets.

            Example:
                >>> MBReportAsk("Summarize RAD findings for Pará", k=4)
            """

            snippets = self._search(query, k)
            if not snippets:
                return {"answer": "No relevant snippets matched the MapBiomas RAD index.", "passages": []}

            context = "\n\n".join(self._format_snippet_for_prompt(snippet) for snippet in snippets)
            system_prompt = (
                "You are a MapBiomas RAD (Annual Deforestation Report) expert. Provide concise, evidence-based, "
                "and policy-relevant answers grounded strictly in the provided snippets. Cite inline as (filename p.page). "
                "Highlight biome-specific and governance implications and note confidence or uncertainty."
            )
            user_prompt = f"Question: {query}\n\nSnippets:\n{context}"

            try:
                answer = self._call_chat_model(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
            except Exception as exc:  # pragma: no cover - network dependent
                return {"answer": f"LLM error: {exc}", "passages": [snippet.__dict__ for snippet in snippets]}

            return {
                "answer": answer,
                "passages": [snippet.__dict__ for snippet in snippets],
                "metadata": {"model": DEFAULT_CHAT_MODEL, "top_k": k},
            }

    def _register_ipcc_report_ask_tool(self) -> None:
        @self.mcp.tool()
        def IPCCReportAsk(query: str, k: int = TOP_K, max_tokens: int = 800) -> Dict[str, Any]:  # noqa: N802 - legacy casing
            """
            Gets context from the IPCC reports focused on state of climate globally and in Latin America and the Caribbean,
            using retrieved snippets. Returns model answer + citations and snippets.

            Example:
                >>> IPCCReportAsk("extreme events LAC", k=4)
            """

            snippets = self._search(query, k)
            if not snippets:
                return {"answer": "No relevant snippets matched the MapBiomas RAD index.", "passages": []}

            context = "\n\n".join(self._format_snippet_for_prompt(snippet) for snippet in snippets)
            system_prompt = (
                "You are summarizing findings from WMO and IPCC reports relevant to Latin America. Provide precise, evidence-based responses "
                "grounded in the snippets, cite inline, and call out uncertainty or confidence language."
            )
            user_prompt = f"Question: {query}\n\nSnippets:\n{context}"

            try:
                answer = self._call_chat_model(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
            except Exception as exc:  # pragma: no cover - network dependent
                return {"answer": f"LLM error: {exc}", "passages": [snippet.__dict__ for snippet in snippets]}

            return {
                "answer": answer,
                "passages": [snippet.__dict__ for snippet in snippets],
                "metadata": {"model": DEFAULT_CHAT_MODEL, "top_k": k},
            }

    # ------------------------------------------------------------------ run_query
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        snippets = self._search(query, TOP_K)

        if not snippets:
            notice_citation = CitationPayload(
                id="mb_no_results",
                server="mb_deforest",
                tool="run_query",
                title="No evidence available",
                source_type="Notice",
                description="Search returned no passages for the current query.",
                url=None,
            )
            notice_fact = FactPayload(
                id="mb_no_results_fact",
                text="No relevant passages matched the query; adjust the geography, timeframe, or keywords and retry.",
                citation_id=notice_citation.id,
            )
            return RunQueryResponse(
                server="mb_deforest",
                query=query,
                facts=[notice_fact],
                citations=[notice_citation],
                artifacts=[],
                messages=[
                    MessagePayload(level="warning", text="No MapBiomas passages matched the query."),
                ],
                next_actions=["Run MBReportSearch with alternative wording"],
            )

        dataset_citation = CitationPayload(
            id="mb_rad_2024",
            server="mb_deforest",
            tool="run_query",
            title=DATASET_NAME,
            source_type="Report",
            description="MapBiomas Annual Deforestation (RAD) 2024 dataset",  # noqa: E501
            url=DATASET_URL,
        )

        context_block = "\n\n".join(self._format_snippet_for_prompt(snippet) for snippet in snippets)
        system_prompt = (
            "You are drafting a policy-ready summary grounded in MapBiomas RAD deforestation findings. "
            "Use only the snippets provided. Highlight quantitative results, biome or state differences, and governance signals."
        )
        user_prompt = f"Question: {query}\n\nSnippets:\n{context_block}"

        try:
            answer = self._call_chat_model(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:  # pragma: no cover - network dependent
            messages = [MessagePayload(level="error", text=f"LLM error while generating answer: {exc}")]
            snippet_citations = self._build_snippet_citations(snippets)
            fallback_facts: List[FactPayload] = []
            for idx, snippet in enumerate(snippets[:2]):
                fallback_facts.append(
                    FactPayload(
                        id=f"mb_fallback_{idx+1}",
                        text=snippet.preview,
                        citation_id=snippet.chunk_id,
                    )
                )
            if not fallback_facts:
                fallback_facts.append(
                    FactPayload(
                        id="mb_fallback_notice",
                        text="Retrieved evidence but summarization failed due to LLM availability issues.",
                        citation_id=dataset_citation.id,
                    )
                )
            table_rows = [
                [
                    Path(snippet.file).name if snippet.file else snippet.doc_id,
                    snippet.page,
                    snippet.preview,
                    snippet.similarity,
                ]
                for snippet in snippets
            ]
            artifacts = [
                ArtifactPayload(
                    id="mb_snippets",
                    type="table",
                    title="Top retrieved MapBiomas passages",
                    data={
                        "columns": ["Document", "Page", "Preview", "Similarity"],
                        "rows": table_rows,
                    },
                    metadata={"query": query},
                )
            ]
            return RunQueryResponse(
                server="mb_deforest",
                query=query,
                facts=fallback_facts,
                citations=[dataset_citation, *snippet_citations],
                artifacts=artifacts,
                messages=messages,
                next_actions=["Retry MBReportAsk after restoring LLM connectivity"],
            )

        snippet_citations = self._build_snippet_citations(snippets)
        facts = self._split_answer_into_facts(answer, dataset_citation.id)
        if not facts:
            facts = [
                FactPayload(
                    id="mb_summary_fallback",
                    text=snippets[0].preview,
                    citation_id=snippets[0].chunk_id,
                )
            ]

        table_rows = []
        for snippet in snippets:
            table_rows.append(
                [
                    Path(snippet.file).name if snippet.file else snippet.doc_id,
                    snippet.page,
                    snippet.preview,
                    snippet.similarity,
                ]
            )

        artifacts = [
            ArtifactPayload(
                id="mb_snippets",
                type="table",
                title="Top retrieved MapBiomas passages",
                data={
                    "columns": ["Document", "Page", "Preview", "Similarity"],
                    "rows": table_rows,
                },
                metadata={"query": query},
            ),
        ]

        kg = KnowledgeGraphPayload(
            nodes=[
                {"id": "mapbiomas", "label": "MapBiomas", "type": "Organization"},
                {"id": "deforestation", "label": "Deforestation", "type": "Concept"},
            ],
            edges=[
                {"source": "mapbiomas", "target": "deforestation", "type": "REPORTS_ON"},
            ],
        )

        return RunQueryResponse(
            server="mb_deforest",
            query=query,
            facts=facts,
            citations=[dataset_citation, *snippet_citations],
            artifacts=artifacts,
            messages=[],
            kg=kg,
            next_actions=[
                "Call MBReportSearch to inspect additional passages",
                "Use MBReportAsk for a narrative grounded in specific snippets",
            ],
        )


def create_server() -> FastMCP:
    """Factory for CLI execution (`python -m mcp.servers_v2.mb_deforest_server_v2`)."""

    server = MBDeforestServerV2()
    return server.mcp


if __name__ == "__main__":  # pragma: no cover - manual execution path
    create_server().run()
