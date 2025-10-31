"""Science Panel for the Amazon (SPA) MCP server using the v2 contract.

The legacy SPA server exposed a trio of retrieval-augmented generation tools
backed by a Chroma vector index.  This module reimplements the same surface in
the v2 contract while providing structured `run_query` responses that include
facts, citations, and artefacts that downstream orchestrators can compose.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastmcp import FastMCP

# Optional anthropic client for routing
try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

# Optional dotenv support (mirrors legacy behaviour)
try:  # pragma: no cover - defensive import
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
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from ..servers_v2.base import RunQueryMixin
    from ..support_intent import SupportIntent


INDEX_DIR = Path("data/spa_index")
COLLECTION_NAME = "spa_pdfs"
TOP_K = 6

DATASET_ID = "spa"
DATASET_URL = "https://www.theamazonwewant.org/spa-publications"


def _load_dataset_metadata() -> Dict[str, Dict[str, Optional[str]]]:
    """Load dataset titles/citations from the shared catalog."""

    path = Path(__file__).resolve().parents[2] / "static" / "meta" / "datasets.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    for item in payload.get("items", []):
        dataset_id = item.get("id")
        if not dataset_id:
            continue
        mapping[str(dataset_id)] = {
            "title": str(item.get("title", "")) or None,
            "citation": str(item.get("citation", "")) or None,
        }
    return mapping


DATASET_METADATA = _load_dataset_metadata()


def _dataset_title(dataset_id: str, *, default: str) -> str:
    entry = DATASET_METADATA.get(dataset_id)
    if entry and entry.get("title"):
        return entry["title"]  # type: ignore[return-value]
    return default


def _dataset_citation(dataset_id: str) -> Optional[str]:
    entry = DATASET_METADATA.get(dataset_id)
    if entry:
        return entry.get("citation")
    return None


DATASET_NAME = _dataset_title(DATASET_ID, default="Science Panel for the Amazon (SPA) Amazon Assessment")

DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


@dataclass
class RetrievalSnippet:
    """Captured passage returned from the vector store."""

    doc_id: str
    file: Optional[str]
    page: Optional[int]
    text: str
    preview: str
    similarity: float
    chunk_id: str


class SpaServerV2(RunQueryMixin):
    """FastMCP server exposing SPA Amazon Assessment retrieval tools."""

    def __init__(self) -> None:
        self.mcp = FastMCP("spa-server-v2")

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("Set OPENAI_API_KEY in the environment before starting spa_server_v2")
        self.openai = OpenAI(api_key=openai_key)

        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential issues
                print(f"[spa-server] Warning: Anthropic client unavailable: {exc}")

        self._register_capabilities_tool()
        self._register_query_support_tool()
        self._register_list_docs_tool()
        self._register_search_tool()
        self._register_ask_tool()
        self._register_run_query_tool()

    # ------------------------------------------------------------------ helpers
    def _get_collection(self) -> Any:
        if not INDEX_DIR.exists():
            raise RuntimeError("SPA index not found. Build it with build_spa_index.py")
        client = chromadb.Client(Settings(persist_directory=str(INDEX_DIR), is_persistent=True))
        return client.get_or_create_collection(COLLECTION_NAME)

    def _capabilities_metadata(self) -> Dict[str, Any]:
        return {
            "name": "spa",
            "description": "Science Panel for the Amazon assessment passages with retrieval and QA tools.",
            "version": "2.0.0",
            "tags": ["amazon", "science panel", "climate", "biodiversity"],
            "dataset": DATASET_NAME,
            "url": DATASET_URL,
            "tools": [
                "describe_capabilities",
                "query_support",
                "AmazonAssessmentListDocs",
                "AmazonAssessmentSearch",
                "AmazonAssessmentAsk",
                "run_query",
            ],
        }

    def _capability_summary(self) -> str:
        metadata = self._capabilities_metadata()
        return (
            f"Dataset: {metadata['dataset']} ({metadata['description']}). "
            "Provides vector search and question answering over SPA Assessment passages with citations and document metadata."
        )

    def _classify_support(self, query: str) -> SupportIntent:
        if self._anthropic_client:
            try:
                prompt = (
                    "Determine whether the Science Panel for the Amazon assessment dataset should handle the question."
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
                "Determine whether the Science Panel for the Amazon assessment dataset should handle the question."
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
                max_tokens=100,
                temperature=0,
            )
            text = completion.choices[0].message.content.strip()
            intent = self._parse_support_intent(text)
            if intent:
                return intent
        except Exception as exc:  # pragma: no cover - network issues
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
        def _parse(payload: str) -> Optional[Dict[str, Any]]:
            try:
                data = json.loads(payload)
                return data if isinstance(data, dict) else None
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
            file_path = metadata.get("file") if isinstance(metadata, dict) else None
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
        for idx, sentence in enumerate(sentences[:4]):  # limit to first few insights
            facts.append(
                FactPayload(
                    id=f"answer_fact_{idx+1}",
                    text=sentence,
                    citation_id=citation_id,
                )
            )
        if not facts:
            facts.append(
                FactPayload(
                    id="answer_fact",
                    text=cleaned,
                    citation_id=citation_id,
                )
            )
        return facts

    def _build_snippet_citations(self, snippets: Iterable[RetrievalSnippet]) -> List[CitationPayload]:
        citations: List[CitationPayload] = []
        for snippet in snippets:
            file_name = Path(snippet.file).name if snippet.file else snippet.doc_id
            snippet_text = snippet.text.strip() if snippet.text else ""
            if not snippet_text:
                snippet_text = snippet.preview.strip()
            if snippet_text:
                snippet_text = snippet_text[:500]
            citations.append(
                CitationPayload(
                    id=snippet.chunk_id,
                    server="spa",
                    tool="AmazonAssessmentSearch",
                    title=file_name,
                    source_type="Report",
                    description=snippet_text or None,
                    url=None,
                    metadata={
                        "doc_id": snippet.doc_id,
                        "file": snippet.file,
                        "page": snippet.page,
                        "similarity": snippet.similarity,
                        "snippet": snippet_text,
                    },
                )
            )
        return citations

    # ------------------------------------------------------------------ tools
    def _register_capabilities_tool(self) -> None:
        @self.mcp.tool()
        def describe_capabilities(format: str = "json") -> str:  # type: ignore[misc]
            """Describe the SPA dataset, provenance, and exposed tools.

            Example:
                >>> describe_capabilities()
            """

            payload = self._capabilities_metadata()
            return json.dumps(payload) if format == "json" else str(payload)

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> str:  # type: ignore[misc]
            """Estimate whether the SPA dataset can address the question.

            Example:
                >>> query_support("How is deforestation affecting Indigenous health?", {})
            """

            intent = self._classify_support(query)
            payload = {
                "server": "spa",
                "query": query,
                "supported": intent.supported,
                "score": intent.score,
                "reasons": intent.reasons,
            }
            return json.dumps(payload)

    def _register_list_docs_tool(self) -> None:
        @self.mcp.tool()
        def AmazonAssessmentListDocs(limit: int = 50) -> List[Dict[str, Any]]:  # noqa: N802 - legacy casing
            """
            Amazon Assessment index overview. Returns a sample of (file, page).

            Example:
                >>> AmazonAssessmentListDocs(limit=3)
            """

            manifest_path = INDEX_DIR / "manifest.json"
            if not manifest_path.exists():
                return []

            data = json.loads(manifest_path.read_text())
            items = data.get("items", [])[: max(1, limit)]
            return [{"file": item.get("file"), "page": item.get("page") } for item in items]

    def _register_search_tool(self) -> None:
        @self.mcp.tool()
        def AmazonAssessmentSearch(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:  # noqa: N802 - legacy casing
            """
            Vector search over the SPA Amazon Assessment index.

            Example:
                >>> AmazonAssessmentSearch("climate justice", k=3)
            """

            snippets = self._search(query, k)
            return [snippet.__dict__ for snippet in snippets]

    def _register_ask_tool(self) -> None:
        @self.mcp.tool()
        def AmazonAssessmentAsk(query: str, k: int = TOP_K, max_tokens: int = 800) -> Dict[str, Any]:  # noqa: N802 - legacy casing
            """
            Answer a question as a SPA expert using retrieved snippets.

            Returns the model answer, supporting passages, and metadata about the run.

            Example:
                >>> AmazonAssessmentAsk("summarize drivers of deforestation", k=4)
            """

            snippets = self._search(query, k)
            if not snippets:
                return {"answer": "No relevant snippets found in the Amazon Assessment index.", "passages": []}

            context = "\n\n".join(self._format_snippet_for_prompt(snippet) for snippet in snippets)
            system_prompt = (
                "You are a subject-matter expert from the Science Panel for the Amazon. "
                "Provide precise, evidence-based, and policy-relevant answers grounded in the provided snippets. "
                "Cite inline as (filename p.page). If the snippets do not support a claim, say so. Be concise."
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
            return RunQueryResponse(
                server="spa",
                query=query,
                facts=[],
                citations=[],
                artifacts=[],
                messages=[
                    MessagePayload(
                        level="warning",
                        text="No relevant SPA passages were retrieved for the query.",
                    )
                ],
                next_actions=["Invoke AmazonAssessmentSearch with refined terms"],
            )

        context_block = "\n\n".join(self._format_snippet_for_prompt(snippet) for snippet in snippets)
        system_prompt = (
            "You are responding for the Science Panel for the Amazon (SPA). "
            "Use only the provided snippets to craft a short, policy-relevant summary. "
            "Mention uncertainties when present and avoid speculation."
        )
        user_prompt = f"Question: {query}\n\nSnippets:\n{context_block}"

        answer = ""
        try:
            answer = self._call_chat_model(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:  # pragma: no cover - network dependent
            messages = [
                MessagePayload(level="error", text=f"LLM error while generating answer: {exc}"),
            ]
            return RunQueryResponse(
                server="spa",
                query=query,
                facts=[],
                citations=self._build_snippet_citations(snippets),
                artifacts=[],
                messages=messages,
                next_actions=["Retry AmazonAssessmentAsk once API connectivity is restored"],
            )

        dataset_citation = CitationPayload(
            id="spa_dataset",
            server="spa",
            tool="run_query",
            title=DATASET_NAME,
            source_type="Report",
            description=_dataset_citation(DATASET_ID) or "Science Panel for the Amazon consolidated assessment",
            url=DATASET_URL,
        )

        snippet_citations = self._build_snippet_citations(snippets)

        facts = self._split_answer_into_facts(answer, dataset_citation.id)
        if not facts:
            facts = [
                FactPayload(
                    id="spa_summary",
                    text="The SPA dataset did not yield a confident textual summary.",
                    citation_id=dataset_citation.id,
                )
            ]

        kg = KnowledgeGraphPayload(
            nodes=[
                {"id": "science_panel_for_the_amazon", "label": "Science Panel for the Amazon", "type": "Organization"},
                {"id": "amazon_biome", "label": "Amazon Biome", "type": "Region"},
            ],
            edges=[
                {"source": "science_panel_for_the_amazon", "target": "amazon_biome", "type": "STUDIES"}
            ],
        )

        return RunQueryResponse(
            server="spa",
            query=query,
            facts=facts,
            citations=[dataset_citation, *snippet_citations],
            artifacts=[],
            messages=[],
            kg=kg,
            next_actions=[
                "Call AmazonAssessmentSearch to explore additional passages",
                "Use AmazonAssessmentAsk for a focused narrative",
            ],
        )


def create_server() -> FastMCP:
    """Factory for CLI execution (`python -m mcp.servers_v2.spa_server_v2`)."""

    server = SpaServerV2()
    return server.mcp


if __name__ == "__main__":  # pragma: no cover - manual execution path
    create_server().run()
