"""WMO/IPCC climate adaptation MCP server using the v2 contract."""

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastmcp import FastMCP

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

try:  # optional dotenv support
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


def _dataset_citation(dataset_id: Optional[str]) -> Optional[str]:
    if not dataset_id:
        return None
    return DATASET_CITATIONS.get(dataset_id)


DATASETS = {
    "combined": {
        "index": Path("data/wmo-ipcc-clim-adapt"),
        "collection": "wmo-ipcc-clim-adapt",
        "title": "WMO/IPCC Climate Adaptation Corpus",
        "url": None,
        "dataset_id": "wmo_ipcc_climate_assessments",
    },
    "wmo": {
        "index": Path("data/wmo-lac"),
        "collection": "wmo-lac",
        "title": "WMO State of the Climate in Latin America and the Caribbean 2024",
        "url": "https://storymaps.arcgis.com/stories/d5cbf0dac8271a5341e20a9933d7b8a3",
        "dataset_id": "wmo_ipcc_climate_assessments",
    },
    "ipcc_ch11": {
        "index": Path("data/ipcc-ch11"),
        "collection": "ipcc-ch11",
        "title": "IPCC AR6 Weather and Climate Extreme Events in a Changing Climate",
        "url": "https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-11",
        "dataset_id": "ipcc_chapter_11",
    },
    "ipcc_ch12": {
        "index": Path("data/ipcc-ch12"),
        "collection": "ipcc-ch12",
        "title": "IPCC AR6 Central and South America",
        "url": "https://www.ipcc.ch/report/ar6/wg2/chapter/chapter-12",
        "dataset_id": "ipcc_chapter_12",
    },
}

TOP_K_DEFAULT = 6
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


@dataclass
class RetrievalSnippet:
    doc_id: str
    file: Optional[str]
    page: Optional[int]
    text: str
    preview: str
    similarity: float
    chunk_id: str


class WmoCliServerV2(RunQueryMixin):
    """FastMCP server exposing WMO/IPCC retrieval and synthesis tools."""

    def __init__(self) -> None:
        self.mcp = FastMCP("wmo-cli-server-v2")

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("Set OPENAI_API_KEY before launching wmo_cli_server_v2")
        self.openai = OpenAI(api_key=openai_key)

        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential issues
                print(f"[wmo-cli-server] Warning: Anthropic client unavailable: {exc}")

        self._register_capabilities_tool()
        self._register_query_support_tool()
        self._register_search_tool()
        self._register_wmo_ask_tool()
        self._register_ipcc_ch11_ask_tool()
        self._register_ipcc_ch12_ask_tool()
        self._register_run_query_tool()

    # ------------------------------------------------------------------ helpers
    def _get_collection(self, key: str):
        config = DATASETS[key]
        index_dir = config["index"]
        if not index_dir.exists():
            raise RuntimeError(f"Index not found for {key}. Build the embeddings first.")
        client = chromadb.Client(Settings(persist_directory=str(index_dir), is_persistent=True))
        return client.get_or_create_collection(config["collection"])

    def _capabilities_metadata(self) -> Dict[str, Any]:
        return {
            "name": "wmo_cli",
            "description": "WMO State of the Climate in LAC 2024 and IPCC AR6 Chapters 11/12 with retrieval and QA tools.",
            "version": "2.0.0",
            "tags": ["climate", "wmo", "ipcc", "adaptation", "latin america"],
            "dataset": DATASETS["combined"]["title"],
            "url": "https://public.wmo.int/en",
            "tools": [
                "describe_capabilities",
                "query_support",
                "WMOIPCCReportSearch",
                "WMOReportAsk",
                "IPCCCh11ReportAsk",
                "IPCCCh12ReportAsk",
                "run_query",
            ],
        }

    def _capability_summary(self) -> str:
        metadata = self._capabilities_metadata()
        return (
            f"Dataset: {metadata['dataset']} ({metadata['description']}). "
            "Supports adaptation analysis with passages from WMO regional reports and IPCC AR6 chapters."
        )

    def _classify_support(self, query: str) -> SupportIntent:
        if self._anthropic_client:
            try:
                prompt = (
                    "Determine whether the WMO/IPCC climate adaptation corpus should handle the question."
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
                "Determine whether the WMO/IPCC climate adaptation corpus should handle the question."
                " Respond strictly with JSON containing keys 'supported' (true/false) and 'reason' (short explanation).\n"
                f"Dataset capabilities: {self._capability_summary()}\n"
                f"Question: {query}"
            )
            completion = self.openai.chat.completions.create(
                model=CHAT_MODEL,
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
        result = self.openai.embeddings.create(model="text-embedding-3-small", input=[text])
        return result.data[0].embedding

    def _search(self, query: str, dataset_key: str, k: int) -> List[RetrievalSnippet]:
        collection = self._get_collection(dataset_key)
        embeddings = self._embed_query(query)
        response = collection.query(
            query_embeddings=[embeddings],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        snippets: List[RetrievalSnippet] = []
        for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            similarity = 1.0 - float(dist) if dist is not None else 0.0
            file_path = meta.get("file") if isinstance(meta, dict) else None
            doc_id = Path(file_path).stem if file_path else meta.get("doc_id") if isinstance(meta, dict) else f"doc_{idx}"
            snippets.append(
                RetrievalSnippet(
                    doc_id=doc_id or f"doc_{idx}",
                    file=file_path,
                    page=meta.get("page") if isinstance(meta, dict) else None,
                    text=doc,
                    preview=(doc or "")[:240],
                    similarity=round(similarity, 4),
                    chunk_id=f"{doc_id or 'doc'}::chunk_{idx}",
                )
            )
        return snippets

    def _format_snippet(self, snippet: RetrievalSnippet) -> str:
        file_name = Path(snippet.file).name if snippet.file else snippet.doc_id
        page = f" p.{snippet.page}" if snippet.page is not None else ""
        return f"[{file_name}{page}] {snippet.text}"

    def _call_chat(self, *, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        completion = self.openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()

    def _split_answer_into_facts(self, answer: str, default_citation_id: str) -> List[Dict[str, str]]:
        cleaned = answer.strip()
        if not cleaned:
            return []
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
        payloads: List[Dict[str, str]] = []
        for idx, sentence in enumerate(sentences[:4]):
            payloads.append({
                "id": f"wmo_fact_{idx+1}",
                "text": sentence,
                "citation_id": default_citation_id,
            })
        if not payloads:
            payloads.append({"id": "wmo_fact", "text": cleaned, "citation_id": default_citation_id})
        return payloads

    def _infer_source(self, snippet: RetrievalSnippet) -> Dict[str, Optional[str]]:
        file_name = (Path(snippet.file).name if snippet.file else snippet.doc_id).lower()
        if "ipcc" in file_name and "11" in file_name:
            dataset = DATASETS["ipcc_ch11"]
            return {
                "dataset": dataset["title"],
                "url": dataset["url"],
                "tool": "IPCCCh11ReportAsk",
                "dataset_id": dataset.get("dataset_id"),
            }
        if "ipcc" in file_name and "12" in file_name:
            dataset = DATASETS["ipcc_ch12"]
            return {
                "dataset": dataset["title"],
                "url": dataset["url"],
                "tool": "IPCCCh12ReportAsk",
                "dataset_id": dataset.get("dataset_id"),
            }
        if "wmo" in file_name:
            dataset = DATASETS["wmo"]
            return {
                "dataset": dataset["title"],
                "url": dataset["url"],
                "tool": "WMOReportAsk",
                "dataset_id": dataset.get("dataset_id"),
            }
        dataset = DATASETS["combined"]
        return {
            "dataset": dataset["title"],
            "url": dataset.get("url"),
            "tool": "WMOIPCCReportSearch",
            "dataset_id": dataset.get("dataset_id"),
        }

    def _snippet_citations(self, snippets: Iterable[RetrievalSnippet], default_tool: str) -> List[CitationPayload]:
        citations: List[CitationPayload] = []
        for snippet in snippets:
            file_name = Path(snippet.file).name if snippet.file else snippet.doc_id
            description_parts = [f"similarity {snippet.similarity}"]
            if snippet.page is not None:
                description_parts.append(f"page {snippet.page}")
            source = self._infer_source(snippet)
            citation_text = _dataset_citation(source.get("dataset_id"))
            if citation_text and snippet.page is not None:
                description = f"{citation_text} (p. {snippet.page})"
            elif citation_text:
                description = citation_text
            else:
                description = ", ".join(description_parts)
            citations.append(
                CitationPayload(
                    id=snippet.chunk_id,
                    server="wmo_cli",
                    tool=source["tool"] or default_tool,
                    title=source["dataset"],
                    source_type="Report",
                    description=description,
                    url=source["url"],
                    metadata={
                        "file": snippet.file,
                        "page": snippet.page,
                        "similarity": snippet.similarity,
                        "document": file_name,
                    },
                )
            )
        return citations

    # ------------------------------------------------------------------ tools
    def _register_capabilities_tool(self) -> None:
        @self.mcp.tool()
        def describe_capabilities(format: str = "json") -> str:  # type: ignore[misc]
            """Describe the WMO/IPCC corpus and available tools.

            Example:
                >>> describe_capabilities()
            """

            payload = self._capabilities_metadata()
            return json.dumps(payload) if format == "json" else str(payload)

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> str:  # type: ignore[misc]
            """Estimate whether the WMO/IPCC dataset can answer the query."""

            intent = self._classify_support(query)
            payload = {
                "server": "wmo_cli",
                "query": query,
                "supported": intent.supported,
                "score": intent.score,
                "reasons": intent.reasons,
            }
            return json.dumps(payload)

    def _register_search_tool(self) -> None:
        @self.mcp.tool()
        def WMOIPCCReportSearch(query: str, k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:  # noqa: N802
            """
            Vector search over the combined WMO/IPCC index.

            Example:
                >>> WMOIPCCReportSearch("El Niño adaptation", k=4)
            """

            snippets = self._search(query, "combined", k)
            return [snippet.__dict__ for snippet in snippets]

    def _register_wmo_ask_tool(self) -> None:
        @self.mcp.tool()
        def WMOReportAsk(query: str, k: int = TOP_K_DEFAULT, max_tokens: int = 800) -> Dict[str, Any]:  # noqa: N802
            """Answer using the WMO Latin America State of the Climate report."""

            snippets = self._search(query, "wmo", k)
            if not snippets:
                return {"answer": "No relevant snippets matched the WMO index.", "passages": []}

            context = "\n\n".join(self._format_snippet(snippet) for snippet in snippets)
            system_prompt = (
                "You are summarizing evidence from the WMO State of the Climate in Latin America and the Caribbean 2024 report. "
                "Ground every statement in the provided snippets, cite inline as (filename p.X), and highlight adaptation or policy implications."
            )
            user_prompt = f"Question: {query}\n\nSnippets:\n{context}"

            try:
                answer = self._call_chat(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
            except Exception as exc:  # pragma: no cover - network dependent
                return {"answer": f"LLM error: {exc}", "passages": [snippet.__dict__ for snippet in snippets]}

            return {
                "answer": answer,
                "passages": [snippet.__dict__ for snippet in snippets],
                "metadata": {"model": CHAT_MODEL, "top_k": k},
            }

    def _register_ipcc_ch11_ask_tool(self) -> None:
        @self.mcp.tool()
        def IPCCCh11ReportAsk(query: str, k: int = TOP_K_DEFAULT, max_tokens: int = 800) -> Dict[str, Any]:  # noqa: N802
            """Answer using IPCC AR6 Chapter 11 (extreme events)."""

            snippets = self._search(query, "ipcc_ch11", k)
            if not snippets:
                return {"answer": "No relevant snippets matched the IPCC Chapter 11 index.", "passages": []}

            context = "\n\n".join(self._format_snippet(snippet) for snippet in snippets)
            system_prompt = (
                "You are summarizing findings from IPCC AR6 Chapter 11 on Weather and Climate Extreme Events. "
                "Use the snippets only, cite inline, and emphasise attribution, confidence, and regional implications."
            )
            user_prompt = f"Question: {query}\n\nSnippets:\n{context}"

            try:
                answer = self._call_chat(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
            except Exception as exc:  # pragma: no cover - network dependent
                return {"answer": f"LLM error: {exc}", "passages": [snippet.__dict__ for snippet in snippets]}

            return {
                "answer": answer,
                "passages": [snippet.__dict__ for snippet in snippets],
                "metadata": {"model": CHAT_MODEL, "top_k": k},
            }

    def _register_ipcc_ch12_ask_tool(self) -> None:
        @self.mcp.tool()
        def IPCCCh12ReportAsk(query: str, k: int = TOP_K_DEFAULT, max_tokens: int = 800) -> Dict[str, Any]:  # noqa: N802
            """Answer using IPCC AR6 Chapter 12 (Central and South America)."""

            snippets = self._search(query, "ipcc_ch12", k)
            if not snippets:
                return {"answer": "No relevant snippets matched the IPCC Chapter 12 index.", "passages": []}

            context = "\n\n".join(self._format_snippet(snippet) for snippet in snippets)
            system_prompt = (
                "You are summarizing findings from IPCC AR6 Chapter 12 on Central and South America. "
                "Use the snippets only, cite inline, and connect findings to adaptation, governance, and vulnerable populations."
            )
            user_prompt = f"Question: {query}\n\nSnippets:\n{context}"

            try:
                answer = self._call_chat(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
            except Exception as exc:  # pragma: no cover - network dependent
                return {"answer": f"LLM error: {exc}", "passages": [snippet.__dict__ for snippet in snippets]}

            return {
                "answer": answer,
                "passages": [snippet.__dict__ for snippet in snippets],
                "metadata": {"model": CHAT_MODEL, "top_k": k},
            }

    # ------------------------------------------------------------------ run_query
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        previous_user_message = None
        if isinstance(context, Mapping):
            previous_user_message = context.get("previous_user_message") or None

        search_query = query
        prior_user_trimmed = None
        if previous_user_message:
            prior_user_trimmed = str(previous_user_message).strip()
            if prior_user_trimmed:
                search_query = f"{prior_user_trimmed}\nFollow-up question: {query.strip()}".strip()

        snippets = self._search(search_query, "combined", TOP_K_DEFAULT)

        if not snippets:
            notice_citation = CitationPayload(
                id="wmo_notice",
                server="wmo_cli",
                tool="run_query",
                title="No evidence available",
                source_type="Notice",
                description="Combined WMO/IPCC search returned no passages.",
            )
            notice_fact = FactPayload(
                id="wmo_no_results",
                text="No WMO or IPCC passages matched the query; refine focus or specify a region.",
                citation_id=notice_citation.id,
            )
            return RunQueryResponse(
                server="wmo_cli",
                query=query,
                facts=[notice_fact],
                citations=[notice_citation],
                artifacts=[],
                messages=[
                    MessagePayload(level="warning", text="WMO/IPCC search did not return any passages."),
                ],
                next_actions=[
                    "Call WMOIPCCReportSearch with different keywords",
                    "Try WMOReportAsk or IPCCCh11ReportAsk for focused sources",
                ],
            )

        system_prompt = (
            "You are drafting a concise brief grounded in WMO and IPCC evidence for climate extremes and adaptation in Latin America. "
            "Use only the provided snippets, cite inline, and highlight confidence, impacts, and policy implications."
        )
        context_block = "\n\n".join(self._format_snippet(snippet) for snippet in snippets)
        if prior_user_trimmed:
            user_prompt = (
                "Previous user message (use only if relevant to this follow-up):\n"
                f"{prior_user_trimmed}\n\n"
                f"Current question: {query}\n\nSnippets:\n{context_block}"
            )
        else:
            user_prompt = f"Question: {query}\n\nSnippets:\n{context_block}"

        combined_dataset = DATASETS["combined"]
        dataset_description = _dataset_citation(combined_dataset.get("dataset_id"))
        dataset_citation = CitationPayload(
            id="wmo_ipcc_dataset",
            server="wmo_cli",
            tool="run_query",
            title=combined_dataset["title"],
            source_type="Report",
            description=dataset_description or "Combined WMO/IPCC climate adaptation corpus",
        )

        try:
            answer = self._call_chat(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=800)
        except Exception as exc:  # pragma: no cover - network dependent
            fallback_facts = [
                FactPayload(
                    id=f"wmo_fallback_{idx+1}",
                    text=snippet.preview,
                    citation_id=snippet.chunk_id,
                )
                for idx, snippet in enumerate(snippets[:2])
            ]
            snippet_citations = self._snippet_citations(snippets, "WMOIPCCReportSearch")
            return RunQueryResponse(
                server="wmo_cli",
                query=query,
                facts=fallback_facts,
                citations=[dataset_citation, *snippet_citations],
                artifacts=[],
                messages=[MessagePayload(level="error", text=f"LLM error: {exc}")],
                next_actions=["Retry WMOIPCCReportSearch or a focused Ask tool"],
            )

        snippet_citations = self._snippet_citations(snippets, "WMOIPCCReportSearch")
        fact_payloads = self._split_answer_into_facts(answer, dataset_citation.id)
        if not fact_payloads:
            first = snippets[0]
            fact_payloads = [{"id": "wmo_summary_fallback", "text": first.preview, "citation_id": first.chunk_id}]

        snippet_ids = [snippet.chunk_id for snippet in snippets]
        facts: List[FactPayload] = []
        for idx, payload in enumerate(fact_payloads):
            citation_id = payload["citation_id"]
            if citation_id == dataset_citation.id and snippet_ids:
                citation_id = snippet_ids[idx % len(snippet_ids)]
            facts.append(
                FactPayload(
                    id=payload["id"],
                    text=payload["text"],
                    citation_id=citation_id,
                )
            )

        artifacts: List[ArtifactPayload] = []

        kg = KnowledgeGraphPayload(
            nodes=[
                {"id": "wmo", "label": "World Meteorological Organization", "type": "Organization"},
                {"id": "ipcc", "label": "Intergovernmental Panel on Climate Change", "type": "Organization"},
                {"id": "latin_america", "label": "Latin America and the Caribbean", "type": "Region"},
            ],
            edges=[
                {"source": "wmo", "target": "latin_america", "type": "REPORTS_ON"},
                {"source": "ipcc", "target": "latin_america", "type": "ANALYSES"},
            ],
        )

        return RunQueryResponse(
            server="wmo_cli",
            query=query,
            facts=facts,
            citations=[dataset_citation, *snippet_citations],
            artifacts=artifacts,
            messages=[],
            kg=kg,
            next_actions=[
                "Call WMOReportAsk for WMO-specific insights",
                "Call IPCCCh11ReportAsk to focus on extreme events",
            ],
        )


def create_server() -> FastMCP:
    """Factory for CLI execution (`python -m mcp.servers_v2.wmo_cli_server_v2`)."""

    server = WmoCliServerV2()
    return server.mcp


if __name__ == "__main__":  # pragma: no cover - manual execution path
    create_server().run()
