"""Simplified MCP orchestrator using the v2 run_query contract."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import logging
import os
import re
import uuid
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Set, cast
from urllib.parse import urlparse

from .url_utils import ensure_absolute_url

# ---------------------------------------------------------------------------
# Model/provider configuration (override here instead of env vars if desired)
# ---------------------------------------------------------------------------

FACT_ORDERER_PROVIDER = "openai"  # options: anthropic, openai, auto
FACT_ORDERER_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
FACT_ORDERER_OPENAI_MODEL = "gpt-5.0"

NARRATIVE_SYNTH_PROVIDER = "openai"  # options: anthropic, openai, auto
NARRATIVE_SYNTH_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
NARRATIVE_SYNTH_OPENAI_MODEL = "gpt-4.1-2025-04-14"

_MARKER_PATTERN = re.compile(r"\[\[(F\d+)\]\]")


def _ensure_evidence_markers(paragraphs: List[str], sequence: List[str]) -> List[str]:
    """Ensure paragraphs contain [[F#]] markers covering the evidence sequence."""

    if not paragraphs or not sequence:
        return paragraphs

    combined = "\n".join(paragraphs)
    present = set(_MARKER_PATTERN.findall(combined))
    missing = [fid for fid in sequence if fid not in present]
    if not missing:
        return paragraphs

    updated = list(paragraphs)
    para_count = len(updated)
    idx = 0
    for fid in missing:
        target = idx if idx < para_count else para_count - 1
        if target < 0:
            break
        updated[target] = updated[target].rstrip() + f" [[{fid}]]"
        idx += 1

    return updated


def _extract_openai_text(response: Any) -> str:
    """Extract text from OpenAI responses payloads."""

    # Newer SDKs expose an aggregate field directly
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    parts: List[str] = []

    for item in getattr(response, "output", []) or []:
        content = getattr(item, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "output_text":
                    block_text = block.get("text")
                    if isinstance(block_text, str):
                        parts.append(block_text)
                elif hasattr(block, "text"):
                    parts.append(getattr(block, "text"))
        elif hasattr(item, "text"):
            parts.append(getattr(item, "text"))

    combined = "".join(parts)
    if combined.strip():
        return combined

    raise RuntimeError("OpenAI response missing text output")

try:  # Optional Anthropic routing support
    import anthropic  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    anthropic = None  # type: ignore

try:  # Optional OpenAI routing support
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    OpenAI = None  # type: ignore

try:  # Optional translation support
    from .translation import translate_modules as _translate_modules
except Exception:  # pragma: no cover - translation is optional
    _translate_modules = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


logger = logging.getLogger(__name__)
llm_logger = logging.getLogger("mcp.llm_router")
if not llm_logger.handlers:
    llm_logger.addHandler(logging.NullHandler())

if load_dotenv:
    try:
        load_dotenv()
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[mcp_chat_v2] Warning: load_dotenv failed: {exc}")

from .contract_validation import (
    ContractValidationError,
    validate_final_response,
    validate_query_support_response,
    validate_run_query_response,
)
from .contracts_v2 import (
    ArtifactPayload,
    FactPayload,
    KnowledgeGraphPayload,
    QueryContext,
    QuerySupportPayload,
    RunQueryResponse,
)


DOMAINS_IN_SCOPE = [
    "Climate change, impacts, and policies",
    "Environmental data and sustainability",
    "Energy systems, renewable energy, and solar facilities",
    "Corporate environmental performance and ESG",
    "Water resources, biodiversity, and ecosystems",
    "Environmental regulations, NDCs, and climate governance",
    "Physical climate risks (floods, droughts, heat stress)",
    "GHG emissions and carbon footprint",
    "Environmental justice and climate adaptation",
    "Deforestation and extreme heat",
    "Questions about this project",
]

# Only these servers are trusted to contribute knowledge-graph nodes.
KG_TRUSTED_SERVERS = {"cpr"}


class MultiServerClient:
    """Lightweight multi-server MCP client for v2 orchestration."""

    def __init__(self) -> None:
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self) -> "MultiServerClient":
        await self.exit_stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await self.exit_stack.__aexit__(exc_type, exc, tb)
        finally:
            self.sessions.clear()

    async def connect_to_server(self, server_name: str, server_script_path: str) -> None:
        """Connect to a single MCP server using direct file path execution."""
        print(f"Connecting to {server_name} server at {server_script_path}")
        
        try:
            # Set up environment with proper PYTHONPATH
            env = os.environ.copy()
            
            # Add project root to PYTHONPATH so imports work
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            
            current_py_path = env.get("PYTHONPATH", "")
            if current_py_path:
                if project_root not in current_py_path.split(":"):
                    env["PYTHONPATH"] = f"{project_root}:{current_py_path}"
            else:
                env["PYTHONPATH"] = project_root
            
            server_params = StdioServerParameters(
                command="python",
                args=[server_script_path],
                env=env  # Use the modified environment
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            await session.initialize()
            
            self.sessions[server_name] = session
            print(f"Successfully connected to {server_name} server")
            
        except Exception as e:
            print(f"Failed to connect to {server_name} server: {e}")
            raise


# ----------------------------------------------------------------------------
# Global client management
# ----------------------------------------------------------------------------

_V2_CLIENT: Optional[MultiServerClient] = None
_V2_CLIENT_LOCK = asyncio.Lock()


async def get_v2_client() -> MultiServerClient:
    """Return a singleton MCP client connected to v2 servers."""
    global _V2_CLIENT
    async with _V2_CLIENT_LOCK:
        if _V2_CLIENT is None:
            client = MultiServerClient()
            try:
                await client.__aenter__()

                # Determine directory paths
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                servers_v2_dir = os.path.join(project_root, "mcp", "servers_v2")

                # Connect to v2 servers using direct file paths
                try:
                    await client.connect_to_server("solar", os.path.join(servers_v2_dir, "solar_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to solar server: {e}")

                try:
                    await client.connect_to_server("deforestation", os.path.join(servers_v2_dir, "deforestation_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to deforestation server: {e}")

                try:
                    await client.connect_to_server("extreme_heat", os.path.join(servers_v2_dir, "extreme_heat_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to extreme heat server: {e}")

                try:
                    await client.connect_to_server("brazil_admin", os.path.join(servers_v2_dir, "brazilian_admin_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to brazilian admin server: {e}")

                try:
                    await client.connect_to_server("cpr", os.path.join(servers_v2_dir, "cpr_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to cpr server: {e}")
                
                try:
                    await client.connect_to_server("lse", os.path.join(servers_v2_dir, "lse_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to lse server: {e}")

                try:
                    await client.connect_to_server("gist", os.path.join(servers_v2_dir, "gist_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to gist server: {e}")

                try:
                    await client.connect_to_server("spa", os.path.join(servers_v2_dir, "spa_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to spa server: {e}")

                # try:
                #     await client.connect_to_server("mb_deforest", os.path.join(servers_v2_dir, "mb_deforest_server_v2.py"))
                # except Exception as e:
                #     print(f"Warning: Failed to connect to mb_deforest server: {e}")

                try:
                    await client.connect_to_server("meta", os.path.join(servers_v2_dir, "meta_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to meta server: {e}")

                try:
                    await client.connect_to_server("wmo_cli", os.path.join(servers_v2_dir, "wmo_cli_server_v2.py"))
                except Exception as e:
                    print(f"Warning: Failed to connect to wmo_cli server: {e}")

                # Only set global client if at least one server connected
                if client.sessions:
                    _V2_CLIENT = client
                    print("V2 MCP client initialized successfully")
                else:
                    await client.__aexit__(None, None, None)
                    raise ConnectionError("Failed to connect to any MCP servers")
                    
            except Exception as e:
                try:
                    await client.__aexit__(None, None, None)
                except:
                    pass
                raise e
                
        return _V2_CLIENT


async def cleanup_v2_client():
    """Clean up the global client (useful for testing or shutdown)."""
    global _V2_CLIENT
    async with _V2_CLIENT_LOCK:
        if _V2_CLIENT is not None:
            try:
                await _V2_CLIENT.__aexit__(None, None, None)
            except:
                pass
            finally:
                _V2_CLIENT = None


# ----------------------------------------------------------------------------
# Helper dataclasses
# ----------------------------------------------------------------------------


@dataclass
class ServerManifest:
    name: str
    description: str
    version: Optional[str] = None
    tags: List[str] = None


@dataclass
class NarrativeEvidence:
    id: str
    text: str
    citation_key: str
    citation: CitationPayload
    fact: FactPayload
    server: str
    source_title: str


@dataclass
class NarrativeResult:
    paragraphs: List[str]
    citation_sequence: List[str]


LLM_ROUTING_CONFIG: Dict[str, Dict[str, Any]] = {
    "cpr": {
        "detailed": (
            "Climate and energy policy knowledge graph focused on Brazil. Includes "
            "policy documents, concept relationships (parent/child/related), and "
            "passages with citations. Use when the query mentions laws, policies, "
            "strategies, governance, NDC commitments, or requires textual evidence "
            "from policy documents. If policy or policies are mentioned, you should use this server."
        ),
        "always_include": False,
    },
    "solar": {
        "detailed": (
            "Global solar facility database containing site locations, country and "
            "region coverage, capacity (MW), construction windows, and facility-level "
            "metadata. Use for questions about solar plants, renewable infrastructure, "
            "facility counts, capacity trends, or geographic distribution of solar assets."
        ),
        "always_include": False,
    },
    "gist": {
        "detailed": (
            "Corporate environmental metrics covering water stress (MSA), drought and flood "
            "exposure, extreme heat, biodiversity impacts (PDF/CO2e/LCE), deforestation "
            "proximity, and Scope 3 emissions (with upstream/downstream breakdown). Includes "
            "company profiles and asset-level geospatial risk assessments. Use for questions "
            "about corporate environmental risk, emissions trends, or asset exposure."
            "This server is mostly about company-level environmental risk and emissions."
        ),
        "always_include": False,
    },
    "lse": {
        "detailed": (
            "Comprehensive climate policy database including NDC commitments (targets, "
            "net-zero years, renewable energy goals), domestic policy comparisons, "
            "institutional frameworks, implementation tracking, and TPI emissions "
            "pathways. Use for questions about national or subnational climate policy "
            "details, targets, and governance structures."
        ),
        "always_include": False,
    },
    # "viz": {
    #     "detailed": (
    #         "Visualization server for generating charts and tables from collected data. "
    #         "Supports bar/line/pie charts and comparison tables with percentages or totals. "
    #         "Use only when actual quantitative data is available and needs structured "
    #         "presentation."
    #     ),
    #     "always_include": False,
    # },
    "deforestation": {
        "detailed": (
            "Deforestation area polygons derived from satellite imagery for Brazil. "
            "Use when analyzing forest loss, overlaps with other geographies, or when "
            "visualizing deforestation extents and statistics."
        ),
        "always_include": False,
    },
    "extreme_heat": {
        "detailed": (
            "Top-quintile extreme heat polygons for Brazil derived from ERA5-Land heat index "
            "and MODIS land-surface temperature layers covering 2020-2025. Use when queries "
            "mention heat stress, wet-bulb risk, persistent heat zones, or need geospatial "
            "heat overlays."
        ),
        "always_include": False,
    },
    "wmo_cli": {
        "detailed": (
            "Semantic retrieval across the WMO State of the Climate in Latin America and "
            "Caribbean 2024 report plus IPCC AR6 Chapters 11 and 12. Returns page-cited "
            "passages covering climate extremes, attribution findings, and regional "
            "adaptation insights."
        ),
        "always_include": False,
    },
    "mb_deforest": {
        "detailed": (
            "MapBiomas Annual Deforestation (RAD) 2024 passages detailing land-use change, "
            "biome-level trends, enforcement narratives, and quantitative deforestation "
            "figures with citations."
        ),
        "always_include": False,
    },
    "admin": {
        "detailed": (
            "Brazilian administrative boundaries for municipalities and states. Use to "
            "retrieve polygon geometries, match place names, or constrain analysis to "
            "specific administrative areas within Brazil."
        ),
        "always_include": False,
    },
    # "geospatial": {
    #     "detailed": (
    #         "Spatial correlation engine that analyzes proximity, overlap, or containment "
    #         "between registered datasets. Does not persist data; use to combine facility, "
    #         "deforestation, heat, or boundary layers when spatial relationships matter."
    #     ),
    #     "always_include": False,
    # },
    "meta": {
        "detailed": (
            "Metadata registry covering dataset descriptions, schemas, and provenance. "
            "Use when the query asks about available data sources, their fields, or "
            "dataset documentation. Use when users ask about this project, initiative, system, or tool."
        ),
        "always_include": False,
    },
    "spa": {
        "detailed": (
            "Science Panel for the Amazon Assessment Report 2021 passages offering "
            "evidence-rich excerpts on Amazon ecosystems, Indigenous knowledge, and regional "
            "climate science with citations. Use for Amazon-specific narrative or evidence "
            "requests."
        ),
        "always_include": False,
    },
}


class ManifestRegistry:
    def __init__(self) -> None:
        self._cache: Dict[str, ServerManifest] = {}

    async def ensure_loaded(
        self, client: MultiServerClient, server_name: str
    ) -> Optional[ServerManifest]:
        if server_name in self._cache:
            return self._cache[server_name]

        session = client.sessions.get(server_name)
        if not session:
            return None

        try:
            result = await session.call_tool("describe_capabilities", {})
        except Exception:
            return None

        if not getattr(result, "content", None):
            return None

        try:
            payload = json.loads(result.content[0].text)
        except Exception:
            return None
        manifest = ServerManifest(
            name=payload.get("name", server_name),
            description=payload.get("description", ""),
            version=payload.get("version"),
            tags=list(payload.get("tags", []) or []),
        )
        self._cache[server_name] = manifest
        return manifest


class LLMRelevanceClassifier:
    """Lightweight helper that uses a small LLM to pre-filter servers."""

    _ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
    _OPENAI_MODEL = "gpt-4.1-mini"

    def __init__(self, routing_config: Dict[str, Dict[str, Any]]) -> None:
        self._config = routing_config
        self._anthropic_client = None
        self._openai_client = None

        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
                llm_logger.debug("Initialized Anthropic client for routing")
            except Exception as exc:  # pragma: no cover - network credential issues
                llm_logger.info(f"Anthropic client unavailable: {exc}")

        if self._anthropic_client is None and OpenAI and os.getenv("OPENAI_API_KEY"):
            try:
                self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                llm_logger.debug("Initialized OpenAI client for routing")
            except Exception as exc:  # pragma: no cover
                llm_logger.info(f"OpenAI client unavailable: {exc}")

    @property
    def available(self) -> bool:
        return bool(self._anthropic_client or self._openai_client)

    async def filter_servers(
        self, query: str, server_names: Iterable[str]
    ) -> List[str]:
        if not self.available:
            print("[router-llm] no routing LLM available; using all servers")
            return list(server_names)

        shortlisted: List[str] = []
        evaluation_tasks: List[asyncio.Task] = []
        evaluated_servers: List[str] = []

        for server in server_names:
            config = self._config.get(server)
            if not config:
                shortlisted.append(server)  # No config, keep server by default
                print(f"[router-llm] {server}: no config, default include")
                continue
            if config.get("always_include"):
                shortlisted.append(server)
                print(f"[router-llm] {server}: ALWAYS INCLUDED (config flag)")
                continue
            evaluated_servers.append(server)
            evaluation_tasks.append(
                asyncio.create_task(self._evaluate_server(query, server, config))
            )

        if evaluation_tasks:
            results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            for server, result in zip(evaluated_servers, results):
                if isinstance(result, Exception):
                    print(
                        f"[router-llm] {server}: LLM routing error {result}; including by default"
                    )
                    shortlisted.append(server)
                elif result:
                    shortlisted.append(server)
                    print(f"[router-llm] {server}: INCLUDED by LLM routing")
                else:
                    print(f"[router-llm] {server}: EXCLUDED by LLM routing")

        return shortlisted or list(server_names)

    async def _evaluate_server(
        self, query: str, server_name: str, config: Dict[str, Any]
    ) -> bool:
        description = config.get("detailed") or config.get("brief") or ""
        prompt = (
            f"Query: \"{query}\"\n\n"
            f"Data Source: {server_name}\n"
            f"Capabilities: {description}\n\n"
            "Does this data source seem like it has information that would help answer the query?\n\n"
            "Answer YES if the dataset is likely to help answer the question. Answer NO if it is unlikely to help."
        )

        try:
            print(f"[router-llm] evaluating {server_name} for query: {query}")
            answer = await self._call_model(
                system="You are a precise query router. Reply with only YES or NO.",
                prompt=prompt,
            )
        except Exception as exc:
            raise RuntimeError(exc)

        normalized = answer.strip().upper()
        is_relevant = normalized.startswith("YES")
        print(f"[router-llm] {server_name}: raw answer={normalized}")
        return is_relevant

    async def is_query_in_scope(
        self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """Determine whether the query fits within the assistant's supported domains."""

        if not self.available:
            return True

        recent_context = []
        if conversation_history:
            for message in conversation_history[-4:]:
                role = message.get("role", "user")
                content = (message.get("content") or "").strip()
                if not content:
                    continue
                prefix = "User" if role == "user" else "Assistant"
                recent_context.append(f"{prefix}: {content[:200]}")

        domains = "\n".join(f"- {domain}" for domain in DOMAINS_IN_SCOPE)
        context_block = "\n".join(recent_context) if recent_context else "(no prior context)"

        prompt = (
            "You are a scope classifier for a climate and environmental assistant.\n"
            "Determine if the user query is within supported domains.\n\n"
            f"Conversation context:\n{context_block}\n\n"
            f"Supported domains:\n{domains}\n\n"
            "The assistant also treats meta questions about itself (capabilities, datasets, how to use it) as in scope.\n\n"
            f"Query: \"{query}\"\n\n"
            "Respond with YES if the query is within scope or builds on the context. Respond with NO if it is clearly unrelated."
        )

        try:
            answer = await self._call_model(
                system="You classify queries as in scope or out of scope. Reply YES or NO only.",
                prompt=prompt,
            )
        except Exception as exc:  # pragma: no cover - conservative fallback
            logger.info(f"Scope check failed ({exc}); defaulting to in-scope")
            return True

        normalized = answer.strip().upper()
        in_scope = normalized.startswith("YES")
        if not in_scope:
            logger.info(f"Scope guard flagged query as out-of-scope: {query[:100]}")
        return in_scope

    async def _call_model(self, *, system: str, prompt: str) -> str:
        if self._anthropic_client is not None:
            def _run_anthropic() -> str:
                response = self._anthropic_client.messages.create(
                    model=self._ANTHROPIC_MODEL,
                    max_tokens=16,
                    temperature=0,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            return await asyncio.to_thread(_run_anthropic)

        if self._openai_client is not None:
            def _run_openai() -> str:
                response = self._openai_client.responses.create(
                    model=self._OPENAI_MODEL,
                    input=prompt,
                    max_output_tokens=16,
                )
                first = response.output[0] if getattr(response, "output", None) else None
                if hasattr(first, "text"):
                    return first.text
                raise RuntimeError("OpenAI response missing text output")

            return await asyncio.to_thread(_run_openai)

        raise RuntimeError("No LLM client configured")


class FactOrderer:
    """Use an LLM to propose an evidence ordering."""

    def __init__(self) -> None:
        provider = (FACT_ORDERER_PROVIDER or "anthropic").strip().lower()
        self._provider_preference = (
            provider if provider in {"anthropic", "openai", "auto"} else "anthropic"
        )

        self._anthropic_client = None
        self._openai_client = None

        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
                llm_logger.debug("Initialized Anthropic client for fact ordering")
            except Exception as exc:  # pragma: no cover - credential issues
                llm_logger.info(
                    f"Anthropic client unavailable for fact ordering: {exc}"
                )

        if OpenAI and os.getenv("OPENAI_API_KEY"):
            try:
                self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                llm_logger.debug("Initialized OpenAI client for fact ordering")
            except Exception as exc:  # pragma: no cover - credential issues
                llm_logger.info(
                    f"OpenAI client unavailable for fact ordering: {exc}"
                )

    @property
    def available(self) -> bool:
        return bool(self._anthropic_client or self._openai_client)

    def _choose_provider(self) -> str:
        order: List[str]
        if self._provider_preference == "openai":
            order = ["openai", "anthropic"]
        elif self._provider_preference == "anthropic":
            order = ["anthropic", "openai"]
        else:  # auto
            order = ["openai", "anthropic"]

        for candidate in order:
            if candidate == "anthropic" and self._anthropic_client is not None:
                return "anthropic"
            if candidate == "openai" and self._openai_client is not None:
                return "openai"

        raise RuntimeError("No LLM client configured for FactOrderer")

    async def order(self, query: str, evidences: List[NarrativeEvidence]) -> List[str]:
        default_order = [e.id for e in evidences]
        if len(default_order) <= 1 or not self.available:
            return default_order

        evidence_lines = [f"{item.id}: {item.text}" for item in evidences]
        evidence_blob = "\n".join(evidence_lines)

        prompt = (
            "You organize evidence for analysts. Given a query and evidence items, "
            "return the best presentation order. Respond with JSON of the form "
            "{\"ordered_ids\": [\"F1\", \"F2\", ...]} using only provided IDs."
        )

        user_message = (
            f"Query: {query}\n\n"
            f"Evidence items:\n{evidence_blob}\n\n"
            "Return JSON only."
        )

        def _invoke() -> str:
            provider = self._choose_provider()

            if provider == "anthropic":
                response = self._anthropic_client.messages.create(  # type: ignore[union-attr]
                    model=FACT_ORDERER_ANTHROPIC_MODEL,
                    max_tokens=300,
                    temperature=0.1,
                    system=prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                parts: List[str] = []
                for block in getattr(response, "content", []) or []:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return "\n".join(parts)

            combined_prompt = f"{prompt}\n\n{user_message}"
            response = self._openai_client.responses.create(  # type: ignore[union-attr]
                model=FACT_ORDERER_OPENAI_MODEL,
                input=combined_prompt,
                max_output_tokens=600,
                temperature=0.1,
            )
            return _extract_openai_text(response)

        try:
            raw = await asyncio.to_thread(_invoke)
            data = json.loads(raw)
            ordered = [fid for fid in data.get("ordered_ids", []) if fid in default_order]
            if not ordered:
                raise ValueError("ordered_ids missing or empty")
            remaining = [fid for fid in default_order if fid not in ordered]
            return ordered + remaining
        except Exception as exc:
            llm_logger.warning(f"Fact ordering failed ({exc}); using original order")
            return default_order


class NarrativeSynthesizer:
    """Generate narrative summaries using an LLM when available."""

    def __init__(self) -> None:
        provider = (NARRATIVE_SYNTH_PROVIDER or "anthropic").strip().lower()
        self._provider_preference = (
            provider if provider in {"anthropic", "openai", "auto"} else "anthropic"
        )

        self._anthropic_client = None
        self._openai_client = None

        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
                llm_logger.debug("Initialized Anthropic client for narrative synthesis")
            except Exception as exc:  # pragma: no cover - credential issues
                llm_logger.info(
                    f"Anthropic client unavailable for narrative synthesis: {exc}"
                )

        if OpenAI and os.getenv("OPENAI_API_KEY"):
            try:
                self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                llm_logger.debug("Initialized OpenAI client for narrative synthesis")
            except Exception as exc:  # pragma: no cover - credential issues
                llm_logger.info(
                    f"OpenAI client unavailable for narrative synthesis: {exc}"
                )

    @property
    def available(self) -> bool:
        return bool(self._anthropic_client or self._openai_client)

    def _choose_provider(self) -> str:
        order: List[str]
        if self._provider_preference == "openai":
            order = ["openai", "anthropic"]
        elif self._provider_preference == "anthropic":
            order = ["anthropic", "openai"]
        else:
            order = ["openai", "anthropic"]

        for candidate in order:
            if candidate == "anthropic" and self._anthropic_client is not None:
                return "anthropic"
            if candidate == "openai" and self._openai_client is not None:
                return "openai"

        raise RuntimeError("No LLM client configured for NarrativeSynthesizer")

    async def generate(
        self,
        query: str,
        evidences: List[NarrativeEvidence],
        sequence: Optional[List[str]] = None,
    ) -> NarrativeResult:
        print(
            f"[KGDEBUG] NarrativeSynthesizer.generate start evidences={len(evidences)}",
            flush=True,
        )
        if not evidences:
            llm_logger.info(
                "NarrativeSynthesizer: no citations available; falling back to fact summary"
            )
            return NarrativeResult(paragraphs=[], citation_sequence=[])

        # Trim to avoid exceeding context limits; Sonnet can handle a lot but keep reasonable
        max_items = 40
        trimmed = evidences[:max_items]
        trimmed_ids = [item.id for item in trimmed]
        desired_sequence = sequence or trimmed_ids
        ordered_sequence = [fid for fid in desired_sequence if fid in trimmed_ids]
        if not ordered_sequence:
            ordered_sequence = trimmed_ids

        if not self.available:
            paragraphs = [f"{e.text} [[{e.id}]]" for e in trimmed]
            paragraphs = _ensure_evidence_markers(paragraphs, ordered_sequence)
            return NarrativeResult(paragraphs=paragraphs, citation_sequence=ordered_sequence)

        evidence_lines = []
        for item in trimmed:
            meta = item.source_title or "Dataset"
            evidence_lines.append(f"{item.id}: {item.text} (Source: {meta})")

        evidence_blob = "\n".join(evidence_lines)
        order_instruction = " -> ".join(ordered_sequence)
        current_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        prompt = (
            "The current date is " + current_date + ".\n\n"
            "You are writing an analyst-grade summary for the given query.\n"
            "Assume each question is about Brazil. This is a Brazil-focused assistant. We will add support for other countries in the future.\n"
            "Don't say you're going to answer the question; just answer it.\n"
            "Use the provided evidence items to respond to the user's query. Each item has an ID like F1.\n"
            "Present the evidence in this order: "
            f"{order_instruction}.\n"
            "Open with a single paragraph that directly answers the question, weaving the key takeaway from the strongest evidence and including at least one citation marker (e.g., [[F1]]).\n"
            "Follow with additional paragraphs that expand on each major aspect or module, adding engaging but fact-based detail while keeping the flow cohesive.\n"
            "You must provide the information in an engaging, coherent narrative.\n"
            "Every sentence that uses evidence must include citation markers using the syntax [[F1]] immediately after the relevant clause.\n"
            "If multiple evidence items support a sentence, include multiple markers, e.g., [[F1]][[F3]].\n"
            "Avoid repeating tool scaffolding or status bullet text verbatim—integrate those ideas into polished prose instead of copying headings like 'NDC Overview & Domestic Comparison'.\n"
            "Do NOT write about methodologies, datasets, analysis approaches, or what 'enables' or 'makes possible' any findings.\n"
            "If evidence describes datasets or analytical methods, DO NOT include this information in your response, unless you are asked. Focus only on substantive findings and facts.\n"
            "Do NOT write information ABOUT your data (NDCAlign has these tables, 'I am going to write about this data', etc). Just write the narrative.\n"
            "You have a wealth of information present within your general knowledge and through tools. You are in charge of making this information engaging and compelling. Stay humble but state known facts confidently.\n"
            "Do not start answers with phrases like 'Based on the provided evidence...' OR similar phrasing.\n"
            "Don't mention the names or ids of documents or data sources (e.g., 'According to document F1...'). Instead, just use the citation markers [[F1]].\n"
            "Don't include phrases like 'In this document, we will discuss...' or 'This report provides an analysis of...'. Just present the information directly.\n"
        )

        user_message = (
            f"Query: {query}\n\n"
            f"Evidence items:\n{evidence_blob}\n\n"
            "Return the paragraphs as plain text separated by blank lines."
        )

        system_prompt = "You are a helpful, precise analyst who follows instructions exactly."

        full_system_prompt = f"{system_prompt}\n\n{prompt}"

        def _invoke() -> str:
            provider = self._choose_provider()
            print(f"[KGDEBUG] narrative provider={provider}", flush=True)

            if provider == "anthropic":
                try:
                    print("[KGDEBUG] anthropic call start", flush=True)
                    response = self._anthropic_client.messages.create(  # type: ignore[union-attr]
                        model=NARRATIVE_SYNTH_ANTHROPIC_MODEL,
                        max_tokens=1200,
                        temperature=0.2,
                        system=full_system_prompt,
                        messages=[{"role": "user", "content": user_message}],
                    )
                    print(
                        f"[KGDEBUG] anthropic call success content_blocks={len(getattr(response, 'content', []) or [])}",
                        flush=True,
                    )
                except Exception as anthropic_error:
                    print(f"[KGDEBUG] anthropic call error: {anthropic_error}", flush=True)
                    raise
                parts = []
                for block in getattr(response, "content", []) or []:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                return "\n".join(parts)

            try:
                print("[KGDEBUG] openai call start", flush=True)
                response = self._openai_client.responses.create(  # type: ignore[union-attr]
                    model=NARRATIVE_SYNTH_OPENAI_MODEL,
                    input=[
                        {"role": "system", "content": full_system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_output_tokens=1600,
                    temperature=0.2,
                )
                print("[KGDEBUG] openai call success", flush=True)
            except Exception as openai_error:
                print(f"[KGDEBUG] openai call error: {openai_error}", flush=True)
                raise
            return _extract_openai_text(response)

        print("[KGDEBUG] about to invoke narrative provider", flush=True)
        try:
            print("[KGDEBUG] calling narrative provider", flush=True)
            raw = await asyncio.wait_for(asyncio.to_thread(_invoke), timeout=15)
            print(
                f"[KGDEBUG] narrative provider returned text length={len(raw)}",
                flush=True,
            )
            paragraphs = [p.strip() for p in raw.strip().split("\n\n") if p.strip()]
            if not paragraphs:
                raise ValueError("No paragraphs returned")
            paragraphs = _ensure_evidence_markers(paragraphs, ordered_sequence)
            return NarrativeResult(paragraphs=paragraphs, citation_sequence=ordered_sequence)
        except asyncio.TimeoutError:
            print("[KGDEBUG] narrative provider timed out; falling back to fact list", flush=True)
            llm_logger.info("Narrative synthesizer timed out while calling provider")
        except Exception as exc:
            print(f"[KGDEBUG] narrative provider error: {exc}", flush=True)
            llm_logger.info(f"Narrative synthesis failed ({exc}); falling back to fact list")

        print("[KGDEBUG] returning fallback narrative from evidences", flush=True)
        paragraphs = [f"{e.text} [[{e.id}]]" for e in trimmed]
        paragraphs = _ensure_evidence_markers(paragraphs, ordered_sequence)
        return NarrativeResult(paragraphs=paragraphs, citation_sequence=ordered_sequence)


# ----------------------------------------------------------------------------
# Routing & execution
# ----------------------------------------------------------------------------


class QueryRouter:
    def __init__(
        self,
        client: MultiServerClient,
        manifests: ManifestRegistry,
        llm_classifier: Optional[LLMRelevanceClassifier] = None,
    ) -> None:
        self._client = client
        self._manifests = manifests
        self._llm_classifier = llm_classifier or LLMRelevanceClassifier(
            LLM_ROUTING_CONFIG
        )

    async def route(
        self,
        query: str,
        context: QueryContext,
        progress_callback: Optional[
            Callable[[str, str, Mapping[str, Any]], Awaitable[None]]
        ] = None,
    ) -> List[QuerySupportPayload]:
        server_names = list(self._client.sessions.keys())

        lowered_query = query.lower()
        meta_triggers = (
            "this project",
            "this app",
            "this assistant",
            "how does this work",
            "how does it work",
            "who built you",
            "what can you do",
            "capabilities",
            "what data do you use",
            "what datasets do you use",
            "what sources do you have",
            "tell me about yourself",
            "tell me about the project",
            "what is this project",
            "about this project",
        )

        if any(trigger in lowered_query for trigger in meta_triggers):
            meta_only = [name for name in server_names if name == "meta"]
            if meta_only:
                server_names = meta_only


        if self._llm_classifier and self._llm_classifier.available:
            try:
                filtered = await self._llm_classifier.filter_servers(query, server_names)
                if filtered:
                    server_names = filtered
                else:
                    logger.info("LLM routing returned no servers; using all")
            except Exception as exc:  # pragma: no cover - network failures
                logger.info(f"LLM routing failed ({exc}); falling back to all servers")

        if not server_names:
            server_names = list(self._client.sessions.keys())

        print(f"[router] servers selected: {server_names}")

        async def run_probe(server_name: str, session) -> tuple[str, str, Mapping[str, Any]]:
            try:
                payload = await self._probe_server(session, server_name, query, context)
                return ("success", server_name, payload)
            except ContractValidationError as exc:
                return ("contract_error", server_name, {"error": exc})
            except Exception as exc:
                return ("failure", server_name, {"error": exc})

        tasks = []
        for name in server_names:
            session = self._client.sessions.get(name)
            if not session:
                continue
            llm_logger.debug(f"[router] probing {name} via query_support")
            tasks.append(asyncio.create_task(run_probe(name, session)))

        responses: List[QuerySupportPayload] = []
        for task in asyncio.as_completed(tasks):
            status, server_name, payload = await task
            if status == "success":
                support = payload  # type: ignore[assignment]
                if progress_callback:
                    await progress_callback(
                        server_name,
                        "query_support",
                        {"payload": support},
                    )
                if isinstance(support, QuerySupportPayload) and support.supported:
                    responses.append(support)
            elif status == "contract_error":
                error = payload.get("error")
                print(f"[router] skipping server due to invalid payload: {error}")
                if progress_callback:
                    await progress_callback(
                        server_name,
                        "query_support_error",
                        {"error": error},
                    )
            else:
                error = payload.get("error")
                if progress_callback:
                    await progress_callback(
                        server_name,
                        "query_support_failure",
                        {"error": error},
                    )
                raise error  # type: ignore[misc]

        responses.sort(key=lambda p: p.score, reverse=True)
        return responses

    async def _probe_server(
        self,
        session,
        server_name: str,
        query: str,
        context: QueryContext,
    ) -> QuerySupportPayload:
        await self._manifests.ensure_loaded(self._client, server_name)

        payload = {
            "query": query,
            "context": context.model_dump(mode="json"),
        }

        result = await session.call_tool("query_support", payload)
        if not getattr(result, "content", None):
            raise ContractValidationError(
                f"{server_name} returned empty query_support payload"
            )

        raw = json.loads(result.content[0].text)
        support = validate_query_support_response(raw)
        print(
            f"[router] {server_name} query_support -> supported={support.supported} score={support.score:.2f}"
        )
        return support


RUN_QUERY_TIMEOUT_SECONDS = 60


class QueryExecutor:
    def __init__(self, client: MultiServerClient) -> None:
        self._client = client

    async def execute(
        self,
        supports: Iterable[QuerySupportPayload],
        context: QueryContext,
        progress_callback: Optional[
            Callable[[str, str, Mapping[str, Any]], Awaitable[None]]
        ] = None,
    ) -> List[RunQueryResponse]:
        async def run_query_call(server_name: str, session) -> tuple[str, str, Mapping[str, Any]]:
            try:
                response = await asyncio.wait_for(
                    self._call_run_query(session, server_name, context),
                    timeout=RUN_QUERY_TIMEOUT_SECONDS,
                )
                return ("success", server_name, response)
            except asyncio.TimeoutError:
                timeout_error = TimeoutError(
                    f"run_query exceeded {RUN_QUERY_TIMEOUT_SECONDS}s"
                )
                return ("timeout", server_name, {"error": timeout_error})
            except ContractValidationError as exc:
                return ("contract_error", server_name, {"error": exc})
            except Exception as exc:
                return ("failure", server_name, {"error": exc})

        tasks = []
        for support in supports:
            session = self._client.sessions.get(support.server)
            if not session:
                continue
            tasks.append(asyncio.create_task(run_query_call(support.server, session)))

        responses: List[RunQueryResponse] = []
        for task in asyncio.as_completed(tasks):
            status, server_name, payload = await task
            if status == "success":
                response = payload  # type: ignore[assignment]
                if progress_callback:
                    await progress_callback(
                        server_name,
                        "run_query",
                        {"payload": response},
                    )
                responses.append(response)  # type: ignore[arg-type]
            elif status == "timeout":
                error = payload.get("error")
                if progress_callback:
                    await progress_callback(
                        server_name,
                        "run_query_timeout",
                        {"error": error},
                    )
            elif status == "contract_error":
                error = payload.get("error")
                print(f"[executor] dropping server response: {error}")
                if progress_callback:
                    await progress_callback(
                        server_name,
                        "run_query_error",
                        {"error": error},
                    )
            else:
                error = payload.get("error")
                if progress_callback:
                    await progress_callback(
                        server_name,
                        "run_query_failure",
                        {"error": error},
                    )
                raise error  # type: ignore[misc]
        return responses

    async def _call_run_query(
        self, session, server_name: str, context: QueryContext
    ) -> RunQueryResponse:
        payload = {
            "query": context.query,
            "context": context.model_dump(mode="json"),
        }

        result = await session.call_tool("run_query", payload)
        if not getattr(result, "content", None):
            raise ContractValidationError(
                f"{server_name} returned empty run_query payload"
            )

        raw = json.loads(result.content[0].text)
        response = validate_run_query_response(raw)
        return response


# ----------------------------------------------------------------------------
# Citation registry
# ----------------------------------------------------------------------------


class CitationRegistry:
    def __init__(self) -> None:
        self._counter = 1
        self._mapping: Dict[str, int] = {}
        self._lookup: Dict[int, Any] = {}

    def register(self, citations: Iterable[Any]) -> None:
        for citation in citations:
            key = f"{citation.server}:{citation.tool}:{citation.id}"
            if key in self._mapping:
                continue
            number = self._counter
            self._counter += 1
            self._mapping[key] = number
            self._lookup[number] = citation

    def number_for(self, citation: Any) -> int:
        key = f"{citation.server}:{citation.tool}:{citation.id}"
        if key not in self._mapping:
            raise KeyError(f"citation not registered: {key}")
        return self._mapping[key]

    def to_table_rows(self) -> List[List[str]]:
        rows: List[List[str]] = []
        for number in sorted(self._lookup):
            citation = self._lookup[number]
            rows.append(
                [
                    str(number),
                    citation.title,
                    citation.tool,
                    citation.source_type,
                    citation.description or "",
                    citation.url or "",
                ]
            )
        return rows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citations": {
                number: {
                    "server": citation.server,
                    "tool": citation.tool,
                    "title": citation.title,
                    "source_type": citation.source_type,
                    "description": citation.description,
                    "url": citation.url,
                }
                for number, citation in self._lookup.items()
            }
        }


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------


def detect_language(query: str) -> Optional[str]:
    lowered = query.lower()
    if any(token in lowered for token in (" energia", " clima", " onde", " quais")):
        return "pt"
    return "en"


class SimpleOrchestrator:
    def __init__(self, client: MultiServerClient):
        self._client = client
        self._manifest_registry = ManifestRegistry()
        self._llm_classifier = LLMRelevanceClassifier(LLM_ROUTING_CONFIG)
        self._router = QueryRouter(
            client, self._manifest_registry, self._llm_classifier
        )
        self._executor = QueryExecutor(client)
        self._fact_orderer = FactOrderer()
        self._narrative = NarrativeSynthesizer()
        self._streamed_fact_ids: Set[str] = set()
        self._fact_message_counts: Dict[str, int] = defaultdict(int)
        self._streamed_fact_count: int = 0
        self._max_fact_messages_per_query: int = 12
        self._max_fact_messages_per_server: int = 3
        self._fact_thinking_max_chars: int = 220

    def _format_fact_thinking_message(self, fact: FactPayload) -> Optional[str]:
        """Return a truncated thinking message for textual facts."""

        text = (fact.text or "").strip()
        if not text or fact.kind != "text":
            return None

        max_chars = max(40, self._fact_thinking_max_chars)
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."

        return f'🔍 Relevant Fact Found: "{text}"'

    async def _emit_fact_thinking_events(
        self,
        responses: List[RunQueryResponse],
        emit_callback: Optional[Callable[[str, str], Awaitable[None]]],
    ) -> None:
        if not emit_callback or not responses:
            return

        for response in responses:
            server_name = response.server
            for fact in response.facts:
                if self._streamed_fact_count >= self._max_fact_messages_per_query:
                    return

                fact_key = f"{server_name}:{fact.id}"
                if fact_key in self._streamed_fact_ids:
                    continue

                if self._fact_message_counts[server_name] >= self._max_fact_messages_per_server:
                    continue

                message = self._format_fact_thinking_message(fact)
                if not message:
                    continue

                await emit_callback(message, "fact")
                self._streamed_fact_ids.add(fact_key)
                self._fact_message_counts[server_name] += 1
                self._streamed_fact_count += 1

    async def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        *,
        target_language: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._run_pipeline(
            query,
            conversation_history,
            target_language=target_language,
            session_id=session_id,
        )

    async def _run_pipeline(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        *,
        target_language: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._streamed_fact_ids.clear()
        self._fact_message_counts.clear()
        self._streamed_fact_count = 0

        async def emit(message: str, category: str = "info", event_type: str = "thinking") -> None:
            if not progress_callback:
                return
            await progress_callback(
                {
                    "type": event_type,
                    "data": {"message": message, "category": category},
                }
            )

        previous_user_message: Optional[str] = None
        previous_assistant_message: Optional[str] = None
        if conversation_history:
            for message in reversed(conversation_history):
                role = message.get("role")
                content = message.get("content")
                if role == "assistant" and previous_assistant_message is None:
                    previous_assistant_message = content
                elif role == "user" and previous_user_message is None:
                    previous_user_message = content
                if previous_user_message and previous_assistant_message:
                    break

        session_identifier = session_id or uuid.uuid4().hex

        async def translate_modules_if_needed(
            modules: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            if (
                not modules
                or not target_language
                or not target_language.lower().startswith("pt")
                or _translate_modules is None
            ):
                return modules
            try:
                return await _translate_modules(modules, target_language)
            except Exception:  # pragma: no cover - translation is best effort
                return modules

        language = detect_language(query)
        context = QueryContext(
            query=query,
            conversation=conversation_history or [],
            language=language,
            session_id=session_identifier,
            previous_user_message=previous_user_message,
            previous_assistant_message=previous_assistant_message,
        )

        await emit("🔍 Evaluating query and available tools", "initialization")

        in_scope = True
        if self._llm_classifier:
            try:
                in_scope = await self._llm_classifier.is_query_in_scope(
                    query, conversation_history or []
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.info(f"Scope guard error ({exc}); proceeding with query")

        if not in_scope:
            await emit(
                "🙏 This question looks outside the supported domains; offering guidance instead.",
                "initialization",
            )
            response = self._build_out_of_scope_response(query)
            translated_modules = await translate_modules_if_needed(response.get("modules", []))
            if translated_modules is not response.get("modules"):
                response = dict(response)
                response["modules"] = translated_modules
            validate_final_response(response)
            return response

        pretty_print = {
            "cpr": "Climate Policy Radar Passage Library",
            "solar": "TZ-SAM Database",
            "brazil_admin": "Brazil Administrative Boundaries and Metadata",
            "gist": "GIST Impact",
            "spa": "Science Panel for the Amazon",
            "deforestation": "PRODES",
            "lse": "NDCAlign",
            "wmo_cli": "Scientific Documents for Climate",
            "meta": "Project Specific Knowledge",
        }

        def _pretty_server_name(server_name: str) -> str:
            """Return a human-friendly server name for progress messages."""
            return pretty_print.get(server_name, server_name)

        async def router_progress(
            server_name: str, stage: str, payload: Mapping[str, Any]
        ) -> None:

            if not progress_callback:
                return
            if stage == "query_support":
                support: QuerySupportPayload = payload["payload"]  # type: ignore[index]
                status = "accepted" if support.supported else "rejected"
                message = (
                    f"📡 {_pretty_server_name(server_name)}: relevance score {support.score:.2f} ({status})"
                )
                await emit(message, "routing")
            elif stage == "query_support_error":
                error = payload.get("error")
                await emit(
                    f"⚠️ {_pretty_server_name(server_name)}: invalid query_support payload ({error})",
                    "routing",
                )
            elif stage == "query_support_failure":
                error = payload.get("error")
                await emit(
                    f"❌ {_pretty_server_name(server_name)}: query_support failed ({error})",
                    "routing",
                )

        await emit("🧭 Routing query to candidate MCP servers", "routing")
        supports = await self._router.route(query, context, progress_callback=router_progress)
        if not supports:
            raise ContractValidationError("No servers accepted the query")

        async def executor_progress(
            server_name: str, stage: str, payload: Mapping[str, Any]
        ) -> None:
            if not progress_callback:
                return
            if stage == "run_query":
                response: RunQueryResponse = payload["payload"]  # type: ignore[index]
                message = (
                    f"✅ {_pretty_server_name(server_name)}: returned {len(response.facts)} facts, "
                    f"{len(response.artifacts)} artifacts"
                )
                await emit(message, "execution")
            elif stage == "run_query_error":
                error = payload.get("error")
                await emit(
                    f"⚠️ {_pretty_server_name(server_name)}: invalid run_query payload ({error})",
                    "execution",
                )
            elif stage == "run_query_failure":
                error = payload.get("error")
                await emit(
                    f"❌ {_pretty_server_name(server_name)}: run_query failed ({error})",
                    "execution",
                )
            elif stage == "run_query_timeout":
                error = payload.get("error")
                await emit(
                    f"⏱️ {_pretty_server_name(server_name)}: run_query timed out ({error})",
                    "execution",
                )

        await emit("📥 Gathering detailed responses", "execution")
        responses = await self._executor.execute(
            supports, context, progress_callback=executor_progress
        )
        if not responses:
            raise ContractValidationError("No server produced a response")

        await self._emit_fact_thinking_events(responses, emit)

        await emit("🧠 Synthesizing narrative and assembling modules", "synthesis")
        evidences, evidence_map = self._collect_evidences(responses)
        print(
            f"[KGDEBUG] collected evidences: {len(evidences)} items from {len(responses)} responses",
            flush=True,
        )
        ordered_ids = await self._fact_orderer.order(query, evidences)
        print(
            f"[KGDEBUG] fact orderer returned ids: {ordered_ids[:10]} (total {len(ordered_ids)})",
            flush=True,
        )
        ordered_evidences: List[NarrativeEvidence] = [
            evidence_map[fid] for fid in ordered_ids if fid in evidence_map
        ]
        # Guard against any IDs dropped by the LLM ordering step
        for evidence in evidences:
            if evidence.id not in ordered_ids:
                ordered_evidences.append(evidence)
                ordered_ids.append(evidence.id)

        print(
            f"[KGDEBUG] entering narrative.generate with {len(ordered_evidences)} evidences",
            flush=True,
        )
        narrative_result = await self._narrative.generate(
            query, ordered_evidences, ordered_ids
        )
        print(
            f"[KGDEBUG] narrative.generate finished: paragraphs={len(narrative_result.paragraphs)}",
            flush=True,
        )

        if narrative_result.paragraphs:
            print(
                f"[KGDEBUG] synthesis returned narrative with {len(narrative_result.paragraphs)} paragraphs",
                flush=True,
            )
            sequence = self._determine_citation_sequence(
                narrative_result.paragraphs, narrative_result.citation_sequence
            )
            citation_registry = self._register_citations_in_order(
                responses, sequence, evidence_map
            )
            summary_module = self._build_narrative_summary_module(
                narrative_result.paragraphs,
                sequence,
                evidence_map,
                citation_registry,
            )
        else:
            print("[KGDEBUG] narrative synthesis returned no paragraphs; building summary table", flush=True)
            citation_registry = CitationRegistry()
            for response in responses:
                citation_registry.register(response.citations)
            summary_module = self._build_summary_module(responses, citation_registry)
        artifact_modules = self._build_artifact_modules(responses)
        citation_module = self._build_citation_module(citation_registry)

        modules = [summary_module, *artifact_modules, citation_module]

        modules = self._merge_map_modules(modules)

        kg_nodes, kg_edges, kg_urls = self._combine_kg(responses)
        has_kg_content = bool(kg_nodes and kg_edges)
        print(
            f"[KGDEBUG] _combine_kg -> nodes={len(kg_nodes)} edges={len(kg_edges)} urls={len(kg_urls)} has_content={has_kg_content}",
            flush=True,
        )

        metadata = self._build_metadata(query, modules, kg_available=has_kg_content)
        print(f"[KGDEBUG] metadata built: {metadata}", flush=True)

        modules = await translate_modules_if_needed(modules)

        print(
            "[KGDEBUG] modules after translation:",
            [module.get("type") for module in modules],
            flush=True,
        )

        kg_context_payload: Dict[str, Any] = {
            "nodes": kg_nodes if has_kg_content else [],
            "edges": kg_edges if has_kg_content else [],
        }
        if has_kg_content and kg_urls:
            kg_context_payload["urls"] = kg_urls

        final_payload: Dict[str, Any] = {
            "query": query,
            "modules": modules,
            "metadata": metadata,
            "kg_context": kg_context_payload,
            "citation_registry": citation_registry.to_dict(),
        }
        print(
            "[KGDEBUG] final payload assembled: modules=",
            len(modules),
            " kg_nodes=",
            len(kg_context_payload["nodes"]),
            " citation_count=",
            len(citation_registry.to_dict().get('citations', {})),
            flush=True,
        )

        validate_final_response(final_payload)

        await emit(
            "✅ Response synthesis complete", "synthesis", event_type="thinking_complete"
        )

        return final_payload

    def _build_out_of_scope_response(self, query: str) -> Dict[str, Any]:
        """Return a friendly redirect when the query is out of scope."""

        domain_lines = "\n".join(f"- {domain}" for domain in DOMAINS_IN_SCOPE)
        guidance_module = {
            "type": "text",
            "heading": "",
            "content": (
                "That question is outside of my scope, but I'd be happy to answer questions based on the domains I know more about! "
                "These include:\n"
                f"{domain_lines}\n"
                "Let me know what you'd like to explore next."
            ),
        }
        citation_module = {
            "type": "numbered_citation_table",
            "heading": "References",
            "columns": ["#", "Source", "ID/Tool", "Type", "Description", "SourceURL"],
            "rows": [],
            "allow_empty": True,
        }

        modules = [guidance_module, citation_module]
        metadata = self._build_metadata(query, modules, kg_available=False)
        return {
            "query": query,
            "modules": modules,
            "metadata": metadata,
            "kg_context": {"nodes": [], "edges": []},
            "citation_registry": {"citations": {}},
        }

    def _build_summary_module(
        self, responses: List[RunQueryResponse], registry: CitationRegistry
    ) -> Dict[str, Any]:
        lines: List[str] = []
        metadata_items: List[Dict[str, Any]] = []
        for response in responses:
            citation_index = {
                citation.id: registry.number_for(citation)
                for citation in response.citations
            }
            for fact in response.facts:
                number = citation_index.get(fact.citation_id)
                suffix = f" ^{number}^" if number else ""
                lines.append(f"- {fact.text}{suffix}")
                citation_obj = next(
                    (c for c in response.citations if c.id == fact.citation_id),
                    None,
                )
                metadata_items.append(
                    {
                        "server": response.server,
                        "tool": citation_obj.tool if citation_obj else None,
                        "citation_id": fact.citation_id,
                        "citation_number": number,
                        "source_title": citation_obj.title if citation_obj else None,
                        "source_type": citation_obj.source_type if citation_obj else None,
                        "text": fact.text,
                    }
                )

        if not lines:
            lines.append("No supporting facts were returned.")

        return {
            "type": "text",
            "heading": "Summary",
            "texts": ["\n".join(lines)],
            "metadata": {"facts": metadata_items},
        }

    def _build_narrative_summary_module(
        self,
        paragraphs: List[str],
        sequence: List[str],
        evidence_map: Dict[str, NarrativeEvidence],
        registry: CitationRegistry,
    ) -> Dict[str, Any]:
        pattern = re.compile(r"\[\[(F\d+)\]\]")

        def replace_marker(match: re.Match[str]) -> str:
            fid = match.group(1)
            evidence = evidence_map.get(fid)
            if not evidence:
                return ""
            try:
                number = registry.number_for(evidence.citation)
            except KeyError:
                return ""
            return f"^{number}^"

        rendered: List[str] = []
        for paragraph in paragraphs:
            rendered.append(pattern.sub(replace_marker, paragraph))

        if not rendered:
            rendered = ["No supporting facts were returned."]

        return {
            "type": "text",
            "heading": "Summary",
            "texts": rendered,
            "metadata": {"citations": self._build_citation_metadata(sequence, evidence_map, registry)},
        }

    def _collect_evidences(
        self, responses: List[RunQueryResponse]
    ) -> tuple[List[NarrativeEvidence], Dict[str, NarrativeEvidence]]:
        evidences: List[NarrativeEvidence] = []
        evidence_map: Dict[str, NarrativeEvidence] = {}
        counter = 1

        for response in responses:
            citation_lookup = {citation.id: citation for citation in response.citations}
            for fact in response.facts:
                citation = citation_lookup.get(fact.citation_id)
                if not citation:
                    continue
                evidence_id = f"F{counter}"
                counter += 1
                citation_key = self._citation_key(response.server, citation)
                evidence = NarrativeEvidence(
                    id=evidence_id,
                    text=fact.text,
                    citation_key=citation_key,
                    citation=citation,
                    fact=fact,
                    server=response.server,
                    source_title=citation.title or citation.source_type or response.server,
                )
                evidences.append(evidence)
                evidence_map[evidence_id] = evidence

        return evidences, evidence_map

    def _determine_citation_sequence(
        self, paragraphs: List[str], provided: List[str]
    ) -> List[str]:
        pattern = re.compile(r"\[\[(F\d+)\]\]")
        ordered: List[str] = []

        for paragraph in paragraphs:
            for fid in pattern.findall(paragraph):
                if fid not in ordered:
                    ordered.append(fid)

        for fid in provided:
            if fid not in ordered:
                ordered.append(fid)

        return ordered

    def _register_citations_in_order(
        self,
        responses: List[RunQueryResponse],
        sequence: List[str],
        evidence_map: Dict[str, NarrativeEvidence],
    ) -> CitationRegistry:
        registry = CitationRegistry()
        seen: set[str] = set()

        for fid in sequence:
            evidence = evidence_map.get(fid)
            if not evidence:
                continue
            key = evidence.citation_key
            if key in seen:
                continue
            registry.register([evidence.citation])
            seen.add(key)

        return registry

    @staticmethod
    def _citation_key(server: str, citation: CitationPayload) -> str:
        return f"{server}:{citation.tool}:{citation.id}"

    def _build_citation_metadata(
        self,
        sequence: List[str],
        evidence_map: Dict[str, NarrativeEvidence],
        registry: CitationRegistry,
    ) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for fid in sequence:
            evidence = evidence_map.get(fid)
            if not evidence:
                continue
            try:
                number = registry.number_for(evidence.citation)
            except KeyError:
                number = None
            details.append(
                {
                    "marker": f"^{number}^" if number else None,
                    "number": number,
                    "evidence_id": fid,
                    "server": evidence.server,
                    "tool": evidence.citation.tool,
                    "source_title": evidence.citation.title,
                    "source_type": evidence.citation.source_type,
                    "citation_id": evidence.citation.id,
                }
            )
        return details

    def _build_artifact_modules(
        self, responses: List[RunQueryResponse]
    ) -> List[Dict[str, Any]]:
        modules: List[Dict[str, Any]] = []
        for response in responses:
            for artifact in response.artifacts:
                module = self._artifact_to_module(artifact)
                if module:
                    modules.append(module)
        return modules

    def _artifact_to_module(self, artifact: ArtifactPayload) -> Optional[Dict[str, Any]]:
        if artifact.type == "map":
            if artifact.geojson_url:
                geojson_url = ensure_absolute_url(artifact.geojson_url)
                metadata: Dict[str, Any]
                if isinstance(artifact.metadata, Mapping):
                    metadata = dict(artifact.metadata)
                else:
                    metadata = artifact.metadata or {}

                module: Dict[str, Any] = {
                    "type": "map",
                    "mapType": "geojson_url",
                    "geojson_url": geojson_url,
                    "heading": artifact.title,
                    "metadata": metadata,
                }
                view_state = self._derive_map_view_state(metadata)
                if view_state:
                    module["viewState"] = view_state
                legend = self._derive_map_legend(metadata)
                if legend:
                    module["legend"] = legend
                if "geometry_type" in metadata:
                    module["geometry_type"] = metadata["geometry_type"]
                return module
            if artifact.data:
                metadata: Dict[str, Any]
                if isinstance(artifact.metadata, Mapping):
                    metadata = dict(artifact.metadata)
                else:
                    metadata = artifact.metadata or {}

                module = {
                    "type": "map",
                    "mapType": "geojson",
                    "geojson": artifact.data,
                    "heading": artifact.title,
                    "metadata": metadata,
                }
                view_state = self._derive_map_view_state(metadata)
                if view_state:
                    module["viewState"] = view_state
                legend = self._derive_map_legend(metadata)
                if legend:
                    module["legend"] = legend
                if "geometry_type" in metadata:
                    module["geometry_type"] = metadata["geometry_type"]
                return module
        elif artifact.type == "chart" and artifact.data:
            chart_type = artifact.metadata.get("chartType") if artifact.metadata else "bar"
            return {
                "type": "chart",
                "chartType": chart_type,
                "data": artifact.data,
                "heading": artifact.title,
                "options": artifact.metadata.get("options") if artifact.metadata else {},
            }
        elif artifact.type == "table" and artifact.data:
            columns = list(artifact.data.get("columns", []))
            raw_rows = artifact.data.get("rows", [])
            rows: List[List[Any]] = []
            for row in raw_rows:
                if isinstance(row, Mapping):
                    rows.append([row.get(column) for column in columns])
                elif isinstance(row, (list, tuple)):
                    rows.append(list(row))
                else:
                    rows.append([row])
            return {
                "type": "table",
                "heading": artifact.title,
                "columns": columns,
                "rows": rows,
            }
        return None

    def _merge_map_modules(self, modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        map_indices = sorted(
            [
            idx
            for idx, module in enumerate(modules)
            if module.get("type") == "map" and module.get("mapType") == "geojson_url"
        ]
        )
        if len(map_indices) <= 1:
            return modules

        project_root = Path(__file__).resolve().parents[1]
        static_maps_dir = project_root / "static" / "maps"
        combined_features: List[Dict[str, Any]] = []
        legend_by_key: Dict[str, Dict[str, Any]] = {}
        combined_bounds: Optional[Dict[str, float]] = None
        total_feature_count = 0
        geometry_type = None

        map_modules = [modules[idx] for idx in map_indices]

        for module in map_modules:
            geojson_url = module.get("geojson_url")
            path = self._resolve_geojson_path(geojson_url, project_root)
            if not path or not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            features = payload.get("features", [])
            if isinstance(features, list):
                combined_features.extend(features)
                total_feature_count += len(features)

            metadata = module.get("metadata") or {}
            bounds_candidate = metadata.get("bounds")
            combined_bounds = self._union_bounds(combined_bounds, bounds_candidate)

            legend = metadata.get("legend")
            if isinstance(legend, Mapping):
                items = legend.get("items")
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, Mapping):
                            label = str(item.get("label", "")).strip()
                            if not label:
                                continue
                            key = label.lower()
                            legend_by_key[key] = {
                                "label": label,
                                "color": item.get("color", "#E31A1C"),
                                "description": item.get("description") or "",
                            }

            if geometry_type is None and metadata.get("geometry_type"):
                geometry_type = metadata.get("geometry_type")

        if len(combined_features) <= 1:
            return modules

        combined_filename = f"combined_map_{uuid.uuid4().hex[:10]}.geojson"
        static_maps_dir.mkdir(parents=True, exist_ok=True)
        combined_path = static_maps_dir / combined_filename
        try:
            combined_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": combined_features}),
                encoding="utf-8",
            )
        except Exception:
            return modules

        if geometry_type is None:
            geometry_type = "polygon"

        combined_metadata: Dict[str, Any] = {
            "feature_count": total_feature_count,
            "legend": {
                "title": "Layers",
                "items": list(legend_by_key.values()),
            },
        }
        if geometry_type:
            combined_metadata["geometry_type"] = geometry_type
        if combined_bounds:
            combined_metadata["bounds"] = combined_bounds
            combined_metadata["center"] = {
                "lon": (combined_bounds["west"] + combined_bounds["east"]) / 2,
                "lat": (combined_bounds["south"] + combined_bounds["north"]) / 2,
            }

        combined_heading = self._derive_combined_map_heading(map_modules, legend_by_key)

        combined_module: Dict[str, Any] = {
            "type": "map",
            "mapType": "geojson_url",
            "geojson_url": ensure_absolute_url(f"/static/maps/{combined_filename}"),
            "heading": combined_heading,
            "metadata": combined_metadata,
        }
        view_state = self._derive_map_view_state(combined_metadata)
        if view_state:
            combined_module["viewState"] = view_state
        legend = self._derive_map_legend(combined_metadata)
        if legend:
            combined_module["legend"] = legend
        if combined_metadata.get("geometry_type"):
            combined_module["geometry_type"] = combined_metadata["geometry_type"]

        insert_index = min(map_indices)
        new_modules: List[Dict[str, Any]] = []
        for idx, module in enumerate(modules):
            if idx == insert_index:
                new_modules.append(combined_module)
            if idx in map_indices:
                continue
            new_modules.append(module)

        return new_modules

    @staticmethod
    def _derive_combined_map_heading(
        map_modules: List[Mapping[str, Any]],
        legend_by_key: Mapping[str, Dict[str, Any]],
    ) -> str:
        """Derive a readable heading for a merged map overlay module."""

        def _dedupe(values: Iterable[str]) -> List[str]:
            ordered: List[str] = []
            seen: Set[str] = set()
            for value in values:
                cleaned = value.strip()
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                ordered.append(cleaned)
                seen.add(key)
            return ordered

        headings = _dedupe(str(module.get("heading") or "") for module in map_modules)

        if not headings and legend_by_key:
            legend_labels = _dedupe(str(entry.get("label") or "") for entry in legend_by_key.values())
            headings = legend_labels

        if not headings:
            return "Combined Overlay Map"

        if len(headings) == 1:
            return headings[0]

        if len(headings) == 2:
            return f"{headings[0]} and {headings[1]}"

        truncated = headings[:3]
        if len(headings) <= 3:
            return ", ".join(truncated[:-1]) + f", and {truncated[-1]}"

        remaining = len(headings) - len(truncated)
        prefix = ", ".join(truncated[:-1]) + f", and {truncated[-1]}"
        return f"{prefix} (+{remaining} more overlays)"

    @staticmethod
    def _resolve_geojson_path(geojson_url: Optional[str], project_root: Path) -> Optional[Path]:
        if not geojson_url:
            return None
        parsed = urlparse(str(geojson_url))
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            return None
        path_str = parsed.path if parsed.scheme else geojson_url
        if not path_str:
            return None
        local_path = project_root / path_str.lstrip("/")
        return local_path

    @staticmethod
    def _union_bounds(
        existing: Optional[Dict[str, float]], candidate: Any
    ) -> Optional[Dict[str, float]]:
        if not isinstance(candidate, Mapping):
            return existing
        try:
            c_west = float(candidate["west"])
            c_east = float(candidate["east"])
            c_south = float(candidate["south"])
            c_north = float(candidate["north"])
        except (KeyError, TypeError, ValueError):
            return existing

        if existing is None:
            return {
                "west": c_west,
                "east": c_east,
                "south": c_south,
                "north": c_north,
            }

        return {
            "west": min(existing["west"], c_west),
            "east": max(existing["east"], c_east),
            "south": min(existing["south"], c_south),
            "north": max(existing["north"], c_north),
        }

    @staticmethod
    def _derive_map_view_state(metadata: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if not metadata:
            return None

        def _to_float(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        def _pick(mapping: Mapping[str, Any], *keys: str) -> Optional[float]:
            for key in keys:
                if key in mapping:
                    value = _to_float(mapping.get(key))
                    if value is not None:
                        return value
            return None

        bounds_candidate = metadata.get("bounds") or metadata.get("view_bounds")
        west = south = east = north = None
        if isinstance(bounds_candidate, Mapping):
            west = _pick(bounds_candidate, "west", "min_lon", "min_lng", "minx", "xmin", "left")
            south = _pick(bounds_candidate, "south", "min_lat", "miny", "ymin", "bottom")
            east = _pick(bounds_candidate, "east", "max_lon", "max_lng", "maxx", "xmax", "right")
            north = _pick(bounds_candidate, "north", "max_lat", "maxy", "ymax", "top")
        elif isinstance(bounds_candidate, (list, tuple)) and len(bounds_candidate) >= 4:
            west = _to_float(bounds_candidate[0])
            south = _to_float(bounds_candidate[1])
            east = _to_float(bounds_candidate[2])
            north = _to_float(bounds_candidate[3])

        center_candidate = metadata.get("center") or metadata.get("centroid")
        center_lon = center_lat = None
        if isinstance(center_candidate, Mapping):
            center_lon = _pick(center_candidate, "lon", "lng", "longitude", "x")
            center_lat = _pick(center_candidate, "lat", "latitude", "y")
        elif isinstance(center_candidate, (list, tuple)) and len(center_candidate) >= 2:
            center_lon = _to_float(center_candidate[0])
            center_lat = _to_float(center_candidate[1])

        if center_lon is None or center_lat is None:
            if None not in (west, east, south, north):
                center_lon = (cast(float, west) + cast(float, east)) / 2
                center_lat = (cast(float, south) + cast(float, north)) / 2

        bounds_map: Optional[Dict[str, float]] = None
        if None not in (west, east, south, north):
            bounds_map = {
                "west": cast(float, west),
                "east": cast(float, east),
                "south": cast(float, south),
                "north": cast(float, north),
            }

        zoom_candidate = metadata.get("default_zoom") or metadata.get("zoom") or metadata.get("zoom_hint")
        zoom = _to_float(zoom_candidate)
        if zoom is None and bounds_map is not None:
            lon_span = abs(bounds_map["east"] - bounds_map["west"])
            lat_span = abs(bounds_map["north"] - bounds_map["south"])
            span = max(lon_span, lat_span)
            if span <= 2:
                zoom = 7
            elif span <= 5:
                zoom = 6
            elif span <= 15:
                zoom = 5
            elif span <= 40:
                zoom = 4
            elif span <= 90:
                zoom = 3
            else:
                zoom = 2

        view_state: Dict[str, Any] = {}
        if center_lon is not None and center_lat is not None:
            view_state["center"] = [float(center_lon), float(center_lat)]
        if bounds_map is not None:
            view_state["bounds"] = bounds_map
        if zoom is not None:
            view_state["zoom"] = float(zoom)

        return view_state or None

    @staticmethod
    def _derive_map_legend(metadata: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if not metadata:
            return None

        legend_candidate = metadata.get("legend")
        if isinstance(legend_candidate, Mapping):
            items = legend_candidate.get("items")
            if isinstance(items, (list, tuple)):
                return legend_candidate  # Already properly structured

        layers = metadata.get("layers")
        items: List[Dict[str, Any]] = []

        if isinstance(layers, Mapping):
            color_palette = [
                "#43A047",  # solar / renewable points
                "#1E88E5",
                "#FB8C00",
                "#8E24AA",
                "#F4511E",
                "#00897B",
            ]
            color_index = 0
            for layer_name, layer_meta in layers.items():
                label = str(layer_name).replace("_", " ").title()
                description_parts: List[str] = []
                if isinstance(layer_meta, Mapping):
                    def _safe_int(value: Any) -> Optional[int]:
                        try:
                            if value is None:
                                return None
                            if isinstance(value, int):
                                return value
                            if isinstance(value, float):
                                return int(round(value))
                            return int(float(value))
                        except (TypeError, ValueError):
                            return None

                    plotted_val = layer_meta.get("plotted_count") or layer_meta.get("count")
                    total_val = layer_meta.get("total_facilities") or layer_meta.get("total")
                    plotted_int = _safe_int(plotted_val)
                    total_int = _safe_int(total_val)
                    if plotted_int is not None:
                        if total_int and plotted_int < total_int:
                            description_parts.append(
                                f"{plotted_int} plotted / {total_int} total"
                            )
                        else:
                            description_parts.append(f"{plotted_int} features")
                    total_area = layer_meta.get("total_area_hectares")
                    if total_area:
                        description_parts.append(f"{total_area} ha")
                    capacity = layer_meta.get("total_capacity_mw")
                    if capacity:
                        description_parts.append(f"{capacity} MW")
                    specified_color = layer_meta.get("color") or layer_meta.get("colour")
                else:
                    specified_color = None
                description = ", ".join(description_parts) if description_parts else None

                if isinstance(specified_color, str) and specified_color.strip():
                    color = specified_color.strip()
                else:
                    color = color_palette[color_index % len(color_palette)]
                color_index += 1
                item: Dict[str, Any] = {"label": label, "color": color}
                if description:
                    item["description"] = description
                items.append(item)

        if not items:
            countries = metadata.get("countries")
            if isinstance(countries, (list, tuple)) and countries:
                countries_str = ", ".join(str(country) for country in countries)
                items.append({
                    "label": "Countries",
                    "color": "#1E88E5",
                    "description": countries_str,
                })

        if not items:
            return None

        title = str(metadata.get("legend_title") or metadata.get("heading") or "Layers")
        return {"title": title, "items": items}

    def _build_citation_module(self, registry: CitationRegistry) -> Dict[str, Any]:
        rows = registry.to_table_rows()
        return {
            "type": "numbered_citation_table",
            "heading": "References",
            "columns": [
                "#",
                "Source",
                "ID/Tool",
                "Type",
                "Description",
                "SourceURL",
            ],
            "rows": rows,
        }

    def _normalise_kg_node(
        self, node: Mapping[str, Any], server_name: str
    ) -> Optional[Dict[str, Any]]:
        """Project heterogeneous node payloads into the embed-friendly schema."""

        raw_id = node.get("id") or node.get("node_id") or node.get("wikibase_id")
        node_id = str(raw_id).strip() if raw_id is not None else ""
        if not node_id:
            return None

        label = node.get("label") or node.get("name") or node.get("title") or node_id
        label = str(label).strip() or node_id

        node_type = node.get("type") or node.get("kind") or "Concept"
        if isinstance(node_type, str):
            formatted_type = node_type.replace("_", " ").strip().title() or "Concept"
        else:
            formatted_type = "Concept"

        normalised: Dict[str, Any] = {**node}
        normalised["id"] = node_id
        normalised["label"] = label
        normalised["type"] = formatted_type
        if "importance" not in normalised:
            normalised["importance"] = 1.0 if formatted_type == "Concept" else 0.8
        normalised.setdefault("source_server", server_name)
        return normalised

    def _normalise_kg_edge(self, edge: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Ensure edges expose canonical source/target/type keys."""

        raw_source = edge.get("source") or edge.get("from") or edge.get("src")
        raw_target = edge.get("target") or edge.get("to") or edge.get("dst")
        if raw_source is None or raw_target is None:
            return None

        source = str(raw_source).strip()
        target = str(raw_target).strip()
        if not source or not target:
            return None

        edge_type = edge.get("type") or edge.get("relationship") or edge.get("label") or "related"
        if isinstance(edge_type, str):
            edge_type_norm = edge_type.replace(" ", "_").replace("-", "_").upper()
        else:
            edge_type_norm = "RELATED"

        normalised: Dict[str, Any] = {**edge}
        normalised["source"] = source
        normalised["target"] = target
        normalised["type"] = edge_type_norm or "RELATED"
        normalised.setdefault("label", normalised["type"])
        return normalised

    def _combine_kg(
        self, responses: List[RunQueryResponse]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        urls: List[str] = []

        for response in responses:
            if response.server not in KG_TRUSTED_SERVERS:
                continue

            kg_payload = getattr(response, "kg", None)
            if not kg_payload:
                continue

            for node in getattr(kg_payload, "nodes", []) or []:
                if not isinstance(node, Mapping):
                    continue
                normalised_node = self._normalise_kg_node(node, response.server)
                if not normalised_node:
                    continue
                node_id = normalised_node["id"]
                existing = nodes.get(node_id)
                if existing is None:
                    nodes[node_id] = normalised_node
                else:
                    for key, value in normalised_node.items():
                        if key not in existing or existing[key] in (None, ""):
                            existing[key] = value

            for edge in getattr(kg_payload, "edges", []) or []:
                if not isinstance(edge, Mapping):
                    continue
                normalised_edge = self._normalise_kg_edge(edge)
                if normalised_edge:
                    edges.append(normalised_edge)

            if getattr(kg_payload, "url", None):
                urls.append(kg_payload.url)

        concept_ids = {
            node_id
            for node_id, node in nodes.items()
            if str(node.get("type", "")).lower() == "concept"
        }

        if not concept_ids:
            return [], [], []

        filtered_nodes = [nodes[nid] for nid in concept_ids if nid in nodes]
        filtered_edges = [
            edge
            for edge in edges
            if edge.get("source") in concept_ids and edge.get("target") in concept_ids
        ]

        if not filtered_edges:
            return [], [], []

        return filtered_nodes, filtered_edges, urls

    def _build_metadata(
        self,
        query: str,
        modules: List[Dict[str, Any]],
        *,
        kg_available: bool = False,
    ) -> Dict[str, Any]:
        module_types = [m.get("type", "unknown") for m in modules]
        has_maps = any(t == "map" for t in module_types)
        has_charts = any(t == "chart" for t in module_types)
        has_tables = any(t in {"table", "numbered_citation_table"} for t in module_types)
        metadata: Dict[str, Any] = {
            "modules_count": len(modules),
            "has_maps": has_maps,
            "has_charts": has_charts,
            "has_tables": has_tables,
            "module_types": module_types,
        }
        if kg_available:
            metadata["kg_visualization_url"] = "/kg-viz"
            metadata["kg_query_url"] = f"/kg-viz?query={query.replace(' ', '%20')}"
        return metadata


async def process_query(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    *,
    target_language: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    client = await get_v2_client()
    orchestrator = SimpleOrchestrator(client)
    return await orchestrator.process_query(
        query,
        conversation_history,
        target_language=target_language,
        session_id=session_id,
    )


async def stream_query(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    *,
    target_language: Optional[str] = None,
    session_id: Optional[str] = None,
):
    client = await get_v2_client()
    orchestrator = SimpleOrchestrator(client)

    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

    async def progress(event: Dict[str, Any]) -> None:
        await queue.put(event)

    pipeline_task = asyncio.create_task(
        orchestrator._run_pipeline(
            query,
            conversation_history,
            progress,
            target_language=target_language,
            session_id=session_id,
        )
    )

    try:
        while True:
            if pipeline_task.done():
                while not queue.empty():
                    yield await queue.get()
                break
            event = await queue.get()
            yield event
    finally:
        if not pipeline_task.done():
            pipeline_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pipeline_task

    try:
        final_payload = await pipeline_task
    except Exception as exc:
        yield {
            "type": "error",
            "data": {"message": str(exc)},
        }
        return

    yield {"type": "complete", "data": final_payload}


async def process_chat_query(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    correlation_session_id: Optional[str] = None,
    target_language: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Compatibility wrapper matching legacy orchestrator signature."""

    return await process_query(
        user_query,
        conversation_history,
        target_language=target_language,
        session_id=correlation_session_id,
    )


async def stream_chat_query(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    correlation_session_id: Optional[str] = None,
    target_language: Optional[str] = None,
    **_: Any,
):
    """Compatibility wrapper for streaming interface expected by api_server."""

    async for event in stream_query(
        user_query,
        conversation_history,
        target_language=target_language,
        session_id=correlation_session_id,
    ):
        yield event


async def get_global_client() -> MultiServerClient:
    """Compatibility alias used by api_server startup hooks."""

    return await get_v2_client()


async def cleanup_global_client() -> None:
    """Compatibility alias for shutdown cleanup."""

    await cleanup_v2_client()


__all__ = [
    "process_query",
    "stream_query",
    "process_chat_query",
    "stream_chat_query",
    "get_v2_client",
    "get_global_client",
    "cleanup_v2_client",
    "cleanup_global_client",
    "SimpleOrchestrator",
]
