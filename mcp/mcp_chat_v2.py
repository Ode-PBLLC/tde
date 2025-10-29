"""Simplified MCP orchestrator using the v2 run_query contract."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import inspect
import logging
import os
import re
import uuid
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
from urllib.parse import urlparse

from .url_utils import ensure_absolute_url

# ---------------------------------------------------------------------------
# Model/provider configuration (override here instead of env vars if desired)
# ---------------------------------------------------------------------------

# Sonnet 4.5 Name
# claude-sonnet-4-5-20250929

# Sonnet 4 Name
# claude-sonnet-4-20250514

# Sonnet Haiku
# claude-3-5-haiku-20241022

FACT_ORDERER_PROVIDER = "openai"  # options: anthropic, openai, auto
FACT_ORDERER_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
FACT_ORDERER_OPENAI_MODEL = "gpt-5.0"

NARRATIVE_SYNTH_PROVIDER = "openai"  # options: anthropic, openai, auto
NARRATIVE_SYNTH_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
NARRATIVE_SYNTH_OPENAI_MODEL = "gpt-4.1-2025-04-14"

# Query enrichment configuration
QUERY_ENRICHMENT_ENABLED = True  # options: True, False
QUERY_ENRICHMENT_MODEL = "claude-3-5-haiku-20241022"
NARRATIVE_SYNTH_MAX_ATTEMPTS = 3
NARRATIVE_SYNTH_TIMEOUT_SECONDS = 180
NARRATIVE_SYNTH_BASE_RETRY_DELAY = 0.5

# Governance summary configuration
ENABLE_GOVERNANCE_SUMMARY = False
GOVERNANCE_SUMMARY_OPENAI_MODEL = "gpt-4.1-mini"
GOVERNANCE_SUMMARY_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"

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
    elif len(missing) > 0.5*len(present): # only add fallback citations if more than half are missing
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


def _should_inject_governance(scope_level: str, has_narrative: bool) -> bool:
    """Return True when the governance module should be generated."""

    if not ENABLE_GOVERNANCE_SUMMARY:
        return False
    return scope_level.upper() == "IN_SCOPE" and has_narrative


def _build_governance_followup_query(
    original_query: str, narrative_paragraphs: Sequence[str]
) -> str:
    """Construct a focused LSE follow-up query anchored to the final narrative."""

    cleaned_paragraphs = [
        paragraph.strip()
        for paragraph in narrative_paragraphs
        if paragraph and paragraph.strip()
    ]
    narrative_block = "\n\n".join(cleaned_paragraphs)

    prompt_parts = [
        "Governance follow-up for NDC Align:",
        f"User question: {original_query.strip()}",
    ]
    if narrative_block:
        prompt_parts.append(
            "Answer summary that will be presented to the user (use this to ground your search):"
        )
        prompt_parts.append(narrative_block)
    prompt_parts.extend(
        [
            "Return governance evidence that contextualises the answer, including institutions, policy processes, and state-level implementation where relevant.",
            "Prioritise authoritative Brazilian governance data drawn from the NDC Align dataset.",
        ]
    )
    return "\n\n".join(prompt_parts)

try:  # Optional Anthropic routing support
    import anthropic  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    anthropic = None  # type: ignore

try:  # Optional OpenAI routing support
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    OpenAI = None  # type: ignore


class QueryEnricher:
    """Enriches queries with domain context for Brazilian environmental data."""
    
    def __init__(self):
        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception:
                pass  # Will fall back to no enrichment
    
    def enrich_query_with_llm(self, query: str) -> Dict[str, Any]:
        """
        Enriches the query using an LLM to add domain context for Brazilian environmental data.
        
        Args:
            query: The original user query
        
        Returns:
            Dictionary with enrichment data including enriched_query
        """
        if not self._anthropic_client:
            return {
                "original": query,
                "enriched_query": query,
                "error": "No Anthropic client available"
            }
        
        enrichment_prompt = """You are a query enricher for an environmental and climate data system in Brazil. If the user does not specify the area, assume they are discussing Brazil and the surrounding regions.

Your job is to expand the query with relevant domain context, synonyms, and technical terms to improve search and retrieval.

Add relevant terms from these domains when applicable:
- Climate change, impacts, and policies
- Environmental data and sustainability  
- Energy systems, renewable energy, and solar facilities
- Corporate environmental performance and ESG
- Water resources, biodiversity, and ecosystems
- Environmental regulations, NDCs, and climate governance
- Physical climate risks (floods, droughts, heat stress)
- GHG emissions and carbon footprint
- Environmental justice and climate adaptation
- Deforestation and extreme heat

Return your response in this exact format:
QUERY: [enhanced query with additional relevant terms and context]
DOMAINS: [comma-separated list of relevant domains]
TERMS: [comma-separated list of additional technical terms, synonyms, acronyms]

Keep the enhanced query focused and comprehensive."""

        try:
            response = self._anthropic_client.messages.create(
                model=QUERY_ENRICHMENT_MODEL,
                max_tokens=300,
                temperature=0.2,
                system=enrichment_prompt,
                messages=[{"role": "user", "content": query}]
            )
            
            # Extract text from response
            response_text = response.content[0].text if response.content else ""
            enriched_query = response_text.strip()
            
            # Validate enriched query
            if not enriched_query or enriched_query == query:
                return {
                    "original": query,
                    "enriched_query": query,
                    "error": "No enrichment generated"
                }
            
            return {
                "original": query,
                "enriched_query": enriched_query
            }
            
        except Exception as e:
            return {
                "original": query,
                "enriched_query": query,
                "error": f"Enrichment failed: {str(e)}"
            }

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

from utils.dataset_resolver import resolve_dataset_url


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
    CitationPayload,
    FactPayload,
    KnowledgeGraphPayload,
    MessagePayload,
    QueryContext,
    QuerySupportPayload,
    RunQueryResponse,
)
from .utils_mcp import mcp_payload_from_result


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
            "Comprehensive climate policy database covering NDC commitments (targets, "
            "net-zero years), sectoral policies (energy, transport, agriculture, land use, "
            "forestry/deforestation, REDD+), institutional frameworks, implementation tracking, "
            "subnational state governance, and TPI emissions pathways. Use for questions about "
            "Brazil's climate policy details, governance structures, forest conservation measures, "
            "sectoral mitigation plans, or state-level climate action."
        ),
        "always_include": True,
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
    "brazil_admin": {
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
            "This is predominantly a tool for policy makers and analysts. We should almost always choose YES for the 'lse' dataset." # PROMPT ADDITION
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

    async def determine_scope_level(
        self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Classify queries as IN_SCOPE, NEAR_SCOPE, or OUT_OF_SCOPE."""

        if not self.available:
            return "IN_SCOPE"

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
            "Classify the user query into one of three categories:\n"
            "- IN_SCOPE: clearly covered by the domains or builds directly on prior context.\n"
            "- NEAR_SCOPE: climate or environmental in theme but about geographies or datasets the assistant does not cover (e.g., outside Brazil) or otherwise too general.\n"
            "- OUT_OF_SCOPE: unrelated topics (sports, pop culture, etc.) or requests with no climate/environment tie.\n\n"
            f"Conversation context:\n{context_block}\n\n"
            f"Supported domains:\n{domains}\n\n"
            "The assistant treats meta questions about itself (capabilities, datasets, how to use it) as IN_SCOPE.\n"
            "If the user references the previous conversation, consider that within scope.\n\n"
            f"Query: \"{query}\"\n\n"
            "Respond with exactly one token: IN_SCOPE, NEAR_SCOPE, or OUT_OF_SCOPE."
        )

        try:
            answer = await self._call_model(
                system="You classify queries as IN_SCOPE, NEAR_SCOPE, or OUT_OF_SCOPE. Reply with one label only.",
                prompt=prompt,
            )
        except Exception as exc:  # pragma: no cover - conservative fallback
            logger.info(f"Scope check failed ({exc}); defaulting to IN_SCOPE")
            return "IN_SCOPE"

        normalized = answer.strip().upper()
        if normalized.startswith("IN_" ):
            return "IN_SCOPE"
        if normalized.startswith("NEAR"):
            return "NEAR_SCOPE"
        if normalized.startswith("OUT"):
            logger.info(f"Scope guard flagged query as out-of-scope: {query[:100]}")
            return "OUT_OF_SCOPE"

        # Fallback for unexpected outputs
        return "IN_SCOPE"

    async def is_query_in_scope(
        self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        level = await self.determine_scope_level(query, conversation_history)
        return level != "OUT_OF_SCOPE"

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


class OutOfScopeResponder:
    """Generate contextual out-of-scope nudges using a lightweight LLM."""

    _ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
    _OPENAI_MODEL = "gpt-4.1-mini"

    def __init__(self) -> None:
        self._anthropic_client = None
        self._openai_client = None

        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential issues
                logger.info(f"Out-of-scope Anthropic client unavailable: {exc}")

        if self._anthropic_client is None and OpenAI and os.getenv("OPENAI_API_KEY"):
            try:
                self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as exc:  # pragma: no cover - credential issues
                logger.info(f"Out-of-scope OpenAI client unavailable: {exc}")

    @property
    def available(self) -> bool:
        return bool(self._anthropic_client or self._openai_client)

    async def craft_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]],
        domain_lines: str,
        *,
        mode: str = "redirect",
    ) -> Optional[str]:
        if not self.available:
            return None

        recent_messages: List[str] = []
        if conversation_history:
            for message in conversation_history[-6:]:
                content = (message.get("content") or "").strip()
                if not content:
                    continue
                role = message.get("role", "user")
                role_label = "User" if role == "user" else "Assistant"
                recent_messages.append(f"{role_label}: {content}")

        context_block = "\n".join(recent_messages) if recent_messages else "(no prior context)"

        if mode == "bridge":
            prompt = (
                "You are composing a short reply for a climate-policy assistant when the user asks"
                " about climate topics that sit outside the assistant's primary datasets."
                " Use the conversation context provided. If the user is referencing a specific"
                " detail (like a number or name) from earlier messages, restate it accurately."
                " Offer a concise, general answer (1-2 sentences) based on broad climate knowledge"
                " or the conversation context, without inventing detailed statistics. Follow with"
                " a friendly sentence reminding them that the assistant specialises in the domains"
                " listed below. Keep the total response to three sentences or fewer."
                "\n\nSupported domains:\n"
                f"{domain_lines}\n\n"
                "Conversation so far:\n"
                f"{context_block}\n\n"
                "User follow-up:\n"
                f"{query}\n\n"
                "Return plain text without bullet points."
            )
        else:
            prompt = (
                "You are composing a brief follow-up message for a climate-policy assistant when"
                " a user asks something outside its supported domains. Use only the conversation"
                " context provided. If the user is asking about a detail that appears in the prior"
                " conversation, restate it succinctly. Then add a friendly note steering them back"
                " toward the supported domains. Keep the response to three sentences or fewer."
                "\n\nSupported domains:\n"
                f"{domain_lines}\n\n"
                "Conversation so far:\n"
                f"{context_block}\n\n"
                "User follow-up:\n"
                f"{query}\n\n"
                "Return plain text without bullet points."
            )

        try:
            response_text = await self._call_model(
                system=(
                    "You craft concise, context-aware replies. Answer only using the "
                    "information above and encourage the user to ask about supported domains."
                ),
                prompt=prompt,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.info(f"Out-of-scope LLM reply failed: {exc}")
            return None

        cleaned = (response_text or "").strip()
        return cleaned or None

    async def _call_model(self, *, system: str, prompt: str) -> str:
        if self._anthropic_client is not None:
            def _run_anthropic() -> str:
                response = self._anthropic_client.messages.create(
                    model=self._ANTHROPIC_MODEL,
                    max_tokens=256,
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
                    max_output_tokens=256,
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

    def _candidate_providers(self) -> List[str]:
        order: List[str]
        if self._provider_preference == "openai":
            order = ["openai", "anthropic"]
        elif self._provider_preference == "anthropic":
            order = ["anthropic", "openai"]
        else:
            order = ["openai", "anthropic"]

        providers: List[str] = []
        for candidate in order:
            if candidate == "anthropic" and self._anthropic_client is not None:
                providers.append("anthropic")
            elif candidate == "openai" and self._openai_client is not None:
                providers.append("openai")
        return providers

    def _choose_provider(self) -> str:
        providers = self._candidate_providers()
        print(f"[KGDEBUG] available providers: {providers}")
        if providers:
            return providers[0]
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

        def _build_fallback_result() -> NarrativeResult:
            fallback_paragraphs = [f"{e.text} [[{e.id}]]" for e in trimmed]
            fallback_paragraphs = _ensure_evidence_markers(
                fallback_paragraphs, ordered_sequence
            )
            return NarrativeResult(
                paragraphs=fallback_paragraphs, citation_sequence=ordered_sequence
            )

        if not self.available:
            print("[KGDEBUG] using fallback result")
            return _build_fallback_result()

        evidence_lines = []
        for item in trimmed:
            meta = item.source_title or "Dataset"
            evidence_lines.append(f"{item.id}: {item.text} (Source: {meta})")

        evidence_blob = "\n".join(evidence_lines)
        print(f"[KGDEBUG] evidence blob built, length={len(evidence_blob)}")
        order_instruction = " -> ".join(ordered_sequence)
        current_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        # prompt = f"""CONTEXT:
        # - The current date is {current_date}.
        # - You are writing an analyst-grade summary for the given query.
        # - Assume each question is about Brazil. This is a Brazil-focused assistant. We will add support for other countries in the future.
        # - Use the provided evidence items to respond. Each item has an ID like F1.
        # - Present the evidence in this order: {order_instruction}.
        # - This is predominantly a tool for policy makers and analysts. Heavily favor incorporating facts about governance, sourced from the LSE - NDC Align dataset in your response.

        # TONE & OBJECTIVITY:
        # - Maintain a strictly objective tone. Do not offer value judgements, prescriptions, or 'best policy' recommendations.
        # - Draw on verifiable sources and well-established domain knowledge. Use widely recognized facts without special labeling, and flag any assumptions or uncertainties explicitly.
        # - Don't say you're going to answer the question; just answer it.
        # - Stay humble but state well-established facts confidently, making the information engaging and compelling.

        # LIMITATIONS & DATA GAPS:
        # - Before drafting, check whether the evidence omits any topic, metric, geography, or timeframe the user explicitly requested.
        # - If something is missing, begin the Key Takeaways section with a brief sentence such as "Unfortunately, we don't have information on {missing_topic}." Do not speculate beyond the available evidence.
        # - When possible, steer the reader toward domains where you do have evidence (e.g., solar facilities, heat exposure, deforestation, governance commitments) while staying factual.

        # VISUALS & STRUCTURED DATA:
        # - Prefer to leverage multimodal evidence through tool calls (maps, charts, tables) whenever it clarifies the user's question; only skip visuals when they would be redundant or unsupported.
        # - Reference insights drawn from those visuals directly in the narrative, focusing on what the data shows rather than describing the tooling.
        # - If a relevant visual is unavailable, continue with text but do not invent or imply the existence of one.

        # FORMAT & MARKDOWN:
        # - Use Markdown headings. Start with "## Key Takeaways" followed by a tight paragraph (2-3 sentences) that captures the main answer and includes at least one citation.
        # - After the Key Takeaways, create additional "##" sections whose titles reflect the major themes or evidence clusters. Adjust the number of sections to match the query complexity and available information.
        # - Within each section, use Markdown bullets to enumerate concrete findings, reserving short connective sentences for context. Only include tables when they are provided by tools.

        # STRUCTURE & FLOW:
        # - Ensure the Key Takeaways paragraph directly answers the question using the strongest evidence and includes at least one citation marker (e.g., [[F1]]).
        # - Follow with additional sections that elaborate on each major aspect or module, keeping the narrative cohesive and aligned to the evidence order.
        # - Deliver an engaging, coherent narrative that reads fluidly without copying scaffold headings like 'NDC Overview & Domestic Comparison'.
        # - Do not start the answer with phrases like 'Based on the provided evidence...' or other throat-clearing language; jump straight into the substance.

        # CITATIONS & SOURCE HANDLING:
        # - Every sentence that uses evidence must include citation markers using the syntax [[F1]] immediately after the relevant clause.
        # - If multiple evidence items support a sentence, include multiple markers, e.g., [[F1]][[F3]].
        # - Don't mention the names or IDs of documents or data sources (e.g., 'According to document F1...'); use citation markers instead.
        # - Don't include phrases like 'In this document, we will discuss...' or 'This report provides an analysis of...'; present the information directly.

        # CONTENT RESTRICTIONS:
        # - Do NOT write about methodologies, datasets, analysis approaches, or what 'enables' or 'makes possible' any findings unless the user explicitly requested that detail.
        # - If evidence describes datasets or analytical methods, omit those references unless asked; focus on substantive findings and facts.
        # - Do NOT write information about your data (e.g., 'NDCAlign has these tables'); just present the findings.
        # """

        prompt = """CONTEXT:
            - Summarize the evidence in this order: {order_instruction}. Focus on Brazil unless the user explicitly requests comparisons.
            - This is predominantly a tool for policy makers and analysts. Heavily favor incorporating facts about governance, sourced from the LSE - NDC Align dataset in your response if available.

            TONE & OBJECTIVITY:
            - Remain objective; avoid prescriptions or “best policy” language. Use established facts confidently and flag assumptions or gaps.

            LIMITATIONS & DATA GAPS:
            - If requested information is missing, open Key Takeaways with “Unfortunately, we don't have information on {missing_topic}.” Do not
            speculate beyond the evidence.

            VISUALS & STRUCTURED DATA:
            - Create and reference maps, charts, or tables through tools when they clarify the answer; otherwise explain available evidence succinctly.

            FORMAT & MARKDOWN:
            - Start with “## Key Takeaways” and a tight two-sentence paragraph including at least one citation.
            - Add downstream “##” sections named for the major themes; use bullet lists for concrete findings and include tables only when tools provide
            them.

            CITATIONS & SOURCE HANDLING:
            - Place citation markers like [[F1]] immediately after each supported sentence before the period; list multiple markers like [[F1]][[F2]] when multiple items apply.
            - Do NOT create a References section. We handle that elsewhere.
            """

        user_message = (
            f"Query: {query}\n\n"
            f"Evidence items:\n{evidence_blob}\n\n"
            "Return the paragraphs as plain text separated by blank lines."
        )

        system_prompt = (
            "You are a helpful, precise analyst who follows instructions exactly.\n"
            "Maintain a strictly objective tone. Do not offer value judgements, prescriptions, or 'best policy' recommendations.\n"
            "Draw on verifiable sources and well-established domain knowledge. Use widely recognized facts without special labeling, and flag any assumptions or uncertainties explicitly."
        )

        full_system_prompt = f"{system_prompt}\n\n{prompt}"

        def _invoke_with_provider(provider: str) -> str:
            print(f"[KGDEBUG] narrative provider={provider}", flush=True)

            if provider == "anthropic":
                try:
                    print("[KGDEBUG] anthropic call start", flush=True)
                    response = self._anthropic_client.messages.create(  # type: ignore[union-attr]
                        model=NARRATIVE_SYNTH_ANTHROPIC_MODEL,
                        max_tokens=2000,
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
                parts: List[str] = []
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

        print(f"[KGDEBUG] about to call _candidate_providers()")
        providers = self._candidate_providers()
        if not providers:
            print(
                "[KGDEBUG] no narrative providers available; returning fallback narrative",
                flush=True,
            )
            return _build_fallback_result()
        attempt_providers = list(islice(cycle(providers), NARRATIVE_SYNTH_MAX_ATTEMPTS))
        total_attempts = len(attempt_providers)
        attempt_errors: List[str] = []

        for attempt_index, provider in enumerate(attempt_providers, start=1):
            print(
                f"[KGDEBUG] narrative attempt {attempt_index}/{total_attempts} provider={provider}",
                flush=True,
            )
            try:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(_invoke_with_provider, provider),
                    timeout=NARRATIVE_SYNTH_TIMEOUT_SECONDS,
                )
                print(
                    f"[KGDEBUG] narrative provider returned text length={len(raw)}",
                    flush=True,
                )
                paragraphs = [p.strip() for p in raw.strip().split("\n\n") if p.strip()]
                if not paragraphs:
                    raise ValueError("No paragraphs returned")
                paragraphs = _ensure_evidence_markers(paragraphs, ordered_sequence)
                return NarrativeResult(
                    paragraphs=paragraphs, citation_sequence=ordered_sequence
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                print(
                    f"[KGDEBUG] narrative provider timeout on attempt {attempt_index}",
                    flush=True,
                )
                llm_logger.info(
                    "Narrative synthesizer attempt %s/%s with %s timed out",
                    attempt_index,
                    total_attempts,
                    provider,
                )
                attempt_errors.append(f"{provider} timeout")
            except Exception as exc:
                print(
                    f"[KGDEBUG] narrative provider error on attempt {attempt_index}: {exc}",
                    flush=True,
                )
                llm_logger.info(
                    "Narrative synthesis attempt %s/%s with %s failed (%s)",
                    attempt_index,
                    total_attempts,
                    provider,
                    exc,
                )
                attempt_errors.append(f"{provider}: {exc}")

            if attempt_index < total_attempts:
                delay = min(
                    NARRATIVE_SYNTH_BASE_RETRY_DELAY * (2 ** (attempt_index - 1)), 2.0
                )
                print(
                    f"[KGDEBUG] narrative retry sleeping for {delay} seconds",
                    flush=True,
                )
                await asyncio.sleep(delay)

        if attempt_errors:
            llm_logger.info(
                "Narrative synthesizer retries exhausted; falling back to fact list (%s)",
                "; ".join(attempt_errors),
            )

        print("[KGDEBUG] returning fallback narrative from evidences", flush=True)
        return _build_fallback_result()


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


async def list_server_tools(session) -> List[Dict[str, Any]]:
    """Return a uniform manifest describing the tools exposed by a server."""

    result = await session.list_tools()
    manifest: List[Dict[str, Any]] = []
    for tool in getattr(result, "tools", []) or []:
        name = str(getattr(tool, "name", "") or "")
        if not name:
            continue

        summary = getattr(tool, "description", None) or ""
        defaults = getattr(tool, "default_arguments", None) or {}

        schema = (
            getattr(tool, "parameters", None)
            or getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
            or {}
        )
        parameters: List[Dict[str, Any]] = []
        required_parameters: List[str] = list(schema.get("required", []) or [])

        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        for param_name, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            param_info: Dict[str, Any] = {"name": param_name}
            description = spec.get("description") or spec.get("title")
            if description:
                param_info["description"] = description
            if "type" in spec:
                param_info["type"] = spec["type"]
            if "default" in spec:
                param_info["default"] = spec["default"]
            parameters.append(param_info)

        # print(            {
        #         "name": name,
        #         "summary": summary,
        #         "default_arguments": defaults,
        #         "parameters": parameters,
        #         "required_parameters": required_parameters,
        #         "signature": getattr(tool, "signature", ""),
        #         "doc": getattr(tool, "description", ""),
        #     })

        manifest.append(
            {
                "name": name,
                "summary": summary,
                "default_arguments": defaults,
                "parameters": parameters,
                "required_parameters": required_parameters,
                "signature": getattr(tool, "signature", ""),
                "doc": getattr(tool, "description", ""),
            }
        )
    return manifest
class ServerToolPlanner:
    """LLM-assisted selector for per-server tool plans."""

    def __init__(
        self,
        anthropic_client: Any | None,
        openai_client: Any | None,
        *,
        max_tools: int | None = None,
    ) -> None:
        self._anthropic_client = anthropic_client
        self._openai_client = openai_client
        self._anthropic_model = os.getenv(
            "SERVER_PLANNER_MODEL", "claude-3-5-haiku-20241022"
        )
        self._openai_model = os.getenv(
            "OPENAI_SERVER_PLANNER_MODEL", "gpt-4.1-mini"
        )
        default_cap = 10 if max_tools is None else max_tools
        try:
            env_cap = int(os.getenv("SERVER_PLANNER_MAX_TOOLS", str(default_cap)))
            self._max_tools = max(1, env_cap)
        except ValueError:
            self._max_tools = default_cap

    async def plan(
        self,
        server_name: str,
        tools_manifest: List[Dict[str, Any]],
        context: QueryContext,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if not tools_manifest:
            return []

        manifest_entries = [
            {
                "name": item["name"],
                "summary": item.get("summary", ""),
                "default_arguments": item.get("default_arguments", {}),
            }
            for item in tools_manifest
        ]
        manifest_text = json.dumps(manifest_entries, ensure_ascii=False)

        query = context.query or ""
        previous_user = context.previous_user_message or ""
        previous_assistant = context.previous_assistant_message or ""

        user_prompt = (
            "You are selecting the best tools from ONE MCP server.\n"
            "Return JSON of the form {\"tools\": [{\"name\": str, \"arguments\": dict}]}.\n"
            f"Select up to {self._max_tools} tools; each tool may be used at most once.\n"
            "Prefer tools that return charts, maps, tables, or concise summaries that answer the question.\n"
            "If no tools clearly match, return an empty list.\n\n"
            f"Server: {server_name}\n"
            f"Available tools: {manifest_text}\n\n"
            f"Current query: {query}\n"
            f"Previous user message: {previous_user}\n"
            f"Previous assistant message: {previous_assistant}"
        )

        plan_payload: Optional[Mapping[str, Any]] = None

        if self._anthropic_client is not None:
            def _invoke_anthropic() -> Any:
                kwargs = dict(
                    model=self._anthropic_model,
                    max_tokens=500,
                    temperature=0,
                    system="Select the best combination of tools. Return JSON only.",
                    messages=[{"role": "user", "content": user_prompt}],
                )
                try:
                    return self._anthropic_client.messages.create(
                        response_format={"type": "json"},
                        **kwargs,
                    )
                except TypeError:
                    return self._anthropic_client.messages.create(**kwargs)

            try:
                response = await asyncio.to_thread(_invoke_anthropic)
                plan_payload = self._extract_payload_from_anthropic(response)
            except Exception:
                plan_payload = None

        if plan_payload is None and self._openai_client is not None:
            def _invoke_openai() -> Optional[str]:
                response = self._openai_client.responses.create(
                    model=self._openai_model,
                    input=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_output_tokens=500,
                    temperature=0,
                )
                return _extract_openai_text(response)

            try:
                raw_text = await asyncio.to_thread(_invoke_openai)
                if raw_text:
                    plan_payload = json.loads(raw_text)
            except Exception:
                plan_payload = None

        plan = self._build_plan_from_payload(plan_payload, tools_manifest, context)
        if plan:
            return plan[: self._max_tools]

        return self._fallback_plan(server_name, tools_manifest, context)

    def _extract_payload_from_anthropic(self, response: Any) -> Optional[Mapping[str, Any]]:
        content_blocks = getattr(response, "content", []) or []
        for block in content_blocks:
            if getattr(block, "type", "") == "json" and isinstance(block.json, Mapping):
                return block.json

        combined = "".join(
            getattr(block, "text", "") for block in content_blocks if hasattr(block, "text")
        ).strip()
        if not combined:
            return None
        try:
            payload = json.loads(combined)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, Mapping) else None

    def _build_plan_from_payload(
        self,
        payload: Optional[Mapping[str, Any]],
        manifest: List[Dict[str, Any]],
        context: QueryContext,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if not payload:
            return []
        tools = payload.get("tools") if isinstance(payload, Mapping) else None
        if not isinstance(tools, list):
            return []

        manifest_index = {item["name"]: item for item in manifest}
        plan: List[Tuple[str, Dict[str, Any]]] = []
        seen: Set[str] = set()
        for entry in tools:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or name not in manifest_index:
                continue
            if name in seen:
                continue
            arguments = entry.get("arguments")
            merged_args = dict(manifest_index[name].get("default_arguments", {}) or {})
            if isinstance(arguments, Mapping):
                merged_args.update(arguments)
            required_params = manifest_index[name].get("required_parameters", []) or []
            parameter_specs = manifest_index[name].get("parameters", []) or []
            param_names = {
                spec.get("name")
                for spec in parameter_specs
                if isinstance(spec, Mapping) and spec.get("name")
            }

            if merged_args:
                alias_pairs = (
                    ("radius", "radius_km"),
                    ("distance", "radius_km"),
                    ("assetType", "asset_type"),
                )
                for alias, target in alias_pairs:
                    if alias in merged_args and target not in merged_args:
                        if not param_names or target in param_names:
                            merged_args[target] = merged_args.pop(alias)
                        elif param_names:
                            merged_args.pop(alias)

            for param_name in required_params:
                if param_name not in merged_args:
                    fallback_value = self._fallback_argument_value(
                        param_name, context
                    )
                    if fallback_value is not None:
                        merged_args[param_name] = fallback_value

            if param_names:
                merged_args = {
                    key: value for key, value in merged_args.items() if key in param_names
                }

            missing_required = [
                param_name for param_name in required_params if param_name not in merged_args
            ]
            if missing_required:
                continue
            plan.append((name, merged_args))
            seen.add(name)
            if len(plan) >= self._max_tools:
                break
        return plan

    @staticmethod
    def _fallback_argument_value(param_name: str, context: QueryContext) -> Optional[str]:
        seed = context.query or context.previous_user_message or ""
        if not seed:
            return None
        normalized_name = param_name.lower()
        if normalized_name in {"query", "text", "concept", "concept_a", "concept_b"}:
            return seed
        return None

    def _fallback_plan(
        self,
        server_name: str,
        manifest: List[Dict[str, Any]],
        context: QueryContext,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        plan: List[Tuple[str, Dict[str, Any]]] = []
        manifest_index = {item["name"]: item for item in manifest}

        def _add(
            name: str,
            overrides: Optional[Dict[str, Any]] = None,
            fill_required_with_query: bool = False,
        ) -> None:
            if name not in manifest_index or any(existing[0] == name for existing in plan):
                return
            entry = manifest_index[name]
            merged = dict(entry.get("default_arguments", {}) or {})
            if overrides:
                merged.update({k: v for k, v in overrides.items() if v is not inspect._empty})
            if fill_required_with_query:
                fill_value = context.query or context.previous_user_message or ""
                if fill_value:
                    for param_name in entry.get("required_parameters", []) or []:
                        merged.setdefault(param_name, fill_value)
            if len(plan) < self._max_tools:
                plan.append((name, merged))

        lowered_query = (context.query or "").lower()

        if server_name == "solar":
            wants_map = any(keyword in lowered_query for keyword in ["map", "show", "locat", "where", "distribution"])
            wants_state_rank = any(keyword in lowered_query for keyword in ["state", "rank", "capacity", "mw"])
            wants_timeline = any(keyword in lowered_query for keyword in ["timeline", "trend", "over time", "year"])

            if wants_map:
                _add("get_solar_facilities_map_data", {"country": "Brazil"})
            if wants_state_rank:
                _add("get_solar_capacity_by_state", {})
            if wants_timeline:
                _add("get_solar_construction_timeline", {"country": "Brazil"})

            if not plan:
                _add("get_solar_facilities_by_country", {"country": "Brazil", "limit": 100})
        elif server_name == "cpr":
            concept_seed = context.query or context.previous_user_message or ""
            if concept_seed:
                _add("GetDescription", {"concept": concept_seed})
                _add("GetRelatedConcepts", {"concept": concept_seed})
                _add("GetSubconcepts", {"concept": concept_seed})
                _add("GetConceptGraphNeighbors", {"concept": concept_seed, "limit": 15})
            else:
                _add("GetDescription", fill_required_with_query=True)
        elif server_name == "spa":
            query_seed = context.query or context.previous_user_message or ""
            if query_seed:
                _add("AmazonAssessmentSearch", {"query": query_seed, "k": 5})
                _add("AmazonAssessmentAsk", {"query": query_seed, "k": 5})
            else:
                _add("AmazonAssessmentSearch", fill_required_with_query=True)
        elif server_name == "gist":
            wants_risk = any(
                keyword in lowered_query
                for keyword in [
                    "risk",
                    "hazard",
                    "exposure",
                    "vulnerability",
                    "high-risk",
                ]
            )
            wants_company_rollup = any(
                keyword in lowered_query
                for keyword in [
                    "company",
                    "sector",
                    "portfolio",
                    "industry",
                ]
            )

            if wants_risk:
                _add("GetGistVisualizationData", {"viz_type": "risk_distribution"})
            if wants_company_rollup:
                _add("GetGistCompaniesBySector", {})

        return plan


async def execute_server_plan(
    server_name: str,
    session,
    plan: List[Tuple[str, Dict[str, Any]]],
    *,
    context: QueryContext,
    default_citation: Optional[CitationPayload] = None,
    debug_details: Optional[Dict[str, Any]] = None,
) -> Optional[RunQueryResponse]:
    """Execute a planned subset of tools and normalize into RunQueryResponse."""

    if not plan:
        if debug_details is not None:
            debug_details.clear()
            debug_details.update(
                {
                    "requested_tools": [],
                    "tool_results": [],
                    "total_facts": 0,
                    "total_artifacts": 0,
                    "fallback_reason": "plan is empty",
                }
            )
        return None

    facts: List[FactPayload] = []
    artifacts: List[ArtifactPayload] = []
    messages: List[MessagePayload] = []
    citations: Dict[str, CitationPayload] = {}
    kg_nodes: List[Dict[str, Any]] = []
    kg_edges: List[Dict[str, Any]] = []

    if debug_details is not None:
        debug_details.clear()
        debug_details.update(
            {
                "requested_tools": [name for name, _ in plan],
                "tool_results": [],
                "total_facts": 0,
                "total_artifacts": 0,
                "fallback_reason": None,
            }
        )

    if default_citation:
        citations[default_citation.id] = default_citation

    for index, (tool_name, arguments) in enumerate(plan, start=1):
        tool_debug: Dict[str, Any] = {"tool": tool_name, "status": "success"}
        try:
            raw_result = await session.call_tool(tool_name, arguments or {})
            payload = mcp_payload_from_result(raw_result)
        except Exception as exc:
            messages.append(
                MessagePayload(
                    level="warning",
                    text=f"{server_name}.{tool_name} failed: {exc}",
                )
            )
            if debug_details is not None:
                tool_debug.update(
                    {
                        "status": "error",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                debug_details["tool_results"].append(tool_debug)
            continue

        citation_payload: Optional[CitationPayload] = None
        citation_obj = payload.get("citation")
        if isinstance(citation_obj, Mapping):
            citation_id = str(
                citation_obj.get("id") or f"{server_name}:{tool_name}:{index}"
            )
            citation_payload = CitationPayload(
                id=citation_id,
                server=server_name,
                tool=str(citation_obj.get("tool") or tool_name),
                title=str(citation_obj.get("title") or server_name),
                source_type=str(citation_obj.get("source_type") or "Dataset"),
                description=citation_obj.get("description"),
                url=citation_obj.get("url"),
            )
            citations.setdefault(citation_payload.id, citation_payload)
        elif default_citation:
                citation_payload = default_citation

        citation_id = citation_payload.id if citation_payload else None

        summary = payload.get("summary")
        has_summary = bool(summary)
        fact_items = [fact for fact in payload.get("facts", []) or [] if fact]
        artifact_items = payload.get("artifacts", []) or []
        message_items = payload.get("messages", []) or []
        kg_payload = payload.get("kg")

        if summary:
            facts.append(
                FactPayload(
                    id=f"{tool_name}_summary_{index}",
                    text=str(summary),
                    citation_id=citation_id,
                )
            )

        for fact_index, fact_text in enumerate(fact_items, start=1):
            if not fact_text:
                continue
            facts.append(
                FactPayload(
                    id=f"{tool_name}_fact_{index}_{fact_index}",
                    text=str(fact_text),
                    citation_id=citation_id,
                )
            )

        for artifact_index, artifact in enumerate(artifact_items, start=1):
            artifact_type = str(artifact.get("type", "") or "")
            title = str(artifact.get("title") or f"{server_name} artifact {artifact_index}")
            metadata = dict(artifact.get("metadata") or {})
            artifact_kwargs: Dict[str, Any] = {
                "id": f"{tool_name}_artifact_{index}_{artifact_index}",
                "type": artifact_type or "table",
                "title": title,
                "metadata": metadata,
            }

            if artifact_type == "map":
                geojson_url = artifact.get("geojson_url") or artifact.get("url")
                if geojson_url:
                    artifact_kwargs["geojson_url"] = ensure_absolute_url(geojson_url)
                artifact_kwargs["url"] = geojson_url
                if artifact.get("data") is not None:
                    artifact_kwargs["data"] = artifact.get("data")
            elif artifact_type == "chart":
                artifact_kwargs["data"] = artifact.get("data")
            elif artifact_type == "table":
                artifact_kwargs["data"] = {
                    "columns": artifact.get("columns", []),
                    "rows": artifact.get("rows", []),
                }
            else:
                artifact_kwargs["data"] = artifact.get("data")
                if artifact.get("url"):
                    artifact_kwargs["url"] = artifact.get("url")

            artifacts.append(ArtifactPayload(**artifact_kwargs))

        for message in payload.get("messages", []) or []:
            if isinstance(message, Mapping):
                text = str(message.get("text", "")).strip()
                if not text:
                    continue
                level = str(message.get("level", "info"))
            else:
                text = str(message).strip()
                if not text:
                    continue
                level = "info"
            messages.append(MessagePayload(level=level, text=text))

        if isinstance(kg_payload, Mapping):
            nodes = kg_payload.get("nodes")
            edges = kg_payload.get("edges")
            if isinstance(nodes, list):
                kg_nodes.extend([node for node in nodes if isinstance(node, Mapping)])
            if isinstance(edges, list):
                kg_edges.extend([edge for edge in edges if isinstance(edge, Mapping)])

        if debug_details is not None:
            produced_output = any(
                [
                    has_summary,
                    bool(fact_items),
                    bool(artifact_items),
                    bool(message_items),
                    isinstance(kg_payload, Mapping)
                    and bool(kg_payload.get("nodes") or kg_payload.get("edges")),
                ]
            )
            tool_debug.update(
                {
                    "status": tool_debug.get("status", "success"),
                    "produced_output": produced_output,
                    "fact_count": len(fact_items) + (1 if has_summary else 0),
                    "artifact_count": len(artifact_items),
                    "message_count": len(message_items),
                }
            )
            debug_details["tool_results"].append(tool_debug)

    if not facts and not artifacts:
        if debug_details is not None:
            debug_details.update(
                {
                    "total_facts": 0,
                    "total_artifacts": 0,
                    "fallback_reason": "planned tools returned no facts or artifacts",
                }
            )
        return None

    citation_list = list(citations.values())

    if debug_details is not None:
        debug_details.update(
            {
                "total_facts": len(facts),
                "total_artifacts": len(artifacts),
                "fallback_reason": None,
            }
        )

    return RunQueryResponse(
        server=server_name,
        query=context.query,
        facts=facts,
        citations=citation_list,
        artifacts=artifacts,
        messages=messages,
        kg=KnowledgeGraphPayload(nodes=kg_nodes, edges=kg_edges),
    )


class SimpleOrchestrator:
    def __init__(self, client: MultiServerClient):
        self._client = client
        self._manifest_registry = ManifestRegistry()
        self._llm_classifier = LLMRelevanceClassifier(LLM_ROUTING_CONFIG)
        self._router = QueryRouter(
            client, self._manifest_registry, self._llm_classifier
        )
        self._fact_orderer = FactOrderer()
        self._narrative = NarrativeSynthesizer()
        self._query_enricher = QueryEnricher()
        self._enable_enrichment = QUERY_ENRICHMENT_ENABLED
        self._streamed_fact_ids: Set[str] = set()
        self._fact_message_counts: Dict[str, int] = defaultdict(int)
        self._streamed_fact_count: int = 0
        self._max_fact_messages_per_query: int = 12
        self._max_fact_messages_per_server: int = 3
        self._fact_thinking_max_chars: int = 160
        self._server_tool_manifests: Dict[str, List[Dict[str, Any]]] = {}
        planner_anthropic = getattr(self._narrative, "_anthropic_client", None)
        planner_openai = getattr(self._narrative, "_openai_client", None)
        if planner_anthropic is None and anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                planner_anthropic = anthropic.Anthropic()
            except Exception:
                planner_anthropic = None
        if planner_openai is None and OpenAI and os.getenv("OPENAI_API_KEY"):
            try:
                planner_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                planner_openai = None
        self._tool_planner = ServerToolPlanner(
            planner_anthropic,
            planner_openai,
        )
        self._out_of_scope_responder = OutOfScopeResponder()
        self._governance_anthropic_client = None
        self._governance_openai_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._governance_anthropic_client = anthropic.Anthropic()
            except Exception as exc:
                logger.info(f"Governance Anthropic client unavailable: {exc}")
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            try:
                self._governance_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as exc:
                logger.info(f"Governance OpenAI client unavailable: {exc}")

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

    def _format_fact_thinking_message(self, fact: FactPayload) -> Optional[str]:
        """Return a truncated thinking message for textual facts."""

        text = (fact.text or "").strip()
        if not text or fact.kind != "text":
            return None

        max_chars = max(40, self._fact_thinking_max_chars)
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."

        return f'🔍 Considering the following information "{text}"'

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

    async def _execute_with_planning(
        self,
        supports: Iterable[QuerySupportPayload],
        context: QueryContext,
        progress_callback: Optional[
            Callable[[str, str, Mapping[str, Any]], Awaitable[None]]
        ] = None,
    ) -> List[RunQueryResponse]:
        responses: List[RunQueryResponse] = []

        for support in supports:
            if not support.supported:
                continue
            session = self._client.sessions.get(support.server)
            if not session:
                continue

            manifest = self._server_tool_manifests.get(support.server)
            if manifest is None:
                try:
                    manifest = await list_server_tools(session)
                except Exception as exc:
                    manifest = []
                    print(
                        f"[planner] Failed to load manifest for {support.server}: {exc}",
                        flush=True,
                    )
                self._server_tool_manifests[support.server] = manifest

            plan: List[Tuple[str, Dict[str, Any]]] = []
            fallback_reason: Optional[str] = None
            try:
                print(
                    f"[planner] Planning tools for {support.server} (tool_count={len(manifest)})",
                    flush=True,
                )
                plan = await self._tool_planner.plan(support.server, manifest, context)
            except Exception as exc:
                print(
                    f"[planner] Tool planning failed for {support.server}: {exc}",
                    flush=True,
                )
                plan = []
                fallback_reason = (
                    f"planner error ({exc.__class__.__name__}: {exc})"
                )

            skip_run_query_on_empty_plan = False
            if plan:
                print(
                    f"[planner] {support.server} selected tools: {[name for name, _ in plan]}",
                    flush=True,
                )
            else:
                print(
                    f"[planner] {support.server} returned no tools; will fallback to run_query",
                    flush=True,
                )
                fallback_reason = fallback_reason or "planner returned no tools"
                skip_run_query_on_empty_plan = support.server in {"deforestation"}

            if plan and progress_callback:
                await progress_callback(
                    support.server,
                    "tool_plan",
                    {"tools": [name for name, _ in plan]},
                )

            response: Optional[RunQueryResponse] = None
            plan_debug: Dict[str, Any] = {}
            if plan:
                try:
                    response = await execute_server_plan(
                        support.server,
                        session,
                        plan,
                        context=context,
                        debug_details=plan_debug,
                    )
                    if response and progress_callback:
                        await progress_callback(
                            support.server,
                            "tool_execute",
                            {
                                "tools": [name for name, _ in plan],
                                "payload": response,
                            },
                        )
                    if response:
                        print(
                            f"[planner] {support.server} tool execution succeeded",
                            flush=True,
                        )
                    else:
                        if fallback_reason is None:
                            fallback_reason = plan_debug.get("fallback_reason")
                        failed_tools = [
                            entry
                            for entry in plan_debug.get("tool_results", [])
                            if entry.get("status") == "error"
                        ]
                        if failed_tools:
                            failure_summaries = \
                                ", ".join(
                                    f"{item['tool']}: {item.get('error')}"
                                    for item in failed_tools
                                    if item.get("tool")
                                )
                            if failure_summaries:
                                print(
                                    f"[planner] {support.server} tool errors: {failure_summaries}",
                                    flush=True,
                                )
                        no_output_tools = [
                            entry["tool"]
                            for entry in plan_debug.get("tool_results", [])
                            if entry.get("status") == "success"
                            and not entry.get("produced_output")
                        ]
                        if no_output_tools:
                            no_output_summary = ", ".join(no_output_tools)
                            print(
                                f"[planner] {support.server} tools produced no output: {no_output_summary}",
                                flush=True,
                            )
                except Exception as exc:
                    if progress_callback:
                        await progress_callback(
                            support.server,
                            "tool_execute_failure",
                            {"error": exc, "tools": [name for name, _ in plan]},
                        )
                    print(
                        f"[planner] {support.server} tool execution failed: {exc}; falling back",
                        flush=True,
                    )
                    response = None
                    fallback_reason = (
                        f"tool execution error ({exc.__class__.__name__}: {exc})"
                    )

            if response is None:
                if fallback_reason is None and plan_debug:
                    fallback_reason = plan_debug.get("fallback_reason")
                reason_to_log = fallback_reason or "unknown planner fallback reason"
                if skip_run_query_on_empty_plan:
                    print(
                        f"[planner] {support.server} skipping run_query fallback ({reason_to_log})",
                        flush=True,
                    )
                    continue
                print(
                    f"[planner] {support.server} falling back to run_query ({reason_to_log})",
                    flush=True,
                )
                try:
                    response = await asyncio.wait_for(
                        self._call_run_query(session, support.server, context),
                        timeout=RUN_QUERY_TIMEOUT_SECONDS,
                    )
                    if progress_callback:
                        await progress_callback(
                            support.server,
                            "run_query",
                            {"payload": response},
                        )
                    print(
                        f"[planner] {support.server} run_query fallback succeeded",
                        flush=True,
                    )
                except asyncio.TimeoutError as exc:
                    if progress_callback:
                        await progress_callback(
                            support.server,
                            "run_query_timeout",
                            {"error": exc},
                        )
                    print(
                        f"[planner] {support.server} run_query timed out: {exc}",
                        flush=True,
                    )
                    continue
                except ContractValidationError as exc:
                    if progress_callback:
                        await progress_callback(
                            support.server,
                            "run_query_error",
                            {"error": exc},
                        )
                    print(
                        f"[planner] {support.server} run_query contract error: {exc}",
                        flush=True,
                    )
                    continue
                except Exception as exc:
                    if progress_callback:
                        await progress_callback(
                            support.server,
                            "run_query_failure",
                            {"error": exc},
                        )
                    print(
                        f"[planner] {support.server} run_query failed: {exc}",
                        flush=True,
                    )
                    continue

            if response:
                responses.append(response)

        return responses

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
        previous_response_modules: Optional[List[Dict[str, Any]]] = None
        if conversation_history:
            for message in reversed(conversation_history):
                role = message.get("role")
                content = message.get("content")
                if role == "assistant" and previous_assistant_message is None:
                    previous_assistant_message = content
                    structured_payload = message.get("structured")
                    if (
                        previous_response_modules is None
                        and isinstance(structured_payload, Mapping)
                    ):
                        modules_candidate = structured_payload.get("modules")
                        if isinstance(modules_candidate, list):
                            previous_response_modules = modules_candidate
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

        # Store original query for final response
        original_query = query
        processing_query = query
        
        # Enrich query for better internal processing (routing, fact extraction)
        if self._enable_enrichment:
            try:
                enrichment_data = self._query_enricher.enrich_query_with_llm(query)
                if "error" not in enrichment_data:
                    enriched_query = enrichment_data.get("enriched_query", query)
                    # if enriched_query and enriched_query.strip() and enriched_query != query:
                    #     processing_query = enriched_query
                    #     await emit("🔍 Query enriched for better processing", "initialization")
                    # else:
                    #     await emit("🔍 Query enrichment returned no changes", "initialization")
                else:
                    await emit("⚠️ Query enrichment failed, using original query", "initialization")
            except Exception as exc:
                await emit(f"⚠️ Query enrichment error: {exc}, using original query", "initialization")

        language = detect_language(processing_query)
        context = QueryContext(
            query=processing_query,  # Use enriched query for internal processing
            conversation=conversation_history or [],
            language=language,
            session_id=session_identifier,
            previous_user_message=previous_user_message,
            previous_assistant_message=previous_assistant_message,
            previous_response_modules=previous_response_modules,
        )

        await emit("🔍 Considering which sources are relevant to your query...", "initialization")

        scope_level = "IN_SCOPE"
        if self._llm_classifier:
            try:
                scope_level = await self._llm_classifier.determine_scope_level(
                    query, conversation_history or []
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.info(f"Scope guard error ({exc}); proceeding with query")

        if scope_level == "OUT_OF_SCOPE":
            await emit(
                "🙏 This question looks outside the supported domains; offering guidance instead.",
                "initialization",
            )
            response = await self._build_out_of_scope_response(
                query,
                conversation_history or [],
            )
            translated_modules = await translate_modules_if_needed(response.get("modules", []))
            if translated_modules is not response.get("modules"):
                response = dict(response)
                response["modules"] = translated_modules
            validate_final_response(response)
            return response

        if scope_level == "NEAR_SCOPE":
            await emit(
                "ℹ️ Sharing a brief high-level answer, then steering back to core domains",
                "initialization",
            )
            response = await self._build_near_scope_response(
                query,
                conversation_history or [],
            )
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
            "wmo_cli": "Scientific Documents for Climate (WMO and IPCC)",
            "meta": "Project Specific Knowledge",
            "extreme_heat": "Extreme Heat Index"
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

                if status == "accepted":
                    message = (
                        f"📡 {_pretty_server_name(server_name)} seems helpful here."
                    )
                else:
                    message = (
                        f"🚫 {_pretty_server_name(server_name)} does not seem relevant."
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

        # await emit("🧭 Confirming the relevance of servers...", "routing")
        await emit(
            "**Step 1: Selecting relevant datasets**\n\nDetermining which curated datasets are most relevant to your question. We are in the constant process of expanding the range and depth of datasets across regions and topics, ensuring comprehensive coverage and interoperability.",
            "routing"
        )
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
                    f"✅ {_pretty_server_name(server_name)} server shared {len(response.facts)} passages "
                    f"and {len(response.artifacts)} visuals"
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

        # await emit("📥 Gathering passages and data from servers...", "execution")
        await emit(
            "**Step 2: Retrieving source data**\n\nBringing together evidence from the selected datasets, verifying accuracy, and preparing to synthesise your answer.",
            "execution"
        )
        responses = await self._execute_with_planning(
            supports, context, progress_callback=executor_progress
        )
        self._populate_citation_urls(responses)
        if not responses:
            raise ContractValidationError("No server produced a response")

        await self._emit_fact_thinking_events(responses, emit)

        # await emit("🧠 Pulling everything together...", "synthesis")
        await emit(
            "**Step 3: Synthesising the answer**\n\nCombining verified evidence into a coherent, cited response.",
            "synthesis"
        )

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

        synthesis_done = asyncio.Event()

        async def _synthesis_heartbeat() -> None:
            try:
                while not synthesis_done.is_set():
                    await asyncio.sleep(15)
                    if synthesis_done.is_set():
                        break
                    await emit("⏳ Still synthesizing…", "synthesis")
            except asyncio.CancelledError:
                raise

        heartbeat_task: Optional[asyncio.Task] = None
        if progress_callback:
            heartbeat_task = asyncio.create_task(_synthesis_heartbeat())

        try:
            narrative_result = await self._narrative.generate(
                query, ordered_evidences, ordered_ids
            )
            print(
                f"[KGDEBUG] narrative.generate finished: paragraphs={len(narrative_result.paragraphs)}",
                flush=True,
            )
        finally:
            synthesis_done.set()
            if heartbeat_task:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task

        # narrative_result and ordered_evidences now available from synthesis block above

        governance_module: Optional[Dict[str, Any]] = None
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
            governance_module = await self._maybe_build_governance_summary(
                original_query=original_query,
                scope_level=scope_level,
                narrative_paragraphs=narrative_result.paragraphs,
                citation_registry=citation_registry,
                responses=responses,
                context=context,
                emit=emit,
            )
        else:
            print("[KGDEBUG] narrative synthesis returned no paragraphs; building summary table", flush=True)
            citation_registry = CitationRegistry()
            for response in responses:
                citation_registry.register(response.citations)
            summary_module = self._build_summary_module(responses, citation_registry)
        artifact_modules = self._build_artifact_modules(responses)
        citation_module = self._build_citation_module(citation_registry)

        modules = [summary_module]
        if governance_module:
            modules.append(governance_module)
        modules.extend(artifact_modules)
        modules.append(citation_module)

        modules = self._merge_map_modules(modules)

        kg_nodes, kg_edges, kg_urls = self._combine_kg(responses)
        has_kg_content = bool(kg_nodes and kg_edges)
        print(
            f"[KGDEBUG] _combine_kg -> nodes={len(kg_nodes)} edges={len(kg_edges)} urls={len(kg_urls)} has_content={has_kg_content}",
            flush=True,
        )

        metadata = self._build_metadata(original_query, modules, kg_available=has_kg_content)
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
            "query": original_query,  # Always return original query to user
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
            "✅ All set—here's what we found", "synthesis", event_type="thinking_complete"
        )

        return final_payload

    async def _build_out_of_scope_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Return a contextual redirect when the query is out of scope."""

        domain_lines = "\n".join(f"- {domain}" for domain in DOMAINS_IN_SCOPE)
        guidance_text = None

        if self._out_of_scope_responder and self._out_of_scope_responder.available:
            guidance_text = await self._out_of_scope_responder.craft_response(
                query,
                conversation_history,
                domain_lines,
            )

        if not guidance_text:
            guidance_text = (
                "That one's outside what I can cover, since I'm focused on our core climate datasets."
                " These include:\n"
                f"{domain_lines}\n\n"
                "Feel free to pivot back to any of those areas and I can go much deeper."
            )

        guidance_module = {
            "type": "text",
            "heading": "",
            "texts": [guidance_text],
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

    async def _build_near_scope_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Provide a brief answer for near-scope queries and nudge toward core domains."""

        domain_lines = "\n".join(f"- {domain}" for domain in DOMAINS_IN_SCOPE)
        bridge_text: Optional[str] = None

        if self._out_of_scope_responder and self._out_of_scope_responder.available:
            bridge_text = await self._out_of_scope_responder.craft_response(
                query,
                conversation_history,
                domain_lines,
                mode="bridge",
            )

        if not bridge_text:
            snippet = query.strip()
            if len(snippet) > 120:
                snippet = snippet[:117].rstrip() + "..."

            bridge_text = (
                f"I don't have detailed datasets on \"{snippet}\", but I can acknowledge it at a high level. "
                "For deep dives with evidence, I'm strongest on the domains listed below."
            )

        reminder_text = (
            "If you'd like to stay in the areas where I have primary evidence, try one of these focus domains:\n"
            f"{domain_lines}"
        )

        guidance_module = {
            "type": "text",
            "heading": "Quick Note",
            "texts": [bridge_text],
        }
        reminder_module = {
            "type": "text",
            "heading": "Where I Can Help Most",
            "texts": [reminder_text],
        }
        citation_module = {
            "type": "numbered_citation_table",
            "heading": "References",
            "columns": ["#", "Source", "ID/Tool", "Type", "Description", "SourceURL"],
            "rows": [],
            "allow_empty": True,
        }

        modules = [guidance_module, reminder_module, citation_module]
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

    # def _build_narrative_summary_module(
    #     self,
    #     paragraphs: List[str],
    #     sequence: List[str],
    #     evidence_map: Dict[str, NarrativeEvidence],
    #     registry: CitationRegistry,
    # ) -> Dict[str, Any]:
    #     pattern = re.compile(r"\[\[(F\d+)\]\]")

    #      # Fix grouped pattern like [[F1][F2][F7]]
    #     grouped_pattern = re.compile(r"\[\[(F\d+(?:\]\[F\d+)+)\]\]")

    #     def expand_grouped_refs(text: str) -> str:
    #         """
    #         Expands malformed grouped citations like [[F1][F2][F7]]
    #         into [[F1]][[F2]][[F7]].
    #         """
    #         def expand(match: re.Match[str]) -> str:
    #             inner = match.group(1)  # e.g. "F1][F2][F7"
    #             # Extract all F# inside it
    #             fids = re.findall(r"F\d+", inner)
    #             # Rebuild properly spaced [[F#]] sequence
    #             return "".join(f"[[{fid}]]" for fid in fids)
            
    #         return grouped_pattern.sub(expand, text)

    #     def replace_marker(match: re.Match[str]) -> str:
    #         fid = match.group(1)
    #         evidence = evidence_map.get(fid)
    #         if not evidence:
    #             return ""
    #         try:
    #             number = registry.number_for(evidence.citation)
    #         except KeyError:
    #             return ""
    #         return f"^{number}^"

    #     rendered: List[str] = []
    #     for paragraph in paragraphs:
    #         paragraph = expand_grouped_refs(paragraph)

    #         rendered.append(pattern.sub(replace_marker, paragraph))

    #     if not rendered:
    #         rendered = ["No supporting facts were returned."]

    #     return {
    #         "type": "text",
    #         "heading": "Summary",
    #         "texts": rendered,
    #         "metadata": {"citations": self._build_citation_metadata(sequence, evidence_map, registry)},
    #     }

    def _build_narrative_summary_module(
        self,
        paragraphs: List[str],
        sequence: List[str],
        evidence_map: Dict[str, NarrativeEvidence],
        registry: CitationRegistry,
    ) -> Dict[str, Any]:
        pattern = re.compile(r"\[\[(F\d+)\]\]")

        # Fix grouped pattern like [[F1][F2][F7]]
        grouped_pattern = re.compile(r"\[\[(F\d+(?:\]\[F\d+)+)\]\]")

        def expand_grouped_refs(text: str) -> str:
            """
            Expands malformed grouped citations like [[F1][F2][F7]]
            into [[F1]][[F2]][[F7]].
            """
            def expand(match: re.Match[str]) -> str:
                inner = match.group(1)  # e.g. "F1][F2][F7"
                # Extract all F# inside it
                fids = re.findall(r"F\d+", inner)
                # Rebuild properly spaced [[F#]] sequence
                return "".join(f"[[{fid}]]" for fid in fids)
            
            return grouped_pattern.sub(expand, text)

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
        
        def collapse_repeated_refs(text: str) -> str:
            """
            Collapses repeated citation markers like:
            ^1^^1^^1^  → ^1^
            ^2^ ^2^    → ^2^
            ^3^   ^3^  → ^3^
            Works for any number.
            """
            # (\^\d+\^) captures a ^number^ group
            # (?:\s*\1)+ matches one or more repeats of the same marker,
            # possibly separated by spaces
            pattern = re.compile(r'(\^\d+\^)(?:\s*\1)+(?!\s*\^)')
            return pattern.sub(r'\1', text)

        def reorder_citation_groups(text: str) -> str:
            """
            Reorders any run of ^n^ markers (with optional internal spaces) into ascending order,
            preserving spacing/punctuation and working at end-of-line too.
            """
            # Match ≥2 markers possibly separated by spaces, stop before space/punct/EOL
            pattern = re.compile(r'(?P<group>(?:\^\d+\^\s*){2,})(?=(?:\s|\W|$))')

            def sort_group(match: re.Match[str]) -> str:
                group = match.group("group")
                # Extract, deduplicate, sort
                numbers = sorted({int(n) for n in re.findall(r"\^(\d+)\^", group)})
                # Rebuild normalized group
                return "".join(f"^{n}^" for n in numbers)

            return pattern.sub(sort_group, text)

        rendered: List[str] = []
        for paragraph in paragraphs:
            print(paragraph)
            paragraph = expand_grouped_refs(paragraph)
            text = pattern.sub(replace_marker, paragraph)
            print(text)
            text = collapse_repeated_refs(text)
            print(text)
            text = reorder_citation_groups(text)
            rendered.append(text)

        if not rendered:
            rendered = ["No supporting facts were returned."]

        return {
            "type": "text",
            "heading": "Summary",
            "texts": rendered,
            "metadata": {"citations": self._build_citation_metadata(sequence, evidence_map, registry)},
        }

    def _populate_citation_urls(self, responses: List[RunQueryResponse]) -> None:
        """Fill missing citation URLs using the dataset resolver."""

        for response in responses:
            for citation in response.citations:
                if getattr(citation, "url", None):
                    continue
                metadata = citation.metadata if isinstance(citation.metadata, Mapping) else None
                _ds_id, url = resolve_dataset_url(
                    tool_name=citation.tool,
                    tool_metadata=metadata,
                    server_name=response.server,
                )
                if url:
                    citation.url = url

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

    async def _maybe_build_governance_summary(
        self,
        *,
        original_query: str,
        scope_level: str,
        narrative_paragraphs: List[str],
        citation_registry: CitationRegistry,
        responses: List[RunQueryResponse],
        context: QueryContext,
        emit: Optional[Callable[[str, str], Awaitable[None]]],
    ) -> Optional[Dict[str, Any]]:
        if not _should_inject_governance(scope_level, bool(narrative_paragraphs)):
            return None

        if emit is not None:
            try:
                await emit("Considering governance implications...", "synthesis")
            except Exception:  # pragma: no cover - thinking is best effort
                pass

        followup_response: Optional[RunQueryResponse] = None
        try:
            followup_response = await self._invoke_lse_followup(
                original_query=original_query,
                narrative_paragraphs=narrative_paragraphs,
                context=context,
            )
        except Exception as exc:  # pragma: no cover - logged for diagnostics
            logger.info(f"Governance follow-up run_query failed: {exc}")

        lse_responses: List[RunQueryResponse] = list(responses)
        if followup_response:
            self._populate_citation_urls([followup_response])
            lse_responses.append(followup_response)

        self._register_lse_citations(citation_registry, lse_responses)
        fact_lines = self._gather_lse_fact_lines(citation_registry, lse_responses)
        if not fact_lines:
            return None

        try:
            governance_text = await self._synthesize_governance_summary_text(
                original_query,
                narrative_paragraphs,
                fact_lines,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.info(f"Governance summary synthesis failed: {exc}")
            return None

        if not governance_text or not governance_text.strip():
            return None

        return {
            "type": "text",
            "heading": "Governance Implications",
            "texts": [governance_text.strip()],
        }

    async def _invoke_lse_followup(
        self,
        *,
        original_query: str,
        narrative_paragraphs: List[str],
        context: QueryContext,
    ) -> Optional[RunQueryResponse]:
        session = self._client.sessions.get("lse")
        if session is None:
            logger.info("LSE session unavailable; skipping governance follow-up")
            return None

        followup_query = _build_governance_followup_query(original_query, narrative_paragraphs)
        followup_context = context.model_copy(
            update={
                "query": followup_query,
                "timestamp": datetime.datetime.utcnow(),
            }
        )

        try:
            response = await asyncio.wait_for(
                self._call_run_query(session, "lse", followup_context),
                timeout=RUN_QUERY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            logger.info(f"Governance follow-up timed out: {exc}")
            return None
        except Exception as exc:
            logger.info(f"Governance follow-up errored: {exc}")
            return None

        return response

    @staticmethod
    def _register_lse_citations(
        citation_registry: CitationRegistry, responses: List[RunQueryResponse]
    ) -> None:
        for response in responses:
            if response.server != "lse":
                continue
            try:
                citation_registry.register(response.citations)
            except Exception:
                # Registry handles duplicates; continue on unexpected issues
                continue

    @staticmethod
    def _gather_lse_fact_lines(
        citation_registry: CitationRegistry, responses: List[RunQueryResponse]
    ) -> List[str]:
        lines: List[str] = []
        for response in responses:
            if response.server != "lse":
                continue
            lookup = {citation.id: citation for citation in response.citations}
            for fact in response.facts:
                citation = lookup.get(fact.citation_id)
                if not citation:
                    continue
                try:
                    number = citation_registry.number_for(citation)
                except KeyError:
                    continue
                text = (fact.text or "").strip()
                if not text:
                    continue
                normalised = re.sub(r"\s+", " ", text)
                lines.append(f"^{number}^ {normalised}")
                if len(lines) >= 16:
                    return lines
        return lines

    async def _synthesize_governance_summary_text(
        self,
        user_query: str,
        narrative_paragraphs: Sequence[str],
        fact_lines: Sequence[str],
    ) -> Optional[str]:
        if not fact_lines:
            return None

        narrative_block = "\n\n".join(
            paragraph.strip() for paragraph in narrative_paragraphs if paragraph and paragraph.strip()
        )
        evidence_block = "\n".join(str(line) for line in fact_lines if line)

        system_prompt = (
            "You are a climate governance analyst. Write a concise paragraph labelled Governance Summary"
            " that interprets LSE NDC Align evidence."
            " Use the provided citation markers like ^3^ immediately after supporting claims."
            " Do not fabricate citations or mention unavailable evidence."
        )
        user_prompt = (
            f"User question:\n{user_query.strip()}\n\n"
            "Narrative summary that precedes your paragraph:\n"
            f"{narrative_block or '(narrative summary unavailable)'}\n\n"
            "Evidence from LSE (each line begins with the citation number you must reuse):\n"
            f"{evidence_block}\n\n"
            "Write 2-4 sentences covering governance implications, institutions, or implementation."
            " Return only the paragraph text without headings or bullet lists."
        )

        if self._governance_openai_client is not None:
            def _run_openai() -> str:
                response = self._governance_openai_client.responses.create(  # type: ignore[union-attr]
                    model=GOVERNANCE_SUMMARY_OPENAI_MODEL,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_output_tokens=400,
                )
                return _extract_openai_text(response)

            return await asyncio.to_thread(_run_openai)

        if self._governance_anthropic_client is not None:
            def _run_anthropic() -> str:
                response = self._governance_anthropic_client.messages.create(  # type: ignore[union-attr]
                    model=GOVERNANCE_SUMMARY_ANTHROPIC_MODEL,
                    max_tokens=400,
                    temperature=0,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text

            return await asyncio.to_thread(_run_anthropic)

        logger.info("No LLM client configured for governance summary")
        return None

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
        map_indices = [
            idx
            for idx, module in enumerate(modules)
            if module.get("type") == "map" and module.get("mapType") == "geojson_url"
        ]
        if len(map_indices) <= 1:
            return modules

        project_root = Path(__file__).resolve().parents[1]
        static_maps_dir = project_root / "static" / "maps"

        groups: Dict[Any, List[int]] = {}
        for idx in map_indices:
            metadata = modules[idx].get("metadata") or {}
            merge_group = metadata.get("merge_group")
            if merge_group:
                groups.setdefault(merge_group, []).append(idx)

        consumed: Set[int] = set()
        result: List[Dict[str, Any]] = []

        for idx, module in enumerate(modules):
            if idx in consumed:
                continue

            if (
                module.get("type") == "map"
                and module.get("mapType") == "geojson_url"
            ):
                metadata = module.get("metadata") or {}
                merge_group = metadata.get("merge_group")
                group_indices = groups.get(merge_group, []) if merge_group else []
                if merge_group and len(group_indices) > 1:
                    candidate_modules = [modules[i] for i in group_indices if i not in consumed]
                    combined = self._combine_map_group(candidate_modules, project_root, static_maps_dir)
                    if combined:
                        result.append(combined)
                        consumed.update(group_indices)
                        continue

            result.append(module)
            consumed.add(idx)

        return result

    def _combine_map_group(
        self,
        map_modules: List[Mapping[str, Any]],
        project_root: Path,
        static_maps_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        if len(map_modules) <= 1:
            return None

        combined_features: List[Dict[str, Any]] = []
        legend_by_key: Dict[str, Dict[str, Any]] = {}
        combined_bounds: Optional[Dict[str, float]] = None
        total_feature_count = 0
        geometry_type = None

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
            combined_bounds = self._union_bounds(combined_bounds, metadata.get("bounds"))

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
            return None

        combined_filename = f"combined_map_{uuid.uuid4().hex[:10]}.geojson"
        static_maps_dir.mkdir(parents=True, exist_ok=True)
        combined_path = static_maps_dir / combined_filename
        try:
            combined_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": combined_features}),
                encoding="utf-8",
            )
        except Exception:
            return None

        if geometry_type is None:
            geometry_type = "polygon"

        combined_metadata: Dict[str, Any] = {
            "feature_count": total_feature_count,
            "legend": {
                "title": "Layers",
                "items": list(legend_by_key.values()),
            },
        }
        first_metadata = map_modules[0].get("metadata") or {}
        merge_group = first_metadata.get("merge_group")
        if merge_group:
            combined_metadata["merge_group"] = merge_group
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

        return combined_module

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
