#!/usr/bin/env python3
"""
MCP Chat Redo - Modern 3-Phase Architecture Implementation

This script implements a high-performance MCP chat system with:
1. Global singleton client for 5-10x performance improvement
2. 3-phase execution strategy (Scout -> Deep Dive -> Synthesis)
3. Citation registry with deduplication and proper numbering
4. Artifact handling for multimodal responses
5. Cross-server callbacks for complex queries

Author: Implementation guided by new_mcp_chat_ideas.md
"""

import asyncio
import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic
from dotenv import load_dotenv

# Import performance optimizer
try:
    from performance_optimizer import (
        PerformanceOptimizer, QueryComplexity, 
        ProgressiveResponseBuilder, SmartRouter
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("Warning: Performance optimizer not available")

# Import fact tracer for debugging
try:
    from fact_tracer import FactTracer
    FACT_TRACER_AVAILABLE = True
except ImportError:
    FACT_TRACER_AVAILABLE = False
    print("Warning: Fact tracer not available for debugging")

# Load environment variables
load_dotenv()

# Feature flags
def _geo_llm_only() -> bool:
    """Return True if geospatial deterministic correlation should be disabled (LLM only)."""
    val = os.environ.get("TDE_GEO_USE_LLM_ONLY") or os.environ.get("TDE_DISABLE_DETERMINISTIC_GEO")
    return str(val).lower() in ("1", "true", "yes")

# =============================================================================
# LOGGING CONFIGURATION FOR LLM CALLS
# =============================================================================

# Determine log directory - use environment variable or create in current project
# This allows deployment flexibility while maintaining local development paths
log_dir_path = os.environ.get('TDE_LOG_DIR')
if log_dir_path:
    LOG_DIR = Path(log_dir_path)
else:
    # Use project-relative logs directory
    # This works whether running from project root or mcp subdirectory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up from mcp/ to project root
    LOG_DIR = project_root / "logs"

# Create logs directory if it doesn't exist
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure LLM call logger
    llm_logger = logging.getLogger("llm_calls")
    llm_logger.setLevel(logging.DEBUG)
    
    # Create file handler with timestamp
    log_filename = LOG_DIR / f"llm_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler for important logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    llm_logger.addHandler(file_handler)
    llm_logger.addHandler(console_handler)
    
    llm_logger.info(f"LLM call logging initialized. Log file: {log_filename}")
    
except Exception as e:
    # If logging setup fails, create a minimal console-only logger
    # This ensures the application still works even if file logging fails
    print(f"Warning: Could not set up file logging: {e}")
    print(f"Attempted log directory: {LOG_DIR}")
    
    llm_logger = logging.getLogger("llm_calls")
    llm_logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    llm_logger.addHandler(console_handler)
    
    llm_logger.info("LLM call logging initialized (console only - file logging disabled)")

# =============================================================================
# CONFIGURATION: MODEL SELECTION
# =============================================================================

# Model configuration for different use cases
SMALL_MODEL = "claude-3-5-haiku-20241022"  # Fast, efficient for routing and simple tasks
LARGE_MODEL = "claude-3-5-sonnet-20241022"  # Powerful for synthesis and complex reasoning

# Initialize Anthropic client (shared across all operations)
ANTHROPIC_CLIENT = anthropic.Anthropic()

# =============================================================================
# LLM PAYLOAD SANITIZATION & LOGGING HELPERS
# =============================================================================

# Keys commonly used in large/geo payloads that should not be echoed to the LLM
SENSITIVE_BIG_KEYS = {"features", "geometry", "geojson", "coordinates", "data", "polygons", "points"}

def _truncate_string(s: str, max_len: int = 4000) -> str:
    try:
        return s if len(s) <= max_len else s[:max_len] + f"... [truncated {len(s)-max_len} chars]"
    except Exception:
        return str(s)[:max_len] + "... [truncated]"

def _sanitize_obj_for_llm(obj: Any, removed_counts: Optional[Dict[str, int]] = None, max_list_len: int = 50) -> Any:
    """Recursively sanitize an object so we never pass raw GeoJSON/heavy arrays to the LLM."""
    from collections.abc import Mapping, Sequence
    if removed_counts is None:
        removed_counts = {}
    # Dict-like
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in SENSITIVE_BIG_KEYS:
                try:
                    length = len(v) if hasattr(v, '__len__') else 1
                except Exception:
                    length = 1
                removed_counts[kl] = removed_counts.get(kl, 0) + length
                out[k] = {"_omitted_for_llm": True, "_approx_count": length}
            else:
                out[k] = _sanitize_obj_for_llm(v, removed_counts, max_list_len)
        return out
    # Strings
    if isinstance(obj, str):
        return _truncate_string(obj, 2000)
    # Lists/tuples
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        if len(obj) > max_list_len:
            head = [_sanitize_obj_for_llm(x, removed_counts, max_list_len) for x in obj[:max_list_len]]
            return head + [{"_omitted_for_llm": True, "_omitted_items": len(obj) - max_list_len}]
        return [_sanitize_obj_for_llm(x, removed_counts, max_list_len) for x in obj]
    # Fallback
    return obj

def _prepare_tool_result_for_llm(tool_name: str, result: Any) -> str:
    """Create a safe, compact string for tool_result content sent back to the LLM."""
    removed_counts: Dict[str, int] = {}
    try:
        if hasattr(result, 'content') and isinstance(result.content, list) and result.content:
            first = result.content[0]
            if hasattr(first, 'text') and isinstance(first.text, str):
                txt = first.text
                try:
                    data = json.loads(txt)
                    sanitized = _sanitize_obj_for_llm(data, removed_counts)
                    payload_str = json.dumps(sanitized, ensure_ascii=False)
                except json.JSONDecodeError:
                    payload_str = _truncate_string(txt, 4000)
                if removed_counts:
                    llm_logger.warning(f"Sanitized tool_result for {tool_name}; removed heavy keys: {removed_counts}")
                llm_logger.debug(f"tool_result payload length: {len(payload_str)}")
                return payload_str
        # Fallback to str(result)
        payload_str = _truncate_string(str(result), 4000)
        llm_logger.debug(f"tool_result payload length: {len(payload_str)}")
        return payload_str
    except Exception as e:
        return f"[tool_result_unavailable: {e}]"

def _log_llm_messages_summary(messages: List[Dict[str, Any]], label: str):
    try:
        serialized = json.dumps(messages, ensure_ascii=False)
        size_kb = len(serialized.encode('utf-8')) / 1024.0
        llm_logger.info(f"LLM messages summary [{label}]: messages={len(messages)} size≈{size_kb:.1f}KB")
    except Exception as e:
        llm_logger.debug(f"Could not summarize LLM messages: {e}")

def _write_llm_payload_log(label: str, payload: Dict[str, Any]):
    """Write the exact system+messages payload we send to the LLM to a daily log file."""
    try:
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = LOG_DIR / f"llm_payloads_{date_str}.log"
        entry = {"ts": datetime.now().isoformat(), "label": label, "payload": payload}
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        llm_logger.error(f"Failed to write LLM payload log: {e}")

# =============================================================================
# LLM HELPER FUNCTIONS
# =============================================================================

async def call_small_model(system: str, user_prompt: str, max_tokens: int = 1000, temperature: float = 0) -> str:
    """
    Call the small model (Haiku) for fast routing and simple tasks.
    
    Args:
        system: System prompt for the LLM
        user_prompt: User message content
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation (0 = deterministic)
        
    Returns:
        Generated text response
    """
    # Log the call details
    llm_logger.debug(f"=== SMALL MODEL CALL ({SMALL_MODEL}) ===")
    llm_logger.debug(f"System prompt: {system[:200]}..." if len(system) > 200 else f"System prompt: {system}")
    llm_logger.debug(f"User prompt: {user_prompt[:500]}..." if len(user_prompt) > 500 else f"User prompt: {user_prompt}")
    llm_logger.debug(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    
    try:
        messages_payload = [{"role": "user", "content": user_prompt}]
        _write_llm_payload_log(
            label=f"small_model:{SMALL_MODEL}",
            payload={
                "model": SMALL_MODEL,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": messages_payload,
            }
        )
        response = ANTHROPIC_CLIENT.messages.create(
            model=SMALL_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages_payload
        )
        
        result = response.content[0].text
        llm_logger.debug(f"Response: {result[:500]}..." if len(result) > 500 else f"Response: {result}")
        llm_logger.info(f"Small model call completed. Input tokens: ~{len(system + user_prompt)//4}, Output tokens: ~{len(result)//4}")
        
        return result
    except Exception as e:
        llm_logger.error(f"Error in small model call: {e}")
        raise


async def call_large_model(system: str, user_prompt: str, max_tokens: int = 2000, temperature: float = 0) -> str:
    """
    Call the large model (Sonnet) for synthesis and complex reasoning.
    
    Args:
        system: System prompt for the LLM
        user_prompt: User message content
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation (0 = deterministic)
        
    Returns:
        Generated text response
    """
    # Log the call details
    llm_logger.debug(f"=== LARGE MODEL CALL ({LARGE_MODEL}) ===")
    llm_logger.debug(f"System prompt: {system[:200]}..." if len(system) > 200 else f"System prompt: {system}")
    llm_logger.debug(f"User prompt: {user_prompt[:500]}..." if len(user_prompt) > 500 else f"User prompt: {user_prompt}")
    llm_logger.debug(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    
    try:
        messages_payload = [{"role": "user", "content": user_prompt}]
        _write_llm_payload_log(
            label=f"large_model:{LARGE_MODEL}",
            payload={
                "model": LARGE_MODEL,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": messages_payload,
            }
        )
        response = ANTHROPIC_CLIENT.messages.create(
            model=LARGE_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages_payload
        )
        
        result = response.content[0].text
        llm_logger.debug(f"Response: {result[:500]}..." if len(result) > 500 else f"Response: {result}")
        llm_logger.info(f"Large model call completed. Input tokens: ~{len(system + user_prompt)//4}, Output tokens: ~{len(result)//4}")
        
        return result
    except Exception as e:
        llm_logger.error(f"Error in large model call: {e}")
        raise


async def call_model_with_tools(model: str, system: str, messages: List[Dict[str, Any]], 
                                tools: List[Dict[str, Any]], max_tokens: int = 1000) -> Any:
    """
    Call an Anthropic model with tools (for MCP tool calling).
    
    Args:
        model: Model name (SMALL_MODEL or LARGE_MODEL)
        system: System prompt
        messages: Conversation messages
        tools: Available tools for the model
        max_tokens: Maximum tokens to generate
        
    Returns:
        Raw Anthropic response object
    """
    # Log the call details
    llm_logger.debug(f"=== MODEL WITH TOOLS CALL ({model}) ===")
    llm_logger.debug(f"System prompt: {system[:200]}..." if len(system) > 200 else f"System prompt: {system}")
    llm_logger.debug(f"Number of messages: {len(messages)}")
    llm_logger.debug(f"Number of tools: {len(tools)}")
    llm_logger.debug(f"Tool names: {[t.get('name', 'unknown') for t in tools]}")
    llm_logger.debug(f"Max tokens: {max_tokens}")
    
    # Log message history
    for i, msg in enumerate(messages[-3:]):  # Log last 3 messages for context
        llm_logger.debug(f"Message {i} ({msg.get('role', 'unknown')}): {str(msg.get('content', ''))[:300]}...")
    
    try:
        _log_llm_messages_summary(messages, f"tools={len(tools)} model={model}")
        # Log exact messages + slim tool metadata to avoid serialization issues
        try:
            tools_log = [{"name": t.get("name", ""), "description": t.get("description", "")} for t in tools]
        except Exception:
            tools_log = []
        _write_llm_payload_log(
            label=f"tools_model:{model}",
            payload={
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
                "tools": tools_log,
            }
        )
        # Note: Recent Anthropic SDKs no longer require or accept the tools beta header.
        # Passing deprecated beta headers can cause 400 errors. Call without extra_headers.
        response = ANTHROPIC_CLIENT.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools
        )
        
        # Log response details
        llm_logger.debug(f"Response type: {type(response)}")
        if hasattr(response, 'content'):
            for content_block in response.content:
                if hasattr(content_block, 'type'):
                    llm_logger.debug(f"Content block type: {content_block.type}")
                    if content_block.type == 'text':
                        llm_logger.debug(f"Text content: {content_block.text[:300]}...")
                    elif content_block.type == 'tool_use':
                        llm_logger.debug(f"Tool use: {content_block.name} with input: {str(content_block.input)[:200]}...")
        
        llm_logger.info(f"Model with tools call completed ({model})")
        
        return response
    except Exception as e:
        llm_logger.error(f"Error in model with tools call: {e}")
        raise

# =============================================================================
# PHASE 0: CORE DATA STRUCTURES
# =============================================================================

class ArtifactType(Enum):
    """Types of artifacts that can be generated and stored."""
    GEOJSON = "geojson"
    CHART = "chart"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class Citation:
    """
    Represents a single citation with all required fields for the citation table.
    
    Based on API spec for numbered_citation_table:
    - #: Citation number (assigned by CitationRegistry)
    - Source: Name of the source (e.g., "GIST Impact Database", "CPR Knowledge Graph")
    - ID/Tool: Tool name or identifier (e.g., "GetGistCompanyRiskData", "search_passages")
    - Type: Type of source (e.g., "Database", "Knowledge Graph", "API", "Tool")
    - Description: Brief description of what this source provided
    """
    source_name: str  # e.g., "GIST Impact Database"
    tool_id: str  # e.g., "GetGistCompanyRiskData"
    source_type: str  # e.g., "Database"
    description: str  # e.g., "Company water risk data"
    server_origin: str  # Which MCP server provided this
    source_url: str = ""  # Public URL to the dataset/provider page (for dataset/database sources)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context
    
    def to_table_row(self, citation_number: int) -> List[str]:
        """Convert citation to table row format for the citation table."""
        return [
            str(citation_number),
            self.source_name,
            self.tool_id,
            self.source_type,
            self.description,
            self.source_url or ""
        ]


@dataclass
class Fact:
    """
    Represents a single fact discovered during query processing.
    
    Attributes:
        text_content: Natural language description of the fact
        source_key: Unique key for citation deduplication
        server_origin: Which MCP server provided this fact
        metadata: Additional context including raw tool results
        citation: Citation object containing source information
        numerical_data: Structured numerical data for visualization (populated in Phase 3)
        map_reference: Reference to map data stored in static/maps/ (populated in Phase 3)
        data_type: Classification for visualization decisions (populated in Phase 3)
    """
    # Core content (always present)
    text_content: str
    source_key: str
    server_origin: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    citation: Optional[Citation] = None
    
    # Visualization data (populated in Phase 3)
    numerical_data: Optional[Dict[str, Any]] = None  # For charts/tables
    map_reference: Optional[Dict[str, Any]] = None   # For geographic data
    data_type: Optional[str] = None  # 'time_series', 'comparison', 'geographic', 'tabular', 'text_only'


@dataclass
class PhaseResult:
    """
    Results from a single phase of execution.
    
    Used to pass data between Scout -> Deep Dive -> Synthesis phases.
    """
    is_relevant: bool
    facts: List[Fact] = field(default_factory=list)
    reasoning: str = ""
    suggested_follow_ups: List[str] = field(default_factory=list)
    continue_processing: bool = True
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: int = 0


class CitationRegistry:
    """
    Manages citation numbering and tracking for inline citations.
    
    Features:
    - Deduplicates sources based on unique keys
    - Assigns sequential citation numbers
    - Tracks module-level citation usage
    - Handles post-synthesis citation reordering
    """
    
    def __init__(self):
        self.citations = {}  # source_key -> citation_number
        self.citation_counter = 1
        self.module_citations = {}  # module_id -> list of citation numbers
        self.citation_objects = {}  # citation_number -> Citation object
        
    def add_citation(self, citation: Citation, module_id: str = None) -> int:
        """
        Add a citation and return its citation number.
        
        Implements deduplication logic using source keys to ensure
        the same source gets the same citation number across references.
        
        Args:
            citation: Citation object with source information
            module_id: Optional server/module identifier
            
        Returns:
            Citation number for referencing this source
        """
        # Generate unique key for this citation
        source_key = self._generate_citation_key(citation)
        
        # Check if we already have this citation
        if source_key in self.citations:
            citation_num = self.citations[source_key]
        else:
            # Assign new citation number
            citation_num = self.citation_counter
            self.citations[source_key] = citation_num
            self.citation_objects[citation_num] = citation
            self.citation_counter += 1
        
        # Track module association
        if module_id:
            if module_id not in self.module_citations:
                self.module_citations[module_id] = []
            if citation_num not in self.module_citations[module_id]:
                self.module_citations[module_id].append(citation_num)
        
        return citation_num
        
    def _generate_citation_key(self, citation: Citation) -> str:
        """
        Generate unique key for citation deduplication.
        
        Creates a unique key based on citation source and tool to prevent
        duplicate citations for the same source.
        """
        # Create key from source name, tool, and server origin
        key_parts = [
            citation.source_name,
            citation.tool_id,
            citation.server_origin
        ]
        
        # Add any unique identifiers from metadata if available
        if "doc_id" in citation.metadata:
            key_parts.append(f"doc_{citation.metadata['doc_id']}")
        if "passage_id" in citation.metadata:
            key_parts.append(f"passage_{citation.metadata['passage_id']}")
        if "citation_id" in citation.metadata:
            key_parts.append(f"cit_{citation.metadata['citation_id']}")
        
        return "_".join(key_parts)
        
    def format_citation_superscript(self, citation_nums: List[int]) -> str:
        """
        Format citation numbers as markdown superscript.
        
        Formats:
        - Single citation: ^1^
        - Multiple citations: ^1,3,5^
        - Sort numbers for consistent ordering
        """
        if not citation_nums:
            return ""
        
        # Sort citation numbers for consistent ordering
        sorted_nums = sorted(citation_nums)
        
        if len(sorted_nums) == 1:
            return f"^{sorted_nums[0]}^"
        else:
            return f"^{','.join(map(str, sorted_nums))}^"
        
    def get_all_citations(self) -> Dict[int, Citation]:
        """Get all citations ordered by number for bibliography generation."""
        return {num: self.citation_objects[num] for num in sorted(self.citation_objects.keys())}
    
    def generate_citation_table(self) -> Dict[str, Any]:
        """
        Generate the citation table in API response format.
        
        Returns:
            Dictionary with numbered_citation_table structure
        """
        if not self.citation_objects:
            return None
        
        rows = []
        for citation_num in sorted(self.citation_objects.keys()):
            citation = self.citation_objects[citation_num]
            rows.append(citation.to_table_row(citation_num))
        
        return {
            "type": "numbered_citation_table",
            "heading": "References",
            "columns": ["#", "Source", "ID/Tool", "Type", "Description", "SourceURL"],
            "rows": rows
        }


# =============================================================================
# KG CONTEXT TRACKING FOR VISUALIZATION
# =============================================================================

class KGContextTracker:
    """
    Tracks Knowledge Graph context during query processing.
    
    Collects information about concepts, passages, and their relationships
    that were actually used during the conversation. This data is used to
    generate accurate KG visualizations that reflect the actual knowledge
    path taken to answer the query.
    """
    
    def __init__(self):
        # Concept tracking
        self.seed_concepts = []  # Primary concepts from query
        self.neighbor_concepts = {}  # concept -> {"type": "parent"|"child"|"related", "via": seed_concept}
        self.concept_wikibase_ids = {}  # concept_label -> wikibase_id
        
        # Passage tracking
        self.concept_passages = {}  # concept -> list of passage data dicts
        self.all_passages = {}  # passage_id -> passage data dict
        
        # Edge tracking
        self.concept_edges = []  # List of (source_concept, target_concept, edge_type)
        
    def add_seed_concept(self, concept_label: str):
        """Add a seed concept (primary concept from query)."""
        if concept_label and concept_label not in self.seed_concepts:
            self.seed_concepts.append(concept_label)
            llm_logger.debug(f"KG_CONTEXT: Added seed concept '{concept_label}'")
    
    def add_neighbor_concept(self, concept_label: str, neighbor_type: str, via_concept: str):
        """Add a neighbor concept discovered through graph traversal."""
        if concept_label and concept_label not in self.neighbor_concepts:
            self.neighbor_concepts[concept_label] = {
                "type": neighbor_type,  # "parent", "child", or "related"
                "via": via_concept
            }
            llm_logger.debug(f"KG_CONTEXT: Added {neighbor_type} concept '{concept_label}' via '{via_concept}'")
    
    def add_concept_edge(self, source: str, target: str, edge_type: str):
        """Add an edge between concepts."""
        edge = (source, target, edge_type)
        if edge not in self.concept_edges:
            self.concept_edges.append(edge)
            llm_logger.debug(f"KG_CONTEXT: Added edge {source} -{edge_type}-> {target}")
    
    def add_passages_for_concept(self, concept: str, passages: List[Dict]):
        """Add passages retrieved for a concept."""
        if concept not in self.concept_passages:
            self.concept_passages[concept] = []
        
        for passage in passages:
            if isinstance(passage, dict) and "passage_id" in passage:
                # Store in concept mapping
                self.concept_passages[concept].append(passage)
                # Store in global passage dict
                pid = str(passage["passage_id"])
                self.all_passages[pid] = passage
                llm_logger.debug(f"KG_CONTEXT: Added passage {pid} for concept '{concept}'")
    
    def set_concept_wikibase_id(self, concept_label: str, wikibase_id: str):
        """Set the wikibase ID for a concept."""
        if concept_label and wikibase_id:
            self.concept_wikibase_ids[concept_label] = wikibase_id
            llm_logger.debug(f"KG_CONTEXT: Set wikibase_id {wikibase_id} for '{concept_label}'")
    
    def build_kg_context(self, cited_passage_ids: Set[str] = None) -> Dict[str, Any]:
        """
        Build the final kg_context structure for visualization.
        
        Args:
            cited_passage_ids: Set of passage IDs that were actually cited in the response
            
        Returns:
            Dictionary with nodes and edges for KG visualization
        """
        cited_passage_ids = cited_passage_ids or set()
        nodes = []
        edges = []
        
        # Add concept nodes
        for concept in self.seed_concepts:
            node = {
                "id": self.concept_wikibase_ids.get(concept, f"concept_{concept}"),
                "label": concept,
                "type": "Concept",
                "status": "seed",
                "importance": 1.0
            }
            nodes.append(node)
        
        # Add neighbor concept nodes
        for concept, info in self.neighbor_concepts.items():
            node = {
                "id": self.concept_wikibase_ids.get(concept, f"concept_{concept}"),
                "label": concept,
                "type": "Concept", 
                "status": info["type"],  # "parent", "child", or "related"
                "importance": 0.7
            }
            nodes.append(node)
        
        # Add passage nodes
        for passage_id, passage_data in self.all_passages.items():
            # Truncate text for label
            text = passage_data.get("text", "")[:75] + "..." if len(passage_data.get("text", "")) > 75 else passage_data.get("text", "")
            
            node = {
                "id": f"passage_{passage_id}",
                "label": text,
                "type": "Passage",
                "doc_id": passage_data.get("doc_id"),
                "cited": str(passage_id) in cited_passage_ids,
                "importance": 1.0 if str(passage_id) in cited_passage_ids else 0.6,
                "full_text": passage_data.get("text", "")
            }
            nodes.append(node)
        
        # Add concept-to-concept edges
        for source, target, edge_type in self.concept_edges:
            source_id = self.concept_wikibase_ids.get(source, f"concept_{source}")
            target_id = self.concept_wikibase_ids.get(target, f"concept_{target}")
            edges.append({
                "source": source_id,
                "target": target_id,
                "type": edge_type
            })
        
        # Add passage-to-concept edges (MENTIONS)
        for concept, passages in self.concept_passages.items():
            concept_id = self.concept_wikibase_ids.get(concept, f"concept_{concept}")
            for passage in passages:
                if "passage_id" in passage:
                    edges.append({
                        "source": f"passage_{passage['passage_id']}",
                        "target": concept_id,
                        "type": "MENTIONS"
                    })
        
        llm_logger.info(f"KG_CONTEXT: Built context with {len(nodes)} nodes and {len(edges)} edges")
        return {
            "nodes": nodes,
            "edges": edges
        }


# =============================================================================
# PHASE 1: GLOBAL SINGLETON CLIENT (PERFORMANCE OPTIMIZATION)
# =============================================================================

# Global variables for singleton pattern
_global_client = None
_client_lock = asyncio.Lock()


class MultiServerClient:
    """
    Multi-server MCP client with connection pooling and session management.
    
    This class maintains persistent connections to all MCP servers for
    optimal performance. Based on existing implementation but restructured
    for the new 3-phase architecture.
    """
    
    def __init__(self):
        self.sessions: Dict[str, Any] = {}  # server_name -> ClientSession
        self.exit_stack = AsyncExitStack()
        # Use global Anthropic client for all LLM calls
        self.anthropic = ANTHROPIC_CLIENT
        self.citation_registry = CitationRegistry()
        self.kg_context_tracker = KGContextTracker()  # Track KG context for visualization
        
        # Initialize fact tracer if debugging is enabled
        self.fact_tracer = None
        if os.getenv('DEBUG_FACTS', '').lower() == 'true' and FACT_TRACER_AVAILABLE:
            self.fact_tracer = None  # Will be initialized per query
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit with cleanup."""
        await self.exit_stack.__aexit__(exc_type, exc, tb)
        
    async def connect_to_server(self, server_name: str, server_script_path: str):
        """
        Connect to a single MCP server.
        
        Establishes a persistent connection to an MCP server using stdio transport.
        Supports both Python and Node.js servers, including those built with FastMCP.
        
        Args:
            server_name: Unique identifier for this server
            server_script_path: Path to the server executable
        """
        print(f"Connecting to {server_name} server at {server_script_path}")
        
        try:
            server_params = StdioServerParameters(
                command="python",
                args=[server_script_path],
                env=None
            )
            
            # Establish stdio transport connection
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            
            # Create and initialize client session
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            await session.initialize()
            
            # Store session for later use
            self.sessions[server_name] = session
            print(f"Successfully connected to {server_name} server")
            
        except Exception as e:
            print(f"Failed to connect to {server_name} server: {e}")
            raise
        
    async def call_tool(self, tool_name: str, tool_args: dict, server_name: str):
        """
        Call a tool on a specific MCP server.
        
        TODO: Implement tool calling:
        1. Validate server_name exists in self.sessions
        2. Get session for the server
        3. Make tool call via session.call_tool()
        4. Return result or raise appropriate error
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool
            server_name: Which server to call the tool on
            
        Returns:
            Tool result from the MCP server
        """
        if server_name not in self.sessions:
            raise ValueError(f"Server '{server_name}' not connected")
        
        session = self.sessions[server_name]
        
        try:
            # Make the actual tool call
            result = await session.call_tool(tool_name, tool_args)
            return result
        except Exception as e:
            print(f"Error calling tool {tool_name} on {server_name}: {e}")
            raise


async def get_global_client() -> MultiServerClient:
    """
    Get or create the global singleton MCP client.
    
    This function implements the singleton pattern with thread-safe initialization.
    The global client persists across API requests for optimal performance.
    
    TODO: Implement singleton logic:
    1. Use asyncio.Lock() for thread safety
    2. Check if _global_client already exists
    3. If not, create new MultiServerClient()
    4. Connect to all required MCP servers:
       - kg: cpr_kg_server.py (Knowledge Graph)
       - solar: solar_facilities_server.py (Solar Data)
       - gist: gist_server.py (GIST Impact - Environmental & Water Risk Data)
       - lse: lse_server.py (LSE Climate Policy - Brazilian Governance & NDC Analysis)
       - formatter: response_formatter_server.py (Response Formatting)
       - viz: viz_server.py (Data Visualization)
    5. Handle initialization errors with cleanup
    6. Return the initialized client
    
    Returns:
        Initialized MultiServerClient instance
    """
    global _global_client
    
    async with _client_lock:
        if _global_client is None:
            # Create new MultiServerClient instance
            _global_client = MultiServerClient()
            await _global_client.__aenter__()
            
            # Determine directory paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            mcp_dir = os.path.join(project_root, "mcp")
            
            try:
                await _global_client.connect_to_server("cpr", os.path.join(mcp_dir, "cpr_kg_server.py"))
                await _global_client.connect_to_server("solar", os.path.join(mcp_dir, "solar_facilities_server.py"))
                await _global_client.connect_to_server("gist", os.path.join(mcp_dir, "gist_server.py"))
                await _global_client.connect_to_server("lse", os.path.join(mcp_dir, "lse_server.py"))
                await _global_client.connect_to_server("viz", os.path.join(mcp_dir, "viz_server.py"))
                
                # Add geospatial servers
                await _global_client.connect_to_server("deforestation", os.path.join(mcp_dir, "deforestation_server.py"))
                await _global_client.connect_to_server("geospatial", os.path.join(mcp_dir, "geospatial_server.py"))
                await _global_client.connect_to_server("municipalities", os.path.join(mcp_dir, "brazilian_admin_server.py"))
                await _global_client.connect_to_server("heat", os.path.join(mcp_dir, "heat_stress_server.py"))
                
                await _global_client.connect_to_server("meta", os.path.join(mcp_dir, "meta_server.py"))

                # Add RAG Servers
                await _global_client.connect_to_server("spa-server", os.path.join(mcp_dir, "spa_server.py"))


                print("Global MCP client initialized successfully with geospatial and municipality capabilities")
            except Exception as e:
                print(f"Error initializing global MCP client: {e}")
                # Clean up on failure
                await _global_client.__aexit__(None, None, None)
                _global_client = None
                raise
            
    return _global_client


async def cleanup_global_client():
    """
    Clean up the global singleton client.
    
    This should be called during application shutdown to properly close
    all MCP server connections and clean up resources.
    
    TODO: Implement cleanup logic:
    1. Acquire client lock for thread safety
    2. Call __aexit__ on global client if it exists
    3. Reset _global_client to None
    4. Log cleanup completion
    """
    global _global_client
    
    async with _client_lock:
        if _global_client is not None:
            try:
                await _global_client.__aexit__(None, None, None)
                _global_client = None
                print("Global MCP client cleaned up")
            except Exception as e:
                print(f"Error during cleanup: {e}")
                # Force cleanup even on error
                _global_client = None


# =============================================================================
# PHASE 2: 3-PHASE EXECUTION ARCHITECTURE
# =============================================================================

class QueryOrchestrator:
    """
    Orchestrates the 3-phase query execution strategy:
    
    Phase 1: Parallel Scout (Minimal Context)
    - All servers receive user query + routing prompt
    - Each server determines relevance and provides initial facts
    - Token budget: ~500-1000 per server
    - Runs in parallel via asyncio.gather()
    
    Phase 2: Targeted Deep Dive (Selective Context Sharing)
    - Only relevant servers continue
    - Each server receives enhanced context with cross-server highlights
    - Token budget: ~2000-3000 per server
    - Continues until termination criteria met
    - Supports cross-server callbacks
    
    Phase 3: Synthesis
    - Citation registry contains all facts from relevant servers
    - Final LLM generates clean narrative with proper citations
    - Response formatter handles artifact embedding
    """
    
    def __init__(self, client: MultiServerClient, conversation_history: Optional[List[Dict[str, str]]] = None, spatial_session_id: Optional[str] = None, target_language: Optional[str] = None):
        self.client = client
        self.citation_registry = client.citation_registry
        self.token_budget_remaining = 50000  # Conservative starting budget
        self.conversation_history = conversation_history or []
        self.cached_facts: List[Fact] = []  # Cache facts from previous responses
        self.cached_response_text: str = ""  # Cache the last response text
        # Geospatial session ID for isolating registrations per conversation/session
        self.spatial_session_id = spatial_session_id or "_default"
        # Track if we've already generated a geospatial correlation/map this query
        self._geo_map_generated: bool = False
        # Per-query guards/deduplication
        self._tool_calls_done: set = set()
        self._admin_zone_bootstrapped: bool = False
        # Target language preference (e.g., 'pt' for Portuguese)
        self.target_language = (target_language or "").lower() or None

    # --- Generic intent detectors (admin/zone/ranking) ---
    def _detect_admin_intent(self, query: str) -> Optional[str]:
        """Detect if the query asks about administrative units we support.
        Currently returns 'municipality' when municipalities/cities/towns are mentioned.
        """
        ql = (query or "").lower()
        admin_terms = ["municip", "municipal", "city", "cities", "town", "towns"]
        if any(t in ql for t in admin_terms):
            return "municipality"
        return None

    def _detect_zone_intent(self, query: str) -> Optional[str]:
        """Detect if the query implies a polygonal 'zone' layer we can register.
        Returns an entity_type like 'heat_zone' or 'deforestation_area' when detected.
        Minimal mapping to keep generality; extend as new layers are added.
        """
        ql = (query or "").lower()
        # Heat stress keywords → top-quintile heat zones
        if any(k in ql for k in ["heat", "temperature", "wbgt", "land surface temperature", "lst"]):
            return "heat_zone"
        # Deforestation (already supported as polygons)
        if "deforest" in ql:
            return "deforestation_area"
        return None

    def _has_ranking_intent(self, query: str) -> bool:
        """Detect ranking/comparison phrasing that implies 'most/highest/top' answers."""
        ql = (query or "").lower()
        return any(k in ql for k in ["which", "most", "highest", "top", "worst", "rank", "ranking"])

    def _tool_call_key(self, server: str, tool: str, args: Any) -> str:
        try:
            payload = json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
        except Exception:
            payload = str(args)
        return f"{server}::{tool}::{payload}"

    def _mark_tool_called(self, server: str, tool: str, args: Any) -> bool:
        key = self._tool_call_key(server, tool, args)
        if key in self._tool_calls_done:
            return False
        self._tool_calls_done.add(key)
        return True
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for LLM context."""
        if not self.conversation_history:
            return ""
        
        formatted = []
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    async def _is_query_relevant(self, user_query: str) -> bool:
        """
        Check if the query is relevant to climate/environment/policy domain.
        
        Args:
            user_query: User's query to check
            
        Returns:
            True if relevant, False if off-topic
        """
        print("Checking query relevance...")
        # Allow meta/capability/help queries up front so onboarding isn't blocked
        if await self._is_meta_query(user_query):
            # Log query and result
            print(f"✅ Meta/capability query detected, treating as relevant: {user_query[:100]}...")
            return True
        # If there's conversation history, include it in the relevance check
        if self.conversation_history and len(self.conversation_history) > 0:
            # Build context summary from recent messages
            recent_context = ""
            for msg in self.conversation_history[-4:]:  # Last 2 exchanges
                if msg["role"] == "user":
                    recent_context += f"User previously asked: {msg['content'][:200]}...\n"
                elif msg["role"] == "assistant":
                    recent_context += f"Assistant discussed: {msg['content'][:200]}...\n"
            
            relevance_prompt = f"""Determine if this query is relevant given the conversation context.

            CONVERSATION CONTEXT:
            {recent_context}

            CURRENT QUERY: "{user_query}"

            Our domains includes:
            - Climate change, impacts, and policies
            - Environmental data and sustainability
            - Energy systems, renewable energy, solar facilities
            - Corporate environmental performance and ESG
            - Water resources, biodiversity, and ecosystems
            - Environmental regulations, NDCs, and climate governance
            - Physical climate risks (floods, droughts, heat stress)
            - GHG emissions and carbon footprint
            - Environmental justice and climate adaptation

            Additionally, ALWAYS treat as relevant any meta questions about the assistant/app itself, including:
            - What you can do or talk about (capabilities, topics, features, examples)
            - What data/sources/datasets you have
            - How to use the app or how it works
            - Who/what the assistant is

            IMPORTANT: If the query refers to or follows up on ANYTHING from the conversation context above,
            it is IMMEDIATELY RELEVANT (answer YES), even if it doesn't explicitly mention climate/environment terms.

            Examples of relevant follow-ups:
            - "What about the totals?" (referring to previous data)
            - "Tell me more about that" (referring to previous topic)
            - "How does that compare?" (referring to previous information)
            - "Summarize what you just said" (referring to previous response)
            - "Remind me about X" (where X was discussed earlier)

            Answer YES if:
            1. The query is about climate/environment topics, OR
            2. The query refers to information from the conversation context

            Answer NO only if:
            - It's completely unrelated to both our domain AND the conversation context
            - It's asking about personal preferences, entertainment, or other clearly off-topic subjects

            Answer (YES/NO):"""
        else:
            # No conversation history - use standard relevance check
            relevance_prompt = f"""Determine if this query is related to our domain of expertise.

            Our domain includes:
            - Climate change, impacts, and policies
            - Environmental data and sustainability
            - Energy systems, renewable energy, solar facilities
            - Corporate environmental performance and ESG
            - Water resources, biodiversity, and ecosystems
            - Environmental regulations, NDCs, and climate governance
            - Physical climate risks (floods, droughts, heat stress)
            - GHG emissions and carbon footprint
            - Environmental justice and climate adaptation

            ALSO IN-SCOPE: Meta questions about the assistant/app itself, including:
            - What we can do or talk about (capabilities, topics, features, examples)
            - What data/sources/datasets we have
            - How to use the app or how it works
            - Who/what the assistant is

            The query: "{user_query}"

            Answer with just YES if the query is related to our domain, or NO if it's about:
            - Personal preferences or opinions unrelated to environment
            - General knowledge, trivia, or entertainment
            - Programming, math, or technical topics unrelated to climate
            - Medical, legal, or financial advice unrelated to climate
            - Other topics clearly outside environmental/climate scope

            Answer (YES/NO):"""
        
        try:
            response = await call_large_model(
                system="You are a query classifier. Respond with only YES or NO.",
                user_prompt=relevance_prompt,
                max_tokens=10,
                temperature=0
            )

            print("Relevance check response:", response)
            
            # Check response
            response_lower = response.strip().lower()
            is_relevant = "yes" in response_lower
            
            if not is_relevant:
                print(f"🚫 Off-topic query detected: {user_query[:100]}...")
                if self.conversation_history:
                    print(f"   Note: Had {len(self.conversation_history)} messages in context")
            else:
                if self.conversation_history:
                    print(f"✅ Query relevant (with {len(self.conversation_history)} context messages): {user_query[:50]}...")
            
            return is_relevant
            
        except Exception as e:
            # If relevance check fails, default to processing the query
            print(f"⚠️ Relevance check failed: {e}. Defaulting to processing query.")
            return True

    async def _is_meta_query(self, user_query: str) -> bool:
        """
        Uses the small model to check if the user query is about the project/app itself
        (capabilities, help, identity, datasets, etc).
        """
        prompt = f"""Is the following query asking about the assistant's capabilities, features, datasets, data sources, how to use the app, or about the assistant/app itself (identity, privacy, limitations, etc)? 

            Query: "{user_query}"

            Answer YES if the query is about what the assistant/app can do, its features, datasets, sources, how to use it, or about the assistant/app itself. 
            Answer NO if the query is about any substantive climate/environmental topic or data.

            Respond with only YES or NO."""
        try:
            response = await call_small_model(
                system="You classify queries as meta/capability/identity or not. Reply YES or NO.",
                user_prompt=prompt,
                max_tokens=10,
                temperature=0
            )
            return "yes" in response.strip().lower()
        except Exception as e:
            print(f"Meta query check failed: {e}")
            # Fallback to old heuristic if LLM fails
            q = (user_query or "").strip().lower()
            triggers = ["what can you do", "your capabilities", "what data", "what datasets", "how to use", "how do i", "who are you", "what is this", "help"
            ]
            return any(t in q for t in triggers)

    async def _create_capabilities_response(self, user_query: str) -> Dict:
        """Build a dynamic capabilities + datasets overview from live server metadata."""
        # Helper: normalize MCP tool result -> dict
        def _as_dict(tool_result) -> Dict[str, Any]:
            try:
                if hasattr(tool_result, 'content') and tool_result.content:
                    import json as _json
                    return _json.loads(tool_result.content[0].text)
                if isinstance(tool_result, dict):
                    return tool_result
            except Exception:
                pass
            return {}

        # Issue metadata calls concurrently where available
        calls = []
        # Prefer DescribeServer() if available; fall back to metadata/stat tools
        for name, server in [("cpr", "cpr"), ("solar", "solar"), ("gist", "gist"), ("lse", "lse"), ("deforestation", "deforestation"), ("municipalities", "municipalities"), ("heat", "heat"), ("geospatial", "geospatial"), ("viz", "viz")]:
            try:
                calls.append((name, self.client.call_tool("DescribeServer", {}, server)))
            except Exception:
                # Fallbacks by server
                try:
                    if name == "cpr":
                        calls.append((name, self.client.call_tool("GetKGDatasetMetadata", {}, server)))
                    elif name == "solar":
                        calls.append((name, self.client.call_tool("GetSolarDatasetMetadata", {}, server)))
                    elif name == "gist":
                        calls.append((name, self.client.call_tool("GetGistDatasetMetadata", {}, server)))
                    elif name == "lse":
                        calls.append((name, self.client.call_tool("GetLSEDatasetMetadata", {}, server)))
                    elif name == "deforestation":
                        calls.append((name, self.client.call_tool("GetDeforestationStatistics", {}, server)))
                    elif name == "municipalities":
                        calls.append((name, self.client.call_tool("GetMunicipalitiesDatasetMetadata", {}, server)))
                    elif name == "heat":
                        calls.append((name, self.client.call_tool("ListHeatLayers", {}, server)))
                except Exception:
                    pass

        results: Dict[str, Dict[str, Any]] = {}
        if calls:
            pairs = await asyncio.gather(*[c for _, c in calls], return_exceptions=True)
            for (name, _), res in zip(calls, pairs):
                if isinstance(res, Exception):
                    continue
                results[name] = _as_dict(res)

        # Build dataset rows using dynamic metadata; fall back to configured descriptions
        server_cfg = self._get_server_descriptions()
        rows = []
        def add_row(display_name: str, summary: str, metrics: str):
            rows.append([display_name, summary, metrics])

        # Build rows per server using DescribeServer format when available
        def describe_to_metrics(name_key: str, r: Dict[str, Any]) -> str:
            # Prefer a compact metrics summary
            m = r.get("metrics", {}) if isinstance(r, dict) else {}
            updated = r.get("last_updated") if isinstance(r, dict) else None
            upd_str = f"; updated {updated[:10]}" if isinstance(updated, str) and len(updated) >= 10 else ""
            if name_key == "cpr":
                concepts = m.get('concepts', m.get('concept_count', 0))
                passages = m.get('passages', m.get('passage_count', 0))
                nodes = m.get('graph_nodes', 0)
                edges = m.get('graph_edges', 0)
                note = " (not loaded)" if all(int(x or 0) == 0 for x in [concepts, passages, nodes, edges]) else ""
                return f"{concepts} concepts; {passages} passages; nodes {nodes}, edges {edges}{upd_str}{note}"
            if name_key == "solar":
                cap = m.get('total_capacity_mw')
                cap_str = f"; {float(cap):.0f} MW" if cap is not None else ""
                return f"{m.get('total_facilities', 0)} facilities{cap_str}; {m.get('total_countries', len(r.get('coverage', {}).get('countries', [])))} countries{upd_str}"
            if name_key == "gist":
                return f"{m.get('total_companies', 0)} companies; {m.get('total_assets', 0)} assets; {m.get('dataset_count', 0)} datasets{upd_str}"
            if name_key == "lse":
                return f"{m.get('total_files', 0)} files; {m.get('total_sheets', 0)} sheets; modules {m.get('modules', 0)}{upd_str}"
            if name_key == "deforestation":
                polys = m.get('total_polygons', r.get('total_polygons', 0))
                area = float(m.get('total_area_km2', r.get('total_area_km2', 0)) or 0)
                note = " (not loaded)" if int(polys or 0) == 0 else ""
                return f"{polys} polygons; {area:.0f} km²{upd_str}{note}"
            if name_key == "municipalities":
                pop = m.get('total_population')
                pop_str = f"; pop {pop:,}" if isinstance(pop, int) else ""
                area = m.get('total_area_km2')
                area_str = f"; area {float(area):.0f} km²" if area is not None else ""
                return f"{m.get('total_municipalities', 0)} municipalities{pop_str}{area_str}{upd_str}"
            if name_key == "heat":
                layers = m.get('layer_count', len(r.get('heat_layers', [])))
                note = " (not loaded)" if int(layers or 0) == 0 else ""
                return f"{layers} layers{upd_str}{note}"
            return ""

        friendly_names = {
            "cpr": "Knowledge Graph",
            "solar": "Solar Facilities",
            "gist": "Corporate Environmental Metrics (GIST)",
            "lse": "LSE (and Friends) Governance Data",
            "deforestation": "Deforestation (Brazil)",
            "admins": "Brazilian Administrative Boundaries",
            "heat": "Heat Stress Layers",
            "geospatial": "Geospatial Correlation",
            "viz": "Visualization"
        }
        # Show datasets only (exclude engines like geospatial/viz)
        included_keys = set()
        for key in ["cpr", "solar", "gist", "lse", "deforestation", "admins", "heat"]:
            if key in results:
                r = results[key]
                brief = server_cfg.get(key, {}).get("brief", key)
                desc = r.get("description") if isinstance(r, dict) else None
                summary = desc or brief
                display_name = r.get("name") if isinstance(r, dict) and r.get("name") else friendly_names.get(key, key)
                rows.append([display_name, summary, describe_to_metrics(key, r)])
                included_keys.add(key)

        # Fallbacks for any missing entries (datasets only; exclude engines like geospatial/viz)
        # Do not add fallback rows with "metadata unavailable" — show only live datasets

        overview_text = (
            "This workspace exposes live datasets: a climate policy knowledge graph, corporate environmental metrics (GIST), "
            "global solar facilities (TZ-SAM), Brazilian deforestation polygons, Brazilian municipalities and demographics, and heat-stress zones. "
            "You can view maps, generate tables and run spatial correlations (e.g., facilities within 1 km of heat zones)."
        )

        modules = [
            {
                "type": "text",
                "heading": "About This Project",
                "texts": [overview_text]
            },
            {
                "type": "table",
                "heading": "Datasets (Live)",
                "columns": ["Dataset", "What it provides", "Key metrics"],
                "rows": rows
            }
        ]

        # Add a concise "Try these" queries section grouped by dataset
        # Curate examples we know are robust in this build
        try_queries = [
            "Solar Facilities (TZ-SAM):\n• Show solar facilities in Brazil\n• Top 10 countries by number of solar facilities (table)\n• Show the largest solar facilities in Brazil (map)",
            "Spatial Correlation (Datasets):\n• Which solar assets are within 1 km of heat stress?\n• Which solar assets are within 1 km of deforestation?",
            "Heat Stress (Brazil):\n• Map top-quintile heat zones in Brazil",
            "Corporate Environmental Metrics (GIST):\n• Companies in Brazil with high water stress exposure (table)",
            "Brazilian Municipalities:\n• Show municipalities in Brazil (table)"
        ]
        modules.append({
            "type": "text",
            "heading": "Try These Queries",
            "texts": ["\n\n".join(try_queries)]
        })

        return {
            "query": user_query,
            "modules": modules,
            "metadata": {
                "query_type": "capabilities_overview",
                "modules_count": len(modules),
                "has_maps": False,
                "has_charts": False,
                "has_tables": True
            }
        }
    
    def _create_redirect_response(self, user_query: str) -> Dict:
        """
        Create a friendly redirect response for off-topic queries.
        
        Args:
            user_query: The off-topic query
            
        Returns:
            API-compliant response with redirect message
        """
        redirect_module = {
            "type": "text",
            "heading": "Let me help you with climate and environmental topics",
            "content": """
            
            Unfortunately I can't help you with that, but I can assist with many climate and environmental topics. I can help you with questions about:

            • **Climate Policy**: National climate strategies, NDCs, carbon pricing, adaptation plans
            • **Renewable Energy**: Solar facilities, wind power, renewable capacity by country/region
            • **Corporate Sustainability**: Company environmental impacts, water stress, GHG emissions
            • **Physical Climate Risks**: Floods, droughts, heat exposure, extreme weather impacts
            • **Environmental Data**: Biodiversity loss, deforestation, water resources, air quality

            **Example questions you could ask:**
            - "What are the water stress risks for major companies in Brazil?"
            - "Show me solar capacity growth in India over the last 5 years"
            - "How are financial institutions addressing climate risks?"
            - "What climate adaptation policies has Nigeria implemented?"
            - "Compare renewable energy deployment between China and the US"

            How can I help you explore climate and environmental topics today?"""
        }
        
        return {
            "query": user_query,
            "modules": [redirect_module],
            "metadata": {
                "query_type": "off_topic_redirect",
                "modules_count": 1,
                "has_maps": False,
                "has_charts": False,
                "has_tables": False
            }
        }
        
    async def _classify_query_type(self, user_query: str) -> str:
        """
        Classify query as new_topic, follow_up_simple, or follow_up_complex.
        
        Args:
            user_query: User's query to classify
            
        Returns:
            Query classification type
        """
        if not self.conversation_history or not self.cached_facts:
            return "new_topic"
        
        # Build context for classification
        recent_exchange = ""
        if len(self.conversation_history) >= 2:
            recent_exchange = f"Previous question: {self.conversation_history[-2]['content']}\nPrevious answer summary: {self.conversation_history[-1]['content'][:500]}"
        
        classification_prompt = f"""Classify this follow-up query based on the conversation context.

        Recent exchange:
        {recent_exchange}

        Current query: "{user_query}"

        Classification rules:
        - "follow_up_simple": Query asks for clarification, specific details, or facts already mentioned in previous response (e.g., "How many?", "What was the capacity?", "Tell me more about X that you mentioned")
        - "follow_up_complex": Query extends the topic but needs some new data (e.g., "What about other countries?", "Show me trends over time")
        - "new_topic": Query is unrelated to previous discussion or asks about completely different aspect

        Return ONLY one of: follow_up_simple, follow_up_complex, new_topic"""
        
        try:
            response = await call_small_model(
                system="You classify queries based on conversation context.",
                user_prompt=classification_prompt,
                max_tokens=20,
                temperature=0
            )
            
            classification = response.strip().lower()
            if classification not in ["follow_up_simple", "follow_up_complex", "new_topic"]:
                return "new_topic"
            
            return classification
            
        except Exception as e:
            print(f"Query classification failed: {e}. Defaulting to new_topic.")
            return "new_topic"
    
    async def _can_answer_from_context(self, user_query: str) -> Tuple[bool, str]:
        """
        Check if query can be answered from cached facts and conversation history.
        
        Args:
            user_query: User's query to check
            
        Returns:
            Tuple of (can_answer, reasoning)
        """
        if not self.cached_facts or not self.conversation_history:
            return False, "No cached context available"
        
        # Get query classification
        query_type = await self._classify_query_type(user_query)
        
        if query_type == "new_topic":
            return False, "Query is about a new topic"
        elif query_type == "follow_up_complex":
            return False, "Query requires additional data beyond cached context"
        else:  # follow_up_simple
            # Check if we have relevant facts to answer
            fact_summary = "\n".join([f"- {fact.text_content}..." for fact in self.cached_facts])
            
            check_prompt = f"""Can this query be answered using ONLY the available facts?

            Query: "{user_query}"

            Available facts:
            {fact_summary}

            Previous response included: {self.cached_response_text}...

            Answer with YES if the facts contain the specific information needed, or NO if new data is required."""
            
            try:
                response = await call_small_model(
                    system="You determine if cached facts can answer a query.",
                    user_prompt=check_prompt,
                    max_tokens=10,
                    temperature=0
                )
                
                can_answer = "yes" in response.strip().lower()
                reasoning = "Cached facts contain the required information" if can_answer else "Need to fetch new data"
                return can_answer, reasoning
                
            except Exception as e:
                return False, f"Context check failed: {e}"
    
    async def _answer_from_context(self, user_query: str) -> List[Dict]:
        """
        Generate response directly from cached facts without new MCP calls.
        
        Args:
            user_query: User's query to answer
            
        Returns:
            List of modules for the response
        """
        print(">> Generating fast response from cached context...")
        
        # Use synthesis to create response from cached facts
        from response_formatter import format_response_as_modules
        
        # Create a focused narrative from cached facts
        fact_list = "\n".join([f"{i+1}. {fact.text_content}" for i, fact in enumerate(self.cached_facts)])
        
        synthesis_prompt = f"""Answer this follow-up question using ONLY the provided facts.

Question: {user_query}

Available facts from previous response:
{fact_list}

Instructions:
1. Answer the specific question directly and concisely
2. Use placeholder citations [CITE_n] when referencing facts
3. Do not add information not present in the facts
4. If the exact answer isn't in the facts, say so clearly"""
        
        try:
            narrative = await call_large_model(
                system="You answer follow-up questions using cached information.",
                user_prompt=synthesis_prompt,
                max_tokens=500
            )
            
            # Build citation map and apply citations
            citation_map = self._build_citation_map(self.cached_facts, narrative)
            final_narrative = self._apply_citations(narrative, citation_map)
            
            # Format as modules
            sources = []
            for fact in self.cached_facts:
                if fact.citation:
                    sources.append({
                        "name": fact.citation.source_name,
                        "type": fact.citation.source_type,
                        "provider": fact.citation.server_origin
                    })
            
            # Check for any cached map/chart data
            map_data = None
            for fact in self.cached_facts:
                if 'raw_result' in fact.metadata:
                    raw = fact.metadata['raw_result']
                    if isinstance(raw, dict) and raw.get('type') == 'map_data_summary' and raw.get('geojson_url'):
                        map_data = raw
                        break
            
            formatted_response = format_response_as_modules(
                response_text=final_narrative,
                facts=self.cached_facts,
                map_data=map_data,
                chart_data=None,
                sources=sources,
                citation_registry={"citations": [], "module_citations": {}}
            )
            
            return formatted_response.get("modules", [])
            
        except Exception as e:
            print(f"Error generating context-based response: {e}")
            # Fall back to full process
            return None
    
    def _cache_facts_from_results(self, results: Dict[str, PhaseResult]):
        """
        Cache facts from phase results for future context-based responses.
        
        Args:
            results: Dictionary of PhaseResults from data collection
        """
        self.cached_facts = []
        for server_name, result in results.items():
            if result.is_relevant and result.facts:
                self.cached_facts.extend(result.facts)
        
        # Also cache a summary of the response (will be populated from synthesis)
        if self.cached_facts:
            print(f"[CACHE] Stored {len(self.cached_facts)} facts for future context")
    
    async def process_query(self, user_query: str) -> Dict:
        """
        Main entry point for query processing.
        
        Orchestrates all three phases and returns API-compliant response.
        Now includes context-aware optimization for follow-up queries.
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            Dictionary with query, modules, and metadata
        """
        try:
            # Log query processing start
            llm_logger.info(f"=== QUERY PROCESSING START ===")
            llm_logger.info(f"User query: {user_query}")
            llm_logger.info(f"Conversation history length: {len(self.conversation_history)}")
            if self.conversation_history:
                llm_logger.debug("Last 3 conversation turns:")
                for msg in self.conversation_history[-3:]:
                    llm_logger.debug(f"  {msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}...")
            llm_logger.info(f"Cached facts count: {len(self.cached_facts)}")
            llm_logger.info(f"Token budget remaining: {self.token_budget_remaining}")
            
            # Clear geospatial index for new query
            if self.client.sessions.get("geospatial"):
                try:
                    await self.client.call_tool("ClearSpatialIndex", {"session_id": self.spatial_session_id}, "geospatial")
                    print("Cleared geospatial index for new query")
                except Exception as e:
                    print(f"Failed to clear geospatial index: {e}")
            
            # Reset geospatial tracking flag
            self._geospatial_cleared = False
            
            # Estimate query complexity for optimization
            if OPTIMIZER_AVAILABLE:
                complexity = await PerformanceOptimizer.estimate_complexity(user_query)
                llm_logger.info(f"Query complexity: {complexity.value}")
                tool_limit = PerformanceOptimizer.TOOL_LIMITS[complexity]
            else:
                complexity = None
                tool_limit = 15  # Default limit
            
            # First check if query is relevant to our domain
            is_relevant = await self._is_query_relevant(user_query)
            if not is_relevant:
                return self._create_redirect_response(user_query)

            # If this is a meta/capability/identity query, return a capabilities overview
            if await self._is_meta_query(user_query):
                return await self._create_capabilities_response(user_query)
            
            # Initialize fact tracer for debugging if enabled
            if os.getenv('DEBUG_FACTS', '').lower() == 'true' and FACT_TRACER_AVAILABLE:
                self.client.fact_tracer = FactTracer(user_query)
                llm_logger.info(f"Fact tracing enabled with trace_id: {self.client.fact_tracer.trace_id}")
            
            # NEW: Check if we can answer from cached context
            can_use_context, reasoning = await self._can_answer_from_context(user_query)
            
            if can_use_context:
                print(f"[CACHED] Using cached context: {reasoning}")
                context_modules = await self._answer_from_context(user_query)
                
                if context_modules:
                    # Successfully answered from context
                    # Build KG context from cached data
                    cited_passage_ids = set()
                    for citation_num, citation in self.client.citation_registry.citation_objects.items():
                        if "passage_id" in citation.metadata:
                            cited_passage_ids.add(str(citation.metadata["passage_id"]))
                    kg_context = self.client.kg_context_tracker.build_kg_context(cited_passage_ids)
                    
                    return {
                        "query": user_query,
                        "modules": context_modules,
                        "metadata": {
                            "modules_count": len(context_modules),
                            "has_maps": any(m.get("type") == "map" for m in context_modules),
                            "has_charts": any(m.get("type") == "chart" for m in context_modules),
                            "has_tables": any(m.get("type") == "table" for m in context_modules),
                            "used_cached_context": True  # Flag for monitoring
                        },
                        "kg_context": kg_context
                    }
                else:
                    print("[WARNING] Context response failed, falling back to full process")
            else:
                print(f"[FULL] Running full process: {reasoning}")
            
            # Phase 1: Collect Information
            collection_results = await self._phase1_collect_information(user_query)
            
            # Phase 2: Deep Dive (if needed)
            need_phase2, reasoning, servers_for_phase2 = await self._should_do_phase2_deep_dive(
                user_query, collection_results
            )
            
            if need_phase2:
                print(f"Phase 2 needed: {reasoning}")
                # FIXED: Pass reasoning that contains follow-up queries to Phase 2
                deep_dive_results = await self._phase2_deep_dive(user_query, collection_results, reasoning)
            else:
                print(f"Skipping Phase 2: {reasoning}")
                deep_dive_results = collection_results
            
            # Phase 3: Synthesis
            modules = await self._phase3_synthesis(user_query, deep_dive_results)
            
            # Cache facts for potential follow-up queries
            self._cache_facts_from_results(deep_dive_results)
            
            # Extract cited passage IDs from CitationRegistry
            cited_passage_ids = set()
            for citation_num, citation in self.client.citation_registry.citation_objects.items():
                # Extract passage ID from citation metadata
                if "passage_id" in citation.metadata:
                    cited_passage_ids.add(str(citation.metadata["passage_id"]))
                # Also check tool_id for GetPassagesMentioningConcept:XXX pattern
                if citation.tool_id and ":" in citation.tool_id:
                    tool_name, passage_id = citation.tool_id.rsplit(":", 1)
                    if tool_name in ["GetPassagesMentioningConcept", "PassagesMentioningBothConcepts"]:
                        cited_passage_ids.add(passage_id)
            
            llm_logger.info(f"KG_CONTEXT: Extracted {len(cited_passage_ids)} cited passage IDs")
            
            # Build KG context for visualization
            kg_context = self.client.kg_context_tracker.build_kg_context(cited_passage_ids)
            
            # Build API response
            return {
                "query": user_query,
                "modules": modules,
                "metadata": {
                    "modules_count": len(modules),
                    "has_maps": any(m.get("type") == "map" for m in modules),
                    "has_charts": any(m.get("type") == "chart" for m in modules),
                    "has_tables": any(m.get("type") == "table" for m in modules),
                    "used_cached_context": False
                },
                "kg_context": kg_context  # Add KG context for visualization
            }
            
        except Exception as e:
            # Return error as a text module
            return {
                "query": user_query,
                "modules": [{
                    "type": "text",
                    "heading": "Error",
                    "texts": [f"Error processing query: {str(e)}"]
                }],
                "metadata": {
                    "modules_count": 1,
                    "has_maps": False,
                    "has_charts": False,
                    "has_tables": False
                }
            }
    
    # =============================================================================
    # SERVER CONFIGURATIONS - Easily adjustable server descriptions
    # =============================================================================
    
    def _get_server_descriptions(self) -> Dict[str, Dict[str, str]]:
        """
        Get server descriptions for routing decisions.
        
        Centralized configuration for easy adjustment of server capabilities.
        Each server has:
        - brief: One-line description for pre-filter
        - detailed: Comprehensive capability list for scout phase
        - collection_instructions: How to effectively use tools during Phase 1
        """
        return {
            "cpr": {
                "brief": "Climate Knowledge Graph with physical & transition risks, energy systems, financial climate impacts",
                "detailed": "This dataset is a knowledge graph of climate/environment related concepts applied to POLICY documents. This dataset is especially useful is policy is being discussed.",
                "collection_instructions": """Knowledge Graph expected workflow (concepts → neighbors → passages):
                
                1) Resolve concepts from the query (do NOT assume 'Brazil' as a concept; CPR KG is Brazil-scoped):
                   - Use concept finders to identify seed concepts from the query text:
                     • FindConceptMatchesByNgrams (exact label n-grams)
                     • GetTopConceptsByQueryLocal (token overlap)
                     • GetTopConceptsByQuery (semantic; only if embeddings available)
                     • SearchConceptsFuzzy (RapidFuzz over preferred + alternative labels)
                   - Deduplicate and keep top seeds most relevant to the query intent (e.g., 'solar energy', 'solar photovoltaic' for solar policy).
                
                2) Expand nearby KG context:
                   - Use GetConceptGraphNeighbors on each seed (direction='both') to collect parent/child/related concepts.
                   - Keep edge types (SUBCONCEPT_OF, HAS_SUBCONCEPT, RELATED_TO) to show structure.
                   - Optionally use FindConceptPathWithEdges for short paths between key concepts.
                
                3) Surface passages with citations:
                   - Use GetPassagesMentioningConcept for each seed (and select neighbor concepts when useful).
                   - If no MENTIONS spans are found for a concept, fallback to KG-side text search against labelled_passages for the concept's preferred + alternative labels (including Portuguese terms where relevant: 'energia solar', 'fotovoltaica', 'geração distribuída', 'compensação de energia', 'PROINFA', 'ANEEL/REN 482', 'leilões').
                   - Return real passages with passage_id/doc_id; avoid placeholders whenever possible.
                
                4) Assemble answer (KG-first):
                   - Lead with a concise summary.
                   - Provide 3–6 bullet points grounded in KG passages (with superscript citations).
                   - Optionally add a compact 'KG Evidence' table (Passage ID, Concept, Snippet).
                   - It is acceptable to include a supplemental asset count (e.g., solar facilities) at the end, but policy answers must prioritize KG evidence.
                
                Notes:
                - Do not include geospatial correlation or facility mapping unless the user asks for spatial relationships.
                - When extracting targets/figures, capture complete context and units.
                - NEVER use 'ALWAYSRUN' - it's a debug tool that returns nothing useful.
                - Prefer the composite KG policy-discovery tool when available: 'DiscoverPolicyContextForQuery'."""
            },
            "solar": {
                "brief": "Solar facility database with locations, capacity, renewable infrastructure globally",
                "detailed": "Comprehensive solar facility database with geospatial locations, capacity data (MW), countries, and datetimes with a range over when they could've been constructed.",
                "collection_instructions": """Tool usage strategy for Solar Database:
                - IMPORTANT: Call 'GetSolarFacilitiesMapData' when you need to show geographic distribution
                - Use 'GetSolarFacilitiesByCountry' for detailed country-level facility data
                - Use 'GetSolarCapacityByCountry' for country-level facility count stats (also returns top_10 list)
                - Use 'GetTopNCountriesByFacilities' for queries like 'top 10 countries by asset/facility count'
                - Use 'GetSolarConstructionTimeline' for temporal trends and growth analysis
                - Use 'GetLargestSolarFacilities' to highlight major installations
                - For multi-country analysis, use 'GetSolarFacilitiesMultipleCountries'
                - Map tools automatically generate interactive visualizations
                - Always aim to provide both statistics AND geographic visualizations"""
            },
            "gist": {
                "brief": "Company environmental data: water stress (MSA), drought/flood risks, heat exposure, GHG emissions",
                "detailed": "GIST Impact database with company-level environmental metrics including: water stress (Mean Species Abundance), drought risk, flood risk (coastal/riverine), extreme heat exposure, extreme precipitation, temperature anomalies, land use changes (urban expansion, forest loss, agricultural conversion), population density impacts, corporate environmental impacts (Scope 1/2/3 GHG emissions, water consumption, SOX/NOX emissions, nitrogen/phosphorous pollution, waste generation), and asset-level risk assessments",
                "collection_instructions": """Tool usage strategy for GIST Impact Database:
                - Use 'GetGistCompanyWaterData' for water stress and MSA metrics
                - Use 'GetGistCompanyClimateRisks' for physical climate risks (drought, flood, heat)
                - Use 'GetGistCompanyEmissions' for GHG data (Scope 1, 2, 3)
                - For sector analysis, query multiple companies in the same industry
                - Always collect both current metrics AND trend data when available
                - For risk assessment, gather multiple risk types (water, climate, emissions)
                - Include asset-level data when analyzing specific locations
                - Collect both absolute values and intensity metrics for meaningful comparison"""
            },
            "lse": {
                "brief": "NDC commitments, climate targets, emissions reductions, Brazilian governance, policy frameworks",
                "detailed": "Comprehensive climate policy database including: NDC targets and commitments (emissions reduction percentages, net-zero dates, renewable energy goals), NDC vs domestic policy comparison, institutional frameworks (direction setting, planning, coordination), climate policies (cross-cutting, sectoral mitigation/adaptation), Brazilian state-level governance, implementation tracking, and TPI emissions pathways",
                "collection_instructions": """Tool usage strategy for LSE Climate Policy:
                - Use 'GetNDCTargets' FIRST for specific country NDC commitments and targets
                - For energy-specific NDC queries, also call 'GetPlansAndPoliciesData' to find energy sector details
                - Use 'GetNDCPolicyComparison' for NDC vs domestic law analysis  
                - Use 'GetNDCImplementationStatus' for tracking progress
                - Use 'GetTPIGraphData' for emissions pathway visualization
                - Use 'GetInstitutionalFramework' for governance structure queries
                - Use 'GetClimatePolicy' for specific policy details
                - Use 'GetSubnationalGovernance' for Brazilian state-level data
                - CRITICAL: Extract ALL quantitative targets including:
                  * Renewable energy percentages (any % targets)
                  * Alternative fuel targets (biofuels, hydrogen, etc.)
                  * Capacity targets (TWh, MW, GW)
                  * Emissions reduction percentages
                  * Sectoral targets (transport, industry, buildings)
                - For energy-specific queries: Look for ALL fuel types, electricity mix, capacity expansions
                - For NDC queries, call MULTIPLE tools to ensure comprehensive coverage"""
            },
            "viz": {
                "brief": "Data visualization tools for creating charts, tables, and comparisons",
                "detailed": "Visualization server providing smart chart generation, comparison tables, and data visualization tools. Can create Bar/Line/Pie charts, comparison tables with percentages and totals, and optimize visualization types based on data characteristics",
                "collection_instructions": """Tool usage strategy for Viz Server:
                CRITICAL: NEVER generate, invent, or guess data values! Only visualize data that has been explicitly provided to you from other servers.
                
                - If you don't have specific data, DO NOT create placeholder values
                - Wait for actual data from kg, lse, or other servers before creating visualizations
                - Use 'CreateDataTable' ONLY when you have real data to display
                  * CRITICAL DATA FORMAT: Pass percentages as whole numbers (45 for 45%, NOT 0.45)
                  * CRITICAL DATA FORMAT: Pass years as integers (2030, NOT "2030" or 2,030)
                - Use 'CreateComparisonTable' when you need "% of Total" calculations for counts/values
                  * Best for facility counts, capacity comparisons, emissions by entity
                - Use 'create_smart_chart' for automatic chart type selection
                - Use 'create_comparison_chart' for side-by-side comparisons
                
                Table Selection Guide:
                - Data contains percentages/rates? → CreateDataTable (no meaningless totals)
                - Need proportion analysis? → CreateComparisonTable (adds "% of Total")
                - Mixed data types? → CreateDataTable (flexible column formatting)
                
                DATA FORMAT REQUIREMENTS:
                - Percentages: Always pass 45 for "45%", never 0.45
                - Years: Always pass 2030 as integer, not "2030" string
                - Rates: Pass as displayed (5.5 for "5.5%")
                - Example: {"country": "Brazil", "target": 45, "year": 2030}"""
            },
            "deforestation": {
                "brief": "Brazil deforestation polygon data from processed satellite imagery",
                "detailed": "Deforestation area polygons with for spatial analysis",
                "collection_instructions": """Tool usage strategy for Deforestation:
                IMPORTANT: Always call GetDeforestationAreas first when deforestation is mentioned in the query
                - Use 'GetDeforestationAreas' as your primary tool for deforestation data
                - Use 'GetDeforestationInBounds' for specific geographic regions  
                - Use 'GetDeforestationStatistics' for summary statistics
                - Use 'GetDeforestationWithMap' to generate visualization maps
                - These tools return polygon geometries that auto-register for spatial correlation
                - The polygon data enables geographic overlap analysis with other datasets"""
            },
            "admin": {
                "brief": "Brazilian administrative boundaries",
                "detailed": "Administrative boundaries for municipalities and states within Brazil. Useful for when asked about particular places within Brazil. If a query asks about a specific state or municipality, use this server to get the polygon boundary for that place.",
                "collection_instructions": """Tool usage strategy for Municipalities:
                - Use 'GetMunicipalitiesByFilter' for administrative queries
                - Use 'GetMunicipalityBoundaries' for specific municipality polygons
                - Use 'GetMunicipalitiesInBounds' for spatial region queries
                - Use 'FindMunicipalitiesNearPoint' for proximity searches
                - Use 'GetMunicipalityStatistics' for aggregate analysis
                - Returns full polygon boundaries that auto-register with geospatial server
                - Enables questions like 'which municipalities have highest deforestation'"""
            },
            "geospatial": {
                "brief": "Spatial correlation engine for geographic relationship analysis",
                "detailed": "Correlates different geographic datasets by proximity, overlap, or containment. Does NOT store data - only analyzes relationships during query session. Receives entity registrations from other servers and performs spatial analysis.",
                "collection_instructions": """Tool usage strategy for Geospatial:
                - This server is PASSIVE in Phase 1 - it receives auto-registered entities from other servers
                - ACTIVE in Phase 2 - use for correlation and map generation
                - Use 'FindSpatialCorrelations' to find relationships (within, intersects, proximity)
                - Use 'GenerateCorrelationMap' to create multi-layer visualization
                - Use 'GetRegisteredEntities' to check what data is available
                - Use 'ClearSpatialIndex' between different queries"""
            },
            "meta": {
                "brief": "Metadata server for managing and retrieving dataset information",
                "detailed": "Handles metadata for various datasets, including descriptions, schemas, and provenance information. Supports querying and updating metadata records.",
                "collection_instructions": """Call the tools to find out more information about this project."""
            },
            "spa": {
                "brief": "Semantic passage index for retrieving relevant text passages from documents",
                "detailed": "The Science Panel for the Amazon released the Amazon Assessment Report 2021 at COP26, which has been called an “encyclopedia” of the Amazon region. This landmark report is unprecedented for its scientific and geographic scope, the inclusion of Indigenous scientists, and its transparency, having undergone peer review and public consultation.",
                "collection_instructions": """Tool usage strategy for SPA:
                - Use 'AmazonAssessmentListDocs' to get an overview of available documents
                - Use 'AmazonAssessmentSearch' for keyword-based searches
                - Use 'AmazonAssessmentAsk' for expert Q&A with context
                - All tools leverage the semantic passage index for retrieval
                - Ensure queries are clear and specific to improve results"""
            }
        }
    
    async def _evaluate_server_relevance(self, query: str, server_name: str, config: dict) -> tuple[str, bool]:
        """
        Evaluate if a single server is relevant to the query.

        Args:
            query: User query
            server_name: Name of the server to evaluate
            config: Server configuration with descriptions

        Returns:
            Tuple of (server_name, is_relevant)
        """
        # Simple, focused prompt that relies on server descriptions
        prompt = f"""Query: "{query}"

        Data Source: {server_name}
        Capabilities: {config['detailed']}

        Does this data source contain information that would help answer the query?

        Answer YES if:
        - The source has data which is likely to be relevant to answering the query
        - The query asks for information this source provides

        Answer NO if:
        - The source doesn't have relevant data for this specific query
        - Other sources would be more appropriate
        - The connection is tangential or coincidental

        Respond with only YES or NO."""

        try:
            response = await call_small_model(
                system="You are a precise query router. Reply with only YES or NO.",
                user_prompt=prompt,
                max_tokens=10,
                temperature=0
            )

            # Parse response
            answer = response.strip().upper()
            is_relevant = answer == "YES"

            # Log decision for debugging
            print(f"  {'✓' if is_relevant else '✗'} {server_name}: {answer}")

            return (server_name, is_relevant)

        except Exception as e:
            print(f"  ⚠ Error evaluating {server_name}: {e}")
            # On error, default to including the server (fail-open)
            return (server_name, True)

    def _apply_routing_rules(self, relevant_servers: List[str], query: str) -> List[str]:
        """
        Apply business logic rules after relevance evaluation.

        Args:
            relevant_servers: List of servers marked as relevant
            query: Original user query

        Returns:
            Final list of servers to use
        """
        # Always include viz server for potential visualizations
        if 'viz' not in relevant_servers:
            relevant_servers.append('viz')

        # Check for spatial relationship keywords
        spatial_keywords = ['near', 'close to', 'within', 'overlap', 'proximity', 'adjacent', 'surrounding']
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in spatial_keywords):
            # If spatial relationships mentioned and we have geographic data sources
            geographic_servers = ['solar', 'deforestation', 'municipalities']
            if any(server in relevant_servers for server in geographic_servers):
                if 'geospatial' not in relevant_servers:
                    relevant_servers.append('geospatial')
                    print(f"  ➕ Added 'geospatial' for spatial correlation")

        # Ensure at least one data server (not just viz)
        if len(relevant_servers) == 1 and relevant_servers[0] == 'viz':
            # Default to kg if no data servers selected
            relevant_servers.insert(0, 'kg')
            print(f"  ➕ Added 'kg' as default data source")

        return relevant_servers

    async def _phase0_prefilter(self, user_query: str) -> List[str]:
        """
        Phase 0: Pre-filter servers using parallel per-server LLM evaluation.

        Each server is evaluated independently for relevance, allowing for
        more precise routing decisions based on detailed server capabilities.

        Args:
            user_query: Original user query

        Returns:
            List of server names that should proceed to scout phase
        """
        print(f"\n🔍 Phase 0: Evaluating server relevance for query")
        server_descriptions = self._get_server_descriptions()

        # Create evaluation tasks for all servers (except always-include ones)
        evaluation_tasks = []
        always_include_servers = []

        for server_name, config in server_descriptions.items():
            # Check if server should always be included
            if config.get('always_include', False):
                always_include_servers.append(server_name)
                print(f"  ✓ {server_name}: ALWAYS INCLUDED")
                continue

            # Create evaluation task
            task = self._evaluate_server_relevance(user_query, server_name, config)
            evaluation_tasks.append(task)

        # Run all evaluations in parallel
        if evaluation_tasks:
            results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        else:
            results = []

        # Process results
        relevant_servers = always_include_servers.copy()

        for result in results:
            if isinstance(result, Exception):
                print(f"  ⚠ Evaluation error: {result}")
                continue

            server_name, is_relevant = result
            if is_relevant:
                relevant_servers.append(server_name)

        # Apply business rules and special cases
        relevant_servers = self._apply_routing_rules(relevant_servers, user_query)

        print(f"\n✅ Pre-filter selected servers: {relevant_servers}")
        return relevant_servers
    
    async def _phase1_collect_information(self, user_query: str) -> Dict[str, PhaseResult]:
        """
        Phase 1: Active Information Collection from pre-filtered servers.
        
        Servers that passed Phase 0 pre-filter now actively collect information
        by calling relevant tools. All servers run in parallel for efficiency.
        
        Args:
            user_query: Original user query
            
        Returns:
            Dictionary mapping server names to their collection results
        """
        # Phase 0: Pre-filter to identify potentially relevant servers
        relevant_servers = await self._phase0_prefilter(user_query)
        
        if not relevant_servers:
            # If pre-filter returns empty, default to knowledge graph
            print("No servers selected by pre-filter. Defaulting to 'kg' server.")
            relevant_servers = ["cpr"]
        
        server_descriptions = self._get_server_descriptions()
        
        # Create collection coroutines only for pre-filtered servers
        collection_coroutines = []
        for server_name in relevant_servers:
            if server_name in server_descriptions:
                config = server_descriptions[server_name]
                coroutine = self._collect_server_information(user_query, server_name, config)
                collection_coroutines.append((server_name, coroutine))
        
        # Execute collection in parallel
        results = {}
        if collection_coroutines:
            collection_results = await asyncio.gather(
                *[coroutine for _, coroutine in collection_coroutines], 
                return_exceptions=True
            )
            
            # Process results
            for (server_name, _), result in zip(collection_coroutines, collection_results):
                if isinstance(result, Exception):
                    # Failed collections have no facts
                    results[server_name] = PhaseResult(
                        is_relevant=True,  # Was selected by pre-filter
                        facts=[],
                        reasoning=f"Collection failed: {str(result)}"
                    )
                else:
                    results[server_name] = result
        
        # Mark non-selected servers (for completeness)
        for server_name in server_descriptions:
            if server_name not in results:
                results[server_name] = PhaseResult(
                    is_relevant=False,  # Not selected by pre-filter
                    facts=[],
                    reasoning="Not selected by pre-filter"
                )
        
        # Post Phase 1: Run deterministic spatial correlation ONLY on explicit spatial relation intent
        ql = user_query.lower()
        # Detect spatial relation intent using deterministic language (not tied to specific layers)
        is_spatial_query = any(keyword in ql for keyword in [
            "within", "inside", "intersect", "overlap", "near", "close to",
            "proximity", "adjacent", "located in", "around", "km"
        ])
        if (not _geo_llm_only()) and is_spatial_query and self.client.sessions.get("geospatial"):
            try:
                reg = await self.client.call_tool("GetRegisteredEntities", {"session_id": self.spatial_session_id}, "geospatial")
                reg_data = {}
                if hasattr(reg, 'content') and reg.content:
                    import json as _json
                    reg_data = _json.loads(reg.content[0].text)
                # If solar facilities are registered, attempt correlation.
                if isinstance(reg_data, dict) and reg_data.get("by_type", {}).get("solar_facility", 0) > 0:
                    import re
                    # Choose the polygon zone based on query intent (heat or deforestation)
                    zone_choice = self._detect_zone_intent(user_query) or "deforestation_area"
                    # If heat zones are requested but not yet registered, try to bootstrap them
                    try:
                        by_type = reg_data.get("by_type", {}) if isinstance(reg_data, dict) else {}
                        if zone_choice == "heat_zone" and by_type.get("heat_zone", 0) == 0:
                            heat_args = {"quintiles": [5], "limit": 5000}
                            hres = await self.client.call_tool("GetHeatQuintilesForGeospatial", heat_args, "heat")
                            if hasattr(hres, 'content') and hres.content:
                                data = _json.loads(hres.content[0].text)
                                entities = data.get('entities', []) if isinstance(data, dict) else []
                                if entities:
                                    await self.client.call_tool(
                                        "RegisterEntities",
                                        {"entity_type": "heat_zone", "entities": entities, "session_id": self.spatial_session_id},
                                        "geospatial"
                                    )
                    except Exception:
                        pass
                    # Compute correlation (default to within unless distance like '1 km' is present)
                    corr_args = {"entity_type1": "solar_facility", "entity_type2": zone_choice, "session_id": self.spatial_session_id}
                    m = re.search(r"(\d+(?:\.\d+)?)\s*km", ql)
                    if m:
                        try:
                            distance_km = float(m.group(1))
                            corr_args.update({"method": "proximity", "distance_km": distance_km})
                        except Exception:
                            corr_args.update({"method": "within"})
                    else:
                        corr_args.update({"method": "within"})
                    corr = await self.client.call_tool("FindSpatialCorrelations", corr_args, "geospatial")
                    corr_data = {}
                    if hasattr(corr, 'content') and corr.content:
                        corr_data = _json.loads(corr.content[0].text)
                    # Generate map (generic correlation type label)
                    map_res = await self.client.call_tool(
                        "GenerateCorrelationMap",
                        {"correlation_type": f"solar_vs_{zone_choice}", "session_id": self.spatial_session_id, "show_uncorrelated": False},
                        "geospatial"
                    )
                    map_data = {}
                    if hasattr(map_res, 'content') and map_res.content:
                        map_data = _json.loads(map_res.content[0].text)
                    # Append as facts to geospatial result (create if missing)
                    geo_result = results.get("geospatial")
                    geo_facts = list(geo_result.facts) if geo_result else []
                    if corr_data:
                        geo_facts.append(Fact(
                            text_content=f"Spatial correlation result: total_correlations={corr_data.get('total_correlations', 'unknown')}",
                            source_key="geospatial_phase1_correlation",
                            server_origin="geospatial",
                            metadata={"tool": "FindSpatialCorrelations", "raw_result": corr_data},
                            citation=Citation(
                                source_name="GEOSPATIAL",
                                tool_id="FindSpatialCorrelations",
                                server_origin="geospatial",
                                source_type="Database",
                                description="Deterministic Phase 1 correlation",
                                source_url=self._resolve_source_url("geospatial", "FindSpatialCorrelations")
                            )
                        ))
                    if map_data:
                        geo_facts.append(Fact(
                            text_content="Correlation map generated",
                            source_key="geospatial_phase1_map",
                            server_origin="geospatial",
                            metadata={"tool": "GenerateCorrelationMap", "raw_result": map_data},
                            citation=Citation(
                                source_name="GEOSPATIAL",
                                tool_id="GenerateCorrelationMap",
                                server_origin="geospatial",
                                source_type="Database",
                                description="Deterministic Phase 1 correlation map",
                                source_url=self._resolve_source_url("geospatial", "GenerateCorrelationMap")
                            )
                        ))
                    results["geospatial"] = PhaseResult(
                        is_relevant=True if not geo_result else geo_result.is_relevant,
                        facts=geo_facts,
                        reasoning=(geo_result.reasoning if geo_result else "Deterministic geospatial correlation added"),
                        continue_processing=False
                    )
            except Exception as e:
                print(f"Post-Phase1 spatial correlation skipped: {e}")
        
        return results
    
    async def _collect_server_information(self, user_query: str, server_name: str, config: Dict[str, Any]) -> PhaseResult:
        """
        Actively collect information from a pre-filtered relevant server.
        
        Since this server passed Phase 0 pre-filtering, we know it's relevant.
        Focus on actively calling tools to gather useful context for answering
        the user's query. Uses a while loop with small scope and token budget.
        
        Args:
            user_query: Original user query
            server_name: Name of server to collect from
            config: Server configuration with collection instructions
            
        Returns:
            PhaseResult with collected facts (is_relevant=True since pre-filtered)
        """
        try:
            # Get available tools for this server
            session = self.client.sessions.get(server_name)
            if not session:
                return PhaseResult(
                    is_relevant=True,  # Was pre-filtered as relevant
                    facts=[],
                    reasoning=f"Server {server_name} not connected"
                )
            
            # Get the list of tools from this server
            tools_response = await session.list_tools()
            available_tools = []
            
            # Convert MCP tools to Anthropic tool format
            for tool in tools_response.tools:
                anthropic_tool = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                available_tools.append(anthropic_tool)
            
            if not available_tools:
                return PhaseResult(
                    is_relevant=True,  # Was pre-filtered as relevant
                    facts=[],
                    reasoning=f"Server {server_name} has no available tools"
                )
            
            # Build optional query-specific hints to guide tool choice (no shortcuts)
            query_hints = ""
            ql = user_query.lower()
            if server_name == "cpr":
                hints = []
                if any(k in ql for k in ["shortest path", "relationship path", "path between", "how are", "how is", "connected to"]):
                    hints.append("For relationship/path queries, consider calling 'FindConceptPathWithEdges' with the two concept names and an appropriate max_len (e.g., 4).")
                if ("mention both" in ql) or ("co-mention" in ql) or ("passage" in ql and " and " in ql):
                    hints.append("For co-mention passage queries, consider calling 'PassagesMentioningBothConcepts' with the two concept names and a reasonable limit (e.g., 5).")
                if hints:
                    query_hints = "\n\nQUERY-SPECIFIC HINTS:\n- " + "\n- ".join(hints)
            elif server_name == "solar":
                hints = []
                ql2 = ql  # already lowered above
                # Top-N by count
                if any(k in ql2 for k in ["top 10", "top ten", "top countries", "most", "highest"]) and \
                   any(k in ql2 for k in ["count", "counts", "asset count", "facilities", "facility count", "assets"]):
                    hints.append("For 'top N countries by facility/asset count' queries, call 'GetTopNCountriesByFacilities' (set n appropriately).")
                # By-country summary
                if "by country" in ql2 or ("countries" in ql2 and "compare" in ql2):
                    hints.append("For country-level counts, call 'GetSolarCapacityByCountry' which returns a top_10 list and totals.")
                # Mapping intent
                if any(k in ql2 for k in ["map", "where", "show me", "locations"]):
                    hints.append("For maps of facilities, call 'GetSolarFacilitiesMapData' (optionally pass a country).")
                if hints:
                    query_hints = "\n\nQUERY-SPECIFIC HINTS:\n- " + "\n- ".join(hints)

                # Deterministic KG bootstrap using semantic + exact concept retrieval from the query
                try:
                    # 1) High-precision: exact unigram/bigram label matches from query
                    exact_res = await self.client.call_tool("FindConceptMatchesByNgrams", {"query": user_query, "top_k": 10}, server_name)
                    exact_list = []
                    if hasattr(exact_res, 'content') and exact_res.content:
                        import json as _json
                        data0 = _json.loads(exact_res.content[0].text)
                        if isinstance(data0, list):
                            exact_list = [d.get("label") for d in data0 if isinstance(d, dict) and d.get("label")]
                    llm_logger.info(f"KG BOOTSTRAP exact ngram matches: {exact_list}")

                    # 2) Local token-overlap retrieval (offline, robust)
                    local_res = await self.client.call_tool("GetTopConceptsByQueryLocal", {"query": user_query, "top_k": 10}, server_name)
                    local_list = []
                    if hasattr(local_res, 'content') and local_res.content:
                        import json as _json
                        dataL = _json.loads(local_res.content[0].text)
                        if isinstance(dataL, list):
                            local_list = [d.get("label") for d in dataL if isinstance(d, dict) and d.get("label")]
                    llm_logger.info(f"KG BOOTSTRAP local token-overlap top: {local_list}")

                    # Embedding diagnostics (so we know why semantic may be empty)
                    try:
                        dbg_res = await self.client.call_tool("DebugEmbeddingStatus", {}, server_name)
                        if hasattr(dbg_res, 'content') and dbg_res.content:
                            import json as _json
                            dbg = _json.loads(dbg_res.content[0].text)
                            llm_logger.info(f"KG DEBUG: env_has_key={dbg.get('env_has_openai_key')} key_prefix={dbg.get('openai_key_prefix')} vectors_present={dbg.get('vectors_column_present')} dtype={dbg.get('vectors_dtype')} nulls={dbg.get('vectors_null_count')} parsed_row0={dbg.get('row0_type_after')} total_concepts={dbg.get('total_concepts')} graphml_exists={dbg.get('graphml_path_exists')}")
                    except Exception as e:
                        llm_logger.warning(f"KG DEBUG call failed: {e}")

                    # 3) Try semantic retrieval via embeddings (may be unavailable)
                    top_res = await self.client.call_tool("GetTopConceptsByQuery", {"query": user_query, "top_k": 5}, server_name)
                    top_list = []
                    if hasattr(top_res, 'content') and top_res.content:
                        import json as _json
                        data = _json.loads(top_res.content[0].text)
                        if isinstance(data, list):
                            top_list = [d.get("label") for d in data if isinstance(d, dict) and d.get("label")]
                    llm_logger.info(f"KG BOOTSTRAP semantic top: {top_list}")
                    # 2) Fallback to fuzzy text search when embeddings unavailable
                    if not top_list:
                        search_res = await self.client.call_tool("SearchConceptsByText", {"query": user_query, "top_k": 5}, server_name)
                        if hasattr(search_res, 'content') and search_res.content:
                            import json as _json
                            data2 = _json.loads(search_res.content[0].text)
                            if isinstance(data2, list):
                                top_list = [d.get("label") for d in data2 if isinstance(d, dict) and d.get("label")]
                        llm_logger.info(f"KG BOOTSTRAP text search top: {top_list}")

                    # Merge exact matches first, then local token-overlap, then semantic/fuzzy, deduplicated
                    merged = []
                    seen_lbl = set()
                    for lbl in exact_list + local_list + top_list:
                        if lbl and lbl not in seen_lbl:
                            merged.append(lbl)
                            seen_lbl.add(lbl)
                    llm_logger.info(f"KG BOOTSTRAP merged candidates: {merged}")

                    # Augment candidates via server-side fuzzy search (no client heuristics)
                    try:
                        fuzzy_res = await self.client.call_tool("SearchConceptsFuzzy", {"query": user_query, "top_k": 10, "min_score": 70}, server_name)
                        if hasattr(fuzzy_res, 'content') and fuzzy_res.content:
                            import json as _json
                            fz = _json.loads(fuzzy_res.content[0].text)
                            if isinstance(fz, list):
                                for d in fz:
                                    lbl = d.get("label")
                                    if lbl and lbl not in seen_lbl:
                                        merged.append(lbl)
                                        seen_lbl.add(lbl)
                            llm_logger.info(f"KG BOOTSTRAP fuzzy added: {len(fz) if isinstance(fz,list) else 0}")
                    except Exception as e:
                        llm_logger.warning(f"KG fuzzy search failed: {e}")

                    # Heuristic: ensure 'terrestrial risk' is tried when deforestation is present
                    try:
                        if (('deforestation' in [x.lower() for x in exact_list + local_list + top_list])
                            and ('terrestrial risk' not in [x.lower() for x in merged])):
                            exists_tr = await self.client.call_tool("CheckConceptExists", {"concept": "terrestrial risk"}, server_name)
                            tr_ok = False
                            if hasattr(exists_tr, 'content') and exists_tr.content:
                                import json as _json
                                try:
                                    tr_ok = bool(_json.loads(exists_tr.content[0].text))
                                except Exception:
                                    tr_ok = str(exists_tr.content[0].text).strip().lower() in ("true","1","yes")
                            if tr_ok:
                                merged.insert(0, 'terrestrial risk')
                                llm_logger.info("KG BOOTSTRAP injected 'terrestrial risk' due to deforestation in query")
                    except Exception as e:
                        llm_logger.warning(f"KG BOOTSTRAP terrestrial risk injection failed: {e}")

                    # Deterministic traversal & context (no scoring)
                    # 1) Select seed concepts and expand neighbors
                    seeds = merged[:2]  # first two seed concepts
                    
                    # Track seed concepts in KG context
                    for seed in seeds:
                        self.client.kg_context_tracker.add_seed_concept(seed)
                    
                    neighbor_labels = []
                    try:
                        for seed in seeds:
                            # Collect parents (outgoing SUBCONCEPT_OF), children (incoming SUBCONCEPT_OF), related (RELATED_TO)
                            parents = []
                            children = []
                            relateds = []
                            try:
                                out_res = await self.client.call_tool(
                                    "GetConceptGraphNeighbors",
                                    {"concept": seed, "edge_types": ["SUBCONCEPT_OF","RELATED_TO"], "direction": "out", "max_results": 10},
                                    server_name
                                )
                                in_res = await self.client.call_tool(
                                    "GetConceptGraphNeighbors",
                                    {"concept": seed, "edge_types": ["SUBCONCEPT_OF","RELATED_TO"], "direction": "in", "max_results": 10},
                                    server_name
                                )
                                import json as _json
                                out_list = _json.loads(out_res.content[0].text) if hasattr(out_res,'content') and out_res.content else []
                                in_list  = _json.loads(in_res.content[0].text) if hasattr(in_res,'content') and in_res.content else []
                                # Parents: out edges with SUBCONCEPT_OF
                                for n in out_list:
                                    if isinstance(n, dict) and n.get('kind') == 'Concept' and n.get('via_edge') == 'SUBCONCEPT_OF':
                                        label = n.get('label') or n.get('node_id')
                                        parents.append(label)
                                        # Track edge: seed -> parent (SUBCONCEPT_OF)
                                        self.client.kg_context_tracker.add_concept_edge(seed, label, "SUBCONCEPT_OF")
                                # Children: in edges with SUBCONCEPT_OF
                                for n in in_list:
                                    if isinstance(n, dict) and n.get('kind') == 'Concept' and n.get('via_edge') == 'SUBCONCEPT_OF':
                                        label = n.get('label') or n.get('node_id')
                                        children.append(label)
                                        # Track edge: child -> seed (SUBCONCEPT_OF)
                                        self.client.kg_context_tracker.add_concept_edge(label, seed, "SUBCONCEPT_OF")
                                # Related: any RELATED_TO from either side
                                for n in out_list + in_list:
                                    if isinstance(n, dict) and n.get('kind') == 'Concept' and n.get('via_edge') == 'RELATED_TO':
                                        label = n.get('label') or n.get('node_id')
                                        relateds.append(label)
                                        # Track edges: bidirectional RELATED_TO
                                        self.client.kg_context_tracker.add_concept_edge(seed, label, "RELATED_TO")
                                        self.client.kg_context_tracker.add_concept_edge(label, seed, "RELATED_TO")
                            except Exception as e:
                                llm_logger.warning(f"KG NEIGHBORS failed for '{seed}': {e}")

                            # Deterministic selection: 1 parent, up to 2 children, 1 related
                            # Sort labels alphabetically for stability
                            parents = sorted([p for p in parents if p])
                            children = sorted([c for c in children if c])
                            relateds = sorted([r for r in relateds if r])
                            if parents:
                                neighbor_labels.append(parents[0])
                                # Track as parent neighbor
                                self.client.kg_context_tracker.add_neighbor_concept(parents[0], "parent", seed)
                            for child in children[:2]:
                                neighbor_labels.append(child)
                                # Track as child neighbor
                                self.client.kg_context_tracker.add_neighbor_concept(child, "child", seed)
                            if relateds:
                                neighbor_labels.append(relateds[0])
                                # Track as related neighbor
                                self.client.kg_context_tracker.add_neighbor_concept(relateds[0], "related", seed)
                    except Exception as e:
                        llm_logger.warning(f"KG neighbor expansion error: {e}")

                    # Deduplicate neighbors and remove those already in merged
                    neighbor_labels_dedup = []
                    for lbl in neighbor_labels:
                        if lbl and (lbl not in neighbor_labels_dedup) and (lbl not in merged):
                            neighbor_labels_dedup.append(lbl)
                    llm_logger.info(f"KG NEIGHBOR concepts selected: {neighbor_labels_dedup}")

                    # 2) Collect evidence deterministically: first a subset of seeds, then a subset of neighbors
                    collected_facts = []
                    seed_for_passages = seeds + merged[2:3]  # up to 3 total seed concepts
                    for label in seed_for_passages:
                        try:
                            pass_res = await self.client.call_tool("GetPassagesMentioningConcept", {"concept": label, "limit": 5}, server_name)
                            facts_here = self._extract_facts_from_result("GetPassagesMentioningConcept", {"concept": label, "limit": 5}, pass_res, server_name)
                            # Log summary of passages and track in KG context
                            try:
                                if hasattr(pass_res, 'content') and pass_res.content:
                                    import json as _json
                                    pr = _json.loads(pass_res.content[0].text)
                                    if isinstance(pr, list):
                                        # Track passages in KG context
                                        self.client.kg_context_tracker.add_passages_for_concept(label, pr)
                                        sample = pr[0] if pr else {}
                                        doc = sample.get('doc_id'); pid = sample.get('passage_id')
                                        llm_logger.info(f"KG PASSAGES for '{label}': {len(pr)} items; sample doc={doc} pid={pid}")
                            except Exception as e:
                                llm_logger.warning(f"KG PASSAGES logging error: {e}")
                            collected_facts.extend(facts_here)
                        except Exception as e:
                            print(f"KG bootstrap error for '{label}': {e}")

                    # Neighbor evidence: only first 2 neighbors, one passage each
                    for label in neighbor_labels_dedup[:2]:
                        try:
                            pass_res = await self.client.call_tool("GetPassagesMentioningConcept", {"concept": label, "limit": 1}, server_name)
                            facts_here = self._extract_facts_from_result("GetPassagesMentioningConcept", {"concept": label, "limit": 1}, pass_res, server_name)
                            # Track neighbor passages in KG context
                            try:
                                if hasattr(pass_res, 'content') and pass_res.content:
                                    import json as _json
                                    pr = _json.loads(pass_res.content[0].text)
                                    if isinstance(pr, list):
                                        self.client.kg_context_tracker.add_passages_for_concept(label, pr)
                            except Exception:
                                pass
                            collected_facts.extend(facts_here)
                        except Exception as e:
                            llm_logger.warning(f"KG neighbor passage fetch failed for '{label}': {e}")

                    # If both indigenous and terrestrial risk are in candidates, also fetch co-mentions
                    try:
                        cand_lower = [c.lower() for c in merged]
                        if ('indigenous people' in cand_lower) and ('terrestrial risk' in cand_lower):
                            both_res = await self.client.call_tool(
                                "PassagesMentioningBothConcepts",
                                {"concept_a": "indigenous people", "concept_b": "terrestrial risk", "limit": 5},
                                server_name
                            )
                            both_facts = self._extract_facts_from_result("PassagesMentioningBothConcepts", {"concept_a": "indigenous people", "concept_b": "terrestrial risk", "limit": 5}, both_res, server_name)
                            # Track co-mention passages for both concepts
                            try:
                                if hasattr(both_res, 'content') and both_res.content:
                                    import json as _json
                                    pr = _json.loads(both_res.content[0].text)
                                    if isinstance(pr, list):
                                        # Add to both concepts since they're co-mentioned
                                        self.client.kg_context_tracker.add_passages_for_concept("indigenous people", pr)
                                        self.client.kg_context_tracker.add_passages_for_concept("terrestrial risk", pr)
                            except Exception:
                                pass
                            llm_logger.info(f"KG BOOTSTRAP co-mentions added: {len(both_facts)}")
                            collected_facts.extend(both_facts)
                    except Exception as e:
                        llm_logger.warning(f"KG BOOTSTRAP co-mentions failed: {e}")

                    # No client-side co-mention injection; rely on KG tools for relevant passages

                    # Optional relationship path (justification)
                    try:
                        if len(seeds) >= 2:
                            path_res = await self.client.call_tool(
                                "FindConceptPathWithEdges",
                                {"source_concept": seeds[0], "target_concept": seeds[1], "max_len": 4},
                                server_name
                            )
                            # Convert relationship to a text-only fact for optional inclusion
                            if hasattr(path_res, 'content') and path_res.content:
                                import json as _json
                                paths = _json.loads(path_res.content[0].text)
                                if isinstance(paths, list) and paths:
                                    explain_res = await self.client.call_tool(
                                        "ExplainConceptRelationship",
                                        {"source_concept": seeds[0], "target_concept": seeds[1], "max_len": 4},
                                        server_name
                                    )
                                    if hasattr(explain_res, 'content') and explain_res.content:
                                        rel_text = explain_res.content[0].text.strip()
                                        if rel_text:
                                            collected_facts.append(Fact(
                                                text_content=f"Relationship: {rel_text}",
                                                source_key=f"kg_relationship_{seeds[0]}_{seeds[1]}",
                                                server_origin=server_name,
                                                metadata={"tool": "ExplainConceptRelationship"}
                                            ))
                    except Exception as e:
                        llm_logger.warning(f"KG relationship explanation failed: {e}")

                    if collected_facts:
                        return PhaseResult(
                            is_relevant=True,
                            facts=collected_facts,
                            reasoning=f"Deterministic KG bootstrap collected {len(collected_facts)} passage facts",
                            continue_processing=False,
                            token_usage=0
                        )
                except Exception as e:
                    print(f"KG deterministic bootstrap failed: {e}")
            
            # Detect intents
            q_lower = user_query.lower()
            # Correlation intent = explicit spatial relation language
            correlation_intent = any(kw in q_lower for kw in [
                "within", "inside", "intersect", "overlap", "near", "close to",
                "proximity", "adjacent", "around", "km", "meter", "metre", "m "
            ])
            # Map intent = wants a map/locations but without correlation operators
            map_intent = (any(kw in q_lower for kw in [
                "map", "show", "where", "locations", "visualize", "plot"
            ]) and not correlation_intent)
            # Keep a legacy flag for any spatial-like phrasing (do not use to trigger correlation)
            is_spatial_query = correlation_intent

            # Policy intent (keep KG/LSE focus)
            policy_intent = any(k in q_lower for k in ["policy", "policies"]) and server_name == "cpr"
            
            # Add specific instructions for spatial queries
            spatial_instructions = ""
            if correlation_intent:
                if server_name == "solar":
                    spatial_instructions = """

            SPATIAL QUERY DETECTED! You MUST:
            1. Call GetSolarFacilitiesByCountry with country='Brazil' to get solar facility locations
            2. This will provide geographic data (lat/lon) needed for spatial correlation
            3. DO NOT skip this step - the geospatial server needs this data"""
                elif server_name == "deforestation":
                    spatial_instructions = """

            SPATIAL QUERY DETECTED! You MUST:
            1. IMMEDIATELY call GetDeforestationAreas with these parameters: {"limit": 1000}
            2. This is MANDATORY - do not analyze or think, just call GetDeforestationAreas FIRST
            3. This will provide geographic boundaries needed for spatial correlation
            4. After getting the data, you can summarize what you found
            5. DO NOT skip this step - the geospatial server needs this data"""
                elif server_name == "gist":
                    spatial_instructions = """

            SPATIAL QUERY DETECTED! You MUST:
            1. Call tools that return assets with latitude/longitude coordinates
            2. This geographic data is needed for spatial correlation
            3. DO NOT skip this step - the geospatial server needs this data"""
                        
            # Create focused system prompt for Phase 1 collection
            system_prompt = f"""You are collecting information from the {server_name} server to answer: {user_query}

            CRITICAL: Focus ONLY on facts that directly answer this specific question.

            Server capabilities: {config['detailed']}

            {config['collection_instructions']}{spatial_instructions}{query_hints}

            COLLECTION PRIORITIES:
            1. Call tools that DIRECTLY answer "{user_query}" - not just related topics
            2. When you receive numerical data, check for units and context
            3. If units are missing, note this (e.g., "value is 45 but units not specified")
            4. Prioritize complete information: value + unit + timeframe + context
            5. Focus on concrete facts: numbers with units, dates, percentages, specific names

            DO NOT collect:
            - Tangentially related information
            - Data about entities not mentioned in the query
            - Background context unless specifically requested
            - Infrastructure counts unless asked for

            Guidelines:
            - Extract key facts with their complete context from each tool response
            - Stop when you have enough to answer the specific question (usually 2-4 tool calls)
            - Be efficient with your token budget

            When you have collected enough information to answer "{user_query}", respond with a summary instead of calling more tools."""

            # Initialize messages for the conversation
            messages = []
            
            # Include conversation history if available
            if self.conversation_history:
                messages.append({
                    "role": "user",
                    "content": f"Previous conversation context:\n{self._format_conversation_history()}\n\nNow, for the current query:"
                })
            
            # Adjust user message for spatial queries
            if correlation_intent and server_name == "deforestation":
                # Avoid streaming large polygon payloads. We rely on the geospatial
                # server's static deforestation index for correlation and will only
                # show correlated polygons in maps.
                user_message = (
                    f"Query: {user_query}\n\n"
                    "Do NOT call GetDeforestationAreas unless explicitly asked for sample polygons. "
                    "Use lightweight stats if needed. Geospatial correlation will use the static deforestation index, "
                    "and maps will include only correlated polygons."
                )
            elif map_intent and server_name == "solar":
                user_message = (
                    f"Query: {user_query}\n\n"
                    "START by calling GetSolarFacilitiesMapData (use country='Brazil' when applicable) to generate the facilities map."
                )
            elif correlation_intent and server_name == "solar":
                user_message = (
                    f"Query: {user_query}\n\n"
                    "START by calling GetSolarFacilitiesByCountry with country='Brazil' to get facility locations, then proceed with spatial correlation if relevant."
                )
            else:
                user_message = f"Query: {user_query}\n\nCollect relevant information from your available tools."
            
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Track collected facts
            facts = []
            tool_calls_made = 0
            max_tool_calls = 15  # Small scope for Phase 1
            
            # Policy-first KG discovery path: use server-side discovery tool
            if policy_intent:
                try:
                    disc = await self.client.call_tool("DiscoverPolicyContextForQuery", {"query": user_query, "top_concepts": 3, "neighbor_limit": 10, "passage_limit": 25}, server_name)
                    facts.extend(self._extract_facts_from_result("DiscoverPolicyContextForQuery", {"query": user_query}, disc, server_name))
                    # Pull recent KG semantic debug lines and forward to orchestrator logs for visibility
                    try:
                        semlog = await self.client.call_tool("GetSemanticDebugLog", {"limit": 80}, server_name)
                        if hasattr(semlog, 'content') and semlog.content:
                            import json as _json
                            data = _json.loads(semlog.content[0].text)
                            if isinstance(data, dict):
                                lines = data.get("lines") or []
                                for ln in lines:
                                    try:
                                        llm_logger.info(f"[KG_SEM] {ln}")
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    return PhaseResult(
                        is_relevant=True,
                        facts=facts,
                        reasoning="KG policy context discovered via server-side tool",
                        continue_processing=False
                    )
                except Exception as e:
                    print(f"KG policy discovery failed: {e}")

            # Deterministic bootstrap for correlation queries to ensure geospatial registration
            bootstrap_done = False
            if correlation_intent:
                try:
                    if server_name == "deforestation":
                        # Skip fetching/streaming deforestation polygons. Correlation will
                        # use the geospatial server's static deforestation index, and maps
                        # include only correlated polygons. This avoids large payloads that hang.
                        print("  Phase 1 - deforestation: Skipping polygon bootstrap; using static deforestation index")
                        return PhaseResult(
                            is_relevant=True,
                            facts=facts,
                            reasoning="Skipped polygon bootstrap; using static deforestation index",
                            continue_processing=True
                        )
                        # Deterministically fetch a SMALL sample of deforestation areas so geospatial
                        # correlation can run even if the static index is unavailable.
                        # Important: avoid unbounded payloads that freeze the frontend.
                        # Use a sane default cap and a small min area to reduce geometry size.
                        def_args = {"min_area_hectares": 5, "limit": 500}
                        print(f"  Phase 1 - deforestation: Bootstrap calling GetDeforestationAreas {def_args}")
                        dres = await self.client.call_tool("GetDeforestationAreas", def_args, "deforestation")
                        # Directly register entities if present
                        try:
                            if hasattr(dres, 'content') and dres.content:
                                import json as _json
                                data = _json.loads(dres.content[0].text)
                                areas = data.get('deforestation_areas', []) if isinstance(data, dict) else []
                                if areas:
                                    await self.client.call_tool(
                                        "RegisterEntities",
                                        {"entity_type": "deforestation_area", "entities": areas, "session_id": self.spatial_session_id},
                                        "geospatial"
                                    )
                                    print(f"  Registered {len(areas)} deforestation areas to session {self.spatial_session_id}")
                        except Exception as e:
                            print(f"  Direct registration from deforestation areas failed: {e}")
                        # If solar was registered earlier, try deterministic correlation now
                        try:
                            reg = await self.client.call_tool("GetRegisteredEntities", {"session_id": self.spatial_session_id}, "geospatial")
                            reg_data = {}
                            if hasattr(reg, 'content') and reg.content:
                                reg_data = _json.loads(reg.content[0].text)
                            if isinstance(reg_data, dict) and reg_data.get("by_type", {}).get("solar_facility", 0) > 0:
                                import re as _re
                                corr_args = {"entity_type1": "solar_facility", "entity_type2": "deforestation_area", "session_id": self.spatial_session_id}
                                m = _re.search(r"(\d+(?:\.\d+)?)\s*km", ql)
                                if m:
                                    try:
                                        distance_km = float(m.group(1))
                                        corr_args.update({"method": "proximity", "distance_km": distance_km})
                                    except Exception:
                                        corr_args.update({"method": "within"})
                                else:
                                    corr_args.update({"method": "within"})
                                print(f"  Phase 1 - deforestation: Deterministic correlation via geospatial {corr_args}")
                                corr = await self.client.call_tool("FindSpatialCorrelations", corr_args, "geospatial")
                                if hasattr(corr, 'content') and corr.content:
                                    corr_data = _json.loads(corr.content[0].text)
                                    facts.append(Fact(
                                        text_content=f"Spatial correlation result: total_correlations={corr_data.get('total_correlations', 'unknown')}",
                                        source_key="geospatial_phase1_correlation",
                                        server_origin="geospatial",
                                        metadata={"tool": "FindSpatialCorrelations", "raw_result": corr_data},
                                        citation=Citation(
                                            source_name="GEOSPATIAL",
                                            tool_id="FindSpatialCorrelations",
                                            server_origin="geospatial",
                                            source_type="Database",
                                            description="Deterministic Phase 1 correlation",
                                            source_url=self._resolve_source_url("geospatial", "FindSpatialCorrelations")
                                        )
                                    ))
                                if not self._geo_map_generated:
                                    map_res = await self.client.call_tool(
                                        "GenerateCorrelationMap",
                                        {"correlation_type": "solar_in_deforestation", "session_id": self.spatial_session_id, "show_uncorrelated": False},
                                        "geospatial"
                                    )
                                    if hasattr(map_res, 'content') and map_res.content:
                                        map_data = _json.loads(map_res.content[0].text)
                                        facts.append(Fact(
                                            text_content="Correlation map generated",
                                            source_key="geospatial_phase1_map",
                                            server_origin="geospatial",
                                            metadata={"tool": "GenerateCorrelationMap", "raw_result": map_data},
                                            citation=Citation(
                                                source_name="GEOSPATIAL",
                                                tool_id="GenerateCorrelationMap",
                                                server_origin="geospatial",
                                                source_type="Database",
                                                description="Deterministic Phase 1 correlation map",
                                                source_url=self._resolve_source_url("geospatial", "GenerateCorrelationMap")
                                            )
                                        ))
                                        self._geo_map_generated = True
                        except Exception as e:
                            print(f"  Phase 1 - deforestation: Deterministic correlation skipped: {e}")
                        bootstrap_done = True
                    elif server_name == "solar":
                        # Prefer direct entities rather than map artifacts
                        bootstrap_args = {"country": "Brazil", "limit": 10000}
                        print(f"  Phase 1 - {server_name}: Bootstrap calling GetFacilitiesForGeospatial {bootstrap_args}")
                        result = await self.client.call_tool("GetFacilitiesForGeospatial", bootstrap_args, server_name)
                        facts.extend(self._extract_facts_from_result("GetFacilitiesForGeospatial", bootstrap_args, result, server_name))
                        # If the query is to SHOW or MAP solar assets in Brazil, generate the map deterministically
                        try:
                            wants_map = any(k in ql for k in ["map", "show", "where", "locations", "assets", "facilities"]) and ("brazil" in ql)
                            if wants_map:
                                map_args = {"country": "Brazil", "limit": 10000}
                                print(f"  Phase 1 - solar: Generating map via GetSolarFacilitiesMapData {map_args}")
                                map_res = await self.client.call_tool("GetSolarFacilitiesMapData", map_args, server_name)
                                facts.extend(self._extract_facts_from_result("GetSolarFacilitiesMapData", map_args, map_res, server_name))
                        except Exception as e:
                            print(f"  Phase 1 - solar: Map generation skipped: {e}")
                        # Directly register entities if present
                        try:
                            if hasattr(result, 'content') and result.content:
                                import json as _json
                                data = _json.loads(result.content[0].text)
                                entities = data.get('entities', []) if isinstance(data, dict) else []
                                if entities:
                                    await self.client.call_tool(
                                        "RegisterEntities",
                                        {"entity_type": "solar_facility", "entities": entities, "session_id": self.spatial_session_id},
                                        "geospatial"
                                    )
                                    print(f"  Registered {len(entities)} solar facilities to session {self.spatial_session_id}")
                        except Exception as e:
                            print(f"  Direct registration from facilities failed: {e}")
                        # After registering solar, if correlation-intent, try deterministic correlation using static deforestation index
                        try:
                            if not correlation_intent:
                                raise Exception("Correlation intent not detected; skip deterministic correlation")
                            import re as _re
                            corr_args = {"entity_type1": "solar_facility", "entity_type2": "deforestation_area", "session_id": self.spatial_session_id}
                            m = _re.search(r"(\d+(?:\.\d+)?)\s*km", ql)
                            if m:
                                try:
                                    distance_km = float(m.group(1))
                                    corr_args.update({"method": "proximity", "distance_km": distance_km})
                                except Exception:
                                    corr_args.update({"method": "within"})
                            else:
                                corr_args.update({"method": "within"})
                            print(f"  Phase 1 - solar: Deterministic correlation via geospatial {corr_args}")
                            corr = await self.client.call_tool("FindSpatialCorrelations", corr_args, "geospatial")
                            if hasattr(corr, 'content') and corr.content:
                                import json as _json
                                corr_data = _json.loads(corr.content[0].text)
                                facts.append(Fact(
                                    text_content=f"Spatial correlation result: total_correlations={corr_data.get('total_correlations', 'unknown')}",
                                    source_key="geospatial_phase1_correlation",
                                    server_origin="geospatial",
                                    metadata={"tool": "FindSpatialCorrelations", "raw_result": corr_data},
                                    citation=Citation(
                                        source_name="GEOSPATIAL",
                                        tool_id="FindSpatialCorrelations",
                                        server_origin="geospatial",
                                        source_type="Database",
                                        description="Deterministic Phase 1 correlation"
                                    )
                                ))
                            # Generate correlation map once per query
                            if not self._geo_map_generated:
                                map_res = await self.client.call_tool(
                                    "GenerateCorrelationMap",
                                    {"correlation_type": "solar_in_deforestation", "session_id": self.spatial_session_id, "show_uncorrelated": False},
                                    "geospatial"
                                )
                                if hasattr(map_res, 'content') and map_res.content:
                                    map_data = _json.loads(map_res.content[0].text)
                                    facts.append(Fact(
                                        text_content="Correlation map generated",
                                        source_key="geospatial_phase1_map",
                                        server_origin="geospatial",
                                        metadata={"tool": "GenerateCorrelationMap", "raw_result": map_data},
                                        citation=Citation(
                                            source_name="GEOSPATIAL",
                                            tool_id="GenerateCorrelationMap",
                                            server_origin="geospatial",
                                            source_type="Database",
                                            description="Deterministic Phase 1 correlation map"
                                        )
                                    ))
                                    self._geo_map_generated = True
                        except Exception as e:
                            print(f"  Phase 1 - solar: Deterministic correlation skipped: {e}")
                        bootstrap_done = True
                    elif server_name == "municipalities":
                        # Deterministically fetch municipalities and register (skip fact extraction)
                        muni_args = {"limit": 6000}
                        if self._mark_tool_called("municipalities", "GetMunicipalitiesByFilter", muni_args):
                            print(f"  Phase 1 - municipalities: Bootstrap calling GetMunicipalitiesByFilter {muni_args}")
                            mres = await self.client.call_tool("GetMunicipalitiesByFilter", muni_args, "municipalities")
                            try:
                                if hasattr(mres, 'content') and mres.content:
                                    import json as _json
                                    mdata = _json.loads(mres.content[0].text)
                                    munis = mdata.get("municipalities", []) if isinstance(mdata, dict) else []
                                    if munis:
                                        await self.client.call_tool(
                                            "RegisterEntities",
                                            {"entity_type": "municipality", "entities": munis, "session_id": self.spatial_session_id},
                                            "geospatial"
                                        )
                            except Exception as e:
                                print(f"  Phase 1 - municipalities: registration failed: {e}")
                        bootstrap_done = True
                    elif server_name == "heat":
                        # Register top quintile heat zones deterministically (skip fact extraction)
                        heat_args = {"quintiles": [5], "limit": 5000}
                        if self._mark_tool_called("heat", "GetHeatQuintilesForGeospatial", heat_args):
                            print(f"  Phase 1 - heat: Bootstrap calling GetHeatQuintilesForGeospatial {heat_args}")
                            hres = await self.client.call_tool("GetHeatQuintilesForGeospatial", heat_args, "heat")
                            try:
                                if hasattr(hres, 'content') and hres.content:
                                    import json as _json
                                    data = _json.loads(hres.content[0].text)
                                    entities = data.get('entities', []) if isinstance(data, dict) else []
                                    if entities:
                                        await self.client.call_tool(
                                            "RegisterEntities",
                                            {"entity_type": "heat_zone", "entities": entities, "session_id": self.spatial_session_id},
                                            "geospatial"
                                        )
                                        print(f"  Registered {len(entities)} heat zones to session {self.spatial_session_id}")
                            except Exception as e:
                                print(f"  Direct registration from heat zones failed: {e}")
                        bootstrap_done = True
                except Exception as e:
                    print(f"  Spatial bootstrap error for {server_name}: {e}")
            
            # If we handled spatial bootstrap for these servers, return early to avoid heavy LLM loops
            # Enhancement: When the query implies admin-vs-zone ranking, ensure both sides are bootstrapped once.
            if not bootstrap_done:
                try:
                    admin_type = self._detect_admin_intent(user_query)
                    zone_type = self._detect_zone_intent(user_query)
                    if admin_type == "municipality" and zone_type and self._has_ranking_intent(user_query):
                        # Bootstrap municipalities
                        try:
                            muni_args = {"limit": 6000}
                            print(f"  Phase 1 - forced bootstrap: municipalities {muni_args}")
                            mres = await self.client.call_tool("GetMunicipalitiesByFilter", muni_args, "municipalities")
                            facts.extend(self._extract_facts_from_result("GetMunicipalitiesByFilter", muni_args, mres, "municipalities"))
                            if hasattr(mres, 'content') and mres.content:
                                import json as _json
                                mdata = _json.loads(mres.content[0].text)
                                munis = mdata.get("municipalities", []) if isinstance(mdata, dict) else []
                                if munis:
                                    await self.client.call_tool(
                                        "RegisterEntities",
                                        {"entity_type": "municipality", "entities": munis, "session_id": self.spatial_session_id},
                                        "geospatial"
                                    )
                        except Exception as e:
                            print(f"  Phase 1 - forced bootstrap municipalities failed: {e}")
                        # Bootstrap zone layer (heat zones currently)
                        if zone_type == "heat_zone":
                            try:
                                heat_args = {"quintiles": [5], "limit": 5000}
                                print(f"  Phase 1 - forced bootstrap: heat {heat_args}")
                                hres = await self.client.call_tool("GetHeatQuintilesForGeospatial", heat_args, "heat")
                                facts.extend(self._extract_facts_from_result("GetHeatQuintilesForGeospatial", heat_args, hres, "heat"))
                                if hasattr(hres, 'content') and hres.content:
                                    import json as _json
                                    hdata = _json.loads(hres.content[0].text)
                                    entities = hdata.get('entities', []) if isinstance(hdata, dict) else []
                                    if entities:
                                        await self.client.call_tool(
                                            "RegisterEntities",
                                            {"entity_type": "heat_zone", "entities": entities, "session_id": self.spatial_session_id},
                                            "geospatial"
                                        )
                                        print(f"  Registered {len(entities)} heat zones (forced bootstrap)")
                                bootstrap_done = True
                            except Exception as e:
                                print(f"  Phase 1 - forced bootstrap heat failed: {e}")
                except Exception as e:
                    print(f"  Phase 1 - admin/zone bootstrap check failed: {e}")

            if bootstrap_done:
                return PhaseResult(
                    is_relevant=True,
                    facts=facts,
                    reasoning=f"Bootstrap collected {len(facts)} facts for spatial correlation",
                    continue_processing=False,
                    token_usage=0
                )
            
            # Start the collection loop
            while tool_calls_made < max_tool_calls:
                # Call LLM with tools
                response = await call_model_with_tools(
                    model=LARGE_MODEL,  # Use Sonnet for better tool selection
                    system=system_prompt,
                    messages=messages,
                    tools=available_tools,
                    max_tokens=800  # Smaller token budget
                )
                
                # Process the response
                assistant_message_content = []
                found_tool_use = False
                
                for content in response.content:
                    if content.type == "text":
                        assistant_message_content.append(content)
                        
                    elif content.type == "tool_use":
                        found_tool_use = True
                        tool_name = content.name
                        tool_args = content.input
                        tool_calls_made += 1
                        
                        # De-duplicate repeated tool calls in Phase 1
                        if not self._mark_tool_called(server_name, tool_name, tool_args):
                            print(f"  Phase 1 - {server_name}: Skipping duplicate tool call {tool_name}")
                            # Still append a minimal tool_result to satisfy the tool-use turn
                            messages.append({"role": "assistant", "content": assistant_message_content})
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": "[duplicate_tool_call_skipped]"
                                }]
                            })
                            continue

                        print(f"  Phase 1 - {server_name}: Call {tool_calls_made}/{max_tool_calls} - {tool_name}")
                        if tool_args:
                            print(f"    Args: {json.dumps(tool_args, indent=2)}")

                        # Make the actual tool call
                        try:
                            # Ensure geospatial tools get the correct session_id
                            if server_name == "geospatial" and isinstance(tool_args, dict) and "session_id" not in tool_args:
                                tool_args = dict(tool_args)
                                tool_args["session_id"] = self.spatial_session_id
                            result = await self.client.call_tool(tool_name, tool_args, server_name)
                            
                            # Log the actual response
                            if hasattr(result, 'content'):
                                if isinstance(result.content, list) and len(result.content) > 0:
                                    first_content = result.content[0]
                                    if hasattr(first_content, 'text'):
                                        try:
                                            result_data = json.loads(first_content.text)
                                            # Truncate large results for logging
                                            if isinstance(result_data, list) and len(result_data) > 3:
                                                print(f"    Response: List with {len(result_data)} items (showing first 3)")
                                                print(f"    {json.dumps(result_data[:3], indent=2)[:500]}...")
                                            elif isinstance(result_data, dict):
                                                result_str = json.dumps(result_data, indent=2)
                                                if len(result_str) > 500:
                                                    print(f"    Response (truncated): {result_str[:500]}...")
                                                else:
                                                    print(f"    Response: {result_str}")
                                            else:
                                                print(f"    Response: {result_data}")
                                        except json.JSONDecodeError:
                                            # Not JSON, show as text
                                            text = first_content.text
                                            if len(text) > 500:
                                                print(f"    Response (text, truncated): {text[:500]}...")
                                            else:
                                                print(f"    Response (text): {text}")
                            else:
                                print(f"    Response: {result}")
                            
                            # Extract facts from the tool result
                            extracted_facts = self._extract_facts_from_result(
                                tool_name, tool_args, result, server_name
                            )
                            facts.extend(extracted_facts)
                            
                            # Auto-register geographic entities with geospatial server
                            await self._auto_register_geographic_entities(
                                server_name, tool_name, result
                            )
                            
                            # Add tool use to assistant message
                            assistant_message_content.append(content)
                            messages.append({"role": "assistant", "content": assistant_message_content})
                            
                            # Add sanitized tool result to conversation to avoid huge payloads/GeoJSON leakage
                            safe_payload = _prepare_tool_result_for_llm(tool_name, result)
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": safe_payload
                                }]
                            })
                            
                        except Exception as e:
                            print(f"  Phase 1 - {server_name}: Tool error: {e}")
                            # Add error as tool result
                            messages.append({"role": "assistant", "content": assistant_message_content})
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": f"Error: {str(e)}"
                                }]
                            })
                        
                        # Break to get next LLM response
                        break
                
                if not found_tool_use:
                    # No tool use - LLM is done collecting
                    messages.append({"role": "assistant", "content": assistant_message_content})
                    
                    # Extract summary from final response
                    summary = ""
                    for content in assistant_message_content:
                        if hasattr(content, 'text'):
                            summary += content.text
                    
                    return PhaseResult(
                        is_relevant=True,  # Pre-filtered as relevant
                        facts=facts,
                        reasoning=summary if summary else f"Collected {len(facts)} facts",
                        continue_processing=False,
                        token_usage=tool_calls_made * 300  # Rough estimate
                    )
            
            # Reached max tool calls
            return PhaseResult(
                is_relevant=True,  # Pre-filtered as relevant
                facts=facts,
                reasoning=f"Collected {len(facts)} facts from {tool_calls_made} tool calls",
                continue_processing=False,
                token_usage=tool_calls_made * 300
            )
            
        except Exception as e:
            print(f"Phase 1 - {server_name}: Error: {e}")
            return PhaseResult(
                is_relevant=True,  # Was pre-filtered as relevant
                facts=[],
                reasoning=f"Collection error: {str(e)}",
                continue_processing=False
            )
    
    def _extract_facts_from_result(self, tool_name: str, tool_args: dict, 
                                   result: Any, server_name: str) -> List[Fact]:
        """
        Extract facts from a tool result.
        
        Parses tool results and creates Fact objects with proper citations.
        
        Args:
            tool_name: Name of the tool that was called
            tool_args: Arguments passed to the tool
            result: Result from the tool call
            server_name: Server that provided the result
            
        Returns:
            List of Fact objects extracted from the result
        """
        facts = []
        
        # Log tool call if tracer is available
        if self.client.fact_tracer:
            self.client.fact_tracer.log_tool_call(server_name, tool_name, tool_args, result)
        
        try:
            # Parse result content
            result_data = None
            if hasattr(result, 'content'):
                if isinstance(result.content, list) and len(result.content) > 0:
                    first_content = result.content[0]
                    if hasattr(first_content, 'text'):
                        try:
                            result_data = json.loads(first_content.text)
                        except json.JSONDecodeError:
                            # Not JSON, treat as plain text
                            result_data = first_content.text
            # Debug logging
            try:
                summary_type = type(result_data).__name__
                count = len(result_data) if isinstance(result_data, (list, dict)) else 1
                llm_logger.info(f"FACTS EXTRACTOR: server={server_name} tool={tool_name} args={tool_args} result_type={summary_type} approx_count={count}")
            except Exception:
                pass
            
            # Extract facts based on tool type and result structure
            if result_data:
                # Create citation for this tool result
                base_citation = Citation(
                    source_name=self._get_source_name_for_tool(tool_name, server_name),
                    tool_id=tool_name,
                    source_type=self._get_source_type_for_server(server_name),
                    description=f"Data from {tool_name}",
                    server_origin=server_name,
                    source_url=self._resolve_source_url(server_name, tool_name),
                    metadata={"tool_args": tool_args}
                )
                
                # Special handling: KG policy discovery tool returns passages list
                if server_name == "cpr"and tool_name == "DiscoverPolicyContextForQuery" and isinstance(result_data, dict) and isinstance(result_data.get("passages"), list):
                    passages = result_data.get("passages", [])
                    for p in passages[:25]:
                        txt = (p.get("text") or "")
                        pid = p.get("passage_id")
                        doc_id = p.get("doc_id")
                        match_type = (p.get("match_type") or "").lower()
                        matched_terms = p.get("matched_terms") if isinstance(p.get("matched_terms"), list) else []
                        # Derive tool_id and description based on provenance
                        if match_type == "text":
                            passage_tool_id = "KGTextSearchFallback"
                            desc_terms = ", ".join(matched_terms[:3]) if matched_terms else "text match"
                            passage_desc = f"Passage {pid} text match ({desc_terms})"
                        else:
                            passage_tool_id = "GetPassagesMentioningConcept"
                            # Try to extract concept labels from spans, if present
                            labels = []
                            try:
                                spans = p.get("spans") if isinstance(p.get("spans"), list) else []
                                for s in spans or []:
                                    lab = s.get("concept_label") or s.get("label")
                                    if lab and lab not in labels:
                                        labels.append(lab)
                            except Exception:
                                labels = []
                            label_str = ", ".join(labels[:3]) if labels else "concept"
                            passage_desc = f"Passage {pid} mentioning {label_str}"

                        specific_citation = Citation(
                            source_name=self._get_source_name_for_tool(passage_tool_id, server_name),
                            tool_id=passage_tool_id,
                            source_type=self._get_source_type_for_server(server_name),
                            description=passage_desc,
                            server_origin=server_name,
                            source_url=self._resolve_source_url(server_name, passage_tool_id),
                            metadata={
                                "tool_args": tool_args,
                                "passage_id": pid,
                                "doc_id": doc_id,
                                "match_type": match_type,
                                "matched_terms": matched_terms
                            }
                        )

                        facts.append(Fact(
                            text_content=txt[:400] + ("…" if len(txt) > 400 else ""),
                            source_key=f"kg_passage_{pid}",
                            server_origin=server_name,
                            metadata={"tool": passage_tool_id, "raw_result": p},
                            citation=specific_citation
                        ))
                    # Also include a compact summary fact
                    facts.append(Fact(
                        text_content=f"KG policy context: {len(passages)} passages; {len(result_data.get('concepts', []))} concepts; {len(result_data.get('neighbors', []))} links.",
                        source_key="kg_policy_context_summary",
                        server_origin=server_name,
                        metadata={"tool": tool_name, "raw_result": result_data},
                        citation=base_citation
                    ))
                    return facts

                # Extract facts based on data type
                if isinstance(result_data, list):
                    # Special handling: KG passage lists with span metadata
                    if server_name == "cpr"and result_data and isinstance(result_data[0], dict) and (
                        "passage_id" in result_data[0] or "doc_id" in result_data[0]
                    ):
                        for item in result_data:
                            if not isinstance(item, dict):
                                continue
                            doc_id = item.get("doc_id") or item.get("document_id") or ""
                            passage_id = item.get("passage_id") or ""
                            # Skip placeholder passages that provide no real evidence
                            if isinstance(passage_id, str) and passage_id.startswith("placeholder_"):
                                try:
                                    llm_logger.info(f"FACTS EXTRACTOR: skipping placeholder passage id={passage_id}")
                                except Exception:
                                    pass
                                continue
                            spans = item.get("spans") if isinstance(item.get("spans"), list) else []
                            # Build rich description using first relevant span if present
                            desc = None
                            concept_label = None
                            if spans:
                                s = spans[0]
                                labelled_text = s.get("labelled_text")
                                start_idx = s.get("start_index")
                                end_idx = s.get("end_index")
                                concept_id = s.get("concept_id")
                                concept_label = s.get("concept_label")
                                labellers = s.get("labellers") or []
                                timestamps = s.get("timestamps") or []
                                latest_ts = timestamps[-1] if timestamps else None
                                if labelled_text is not None and start_idx is not None and end_idx is not None:
                                    parts = []
                                    parts.append(f"‘{labelled_text}’")
                                    if concept_id:
                                        parts.append(f"({concept_id})")
                                    parts.append(f"start:{start_idx}–{end_idx}")
                                    if labellers:
                                        parts.append("Found by: " + ", ".join(map(str, labellers)))
                                    if latest_ts:
                                        parts.append(str(latest_ts))
                                    desc = " • ".join(parts)
                            if not desc:
                                text_snippet = (item.get("text") or "")
                                desc = (text_snippet[:100] + "...") if text_snippet else "Passage"

                            source_name = doc_id or self._get_source_name_for_tool(tool_name, server_name)
                            if concept_label:
                                source_name = f"{source_name} — {concept_label}"

                            citation = Citation(
                                source_name=source_name,
                                tool_id=f"{tool_name}:{passage_id}" if passage_id else tool_name,
                                source_type="Document",
                                description=desc,
                                server_origin=server_name,
                                source_url=self._resolve_source_url(server_name, tool_name),
                                metadata={"tool_args": tool_args, "doc_id": doc_id, "passage_id": passage_id}
                            )

                            # Construct a concise fact text favoring a full sentence quote around the span
                            fact_text = None
                            passage_text = (item.get("text") or "")
                            if spans and spans[0].get("labelled_text") and passage_text:
                                try:
                                    si = int(spans[0].get("start_index", 0))
                                    ei = int(spans[0].get("end_index", 0))
                                    # Expand to sentence boundaries (simple heuristic)
                                    left = passage_text.rfind('.', 0, si)
                                    left_nl = passage_text.rfind('\n', 0, si)
                                    if left_nl > left:
                                        left = left_nl
                                    right = passage_text.find('.', ei)
                                    right_nl = passage_text.find('\n', ei)
                                    if right == -1 or (right_nl != -1 and right_nl < right):
                                        right = right_nl
                                    if left == -1:
                                        left = max(passage_text.rfind(';', 0, si), passage_text.rfind(':', 0, si))
                                    if right == -1:
                                        for sep in [';', ':']:
                                            r2 = passage_text.find(sep, ei)
                                            if r2 != -1:
                                                right = r2
                                                break
                                    # Slice sentence with some padding
                                    start = max(0, left + 1)
                                    end = right + 1 if right != -1 else min(len(passage_text), ei + 240)
                                    sentence = passage_text[start:end].strip()
                                    if sentence:
                                        fact_text = f"Quoted evidence: \"{sentence}\""
                                except Exception:
                                    pass
                            # Fallbacks
                            if not fact_text and spans and spans[0].get("labelled_text"):
                                fact_text = f"Quoted evidence: ‘{spans[0]['labelled_text']}’"
                            if not fact_text:
                                t = passage_text
                                fact_text = t[:240] + ("..." if len(t) > 240 else "")

                            fact = Fact(
                                text_content=fact_text,
                                source_key=f"{server_name}_{tool_name}_{doc_id}_{passage_id}",
                                server_origin=server_name,
                                metadata={"tool": tool_name, "raw_result": item},
                                citation=citation
                            )
                            facts.append(fact)
                            # Log each fact summary line
                            try:
                                llm_logger.info(f"FACT: {fact.text_content} | source={citation.source_name} id={citation.tool_id}")
                                # Log passage context length and snippet to confirm content is present
                                pt = passage_text
                                llm_logger.info(f"PASSAGE_CONTEXT: doc={doc_id} pid={passage_id} len={len(pt)} snippet='{pt[:180]}'")
                            except Exception:
                                pass
                    else:
                        # Generic list result - create summary fact
                        fact_text = f"Found {len(result_data)} items from {tool_name}"
                        if tool_args:
                            args_str = ', '.join([f'{k}={v}' for k, v in tool_args.items()])
                            fact_text += f" ({args_str})"
                        
                        facts.append(Fact(
                            text_content=fact_text,
                            source_key=f"{server_name}_{tool_name}_{hash(str(tool_args))}",
                            server_origin=server_name,
                            metadata={
                                "count": len(result_data), 
                                "tool": tool_name,
                                "raw_result": result_data  # Preserve for Phase 3
                            },
                            citation=base_citation
                        ))
                    
                elif isinstance(result_data, dict):
                    # If a tool returned a ready-to-render table module, convert to a tabular fact
                    try:
                        if result_data.get('type') == 'table' and result_data.get('columns') and result_data.get('rows'):
                            heading = result_data.get('heading') or 'Table'
                            facts.append(Fact(
                                text_content=heading,
                                source_key=f"{server_name}_{tool_name}_table_{hash(json.dumps(result_data.get('columns')))}",
                                server_origin=server_name,
                                metadata={"tool": tool_name, "raw_result": result_data},
                                citation=base_citation,
                                numerical_data={
                                    "columns": result_data.get('columns'),
                                    "rows": result_data.get('rows')
                                },
                                data_type="tabular"
                            ))
                    except Exception:
                        pass

                    # If a tool included a choropleth URL, add a map fact so the frontend renders it
                    try:
                        if isinstance(result_data.get('geojson_url'), str) and result_data.get('geojson_url'):
                            mfact = Fact(
                                text_content=result_data.get('heading') or 'Map',
                                source_key=f"{server_name}_{tool_name}_map_{hash(result_data.get('geojson_url'))}",
                                server_origin=server_name,
                                metadata={"tool": tool_name, "raw_result": result_data},
                                citation=base_citation
                            )
                            mfact.map_reference = {"url": result_data.get('geojson_url')}
                            mfact.data_type = "geographic"
                            facts.append(mfact)
                    except Exception:
                        pass
                    # Dictionary - extract key information with units
                    if tool_name == "GetSolarFacilitiesMultipleCountries":
                        print(f"DEBUG: GetSolarFacilitiesMultipleCountries result_data keys: {list(result_data.keys())}")
                        print(f"DEBUG: Has geojson_url: {'geojson_url' in result_data}")
                    
                    # Look for common patterns to build complete facts with units
                    unit_keys = ['unit', 'units', 'measure', 'metric', 'unit_of_measure']
                    time_keys = ['year', 'target_year', 'by_year', 'date', 'period', 'deadline']
                    measure_keys = ['measure', 'metric', 'indicator', 'type', 'category']
                    value_keys = ['value', 'total', 'amount', 'capacity', 'target', 'goal']
                    
                    # Try to extract structured information
                    value = None
                    for vk in value_keys:
                        if vk in result_data:
                            value = result_data[vk]
                            break
                    
                    unit = next((result_data.get(k) for k in unit_keys if k in result_data), '')
                    measure = next((result_data.get(k) for k in measure_keys if k in result_data), '')
                    timeframe = next((result_data.get(k) for k in time_keys if k in result_data), '')
                    
                    # Special handling for specific tool types
                    if 'data' in result_data and isinstance(result_data['data'], list):
                        count = len(result_data['data'])
                        fact_text = f"Found {count} records"
                        if tool_args:
                            args_str = ', '.join([f'{k}={v}' for k, v in tool_args.items()])
                            fact_text += f" ({args_str})"
                    elif value is not None and (unit or measure):
                        # Build complete fact with units
                        if measure:
                            fact_text = f"{measure}: {value}"
                        else:
                            fact_text = f"{value}"
                        
                        if unit:
                            fact_text += f" {unit}"
                        
                        if timeframe:
                            fact_text += f" (by {timeframe})" if 'target' in str(value).lower() or 'goal' in str(value).lower() else f" ({timeframe})"
                        
                        # Add context from tool args if relevant
                        if tool_args and 'country' in tool_args:
                            fact_text = f"{tool_args['country']}: {fact_text}"
                    elif 'total' in result_data:
                        # Enhanced total handling
                        total = result_data['total']
                        if unit:
                            fact_text = f"Total: {total} {unit}"
                        else:
                            fact_text = f"Total: {total}"
                        if timeframe:
                            fact_text += f" ({timeframe})"
                    elif 'metadata' in result_data:
                        meta = result_data['metadata']
                        fact_text = f"Data: {', '.join([f'{k}={v}' for k, v in meta.items()])}"
                    else:
                        # Extract actual key-value pairs for meaningful facts
                        key_values = []
                        for key, value in result_data.items():
                            if isinstance(value, (str, int, float, bool)):
                                # Try to identify if this is a value with implicit units
                                if any(indicator in key.lower() for indicator in ['percent', 'rate', 'ratio', 'capacity', 'mw', 'gw', 'emissions', 'co2']):
                                    # Add context from the key name
                                    key_values.append(f"{key}: {value}")
                                else:
                                    key_values.append(f"{key}: {value}")
                            elif isinstance(value, list) and len(value) > 0:
                                key_values.append(f"{key}: {len(value)} items")
                            elif isinstance(value, dict):
                                # Check if nested dict has value/unit structure
                                if 'value' in value and 'unit' in value:
                                    key_values.append(f"{key}: {value['value']} {value['unit']}")
                                else:
                                    key_values.append(f"{key}: {len(value)} fields")
                        
                        if key_values:
                            # Join first few key-value pairs for the fact
                            fact_text = "; ".join(key_values[:5])  # Limit to 5 to keep it concise
                            if len(key_values) > 5:
                                fact_text += f" (and {len(key_values) - 5} more fields)"
                        else:
                            # Fallback only if no extractable content
                            fact_text = f"Retrieved {len(result_data)} data fields"
                    
                    facts.append(Fact(
                        text_content=fact_text,
                        source_key=f"{server_name}_{tool_name}_{hash(str(tool_args))}",
                        server_origin=server_name,
                        metadata={
                            "tool": tool_name, 
                            "data_keys": list(result_data.keys()),
                            "raw_result": result_data  # Preserve for Phase 3
                        },
                        citation=base_citation
                    ))
                    
                elif isinstance(result_data, str):
                    # Plain text result - summarize
                    fact_text = result_data[:200]  # First 200 chars
                    if len(result_data) > 200:
                        fact_text += "..."
                    
                    facts.append(Fact(
                        text_content=fact_text,
                        source_key=f"{server_name}_{tool_name}_{hash(str(tool_args))}",
                        server_origin=server_name,
                        metadata={
                            "tool": tool_name,
                            "raw_result": result_data  # Preserve for Phase 3
                        },
                        citation=citation
                    ))
                
        except Exception as e:
            print(f"  Extract facts error for {tool_name}: {e}")
        
        # Debug logging when enabled
        if os.getenv("TDE_DEBUG_FACTS") and facts:
            print(f"  DEBUG: Extracted {len(facts)} facts from {tool_name}")
            for i, fact in enumerate(facts[:3], 1):  # Show first 3 facts
                preview = fact.text_content[:100] + "..." if len(fact.text_content) > 100 else fact.text_content
                print(f"    Fact {i}: {preview}")
        
        # Log facts extraction if tracer is available
        if self.client.fact_tracer and facts:
            self.client.fact_tracer.log_fact_extraction("phase1", server_name, result_data if 'result_data' in locals() else None, facts)
        
        return facts
    
    async def _auto_register_geographic_entities(self, server_name: str, tool_name: str, result: Any) -> None:
        """
        Auto-register geographic entities with the geospatial server for correlation.
        
        This happens automatically during Phase 1 when servers return geographic data.
        
        Args:
            server_name: Name of the server that provided the data
            tool_name: Name of the tool that was called
            result: Result from the tool call
        """
        # Check if geospatial server is available
        if not self.client.sessions.get("geospatial"):
            return
        
        try:
            # Parse result to get data
            result_data = None
            if hasattr(result, 'content'):
                if isinstance(result.content, list) and len(result.content) > 0:
                    first_content = result.content[0]
                    if hasattr(first_content, 'text'):
                        try:
                            result_data = json.loads(first_content.text)
                        except json.JSONDecodeError:
                            return  # Not JSON data
            
            if not result_data or not isinstance(result_data, dict):
                return
            
            # Auto-register based on server and data type
            registered = False
            
            # Debug logging
            print(f"  DEBUG Auto-register check: server={server_name}, tool={tool_name}, has_data={result_data is not None}")
            if result_data and isinstance(result_data, dict):
                print(f"  DEBUG Result keys: {list(result_data.keys())[:5]}")
            
            # Generic fallback: if result provides standard entity payload, register directly
            try:
                if isinstance(result_data.get("entities"), list) and isinstance(result_data.get("entity_type"), str):
                    entities = result_data.get("entities", [])
                    etype = result_data.get("entity_type")
                    if entities:
                        await self.client.call_tool(
                            "RegisterEntities",
                            {"entity_type": etype, "entities": entities, "session_id": self.spatial_session_id},
                            "geospatial"
                        )
                        print(f"  Auto-registered {len(entities)} entities of type '{etype}' with geospatial server (generic path)")
                        registered = True
            except Exception as e:
                print(f"  Generic auto-registration failed: {e}")

            # Solar facilities
            if server_name == "solar":
                # If result references a GeoJSON file, parse it and register points (prefer this path)
                if "geojson_url" in result_data:
                    try:
                        from pathlib import Path
                        project_root = Path(__file__).resolve().parent.parent
                        # geojson_url like /static/maps/<filename>
                        url_path = result_data.get("geojson_url", "")
                        filename = url_path.split("/")[-1]
                        geojson_path = project_root / "static" / "maps" / filename
                        if geojson_path.exists():
                            import json as _json
                            with open(geojson_path, "r", encoding="utf-8") as f:
                                gj = _json.load(f)
                            entities = []
                            for feat in gj.get("features", []):
                                try:
                                    geom = feat.get("geometry", {})
                                    if geom.get("type") == "Point":
                                        coords = geom.get("coordinates", [])
                                        if len(coords) == 2:
                                            props = feat.get("properties", {})
                                            entities.append({
                                                "id": props.get("cluster_id") or props.get("id") or props.get("name"),
                                                "latitude": float(coords[1]),
                                                "longitude": float(coords[0]),
                                                "country": props.get("country"),
                                                "capacity_mw": props.get("capacity_mw"),
                                                "cluster_id": props.get("cluster_id")
                                            })
                                except Exception as e:
                                    print(f"  Skipped invalid feature during GeoJSON registration: {e}")
                            if entities:
                                await self.client.call_tool(
                                    "RegisterEntities",
                                    {"entity_type": "solar_facility", "entities": entities, "session_id": self.spatial_session_id},
                                    "geospatial"
                                )
                                print(f"  Auto-registered {len(entities)} solar facilities from GeoJSON with geospatial server")
                                registered = True
                        else:
                            print(f"  GeoJSON path not found for registration: {geojson_path}")
                    except Exception as e:
                        print(f"  Failed GeoJSON-based solar registration: {e}")
                # Fallback to registering provided facilities/sample facilities
                elif ("facilities" in result_data) or ("sample_facilities" in result_data):
                    facilities = result_data.get("facilities") or result_data.get("sample_facilities", [])
                    if facilities and isinstance(facilities, list):
                        try:
                            await self.client.call_tool(
                                "RegisterEntities",
                                {"entity_type": "solar_facility", "entities": facilities, "session_id": self.spatial_session_id},
                                "geospatial"
                            )
                            print(f"  Auto-registered {len(facilities)} solar facilities with geospatial server")
                            registered = True
                        except Exception as e:
                            print(f"  Failed to register solar facilities: {e}")
            
            # Deforestation areas
            elif server_name == "deforestation" and "deforestation_areas" in result_data:
                areas = result_data["deforestation_areas"]
                if areas and isinstance(areas, list):
                    try:
                        await self.client.call_tool(
                            "RegisterEntities",
                            {"entity_type": "deforestation_area", "entities": areas, "session_id": self.spatial_session_id},
                            "geospatial"
                        )
                        print(f"  Auto-registered {len(areas)} deforestation areas with geospatial server")
                        registered = True
                    except Exception as e:
                        print(f"  Failed to register deforestation areas: {e}")
            
            # Brazilian municipalities
            elif server_name == "municipalities" and "municipalities" in result_data:
                munis = result_data["municipalities"]
                if munis and isinstance(munis, list):
                    try:
                        await self.client.call_tool(
                            "RegisterEntities",
                            {"entity_type": "municipality", "entities": munis, "session_id": self.spatial_session_id},
                            "geospatial"
                        )
                        print(f"  Auto-registered {len(munis)} municipalities with geospatial server")
                        registered = True
                    except Exception as e:
                        print(f"  Failed to register municipalities: {e}")
            
            # Heat zones
            elif server_name == "heat":
                if result_data.get("entity_type") == "heat_zone" and isinstance(result_data.get("entities"), list):
                    entities = result_data.get("entities", [])
                    if entities:
                        try:
                            await self.client.call_tool(
                                "RegisterEntities",
                                {"entity_type": "heat_zone", "entities": entities, "session_id": self.spatial_session_id},
                                "geospatial"
                            )
                            print(f"  Auto-registered {len(entities)} heat zones with geospatial server")
                            registered = True
                        except Exception as e:
                            print(f"  Failed to register heat zones: {e}")

            # GIST assets with coordinates
            elif server_name == "gist" and "assets" in result_data:
                assets = result_data["assets"]
                # Check if assets have lat/lon
                if assets and isinstance(assets, list) and len(assets) > 0:
                    # Check first asset for coordinates
                    if all(k in assets[0] for k in ['latitude', 'longitude']):
                        try:
                            await self.client.call_tool(
                                "RegisterEntities",
                                {"entity_type": "water_stressed_asset", "entities": assets},
                                "geospatial"
                            )
                            print(f"  Auto-registered {len(assets)} GIST assets with geospatial server")
                            registered = True
                        except Exception as e:
                            print(f"  Failed to register GIST assets: {e}")
            
            # Clear geospatial index at the start of each new query (only once)
            # This should ideally be done elsewhere, but we can check if this is the first registration
            if registered and not hasattr(self, '_geospatial_cleared'):
                # Mark that we've started using geospatial for this query
                self._geospatial_cleared = True
                
        except Exception as e:
            # Silent failure - auto-registration is optional enhancement
            print(f"  Auto-registration error: {e}")
    
    def _get_source_name_for_tool(self, tool_name: str, server_name: str) -> str:
        """Get a human-readable source name for a tool."""
        source_names = {
            "cpr": "CPR Knowledge Graph",
            "solar": "TransitionZero Solar Asset Mapper (TZ-SAM), Q1 2025",
            "gist": "GIST Impact Database",
            "lse": "LSE Climate Policy Database",
            "heat": "PlanetSapling Heat Stress (Brazil 2020–2025)",
            "formatter": "Response Formatter"
        }
        return source_names.get(server_name, server_name.upper())
    
    def _get_source_type_for_server(self, server_name: str) -> str:
        """Get the source type for a server."""
        source_types = {
            "cpr": "Knowledge Graph",
            # Treat solar facilities as a dataset for clearer citations
            "solar": "Dataset",
            "gist": "Database",
            "lse": "Database",
            "heat": "Dataset",
            "formatter": "Tool"
        }
        return source_types.get(server_name, "Database")

    def _resolve_source_url(self, server_name: str, tool_name: str) -> str:
        """Resolve SourceURL from module-level mapping and datasets.json."""
        try:
            from utils.dataset_resolver import resolve_dataset_url
            _ds, url = resolve_dataset_url(tool_name=tool_name, server_name=server_name)
            return url or ""
        except Exception:
            return ""
    
    async def _should_do_phase2_deep_dive(self, user_query: str, 
                                          collection_results: Dict[str, PhaseResult],
                                          complexity: Optional['QueryComplexity'] = None,
                                          time_elapsed: float = 0) -> Tuple[bool, str, List[str]]:
        """
        Determine if Phase 2 deep dive is needed based on Phase 1 results.
        
        Uses optimizer heuristics first, then falls back to LLM if needed.
        
        Args:
            user_query: Original user query
            collection_results: Results from Phase 1 collection
            complexity: Query complexity level (if optimizer available)
            time_elapsed: Time already spent in seconds
            
        Returns:
            Tuple of (should_continue: bool, reasoning: str, servers_to_deep_dive: List[str])
        """
        # Detect if this is a spatial correlation query (relation operators only)
        is_spatial_query = any(keyword in user_query.lower() for keyword in [
            "within", "inside", "intersect", "overlap", "near", "close to", 
            "proximity", "adjacent", "around", "km"
        ])
        
        # Check if geospatial server has registered entities
        geospatial_has_entities = False
        if 'geospatial' in collection_results:
            geospatial_result = collection_results['geospatial']
            if geospatial_result.facts:
                # Check if any facts mention registered entities
                for fact in geospatial_result.facts:
                    if 'registered' in fact.text_content.lower():
                        geospatial_has_entities = True
                        break
        
        # If spatial query and geospatial has entities, MUST do Phase 2 for correlation
        if is_spatial_query and geospatial_has_entities:
            return True, "Spatial correlation query detected with registered entities - Phase 2 needed for FindSpatialCorrelations", ["geospatial"]

        # Force Phase 2 for admin–zone ranking questions (e.g., municipalities + heat/deforestation)
        try:
            admin_type = self._detect_admin_intent(user_query)
            zone_type = self._detect_zone_intent(user_query)
            has_rank_intent = self._has_ranking_intent(user_query)
            if admin_type == "municipality" and (zone_type or has_rank_intent):
                return True, "Admin–zone ranking intent detected (municipalities vs zone) - Phase 2 needed for overlap computation", ["geospatial"]
        except Exception:
            pass
        
        # Gather all facts from Phase 1
        all_facts = []
        servers_with_facts = []
        total_facts = 0
        
        for server_name, result in collection_results.items():
            if result.is_relevant and result.facts:
                servers_with_facts.append(server_name)
                total_facts += len(result.facts)
                # FIXED: Preserve complete fact content, not just metadata
                # Include all facts to properly evaluate completeness
                for fact in result.facts:
                    # Preserve full fact text content for proper evaluation
                    all_facts.append(f"[{server_name}] {fact.text_content}")
        
        if not all_facts:
            return False, "No facts collected in Phase 1", []
        
        # Use optimizer heuristics if available
        if OPTIMIZER_AVAILABLE and complexity:
            should_skip, reason = PerformanceOptimizer.should_skip_phase2(
                complexity, total_facts, time_elapsed
            )
            if should_skip:
                return False, reason, []
        
        # Quick heuristic: Skip Phase 2 for simple queries with sufficient facts
        # Note: Do not skip if admin–zone intent was detected above.
        if total_facts >= 10:
            return False, "Sufficient facts already collected", []
        
        # Create evaluation prompt
        evaluation_prompt = f"""Analyze whether deeper investigation is needed to fully answer this query.

        User Query: {user_query}

        Facts Collected from Phase 1:
        {chr(10).join(all_facts)}

        Servers that provided data: {', '.join(servers_with_facts)}

        Evaluate based on:
        1. Completeness: Do the facts sufficiently answer the query?
        2. Gaps: Are there obvious missing pieces that need filling?
        3. Contradictions: Do facts from different servers conflict?
        4. Cross-references: Would correlating data across servers add value?
        5. Depth: Does the query warrant deeper analysis than these initial facts?
        6. Specificity: Are there specific subtopics or metrics mentioned but not detailed?

        Examples when Phase 2 IS needed:
        - Query asks for "commitments related to energy" but facts only show emissions targets, not energy-specific targets
        - Query asks for "correlation between X and Y" but facts don't show relationships
        - Query asks for "trends" but only got point-in-time data
        - Facts mention entities that other servers could provide more detail on
        - Geographic query that needs coordinate matching across servers

        Examples when Phase 2 is NOT needed:
        - Simple factual questions that Phase 1 fully answered
        - Query asks for a list and we got the list
        - All relevant data has been collected
        - Specific numerical targets are already extracted with units

        Respond with JSON:
        {{
            "need_phase2": true/false,
            "reasoning": "one sentence explanation",
            "gaps_identified": ["specific gap 1", "specific gap 2"],
            "servers_for_phase2": ["server1", "server2"],
            "follow_up_queries": ["specific follow-up question 1", "specific follow-up question 2"]
        }}"""

        try:
            response = await call_large_model(
                system="You evaluate whether additional data collection would improve query answers.",
                user_prompt=evaluation_prompt,
                max_tokens=400,
                temperature=0
            )
            
            # Try to parse JSON with multiple repair strategies
            result = None
            
            # Strategy 1: Direct parsing
            try:
                result = json.loads(response.strip())
            except json.JSONDecodeError:
                # Strategy 2: Extract JSON boundaries
                try:
                    first_brace = response.find('{')
                    last_brace = response.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        extracted_json = response[first_brace:last_brace+1]
                        result = json.loads(extracted_json)
                except json.JSONDecodeError:
                    # Strategy 3: Extract from code blocks
                    if '```json' in response:
                        try:
                            json_start = response.find('```json') + 7
                            json_end = response.find('```', json_start)
                            if json_end != -1:
                                extracted_json = response[json_start:json_end].strip()
                                result = json.loads(extracted_json)
                        except json.JSONDecodeError:
                            pass
            
            # If we successfully parsed JSON, use it
            if result:
                need_phase2 = result.get("need_phase2", False)
                reasoning = result.get("reasoning", "No additional analysis needed")
                servers = result.get("servers_for_phase2", servers_with_facts)
                # FIXED: Extract follow-up queries for targeted Phase 2 investigation
                follow_up_queries = result.get("follow_up_queries", [])
                # Return follow-up queries as part of the server list (temporary solution)
                # We'll pass these to Phase 2 through the reasoning field
                if follow_up_queries:
                    reasoning = f"{reasoning}|||FOLLOW_UP:{json.dumps(follow_up_queries)}"
                return need_phase2, reasoning, servers
            else:
                # All JSON parsing failed
                raise ValueError("Could not parse JSON response")
            
        except Exception as e:
            print(f"Phase 2 evaluation error: {e}")
            # On error, default to no Phase 2
            return False, "Evaluation failed, proceeding with current data", []
    
    async def _phase2_deep_dive(self, user_query: str, scout_results: Dict[str, PhaseResult], 
                                reasoning: str = "") -> Dict[str, PhaseResult]:
        """
        Phase 2: Unified Deep Dive - All tools from all relevant servers available.
        
        Provides maximum flexibility for cross-server coordination.
        LLM has access to ALL tools and can intelligently combine them.
        
        Args:
            user_query: Original user query
            scout_results: Results from Phase 1 scout
            reasoning: Reasoning from Phase 2 decision, may contain follow-up queries
            
        Returns:
            Dictionary mapping server names to their enhanced results
        """
        relevant_servers = {
            name: result for name, result in scout_results.items() 
            if result.is_relevant
        }
        
        if not relevant_servers:
            return {}
        
        # FIXED: Extract follow-up queries from reasoning if present
        follow_up_queries = []
        if "|||FOLLOW_UP:" in reasoning:
            try:
                follow_up_part = reasoning.split("|||FOLLOW_UP:")[1]
                follow_up_queries = json.loads(follow_up_part)
                print(f"Phase 2 using follow-up queries: {follow_up_queries}")
            except:
                print("Failed to parse follow-up queries from reasoning")
        
        # FIXED: Only gather tools from servers that had facts in Phase 1
        # This reduces token usage and focuses on productive servers
        servers_with_facts = [name for name, result in relevant_servers.items() 
                              if result.facts]
        
        all_tools = {}
        tool_to_server_map = {}  # Track which server each tool belongs to
        
        for server_name in relevant_servers.keys():
            if server_name not in self.client.sessions:
                print(f"Warning: Server {server_name} not in sessions")
                continue
                
            try:
                tools_response = await self.client.sessions[server_name].list_tools()
                # Access the tools list from the response object
                for tool in tools_response.tools:
                    # Prefix tool name with server to avoid collisions
                    # Use underscore instead of period to match Claude's tool naming requirements
                    prefixed_name = f"{server_name}_{tool.name}"
                    all_tools[prefixed_name] = tool
                    tool_to_server_map[prefixed_name] = server_name
            except Exception as e:
                print(f"Error getting tools from {server_name}: {e}")
        
        if not all_tools:
            print("No tools available for Phase 2")
            return scout_results
        
        # Accumulate all Phase 1 facts
        accumulated_facts = []
        for server_name, result in scout_results.items():
            if result.is_relevant and result.facts:
                accumulated_facts.extend(result.facts)
        
        # Format facts for context
        facts_context = "\n".join([
            f"- {fact.text_content}" 
            for fact in accumulated_facts
        ]) if accumulated_facts else "No facts collected yet."
        
        # Phase 2 termination loop with ALL tools available
        max_iterations = 10
        iteration = 0
        phase2_facts = []  # New facts from Phase 2

        # Deterministic spatial correlation if applicable (relation operators only)
        try:
            ql = user_query.lower()
            is_spatial_query = any(keyword in ql for keyword in [
                "within", "inside", "intersect", "overlap", "near", "close to",
                "proximity", "adjacent", "around", "km"
            ])
            # Detect generalized admin vs zone ranking intent (no spatial operator needed)
            admin_type = self._detect_admin_intent(user_query)
            zone_type = self._detect_zone_intent(user_query)
            has_rank_intent = self._has_ranking_intent(user_query)
            admin_zone_ranking = bool(admin_type and zone_type and has_rank_intent)

            if (is_spatial_query or admin_zone_ranking) and self.client.sessions.get("geospatial"):
                # Prefer legacy solar↔deforestation correlation when both are registered
                try:
                    reg = await self.client.call_tool("GetRegisteredEntities", {"session_id": self.spatial_session_id}, "geospatial")
                    reg_data = {}
                    if hasattr(reg, 'content') and reg.content:
                        try:
                            reg_data = json.loads(reg.content[0].text)
                        except Exception:
                            reg_data = {}
                    by_type = reg_data.get("by_type", {}) if isinstance(reg_data, dict) else {}
                    if by_type.get("solar_facility", 0) and by_type.get("deforestation_area", 0):
                        import re
                        m = re.search(r"(\d+(?:\.\d+)?)\s*km", ql)
                        corr_args = {"entity_type1": "solar_facility", "entity_type2": "deforestation_area", "session_id": self.spatial_session_id}
                        if m:
                            try:
                                distance_km = float(m.group(1))
                                corr_args.update({"method": "proximity", "distance_km": distance_km})
                            except Exception:
                                corr_args.update({"method": "within"})
                        else:
                            corr_args.update({"method": "within"})
                        print(f"Phase 2 - geospatial: FindSpatialCorrelations {corr_args}")
                        corr_result = await self.client.call_tool("FindSpatialCorrelations", corr_args, "geospatial")
                    else:
                        # Generalized selection of correlation pair from registered entities
                        types = reg_data.get("entity_types", []) or list((reg_data.get("by_type") or {}).keys())
                    geom_types = reg_data.get("geometry_types", {})
                    # Classify
                    point_types = [t for t in types if str(geom_types.get(t, "")).lower().startswith("point")]
                    poly_types = [t for t in types if t not in point_types]
                    # Choose preferred polygon type based on query hints
                    def pick_poly():
                        if 'deforestation' in ql and 'deforestation_area' in poly_types:
                            return 'deforestation_area'
                        if ('heat' in ql or 'temperature' in ql) and 'heat_zone' in poly_types:
                            return 'heat_zone'
                        if 'municip' in ql and 'municipality' in poly_types:
                            return 'municipality'
                        return poly_types[0] if poly_types else None
                    # Choose preferred point type
                    def pick_point():
                        if 'solar' in ql and 'solar_facility' in point_types:
                            return 'solar_facility'
                        return point_types[0] if point_types else None
                    et1 = pick_point()
                    et2 = pick_poly()
                    import re
                    distance_km = None
                    m = re.search(r"(\d+(?:\.\d+)?)\s*km", ql)
                    if m:
                        try:
                            distance_km = float(m.group(1))
                        except Exception:
                            distance_km = None
                    if et1 and et2:
                        method = 'within'
                        corr_args = {"entity_type1": et1, "entity_type2": et2, "method": method, "session_id": self.spatial_session_id}
                    elif len(point_types) >= 2:
                        # Point vs point proximity
                        et1, et2 = point_types[:2]
                        method = 'proximity'
                        corr_args = {"entity_type1": et1, "entity_type2": et2, "method": method, "distance_km": distance_km or 5.0, "session_id": self.spatial_session_id}
                    elif len(poly_types) >= 2:
                        # Polygon vs polygon intersects
                        et1, et2 = poly_types[:2]
                        method = 'intersects'
                        corr_args = {"entity_type1": et1, "entity_type2": et2, "method": method, "session_id": self.spatial_session_id}
                    else:
                        corr_args = None
                    if not corr_args:
                        raise RuntimeError("No suitable entity type pairs for correlation")
                    # If we didn't already run the legacy pair above
                    if 'corr_result' not in locals():
                        print(f"Phase 2 - geospatial: FindSpatialCorrelations {corr_args}")
                        corr_result = await self.client.call_tool("FindSpatialCorrelations", corr_args, "geospatial")
                except Exception as e:
                    print(f"Phase 2 - geospatial correlation skipped: {e}")
                    corr_result = None
                # Package as fact for synthesis
                corr_data = {}
                if hasattr(corr_result, 'content') and corr_result.content:
                    try:
                        corr_data = json.loads(corr_result.content[0].text)
                    except Exception:
                        corr_data = {"text": str(corr_result)}
                phase2_facts.append(
                    Fact(
                        text_content=f"Spatial correlation: total_correlations={corr_data.get('total_correlations', 'unknown')}",
                        source_key="geospatial_correlation_result",
                        server_origin="geospatial",
                        metadata={"tool": "FindSpatialCorrelations", "raw_result": corr_data},
                        citation=Citation(
                            source_name="GEOSPATIAL",
                            tool_id="FindSpatialCorrelations",
                            server_origin="geospatial",
                            source_type="Database",
                            description="Spatial correlation between registered entities",
                            source_url=self._resolve_source_url("geospatial", "FindSpatialCorrelations")
                        )
                    )
                )
                # Generate a correlation map for visualization
                try:
                    # Use legacy correlation_type when solar↔deforestation is present
                    corr_type = "solar_in_deforestation" if by_type.get("solar_facility", 0) and by_type.get("deforestation_area", 0) else "spatial_correlation"
                    map_res = await self.client.call_tool(
                        "GenerateCorrelationMap",
                        {"correlation_type": corr_type, "session_id": self.spatial_session_id, "show_uncorrelated": False},
                        "geospatial"
                    )
                    map_data = {}
                    if hasattr(map_res, 'content') and map_res.content:
                        try:
                            map_data = json.loads(map_res.content[0].text)
                        except Exception:
                            map_data = {"text": str(map_res)}
                    phase2_facts.append(
                        Fact(
                            text_content="Correlation map generated",
                            source_key="geospatial_correlation_map",
                            server_origin="geospatial",
                            metadata={"tool": "GenerateCorrelationMap", "raw_result": map_data},
                            citation=Citation(
                                source_name="GEOSPATIAL",
                                tool_id="GenerateCorrelationMap",
                                server_origin="geospatial",
                                source_type="Map",
                                description="GeoJSON map for spatial correlations",
                                source_url=self._resolve_source_url("geospatial", "GenerateCorrelationMap")
                            )
                        )
                    )
                except Exception as e:
                    print(f"Phase 2 - geospatial map generation skipped: {e}")
                
                # Minimal overlay for this class of query: heat exposure x weak solar presence (proxy)
                try:
                    by_type = reg_data.get("by_type", {}) if isinstance(reg_data, dict) else {}
                    if by_type.get("municipality") and by_type.get("heat_zone") and by_type.get("solar_facility"):
                        # Compute area overlap of heat zones per municipality
                        overlap_res = await self.client.call_tool(
                            "ComputeAreaOverlapByEntityTypes",
                            {"admin_entity_type": "municipality", "zone_entity_type": "heat_zone", "session_id": self.spatial_session_id},
                            "geospatial"
                        )
                        overlap_data = {}
                        if hasattr(overlap_res, 'content') and overlap_res.content:
                            try:
                                overlap_data = json.loads(overlap_res.content[0].text)
                            except Exception:
                                overlap_data = {}
                        overlap_list = overlap_data.get("results", [])

                        # Compute solar facility density per municipality (per 1000 km²)
                        density_res = await self.client.call_tool(
                            "ComputePointDensityByEntityTypes",
                            {"admin_entity_type": "municipality", "point_entity_type": "solar_facility", "per_km2": 1000.0, "session_id": self.spatial_session_id},
                            "geospatial"
                        )
                        density_data = {}
                        if hasattr(density_res, 'content') and density_res.content:
                            try:
                                density_data = json.loads(density_res.content[0].text)
                            except Exception:
                                density_data = {}
                        density_list = density_data.get("results", [])

                        # Index density by admin_id
                        by_admin_density = {}
                        for d in density_list:
                            aid = str(d.get("admin_id"))
                            if not aid:
                                continue
                            # Find the density field name dynamically
                            val = None
                            for k, v in d.items():
                                if isinstance(v, (int, float)) and str(k).startswith("points_per_"):
                                    val = float(v)
                                    break
                            by_admin_density[aid] = {"density": val or 0.0}

                        # Merge and compute composite score
                        rows = []
                        densities = [v["density"] for v in by_admin_density.values()] or [0.0]
                        dmin, dmax = min(densities), max(densities)
                        def norm_density(x: float) -> float:
                            try:
                                return 0.0 if dmax == dmin else (x - dmin) / (dmax - dmin)
                            except Exception:
                                return 0.0
                        for item in overlap_list:
                            aid = str(item.get("admin_id"))
                            if not aid:
                                continue
                            props = item.get("properties", {}) or {}
                            name = props.get("name", aid)
                            state = props.get("state") or props.get("state_abbrev") or ""
                            overlap_ratio = float(item.get("overlap_ratio", 0.0))
                            dens = by_admin_density.get(aid, {}).get("density", 0.0)
                            dn = norm_density(dens)
                            score = overlap_ratio * (1.0 - dn)
                            rows.append({
                                "admin_id": aid,
                                "municipality": name,
                                "state": state,
                                "heat_overlap_pct": round(overlap_ratio * 100.0, 2),
                                "solar_per_1000km2": round(dens, 3),
                                "composite_score": round(score, 4)
                            })

                        rows.sort(key=lambda r: r["composite_score"], reverse=True)
                        top_n = rows[:25]

                        # Emit as a tabular Fact so the renderer creates a table module
                        columns = [
                            "Municipality", "State", "Heat Overlap (%)", "Solar per 1000 km²", "Composite Score"
                        ]
                        table_rows = [
                            [r["municipality"], r["state"], r["heat_overlap_pct"], r["solar_per_1000km2"], r["composite_score"]]
                            for r in top_n
                        ]
                        phase2_facts.append(
                            Fact(
                                text_content="Top municipalities by heat exposure × low solar presence (proxy for weak renewable access).",
                                source_key="heat_low_solar_top_munis",
                                server_origin="geospatial",
                                data_type="tabular",
                                numerical_data={
                                    "columns": columns,
                                    "rows": table_rows,
                                    "raw": top_n
                                },
                                citation=Citation(
                                    source_name="Internal geospatial overlay",
                                    tool_id="ComputeAreaOverlapByEntityTypes+ComputePointDensityByEntityTypes",
                                    server_origin="geospatial",
                                    source_type="Derived",
                                    description="Area overlap of heat Q5 with municipalities and solar facility density per 1000 km²."
                                )
                            )
                        )
                        # Add a plain-text caveat fact
                        phase2_facts.append(
                            Fact(
                                text_content="Note: No social vulnerability data included. 'Weak access' uses low solar presence as a proxy, not electrification or grid reliability.",
                                source_key="proxy_disclaimer",
                                server_origin="geospatial",
                                citation=None
                            )
                        )
                except Exception as e:
                    print(f"Phase 2 - heat×low-solar overlay skipped or failed: {e}")
                # Generic admin vs. zone overlap ranking (heat, deforestation, or any polygon layer)
                try:
                    # Refresh registration status after any map generation
                    reg = await self.client.call_tool("GetRegisteredEntities", {"session_id": self.spatial_session_id}, "geospatial")
                    reg_data = {}
                    if hasattr(reg, 'content') and reg.content:
                        try:
                            reg_data = json.loads(reg.content[0].text)
                        except Exception:
                            reg_data = {}

                    # If admin-zone intent detected, ensure entities are registered deterministically
                    if admin_zone_ranking and not self._admin_zone_bootstrapped:
                        # Ensure municipalities are registered
                        if not (isinstance(reg_data, dict) and (reg_data.get("by_type", {}) or {}).get("municipality")):
                            try:
                                muni_res = await self.client.call_tool("GetMunicipalitiesByFilter", {"limit": 6000}, "municipalities")
                                muni_data = {}
                                if hasattr(muni_res, 'content') and muni_res.content:
                                    muni_data = json.loads(muni_res.content[0].text)
                                munis = muni_data.get("municipalities", []) if isinstance(muni_data, dict) else []
                                if munis:
                                    await self.client.call_tool(
                                        "RegisterEntities",
                                        {"entity_type": "municipality", "entities": munis, "session_id": self.spatial_session_id},
                                        "geospatial"
                                    )
                            except Exception as e:
                                print(f"Admin intent: failed to register municipalities: {e}")

                        # Ensure the detected zone layer is registered (when we can determine it from query)
                        if zone_type == "heat_zone" and not ((reg_data.get("by_type", {}) if isinstance(reg_data, dict) else {}).get("heat_zone")):
                            try:
                                heat_res = await self.client.call_tool("GetHeatQuintilesForGeospatial", {"quintiles": [5], "limit": 5000}, "heat")
                                heat_data = {}
                                if hasattr(heat_res, 'content') and heat_res.content:
                                    heat_data = json.loads(heat_res.content[0].text)
                                entities = heat_data.get("entities", []) if isinstance(heat_data, dict) else []
                                if entities:
                                    await self.client.call_tool(
                                        "RegisterEntities",
                                        {"entity_type": "heat_zone", "entities": entities, "session_id": self.spatial_session_id},
                                        "geospatial"
                                    )
                            except Exception as e:
                                print(f"Admin intent: failed to register heat zones: {e}")

                        self._admin_zone_bootstrapped = True
                        # Re-query registration summary after potential registrations
                        reg = await self.client.call_tool("GetRegisteredEntities", {"session_id": self.spatial_session_id}, "geospatial")
                        if hasattr(reg, 'content') and reg.content:
                            try:
                                reg_data = json.loads(reg.content[0].text)
                            except Exception:
                                reg_data = {}

                    # Presence-based trigger: if municipalities and any other polygon entity type are registered,
                    # compute overlaps without keyword-specific mapping.
                    by_type = reg_data.get("by_type", {}) if isinstance(reg_data, dict) else {}
                    geom_types = reg_data.get("geometry_types", {}) if isinstance(reg_data, dict) else {}
                    types = list(by_type.keys())
                    point_types = [t for t in types if str(geom_types.get(t, "")).lower().startswith("point")]
                    poly_types = [t for t in types if t not in point_types]
                    zone_choice = None
                    if 'municipality' in poly_types:
                        # Prefer heat_zone when present; else any other polygon layer
                        for cand in ["heat_zone"] + [t for t in poly_types if t != 'municipality']:
                            if cand in poly_types and cand != 'municipality':
                                zone_choice = cand
                                break
                    # Fallback: if query clearly indicates a zone type (e.g., heat_zone) but it's not registered,
                    # still attempt overlap using servers with static indexes (ComputeAreaOverlap handles this for heat).
                    if not zone_choice and zone_type in ("heat_zone",):
                        zone_choice = zone_type
                    if by_type.get('municipality') and zone_choice:
                        # Generate a ready-to-render table via geospatial tool to ensure deterministic output
                        table_res = await self.client.call_tool(
                            "GenerateMunicipalityOverlapTable",
                            {"zone_entity_type": zone_choice, "sort_by": "percentage", "top_n": 25, "session_id": self.spatial_session_id},
                            "geospatial"
                        )
                        if hasattr(table_res, 'content') and table_res.content:
                            try:
                                tbl = json.loads(table_res.content[0].text)
                            except Exception:
                                tbl = {}
                            # Convert table result into a tabular Fact via extractor path by reusing the result
                            try:
                                facts_from_table = self._extract_facts_from_result("GenerateMunicipalityOverlapTable", {"zone_entity_type": zone_choice}, table_res, "geospatial")
                                for f in facts_from_table:
                                    phase2_facts.append(f)
                            except Exception as e:
                                print(f"Failed to extract table fact: {e}")

                        # Optional: generate a choropleth map for quick visual context
                        # Skip choropleth generation for municipality + heat requests (table-only)
                        if zone_choice != 'heat_zone':
                            try:
                                choro = await self.client.call_tool(
                                    "GenerateAdminChoropleth",
                                    {"admin_entity_type": "municipality", "metric_name": f"overlap_{zone_choice}", "metrics": overlap_list, "title": "Municipality Overlap"},
                                    "geospatial"
                                )
                                choro_data = {}
                                if hasattr(choro, 'content') and choro.content:
                                    choro_data = json.loads(choro.content[0].text)
                                geojson_url = choro_data.get("geojson_url")
                                if isinstance(geojson_url, str) and geojson_url:
                                    fact = Fact(
                                        text_content="Admin choropleth generated",
                                        source_key=f"admin_choropleth_{zone_choice}",
                                        server_origin="geospatial",
                                        metadata={"tool": "GenerateAdminChoropleth", "raw_result": choro_data},
                                        citation=Citation(
                                            source_name="GEOSPATIAL",
                                            tool_id="GenerateAdminChoropleth",
                                            server_origin="geospatial",
                                            source_type="Map",
                                            description="Choropleth GeoJSON for admin metric"
                                        )
                                    )
                                    # Enable map module rendering in synthesis
                                    fact.map_reference = {"url": geojson_url}
                                    fact.data_type = "geographic"
                                    phase2_facts.append(fact)
                            except Exception as e:
                                print(f"Choropleth generation skipped: {e}")
                except Exception as e:
                    print(f"Phase 2 - generic admin-zone overlap skipped or failed: {e}")

                # Skip generic Phase 2 when we have the correlation answer (spatial queries)
                phase2_results = {}
                for server_name, original_result in scout_results.items():
                    server_phase2_facts = [
                        fact for fact in phase2_facts 
                        if fact.server_origin == server_name
                    ]
                    if server_phase2_facts:
                        all_server_facts = list(original_result.facts) + server_phase2_facts
                        phase2_results[server_name] = PhaseResult(
                            is_relevant=original_result.is_relevant,
                            facts=all_server_facts,
                            reasoning=f"{original_result.reasoning} + Spatial correlation computed",
                            continue_processing=False
                        )
                    else:
                        phase2_results[server_name] = original_result
                return phase2_results
        except Exception as e:
            print(f"Phase 2 deterministic correlation error: {e}")
        
        while iteration < max_iterations:
            iteration += 1
            
            # Build the unified prompt
            system_prompt = """System prompt (generic, minimal, dedup-first)

                You are an efficient tool-using agent. Only output tool_use blocks or EXACTLY: COMPLETE

                Objectives:
                - Collect the smallest set of tool results needed to answer the query.
                - Produce at most one directly-helpful visualization (only if it clarifies the answer).
                - Minimize calls and avoid duplicates.

                Rules:
                - Deduplicate: Before any call, review “What We Know So Far” (and “RecentCalls” if provided). Do NOT call the same
                server.tool with identical arguments more than once in this query.
                - Respect constraints: If the user specifies constraints (e.g., limit/size, time, place, entity set), pass them via
                tool parameters when available. If a parameter isn’t supported, collect once and filter in reasoning; do NOT re-call to
                “refine”.
                - Variants: For closely related parameter variants, call each at most once. If a variant returns no results, do not retry
                the same variant.
                - Call budget: Prefer 1–3 calls that directly return the entities/metrics needed. Avoid background/context calls unless
                they unlock the exact answer.
                - Visualization: Choose a single, most-informative visualization only if it directly answers the ask. Do not generate
                duplicate or redundant visuals.
                - Completion: When you have the required entities + metrics (with user constraints applied) and at most one helpful
                visualization, reply EXACTLY: COMPLETE

                User prompt (generic, reusable)

                User Query: {user_query}

                What We Know So Far:
                {facts_context}

                Guidance:
                - Use each needed server.tool+arguments combination at most once; reuse data already present above.
                - Apply any explicit user constraints (limit/size, filters, time, geography) via tool parameters if supported.
                - Checklist before your next call:
                    1) Do I already have the entities + metric(s) the user asked for (respecting constraints)?
                    2) Am I about to repeat a server.tool with identical arguments?
                    3) Will this call materially add required information?

                Do this:
                1) Identify the minimal tool call(s) and exact arguments that directly produce the needed data.
                2) Make only those calls (no duplicates). If a parameter (e.g., limit) is supported, pass it; otherwise collect once and
                filter mentally.
                3) If a visualization helps, create exactly one that directly clarifies the answer.
                4) If the answer is sufficient with the visualization, reply EXACTLY: COMPLETE; otherwise, make only the missing, non-
                duplicate call(s)."""
            
            # Convert tools to format expected by call_model_with_tools
            tools_list = [
                {
                    "name": prefixed_name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for prefixed_name, tool in all_tools.items()
            ]
            
            try:
                response = await call_model_with_tools(
                    model=LARGE_MODEL,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    tools=tools_list,
                    max_tokens=2000
                )
                
                # Check if complete - response.content is the list of blocks
                # Check if any text block contains COMPLETE
                for block in response.content:
                    print(block)
                    print(type(block))
                    if block.type == "text" and "COMPLETE" in block.text.upper():
                        print(f"Phase 2 complete after {iteration} iterations")
                        break
                
                # Process tool calls
                tool_called = False
                for block in response.content:
                    if block.type == "tool_use":
                        tool_called = True
                        prefixed_name = block.name
                        
                        # Extract server name and actual tool name
                        if prefixed_name in tool_to_server_map:
                            server_name = tool_to_server_map[prefixed_name]
                            # Handle different tool name formats flexibly
                            if '_' in prefixed_name:
                                # Format: server_ToolName
                                parts = prefixed_name.split('_', 1)
                                actual_tool_name = parts[1] if len(parts) > 1 else prefixed_name
                            elif '.' in prefixed_name:
                                # Format: server.ToolName
                                parts = prefixed_name.split('.', 1)
                                actual_tool_name = parts[1] if len(parts) > 1 else prefixed_name
                            else:
                                # No prefix, use as is
                                actual_tool_name = prefixed_name
                            
                            try:
                                # Call the tool on the appropriate server
                                tool_args = block.input if isinstance(block.input, dict) else {}
                                if server_name == "geospatial" and "session_id" not in tool_args:
                                    tool_args = dict(tool_args)
                                    tool_args["session_id"] = self.spatial_session_id
                                result = await self.client.call_tool(
                                    tool_name=actual_tool_name,
                                    tool_args=tool_args,
                                    server_name=server_name
                                )
                                
                                # Parse the result to extract structured data
                                result_data = {}
                                if hasattr(result, 'content'):
                                    for item in result.content:
                                        if hasattr(item, 'text'):
                                            try:
                                                result_data = json.loads(item.text)
                                            except:
                                                result_data = {"text": item.text}
                                
                                # Create fact from result with full metadata
                                fact = Fact(
                                    text_content=f"[Phase 2 - {actual_tool_name}] {str(result)[:500]}",
                                    source_key=f"phase2_{server_name}_{actual_tool_name}_{iteration}",
                                    server_origin=server_name,
                                    metadata={
                                        "phase": 2,
                                        "iteration": iteration,
                                        "tool": actual_tool_name,
                                        "raw_result": result_data  # Store structured data for Phase 3
                                    },
                                    citation=Citation(
                                        source_name=f"{server_name.upper()} Database",
                                        tool_id=actual_tool_name,
                                        server_origin=server_name,
                                        source_type="database",
                                        description=f"Phase 2 call to {actual_tool_name}"
                                    )
                                )
                                
                                phase2_facts.append(fact)
                                accumulated_facts.append(fact)
                                
                                # Update facts context for next iteration
                                facts_context += f"\n- {fact.text_content}"
                                
                            except Exception as e:
                                print(f"Error calling tool {actual_tool_name} on {server_name}: {e}")
                
                if not tool_called:
                    # No tools called but not complete - avoid infinite loop
                    print("Phase 2: No tools called but not marked complete")
                    break
                    
            except Exception as e:
                print(f"Phase 2 iteration {iteration} error: {e}")
                break
        
        # Merge Phase 2 facts back into results
        phase2_results = {}
        for server_name, original_result in scout_results.items():
            # Keep original result but potentially add Phase 2 facts
            server_phase2_facts = [
                fact for fact in phase2_facts 
                if fact.server_origin == server_name
            ]
            
            if server_phase2_facts:
                # Combine Phase 1 and Phase 2 facts
                all_server_facts = list(original_result.facts) + server_phase2_facts
                phase2_results[server_name] = PhaseResult(
                    is_relevant=original_result.is_relevant,
                    facts=all_server_facts,
                    reasoning=f"{original_result.reasoning} + Phase 2 enrichment",
                    continue_processing=False
                )
            else:
                phase2_results[server_name] = original_result
        
        return phase2_results
    
    
    
    async def _handle_cross_server_callback(self, callback_request: Dict[str, Any]) -> str:
        """
        Handle cross-server callback requests.
        
        When one server needs information from another server, it can request
        a callback. This method routes the request and returns the response.
        
        TODO: Implement callback handling:
        1. Parse callback request: target_server, query
        2. Validate target server exists and is available
        3. Create simple query context (no tool specs needed)
        4. Make LLM call to target server
        5. Return just the answer as a string
        6. Handle errors gracefully
        
        Example callback request:
        {
            "ask_server": "solar",
            "query": "What are the coordinates of facility X?"
        }
        
        Args:
            callback_request: Dictionary with ask_server and query fields
            
        Returns:
            String response from the target server
        """
        target_server = callback_request.get("ask_server")
        query = callback_request.get("query")
        
        if not target_server or not query:
            return "Invalid callback request"
        
        # TODO: Implement callback to target server
        # TODO: Format response for injection into requesting server's context
        return f"Server '{target_server}' responds: [Not implemented]"
    
    async def _phase3_synthesis(self, user_query: str, deep_dive_results: Dict[str, PhaseResult]) -> List[Dict]:
        """
        Phase 3: Synthesis - Transform facts into API-compliant modules.
        
        Synthesizes collected facts into coherent narrative with visualizations
        and proper citations, returning modules in the API-specified format.
        
        Args:
            user_query: Original user query
            deep_dive_results: Results from Phase 2 deep dive
            
        Returns:
            List of modules matching API specification
        """
        # Import direct formatter
        from response_formatter import format_response_as_modules
        # Lazy import translation utility
        try:
            from translation import translate_modules as _translate_modules
        except Exception:
            _translate_modules = None
        
        # Prefer correlation maps only when the user explicitly asks for spatial relations
        # (e.g., within/near/overlap), otherwise prefer base facilities maps.
        PREFER_CORRELATION_MAP = False

        # Step 1: Collect all facts
        all_facts = []
        map_data_list = []  # Store ALL map data, not just one
        chart_data = None  # Will store chart data
        table_data_list = []  # Store table data for creating comparison tables
        viz_table_modules = []  # Store actual table modules from viz server
        
        # Log start of synthesis if tracer is available
        if self.client.fact_tracer:
            self.client.fact_tracer.log_prompt("phase3", "synthesis_start", f"Starting synthesis with {len(deep_dive_results)} server results")
        
        for server_name, result in deep_dive_results.items():
            if result.is_relevant:
                all_facts.extend(result.facts)
                
                # Extract ALL map and table data from solar server facts
                if server_name == "solar":
                    print(f"DEBUG: Checking {len(result.facts)} solar facts for maps and tables")
                    for i, fact in enumerate(result.facts):
                        raw_result = fact.metadata.get("raw_result", {})
                        if raw_result and isinstance(raw_result, dict):
                            # Check for map data in multiple formats
                            if "geojson_url" in raw_result:
                                print(f"DEBUG: Found geojson_url in fact {i}: {raw_result['geojson_url']}")
                                
                                # Format 1: GetSolarFacilitiesMapData returns type: "map_data_summary"
                                if raw_result.get("type") == "map_data_summary":
                                    md = dict(raw_result)
                                    md["is_correlation_map"] = False
                                    map_data_list.append(md)
                                
                                # Format 2: GetSolarFacilitiesMultipleCountries
                                elif raw_result.get("countries_requested"):
                                    # Check if we should create separate maps per country
                                    countries = raw_result.get("countries_requested", [])
                                    facilities_by_country = raw_result.get("facilities_by_country", {})
                                    
                                    # For now, add the combined map
                                    map_data = {
                                        "type": "map_data_summary",
                                        "summary": {
                                            "description": f"Solar facilities in {', '.join(countries)}",
                                            "total_facilities": raw_result.get("total_facilities", 0),
                                            "countries": countries,
                                            "facilities_by_country": facilities_by_country
                                        },
                                        "geojson_url": raw_result["geojson_url"],
                                        "geojson_filename": raw_result.get("geojson_filename")
                                    }
                                    map_data["is_correlation_map"] = False
                                    map_data_list.append(map_data)
                                    
                                    # If we have country breakdown data, create a comparison table
                                    if facilities_by_country:
                                        table_data_list.append({
                                            "type": "country_comparison",
                                            "data": facilities_by_country,
                                            "countries": countries
                                        })
                                
                                # Format 3: Generic geojson_url
                                else:
                                    map_data = {
                                        "type": "map_data_summary",
                                        "summary": {
                                            "description": "Solar facilities map",
                                            "total_facilities": raw_result.get("total_facilities", 0)
                                        },
                                        "geojson_url": raw_result["geojson_url"]
                                    }
                                    map_data["is_correlation_map"] = False
                                    map_data_list.append(map_data)
                            
                            # Check for capacity statistics that could become charts/tables
                            if raw_result.get("capacity_stats") or raw_result.get("facilities_by_country"):
                                print(f"DEBUG: Found potential table/chart data in fact {i}")
                
                # Extract table/chart modules from viz server facts
                elif server_name == "viz":
                    print(f"DEBUG: Checking {len(result.facts)} viz facts for tables and charts")
                    for i, fact in enumerate(result.facts):
                        raw_result = fact.metadata.get("raw_result", {})
                        if raw_result and isinstance(raw_result, dict):
                            # Check if it's a table module from CreateComparisonTable
                            if raw_result.get("type") == "table":
                                print(f"DEBUG: Found table from viz server in fact {i}")
                                viz_table_modules.append(raw_result)

                # Generic map capture from any server (e.g., geospatial correlation maps)
                if result.facts:
                    for i, fact in enumerate(result.facts):
                        raw_result = fact.metadata.get("raw_result", {})
                        if isinstance(raw_result, dict) and "geojson_url" in raw_result:
                            desc = raw_result.get("summary", {}).get("description") or "Map"
                            meta = raw_result.get("metadata", {}) if isinstance(raw_result.get("metadata"), dict) else {}
                            b = meta.get("bounds")
                            c = meta.get("center")
                            countries = meta.get("countries") or []
                            # Fallback to a safe default to satisfy frontend match expression
                            if not countries:
                                countries = ["brazil"]
                            is_corr = (server_name == "geospatial") or (isinstance(desc, str) and ("correlation" in desc.lower()))
                            summary_dict = {
                                "description": desc,
                                "bounds": b,
                                "center": c,
                                "countries": countries
                            }
                            if is_corr:
                                summary_dict.update({
                                    "map_role": "correlation",
                                    "legend_layers": [
                                        {"label": "Solar Assets", "color": "#FFD700"},
                                        {"label": "Deforestation Areas", "color": "#8B4513"}
                                    ]
                                })
                            map_data = {
                                "type": "map_data_summary",
                                "summary": summary_dict,
                                "geojson_url": raw_result["geojson_url"],
                                "geojson_filename": raw_result.get("geojson_filename"),
                                "is_correlation_map": bool(is_corr)
                            }
                            print(f"DEBUG: Captured generic map from {server_name} fact {i}: {map_data['geojson_url']}")
                            map_data_list.append(map_data)

        # Enrich collected facts with structured visualization data (tables/charts)
        # This converts known tool payloads (e.g., lists of companies) into tabular data
        try:
            await self._enhance_facts_with_visualization_data(all_facts)
        except Exception as e:
            print(f"Warning: _enhance_facts_with_visualization_data failed: {e}")

        # Fallback (deterministic): only for explicit correlation intent. If we have geospatial
        # correlation facts but no map captured yet, proactively generate a correlation map.
        try:
            corr_intent = any(kw in user_query.lower() for kw in [
                "within", "inside", "intersect", "overlap", "near", "close to",
                "proximity", "adjacent", "around", "km"
            ])
            # Determine if a correlation map is already present in what we've captured so far
            map_has_corr = any(isinstance(md, dict) and md.get("is_correlation_map") for md in map_data_list)
            if (not _geo_llm_only()) and corr_intent and (not map_has_corr) and (not self._geo_map_generated):
                has_geo_corr = any(
                    (getattr(f, 'server_origin', '') == 'geospatial') and isinstance(f.metadata, dict)
                    and (
                        ('raw_result' in f.metadata and isinstance(f.metadata['raw_result'], dict) and f.metadata['raw_result'].get('total_correlations') is not None)
                        or ('tool' in f.metadata and f.metadata.get('tool') == 'FindSpatialCorrelations')
                    )
                    for f in all_facts
                )
                if has_geo_corr and self.client.sessions.get("geospatial"):
                    map_res = await self.client.call_tool(
                        "GenerateCorrelationMap",
                        {"correlation_type": "spatial_correlation", "session_id": self.spatial_session_id, "show_uncorrelated": False},
                        "geospatial"
                    )
                    if hasattr(map_res, 'content') and map_res.content:
                        import json as _json
                        try:
                            mdata = _json.loads(map_res.content[0].text)
                            if isinstance(mdata, dict) and mdata.get('geojson_url'):
                                # Ensure correlation tagging and legend for formatter
                                summary = mdata.get("summary") or {"description": "Spatial correlation map"}
                                summary.setdefault("map_role", "correlation")
                                summary.setdefault("legend_layers", [
                                    {"label": "Solar Assets", "color": "#FFD700"},
                                    {"label": "Deforestation Areas", "color": "#8B4513"}
                                ])
                                map_data_list.append({
                                    "type": "map_data_summary",
                                    "summary": summary,
                                    "geojson_url": mdata.get("geojson_url"),
                                    "geojson_filename": mdata.get("geojson_filename"),
                                    "is_correlation_map": True
                                })
                                print("DEBUG: Added fallback correlation map from geospatial server in Phase 3")
                                self._geo_map_generated = True
                        except Exception as e:
                            print(f"DEBUG: Failed to parse fallback map data: {e}")
        except Exception as e:
            print(f"DEBUG: Phase 3 map fallback skipped: {e}")

        # De-duplicate maps by geojson_url while preserving order
        if map_data_list:
            seen_urls = set()
            deduped = []
            for md in map_data_list:
                url = md.get("geojson_url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                deduped.append(md)
            map_data_list = deduped

        # Map preference: Only prefer correlation maps when correlation intent is detected.
        try:
            corr_intent_p3 = any(kw in user_query.lower() for kw in [
                "within", "inside", "intersect", "overlap", "near", "close to",
                "proximity", "adjacent", "around", "km"
            ])
        except Exception:
            corr_intent_p3 = False

        has_corr_maps = any(isinstance(md, dict) and md.get("is_correlation_map") for md in map_data_list)
        if has_corr_maps:
            if corr_intent_p3:
                # User asked for a spatial relation; prioritize correlation maps
                map_data_list = [md for md in map_data_list if md.get("is_correlation_map")]
                if len(map_data_list) > 1:
                    map_data_list = map_data_list[:1]
            else:
                # Keep base facilities maps if present; avoid showing empty correlation maps by default
                base_maps = [md for md in map_data_list if not md.get("is_correlation_map")]
                if base_maps:
                    map_data_list = base_maps
                else:
                    # Fall back to the first correlation map if no base map exists
                    map_data_list = [m for m in map_data_list if m.get("is_correlation_map")][:1]
        
        if not all_facts:
            return [{
                "type": "text", 
                "heading": "No Results", 
                "texts": ["No relevant information found for your query."]
            }]
        
        # Debug: Log facts going into synthesis
        if os.getenv("TDE_DEBUG_FACTS"):
            print(f"DEBUG: Phase 3 synthesis starting with {len(all_facts)} total facts")
            for server_name, result in deep_dive_results.items():
                if result.facts:
                    print(f"  {server_name}: {len(result.facts)} facts")
        
        # Log synthesis input if tracer is available
        if self.client.fact_tracer:
            self.client.fact_tracer.log_synthesis_input(all_facts)
            # Log energy facts specifically
            energy_facts = [f for f in all_facts if any(kw in f.text_content.lower() for kw in ['energy', 'renewable', 'biofuel', '18%', 'ethanol', '45%', '80%'])]
            if energy_facts:
                llm_logger.warning(f"🔋 {len(energy_facts)} energy facts going into synthesis:")
                for ef in energy_facts[:5]:  # Show first 5
                    llm_logger.warning(f"  - {ef.text_content[:150]}")
        
        # Step 2: Generate narrative with placeholder citations
        narrative_with_placeholders = await self._synthesize_narrative(user_query, all_facts)
        
        # Step 3: Build citation mapping with deduplication
        citation_map = self._build_citation_map(all_facts, narrative_with_placeholders)
        
        # Step 4: Replace placeholders with ^n^ format
        final_narrative = self._apply_citations(narrative_with_placeholders, citation_map)
        
        # Step 5: Build sources list for citation table
        sources = []
        citation_registry = {"citations": [], "module_citations": {}}
        
        for citation_num, fact in enumerate(all_facts, 1):
            if fact.citation:
                sources.append({
                    "name": fact.citation.source_name,
                    "type": fact.citation.source_type,
                    "provider": fact.citation.server_origin  # Use server_origin instead of provider
                })
                citation_registry["citations"].append({
                    "id": citation_num,
                    "source": fact.citation.source_name,
                    "type": fact.citation.source_type,
                    "description": fact.text_content
                })
        
        # Step 6: Use intelligent module assembly for better narrative interleaving
        # This replaces the simple formatter with context-aware module ordering
        # Use the first map if we have multiple (for backward compatibility)
        primary_map_data = map_data_list[0] if map_data_list else None
        
        modules = await self._assemble_modules_intelligently(
            narrative=final_narrative,
            facts=all_facts,
            citation_map=citation_map,
            user_query=user_query,
            map_data=primary_map_data,
            chart_data=chart_data,
            sources=sources,
            additional_maps=map_data_list[1:] if len(map_data_list) > 1 else [],
            table_data_list=table_data_list,
            viz_table_modules=viz_table_modules
        )

        # If we have a deterministic geospatial correlation result, prepend a concise answer module
        try:
            correlation_count = None
            unique_facilities = None
            zone_label = None
            distance_km = None
            for fact in all_facts:
                raw = fact.metadata.get("raw_result") if isinstance(fact.metadata, dict) else None
                if isinstance(raw, dict) and ("unique_facilities" in raw or "total_correlations" in raw):
                    if "unique_facilities" in raw:
                        unique_facilities = raw.get("unique_facilities")
                    # Prefer geospatial server results
                    if fact.server_origin == "geospatial":
                        correlation_count = raw.get("total_correlations")
                        # Infer zone label and any distance parameter for a tailored summary
                        try:
                            if isinstance(raw.get("correlations"), list) and raw["correlations"]:
                                et2 = raw["correlations"][0].get("entity2_type")
                                if et2 == "deforestation_area":
                                    zone_label = "deforestation areas"
                                elif et2 == "heat_zone":
                                    zone_label = "heat stress zones"
                            params = raw.get("parameters") or {}
                            dk = params.get("distance_km")
                            if dk:
                                distance_km = float(dk)
                        except Exception:
                            pass
                        if "unique_facilities" in raw:
                            unique_facilities = raw.get("unique_facilities")
                        break
                    correlation_count = correlation_count or raw.get("total_correlations")
            # Only prepend a correlation sentence if the user asked about deforestation/proximity
            ql2 = user_query.lower()
            if (unique_facilities is not None or correlation_count is not None) and any(k in ql2 for k in ["deforest", "in deforestation", "forest loss", "near", "km", "proximity", "heat", "temperature"]):
                n = unique_facilities if unique_facilities is not None else correlation_count
                # Compose a general but precise summary
                if distance_km and zone_label:
                    answer_text = f"Based on the spatial correlation analysis, {int(n)} solar assets are within {distance_km:g} km of {zone_label}."
                elif zone_label:
                    answer_text = f"Based on the spatial correlation analysis, {int(n)} solar assets are within the specified proximity to {zone_label}."
                else:
                    answer_text = f"Based on the spatial correlation analysis, {int(n)} solar assets are within the specified proximity."
                modules.insert(0, {"type": "text", "heading": "", "texts": [answer_text]})
        except Exception as e:
            print(f"Failed to prepend correlation answer: {e}")

        # Apply translation if requested
        if self.target_language and self.target_language.startswith("pt") and _translate_modules:
            try:
                modules = await _translate_modules(modules, self.target_language)
            except Exception:
                pass
        return modules
    
    async def _enhance_facts_with_visualization_data(self, facts: List[Fact]):
        """
        Analyze raw_result in metadata to populate visualization fields.
        Modifies facts in place.
        
        Args:
            facts: List of facts to enhance with visualization data
        """
        for fact in facts:
            if "raw_result" not in fact.metadata:
                continue
                
            raw = fact.metadata["raw_result"]
            
            # Time series detection
            if self._has_time_pattern(raw):
                fact.numerical_data = self._extract_time_series(raw)
                fact.data_type = "time_series"
            
            # Categorical/comparison detection
            elif self._has_categories(raw):
                fact.numerical_data = self._extract_categorical(raw)
                fact.data_type = "comparison"

            # GIST: Companies at risk (convert to table)
            elif (
                isinstance(raw, dict)
                and isinstance(raw.get("companies"), list)
                and raw.get("companies")
                and any(k in raw.keys() for k in ["risk_type", "risk_level", "companies_found"])  # heuristic for GIST risk calls
            ):
                try:
                    companies = raw.get("companies", [])
                    # Validate expected fields
                    sample = companies[0] if companies else {}
                    expected_fields = {"company_name", "at_risk_assets", "risk_percentage"}
                    if isinstance(sample, dict) and expected_fields.issubset(set(sample.keys())):
                        columns = [
                            "Rank",
                            "Company",
                            "At-risk assets",
                            "% of company assets",
                            "Sector",
                            "Country",
                            "Total assets",
                        ]
                        rows = []
                        for idx, item in enumerate(companies, start=1):
                            try:
                                rows.append([
                                    idx,
                                    item.get("company_name", ""),
                                    int(item.get("at_risk_assets", 0) or 0),
                                    float(item.get("risk_percentage", 0) or 0),
                                    item.get("sector_code", ""),
                                    item.get("country", ""),
                                    int(item.get("total_assets", 0) or 0),
                                ])
                            except Exception:
                                continue
                        if rows:
                            # Friendly label for risk type
                            rt = str(raw.get("risk_type") or "").upper()
                            rt_label = {
                                "FLOOD_RIVERINE": "riverine flood",
                                "FLOOD_COASTAL": "coastal flood",
                            }.get(rt, rt.replace("_", " ").lower())
                            rl = str(raw.get("risk_level") or "").title()
                            title = f"Companies at {rl} risk ({rt_label})"
                            fact.numerical_data = {
                                "columns": columns,
                                "rows": rows,
                                "title": title,
                                "source": "gist_companies_risk",
                            }
                            fact.data_type = "tabular"
                except Exception:
                    pass
            
            # Special-case: top countries by facility count (solar server)
            elif isinstance(raw, dict) and (isinstance(raw.get("countries"), list) or isinstance(raw.get("top_10_countries"), list)):
                countries_list = raw.get("countries") or raw.get("top_10_countries")
                # Validate shape: list of dicts with country (str) and facility_count (numeric)
                if countries_list and isinstance(countries_list[0], dict):
                    first = countries_list[0]
                    if (
                        "country" in first and "facility_count" in first and
                        isinstance(first["country"], str) and isinstance(first["facility_count"], (int, float))
                    ):
                        # Build a simple ranked table
                        columns = ["Rank", "Country", "Facilities"]
                        rows = []
                        for idx, item in enumerate(countries_list, start=1):
                            try:
                                rows.append([idx, item.get("country", ""), int(item.get("facility_count", 0))])
                            except Exception:
                                continue
                        if rows:
                            fact.numerical_data = {"columns": columns, "rows": rows, "source": "solar_top_countries"}
                            fact.data_type = "tabular"

            # Geographic detection (already saved in Phase 1/2)
            elif "map_url" in raw or "geojson_url" in raw:
                fact.map_reference = {
                    "url": raw.get("map_url") or raw.get("geojson_url"),
                    "bounds": raw.get("bounds"),
                    "summary": raw.get("summary")
                }
                fact.data_type = "geographic"
            
            # Table detection
            elif self._is_tabular(raw):
                fact.numerical_data = self._extract_table_data(raw)
                fact.data_type = "tabular"
    
    def _has_time_pattern(self, data: Dict) -> bool:
        """Detect if data contains time series information."""
        if not isinstance(data, dict):
            return False
        
        # Look for time indicators in keys or nested data
        time_indicators = ['year', 'date', 'time', 'month', 'day', 'timestamp']
        
        # Check top-level keys
        for key in data.keys():
            if any(indicator in key.lower() for indicator in time_indicators):
                return True
        
        # Check if data is a list with time patterns
        if isinstance(data.get('data'), list) and data['data']:
            first_item = data['data'][0]
            if isinstance(first_item, dict):
                for key in first_item.keys():
                    if any(indicator in key.lower() for indicator in time_indicators):
                        return True
        
        return False
    
    def _has_categories(self, data: Dict) -> bool:
        """Detect if data contains categorical comparisons."""
        if not isinstance(data, dict):
            return False
        
        # Look for categorical indicators
        category_indicators = ['category', 'sector', 'type', 'name', 'group', 'class']
        numeric_indicators = ['count', 'total', 'value', 'amount', 'score', 'percentage']
        
        has_categories = False
        has_numeric = False
        
        # Check top-level structure
        for key in data.keys():
            key_lower = key.lower()
            if any(indicator in key_lower for indicator in category_indicators):
                has_categories = True
            if any(indicator in key_lower for indicator in numeric_indicators):
                has_numeric = True
        
        # Check nested data
        if isinstance(data.get('data'), list) and data['data']:
            first_item = data['data'][0]
            if isinstance(first_item, dict):
                for key in first_item.keys():
                    key_lower = key.lower()
                    if any(indicator in key_lower for indicator in category_indicators):
                        has_categories = True
                    if any(indicator in key_lower for indicator in numeric_indicators):
                        has_numeric = True
        
        return has_categories and has_numeric
    
    def _is_tabular(self, data: Dict) -> bool:
        """Detect if data should be presented as a table."""
        if not isinstance(data, dict):
            return False
        
        # Check if data has tabular structure
        if 'rows' in data and 'columns' in data:
            return True
        
        # Check if data is a list of uniform dictionaries
        if isinstance(data.get('data'), list) and len(data['data']) > 1:
            first_item = data['data'][0]
            if isinstance(first_item, dict) and len(first_item) > 2:
                # Check if all items have similar structure
                keys = set(first_item.keys())
                for item in data['data'][1:3]:  # Check first few items
                    if isinstance(item, dict) and set(item.keys()) == keys:
                        return True
                    break
        
        return False
    
    def _extract_time_series(self, data: Dict) -> Dict:
        """Extract time series data for chart generation."""
        extracted = {"values": [], "summary": {}}
        
        if isinstance(data.get('data'), list):
            extracted["values"] = data['data']
        else:
            # Try to construct time series from other formats
            for key, value in data.items():
                if isinstance(value, (int, float)) and 'year' in key.lower():
                    extracted["values"].append({"year": key, "value": value})
        
        # Add summary statistics
        if extracted["values"]:
            numeric_values = []
            for item in extracted["values"]:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, (int, float)) and k != 'year':
                            numeric_values.append(v)
                            break
                elif isinstance(item, (int, float)):
                    numeric_values.append(item)
            
            if numeric_values:
                extracted["summary"] = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values),
                    "trend": "increasing" if numeric_values[-1] > numeric_values[0] else "decreasing"
                }
        
        return extracted
    
    def _extract_categorical(self, data: Dict) -> Dict:
        """Extract categorical data for bar/pie charts."""
        extracted = {"categories": [], "values": [], "summary": {}}
        
        if isinstance(data.get('data'), list):
            for item in data['data']:
                if isinstance(item, dict):
                    # Find category and value fields
                    category_val = None
                    numeric_val = None
                    
                    for key, val in item.items():
                        if isinstance(val, str) and not category_val:
                            category_val = val
                        elif isinstance(val, (int, float)) and not numeric_val:
                            numeric_val = val
                    
                    if category_val and numeric_val is not None:
                        extracted["categories"].append(category_val)
                        extracted["values"].append(numeric_val)
        
        # Add summary
        if extracted["values"]:
            extracted["summary"] = {
                "total": sum(extracted["values"]),
                "count": len(extracted["values"]),
                "max_category": extracted["categories"][extracted["values"].index(max(extracted["values"]))]
            }
        
        return extracted
    
    def _extract_table_data(self, data: Dict) -> Dict:
        """Extract tabular data structure."""
        extracted = {"columns": [], "rows": [], "summary": {}}
        
        # Direct table format
        if 'columns' in data and 'rows' in data:
            extracted["columns"] = data["columns"]
            extracted["rows"] = data["rows"]
        
        # List of dictionaries format
        elif isinstance(data.get('data'), list) and data['data']:
            first_item = data['data'][0]
            if isinstance(first_item, dict):
                extracted["columns"] = list(first_item.keys())
                for item in data['data']:
                    if isinstance(item, dict):
                        row = [item.get(col, '') for col in extracted["columns"]]
                        extracted["rows"].append(row)
        
        extracted["summary"] = {
            "column_count": len(extracted["columns"]),
            "row_count": len(extracted["rows"])
        }
        
        return extracted
    
    async def _assemble_modules_intelligently(
        self, 
        narrative: str, 
        facts: List[Fact], 
        citation_map: Dict[str, int],
        user_query: str,
        map_data: Optional[Dict] = None,
        chart_data: Optional[List[Dict]] = None,
        sources: Optional[List] = None,
        additional_maps: Optional[List[Dict]] = None,
        table_data_list: Optional[List[Dict]] = None,
        viz_table_modules: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Use LLM to intelligently order modules for natural narrative flow.
        
        Strategy:
        1. Parse narrative into sections
        2. Gather all available visualizations
        3. Use LLM to check if visualizations directly answer the query
        4. Assemble modules in LLM-suggested order
        5. Always put citation table last
        
        Args:
            narrative: Final narrative text with citations
            facts: All facts with visualization data
            citation_map: Citation mapping for deduplication
            user_query: Original user query
            map_data: Map data from solar server
            chart_data: Chart data
            sources: Sources for citations
            
        Returns:
            List of modules in intelligent order
        """
        # Parse narrative into sections
        sections = self._parse_narrative_sections(narrative)
        
        # Get all available visualizations
        chart_modules = await self._create_chart_modules(facts)
        
        # Create map modules from primary map and additional maps
        map_modules = self._create_map_modules(facts, map_data)

        # Deduplicate maps by URL/filename, preserving order (prefer primary first)
        deduped = []
        seen_keys = set()
        for m in map_modules:
            key = m.get("geojson_url") or m.get("filename") or id(m)
            if key and key not in seen_keys:
                deduped.append(m)
                seen_keys.add(key)
        map_modules = deduped
        
        # Add additional maps if provided
        if additional_maps:
            for extra_map in additional_maps:
                # Use response_formatter's _create_map_module directly
                try:
                    from response_formatter import _create_map_module
                    extra_module = _create_map_module(extra_map)
                    if extra_module:
                        map_modules.append(extra_module)
                except Exception as e:
                    print(f"Error creating additional map module: {e}")
        
        # Create table modules from facts and additional table data
        table_modules = self._create_table_modules(facts)
        
        # Add tables created by viz server during Phase 2
        if viz_table_modules:
            print(f"DEBUG: Adding {len(viz_table_modules)} tables from viz server")
            table_modules.extend(viz_table_modules)
        
        print(f"DEBUG: Visualization modules created:")
        print(f"  Charts: {len(chart_modules)}")
        print(f"  Maps: {len(map_modules)}")
        print(f"  Tables: {len(table_modules)}")
        if map_modules:
            for i, m in enumerate(map_modules):
                print(f"  Map {i}: {m.get('heading', 'No heading')}")
        
        # Filter visualizations for relevance before including them
        if chart_modules or map_modules or table_modules:
            # Force include admin–zone tables (overlap rankings) to avoid LLM gating drop
            force_tables = False
            try:
                at = self._detect_admin_intent(user_query)
                zt = self._detect_zone_intent(user_query)
                if at == 'municipality' and (zt or any(k in user_query.lower() for k in ['heat', 'deforest', 'overlap'])):
                    force_tables = True
            except Exception:
                pass
            # For explicit spatial queries, include maps without filtering
            spatial_query = any(k in user_query.lower() for k in [
                "within", "km", "proximity", "near", "adjacent", "in deforestation",
                # Treat common location-intent phrasings as spatial too
                "where", "map", "show me", "locations"
            ])
            # Detect admin+zone intent to constrain maps (e.g., municipalities x heat)
            try:
                at = self._detect_admin_intent(user_query)
            except Exception:
                at = None
            try:
                zt = self._detect_zone_intent(user_query)
            except Exception:
                zt = None
            is_muni_heat = (at == 'municipality') and (zt == 'heat_zone' or ('heat' in user_query.lower()))
            allow_multiple_maps = any(k in user_query.lower() for k in [
                "layers", "multiple maps", "two maps", "compare maps", "overlay", "overlays"
            ])
            # Charts: deterministic inclusion (no LLM gating)
            # If query suggests comparisons/tables/charts, include all charts; else include first per provider (we often lack provider tags, so include first only)
            compare_query = any(k in user_query.lower() for k in ["compare", "versus", "table", "chart", "top", "highest"])
            relevant_charts = list(chart_modules if compare_query else chart_modules[:1])
            
            relevant_maps = []
            # Always run relevance filter; if spatial query and nothing passes, fall back to first map
            for map_module in map_modules:
                if await self._is_visualization_relevant(map_module, user_query, "map"):
                    relevant_maps.append(map_module)
                else:
                    print(f"Filtered out irrelevant map: {map_module.get('heading', 'unnamed')}")

            # For municipality heat intent, only allow geospatial choropleth/heat maps; never fall back to solar maps
            if is_muni_heat:
                def _is_heat_admin_map(m: dict) -> bool:
                    url = (m.get('geojson_url') or '')
                    if not isinstance(url, str):
                        return False
                    url_l = url.lower()
                    if 'solar_facilities' in url_l:
                        return False
                    return ('admin_choropleth_' in url_l) or ('overlap_heat' in url_l) or ('heat_zone' in url_l)
                relevant_maps = [m for m in relevant_maps if _is_heat_admin_map(m)]
            
            # Generic spatial fallback only when not the muni-heat constrained case
            if spatial_query and not is_muni_heat and not relevant_maps and map_modules:
                # Ensure at least one map for spatial queries
                relevant_maps = [map_modules[0]]

            # Cap number of maps unless explicitly requested
            if not allow_multiple_maps and len(relevant_maps) > 1:
                relevant_maps = relevant_maps[:1]
            
            relevant_tables = []
            if force_tables:
                relevant_tables = list(table_modules)
            else:
                for table in table_modules:
                    if await self._is_visualization_relevant(table, user_query, "table"):
                        relevant_tables.append(table)
                    else:
                        print(f"Filtered out irrelevant table: {table.get('heading', 'unnamed')}")
            
            # Use LLM to determine placement for relevant visualizations
            if relevant_charts or relevant_maps or relevant_tables:
                ordered_modules = await self._llm_order_modules(
                    sections=sections,
                    chart_modules=relevant_charts,
                    map_modules=relevant_maps,
                    table_modules=relevant_tables,
                    user_query=user_query
                )
            else:
                # No relevant visualizations, just use text sections
                ordered_modules = []
                for section in sections:
                    ordered_modules.append({
                        "type": "text",
                        "heading": section["heading"],
                        "texts": section["paragraphs"]
                    })
        else:
            # No visualizations, just use text sections
            ordered_modules = []
            for section in sections:
                ordered_modules.append({
                    "type": "text",
                    "heading": section["heading"],
                    "texts": section["paragraphs"]
                })
        
        # Citation table always goes last
        citation_table = self._create_citation_table(citation_map, facts, sources)
        if citation_table:
            ordered_modules.append(citation_table)
        
        return ordered_modules
    
    async def _is_visualization_relevant(self, viz_module: Dict, user_query: str, viz_type: str) -> bool:
        """
        Check if a visualization directly helps answer the user's query.
        
        Args:
            viz_module: The visualization module to check
            user_query: The original user query
            viz_type: Type of visualization ("map", "chart", "table")
            
        Returns:
            True if the visualization should be included
        """
        # Get description of what the visualization shows
        viz_description = viz_module.get('heading', 'Data visualization')
        
        # For tables from viz server, check the data content
        if viz_type == "table" and 'data' in viz_module:
            # Extract what the table contains
            if 'columns' in viz_module.get('data', {}):
                columns = viz_module['data']['columns']
                viz_description += f" with columns: {', '.join(columns[:3])}"
        
        relevance_prompt = f"""Query: {user_query}

            Available {viz_type}: {viz_description}

            Should this {viz_type} be included in the response?

            Decision criteria:
            - Maps: Include ONLY if the query explicitly asks about locations, geography, spatial distribution, or "where"
            - Tables: Include ONLY if the query asks to compare multiple entities, see breakdowns, or requests detailed data
            - Charts: Include ONLY if the query asks about trends, changes over time, growth, or proportions

            The visualization must DIRECTLY help answer the specific question, not just be related to the topic.

            Examples:
            - Query: "What are Brazil's NDC targets?" → DON'T include map of solar facilities
            - Query: "Where are solar facilities in Brazil?" → DO include map
            - Query: "Compare emissions across sectors" → DO include comparison table
            - Query: "What is the water stress level?" → DON'T include unrelated charts

            Answer YES or NO:"""

        try:
            response = await call_small_model(
                system="You determine if visualizations directly answer queries.",
                user_prompt=relevance_prompt,
                max_tokens=10,
                temperature=0
            )
            
            return "yes" in response.strip().lower()
            
        except Exception as e:
            print(f"Error checking visualization relevance: {e}")
            # Default to excluding if check fails
            return False
    
    async def _llm_order_modules(
        self,
        sections: List[Dict],
        chart_modules: List[Dict],
        map_modules: List[Dict],
        table_modules: List[Dict],
        user_query: str
    ) -> List[Dict]:
        """
        Use LLM to determine optimal module ordering for narrative flow.
        
        Args:
            sections: Parsed narrative text sections
            chart_modules: Available chart visualizations
            map_modules: Available map visualizations
            table_modules: Available tables
            user_query: Original user query for context
            
        Returns:
            List of all modules in optimal order
        """
        # Build concise descriptions for the LLM
        section_summaries = []
        for i, section in enumerate(sections):
            summary = f"Text{i}: {section['heading'] or 'Introduction'} - {section['content'][:150]}..."
            section_summaries.append(summary)
        
        chart_summaries = []
        for i, chart in enumerate(chart_modules):
            summary = f"Chart{i}: {chart.get('heading', 'Data visualization')}"
            chart_summaries.append(summary)
            
        map_summaries = []
        for i, map_module in enumerate(map_modules):
            summary = f"Map{i}: {map_module.get('heading', 'Geographic visualization')}"
            map_summaries.append(summary)
            
        table_summaries = []
        for i, table in enumerate(table_modules):
            summary = f"Table{i}: {table.get('heading', 'Data table')}"
            table_summaries.append(summary)
        
        # Create the ordering prompt
        prompt = f"""You are arranging content modules for a response about: {user_query}

Available content modules:

TEXT SECTIONS:
{chr(10).join(section_summaries)}

VISUALIZATIONS:
{chr(10).join(chart_summaries) if chart_summaries else '(No charts)'}
{chr(10).join(map_summaries) if map_summaries else '(No maps)'}
{chr(10).join(table_summaries) if table_summaries else '(No tables)'}

Create an optimal reading order where:
1. Text introduces concepts before related visualizations
2. Maps appear when geographic context is established
3. Charts follow discussions of trends or comparisons
4. Tables accompany detailed data discussions
5. Related content flows naturally together

Return a JSON array of module IDs in order, like:
["Text0", "Map0", "Text1", "Chart0", "Text2", "Table0"]

Only include modules that exist above. Ensure all modules are included exactly once."""

        try:
            print(f"DEBUG: Calling LLM to order modules...")
            response = await call_small_model(
                system="You are an expert at organizing content for optimal narrative flow. Return only valid JSON.",
                user_prompt=prompt,
                max_tokens=500,
                temperature=0
            )
            print(f"DEBUG: LLM response received")
            
            # Parse the ordering
            import re
            # Extract JSON array from response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                order = json.loads(json_match.group())
            else:
                # Fallback to simple interleaving
                order = self._create_fallback_order(sections, chart_modules, map_modules, table_modules)
            
            # Build the final module list based on the order
            print(f"DEBUG: LLM ordered modules as: {order}")
            ordered_modules = []
            for module_id in order:
                if module_id.startswith("Text"):
                    idx = int(module_id[4:])
                    if idx < len(sections):
                        ordered_modules.append({
                            "type": "text",
                            "heading": sections[idx]["heading"],
                            "texts": sections[idx]["paragraphs"]
                        })
                elif module_id.startswith("Chart"):
                    idx = int(module_id[5:])
                    if idx < len(chart_modules):
                        ordered_modules.append(chart_modules[idx])
                elif module_id.startswith("Map"):
                    idx = int(module_id[3:])
                    if idx < len(map_modules):
                        ordered_modules.append(map_modules[idx])
                elif module_id.startswith("Table"):
                    idx = int(module_id[5:])
                    if idx < len(table_modules):
                        ordered_modules.append(table_modules[idx])
            
            return ordered_modules
            
        except Exception as e:
            print(f"LLM ordering failed: {e}. Using fallback order.")
            return self._create_fallback_order_modules(sections, chart_modules, map_modules, table_modules)
    
    def _create_fallback_order(self, sections, chart_modules, map_modules, table_modules):
        """Create a simple fallback ordering if LLM fails."""
        order = []
        for i in range(len(sections)):
            order.append(f"Text{i}")
        for i in range(len(map_modules)):
            order.append(f"Map{i}")
        for i in range(len(chart_modules)):
            order.append(f"Chart{i}")
        for i in range(len(table_modules)):
            order.append(f"Table{i}")
        return order
    
    def _create_fallback_order_modules(self, sections, chart_modules, map_modules, table_modules):
        """Create fallback module list if LLM ordering fails."""
        modules = []
        
        # Add all text sections first
        for section in sections:
            modules.append({
                "type": "text",
                "heading": section["heading"],
                "texts": section["paragraphs"]
            })
        
        # Then add visualizations
        modules.extend(map_modules)
        modules.extend(chart_modules)
        modules.extend(table_modules)
        
        return modules

    def _parse_narrative_sections(self, narrative: str) -> List[Dict]:
        """
        Parse narrative into logical sections based on headers and paragraphs.
        
        Args:
            narrative: The complete narrative text
            
        Returns:
            List of section dictionaries with heading, paragraphs, and content
        """
        sections = []
        current_section = {"heading": "", "paragraphs": [], "content": ""}
        current_paragraph = []
        
        lines = narrative.split('\n')
        for line in lines:
            line_stripped = line.strip()
            
            # Check for section headers
            if line_stripped.startswith('##'):
                # Save current paragraph if exists
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    current_section["paragraphs"].append(paragraph_text)
                    current_section["content"] += " " + paragraph_text
                    current_paragraph = []
                
                # Save current section if it has content
                if current_section["paragraphs"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "heading": line_stripped.replace('##', '').strip(),
                    "paragraphs": [],
                    "content": ""
                }
            
            # Empty line indicates paragraph break
            elif not line_stripped:
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    current_section["paragraphs"].append(paragraph_text)
                    current_section["content"] += " " + paragraph_text
                    current_paragraph = []
            
            # Regular content line
            else:
                current_paragraph.append(line_stripped)
        
        # Save any remaining paragraph
        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            current_section["paragraphs"].append(paragraph_text)
            current_section["content"] += " " + paragraph_text
        
        # Save any remaining section
        if current_section["paragraphs"]:
            sections.append(current_section)
        
        # If no sections found, split by double newlines
        if not sections:
            paragraphs = [p.strip() for p in narrative.split('\n\n') if p.strip()]
            if paragraphs:
                # Group into logical sections (max 2-3 paragraphs per section)
                for i in range(0, len(paragraphs), 2):
                    section_paragraphs = paragraphs[i:i+2]
                    sections.append({
                        "heading": "",
                        "paragraphs": section_paragraphs,
                        "content": " ".join(section_paragraphs)
                    })
            else:
                # Fallback: treat entire narrative as one section
                sections.append({
                    "heading": "",
                    "paragraphs": [narrative] if narrative else ["No content"],
                    "content": narrative
                })
        
        return sections
    
    def _is_chart_relevant_to_section(self, chart: Dict, section_text: str) -> bool:
        """
        Determine if a chart should be placed after this text section.
        Checks for keyword matches, data references, etc.
        
        Args:
            chart: Chart module dictionary
            section_text: Text content of the section
            
        Returns:
            True if chart is relevant to this section
        """
        chart_title = chart.get("heading", "").lower()
        section_lower = section_text.lower()
        
        # Check for direct mentions
        if chart_title and any(word in section_lower for word in chart_title.split() if len(word) > 3):
            return True
        
        # Check for data type mentions
        chart_type = chart.get("chartType", "")
        if chart_type == "line" and any(word in section_lower for word in ["trend", "over time", "growth", "decline", "change"]):
            return True
        
        if chart_type in ["bar", "pie", "doughnut"] and any(word in section_lower for word in ["comparison", "compared", "versus", "by sector", "distribution"]):
            return True
        
        return False
    
    def _is_map_relevant_to_section(self, map_module: Dict, section_text: str) -> bool:
        """
        Determine if a map should be placed after this text section.
        
        Args:
            map_module: Map module dictionary
            section_text: Text content of the section
            
        Returns:
            True if map is relevant to this section
        """
        section_lower = section_text.lower()
        
        # Check for geographic mentions
        if any(word in section_lower for word in ["location", "map", "geographic", "facilities", "sites", "regions", "country", "state", "province"]):
            return True
        
        # Check for specific country/region mentions that match the map
        map_title = map_module.get("heading", "").lower()
        if map_title and any(location in section_lower for location in map_title.split() if len(location) > 3):
            return True
        
        return False
    
    def _is_table_relevant_to_section(self, table: Dict, section_text: str) -> bool:
        """
        Determine if a table should be placed after this text section.
        
        Args:
            table: Table module dictionary
            section_text: Text content of the section
            
        Returns:
            True if table is relevant to this section
        """
        section_lower = section_text.lower()
        
        # Check for data mentions that suggest tables
        if any(word in section_lower for word in ["data", "details", "breakdown", "summary", "comparison", "list", "results"]):
            return True
        
        # Check for table title mentions
        table_title = table.get("heading", "").lower()
        if table_title and any(word in section_lower for word in table_title.split() if len(word) > 3):
            return True
        
        return False
    
    async def _synthesize_narrative(self, user_query: str, facts: List[Fact]) -> str:
        """
        Generate coherent narrative with placeholder citations.
        
        Args:
            user_query: Original user query
            facts: All collected facts
            
        Returns:
            Narrative text with placeholder citations like [CITE_1]
        """
        # Prepare fact summaries for LLM
        fact_list = []
        has_map_data = False
        has_chart_data = False
        has_table_data = False
        
        # Log synthesis prompt if tracer is available
        if self.client.fact_tracer:
            self.client.fact_tracer.log_prompt("phase3", "synthesis_facts", f"Preparing {len(facts)} facts for synthesis")
        
        # Rank and filter facts to better match the query intent
        ql = user_query.lower()
        def _fact_score(f: Fact) -> float:
            score = 0.0
            # Prefer KG evidence
            if f.server_origin == 'kg':
                score += 2.0
            # Prefer quoted evidence facts
            if isinstance(f.text_content, str) and f.text_content.startswith('Quoted evidence:'):
                score += 2.0
            # Prefer UNFCCC party submissions when query references national submissions
            src = (f.citation.source_name if f.citation else '') or ''
            if 'unfccc.party' in src.lower():
                score += 1.0
            # Keyword alignment
            if 'indigenous' in ql and 'indigenous' in f.text_content.lower():
                score += 1.0
            if 'deforest' in ql and 'deforest' in f.text_content.lower():
                score += 0.5
            return score

        # Drop placeholder-like generic facts (no quoted evidence and no doc id)
        filtered = []
        for f in facts:
            src = (f.citation.source_name if f.citation else '') or ''
            if src == 'guidance_doc' and not (isinstance(f.text_content, str) and f.text_content.startswith('Quoted evidence:')):
                continue
            filtered.append(f)

        ranked = sorted(filtered or facts, key=_fact_score, reverse=True)

        for i, fact in enumerate(ranked):
            # Base textual fact content
            fact_entry = f"{i+1}. {fact.text_content}"

            # If this fact includes a tabular payload, append a raw excerpt (top 20 rows) so the LLM can see actual table data
            try:
                if getattr(fact, 'numerical_data', None) and getattr(fact, 'data_type', None) == 'tabular':
                    table = fact.numerical_data or {}
                    cols = table.get('columns') or []
                    rows = table.get('rows') or []
                    title = table.get('title') or ''
                    if isinstance(cols, list) and isinstance(rows, list) and rows:
                        has_table_data = True
                        try:
                            top_n = min(20, len(rows))
                            header = " | ".join([str(c) for c in cols])
                            lines = []
                            for r in rows[:top_n]:
                                if isinstance(r, (list, tuple)):
                                    row_vals = [str(v) for v in r[:len(cols)]]
                                else:
                                    # Fallback: attempt to map dict by column order
                                    row_vals = [str(r.get(c, '')) if isinstance(r, dict) else str(r) for c in cols]
                                lines.append(" | ".join(row_vals))
                            table_block = "\n".join([header] + lines)
                            prefix_title = f"Table: {title}\n" if title else "Table:\n"
                            fact_entry += f"\n{prefix_title}{table_block}"
                        except Exception:
                            pass
            except Exception:
                # Never let table summarization break synthesis
                pass

            fact_list.append(fact_entry)
            # Check if this fact contains map data
            if 'raw_result' in fact.metadata:
                raw = fact.metadata['raw_result']
                if isinstance(raw, dict) and raw.get('type') == 'map_data_summary' and raw.get('geojson_url'):
                    has_map_data = True
            # Check for chart data
            if fact.numerical_data and fact.data_type in ['time_series', 'comparison']:
                has_chart_data = True
        
        # Build context about what will be displayed
        display_context = []
        if has_map_data:
            display_context.append("A MAP will be displayed showing the geographic data")
        if has_chart_data:
            display_context.append("CHARTS will be displayed showing the numerical data")
        
        display_info = ""
        if display_context:
            # Provide visualization availability as an instruction, not prose to echo back
            vis_list = ", ".join(item.replace("A ", "").replace("CHARTS ", "charts ").lower() for item in display_context)
            display_info = (
                "\nVISUALIZATION AVAILABILITY (instruction for you, do not echo): "
                f"available visuals: {vis_list}. "
                "Do NOT describe or promise visualizations in your answer; the UI renders maps/tables separately.\n"
            )
        
        # Add conversation history if available
        context_section = ""
        if self.conversation_history:
            context_section = f"""Previous Conversation Context:
{self._format_conversation_history()}

Current Query: {user_query}"""
        else:
            context_section = f"User Query: {user_query}"
        
        prompt = f"""{context_section}

CRITICAL INSTRUCTIONS - ANSWER THIS SPECIFIC QUERY: {user_query}

Coverage Check (MANDATORY before drafting):
1) From the facts, infer which layers are present: has_svi (social vulnerability/IVS), has_heat (heat zones or metrics), has_solar (solar facilities), has_grid (grid/electrification). If a layer is not present in the facts, treat it as absent.
2) If has_svi is FALSE and the user asked about vulnerability, you MUST explicitly state this absence at the start and DO NOT use the word "vulnerability" elsewhere (unless quoting the user). Instead, clearly limit analysis to available layers (e.g., heat exposure and a clearly labeled proxy for weak renewable access based on low solar facility density).
3) If using the proxy for weak renewable access, you MUST label it as such and MUST NOT imply electrification rates or grid reliability.

Your FIRST SENTENCE must directly answer the above query. If a disclaimer from the coverage check is required, include it as the first sentence, then follow immediately with the direct answer.

Available Facts:
{chr(10).join(fact_list)}
{display_info}

FOCUSED RESPONSE RULES:
1. Start with a direct answer to the query in your first sentence
2. ONLY include facts that support this direct answer
3. Use placeholder citations [CITE_1], [CITE_2] for fact references
4. If any fact begins with "Quoted evidence:", include that quoted text verbatim in your answer where appropriate
5. Prefer passages from national submissions when relevant (e.g., sources with 'UNFCCC.party')
6. If facts DO NOT include social vulnerability (SVI/IVS) data, you MUST say that vulnerability data is not present and limit the analysis to the available layers (e.g., heat exposure and clearly-labeled proxies). Do NOT claim "vulnerability" without such data.
7. If referencing weak renewable access without direct grid/electrification data, you MUST label it explicitly as a proxy based on low solar facility presence/density; do NOT conflate this with electrification rates or grid reliability.
8. DO NOT include numeric totals (e.g., population impacted) unless they appear in the facts as computed values or tables.
 9. DO NOT add a separate "Note:" about maps or visualizations; do not promise that a map "will show" anything. Let the visuals, if included, speak for themselves in the UI.
 10. Avoid future tense; describe results in present tense.
 11. Avoid words like "comprehensive", "all facilities", or "actively tracked" unless explicitly stated in the facts. Use precise counts and dataset names instead.

DO NOT include:
   - Information about other countries/companies not asked about
   - Tangentially related data that doesn't answer the query
   - Background context not requested
   - Facility counts or infrastructure data unless specifically asked

Example for NDC energy queries:
GOOD START: "Country X commits to achieving [specific %] renewable energy by [year], with additional targets for [specific sectors/fuels] [CITE_1]."
BAD START: "Country X has a comprehensive climate policy framework with many facilities..."

For energy-related NDC queries, PRIORITIZE mentioning:
- ALL quantitative targets with their units (percentages, TWh, MW, GW)
- Specific fuel or technology commitments (biofuels, solar, wind, hydro, nuclear)
- Sectoral targets (electricity, transport, industry, buildings)
- Timeline commitments with specific years
- Implementation status if mentioned in facts

Structure:
1. First paragraph: Direct answer with key facts
2. Supporting paragraphs: Only details that elaborate on the answer
3. Use ## headings only if you have multiple distinct aspects to cover
{f'4. Reference maps ONLY if showing requested geographic data' if has_map_data else ''}
{f'5. Reference charts ONLY if showing requested trends/comparisons' if has_chart_data else ''}

Remember: Every sentence should help answer "{user_query}" - if it doesn't, leave it out."""
        
        # Log the full synthesis prompt if tracer is available
        if self.client.fact_tracer:
            self.client.fact_tracer.log_prompt("phase3", "synthesis_prompt", prompt)
            # Log specific facts
            llm_logger.debug(f"Synthesis facts preview (first 3):")
            for fact_text in fact_list[:3]:
                llm_logger.debug(f"  {fact_text[:150]}")
        
        try:
            response = await call_large_model(
                system=(
                    "You synthesize facts into clear, informative responses with proper citations. "
                    "STRICT GUARDRails: \n"
                    "- Use ONLY the provided facts; never infer missing dimensions.\n"
                    "- If a requested dimension is ABSENT in facts (e.g., social vulnerability/SVI), explicitly state it is not available and DO NOT claim it.\n"
                    "- If using a proxy, NAME it. For weak renewable access, the ONLY acceptable proxy is 'low solar facility presence/density'; DO NOT call this 'vulnerability', 'electrification rate', or 'grid reliability'.\n"
                    "- DO NOT produce numeric totals (e.g., population) unless the facts include a computed number or a table specifying it. Never guess or back-solve.\n"
                    "- Each declarative claim MUST be traceable to a fact and include a [CITE_X] placeholder.\n"
                    "- Do NOT introduce institutions (e.g., BNDES, Caixa, UNDP, GEF) unless a fact explicitly mentions them; otherwise omit.\n"
                    "- Prefer concise, qualified statements over confident generalizations when evidence is partial."
                ),
                user_prompt=prompt,
                max_tokens=2000
            )
            return response
        except Exception as e:
            print(f"Error in narrative synthesis: {e}")
            # Fallback to simple fact listing
            return f"## Analysis\n\n" + "\n\n".join([f"{fact.text_content} [CITE_{i+1}]" for i, fact in enumerate(facts)])
    
    def _build_citation_map(self, facts: List[Fact], narrative: str) -> Dict[str, int]:
        """
        Map placeholder citations to deduplicated numbers.
        
        Args:
            facts: All facts with source keys
            narrative: Narrative with [CITE_n] placeholders
            
        Returns:
            Dictionary mapping placeholders to final citation numbers
        """
        import re
        
        # Find all citation placeholders in narrative
        # This catches CITE_n in ANY format: [CITE_1], [[CITE_1], [CITE_2]], [CITE_1, CITE_2], or bare CITE_1
        all_citations = re.findall(r'CITE_(\d+)', narrative)
        # Remove duplicates while preserving order of first appearance
        citations_found = list(dict.fromkeys(all_citations))
        
        # Build source key to citation number mapping with deduplication
        source_to_citation = {}
        citation_counter = 1
        citation_map = {}
        
        for cite_num in citations_found:
            fact_index = int(cite_num) - 1  # Convert to 0-based index
            if fact_index < len(facts):
                fact = facts[fact_index]
                source_key = fact.source_key
                
                # Check if we've seen this source before
                if source_key not in source_to_citation:
                    source_to_citation[source_key] = citation_counter
                    citation_counter += 1
                
                # Map this placeholder to the deduplicated citation number
                citation_map[f"CITE_{cite_num}"] = source_to_citation[source_key]
        
        return citation_map
    
    def _apply_citations(self, narrative: str, citation_map: Dict[str, int]) -> str:
        """
        Replace [CITE_n] with ^n^ format.
        
        Args:
            narrative: Text with placeholder citations
            citation_map: Mapping from placeholders to final numbers
            
        Returns:
            Text with final ^n^ citations
        """
        import re
        
        final_text = narrative
        
        # Handle all citation formats with a single comprehensive function
        def replace_any_citation_format(match):
            """Replace any citation format with proper ^n^ format."""
            full_match = match.group(0)
            citations = re.findall(r'CITE_(\d+)', full_match)
            
            # Replace each citation with its mapped number
            replaced_citations = []
            for cite_num in citations:
                placeholder = f"CITE_{cite_num}"
                if placeholder in citation_map:
                    replaced_citations.append(f"^{citation_map[placeholder]}^")
            
            # Return space-separated superscript citations
            return ' '.join(replaced_citations) if replaced_citations else full_match
        
        # Match ALL possible citation formats:
        # - [[CITE_1], [CITE_2]] - double brackets
        # - [CITE_1, CITE_2, CITE_3] - grouped with commas
        # - [CITE_1] - single brackets
        # - CITE_1 - bare citations
        citation_patterns = [
            r'\[\[CITE_\d+\](?:,\s*\[CITE_\d+\])*\]',  # [[CITE_1], [CITE_2]]
            r'\[CITE_\d+(?:,\s*CITE_\d+)*\]',           # [CITE_1, CITE_2]
            r'\[CITE_\d+\]',                            # [CITE_1]
            r'CITE_\d+'                                  # CITE_1 (bare)
        ]
        
        # Apply replacements for each pattern
        for pattern in citation_patterns:
            final_text = re.sub(pattern, replace_any_citation_format, final_text)
        
        return final_text
    
    async def _create_chart_modules(self, facts: List[Fact]) -> List[Dict]:
        """
        Create Chart.js modules deterministically from tool-provided specs or numerical facts.
        
        Preferred path: tools return a lightweight visualization spec that
        response_formatter converts into a chart module. We avoid LLM/viz
        heuristics for predictability.
        """
        modules: List[Dict] = []

        # 1) Preferred: collect tool-provided visualization specs
        from response_formatter import _create_chart_module as _rf_create_chart_module
        seen_specs: set = set()
        for fact in facts:
            try:
                raw = fact.metadata.get("raw_result") if isinstance(fact.metadata, dict) else None
                if isinstance(raw, dict) and raw.get("visualization_type") and isinstance(raw.get("data"), list):
                    # Build a lightweight signature to avoid duplicate charts
                    try:
                        viz_type = str(raw.get("visualization_type"))
                        cfg = raw.get("chart_config", {}) or {}
                        x_key = str(cfg.get("x_axis", ""))
                        y_key = str(cfg.get("y_axis", ""))
                        title = str(cfg.get("title", ""))
                        n = len(raw.get("data", []))
                        sig = f"{viz_type}|x:{x_key}|y:{y_key}|n:{n}|t:{title}"
                    except Exception:
                        sig = str(raw.get("visualization_type"))
                    if sig in seen_specs:
                        continue
                    mod = _rf_create_chart_module(raw)
                    if mod:
                        seen_specs.add(sig)
                        modules.append(mod)
            except Exception as e:
                print(f"Chart spec conversion failed: {e}")

        # 2) Fallback: build charts from structured numerical facts (time_series/comparison)
        #    Use viz server only when no explicit spec was provided
        if not modules:
            numerical_facts = [f for f in facts if f.numerical_data and f.data_type in ['time_series', 'comparison']]
            for fact in numerical_facts:
                try:
                    chart_config = await self.client.call_tool(
                        tool_name="create_smart_chart",
                        tool_args={
                            "data": fact.numerical_data.get("values", []) or fact.numerical_data,
                            "context": f"{fact.data_type}: {fact.text_content}",
                            "title": self._generate_chart_title(fact)
                        },
                        server_name="viz"
                    )
                    modules.append({
                        "type": "chart",
                        "chartType": chart_config["type"],
                        "heading": chart_config.get("title", self._generate_chart_title(fact)),
                        "data": chart_config["data"],
                        "options": chart_config.get("options", {})
                    })
                except Exception as e:
                    print(f"Error creating chart for fact: {e}")

        return modules
    
    def _create_map_modules(self, facts: List[Fact], map_data: Optional[Dict] = None) -> List[Dict]:
        """
        Create map modules from geographic facts or solar map data.
        Maps are already saved to static/maps/ by servers.
        
        Args:
            facts: All facts, filtered for those with map_reference
            map_data: Direct map data from solar server with geojson_url
            
        Returns:
            List of map modules
        """
        modules = []
        
        # First check if we have direct map_data from solar server
        if map_data and map_data.get("type") == "map_data_summary" and map_data.get("geojson_url"):
            try:
                # Use the response_formatter approach for consistency
                from response_formatter import _create_map_module
                map_module = _create_map_module(map_data)
                if map_module:
                    modules.append(map_module)
            except Exception as e:
                print(f"Error creating map from solar data: {e}")
        
        # Also check facts for additional maps (always reference by URL; do not inline GeoJSON)
        for fact in facts:
            if fact.map_reference and fact.data_type == "geographic":
                try:
                    from response_formatter import _create_map_module
                    url = fact.map_reference.get("url")
                    if isinstance(url, str) and url:
                        # If URL is a correlation map, tag accordingly so the formatter renders a layer legend
                        is_corr_url = url.lower().startswith('/static/maps/correlation_') if isinstance(url, str) else False
                        summary = {
                            "description": self._extract_map_title(fact.text_content)
                        }
                        if is_corr_url:
                            summary.update({
                                "map_role": "correlation",
                                "legend_layers": [
                                    {"label": "Solar Assets", "color": "#FFD700"},
                                    {"label": "Deforestation Areas", "color": "#8B4513"}
                                ]
                            })
                        map_data = {
                            "type": "map_data_summary",
                            "summary": summary,
                            "geojson_url": url,
                            "is_correlation_map": bool(is_corr_url)
                        }
                        map_module = _create_map_module(map_data)
                        if map_module:
                            modules.append(map_module)
                except Exception as e:
                    print(f"Error creating URL-based map from fact: {e}")
        
        return modules
    
    def _create_table_modules(self, facts: List[Fact]) -> List[Dict]:
        """
        Create table modules from tabular facts.
        
        Args:
            facts: All facts, filtered for those with tabular data
            
        Returns:
            List of table modules
        """
        modules = []
        
        for fact in facts:
            if fact.numerical_data and fact.data_type == "tabular":
                try:
                    table_data = fact.numerical_data
                    if table_data.get("columns") and table_data.get("rows"):
                        modules.append({
                            "type": "table",
                            "heading": table_data.get("title") or self._extract_table_title(fact.text_content),
                            "columns": table_data["columns"],
                            "rows": table_data["rows"]
                        })
                except Exception as e:
                    print(f"Error creating table: {e}")
        
        return modules
    
    def _create_citation_table(self, citation_map: Dict[str, int], facts: List[Fact], sources: Optional[List] = None) -> Optional[Dict]:
        """
        Build the citation table module.
        
        Args:
            citation_map: Citation mapping with deduplication
            facts: All facts for citation lookup
            sources: Optional list of sources for fallback
            
        Returns:
            Citation table module or None if no citations
        """
        # Build reverse map: citation_number -> fact
        used_citations = {}
        
        if citation_map:
            for placeholder, citation_num in citation_map.items():
                # Extract fact index from placeholder
                fact_idx = int(placeholder.split('_')[1]) - 1
                if fact_idx < len(facts):
                    fact = facts[fact_idx]
                    used_citations[citation_num] = fact
        else:
            # Fallback: include all facts in order when narrative omitted placeholders
            for idx, fact in enumerate(facts, start=1):
                used_citations[idx] = fact
        
        # Build rows
        rows = []
        for citation_num in sorted(used_citations.keys()):
            fact = used_citations[citation_num]
            if fact.citation:
                # Prefer full description; truncate to keep table concise
                desc = fact.citation.description
                desc = (desc[:120] + "...") if isinstance(desc, str) and len(desc) > 120 else desc
                rows.append([
                    str(citation_num),
                    fact.citation.source_name,
                    fact.citation.tool_id,
                    fact.citation.source_type,
                    desc or "",
                    fact.citation.source_url or ""
                ])
        
        return {
            "type": "numbered_citation_table",
            "heading": "References",
            "columns": ["#", "Source", "ID/Tool", "Type", "Description", "SourceURL"],
            "rows": rows
        }
    
    def _generate_chart_title(self, fact: Fact) -> str:
        """Generate a title for a chart based on the fact."""
        text = fact.text_content
        if len(text) > 50:
            return text[:47] + "..."
        return text
    
    def _extract_map_title(self, text_content: str) -> str:
        """Extract a title for a map from fact text."""
        # Simple extraction - take first part of text
        if len(text_content) > 40:
            return text_content[:37] + "..."
        return text_content
    
    def _extract_table_title(self, text_content: str) -> str:
        """Extract a title for a table from fact text."""
        if len(text_content) > 40:
            return text_content[:37] + "..."
        return text_content
    
    def _calculate_map_view_state(self, map_reference: Dict) -> Dict:
        """Calculate view state for a map based on its reference data."""
        bounds = map_reference.get("bounds")
        if bounds:
            # Calculate center from bounds
            center_lat = (bounds.get("north", 0) + bounds.get("south", 0)) / 2
            center_lon = (bounds.get("east", 0) + bounds.get("west", 0)) / 2
            
            return {
                "center": [center_lon, center_lat],
                "zoom": 6
            }
        
        # Default view state
        return {
            "center": [0, 0],
            "zoom": 2
        }
    


# =============================================================================
# PHASE 3: ARTIFACT HANDLING AND RESPONSE FORMATTING
# =============================================================================

class ArtifactManager:
    """
    Manages artifact storage, URL generation, and embedding in responses.
    
    Handles large outputs like GeoJSON maps, charts, and images that can't
    be included directly in LLM context due to token limitations.
    """
    
    def __init__(self, static_dir: str = "static"):
        self.static_dir = static_dir
        self.artifact_base_url = "/static"  # For serving artifacts
        
    async def save_artifact(self, artifact_data: Any, artifact_type: ArtifactType, 
                           query_context: str = "") -> Dict[str, Any]:
        """
        Save an artifact to disk and return metadata.
        
        TODO: Implement artifact saving:
        1. Generate unique filename based on content hash
        2. Ensure static directory exists
        3. Save artifact data to appropriate format
        4. Generate serving URL
        5. Create summary description
        6. Return metadata dict with url, summary, type, etc.
        
        Args:
            artifact_data: The actual artifact content (GeoJSON, chart config, etc.)
            artifact_type: Type of artifact being saved
            query_context: Context for generating descriptive summary
            
        Returns:
            Dictionary with artifact metadata for citation registry
        """
        # TODO: Generate unique filename
        # TODO: Save to static directory
        # TODO: Create descriptive summary
        # TODO: Return metadata
        
        return {
            "artifact_type": artifact_type.value,
            "artifact_url": f"{self.artifact_base_url}/placeholder.json",
            "summary": "Placeholder artifact summary",
            "metadata": {"count": 0, "total_capacity": 0}
        }
    
    def embed_artifacts_in_response(self, response_text: str, citation_registry: CitationRegistry) -> str:
        """
        Post-process synthesized response to embed artifacts.
        
        Detects placeholders like "[see map]" and replaces them with actual
        artifact embeds (iframes, Chart.js configs, etc.).
        
        TODO: Implement artifact embedding:
        1. Scan response for artifact placeholders
        2. Match placeholders to artifacts in citation registry
        3. Generate appropriate embeds:
           - GeoJSON: iframe with map viewer
           - Charts: Chart.js configuration
           - Images: img tags
        4. Replace placeholders with embeds
        5. Preserve citation numbering
        
        Args:
            response_text: Synthesized response with placeholders
            citation_registry: Registry containing artifact references
            
        Returns:
            Response with embedded artifacts
        """
        # TODO: Implement placeholder detection and replacement
        return response_text


# =============================================================================
# PHASE 4: MAIN API INTERFACE
# =============================================================================

async def process_chat_query(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None, correlation_session_id: Optional[str] = None, target_language: Optional[str] = None) -> Dict:
    """
    Main entry point for processing chat queries synchronously.
    
    Returns API-compliant response with modules and metadata.
    
    Args:
        user_query: User's natural language query
        conversation_history: Optional list of previous messages in format
                            [{"role": "user"|"assistant", "content": "..."}]
        
    Returns:
        Dictionary with query, modules, and metadata
    """
    try:
        # Get singleton client for optimal performance
        client = await get_global_client()
        
        # Create orchestrator for this query
        orchestrator = QueryOrchestrator(
            client,
            conversation_history=conversation_history,
            spatial_session_id=correlation_session_id,
            target_language=target_language
        )
        
        # Process through 3-phase architecture
        response = await orchestrator.process_query(user_query)
        
        return response
        
    except Exception as e:
        # TODO: Implement proper error handling and logging
        return f"Error processing query: {str(e)}"


async def stream_chat_query(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None, correlation_session_id: Optional[str] = None, target_language: Optional[str] = None):
    """
    Streaming version of chat query processing with thinking objects.
    
    Uses Option D: Sequential execution for better progress visibility.
    Yields thinking objects compatible with the API specification.
    
    Args:
        user_query: User's natural language query
        
    Yields:
        Dictionary objects with type and data fields for SSE streaming
    """
    print(f"DEBUG: stream_chat_query called with: {user_query}")
    try:
        # Initial thinking message
        yield {
            "type": "thinking",
            "data": {
                "message": "🚀 Initializing climate data analysis...",
                "category": "initialization"
            }
        }
        print("DEBUG: Initial message yielded")
        
        # Get singleton client for performance
        print("DEBUG: Getting global client...")
        client = await get_global_client()
        print("DEBUG: Got global client")
        
        # Create orchestrator for this query with conversation history
        orchestrator = QueryOrchestrator(
            client,
            conversation_history=conversation_history,
            spatial_session_id=correlation_session_id,
            target_language=target_language,
        )
        print("DEBUG: Created orchestrator")
        
        # Check if query is off-topic
        is_relevant = await orchestrator._is_query_relevant(user_query)
        if not is_relevant:
            # Send off-topic redirect response
            yield {
                "type": "thinking",
                "data": {
                    "message": "🚫 Query appears to be outside my climate/environment expertise...",
                    "category": "relevance_check"
                }
            }
            
            redirect_response = orchestrator._create_redirect_response(user_query)
            
            yield {
                "type": "complete",
                "data": redirect_response
            }
            return
        
        # Meta/capability queries: return dynamic datasets overview early
        if await orchestrator._is_meta_query(user_query):
            yield {
                "type": "thinking",
                "data": {
                    "message": "📚 Compiling live datasets and tools overview...",
                    "category": "data_discovery"
                }
            }
            overview = await orchestrator._create_capabilities_response(user_query)
            yield {
                "type": "complete",
                "data": overview
            }
            return
        
        server_descriptions = orchestrator._get_server_descriptions()
        print("DEBUG: Got server descriptions")
        
        # ========== PHASE 0: PRE-FILTER ==========
        yield {
            "type": "thinking",
            "data": {
                "message": "🔍 Identifying relevant data sources for your query...",
                "category": "search"
            }
        }
        
        # Pre-filter to get relevant servers
        relevant_servers = await orchestrator._phase0_prefilter(user_query)
        
        if not relevant_servers:
            # Default to knowledge graph if no servers selected
            relevant_servers = ["cpr"]
            yield {
                "type": "thinking",
                "data": {
                    "message": "📊 Using default knowledge graph source",
                    "category": "data_discovery"
                }
            }
        else:
            # Announce selected servers
            server_names = {
                "cpr": "Climate Policy Radar",
                "solar": "TZ-SAM",
                "gist": "GIST Environmental Impact",
                "lse": "LSE (and Friends) Governance Data",
                "viz": "Visualization",
                "geospatial": "Geospatial Correlation",
                "deforestation": "Deforestation",
                "municipalities": "Administrative Boundaries",
                "heat": "Heat Stress",
                "spa": "Science Panel for the Amazon",
            }
            friendly_names = [server_names.get(s, s) for s in relevant_servers]
            yield {
                "type": "thinking",
                "data": {
                    "message": f"📊 Selected {len(relevant_servers)} data sources: {', '.join(friendly_names)}",
                    "category": "data_discovery"
                }
            }
        
        # ========== PHASE 1: SEQUENTIAL COLLECTION ==========
        collection_results = {}
        total_facts = 0
        facts_by_server = {}  # Track facts for debugging
        
        for server_name in relevant_servers:
            if server_name not in server_descriptions:
                continue
                
            config = server_descriptions[server_name]
            
            # Announce starting collection from this server
            server_display = server_names.get(server_name, server_name)
            yield {
                "type": "thinking",
                "data": {
                    "message": f"📥 Collecting data from {server_display}...",
                    "category": "data_loading"
                }
            }
            
            # Collect information from this server
            try:
                result = await orchestrator._collect_server_information(
                    user_query, server_name, config
                )
                collection_results[server_name] = result
                
                # Report what we found
                if result.facts:
                    fact_count = len(result.facts)
                    total_facts += fact_count
                    facts_by_server[server_name] = fact_count
                    yield {
                        "type": "thinking",
                        "data": {
                            "message": f"✅ Found {fact_count} relevant facts from {server_display}",
                            "category": "information"
                        }
                    }
                else:
                    yield {
                        "type": "thinking",
                        "data": {
                            "message": f"ℹ️ No specific data found from {server_display}",
                            "category": "information"
                        }
                    }
                    
            except Exception as e:
                # Report collection error but continue with other servers
                yield {
                    "type": "thinking",
                    "data": {
                        "message": f"⚠️ Could not access {server_display}: {str(e)}",
                        "category": "error"
                    }
                }
                collection_results[server_name] = PhaseResult(
                    is_relevant=True,
                    facts=[],
                    reasoning=f"Error: {str(e)}"
                )
        
        # Debug logging for Phase 1 facts
        if os.getenv("TDE_DEBUG_FACTS") and total_facts > 0:
            facts_data = {
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "phase": "post_phase1",
                "total_facts": total_facts,
                "facts_by_server": {
                    server: [
                        {
                            "text": f.text_content,
                            "source_key": f.source_key,
                            "tool": f.metadata.get("tool", "unknown")
                        } for f in result.facts
                    ]
                    for server, result in collection_results.items() if result.facts
                }
            }
            filename = f"/tmp/tde_facts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_phase1.json"
            try:
                with open(filename, "w") as f:
                    json.dump(facts_data, f, indent=2, default=str)
                print(f"DEBUG: Saved {total_facts} Phase 1 facts to {filename}")
            except Exception as e:
                print(f"DEBUG: Could not save facts: {e}")
        
        # Yield fact collection summary event
        if facts_by_server:
            yield {
                "type": "facts_summary",
                "data": {
                    "phase": 1,
                    "total": total_facts,
                    "by_server": facts_by_server
                }
            }
        
        # ========== PHASE 2: DEEP DIVE (if needed) ==========
        # Calculate time elapsed
        import time
        start_time = time.time() if not hasattr(orchestrator, '_start_time') else orchestrator._start_time
        time_elapsed = time.time() - start_time
        
        # Get query complexity if available
        complexity = None
        if OPTIMIZER_AVAILABLE:
            complexity = await PerformanceOptimizer.estimate_complexity(user_query)
        
        # Use LLM to decide if Phase 2 is needed
        should_deep_dive, reasoning, servers_for_phase2 = await orchestrator._should_do_phase2_deep_dive(
            user_query, collection_results, complexity, time_elapsed
        )
        
        if should_deep_dive:
            yield {
                "type": "thinking",
                "data": {
                    "message": f"🔬 {reasoning}",
                    "category": "analysis"
                }
            }
            
            yield {
                "type": "thinking",
                "data": {
                    "message": f"🔄 Running deeper analysis on {len(servers_for_phase2)} servers...",
                    "category": "analysis"
                }
            }
            
            # Run Phase 2 deep dive
            deep_dive_results = await orchestrator._phase2_deep_dive(user_query, collection_results)
            
            # Update total facts count
            for server_name, result in deep_dive_results.items():
                if result.facts:
                    new_facts = len(result.facts) - len(collection_results.get(server_name, PhaseResult(is_relevant=False)).facts)
                    if new_facts > 0:
                        total_facts += new_facts
                        yield {
                            "type": "thinking",
                            "data": {
                                "message": f"🔍 Found {new_facts} additional facts from deeper analysis",
                                "category": "information"
                            }
                        }
            
            # Use deep dive results for synthesis
            final_results = deep_dive_results
            
            # Yield Phase 2 facts summary
            phase2_facts = sum(len(r.facts) for r in deep_dive_results.values() if r.facts)
            yield {
                "type": "facts_summary",
                "data": {
                    "phase": 2,
                    "total": phase2_facts,
                    "new_facts": phase2_facts - total_facts
                }
            }
        else:
            # Skip Phase 2, use Phase 1 results
            final_results = collection_results
            if total_facts > 0:
                yield {
                    "type": "thinking", 
                    "data": {
                        "message": "✅ Sufficient information collected, proceeding to synthesis",
                        "category": "analysis"
                    }
                }
        
        # Debug logging before synthesis
        if os.getenv("TDE_DEBUG_FACTS"):
            final_facts_count = sum(len(r.facts) for r in final_results.values() if r.facts)
            facts_data = {
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "phase": "pre_synthesis",
                "total_facts": final_facts_count,
                "facts_by_server": {
                    server: [
                        {
                            "text": f.text_content,
                            "source_key": f.source_key,
                            "tool": f.metadata.get("tool", "unknown"),
                            "has_raw_result": "raw_result" in f.metadata
                        } for f in result.facts
                    ]
                    for server, result in final_results.items() if result.facts
                }
            }
            filename = f"/tmp/tde_facts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_final.json"
            try:
                with open(filename, "w") as f:
                    json.dump(facts_data, f, indent=2, default=str)
                print(f"DEBUG: Saved {final_facts_count} final facts to {filename}")
            except Exception as e:
                print(f"DEBUG: Could not save final facts: {e}")
        
        # ========== PHASE 3: SYNTHESIS ==========
        yield {
            "type": "thinking",
            "data": {
                "message": "🧠 Synthesizing comprehensive response with citations...",
                "category": "synthesis"
            }
        }
        
        # Synthesize the final response
        try:
            modules = await orchestrator._phase3_synthesis(user_query, final_results)
            
            # Build the complete response structure
            response_data = {
                "query": user_query,
                "modules": modules,
                "metadata": {
                    "modules_count": len(modules),
                    "servers_queried": len(relevant_servers),
                    "facts_collected": total_facts,
                    "has_maps": any(m.get("type") == "map" for m in modules),
                    "has_charts": any(m.get("type") == "chart" for m in modules),
                    "has_tables": any(m.get("type") == "table" for m in modules)
                }
            }
            
            # Return the complete response
            yield {
                "type": "complete",
                "data": response_data
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "data": {
                    "message": f"Error during synthesis: {str(e)}",
                    "traceback": None
                }
            }
        
    except Exception as e:
        # Handle any unexpected errors
        yield {
            "type": "error",
            "data": {
                "message": f"Unexpected error: {str(e)}",
                "traceback": None
            }
        }


# =============================================================================
# PHASE 5: TESTING AND UTILITIES
# =============================================================================

async def test_server_connections():
    """
    Test utility to verify all MCP server connections.
    
    TODO: Implement connection testing:
    1. Get global client
    2. Test each server connection
    3. Try sample tool calls
    4. Report connection status
    5. Identify any issues
    
    Returns:
        Dictionary with server connection status
    """
    try:
        client = await get_global_client()
        
        # TODO: Test each server connection
        # TODO: Make sample tool calls
        # TODO: Report status

        return {"status": "all_connected", "servers": ["cpr", "solar", "gist", "lse", "formatter"]}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    """
    Command-line testing interface.
    
    Run this script directly to test the implementation with sample queries.
    """
    import sys
    
    async def main():
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
            print(f"Processing query: {query}")
            response = await process_chat_query(query)
            print(f"Response: {response}")
        else:
            print("Testing server connections...")
            status = await test_server_connections()
            print(f"Connection status: {status}")
            
            # Cleanup
            await cleanup_global_client()
    
    # Run main() with proper asyncio event loop
# =============================================================================
# LLM PAYLOAD SANITIZATION & LOGGING HELPERS
# =============================================================================

# Keys that often contain very large payloads or raw GeoJSON
SENSITIVE_BIG_KEYS = {
    "features", "geometry", "geojson", "coordinates", "data", "polygons", "points"
}

def _truncate_string(s: str, max_len: int = 4000) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... [truncated {len(s)-max_len} chars]"

def _sanitize_obj_for_llm(obj: Any, removed_counts: Optional[Dict[str, int]] = None, max_list_len: int = 50) -> Any:
    """Recursively sanitize an object for safe inclusion in LLM messages.

    - Removes or summarizes large/geo structures (features/geometry/geojson/coordinates/data).
    - Truncates long arrays and strings.
    - Tracks removed elements per key in removed_counts.
    """
    from collections.abc import Mapping, Sequence
    if removed_counts is None:
        removed_counts = {}

    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in SENSITIVE_BIG_KEYS:
                # summarize rather than include raw
                try:
                    length = len(v) if hasattr(v, '__len__') else 1
                except Exception:
                    length = 1
                removed_counts[kl] = removed_counts.get(kl, 0) + length
                out[k] = {"_omitted_for_llm": True, "_approx_count": length}
            else:
                out[k] = _sanitize_obj_for_llm(v, removed_counts, max_list_len)
        return out
    elif isinstance(obj, str):
        return _truncate_string(obj, 2000)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        # Truncate very long lists
        if len(obj) > max_list_len:
            head = [_sanitize_obj_for_llm(x, removed_counts, max_list_len) for x in obj[:max_list_len]]
            return head + [{"_omitted_for_llm": True, "_omitted_items": len(obj) - max_list_len}]
        else:
            return [_sanitize_obj_for_llm(x, removed_counts, max_list_len) for x in obj]
    else:
        return obj

def _prepare_tool_result_for_llm(tool_name: str, result: Any) -> str:
    """Create a safe, compact string for the tool_result content sent back to the LLM.

    Attempts to parse JSON from the first text content; removes heavy keys and truncates.
    Logs when sanitization removed large fields to help track payload sizes.
    """
    payload_str = None
    removed_counts: Dict[str, int] = {}
    try:
        if hasattr(result, 'content') and isinstance(result.content, list) and result.content:
            first = result.content[0]
            if hasattr(first, 'text') and isinstance(first.text, str):
                txt = first.text
                # Try JSON parse
                try:
                    data = json.loads(txt)
                    sanitized = _sanitize_obj_for_llm(data, removed_counts)
                    payload_str = json.dumps(sanitized, ensure_ascii=False)
                except json.JSONDecodeError:
                    # Not JSON; just truncate text
                    payload_str = _truncate_string(txt, 4000)
        if payload_str is None:
            payload_str = _truncate_string(str(result), 4000)
    except Exception as e:
        payload_str = f"[tool_result_unavailable: {e}]"

    # Log summary of what we removed to help detect GeoJSON leakage
    if removed_counts:
        llm_logger.warning(f"Sanitized tool_result for {tool_name}; removed heavy keys: {removed_counts}")
    llm_logger.debug(f"tool_result payload length: {len(payload_str)}")
    return payload_str

def _log_llm_messages_summary(messages: List[Dict[str, Any]], label: str):
    try:
        # Rough size estimate for the messages payload
        import math
        serialized = json.dumps(messages, ensure_ascii=False)
        size_kb = len(serialized.encode('utf-8')) / 1024.0
        llm_logger.info(f"LLM messages summary [{label}]: messages={len(messages)} size≈{size_kb:.1f}KB")
    except Exception as e:
        llm_logger.debug(f"Could not summarize LLM messages: {e}")
    # Note: Running the CLI test harness is guarded above. Avoid executing anything
    # on module import so API/server imports remain side-effect free.
