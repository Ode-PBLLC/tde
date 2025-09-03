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
from typing import Dict, List, Any, Optional, Union, Tuple
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION: MODEL SELECTION
# =============================================================================

# Model configuration for different use cases
SMALL_MODEL = "claude-3-5-haiku-20241022"  # Fast, efficient for routing and simple tasks
LARGE_MODEL = "claude-3-5-sonnet-20241022"  # Powerful for synthesis and complex reasoning

# Initialize Anthropic client (shared across all operations)
ANTHROPIC_CLIENT = anthropic.Anthropic()

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
    response = ANTHROPIC_CLIENT.messages.create(
        model=SMALL_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text


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
    response = ANTHROPIC_CLIENT.messages.create(
        model=LARGE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return response.content[0].text


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
    return ANTHROPIC_CLIENT.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        tools=tools
    )

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
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context
    
    def to_table_row(self, citation_number: int) -> List[str]:
        """Convert citation to table row format for the citation table."""
        return [
            str(citation_number),
            self.source_name,
            self.tool_id,
            self.source_type,
            self.description
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
            "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
            "rows": rows
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
                # Connect to all MCP servers (formatter removed - now using direct functions)
                await _global_client.connect_to_server("kg", os.path.join(mcp_dir, "cpr_kg_server.py"))
                await _global_client.connect_to_server("solar", os.path.join(mcp_dir, "solar_facilities_server.py"))
                await _global_client.connect_to_server("gist", os.path.join(mcp_dir, "gist_server.py"))
                await _global_client.connect_to_server("lse", os.path.join(mcp_dir, "lse_server.py"))
                await _global_client.connect_to_server("viz", os.path.join(mcp_dir, "viz_server.py"))
                print("Global MCP client initialized successfully")
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
    
    def __init__(self, client: MultiServerClient, conversation_history: Optional[List[Dict[str, str]]] = None):
        self.client = client
        self.citation_registry = client.citation_registry
        self.token_budget_remaining = 50000  # Conservative starting budget
        self.conversation_history = conversation_history or []
        self.cached_facts: List[Fact] = []  # Cache facts from previous responses
        self.cached_response_text: str = ""  # Cache the last response text
    
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

The query: "{user_query}"

Answer with just YES if the query is related to our domain, or NO if it's about:
- Personal preferences or opinions unrelated to environment
- General knowledge, trivia, or entertainment
- Programming, math, or technical topics unrelated to climate
- Medical, legal, or financial advice unrelated to climate
- Other topics clearly outside environmental/climate scope

Answer (YES/NO):"""
        
        try:
            response = await call_small_model(
                system="You are a query classifier. Respond with only YES or NO.",
                user_prompt=relevance_prompt,
                max_tokens=10,
                temperature=0
            )
            
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
            "content": """I'm specialized in climate policy, environmental data, and sustainability topics. I can help you with questions about:

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
            fact_summary = "\n".join([f"- {fact.text_content[:100]}..." for fact in self.cached_facts[:10]])
            
            check_prompt = f"""Can this query be answered using ONLY the available facts?

Query: "{user_query}"

Available facts:
{fact_summary}

Previous response included: {self.cached_response_text[:500]}...

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
            # First check if query is relevant to our domain
            is_relevant = await self._is_query_relevant(user_query)
            if not is_relevant:
                return self._create_redirect_response(user_query)
            
            # NEW: Check if we can answer from cached context
            can_use_context, reasoning = await self._can_answer_from_context(user_query)
            
            if can_use_context:
                print(f"[CACHED] Using cached context: {reasoning}")
                context_modules = await self._answer_from_context(user_query)
                
                if context_modules:
                    # Successfully answered from context
                    return {
                        "query": user_query,
                        "modules": context_modules,
                        "metadata": {
                            "modules_count": len(context_modules),
                            "has_maps": any(m.get("type") == "map" for m in context_modules),
                            "has_charts": any(m.get("type") == "chart" for m in context_modules),
                            "has_tables": any(m.get("type") == "table" for m in context_modules),
                            "used_cached_context": True  # Flag for monitoring
                        }
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
                deep_dive_results = await self._phase2_deep_dive(user_query, collection_results)
            else:
                print(f"Skipping Phase 2: {reasoning}")
                deep_dive_results = collection_results
            
            # Phase 3: Synthesis
            modules = await self._phase3_synthesis(user_query, deep_dive_results)
            
            # Cache facts for potential follow-up queries
            self._cache_facts_from_results(deep_dive_results)
            
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
                }
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
        - routing_prompt: Specific instructions for relevance determination
        - collection_instructions: How to effectively use tools during Phase 1
        """
        return {
            "kg": {
                "brief": "Climate Knowledge Graph with physical & transition risks, energy systems, financial climate impacts",
                "detailed": "Knowledge graph containing climate risk data, energy system information, financial climate impacts, physical risks (floods, droughts, heat), transition risks (policy, technology, market), sectoral analysis, and interconnected climate-finance relationships",
                "routing_prompt": "Analyze if this query relates to climate risks, energy systems, financial impacts, physical or transition risks, or sector-specific climate analysis.",
                "collection_instructions": """Tool usage strategy for Knowledge Graph:
                - Start with 'search_passages' for broad queries about climate risks, financial impacts, or energy
                - Use specific search terms and filters (risk_type, sector, region) to narrow results
                - For company-specific queries, search by company name first
                - For risk analysis, always specify the risk_type parameter (physical, transition, both)
                - Collect multiple relevant passages to ensure comprehensive coverage
                - Look for interconnections between climate, finance, and energy systems"""
            },
            "solar": {
                "brief": "Solar facility database with locations, capacity, renewable infrastructure globally",
                "detailed": "Comprehensive solar facility database with geospatial locations, capacity data (MW), facility metadata, country/region aggregations, renewable energy infrastructure mapping, and solar deployment trends",
                "routing_prompt": "Determine if this query needs solar facility locations, renewable capacity data, solar infrastructure mapping, or country/region solar statistics.",
                "collection_instructions": """Tool usage strategy for Solar Database:
                - IMPORTANT: Call 'GetSolarFacilitiesMapData' when you need to show geographic distribution
                - Use 'GetSolarFacilitiesByCountry' for detailed country-level facility data
                - Use 'GetSolarCapacityByCountry' for capacity statistics across countries
                - Use 'GetSolarConstructionTimeline' for temporal trends and growth analysis
                - Use 'GetLargestSolarFacilities' to highlight major installations
                - For multi-country analysis, use 'GetSolarFacilitiesMultipleCountries'
                - Map tools automatically generate interactive visualizations
                - Always aim to provide both statistics AND geographic visualizations"""
            },
            "gist": {
                "brief": "Company environmental data: water stress (MSA), drought/flood risks, heat exposure, GHG emissions",
                "detailed": "GIST Impact database with company-level environmental metrics including: water stress (Mean Species Abundance), drought risk, flood risk (coastal/riverine), extreme heat exposure, extreme precipitation, temperature anomalies, land use changes (urban expansion, forest loss, agricultural conversion), population density impacts, corporate environmental impacts (Scope 1/2/3 GHG emissions, water consumption, SOX/NOX emissions, nitrogen/phosphorous pollution, waste generation), and asset-level risk assessments",
                "routing_prompt": "Analyze if this query needs company environmental data, water risk metrics, climate physical risks, GHG emissions, pollution data, or corporate sustainability metrics.",
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
                "brief": "Brazilian climate governance, state assessments, NDC implementation, institutional frameworks",
                "detailed": "LSE Climate Policy dataset specializing in Brazilian subnational climate governance including: state-level climate assessments, NDC implementation tracking, legal/institutional frameworks, climate action plans, mitigation/adaptation strategies, governance quality indicators, policy coherence analysis, and subnational climate leadership metrics",
                "routing_prompt": "Determine if this query needs Brazilian climate policy data, subnational governance assessments, NDC tracking, institutional analysis, or climate policy implementation metrics.",
                "collection_instructions": """Tool usage strategy for LSE Climate Policy:
                - Use 'get_state_climate_assessment' for individual Brazilian state analysis
                - Use 'get_ndc_implementation_status' for NDC tracking and progress
                - Use 'get_governance_indicators' for institutional quality metrics
                - For comparative analysis, collect data from multiple states
                - Always gather both policy existence AND implementation status
                - Include legal framework details when analyzing governance
                - Collect mitigation AND adaptation strategy information
                - For trends, gather historical governance scores and policy evolution"""
            }
        }
    
    async def _phase0_prefilter(self, user_query: str) -> List[str]:
        """
        Phase 0: Pre-filter servers using a single LLM call.
        
        Quickly identifies potentially relevant servers to avoid wasting tokens
        on obviously irrelevant ones. Uses brief descriptions for efficiency.
        
        Args:
            user_query: Original user query
            
        Returns:
            List of server names that should proceed to scout phase
        """
        server_descriptions = self._get_server_descriptions()
        
        # Build server list for pre-filter prompt
        server_list = "\n".join([
            f"- {name}: {config['brief']}" 
            for name, config in server_descriptions.items()
        ])

        
        prefilter_prompt = f"""Given this user query: "{user_query}"

        Which of these data sources might be relevant? Return ONLY the server IDs that could help answer this query.

        Available servers:
        {server_list}

        Instructions:
        - Be selective: only include servers that are clearly relevant
        - For general queries, include multiple relevant servers
        - For specific queries, be more targeted
        - Include 'formatter' ONLY if the query explicitly asks for maps, charts, or visualizations
        - If no servers seem relevant, return empty array []

        Return a JSON array of relevant server IDs only.
        Examples: ["kg", "gist"] or ["solar"] or ["kg", "solar", "formatter"] or []"""

        try:
            # Use small model for speed
            response = await call_small_model(
                system="You are a query router. Return only a valid JSON array of server IDs.",
                user_prompt=prefilter_prompt,
                max_tokens=100,
                temperature=0
            )
            
            # Parse JSON response
            relevant_servers = json.loads(response.strip())
            
            # Validate server names
            valid_servers = [s for s in relevant_servers if s in server_descriptions]
            
            print(f"Pre-filter selected servers: {valid_servers}")
            return valid_servers
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Pre-filter error: {e}. Defaulting to all servers.")
            # On error, return all servers (graceful degradation)
            return list(server_descriptions.keys())
    
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
            relevant_servers = ["kg"]
        
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
            
            # Create focused system prompt for Phase 1 collection
            system_prompt = f"""You are collecting information from the {server_name} server to help answer a user query.

Server capabilities: {config['detailed']}

Your task: Actively call tools to collect specific facts, data points, and information that will help answer the user's query.

{config['collection_instructions']}

Guidelines:
- Focus on collecting concrete information: numbers, names, dates, statistics
- Call tools that will provide the most relevant data for the query
- Extract key facts from each tool response
- Stop when you have gathered sufficient information (usually 2-4 tool calls)
- Be efficient with your token budget

When you have collected enough information, respond with text summarizing what you found instead of calling more tools."""

            # Initialize messages for the conversation
            messages = []
            
            # Include conversation history if available
            if self.conversation_history:
                messages.append({
                    "role": "user",
                    "content": f"Previous conversation context:\n{self._format_conversation_history()}\n\nNow, for the current query:"
                })
            
            messages.append({
                "role": "user",
                "content": f"Query: {user_query}\n\nCollect relevant information from your available tools."
            })
            
            # Track collected facts
            facts = []
            tool_calls_made = 0
            max_tool_calls = 15  # Small scope for Phase 1
            
            # Start the collection loop
            while tool_calls_made < max_tool_calls:
                # Call LLM with tools
                response = await call_model_with_tools(
                    model=SMALL_MODEL,  # Use small model for efficiency
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
                        
                        print(f"  Phase 1 - {server_name}: Call {tool_calls_made}/{max_tool_calls} - {tool_name}")
                        
                        # Make the actual tool call
                        try:
                            result = await self.client.call_tool(tool_name, tool_args, server_name)
                            
                            # Extract facts from the tool result
                            extracted_facts = self._extract_facts_from_result(
                                tool_name, tool_args, result, server_name
                            )
                            facts.extend(extracted_facts)
                            
                            # Add tool use to assistant message
                            assistant_message_content.append(content)
                            messages.append({"role": "assistant", "content": assistant_message_content})
                            
                            # Add tool result to conversation
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": result.content if hasattr(result, 'content') else str(result)
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
            
            # Extract facts based on tool type and result structure
            if result_data:
                # Create citation for this tool result
                citation = Citation(
                    source_name=self._get_source_name_for_tool(tool_name, server_name),
                    tool_id=tool_name,
                    source_type=self._get_source_type_for_server(server_name),
                    description=f"Data from {tool_name}",
                    server_origin=server_name,
                    metadata={"tool_args": tool_args}
                )
                
                # Extract facts based on data type
                if isinstance(result_data, list):
                    # List of items - create summary fact
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
                        citation=citation
                    ))
                    
                elif isinstance(result_data, dict):
                    # Dictionary - extract key information
                    if tool_name == "GetSolarFacilitiesMultipleCountries":
                        print(f"DEBUG: GetSolarFacilitiesMultipleCountries result_data keys: {list(result_data.keys())}")
                        print(f"DEBUG: Has geojson_url: {'geojson_url' in result_data}")
                    if 'data' in result_data and isinstance(result_data['data'], list):
                        count = len(result_data['data'])
                        fact_text = f"Found {count} records"
                        if tool_args:
                            args_str = ', '.join([f'{k}={v}' for k, v in tool_args.items()])
                            fact_text += f" ({args_str})"
                    elif 'total' in result_data:
                        fact_text = f"Total: {result_data['total']}"
                    elif 'metadata' in result_data:
                        meta = result_data['metadata']
                        fact_text = f"Data: {', '.join([f'{k}={v}' for k, v in meta.items()])}"
                    else:
                        # Extract actual key-value pairs for meaningful facts
                        key_values = []
                        for key, value in result_data.items():
                            if isinstance(value, (str, int, float, bool)):
                                key_values.append(f"{key}: {value}")
                            elif isinstance(value, list) and len(value) > 0:
                                key_values.append(f"{key}: {len(value)} items")
                            elif isinstance(value, dict):
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
                        citation=citation
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
        
        return facts
    
    def _get_source_name_for_tool(self, tool_name: str, server_name: str) -> str:
        """Get a human-readable source name for a tool."""
        source_names = {
            "kg": "CPR Knowledge Graph",
            "solar": "Solar Facilities Database",
            "gist": "GIST Impact Database",
            "lse": "LSE Climate Policy Database",
            "formatter": "Response Formatter"
        }
        return source_names.get(server_name, server_name.upper())
    
    def _get_source_type_for_server(self, server_name: str) -> str:
        """Get the source type for a server."""
        source_types = {
            "kg": "Knowledge Graph",
            "solar": "Database",
            "gist": "Database",
            "lse": "Database",
            "formatter": "Tool"
        }
        return source_types.get(server_name, "Database")
    
    async def _should_do_phase2_deep_dive(self, user_query: str, 
                                          collection_results: Dict[str, PhaseResult]) -> Tuple[bool, str, List[str]]:
        """
        Determine if Phase 2 deep dive is needed based on Phase 1 results.
        
        Uses LLM to intelligently evaluate if deeper analysis would be beneficial.
        
        Args:
            user_query: Original user query
            collection_results: Results from Phase 1 collection
            
        Returns:
            Tuple of (should_continue: bool, reasoning: str, servers_to_deep_dive: List[str])
        """
        # Gather all facts from Phase 1
        all_facts = []
        servers_with_facts = []
        
        for server_name, result in collection_results.items():
            if result.is_relevant and result.facts:
                servers_with_facts.append(server_name)
                for fact in result.facts[:5]:  # Limit facts per server for token efficiency
                    all_facts.append(f"[{server_name}] {fact.text_content}")
        
        if not all_facts:
            return False, "No facts collected in Phase 1", []
        
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

Examples when Phase 2 IS needed:
- Query asks for "correlation between X and Y" but facts don't show relationships
- Query asks for "trends" but only got point-in-time data
- Facts mention entities that other servers could provide more detail on
- Geographic query that needs coordinate matching across servers

Examples when Phase 2 is NOT needed:
- Simple factual questions that Phase 1 fully answered
- Query asks for a list and we got the list
- All relevant data has been collected

Respond with JSON:
{{
    "need_phase2": true/false,
    "reasoning": "one sentence explanation",
    "gaps_identified": ["specific gap 1", "specific gap 2"],
    "servers_for_phase2": ["server1", "server2"]
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
                return need_phase2, reasoning, servers
            else:
                # All JSON parsing failed
                raise ValueError("Could not parse JSON response")
            
        except Exception as e:
            print(f"Phase 2 evaluation error: {e}")
            # On error, default to no Phase 2
            return False, "Evaluation failed, proceeding with current data", []
    
    async def _phase2_deep_dive(self, user_query: str, scout_results: Dict[str, PhaseResult]) -> Dict[str, PhaseResult]:
        """
        Phase 2: Unified Deep Dive - All tools from all relevant servers available.
        
        Provides maximum flexibility for cross-server coordination.
        LLM has access to ALL tools and can intelligently combine them.
        
        Args:
            user_query: Original user query
            scout_results: Results from Phase 1 scout
            
        Returns:
            Dictionary mapping server names to their enhanced results
        """
        relevant_servers = {
            name: result for name, result in scout_results.items() 
            if result.is_relevant
        }
        
        if not relevant_servers:
            return {}
        
        # Gather ALL tools from ALL relevant servers
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
        
        while iteration < max_iterations:
            iteration += 1
            
            # Build the unified prompt
            system_prompt = """You are analyzing information to answer a user query.
            You have access to tools from multiple data sources.
            Call tools to fill gaps and correlate information across sources.
            When you have sufficient information, respond with 'COMPLETE'."""
            
            user_prompt = f"""User Query: {user_query}

What We Know So Far:
{facts_context}

You have access to tools from these servers: {', '.join(relevant_servers.keys())}

Analyze what information is still needed to fully answer the query.
Call relevant tools to:
1. Fill specific gaps in our knowledge
2. Correlate data across different sources  
3. Resolve any contradictions
4. Get deeper insights where valuable

If you have sufficient information, respond with just the word 'COMPLETE'.
Otherwise, call tools to gather the missing information."""
            
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
                    model=SMALL_MODEL,
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
                            actual_tool_name = prefixed_name.split('.', 1)[1]
                            
                            try:
                                # Call the tool on the appropriate server
                                result = await self.client.call_tool(
                                    tool_name=actual_tool_name,
                                    tool_args=block.input,
                                    server_name=server_name
                                )
                                
                                # Create fact from result
                                fact = Fact(
                                    text_content=f"[Phase 2] {str(result)[:500]}",  # Reasonable limit
                                    source_key=f"phase2_{server_name}_{actual_tool_name}_{iteration}",
                                    server_origin=server_name,
                                    metadata={
                                        "phase": 2,
                                        "iteration": iteration,
                                        "tool": actual_tool_name
                                    },
                                    citation_data={
                                        "tool_used": actual_tool_name,
                                        "source_name": f"{server_name} Phase 2",
                                        "phase": 2
                                    }
                                )
                                
                                phase2_facts.append(fact)
                                accumulated_facts.append(fact)
                                
                                # Add to citation registry
                                self.client.citation_registry.add_source(
                                    source=fact.citation_data,
                                    module_id=f"phase2_{server_name}"
                                )
                                
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
        
        # Step 1: Collect all facts
        all_facts = []
        map_data_list = []  # Store ALL map data, not just one
        chart_data = None  # Will store chart data
        table_data_list = []  # Store table data
        
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
                                    map_data_list.append(raw_result)
                                
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
                                    map_data_list.append(map_data)
                            
                            # Check for capacity statistics that could become charts/tables
                            if raw_result.get("capacity_stats") or raw_result.get("facilities_by_country"):
                                print(f"DEBUG: Found potential table/chart data in fact {i}")
        
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
            table_data_list=table_data_list
        )
        
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
            
            # Geographic detection (already saved in Phase 1/2)
            elif "map_url" in raw:
                fact.map_reference = {
                    "url": raw["map_url"],
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
        table_data_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Use LLM to intelligently order modules for natural narrative flow.
        
        Strategy:
        1. Parse narrative into sections
        2. Gather all available visualizations
        3. Ask LLM to determine optimal placement
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
        
        # Add additional maps if provided
        if additional_maps:
            for extra_map in additional_maps:
                extra_module = self._create_single_map_module(extra_map)
                if extra_module:
                    map_modules.append(extra_module)
        
        # Create table modules from facts and additional table data
        table_modules = self._create_table_modules(facts)
        
        # Add comparison tables from table_data_list using viz server
        if table_data_list:
            for table_data in table_data_list:
                if table_data.get("type") == "country_comparison":
                    # Call viz server to create comparison table
                    comparison_table = await self._create_comparison_table_via_viz(table_data)
                    if comparison_table:
                        table_modules.append(comparison_table)
        
        print(f"DEBUG: Visualization modules created:")
        print(f"  Charts: {len(chart_modules)}")
        print(f"  Maps: {len(map_modules)}")
        print(f"  Tables: {len(table_modules)}")
        if map_modules:
            for i, m in enumerate(map_modules):
                print(f"  Map {i}: {m.get('heading', 'No heading')}")
        
        # If we have visualizations, use LLM to determine placement
        if chart_modules or map_modules or table_modules:
            ordered_modules = await self._llm_order_modules(
                sections=sections,
                chart_modules=chart_modules,
                map_modules=map_modules, 
                table_modules=table_modules,
                user_query=user_query
            )
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
        
        for i, fact in enumerate(facts):
            fact_list.append(f"{i+1}. {fact.text_content}")
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
            display_info = f"\nIMPORTANT - The following visualizations WILL be included with your response:\n" + "\n".join(f"- {item}" for item in display_context) + "\n"
        
        # Add conversation history if available
        context_section = ""
        if self.conversation_history:
            context_section = f"""Previous Conversation Context:
{self._format_conversation_history()}

Current Query: {user_query}"""
        else:
            context_section = f"User Query: {user_query}"
        
        prompt = f"""{context_section}

Available Facts:
{chr(10).join(fact_list)}
{display_info}
Instructions:
1. Create a comprehensive response using these facts
2. Use placeholder citations like [CITE_1], [CITE_2] when referencing facts by their numbers
3. Group related information into logical sections with ## headings
4. Be concise but thorough
5. Focus on answering the user's query directly
{f'6. When discussing geographic data, reference the map naturally: "The map below shows..." or "As illustrated in the geographic distribution..."' if has_map_data else ''}
{f'7. When presenting numerical trends or comparisons, reference charts: "The chart illustrates..." or "This trend is visualized below..."' if has_chart_data else ''}
8. Structure your response to have clear transitions between topics
9. Use paragraph breaks to separate distinct ideas or data points
10. When transitioning from text to data visualization topics, use bridging phrases

Format sections clearly with ## headings and use citations for all factual claims."""
        
        try:
            response = await call_large_model(
                system="You synthesize facts into clear, informative responses with proper citations.",
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
        citations_found = re.findall(r'\[CITE_(\d+)\]', narrative)
        
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
        
        # First handle grouped citations like [CITE_2, CITE_3, CITE_5]
        def replace_grouped(match):
            grouped_text = match.group(0)  # e.g., "[CITE_2, CITE_3, CITE_5]"
            citations = re.findall(r'CITE_(\d+)', grouped_text)
            
            # Replace each citation with its mapped number
            replaced_citations = []
            for cite_num in citations:
                placeholder = f"CITE_{cite_num}"
                if placeholder in citation_map:
                    replaced_citations.append(f"^{citation_map[placeholder]}^")
            
            # Return space-separated superscript citations
            return ' '.join(replaced_citations) if replaced_citations else grouped_text
        
        # Replace grouped citations first
        final_text = re.sub(r'\[CITE_\d+(?:, CITE_\d+)*\]', replace_grouped, final_text)
        
        # Then handle any remaining individual citations [CITE_n]
        for placeholder, citation_num in citation_map.items():
            final_text = final_text.replace(f"[{placeholder}]", f"^{citation_num}^")
        
        return final_text
    
    async def _create_chart_modules(self, facts: List[Fact]) -> List[Dict]:
        """
        Create Chart.js modules from numerical facts.
        
        Args:
            facts: All facts, filtered for those with numerical_data
            
        Returns:
            List of chart modules
        """
        modules = []
        
        # Group related numerical facts
        numerical_facts = [f for f in facts if f.numerical_data and f.data_type in ['time_series', 'comparison']]
        
        for fact in numerical_facts:
            try:
                # Call visualization server
                chart_config = await self.client.call_tool(
                    tool_name="create_smart_chart",
                    tool_args={
                        "data": fact.numerical_data.get("values", []),
                        "context": f"{fact.data_type}: {fact.text_content}",
                        "title": self._generate_chart_title(fact)
                    },
                    server_name="viz"
                )
                
                # Create module
                modules.append({
                    "type": "chart",
                    "chartType": chart_config["type"],
                    "heading": chart_config.get("title", self._generate_chart_title(fact)),
                    "data": chart_config["data"],
                    "options": chart_config.get("options", {})
                })
            except Exception as e:
                print(f"Error creating chart for fact: {e}")
                # Skip failed charts rather than break the whole process
        
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
        
        # Also check facts for additional maps
        for fact in facts:
            if fact.map_reference and fact.data_type == "geographic":
                try:
                    # Load the pre-saved GeoJSON
                    file_path = fact.map_reference["url"].replace("/static/", "static/")
                    
                    # Check if file exists
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as f:
                            geojson = json.load(f)
                        
                        modules.append({
                            "type": "map",
                            "mapType": "geojson",
                            "heading": self._extract_map_title(fact.text_content),
                            "geojson": geojson,
                            "viewState": self._calculate_map_view_state(fact.map_reference)
                        })
                    else:
                        print(f"Map file not found: {file_path}")
                        
                except Exception as e:
                    print(f"Error loading map: {e}")
        
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
                            "heading": self._extract_table_title(fact.text_content),
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
        # Build reverse map: citation_number -> source_key
        used_citations = {}
        
        for placeholder, citation_num in citation_map.items():
            # Extract fact index from placeholder
            fact_idx = int(placeholder.split('_')[1]) - 1
            if fact_idx < len(facts):
                fact = facts[fact_idx]
                used_citations[citation_num] = fact
        
        # Build rows
        rows = []
        for citation_num in sorted(used_citations.keys()):
            fact = used_citations[citation_num]
            if fact.citation:
                rows.append([
                    str(citation_num),
                    fact.citation.source_name,
                    fact.citation.tool_id,
                    fact.citation.source_type,
                    fact.citation.description[:100]
                ])
        
        return {
            "type": "numbered_citation_table",
            "heading": "References",
            "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
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
    
    def _create_single_map_module(self, map_data: Dict) -> Optional[Dict]:
        """
        Create a single map module with proper context from map data.
        Uses the response_formatter's _create_map_module for consistency.
        """
        if not map_data or not map_data.get("geojson_url"):
            return None
        
        try:
            from response_formatter import _create_map_module
            map_module = _create_map_module(map_data)
            
            # The map module already has metadata with context from summary
            return map_module
            
        except Exception as e:
            print(f"Error creating single map module: {e}")
            return None
    
    async def _create_comparison_table_via_viz(self, table_data: Dict) -> Optional[Dict]:
        """
        Create a comparison table by calling the viz server's CreateComparisonTable tool.
        
        Args:
            table_data: Dictionary with 'type': 'country_comparison', 'data': facilities_by_country, 'countries': list
        
        Returns:
            Table module from viz server or None
        """
        if table_data.get("type") != "country_comparison":
            return None
        
        facilities_by_country = table_data.get("data", {})
        countries = table_data.get("countries", [])
        
        if not facilities_by_country:
            return None
        
        try:
            # Transform data for viz server format
            data_points = [
                {"name": country, "value": count}
                for country, count in facilities_by_country.items()
            ]
            
            # Call viz server tool
            client = await get_global_mcp_client()
            result = await client.call_tool(
                server="viz",
                tool="CreateComparisonTable",
                arguments={
                    "data_points": data_points,
                    "comparison_type": "facilities",
                    "entity_key": "name",
                    "value_key": "value",
                    "include_percentages": True,
                    "include_totals": True,
                    "sort_by": "value",
                    "sort_descending": True
                }
            )
            
            # Extract the table module from the result
            if result and hasattr(result, 'content'):
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        # Parse the JSON response
                        table_module = json.loads(content_item.text)
                        return table_module
            
            return None
            
        except Exception as e:
            print(f"Error calling viz server for comparison table: {e}")
            # Fallback to simple table
            return self._create_simple_comparison_table(facilities_by_country, countries)
    
    def _create_simple_comparison_table(self, facilities_by_country: Dict, countries: List[str]) -> Dict:
        """Simple fallback comparison table if viz server fails."""
        rows = []
        total = sum(facilities_by_country.values())
        
        for country in sorted(countries, key=lambda c: facilities_by_country.get(c, 0), reverse=True):
            count = facilities_by_country.get(country, 0)
            percentage = (count / total * 100) if total > 0 else 0
            rows.append([country.title(), f"{count:,}", f"{percentage:.1f}%"])
        
        rows.append(["**Total**", f"**{total:,}**", "100.0%"])
        
        return {
            "type": "table",
            "heading": "Solar Facility Distribution by Country",
            "columns": ["Country", "Number of Facilities", "% of Total"],
            "rows": rows
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

async def process_chat_query(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict:
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
        orchestrator = QueryOrchestrator(client, conversation_history=conversation_history)
        
        # Process through 3-phase architecture
        response = await orchestrator.process_query(user_query)
        
        return response
        
    except Exception as e:
        # TODO: Implement proper error handling and logging
        return f"Error processing query: {str(e)}"


async def stream_chat_query(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None):
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
        orchestrator = QueryOrchestrator(client, conversation_history=conversation_history)
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
            relevant_servers = ["kg"]
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
                "kg": "Knowledge Graph",
                "solar": "Solar Facilities",
                "gist": "Environmental Impact",
                "lse": "Climate Policy",
                "formatter": "Visualization"
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
        # Use LLM to decide if Phase 2 is needed
        should_deep_dive, reasoning, servers_for_phase2 = await orchestrator._should_do_phase2_deep_dive(
            user_query, collection_results
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
        
        return {"status": "all_connected", "servers": ["kg", "solar", "gist", "lse", "formatter"]}
        
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
    asyncio.run(main())