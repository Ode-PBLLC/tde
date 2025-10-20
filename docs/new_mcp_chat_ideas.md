I think a new mcp_chat.py file should be made. Let's call it mcp_chat_redo.py

The script should have:
## Server Initialization (Global Singleton)
```python
# Global singleton client for performance optimization
_global_client = None
_client_lock = asyncio.Lock()

async def get_global_client():
    """Get or create the global singleton MCP client."""
    global _global_client
    
    async with _client_lock:
        if _global_client is None:
            _global_client = MultiServerClient()
            await _global_client.__aenter__()
            
            # Connect to all servers
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            mcp_dir = os.path.join(project_root, "mcp")
            
            try:
                await _global_client.connect_to_server("kg", os.path.join(mcp_dir, "cpr_kg_server.py"))
                await _global_client.connect_to_server("solar", os.path.join(mcp_dir, "solar_facilities_server.py"))
                await _global_client.connect_to_server("gist", os.path.join(mcp_dir, "gist_server.py"))
                await _global_client.connect_to_server("lse", os.path.join(mcp_dir, "lse_server.py"))
                await _global_client.connect_to_server("formatter", os.path.join(mcp_dir, "response_formatter_server.py"))
                print("Global MCP client initialized successfully")
            except Exception as e:
                print(f"Error initializing global MCP client: {e}")
                # Clean up on failure
                await _global_client.__aexit__(None, None, None)
                _global_client = None
                raise
                
    return _global_client

async def cleanup_global_client():
    """Clean up the global singleton client."""
    global _global_client
    
    async with _client_lock:
        if _global_client is not None:
            await _global_client.__aexit__(None, None, None)
            _global_client = None
            print("Global MCP client cleaned up")
```

**Why this pattern:**
- Thread-safe initialization with asyncio.Lock()
- Lazy loading (only connects when first needed)
- Proper error handling and cleanup
- Reusable across all API requests for 5-10x performance gain

## Core Infrastructure (from existing implementation)

### MultiServerClient Base Structure
```python
class MultiServerClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.anthropic = anthropic.Anthropic()
        self.citation_registry = CitationRegistry()
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.exit_stack.__aexit__(exc_type, exc, tb)
```

### Server Connection Method
```python
async def connect_to_server(self, server_name: str, server_script_path: str):
    is_python = server_script_path.endswith('.py')
    is_js = server_script_path.endswith('.js')
    
    if not (is_python or is_js):
        raise ValueError("Server script must be a .py or .js file")

    command = "python" if is_python else "node"
    server_params = StdioServerParameters(
        command=command,
        args=[server_script_path],
        env=None
    )

    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()
    
    self.sessions[server_name] = session
```

### Tool Calling Method
```python
async def call_tool(self, tool_name: str, tool_args: dict, server_name: str):
    """Call a tool on a specific server."""
    if server_name not in self.sessions:
        raise ValueError(f"Server '{server_name}' not connected")
    
    session = self.sessions[server_name]
    return await session.call_tool(tool_name, tool_args)
```

**Note**: The rest of the implementation (query processing, orchestration logic) will be rewritten from scratch to follow the new 3-phase architecture.

## Citation Registry
- citation registry: Essentially, the registry will be a collection of facts and citations to be later synthesized into our responses to the user. A possible implementation is down below.
- llm orchestration
   - user query
   - Query a "routing LLM" with "Which of the following servers are relevant for this user query:
      server_1 name and description
      server_2 name and description
      server_3 name and description..."
   - From the servers above, carry forward the descriptions of the tools and servers and ask query the LLM again with "Given these tools and servers"
     - **Hybrid Parallel-Sequential Execution Strategy**:

         Phase 1: Parallel Scout (Minimal Context)
         - Each server receives: user query + server-specific routing prompt
         - Token budget: ~500-1000 tokens per server
         - Returns:
           - is_relevant: boolean (LLM decides yes/no)
           - initial_facts: list of discovered facts (if relevant)
           - reasoning: brief explanation of relevance/irrelevance
           - suggested_follow_ups: queries for Phase 2 (if relevant)
         - All servers run in parallel via asyncio.gather()

        Phase 2: Targeted Deep Dive (Selective Context Sharing)
        - Only servers where is_relevant=True continue
        - Each relevant server receives:
            - Original query
            - Its own Phase 1 results
            - "Highlights" from other relevant servers (key facts only, <200 
        tokens)
        - Token budget: ~2000-3000 per server
        - **Termination**: Each server runs in a loop until the LLM returns:
            - continue: True/False (should I keep searching?)
            - reasoning: why to continue or stop
            - next_action: specific tool call to make (if continuing)
        - **Fact & Citation Collection**:
            - As each tool returns data, immediately create Fact objects
            - Call citation_registry.add_source() when fact is created
            - Facts and citations are collected in real-time during execution
            - Each server maintains its own fact list during Phase 2
        - **Cross-Server Callbacks** (lightweight):
            - Server can request: "ask_server: 'solar', query: 'coordinates of facility X'"
            - Orchestrator routes this as a simple query to the target server
            - Target server responds with just the answer (no tool specs needed)
            - Response injected into requesting server's context as a "message"
            - Example: "Server 'solar' responds: Facility X is at lat: -15.7, lon: -47.9"
        - **Error Handling**: 
            - If a server fails after is_relevant=True, log the error but continue
            - Treat failed servers as having no additional facts to contribute
            - Synthesis uses only successful server outputs
            - Empty results (relevant but no data) are valid - server contributes no facts
        - **Artifact Handling** (for multimodal responses):
            - Large outputs (GeoJSON, charts) saved to static/ directory
            - Citation registry stores:
                - artifact_type: 'geojson' | 'chart' | 'image'
                - artifact_url: path to saved file (internal use only, NOT in synthesized text)
                - summary: natural language description (e.g., "interactive map showing 2,273 solar facilities across Brazil")
                - metadata: key stats (count, total_capacity, bounds, etc.)
            - Context passed forward contains only summary + metadata, not full artifact data
            - Example fact stored: "Generated an interactive map showing 2,273 solar facilities in Brazil with 15.2GW total capacity"

         Phase 3: Synthesis
         - Citation registry contains all collected facts from relevant servers AND artifact references
         - Final synthesis LLM receives ONLY:
             - Text facts with natural language descriptions
             - Artifact summaries (no URLs or file paths)
             - Example input: "Analysis found 2,273 solar facilities in Brazil with 15.2GW capacity"
         - LLM generates clean narrative with placeholder references:
             - Example output: "Brazil has 2,273 solar facilities [see map] concentrated in São Paulo region"
         - **Citation Deduplication & Numbering**:
             - After facts are reordered narratively, citations are reordered to match
             - Deduplicate citations based on source_key (same source = same citation number)
             - Assign citation numbers based on first appearance in the narrative
             - Example flow:
                 - Fact 1: "Brazil leads in solar" [source A]
                 - Fact 2: "São Paulo has most facilities" [source B] 
                 - Fact 3: "Capacity grew 40% in 2024" [source A]
                 - Result: "Brazil leads in solar¹. São Paulo has most facilities². Capacity grew 40% in 2024¹."
             - Citation registry maintains mapping: source_key -> final_citation_number
         - Response formatter (POST-synthesis):
             - Detects artifact placeholders like "[see map]" or "[view chart]"
             - Enriches with actual URLs/embeds from citation registry
             - Injects Chart.js configs
             - Maintains clean separation between content and presentation
         - Can make targeted callbacks to specific servers for clarification

         Token Control:
         - Track token_budget_remaining across phases
         - Dynamically adjust Phase 2 depth based on Phase 1 consumption
         - Emergency brake if approaching rate limits (see CLAUDE.md)
         - For now: Keep budgets high and let models work naturally

   - Surface a context with a while loop for each that goes until termination. Make sure all tool calls that return information are captured with the citation registry

## Fact Structure Definition
```python
class Fact:
    text_content: str  # The actual fact text (natural language)
    source_key: str  # For citation deduplication
    server_origin: str  # Which server provided this fact
    metadata: dict  # Additional context including:
                    # - artifact_urls (geojson, charts, etc.)
                    # - artifact_type ('geojson', 'chart', 'image')
                    # - statistical summaries
                    # - timestamp
    citation_data: dict  # Original source information for citations
```

Note: text_content is explicitly for narrative text. Large data artifacts (maps, charts) are stored separately and referenced via metadata.

POSSIBLE CitationRegistry implementation
class CitationRegistry:
    """
    Manages citation numbering and tracking for inline citations.
    
    Assigns unique citation numbers to sources and tracks which modules use which citations.
    """
    
    def __init__(self):
        self.citations = {}  # source_key -> citation_number
        self.citation_counter = 1
        self.module_citations = {}  # module_id -> list of citation numbers
        self.citation_details = {}  # citation_number -> full source dict
        
    def add_source(self, source: Dict[str, Any], module_id: str = None) -> int:
        """
        Add a source and return its citation number.
        
        Args:
            source: Source dictionary (passage or dataset)
            module_id: Optional module identifier for tracking
            
        Returns:
            Citation number for this source
        """
        # Create unique key for this source
        source_key = self._generate_source_key(source)
        
        # Check if we already have this source
        if source_key in self.citations:
            citation_num = self.citations[source_key]
        else:
            # Assign new citation number
            citation_num = self.citation_counter
            self.citations[source_key] = citation_num
            self.citation_details[citation_num] = source
            self.citation_counter += 1
        
        # Track module association
        if module_id:
            if module_id not in self.module_citations:
                self.module_citations[module_id] = []
            if citation_num not in self.module_citations[module_id]:
                self.module_citations[module_id].append(citation_num)
        
        return citation_num
    
    def _generate_source_key(self, source: Dict[str, Any]) -> str:
        """Generate unique key for source deduplication."""
        if isinstance(source, dict):
            # For passage sources
            if "doc_id" in source and "passage_id" in source:
                return f"passage_{source['doc_id']}_{source['passage_id']}"
            # For dataset sources
            elif "citation_id" in source:
                return f"dataset_{source['citation_id']}"
            # For tool-based sources
            elif "tool_used" in source and "source_name" in source:
                return f"tool_{source['tool_used']}_{source['source_name']}"
        
        # Fallback for other source types
        return f"generic_{hash(str(source))}"
    
    def get_module_citations(self, module_id: str) -> List[int]:
        """Get all citation numbers used by a specific module."""
        return self.module_citations.get(module_id, [])
    
    def get_citation_details(self, citation_num: int) -> Dict[str, Any]:
        """Get full source details for a citation number."""
        return self.citation_details.get(citation_num, {})
    
    def get_all_citations(self) -> Dict[int, Dict[str, Any]]:
        """Get all citations ordered by number."""
        return {num: self.citation_details[num] for num in sorted(self.citation_details.keys())}
    
    def format_citation_superscript(self, citation_nums: List[int]) -> str:
        """Format citation numbers as superscript text."""
        if not citation_nums:
            return ""
        if len(citation_nums) == 1:
            return f"^{citation_nums[0]}^"
        else:
            return f"^{','.join(map(str, sorted(citation_nums)))}^"