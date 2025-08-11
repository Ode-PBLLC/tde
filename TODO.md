# TODO.md - Feature Ideas and Improvements

## Response Formatter Enhancements

### Fix Inline Citations
**Date Added**: January 31, 2025  
**Priority**: High  
**Component**: `mcp/response_formatter_server.py`

**Current Issues**:
- Citations sometimes appear at the end of paragraphs rather than immediately after the relevant fact
- LLM-based citation placement can be inconsistent
- Some citations may be missed or duplicated
- Citation numbers don't always match the actual sources

**Desired Behavior**:
- Citations appear immediately after the specific fact they support
- Example: "Brazil has 2,273 facilities^1^ with total capacity of 26,022 MW^2^"
- All citations from the registry are used and properly distributed
- Consistent and reliable citation placement

**Implementation Ideas**:
1. Improve the LLM prompt for citation placement with more specific examples
2. Add validation to ensure all citations are used
3. Implement a more robust fallback mechanism
4. Consider rule-based citation placement for specific data patterns
5. Add citation preview/debugging tools

### Implement True Content Interweaving
**Date Added**: January 31, 2025  
**Priority**: Medium  
**Component**: `mcp/response_formatter_server.py`

**Current Behavior**: 
- All text content is bundled into a single text module at the beginning
- Charts, maps, and tables follow as separate discrete modules
- Limited narrative flow between different content types

**Desired Behavior**:
- Enable true interweaving of text and visual elements
- Support patterns like: text → chart → text → map → text
- Create more engaging, magazine-style reports

**Example Use Case**:
```
1. "Brazil has emerged as a leader in solar capacity..."
2. [Bar chart: Brazil's solar capacity growth 2020-2025]
3. "This growth is geographically concentrated in key regions..."
4. [Interactive map: Solar facilities across Brazilian states]
5. "Policy incentives have been crucial to this expansion..."
```

**Implementation Ideas**:
1. Modify text processing to split content into semantic sections
2. Create associations between text sections and relevant visualizations
3. Develop new "composite" module types that combine text + visual elements
4. Enhance `OrganizeModulesIntoNarrative` to work with finer-grained content blocks
5. Maintain citation integrity across interwoven content

**Benefits**:
- More engaging and readable reports
- Better contextual placement of visualizations
- Improved user comprehension through visual breaks
- Professional, publication-quality output

### Synchronize KG Response and Visualization Concepts
**Date Added**: January 31, 2025  
**Priority**: High  
**Component**: `mcp/kg_query_server.py`, `mcp/kg_visualization_server.py`

**Current Issues**:
- Knowledge Graph text responses and visualizations may use different concepts/entities
- Visualization might show different relationships than what's described in text
- No guarantee that the graph visualization matches the narrative response
- Concepts mentioned in text might not appear in the visual graph

**Desired Behavior**:
- KG visualization should directly reflect the concepts discussed in the text response
- Entities and relationships mentioned in text should be visible in the graph
- Graph should highlight the specific subgraph relevant to the query
- Consistent concept selection between text and visual representations

**Example**:
- Text: "Brazil's REDD+ policy connects to deforestation monitoring through..."
- Graph: Should show Brazil, REDD+ policy node, deforestation monitoring node, and their connections

**Implementation Ideas**:
1. Pass concept list from text response to visualization generator
2. Implement concept extraction from KG query results
3. Create a "focused subgraph" mode that shows only mentioned entities
4. Add concept highlighting in visualizations
5. Develop shared concept registry between text and visualization components
6. Consider two-pass approach: extract concepts first, then generate both text and visual

**Benefits**:
- Better alignment between narrative and visual elements
- Reduced cognitive load for users
- More coherent and integrated responses
- Improved trust in the system's outputs

### Add LLM Provider Flexibility and Fallback Support
**Date Added**: January 31, 2025  
**Priority**: High  
**Component**: Multiple files using Anthropic API

**Current Issues**:
- System is tightly coupled to Anthropic's Claude API
- No fallback when Claude API is down or rate limited
- Single point of failure for LLM-dependent features
- Hard-coded model selection throughout codebase

**Desired Behavior**:
- Support multiple LLM providers (OpenAI, Google, Mistral, local models)
- Automatic fallback to secondary providers when primary is unavailable
- Configurable model selection via environment variables
- Graceful degradation when all LLM services are down

**Affected Components**:
- `mcp/response_formatter_server.py` - Citation placement
- `mcp/kg_query_server.py` - Query interpretation
- `mcp/mcp_orchestrator.py` - Response generation
- Any other components using Anthropic directly

**Implementation Ideas**:
1. Create an LLM abstraction layer/interface
2. Implement provider adapters (OpenAI, Google Vertex, AWS Bedrock, etc.)
3. Add retry logic with exponential backoff
4. Implement circuit breaker pattern for failed providers
5. Create provider priority queue (primary → secondary → tertiary)
6. Add health checks for LLM endpoints
7. Implement prompt adaptation for different models
8. Add cost tracking per provider

**Example Configuration**:
```yaml
llm_providers:
  primary: 
    provider: "anthropic"
    model: "claude-3-5-sonnet-20241022"
  fallback:
    - provider: "openai"
      model: "gpt-4-turbo"
    - provider: "google"
      model: "gemini-1.5-pro"
```

**Benefits**:
- Improved system reliability and uptime
- Cost optimization through provider selection
- Flexibility to use best model for each task
- Reduced vendor lock-in
- Better handling of rate limits and outages

### Fix Critical Concurrency Issue in Global MCP Client
**Date Added**: January 31, 2025  
**Priority**: CRITICAL  
**Component**: `mcp/mcp_chat.py`, `api_server.py`

**Current Issues**:
- Global `CitationRegistry` instance shared across ALL concurrent requests
- Citation numbers mix between different users' queries
- Race conditions corrupt citation tracking and module mappings
- State contamination causes data leakage between users
- Security/privacy risk in multi-user environment

**Failure Scenario**:
```python
# Request 1: "Brazil climate policy" → adds citations 1,2,3
# Request 2: "India solar data" (concurrent) → adds citations 4,5,6 to SAME registry
# Result: Both responses contain mixed citations from both queries
```

**Root Cause**:
- `MultiServerClient` has single `self.citation_registry = CitationRegistry()`
- Global singleton pattern shares this mutable state across all requests
- AsyncIO cooperative multitasking allows requests to interleave

**Desired Behavior**:
- Each request should have isolated citation tracking
- No shared mutable state between concurrent requests
- Connection pooling benefits retained without state contamination

**Implementation Options**:
1. **Per-Request Citation Registry** (Recommended)
   - Pass citation_registry as parameter through call chain
   - Create new registry for each request in `run_query_*` functions
   - Keep global client for connection pooling only

2. **Request Context Pattern**
   - Use contextvars to maintain per-request state
   - AsyncIO-aware context isolation
   - Clean separation of shared vs. request-specific data

3. **Client Factory Pattern**
   - Global connection pool, per-request client instances
   - Each client gets fresh citation registry
   - More complex but very clean separation

**Example Fix (Option 1)**:
```python
async def run_query_streaming(query: str):
    client = await get_global_client()
    citation_registry = CitationRegistry()  # Per-request instance
    
    # Pass citation_registry through the call chain
    async for event in client.process_query_streaming(query, citation_registry):
        yield event
```

**Testing Requirements**:
- Concurrent request test with different queries
- Verify citation isolation between requests
- Load test with 10+ simultaneous requests
- Check for any other shared mutable state

**Benefits**:
- Eliminates data leakage between users
- Correct citation tracking per request
- Maintains performance benefits of connection pooling
- Production-ready multi-user support

### Implement Brazilian Portuguese Response Translation
**Date Added**: August 1, 2025  
**Priority**: Medium  
**Component**: Response formatter and final output stage

**Current Behavior**:
- All responses are delivered in English regardless of query language
- No language detection for incoming queries
- Processing pipeline operates entirely in English

**Desired Behavior**:
- Detect when queries are submitted in Brazilian Portuguese
- Process queries normally in English (knowledge graph, data retrieval, analysis)
- Translate final formatted response to Brazilian Portuguese as the last step
- Maintain proper Portuguese grammar, technical terminology, and cultural context

**Implementation Ideas**:
1. Add language detection to initial query processing
2. Store detected language preference with query context
3. Keep all MCP processing (KG queries, data analysis, formatting) in English
4. Add translation step in response formatter before final output
5. Use LLM for high-quality translation with climate/energy domain knowledge
6. Preserve citations, formatting, and technical accuracy during translation

**Example Flow**:
- Query: "Analise a capacidade solar do Brasil"
- Detection: Portuguese detected
- Processing: All MCP tools work in English
- Final step: Translate complete English response to Portuguese

**Benefits**:
- Minimal changes to existing processing pipeline
- Leverages existing English-optimized knowledge base
- High-quality final output in user's preferred language
- Maintains technical accuracy through domain-aware translation

### Implement Multi-Turn Conversation Support
**Date Added**: August 1, 2025  
**Priority**: High  
**Component**: API server, MCP orchestrator, and conversation management

**Current Behavior**:
- Each query is processed independently without context
- No memory of previous questions or responses
- Users cannot build on previous queries or ask follow-up questions
- No session management or conversation continuity

**Desired Behavior**:
- Support multi-turn conversations with context awareness
- Remember previous queries and responses within a session
- Enable follow-up questions like "What about India?" or "Show me more details on that policy"
- Maintain conversation state across multiple API calls
- Allow users to refine and build upon previous analyses

**Implementation Ideas**:
1. Add session management with unique conversation IDs
2. Store conversation history (queries + responses) in memory or database
3. Include relevant conversation context in MCP tool queries
4. Implement context window management to handle long conversations
5. Add conversation endpoints: start, continue, end session
6. Create conversation memory that includes:
   - Previous queries and their results
   - Generated visualizations and their contexts
   - Key entities and concepts discussed
   - Current conversation focus/topic

**Example Conversation Flow**:
```
User: "Show me Brazil's solar capacity"
System: [Provides detailed analysis with charts and maps]

User: "How does this compare to India?"
System: [Uses context from previous Brazil analysis to make comparison]

User: "What policies drove this growth in both countries?"
System: [References both Brazil and India context from conversation history]
```

**Technical Considerations**:
- Session storage (Redis, in-memory, or database)
- Context summarization for long conversations
- Privacy and data retention policies
- Session timeout and cleanup
- Conversation context injection into MCP queries

**API Changes**:
- Add session_id parameter to query endpoints
- New endpoints: `/session/start`, `/session/end`
- Conversation history retrieval endpoint
- Session management in streaming responses

**Benefits**:
- More natural and engaging user experience
- Enables deeper exploration of topics
- Reduces need to repeat context in follow-up questions
- Better support for complex analytical workflows
- Competitive feature for conversational AI platforms

### Test GeoJSON Mixed Geometry Rendering
**Date Added**: August 1, 2025  
**Priority**: Medium  
**Component**: Map generation and frontend rendering

**Current Behavior**:
- Current GeoJSON outputs contain only points (solar facilities)
- Map rendering has only been tested with single geometry type
- Unknown if frontend can handle mixed geometry types in same GeoJSON

**Desired Behavior**:
- Test rendering of GeoJSON files containing both points and polygons
- Verify that maps can display mixed geometry types simultaneously
- Ensure proper styling and interaction for different geometry types

**Test Scenarios**:
1. Solar facilities (points) + administrative boundaries (polygons)
2. Solar facilities (points) + environmental zones (polygons)
3. Energy infrastructure (points) + protected areas (polygons)
4. Multiple geometry types with different styling requirements

**Implementation Ideas**:
1. Create test GeoJSON files with mixed geometry types
2. Test with existing map rendering system
3. Verify frontend JavaScript handles FeatureCollection with mixed types
4. Document any limitations or required styling changes
5. Update map generation logic if needed to support mixed types

**Example Use Cases**:
- "Show solar facilities in Brazil with state boundaries"
- "Display mining sites within protected environmental zones"
- "Map energy infrastructure overlaid on watershed boundaries"

**Technical Considerations**:
- GeoJSON FeatureCollection with mixed geometry types
- Frontend map library compatibility (Leaflet, MapBox, etc.)
- Styling differentiation between points and polygons
- Performance impact of complex mixed geometry rendering
- Legend and interaction behavior for mixed types

**Benefits**:
- More comprehensive and informative maps
- Better context for spatial analysis
- Enhanced visual storytelling capabilities
- Support for complex geographical relationships

### Add Complete Knowledge Graph File
**Date Added**: August 1, 2025  
**Priority**: Medium  
**Component**: Knowledge graph data files

**Current Status**:
- Knowledge graph may be incomplete or missing critical connections
- Need to verify completeness and accuracy of current KG data

**Action Required**:
- Review and add the complete/correct knowledge graph file to ensure comprehensive data coverage

### Integrate Governance Data from Alex
**Date Added**: August 1, 2025  
**Priority**: Medium  
**Component**: Data integration and MCP servers

**Action Required**:
- Obtain governance data from Alex
- Integrate governance data into existing MCP server infrastructure
- Update knowledge graph and query capabilities to include governance information

### Develop Stress Testing Framework for Response Times and Costs
**Date Added**: August 5, 2025  
**Priority**: High  
**Component**: Testing infrastructure and performance analysis

**Current Behavior**:
- No systematic way to measure system performance under load
- Response times and API costs not tracked during burst traffic
- No benchmarking framework for stress testing

**Desired Behavior**:
- Develop comprehensive stress testing framework
- Benchmark response times under various load conditions
- Track API costs during burst queries
- Identify performance bottlenecks and scalability limits

**Implementation Ideas**:
1. Create test suite that generates burst queries with configurable:
   - Concurrent request counts (10, 50, 100+ simultaneous queries)
   - Query complexity levels (simple vs. complex multi-tool queries)
   - Duration and frequency patterns
2. Implement metrics collection for:
   - Time to first token (TTFT)
   - Total response time
   - API token usage and costs per provider
   - Error rates and failure modes
   - Memory usage and system resources
3. Add performance regression detection
4. Create load testing scenarios that simulate real user behavior
5. Generate performance reports with visualizations

**Testing Scenarios**:
- Sustained load: 20 concurrent users for 10 minutes
- Burst traffic: 100 simultaneous queries
- Complex query stress: KG + visualization + formatting pipeline
- Rate limit testing: Push API limits to measure graceful degradation
- Mixed query types: Combine simple and complex queries

**Deliverables**:
- Stress testing script in test_scripts directory
- Performance benchmarking dashboard
- Cost analysis reporting
- System scalability recommendations
- Performance regression test suite

**Benefits**:
- Identify system bottlenecks before production issues
- Optimize API costs through better understanding of usage patterns
- Validate system reliability under load
- Data-driven capacity planning and scaling decisions