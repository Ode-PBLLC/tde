# Climate Policy API System Architecture

## Overview

The Climate Policy API is an intelligent system that combines knowledge graphs, structured datasets, and AI reasoning to provide comprehensive responses about climate policy topics. It automatically discovers and surfaces relevant data, policy documents, and visualizations based on user queries.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                          │
│                      (api_server.py)                           │
├─────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                     │
│  • POST /query              - Structured responses             │
│  • POST /thorough-response  - Raw MCP data (debugging)         │
│  • GET  /health             - Health check                     │
│  • GET  /example-response   - Sample response format           │
│  • Static files at /static  - Generated maps/charts            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────────┐
│                   MCP Orchestration Layer                      │
│                     (mcp_chat.py)                              │
├─────────────────────────────────────────────────────────────────┤
│  • Multi-server client management                              │
│  • AI reasoning with Claude Sonnet 4                           │
│  • Automatic dataset discovery                                 │
│  • Response formatting and structuring                         │
└─────────┬───────────┬───────────┬───────────────────────────────┘
          │           │           │
          v           v           v
┌─────────────┐ ┌─────────────┐ ┌─────────────────────────────────┐
│   Knowledge │ │    Solar    │ │      Response Formatter        │
│    Graph    │ │ Facilities  │ │         Server                 │
│   Server    │ │   Server    │ │   (response_formatter_server.py)│
│             │ │             │ │                                │
│ • Concepts  │ │ • Geographic│ │ • Structures raw data          │
│ • Passages  │ │   data      │ │ • Creates visualizations       │
│ • Relations │ │ • Map gen   │ │ • Formats for frontend         │
└─────────────┘ └─────────────┘ └─────────────────────────────────┘
```

## Core Components

### 1. FastAPI Application (`api_server.py`)

**Purpose**: HTTP API layer that provides structured responses for frontend consumption.

**Key Features**:
- CORS-enabled for cross-origin requests
- Static file serving for generated maps/charts
- Multiple response formats (structured vs. raw)
- Error handling with detailed tracebacks

**Endpoints**:
- `POST /query` - Main endpoint returning structured JSON modules
- `POST /thorough-response` - Debug endpoint with complete raw MCP data
- `GET /health` - System health check
- `GET /example-response` - Sample response format for developers

### 2. MCP Orchestration (`mcp_chat.py`)

**Purpose**: Central coordinator that manages multiple MCP servers and AI reasoning.

**Key Features**:
- **Multi-server management**: Connects to multiple specialized MCP servers
- **AI reasoning**: Uses Claude Sonnet 4 for query processing and synthesis
- **Automatic dataset discovery**: Enhanced system prompt automatically calls `GetAvailableDatasets()`
- **Response structuring**: Converts raw data into frontend-ready modules

**AI Behavior**:
```python
# Enhanced workflow for ANY query:
1. GetPassagesMentioningConcept(concept)     # Knowledge graph search
2. GetAvailableDatasets()                    # Automatic dataset discovery  
3. GetDatasetContent(dataset_id)             # If datasets found
4. [Additional tools based on query type]    # Solar, formatting, etc.
5. Synthesize comprehensive response         # Text + Data + Visualizations
```

### 3. Knowledge Graph Server (`cpr_kg_server.py`)

**Purpose**: Provides access to climate policy concepts, passages, and dataset relationships.

**Data Sources**:
- `extras/concepts.csv` - 1,325 climate concepts with embeddings
- `extras/labelled_passages.jsonl` - 6,813 labeled text passages
- `aggregated/` - 200+ policy documents (CCLW, UNFCCC, CPR, GCF)

**Key Tools**:
- `GetPassagesMentioningConcept` - Semantic search through policy documents
- `GetAvailableDatasets` - Discovery of datasets linked to concepts
- `GetDatasetContent` - Retrieval of structured dataset content

**Concept-Dataset Linking**:
```python
# Example relationships in the knowledge graph:
"extreme weather" ──HAS_DATASET_ABOUT──> "DUMMY_DATASET_EXTREME_WEATHER"
"solar energy" ────HAS_DATASET_ABOUT──> "SOLAR_FACILITIES_DATASET"
"renewable energy" ─HAS_DATASET_ABOUT──> "SOLAR_FACILITIES_DATASET"
```

### 4. Solar Facilities Server (`solar_facilities_server.py`)

**Purpose**: Provides geographic data and visualizations for solar energy infrastructure.

**Data Source**:
- `data/tz-sam-q1-2025/solar_facilities_demo.csv` - 8,319 solar facilities
- Coverage: Brazil, India, South Africa, Vietnam
- 124.9 GW total capacity

**Key Tools**:
- `GetSolarFacilitiesByCountry` - Filter facilities by country
- `GetSolarFacilitiesMapData` - Generate GeoJSON for mapping
- `GetLargestSolarFacilities` - Find biggest installations
- `GetSolarCapacityVisualizationData` - Chart data generation

### 5. Response Formatter Server (`response_formatter_server.py`)

**Purpose**: Structures raw data into frontend-ready modules with consistent formatting.

**Module Types**:
- `text` - Policy summaries and analysis
- `map` - Interactive maps with GeoJSON data
- `chart` - Data visualizations (bar, line, pie charts)
- `table` - Structured data tables

## Data Flow

### Standard Query Flow

```
1. User Query: "extreme weather in Brazil"
   ↓
2. MCP Orchestration:
   • GetPassagesMentioningConcept("extreme weather")
   • GetAvailableDatasets() → Discovers extreme weather dataset
   • GetDatasetContent("DUMMY_DATASET_EXTREME_WEATHER")
   • GetSolarFacilitiesByCountry("Brazil") [if solar-related]
   ↓
3. AI Synthesis:
   • Combines policy text + structured data + geographic data
   • Generates comprehensive response
   ↓
4. Response Formatting:
   • CreateStructuredResponse() formats into modules
   • Maps, charts, and tables generated as needed
   ↓
5. Frontend Response:
   {
     "modules": [
       {"type": "text", "heading": "Analysis", "texts": [...]},
       {"type": "table", "heading": "Data", "rows": [...]},
       {"type": "map", "geojson_url": "/static/maps/..."}
     ],
     "metadata": {"has_maps": true, "has_charts": false}
   }
```

### Thorough Response Flow

The `/thorough-response` endpoint provides complete raw MCP data for debugging:

```json
{
  "query": "extreme weather",
  "raw_mcp_response": {
    "response": "AI's final summary...",
    "sources": ["Document sources"],
    "chart_data": [...],
    "map_data": [...],
    "all_tool_outputs_for_debug": [
      {"tool_name": "GetPassagesMentioningConcept", "tool_result": "..."},
      {"tool_name": "GetAvailableDatasets", "tool_result": "..."}
    ],
    "ai_thought_process": "Step-by-step reasoning..."
  }
}
```

## Key Features

### 1. Automatic Dataset Discovery

**Innovation**: System automatically discovers and surfaces datasets connected to any concept, not just when explicitly requested.

**Implementation**: Enhanced system prompt mandates `GetAvailableDatasets()` call for every query.

**Result**: Users get comprehensive responses combining policy analysis + structured data without asking.

### 2. Multi-Modal Responses

**Text Modules**: Policy summaries, analysis, explanations
**Data Tables**: Structured datasets (extreme weather events, facility data)
**Interactive Maps**: GeoJSON-based maps with facility locations
**Charts**: Data visualizations (capacity distributions, comparisons)

### 3. Cross-Referenced Intelligence

**Knowledge Graph**: Semantic search through policy documents
**Geographic Data**: Real-world infrastructure locations
**Structured Datasets**: Quantitative data for analysis
**AI Reasoning**: Synthesis of all sources into coherent responses

### 4. Developer-Friendly API

**Structured Responses**: Consistent module-based format
**Raw Data Access**: Complete MCP pipeline visibility
**Static File Serving**: Generated maps and charts
**CORS Support**: Frontend integration ready
**Error Handling**: Detailed debugging information

## Deployment Architecture

```
Production Environment:
├── EC2 Instance (Ubuntu)
├── Conda Environment (tde-api)
├── FastAPI Server (Port 8099)
├── Static File Serving (/static/maps/)
├── Security Group (Ports 22, 8099)
└── Git-based Deployment
```

**Key Configuration**:
- Environment variables in `.env` file
- Anthropic API key required
- OpenAI API key for embeddings
- Cross-platform path handling
- Automatic working directory detection

## Performance Characteristics

**Query Response Times**:
- Simple concept queries: ~3-5 seconds
- Complex multi-dataset queries: ~10-15 seconds
- Geographic queries with maps: ~15-20 seconds

**Data Volumes**:
- Knowledge Graph: 1,325 concepts, 6,813 passages
- Solar Dataset: 8,319 facilities across 4 countries
- Policy Documents: 200+ documents from major climate organizations
- Generated Files: Maps and charts saved to `/static/` directory

**Scalability Considerations**:
- MCP servers can be horizontally scaled
- Static files cached for performance
- Dataset size limited by memory (CSV loading)
- AI API rate limits apply

## Security Model

**API Security**:
- No authentication currently implemented
- CORS enabled for cross-origin access
- Input validation on query parameters
- Error message sanitization

**Data Security**:
- No sensitive data storage
- API keys in environment variables
- Static files publicly accessible
- No user data persistence

## Integration Points

**Frontend Integration**:
```javascript
// Standard query
const response = await fetch('/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: "solar energy in Brazil" })
});

// Response structure
{
  "modules": [
    {"type": "text", "heading": "Analysis", "texts": [...]},
    {"type": "map", "geojson_url": "/static/maps/file.geojson"}
  ],
  "metadata": {"modules_count": 2, "has_maps": true}
}
```

**Backend Integration**:
- MCP server protocol for adding new data sources
- FastAPI framework for additional endpoints
- Python ecosystem for data processing
- Git-based deployment workflow

This architecture provides a robust foundation for climate policy intelligence with automatic dataset discovery, multi-modal responses, and comprehensive data integration.