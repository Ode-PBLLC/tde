# TDE System Architecture

## Overview
The TDE (Transitional Data Exchange) system is a three-tier architecture for querying and visualizing climate policy data.

## Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        BROWSER[Web Browser]
        STREAMLIT[Streamlit Demo<br/>:8501]
        SCRIPTS[Scripts<br/>generate_featured_cache.py<br/>migrate_solar_to_db.py]
    end

    subgraph "API Layer - Port 8098"
        API[api_server.py<br/>FastAPI Server]
        MCP_CHAT[mcp_chat.py<br/>MCP Client Manager<br/>Singleton Pattern]
        KG_GEN[kg_embed_generator.py<br/>KG Embed Generator]
    end

    subgraph "Visualization Layer - Port 8100"
        KG_VIZ[kg_visualization_server.py<br/>D3.js Graph Server]
        KG_API[KG Query API<br/>/api/kg/query-subgraph]
    end

    subgraph "Data Layer - MCP Servers"
        MCP_KG[cpr_kg_server.py<br/>Knowledge Graph]
        MCP_SOLAR[solar_facilities_server.py<br/>Solar Facilities]
        MCP_GIST[gist_server.py<br/>GIST Data]
        MCP_LSE[lse_server.py<br/>LSE Data]
        MCP_FORMAT[response_formatter_server.py<br/>Response Formatter]
    end

    subgraph "Storage"
        SQLITE[(solar_facilities.db<br/>SQLite Database)]
        GRAPHML[knowledge_graph.graphml]
        CSV[concepts.csv]
        CACHE[Static Cache<br/>featured queries]
        EXCEL[Excel Data Files<br/>gist.xlsx, LSE files]
    end

    %% Client connections
    BROWSER --> STREAMLIT
    BROWSER --> API
    BROWSER --> KG_VIZ
    SCRIPTS --> API

    %% API Layer connections
    STREAMLIT --> API
    API --> MCP_CHAT
    API --> KG_GEN
    API -.proxy.-> KG_API
    MCP_CHAT --> MCP_KG
    MCP_CHAT --> MCP_SOLAR
    MCP_CHAT --> MCP_GIST
    MCP_CHAT --> MCP_LSE
    MCP_CHAT --> MCP_FORMAT

    %% KG Viz connections
    KG_GEN --> KG_API
    KG_VIZ --> GRAPHML
    KG_VIZ --> CSV
    KG_API --> KG_VIZ

    %% Data connections
    MCP_SOLAR --> SQLITE
    MCP_KG --> GRAPHML
    MCP_KG --> CSV
    MCP_GIST --> EXCEL
    MCP_LSE --> EXCEL
    API --> CACHE

    %% Styling
    classDef client fill:#e1f5e1,stroke:#4caf50,stroke-width:2px
    classDef api fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef viz fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef mcp fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    classDef storage fill:#f5f5f5,stroke:#757575,stroke-width:2px

    class BROWSER,STREAMLIT,SCRIPTS client
    class API,MCP_CHAT,KG_GEN api
    class KG_VIZ,KG_API viz
    class MCP_KG,MCP_SOLAR,MCP_GIST,MCP_LSE,MCP_FORMAT mcp
    class SQLITE,GRAPHML,CSV,CACHE,EXCEL storage
```

## Key Components

### 1. Client Layer
- **Web Browser**: Direct access to API endpoints and KG visualization
- **Streamlit Demo** (port 8501): Interactive web UI for testing all features
- **Scripts**: Utility scripts for cache generation and data migration

### 2. API Layer (Port 8098)
- **api_server.py**: Main FastAPI server handling all API requests
- **mcp_chat.py**: Singleton MCP client manager for efficient connection pooling
- **kg_embed_generator.py**: Generates embeddable KG visualizations

### 3. Visualization Layer (Port 8100)
- **kg_visualization_server.py**: Standalone D3.js graph visualization server
- **KG Query API**: RESTful API for graph queries and subgraph extraction
- Clean iframe-embeddable interface

### 4. Data Layer (MCP Servers)
- **cpr_kg_server.py**: Knowledge graph concepts and relationships
- **solar_facilities_server.py**: Geospatial solar facility data
- **gist_server.py**: GIST climate data
- **lse_server.py**: LSE policy data
- **response_formatter_server.py**: Formats and structures responses

### 5. Storage
- **SQLite Database**: Optimized solar facilities data (30s → <1s queries)
- **GraphML/CSV**: Knowledge graph structure and concepts
- **Static Cache**: Pre-computed featured query responses
- **Excel Files**: Source data for GIST and LSE

## Data Flow

1. **Query Flow**:
   - Client → API Server → MCP Client → MCP Servers → Data Sources
   - Responses flow back through formatters to client

2. **Visualization Flow**:
   - API Server generates KG URLs → Client opens KG Viz Server
   - KG Viz Server queries graph data directly

3. **Caching**:
   - Featured queries are pre-computed and cached
   - Static files served directly for performance

## Key Design Decisions

1. **Separation of Concerns**: Each layer has a specific responsibility
2. **Singleton MCP Client**: Reduces connection overhead (5-10x performance gain)
3. **Separate Visualization Server**: Clean embedding and independent scaling
4. **SQLite Migration**: Massive performance improvement for geospatial queries
5. **Static Caching**: Instant responses for common queries