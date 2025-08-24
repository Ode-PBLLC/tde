# Interactive Query-Specific Knowledge Graph Visualization

## Overview

This implementation creates an interactive, zoomable visualization of the Knowledge Graph (KG) that's specifically relevant to user queries. The system extracts concepts from queries and displays a focused subgraph showing the relationships between those concepts, datasets, and document passages.

## Key Features

### 1. Query-Specific Subgraph Generation
- **Endpoint**: `/api/kg/query-subgraph` in `kg_visualization_server.py`
- **Concept Extraction**: Uses both exact matching and semantic similarity (via MCP tools)
- **Intelligent Expansion**: BFS expansion around query concepts with configurable depth
- **Node Type Filtering**: Selectively includes concepts, datasets, and passages

### 2. Interactive Frontend Enhancements
- **Query Mode**: New section in sidebar for entering natural language queries
- **Visual Highlighting**: Query-relevant nodes highlighted in red with thicker borders
- **Auto-loading**: URL query parameter support for direct links to specific queries
- **Configurable Options**: Checkboxes to include/exclude datasets and passages

### 3. API Integration
- **Enhanced Metadata**: Main API responses now include KG visualization URLs
- **Direct Links**: Query-specific URLs automatically generated for each API response
- **Seamless Integration**: KG visualization accessible from any API response

## Implementation Details

### Backend Changes

#### 1. Enhanced KG Visualization Server (`kg_visualization_server.py`)

**New Dependencies**:
```python
import re
import sys
from typing import Set
# MCP tools integration for concept extraction
```

**New Request Model**:
```python
class QuerySubgraphRequest(BaseModel):
    query: str
    depth: int = 2
    max_nodes: int = 100
    include_datasets: bool = True
    include_passages: bool = True
```

**Core Functions**:
- `extract_concepts_from_query()`: Multi-strategy concept extraction
- `build_query_subgraph()`: Intelligent subgraph construction
- `get_query_subgraph()`: API endpoint handler

#### 2. Main API Server Integration (`api_server.py`)

**Enhanced Metadata Function**:
```python
def _generate_enhanced_metadata(structured_response, full_result=None, query_text=""):
    # ... existing metadata generation ...
    return {
        # ... existing fields ...
        "kg_visualization_url": "http://localhost:8100",
        "kg_query_url": f"http://localhost:8100?query={query_text.replace(' ', '%20')}"
    }
```

### Frontend Changes

#### 1. HTML Structure (`static/kg_visualization.html`)

**New Query Mode Section**:
```html
<div class="section">
    <h3>Query Mode</h3>
    <textarea id="queryInput" placeholder="Enter your query..." rows="3"></textarea>
    <div class="query-options">
        <label><input type="checkbox" id="includeDatasets" checked> Include Datasets</label>
        <label><input type="checkbox" id="includePassages" checked> Include Passages</label>
    </div>
    <button id="queryBtn">Analyze Query</button>
    <div id="queryResults"></div>
</div>
```

**Enhanced Legend**:
- Added "Query Relevant" indicator with red highlighting

#### 2. JavaScript Functionality (`static/js/kg_graph.js`)

**New Methods**:
- `analyzeQuery()`: Processes natural language queries
- `showQueryResults()`: Displays analysis results
- `highlightQueryNodes()`: Visual highlighting of relevant nodes
- `checkQueryParameter()`: Auto-loads queries from URL parameters

**Enhanced Features**:
- Query concept storage for persistent highlighting
- Visual encoding for query relevance
- Automatic query processing from URL parameters

#### 3. CSS Styling (`static/css/kg_graph.css`)

**New Styles**:
- Query input styling (textarea, checkboxes)
- Result display styling (success, error, loading states)
- Query options layout and spacing

## Usage Examples

### 1. Direct Query Analysis
1. Visit `http://localhost:8100`
2. Enter query: "Water stress impacts on financial sector"
3. Click "Analyze Query"
4. View relevant subgraph with highlighted concepts

### 2. API Integration
```bash
# API call returns metadata with KG links
POST /query
{
  "query": "Climate policy effectiveness in Brazil",
  "include_thinking": false
}

# Response includes:
{
  "metadata": {
    "kg_visualization_url": "http://localhost:8100",
    "kg_query_url": "http://localhost:8100?query=Climate%20policy%20effectiveness%20in%20Brazil"
  }
}
```

### 3. Direct URL Access
```
http://localhost:8100?query=renewable%20energy%20policy%20Brazil
```
Automatically loads and analyzes the specified query.

## Architecture Benefits

### 1. Performance Optimizations
- **Focused Subgraphs**: Only loads relevant portions of the KG
- **Intelligent Filtering**: Reduces visual complexity by filtering node types
- **Efficient Expansion**: BFS algorithm with configurable limits

### 2. Educational Value
- **Concept Discovery**: Shows how the system interprets queries
- **Relationship Visualization**: Displays connections between concepts
- **Interactive Exploration**: Click nodes to see details and expand neighborhoods

### 3. Integration Benefits
- **Seamless Workflow**: Direct links from API responses
- **URL Shareability**: Query-specific URLs for collaboration
- **Context Preservation**: Visual representation complements textual responses

## Future Enhancements

### Potential Improvements
1. **Enhanced Concept Extraction**: More sophisticated NLP for concept identification
2. **Path Analysis**: Shortest path visualization between query concepts
3. **Temporal Filtering**: Date-based filtering of passages and documents
4. **Export Functionality**: Save subgraphs as images or data files
5. **Collaborative Features**: Share and annotate specific subgraphs

### Technical Considerations
1. **Scalability**: Caching for frequently queried concept combinations
2. **Performance**: Lazy loading for large subgraphs
3. **Accessibility**: Keyboard navigation and screen reader support
4. **Mobile Optimization**: Responsive design for mobile devices

## Running the System

### Prerequisites
- Python 3.8+
- Required packages: `fastapi`, `uvicorn`, `networkx`, `pandas`
- Knowledge Graph files: `extras/knowledge_graph.graphml`, `extras/concepts.csv`

### Starting the Server
```bash
# Start KG visualization server
python kg_visualization_server.py

# Server runs on http://localhost:8100
```

### Integration with Main API
The KG visualization server runs independently but integrates seamlessly with the main API server (`api_server.py`) through metadata links.

## Conclusion

This implementation provides a comprehensive solution for interactive, query-specific Knowledge Graph visualization. It bridges the gap between textual API responses and visual understanding, offering users an intuitive way to explore the conceptual relationships underlying their queries.

The system is designed to be both educational and functional, helping users understand how the AI system interprets and processes their queries while providing a powerful tool for exploring the broader knowledge landscape around their topics of interest.