# Generic MCP API - Developer Guide

Complete reference for integrating with the Generic MCP API.

---

## Quick Start

```bash
# Health check
curl http://localhost:8098/health

# Stream a query (recommended)
curl -X POST http://localhost:8098/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Your analysis question here"}'

# Get featured queries for gallery
curl http://localhost:8098/featured-queries
```

---

## Endpoints

### 1. Streaming Analysis: `POST /query/stream`

**Primary endpoint** for real-time analysis with progress indicators.

#### Request
```json
{
  "query": "Your analysis question here"
}
```

#### Response: Server-Sent Events (SSE)
```
Content-Type: text/event-stream
Cache-Control: no-cache
```

**Event Types:**
- **Progress**: `{"type": "thinking", "data": {"message": "🚀 Initializing...", "category": "initialization"}}`
- **Tool calls**: `{"type": "tool_call", "data": {"tool": "ToolName", "args": {...}}}`
- **Final result**: `{"type": "complete", "data": {"query": "...", "modules": [...]}}`
- **Errors**: `{"type": "error", "data": {"message": "...", "traceback": "..."}}`

**Timeout**: 120 seconds

---

### 2. Synchronous Query: `POST /query`

Immediate response without streaming (for simple integrations).

#### Request
```json
{
  "query": "Your question here",
  "include_thinking": false
}
```

#### Response
```json
{
  "query": "Your question",
  "modules": [...],
  "thinking_process": "string | null",
  "metadata": {
    "modules_count": 4,
    "has_maps": true,
    "has_charts": true,
    "has_tables": true
  }
}
```

---

### 3. Featured Queries: `GET /featured-queries`

Content management endpoint for frontend gallery boxes.

#### Response
```json
{
  "featured_queries": [
    {
      "id": "example-analysis",
      "title": "Example Analysis",
      "query": "Analyze data patterns and provide insights...",
      "image": "/static/images/example-analysis.jpg",
      "category": "Data Analysis",
      "description": "Comprehensive data analysis example"
    }
  ],
  "metadata": {
    "total_queries": 6,
    "categories": ["Data Analysis", "Reporting", "Visualization"]
  }
}
```

**Content updates**: Edit `static/featured_queries.json` - no backend changes needed.

---

### 4. Health Check: `GET /health`

System health endpoint.

#### Response
```json
{
  "status": "healthy", 
  "message": "Service Name is running"
}
```

---

## Configuration

### Environment Variables
```bash
# API Configuration
API_TITLE="Your Project API"
API_VERSION="1.0.0" 
API_DESCRIPTION="Your project description"
SERVICE_NAME="Your Project Service"

# Directory Configuration
DATA_DIR=./data
CONFIG_DIR=./config
STATIC_DIR=./static

# Server Configuration  
PORT=8098
```

### Directory Structure
```
project/
  api_server.py          # Main API server
  pathing.py            # Path helper
  config/
    servers.base.json    # Generic MCP servers
    servers.local.json   # Local overrides (gitignored)
  data/                 # Data files (gitignored)
  static/              # Static assets
    featured_queries.json
    images/
  mcp/                 # MCP server implementations
```

---

## Response Structure

The API returns data as **modules** - an array of 5 possible types:

### 1. Text Module
```json
{
  "type": "text",
  "heading": "Analysis Title",
  "texts": [
    "Data shows X pattern ^1^.",
    "Y metric indicates Z trend ^2^."
  ]
}
```

### 2. Chart Module
Chart.js v3+ compatible - use data directly, no transformation needed.

```json
{
  "type": "chart", 
  "chartType": "bar|line|pie",
  "heading": "Chart Title",
  "data": {
    "labels": ["Category A", "Category B", "Category C"],
    "datasets": [{
      "label": "Values",
      "data": [420, 290, 180],
      "backgroundColor": ["#4CAF50", "#FF9800", "#F44336"]
    }]
  }
}
```

### 3. Table Module
```json
{
  "type": "table",
  "heading": "Data Summary",
  "columns": ["Item", "Value", "Category"],
  "rows": [
    ["Item A", "420", "Type 1"],
    ["Item B", "290", "Type 2"]
  ]
}
```

### 4. Map Module
Standard GeoJSON - works with Leaflet, Mapbox, etc.

```json
{
  "type": "map",
  "mapType": "geojson", 
  "heading": "Location Data",
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-75.5, 45.5]},
        "properties": {
          "id": "LOC_001",
          "popup_title": "Location Name",
          "popup_content": "Description: Details here",
          "marker_color": "#4CAF50",
          "marker_size": 20
        }
      }
    ]
  },
  "viewState": {
    "center": [-75.5, 45.5],
    "zoom": 6
  }
}
```

### 5. Citation Table
Always appears last. Contains references for all `^1,2,3^` citations in text.

```json
{
  "type": "numbered_citation_table",
  "heading": "References",
  "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
  "rows": [
    ["1", "Data Source A", "ToolName", "Database", "Source description"],
    ["2", "Data Source B", "AnotherTool", "API", "Another source"]
  ]
}
```

---

## Citation System

**Format**: Citations appear after text as `^1^`, `^2,3^`, `^1,2,3^`

**Example text**: `"Data shows 4 items ^1^. Analysis indicates 64% increase ^2^."`

**Processing**: Convert to your preferred display format:
```javascript
function processCitations(text) {
  return text.replace(/\^(\d+(?:,\d+)*)\^/g, (match, nums) => {
    const citationNums = nums.split(',');
    const links = citationNums.map(num => 
      `<a href="#citation-${num}" class="citation">${num}</a>`
    ).join(',');
    return `<sup>[${links}]</sup>`;
  });
}
```

**Cross-references**: Citation numbers match the `#` column in the citation table.

---

## Frontend Integration

### Streaming Client Example
```javascript
async function queryAPI(query) {
  const response = await fetch('/query/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query})
  });
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const event = JSON.parse(line.slice(6));
        handleEvent(event);
      }
    }
  }
}

function handleEvent(event) {
  switch (event.type) {
    case 'thinking':
      showProgress(event.data.message);
      break;
    case 'complete':
      renderModules(event.data.modules);
      break;
    case 'error':
      showError(event.data.message);
      break;
  }
}
```

### Module Rendering
```javascript
function renderModules(modules) {
  modules.forEach(module => {
    switch (module.type) {
      case 'text':
        renderText(module.heading, module.texts);
        break;
      case 'chart':
        // Direct Chart.js usage - no transformation needed
        new Chart(canvas, {
          type: module.chartType,
          data: module.data,
          options: {responsive: true}
        });
        break;
      case 'table':
        renderTable(module.columns, module.rows);
        break;
      case 'map':
        // Standard GeoJSON - works with any mapping library
        renderMap(module.geojson, module.viewState);
        break;
      case 'numbered_citation_table':
        renderCitations(module.rows);
        break;
    }
  });
}
```

### Featured Queries Gallery
```javascript
async function loadFeaturedQueries() {
  const response = await fetch('/featured-queries');
  const data = await response.json();
  
  data.featured_queries.forEach(query => {
    createQueryCard({
      title: query.title,
      description: query.description,
      image: query.image,
      category: query.category,
      onClick: () => queryAPI(query.query)
    });
  });
}
```

---

## Error Handling

### HTTP Status Codes
- **200**: Success
- **500**: Internal server error with detailed message

### Error Response Format
```json
{
  "detail": "Error description with traceback information"
}
```

### Stream Error Events
```javascript
data: {"type": "error", "data": {"message": "Description", "traceback": "..."}}
```

---

## Technical Notes

- **CORS**: Enabled for all origins (`*`) - configure appropriately for production
- **Timeout**: 120 seconds for all endpoints
- **Chart.js**: v3+ compatible, no data transformation needed
- **GeoJSON**: Standard format, compatible with all mapping libraries
- **Citations**: Always appear after referenced text
- **Module order**: Text → Charts → Tables → Maps → Citations
- **Static files**: Images served via `/static/` route

---

## Content Management

### Updating Featured Queries
1. Edit `static/featured_queries.json`
2. Add images to `static/images/` 
3. No backend restart needed

### JSON Structure
```json
{
  "featured_queries": [
    {
      "id": "unique-slug",
      "title": "Display Title", 
      "query": "Full query text for /query/stream",
      "image": "/static/images/filename.jpg",
      "category": "Category Name",
      "description": "Brief description"
    }
  ]
}
```

### Image Guidelines
- **Size**: 400x300px (4:3 ratio recommended)
- **Format**: JPG, PNG, WebP
- **Location**: `static/images/` directory
- **Naming**: Use kebab-case matching query ID

---

This guide covers everything your frontend team needs to integrate with the Generic MCP API. Customize the examples and content for your specific domain and use cases.