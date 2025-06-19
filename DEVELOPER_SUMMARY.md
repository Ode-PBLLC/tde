# Climate Policy API - Developer Summary

## Core Endpoints

### 1. Streaming Analysis: `POST /query/stream`

**Primary endpoint** for real-time climate policy analysis.

```bash
curl -X POST http://localhost:8099/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze Brazilian oil companies environmental risks"}'
```

**Response**: Server-Sent Events (SSE) stream
- **Progress events**: `{"type": "thinking", "data": {...}}`
- **Tool events**: `{"type": "tool_call", "data": {...}}`
- **Final result**: `{"type": "complete", "data": {"query": "...", "modules": [...]}}`

**Timeout**: 120 seconds max

---

### 2. Featured Queries: `GET /featured-queries`

**Content management** endpoint for frontend gallery.

```bash
curl http://localhost:8099/featured-queries
```

**Response**: Curated list of query/image pairs
```json
{
  "featured_queries": [
    {
      "id": "brazil-oil-risks",
      "title": "Brazilian Oil Company Risk Analysis",
      "query": "Full query text to send to /query/stream",
      "image": "/static/images/brazil-oil-risks.jpg",
      "category": "Environmental Risk",
      "description": "Brief description for UI"
    }
  ]
}
```

**Content Updates**: Edit `static/featured_queries.json` - no code changes needed.

---

## Response Module Types

The API returns an array of **5 module types**:

### 1. Text Module
```json
{
  "type": "text",
  "heading": "Analysis Title",
  "texts": [
    "Brazil has 4 oil companies ^1^.",
    "Water stress affects 64 companies ^2^."
  ]
}
```

### 2. Chart Module (Chart.js Compatible)
```json
{
  "type": "chart", 
  "chartType": "bar|line|pie",
  "heading": "Chart Title",
  "data": {
    "labels": ["Brazil", "India"],
    "datasets": [{
      "label": "Emissions (Mt CO2)",
      "data": [420, 290],
      "backgroundColor": ["#4CAF50", "#FF9800"]
    }]
  }
}
```

### 3. Table Module
```json
{
  "type": "table",
  "heading": "Data Table",
  "columns": ["Company", "Emissions", "Risk Level"],
  "rows": [
    ["Petrobras", "420 Mt", "High"],
    ["Vibra", "290 Mt", "Medium"]
  ]
}
```

### 4. Map Module (GeoJSON)
```json
{
  "type": "map",
  "mapType": "geojson", 
  "heading": "Facility Locations",
  "geojson": {
    "type": "FeatureCollection",
    "features": [...]
  },
  "viewState": {
    "center": [-51.9253, -14.235],
    "zoom": 6
  }
}
```

### 5. Citation Table
```json
{
  "type": "numbered_citation_table",
  "heading": "References",
  "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
  "rows": [
    ["1", "GIST Corporate Directory", "GetGistCompanies", "Database", "Company data"],
    ["2", "GIST Water Risk", "GetGistRiskData", "Database", "Risk assessment"]
  ]
}
```

---

## Citation System

**Format**: `^1,2,3^` appears after referenced text
**Processing**: Convert to your preferred display format
```javascript
text.replace(/\^(\d+(?:,\d+)*)\^/g, (match, nums) => {
  return `<sup>[${nums.split(',').join(',')}]</sup>`;
});
```

---

## Frontend Integration

### Basic Streaming Client
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
  }
}
```

### Module Rendering
```javascript
function renderModules(modules) {
  modules.forEach(module => {
    switch (module.type) {
      case 'text':
        renderText(module);
        break;
      case 'chart':
        new Chart(canvas, {
          type: module.chartType,
          data: module.data  // Use directly
        });
        break;
      case 'table':
        renderTable(module.columns, module.rows);
        break;
      case 'map':
        renderMap(module.geojson, module.viewState);
        break;
    }
  });
}
```

---

## Key Technical Notes

- **CORS**: Enabled for all origins
- **Chart.js**: v3+ compatible, no data transformation needed
- **GeoJSON**: Standard format, works with Leaflet/Mapbox
- **Citations**: Always appear after referenced text as `^1,2,3^`
- **Module Order**: Text → Charts → Tables → Maps → Citations
- **Error Handling**: 500 status with detail message

---

## Content Management

**Update Featured Queries**: 
1. Edit `static/featured_queries.json`
2. Add images to `static/images/`
3. No backend restart needed

**Add New Categories**: Just include in the JSON metadata.

Complete API specification: See `API_SPECIFICATION.md`