# Climate Policy API - Developer Guide

Complete reference for integrating with the Climate Policy API.

---

## Quick Start

```bash
# Health check
curl http://localhost:8099/health

# Stream a query (recommended)
curl -X POST http://localhost:8099/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze Brazilian oil companies environmental risks"}'

# Get featured queries for gallery
curl http://localhost:8099/featured-queries
```

---

## Endpoints

### 1. Streaming Analysis: `POST /query/stream`

**Primary endpoint** for real-time climate analysis with progress indicators.

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
- **Tool calls**: `{"type": "tool_call", "data": {"tool": "GetGistCompanies", "args": {...}}}`
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
      "id": "brazil-oil-risks",
      "title": "Brazilian Oil Companies Risk Analysis",
      "query": "Analyze environmental risk exposure of oil companies...",
      "image": "/static/images/brazil-oil-risks.jpg",
      "category": "Environmental Risk",
      "description": "Water stress, flood risks, and emissions analysis"
    }
  ],
  "metadata": {
    "total_queries": 6,
    "categories": ["Environmental Risk", "Renewable Energy", "Policy Analysis"]
  }
}
```

**Content updates**: Edit `static/featured_queries.json` - no backend changes needed.

---

## Response Structure

The API returns data as **modules** - an array of 5 possible types:

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

### 2. Chart Module
Chart.js v3+ compatible - use data directly, no transformation needed.

```json
{
  "type": "chart", 
  "chartType": "bar|line|pie",
  "heading": "Chart Title",
  "data": {
    "labels": ["Brazil", "India", "South Africa"],
    "datasets": [{
      "label": "Emissions (Mt CO2)",
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
  "columns": ["Company", "Emissions", "Risk Level"],
  "rows": [
    ["Petrobras", "420 Mt", "High"],
    ["Vibra", "290 Mt", "Medium"]
  ]
}
```

### 4. Map Module
Standard GeoJSON - works with Leaflet, Mapbox, etc.

```json
{
  "type": "map",
  "mapType": "geojson", 
  "heading": "Facility Locations",
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-38.5, -12.9]},
        "properties": {
          "facility_id": "BR_001",
          "popup_title": "Solar Facility",
          "popup_content": "Capacity: 2,511 MW",
          "marker_color": "#4CAF50",
          "marker_size": 20
        }
      }
    ]
  },
  "viewState": {
    "center": [-51.9253, -14.235],
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
    ["1", "GIST Corporate Directory", "GetGistCompanies", "Database", "Company data"],
    ["2", "GIST Water Risk", "GetGistRiskData", "Database", "Risk assessment"]
  ]
}
```

---

## Citation System

**Format**: Citations appear after text as `^1^`, `^2,3^`, `^1,2,3^`

**Example text**: `"Brazil has 4 oil companies ^1^. Water stress affects 64 companies ^2^."`

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

## React Integration Example

### Custom Hook
```jsx
import {useState, useEffect} from 'react';

export function useEnvironmentalQuery() {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState([]);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const executeQuery = async (queryText) => {
    setLoading(true);
    setProgress([]);
    setResults(null);
    setError(null);

    try {
      const response = await fetch('/query/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: queryText})
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
            const data = JSON.parse(line.slice(6));
            
            if (data.type === 'thinking') {
              setProgress(prev => [...prev, data.data]);
            } else if (data.type === 'complete') {
              setResults(data.data);
              setLoading(false);
            } else if (data.type === 'error') {
              setError(data.data.message);
              setLoading(false);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  return {loading, progress, results, error, executeQuery};
}
```

### Chart Component
```jsx
import {Chart} from 'chart.js/auto';

function ChartModule({module}) {
  useEffect(() => {
    const canvas = document.getElementById(`chart-${module.id}`);
    new Chart(canvas, {
      type: module.chartType,
      data: module.data, // Use directly - no transformation
      options: {
        responsive: true,
        maintainAspectRatio: false
      }
    });
  }, [module]);

  return (
    <div className="chart-container">
      <h3>{module.heading}</h3>
      <canvas id={`chart-${module.id}`} />
    </div>
  );
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

- **CORS**: Enabled for all origins (`*`)
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

## Example Queries

### Environmental Risk Analysis
```json
{"query": "Analyze environmental risk exposure of oil and gas companies in Brazil. Show companies with high water stress and flood risks."}
```
**Returns**: Text analysis + risk assessment table + map of at-risk assets

### Solar Infrastructure
```json  
{"query": "Show solar facilities across Brazil, India, and South Africa with capacity comparisons."}
```
**Returns**: Interactive map + capacity charts + facility data tables

### Policy Analysis
```json
{"query": "Compare climate policy effectiveness across major economies including NDC commitments."}
```
**Returns**: Policy framework analysis + implementation progress tables

### Emissions Trends
```json
{"query": "Show Scope 3 emissions trends for oil companies from 2020-2023 with upstream vs downstream breakdown."}
```
**Returns**: Time series charts + emissions data tables + trend analysis

---

This guide covers everything your frontend team needs to integrate with the API. For deployment and backend setup, see `DEPLOYMENT.md`.