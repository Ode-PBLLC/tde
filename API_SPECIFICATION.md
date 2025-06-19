# Climate Policy Radar API Specification

**Version:** 1.0  
**Base URL:** `http://localhost:8099`  
**Protocol:** HTTP/1.1  
**Content-Type:** `application/json`

---

## Overview

This API provides structured climate policy and environmental risk analysis through multiple endpoints. The primary endpoint streams real-time analysis with progress indicators, while synchronous endpoints provide immediate responses.

---

## Endpoints

### 1. Primary Streaming Endpoint

**POST** `/query/stream`

Real-time streaming analysis with Server-Sent Events (SSE).

#### Request
```json
{
  "query": "string"
}
```

#### Response Format
Server-Sent Events stream with the following event types:

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

#### Event Types

**Progress Events:**
```javascript
data: {"type": "thinking", "data": {"message": "🚀 Initializing search...", "category": "initialization"}}
data: {"type": "thinking_complete", "data": {"message": "✅ Completed analysis", "category": "processing"}}
```

**Tool Events:**
```javascript
data: {"type": "tool_call", "data": {"tool": "GetGistCompanies", "args": {"sector": "OGES"}}}
data: {"type": "tool_result", "data": {"tool": "GetGistCompanies", "result": {...}}}
```

**Final Response:**
```javascript
data: {"type": "complete", "data": {
  "query": "string",
  "modules": [...],
  "metadata": {...}
}}
```

**Error Event:**
```javascript
data: {"type": "error", "data": {"message": "string", "traceback": "string"}}
```

---

### 2. Synchronous Query Endpoint

**POST** `/query`

Immediate structured response without streaming.

#### Request
```json
{
  "query": "string",
  "include_thinking": false  // optional, default: false
}
```

#### Response
```json
{
  "query": "string",
  "modules": [...],
  "thinking_process": "string | null",
  "metadata": {
    "modules_count": 4,
    "has_maps": true,
    "has_charts": true,
    "has_tables": true,
    "module_types": ["text", "chart", "table", "map", "numbered_citation_table"]
  }
}
```

---

### 3. Health Check

**GET** `/health`

#### Response
```json
{
  "status": "healthy",
  "message": "Climate Policy Radar API is running"
}
```

---

### 4. Featured Queries (Pseudo-CMS)

**GET** `/featured-queries`

Returns curated list of featured queries with images for frontend gallery.

#### Response
```json
{
  "featured_queries": [
    {
      "id": "brazil-oil-environmental-risks",
      "title": "Brazilian Oil Companies Environmental Risk Analysis",
      "query": "Analyze the environmental risk exposure of oil and gas companies in Brazil...",
      "image": "/static/images/brazil-oil-risks.jpg",
      "category": "Environmental Risk",
      "description": "Comprehensive analysis of water stress, flood risks, and emissions trends"
    },
    {
      "id": "global-solar-capacity",
      "title": "Global Solar Capacity Analysis", 
      "query": "Show me solar facilities and capacity data across Brazil, India...",
      "image": "/static/images/global-solar-capacity.jpg",
      "category": "Renewable Energy",
      "description": "Interactive maps and charts showing global solar infrastructure"
    }
  ],
  "metadata": {
    "last_updated": "2024-01-15T10:00:00Z",
    "total_queries": 6,
    "categories": ["Environmental Risk", "Renewable Energy", "Policy Analysis"],
    "served_at": "2024-01-15T10:30:00Z"
  }
}
```

**Usage:**
- Use returned queries as clickable cards/boxes in frontend gallery
- Images are served via existing `/static/images/` route
- Update content by modifying `static/featured_queries.json` file
- No database or backend code changes needed for content updates

---

## Module Types

The API returns data as an array of modules. Each module has a `type` field that determines its structure.

### 1. Text Module

Contains text content with inline citations.

```json
{
  "type": "text",
  "heading": "Climate Policy Analysis",
  "texts": [
    "Brazil's oil sector shows significant emissions growth ^1,3^.",
    "Water stress impacts 15.25% of banking assets ^2^."
  ]
}
```

**Fields:**
- `type`: Always `"text"`
- `heading`: String title for the text section
- `texts`: Array of strings, each may contain citations in `^1,2,3^` format

---

### 2. Chart Module

Chart.js v3+ compatible visualization data.

```json
{
  "type": "chart",
  "chartType": "bar",
  "heading": "Emissions by Country",
  "data": {
    "labels": ["Brazil", "India", "South Africa"],
    "datasets": [{
      "label": "Total Emissions (Mt CO2)",
      "data": [420, 290, 180],
      "backgroundColor": ["#4CAF50", "#FF9800", "#F44336"],
      "borderColor": "#333",
      "borderWidth": 1
    }]
  }
}
```

**Fields:**
- `type`: Always `"chart"`
- `chartType`: `"bar"` | `"line"` | `"pie"`
- `heading`: String title for the chart
- `data`: Chart.js compatible data object
  - `labels`: Array of x-axis labels
  - `datasets`: Array of dataset objects with `label`, `data`, `backgroundColor`, etc.

**Chart Types Generated:**
- **Bar Charts**: Country comparisons, company rankings, capacity distributions
- **Line Charts**: Time series data, emissions trends over time
- **Pie Charts**: Risk level distributions, category breakdowns

---

### 3. Table Module

Standard data table with rows and columns.

```json
{
  "type": "table",
  "heading": "Companies Summary",
  "columns": ["Company Code", "Company Name", "Sector", "Country"],
  "rows": [
    ["PETROB00002", "Vibra Energia SA", "OGES", "Brazil"],
    ["COSANO00001", "Cosan SA", "OGES", "Brazil"]
  ]
}
```

**Fields:**
- `type`: Always `"table"`
- `heading`: String title for the table
- `columns`: Array of column header strings
- `rows`: Array of arrays, each inner array represents a table row

---

### 4. Map Module

Interactive map with GeoJSON data and markers.

```json
{
  "type": "map",
  "mapType": "geojson",
  "heading": "Solar Facilities",
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Point",
          "coordinates": [-38.5014, -12.9734]
        },
        "properties": {
          "facility_id": "BR_FAC_001",
          "capacity_mw": 2511.1,
          "popup_title": "Large Solar Complex",
          "popup_content": "Capacity: 2,511.1 MW<br>Location: Bahia",
          "marker_color": "#4CAF50",
          "marker_size": 20,
          "marker_opacity": 0.8
        }
      }
    ]
  },
  "viewState": {
    "center": [-51.9253, -14.235],
    "zoom": 6,
    "bounds": {
      "north": -12.9734,
      "south": -19.9167,
      "east": -38.5014,
      "west": -43.9345
    }
  },
  "legend": {
    "title": "Solar Facilities",
    "items": [
      {
        "label": "Brazil",
        "color": "#4CAF50",
        "description": "Size represents capacity"
      }
    ]
  }
}
```

**Fields:**
- `type`: Always `"map"`
- `mapType`: Always `"geojson"`
- `heading`: String title for the map
- `geojson`: Standard GeoJSON FeatureCollection
- `viewState`: Initial map view configuration
- `legend`: Map legend configuration

---

### 5. Citation Table Module

References and sources table, always appears last.

```json
{
  "type": "numbered_citation_table",
  "heading": "References",
  "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
  "rows": [
    ["1", "GIST Corporate Directory", "GetGistCompanies", "Database", "Company sector and location data"],
    ["2", "GIST Water Risk Database", "GetGistRiskData", "Database", "Water stress exposure analysis"],
    ["3", "GIST Scope 3 Emissions", "GetGistScope3Emissions", "Time Series", "2020-2023 emissions data"]
  ]
}
```

**Fields:**
- `type`: Always `"numbered_citation_table"`
- `heading`: Always `"References"`
- `columns`: Always `["#", "Source", "ID/Tool", "Type", "Description"]`
- `rows`: Array of citation entries, first column is citation number

---

## Citation System

### Format
Citations appear inline as superscript notation: `^1^`, `^2,3^`, `^1,2,3^`

### Placement
Citations appear **after** the text they reference:
```
"Brazil has 45 oil companies ^1^."
"Emissions increased 23% from 2020-2023 ^2,3^."
```

### Cross-References
- Citation numbers in text match the `#` column in the `numbered_citation_table`
- Numbers are sequential starting from 1
- Multiple citations are comma-separated: `^1,2,3^`

### Frontend Processing
Convert `^1,2,3^` to your preferred display format:

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

---

## Error Handling

### HTTP Status Codes
- `200`: Success
- `500`: Internal server error

### Error Response Format
```json
{
  "detail": "Error message with traceback information"
}
```

### Stream Error Events
```javascript
data: {"type": "error", "data": {"message": "Error description", "traceback": "..."}}
```

---

## Performance & Limits

### Timeouts
- Streaming endpoint: 120 seconds maximum
- Synchronous endpoint: 120 seconds maximum

### CORS
- All origins allowed (`Access-Control-Allow-Origin: *`)
- All methods and headers permitted

### Caching
- Responses include `Cache-Control: no-cache` for streaming
- No response caching recommended due to real-time nature

---

## Implementation Notes

### Frontend Integration
1. **Use the streaming endpoint** (`/query/stream`) as primary interface
2. **Process events incrementally** to show progress
3. **Handle citations** by converting `^1,2,3^` format to your UI needs
4. **Render modules sequentially** as they appear in the array
5. **Always display citation table last** for reference

### Module Rendering Order
Modules appear in this typical order:
1. `text` - Main analysis content
2. `chart` - Data visualizations 
3. `table` - Supporting data tables
4. `map` - Geographic visualizations
5. `numbered_citation_table` - References (always last)

### Chart Integration
Charts use Chart.js v3+ format - no data transformation needed:

```javascript
new Chart(canvas, {
  type: module.chartType,
  data: module.data,  // Use directly
  options: { responsive: true }
});
```

### Map Integration
Maps use standard GeoJSON - compatible with Leaflet, Mapbox, etc.:

```javascript
// Leaflet example
const map = L.map('map').setView(module.viewState.center, module.viewState.zoom);
L.geoJSON(module.geojson).addTo(map);
```

---

## Examples

### Complete Response Example
```json
{
  "query": "Analyze Brazilian oil companies' water risks",
  "modules": [
    {
      "type": "text",
      "heading": "Water Risk Analysis",
      "texts": [
        "Brazil has 4 major oil & gas companies in our database ^1^.",
        "Water stress affects 64 companies with 15.25% of assets at risk ^2^."
      ]
    },
    {
      "type": "chart",
      "chartType": "bar",
      "heading": "Companies by Water Risk Level",
      "data": {
        "labels": ["High Risk", "Medium Risk", "Low Risk"],
        "datasets": [{
          "label": "Number of Companies",
          "data": [25, 30, 9],
          "backgroundColor": ["#f44336", "#ff9800", "#4caf50"]
        }]
      }
    },
    {
      "type": "table",
      "heading": "Brazilian Oil & Gas Companies",
      "columns": ["Company Code", "Company Name", "Datasets Available"],
      "rows": [
        ["PETROB00002", "Vibra Energia SA", "5 datasets"],
        ["COSANO00001", "Cosan SA", "5 datasets"]
      ]
    },
    {
      "type": "numbered_citation_table",
      "heading": "References", 
      "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
      "rows": [
        ["1", "GIST Corporate Directory", "GetGistCompanies", "Database", "Oil & gas companies in Brazil"],
        ["2", "GIST Water Risk Database", "GetGistRiskData", "Database", "Water stress risk assessment data"]
      ]
    }
  ],
  "metadata": {
    "modules_count": 4,
    "has_maps": false,
    "has_charts": true,
    "has_tables": true,
    "module_types": ["text", "chart", "table", "numbered_citation_table"]
  }
}
```

This specification defines the complete, fixed API contract. Backend implementations must conform to these exact structures, and frontend applications can rely on this format remaining stable.