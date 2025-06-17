# API Reference

## Base URL

```
Production: http://54.146.227.119:8099
Local Development: http://localhost:8099
```

## Authentication

Currently no authentication is required. All endpoints are publicly accessible.

## Content Types

All POST requests expect `Content-Type: application/json`
All responses return `Content-Type: application/json`

## Endpoints

### GET /health

Health check endpoint to verify API status.

**Response**:
```json
{
  "status": "healthy",
  "message": "Climate Policy Radar API is running"
}
```

**Status Codes**:
- `200` - API is healthy and operational

---

### POST /query

Main endpoint that returns structured responses for frontend consumption.

**Request Body**:
```json
{
  "query": "string",              // Required: User's question or topic
  "include_thinking": false       // Optional: Include AI reasoning process
}
```

**Example Request**:
```bash
curl -X POST http://54.146.227.119:8099/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "solar facilities in Brazil",
    "include_thinking": false
  }'
```

**Response Structure**:
```json
{
  "query": "solar facilities in Brazil",
  "modules": [
    {
      "type": "text",
      "heading": "Solar Facilities in Brazil",
      "texts": [
        "Brazil has 2,273 solar installations...",
        "The facilities range from small 0.3 MW..."
      ]
    },
    {
      "type": "map", 
      "mapType": "geojson",
      "geojson_url": "/static/maps/solar_facilities_brazil_abc123.geojson",
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
          {"label": "Brazil", "color": "#4CAF50", "size": "Large"}
        ]
      }
    },
    {
      "type": "table",
      "heading": "Facility Data",
      "columns": ["facility_id", "capacity_mw", "location"],
      "rows": [
        ["BR_001", 996.1, "Minas Gerais"],
        ["BR_002", 839.6, "Bahia"]
      ]
    },
    {
      "type": "chart",
      "chartType": "bar",
      "heading": "Capacity Distribution", 
      "data": {
        "labels": ["0-10 MW", "10-50 MW", "50+ MW"],
        "datasets": [{
          "label": "Facility Count",
          "data": [45, 123, 67],
          "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56"]
        }]
      }
    }
  ],
  "thinking_process": null,        // Only if include_thinking: true
  "metadata": {
    "modules_count": 4,
    "has_maps": true,
    "has_charts": true,
    "has_tables": true
  }
}
```

**Module Types**:

#### Text Module
```json
{
  "type": "text",
  "heading": "Section Title",
  "texts": ["Paragraph 1", "Paragraph 2", "..."]
}
```

#### Map Module  
```json
{
  "type": "map",
  "mapType": "geojson",
  "geojson_url": "/static/maps/filename.geojson",
  "viewState": {
    "center": [longitude, latitude],
    "zoom": 6,
    "bounds": {"north": lat, "south": lat, "east": lng, "west": lng}
  },
  "legend": {
    "title": "Legend Title",
    "items": [{"label": "Item", "color": "#hex", "size": "Large"}]
  }
}
```

#### Table Module
```json
{
  "type": "table", 
  "heading": "Table Title",
  "columns": ["col1", "col2", "col3"],
  "rows": [
    ["value1", "value2", "value3"],
    ["value4", "value5", "value6"]
  ]
}
```

#### Chart Module
```json
{
  "type": "chart",
  "chartType": "bar|line|pie|doughnut",
  "heading": "Chart Title",
  "data": {
    "labels": ["Label1", "Label2"],
    "datasets": [{
      "label": "Dataset Label", 
      "data": [10, 20],
      "backgroundColor": ["#color1", "#color2"]
    }]
  }
}
```

**Status Codes**:
- `200` - Successful response
- `500` - Server error with detailed error message

---

### POST /thorough-response

Debug endpoint that returns complete raw MCP data including all intermediate processing steps.

**Request Body**:
```json
{
  "query": "string",              // Required: User's question
  "include_thinking": false       // Optional: Include thinking process
}
```

**Response Structure**:
```json
{
  "query": "extreme weather",
  "raw_mcp_response": {
    "response": "The AI's final synthesized response text",
    "sources": ["Source document IDs"],
    "chart_data": [{
      "event_id": "EW001",
      "year": 2023,
      "type": "Hurricane", 
      "location": "Florida",
      "impact_rating": 5
    }],
    "map_data": [{
      "cluster_id": 14079,
      "capacity_mw": 996.1,
      "latitude": -15.94,
      "longitude": -43.50,
      "country": "Brazil"
    }],
    "visualization_data": null,
    "all_tool_outputs_for_debug": [
      {
        "tool_name": "GetPassagesMentioningConcept",
        "tool_args": {"concept": "extreme weather"},
        "tool_result_content": [...]
      },
      {
        "tool_name": "GetAvailableDatasets", 
        "tool_args": {},
        "tool_result_content": [...]
      }
    ],
    "ai_thought_process": "Step-by-step reasoning...",
    "formatted_response": {
      "modules": [...]  // Same structure as /query endpoint
    }
  },
  "metadata": {
    "endpoint": "thorough-response",
    "timestamp": "2025-06-06T04:22:16.585696",
    "note": "Contains all raw MCP data including debug info"
  }
}
```

**Use Cases**:
- Debugging MCP server interactions
- Understanding AI reasoning process
- Accessing raw dataset content
- Developing new response formatting

**Status Codes**:
- `200` - Successful response
- `500` - Server error with detailed traceback

---

### GET /example-response

Returns a sample response structure for frontend developers to understand the expected format.

**Response**: Sample structured response showing all possible module types and data structures.

**Status Codes**:
- `200` - Returns example response

---

### GET /static/maps/{filename}

Serves generated GeoJSON files for map visualization.

**Parameters**:
- `filename` - Generated GeoJSON filename (e.g., `solar_facilities_brazil_abc123.geojson`)

**Response**: GeoJSON file content

**Example GeoJSON Structure**:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-43.50, -15.94]
      },
      "properties": {
        "facility_id": "BR_FAC_001",
        "capacity_mw": 996.1,
        "country": "Brazil",
        "popup_title": "Solar Facility (Minas Gerais)",
        "popup_content": "Capacity: 996.1 MW<br>Location: Minas Gerais",
        "marker_color": "#4CAF50",
        "marker_size": 10,
        "marker_opacity": 0.8
      }
    }
  ]
}
```

**Status Codes**:
- `200` - File found and served
- `404` - File not found

## Query Types and Examples

### Geographic Queries

**Solar Facilities by Country**:
```json
{"query": "solar facilities in India"}
{"query": "show me solar installations in South Africa"}
{"query": "renewable energy infrastructure in Vietnam"}
```

**Response**: Text analysis + interactive map + facility data tables

### Policy Analysis Queries

**Climate Policy Topics**:
```json
{"query": "climate legislation"}
{"query": "NDCs and emissions targets"}
{"query": "green finance policies"}
```

**Response**: Policy document analysis + related data if available

### Data-Driven Queries

**Explicit Data Requests**:
```json
{"query": "extreme weather data"}
{"query": "show me climate change statistics"}
{"query": "energy capacity by country"}
```

**Response**: Structured datasets + visualizations + contextual analysis

### Comparative Queries

**Cross-Country Analysis**:
```json
{"query": "compare solar capacity between Brazil and India"}
{"query": "climate policies in developing countries"}
```

**Response**: Comparative analysis + multi-country data + charts

## Response Times

**Typical Performance**:
- Simple concept queries: 3-5 seconds
- Geographic queries with maps: 10-15 seconds  
- Complex multi-dataset queries: 15-20 seconds
- Thorough response queries: +5-10 seconds overhead

## Error Handling

### Standard Error Response

```json
{
  "detail": "Error description with context",
  "error_type": "ValueError|ConnectionError|APIError",
  "traceback": "Detailed Python traceback for debugging"
}
```

### Common Error Scenarios

**Missing API Keys**:
```json
{
  "detail": "Could not resolve authentication method. Expected either api_key or auth_token to be set"
}
```

**MCP Server Connection Issues**:
```json
{
  "detail": "Query processing failed: Connection closed\n\nTraceback: ..."
}
```

**Invalid Query Format**:
```json
{
  "detail": "Request validation error: field required"
}
```

## Rate Limits

Currently no rate limiting is implemented. However, the system is subject to:
- Anthropic API rate limits (varies by plan)
- OpenAI API rate limits (for embeddings)
- Server resource constraints

## CORS Policy

The API supports Cross-Origin Resource Sharing (CORS) with the following configuration:
- **Allowed Origins**: `["*"]` (all origins)
- **Allowed Methods**: `["GET", "POST"]`
- **Allowed Headers**: `["*"]`

## Frontend Integration Examples

### JavaScript/React

```javascript
// Basic query
async function queryClimateAPI(query) {
  const response = await fetch('http://54.146.227.119:8099/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, include_thinking: false })
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return await response.json();
}

// Render modules
function renderModules(modules) {
  return modules.map(module => {
    switch(module.type) {
      case 'text':
        return renderTextModule(module);
      case 'map':
        return renderMapModule(module);
      case 'table':
        return renderTableModule(module);
      case 'chart':
        return renderChartModule(module);
      default:
        return null;
    }
  });
}

// Map integration with Leaflet
function renderMapModule(module) {
  const map = L.map('map').setView(
    module.viewState.center, 
    module.viewState.zoom
  );
  
  // Load GeoJSON
  fetch(module.geojson_url)
    .then(response => response.json())
    .then(geojson => {
      L.geoJSON(geojson, {
        pointToLayer: (feature, latlng) => {
          return L.circleMarker(latlng, {
            radius: feature.properties.marker_size,
            fillColor: feature.properties.marker_color,
            fillOpacity: feature.properties.marker_opacity
          });
        },
        onEachFeature: (feature, layer) => {
          if (feature.properties.popup_content) {
            layer.bindPopup(feature.properties.popup_content);
          }
        }
      }).addTo(map);
    });
}
```

### Python Client

```python
import requests
import json

class ClimateAPIClient:
    def __init__(self, base_url="http://54.146.227.119:8099"):
        self.base_url = base_url
    
    def query(self, query_text, include_thinking=False):
        """Make a standard query to the API."""
        response = requests.post(
            f"{self.base_url}/query",
            json={
                "query": query_text,
                "include_thinking": include_thinking
            }
        )
        response.raise_for_status()
        return response.json()
    
    def thorough_query(self, query_text):
        """Get complete raw response for debugging."""
        response = requests.post(
            f"{self.base_url}/thorough-response",
            json={"query": query_text}
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage example
client = ClimateAPIClient()

# Basic query
result = client.query("solar energy in Brazil")
print(f"Found {result['metadata']['modules_count']} modules")

# Debug query  
debug_result = client.thorough_query("extreme weather")
print(f"Raw tools called: {len(debug_result['raw_mcp_response']['all_tool_outputs_for_debug'])}")
```

This API reference provides comprehensive documentation for integrating with the Climate Policy API system.