# Complete KG Integration Guide

## Overview

The Knowledge Graph concepts and relationships are now **fully integrated** into the main API server responses. Your dev team gets KG data directly in both regular and streaming API calls.

## What's Changed

### ✅ **Main API Server Enhanced**
Both `/query` and `/query/stream` endpoints now include:
- **`concepts[]`** - List of relevant concepts with query relevance flags
- **`relationships[]`** - Formatted concept relationships ("ConceptA -> ConceptB")

### ✅ **No Additional API Calls Needed**
- KG data is automatically fetched and included
- Graceful fallback if KG server unavailable
- Single request gets everything your team needs

## API Response Format

### Regular `/query` Endpoint
```json
{
  "query": "Solar energy in Brazil",
  "modules": [
    {"type": "text", "heading": "Solar Energy Analysis", "texts": [...]},
    {"type": "map", "mapType": "geojson_url", "geojson_url": "..."},
    {"type": "table", "heading": "Solar Facilities", "rows": [...]}
  ],
  "concepts": [
    {
      "id": "Q123",
      "label": "Solar Energy",
      "is_query_relevant": true
    },
    {
      "id": "Q456", 
      "label": "Renewable Energy",
      "is_query_relevant": false
    }
  ],
  "relationships": [
    {
      "source_label": "Solar Energy",
      "target_label": "Renewable Energy", 
      "relationship_type": "SUBCONCEPT_OF",
      "formatted": "Solar Energy -> Renewable Energy (SUBCONCEPT_OF)"
    }
  ],
  "metadata": {
    "modules_count": 3,
    "has_maps": true,
    "kg_visualization_url": "http://localhost:8100"
  }
}
```

### Streaming `/query/stream` Endpoint
```
data: {"type": "thinking", "data": {"message": "🔍 Searching for solar energy data"}}

data: {"type": "complete", "data": {
  "query": "Solar energy in Brazil",
  "modules": [...],
  "concepts": [
    {"id": "Q123", "label": "Solar Energy", "is_query_relevant": true}
  ],
  "relationships": [
    {"formatted": "Solar Energy -> Renewable Energy (SUBCONCEPT_OF)"}
  ],
  "metadata": {...}
}}
```

## Running the Complete System

### 1. **Start Both Servers**
```bash
# Terminal 1 - KG Visualization Server
python kg_visualization_server.py
# Runs on http://localhost:8100

# Terminal 2 - Main API Server (with KG integration)
python api_server.py  
# Runs on http://localhost:8098
```

### 2. **Test Integration**
```bash
# Test the integrated API
python test_integrated_api.py

# Or test manually:
curl -X POST "http://localhost:8098/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Solar energy in Brazil"}' | jq '.concepts, .relationships'
```

## Benefits for Your Dev Team

### 🎯 **Simplified Architecture**
```
Before:
├── Call main API → Get modules  
└── Call KG API → Get concepts/relationships

After:  
└── Call main API → Get modules + concepts + relationships
```

### 🔄 **Consistent Data Flow**
- **Regular requests**: Everything in one response
- **Streaming requests**: KG data in final "complete" event
- **Graceful degradation**: Works even if KG server is down

### 📊 **Rich Context**
Your UI can now show:
- **Climate analysis modules** (text, maps, charts, tables)
- **Concept lists** with query relevance highlighting  
- **Relationship networks** in "ConceptA -> ConceptB" format
- **Visual graph links** for deeper exploration

## Example Frontend Integration

```javascript
// Single API call gets everything
const response = await fetch('/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query: 'Solar energy in Brazil'})
});

const data = await response.json();

// Display main content
renderModules(data.modules);

// Display KG data  
renderConcepts(data.concepts);
renderRelationships(data.relationships);

// Link to visual graph
const kgUrl = data.metadata.kg_visualization_url;
```

## Error Handling

The system gracefully handles KG server issues:

```json
// If KG server unavailable:
{
  "concepts": [],
  "relationships": [],
  "metadata": {
    "kg_extraction_method": "unavailable"
  }
}
```

## Configuration

### Timeout Settings
- **Regular API**: 10 second timeout for KG fetch
- **Streaming API**: 8 second timeout for KG fetch
- **Fallback**: Empty arrays if KG server unreachable

### KG Parameters
```python
# In api_server.py and mcp_chat.py
payload = {
    "query": query,
    "depth": 2,           # Concept expansion depth
    "max_nodes": 80,      # Maximum nodes in subgraph
    "include_datasets": True,
    "include_passages": False
}
```

## Performance

### Response Times
- **KG fetch adds ~1-3 seconds** to total response time
- **Parallel processing** with main query analysis  
- **Timeout protection** prevents hanging requests

### Caching Opportunities
- KG responses could be cached by query
- Concept mappings are relatively stable
- Relationship structures change infrequently

## Troubleshooting

### KG Data Missing
```bash
# Check KG server status
curl http://localhost:8100/api/kg/stats

# Check main API logs
tail -f api.log  # Look for KG fetch errors
```

### Integration Testing
```bash
# Test KG server alone
python test_kg_direct.py

# Test integrated APIs
python test_integrated_api.py
```

## Summary

Your dev team now has **complete KG integration**:
- ✅ **Concepts and relationships** in every API response
- ✅ **Both regular and streaming** endpoints enhanced  
- ✅ **Graceful fallback** if KG server unavailable
- ✅ **No additional API calls** required

The system provides the perfect balance of **rich climate analysis** (modules) with **structured knowledge context** (concepts and relationships) in a single, consistent API interface.