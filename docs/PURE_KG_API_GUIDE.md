# Pure Knowledge Graph API

## Overview

The KG visualization is now a **pure, clean graph interface** with **all concepts and relationships data available exclusively through the API**. Perfect separation of concerns for your dev team.

## What You Get

### 🎨 **Pure Graph Visualization**
- **URL**: `http://localhost:8100`
- **Interface**: Just the interactive D3.js graph, nothing else
- **Features**: Zoom, pan, click nodes for tooltips
- **No UI clutter**: No input boxes, no overlays, no buttons

### 📊 **Rich API Data**
- **Concepts list**: All concepts with query relevance flags
- **Relationships list**: Formatted as "ConceptA -> ConceptB (Type)"
- **Graph data**: Nodes and edges for visualization
- **Query context**: Which concepts are most relevant

## API Endpoints

### POST `/api/kg/query-subgraph`
**Input:**
```json
{
  "query": "Climate policy effectiveness in Brazil",
  "depth": 2,
  "max_nodes": 50,
  "include_datasets": true,
  "include_passages": false
}
```

**Output:**
```json
{
  "nodes": [
    {"id": "Q123", "label": "Climate Policy", "kind": "Concept"},
    {"id": "Q456", "label": "Carbon Pricing", "kind": "Concept"}
  ],
  "edges": [
    {"source": "Q123", "target": "Q456", "type": "RELATED_TO"}
  ],
  "concepts": [
    {"id": "Q123", "label": "Climate Policy", "is_query_relevant": true},
    {"id": "Q456", "label": "Carbon Pricing", "is_query_relevant": false}
  ],
  "relationships": [
    {
      "source_label": "Climate Policy",
      "target_label": "Carbon Pricing",
      "relationship_type": "RELATED_TO",
      "formatted": "Climate Policy -> Carbon Pricing (RELATED_TO)"
    }
  ],
  "query_concepts": ["Q123"],
  "query_concept_labels": ["Climate Policy"],
  "total_found": 25
}
```

### POST `/api/kg/subgraph`
**Input:**
```json
{
  "node_id": "Q123",
  "depth": 1,
  "max_nodes": 30
}
```

**Output:** Same structure as query-subgraph (but `is_query_relevant` will be false for all concepts)

## Integration for Your Dev Team

### 1. **Embed the Pure Graph**
```html
<iframe 
  src="http://localhost:8100" 
  width="800" 
  height="600"
  frameborder="0">
</iframe>
```

### 2. **Fetch API Data**
```javascript
// Get concepts and relationships data
const response = await fetch('/api/kg/query-subgraph', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: "Climate policy effectiveness",
    depth: 2,
    max_nodes: 50
  })
});

const data = await response.json();

// Use the clean data in your UI
const concepts = data.concepts;
const relationships = data.relationships;
```

### 3. **Display in Your UI**
```javascript
// Concepts list component
concepts.forEach(concept => {
  const item = document.createElement('li');
  item.textContent = concept.is_query_relevant 
    ? `★ ${concept.label}` 
    : concept.label;
  conceptsList.appendChild(item);
});

// Relationships list component  
relationships.forEach(rel => {
  const item = document.createElement('li');
  item.textContent = rel.formatted; // "ConceptA -> ConceptB (Type)"
  relationshipsList.appendChild(item);
});
```

## Example Integration Architecture

```
┌─────────────────────────────────────────┐
│              Your Application           │
├─────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────────┐│
│  │   Graph     │  │    Your UI          ││
│  │  (iframe)   │  │                     ││
│  │             │  │  📝 Concepts List   ││
│  │   D3.js     │  │  🔗 Relationships   ││
│  │ Visualization│  │  🎯 Query Results   ││
│  │             │  │  📊 Statistics      ││
│  └─────────────┘  └─────────────────────┘│
├─────────────────────────────────────────┤
│              API Integration            │
│  POST /api/kg/query-subgraph            │
│  → Get concepts + relationships data    │
└─────────────────────────────────────────┘
```

## Testing

### Start Server
```bash
python kg_visualization_server.py
# Runs on http://localhost:8100
```

### Test Pure API
```bash
python test_pure_api.py
```

### Manual API Test
```bash
curl -X POST "http://localhost:8100/api/kg/query-subgraph" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Climate policy effectiveness",
    "depth": 2,
    "max_nodes": 50
  }' | jq '.concepts, .relationships'
```

## Benefits

### ✅ **Clean Separation**
- **Graph**: Pure visualization, no UI clutter
- **Data**: Structured API responses for your components
- **Integration**: Easy to embed and extend

### ✅ **Developer Friendly**  
- **Standard JSON**: Easy to parse and use
- **Formatted strings**: Ready for display ("ConceptA -> ConceptB")
- **Query relevance**: Know which concepts matter most
- **Flexible**: Use the data however you want

### ✅ **Performance**
- **Lightweight**: No unnecessary UI in the graph
- **Focused**: Only the data you need
- **Cacheable**: API responses can be cached

This approach gives your dev team maximum flexibility while keeping the graph visualization clean and focused!