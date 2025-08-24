# Knowledge Graph Visualization Testing Guide

## Overview

The interactive KG visualization system is now integrated with both the regular `/query` and streaming `/query/stream` endpoints. Both endpoints include KG visualization links in their metadata.

## Running the System

### 1. Start the KG Visualization Server

```bash
# Terminal 1 - Start KG visualization server
cd /Users/mason/Documents/GitHub/tde
python kg_visualization_server.py
```

You should see:
```
Starting KG Visualization Server...
Graph file: /Users/mason/Documents/GitHub/tde/extras/knowledge_graph.graphml
Concepts file: /Users/mason/Documents/GitHub/tde/extras/concepts.csv
Visit http://localhost:8100 to see the visualization
INFO:     Uvicorn running on http://0.0.0.0:8100
```

### 2. Start the Main API Server

```bash
# Terminal 2 - Start main API server
cd /Users/mason/Documents/GitHub/tde
python api_server.py
```

You should see:
```
INFO:     Started server process [xxxxx]
INFO:     Uvicorn running on http://0.0.0.0:8099
```

## Testing Methods

### Method 1: Direct KG Visualization Access

Visit: `http://localhost:8100`

1. **Test Query Mode**:
   - Enter: "Water stress impacts on financial sector"
   - Click "Analyze Query"
   - See relevant subgraph with highlighted nodes

2. **Test URL Parameters**:
   - Visit: `http://localhost:8100?query=renewable%20energy%20policy%20Brazil`
   - Should auto-load and analyze the query

### Method 2: Test Regular API Endpoint

```bash
# Test regular /query endpoint
curl -X POST "http://localhost:8098/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Climate policy effectiveness in Brazil",
    "include_thinking": false
  }' | jq '.metadata'
```

**Expected Response** (metadata section):
```json
{
  "modules_count": 3,
  "has_maps": true,
  "has_charts": true,
  "has_tables": true,
  "kg_visualization_url": "http://localhost:8100",
  "kg_query_url": "http://localhost:8100?query=Climate%20policy%20effectiveness%20in%20Brazil"
}
```

### Method 3: Test Streaming API Endpoint

#### Option A: Using the Test Script

```bash
# Run the automated test
python test_streaming_kg.py
```

#### Option B: Manual curl Test

```bash
# Test streaming endpoint
curl -X POST "http://localhost:8098/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "Solar energy growth in emerging markets"}' \
  --no-buffer
```

**Expected Streaming Events**:
```
data: {"type": "thinking", "data": {"message": "🔍 Checking concept database for query-related terms", "category": "search"}}

data: {"type": "thinking_complete", "data": {"message": "✅ Found relevant policy concepts", "category": "search"}}

...

data: {"type": "complete", "data": {"query": "Solar energy growth in emerging markets", "modules": [...], "metadata": {"kg_visualization_url": "http://localhost:8100", "kg_query_url": "http://localhost:8100?query=Solar%20energy%20growth%20in%20emerging%20markets"}}}
```

### Method 4: Frontend Integration Test

If you have a frontend client, the metadata now includes:

```javascript
// Example frontend usage
fetch('/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query: 'Climate adaptation policies'})
})
.then(response => response.json())
.then(data => {
  const kgUrl = data.metadata.kg_visualization_url;
  const querySpecificUrl = data.metadata.kg_query_url;
  
  // Add links to your UI
  console.log('General KG:', kgUrl);
  console.log('Query-specific KG:', querySpecificUrl);
});
```

## Expected Behavior

### ✅ What Should Work

1. **KG Visualization Server**:
   - Loads at `http://localhost:8100`
   - Query Mode accepts natural language queries
   - Displays relevant subgraphs with red highlighting
   - Auto-loads queries from URL parameters

2. **API Integration**:
   - Both `/query` and `/query/stream` include KG metadata
   - Links are properly URL-encoded
   - Query-specific URLs work when clicked

3. **Interactive Features**:
   - Zoom and pan the graph
   - Click nodes to see details
   - Expand neighborhoods by shift-clicking
   - Toggle node types (datasets, passages)

### 🔧 Troubleshooting

#### KG Server Won't Start
```bash
# Check dependencies
pip install fastapi uvicorn networkx pandas

# Check required files
ls -la extras/knowledge_graph.graphml
ls -la extras/concepts.csv
```

#### Port Conflicts
```bash
# Change port in kg_visualization_server.py (line ~317)
uvicorn.run(app, host="0.0.0.0", port=8101)  # Use different port

# Update URLs in api_server.py accordingly
```

#### No Concepts Found
- The system falls back to keyword matching if MCP tools aren't available
- Check that `extras/concepts.csv` contains the concept mappings
- Try simpler queries like "climate policy" or "renewable energy"

## Example Test Queries

### Good Test Queries
- "Water stress impacts on financial sector"
- "Renewable energy policy in Brazil"
- "Climate adaptation strategies"
- "Carbon pricing mechanisms"
- "Solar capacity in emerging markets"

### What Each Test Shows
1. **Concept Extraction**: How the system identifies relevant concepts
2. **Subgraph Construction**: Which related concepts/datasets are included
3. **Visual Highlighting**: Query-relevant nodes in red
4. **API Integration**: Links from API responses to visualizations

## Integration Benefits

### For Developers
- **Debug Tool**: See how queries map to KG concepts
- **Educational**: Understand system reasoning
- **Visual Feedback**: Immediate sense of data coverage

### For Users
- **Context Understanding**: See broader topic relationships
- **Interactive Exploration**: Discover related concepts
- **Knowledge Discovery**: Find unexpected connections

## Next Steps

After confirming the system works:

1. **Share KG URLs**: Include in API documentation
2. **Frontend Integration**: Add KG links to your UI
3. **User Training**: Show users how to explore concepts
4. **Analytics**: Track which concepts are most queried

The system provides a powerful bridge between textual API responses and visual knowledge exploration!