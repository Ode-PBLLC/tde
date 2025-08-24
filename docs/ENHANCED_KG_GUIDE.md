# Enhanced Knowledge Graph Visualization

## Overview

The KG visualization has been enhanced with a clean, full-screen interface that focuses on the graph itself plus structured data lists. The sidebar has been removed in favor of floating overlays that show:

1. **Interactive Graph** - Full-screen, zoomable knowledge graph
2. **Concepts List** - All concepts in the current view 
3. **Relationships List** - Concept-to-concept relationships in "ConceptA -> ConceptB" format

## Key Features

### 🎯 Clean Interface
- **Full-screen graph** with no sidebar clutter
- **Floating query input** overlay (top-left)
- **Concepts list** overlay (top-right)  
- **Relationships list** overlay (bottom-right)

### 📊 Enhanced API Responses
All subgraph endpoints now return:
```json
{
  "nodes": [...],
  "edges": [...],
  "concepts": [
    {
      "id": "Q123",
      "label": "Climate Policy",
      "is_query_relevant": true
    }
  ],
  "relationships": [
    {
      "source_label": "Climate Policy",
      "target_label": "Carbon Pricing", 
      "relationship_type": "RELATED_TO",
      "formatted": "Climate Policy -> Carbon Pricing (RELATED_TO)"
    }
  ]
}
```

### ⭐ Query Relevance Highlighting
- Query-relevant concepts marked with ★ in the list
- Red highlighting on the graph for query-relevant nodes
- Concepts sorted with query-relevant items first

## Running the Enhanced System

### 1. Start the KG Visualization Server
```bash
cd /Users/mason/Documents/GitHub/tde
python kg_visualization_server.py
```

### 2. Test the Enhanced Features

#### Direct Browser Access
```
http://localhost:8100
```

#### With Query Parameter
```
http://localhost:8100?query=water%20stress%20financial%20sector
```

#### API Testing
```bash
# Test enhanced query endpoint
python test_enhanced_kg.py
```

## Interface Usage

### Query Input
- **Location**: Floating overlay (top-left)
- **Usage**: Enter natural language queries
- **Keyboard**: Ctrl+Enter to analyze

### Concepts List  
- **Location**: Floating overlay (top-right)
- **Content**: All concept nodes in current view
- **Sorting**: Query-relevant first (★), then alphabetical
- **Example**:
  ```
  ★ Climate Policy
  ★ Water Stress  
    Carbon Pricing
    Environmental Risk
    Financial Regulation
  ```

### Relationships List
- **Location**: Floating overlay (bottom-right) 
- **Format**: "ConceptA -> ConceptB (RelationType)"
- **Content**: Only concept-to-concept relationships
- **Example**:
  ```
  Climate Policy -> Carbon Pricing (RELATED_TO)
  Climate Policy -> Environmental Risk (HAS_SUBCONCEPT)
  Water Stress -> Financial Risk (MENTIONS)
  ```

## API Endpoints Enhanced

### `/api/kg/query-subgraph` 
- **Input**: Natural language query
- **Output**: Nodes, edges, concepts list, relationships list
- **New Fields**: `concepts[]`, `relationships[]`

### `/api/kg/subgraph`
- **Input**: Node ID for expansion  
- **Output**: Subgraph + concepts/relationships lists
- **New Fields**: `concepts[]`, `relationships[]`

## Example API Response

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
      "source_id": "Q123",
      "target_id": "Q456", 
      "source_label": "Climate Policy",
      "target_label": "Carbon Pricing",
      "relationship_type": "RELATED_TO",
      "formatted": "Climate Policy -> Carbon Pricing (RELATED_TO)"
    }
  ],
  "query": "climate policy effectiveness",
  "total_found": 25
}
```

## Benefits

### For Users
- **Clean Focus**: Graph is the main focus without UI clutter
- **Quick Reference**: Concepts and relationships always visible
- **Easy Navigation**: Query-relevant items clearly marked
- **Full Context**: See all relationships at a glance

### For Developers  
- **Structured Data**: Concepts and relationships as separate arrays
- **Machine Readable**: Formatted strings perfect for display
- **Query Context**: Know which concepts are most relevant
- **Integration Ready**: Lists perfect for other UI components

## Keyboard Shortcuts

- **Ctrl+Enter**: Analyze query
- **Mouse Wheel**: Zoom graph
- **Click+Drag**: Pan graph  
- **Click Node**: Show details in tooltip

## Customization

### Overlay Positioning
Edit `static/css/kg_graph.css`:
```css
#conceptsList {
    top: 20px;     /* Distance from top */
    right: 20px;   /* Distance from right */
}
```

### Relationship Format
Edit `kg_visualization_server.py`:
```python
"formatted": f"{source_label} -> {target_label} ({relationship_type})"
```

The enhanced interface provides a much cleaner, more focused experience while giving you exactly the structured data you need!