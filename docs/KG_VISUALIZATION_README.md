# Knowledge Graph Interactive Visualization

This interactive visualization allows you to explore the Climate Policy Radar knowledge graph in an intuitive, visual way.

## Features

- **Interactive Graph Exploration**: Click and drag nodes, zoom in/out, and pan around the graph
- **Concept Search**: Search for specific concepts by name
- **Path Finding**: Find shortest paths between two concepts
- **Node Details**: Click on nodes to see detailed information
- **Visual Encoding**: Different shapes and colors for different node types:
  - 🔵 **Circles**: Concepts
  - 🟧 **Squares**: Datasets
  - 🟩 **Triangles**: Passages
- **Expandable Graph**: Shift+click on nodes to expand and see their neighbors
- **Filtering**: Toggle passage visibility to focus on concepts and datasets

## Getting Started

1. **Start the server**:
   ```bash
   python kg_visualization_server.py
   ```

2. **Open the visualization**:
   Navigate to `http://localhost:8100` in your web browser

3. **Test the API** (optional):
   ```bash
   python test_kg_visualization.py
   ```

## How to Use

### Basic Navigation
- **Pan**: Click and drag on the background
- **Zoom**: Use mouse wheel or pinch gesture
- **Move nodes**: Click and drag individual nodes

### Exploring Concepts
1. **Search**: Type a concept name in the search box and click "Search"
2. **Expand**: Shift+click on any node to load its neighbors
3. **Details**: Click on a node to see its details in the sidebar

### Finding Relationships
1. Enter source and target concept names in the "Find Path" section
2. Click "Find Path" to highlight the shortest path between them
3. Path nodes and edges will be highlighted in green

### Controls
- **Reset View**: Clear all highlights and reset zoom
- **Toggle Labels**: Show/hide node labels
- **Show Passages**: Toggle visibility of passage nodes

## API Endpoints

The visualization is powered by a FastAPI backend with the following endpoints:

- `GET /api/kg/stats` - Get graph statistics
- `GET /api/kg/top_concepts` - Get most connected concepts
- `POST /api/kg/search` - Search for concepts by name
- `POST /api/kg/subgraph` - Get subgraph around a node
- `POST /api/kg/path` - Find path between two nodes
- `GET /api/kg/node/{id}` - Get detailed node information

## Technical Details

- **Backend**: FastAPI server reading from GraphML knowledge graph
- **Frontend**: D3.js force-directed graph visualization
- **Data**: NetworkX MultiDiGraph with concepts, passages, and datasets

## Customization

You can customize the visualization by editing:
- `static/css/kg_graph.css` - Visual styling
- `static/js/kg_graph.js` - Interaction behavior
- `kg_visualization_server.py` - API logic and data processing

## Troubleshooting

- **Server won't start**: Make sure the GraphML file exists at `extras/knowledge_graph.graphml`
- **No concepts found**: Verify `extras/concepts.csv` is present and properly formatted
- **Visualization is slow**: Try reducing the depth parameter when expanding nodes or hiding passages