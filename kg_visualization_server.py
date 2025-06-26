#!/usr/bin/env python3
"""
Knowledge Graph Visualization Server
Serves graph data for interactive D3.js visualization
"""

import os
import json
import re
import sys
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import List, Dict, Optional, Any, Set
from pydantic import BaseModel
import pandas as pd
from functools import lru_cache

# Add MCP directory to path for concept extraction
sys.path.append('mcp')
try:
    from cpr_kg_server import (
        CheckConceptExists, GetSemanticallySimilarConcepts,
        GetPassagesMentioningConcept, GetConceptGraphNeighbors
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP tools not available, using fallback concept extraction")

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
GRAPHML_PATH = os.path.join(script_dir, "extras", "knowledge_graph.graphml")
CONCEPTS_PATH = os.path.join(script_dir, "extras", "concepts.csv")

app = FastAPI(title="Knowledge Graph Visualization API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load concepts for label mapping
concepts_df = pd.read_csv(CONCEPTS_PATH)
LABEL_TO_ID = concepts_df.set_index("preferred_label")["wikibase_id"].to_dict()
ID_TO_LABEL = concepts_df.set_index("wikibase_id")["preferred_label"].to_dict()

@lru_cache(maxsize=1)
def load_graph() -> nx.MultiDiGraph:
    """Load the knowledge graph from GraphML file"""
    return nx.read_graphml(GRAPHML_PATH)

def get_node_info(G: nx.MultiDiGraph, node_id: str) -> Dict[str, Any]:
    """Extract node information for visualization"""
    node_data = G.nodes[node_id]
    node_kind = node_data.get("kind", "Unknown")
    
    info = {
        "id": node_id,
        "kind": node_kind,
        "label": "",
        "description": ""
    }
    
    if node_kind == "Concept":
        info["label"] = ID_TO_LABEL.get(node_id, node_id)
        info["description"] = node_data.get("description", "")
    elif node_kind == "Dataset":
        info["label"] = node_data.get("label", node_id)
        info["description"] = node_data.get("description", "")
    elif node_kind == "Passage":
        info["label"] = f"Passage {node_id[:8]}..."
        info["description"] = node_data.get("text", "")[:200] + "..."
    else:
        info["label"] = node_id
    
    return info

def get_edge_info(edge_data: Dict) -> Dict[str, Any]:
    """Extract edge information for visualization"""
    return {
        "type": edge_data.get("type", "UNKNOWN"),
        "weight": 1.0  # Can be customized based on edge type
    }

class SubgraphRequest(BaseModel):
    node_id: str
    depth: int = 1
    max_nodes: int = 50

class PathRequest(BaseModel):
    source_id: str
    target_id: str
    max_length: int = 4

class SearchRequest(BaseModel):
    query: str
    limit: int = 20

class QuerySubgraphRequest(BaseModel):
    query: str
    depth: int = 2
    max_nodes: int = 100
    include_datasets: bool = True
    include_passages: bool = True

def extract_concepts_from_query(query: str) -> List[str]:
    """Extract potential concepts from a query string using simple heuristics"""
    # Simple keyword extraction - look for known concepts in query
    query_lower = query.lower()
    found_concepts = []
    
    # Check for exact matches in concept labels
    for label, concept_id in LABEL_TO_ID.items():
        if label.lower() in query_lower:
            found_concepts.append(concept_id)
    
    # If no exact matches, use semantic similarity
    if not found_concepts and MCP_AVAILABLE:
        try:
            # Use MCP semantic similarity
            similar_concepts = GetSemanticallySimilarConcepts(query, top_k=5)
            if similar_concepts.get("concepts"):
                found_concepts = [c["id"] for c in similar_concepts["concepts"][:3]]
        except Exception as e:
            print(f"Error in semantic concept extraction: {e}")
    
    # Fallback: find concepts with partial matches
    if not found_concepts:
        # Split query into words and find partial matches
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in query_words:
            if len(word) > 3:  # Skip short words
                for label, concept_id in LABEL_TO_ID.items():
                    if word in label.lower() and concept_id not in found_concepts:
                        found_concepts.append(concept_id)
                        if len(found_concepts) >= 5:
                            break
    
    return found_concepts

def build_query_subgraph(G: nx.MultiDiGraph, query_concepts: List[str], 
                        depth: int = 2, max_nodes: int = 100,
                        include_datasets: bool = True, include_passages: bool = True) -> Dict[str, Any]:
    """Build a subgraph focused on query-relevant concepts"""
    if not query_concepts:
        return {"nodes": [], "edges": [], "query_concepts": []}
    
    # Start with query concepts
    nodes_to_include = set(query_concepts)
    query_concept_labels = [ID_TO_LABEL.get(cid, cid) for cid in query_concepts]
    
    # Expand around each query concept
    for concept_id in query_concepts:
        if not G.has_node(concept_id):
            continue
            
        # BFS expansion
        current_level = {concept_id}
        for d in range(depth):
            if len(nodes_to_include) >= max_nodes:
                break
                
            next_level = set()
            for node in current_level:
                # Get neighbors
                neighbors = set(G.predecessors(node)) | set(G.successors(node))
                
                # Filter by node type preferences
                for neighbor in neighbors:
                    if len(nodes_to_include) >= max_nodes:
                        break
                        
                    node_data = G.nodes.get(neighbor, {})
                    node_kind = node_data.get("kind", "Unknown")
                    
                    # Always include concepts
                    if node_kind == "Concept":
                        next_level.add(neighbor)
                    # Include datasets if requested
                    elif node_kind == "Dataset" and include_datasets:
                        next_level.add(neighbor)
                    # Include passages if requested and not too many
                    elif node_kind == "Passage" and include_passages and len(nodes_to_include) < max_nodes * 0.7:
                        next_level.add(neighbor)
            
            current_level = next_level - nodes_to_include
            nodes_to_include.update(current_level)
    
    # Create subgraph
    valid_nodes = [n for n in nodes_to_include if G.has_node(n)]
    subgraph = G.subgraph(valid_nodes)
    
    # Convert to visualization format
    nodes = []
    edges = []
    
    for node in subgraph.nodes():
        node_info = get_node_info(G, node)
        nodes.append(node_info)
    
    for u, v, key, data in subgraph.edges(keys=True, data=True):
        edge_info = get_edge_info(data)
        edges.append({
            "source": u,
            "target": v,
            "type": edge_info["type"],
            "weight": edge_info["weight"]
        })
    
    # Generate concepts list (only concept nodes)
    concepts_list = []
    for node in nodes:
        if node["kind"] == "Concept":
            concepts_list.append({
                "id": node["id"],
                "label": node["label"]
            })
    
    # Generate relationships list (ConceptA -> ConceptB format)
    relationships_list = []
    for edge in edges:
        source_node = next((n for n in nodes if n["id"] == edge["source"]), None)
        target_node = next((n for n in nodes if n["id"] == edge["target"]), None)
        
        # Only include concept-to-concept relationships
        if (source_node and target_node and 
            source_node["kind"] == "Concept" and target_node["kind"] == "Concept"):
            
            source_label = ID_TO_LABEL.get(edge["source"], edge["source"])
            target_label = ID_TO_LABEL.get(edge["target"], edge["target"])
            relationship_type = edge["type"]
            
            relationships_list.append({
                "source_id": edge["source"],
                "target_id": edge["target"],
                "source_label": source_label,
                "target_label": target_label,
                "relationship_type": relationship_type,
                "formatted": f"{source_label} -> {target_label} ({relationship_type})"
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "query_concepts": query_concepts,
        "query_concept_labels": query_concept_labels,
        "total_found": len(nodes),
        "concepts": concepts_list,
        "relationships": relationships_list
    }

@app.get("/")
async def read_root():
    """Serve the main visualization page"""
    with open("static/kg_visualization.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/kg/stats")
async def get_graph_stats():
    """Get basic statistics about the knowledge graph"""
    G = load_graph()
    
    node_counts = {}
    for node, data in G.nodes(data=True):
        kind = data.get("kind", "Unknown")
        node_counts[kind] = node_counts.get(kind, 0) + 1
    
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_counts": node_counts,
        "edge_types": list(set(data.get("type", "UNKNOWN") for _, _, data in G.edges(data=True)))
    }

@app.post("/api/kg/subgraph")
async def get_subgraph(request: SubgraphRequest):
    """Get a subgraph around a specific node"""
    G = load_graph()
    
    if not G.has_node(request.node_id):
        raise HTTPException(status_code=404, detail=f"Node {request.node_id} not found")
    
    # Use BFS to get nodes within specified depth
    nodes_to_include = set()
    current_level = {request.node_id}
    
    for _ in range(request.depth + 1):
        nodes_to_include.update(current_level)
        if len(nodes_to_include) >= request.max_nodes:
            break
        
        next_level = set()
        for node in current_level:
            # Get both in and out neighbors
            next_level.update(G.predecessors(node))
            next_level.update(G.successors(node))
        
        current_level = next_level - nodes_to_include
    
    # Create subgraph
    subgraph = G.subgraph(list(nodes_to_include)[:request.max_nodes])
    
    # Convert to visualization format
    nodes = []
    edges = []
    
    for node in subgraph.nodes():
        nodes.append(get_node_info(G, node))
    
    for u, v, key, data in subgraph.edges(keys=True, data=True):
        edge_info = get_edge_info(data)
        edges.append({
            "source": u,
            "target": v,
            "type": edge_info["type"],
            "weight": edge_info["weight"]
        })
    
    # Generate concepts list (only concept nodes)
    concepts_list = []
    for node in subgraph.nodes():
        node_data = G.nodes[node]
        if node_data.get("kind") == "Concept":
            concepts_list.append({
                "id": node,
                "label": ID_TO_LABEL.get(node, node)
            })
    
    # Generate relationships list (ConceptA -> ConceptB format)
    relationships_list = []
    for u, v, key, data in subgraph.edges(keys=True, data=True):
        source_node_data = G.nodes.get(u, {})
        target_node_data = G.nodes.get(v, {})
        
        # Only include concept-to-concept relationships
        if (source_node_data.get("kind") == "Concept" and 
            target_node_data.get("kind") == "Concept"):
            
            source_label = ID_TO_LABEL.get(u, u)
            target_label = ID_TO_LABEL.get(v, v)
            relationship_type = data.get("type", "UNKNOWN")
            
            relationships_list.append({
                "source_id": u,
                "target_id": v,
                "source_label": source_label,
                "target_label": target_label,
                "relationship_type": relationship_type,
                "formatted": f"{source_label} -> {target_label} ({relationship_type})"
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "center_node": request.node_id,
        "concepts": concepts_list,
        "relationships": relationships_list
    }

@app.post("/api/kg/search")
async def search_concepts(request: SearchRequest):
    """Search for concepts by label"""
    query_lower = request.query.lower()
    results = []
    
    # Search in concept labels
    for label, concept_id in LABEL_TO_ID.items():
        if query_lower in label.lower():
            results.append({
                "id": concept_id,
                "label": label,
                "kind": "Concept",
                "score": 1.0 if query_lower == label.lower() else 0.5
            })
    
    # Sort by score and limit
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return {"results": results[:request.limit]}

@app.post("/api/kg/path")
async def find_path(request: PathRequest):
    """Find shortest path between two nodes"""
    G = load_graph()
    
    if not G.has_node(request.source_id):
        raise HTTPException(status_code=404, detail=f"Source node {request.source_id} not found")
    if not G.has_node(request.target_id):
        raise HTTPException(status_code=404, detail=f"Target node {request.target_id} not found")
    
    # Convert to undirected for path finding
    UG = G.to_undirected()
    
    try:
        path = nx.shortest_path(UG, request.source_id, request.target_id, 
                               weight=None)[:request.max_length + 1]
        
        # Get path with edges
        path_nodes = []
        path_edges = []
        
        for node in path:
            path_nodes.append(get_node_info(G, node))
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Get edge data (check both directions)
            edge_data = G.get_edge_data(u, v, default={})
            if not edge_data:
                edge_data = G.get_edge_data(v, u, default={})
            
            if edge_data:
                # Get first edge if multiple exist
                first_edge = next(iter(edge_data.values()))
                path_edges.append({
                    "source": u,
                    "target": v,
                    "type": first_edge.get("type", "UNKNOWN")
                })
        
        return {
            "path_exists": True,
            "path_nodes": path_nodes,
            "path_edges": path_edges,
            "length": len(path) - 1
        }
        
    except nx.NetworkXNoPath:
        return {
            "path_exists": False,
            "message": f"No path found between {request.source_id} and {request.target_id}"
        }

@app.get("/api/kg/node/{node_id}")
async def get_node_details(node_id: str):
    """Get detailed information about a specific node"""
    G = load_graph()
    
    if not G.has_node(node_id):
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    node_info = get_node_info(G, node_id)
    node_data = G.nodes[node_id]
    
    # Add additional details based on node type
    if node_data.get("kind") == "Concept":
        # Get related concepts
        neighbors = []
        for target in G.successors(node_id):
            edge_data = next(iter(G.get_edge_data(node_id, target).values()))
            if edge_data.get("type") in ["RELATED_TO", "HAS_SUBCONCEPT"]:
                neighbors.append({
                    "id": target,
                    "label": ID_TO_LABEL.get(target, target),
                    "relationship": edge_data.get("type")
                })
        node_info["related_concepts"] = neighbors
    
    elif node_data.get("kind") == "Dataset":
        node_info["server_name"] = node_data.get("server_name", "")
        node_info["countries"] = node_data.get("countries", [])
        node_info["total_facilities"] = node_data.get("total_facilities", 0)
        node_info["total_capacity_gw"] = node_data.get("total_capacity_gw", 0)
    
    # Get degree information
    node_info["in_degree"] = G.in_degree(node_id)
    node_info["out_degree"] = G.out_degree(node_id)
    
    return node_info

@app.get("/api/kg/top_concepts")
async def get_top_concepts(limit: int = 20):
    """Get top-level concepts to start visualization"""
    G = load_graph()
    
    # Find concepts with high degree centrality
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "Concept"]
    subgraph = G.subgraph(concept_nodes)
    
    # Calculate degree centrality
    centrality = nx.degree_centrality(subgraph)
    
    # Sort by centrality
    top_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    results = []
    for concept_id, score in top_concepts:
        results.append({
            "id": concept_id,
            "label": ID_TO_LABEL.get(concept_id, concept_id),
            "centrality": score,
            "degree": G.degree(concept_id)
        })
    
    return {"concepts": results}

@app.post("/api/kg/query-subgraph")
async def get_query_subgraph(request: QuerySubgraphRequest):
    """Get a subgraph relevant to a specific query"""
    G = load_graph()
    
    # Extract concepts from query
    query_concepts = extract_concepts_from_query(request.query)
    
    if not query_concepts:
        raise HTTPException(
            status_code=404, 
            detail=f"No relevant concepts found for query: '{request.query}'"
        )
    
    # Build query-specific subgraph
    result = build_query_subgraph(
        G, query_concepts, 
        depth=request.depth,
        max_nodes=request.max_nodes,
        include_datasets=request.include_datasets,
        include_passages=request.include_passages
    )
    
    # Add query information
    result["query"] = request.query
    result["extraction_method"] = "semantic" if MCP_AVAILABLE else "keyword_matching"
    
    return result

if __name__ == "__main__":
    import uvicorn
    
    # Create static directory if it doesn't exist
    os.makedirs("static/js", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    
    print(f"Starting KG Visualization Server...")
    print(f"Graph file: {GRAPHML_PATH}")
    print(f"Concepts file: {CONCEPTS_PATH}")
    print(f"Visit http://localhost:8100 to see the visualization")
    
    uvicorn.run(app, host="0.0.0.0", port=8100)