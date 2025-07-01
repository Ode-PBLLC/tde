#!/usr/bin/env python3
"""
Generate static KG visualization files for queries.
Creates embeddable HTML files with D3.js visualizations in the static directory.
"""
import os
import json
import hashlib
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path


class KGEmbedGenerator:
    """Generate static KG visualization files for queries."""
    
    def __init__(self, kg_server_url: str = "http://localhost:8100", static_dir: str = "static", base_url: str = "https://api.transitiondigital.org"):
        self.kg_server_url = kg_server_url
        self.static_dir = Path(static_dir)
        self.kg_dir = self.static_dir / "kg"
        self.base_url = base_url.rstrip('/')  # Remove trailing slash
        
        # Ensure kg directory exists
        self.kg_dir.mkdir(exist_ok=True)
        
        # Template for embeddable visualization
        self.html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph: {query}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; padding: 10px; font-family: Arial, sans-serif; }}
        .container {{ width: 100%; height: 500px; border: 1px solid #ddd; position: relative; }}
        .graph {{ width: 100%; height: 100%; }}
        .tooltip {{ 
            position: absolute; 
            background: rgba(0,0,0,0.8); 
            color: white; 
            padding: 8px; 
            border-radius: 4px; 
            font-size: 12px; 
            pointer-events: none; 
            opacity: 0;
        }}
        .query-title {{ 
            font-size: 14px; 
            font-weight: bold; 
            margin-bottom: 10px; 
            color: #333;
        }}
        .node {{ cursor: pointer; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; stroke-width: 1.5px; }}
        .node-label {{ font-size: 10px; fill: #333; text-anchor: middle; }}
    </style>
</head>
<body>
    <div class="query-title">Knowledge Graph: {query}</div>
    <div class="container">
        <svg class="graph" id="kg-graph"></svg>
        <div class="tooltip" id="tooltip"></div>
    </div>
    
    <script>
        // Embedded KG data
        const kgData = {kg_data_json};
        
        // Initialize visualization
        const width = 800;
        const height = 500;
        
        const svg = d3.select("#kg-graph")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", `0 0 ${{width}} ${{height}}`);
            
        const g = svg.append("g");
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        svg.call(zoom);
        
        // Process data
        const nodes = kgData.nodes || [];
        const links = kgData.links || [];
        
        // Color mapping
        const nodeColors = {{
            'Concept': '#1f77b4',
            'Dataset': '#ff7f0e', 
            'Passage': '#2ca02c',
            'Document': '#d62728',
            'Unknown': '#7f7f7f'
        }};
        
        // Create simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(60))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link");
        
        // Create nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => Math.max(5, Math.min(15, (d.importance || 1) * 8)))
            .attr("fill", d => nodeColors[d.type] || nodeColors['Unknown'])
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // Add labels
        const label = g.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .attr("dy", -18)
            .text(d => d.label || d.id);
        
        // Tooltip
        const tooltip = d3.select("#tooltip");
        
        node.on("mouseover", function(event, d) {{
            tooltip.transition().duration(200).style("opacity", .9);
            tooltip.html(`<strong>${{d.label || d.id}}</strong><br/>Type: ${{d.type}}<br/>Importance: ${{d.importance || 'N/A'}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        }})
        .on("mouseout", function(d) {{
            tooltip.transition().duration(500).style("opacity", 0);
        }});
        
        // Update positions on simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>"""

    def _generate_query_hash(self, query: str) -> str:
        """Generate a short hash for the query to use as filename."""
        return hashlib.md5(query.encode()).hexdigest()[:12]
    
    async def fetch_kg_data(self, query: str, mcp_response: Optional[Dict[str, Any]] = None, citation_registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch KG data from the KG visualization server, optionally enhanced with MCP response data."""
        
        # Try enhanced endpoint first if MCP response data is available
        if mcp_response:
            endpoint = f"{self.kg_server_url}/api/kg/query-subgraph-with-mcp"
            payload = {
                "query": query,
                "mcp_response": mcp_response,
                "citation_registry": citation_registry,
                "depth": 2,
                "max_nodes": 50,  # Smaller for embedded view
                "include_datasets": True,
                "include_passages": False
            }
            
            print(f"📡 Fetching KG data from MCP endpoint: {endpoint}")
            print(f"📡 Payload: {json.dumps(payload, indent=2)}")
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(f"📡 KG server response keys: {list(data.keys())}")
                            if "nodes" in data:
                                print(f"📡 Nodes received: {len(data['nodes'])}")
                            if "links" in data:
                                print(f"📡 Links received: {len(data['links'])}")
                            if "edges" in data:
                                print(f"📡 Edges received: {len(data['edges'])}")
                            return data
                        else:
                            print(f"⚠️  MCP endpoint failed with status {response.status}, falling back to regular endpoint")
                            error_text = await response.text()
                            print(f"⚠️  MCP endpoint error: {error_text}")
            except Exception as e:
                print(f"⚠️  MCP endpoint exception: {e}, falling back to regular endpoint")
        
        # Fallback to original endpoint (either no MCP data or MCP endpoint failed)
        endpoint = f"{self.kg_server_url}/api/kg/query-subgraph"
        payload = {
            "query": query,
            "depth": 2,
            "max_nodes": 50,  # Smaller for embedded view
            "include_datasets": True,
            "include_passages": False
        }
        
        print(f"📡 Fetching KG data from regular endpoint: {endpoint}")
        print(f"📡 Payload: {json.dumps(payload, indent=2)}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"📡 KG server response keys: {list(data.keys())}")
                        if "nodes" in data:
                            print(f"📡 Nodes received: {len(data['nodes'])}")
                        if "links" in data:
                            print(f"📡 Links received: {len(data['links'])}")
                        if "edges" in data:
                            print(f"📡 Edges received: {len(data['edges'])}")
                        return data
                    else:
                        print(f"❌ KG server returned status {response.status}")
                        error_text = await response.text()
                        print(f"❌ KG server error: {error_text}")
                        return {"nodes": [], "links": [], "concepts": [], "relationships": []}
        except Exception as e:
            print(f"❌ Error fetching KG data: {e}")
            import traceback
            print(f"❌ KG fetch traceback: {traceback.format_exc()}")
            return {"nodes": [], "links": [], "concepts": [], "relationships": []}
    
    async def generate_embed(self, query: str, mcp_response: Optional[Dict[str, Any]] = None, citation_registry: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
        """
        Generate a static KG visualization file for a query, optionally enhanced with MCP response data.
        Returns the relative path to the generated file.
        """
        # Get KG data (enhanced with MCP response if available)
        kg_data = await self.fetch_kg_data(query, mcp_response, citation_registry)
        
        # Handle both "links" and "edges" formats from KG server
        if "edges" in kg_data and "links" not in kg_data:
            kg_data["links"] = kg_data["edges"]
        
        # Check for valid data
        nodes = kg_data.get("nodes", [])
        links = kg_data.get("links", [])
        
        if not nodes and not links:
            print(f"No KG nodes or links found for query: {query}")
            print(f"KG server response keys: {list(kg_data.keys())}")
            return None
        
        print(f"📊 KG data found: {len(nodes)} nodes, {len(links)} links")
        
        # Generate filename
        query_hash = self._generate_query_hash(query)
        filename = f"kg_{query_hash}.html"
        filepath = self.kg_dir / filename
        
        # Generate HTML content
        html_content = self.html_template.format(
            query=query.replace('"', '&quot;'),
            kg_data_json=json.dumps(kg_data, indent=2)
        )
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Also save just the data as JSON for API access
        json_filename = f"kg_{query_hash}.json"
        json_filepath = self.kg_dir / json_filename
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "query": query,
                "kg_data": kg_data,
                "generated_at": str(asyncio.get_event_loop().time())
            }, f, indent=2)
        
        # Return dict with both relative and absolute paths
        return {
            "relative_path": f"kg/{filename}",
            "absolute_path": str(filepath.absolute()),
            "filename": filename,
            "url_path": f"{self.base_url}/static/kg/{filename}"
        }
    
    def get_embed_url(self, query: str, base_url: str = "") -> str:
        """Get the URL for a query's KG embed file."""
        query_hash = self._generate_query_hash(query)
        filename = f"kg_{query_hash}.html"
        return f"{base_url}/static/kg/{filename}"
    
    def embed_exists(self, query: str) -> bool:
        """Check if an embed file already exists for this query."""
        query_hash = self._generate_query_hash(query)
        filename = f"kg_{query_hash}.html"
        return (self.kg_dir / filename).exists()


async def main():
    """Test the KG embed generator."""
    generator = KGEmbedGenerator()
    
    test_queries = [
        "What is climate change?",
        "Renewable energy policies in Brazil",
        "Carbon pricing mechanisms"
    ]
    
    print("Generating KG embeds...")
    for query in test_queries:
        print(f"Processing: {query}")
        embed_path = await generator.generate_embed(query)
        if embed_path:
            print(f"  ✅ Generated: {embed_path}")
        else:
            print(f"  ❌ Failed to generate embed")
    
    print(f"\nFiles generated in: {generator.kg_dir}")


if __name__ == "__main__":
    asyncio.run(main())