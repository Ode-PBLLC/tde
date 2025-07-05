#!/usr/bin/env python3
"""
Simple API server that returns structured JSON responses with thinking processes
for front-end consumption.
"""
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, AsyncGenerator
import json
import sys
import os
import aiohttp
from kg_embed_generator import KGEmbedGenerator

# Add the mcp directory to the path
sys.path.append('mcp')
from mcp_chat import run_query_structured, run_query, run_query_streaming

app = FastAPI(title="Climate Policy Radar API", version="1.0.0")

# Initialize KG embed generator - will use environment variable or default
kg_generator = KGEmbedGenerator()

# Mount static files for serving images, GeoJSON, and other static content
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS for front-end access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    include_thinking: bool = False

class StreamQueryRequest(BaseModel):
    query: str

async def _fetch_kg_data(query: str) -> Dict[str, Any]:
    """
    Fetch concepts and relationships data from the KG visualization server.
    Returns empty data if KG server is unavailable.
    """
    kg_server_url = "http://localhost:8100/api/kg/query-subgraph"
    
    payload = {
        "query": query,
        "depth": 2,
        "max_nodes": 80,
        "include_datasets": True,
        "include_passages": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(kg_server_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "concepts": data.get("concepts", []),
                        "relationships": data.get("relationships", []),
                        "query_concepts": data.get("query_concepts", []),
                        "query_concept_labels": data.get("query_concept_labels", []),
                        "kg_extraction_method": data.get("extraction_method", "unknown")
                    }
                else:
                    print(f"KG server returned status {response.status}")
                    return _empty_kg_data()
    except asyncio.TimeoutError:
        print("KG server timeout - continuing without KG data")
        return _empty_kg_data()
    except Exception as e:
        print(f"Error fetching KG data: {e}")
        return _empty_kg_data()

def _empty_kg_data() -> Dict[str, Any]:
    """Return empty KG data structure when KG server is unavailable"""
    return {
        "concepts": [],
        "relationships": [],
        "query_concepts": [],
        "query_concept_labels": [],
        "kg_extraction_method": "unavailable"
    }

def _generate_enhanced_metadata(structured_response: Dict[str, Any], full_result: Optional[Dict[str, Any]] = None, query_text: str = "", kg_embed_info: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Generate enhanced metadata that better detects charts and visualization potential.
    """
    modules = structured_response.get("modules", [])
    
    # Standard module type detection
    has_maps = any(m.get("type") == "map" for m in modules)
    has_charts = any(m.get("type") == "chart" for m in modules)
    has_tables = any(m.get("type") in ["table", "numbered_citation_table"] for m in modules)
    
    # Enhanced chart potential detection
    chart_potential = False
    visualization_data_available = False
    
    # Check for chart data in full result if available
    if full_result:
        # Check legacy chart_data
        chart_data = full_result.get("chart_data")
        if chart_data and isinstance(chart_data, list) and len(chart_data) > 0:
            chart_potential = True
            
        # Check structured visualization_data
        viz_data = full_result.get("visualization_data")
        if viz_data and isinstance(viz_data, dict) and "data" in viz_data:
            visualization_data_available = True
            chart_potential = True
            
        # Check map_data for potential charts
        map_data = full_result.get("map_data")
        if map_data and isinstance(map_data, dict) and "data" in map_data:
            chart_potential = True
    
    # Analyze table data for chart potential
    chart_worthy_tables = 0
    for module in modules:
        if module.get("type") == "table":
            rows = module.get("rows", [])
            columns = module.get("columns", [])
            
            # Check if table has numerical data suitable for charts
            if len(rows) > 1 and len(columns) >= 2:
                # Look for numerical data patterns
                has_numeric_data = False
                if rows:
                    try:
                        # Check second column for numeric values (first is often labels)
                        for row in rows[:3]:  # Check first few rows
                            if len(row) > 1:
                                val = str(row[1]).replace(',', '').replace('%', '')
                                try:
                                    float(val)
                                    has_numeric_data = True
                                    break
                                except (ValueError, TypeError):
                                    continue
                    except (IndexError, TypeError):
                        pass
                
                if has_numeric_data:
                    chart_worthy_tables += 1
                    chart_potential = True
    
    # If we have charts OR chart potential, mark has_charts as true
    effective_has_charts = has_charts or chart_potential
    
    return {
        "modules_count": len(modules),
        "has_maps": has_maps,
        "has_charts": effective_has_charts,
        "has_tables": has_tables,
        "chart_potential": chart_potential,
        "chart_worthy_tables": chart_worthy_tables,
        "visualization_data_available": visualization_data_available,
        "module_types": list(set(m.get("type", "unknown") for m in modules)),
        "kg_visualization_url": "/kg-viz",
        "kg_query_url": f"/kg-viz?query={query_text.replace(' ', '%20')}" if query_text else "/kg-viz",
        "kg_embed_url": kg_embed_info.get("url_path") if kg_embed_info else None,
        "kg_embed_path": kg_embed_info.get("relative_path") if kg_embed_info else None,
        "kg_embed_absolute_path": kg_embed_info.get("absolute_path") if kg_embed_info else None,
        "kg_embed_filename": kg_embed_info.get("filename") if kg_embed_info else None
    }

class QueryResponse(BaseModel):
    query: str
    modules: list
    thinking_process: Optional[str] = None
    metadata: Dict[str, Any]
    concepts: list = []
    relationships: list = []

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a climate policy query and return structured JSON response.
    
    Returns:
    - modules: Array of structured content (text, charts, tables, maps with GeoJSON)
    - thinking_process: AI's step-by-step reasoning (if requested)
    - metadata: Query metadata and performance info
    """
    print(f"🔥 RECEIVED QUERY REQUEST: {request.query}")
    try:
        # Set working directory for MCP servers - use current script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Always get full result for KG generation (needed for citation registry)
        full_result = await run_query(request.query)
        structured_response = full_result.get("formatted_response", {"modules": []})
        
        if request.include_thinking:
            thinking_process = full_result.get("ai_thought_process", "")
        else:
            thinking_process = None
        
        # Fetch KG data (concepts and relationships)
        kg_data = await _fetch_kg_data(request.query)
        
        # Initialize KG variables
        kg_embed_path = None
        kg_embed_absolute_path = None
        kg_embed_url = None
        
        # Generate static KG visualization file
        try:
            # Extract citation registry from full result
            citation_registry = None
            if "citation_registry" in full_result:
                citation_registry = full_result["citation_registry"]
                print(f"📋 Citation registry found with {len(citation_registry.get('citations', {}))} citations")
            else:
                print(f"⚠️  No citation registry in full_result. Keys: {list(full_result.keys())}")
            
            print(f"🔧 Starting KG embed generation for query: {request.query}")
            
            # Generate enhanced KG embed with MCP response and citation data
            kg_embed_result = await kg_generator.generate_embed(
                request.query, 
                structured_response,  # Pass the structured response as MCP data
                citation_registry
            )
            
            # Extract path info from result
            if kg_embed_result:
                print(f"✅ KG embed generation successful! Result type: {type(kg_embed_result)}")
                if isinstance(kg_embed_result, dict):
                    kg_embed_path = kg_embed_result["relative_path"]
                    kg_embed_absolute_path = kg_embed_result["absolute_path"] 
                    kg_embed_url = kg_embed_result["url_path"]
                    print(f"📁 KG file created: {kg_embed_absolute_path}")
                else:
                    # Backward compatibility
                    kg_embed_path = kg_embed_result
                    kg_embed_absolute_path = None
                    kg_embed_url = f"/static/{kg_embed_path}"
                    print(f"📁 KG file created (legacy format): {kg_embed_path}")
            else:
                print(f"❌ KG embed generation returned None")
                kg_embed_path = None
                kg_embed_absolute_path = None
                kg_embed_url = None
        except Exception as e:
            print(f"Failed to generate KG embed: {e}")
            import traceback
            print(f"KG generation traceback: {traceback.format_exc()}")
            kg_embed_path = None
            kg_embed_absolute_path = None
            kg_embed_url = None
        
        # Create kg_embed_info dict for metadata
        kg_embed_info = None
        if kg_embed_path:
            kg_embed_info = {
                "relative_path": kg_embed_path,
                "absolute_path": kg_embed_absolute_path,
                "url_path": kg_embed_url,
                "filename": kg_embed_url.split('/')[-1] if kg_embed_url else None
            }
            print(f"📊 KG embed info created: {kg_embed_info}")
        else:
            print(f"⚠️  No KG embed path - KG embed info will be None")
        
        return QueryResponse(
            query=request.query,
            modules=structured_response.get("modules", []),
            thinking_process=thinking_process,
            metadata=_generate_enhanced_metadata(structured_response, full_result if request.include_thinking else None, request.query, kg_embed_info),
            concepts=kg_data["concepts"],
            relationships=kg_data["relationships"]
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Query processing failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"API ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Climate Policy Radar API is running"}

@app.get("/featured-queries")
async def get_featured_queries(include_cached: bool = False):
    """
    Returns curated list of featured queries with images for frontend gallery.
    
    Args:
        include_cached: If True, includes cached API responses for each query
    
    This endpoint serves as a pseudo-CMS for maintaining featured content
    without requiring database changes. Update static/featured_queries.json
    to modify the content.
    """
    try:
        # Read the featured queries JSON file
        featured_queries_path = os.path.join(os.path.dirname(__file__), "static", "featured_queries.json")
        
        if not os.path.exists(featured_queries_path):
            # Return empty response if file doesn't exist
            return {
                "featured_queries": [],
                "metadata": {
                    "total_queries": 0,
                    "categories": [],
                    "error": "Featured queries file not found"
                }
            }
        
        with open(featured_queries_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate basic structure
        if "featured_queries" not in data:
            raise ValueError("Invalid featured queries file structure")
        
        # If include_cached is True, load cached responses
        if include_cached:
            cache_dir = os.path.join(os.path.dirname(__file__), "static", "cache")
            for query in data["featured_queries"]:
                query_id = query.get("id")
                if query_id:
                    cache_file = os.path.join(cache_dir, f"{query_id}.json")
                    if os.path.exists(cache_file):
                        try:
                            with open(cache_file, 'r', encoding='utf-8') as cf:
                                cached_data = json.load(cf)
                                query["cached_response"] = cached_data.get("response")
                                query["cached_at"] = cached_data.get("cached_at")
                        except Exception as e:
                            # Log error but continue
                            query["cache_error"] = str(e)
        
        # Add timestamp for caching
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["served_at"] = datetime.now().isoformat()
        
        return data
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in featured queries file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load featured queries: {str(e)}")

@app.get("/featured-queries/{query_id}/cached")
async def get_cached_query(query_id: str):
    """
    Get cached response for a specific featured query by ID.
    
    Args:
        query_id: The ID of the featured query
        
    Returns:
        The cached API response if available, or 404 if not found
    """
    try:
        cache_file = os.path.join(os.path.dirname(__file__), "static", "cache", f"{query_id}.json")
        
        if not os.path.exists(cache_file):
            raise HTTPException(status_code=404, detail=f"No cached response found for query ID: {query_id}")
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        
        return cached_data
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid cached data format: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load cached query: {str(e)}")

@app.get("/example-response")
async def get_example_response():
    """
    Returns an example of the structured response format for front-end developers.
    """
    return {
        "query": "Show me solar facilities in Brazil",
        "modules": [
            {
                "type": "text",
                "heading": "Solar Facilities in Brazil",
                "texts": [
                    "Brazil has 2,273 solar installations with a total capacity of 26,022 MW.",
                    "The facilities range from small 0.3 MW installations to large utility-scale projects exceeding 2,500 MW."
                ]
            },
            {
                "type": "map",
                "mapType": "geojson",
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
                                "country": "Brazil",
                                "popup_title": "Large Solar Complex (Bahia)",
                                "popup_content": "Capacity: 2,511.1 MW<br>Location: Bahia State",
                                "marker_color": "#4CAF50",
                                "marker_size": 20,
                                "marker_opacity": 0.8
                            }
                        },
                        {
                            "type": "Feature", 
                            "geometry": {
                                "type": "Point",
                                "coordinates": [-43.9345, -19.9167]
                            },
                            "properties": {
                                "facility_id": "BR_FAC_002",
                                "capacity_mw": 145.8,
                                "country": "Brazil",
                                "popup_title": "Solar Facility (Minas Gerais)",
                                "popup_content": "Capacity: 145.8 MW<br>Location: Minas Gerais",
                                "marker_color": "#4CAF50",
                                "marker_size": 8,
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
                },
                "metadata": {
                    "total_facilities": 2273,
                    "total_capacity_mw": 26022.5,
                    "data_source": "TZ-SAM Q1 2025",
                    "countries": ["brazil"],
                    "feature_count": 2
                }
            },
            {
                "type": "chart",
                "chartType": "bar",
                "data": {
                    "labels": ["Brazil", "India", "South Africa", "Vietnam"],
                    "datasets": [{
                        "label": "Total Capacity (MW)",
                        "data": [26022, 79734, 6075, 13063],
                        "backgroundColor": ["#4CAF50", "#FF9800", "#F44336", "#2196F3"]
                    }]
                }
            },
            {
                "type": "table",
                "heading": "Sources and References", 
                "columns": ["#", "Source", "ID/Tool", "Type", "Method", "Description"],
                "rows": [
                    ["1", "TZ-SAM Q1 2025 Solar Facilities Database | TransitionZero | Brazil, India, South Africa, Vietnam", "GetSolarFacilitiesMapData", "Dataset", "Tool/API", "TransitionZero Solar Asset Mapper - Global solar facility locations and capacity data (2273 facilities)"],
                    ["2", "Brazilian Climate Policy Framework (CCLW.executive.4934.1571)", "passage_12345", "Policy", "Knowledge Graph", "Brazil has implemented various renewable energy policies to promote solar power development across..."],
                    ["3", "UNFCCC National Communication (UNFCCC.party.492.0)", "passage_24680", "Document", "Knowledge Graph", "Solar capacity in Brazil is expected to reach 30 GW by 2030 according to government projections..."],
                    ["4", "TZ-SAM Solar Capacity Database | TransitionZero | Global", "GetSolarCapacityByCountry", "Dataset", "Tool/API", "Solar capacity statistics and aggregations for Brazil (26,022 MW total capacity)"]
                ]
            }
        ],
        "thinking_process": "First, I called GetSolarFacilitiesByCountry for Brazil...",
        "metadata": {
            "modules_count": 4,
            "has_maps": True,
            "has_charts": True,
            "has_tables": True
        }
    }

@app.post("/thorough-response")
async def thorough_query_response(request: QueryRequest):
    """
    Returns the complete raw response from MCP servers including all thinking process,
    tool calls, intermediate data, and debug information.
    
    This endpoint surfaces EVERYTHING that the MCP servers return without filtering
    or formatting - useful for debugging and development.
    """
    try:
        # Set working directory for MCP servers - use current script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Get the complete raw response from MCP
        full_result = await run_query(request.query)
        
        # Return everything - no filtering or formatting
        return {
            "query": request.query,
            "raw_mcp_response": full_result,
            "metadata": {
                "endpoint": "thorough-response",
                "timestamp": datetime.now().isoformat(),
                "note": "This contains all raw MCP data including debug info and thinking process"
            }
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Thorough query processing failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"THOROUGH API ERROR: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/query/stream")
async def stream_query(request: StreamQueryRequest):
    """
    Stream query processing with real-time thinking traces.
    
    Returns Server-Sent Events (SSE) stream with:
    - thinking: AI reasoning steps as they happen
    - tool_call: Tool execution notifications
    - tool_result: Tool execution results
    - complete: Final structured response
    - error: Any errors that occur
    """
    print(f"🔥 RECEIVED STREAM REQUEST: {request.query}")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Set working directory for MCP servers
            script_dir = os.path.dirname(os.path.abspath(__file__))
            os.chdir(script_dir)
            
            # Process query with streaming callback
            async for event in run_query_streaming(request.query):
                # Format as Server-Sent Events
                event_data = json.dumps(event, ensure_ascii=False)
                yield f"data: {event_data}\n\n"
                
        except Exception as e:
            import traceback
            error_detail = f"Streaming query failed: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(f"STREAM API ERROR: {error_detail}")
            
            error_event = {
                "type": "error",
                "data": {
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.get("/static/maps/{filename}")
async def get_geojson(filename: str):
    """
    Generate GeoJSON data dynamically for solar facilities.
    """
    try:
        # Read the CSV file directly
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), "mcp", "solar_facilities_demo.csv")
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Solar facilities data not found")
        
        # Read facilities data
        df = pd.read_csv(csv_path)
        
        # Sort by capacity and limit to top 1000
        df = df.sort_values('capacity_mw', ascending=False).head(1000)
        
        # Convert to list of facilities
        facilities = df.to_dict('records')
        
        # Convert to GeoJSON format
        features = []
        for facility in facilities:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [facility["longitude"], facility["latitude"]]
                },
                "properties": {
                    "name": facility.get("name", f"Facility {facility.get('cluster_id', '')}"),
                    "country": facility["country"],
                    "capacity_mw": facility["capacity_mw"],
                    "technology": "Solar PV",
                    "cluster_id": facility.get("cluster_id", ""),
                    "completion_year": facility.get("completion_year", "Unknown"),
                    "marker_color": {
                        'brazil': '#4CAF50',
                        'india': '#FF9800', 
                        'south africa': '#F44336',
                        'vietnam': '#2196F3'
                    }.get(facility["country"].lower(), '#9E9E9E')
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        return geojson
        
    except Exception as e:
        print(f"Error generating GeoJSON: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating GeoJSON: {str(e)}")

# Proxy routes for KG visualization
@app.get("/kg-viz")
async def proxy_kg_visualization(query: Optional[str] = None):
    """Proxy to KG visualization server main page."""
    kg_url = f"http://localhost:8100/"
    if query:
        kg_url += f"?query={query}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(kg_url) as response:
                content = await response.text()
                # Update API endpoints in the HTML to use relative paths
                content = content.replace('"/api/kg/', '"/api/kg/')
                content = content.replace("'/api/kg/", "'/api/kg/")
                return HTMLResponse(content=content, status_code=response.status)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"KG visualization server unavailable: {str(e)}")

@app.get("/api/kg/{path:path}")
async def proxy_kg_api(path: str, request: Request):
    """Proxy API requests to KG visualization server."""
    kg_url = f"http://localhost:8100/api/kg/{path}"
    
    # Forward query parameters
    if request.query_params:
        kg_url += f"?{str(request.query_params)}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(kg_url) as response:
                content = await response.json()
                return JSONResponse(content=content, status_code=response.status)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"KG server unavailable: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099) # Changed port to 8099 to avoid VS Code conflict