#!/usr/bin/env python3
"""
Simple API server that returns structured JSON responses with thinking processes
for front-end consumption.
"""
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, AsyncGenerator
import json
import sys
import os

# Add the mcp directory to the path
sys.path.append('mcp')
from mcp_chat import run_query_structured, run_query, run_query_streaming

app = FastAPI(title="Climate Policy Radar API", version="1.0.0")

# Mount static files for GeoJSON serving
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

class QueryResponse(BaseModel):
    query: str
    modules: list
    thinking_process: Optional[str] = None
    metadata: Dict[str, Any]

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a climate policy query and return structured JSON response.
    
    Returns:
    - modules: Array of structured content (text, charts, tables, maps with GeoJSON)
    - thinking_process: AI's step-by-step reasoning (if requested)
    - metadata: Query metadata and performance info
    """
    try:
        # Set working directory for MCP servers - use current script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        if request.include_thinking:
            # Get full response with thinking process
            full_result = await run_query(request.query)
            structured_response = full_result.get("formatted_response", {"modules": []})
            thinking_process = full_result.get("ai_thought_process", "")
        else:
            # Get just structured response
            structured_response = await run_query_structured(request.query)
            thinking_process = None
        
        return QueryResponse(
            query=request.query,
            modules=structured_response.get("modules", []),
            thinking_process=thinking_process,
            metadata={
                "modules_count": len(structured_response.get("modules", [])),
                "has_maps": any(m.get("type") == "map" for m in structured_response.get("modules", [])),
                "has_charts": any(m.get("type") == "chart" for m in structured_response.get("modules", [])),
                "has_tables": any(m.get("type") == "table" for m in structured_response.get("modules", []))
            }
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8099, reload=True) # TODO change back to reload = False and app as first input