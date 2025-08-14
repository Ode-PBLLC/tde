#!/usr/bin/env python3
"""
ODE MCP Generic API Server
Domain-agnostic FastAPI server for MCP orchestration with advanced citation system.
"""
import asyncio
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import sys

# Add the mcp directory to the path
sys.path.append('mcp')
from mcp_chat import run_query_structured, run_query, run_query_streaming, get_global_client, cleanup_global_client
from citation_validation import validate_citation_coverage_sync
from pathing import get_path

# Get API configuration from environment
API_TITLE = os.getenv("API_TITLE", "ODE MCP Generic API")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
API_DESCRIPTION = os.getenv("API_DESCRIPTION", "Domain-agnostic MCP API with advanced citation system")

# Generic FastAPI app - customize title/version for your domain
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

@app.on_event("startup")
async def startup_event():
    """Initialize the global MCP client singleton on server startup."""
    try:
        print("🚀 Starting ODE MCP Generic API...")
        
        # Validate citation configuration first
        print("📋 Validating citation configuration...")
        citation_result = validate_citation_coverage_sync()
        if citation_result.valid:
            print(f"✅ Citation validation passed: {len(citation_result.tools_validated)} tools with valid citations")
        else:
            print(f"⚠️ Citation validation issues: {len(citation_result.errors)} errors, {len(citation_result.warnings)} warnings")
            for error in citation_result.errors:
                print(f"  ❌ {error}")
            for warning in citation_result.warnings[:3]:  # Show first 3 warnings
                print(f"  ⚠️ {warning}")
            if len(citation_result.warnings) > 3:
                print(f"  ... and {len(citation_result.warnings) - 3} more warnings")
        
        print("Warming up global MCP client...")
        await get_global_client()
        print("✅ Global MCP client warmed up successfully")
        print(f"🌐 API server ready at http://0.0.0.0:{os.getenv('API_PORT', 8098)}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to warm up global MCP client: {e}")
        print("Individual requests will handle fallback initialization")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up the global MCP client on server shutdown."""
    try:
        print("🔄 Shutting down ODE MCP Generic API...")
        print("Cleaning up global MCP client...")
        await cleanup_global_client()
        print("✅ Global MCP client cleanup completed")
    except Exception as e:
        print(f"⚠️ Warning: Error during MCP client cleanup: {e}")

# Mount static files for serving domain-specific content
static_dir = get_path("static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print(f"📁 Static files mounted at /static (from {static_dir})")

# Enable CORS for front-end access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    include_thinking: bool = False

class StreamQueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    modules: list
    metadata: dict
    processing_time: float
    thinking: Optional[str] = None

# Core API Endpoints
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process queries using MCP orchestration with citation system.
    Returns structured response with modules and metadata.
    """
    try:
        start_time = time.time()
        
        # Run query through MCP orchestration
        result = await run_query_structured(request.query)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=result.get("response", ""),
            modules=result.get("modules", []),
            metadata={
                **result.get("metadata", {}),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0"
            },
            processing_time=processing_time,
            thinking=result.get("thinking") if request.include_thinking else None
        )
        
    except Exception as e:
        print(f"❌ Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/query/stream")
async def stream_query_endpoint(request: StreamQueryRequest):
    """
    Stream query processing with real-time updates.
    Returns server-sent events with progressive results.
    """
    async def generate_stream():
        try:
            async for chunk in run_query_streaming(request.query):
                yield f"data: {json.dumps(chunk)}\\n\\n"
        except Exception as e:
            error_chunk = {
                "type": "error",
                "content": f"Stream processing failed: {str(e)}"
            }
            yield f"data: {json.dumps(error_chunk)}\\n\\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    service_name = os.getenv("SERVICE_NAME", API_TITLE)
    try:
        # Test MCP client connectivity
        client = await get_global_client()
        if client and hasattr(client, 'health_manager'):
            # Get detailed server health information
            server_health = client.health_manager.get_all_server_health()
            client_status = "healthy"
            
            # Determine overall status based on server health
            overall_status = "healthy"
            down_servers = []
            degraded_servers = []
            
            for server_name, health_info in server_health.items():
                if health_info["status"] == "down":
                    down_servers.append(server_name)
                    overall_status = "degraded"
                elif health_info["status"] in ["degraded", "recovering"]:
                    degraded_servers.append(server_name)
                    if overall_status == "healthy":
                        overall_status = "degraded"
            
            if len(down_servers) == len(server_health):
                overall_status = "unhealthy"
            
        else:
            client_status = "healthy" if client else "unavailable"
            server_health = {}
            overall_status = "healthy" if client else "degraded"
            down_servers = []
            degraded_servers = []
            
    except Exception as e:
        client_status = "error"
        server_health = {}
        overall_status = "unhealthy"
        down_servers = []
        degraded_servers = []
    
    response = {
        "status": overall_status,
        "message": f"{service_name} is {overall_status}",
        "timestamp": datetime.now().isoformat(),
        "mcp_client": client_status,
        "api_version": API_VERSION,
        "server_health": server_health
    }
    
    # Add summary for monitoring systems
    if server_health:
        response["summary"] = {
            "total_servers": len(server_health),
            "healthy_servers": len([s for s in server_health.values() if s["status"] == "healthy"]),
            "degraded_servers": len(degraded_servers),
            "down_servers": len(down_servers)
        }
        
        if down_servers:
            response["down_servers"] = down_servers
        if degraded_servers:
            response["degraded_servers"] = degraded_servers
    
    return response

@app.get("/featured-queries")
async def get_featured_queries():
    """
    Return domain-specific featured queries.
    Load from config/featured_queries.json or static/featured_queries.json
    """
    try:
        # Try config first, fallback to static
        for path in [
            get_path("config", "featured_queries.json"), 
            get_path("static", "featured_queries.json")
        ]:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    queries = json.load(f)
                    print(f"📋 Loaded {len(queries.get('queries', []))} featured queries from {path}")
                    return queries
        
        # Default empty response if no featured queries configured
        return {
            "queries": [],
            "message": "No featured queries configured for this domain",
            "setup_instructions": "Add queries to config/featured_queries.json to customize this endpoint"
        }
    except Exception as e:
        print(f"❌ Failed to load featured queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load featured queries: {str(e)}")

@app.get("/config/info")
async def get_config_info():
    """
    Return information about the current configuration.
    Useful for debugging and setup validation.
    """
    config_files = {
        "servers": os.path.exists("config/servers.json"),
        "citation_sources": os.path.exists("config/citation_sources.json"),
        "featured_queries": os.path.exists("config/featured_queries.json")
    }
    
    return {
        "domain": os.getenv("DOMAIN_NAME", "generic"),
        "organization": os.getenv("ORGANIZATION_NAME", "Unknown"),
        "config_files_present": config_files,
        "static_directory_exists": os.path.exists("static"),
        "data_directory_exists": os.path.exists("data"),
        "api_version": "1.0.0"
    }

# Domain-specific endpoints can be added here
# Example:
# @app.get("/domain-specific-endpoint")
# async def domain_endpoint():
#     """Add domain-specific endpoints as needed."""
#     return {"message": "Add domain-specific endpoints as needed"}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8098))
    
    print(f"🚀 Starting ODE MCP Generic API on {host}:{port}")
    print("📋 Available endpoints:")
    print("  POST /query - Process queries with full response")
    print("  POST /query/stream - Process queries with streaming")
    print("  GET /health - Health check")
    print("  GET /featured-queries - Domain featured queries")
    print("  GET /config/info - Configuration information")
    print("")
    
    uvicorn.run(app, host=host, port=port)