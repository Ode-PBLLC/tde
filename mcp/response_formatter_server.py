import json
from fastmcp import FastMCP
from typing import List, Dict, Any, Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
from datetime import datetime
import hashlib

mcp = FastMCP("response-formatter-server")

metadata = {
    "Name": "Response Formatter Server",
    "Description": "Formats responses into structured modules for front-end consumption", 
    "Version": "1.0.0",
    "Author": "Climate Policy Radar Team"
}

@mcp.tool()
def FormatResponseAsModules(
    response_text: str,
    chart_data: Optional[List[Dict]] = None,
    visualization_data: Optional[Dict] = None,
    map_data: Optional[Dict] = None,
    sources: Optional[List] = None,
    title: str = "Climate Policy Analysis"
) -> Dict[str, Any]:
    """
    Convert a raw response into structured modules format for front-end.
    
    Parameters:
    - response_text: Main text response from Claude
    - chart_data: Legacy chart data (list of dicts)
    - visualization_data: Structured visualization data
    - map_data: Map data for display
    - sources: Source information
    - title: Main heading for the response
    """
    modules = []
    
    # 1. Add main text response as text module
    if response_text and response_text.strip():
        # Split into paragraphs for better formatting
        paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
        
        modules.append({
            "type": "text", 
            "heading": title,
            "texts": paragraphs
        })
    
    # 2. Add visualization data as chart module
    if visualization_data and visualization_data.get("data"):
        chart_module = _create_chart_module(visualization_data)
        if chart_module:
            modules.append(chart_module)
    
    # 3. Add legacy chart data as table or chart
    if chart_data and isinstance(chart_data, list) and chart_data:
        table_module = _create_table_from_data(chart_data, "Data Summary")
        if table_module:
            modules.append(table_module)
    
    # 4. Add interactive map module
    if map_data and map_data.get("data"):
        map_module = _create_map_module(map_data)
        if map_module:
            modules.append(map_module)
        
        # Also add a summary table as backup
        map_summary = _create_map_summary_table(map_data)
        if map_summary:
            modules.append(map_summary)
    
    # 5. Add sources as table (always include)
    sources_module = _create_sources_table(sources)
    if sources_module:
        modules.append(sources_module)
    
    return {"modules": modules}

def _create_chart_module(viz_data: Dict) -> Optional[Dict]:
    """Create a Chart.js compatible chart module from visualization data."""
    viz_type = viz_data.get("visualization_type", "")
    data = viz_data.get("data", [])
    chart_config = viz_data.get("chart_config", {})
    
    if not data:
        return None
    
    try:
        df = pd.DataFrame(data)
        
        if viz_type == "by_country":
            # Country comparison bar chart
            return {
                "type": "chart",
                "chartType": "bar",
                "data": {
                    "labels": df["country"].tolist(),
                    "datasets": [{
                        "label": "Total Capacity (MW)",
                        "data": df["total_capacity_mw"].tolist(),
                        "backgroundColor": ["#4CAF50", "#FF9800", "#F44336", "#2196F3"]
                    }]
                }
            }
            
        elif viz_type == "capacity_distribution":
            # Capacity distribution bar chart
            return {
                "type": "chart", 
                "chartType": "bar",
                "data": {
                    "labels": df["capacity_range"].tolist(),
                    "datasets": [{
                        "label": "Number of Facilities",
                        "data": df["facility_count"].tolist(),
                        "backgroundColor": "#36A2EB"
                    }]
                }
            }
            
        elif viz_type == "timeline":
            # Timeline line chart
            countries = df["country"].unique()
            datasets = []
            colors = ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"]
            
            for i, country in enumerate(countries):
                country_data = df[df["country"] == country]
                datasets.append({
                    "label": country,
                    "data": country_data["capacity_mw"].tolist(),
                    "borderColor": colors[i % len(colors)],
                    "fill": False
                })
            
            return {
                "type": "chart",
                "chartType": "line", 
                "data": {
                    "labels": sorted(df["completion_year"].unique().tolist()),
                    "datasets": datasets
                }
            }
            
    except Exception as e:
        print(f"Error creating chart module: {e}")
        return None
    
    return None

def _create_table_from_data(data: List[Dict], heading: str) -> Optional[Dict]:
    """Create a table module from list of dictionaries."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Limit to reasonable number of rows for display
        display_df = df.head(10)
        
        return {
            "type": "table",
            "heading": heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.values.tolist()
        }
        
    except Exception as e:
        print(f"Error creating table: {e}")
        return None

def _create_map_module(map_data: Dict) -> Optional[Dict]:
    """Create an interactive map module from facility data using GeoPandas."""
    print(f"FORMATTER DEBUG: _create_map_module called with map_data type = {type(map_data)}")
    if map_data:
        print(f"FORMATTER DEBUG: map_data keys = {list(map_data.keys())}")
    
    facilities = map_data.get("data", [])
    metadata = map_data.get("metadata", {})
    
    print(f"FORMATTER DEBUG: facilities count = {len(facilities)}")
    print(f"FORMATTER DEBUG: metadata = {metadata}")
    
    if not facilities:
        print(f"FORMATTER DEBUG: No facilities found, returning None")
        return None
    
    try:
        # Convert to DataFrame first
        df = pd.DataFrame(facilities[:500])  # Limit for performance
        
        # Create GeoPandas DataFrame with Point geometries
        geometry = [Point(row['longitude'], row['latitude']) for _, row in df.iterrows()]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        # Color mapping by country
        country_colors = {
            'brazil': '#4CAF50',
            'india': '#FF9800', 
            'south africa': '#F44336',
            'vietnam': '#2196F3'
        }
        
        # Add styling columns
        gdf['marker_color'] = gdf['country'].str.lower().map(country_colors).fillna('#9E9E9E')
        gdf['marker_size'] = 4 + ((gdf['capacity_mw'] / 100) * 2).clip(upper=16)
        gdf['marker_opacity'] = 0.8
        
        # Add popup content
        gdf['popup_title'] = 'Solar Facility (' + gdf['country'].astype(str) + ')'
        gdf['popup_content'] = (
            'Capacity: ' + gdf['capacity_mw'].round(1).astype(str) + ' MW<br>' +
            'Coordinates: ' + gdf['latitude'].round(3).astype(str) + ', ' + 
            gdf['longitude'].round(3).astype(str)
        )
        
        # Get countries for filename (before using it)
        countries_in_data = set(gdf['country'].str.lower())
        
        # Generate GeoJSON using GeoPandas (much cleaner!)
        geojson = json.loads(gdf.to_json())
        
        # Save GeoJSON to static file instead of embedding
        # Create unique filename based on data hash for caching
        data_hash = hashlib.md5(str(sorted(gdf['cluster_id'].tolist())).encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        countries_str = "_".join(sorted(countries_in_data))
        filename = f"solar_facilities_{countries_str}_{data_hash}.geojson"
        
        # Get script directory and create static path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        static_dir = os.path.join(project_root, "static", "maps")
        os.makedirs(static_dir, exist_ok=True)
        
        file_path = os.path.join(static_dir, filename)
        
        # Save GeoJSON to file
        with open(file_path, 'w') as f:
            json.dump(geojson, f, separators=(',', ':'))  # Compact JSON
        
        print(f"Saved GeoJSON with {len(gdf)} facilities to {filename}")
        
        # Calculate bounds and center
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        
        # Create legend
        legend_items = []
        for country, color in country_colors.items():
            if country in countries_in_data:
                legend_items.append({
                    "label": country.title(),
                    "color": color,
                    "description": "Size represents capacity"
                })
        
        return {
            "type": "map",
            "mapType": "geojson_url",
            "geojson_url": f"/static/maps/{filename}",  # URL instead of embedded data
            "filename": filename,  # For debugging/caching
            "viewState": {
                "center": [center_lon, center_lat],
                "zoom": 6,
                "bounds": {
                    "north": bounds[3],  # maxy
                    "south": bounds[1],  # miny
                    "east": bounds[2],   # maxx
                    "west": bounds[0]    # minx
                }
            },
            "legend": {
                "title": "Solar Facilities",
                "items": legend_items
            },
            "metadata": {
                "total_facilities": len(gdf),
                "total_capacity_mw": float(gdf['capacity_mw'].sum()),
                "data_source": "TZ-SAM Q1 2025",
                "countries": list(countries_in_data),
                "feature_count": len(gdf),
                "file_size_kb": round(os.path.getsize(file_path) / 1024, 1)
            }
        }
        
    except Exception as e:
        print(f"Error creating map module with GeoPandas: {e}")
        import traceback
        traceback.print_exc()
        return None

def _create_map_summary_table(map_data: Dict) -> Optional[Dict]:
    """Create a summary table for map data since we can't embed interactive maps."""
    metadata = map_data.get("metadata", {})
    
    if not metadata:
        return None
    
    rows = [
        ["Total Facilities", str(metadata.get("total_facilities", "N/A"))],
        ["Total Capacity", f"{metadata.get('total_capacity', 0):.0f} MW"],
        ["Countries", ", ".join(metadata.get("countries", []))]
    ]
    
    return {
        "type": "table",
        "heading": "Solar Facilities Summary",
        "columns": ["Metric", "Value"],
        "rows": rows
    }

def _create_sources_table(sources: List) -> Optional[Dict]:
    """Create a comprehensive sources table for all data types (passages, datasets, databases)."""
    rows = []
    
    print(f"FORMATTER DEBUG: Creating comprehensive sources table with {len(sources) if sources else 0} sources")
    
    # Handle empty sources or "No source captured"
    if not sources or sources == ["No source captured"]:
        rows.append(["1", "N/A", "N/A", "N/A", "N/A", "No sources available for this response"])
    else:
        for i, source in enumerate(sources[:20], 1):  # Increased to 20 sources for comprehensive display
            print(f"FORMATTER DEBUG: Processing source {i}: {source}")
            if isinstance(source, dict):
                source_type = source.get("type", "Document")
                
                # Handle different source types differently
                if source_type.lower() in ["dataset", "database"]:
                    # Dataset/Database citation format
                    source_name = source.get("title", source.get("source_name", "Unknown Source"))
                    provider = source.get("provider", "Unknown Provider")
                    coverage = source.get("coverage", "")
                    tool_used = source.get("passage_id", source.get("tool_used", "N/A"))  # passage_id stores tool name for datasets
                    description = source.get("text", "")[:150] + "..." if source.get("text") else "N/A"
                    
                    # Create comprehensive source reference for datasets
                    source_ref = f"{source_name}"
                    if provider and provider != "Unknown Provider":
                        source_ref += f" | {provider}"
                    if coverage:
                        source_ref += f" | {coverage}"
                    
                    rows.append([
                        str(i),
                        source_ref,
                        tool_used,
                        source_type.title(),
                        "Tool/API",
                        description
                    ])
                    
                else:
                    # Traditional document/passage citation format
                    doc_id = source.get("doc_id", "N/A")
                    passage_id = source.get("passage_id", "N/A")
                    title = source.get("title", "")
                    
                    # Create document reference
                    doc_ref = doc_id
                    if title:
                        doc_ref = f"{title} ({doc_id})"
                    elif source_type:
                        doc_ref = f"{source_type.title()}: {doc_id}"
                    
                    text_snippet = source.get("text", "")[:150] + "..." if source.get("text") else "N/A"
                    
                    rows.append([
                        str(i),
                        doc_ref,
                        passage_id,
                        source_type.title() if source_type else "Document", 
                        "Knowledge Graph",
                        text_snippet
                    ])
            else:
                # Handle legacy string sources
                rows.append([str(i), "General Reference", "N/A", "General", "Legacy", str(source)[:150]])
    
    return {
        "type": "source_table",
        "heading": "Sources and References",
        "columns": ["#", "Source", "ID/Tool", "Type", "Method", "Description"],
        "rows": rows
    }

@mcp.tool()
def GetFormatterMetadata() -> Dict[str, Any]:
    """Get metadata about the response formatter."""
    return metadata

if __name__ == "__main__":
    mcp.run()