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

def _insert_inline_citations(text: str, module_id: str, citation_registry: Optional[Dict] = None) -> str:
    """
    Insert inline citation superscripts into text based on content analysis.
    
    Args:
        text: Text content to add citations to
        module_id: Module identifier to look up relevant citations
        citation_registry: Registry containing citation information
        
    Returns:
        Text with inline citation superscripts added
    """
    if not citation_registry or not citation_registry.get("module_citations"):
        return text
    
    # Get citations for this module
    module_citations = citation_registry.get("module_citations", {})
    relevant_citations = []
    
    # Look for citations from tools that contributed to this module
    for mod_id, citations in module_citations.items():
        if module_id in mod_id or any(keyword in text.lower() for keyword in ["solar", "gist", "climate", "emission"]):
            relevant_citations.extend(citations)
    
    # Remove duplicates and sort
    relevant_citations = sorted(list(set(relevant_citations)))
    
    if not relevant_citations:
        return text
    
    # Simple citation insertion at sentence ends
    sentences = text.split('. ')
    if len(sentences) > 1:
        # Add citations to the first few sentences
        for i in range(min(2, len(sentences))):
            if sentences[i] and not sentences[i].endswith('^'):
                citation_nums = relevant_citations[:min(3, len(relevant_citations))]
                superscript = f"^{','.join(map(str, citation_nums))}^"
                sentences[i] = sentences[i] + superscript
        
        return '. '.join(sentences)
    else:
        # Single sentence - add citation at end
        citation_nums = relevant_citations[:3]
        superscript = f"^{','.join(map(str, citation_nums))}^"
        return text + superscript
    
def _create_numbered_citation_table(citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create a numbered citation table from the citation registry."""
    if not citation_registry or not citation_registry.get("citations"):
        return None
    
    citations = citation_registry.get("citations", {})
    rows = []
    
    for citation_num in sorted(citations.keys()):
        source = citations[citation_num]
        
        if isinstance(source, dict):
            source_type = source.get("type", "Document")
            
            if source_type.lower() in ["dataset", "database"]:
                # Dataset citation format
                source_name = source.get("title", "Unknown Source")
                provider = source.get("provider", "Unknown Provider")
                tool_used = source.get("passage_id", "N/A")
                description = source.get("text", "")[:100] + "..." if source.get("text") else "N/A"
                
                source_ref = f"{source_name}"
                if provider and provider != "Unknown Provider":
                    source_ref += f" | {provider}"
                
                rows.append([
                    str(citation_num),
                    source_ref,
                    tool_used,
                    source_type.title(),
                    description
                ])
            else:
                # Document citation format
                title = source.get("title", "")
                doc_id = source.get("doc_id", "N/A")
                doc_ref = f"{title} ({doc_id})" if title else doc_id
                passage_id = source.get("passage_id", "N/A")
                text_snippet = source.get("text", "")[:100] + "..." if source.get("text") else "N/A"
                
                rows.append([
                    str(citation_num),
                    doc_ref,
                    passage_id,
                    "Document",
                    text_snippet
                ])
        else:
            # Legacy string source
            rows.append([str(citation_num), str(source)[:100], "N/A", "General", ""])
    
    return {
        "type": "numbered_citation_table",
        "heading": "References",
        "columns": ["#", "Source", "ID/Tool", "Type", "Description"],
        "rows": rows
    } if rows else None

def _add_citations_to_table_heading(heading: str, tool_name: str, citation_registry: Optional[Dict] = None) -> str:
    """Add citation superscripts to table headings based on the tool that generated the data."""
    if not citation_registry or not citation_registry.get("module_citations"):
        return heading
    
    # Find citations associated with this tool
    module_citations = citation_registry.get("module_citations", {})
    relevant_citations = []
    
    for module_id, citations in module_citations.items():
        if tool_name in module_id:
            relevant_citations.extend(citations)
    
    # Remove duplicates and sort
    relevant_citations = sorted(list(set(relevant_citations)))
    
    if relevant_citations:
        citation_nums = relevant_citations[:3]  # Limit to first 3 citations
        superscript = f"^{','.join(map(str, citation_nums))}^"
        return f"{heading} {superscript}"
    
    return heading

@mcp.tool()
def FormatResponseAsModules(
    response_text: str,
    chart_data: Optional[List[Dict]] = None,
    visualization_data: Optional[Dict] = None,
    map_data: Optional[Dict] = None,
    sources: Optional[List] = None,
    title: str = "Climate Policy Analysis",
    citation_registry: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convert a raw response into structured modules format for front-end with inline citations.
    
    Parameters:
    - response_text: Main text response from Claude
    - chart_data: Legacy chart data (list of dicts)
    - visualization_data: Structured visualization data
    - map_data: Map data for display
    - sources: Source information
    - title: Main heading for the response
    - citation_registry: Citation registry with numbered sources and module mappings
    """
    modules = []
    
    # 1. Add main text response as text module with inline citations
    if response_text and response_text.strip():
        # Split into paragraphs for better formatting
        paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
        
        # Add inline citations to paragraphs
        if citation_registry:
            paragraphs_with_citations = []
            for i, paragraph in enumerate(paragraphs):
                cited_paragraph = _insert_inline_citations(paragraph, f"text_module_{i}", citation_registry)
                paragraphs_with_citations.append(cited_paragraph)
            paragraphs = paragraphs_with_citations
        
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
    
    # 3. Add legacy chart data as enhanced table
    if chart_data and isinstance(chart_data, list) and chart_data:
        table_module = _create_enhanced_table_from_data(chart_data, "Data Summary", "", citation_registry)
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
    
    # 5. Add sources as numbered citation table or legacy sources table
    if citation_registry:
        # Use new numbered citation table
        citation_table = _create_numbered_citation_table(citation_registry)
        if citation_table:
            modules.append(citation_table)
    else:
        # Fallback to legacy sources table
        sources_module = _create_sources_table(sources)
        if sources_module:
            modules.append(sources_module)
    
    # 6. OPTIONAL: Organize modules into narrative flow if we have enough content
    if len(modules) > 2 and citation_registry:
        try:
            # Extract query from title or use default
            query_context = title if title != "Climate Policy Analysis" else "Analyze climate policy data"
            
            print(f"🎯 FORMATTER DEBUG: Attempting narrative organization for {len(modules)} modules")
            
            # Call the narrative organizer
            organized_result = OrganizeModulesIntoNarrative(modules, query_context, citation_registry)
            
            if organized_result and "modules" in organized_result:
                organized_modules = organized_result["modules"]
                print(f"🎯 FORMATTER DEBUG: Narrative organization successful: {len(modules)} -> {len(organized_modules)} modules")
                
                # Use organized modules instead of original
                modules = organized_modules
            else:
                print(f"🎯 FORMATTER DEBUG: Narrative organization returned no modules, using original")
                
        except Exception as e:
            print(f"🎯 FORMATTER DEBUG: Error in narrative organization: {e}")
            # Fall back to original modules on any error
            pass
    else:
        print(f"🎯 FORMATTER DEBUG: Skipping narrative organization (modules: {len(modules)}, has_citations: {bool(citation_registry)})")
    
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

def detect_table_type(tool_name: str, data: List[Dict]) -> str:
    """Automatically determine appropriate table type based on tool and data structure."""
    
    # Ranking tables (ordered by numeric value)
    if tool_name in ["GetGistTopEmitters", "GetLargestSolarFacilities", "GetGistHighRiskCompanies"]:
        return "ranking_table"
    
    # Comparison tables (multiple entities compared)
    if tool_name in ["CompareBrazilianStates", "GetSolarCapacityByCountry", "GetGistEmissionsBySector"]:
        return "comparison_table"
    
    # Trend tables (time series data)
    if tool_name in ["GetSolarConstructionTimeline", "GetGistEmissionsTrends", "GetGistBiodiversityTrends"]:
        return "trend_table"
        
    # Geographic tables (location-based)
    if tool_name in ["GetGistAssetsByCountry", "GetSolarFacilitiesInRadius", "GetGistAssetsMapData"]:
        return "geographic_table"
        
    # Summary tables (aggregated overviews)
    if tool_name in ["GetGistCompaniesBySector", "GetBrazilianStatesOverview", "GetAvailableDatasets"]:
        return "summary_table"
        
    # Detail tables (specific breakdowns)
    if tool_name in ["GetGistScope3Emissions", "GetGistCompanyProfile", "GetSolarFacilitiesByCountry"]:
        return "detail_table"
        
    # Default fallback
    return "table"

def _create_comparison_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create side-by-side comparison table for entities like countries, sectors, states."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Limit rows for readability
        display_df = df.head(15)
        
        # Add citations to heading if available
        cited_heading = _add_citations_to_table_heading(heading, tool_name, citation_registry)
        
        return {
            "type": "comparison_table",
            "heading": cited_heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "total_entities": len(df),
                "displayed_entities": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating comparison table: {e}")
        return None

def _create_ranking_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create ordered ranking/leaderboard table."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Add rank column if not present
        if 'rank' not in df.columns and 'Rank' not in df.columns:
            df.insert(0, 'Rank', range(1, len(df) + 1))
        
        # Limit to top 20 for rankings
        display_df = df.head(20)
        
        # Add citations to heading if available
        cited_heading = _add_citations_to_table_heading(heading, tool_name, citation_registry)
        
        return {
            "type": "ranking_table", 
            "heading": cited_heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "total_entries": len(df),
                "displayed_entries": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating ranking table: {e}")
        return None

def _create_trend_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create time series analysis table."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Sort by time column if present
        time_columns = ['year', 'Year', 'date', 'Date', 'reporting_year', 'Reporting_Year']
        for col in time_columns:
            if col in df.columns:
                df = df.sort_values(col)
                break
        
        # Limit to reasonable time range (last 10 years/periods)
        display_df = df.tail(10)
        
        return {
            "type": "trend_table",
            "heading": heading,
            "columns": display_df.columns.tolist(), 
            "rows": display_df.values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "total_periods": len(df),
                "displayed_periods": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating trend table: {e}")
        return None

def _create_summary_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create aggregated overview table."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # For summary tables, show more rows (up to 25)
        display_df = df.head(25)
        
        return {
            "type": "summary_table",
            "heading": heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "total_items": len(df),
                "displayed_items": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating summary table: {e}")
        return None

def _create_detail_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create detailed breakdown table for specific entity."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Detail tables show fewer rows but more columns
        display_df = df.head(12)
        
        return {
            "type": "detail_table",
            "heading": heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "total_records": len(df),
                "displayed_records": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating detail table: {e}")
        return None

def _create_geographic_table(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create location-based analysis table."""
    if not data or not isinstance(data, list):
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Limit geographic tables to 15 entries for readability
        display_df = df.head(15)
        
        return {
            "type": "geographic_table", 
            "heading": heading,
            "columns": display_df.columns.tolist(),
            "rows": display_df.values.tolist(),
            "metadata": {
                "tool_used": tool_name,
                "total_locations": len(df),
                "displayed_locations": len(display_df)
            }
        }
        
    except Exception as e:
        print(f"Error creating geographic table: {e}")
        return None

def _create_enhanced_table_from_data(data: List[Dict], heading: str, tool_name: str = "", citation_registry: Optional[Dict] = None) -> Optional[Dict]:
    """Create appropriately typed table based on tool and data structure with citation support."""
    if not data or not isinstance(data, list):
        return None
    
    # Detect appropriate table type
    table_type = detect_table_type(tool_name, data)
    
    # Create table using appropriate specialized function
    if table_type == "comparison_table":
        return _create_comparison_table(data, heading, tool_name, citation_registry)
    elif table_type == "ranking_table":
        return _create_ranking_table(data, heading, tool_name, citation_registry)
    elif table_type == "trend_table":
        return _create_trend_table(data, heading, tool_name, citation_registry)
    elif table_type == "summary_table":
        return _create_summary_table(data, heading, tool_name, citation_registry)
    elif table_type == "detail_table":
        return _create_detail_table(data, heading, tool_name, citation_registry)
    elif table_type == "geographic_table":
        return _create_geographic_table(data, heading, tool_name, citation_registry)
    else:
        # Fallback to original function
        return _create_table_from_data(data, heading)

@mcp.tool()
def CreateMultipleTablesFromToolResults(
    tool_results: List[Dict],
    query_context: Optional[str] = None,
    citation_registry: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create multiple enhanced tables from tool results with intelligent table typing.
    
    Parameters:
    - tool_results: List of tool results with tool names and data
    - query_context: Optional context about the query for better table generation
    
    Returns modules with multiple appropriately typed tables.
    """
    print(f"🔧 FORMATTER DEBUG: CreateMultipleTablesFromToolResults called")
    print(f"  - Received {len(tool_results)} tool results")
    print(f"  - Query context: '{query_context[:50] if query_context else 'None'}...'")
    
    modules = []
    
    # Process each tool result into appropriate table type
    for i, result in enumerate(tool_results):
        tool_name = result.get("tool_name", "")
        tool_data = result.get("data", [])
        
        print(f"🔧 FORMATTER DEBUG: Processing tool result [{i+1}]: {tool_name}")
        print(f"  - Data type: {type(tool_data)}")
        print(f"  - Data length: {len(tool_data) if isinstance(tool_data, list) else 'not a list'}")
        
        if not tool_data or not isinstance(tool_data, list):
            print(f"🔧 FORMATTER DEBUG: ❌ Skipping {tool_name} - no data or not a list")
            continue
            
        # Generate appropriate heading based on tool name
        heading = _generate_table_heading(tool_name, tool_data)
        print(f"🔧 FORMATTER DEBUG: Generated heading for {tool_name}: '{heading}'")
        
        # Create enhanced table with appropriate type and citations
        table_module = _create_enhanced_table_from_data(tool_data, heading, tool_name, citation_registry)
        
        if table_module:
            table_type = table_module.get("type", "unknown")
            row_count = len(table_module.get("rows", [])) if "rows" in table_module else "no rows"
            print(f"🔧 FORMATTER DEBUG: ✅ Created {table_type} table for {tool_name} ({row_count} rows)")
            modules.append(table_module)
        else:
            print(f"🔧 FORMATTER DEBUG: ❌ Failed to create table for {tool_name}")
    
    print(f"🔧 FORMATTER DEBUG: Returning {len(modules)} table modules")
    return {"modules": modules}

@mcp.tool()
def OrganizeModulesIntoNarrative(
    modules: List[Dict],
    query: str,
    citation_registry: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Intelligently organize and interweave modules into a cohesive narrative using Sonnet model.
    
    Parameters:
    - modules: List of module dictionaries (text, tables, charts, maps)
    - query: Original user query for context
    - citation_registry: Citation registry for maintaining citation continuity
    
    Returns:
    - Dictionary with reorganized modules and transition text
    """
    print(f"🎯 NARRATIVE DEBUG: Organizing {len(modules)} modules into narrative")
    
    if not modules:
        return {"modules": []}
    
    # Create module summary for the narrative organizer
    module_summary = []
    for i, module in enumerate(modules):
        module_type = module.get("type", "unknown")
        heading = module.get("heading", f"Module {i+1}")
        
        summary = {
            "index": i,
            "type": module_type,
            "heading": heading,
            "preview": _get_module_preview(module)
        }
        module_summary.append(summary)
    
    # Use Claude Sonnet to organize the narrative
    try:
        import anthropic
        client = anthropic.Anthropic()
        
        system_prompt = """You are an expert at organizing information into compelling narratives. Given a user query and a list of data modules, organize them into the most logical and engaging flow.

Your task:
1. Determine the optimal order for presenting the modules
2. Identify where transition text would help connect modules
3. Suggest where modules could be grouped or sections created
4. Maintain citation integrity - don't break citation numbering

Return a JSON object with:
{
    "narrative_plan": "Brief description of your organization strategy",
    "sections": [
        {
            "title": "Section title",
            "modules": [0, 1, 2], // Module indices in this section
            "transition_text": "Optional text to introduce this section"
        }
    ],
    "final_order": [0, 3, 1, 2, 4], // Final module order
    "reasoning": "Explanation of your choices"
}

Focus on creating a logical flow that answers the user's question comprehensively."""

        user_prompt = f"""Original query: {query}

Available modules:
{json.dumps(module_summary, indent=2)}

Organize these modules into the most effective narrative structure."""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        narrative_plan = json.loads(response.content[0].text)
        print(f"🎯 NARRATIVE DEBUG: Received narrative plan: {narrative_plan.get('narrative_plan', 'No plan')}")
        
        # Reorganize modules according to the narrative plan
        reorganized_modules = []
        final_order = narrative_plan.get("final_order", list(range(len(modules))))
        
        # Add sections with transition text
        sections = narrative_plan.get("sections", [])
        current_section_idx = 0
        
        for module_idx in final_order:
            if module_idx < len(modules):
                # Check if we're starting a new section
                if current_section_idx < len(sections):
                    section = sections[current_section_idx]
                    if module_idx == section.get("modules", [None])[0]:  # First module in section
                        # Add section transition if provided
                        if section.get("transition_text"):
                            transition_module = {
                                "type": "narrative_transition",
                                "heading": section.get("title", ""),
                                "text": section.get("transition_text", "")
                            }
                            reorganized_modules.append(transition_module)
                        current_section_idx += 1
                
                reorganized_modules.append(modules[module_idx])
        
        return {
            "modules": reorganized_modules,
            "narrative_plan": narrative_plan,
            "original_count": len(modules),
            "organized_count": len(reorganized_modules)
        }
        
    except Exception as e:
        print(f"🎯 NARRATIVE DEBUG: Error organizing narrative: {e}")
        # Fallback to original order
        return {"modules": modules}

def _get_module_preview(module: Dict) -> str:
    """Generate a preview description of a module for narrative planning."""
    module_type = module.get("type", "unknown")
    
    if module_type == "text":
        texts = module.get("texts", [])
        preview = texts[0][:100] + "..." if texts else "Text content"
        return f"Text: {preview}"
    
    elif module_type.endswith("_table"):
        rows = module.get("rows", [])
        columns = module.get("columns", [])
        preview = f"Table with {len(rows)} rows, {len(columns)} columns"
        if columns:
            preview += f" (columns: {', '.join(columns[:3])}...)"
        return preview
    
    elif module_type == "chart":
        chart_type = module.get("chartType", "unknown")
        return f"Chart: {chart_type} visualization"
    
    elif module_type == "map":
        data = module.get("data", [])
        return f"Map with {len(data)} data points"
    
    elif module_type == "numbered_citation_table":
        rows = module.get("rows", [])
        return f"References table with {len(rows)} citations"
    
    else:
        return f"{module_type.replace('_', ' ').title()} module"

def _generate_table_heading(tool_name: str, data: List[Dict]) -> str:
    """Generate appropriate table heading based on tool name and data."""
    
    # Custom headings for specific tools
    headings_map = {
        "GetGistCompaniesBySector": "Companies by Sector",
        "GetGistTopEmitters": "Top Emitting Companies",
        "GetGistHighRiskCompanies": "Highest Environmental Risk Companies",
        "GetSolarCapacityByCountry": "Solar Capacity by Country",
        "GetLargestSolarFacilities": "Largest Solar Facilities",
        "GetSolarConstructionTimeline": "Solar Construction Timeline",
        "GetBrazilianStatesOverview": "Brazilian State Climate Policies",
        "CompareBrazilianStates": "State Policy Comparison",
        "GetGistEmissionsBySector": "Emissions by Sector",
        "GetGistRiskByCategory": "Environmental Risk Assessment",
        "GetGistEmissionsTrends": "Emissions Trends Analysis",
        "GetGistAssetsByCountry": "Assets by Geographic Location",
        "GetGistScope3Emissions": "Scope 3 Emissions Breakdown",
        "GetGistCompanyProfile": "Company Sustainability Profile",
        "GetAvailableDatasets": "Available Datasets",
        "GetInstitutionsProcessesData": "Climate Governance Institutions",
        "GetPlansAndPoliciesData": "Climate Plans and Policies"
    }
    
    if tool_name in headings_map:
        return headings_map[tool_name]
    
    # Generate heading from tool name
    heading = tool_name.replace("Get", "").replace("Gist", "").replace("Solar", "Solar ")
    heading = " ".join([word.capitalize() for word in heading.split()])
    
    # Add context based on data size
    if data and len(data) > 0:
        if len(data) == 1:
            heading += " Details"
        elif len(data) < 5:
            heading += " Summary"
        else:
            heading += " Analysis"
    
    return heading

@mcp.tool()
def GetFormatterMetadata() -> Dict[str, Any]:
    """Get metadata about the response formatter."""
    return metadata

if __name__ == "__main__":
    mcp.run()