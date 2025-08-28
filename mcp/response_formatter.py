"""
Direct response formatter module - converts raw MCP data into structured modules.
No LLM calls, just deterministic data transformation.
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib
import os
from datetime import datetime


def format_response_as_modules(
    response_text: str,
    facts: List[Dict] = None,
    map_data: Optional[Dict] = None,
    chart_data: Optional[List[Dict]] = None,
    visualization_data: Optional[Dict] = None,
    sources: Optional[List] = None,
    citation_registry: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Main formatting function - converts collected data into display modules.
    
    Args:
        response_text: Synthesized text response
        facts: List of facts collected from various servers
        map_data: Map data from solar/gist servers (includes geojson_url)
        chart_data: Data for charts
        visualization_data: Structured visualization configuration
        sources: Legacy sources list
        citation_registry: Citation tracking information
    
    Returns:
        Dictionary with 'modules' list and metadata
    """
    modules = []
    
    # 1. Add main text response as text modules (split by sections if needed)
    if response_text:
        text_modules = _create_text_modules(response_text)
        modules.extend(text_modules)
    
    # 2. Add map module if we have map data
    if map_data:
        map_module = _create_map_module(map_data)
        if map_module:
            modules.append(map_module)
            # Also add a summary table for the map data
            map_summary = _create_map_summary_table(map_data)
            if map_summary:
                modules.append(map_summary)
    
    # 3. Add chart module if we have chart/visualization data
    if chart_data or visualization_data:
        chart_module = _create_chart_module(chart_data or visualization_data)
        if chart_module:
            modules.append(chart_module)
    
    # 4. Add citations/sources table
    if citation_registry and citation_registry.get("citations"):
        citation_module = _create_citation_table(citation_registry)
        if citation_module:
            modules.append(citation_module)
    elif sources:
        sources_module = _create_sources_table(sources)
        if sources_module:
            modules.append(sources_module)
    
    # Calculate metadata
    metadata = {
        "modules_count": len(modules),
        "has_maps": any(m.get("type") == "map" for m in modules),
        "has_charts": any(m.get("type") == "chart" for m in modules),
        "has_tables": any("table" in m.get("type", "") for m in modules)
    }
    
    return {
        "modules": modules,
        "metadata": metadata
    }


def _create_text_modules(response_text: str) -> List[Dict]:
    """
    Convert response text into text modules.
    Splits on headers if present, otherwise creates a single module.
    """
    modules = []
    
    # Split by markdown headers (##)
    sections = []
    current_section = {"heading": "", "content": []}
    
    for line in response_text.split('\n'):
        if line.startswith('## '):
            # Save previous section if it has content
            if current_section["content"]:
                sections.append(current_section)
            # Start new section
            current_section = {
                "heading": line[3:].strip(),
                "content": []
            }
        else:
            current_section["content"].append(line)
    
    # Don't forget the last section
    if current_section["content"]:
        sections.append(current_section)
    
    # Create modules from sections
    if sections:
        for section in sections:
            content = '\n'.join(section["content"]).strip()
            if content:
                modules.append({
                    "type": "text",
                    "heading": section["heading"],
                    "texts": [content]
                })
    else:
        # No sections found, create single module
        modules.append({
            "type": "text",
            "heading": "",
            "texts": [response_text]
        })
    
    return modules


def _create_map_module(map_data: Dict) -> Optional[Dict]:
    """
    Create a map module from solar/gist map data.
    Expects map_data with type: "map_data_summary" and geojson_url.
    """
    if not map_data:
        return None
    
    # Handle map_data_summary format from solar facilities
    if map_data.get("type") == "map_data_summary":
        summary = map_data.get("summary", {})
        geojson_url = map_data.get("geojson_url")
        
        if not geojson_url:
            return None
        
        # Determine map bounds based on countries
        countries = summary.get("countries", [])
        bounds, center = _calculate_map_bounds(countries)
        
        return {
            "type": "map",
            "mapType": "geojson_url",
            "geojson_url": geojson_url,
            "viewState": {
                "center": center,
                "zoom": 6,
                "bounds": bounds
            },
            "metadata": {
                "total_facilities": summary.get("total_facilities", 0),
                "total_capacity_mw": summary.get("total_capacity_mw", 0),
                "countries": countries,
                "data_source": "TZ-SAM Q1 2025"
            }
        }
    
    # Handle legacy format with facilities list
    elif map_data.get("data"):
        # Legacy handling - would need to generate GeoJSON
        return None
    
    return None


def _calculate_map_bounds(countries: List[str]) -> Tuple[Dict, List[float]]:
    """Calculate map bounds and center based on countries."""
    country_bounds = {
        "brazil": {"lat": [-33.75, 5.27], "lon": [-73.98, -34.73]},
        "india": {"lat": [8.08, 35.50], "lon": [68.18, 97.40]},
        "vietnam": {"lat": [8.55, 23.39], "lon": [102.14, 109.46]},
        "south africa": {"lat": [-34.83, -22.13], "lon": [16.45, 32.89]},
        "china": {"lat": [18.0, 53.56], "lon": [73.66, 135.05]},
        "united states of america": {"lat": [24.52, 49.38], "lon": [-125.0, -66.93]},
        "japan": {"lat": [24.0, 46.0], "lon": [123.0, 146.0]}
    }
    
    all_lats = []
    all_lons = []
    
    for country in countries:
        country_key = country.lower().replace(" ", "_")
        if country_key in country_bounds:
            bounds = country_bounds[country_key]
            all_lats.extend(bounds["lat"])
            all_lons.extend(bounds["lon"])
    
    if all_lats and all_lons:
        bounds = {
            "north": max(all_lats),
            "south": min(all_lats),
            "east": max(all_lons),
            "west": min(all_lons)
        }
        center = [(bounds["west"] + bounds["east"]) / 2, (bounds["north"] + bounds["south"]) / 2]
    else:
        # Default world view
        bounds = {"north": 50, "south": -50, "east": 180, "west": -180}
        center = [0, 0]
    
    return bounds, center


def _create_map_summary_table(map_data: Dict) -> Optional[Dict]:
    """Create a summary table for map data."""
    if not map_data or map_data.get("type") != "map_data_summary":
        return None
    
    summary = map_data.get("summary", {})
    
    rows = []
    
    # Add facility count
    if summary.get("total_facilities"):
        rows.append(["Total Facilities", f"{summary['total_facilities']:,}"])
    
    # Add capacity
    if summary.get("total_capacity_mw"):
        rows.append(["Total Capacity", f"{summary['total_capacity_mw']:,.1f} MW"])
    
    # Add capacity range
    if summary.get("capacity_range_mw"):
        cap_range = summary["capacity_range_mw"]
        if cap_range.get("min") and cap_range.get("max"):
            rows.append(["Capacity Range", f"{cap_range['min']:.1f} - {cap_range['max']:.1f} MW"])
        if cap_range.get("average"):
            rows.append(["Average Capacity", f"{cap_range['average']:.1f} MW"])
    
    # Add countries
    if summary.get("countries"):
        countries_list = summary["countries"]
        if len(countries_list) <= 5:
            rows.append(["Countries", ", ".join(countries_list)])
        else:
            rows.append(["Countries", f"{len(countries_list)} countries"])
    
    if not rows:
        return None
    
    return {
        "type": "table",
        "heading": "Solar Facilities Summary",
        "columns": ["Metric", "Value"],
        "rows": rows
    }


def _create_chart_module(data: Any) -> Optional[Dict]:
    """Create a chart module from visualization data."""
    if not data:
        return None
    
    # Handle different data formats
    if isinstance(data, dict) and data.get("visualization_type"):
        # Structured visualization data
        viz_type = data["visualization_type"]
        chart_data = data.get("data", [])
        config = data.get("chart_config", {})
        
        if viz_type == "by_country" and chart_data:
            return {
                "type": "chart",
                "chartType": "bar",
                "data": {
                    "labels": [d["country"] for d in chart_data[:10]],  # Top 10
                    "datasets": [{
                        "label": "Facility Count",
                        "data": [d.get("facility_count", 0) for d in chart_data[:10]],
                        "backgroundColor": "#4CAF50"
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": config.get("title", "Solar Facilities by Country")
                        }
                    }
                }
            }
    
    return None


def _create_citation_table(citation_registry: Dict) -> Optional[Dict]:
    """Create a numbered citation table from registry."""
    citations = citation_registry.get("citations", [])
    
    if not citations:
        return None
    
    rows = []
    for i, citation in enumerate(citations, 1):
        source = citation.get("source", "Unknown")
        source_type = citation.get("type", "Document")
        description = citation.get("description", "")
        
        rows.append([str(i), source, source_type, description])
    
    return {
        "type": "numbered_citation_table",
        "heading": "References",
        "columns": ["#", "Source", "Type", "Description"],
        "rows": rows
    }


def _create_sources_table(sources: List) -> Optional[Dict]:
    """Create a simple sources table (legacy format)."""
    if not sources:
        return None
    
    rows = []
    for i, source in enumerate(sources, 1):
        if isinstance(source, dict):
            name = source.get("name", source.get("source", "Unknown"))
            rows.append([str(i), name])
        else:
            rows.append([str(i), str(source)])
    
    return {
        "type": "table",
        "heading": "Sources",
        "columns": ["#", "Source"],
        "rows": rows
    }


def create_map_module_from_facts(facts: List[Dict]) -> Optional[Dict]:
    """
    Intelligently create a map module by analyzing facts for map data.
    Called after synthesis when we have all facts.
    """
    # Look for facts that contain map data
    for fact in facts:
        if fact.get("has_map_data") or fact.get("geojson_url"):
            return _create_map_module({
                "type": "map_data_summary",
                "geojson_url": fact.get("geojson_url"),
                "summary": fact.get("summary", {})
            })
    
    return None


def organize_modules_by_relevance(modules: List[Dict], query: str) -> List[Dict]:
    """
    Simple heuristic-based organization of modules.
    No LLM needed - just logical ordering.
    """
    # Group modules by type
    text_modules = [m for m in modules if m.get("type") == "text"]
    map_modules = [m for m in modules if m.get("type") == "map"]
    chart_modules = [m for m in modules if m.get("type") == "chart"]
    table_modules = [m for m in modules if "table" in m.get("type", "")]
    
    organized = []
    
    # 1. Start with introductory text
    intro_texts = [m for m in text_modules if not m.get("heading") or "introduction" in m.get("heading", "").lower()]
    organized.extend(intro_texts)
    
    # 2. Add maps early if query mentions "map", "where", "location"
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["map", "where", "location", "show me"]):
        organized.extend(map_modules)
    
    # 3. Add remaining text modules
    remaining_texts = [m for m in text_modules if m not in intro_texts]
    organized.extend(remaining_texts)
    
    # 4. Add visualizations
    if map_modules and not any(m in organized for m in map_modules):
        organized.extend(map_modules)
    organized.extend(chart_modules)
    
    # 5. Add data tables (except citations)
    data_tables = [m for m in table_modules if m.get("type") != "numbered_citation_table"]
    organized.extend(data_tables)
    
    # 6. Always end with citations
    citation_tables = [m for m in table_modules if m.get("type") == "numbered_citation_table"]
    organized.extend(citation_tables)
    
    return organized