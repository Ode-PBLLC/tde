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
    
    # Handle map_data_summary format (and treat any geojson_url-bearing payload as compatible)
    if map_data.get("type") == "map_data_summary" or (map_data.get("geojson_url") and isinstance(map_data.get("geojson_url"), str)):
        summary = map_data.get("summary", {})
        metadata = map_data.get("metadata", {}) or {}
        geojson_url = map_data.get("geojson_url")
        filename_hint = map_data.get("geojson_filename")
        # Strict correlation detection: explicit flag or correlation_* filename/URL only
        is_correlation_map = bool(map_data.get("is_correlation_map") is True)
        if not is_correlation_map:
            try:
                url_l = geojson_url.lower() if isinstance(geojson_url, str) else ""
                fn_l = filename_hint.lower() if isinstance(filename_hint, str) else ""
            except Exception:
                url_l, fn_l = "", ""
            if url_l.startswith("/static/maps/correlation_") or fn_l.startswith("correlation_"):
                is_correlation_map = True
        # Safety: force non-correlation for plain solar files
        try:
            url_l = geojson_url.lower() if isinstance(geojson_url, str) else ""
            fn_l = filename_hint.lower() if isinstance(filename_hint, str) else ""
        except Exception:
            url_l, fn_l = "", ""
        if ('solar_facilities_' in url_l) or ('solar_facilities_' in fn_l):
            is_correlation_map = False
        
        if not geojson_url:
            return None

        # Ensure we have a complete URL for the frontend
        import os
        if geojson_url.startswith('/'):
            # Get base URL from environment or use HTTPS default to avoid mixed content
            base_url = os.getenv('API_BASE_URL', 'https://api.transitiondigital.org')
            # Remove trailing slash from base URL if present
            base_url = base_url.rstrip('/')
            geojson_url = base_url + geojson_url

        # Determine geometry type(s) for downstream renderers
        geom_candidates = []

        def _collect_geometry_types(value):
            if not value:
                return
            if isinstance(value, str):
                geom_candidates.append(value.lower())
            elif isinstance(value, dict):
                for nested in value.values():
                    _collect_geometry_types(nested)
            elif isinstance(value, (list, tuple, set)):
                for nested in value:
                    _collect_geometry_types(nested)

        _collect_geometry_types(map_data.get("geometry_type"))
        _collect_geometry_types(map_data.get("geometry_types"))
        _collect_geometry_types(summary.get("geometry_type"))
        _collect_geometry_types(summary.get("geometry_types"))
        _collect_geometry_types(metadata.get("geometry_type"))
        _collect_geometry_types(metadata.get("geometry_types"))

        geometry_types = sorted({candidate for candidate in geom_candidates if isinstance(candidate, str) and candidate})
        geometry_type = "point"
        if geometry_types:
            has_point = any(candidate.startswith("point") for candidate in geometry_types)
            has_polygon = any(candidate.startswith("poly") or candidate.startswith("multi") for candidate in geometry_types)
            if has_polygon and not has_point:
                geometry_type = "polygon"
            elif has_point:
                geometry_type = "point"
            else:
                geometry_type = geometry_types[0]
        if not geometry_types:
            geometry_types = [geometry_type]

        # Determine map bounds
        bounds = summary.get("bounds")
        center = summary.get("center")
        countries = summary.get("countries", [])
        if not bounds:
            # Compute from countries; fallback to Brazil center
            bounds, center_calc = _calculate_map_bounds(countries)
            if not center:
                center = center_calc if center_calc else [-51.9253, -14.2350]
        if not center:
            center = [-51.9253, -14.2350]
        
        # Generate legend items for countries or layers
        country_colors = {
            "brazil": "#4CAF50",
            "india": "#2196F3", 
            "china": "#FF9800",
            "united states": "#9C27B0",
            "japan": "#F44336",
            "germany": "#009688",
            "australia": "#FFEB3B",
            "south africa": "#795548",
            "mexico": "#607D8B",
            "chile": "#E91E63",
            # Extended labels for correlation maps
            "solar facilities": "#FFD700",  # gold
            "deforestation": "#8B4513"      # brown
        }
        
        legend_items = []
        if is_correlation_map:
            # Force a layer-based legend regardless of countries list
            legend_items = [
                {"label": "Solar Assets", "color": "#FFD700"},
                {"label": "Deforestation Areas", "color": "#8B4513"}
            ]
        else:
            if countries:
                legend_items = [
                    {
                        "label": country.title(),
                        "color": country_colors.get(country.lower(), "#9E9E9E")
                    }
                    for country in countries[:10]
                ]
            # No heuristic fallback to correlation here to avoid misclassification
        
        return {
            "type": "map",
            "mapType": "geojson_url",
            "geojson_url": geojson_url,
            "geometry_type": geometry_type,
            "geometry_types": geometry_types,
            "viewState": {
                "center": center,
                "zoom": 6,
                "bounds": bounds
            },
            "legend": {
                "title": ("Correlation Map" if is_correlation_map else summary.get("title", "Spatial Map")),
                "items": legend_items
            },
            "metadata": {
                "total_facilities": summary.get("total_facilities", 0),
                "total_capacity_mw": summary.get("total_capacity_mw", 0),
                "countries": countries,
                "geometry_type": geometry_type,
                "geometry_types": geometry_types,
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

        # Helper getters
        x_key = (config.get("x_axis") or "country") if isinstance(config, dict) else "country"
        y_key = (config.get("y_axis") or "facility_count") if isinstance(config, dict) else "facility_count"
        title = (config.get("title") or "Chart") if isinstance(config, dict) else "Chart"
        chart_type = (config.get("chart_type") or "bar") if isinstance(config, dict) else "bar"
        color_key = config.get("color") if isinstance(config, dict) else None

        if not chart_data:
            return None

        # by_country: generic bar using x/y keys
        if viz_type == "by_country":
            labels = [str(d.get(x_key, "")) for d in chart_data]
            values = [d.get(y_key, 0) or 0 for d in chart_data]
            return {
                "type": "chart",
                "chartType": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": y_key.replace("_", " ").title(),
                        "data": values,
                        "backgroundColor": "#4CAF50"
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": title}}
                }
            }

        # time_series: line; optional color series
        if viz_type == "time_series":
            # Build series by color_key or single series
            if color_key:
                # Group by series key, x is year
                series = {}
                years_set = set()
                for d in chart_data:
                    series_key = str(d.get(color_key, "Unknown"))
                    year = d.get(x_key)
                    val = d.get(y_key, 0) or 0
                    if year is None:
                        continue
                    years_set.add(int(year))
                    series.setdefault(series_key, {}).update({int(year): val})
                years = sorted(years_set)
                datasets = []
                palette = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336", "#009688"]
                for idx, (name, points) in enumerate(series.items()):
                    data_vals = [points.get(y, 0) for y in years]
                    datasets.append({
                        "label": name,
                        "data": data_vals,
                        "borderColor": palette[idx % len(palette)],
                        "tension": 0.1,
                        "fill": False
                    })
                return {
                    "type": "chart",
                    "chartType": "line",
                    "data": {"labels": years, "datasets": datasets},
                    "options": {"responsive": True, "plugins": {"title": {"display": True, "text": title}}}
                }
            else:
                # Single series
                pairs = [(d.get(x_key), d.get(y_key, 0) or 0) for d in chart_data if d.get(x_key) is not None]
                pairs = sorted(((int(x), y) for x, y in pairs), key=lambda t: t[0])
                years = [p[0] for p in pairs]
                values = [p[1] for p in pairs]
                return {
                    "type": "chart",
                    "chartType": "line",
                    "data": {
                        "labels": years,
                        "datasets": [{"label": y_key.replace("_", " ").title(), "data": values, "borderColor": "#2196F3", "tension": 0.1, "fill": False}]
                    },
                    "options": {"responsive": True, "plugins": {"title": {"display": True, "text": title}}}
                }

        # comparison: generic bar using x/y keys
        if viz_type == "comparison":
            labels = [str(d.get(x_key, "")) for d in chart_data]
            values = [d.get(y_key, 0) or 0 for d in chart_data]
            return {
                "type": "chart",
                "chartType": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": y_key.replace("_", " ").title(),
                        "data": values,
                        "backgroundColor": "#FF9800"
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {"title": {"display": True, "text": title}}
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
