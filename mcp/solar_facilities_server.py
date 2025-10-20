import pandas as pd
from fastmcp import FastMCP
from typing import List, Optional, Dict, Any
import folium
import os
import json
import hashlib
from solar_db import SolarDatabase

mcp = FastMCP("solar-facilities-server")

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize database connection
try:
    db = SolarDatabase()
    print(f"Connected to solar facilities database with {db.get_total_count():,} facilities")
    metadata = {
        "Name": "Solar Facilities Server", 
        "Description": "Global solar facility data from TransitionZero Solar Asset Mapper (TZ-SAM) Q1 2025",
        "Version": "2.0.0 - Database Edition",
        "Author": "Climate Policy Radar Team",
        "Dataset": "TZ-SAM Q1 2025 - Complete Global Dataset (SQLite)",
        **db.get_metadata()
    }
except Exception as e:
    print(f"Error: Could not connect to solar facilities database: {e}")
    db = None
    metadata = {"error": "Database not available"}

def generate_and_save_geojson(facilities: List[Dict], identifier: str = "world", extra_metadata: Dict = None) -> Dict[str, Any]:
    """
    Generate GeoJSON file from facilities data and save to static/maps/.
    Returns URL, filename, and statistics about the data.
    """
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Color mapping by country
    country_colors = {
        'brazil': '#4CAF50',
        'india': '#FF9800',
        'south africa': '#F44336',
        'vietnam': '#2196F3',
        'china': '#9C27B0',
        'united states of america': '#FF5722',
        'japan': '#3F51B5',
        'germany': '#009688',
        'spain': '#FFC107',
        'italy': '#795548'
    }
    
    # Statistics tracking
    total_capacity = 0
    capacities = []
    countries = set()
    country_counts = {}
    geometry_type = "point"  # Facilities are rendered as point features for maps
    # Track geographic bounds (min/max lat/lon)
    min_lat = None
    max_lat = None
    min_lon = None
    max_lon = None
    
    for facility in facilities:
        # Track statistics
        country = facility.get('country', 'Unknown')
        countries.add(country)
        country_counts[country] = country_counts.get(country, 0) + 1
        capacity = facility.get('capacity_mw')
        if capacity is not None and capacity > 0:
            capacities.append(capacity)
            total_capacity += capacity
        
        # Update bounds
        try:
            lat = float(facility['latitude'])
            lon = float(facility['longitude'])
            min_lat = lat if min_lat is None else min(min_lat, lat)
            max_lat = lat if max_lat is None else max(max_lat, lat)
            min_lon = lon if min_lon is None else min(min_lon, lon)
            max_lon = lon if max_lon is None else max(max_lon, lon)
        except Exception:
            pass

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(facility['longitude']), float(facility['latitude'])]
            },
            "properties": {
                "name": facility.get('name', f"Solar Facility {facility.get('cluster_id', '')}"),
                "capacity_mw": capacity if capacity is not None else 0.0,
                "country": facility.get('country', 'Unknown'),
                "cluster_id": facility.get('cluster_id', ''),
                "technology": "Solar PV",
                "constructed_before": facility.get('constructed_before'),
                "constructed_after": facility.get('constructed_after'),
                "marker_color": country_colors.get(facility.get('country', '').lower(), "#9E9E9E")
            }
        }
        geojson["features"].append(feature)
    
    # Calculate statistics
    stats = {
        "total_facilities": len(facilities),
        "total_capacity_mw": round(total_capacity, 1) if total_capacity > 0 else None,
        "countries": sorted(list(countries)),
        "geometry_type": geometry_type,
        "geometry_types": [geometry_type],
        "capacity_range_mw": {
            "min": round(min(capacities), 1) if capacities else None,
            "max": round(max(capacities), 1) if capacities else None,
            "average": round(sum(capacities) / len(capacities), 1) if capacities else None
        } if capacities else None,
        "bounds": (
            {
                "north": max_lat,
                "south": min_lat,
                "east": max_lon,
                "west": min_lon,
            }
            if None not in (min_lat, max_lat, min_lon, max_lon) else None
        )
    }
    
    # Save GeoJSON file
    try:
        project_root = os.path.dirname(script_dir)
        static_maps_dir = os.path.join(project_root, "static", "maps")
        os.makedirs(static_maps_dir, exist_ok=True)
        
        # Create unique filename
        identifier_str = identifier.lower().replace(" ", "_").replace(",", "")[:50]
        data_hash = hashlib.md5(f"{identifier_str}_{len(facilities)}".encode()).hexdigest()[:8]
        filename = f"solar_facilities_{identifier_str}_{data_hash}.geojson"
        
        geojson_path = os.path.join(static_maps_dir, filename)
        
        # Add metadata to GeoJSON
        if extra_metadata:
            geojson["metadata"] = extra_metadata
        # Include per-country counts to support front-end legends expecting counts
        stats_with_counts = {**stats, "country_counts": country_counts}
        geojson["metadata"] = {**geojson.get("metadata", {}), **stats_with_counts}
        
        with open(geojson_path, 'w') as f:
            json.dump(geojson, f)
        
        print(f"Generated GeoJSON file: {filename} with {len(geojson['features'])} features")
        
        return {
            "success": True,
            "geojson_url": f"/static/maps/{filename}",
            "geojson_filename": filename,
            "file_size_kb": round(os.path.getsize(geojson_path) / 1024, 1),
            "stats": stats,
            "geometry_type": geometry_type,
            "geometry_types": [geometry_type]
        }
        
    except Exception as e:
        print(f"Error generating GeoJSON file: {e}")
        return {
            "success": False,
            "error": str(e),
            "stats": stats,
            "geometry_type": geometry_type,
            "geometry_types": [geometry_type]
        }

@mcp.tool()
def GetTopNCountriesByFacilities(n: int = 10) -> Dict[str, Any]:
    """Get top N countries by number of solar facilities."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        country_stats = db.get_country_statistics()
        
        if not country_stats:
            return {"error": "No country statistics available"}
        
        # Sort by facility count
        country_stats.sort(key=lambda x: x['facility_count'], reverse=True)
        
        top_countries = country_stats[:n]
        
        return {
            "top_n": n,
            "countries": top_countries,
            "data_available": True,
            "note": f"Top {n} countries by number of solar facilities"
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetTopNCountriesByCapacity(n: int = 10) -> Dict[str, Any]:
    """Get top N countries by total solar capacity (capacity data not available in current schema)."""
    if not db:
        return {"error": "Database not available"}
    
    return {
        "error": "Capacity data not available - current database schema doesn't include capacity_mw field",
        "available_fields": ["cluster_id", "source_id", "source", "source_date", "latitude", "longitude", "country"],
        "suggestion": "Use GetSolarFacilitiesByCountry or GetSolarFacilitiesInRadius instead",
        "note": "Original TZ-SAM dataset doesn't include capacity information in the raw polygons file"
    }

@mcp.tool()
def GetSolarFacilitiesByCountry(country: str, min_capacity_mw: float = 0, limit: int = 10000) -> Dict[str, Any]:
    """Get solar facilities summary for a country.
    Default limit of 10,000 allows for complete country datasets (Brazil has ~2,273 facilities)."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        facilities = db.get_facilities_by_country(country, limit=limit)
        
        if not facilities:
            return {"error": f"No facilities found for {country}"}
        
        # Return summary
        summary = {
            "country": country,
            "total_facilities": len(facilities),
            "sample_facilities": facilities[:3],  # First 3 as examples
            "coordinate_range": {
                "lat_min": min(f['latitude'] for f in facilities),
                "lat_max": max(f['latitude'] for f in facilities),
                "lon_min": min(f['longitude'] for f in facilities),
                "lon_max": max(f['longitude'] for f in facilities)
            },
            "data_available": True
        }
        
        return summary
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarFacilitiesMapData(country: Optional[str] = None, limit: int = 10000) -> Dict[str, Any]:
    """PRIMARY MAP FUNCTION: Get facilities data for map visualization - generates GeoJSON file.
    Returns GeoJSON URL for frontend consumption. 
    Default limit of 10,000 allows for complete country datasets (Brazil has ~2,273 facilities)."""
    print(f"DEBUG GetSolarFacilitiesMapData called with country={country}, limit={limit}")
    if not db:
        return {"error": "Database not available"}
    
    try:
        if country:
            facilities = db.get_facilities_by_country(country, limit=limit)
        else:
            facilities = db.search_facilities(limit=limit)
        
        if not facilities:
            return {"error": f"No facilities found{' for ' + country if country else ''}"}
        
        # Get country statistics for summary
        country_stats = db.get_country_statistics()
        total_facilities_global = sum(stat['facility_count'] for stat in country_stats)
        countries_global = [stat['country'] for stat in country_stats]
        
        # Generate GeoJSON file
        identifier = country if country else "world"
        geojson_result = generate_and_save_geojson(
            facilities, 
            identifier,
            extra_metadata={
                "data_source": "TZ-SAM Q1 2025",
                "query_limit": limit,
                "country_filter": country
            }
        )
        
        if not geojson_result["success"]:
            return {"error": f"Failed to generate GeoJSON: {geojson_result.get('error')}"}
        
        stats = geojson_result["stats"]
        
        # Create description for LLM
        description = f"Generated map with {stats['total_facilities']:,} solar facilities"
        if country:
            description += f" in {country}"
        if stats.get('total_capacity_mw'):
            description += f" totaling {stats['total_capacity_mw']:,.1f} MW"
        if stats.get('capacity_range_mw'):
            cap_range = stats['capacity_range_mw']
            if cap_range.get('min') and cap_range.get('max'):
                description += f" (range: {cap_range['min']}-{cap_range['max']} MW)"

        geometry_type = geojson_result.get("geometry_type", stats.get("geometry_type", "point"))
        geometry_types = geojson_result.get("geometry_types", stats.get("geometry_types") or [geometry_type])

        result = {
            "type": "map_data_summary",
            "summary": {
                "description": description,
                "total_facilities": stats['total_facilities'],
                "total_capacity_mw": stats.get('total_capacity_mw'),
                "capacity_range_mw": stats.get('capacity_range_mw'),
                "countries": stats['countries'],
                "facilities_shown_on_map": stats['total_facilities'],
                "bounds": stats.get('bounds'),
                "geometry_type": geometry_type,
                "geometry_types": geometry_types,
                "global_context": {
                    "total_facilities_global": total_facilities_global,
                    "countries_with_data": countries_global[:10]
                },
                "data_from_database": True
            },
            "geojson_url": geojson_result["geojson_url"],
            "geojson_filename": geojson_result["geojson_filename"],
            "file_size_kb": geojson_result["file_size_kb"],
            "sample_facilities": facilities[:3],  # First 3 as examples
            "metadata": {
                "data_source": "TZ-SAM Q1 2025 (SQLite Database)",
                "geojson_available": True,
                "query_time_optimized": True,
                "note": "Full facility data saved to GeoJSON file",
                "bounds": stats.get('bounds'),
                "geometry_type": geometry_type,
                "geometry_types": geometry_types
            },
            "geometry_type": geometry_type,
            "geometry_types": geometry_types
        }
        print(f"DEBUG GetSolarFacilitiesMapData returning keys: {list(result.keys())}, NO full_data present")
        return result
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetFacilitiesForGeospatial(country: Optional[str] = None, limit: int = 10000) -> Dict[str, Any]:
    """Return condensed facility entities for direct geospatial registration.

    Fields: id/cluster_id, latitude, longitude, country, capacity_mw.
    """
    if not db:
        return {"error": "Database not available"}
    try:
        if country:
            facilities = db.get_facilities_by_country(country, limit=limit)
        else:
            facilities = db.search_facilities(limit=limit)
        if not facilities:
            return {"error": f"No facilities found{' for ' + country if country else ''}"}
        entities = []
        for f in facilities:
            try:
                entities.append({
                    "id": f.get('cluster_id') or f.get('id'),
                    "cluster_id": f.get('cluster_id'),
                    "latitude": float(f['latitude']),
                    "longitude": float(f['longitude']),
                    "country": f.get('country'),
                    "capacity_mw": f.get('capacity_mw')
                })
            except Exception:
                continue
        return {
            "entity_type": "solar_facility",
            "country_filter": country,
            "limit": limit,
            "entities": entities,
            "count": len(entities)
        }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarFacilitiesForGeoJSON(country: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """Get facilities for GeoJSON generation - generates file and returns reference."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        if country:
            facilities = db.get_facilities_by_country(country, limit=limit)
        else:
            facilities = db.search_facilities(limit=limit)
        
        if not facilities:
            return {"error": f"No facilities found{' for ' + country if country else ''}"}
        
        # Generate GeoJSON file
        identifier = country if country else "world"
        geojson_result = generate_and_save_geojson(
            facilities, 
            identifier,
            extra_metadata={
                "data_source": "TZ-SAM Q1 2025 Database",
                "query_limit": limit,
                "country_filter": country,
                "note": "Combined raw and analysis dataset with capacity where available"
            }
        )
        
        if not geojson_result["success"]:
            return {"error": f"Failed to generate GeoJSON: {geojson_result.get('error')}"}
        
        stats = geojson_result["stats"]
        
        return {
            "type": "map",
            "geometry_type": "point",
            "geojson_url": geojson_result["geojson_url"],
            "geojson_filename": geojson_result["geojson_filename"],
            "metadata": {
                "total_facilities": stats['total_facilities'],
                "countries": stats['countries'],
                "total_capacity_mw": stats.get('total_capacity_mw'),
                "data_source": "TZ-SAM Q1 2025 Database",
                "note": "GeoJSON file generated with full facility data"
            }
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarCapacityByCountry() -> Dict[str, Any]:
    """Get solar facility counts by country."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        country_stats = db.get_country_statistics()
        
        if not country_stats:
            return {"error": "No country statistics available"}
        
        # Sort by facility count
        country_stats.sort(key=lambda x: x['facility_count'], reverse=True)
        
        total_facilities = sum(stat['facility_count'] for stat in country_stats)
        top_country = country_stats[0]
        
        return {
            "countries_analyzed": [stat['country'] for stat in country_stats],
            "total_global_facilities": total_facilities,
            "total_countries": len(country_stats),
            "country_with_most_facilities": top_country['country'],
            "top_10_countries": country_stats[:10],
            "data_available": True,
            "query_optimized": True
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetDataForDirectVisualization(request_type: str, country: Optional[str] = None) -> str:
    """Get data ID for direct visualization."""
    if not db:
        return "NO_DATA_AVAILABLE"
    
    # Generate a simple identifier that the client can interpret
    if request_type == "country_comparison":
        return "SOLAR_COUNTRY_STATS"
    elif request_type == "facilities_map":
        country_filter = country or 'all'
        return f"SOLAR_MAP_{country_filter.upper()}"
    elif request_type == "capacity_distribution":
        return "SOLAR_CAPACITY_DIST"
    elif request_type == "timeline":
        return "SOLAR_TIMELINE"
    else:
        return f"SOLAR_DATA_{request_type.upper()}"

@mcp.tool()
def GetSolarFacilitiesInRadius(latitude: float, longitude: float, radius_km: float = 50, country: Optional[str] = None) -> Dict[str, Any]:
    """Find facilities within radius of coordinates."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        facilities = db.get_facilities_in_radius(
            latitude, longitude, radius_km, country, limit=100
        )
        
        if not facilities:
            return {"error": f"No facilities found within {radius_km}km of ({latitude}, {longitude})"}
        
        return {
            "search_center": [latitude, longitude],
            "radius_km": radius_km,
            "facilities_found": len(facilities),
            "closest_facilities": facilities[:5],  # Top 5 closest
            "data_available": True,
            "query_optimized": True
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarConstructionTimeline(start_year: int = 2010, end_year: int = 2030, country: Optional[str] = None, countries: Optional[List[str]] = None) -> Dict[str, Any]:
    """Get construction timeline based on constructed_after/constructed_before timestamps.

    Heuristics:
    - Prefer constructed_after (earliest known commissioning) if available
    - Fallback to constructed_before
    - If neither present, fallback to source_date when available
    Returns an annual count series under `timeline_data` for downstream viz.
    Accepts either a single `country` or list `countries`.
    """
    if not db:
        return {"error": "Database not available"}

    try:
        # Gather facilities for the requested scope
        if countries and not country:
            facilities: List[Dict[str, Any]] = []
            for c in countries:
                facilities.extend(db.get_facilities_by_country(c, limit=3000))
        elif country:
            facilities = db.get_facilities_by_country(country, limit=10000)
        else:
            facilities = db.search_facilities(limit=10000)

        if not facilities:
            scope = f" for {country}" if country else (f" for {', '.join(countries)}" if countries else "")
            return {"error": f"No facilities found{scope}"}

        # Build year for each facility using constructed_after/before/source_date
        per_facility_years: List[Tuple[int, Dict[str, Any]]] = []
        for f in facilities:
            year_val = None
            # Try constructed_after
            ca = f.get("constructed_after")
            cb = f.get("constructed_before")
            sd = f.get("source_date")
            try:
                if ca:
                    y = pd.to_datetime(ca, errors='coerce')
                    if pd.notnull(y):
                        year_val = int(y.year)
                if year_val is None and cb:
                    y = pd.to_datetime(cb, errors='coerce')
                    if pd.notnull(y):
                        year_val = int(y.year)
                if year_val is None and sd:
                    y = pd.to_datetime(sd, errors='coerce')
                    if pd.notnull(y):
                        year_val = int(y.year)
            except Exception:
                year_val = None

            if year_val is not None and start_year <= year_val <= end_year:
                per_facility_years.append((year_val, f))

        if not per_facility_years:
            return {"error": f"No facilities with valid timestamps between {start_year}-{end_year}"}

        # Aggregate counts by year (and optionally by country when multiple)
        year_counts: Dict[int, int] = {}
        for y, _ in per_facility_years:
            year_counts[y] = year_counts.get(y, 0) + 1

        # Prepare timeline_data list for downstream viz/table modules
        years_sorted = sorted(year_counts.keys())
        timeline_data = [{"year": y, "facilities": year_counts[y]} for y in years_sorted]

        result: Dict[str, Any] = {
            "period": f"{start_year}-{end_year}",
            "country_filter": country,
            "total_facilities_in_period": int(sum(year_counts.values())),
            "years_with_data": years_sorted,
            "yearly_counts": year_counts,
            "timeline_data": timeline_data,  # primary data payload for viz
            "data": timeline_data,           # alias to support generic consumers
            "data_available": True,
            "note": "Based on constructed_after/constructed_before with fallback to source_date"
        }

        # If multiple countries requested, also return per-country breakdown
        if countries and not country:
            # country -> year -> count
            per_country: Dict[str, Dict[int, int]] = {}
            for y, fac in per_facility_years:
                ctry = fac.get("country", "Unknown")
                per_country.setdefault(ctry, {})
                per_country[ctry][y] = per_country[ctry].get(y, 0) + 1

            per_country_timeline = {
                c: [{"year": y, "facilities": cnt} for y, cnt in sorted(yc.items())]
                for c, yc in per_country.items()
            }
            result["per_country_timeline"] = per_country_timeline

        return result

    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetLargestSolarFacilities(limit: int = 20, country: Optional[str] = None) -> Dict[str, Any]:
    """Get largest solar facilities by capacity - generates GeoJSON for mapping."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        facilities = db.get_largest_facilities(limit=limit, country=country)
        
        if not facilities:
            return {"error": f"No facilities found{' in ' + country if country else ''}"}
        
        # Generate GeoJSON file for these large facilities
        identifier = f"largest_{country}" if country else "largest_global"
        geojson_result = generate_and_save_geojson(
            facilities, 
            identifier,
            extra_metadata={
                "data_source": "TZ-SAM Q1 2025",
                "query_type": "largest_facilities",
                "limit": limit,
                "country_filter": country
            }
        )
        
        # Prepare facility summaries
        facilities_summary = []
        for i, facility in enumerate(facilities[:10], 1):  # Top 10 for summary
            facilities_summary.append({
                "rank": i,
                "cluster_id": facility.get('cluster_id', ''),
                "capacity_mw": facility.get('capacity_mw'),
                "country": facility['country'],
                "name": f"Solar Facility {facility.get('cluster_id', '')}"
            })
        
        stats = geojson_result["stats"] if geojson_result["success"] else {}
        
        return {
            "search_criteria": {"limit": limit, "country": country},
            "facilities_found": len(facilities),
            "top_10_facilities": facilities_summary,
            "capacity_stats": {
                "total_mw": stats.get('total_capacity_mw'),
                "range_mw": stats.get('capacity_range_mw')
            },
            "type": "map",
            "geometry_type": "point",
            "geojson_url": geojson_result.get("geojson_url") if geojson_result["success"] else None,
            "geojson_filename": geojson_result.get("geojson_filename") if geojson_result["success"] else None,
            "note": "Sorted by capacity (MW) - full list available in GeoJSON file",
            "data_available": True
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarFacilityDetails(cluster_id: str) -> Dict[str, Any]:
    """Get details for specific facility."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        facility = db.get_facility_by_id(cluster_id)
        
        if not facility:
            return {"error": f"Facility with cluster_id '{cluster_id}' not found"}
        
        # Add derived information if source_date is available
        if facility.get('source_date'):
            try:
                source_date = pd.to_datetime(facility['source_date'])
                facility['source_year'] = source_date.year
                facility['source_date_formatted'] = source_date.strftime('%Y-%m-%d')
            except Exception as e:
                facility['date_parse_error'] = str(e)
        
        return facility
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def SearchSolarFacilitiesByCapacity(min_capacity_mw: float, max_capacity_mw: float, country: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """Search facilities by capacity range (capacity data not available in current schema)."""
    if not db:
        return {"error": "Database not available"}
    
    return {
        "error": "Capacity search not available - current database schema doesn't include capacity_mw field",
        "available_fields": ["cluster_id", "source_id", "source", "source_date", "latitude", "longitude", "country"],
        "suggestion": "Use GetSolarFacilitiesByCountry or GetSolarFacilitiesInRadius instead",
        "note": "Original TZ-SAM dataset doesn't include capacity information in the raw polygons file"
    }

# CreateSolarFacilitiesMap tool removed - was causing context explosion
# Maps are now generated purely client-side

@mcp.tool()
def GetSolarCapacityVisualizationData(
    visualization_type: str = "by_country",
    metric: str = "facility_count",
    top_n: int = 20
) -> Dict[str, Any]:
    """Get data for visualization. Types: by_country, source_timeline.

    For by_country, supports metric "facility_count" (default) or "total_capacity_mw" and optional top_n limit.
    """
    if not db:
        return {"error": "Database not available"}
    
    try:
        if visualization_type == "by_country":
            # Country-level statistics
            country_stats = db.get_country_statistics()
            # Validate metric
            y_key = metric if metric in ["facility_count", "total_capacity_mw"] else "facility_count"
            # Sort and limit
            data_sorted = sorted(country_stats, key=lambda d: (d.get(y_key) or 0), reverse=True)
            data_limited = data_sorted[:max(1, int(top_n))]
            title_metric = "Facility Count" if y_key == "facility_count" else "Total Capacity (MW)"
            return {
                "visualization_type": "by_country",
                "data": data_limited,
                "chart_config": {
                    "x_axis": "country",
                    "y_axis": y_key, 
                    "title": f"Solar by Country — {title_metric}",
                    "chart_type": "bar"
                }
            }
        
        elif visualization_type == "source_timeline":
            # Timeline based on source dates - use unlimited data for accurate charts
            facilities = db.search_facilities(limit=None)  # No limit for chart accuracy
            
            # Group by source and year
            timeline_data = {}
            for facility in facilities:
                if facility.get('source_date'):
                    try:
                        source_date = pd.to_datetime(facility['source_date'])
                        year = source_date.year
                        source = facility.get('source', 'Unknown')
                        
                        key = f"{source}_{year}"
                        if key not in timeline_data:
                            timeline_data[key] = {
                                "source": source,
                                "year": year,
                                "facility_count": 0
                            }
                        timeline_data[key]["facility_count"] += 1
                    except:
                        continue
            
            data = list(timeline_data.values())
            
            return {
                "visualization_type": "source_timeline",
                "data": data,
                "chart_config": {
                    "x_axis": "year",
                    "y_axis": "facility_count",
                    "color": "source",
                    "title": "Solar Facilities by Source and Year",
                    "chart_type": "line"
                }
            }
        
        else:
            return {"error": f"Unknown visualization type: {visualization_type}. Available: by_country, source_timeline"}
            
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

# DEPRECATED: HTML map generation - use GetSolarFacilitiesMapData for GeoJSON URLs instead
# @mcp.tool()
def GetSolarFacilitiesMap_DEPRECATED(country: Optional[str] = None, limit: int = 10000) -> Dict[str, Any]:
    """DEPRECATED: Generate an interactive HTML map of solar facilities.
    Use GetSolarFacilitiesMapData instead for GeoJSON URL generation."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        if country:
            facilities = db.get_facilities_by_country(country, limit=limit)
        else:
            facilities = db.search_facilities(limit=limit)
        
        if not facilities:
            return {"error": f"No facilities found{' for ' + country if country else ''}"}
        
        # Calculate map center based on facilities
        lats = [f['latitude'] for f in facilities if f['latitude']]
        lons = [f['longitude'] for f in facilities if f['longitude']]
        
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
        
        # Add markers
        for facility in facilities:
            if facility.get('latitude') and facility.get('longitude'):
                popup_text = f"""
                <b>Capacity:</b> {facility.get('capacity_mw', 'Unknown')} MW<br>
                <b>Country:</b> {facility.get('country', 'Unknown')}<br>
                <b>Cluster ID:</b> {facility.get('cluster_id', 'Unknown')}
                """
                
                folium.CircleMarker(
                    location=[facility['latitude'], facility['longitude']],
                    radius=5,
                    popup=popup_text,
                    color='red',
                    fillColor='orange',
                    fillOpacity=0.7
                ).add_to(m)
        
        # Save to temp file and get HTML
        import uuid
        
        temp_id = str(uuid.uuid4())[:8]
        temp_file = f"/tmp/solar_map_{temp_id}.html"
        m.save(temp_file)
        
        with open(temp_file, 'r') as f:
            map_html = f.read()
        
        # Clean up temp file
        os.remove(temp_file)
        
        return {
            "type": "interactive_map",
            "country_filter": country,
            "facilities_count": len(facilities),
            "map_html": map_html,
            "center": [center_lat, center_lon],
            "data_available": True
        }
        
    except Exception as e:
        return {"error": f"Map generation failed: {str(e)}"}

@mcp.tool()
def GetSolarFacilitiesInBounds(north: float, south: float, east: float, west: float, limit: int = 500) -> Dict[str, Any]:
    """Get solar facilities within geographic bounds - generates GeoJSON."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        facilities = db.get_facilities_in_bounds(north, south, east, west, limit=limit)
        
        if not facilities:
            return {"error": f"No facilities found in bounds: N{north} S{south} E{east} W{west}"}
        
        # Generate GeoJSON file
        identifier = f"bounds_n{north:.1f}_s{south:.1f}_e{east:.1f}_w{west:.1f}"
        geojson_result = generate_and_save_geojson(
            facilities, 
            identifier,
            extra_metadata={
                "data_source": "TZ-SAM Q1 2025",
                "query_type": "geographic_bounds",
                "bounds": {"north": north, "south": south, "east": east, "west": west}
            }
        )
        
        # Group by country for summary
        countries = {}
        for facility in facilities:
            country = facility.get('country', 'Unknown')
            if country not in countries:
                countries[country] = 0
            countries[country] += 1
        
        stats = geojson_result["stats"] if geojson_result["success"] else {}
        
        return {
            "bounds": {"north": north, "south": south, "east": east, "west": west},
            "facilities_found": len(facilities),
            "countries_in_bounds": list(countries.keys()),
            "facilities_by_country": countries,
            "capacity_stats": {
                "total_mw": stats.get('total_capacity_mw'),
                "range_mw": stats.get('capacity_range_mw')
            },
            "sample_facilities": facilities[:5],
            "type": "map",
            "geometry_type": "point",
            "geojson_url": geojson_result.get("geojson_url") if geojson_result["success"] else None,
            "geojson_filename": geojson_result.get("geojson_filename") if geojson_result["success"] else None,
            "data_available": True
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarFacilitiesMultipleCountries(countries: List[str], limit: int = 100000) -> Dict[str, Any]:
    """Get solar facilities for multiple countries - generates GeoJSON."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        all_facilities = []
        country_results = {}
        
        for country in countries:
            facilities = db.get_facilities_by_country(country, limit=limit//len(countries))
            all_facilities.extend(facilities)
            country_results[country] = len(facilities)
        
        if not all_facilities:
            return {"error": f"No facilities found for countries: {', '.join(countries)}"}
        
        # Generate GeoJSON file
        identifier = "_".join(c.lower().replace(" ", "")[:10] for c in countries[:3])  # First 3 countries
        geojson_result = generate_and_save_geojson(
            all_facilities, 
            identifier,
            extra_metadata={
                "data_source": "TZ-SAM Q1 2025",
                "query_type": "multiple_countries",
                "countries": countries
            }
        )
        
        stats = geojson_result["stats"] if geojson_result["success"] else {}
        
        return {
            "countries_requested": countries,
            "total_facilities": len(all_facilities),
            "facilities_by_country": country_results,
            "capacity_stats": {
                "total_mw": stats.get('total_capacity_mw'),
                "range_mw": stats.get('capacity_range_mw')
            },
            "sample_facilities": all_facilities[:5],
            "type": "map",
            "geometry_type": "point",
            "geojson_url": geojson_result.get("geojson_url") if geojson_result["success"] else None,
            "geojson_filename": geojson_result.get("geojson_filename") if geojson_result["success"] else None,
            "data_available": True
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarFacilitiesCountries() -> Dict[str, Any]:
    """Get list of all countries with solar facilities in the database."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        all_countries = db.get_all_country_names()
        country_stats = db.get_country_statistics()
        
        # Create a lookup for facility counts
        country_counts = {stat['country']: stat['facility_count'] for stat in country_stats}
        
        countries_with_counts = [
            {
                "country": country,
                "facility_count": country_counts.get(country, 0)
            }
            for country in all_countries
        ]
        
        # Sort by facility count descending
        countries_with_counts.sort(key=lambda x: x['facility_count'], reverse=True)
        
        return {
            "total_countries": len(all_countries),
            "countries": countries_with_counts,
            "top_10_countries": countries_with_counts[:10],
            "data_available": True,
            "note": "Use exact country names for filtering. Case insensitive matching supported."
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def FindSolarFacilitiesCountries(partial_name: str) -> Dict[str, Any]:
    """Find countries with solar facilities by partial name match."""
    if not db:
        return {"error": "Database not available"}
    
    try:
        matching_countries = db.find_country_by_partial_name(partial_name)
        
        if not matching_countries:
            return {"error": f"No countries found matching '{partial_name}'"}
        
        return {
            "search_term": partial_name,
            "matching_countries": matching_countries,
            "match_count": len(matching_countries),
            "data_available": True
        }
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool()
def GetSolarDatasetMetadata() -> Dict[str, Any]:
    """Get dataset metadata."""
    if not db:
        return {"error": "Database not available"}
    
    return metadata

@mcp.tool()
def DescribeServer() -> Dict[str, Any]:
    """Describe this server, its dataset, key tools, and live metrics."""
    if not db:
        return {"error": "Database not available"}
    m = metadata.copy()
    tools = [
        "GetSolarFacilitiesByCountry",
        "GetSolarCapacityByCountry",
        "GetSolarFacilitiesMapData",
        "GetSolarConstructionTimeline",
        "GetSolarFacilitiesInBounds",
        "GetSolarFacilitiesMultipleCountries",
        "FindSolarFacilitiesCountries",
        "GetSolarDatasetMetadata"
    ]
    # Derive last_updated from database file mtime if present
    last_updated = None
    try:
        db_path = m.get("database_path")
        if db_path and os.path.exists(db_path):
            import datetime as _dt
            last_updated = _dt.datetime.fromtimestamp(os.path.getmtime(db_path)).isoformat()
    except Exception:
        pass
    return {
        "name": m.get("Name", "Solar Facilities Server"),
        "description": m.get("Description", "Global solar facility data"),
        "version": m.get("Version"),
        "dataset": m.get("Dataset"),
        "metrics": {
            "total_facilities": m.get("total_facilities"),
            "total_countries": m.get("total_countries"),
            "total_capacity_mw": m.get("total_capacity")
        },
        "coverage": {
            "countries": m.get("countries", [])
        },
        "tools": tools,
        "examples": [
            "Map solar facilities in India",
            "Top countries by solar facility count",
            "Show construction timeline for Brazil"
        ],
        "source": m.get("data_source", "TZ-SAM Q1 2025"),
        "last_updated": last_updated
    }

if __name__ == "__main__":
    mcp.run()
