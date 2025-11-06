import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from fastmcp import FastMCP
from shapely import wkb, wkt
from shapely.geometry import box

try:  # Optional dotenv support
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

try:  # Optional dependency for LLM-backed routing
    import anthropic  # type: ignore
except ImportError:
    anthropic = None

if load_dotenv:  # pragma: no cover - best-effort env loading
    try:
        load_dotenv()
    except Exception as exc:
        print(f"[solar_clay] Warning: load_dotenv failed: {exc}", file=sys.stderr)

mcp = FastMCP("solar-clay-server")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SOLAR_CHIP_DATA_PATH = PROJECT_ROOT / "data" / "top_potential_solar.parquet"
BRAZILIAN_STATES_PATH = (
    PROJECT_ROOT / "data" / "brazilian_admin" / "brazilian_states.geojson"
)
STATIC_MAPS_DIR = PROJECT_ROOT / "static" / "maps"
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"


class DatasetUnavailable(RuntimeError):
    """Raised when the Clay solar dataset cannot be accessed."""


def _create_anthropic_client() -> Optional["anthropic.Anthropic"]:
    if not anthropic or not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        return anthropic.Anthropic()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: Failed to initialize Anthropic client: {exc}", file=sys.stderr)
        return None


ANTHROPIC_CLIENT = _create_anthropic_client()


def convert_to_json_serializable(obj: Any) -> Any:
    """Best-effort conversion for numpy/pandas types."""
    if isinstance(obj, np.ndarray):
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    return obj


@lru_cache(maxsize=1)
def _load_states() -> Optional[gpd.GeoDataFrame]:
    if not BRAZILIAN_STATES_PATH.exists():
        return None
    try:
        return gpd.read_file(BRAZILIAN_STATES_PATH)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: Failed to load Brazilian states: {exc}", file=sys.stderr)
        return None


def _convert_geometry_column(df: pd.DataFrame) -> pd.DataFrame:
    if "geometry" not in df.columns or df.empty:
        return df

    sample = df["geometry"].iloc[0]
    if isinstance(sample, bytes):
        df["geometry"] = df["geometry"].apply(lambda value: wkb.loads(value) if value else None)
    elif isinstance(sample, str):
        df["geometry"] = df["geometry"].apply(lambda value: wkt.loads(value) if value else None)
    return df


@lru_cache(maxsize=1)
def _load_dataset() -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Dict[str, Any]]:
    states_gdf = _load_states()

    if not SOLAR_CHIP_DATA_PATH.exists():
        metadata = {
            "error": f"Solar data file not found at {SOLAR_CHIP_DATA_PATH}",
            "Name": "Solar Geometries Server",
            "Description": "Top potential sites for solar development generated from Clay embeddings",
            "Version": "1.0.0",
            "Author": "Ode Partners",
            "Dataset": "Top potential Solar Non-Intersecting Geometries",
        }
        return None, states_gdf, metadata

    try:
        df = pd.read_parquet(SOLAR_CHIP_DATA_PATH)
        df = _convert_geometry_column(df)
        solar_gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        if states_gdf is not None and not states_gdf.empty:
            joined = gpd.sjoin(
                solar_gdf,
                states_gdf[["NM_UF", "geometry"]],
                how="left",
                predicate="intersects",
            )
            if "index_right" in joined.columns:
                joined = joined.drop(columns="index_right")
            if "NM_UF" in joined.columns:
                joined = joined.rename(columns={"NM_UF": "state"})
            solar_gdf = joined

        metadata = {
            "Name": "Solar Geometries Server",
            "Description": "Top potential sites where solar farms could be installed based on Clay embeddings and PVGIS data",
            "Version": "1.0.0",
            "Author": "Ode Partners",
            "Dataset": "Top potential Solar Non-Intersecting Geometries",
            "total_geometries": len(solar_gdf),
            "columns": list(solar_gdf.columns),
            "has_yield_data": "specific_yield_kwh_per_kwp_yr" in solar_gdf.columns,
        }

        if "state" in solar_gdf.columns:
            metadata["total_states"] = int(solar_gdf["state"].dropna().nunique())

        return solar_gdf, states_gdf, metadata

    except Exception as exc:
        metadata = {"error": f"Data loading failed: {exc}"}
        return None, states_gdf, metadata


def _get_dataset_or_raise() -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Dict[str, Any]]:
    solar_gdf, states_gdf, metadata = _load_dataset()
    if solar_gdf is None:
        raise DatasetUnavailable(metadata.get("error", "Solar data not available"))
    return solar_gdf, states_gdf, metadata


def _determine_point_columns(gdf: gpd.GeoDataFrame, include_columns: Optional[Sequence[str]]) -> List[str]:
    if include_columns:
        missing = [col for col in include_columns if col not in gdf.columns]
        if missing:
            raise KeyError(f"Columns not found in data: {', '.join(missing)}")
        return list(include_columns)

    preferred = [
        "cluster_id",
        "state",
        "specific_yield_kwh_per_kwp_yr",
        "potential_mwp",
        "mean_ghi",
        "score",
    ]
    columns = [col for col in preferred if col in gdf.columns]
    if columns:
        return columns

    fallback = [col for col in gdf.columns if col != "geometry"]
    return fallback[: min(8, len(fallback))]


def _point_records_from_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    include_columns: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if gdf is None or gdf.empty:
        return [], {"total_points": 0}

    working = gdf.dropna(subset=["geometry"]).copy()
    if working.empty:
        return [], {"total_points": 0}

    if limit is not None and limit > 0:
        working = working.head(limit)

    columns = _determine_point_columns(working, include_columns)

    records: List[Dict[str, Any]] = []
    latitudes: List[float] = []
    longitudes: List[float] = []

    for idx, row in working.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        try:
            centroid = geom.centroid
        except Exception:
            continue

        lat = float(centroid.y)
        lon = float(centroid.x)

        payload: Dict[str, Any] = {
            "id": str(row.get("cluster_id") or row.get("id") or idx),
            "latitude": lat,
            "longitude": lon,
        }
        for col in columns:
            payload[col] = convert_to_json_serializable(row.get(col))

        records.append(payload)
        latitudes.append(lat)
        longitudes.append(lon)

    summary: Dict[str, Any] = {
        "total_points": len(records),
        "included_columns": list(columns),
    }

    if records:
        summary["bounds"] = {
            "north": max(latitudes),
            "south": min(latitudes),
            "east": max(longitudes),
            "west": min(longitudes),
        }

    return records, summary


def _dataset_error_response(error: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"error": error}
    payload.update(extra)
    return payload


@mcp.tool()
def GetClayCandidatePoints(
    limit: Optional[int] = 500,
    include_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Retrieve centroid points for Clay solar candidate sites across all Brazilian states.

    Use this for queries requesting candidate sites across Brazil without state filtering.
    Returns point collection with latitude, longitude, and yield data.
    """
    try:
        solar_gdf, _, _ = _get_dataset_or_raise()
        points, summary = _point_records_from_gdf(
            solar_gdf,
            include_columns=include_columns,
            limit=limit,
        )
    except DatasetUnavailable as exc:
        return _dataset_error_response(str(exc))
    except KeyError as exc:
        return _dataset_error_response(str(exc))

    summary.update(
        {
            "limit_requested": limit,
            "data_source": "Solar Top potential",
        }
    )

    return {
        "type": "point_collection",
        "points": points,
        "summary": summary,
    }


@mcp.tool()
def GetClayCandidatePointsByState(
    state: str,
    limit: Optional[int] = 200,
    include_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Retrieve Clay solar candidate points filtered to a specific Brazilian state.

    Use this when the query mentions a specific state name like Santa Catarina, Minas Gerais,
    Rio Grande do Sul, São Paulo, etc. Returns filtered point collection for that state only.
    """
    try:
        solar_gdf, _, _ = _get_dataset_or_raise()
    except DatasetUnavailable as exc:
        return {"summary": f"Error: {exc}"}

    if "state" not in solar_gdf.columns:
        return {"summary": "State column not found in solar data"}

    subset = solar_gdf[solar_gdf["state"].str.lower() == state.lower()]
    if subset.empty:
        available = sorted(solar_gdf["state"].dropna().unique().tolist())
        return {"summary": f"No Clay solar candidates found for state '{state}'. Available states: {', '.join(available[:5])}"}

    try:
        points, point_summary = _point_records_from_gdf(
            subset,
            include_columns=include_columns,
            limit=limit,
        )
    except KeyError as exc:
        return {"summary": f"Error retrieving data: {exc}"}

    total = point_summary.get("total_points", 0)
    avg_yield = None
    if points and "specific_yield_kwh_per_kwp_yr" in points[0]:
        yields = [p.get("specific_yield_kwh_per_kwp_yr") for p in points if p.get("specific_yield_kwh_per_kwp_yr")]
        if yields:
            avg_yield = sum(yields) / len(yields)

    summary_text = f"Found {total} Clay solar candidate sites in {state}"
    if avg_yield:
        summary_text += f" with average solar yield of {avg_yield:.1f} kWh/kWp/yr"

    return {
        "citation": {
            "id": "clay-solar-state",
            "tool": "GetClayCandidatePointsByState",
            "title": "Clay Solar Site Potential Dataset",
            "source_type": "Dataset",
            "description": "AI-powered Earth observation analysis identifying optimal solar development sites",
            "url": "https://clay-foundation.github.io/model/",
        },
        "summary": summary_text,
        "artifacts": [],  # Could add a map here if needed
    }


@mcp.tool()
def GetClayCandidatePointsInBounds(
    north: float,
    south: float,
    east: float,
    west: float,
    limit: Optional[int] = 200,
    include_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Retrieve Clay solar candidate points within specified latitude/longitude bounding box.

    Use this when the query specifies geographic coordinates or a bounding box.
    Returns points where centroid falls within the specified bounds.
    """
    try:
        solar_gdf, _, _ = _get_dataset_or_raise()
    except DatasetUnavailable as exc:
        return _dataset_error_response(str(exc))

    if north < south:
        return _dataset_error_response("north bound must be >= south bound")
    if east < west:
        return _dataset_error_response("east bound must be >= west bound")

    bounds_polygon = box(west, south, east, north)
    try:
        centroids = solar_gdf.geometry.centroid
    except Exception as exc:
        return _dataset_error_response(f"Failed to compute centroids: {exc}")

    mask = centroids.apply(lambda pt: pt is not None and bounds_polygon.covers(pt))
    subset = solar_gdf[mask]

    if subset.empty:
        return _dataset_error_response(
            "No points found within the requested bounds",
            bounds={"north": north, "south": south, "east": east, "west": west},
        )

    try:
        points, summary = _point_records_from_gdf(
            subset,
            include_columns=include_columns,
            limit=limit,
        )
    except KeyError as exc:
        return _dataset_error_response(
            str(exc),
            available_columns=[col for col in subset.columns if col != "geometry"],
        )

    summary.update(
        {
            "bounds": {"north": north, "south": south, "east": east, "west": west},
            "limit_requested": limit,
            "data_source": "Solar Top potential",
        }
    )

    return {
        "type": "point_collection",
        "points": points,
        "summary": summary,
    }


@mcp.tool()
def GetGeometriesByState() -> Dict[str, Any]:
    """Get summary statistics of Clay candidate sites grouped by Brazilian state.

    Use this for queries asking about distribution across states, top states by count,
    or comparative state statistics. Returns counts and total area per state.
    """
    try:
        solar_gdf, _, _ = _get_dataset_or_raise()
    except DatasetUnavailable as exc:
        return {"summary": f"Error: {exc}"}

    if "state" not in solar_gdf.columns:
        return {"summary": "State column not found in solar data"}

    state_counts = solar_gdf["state"].value_counts().to_dict()
    state_stats: List[Dict[str, Any]] = []

    for state, count in sorted(state_counts.items(), key=lambda item: item[1], reverse=True):
        state_gdf = solar_gdf[solar_gdf["state"] == state]
        total_area = float(state_gdf.geometry.area.sum() * 111 * 111 * 100)
        state_stats.append(
            {
                "state": state,
                "geometry_count": int(count),
                "total_area_ha": round(total_area, 1),
            }
        )

    # Build summary text
    top_5 = state_stats[:5]
    top_states_text = ", ".join([f"{s['state']} ({s['geometry_count']} sites)" for s in top_5])
    summary_text = f"Clay solar candidate distribution across {len(state_counts)} Brazilian states. Top 5 states: {top_states_text}"

    # Build facts for each top state
    facts = []
    for i, s in enumerate(top_5, 1):
        facts.append(f"{s['state']} has {s['geometry_count']} Clay solar candidate sites covering approximately {s['total_area_ha']:,.0f} hectares")

    return {
        "citation": {
            "id": "clay-solar-stats",
            "tool": "GetGeometriesByState",
            "title": "Clay Solar Site Potential Dataset",
            "source_type": "Dataset",
            "description": "AI-powered Earth observation analysis identifying optimal solar development sites",
            "url": "https://clay-foundation.github.io/model/",
        },
        "summary": summary_text,
        "facts": facts,
    }


@mcp.tool()
def GetSolarDataSummary() -> Dict[str, Any]:
    """Get dataset-level statistics and metadata for the Clay solar candidate dataset.

    Use this for queries about dataset coverage, available fields, yield statistics,
    geographic bounds, or general dataset information.
    """
    try:
        solar_gdf, _, _ = _get_dataset_or_raise()
    except DatasetUnavailable as exc:
        return _dataset_error_response(str(exc))

    summary: Dict[str, Any] = {
        "total_geometries": len(solar_gdf),
        "columns": list(solar_gdf.columns),
        "geometry_types": solar_gdf.geometry.geom_type.unique().tolist(),
        "bounds": {
            "west": float(solar_gdf.total_bounds[0]),
            "south": float(solar_gdf.total_bounds[1]),
            "east": float(solar_gdf.total_bounds[2]),
            "north": float(solar_gdf.total_bounds[3]),
        },
    }

    if "specific_yield_kwh_per_kwp_yr" in solar_gdf.columns:
        stats = solar_gdf["specific_yield_kwh_per_kwp_yr"].describe()
        summary["yield_statistics"] = {
            "min": float(stats["min"]),
            "max": float(stats["max"]),
            "mean": float(stats["mean"]),
            "median": float(stats["50%"]),
            "std": float(stats["std"]),
        }

    if "state" in solar_gdf.columns:
        states = sorted(solar_gdf["state"].dropna().unique().tolist())
        summary["states"] = states
        summary["total_states"] = len(states)

    return summary


@mcp.tool()
def GetBrazilianStates() -> Dict[str, Any]:
    """List all Brazilian states available in the administrative boundaries data.

    Use this when the query asks for a list of states or available geographic coverage.
    """
    _, states_gdf, _ = _load_dataset()
    if states_gdf is None or states_gdf.empty:
        return _dataset_error_response("Brazilian states data not available")

    states = sorted(states_gdf["NM_UF"].dropna().unique().tolist())
    return {
        "total_states": len(states),
        "states": states,
        "data_available": True,
    }


@mcp.tool()
def GetClaySupport() -> Dict[str, Any]:
    """Get usage guidance and capabilities description for the Clay solar candidate dataset.

    Use this when the query asks about the dataset purpose, best use cases,
    available tools, or how to work with the Clay data.
    """
    _, _, metadata = _load_dataset()
    available_columns = metadata.get("columns", [])

    return {
        "dataset": "Clay Earth Observation Solar Candidate Set",
        "description": (
            "Spatial candidates generated via Clay's EO foundation model and PVGIS yield scoring. "
            "Each record represents a high-potential solar development polygon; this server exposes "
            "centroid points for lightweight visualisation."
        ),
        "best_for": [
            "Highlighting top Clay solar candidate locations by yield or score",
            "Filtering candidates to a single Brazilian state",
            "Requesting Clay candidates within custom latitude/longitude bounds",
        ],
        "not_suitable_for": [
            "Existing, operational solar facilities (use TZ-SAM / solar server)",
            "Large multi-polygon visualisations—use these points instead",
        ],
        "primary_tools": [
            "GetClayCandidatePoints",
            "GetClayCandidatePointsByState",
            "GetClayCandidatePointsInBounds",
            "GetGeometriesByState",
            "GetSolarDataSummary",
        ],
        "available_columns": [col for col in available_columns if col != "geometry"],
        "data_available": "error" not in metadata,
    }


@mcp.tool()
def DescribeServer() -> Dict[str, Any]:
    """Describe the Clay solar candidate server's capabilities and available tools.

    Use this for meta-queries about the server itself, available tools, or dataset provenance.
    """
    _, _, metadata = _load_dataset()
    tools = [
        "GetClayCandidatePoints - Retrieve centroid points for all Clay solar candidates",
        "GetClayCandidatePointsByState - Retrieve centroid points filtered to a specific state",
        "GetClayCandidatePointsInBounds - Retrieve centroid points within bounding coordinates",
        "GetGeometriesByState - Get count of geometries per state",
        "GetSolarDataSummary - Get dataset statistics and metadata",
        "GetBrazilianStates - Get list of Brazilian states",
        "GetClaySupport - Usage guidance for the Clay solar candidate dataset",
    ]

    return {
        "name": metadata.get("Name", "Solar Geometries Server"),
        "description": metadata.get(
            "Description",
            "Top potential Clay solar candidate geometries.",
        ),
        "version": metadata.get("Version"),
        "dataset": metadata.get("Dataset"),
        "metrics": {
            "total_geometries": metadata.get("total_geometries"),
            "total_states": metadata.get("total_states"),
        },
        "tools": tools,
        "examples": [
            "List the highest-yield Clay solar candidates and their locations",
            "Show Clay solar candidate points in Mato Grosso",
            "Find Clay solar candidate points within a given latitude/longitude box",
            "Summarise how to use the Clay solar candidate dataset",
        ],
        "data_available": "error" not in metadata,
    }


def _classify_query_with_llm(query: str) -> Tuple[bool, str]:
    if not ANTHROPIC_CLIENT:
        return False, "LLM unavailable; cannot determine support"

    prompt = (
        "You decide whether to route a question to the Clay Solar Candidate dataset. "
        "This dataset contains POTENTIAL solar development sites in Brazil identified via AI-powered Earth observation analysis. "
        "Route questions about: potential sites, candidate locations, where to build solar, good places for solar, "
        "future solar development opportunities, site suitability, or solar potential. "
        "DO NOT route questions about existing/operational solar facilities (those go to TZ-SAM dataset). "
        "Respond ONLY with JSON of the form "
        '{{"supported": true|false, "reason": "short explanation"}}. '
        "Question: {question}"
    ).format(question=query)

    try:
        response = ANTHROPIC_CLIENT.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=128,
            temperature=0,
            system="Respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
            text = text.strip()
        parsed = json.loads(text)
        # Handle keys with or without extra quotes
        supported = bool(parsed.get("supported") or parsed.get('"supported"'))
        reason = str(parsed.get("reason") or parsed.get('"reason"') or "").strip()
        return supported, reason or "LLM classified the query"
    except json.JSONDecodeError as exc:
        return False, f"LLM response parsing failed: {exc}"
    except Exception as exc:  # pragma: no cover - network/API failure
        return False, f"LLM classification failed: {exc}"


@mcp.tool()
def query_support(query: str, context: dict) -> str:
    """Decide whether the Clay solar candidate dataset can assist with the query.

    Used by the orchestrator for routing decisions. Returns support score and reasoning.
    """
    supported, reason = _classify_query_with_llm(query)
    payload = {
        "server": "solar_clay",
        "query": query,
        "supported": supported,
        "score": 0.9 if supported else 0.0,
        "reasons": [reason] if reason else [],
    }
    return json.dumps(payload)


def _build_geojson_artifact(solar_gdf: gpd.GeoDataFrame) -> Optional[Dict[str, Any]]:
    if solar_gdf.empty:
        return None

    STATIC_MAPS_DIR.mkdir(parents=True, exist_ok=True)

    features: List[Dict[str, Any]] = []
    for idx, row in solar_gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == "Point":
            point = geom
        else:
            try:
                point = geom.centroid
            except Exception:
                continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(point.x), float(point.y)],
                },
                "properties": {
                    "site_id": idx,
                    "state": row.get("state", "Unknown"),
                    "specific_yield_kwh_per_kwp_yr": row.get("specific_yield_kwh_per_kwp_yr"),
                    "layer": "clay_solar_candidate",
                    "color_value": 1,
                    "color_hex": "#4CAF50",
                    "color": "#4CAF50",
                },
            }
        )

    if not features:
        return None

    signature = hashlib.md5(f"clay_solar_candidates_{len(features)}".encode()).hexdigest()[:8]
    filename = f"clay_solar_candidates_brazil_{signature}.geojson"
    output_path = STATIC_MAPS_DIR / filename

    geojson_payload = {
        "type": "FeatureCollection",
        "features": features,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(geojson_payload, handle)

    bounds = solar_gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    return {
        "id": f"clay_map_{signature}",
        "type": "map",
        "title": "Clay Solar Candidate Sites in Brazil",
        "geojson_url": f"/static/maps/{filename}",
        "metadata": {
            "plotted_sites": len(features),
            "countries": ["Brazil"],
            "country_counts": {"Brazil": len(features)},
            "layers": {
                "clay_solar_candidate": {
                    "count": len(features),
                    "color": "#4CAF50",
                    "color_property": "color_value",
                    "fill_style": {
                        "type": "interpolate",
                        "color_property": "color_value",
                        "range": [0, 1],
                        "colorMin": "#4CAF50",
                        "colorMax": "#4CAF50",
                    },
                    "z_index": 1,
                }
            },
            "legend": {
                "title": "Layers",
                "items": [
                    {
                        "label": "Clay Solar Candidate",
                        "color": "#4CAF50",
                        "description": f"{len(features)} sites",
                    }
                ],
            },
            "bounds": {
                "west": float(bounds[0]),
                "east": float(bounds[2]),
                "south": float(bounds[1]),
                "north": float(bounds[3]),
            },
            "center": {
                "lon": float(center_lon),
                "lat": float(center_lat),
            },
            "geometry_type": "point",
        },
        "viewState": {
            "center": [float(center_lon), float(center_lat)],
            "bounds": {
                "west": float(bounds[0]),
                "east": float(bounds[2]),
                "south": float(bounds[1]),
                "north": float(bounds[3]),
            },
            "zoom": 4.0,
        },
        "legend": {
            "title": "Layers",
            "items": [
                {
                    "label": "Clay Solar Candidate",
                    "color": "#4CAF50",
                    "description": f"{len(features)} sites",
                }
            ],
        },
        "geometry_type": "point",
    }


@mcp.tool()
def run_query(query: str, context: dict) -> str:
    """Execute a general query against the Clay solar candidate dataset.

    Fallback tool that returns dataset summary and map. The LLM tool planner should
    prefer specific tools like GetClayCandidatePointsByState for targeted queries.
    """
    facts: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    messages: List[Dict[str, Any]] = []

    citation = {
        "id": "clay-solar-candidates",
        "server": "solar_clay",
        "tool": "run_query",
        "title": "Clay Solar Site Potential Dataset",
        "source_type": "Dataset",
        "description": (
            "TransitionDigital. Clay Solar Site Potential [dataset]. Ode Partners. "
            "Based on Clay Earth Observation foundation model and PVGIS yield data, 2025."
        ),
        "url": "https://clay-foundation.github.io/model/",
    }
    citations.append(citation)

    try:
        solar_gdf, _, _ = _get_dataset_or_raise()
    except DatasetUnavailable as exc:
        messages.append(
            {
                "level": "error",
                "text": f"Clay solar candidate data is unavailable: {exc}",
            }
        )
        response = {
            "server": "solar_clay",
            "query": query,
            "facts": facts,
            "citations": citations,
            "artifacts": artifacts,
            "messages": messages,
            "kg": {"nodes": [], "edges": []},
            "next_actions": [],
            "duration_ms": 0,
        }
        return json.dumps(response)

    total_sites = len(solar_gdf)
    facts.append(
        {
            "id": "f1",
            "kind": "text",
            "text": (
                f"The Clay solar candidate dataset contains {total_sites} potential solar "
                "development sites in Brazil identified using AI-powered Earth observation analysis."
            ),
            "citation_id": "clay-solar-candidates",
        }
    )

    if "state" in solar_gdf.columns:
        state_counts = solar_gdf["state"].value_counts().head(5)
        state_list = ", ".join(
            [f"{state} ({count} sites)" for state, count in state_counts.items()]
        )
        facts.append(
            {
                "id": "f2",
                "kind": "text",
                "text": f"Top states by candidate site count: {state_list}.",
                "citation_id": "clay-solar-candidates",
            }
        )

    artifact = _build_geojson_artifact(solar_gdf)
    if artifact:
        artifacts.append(artifact)
    else:
        messages.append(
            {
                "level": "warning",
                "text": "Could not generate map artifact for Clay solar candidates.",
            }
        )

    messages.append(
        {
            "level": "info",
            "text": (
                "Returning basic dataset summary (tool planner fallback). "
                "For detailed analysis, the LLM tool planner will select specific tools like "
                "GetClayCandidatePointsByState."
            ),
        }
    )

    response = {
        "server": "solar_clay",
        "query": query,
        "facts": facts,
        "citations": citations,
        "artifacts": artifacts,
        "messages": messages,
        "kg": {"nodes": [], "edges": []},
        "next_actions": [],
        "duration_ms": 0,
    }
    return json.dumps(response)


if __name__ == "__main__":
    mcp.run()
