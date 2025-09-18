#!/usr/bin/env python3
"""
Heat Stress Quintiles Server

Serves preprocessed heat-stress polygons (quintiles 1..5) for geospatial analysis.
- Optimized to load and index only the top quintile (5) by default.
- Provides tools to list layers, fetch entities for geospatial registration,
  and generate a map GeoJSON for front-end display.

Uses FastMCP for consistent server architecture.
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from fastmcp import FastMCP

mcp = FastMCP("heat-stress-server")

# Try to import geospatial libraries
try:
    import geopandas as gpd
    from shapely.geometry import shape, box
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
    GEOSPATIAL_AVAILABLE = True
except Exception:
    GEOSPATIAL_AVAILABLE = False
    print("Warning: GeoPandas/Shapely not available. Install with: pip install geopandas shapely")

# Data directory with preprocessed files (GPKG or simplified GeoJSONs)
BASE_DIR = Path(__file__).resolve().parent.parent
HEAT_DIR = BASE_DIR / "data" / "heat_stress" / "preprocessed"
HEAT_GEOJSON_DIR = HEAT_DIR / "geojsons"
HEAT_DOC_PATH = BASE_DIR / "data" / "heat_stress" / "Brazil Datasets Info.txt"

# Global, lazy-loaded index of top quintile polygons
HEAT_INDEX = {
    "loaded": False,
    "gdf": None,               # GeoDataFrame EPSG:4326 with columns: id, source, quintile, geometry
    "tree": None,              # STRtree of geometries
    "geom_list": None,         # list of shapely geometries matching index order
    "id_list": None,           # list of feature ids matching index order
}

# Basic dataset metadata derived from Brazil Datasets Info.txt (hardcoded mapping)
HEAT_DATASET_INFO = {
    "Heat_Index": {
        "variable": "Heat Index (WBGT proxy)",
        "spatial_coverage": "Brazil",
        "temporal_coverage": "2020-01-01 to 2025-01-01",
        "source": "ERA5-Land Daily Aggregated — ECMWF",
        "citation": "Muñoz Sabater, J. (2019). ERA5-Land monthly averaged data from 1981 to present. Copernicus C3S. doi:10.24381/cds.68d2bb30",
        "notes": "ERA5-Land lacks coverage over large water bodies and some coastal margins.",
    },
    "LST": {
        "variable": "Land Surface Temperature (LST)",
        "spatial_coverage": "Brazil",
        "temporal_coverage": "2020-01-01 to 2025-01-01",
        "source": "MOD11A1.061 — MODIS/Terra LST & Emissivity, Daily, 1 km",
        "citation": "Wan, Z., Hook, S., & Hulley, G. (2021). MODIS/Terra LST/Emissivity Daily L3 Global 1km SIN Grid V061. NASA LP DAAC. https://doi.org/10.5067/MODIS/MOD11A1.061",
        "notes": "Daytime LST daily means; southern-summer variants aggregate Nov–Mar seasons.",
    }
}

def _dataset_info_for_source(source: str) -> Dict[str, Any]:
    key = "Heat_Index" if "Heat_Index" in source else ("LST" if "LST" in source else None)
    return HEAT_DATASET_INFO.get(key, {})


def _explode_geoms(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """Explode MultiPolygon geometries to individual Polygon parts, preserving attributes."""
    try:
        egdf = gdf.explode(index_parts=False).reset_index(drop=True)
        return egdf
    except Exception:
        # Fallback manual explode
        rows = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            try:
                geoms = getattr(geom, "geoms", [geom])
            except Exception:
                geoms = [geom]
            for part in geoms:
                new_row = row.copy()
                new_row.geometry = part
                rows.append(new_row)
        return gpd.GeoDataFrame(rows, crs=gdf.crs)


def _normalize_source_name(stem: str) -> str:
    """Derive a source name from filename stem, dropping quintile suffixes."""
    for suf in ["_quintiles_simplified", "_quintiles"]:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _load_top_quintiles_index() -> bool:
    """Load all top-quintile (5) polygons across available GPKGs and build an STRtree.

    Returns True if loaded successfully or already loaded.
    """
    if not GEOSPATIAL_AVAILABLE:
        return False
    if HEAT_INDEX["loaded"]:
        return True
    # Prefer simplified GeoJSONs if available
    search_dir = HEAT_GEOJSON_DIR if HEAT_GEOJSON_DIR.exists() else HEAT_DIR
    if not search_dir.exists():
        print(f"[heat] Heat directory not found: {search_dir}")
        return False

    # Accept both simplified and non-simplified geojsons
    files = sorted([
        *search_dir.glob("*_quintiles_simplified.geojson"),
        *search_dir.glob("*_quintiles.geojson"),
        *search_dir.glob("*_quintiles.gpkg")  # fallback if GeoPackage still used
    ])
    if not files:
        print(f"[heat] No heat-stress files found in {search_dir}")
        return False

    parts = []
    for path in files:
        try:
            # Read either GeoJSON or GPKG (layer name only for GPKG)
            if str(path).lower().endswith(".gpkg"):
                gdf = gpd.read_file(path, layer="quintiles")
            else:
                gdf = gpd.read_file(path)
            # Ensure CRS -> WGS84 for registration
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            # Filter to top quintile only
            if "quintile" not in gdf.columns:
                print(f"[heat] Missing 'quintile' column in {path.name}; skipping")
                continue
            top = gdf[gdf["quintile"] == 5].copy()
            if top.empty:
                continue

            # Add source name and stable id base
            src = _normalize_source_name(path.stem)
            top["source"] = src

            # Explode multipolygons to individual parts
            top_parts = _explode_geoms(top)
            # Drop empties
            top_parts = top_parts[~top_parts.geometry.is_empty]
            # Assign ids
            top_parts = top_parts.reset_index(drop=True)
            top_parts["id"] = [f"heat_{src}_q5_{i}" for i in range(len(top_parts))]
            parts.append(top_parts[["id", "source", "quintile", "geometry"]])
        except Exception as e:
            print(f"[heat] Failed to read {path.name}: {e}")
            continue

    if not parts:
        print("[heat] No top quintile polygons loaded")
        return False

    all_top = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")

    # Build STRtree index
    try:
        geoms = list(all_top.geometry.values)
        tree = STRtree(geoms)
        HEAT_INDEX.update({
            "gdf": all_top,
            "tree": tree,
            "geom_list": geoms,
            "id_list": list(all_top["id"].values),
            "loaded": True,
        })
        print(f"[heat] Top quintile index loaded: {len(all_top)} polygons from {len(files)} sources ({search_dir})")
        return True
    except Exception as e:
        print(f"[heat] Failed to build STRtree: {e}")
        return False


# Pandas is optional but used for concat/stats
try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore


@mcp.tool()
def ListHeatLayers() -> Dict[str, Any]:
    """
    List available preprocessed heat-stress GPKG layers and basic metadata.
    Focus on quintiles layer and presence of top quintile.
    """
    files = []
    search_dirs = [d for d in [HEAT_GEOJSON_DIR, HEAT_DIR] if d.exists()]
    for dirp in search_dirs:
        for p in sorted(list(dirp.glob("*_quintiles_simplified.geojson")) +
                         list(dirp.glob("*_quintiles.geojson")) +
                         list(dirp.glob("*_quintiles.gpkg"))):
            info = {
                "filename": p.name,
                "path": str(p),
                "source": _normalize_source_name(p.stem),
                "has_quintiles": True,
                "has_top_quintile": None,
            }
            try:
                if GEOSPATIAL_AVAILABLE:
                    if str(p).lower().endswith(".gpkg"):
                        gdf = gpd.read_file(p, layer="quintiles")
                    else:
                        gdf = gpd.read_file(p)
                    info["crs"] = (gdf.crs.to_string() if gdf.crs else "None")
                    if "quintile" in gdf.columns:
                        qs = sorted([int(x) for x in gdf["quintile"].dropna().unique().tolist()])
                        info["available_quintiles"] = qs
                        info["has_top_quintile"] = (5 in qs)
                    info["feature_count"] = int(len(gdf))
                    info["dataset_info"] = _dataset_info_for_source(info["source"]) or None
            except Exception as e:
                info["error"] = str(e)
            files.append(info)
    return {
        "heat_layers": files,
        "directories": [str(d) for d in search_dirs],
        "note": "Server optimized for top quintile (5) polygons",
    }


@mcp.tool()
def GetHeatQuintileCounts() -> Dict[str, Any]:
    """Return a chart spec for counts by heat-stress quintile (1..5)."""
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    search_dir = HEAT_GEOJSON_DIR if HEAT_GEOJSON_DIR.exists() else HEAT_DIR
    files = sorted(list(search_dir.glob("*_quintiles_simplified.geojson")) + list(search_dir.glob("*_quintiles.geojson")))
    if not files:
        return {"error": f"No heat-stress quintiles files found under {search_dir}"}
    try:
        gdf = gpd.read_file(files[0])
    except Exception as e:
        return {"error": f"Failed to read {files[0]}: {e}"}
    if 'quintile' not in gdf.columns:
        return {"error": "Dataset missing 'quintile' column"}
    vc = gdf['quintile'].value_counts().sort_index()
    data = [{"quintile": int(k), "count": int(v)} for k, v in vc.items()]
    return {
        "visualization_type": "comparison",
        "data": data,
        "chart_config": {
            "x_axis": "quintile",
            "y_axis": "count",
            "title": "Heat-Stress Features by Quintile",
            "chart_type": "bar"
        },
        "source_file": str(files[0])
    }


@mcp.tool()
def GetHeatAreaByQuintileChart() -> Dict[str, Any]:
    """Return a pie (or bar) chart spec of total area (km²) by heat-stress quintile.

    Computes polygon areas in an equal-area projection (EPSG:5880 – Brazil Polyconic)
    and aggregates by the 'quintile' column.
    """
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    search_dir = HEAT_GEOJSON_DIR if HEAT_GEOJSON_DIR.exists() else HEAT_DIR
    files = sorted(list(search_dir.glob("*_quintiles_simplified.geojson")) + list(search_dir.glob("*_quintiles.geojson")))
    if not files:
        return {"error": f"No heat-stress quintiles files found under {search_dir}"}
    f = files[0]
    try:
        gdf = gpd.read_file(f)
    except Exception as e:
        return {"error": f"Failed to read {f}: {e}"}
    if 'quintile' not in gdf.columns:
        return {"error": "Dataset missing 'quintile' column"}

    # Ensure CRS and compute area in km^2 using equal-area projection for Brazil
    try:
        gdf_proj = gdf.to_crs('EPSG:5880') if (gdf.crs is not None) else gdf.set_crs('EPSG:4326').to_crs('EPSG:5880')
    except Exception:
        # Fallback: use EPSG:3857 with note (less accurate)
        try:
            gdf_proj = gdf.to_crs('EPSG:3857') if (gdf.crs is not None) else gdf.set_crs('EPSG:4326').to_crs('EPSG:3857')
        except Exception as e:
            return {"error": f"Failed to project dataset for area calculation: {e}"}

    gdf_proj = gdf_proj[~gdf_proj.geometry.is_empty]
    gdf_proj['area_km2'] = gdf_proj.geometry.area / 1_000_000.0
    area_by_q = gdf_proj.groupby('quintile', dropna=False)['area_km2'].sum().sort_index()
    data = [{"quintile": int(q), "area_km2": float(a)} for q, a in area_by_q.items()]

    return {
        "visualization_type": "comparison",
        "data": data,
        "chart_config": {
            "x_axis": "quintile",
            "y_axis": "area_km2",
            "title": "Heat-Stress Area by Quintile (km²)",
            "chart_type": "pie"  # Frontend will render a pie from the standard chart module
        },
        "source_file": str(f)
    }

@mcp.tool()
def GetHeatDatasetInfo(source: Optional[str] = None) -> Dict[str, Any]:
    """Return dataset metadata/citation for heat-stress layers.

    Args:
        source: Optional source hint ('Heat_Index' or 'LST' in filename).
    """
    if source:
        info = _dataset_info_for_source(source)
        if info:
            return {"dataset": info, "matched_source": source}
    # Return both when not specified
    return {"datasets": HEAT_DATASET_INFO}

@mcp.tool()
def DescribeServer() -> Dict[str, Any]:
    """Describe heat-stress layers, tools, and live availability."""
    try:
        # Count available heat layer files without invoking MCP tool wrappers
        search_dirs = [d for d in [HEAT_GEOJSON_DIR, HEAT_DIR] if d.exists()]
        count = 0
        for dirp in search_dirs:
            count += len(list(dirp.glob("*_quintiles_simplified.geojson")))
            count += len(list(dirp.glob("*_quintiles.geojson")))
            count += len(list(dirp.glob("*_quintiles.gpkg")))
        # Derive last_updated from available files
        last_updated = None
        try:
            from datetime import datetime as _dt
            mtimes = []
            for dirp in search_dirs:
                for p in dirp.glob("*.geojson"):
                    try:
                        mtimes.append(p.stat().st_mtime)
                    except Exception:
                        pass
                for p in dirp.glob("*.gpkg"):
                    try:
                        mtimes.append(p.stat().st_mtime)
                    except Exception:
                        pass
            if mtimes:
                last_updated = _dt.fromtimestamp(max(mtimes)).isoformat()
        except Exception:
            pass

        # Build a richer description from Brazil Datasets Info.txt (if present)
        description = "Preprocessed heat-stress quintile layers for Brazil (top-quintile optimized) derived from: " \
            + "ERA5-Land Heat Index (WBGT proxy) and MODIS/Terra Land Surface Temperature (LST). " \
            + "Aggregations include multi-year daily means for 2020–2025 and southern-summer (Nov–Mar) seasonal means. " \
            + "Units in °C; CRS EPSG:4326."
        try:
            if HEAT_DOC_PATH.exists():
                # Extract a few salient bullets from the doc to enrich the description
                txt = HEAT_DOC_PATH.read_text(encoding="utf-8", errors="ignore")
                bullets = []
                for key in ["Variable:", "Temporal Coverage:", "Temporal Aggregation:", "Source:", "Notes:"]:
                    # take the first two occurrences across both datasets
                    parts = [line.strip("\r\n ") for line in txt.splitlines() if key in line]
                    for p in parts[:2]:
                        # compact line
                        bullets.append(p.replace("?", "").strip())
                if bullets:
                    description += " " + " ".join(bullets[:4])
        except Exception:
            pass
        return {
            "name": "Heat Stress Server",
            "description": description,
            "version": "1.0",
            "dataset": "Heat index and land surface temperature derived layers",
            "metrics": {"layer_count": count},
            "tools": [
                "ListHeatLayers",
                "GetHeatQuintileCounts",
                "GetHeatAreaByQuintileChart",
                "GetHeatDatasetInfo",
                "GetHeatQuintilesForGeospatial"
            ],
            "examples": [
                "List available heat-stress layers",
                "Register top quintile heat zones for correlation"
            ],
            "last_updated": last_updated
        }
    except Exception as e:
        return {"error": str(e)}


def _filter_by_bbox(gdf: "gpd.GeoDataFrame", bbox_dict: Optional[Dict[str, float]]) -> "gpd.GeoDataFrame":
    if not bbox_dict:
        return gdf
    try:
        b = box(bbox_dict["west"], bbox_dict["south"], bbox_dict["east"], bbox_dict["north"])
    except Exception:
        return gdf
    if HEAT_INDEX["tree"] is not None and gdf is HEAT_INDEX["gdf"]:
        # Use the STRtree for fast candidate selection
        cand_idxs = list(HEAT_INDEX["tree"].query(b))
        sub = gdf.iloc[cand_idxs]
        return sub[sub.geometry.intersects(b)]
    # Fallback: direct intersection
    return gdf[gdf.geometry.intersects(b)]


@mcp.tool()
def GetHeatQuintilesForGeospatial(
    source: Optional[str] = None,
    quintiles: Optional[List[int]] = None,
    limit: int = 5000,
    bbox: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Return condensed heat zone entities for geospatial registration.
    Defaults to top quintile (5) and uses STRtree for bbox filtering.

    Args:
        source: Optional basename to filter a single GPKG source (without _quintiles)
        quintiles: List of quintiles (1..5); defaults to [5]
        limit: Max number of polygon entities to return
        bbox: Optional geographic bounding box dict {north, south, east, west}
    """
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    if pd is None:
        return {"error": "Pandas not installed"}

    if not _load_top_quintiles_index():
        return {"error": "Failed to load top quintile index"}

    qs = quintiles if quintiles else [5]
    # For now, only top quintile is indexed; ignore others gracefully
    if qs != [5]:
        return {"warning": "Only top quintile (5) is available in optimized index", "entities": [], "count": 0, "entity_type": "heat_zone"}

    gdf = HEAT_INDEX["gdf"]
    if source:
        gdf = gdf[gdf["source"].str.lower() == source.lower()]
    if bbox:
        gdf = _filter_by_bbox(gdf, bbox)

    if gdf.empty:
        return {"entity_type": "heat_zone", "entities": [], "count": 0, "note": "No matching heat zones"}

    # Limit
    out = gdf.head(max(0, int(limit))).copy()

    entities: List[Dict[str, Any]] = []
    for _, row in out.iterrows():
        try:
            entities.append({
                "id": row["id"],
                "geometry": row.geometry.__geo_interface__,
                "quintile": int(row.get("quintile", 5)),
                "source": row.get("source"),
            })
        except Exception:
            continue

    meta_info = _dataset_info_for_source(source or (gdf.iloc[0]["source"] if not gdf.empty else ""))
    return {
        "entity_type": "heat_zone",
        "quintiles": qs,
        "entities": entities,
        "count": len(entities),
        "source_filter": source,
        "bbox": bbox,
        "optimized_top_quintile_only": True,
        "citation": meta_info,
        "spatial_coverage": meta_info.get("spatial_coverage"),
        "temporal_coverage": meta_info.get("temporal_coverage"),
        "dataset_variable": meta_info.get("variable")
    }


def _color_for_quintile(q: int) -> str:
    # Red gradient for heat
    colors = {
        1: "#FFEDA0",
        2: "#FEB24C",
        3: "#FD8D3C",
        4: "#FC4E2A",
        5: "#E31A1C",
    }
    return colors.get(int(q), "#E31A1C")


@mcp.tool()
def GetHeatQuintilesMap(
    source: Optional[str] = None,
    quintiles: Optional[List[int]] = None,
    limit: int = 5000,
    bbox: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Generate a GeoJSON map of heat-stress polygons (default: top quintile only).
    Saves to static/maps and returns URL and summary.
    """
    if not GEOSPATIAL_AVAILABLE:
        return {"error": "GeoPandas not installed"}
    if pd is None:
        return {"error": "Pandas not installed"}

    if not _load_top_quintiles_index():
        return {"error": "Failed to load top quintile index"}

    qs = quintiles if quintiles else [5]
    if qs != [5]:
        return {"error": "Only top quintile (5) supported for map generation currently"}

    gdf = HEAT_INDEX["gdf"]
    if source:
        gdf = gdf[gdf["source"].str.lower() == source.lower()]
    if bbox:
        gdf = _filter_by_bbox(gdf, bbox)
    out = gdf.head(max(0, int(limit))).copy()

    # Build FeatureCollection
    features = []
    countries = set()
    for _, row in out.iterrows():
        q = int(row.get("quintile", 5))
        props = {
            "layer": "heat_zone",
            "quintile": q,
            "source": row.get("source"),
            "fill": _color_for_quintile(q),
            "fill-opacity": 0.35,
            "stroke": "#9E9E9E",
            "stroke-width": 0.5,
            "title": f"Heat zone Q{q} • {row.get('source')}"
        }
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": props
        })
    geojson = {"type": "FeatureCollection", "features": features, "metadata": {}}

    # Save to static/maps
    project_root = BASE_DIR
    static_maps_dir = os.path.join(project_root, "static", "maps")
    os.makedirs(static_maps_dir, exist_ok=True)
    ident = source or "all"
    data_hash = hashlib.md5(f"heat_{ident}_{len(features)}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8]
    filename = f"heat_quintiles_{ident}_{data_hash}.geojson"
    out_path = os.path.join(static_maps_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)

    meta_info = _dataset_info_for_source(source or (gdf.iloc[0]["source"] if not gdf.empty else ""))
    summary = {
        "description": f"Heat stress map (quintile {qs})",
        "total_features": len(features),
        "layers": [{"type": "heat_zone", "count": len(features)}],
        "title": "Heat Stress Map",
        "spatial_coverage": meta_info.get("spatial_coverage"),
        "temporal_coverage": meta_info.get("temporal_coverage"),
        "dataset_variable": meta_info.get("variable")
    }

    return {
        "type": "map",
        "geojson_url": f"/static/maps/{filename}",
        "geojson_filename": filename,
        "summary": summary,
        "citation": meta_info,
        "optimized_top_quintile_only": True
    }


if __name__ == "__main__":
    mcp.run()
