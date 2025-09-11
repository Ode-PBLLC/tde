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

# Data directory with preprocessed GPKGs (from preprocess_heat_rasters.py)
BASE_DIR = Path(__file__).resolve().parent.parent
HEAT_DIR = BASE_DIR / "data" / "heat_stress" / "preprocessed"

# Global, lazy-loaded index of top quintile polygons
HEAT_INDEX = {
    "loaded": False,
    "gdf": None,               # GeoDataFrame EPSG:4326 with columns: id, source, quintile, geometry
    "tree": None,              # STRtree of geometries
    "geom_list": None,         # list of shapely geometries matching index order
    "id_list": None,           # list of feature ids matching index order
}


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


def _load_top_quintiles_index() -> bool:
    """Load all top-quintile (5) polygons across available GPKGs and build an STRtree.

    Returns True if loaded successfully or already loaded.
    """
    if not GEOSPATIAL_AVAILABLE:
        return False
    if HEAT_INDEX["loaded"]:
        return True
    if not HEAT_DIR.exists():
        print(f"[heat] Heat directory not found: {HEAT_DIR}")
        return False

    files = sorted([p for p in HEAT_DIR.glob("*_quintiles.gpkg") if p.is_file()])
    if not files:
        print(f"[heat] No GPKG files found in {HEAT_DIR}")
        return False

    parts = []
    for path in files:
        try:
            gdf = gpd.read_file(path, layer="quintiles")
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
            src = path.stem.replace("_quintiles", "")
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
        print(f"[heat] Top quintile index loaded: {len(all_top)} polygons from {len(files)} sources")
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
    if HEAT_DIR.exists():
        for p in sorted(HEAT_DIR.glob("*_quintiles.gpkg")):
            info = {
                "filename": p.name,
                "path": str(p),
                "source": p.stem.replace("_quintiles", ""),
                "has_quintiles": True,
                "has_top_quintile": None,
            }
            try:
                if GEOSPATIAL_AVAILABLE:
                    gdf = gpd.read_file(p, layer="quintiles")
                    info["crs"] = (gdf.crs.to_string() if gdf.crs else "None")
                    if "quintile" in gdf.columns:
                        qs = sorted([int(x) for x in gdf["quintile"].dropna().unique().tolist()])
                        info["available_quintiles"] = qs
                        info["has_top_quintile"] = (5 in qs)
                    info["feature_count"] = int(len(gdf))
            except Exception as e:
                info["error"] = str(e)
            files.append(info)
    return {
        "heat_layers": files,
        "directory": str(HEAT_DIR),
        "note": "Server optimized for top quintile (5) polygons",
    }


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

    return {
        "entity_type": "heat_zone",
        "quintiles": qs,
        "entities": entities,
        "count": len(entities),
        "source_filter": source,
        "bbox": bbox,
        "optimized_top_quintile_only": True
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

    summary = {
        "description": f"Heat stress map (quintile {qs})",
        "total_features": len(features),
        "layers": [{"type": "heat_zone", "count": len(features)}],
        "title": "Heat Stress Map"
    }

    return {
        "type": "map",
        "geojson_url": f"/static/maps/{filename}",
        "geojson_filename": filename,
        "summary": summary,
        "optimized_top_quintile_only": True
    }


if __name__ == "__main__":
    mcp.run()

