"""Precompute deforestation area overlays against Brazilian states and municipalities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.geospatial_datasets.deforestation import DeforestationPolygonProvider

ADMIN_DIR = PROJECT_ROOT / "data" / "brazilian_admin"
OUTPUT_DIR = PROJECT_ROOT / "static" / "meta"
STATE_OUTPUT = OUTPUT_DIR / "deforestation_by_state.json"
MUNICIPALITY_OUTPUT = OUTPUT_DIR / "deforestation_by_municipality.json"


def _first_column(gdf: gpd.GeoDataFrame, candidates: List[str], fallback: str) -> pd.Series:
    for column in candidates:
        if column in gdf.columns:
            return gdf[column].astype(str)
    return gdf[fallback].astype(str)


def load_states() -> gpd.GeoDataFrame:
    path = ADMIN_DIR / "brazilian_states.geojson"
    if not path.exists():
        raise RuntimeError(f"State GeoJSON missing: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.reset_index(drop=True)
    gdf["state_name"] = _first_column(gdf, ["NM_UF", "NOME", "NAME", "state", "UF_NAME"], "index")
    gdf["state_code"] = _first_column(gdf, ["SIGLA_UF", "UF", "STATE_ABBR", "abbrev", "state_code"], "state_name")
    return gdf[["state_code", "state_name", "geometry"]]


def load_municipalities() -> gpd.GeoDataFrame:
    geojson_path = ADMIN_DIR / "BR_Municipios_2024_simplified.geojson"
    shapefile_path = ADMIN_DIR / "municipality_shapes" / "BR_Municipios_2024.shp"
    if geojson_path.exists():
        gdf = gpd.read_file(geojson_path)
    elif shapefile_path.exists():
        gdf = gpd.read_file(shapefile_path)
    else:
        raise RuntimeError("Municipality geometries unavailable")
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.reset_index(drop=True)
    gdf["muni_code"] = _first_column(gdf, ["CD_MUN", "CD_GEOCMU", "muni_code"], "index")
    gdf["municipality_name"] = _first_column(gdf, ["NM_MUN", "NM_MUNICIP", "NOME", "NAME", "municipio"], "muni_code")
    gdf["state"] = _first_column(gdf, ["SIGLA_UF", "state", "UF", "state_abbr", "NM_UF", "nome_uf"], "municipality_name")
    return gdf[["muni_code", "municipality_name", "state", "geometry"]]


def load_deforestation_polygons() -> gpd.GeoDataFrame:
    provider = DeforestationPolygonProvider()
    gdf = provider._gdf.copy()  # type: ignore[attr-defined]
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf["area_hectares"] = pd.to_numeric(gdf.get("area_hectares"), errors="coerce").fillna(0.0)
    gdf["geometry"] = gdf.geometry
    return gdf[["__polygon_id", "area_hectares", "geometry"]]


def aggregate_area(
    polygons: gpd.GeoDataFrame,
    admin_gdf: gpd.GeoDataFrame,
    mapping: Dict[str, str],
) -> List[Dict[str, object]]:
    overlay = gpd.overlay(polygons, admin_gdf, how="intersection")
    if overlay.empty:
        return []
    overlay["area_hectares"] = pd.to_numeric(overlay.get("area_hectares"), errors="coerce").fillna(0.0)
    grouped = overlay.groupby(list(mapping.values()), as_index=False).agg(
        polygons=("__polygon_id", "nunique"),
        total_area_hectares=("area_hectares", "sum"),
    )
    grouped["total_area_hectares"] = grouped["total_area_hectares"].astype(float)
    grouped["polygons"] = grouped["polygons"].astype(int)
    grouped.sort_values("total_area_hectares", ascending=False, inplace=True)
    records = grouped.to_dict("records")
    for record in records:
        record["total_area_hectares"] = round(float(record.get("total_area_hectares", 0.0)), 3)
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    polygons = load_deforestation_polygons()
    states = load_states()
    municipalities = load_municipalities()

    state_stats = aggregate_area(
        polygons,
        states,
        {"state_code": "state_code", "state_name": "state_name"},
    )
    municipality_stats = aggregate_area(
        polygons,
        municipalities,
        {
            "muni_code": "muni_code",
            "municipality_name": "municipality_name",
            "state": "state",
        },
    )

    STATE_OUTPUT.write_text(json.dumps({"items": state_stats}, indent=2), encoding="utf-8")
    MUNICIPALITY_OUTPUT.write_text(json.dumps({"items": municipality_stats}, indent=2), encoding="utf-8")
    print(f"Wrote {len(state_stats)} state rows to {STATE_OUTPUT}")
    print(f"Wrote {len(municipality_stats)} municipality rows to {MUNICIPALITY_OUTPUT}")


if __name__ == "__main__":
    main()
