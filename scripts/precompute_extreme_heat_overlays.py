"""Precompute extreme heat overlays against Brazilian states and municipalities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEAT_DIR = PROJECT_ROOT / "data" / "heat_stress" / "preprocessed"
HEAT_GEOJSON_DIR = HEAT_DIR / "geojsons"
ADMIN_DIR = PROJECT_ROOT / "data" / "brazilian_admin"

OUTPUT_DIR = PROJECT_ROOT / "static" / "meta"
STATE_OUTPUT = OUTPUT_DIR / "extreme_heat_by_state.json"
MUNICIPALITY_OUTPUT = OUTPUT_DIR / "extreme_heat_by_municipality.json"


def _list_heat_files() -> List[Path]:
    files: List[Path] = []
    seen: set[Path] = set()
    for directory in (HEAT_GEOJSON_DIR, HEAT_DIR):
        if not directory.exists():
            continue
        for pattern in (
            "*_quintiles_simplified.geojson",
            "*_quintiles.geojson",
            "*_quintiles.gpkg",
        ):
            for path in sorted(directory.glob(pattern)):
                if path not in seen:
                    files.append(path)
                    seen.add(path)
    return files


def load_heat_polygons() -> gpd.GeoDataFrame:
    files = _list_heat_files()
    parts: List[gpd.GeoDataFrame] = []
    for path in files:
        if str(path).lower().endswith(".gpkg"):
            gdf = gpd.read_file(path, layer="quintiles")
        else:
            gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        if "quintile" not in gdf.columns:
            continue
        top = gdf[gdf["quintile"] == 5].copy()
        if top.empty:
            continue
        source = Path(path).stem
        top["source"] = source
        top = top.reset_index(drop=True)
        parts.append(top[["source", "geometry"]])
    if not parts:
        raise RuntimeError("No heat polygons found")
    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    combined["id"] = [f"heat_{idx}" for idx in range(len(combined))]
    return combined


def load_states() -> gpd.GeoDataFrame:
    path = ADMIN_DIR / "brazilian_states.geojson"
    if not path.exists():
        raise RuntimeError(f"State GeoJSON missing: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    def _first(series_names):
        for column in series_names:
            if column in gdf.columns:
                return gdf[column].astype(str)
        return gdf.index.astype(str)

    gdf["state_name"] = _first(["NM_UF", "NOME", "NAME", "state", "UF_NAME"])
    gdf["state_code"] = _first(["SIGLA_UF", "UF", "STATE_ABBR", "abbrev", "state_code"])
    return gdf[["state_code", "state_name", "geometry"]]


def load_municipalities() -> gpd.GeoDataFrame:
    geojson_path = ADMIN_DIR / "BR_Municipios_2024_simplified.geojson"
    shapefile_path = ADMIN_DIR / "municipality_shapes" / "BR_Municipios_2024.shp"
    if geojson_path.exists():
        gdf = gpd.read_file(geojson_path)
    elif shapefile_path.exists():
        gdf = gpd.read_file(shapefile_path)
    else:
        raise RuntimeError("Municipality geometries missing")
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    def _first(columns):
        for column in columns:
            if column in gdf.columns:
                return gdf[column].astype(str)
        return gdf.index.astype(str)

    gdf["muni_code"] = _first(["CD_MUN", "CD_GEOCMU", "muni_code"])
    gdf["municipality_name"] = _first(["NM_MUN", "NM_MUNICIP", "NOME", "NAME", "municipio"])
    gdf["state"] = _first(["SIGLA_UF", "state", "UF", "state_abbr", "NM_UF", "nome_uf"])
    return gdf[["muni_code", "municipality_name", "state", "geometry"]]


def aggregate_heat_by_polygons(
    heat_gdf: gpd.GeoDataFrame,
    admin_gdf: gpd.GeoDataFrame,
    admin_columns: Dict[str, str],
) -> List[Dict[str, Any]]:
    heat_proj = heat_gdf.to_crs("EPSG:5880")
    admin_proj = admin_gdf.to_crs("EPSG:5880")
    overlay = gpd.overlay(
        heat_proj[["id", "source", "geometry"]],
        admin_proj[[*admin_columns.values(), "geometry"]],
        how="intersection",
    )
    if overlay.empty:
        return []
    overlay["area_km2"] = overlay.geometry.area / 1_000_000.0
    group_fields = list(admin_columns.values())
    grouped = overlay.groupby(group_fields, as_index=False).agg(
        area_km2=("area_km2", "sum"),
        polygon_count=("id", "nunique"),
    )
    grouped = grouped.sort_values("area_km2", ascending=False)
    records = grouped.to_dict("records")
    for record in records:
        record["area_km2"] = round(float(record.get("area_km2", 0.0)), 6)
        record["polygon_count"] = int(record.get("polygon_count", 0))
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heat_gdf = load_heat_polygons()
    states = load_states()
    municipalities = load_municipalities()

    state_stats = aggregate_heat_by_polygons(
        heat_gdf,
        states,
        {"state_code": "state_code", "state_name": "state_name"},
    )
    municipality_stats = aggregate_heat_by_polygons(
        heat_gdf,
        municipalities,
        {"muni_code": "muni_code", "municipality_name": "municipality_name", "state": "state"},
    )

    STATE_OUTPUT.write_text(json.dumps({"items": state_stats}, indent=2), encoding="utf-8")
    MUNICIPALITY_OUTPUT.write_text(json.dumps({"items": municipality_stats}, indent=2), encoding="utf-8")
    print(f"Wrote {len(state_stats)} state entries to {STATE_OUTPUT}")
    print(f"Wrote {len(municipality_stats)} municipality entries to {MUNICIPALITY_OUTPUT}")


if __name__ == "__main__":
    main()
