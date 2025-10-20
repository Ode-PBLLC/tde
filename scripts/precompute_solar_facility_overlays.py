"""Precompute solar facility overlays against Brazilian states and municipalities."""

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

from mcp.solar_db import SolarDatabase  # noqa: E402

ADMIN_DIR = PROJECT_ROOT / "data" / "brazilian_admin"
OUTPUT_DIR = PROJECT_ROOT / "static" / "meta"
STATE_OUTPUT = OUTPUT_DIR / "solar_facilities_by_state.json"
MUNICIPALITY_OUTPUT = OUTPUT_DIR / "solar_facilities_by_municipality.json"


def load_facilities() -> gpd.GeoDataFrame:
    db = SolarDatabase()
    records = db.get_all_facilities()
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("Solar facilities table is empty")

    df = df.dropna(subset=["latitude", "longitude"])
    df["capacity_mw"] = pd.to_numeric(df.get("capacity_mw"), errors="coerce")
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"], crs="EPSG:4326")
    facilities = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    facilities["cluster_id"] = facilities["cluster_id"].astype(str)
    return facilities


def _first_available_column(gdf: gpd.GeoDataFrame, candidates: List[str], fallback: str) -> pd.Series:
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
    gdf["state_name"] = _first_available_column(
        gdf,
        ["NM_UF", "NOME", "NAME", "state", "UF_NAME"],
        fallback="index",
    )
    gdf["state_code"] = _first_available_column(
        gdf,
        ["SIGLA_UF", "UF", "STATE_ABBR", "abbrev", "state_code"],
        fallback="state_name",
    )
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
    gdf["muni_code"] = _first_available_column(
        gdf,
        ["CD_MUN", "CD_GEOCMU", "muni_code"],
        fallback="index",
    )
    gdf["municipality_name"] = _first_available_column(
        gdf,
        ["NM_MUN", "NM_MUNICIP", "NOME", "NAME", "municipio"],
        fallback="muni_code",
    )
    gdf["state"] = _first_available_column(
        gdf,
        ["SIGLA_UF", "state", "UF", "state_abbr", "NM_UF", "nome_uf"],
        fallback="municipality_name",
    )
    return gdf[["muni_code", "municipality_name", "state", "geometry"]]


def aggregate_facilities(
    facilities: gpd.GeoDataFrame,
    admin_gdf: gpd.GeoDataFrame,
    group_columns: Dict[str, str],
) -> List[Dict[str, object]]:
    joined = gpd.sjoin(facilities, admin_gdf, how="inner", predicate="within")
    if joined.empty:
        return []

    joined["capacity_mw"] = pd.to_numeric(joined["capacity_mw"], errors="coerce")

    grouped = joined.groupby(list(group_columns.values()), as_index=False).agg(
        facility_count=("cluster_id", "count"),
        total_capacity_mw=("capacity_mw", "sum"),
        facilities_with_capacity=("capacity_mw", lambda series: int(series.notna().sum())),
    )
    grouped["total_capacity_mw"] = grouped["total_capacity_mw"].astype(float)
    grouped.sort_values("total_capacity_mw", ascending=False, inplace=True)

    records = grouped.to_dict("records")
    for record in records:
        record["total_capacity_mw"] = round(float(record.get("total_capacity_mw", 0.0)), 3)
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    facilities = load_facilities()
    states = load_states()
    municipalities = load_municipalities()

    state_stats = aggregate_facilities(
        facilities,
        states,
        {"state_code": "state_code", "state_name": "state_name"},
    )
    municipality_stats = aggregate_facilities(
        facilities,
        municipalities,
        {
            "muni_code": "muni_code",
            "municipality_name": "municipality_name",
            "state": "state",
        },
    )

    STATE_OUTPUT.write_text(json.dumps({"items": state_stats}, indent=2), encoding="utf-8")
    MUNICIPALITY_OUTPUT.write_text(
        json.dumps({"items": municipality_stats}, indent=2),
        encoding="utf-8",
    )

    print(
        f"Wrote {len(state_stats)} state rows to {STATE_OUTPUT}"
    )
    print(
        f"Wrote {len(municipality_stats)} municipality rows to {MUNICIPALITY_OUTPUT}"
    )


if __name__ == "__main__":
    main()
