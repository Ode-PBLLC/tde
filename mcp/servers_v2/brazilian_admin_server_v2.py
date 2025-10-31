"""Brazilian administrative boundaries MCP server (v2 contract).

This server exposes municipality and state boundaries for Brazil, providing
search/filter tools plus a `run_query` implementation compatible with the
new contracts_v2 schema.  The implementation is standalone and does not
import the legacy FastMCP server; instead, it reimplements the data loading
and tool surface following the v2 style used by other geospatial servers.
"""

import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional for intent classification
    import anthropic  # type: ignore
except Exception:  # pragma: no cover - optional dependency may be missing
    anthropic = None  # type: ignore

# Geo dependencies are optional because the repo can run without them in CI.
try:  # pragma: no cover - heavy dependencies usually stubbed in tests
    import geopandas as gpd  # type: ignore
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    from shapely.geometry import Point, box  # type: ignore
    from shapely.geometry.base import BaseGeometry  # type: ignore
    from shapely.strtree import STRtree  # type: ignore
    GEOSPATIAL_AVAILABLE = True
except Exception:  # noqa: BLE001 - broad catch to keep optional deps optional
    gpd = None  # type: ignore
    pd = None  # type: ignore
    np = None  # type: ignore
    Point = None  # type: ignore
    box = None  # type: ignore
    BaseGeometry = None  # type: ignore
    STRtree = None  # type: ignore
    GEOSPATIAL_AVAILABLE = False

from utils.llm_retry import call_llm_with_retries_sync

from fastmcp import FastMCP

if __package__ in {None, ""}:
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from mcp.contracts_v2 import (  # type: ignore
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from mcp.servers_v2.base import RunQueryMixin  # type: ignore
    from mcp.url_utils import ensure_absolute_url  # type: ignore
    from mcp.servers_v2.support_intent import SupportIntent  # type: ignore
else:  # pragma: no cover - normal import path
    from ..contracts_v2 import (
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from ..servers_v2.base import RunQueryMixin
    from ..url_utils import ensure_absolute_url
    from .support_intent import SupportIntent


SERVER_ID = "brazil_admin"
DATASET_ID = "admin"


@dataclass
class GeoJSONExport:
    url: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        self.url = ensure_absolute_url(self.url)


class BrazilianAdminServerV2(RunQueryMixin):
    """Expose Brazilian municipalities/states through the v2 MCP contract."""

    def __init__(self) -> None:
        self.mcp = FastMCP("brazilian-admin-server-v2")
        self.base_dir = Path(__file__).resolve().parents[2]
        self.data_dir = self.base_dir / "data" / "brazilian_admin"
        self.static_maps_dir = self.base_dir / "static" / "maps"

        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential issues
                print(f"[brazil-admin] Warning: Anthropic client unavailable: {exc}")

        self.municipalities_gdf: Optional["gpd.GeoDataFrame"] = None
        self.states_gdf: Optional["gpd.GeoDataFrame"] = None
        self.municipalities_tree: Optional["STRtree"] = None
        self.states_tree: Optional["STRtree"] = None
        self.csv_data: Optional["pd.DataFrame"] = None

        self.dataset_metadata = self._load_dataset_metadata()
        self._municipalities_loaded = self._load_municipalities()
        self._states_loaded = self._load_states()

        self._register_capabilities_tool()
        self._register_query_support_tool()
        self._register_dataset_metadata_tool()
        self._register_boundary_counts_tool()
        self._register_state_boundaries_tool()
        self._register_list_states_tool()
        self._register_search_municipalities_tool()
        self._register_municipalities_dataset_metadata_tool()
        self._register_municipalities_filter_tool()
        self._register_municipality_boundaries_tool()
        self._register_municipalities_bounds_tool()
        self._register_municipality_statistics_tool()
        self._register_municipalities_near_point_tool()
        self._register_top_cities_tool()
        self._register_population_by_state_tool()
        self._register_area_histogram_tool()
        self._register_run_query_tool()

    # ------------------------------------------------------------------ helpers
    def _load_dataset_metadata(self) -> Dict[str, Dict[str, str]]:
        path = self.base_dir / "static" / "meta" / "datasets.json"
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}

        metadata: Dict[str, Dict[str, str]] = {}
        for item in payload.get("items", []):
            identifier = item.get("id")
            if not identifier:
                continue
            metadata[str(identifier)] = {
                "title": str(item.get("title", "")),
                "source": str(item.get("source", "")),
                "citation": str(item.get("citation", "")),
                "description": str(item.get("description", "")),
            }
        return metadata

    def _municipality_csv_path(self) -> Path:
        return self.data_dir / "br.csv"

    def _municipality_geojson_path(self) -> Path:
        return self.data_dir / "BR_Municipios_2024_simplified.geojson"

    def _municipality_shapefile_path(self) -> Path:
        return self.data_dir / "municipality_shapes" / "BR_Municipios_2024.shp"

    def _states_geojson_path(self) -> Path:
        return self.data_dir / "brazilian_states.geojson"

    def _load_municipalities(self) -> bool:
        if not GEOSPATIAL_AVAILABLE:
            print("[brazil-admin] GeoPandas not available; municipality features disabled")
            return False

        csv_path = self._municipality_csv_path()
        geojson_path = self._municipality_geojson_path()
        shapefile_path = self._municipality_shapefile_path()

        if not csv_path.exists():
            print(f"[brazil-admin] Municipality CSV missing at {csv_path}")
            return False

        try:
            csv_df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[brazil-admin] Failed to load municipality CSV: {exc}")
            return False

        geometry_df = None
        if geojson_path.exists():
            try:
                geometry_df = gpd.read_file(geojson_path)
            except Exception as exc:
                print(f"[brazil-admin] Failed to read simplified GeoJSON: {exc}")
        if geometry_df is None and shapefile_path.exists():
            try:
                geometry_df = gpd.read_file(shapefile_path)
            except Exception as exc:
                print(f"[brazil-admin] Failed to read shapefile: {exc}")

        if geometry_df is None or geometry_df.empty:
            print("[brazil-admin] Municipality geometries unavailable")
            return False

        if geometry_df.crs is None:
            geometry_df.set_crs(epsg=4326, inplace=True)
        elif geometry_df.crs.to_string() != "EPSG:4326":
            geometry_df = geometry_df.to_crs("EPSG:4326")

        csv_df["city_normalized"] = csv_df["city"].astype(str).map(self._normalize)

        geometry_df = self._standardise_municipality_columns(geometry_df)
        geometry_df["name_normalized"] = geometry_df["name"].astype(str).map(self._normalize)

        merged = geometry_df.merge(
            csv_df,
            left_on="name_normalized",
            right_on="city_normalized",
            how="left",
            suffixes=("", "_csv"),
        )

        merged["area_km2"] = self._ensure_area_series(merged)
        merged["latitude"], merged["longitude"] = self._compute_centroids(merged)

        dataset = gpd.GeoDataFrame(
            {
                "muni_id": "muni_" + merged["muni_code"].astype(str),
                "muni_code": merged["muni_code"].astype(str),
                "name": merged["name"].astype(str),
                "state": merged["state"].astype(str),
                "state_code": merged["state_code"],
                "geometry": merged.geometry,
                "area_km2": merged["area_km2"],
                "population": self._coerce_int_series(merged.get("population")),
                "population_proper": self._coerce_int_series(merged.get("population_proper")),
                "capital": merged.get("capital", "").fillna(""),
                "latitude": merged["latitude"],
                "longitude": merged["longitude"],
            },
            geometry=merged.geometry,
            crs="EPSG:4326",
        )

        self.municipalities_gdf = dataset
        self.csv_data = csv_df

        try:
            self.municipalities_tree = STRtree(dataset.geometry.values)
        except Exception as exc:
            print(f"[brazil-admin] Warning: failed to build municipality STRtree: {exc}")
            self.municipalities_tree = None

        print(f"[brazil-admin] Loaded {len(dataset)} municipalities")
        return True

    def _standardise_municipality_columns(self, gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
        gdf = gdf.copy()

        if "CD_MUN" in gdf.columns:
            gdf["muni_code"] = gdf["CD_MUN"].astype(str)
        elif "CD_GEOCMU" in gdf.columns:
            gdf["muni_code"] = gdf["CD_GEOCMU"].astype(str)
        else:
            gdf["muni_code"] = gdf.index.astype(str)

        name_columns = [
            "NM_MUN",
            "NM_MUNICIP",
            "NOME",
            "NAME",
            "municipio",
        ]
        for column in name_columns:
            if column in gdf.columns:
                gdf["name"] = gdf[column].astype(str)
                break
        else:
            gdf["name"] = gdf["muni_code"].astype(str)

        state_name_columns = ["NM_UF", "state", "UF_NAME", "nome_uf"]
        for column in state_name_columns:
            if column in gdf.columns:
                gdf["state"] = gdf[column].astype(str)
                break
        else:
            gdf["state"] = "Unknown"

        state_code_columns = ["SIGLA_UF", "state_code", "UF", "state_abbr"]
        for column in state_code_columns:
            if column in gdf.columns:
                gdf["state_code"] = gdf[column].astype(str)
                break
        else:
            gdf["state_code"] = None

        return gdf

    def _coerce_int_series(self, series: Optional["pd.Series"]) -> "pd.Series":
        if series is None:
            return pd.Series([0] * len(self.municipalities_gdf or []))  # type: ignore[arg-type]
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    def _ensure_area_series(self, gdf: "gpd.GeoDataFrame") -> "pd.Series":
        if "AREA_KM2" in gdf.columns:
            return pd.to_numeric(gdf["AREA_KM2"], errors="coerce").fillna(0.0)

        if GEOSPATIAL_AVAILABLE:
            try:
                projected = gdf.to_crs("EPSG:5880")
                areas = projected.geometry.area / 1_000_000
                return pd.Series(areas, index=gdf.index).fillna(0.0)
            except Exception as exc:
                print(f"[brazil-admin] Failed to compute municipality area: {exc}")
        return pd.Series([0.0] * len(gdf), index=gdf.index)

    def _compute_centroids(self, gdf: "gpd.GeoDataFrame") -> Tuple["pd.Series", "pd.Series"]:
        try:
            projected = gdf.to_crs("EPSG:5880")
            centroids_proj = projected.geometry.centroid
            centroids_ll = gpd.GeoSeries(centroids_proj, crs="EPSG:5880").to_crs("EPSG:4326")
            lat_series = pd.Series(centroids_ll.y.values, index=gdf.index)
            lon_series = pd.Series(centroids_ll.x.values, index=gdf.index)
            return lat_series, lon_series
        except Exception as exc:
            print(f"[brazil-admin] Failed to compute centroids: {exc}")
            zero_series = pd.Series([0.0] * len(gdf), index=gdf.index)
            return zero_series, zero_series

    def _load_states(self) -> bool:
        if not GEOSPATIAL_AVAILABLE:
            return False

        path = self._states_geojson_path()
        if not path.exists():
            print(f"[brazil-admin] State GeoJSON missing at {path}")
            return False

        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            print(f"[brazil-admin] Failed to load states GeoJSON: {exc}")
            return False

        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        name_column = self._first_available_column(
            gdf,
            ["NM_UF", "NOME", "NAME", "state", "UF_NAME"],
            default="name",
        )
        code_column = self._first_available_column(
            gdf,
            ["SIGLA_UF", "UF", "STATE_ABBR", "abbrev"],
            default=None,
        )

        if code_column is None:
            gdf["state_code"] = None
        else:
            gdf["state_code"] = gdf[code_column].astype(str)

        gdf["name"] = gdf[name_column].astype(str)

        try:
            projected = gdf.to_crs("EPSG:5880")
            gdf["area_km2"] = projected.geometry.area / 1_000_000
            centroids_proj = projected.geometry.centroid
            centroids_ll = gpd.GeoSeries(centroids_proj, crs="EPSG:5880").to_crs("EPSG:4326")
            gdf["latitude"] = centroids_ll.y.values
            gdf["longitude"] = centroids_ll.x.values
        except Exception as exc:
            print(f"[brazil-admin] Failed to compute state metadata: {exc}")
            gdf["area_km2"] = 0.0
            gdf["latitude"] = 0.0
            gdf["longitude"] = 0.0

        default_codes = pd.Series(gdf.index.astype(str), index=gdf.index)
        filled_codes = gdf["state_code"].fillna(default_codes).astype(str)
        gdf["state_code"] = filled_codes
        gdf["state_id"] = filled_codes.map(lambda value: f"state_{value}")

        self.states_gdf = gdf
        try:
            self.states_tree = STRtree(gdf.geometry.values)
        except Exception as exc:
            print(f"[brazil-admin] Warning: failed to build state STRtree: {exc}")
            self.states_tree = None

        print(f"[brazil-admin] Loaded {len(gdf)} states")
        return True

    def _first_available_column(
        self,
        gdf: "gpd.GeoDataFrame",
        candidates: Sequence[str],
        *,
        default: Optional[str],
    ) -> str:
        for column in candidates:
            if column in gdf.columns:
                return column
        if default is not None:
            if default not in gdf.columns:
                gdf[default] = None
            return default
        return None

    @staticmethod
    def _geo_interface(geometry: Any) -> Optional[Dict[str, Any]]:
        if geometry is None:
            return None
        return geometry.__geo_interface__ if hasattr(geometry, "__geo_interface__") else None

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        text = unicodedata.normalize("NFKD", text)
        return "".join(char for char in text if not unicodedata.combining(char))

    def _fuzzy_select(
        self,
        gdf: "pd.DataFrame",
        column: str,
        query: str,
        *,
        max_results: int = 5,
        cutoff: float = 0.75,
    ) -> "pd.DataFrame":
        import difflib

        if column not in gdf.columns:
            return gdf.iloc[0:0]
        series = gdf[column].astype(str)
        normalized = series.map(self._normalize)
        target = self._normalize(query)
        unique_values = list(dict.fromkeys(normalized.tolist()))
        candidates = difflib.get_close_matches(target, unique_values, n=max_results, cutoff=cutoff)
        if not candidates:
            return gdf.iloc[0:0]
        return gdf[normalized.isin(candidates)]

    def _capabilities_payload(self) -> Dict[str, Any]:
        meta = self.dataset_metadata.get(DATASET_ID, {})
        muni_total = (
            int(self.municipalities_gdf.shape[0])
            if self.municipalities_gdf is not None
            else 0
        )
        state_total = (
            int(self.states_gdf.shape[0])
            if self.states_gdf is not None
            else 0
        )
        geometry_source = (
            "simplified_geojson"
            if self._municipality_geojson_path().exists()
            else "shapefile"
        )

        payload = {
            "name": "brazil_admin",
            "description": (
                "Brazilian municipalities and states with population, area, "
                "and geospatial boundaries."
            ),
            "version": meta.get("last_updated", "2024"),
            "dataset": meta.get("title", "Brazilian Administrative Boundaries"),
            "geometry_source": geometry_source,
            "coverage": {
                "total_municipalities": muni_total,
                "total_states": state_total,
            },
            "tools": [
                "describe_capabilities",
                "query_support",
                "get_admin_dataset_metadata",
                "get_boundary_counts",
                "get_state_boundaries",
                "list_state_names",
                "search_municipalities",
                "get_municipalities_dataset_metadata",
                "get_municipalities_by_filter",
                "get_municipality_boundaries",
                "get_municipalities_in_bounds",
                "get_municipality_statistics",
                "find_municipalities_near_point",
                "get_top_brazilian_cities_by_population",
                "get_population_by_state",
                "get_municipality_area_histogram",
                "run_query",
            ],
        }
        return payload

    # ------------------------------------------------------------------ tool registration
    def _register_capabilities_tool(self) -> None:
        @self.mcp.tool()
        def describe_capabilities(format: str = "json") -> str:  # type: ignore[misc]
            """Describe the Brazilian administrative dataset and capabilities."""

            payload = self._capabilities_payload()
            if format == "json":
                return json.dumps(payload)
            return str(payload)

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> str:  # type: ignore[misc]
            """Judge whether the dataset can help answer the query."""

            intent = self._classify_intent(query)
            response = {
                "server": SERVER_ID,
                "query": query,
                "supported": intent.supported,
                "score": intent.score,
                "reasons": intent.reasons,
            }
            return json.dumps(response)

    def _register_dataset_metadata_tool(self) -> None:
        @self.mcp.tool()
        def get_admin_dataset_metadata() -> Dict[str, Any]:  # type: ignore[misc]
            """Return metadata for Brazilian states and municipalities."""

            return self._admin_metadata()

    def _register_boundary_counts_tool(self) -> None:
        @self.mcp.tool()
        def get_boundary_counts() -> Dict[str, Any]:  # type: ignore[misc]
            """Return counts and total areas for states and municipalities."""

            meta = self._admin_metadata()
            if "error" in meta:
                return meta
            return {
                "boundary_types": meta.get("boundary_types"),
                "counts": meta.get("counts"),
                "total_area_km2": meta.get("total_area_km2"),
            }

    def _register_state_boundaries_tool(self) -> None:
        @self.mcp.tool()
        def get_state_boundaries(
            state_names: List[str],
            include_metadata: bool = True,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Return state polygons given names or ISO codes."""

            return self._get_state_boundaries(state_names, include_metadata=include_metadata)

    def _register_list_states_tool(self) -> None:
        @self.mcp.tool()
        def list_state_names() -> Dict[str, Any]:  # type: ignore[misc]
            """List state names and two-letter codes."""

            if not self._states_loaded or self.states_gdf is None or self.states_gdf.empty:
                return {"error": "State data not available"}
            records = []
            for _, row in self.states_gdf.iterrows():
                records.append(
                    {
                        "name": str(row.get("name")),
                        "state_code": str(row.get("state_code")) if row.get("state_code") else None,
                    }
                )
            return {"states": records, "total_count": len(records)}

    def _register_search_municipalities_tool(self) -> None:
        @self.mcp.tool()
        def search_municipalities(
            query: str,
            state: Optional[str] = None,
            max_results: int = 20,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Fuzzy search for municipality names, optionally scoped by state."""

            return self._search_municipalities(query, state=state, max_results=max_results)

    def _register_municipalities_dataset_metadata_tool(self) -> None:
        @self.mcp.tool()
        def get_municipalities_dataset_metadata() -> Dict[str, Any]:  # type: ignore[misc]
            """Return metadata about the municipalities dataset."""

            return self._municipalities_metadata()

    def _register_municipalities_filter_tool(self) -> None:
        @self.mcp.tool()
        def get_municipalities_by_filter(
            state: Optional[str] = None,
            min_population: Optional[int] = None,
            max_population: Optional[int] = None,
            capital_only: bool = False,
            limit: int = 6000,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Filter municipalities by state, population, and capital status."""

            return self._municipalities_by_filter(
                state=state,
                min_population=min_population,
                max_population=max_population,
                capital_only=capital_only,
                limit=limit,
            )

    def _register_municipality_boundaries_tool(self) -> None:
        @self.mcp.tool()
        def get_municipality_boundaries(
            municipality_names: List[str],
            include_metadata: bool = True,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Return boundaries for specific municipalities (accent-insensitive)."""

            return self._get_municipality_boundaries(
                municipality_names,
                include_metadata=include_metadata,
            )

    def _register_municipalities_bounds_tool(self) -> None:
        @self.mcp.tool()
        def get_municipalities_in_bounds(
            north: float,
            south: float,
            east: float,
            west: float,
            min_area_km2: Optional[float] = None,
            limit: int = 6000,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Return municipalities intersecting a bounding box."""

            return self._municipalities_in_bounds(
                north=north,
                south=south,
                east=east,
                west=west,
                min_area_km2=min_area_km2,
                limit=limit,
            )

    def _register_municipality_statistics_tool(self) -> None:
        @self.mcp.tool()
        def get_municipality_statistics(
            group_by: str = "state",
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Aggregate statistics for municipalities by state, capital status, or overall."""

            return self._municipality_statistics(group_by=group_by)

    def _register_municipalities_near_point_tool(self) -> None:
        @self.mcp.tool()
        def find_municipalities_near_point(
            latitude: float,
            longitude: float,
            radius_km: float = 50.0,
            limit: int = 20,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Return municipalities near a coordinate within a radius."""

            return self._municipalities_near_point(
                latitude=latitude,
                longitude=longitude,
                radius_km=radius_km,
                limit=limit,
            )

    def _register_top_cities_tool(self) -> None:
        @self.mcp.tool()
        def get_top_brazilian_cities_by_population(
            top_n: int = 10,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Return the top-N cities ranked by population."""

            return self._top_cities(top_n=top_n)

    def _register_population_by_state_tool(self) -> None:
        @self.mcp.tool()
        def get_population_by_state() -> Dict[str, Any]:  # type: ignore[misc]
            """Return total population per state."""

            return self._population_by_state()

    def _register_area_histogram_tool(self) -> None:
        @self.mcp.tool()
        def get_municipality_area_histogram(bins: int = 10) -> Dict[str, Any]:  # type: ignore[misc]
            """Return a histogram-style distribution of municipality area sizes."""

            return self._municipality_area_histogram(bins=bins)

    # ------------------------------------------------------------------ tool implementations
    def _admin_metadata(self) -> Dict[str, Any]:
        if not GEOSPATIAL_AVAILABLE or self.municipalities_gdf is None or self.states_gdf is None:
            return {"error": "Geospatial data not available"}

        muni_total = int(self.municipalities_gdf.shape[0])
        state_total = int(self.states_gdf.shape[0])
        muni_area_total = (
            float(self.municipalities_gdf["area_km2"].sum())
            if "area_km2" in self.municipalities_gdf
            else 0.0
        )
        state_area_total = (
            float(self.states_gdf["area_km2"].sum())
            if "area_km2" in self.states_gdf
            else 0.0
        )
        return {
            "Name": "Brazilian Administrative Boundaries",
            "Description": "States and municipalities for Brazil (geometries, areas)",
            "Version": "2024",
            "boundary_types": ["state", "municipality"],
            "counts": {"states": state_total, "municipalities": muni_total},
            "total_area_km2": {"states": state_area_total, "municipalities": muni_area_total},
        }

    def _municipalities_metadata(self) -> Dict[str, Any]:
        if (
            not GEOSPATIAL_AVAILABLE
            or self.municipalities_gdf is None
        ):
            return {"error": "Municipality data not available"}
        meta = self._admin_metadata()
        geometry_source = (
            "simplified_geojson"
            if self._municipality_geojson_path().exists()
            else "shapefile"
        )
        return {
            "Name": "Brazilian Municipalities",
            "Description": "Administrative boundaries and demographics for Brazil's municipalities",
            "Version": "2024",
            "geometry_source": geometry_source,
            "total_municipalities": meta.get("counts", {}).get("municipalities"),
            "states": meta.get("counts", {}).get("states"),
            "total_population": int(self.municipalities_gdf["population"].sum()),
            "total_area_km2": float(self.municipalities_gdf["area_km2"].sum()),
        }

    def _get_state_boundaries(
        self,
        state_names: Iterable[str],
        *,
        include_metadata: bool,
    ) -> Dict[str, Any]:
        if not GEOSPATIAL_AVAILABLE or self.states_gdf is None or self.states_gdf.empty:
            return {"error": "State data not available"}

        if not state_names:
            return {"error": "No state names provided"}

        results = []
        not_found: List[str] = []

        names_normalized = self.states_gdf["name"].astype(str).map(self._normalize)
        codes_normalized = (
            self.states_gdf["state_code"].astype(str).map(self._normalize)
            if "state_code" in self.states_gdf.columns
            else None
        )

        for raw_query in state_names:
            target = self._normalize(raw_query)
            matches = self.states_gdf[names_normalized == target]
            if codes_normalized is not None:
                code_matches = self.states_gdf[codes_normalized == target]
                matches = pd.concat([matches, code_matches]).drop_duplicates()

            if matches.empty:
                contains = self.states_gdf[names_normalized.str.contains(target, na=False)]
                if codes_normalized is not None:
                    code_contains = self.states_gdf[codes_normalized.str.contains(target, na=False)]
                    contains = pd.concat([contains, code_contains]).drop_duplicates()
                matches = contains

            if matches.empty:
                matches = self._fuzzy_select(
                self.states_gdf,
                "name",
                raw_query,
                max_results=5,
                cutoff=0.7,
            )

            if matches.empty:
                not_found.append(str(raw_query))
                continue

            for _, record in matches.iterrows():
                geometry = self._geo_interface(record.geometry)
                state_payload: Dict[str, Any] = {
                    "id": record.get("state_id"),
                    "name": record.get("name"),
                    "state_code": record.get("state_code"),
                    "geometry": geometry,
                }
                if include_metadata:
                    state_payload.update(
                        {
                            "area_km2": float(record.get("area_km2", 0.0)),
                            "latitude": float(record.get("latitude", 0.0)),
                            "longitude": float(record.get("longitude", 0.0)),
                        }
                    )
                results.append(state_payload)

        metadata = {
            "requested": len(list(state_names)),
            "found": len(results),
            "missing": len(not_found),
        }
        if include_metadata and results:
            metadata["total_area_km2"] = float(sum(item.get("area_km2", 0.0) for item in results))

        return {
            "states": results,
            "total_count": len(results),
            "not_found": not_found,
            "metadata": metadata,
        }

    def _search_municipalities(
        self,
        query: str,
        *,
        state: Optional[str],
        max_results: int,
    ) -> Dict[str, Any]:
        if (
            not GEOSPATIAL_AVAILABLE
            or self.municipalities_gdf is None
            or self.municipalities_gdf.empty
        ):
            return {"error": "Municipality data not available"}

        df = self.municipalities_gdf
        if state:
            normalized_state = self._normalize(state)
            df = df[
                (df["state"].map(self._normalize) == normalized_state)
                | (df["state_code"].map(self._normalize) == normalized_state)
            ]

        subset = self._fuzzy_select(df, "name", query, max_results=max_results, cutoff=0.65)
        rows = []
        for _, record in subset.head(max_results).iterrows():
            rows.append(
                {
                    "name": record.get("name"),
                    "state": record.get("state"),
                    "state_code": record.get("state_code"),
                    "population": int(record.get("population") or 0),
                    "area_km2": float(record.get("area_km2") or 0.0),
                }
            )

        return {
            "municipalities": rows,
            "total_count": len(rows),
            "query": query,
            "state_scope": state,
        }

    def _municipalities_by_filter(
        self,
        *,
        state: Optional[str],
        min_population: Optional[int],
        max_population: Optional[int],
        capital_only: bool,
        limit: int,
    ) -> Dict[str, Any]:
        if (
            not GEOSPATIAL_AVAILABLE
            or self.municipalities_gdf is None
        ):
            return {"error": "Municipality data not available"}

        df = self.municipalities_gdf.copy()

        if state:
            state_upper = state.upper()
            df = df[
                (df["state"].str.upper() == state_upper)
                | (df["state"].str.contains(state, case=False, na=False))
                | (df["state_code"].str.upper() == state_upper)
            ]

        if min_population is not None:
            df = df[df["population"] >= int(min_population)]

        if max_population is not None:
            df = df[df["population"] <= int(max_population)]

        if capital_only:
            df = df[df["capital"].isin(["admin", "primary", "minor"])]

        df = df.sort_values("population", ascending=False).head(max(1, int(limit)))

        rows = []
        for _, record in df.iterrows():
            rows.append(
                {
                    "id": record.get("muni_id"),
                    "name": record.get("name"),
                    "state": record.get("state"),
                    "population": int(record.get("population", 0)),
                    "population_proper": int(record.get("population_proper", 0)),
                    "area_km2": float(record.get("area_km2", 0.0)),
                    "capital": record.get("capital", ""),
                    "latitude": float(record.get("latitude", 0.0)),
                    "longitude": float(record.get("longitude", 0.0)),
                    "geometry": self._geo_interface(record.geometry),
                }
            )

        total_population = int(df["population"].sum()) if not df.empty else 0
        average_population = int(df["population"].mean()) if not df.empty else 0
        total_area = float(df["area_km2"].sum()) if not df.empty else 0.0

        metadata = {
            "filter_applied": {
                "state": state,
                "min_population": min_population,
                "max_population": max_population,
                "capital_only": capital_only,
            },
            "total_population": total_population,
            "average_population": average_population,
            "total_area_km2": total_area,
        }

        return {
            "municipalities": rows,
            "total_count": len(rows),
            "metadata": metadata,
            "summary": f"Found {len(rows)} municipalities matching criteria",
        }

    def _get_municipality_boundaries(
        self,
        municipality_names: Iterable[str],
        *,
        include_metadata: bool,
    ) -> Dict[str, Any]:
        if (
            not GEOSPATIAL_AVAILABLE
            or self.municipalities_gdf is None
        ):
            return {"error": "Municipality data not available"}

        if not municipality_names:
            return {"error": "No municipality names provided"}

        name_series = self.municipalities_gdf["name"].astype(str)
        normalized_names = name_series.map(self._normalize)

        results = []
        missing: List[str] = []

        for raw_query in municipality_names:
            normalized_query = self._normalize(raw_query)
            matches = self.municipalities_gdf[normalized_names == normalized_query]
            if matches.empty:
                matches = self.municipalities_gdf[
                normalized_names.str.contains(normalized_query, na=False)
            ]
            if matches.empty:
                matches = self._fuzzy_select(
                self.municipalities_gdf,
                "name",
                raw_query,
                max_results=5,
                cutoff=0.7,
            )

            if matches.empty:
                missing.append(str(raw_query))
                continue

            match = matches.sort_values("population", ascending=False).iloc[0]
            payload: Dict[str, Any] = {
                "id": match.get("muni_id"),
                "name": match.get("name"),
                "state": match.get("state"),
                "geometry": self._geo_interface(match.geometry),
            }
            if include_metadata:
                payload.update(
                    {
                        "population": int(match.get("population", 0)),
                        "population_proper": int(match.get("population_proper", 0)),
                        "area_km2": float(match.get("area_km2", 0.0)),
                        "capital": match.get("capital", ""),
                        "latitude": float(match.get("latitude", 0.0)),
                        "longitude": float(match.get("longitude", 0.0)),
                    }
                )
            results.append(payload)

        metadata = {
            "requested": len(list(municipality_names)),
            "found": len(results),
            "missing": len(missing),
        }
        if include_metadata and results:
            metadata["total_population"] = int(sum(item.get("population", 0) for item in results))
            metadata["total_area_km2"] = float(sum(item.get("area_km2", 0.0) for item in results))

        return {
            "municipalities": results,
            "total_count": len(results),
            "not_found": missing,
            "metadata": metadata,
        }

    def _municipalities_in_bounds(
        self,
        *,
        north: float,
        south: float,
        east: float,
        west: float,
        min_area_km2: Optional[float],
        limit: int,
    ) -> Dict[str, Any]:
        if (
            not GEOSPATIAL_AVAILABLE
            or self.municipalities_gdf is None
        ):
            return {"error": "Municipality data not available"}
        if box is None:
            return {"error": "Shapely not available"}

        bbox = box(west, south, east, north)

        if self.municipalities_tree is not None:
            try:
                possible_indices = list(self.municipalities_tree.query(bbox))
                candidates = self.municipalities_gdf.iloc[possible_indices]
                intersects = candidates[candidates.geometry.intersects(bbox)]
            except Exception as exc:
                print(f"[brazil-admin] STRtree query failed: {exc}")
                intersects = self.municipalities_gdf[
            self.municipalities_gdf.geometry.intersects(bbox)
        ]
        else:
            intersects = self.municipalities_gdf[
            self.municipalities_gdf.geometry.intersects(bbox)
        ]

        if min_area_km2 is not None:
            intersects = intersects[intersects["area_km2"] >= float(min_area_km2)]

        intersects = intersects.sort_values("population", ascending=False).head(max(1, int(limit)))

        rows = []
        for _, record in intersects.iterrows():
            rows.append(
                {
                    "id": record.get("muni_id"),
                    "name": record.get("name"),
                    "state": record.get("state"),
                    "population": int(record.get("population", 0)),
                    "area_km2": float(record.get("area_km2", 0.0)),
                    "latitude": float(record.get("latitude", 0.0)),
                    "longitude": float(record.get("longitude", 0.0)),
                    "geometry": self._geo_interface(record.geometry),
                }
            )

        metadata = {
            "total_population": int(intersects["population"].sum()) if not intersects.empty else 0,
            "total_area_km2": float(intersects["area_km2"].sum()) if not intersects.empty else 0.0,
            "states": list(intersects["state"].unique()),
        }

        return {
            "municipalities": rows,
            "total_count": len(rows),
            "bounds": {"north": north, "south": south, "east": east, "west": west},
            "metadata": metadata,
            "summary": f"Found {len(rows)} municipalities in bounding box",
        }

    def _municipality_statistics(self, *, group_by: str) -> Dict[str, Any]:
        if (
            not GEOSPATIAL_AVAILABLE
            or self.municipalities_gdf is None
        ):
            return {"error": "Municipality data not available"}

        df = self.municipalities_gdf
        if group_by == "state":
            grouped = df.groupby("state").agg({
                "muni_id": "count",
                "population": ["sum", "mean", "median"],
                "area_km2": "sum",
            }).round(2)
            stats: Dict[str, Any] = {}
            for state in grouped.index:
                stats[str(state)] = {
                    "municipality_count": int(grouped.loc[state, ("muni_id", "count")]),
                    "total_population": int(grouped.loc[state, ("population", "sum")]),
                    "avg_population": int(grouped.loc[state, ("population", "mean")]),
                    "median_population": int(grouped.loc[state, ("population", "median")]),
                    "total_area_km2": float(grouped.loc[state, ("area_km2", "sum")]),
                }
            return {
                "statistics": stats,
                "group_by": "state",
                "total_municipalities": len(df),
                "total_states": len(stats),
            }

        if group_by == "capital":
            stats = {}
            for capital_type in ["primary", "admin", "minor", ""]:
                if capital_type:
                    subset = df[df["capital"] == capital_type]
                    label = f"capital_{capital_type}"
                else:
                    subset = df[df["capital"] == ""]
                    label = "non_capital"
                if subset.empty:
                    continue
                stats[label] = {
                    "count": int(len(subset)),
                    "total_population": int(subset["population"].sum()),
                    "avg_population": int(subset["population"].mean()),
                    "total_area_km2": float(subset["area_km2"].sum()),
                }
            return {
                "statistics": stats,
                "group_by": "capital",
                "total_municipalities": len(df),
            }

        pop_series = df["population"]
        area_series = df["area_km2"]
        return {
            "statistics": {
                "total_municipalities": len(df),
                "total_population": int(pop_series.sum()),
                "population": {
                    "mean": int(pop_series.mean()),
                    "median": int(pop_series.median()),
                    "min": int(pop_series.min()),
                    "max": int(pop_series.max()),
                    "std": int(pop_series.std()),
                },
                "area_km2": {
                    "total": float(area_series.sum()),
                    "mean": float(area_series.mean()),
                    "median": float(area_series.median()),
                    "min": float(area_series.min()),
                    "max": float(area_series.max()),
                },
                "states": list(df["state"].unique()),
                "state_count": int(df["state"].nunique()),
                "capitals_count": int(len(df[df["capital"] != ""])),
            },
            "group_by": "all",
        }

    def _municipalities_near_point(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_km: float,
        limit: int,
    ) -> Dict[str, Any]:
        if not GEOSPATIAL_AVAILABLE or self.municipalities_gdf is None or Point is None:
            return {"error": "Municipality data not available"}

        point = Point(float(longitude), float(latitude))

        df_projected = self.municipalities_gdf.to_crs("EPSG:3857")
        point_projected = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

        buffer_m = float(radius_km) * 1000

        centroids_proj = df_projected.geometry.centroid
        distances = centroids_proj.distance(point_projected)
        nearby_mask = distances <= buffer_m
        nearby = self.municipalities_gdf.loc[nearby_mask].copy()

        if not nearby.empty:
            nearby_proj = nearby.to_crs("EPSG:3857")
            nearby["distance_km"] = nearby_proj.geometry.centroid.distance(point_projected) / 1000
            nearby = nearby.sort_values("distance_km").head(max(1, int(limit)))

        rows = []
        for _, record in nearby.iterrows():
            rows.append(
                {
                    "id": record.get("muni_id"),
                    "name": record.get("name"),
                    "state": record.get("state"),
                    "population": int(record.get("population", 0)),
                    "area_km2": float(record.get("area_km2", 0.0)),
                    "distance_km": round(float(record.get("distance_km", 0.0)), 2),
                    "latitude": float(record.get("latitude", 0.0)),
                    "longitude": float(record.get("longitude", 0.0)),
                    "geometry": self._geo_interface(record.geometry),
                }
            )

        metadata = {
            "nearest": rows[0]["name"] if rows else None,
            "farthest": rows[-1]["name"] if rows else None,
            "total_population_in_radius": int(sum(item.get("population", 0) for item in rows)),
        }

        return {
            "municipalities": rows,
            "total_count": len(rows),
            "search_point": {"latitude": latitude, "longitude": longitude, "radius_km": radius_km},
            "metadata": metadata,
            "summary": f"Found {len(rows)} municipalities within {radius_km}km",
        }

    def _top_cities(self, *, top_n: int) -> Dict[str, Any]:
        if pd is None or self.csv_data is None or self.csv_data.empty:
            return {"error": "CSV data not available"}
        df = self.csv_data.copy()
        if "population" not in df.columns or "city" not in df.columns:
            return {"error": "CSV missing required columns"}
        df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0)
        df = df.sort_values("population", ascending=False).head(max(1, int(top_n)))
        data = [
            {"city": row["city"], "population": int(row["population"])}
            for _, row in df.iterrows()
        ]
        return {
            "visualization_type": "comparison",
            "data": data,
            "chart_config": {
                "x_axis": "city",
                "y_axis": "population",
                "title": f"Top {max(1, int(top_n))} Brazilian Cities by Population",
                "chart_type": "bar",
            },
        }

    def _population_by_state(self) -> Dict[str, Any]:
        if pd is None:
            return {"error": "pandas not available"}

        if self.municipalities_gdf is not None and not self.municipalities_gdf.empty:
            df = self.municipalities_gdf[["state", "population"]].copy()
            df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0)
            agg = df.groupby("state", dropna=False)["population"].sum().reset_index()
        elif (
            self.csv_data is not None
            and not self.csv_data.empty
            and {"admin_name", "population"}.issubset(self.csv_data.columns)
        ):
            df = self.csv_data[["admin_name", "population"]].copy()
            df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0)
            agg = df.groupby("admin_name", dropna=False)["population"].sum().reset_index().rename(
                columns={"admin_name": "state"}
            )
        else:
            return {"error": "No state/population columns available"}

        agg = agg.sort_values("population", ascending=False)
        data = [
            {"state": str(row["state"]), "population": int(row["population"])}
            for _, row in agg.iterrows()
        ]
        return {
            "visualization_type": "comparison",
            "data": data,
            "chart_config": {
                "x_axis": "state",
                "y_axis": "population",
                "title": "Total Population by Brazilian State",
                "chart_type": "bar",
            },
        }

    def _municipality_area_histogram(self, *, bins: int) -> Dict[str, Any]:
        if pd is None or np is None or self.municipalities_gdf is None:
            return {"error": "Municipality areas not available"}
        if "area_km2" not in self.municipalities_gdf:
            return {"error": "Municipality area column missing"}
        series = pd.to_numeric(self.municipalities_gdf["area_km2"], errors="coerce").dropna()
        if series.empty:
            return {"error": "No area data"}
        bins = max(2, int(bins))
        counts, edges = np.histogram(series.values, bins=bins)
        labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(len(edges) - 1)]
        data = [{"bin": labels[i], "count": int(counts[i])} for i in range(len(counts))]
        return {
            "visualization_type": "comparison",
            "data": data,
            "chart_config": {
                "x_axis": "bin",
                "y_axis": "count",
                "title": "Distribution of Municipality Areas (km²)",
                "chart_type": "bar",
            },
        }

    # ------------------------------------------------------------------ query support
    def _classify_intent(self, query: str) -> SupportIntent:
        if not query.strip():
            return SupportIntent(supported=False, score=0.0, reasons=["Empty query"])

        if self._anthropic_client:
            prompt = self._intent_prompt(query)
            try:
                response = call_llm_with_retries_sync(
                    lambda: self._anthropic_client.messages.create(
                        model="claude-3-5-haiku-20241022",
                        max_tokens=128,
                        temperature=0,
                        system="Respond with valid JSON only.",
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    provider="anthropic.brazil_admin_router",
                )
                text = response.content[0].text.strip()
                data = self._parse_json_response(text)
                if data is None:
                    raise ValueError("LLM returned non-JSON")
                supported = bool(data.get("supported", False))
                reason = str(data.get("reason")) if data.get("reason") else "LLM routing"
                score = 0.85 if supported else 0.1
                return SupportIntent(supported=supported, score=score, reasons=[reason])
            except Exception as exc:  # pragma: no cover - network failures
                return SupportIntent(
                    supported=self._heuristic_support(query),
                    score=0.35,
                    reasons=[f"LLM intent unavailable: {exc}"],
                )

        supported = self._heuristic_support(query)
        score = 0.6 if supported else 0.15
        reasons = (
            ["Heuristic location/administrative keyword match"]
            if supported
            else ["No obvious administrative keywords"]
        )
        return SupportIntent(supported=supported, score=score, reasons=reasons)

    def _intent_prompt(self, query: str) -> str:
        capabilities = self._capabilities_payload()
        return (
            "Decide whether the following question should be answered using the"
            " Brazilian administrative boundaries dataset. The dataset includes"
            " polygons, centroids, and demographic attributes for all Brazilian"
            " municipalities and states. Respond with JSON of the form"
            " {\"supported\": true|false, \"reason\": \"short explanation\"}."
            f"\nDataset summary: {json.dumps(capabilities)}"
            f"\nQuestion: {query}"
            "\nMark supported=true when the query needs Brazilian municipal or state"
            " context (boundaries, population, locations, filtering by area)."
        )

    @staticmethod
    def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                try:
                    data = json.loads(text[start : end + 1])
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    return None
        return None

    def _heuristic_support(self, query: str) -> bool:
        print("[brazil-admin] Heuristic support check")
        lowered = query.lower()
        keywords = [
            "municipality",
            "municipalities",
            "prefeitura",
            "boundary",
            "boundaries",
            "geojson",
            "polygon",
            "shapefile",
            "microregion",
            "mesoregion",
            "ibge",
            "map",
        ]
        return any(word in lowered for word in keywords)

    # ------------------------------------------------------------------ run_query
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        if not GEOSPATIAL_AVAILABLE or self.municipalities_gdf is None or self.states_gdf is None:
            message = MessagePayload(
                level="warning",
                text="Geospatial dependencies unavailable; dataset is offline.",
            )
            return RunQueryResponse(
                server=SERVER_ID,
                query=query,
                facts=[],
                citations=[],
                artifacts=[],
                messages=[message],
                kg=KnowledgeGraphPayload(),
            )

        matched_states = self._detect_states(query)
        matched_municipalities = self._detect_municipalities(query, max_results=5)

        citation = self._dataset_citation()
        facts: List[FactPayload] = []

        overview = self._admin_metadata()
        facts.append(
            FactPayload(
                id="brazil_admin_overview",
                text=(
                    f"The dataset tracks {overview['counts']['municipalities']} municipalities "
                    f"across {overview['counts']['states']} states in Brazil, "
                    f"covering roughly {int(overview['total_area_km2']['municipalities']):,} km² "
                    "of municipal territory."
                ),
                citation_id=citation.id,
            )
        )

        for _, row in matched_states.iterrows():
            facts.append(
                FactPayload(
                    id=f"state_{self._slugify(row.get('state_code') or row.get('name'))}",
                    text=(
                        f"State {row.get('name')} spans about "
                        f"{round(float(row.get('area_km2', 0.0)), 1):,} km² "
                        f"and its centroid sits near {round(float(row.get('latitude', 0.0)), 3)}, "
                        f"{round(float(row.get('longitude', 0.0)), 3)}."
                    ),
                    citation_id=citation.id,
                )
            )

        if matched_municipalities is not None and not matched_municipalities.empty:
            for _, row in matched_municipalities.iterrows():
                facts.append(
                    FactPayload(
                        id=f"muni_{self._slugify(row.get('muni_code'))}",
                        text=(
                            f"Municipality {row.get('name')} ({row.get('state')}) has around "
                            f"{int(row.get('population', 0)):,} residents and covers roughly "
                            f"{round(float(row.get('area_km2', 0.0)), 1):,} km²."
                        ),
                        citation_id=citation.id,
                    )
                )

        artifacts: List[ArtifactPayload] = []
        if matched_municipalities is not None and not matched_municipalities.empty:
            export = self._export_geojson(
                matched_municipalities.geometry,
                identifier="municipalities_focus",
                legend_label="Municipalities",
                legend_color="#1E88E5",
            )
            artifacts.append(
                ArtifactPayload(
                    id="municipality_map",
                    type="map",
                    title="Highlighted Brazilian municipalities",
                    geojson_url=export.url,
                    metadata=export.metadata,
                )
            )

        messages: List[MessagePayload] = []

        kg = self._assemble_kg(
            matched_states=matched_states,
            matched_municipalities=matched_municipalities,
        )

        return RunQueryResponse(
            server=SERVER_ID,
            query=query,
            facts=facts,
            citations=[citation],
            artifacts=artifacts,
            messages=messages,
            kg=kg,
        )

    def _dataset_citation(self) -> CitationPayload:
        meta = self.dataset_metadata.get(DATASET_ID, {})
        return CitationPayload(
            id="brazil_admin_dataset",
            server=SERVER_ID,
            tool="run_query",
            title=meta.get("title", "Brazilian Administrative Boundaries"),
            source_type="Dataset",
            description=meta.get("citation") or meta.get("description", ""),
            url=meta.get("source"),
        )

    def _detect_states(self, query: str) -> "pd.DataFrame":
        if self.states_gdf is None or self.states_gdf.empty:
            return pd.DataFrame()
        lowered = query.lower()
        matches = []
        for _, row in self.states_gdf.iterrows():
            name = str(row.get("name", "")).lower()
            code = str(row.get("state_code", "")).lower()
            if name and name in lowered:
                matches.append(row)
            elif code and code in lowered:
                matches.append(row)
        if matches:
            return self.states_gdf.loc[[r.name for r in matches]]
        subset = self._fuzzy_select(self.states_gdf, "name", query, max_results=5, cutoff=0.6)
        return subset

    def _detect_municipalities(self, query: str, *, max_results: int) -> "pd.DataFrame":
        if self.municipalities_gdf is None or self.municipalities_gdf.empty:
            return pd.DataFrame()
        lowered = query.lower()
        matches = self.municipalities_gdf[
            self.municipalities_gdf["name"].str.lower().apply(lambda value: value in lowered)
        ]
        if matches.empty:
            matches = self._fuzzy_select(
            self.municipalities_gdf,
            "name",
            query,
            max_results=max_results,
            cutoff=0.7,
        )
        return matches.head(max_results)

    def _export_geojson(
        self,
        geometries: "pd.Series",
        *,
        identifier: str,
        legend_label: Optional[str] = None,
        legend_color: str = "#1E88E5",
    ) -> GeoJSONExport:
        self.static_maps_dir.mkdir(parents=True, exist_ok=True)
        features: List[Dict[str, Any]] = []
        total_area = 0.0
        min_lon: Optional[float] = None
        min_lat: Optional[float] = None
        max_lon: Optional[float] = None
        max_lat: Optional[float] = None
        label = legend_label or identifier.replace("_", " ").title()
        label_lower = label.lower()
        for geom in geometries:
            if geom is None or not hasattr(geom, "__geo_interface__"):
                continue
            geo_copy = geom
            if not isinstance(geo_copy, BaseGeometry):  # type: ignore[arg-type]
                try:
                    from shapely.geometry import shape  # type: ignore

                    geo_copy = shape(geom)
                except Exception:
                    continue
            geojson = geo_copy.__geo_interface__
            features.append({
                "type": "Feature",
                "geometry": geojson,
                "properties": {"country": label_lower},
            })
            try:
                area_km2 = (
                    geo_copy.to_crs("EPSG:5880").area / 1_000_000
                )  # type: ignore[attr-defined]
            except Exception:
                area_km2 = 0.0
            total_area += float(area_km2)
            try:
                minx, miny, maxx, maxy = geo_copy.bounds
                min_lon = minx if min_lon is None else min(min_lon, minx)
                min_lat = miny if min_lat is None else min(min_lat, miny)
                max_lon = maxx if max_lon is None else max(max_lon, maxx)
                max_lat = maxy if max_lat is None else max(max_lat, maxy)
            except Exception:
                pass

        identifier_bytes = identifier.encode("utf-8")
        digest = hashlib.sha256(
            identifier_bytes + json.dumps(features).encode("utf-8")
        ).hexdigest()[:10]
        filename = f"brazil_admin_{identifier}_{digest}.geojson"
        path = self.static_maps_dir / filename

        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"type": "FeatureCollection", "features": features}, handle)

        metadata: Dict[str, Any] = {
            "feature_count": len(features),
            "approx_area_km2": round(total_area, 2),
            "geometry_type": "polygon",
            "legend": {
                "title": label,
                "items": [
                    {
                        "label": label,
                        "color": legend_color,
                        "description": f"{len(features)} features",
                    }
                ],
            },
        }
        if (
            min_lon is not None
            and min_lat is not None
            and max_lon is not None
            and max_lat is not None
        ):
            padding_lon = max((max_lon - min_lon) * 0.05, 0.25)
            padding_lat = max((max_lat - min_lat) * 0.05, 0.25)
            metadata["bounds"] = {
                "west": float(min_lon - padding_lon),
                "south": float(min_lat - padding_lat),
                "east": float(max_lon + padding_lon),
                "north": float(max_lat + padding_lat),
            }
            metadata["center"] = {
                "lon": float((min_lon + max_lon) / 2),
                "lat": float((min_lat + max_lat) / 2),
            }
        return GeoJSONExport(url=f"/static/maps/{filename}", metadata=metadata)

    def _assemble_kg(
        self,
        *,
        matched_states: "pd.DataFrame",
        matched_municipalities: "pd.DataFrame",
    ) -> KnowledgeGraphPayload:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        nodes.append({"id": "brazil", "label": "Brazil", "type": "Location"})

        if matched_states is not None and not matched_states.empty:
            for _, row in matched_states.iterrows():
                state_id = f"state::{self._slugify(row.get('state_code') or row.get('name'))}"
                nodes.append({"id": state_id, "label": str(row.get("name")), "type": "Location"})
                edges.append({"source": state_id, "target": "brazil", "type": "PART_OF"})

        if matched_municipalities is not None and not matched_municipalities.empty:
            for _, row in matched_municipalities.iterrows():
                muni_id = f"municipality::{self._slugify(row.get('muni_code') or row.get('name'))}"
                nodes.append({"id": muni_id, "label": str(row.get("name")), "type": "Location"})
                state_label = str(row.get("state"))
                state_nodes = [
            node
            for node in nodes
            if node.get("label") == state_label and node.get("type") == "Location"
        ]
                if state_nodes:
                    edges.append(
                        {
                            "source": muni_id,
                            "target": state_nodes[0]["id"],
                            "type": "PART_OF",
                        }
                    )
                else:
                    edges.append({"source": muni_id, "target": "brazil", "type": "PART_OF"})

        return KnowledgeGraphPayload(nodes=nodes, edges=edges)

    @staticmethod
    def _slugify(value: Any) -> str:
        text = BrazilianAdminServerV2._normalize(value)
        return text.replace(" ", "-").replace("/", "-") or "unknown"


def create_server() -> FastMCP:
    server = BrazilianAdminServerV2()
    return server.mcp


if __name__ == "__main__":  # pragma: no cover
    create_server().run()
