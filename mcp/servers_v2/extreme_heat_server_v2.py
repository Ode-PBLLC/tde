"""Extreme heat (legacy heat-stress) MCP server compliant with the v2 contract."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # Optional dotenv support keeps parity with legacy servers
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

try:  # Optional Anthropic client for intent classification
    import anthropic  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    anthropic = None  # type: ignore

try:  # Optional geospatial stack
    import geopandas as gpd  # type: ignore
    from shapely.geometry import box  # type: ignore
    from shapely.strtree import STRtree  # type: ignore
    GEOSPATIAL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    gpd = None  # type: ignore
    box = None  # type: ignore
    STRtree = None  # type: ignore
    GEOSPATIAL_AVAILABLE = False

try:  # Pandas is required when GeoPandas succeeds
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from fastmcp import FastMCP

if load_dotenv:  # pragma: no cover - best-effort env loading
    try:
        load_dotenv()
    except Exception as exc:
        print(f"[extreme_heat] Warning: load_dotenv failed: {exc}")

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
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
    from mcp.servers_v2.support_intent import SupportIntent  # type: ignore
else:  # pragma: no cover - package imports when spawned by orchestrator
    from ..contracts_v2 import (
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from ..servers_v2.base import RunQueryMixin
    from .support_intent import SupportIntent


DATASET_ID = "heat"
DEFAULT_LIMIT = 2000
DEFAULT_QUINTILES = [5]
DATASET_SERVER_NAME = "extreme_heat"
DATASET_SOURCE_TYPE = "Dataset"

SOURCE_DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "Heat_Index": {
        "variable": "Heat Index (WBGT proxy)",
        "spatial_coverage": "Brazil",
        "temporal_coverage": "2020-01-01 to 2025-01-01",
        "source": "ERA5-Land Daily Aggregated - ECMWF",
        "notes": "ERA5-Land lacks coverage over large water bodies and some coastal margins.",
    },
    "LST": {
        "variable": "Land Surface Temperature (LST)",
        "spatial_coverage": "Brazil",
        "temporal_coverage": "2020-01-01 to 2025-01-01",
        "source": "MOD11A1.061 - MODIS/Terra LST & Emissivity, Daily, 1 km",
        "notes": "Daytime Land Surface Temperature daily means; southern-summer variants aggregate Nov-Mar seasons.",
    },
}

QUINTILE_COLORS: Dict[int, str] = {
    1: "#FFEDA0",
    2: "#FEB24C",
    3: "#FD8D3C",
    4: "#FC4E2A",
    5: "#E31A1C",
}

STATE_STATS_PATH = Path(__file__).resolve().parents[2] / "static" / "meta" / "extreme_heat_by_state.json"
MUNICIPALITY_STATS_PATH = (
    Path(__file__).resolve().parents[2] / "static" / "meta" / "extreme_heat_by_municipality.json"
)


class ExtremeHeatServerV2(RunQueryMixin):
    """FastMCP server exposing extreme heat polygons and v2 run_query."""

    def __init__(self) -> None:
        self.mcp = FastMCP("extreme-heat-server-v2")
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_root = self.project_root / "data" / "heat_stress"
        self.preprocessed_dir = self.data_root / "preprocessed"
        self.geojson_dir = self.preprocessed_dir / "geojsons"
        self.doc_path = self.data_root / "Brazil Datasets Info.txt"
        self.static_maps_dir = self.project_root / "static" / "maps"
        self.dataset_metadata = self._load_dataset_metadata()
        self._anthropic_client = self._build_anthropic_client()
        self._top_quintile_index: Dict[str, Any] = {
            "loaded": False,
            "gdf": None,
            "tree": None,
            "geom_list": [],
            "geom_positions": {},
        }
        self._state_heat_stats: Optional[List[Dict[str, Any]]] = None
        self._municipality_heat_stats: Optional[List[Dict[str, Any]]] = None

        self._register_capabilities_tool()
        self._register_query_support_tool()
        self._register_list_layers_tool()
        self._register_quintile_counts_tool()
        self._register_area_chart_tool()
        self._register_dataset_info_tool()
        self._register_geospatial_tool()
        self._register_map_tool()
        self._register_heat_by_state_tool()
        self._register_heat_by_municipality_tool()
        self._register_run_query_tool()
        self._load_precomputed_admin_stats()

    # ------------------------------------------------------------------ helpers
    def _build_anthropic_client(self) -> Optional["anthropic.Anthropic"]:
        """Initialise the Anthropic client when credentials are present."""

        if not anthropic:
            return None
        if not os.getenv("ANTHROPIC_API_KEY"):
            return None
        try:
            return anthropic.Anthropic()
        except Exception as exc:  # pragma: no cover - credential or import failure
            print(f"[extreme_heat] Warning: Anthropic client unavailable: {exc}")
            return None

    def _load_dataset_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load dataset metadata entries from the shared datasets registry."""

        path = self.project_root / "static" / "meta" / "datasets.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        metadata: Dict[str, Dict[str, Any]] = {}
        for item in payload.get("items", []):
            dataset_id = item.get("id")
            if not dataset_id:
                continue
            metadata[str(dataset_id)] = {
                "title": str(item.get("title", "")),
                "description": str(item.get("description", "")),
                "citation": str(item.get("citation", "")),
                "last_updated": str(item.get("last_updated", "")),
            }
        return metadata

    def _dataset_entry(self) -> Dict[str, Any]:
        """Return the metadata block for the extreme heat dataset."""

        return self.dataset_metadata.get(DATASET_ID, {})

    def _dataset_info_for_source(self, source: Optional[str]) -> Dict[str, Any]:
        """Return metadata for a specific source stem when available."""

        if not source:
            return {}
        key = "Heat_Index" if "heat_index" in source.lower() else None
        if not key and "lst" in source.lower():
            key = "LST"
        return SOURCE_DATASET_INFO.get(key or "", {})

    def _normalize_source_name(self, stem: str) -> str:
        """Derive a source identifier from a filename stem."""

        for suffix in ("_quintiles_simplified", "_quintiles"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def _list_data_files(self) -> List[Path]:
        """Return candidate preprocessed files ordered by preference."""

        files: List[Path] = []
        seen: set[Path] = set()
        for directory in (self.geojson_dir, self.preprocessed_dir):
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

    def _explode_geometries(self, gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
        """Explode MultiPolygon geometries into Polygon rows."""

        try:
            return gdf.explode(index_parts=False).reset_index(drop=True)
        except Exception:
            rows = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                parts = getattr(geom, "geoms", [geom])
                for part in parts:
                    new_row = row.copy()
                    new_row.geometry = part
                    rows.append(new_row)
            return gpd.GeoDataFrame(rows, crs=gdf.crs)

    def _ensure_top_quintile_index(self) -> bool:
        """Load top-quintile polygons and build an STRtree for filtering."""

        if not GEOSPATIAL_AVAILABLE or gpd is None or pd is None:
            return False
        if self._top_quintile_index.get("loaded") and self._top_quintile_index.get("gdf") is not None:
            return True

        files = self._list_data_files()
        if not files:
            print("[extreme_heat] No extreme heat files found")
            return False

        parts: List["gpd.GeoDataFrame"] = []
        for path in files:
            try:
                if str(path).lower().endswith(".gpkg"):
                    gdf = gpd.read_file(path, layer="quintiles")
                else:
                    gdf = gpd.read_file(path)
            except Exception as exc:
                print(f"[extreme_heat] Failed to read {path.name}: {exc}")
                continue

            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            if "quintile" not in gdf.columns:
                print(f"[extreme_heat] Missing 'quintile' column in {path.name}; skipping")
                continue

            top = gdf[gdf["quintile"] == 5].copy()
            if top.empty:
                continue
            source = self._normalize_source_name(path.stem)
            top["source"] = source
            top_parts = self._explode_geometries(top)
            top_parts = top_parts[~top_parts.geometry.is_empty]
            top_parts = top_parts.reset_index(drop=True)
            top_parts["id"] = [f"heat_{source}_q5_{idx}" for idx in range(len(top_parts))]
            parts.append(top_parts[["id", "source", "quintile", "geometry"]])

        if not parts:
            print("[extreme_heat] No top-quintile polygons available after filtering")
            return False

        try:
            combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
        except Exception as exc:
            print(f"[extreme_heat] Failed to concatenate GeoDataFrames: {exc}")
            return False

        try:
            geometries = list(combined.geometry.values)
            tree = STRtree(geometries) if STRtree is not None else None
            geom_positions = {id(geom): idx for idx, geom in enumerate(geometries)} if tree else {}
        except Exception as exc:
            print(f"[extreme_heat] Failed to build spatial index: {exc}")
            tree = None
            geom_positions = {}

        self._top_quintile_index.update(
            {
                "loaded": True,
                "gdf": combined,
                "tree": tree,
                "geom_list": geometries,
                "geom_positions": geom_positions,
            }
        )
        print(
            "[extreme_heat] Top quintile index loaded: "
            f"{len(combined)} polygons across {len(parts)} sources"
        )
        return True

    def _filter_by_bbox(
        self,
        gdf: "gpd.GeoDataFrame",
        bbox_dict: Optional[Dict[str, float]],
    ) -> "gpd.GeoDataFrame":
        """Filter a GeoDataFrame by a bounding box when provided."""

        if not bbox_dict or box is None:
            return gdf
        try:
            candidate_box = box(
                bbox_dict["west"],
                bbox_dict["south"],
                bbox_dict["east"],
                bbox_dict["north"],
            )
        except Exception:
            return gdf

        index_gdf = self._top_quintile_index.get("gdf")
        tree = self._top_quintile_index.get("tree")
        if tree is not None and index_gdf is not None and gdf is index_gdf:
            positions: Dict[int, int] = self._top_quintile_index.get("geom_positions", {})
            try:
                hits = tree.query(candidate_box)
                idxs = [positions.get(id(geom)) for geom in hits if positions.get(id(geom)) is not None]
                if idxs:
                    subset = gdf.iloc[idxs]
                    return subset[subset.geometry.intersects(candidate_box)]
            except Exception:
                pass
        return gdf[gdf.geometry.intersects(candidate_box)]

    def _total_area_km2(self, gdf: "gpd.GeoDataFrame") -> Optional[float]:
        """Compute total area in square kilometres using an equal-area projection."""

        if gdf.empty:
            return None
        try:
            working = gdf.set_crs(epsg=4326) if gdf.crs is None else gdf
        except Exception:
            return None
        try:
            projected = working.to_crs("EPSG:5880")
        except Exception:
            try:
                projected = working.to_crs("EPSG:3857")
            except Exception:
                return None
        projected = projected[~projected.geometry.is_empty]
        return float(projected.geometry.area.sum() / 1_000_000.0)

    def _extract_bullet_description(self) -> Optional[str]:
        """Read the legacy dataset notes file to enrich the description."""

        if not self.doc_path.exists():
            return None
        try:
            text = self.doc_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None
        bullets: List[str] = []
        for key in ("Variable:", "Temporal Coverage:", "Source:", "Notes:"):
            matches = [line.strip() for line in text.splitlines() if key in line]
            bullets.extend(matches[:2])
        if bullets:
            return " ".join(bullets[:4])
        return None

    def _capability_summary(self) -> str:
        """Summary string describing dataset coverage for LLM routing."""

        dataset_entry = self._dataset_entry()
        parts = [
            "Dataset: Extreme heat quintile polygons for Brazil.",
            f"Title: {dataset_entry.get('title')}" if dataset_entry.get("title") else "",
            f"Updated: {dataset_entry.get('last_updated')}" if dataset_entry.get("last_updated") else "",
            "Top quintile (5) polygons optimised for geospatial registration.",
            "Supports polygon exports, bounding-box filters, and quick map generation.",
        ]
        appendix = self._extract_bullet_description()
        if appendix:
            parts.append(appendix)
        return " ".join(segment for segment in parts if segment)

    def _classify_intent(self, query: str) -> SupportIntent:
        """Use an LLM or keyword heuristic to judge dataset relevance."""

        if not self._anthropic_client:
            lowered = query.lower()
            keywords = [
                "extreme heat",
                "heat stress",
                "wet bulb",
                "heatwave",
                "heat risk",
                "wbgt",
            ]
            supported = any(token in lowered for token in keywords)
            score = 0.8 if supported else 0.2
            reasons = ["Keyword heuristic"]
            return SupportIntent(supported=supported, score=score, reasons=reasons)

        prompt = (
            "Decide whether the following question should be answered using the"
            " extreme heat geospatial dataset. Respond with JSON like"
            " {\"supported\": true|false, \"reason\": \"short explanation\"}.\n"
            f"Dataset summary: {self._capability_summary()}\n"
            f"Question: {query}\n"
            "Answer true when the query involves heat stress, wet-bulb temperature,"
            " heat risk, or requests for heat polygons in Brazil."
        )

        try:
            response = self._anthropic_client.messages.create(  # type: ignore[no-untyped-call]
                model="claude-3-5-haiku-20241022",
                max_tokens=128,
                temperature=0,
                system="Respond with valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        except Exception as exc:  # pragma: no cover - API/network issues
            return SupportIntent(
                supported=True,
                score=0.3,
                reasons=[f"LLM intent unavailable: {exc}"],
            )

        def _parse(blob: str) -> Optional[Dict[str, Any]]:
            try:
                parsed = json.loads(blob)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None

        data = _parse(text)
        if not data:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                data = _parse(text[start : end + 1])

        if not data:
            return SupportIntent(
                supported=True,
                score=0.3,
                reasons=["LLM returned non-JSON response"],
            )

        supported = bool(data.get("supported", False))
        reason = str(data.get("reason")) if data.get("reason") else "LLM classification"
        return SupportIntent(supported=supported, score=0.9 if supported else 0.1, reasons=[reason])

    def _quintile_counts(self) -> Optional[List[Dict[str, Any]]]:
        """Return counts per quintile using the first available dataset."""

        if not GEOSPATIAL_AVAILABLE or gpd is None:
            return None
        files = self._list_data_files()
        if not files:
            return None
        try:
            gdf = gpd.read_file(files[0])
        except Exception:
            return None
        if "quintile" not in gdf.columns:
            return None
        counts = gdf["quintile"].value_counts().sort_index()
        return [{"quintile": int(key), "count": int(value)} for key, value in counts.items()]

    def _area_by_quintile(self) -> Optional[List[Dict[str, Any]]]:
        """Return area (km^2) per quintile using the first dataset."""

        if not GEOSPATIAL_AVAILABLE or gpd is None:
            return None
        files = self._list_data_files()
        if not files:
            return None
        try:
            gdf = gpd.read_file(files[0])
        except Exception:
            return None
        if "quintile" not in gdf.columns:
            return None
        try:
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            projected = gdf.to_crs("EPSG:5880")
        except Exception:
            try:
                projected = gdf.to_crs("EPSG:3857")
            except Exception:
                return None
        projected = projected[~projected.geometry.is_empty]
        grouped = projected.groupby("quintile", dropna=False)["geometry"].apply(
            lambda series: float(series.area.sum() / 1_000_000.0)
        )
        return [{"quintile": int(key), "area_km2": float(value)} for key, value in grouped.sort_index().items()]

    def _sanitize_limit(self, value: Any) -> int:
        """Clamp limit values to a reasonable range."""

        try:
            limit = int(value)
        except Exception:
            limit = DEFAULT_LIMIT
        return max(0, min(limit, DEFAULT_LIMIT))

    def _prepare_filtered_index(
        self,
        source: Optional[str],
        quintiles: Optional[Iterable[int]],
        limit: int,
        bbox: Optional[Dict[str, float]],
    ) -> Tuple[List[int], Optional["gpd.GeoDataFrame"], List[int]]:
        """Return filtered record indexes, GeoDataFrame, and requested quintiles."""

        if not self._ensure_top_quintile_index():
            return [], None, list(DEFAULT_QUINTILES)

        gdf = self._top_quintile_index.get("gdf")
        if gdf is None:
            return [], None, list(DEFAULT_QUINTILES)

        filtered = gdf
        if source:
            filtered = filtered[filtered["source"].str.lower() == source.lower()]
        requested_quintiles = list(quintiles) if quintiles else list(DEFAULT_QUINTILES)
        if requested_quintiles != [5]:
            requested_quintiles = list(DEFAULT_QUINTILES)
        if bbox:
            filtered = self._filter_by_bbox(filtered, bbox)
        if filtered.empty:
            return [], filtered, requested_quintiles
        limited = filtered.head(limit)
        return list(limited.index.values), limited, requested_quintiles

    def _preferred_map_source(self) -> Optional[str]:
        """Return the default source to use when rendering maps."""

        if not self._ensure_top_quintile_index():
            return None
        gdf = self._top_quintile_index.get("gdf")
        if gdf is None or gdf.empty:
            return None
        sources = [str(src) for src in gdf["source"].dropna().unique().tolist()]
        for src in sources:
            if "heat_index" in src.lower():
                return src
        return sources[0] if sources else None

    def _build_map_geojson(
        self,
        gdf: "gpd.GeoDataFrame",
        identifier: str,
        source: Optional[str],
        quintiles: Iterable[int],
    ) -> Tuple[str, Dict[str, Any]]:
        """Persist a GeoJSON file and return filename plus summary metadata."""

        features: List[Dict[str, Any]] = []
        legend_label = "Extreme Heat Zones"
        legend_key = legend_label.lower()
        for _, row in gdf.iterrows():
            quintile = int(row.get("quintile", 5))
            props = {
                "layer": "heat_zone",
                "quintile": quintile,
                "source": row.get("source"),
                "fill": QUINTILE_COLORS.get(quintile, "#E31A1C"),
                "fill-opacity": 0.35,
                "stroke": "#737373",
                "stroke-width": 0.5,
                "title": f"Heat zone Q{quintile} - {row.get('source')}",
                "country": legend_key,
            }
            features.append(
                {
                    "type": "Feature",
                    "geometry": row.geometry.__geo_interface__,
                    "properties": props,
                }
            )

        geojson = {"type": "FeatureCollection", "features": features}
        os.makedirs(self.static_maps_dir, exist_ok=True)
        data_hash = hashlib.md5(
            f"extreme_heat_{identifier}_{len(features)}_{datetime.utcnow().isoformat()}".encode("utf-8")
        ).hexdigest()[:8]
        filename = f"extreme_heat_{identifier}_{data_hash}.geojson"
        path = self.static_maps_dir / filename
        path.write_text(json.dumps(geojson), encoding="utf-8")

        dataset_info = self._dataset_info_for_source(source or "")
        try:
            minx, miny, maxx, maxy = gdf.total_bounds
            bounds = {
                "west": float(minx),
                "south": float(miny),
                "east": float(maxx),
                "north": float(maxy),
            }
            center = {
                "lon": float((minx + maxx) / 2),
                "lat": float((miny + maxy) / 2),
            }
        except Exception:
            bounds = None
            center = None
        summary = {
            "description": f"Extreme heat map (quintiles {quintiles})",
            "total_features": len(features),
            "layers": [{"type": "heat_zone", "count": len(features)}],
            "title": "Extreme Heat Map",
            "spatial_coverage": dataset_info.get("spatial_coverage"),
            "temporal_coverage": dataset_info.get("temporal_coverage"),
            "dataset_variable": dataset_info.get("variable"),
        }
        if bounds:
            summary["bounds"] = bounds
        if center:
            summary["center"] = center
        return filename, summary

    def _load_precomputed_admin_stats(self) -> None:
        self._state_heat_stats = self._read_stats_file(STATE_STATS_PATH)
        self._municipality_heat_stats = self._read_stats_file(MUNICIPALITY_STATS_PATH)

    def _read_stats_file(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            print(f"[extreme_heat] Precomputed stats missing: {path}")
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[extreme_heat] Failed to load stats file {path.name}: {exc}")
            return []
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        normalised: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                normalised.append(dict(item))
        normalised.sort(key=lambda row: float(row.get("area_km2", 0.0)), reverse=True)
        return normalised

    @staticmethod
    def _chart_dataset(label: str, values: List[float], color: str) -> Dict[str, Any]:
        return {
            "label": label,
            "data": values,
            "backgroundColor": color,
        }

    def _build_state_chart_module(self, stats: List[Dict[str, Any]], limit: int = 10) -> Optional[Dict[str, Any]]:
        if not stats:
            return None
        limited = stats[: max(1, limit)]
        labels = [str(entry.get("state_name")) for entry in limited]
        values = [round(float(entry.get("area_km2", 0.0)), 2) for entry in limited]
        dataset = self._chart_dataset("Extreme Heat Area (km^2)", values, "#E31A1C")
        return {
            "labels": labels,
            "datasets": [dataset],
        }

    def _build_municipality_chart_module(self, stats: List[Dict[str, Any]], limit: int = 10) -> Optional[Dict[str, Any]]:
        if not stats:
            return None
        limited = stats[: max(1, limit)]
        labels = [f"{entry.get('municipality_name')} ({entry.get('state')})" for entry in limited]
        values = [round(float(entry.get("area_km2", 0.0)), 2) for entry in limited]
        dataset = self._chart_dataset("Extreme Heat Area (km^2)", values, "#1E88E5")
        return {
            "labels": labels,
            "datasets": [dataset],
        }

    # ------------------------------------------------------------------ tool registration
    def _register_capabilities_tool(self) -> None:
        payload_cache: Dict[str, Any] = {}

        def _payload() -> Dict[str, Any]:
            if payload_cache:
                return payload_cache
            dataset_entry = self._dataset_entry()
            description_parts = [
                "Preprocessed extreme heat quintile polygons for Brazil (legacy heat-stress dataset).",
                "Top-quintile geometries optimised for bounding-box queries and mapping.",
            ]
            appendix = self._extract_bullet_description()
            if appendix:
                description_parts.append(appendix)
            payload_cache.update(
                {
                    "name": "Extreme Heat Server",
                    "description": " ".join(description_parts),
                    "version": "2.0",
                    "dataset": dataset_entry.get("title") or "Extreme Heat Indices",
                    "last_updated": dataset_entry.get("last_updated"),
                    "tools": [
                        "DescribeServer",
                        "describe_capabilities",
                        "query_support",
                        "ListHeatLayers",
                        "GetHeatQuintileCounts",
                        "GetHeatAreaByQuintileChart",
                        "GetHeatDatasetInfo",
                        "GetHeatQuintilesForGeospatial",
                        "GetHeatQuintilesMap",
                        "GetHeatByStateChart",
                        "GetHeatByMunicipalityChart",
                        "run_query",
                    ],
                }
            )
            return payload_cache

        @self.mcp.tool()
        def DescribeServer() -> Dict[str, Any]:  # type: ignore[misc]
            """Describe heat-stress layers, tools, and live availability."""

            return _payload()

        @self.mcp.tool()
        def describe_capabilities(format: str = "json") -> str:  # type: ignore[misc]
            """Describe the extreme heat dataset and supported tools."""

            payload = _payload()
            return json.dumps(payload) if format == "json" else str(payload)

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> str:  # type: ignore[misc]
            """Return whether the extreme heat dataset can answer the query."""

            intent = self._classify_intent(query)
            payload = {
                "server": DATASET_SERVER_NAME,
                "query": query,
                "supported": intent.supported,
                "score": intent.score,
                "reasons": intent.reasons,
            }
            return json.dumps(payload)

    def _register_list_layers_tool(self) -> None:
        @self.mcp.tool()
        def ListHeatLayers() -> Dict[str, Any]:  # type: ignore[misc]
            """
            List available preprocessed heat-stress GPKG layers and basic metadata.
            Focus on quintiles layer and presence of top quintile.
            """

            files_info: List[Dict[str, Any]] = []
            for path in self._list_data_files():
                info: Dict[str, Any] = {
                    "filename": path.name,
                    "path": str(path),
                    "source": self._normalize_source_name(path.stem),
                    "has_quintiles": True,
                    "has_top_quintile": None,
                }
                if GEOSPATIAL_AVAILABLE and gpd is not None:
                    try:
                        if str(path).lower().endswith(".gpkg"):
                            gdf = gpd.read_file(path, layer="quintiles")
                        else:
                            gdf = gpd.read_file(path)
                        info["crs"] = gdf.crs.to_string() if gdf.crs else "None"
                        if "quintile" in gdf.columns:
                            quintiles = [int(x) for x in gdf["quintile"].dropna().unique().tolist()]
                            info["available_quintiles"] = sorted(quintiles)
                            info["has_top_quintile"] = 5 in quintiles
                        info["feature_count"] = int(len(gdf))
                        metadata = self._dataset_info_for_source(info["source"])
                        if metadata:
                            info["dataset_info"] = metadata
                    except Exception as exc:
                        info["error"] = str(exc)
                files_info.append(info)
            return {
                "heat_layers": files_info,
                "directories": [str(directory) for directory in (self.geojson_dir, self.preprocessed_dir) if directory.exists()],
                "note": "Server optimised for top quintile (5) polygons",
            }

    def _register_quintile_counts_tool(self) -> None:
        @self.mcp.tool()
        def GetHeatQuintileCounts() -> Dict[str, Any]:  # type: ignore[misc]
            """Return a chart spec for counts by heat-stress quintile (1..5)."""

            if not GEOSPATIAL_AVAILABLE or gpd is None:
                return {"error": "GeoPandas not installed"}
            counts = self._quintile_counts()
            if not counts:
                return {"error": "No heat-stress quintiles data available"}
            files = self._list_data_files()
            source_file = str(files[0]) if files else None
            return {
                "visualization_type": "comparison",
                "data": counts,
                "chart_config": {
                    "x_axis": "quintile",
                    "y_axis": "count",
                    "title": "Heat-Stress Features by Quintile",
                    "chart_type": "bar",
                },
                "source_file": source_file,
            }

    def _register_area_chart_tool(self) -> None:
        @self.mcp.tool()
        def GetHeatAreaByQuintileChart() -> Dict[str, Any]:  # type: ignore[misc]
            """Return a pie (or bar) chart spec of total area (km^2) by heat-stress quintile.

            Computes polygon areas in an equal-area projection (EPSG:5880 - Brazil Polyconic)
            and aggregates by the 'quintile' column.
            """

            if not GEOSPATIAL_AVAILABLE or gpd is None:
                return {"error": "GeoPandas not installed"}
            data = self._area_by_quintile()
            if not data:
                return {"error": "No heat-stress quintiles data available"}
            files = self._list_data_files()
            source_file = str(files[0]) if files else None
            return {
                "visualization_type": "comparison",
                "data": data,
                "chart_config": {
                    "x_axis": "quintile",
                    "y_axis": "area_km2",
                    "title": "Heat-Stress Area by Quintile (km^2)",
                    "chart_type": "pie",
                },
                "source_file": source_file,
            }

    def _register_dataset_info_tool(self) -> None:
        @self.mcp.tool()
        def GetHeatDatasetInfo(source: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[misc]
            """Return dataset metadata/citation for heat-stress layers.

            Args:
                source: Optional source hint ('Heat_Index' or 'LST' in filename).
            """

            if source:
                info = self._dataset_info_for_source(source)
                if info:
                    return {"dataset": info, "matched_source": source}
            return {"datasets": SOURCE_DATASET_INFO}

    def _register_geospatial_tool(self) -> None:
        @self.mcp.tool()
        def GetHeatQuintilesForGeospatial(
            source: Optional[str] = None,
            quintiles: Optional[List[int]] = None,
            limit: int = 5000,
            bbox: Optional[Dict[str, float]] = None,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """
            Return condensed heat zone entities for geospatial registration.
            Defaults to top quintile (5) and uses STRtree for bbox filtering.

            Args:
                source: Optional basename to filter a single GPKG source (without _quintiles)
                quintiles: List of quintiles (1..5); defaults to [5]
                limit: Max number of polygon entities to return
                bbox: Optional geographic bounding box dict {north, south, east, west}
            """

            if not GEOSPATIAL_AVAILABLE or gpd is None or pd is None:
                return {"error": "GeoPandas not installed"}

            cleaned_quintiles = list(quintiles) if quintiles else list(DEFAULT_QUINTILES)
            if cleaned_quintiles != [5]:
                return {
                    "warning": "Only top quintile (5) is available in optimised index",
                    "entities": [],
                    "count": 0,
                    "entity_type": "heat_zone",
                }

            _, filtered, _ = self._prepare_filtered_index(
                source=source,
                quintiles=cleaned_quintiles,
                limit=min(limit, DEFAULT_LIMIT),
                bbox=bbox,
            )
            if filtered is None or filtered.empty:
                return {
                    "entity_type": "heat_zone",
                    "entities": [],
                    "count": 0,
                    "note": "No matching heat zones",
                }

            entities: List[Dict[str, Any]] = []
            for _, row in filtered.iterrows():
                try:
                    entities.append(
                        {
                            "id": row["id"],
                            "geometry": row.geometry.__geo_interface__,
                            "quintile": int(row.get("quintile", 5)),
                            "source": row.get("source"),
                        }
                    )
                except Exception:
                    continue

            meta_info = self._dataset_info_for_source(source or (filtered.iloc[0]["source"] if not filtered.empty else ""))
            return {
                "entity_type": "heat_zone",
                "quintiles": cleaned_quintiles,
                "entities": entities,
                "count": len(entities),
                "source_filter": source,
                "bbox": bbox,
                "optimized_top_quintile_only": True,
                "citation": meta_info,
                "spatial_coverage": meta_info.get("spatial_coverage"),
                "temporal_coverage": meta_info.get("temporal_coverage"),
                "dataset_variable": meta_info.get("variable"),
            }

    def _register_map_tool(self) -> None:
        @self.mcp.tool()
        def GetHeatQuintilesMap(
            source: Optional[str] = None,
            quintiles: Optional[List[int]] = None,
            limit: int = 5000,
            bbox: Optional[Dict[str, float]] = None,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """
            Generate a GeoJSON map of heat-stress polygons (default: top quintile only).
            Saves to static/maps and returns URL and summary.
            """

            if not GEOSPATIAL_AVAILABLE or gpd is None or pd is None:
                return {
                    "summary": None,
                    "facts": [],
                    "artifacts": [],
                    "messages": [
                        {"level": "warning", "text": "GeoPandas not installed"}
                    ],
                    "details": {"error": "GeoPandas not installed"},
                }

            cleaned_quintiles = list(quintiles) if quintiles else list(DEFAULT_QUINTILES)
            if cleaned_quintiles != [5]:
                return {
                    "summary": None,
                    "facts": [],
                    "artifacts": [],
                    "messages": [
                        {
                            "level": "warning",
                            "text": "Only top quintile (5) supported for map generation currently",
                        }
                    ],
                    "details": {
                        "error": "Only top quintile (5) supported for map generation currently",
                        "requested_quintiles": cleaned_quintiles,
                    },
                }

            source_to_use = source or self._preferred_map_source()

            _, filtered, _ = self._prepare_filtered_index(
                source=source_to_use,
                quintiles=cleaned_quintiles,
                limit=min(limit, DEFAULT_LIMIT),
                bbox=bbox,
            )
            if filtered is None or filtered.empty:
                return {
                    "summary": None,
                    "facts": [],
                    "artifacts": [],
                    "messages": [
                        {"level": "info", "text": "No matching heat zones"}
                    ],
                    "details": {
                        "error": "No matching heat zones",
                        "source": source_to_use,
                        "quintiles": cleaned_quintiles,
                        "limit": limit,
                        "bbox": bbox,
                    },
                }

            identifier = source_to_use or "all"
            filename, summary = self._build_map_geojson(
                filtered,
                identifier=identifier,
                source=source_to_use,
                quintiles=cleaned_quintiles,
            )
            dataset_entry = self._dataset_entry()
            dataset_info = self._dataset_info_for_source(
                source_to_use or (filtered.iloc[0]["source"] if not filtered.empty else "")
            )
            feature_count = int(summary.get("total_features", len(filtered))) if isinstance(summary, dict) else len(filtered)
            source_label = source_to_use or "all sources"
            summary_text = (
                f"Generated an extreme-heat quintile map with {feature_count:,} zones from {source_label}."
            )

            legend_label = "Extreme Heat Zones"
            artifact_metadata = {
                "dataset_id": DATASET_ID,
                "quintiles": cleaned_quintiles,
                "source": source_to_use,
                "limit": min(limit, DEFAULT_LIMIT),
                "bbox": bbox,
                "optimized_top_quintile_only": True,
                "geometry_type": "polygon",
                "merge_group": "extreme_heat_quintiles",
            }
            if isinstance(summary, dict):
                for key in ("bounds", "center", "legend", "layers"):
                    if summary.get(key) is not None:
                        artifact_metadata[key] = summary[key]

            legend_items = []
            layers_payload = artifact_metadata.get("layers")
            if isinstance(layers_payload, dict):
                for name, layer_data in layers_payload.items():
                    count = layer_data.get("count")
                    legend_items.append(
                        {
                            "label": name.replace("_", " ").title(),
                            "color": layer_data.get("color") or QUINTILE_COLORS.get(5, "#E31A1C"),
                            "description": f"{count} features" if count is not None else None,
                        }
                    )
            elif isinstance(layers_payload, list):
                for entry in layers_payload:
                    if not isinstance(entry, dict):
                        continue
                    count = entry.get("count")
                    legend_items.append(
                        {
                            "label": legend_label,
                            "color": QUINTILE_COLORS.get(5, "#E31A1C"),
                            "description": f"{count} zones" if count is not None else None,
                        }
                    )

            if not legend_items:
                legend_items.append(
                    {
                        "label": legend_label,
                        "color": QUINTILE_COLORS.get(5, "#E31A1C"),
                        "description": f"{feature_count:,} zones",
                    }
                )

            artifact_metadata["legend"] = {
                "title": "Extreme heat quintiles",
                "items": legend_items,
            }

            artifact = {
                "type": "map",
                "title": "Extreme heat quintile map",
                "metadata": artifact_metadata,
                "geojson_url": f"/static/maps/{filename}",
                "summary": summary,
            }
            facts = [
                f"Top-quintile extreme heat polygons span {feature_count:,} zones in {source_label}, optimized for mapping."
            ]

            citation = {
                "tool": "GetHeatQuintilesMap",
                "title": dataset_info.get("title") or dataset_entry.get("title") or "Extreme Heat Indices",
                "source_type": "Dataset",
                "description": dataset_info.get("citation")
                or dataset_info.get("description")
                or dataset_entry.get("citation")
                or dataset_entry.get("description"),
                "metadata": {
                    "dataset_id": DATASET_ID,
                    "source": source_to_use,
                    "quintiles": cleaned_quintiles,
                },
            }

            return {
                "summary": summary_text,
                "facts": facts,
                "artifacts": [artifact],
                "messages": [],
                "citation": citation,
                "details": {
                    "map_summary": summary,
                    "source": source_to_use,
                    "quintiles": cleaned_quintiles,
                    "feature_count": feature_count,
                    "filename": filename,
                },
            }

    # ------------------------------------------------------------------ run_query implementation
    def _register_heat_by_state_tool(self) -> None:
        @self.mcp.tool()
        def GetHeatByStateChart(top_n: int = 10) -> Dict[str, Any]:  # type: ignore[misc]
            """Return a bar chart of top Brazilian states by extreme heat area (km^2)."""

            stats = self._state_heat_stats or []
            if not stats:
                return {
                    "summary": None,
                    "facts": [],
                    "artifacts": [],
                    "messages": [
                        {"level": "warning", "text": "State overlays unavailable"}
                    ],
                    "details": {"error": "State overlays unavailable"},
                }
            top_n_int = max(1, int(top_n))
            chart_payload = self._build_state_chart_module(stats, limit=top_n_int)
            if not chart_payload:
                return {
                    "summary": None,
                    "facts": [],
                    "artifacts": [],
                    "messages": [
                        {"level": "warning", "text": "State overlays unavailable"}
                    ],
                    "details": {"error": "State overlays unavailable"},
                }

            limited_stats = stats[:top_n_int]
            top_state = limited_stats[0]
            top_name = str(top_state.get("state_name", "Unknown"))
            top_area = round(float(top_state.get("area_km2", 0.0)), 2)
            summary = (
                f"{top_name} shows the largest area of extreme heat exposure, covering {top_area:,} km² in the top quintile."
            )

            facts: List[str] = []
            for idx, entry in enumerate(limited_stats[:3], start=1):
                state_label = str(entry.get("state_name", "Unknown"))
                area = round(float(entry.get("area_km2", 0.0)), 2)
                facts.append(
                    f"#{idx}: {state_label} has {area:,} km² of land in the extreme-heat top quintile."
                )

            artifact = {
                "type": "chart",
                "title": "Extreme heat exposure by state",
                "metadata": {
                    "chartType": "bar",
                    "dataset_id": DATASET_ID,
                    "top_n": min(top_n_int, len(limited_stats)),
                },
                "data": {
                    "labels": chart_payload["labels"],
                    "datasets": chart_payload["datasets"],
                },
            }

            dataset_entry = self._dataset_entry()
            citation = {
                "tool": "GetHeatByStateChart",
                "title": dataset_entry.get("title") or "Extreme Heat Indices",
                "source_type": "Dataset",
                "description": dataset_entry.get("citation") or dataset_entry.get("description"),
                "metadata": {
                    "dataset_id": DATASET_ID,
                    "top_n": min(top_n_int, len(limited_stats)),
                },
            }

            return {
                "summary": summary,
                "facts": facts,
                "artifacts": [artifact],
                "messages": [],
                "citation": citation,
                "details": {
                    "labels": chart_payload["labels"],
                    "datasets": chart_payload["datasets"],
                    "states": [dict(entry) for entry in limited_stats],
                },
            }

    def _register_heat_by_municipality_tool(self) -> None:
        @self.mcp.tool()
        def GetHeatByMunicipalityChart(top_n: int = 10) -> Dict[str, Any]:  # type: ignore[misc]
            """Return a bar chart of municipalities most exposed to extreme heat."""

            stats = self._municipality_heat_stats or []
            if not stats:
                return {
                    "summary": None,
                    "facts": [],
                    "artifacts": [],
                    "messages": [
                        {"level": "warning", "text": "Municipality overlays unavailable"}
                    ],
                    "details": {"error": "Municipality overlays unavailable"},
                }

            top_n_int = max(1, int(top_n))
            chart_payload = self._build_municipality_chart_module(stats, limit=top_n_int)
            if not chart_payload:
                return {
                    "summary": None,
                    "facts": [],
                    "artifacts": [],
                    "messages": [
                        {"level": "warning", "text": "Municipality overlays unavailable"}
                    ],
                    "details": {"error": "Municipality overlays unavailable"},
                }

            limited_stats = stats[:top_n_int]
            top_entry = limited_stats[0]
            top_name = f"{top_entry.get('municipality_name', 'Unknown')} ({top_entry.get('state')})"
            top_area = round(float(top_entry.get("area_km2", 0.0)), 2)
            summary = (
                f"{top_name} has the largest municipal area in the extreme-heat top quintile, covering {top_area:,} km²."
            )

            facts: List[str] = []
            for idx, entry in enumerate(limited_stats[:3], start=1):
                label = f"{entry.get('municipality_name', 'Unknown')} ({entry.get('state')})"
                area = round(float(entry.get("area_km2", 0.0)), 2)
                facts.append(
                    f"#{idx}: {label} has {area:,} km² exposed to top-quintile extreme heat."
                )

            return {
                "summary": summary,
                "facts": facts,
                "artifacts": [
                    {
                        "type": "chart",
                        "title": "Extreme heat exposure by municipality",
                        "metadata": {
                            "chartType": "bar",
                            "dataset_id": DATASET_ID,
                            "top_n": min(top_n_int, len(limited_stats)),
                        },
                        "data": {
                            "labels": chart_payload["labels"],
                            "datasets": chart_payload["datasets"],
                        },
                    }
                ],
                "messages": [],
                "citation": {
                    "tool": "GetHeatByMunicipalityChart",
                    "title": self._dataset_entry().get("title") or "Extreme Heat Indices",
                    "source_type": "Dataset",
                    "description": self._dataset_entry().get("citation")
                    or self._dataset_entry().get("description"),
                    "metadata": {
                        "dataset_id": DATASET_ID,
                        "top_n": min(top_n_int, len(limited_stats)),
                    },
                },
                "details": {
                    "labels": chart_payload["labels"],
                    "datasets": chart_payload["datasets"],
                    "municipalities": [dict(entry) for entry in limited_stats],
                },
            }

    def _ensure_top_quintiles_index_ready(self) -> bool:
        return bool(self._top_quintile_index.get("loaded") and self._top_quintile_index.get("gdf") is not None)

    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        """Run a structured query against the extreme heat dataset."""

        if not isinstance(context, dict):
            context = {}

        source = context.get("source") if isinstance(context.get("source"), str) else None
        quintiles_value = context.get("quintiles") if isinstance(context.get("quintiles"), (list, tuple)) else None
        bbox = context.get("bbox") if isinstance(context.get("bbox"), dict) else None
        limit = self._sanitize_limit(context.get("limit", DEFAULT_LIMIT))

        messages: List[MessagePayload] = []
        artifacts: List[ArtifactPayload] = []
        facts: List[FactPayload] = []
        citations: List[CitationPayload] = []

        auto_selected_source = False
        if not source:
            preferred = self._preferred_map_source()
            if preferred:
                source = preferred
                auto_selected_source = True

        dataset_entry = self._dataset_entry()
        citation = CitationPayload(
            id="extreme_heat_dataset",
            server=DATASET_SERVER_NAME,
            tool="run_query",
            title=dataset_entry.get("title") or "Extreme Heat Indices",
            source_type=DATASET_SOURCE_TYPE,
            description=dataset_entry.get("description"),
            url=None,
            metadata={
                "dataset_id": DATASET_ID,
                "last_updated": dataset_entry.get("last_updated"),
            },
        )
        citations.append(citation)

        if quintiles_value and list(quintiles_value) != [5]:
            messages.append(
                MessagePayload(
                    level="warning",
                    text="Only top quintile (5) polygons are indexed; falling back to quintile 5.",
                )
            )

        if auto_selected_source and source and "heat_index" in source.lower():
            messages.append(
                MessagePayload(
                    level="info",
                    text="Defaulting to the Heat Index dataset for map generation.",
                )
            )

        if not GEOSPATIAL_AVAILABLE or gpd is None or pd is None:
            messages.append(
                MessagePayload(
                    level="error",
                    text="GeoPandas is not installed; extreme heat polygons are unavailable on this host.",
                )
            )
            return RunQueryResponse(
                server=DATASET_SERVER_NAME,
                query=query,
                facts=facts,
                citations=citations,
                artifacts=artifacts,
                messages=messages,
            )

        _, filtered, quintiles_used = self._prepare_filtered_index(
            source=source,
            quintiles=quintiles_value,
            limit=limit,
            bbox=bbox,
        )

        if filtered is None or filtered.empty:
            messages.append(
                MessagePayload(
                    level="warning",
                    text="No extreme heat polygons matched the provided filters.",
                )
            )
            return RunQueryResponse(
                server=DATASET_SERVER_NAME,
                query=query,
                facts=facts,
                citations=citations,
                artifacts=artifacts,
                messages=messages,
            )

        feature_count = len(filtered)
        unique_sources = sorted({str(item) for item in filtered["source"].dropna().unique().tolist()})
        total_area = self._total_area_km2(filtered) or 0.0
        bbox_summary = bbox or {}

        fact_text = (
            f"Extracted {feature_count} top-quintile extreme heat polygons"
            f"{' for ' + ', '.join(unique_sources) if unique_sources else ''}"
            " to characterise persistent heat stress zones in Brazil."
        )
        facts.append(
            FactPayload(
                id="F1",
                text=fact_text,
                citation_id=citation.id,
            )
        )

        if total_area:
            facts.append(
                FactPayload(
                    id="F2",
                    text=f"The selected polygons cover approximately {total_area:,.0f} km^2 of land.",
                    citation_id=citation.id,
                    kind="metric",
                    data={"area_km2": total_area, "feature_count": feature_count},
                )
            )

        if dataset_entry.get("last_updated"):
            facts.append(
                FactPayload(
                    id="F3",
                    text=(
                        "Dataset last updated on "
                        f"{dataset_entry['last_updated']}; current export contains top-quintile polygons only."
                    ),
                    citation_id=citation.id,
                )
            )

        identifier = source or "all"
        filename, summary = self._build_map_geojson(
            filtered, identifier=identifier, source=source, quintiles=quintiles_used
        )
        legend_label = "Extreme Heat Zones"
        map_metadata = {
            "source_filter": source,
            "feature_count": feature_count,
            "geometry_type": "polygon",
        }
        if summary.get("bounds"):
            map_metadata["bounds"] = summary["bounds"]
        if summary.get("center"):
            map_metadata["center"] = summary["center"]
        if summary.get("layers"):
            map_metadata["layers"] = summary["layers"]
        map_metadata["legend"] = {
            "title": "Extreme Heat Zones",
            "items": [
                {
                    "label": legend_label,
                    "color": QUINTILE_COLORS.get(5, "#E31A1C"),
                    "description": f"{feature_count} polygons",
                }
            ],
        }
        artifacts.append(
            ArtifactPayload(
                id="extreme_heat_map",
                type="map",
                title=summary.get("title", "Extreme Heat Map"),
                url=f"/static/maps/{filename}",
                geojson_url=f"/static/maps/{filename}",
                data={"summary": summary, "bbox": bbox_summary, "quintiles": quintiles_used},
                description=summary.get("description"),
                metadata=map_metadata,
            )
        )

        if self._state_heat_stats:
            top_states = self._state_heat_stats[:3]
            if top_states:
                state_parts = [
                    f"{item.get('state_name')} ({float(item.get('area_km2', 0.0)):,.0f} km^2)"
                    for item in top_states
                ]
                facts.append(
                    FactPayload(
                        id="F4",
                        text="Top extreme-heat exposure by area concentrates in " + ", ".join(state_parts) + ".",
                        citation_id=citation.id,
                    )
                )

            chart_payload = self._build_state_chart_module(self._state_heat_stats, limit=10)
            if chart_payload:
                artifacts.append(
                    ArtifactPayload(
                        id="extreme_heat_states_chart",
                        type="chart",
                        title="Top States by Extreme Heat Area",
                        data=chart_payload,
                        metadata={"chartType": "bar", "datasetLabel": "Extreme Heat Area (km^2)"},
                    )
                )

        return RunQueryResponse(
            server=DATASET_SERVER_NAME,
            query=query,
            facts=facts,
            citations=citations,
            artifacts=artifacts,
            messages=messages,
        )


if __name__ == "__main__":
    server = ExtremeHeatServerV2()
    server.mcp.run()
