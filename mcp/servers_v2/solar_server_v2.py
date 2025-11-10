"""Solar Facilities MCP server (v2 contract).

This implementation keeps the rich tool surface from the legacy server while
returning results in the stricter v2 schema so downstream orchestrators can
compose facts, maps, and tables deterministically.
"""

# from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

from fastmcp import FastMCP
from shapely import wkt as shapely_wkt
from shapely.geometry import Point, shape, mapping, box
from shapely.ops import transform
from shapely.prepared import prep

try:  # Optional projection support for accurate buffering
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore

try:  # pragma: no cover - optional heavy dependency
    import geopandas as gpd  # type: ignore
except Exception:  # pragma: no cover - geopandas often optional in CI
    gpd = None  # type: ignore

if __package__ in {None, ""}:
    # When executed as ``python mcp/servers_v2/solar_server_v2.py`` the package
    # context is empty; add project root so we can import shared modules.
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from mcp.contracts_v2 import (  # type: ignore
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        MessagePayload,
        KnowledgeGraphPayload,
        RunQueryResponse,
    )
    from mcp.servers_v2.base import RunQueryMixin  # type: ignore
    from mcp.geospatial_datasets import (  # type: ignore
        DeforestationPolygonProvider,
        SolarFacilityProvider,
    )
    from mcp.geospatial_bridge import SpatialCorrelation  # type: ignore
    from mcp.solar_db import SolarDatabase  # type: ignore
    from mcp.servers_v2.support_intent import SupportIntent  # type: ignore
else:  # pragma: no cover - import path when used as package
    from ..contracts_v2 import (
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        MessagePayload,
        KnowledgeGraphPayload,
        RunQueryResponse,
    )
    from ..servers_v2.base import RunQueryMixin
    from ..geospatial_datasets import DeforestationPolygonProvider, SolarFacilityProvider
    from ..geospatial_bridge import SpatialCorrelation
    from ..solar_db import SolarDatabase
    from ..servers_v2.support_intent import SupportIntent


DATASET_NAME = "TransitionZero Solar Asset Mapper (Q1 2025)"
DATASET_URL = "https://transitionzero.org/"
DATASET_ID = "solar_facilities"
DEFORESTATION_DATASET_ID = "brazil_deforestation"


def _load_dataset_citations() -> Dict[str, str]:
    path = Path(__file__).resolve().parents[2] / "static" / "meta" / "datasets.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    mapping: Dict[str, str] = {}
    for item in payload.get("items", []):
        dataset_id = item.get("id")
        citation = item.get("citation")
        if dataset_id and citation:
            mapping[str(dataset_id)] = str(citation)
    return mapping


DATASET_CITATIONS = _load_dataset_citations()


def _dataset_citation(dataset_id: str) -> Optional[str]:
    return DATASET_CITATIONS.get(dataset_id)


def _load_dataset_metadata() -> Dict[str, Dict[str, str]]:
    path = Path(__file__).resolve().parents[2] / "static" / "meta" / "datasets.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    mapping: Dict[str, Dict[str, str]] = {}
    for item in payload.get("items", []):
        dataset_id = item.get("id")
        if not dataset_id:
            continue
        mapping[str(dataset_id)] = {
            "title": str(item.get("title", "")),
            "source": str(item.get("source", "")),
            "citation": str(item.get("citation", "")),
        }
    return mapping


DATASET_METADATA = _load_dataset_metadata()


def _dataset_metadata(dataset_id: str) -> Optional[Dict[str, str]]:
    return DATASET_METADATA.get(dataset_id)


@dataclass(frozen=True)
class BrazilState:
    """Lightweight cache entry describing a Brazilian state polygon."""

    name: str
    abbreviation: Optional[str]
    geometry: Any


@dataclass(frozen=True)
class SimplePolygon:
    """Minimal structure for passing polygons into GeoJSON generation."""

    polygon_id: str
    geometry_wkt: str
    properties: Dict[str, Any]


# Canonical Brazilian state abbreviations for alias expansion.
BRAZIL_STATE_ABBREVIATIONS: Dict[str, str] = {
    "Acre": "AC",
    "Alagoas": "AL",
    "Amapá": "AP",
    "Amazonas": "AM",
    "Bahia": "BA",
    "Ceará": "CE",
    "Distrito Federal": "DF",
    "Espírito Santo": "ES",
    "Goiás": "GO",
    "Maranhão": "MA",
    "Mato Grosso": "MT",
    "Mato Grosso do Sul": "MS",
    "Minas Gerais": "MG",
    "Pará": "PA",
    "Paraíba": "PB",
    "Paraná": "PR",
    "Pernambuco": "PE",
    "Piauí": "PI",
    "Rio de Janeiro": "RJ",
    "Rio Grande do Norte": "RN",
    "Rio Grande do Sul": "RS",
    "Rondônia": "RO",
    "Roraima": "RR",
    "Santa Catarina": "SC",
    "São Paulo": "SP",
    "Sergipe": "SE",
    "Tocantins": "TO",
}


POLYGON_LAYER_COLORS: Dict[str, str] = {
    "deforestation_polygon": "#F4511E",
    "brazil_state": "#1E88E5",
}

SOLAR_STATE_STATS_PATH = Path(__file__).resolve().parents[2] / "static" / "meta" / "solar_facilities_by_state.json"
SOLAR_MUNICIPALITY_STATS_PATH = (
    Path(__file__).resolve().parents[2] / "static" / "meta" / "solar_facilities_by_municipality.json"
)

CIRCLE_BUFFER_METERS = 1_000.0
FACILITY_BUFFER_MIN_METERS = 2_000.0
FACILITY_BUFFER_MAX_METERS = 10_000.0
FACILITY_BUFFER_SCALE_PER_KM = 600.0  # metres to use per requested kilometre radius
FACILITY_BUFFER_COLOR = "#1E88E5"


def _buffer_point_as_circle(lon: float, lat: float, radius_m: float = CIRCLE_BUFFER_METERS) -> Dict[str, Any]:
    """Return a circular polygon around a WGS84 point for map rendering."""

    if Transformer:
        to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        def _to_merc(x: float, y: float) -> tuple[float, float]:
            return to_merc.transform(x, y)

        def _to_lonlat(x: float, y: float) -> tuple[float, float]:
            return to_lonlat.transform(x, y)

        center_merc = transform(_to_merc, Point(lon, lat))
        circle_merc = center_merc.buffer(radius_m)
        circle_lonlat = transform(_to_lonlat, circle_merc)
    else:  # fallback to simple degree buffer
        radius_deg = radius_m / 111_320.0
        circle_lonlat = Point(lon, lat).buffer(radius_deg)

    return mapping(circle_lonlat)


@dataclass
class GeoJSONSummary:
    url: str
    metadata: Dict[str, Any]


class SolarServerV2(RunQueryMixin):
    """FastMCP server exposing solar database tools and v2 `run_query`."""

    def __init__(self) -> None:
        self.mcp = FastMCP("solar-server-v2")
        self.db = SolarDatabase()
        try:
            self.facility_provider: Optional[SolarFacilityProvider] = SolarFacilityProvider()
        except Exception as exc:
            print(f"[solar] Warning: failed to preload solar facilities: {exc}")
            self.facility_provider = None
        try:
            self.deforestation_provider: Optional[DeforestationPolygonProvider] = DeforestationPolygonProvider()
        except Exception as exc:
            print(f"[solar] Warning: failed to preload deforestation polygons: {exc}")
            self.deforestation_provider = None
        try:
            self.spatial_correlator: Optional[SpatialCorrelation] = SpatialCorrelation()
        except Exception as exc:
            print(f"[solar] Warning: spatial correlation disabled ({exc})")
            self.spatial_correlator = None

        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential issues
                print(f"[solar] Warning: Anthropic client unavailable: {exc}")

        self._register_capabilities_tool()
        self._register_query_support_tool()
        self._register_tool_top_countries()
        self._register_tool_countries()
        self._register_tool_find_country()
        self._register_tool_facilities_by_country()
        self._register_tool_map_data()
        self._register_tool_facilities_for_geospatial()
        self._register_tool_facilities_in_radius()
        self._register_tool_facilities_in_brazil_state()
        self._register_tool_facilities_in_bounds()
        self._register_tool_facilities_multiple_countries()
        self._register_tool_capacity_by_country()
        self._register_tool_capacity_by_state()
        self._register_tool_capacity_by_municipality()
        self._register_tool_construction_timeline()
        self._register_tool_largest_facilities()
        self._register_tool_facility_details()
        self._register_tool_facility_location()
        self._register_tool_capacity_visualization()
        self._register_tool_facilities_near_polygon()
        self._register_tool_facilities_near_deforestation()
        self._register_run_query_tool()
        self._country_stats_index: Dict[str, Dict[str, Any]] = {}
        self._state_facility_stats: List[Dict[str, Any]] = []
        self._municipality_facility_stats: List[Dict[str, Any]] = []
        self._timeline_cache: Dict[str, List[Tuple[str, int]]] = {}
        self._load_precomputed_admin_stats()

        self._brazil_states: Dict[str, BrazilState] = {}
        self._brazil_state_aliases: Dict[str, Dict[str, Any]] = {}
        self._load_brazil_states()

    # ------------------------------------------------------------------ shared formatting helpers
    def _dataset_citation_dict(self, *, description: Optional[str] = None) -> Dict[str, Any]:
        meta = _dataset_metadata(DATASET_ID) or {}
        citation_text = description or _dataset_citation(DATASET_ID) or DATASET_NAME
        return {
            "id": "tz_sam_q1_2025",
            "title": meta.get("title") or DATASET_NAME,
            "source_type": "Dataset",
            "description": citation_text,
            "url": meta.get("source") or DATASET_URL,
            "provider": "TransitionZero",
        }

    @staticmethod
    def _format_capacity(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if abs(numeric) >= 1_000:
            return f"{numeric:,.0f}"
        return f"{numeric:,.1f}"

    @staticmethod
    def _format_integer(value: Any) -> str:
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)

    # ------------------------------------------------------------------ helpers
    def _normalize_text(self, text: str) -> str:
        base = unicodedata.normalize("NFKD", text)
        without_accents = "".join(ch for ch in base if not unicodedata.combining(ch))
        lowered = without_accents.lower()
        cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
        collapsed = re.sub(r"\s+", " ", cleaned).strip()
        return collapsed

    def _brazil_state_geojson_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "data" / "brazilian_admin" / "brazilian_states.geojson"

    def _add_state_alias(
        self,
        alias: str,
        canonical_key: str,
        *,
        is_abbreviation: bool = False,
    ) -> None:
        normalized_alias = self._normalize_text(alias)
        if not normalized_alias:
            return
        self._brazil_state_aliases[normalized_alias] = {
            "canonical": canonical_key,
            "alias": normalized_alias,
            "length": len(normalized_alias),
            "is_abbreviation": is_abbreviation,
        }

    def _load_brazil_states(self) -> None:
        """Preload Brazilian state polygons for sub-national facility filters."""

        self._brazil_states.clear()
        self._brazil_state_aliases.clear()

        if gpd is None:
            print("[solar] Warning: geopandas unavailable; state-level queries disabled")
            return

        path = self._brazil_state_geojson_path()
        if not path.exists():
            print(f"[solar] Warning: Brazilian state GeoJSON missing at {path}")
            return

        try:
            states_gdf = gpd.read_file(path)
        except Exception as exc:
            print(f"[solar] Warning: failed to load Brazilian states GeoJSON ({exc})")
            return

        for record in states_gdf.itertuples():
            name = getattr(record, "NM_UF", None)
            geometry = getattr(record, "geometry", None)
            if not name or geometry is None:
                continue

            canonical = self._normalize_text(str(name))
            abbreviation = BRAZIL_STATE_ABBREVIATIONS.get(str(name))
            self._brazil_states[canonical] = BrazilState(
                name=str(name),
                abbreviation=abbreviation,
                geometry=geometry,
            )

            self._add_state_alias(str(name), canonical)
            self._add_state_alias(f"estado de {name}", canonical)
            self._add_state_alias(f"state of {name}", canonical)

            ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
            if ascii_name and ascii_name != name:
                self._add_state_alias(ascii_name, canonical)
                self._add_state_alias(f"estado de {ascii_name}", canonical)
                self._add_state_alias(f"state of {ascii_name}", canonical)

            if abbreviation:
                self._add_state_alias(abbreviation, canonical, is_abbreviation=True)

    def _load_precomputed_admin_stats(self) -> None:
        self._state_facility_stats = self._read_stats_file(SOLAR_STATE_STATS_PATH)
        self._municipality_facility_stats = self._read_stats_file(SOLAR_MUNICIPALITY_STATS_PATH)

    def _read_stats_file(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            print(f"[solar] Warning: precomputed stats missing at {path}")
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[solar] Warning: failed to load stats file {path.name}: {exc}")
            return []
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        cleaned: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                entry = dict(item)
                entry["facility_count"] = int(entry.get("facility_count", 0))
                entry["facilities_with_capacity"] = int(entry.get("facilities_with_capacity", 0))
                entry["total_capacity_mw"] = float(entry.get("total_capacity_mw", 0.0))
                cleaned.append(entry)
        cleaned.sort(key=lambda row: row.get("total_capacity_mw", 0.0), reverse=True)
        return cleaned

    def _match_brazil_state(self, value: Optional[str]) -> Optional[BrazilState]:
        if not value:
            return None
        normalized = self._normalize_text(value)
        if not normalized:
            return None
        alias = self._brazil_state_aliases.get(normalized)
        if alias:
            return self._brazil_states.get(alias["canonical"])
        return None

    _LETTER_REF_PATTERN = re.compile(
        r"\b(?:state|row|item|entry|option|label)\s+(?:you\s+)?(?:labeled|labelled)?\s*([A-Z])\b",
        re.IGNORECASE,
    )

    def _extract_letter_reference(self, query: str) -> Optional[str]:
        if not query:
            return None
        match = self._LETTER_REF_PATTERN.search(query)
        if match:
            letter = match.group(1).upper()
            if len(letter) == 1 and letter.isalpha():
                return letter
        # Fallback for phrasing like "state B"
        fallback = re.search(r"\bstate\s+([A-Z])\b", query)
        if fallback:
            letter = fallback.group(1).upper()
            if len(letter) == 1:
                return letter
        return None

    def _build_label_to_state_map(
        self, modules: Iterable[Mapping[str, Any]]
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for module in modules:
            if not isinstance(module, Mapping):
                continue
            module_type = str(module.get("type") or "").lower()
            texts: List[str] = []
            if module_type == "text":
                if isinstance(module.get("texts"), list):
                    texts.extend(str(item) for item in module.get("texts") if item)
                content = module.get("content")
                if isinstance(content, str):
                    texts.append(content)
            elif module_type == "focus":
                content = module.get("content")
                if isinstance(content, str):
                    texts.append(content)
            elif module_type == "table":
                rows = module.get("rows")
                columns = module.get("columns") or []
                if isinstance(rows, list) and rows:
                    # Attempt to infer from table structure if first column contains labels
                    for row in rows:
                        if not isinstance(row, Sequence) or not row:
                            continue
                        label_candidate = str(row[0]).strip()
                        if len(label_candidate) == 1 and label_candidate.isalpha():
                            if len(row) > 1:
                                texts.append(f"{label_candidate}. {row[1]}")
            for text in texts:
                if not text:
                    continue
                for match in re.finditer(r"\b([A-Z])\.[\s\u202f]+([^\n]+)", text):
                    letter = match.group(1).upper()
                    remainder = match.group(2).strip()
                    if not remainder:
                        continue
                    # Trim at common delimiters
                    name_part = re.split(r"[:–\-]|\(|,", remainder, maxsplit=1)[0].strip()
                    if not name_part:
                        continue
                    state = self._match_brazil_state(name_part)
                    if state:
                        mapping[letter] = state.name
        return mapping

    def _infer_state_from_context(
        self,
        query: str,
        context: Mapping[str, Any],
    ) -> Optional[BrazilState]:
        if not isinstance(context, Mapping):
            return None
        modules = context.get("previous_response_modules")
        if not isinstance(modules, list):
            return None
        letter = self._extract_letter_reference(query)
        if not letter:
            return None
        label_map = self._build_label_to_state_map(modules)
        state_name = label_map.get(letter)
        if not state_name:
            return None
        return self._match_brazil_state(state_name)

    @staticmethod
    def _chart_dataset(label: str, values: List[float], color: str, *, fill: bool = False) -> Dict[str, Any]:
        dataset: Dict[str, Any] = {
            "label": label,
            "data": values,
            "borderColor": color,
            "backgroundColor": color if fill else color,
        }
        if fill:
            dataset["backgroundColor"] = "rgba(76, 175, 80, 0.35)"
            dataset["fill"] = True
        return dataset

    def _build_state_capacity_chart(self, *, top_n: int = 10) -> Optional[Dict[str, Any]]:
        if not self._state_facility_stats:
            return None
        stats = self._state_facility_stats[: max(1, top_n)]
        labels = [entry.get("state_name") for entry in stats]
        values = [round(float(entry.get("total_capacity_mw", 0.0)), 2) for entry in stats]
        dataset = {
            "label": "Solar Capacity (MW)",
            "data": values,
            "backgroundColor": "#43A047",
        }
        return {
            "labels": labels,
            "datasets": [dataset],
            "options": {
                "scales": {
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Capacity (MW)",
                        }
                    }
                }
            },
        }

    def _build_municipality_capacity_chart(self, *, top_n: int = 10) -> Optional[Dict[str, Any]]:
        if not self._municipality_facility_stats:
            return None
        stats = self._municipality_facility_stats[: max(1, top_n)]
        labels = [
            f"{entry.get('municipality_name')} ({entry.get('state')})" for entry in stats
        ]
        values = [round(float(entry.get("total_capacity_mw", 0.0)), 2) for entry in stats]
        dataset = {
            "label": "Solar Capacity (MW)",
            "data": values,
            "backgroundColor": "#1E88E5",
        }
        return {
            "labels": labels,
            "datasets": [dataset],
            "options": {
                "scales": {
                    "y": {
                        "title": {
                            "display": True,
                            "text": "Capacity (MW)",
                        }
                    }
                }
            },
        }

    def _construction_timeline_series(self, country: str) -> List[Tuple[str, int]]:
        """Generate time series of solar facility construction by year for a country.

        Extracts construction years from `constructed_before` or `constructed_after`
        fields and counts facilities commissioned each year. Results are cached for
        performance.

        Args:
            country: Country name (e.g., "Brazil", "United States")

        Returns:
            List of (year, count) tuples sorted chronologically. Example:
            [("2017", 44), ("2018", 48), ("2019", 89), ...]
        """
        key = country.lower()
        if key in self._timeline_cache:
            return self._timeline_cache[key]

        facilities = self.db.get_all_facilities()
        normalized_country = self._normalize_country(country)
        series: Dict[str, int] = {}
        for facility in facilities:
            facility_country = self._normalize_country(str(facility.get("country") or ""))
            if facility_country != normalized_country:
                continue
            year = self._extract_year(facility.get("constructed_before") or facility.get("constructed_after"))
            if year:
                series[year] = series.get(year, 0) + 1
        timeline = sorted(series.items())
        self._timeline_cache[key] = timeline
        return timeline

    def _construction_timeline_with_capacity(self, country: str) -> Dict[str, Any]:
        """Generate comprehensive timeline with annual and cumulative capacity data.

        Tracks both annual additions and running totals of solar facilities and
        capacity over time. This provides a complete picture of solar deployment
        growth, showing both installation rate and total accumulated capacity.

        Args:
            country: Country name (e.g., "Brazil", "United States")

        Returns:
            Dict with keys:
            - annual: List[Dict] with year, facility_count, capacity_mw
            - cumulative: List[Dict] with year, total_facilities, total_capacity_mw
            - summary: Dict with first_year, last_year, total_facilities, total_capacity

        Example:
            {
                "annual": [
                    {"year": "2017", "facilities": 44, "capacity_mw": 1509.92},
                    {"year": "2018", "facilities": 48, "capacity_mw": 1184.97},
                ],
                "cumulative": [
                    {"year": "2017", "total_facilities": 44, "total_capacity_mw": 1509.92},
                    {"year": "2018", "total_facilities": 92, "total_capacity_mw": 2694.89},
                ],
                "summary": {
                    "first_year": "2017",
                    "last_year": "2025",
                    "total_facilities": 2273,
                    "total_capacity_mw": 26022.51
                }
            }
        """
        facilities = self.db.get_all_facilities()
        normalized_country = self._normalize_country(country)

        # Aggregate by year
        annual_data: Dict[str, Dict[str, float]] = {}
        for facility in facilities:
            facility_country = self._normalize_country(str(facility.get("country") or ""))
            if facility_country != normalized_country:
                continue

            year = self._extract_year(facility.get("constructed_before") or facility.get("constructed_after"))
            if not year:
                continue

            if year not in annual_data:
                annual_data[year] = {"count": 0, "capacity": 0.0}

            annual_data[year]["count"] += 1
            capacity = facility.get("capacity_mw")
            if capacity is not None:
                try:
                    annual_data[year]["capacity"] += float(capacity)
                except (TypeError, ValueError):
                    pass

        # Sort and build cumulative
        sorted_years = sorted(annual_data.keys())
        annual_list = []
        cumulative_list = []
        cumulative_facilities = 0
        cumulative_capacity = 0.0

        for year in sorted_years:
            data = annual_data[year]
            annual_facilities = data["count"]
            annual_capacity = data["capacity"]

            cumulative_facilities += annual_facilities
            cumulative_capacity += annual_capacity

            annual_list.append({
                "year": year,
                "facilities": annual_facilities,
                "capacity_mw": round(annual_capacity, 2)
            })

            cumulative_list.append({
                "year": year,
                "total_facilities": cumulative_facilities,
                "total_capacity_mw": round(cumulative_capacity, 2)
            })

        summary = {}
        if sorted_years:
            summary = {
                "first_year": sorted_years[0],
                "last_year": sorted_years[-1],
                "total_facilities": cumulative_facilities,
                "total_capacity_mw": round(cumulative_capacity, 2)
            }

        return {
            "annual": annual_list,
            "cumulative": cumulative_list,
            "summary": summary
        }

    @staticmethod
    def _extract_year(value: Any) -> Optional[str]:
        """Extract 4-digit year from date string or timestamp.

        Args:
            value: Date string, timestamp, or None

        Returns:
            Four-digit year string (e.g., "2023") or None if unable to parse
        """
        if value is None:
            return None
        text = str(value)
        if len(text) < 4:
            return None
        year = text[:4]
        return year if year.isdigit() else None

    def _build_timeline_chart(self, timeline: List[Tuple[str, int]]) -> Optional[Dict[str, Any]]:
        """Convert timeline series into Chart.js compatible visualization data.

        Creates a line chart configuration showing facility commissioning trends
        over time with labels and datasets ready for frontend rendering.

        Args:
            timeline: List of (year, count) tuples from _construction_timeline_series()

        Returns:
            Chart.js data object with labels and datasets, or None if timeline is empty.
            Format: {"labels": [...], "datasets": [{"label": ..., "data": ...}]}
        """
        if not timeline:
            return None
        labels = [year for year, _ in timeline]
        counts = [count for _, count in timeline]
        dataset = {
            "label": "Commissioned Facilities",
            "data": counts,
            "borderColor": "#4CAF50",
            "backgroundColor": "rgba(76, 175, 80, 0.35)",
            "fill": True,
        }
        return {
            "labels": labels,
            "datasets": [dataset],
        }

    def _detect_brazil_state(self, query: str) -> Optional[BrazilState]:
        if not query:
            return None
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            return None

        best_state: Optional[BrazilState] = None
        best_length = 0
        for alias_norm, info in self._brazil_state_aliases.items():
            if info.get("is_abbreviation"):
                pattern = rf"\b{re.escape(alias_norm)}\b"
                if not re.search(pattern, normalized_query):
                    continue
            else:
                if alias_norm not in normalized_query:
                    continue

            candidate = self._brazil_states.get(info["canonical"])
            if not candidate:
                continue

            alias_length = int(info.get("length", 0))
            if alias_length > best_length:
                best_state = candidate
                best_length = alias_length

        return best_state

    def _facilities_within_geometry(
        self,
        geometry: Any,
        *,
        limit: Optional[int] = None,
        country: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if geometry is None:
            return []

        prepared = prep(geometry)

        facilities: List[Any]
        if self.facility_provider:
            if country:
                facilities = self.facility_provider.facilities_by_country(country)
            else:
                facilities = self.facility_provider.all_facilities()
        else:
            if country:
                facilities = self.db.get_facilities_by_country(country, limit=5000)
            else:
                facilities = self.db.get_all_facilities(limit=10000)

        matches: List[Dict[str, Any]] = []
        for facility in facilities:
            facility_dict = facility.as_dict() if hasattr(facility, "as_dict") else facility
            if not isinstance(facility_dict, dict):
                continue

            lat = facility_dict.get("latitude")
            lon = facility_dict.get("longitude")
            if lat is None or lon is None:
                continue

            try:
                point = Point(float(lon), float(lat))
            except Exception:
                continue

            if prepared.contains(point):
                matches.append(facility_dict)
                if limit is not None and len(matches) >= limit:
                    break

        return matches

    def _facilities_within_state(self, state: BrazilState, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self._facilities_within_geometry(
            state.geometry,
            limit=limit,
            country="Brazil",
        )

    def _build_state_run_query(self, query: str, state: BrazilState) -> RunQueryResponse:
        facilities = self._facilities_within_state(state)
        facility_count = len(facilities)
        total_capacity = round(sum((f.get("capacity_mw") or 0.0) for f in facilities), 1)

        polygon = SimplePolygon(
            polygon_id=f"state_{self._normalize_text(state.name).replace(' ', '_')}",
            geometry_wkt=state.geometry.wkt,
            properties={
                "layer": "brazil_state",
                "name": state.name,
                "abbreviation": state.abbreviation,
            },
        )
        geojson = self._generate_geojson(
            facilities,
            identifier=f"brazil_state_{self._normalize_text(state.name).replace(' ', '_')}",
            polygons=[polygon],
        )

        if isinstance(geojson.metadata, dict):
            geojson.metadata.setdefault("scope", {})
            if isinstance(geojson.metadata["scope"], dict):
                scope_meta = geojson.metadata["scope"]
            else:
                scope_meta = {}
                geojson.metadata["scope"] = scope_meta
            scope_meta.update(
                {
                    "admin_level": "state",
                    "state": state.name,
                    "country": "Brazil",
                    "abbreviation": state.abbreviation,
                }
            )
            geojson.metadata["total_facilities_state"] = facility_count
            geojson.metadata["total_capacity_state_mw"] = total_capacity

        brazil_stats = self._country_statistics("Brazil") or {}
        national_facility_count = int(brazil_stats.get("facility_count") or 0)
        national_capacity = float(brazil_stats.get("total_capacity_mw") or 0.0)

        dataset_meta = _dataset_metadata(DATASET_ID) or {}
        citation = CitationPayload(
            id=f"tz_sam_{self._normalize_text(state.name).replace(' ', '_')}",
            server="solar",
            tool="run_query",
            title=dataset_meta.get("title") or DATASET_NAME,
            source_type="Dataset",
            description=_dataset_citation(DATASET_ID)
            or f"Solar facilities filtered for {state.name}, Brazil",
            url=dataset_meta.get("source") or DATASET_URL,
        )

        facts: List[FactPayload] = [
            FactPayload(
                id="state_facility_count",
                text=f"{state.name} contains {facility_count} tracked solar facilities in the TransitionZero dataset.",
                citation_id=citation.id,
                metadata={"state": state.name},
            )
        ]

        if total_capacity:
            facts.append(
                FactPayload(
                    id="state_capacity_total",
                    text=f"Their combined nameplate capacity is about {total_capacity:.1f} MW.",
                    citation_id=citation.id,
                    metadata={"state": state.name},
                )
            )

        if national_facility_count:
            share = (facility_count / national_facility_count) * 100
            share_capacity = (
                (total_capacity / national_capacity) * 100 if national_capacity else None
            )
            share_text = (
                f"These sites represent roughly {share:.1f}% of Brazil's {national_facility_count} tracked facilities"
            )
            if share_capacity:
                share_text += f" and {share_capacity:.1f}% of national solar capacity"
            share_text += "."
            facts.append(
                FactPayload(
                    id="state_national_share",
                    text=share_text,
                    citation_id=citation.id,
                    metadata={
                        "state": state.name,
                        "national_facility_count": national_facility_count,
                        "national_capacity_mw": round(national_capacity, 1) if national_capacity else None,
                    },
                )
            )

        largest_facility = None
        if facilities:
            largest_facility = max(
                facilities,
                key=lambda record: record.get("capacity_mw") or 0.0,
            )
        if largest_facility and (largest_facility.get("capacity_mw") or 0.0) > 0:
            capacity = float(largest_facility.get("capacity_mw") or 0.0)
            facts.append(
                FactPayload(
                    id="state_largest_facility",
                    text=(
                        f"The largest installation in {state.name} is {largest_facility.get('cluster_id')} "
                        f"at approximately {capacity:.1f} MW."
                    ),
                    citation_id=citation.id,
                    metadata={
                        "cluster_id": largest_facility.get("cluster_id"),
                        "capacity_mw": capacity,
                    },
                )
            )

        plotted_count = geojson.metadata.get("layers", {}).get("solar_facilities", {}).get("plotted_count") if isinstance(geojson.metadata, dict) else None
        map_metadata = geojson.metadata if isinstance(geojson.metadata, dict) else {}
        if plotted_count:
            facts.append(
                FactPayload(
                    id="state_map_fact",
                    kind="map",
                    text=(
                        f"Map data is available, plotting {plotted_count} facilities across {state.name}."
                    ),
                    citation_id=citation.id,
                    metadata={
                        "geojson_url": geojson.url,
                        "plotted_count": plotted_count,
                        "state": state.name,
                    },
                )
            )

        artifacts = [
            ArtifactPayload(
                id=f"map_{self._normalize_text(state.name).replace(' ', '_')}",
                type="map",
                title=f"Solar facilities in {state.name}",
                geojson_url=geojson.url,
                metadata=map_metadata,
            )
        ]

        messages: List[MessagePayload] = []
        if facility_count == 0:
            messages.append(
                MessagePayload(
                    level="warning",
                    text=(
                        f"No solar facilities matched the {state.name} boundary in the dataset."
                    ),
                )
            )

        kg = KnowledgeGraphPayload(
            nodes=[
                {"id": "solar_energy", "label": "Solar energy", "type": "Concept"},
                {"id": "brazil", "label": "Brazil", "type": "Location"},
                {
                    "id": self._normalize_text(state.name).replace(" ", "_"),
                    "label": state.name,
                    "type": "Location",
                },
            ],
            edges=[
                {"source": "solar_energy", "target": "brazil", "type": "LOCATES"},
                {
                    "source": "solar_energy",
                    "target": self._normalize_text(state.name).replace(" ", "_"),
                    "type": "LOCATES",
                },
                {
                    "source": self._normalize_text(state.name).replace(" ", "_"),
                    "target": "brazil",
                    "type": "WITHIN",
                },
            ],
        )

        return RunQueryResponse(
            server="solar",
            query=query,
            facts=facts,
            citations=[citation],
            artifacts=artifacts,
            messages=messages,
            kg=kg,
        )

    def _capabilities_metadata(self) -> Dict[str, Any]:
        """Return the capabilities payload shared by tools and LLM prompts."""

        return {
            "name": "solar",
            "description": "Global solar facility records with geospatial coordinates, capacity, commissioning dates, and temporal deployment analysis.",
            "version": "2.0.0",
            "tags": ["solar", "facilities", "geospatial", "renewable", "timeline", "growth", "deployment", "expansion"],
            "dataset": DATASET_NAME,
            "url": DATASET_URL,
            "tools": [
                "describe_capabilities",
                "query_support",
                "get_top_countries_by_facilities",
                "get_solar_facilities_countries",
                "find_country_for_facility",
                "get_solar_facilities_by_country",
                "get_solar_facilities_map_data",
                "get_solar_facilities_for_geospatial",
                "get_solar_facilities_in_radius",
                "get_solar_facilities_in_brazil_state",
                "get_solar_facilities_in_bounds",
                "get_solar_facilities_multiple_countries",
                "get_capacity_by_country",
                "get_construction_timeline",
                "get_largest_facilities",
                "get_facility_details",
                "get_facility_location",
                "get_capacity_visualization",
                "get_facilities_near_polygon",
                "get_facilities_near_deforestation",
                "run_query",
            ],
        }

    def _capability_summary(self) -> str:
        metadata = self._capabilities_metadata()
        return (
            f"Dataset: {metadata['dataset']} ({metadata['description']}). "
            "Provides facility locations, capacities, commissioning dates (99% coverage), and tools for: "
            "geospatial search, temporal deployment analysis (annual & cumulative timelines), growth rate calculations, "
            "correlation with deforestation, and aggregation by country/state/municipality."
        )

    def _country_statistics(self, country: str) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_country(country)
        cache_key = normalized.lower()
        if not self._country_stats_index:
            try:
                stats = self.db.get_country_statistics()
            except Exception as exc:  # pragma: no cover - database access issues
                print(f"[solar] Warning: failed to load country statistics ({exc})")
                self._country_stats_index = {}
            else:
                self._country_stats_index = {
                    str(row.get("country") or "").lower(): row for row in stats if row.get("country")
                }

        return self._country_stats_index.get(cache_key)

    def _classify_support(self, query: str) -> SupportIntent:
        """Use an LLM to judge whether the solar dataset should handle a query."""

        if not self._anthropic_client:
            return SupportIntent(
                supported=True,
                score=0.3,
                reasons=["LLM unavailable; defaulting to dataset summary"],
            )

        prompt = (
            "You decide whether to route a question to the TransitionZero Solar Asset Mapper dataset."
            " The dataset contains EXISTING, OPERATIONAL solar facility points with capacity (MW), commissioning dates (99% coverage),"
            " and tools for country summaries, geospatial lookups, temporal deployment analysis, growth trends, expansion rates,"
            " and correlations with deforestation activity."
            " This dataset is IDEAL for questions about: solar deployment timelines, capacity expansion over time, growth rates,"
            " historical installation trends, cumulative capacity analysis, and year-over-year commissioning statistics."
            " This dataset is for EXISTING facilities only. DO NOT route questions about potential sites, candidate locations,"
            " where to build solar, good places for solar, or future solar development—those go to the Clay Solar Candidate dataset."
            " Treat questions about the dataset's contents, coverage, provenance, maintainers, data quality, or how to use the tools as supported."
            " Respond only with JSON of the form {\"supported\": true|false, \"reason\": \"short explanation\"}.\n"
            f"Dataset capabilities: {self._capability_summary()}\n"
            f"Question: {query}"
        )

        try:
            response = self._anthropic_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=128,
                temperature=0,
                system="Respond with valid JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        except Exception as exc:  # pragma: no cover - network failures
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
        reason = str(data.get("reason")) if data.get("reason") else None
        score = 0.9 if supported else 0.1

        reasons = [reason] if reason else []
        if not reasons:
            reasons.append("LLM classification")

        return SupportIntent(supported=supported, score=score, reasons=reasons)

    @staticmethod
    def _normalize_country(name: str) -> str:
        name = name.strip()

        mapping = {
            "usa": "United States of America",
            "united states": "United States of America",
            "us": "United States of America",
            "america": "United States of America",
            "uk": "United Kingdom",
            "britain": "United Kingdom",
            "england": "United Kingdom",
            "brasil": "Brazil",
        }
        return mapping.get(name.lower(), name)

    @staticmethod
    def _facility_buffer_radius_m(radius_km: Optional[float]) -> float:
        if radius_km is None:
            return FACILITY_BUFFER_MIN_METERS
        try:
            radius_value = max(0.0, float(radius_km))
        except (TypeError, ValueError):
            return FACILITY_BUFFER_MIN_METERS

        scaled = radius_value * FACILITY_BUFFER_SCALE_PER_KM
        if scaled <= 0:
            return FACILITY_BUFFER_MIN_METERS
        return max(
            FACILITY_BUFFER_MIN_METERS,
            min(FACILITY_BUFFER_MAX_METERS, scaled),
        )

    def _generate_geojson(
        self,
        facilities: List[Any],
        identifier: str,
        *,
        polygons: Optional[Iterable[Any]] = None,
        point_buffer_radius_km: Optional[float] = None,
    ) -> GeoJSONSummary:
        project_root = Path(__file__).resolve().parents[2]
        static_maps_dir = project_root / "static" / "maps"
        static_maps_dir.mkdir(parents=True, exist_ok=True)

        feature_collection: Dict[str, Any] = {"type": "FeatureCollection", "features": []}
        total_capacity = 0.0
        capacities: List[float] = []
        countries: Dict[str, int] = {}
        min_lat: Optional[float] = None
        max_lat: Optional[float] = None
        min_lon: Optional[float] = None
        max_lon: Optional[float] = None

        facility_label = "Solar Asset"
        facility_key = facility_label.lower()
        use_polygon_markers = polygons is not None
        point_buffer_m = (
            self._facility_buffer_radius_m(point_buffer_radius_km)
            if use_polygon_markers
            else None
        )

        polygon_records = list(polygons or [])
        polygon_features: List[Dict[str, Any]] = []
        polygon_stats: Dict[str, Dict[str, Any]] = {}
        for polygon in polygon_records:
            geometry_wkt = getattr(polygon, "geometry_wkt", None)
            properties = getattr(polygon, "properties", {}) or {}
            polygon_id = getattr(polygon, "polygon_id", None)
            if not geometry_wkt:
                continue
            try:
                geometry = shapely_wkt.loads(geometry_wkt)
            except Exception:
                continue

            layer_name = "deforestation_polygon"
            properties = {
                **properties,
                "country": properties.get("country")
                or layer_name.replace("_", " ").lower(),
                "polygon_id": polygon_id,
                "layer": layer_name,
                "color_value": 1,
                "color_hex": POLYGON_LAYER_COLORS.get(layer_name, "#F4511E"),
                "color": POLYGON_LAYER_COLORS.get(layer_name, "#F4511E"),
            }
            polygon_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(geometry),
                    "properties": properties,
                }
            )
            stats = polygon_stats.setdefault(
                layer_name,
                {
                    "count": 0,
                    "total_area_hectares": 0.0,
                    "years": set(),
                },
            )
            stats["count"] += 1

            area = properties.get("area_hectares")
            try:
                if area is not None:
                    stats["total_area_hectares"] += float(area)
            except (TypeError, ValueError):
                pass

            year_value = properties.get("year")
            if isinstance(year_value, (int, float)):
                stats["years"].add(str(int(year_value)))
            elif isinstance(year_value, str) and year_value.strip():
                stats["years"].add(year_value.strip())

            bounds = geometry.bounds
            minx, miny, maxx, maxy = bounds
            min_lon = minx if min_lon is None else min(min_lon, minx)
            min_lat = miny if min_lat is None else min(min_lat, miny)
            max_lon = maxx if max_lon is None else max(max_lon, maxx)
            max_lat = maxy if max_lat is None else max(max_lat, maxy)

        facility_records: List[Dict[str, Any]] = []
        for facility in facilities:
            facility_dict = facility.as_dict() if hasattr(facility, "as_dict") else facility
            if not isinstance(facility_dict, dict):
                continue

            lat = facility_dict.get("latitude")
            lon = facility_dict.get("longitude")
            if lat is None or lon is None:
                continue

            facility_records.append(facility_dict)

            country = facility_dict.get("country", "Unknown")
            countries[country] = countries.get(country, 0) + 1
            capacity = facility_dict.get("capacity_mw") or 0.0
            total_capacity += capacity
            if capacity:
                capacities.append(capacity)

            min_lat = lat if min_lat is None else min(min_lat, lat)
            max_lat = lat if max_lat is None else max(max_lat, lat)
            min_lon = lon if min_lon is None else min(min_lon, lon)
            max_lon = lon if max_lon is None else max(max_lon, lon)

            properties = {
                "cluster_id": facility_dict.get("cluster_id"),
                "name": facility_dict.get("name") or facility_dict.get("cluster_id"),
                "capacity_mw": round(capacity, 2) if capacity else 0,
                "country": country,
                "facility_country": country,
                "constructed_before": facility_dict.get("constructed_before"),
                "constructed_after": facility_dict.get("constructed_after"),
                "layer": "solar_facility",
                "color_value": 1,
                "color_hex": FACILITY_BUFFER_COLOR,
                "color": FACILITY_BUFFER_COLOR,
            }

            if use_polygon_markers:
                geometry_geojson = _buffer_point_as_circle(
                    float(lon),
                    float(lat),
                    radius_m=point_buffer_m or CIRCLE_BUFFER_METERS,
                )
            else:
                geometry_geojson = {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                }

            feature_collection["features"].append(
                {
                    "type": "Feature",
                    "geometry": geometry_geojson,
                    "properties": properties,
                }
            )

        identifier_slug = identifier.lower().replace(" ", "_").replace(",", "")[:50]
        data_hash = hashlib.md5(f"{identifier_slug}_{len(feature_collection['features'])}".encode()).hexdigest()[:8]
        filename = f"solar_facilities_{identifier_slug}_{data_hash}.geojson"
        output_path = static_maps_dir / filename

        plotted_count = len(facility_records)
        metadata: Dict[str, Any] = {
            "plotted_facilities": plotted_count,
            "total_capacity_mw": round(total_capacity, 1),
            "countries": sorted(countries.keys()),
            "country_counts": countries,
        }
        metadata["layers"] = {
            "solar_asset": {
                "count": plotted_count,
                "plotted_count": plotted_count,
                "total_capacity_mw": round(total_capacity, 1),
                "color": FACILITY_BUFFER_COLOR,
                "color_property": "color_value",
                "fill_style": {
                    "type": "interpolate",
                    "color_property": "color_value",
                    "range": [0, 1],
                    "colorMin": FACILITY_BUFFER_COLOR,
                    "colorMax": FACILITY_BUFFER_COLOR,
                },
                "z_index": 1,
            }
        }
        if point_buffer_m:
            metadata["layers"]["solar_asset"]["buffer_radius_m"] = point_buffer_m
        has_facilities = bool(facility_records)

        filtered_polygon_stats: Dict[str, Dict[str, Any]] = {}
        for layer_name, stats in polygon_stats.items():
            if layer_name == "deforestation_polygon" and not has_facilities:
                continue
            filtered_polygon_stats[layer_name] = stats

        for layer_name, stats in filtered_polygon_stats.items():
            metadata_key = "deforestation_polygons" if layer_name == "deforestation_polygon" else layer_name
            layer_color = POLYGON_LAYER_COLORS.get(layer_name, "#1E88E5")
            layer_payload: Dict[str, Any] = {
                "count": stats.get("count", 0),
                "color": layer_color,
                "color_property": "color_value",
                "fill_style": {
                    "type": "interpolate",
                    "color_property": "color_value",
                    "range": [0, 1],
                    "colorMin": layer_color,
                    "colorMax": layer_color,
                },
                "z_index": 2,
            }
            total_area = stats.get("total_area_hectares")
            if total_area:
                layer_payload["total_area_hectares"] = round(float(total_area), 2)
            years = stats.get("years")
            if isinstance(years, set) and years:
                layer_payload["years"] = sorted(years)
            metadata["layers"][metadata_key] = layer_payload

        legend = metadata.setdefault(
            "legend",
            {
                "title": "Layers",
                "items": [],
            },
        )
        legend_items = legend.setdefault("items", [])
        legend_items.append(
            {
                "label": facility_label,
                "color": FACILITY_BUFFER_COLOR,
                # "description": f"{plotted_count} facilities",
            }
        )
        for layer_name, layer_payload in metadata["layers"].items():
            if layer_name == "solar_asset":
                continue
            color = layer_payload.get("color") or "#1E88E5"
            count = layer_payload.get("count")
            legend_item = {
                "label": layer_name.replace("_", " ").title(),
                "color": color,
            }
            if count is not None:
                legend_item["description"] = f"{count} features"
            legend_items.append(legend_item)

        if min_lat is not None and max_lat is not None and min_lon is not None and max_lon is not None:
            lat_span = max_lat - min_lat
            lon_span = max_lon - min_lon
            padding_lat = max(lat_span * 0.1, 0.5) if lat_span else 0.5
            padding_lon = max(lon_span * 0.1, 0.5) if lon_span else 0.5

            bounds = {
                "north": float(max_lat + padding_lat),
                "south": float(min_lat - padding_lat),
                "east": float(max_lon + padding_lon),
                "west": float(min_lon - padding_lon),
            }
            metadata["bounds"] = bounds
            metadata["center"] = {
                "lon": float((min_lon + max_lon) / 2),
                "lat": float((min_lat + max_lat) / 2),
            }
        if capacities:
            metadata["capacity_range_mw"] = {
                "min": round(min(capacities), 1),
                "max": round(max(capacities), 1),
                "average": round(sum(capacities) / len(capacities), 1),
            }

        # Append polygons last so they draw above buffered facilities in most renderers.
        if polygon_features:
            if not has_facilities:
                polygon_features = [
                    feature
                    for feature in polygon_features
                    if feature.get("properties", {}).get("layer") != "deforestation_polygon"
                ]
            if polygon_features:
                feature_collection["features"].extend(polygon_features)

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({"type": "FeatureCollection", **feature_collection}, handle)

        metadata["geometry_type"] = "polygon" if use_polygon_markers or polygon_records else "point"
        metadata["merge_group"] = "solar_facilities"

        return GeoJSONSummary(url=f"/static/maps/{filename}", metadata=metadata)

    # ------------------------------------------------------------------ tools
    def _register_capabilities_tool(self) -> None:
        @self.mcp.tool()
        def describe_capabilities(format: str = "json") -> str:  # type: ignore[misc]
            """Describe the dataset, provenance, and key tools.

            Example:
                >>> describe_capabilities()
            """

            payload = self._capabilities_metadata()
            return json.dumps(payload) if format == "json" else str(payload)

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> str:  # type: ignore[misc]
            """Decide whether the solar dataset is relevant to the query.

            Example:
                >>> query_support("Map solar plants in Brazil", {})
            """

            context_mapping = context if isinstance(context, dict) else {}
            inferred_state = self._infer_state_from_context(query, context_mapping)
            if inferred_state:
                intent = SupportIntent(
                    supported=True,
                    score=0.85,
                    reasons=[
                        f"Follow-up referencing {inferred_state.name} from earlier solar summary"
                    ],
                )
            else:
                intent = self._classify_support(query)
            payload = {
                "server": "solar",
                "query": query,
                "supported": intent.supported,
                "score": intent.score,
                "reasons": intent.reasons,
            }
            return json.dumps(payload)

    def _register_tool_top_countries(self) -> None:
        @self.mcp.tool()
        def get_top_countries_by_facilities(limit: int = 10) -> dict:  # type: ignore[misc]
            """Return the countries with the highest facility counts.

            Example:
                >>> get_top_countries_by_facilities(limit=5)
            """

            stats = self.db.get_country_statistics()
            ranked = sorted(stats, key=lambda item: item.get("facility_count", 0), reverse=True)[:limit]
            rows = [
                {
                    "country": item["country"],
                    "facility_count": item["facility_count"],
                    "total_capacity_mw": item.get("total_capacity_mw"),
                }
                for item in ranked
            ]
            return {"rows": rows, "limit": limit}

    def _register_tool_countries(self) -> None:
        @self.mcp.tool()
        def get_solar_facilities_countries() -> dict:  # type: ignore[misc]
            """List every country present in the dataset."""

            return {"countries": self.db.get_all_country_names()}

    def _register_tool_find_country(self) -> None:
        @self.mcp.tool()
        def find_solar_facilities_countries(partial_name: str) -> dict:  # type: ignore[misc]
            """Suggest country names matching the provided text.

            Example:
                >>> find_solar_facilities_countries("bra")
            """

            return {"matches": self.db.find_country_by_partial_name(partial_name)}

    def _register_tool_facilities_by_country(self) -> None:
        @self.mcp.tool()
        def get_solar_facilities_by_country(country: str, limit: int = 200) -> dict:  # type: ignore[misc]
            """Fetch facility-level details and summary statistics for a country.

            Example:
                >>> get_solar_facilities_by_country(country="Brazil", limit=50)
            """

            country_norm = self._normalize_country(country)
            facilities = self.db.get_facilities_by_country(country_norm, limit=limit)
            total = len(facilities)
            capacity = sum((f.get("capacity_mw") or 0.0) for f in facilities)
            return {
                "country": country_norm,
                "limit": limit,
                "count": total,
                "total_capacity_mw": round(capacity, 1),
                "facilities": facilities,
            }

    def _register_tool_map_data(self) -> None:
        @self.mcp.tool()
        def get_solar_facilities_map_data(country: str | None = None, limit: int = 5000) -> dict:  # type: ignore[misc]
            """Generate GeoJSON for facilities (optionally filtered by country).

            Example:
                >>> get_solar_facilities_map_data(country="Brazil")
            """

            target_country = self._normalize_country(country) if country else "Brazil"
            facilities = self.db.get_facilities_by_country(target_country, limit=limit)
            geojson = self._generate_geojson(facilities, identifier=target_country.lower())

            stats = self._country_statistics(target_country) or {}
            facility_count = int(stats.get("facility_count") or len(facilities))
            total_capacity = float(stats.get("total_capacity_mw") or 0.0)
            if not total_capacity:
                total_capacity = sum((f.get("capacity_mw") or 0.0) for f in facilities)

            summary = f"Mapped {facility_count:,} tracked solar facilities in {target_country}."
            facts: List[str] = []
            if total_capacity:
                facts.append(
                    f"These sites represent roughly {self._format_capacity(total_capacity)} MW of capacity in {target_country}."
                )

            artifacts = [
                {
                    "type": "map",
                    "title": f"Solar facilities in {target_country} (TZ-SAM Q1 2025)",
                    "geojson_url": geojson.url,
                    "metadata": geojson.metadata,
                }
            ]

            return {
                "country": target_country,
                "geojson_url": geojson.url,
                "metadata": geojson.metadata,
                "summary": summary,
                "facts": facts,
                "artifacts": artifacts,
                "citation": self._dataset_citation_dict(),
            }

    def _register_tool_facilities_for_geospatial(self) -> None:
        @self.mcp.tool()
        def get_facilities_for_geospatial(country: str | None = None, limit: int = 1000) -> dict:  # type: ignore[misc]
            """Return simplified entities suitable for geospatial registration."""

            if country:
                facilities = self.db.get_facilities_by_country(self._normalize_country(country), limit=limit)
            else:
                facilities = self.db.get_facilities_by_country("Brazil", limit=limit)
            entities = [
                {
                    "id": f.get("cluster_id"),
                    "latitude": f.get("latitude"),
                    "longitude": f.get("longitude"),
                    "properties": {
                        "country": f.get("country"),
                        "capacity_mw": f.get("capacity_mw"),
                    },
                }
                for f in facilities
                if f.get("latitude") is not None and f.get("longitude") is not None
            ]
            return {"entities": entities, "count": len(entities)}

    def _register_tool_facilities_in_radius(self) -> None:
        @self.mcp.tool()
        def get_solar_facilities_in_radius(latitude: float, longitude: float, radius_km: float = 50, limit: int = 200) -> dict:  # type: ignore[misc]
            """Find facilities within a radius of the provided coordinate."""

            facilities = self.db.get_facilities_in_radius(latitude, longitude, radius_km, limit=limit)
            return {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius_km": radius_km,
                "count": len(facilities),
                "facilities": facilities,
            }

    def _register_tool_facilities_in_brazil_state(self) -> None:
        @self.mcp.tool()
        def get_solar_facilities_in_brazil_state(state: str, limit: int = 5000) -> dict:  # type: ignore[misc]
            """Return facilities located within a Brazilian state polygon."""

            state_match = self._match_brazil_state(state)
            if not state_match:
                return {
                    "error": f"State '{state}' not found in Brazilian administrative dataset.",
                    "available": sorted(state.name for state in self._brazil_states.values()),
                }

            facilities = self._facilities_within_state(state_match, limit=limit)
            geojson = self._generate_geojson(
                facilities,
                identifier=f"brazil_state_{self._normalize_text(state_match.name).replace(' ', '_')}",
                polygons=[
                    SimplePolygon(
                        polygon_id=f"state_{self._normalize_text(state_match.name).replace(' ', '_')}",
                        geometry_wkt=state_match.geometry.wkt,
                        properties={
                            "layer": "brazil_state",
                            "name": state_match.name,
                            "abbreviation": state_match.abbreviation,
                        },
                    )
                ],
            )

            total_capacity = sum((f.get("capacity_mw") or 0.0) for f in facilities)
            return {
                "state": state_match.name,
                "abbreviation": state_match.abbreviation,
                "count": len(facilities),
                "total_capacity_mw": round(total_capacity, 1),
                "geojson_url": geojson.url,
                "metadata": geojson.metadata,
                "facilities": facilities[:limit],
            }

    def _register_tool_facilities_in_bounds(self) -> None:
        @self.mcp.tool()
        def get_solar_facilities_in_bounds(north: float, south: float, east: float, west: float, limit: int = 500) -> dict:  # type: ignore[misc]
            """Return facilities inside a bounding box (west/east/north/south)."""

            facilities = self.db.get_facilities_in_bounds(north=north, south=south, east=east, west=west, limit=limit)
            geojson = self._generate_geojson(facilities, identifier="bounds")
            return {"geojson_url": geojson.url, "metadata": geojson.metadata}

    def _register_tool_facilities_multiple_countries(self) -> None:
        @self.mcp.tool()
        def get_solar_facilities_multiple_countries(countries: list[str], limit: int = 500) -> dict:  # type: ignore[misc]
            """Create a combined GeoJSON for several countries."""

            combined: List[Dict[str, Any]] = []
            for country in countries:
                combined.extend(self.db.get_facilities_by_country(self._normalize_country(country), limit=limit))
            geojson = self._generate_geojson(combined, identifier="_".join(countries))
            return {"geojson_url": geojson.url, "metadata": geojson.metadata}

    def _register_tool_capacity_by_country(self) -> None:
        @self.mcp.tool()
        def get_solar_capacity_by_country(limit: int = 20) -> dict:  # type: ignore[misc]
            """Return aggregated counts and capacity per country."""

            stats = self.db.get_country_statistics()
            ranked = sorted(
                stats,
                key=lambda item: item.get("total_capacity_mw") or item.get("facility_count", 0),
                reverse=True,
            )[:limit]

            labels = [item.get("country", "Unknown") for item in ranked]
            values = [float(item.get("total_capacity_mw") or 0.0) for item in ranked]
            chart = {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Solar Capacity (MW)",
                        "data": values,
                        "backgroundColor": "#43A047",
                    }
                ],
                "options": {
                    "scales": {
                        "y": {
                            "title": {
                                "display": True,
                                "text": "Capacity (MW)",
                            }
                        }
                    }
                },
            }

            leader = ranked[0] if ranked else {}
            leader_name = leader.get("country", "Unknown")
            leader_capacity = self._format_capacity(leader.get("total_capacity_mw") or 0.0)
            summary = (
                f"{leader_name} leads tracked solar capacity with {leader_capacity} MW among the top {len(ranked)} countries."
            )

            facts: List[str] = []
            total_capacity = sum(values)
            if total_capacity:
                facts.append(
                    f"These {len(ranked)} countries account for {self._format_capacity(total_capacity)} MW combined."
                )

            target_country = "Brazil"
            target_entry = next((item for item in ranked if item.get("country") == target_country), None)
            if target_entry:
                target_rank = ranked.index(target_entry) + 1
                target_capacity = self._format_capacity(target_entry.get("total_capacity_mw") or 0.0)
                target_facilities = self._format_integer(target_entry.get("facility_count"))
                facts.append(
                    f"{target_country} ranks {target_rank} with {target_capacity} MW across {target_facilities} tracked facilities."
                )
            elif any(item.get("country") == target_country for item in stats):
                facts.append(
                    f"{target_country} does not appear in the top {len(ranked)} tracked countries but remains present in the global dataset."
                )

            table_rows = [
                [
                    item.get("country"),
                    self._format_integer(item.get("facility_count")),
                    self._format_capacity(item.get("total_capacity_mw") or 0.0),
                ]
                for item in ranked
            ]

            artifacts = [
                {
                    "type": "chart",
                    "title": "Top countries by tracked solar capacity",
                    "data": {
                        "labels": chart["labels"],
                        "datasets": chart["datasets"],
                    },
                    "metadata": {
                        "chartType": "bar",
                        "datasetLabel": "Solar Capacity (MW)",
                        "options": chart.get("options", {}),
                    },
                },
                {
                    "type": "table",
                    "title": "Top countries by tracked solar capacity",
                    "columns": ["Country", "Facilities", "Capacity (MW)"],
                    "rows": table_rows,
                },
            ]

            return {
                "countries": ranked,
                "limit": limit,
                "summary": summary,
                "facts": facts,
                "artifacts": artifacts,
                "citation": self._dataset_citation_dict(),
            }

    def _register_tool_capacity_by_state(self) -> None:
        @self.mcp.tool()
        def get_solar_capacity_by_state(top_n: int = 10) -> dict:  # type: ignore[misc]
            """Return a bar chart of Brazilian states by solar capacity."""

            chart = self._build_state_capacity_chart(top_n=top_n)
            if not chart:
                return {"error": "State aggregation unavailable"}

            stats = self._state_facility_stats[: max(1, top_n)]
            top_entries = [
                f"{item.get('state_name')} ({self._format_capacity(item.get('total_capacity_mw') or 0.0)} MW)"
                for item in stats[: min(len(stats), 3)]
            ]
            summary = (
                "Top Brazilian states by tracked solar capacity are "
                + ", ".join(top_entries)
                + "."
            )

            brazil_stats = self._country_statistics("Brazil") or {}
            national_capacity = float(brazil_stats.get("total_capacity_mw") or 0.0)
            national_facilities = int(brazil_stats.get("facility_count") or 0)

            facts: List[str] = []
            if national_capacity and stats:
                subset_capacity = sum(float(item.get("total_capacity_mw") or 0.0) for item in stats)
                share = (subset_capacity / national_capacity) * 100 if national_capacity else 0.0
                facts.append(
                    f"The top {len(stats)} states account for {self._format_capacity(subset_capacity)} MW (≈{share:.1f}% of Brazil's tracked capacity)."
                )
            if national_facilities and stats:
                subset_facilities = sum(int(item.get("facility_count") or 0) for item in stats)
                share_facilities = (subset_facilities / national_facilities) * 100 if national_facilities else 0.0
                facts.append(
                    f"They include {self._format_integer(subset_facilities)} facilities (≈{share_facilities:.1f}% of national sites)."
                )

            table_rows = [
                [
                    item.get("state_name"),
                    self._format_integer(item.get("facility_count")),
                    self._format_capacity(item.get("total_capacity_mw") or 0.0),
                ]
                for item in stats
            ]

            artifacts = [
                {
                    "type": "chart",
                    "title": "Top Brazilian states by solar capacity",
                    "data": {
                        "labels": chart["labels"],
                        "datasets": chart["datasets"],
                    },
                    "metadata": {
                        "chartType": "bar",
                        "datasetLabel": "Solar Capacity (MW)",
                        "options": chart.get("options", {}),
                    },
                },
                {
                    "type": "table",
                    "title": "Top Brazilian states by solar capacity",
                    "columns": ["State", "Facilities", "Capacity (MW)"],
                    "rows": table_rows,
                },
            ]

            return {
                "labels": chart["labels"],
                "datasets": chart["datasets"],
                "options": chart.get("options"),
                "metadata": {"chartType": "bar"},
                "items": stats,
                "summary": summary,
                "facts": facts,
                "artifacts": artifacts,
                "citation": self._dataset_citation_dict(),
            }

    def _register_tool_capacity_by_municipality(self) -> None:
        @self.mcp.tool()
        def get_solar_capacity_by_municipality(top_n: int = 10) -> dict:  # type: ignore[misc]
            """Return a bar chart of municipalities by solar capacity."""

            chart = self._build_municipality_capacity_chart(top_n=top_n)
            if not chart:
                return {"error": "Municipality aggregation unavailable"}

            stats = self._municipality_facility_stats[: max(1, top_n)]
            top_entries = [
                (
                    f"{item.get('municipality_name')} ({item.get('state')}) "
                    f"{self._format_capacity(item.get('total_capacity_mw') or 0.0)} MW"
                )
                for item in stats[: min(len(stats), 3)]
            ]
            summary = (
                "Top municipalities by tracked capacity include "
                + ", ".join(top_entries)
                + "."
            )

            facts: List[str] = []
            total_capacity = sum(float(item.get("total_capacity_mw") or 0.0) for item in stats)
            if total_capacity:
                facts.append(
                    f"Combined, these municipalities host {self._format_capacity(total_capacity)} MW of solar capacity."
                )

            table_rows = [
                [
                    item.get("municipality_name"),
                    item.get("state"),
                    self._format_integer(item.get("facility_count")),
                    self._format_capacity(item.get("total_capacity_mw") or 0.0),
                ]
                for item in stats
            ]

            artifacts = [
                {
                    "type": "chart",
                    "title": "Top Brazilian municipalities by solar capacity",
                    "data": {
                        "labels": chart["labels"],
                        "datasets": chart["datasets"],
                    },
                    "metadata": {
                        "chartType": "bar",
                        "datasetLabel": "Solar Capacity (MW)",
                        "options": chart.get("options", {}),
                    },
                },
                {
                    "type": "table",
                    "title": "Top Brazilian municipalities by solar capacity",
                    "columns": ["Municipality", "State", "Facilities", "Capacity (MW)"],
                    "rows": table_rows,
                },
            ]

            return {
                "labels": chart["labels"],
                "datasets": chart["datasets"],
                "options": chart.get("options"),
                "metadata": {"chartType": "bar"},
                "items": stats,
                "summary": summary,
                "facts": facts,
                "artifacts": artifacts,
                "citation": self._dataset_citation_dict(),
            }

    def _register_tool_construction_timeline(self) -> None:
        @self.mcp.tool()
        def get_solar_construction_timeline(country: str | None = None) -> dict:  # type: ignore[misc]
            """Get comprehensive solar deployment timeline with annual and cumulative capacity.

            Returns year-over-year solar facility commissioning data including both
            annual additions and running cumulative totals. Shows facility counts and
            capacity (MW) to track deployment growth over time.

            Data is derived from construction date fields in the TransitionZero Solar
            Asset Mapper dataset (99% temporal coverage across 103K+ facilities).

            Use this tool to:
            - Analyze solar deployment growth rates and trends
            - Identify peak installation years and capacity additions
            - Track total accumulated solar capacity over time
            - Compare annual build-out rates across different periods
            - Generate time-series visualizations showing growth trajectory

            The response includes both annual additions and cumulative totals, plus
            Chart.js-compatible visualization payloads for immediate rendering.

            Args:
                country: Country name for analysis. Defaults to "Brazil" if not specified.
                    Examples: "Brazil", "United States", "India", "China", "Germany"

            Returns:
                dict with the following structure:
                {
                    "country": str,                    # Country analyzed
                    "annual": List[Dict],              # Annual additions by year
                        # [{"year": "2023", "facilities": 682, "capacity_mw": 8543.25}, ...]
                    "cumulative": List[Dict],          # Running totals by year
                        # [{"year": "2023", "total_facilities": 2111, "total_capacity_mw": 20901.81}, ...]
                    "summary": str,                    # Human-readable growth summary
                    "facts": List[str],                # Key insights (peak years, growth rates)
                    "artifacts": List[Dict],           # Chart.js payloads (2 charts)
                    "citation": Dict                   # Dataset citation information
                }

            Example:
                >>> get_solar_construction_timeline("Brazil")
                {
                    "country": "Brazil",
                    "annual": [
                        {"year": "2017", "facilities": 44, "capacity_mw": 1509.92},
                        {"year": "2023", "facilities": 682, "capacity_mw": 8543.25}
                    ],
                    "cumulative": [
                        {"year": "2017", "total_facilities": 44, "total_capacity_mw": 1509.92},
                        {"year": "2025", "total_facilities": 2273, "total_capacity_mw": 26022.51}
                    ],
                    "summary": "Solar deployment in Brazil grew from 2017 to 2025, commissioning 2,273 facilities with total capacity of 26,022.51 MW.",
                    "facts": [
                        "Peak installation year: 2023 (8,543.25 MW commissioned)",
                        "Capacity increased 17.2x from 2017 to 2025"
                    ],
                    ...
                }
            """

            target = country or "Brazil"

            # Get comprehensive timeline data with capacity
            timeline_data = self._construction_timeline_with_capacity(target)
            annual = timeline_data.get("annual", [])
            cumulative = timeline_data.get("cumulative", [])
            data_summary = timeline_data.get("summary", {})

            # Build summary and facts
            summary: Optional[str] = None
            facts: List[str] = []

            if data_summary:
                total_capacity = data_summary.get("total_capacity_mw", 0)
                total_facilities = data_summary.get("total_facilities", 0)
                first_year = data_summary.get("first_year")
                last_year = data_summary.get("last_year")

                summary = (
                    f"Solar deployment in {target} grew from {first_year} to {last_year}, "
                    f"commissioning {total_facilities} facilities with total capacity of "
                    f"{total_capacity:,.2f} MW."
                )

                # Find peak installation year
                if annual:
                    peak_year_data = max(annual, key=lambda x: x.get("capacity_mw", 0))
                    peak_year = peak_year_data["year"]
                    peak_capacity = peak_year_data["capacity_mw"]
                    facts.append(f"Peak installation year: {peak_year} ({peak_capacity:,.2f} MW commissioned)")

                # Calculate growth rate
                if len(cumulative) >= 2:
                    start_capacity = cumulative[0]["total_capacity_mw"]
                    end_capacity = cumulative[-1]["total_capacity_mw"]
                    if start_capacity > 0:
                        growth_multiple = end_capacity / start_capacity
                        facts.append(f"Capacity increased {growth_multiple:.1f}x from {first_year} to {last_year}")

            # Build charts
            artifacts = []

            # Annual additions chart
            if annual:
                annual_labels = [x["year"] for x in annual]
                annual_capacity_values = [x["capacity_mw"] for x in annual]
                artifacts.append({
                    "type": "chart",
                    "title": f"Annual Solar Capacity Additions - {target}",
                    "data": {
                        "labels": annual_labels,
                        "datasets": [{
                            "label": "Annual Capacity Additions (MW)",
                            "data": annual_capacity_values,
                            "borderColor": "#4CAF50",
                            "backgroundColor": "rgba(76, 175, 80, 0.35)",
                            "fill": True,
                            "yAxisID": "y"
                        }]
                    },
                    "metadata": {
                        "chartType": "bar",
                        "options": {
                            "scales": {
                                "y": {
                                    "type": "linear",
                                    "display": True,
                                    "position": "left",
                                    "title": {
                                        "display": True,
                                        "text": "Capacity (MW)"
                                    }
                                }
                            }
                        }
                    }
                })

            # Cumulative capacity chart
            if cumulative:
                cumulative_labels = [x["year"] for x in cumulative]
                cumulative_capacity_values = [x["total_capacity_mw"] for x in cumulative]
                artifacts.append({
                    "type": "chart",
                    "title": f"Cumulative Solar Capacity - {target}",
                    "data": {
                        "labels": cumulative_labels,
                        "datasets": [{
                            "label": "Total Installed Capacity (MW)",
                            "data": cumulative_capacity_values,
                            "borderColor": "#2196F3",
                            "backgroundColor": "rgba(33, 150, 243, 0.35)",
                            "fill": True,
                            "yAxisID": "y"
                        }]
                    },
                    "metadata": {
                        "chartType": "bar",
                        "options": {
                            "scales": {
                                "y": {
                                    "type": "linear",
                                    "display": True,
                                    "position": "left",
                                    "title": {
                                        "display": True,
                                        "text": "Total Capacity (MW)"
                                    }
                                }
                            }
                        }
                    }
                })

            return {
                "country": target,
                "annual": annual,
                "cumulative": cumulative,
                "summary": summary,
                "facts": facts,
                "artifacts": artifacts,
                "citation": self._dataset_citation_dict(),
            }

    def _register_tool_largest_facilities(self) -> None:
        @self.mcp.tool()
        def get_largest_solar_facilities(limit: int = 20, country: str | None = None) -> dict:  # type: ignore[misc]
            """Return the highest-capacity facilities (optionally filtered by country)."""

            facilities = self.db.get_largest_facilities(limit=limit, country=self._normalize_country(country) if country else None)
            geojson = self._generate_geojson(facilities, identifier=f"largest_{country or 'global'}")
            return {
                "geojson_url": geojson.url,
                "metadata": geojson.metadata,
                "facilities": facilities,
            }

    def _register_tool_facility_details(self) -> None:
        @self.mcp.tool()
        def get_solar_facility_details(cluster_id: str) -> dict:  # type: ignore[misc]
            """Return detailed attributes for a specific facility."""

            facility = self.db.get_facility_by_id(cluster_id)
            return facility or {}

    def _register_tool_facility_location(self) -> None:
        @self.mcp.tool()
        def get_solar_facility_location(cluster_id: str) -> dict:  # type: ignore[misc]
            """Return coordinates and a GeoJSON snippet for a facility."""

            return self._facility_location_payload(cluster_id)

    def _facility_location_payload(self, cluster_id: str) -> Dict[str, Any]:
        facility = self.db.get_facility_by_id(cluster_id)
        if not facility:
            return {
                "cluster_id": cluster_id,
                "error": f"No solar facility found for cluster_id {cluster_id}",
            }

        latitude = facility.get("latitude")
        longitude = facility.get("longitude")
        if latitude is None or longitude is None:
            return {
                "cluster_id": cluster_id,
                "facility": facility,
                "error": "Facility lacks coordinate data",
            }

        summary = self._generate_geojson([facility], identifier=f"facility_{cluster_id}")
        response: Dict[str, Any] = {
            "cluster_id": facility.get("cluster_id"),
            "country": facility.get("country"),
            "capacity_mw": facility.get("capacity_mw"),
            "constructed_before": facility.get("constructed_before"),
            "constructed_after": facility.get("constructed_after"),
            "coordinates": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "geojson_url": summary.url,
            "geojson_metadata": summary.metadata,
            "facility": facility,
        }
        return response

    def _register_tool_capacity_visualization(self) -> None:
        @self.mcp.tool()
        def get_solar_capacity_visualization_data(limit: int = 10) -> dict:  # type: ignore[misc]
            """Provide bar-chart ready data of facilities per country."""

            stats = self.db.get_country_statistics()
            ranked = sorted(stats, key=lambda item: item.get("facility_count", 0), reverse=True)[:limit]
            labels = [item["country"] for item in ranked]
            counts = [item["facility_count"] for item in ranked]
            return {
                "chartType": "bar",
                "labels": labels,
                "datasets": [
                    {
                        "label": "Facility count",
                        "data": counts,
                    }
                ],
            }

    def _register_tool_facilities_near_polygon(self) -> None:
        @self.mcp.tool()
        def get_solar_facilities_near_polygon(
            polygon_geojson: dict, buffer_km: float = 0.0, limit: int = 1000
        ) -> dict:  # type: ignore[misc]
            """Return facilities intersecting the provided polygon.

            Args:
                polygon_geojson: GeoJSON feature or feature collection describing the area.
                buffer_km: Optional buffer distance (in kilometres).
                limit: Maximum facilities to process.

            Example:
                >>> get_solar_facilities_near_polygon({"type": "Feature", ...})
            """

            facilities = self.db.get_facilities_by_country("Brazil", limit=limit)
            geometries = []
            features = polygon_geojson.get("features") if polygon_geojson.get("type") == "FeatureCollection" else [polygon_geojson]
            for feature in features or []:
                try:
                    geometries.append(shape(feature.get("geometry")))
                except Exception:
                    continue

            matched: List[Dict[str, Any]] = []
            for facility in facilities:
                lat = facility.get("latitude")
                lon = facility.get("longitude")
                if lat is None or lon is None:
                    continue
                point = Point(lon, lat)
                for geom in geometries:
                    target = geom.buffer(buffer_km / 111.0) if buffer_km else geom
                    if target.contains(point):
                        matched.append(facility)
                        break

            geojson = self._generate_geojson(
                matched,
                identifier="polygon",
                point_buffer_radius_km=buffer_km or None,
            )
            aggregation: Dict[str, Dict[str, Any]] = {}
            for facility in matched:
                key = facility.get("country", "Unknown")
                entry = aggregation.setdefault(key, {"facility_count": 0, "capacity_mw": 0.0})
                entry["facility_count"] += 1
                entry["capacity_mw"] += facility.get("capacity_mw") or 0.0

            return {
                "geojson_url": geojson.url,
                "metadata": geojson.metadata,
                "aggregations": aggregation,
            }

    def _register_tool_facilities_near_deforestation(self) -> None:
        @self.mcp.tool()
        def get_solar_facilities_near_deforestation(
            country: Optional[str] = None,
            radius_km: float = 1.0,
            limit: int = 200,
        ) -> dict:  # type: ignore[misc]
            """Correlate solar facilities with deforestation polygons within ``radius_km``.

            Use when the user explicitly mentions deforestation overlap or proximity (e.g.,
            "within X km of deforestation"). For general renewable maps prefer
            ``get_solar_facilities_map_data``.
            """

            cleaned_country = country.strip() if isinstance(country, str) else ""
            assumed_country = not cleaned_country
            resolved_country = cleaned_country or "Brazil"

            match_limit = limit if limit > 0 else None
            correlation = self._compute_deforestation_correlation(
                resolved_country,
                radius_km=radius_km,
                match_limit=match_limit,
            )
            if correlation is None:
                return {
                    "error": "Spatial correlation prerequisites missing; ensure providers initialised.",
                    "country": resolved_country,
                }

            payload = {
                "country": correlation["country"],
                "radius_km": correlation["radius_km"],
                "match_count": len(correlation["matches"]),
                "matches": correlation["matches"],
            }
            notes: List[str] = []
            if assumed_country:
                notes.append(
                    "Country not provided; defaulting to Brazil deforestation polygons."
                )
            geojson_summary = correlation.get("geojson")
            if geojson_summary:
                payload["geojson_url"] = geojson_summary.url
                payload["metadata"] = geojson_summary.metadata
                payload.setdefault("artifacts", []).append(
                    {
                        "type": "map",
                        "title": f"Solar facilities near deforestation in {resolved_country}",
                        "geojson_url": geojson_summary.url,
                        "metadata": geojson_summary.metadata,
                    }
                )
            facilities = correlation.get("facilities") or []
            polygons = correlation.get("polygons") or []
            facility_ids = {
                getattr(facility, "cluster_id", None)
                for facility in facilities
                if getattr(facility, "cluster_id", None)
            }
            facility_count = len(facility_ids)
            polygon_count = len({getattr(polygon, "polygon_id", None) for polygon in polygons if getattr(polygon, "polygon_id", None)})
            total_capacity = sum(
                (getattr(facility, "capacity_mw", 0.0) or 0.0)
                for facility in facilities
            )
            total_area = 0.0
            years: set[str] = set()
            for polygon in polygons:
                properties = getattr(polygon, "properties", {}) or {}
                area_value = properties.get("area_hectares")
                try:
                    if area_value is not None:
                        total_area += float(area_value)
                except (TypeError, ValueError):
                    pass
                year_value = properties.get("year")
                if isinstance(year_value, (int, float)):
                    years.add(str(int(year_value)))
                elif isinstance(year_value, str) and year_value.strip():
                    years.add(year_value.strip())

            facts: List[str] = []
            if facility_count:
                facts.append(
                    f"Identified {facility_count} solar facilities in {resolved_country} within {radius_km} km of tracked deforestation polygons."
                )
            if polygon_count:
                text = f"These overlaps intersect {polygon_count} deforestation polygons"
                if total_area:
                    text += f" spanning roughly {round(total_area, 1)} hectares"
                text += "."
                facts.append(text)
            if total_capacity:
                facts.append(
                    f"The matched facilities represent about {round(total_capacity, 1)} MW of nameplate capacity."
                )
            if facts:
                payload["facts"] = facts
            if years:
                payload.setdefault("metadata", {}).setdefault("layers", {}).setdefault(
                    "deforestation_polygons",
                    {},
                ).setdefault("years", sorted(years))

            citation = self._dataset_citation_dict(
                description=(
                    f"Solar facilities correlated with deforestation polygons within {radius_km} km."
                )
            )
            citation.setdefault("metadata", {}).update(
                {
                    "country": resolved_country,
                    "radius_km": radius_km,
                    "datasets": [DATASET_ID, DEFORESTATION_DATASET_ID],
                    "facility_count": facility_count,
                    "polygon_count": polygon_count,
                }
            )
            payload["citation"] = citation
            if not correlation["matches"]:
                payload["note"] = "No overlapping deforestation polygons detected within the specified distance."
            if notes:
                payload["notes"] = notes
            return payload

    # ------------------------------------------------------------------ run_query
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        if not isinstance(context, dict):
            context = {}
        context_mapping = context if isinstance(context, Mapping) else {}

        state_context = (
            context.get("state")
            or context.get("state_name")
            or context.get("admin_area")
            or context.get("region")
        )

        query_lower = query.lower()

        state_match: Optional[BrazilState] = None
        inferred_state_from_context: Optional[BrazilState] = None
        if context_mapping:
            try:
                inferred_state_from_context = self._infer_state_from_context(query, context_mapping)
            except Exception:
                inferred_state_from_context = None

        if self._brazil_states:
            state_match = self._match_brazil_state(state_context)
            if not state_match:
                state_match = self._detect_brazil_state(query)
            if not state_match and inferred_state_from_context:
                state_match = inferred_state_from_context

        country_input = self._detect_country(query) or context.get("country") or "Brazil"
        country = self._normalize_country(country_input)
        radius_km = self._extract_radius_from_context(context, default=1.0)
        radius_from_query = self._extract_radius_from_query(query)
        if radius_from_query is not None:
            radius_km = radius_from_query

        if self._should_correlate_with_deforestation(query, context):
            correlation = self._compute_deforestation_correlation(
                country,
                radius_km=radius_km,
                match_limit=200,
            )
            if correlation is not None:
                return self._build_deforestation_run_query(query, correlation)
            deforestation_warning = (
                "Deforestation correlation unavailable; install geopandas/shapely to enable the geospatial bridge."
            )
        else:
            deforestation_warning = None

        if state_match and country.lower() == "brazil":
            return self._build_state_run_query(query, state_match)

        sample_limit = 3000
        raw_country_data = self.db.get_facilities_by_country(country, limit=sample_limit)
        target_country_norm = self._normalize_country(country)
        country_data = [
            facility
            for facility in raw_country_data
            if self._normalize_country(str(facility.get("country") or ""))
            == target_country_norm
        ]
        if not country_data and raw_country_data:
            country_data = raw_country_data
        map_data = self._generate_geojson(country_data, identifier=country.lower())
        stats = self._country_statistics(country)
        total_facilities_from_stats = (
            int(stats.get("facility_count"))
            if isinstance(stats, dict) and stats.get("facility_count") is not None
            else len(country_data)
        )
        if isinstance(map_data.metadata, dict):
            meta = map_data.metadata
            meta["sample_size"] = len(country_data)
            meta["total_facilities"] = total_facilities_from_stats
            meta.setdefault("legend_title", f"Solar Facilities in {country}")

            layers_meta = meta.get("layers")
            if isinstance(layers_meta, Mapping):
                solar_layer = layers_meta.get("solar_facilities")
                if isinstance(solar_layer, dict):
                    plotted_count = solar_layer.get("plotted_count") or solar_layer.get("count")
                    if plotted_count is None:
                        plotted_count = len(country_data)
                    solar_layer["plotted_count"] = int(plotted_count)
                    solar_layer["total_facilities"] = total_facilities_from_stats

        solar_meta = _dataset_metadata(DATASET_ID) or {}
        citation = CitationPayload(
            id="tz_sam_q1_2025",
            server="solar",
            tool="run_query",
            title=solar_meta.get("title") or DATASET_NAME,
            source_type="Dataset",
            description=_dataset_citation(DATASET_ID)
            or f"Solar facilities dataset filtered for {country}.",
            url=solar_meta.get("source") or DATASET_URL,
        )

        facility_count = int(stats.get("facility_count")) if isinstance(stats, dict) and stats.get("facility_count") is not None else len(country_data)
        total_capacity = float(stats.get("total_capacity_mw")) if isinstance(stats, dict) and stats.get("total_capacity_mw") is not None else sum((f.get("capacity_mw") or 0.0) for f in country_data)
        largest_records = self.db.get_largest_facilities(limit=1, country=country)
        largest = largest_records[0] if largest_records else None

        facts: List[FactPayload] = [
            FactPayload(
                id="facility_count",
                text=f"The dataset lists {facility_count} solar facilities in {country}.",
                citation_id=citation.id,
                metadata={"country": country},
            )
        ]
        if total_capacity:
            facts.append(
                FactPayload(
                    id="capacity_sum",
                    text=f"These facilities represent roughly {round(total_capacity, 1)} MW of nameplate capacity in {country}.",
                    citation_id=citation.id,
                )
            )
        if largest and largest.get("capacity_mw"):
            facts.append(
                FactPayload(
                    id="largest_facility",
                    text=f"The largest tracked installation in {country} is {largest.get('name') or largest.get('cluster_id')} at {round(largest.get('capacity_mw'), 1)} MW.",
                    citation_id=citation.id,
                    metadata={"cluster_id": largest.get("cluster_id")},
                )
            )

        if isinstance(map_data.metadata, dict):
            def _to_int(value: Any) -> Optional[int]:
                try:
                    if value is None:
                        return None
                    if isinstance(value, int):
                        return value
                    if isinstance(value, float):
                        return int(round(value))
                    return int(float(value))
                except (TypeError, ValueError):
                    return None

            solar_layer_meta: Dict[str, Any] = {}
            layers_meta = map_data.metadata.get("layers")
            if isinstance(layers_meta, Mapping):
                layer_meta = layers_meta.get("solar_facilities")
                if isinstance(layer_meta, Mapping):
                    solar_layer_meta = dict(layer_meta)
                    layers_meta["solar_facilities"] = solar_layer_meta

            plotted_count = _to_int(solar_layer_meta.get("plotted_count") or solar_layer_meta.get("count"))
            sample_size = _to_int(map_data.metadata.get("sample_size"))
            total_facilities_meta = _to_int(
                solar_layer_meta.get("total_facilities")
                or map_data.metadata.get("total_facilities")
            )

            if plotted_count:
                sample_clause = ""
                if total_facilities_meta and plotted_count < total_facilities_meta:
                    sample_clause = f" (sample of {total_facilities_meta} tracked nationally)"
                map_fact_text = (
                    f"An interactive GeoJSON map accompanies this answer, plotting {plotted_count} solar facilities in {country}{sample_clause}."
                )
                facts.append(
                    FactPayload(
                        id="map_availability",
                        kind="map",
                        text=map_fact_text,
                        citation_id=citation.id,
                        metadata={
                            "geojson_url": map_data.url,
                            "plotted_count": plotted_count,
                            "total_facilities": total_facilities_meta,
                            "sample_size": sample_size,
                        },
                    )
                )

        artifacts = [
            ArtifactPayload(
                id="solar_map",
                type="map",
                title=f"Solar facilities in {country}",
                geojson_url=map_data.url,
                metadata=map_data.metadata,
            ),
        ]

        timeline_requested = any(
            keyword in query_lower
            for keyword in ("timeline", "trend", "trendline", "historical", "over time")
        )
        if timeline_requested:
            timeline_series = self._construction_timeline_series(country)
            timeline_chart = self._build_timeline_chart(timeline_series)
            if timeline_chart:
                if len(timeline_series) >= 2:
                    first_year, first_count = timeline_series[0]
                    last_year, last_count = timeline_series[-1]
                    facts.append(
                        FactPayload(
                            id="timeline_trend",
                            text=(
                                f"Commissioning records show {first_count} facilities in {first_year} "
                                f"rising to {last_count} in {last_year} for {country}."
                            ),
                            citation_id=citation.id,
                        )
                    )
                artifacts.append(
                    ArtifactPayload(
                        id="solar_construction_timeline",
                        type="chart",
                        title=f"Solar construction timeline ({country})",
                        data=timeline_chart,
                        metadata={"chartType": "line"},
                    )
                )
            else:
                messages.append(
                    MessagePayload(
                        level="warning",
                        text="Commissioning timeline unavailable; missing construction dates in source data.",
                    )
                )

        kg = KnowledgeGraphPayload(
            nodes=[
                {"id": "solar_energy", "label": "solar energy", "type": "Concept"},
                {"id": country.lower().replace(" ", "_"), "label": country, "type": "Location"},
            ],
            edges=[
                {"source": "solar_energy", "target": country.lower().replace(" ", "_"), "type": "LOCATES"}
            ],
        )

        messages: List[MessagePayload] = []
        if deforestation_warning:
            messages.append(MessagePayload(level="warning", text=deforestation_warning))
        visible_sample = len(country_data)
        if facility_count > visible_sample:
            messages.append(
                MessagePayload(
                    level="info",
                    text=(
                        f"Showing a sample of {visible_sample} facilities out of {facility_count} available for {country}."
                    ),
                )
            )
        if visible_sample < len(raw_country_data):
            messages.append(
                MessagePayload(
                    level="info",
                    text="Filtered out facilities with ambiguous country labels to focus the map on the requested country.",
                )
            )

        return RunQueryResponse(
            server="solar",
            query=query,
            facts=facts,
            citations=[citation],
            artifacts=artifacts,
            messages=messages,
            kg=kg,
        )

    # ------------------------------------------------------------------ helpers
    def _compute_deforestation_correlation(
        self,
        country: str,
        *,
        radius_km: float,
        match_limit: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not (self.facility_provider and self.deforestation_provider and self.spatial_correlator):
            return None

        normalized_country = self._normalize_country(country)
        facilities = self.facility_provider.facilities_by_country(normalized_country)
        if not facilities:
            return {
                "country": normalized_country,
                "radius_km": radius_km,
                "matches": [],
                "facilities": [],
                "polygons": [],
                "geojson": None,
            }

        facility_points = self.facility_provider.facilities_to_points(facilities)
        polygons = self.deforestation_provider.polygons_near_points(
            facility_points,
            distance_km=radius_km,
        )
        if not polygons:
            return {
                "country": normalized_country,
                "radius_km": radius_km,
                "matches": [],
                "facilities": [],
                "polygons": [],
                "geojson": None,
            }

        polygon_features = [polygon.to_polygon_feature() for polygon in polygons]
        matches = self.spatial_correlator.points_within_polygons(
            facility_points,
            polygon_features,
            distance_km=radius_km,
        )

        if match_limit is not None and match_limit > 0:
            matches = matches[:match_limit]

        facility_ids = [match.get("facility_id") for match in matches if match.get("facility_id")]
        polygon_ids = [match.get("polygon_id") for match in matches if match.get("polygon_id")]

        facility_ids_unique = list(dict.fromkeys(facility_ids))
        polygon_ids_unique = list(dict.fromkeys(polygon_ids))

        matched_facilities = []
        for fid in facility_ids_unique:
            if not fid:
                continue
            facility = self.facility_provider.get(fid)
            if facility:
                matched_facilities.append(facility)

        polygon_map = {polygon.polygon_id: polygon for polygon in polygons}
        matched_polygons = [polygon_map[pid] for pid in polygon_ids_unique if pid in polygon_map]

        geojson_summary: Optional[GeoJSONSummary] = None
        if matched_facilities and matched_polygons:
            geojson_summary = self._generate_geojson(
                matched_facilities,
                identifier=f"{normalized_country}_deforestation",
                polygons=matched_polygons,
                point_buffer_radius_km=radius_km,
            )

        return {
            "country": normalized_country,
            "radius_km": radius_km,
            "matches": matches,
            "facilities": matched_facilities,
            "polygons": matched_polygons,
            "geojson": geojson_summary,
        }

    @staticmethod
    def _extract_radius_from_context(context: dict, default: float = 1.0) -> float:
        candidates = []
        if isinstance(context, dict):
            direct = context.get("radius_km")
            if direct is not None:
                candidates.append(direct)
            filters = context.get("filters")
            if isinstance(filters, dict):
                candidates.append(filters.get("radius_km"))
        for candidate in candidates:
            try:
                value = float(candidate)  # type: ignore[arg-type]
                if value > 0:
                    return value
            except (TypeError, ValueError):
                continue
        return default

    @staticmethod
    def _extract_radius_from_query(query: str, default: Optional[float] = None) -> Optional[float]:
        if not query:
            return default
        lowered = query.lower()
        meter_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(m|meter|metre|meters|metres)\b",
            lowered,
        )
        if meter_match:
            try:
                value = float(meter_match.group(1)) / 1000.0
                return value if value > 0 else default
            except ValueError:
                return default

        km_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(km|kilometer|kilometre|kilometers|kilometres)\b",
            lowered,
        )
        if km_match:
            try:
                value = float(km_match.group(1))
                return value if value > 0 else default
            except ValueError:
                return default

        return default

    @staticmethod
    def _should_correlate_with_deforestation(query: str, context: dict) -> bool:
        if isinstance(context, dict):
            if context.get("correlate_deforestation"):
                return True
            filters = context.get("filters")
            if isinstance(filters, dict) and filters.get("dataset") == "deforestation":
                return True

        lowered = query.lower()
        defo_tokens = [
            "deforest",
            "forest loss",
            "clearance",
            "logging",
            "rainforest",
            "amazon",
        ]
        proximity_tokens = ["within", "near", "close", "adjacent", "km", "kilometre", "kilometer"]
        if any(token in lowered for token in defo_tokens):
            if any(token in lowered for token in proximity_tokens):
                return True
        return False

    def _build_deforestation_run_query(self, query: str, correlation: Dict[str, Any]) -> RunQueryResponse:
        country = correlation["country"]
        radius_km = correlation["radius_km"]
        matches: List[Dict[str, Any]] = correlation["matches"]
        matched_facilities: List[Any] = correlation["facilities"]
        matched_polygons: List[Any] = correlation["polygons"]
        geojson_summary: Optional[GeoJSONSummary] = correlation.get("geojson")

        solar_meta = _dataset_metadata(DATASET_ID) or {}
        solar_citation = CitationPayload(
            id="tz_sam_q1_2025",
            server="solar",
            tool="run_query",
            title=solar_meta.get("title") or DATASET_NAME,
            source_type="Dataset",
            description=_dataset_citation(DATASET_ID) or DATASET_NAME,
            url=solar_meta.get("source") or DATASET_URL,
        )

        defo_meta = _dataset_metadata(DEFORESTATION_DATASET_ID) or {}
        defo_citation = CitationPayload(
            id="brazil_deforestation_prodes",
            server="solar",
            tool="run_query",
            title=defo_meta.get("title") or "Brazil deforestation polygons",
            source_type="Dataset",
            description=_dataset_citation(DEFORESTATION_DATASET_ID)
            or "PRODES deforestation polygons maintained by INPE.",
            url=defo_meta.get("source") or None,
        )

        facility_count = len({match.get("facility_id") for match in matches if match.get("facility_id")})
        polygon_count = len({match.get("polygon_id") for match in matches if match.get("polygon_id")})
        total_capacity = sum((facility.capacity_mw or 0.0) for facility in matched_facilities)
        total_area = 0.0
        years: set[str] = set()
        for polygon in matched_polygons:
            properties = getattr(polygon, "properties", {}) or {}
            area_value = properties.get("area_hectares")
            try:
                if area_value is not None:
                    total_area += float(area_value)
            except (TypeError, ValueError):
                pass
            year_value = properties.get("year")
            if isinstance(year_value, (int, float)):
                years.add(str(int(year_value)))
            elif isinstance(year_value, str) and year_value.strip():
                years.add(year_value.strip())

        facts: List[FactPayload] = []
        if matches:
            facts.append(
                FactPayload(
                    id="solar_deforest_facilities",
                    text=(
                        f"Identified {facility_count} solar facilities in {country} within {radius_km} km of tracked deforestation polygons."
                    ),
                    citation_id=solar_citation.id,
                    metadata={"radius_km": radius_km, "facility_count": facility_count},
                )
            )
            facts.append(
                FactPayload(
                    id="solar_deforest_area",
                    text=(
                        f"These overlaps intersect {polygon_count} deforestation polygons covering roughly {round(total_area, 1)} hectares"
                    ),
                    citation_id=defo_citation.id,
                    metadata={"polygon_count": polygon_count, "years": sorted(years)},
                )
            )
            if total_capacity:
                facts.append(
                    FactPayload(
                        id="solar_deforest_capacity",
                        text=(
                            f"The matched facilities represent about {round(total_capacity, 1)} MW of nameplate capacity."
                        ),
                        citation_id=solar_citation.id,
                    )
                )
        else:
            facts.append(
                FactPayload(
                    id="solar_deforest_none",
                    text=(
                        f"No solar facilities in {country} fall within {radius_km} km of recorded deforestation polygons."
                    ),
                    citation_id=defo_citation.id,
                    metadata={"radius_km": radius_km},
                )
            )

        artifacts: List[ArtifactPayload] = []
        if geojson_summary:
            artifacts.append(
                ArtifactPayload(
                    id="solar_deforest_map",
                    type="map",
                    title=f"Solar facilities near deforestation in {country}",
                    geojson_url=geojson_summary.url,
                    metadata=geojson_summary.metadata,
                )
            )

        kg = KnowledgeGraphPayload(
            nodes=[
                {"id": "solar_energy", "label": "solar energy", "type": "Concept"},
                {"id": "deforestation", "label": "deforestation", "type": "Concept"},
                {"id": country.lower().replace(" ", "_"), "label": country, "type": "Location"},
            ],
            edges=[
                {"source": "solar_energy", "target": country.lower().replace(" ", "_"), "type": "LOCATES"},
                {"source": "deforestation", "target": country.lower().replace(" ", "_"), "type": "LOCATES"},
                {"source": "solar_energy", "target": "deforestation", "type": "INTERSECTS"},
            ],
        )

        return RunQueryResponse(
            server="solar",
            query=query,
            facts=facts,
            citations=[solar_citation, defo_citation],
            artifacts=artifacts,
            messages=[],
            kg=kg,
        )

    def _detect_country(self, query: str) -> Optional[str]:
        lowered = query.lower()
        for country in self.db.get_all_country_names():
            if country and country.lower() in lowered:
                return country
        return None


def create_server() -> FastMCP:
    """Entry point used by ``python -m mcp.servers_v2.solar_server_v2``."""

    server = SolarServerV2()
    return server.mcp


if __name__ == "__main__":  # pragma: no cover - manual execution
    create_server().run()
