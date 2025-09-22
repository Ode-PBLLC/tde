"""MapBiomas/PRODES deforestation polygons exposed via the MCP v2 contract."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

from fastmcp import FastMCP
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping

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
    from mcp.geospatial_datasets import DeforestationPolygonProvider  # type: ignore
    from mcp.servers_v2.support_intent import SupportIntent  # type: ignore
else:
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
    from ..geospatial_datasets import DeforestationPolygonProvider
    from .support_intent import SupportIntent


DATASET_ID = "brazil_deforestation"
DEFAULT_LIMIT = 200

STATE_STATS_PATH = Path(__file__).resolve().parents[2] / "static" / "meta" / "deforestation_by_state.json"
MUNICIPALITY_STATS_PATH = (
    Path(__file__).resolve().parents[2] / "static" / "meta" / "deforestation_by_municipality.json"
)

POLYGON_LAYER_COLORS: Dict[str, str] = {
    "deforestation_polygon": "#D84315",
    "brazil_state": "#1E88E5",
    "municipalities": "#3949AB",
}


def _load_dataset_metadata() -> Dict[str, Dict[str, str]]:
    path = Path(__file__).resolve().parents[2] / "static" / "meta" / "datasets.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    metadata: Dict[str, Dict[str, str]] = {}
    for item in payload.get("items", []):
        dataset_id = item.get("id")
        if not dataset_id:
            continue
        metadata[str(dataset_id)] = {
            "title": str(item.get("title", "")),
            "source": str(item.get("source", "")),
            "citation": str(item.get("citation", "")),
        }
    return metadata


DATASET_METADATA = _load_dataset_metadata()


@dataclass
class GeoJSONSummary:
    url: str
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        self.url = ensure_absolute_url(self.url)

class DeforestationServerV2(RunQueryMixin):
    """FastMCP server providing deforestation polygons and v2 run_query."""

    def __init__(self) -> None:
        self.mcp = FastMCP("deforestation-server-v2")
        self.provider = DeforestationPolygonProvider()
        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential issues
                print(f"[deforestation-server] Warning: Anthropic client unavailable: {exc}")

        self._register_capabilities_tool()
        self._register_query_support_tool()
        self._register_tool_dataset_overview()
        self._register_tool_polygons_by_area()
        self._register_tool_polygons_in_bounds()
        self._register_tool_area_by_year()
        self._register_tool_area_by_state()
        self._register_tool_area_by_municipality()
        self._register_run_query_tool()
        self._state_area_stats: List[Dict[str, Any]] = []
        self._municipality_area_stats: List[Dict[str, Any]] = []
        self._load_precomputed_admin_stats()

    # ------------------------------------------------------------------ tools
    def _register_capabilities_tool(self) -> None:
        @self.mcp.tool()
        def describe_capabilities(format: str = "json") -> str:  # type: ignore[misc]
            """Describe the deforestation dataset and supported tools."""

            overview = self.provider.dataset_overview()
            payload = {
                "name": "deforestation",
                "description": "Brazilian deforestation polygons from INPE/PRODES prepared for spatial analytics.",
                "dataset": DATASET_METADATA.get(DATASET_ID, {}).get("title", "PRODES Deforestation"),
                "polygon_count": overview.get("polygon_count"),
                "years": overview.get("years"),
                "tools": [
                    "describe_capabilities",
                    "query_support",
                    "get_deforestation_overview",
                    "get_deforestation_polygons",
                    "get_deforestation_polygons_in_bounds",
                    "get_deforestation_area_by_year",
                    "get_deforestation_area_by_state",
                    "get_deforestation_area_by_municipality",
                    "run_query",
                ],
            }
            return json.dumps(payload) if format == "json" else str(payload)

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> str:  # type: ignore[misc]
            """Return whether the deforestation dataset can help with the query."""

            intent = self._classify_intent(query)
            payload = {
                "server": "deforestation",
                "query": query,
                "supported": intent.supported,
                "score": intent.score,
                "reasons": intent.reasons,
            }
            return json.dumps(payload)

    def _register_tool_dataset_overview(self) -> None:
        @self.mcp.tool()
        def get_deforestation_overview() -> dict:  # type: ignore[misc]
            """Return dataset-wide statistics (polygon counts, area, temporal coverage)."""

            overview = self.provider.dataset_overview()
            return overview

    def _register_tool_polygons_by_area(self) -> None:
        @self.mcp.tool()
        def get_deforestation_polygons(
            min_area_hectares: float = 0.0,
            max_area_hectares: float | None = None,
            limit: int = 200,
        ) -> dict:  # type: ignore[misc]
            """Retrieve polygons filtered by area thresholds."""

            polygons = self.provider.polygons_by_area(
                min_area_hectares=min_area_hectares,
                max_area_hectares=max_area_hectares,
                limit=limit,
            )
            summary = self._generate_geojson(polygons, identifier="area_filter")
            return {
                "count": len(polygons),
                "geojson_url": summary.url,
                "metadata": summary.metadata,
                "polygons": [self._polygon_to_dict(polygon) for polygon in polygons[:50]],
            }

    def _register_tool_polygons_in_bounds(self) -> None:
        @self.mcp.tool()
        def get_deforestation_polygons_in_bounds(
            north: float,
            south: float,
            east: float,
            west: float,
            limit: int = 200,
        ) -> dict:  # type: ignore[misc]
            """Return polygons intersecting a bounding box."""

            polygons = self.provider.polygons_in_bounds(north=north, south=south, east=east, west=west, limit=limit)
            summary = self._generate_geojson(polygons, identifier="bounds")
            return {
                "count": len(polygons),
                "geojson_url": summary.url,
                "metadata": summary.metadata,
                "polygons": [self._polygon_to_dict(polygon) for polygon in polygons[:50]],
            }

    def _register_tool_area_by_year(self) -> None:
        @self.mcp.tool()
        def get_deforestation_area_by_year(
            min_total_area_hectares: float = 0.0,
            min_year: str | None = None,
            max_year: str | None = None,
            limit: int = 30,
            chart_type: str = "line",
        ) -> dict:  # type: ignore[misc]
            """Return precomputed deforestation area summaries grouped by year."""

            summary = self.provider.area_by_year(min_year=min_year, max_year=max_year)
            filtered = [entry for entry in summary if entry["total_area_hectares"] >= min_total_area_hectares]
            if limit > 0:
                filtered = filtered[:limit]

            labels = [entry["year"] for entry in reversed(filtered)]
            dataset_values = [entry["total_area_hectares"] for entry in reversed(filtered)]
            chart = {
                "type": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [
                        {
                            "label": "Total deforestation area (ha)",
                            "data": dataset_values,
                        }
                    ],
                },
            }
            return {
                "summaries": filtered,
                "min_total_area_hectares": min_total_area_hectares,
                "min_year": min_year,
                "max_year": max_year,
                "limit": limit,
                "chart": chart,
            }

    def _register_tool_area_by_state(self) -> None:
        @self.mcp.tool()
        def get_deforestation_area_by_state(top_n: int = 10) -> dict:  # type: ignore[misc]
            """Return total deforested area by Brazilian state."""

            chart = self._build_state_chart(top_n)
            if not chart:
                return {"error": "State aggregation unavailable"}
            return {
                "labels": chart["labels"],
                "datasets": chart["datasets"],
                "metadata": {"chartType": "bar"},
                "items": self._state_area_stats[: max(1, top_n)],
            }

    def _register_tool_area_by_municipality(self) -> None:
        @self.mcp.tool()
        def get_deforestation_area_by_municipality(top_n: int = 10) -> dict:  # type: ignore[misc]
            """Return total deforested area by municipality."""

            chart = self._build_municipality_chart(top_n)
            if not chart:
                return {"error": "Municipality aggregation unavailable"}
            return {
                "labels": chart["labels"],
                "datasets": chart["datasets"],
                "metadata": {"chartType": "bar"},
                "items": self._municipality_area_stats[: max(1, top_n)],
            }

    @lru_cache(maxsize=1)
    def _capability_summary(self) -> str:
        """Return a cached natural-language summary of dataset capabilities."""

        overview = self.provider.dataset_overview()
        dataset_meta = DATASET_METADATA.get(DATASET_ID, {})
        parts = [
            "Dataset: INPE/PRODES deforestation polygons for Brazil.",
            f"Records: {overview.get('polygon_count')} polygons" if overview.get("polygon_count") else "",
            f"Years: {overview.get('years')}" if overview.get("years") else "",
            f"Total area: {overview.get('total_area_hectares')} hectares" if overview.get("total_area_hectares") else "",
            f"Source: {dataset_meta.get('source')}" if dataset_meta.get("source") else "",
        ]
        return " ".join(segment for segment in parts if segment)

    def _classify_intent(self, query: str) -> SupportIntent:
        """Use an LLM (if available) to judge dataset relevance."""

        if not self._anthropic_client:
            return SupportIntent(
                supported=True,
                score=0.3,
                reasons=["LLM unavailable; defaulting to dataset summary"],
            )

        capability_summary = self._capability_summary()
        prompt = (
            "Decide whether the following question should be answered using the"
            " INPE/PRODES deforestation polygons dataset."
            " Respond exclusively with JSON of the form {\"supported\": true|false,"
            " \"reason\": \"short explanation\"}.\n"
            f"Dataset summary: {capability_summary}\n"
            f"Question: {query}\n"
            "If the dataset clearly applies (e.g., questions about forest loss, PRODES,"
            " deforestation geospatial analysis in Brazil), set supported to true."
            " Otherwise set it to false."
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

    # ------------------------------------------------------------------ run_query
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        if not isinstance(context, dict):
            context = {}
        query_lower = query.lower()

        min_area = self._extract_numeric(context, "min_area_hectares", default=100.0)
        inferred_min = self._infer_min_area_from_query(query)
        if inferred_min is not None and inferred_min > min_area:
            min_area = inferred_min
        limit = int(context.get("limit", DEFAULT_LIMIT)) if isinstance(context.get("limit"), (int, float)) else DEFAULT_LIMIT

        polygons = self.provider.polygons_by_area(min_area_hectares=min_area, limit=limit)
        summary = self._generate_geojson(polygons, identifier="run_query") if polygons else None
        area_by_year = self.provider.area_by_year()
        chart_labels = [entry["year"] for entry in reversed(area_by_year[:30])]
        chart_values = [entry["total_area_hectares"] for entry in reversed(area_by_year[:30])]

        dataset_meta = DATASET_METADATA.get(DATASET_ID, {})
        citation = CitationPayload(
            id="brazil_deforestation_prodes",
            server="deforestation",
            tool="run_query",
            title=dataset_meta.get("title") or "PRODES Deforestation Polygons",
            source_type="Dataset",
            description=dataset_meta.get("citation") or "PRODES deforestation dataset maintained by INPE.",
            url=dataset_meta.get("source") or None,
        )

        overview = self.provider.dataset_overview()
        facts: List[FactPayload] = []
        facts.append(
            FactPayload(
                id="deforest_total",
                text=(
                    f"The dataset contains {overview.get('polygon_count')} deforestation polygons covering roughly {overview.get('total_area_hectares')} hectares."
                ),
                citation_id=citation.id,
            )
        )
        if polygons:
            selected_area = sum((polygon.properties.get("area_hectares") or 0.0) for polygon in polygons)
            facts.append(
                FactPayload(
                    id="deforest_filter",
                    text=(
                        f"Applied a minimum area filter of {min_area} hectares, returning {len(polygons)} polygons totalling about {round(selected_area, 1)} hectares."
                    ),
                    citation_id=citation.id,
                    metadata={"min_area_hectares": min_area},
                )
            )
        if area_by_year:
            top_year = max(area_by_year, key=lambda row: row["total_area_hectares"])
            top_area = round(float(top_year["total_area_hectares"]), 1)
            facts.append(
                FactPayload(
                    id="deforest_peak_year",
                    text=(
                        f"The heaviest recent activity occurred in {top_year['year']}, with approximately {top_area} hectares across {top_year['polygon_count']} polygons."
                    ),
                    citation_id=citation.id,
                    metadata={
                        "year": top_year["year"],
                        "total_area_hectares": top_area,
                    },
                )
            )
        else:
            facts.append(
                FactPayload(
                    id="deforest_no_matches",
                    text=f"No polygons exceeded the minimum area threshold of {min_area} hectares.",
                    citation_id=citation.id,
                )
            )

        rows: List[List[Any]] = []
        for polygon in polygons[:30]:
            props = polygon.properties
            rows.append(
                [
                    polygon.polygon_id,
                    props.get("area_hectares"),
                    props.get("year"),
                    props.get("state") or props.get("estado"),
                    props.get("biome"),
                ]
            )

        artifacts: List[ArtifactPayload] = []
        if summary:
            artifacts.append(
                ArtifactPayload(
                    id="deforest_map",
                    type="map",
                    title="Deforestation polygons",
                    geojson_url=summary.url,
                    metadata=summary.metadata,
                )
            )

        wants_state_chart = "state" in query_lower or "estado" in query_lower
        if wants_state_chart and self._state_area_stats:
            state_chart = self._build_state_chart(top_n=10)
            if state_chart:
                top_states = self._state_area_stats[:3]
                if top_states:
                    summary_text = ", ".join(
                        f"{entry.get('state_name')} ({float(entry.get('total_area_hectares', 0.0)):,.0f} ha)"
                        for entry in top_states
                    )
                    facts.append(
                        FactPayload(
                            id="deforest_state_summary",
                            text=f"States with the most recorded deforestation include {summary_text}.",
                            citation_id=citation.id,
                        )
                    )
                artifacts.append(
                    ArtifactPayload(
                        id="deforest_state_chart",
                        type="chart",
                        title="Top states by deforested area",
                        data=state_chart,
                        metadata={"chartType": "bar"},
                    )
                )

        if area_by_year:
            year_rows = [
                [
                    entry["year"],
                    entry["polygon_count"],
                    entry["total_area_hectares"],
                    entry["average_area_hectares"],
                ]
                for entry in area_by_year[:20]
            ]
            artifacts.append(
                ArtifactPayload(
                    id="deforest_area_by_year",
                    type="table",
                    title="Deforestation area by year",
                    data={
                        "columns": [
                            "Year",
                            "Polygon count",
                            "Total area (ha)",
                            "Average area (ha)",
                        ],
                        "rows": year_rows,
                    },
                )
            )
            artifacts.append(
                ArtifactPayload(
                    id="deforest_area_trend",
                    type="chart",
                    title="Deforestation area trend",
                    data={
                        "labels": chart_labels,
                        "datasets": [
                            {
                                "label": "Total area (ha)",
                                "data": chart_values,
                                "borderColor": "#EF6C00",
                                "backgroundColor": "rgba(239, 108, 0, 0.35)",
                                "fill": True,
                            }
                        ],
                    },
                    metadata={"chartType": "line"},
                )
            )

        messages: List[MessagePayload] = []
        if not polygons:
            messages.append(
                MessagePayload(
                    level="warning",
                    text="No polygons matched the requested filters; consider lowering the minimum area threshold.",
                )
            )

        kg = KnowledgeGraphPayload(
            nodes=[
                {"id": "deforestation", "label": "deforestation", "type": "Concept"},
                {"id": "brazil", "label": "Brazil", "type": "Location"},
            ],
            edges=[
                {"source": "deforestation", "target": "brazil", "type": "LOCATES"},
            ],
        )

        return RunQueryResponse(
            server="deforestation",
            query=query,
            facts=facts,
            citations=[citation],
            artifacts=artifacts,
            messages=messages,
            kg=kg,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _extract_numeric(context: dict, key: str, default: float) -> float:
        value = context.get(key)
        if value is None and isinstance(context.get("filters"), dict):
            value = context["filters"].get(key)
        try:
            if value is not None:
                number = float(value)
                if number > 0:
                    return number
        except (TypeError, ValueError):
            pass
        return default

    @staticmethod
    def _infer_min_area_from_query(query: str) -> float | None:
        text = query.lower()
        patterns = [
            r"(?:over|more than|greater than|larger than|at least|minimum of)\s+([0-9][\d,\.]*)\s*(?:hectare|hectares|ha)\b",
            r"([0-9][\d,\.]*)\s*(?:hectare|hectares|ha)\s*(?:or more|and above|minimum|min)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            raw_value = match.group(1).replace(",", "")
            try:
                return float(raw_value)
            except ValueError:
                continue
        return None

    def _generate_geojson(self, polygons: List[Any], identifier: str) -> GeoJSONSummary:
        project_root = Path(__file__).resolve().parents[2]
        static_maps_dir = project_root / "static" / "maps"
        static_maps_dir.mkdir(parents=True, exist_ok=True)

        features: List[Dict[str, Any]] = []
        total_area = 0.0
        years: set[str] = set()
        min_lon: Optional[float] = None
        min_lat: Optional[float] = None
        max_lon: Optional[float] = None
        max_lat: Optional[float] = None
        for polygon in polygons:
            try:
                geometry = shapely_wkt.loads(polygon.geometry_wkt)
            except Exception:
                continue
            properties = {**polygon.properties, "polygon_id": polygon.polygon_id}
            properties.setdefault("country", "deforestation")
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

            features.append({"type": "Feature", "geometry": mapping(geometry), "properties": properties})

            bounds = geometry.bounds
            minx, miny, maxx, maxy = bounds
            min_lon = minx if min_lon is None else min(min_lon, minx)
            min_lat = miny if min_lat is None else min(min_lat, miny)
            max_lon = maxx if max_lon is None else max(max_lon, maxx)
            max_lat = maxy if max_lat is None else max(max_lat, maxy)

        identifier_slug = identifier.replace(" ", "_")[:40]
        filename = f"deforestation_{identifier_slug}.geojson"
        output_path = static_maps_dir / filename
        payload = {"type": "FeatureCollection", "features": features}
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        metadata: Dict[str, Any] = {
            "polygon_count": len(features),
            "total_area_hectares": round(total_area, 2),
            "years": sorted(years),
            "geometry_type": "polygon",
        }
        metadata["layers"] = {
            "deforestation_polygon": {
                "count": len(features),
                "color": POLYGON_LAYER_COLORS.get("deforestation_polygon", "#D84315"),
            }
        }
        metadata["legend"] = {
            "title": "Layers",
            "items": [
                {
                    "label": "Deforestation polygons",
                    "color": POLYGON_LAYER_COLORS.get("deforestation_polygon", "#D84315"),
                    "description": f"{len(features)} polygons",
                }
            ],
        }
        if (
            min_lon is not None
            and max_lon is not None
            and min_lat is not None
            and max_lat is not None
        ):
            lon_span = max_lon - min_lon
            lat_span = max_lat - min_lat
            padding_lon = max(lon_span * 0.05, 0.25) if lon_span else 0.25
            padding_lat = max(lat_span * 0.05, 0.25) if lat_span else 0.25
            metadata["bounds"] = {
                "west": float(min_lon - padding_lon),
                "east": float(max_lon + padding_lon),
                "south": float(min_lat - padding_lat),
                "north": float(max_lat + padding_lat),
            }
            metadata["center"] = {
                "lon": float((min_lon + max_lon) / 2),
                "lat": float((min_lat + max_lat) / 2),
            }
        return GeoJSONSummary(url=f"/static/maps/{filename}", metadata=metadata)

    @staticmethod
    def _polygon_to_dict(polygon: Any) -> Dict[str, Any]:
        return {
            "polygon_id": polygon.polygon_id,
            "properties": polygon.properties,
        }

    def _load_precomputed_admin_stats(self) -> None:
        self._state_area_stats = self._read_stats_file(STATE_STATS_PATH)
        self._municipality_area_stats = self._read_stats_file(MUNICIPALITY_STATS_PATH)

    def _read_stats_file(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            print(f"[deforestation-server] Warning: precomputed stats missing at {path}")
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[deforestation-server] Warning: failed to load stats file {path.name}: {exc}")
            return []
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        cleaned: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                entry = dict(item)
                entry["polygons"] = int(entry.get("polygons", 0) or 0)
                entry["total_area_hectares"] = float(entry.get("total_area_hectares", 0.0) or 0.0)
                cleaned.append(entry)
        cleaned.sort(key=lambda row: row.get("total_area_hectares", 0.0), reverse=True)
        return cleaned

    def _build_state_chart(self, top_n: int = 10) -> Optional[Dict[str, Any]]:
        if not self._state_area_stats:
            return None
        stats = self._state_area_stats[: max(1, top_n)]
        labels = [entry.get("state_name") for entry in stats]
        values = [round(float(entry.get("total_area_hectares", 0.0)), 2) for entry in stats]
        dataset = {
            "label": "Deforested Area (hectares)",
            "data": values,
            "backgroundColor": "#D84315",
        }
        return {"labels": labels, "datasets": [dataset]}

    def _build_municipality_chart(self, top_n: int = 10) -> Optional[Dict[str, Any]]:
        if not self._municipality_area_stats:
            return None
        stats = self._municipality_area_stats[: max(1, top_n)]
        labels = [f"{entry.get('municipality_name')} ({entry.get('state')})" for entry in stats]
        values = [round(float(entry.get("total_area_hectares", 0.0)), 2) for entry in stats]
        dataset = {
            "label": "Deforested Area (hectares)",
            "data": values,
            "backgroundColor": "#6A1B9A",
        }
        return {"labels": labels, "datasets": [dataset]}


def create_server() -> FastMCP:
    server = DeforestationServerV2()
    return server.mcp


if __name__ == "__main__":  # pragma: no cover
    create_server().run()
