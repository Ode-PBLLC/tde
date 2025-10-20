"""In-memory deforestation polygons with STRtree acceleration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import geopandas as gpd
    from shapely.geometry import Point, box
    from shapely.strtree import STRtree
except ImportError as exc:  # pragma: no cover - geo stack optional in CI
    raise RuntimeError(
        "geopandas and shapely are required for deforestation correlation workflows"
    ) from exc

from mcp.geospatial_bridge import FacilityPoint, PolygonFeature


@dataclass(frozen=True)
class DeforestationPolygon:
    """Canonical polygon representation backed by the loaded GeoDataFrame."""

    polygon_id: str
    geometry_wkt: str
    properties: Dict[str, object]

    def to_polygon_feature(self) -> PolygonFeature:
        return PolygonFeature(id=self.polygon_id, geometry_wkt=self.geometry_wkt, properties=self.properties)


class DeforestationPolygonProvider:
    """Loads deforestation polygons once and exposes STRtree-based queries."""

    DATASET_CANDIDATES: Sequence[Tuple[str, Path]] = (
        ("parquet", Path("data/deforestation/deforestation.parquet")),
        ("parquet", Path("data/deforestation/deforestation_old.parquet")),
        ("geojson", Path("data/brazil_deforestation.geojson")),
    )

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parents[2]
        self.dataset_name: Optional[str] = None
        self._gdf: gpd.GeoDataFrame = self._load_dataset()
        self._gdf_metric = self._gdf.to_crs("EPSG:3857")
        self._geoms_metric = list(self._gdf_metric.geometry)
        self._id_lookup = {id(geom): idx for idx, geom in enumerate(self._geoms_metric)}
        self._tree = STRtree(self._geoms_metric)
        self._index_by_id: Dict[str, int] = {
            str(pid): idx for idx, pid in enumerate(self._gdf["__polygon_id"])
        }
        self._area_by_year: List[Dict[str, object]] = self._compute_area_by_year()

    def _load_dataset(self) -> "gpd.GeoDataFrame":
        candidates = self.DATASET_CANDIDATES
        for fmt, relative_path in candidates:
            path = (self.root / relative_path).resolve()
            if not path.exists():
                continue
            try:
                if fmt == "parquet":
                    gdf = gpd.read_parquet(path)
                else:
                    gdf = gpd.read_file(path)
            except Exception:
                continue
            if gdf.empty:
                continue
            gdf = gdf.copy()
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_string() != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")

            if "area_hectares" not in gdf.columns:
                try:
                    projected = gdf.to_crs("EPSG:5880")
                    gdf["area_hectares"] = projected.geometry.area / 10_000
                except Exception:
                    gdf["area_hectares"] = 0.0
            if "year" not in gdf.columns:
                gdf["year"] = None

            gdf.reset_index(drop=True, inplace=True)
            gdf["__polygon_id"] = [f"deforest_{idx}" for idx in range(len(gdf))]
            self.dataset_name = path.name
            return gdf
        raise RuntimeError("No deforestation dataset found in expected locations")

    def all_polygons(self) -> List[DeforestationPolygon]:
        return [self._row_to_polygon(idx) for idx in range(len(self._gdf))]

    def polygons_by_ids(self, polygon_ids: Iterable[str]) -> List[DeforestationPolygon]:
        indices = [self._index_by_id[pid] for pid in polygon_ids if pid in self._index_by_id]
        return [self._row_to_polygon(idx) for idx in indices]

    def polygons_near_points(
        self,
        facilities: Iterable[FacilityPoint],
        *,
        distance_km: float = 1.0,
        limit: int = 200,
    ) -> List[DeforestationPolygon]:
        if not facilities:
            return []

        points_gdf = gpd.GeoDataFrame(
            [{"id": facility.id, "geometry": Point(facility.longitude, facility.latitude)} for facility in facilities],
            geometry="geometry",
            crs="EPSG:4326",
        )
        points_metric = points_gdf.to_crs("EPSG:3857")
        buffer_distance_m = distance_km * 1_000
        buffered = gpd.GeoDataFrame(
            geometry=points_metric.geometry.buffer(buffer_distance_m),
            crs="EPSG:3857",
        )

        polygons_metric = self._gdf_metric
        try:
            joined = gpd.sjoin(polygons_metric, buffered, predicate="intersects", how="inner")
        except Exception:
            return []

        unique_indices = joined.index.unique()
        if unique_indices.empty:
            return []
        if limit is not None and limit > 0:
            unique_indices = unique_indices[:limit]
        return [self._row_to_polygon(int(idx)) for idx in unique_indices]

    def polygons_by_area(
        self,
        *,
        min_area_hectares: float = 0.0,
        max_area_hectares: Optional[float] = None,
        limit: int = 200,
    ) -> List[DeforestationPolygon]:
        filtered = self._gdf
        if min_area_hectares > 0:
            filtered = filtered[filtered["area_hectares"] >= min_area_hectares]
        if max_area_hectares is not None:
            filtered = filtered[filtered["area_hectares"] <= max_area_hectares]
        filtered = filtered.sort_values(by="area_hectares", ascending=False).head(limit)
        return [self._row_to_polygon(idx) for idx in filtered.index]

    def polygons_in_bounds(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        *,
        limit: int = 200,
    ) -> List[DeforestationPolygon]:
        bbox_geom = box(west, south, east, north)
        filtered = self._gdf[self._gdf.intersects(bbox_geom)].head(limit)
        return [self._row_to_polygon(idx) for idx in filtered.index]

    def dataset_overview(self) -> Dict[str, object]:
        years: set[str] = set()
        if "year" in self._gdf.columns:
            for year in self._gdf["year"]:
                if isinstance(year, (int, float)) and not math.isnan(year):
                    years.add(str(int(year)))
                elif isinstance(year, str) and year.strip():
                    years.add(year.strip())

        total_area = float(self._gdf["area_hectares"].sum()) if "area_hectares" in self._gdf.columns else 0.0
        return {
            "polygon_count": int(len(self._gdf)),
            "total_area_hectares": round(total_area, 2),
            "years": sorted(years),
        }

    def area_by_year(
        self,
        *,
        min_year: Optional[str] = None,
        max_year: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Return precomputed area aggregation grouped by year (with optional bounds)."""

        if min_year is None and max_year is None:
            return list(self._area_by_year)

        min_key = self._normalise_year(min_year) if min_year is not None else None
        max_key = self._normalise_year(max_year) if max_year is not None else None

        filtered: List[Dict[str, object]] = []
        for entry in self._area_by_year:
            year_key = entry.get("year")
            if not self._year_within_bounds(year_key, min_key, max_key):
                continue
            filtered.append(entry)
        return filtered

    def _compute_area_by_year(self) -> List[Dict[str, object]]:
        if "year" not in self._gdf.columns or "area_hectares" not in self._gdf.columns:
            return []

        summary: Dict[str, Dict[str, float]] = {}
        for _, row in self._gdf.iterrows():
            year_key = self._normalise_year(row.get("year"))
            if year_key is None:
                continue
            area_value = self._coerce_float(row.get("area_hectares"))
            if area_value is None or area_value <= 0:
                continue
            bucket = summary.setdefault(year_key, {"area": 0.0, "count": 0.0})
            bucket["area"] += float(area_value)
            bucket["count"] += 1.0

        results: List[Dict[str, object]] = []
        for year_key, data in summary.items():
            polygon_count = int(data["count"])
            total_area = data["area"]
            average_area = total_area / polygon_count if polygon_count else 0.0
            results.append(
                {
                    "year": year_key,
                    "polygon_count": polygon_count,
                    "total_area_hectares": round(total_area, 2),
                    "average_area_hectares": round(average_area, 2),
                }
            )

        results.sort(key=self._year_sort_key, reverse=True)
        return results

    def _row_to_polygon(self, idx: int) -> DeforestationPolygon:
        row = self._gdf.iloc[idx]
        polygon_id = str(row["__polygon_id"])
        geometry_wkt = row.geometry.wkt
        properties = {}
        for key in self._gdf.columns:
            if key in {"geometry", "__polygon_id"}:
                continue
            properties[key] = self._coerce_value(row[key])
        return DeforestationPolygon(polygon_id=polygon_id, geometry_wkt=geometry_wkt, properties=properties)

    @staticmethod
    def _coerce_value(value: object) -> object:
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                value = value.item()
        except Exception:
            pass
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return value

    @staticmethod
    def _coerce_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number):
            return None
        return number

    @staticmethod
    def _normalise_year(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if math.isnan(value):
                return None
            return str(int(value))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.isdigit():
                return str(int(text))
            return text
        return None

    @staticmethod
    def _year_sort_key(entry: Dict[str, object]) -> float:
        year_value = entry.get("year")
        if isinstance(year_value, (int, float)):
            return float(year_value)
        if isinstance(year_value, str):
            try:
                return float(int(year_value))
            except ValueError:
                return float("-inf")
        return float("-inf")

    @staticmethod
    def _year_within_bounds(year: object, min_year: Optional[str], max_year: Optional[str]) -> bool:
        def as_int(value: Optional[str]) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        year_norm = DeforestationPolygonProvider._normalise_year(year)
        if year_norm is None:
            return False

        year_int = as_int(year_norm)
        min_int = as_int(min_year)
        max_int = as_int(max_year)

        if min_int is not None and (year_int is None or year_int < min_int):
            return False
        if max_int is not None and (year_int is None or year_int > max_int):
            return False
        return True
