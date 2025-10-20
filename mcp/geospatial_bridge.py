"""Shared utilities for cross-server spatial workflows.

Servers that need proximity/overlap logic (e.g. solar assets near
deforestation areas) can import this module and keep *all* spatial
computations server-side, removing the need for Phase 2 orchestration.

The bridge exposes thin wrappers that:

* normalise incoming feature collections,
* perform shapely/geopandas powered spatial joins,
* return data in the same schema required by `run_query`.

This means a server such as `solar` can do:

```python
from mcp.geospatial_bridge import SpatialCorrelation

correlator = SpatialCorrelation()
matches = correlator.points_within_polygons(points, polygons, distance_km=1)
```

and simply convert `matches` into facts/citations before returning from
`run_query`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

try:  # Optional dependency – servers must handle ImportError gracefully.
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:  # pragma: no cover - geo stack may be absent in CI
    gpd = None  # type: ignore
    Point = None  # type: ignore


@dataclass
class FacilityPoint:
    """Simple representation of a point feature."""

    id: str
    latitude: float
    longitude: float
    properties: dict


@dataclass
class PolygonFeature:
    """Simple representation of a polygon feature."""

    id: str
    geometry_wkt: str
    properties: dict


class SpatialCorrelation:
    """Utility class that performs common spatial joins."""

    def __init__(self) -> None:
        if gpd is None:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "geopandas is required for SpatialCorrelation. Install via pip."
            )

    def points_within_polygons(
        self,
        points: Iterable[FacilityPoint],
        polygons: Iterable[PolygonFeature],
        *,
        distance_km: float = 1.0,
    ) -> List[dict]:
        """Return polygons that intersect a buffered point (in kilometres)."""

        points_gdf = self._points_to_gdf(points)
        polys_gdf = self._polygons_to_gdf(polygons)

        if points_gdf.empty or polys_gdf.empty:
            return []

        # Project to metric CRS for buffering and distance calculations.
        points_metric = points_gdf.to_crs(epsg=3857)
        buffer_m = distance_km * 1000
        buffered = points_metric.buffer(buffer_m)
        polygons_metric = polys_gdf.to_crs(epsg=3857)

        points_metric["geometry"] = buffered
        joined = gpd.sjoin(polygons_metric, points_metric, predicate="intersects")

        records: List[dict] = []
        for _, row in joined.iterrows():
            facility_props = row.get("properties_right", {}) or {}
            polygon_props = row.get("properties_left", {}) or {}
            records.append(
                {
                    "facility_id": row.get("id_right"),
                    "polygon_id": row.get("id_left"),
                    "distance_km": distance_km,
                    "facility": facility_props,
                    "polygon": polygon_props,
                }
            )
        return records

    @staticmethod
    def _points_to_gdf(points: Iterable[FacilityPoint]) -> "gpd.GeoDataFrame":
        data = []
        for point in points:
            data.append(
                {
                    "id": point.id,
                    "properties": point.properties,
                    "geometry": Point(point.longitude, point.latitude),
                }
            )
        return gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")

    @staticmethod
    def _polygons_to_gdf(polygons: Iterable[PolygonFeature]) -> "gpd.GeoDataFrame":
        data = []
        for poly in polygons:
            geometry = gpd.GeoSeries.from_wkt([poly.geometry_wkt], crs="EPSG:4326")[0]
            data.append({"id": poly.id, "properties": poly.properties, "geometry": geometry})
        return gpd.GeoDataFrame(data, geometry="geometry", crs="EPSG:4326")


__all__ = [
    "FacilityPoint",
    "PolygonFeature",
    "SpatialCorrelation",
]

