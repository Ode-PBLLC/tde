"""In-memory cache for solar facilities used by geospatial servers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from mcp.geospatial_bridge import FacilityPoint
from mcp.solar_db import SolarDatabase


@dataclass(frozen=True)
class SolarFacility:
    """Canonical in-memory representation of a solar facility."""

    cluster_id: str
    latitude: float
    longitude: float
    country: str
    capacity_mw: Optional[float]
    constructed_before: Optional[str]
    constructed_after: Optional[str]
    source: Optional[str]
    source_date: Optional[str]

    @classmethod
    def from_row(cls, row: Dict[str, object]) -> "SolarFacility":
        return cls(
            cluster_id=str(row.get("cluster_id")),
            latitude=float(row.get("latitude")),
            longitude=float(row.get("longitude")),
            country=str(row.get("country")),
            capacity_mw=_coerce_optional_float(row.get("capacity_mw")),
            constructed_before=_coerce_optional_str(row.get("constructed_before")),
            constructed_after=_coerce_optional_str(row.get("constructed_after")),
            source=_coerce_optional_str(row.get("source")),
            source_date=_coerce_optional_str(row.get("source_date")),
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "country": self.country,
            "capacity_mw": self.capacity_mw,
            "constructed_before": self.constructed_before,
            "constructed_after": self.constructed_after,
            "source": self.source,
            "source_date": self.source_date,
        }

    def to_facility_point(self) -> FacilityPoint:
        return FacilityPoint(
            id=self.cluster_id,
            latitude=self.latitude,
            longitude=self.longitude,
            properties=self.as_dict(),
        )


def _coerce_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


class SolarFacilityProvider:
    """Loads the solar facility database into memory for fast geospatial access."""

    def __init__(self, *, preload_limit: Optional[int] = None) -> None:
        self.db = SolarDatabase()
        rows = self.db.get_all_facilities(limit=preload_limit)
        self._facilities: List[SolarFacility] = [SolarFacility.from_row(row) for row in rows]
        self._by_id: Dict[str, SolarFacility] = {facility.cluster_id: facility for facility in self._facilities}

    def all_facilities(self) -> List[SolarFacility]:
        return list(self._facilities)

    def get(self, cluster_id: str) -> Optional[SolarFacility]:
        return self._by_id.get(cluster_id)

    def facilities_in_radius(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        *,
        limit: Optional[int] = None,
    ) -> List[SolarFacility]:
        matches: List[SolarFacility] = []
        for facility in self._facilities:
            distance = _haversine_km(latitude, longitude, facility.latitude, facility.longitude)
            if distance <= radius_km:
                matches.append(facility)
                if limit is not None and len(matches) >= limit:
                    break
        return matches

    def facilities_to_points(self, facilities: Iterable[SolarFacility]) -> List[FacilityPoint]:
        return [facility.to_facility_point() for facility in facilities]

    def facilities_by_country(self, country: str, *, limit: Optional[int] = None) -> List[SolarFacility]:
        matches = [facility for facility in self._facilities if facility.country.lower() == country.lower()]
        if limit is not None:
            return matches[:limit]
        return matches

