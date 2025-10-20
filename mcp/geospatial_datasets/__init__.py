"""Shared in-memory geospatial datasets used by v2 MCP servers."""

from .solar import SolarFacilityProvider
from .deforestation import DeforestationPolygonProvider

__all__ = [
    "SolarFacilityProvider",
    "DeforestationPolygonProvider",
]
