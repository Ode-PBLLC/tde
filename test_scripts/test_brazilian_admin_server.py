#!/usr/bin/env python3
"""
Smoke tests for mcp/brazilian_admin_server.py

Checks minimal functionality without running the MCP server:
- Loads municipalities and states
- Boundary counts and metadata
- Retrieves polygons for specific states and municipalities
- Basic filters and bounds queries

Note: Provides a FastMCP stub so we can import the module directly.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_fastmcp_stub():
    if 'fastmcp' in sys.modules:
        return
    m = types.ModuleType('fastmcp')
    class FastMCP:  # minimal no-op stub
        def __init__(self, name: str):
            self.name = name
        def tool(self, *args, **kwargs):
            def _decorator(func):
                return func
            return _decorator
        def run(self):
            return None
    m.FastMCP = FastMCP
    sys.modules['fastmcp'] = m


def _load_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def main():
    _install_fastmcp_stub()
    mod = _load_module('brazilian_admin_server', 'mcp/brazilian_admin_server.py')

    # 1) Counts and metadata
    counts = mod.GetBoundaryCounts()
    assert not counts.get('error'), counts
    print('Counts:', counts)
    assert counts['counts']['states'] >= 26
    assert counts['counts']['municipalities'] >= 5500

    meta = mod.GetAdminDatasetMetadata()
    assert not meta.get('error'), meta
    print('Admin meta:', {k: meta[k] for k in ['boundary_types', 'counts']})

    # 2) State polygons by name and code
    states = mod.GetStateBoundaries(['São Paulo', 'AM'])
    assert not states.get('error'), states
    print('State results:', states['total_count'], 'items; missing:', states['not_found'])
    assert states['total_count'] >= 2
    assert all('geometry' in s and s['geometry'] for s in states['states'])

    # 3) Municipality filters
    mun = mod.GetMunicipalitiesByFilter(state='São Paulo', limit=5)
    assert not mun.get('error'), mun
    print('Municipalities (SP):', mun['total_count'])
    assert mun['total_count'] > 0
    assert all('geometry' in m and m['geometry'] for m in mun['municipalities'])

    # 4) Bounds query over a small São Paulo city area
    # Rough bounds around São Paulo metro
    bounds = mod.GetMunicipalitiesInBounds(
        north=-23.3, south=-24.1, east=-46.2, west=-47.2, limit=50
    )
    assert not bounds.get('error'), bounds
    print('Bounds query count:', bounds['total_count'])
    assert bounds['total_count'] > 0

    print('Brazilian admin server smoke tests passed.')


if __name__ == '__main__':
    main()

