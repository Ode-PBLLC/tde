#!/usr/bin/env python3
"""
Compute municipalities most at risk of heat stress by overlapping
Brazilian municipality polygons with top-quintile heat-stress zones.

Method:
- Load top-quintile (Q5) heat zones via mcp.heat_stress_server
- Load all Brazilian municipalities via mcp.brazilian_municipalities_server
- Register both with mcp.geospatial_server and compute area overlap ratio
- Rank municipalities by overlap_ratio descending

Outputs a concise top list plus a CSV file in temp/ for further use.
"""

import json
import importlib.util
import sys
import types
from pathlib import Path

# Load MCP modules directly by path (mcp is not a package)
REPO_ROOT = Path(__file__).resolve().parents[1]

def _load_module(name: str, rel_path: str):
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# Provide a lightweight stub for fastmcp so servers can import
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

_install_fastmcp_stub()

heat = _load_module("heat_stress_server", "mcp/heat_stress_server.py")
mun = _load_module("brazilian_municipalities_server", "mcp/brazilian_municipalities_server.py")
geo = _load_module("geospatial_server", "mcp/geospatial_server.py")


def main():
    session_id = "heat_risk_run"

    # 1) Load heat zones (top quintile)
    heat_res = heat.GetHeatQuintilesForGeospatial(limit=20000)
    if heat_res.get("error"):
        raise SystemExit(
            f"Heat server error: {heat_res['error']}\n"
            "Note: ensure GeoPandas/Shapely installed and data present under data/heat_stress/preprocessed"
        )
    heat_entities = heat_res.get("entities", [])
    if not heat_entities:
        raise SystemExit("No heat-zone entities returned (top quintile).")

    # 2) Load municipalities (all)
    muni_res = mun.GetMunicipalitiesByFilter(limit=7000)
    if muni_res.get("error"):
        raise SystemExit(f"Municipality server error: {muni_res['error']}")
    municipalities = muni_res.get("municipalities", [])
    if not municipalities:
        raise SystemExit("No municipalities returned.")

    # 3) Register entities in geospatial session
    reg1 = geo.RegisterEntities(entity_type="heat_zone", entities=heat_entities, session_id=session_id)
    if reg1.get("error"):
        raise SystemExit(f"Geospatial registration error (heat): {reg1['error']}")
    reg2 = geo.RegisterEntities(entity_type="municipality", entities=municipalities, session_id=session_id)
    if reg2.get("error"):
        raise SystemExit(f"Geospatial registration error (municipalities): {reg2['error']}")

    # 4) Compute area overlap per municipality
    overlap = geo.ComputeAreaOverlapByEntityTypes(
        admin_entity_type="municipality",
        zone_entity_type="heat_zone",
        session_id=session_id,
    )
    if overlap.get("error"):
        raise SystemExit(f"Overlap computation error: {overlap['error']}")

    results = overlap.get("results", [])
    if not results:
        print("No overlaps found between municipalities and top-quintile heat zones.")
        return

    # 5) Build readable ranking using attached properties
    def name_state(props: dict):
        name = props.get("name") or props.get("muni_name") or "Unknown"
        state = props.get("state") or "?"
        return name, state

    ranked = []
    for r in results:
        props = r.get("properties") or {}
        name, state = name_state(props)
        ranked.append({
            "municipality": name,
            "state": state,
            "overlap_pct": round(float(r.get("overlap_ratio", 0.0)) * 100.0, 2),
            "overlap_km2": round(float(r.get("overlap_km2", 0.0)), 2),
            "total_area_km2": round(float(r.get("total_area_km2", 0.0)), 2),
            "population": int(props.get("population") or 0),
        })

    # Sort by overlap percentage desc, then by overlap area desc
    ranked.sort(key=lambda x: (x["overlap_pct"], x["overlap_km2"]), reverse=True)

    # Save CSV for inspection
    out_dir = Path("temp")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "municipalities_heat_risk_ranking.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("Municipality,State,Overlap (%),Overlap (km2),Total Area (km2),Population\n")
        for row in ranked:
            f.write(
                f"{row['municipality']},{row['state']},{row['overlap_pct']},{row['overlap_km2']},{row['total_area_km2']},{row['population']}\n"
            )

    # Print concise top 15
    top_n = 15
    print(f"Top {top_n} municipalities by heat-stress exposure (Q5 overlap %):")
    for i, row in enumerate(ranked[:top_n], start=1):
        print(
            f"{i:>2}. {row['municipality']} ({row['state']}): {row['overlap_pct']}% overlap, "
            f"{row['overlap_km2']} km² of {row['total_area_km2']} km²; pop {row['population']:,}"
        )
    print(f"\nFull ranking saved to: {csv_path}")


if __name__ == "__main__":
    main()
