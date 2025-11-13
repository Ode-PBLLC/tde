# Data Directory

This directory contains datasets used by the MCP servers. Due to size constraints, these files are **not pushed to the repository** (excluded via `.gitignore`).

## Directory Overview

Total size: ~2.8GB

| Directory | Size | Server | Description |
|-----------|------|--------|-------------|
| `deforestation/` | 2.0GB | Deforestation | PRODES Deforestation Polygons Since 2008 - GeoJSON/Parquet polygons for deforestation analyses |
| `deforestation/deforestation_prodes_2024.parquet` | ~450MB | Deforestation | 2024-only PRODES polygons (server default) |
| `deforestation/deforestation_PRODES_recent.parquet` | ~120MB | Prototype | Centroids for 2022-2024 PRODES polygons (lean subset) |
| `tz-sam-q1-2025/` | 384MB | Solar Clay | TransitionZero Solar Asset Mapper (TZ-SAM) - Asset-level solar facilities with ML-powered footprints and capacity estimates |
| `heat_stress/` | 308MB | Extreme Heat | PlanetSapling Heat Index - Extreme heat quintiles from ERA5-Land measurements (5-year mean daily temps) |
| `wmo-ipcc-clim-adapt/` | 91MB | WMO/Climate | WMO State of the Climate in Latin America and the Caribbean 2024 |
| `ipcc-ch11/` | 79MB | GIST | IPCC WG1 Chapter 11: Weather and Climate Extreme Events in a Changing Climate |
| `mb-deforest/` | 68MB | MapBiomas | MapBiomas Annual Deforestation Report 2024 (RAD2024) - Brazil deforestation findings |
| `spa_index/` | 74MB | SPA | Science Panel for the Amazon (SPA) 2021 Report - Amazon Assessment Report from COP26 |
| `ipcc-ch12/` | 16MB | GIST | IPCC WG1 Chapter 12: Climate Change Information for Regional Impact and Risk Assessment |
| `gist/` | 11MB | GIST | GIST Impact Datasets - Company and asset-level sustainability metrics, biodiversity impacts |
| `brazilian_admin/` | 7.2MB | Brazilian Admin | Brazilian Administrative Boundaries (IBGE) - Geospatial data for municipalities and states |
| `wmo-lac/` | 1.9MB | WMO/Climate | WMO Latin America & Caribbean climate data |
| `lse/` | 1.1MB | LSE | NDC Align - Brazil's climate policy response and NDC implementation tracking |
| `lse_processed/` | 840KB | LSE | Processed NDC Align policy data |
| `solar_facilities.db` | 44MB | Solar | SQLite database of global solar facilities from TZ-SAM dataset |

## Regenerating Datasets

### Solar Database

```bash
python scripts/migrate_solar_to_db.py
```

Creates `data/solar_facilities.db` from source CSV files.

### Precomputed Overlays

These scripts generate spatial correlation data for faster queries:

```bash
# Deforestation overlays
python scripts/precompute_deforestation_overlays.py

# 2024-only PRODES subset (polygons)
python scripts/build_deforestation_subset.py --min-year 2024 --max-year 2024

# Extreme heat overlays
python scripts/precompute_extreme_heat_overlays.py

# Solar facility overlays
python scripts/precompute_solar_facility_overlays.py

# Lightweight centroids for prototypes (2022+)
python scripts/build_recent_deforestation_points.py --min-year 2022 --max-year 2024

# Yearly area aggregates (for MCP tools)
python scripts/precompute_deforestation_area_by_year.py
```

## Inspection Scripts

Located in `scripts/`, these help verify data integrity:

- `inspect_solar_db.py` - Check solar database contents
- `inspect_deforestation.py` - Verify deforestation data
- `inspect_heat_stress.py` - Check heat stress data
- `inspect_lse.py` - Inspect LSE policy data
- `inspect_gist.py` - Check GIST/IPCC data
- `inspect_municipalities.py` - Verify Brazilian admin data

## Development Setup

For local development, you'll need to either:

1. **Obtain the data** from another developer or data source
2. **Generate subsets** using the regeneration scripts above
3. **Run with missing data** - servers will warn but some functionality will be degraded

## Notes

- The `test_write.tmp` file is a leftover test file and can be deleted
- Large directories like `deforestation/` should only be loaded if you need spatial correlation features
- Most queries work with just the core datasets (~60MB total)
