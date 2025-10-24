# Data Directory

This directory contains datasets used by the MCP servers. Due to size constraints, these files are **not pushed to the repository** (excluded via `.gitignore`).

## Directory Overview

Total size: ~2.8GB

| Directory | Size | Server | Description |
|-----------|------|--------|-------------|
| `deforestation/` | 2.0GB | Deforestation | Polygon data for deforestation analysis |
| `tz-sam-q1-2025/` | 384MB | Solar | TZ-SAM analysis outputs and polygons |
| `heat_stress/` | 308MB | Extreme Heat | Heat index and stress data |
| `wmo-ipcc-clim-adapt/` | 91MB | WMO/Climate | WMO climate adaptation index |
| `ipcc-ch11/` | 79MB | GIST | IPCC Chapter 11 processed data |
| `mb-deforest/` | 68MB | MapBiomas Deforestation | MapBiomas deforestation index |
| `spa_index/` | 74MB | SPA | Sectoral Policy Analysis index |
| `ipcc-ch12/` | 16MB | GIST | IPCC Chapter 12 processed data |
| `gist/` | 11MB | GIST | General index and search data |
| `brazilian_admin/` | 7.2MB | Brazilian Admin | Administrative boundaries |
| `wmo-lac/` | 1.9MB | WMO/Climate | WMO Latin America & Caribbean data |
| `lse/` | 1.1MB | LSE | London School of Economics policy data |
| `lse_processed/` | 840KB | LSE | Processed LSE policy data |
| `solar_facilities.db` | 44MB | Solar | SQLite database of solar facilities worldwide |

## Required vs Optional

### Core Datasets (Servers will fail without these)

These are required for the respective servers to function:

- **`solar_facilities.db`** - Solar server (regenerate with `scripts/migrate_solar_to_db.py`)
- **`data/lse/`** - LSE server (climate policy data)
- **`data/gist/`** - GIST server (IPCC data)
- **`data/brazilian_admin/`** - Brazilian admin boundaries server

### Optional/Large Datasets

These enhance server capabilities but aren't strictly required for basic operation:

- **`deforestation/`** (2GB) - Used for spatial correlation overlays
- **`heat_stress/`** (308MB) - Extreme heat analysis
- **`tz-sam-q1-2025/`** (384MB) - Transition zone analysis outputs

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

# Extreme heat overlays
python scripts/precompute_extreme_heat_overlays.py

# Solar facility overlays
python scripts/precompute_solar_facility_overlays.py
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
