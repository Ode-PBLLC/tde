#!/usr/bin/env python3
"""
Process TPI Graph data from raw Excel format to structured JSON.

This script transforms the messy tabular TPI data into a query-friendly format
with proper year indexing and scenario separation.

Input: data/lse_processed/tpi_graphs/tpi_graphs-sheet1.json
Output: data/lse_processed/tpi_graphs/tpi-pathways-structured.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_TPI_FILE = PROJECT_ROOT / "data/lse_processed/tpi_graphs/tpi_graphs-sheet1.json"
OUTPUT_FILE = PROJECT_ROOT / "data/lse_processed/tpi_graphs/tpi-pathways-structured.json"


def load_raw_data() -> Dict[str, Any]:
    """Load the raw TPI data."""
    with open(RAW_TPI_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_years_from_header(records: List[Dict]) -> List[int]:
    """
    Extract years from the first row where column names are 'unnamed-X'.

    The first row contains years like 2005, 2006, ..., 2035 in columns
    'unnamed-1' through 'unnamed-31'.
    """
    header_row = records[0]
    years = []

    # Start from unnamed-1 (skip mtco2e column which is unnamed-1's position)
    for i in range(1, 32):  # unnamed-1 through unnamed-31
        col_name = f"unnamed-{i}"
        year_value = header_row.get(col_name)
        if year_value is not None:
            years.append(int(year_value))

    return years


def extract_scenario_timeseries(
    records: List[Dict],
    scenario_index: int,
    years: List[int]
) -> List[Dict[str, Any]]:
    """
    Extract timeseries for a specific scenario.

    Args:
        records: All data records
        scenario_index: Row index of the scenario (1=historical, 2=NDC, etc.)
        years: List of years corresponding to columns

    Returns:
        List of {year, emissions_mtco2e} dicts
    """
    scenario_row = records[scenario_index]
    timeseries = []

    for i, year in enumerate(years):
        col_name = f"unnamed-{i+1}"
        emissions = scenario_row.get(col_name)

        if emissions is not None:
            timeseries.append({
                "year": year,
                "emissions_mtco2e": round(emissions, 2) if isinstance(emissions, float) else emissions
            })

    return timeseries


def calculate_reduction_percent(current: float, base: float) -> float:
    """Calculate percentage reduction from base year."""
    return round(((base - current) / base) * 100, 1)


def process_tpi_data() -> Dict[str, Any]:
    """Main processing function."""

    raw_data = load_raw_data()
    records = raw_data["records"]

    # Extract years from header row (index 0)
    years = extract_years_from_header(records)
    print(f"✓ Extracted {len(years)} years: {years[0]} to {years[-1]}")

    # Extract each scenario
    # Row indices based on raw data structure:
    # 0: Header (years)
    # 1: Historical emissions
    # 2: NDC targets
    # 3: High ambition
    # 4: 1.5C fair share
    # 5: 1.5C benchmark
    # 6-8: Text rows with assessment

    historical = extract_scenario_timeseries(records, 1, years)
    ndc_pathway = extract_scenario_timeseries(records, 2, years)
    high_ambition = extract_scenario_timeseries(records, 3, years)
    fair_share = extract_scenario_timeseries(records, 4, years)
    benchmark = extract_scenario_timeseries(records, 5, years)

    print(f"✓ Historical: {len(historical)} data points (2005-2023)")
    print(f"✓ NDC pathway: {len(ndc_pathway)} data points")
    print(f"✓ High ambition: {len(high_ambition)} data points")
    print(f"✓ Fair share: {len(fair_share)} data points")
    print(f"✓ Benchmark: {len(benchmark)} data points")

    # Get key values for calculations
    base_year_emissions = 931.64  # 2005
    latest_historical = 1201.89  # 2023

    # Find specific year values
    def get_year_value(timeseries: List[Dict], year: int) -> Optional[float]:
        for entry in timeseries:
            if entry["year"] == year:
                return entry["emissions_mtco2e"]
        return None

    ndc_2030 = get_year_value(ndc_pathway, 2030)
    ndc_2035_low = get_year_value(ndc_pathway, 2035)
    high_amb_2035 = get_year_value(high_ambition, 2035)
    fair_2030 = get_year_value(fair_share, 2030)
    fair_2035 = get_year_value(fair_share, 2035)
    bench_2030 = get_year_value(benchmark, 2030)
    bench_2035 = get_year_value(benchmark, 2035)

    print(f"\n✓ Key values:")
    print(f"  2030 NDC: {ndc_2030} MtCO2e")
    print(f"  2035 NDC (low): {ndc_2035_low} MtCO2e")
    print(f"  2035 High ambition: {high_amb_2035} MtCO2e")
    print(f"  2030 Gap vs fair share: {ndc_2030 - fair_2030:.2f} MtCO2e" if ndc_2030 and fair_2030 else "")
    print(f"  2030 Gap vs benchmark: {ndc_2030 - bench_2030:.2f} MtCO2e" if ndc_2030 and bench_2030 else "")

    # Extract alignment assessment text
    assessment_2030 = records[7].get("unnamed-12", "")
    assessment_2035 = records[8].get("unnamed-12", "")

    # Build structured output
    structured_data = {
        "module": "tpi_graphs",
        "group": "transition_pathways",
        "title": "TPI Emissions Pathways - Structured",
        "slug": "tpi-pathways-structured",
        "country": "Brazil",
        "source": "TPI (Transition Pathway Initiative)",
        "source_file": raw_data["source_file"],
        "description": "Brazil's emissions pathways comparing historical data, NDC targets, and Paris Agreement alignment scenarios",
        "last_updated": "2025-10-29",
        "scenarios": {
            "historical": {
                "name": "Historical emissions",
                "type": "observed",
                "start_year": years[0] if historical else None,
                "end_year": historical[-1]["year"] if historical else None,
                "data_quality": "Observed historical data",
                "timeseries": historical,
                "summary": {
                    "peak_year": 2023,
                    "peak_emissions": latest_historical,
                    "lowest_year": 2005,
                    "lowest_emissions": base_year_emissions,
                    "trend": "Generally increasing with fluctuations"
                }
            },
            "ndc_target_pathway": {
                "name": "NDC targets",
                "type": "official_commitment",
                "start_year": ndc_pathway[0]["year"] if ndc_pathway else None,
                "end_year": ndc_pathway[-1]["year"] if ndc_pathway else None,
                "description": "Brazil's official NDC trajectory showing linear decline from 2023 to 2035 target",
                "timeseries": ndc_pathway,
                "key_years": {
                    "2030": {
                        "emissions_mtco2e": ndc_2030,
                        "reduction_from_2005_percent": calculate_reduction_percent(ndc_2030, base_year_emissions) if ndc_2030 else None,
                        "note": "No formal 2030 target in current NDC; this is interpolated value"
                    },
                    "2035": {
                        "emissions_mtco2e": ndc_2035_low,
                        "reduction_from_2005_percent": calculate_reduction_percent(ndc_2035_low, base_year_emissions) if ndc_2035_low else None,
                        "reduction_from_2005_range": "59-67%",
                        "note": "Official NDC commitment: 59-67% below 2005 levels (this shows low end at 59%)"
                    }
                }
            },
            "high_ambition": {
                "name": "High ambition end of target range",
                "type": "ambitious_scenario",
                "start_year": high_ambition[0]["year"] if high_ambition else None,
                "end_year": high_ambition[-1]["year"] if high_ambition else None,
                "description": "More aggressive reduction pathway representing the high end (67%) of NDC target range",
                "timeseries": high_ambition,
                "key_years": {
                    "2035": {
                        "emissions_mtco2e": high_amb_2035,
                        "reduction_from_2005_percent": calculate_reduction_percent(high_amb_2035, base_year_emissions) if high_amb_2035 else None,
                        "note": "High ambition scenario (67% reduction from 2005)"
                    }
                }
            },
            "paris_1_5c_fair_share": {
                "name": "1.5°C fair share allocation",
                "type": "paris_benchmark",
                "start_year": fair_share[0]["year"] if fair_share else None,
                "end_year": fair_share[-1]["year"] if fair_share else None,
                "description": "Pathway representing Brazil's fair share contribution to limiting warming to 1.5°C",
                "timeseries": fair_share,
                "key_years": {
                    "2030": {
                        "emissions_mtco2e": fair_2030,
                        "reduction_from_2019_percent": calculate_reduction_percent(fair_2030, latest_historical) if fair_2030 else None,
                        "note": "Fair share pathway requires steep reductions"
                    },
                    "2035": {
                        "emissions_mtco2e": fair_2035,
                        "reduction_from_2005_percent": calculate_reduction_percent(fair_2035, base_year_emissions) if fair_2035 else None,
                        "note": "Most ambitious pathway shown"
                    }
                }
            },
            "paris_1_5c_benchmark": {
                "name": "1.5°C benchmark",
                "type": "paris_benchmark",
                "start_year": benchmark[0]["year"] if benchmark else None,
                "end_year": benchmark[-1]["year"] if benchmark else None,
                "description": "General 1.5°C alignment benchmark pathway for Brazil",
                "timeseries": benchmark,
                "key_years": {
                    "2030": {
                        "emissions_mtco2e": bench_2030,
                        "reduction_from_2019_percent": calculate_reduction_percent(bench_2030, latest_historical) if bench_2030 else None,
                        "note": "Moderate 1.5°C alignment pathway"
                    },
                    "2035": {
                        "emissions_mtco2e": bench_2035,
                        "reduction_from_2005_percent": calculate_reduction_percent(bench_2035, base_year_emissions) if bench_2035 else None,
                        "note": "Similar to high ambition NDC scenario"
                    }
                }
            }
        },
        "alignment_assessment": {
            "summary": "Brazil's NDC targets are not aligned with 1.5°C pathways for 2030, but 2035 targets show better alignment",
            "2030": {
                "ndc_value_mtco2e": ndc_2030,
                "fair_share_value_mtco2e": fair_2030,
                "benchmark_value_mtco2e": bench_2030,
                "aligned_with_fair_share": False,
                "aligned_with_benchmark": False,
                "fair_share_gap_mtco2e": round(ndc_2030 - fair_2030, 2) if ndc_2030 and fair_2030 else None,
                "fair_share_gap_percent": round(((ndc_2030 - fair_2030) / fair_2030) * 100, 1) if ndc_2030 and fair_2030 else None,
                "benchmark_gap_mtco2e": round(ndc_2030 - bench_2030, 2) if ndc_2030 and bench_2030 else None,
                "benchmark_gap_percent": round(((ndc_2030 - bench_2030) / bench_2030) * 100, 1) if ndc_2030 and bench_2030 else None,
                "assessment": assessment_2030,
                "note": "Brazil has no formal 2030 quantitative target in current NDC; this is interpolated value"
            },
            "2035": {
                "ndc_value_low_mtco2e": ndc_2035_low,
                "ndc_value_high_mtco2e": high_amb_2035,
                "fair_share_value_mtco2e": fair_2035,
                "benchmark_value_mtco2e": bench_2035,
                "aligned_with_fair_share": False,
                "aligned_with_benchmark_at_high_ambition": abs(high_amb_2035 - bench_2035) < 5 if high_amb_2035 and bench_2035 else False,
                "assessment": assessment_2035,
                "note": "High ambition NDC target aligns well with 1.5°C benchmark"
            }
        },
        "comparison_table": {
            "description": "Side-by-side comparison of all scenarios at key years",
            "years": [
                {
                    "year": 2005,
                    "historical": base_year_emissions,
                    "ndc": None,
                    "high_ambition": None,
                    "fair_share": None,
                    "benchmark": None
                },
                {
                    "year": 2023,
                    "historical": latest_historical,
                    "ndc": get_year_value(ndc_pathway, 2023),
                    "high_ambition": None,
                    "fair_share": get_year_value(fair_share, 2023),
                    "benchmark": get_year_value(benchmark, 2023)
                },
                {
                    "year": 2030,
                    "historical": None,
                    "ndc": ndc_2030,
                    "high_ambition": get_year_value(high_ambition, 2030),
                    "fair_share": fair_2030,
                    "benchmark": bench_2030
                },
                {
                    "year": 2035,
                    "historical": None,
                    "ndc": ndc_2035_low,
                    "high_ambition": high_amb_2035,
                    "fair_share": fair_2035,
                    "benchmark": bench_2035
                }
            ]
        },
        "metadata": {
            "units": "MtCO2e (million tonnes CO2 equivalent)",
            "base_year_for_targets": 2005,
            "base_year_emissions": base_year_emissions,
            "latest_historical_year": 2023,
            "latest_historical_emissions": latest_historical,
            "target_years": [2030, 2035],
            "data_source": "TPI (Transition Pathway Initiative)",
            "source_file": raw_data["source_file"],
            "processed_from": "tpi_graphs-sheet1.json",
            "citation": "NDC Align via TPI - Transition Pathway Initiative"
        }
    }

    return structured_data


def main():
    """Main execution."""
    print("=" * 60)
    print("TPI Data Processing Script")
    print("=" * 60)
    print()

    # Check input file exists
    if not RAW_TPI_FILE.exists():
        print(f"❌ Error: Input file not found: {RAW_TPI_FILE}")
        return 1

    print(f"📂 Input:  {RAW_TPI_FILE}")
    print(f"📂 Output: {OUTPUT_FILE}")
    print()

    # Process data
    try:
        structured_data = process_tpi_data()

        # Write output
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

        print()
        print(f"✅ Successfully created structured TPI dataset")
        print(f"   {OUTPUT_FILE}")
        print()
        print("📊 Summary:")
        print(f"   - Historical data: 2005-2023 ({len(structured_data['scenarios']['historical']['timeseries'])} points)")
        print(f"   - NDC pathway: {len(structured_data['scenarios']['ndc_target_pathway']['timeseries'])} points")
        print(f"   - High ambition: {len(structured_data['scenarios']['high_ambition']['timeseries'])} points")
        print(f"   - Fair share: {len(structured_data['scenarios']['paris_1_5c_fair_share']['timeseries'])} points")
        print(f"   - Benchmark: {len(structured_data['scenarios']['paris_1_5c_benchmark']['timeseries'])} points")
        print()

        return 0

    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
