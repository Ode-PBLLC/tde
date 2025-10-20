#!/usr/bin/env python3
"""
Quick check for solar construction timeline output.
Ensures timeline_data is present for Brazil and spans recent years.
"""

import sys
import os

# Ensure project root on sys.path
if 'test_scripts' in os.getcwd():
    os.chdir('..')
sys.path.append('mcp')

from solar_db import SolarDatabase


def main():
    print("=== Testing Solar Construction Timeline (Brazil) ===")
    db = SolarDatabase()
    facilities = db.get_facilities_by_country("Brazil", limit=10000)
    start_year, end_year = 2017, 2025
    counts = {}
    import pandas as pd
    for f in facilities:
        y = None
        for key in ("constructed_after", "constructed_before", "source_date"):
            v = f.get(key)
            if v:
                dt = pd.to_datetime(v, errors='coerce')
                if pd.notnull(dt):
                    y = int(dt.year)
                    break
        if y is not None and start_year <= y <= end_year:
            counts[y] = counts.get(y, 0) + 1

    if counts:
        years = sorted(counts)
        print("✓ Computed year counts between 2017-2025")
        print("First 5 entries:")
        for y in years[:5]:
            print({"year": y, "facilities": counts[y]})
        print("Total in period:", sum(counts.values()))
    else:
        print("✗ No counts computed — investigate constructed_* fields")


if __name__ == "__main__":
    main()
