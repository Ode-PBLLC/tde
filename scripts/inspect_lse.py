#!/usr/bin/env python3
"""
Inspect LSE climate policy Excel files and print sheet names + sample columns.
Looks in: data/lse/
"""

import os
from pathlib import Path
import pandas as pd


def main():
    base = Path(__file__).resolve().parent.parent
    lse_dir = base / 'data' / 'lse'
    if not lse_dir.exists():
        print('✗ LSE directory not found at', lse_dir)
        return

    files = [f for f in lse_dir.iterdir() if f.suffix.lower() in ('.xlsx', '.xls')]
    if not files:
        print('✗ No Excel files found in', lse_dir)
        return

    print('✓ Found LSE files:')
    for f in files:
        print('  -', f.name)

    # Inspect first 2 sheets of each file
    for f in files:
        try:
            print(f'\n== {f.name} ==')
            xl = pd.ExcelFile(f)
            print('Sheets:', xl.sheet_names)
            for sheet in xl.sheet_names[:2]:
                print(f'  -- {sheet} --')
                df = xl.parse(sheet)
                print('  Columns:', list(df.columns))
                print('  Head(3):')
                print(df.head(3))
        except Exception as e:
            print('  (error reading file):', e)


if __name__ == '__main__':
    main()

