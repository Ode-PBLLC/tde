#!/usr/bin/env python3
"""
Inspect GIST Excel datasets and print sheet names, columns, and samples for key sheets.
Looks for: data/gist/gist.xlsx
"""

import os
from pathlib import Path
import pandas as pd


KEY_SHEETS = [
    'EXSITU',
    'EXSITU_ASSET_DATA',
    'SCOPE_3_DATA',
    'DEFORESTATION',
    'BIODIVERSITY_PDF_DATA',
    'Data Dictionary',
]


def main():
    base = Path(__file__).resolve().parent.parent
    xls_path = base / 'data' / 'gist' / 'gist.xlsx'
    if not xls_path.exists():
        print('✗ GIST Excel not found at', xls_path)
        return

    print('✓ Loading GIST from:', xls_path)
    xl = pd.ExcelFile(xls_path)
    print('Sheets:', xl.sheet_names)

    for sheet in KEY_SHEETS:
        if sheet not in xl.sheet_names:
            print(f'\n— Sheet missing: {sheet}')
            continue
        print(f'\n== {sheet} ==')
        df = xl.parse(sheet)
        print('Columns:', list(df.columns))
        print('Head(5):')
        print(df.head(5))


if __name__ == '__main__':
    main()

