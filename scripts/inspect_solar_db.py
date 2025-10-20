#!/usr/bin/env python3
"""
Inspect the Solar SQLite database and print schema + sample rows.

Looks for DB at:
- data/solar_facilities.db
- ./solar_facilities.db

Prints:
- Tables present
- Columns for key tables (solar_facilities, dataset_metadata)
- Sample rows from solar_facilities
- Quick aggregates by country (top 5)
"""

import os
import sqlite3


def find_db_path() -> str | None:
    candidates = [
        os.path.join('data', 'solar_facilities.db'),
        'solar_facilities.db',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    db_path = find_db_path()
    if not db_path:
        print('✗ No SQLite DB found at data/solar_facilities.db or ./solar_facilities.db')
        return

    print(f'✓ Using DB: {db_path}')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print('Tables:', tables)

    def show_table_info(name: str):
        try:
            cur.execute(f"PRAGMA table_info({name})")
            cols = cur.fetchall()
            print(f'\nColumns in {name}:')
            for c in cols:
                print('  -', c[1], c[2])
        except Exception as e:
            print(f'  (could not inspect {name}: {e})')

    for t in ['solar_facilities', 'dataset_metadata']:
        show_table_info(t)

    # Sample rows
    try:
        print('\nSample rows from solar_facilities:')
        cur.execute("SELECT cluster_id, country, latitude, longitude, capacity_mw, source, source_date FROM solar_facilities LIMIT 5")
        for row in cur.fetchall():
            print('  ', row)
    except Exception as e:
        print('  (no sample rows available:', e, ')')

    # Quick aggregates
    try:
        print('\nTop 5 countries by facility count:')
        cur.execute("SELECT country, COUNT(*) as cnt FROM solar_facilities GROUP BY country ORDER BY cnt DESC LIMIT 5")
        for row in cur.fetchall():
            print('  ', row)
    except Exception as e:
        print('  (aggregate failed:', e, ')')

    conn.close()


if __name__ == '__main__':
    main()

