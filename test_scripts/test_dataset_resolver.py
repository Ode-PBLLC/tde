"""
Quick, network-independent smoke test for dataset resolver.

Run: python test_scripts/test_dataset_resolver.py
Expected: For TZ-SAM tools (GetSolar*), resolves a non-empty URL from static/meta/datasets.json.
"""
from utils.dataset_resolver import resolve_dataset_url


def main():
    # TZ-SAM example
    ds_id, url = resolve_dataset_url("GetSolarCapacityByCountry", server_name="solar")
    print("TZ-SAM dataset_id:", ds_id)
    print("TZ-SAM source_url:", url)

    # GIST example (likely blank URL due to proprietary source)
    ds_id2, url2 = resolve_dataset_url("GetGistCompanies", server_name="gist")
    print("GIST dataset_id:", ds_id2)
    print("GIST source_url:", url2)

    # Unknown tool -> None
    ds_id3, url3 = resolve_dataset_url("GetUnknownTool", server_name="unknown")
    print("Unknown dataset_id:", ds_id3)
    print("Unknown source_url:", url3)


if __name__ == "__main__":
    main()
