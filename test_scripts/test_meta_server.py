#!/usr/bin/env python3
"""Quick smoke tests for the Meta MCP server tools.

Runs without network or external services; imports tool functions directly.
"""

import os
import sys
from typing import Any, Dict


def _project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def main() -> None:
    # Ensure we can import from mcp/
    proj_root = _project_root()
    mcp_dir = os.path.join(proj_root, "mcp")
    sys.path.insert(0, mcp_dir)

    try:
        from meta_server import (
            _get_meta_summary,
            _get_repo_meta,
            _get_datasets_meta,
            _get_organizations_meta,
            _get_project_links,
        )
    except Exception as e:
        print(f"✗ Failed to import meta_server tools: {e}")
        sys.exit(1)

    # Run tools and print compact summaries
    def _pp(title: str, data: Dict[str, Any]) -> None:
        print(f"\n=== {title} ===")
        keys = list(data.keys())
        print(f"keys: {keys[:6]}{' ...' if len(keys) > 6 else ''}")

    try:
        _pp("MetaSummary", _get_meta_summary())
        _pp("RepoMeta", _get_repo_meta())
        _pp("DatasetsMeta", _get_datasets_meta())
        _pp("OrganizationsMeta", _get_organizations_meta())
        _pp("ProjectLinks", _get_project_links())
        print("\n✓ Meta server tools basic checks passed")
    except Exception as e:
        print(f"✗ Meta server tools execution error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # If invoked from within test_scripts, jump to project root for stable paths
    if os.path.basename(os.getcwd()) == "test_scripts":
        os.chdir("..")
    main()
