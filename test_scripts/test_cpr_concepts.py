#!/usr/bin/env python3
"""Inspect CPR concept candidates for an arbitrary query."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from textwrap import shorten
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_MODULE = REPO_ROOT / "mcp" / "servers_v2" / "cpr_server_v2.py"


def _load_tools_module() -> Any:
    """Import the CPR tools module once paths are configured."""

    import importlib

    return importlib.import_module("mcp.servers_v2.cpr_tools")


def _load_cpr_server() -> Any:
    """Load `CPRServerV2` in the same import mode as when run via FastMCP."""

    if not SERVER_MODULE.exists():
        raise FileNotFoundError(f"Unable to locate {SERVER_MODULE}")

    module_globals: Dict[str, Any] = runpy.run_path(str(SERVER_MODULE))
    try:
        return module_globals["CPRServerV2"]()
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("CPRServerV2 not exported from module") from exc


def _resolve_concepts(
    server: Any,
    query: str,
    limit: int,
    passage_limit: int,
) -> Dict[str, Any]:
    """Return concept candidates and optional passage payloads for `query`."""

    candidates: List[Dict[str, Any]] = server._candidate_concepts(query, limit=limit)  # type: ignore[attr-defined]

    if passage_limit <= 0:
        return {
            "query": query,
            "limit": limit,
            "concepts": candidates,
        }

    tools = _load_tools_module()
    enriched: List[Dict[str, Any]] = []
    for concept in candidates:
        label = concept.get("label")
        if not label:
            enriched.append({**concept, "passages": []})
            continue

        passages = tools.get_passages_mentioning_concept(label, limit=passage_limit)
        formatted_passages: List[Dict[str, Any]] = []
        for passage in passages:
            metadata = (passage.get("metadata") or {}) if isinstance(passage, dict) else {}
            text = passage.get("text") if isinstance(passage, dict) else ""
            snippet = text.split(":", 1)[0].strip() if text else ""

            formatted_passages.append(
                {
                    "document_id": metadata.get("document_id"),
                    "passage_id": metadata.get("passage_id"),
                    "text": text,
                    "snippet": snippet,
                }
            )

        enriched.append({**concept, "passages": formatted_passages})

    return {
        "query": query,
        "limit": limit,
        "passage_limit": passage_limit,
        "concepts": enriched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect which CPR concepts the server would chase for a query.",
    )
    parser.add_argument(
        "query",
        help="Natural language question to analyse",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of candidate concepts to display (default: 10)",
    )
    parser.add_argument(
        "--passage-limit",
        type=int,
        default=0,
        help="If > 0, include up to this many passages per concept",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of pretty text",
    )

    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    server = _load_cpr_server()

    payload = _resolve_concepts(server, args.query, args.limit, args.passage_limit)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    concepts = payload.get("concepts", [])
    print(f"Query        : {payload['query']}")
    print(f"Limit        : {payload['limit']}")
    print(f"Concept count: {len(concepts)}")
    for idx, concept in enumerate(concepts, start=1):
        label = concept.get("label") or "<missing label>"
        wikibase_id = concept.get("wikibase_id") or "<unknown>"
        print(f" {idx:>2}. {label} (wikibase: {wikibase_id})")
        passages = concept.get("passages") or []
        if not passages:
            continue
        for p_idx, passage in enumerate(passages, start=1):
            doc_id = passage.get("document_id") or "<unknown>"
            passage_id = passage.get("passage_id") or "<unknown>"
            snippet = passage.get("snippet") or passage.get("text") or ""
            display = shorten(snippet, width=160, placeholder="…") if snippet else "<no text>"
            print(f"      - [{p_idx}] doc={doc_id} passage={passage_id}: {display}")


if __name__ == "__main__":
    main()
