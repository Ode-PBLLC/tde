from __future__ import annotations

import unittest

import importlib.util
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if "mcp" not in sys.modules:
    package = types.ModuleType("mcp")
    package.__path__ = [str(ROOT / "mcp")]
    sys.modules["mcp"] = package

contract_validation = _load_module(
    "mcp.contract_validation", "mcp/contract_validation.py"
)

ContractValidationError = contract_validation.ContractValidationError
validate_final_response = contract_validation.validate_final_response
validate_run_query_response = contract_validation.validate_run_query_response


def _sample_run_query_response() -> dict:
    return {
        "server": "solar",
        "query": "Where are solar assets near deforestation?",
        "facts": [
            {
                "id": "fact_1",
                "text": "Three facilities fall within 1 km of recent deforestation polygons.",
                "citation_id": "cite_1",
                "kind": "text",
                "metadata": {
                    "facility_ids": ["BR001", "BR002", "BR003"],
                },
            }
        ],
        "citations": [
            {
                "id": "cite_1",
                "server": "solar",
                "tool": "run_query",
                "title": "TransitionZero Solar Asset Mapper",
                "source_type": "Dataset",
                "description": "Brazilian facility locations and proximity analysis",
                "url": "https://example.com/solar",
            }
        ],
        "artifacts": [
            {
                "id": "map_1",
                "type": "map",
                "title": "Solar assets near deforestation",
                "geojson_url": "https://example.com/map.geojson",
            }
        ],
        "messages": [],
        "kg": {"nodes": [], "edges": []},
        "next_actions": [],
    }


def _sample_final_response() -> dict:
    return {
        "query": "Where are solar assets near deforestation?",
        "modules": [
            {
                "type": "text",
                "heading": "Summary",
                "texts": [
                    "Three solar facilities fall within one kilometre of recent deforestation activity ^1^."
                ],
            },
            {
                "type": "map",
                "mapType": "geojson_url",
                "geojson_url": "https://example.com/map.geojson",
                "viewState": {"center": [-54.0, -14.0], "zoom": 6},
            },
            {
                "type": "numbered_citation_table",
                "heading": "References",
                "columns": [
                    "#",
                    "Source",
                    "ID/Tool",
                    "Type",
                    "Description",
                    "SourceURL",
                ],
                "rows": [
                    [
                        "1",
                        "TransitionZero Solar Asset Mapper",
                        "run_query",
                        "Dataset",
                        "Brazilian facility locations and proximity analysis",
                        "https://example.com/solar",
                    ]
                ],
            },
        ],
        "metadata": {
            "modules_count": 3,
            "has_maps": True,
            "has_charts": False,
            "has_tables": True,
        },
        "kg_context": {
            "nodes": [
                {"id": "concept_solar", "label": "solar energy", "type": "Concept"}
            ],
            "edges": [],
        },
    }


class ContractValidationTests(unittest.TestCase):
    def test_validate_run_query_response_accepts_valid_payload(self) -> None:
        validate_run_query_response(_sample_run_query_response())

    def test_validate_run_query_response_rejects_missing_citations(self) -> None:
        payload = _sample_run_query_response()
        payload["citations"] = []
        with self.assertRaises(ContractValidationError):
            validate_run_query_response(payload)

    def test_validate_final_response_accepts_valid_payload(self) -> None:
        validate_final_response(_sample_final_response())

    def test_validate_final_response_requires_citation_table(self) -> None:
        payload = _sample_final_response()
        payload["modules"] = payload["modules"][:-1]
        payload["metadata"]["modules_count"] = len(payload["modules"])
        with self.assertRaises(ContractValidationError):
            validate_final_response(payload)

    def test_validate_final_response_requires_geojson_urls(self) -> None:
        payload = _sample_final_response()
        payload["modules"][1]["geojson_url"] = ""
        with self.assertRaises(ContractValidationError):
            validate_final_response(payload)


if __name__ == "__main__":
    unittest.main()
