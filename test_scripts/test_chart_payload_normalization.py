"""Tests for chart payload normalization in mcp_chat_redo."""

from pathlib import Path
import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "mcp" / "mcp_chat_redo.py"
spec = importlib.util.spec_from_file_location("mcp_chat_redo_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
QueryOrchestrator = module.QueryOrchestrator


def test_empty_categorical_payload_returns_none():
    payload = QueryOrchestrator._normalize_chart_payload(
        {"categories": [], "values": [], "summary": {}},
        "comparison",
    )
    assert payload is None


def test_categorical_payload_pairs_categories_and_values():
    payload = QueryOrchestrator._normalize_chart_payload(
        {"categories": ["Solar", "Wind"], "values": [70, 30]},
        "comparison",
    )
    assert payload == [
        {"category": "Solar", "value": 70},
        {"category": "Wind", "value": 30},
    ]


def test_time_series_payload_passthrough():
    values = [{"year": 2020, "value": 10}, {"year": 2021, "value": 15}]
    payload = QueryOrchestrator._normalize_chart_payload(
        {"values": values, "summary": {}},
        "time_series",
    )
    assert payload == values


if __name__ == "__main__":
    test_empty_categorical_payload_returns_none()
    test_categorical_payload_pairs_categories_and_values()
    test_time_series_payload_passthrough()
    print("All chart payload normalization tests passed.")
