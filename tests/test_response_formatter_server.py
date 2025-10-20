from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.response_formatter_server import _insert_inline_citations


def test_insert_inline_citations_skips_existing_numbers() -> None:
    text = "Evidence already cited.^1^"
    citation_registry = {
        "module_citations": {"text_module_0": [1]},
        "citations": {},
    }

    result = _insert_inline_citations(text, "text_module_0", citation_registry)

    assert result == text


def test_insert_inline_citations_appends_missing_only_once() -> None:
    text = "Evidence needs support."
    citation_registry = {
        "module_citations": {"text_module_0": [1, 2]},
        "citations": {},
    }

    result = _insert_inline_citations(text, "text_module_0", citation_registry)

    assert result.endswith("^1^ ^2^")
    # Ensure we only appended once
    assert result.count("^1^") == 1
    assert result.count("^2^") == 1
