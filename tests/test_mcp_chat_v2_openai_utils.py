from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.mcp_chat_v2 import _extract_openai_text


def test_extract_openai_text_uses_output_text_when_available() -> None:
    response = SimpleNamespace(output_text="Primary text", output=[])

    assert _extract_openai_text(response) == "Primary text"


def test_extract_openai_text_collects_from_nested_content() -> None:
    content_block = {"type": "output_text", "text": "Nested"}
    response = SimpleNamespace(output_text="", output=[SimpleNamespace(content=[content_block])])

    assert _extract_openai_text(response) == "Nested"
