import asyncio
import os
import sys
from pathlib import Path

# Ensure the project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)


def test_detect_portuguese():
    from utils.language import detect_portuguese
    assert detect_portuguese("Mostre políticas de energia no Brasil") is True
    assert detect_portuguese("Show energy policies in Brazil") is False


async def test_should_use_portuguese_fallback():
    from utils.language import should_respond_in_portuguese

    # Ensure the test exercises the heuristic path by clearing the API key.
    prior_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        assert await should_respond_in_portuguese("Mostre políticas climáticas do Brasil.") is True
        assert await should_respond_in_portuguese("Please answer in Brazilian Portuguese about energy policy.") is True
        assert await should_respond_in_portuguese("Give me a quick English summary of solar targets.") is False
    finally:
        if prior_key is not None:
            os.environ["OPENAI_API_KEY"] = prior_key


async def test_translate_modules_monkeypatch():
    # Monkeypatch translate_strings to avoid network and verify mapping
    import mcp.translation as tr

    async def fake_translate(strings, lang):
        return [f"[PT]{s}" for s in strings]

    tr.translate_strings = fake_translate  # type: ignore

    modules = [
        {
            "type": "text",
            "heading": "Overview",
            "texts": ["Solar capacity has increased."]
        },
        {
            "type": "map",
            "legend": {
                "title": "Spatial Map",
                "items": [{"label": "Brazil"}, {"label": "China"}]
            }
        },
        {
            "type": "chart",
            "heading": "Capacity by Year",
            "options": {"plugins": {"title": {"text": "Capacity Trend"}}}
        },
        {
            "type": "table",
            "heading": "Summary",
            "columns": ["Country", "Capacity (MW)"]
        },
    ]

    out = await tr.translate_modules(modules, "pt")

    # Check a few key fields got "translated"
    assert out[0]["heading"].startswith("[PT]")
    assert out[0]["texts"][0].startswith("[PT]")
    assert out[1]["legend"]["title"].startswith("[PT]")
    assert out[1]["legend"]["items"][0]["label"].startswith("[PT]")
    assert out[2]["options"]["plugins"]["title"]["text"].startswith("[PT]")
    assert out[3]["columns"][0].startswith("[PT]")


def main():
    test_detect_portuguese()
    asyncio.run(test_should_use_portuguese_fallback())
    asyncio.run(test_translate_modules_monkeypatch())
    print("OK: translation plumbing and language detection pass")


if __name__ == "__main__":
    main()
