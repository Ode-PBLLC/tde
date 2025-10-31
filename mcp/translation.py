"""
Translation utilities using OpenAI API.

Provides helper to translate human-facing fields in response modules while
preserving numbers, IDs, URLs, and citation markers.
"""

from typing import List, Dict, Any, Tuple
import os

try:
    from openai import AsyncOpenAI
    _HAS_OPENAI = True
except Exception:
    # Allow importing module without OpenAI installed; callers should handle fallback
    _HAS_OPENAI = False


def _collect_translatable_strings(modules: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[str, int, List[str]]]]:
    """
    Walk modules and collect strings to translate.

    Returns a tuple of:
    - strings: list of strings to translate in order
    - locations: list of (module_type, module_index, path_parts) to write back
      where path_parts is a list describing how to reach the field:
        e.g., ["heading"], ["texts", 0], ["legend", "title"], ["legend", "items", i, "label"],
              ["options", "plugins", "title", "text"], ["columns", j]
    """
    strings: List[str] = []
    locations: List[Tuple[str, int, List[Any]]] = []

    def add(module_type: str, idx: int, path: List[Any], value: Any):
        if isinstance(value, str) and value.strip():
            strings.append(value)
            locations.append((module_type, idx, path))

    for i, m in enumerate(modules or []):
        mtype = str(m.get("type", ""))
        # Common: heading
        add(mtype, i, ["heading"], m.get("heading"))

        if mtype == "text":
            texts = m.get("texts") or []
            for j, t in enumerate(texts):
                add(mtype, i, ["texts", j], t)
        elif mtype == "map":
            legend = (m.get("legend") or {})
            add(mtype, i, ["legend", "title"], legend.get("title"))
            items = legend.get("items") or []
            for j, it in enumerate(items):
                add(mtype, i, ["legend", "items", j, "label"], (it or {}).get("label"))
        elif mtype == "chart":
            # Translate heading and Chart.js title
            opts = (m.get("options") or {})
            plugins = (opts.get("plugins") or {})
            title = (plugins.get("title") or {})
            add(mtype, i, ["options", "plugins", "title", "text"], title.get("text"))
        elif mtype == "table":
            add(mtype, i, ["heading"], m.get("heading"))
            cols = m.get("columns") or []
            for j, col in enumerate(cols):
                add(mtype, i, ["columns", j], col)
        # Do not translate rows/data to avoid altering proper nouns and values

    return strings, locations


async def translate_strings(strings: List[str], target_lang: str) -> List[str]:
    """Translate a list of strings into the target language using OpenAI.

    Returns translated strings in the same order. If translation is unavailable
    (no API key/client error), returns the original strings.
    """
    if not strings:
        return []

    if not _HAS_OPENAI:
        return strings

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return strings

    client = AsyncOpenAI(api_key=api_key)

    # Build a compact JSON payload to preserve order and enable deterministic parsing
    import json
    payload = json.dumps(strings, ensure_ascii=False)
    system = (
        "You are a precise translator. Translate the provided JSON array of strings "
        f"into {('Portuguese' if target_lang.lower().startswith('pt') else target_lang)}. "
        "Output ONLY a valid JSON array with the translated strings in the exact same order. "
        "Preserve citation markers like ^1^ or [CITE_1], do not translate URLs, IDs, numbers, or currency symbols. "
        "Keep markdown and punctuation."
    )
    user = (
        "Translate these strings. Return a JSON array only.\n\n" + payload
    )

    try:
        resp = await client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content if resp.choices else None
        if not content:
            return strings
        # Some models may wrap JSON in code fences; strip if present
        content_str = content.strip()
        if content_str.startswith("```"):
            content_str = content_str.strip("`")
            # Remove a potential language hint
            if "[" in content_str:
                content_str = content_str[content_str.index("[") : ]
        translated = json.loads(content_str)
        if not isinstance(translated, list):
            return strings
        # Ensure lengths match
        if len(translated) != len(strings):
            return strings
        # Ensure all are strings
        return [str(s) for s in translated]
    except Exception:
        return strings


async def translate_modules(modules: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
    """Translate human-readable fields in modules to target_lang.

    Returns a new modules list with translations applied. On failure, returns original modules.
    """
    if not modules or not target_lang:
        return modules

    strings, locations = _collect_translatable_strings(modules)
    if not strings:
        return modules

    translated = await translate_strings(strings, target_lang)
    if not translated or len(translated) != len(strings):
        return modules

    # Apply translations back by walking locations
    out = [dict(m) for m in modules]
    si = 0
    for (mtype, idx, path) in locations:
        try:
            ref = out[idx]
            # Walk/create nested structure as needed
            parent = ref
            for k in path[:-1]:
                if isinstance(k, int):
                    # list index
                    if not isinstance(parent, list) or k >= len(parent):
                        raise IndexError
                    parent = parent[k]
                else:
                    if k not in parent or not isinstance(parent[k], (dict, list)):
                        # create a dict to place value sensibly; but if structure is missing, skip
                        raise KeyError
                    parent = parent[k]
            last = path[-1]
            if isinstance(last, int):
                if isinstance(parent, list) and last < len(parent):
                    parent[last] = translated[si]
            else:
                if isinstance(parent, dict):
                    parent[last] = translated[si]
        except Exception:
            # Skip malformed paths gracefully
            pass
        finally:
            si += 1

    return out
