"""
Meta Server (MCP)

Provides project metadata via MCP tools: repo, datasets, orgs, and links.

Design goals:
- Keep sources primarily static in `static/meta/*.json` to avoid drift.
- Add small, safe dynamic fields (git commit/branch when available).
- Avoid secrets and large data; aggregate only.
"""

import ast
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP


mcp = FastMCP("meta-server")


# Paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_STATIC_META_DIR = os.path.join(_PROJECT_ROOT, "static", "meta")
_EXCLUDED_SUMMARY_DIRS = {
    ".git",
    "__pycache__",
    "data",
    "logs",
    "static",
    "temp",
}
_EXCLUDED_SUMMARY_FILES = {"__init__.py"}
_MAX_FUNCTION_SUMMARIES = 3
_MAX_CLASS_SUMMARIES = 2
_MAX_METHOD_NAMES = 4
_SOFTEN_PATTERNS = [
    ("API server", "web service"),
    ("GeoJSON", "map data"),
    ("API", "service"),
    ("JSON", "structured data"),
    ("MCP client", "helper process"),
    ("MCP", "assistant"),
]


def _read_json(path: str) -> Dict[str, Any]:
    """Read JSON file content; return {} if missing or invalid.

    Args:
        path: Absolute file path.
    Returns:
        Parsed JSON dict or empty dict on error.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {"items": data} if isinstance(data, list) else {}
    except Exception:
        return {}


def _git_info(project_root: str) -> Dict[str, Optional[str]]:
    """Best-effort git info without running subprocesses.

    Reads `.git/HEAD` and referenced ref file if available.
    Returns nulls if not in a git checkout.
    """
    try:
        git_dir = os.path.join(project_root, ".git")
        head_path = os.path.join(git_dir, "HEAD")
        if not os.path.exists(head_path):
            return {"branch": None, "commit": None}

        with open(head_path, "r", encoding="utf-8") as f:
            head = f.read().strip()

        # HEAD can be like: ref: refs/heads/main
        if head.startswith("ref: "):
            ref = head.split(": ", 1)[1]
            branch = os.path.basename(ref)
            ref_path = os.path.join(git_dir, ref)
            commit = None
            if os.path.exists(ref_path):
                with open(ref_path, "r", encoding="utf-8") as rf:
                    commit = rf.read().strip() or None
            return {"branch": branch, "commit": commit}

        # Detached HEAD with commit hash directly
        return {"branch": None, "commit": head or None}
    except Exception:
        return {"branch": None, "commit": None}


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _get_meta_summary() -> Dict[str, Any]:
    """Implementation: summary of available meta sections and basic stats."""
    datasets = _read_json(os.path.join(_STATIC_META_DIR, "datasets.json"))
    repo = _read_json(os.path.join(_STATIC_META_DIR, "repo.json"))
    orgs = _read_json(os.path.join(_STATIC_META_DIR, "orgs.json"))
    links = _read_json(os.path.join(_STATIC_META_DIR, "links.json"))

    git = _git_info(_PROJECT_ROOT)

    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "sections": {
            "repo": {
                "present": bool(repo),
                "branch": git.get("branch"),
                "commit": git.get("commit"),
            },
            "datasets": {
                "present": bool(datasets),
                "count": len(datasets.get("items", [])) if isinstance(datasets.get("items"), list) else 0,
            },
            "orgs": {
                "present": bool(orgs),
                "count": len(orgs.get("items", [])) if isinstance(orgs.get("items"), list) else 0,
            },
            "links": {
                "present": bool(links),
                "keys": list(links.keys()),
            },
        },
        "notes": "Static files live in static/meta/*.json. Keep secrets out.",
    }


@mcp.tool()
def GetMetaSummary() -> Dict[str, Any]:
    """Summary of available meta sections and basic stats."""
    return _get_meta_summary()


def _get_repo_meta() -> Dict[str, Any]:
    """Implementation: project repository metadata (safe subset)."""
    repo = _read_json(os.path.join(_STATIC_META_DIR, "repo.json"))
    git = _git_info(_PROJECT_ROOT)

    # Safe runtime facts
    safe_runtime = {
        "api_default_port": 8098,  # per repository guidelines
        "kg_default_port": 8100,
    }

    repo_out = {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "git": git,
        "runtime": safe_runtime,
    }
    # Merge static fields (static takes precedence for display text/links)
    repo_out.update({k: v for k, v in repo.items() if k not in repo_out})
    return repo_out


@mcp.tool()
def GetRepoMeta() -> Dict[str, Any]:
    """Project repository metadata (safe subset)."""
    return _get_repo_meta()


def _get_datasets_meta() -> Dict[str, Any]:
    """Implementation: datasets overview."""
    datasets = _read_json(os.path.join(_STATIC_META_DIR, "datasets.json"))
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        **datasets,
    }


@mcp.tool()
def GetDatasetsMeta() -> Dict[str, Any]:
    """Datasets overview: id, title, description, source, license, last_updated."""
    return _get_datasets_meta()


def _get_organizations_meta() -> Dict[str, Any]:
    """Implementation: organizations and roles (no PII)."""
    orgs = _read_json(os.path.join(_STATIC_META_DIR, "orgs.json"))
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        **orgs,
    }


@mcp.tool()
def GetOrganizationsMeta() -> Dict[str, Any]:
    """Organizations and roles involved in the project (no PII)."""
    return _get_organizations_meta()


def _get_project_links() -> Dict[str, Any]:
    """Implementation: useful project links."""
    links = _read_json(os.path.join(_STATIC_META_DIR, "links.json"))
    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        **links,
    }


@mcp.tool()
def GetProjectLinks() -> Dict[str, Any]:
    """Useful project links: docs, dashboards, KG UI, API routes, trackers."""
    return _get_project_links()


@mcp.tool()
def DescribeServer() -> Dict[str, Any]:
    """Describe this meta MCP server and available tools."""
    tools = [
        "GetMetaSummary",
        "GetRepoMeta",
        "GetDatasetsMeta",
        "GetOrganizationsMeta",
        "GetProjectLinks",
        "GetRepoFunctionSummary",
    ]
    return {
        "name": "Meta Server",
        "description": "Project metadata server exposing repo/dataset/org/link info",
        "version": "1.0.0",
        "tools": tools,
        "static_dir": "/static/meta",
        "schema_version": "1.0",
        "generated_at": _now_iso(),
    }


def _trim_summary(text: Optional[str]) -> Optional[str]:
    """Return the first line of text truncated for readability."""
    if not text:
        return None
    first_line = text.strip().splitlines()[0]
    return (first_line[:157] + "...") if len(first_line) > 160 else first_line


def _sentence_case(text: str) -> str:
    """Lower-case the leading character for smoother mid-sentence inserts."""
    if not text:
        return text
    stripped = text.strip()
    if not stripped:
        return stripped
    first_char = stripped[0]
    if first_char.isupper():
        return first_char.lower() + stripped[1:]
    return stripped


def _articleize(text: str) -> str:
    """Prefix text with a natural-sounding article when missing."""
    stripped = text.strip()
    if not stripped:
        return stripped
    lowered = stripped.lower()
    if lowered.startswith(("a ", "an ", "the ")):
        return stripped
    article = "an" if stripped[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {stripped}"


def _build_list_phrase(items: List[str]) -> str:
    """Human-friendly list joining (oxford comma style)."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _soften_language(text: Optional[str]) -> Optional[str]:
    """Swap technical jargon for plain-language equivalents."""
    if not text:
        return text
    softened = text
    for target, replacement in _SOFTEN_PATTERNS:
        softened = softened.replace(target, replacement)
    softened = softened.replace("structured structured data", "structured data")
    softened = softened.replace("map structured data", "map data")
    softened = " ".join(softened.split())
    return softened


def _summarize_module(abs_path: str, rel_path: str) -> Optional[Dict[str, Any]]:
    """Parse a module and return high-level summaries of its symbols."""
    try:
        with open(abs_path, "r", encoding="utf-8") as source_file:
            source = source_file.read()
    except (OSError, UnicodeDecodeError):
        return None

    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return None
    module_doc = _trim_summary(ast.get_docstring(tree))
    function_nodes: List[ast.AST] = []
    class_nodes: List[ast.ClassDef] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            class_nodes.append(node)

    if not module_doc and not function_nodes and not class_nodes:
        return None

    public_function_nodes = [fn for fn in function_nodes if not fn.name.startswith("_")]
    public_class_nodes = [cls for cls in class_nodes if not cls.name.startswith("_")]

    chosen_function_nodes = (public_function_nodes or function_nodes)[:_MAX_FUNCTION_SUMMARIES]
    chosen_class_nodes = (public_class_nodes or class_nodes)[:_MAX_CLASS_SUMMARIES]

    function_summaries = [
        {
            "name": fn.name,
            "description": _trim_summary(ast.get_docstring(fn)) or "No docstring provided.",
        }
        for fn in chosen_function_nodes
    ]

    class_summaries = []
    for cls in chosen_class_nodes:
        methods = [
            item
            for item in cls.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        public_method_names = [m.name for m in methods if not m.name.startswith("_")]
        chosen_methods = (public_method_names or [m.name for m in methods])[:_MAX_METHOD_NAMES]
        class_summaries.append(
            {
                "name": cls.name,
                "description": _trim_summary(ast.get_docstring(cls)) or "No docstring provided.",
                "method_count": len(methods),
                "key_methods": chosen_methods,
            }
        )

    if module_doc:
        doc_sentence = _soften_language(module_doc.strip().rstrip("."))
        lowered_first = doc_sentence.split()[0].lower() if doc_sentence.split() else ""
        action_verbs = {
            "provides",
            "handles",
            "coordinates",
            "offers",
            "manages",
            "hosts",
            "processes",
            "keeps",
            "stores",
            "routes",
            "exposes",
            "contains",
            "collects",
        }
        if lowered_first in action_verbs:
            summary_text = f"This file { _sentence_case(doc_sentence) }."
        else:
            summary_text = f"This file is {_articleize(_sentence_case(doc_sentence))}."
    else:
        parts = []
        if function_nodes:
            parts.append("brings together several helper routines")
        if class_nodes:
            parts.append("and a few supporting classes" if parts else "gathers a few supporting classes")
        if parts:
            summary_text = "This file " + " ".join(parts) + "."
        else:
            summary_text = "This file keeps the project running but lacks an overview docstring."

    def _append_sentence(text: str, sentence: str) -> str:
        text = text.strip()
        if not text:
            return sentence
        if not text.endswith("."):
            text += "."
        return f"{text} {sentence}"

    function_phrases: List[str] = []
    for fn in function_summaries:
        desc = fn["description"]
        desc = _soften_language(desc)
        if desc == "No docstring provided." or not desc:
            phrase = f"{fn['name']}, which keeps things running behind the scenes"
        else:
            phrase = f"{fn['name']}, which helps {_sentence_case(desc.rstrip('.'))}"
        function_phrases.append(phrase)

    class_phrases: List[str] = []
    for cls in class_summaries:
        desc = _soften_language(cls["description"])
        if desc == "No docstring provided." or not desc:
            phrase = f"{cls['name']}, which keeps the project data tidy"
        else:
            phrase = f"{cls['name']}, {_articleize(_sentence_case(desc.rstrip('.')))}"
        class_phrases.append(phrase)

    if function_phrases:
        summary_text = _append_sentence(summary_text, f"Notable touches include {_build_list_phrase(function_phrases)}.")
    if class_phrases:
        summary_text = _append_sentence(summary_text, f"It also introduces {_build_list_phrase(class_phrases)}.")

    return {
        "path": rel_path,
        "summary": _soften_language(summary_text),
        "function_count": len(function_nodes),
        "class_count": len(class_nodes),
        "method_count": sum(
            len(
                [
                    item
                    for item in cls.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
            )
            for cls in class_nodes
        ),
        "functions": function_summaries,
        "classes": class_summaries,
    }


def _iter_python_files(project_root: str) -> List[str]:
    """Return sorted list of repository Python files for summarization."""
    collected: List[str] = []
    for root, dirs, files in os.walk(project_root):
        rel_root = os.path.relpath(root, project_root)
        if rel_root == ".":
            rel_root = ""
        dirs[:] = [
            d
            for d in dirs
            if d not in _EXCLUDED_SUMMARY_DIRS and not d.startswith(".")
        ]
        for filename in files:
            if not filename.endswith(".py") or filename in _EXCLUDED_SUMMARY_FILES:
                continue
            rel_path = os.path.join(rel_root, filename) if rel_root else filename
            if rel_path.startswith("."):
                continue
            collected.append(rel_path)
    return sorted(collected)


def _get_repo_function_summary() -> Dict[str, Any]:
    """Aggregate module-level overviews so new teammates can orient quickly."""
    modules: List[Dict[str, Any]] = []
    total_functions = 0
    total_classes = 0
    total_methods = 0

    for rel_path in _iter_python_files(_PROJECT_ROOT):
        abs_path = os.path.join(_PROJECT_ROOT, rel_path)
        module_info = _summarize_module(abs_path, rel_path)
        if not module_info:
            continue
        modules.append(module_info)
        total_functions += module_info.get("function_count", 0)
        total_classes += module_info.get("class_count", 0)
        total_methods += module_info.get("method_count", 0)

    return {
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "module_count": len(modules),
        "function_count": total_functions,
        "class_count": total_classes,
        "method_count": total_methods,
        "modules": modules,
    }


@mcp.tool()
def GetRepoFunctionSummary() -> Dict[str, Any]:
    """High-level summary of repository modules, functions, and methods."""
    return _get_repo_function_summary()


if __name__ == "__main__":
    mcp.run()
