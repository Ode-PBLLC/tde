"""NDC Align (LSE) MCP server implementing the v2 run_query contract.

This module re-implements the legacy `lse_server` FastMCP tools so the
dataset can participate in the structured v2 orchestration layer.  All
parsing, cataloguing, and tool logic lives in this file - we do not import
the legacy module to keep the new server self-contained.
"""

import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # pragma: no cover - optional dependencies for query routing
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

try:  # pragma: no cover - optional dependencies for query routing
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:  # pragma: no cover - dotenv is optional in production
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __package__ in {None, ""}:
    from mcp.contracts_v2 import (  # type: ignore
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from mcp.servers_v2.base import RunQueryMixin  # type: ignore
    from mcp.servers_v2.support_intent import SupportIntent  # type: ignore
else:  # pragma: no cover - package execution path
    from ..contracts_v2 import (
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from ..servers_v2.base import RunQueryMixin
    from ..servers_v2.support_intent import SupportIntent

if load_dotenv:
    try:
        load_dotenv(ROOT / ".env")
    except Exception as exc:  # pragma: no cover - best effort warning
        print(f"[lse] Warning: failed to load .env file: {exc}")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "lse"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "lse_processed"


def slugify(value: str, fallback: str = "item") -> str:
    """Convert arbitrary text into a filesystem-friendly slug."""

    if not value:
        return fallback
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return ascii_text or fallback


def sanitize_value(value: Any) -> Any:
    """Normalize values for JSON serialization."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype, np.datetime64)):
        try:
            return pd.to_datetime(value).isoformat()
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_value(val) for key, val in value.items()}
    return value


def sanitize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize each value in a record."""

    return {key: sanitize_value(val) for key, val in record.items()}


def collect_sources(row: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    """Collect primary/secondary/tertiary sources into a uniform list."""

    sources: List[Dict[str, Optional[str]]] = []
    for prefix in ("primary", "secondary", "tertiary", "quaternary"):
        src = sanitize_value(row.get(f"{prefix}_source"))
        src_type = sanitize_value(row.get(f"{prefix}_source_type"))
        if src or src_type:
            sources.append({"kind": prefix, "source": src, "source_type": src_type})
    return sources


def count_yes_no(responses: Iterable[str]) -> Dict[str, int]:
    """Count simple yes/no style responses."""

    yes = no = other = 0
    for response in responses:
        if not response:
            continue
        value = response.strip().lower()
        if value.startswith("yes"):
            yes += 1
        elif value.startswith("no"):
            no += 1
        else:
            other += 1
    return {"yes": yes, "no": no, "other": other}


def build_snippet(record: Dict[str, Any], term: str) -> Optional[str]:
    """Create a short snippet containing the search term, if possible."""

    term_lower = term.lower()
    for value in record.values():
        if isinstance(value, str) and term_lower in value.lower():
            return value
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    snippet = build_snippet(item, term)
                    if snippet:
                        return snippet
                if isinstance(item, str) and term_lower in item.lower():
                    return item
    return None


def drop_empty_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove records that are entirely empty."""

    cleaned: List[Dict[str, Any]] = []
    for record in records:
        if any(value not in (None, "") for value in record.values()):
            cleaned.append(record)
    return cleaned


def detect_header_row(df: pd.DataFrame, marker: str) -> int:
    """Detect the header row by searching for a marker string."""

    marker_lower = marker.lower()
    for idx, row in df.iterrows():
        for cell in row.tolist():
            if isinstance(cell, str) and marker_lower in cell.lower():
                return idx
    return 0


@dataclass
class ProcessedSheet:
    """Normalized representation of a single Excel sheet."""

    module: str
    group: str
    source_file: str
    sheet_name: str
    slug: str
    title: str
    columns: List[str]
    records: List[Dict[str, Any]]
    summary: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_records: bool = True) -> Dict[str, Any]:
        payload = {
            "module": self.module,
            "group": self.group,
            "title": self.title,
            "slug": self.slug,
            "source_file": self.source_file,
            "sheet_name": self.sheet_name,
            "columns": self.columns,
            "summary": self.summary,
            "metadata": self.metadata,
        }
        if include_records:
            payload["records"] = self.records
        return payload

    def write_json(self, base_dir: Path) -> Path:
        """Persist the sheet to JSON under the processed directory."""

        module_dir = base_dir / self.module
        module_dir.mkdir(parents=True, exist_ok=True)
        file_path = module_dir / f"{self.slug}.json"
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(include_records=True), handle, ensure_ascii=False, indent=2)
        return file_path


@dataclass
class ModuleSpec:
    """Configuration for a workbook module."""

    module: str
    patterns: Sequence[str]
    parser: Callable[[Path, "ModuleSpec"], List[ProcessedSheet]]
    group: str
    exclude_sheets: Sequence[str] = ()
    friendly_name: Optional[str] = None


GROUP_LABELS = {
    "ndc_overview": "national_commitments",
    "institutions": "governance_processes",
    "plans_policies": "policy_frameworks",
    "subnational": "brazilian_states",
    "tpi_graphs": "transition_pathways",
}


def parse_ndc_workbook(path: Path, spec: ModuleSpec) -> List[ProcessedSheet]:
    """Parse the NDC overview workbook into a single structured sheet."""

    sheets: List[ProcessedSheet] = []
    if not path.exists():
        return sheets

    excel = pd.ExcelFile(path)
    if "Sheet1" not in excel.sheet_names:
        return sheets

    raw = excel.parse("Sheet1", header=None)
    header_idx = detect_header_row(raw, "Response from the NDC") or 3
    df = excel.parse("Sheet1", header=header_idx)
    df = df.dropna(how="all")

    columns = list(df.columns)
    rename_map: Dict[str, str] = {}
    if columns:
        rename_map[columns[0]] = "entry"
    if len(columns) > 1:
        rename_map[columns[1]] = "ndc_answer"
    if len(columns) > 2:
        rename_map[columns[2]] = "ndc_summary"
    if len(columns) > 3:
        rename_map[columns[3]] = "ndc_source"
    if len(columns) > 4:
        rename_map[columns[4]] = "domestic_alignment"
    if len(columns) > 5:
        rename_map[columns[5]] = "domestic_summary"
    if len(columns) > 6:
        rename_map[columns[6]] = "status"
    if len(columns) > 7:
        rename_map[columns[7]] = "primary_source"
    if len(columns) > 8:
        rename_map[columns[8]] = "primary_source_type"
    if len(columns) > 9:
        rename_map[columns[9]] = "secondary_source"
    if len(columns) > 10:
        rename_map[columns[10]] = "secondary_source_type"
    if len(columns) > 11:
        rename_map[columns[11]] = "tertiary_source"
    if len(columns) > 12:
        rename_map[columns[12]] = "tertiary_source_type"

    df = df.rename(columns=rename_map)
    df = df.dropna(how="all")

    if not df.empty and df.iloc[0].astype(str).str.contains("yes/ no/ na", case=False).any():
        df = df.iloc[1:]

    df = df.replace({np.nan: None})

    records: List[Dict[str, Any]] = []
    current_section: Optional[str] = None
    question_count = 0
    sources_count = 0

    for _, row in df.iterrows():
        row_dict = {col: sanitize_value(row.get(col)) for col in df.columns}
        entry = row_dict.get("entry")
        ndc_answer = row_dict.get("ndc_answer")
        ndc_summary = row_dict.get("ndc_summary")
        domestic_summary = row_dict.get("domestic_summary")

        if not entry and not ndc_summary and not domestic_summary:
            continue

        is_section = bool(entry and not ndc_summary and not domestic_summary and not ndc_answer)
        if is_section:
            current_section = entry
            records.append(
                {
                    "type": "section",
                    "title": entry,
                    "slug": slugify(entry, fallback="section"),
                }
            )
            continue

        record = {
            "type": "question",
            "label": entry,
            "section": current_section,
            "ndc_answer": ndc_answer,
            "ndc_summary": ndc_summary,
            "ndc_source": row_dict.get("ndc_source"),
            "domestic_alignment": row_dict.get("domestic_alignment"),
            "domestic_summary": domestic_summary,
            "status": row_dict.get("status"),
        }
        sources = collect_sources(row_dict)
        if sources:
            sources_count += len(sources)
            record["sources"] = sources
        records.append(sanitize_record(record))
        question_count += 1

    summary = {
        "sections": sum(1 for item in records if item.get("type") == "section"),
        "questions": question_count,
        "with_domestic_summary": sum(
            1 for item in records if item.get("type") == "question" and item.get("domestic_summary")
        ),
        "with_sources": sources_count,
    }

    sheet = ProcessedSheet(
        module=spec.module,
        group=spec.group,
        source_file=path.name,
        sheet_name="Sheet1",
        slug="ndc-overview-domestic-comparison",
        title=spec.friendly_name or "NDC Overview & Domestic Comparison",
        columns=["type", "label", "section", "ndc_answer", "ndc_summary", "domestic_summary"],
        records=records,
        summary=summary,
        metadata={"header_row": header_idx, "question_count": question_count},
    )
    sheets.append(sheet)
    return sheets


def rename_columns_with_keywords(columns: Sequence[str], module: str) -> List[str]:
    """Rename columns using keyword heuristics."""

    aliases = [col.strip() for col in columns]
    mapping: List[tuple[str, str]] = [
        ("pergunta", "question"),
        ("question", "question"),
        ("indicator", "indicator"),
        ("questions", "question"),
        ("resposta", "response"),
        ("response", "response"),
        ("status", "status"),
        ("summary", "summary"),
        ("justification", "justification"),
        ("implementation information", "implementation_information"),
        ("implementation", "implementation_information"),
        ("notes on and source for implementation information", "implementation_notes"),
        ("primary source for answer", "primary_source"),
        ("primary source type", "primary_source_type"),
        ("second source for answer", "secondary_source"),
        ("second source type", "secondary_source_type"),
        ("third source for answer", "tertiary_source"),
        ("third source type", "tertiary_source_type"),
        ("fourth source", "quaternary_source"),
        ("adaptation", "adaptation_mitigation"),
        ("mitigation", "adaptation_mitigation"),
        ("sector", "sector"),
        ("implementation stage", "implementation_stage"),
    ]

    renamed: List[str] = []
    counters: Dict[str, int] = defaultdict(int)

    for original in aliases:
        lower = original.lower()
        replacement: Optional[str] = None
        for needle, target in mapping:
            if needle in lower:
                replacement = target
                break
        if replacement is None:
            replacement = slugify(original, fallback="column")
        counters[replacement] += 1
        if counters[replacement] > 1:
            renamed.append(f"{replacement}_{counters[replacement]}")
        else:
            renamed.append(replacement)
    return renamed


def parse_generic_module(path: Path, spec: ModuleSpec) -> List[ProcessedSheet]:
    """Parse institutions or plans/policies workbooks."""

    excel = pd.ExcelFile(path)
    sheets: List[ProcessedSheet] = []

    for sheet_name in excel.sheet_names:
        if sheet_name in spec.exclude_sheets:
            continue
        df = excel.parse(sheet_name).dropna(how="all")
        if df.empty:
            continue

        df.columns = rename_columns_with_keywords(df.columns, spec.module)
        df = df.replace({np.nan: None})
        records = [sanitize_record(record) for record in df.to_dict(orient="records")]
        records = drop_empty_records(records)
        if not records:
            continue

        for record in records:
            record.setdefault("topic", sheet_name)

        yes_counts = count_yes_no(record.get("response") for record in records if record.get("response"))
        summary = {
            "rows": len(records),
            "columns": df.columns.tolist(),
            "yes_responses": yes_counts["yes"],
            "no_responses": yes_counts["no"],
            "with_sources": sum(1 for r in records if r.get("primary_source")),
        }

        sheet = ProcessedSheet(
            module=spec.module,
            group=spec.group,
            source_file=path.name,
            sheet_name=sheet_name,
            slug=f"{spec.module}-{slugify(sheet_name)}",
            title=sheet_name,
            columns=df.columns.tolist(),
            records=records,
            summary=summary,
            metadata={"topic": sheet_name},
        )
        sheets.append(sheet)

    return sheets


def parse_subnational_workbook(path: Path, spec: ModuleSpec) -> List[ProcessedSheet]:
    """Parse Brazilian subnational governance workbook."""

    excel = pd.ExcelFile(path)
    sheets: List[ProcessedSheet] = []

    for sheet_name in excel.sheet_names:
        if sheet_name in spec.exclude_sheets:
            continue

        df = excel.parse(sheet_name).dropna(how="all")
        if df.empty:
            continue

        original_columns = list(df.columns)
        new_columns: List[Optional[str]] = []
        source_counter = 0
        note_counter = 0

        for idx, column in enumerate(original_columns):
            column_str = str(column).strip()
            lower = column_str.lower()
            alias: Optional[str] = None
            if idx == 0:
                alias = "question"
            elif "display logic" in lower:
                alias = None
            elif "yes" in lower and "no" in lower:
                alias = "response"
            elif "summary" in lower:
                alias = "summary"
            elif "status" in lower:
                alias = "status"
            elif "source document" in lower:
                source_counter += 1
                alias = f"source_document_{source_counter}"
            elif "please specify" in lower:
                note_counter += 1
                alias = f"source_note_{note_counter}"
            else:
                alias = slugify(column_str, fallback=f"column_{idx}")
            new_columns.append(alias)

        filtered_columns = [alias for alias in new_columns if alias]
        df.columns = new_columns
        df = df[[col for col in filtered_columns if col in df.columns]]
        df = df.replace({np.nan: None})

        records = []
        for record in df.to_dict(orient="records"):
            cleaned = sanitize_record(record)
            if any(value is not None for value in cleaned.values()):
                records.append(cleaned)

        if not records:
            continue

        state_name = sheet_name
        state_code = None
        if "(" in sheet_name and ")" in sheet_name:
            name_part, code_part = sheet_name.split("(", 1)
            state_name = name_part.strip()
            state_code = code_part.replace(")", "").strip()
        records_with_state = []
        responses = []
        for record in records:
            record["state"] = state_name
            record["state_code"] = state_code or state_name
            responses.append(record.get("response"))
            records_with_state.append(record)

        yes_counts = count_yes_no(filter(None, responses))
        summary = {
            "rows": len(records_with_state),
            "state_name": state_name,
            "state_code": state_code or state_name,
            "yes_responses": yes_counts["yes"],
            "no_responses": yes_counts["no"],
        }

        sheet = ProcessedSheet(
            module=spec.module,
            group=spec.group,
            source_file=path.name,
            sheet_name=sheet_name,
            slug=f"{spec.module}-{slugify(sheet_name)}",
            title=state_name,
            columns=list(df.columns),
            records=records_with_state,
            summary=summary,
            metadata={"state_name": state_name, "state_code": state_code},
        )
        sheets.append(sheet)

    return sheets


def parse_tpi_workbook(path: Path, spec: ModuleSpec) -> List[ProcessedSheet]:
    """Parse the TPI graph workbook."""

    excel = pd.ExcelFile(path)
    sheets: List[ProcessedSheet] = []

    for sheet_name in excel.sheet_names:
        df = excel.parse(sheet_name).dropna(how="all")
        if df.empty:
            continue
        df = df.replace({np.nan: None})
        columns = rename_columns_with_keywords(df.columns, spec.module)
        df.columns = columns
        records = [sanitize_record(record) for record in df.to_dict(orient="records")]
        sheet = ProcessedSheet(
            module=spec.module,
            group=spec.group,
            source_file=path.name,
            sheet_name=sheet_name,
            slug=f"{spec.module}-{slugify(sheet_name)}",
            title="TPI Emissions Pathways",
            columns=columns,
            records=records,
            summary={"rows": len(records), "columns": columns},
            metadata={},
        )
        sheets.append(sheet)
    return sheets


MODULE_SPECS: Sequence[ModuleSpec] = (
    ModuleSpec(
        module="ndc_overview",
        patterns=("1 NDC Overview",),
        parser=parse_ndc_workbook,
        group=GROUP_LABELS["ndc_overview"],
        friendly_name="NDC Overview & Domestic Comparison",
    ),
    ModuleSpec(
        module="institutions",
        patterns=("2 Institutions",),
        parser=parse_generic_module,
        group=GROUP_LABELS["institutions"],
        exclude_sheets=("How to use", "Metadata"),
    ),
    ModuleSpec(
        module="plans_policies",
        patterns=("3 Plans and Policies",),
        parser=parse_generic_module,
        group=GROUP_LABELS["plans_policies"],
        exclude_sheets=("How to use", "Metadata lists"),
    ),
    ModuleSpec(
        module="subnational",
        patterns=("4 Subnational",),
        parser=parse_subnational_workbook,
        group=GROUP_LABELS["subnational"],
        exclude_sheets=("How to use", "Metadata"),
    ),
    ModuleSpec(
        module="tpi_graphs",
        patterns=("1_1 TPI Graph",),
        parser=parse_tpi_workbook,
        group=GROUP_LABELS["tpi_graphs"],
    ),
)


class LSECatalog:
    """Load, normalize, and index the LSE workbooks."""

    def __init__(self, raw_dir: Path, processed_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.sheets: Dict[str, ProcessedSheet] = {}
        self.module_index: Dict[str, List[str]] = defaultdict(list)
        self.group_index: Dict[str, List[str]] = defaultdict(list)
        self.metadata: Dict[str, Any] = {}
        self._load()

    def _identify_spec(self, filename: str) -> Optional[ModuleSpec]:
        for spec in MODULE_SPECS:
            if any(pattern in filename for pattern in spec.patterns):
                return spec
        return None

    def _load(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        total_records = 0

        if not self.raw_dir.exists():
            self.metadata = {
                "raw_directory": str(self.raw_dir),
                "processed_directory": str(self.processed_dir),
                "modules": 0,
                "tabs": 0,
                "records": 0,
            }
            return

        for workbook in sorted(self.raw_dir.glob("*.xlsx")):
            spec = self._identify_spec(workbook.name)
            if not spec:
                continue
            sheets = spec.parser(workbook, spec)
            for sheet in sheets:
                self.sheets[sheet.slug] = sheet
                self.module_index[sheet.module].append(sheet.slug)
                self.group_index[sheet.group].append(sheet.slug)
                sheet.write_json(self.processed_dir)
                total_records += len(sheet.records)

        modules_summary = []
        for module, slugs in self.module_index.items():
            records_count = sum(len(self.sheets[slug].records) for slug in slugs)
            modules_summary.append(
                {
                    "module": module,
                    "group": self.sheets[slugs[0]].group if slugs else None,
                    "tabs": len(slugs),
                    "records": records_count,
                }
            )

        last_updated = None
        mtimes = []
        for workbook in self.raw_dir.glob("*.xlsx"):
            mtimes.append(os.path.getmtime(workbook))
        if mtimes:
            last_updated = max(mtimes)

        self.metadata = {
            "raw_directory": str(self.raw_dir),
            "processed_directory": str(self.processed_dir),
            "modules": len(self.module_index),
            "tabs": len(self.sheets),
            "records": total_records,
            "module_summary": modules_summary,
            "groups": {group: len(slugs) for group, slugs in self.group_index.items()},
            "last_updated": last_updated,
        }

    def list_groups(self) -> Dict[str, Any]:
        groups = []
        for group, slugs in sorted(self.group_index.items()):
            modules = sorted({self.sheets[slug].module for slug in slugs})
            groups.append(
                {
                    "group": group,
                    "modules": modules,
                    "tabs": len(slugs),
                }
            )
        return {"groups": groups, "total_groups": len(groups)}

    def list_tabs(
        self,
        group: Optional[str] = None,
        module: Optional[str] = None,
        include_preview: bool = False,
    ) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        for slug, sheet in sorted(self.sheets.items()):
            if group and sheet.group != group:
                continue
            if module and sheet.module != module:
                continue
            item = {
                "slug": sheet.slug,
                "title": sheet.title,
                "module": sheet.module,
                "group": sheet.group,
                "source_file": sheet.source_file,
                "sheet_name": sheet.sheet_name,
                "record_count": len(sheet.records),
                "summary": sheet.summary,
            }
            if include_preview:
                item["preview"] = sheet.records[: min(3, len(sheet.records))]
            items.append(item)
        return {"count": len(items), "tabs": items}

    def get_sheet(self, slug: str) -> Optional[ProcessedSheet]:
        return self.sheets.get(slug)

    def get_sheet_by_state(self, state: str) -> Optional[ProcessedSheet]:
        state_lower = state.lower()
        for slug in self.module_index.get("subnational", []):
            sheet = self.sheets[slug]
            state_name = sheet.metadata.get("state_name") or sheet.title
            state_code = sheet.metadata.get("state_code")
            identifiers = [sheet.title, state_name or "", state_code or ""]
            if any(state_lower in str(identifier).lower() for identifier in identifiers):
                return sheet
        return None

    def get_ndc_sheet(self) -> Optional[ProcessedSheet]:
        slugs = self.module_index.get("ndc_overview", [])
        if not slugs:
            return None
        return self.sheets[slugs[0]]

    def search(
        self,
        term: str,
        *,
        group: Optional[str] = None,
        module: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        term_lower = term.lower()
        results: List[Dict[str, Any]] = []
        for slug, sheet in self.sheets.items():
            if group and sheet.group != group:
                continue
            if module and sheet.module != module:
                continue
            for record in sheet.records:
                haystack_parts: List[str] = []
                for value in record.values():
                    if isinstance(value, str):
                        haystack_parts.append(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                haystack_parts.extend(
                                    str(v) for v in item.values() if isinstance(v, str)
                                )
                            elif isinstance(item, str):
                                haystack_parts.append(item)
                haystack = " ".join(haystack_parts).lower()
                if term_lower in haystack:
                    snippet = build_snippet(record, term_lower)
                    results.append(
                        {
                            "slug": slug,
                            "title": sheet.title,
                            "module": sheet.module,
                            "group": sheet.group,
                            "snippet": snippet,
                            "record": record,
                        }
                    )
                    break
            if len(results) >= limit:
                break
        return {"term": term, "results": results, "count": len(results)}


def _load_dataset_metadata() -> Dict[str, Dict[str, str]]:
    path = PROJECT_ROOT / "static" / "meta" / "datasets.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    mapping: Dict[str, Dict[str, str]] = {}
    for item in payload.get("items", []):
        dataset_id = item.get("id")
        if not dataset_id:
            continue
        mapping[str(dataset_id)] = {
            "title": str(item.get("title", "")),
            "description": str(item.get("description", "")),
            "source": str(item.get("source", "")),
            "citation": str(item.get("citation", "")),
        }
    return mapping


DATASET_ID = "lse"
DATASET_METADATA = _load_dataset_metadata()
DATASET_INFO = DATASET_METADATA.get(
    DATASET_ID,
    {
        "title": "NDC Align",
        "description": "Brazil's climate governance dataset",
        "source": "https://governance.transitiondigital.org/en/tabs/ndc-overview",
        "citation": "Grantham Research Institute. NDCAlign dataset.",
    },
)

EXTRAS_DIR = PROJECT_ROOT / "extras"
SEMANTIC_INDEX_PATH = EXTRAS_DIR / "lse_semantic_index.jsonl"


class LSEServerV2(RunQueryMixin):
    """FastMCP server exposing the normalized LSE / NDC Align dataset."""

    def __init__(self) -> None:
        self.mcp = FastMCP("lse-server-v2")
        self.catalog = LSECatalog(RAW_DATA_DIR, PROCESSED_DATA_DIR)
        self.metadata = self.catalog.metadata

        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential failure
                print(f"[lse] Warning: Anthropic client unavailable: {exc}")

        self._openai_client = None
        if OpenAI and os.getenv("OPENAI_API_KEY"):
            try:
                self._openai_client = OpenAI()
            except Exception as exc:  # pragma: no cover - credential failure
                print(f"[lse] Warning: OpenAI client unavailable: {exc}")

        self._semantic_records: List[Dict[str, Any]] = []
        self._semantic_matrix: Optional[np.ndarray] = None
        self._load_semantic_index()

        self._register_capabilities_tool()
        self._register_query_support_tool()
        self._register_tool_list_groups()
        self._register_tool_list_tabs()
        self._register_tool_get_tab()
        self._register_tool_tpi_graph()
        self._register_tool_institutional_framework()
        self._register_tool_climate_policy()
        self._register_tool_subnational()
        self._register_tool_dataset_overview()
        self._register_tool_file_structure()
        self._register_tool_search()
        self._register_tool_states_overview()
        self._register_tool_state_policy()
        self._register_tool_compare_states()
        self._register_tool_ndc_targets()
        self._register_tool_ndc_policy_comparison()
        self._register_tool_ndc_implementation()
        self._register_tool_all_ndc()
        self._register_tool_ndc_overview()
        self._register_tool_institutions_all()
        self._register_tool_plans_all()
        self._register_tool_visualization()
        self._register_tool_dataset_metadata()
        self._register_tool_describe_server()
        self._register_run_query_tool()

    # ------------------------------------------------------------------ shared helpers
    def _capabilities_metadata(self) -> Dict[str, Any]:
        module_summary = self.metadata.get("module_summary", [])
        return {
            "name": "lse",
            "description": (
                "Brazil's NDC Align catalog with governance institutions, plans and policies,"
                " subnational implementation, and Transition Pathway Initiative pathways."
            ),
            "version": "2.0.0",
            "dataset": DATASET_INFO.get("title", "NDC Align"),
            "url": DATASET_INFO.get("source"),
            "tags": [
                "brazil",
                "ndc",
                "policy",
                "governance",
                "subnational",
            ],
            "modules": module_summary,
            "tools": [
                "ListLSEGroups",
                "ListLSETabs",
                "GetLSETab",
                "GetTPIGraphData",
                "GetInstitutionalFramework",
                "GetClimatePolicy",
                "GetSubnationalGovernance",
                "GetBrazilianStatesOverview",
                "GetStateClimatePolicy",
                "CompareBrazilianStates",
                "GetNDCTargets",
                "GetNDCPolicyComparison",
                "GetNDCImplementationStatus",
                "GetAllNDCData",
                "GetNDCOverviewData",
                "GetInstitutionsProcessesData",
                "GetPlansAndPoliciesData",
                "GetLSEVisualizationData",
                "DescribeServer",
                "run_query",
            ],
        }

    def _capability_summary(self) -> str:
        meta = self._capabilities_metadata()
        return (
            f"Dataset: {meta['dataset']} - {meta['description']} Modules: "
            + ", ".join(summary.get("module", "") for summary in meta.get("modules", []))
        )

    def _register_capabilities_tool(self) -> None:
        @self.mcp.tool()
        def describe_capabilities() -> Dict[str, Any]:  # type: ignore[misc]
            """Describe dataset coverage, modules, and available tools."""

            return self._capabilities_metadata()

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> str:  # type: ignore[misc]
            """Decide whether the LSE dataset can assist with the query."""

            intent = self._classify_support(query)
            payload = {
                "server": "lse",
                "query": query,
                "supported": intent.supported,
                "score": intent.score,
                "reasons": intent.reasons,
            }
            return json.dumps(payload)

    def _classify_support(self, query: str) -> SupportIntent:
        if self._anthropic_client:
            try:
                prompt = (
                    "Decide if the Brazil-focused NDC Align governance dataset should answer the question."
                    " Respond with JSON keys 'supported' (true/false) and 'reason'.\n"
                    f"Dataset summary: {self._capability_summary()}\n"
                    f"Question: {query}"
                )
                response = self._anthropic_client.messages.create(
                    model=os.getenv("LSE_ROUTER_MODEL", "claude-3-5-haiku-20241022"),
                    max_tokens=128,
                    temperature=0,
                    system="Respond with strict JSON only.",
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                intent = self._parse_support_intent(text)
                if intent:
                    return intent
            except Exception as exc:  # pragma: no cover
                return SupportIntent(True, 0.3, [f"Anthropic routing unavailable: {exc}"])

        if self._openai_client:
            try:
                prompt = (
                    "Decide if the Brazil-focused NDC Align governance dataset should answer the question."
                    " Respond with JSON keys 'supported' (true/false) and 'reason'.\n"
                    f"Dataset summary: {self._capability_summary()}\n"
                    f"Question: {query}"
                )
                response = self._openai_client.responses.create(
                    model=os.getenv("LSE_ROUTER_MODEL", "gpt-4.1-mini"),
                    input=prompt,
                    temperature=0,
                    max_output_tokens=128,
                )
                text = "".join(
                    part.text for part in response.output if hasattr(part, "text") and part.text
                ).strip()
                intent = self._parse_support_intent(text)
                if intent:
                    return intent
            except Exception as exc:  # pragma: no cover
                return SupportIntent(True, 0.3, [f"OpenAI routing unavailable: {exc}"])

        return SupportIntent(True, 0.3, ["LLM unavailable; defaulting to dataset availability"])

    @staticmethod
    def _parse_support_intent(text: str) -> Optional[SupportIntent]:
        def _attempt(blob: str) -> Optional[Dict[str, Any]]:
            try:
                data = json.loads(blob)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None

        data = _attempt(text)
        if not data:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                data = _attempt(text[start : end + 1])
        if not data:
            return None

        supported = bool(data.get("supported", False))
        reason = str(data.get("reason")) if data.get("reason") else "LLM routing"
        score = 0.9 if supported else 0.1
        return SupportIntent(supported=supported, score=score, reasons=[reason])

    # ------------------------------------------------------------------ tool registration helpers
    def _register_tool_list_groups(self) -> None:
        @self.mcp.tool()
        def ListLSEGroups() -> Dict[str, Any]:  # type: ignore[misc]
            """List dataset groups and their module coverage."""

            if not self.catalog.sheets:
                return {"error": "LSE data not available"}
            return self.catalog.list_groups()

    def _register_tool_list_tabs(self) -> None:
        @self.mcp.tool()
        def ListLSETabs(
            group: Optional[str] = None,
            module: Optional[str] = None,
            include_preview: bool = False,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """List normalized tabs with optional filters."""

            if not self.catalog.sheets:
                return {"error": "LSE data not available"}
            return self.catalog.list_tabs(group=group, module=module, include_preview=include_preview)

    def _register_tool_get_tab(self) -> None:
        @self.mcp.tool()
        def GetLSETab(slug: str, include_records: bool = True) -> Dict[str, Any]:  # type: ignore[misc]
            """Fetch a specific tab by slug."""

            sheet = self.catalog.get_sheet(slug)
            if not sheet:
                return {"error": f"Tab '{slug}' not found"}
            return sheet.to_dict(include_records=include_records)

    def _register_tool_tpi_graph(self) -> None:
        @self.mcp.tool()
        def GetTPIGraphData() -> Dict[str, Any]:  # type: ignore[misc]
            """Return Transition Pathway Initiative emissions data."""

            slugs = self.catalog.module_index.get("tpi_graphs", [])
            if not slugs:
                return {"error": "TPI graph data not available"}
            sheet = self.catalog.get_sheet(slugs[0])
            if not sheet:
                return {"error": "TPI graph data not available"}
            return {
                "description": "Transition Pathway Initiative pathways for Brazil",
                "columns": sheet.columns,
                "summary": sheet.summary,
                "records": sheet.records,
            }

    def _register_tool_institutional_framework(self) -> None:
        @self.mcp.tool()
        def GetInstitutionalFramework(topic: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[misc]
            """Fetch institutional governance entries, optionally filtered by topic."""

            slugs = self.catalog.module_index.get("institutions", [])
            if not slugs:
                return {"error": "Institutional data not available"}

            matches = []
            available_topics = []
            for slug in slugs:
                sheet = self.catalog.get_sheet(slug)
                if not sheet:
                    continue
                available_topics.append(sheet.title)
                if topic and topic.lower() not in sheet.title.lower():
                    continue
                matches.append(
                    {
                        "topic": sheet.title,
                        "slug": sheet.slug,
                        "summary": sheet.summary,
                        "records": sheet.records,
                    }
                )

            if topic and not matches:
                return {"error": f"No institutional data found for topic '{topic}'."}

            return {
                "topics": matches,
                "available_topics": available_topics,
                "count": len(matches),
            }

    def _register_tool_climate_policy(self) -> None:
        @self.mcp.tool()
        def GetClimatePolicy(policy_type: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[misc]
            """Retrieve climate plans and policies entries by policy type."""

            slugs = self.catalog.module_index.get("plans_policies", [])
            if not slugs:
                return {"error": "Plans and policies data not available"}

            matches = []
            available_types = []
            for slug in slugs:
                sheet = self.catalog.get_sheet(slug)
                if not sheet:
                    continue
                available_types.append(sheet.title)
                if policy_type and policy_type.lower() not in sheet.title.lower():
                    continue
                matches.append(
                    {
                        "policy_type": sheet.title,
                        "slug": sheet.slug,
                        "summary": sheet.summary,
                        "records": sheet.records,
                    }
                )

            if policy_type and not matches:
                return {"error": f"No policy data found for type '{policy_type}'."}

            return {
                "policy_types": matches,
                "available_policy_types": available_types,
                "count": len(matches),
                "citation": self._dataset_citation_dict(),
            }

    def _register_tool_subnational(self) -> None:
        @self.mcp.tool()
        def GetSubnationalGovernance(
            state: Optional[str] = None,
            metric: Optional[str] = None,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Retrieve Brazilian subnational governance data."""

            slugs = self.catalog.module_index.get("subnational", [])
            if not slugs:
                return {"error": "Subnational data not available"}

            if state:
                sheet = self.catalog.get_sheet_by_state(state)
                if not sheet:
                    return {"error": f"State '{state}' not found"}
                records = sheet.records
                if metric:
                    metric_lower = metric.lower()
                    records = [
                        record
                        for record in records
                        if any(
                            metric_lower in str(value).lower()
                            for key, value in record.items()
                            if key in ("question", "summary") and value
                        )
                    ]
                return {
                    "state": sheet.metadata.get("state_name") or sheet.title,
                    "state_code": sheet.metadata.get("state_code"),
                    "slug": sheet.slug,
                    "summary": sheet.summary,
                    "records": records,
                    "metric_filter": metric,
                    "citation": self._dataset_citation_dict(),
                }

            overview = []
            for slug in slugs:
                sheet = self.catalog.get_sheet(slug)
                if not sheet:
                    continue
                entry = {
                    "state": sheet.metadata.get("state_name") or sheet.title,
                    "state_code": sheet.metadata.get("state_code"),
                    "slug": sheet.slug,
                    "summary": sheet.summary,
                }
                overview.append(entry)

            return {
                "total_states": len(overview),
                "states": overview,
                "citation": self._dataset_citation_dict(),
            }

        @self.mcp.tool()
        def GetStatePolicyCoverageRanking(limit: int = 27) -> Dict[str, Any]:  # type: ignore[misc]
            """Rank Brazilian states by share of 'yes' responses in subnational questionnaire."""

            slugs = self.catalog.module_index.get("subnational", [])
            if not slugs:
                return {"error": "Subnational data not available"}

            rankings: List[Dict[str, Any]] = []
            for slug in slugs:
                sheet = self.catalog.get_sheet(slug)
                if not sheet:
                    continue

                summary = sheet.summary or {}
                total_questions = int(summary.get("rows", 0) or len(sheet.records))

                if total_questions <= 0:
                    continue

                yes_count = int(summary.get("yes_responses", 0))
                no_count = int(summary.get("no_responses", 0))
                other_count = max(total_questions - (yes_count + no_count), 0)
                coverage = round((yes_count / total_questions) * 100, 2)

                rankings.append(
                    {
                        "state": sheet.metadata.get("state_name") or sheet.title,
                        "state_code": sheet.metadata.get("state_code"),
                        "questions": total_questions,
                        "yes_responses": yes_count,
                        "no_responses": no_count,
                        "other_responses": other_count,
                        "coverage_percent": coverage,
                        "slug": sheet.slug,
                    }
                )

            rankings.sort(key=lambda item: item["coverage_percent"], reverse=True)
            limited_rankings = rankings[: max(1, limit)]

            chart_payload = {
                "labels": [entry["state"] for entry in limited_rankings],
                "datasets": [
                    {
                        "label": "Yes response share (%)",
                        "data": [round(entry["coverage_percent"], 2) for entry in limited_rankings],
                        "backgroundColor": "#1E88E5",
                    }
                ],
            }

            artifact = {
                "type": "chart",
                "title": "State climate policy coverage",
                "metadata": {
                    "chartType": "bar",
                    "metric": "coverage_percent",
                    "limit": max(1, limit),
                },
                "data": chart_payload,
            }

            return {
                "states": limited_rankings,
                "limit": limit,
                "artifacts": [artifact],
                "summary": (
                    f"Top {min(limit, len(rankings))} states by share of 'yes' responses in the subnational governance questionnaire."
                ),
                "citation": self._dataset_citation_dict(),
            }

        # @self.mcp.tool()
        # def GetStatePolicyCoverageMap(limit: int = 27) -> Dict[str, Any]:  # type: ignore[misc]
        #     """Produce a choropleth-ready GeoJSON of state policy coverage percentages."""

        #     try:
        #         import runpy
        #         mod = runpy.run_path(str(PROJECT_ROOT / "mcp" / "servers_v2" / "brazilian_admin_server_v2.py"))
        #         admin_server_cls = mod.get("BrazilianAdminServerV2")
        #     except Exception as exc:
        #         return {"error": f"Brazilian admin server unavailable: {exc}"}

        #     if not admin_server_cls:
        #         return {"error": "Brazilian admin server unavailable"}

        #     admin_server = admin_server_cls()

        #     slugs = self.catalog.module_index.get("subnational", [])
        #     if not slugs:
        #         return {"error": "Subnational data not available"}

        #     rows: List[Dict[str, Any]] = []
        #     for slug in slugs:
        #         sheet = self.catalog.get_sheet(slug)
        #         if not sheet:
        #             continue
        #         summary = sheet.summary or {}
        #         total_questions = int(summary.get("rows", 0) or len(sheet.records))
        #         if total_questions <= 0:
        #             continue
        #         yes_count = int(summary.get("yes_responses", 0))
        #         coverage = round((yes_count / total_questions) * 100, 2)
        #         state_code = sheet.metadata.get("state_code") or sheet.title
        #         rows.append(
        #             {
        #                 "state": sheet.metadata.get("state_name") or sheet.title,
        #                 "state_code": state_code,
        #                 "coverage_percent": coverage,
        #             }
        #         )

        #     rows.sort(key=lambda item: item["coverage_percent"], reverse=True)
        #     limited_rows = rows[: max(1, limit)]
        #     states_to_color = {row["state_code"]: row for row in limited_rows}

        #     state_gdf = admin_server.states_gdf
        #     if state_gdf is None or state_gdf.empty:
        #         return {"error": "Brazilian state geometries unavailable"}

        #     def _normalise_code(value: Optional[str]) -> str:
        #         if value is None:
        #             return ""
        #         text = str(value).strip()
        #         if text.isdigit():
        #             mapping = [
        #                 "AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB",
        #                 "PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"
        #             ]
        #             idx = int(text)
        #             if 0 <= idx < len(mapping):
        #                 return mapping[idx]
        #         return text.upper()

        #     joined = []
        #     min_lon = min_lat = max_lon = max_lat = None
        #     for _, geom_row in state_gdf.iterrows():
        #         code = _normalise_code(geom_row.get("state_code") or geom_row.get("name"))
        #         match = states_to_color.get(code)
        #         if match:
        #             joined.append((geom_row.geometry, match))
        #             try:
        #                 minx, miny, maxx, maxy = geom_row.geometry.bounds
        #                 min_lon = minx if min_lon is None else min(min_lon, minx)
        #                 min_lat = miny if min_lat is None else min(min_lat, miny)
        #                 max_lon = maxx if max_lon is None else max(max_lon, maxx)
        #                 max_lat = maxy if max_lat is None else max(max_lat, maxy)
        #             except Exception:
        #                 pass

        #     if not joined:
        #         return {"error": "No matching states found for coverage map"}

        #     min_value: Optional[float] = None
        #     max_value: Optional[float] = None
        #     coverages: List[float] = []
        #     for geometry, record in joined:
        #         coverage_value = float(record["coverage_percent"])
        #         coverages.append(coverage_value)
        #         if min_value is None or coverage_value < min_value:
        #             min_value = coverage_value
        #         if max_value is None or coverage_value > max_value:
        #             max_value = coverage_value

        #     range_min = min_value if min_value is not None else (min(coverages) if coverages else 0.0)
        #     range_max = max_value if max_value is not None else (max(coverages) if coverages else 100.0)

        #     # Update features with category now that range is known
        #     features = []
        #     for geometry, record in joined:
        #         coverage_value = float(record["coverage_percent"])
        #         category = "high coverage" if coverage_value >= range_max else "lower coverage"
        #         features.append(
        #             {
        #                 "type": "Feature",
        #                 "geometry": geometry.__geo_interface__,
        #                 "properties": {
        #                     "state": record["state"],
        #                     "state_code": record["state_code"],
        #                     "country": category,
        #                     "coverage_category": category,
        #                     "coverage_percent": coverage_value,
        #                 },
        #             }
        #         )

        #     if min_lon is not None and min_lat is not None and max_lon is not None and max_lat is not None:
        #         padding_lon = max((max_lon - min_lon) * 0.05, 0.25)
        #         padding_lat = max((max_lat - min_lat) * 0.05, 0.25)
        #         bounds = {
        #             "west": float(min_lon - padding_lon),
        #             "south": float(min_lat - padding_lat),
        #             "east": float(max_lon + padding_lon),
        #             "north": float(max_lat + padding_lat),
        #         }
        #         center = {
        #             "lon": float((min_lon + max_lon) / 2),
        #             "lat": float((min_lat + max_lat) / 2),
        #         }
        #     else:
        #         bounds = None
        #         center = None

        #     identifier = f"lse_state_policy_coverage_{len(features)}"
        #     filename = f"lse_state_policy_coverage_{len(features)}.geojson"
        #     path = admin_server.static_maps_dir / filename
        #     admin_server.static_maps_dir.mkdir(parents=True, exist_ok=True)
        #     with open(path, "w", encoding="utf-8") as handle:
        #         json.dump({"type": "FeatureCollection", "features": features}, handle)

        #     color_property = "coverage_percent"
        #     metadata = {
        #         "dataset_id": DATASET_ID,
        #         "geometry_type": "polygon",
        #         "metric": "coverage_percent",
        #         "range": [range_min, range_max],
        #         "fill_style": {
        #             "type": "interpolate",
        #             "color_property": color_property,
        #             "palette": ["#d4e4ff", "#1e88e5"],
        #         },
        #     }
        #     legend_items = [
        #         {
        #             "label": "High coverage",
        #             "color": "#1e88e5",
        #             "description": f">= {round(range_max, 2)}% yes responses",
        #         },
        #         {
        #             "label": "Lower coverage",
        #             "color": "#d4e4ff",
        #             "description": f"<= {round(range_min, 2)}% yes responses",
        #         },
        #     ]
        #     metadata["legend"] = {"title": "State coverage", "items": legend_items}
        #     metadata["fill_style"]["match_property"] = "coverage_category"
        #     if bounds and center:
        #         metadata["bounds"] = bounds
        #         metadata["center"] = center

        #     artifact = {
        #         "type": "map",
        #         "title": "State climate policy coverage",
        #         "metadata": metadata,
        #         "geojson_url": f"/static/maps/{filename}",
        #     }

        #     view_state = None
        #     if bounds and center:
        #         view_state = {
        #             "center": [center["lon"], center["lat"]],
        #             "bounds": bounds,
        #             "zoom": 4.0,
        #         }

        #     return {
        #         "states": limited_rows,
        #         "range": {
        #             "min": range_min,
        #             "max": range_max,
        #         },
        #         "summary": (
        #             f"Top {min(limit, len(rows))} states by 'yes' response share in the subnational questionnaire, rendered as a choropleth."
        #         ),
        #         "artifacts": [artifact],
        #         "view_state": view_state,
        #         "citation": self._dataset_citation_dict(),
        #     }

    def _register_tool_dataset_overview(self) -> None:
        @self.mcp.tool()
        def GetLSEDatasetOverview() -> Dict[str, Any]:  # type: ignore[misc]
            """Provide high-level dataset overview."""

            if not self.catalog.sheets:
                return {"error": "LSE data not available"}
            return self.catalog.metadata

    def _register_tool_file_structure(self) -> None:
        @self.mcp.tool()
        def GetLSEFileStructure(filename: str) -> Dict[str, Any]:  # type: ignore[misc]
            """Describe processed tabs originating from a given workbook."""

            matches = [sheet for sheet in self.catalog.sheets.values() if sheet.source_file == filename]
            if not matches:
                return {"error": f"File '{filename}' not found in processed catalog"}
            tabs = [
                {
                    "slug": sheet.slug,
                    "title": sheet.title,
                    "sheet_name": sheet.sheet_name,
                    "records": len(sheet.records),
                    "columns": sheet.columns,
                }
                for sheet in matches
            ]
            return {
                "filename": filename,
                "tabs": tabs,
                "total_tabs": len(tabs),
            }

    def _register_tool_search(self) -> None:
        @self.mcp.tool()
        def SearchLSEContent(
            search_term: Optional[str] = None,
            *,
            query: Optional[str] = None,
            module_type: Optional[str] = None,
            limit: int = 10,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Search across normalized LSE content."""

            if not self.catalog.sheets:
                return {"error": "LSE data not available"}
            term = search_term or query
            if not term:
                return {"error": "Missing search term"}
            semantic_hits = self._semantic_search(term, limit=limit)
            if semantic_hits:
                return {
                    "term": term,
                    "results": semantic_hits,
                    "count": len(semantic_hits),
                    "method": "semantic",
                }

            result = self.catalog.search(term, module=module_type, limit=limit)
            if not result.get("results"):
                result["guidance"] = (
                    "No direct matches found. Try broader keywords or explore modules via ListLSEGroups."
                )
            return result

    def _register_tool_states_overview(self) -> None:
        @self.mcp.tool()
        def GetBrazilianStatesOverview() -> Dict[str, Any]:  # type: ignore[misc]
            """List available Brazilian states with summarized metrics."""

            slugs = self.catalog.module_index.get("subnational", [])
            if not slugs:
                return {"error": "Subnational data not available"}
            overview = []
            for slug in slugs:
                sheet = self.catalog.get_sheet(slug)
                if sheet:
                    overview.append(
                        {
                            "state": sheet.metadata.get("state_name") or sheet.title,
                            "state_code": sheet.metadata.get("state_code"),
                            "slug": sheet.slug,
                            "summary": sheet.summary,
                        }
                    )
            return {
                "total_states": len(overview),
                "states": overview,
            }

    def _register_tool_state_policy(self) -> None:
        @self.mcp.tool()
        def GetStateClimatePolicy(
            state_name: Optional[str] = None,
            *,
            state: Optional[str] = None,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Retrieve the full policy table for a specific Brazilian state."""

            target = state_name or state
            if not target:
                return {"error": "State name is required"}

            sheet = self.catalog.get_sheet_by_state(target)
            if not sheet:
                return {"error": f"State '{target}' not found"}
            return {
                "state": sheet.metadata.get("state_name") or sheet.title,
                "state_code": sheet.metadata.get("state_code"),
                "slug": sheet.slug,
                "summary": sheet.summary,
                "records": sheet.records,
            }

    def _register_tool_compare_states(self) -> None:
        @self.mcp.tool()
        def CompareBrazilianStates(
            states: List[str],
            policy_area: Optional[str] = None,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Compare policy coverage across multiple Brazilian states."""

            comparison = {
                "states": [],
                "policy_area": policy_area,
                "metrics": {},
            }

            for state in states:
                sheet = self.catalog.get_sheet_by_state(state)
                if not sheet:
                    continue
                records = sheet.records
                if policy_area:
                    area_lower = policy_area.lower()
                    records = [
                        record
                        for record in records
                        if area_lower in str(record.get("question", "")).lower()
                    ]
                responses = [record.get("response") for record in records if record.get("response")]
                counts = count_yes_no(responses)
                coverage = 0.0
                if records:
                    coverage = round((counts["yes"] / len(records)) * 100, 2)
                key = sheet.metadata.get("state_name") or sheet.title
                comparison["states"].append(key)
                comparison["metrics"][key] = {
                    "questions": len(records),
                    "yes": counts["yes"],
                    "no": counts["no"],
                    "other": counts["other"],
                    "coverage_percent": coverage,
                }

            if not comparison["states"]:
                comparison["error"] = "No states found for comparison"
            comparison["citation"] = self._dataset_citation_dict()
            comparison["citation"] = self._dataset_citation_dict()
            comparison["citation"] = self._dataset_citation_dict()
            comparison["citation"] = self._dataset_citation_dict()
            return comparison

    def _register_tool_ndc_targets(self) -> None:
        @self.mcp.tool()
        def GetNDCTargets(country: str = "Brazil") -> Dict[str, Any]:  # type: ignore[misc]
            """Extract key NDC targets and commitments."""

            records = self._ndc_records()
            if not records:
                return {"error": "NDC data not available"}

            targets: Dict[str, Any] = {
                "country": country,
                "sections": defaultdict(list),
                "long_term": None,
                "interim_targets": {},
                "adaptation": None,
                "principles": [],
                "sources": set(),
            }

            current_section: Optional[str] = None
            for record in records:
                if record.get("type") == "section":
                    current_section = record.get("title")
                    continue
                if record.get("type") != "question":
                    continue

                label = record.get("label") or ""
                ndc_summary = record.get("ndc_summary")
                ndc_answer = (record.get("ndc_answer") or "").lower()
                domestic_summary = record.get("domestic_summary")
                section = record.get("section") or current_section

                if section:
                    targets["sections"][section].append(record)

                if "long term" in label.lower():
                    targets["long_term"] = {
                        "question": label,
                        "answer": ndc_summary,
                        "domestic": domestic_summary,
                    }
                if "2030" in label:
                    targets["interim_targets"]["2030"] = {
                        "question": label,
                        "answer": ndc_summary,
                        "contains_target": "yes" in ndc_answer,
                    }
                if "2035" in label:
                    targets["interim_targets"]["2035"] = {
                        "question": label,
                        "answer": ndc_summary,
                        "contains_target": "yes" in ndc_answer,
                    }
                if "2040" in label:
                    targets["interim_targets"]["2040"] = {
                        "question": label,
                        "answer": ndc_summary,
                        "contains_target": "yes" in ndc_answer,
                    }
                if "adaptation" in label.lower():
                    targets["adaptation"] = {
                        "question": label,
                        "answer": ndc_summary,
                        "domestic": domestic_summary,
                    }
                if "principle" in label.lower():
                    targets["principles"].append(
                        {
                            "question": label,
                            "answer": ndc_summary,
                            "domestic": domestic_summary,
                        }
                    )
                for source in record.get("sources", []):
                    if isinstance(source, dict) and source.get("source"):
                        targets["sources"].add(source["source"])

            targets["sources"] = sorted(targets["sources"])
            targets["citation"] = self._dataset_citation_dict()
            return targets

    def _register_tool_ndc_policy_comparison(self) -> None:
        @self.mcp.tool()
        def GetNDCPolicyComparison() -> Dict[str, Any]:  # type: ignore[misc]
            """Compare NDC commitments with domestic policy responses."""

            records = self._ndc_records()
            if not records:
                return {"error": "NDC data not available"}

            comparison = {
                "aligned": [],
                "gaps": [],
                "legal_framework": [],
            }

            for record in records:
                if record.get("type") != "question":
                    continue
                label = record.get("label") or ""
                ndc_answer = (record.get("ndc_answer") or "").lower()
                ndc_summary = record.get("ndc_summary")
                domestic_summary = record.get("domestic_summary")
                if not ndc_summary and not domestic_summary:
                    continue

                if "enforceable" in label.lower() or "paris agreement" in label.lower():
                    comparison["legal_framework"].append(
                        {
                            "question": label,
                            "ndc_position": ndc_summary,
                            "domestic_position": domestic_summary,
                        }
                    )
                    continue

                ndc_has_commitment = "yes" in ndc_answer or (ndc_summary is not None and len(str(ndc_summary)) > 0)
                domestic_has_commitment = domestic_summary is not None and len(str(domestic_summary)) > 0

                if ndc_has_commitment and domestic_has_commitment:
                    comparison["aligned"].append(
                        {
                            "question": label,
                            "ndc": ndc_summary,
                            "domestic": domestic_summary,
                        }
                    )
                elif ndc_has_commitment and not domestic_has_commitment:
                    comparison["gaps"].append(
                        {
                            "question": label,
                            "ndc": ndc_summary,
                            "domestic": domestic_summary,
                        }
                    )

            comparison["citation"] = self._dataset_citation_dict()
            return comparison

    def _register_tool_ndc_implementation(self) -> None:
        @self.mcp.tool()
        def GetNDCImplementationStatus(country: str = "Brazil") -> Dict[str, Any]:  # type: ignore[misc]
            """Summarize implementation evidence for NDC commitments."""

            records = self._ndc_records()
            if not records:
                return {"error": "NDC data not available"}

            implemented = []
            pending = []
            instruments = []

            for record in records:
                if record.get("type") != "question":
                    continue
                ndc_summary = record.get("ndc_summary") or ""
                domestic_summary = record.get("domestic_summary") or ""
                question = record.get("label")
                ndc_answer = (record.get("ndc_answer") or "").lower()

                has_ndc_commitment = "yes" in ndc_answer or bool(ndc_summary)
                has_domestic_action = any(
                    keyword in domestic_summary.lower()
                    for keyword in ("law", "decree", "resolution", "implemented", "policy")
                )

                entry = {
                    "question": question,
                    "ndc": ndc_summary,
                    "domestic": domestic_summary,
                }

                if has_ndc_commitment and has_domestic_action:
                    implemented.append(entry)
                elif has_ndc_commitment:
                    pending.append(entry)

                if any(
                    keyword in domestic_summary
                    for keyword in ("Law", "Decree", "Resolution", "Ordinance")
                ):
                    instruments.append(entry)

            total_targets = len(implemented) + len(pending)
            implementation_rate = round((len(implemented) / total_targets) * 100, 2) if total_targets else 0.0

            return {
                "country": country,
                "implemented_targets": implemented,
                "pending_targets": pending,
                "implementation_rate_percent": implementation_rate,
                "instruments": instruments,
                "citation": self._dataset_citation_dict(),
            }

    def _register_tool_all_ndc(self) -> None:
        @self.mcp.tool()
        def GetAllNDCData() -> Dict[str, Any]:  # type: ignore[misc]
            """Return the full normalized NDC overview dataset."""

            sheet = self.catalog.get_ndc_sheet()
            if not sheet:
                return {"error": "NDC data not available"}
            return sheet.to_dict(include_records=True)

    def _register_tool_ndc_overview(self) -> None:
        @self.mcp.tool()
        def GetNDCOverviewData(country: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[misc]
            """Provide NDC overview summary for the given country."""

            sheet = self.catalog.get_ndc_sheet()
            if not sheet:
                return {"error": "NDC data not available"}
            summary = sheet.summary.copy()
            if country:
                summary["country"] = country
            return {
                "metadata": sheet.metadata,
                "summary": summary,
                "records": sheet.records,
                "citation": self._dataset_citation_dict(),
            }

    def _register_tool_institutions_all(self) -> None:
        @self.mcp.tool()
        def GetInstitutionsProcessesData() -> Dict[str, Any]:  # type: ignore[misc]
            """Return all institutions & processes tabs."""

            slugs = self.catalog.module_index.get("institutions", [])
            if not slugs:
                return {"error": "Institutional data not available"}
            tabs = [
                self.catalog.get_sheet(slug).to_dict(include_records=True)
                for slug in slugs
                if self.catalog.get_sheet(slug)
            ]
            return {
                "tabs": tabs,
                "count": len(tabs),
            }

    def _register_tool_plans_all(self) -> None:
        @self.mcp.tool()
        def GetPlansAndPoliciesData() -> Dict[str, Any]:  # type: ignore[misc]
            """Return all plans & policies tabs."""

            slugs = self.catalog.module_index.get("plans_policies", [])
            if not slugs:
                return {"error": "Plans and policies data not available"}
            tabs = [
                self.catalog.get_sheet(slug).to_dict(include_records=True)
                for slug in slugs
                if self.catalog.get_sheet(slug)
            ]
            return {
                "tabs": tabs,
                "count": len(tabs),
            }

    def _register_tool_visualization(self) -> None:
        @self.mcp.tool()
        def GetLSEVisualizationData(
            viz_type: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Provide pre-aggregated data for simple visualizations."""

            if not self.catalog.sheets:
                return {"error": "LSE data not available"}
            filters = filters or {}
            viz_type_value = (viz_type or "states_comparison").strip()
            viz_type_lower = viz_type_value.lower()

            if viz_type_lower == "states_comparison":
                slugs = self.catalog.module_index.get("subnational", [])
                data = []
                for slug in slugs:
                    sheet = self.catalog.get_sheet(slug)
                    if not sheet:
                        continue
                    responses = [record.get("response") for record in sheet.records if record.get("response")]
                    counts = count_yes_no(responses)
                    total = len(sheet.records)
                    coverage = round((counts["yes"] / total) * 100, 2) if total else 0.0
                    data.append(
                        {
                            "state": sheet.metadata.get("state_name") or sheet.title,
                            "state_code": sheet.metadata.get("state_code"),
                            "yes": counts["yes"],
                            "no": counts["no"],
                            "coverage": coverage,
                        }
                    )
                return {
                    "visualization": "states_comparison",
                    "data": data,
                    "chart": {"type": "bar", "x": "state", "y": "coverage"},
                }

            if viz_type_lower == "module_overview":
                return {
                    "visualization": "module_overview",
                    "data": self.catalog.metadata.get("module_summary", []),
                    "chart": {"type": "bar", "x": "module", "y": "tabs"},
                }

            if viz_type_lower == "policy_coverage":
                modules = []
                for module, slugs in self.catalog.module_index.items():
                    modules.append(
                        {
                            "module": module,
                            "records": sum(
                                len(self.catalog.get_sheet(slug).records)
                                for slug in slugs
                                if self.catalog.get_sheet(slug)
                            ),
                            "tabs": len(slugs),
                        }
                    )
                return {
                    "visualization": "policy_coverage",
                    "data": modules,
                    "chart": {"type": "bar", "x": "module", "y": "records"},
                }

            if viz_type_lower == "governance_status":
                status_counts: Dict[str, int] = defaultdict(int)
                for slug in self.catalog.module_index.get("subnational", []):
                    sheet = self.catalog.get_sheet(slug)
                    if not sheet:
                        continue
                    for record in sheet.records:
                        status = record.get("status") or "Unknown"
                        status_counts[status] += 1
                data = [{"status": status, "count": count} for status, count in status_counts.items()]
                return {
                    "visualization": "governance_status",
                    "data": data,
                    "chart": {"type": "pie", "label": "status", "value": "count"},
                }

            return {"error": f"Unknown visualization type '{viz_type}'."}

    def _register_tool_dataset_metadata(self) -> None:
        @self.mcp.tool()
        def GetLSEDatasetMetadata() -> Dict[str, Any]:  # type: ignore[misc]
            """Expose raw metadata for the normalized dataset."""

            return self.catalog.metadata

    def _register_tool_describe_server(self) -> None:
        @self.mcp.tool()
        def DescribeServer() -> Dict[str, Any]:  # type: ignore[misc]
            """Describe the server, modules, and tooling."""

            if not self.catalog.sheets:
                return {"error": "LSE data not available"}
            metadata = self.catalog.metadata.copy()
            metadata["tools"] = [
                "ListLSEGroups",
                "ListLSETabs",
                "SearchLSEContent",
                "GetLSETab",
                "GetTPIGraphData",
                "GetSubnationalGovernance",
                "run_query",
            ]
            return metadata

    # ------------------------------------------------------------------ run_query helpers
    def _ndc_records(self) -> List[Dict[str, Any]]:
        sheet = self.catalog.get_ndc_sheet()
        if not sheet:
            return []
        return sheet.records

    def _dataset_citation(self) -> CitationPayload:
        return CitationPayload(
            id="lse-dataset",
            server="lse",
            tool="GetLSEDatasetOverview",
            title=DATASET_INFO.get("title", "NDC Align"),
            source_type="Dataset",
            description=DATASET_INFO.get("citation"),
            url=DATASET_INFO.get("source"),
            metadata={
                "modules": [summary.get("module") for summary in self.metadata.get("module_summary", [])],
                "records": self.metadata.get("records"),
            },
        )

    def _dataset_citation_dict(self) -> Dict[str, Any]:
        citation = self._dataset_citation()
        return {
            "id": citation.id,
            "server": citation.server,
            "tool": citation.tool,
            "title": citation.title,
            "source_type": citation.source_type,
            "description": citation.description,
            "url": citation.url,
            "metadata": citation.metadata,
        }

    @staticmethod
    def _ensure_citation(
        citations: List[CitationPayload], citation: CitationPayload
    ) -> None:
        if any(existing.id == citation.id for existing in citations):
            return
        citations.append(citation)

    def _citation_for_record(
        self,
        sheet: ProcessedSheet,
        record: Dict[str, Any],
        citations: List[CitationPayload],
        citation_lookup: Dict[str, str],
    ) -> str:
        return "lse-dataset"

    @staticmethod
    def _format_record_text(sheet: ProcessedSheet, record: Dict[str, Any]) -> Optional[str]:
        label = (record.get("label") or record.get("question") or "").strip()
        if record.get("type") == "section" or not label:
            return None

        parts: List[str] = []

        ndc_summary = record.get("ndc_summary")
        if isinstance(ndc_summary, str) and ndc_summary.strip():
            parts.append(f"NDC summary: {ndc_summary.strip()}")

        domestic_alignment = record.get("domestic_alignment")
        if isinstance(domestic_alignment, str) and domestic_alignment.strip():
            parts.append(f"Domestic alignment: {domestic_alignment.strip()}")

        domestic_summary = record.get("domestic_summary") or record.get("summary")
        if isinstance(domestic_summary, str) and domestic_summary.strip():
            parts.append(f"Domestic summary: {domestic_summary.strip()}")

        response = record.get("response")
        if isinstance(response, str) and response.strip():
            parts.append(f"Response: {response.strip()}")

        status = record.get("status")
        if isinstance(status, str) and status.strip():
            parts.append(f"Status: {status.strip()}")

        implementation_notes = record.get("implementation_information") or record.get(
            "implementation_information_2"
        )
        if isinstance(implementation_notes, str) and implementation_notes.strip():
            parts.append(f"Implementation notes: {implementation_notes.strip()}")

        if not parts:
            return None

        header = f"{sheet.title}: {label}"
        return header + "\n" + "\n".join(parts)

    def _record_text_for_embedding(self, sheet: ProcessedSheet, record: Dict[str, Any]) -> Optional[str]:
        text = self._format_record_text(sheet, record)
        if not text:
            return None
        module = sheet.module.replace("_", " ")
        group = sheet.group.replace("_", " ")
        prefix = f"Module: {module} | Group: {group}"
        return f"{prefix}\n{text}"

    def _load_semantic_index(self) -> None:
        if not SEMANTIC_INDEX_PATH.exists():
            self._build_semantic_index()
            return

        records: List[Dict[str, Any]] = []
        try:
            with SEMANTIC_INDEX_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                        embedding = np.array(entry.get("embedding"), dtype=float)
                        if embedding.size == 0:
                            continue
                        entry["embedding"] = embedding
                        records.append(entry)
                    except Exception:
                        continue
        except Exception as exc:
            print(f"[lse] Warning: failed to load semantic index: {exc}")
            records = []

        if records:
            self._semantic_records = records
            self._semantic_matrix = np.vstack([entry["embedding"] for entry in records])
        else:
            self._semantic_records = []
            self._semantic_matrix = None
            self._build_semantic_index()

    def _build_semantic_index(self) -> None:
        if self._openai_client is None:
            return

        texts: List[str] = []
        metadata: List[Dict[str, Any]] = []

        for slug in self.catalog.sheets:
            sheet = self.catalog.get_sheet(slug)
            if not sheet:
                continue
            for idx, record in enumerate(sheet.records):
                text = self._record_text_for_embedding(sheet, record)
                if not text:
                    continue
                texts.append(text)
                metadata.append({
                    "slug": sheet.slug,
                    "record_index": idx,
                    "module": sheet.module,
                    "group": sheet.group,
                    "title": sheet.title,
                    "snippet": text.split("\n", 1)[-1][:280],
                })

        if not texts:
            return

        vectors: List[np.ndarray] = []
        try:
            batch_size = 90
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                response = self._openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk,
                )
                for item in response.data:
                    vectors.append(np.array(item.embedding, dtype=float))
        except Exception as exc:
            print(f"[lse] Warning: failed to generate semantic index: {exc}")
            return

        if len(vectors) != len(metadata):
            print("[lse] Warning: embedding count mismatch; skipping semantic index")
            return

        EXTRAS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with SEMANTIC_INDEX_PATH.open("w", encoding="utf-8") as handle:
                for meta, vector in zip(metadata, vectors):
                    entry = meta.copy()
                    entry["embedding"] = vector.tolist()
                    json.dump(entry, handle)
                    handle.write("\n")
        except Exception as exc:
            print(f"[lse] Warning: failed to persist semantic index: {exc}")

        self._semantic_records = []
        matrix = []
        for meta, vector in zip(metadata, vectors):
            meta_copy = meta.copy()
            meta_copy["embedding"] = vector
            self._semantic_records.append(meta_copy)
            matrix.append(vector)

        if matrix:
            self._semantic_matrix = np.vstack(matrix)
        else:
            self._semantic_matrix = None

    def _embed_query(self, text: str) -> Optional[np.ndarray]:
        if not text or self._openai_client is None:
            return None
        try:
            response = self._openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[text],
            )
            return np.array(response.data[0].embedding, dtype=float)
        except Exception as exc:
            print(f"[lse] Warning: query embedding failed: {exc}")
            return None

    def _semantic_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if (
            not query
            or self._semantic_matrix is None
            or not len(self._semantic_records)
        ):
            return []

        query_vector = self._embed_query(query)
        if query_vector is None or query_vector.size == 0:
            return []

        similarities = cosine_similarity(query_vector.reshape(1, -1), self._semantic_matrix)[0]
        top_indices = similarities.argsort()[::-1][:limit]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            entry = self._semantic_records[idx]
            sheet = self.catalog.get_sheet(entry.get("slug"))
            if not sheet:
                continue
            record_index = entry.get("record_index")
            if record_index is None or record_index >= len(sheet.records):
                continue
            record = sheet.records[record_index]
            snippet = entry.get("snippet") or self._format_record_text(sheet, record) or ""
            results.append(
                {
                    "slug": sheet.slug,
                    "title": sheet.title,
                    "module": sheet.module,
                    "group": sheet.group,
                    "snippet": snippet,
                    "record": record,
                    "score": float(similarities[idx]),
                }
            )
        return results

    @staticmethod
    def _summarize_record(sheet: ProcessedSheet, record: Dict[str, Any]) -> Optional[str]:
        return LSEServerV2._format_record_text(sheet, record)

    def _ndc_focus_records(
        self, query: str
    ) -> List[Tuple[ProcessedSheet, Dict[str, Any], str]]:
        lowered = query.lower()
        if not any(keyword in lowered for keyword in ["ndc", "nationally determined", "long term", "implementation"]):
            return []

        sheet = self.catalog.get_ndc_sheet()
        if not sheet:
            return []

        prioritized_keywords = [
            "long term",
            "2030",
            "2035",
            "2040",
            "adaptation",
            "principle",
            "implementation",
        ]

        selected: List[Tuple[ProcessedSheet, Dict[str, Any], str]] = []
        seen_labels: Set[str] = set()

        for keyword in prioritized_keywords:
            for record in sheet.records:
                if record.get("type") != "question":
                    continue
                label = str(record.get("label") or "").strip()
                if not label or label.lower() in seen_labels:
                    continue
                if keyword in label.lower():
                    text = self._format_record_text(sheet, record)
                    if text:
                        selected.append((sheet, record, text))
                        seen_labels.add(label.lower())
                        break

        if not selected:
            for record in sheet.records:
                if record.get("type") != "question":
                    continue
                label = str(record.get("label") or "").strip()
                if not label or label.lower() in seen_labels:
                    continue
                text = self._format_record_text(sheet, record)
                if text:
                    selected.append((sheet, record, text))
                    seen_labels.add(label.lower())
                if len(selected) >= 5:
                    break

        return selected[:5]

    def _state_coverage_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for slug in self.catalog.module_index.get("subnational", []):
            sheet = self.catalog.get_sheet(slug)
            if not sheet:
                continue
            responses = [record.get("response") for record in sheet.records if record.get("response")]
            counts = count_yes_no(responses)
            total = len(sheet.records)
            coverage = round((counts["yes"] / total) * 100, 2) if total else 0.0
            rows.append(
                {
                    "state": sheet.metadata.get("state_name") or sheet.title,
                    "yes": counts["yes"],
                    "no": counts["no"],
                    "other": counts["other"],
                    "coverage_percent": coverage,
                }
            )
        return rows

    def _module_artifact(self) -> Optional[ArtifactPayload]:
        modules = self.metadata.get("module_summary", [])
        if not modules:
            return None
        return ArtifactPayload(
            id="lse-modules-table",
            type="table",
            title="Module coverage in the NDC Align catalog",
            data={
                "columns": ["module", "group", "tabs", "records"],
                "rows": modules,
            },
        )

    def _state_artifact(self, query: str) -> Optional[ArtifactPayload]:
        lowered = query.lower()
        if not any(keyword in lowered for keyword in ["state", "subnational", "municip", "rio", "sao", "amazon"]):
            return None
        rows = self._state_coverage_rows()
        if not rows:
            return None
        return ArtifactPayload(
            id="lse-state-coverage",
            type="table",
            title="State-level governance coverage",
            data={
                "columns": ["state", "yes", "no", "other", "coverage_percent"],
                "rows": rows,
            },
            description="Comparison of subnational governance responses across Brazilian states",
        )

    @staticmethod
    def _suggest_next_actions(query: str, has_results: bool) -> List[str]:
        suggestions: List[str] = []
        lowered = query.lower()
        if "state" in lowered or "subnational" in lowered:
            suggestions.append("GetBrazilianStatesOverview()")
            suggestions.append('CompareBrazilianStates(states=["Para", "Maranhao"])')
        if "ndc" in lowered or "commitment" in lowered:
            suggestions.append("GetNDCTargets(country=\"Brazil\")")
            suggestions.append("GetNDCPolicyComparison()")
        if not suggestions or not has_results:
            suggestions.append("ListLSEGroups()")
            suggestions.append("SearchLSEContent(search_term=\"policy\", limit=10)")
        return suggestions[:5]

    # ------------------------------------------------------------------ run_query implementation
    def handle_run_query(self, *, query: str, context: dict) -> RunQueryResponse:
        start_time = time.perf_counter()
        facts: List[FactPayload] = []
        citations: List[CitationPayload] = []
        artifacts: List[ArtifactPayload] = []
        messages: List[MessagePayload] = []
        kg_nodes: List[Dict[str, Any]] = []
        kg_edges: List[Dict[str, Any]] = []
        next_actions: List[str] = []

        if not self.catalog.sheets:
            messages.append(
                MessagePayload(
                    level="error",
                    text="LSE data is unavailable. Ensure data/lse workbooks are present on the server.",
                )
            )
            return RunQueryResponse(
                server="lse",
                query=query,
                facts=facts,
                citations=citations,
                artifacts=artifacts,
                messages=messages,
                kg=KnowledgeGraphPayload(nodes=kg_nodes, edges=kg_edges),
                next_actions=next_actions,
                duration_ms=int((time.perf_counter() - start_time) * 1000),
            )

        dataset_citation = self._dataset_citation()
        self._ensure_citation(citations, dataset_citation)

        module_summary = self.metadata.get("module_summary", [])
        lowered_query = query.lower()
        include_overview = any(
            keyword in lowered_query
            for keyword in [
                "dataset",
                "module",
                "catalog",
                "coverage",
                "overview",
                "this project",
                "this app",
                "this assistant",
                "what can you do",
                "capabilities",
                "how does this work",
                "what data do you use",
                "what datasets do you use",
                "what sources do you have",
            ]
        )

        if include_overview:
            module_names = [
                summary.get("module") for summary in module_summary if summary.get("module")
            ]
            module_text = ", ".join(module_names[:4])
            if len(module_names) > 4:
                module_text += " and others"

            facts.append(
                FactPayload(
                    id="lse-overview",
                    text=(
                        "NDC Align catalogues Brazil's climate governance, covering "
                        f"{len(module_summary)} modules ({module_text}) with {self.metadata.get('records', 0)} "
                        "normalized records drawn from official policies and institutional documentation."
                    ),
                    citation_id=dataset_citation.id,
                    kind="text",
                    metadata={"modules": module_summary},
                )
            )

        search_results = self.catalog.search(query, limit=5)
        results = search_results.get("results", [])
        citation_lookup: Dict[str, str] = {}
        seen_record_keys: Set[Tuple[str, str]] = set()
        for idx, item in enumerate(results, start=1):
            slug = item.get("slug")
            sheet = self.catalog.get_sheet(slug) if slug else None
            if not sheet:
                continue
            record = item.get("record", {})
            fact_text = self._summarize_record(sheet, record)
            if not fact_text:
                continue
            label_key = (
                sheet.slug,
                (str(record.get("label") or record.get("question") or "").strip().lower()),
            )
            seen_record_keys.add(label_key)
            citation_id = self._citation_for_record(sheet, record, citations, citation_lookup)
            if not citation_id:
                citation_id = dataset_citation.id
            facts.append(
                FactPayload(
                    id=f"lse-record-{idx}",
                    text=fact_text,
                    citation_id=citation_id,
                    kind="text",
                    metadata={
                        "slug": sheet.slug,
                        "module": sheet.module,
                        "group": sheet.group,
                        "question": record.get("label") or record.get("question"),
                        "record": record,
                    },
                )
            )

        ndc_focus = self._ndc_focus_records(query)
        for sheet, record, text in ndc_focus:
            label_key = (
                sheet.slug,
                (str(record.get("label") or record.get("question") or "").strip().lower()),
            )
            if label_key in seen_record_keys:
                continue
            citation_id = self._citation_for_record(sheet, record, citations, citation_lookup)
            if not citation_id:
                citation_id = dataset_citation.id
            facts.append(
                FactPayload(
                    id=f"lse-ndc-{len(facts) + 1}",
                    text=text,
                    citation_id=citation_id,
                    kind="text",
                    metadata={
                        "slug": sheet.slug,
                        "module": sheet.module,
                        "group": sheet.group,
                        "question": record.get("label") or record.get("question"),
                        "record": record,
                    },
                )
            )
            seen_record_keys.add(label_key)

        if not results:
            messages.append(
                MessagePayload(
                    level="info",
                    text=(
                        "No direct record matches were found. Try focusing on specific NDC topics, "
                        "policy areas, or mention a Brazilian state to tap into subnational data."
                    ),
                )
            )

        if include_overview:
            module_artifact = self._module_artifact()
            if module_artifact:
                artifacts.append(module_artifact)
        state_artifact = self._state_artifact(query)
        if state_artifact:
            artifacts.append(state_artifact)

        if not facts:
            ndc_sheet = self.catalog.get_ndc_sheet()
            if ndc_sheet:
                for record in ndc_sheet.records:
                    if record.get("type") != "question":
                        continue
                    text = self._format_record_text(ndc_sheet, record)
                    if not text:
                        continue
                    facts.append(
                        FactPayload(
                            id=f"lse-ndc-fallback-{len(facts) + 1}",
                            text=text,
                            citation_id=dataset_citation.id,
                            kind="text",
                            metadata={
                                "slug": ndc_sheet.slug,
                                "module": ndc_sheet.module,
                                "group": ndc_sheet.group,
                                "question": record.get("label") or record.get("question"),
                                "record": record,
                            },
                        )
                    )
                    if len(facts) >= 2:
                        break

        if not facts:
            for slug in self.catalog.module_index.get("plans_policies", [])[:2]:
                sheet = self.catalog.get_sheet(slug)
                if not sheet:
                    continue
                for record in sheet.records:
                    if record.get("type") == "section":
                        continue
                    text = self._format_record_text(sheet, record)
                    if not text:
                        continue
                    facts.append(
                        FactPayload(
                            id=f"lse-policy-fallback-{len(facts) + 1}",
                            text=text,
                            citation_id=dataset_citation.id,
                            kind="text",
                            metadata={
                                "slug": sheet.slug,
                                "module": sheet.module,
                                "group": sheet.group,
                                "question": record.get("label") or record.get("question"),
                                "record": record,
                            },
                        )
                    )
                    if len(facts) >= 2:
                        break
                if facts:
                    break

        kg_nodes.append(
            {
                "id": "dataset:lse",
                "name": DATASET_INFO.get("title", "NDC Align"),
                "type": "dataset",
            }
        )
        for summary in module_summary:
            module_name = summary.get("module")
            if not module_name:
                continue
            node_id = f"module:{module_name}"
            kg_nodes.append(
                {
                    "id": node_id,
                    "name": module_name,
                    "type": "module",
                    "group": summary.get("group"),
                }
            )
            kg_edges.append(
                {
                    "source": "dataset:lse",
                    "target": node_id,
                    "type": "contains",
                }
            )

        next_actions.extend(self._suggest_next_actions(query, bool(results)))

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return RunQueryResponse(
            server="lse",
            query=query,
            facts=facts,
            citations=citations,
            artifacts=artifacts,
            messages=messages,
            kg=KnowledgeGraphPayload(nodes=kg_nodes, edges=kg_edges),
            next_actions=next_actions,
            duration_ms=duration_ms,
        )


def create_server() -> FastMCP:
    """Factory used by the orchestrator to spawn the MCP server."""

    server = LSEServerV2()
    return server.mcp


if __name__ == "__main__":  # pragma: no cover - manual execution
    create_server().run()
