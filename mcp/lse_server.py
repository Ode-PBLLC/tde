import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
import typing as _typing
from collections import defaultdict

import numpy as np
import pandas as pd
from fastmcp import FastMCP

# =============================================================================
# Paths and globals
# =============================================================================

mcp = FastMCP("lse-server")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "lse"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "lse_processed"

Any = _typing.Any  # ensure `Any` survives forward-ref evaluation when run as __main__

GROUP_LABELS = {
    "ndc_overview": "national_commitments",
    "institutions": "governance_processes",
    "plans_policies": "policy_frameworks",
    "subnational": "brazilian_states",
    "tpi_graphs": "transition_pathways",
}

# =============================================================================
# Utility helpers
# =============================================================================


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


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Sanitize each value in a record."""
    return {key: sanitize_value(val) for key, val in record.items()}


def collect_sources(row: dict[str, Any]) -> list[dict[str, Optional[str]]]:
    """Collect primary/secondary/tertiary sources into a uniform list."""
    sources: list[dict[str, Optional[str]]] = []
    for prefix in ("primary", "secondary", "tertiary", "quaternary"):
        src = sanitize_value(row.get(f"{prefix}_source"))
        src_type = sanitize_value(row.get(f"{prefix}_source_type"))
        if src or src_type:
            sources.append({"kind": prefix, "source": src, "source_type": src_type})
    return sources


def count_yes_no(responses: Iterable[str]) -> dict[str, int]:
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


def build_snippet(record: dict[str, Any], term: str) -> Optional[str]:
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


def drop_empty_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove records that are entirely empty."""
    cleaned: list[dict[str, Any]] = []
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


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class ProcessedSheet:
    """Normalized representation of a single Excel sheet."""

    module: str
    group: str
    source_file: str
    sheet_name: str
    slug: str
    title: str
    columns: list[str]
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_records: bool = True) -> dict[str, Any]:
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
    patterns: tuple[str, ...]
    parser: Callable[[Path, "ModuleSpec"], list[ProcessedSheet]]
    group: str
    exclude_sheets: tuple[str, ...] = ()
    friendly_name: Optional[str] = None


# =============================================================================
# Module parsers
# =============================================================================


def parse_ndc_workbook(path: Path, spec: ModuleSpec) -> list[ProcessedSheet]:
    """Parse the NDC overview workbook into a single structured sheet."""
    sheets: list[ProcessedSheet] = []
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
    rename_map: dict[str, str] = {}
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

    records: list[dict[str, Any]] = []
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
        title=spec.friendly_name or "NDC Overview & Domestic Policy Comparison",
        columns=["type", "label", "section", "ndc_answer", "ndc_summary", "domestic_summary"],
        records=records,
        summary=summary,
        metadata={"header_row": header_idx, "question_count": question_count},
    )
    sheets.append(sheet)
    return sheets


def rename_columns_with_keywords(columns: Sequence[str], module: str) -> list[str]:
    """Rename columns using keyword heuristics."""
    aliases = [col.strip() for col in columns]
    mapping: list[tuple[str, str]] = [
        ("pergunta", "question"),
        ("question", "question"),
        ("indicator", "indicator"),
        ("questions", "question"),
        ("resposta","response"),
        ("response","response"),
        ("status","status"),
        ("summary","summary"),
        ("justification","justification"),
        ("implementation information","implementation_information"),
        ("implementation","implementation_information"),
        ("notes on and source for implementation information","implementation_notes"),
        ("primary source for answer","primary_source"),
        ("primary source type","primary_source_type"),
        ("second source for answer","secondary_source"),
        ("second source type","secondary_source_type"),
        ("third source for answer","tertiary_source"),
        ("third source type","tertiary_source_type"),
        ("fourth source","quaternary_source"),
        ("adaptation","adaptation_mitigation"),
        ("mitigation","adaptation_mitigation"),
        ("sector","sector"),
        ("implementation stage","implementation_stage"),
    ]

    renamed: list[str] = []
    counters: dict[str, int] = defaultdict(int)

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


def parse_generic_module(path: Path, spec: ModuleSpec) -> list[ProcessedSheet]:
    """Parse institutions or plans/policies workbooks."""
    excel = pd.ExcelFile(path)
    sheets: list[ProcessedSheet] = []

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


def parse_subnational_workbook(path: Path, spec: ModuleSpec) -> list[ProcessedSheet]:
    """Parse Brazilian subnational governance workbook."""
    excel = pd.ExcelFile(path)
    sheets: list[ProcessedSheet] = []

    for sheet_name in excel.sheet_names:
        if sheet_name in spec.exclude_sheets:
            continue

        df = excel.parse(sheet_name).dropna(how="all")
        if df.empty:
            continue

        original_columns = list(df.columns)
        new_columns: list[Optional[str]] = []
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


def parse_tpi_workbook(path: Path, spec: ModuleSpec) -> list[ProcessedSheet]:
    """Parse the TPI graph workbook."""
    excel = pd.ExcelFile(path)
    sheets: list[ProcessedSheet] = []

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


MODULE_SPECS: tuple[ModuleSpec, ...] = (
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


# =============================================================================
# Catalog builder
# =============================================================================


class LSECatalog:
    """Load, normalize, and index the LSE workbooks."""

    def __init__(self, raw_dir: Path, processed_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.sheets: dict[str, ProcessedSheet] = {}
        self.module_index: dict[str, list[str]] = defaultdict(list)
        self.group_index: dict[str, list[str]] = defaultdict(list)
        self.metadata: dict[str, Any] = {}
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

    # ---------------------------------------------------------------------
    # Access helpers
    # ---------------------------------------------------------------------

    def list_groups(self) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
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
    ) -> dict[str, Any]:
        term_lower = term.lower()
        results: list[dict[str, Any]] = []
        for slug, sheet in self.sheets.items():
            if group and sheet.group != group:
                continue
            if module and sheet.module != module:
                continue
            for record in sheet.records:
                haystack_parts: list[str] = []
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


# Initialize catalog
catalog = LSECatalog(RAW_DATA_DIR, PROCESSED_DATA_DIR)


# =============================================================================
# Tool definitions
# =============================================================================


@mcp.tool()
def ListLSEGroups() -> dict[str, Any]:
    """List dataset groups and their module coverage."""
    if not catalog.sheets:
        return {"error": "LSE data not available"}
    return catalog.list_groups()


@mcp.tool()
def ListLSETabs(
    group: Optional[str] = None,
    module: Optional[str] = None,
    include_preview: bool = False,
) -> dict[str, Any]:
    """List normalized tabs with optional filters."""
    if not catalog.sheets:
        return {"error": "LSE data not available"}
    return catalog.list_tabs(group=group, module=module, include_preview=include_preview)


@mcp.tool()
def GetLSETab(slug: str, include_records: bool = True) -> dict[str, Any]:
    """Fetch a specific tab by slug."""
    sheet = catalog.get_sheet(slug)
    if not sheet:
        return {"error": f"Tab '{slug}' not found"}
    return sheet.to_dict(include_records=include_records)


@mcp.tool()
def GetTPIGraphData() -> dict[str, Any]:
    """Return Transition Pathway Initiative emissions data."""
    slugs = catalog.module_index.get("tpi_graphs", [])
    if not slugs:
        return {"error": "TPI graph data not available"}
    sheet = catalog.get_sheet(slugs[0])
    if not sheet:
        return {"error": "TPI graph data not available"}
    return {
        "description": "Transition Pathway Initiative pathways for Brazil",
        "columns": sheet.columns,
        "summary": sheet.summary,
        "records": sheet.records,
    }


@mcp.tool()
def GetInstitutionalFramework(topic: Optional[str] = None) -> dict[str, Any]:
    """Fetch institutional governance entries, optionally filtered by topic."""
    slugs = catalog.module_index.get("institutions", [])
    if not slugs:
        return {"error": "Institutional data not available"}

    matches = []
    available_topics = []
    for slug in slugs:
        sheet = catalog.get_sheet(slug)
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


@mcp.tool()
def GetClimatePolicy(policy_type: Optional[str] = None) -> dict[str, Any]:
    """Retrieve climate plans and policies entries by policy type."""
    slugs = catalog.module_index.get("plans_policies", [])
    if not slugs:
        return {"error": "Plans and policies data not available"}

    matches = []
    available_types = []
    for slug in slugs:
        sheet = catalog.get_sheet(slug)
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
    }


@mcp.tool()
def GetSubnationalGovernance(
    state: Optional[str] = None,
    metric: Optional[str] = None,
) -> dict[str, Any]:
    """Retrieve Brazilian subnational governance data."""
    slugs = catalog.module_index.get("subnational", [])
    if not slugs:
        return {"error": "Subnational data not available"}

    if state:
        sheet = catalog.get_sheet_by_state(state)
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
        }

    overview = []
    for slug in slugs:
        sheet = catalog.get_sheet(slug)
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
    }


@mcp.tool()
def GetLSEDatasetOverview() -> dict[str, Any]:
    """Provide high-level dataset overview."""
    if not catalog.sheets:
        return {"error": "LSE data not available"}
    return catalog.metadata


@mcp.tool()
def GetLSEFileStructure(filename: str) -> dict[str, Any]:
    """Describe processed tabs originating from a given workbook."""
    matches = [sheet for sheet in catalog.sheets.values() if sheet.source_file == filename]
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


@mcp.tool()
def SearchLSEContent(
    search_term: str,
    module_type: Optional[str] = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Search across normalized LSE content."""
    if not catalog.sheets:
        return {"error": "LSE data not available"}
    result = catalog.search(search_term, module=module_type, limit=limit)
    if not result.get("results"):
        result["guidance"] = (
            "No direct matches found. Try broader keywords or explore modules via ListLSEGroups."
        )
    return result


@mcp.tool()
def GetBrazilianStatesOverview() -> dict[str, Any]:
    """List available Brazilian states with summarized metrics."""
    slugs = catalog.module_index.get("subnational", [])
    if not slugs:
        return {"error": "Subnational data not available"}
    overview = []
    for slug in slugs:
        sheet = catalog.get_sheet(slug)
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


@mcp.tool()
def GetStateClimatePolicy(state_name: str) -> dict[str, Any]:
    """Retrieve the full policy table for a specific Brazilian state."""
    sheet = catalog.get_sheet_by_state(state_name)
    if not sheet:
        return {"error": f"State '{state_name}' not found"}
    return {
        "state": sheet.metadata.get("state_name") or sheet.title,
        "state_code": sheet.metadata.get("state_code"),
        "slug": sheet.slug,
        "summary": sheet.summary,
        "records": sheet.records,
    }


@mcp.tool()
def CompareBrazilianStates(
    states: list[str],
    policy_area: Optional[str] = None,
) -> dict[str, Any]:
    """Compare policy coverage across multiple Brazilian states."""
    comparison = {
        "states": [],
        "policy_area": policy_area,
        "metrics": {},
    }

    for state in states:
        sheet = catalog.get_sheet_by_state(state)
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
        comparison["states"].append(sheet.metadata.get("state_name") or sheet.title)
        comparison["metrics"][sheet.metadata.get("state_name") or sheet.title] = {
            "questions": len(records),
            "yes": counts["yes"],
            "no": counts["no"],
            "other": counts["other"],
            "coverage_percent": coverage,
        }

    if not comparison["states"]:
        comparison["error"] = "No states found for comparison"
    return comparison


# -----------------------------------------------------------------------------
# NDC-focused tools
# -----------------------------------------------------------------------------


def _ndc_records() -> list[dict[str, Any]]:
    sheet = catalog.get_ndc_sheet()
    if not sheet:
        return []
    return sheet.records


@mcp.tool()
def GetNDCTargets(country: str = "Brazil") -> dict[str, Any]:
    """Extract key NDC targets and commitments."""
    records = _ndc_records()
    if not records:
        return {"error": "NDC data not available"}

    targets: dict[str, Any] = {
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
    return targets


@mcp.tool()
def GetNDCPolicyComparison() -> dict[str, Any]:
    """Compare NDC commitments with domestic policy responses."""
    records = _ndc_records()
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

    return comparison


@mcp.tool()
def GetNDCImplementationStatus(country: str = "Brazil") -> dict[str, Any]:
    """Summarize implementation evidence for NDC commitments."""
    records = _ndc_records()
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

        if any(keyword in domestic_summary for keyword in ("Law", "Decree", "Resolution", "Ordinance")):
            instruments.append(entry)

    total_targets = len(implemented) + len(pending)
    implementation_rate = round((len(implemented) / total_targets) * 100, 2) if total_targets else 0.0

    return {
        "country": country,
        "implemented_targets": implemented,
        "pending_targets": pending,
        "implementation_rate_percent": implementation_rate,
        "instruments": instruments,
    }


@mcp.tool()
def GetAllNDCData() -> dict[str, Any]:
    """Return the full normalized NDC overview dataset."""
    sheet = catalog.get_ndc_sheet()
    if not sheet:
        return {"error": "NDC data not available"}
    return sheet.to_dict(include_records=True)


@mcp.tool()
def GetNDCOverviewData(country: Optional[str] = None) -> dict[str, Any]:
    """Provide NDC overview summary for the given country."""
    sheet = catalog.get_ndc_sheet()
    if not sheet:
        return {"error": "NDC data not available"}
    summary = sheet.summary.copy()
    if country:
        summary["country"] = country
    return {
        "metadata": sheet.metadata,
        "summary": summary,
        "records": sheet.records,
    }


@mcp.tool()
def GetInstitutionsProcessesData() -> dict[str, Any]:
    """Return all institutions & processes tabs."""
    slugs = catalog.module_index.get("institutions", [])
    if not slugs:
        return {"error": "Institutional data not available"}
    tabs = [catalog.get_sheet(slug).to_dict(include_records=True) for slug in slugs if catalog.get_sheet(slug)]
    return {
        "tabs": tabs,
        "count": len(tabs),
    }


@mcp.tool()
def GetPlansAndPoliciesData() -> dict[str, Any]:
    """Return all plans & policies tabs."""
    slugs = catalog.module_index.get("plans_policies", [])
    if not slugs:
        return {"error": "Plans and policies data not available"}
    tabs = [catalog.get_sheet(slug).to_dict(include_records=True) for slug in slugs if catalog.get_sheet(slug)]
    return {
        "tabs": tabs,
        "count": len(tabs),
    }


@mcp.tool()
def GetLSEVisualizationData(viz_type: str, filters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Provide pre-aggregated data for simple visualizations."""
    if not catalog.sheets:
        return {"error": "LSE data not available"}
    filters = filters or {}
    viz_type_lower = viz_type.lower()

    if viz_type_lower == "states_comparison":
        slugs = catalog.module_index.get("subnational", [])
        data = []
        for slug in slugs:
            sheet = catalog.get_sheet(slug)
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
            "data": catalog.metadata.get("module_summary", []),
            "chart": {"type": "bar", "x": "module", "y": "tabs"},
        }

    if viz_type_lower == "policy_coverage":
        modules = []
        for module, slugs in catalog.module_index.items():
            modules.append(
                {
                    "module": module,
                    "records": sum(len(catalog.get_sheet(slug).records) for slug in slugs if catalog.get_sheet(slug)),
                    "tabs": len(slugs),
                }
            )
        return {
            "visualization": "policy_coverage",
            "data": modules,
            "chart": {"type": "bar", "x": "module", "y": "records"},
        }

    if viz_type_lower == "governance_status":
        status_counts: dict[str, int] = defaultdict(int)
        for slug in catalog.module_index.get("subnational", []):
            sheet = catalog.get_sheet(slug)
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


@mcp.tool()
def GetLSEDatasetMetadata() -> dict[str, Any]:
    """Expose raw metadata for the normalized dataset."""
    return catalog.metadata


@mcp.tool()
def DescribeServer() -> dict[str, Any]:
    """Describe the server, modules, and tooling."""
    if not catalog.sheets:
        return {"error": "LSE data not available"}
    metadata = catalog.metadata.copy()
    metadata["tools"] = [
        "ListLSEGroups",
        "ListLSETabs",
        "SearchLSEContent",
        "GetLSETab",
        "GetTPIGraphData",
        "GetSubnationalGovernance",
    ]
    return metadata


if __name__ == "__main__":
    mcp.run()
