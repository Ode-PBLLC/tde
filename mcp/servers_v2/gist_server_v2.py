"""GIST dataset MCP server implementing the v2 `run_query` contract.

This module ports the legacy GIST FastMCP tools into the v2 contract so the
server can provide structured responses alongside the existing tool surface.
All tool implementations are inlined here (no runtime dependency on the
legacy server module) while reusing the shared data loading utilities.
"""

import json
import os
import re
import sys
import time
from collections.abc import Mapping
from datetime import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastmcp import FastMCP

try:  # pragma: no cover - optional dependency
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

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
else:  # pragma: no cover - package execution
    from ..contracts_v2 import (
        ArtifactPayload,
        CitationPayload,
        FactPayload,
        KnowledgeGraphPayload,
        MessagePayload,
        RunQueryResponse,
    )
    from ..servers_v2.base import RunQueryMixin
    from ..support_intent import SupportIntent

from utils.llm_retry import call_llm_with_retries_sync

load_dotenv(ROOT / ".env")

GIST_FILE_PATH = ROOT / "data" / "gist" / "gist.xlsx"
DATASET_TITLE = "GIST Impact Datasets"
DATASET_SOURCE = "GIST Environmental Research"
DATASET_NODE_ID = "GIST_DATASET"
DATASET_ID = "gist_multi_dataset"


def _load_dataset_citations() -> Dict[str, str]:
    path = ROOT / "static" / "meta" / "datasets.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    mapping: Dict[str, str] = {}
    for item in payload.get("items", []):
        dataset_id = item.get("id")
        citation = item.get("citation")
        if dataset_id and citation:
            mapping[str(dataset_id)] = str(citation)
    return mapping


DATASET_CITATIONS = _load_dataset_citations()


def _dataset_citation(dataset_id: str) -> Optional[str]:
    return DATASET_CITATIONS.get(dataset_id)


class GistDataManager:
    """Load and cache GIST Excel sheets for tool access."""

    def __init__(self) -> None:
        self.excel_path = self._resolve_excel_path()
        self.excel_file: Optional[pd.ExcelFile] = None
        self.sheets: Dict[str, pd.DataFrame] = {}
        self.companies_cache: Dict[str, Dict[str, Any]] = {}
        self.asset_cache: Dict[str, Any] = {}
        self._load_data()

    def _resolve_excel_path(self) -> Optional[Path]:
        candidates = [
            GIST_FILE_PATH,
            Path.cwd() / "data" / "gist" / "gist.xlsx",
            Path.cwd().parent / "data" / "gist" / "gist.xlsx",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_data(self) -> None:
        if not self.excel_path or not self.excel_path.exists():
            self.sheets = {}
            self.companies_cache = {}
            self.asset_cache = {}
            return

        try:
            self.excel_file = pd.ExcelFile(self.excel_path)
            for sheet_name in self.excel_file.sheet_names:
                self.sheets[sheet_name] = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            self._build_company_index()
            self._build_asset_index()
        except Exception:
            self.sheets = {}
            self.companies_cache = {}
            self.asset_cache = {}

    def _build_company_index(self) -> None:
        self.companies_cache = {}
        for sheet_name, df in self.sheets.items():
            if sheet_name == "Data Dictionary" or "COMPANY_CODE" not in df.columns:
                continue
            for _, row in df.iterrows():
                company_code = row.get("COMPANY_CODE")
                if pd.isna(company_code):
                    continue
                company_code = str(company_code)
                company_name = row.get("COMPANY_NAME", "Unknown")
                entry = self.companies_cache.setdefault(
                    company_code,
                    {
                        "company_code": company_code,
                        "company_name": company_name,
                        "datasets": [],
                        "sector_code": row.get("SECTOR_CODE", "Unknown"),
                        "country": row.get("COUNTRY_NAME", row.get("COUNTRY_CODE", "Unknown")),
                    },
                )
                if sheet_name not in entry["datasets"]:
                    entry["datasets"].append(sheet_name)

    def _build_asset_index(self) -> None:
        if "EXSITU_ASSET_DATA" not in self.sheets:
            self.asset_cache = {}
            return
        asset_df = self.sheets["EXSITU_ASSET_DATA"]
        self.asset_cache = {
            "by_company": asset_df.groupby("COMPANY_CODE").size().to_dict(),
            "by_country": asset_df.groupby("COUNTRY_CODE").size().to_dict(),
            "total_assets": int(len(asset_df)),
        }

    def get_sheet(self, sheet_name: str) -> pd.DataFrame:
        return self.sheets.get(sheet_name, pd.DataFrame())

    def get_companies(
        self, *, sector: Optional[str] = None, country: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        companies = list(self.companies_cache.values())
        if sector:
            companies = [c for c in companies if c.get("sector_code") == sector]
        if country:
            companies = [c for c in companies if country.lower() in str(c.get("country", "")).lower()]
        return companies

    def get_company_data(self, company_code: str) -> Dict[str, Any]:
        company_data: Dict[str, Any] = {"company_code": company_code, "datasets": {}}
        for sheet_name, df in self.sheets.items():
            if sheet_name == "Data Dictionary" or "COMPANY_CODE" not in df.columns:
                continue
            company_rows = df[df["COMPANY_CODE"] == company_code]
            if not company_rows.empty:
                company_data["datasets"][sheet_name] = company_rows.to_dict("records")
        return company_data


class GistServerV2(RunQueryMixin):
    """FastMCP server that exposes GIST tools and structured run_query."""

    RISK_SYNONYM_MAP = {
        "msa": "MSA",
        "mean species abundance": "MSA",
        "biodiversity": "MSA",
        "water stress": "WATER_STRESS",
        "water demand": "WATER_DEMAND",
        "water variability": "WATER_VARIABILITY",
        "drought": "DROUGHT",
        "riverine flood": "FLOOD_RIVERINE",
        "riverine flooding": "FLOOD_RIVERINE",
        "flood": "FLOOD_RIVERINE",
        "flooding": "FLOOD_RIVERINE",
        "coastal flood": "FLOOD_COASTAL",
        "coastal flooding": "FLOOD_COASTAL",
        "extreme heat": "EXTREME_HEAT",
        "heat stress": "EXTREME_HEAT",
        "extreme precipitation": "EXTREME_PRECIPITATION",
        "precipitation": "EXTREME_PRECIPITATION",
        "temperature anomaly": "TEMPERATURE_ANOMALY",
        "temperature": "TEMPERATURE_ANOMALY",
        "urban": "URBAN_AREA_CHANGE",
        "urban change": "URBAN_AREA_CHANGE",
        "agriculture": "AGRICULTURE_AREA_CHANGE",
        "agricultural": "AGRICULTURE_AREA_CHANGE",
        "forest": "FOREST_AREA_CHANGE",
        "deforestation": "FOREST_AREA_CHANGE",
    }

    def __init__(self) -> None:
        self.mcp = FastMCP("gist-server-v2")
        self.data_manager = GistDataManager()
        self.metadata = self._build_metadata()

        self._anthropic_client = None
        if anthropic and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self._anthropic_client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover - credential issues
                print(f"[gist-server] Warning: Anthropic client unavailable: {exc}")

        self._openai_client = None
        openai_key = os.getenv("OPENAI_API_KEY")
        if OpenAI and openai_key:
            try:
                self._openai_client = OpenAI(api_key=openai_key)
            except Exception as exc:  # pragma: no cover - credential issues
                print(f"[gist-server] Warning: OpenAI client unavailable: {exc}")

        self._register_capabilities_tool()
        self._register_query_support_tool()
        self._register_data_dictionary_tools()
        self._register_company_tools()
        self._register_risk_and_geospatial_tools()
        self._register_emissions_tools()
        self._register_biodiversity_tools()
        self._register_deforestation_tools()
        self._register_visualization_tools()
        self._register_metadata_tools()
        self._register_run_query_tool()

    # ------------------------------------------------------------------ metadata helpers
    def _capabilities_metadata(self) -> Dict[str, Any]:
        return {
            "name": "gist",
            "description": "Corporate environmental datasets covering risk exposures, asset-level geospatial data, biodiversity impacts, deforestation risk, and Scope 3 emissions.",
            "version": "2.0.0",
            "tags": [
                "corporate",
                "risk",
                "biodiversity",
                "scope 3",
                "geospatial",
                "deforestation",
            ],
            "dataset": DATASET_TITLE,
            "url": "https://www.gistimpact.com/",
            "tables": list(self.metadata.get("Datasets", [])),
            "tools": [
                "describe_capabilities",
                "query_support",
                "GetGistDataDictionary",
                "SearchGistFields",
                "GetGistDatasetSchemas",
                "GetGistCompanies",
                "GetGistCompanyProfile",
                "GetGistCompanyRiskSummary",
                "GetGistCompanyRiskDetail",
                "GetGistScope3Summary",
                "GetGistScope3Timeseries",
                "GetGistBiodiversitySummary",
                "GetGistBiodiversityTimeSeries",
                "GetGistDeforestationSummary",
                "GetGistDeforestationAssets",
                "GetGistExtremeHeatSummary",
                "GetGistWaterStressSummary",
                "GetGistAssetLookup",
                "GetGistCompanyAssets",
                "run_query",
            ],
        }

    def _capability_summary(self) -> str:
        metadata = self._capabilities_metadata()
        return (
            f"Dataset: {metadata['dataset']} ({metadata['description']}). "
            "Supports company rankings for hazards (e.g., flood, heat, drought) via GetGistRiskByCategory plus deep single-company risk, emissions, and asset profiles."
        )

    def _classify_support(self, query: str) -> SupportIntent:
        # Prefer Anthropic, fall back to OpenAI, otherwise allow by default.
        if self._anthropic_client:
            try:
                prompt = (
                    "Decide whether the GIST corporate environmental dataset should answer the question."
                    " Respond strictly with JSON containing keys 'supported' (true/false) and 'reason' (short explanation).\n"
                    f"Dataset capabilities: {self._capability_summary()}\n"
                    f"Question: {query}"
                )
                response = call_llm_with_retries_sync(
                    lambda: self._anthropic_client.messages.create(
                        model="claude-3-5-haiku-20241022",
                        max_tokens=128,
                        temperature=0,
                        system="Respond with valid JSON only.",
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    provider="anthropic.gist_router",
                )
                text = response.content[0].text.strip()
                intent = self._parse_support_intent(text)
                if intent:
                    return intent
            except Exception as exc:  # pragma: no cover
                return SupportIntent(
                    supported=True,
                    score=0.3,
                    reasons=[f"Anthropic intent unavailable: {exc}"],
                )

        if self._openai_client:
            try:
                prompt = (
                    "Decide whether the GIST corporate environmental dataset should answer the question.\n"
                    "Respond strictly with JSON containing keys 'supported' (true/false) and 'reason' (short explanation).\n"
                    f"Dataset capabilities: {self._capability_summary()}\n"
                    f"Question: {query}"
                )
                response = call_llm_with_retries_sync(
                    lambda: self._openai_client.responses.create(
                        model=os.getenv("GIST_ROUTER_MODEL", "gpt-4.1-mini"),
                        input=prompt,
                        temperature=0,
                        max_output_tokens=128,
                    ),
                    provider="openai.gist_router",
                )
                text = "".join(part.text for part in response.output if hasattr(part, "text"))
                intent = self._parse_support_intent(text)
                if intent:
                    return intent
            except Exception as exc:  # pragma: no cover
                return SupportIntent(
                    supported=True,
                    score=0.3,
                    reasons=[f"OpenAI intent unavailable: {exc}"],
                )

        return SupportIntent(
            supported=True,
            score=0.3,
            reasons=["LLM unavailable; defaulting to dataset summary"],
        )

    @staticmethod
    def _parse_support_intent(text: str) -> Optional[SupportIntent]:
        def _parse(blob: str) -> Optional[Dict[str, Any]]:
            try:
                payload = json.loads(blob)
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return None

        data = _parse(text)
        if not data:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and start < end:
                data = _parse(text[start : end + 1])
        if not data:
            return None

        supported = bool(data.get("supported", False))
        reason = str(data.get("reason")) if data.get("reason") else None
        score = 0.9 if supported else 0.1
        reasons = [reason] if reason else ["LLM classification"]
        return SupportIntent(supported=supported, score=score, reasons=reasons)

    def _build_metadata(self) -> Dict[str, Any]:
        total_companies = len(self.data_manager.companies_cache)
        total_assets = self.data_manager.asset_cache.get("total_assets", 0)
        datasets = list(self.data_manager.sheets.keys()) if self.data_manager.sheets else []
        return {
            "Name": "GIST Server",
            "Description": "Global Infrastructure Sustainability Toolkit (GIST) data access server",
            "Version": "2.0.0",
            "Author": "Climate Policy Radar Team",
            "Dataset": "GIST Multi-Dataset Collection",
            "Total_Companies": total_companies,
            "Total_Assets": total_assets,
            "Datasets": datasets,
        }

    # ------------------------------------------------------------------ shared helper methods
    @staticmethod
    def _get_dataset_description(sheet_name: str) -> str:
        descriptions = {
            "Data Dictionary": "Field definitions and schema for all datasets",
            "EXSITU": "Company-level aggregated environmental risk data across 75 metrics",
            "EXSITU_ASSET_DATA": "Individual asset-level environmental risk data with coordinates (40K+ assets)",
            "DEFORESTATION": "Company deforestation risk proximity indicators",
            "SCOPE_3_DATA": "Multi-year Scope 3 emissions data by company (2016-2024)",
            "BIODIVERSITY_PDF_DATA": "Multi-year biodiversity impact data with PDF, CO2E, and LCE metrics",
        }
        return descriptions.get(sheet_name, "Dataset information")

    @staticmethod
    def _get_data_summary(dataset_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {}
        if dataset_name in {"SCOPE_3_DATA", "BIODIVERSITY_PDF_DATA"}:
            years = [r.get("REPORTING_YEAR") for r in records if r.get("REPORTING_YEAR")]
            return {"years_available": sorted(list({y for y in years if y}))}
        if dataset_name == "EXSITU_ASSET_DATA":
            countries = [r.get("COUNTRY_CODE") for r in records if r.get("COUNTRY_CODE")]
            return {"countries": sorted(list({c for c in countries if c}))}
        return {"records": len(records)}

    @staticmethod
    def _extract_risk_metrics(row: pd.Series, risk_type: str) -> Dict[str, Any]:
        risk_levels = ["VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH"]
        counts = {}
        for level in risk_levels:
            col_name = f"COUNT_OF_ASSETS_WITH_{level}_{risk_type}"
            counts[level.lower()] = row.get(col_name, 0)
        total = int(sum(counts.values()))
        percentages = {level: (count / total * 100) if total else 0 for level, count in counts.items()}
        return {
            "counts": counts,
            "percentages": percentages,
            "total_assets": total,
            "high_risk_assets": counts.get("high", 0) + counts.get("very_high", 0),
        }

    @staticmethod
    def _calculate_high_risk_summary(
        risk_categories: Dict[str, Dict[str, Any]], total_assets: int
    ) -> Dict[str, Any]:
        summary: Dict[str, Dict[str, Any]] = {}
        for category, data in risk_categories.items():
            high_risk_count = data.get("high_risk_assets", 0)
            high_risk_pct = (high_risk_count / total_assets * 100) if total_assets else 0
            summary[category] = {
                "high_risk_assets": high_risk_count,
                "high_risk_percentage": round(high_risk_pct, 2),
            }
        top = sorted(
            summary.items(), key=lambda item: item[1]["high_risk_percentage"], reverse=True
        )[:5]
        return {
            "by_category": summary,
            "top_risk_categories": [
                {"category": category, **data} for category, data in top
            ],
        }

    # ------------------------------------------------------------------ tool implementations (non-decorated)
    def get_data_dictionary(self, dataset_name: Optional[str]) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        dict_df = self.data_manager.get_sheet("Data Dictionary")
        if dict_df.empty:
            return {"error": "Data Dictionary not found"}
        if dataset_name:
            dict_df = dict_df[dict_df["Dataset"].str.contains(dataset_name, case=False, na=False)]
        result = {"datasets": {}, "total_fields": len(dict_df)}
        for dataset, group in dict_df.groupby("Dataset"):
            if pd.isna(dataset):
                continue
            records = []
            for _, row in group.iterrows():
                records.append(
                    {
                        "field_name": row.get("Field Name"),
                        "unit": row.get("Unit"),
                        "definition": row.get("Definition"),
                    }
                )
            result["datasets"][dataset] = {
                "field_count": len(records),
                "fields": records,
            }
        return result

    def search_fields(self, search_term: str) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        dict_df = self.data_manager.get_sheet("Data Dictionary")
        if dict_df.empty:
            return {"error": "Data Dictionary not found"}
        mask = (
            dict_df["Field Name"].str.contains(search_term, case=False, na=False)
            | dict_df["Definition"].str.contains(search_term, case=False, na=False)
        )
        matches = dict_df[mask]
        results = []
        for _, row in matches.iterrows():
            definition = str(row.get("Definition", ""))
            if len(definition) > 200:
                definition = definition[:200] + "..."
            results.append(
                {
                    "dataset": row.get("Dataset"),
                    "field_name": row.get("Field Name"),
                    "unit": row.get("Unit"),
                    "definition": definition,
                }
            )
        return {
            "search_term": search_term,
            "matches_found": len(results),
            "matches": results,
        }

    def get_dataset_schemas(self) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        payload: Dict[str, Any] = {"total_datasets": len(self.data_manager.sheets), "datasets": {}}
        for sheet_name, df in self.data_manager.sheets.items():
            payload["datasets"][sheet_name] = {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": list(df.columns),
                "description": self._get_dataset_description(sheet_name),
            }
        return payload

    def get_companies(
        self, *, sector: Optional[str], country: Optional[str], limit: int
    ) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        companies = self.data_manager.get_companies(sector=sector, country=country)[:limit]
        return {
            "total_companies": len(self.data_manager.companies_cache),
            "filtered_companies": len(companies),
            "filters_applied": {"sector": sector, "country": country},
            "companies": companies,
        }

    def get_company_profile(self, company_code: str) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        if company_code not in self.data_manager.companies_cache:
            return {"error": f"Company {company_code} not found"}
        company_info = self.data_manager.companies_cache[company_code]
        company_data = self.data_manager.get_company_data(company_code)
        profile = {
            "company_code": company_code,
            "company_name": company_info.get("company_name"),
            "sector_code": company_info.get("sector_code"),
            "country": company_info.get("country"),
            "datasets_available": company_info.get("datasets", []),
            "data_summary": {},
        }
        for dataset, records in company_data["datasets"].items():
            if dataset == "EXSITU_ASSET_DATA":
                profile["data_summary"][dataset] = {
                    "total_assets": len(records),
                    "countries": sorted(list({r.get("COUNTRY_CODE", "Unknown") for r in records})),
                    "asset_types": sorted(list({r.get("ASSET_TYPE_LEVEL_1", "Unknown") for r in records})),
                }
            elif dataset == "SCOPE_3_DATA":
                years = [r.get("REPORTING_YEAR") for r in records if r.get("REPORTING_YEAR")]
                profile["data_summary"][dataset] = {
                    "reporting_years": sorted(list({y for y in years if y})),
                    "total_records": len(records),
                }
            else:
                profile["data_summary"][dataset] = {"total_records": len(records)}
        return profile

    def get_companies_by_sector(self) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        sector_groups: Dict[str, Dict[str, Any]] = {}
        for company in self.data_manager.companies_cache.values():
            sector = company.get("sector_code")
            bucket = sector_groups.setdefault(
                sector,
                {"companies": [], "countries": set(), "total_count": 0},
            )
            bucket["companies"].append(
                {
                    "company_code": company.get("company_code"),
                    "company_name": company.get("company_name"),
                    "country": company.get("country"),
                }
            )
            bucket["countries"].add(company.get("country"))
            bucket["total_count"] += 1
        for sector, data in sector_groups.items():
            data["countries"] = sorted(list(data["countries"]))
            data["companies"] = data["companies"][:10]
        return {"total_sectors": len(sector_groups), "sectors": sector_groups}

    def get_company_data_availability(self, company_code: str) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        if company_code not in self.data_manager.companies_cache:
            return {"error": f"Company {company_code} not found"}
        company_data = self.data_manager.get_company_data(company_code)
        availability: Dict[str, Any] = {
            "company_code": company_code,
            "company_name": self.data_manager.companies_cache[company_code].get("company_name"),
            "data_availability": {},
        }
        for sheet_name in self.data_manager.sheets.keys():
            if sheet_name == "Data Dictionary":
                continue
            records = company_data["datasets"].get(sheet_name)
            if records:
                availability["data_availability"][sheet_name] = {
                    "available": True,
                    "record_count": len(records),
                    "data_summary": self._get_data_summary(sheet_name, records),
                }
            else:
                availability["data_availability"][sheet_name] = {
                    "available": False,
                    "record_count": 0,
                }
        return availability

    def get_company_risks(self, company_code: str) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        exsitu_df = self.data_manager.get_sheet("EXSITU")
        if exsitu_df.empty:
            return {"error": "EXSITU risk data not available"}
        company_data = exsitu_df[exsitu_df["COMPANY_CODE"] == company_code]
        if company_data.empty:
            return {"error": f"No risk data found for company {company_code}"}
        row = company_data.iloc[0]
        risk_categories = {
            "biodiversity": self._extract_risk_metrics(row, "MSA"),
            "water_stress": self._extract_risk_metrics(row, "WATER_STRESS"),
            "water_demand": self._extract_risk_metrics(row, "WATER_DEMAND"),
            "water_variability": self._extract_risk_metrics(row, "WATER_VARIABILITY"),
            "drought": self._extract_risk_metrics(row, "DROUGHT"),
            "flood_coastal": self._extract_risk_metrics(row, "FLOOD_COASTAL"),
            "flood_riverine": self._extract_risk_metrics(row, "FLOOD_RIVERINE"),
            "extreme_heat": self._extract_risk_metrics(row, "EXTREME_HEAT"),
            "extreme_precipitation": self._extract_risk_metrics(row, "EXTREME_PRECIPITATION"),
            "temperature_anomaly": self._extract_risk_metrics(row, "TEMPERATURE_ANOMALY"),
            "urban_area_change": self._extract_risk_metrics(row, "URBAN_AREA_CHANGE"),
            "agriculture_area_change": self._extract_risk_metrics(row, "AGRICULTURE_AREA_CHANGE"),
            "forest_area_change": self._extract_risk_metrics(row, "FOREST_AREA_CHANGE"),
        }
        total_assets = int(row.get("TOTAL_NUMBER_OF_ASSETS_ASSESSED_WITHIN_A_COMPANY", 0))
        return {
            "company_code": company_code,
            "company_name": row.get("COMPANY_NAME", "Unknown"),
            "sector_code": row.get("SECTOR_CODE", "Unknown"),
            "country": row.get("COUNTRY_NAME", "Unknown"),
            "total_assets": total_assets,
            "risk_categories": risk_categories,
            "high_risk_summary": self._calculate_high_risk_summary(risk_categories, total_assets),
        }

    @staticmethod
    def _normalize_risk_keyword(value: str) -> Optional[str]:
        if not value:
            return None
        key = value.strip().lower().replace("-", " ")
        mapping = GistServerV2.RISK_SYNONYM_MAP
        if key in mapping:
            return mapping[key]
        for synonym, canonical in mapping.items():
            if synonym in key:
                return canonical
        return value.upper()

    def get_risk_by_category(
        self,
        risk_type: str,
        risk_level: str,
        limit: int,
        *,
        country: Optional[str] = None,
        sector: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        exsitu_df = self.data_manager.get_sheet("EXSITU")
        if exsitu_df.empty:
            return {"error": "EXSITU risk data not available"}

        normalized = self._normalize_risk_keyword(risk_type)
        if not normalized:
            return {"error": f"Unknown risk type {risk_type}"}
        risk_level = risk_level.upper()
        if risk_level not in {"VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH"}:
            return {"error": f"Invalid risk level {risk_level}"}
        col = f"COUNT_OF_ASSETS_WITH_{risk_level}_{normalized}"
        if col not in exsitu_df.columns:
            return {"error": f"Risk column {col} not available"}
        filtered = exsitu_df[exsitu_df[col] > 0].copy()
        if country:
            country_normalized = country.strip().lower()
            mask = pd.Series(False, index=filtered.index)
            if "COUNTRY_NAME" in filtered.columns:
                mask = mask | filtered["COUNTRY_NAME"].astype(str).str.lower().str.contains(
                    country_normalized, na=False
                )
            if "COUNTRY_CODE" in filtered.columns:
                mask = mask | (
                    filtered["COUNTRY_CODE"].astype(str).str.lower() == country_normalized
                )
            filtered = filtered[mask]
        if sector:
            sector_normalized = sector.strip().upper()
            filtered = filtered[
                filtered["SECTOR_CODE"].astype(str).str.upper() == sector_normalized
            ]
        results = filtered[[
            "COMPANY_CODE",
            "COMPANY_NAME",
            "SECTOR_CODE",
            "COUNTRY_NAME",
            col,
            "TOTAL_NUMBER_OF_ASSETS_ASSESSED_WITHIN_A_COMPANY",
        ]]
        if results.empty:
            return {"error": "No companies meet the criteria"}
        results = results.sort_values(col, ascending=False).head(limit)
        companies: List[Dict[str, Any]] = []
        for _, row in results.iterrows():
            total_assets = row.get("TOTAL_NUMBER_OF_ASSETS_ASSESSED_WITHIN_A_COMPANY", 0)
            count = row.get(col, 0)
            percentage = (count / total_assets * 100) if total_assets else 0
            companies.append(
                {
                    "company_code": row.get("COMPANY_CODE"),
                    "company_name": row.get("COMPANY_NAME"),
                    "sector_code": row.get("SECTOR_CODE"),
                    "country": row.get("COUNTRY_NAME"),
                    "high_risk_assets": count,
                    "high_risk_percentage": round(percentage, 2),
                }
            )
        return {
            "risk_type": normalized,
            "risk_level": risk_level,
            "companies_found": len(companies),
            "companies": companies,
        }

    def get_high_risk_companies(self, risk_threshold: float, limit: int) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        exsitu_df = self.data_manager.get_sheet("EXSITU")
        if exsitu_df.empty:
            return {"error": "EXSITU risk data not available"}
        high_risk_companies: List[Dict[str, Any]] = []
        for _, row in exsitu_df.iterrows():
            total_assets = row.get("TOTAL_NUMBER_OF_ASSETS_ASSESSED_WITHIN_A_COMPANY", 0)
            if not total_assets:
                continue
            high_risk_counts = [
                row[col]
                for col in row.index
                if "COUNT_OF_ASSETS_WITH_HIGH_" in col or "COUNT_OF_ASSETS_WITH_VERY_HIGH_" in col
            ]
            if not high_risk_counts:
                continue
            avg_high_risk_pct = (sum(high_risk_counts) / len(high_risk_counts) / total_assets * 100)
            if avg_high_risk_pct >= risk_threshold:
                high_risk_companies.append(
                    {
                        "company_code": row.get("COMPANY_CODE"),
                        "company_name": row.get("COMPANY_NAME"),
                        "sector_code": row.get("SECTOR_CODE"),
                        "country": row.get("COUNTRY_NAME"),
                        "total_assets": total_assets,
                        "avg_high_risk_percentage": round(avg_high_risk_pct, 2),
                    }
                )
        high_risk_companies.sort(key=lambda item: item["avg_high_risk_percentage"], reverse=True)
        return {
            "risk_threshold": risk_threshold,
            "companies_found": len(high_risk_companies),
            "companies": high_risk_companies[:limit],
        }

    def get_assets_map_data(
        self, company_code: Optional[str], country: Optional[str], limit: int
    ) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        asset_df = self.data_manager.get_sheet("EXSITU_ASSET_DATA")
        if asset_df.empty:
            return {"error": "Asset data not available"}
        filtered_df = asset_df
        if company_code:
            filtered_df = filtered_df[filtered_df["COMPANY_CODE"] == company_code]
        if country:
            filtered_df = filtered_df[filtered_df["COUNTRY_CODE"] == country.upper()]
        filtered_df = filtered_df.head(limit)
        if filtered_df.empty:
            return {"error": "No assets found with specified filters"}
        assets_data = []
        for _, row in filtered_df.iterrows():
            assets_data.append(
                {
                    "asset_id": row.get("ASSET_ID"),
                    "company_code": row.get("COMPANY_CODE"),
                    "company_name": row.get("COMPANY_NAME"),
                    "latitude": float(row.get("LATITUDE")),
                    "longitude": float(row.get("LONGITUDE")),
                    "country": row.get("COUNTRY_CODE"),
                    "asset_type": row.get("ASSET_TYPE_LEVEL_1", "Unknown"),
                    "msa_risk": row.get("MSA_RISKLEVEL", "Unknown"),
                    "water_stress_risk": row.get("WATER_STRESS_RISKLEVEL", "Unknown"),
                }
            )
        return {
            "type": "map",
            "filters_applied": {"company_code": company_code, "country": country, "limit": limit},
            "data": assets_data,
            "metadata": {
                "total_assets": len(assets_data),
                "countries": sorted(list(filtered_df["COUNTRY_CODE"].unique())),
                "companies": sorted(list(filtered_df["COMPANY_CODE"].unique())),
                "asset_types": sorted(list(filtered_df.get("ASSET_TYPE_LEVEL_1", pd.Series()).dropna().unique())),
            },
        }

    def get_assets_in_radius(
        self, latitude: float, longitude: float, radius_km: float, limit: int
    ) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        asset_df = self.data_manager.get_sheet("EXSITU_ASSET_DATA")
        if asset_df.empty:
            return {"error": "Asset data not available"}
        lat1, lon1 = np.radians(latitude), np.radians(longitude)
        lat2, lon2 = np.radians(asset_df["LATITUDE"]), np.radians(asset_df["LONGITUDE"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))
        within_radius = asset_df[distance_km <= radius_km].copy()
        within_radius["distance_km"] = distance_km[distance_km <= radius_km]
        within_radius = within_radius.sort_values("distance_km").head(limit)
        if within_radius.empty:
            return {
                "error": f"No assets found within {radius_km}km of ({latitude}, {longitude})",
            }
        assets_data = []
        for _, row in within_radius.iterrows():
            assets_data.append(
                {
                    "asset_id": row.get("ASSET_ID"),
                    "company_code": row.get("COMPANY_CODE"),
                    "company_name": row.get("COMPANY_NAME"),
                    "latitude": float(row.get("LATITUDE")),
                    "longitude": float(row.get("LONGITUDE")),
                    "distance_km": round(float(row.get("distance_km", 0.0)), 2),
                    "country": row.get("COUNTRY_CODE"),
                    "asset_type": row.get("ASSET_TYPE_LEVEL_1", "Unknown"),
                }
            )
        return {
            "search_center": [latitude, longitude],
            "radius_km": radius_km,
            "assets_found": len(assets_data),
            "assets": assets_data,
        }

    def get_assets_by_country(self) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        asset_df = self.data_manager.get_sheet("EXSITU_ASSET_DATA")
        if asset_df.empty:
            return {"error": "Asset data not available"}
        country_stats = asset_df.groupby("COUNTRY_CODE").agg(
            {
                "ASSET_ID": "count",
                "COMPANY_CODE": "nunique",
                "LATITUDE": ["min", "max"],
                "LONGITUDE": ["min", "max"],
            }
        ).round(2)
        country_stats.columns = [
            "total_assets",
            "unique_companies",
            "min_lat",
            "max_lat",
            "min_lon",
            "max_lon",
        ]
        country_stats = country_stats.reset_index()
        return {
            "total_countries": int(len(country_stats)),
            "countries": country_stats.head(50).to_dict("records"),
        }

    def get_asset_details(self, asset_id: str) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        asset_df = self.data_manager.get_sheet("EXSITU_ASSET_DATA")
        if asset_df.empty:
            return {"error": "Asset data not available"}
        asset_data = asset_df[asset_df["ASSET_ID"] == asset_id]
        if asset_data.empty:
            return {"error": f"Asset {asset_id} not found"}
        asset = asset_data.iloc[0].to_dict()
        return {
            "asset_id": asset_id,
            "company_info": {
                "company_code": asset.get("COMPANY_CODE"),
                "company_name": asset.get("COMPANY_NAME"),
            },
            "location": {
                "latitude": asset.get("LATITUDE"),
                "longitude": asset.get("LONGITUDE"),
                "country_code": asset.get("COUNTRY_CODE"),
            },
            "asset_type": {
                "level_1": asset.get("ASSET_TYPE_LEVEL_1"),
                "level_2": asset.get("ASSET_TYPE_LEVEL_2"),
            },
            "environmental_risks": {
                "msa": {
                    "value": asset.get("MSA"),
                    "risk_level": asset.get("MSA_RISKLEVEL"),
                },
                "water_stress": {
                    "value": asset.get("WATER_STRESS"),
                    "risk_level": asset.get("WATER_STRESS_RISKLEVEL"),
                },
                "water_variability": {
                    "value": asset.get("WATER_VARIABILITY"),
                    "risk_level": asset.get("WATER_VARIABILITY_RISKLEVEL"),
                },
                "water_demand": {
                    "value": asset.get("WATER_DEMAND"),
                    "risk_level": asset.get("WATER_DEMAND_RISKLEVEL"),
                },
                "drought": {
                    "value": asset.get("DROUGHT"),
                    "risk_level": asset.get("DROUGHT_RISKLEVEL"),
                },
                "flood_riverine": {
                    "value": asset.get("FLOOD_RIVERINE"),
                    "risk_level": asset.get("FLOOD_RIVERINE_RISKLEVEL"),
                },
                "flood_coastal": {
                    "value": asset.get("FLOOD_COASTAL"),
                    "risk_level": asset.get("FLOOD_COASTAL_RISKLEVEL"),
                },
                "extreme_heat": {
                    "value": asset.get("EXTREME_HEAT"),
                    "risk_level": asset.get("EXTREME_HEAT_RISK"),
                },
                "extreme_precipitation": {
                    "value": asset.get("EXTREME_PRECIPITATION"),
                    "risk_level": asset.get("EXTREME_PRECIPITATION_RISK"),
                },
                "temperature_anomaly": {
                    "value": asset.get("TEMPERATURE_ANOMALY"),
                    "risk_level": asset.get("TEMPERATURE_ANOMALY_RISK"),
                },
            },
        }

    def get_scope3_emissions(
        self, company_code: str, year: Optional[int]
    ) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        scope3_df = self.data_manager.get_sheet("SCOPE_3_DATA")
        if scope3_df.empty:
            return {"error": "Scope 3 data not available"}
        company_data = scope3_df[scope3_df["COMPANY_CODE"] == company_code]
        if company_data.empty:
            return {"error": f"No Scope 3 data found for company {company_code}"}
        if year:
            company_data = company_data[company_data["REPORTING_YEAR"] == year]
            if company_data.empty:
                return {
                    "error": f"No Scope 3 data found for company {company_code} in year {year}",
                }
        company_data = company_data.sort_values("REPORTING_YEAR")
        emissions_data = {
            "company_code": company_code,
            "company_name": company_data.iloc[0].get("COMPANY_NAME"),
            "sector_code": company_data.iloc[0].get("SECTOR_CODE"),
            "years_available": sorted(company_data["REPORTING_YEAR"].unique().tolist()),
            "emissions_by_year": [],
        }
        for _, row in company_data.iterrows():
            year_data = {
                "reporting_year": row.get("REPORTING_YEAR"),
                "revenue": row.get("REVENUE"),
                "total_scope3_emissions": row.get("SCOPE_3_EMISSIONS_TOTAL"),
                "upstream_emissions": row.get("SCOPE_3_EMISSIONS_TOTAL_UPSTREAM"),
                "downstream_emissions": row.get("SCOPE_3_EMISSIONS_TOTAL_DOWNSTREAM"),
                "breakdown": {
                    "purchased_goods_services": row.get("SCOPE_3_PURCHASED_GOODS_AND_SERVICES"),
                    "capital_goods": row.get("SCOPE_3_CAPITAL_GOODS"),
                    "fuel_energy_activities": row.get(
                        "SCOPE_3_FUEL_AND_ENERGY_RELATED_ACTIVITIES_NOT_INCLUDED_IN_SCOPE_1_OR_SCOPE_2"
                    ),
                    "upstream_transportation": row.get("SCOPE_3_UPSTREAM_TRANSPORTATION_AND_DISTRIBUTION"),
                    "waste_operations": row.get("SCOPE_3_WASTE_GENERATED_IN_OPERATIONS"),
                    "business_travel": row.get("SCOPE_3_BUSINESS_TRAVEL"),
                    "employee_commuting": row.get("SCOPE_3_EMPLOYEE_COMMUTING"),
                    "downstream_transportation": row.get("SCOPE_3_DOWNSTREAM_TRANSPORTATION_AND_DISTRIBUTION"),
                    "use_of_sold_products": row.get("SCOPE_3_USE_OF_SOLD_PRODUCTS"),
                    "end_of_life_treatment": row.get("SCOPE_3_END_OF_LIFE_TREATMENT_OF_SOLD_PRODUCTS"),
                    "investments": row.get("SCOPE_3_INVESTMENTS"),
                },
            }
            if row.get("REVENUE") and row.get("SCOPE_3_EMISSIONS_TOTAL"):
                year_data["emissions_intensity"] = row.get("SCOPE_3_EMISSIONS_TOTAL") / row.get("REVENUE")
            emissions_data["emissions_by_year"].append(year_data)
        return emissions_data

    def get_emissions_trends(self, company_code: str) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        scope3_df = self.data_manager.get_sheet("SCOPE_3_DATA")
        if scope3_df.empty:
            return {"error": "Scope 3 data not available"}
        company_data = scope3_df[scope3_df["COMPANY_CODE"] == company_code].sort_values(
            "REPORTING_YEAR"
        )
        if company_data.empty:
            return {"error": f"No emissions data found for company {company_code}"}
        if len(company_data) < 2:
            return {"error": "Insufficient data for trend analysis (need at least 2 years)"}
        trends: Dict[str, Any] = {
            "company_code": company_code,
            "company_name": company_data.iloc[0].get("COMPANY_NAME"),
            "analysis_period": {
                "start_year": int(company_data["REPORTING_YEAR"].min()),
                "end_year": int(company_data["REPORTING_YEAR"].max()),
                "years_analyzed": len(company_data),
            },
            "trends": {},
        }
        metrics = {
            "total_emissions": "SCOPE_3_EMISSIONS_TOTAL",
            "upstream_emissions": "SCOPE_3_EMISSIONS_TOTAL_UPSTREAM",
            "downstream_emissions": "SCOPE_3_EMISSIONS_TOTAL_DOWNSTREAM",
            "revenue": "REVENUE",
        }
        for name, column in metrics.items():
            if column not in company_data.columns:
                continue
            values = company_data[column].dropna()
            if len(values) < 2:
                continue
            first_value = values.iloc[0]
            last_value = values.iloc[-1]
            if first_value == 0:
                continue
            percent_change = ((last_value - first_value) / abs(first_value)) * 100
            trends["trends"][name] = {
                "start_value": first_value,
                "end_value": last_value,
                "absolute_change": last_value - first_value,
                "percent_change": round(percent_change, 2),
                "trend_direction": "increasing" if percent_change > 0 else "decreasing",
            }
        return trends

    def get_emissions_by_sector(self, year: Optional[int]) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        scope3_df = self.data_manager.get_sheet("SCOPE_3_DATA")
        if scope3_df.empty:
            return {"error": "Scope 3 data not available"}
        if year:
            data = scope3_df[scope3_df["REPORTING_YEAR"] == year]
            if data.empty:
                return {"error": f"No emissions data available for year {year}"}
        else:
            latest_year = scope3_df["REPORTING_YEAR"].max()
            data = scope3_df[scope3_df["REPORTING_YEAR"] == latest_year]
            year = latest_year
        sector_stats = data.groupby("SECTOR_CODE").agg(
            {
                "SCOPE_3_EMISSIONS_TOTAL": "sum",
                "COMPANY_CODE": "nunique",
            }
        ).round(2)
        sector_stats = sector_stats.reset_index()
        return {
            "analysis_year": year,
            "total_sectors": int(len(sector_stats)),
            "sector_emissions": sector_stats.head(50).to_dict("records"),
        }

    def get_top_emitters(self, limit: int, year: Optional[int]) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        scope3_df = self.data_manager.get_sheet("SCOPE_3_DATA")
        if scope3_df.empty:
            return {"error": "Scope 3 data not available"}
        if year:
            data = scope3_df[scope3_df["REPORTING_YEAR"] == year]
            if data.empty:
                return {"error": f"No emissions data available for year {year}"}
        else:
            latest_year = scope3_df["REPORTING_YEAR"].max()
            data = scope3_df[scope3_df["REPORTING_YEAR"] == latest_year]
            year = latest_year
        data = data.dropna(subset=["SCOPE_3_EMISSIONS_TOTAL"])
        data = data[data["SCOPE_3_EMISSIONS_TOTAL"] > 0]
        top_emitters = data.nlargest(limit, "SCOPE_3_EMISSIONS_TOTAL")
        emitters_list = []
        for _, row in top_emitters.iterrows():
            entry = {
                "rank": len(emitters_list) + 1,
                "company_code": row.get("COMPANY_CODE"),
                "company_name": row.get("COMPANY_NAME"),
                "sector_code": row.get("SECTOR_CODE"),
                "reporting_year": row.get("REPORTING_YEAR"),
                "total_scope3_emissions": row.get("SCOPE_3_EMISSIONS_TOTAL"),
                "upstream_emissions": row.get("SCOPE_3_EMISSIONS_TOTAL_UPSTREAM"),
                "downstream_emissions": row.get("SCOPE_3_EMISSIONS_TOTAL_DOWNSTREAM"),
                "revenue": row.get("REVENUE"),
            }
            revenue = row.get("REVENUE")
            total = row.get("SCOPE_3_EMISSIONS_TOTAL")
            if revenue and revenue > 0 and total:
                entry["emissions_intensity"] = total / revenue
            emitters_list.append(entry)
        return {
            "analysis_year": year,
            "companies_analyzed": len(data),
            "top_emitters": emitters_list,
        }

    def get_biodiversity_impacts(
        self, company_code: str, year: Optional[int]
    ) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        bio_df = self.data_manager.get_sheet("BIODIVERSITY_PDF_DATA")
        if bio_df.empty:
            return {"error": "Biodiversity data not available"}
        company_data = bio_df[bio_df["COMPANY_CODE"] == company_code]
        if company_data.empty:
            return {"error": f"No biodiversity data found for company {company_code}"}
        if year:
            company_data = company_data[company_data["REPORTING_YEAR"] == year]
            if company_data.empty:
                return {
                    "error": f"No biodiversity data found for company {company_code} in year {year}",
                }
        company_data = company_data.sort_values("REPORTING_YEAR")
        biodiversity_data = {
            "company_code": company_code,
            "company_name": company_data.iloc[0].get("COMPANY_NAME"),
            "sector_code": company_data.iloc[0].get("SECTOR_CODE"),
            "years_available": sorted(company_data["REPORTING_YEAR"].unique().tolist()),
            "impacts_by_year": [],
        }
        for _, row in company_data.iterrows():
            biodiversity_data["impacts_by_year"].append(
                {
                    "reporting_year": row.get("REPORTING_YEAR"),
                    "total_impacts": {
                        "pdf": row.get("TOTAL_COMPANY_IMPACTS_PDF"),
                        "co2e": row.get("TOTAL_COMPANY_IMPACTS_CO2E"),
                        "lce": row.get("TOTAL_COMPANY_IMPACTS_LCE"),
                    },
                    "impact_categories_pdf": {
                        "ghg_100_years": row.get("GHG_IMPACTS_PDF_100_YRS"),
                        "ghg_1000_years": row.get("GHG_IMPACTS_PDF_1000_YRS"),
                        "water_consumption": row.get("WATER_CONSUMPTION_IMPACTS_PDF"),
                        "sox_impacts": row.get("SOX_IMPACTS_PDF"),
                        "nox_impacts": row.get("NOX_IMPACTS_PDF"),
                        "nitrogen_impacts": row.get("TOTAL_NITROGEN_IMPACTS_PDF"),
                        "phosphorous_impacts": row.get("TOTAL_PHOSPHOROUS_IMPACTS_PDF"),
                        "land_use_change": row.get("LUC_IMPACTS_PDF"),
                        "waste_generation_100": row.get("WASTE_GENERATION_IMPACTS_PDF_100_YRS"),
                        "waste_generation_1000": row.get("WASTE_GENERATION_IMPACTS_PDF_1000_YRS"),
                    },
                    "impact_categories_co2e": {
                        "ghg_100_years": row.get("GHG_IMPACTS_CO2E_100_YRS"),
                        "ghg_1000_years": row.get("GHG_IMPACTS_CO2E_1000_YRS"),
                        "water_consumption": row.get("WATER_CONSUMPTION_IMPACTS_CO2E"),
                        "sox_impacts": row.get("SOX_IMPACTS_CO2E"),
                        "nox_impacts": row.get("NOX_IMPACTS_CO2E"),
                        "nitrogen_impacts": row.get("TOTAL_NITROGEN_IMPACTS_CO2E"),
                        "phosphorous_impacts": row.get("TOTAL_PHOSPHOROUS_IMPACTS_CO2E"),
                        "land_use_change": row.get("LUC_IMPACTS_CO2E"),
                        "waste_generation_100": row.get("WASTE_GENERATION_IMPACTS_CO2E_100_YRS"),
                        "waste_generation_1000": row.get("WASTE_GENERATION_IMPACTS_CO2E_1000_YRS"),
                    },
                }
            )
        return biodiversity_data

    def get_biodiversity_trends(self, company_code: str) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        bio_df = self.data_manager.get_sheet("BIODIVERSITY_PDF_DATA")
        if bio_df.empty:
            return {"error": "Biodiversity data not available"}
        company_data = bio_df[bio_df["COMPANY_CODE"] == company_code].sort_values(
            "REPORTING_YEAR"
        )
        if company_data.empty:
            return {"error": f"No biodiversity data found for company {company_code}"}
        if len(company_data) < 2:
            return {"error": "Insufficient data for trend analysis (need at least 2 years)"}
        trends: Dict[str, Any] = {
            "company_code": company_code,
            "company_name": company_data.iloc[0].get("COMPANY_NAME"),
            "analysis_period": {
                "start_year": int(company_data["REPORTING_YEAR"].min()),
                "end_year": int(company_data["REPORTING_YEAR"].max()),
                "years_analyzed": len(company_data),
            },
            "trends": {},
        }
        metrics = {
            "pdf_impacts": "TOTAL_COMPANY_IMPACTS_PDF",
            "co2e_impacts": "TOTAL_COMPANY_IMPACTS_CO2E",
            "lce_impacts": "TOTAL_COMPANY_IMPACTS_LCE",
            "ghg_pdf_100": "GHG_IMPACTS_PDF_100_YRS",
            "water_consumption_pdf": "WATER_CONSUMPTION_IMPACTS_PDF",
            "land_use_change_pdf": "LUC_IMPACTS_PDF",
        }
        for metric_name, column in metrics.items():
            if column not in company_data.columns:
                continue
            values = company_data[column].dropna()
            if len(values) < 2:
                continue
            first_value = values.iloc[0]
            last_value = values.iloc[-1]
            if first_value == 0:
                continue
            percent_change = ((last_value - first_value) / abs(first_value)) * 100
            trends["trends"][metric_name] = {
                "start_value": first_value,
                "end_value": last_value,
                "absolute_change": last_value - first_value,
                "percent_change": round(percent_change, 2),
                "trend_direction": "increasing" if percent_change > 0 else "decreasing",
            }
        return trends

    def get_biodiversity_by_sector(self, year: Optional[int]) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        bio_df = self.data_manager.get_sheet("BIODIVERSITY_PDF_DATA")
        if bio_df.empty:
            return {"error": "Biodiversity data not available"}
        if year:
            data = bio_df[bio_df["REPORTING_YEAR"] == year]
            if data.empty:
                return {"error": f"No biodiversity data available for year {year}"}
        else:
            latest_year = bio_df["REPORTING_YEAR"].max()
            data = bio_df[bio_df["REPORTING_YEAR"] == latest_year]
            year = latest_year
        sector_stats = data.groupby("SECTOR_CODE").agg(
            {
                "TOTAL_COMPANY_IMPACTS_PDF": ["sum", "mean", "median"],
                "TOTAL_COMPANY_IMPACTS_CO2E": ["sum", "mean", "median"],
                "TOTAL_COMPANY_IMPACTS_LCE": ["sum", "mean", "median"],
                "GHG_IMPACTS_PDF_100_YRS": "sum",
                "WATER_CONSUMPTION_IMPACTS_PDF": "sum",
                "LUC_IMPACTS_PDF": "sum",
                "COMPANY_CODE": "nunique",
            }
        ).round(6)
        sector_stats.columns = [
            "total_pdf_sum",
            "mean_pdf",
            "median_pdf",
            "total_co2e_sum",
            "mean_co2e",
            "median_co2e",
            "total_lce_sum",
            "mean_lce",
            "median_lce",
            "total_ghg_pdf",
            "total_water_pdf",
            "total_luc_pdf",
            "unique_companies",
        ]
        sector_stats = sector_stats.reset_index()
        sector_stats = sector_stats.sort_values("total_pdf_sum", ascending=False)
        return {
            "analysis_year": year,
            "total_sectors": int(len(sector_stats)),
            "sector_biodiversity_impacts": sector_stats.head(50).to_dict("records"),
        }

    def get_biodiversity_worst_performers(
        self, metric: str, limit: int, year: Optional[int]
    ) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        bio_df = self.data_manager.get_sheet("BIODIVERSITY_PDF_DATA")
        if bio_df.empty:
            return {"error": "Biodiversity data not available"}
        if year:
            data = bio_df[bio_df["REPORTING_YEAR"] == year]
            if data.empty:
                return {"error": f"No biodiversity data available for year {year}"}
        else:
            latest_year = bio_df["REPORTING_YEAR"].max()
            data = bio_df[bio_df["REPORTING_YEAR"] == latest_year]
            year = latest_year
        metric_columns = {
            "PDF": "TOTAL_COMPANY_IMPACTS_PDF",
            "CO2E": "TOTAL_COMPANY_IMPACTS_CO2E",
            "LCE": "TOTAL_COMPANY_IMPACTS_LCE",
        }
        key = metric.upper()
        if key not in metric_columns:
            return {"error": f"Invalid metric {metric}. Available: {list(metric_columns.keys())}"}
        column = metric_columns[key]
        data = data.dropna(subset=[column])
        worst_performers = data.nlargest(limit, column)
        performers_list = []
        for _, row in worst_performers.iterrows():
            performers_list.append(
                {
                    "rank": len(performers_list) + 1,
                    "company_code": row.get("COMPANY_CODE"),
                    "company_name": row.get("COMPANY_NAME"),
                    "sector_code": row.get("SECTOR_CODE"),
                    "reporting_year": row.get("REPORTING_YEAR"),
                    "impact_value": row.get(column),
                    "impact_metric": key,
                    "other_metrics": {
                        "pdf_impact": row.get("TOTAL_COMPANY_IMPACTS_PDF"),
                        "co2e_impact": row.get("TOTAL_COMPANY_IMPACTS_CO2E"),
                        "lce_impact": row.get("TOTAL_COMPANY_IMPACTS_LCE"),
                    },
                }
            )
        return {
            "analysis_year": year,
            "impact_metric": key,
            "companies_analyzed": len(data),
            "worst_performers": performers_list,
        }

    def get_deforestation_risks(self, company_code: Optional[str]) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        deforest_df = self.data_manager.get_sheet("DEFORESTATION")
        if deforest_df.empty:
            return {"error": "Deforestation data not available"}
        if company_code:
            company_data = deforest_df[deforest_df["COMPANY_CODE"] == company_code]
            if company_data.empty:
                return {"error": f"No deforestation data found for company {company_code}"}
            row = company_data.iloc[0]
            return {
                "company_code": company_code,
                "company_name": row.get("COMPANY_NAME", "Unknown"),
                "deforestation_indicators": {
                    "high_fraction_assets_forest_change": bool(
                        row.get("company_high_fraction_assets_forest_change_proximity", False)
                    ),
                    "high_average_forest_change": bool(
                        row.get("company_high_average_forest_change_proximity", False)
                    ),
                    "extreme_forest_change_proximity": bool(
                        row.get("company_asset_extreme_forest_change_proximity", False)
                    ),
                },
                "risk_level": self._calculate_deforestation_risk_level(row),
            }
        total_companies = int(len(deforest_df))
        high_fraction = int(
            deforest_df["company_high_fraction_assets_forest_change_proximity"].sum()
        )
        high_average = int(
            deforest_df["company_high_average_forest_change_proximity"].sum()
        )
        extreme_proximity = int(
            deforest_df["company_asset_extreme_forest_change_proximity"].sum()
        )
        return {
            "total_companies": total_companies,
            "companies_high_fraction": high_fraction,
            "companies_high_average": high_average,
            "companies_extreme_proximity": extreme_proximity,
        }

    @staticmethod
    def _calculate_deforestation_risk_level(row: pd.Series) -> str:
        indicators = [
            row.get("company_high_fraction_assets_forest_change_proximity", False),
            row.get("company_high_average_forest_change_proximity", False),
            row.get("company_asset_extreme_forest_change_proximity", False),
        ]
        score = sum(bool(x) for x in indicators)
        if score == 0:
            return "low"
        if score == 1:
            return "moderate"
        if score == 2:
            return "high"
        return "very_high"

    def get_deforestation_exposed(self, limit: int) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        deforest_df = self.data_manager.get_sheet("DEFORESTATION")
        if deforest_df.empty:
            return {"error": "Deforestation data not available"}
        deforest_df = deforest_df[
            deforest_df[
                [
                    "company_high_fraction_assets_forest_change_proximity",
                    "company_high_average_forest_change_proximity",
                    "company_asset_extreme_forest_change_proximity",
                ]
            ].any(axis=1)
        ]
        records = []
        for _, row in deforest_df.iterrows():
            records.append(
                {
                    "company_code": row.get("COMPANY_CODE"),
                    "company_name": row.get("COMPANY_NAME"),
                    "high_fraction_assets": bool(
                        row.get("company_high_fraction_assets_forest_change_proximity", False)
                    ),
                    "high_average_proximity": bool(
                        row.get("company_high_average_forest_change_proximity", False)
                    ),
                    "extreme_proximity": bool(
                        row.get("company_asset_extreme_forest_change_proximity", False)
                    ),
                }
            )
        return {
            "total_companies_analyzed": len(deforest_df),
            "companies_with_deforestation_risk": len(records),
            "high_risk_companies": records[:limit],
        }

    def get_forest_change_proximity(self) -> Dict[str, Any]:
        if not self.data_manager.sheets:
            return {"error": "GIST data not available"}
        deforest_df = self.data_manager.get_sheet("DEFORESTATION")
        if deforest_df.empty:
            return {"error": "Deforestation data not available"}
        sector_analysis: Dict[str, Dict[str, Any]] = {}
        for _, row in deforest_df.iterrows():
            company_code = row.get("COMPANY_CODE")
            company_info = self.data_manager.companies_cache.get(company_code, {})
            sector = company_info.get("sector_code", "Unknown")
            bucket = sector_analysis.setdefault(
                sector,
                {
                    "total_companies": 0,
                    "high_fraction_risk": 0,
                    "high_average_risk": 0,
                    "extreme_proximity": 0,
                },
            )
            bucket["total_companies"] += 1
            if row.get("company_high_fraction_assets_forest_change_proximity", False):
                bucket["high_fraction_risk"] += 1
            if row.get("company_high_average_forest_change_proximity", False):
                bucket["high_average_risk"] += 1
            if row.get("company_asset_extreme_forest_change_proximity", False):
                bucket["extreme_proximity"] += 1
        for stats in sector_analysis.values():
            total = stats["total_companies"]
            if not total:
                continue
            stats["high_fraction_percentage"] = round(
                stats["high_fraction_risk"] / total * 100, 2
            )
            stats["high_average_percentage"] = round(
                stats["high_average_risk"] / total * 100, 2
            )
            stats["extreme_proximity_percentage"] = round(
                stats["extreme_proximity"] / total * 100, 2
            )
        return {
            "total_companies_analyzed": len(deforest_df),
            "sector_analysis": sector_analysis,
        }

    def get_visualization_data(self, viz_type: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        filters = filters or {}
        if viz_type == "emissions_by_sector":
            return self._get_emissions_by_sector_viz(filters)
        if viz_type == "risk_distribution":
            return self._get_risk_distribution_viz(filters)
        if viz_type == "asset_map":
            return self._get_asset_map_viz(filters)
        if viz_type == "biodiversity_trends":
            return self._get_biodiversity_trends_viz(filters)
        if viz_type == "scope3_breakdown":
            return self._get_scope3_breakdown_viz(filters)
        return {
            "error": (
                f"Unknown visualization type: {viz_type}. Available: emissions_by_sector, "
                "risk_distribution, asset_map, biodiversity_trends, scope3_breakdown"
            )
        }

    def _get_emissions_by_sector_viz(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        scope3_df = self.data_manager.get_sheet("SCOPE_3_DATA")
        if scope3_df.empty:
            return {"error": "Scope 3 data not available"}
        year = filters.get("year", scope3_df["REPORTING_YEAR"].max())
        data = scope3_df[scope3_df["REPORTING_YEAR"] == year]
        sector_stats = data.groupby("SECTOR_CODE").agg(
            {
                "SCOPE_3_EMISSIONS_TOTAL": "sum",
                "COMPANY_CODE": "nunique",
            }
        ).round(2)
        sector_stats = sector_stats.reset_index()
        return {
            "visualization_type": "emissions_by_sector",
            "data": sector_stats.to_dict("records"),
            "chart_config": {
                "x_axis": "SECTOR_CODE",
                "y_axis": "SCOPE_3_EMISSIONS_TOTAL",
                "title": f"Scope 3 Emissions by Sector ({year})",
                "chart_type": "bar",
            },
            "metadata": {"year": year, "total_sectors": int(len(sector_stats))},
        }

    def _get_risk_distribution_viz(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        exsitu_df = self.data_manager.get_sheet("EXSITU")
        if exsitu_df.empty:
            return {"error": "EXSITU risk data not available"}
        risk_types = ["MSA", "WATER_STRESS", "DROUGHT", "FLOOD_COASTAL", "EXTREME_HEAT"]
        risk_data = []
        for risk_type in risk_types:
            high_col = f"COUNT_OF_ASSETS_WITH_HIGH_{risk_type}"
            very_high_col = f"COUNT_OF_ASSETS_WITH_VERY_HIGH_{risk_type}"
            if high_col in exsitu_df.columns and very_high_col in exsitu_df.columns:
                companies_at_risk = len(exsitu_df[(exsitu_df[high_col] > 0) | (exsitu_df[very_high_col] > 0)])
                risk_data.append(
                    {
                        "risk_type": risk_type,
                        "companies_high_risk": companies_at_risk,
                        "high_risk_assets": int(exsitu_df[high_col].sum()),
                        "very_high_risk_assets": int(exsitu_df[very_high_col].sum()),
                    }
                )
        return {
            "visualization_type": "risk_distribution",
            "data": risk_data,
            "chart_config": {
                "x_axis": "risk_type",
                "y_axis": "companies_high_risk",
                "title": "Companies with High/Very High Risk by Category",
                "chart_type": "column",
            },
            "metadata": {"risk_types_analyzed": len(risk_data)},
        }

    def _get_asset_map_viz(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        company_code = filters.get("company_code")
        country = filters.get("country")
        limit = filters.get("limit", 200)
        response = self.get_assets_map_data(company_code, country, limit)
        if "error" in response:
            return response
        return {
            "visualization_type": "asset_map",
            "data": response["data"],
            "metadata": response.get("metadata", {}),
        }

    # def _get_biodiversity_trends_viz(self, filters: Dict[str, Any]) -> Dict[str, Any]:
    #     company_code = filters.get("company_code")
    #     if not company_code:
    #         return {"error": "company_code filter required"}
    #     response = self.get_biodiversity_impacts(company_code, year=None)
    #     if "error" in response:
    #         return response
    #     data = response.get("impacts_by_year", [])
    #     return {
    #         "visualization_type": "biodiversity_trends",
    #         "data": data,
    #         "chart_config": {
    #             "x_axis": "reporting_year",
    #             "y_axis": "total_impacts.pdf",
    #             "title": f"Biodiversity PDF impacts for {company_code}",
    #             "chart_type": "line",
    #         },
    #         "metadata": {
    #             "company_code": company_code,
    #             "years": [item.get("reporting_year") for item in data],
    #         },
    #     }

    def _get_scope3_breakdown_viz(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        company_code = filters.get("company_code")
        year = filters.get("year")
        scope3_df = self.data_manager.get_sheet("SCOPE_3_DATA")
        if scope3_df.empty:
            return {"error": "Scope 3 data not available"}
        if not company_code:
            available_companies = scope3_df.groupby("COMPANY_CODE").agg(
                {
                    "COMPANY_NAME": "first",
                    "SECTOR_CODE": "first",
                    "REPORTING_YEAR": ["min", "max", "count"],
                    "SCOPE_3_EMISSIONS_TOTAL": "sum",
                }
            )
            available_companies.columns = [
                "company_name",
                "sector_code",
                "min_year",
                "max_year",
                "years_count",
                "total_emissions",
            ]
            available_companies = available_companies.reset_index()
            available_companies = available_companies.sort_values("total_emissions", ascending=False)
            return {
                "error": "company_code required for scope3 breakdown",
                "help": "Specify a company_code filter to view Scope 3 emissions breakdown for a company",
                "available_companies": available_companies.head(20).to_dict("records"),
                "total_companies_with_data": int(len(available_companies)),
                "suggested_company": available_companies.iloc[0]["COMPANY_CODE"]
                if len(available_companies) > 0
                else None,
            }
        company_data = scope3_df[scope3_df["COMPANY_CODE"] == company_code]
        if company_data.empty:
            similar = scope3_df[scope3_df["COMPANY_CODE"].str.contains(company_code, case=False, na=False)][
                "COMPANY_CODE"
            ].unique()
            message = f"No Scope 3 data for company {company_code}"
            if len(similar) > 0:
                message += f". Did you mean one of: {list(similar)[:5]}"
            return {"error": message}
        if year:
            company_data = company_data[company_data["REPORTING_YEAR"] == year]
            if company_data.empty:
                available_years = scope3_df[scope3_df["COMPANY_CODE"] == company_code]["REPORTING_YEAR"].unique()
                return {
                    "error": f"No Scope 3 data for company {company_code} in year {year}. Available years: {sorted(available_years)}",
                }
        else:
            latest_year = company_data["REPORTING_YEAR"].max()
            company_data = company_data[company_data["REPORTING_YEAR"] == latest_year]
            year = latest_year
        row = company_data.iloc[0]
        breakdown_data = [
            {
                "category": "Purchased Goods & Services",
                "emissions": row.get("SCOPE_3_PURCHASED_GOODS_AND_SERVICES", 0),
            },
            {"category": "Capital Goods", "emissions": row.get("SCOPE_3_CAPITAL_GOODS", 0)},
            {
                "category": "Fuel & Energy Activities",
                "emissions": row.get(
                    "SCOPE_3_FUEL_AND_ENERGY_RELATED_ACTIVITIES_NOT_INCLUDED_IN_SCOPE_1_OR_SCOPE_2", 0
                ),
            },
            {
                "category": "Upstream Transportation",
                "emissions": row.get("SCOPE_3_UPSTREAM_TRANSPORTATION_AND_DISTRIBUTION", 0),
            },
            {"category": "Business Travel", "emissions": row.get("SCOPE_3_BUSINESS_TRAVEL", 0)},
            {"category": "Employee Commuting", "emissions": row.get("SCOPE_3_EMPLOYEE_COMMUTING", 0)},
            {
                "category": "Use of Sold Products",
                "emissions": row.get("SCOPE_3_USE_OF_SOLD_PRODUCTS", 0),
            },
            {"category": "Investments", "emissions": row.get("SCOPE_3_INVESTMENTS", 0)},
        ]
        breakdown_data = [item for item in breakdown_data if item["emissions"] > 0]
        breakdown_data.sort(key=lambda item: item["emissions"], reverse=True)
        return {
            "visualization_type": "scope3_breakdown",
            "data": breakdown_data,
            "chart_config": {
                "x_axis": "category",
                "y_axis": "emissions",
                "title": f"Scope 3 Emissions Breakdown - {company_code} ({year})",
                "chart_type": "pie",
            },
            "metadata": {
                "company_code": company_code,
                "year": year,
                "total_categories": len(breakdown_data),
            },
        }

    def get_dataset_metadata(self) -> Dict[str, Any]:
        return self.metadata

    def describe_server(self) -> Dict[str, Any]:
        m = self.metadata.copy()
        try:
            last_updated = None
            if self.data_manager.excel_path and self.data_manager.excel_path.exists():
                last_updated = _dt.fromtimestamp(
                    self.data_manager.excel_path.stat().st_mtime
                ).isoformat()
        except Exception:
            last_updated = None
        return {
            "name": m.get("Name", "GIST Server"),
            "description": m.get("Description", "GIST environmental datasets"),
            "version": m.get("Version"),
            "dataset": m.get("Dataset"),
            "metrics": {
                "total_companies": m.get("Total_Companies"),
                "total_assets": m.get("Total_Assets"),
                "dataset_count": len(m.get("Datasets", [])),
            },
            "coverage": {"datasets": m.get("Datasets", [])},
            "tools": [
                "GetGistCompanyProfile",
                "GetGistCompanyRisks",
                "GetGistScope3Emissions",
                "GetGistCompanies",
                "GetGistDataDictionary",
                "SearchGistFields",
                "GetGistDatasetMetadata",
            ],
            "examples": [
                "Rank the most exposed companies to riverine flooding in Brazil",
                "Water stress breakdown for company CODE",
                "Compare Scope 1/2/3 for company CODE",
            ],
            "last_updated": last_updated,
        }

    # ------------------------------------------------------------------ tool registration helpers
    def _register_capabilities_tool(self) -> None:
        @self.mcp.tool()
        def describe_capabilities(format: str = "json") -> str:  # type: ignore[misc]
            """Describe the GIST dataset coverage and available tooling.

            Example:
                >>> describe_capabilities()
            """

            payload = self._capabilities_metadata()
            return json.dumps(payload) if format == "json" else payload  # type: ignore[return-value]

    def _register_query_support_tool(self) -> None:
        @self.mcp.tool()
        def query_support(query: str, context: dict) -> str:  # type: ignore[misc]
            """Assess whether the GIST dataset can address the query.

            Example:
                >>> query_support("Which Brazilian miners face high water stress?", {})
            """

            intent = self._classify_support(query)
            payload = {
                "server": "gist",
                "query": query,
                "supported": intent.supported,
                "score": round(intent.score, 2),
                "reasons": intent.reasons,
            }
            return json.dumps(payload)

    def _register_data_dictionary_tools(self) -> None:
        @self.mcp.tool()
        def GetGistDataDictionary(dataset_name: Optional[str] = None) -> Dict[str, Any]:  # type: ignore[misc]
            """Get field definitions from GIST Data Dictionary."""

            return self.get_data_dictionary(dataset_name)

        @self.mcp.tool()
        def SearchGistFields(search_term: str) -> Dict[str, Any]:  # type: ignore[misc]
            """Search field names and definitions in GIST dictionary."""

            return self.search_fields(search_term)

        @self.mcp.tool()
        def GetGistDatasetSchemas() -> Dict[str, Any]:  # type: ignore[misc]
            """List all GIST datasets with field counts."""

            return self.get_dataset_schemas()

    def _register_company_tools(self) -> None:
        @self.mcp.tool()
        def GetGistCompanies(
            sector: Optional[str] = None,
            country: Optional[str] = None,
            limit: int = 50,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Return basic profiles for companies, optionally filtered by sector/country."""

            return self.get_companies(sector=sector, country=country, limit=limit)

        @self.mcp.tool()
        def GetGistCompanyProfile(company_code: str) -> Dict[str, Any]:  # type: ignore[misc]
            """Summarise every GIST dataset available for a company code."""

            return self.get_company_profile(company_code)

        @self.mcp.tool()
        def GetGistCompaniesBySector() -> Dict[str, Any]:  # type: ignore[misc]
            """Get companies grouped by sector."""

            return self.get_companies_by_sector()

        @self.mcp.tool()
        def GetGistCompanyDataAvailability(
            company_code: str,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """List which GIST tables contain rows for the provided company."""

            return self.get_company_data_availability(company_code)

    def _register_risk_and_geospatial_tools(self) -> None:
        @self.mcp.tool()
        def GetGistCompanyRisks(company_code: str) -> Dict[str, Any]:  # type: ignore[misc]
            """Return a detailed hazard profile for a single company (all risk categories).

            Use when the user explicitly requests data for a known company code.
            """

            return self.get_company_risks(company_code)

        @self.mcp.tool()
        def GetGistRiskByCategory(
            risk_type: str,
            risk_level: str = "HIGH",
            limit: int = 20,
            country: Optional[str] = None,
            sector: Optional[str] = None,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Rank companies by exposure to a hazard (e.g., riverine flood) with optional country/sector filters.

            Example:
                >>> GetGistRiskByCategory(risk_type="flood", country="Brazil", limit=10)
                # returns top companies with the most assets at high riverine-flood risk in Brazil
            """

            ranking = self.get_risk_by_category(
                risk_type=risk_type,
                risk_level=risk_level,
                limit=limit,
                country=country,
                sector=sector,
            )

            # Surface errors directly for planner diagnostics.
            if ranking.get("error"):
                return {
                    "summary": None,
                    "facts": [],
                    "artifacts": [],
                    "messages": [
                        {
                            "level": "warning",
                            "text": str(ranking["error"]),
                        }
                    ],
                    "citation": {
                        "tool": "GetGistRiskByCategory",
                        "title": "GIST hazard ranking",
                        "source_type": "Dataset",
                        "description": _dataset_citation(DATASET_ID),
                        "metadata": {
                            "risk_type": risk_type,
                            "risk_level": risk_level,
                            "country": country,
                            "sector": sector,
                        },
                    },
                    "details": ranking,
                }

            companies = ranking.get("companies") or []
            if not companies:
                return {
                    "summary": None,
                    "facts": [],
                    "artifacts": [],
                    "messages": [
                        {
                            "level": "info",
                            "text": "No companies meet the requested hazard criteria.",
                        }
                    ],
                    "citation": {
                        "tool": "GetGistRiskByCategory",
                        "title": "GIST hazard ranking",
                        "source_type": "Dataset",
                        "description": _dataset_citation(DATASET_ID),
                        "metadata": {
                            "risk_type": risk_type,
                            "risk_level": risk_level,
                            "country": country,
                            "sector": sector,
                        },
                    },
                    "details": ranking,
                }

            # Ensure downstream consumers receive native Python primitives.
            ranking_native = self._to_native(ranking)
            companies_native: List[Dict[str, Any]] = [
                self._to_native(company) for company in companies
                if isinstance(company, Mapping)
            ]
            if companies_native:
                ranking_native["companies"] = companies_native
                top_company = companies_native[0]
            else:
                top_company = companies[0]

            risk_label = str(ranking.get("risk_type", risk_type)).replace("_", " ").lower()
            level_label = str(ranking.get("risk_level", risk_level)).replace("_", " ").lower()
            location_phrase = f" in {country}" if country else ""

            top_name = top_company.get("company_name", "Unknown")
            top_code = top_company.get("company_code", "Unknown")
            top_assets = top_company.get("high_risk_assets", 0)
            top_pct = top_company.get("high_risk_percentage", 0)

            summary = (
                f"{top_name} ({top_code}) shows the highest share of assets at {level_label} "
                f"{risk_label}{location_phrase}, with {top_assets} assets accounting for "
                f"{top_pct}% of those assessed."
            )

            facts: List[str] = []
            for idx, company in enumerate(companies_native[: min(3, len(companies_native))], start=1):
                facts.append(
                    (
                        f"#{idx}: {company.get('company_name', 'Unknown')} ({company.get('company_code', 'Unknown')}) "
                        f"has {company.get('high_risk_assets', 0)} assets at {risk_label} {level_label} "
                        f"risk, representing {company.get('high_risk_percentage', 0)}% of assessed assets."
                    )
                )

            table_rows: List[List[Any]] = []
            for idx, company in enumerate(companies_native, start=1):
                table_rows.append(
                    [
                        idx,
                        company.get("company_name"),
                        company.get("company_code"),
                        company.get("sector_code"),
                        company.get("country"),
                        company.get("high_risk_assets"),
                        company.get("high_risk_percentage"),
                    ]
                )

            artifact = {
                "type": "table",
                "title": "Company exposure ranking",
                "metadata": {
                    "risk_type": ranking.get("risk_type", risk_type),
                    "risk_level": ranking.get("risk_level", risk_level),
                    "country": country,
                    "sector": sector,
                },
                "columns": [
                    "rank",
                    "company_name",
                    "company_code",
                    "sector_code",
                    "country",
                    "high_risk_assets",
                    "high_risk_percentage",
                ],
                "rows": table_rows,
            }

            citation = {
                "tool": "GetGistRiskByCategory",
                "title": "GIST hazard ranking",
                "source_type": "Dataset",
                "description": _dataset_citation(DATASET_ID),
                "metadata": {
                    "risk_type": ranking.get("risk_type", risk_type),
                    "risk_level": ranking.get("risk_level", risk_level),
                    "country": country,
                    "sector": sector,
                },
            }

            return {
                "summary": summary,
                "facts": facts,
                "artifacts": [artifact],
                "messages": [],
                "citation": citation,
                "details": ranking_native,
            }

        @self.mcp.tool()
        def GetGistHighRiskCompanies(
            risk_threshold: float = 25.0, limit: int = 20
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Surface companies whose assets show systemic high/very-high exposure across hazards.

            Example:
                >>> GetGistHighRiskCompanies(risk_threshold=30, limit=15)
                # returns companies with ≥30% of assessed assets in high/very-high risk buckets overall
            """

            return self.get_high_risk_companies(risk_threshold, limit)

        @self.mcp.tool()
        def GetGistAssetsMapData(
            company_code: Optional[str] = None,
            country: Optional[str] = None,
            limit: int = 100,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Retrieve asset coordinates for mapping (filterable by company or country)."""

            return self.get_assets_map_data(company_code, country, limit)

        @self.mcp.tool()
        def GetGistAssetsInRadius(
            latitude: float,
            longitude: float,
            radius_km: float = 50.0,
            limit: int = 100,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Find assets within radius of coordinates."""

            return self.get_assets_in_radius(latitude, longitude, radius_km, limit)

        @self.mcp.tool()
        def GetGistAssetsByCountry() -> Dict[str, Any]:  # type: ignore[misc]
            """Get asset distribution by country."""

            return self.get_assets_by_country()

        @self.mcp.tool()
        def GetGistAssetDetails(asset_id: str) -> Dict[str, Any]:  # type: ignore[misc]
            """Get detailed asset information."""

            return self.get_asset_details(asset_id)

    def _register_emissions_tools(self) -> None:
        @self.mcp.tool()
        def GetGistScope3Emissions(
            company_code: str, year: Optional[int] = None
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Get Scope 3 emissions for company."""

            return self.get_scope3_emissions(company_code, year)

        @self.mcp.tool()
        def GetGistEmissionsTrends(company_code: str) -> Dict[str, Any]:  # type: ignore[misc]
            """Get emissions trends for company."""

            return self.get_emissions_trends(company_code)

        @self.mcp.tool()
        def GetGistEmissionsBySector(year: Optional[int] = None) -> Dict[str, Any]:  # type: ignore[misc]
            """Aggregate emissions by sector."""

            return self.get_emissions_by_sector(year)

        @self.mcp.tool()
        def GetGistTopEmitters(
            limit: int = 20, year: Optional[int] = None
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Get highest emitting companies."""

            return self.get_top_emitters(limit, year)

    def _register_biodiversity_tools(self) -> None:
        @self.mcp.tool()
        def GetGistBiodiversityImpacts(
            company_code: str, year: Optional[int] = None
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Get biodiversity impacts for company."""

            return self.get_biodiversity_impacts(company_code, year)

        @self.mcp.tool()
        def GetGistBiodiversityTrends(company_code: str) -> Dict[str, Any]:  # type: ignore[misc]
            """Get biodiversity impact trends."""

            return self.get_biodiversity_trends(company_code)

        @self.mcp.tool()
        def GetGistBiodiversityBySector(year: Optional[int] = None) -> Dict[str, Any]:  # type: ignore[misc]
            """Get biodiversity impacts by sector."""

            return self.get_biodiversity_by_sector(year)

        @self.mcp.tool()
        def GetGistBiodiversityWorstPerformers(
            metric: str = "PDF", limit: int = 20, year: Optional[int] = None
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Get companies with highest biodiversity impacts."""

            return self.get_biodiversity_worst_performers(metric, limit, year)

    def _register_deforestation_tools(self) -> None:
        @self.mcp.tool()
        def GetGistDeforestationRisks(
            company_code: Optional[str] = None,
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Get deforestation proximity indicators."""

            payload = self.get_deforestation_risks(company_code)
            facts: List[str] = []
            if payload.get("error"):
                facts.append(str(payload["error"]))
            else:
                if company_code:
                    indicators = payload.get("deforestation_indicators", {}) or {}
                    yes_flags = [name for name, value in indicators.items() if value]
                    risk_level = payload.get("risk_level", "unknown").lower()
                    if yes_flags:
                        formatted = ", ".join(flag.replace("_", " ") for flag in yes_flags)
                        facts.append(
                            f"{payload.get('company_name', company_code)} triggers deforestation risk indicators: {formatted}."
                        )
                    else:
                        facts.append(
                            f"{payload.get('company_name', company_code)} shows no high deforestation proximity signals (risk level: {risk_level})."
                        )
                else:
                    total = payload.get("total_companies", 0)
                    high_fraction = payload.get("companies_high_fraction", 0)
                    extreme = payload.get("companies_extreme_proximity", 0)
                    facts.append(
                        f"GIST identifies {high_fraction} of {total} tracked companies with a high fraction of assets near recent forest change, and {extreme} with extreme proximity."
                    )
            if not facts:
                facts.append("No deforestation proximity indicators were available for this request.")
            payload["summary_fact"] = facts[0]
            payload["facts"] = facts
            payload["citation"] = {
                "id": "gist-deforestation",
                "server": "gist",
                "tool": "GetGistDeforestationRisks",
                "title": "GIST Deforestation Proximity Summary",
                "source_type": "Dataset",
                "description": _dataset_citation(DATASET_ID),
                "metadata": {"dataset": "DEFORESTATION"},
            }
            return payload

        @self.mcp.tool()
        def GetGistForestChangeProximity() -> Dict[str, Any]:  # type: ignore[misc]
            """Analyze forest change proximity across companies."""

            payload = self.get_forest_change_proximity()
            facts: List[str] = []
            if payload.get("error"):
                facts.append(str(payload["error"]))
            else:
                sectors = payload.get("sector_analysis") or {}
                if sectors:
                    top_sector, stats = max(
                        sectors.items(),
                        key=lambda item: item[1].get("high_fraction_percentage", 0),
                    )
                    facts.append(
                        f"{stats.get('high_fraction_percentage', 0)}% of companies in the {top_sector} sector have a high fraction of assets near forest change."
                    )
                total = payload.get("total_companies", 0)
                if total:
                    overall = payload.get("companies_high_fraction", 0)
                    facts.append(
                        f"Across all sectors, {overall} of {total} companies show high forest-change proximity indicators."
                    )
            if not facts:
                facts.append("No forest change proximity statistics were available for this request.")
            payload["summary_fact"] = facts[0]
            payload["facts"] = facts
            payload["citation"] = {
                "id": "gist-deforestation-sectors",
                "server": "gist",
                "tool": "GetGistForestChangeProximity",
                "title": "GIST Deforestation Proximity Summary",
                "source_type": "Dataset",
                "description": _dataset_citation(DATASET_ID),
                "metadata": {"dataset": "DEFORESTATION"},
            }
            return payload

    def _register_visualization_tools(self) -> None:
        @self.mcp.tool()
        def GetGistVisualizationData(
            viz_type: str, filters: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:  # type: ignore[misc]
            """Get data for visualization types: emissions_by_sector, risk_distribution, asset_map, biodiversity_trends, scope3_breakdown."""

            return self.get_visualization_data(viz_type, filters)

    def _register_metadata_tools(self) -> None:
        @self.mcp.tool()
        def GetGistDatasetMetadata() -> Dict[str, Any]:  # type: ignore[misc]
            """Get GIST dataset metadata."""

            return self.get_dataset_metadata()

        @self.mcp.tool()
        def DescribeServer() -> Dict[str, Any]:  # type: ignore[misc]
            """Describe this server, its datasets, key tools, and live metrics."""

            return self.describe_server()

    # ------------------------------------------------------------------ run_query helpers
    def _find_companies(self, query: str, limit: int = 3) -> List[Tuple[str, Dict[str, Any], float]]:
        if not self.data_manager.companies_cache:
            return []
        lowered = query.lower()
        tokens = set(re.findall(r"[a-z0-9]+", lowered))
        stop_tokens = {
            "brazil",
            "companies",
            "company",
            "risk",
            "risks",
            "most",
            "top",
            "highest",
            "which",
            "are",
            "exposed",
            "exposure",
            "in",
            "of",
            "to",
            "flood",
            "flooding",
            "riverine",
            "coastal",
            "heat",
            "climate",
            "environmental",
        }
        results: List[Tuple[str, Dict[str, Any], float]] = []
        for code, info in self.data_manager.companies_cache.items():
            score = 0.0
            if code and code.lower() in lowered:
                score += 1.0
            name = str(info.get("company_name") or "").strip()
            if name:
                name_lower = name.lower()
                if name_lower and name_lower in lowered:
                    score += 0.9
                else:
                    name_tokens = set(re.findall(r"[a-z0-9]+", name_lower))
                    shared = {
                        token for token in name_tokens.intersection(tokens)
                        if token and token not in stop_tokens and len(token) > 2
                    }
                    if shared:
                        score += min(0.7, 0.3 * len(shared))
            if score > 0:
                results.append((code, info, score))
        results.sort(key=lambda item: (-item[2], item[1].get("company_name", "")))
        return results[:limit]

    @staticmethod
    def _format_number(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if abs(number) >= 1_000_000_000:
            return f"{number/1_000_000_000:.2f}B"
        if abs(number) >= 1_000_000:
            return f"{number/1_000_000:.2f}M"
        if abs(number) >= 1_000:
            return f"{number/1_000:.2f}K"
        return f"{number:.2f}"

    def _extract_risk_ranking_intent(
        self,
        query: str,
        previous_user_message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not query and not previous_user_message:
            return None
        lowered_source = " ".join(
            part for part in [previous_user_message or "", query]
            if part
        )
        lowered = lowered_source.lower()
        ranking_tokens = {"most", "top", "highest", "leading", "biggest", "largest"}
        if not any(token in lowered for token in ranking_tokens):
            return None

        risk_type: Optional[str] = None
        for synonym, canonical in self.RISK_SYNONYM_MAP.items():
            if synonym in lowered:
                risk_type = canonical
                break
        if not risk_type:
            return None

        risk_level = "HIGH"
        if "very high" in lowered or "extreme" in lowered:
            risk_level = "VERY_HIGH"
        elif "moderate" in lowered and "very" not in lowered:
            risk_level = "MODERATE"
        elif "low" in lowered and "very" not in lowered:
            risk_level = "LOW"

        country = None
        for token, name in {"brazil": "Brazil", "brasil": "Brazil", "br": "Brazil"}.items():
            if token in lowered:
                country = name
                break

        return {
            "risk_type": risk_type,
            "risk_level": risk_level,
            "country": country,
        }

    @staticmethod
    def _ensure_citation(
        citations: List[CitationPayload], citation: CitationPayload
    ) -> None:
        if citation.id not in {c.id for c in citations}:
            citations.append(citation)

    @staticmethod
    def _to_native(value: Any) -> Any:
        """Convert numpy/pandas types to plain Python types for serialization."""

        if isinstance(value, dict):
            return {k: GistServerV2._to_native(v) for k, v in value.items()}
        if isinstance(value, list):
            return [GistServerV2._to_native(v) for v in value]
        if isinstance(value, tuple):
            return tuple(GistServerV2._to_native(v) for v in value)
        if isinstance(value, (np.integer,)):  # type: ignore[arg-type]
            return int(value)
        if isinstance(value, (np.floating,)):  # type: ignore[arg-type]
            return float(value)
        if isinstance(value, (np.ndarray,)):  # type: ignore[arg-type]
            return value.tolist()
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _build_company_summary(
        self,
        company_code: str,
        company_info: Dict[str, Any],
        facts: List[FactPayload],
        citations: List[CitationPayload],
        artifacts: List[ArtifactPayload],
        kg_nodes: List[Dict[str, Any]],
        kg_edges: List[Dict[str, Any]],
        messages: List[MessagePayload],
        next_actions: List[str],
    ) -> None:
        profile = self.get_company_profile(company_code)
        if profile.get("error"):
            messages.append(MessagePayload(level="warning", text=profile["error"]))
            return
        company_name = profile.get("company_name", company_code)
        sector = profile.get("sector_code", "Unknown")
        country = profile.get("country", "Unknown")
        _ensure_root_node(kg_nodes)
        if not any(node.get("id") == company_code for node in kg_nodes):
            kg_nodes.append(
                {
                    "id": company_code,
                    "type": "company",
                    "label": company_name,
                    "sector": sector,
                    "country": country,
                }
            )
            kg_edges.append(
                {
                    "source": company_code,
                    "target": DATASET_NODE_ID,
                    "type": "has_data",
                    "dataset": "GIST",
                }
            )

        risk_data = self.get_company_risks(company_code)
        risk_citation_id = f"gist-risk-{company_code.lower()}"
        if not risk_data.get("error"):
            self._ensure_citation(
                citations,
                CitationPayload(
                    id=risk_citation_id,
                    server="gist",
                    tool="GetGistCompanyRisks",
                    title=f"GIST Environmental Risk Assessment – {company_name}",
                    source_type="Dataset",
                    description=_dataset_citation(DATASET_ID),
                    metadata={"dataset": "EXSITU", "company_code": company_code},
                ),
            )
            top_risks = (risk_data.get("high_risk_summary") or {}).get(
                "top_risk_categories", []
            )
            if top_risks:
                top_strings = [
                    f"{item['category'].replace('_', ' ')} ({item['high_risk_percentage']}% high risk)"
                    for item in top_risks[:3]
                ]
                facts.append(
                    FactPayload(
                        id=f"gist-risk-summary-{company_code.lower()}",
                        text=(
                            f"{company_name} ({company_code}) shows the highest high-risk exposure in "
                            f"{', '.join(top_strings)} based on {risk_data.get('total_assets', 0)} assessed assets."
                        ),
                        citation_id=risk_citation_id,
                        kind="text",
                        metadata={"company_code": company_code, "sector": sector},
                    )
                )
                table_rows = [
                    {
                        "category": item["category"],
                        "high_risk_assets": item["high_risk_assets"],
                        "percentage": item["high_risk_percentage"],
                    }
                    for item in top_risks[:5]
                ]
                artifacts.append(
                    ArtifactPayload(
                        id=f"gist-risk-table-{company_code.lower()}",
                        type="table",
                        title=f"Top risk categories for {company_name}",
                        data={
                            "columns": ["category", "high_risk_assets", "percentage"],
                            "rows": table_rows,
                        },
                    )
                )
        elif risk_data.get("error"):
            messages.append(MessagePayload(level="info", text=risk_data["error"]))

        emissions_trends = self.get_emissions_trends(company_code)
        emissions_citation_id = f"gist-emissions-{company_code.lower()}"
        scope3_series = self.get_scope3_emissions(company_code, year=None)
        if not emissions_trends.get("error"):
            trends = emissions_trends.get("trends", {})
            totals = trends.get("total_emissions")
            analysis_period = emissions_trends.get("analysis_period", {})
            if totals:
                start_val = totals.get("start_value")
                end_val = totals.get("end_value")
                pct_change = totals.get("percent_change")
                start_year = analysis_period.get("start_year")
                end_year = analysis_period.get("end_year")
                self._ensure_citation(
                    citations,
                CitationPayload(
                    id=emissions_citation_id,
                    server="gist",
                    tool="GetGistEmissionsTrends",
                    title=f"GIST Scope 3 Emissions Trends – {company_name}",
                    source_type="Dataset",
                    description=_dataset_citation(DATASET_ID),
                    metadata={"dataset": "SCOPE_3_DATA", "company_code": company_code},
                ),
            )
                facts.append(
                    FactPayload(
                        id=f"gist-emissions-trend-{company_code.lower()}",
                        text=(
                            f"Between {start_year} and {end_year}, {company_name} changed total Scope 3 emissions "
                            f"from {self._format_number(start_val)} to {self._format_number(end_val)} tonnes CO₂e, "
                            f"a {self._format_number(pct_change)}% shift."
                        ),
                        citation_id=emissions_citation_id,
                        kind="text",
                        metadata={"company_code": company_code},
                    )
                )
        if not scope3_series.get("error"):
            series = scope3_series.get("emissions_by_year", [])
            if series:
                self._ensure_citation(
                    citations,
                CitationPayload(
                    id=emissions_citation_id,
                    server="gist",
                    tool="GetGistScope3Emissions",
                    title=f"GIST Scope 3 Emissions Records – {company_name}",
                    source_type="Dataset",
                    description=_dataset_citation(DATASET_ID),
                    metadata={"dataset": "SCOPE_3_DATA", "company_code": company_code},
                ),
            )
                chronological = [
                    item for item in series if item.get("reporting_year") is not None
                ]
                chronological.sort(key=lambda row: row.get("reporting_year"))

                labels = [row["reporting_year"] for row in chronological]

                datasets = []
                palette = [
                    "#2196F3",  # total
                    "#4CAF50",  # upstream
                    "#FF9800",  # downstream
                ]

                def _append_series(label: str, key: str, color: str, fill: bool = False) -> None:
                    values = [row.get(key, 0) or 0 for row in chronological]
                    if any(values):
                        datasets.append(
                            {
                                "label": label,
                                "data": values,
                                "borderColor": color,
                                "tension": 0.1,
                                "fill": fill,
                            }
                        )

                # _append_series("Total Scope 3 Emissions", "total_scope3_emissions", palette[0])
                # _append_series("Upstream Emissions", "upstream_emissions", palette[1])
                # _append_series("Downstream Emissions", "downstream_emissions", palette[2])

                # if datasets:
                #     artifacts.append(
                #         ArtifactPayload(
                #             id=f"gist-scope3-trend-{company_code.lower()}",
                #             type="chart",
                #             title=f"Scope 3 emissions by year – {company_name}",
                #             data={
                #                 "labels": labels,
                #                 "datasets": datasets,
                #             },
                #             metadata={
                #                 "chartType": "line",
                #                 "options": {
                #                     "responsive": True,
                #                     "plugins": {
                #                         "title": {
                #                             "display": True,
                #                             "text": f"Scope 3 emissions by year – {company_name}",
                #                         }
                #                     },
                #                 },
                #             },
                #         )
                #     )
        elif scope3_series.get("error"):
            messages.append(MessagePayload(level="info", text=scope3_series["error"]))

        biodiversity = self.get_biodiversity_impacts(company_code, year=None)
        biodiversity_citation_id = f"gist-biodiversity-{company_code.lower()}"
        if not biodiversity.get("error"):
            impacts = biodiversity.get("impacts_by_year", [])
            if impacts:
                latest = impacts[-1]
                self._ensure_citation(
                    citations,
                CitationPayload(
                    id=biodiversity_citation_id,
                    server="gist",
                    tool="GetGistBiodiversityImpacts",
                    title=f"GIST Biodiversity Impacts – {company_name}",
                    source_type="Dataset",
                    description=_dataset_citation(DATASET_ID),
                    metadata={"dataset": "BIODIVERSITY_PDF_DATA", "company_code": company_code},
                ),
            )
                total_pdf = latest.get("total_impacts", {}).get("pdf")
                facts.append(
                    FactPayload(
                        id=f"gist-biodiversity-{company_code.lower()}",
                        text=(
                            f"GIST reports biodiversity pressure for {company_name} at {self._format_number(total_pdf)} PDF in "
                            f"{latest.get('reporting_year')} across {len(impacts)} recorded years."
                        ),
                        citation_id=biodiversity_citation_id,
                        kind="text",
                        metadata={"company_code": company_code},
                    )
                )
        next_actions.extend(
            [
                f"GetGistCompanyProfile(company_code='{company_code}')",
                f"GetGistCompanyRisks(company_code='{company_code}')",
                f"GetGistScope3Emissions(company_code='{company_code}')",
            ]
        )

    def _build_aggregate_summary(
        self,
        topic: str,
        query: str,
        facts: List[FactPayload],
        citations: List[CitationPayload],
        artifacts: List[ArtifactPayload],
        messages: List[MessagePayload],
        next_actions: List[str],
        ) -> None:
        if topic == "emissions":
            ranking = self.get_top_emitters(20, year=None)
            if not ranking.get("error"):
                top_emitters = ranking.get("top_emitters", [])
                if top_emitters:
                    analysis_year = ranking.get("analysis_year")
                    leader = top_emitters[0]
                    citation_id = "gist-emissions-summary"
                    self._ensure_citation(
                        citations,
                CitationPayload(
                    id=citation_id,
                    server="gist",
                    tool="GetGistTopEmitters",
                    title="GIST Scope 3 Top Emitters",
                    source_type="Dataset",
                    description=_dataset_citation(DATASET_ID),
                    metadata={"dataset": "SCOPE_3_DATA", "analysis_year": analysis_year},
                ),
            )
                    facts.append(
                        FactPayload(
                            id="gist-top-emitter",
                            text=(
                                f"In {analysis_year}, {leader['company_name']} ({leader['company_code']}) "
                                f"reported the highest Scope 3 emissions at {self._format_number(leader['total_scope3_emissions'])} tonnes CO₂e."
                            ),
                            citation_id=citation_id,
                            kind="text",
                            metadata={"analysis_year": analysis_year},
                        )
                    )
                    artifacts.append(
                        ArtifactPayload(
                            id="gist-top-emitters-table",
                            type="table",
                            title=f"Top Scope 3 emitters ({analysis_year})",
                            data={
                                "columns": ["rank", "company_name", "company_code", "total_scope3_emissions"],
                                "rows": [
                                    {
                                        "rank": item.get("rank"),
                                        "company_name": item.get("company_name"),
                                        "company_code": item.get("company_code"),
                                        "total_scope3_emissions": item.get("total_scope3_emissions"),
                                    }
                                    for item in top_emitters
                                ],
                            },
                        )
                    )
                    next_actions.append("GetGistTopEmitters(limit=10)")
                else:
                    messages.append(
                        MessagePayload(
                            level="info",
                            text="Scope 3 emissions data is available but no emitters met the filtering criteria.",
                        )
                    )
            elif ranking.get("error"):
                messages.append(MessagePayload(level="info", text=ranking["error"]))
        elif topic == "biodiversity":
            worst = self.get_biodiversity_worst_performers(metric="PDF", limit=20, year=None)
            if not worst.get("error"):
                performers = worst.get("worst_performers", [])
                if performers:
                    year = worst.get("analysis_year")
                    citation_id = "gist-biodiversity-summary"
                    self._ensure_citation(
                        citations,
                CitationPayload(
                    id=citation_id,
                    server="gist",
                    tool="GetGistBiodiversityWorstPerformers",
                    title="GIST Biodiversity Impacts Leaders",
                    source_type="Dataset",
                    description=_dataset_citation(DATASET_ID),
                    metadata={"dataset": "BIODIVERSITY_PDF_DATA", "analysis_year": year},
                ),
            )
                    leader = performers[0]
                    facts.append(
                        FactPayload(
                            id="gist-biodiversity-leader",
                            text=(
                                f"{leader['company_name']} ({leader['company_code']}) recorded the highest biodiversity pressure "
                                f"in {year} with {self._format_number(leader['impact_value'])} {leader['impact_metric']}."
                            ),
                            citation_id=citation_id,
                            kind="text",
                            metadata={"analysis_year": year},
                        )
                    )
                    artifacts.append(
                        ArtifactPayload(
                            id="gist-biodiversity-table",
                            type="table",
                            title=f"Highest biodiversity impacts ({year})",
                            data={
                                "columns": [
                                    "rank",
                                    "company_name",
                                    "company_code",
                                    "impact_metric",
                                    "impact_value",
                                ],
                                "rows": performers,
                            },
                        )
                    )
                    next_actions.append("GetGistBiodiversityWorstPerformers(limit=10)")
            elif worst.get("error"):
                messages.append(MessagePayload(level="info", text=worst["error"]))
        elif topic == "deforestation":
            risks = self.get_deforestation_risks(company_code=None)
            if not risks.get("error"):
                citation_id = "gist-deforestation-summary"
                self._ensure_citation(
                    citations,
                CitationPayload(
                    id=citation_id,
                    server="gist",
                    tool="GetGistDeforestationRisks",
                    title="GIST Deforestation Proximity Summary",
                    source_type="Dataset",
                    description=_dataset_citation(DATASET_ID),
                    metadata={"dataset": "DEFORESTATION"},
                ),
            )
                high_fraction = risks.get("companies_high_fraction", 0)
                total = risks.get("total_companies", 0)
                facts.append(
                    FactPayload(
                        id="gist-deforestation-summary",
                        text=(
                            f"GIST identifies {high_fraction} of {total} tracked companies with a high fraction of assets near recent forest change."
                        ),
                        citation_id=citation_id,
                        kind="text",
                    )
                )
                artifacts.append(
                    ArtifactPayload(
                        id="gist-deforestation-stats",
                        type="table",
                        title="Deforestation proximity indicators",
                        data={
                            "columns": list(risks.keys()),
                            "rows": [risks],
                        },
                    )
                )
                next_actions.append("GetGistDeforestationRisks()")
            elif risks.get("error"):
                messages.append(MessagePayload(level="info", text=risks["error"]))
        elif topic == "assets":
            # Deprecated in this context – avoid adding noisy global tables.
            pass

    def _build_risk_ranking_summary(
        self,
        intent: Dict[str, Any],
        query: str,
        facts: List[FactPayload],
        citations: List[CitationPayload],
        artifacts: List[ArtifactPayload],
        messages: List[MessagePayload],
        next_actions: List[str],
        limit: int = 5,
    ) -> None:
        risk_type = intent["risk_type"]
        risk_level = intent.get("risk_level", "HIGH")
        country = intent.get("country")
        context_guidance = intent.get("context_guidance")

        ranking = self.get_risk_by_category(
            risk_type=risk_type,
            risk_level=risk_level,
            limit=limit,
            country=country,
            sector=None,
        )
        if ranking.get("error"):
            messages.append(MessagePayload(level="info", text=ranking["error"]))
            return

        companies = ranking.get("companies") or []
        if not companies:
            messages.append(
                MessagePayload(
                    level="info",
                    text="GIST risk rankings did not identify companies that meet the requested criteria.",
                )
            )
            return

        risk_label = risk_type.replace("_", " ").lower()
        level_label = risk_level.replace("_", " ").lower()
        location_phrase = f" in {country}" if country else ""

        citation_id = f"gist-{risk_type.lower()}-{risk_level.lower()}-ranking"
        self._ensure_citation(
            citations,
            CitationPayload(
                id=citation_id,
                server="gist",
                tool="GetGistRiskByCategory",
                title=f"GIST hazard ranking: {risk_label.title()}",
                source_type="Dataset",
                description=_dataset_citation(DATASET_ID),
                metadata={
                    "risk_type": risk_type,
                    "risk_level": risk_level,
                    "country": country,
                },
            ),
        )

        raw_leader = companies[0]
        if not isinstance(raw_leader, dict):
            messages.append(
                MessagePayload(
                    level="info",
                    text="Unable to parse risk ranking results for summary generation.",
                )
            )
            return

        leader = {key: self._to_native(value) for key, value in raw_leader.items()}
        facts.append(
            FactPayload(
                id=f"gist-{risk_type.lower()}-leader",
                text=(
                    f"{leader['company_name']} ({leader['company_code']}) has the greatest share of assets at {level_label} {risk_label}{location_phrase}, "
                    f"with {leader['high_risk_assets']} assets representing {leader['high_risk_percentage']}% of those assessed."
                ),
                citation_id=citation_id,
                kind="text",
                metadata={
                    "risk_type": risk_type,
                    "risk_level": risk_level,
                    "country": country,
                    "query": query,
                    **({"context_guidance": context_guidance} if context_guidance else {}),
                },
            )
        )

        rows: List[Dict[str, Any]] = []
        for idx, company in enumerate(companies, start=1):
            normalized = {
                key: self._to_native(value)
                for key, value in (company.items() if isinstance(company, dict) else [])
            }
            rows.append(
                {
                    "rank": idx,
                    "company_name": normalized.get("company_name"),
                    "company_code": normalized.get("company_code"),
                    "country": normalized.get("country"),
                    "high_risk_assets": normalized.get("high_risk_assets"),
                    "high_risk_percentage": normalized.get("high_risk_percentage"),
                }
            )

        artifacts.append(
            ArtifactPayload(
                id=f"gist-{risk_type.lower()}-ranking-table",
                type="table",
                title=(
                    f"Top companies by {risk_label} exposure"
                    + (f" ({country})" if country else "")
                ),
                data={
                    "columns": [
                        "rank",
                        "company_name",
                        "company_code",
                        "country",
                        "high_risk_assets",
                        "high_risk_percentage",
                    ],
                    "rows": rows,
                },
            )
        )

        next_actions.append(
            "GetGistRiskByCategory(risk_type=..., risk_level='HIGH', country='BRA')"
        )
        return


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

        previous_user_message = None
        previous_assistant_message = None
        if isinstance(context, Mapping):
            previous_user_message = context.get("previous_user_message") or None
            previous_assistant_message = context.get("previous_assistant_message") or None

        analysis_query = query
        if previous_user_message:
            prior = str(previous_user_message).strip()
            if prior:
                analysis_query = f"{prior}\nFollow-up question: {query.strip()}".strip()

        guidance_text = None
        if previous_user_message:
            guidance_text = (
                "Previous user message (use only if relevant to the current request): "
                f"{str(previous_user_message).strip()}"
            )

        if not self.data_manager.sheets:
            messages.append(
                MessagePayload(
                    level="error",
                    text="GIST data is unavailable. Ensure data/gist/gist.xlsx is present on the server.",
                )
            )
            return RunQueryResponse(
                server="gist",
                query=query,
                facts=facts,
                citations=citations,
                artifacts=artifacts,
                messages=messages,
                kg=KnowledgeGraphPayload(nodes=kg_nodes, edges=kg_edges),
                next_actions=next_actions,
                duration_ms=int((time.perf_counter() - start_time) * 1000),
            )

        dataset_citation = CitationPayload(
            id="gist-dataset",
            server="gist",
            tool="GetGistDatasetMetadata",
            title=DATASET_TITLE,
            source_type="Dataset",
            description=_dataset_citation(DATASET_ID)
            or "Comprehensive corporate environmental metrics spanning risk, biodiversity, emissions, and asset-level exposure.",
            metadata={
                "tables": self.metadata.get("Datasets", []),
                "total_companies": self.metadata.get("Total_Companies"),
            },
        )
        self._ensure_citation(citations, dataset_citation)
        facts.append(
            FactPayload(
                id="gist-overview",
                text=(
                    "The GIST Impact Datasets combine multi-sheet Excel sources covering corporate environmental risks, Scope 3 emissions, "
                    "biodiversity pressure, deforestation proximity, and asset-level geospatial exposure."
                ),
                citation_id="gist-dataset",
                kind="text",
                metadata={"tables": self.metadata.get("Datasets", [])},
            )
        )

        company_matches = self._find_companies(analysis_query)
        if company_matches:
            for company_code, info, _score in company_matches:
                self._build_company_summary(
                    company_code,
                    info,
                    facts,
                    citations,
                    artifacts,
                    kg_nodes,
                    kg_edges,
                    messages,
                    next_actions,
                )
        else:
            ranking_intent = self._extract_risk_ranking_intent(query, previous_user_message)
            if ranking_intent:
                if guidance_text:
                    ranking_intent.setdefault("context_guidance", guidance_text)
                self._build_risk_ranking_summary(
                    ranking_intent,
                    query,
                    facts,
                    citations,
                    artifacts,
                    messages,
                    next_actions,
                )

            lowered = analysis_query.lower()
            if any(keyword in lowered for keyword in ["scope 3", "emission", "ghg"]):
                self._build_aggregate_summary(
                    "emissions", query, facts, citations, artifacts, messages, next_actions
                )
            if "biodiversity" in lowered:
                self._build_aggregate_summary(
                    "biodiversity", query, facts, citations, artifacts, messages, next_actions
                )
            if "deforest" in lowered or "forest" in lowered:
                self._build_aggregate_summary(
                    "deforestation", query, facts, citations, artifacts, messages, next_actions
                )
            if any(keyword in lowered for keyword in ["asset", "map", "geospatial", "location"]):
                self._build_aggregate_summary(
                    "assets", query, facts, citations, artifacts, messages, next_actions
                )
            if len(facts) <= 1:
                messages.append(
                    MessagePayload(
                        level="info",
                        text=(
                            "Specify a company code, sector, or metric (e.g., Scope 3 emissions, biodiversity impacts, deforestation risk) "
                            "to receive focused metrics from the GIST dataset."
                        ),
                    )
                )
                next_actions.extend(
                    [
                        "GetGistCompanies(limit=25)",
                        "GetGistDatasetSchemas()",
                    ]
                )

        for citation in citations:
            citation.metadata = self._to_native(citation.metadata)
        for fact in facts:
            if fact.metadata:
                fact.metadata = self._to_native(fact.metadata)
            if fact.data:
                fact.data = self._to_native(fact.data)
        for artifact in artifacts:
            if artifact.metadata:
                artifact.metadata = self._to_native(artifact.metadata)
            if artifact.data:
                artifact.data = self._to_native(artifact.data)

        kg_nodes = self._to_native(kg_nodes)
        kg_edges = self._to_native(kg_edges)

        duration_ms = int((time.perf_counter() - start_time) * 1000)
        return RunQueryResponse(
            server="gist",
            query=query,
            facts=facts,
            citations=citations,
            artifacts=artifacts,
            messages=messages,
            kg=KnowledgeGraphPayload(nodes=kg_nodes, edges=kg_edges),
            next_actions=list(dict.fromkeys(next_actions)),
            duration_ms=duration_ms,
        )


def _ensure_root_node(kg_nodes: List[Dict[str, Any]]) -> None:
    if not any(node.get("id") == DATASET_NODE_ID for node in kg_nodes):
        kg_nodes.append(
            {
                "id": DATASET_NODE_ID,
                "type": "dataset",
                "label": DATASET_TITLE,
                "source": DATASET_SOURCE,
            }
        )


def create_server() -> FastMCP:
    """Entry point used by ``python -m mcp.servers_v2.gist_server_v2``."""

    server = GistServerV2()
    return server.mcp


if __name__ == "__main__":  # pragma: no cover - manual execution
    create_server().run()
