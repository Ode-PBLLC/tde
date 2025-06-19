import pandas as pd
import numpy as np
from fastmcp import FastMCP
from typing import List, Optional, Dict, Any, Union
import json
import os
from functools import lru_cache

mcp = FastMCP("gist-server")

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
GIST_FILE_PATH = os.path.join(project_root, "data", "gist", "gist.xlsx")

class GistDataManager:
    """
    Manages loading and caching of GIST data across all 6 Excel sheets.
    Provides efficient access to sustainability, environmental, and emissions data.
    """
    
    def __init__(self):
        self.excel_file = None
        self.sheets = {}
        self.companies_cache = {}
        self.asset_cache = {}
        self._load_data()
    
    def _load_data(self):
        """Load all sheets from the GIST Excel file."""
        global GIST_FILE_PATH
        try:
            print(f"Attempting to load GIST data from: {GIST_FILE_PATH}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"File exists: {os.path.exists(GIST_FILE_PATH)}")
            
            if not os.path.exists(GIST_FILE_PATH):
                # Try alternative paths
                alternative_paths = [
                    os.path.join(os.getcwd(), "data", "gist", "gist.xlsx"),
                    os.path.join(os.path.dirname(os.getcwd()), "data", "gist", "gist.xlsx"),
                    "./data/gist/gist.xlsx",
                    "../data/gist/gist.xlsx"
                ]
                
                for alt_path in alternative_paths:
                    print(f"Trying alternative path: {alt_path}")
                    if os.path.exists(alt_path):
                        GIST_FILE_PATH = alt_path
                        print(f"✓ Found GIST file at: {GIST_FILE_PATH}")
                        break
                else:
                    raise FileNotFoundError(f"GIST file not found at {GIST_FILE_PATH} or alternative paths")
            
            self.excel_file = pd.ExcelFile(GIST_FILE_PATH)
            print(f"Loading GIST data from {len(self.excel_file.sheet_names)} sheets...")
            
            # Load all sheets with progress indication
            for i, sheet_name in enumerate(self.excel_file.sheet_names, 1):
                print(f"Loading sheet {i}/{len(self.excel_file.sheet_names)}: {sheet_name}...")
                self.sheets[sheet_name] = pd.read_excel(GIST_FILE_PATH, sheet_name=sheet_name)
                print(f"✓ Loaded {sheet_name}: {self.sheets[sheet_name].shape}")
            
            # Build indexes for fast lookups
            self._build_company_index()
            self._build_asset_index()
            
            print("GIST data loading completed successfully")
            
        except FileNotFoundError as e:
            print(f"Error: GIST file not found - {e}")
            self.sheets = {}
        except Exception as e:
            print(f"Error loading GIST data: {e}")
            import traceback
            traceback.print_exc()
            self.sheets = {}
    
    def _build_company_index(self):
        """Build a comprehensive index of all companies across datasets."""
        self.companies_cache = {}
        
        # Collect all companies from data sheets (skip Data Dictionary)
        for sheet_name, df in self.sheets.items():
            if sheet_name == 'Data Dictionary' or 'COMPANY_CODE' not in df.columns:
                continue
                
            for _, row in df.iterrows():
                company_code = row['COMPANY_CODE']
                company_name = row.get('COMPANY_NAME', 'Unknown')
                
                if company_code not in self.companies_cache:
                    self.companies_cache[company_code] = {
                        'company_code': company_code,
                        'company_name': company_name,
                        'datasets': [],
                        'sector_code': row.get('SECTOR_CODE', 'Unknown'),
                        'country': row.get('COUNTRY_NAME', row.get('COUNTRY_CODE', 'Unknown'))
                    }
                
                if sheet_name not in self.companies_cache[company_code]['datasets']:
                    self.companies_cache[company_code]['datasets'].append(sheet_name)
    
    def _build_asset_index(self):
        """Build index for asset-level data."""
        if 'EXSITU_ASSET_DATA' in self.sheets:
            asset_df = self.sheets['EXSITU_ASSET_DATA']
            self.asset_cache = {
                'by_company': asset_df.groupby('COMPANY_CODE').size().to_dict(),
                'by_country': asset_df.groupby('COUNTRY_CODE').size().to_dict(),
                'total_assets': len(asset_df)
            }
    
    def get_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Get a specific sheet by name."""
        if not self.sheets:
            print(f"WARNING: No GIST data loaded, returning empty DataFrame for {sheet_name}")
        return self.sheets.get(sheet_name, pd.DataFrame())
    
    def get_companies(self, sector: Optional[str] = None, country: Optional[str] = None) -> List[Dict]:
        """Get list of companies with optional filtering."""
        companies = list(self.companies_cache.values())
        
        if sector:
            companies = [c for c in companies if c['sector_code'] == sector]
        
        if country:
            companies = [c for c in companies if country.lower() in c['country'].lower()]
        
        return companies
    
    def get_company_data(self, company_code: str) -> Dict[str, Any]:
        """Get all data for a specific company across all datasets."""
        company_data = {'company_code': company_code, 'datasets': {}}
        
        for sheet_name, df in self.sheets.items():
            if sheet_name == 'Data Dictionary' or 'COMPANY_CODE' not in df.columns:
                continue
                
            company_rows = df[df['COMPANY_CODE'] == company_code]
            if not company_rows.empty:
                company_data['datasets'][sheet_name] = company_rows.to_dict('records')
        
        return company_data

# Initialize the data manager
data_manager = GistDataManager()

metadata = {
    "Name": "GIST Server",
    "Description": "Global Infrastructure Sustainability Toolkit (GIST) data access server",
    "Version": "1.0.0", 
    "Author": "Climate Policy Radar Team",
    "Dataset": "GIST Multi-Dataset Collection",
    "Total_Companies": len(data_manager.companies_cache),
    "Total_Assets": data_manager.asset_cache.get('total_assets', 0),
    "Datasets": list(data_manager.sheets.keys()) if data_manager.sheets else []
}

# =============================================================================
# SCHEMA/DICTIONARY TOOLS
# =============================================================================

@mcp.tool()
def GetGistDataDictionary(dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get field definitions from the GIST Data Dictionary.
    
    Parameters:
    - dataset_name: Optional filter for specific dataset
    
    Returns dictionary with field definitions and metadata.
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    dict_df = data_manager.get_sheet('Data Dictionary')
    if dict_df.empty:
        return {"error": "Data Dictionary not found"}
    
    if dataset_name:
        dict_df = dict_df[dict_df['Dataset'].str.contains(dataset_name, case=False, na=False)]
    
    # Group by dataset
    result = {"datasets": {}, "total_fields": len(dict_df)}
    
    for dataset, group in dict_df.groupby('Dataset'):
        if pd.isna(dataset):
            continue
            
        result["datasets"][dataset] = {
            "field_count": len(group),
            "fields": []
        }
        
        for _, row in group.iterrows():
            result["datasets"][dataset]["fields"].append({
                "field_name": row['Field Name'],
                "unit": row['Unit'], 
                "definition": row['Definition']
            })
    
    return result

@mcp.tool()
def SearchGistFields(search_term: str) -> Dict[str, Any]:
    """
    Search across all field names and definitions in the GIST data dictionary.
    
    Parameters:
    - search_term: Term to search for in field names and definitions
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    dict_df = data_manager.get_sheet('Data Dictionary')
    if dict_df.empty:
        return {"error": "Data Dictionary not found"}
    
    # Search in field names and definitions
    mask = (dict_df['Field Name'].str.contains(search_term, case=False, na=False) |
            dict_df['Definition'].str.contains(search_term, case=False, na=False))
    
    matches = dict_df[mask]
    
    result = {
        "search_term": search_term,
        "matches_found": len(matches),
        "matches": []
    }
    
    for _, row in matches.iterrows():
        result["matches"].append({
            "dataset": row['Dataset'],
            "field_name": row['Field Name'],
            "unit": row['Unit'],
            "definition": row['Definition'][:200] + "..." if len(str(row['Definition'])) > 200 else row['Definition']
        })
    
    return result

@mcp.tool()
def GetGistDatasetSchemas() -> Dict[str, Any]:
    """
    List all available GIST datasets with their field counts and basic information.
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    result = {
        "total_datasets": len(data_manager.sheets),
        "datasets": {}
    }
    
    for sheet_name, df in data_manager.sheets.items():
        result["datasets"][sheet_name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "description": _get_dataset_description(sheet_name)
        }
    
    return result

def _get_dataset_description(sheet_name: str) -> str:
    """Get description for each dataset."""
    descriptions = {
        'Data Dictionary': 'Field definitions and schema for all datasets',
        'EXSITU': 'Company-level aggregated environmental risk data across 75 metrics',
        'EXSITU_ASSET_DATA': 'Individual asset-level environmental risk data with coordinates (40K+ assets)',
        'DEFORESTATION': 'Company deforestation risk proximity indicators',
        'SCOPE_3_DATA': 'Multi-year Scope 3 emissions data by company (2016-2024)',
        'BIODIVERSITY_PDF_DATA': 'Multi-year biodiversity impact data with PDF, CO2E, and LCE metrics'
    }
    return descriptions.get(sheet_name, 'Dataset information')

# =============================================================================
# COMPANY DISCOVERY & ANALYSIS TOOLS  
# =============================================================================

@mcp.tool()
def GetGistCompanies(sector: Optional[str] = None, country: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Get list of companies in GIST database with optional filtering.
    
    Parameters:
    - sector: Filter by sector code (OGES, FINS, WHRE, MOMI, REEN)
    - country: Filter by country name
    - limit: Maximum number of companies to return
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    companies = data_manager.get_companies(sector=sector, country=country)
    
    # Limit results
    companies = companies[:limit]
    
    result = {
        "total_companies": len(data_manager.companies_cache),
        "filtered_companies": len(companies),
        "filters_applied": {"sector": sector, "country": country},
        "companies": companies
    }
    
    return result

@mcp.tool()
def GetGistCompanyProfile(company_code: str) -> Dict[str, Any]:
    """
    Get complete profile for a specific company across all GIST datasets.
    
    Parameters:
    - company_code: Unique company identifier
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    if company_code not in data_manager.companies_cache:
        return {"error": f"Company {company_code} not found"}
    
    company_info = data_manager.companies_cache[company_code]
    company_data = data_manager.get_company_data(company_code)
    
    # Get summary statistics
    profile = {
        "company_code": company_code,
        "company_name": company_info['company_name'],
        "sector_code": company_info['sector_code'],
        "country": company_info['country'],
        "datasets_available": company_info['datasets'],
        "data_summary": {}
    }
    
    # Add summary for each dataset
    for dataset, records in company_data['datasets'].items():
        if dataset == 'EXSITU_ASSET_DATA':
            profile["data_summary"][dataset] = {
                "total_assets": len(records),
                "countries": list(set([r.get('COUNTRY_CODE', 'Unknown') for r in records])),
                "asset_types": list(set([r.get('ASSET_TYPE_LEVEL_1', 'Unknown') for r in records]))
            }
        elif dataset == 'SCOPE_3_DATA':
            years = [r.get('REPORTING_YEAR') for r in records if r.get('REPORTING_YEAR')]
            profile["data_summary"][dataset] = {
                "reporting_years": sorted(years) if years else [],
                "total_records": len(records)
            }
        else:
            profile["data_summary"][dataset] = {
                "total_records": len(records)
            }
    
    return profile

@mcp.tool()
def GetGistCompaniesBySector() -> Dict[str, Any]:
    """
    Get companies grouped by sector with counts and basic statistics.
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    sector_groups = {}
    for company in data_manager.companies_cache.values():
        sector = company['sector_code']
        if sector not in sector_groups:
            sector_groups[sector] = {
                "companies": [],
                "countries": set(),
                "total_count": 0
            }
        
        sector_groups[sector]["companies"].append({
            "company_code": company['company_code'],
            "company_name": company['company_name'],
            "country": company['country']
        })
        sector_groups[sector]["countries"].add(company['country'])
        sector_groups[sector]["total_count"] += 1
    
    # Convert sets to lists for JSON serialization
    for sector in sector_groups:
        sector_groups[sector]["countries"] = list(sector_groups[sector]["countries"])
        sector_groups[sector]["companies"] = sector_groups[sector]["companies"][:10]  # Limit for display
    
    return {
        "total_sectors": len(sector_groups),
        "sectors": sector_groups
    }

@mcp.tool()
def GetGistCompanyDataAvailability(company_code: str) -> Dict[str, Any]:
    """
    Show which datasets contain data for a specific company.
    
    Parameters:
    - company_code: Unique company identifier
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    if company_code not in data_manager.companies_cache:
        return {"error": f"Company {company_code} not found"}
    
    company_data = data_manager.get_company_data(company_code)
    
    availability = {
        "company_code": company_code,
        "company_name": data_manager.companies_cache[company_code]['company_name'],
        "data_availability": {}
    }
    
    for sheet_name in data_manager.sheets.keys():
        if sheet_name == 'Data Dictionary':
            continue
            
        if sheet_name in company_data['datasets']:
            records = company_data['datasets'][sheet_name]
            availability["data_availability"][sheet_name] = {
                "available": True,
                "record_count": len(records),
                "data_summary": _get_data_summary(sheet_name, records)
            }
        else:
            availability["data_availability"][sheet_name] = {
                "available": False,
                "record_count": 0
            }
    
    return availability

def _get_data_summary(dataset_name: str, records: List[Dict]) -> Dict[str, Any]:
    """Get summary information for dataset records."""
    if not records:
        return {}
    
    if dataset_name == 'SCOPE_3_DATA' or dataset_name == 'BIODIVERSITY_PDF_DATA':
        years = [r.get('REPORTING_YEAR') for r in records if r.get('REPORTING_YEAR')]
        return {"years_available": sorted(list(set(years))) if years else []}
    
    elif dataset_name == 'EXSITU_ASSET_DATA':
        countries = [r.get('COUNTRY_CODE') for r in records if r.get('COUNTRY_CODE')]
        return {"countries": list(set(countries)) if countries else []}
    
    return {"records": len(records)}

# =============================================================================
# ENVIRONMENTAL RISK ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
def GetGistCompanyRisks(company_code: str) -> Dict[str, Any]:
    """
    Get environmental risk summary for a company from EXSITU data.
    
    Parameters:
    - company_code: Unique company identifier
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    exsitu_df = data_manager.get_sheet('EXSITU')
    if exsitu_df.empty:
        return {"error": "EXSITU risk data not available"}
    
    company_data = exsitu_df[exsitu_df['COMPANY_CODE'] == company_code]
    if company_data.empty:
        return {"error": f"No risk data found for company {company_code}"}
    
    row = company_data.iloc[0]
    
    # Extract risk categories
    risk_categories = {
        "biodiversity": _extract_risk_metrics(row, "MSA"),
        "water_stress": _extract_risk_metrics(row, "WATER_STRESS"), 
        "water_demand": _extract_risk_metrics(row, "WATER_DEMAND"),
        "water_variability": _extract_risk_metrics(row, "WATER_VARIABILITY"),
        "drought": _extract_risk_metrics(row, "DROUGHT"),
        "flood_coastal": _extract_risk_metrics(row, "FLOOD_COASTAL"),
        "flood_riverine": _extract_risk_metrics(row, "FLOOD_RIVERINE"),
        "extreme_heat": _extract_risk_metrics(row, "EXTREME_HEAT"),
        "extreme_precipitation": _extract_risk_metrics(row, "EXTREME_PRECIPITATION"),
        "temperature_anomaly": _extract_risk_metrics(row, "TEMPERATURE_ANOMALY"),
        "urban_area_change": _extract_risk_metrics(row, "URBAN_AREA_CHANGE"),
        "agriculture_area_change": _extract_risk_metrics(row, "AGRICULTURE_AREA_CHANGE"),
        "forest_area_change": _extract_risk_metrics(row, "FOREST_AREA_CHANGE")
    }
    
    # Calculate overall risk score
    total_assets = row.get('TOTAL_NUMBER_OF_ASSETS_ASSESSED_WITHIN_A_COMPANY', 0)
    
    return {
        "company_code": company_code,
        "company_name": row.get('COMPANY_NAME', 'Unknown'),
        "sector_code": row.get('SECTOR_CODE', 'Unknown'),
        "country": row.get('COUNTRY_NAME', 'Unknown'),
        "total_assets": total_assets,
        "risk_categories": risk_categories,
        "high_risk_summary": _calculate_high_risk_summary(risk_categories, total_assets)
    }

def _extract_risk_metrics(row: pd.Series, risk_type: str) -> Dict[str, Any]:
    """Extract risk level counts for a specific risk type."""
    risk_levels = ["VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    counts = {}
    
    for level in risk_levels:
        col_name = f"COUNT_OF_ASSETS_WITH_{level}_{risk_type}"
        counts[level.lower()] = row.get(col_name, 0)
    
    total = sum(counts.values())
    percentages = {level: (count / total * 100) if total > 0 else 0 
                  for level, count in counts.items()}
    
    return {
        "counts": counts,
        "percentages": percentages,
        "total_assets": total,
        "high_risk_assets": counts.get("high", 0) + counts.get("very_high", 0)
    }

def _calculate_high_risk_summary(risk_categories: Dict, total_assets: int) -> Dict[str, Any]:
    """Calculate summary of high-risk exposures."""
    high_risk_summary = {}
    
    for category, data in risk_categories.items():
        high_risk_count = data["high_risk_assets"]
        high_risk_pct = (high_risk_count / total_assets * 100) if total_assets > 0 else 0
        
        high_risk_summary[category] = {
            "high_risk_assets": high_risk_count,
            "high_risk_percentage": round(high_risk_pct, 2)
        }
    
    # Find top risk categories
    sorted_risks = sorted(high_risk_summary.items(), 
                         key=lambda x: x[1]["high_risk_percentage"], 
                         reverse=True)
    
    return {
        "by_category": dict(high_risk_summary),
        "top_risk_categories": [{"category": cat, **data} for cat, data in sorted_risks[:5]]
    }

@mcp.tool()
def GetGistRiskByCategory(risk_type: str, risk_level: str = "HIGH") -> Dict[str, Any]:
    """
    Get companies by specific risk category and level.
    
    Parameters:
    - risk_type: Type of risk (MSA, WATER_STRESS, DROUGHT, FLOOD_COASTAL, etc.)
    - risk_level: Risk level (VERY_LOW, LOW, MODERATE, HIGH, VERY_HIGH)
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    exsitu_df = data_manager.get_sheet('EXSITU')
    if exsitu_df.empty:
        return {"error": "EXSITU risk data not available"}
    
    col_name = f"COUNT_OF_ASSETS_WITH_{risk_level.upper()}_{risk_type.upper()}"
    
    if col_name not in exsitu_df.columns:
        available_risks = [col.split('_WITH_')[1].split('_')[1] for col in exsitu_df.columns 
                          if col.startswith('COUNT_OF_ASSETS_WITH_')]
        return {"error": f"Risk type {risk_type} not found. Available: {list(set(available_risks))}"}
    
    # Filter companies with assets in this risk category
    companies_at_risk = exsitu_df[exsitu_df[col_name] > 0].copy()
    companies_at_risk = companies_at_risk.sort_values(col_name, ascending=False)
    
    result = {
        "risk_type": risk_type,
        "risk_level": risk_level,
        "companies_found": len(companies_at_risk),
        "companies": []
    }
    
    for _, row in companies_at_risk.head(20).iterrows():  # Limit to top 20
        total_assets = row.get('TOTAL_NUMBER_OF_ASSETS_ASSESSED_WITHIN_A_COMPANY', 0)
        at_risk_assets = row[col_name]
        risk_percentage = (at_risk_assets / total_assets * 100) if total_assets > 0 else 0
        
        result["companies"].append({
            "company_code": row['COMPANY_CODE'],
            "company_name": row['COMPANY_NAME'],
            "sector_code": row['SECTOR_CODE'],
            "country": row['COUNTRY_NAME'],
            "total_assets": total_assets,
            "at_risk_assets": at_risk_assets,
            "risk_percentage": round(risk_percentage, 2)
        })
    
    return result

@mcp.tool()
def GetGistHighRiskCompanies(risk_threshold: float = 14.67, limit: int = 20) -> Dict[str, Any]:
    """
    Get companies with highest overall environmental risk exposure.
    
    Parameters:
    - risk_threshold: Minimum percentage of assets at high/very high risk
    - limit: Maximum number of companies to return
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    exsitu_df = data_manager.get_sheet('EXSITU')
    if exsitu_df.empty:
        return {"error": "EXSITU risk data not available"}
    
    high_risk_companies = []
    
    for _, row in exsitu_df.iterrows():
        total_assets = row.get('TOTAL_NUMBER_OF_ASSETS_ASSESSED_WITHIN_A_COMPANY', 0)
        if total_assets == 0:
            continue
        
        # Calculate percentage of assets at high/very high risk across all categories
        high_risk_counts = []
        for col in row.index:
            if 'COUNT_OF_ASSETS_WITH_HIGH_' in col or 'COUNT_OF_ASSETS_WITH_VERY_HIGH_' in col:
                high_risk_counts.append(row[col])
        
        # Average high risk percentage across all risk categories
        avg_high_risk_pct = (sum(high_risk_counts) / len(high_risk_counts) / total_assets * 100) if high_risk_counts else 0
        
        if avg_high_risk_pct >= risk_threshold:
            high_risk_companies.append({
                "company_code": row['COMPANY_CODE'],
                "company_name": row['COMPANY_NAME'],
                "sector_code": row['SECTOR_CODE'],
                "country": row['COUNTRY_NAME'],
                "total_assets": total_assets,
                "avg_high_risk_percentage": round(avg_high_risk_pct, 2)
            })
    
    # Sort by risk percentage
    high_risk_companies.sort(key=lambda x: x['avg_high_risk_percentage'], reverse=True)
    
    return {
        "risk_threshold": risk_threshold,
        "companies_found": len(high_risk_companies),
        "companies": high_risk_companies[:limit]
    }

# =============================================================================
# GEOGRAPHIC/ASSET ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
def GetGistAssetsMapData(company_code: Optional[str] = None, country: Optional[str] = None, limit: int = 1000) -> Dict[str, Any]:
    """
    Get asset coordinates for mapping visualization.
    
    Parameters:
    - company_code: Filter by specific company
    - country: Filter by country code
    - limit: Maximum number of assets to return
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    asset_df = data_manager.get_sheet('EXSITU_ASSET_DATA')
    if asset_df.empty:
        return {"error": "Asset data not available"}
    
    # Apply filters
    filtered_df = asset_df.copy()
    
    if company_code:
        filtered_df = filtered_df[filtered_df['COMPANY_CODE'] == company_code]
    
    if country:
        filtered_df = filtered_df[filtered_df['COUNTRY_CODE'] == country.upper()]
    
    # Limit results for performance
    filtered_df = filtered_df.head(limit)
    
    if filtered_df.empty:
        return {"error": "No assets found with specified filters"}
    
    # Convert to map data format
    assets_data = []
    for _, row in filtered_df.iterrows():
        assets_data.append({
            "asset_id": row['ASSET_ID'],
            "company_code": row['COMPANY_CODE'],
            "company_name": row['COMPANY_NAME'],
            "latitude": float(row['LATITUDE']),
            "longitude": float(row['LONGITUDE']),
            "country": row['COUNTRY_CODE'],
            "asset_type": row.get('ASSET_TYPE_LEVEL_1', 'Unknown'),
            "msa_risk": row.get('MSA_RISKLEVEL', 'Unknown'),
            "water_stress_risk": row.get('WATER_STRESS_RISKLEVEL', 'Unknown')
        })
    
    return {
        "type": "map",
        "filters_applied": {"company_code": company_code, "country": country, "limit": limit},
        "data": assets_data,
        "metadata": {
            "total_assets": len(assets_data),
            "countries": list(filtered_df['COUNTRY_CODE'].unique()),
            "companies": list(filtered_df['COMPANY_CODE'].unique()),
            "asset_types": list(filtered_df['ASSET_TYPE_LEVEL_1'].unique()) if 'ASSET_TYPE_LEVEL_1' in filtered_df.columns else []
        }
    }

@mcp.tool()
def GetGistAssetsInRadius(latitude: float, longitude: float, radius_km: float = 50.0, limit: int = 100) -> Dict[str, Any]:
    """
    Find assets within a radius of given coordinates.
    
    Parameters:
    - latitude: Center latitude
    - longitude: Center longitude
    - radius_km: Search radius in kilometers
    - limit: Maximum number of assets to return
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    asset_df = data_manager.get_sheet('EXSITU_ASSET_DATA')
    if asset_df.empty:
        return {"error": "Asset data not available"}
    
    # Calculate distances using Haversine formula
    lat1, lon1 = np.radians(latitude), np.radians(longitude)
    lat2, lon2 = np.radians(asset_df['LATITUDE']), np.radians(asset_df['LONGITUDE'])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))  # Earth radius = 6371 km
    
    # Filter by radius
    within_radius = asset_df[distance_km <= radius_km].copy()
    within_radius['distance_km'] = distance_km[distance_km <= radius_km]
    within_radius = within_radius.sort_values('distance_km').head(limit)
    
    if within_radius.empty:
        return {"error": f"No assets found within {radius_km}km of ({latitude}, {longitude})"}
    
    assets_data = []
    for _, row in within_radius.iterrows():
        assets_data.append({
            "asset_id": row['ASSET_ID'],
            "company_code": row['COMPANY_CODE'],
            "company_name": row['COMPANY_NAME'],
            "latitude": float(row['LATITUDE']),
            "longitude": float(row['LONGITUDE']),
            "distance_km": round(row['distance_km'], 2),
            "country": row['COUNTRY_CODE'],
            "asset_type": row.get('ASSET_TYPE_LEVEL_1', 'Unknown')
        })
    
    return {
        "search_center": [latitude, longitude],
        "radius_km": radius_km,
        "assets_found": len(assets_data),
        "assets": assets_data
    }

@mcp.tool()
def GetGistAssetsByCountry() -> Dict[str, Any]:
    """
    Get asset distribution by country.
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    asset_df = data_manager.get_sheet('EXSITU_ASSET_DATA')
    if asset_df.empty:
        return {"error": "Asset data not available"}
    
    country_stats = asset_df.groupby('COUNTRY_CODE').agg({
        'ASSET_ID': 'count',
        'COMPANY_CODE': 'nunique',
        'LATITUDE': ['min', 'max'],
        'LONGITUDE': ['min', 'max']
    }).round(2)
    
    country_stats.columns = ['total_assets', 'unique_companies', 'min_lat', 'max_lat', 'min_lon', 'max_lon']
    country_stats = country_stats.reset_index()
    
    return {
        "total_countries": len(country_stats),
        "countries": country_stats.to_dict('records')
    }

@mcp.tool()
def GetGistAssetDetails(asset_id: str) -> Dict[str, Any]:
    """
    Get detailed information for a specific asset.
    
    Parameters:
    - asset_id: Unique asset identifier
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    asset_df = data_manager.get_sheet('EXSITU_ASSET_DATA')
    if asset_df.empty:
        return {"error": "Asset data not available"}
    
    asset_data = asset_df[asset_df['ASSET_ID'] == asset_id]
    if asset_data.empty:
        return {"error": f"Asset {asset_id} not found"}
    
    asset = asset_data.iloc[0].to_dict()
    
    # Clean up the data for better presentation
    return {
        "asset_id": asset_id,
        "company_info": {
            "company_code": asset.get('COMPANY_CODE'),
            "company_name": asset.get('COMPANY_NAME')
        },
        "location": {
            "latitude": asset.get('LATITUDE'),
            "longitude": asset.get('LONGITUDE'),
            "country_code": asset.get('COUNTRY_CODE')
        },
        "asset_type": {
            "level_1": asset.get('ASSET_TYPE_LEVEL_1'),
            "level_2": asset.get('ASSET_TYPE_LEVEL_2')
        },
        "environmental_risks": {
            "msa": {"value": asset.get('MSA'), "risk_level": asset.get('MSA_RISKLEVEL')},
            "water_stress": {"value": asset.get('WATER_STRESS'), "risk_level": asset.get('WATER_STRESS_RISKLEVEL')},
            "water_variability": {"value": asset.get('WATER_VARIABILITY'), "risk_level": asset.get('WATER_VARIABILITY_RISKLEVEL')},
            "water_demand": {"value": asset.get('WATER_DEMAND'), "risk_level": asset.get('WATER_DEMAND_RISKLEVEL')},
            "drought": {"value": asset.get('DROUGHT'), "risk_level": asset.get('DROUGHT_RISKLEVEL')},
            "flood_riverine": {"value": asset.get('FLOOD_RIVERINE'), "risk_level": asset.get('FLOOD_RIVERINE_RISKLEVEL')},
            "flood_coastal": {"value": asset.get('FLOOD_COASTAL'), "risk_level": asset.get('FLOOD_COASTAL_RISKLEVEL')},
            "extreme_heat": {"value": asset.get('EXTREME_HEAT'), "risk_level": asset.get('EXTREME_HEAT_RISK')},
            "extreme_precipitation": {"value": asset.get('EXTREME_PRECIPITATION'), "risk_level": asset.get('EXTREME_PRECIPITATION_RISK')},
            "temperature_anomaly": {"value": asset.get('TEMPERATURE_ANOMALY'), "risk_level": asset.get('TEMPERATURE_ANOMALY_RISK')}
        }
    }

# =============================================================================
# EMISSIONS & CARBON ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
def GetGistScope3Emissions(company_code: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get Scope 3 emissions data for a company.
    
    Parameters:
    - company_code: Unique company identifier
    - year: Optional filter for specific reporting year
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    scope3_df = data_manager.get_sheet('SCOPE_3_DATA')
    if scope3_df.empty:
        return {"error": "Scope 3 data not available"}
    
    company_data = scope3_df[scope3_df['COMPANY_CODE'] == company_code]
    if company_data.empty:
        return {"error": f"No Scope 3 data found for company {company_code}"}
    
    if year:
        company_data = company_data[company_data['REPORTING_YEAR'] == year]
        if company_data.empty:
            return {"error": f"No Scope 3 data found for company {company_code} in year {year}"}
    
    # Sort by year
    company_data = company_data.sort_values('REPORTING_YEAR')
    
    emissions_data = {
        "company_code": company_code,
        "company_name": company_data.iloc[0]['COMPANY_NAME'],
        "sector_code": company_data.iloc[0]['SECTOR_CODE'],
        "years_available": sorted(company_data['REPORTING_YEAR'].unique().tolist()),
        "emissions_by_year": []
    }
    
    for _, row in company_data.iterrows():
        year_data = {
            "reporting_year": row['REPORTING_YEAR'],
            "revenue": row.get('REVENUE'),
            "total_scope3_emissions": row.get('SCOPE_3_EMISSIONS_TOTAL'),
            "upstream_emissions": row.get('SCOPE_3_EMISSIONS_TOTAL_UPSTREAM'),
            "downstream_emissions": row.get('SCOPE_3_EMISSIONS_TOTAL_DOWNSTREAM'),
            "breakdown": {
                "purchased_goods_services": row.get('SCOPE_3_PURCHASED_GOODS_AND_SERVICES'),
                "capital_goods": row.get('SCOPE_3_CAPITAL_GOODS'),
                "fuel_energy_activities": row.get('SCOPE_3_FUEL_AND_ENERGY_RELATED_ACTIVITIES_NOT_INCLUDED_IN_SCOPE_1_OR_SCOPE_2'),
                "upstream_transportation": row.get('SCOPE_3_UPSTREAM_TRANSPORTATION_AND_DISTRIBUTION'),
                "waste_operations": row.get('SCOPE_3_WASTE_GENERATED_IN_OPERATIONS'),
                "business_travel": row.get('SCOPE_3_BUSINESS_TRAVEL'),
                "employee_commuting": row.get('SCOPE_3_EMPLOYEE_COMMUTING'),
                "downstream_transportation": row.get('SCOPE_3_DOWNSTREAM_TRANSPORTATION_AND_DISTRIBUTION'),
                "use_of_sold_products": row.get('SCOPE_3_USE_OF_SOLD_PRODUCTS'),
                "end_of_life_treatment": row.get('SCOPE_3_END_OF_LIFE_TREATMENT_OF_SOLD_PRODUCTS'),
                "investments": row.get('SCOPE_3_INVESTMENTS')
            }
        }
        
        # Calculate emissions intensity
        if row.get('REVENUE') and row.get('SCOPE_3_EMISSIONS_TOTAL'):
            year_data["emissions_intensity"] = row['SCOPE_3_EMISSIONS_TOTAL'] / row['REVENUE']
        
        emissions_data["emissions_by_year"].append(year_data)
    
    return emissions_data

@mcp.tool()
def GetGistEmissionsTrends(company_code: str) -> Dict[str, Any]:
    """
    Get multi-year emissions trends for a company.
    
    Parameters:
    - company_code: Unique company identifier
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    scope3_df = data_manager.get_sheet('SCOPE_3_DATA')
    if scope3_df.empty:
        return {"error": "Scope 3 data not available"}
    
    company_data = scope3_df[scope3_df['COMPANY_CODE'] == company_code]
    if company_data.empty:
        return {"error": f"No emissions data found for company {company_code}"}
    
    # Sort by year for trend analysis
    company_data = company_data.sort_values('REPORTING_YEAR')
    
    if len(company_data) < 2:
        return {"error": f"Insufficient data for trend analysis (need at least 2 years)"}
    
    trends = {
        "company_code": company_code,
        "company_name": company_data.iloc[0]['COMPANY_NAME'],
        "analysis_period": {
            "start_year": int(company_data['REPORTING_YEAR'].min()),
            "end_year": int(company_data['REPORTING_YEAR'].max()),
            "years_analyzed": len(company_data)
        },
        "trends": {}
    }
    
    # Calculate trends for key metrics
    metrics = {
        "total_emissions": "SCOPE_3_EMISSIONS_TOTAL",
        "upstream_emissions": "SCOPE_3_EMISSIONS_TOTAL_UPSTREAM", 
        "downstream_emissions": "SCOPE_3_EMISSIONS_TOTAL_DOWNSTREAM",
        "revenue": "REVENUE"
    }
    
    for metric_name, column in metrics.items():
        if column in company_data.columns:
            values = company_data[column].dropna()
            if len(values) >= 2:
                first_value = values.iloc[0]
                last_value = values.iloc[-1]
                
                if first_value > 0:
                    percent_change = ((last_value - first_value) / first_value) * 100
                    trends["trends"][metric_name] = {
                        "start_value": first_value,
                        "end_value": last_value,
                        "absolute_change": last_value - first_value,
                        "percent_change": round(percent_change, 2),
                        "trend_direction": "increasing" if percent_change > 0 else "decreasing"
                    }
    
    # Calculate emissions intensity trend if both emissions and revenue available
    if "total_emissions" in trends["trends"] and "revenue" in trends["trends"]:
        intensity_data = []
        for _, row in company_data.iterrows():
            if pd.notna(row.get('SCOPE_3_EMISSIONS_TOTAL')) and pd.notna(row.get('REVENUE')) and row['REVENUE'] > 0:
                intensity_data.append(row['SCOPE_3_EMISSIONS_TOTAL'] / row['REVENUE'])
        
        if len(intensity_data) >= 2:
            first_intensity = intensity_data[0]
            last_intensity = intensity_data[-1]
            intensity_change = ((last_intensity - first_intensity) / first_intensity) * 100 if first_intensity > 0 else 0
            
            trends["trends"]["emissions_intensity"] = {
                "start_value": first_intensity,
                "end_value": last_intensity,
                "percent_change": round(intensity_change, 2),
                "trend_direction": "increasing" if intensity_change > 0 else "decreasing"
            }
    
    return trends

@mcp.tool()
def GetGistEmissionsBySector(year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get emissions comparison by sector.
    
    Parameters:
    - year: Optional filter for specific year (default: latest available year)
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    scope3_df = data_manager.get_sheet('SCOPE_3_DATA')
    if scope3_df.empty:
        return {"error": "Scope 3 data not available"}
    
    if year:
        data = scope3_df[scope3_df['REPORTING_YEAR'] == year]
        if data.empty:
            return {"error": f"No emissions data available for year {year}"}
    else:
        # Use most recent year available
        latest_year = scope3_df['REPORTING_YEAR'].max()
        data = scope3_df[scope3_df['REPORTING_YEAR'] == latest_year]
        year = latest_year
    
    sector_stats = data.groupby('SECTOR_CODE').agg({
        'SCOPE_3_EMISSIONS_TOTAL': ['sum', 'mean', 'median', 'count'],
        'SCOPE_3_EMISSIONS_TOTAL_UPSTREAM': 'sum',
        'SCOPE_3_EMISSIONS_TOTAL_DOWNSTREAM': 'sum',
        'REVENUE': 'sum',
        'COMPANY_CODE': 'nunique'
    }).round(2)
    
    sector_stats.columns = ['total_emissions', 'mean_emissions', 'median_emissions', 'company_count',
                           'total_upstream', 'total_downstream', 'total_revenue', 'unique_companies']
    sector_stats = sector_stats.reset_index()
    
    # Calculate emissions intensity by sector
    sector_stats['emissions_intensity'] = (sector_stats['total_emissions'] / 
                                         sector_stats['total_revenue']).round(4)
    
    # Sort by total emissions
    sector_stats = sector_stats.sort_values('total_emissions', ascending=False)
    
    return {
        "analysis_year": year,
        "total_sectors": len(sector_stats),
        "sector_emissions": sector_stats.to_dict('records')
    }

@mcp.tool()
def GetGistTopEmitters(limit: int = 20, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get highest emitting companies.
    
    Parameters:
    - limit: Maximum number of companies to return
    - year: Optional filter for specific year (default: latest available)
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    scope3_df = data_manager.get_sheet('SCOPE_3_DATA')
    if scope3_df.empty:
        return {"error": "Scope 3 data not available"}
    
    if year:
        data = scope3_df[scope3_df['REPORTING_YEAR'] == year]
        if data.empty:
            return {"error": f"No emissions data available for year {year}"}
    else:
        # Use most recent year for each company
        latest_year = scope3_df['REPORTING_YEAR'].max()
        data = scope3_df[scope3_df['REPORTING_YEAR'] == latest_year]
        year = latest_year
    
    # Filter out companies without emissions data
    data = data.dropna(subset=['SCOPE_3_EMISSIONS_TOTAL'])
    data = data[data['SCOPE_3_EMISSIONS_TOTAL'] > 0]
    
    # Sort by total emissions
    top_emitters = data.nlargest(limit, 'SCOPE_3_EMISSIONS_TOTAL')
    
    emitters_list = []
    for _, row in top_emitters.iterrows():
        emitter_data = {
            "rank": len(emitters_list) + 1,
            "company_code": row['COMPANY_CODE'],
            "company_name": row['COMPANY_NAME'],
            "sector_code": row['SECTOR_CODE'],
            "reporting_year": row['REPORTING_YEAR'],
            "total_scope3_emissions": row['SCOPE_3_EMISSIONS_TOTAL'],
            "upstream_emissions": row.get('SCOPE_3_EMISSIONS_TOTAL_UPSTREAM'),
            "downstream_emissions": row.get('SCOPE_3_EMISSIONS_TOTAL_DOWNSTREAM'),
            "revenue": row.get('REVENUE')
        }
        
        # Calculate emissions intensity if revenue available
        if row.get('REVENUE') and row['REVENUE'] > 0:
            emitter_data["emissions_intensity"] = row['SCOPE_3_EMISSIONS_TOTAL'] / row['REVENUE']
        
        emitters_list.append(emitter_data)
    
    return {
        "analysis_year": year,
        "companies_analyzed": len(data),
        "top_emitters": emitters_list
    }

# =============================================================================
# BIODIVERSITY IMPACT TOOLS
# =============================================================================

@mcp.tool()
def GetGistBiodiversityImpacts(company_code: str, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get biodiversity impact data for a company.
    
    Parameters:
    - company_code: Unique company identifier
    - year: Optional filter for specific reporting year
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    bio_df = data_manager.get_sheet('BIODIVERSITY_PDF_DATA')
    if bio_df.empty:
        return {"error": "Biodiversity data not available"}
    
    company_data = bio_df[bio_df['COMPANY_CODE'] == company_code]
    if company_data.empty:
        return {"error": f"No biodiversity data found for company {company_code}"}
    
    if year:
        company_data = company_data[company_data['REPORTING_YEAR'] == year]
        if company_data.empty:
            return {"error": f"No biodiversity data found for company {company_code} in year {year}"}
    
    # Sort by year
    company_data = company_data.sort_values('REPORTING_YEAR')
    
    biodiversity_data = {
        "company_code": company_code,
        "company_name": company_data.iloc[0]['COMPANY_NAME'],
        "sector_code": company_data.iloc[0]['SECTOR_CODE'],
        "years_available": sorted(company_data['REPORTING_YEAR'].unique().tolist()),
        "impacts_by_year": []
    }
    
    for _, row in company_data.iterrows():
        year_data = {
            "reporting_year": row['REPORTING_YEAR'],
            "total_impacts": {
                "pdf": row.get('TOTAL_COMPANY_IMPACTS_PDF'),
                "co2e": row.get('TOTAL_COMPANY_IMPACTS_CO2E'),
                "lce": row.get('TOTAL_COMPANY_IMPACTS_LCE')
            },
            "impact_categories_pdf": {
                "ghg_100_years": row.get('GHG_IMPACTS_PDF_100_YRS'),
                "ghg_1000_years": row.get('GHG_IMPACTS_PDF_1000_YRS'),
                "water_consumption": row.get('WATER_CONSUMPTION_IMPACTS_PDF'),
                "sox_impacts": row.get('SOX_IMPACTS_PDF'),
                "nox_impacts": row.get('NOX_IMPACTS_PDF'),
                "nitrogen_impacts": row.get('TOTAL_NITROGEN_IMPACTS_PDF'),
                "phosphorous_impacts": row.get('TOTAL_PHOSPHOROUS_IMPACTS_PDF'),
                "land_use_change": row.get('LUC_IMPACTS_PDF'),
                "waste_generation_100": row.get('WASTE_GENERATION_IMPACTS_PDF_100_YRS'),
                "waste_generation_1000": row.get('WASTE_GENERATION_IMPACTS_PDF_1000_YRS')
            },
            "impact_categories_co2e": {
                "ghg_100_years": row.get('GHG_IMPACTS_CO2E_100_YRS'),
                "ghg_1000_years": row.get('GHG_IMPACTS_CO2E_1000_YRS'),
                "water_consumption": row.get('WATER_CONSUMPTION_IMPACTS_CO2E'),
                "sox_impacts": row.get('SOX_IMPACTS_CO2E'),
                "nox_impacts": row.get('NOX_IMPACTS_CO2E'),
                "nitrogen_impacts": row.get('TOTAL_NITROGEN_IMPACTS_CO2E'),
                "phosphorous_impacts": row.get('TOTAL_PHOSPHOROUS_IMPACTS_CO2E'),
                "land_use_change": row.get('LUC_IMPACTS_CO2E'),
                "waste_generation_100": row.get('WASTE_GENERATION_IMPACTS_CO2E_100_YRS'),
                "waste_generation_1000": row.get('WASTE_GENERATION_IMPACTS_CO2E_1000_YRS')
            }
        }
        
        biodiversity_data["impacts_by_year"].append(year_data)
    
    return biodiversity_data

@mcp.tool()
def GetGistBiodiversityTrends(company_code: str) -> Dict[str, Any]:
    """
    Get multi-year biodiversity impact trends for a company.
    
    Parameters:
    - company_code: Unique company identifier
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    bio_df = data_manager.get_sheet('BIODIVERSITY_PDF_DATA')
    if bio_df.empty:
        return {"error": "Biodiversity data not available"}
    
    company_data = bio_df[bio_df['COMPANY_CODE'] == company_code]
    if company_data.empty:
        return {"error": f"No biodiversity data found for company {company_code}"}
    
    # Sort by year for trend analysis
    company_data = company_data.sort_values('REPORTING_YEAR')
    
    if len(company_data) < 2:
        return {"error": f"Insufficient data for trend analysis (need at least 2 years)"}
    
    trends = {
        "company_code": company_code,
        "company_name": company_data.iloc[0]['COMPANY_NAME'],
        "analysis_period": {
            "start_year": int(company_data['REPORTING_YEAR'].min()),
            "end_year": int(company_data['REPORTING_YEAR'].max()),
            "years_analyzed": len(company_data)
        },
        "trends": {}
    }
    
    # Calculate trends for key biodiversity metrics
    metrics = {
        "total_pdf_impact": "TOTAL_COMPANY_IMPACTS_PDF",
        "total_co2e_impact": "TOTAL_COMPANY_IMPACTS_CO2E",
        "total_lce_impact": "TOTAL_COMPANY_IMPACTS_LCE",
        "ghg_pdf_100": "GHG_IMPACTS_PDF_100_YRS",
        "water_consumption_pdf": "WATER_CONSUMPTION_IMPACTS_PDF",
        "land_use_change_pdf": "LUC_IMPACTS_PDF"
    }
    
    for metric_name, column in metrics.items():
        if column in company_data.columns:
            values = company_data[column].dropna()
            if len(values) >= 2:
                first_value = values.iloc[0]
                last_value = values.iloc[-1]
                
                if first_value != 0:
                    percent_change = ((last_value - first_value) / abs(first_value)) * 100
                    trends["trends"][metric_name] = {
                        "start_value": first_value,
                        "end_value": last_value,
                        "absolute_change": last_value - first_value,
                        "percent_change": round(percent_change, 2),
                        "trend_direction": "increasing" if percent_change > 0 else "decreasing"
                    }
    
    return trends

@mcp.tool()
def GetGistBiodiversityBySector(year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get biodiversity impact comparison by sector.
    
    Parameters:
    - year: Optional filter for specific year (default: latest available year)
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    bio_df = data_manager.get_sheet('BIODIVERSITY_PDF_DATA')
    if bio_df.empty:
        return {"error": "Biodiversity data not available"}
    
    if year:
        data = bio_df[bio_df['REPORTING_YEAR'] == year]
        if data.empty:
            return {"error": f"No biodiversity data available for year {year}"}
    else:
        # Use most recent year available
        latest_year = bio_df['REPORTING_YEAR'].max()
        data = bio_df[bio_df['REPORTING_YEAR'] == latest_year]
        year = latest_year
    
    sector_stats = data.groupby('SECTOR_CODE').agg({
        'TOTAL_COMPANY_IMPACTS_PDF': ['sum', 'mean', 'median'],
        'TOTAL_COMPANY_IMPACTS_CO2E': ['sum', 'mean', 'median'],
        'TOTAL_COMPANY_IMPACTS_LCE': ['sum', 'mean', 'median'],
        'GHG_IMPACTS_PDF_100_YRS': 'sum',
        'WATER_CONSUMPTION_IMPACTS_PDF': 'sum',
        'LUC_IMPACTS_PDF': 'sum',
        'COMPANY_CODE': 'nunique'
    }).round(6)
    
    sector_stats.columns = ['total_pdf_sum', 'mean_pdf', 'median_pdf',
                           'total_co2e_sum', 'mean_co2e', 'median_co2e',
                           'total_lce_sum', 'mean_lce', 'median_lce',
                           'total_ghg_pdf', 'total_water_pdf', 'total_luc_pdf',
                           'unique_companies']
    sector_stats = sector_stats.reset_index()
    
    # Sort by total PDF impact
    sector_stats = sector_stats.sort_values('total_pdf_sum', ascending=False)
    
    return {
        "analysis_year": year,
        "total_sectors": len(sector_stats),
        "sector_biodiversity_impacts": sector_stats.to_dict('records')
    }

@mcp.tool()
def GetGistBiodiversityWorstPerformers(metric: str = 'PDF', limit: int = 20, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get companies with highest biodiversity impacts.
    
    Parameters:
    - metric: Impact metric to rank by (PDF, CO2E, LCE)
    - limit: Maximum number of companies to return
    - year: Optional filter for specific year (default: latest available)
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    bio_df = data_manager.get_sheet('BIODIVERSITY_PDF_DATA')
    if bio_df.empty:
        return {"error": "Biodiversity data not available"}
    
    if year:
        data = bio_df[bio_df['REPORTING_YEAR'] == year]
        if data.empty:
            return {"error": f"No biodiversity data available for year {year}"}
    else:
        # Use most recent year for each company
        latest_year = bio_df['REPORTING_YEAR'].max()
        data = bio_df[bio_df['REPORTING_YEAR'] == latest_year]
        year = latest_year
    
    # Select the appropriate column based on metric
    metric_columns = {
        'PDF': 'TOTAL_COMPANY_IMPACTS_PDF',
        'CO2E': 'TOTAL_COMPANY_IMPACTS_CO2E', 
        'LCE': 'TOTAL_COMPANY_IMPACTS_LCE'
    }
    
    if metric.upper() not in metric_columns:
        return {"error": f"Invalid metric {metric}. Available: {list(metric_columns.keys())}"}
    
    column = metric_columns[metric.upper()]
    
    # Filter out companies without data
    data = data.dropna(subset=[column])
    
    # Sort by impact metric
    worst_performers = data.nlargest(limit, column)
    
    performers_list = []
    for _, row in worst_performers.iterrows():
        performer_data = {
            "rank": len(performers_list) + 1,
            "company_code": row['COMPANY_CODE'],
            "company_name": row['COMPANY_NAME'],
            "sector_code": row['SECTOR_CODE'],
            "reporting_year": row['REPORTING_YEAR'],
            "impact_value": row[column],
            "impact_metric": metric.upper(),
            "other_metrics": {
                "pdf_impact": row.get('TOTAL_COMPANY_IMPACTS_PDF'),
                "co2e_impact": row.get('TOTAL_COMPANY_IMPACTS_CO2E'),
                "lce_impact": row.get('TOTAL_COMPANY_IMPACTS_LCE')
            }
        }
        
        performers_list.append(performer_data)
    
    return {
        "analysis_year": year,
        "impact_metric": metric.upper(),
        "companies_analyzed": len(data),
        "worst_performers": performers_list
    }

# =============================================================================
# DEFORESTATION ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
def GetGistDeforestationRisks(company_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Get deforestation proximity indicators for companies.
    
    Parameters:
    - company_code: Optional filter for specific company
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    deforest_df = data_manager.get_sheet('DEFORESTATION')
    if deforest_df.empty:
        return {"error": "Deforestation data not available"}
    
    if company_code:
        company_data = deforest_df[deforest_df['COMPANY_CODE'] == company_code]
        if company_data.empty:
            return {"error": f"No deforestation data found for company {company_code}"}
        
        row = company_data.iloc[0]
        return {
            "company_code": company_code,
            "company_name": row.get('COMPANY_NAME', 'Unknown'),
            "deforestation_indicators": {
                "high_fraction_assets_forest_change": bool(row.get('company_high_fraction_assets_forest_change_proximity', False)),
                "high_average_forest_change": bool(row.get('company_high_average_forest_change_proximity', False)),
                "extreme_forest_change_proximity": bool(row.get('company_asset_extreme_forest_change_proximity', False))
            },
            "risk_level": _calculate_deforestation_risk_level(row)
        }
    else:
        # Return summary for all companies
        total_companies = len(deforest_df)
        high_fraction = deforest_df['company_high_fraction_assets_forest_change_proximity'].sum()
        high_average = deforest_df['company_high_average_forest_change_proximity'].sum()
        extreme_proximity = deforest_df['company_asset_extreme_forest_change_proximity'].sum()
        
        return {
            "total_companies_analyzed": total_companies,
            "summary": {
                "companies_with_high_fraction_risk": int(high_fraction),
                "companies_with_high_average_risk": int(high_average),
                "companies_with_extreme_proximity": int(extreme_proximity),
                "percentage_high_fraction": round((high_fraction / total_companies) * 100, 2) if total_companies > 0 else 0,
                "percentage_high_average": round((high_average / total_companies) * 100, 2) if total_companies > 0 else 0,
                "percentage_extreme_proximity": round((extreme_proximity / total_companies) * 100, 2) if total_companies > 0 else 0
            }
        }

def _calculate_deforestation_risk_level(row: pd.Series) -> str:
    """Calculate overall deforestation risk level for a company."""
    risk_indicators = [
        row.get('company_high_fraction_assets_forest_change_proximity', False),
        row.get('company_high_average_forest_change_proximity', False),
        row.get('company_asset_extreme_forest_change_proximity', False)
    ]
    
    risk_count = sum(risk_indicators)
    
    if risk_count >= 3:
        return "Very High"
    elif risk_count == 2:
        return "High"
    elif risk_count == 1:
        return "Moderate"
    else:
        return "Low"

@mcp.tool()
def GetGistDeforestationExposed() -> Dict[str, Any]:
    """
    Get companies with high deforestation exposure.
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    deforest_df = data_manager.get_sheet('DEFORESTATION')
    if deforest_df.empty:
        return {"error": "Deforestation data not available"}
    
    # Find companies with any deforestation risk indicators
    high_risk_companies = []
    
    for _, row in deforest_df.iterrows():
        risk_indicators = [
            row.get('company_high_fraction_assets_forest_change_proximity', False),
            row.get('company_high_average_forest_change_proximity', False),
            row.get('company_asset_extreme_forest_change_proximity', False)
        ]
        
        if any(risk_indicators):
            company_info = data_manager.companies_cache.get(row['COMPANY_CODE'], {})
            
            high_risk_companies.append({
                "company_code": row['COMPANY_CODE'],
                "company_name": row.get('COMPANY_NAME', 'Unknown'),
                "sector_code": company_info.get('sector_code', 'Unknown'),
                "country": company_info.get('country', 'Unknown'),
                "risk_indicators": {
                    "high_fraction_assets": bool(risk_indicators[0]),
                    "high_average_forest_change": bool(risk_indicators[1]),
                    "extreme_proximity": bool(risk_indicators[2])
                },
                "risk_level": _calculate_deforestation_risk_level(row),
                "total_risk_indicators": sum(risk_indicators)
            })
    
    # Sort by number of risk indicators
    high_risk_companies.sort(key=lambda x: x['total_risk_indicators'], reverse=True)
    
    return {
        "total_companies_analyzed": len(deforest_df),
        "companies_with_deforestation_risk": len(high_risk_companies),
        "high_risk_companies": high_risk_companies
    }

@mcp.tool()
def GetGistForestChangeProximity() -> Dict[str, Any]:
    """
    Analysis of forest change proximity across companies.
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    deforest_df = data_manager.get_sheet('DEFORESTATION')
    if deforest_df.empty:
        return {"error": "Deforestation data not available"}
    
    # Analyze by sector if possible
    sector_analysis = {}
    
    for _, row in deforest_df.iterrows():
        company_code = row['COMPANY_CODE']
        company_info = data_manager.companies_cache.get(company_code, {})
        sector = company_info.get('sector_code', 'Unknown')
        
        if sector not in sector_analysis:
            sector_analysis[sector] = {
                "total_companies": 0,
                "high_fraction_risk": 0,
                "high_average_risk": 0,
                "extreme_proximity": 0
            }
        
        sector_analysis[sector]["total_companies"] += 1
        
        if row.get('company_high_fraction_assets_forest_change_proximity', False):
            sector_analysis[sector]["high_fraction_risk"] += 1
        if row.get('company_high_average_forest_change_proximity', False):
            sector_analysis[sector]["high_average_risk"] += 1
        if row.get('company_asset_extreme_forest_change_proximity', False):
            sector_analysis[sector]["extreme_proximity"] += 1
    
    # Calculate percentages
    for sector_data in sector_analysis.values():
        total = sector_data["total_companies"]
        if total > 0:
            sector_data["high_fraction_percentage"] = round((sector_data["high_fraction_risk"] / total) * 100, 2)
            sector_data["high_average_percentage"] = round((sector_data["high_average_risk"] / total) * 100, 2)
            sector_data["extreme_proximity_percentage"] = round((sector_data["extreme_proximity"] / total) * 100, 2)
    
    return {
        "total_companies_analyzed": len(deforest_df),
        "sector_analysis": sector_analysis
    }

# =============================================================================
# VISUALIZATION DATA TOOLS
# =============================================================================

@mcp.tool()
def GetGistVisualizationData(viz_type: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get structured data for specific visualization types.
    
    Parameters:
    - viz_type: Type of visualization ('emissions_by_sector', 'risk_distribution', 'asset_map', 'biodiversity_trends', 'scope3_breakdown')
    - filters: Optional filters like {'year': 2024, 'sector': 'OGES', 'company_code': 'COMPANY123'}
    
    Note: 'biodiversity_trends' and 'scope3_breakdown' require a 'company_code' in filters.
    If not provided, the function will return available companies to choose from.
    """
    if not data_manager.sheets:
        return {"error": "GIST data not available"}
    
    filters = filters or {}
    
    if viz_type == "emissions_by_sector":
        return _get_emissions_by_sector_viz(filters)
    elif viz_type == "risk_distribution":
        return _get_risk_distribution_viz(filters)
    elif viz_type == "asset_map":
        return _get_asset_map_viz(filters)
    elif viz_type == "biodiversity_trends":
        return _get_biodiversity_trends_viz(filters)
    elif viz_type == "scope3_breakdown":
        return _get_scope3_breakdown_viz(filters)
    else:
        return {"error": f"Unknown visualization type: {viz_type}. Available: emissions_by_sector, risk_distribution, asset_map, biodiversity_trends, scope3_breakdown"}

def _get_emissions_by_sector_viz(filters: Dict) -> Dict[str, Any]:
    """Get data for emissions by sector visualization."""
    scope3_df = data_manager.get_sheet('SCOPE_3_DATA')
    if scope3_df.empty:
        return {"error": "Scope 3 data not available"}
    
    year = filters.get('year', scope3_df['REPORTING_YEAR'].max())
    data = scope3_df[scope3_df['REPORTING_YEAR'] == year]
    
    sector_stats = data.groupby('SECTOR_CODE').agg({
        'SCOPE_3_EMISSIONS_TOTAL': 'sum',
        'COMPANY_CODE': 'nunique'
    }).round(2).reset_index()
    
    return {
        "visualization_type": "emissions_by_sector",
        "data": sector_stats.to_dict('records'),
        "chart_config": {
            "x_axis": "SECTOR_CODE",
            "y_axis": "SCOPE_3_EMISSIONS_TOTAL",
            "title": f"Scope 3 Emissions by Sector ({year})",
            "chart_type": "bar"
        },
        "metadata": {"year": year, "total_sectors": len(sector_stats)}
    }

def _get_risk_distribution_viz(filters: Dict) -> Dict[str, Any]:
    """Get data for environmental risk distribution visualization."""
    exsitu_df = data_manager.get_sheet('EXSITU')
    if exsitu_df.empty:
        return {"error": "EXSITU risk data not available"}
    
    # Count companies by risk level for different risk types
    risk_types = ['MSA', 'WATER_STRESS', 'DROUGHT', 'FLOOD_COASTAL', 'EXTREME_HEAT']
    risk_data = []
    
    for risk_type in risk_types:
        high_col = f"COUNT_OF_ASSETS_WITH_HIGH_{risk_type}"
        very_high_col = f"COUNT_OF_ASSETS_WITH_VERY_HIGH_{risk_type}"
        
        if high_col in exsitu_df.columns and very_high_col in exsitu_df.columns:
            companies_at_risk = len(exsitu_df[(exsitu_df[high_col] > 0) | (exsitu_df[very_high_col] > 0)])
            risk_data.append({
                "risk_type": risk_type,
                "companies_at_high_risk": companies_at_risk,
                "total_companies": len(exsitu_df)
            })
    
    return {
        "visualization_type": "risk_distribution",
        "data": risk_data,
        "chart_config": {
            "x_axis": "risk_type",
            "y_axis": "companies_at_high_risk",
            "title": "Companies at High Environmental Risk by Category",
            "chart_type": "bar"
        }
    }

def _get_asset_map_viz(filters: Dict) -> Dict[str, Any]:
    """Get data for asset mapping visualization."""
    asset_df = data_manager.get_sheet('EXSITU_ASSET_DATA')
    if asset_df.empty:
        return {"error": "Asset data not available"}
    
    # Apply filters
    filtered_df = asset_df.copy()
    if 'country' in filters:
        filtered_df = filtered_df[filtered_df['COUNTRY_CODE'] == filters['country']]
    if 'company' in filters:
        filtered_df = filtered_df[filtered_df['COMPANY_CODE'] == filters['company']]
    
    # Limit for performance
    limit = filters.get('limit', 1000)
    filtered_df = filtered_df.head(limit)
    
    # Convert to GeoJSON-like structure
    features = []
    for _, row in filtered_df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['LONGITUDE']), float(row['LATITUDE'])]
            },
            "properties": {
                "asset_id": row['ASSET_ID'],
                "company_code": row['COMPANY_CODE'],
                "company_name": row['COMPANY_NAME'],
                "country": row['COUNTRY_CODE'],
                "msa_risk": row.get('MSA_RISKLEVEL', 'Unknown'),
                "water_stress_risk": row.get('WATER_STRESS_RISKLEVEL', 'Unknown')
            }
        })
    
    return {
        "visualization_type": "asset_map",
        "data": {
            "type": "FeatureCollection",
            "features": features
        },
        "metadata": {
            "total_assets": len(features),
            "filters_applied": filters
        }
    }

def _get_biodiversity_trends_viz(filters: Dict) -> Dict[str, Any]:
    """Get data for biodiversity trends visualization."""
    bio_df = data_manager.get_sheet('BIODIVERSITY_PDF_DATA')
    if bio_df.empty:
        return {"error": "Biodiversity data not available"}
    
    company_code = filters.get('company_code')
    if not company_code:
        # Provide helpful error with available companies
        available_companies = bio_df.groupby('COMPANY_CODE').agg({
            'COMPANY_NAME': 'first',
            'REPORTING_YEAR': ['min', 'max', 'count']
        }).reset_index()
        available_companies.columns = ['COMPANY_CODE', 'COMPANY_NAME', 'MIN_YEAR', 'MAX_YEAR', 'YEARS_COUNT']
        
        # Sort by number of years (most complete data first)
        available_companies = available_companies.sort_values('YEARS_COUNT', ascending=False)
        
        return {
            "error": "company_code required for biodiversity trends",
            "help": "Please specify a company_code filter to view biodiversity trends for a specific company",
            "available_companies": available_companies.head(20).to_dict('records'),
            "total_companies_with_data": len(available_companies),
            "suggested_company": available_companies.iloc[0]['COMPANY_CODE'] if len(available_companies) > 0 else None
        }
    
    company_data = bio_df[bio_df['COMPANY_CODE'] == company_code].sort_values('REPORTING_YEAR')
    if company_data.empty:
        # Provide suggestions for similar company codes if exact match not found
        similar_companies = bio_df[bio_df['COMPANY_CODE'].str.contains(company_code, case=False, na=False)]['COMPANY_CODE'].unique()
        error_msg = f"No biodiversity data for company {company_code}"
        if len(similar_companies) > 0:
            error_msg += f". Did you mean one of: {list(similar_companies)[:5]}"
        return {"error": error_msg}
    
    trend_data = []
    for _, row in company_data.iterrows():
        trend_data.append({
            "year": row['REPORTING_YEAR'],
            "total_pdf_impact": row.get('TOTAL_COMPANY_IMPACTS_PDF', 0),
            "total_co2e_impact": row.get('TOTAL_COMPANY_IMPACTS_CO2E', 0),
            "ghg_impact": row.get('GHG_IMPACTS_PDF_100_YRS', 0),
            "water_impact": row.get('WATER_CONSUMPTION_IMPACTS_PDF', 0)
        })
    
    return {
        "visualization_type": "biodiversity_trends",
        "data": trend_data,
        "chart_config": {
            "x_axis": "year",
            "y_axis": "total_pdf_impact",
            "title": f"Biodiversity Impact Trends - {company_code}",
            "chart_type": "line"
        },
        "metadata": {"company_code": company_code, "years_analyzed": len(trend_data)}
    }

def _get_scope3_breakdown_viz(filters: Dict) -> Dict[str, Any]:
    """Get data for Scope 3 emissions breakdown visualization."""
    scope3_df = data_manager.get_sheet('SCOPE_3_DATA')
    if scope3_df.empty:
        return {"error": "Scope 3 data not available"}
    
    company_code = filters.get('company_code')
    year = filters.get('year')
    
    if not company_code:
        # Provide helpful error with available companies
        available_companies = scope3_df.groupby('COMPANY_CODE').agg({
            'COMPANY_NAME': 'first',
            'REPORTING_YEAR': ['min', 'max', 'count'],
            'SCOPE_3_EMISSIONS_TOTAL': 'sum'
        }).reset_index()
        available_companies.columns = ['COMPANY_CODE', 'COMPANY_NAME', 'MIN_YEAR', 'MAX_YEAR', 'YEARS_COUNT', 'TOTAL_EMISSIONS']
        
        # Sort by total emissions (highest emitters first, as they're likely most relevant)
        available_companies = available_companies.sort_values('TOTAL_EMISSIONS', ascending=False)
        
        return {
            "error": "company_code required for scope3 breakdown",
            "help": "Please specify a company_code filter to view Scope 3 emissions breakdown for a specific company",
            "available_companies": available_companies.head(20).to_dict('records'),
            "total_companies_with_data": len(available_companies),
            "suggested_company": available_companies.iloc[0]['COMPANY_CODE'] if len(available_companies) > 0 else None
        }
    
    company_data = scope3_df[scope3_df['COMPANY_CODE'] == company_code]
    if company_data.empty:
        # Provide suggestions for similar company codes if exact match not found
        similar_companies = scope3_df[scope3_df['COMPANY_CODE'].str.contains(company_code, case=False, na=False)]['COMPANY_CODE'].unique()
        error_msg = f"No Scope 3 data for company {company_code}"
        if len(similar_companies) > 0:
            error_msg += f". Did you mean one of: {list(similar_companies)[:5]}"
        return {"error": error_msg}
    
    if year:
        company_data = company_data[company_data['REPORTING_YEAR'] == year]
        if company_data.empty:
            available_years = scope3_df[scope3_df['COMPANY_CODE'] == company_code]['REPORTING_YEAR'].unique()
            return {"error": f"No Scope 3 data for company {company_code} in year {year}. Available years: {sorted(available_years)}"}
    else:
        # Use latest year
        latest_year = company_data['REPORTING_YEAR'].max()
        company_data = company_data[company_data['REPORTING_YEAR'] == latest_year]
        year = latest_year
    
    row = company_data.iloc[0]
    
    # Extract Scope 3 categories
    breakdown_data = [
        {"category": "Purchased Goods & Services", "emissions": row.get('SCOPE_3_PURCHASED_GOODS_AND_SERVICES', 0)},
        {"category": "Capital Goods", "emissions": row.get('SCOPE_3_CAPITAL_GOODS', 0)},
        {"category": "Fuel & Energy Activities", "emissions": row.get('SCOPE_3_FUEL_AND_ENERGY_RELATED_ACTIVITIES_NOT_INCLUDED_IN_SCOPE_1_OR_SCOPE_2', 0)},
        {"category": "Upstream Transportation", "emissions": row.get('SCOPE_3_UPSTREAM_TRANSPORTATION_AND_DISTRIBUTION', 0)},
        {"category": "Business Travel", "emissions": row.get('SCOPE_3_BUSINESS_TRAVEL', 0)},
        {"category": "Employee Commuting", "emissions": row.get('SCOPE_3_EMPLOYEE_COMMUTING', 0)},
        {"category": "Use of Sold Products", "emissions": row.get('SCOPE_3_USE_OF_SOLD_PRODUCTS', 0)},
        {"category": "Investments", "emissions": row.get('SCOPE_3_INVESTMENTS', 0)}
    ]
    
    # Filter out zero emissions and sort
    breakdown_data = [item for item in breakdown_data if item['emissions'] > 0]
    breakdown_data.sort(key=lambda x: x['emissions'], reverse=True)
    
    return {
        "visualization_type": "scope3_breakdown",
        "data": breakdown_data,
        "chart_config": {
            "x_axis": "category",
            "y_axis": "emissions",
            "title": f"Scope 3 Emissions Breakdown - {company_code} ({year})",
            "chart_type": "pie"
        },
        "metadata": {
            "company_code": company_code,
            "year": year,
            "total_categories": len(breakdown_data)
        }
    }

@mcp.tool()
def GetGistDatasetMetadata() -> Dict[str, Any]:
    """
    Get comprehensive metadata about the GIST dataset.
    """
    return metadata

if __name__ == "__main__":
    mcp.run()