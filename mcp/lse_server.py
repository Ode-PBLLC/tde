import pandas as pd
import numpy as np
from fastmcp import FastMCP
from typing import List, Optional, Dict, Any, Union
import json
import os
from functools import lru_cache

mcp = FastMCP("lse-server")

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
LSE_DATA_PATH = os.path.join(project_root, "data", "lse")

class LSEDataManager:
    """
    Manages loading and caching of LSE (London School of Economics) climate policy data
    across all Excel files and sheets. Provides efficient access to NDC overview, 
    institutions, plans, policies, and subnational governance data.
    """
    
    def __init__(self):
        self.excel_files = {}
        self.all_sheets = {}
        self.content_cache = {}
        self.states_cache = {}
        self._load_data()
    
    def _load_data(self):
        """Load all LSE Excel files and their sheets."""
        try:
            print(f"Loading LSE data from: {LSE_DATA_PATH}")
            
            if not os.path.exists(LSE_DATA_PATH):
                print(f"LSE data directory not found: {LSE_DATA_PATH}")
                return
            
            # Find all Excel files
            excel_files = [f for f in os.listdir(LSE_DATA_PATH) if f.endswith('.xlsx')]
            print(f"Found {len(excel_files)} Excel files")
            
            for filename in excel_files:
                file_path = os.path.join(LSE_DATA_PATH, filename)
                print(f"Loading {filename}...")
                
                try:
                    excel_file = pd.ExcelFile(file_path)
                    self.excel_files[filename] = excel_file
                    
                    # Load each sheet
                    for sheet_name in excel_file.sheet_names:
                        try:
                            df = pd.read_excel(file_path, sheet_name=sheet_name)
                            sheet_key = f"{filename}::{sheet_name}"
                            self.all_sheets[sheet_key] = df
                            print(f"  ✓ {sheet_name}: {df.shape}")
                        except Exception as e:
                            print(f"  ✗ Error loading sheet {sheet_name}: {e}")
                
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            
            # Build content indexes
            self._build_content_index()
            self._build_states_index()
            
            print(f"LSE data loading completed: {len(self.all_sheets)} sheets loaded")
            
        except Exception as e:
            print(f"Error loading LSE data: {e}")
            import traceback
            traceback.print_exc()
    
    def _build_content_index(self):
        """Build index of content by module type."""
        self.content_cache = {
            'ndc_overview': [],
            'institutions': [],
            'plans_policies': [],
            'subnational': [],
            'tpi_graphs': []
        }
        
        for sheet_key, df in self.all_sheets.items():
            filename, sheet_name = sheet_key.split('::', 1)
            
            # Categorize by filename
            if 'NDC Overview' in filename:
                self.content_cache['ndc_overview'].append((filename, sheet_name, df))
            elif 'Institutions' in filename:
                self.content_cache['institutions'].append((filename, sheet_name, df))
            elif 'Plans and Policies' in filename:
                self.content_cache['plans_policies'].append((filename, sheet_name, df))
            elif 'Subnational' in filename:
                self.content_cache['subnational'].append((filename, sheet_name, df))
            elif 'TPI Graph' in filename:
                self.content_cache['tpi_graphs'].append((filename, sheet_name, df))
    
    def _build_states_index(self):
        """Build index of Brazilian states from subnational data."""
        self.states_cache = {}
        
        for filename, sheet_name, df in self.content_cache['subnational']:
            # Skip metadata sheets
            if sheet_name in ['How to use', 'Metadata']:
                continue
            
            # Extract state name (remove state code if present)
            state_name = sheet_name
            if '(' in state_name and ')' in state_name:
                state_name = sheet_name.split('(')[0].strip()
            
            self.states_cache[sheet_name] = {
                'state_name': state_name,
                'sheet_name': sheet_name,
                'filename': filename,
                'data': df,
                'questions_count': len(df) if not df.empty else 0
            }
    
    def get_sheet(self, filename: str, sheet_name: str) -> pd.DataFrame:
        """Get a specific sheet by filename and sheet name."""
        sheet_key = f"{filename}::{sheet_name}"
        return self.all_sheets.get(sheet_key, pd.DataFrame())
    
    def get_files_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded files and their contents."""
        summary = {}
        for filename, excel_file in self.excel_files.items():
            summary[filename] = {
                'sheet_count': len(excel_file.sheet_names),
                'sheet_names': excel_file.sheet_names
            }
        return summary
    
    def search_content(self, search_term: str, module_type: Optional[str] = None) -> List[Dict]:
        """Search across all content for a specific term."""
        results = []
        search_term_lower = search_term.lower()
        
        # Determine which modules to search
        modules_to_search = [module_type] if module_type else self.content_cache.keys()
        
        for module in modules_to_search:
            if module not in self.content_cache:
                continue
                
            for filename, sheet_name, df in self.content_cache[module]:
                if df.empty:
                    continue
                
                # Search in all text columns
                for col in df.columns:
                    if df[col].dtype == 'object':  # Text columns
                        matches = df[df[col].astype(str).str.lower().str.contains(search_term_lower, na=False)]
                        
                        for idx, row in matches.iterrows():
                            results.append({
                                'module': module,
                                'filename': filename,
                                'sheet_name': sheet_name,
                                'row_index': idx,
                                'column': col,
                                'content': str(row[col])[:200] + "..." if len(str(row[col])) > 200 else str(row[col]),
                                'context': {k: str(v)[:100] for k, v in row.to_dict().items() if k != col}
                            })
        
        return results[:20]  # Limit results

# Initialize the data manager
data_manager = LSEDataManager()

metadata = {
    "Name": "LSE Climate Policy Server",
    "Description": "London School of Economics climate policy governance data server",
    "Version": "1.0.0", 
    "Author": "Climate Policy Radar Team",
    "Dataset": "LSE Climate Policy Analysis Collection",
    "Total_Files": len(data_manager.excel_files),
    "Total_Sheets": len(data_manager.all_sheets),
    "Modules": list(data_manager.content_cache.keys()),
    "Brazilian_States": len(data_manager.states_cache)
}

# =============================================================================
# SCHEMA/DISCOVERY TOOLS
# =============================================================================

@mcp.tool()
def GetLSEDatasetOverview() -> Dict[str, Any]:
    """
    Get overview of all LSE climate policy datasets and their structure.
    """
    if not data_manager.all_sheets:
        return {"error": "LSE data not available"}
    
    overview = {
        "total_files": len(data_manager.excel_files),
        "total_sheets": len(data_manager.all_sheets),
        "modules": {},
        "files_summary": data_manager.get_files_summary()
    }
    
    # Add module summaries
    for module, content_list in data_manager.content_cache.items():
        overview["modules"][module] = {
            "files_count": len(set([filename for filename, _, _ in content_list])),
            "sheets_count": len(content_list),
            "description": _get_module_description(module)
        }
    
    return overview

def _get_module_description(module: str) -> str:
    """Get description for each module."""
    descriptions = {
        'ndc_overview': 'National Determined Contributions overview and domestic policy comparison',
        'institutions': 'Climate governance institutions and processes analysis',
        'plans_policies': 'Climate plans and policies assessment',
        'subnational': 'Brazilian state-level climate governance analysis',
        'tpi_graphs': 'Transition Pathway Initiative graphical data'
    }
    return descriptions.get(module, 'Climate policy analysis module')

@mcp.tool()
def GetLSEFileStructure(filename: str) -> Dict[str, Any]:
    """
    Get detailed structure of a specific LSE file.
    
    Parameters:
    - filename: Name of the Excel file to examine
    """
    if not data_manager.excel_files:
        return {"error": "LSE data not available"}
    
    if filename not in data_manager.excel_files:
        available_files = list(data_manager.excel_files.keys())
        return {"error": f"File '{filename}' not found. Available files: {available_files}"}
    
    excel_file = data_manager.excel_files[filename]
    structure = {
        "filename": filename,
        "total_sheets": len(excel_file.sheet_names),
        "sheets": {}
    }
    
    for sheet_name in excel_file.sheet_names:
        df = data_manager.get_sheet(filename, sheet_name)
        structure["sheets"][sheet_name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "has_data": not df.empty,
            "sample_data": df.head(2).to_dict('records') if not df.empty else []
        }
    
    return structure

@mcp.tool()
def SearchLSEContent(search_term: str, module_type: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """
    Search across LSE content for specific terms.
    
    Parameters:
    - search_term: Term to search for in content
    - module_type: Optional filter by module (ndc_overview, institutions, plans_policies, subnational, tpi_graphs)
    - limit: Maximum number of results to return
    """
    if not data_manager.all_sheets:
        return {"error": "LSE data not available"}
    
    results = data_manager.search_content(search_term, module_type)
    
    return {
        "search_term": search_term,
        "module_filter": module_type,
        "results_found": len(results),
        "results": results[:limit]
    }

# =============================================================================
# SUBNATIONAL GOVERNANCE TOOLS
# =============================================================================

@mcp.tool()
def GetBrazilianStatesOverview() -> Dict[str, Any]:
    """
    Get overview of Brazilian state climate governance data.
    """
    if not data_manager.states_cache:
        return {"error": "Brazilian states data not available"}
    
    overview = {
        "total_states": len(data_manager.states_cache),
        "states": {}
    }
    
    for sheet_key, state_info in data_manager.states_cache.items():
        overview["states"][state_info['state_name']] = {
            "sheet_name": sheet_key,
            "questions_analyzed": state_info['questions_count'],
            "has_data": state_info['questions_count'] > 0
        }
    
    return overview

@mcp.tool()
def GetStateClimatePolicy(state_name: str) -> Dict[str, Any]:
    """
    Get climate policy information for a specific Brazilian state.
    
    Parameters:
    - state_name: Name of the Brazilian state (e.g., "São Paulo", "Rio de Janeiro")
    """
    if not data_manager.states_cache:
        return {"error": "Brazilian states data not available"}
    
    # Find matching state (flexible matching)
    matching_state = None
    for sheet_key, state_info in data_manager.states_cache.items():
        if (state_name.lower() in state_info['state_name'].lower() or 
            state_name.lower() in sheet_key.lower()):
            matching_state = state_info
            break
    
    if not matching_state:
        available_states = [info['state_name'] for info in data_manager.states_cache.values()]
        return {"error": f"State '{state_name}' not found. Available states: {available_states[:10]}..."}
    
    df = matching_state['data']
    if df.empty:
        return {"error": f"No data available for {state_name}"}
    
    # Extract key policy information
    policy_info = {
        "state_name": matching_state['state_name'],
        "sheet_name": matching_state['sheet_name'],
        "total_questions": len(df),
        "policy_areas": []
    }
    
    # Process each row as a policy question/area
    for idx, row in df.iterrows():
        # Look for key columns that contain policy information
        question_col = None
        answer_col = None
        summary_col = None
        status_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'direction setting' in col_lower or 'question' in col_lower:
                question_col = col
            elif 'yes' in col_lower and 'no' in col_lower:
                answer_col = col
            elif 'summary' in col_lower:
                summary_col = col
            elif 'status' in col_lower:
                status_col = col
        
        if question_col and not pd.isna(row[question_col]):
            policy_area = {
                "question": str(row[question_col]),
                "answer": str(row[answer_col]) if answer_col and not pd.isna(row[answer_col]) else None,
                "summary": str(row[summary_col]) if summary_col and not pd.isna(row[summary_col]) else None,
                "status": str(row[status_col]) if status_col and not pd.isna(row[status_col]) else None
            }
            policy_info["policy_areas"].append(policy_area)
    
    return policy_info

@mcp.tool()
def CompareBrazilianStates(states: List[str], policy_area: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare climate policies across multiple Brazilian states.
    
    Parameters:
    - states: List of state names to compare
    - policy_area: Optional filter for specific policy area
    """
    if not data_manager.states_cache:
        return {"error": "Brazilian states data not available"}
    
    comparison = {
        "states_compared": [],
        "comparison_matrix": {},
        "summary_stats": {}
    }
    
    valid_states = []
    for state_name in states:
        # Find matching state
        for sheet_key, state_info in data_manager.states_cache.items():
            if (state_name.lower() in state_info['state_name'].lower() or 
                state_name.lower() in sheet_key.lower()):
                valid_states.append((state_name, state_info))
                break
    
    if not valid_states:
        return {"error": f"No valid states found from: {states}"}
    
    comparison["states_compared"] = [state[0] for state in valid_states]
    
    # Compare policy responses
    for state_name, state_info in valid_states:
        df = state_info['data']
        state_policies = []
        
        for idx, row in df.iterrows():
            # Extract Yes/No responses
            for col in df.columns:
                if 'yes' in col.lower() and 'no' in col.lower():
                    answer = str(row[col]) if not pd.isna(row[col]) else 'Unknown'
                    state_policies.append(answer)
                    break
        
        # Count policy responses
        yes_count = sum(1 for answer in state_policies if 'yes' in answer.lower())
        no_count = sum(1 for answer in state_policies if 'no' in answer.lower())
        
        comparison["comparison_matrix"][state_name] = {
            "total_policies": len(state_policies),
            "yes_responses": yes_count,
            "no_responses": no_count,
            "response_rate": round((yes_count / len(state_policies)) * 100, 2) if state_policies else 0
        }
    
    # Calculate summary statistics
    response_rates = [data["response_rate"] for data in comparison["comparison_matrix"].values()]
    comparison["summary_stats"] = {
        "avg_response_rate": round(np.mean(response_rates), 2) if response_rates else 0,
        "highest_response_rate": max(response_rates) if response_rates else 0,
        "lowest_response_rate": min(response_rates) if response_rates else 0
    }
    
    return comparison

# =============================================================================
# POLICY ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
def GetNDCOverviewData(country: Optional[str] = None) -> Dict[str, Any]:
    """
    Get NDC (Nationally Determined Contributions) overview data.
    
    Parameters:
    - country: Optional filter for specific country
    """
    if 'ndc_overview' not in data_manager.content_cache:
        return {"error": "NDC overview data not available"}
    
    ndc_data = {
        "available_files": [],
        "content_summary": {}
    }
    
    for filename, sheet_name, df in data_manager.content_cache['ndc_overview']:
        ndc_data["available_files"].append({
            "filename": filename,
            "sheet_name": sheet_name,
            "rows": len(df),
            "columns": len(df.columns)
        })
        
        if not df.empty:
            # Extract key information from the sheet
            key_info = []
            for idx, row in df.head(5).iterrows():  # Sample first 5 rows
                row_info = {}
                for col in df.columns[:3]:  # First 3 columns
                    if not pd.isna(row[col]):
                        row_info[col] = str(row[col])[:100]
                key_info.append(row_info)
            
            ndc_data["content_summary"][sheet_name] = key_info
    
    return ndc_data

@mcp.tool()
def GetInstitutionsProcessesData() -> Dict[str, Any]:
    """
    Get institutions and processes module data.
    """
    if 'institutions' not in data_manager.content_cache:
        return {"error": "Institutions and processes data not available"}
    
    institutions_data = {
        "available_datasets": [],
        "analysis_summary": {}
    }
    
    for filename, sheet_name, df in data_manager.content_cache['institutions']:
        institutions_data["available_datasets"].append({
            "filename": filename,
            "sheet_name": sheet_name,
            "dimensions": f"{len(df)} rows x {len(df.columns)} columns",
            "has_data": not df.empty
        })
        
        if not df.empty:
            # Analyze institutional data
            institutions_data["analysis_summary"][sheet_name] = {
                "total_entries": len(df),
                "column_structure": list(df.columns),
                "sample_content": df.head(2).to_dict('records') if len(df) > 0 else []
            }
    
    return institutions_data

@mcp.tool()
def GetPlansAndPoliciesData() -> Dict[str, Any]:
    """
    Get plans and policies module data.
    """
    if 'plans_policies' not in data_manager.content_cache:
        return {"error": "Plans and policies data not available"}
    
    policies_data = {
        "available_datasets": [],
        "content_analysis": {}
    }
    
    for filename, sheet_name, df in data_manager.content_cache['plans_policies']:
        policies_data["available_datasets"].append({
            "filename": filename,
            "sheet_name": sheet_name,
            "data_points": len(df),
            "variables": len(df.columns)
        })
        
        if not df.empty:
            policies_data["content_analysis"][sheet_name] = {
                "total_policies": len(df),
                "data_structure": list(df.columns),
                "preview": df.head(2).to_dict('records') if len(df) > 0 else []
            }
    
    return policies_data

# =============================================================================
# VISUALIZATION & EXPORT TOOLS
# =============================================================================

@mcp.tool()
def GetLSEVisualizationData(viz_type: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get structured data for specific visualization types.
    
    Parameters:
    - viz_type: Type of visualization ('states_comparison', 'policy_coverage', 'module_overview', 'governance_status')
    - filters: Optional filters like {'module': 'subnational', 'region': 'southeast'}
    """
    if not data_manager.all_sheets:
        return {"error": "LSE data not available"}
    
    filters = filters or {}
    
    if viz_type == "states_comparison":
        return _get_states_comparison_viz(filters)
    elif viz_type == "policy_coverage":
        return _get_policy_coverage_viz(filters)
    elif viz_type == "module_overview":
        return _get_module_overview_viz(filters)
    elif viz_type == "governance_status":
        return _get_governance_status_viz(filters)
    else:
        return {"error": f"Unknown visualization type: {viz_type}. Available: states_comparison, policy_coverage, module_overview, governance_status"}

def _get_states_comparison_viz(filters: Dict) -> Dict[str, Any]:
    """Get data for Brazilian states comparison visualization."""
    if not data_manager.states_cache:
        return {"error": "States data not available"}
    
    comparison_data = []
    for sheet_key, state_info in data_manager.states_cache.items():
        df = state_info['data']
        if df.empty:
            continue
        
        # Count Yes/No responses
        yes_count = 0
        no_count = 0
        total_questions = 0
        
        for idx, row in df.iterrows():
            for col in df.columns:
                if 'yes' in col.lower() and 'no' in col.lower():
                    answer = str(row[col]) if not pd.isna(row[col]) else ''
                    if answer:
                        total_questions += 1
                        if 'yes' in answer.lower():
                            yes_count += 1
                        elif 'no' in answer.lower():
                            no_count += 1
                    break
        
        if total_questions > 0:
            comparison_data.append({
                "state": state_info['state_name'],
                "yes_responses": yes_count,
                "no_responses": no_count,
                "total_questions": total_questions,
                "policy_coverage": round((yes_count / total_questions) * 100, 2)
            })
    
    return {
        "visualization_type": "states_comparison",
        "data": comparison_data,
        "chart_config": {
            "x_axis": "state",
            "y_axis": "policy_coverage",
            "title": "Climate Policy Coverage by Brazilian State",
            "chart_type": "bar"
        }
    }

def _get_policy_coverage_viz(filters: Dict) -> Dict[str, Any]:
    """Get data for policy coverage visualization."""
    coverage_data = []
    
    for module, content_list in data_manager.content_cache.items():
        total_entries = 0
        total_sheets = len(content_list)
        
        for filename, sheet_name, df in content_list:
            total_entries += len(df)
        
        coverage_data.append({
            "module": module.replace('_', ' ').title(),
            "total_sheets": total_sheets,
            "total_entries": total_entries,
            "avg_entries_per_sheet": round(total_entries / total_sheets, 2) if total_sheets > 0 else 0
        })
    
    return {
        "visualization_type": "policy_coverage",
        "data": coverage_data,
        "chart_config": {
            "x_axis": "module",
            "y_axis": "total_entries",
            "title": "Policy Data Coverage by Module",
            "chart_type": "bar"
        }
    }

def _get_module_overview_viz(filters: Dict) -> Dict[str, Any]:
    """Get data for module overview visualization."""
    module_data = []
    
    for module, content_list in data_manager.content_cache.items():
        files_count = len(set([filename for filename, _, _ in content_list]))
        sheets_count = len(content_list)
        
        module_data.append({
            "module": module.replace('_', ' ').title(),
            "files": files_count,
            "sheets": sheets_count,
            "description": _get_module_description(module)
        })
    
    return {
        "visualization_type": "module_overview",
        "data": module_data,
        "chart_config": {
            "x_axis": "module",
            "y_axis": "sheets",
            "title": "LSE Climate Policy Data Modules",
            "chart_type": "bar"
        }
    }

def _get_governance_status_viz(filters: Dict) -> Dict[str, Any]:
    """Get data for governance status visualization."""
    if not data_manager.states_cache:
        return {"error": "Governance status data not available"}
    
    status_counts = {}
    
    for sheet_key, state_info in data_manager.states_cache.items():
        df = state_info['data']
        if df.empty:
            continue
        
        # Look for status column
        for col in df.columns:
            if 'status' in col.lower():
                for idx, row in df.iterrows():
                    status = str(row[col]) if not pd.isna(row[col]) else 'Unknown'
                    if status and status != 'nan':
                        status_counts[status] = status_counts.get(status, 0) + 1
                break
    
    status_data = [{"status": k, "count": v} for k, v in status_counts.items()]
    
    return {
        "visualization_type": "governance_status",
        "data": status_data,
        "chart_config": {
            "x_axis": "status",
            "y_axis": "count",
            "title": "Climate Governance Implementation Status",
            "chart_type": "pie"
        }
    }

@mcp.tool()
def GetLSEDatasetMetadata() -> Dict[str, Any]:
    """
    Get comprehensive metadata about the LSE dataset.
    """
    return metadata

# Clean up the exploration script
if os.path.exists("/Users/mason/Documents/GitHub/tde/explore_lse.py"):
    os.remove("/Users/mason/Documents/GitHub/tde/explore_lse.py")

if __name__ == "__main__":
    mcp.run()