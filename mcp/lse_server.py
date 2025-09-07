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
            self._build_ndc_index()
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
    
    def _build_ndc_index(self):
        """Build index of NDC data from Sheet1."""
        self.ndc_cache = {}
        
        # Look for NDC Overview file
        ndc_file = '1 NDC Overview and Domestic Policy Comparison Content.xlsx::Sheet1'
        if ndc_file in self.all_sheets:
            df = self.all_sheets[ndc_file]
            
            # Parse the Q&A format
            for idx, row in df.iterrows():
                # Extract question from first column
                question = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ''
                
                # Extract NDC response (usually column 2)
                ndc_response = str(row.iloc[1]) if len(row) > 1 and not pd.isna(row.iloc[1]) else ''
                
                # Extract domestic policy response and sources
                policy_response = str(row.iloc[2]) if len(row) > 2 and not pd.isna(row.iloc[2]) else ''
                source = str(row.iloc[3]) if len(row) > 3 and not pd.isna(row.iloc[3]) else ''
                
                if question and (ndc_response or policy_response):
                    # Clean up the question for use as a key
                    question_key = question.strip().replace('?', '').lower()
                    
                    self.ndc_cache[question_key] = {
                        'question': question,
                        'ndc_response': ndc_response,
                        'policy_response': policy_response,
                        'source': source,
                        'row_index': idx
                    }
        
        print(f"Built NDC index with {len(self.ndc_cache)} entries")
    
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
def GetTPIGraphData() -> Dict[str, Any]:
    """Get TPI (Transition Pathway Initiative) emissions pathway graph data.
    
    Returns emissions data in MtCO₂e for different scenarios and pathways.
    """
    tpi_file = '1_1 TPI Graph [on NDC Overview].xlsx::Sheet1'
    
    if tpi_file not in data_manager.all_sheets:
        return {"error": "TPI graph data not available"}
    
    df = data_manager.all_sheets[tpi_file]
    
    tpi_data = {
        "description": "TPI emissions pathway data for Brazil",
        "unit": "MtCO₂e",
        "data_points": [],
        "columns": list(df.columns),
        "shape": {"rows": len(df), "columns": len(df.columns)}
    }
    
    # Extract emissions data
    for idx, row in df.iterrows():
        row_data = {}
        for col in df.columns:
            value = row[col]
            if not pd.isna(value):
                # Clean column name
                col_name = str(col).replace('Unnamed: ', 'Column_')
                row_data[col_name] = value
        
        if row_data:  # Only add non-empty rows
            tpi_data['data_points'].append(row_data)
    
    return tpi_data

@mcp.tool()
def GetInstitutionalFramework(topic: Optional[str] = None) -> Dict[str, Any]:
    """Get institutional framework and governance data.
    
    Topics include: Direction setting, Planning and strategy, Knowledge and evidence,
    Integration, Coordination, Transparency and accountability, Participation,
    Public Finance, Private finance.
    """
    institutions_data = {}
    
    # Get relevant sheets from Institutions file
    for sheet_key, df in data_manager.all_sheets.items():
        if '2 Institutions and Processes Module Content.xlsx' not in sheet_key:
            continue
            
        sheet_name = sheet_key.split('::')[1]
        
        # Skip metadata sheets
        if sheet_name in ['How to use', 'Metadata']:
            continue
        
        # Filter by topic if specified
        if topic and topic.lower() not in sheet_name.lower():
            continue
        
        # Parse the institutional data
        sheet_data = {
            'topic': sheet_name,
            'entries': [],
            'summary_stats': {}
        }
        
        for idx, row in df.iterrows():
            entry = {}
            
            # Common columns in institutional sheets
            if 'Pergunta / Indicador' in df.columns:
                entry['question'] = row.get('Pergunta / Indicador')
            if 'Resposta' in df.columns:
                entry['response'] = row.get('Resposta')
            if 'Status' in df.columns:
                entry['status'] = row.get('Status')
            if 'Primary source for answer' in df.columns:
                entry['primary_source'] = row.get('Primary source for answer')
            
            # Only add entries with content
            if any(v and str(v) != 'nan' for v in entry.values()):
                sheet_data['entries'].append(entry)
        
        # Calculate summary statistics
        if sheet_data['entries']:
            responses = [e.get('response', '') for e in sheet_data['entries']]
            sheet_data['summary_stats'] = {
                'total_questions': len(sheet_data['entries']),
                'yes_responses': sum(1 for r in responses if 'yes' in str(r).lower()),
                'no_responses': sum(1 for r in responses if 'no' in str(r).lower()),
                'with_sources': sum(1 for e in sheet_data['entries'] if e.get('primary_source'))
            }
        
        institutions_data[sheet_name] = sheet_data
    
    if not institutions_data:
        return {"error": f"No institutional data found{' for topic: ' + topic if topic else ''}"}
    
    return {
        "topics_available": list(institutions_data.keys()),
        "data": institutions_data
    }

@mcp.tool()
def GetClimatePolicy(policy_type: Optional[str] = None) -> Dict[str, Any]:
    """Get climate plans and policies data.
    
    Policy types include: Cross Cutting Policies, Sectoral mitigation plans,
    Sectoral adaptation plans.
    """
    policies_data = {}
    
    # Get relevant sheets from Plans and Policies file
    for sheet_key, df in data_manager.all_sheets.items():
        if '3 Plans and Policies Module Content.xlsx' not in sheet_key:
            continue
            
        sheet_name = sheet_key.split('::')[1]
        
        # Skip metadata sheets
        if sheet_name in ['How to use', 'Metadata lists']:
            continue
        
        # Filter by policy type if specified
        if policy_type and policy_type.lower() not in sheet_name.lower():
            continue
        
        # Parse the policy data
        sheet_data = {
            'policy_type': sheet_name,
            'policies': [],
            'implementation_status': {}
        }
        
        for idx, row in df.iterrows():
            policy = {}
            
            # Common columns in policy sheets
            if 'Questions' in df.columns:
                policy['question'] = row.get('Questions')
            if 'Justification' in df.columns:
                policy['justification'] = row.get('Justification')
            if 'Summary' in df.columns or 'Summary ' in df.columns:
                policy['summary'] = row.get('Summary', row.get('Summary '))
            if 'Status' in df.columns:
                policy['status'] = row.get('Status')
            if 'Implementation information' in df.columns:
                policy['implementation'] = row.get('Implementation information')
            if 'Primary source for answer' in df.columns:
                policy['source'] = row.get('Primary source for answer')
            
            # Only add policies with content
            if any(v and str(v) != 'nan' for v in policy.values()):
                sheet_data['policies'].append(policy)
        
        # Track implementation status
        if sheet_data['policies']:
            statuses = [p.get('status', '') for p in sheet_data['policies']]
            sheet_data['implementation_status'] = {
                'total_policies': len(sheet_data['policies']),
                'implemented': sum(1 for s in statuses if 'implement' in str(s).lower()),
                'planned': sum(1 for s in statuses if 'plan' in str(s).lower()),
                'with_sources': sum(1 for p in sheet_data['policies'] if p.get('source'))
            }
        
        policies_data[sheet_name] = sheet_data
    
    if not policies_data:
        return {"error": f"No policy data found{' for type: ' + policy_type if policy_type else ''}"}
    
    return {
        "policy_types_available": list(policies_data.keys()),
        "data": policies_data
    }

@mcp.tool()
def GetSubnationalGovernance(state: Optional[str] = None, metric: Optional[str] = None) -> Dict[str, Any]:
    """Get Brazilian state-level climate governance data.
    
    Access detailed climate governance assessments for Brazilian states.
    States use format like 'São Paulo (SP)' or just 'SP'.
    """
    subnational_data = {}
    
    # Get relevant sheets from Subnational file
    for sheet_key, df in data_manager.all_sheets.items():
        if '4 Subnational Module Content.xlsx' not in sheet_key:
            continue
            
        sheet_name = sheet_key.split('::')[1]
        
        # Skip metadata sheets
        if sheet_name in ['How to use', 'Metadata']:
            continue
        
        # Filter by state if specified
        if state:
            # Handle both full name and abbreviation
            state_upper = state.upper()
            if state_upper not in sheet_name and f'({state_upper})' not in sheet_name:
                continue
        
        # Parse the state data
        state_data = {
            'state_name': sheet_name,
            'governance_metrics': [],
            'climate_policies': [],
            'summary': {}
        }
        
        # Extract data based on sheet structure
        for idx, row in df.iterrows():
            # Try to identify the type of data
            first_col = row.iloc[0] if len(row) > 0 else None
            
            if pd.notna(first_col):
                entry = {'indicator': str(first_col)}
                
                # Add other columns
                for i, col in enumerate(df.columns[1:], 1):
                    if pd.notna(row.iloc[i]):
                        entry[f'value_{i}'] = str(row.iloc[i])
                
                # Categorize the entry
                if metric:
                    if metric.lower() in str(first_col).lower():
                        state_data['governance_metrics'].append(entry)
                else:
                    state_data['governance_metrics'].append(entry)
        
        # Calculate summary
        if state_data['governance_metrics']:
            state_data['summary'] = {
                'total_indicators': len(state_data['governance_metrics']),
                'state_code': sheet_name.split('(')[-1].replace(')', '') if '(' in sheet_name else sheet_name
            }
        
        if state_data['governance_metrics'] or state_data['climate_policies']:
            subnational_data[sheet_name] = state_data
    
    if not subnational_data:
        return {"error": f"No subnational data found{' for state: ' + state if state else ''}"}
    
    # Get list of all available states
    all_states = []
    for sheet_key in data_manager.all_sheets.keys():
        if '4 Subnational Module Content.xlsx' in sheet_key:
            sheet_name = sheet_key.split('::')[1]
            if sheet_name not in ['How to use', 'Metadata']:
                all_states.append(sheet_name)
    
    return {
        "states_available": all_states,
        "data": subnational_data,
        "total_states": len(all_states)
    }

@mcp.tool()
def GetLSEDatasetOverview() -> Dict[str, Any]:
    """Get overview of LSE climate policy datasets."""
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
    """Get structure of specific LSE file."""
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
    """Search LSE content for specific terms."""
    if not data_manager.all_sheets:
        return {"error": "LSE data not available"}
    
    results = data_manager.search_content(search_term, module_type)
    
    # If no results found, provide helpful placeholder content
    if len(results) == 0:
        placeholder_results = [{
            "content": f"No specific results found for '{search_term}'. The LSE climate governance database contains comprehensive information on climate policies, institutional frameworks, and governance approaches. Consider broader search terms or explore available modules: ndc_overview, institutions, plans_policies, subnational.",
            "module": "search_guidance",
            "relevance_score": 0.5,
            "data_type": "guidance"
        }]
        results = placeholder_results
    
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
    """Get Brazilian states climate governance overview."""
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
    """Get climate policy for specific Brazilian state."""
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
    """Compare climate policies across Brazilian states."""
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
# NDC-SPECIFIC TOOLS
# =============================================================================

@mcp.tool()
def GetNDCTargets(country: str = "Brazil") -> Dict[str, Any]:
    """Get specific NDC targets and commitments for a country.
    
    Returns emissions reduction targets, net-zero dates, renewable energy goals,
    and other quantitative commitments from the NDC.
    """
    if not data_manager.ndc_cache:
        return {"error": "NDC data not available"}
    
    targets = {
        "country": country,
        "long_term_target": None,
        "interim_targets": {},
        "adaptation_goals": None,
        "principles": None,
        "key_commitments": [],
        "sources": []
    }
    
    # Extract specific targets from NDC cache
    for key, data in data_manager.ndc_cache.items():
        question = data['question']
        ndc_response = data['ndc_response']
        
        # Long-term target (net-zero)
        if 'long term' in key and 'emissions reduction' in key:
            if ndc_response and ndc_response != 'nan':
                targets['long_term_target'] = {
                    'commitment': ndc_response,
                    'question': question
                }
                # Extract specific values
                if '2050' in ndc_response:
                    targets['key_commitments'].append("Climate neutrality by 2050")
        
        # 2030 target
        elif 'interim target for 2030' in key:
            targets['interim_targets']['2030'] = {
                'has_target': 'yes' in ndc_response.lower() if ndc_response else False,
                'details': ndc_response if ndc_response and ndc_response != 'nan' else None
            }
        
        # 2035 target
        elif 'interim target for 2035' in key:
            targets['interim_targets']['2035'] = {
                'has_target': 'yes' in ndc_response.lower() if ndc_response else False,
                'details': ndc_response if ndc_response and ndc_response != 'nan' else None
            }
            # Extract specific percentage
            if '59' in ndc_response and '67' in ndc_response:
                targets['key_commitments'].append("59-67% GHG reduction below 2005 levels by 2035")
        
        # 2040 target
        elif 'interim target for 2040' in key:
            targets['interim_targets']['2040'] = {
                'has_target': 'yes' in ndc_response.lower() if ndc_response else False,
                'details': ndc_response if ndc_response and ndc_response != 'nan' else None
            }
        
        # Adaptation goals
        elif 'adaptation' in key and 'goal' in key:
            if ndc_response and ndc_response != 'nan':
                targets['adaptation_goals'] = {
                    'has_goals': 'yes' in ndc_response.lower(),
                    'details': ndc_response
                }
        
        # Guiding principles
        elif 'principles' in key:
            if ndc_response and ndc_response != 'nan':
                targets['principles'] = ndc_response
        
        # Collect sources
        if data['source'] and data['source'] != 'nan':
            targets['sources'].append(data['source'])
    
    # Remove duplicates from sources
    targets['sources'] = list(set(targets['sources']))
    
    return targets

@mcp.tool()
def GetNDCPolicyComparison() -> Dict[str, Any]:
    """Get comparison between NDC commitments and domestic policy.
    
    Shows how NDC targets are reflected in national laws and policies.
    """
    if not data_manager.ndc_cache:
        return {"error": "NDC comparison data not available"}
    
    comparison = {
        "alignment_status": {},
        "gaps_identified": [],
        "implementation_mechanisms": [],
        "legal_framework": {}
    }
    
    for key, data in data_manager.ndc_cache.items():
        question = data['question']
        ndc_response = data['ndc_response']
        policy_response = data['policy_response']
        
        if not ndc_response or ndc_response == 'nan':
            continue
            
        # Check if Paris Agreement is enforceable
        if 'paris agreement' in key and 'enforceable' in key:
            comparison['legal_framework']['paris_agreement_status'] = {
                'enforceable': 'yes' in ndc_response.lower(),
                'details': ndc_response,
                'domestic_implementation': policy_response if policy_response != 'nan' else None
            }
        
        # Compare NDC vs domestic responses
        if policy_response and policy_response != 'nan':
            # Both responses exist - check alignment
            ndc_has_target = 'yes' in ndc_response.lower()
            policy_has_target = 'yes' in policy_response.lower() or 'evidence' in policy_response.lower()
            
            topic = question.split('?')[0] if '?' in question else question
            
            if ndc_has_target and not policy_has_target:
                comparison['gaps_identified'].append({
                    'topic': topic,
                    'gap': 'NDC commitment not yet reflected in domestic policy',
                    'ndc_position': ndc_response[:200],
                    'policy_position': policy_response[:200]
                })
            elif ndc_has_target and policy_has_target:
                comparison['alignment_status'][topic] = {
                    'aligned': True,
                    'ndc_commitment': ndc_response[:200],
                    'domestic_policy': policy_response[:200]
                }
            
            # Extract implementation mechanisms
            if any(word in policy_response.lower() for word in ['law', 'decree', 'regulation', 'act']):
                comparison['implementation_mechanisms'].append({
                    'topic': topic,
                    'mechanism': policy_response[:300]
                })
    
    return comparison

@mcp.tool()
def GetNDCImplementationStatus(country: str = "Brazil") -> Dict[str, Any]:
    """Track NDC implementation progress and status.
    
    Returns information on how NDC commitments are being implemented
    through domestic policies and actions.
    """
    if not data_manager.ndc_cache:
        return {"error": "NDC implementation data not available"}
    
    implementation = {
        "country": country,
        "targets_with_implementation": [],
        "targets_pending_implementation": [],
        "implementation_instruments": [],
        "carbon_pricing": None,
        "monitoring_systems": None
    }
    
    for key, data in data_manager.ndc_cache.items():
        ndc_response = data['ndc_response']
        policy_response = data['policy_response']
        
        if not ndc_response or ndc_response == 'nan':
            continue
        
        # Check carbon budgets/pricing
        if 'carbon budget' in key or 'emissions trading' in key:
            implementation['carbon_pricing'] = {
                'has_system': 'yes' in ndc_response.lower() or 'SBCE' in str(policy_response),
                'details': policy_response if policy_response != 'nan' else ndc_response
            }
        
        # Track implementation status
        if ndc_response and policy_response and policy_response != 'nan':
            has_ndc_commitment = 'yes' in ndc_response.lower()
            has_policy = any(word in policy_response.lower() for word in ['yes', 'law', 'decree', 'established'])
            
            target_info = {
                'target': data['question'],
                'ndc_commitment': ndc_response[:200],
                'implementation': policy_response[:200] if policy_response else 'Pending'
            }
            
            if has_ndc_commitment:
                if has_policy:
                    implementation['targets_with_implementation'].append(target_info)
                else:
                    implementation['targets_pending_implementation'].append(target_info)
            
            # Extract specific instruments
            if 'Law No.' in str(policy_response) or 'Decree' in str(policy_response):
                implementation['implementation_instruments'].append({
                    'topic': data['question'],
                    'instrument': policy_response[:300]
                })
    
    # Calculate implementation rate
    total_targets = len(implementation['targets_with_implementation']) + len(implementation['targets_pending_implementation'])
    if total_targets > 0:
        implementation['implementation_rate'] = round(
            len(implementation['targets_with_implementation']) / total_targets * 100, 1
        )
    
    return implementation

@mcp.tool()
def GetAllNDCData() -> Dict[str, Any]:
    """Get all NDC data for comprehensive analysis.
    
    Returns complete NDC dataset including all questions, responses,
    and domestic policy comparisons.
    """
    if not data_manager.ndc_cache:
        return {"error": "NDC data not available"}
    
    # Structure all NDC data
    ndc_data = {
        "total_entries": len(data_manager.ndc_cache),
        "categories": {},
        "full_data": []
    }
    
    # Categorize entries
    categories = {
        'long_term_targets': [],
        'interim_targets': [],
        'adaptation': [],
        'implementation': [],
        'principles': [],
        'other': []
    }
    
    for key, data in data_manager.ndc_cache.items():
        entry = {
            'question': data['question'],
            'ndc_response': data['ndc_response'] if data['ndc_response'] != 'nan' else None,
            'policy_response': data['policy_response'] if data['policy_response'] != 'nan' else None,
            'source': data['source'] if data['source'] != 'nan' else None
        }
        
        ndc_data['full_data'].append(entry)
        
        # Categorize
        if 'long term' in key:
            categories['long_term_targets'].append(entry)
        elif any(year in key for year in ['2030', '2035', '2040', '2050']):
            categories['interim_targets'].append(entry)
        elif 'adaptation' in key:
            categories['adaptation'].append(entry)
        elif any(word in key for word in ['carbon', 'implementation', 'monitoring']):
            categories['implementation'].append(entry)
        elif 'principle' in key:
            categories['principles'].append(entry)
        else:
            categories['other'].append(entry)
    
    ndc_data['categories'] = {k: len(v) for k, v in categories.items()}
    ndc_data['categorized_data'] = categories
    
    return ndc_data

# =============================================================================
# POLICY ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
def GetNDCOverviewData(country: Optional[str] = None) -> Dict[str, Any]:
    """Get NDC overview data."""
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
    """Get institutions and processes data."""
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
    """Get plans and policies data."""
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
    """Get structured data for visualization."""
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
    """Get LSE dataset metadata."""
    return metadata

# Clean up the exploration script
if os.path.exists("/Users/mason/Documents/GitHub/tde/explore_lse.py"):
    os.remove("/Users/mason/Documents/GitHub/tde/explore_lse.py")

if __name__ == "__main__":
    mcp.run()