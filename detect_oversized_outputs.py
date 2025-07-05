#!/usr/bin/env python3
"""
Detect oversized tool outputs that could cause prompt length issues.

This script analyzes MCP tool implementations to identify functions
that might return large datasets causing rate limit problems.
"""

import re
import os
from typing import Dict, List, Any, Tuple

def analyze_function_for_size_risk(func_content: str, func_name: str) -> Dict[str, Any]:
    """Analyze a function for potential size risks."""
    
    risk_score = 0
    issues = []
    
    # Check for high default limits
    limit_matches = re.findall(r'limit:\s*int\s*=\s*(\d+)', func_content)
    for limit in limit_matches:
        limit_val = int(limit)
        if limit_val >= 1000:
            risk_score += 10
            issues.append(f"High default limit: {limit_val}")
        elif limit_val >= 500:
            risk_score += 5
            issues.append(f"Medium default limit: {limit_val}")
    
    # Check for .to_dict('records') without limits
    if ".to_dict('records')" in func_content:
        # Check if there's a limit applied
        if not re.search(r'\.head\(\d+\)|\.iloc\[:\d+\]|limit.*\d+', func_content):
            risk_score += 15
            issues.append("Uses .to_dict('records') without apparent limits")
        else:
            risk_score += 2
            issues.append("Uses .to_dict('records') with limits")
    
    # Check for iteration over large datasets
    if "for _, " in func_content and ("append" in func_content or "full_data" in func_content):
        risk_score += 8
        issues.append("Iterates over dataset building arrays")
    
    # Check for full_data arrays
    if "full_data" in func_content:
        risk_score += 12
        issues.append("Returns full_data arrays")
    
    # Check for keywords indicating large returns
    large_return_patterns = [
        r'return.*facilities.*\[',
        r'return.*assets.*\[',
        r'return.*companies.*\[',
        r'\.to_dict\(.*\)',
    ]
    
    for pattern in large_return_patterns:
        if re.search(pattern, func_content, re.IGNORECASE):
            risk_score += 3
            issues.append(f"Potentially large return pattern: {pattern}")
    
    # Categorize risk level
    if risk_score >= 20:
        risk_level = "HIGH"
    elif risk_score >= 10:
        risk_level = "MEDIUM"
    elif risk_score >= 5:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"
    
    return {
        "function_name": func_name,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "issues": issues,
        "function_length": len(func_content)
    }

def extract_mcp_functions(file_path: str) -> List[Tuple[str, str]]:
    """Extract MCP tool functions from a file."""
    
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all @mcp.tool() decorated functions
    functions = []
    
    # Pattern to match @mcp.tool() followed by function definition
    pattern = r'@mcp\.tool\(\)\s*\ndef\s+(\w+)\((.*?)\).*?:\s*(""".*?""")?(.*?)(?=@mcp\.tool\(\)|def\s+\w+|class\s+\w+|\Z)'
    
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        func_name = match.group(1)
        func_params = match.group(2)
        func_docstring = match.group(3) or ""
        func_body = match.group(4)
        
        full_function = func_params + func_docstring + func_body
        functions.append((func_name, full_function))
    
    return functions

def analyze_server_file(file_path: str, server_name: str) -> Dict[str, Any]:
    """Analyze an entire MCP server file for size risks."""
    
    functions = extract_mcp_functions(file_path)
    
    if not functions:
        return {
            "server_name": server_name,
            "error": f"No MCP functions found in {file_path}"
        }
    
    function_analyses = []
    total_risk_score = 0
    high_risk_functions = []
    
    for func_name, func_content in functions:
        analysis = analyze_function_for_size_risk(func_content, func_name)
        function_analyses.append(analysis)
        total_risk_score += analysis["risk_score"]
        
        if analysis["risk_level"] in ["HIGH", "MEDIUM"]:
            high_risk_functions.append(analysis)
    
    return {
        "server_name": server_name,
        "total_functions": len(functions),
        "total_risk_score": total_risk_score,
        "average_risk_score": total_risk_score / len(functions) if functions else 0,
        "high_risk_functions": high_risk_functions,
        "all_functions": function_analyses
    }

def main():
    """Main analysis function."""
    
    print("🔍 MCP Tool Output Size Risk Analysis")
    print("=" * 60)
    
    servers = [
        {"name": "Knowledge Graph", "path": "mcp/cpr_kg_server.py"},
        {"name": "Solar Facilities", "path": "mcp/solar_facilities_server.py"},
        {"name": "GIST Environmental", "path": "mcp/gist_server.py"},
        {"name": "LSE Policy", "path": "mcp/lse_server.py"},
        {"name": "Response Formatter", "path": "mcp/response_formatter_server.py"}
    ]
    
    all_high_risk = []
    total_risk_score = 0
    
    for server in servers:
        print(f"\n📋 {server['name']}:")
        
        analysis = analyze_server_file(server["path"], server["name"])
        
        if "error" in analysis:
            print(f"   ❌ {analysis['error']}")
            continue
        
        total_functions = analysis["total_functions"]
        server_risk = analysis["total_risk_score"]
        avg_risk = analysis["average_risk_score"]
        high_risk_funcs = analysis["high_risk_functions"]
        
        print(f"   Functions: {total_functions}")
        print(f"   Total Risk Score: {server_risk}")
        print(f"   Average Risk: {avg_risk:.1f}")
        print(f"   High Risk Functions: {len(high_risk_funcs)}")
        
        if high_risk_funcs:
            print(f"   🚨 HIGH RISK FUNCTIONS:")
            for func in high_risk_funcs:
                print(f"     • {func['function_name']} ({func['risk_level']}, score: {func['risk_score']})")
                for issue in func['issues'][:2]:  # Show first 2 issues
                    print(f"       - {issue}")
        
        all_high_risk.extend(high_risk_funcs)
        total_risk_score += server_risk
    
    print(f"\n" + "=" * 60)
    print("🎯 OVERALL RISK ASSESSMENT")
    print(f"   Total High Risk Functions: {len(all_high_risk)}")
    print(f"   Combined Risk Score: {total_risk_score}")
    
    # Sort high risk functions by score
    all_high_risk.sort(key=lambda x: x["risk_score"], reverse=True)
    
    print(f"\n🚨 TOP RISK FUNCTIONS (Immediate Action Required):")
    for i, func in enumerate(all_high_risk[:5], 1):
        print(f"   {i}. {func['function_name']} (Score: {func['risk_score']})")
        print(f"      Risk Level: {func['risk_level']}")
        print(f"      Issues: {', '.join(func['issues'][:3])}")
        
        # Provide specific recommendations
        recommendations = []
        if any("High default limit" in issue for issue in func['issues']):
            recommendations.append("Reduce default limit to 100-200")
        if any("without apparent limits" in issue for issue in func['issues']):
            recommendations.append("Add .head(limit) before .to_dict('records')")
        if any("full_data arrays" in issue for issue in func['issues']):
            recommendations.append("Return summary only, generate files server-side")
        
        if recommendations:
            print(f"      Recommendations: {'; '.join(recommendations)}")
        print()
    
    print(f"💡 SUMMARY RECOMMENDATIONS:")
    print(f"   1. Functions with score ≥20: Immediate fixes required")
    print(f"   2. Functions with score ≥10: Review and optimize")
    print(f"   3. Add default limits of 100-200 for map/list functions")
    print(f"   4. Use server-side file generation instead of returning full_data")
    print(f"   5. Implement token counting in MCP orchestration")

if __name__ == "__main__":
    main()