#!/usr/bin/env python3
"""
Example script showing how to get structured JSON responses 
for front-end consumption.
"""
import asyncio
import json
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from mcp.mcp_chat import run_query_structured

async def get_structured_response(query: str):
    """
    Get a structured JSON response that matches your response.json format.
    
    Returns a dictionary with 'modules' array containing:
    - text modules (with heading and texts)
    - chart modules (Chart.js compatible)
    - table modules (with columns and rows)
    - image modules (with src and alt)
    """
    
    # Set working directory to ensure relative paths work
    os.chdir("/mnt/o/Ode/Github/tde")
    
    try:
        structured_response = await run_query_structured(query)
        return structured_response
    except Exception as e:
        print(f"Error: {e}")
        return {"modules": []}

async def main():
    """Example usage for different types of queries."""
    
    print("🔍 Testing structured response formatting...\n")
    
    # Example queries
    test_queries = [
        "What datasets are available?",
        "Show me solar capacity by country", 
        "Tell me about solar energy in Brazil",
        "Compare solar facilities across countries"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("=" * 50)
        
        response = await get_structured_response(query)
        
        # Pretty print the JSON
        print(json.dumps(response, indent=2))
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    # Run example
    asyncio.run(main())
