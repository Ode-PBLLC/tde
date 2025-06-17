import asyncio
import sys
import os
from pathlib import Path

# Automatically add the mcp module directory to sys.path
project_root = Path("/mnt/o/Ode/Github/tde")
mcp_path = project_root / "mcp"
sys.path.insert(0, str(mcp_path))

from mcp_chat import run_query_structured  # Import from mcp directly

async def test():
    result = await run_query_structured('Can you tell me about Solar Power? Is there Solar Power in Brazil?')
    print('SUCCESS! Got structured response with', len(result.get('modules', [])), 'modules')

    # Check for maps
    for module in result.get('modules', []):
        if module.get('type') == 'map':
            geojson = module.get('geojson', {})
            print(f'Found map with {len(geojson.get("features", []))} GeoJSON features')

asyncio.run(test())
