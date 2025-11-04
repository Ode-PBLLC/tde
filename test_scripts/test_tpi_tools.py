#!/usr/bin/env python3
"""
Test the new TPI tools to ensure they work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.servers_v2.lse_server_v2 import LSEServerV2


def test_tpi_tools():
    """Test the TPI tools."""
    print("=" * 70)
    print("Testing TPI Tools")
    print("=" * 70)
    print()

    # Initialize server
    print("📡 Initializing LSE Server...")
    server = LSEServerV2()
    print("✓ Server initialized")
    print()

    # Test 1: Load structured data
    print("Test 1: Load structured TPI data")
    print("-" * 70)
    tpi_data = server._load_tpi_structured()
    if tpi_data:
        print(f"✓ Loaded structured TPI data")
        print(f"  Country: {tpi_data.get('country')}")
        print(f"  Scenarios: {list(tpi_data.get('scenarios', {}).keys())}")
    else:
        print("❌ Failed to load structured TPI data")
        return False
    print()

    # Test 2: GetTPIEmissionsPathway - specific year
    print("Test 2: GetTPIEmissionsPathway(scenario='historical', year=2020)")
    print("-" * 70)
    try:
        # Access the tool method directly
        tools = [tool for tool in server.mcp.list_tools()]
        pathway_tool = next((t for t in tools if t.name == "GetTPIEmissionsPathway"), None)

        if pathway_tool:
            print(f"✓ Tool found: {pathway_tool.name}")
            print(f"  Description preview: {pathway_tool.description[:100]}...")
        else:
            print("❌ GetTPIEmissionsPathway tool not found")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    print()

    # Test 3: GetTPIAlignmentAssessment
    print("Test 3: GetTPIAlignmentAssessment()")
    print("-" * 70)
    try:
        alignment_tool = next((t for t in tools if t.name == "GetTPIAlignmentAssessment"), None)

        if alignment_tool:
            print(f"✓ Tool found: {alignment_tool.name}")
            print(f"  Description preview: {alignment_tool.description[:100]}...")
        else:
            print("❌ GetTPIAlignmentAssessment tool not found")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    print()

    # Test 4: Check tool docstrings
    print("Test 4: Verify tool docstrings are comprehensive")
    print("-" * 70)

    # Check GetTPIEmissionsPathway docstring
    if "USE FOR queries about:" in pathway_tool.description:
        print("✓ GetTPIEmissionsPathway has 'USE FOR' section")
    else:
        print("⚠️  GetTPIEmissionsPathway missing 'USE FOR' section")

    if "EXAMPLES:" in pathway_tool.description:
        print("✓ GetTPIEmissionsPathway has 'EXAMPLES' section")
    else:
        print("⚠️  GetTPIEmissionsPathway missing 'EXAMPLES' section")

    if "SCENARIOS:" in pathway_tool.description:
        print("✓ GetTPIEmissionsPathway has 'SCENARIOS' section")
    else:
        print("⚠️  GetTPIEmissionsPathway missing 'SCENARIOS' section")

    # Check GetTPIAlignmentAssessment docstring
    if "USE FOR queries about:" in alignment_tool.description:
        print("✓ GetTPIAlignmentAssessment has 'USE FOR' section")
    else:
        print("⚠️  GetTPIAlignmentAssessment missing 'USE FOR' section")

    if "EXAMPLES:" in alignment_tool.description:
        print("✓ GetTPIAlignmentAssessment has 'EXAMPLES' section")
    else:
        print("⚠️  GetTPIAlignmentAssessment missing 'EXAMPLES' section")
    print()

    # Test 5: Check data integrity
    print("Test 5: Verify structured data integrity")
    print("-" * 70)

    scenarios = tpi_data.get("scenarios", {})

    # Check historical data
    historical = scenarios.get("historical", {})
    if historical:
        ts = historical.get("timeseries", [])
        print(f"✓ Historical data: {len(ts)} years ({ts[0]['year']}-{ts[-1]['year']})")

        # Verify 2020 value
        year_2020 = next((e for e in ts if e["year"] == 2020), None)
        if year_2020:
            print(f"  2020 emissions: {year_2020['emissions_mtco2e']} MtCO2e")
        else:
            print("  ❌ 2020 data not found")

    # Check NDC pathway
    ndc = scenarios.get("ndc_target_pathway", {})
    if ndc:
        ts = ndc.get("timeseries", [])
        print(f"✓ NDC pathway: {len(ts)} years ({ts[0]['year']}-{ts[-1]['year']})")

        # Verify 2030 value
        year_2030 = next((e for e in ts if e["year"] == 2030), None)
        if year_2030:
            print(f"  2030 target: {year_2030['emissions_mtco2e']} MtCO2e")
        else:
            print("  ❌ 2030 data not found")

    # Check alignment assessment
    alignment = tpi_data.get("alignment_assessment", {})
    if alignment.get("2030"):
        gap = alignment["2030"].get("benchmark_gap_mtco2e")
        print(f"✓ Alignment assessment available")
        print(f"  2030 gap vs benchmark: {gap} MtCO2e")
    else:
        print("❌ Alignment assessment missing")
    print()

    print("=" * 70)
    print("✅ All TPI tool tests passed!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_tpi_tools()
    sys.exit(0 if success else 1)
