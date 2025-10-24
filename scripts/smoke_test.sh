#!/bin/bash
# Smoke test script - verifies critical functionality before deployment
# Run this before pushing to ensure v2 architecture is working

# Note: We don't use set -e because we want to collect all test results

echo "======================================"
echo "TDE v2 Smoke Tests"
echo "======================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to report test results
test_result() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $1"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $1"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "1. Testing Python imports..."
echo "--------------------------------------"

python -c "from mcp.mcp_chat_v2 import process_chat_query, stream_chat_query, get_global_client, cleanup_global_client" 2>&1 > /dev/null
test_result "MCP v2 orchestrator imports"

python -c "from mcp.contracts_v2 import RunQueryResponse, FactPayload, CitationPayload" 2>&1 > /dev/null
test_result "MCP v2 contracts imports"

python -c "import api_server" 2>&1 | grep -q "Using mcp_chat_v2"
test_result "API server uses mcp_chat_v2 orchestrator"

echo ""
echo "2. Checking server files..."
echo "--------------------------------------"

[ -f "mcp/servers_v2/solar_server_v2.py" ]
test_result "Solar server v2 exists"

[ -f "mcp/servers_v2/lse_server_v2.py" ]
test_result "LSE server v2 exists"

[ -f "mcp/servers_v2/cpr_server_v2.py" ]
test_result "CPR server v2 exists"

! [ -f "mcp/servers_v2/cpr_server_v2 copy.py" ]
test_result "No duplicate server files"

echo ""
echo "3. Checking for legacy code..."
echo "--------------------------------------"

! find mcp/ -name "mcp_chat.py" -o -name "mcp_chat_redo.py" -o -name "mcp_chat_plan_execute.py" 2>/dev/null | grep -q "."
test_result "No legacy orchestrator files present"

! grep -r "import streamlit" --include="*.py" mcp/ api_server.py 2>&1 | grep -v "__pycache__" | grep -q "."
test_result "No streamlit imports in codebase"

echo ""
echo "4. Checking configuration..."
echo "--------------------------------------"

[ -f ".env.example" ]
test_result ".env.example exists"

[ -f "requirements.txt" ]
test_result "requirements.txt exists"

! grep -q "streamlit" requirements.txt
test_result "Streamlit removed from requirements.txt"

grep -q "fastapi" requirements.txt
test_result "FastAPI in requirements.txt"

echo ""
echo "5. Checking documentation..."
echo "--------------------------------------"

[ -f "data/README.md" ]
test_result "data/README.md exists"

[ -f "CLAUDE.md" ]
test_result "CLAUDE.md exists"

echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"
echo -e "${GREEN}Passed: ${TESTS_PASSED}${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed: ${TESTS_FAILED}${NC}"
    echo ""
    echo "Some tests failed. Please fix issues before deploying."
    exit 1
else
    echo -e "${YELLOW}Failed: ${TESTS_FAILED}${NC}"
    echo ""
    echo -e "${GREEN}All smoke tests passed! ✓${NC}"
    echo "v2 architecture is ready."
    exit 0
fi
