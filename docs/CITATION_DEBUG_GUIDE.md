# Citation System Debug Guide

## Primary Files to Examine

### 1. `/Users/mason/Documents/GitHub/tde/mcp/response_formatter_server.py`

**Key sections:**
- **Lines 314-323**: Where `_insert_contextual_citations` is called for text modules
- **Lines 93-157**: The `_insert_contextual_citations` function itself  
- **Lines 124-142**: The citation matching logic I just updated

### 2. `/Users/mason/Documents/GitHub/tde/mcp/mcp_chat.py`

**Key sections:**
- **Lines 2110-2113 and 2165-2168**: Where citation registry is passed to formatter
- **Lines 1288 and 1308**: Where citations are added to registry with module IDs

## Key Things to Check

### 1. Are citations being added to the registry?
Look for calls to `self.citation_registry.add_source()` in mcp_chat.py

### 2. Is the registry being passed correctly?
Check the `citation_registry` parameter in formatter calls

### 3. Are the module IDs matching?
The formatter looks for modules starting with `"tool_"` but citations might be stored with different IDs

### 4. Is the citation matching logic working?
The text contains words like "solar", "capacity", "data" which should match our updated criteria

## Quick Debugging Test

Add a print statement in `_insert_contextual_citations` at line 110:

```python
print(f"DEBUG: Function called with {len(all_citations)} citations, registry keys: {list(citation_registry.keys()) if citation_registry else 'None'}")
```

This will show if the function is being called and what data it receives.

## Expected Flow

1. Tools add sources to citation registry with module IDs like `tool_{tool_name}`
2. Registry gets converted to dict format with `"citations"` and `"module_citations"` keys
3. Formatter calls `_insert_contextual_citations` with text and all available citations
4. Function should match citations to text content and insert `^1^` markers
5. Streamlit app should convert `^1^` markers to HTML superscripts

## Current Issue

The Streamlit app isn't displaying citation text because the API isn't generating citation markers like `^1^` in the text. The Streamlit formatting function works correctly - the problem is in the API citation generation.

## Status

- ✅ Streamlit app citation formatting function is working
- ❌ API is not inserting citation markers into text
- 🔧 Updated citation matching logic to be more permissive
- ❓ Need to verify if citations are being added to registry and passed to formatter