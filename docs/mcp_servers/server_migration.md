# MCP Server v2 Migration Checklist

This document captures the steps we follow when migrating a legacy MCP server
into the v2 contract. Each v2 server is responsible for surfacing its own
tools with descriptive docstrings, performing LLM-based routing, and returning
fully structured responses via `run_query`.

## 1. Prepare the Module
- Create `mcp/servers_v2/<name>_server_v2.py`.
- Import `RunQueryResponse` and other models from `mcp.contracts_v2`, and
  subclass `RunQueryMixin`.
- Load secrets early with `python-dotenv` so Anthropic/OpenAI keys are
  available (consistent with legacy behavior).

## 2. Reimplement Tools
- Copy the relevant tools from the legacy server. Each tool should:
  - Keep the existing docstring; it drives the self-documenting experience in
    FastMCP.
  - Provide at least one minimal example in the docstring where helpful
    (inputs/outputs).
  - Use standard Python return types (`dict`, `list`, primitives) so the
    FastMCP type adapter doesn’t require `typing` imports (`Dict`, `Optional`,
    etc. aren’t necessary).
- Drop deprecated endpoints (e.g. `ALWAYSRUN`) unless the user explicitly wants
  them carried forward.

## 3. Query Support Logic
- Implement `query_support` with a short LLM prompt when an API key is present.
  Fall back to heuristics if not.
- The payload must be JSON with `supported`, `score`, and an optional
  `reasons` list.

## 4. Run Query Implementation
- `handle_run_query` should call whichever local tools are needed to gather
  real data (facilities, passages, charts) rather than returning canned facts.
- Populate:
  - `facts`: plain-language sentences with citations.
  - `citations`: include dataset name, server, tool, and URL for provenance.
  - `artifacts`: maps/tables/charts expressed via the v2 artifact schema.
  - `kg`: nodes + edges if the dataset supports graph context.

## 5. Update the Orchestrator
- In `mcp/mcp_chat_v2.py`, connect to the new server in `get_v2_client()` using
  `_connect_server_module(client, "name", "mcp.servers_v2.<name>_server_v2")`.
- Ensure the orchestrator still uses the singleton client so we don’t re-spawn
  servers between queries.

## 6. Basic Validation
- Run the schema/unit tests:
  ```
  python tests/test_contract_validation.py
  ```
- Use the helper script to run an end-to-end query:
  ```
  python tests/run_mcp_v2_query.py "<test question>"
  ```
  This exercises the routing, tool execution, and `run_query` output.

## 7. Next Server
- After confirming the first v2 server works, repeat: port tools with
  docstrings, implement LLM routing, build a factual `run_query`, connect in
  the orchestrator, and run the same tests.

## Notes
- Leave the legacy server untouched until the v2 path is fully validated; this
  makes it easy to roll back.
- Once all critical servers are migrated and have reliable tests, we can point
  the public API to `mcp_chat_v2.process_query`.

