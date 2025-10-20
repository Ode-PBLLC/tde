# LLM-Guided Orchestration Pattern

This document summarizes the pattern we now use for v2 servers (example: solar) so
other datasets can adopt the same orchestration workflow.

## Step-by-step

1. **Expose focused tools**
   - Keep `run_query` only as a fallback.
   - Every capability (map, ranking, timeline, lookup, etc.) gets a dedicated tool.
   - Each tool returns a JSON payload with:
     - `summary`: one-sentence description.
     - `facts`: list of extra sentences.
     - `artifacts`: ready-to-use map/chart/table definitions.
     - `citation`: dataset reference (consistent ID/title/url).
     - Optional raw rows/items if downstream code needs them.
   - No heuristics inside tools—each tool does one thing.

2. **Discover tool metadata dynamically**
   - In the orchestrator, call `await session.list_tools()` to fetch the live tool
     descriptions and default arguments.
   - Cache this manifest so it updates automatically when tools change.

3. **Use an LLM planner for tool selection**
   - When a server is chosen (e.g., by `query_support`), hand the query, prior turn,
     and tool manifest to an LLM that emits a JSON plan (e.g., `{"tools": [{"name": ...}]}`).
   - Log the plan for inspection.
   - If the planner fails or declines, do **not** fall back to heuristics—log the issue
     and skip these tools. Better to fix the planner/credentials than silently inject
     unwanted results.

4. **Execute the plan and assemble responses**
   - Call each tool via `session.call_tool`.
   - Convert the returned JSON into a synthetic `RunQueryResponse` (facts, artifacts,
     citations, messages, KG nodes).
   - Merge that response with any other server outputs and pass through the existing
     narrative synthesis + response formatter.

5. **Prevent map collisions with metadata**
   - Add a `merge_group` (or similar tag) to map metadata.
   - The formatter only merges maps with matching `merge_group`; other maps stay separate.

6. **Remove broad heuristics**
   - Servers should only emit content when the planner invokes them.
   - Keep `query_support` LLM-driven (with narrow heuristics as fallback if absolutely
     necessary) and avoid automatic “dataset highlights.”

7. **Add planner-validation scripts**
   - Create scripts under `test_scripts/` that list representative queries and the
     expected tool list.
   - Run the planner to confirm the selected tools match expectations.
   - Developers should ask the maintainer to run these scripts (they require real
     credentials).

## Resulting Flow

- LLM planner chooses the tools based on the live manifest.
- Orchestrator gathers the tool outputs (artifacts/facts/citations).
- Narrative synthesizer and response formatter turn those into the final answer.

Apply these steps to each server in turn so the orchestration logic becomes consistent,
LLM-driven, and easier to debug across the stack.
