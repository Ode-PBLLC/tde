# Plan + Execute Orchestrator (mcp_chat_plan_execute.py)

This module introduces a minimal, deterministic Plan→Execute orchestrator that keeps the “magical” feel while making execution easier to debug and reason about. It preserves the exact output structure and streaming events that the frontend already expects.

## Why

- Hard to debug multi‑phase flow and LLM‑driven tool sequencing
- Interleaved responsibilities across orchestrator and servers
- Opaque failure modes (e.g., spatial correlation silently missing a prerequisite)

The Plan→Execute approach makes each step explicit, traceable, and time‑bounded, without changing the client contract.

## What It Is

- A small planner that detects key intents and produces a concrete plan: an ordered list of tool calls with arguments and timeouts.
- A deterministic executor that runs the plan, logs a decision timeline, and builds the same response shape as `mcp_chat_redo`.
- A focused first recipe: spatial correlation for “solar facilities near heat stress” (and similar phrasing). Non‑spatial queries are delegated to `mcp_chat_redo` for full coverage.

## Public API (unchanged)

- `process_chat_query(user_query, conversation_history=None, correlation_session_id=None, target_language=None) -> dict`
- `stream_chat_query(user_query, conversation_history=None, correlation_session_id=None, target_language=None) -> async iterator`

These functions mirror `mcp_chat_redo` so you can switch orchestrators without frontend changes.

## Streaming Events (unchanged)

- `thinking` – progress updates
- `facts_summary` – small numeric summary
- `complete` – final response object
- `error` – error details

## Response Structure (unchanged)

Returns a dict with:
- `query: str`
- `modules: List[Module]` (text/table/map/chart, same schema)
- `metadata: dict` including counts and flags; additionally includes `decision_timeline` for debugging.

## Where It Lives

- File: `mcp_chat_plan_execute.py` (same directory as `mcp_chat_redo.py`)
- Reuses the global MCP client from `mcp_chat_redo.get_global_client()` (no server changes required)

## Current Recipe: Spatial Correlation (Solar ↔ Heat)

Triggered when the query contains spatial operators like “within/near/overlap/inside/…”, or a distance (e.g., “1 km”, “500 m”).

Plan steps:
1. `solar.GetFacilitiesForGeospatial(country='Brazil', limit=10000)`
2. `geospatial.RegisterEntities(entity_type='solar_facility', entities=<step1.entities>, session_id=<query_id>)`
3. `heat.GetHeatQuintilesForGeospatial(quintiles=[5], limit=5000)`
4. `geospatial.RegisterEntities(entity_type='heat_zone', entities=<step3.entities>, session_id=<query_id>)`
5. `geospatial.FindSpatialCorrelations(entity_type1='solar_facility', entity_type2='heat_zone', method='within' | 'proximity', distance_km=…?, session_id=<query_id>)`
6. `geospatial.GenerateCorrelationMap(session_id=<query_id>)`

Notes:
- If the query says “near 1 km” we switch to `method='proximity'` and parse the distance; otherwise default to `within`.
- If the query mentions deforestation instead of heat, step 5 uses `entity_type2='deforestation_area'` (geospatial server uses its static index).
- Country defaults to Brazil if not stated, keeping behavior predictable.

## Decision Timeline (debugging)

Every tool call is logged in `metadata.decision_timeline` with:
- `ts` – timestamp
- `event` – `tool_start` or `tool_end`
- `call_id` – unique id for the call
- `server` / `tool`
- `args_preview` – sanitized preview of arguments
- `status` – `ok` | `timeout` | `error`
- `result_keys` – top‑level keys of parsed result (when available)

This gives a concise, query‑scoped trace of exactly what ran and how it ended.

## Output Modules Produced by the Correlation Recipe

- Text overview with method and counts
- Map module with `geojson_url` from `GenerateCorrelationMap`
- Table summarizing entity counts in the session

These modules adhere to the existing schema the frontend renders.

## Fallback to Full Orchestrator

If the planner does not detect a spatial correlation intent, the call is delegated to `mcp_chat_redo.process_chat_query`. This preserves all current non‑spatial behaviors without duplication.

## How to Switch (backend‑only)

You can switch the API server to use the new orchestrator without frontend changes by importing:

```python
from mcp_chat_plan_execute import process_chat_query, stream_chat_query
```

The function signatures and outputs are the same.

## Extending with More Recipes

Add to the `Planner.plan()` method and return a `Plan` with `PlanStep`s. The `Executor` takes care of variable interpolation (`${alias.key}`), timeouts, and timeline logging. Suggested next recipes:
- Solar map (no correlation): call `solar.GetSolarFacilitiesMapData` and return map + summary
- Deforestation overlap (explicit): register deforestation polygons when needed, correlate, map
- Policy summary: call KG/LSE composite tools and return a text + table with citations (can later bring over the citation machinery as needed)

## Limitations / Assumptions

- Heat server and geospatial server must be available for the correlation recipe
- Defaults country to Brazil if not specified (tunable)
- Distance parsing is heuristic (supports plain “km”/“m” patterns)
- Uses the existing singleton MCP client and server tools; does not modify servers

## Rationale

This hybrid approach keeps the “magic” (LLM can still choose or narrate) while execution is deterministic and easy to debug. You gain:
- Clear, bounded steps with timeouts
- A per‑query decision timeline
- Identical output and streaming shapes for the frontend

