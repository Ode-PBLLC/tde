#!/usr/bin/env python3
"""
MCP Chat (Plan + Execute)

Goal: Keep the magical feel while making execution predictable and debuggable.
This module implements a minimal hybrid Plan→Execute flow that:

- Reuses the existing MCP singleton connections from mcp_chat_redo
- Detects common intents (esp. spatial correlation like "solar near heat")
- Builds a small, explicit plan
- Executes deterministically with timeouts + a decision timeline
- Returns the same output structure and SSE event shapes as mcp_chat_redo

Scope in this initial version:
- Implements a deterministic correlation recipe: solar facilities ↔ heat zones (top quintile)
- For other queries, delegates to mcp_chat_redo to keep full capability coverage

You can incrementally add more recipes (maps, policy summaries, etc.) into this planner
without touching the frontend.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Reuse the existing global client (sessions to MCP servers) and helpers
from .mcp_chat_redo import get_global_client, cleanup_global_client  # noqa: F401


# -------------------------------
# Utilities
# -------------------------------

def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _parse_distance_km(text: str) -> Optional[float]:
    """Try to parse a distance like '1 km' or '500 m' from the query."""
    ql = (text or "").lower()
    # meters first
    m_match = re.search(r"(\d+(?:\.\d+)?)\s*(m|meter|metre|meters|metres)\b", ql)
    if m_match:
        val = float(m_match.group(1))
        return max(val / 1000.0, 0.001)
    # kilometers
    km_match = re.search(r"(\d+(?:\.\d+)?)\s*(km|kilometer|kilometre|kilometers|kilometres)\b", ql)
    if km_match:
        val = float(km_match.group(1))
        return max(val, 0.001)
    return None


def _extract_json_result(tool_result: Any) -> Any:
    """Best-effort: parse the first text block as JSON; otherwise return the raw object."""
    try:
        if hasattr(tool_result, "content") and tool_result.content:
            first = tool_result.content[0]
            if hasattr(first, "text") and isinstance(first.text, str):
                try:
                    return json.loads(first.text)
                except Exception:
                    return {"_text": first.text}
        # Already a dict/obj
        return tool_result
    except Exception:
        return tool_result


# -------------------------------
# Planning model
# -------------------------------

@dataclass
class PlanStep:
    server: str
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    alias: Optional[str] = None  # name to store result in the execution context
    timeout_s: int = 60


@dataclass
# A very small plan container for now
class Plan:
    intent: str
    steps: List[PlanStep]
    notes: List[str] = field(default_factory=list)


class Planner:
    """Detect intent and build a small, explicit plan."""

    def __init__(self, user_query: str):
        self.query = user_query

    def plan(self) -> Plan:
        ql = (self.query or "").lower()

        # Detect correlation intent
        correlation = any(k in ql for k in [
            "within", "inside", "intersect", "overlap", "near", "close to",
            "proximity", "adjacent", "around", "km", "meter", "metre", "m "
        ])

        if correlation:
            # Defaults for correlation
            distance_km = _parse_distance_km(self.query)
            method = "proximity" if (distance_km is not None) else "within"
            distance_km = 1.0 if distance_km is None and method == "proximity" else distance_km

            steps = [
                # 1) Get solar facility entities for geospatial registration (Brazil default if none inferred)
                PlanStep(
                    server="solar",
                    tool="GetFacilitiesForGeospatial",
                    args={"country": "Brazil", "limit": 10000},
                    alias="solar_entities",
                    timeout_s=60,
                ),
                # 2) Register solar facilities
                PlanStep(
                    server="geospatial",
                    tool="RegisterEntities",
                    args={"entity_type": "solar_facility", "entities": "${solar_entities.entities}", "session_id": "${session_id}"},
                    alias="solar_registered",
                ),
                # 3) Ensure heat zones (top quintile) and register
                PlanStep(
                    server="heat",
                    tool="GetHeatQuintilesForGeospatial",
                    args={"quintiles": [5], "limit": 5000},
                    alias="heat_entities",
                ),
                PlanStep(
                    server="geospatial",
                    tool="RegisterEntities",
                    args={"entity_type": "heat_zone", "entities": "${heat_entities.entities}", "session_id": "${session_id}"},
                    alias="heat_registered",
                ),
                # 4) Correlate
                PlanStep(
                    server="geospatial",
                    tool="FindSpatialCorrelations",
                    args={
                        "entity_type1": "solar_facility",
                        "entity_type2": "deforestation_area" if "deforest" in ql else "heat_zone",
                        "method": method,
                        **({"distance_km": distance_km} if method == "proximity" and distance_km else {}),
                        "session_id": "${session_id}",
                    },
                    alias="correlations",
                    timeout_s=120,
                ),
                # 5) Map
                PlanStep(
                    server="geospatial",
                    tool="GenerateCorrelationMap",
                    args={"session_id": "${session_id}"},
                    alias="corr_map",
                ),
            ]

            notes = [
                f"intent=correlate method={method}",
                (f"distance_km={distance_km}" if distance_km else "distance_km=default"),
                "country=Brazil (assumed if not specified)",
            ]
            return Plan(intent="correlation", steps=steps, notes=notes)

        # Fallback: defer to full orchestrator for non-spatial queries
        return Plan(intent="delegate", steps=[], notes=["delegated_to_mcp_chat_redo"])


# -------------------------------
# Executor with decision timeline
# -------------------------------

class Executor:
    def __init__(self, client, session_id: str, timeline: List[Dict[str, Any]]):
        self.client = client
        self.session_id = session_id
        self.ctx: Dict[str, Any] = {}
        self.timeline = timeline

    async def call_tool_with_trace(self, server: str, tool: str, args: Dict[str, Any], timeout_s: int = 60) -> Any:
        """Call a tool with timeout, variable interpolation, and timeline logging."""
        # Interpolate simple ${...} variables from context
        def _interpolate(val):
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                key = val[2:-1]
                # support nested extraction like solar_entities.entities
                parts = key.split('.')
                data = self.ctx.get(parts[0])
                for p in parts[1:]:
                    if isinstance(data, dict):
                        data = data.get(p)
                    else:
                        data = None
                        break
                return data
            return val

        resolved_args = {}
        for k, v in args.items():
            resolved_args[k] = _interpolate(v)

        # Attach session_id if not present and geospatial/flow needs it
        if "session_id" in args and isinstance(resolved_args.get("session_id"), str) and resolved_args["session_id"] == "${session_id}":
            resolved_args["session_id"] = self.session_id

        call_id = f"{server}.{tool}.{datetime.utcnow().strftime('%H%M%S%f')}"
        start = _now_iso()
        self.timeline.append({
            "ts": start,
            "event": "tool_start",
            "call_id": call_id,
            "server": server,
            "tool": tool,
            "args_preview": {k: ("<list>" if isinstance(v, list) else v) for k, v in resolved_args.items()},
        })

        try:
            coro = self.client.call_tool(tool, resolved_args, server)
            result = await asyncio.wait_for(coro, timeout=timeout_s)
            parsed = _extract_json_result(result)
            end = _now_iso()
            size_hint = None
            try:
                s = json.dumps(parsed)
                size_hint = len(s)
            except Exception:
                size_hint = None
            self.timeline.append({
                "ts": end,
                "event": "tool_end",
                "call_id": call_id,
                "status": "ok",
                "result_keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                "size": size_hint,
            })
            return parsed
        except asyncio.TimeoutError:
            end = _now_iso()
            self.timeline.append({
                "ts": end,
                "event": "tool_end",
                "call_id": call_id,
                "status": "timeout",
            })
            raise
        except Exception as e:
            end = _now_iso()
            self.timeline.append({
                "ts": end,
                "event": "tool_end",
                "call_id": call_id,
                "status": "error",
                "error": str(e),
            })
            raise

    async def execute(self, plan: Plan) -> Dict[str, Any]:
        for step in plan.steps:
            parsed = await self.call_tool_with_trace(step.server, step.tool, step.args, timeout_s=step.timeout_s)
            if step.alias:
                self.ctx[step.alias] = parsed
        return self.ctx


# -------------------------------
# Response builders (same structure as existing frontend expects)
# -------------------------------

def _correlation_modules(user_query: str, ctx: Dict[str, Any], plan: Plan, session_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    modules: List[Dict[str, Any]] = []
    corr = ctx.get("correlations") or {}
    summary = {
        "pairs": corr.get("pairs_count") or corr.get("total_correlations"),
        "unique_facilities": corr.get("unique_facilities"),
        "unique_polygons": corr.get("unique_polygons"),
        "method": corr.get("method"),
        "entity_counts": corr.get("entity_counts"),
    }

    # Text overview
    overview_lines = [
        f"Correlation method: {summary.get('method')}",
        f"Pairs found: {summary.get('pairs')} | Unique facilities: {summary.get('unique_facilities')} | Unique polygons: {summary.get('unique_polygons')}",
    ]
    if plan.notes:
        overview_lines.append("Notes: " + "; ".join(plan.notes))
    modules.append({
        "type": "text",
        "heading": "Spatial Correlation Overview",
        "content": "\n".join(overview_lines),
    })

    # Map (if available)
    corr_map = ctx.get("corr_map") or {}
    geojson_url = corr_map.get("geojson_url")
    if isinstance(geojson_url, str) and geojson_url:
        modules.append({
            "type": "map",
            "mapType": "geojson_url",
            "heading": "Correlation Map",
            "geojson_url": geojson_url,
            "legend": {
                "title": "Correlation Map",
                "items": [
                    {"label": "Solar Assets", "color": "#FFD700"},
                    {"label": "Deforestation Areas", "color": "#8B4513"}
                ]
            },
            "metadata": {
                "session_id": session_id,
                "layers": ["solar_facility", "deforestation_area"],
                "is_correlation_map": True,
                "map_role": "correlation"
            },
        })

    # Table summary (small inline table)
    entity_counts = summary.get("entity_counts") or {}
    if entity_counts:
        rows = [[k, v] for k, v in entity_counts.items()]
        modules.append({
            "type": "table",
            "heading": "Entities in Session",
            "columns": ["Entity Type", "Count"],
            "rows": rows,
        })

    metadata = {
        "modules_count": len(modules),
        "has_maps": any(m.get("type") == "map" for m in modules),
        "has_charts": any(m.get("type") == "chart" for m in modules),
        "has_tables": any(m.get("type") == "table" for m in modules),
        "intent": plan.intent,
    }
    return modules, metadata


# -------------------------------
# Public API (same as mcp_chat_redo)
# -------------------------------

async def process_chat_query(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None, correlation_session_id: Optional[str] = None, target_language: Optional[str] = None) -> Dict:
    """Synchronous variant that returns the final response object."""
    # For consistency with the streaming path, run stream and capture the final event
    async def _drain():
        last_complete = None
        async for ev in stream_chat_query(user_query, conversation_history, correlation_session_id, target_language):
            if isinstance(ev, dict) and ev.get("type") == "complete":
                last_complete = ev.get("data")
        return last_complete or {"query": user_query, "modules": [], "metadata": {"modules_count": 0}}

    return await _drain()


async def stream_chat_query(user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None, correlation_session_id: Optional[str] = None, target_language: Optional[str] = None):
    """Streaming variant that yields thinking events and final response.

    Event shapes mirror mcp_chat_redo: thinking, facts_summary, complete, error.
    """
    query_id = f"q_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
    session_id = correlation_session_id or query_id  # isolate geospatial registrations per query

    # Kickoff message
    yield {"type": "thinking", "data": {"message": "🚀 Initializing analysis...", "category": "initialization", "query_id": query_id}}

    try:
        client = await get_global_client()

        # Plan
        planner = Planner(user_query)
        plan = planner.plan()
        yield {"type": "thinking", "data": {"message": f"🧭 Intent: {plan.intent}", "category": "routing"}}

        # Delegate non-spatial queries to existing orchestrator for now
        if plan.intent == "delegate":
            from .mcp_chat_redo import process_chat_query as _fallback_process
            result = await _fallback_process(user_query, conversation_history, session_id, target_language)
            yield {"type": "complete", "data": result}
            return

        # Execute with timeline
        timeline: List[Dict[str, Any]] = []
        executor = Executor(client, session_id=session_id, timeline=timeline)

        yield {"type": "thinking", "data": {"message": "🔧 Executing plan steps...", "category": "analysis"}}

        try:
            ctx = await executor.execute(plan)
        except asyncio.TimeoutError:
            yield {"type": "error", "data": {"message": "A tool call timed out while executing the plan."}}
            return
        except Exception as e:
            yield {"type": "error", "data": {"message": f"Plan execution failed: {e}"}}
            return

        # Small facts summary: use correlation counts if present
        corr = ctx.get("correlations") or {}
        total_pairs = corr.get("pairs_count") or corr.get("total_correlations") or 0
        yield {"type": "facts_summary", "data": {"phase": 1, "total": int(total_pairs), "new_facts": int(total_pairs)}}

        # Build response modules
        modules, meta = _correlation_modules(user_query, ctx, plan, session_id)

        response_data = {
            "query": user_query,
            "modules": modules,
            "metadata": {
                **meta,
                "modules_count": len(modules),
                "servers_queried": len({s.server for s in plan.steps}),
                "facts_collected": int(total_pairs),
                "decision_timeline": timeline,
            },
        }

        yield {"type": "complete", "data": response_data}

    except Exception as e:
        yield {"type": "error", "data": {"message": f"Unexpected error: {str(e)}"}}
