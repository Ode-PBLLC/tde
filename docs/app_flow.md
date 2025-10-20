# Application Flow

This document outlines how the application works at two levels: a general, high‑level story (the "what") and a more detailed, technical view (the "how").

## General Flow (What happens)

We use AI to surface data that is immediately relevant to a user query.

1. User asks a question
   - A user provides a natural‑language prompt (e.g., “Where are the largest solar facilities near deforestation risk?”).

2. Understand and focus the intent
   - The system interprets the question, identifies what’s being asked, and narrows to the most relevant data sources and views.

3. Fetch the right data
   - Specialized data “servers” (for solar facilities, deforestation, geospatial, etc.) provide the needed facts without the user having to know where they live.

4. Assemble a clear answer
   - The system organizes results into easy‑to‑read modules: short explanations, tables, maps, and charts with source citations.

5. Show, cite, and iterate
   - Answers include links, sources, and optional visualizations (e.g., a map). The user can refine the question, and the system repeats the focus → fetch → assemble loop quickly.

Result: fast, targeted insights with just enough context for decision‑making, backed by transparent data sources.

---

## Technical Flow (How it works)

Key components
- `api_server.py` (FastAPI): entrypoint for requests such as `POST /query/stream` and `GET /health`.
- MCP servers (`mcp/*.py`): tool‑style servers that expose structured functions for domain data (e.g., solar, deforestation, geospatial, GIST, response formatting, meta).
- Response formatting: `mcp/response_formatter_server.py` generates user‑facing modules (text, tables, charts, maps) and manages citation bookkeeping.
- KG UI/API: `kg_visualization_server.py` serves knowledge‑graph content and embeds.
- Utilities and data: helpers in `utils/`, datasets under `data/` (no secrets or large raw data in git), static artifacts under `static/`.

End‑to‑end request path
1. Request in
   - Client calls `POST /query/stream` on the FastAPI app (`api_server.py`). The server logs context and starts streaming events.

2. Language handling and planning
   - Translation/normalization (see `mcp/translation.py`) ensures prompts are processable.
   - Orchestration logic (see `mcp/mcp_chat_redo.py` and related) analyzes the prompt, plans which MCP tools to call, and in what order.

3. Tool selection and execution (MCP)
   - The orchestrator invokes MCP tools like:
     - `solar_facilities_server.py` for facility lookups and summaries.
     - `deforestation_server.py` for polygons and stats; can persist map GeoJSON to `static/maps/`.
     - `geospatial_server.py` for spatial joins, bounds, buffering.
     - `gist_server.py` for sustainability datasets (if available locally).
     - `meta_server.py` for project metadata (repo, datasets, orgs, links).
   - Tools return structured JSON payloads (counts, records, geojson paths, etc.).

4. Aggregation and shaping
   - Orchestrator merges tool outputs, resolves conflicts, and prepares context for formatting.
   - Citations and provenance are collected into a registry keyed by module/tool outputs.

5. Response formatting and streaming
   - `response_formatter_server.py` converts results into modules:
     - Text summaries with inline citations.
     - Tables and charts with consistent IDs.
     - Map references to artifacts under `static/maps/`.
   - `api_server.py` streams the assembled modules back to the client as they’re ready.

6. Optional KG views
   - `kg_visualization_server.py` can provide graph views or entity pages linked from the answer.

Operational notes
- Ports: API `8098` (or `8099` via uvicorn), KG `8100`; configurable via env where deployed.
- Secrets: use `.env` (see `.env.example`) and never surface via meta or logs.
- Testing: smoke scripts in `test_scripts/` validate translation, focused responses, and MCP data paths.
- Caching/artifacts: generated assets (e.g., GeoJSON maps) are written under `static/` for reuse and UI embedding.

Extensibility
- Add a new domain MCP server under `mcp/` with `FastMCP` and expose typed tool functions.
- Register/use the new tools in orchestration so the planner can discover them.
- Update `static/meta/*.json` and the Meta MCP server (`mcp/meta_server.py`) to advertise new datasets/links.

This flow keeps “what the user sees” focused and trustworthy, while the “how” remains modular and testable.

