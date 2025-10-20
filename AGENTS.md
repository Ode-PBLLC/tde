# Repository Guidelines

## Project Structure & Modules
- `api_server.py` — FastAPI app (defaults `PORT=8098`).
- `kg_visualization_server.py` — KG UI/API on `8100`.
- `mcp/` — orchestration, tool servers, response formatting.
- `utils/` — helpers (e.g., `language.py`).
- `static/` — assets, cached responses, generated KG embeds.
- `data/` — datasets; avoid committing large or secret data.
- `test_scripts/` — runnable tests and demos.
- `scripts/`, `deploy/`, `docs/` — ops, deployment, and docs.

## Build, Run, and Test
- Install deps: `pip install -r requirements.txt`
- Run API (auto‑reload): `uvicorn api_server:app --reload --host 0.0.0.0 --port 8099`
  - Alt: `PORT=8098 python api_server.py` (matches `start_servers.sh`).
- Run KG server: `python kg_visualization_server.py` (port `8100`).
- Start/stop both: `./start_servers.sh` / `./stop_servers.sh`.
- Smoke tests:
  - Translation plumbing: `python test_scripts/test_translation.py`
  - Focused responses (async): `python test_scripts/test_focused_responses.py`
  - Solar DB checks: `python test_scripts/test_enhanced_solar_mcp.py`

## Coding Style & Naming
- Follow PEP 8; 4‑space indentation; wrap ~100 cols.
- Use type hints and docstrings for new/changed functions.
- Names: modules/files `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`.
- JSON schema in responses is user‑facing; do not rename module keys without coordination.

## Testing Guidelines
- Prefer small, scriptable checks in `test_scripts/`; keep tests network‑independent.
- Add targeted tests for new utilities under `test_scripts/` (prefix with `test_*.py`).
- Ensure API still serves: `GET /health`, and `POST /query/stream` completes for a trivial prompt.

## Commits & Pull Requests
- Commits: imperative, descriptive, scoped (e.g., "Improve KG map proxy error handling").
- PRs: include summary, rationale, runnable steps (commands), and screenshots for UI/maps.
- Link related issues; update docs if behavior or endpoints change.

## Security & Config
- Secrets via `.env` (see `.env.example`): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.
- Never commit secrets or large raw data; respect `.gitignore`.
- Ports: API `8098` (or `8099` via uvicorn), KG `8100`. Configure with `PORT`/`API_BASE_URL` when deploying.

## Agent‑Specific Notes
- Keep changes minimal and localized; avoid breaking API schemas.
- Do not alter default ports, routes, or response module shapes without explicit approval.
