# FastMCP Latency-Reduction Guide  
**Objective:** keep a single `fastmcp.Client` alive inside every FastAPI worker so each user query skips the MCP handshake and subprocess cold-start.

---

## Why this matters
* **Stdio transport spins up a fresh Python subprocess** for every new `Client` context; the child process builds the manifest and parses JSON schemas. That costs **≈ 250-700 ms** per launch.  
* The transport already supports **`keep_alive=True` by default** – we only need to stop recreating the client. :contentReference[oaicite:0]{index=0}  

---

## Tactical checklist

| ✔︎ | Task |
|---|------|
| ☐ **Create a module-level `client` singleton** in `mcp_chat.py`. |
| ☐ **Replace per-request instantiation** with direct `invoke`/`stream` calls that use the singleton. |
| ☐ **Add FastAPI startup/shutdown hooks** in `api_server.py` to open (warm) and close the client once per worker lifecycle. |
| ☐ *(Optional)* Call a cheap tool (e.g. `_ping`) during startup to finish warm-up before the first external request. |
| ☐ *(Optional)* Drop `DetailedTimingMiddleware` into one MCP server to confirm handshake time disappears. |
| ☐ *(Later)* Switch servers to HTTP/SSE transport for another ≈10 ms gain. |

---

## Code patches

### 1 · `mcp_chat.py`

```diff
-from fastmcp import Client
-import asyncio
-
-async def run_query(text: str) -> dict:
-    client = Client("./server.py")           # ❌ subprocess per call
-    async with client:
-        return await client.invoke("Ask", {"query": text})
-
-async def run_query_streaming(text: str):
-    client = Client("./server.py")
-    async with client:
-        async for chunk in client.stream("AskStream", {"query": text}):
-            yield chunk
+"""Shared FastMCP client — one per worker."""
+from fastmcp import Client
+
+# ① create once; keep_alive is True by default for stdio transport
+client = Client("./server.py")
+
+async def run_query(text: str) -> dict:
+    return await client.invoke("Ask", {"query": text})
+
+async def run_query_streaming(text: str):
+    async for chunk in client.stream("AskStream", {"query": text}):
+        yield chunk
2 · api_server.py
Add immediately after imports:

from mcp_chat import client  # singleton from step 1

@app.on_event("startup")
async def _warm_mcp() -> None:
    # ② one-time handshake + manifest fetch
    await client.__aenter__()
    # Optional: await client.invoke("_ping", {})  # pre-warm tool

@app.on_event("shutdown")
async def _close_mcp() -> None:
    await client.__aexit__(None, None, None)
Nothing else in your endpoints needs to change.

Validation

Local test (single worker)
uvicorn api_server:app --reload
First request: still ≈300 ms (warm-up).
Every later request: latency drop (handshake cost gone).
Multi-worker test
uvicorn api_server:app --workers 4
You’ll see four warm-ups (one per worker) and then consistently low latency.
Timing middleware (optional)
from fastmcp.server import DetailedTimingMiddleware
mcp_app.add_middleware(DetailedTimingMiddleware)
Logs should show handshake=0ms after warm-up.
Optional future upgrade — HTTP/SSE transport

Run each MCP server once
python server.py --http 9001
Point the client at the new endpoint
import httpx
client = Client(
    "http://localhost:9001/mcp",
    transport_kwargs={"client": httpx.AsyncClient(http2=True)}
)
Keep the same startup/shutdown hooks.
Median per-call overhead typically falls to < 15 ms (network + tool run only). 
github.com
Done
Ship these two patches; latency improvement ≈ 10× for every request after each worker’s initial warm-up.

::contentReference[oaicite:2]{index=2}
