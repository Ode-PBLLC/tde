#!/usr/bin/env python3
"""
Record Featured Query Streams

This script generates JSONL cache files for featured queries by:
1. Reading featured queries from static/featured_queries.json
2. Calling the streaming API endpoint for each query
3. Recording the full SSE event stream to JSONL files

Usage:
    python scripts/record_featured_streams.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import aiohttp
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stream_cache_manager import StreamCache

# Configuration
DEFAULT_API_URL = "http://localhost:8098"
FEATURED_QUERIES_PATH = Path("static/featured_queries.json")
TIMEOUT_SECONDS = 600  # 10 minutes max per query


async def parse_sse_stream(response: aiohttp.ClientResponse) -> List[Dict[str, Any]]:
    """Parse SSE stream from HTTP response and return list of events."""
    events = []

    try:
        async for line in response.content:
            line = line.decode('utf-8').strip()

            # SSE format: "data: {json}\n\n"
            if line.startswith('data: '):
                data_str = line[6:]  # Remove "data: " prefix

                try:
                    event = json.loads(data_str)
                    events.append(event)

                    # Check for completion
                    if event.get("type") == "done":
                        break

                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse SSE event: {e}")
                    continue

    except Exception as e:
        print(f"  Error reading SSE stream: {e}")
        raise

    return events


async def record_query_stream(
    session: aiohttp.ClientSession,
    api_url: str,
    query_id: str,
    query_text: str,
    cache_manager: StreamCache,
    force: bool = False
) -> tuple[bool, str]:
    """
    Record a single query stream to cache.

    Returns:
        (success: bool, message: str)
    """
    # Check if cache already exists
    if not force and await cache_manager.exists(query_id):
        return True, "Cache already exists (use --force to overwrite)"

    print(f"  📡 Sending query to API...")

    url = f"{api_url}/query/stream"
    payload = {"query": query_text, "reset": False}

    start_time = time.time()

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                return False, f"API returned status {response.status}: {error_text}"

            # Parse SSE stream
            print(f"  📥 Receiving events...")
            events = await parse_sse_stream(response)

            if not events:
                return False, "No events received from API"

            print(f"  💾 Saving {len(events)} events to cache...")

            # Save to cache file with timestamps
            cache_file = cache_manager.get_cache_path(query_id)
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            timestamp_ms = 0
            with open(cache_file, 'w', encoding='utf-8') as f:
                for i, event in enumerate(events):
                    event_with_timing = event.copy()

                    # Add timestamps based on event type (simulate realistic timing)
                    event_type = event.get("type", "")

                    if event_type == "thinking":
                        timestamp_ms += 100  # 100ms between thinking events
                    elif event_type == "tool_call":
                        timestamp_ms += 200  # 200ms for tool calls
                    elif event_type == "tool_result":
                        timestamp_ms += 500  # 500ms for tool results
                    elif event_type == "content":
                        timestamp_ms += 50   # 50ms between content chunks
                    elif event_type == "complete":
                        timestamp_ms += 300  # 300ms before complete
                    else:
                        timestamp_ms += 150  # Default 150ms

                    event_with_timing["timestamp_ms"] = timestamp_ms
                    event_with_timing["recorded_at"] = datetime.now().isoformat()

                    f.write(json.dumps(event_with_timing, ensure_ascii=False) + '\n')

            elapsed = time.time() - start_time
            file_size = cache_file.stat().st_size

            print(f"  ✅ Cached {len(events)} events ({file_size:,} bytes) in {elapsed:.1f}s")
            return True, "Success"

    except asyncio.TimeoutError:
        return False, f"Request timed out after {TIMEOUT_SECONDS} seconds"
    except Exception as e:
        return False, f"Error: {str(e)}"


async def main():
    print("\n🎙️  Featured Query Stream Recorder")
    print("=" * 80)

    # Featured queries to cache (hardcoded for exact matching)
    queries = [
        {
            "id": "brazil-solar-expansion",
            "title": "Brazil Solar Expansion",
            "query": "How fast is Brazil's solar energy capacity expanding, and how does this align with national climate targets and land-use priorities?"
        },
        {
            "id": "brazil-climate-goals",
            "title": "Brazil Climate Goals",
            "query": "How is Brazil currently performing on its national climate goals, and which policies are driving progress or falling behind?"
        },
        {
            "id": "brazil-climate-risks",
            "title": "Brazil Climate Risks",
            "query": "What are the main climate risks facing different regions of Brazil this decade, and which ones are most urgent for policymakers to address?"
        }
    ]

    print(f"\n📋 Recording {len(queries)} featured queries:")
    for q in queries:
        print(f"   • {q.get('title', q.get('id'))}")

    api_url = DEFAULT_API_URL
    print(f"\n🔗 API URL: {api_url}")

    # Check if API is running
    print(f"🔍 Checking API availability...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{api_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    print(f"✅ API is running\n")
                else:
                    print(f"❌ API returned status {response.status}")
                    print(f"   Make sure the API server is running: python api_server.py")
                    sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"   Make sure the API server is running: python api_server.py")
        sys.exit(1)

    # Initialize cache manager
    cache_manager = StreamCache()

    # Process queries
    results = []
    async with aiohttp.ClientSession() as session:
        for i, query_data in enumerate(queries, 1):
            query_id = query_data.get("id")
            query_text = query_data.get("query")
            title = query_data.get("title", query_id)

            print(f"\n[{i}/{len(queries)}] 🔹 {title}")
            print(f"{'─' * 80}")
            print(f"  ID: {query_id}")

            success, message = await record_query_stream(
                session,
                api_url,
                query_id,
                query_text,
                cache_manager
            )

            results.append({
                "query_id": query_id,
                "title": title,
                "success": success,
                "message": message
            })

            # Small delay between queries
            if i < len(queries):
                await asyncio.sleep(2)

    # Summary
    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n  Total queries: {len(results)}")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")

    if failed > 0:
        print(f"\n  Failed queries:")
        for result in results:
            if not result["success"]:
                print(f"    • {result['query_id']}: {result['message']}")

    # Show cache location
    cache_dir = cache_manager.cache_dir
    print(f"\n  📁 Cache location: {cache_dir.absolute()}")

    print("\n" + "=" * 80)
    print("✅ Recording complete!\n")

    if successful > 0:
        print("Next steps:")
        print("  1. The cache is now active - no need to restart the server")
        print("  2. Test by sending one of the featured queries")
        print("  3. Check server logs for '[StreamCache] Cache hit!' messages\n")


if __name__ == "__main__":
    asyncio.run(main())
