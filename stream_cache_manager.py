#!/usr/bin/env python3
"""
Stream Cache Manager - Records and replays SSE event streams for featured queries.

This module provides caching for the full streaming experience, preserving timing
and all events (thinking, tool calls, results, content, etc.)
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, AsyncGenerator, Dict, Any, List
from datetime import datetime
import time


STREAM_CACHE_DIR = Path("static/stream_cache")


class StreamCache:
    """Manages recording and replay of SSE event streams."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or STREAM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, query_id: str) -> Path:
        """Get the cache file path for a query ID."""
        return self.cache_dir / f"{query_id}.jsonl"

    async def exists(self, query_id: str) -> bool:
        """Check if a cache exists for a query ID."""
        return self.get_cache_path(query_id).exists()

    async def get_cached_stream(self, query_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load cached event stream if it exists.

        Returns:
            List of events with timing information, or None if cache doesn't exist
        """
        cache_file = self.get_cache_path(query_id)

        if not cache_file.exists():
            return None

        events = []
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError:
                        continue

            return events

        except Exception:
            return None

    async def replay_stream(
        self,
        events: List[Dict[str, Any]],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Replay cached events with original timing.

        Args:
            events: List of cached events with timestamp_ms

        Yields:
            Events one at a time with appropriate delays
        """
        if not events:
            return

        for i, event in enumerate(events):
            # Calculate delay based on original timing
            if i > 0:
                prev_timestamp = events[i-1].get("timestamp_ms", 0)
                curr_timestamp = event.get("timestamp_ms", 0)
                delay_ms = curr_timestamp - prev_timestamp

                delay_sec = delay_ms / 1000.0

                # Cap delays to avoid long waits (max 3 seconds)
                delay_sec = min(delay_sec, 3.0)

                if delay_sec > 0:
                    await asyncio.sleep(delay_sec)

            # Yield the event data without the timing metadata
            event_copy = event.copy()
            event_copy.pop("timestamp_ms", None)
            event_copy.pop("recorded_at", None)

            yield event_copy

    async def record_stream(
        self,
        query_id: str,
        events_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Record streaming events to cache file while passing them through.

        Args:
            query_id: Unique identifier for the query
            events_stream: Async generator of events to record

        Yields:
            Events are passed through unchanged
        """
        cache_file = self.get_cache_path(query_id)
        start_time = time.time()

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                async for event in events_stream:
                    # Add timing information
                    event_with_timing = event.copy()
                    event_with_timing["timestamp_ms"] = int((time.time() - start_time) * 1000)
                    event_with_timing["recorded_at"] = datetime.now().isoformat()

                    # Write to cache file (one JSON object per line)
                    f.write(json.dumps(event_with_timing, ensure_ascii=False) + '\n')
                    f.flush()  # Ensure it's written immediately

                    # Pass through the original event (without timing metadata)
                    yield event

        except Exception as e:
            # Clean up partial cache file
            if cache_file.exists():
                cache_file.unlink()
            raise


# Convenience function for testing
async def test_cache_replay(query_id: str):
    """Test replaying a cached stream."""
    cache = StreamCache()
    events = await cache.get_cached_stream(query_id)

    if not events:
        print(f"No cache found for: {query_id}")
        return

    print(f"\n🎬 Replaying cached stream for: {query_id}")
    print("=" * 60)

    async for event in cache.replay_stream(events):
        event_type = event.get("type", "unknown")
        print(f"[{event_type}]", end=" ")

        if event_type == "thinking":
            msg = event.get("data", {}).get("message", "")
            if msg:
                print(msg[:80])
        elif event_type == "complete":
            print("✓ Complete")
        else:
            print()

    print("\n" + "=" * 60)
    print("✅ Replay finished")


if __name__ == "__main__":
    # Simple CLI for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python stream_cache_manager.py replay <query_id>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "replay" and len(sys.argv) > 2:
        query_id = sys.argv[2]
        asyncio.run(test_cache_replay(query_id))
    else:
        print(f"Unknown command: {command}")
