"""Version 2 MCP server helpers.

Servers that comply with the new `run_query` contract can import shared
base classes from this package.  We provide a small mixin that takes care
of serialising responses into the JSON envelope expected by the
orchestrator.
"""

from .base import RunQueryMixin

__all__ = ["RunQueryMixin"]

