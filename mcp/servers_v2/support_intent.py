"""Common helpers for LLM-backed query_support decisions."""
from dataclasses import dataclass
from typing import List


@dataclass
class SupportIntent:
    """LLM-backed routing decision for a dataset."""

    supported: bool
    score: float
    reasons: List[str]
