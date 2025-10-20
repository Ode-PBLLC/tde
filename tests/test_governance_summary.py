from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mcp.mcp_chat_v2 as mcp_chat_v2


def test_should_inject_governance_only_for_in_scope() -> None:
    assert mcp_chat_v2._should_inject_governance("IN_SCOPE", True)
    assert not mcp_chat_v2._should_inject_governance("NEAR_SCOPE", True)
    assert not mcp_chat_v2._should_inject_governance("IN_SCOPE", False)


def test_should_inject_governance_respects_flag(monkeypatch) -> None:
    monkeypatch.setattr(mcp_chat_v2, "ENABLE_GOVERNANCE_SUMMARY", False)
    assert not mcp_chat_v2._should_inject_governance("IN_SCOPE", True)


def test_build_governance_followup_query_includes_context() -> None:
    query = "How is Brazil governing solar expansion?"
    paragraphs = [
        "Brazil has expanded solar auctions at the federal level.",
        "Bahia and Ceará manage large state projects.",
    ]

    followup = mcp_chat_v2._build_governance_followup_query(query, paragraphs)

    assert query in followup
    for paragraph in paragraphs:
        assert paragraph in followup
    assert "governance" in followup.lower()
