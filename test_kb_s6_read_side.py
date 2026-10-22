"""S6 kb read-side: kb_search_central tool (read-only central query) +
referral hint on the kb pre_llm_call auto-retrieve face (local hits < 2
-> suggest the tool, never auto-inject central data).

Symmetric to the filemem read-side. Owner rulings (2026-10-22):
- D1 = (b) pointer-only hint; D2 = results stay in this turn's tool
  return, never re-landed locally; fixed threshold 2.
- kb auto-retrieve is a pre_llm_call hook (kb_ingest_hook), NOT a
  provider.prefetch — the hint rides the returned {"context": ...} block.
"""
import json
import sys
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from __init__ import KB_SEARCH_CENTRAL_SCHEMA, _kb_search_central_handler


def _cfg() -> dict:
    return {
        "enabled": True, "agent_id": "local",
        "pg": {"host": "127.0.0.1", "port": 5432, "db": "sharedmem",
                "user": "sharedmem", "password": "pw"},
        "token": "tok",
    }


class _FakeSmModule:
    """Test double standing in for kb_shared_memory (injectable via kwargs)."""
    cfg_value = None
    search_result = []

    @staticmethod
    def _load_shared_memory_config():
        return _FakeSmModule.cfg_value

    @staticmethod
    def search_central(cfg, query, confirmed_only=True, limit=50):
        return _FakeSmModule.search_result


def _fake_sm(cfg, result=None):
    _FakeSmModule.cfg_value = cfg
    _FakeSmModule.search_result = result if result is not None else []
    return _FakeSmModule


class TestKbSearchCentralHandler:
    def test_not_enabled(self):
        # sm module present but config None -> clean "not enabled" error.
        sm = _fake_sm(None)
        out = json.loads(_kb_search_central_handler(
            {"query": "postgres"}, sm_module=sm))
        assert out["success"] is False
        assert ("enabled" in out.get("error", "").lower()
                or "未启用" in out.get("error", ""))

    def test_delegates_and_returns_json(self):
        sm = _fake_sm(_cfg(), result=[
            {"docid": "d1", "title": "T", "content": "C",
             "status": "confirmed", "agent_id": "remote",
             "updated_at": None, "source_profile": "remote"}])
        out = json.loads(_kb_search_central_handler(
            {"query": "postgres", "confirmed_only": True}, sm_module=sm))
        assert out["success"] is True
        assert out["results"][0]["agent_id"] == "remote"
        assert out["count"] == 1

    def test_empty_query_errors(self):
        sm = _fake_sm(_cfg())
        out = json.loads(_kb_search_central_handler(
            {"query": "  "}, sm_module=sm))
        assert out["success"] is False


def test_schema_has_expected_shape():
    assert KB_SEARCH_CENTRAL_SCHEMA["name"] == "kb_search_central"
    props = KB_SEARCH_CENTRAL_SCHEMA["parameters"]["properties"]
    assert "query" in props
    assert "confirmed_only" in props


class TestKbReferralHint:
    """The kb auto-retrieve referral hint (kb_ingest_hook module level)."""
    def test_hint_disabled_is_empty(self):
        from kb_ingest_hook import kb_referral_hint
        assert kb_referral_hint(0, enabled=False) == ""
        assert kb_referral_hint(5, enabled=True) == ""
        assert kb_referral_hint(1, enabled=True) != ""
        assert "kb_search_central" in kb_referral_hint(1, enabled=True)
