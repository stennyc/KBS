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


class TestSearchCentralDocKindFilter:
    """门 2 共识面 (owner 定稿 2026-10-22): confirmed_only=true 只放行
    CONSENSUS_DOC_KINDS (reference/material/tutorial), manual/experience 挡在外.
    confirmed_only=false (全量面) 不过滤 — manual/experience 仍可查 (模型主动翻).
    拦在 _PgConn + _central_auth 层注入 fake cur, 验证 SQL 过滤条件."""
    def _run(self, confirmed_only, cur):
        import kb_shared_memory as sm

        class _FakeConn:
            def __init__(self, pg):
                pass
            def cur(self):
                return cur
            def commit(self):
                pass
            def close(self):
                pass
        with mock.patch.object(sm, "_PgConn", _FakeConn), \
             mock.patch.object(sm, "_central_auth", lambda *a, **k: None):
            return sm.search_central(_cfg(), "postgres",
                                     confirmed_only=confirmed_only)

    def test_confirmed_only_filters_doc_kind_in_sql(self):
        sqls = []

        class _Cur:
            def __init__(self):
                self.sqls = []
                self.params_list = []
            def execute(self, sql, params):
                # _central_auth 已 mock (不调 cur.execute) -> 每次 execute 都是 search SQL
                self.sqls.append(sql)
                self.params_list.append(params)
            def fetchone(self):
                return ("local",)
            def fetchall(self):
                return []

        c = _Cur()
        out = self._run(True, c)
        assert out == []
        assert len(c.sqls) == 1
        s = c.sqls[0]
        # 共识面: status='confirmed' + doc_kind IN 白名单
        assert "status = 'confirmed'" in s
        assert "doc_kind IN" in s
        # 参数里带 3 个门类白名单值
        p = c.params_list[0]
        assert "reference" in p and "material" in p and "tutorial" in p

    def test_full_sediment_no_doc_kind_filter(self):
        class _Cur:
            def __init__(self):
                self.sql = ""
                self.params_list = []
            def execute(self, sql, params):
                self.sql = sql
                self.params_list.append(params)
            def fetchone(self):
                return ("local",)
            def fetchall(self):
                return [("d1", "t", "c", "remote", "remote", 1, None, None,
                         "confirmed", "manual", {})]
        c = _Cur()
        out = self._run(False, c)
        s = c.sql
        assert "status = 'confirmed'" not in s
        assert "doc_kind IN" not in s
        # 全量面: manual 查得到 (模型主动翻全量沉淀池)
        assert out[0]["doc_kind"] == "manual"


def test_consensus_doc_kinds_constant():
    import kb_shared_memory as sm
    assert sm.CONSENSUS_DOC_KINDS == ("reference", "material", "tutorial")
