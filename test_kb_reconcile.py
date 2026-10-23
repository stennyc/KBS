"""kb 启动对账腿 (owner 定稿 2026-10-23, 选 A + 基准源 pkl 双快照):
_KbSearchService.__init__ 后模块级哨兵守一次 "本地 hot+cold 全集 vs 中央
kb_docs 差集补推" (幂等, 只推差集, fail-soft, 根除重启/宕机期间
notify_confirmed 没 fire 的缺口). 与 filemem Q1 对账腿同型."""
import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb_shared_memory as sm
import kb_service


def _cfg():
    return {
        "enabled": True, "agent_id": "local",
        "pg": {"host": "127.0.0.1", "port": 5432, "db": "sharedmem",
                "user": "sharedmem", "password": "pw"},
        "token": "tok",
    }


class _FakeCur:
    def __init__(self, central_docids):
        self.central = central_docids
        self.pushed = []

    def execute(self, sql, params=None):
        if sql.startswith("SELECT docid FROM kb_docs"):
            self._rows = [(d,) for d in self.central]
        elif "upsert_kb_doc" in sql:
            self.pushed.append(params[0])  # upsert_kb_doc 第一参 = docid
        else:
            self._rows = []

    def fetchall(self):
        return getattr(self, "_rows", [])

    def fetchone(self):
        r = getattr(self, "_rows", [])
        return r[0] if r else None

    def close(self):
        pass


class _FakeConn:
    def __init__(self, central_docids):
        self._cur = _FakeCur(central_docids)
        self.central_docids = central_docids
        self.pushed = self._cur.pushed

    def cur(self):
        return self._cur

    def commit(self):
        pass

    def close(self):
        pass


def _fake_svc(docs):
    """svc stub: 只需 _hot/_cold.docs_snapshot 全集 (对账腿本地全集来源)."""
    class _Snap:
        def __init__(self, docs):
            self._docs = docs
        def docs_snapshot(self):
            return self._docs
    class _Svc:
        pass
    s = _Svc()
    s._hot = _Snap([d for d in docs if d.get("_layer") == "hot"])
    s._cold = _Snap([d for d in docs if d.get("_layer") != "hot"])
    return s


class TestReconcileLocalToCentral:
    def test_reconcile_pushes_only_missing_docids(self):
        # 本地 hot+cold 共 3 doc, 中央已有 d2 -> 只补推 d1+d3
        svc = _fake_svc([
            {"docid": "d1", "title": "t1", "content": "c", "status": "confirmed",
             "_layer": "hot"},
            {"docid": "d2", "title": "t2", "content": "c", "status": "confirmed"},
            {"docid": "d3", "title": "t3", "content": "c", "status": "confirmed",
             "_layer": "hot"},
        ])
        conn = _FakeConn(central_docids=["d2"])
        with mock.patch.object(sm, "_PgConn", lambda c: conn):
            with mock.patch.object(sm, "_central_auth"):
                with mock.patch.object(sm, "_load_shared_memory_config",
                                       lambda: _cfg()):
                    with mock.patch.object(sm, "_read_local_doc",
                                           lambda s, did: _doc_for(did)):
                        r = sm.reconcile_local_to_central(svc)
        assert r["status"] == "ok"
        assert sorted(conn.pushed) == ["d1", "d3"]
        assert r["missing"] == 2 and r["pushed"] == 2

    def test_reconcile_in_sync_pushes_zero(self):
        svc = _fake_svc([
            {"docid": "d1", "title": "t1", "content": "c", "status": "confirmed"},
        ])
        conn = _FakeConn(central_docids=["d1"])
        with mock.patch.object(sm, "_PgConn", lambda c: conn):
            with mock.patch.object(sm, "_central_auth"):
                with mock.patch.object(sm, "_load_shared_memory_config",
                                       lambda: _cfg()):
                    r = sm.reconcile_local_to_central(svc)
        assert r["status"] == "ok" and r.get("reason") == "in-sync"
        assert conn.pushed == [] and r["missing"] == 0 and r["pushed"] == 0

    def test_reconcile_skips_when_no_config(self):
        with mock.patch.object(sm, "_load_shared_memory_config", lambda: None):
            r = sm.reconcile_local_to_central(_fake_svc([]))
        assert r["status"] == "skipped" and r["pushed"] == 0

    def test_module_sentinel_gates_once_per_host(self):
        """_svc() 每 session 可能重入, 但 host 生命周期只该跑一次对账."""
        svc = _fake_svc([{"docid": "d1"}])
        calls = {"n": 0}

        def _fake_reconcile(svc):
            calls["n"] += 1
            return {"status": "ok", "missing": 0, "pushed": 0}

        orig_reconcile = sm.reconcile_local_to_central
        # 直接测哨兵短路逻辑 (不 import __init__ 避免拉宿主环境)
        sm._svc_reconciled = False
        sm.reconcile_local_to_central = _fake_reconcile
        sm._svc_reconcile_once(svc)
        first = calls["n"]
        sm._svc_reconcile_once(svc)  # 哨兵已 True, 短路
        second = calls["n"]
        assert second == first, f"host 生命周期第二次不该再对账: {first} -> {second}"
        # 还原
        sm.reconcile_local_to_central = orig_reconcile


def _doc_for(did):
    return {"docid": did, "title": f"t-{did}", "content": "c",
            "status": "confirmed", "version": 1}
