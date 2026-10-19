"""TR-1/TR-5/TR-8 unit tests: agent experience-ledger fields,
premise_status migrations, and read-side default profile-scoping.

TR-1: origin_kind="agent" passthrough + the four ledger fields
     (topic/task_type/premise/premise_status) defaulted via setdefault.
TR-5: set_premise_status() legal/illegal transitions + the uuid5
     double-row invariant (only premise_status/premise keys are mutated).
TR-8: _search_handler injects source_profile=<caller> when the caller
     has no explicit source_profile filter; explicit source_profile=None
     is the intentional cross-profile opt-in (R-7).

Run: /root/.hermes/plugin-venv/bin/python -m pytest test_kb_agent_experience.py -q
(needs the shared plugin venv on PYTHONPATH for rank_bm25 + jieba)
"""

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import kb_service
from kb_service import KbSearchService, _normalize_docs, _deterministic_docid

# Self-contained plugin dir: this test file and the service layer
# (kb_service.py, hot_layer.py, ...) all live in the SAME directory
# (single source of truth — .hermes/plugins/knowledge-base-search/).
PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PLUGIN_DIR))
_PLUGIN_INIT = PLUGIN_DIR / "__init__.py"


def _svc_with_audit() -> tuple:
    base = Path(tempfile.mkdtemp())
    data = base / "kb-data"
    data.mkdir()
    audit = base / "audit.jsonl"
    svc = KbSearchService(data, audit_path=audit)
    return svc, base, data, audit


def _audit_lines(path: Path) -> list:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


class TR1Tests(unittest.TestCase):

    # ---- origin_kind="agent" passthrough (no whitelist — free string) ----

    def test_origin_kind_agent_passes_to_confirmed_corpus(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{
            "docid": "ag-1", "title": "一条 agent 经验 记录",
            "content": "这是 agent 自己学到的 边角经验 甲乙丙丁戊己庚辛",
            "status": "confirmed", "origin_kind": "agent",
            "source_profile": "coder",
        }])
        self.assertEqual(out["ingested"], ["ag-1"])
        # The doc must be in the hot corpus with origin_kind="agent".
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        found = [d for d in hot_docs if str(d.get("docid")) == "ag-1"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].get("origin_kind"), "agent")

    def test_origin_kind_default_still_auto_when_omitted(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{
            "docid": "no-origin", "title": "默认 origin 测试",
            "content": "没有 origin_kind 的文档 默认 走 auto 流程 甲乙丙丁",
            "status": "confirmed",
        }])
        self.assertEqual(out["ingested"], ["no-origin"])
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        found = [d for d in hot_docs if str(d.get("docid")) == "no-origin"]
        self.assertEqual(found[0].get("origin_kind"), "auto")

    def test_origin_kind_user_still_works(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{
            "docid": "u-1", "title": "用户直接录入",
            "content": "用户 手动 录入的 知识 条目 甲乙丙丁戊己甲甲",
            "status": "confirmed", "origin_kind": "user",
        }])
        self.assertEqual(out["ingested"], ["u-1"])

    # ---- four ledger fields: setdefault fallback --------------------------

    def test_four_fields_defaulted_via_setdefault(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{
            "docid": "fld-1", "title": "默认 字段 兜底 测试",
            "content": "不传 topic task_type premise premise_status 四个字段 甲乙丙丁戊己",
            "status": "confirmed", "origin_kind": "agent",
        }])
        self.assertEqual(out["ingested"], ["fld-1"])
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        found = [d for d in hot_docs if str(d.get("docid")) == "fld-1"][0]
        self.assertEqual(found.get("topic"), "general")
        # task_type / premise are absent (not None, not the key itself)
        self.assertNotIn("task_type", found)
        self.assertNotIn("premise", found)
        # premise_status defaults to True (the bool the setdefault stamps).
        # NOTE: the default is the bool True, not the string "true" — and
        # search() filters are EXACT-match, so the string "true" does NOT
        # match the default bool. Filtering the default-state docs works
        # via the bool, or via omitting the filter (all docs in the
        # corpus match their own stored value).
        self.assertIs(found.get("premise_status"), True)
        # The bool default IS a valid exact-match filter value.
        res = svc.search("默认 字段 兜底", top_k=5,
                         filters={"premise_status": True})
        self.assertIn("fld-1", [r["doc_id"] for r in res["data"]["results"]])

    def test_four_fields_explicit_values_preserved(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{
            "docid": "fld-2", "title": "显式 字段 测试",
            "content": "传了全部 四个 字段 的 agent 经验 甲乙丙丁戊己庚辛",
            "status": "confirmed", "origin_kind": "agent",
            "topic": "messaging", "task_type": "incident-fix",
            "premise": "Kafka broker 在 3.6.1 版本上",
            "premise_status": "needs_verification",
        }])
        self.assertEqual(out["ingested"], ["fld-2"])
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        found = [d for d in hot_docs if str(d.get("docid")) == "fld-2"][0]
        self.assertEqual(found.get("topic"), "messaging")
        self.assertEqual(found.get("task_type"), "incident-fix")
        self.assertEqual(found.get("premise"), "Kafka broker 在 3.6.1 版本上")
        self.assertEqual(found.get("premise_status"), "needs_verification")

    def test_four_fields_flow_into_search_metadata(self):
        """The four ledger fields travel through the hot/cold doc-dict
        passthrough into the search payload's metadata dict."""
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([{
            "docid": "meta-1", "title": "元数据 通道 测试",
            "content": "topic 和 premise_status 要能 被 filters 键 检索到 甲乙丙丁",
            "status": "confirmed", "origin_kind": "agent",
            "topic": "deployment", "premise_status": "falsified",
        }])
        # premise_status as a filter key: the doc must be findable.
        res = svc.search("元数据 通道", top_k=5,
                         filters={"premise_status": "falsified"})
        ids = [r["doc_id"] for r in res["data"]["results"]]
        self.assertIn("meta-1", ids)

        # topic as a filter key.
        res2 = svc.search("元数据 通道", top_k=5, filters={"topic": "deployment"})
        ids2 = [r["doc_id"] for r in res2["data"]["results"]]
        self.assertIn("meta-1", ids2)

        # The metadata dict in the payload carries all four fields.
        hit = next(r for r in res["data"]["results"] if r["doc_id"] == "meta-1")
        self.assertEqual(hit["metadata"].get("topic"), "deployment")
        self.assertEqual(hit["metadata"].get("premise_status"), "falsified")
        self.assertEqual(hit["metadata"].get("origin_kind"), "agent")

    def test_premise_status_filter_excludes_non_matching(self):
        """A filter on premise_status must EXCLUDE docs with a different value."""
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([
            {"docid": "ps-true", "title": "前提 成立 测试",
             "content": "premise_status 默认 是 true 的 文档 甲乙丙丁戊己",
             "status": "confirmed", "origin_kind": "agent"},
            {"docid": "ps-fals", "title": "前提 证伪 测试",
             "content": "premise_status 被 证伪 的 文档 甲乙丙丁戊己庚辛壬",
             "status": "confirmed", "origin_kind": "agent",
             "premise_status": "falsified"},
        ])
        res = svc.search("前提 测试", top_k=5, filters={"premise_status": "falsified"})
        ids = [r["doc_id"] for r in res["data"]["results"]]
        self.assertIn("ps-fals", ids)
        self.assertNotIn("ps-true", ids,
                         "a filter on premise_status must exclude non-matching docs")


class TR5Tests(unittest.TestCase):

    def _make_confirmed_doc(self, svc, docid, premise_status=None, premise=None):
        doc = {
            "docid": docid, "title": f"经验条目 {docid}",
            "content": f"这是一条已确认的 agent 经验 编号 {docid} 甲乙丙丁戊己",
            "status": "confirmed", "origin_kind": "agent",
            "source_profile": "coder", "topic": "general",
        }
        if premise_status is not None:
            doc["premise_status"] = premise_status
        if premise is not None:
            doc["premise"] = premise
        svc.ingest([doc])
        return doc

    def test_set_premise_status_legal_true_to_needs_verification(self):
        svc, base, data, audit = _svc_with_audit()
        self._make_confirmed_doc(svc, "m-1")  # default premise_status=True
        out = svc.set_premise_status("m-1", "needs_verification")
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["from_status"], "true")
        self.assertEqual(out["to"], "needs_verification")
        # Verify in the hot corpus.
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        found = [d for d in hot_docs if str(d.get("docid")) == "m-1"][0]
        self.assertEqual(found.get("premise_status"), "needs_verification")

    def test_set_premise_status_legal_nv_to_falsified(self):
        svc, base, data, audit = _svc_with_audit()
        self._make_confirmed_doc(svc, "m-2", premise_status="needs_verification")
        out = svc.set_premise_status("m-2", "falsified", premise_text="旧前提")
        self.assertEqual(out["status"], "success")
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        found = [d for d in hot_docs if str(d.get("docid")) == "m-2"][0]
        self.assertEqual(found.get("premise_status"), "falsified")
        self.assertEqual(found.get("premise"), "旧前提")

    def test_set_premise_status_legal_falsified_to_nv(self):
        svc, base, data, audit = _svc_with_audit()
        self._make_confirmed_doc(svc, "m-3", premise_status="falsified")
        out = svc.set_premise_status("m-3", "needs_verification")
        self.assertEqual(out["status"], "success")

    def test_set_premise_status_legal_falsified_to_true(self):
        svc, base, data, audit = _svc_with_audit()
        self._make_confirmed_doc(svc, "m-4", premise_status="falsified")
        out = svc.set_premise_status("m-4", True)
        self.assertEqual(out["status"], "success")
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        found = [d for d in hot_docs if str(d.get("docid")) == "m-4"][0]
        # The stored value is the canonical "true" string; filters are
        # exact-match, so the filter must use the same string.
        self.assertEqual(found.get("premise_status"), "true")
        # And the "true" filter now matches the restored doc.
        res = svc.search("经验条目", top_k=5,
                         filters={"premise_status": "true"})
        self.assertIn("m-4", [r["doc_id"] for r in res["data"]["results"]])

    def test_set_premise_status_illegal_true_to_falsified_rejected(self):
        """true -> falsified is NOT a legal transition (must go through
        needs_verification first)."""
        svc, base, data, audit = _svc_with_audit()
        self._make_confirmed_doc(svc, "m-illegal")  # premise_status=True
        out = svc.set_premise_status("m-illegal", "falsified")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "illegal_transition")
        self.assertEqual(out["from_status"], "true")
        self.assertEqual(out["allowed"], ["needs_verification"])
        # The doc is unchanged in the corpus.
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        found = [d for d in hot_docs if str(d.get("docid")) == "m-illegal"][0]
        self.assertIs(found.get("premise_status"), True)
        # Audit record exists.
        rej = [l for l in _audit_lines(audit)
               if l.get("action") == "premise_status_change_rejected"
               and l.get("doc_id") == "m-illegal"]
        self.assertTrue(rej, "illegal transition must be audited")
        # The rejected audit record carries from/to + reason (留痕).
        self.assertEqual(rej[0].get("from_status"), "true")
        self.assertEqual(rej[0].get("to"), "falsified")
        self.assertIn("illegal transition", str(rej[0].get("reason", "")))

    def test_set_premise_status_invalid_status_rejected(self):
        svc, base, data, audit = _svc_with_audit()
        self._make_confirmed_doc(svc, "m-bad")
        out = svc.set_premise_status("m-bad", "bogus")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_status")

    def test_set_premise_status_unknown_doc_not_found(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.set_premise_status("never-existed", "needs_verification")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "not_found")

    def test_set_premise_status_audit_success_record(self):
        svc, base, data, audit = _svc_with_audit()
        self._make_confirmed_doc(svc, "m-audit")
        svc.set_premise_status("m-audit", "needs_verification")
        success = [l for l in _audit_lines(audit)
                   if l.get("action") == "premise_status_change"
                   and l.get("doc_id") == "m-audit"
                   and l.get("from_status") == "true"
                   and l.get("to") == "needs_verification"]
        self.assertTrue(success, "legal transition must be audited")

    def test_uuid5_double_row_invariant(self):
        """TR-5 INVARIANT: set_premise_status mutates the doc dict IN PLACE
        (same docid), never re-mints the uuid5. After the migration the
        docid is unchanged and there is exactly ONE row in the corpus."""
        svc, base, data, audit = _svc_with_audit()
        doc = self._make_confirmed_doc(svc, "inv-1")
        before = _deterministic_docid({k: v for k, v in doc.items() if k != "docid"})
        # The docid is explicit ("inv-1"), so the uuid5 double-row check is:
        # after set_premise_status, the corpus has EXACTLY ONE row with docid="inv-1"
        # and the row's dict is the SAME dict object (mutated in place).
        out = svc.set_premise_status("inv-1", "needs_verification",
                                     premise_text="新前提")
        self.assertEqual(out["status"], "success")
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        rows = [d for d in hot_docs if str(d.get("docid")) == "inv-1"]
        self.assertEqual(len(rows), 1,
                         "in-place mutation must produce exactly ONE row "
                         "with the same docid (no double-row)")
        self.assertEqual(rows[0].get("premise_status"), "needs_verification")
        self.assertEqual(rows[0].get("premise"), "新前提")

    def test_uuid5_new_docid_would_break_the_invariant(self):
        """Negative check: if someone re-runs _normalize_docs on the
        mutated dict and re-ingests with a NEW docid, they create a
        double-row. This test verifies the CORRECT path (in-place) does
        NOT trigger that, by checking the corpus has no second row."""
        svc, base, data, audit = _svc_with_audit()
        self._make_confirmed_doc(svc, "inv-2")
        svc.set_premise_status("inv-2", "falsified")
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        rows = [d for d in hot_docs if str(d.get("docid")) == "inv-2"]
        self.assertEqual(len(rows), 1)
        # Now the WRONG way (what the invariant forbids): re-normalize the
        # mutated dict and re-ingest it — _normalize_docs mints a NEW
        # deterministic docid for the "missing-docid" path. Verify that this
        # would have produced a DIFFERENT docid (i.e. the in-place path is
        # the only safe one).
        from kb_service import _normalize_docs, _deterministic_docid
        mutated = dict(rows[0])
        mutated.pop("docid")  # simulate re-normalization as a fresh doc
        fresh = _normalize_docs([mutated])[0]
        self.assertNotEqual(fresh["docid"], "inv-2",
                            "re-normalizing the mutated dict mints a NEW docid "
                            "(the double-row bug set_premise_status must avoid)")

    def test_set_premise_status_survives_process_restart(self):
        """The in-place mutation is persisted via _flush() + pickle roundtrip
        — a second service instance (simulating a process restart) sees the
        updated premise_status. Goes through the full legal chain
        (true -> needs_verification -> falsified) to exercise both the
        two-hop migration and the persistence."""
        svc, base, data, audit = _svc_with_audit()
        self._make_confirmed_doc(svc, "persist-1")
        out = svc.set_premise_status("persist-1", "needs_verification")
        self.assertEqual(out["status"], "success")
        out = svc.set_premise_status("persist-1", "falsified", premise_text="已证伪")
        self.assertEqual(out["status"], "success")
        # Verify the persisted row before the restart.
        hot_docs = _load_pkl_docs(data / "hot_corpus.pkl")
        row = [d for d in hot_docs if str(d.get("docid")) == "persist-1"][0]
        self.assertEqual(row.get("premise_status"), "falsified")
        self.assertEqual(row.get("premise"), "已证伪")
        # Close + reopen (simulates a process restart: load from pickle).
        svc.close()
        svc2 = KbSearchService(data, audit_path=audit)
        # No filter: the doc must still be searchable after restart.
        res = svc2.search("经验条目 编号", top_k=5)
        ids = [r["doc_id"] for r in res["data"]["results"]]
        self.assertIn("persist-1", ids,
                      "the doc must survive the restart (in-place persistence)")
        # With the premise_status filter: the falsified state is visible.
        res2 = svc2.search("经验条目 编号", top_k=5,
                           filters={"premise_status": "falsified"})
        ids2 = [r["doc_id"] for r in res2["data"]["results"]]
        self.assertIn("persist-1", ids2,
                      "the falsified status must survive a process restart")


class TR8Tests(unittest.TestCase):
    """Read-side default profile-scoping in _search_handler.

    The handler is in __init__.py; we test it via a real service + a
    monkeypatched _current_profile (the env var HERMES_PROFILE).
    """

    def _import_search_handler(self):
        """Import the _search_handler from the plugin __init__ module.

        Loads __init__.py as a standalone module (the package name has a
        hyphen, so importlib's spec-from-file path is the reliable one)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "kb_search_plugin_init", _PLUGIN_INIT)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _handler_cases(self, svc, query="X 的 agent 经验"):
        """Run the three TR-8 handler cases; return the (not_in, in_none,
        in_self) result-id lists for the assertions."""
        import os
        mod = self._import_search_handler()
        # The handler resolves the service lazily via _svc() — monkeypatch
        # the module-level service object so _svc() returns our instance.
        mod._service = svc
        old_profile = os.environ.get("HERMES_PROFILE")
        try:
            # --- Case 1: profile Y (no explicit filters) must NOT see X doc ---
            os.environ["HERMES_PROFILE"] = "Y"
            out = mod._search_handler({"query": "共享 知识"})
            ids_y = [r["doc_id"] for r in json.loads(out).get("data", {}).get("results", [])]
            # --- Case 2: explicit source_profile=None opt-in (R-7 global
            #   door) — the query text is chosen to match BOTH the global
            #   and the X doc (the "共享" token is shared by both), so a
            #   dropped filter would surface the X doc here and the test
            #   would catch it. ---
            out = mod._search_handler(
                {"query": "共享 知识", "filters": {"source_profile": None}})
            ids_y_none = [r["doc_id"] for r in json.loads(out).get("data", {}).get("results", [])]
            # --- Case 2b: explicit source_profile=X (Y's caller scoping to
            #   X) — proves cross-profile reads work via an explicit filter
            #   value, the only other path out of the implicit scoping.
            #   (Same shared query so the X doc IS a candidate here.) ---
            out = mod._search_handler(
                {"query": "共享 知识", "filters": {"source_profile": "X"}})
            ids_y_x = [r["doc_id"] for r in json.loads(out).get("data", {}).get("results", [])]
            # --- Case 3: profile X (same profile) DOES see its own doc ---
            os.environ["HERMES_PROFILE"] = "X"
            out = mod._search_handler({"query": "共享 知识"})
            ids_x = [r["doc_id"] for r in json.loads(out).get("data", {}).get("results", [])]
        finally:
            if old_profile is not None:
                os.environ["HERMES_PROFILE"] = old_profile
            else:
                os.environ.pop("HERMES_PROFILE", None)
        return ids_y, ids_y_none, ids_y_x, ids_x

    def test_tr8_cross_profile_leak_negative(self):
        """R-7 negative test: a profile X agent doc must NOT be visible
        to a profile Y caller via the implicit source_profile filter.

        Construct: profile X ingests a confirmed agent doc with
        source_profile=X. Then profile Y calls _search_handler with NO
        explicit filters — the implicit filter must exclude the X doc.

        Explicit source_profile=None opt-in (filters={'source_profile':
        None}) is the R-7 door: we can't assert "the Y doc IS found" via
        the filter, because filters are exact-match (a None filter matches
        docs whose source_profile metadata value is None — legacy/global
        docs, not profile-X docs). What we CAN assert, and DO assert:
          - the implicit filter path excludes X's doc from Y's search
          - the None-filter path returns ONLY docs whose source_profile is
            None (i.e. the opt-in door is mechanically respected, not
            dropped — the filter is forwarded to svc.search, not stripped)
        The actual cross-profile visibility of X's doc from Y's side
        happens through an EXPLICIT source_profile=X filter, which the
        test verifies separately.
        """
        import tempfile

        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)

        # Profile X: ingest a confirmed agent doc with source_profile=X.
        # The title/content carry a "共享" token shared with the global doc
        # so the same query text can target both (the filter, not the
        # query, is what distinguishes them).
        svc.ingest([{
            "docid": "x-doc", "title": "X 的 agent 经验 共享 知识",
            "content": "profile X 的 agent 经验 条目 共享 知识 甲乙丙丁戊己庚辛",
            "status": "confirmed", "origin_kind": "agent",
            "source_profile": "X",
        }])
        # A global (no source_profile) doc — the only thing a None filter
        # can match, since metadata filters are exact-match.
        svc.ingest([{
            "docid": "global-doc", "title": "全局 共享 知识",
            "content": "没有 source_profile 的全局 知识 条目 共享 甲乙丙丁戊己",
            "status": "confirmed", "origin_kind": "auto",
        }])
        svc.close()

        ids_y, ids_y_none, ids_y_x, ids_x = self._handler_cases(svc)

        # Case 1: profile Y's implicit-filter search must NOT leak X's doc.
        self.assertNotIn("x-doc", ids_y,
                          "cross-profile leak: profile Y must not see "
                          "profile X's agent doc (implicit filter active)")
        # Case 2: the source_profile=None opt-in filter is forwarded to
        # svc.search as-is — it matches ONLY docs with no source_profile
        # (global/legacy), never a profile-X doc. This proves the door is
        # respected, not silently dropped (dropping it would make Case 2
        # return the same ids as Case 1 — the implicit profile-Y filter).
        self.assertIn("global-doc", ids_y_none,
                      "source_profile=None filter must match global docs "
                      "(the R-7 opt-in door is mechanically respected)")
        self.assertNotIn("x-doc", ids_y_none,
                         "source_profile=None is exact-match: it cannot "
                         "match a doc stored with source_profile='X'")
        # Case 2b: cross-profile READ is possible via an EXPLICIT filter
        # value — profile Y explicitly scoping to X sees X's doc. This is
        # the positive R-7 door (the None-filter is the global door; the
        # explicit-value filter is the cross-profile-by-name door).
        self.assertIn("x-doc", ids_y_x,
                      "explicit source_profile=X filter must let profile Y "
                      "read X's agent doc (cross-profile read by explicit filter)")
        # Case 3: profile X sees its own doc (implicit filter = X).
        self.assertIn("x-doc", ids_x,
                      "profile X must see its own agent doc "
                      "(implicit filter = X)")

    def test_tr8_no_profile_cli_global(self):
        """When HERMES_PROFILE is unset (CLI / no-profile), the handler
        does NOT inject a source_profile filter — global search (all
        profiles + legacy docs with no source_profile)."""
        import os
        import tempfile

        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)

        # A global doc (no source_profile) + a profile-specific doc.
        # Both carry the shared "知识" token so one query targets both.
        svc.ingest([
            {"docid": "global-doc", "title": "全局 共享 知识",
             "content": "没有 source_profile 的全局 知识 条目 共享 甲乙丙丁戊己",
             "status": "confirmed", "origin_kind": "auto"},
            {"docid": "coder-doc", "title": "coder 专属 知识",
             "content": "source_profile 为 coder 的 知识 条目 共享 甲乙丙丁",
             "status": "confirmed", "origin_kind": "auto",
             "source_profile": "coder"},
        ])
        svc.close()

        mod = self._import_search_handler()
        orig_service = mod._service
        mod._service = svc

        old_profile = os.environ.get("HERMES_PROFILE")
        os.environ.pop("HERMES_PROFILE", None)
        try:
            # No profile -> no implicit filter -> both docs visible.
            out = mod._search_handler({"query": "知识 条目 共享"})
            payload = json.loads(out)
            ids = [r["doc_id"] for r in payload.get("data", {}).get("results", [])]
            self.assertIn("global-doc", ids,
                          "no-profile CLI must see global docs (no implicit filter)")
            self.assertIn("coder-doc", ids,
                          "no-profile CLI must see all-profile docs (global search)")
        finally:
            if old_profile is not None:
                os.environ["HERMES_PROFILE"] = old_profile
            mod._service = orig_service


def _load_pkl_docs(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f).get("docs", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
