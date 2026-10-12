#!/usr/bin/env python3
"""Phase 4 TR-10 unit tests: share_to_global explicit escalation to the
global shared layer.

TR-10 (PRD v2 §TR-10, L46 verbatim): a confirmed origin_kind=agent
experience doc in a source_profile may be escalated to the global
shared layer by an EXPLICIT, AUDITED action only — a new narrow tool
face knowledge_base_share_to_global(doc_id) that writes one global
copy via ingest(status=confirmed, source_profile=None). The owner red
line "agent 经验不走私" is enforced as TWO hard gates on the doc being
escalated:
  (a) source_profile match — the doc row's stored source_profile must
      equal the caller's current profile (same matching rule as
      confirm(): a None caller is admitted only when the stored row is
      also None);
  (b) origin_kind == "agent" — user/auto entries never walk to the
      global layer.
Any violation (not_found / profile_mismatch / not_agent_origin) is
rejected AND written to the EXISTING _audit trail as
action=share_to_global with the error code — no new audit face.

Semantics (additive, not a move): the escalation writes a SECOND row
with the same title/content/metadata but source_profile=None and its
own deterministic docid (uuid5 input differs on source_profile — the
Phase 1 origin_kind-into-uuid5 precedent); the ORIGINAL profile row
stays in the corpus untouched. Other profiles reach the doc through
the TR-8 source_profile=None opt-in full search (plain
knowledge_base_search with the explicit filters opt-in door).

The near-dup gate is SKIPPED for the copy itself (its token containment
against the original row would be 1.0 >= 0.85 block threshold — the
copy is an explicit, caller-authorized twin of a doc that already
passed the gate at original ingest; the length gate still runs on the
copy). This is expected behavior, not a gate bypass: every OTHER
confirmed ingest still runs the full near-dup self-check.

Red lines honored (none touched by this phase): TR-7 (hot/cold, flock,
ReinforceStore write gate, staging isolation, graph gate,
occurred_at/version_ts, near-dup thresholds 0.85/0.60, 7-gateway
topology); the pre_llm_call auto-inject layer; confirm()'s
profile_mismatch hard gate (L594-603 semantics reused, not re-built);
the _deterministic_docid (uuid5) mechanism itself.

Run: PYTHONPATH=/root/.hermes/plugins/knowledge-base-search \
     /root/.hermes/plugin-venv/bin/python -m pytest test_kb_phase4_tr10.py -q
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLUGIN_DIR))
_kb_root = str(_PLUGIN_DIR.parents[1])
if _kb_root not in sys.path:
    sys.path.insert(0, _kb_root)

from kb_service import KbSearchService  # noqa: E402

import kb_ingest_hook as hook  # noqa: E402
from kb_ingest_hook import _current_profile  # noqa: E402

import importlib.util  # noqa: E402


def _load_plugin_pkg() -> object:
    path = _PLUGIN_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "kb_plugin_pkg_tr10", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_env_profile(value):
    import os
    old = os.environ.get("HERMES_PROFILE")
    if value is None:
        os.environ.pop("HERMES_PROFILE", None)
    else:
        os.environ["HERMES_PROFILE"] = value
    return old


def _restore_env_profile(old):
    import os
    if old is not None:
        os.environ["HERMES_PROFILE"] = old
    else:
        os.environ.pop("HERMES_PROFILE", None)


def _read_audit(audit: Path) -> list[dict]:
    out = []
    with open(audit, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _share_records(audit: Path) -> list[dict]:
    return [r for r in _read_audit(audit)
            if r.get("action") == "share_to_global"]


# ---------------------------------------------------------------------------
# (a) happy path: this profile's confirmed agent entry escalates; the
# global copy is reachable via the TR-8 source_profile=None opt-in
# full search, while the profile-scoped default search of ANOTHER
# profile still does not leak it pre-escalation behaviour (the copy
# is global, not profile Y's).
# ---------------------------------------------------------------------------

class TR10HappyPathTests(unittest.TestCase):
    def _live(self, base: Path):
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        svc.ingest([
            {"docid": "ag-x", "title": "网关重试预算经验条目甲乙丙",
             "content": "网关 重试 预算 经验 甲乙丙丁戊己 庚辛壬癸",
             "status": "confirmed", "origin_kind": "agent",
             "topic": "messaging", "source_profile": "coder"},
        ])
        return svc, audit

    def test_own_profile_agent_doc_escapes_to_global(self):
        base = Path(tempfile.mkdtemp())
        svc, audit = self._live(base)
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("ag-x", source_profile="coder")
            self.assertEqual(out["status"], "success")
            self.assertEqual(out["doc_id"], "ag-x")
            self.assertIsNotNone(out["global_doc_id"],
                                 "the global copy gets its OWN "
                                 "deterministic docid (uuid5 input "
                                 "differs on source_profile)")
            self.assertNotEqual(out["global_doc_id"], "ag-x",
                                 "a global copy is a NEW row, not an "
                                 "in-place rewrite of the original")
        finally:
            _restore_env_profile(old)
            svc.close()
        # audit trail: action=share_to_global + success status, on the
        # EXISTING audit face (no new file / new action schema).
        recs = [r for r in _share_records(audit) if r.get("error") is None]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["doc_id"], "ag-x")
        self.assertEqual(recs[0]["status"], "shared")
        self.assertEqual(recs[0]["source_profile"], "coder")
        self.assertEqual(recs[0]["global_doc_id"], out["global_doc_id"])

    def test_tr8_opt_in_search_hits_the_global_copy(self):
        """The PRD's point: OTHER profiles reach the escalated doc via
        the TR-8 source_profile=None opt-in full search (plain
        svc.search with the explicit None filter — the one door out of
        profile scoping)."""
        base = Path(tempfile.mkdtemp())
        svc, _audit = self._live(base)
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("ag-x", source_profile="coder")
            gdoc = out["global_doc_id"]
            # TR-8 opt-in: filters={source_profile: None} is the
            # explicit cross-profile global door (the R-7 one and only
            # way out of scoping).
            res = svc.search("网关 重试 经验", top_k=10,
                             filters={"source_profile": None})
            ids = [r["doc_id"] for r in res["data"]["results"]]
            self.assertIn(gdoc, ids,
                          "the global copy (source_profile=None row) "
                          "must be reachable via the TR-8 "
                          "source_profile=None opt-in full search")
        finally:
            _restore_env_profile(old)
            svc.close()

    def test_original_profile_row_survives_escalation(self):
        """(d) additive semantics: after escalation the corpus holds
        BOTH rows — the original source_profile=coder row AND the new
        source_profile=None copy — not an in-place rewrite or a move.
        Profile-coder's scoped search still sees the original row."""
        base = Path(tempfile.mkdtemp())
        svc, _audit = self._live(base)
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("ag-x", source_profile="coder")
            gdoc = out["global_doc_id"]
            self.assertNotEqual(gdoc, "ag-x")
            # original row still there under its own profile scoping
            res = svc.search("网关 重试 经验", top_k=10,
                             filters={"source_profile": "coder"})
            ids = [r["doc_id"] for r in res["data"]["results"]]
            self.assertIn("ag-x", ids,
                         "the original profile row must remain "
                         "intact after escalation (additive copy, "
                         "never a move/rewrite)")
            self.assertNotIn(gdoc, ids,
                             "the global copy is NOT in the profile-"
                             "scoped view (it belongs to source_"
                             "profile=None)")
            # global opt-in view sees the copy, not the original row
            res2 = svc.search("网关 重试 经验", top_k=10,
                              filters={"source_profile": None})
            ids2 = [r["doc_id"] for r in res2["data"]["results"]]
            self.assertIn(gdoc, ids2)
            self.assertNotIn("ag-x", ids2,
                             "the original row is profile-coder's; "
                             "only the copy is global")
        finally:
            _restore_env_profile(old)
            svc.close()

    def test_idempotent_second_escalation_is_a_noop_row_level(self):
        """Escalating the same doc twice writes NO second global row:
        the second call's copy has the same deterministic docid
        (uuid5 over the same dict) and the idempotency check detects
        the existing source_profile=None row, skipping update +
        record_ingest_ts + _flush entirely. The physical corpus after
        one successful escalation holds exactly TWO rows for this doc
        — the original source_profile=coder row (1) plus the global
        copy (1); a repeat call adds zero rows (not "always 1 row":
        it is 2 rows total, stable across repeats). The second call
        still returns success + an audit line (already_shared=True),
        so the no-op is recorded, not silent."""
        base = Path(tempfile.mkdtemp())
        svc, _audit = self._live(base)
        old = _set_env_profile("coder")
        try:
            out1 = svc.share_to_global("ag-x", source_profile="coder")
            out2 = svc.share_to_global("ag-x", source_profile="coder")
            self.assertEqual(out1["global_doc_id"], out2["global_doc_id"],
                             "deterministic docid: same input dict "
                             "(source_profile=None) -> same global row, "
                             "no duplicate")
            self.assertFalse(out1.get("already_shared"),
                             "the first escalation is the real write")
            self.assertTrue(out2.get("already_shared"),
                             "the second escalation is the idempotent "
                             "hit, not a re-write")
            # physical row count: the global docid appears EXACTLY once
            # in the hot+cold corpus (not twice — no duplicate row).
            hot = svc._hot.docs_snapshot()
            cold = svc._cold.docs_snapshot()
            global_rows = [d for d in hot + cold
                           if str(d.get("docid")) == out1["global_doc_id"]
                           and d.get("source_profile") is None]
            self.assertEqual(len(global_rows), 1,
                             "one successful escalation = original row "
                             "(1) + global copy (1) = 2 physical rows "
                             "total; a repeat adds ZERO new rows")
            res = svc.search("网关 重试 经验", top_k=10,
                             filters={"source_profile": None})
            hits = [r for r in res["data"]["results"]
                    if r["doc_id"] == out1["global_doc_id"]]
            self.assertEqual(len(hits), 1)
        finally:
            _restore_env_profile(old)
            svc.close()


# ---------------------------------------------------------------------------
# (b) cross-profile entry: source_profile != caller -> rejected +
# audit (action=share_to_global, error=profile_mismatch).
# ---------------------------------------------------------------------------

class TR10CrossProfileRejectionTests(unittest.TestCase):
    def _live(self, base: Path):
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        svc.ingest([
            {"docid": "ag-y", "title": "网关灰度发布经验条目庚辛壬",
             "content": "网关 灰度 发布 经验 庚辛壬癸甲乙 子丑寅",
             "status": "confirmed", "origin_kind": "agent",
             "topic": "deployment", "source_profile": "devops"},
        ])
        return svc, audit

    def test_foreign_profile_agent_doc_is_rejected_and_audited(self):
        base = Path(tempfile.mkdtemp())
        svc, audit = self._live(base)
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("ag-y", source_profile="coder")
            self.assertEqual(out["status"], "error")
            self.assertEqual(out["error"], "profile_mismatch")
        finally:
            _restore_env_profile(old)
            svc.close()
        # audit red line: the REJECTION is written to the existing
        # _audit face as action=share_to_global + error code.
        recs = _share_records(audit)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["doc_id"], "ag-y")
        self.assertEqual(recs[0]["error"], "profile_mismatch")
        self.assertEqual(recs[0]["stored_profile"], "devops")
        self.assertEqual(recs[0]["source_profile"], "coder")
        # no global copy was written: the TR-8 opt-in view has zero
        # hits for the foreign doc's content.
        res = svc.search("网关 灰度 发布 经验", top_k=10,
                         filters={"source_profile": None})
        self.assertEqual(res["data"]["results"], [])

    def test_cross_profile_rejection_means_no_corpus_change(self):
        base = Path(tempfile.mkdtemp())
        svc, _audit = self._live(base)
        old = _set_env_profile("coder")
        try:
            svc.share_to_global("ag-y", source_profile="coder")
        finally:
            _restore_env_profile(old)
        # the foreign profile row is untouched and still profile-scoped
        res = svc.search("网关 灰度 发布 经验", top_k=10,
                         filters={"source_profile": "devops"})
        self.assertEqual([r["doc_id"] for r in res["data"]["results"]],
                         ["ag-y"])
        svc.close()


# ---------------------------------------------------------------------------
# (c) own profile but origin_kind != agent -> rejected + audit
# (error=not_agent_origin). The red line's other half: NOT every doc
# of this profile may escalate — only agent-produced ones.
# ---------------------------------------------------------------------------

class TR10NonAgentOriginRejectionTests(unittest.TestCase):
    def _live(self, base: Path, origin_kind: str, docid: str):
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        svc.ingest([
            {"docid": docid, "title": "网关知识录入条目内容 " + docid,
             "content": "网关 知识 用户 录入 内容 甲乙丙丁戊己 " + docid,
             "status": "confirmed", "origin_kind": origin_kind,
             "topic": "messaging", "source_profile": "coder"},
        ])
        return svc, audit

    def test_user_origin_own_profile_is_rejected_and_audited(self):
        base = Path(tempfile.mkdtemp())
        svc, audit = self._live(base, "user", "us-x")
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("us-x", source_profile="coder")
            self.assertEqual(out["status"], "error")
            self.assertEqual(out["error"], "not_agent_origin")
        finally:
            _restore_env_profile(old)
            svc.close()
        recs = _share_records(audit)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["doc_id"], "us-x")
        self.assertEqual(recs[0]["error"], "not_agent_origin")
        self.assertEqual(recs[0]["origin_kind"], "user")

    def test_auto_origin_own_profile_is_rejected_and_audited(self):
        base = Path(tempfile.mkdtemp())
        svc, audit = self._live(base, "auto", "au-x")
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("au-x", source_profile="coder")
            self.assertEqual(out["status"], "error")
            self.assertEqual(out["error"], "not_agent_origin")
        finally:
            _restore_env_profile(old)
            svc.close()
        recs = _share_records(audit)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["error"], "not_agent_origin")
        self.assertEqual(recs[0]["origin_kind"], "auto")

    def test_rejected_non_agent_docs_leave_no_global_row(self):
        base = Path(tempfile.mkdtemp())
        svc, _audit = self._live(base, "user", "us-x")
        old = _set_env_profile("coder")
        try:
            svc.share_to_global("us-x", source_profile="coder")
        finally:
            _restore_env_profile(old)
        res = svc.search("网关 知识 录入", top_k=10,
                         filters={"source_profile": None})
        self.assertEqual(res["data"]["results"], [],
                         "a rejected escalation must not leak a "
                         "global copy into the TR-8 opt-in view")
        svc.close()


# ---------------------------------------------------------------------------
# not_found: an unknown doc_id is rejected + audited too (the audit
# red line covers every outcome of the action, not just the two
# ownership violations).
# ---------------------------------------------------------------------------

class TR10NotFoundTests(unittest.TestCase):
    def test_unknown_doc_id_rejected_and_audited(self):
        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("nope", source_profile="coder")
            self.assertEqual(out["status"], "error")
            self.assertEqual(out["error"], "not_found")
        finally:
            _restore_env_profile(old)
            svc.close()
        recs = _share_records(audit)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["error"], "not_found")
        self.assertEqual(recs[0]["doc_id"], "nope")


# ---------------------------------------------------------------------------
# near-dup interaction: the global copy's token containment against the
# original row is 1.0 (>= 0.85 block) — it is EXPECTED to skip the
# near-dup self-check (explicit caller-authorized twin of an already
# gated doc), while the normal ingest path still blocks near-dups.
# ---------------------------------------------------------------------------

class TR10NearDupInteractionTests(unittest.TestCase):
    def test_global_copy_is_100_containment_of_original(self):
        """Prove the premise of the skip: the copy (title/content
        identical, only source_profile differs) would score 1.0 token
        containment against the original row under the normal
        near-dup gate — i.e. a plain re-ingest of the same content is
        blocked, which is why the escalation must carry its explicit
        skip."""
        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        svc.ingest([
            {"docid": "ag-x", "title": "网关重试预算经验条目甲乙丙",
             "content": "网关 重试 预算 经验 甲乙丙丁戊己 庚辛壬癸",
             "status": "confirmed", "origin_kind": "agent",
             "topic": "messaging", "source_profile": "coder"},
        ])
        # A plain re-ingest of the SAME content (a different docid,
        # different source_profile) is a near-dup and gets blocked:
        re = svc.ingest([
            {"docid": "ag-x-twin", "title": "网关重试预算经验条目甲乙丙",
             "content": "网关 重试 预算 经验 甲乙丙丁戊己 庚辛壬癸",
             "status": "confirmed", "origin_kind": "agent",
             "source_profile": None},
        ])
        self.assertEqual(re["ingested"], [],
                         "the normal ingest path still blocks the "
                         "near-exact twin (gate intact)")
        self.assertEqual(len(re["rejected"]), 1)
        self.assertEqual(re["rejected"][0]["reason"],
                         "near_duplicate")
        # The escalation path is NOT blocked (explicit skip):
        out = svc.share_to_global("ag-x", source_profile="coder")
        self.assertEqual(out["status"], "success")
        svc.close()

    def test_near_dup_gate_is_untouched_for_other_ingest(self):
        """Red-line KPI: the near-dup thresholds (0.85 block / 0.60
        audit) behave exactly as before for ordinary ingests — TR-10
        only skipped the self-check for ITS OWN explicit copy, never
        the gate itself."""
        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        svc.ingest([
            {"docid": "base-1", "title": "网关重试预算经验条目甲乙丙",
             "content": "网关 重试 预算 经验 甲乙丙丁戊己 庚辛壬癸",
             "status": "confirmed", "origin_kind": "auto"},
        ])
        # near-duplicate of base-1: blocked
        r1 = svc.ingest([
            {"docid": "base-1-dup", "title": "网关重试预算经验条目甲乙丙",
             "content": "网关 重试 预算 经验 甲乙丙丁戊己 庚辛壬癸",
             "status": "confirmed", "origin_kind": "auto"},
        ])
        self.assertEqual(r1["ingested"], [])
        self.assertEqual(r1["rejected"][0]["reason"], "near_duplicate")
        svc.close()


# ---------------------------------------------------------------------------
# plugin tool face: knowledge_base_share_to_global is registered with a
# NARROW schema (doc_id only — no batch, no other knobs) and forwards
# the caller's profile from the SAME _current_profile() source as the
# TR-8 implicit scoping (one identity face, no new plumbing).
# ---------------------------------------------------------------------------

class TR10PluginFaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._pkg = _load_plugin_pkg()

    def _register(self):
        """Replay the module's real register() against a fake ctx that
        captures register_tool / register_hook calls."""
        class _Ctx:
            def __init__(self):
                self.register_tool_calls = []
            def register_tool(self, name, **kw):
                self.register_tool_calls.append((name, kw))
            def register_hook(self, *a, **kw):
                pass
        fake = _Ctx()
        self._pkg.register(fake)
        self._ctx = fake

    def test_tool_is_registered_with_narrow_schema(self):
        self._register()
        names = [n for n, _ in self._ctx.register_tool_calls]
        self.assertIn("knowledge_base_share_to_global", names)
        schema = [kw for n, kw in self._ctx.register_tool_calls
                  if n == "knowledge_base_share_to_global"][0]["schema"]
        self.assertEqual(
            set(schema["parameters"]["properties"].keys()),
            {"doc_id"},
            "the face must stay narrow: doc_id only (the PRD's '一次' "
            "action) — no batch arg, no status knob, no profile knob")
        self.assertEqual(schema["parameters"]["required"], ["doc_id"])

    def test_handler_forwards_current_profile_to_service(self):
        """The handler passes _current_profile() as source_profile to
        svc.share_to_global — the same identity source _search_handler
        uses for its TR-8 implicit scoping."""
        fake = mock.Mock()
        fake.share_to_global.return_value = {
            "status": "success", "doc_id": "ag-x",
            "global_doc_id": "g-1", "source_profile": "coder"}
        fake.persist = mock.Mock()
        old = _set_env_profile("coder")
        try:
            with mock.patch.object(self._pkg, "_svc", lambda: fake):
                out = self._pkg._share_to_global_handler({"doc_id": "ag-x"})
            fake.share_to_global.assert_called_once_with(
                "ag-x", source_profile="coder")
        finally:
            _restore_env_profile(old)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "success")
        fake.persist.assert_called_once()  # durable across restarts

    def test_handler_no_profile_env_degrades_to_none_caller(self):
        """With no profile env (CLI/global context), the caller
        identity is None and flows through unchanged — the service
        then applies the confirm()-style rule (None admitted only
        against a stored-None row; here every row is profile-owned so
        the result is not_found/profile_mismatch-class rejection, NOT
        a silent global write)."""
        fake = mock.Mock()
        fake.share_to_global.return_value = {
            "status": "error", "error": "not_found", "doc_id": "ag-x"}
        old = _set_env_profile(None)
        try:
            with mock.patch.object(self._pkg, "_svc", lambda: fake):
                out = self._pkg._share_to_global_handler({"doc_id": "ag-x"})
            fake.share_to_global.assert_called_once_with(
                "ag-x", source_profile=None)
        finally:
            _restore_env_profile(old)
        self.assertEqual(json.loads(out)["status"], "error")

    def test_handler_persists_even_on_rejection(self):
        """persist() runs regardless of outcome so a later successful
        escalation is durable; a rejection writes nothing to the
        corpus, only the audit record (written inside the service)."""
        fake = mock.Mock()
        fake.share_to_global.return_value = {
            "status": "error", "error": "profile_mismatch",
            "doc_id": "ag-y"}
        fake.persist = mock.Mock()
        old = _set_env_profile("coder")
        try:
            with mock.patch.object(self._pkg, "_svc", lambda: fake):
                out = self._pkg._share_to_global_handler({"doc_id": "ag-y"})
        finally:
            _restore_env_profile(old)
        self.assertEqual(json.loads(out)["status"], "error")
        fake.persist.assert_called_once()


# ---------------------------------------------------------------------------
# (e) semantic gate: premise_status == falsified is REJECTED (a falsified
# premise must not walk into the global pool); needs_verification and
# True/"true" stay admissible (same two-state allow rule as the KPI-3
# decay gate — NOT every non-true state blocks, only explicit falsified).
# ---------------------------------------------------------------------------

class TR10PremiseFalsifiedRejectionTests(unittest.TestCase):
    def _live(self, base: Path, docid: str, premise_status):
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        svc.ingest([
            {"docid": docid, "title": "网关重试预算经验条目甲乙丙",
             "content": "网关 重试 预算 经验 甲乙丙丁戊己 庚辛壬癸",
             "status": "confirmed", "origin_kind": "agent",
             "topic": "messaging", "source_profile": "coder",
             "premise_status": premise_status},
        ])
        return svc, audit

    def test_falsified_doc_is_rejected_and_audited(self):
        """premise_status is the falsified state -> rejected +
        audit line error=premise_falsified, on the EXISTING
        _audit face (action=share_to_global, no new face). The falsified
        state bit is the literal False value (the same value
        set_premise_status stores after a falsification migration),
        distinct from "needs_verification" and True/"true"."""
        base = Path(tempfile.mkdtemp())
        svc, audit = self._live(base, "pf-x", False)
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("pf-x", source_profile="coder")
            self.assertEqual(out["status"], "error")
            self.assertEqual(out["error"], "premise_falsified")
        finally:
            _restore_env_profile(old)
            svc.close()
        recs = _share_records(audit)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["doc_id"], "pf-x")
        self.assertEqual(recs[0]["error"], "premise_falsified")
        # no global copy leaked into the TR-8 opt-in view
        res = svc.search("网关 重试 经验", top_k=10,
                         filters={"source_profile": None})
        self.assertEqual(res["data"]["results"], [])

    def test_needs_verification_doc_still_escalates(self):
        """Semantic distinction: "needs_verification" is NOT a block —
        only the explicitly falsified state is. A doc under re-check
        stays admissible (the KPI-3 two-state allow rule: true /
        needs_verification pass, falsified blocks)."""
        base = Path(tempfile.mkdtemp())
        svc, audit = self._live(base, "nv-x", "needs_verification")
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("nv-x", source_profile="coder")
            self.assertEqual(out["status"], "success")
            self.assertIsNotNone(out["global_doc_id"])
        finally:
            _restore_env_profile(old)
            svc.close()
        # success audit line (no error field), action still
        # share_to_global — single channel
        recs = [r for r in _share_records(audit)
                if r.get("error") is None]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["doc_id"], "nv-x")
        self.assertEqual(recs[0]["status"], "shared")

    def test_true_premise_status_doc_still_escalates(self):
        """premise_status=True (the held default) is the third
        admissible state — never blocked."""
        base = Path(tempfile.mkdtemp())
        svc, audit = self._live(base, "pt-x", True)
        old = _set_env_profile("coder")
        try:
            out = svc.share_to_global("pt-x", source_profile="coder")
            self.assertEqual(out["status"], "success")
        finally:
            _restore_env_profile(old)
            svc.close()


# ---------------------------------------------------------------------------
# (f) idempotency: a second escalation of the same doc is a recorded
# no-op — no re-write (update / record_ingest_ts / _flush all skipped),
# no new physical row, but still an audit line (already_shared=True)
# + success return (never a silent skip).
# ---------------------------------------------------------------------------

class TR10IdempotentRepeatTests(unittest.TestCase):
    def _live(self, base: Path):
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        svc.ingest([
            {"docid": "ag-x", "title": "网关重试预算经验条目甲乙丙",
             "content": "网关 重试 预算 经验 甲乙丙丁戊己 庚辛壬癸",
             "status": "confirmed", "origin_kind": "agent",
             "topic": "messaging", "source_profile": "coder"},
        ])
        return svc, audit

    def test_second_call_is_audited_noop_without_rewrite(self):
        """Same doc escalated twice: first call = the real write
        (update + record_ingest_ts + _flush + audit, no
        already_shared marker); second call = idempotent hit — skips
        all three write steps, no new physical row, returns success
        + an audit line carrying already_shared=True."""
        base = Path(tempfile.mkdtemp())
        svc, audit = self._live(base)
        old = _set_env_profile("coder")
        try:
            out1 = svc.share_to_global("ag-x", source_profile="coder")
            out2 = svc.share_to_global("ag-x", source_profile="coder")
            self.assertEqual(out1["status"], "success")
            self.assertEqual(out2["status"], "success")
            self.assertEqual(out1["global_doc_id"], out2["global_doc_id"])
            self.assertNotIn("already_shared", out1,
                             "first escalation: the real write")
            self.assertIs(out2.get("already_shared"), True,
                          "second escalation: idempotent hit marked")
            # physical rows: global copy (source_profile=None) appears
            # EXACTLY once in the hot+cold corpus after both calls —
            # the repeat added zero rows.
            hot = svc._hot.docs_snapshot()
            cold = svc._cold.docs_snapshot()
            global_rows = [d for d in hot + cold
                          if str(d.get("docid")) == out1["global_doc_id"]
                          and d.get("source_profile") is None]
            self.assertEqual(len(global_rows), 1,
                             "repeat call must not add a second global "
                             "row (physical row count: original 1 + "
                             "global copy 1 = 2 rows total, stable)")
        finally:
            _restore_env_profile(old)
            svc.close()
        # audit trail: two share_to_global lines — the first is the
        # real write (no already_shared key), the second carries
        # already_shared=True (the idempotent hit, recorded not
        # silent).
        recs = _share_records(audit)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].get("status"), "shared")
        self.assertNotIn("already_shared", recs[0])
        self.assertIs(recs[1].get("already_shared"), True)
        self.assertEqual(recs[1].get("status"), "shared")
        self.assertEqual(recs[1].get("global_doc_id"),
                         out1["global_doc_id"])

    def test_third_call_still_idempotent_and_flat_rows(self):
        """A third call behaves identically to the second (flat row
        count, another already_shared audit line) — the no-op
        property is stable across N repeats, not a one-shot."""
        base = Path(tempfile.mkdtemp())
        svc, audit = self._live(base)
        old = _set_env_profile("coder")
        try:
            out1 = svc.share_to_global("ag-x", source_profile="coder")
            out2 = svc.share_to_global("ag-x", source_profile="coder")
            out3 = svc.share_to_global("ag-x", source_profile="coder")
            self.assertIs(out3.get("already_shared"), True)
            self.assertEqual(out1["global_doc_id"],
                             out3["global_doc_id"])
            hot = svc._hot.docs_snapshot()
            cold = svc._cold.docs_snapshot()
            global_rows = [d for d in hot + cold
                           if str(d.get("docid")) == out1["global_doc_id"]
                           and d.get("source_profile") is None]
            self.assertEqual(len(global_rows), 1)
        finally:
            _restore_env_profile(old)
            svc.close()
        recs = _share_records(audit)
        self.assertEqual(len(recs), 3)
        self.assertTrue(all(r.get("already_shared") for r in recs[1:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
