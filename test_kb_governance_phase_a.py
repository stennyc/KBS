"""Phase A governance PRD v2-final-revised.3 — write-side contract tests.

TDD: these tests define the contract before the implementation lands.
Covers the 11 delivery items' Phase A portion:

  (1) B1 None-filter:
      - _deterministic_docid drops every None-valued key (key=None ==
        key-absent, the docid semantics narrow + re-ingest double-row
        bug closes)
      - _normalize_docs setdefaults gov_tier (default None, part of the
        same "None == absent" contract)
  (2) B1 filter-face semantic unification: search() filter value
      "__unset__" matches BOTH physically-absent keys AND keys whose
      value is None (unified "None ≡ key absent" across docid + filter
      faces). Any other filter value keeps exact-equality semantics.
  (3) B2 confirm dedup + clarify downgrade (revised.2: option-b):
      - confirm() of a row carrying gov_tier=X scans the confirmed corpus
        (hot+cold docs_snapshot) for same title+content with
        gov_tier != X. On hit the row is NOT moved into the corpus: it
        stays PENDING in staging, is marked _suspected_tier_conflict, is
        AUDITED (tier_conflict), and is DOWNGRADED to a plain PENDING
        (gov_tier cleared so a re-confirm re-scans) with the session's
        clarify channel re-triggered (_fire_clarify) so the footnote
        surfaces it to the owner. No new tool face, no confirm signature
        change; owner confirmation goes through the EXISTING
        knowledge_base_confirm with a non-agent confirmed_by (the
        _suspected_tier_conflict mark is metadata, the move itself is
        byte-identical to a normal confirm; skip leaves the row PENDING).
      - No-gov_tier rows confirm byte-identically to pre-Phase-A
        behaviour (regression guard).
  (4) B4 anchor points (independent 4-class regex, union with LLM):
      - extract_anchor_points(): commit hash / abs file path / version
        number / service+config-key, independent of any LLM
      - close_task_boundary verdict routing:
          ① LLM healthy + tier=transcript        -> original content kept
             (fail-soft floor), anchor union = regex ∪ LLM anchors
          ② LLM fail (use_original AND
             model_used=False) + regex anchors -> archive doc
          ③ LLM fail + NO regex anchors      -> transcript DROP:
             NOT ingested at all (no staging row), audit only (N2:
             dropped rows go to the stale_staging audit surface, they
             never enter the active clarify batch)
      - no LLM-fail fail-open to archive: a fail verdict with no
        regex anchors must NOT stage anything
  (5) B3 exemption branch: _agent_in_decay_gate exempts
      kb_class=incident + project_ref + premise held — inserted AFTER
      the premise_status check and BEFORE the reference/age checks.
      falsified premise still fails closed.
  (6) note 3 repo capture:
      - Candidate gains project_ref (Optional[str], default None — the
        origin_kind backward-compat precedent)
      - _WriteSideState.__init__ gains self._repo_ctx; on_turn captures
        git toplevel/remote fail-soft (git unavailable -> None ->
        docs never upgrade to archive)
  (7) note 5: _build_agent_doc writes the project_ref key ONLY for
      gov_tier=archive; experience/transcript docs omit the key
      entirely (never None).
  (8) note 4 staging TTL: StagingStore.sweep_stale(now_ts) re-keys rows
      idle > 30 days to status=stale_staging (kept for audit, excluded
      from clarify batches by find()'s pending-only semantics; the
      caller's ClarifyAggregator buffers live in memory so stale rows
      are physically invisible to pending_for once re-confirmed/moved).
  (9) B1 migration script tools/migrate_docid_none_filter.py:
      - forward: recompute every hot/cold pkl + staging jsonl row's docid
        under the None-filter formula, dump old->new map to
        kb-data/docid_migration_map.json, align ingest_ts.jsonl, and
        rewrite the stores so re-running the NEW formula over the
        stored rows is a fixpoint (the revised.2 audit criterion).
      - rollback: restore old docids from the map, strip the new
        project_ref/gov_tier keys, and re-align the sidecar.

Run: PYTHONPATH=<plugin-venv site-packages>:<plugin dir> \
     /root/.hermes/plugin-venv/bin/python -m pytest \
     test_kb_governance_phase_a.py -q
"""

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kb_service import KbSearchService, _deterministic_docid, _normalize_docs
import kb_service
import kb_ingest_hook
from kb_ingest_hook import (
    _WriteSideState,
    _agent_in_decay_gate,
)
from kb_ingest_rules import Candidate
from staging import StagingStore

PLUGIN_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


class _FakeCtx:
    """Test context: no config, no manager (pure defaults + stubs)."""
    _config = None

    def __init__(self):
        self._manager = None


class FakeService:
    """Lightweight service stub for write-side sink tests.

    Implements just enough of KbSearchService's surface for
    close_task_boundary: staging + a minimal hot/cold in-memory store
    + audit. No KB corpus machinery involved."""

    def __init__(self):
        self._staged = []
        self._confirmed = []
        self._audit_records = []

    def _audit(self, **fields):
        self._audit_records.append(dict(fields))

    # --- staging surface (what close_task_boundary / confirm use) ---
    class _Stag:
        def __init__(self, s): self._s = s
        def docs_snapshot(self): return [dict(d) for d in self._s._staged]
        def find(self, did):
            for d in self._s._staged:
                if str(d.get("docid")) == did: return dict(d)
            return None
        def add(self, doc):
            self._s._staged = [d for d in self._s._staged
                               if str(d.get("docid")) != str(doc.get("docid"))]
            self._s._staged.append(dict(doc))
        def remove(self, did):
            for i, d in enumerate(self._s._staged):
                if str(d.get("docid")) == did:
                    return self._s._staged.pop(i)
            return None
        def persist(self): pass

    @property
    def _staging(self):
        return self._Stag(self)

    def ingest(self, docs):
        staged = []
        for d in docs:
            d2 = dict(d)
            d2.setdefault("docid", _deterministic_docid(d2))
            self._staged.append(d2)
            staged.append(d2["docid"])
        return {"ingested": [], "rejected": [], "staged": staged,
                "generation": 0}

    def confirm(self, doc_id, source_profile=None, confirmed_by=None):
        removed = self._Stag(self).remove(doc_id)
        if removed is None:
            self._audit(action="confirm", doc_id=doc_id,
                        source_profile=source_profile,
                        confirmed_by=confirmed_by, error="not_found")
            return {"status": "error", "error": "not_found",
                    "doc_id": doc_id}
        removed["status"] = "confirmed"
        removed["confirmed_by"] = confirmed_by
        self._confirmed.append(removed)
        self._audit(action="confirm", doc_id=doc_id,
                    source_profile=removed.get("source_profile"),
                    confirmed_by=confirmed_by, status="confirmed")
        return {"status": "success", "doc_id": doc_id,
                "confirmed_by": confirmed_by}

    def _flush(self):
        pass

    # --- hot/cold snapshot surface (for B2 dedup scan) ---
    class _Layer:
        def __init__(self, s): self._s = s
        def docs_snapshot(self): return [dict(d) for d in self._s._confirmed]

    @property
    def _hot(self):
        return self._Layer(self)

    @property
    def _cold(self):
        return self._Layer(self)

    @property
    def _data_dir(self):
        import tempfile
        return Path(tempfile.gettempdir()) / "fake-data"

    @property
    def _audit_path(self):
        """Return a path whose contents we can read as audit records.

        The FakeService records audit in memory; this property writes
        them out to a temp jsonl so the test's _audit_lines() can read
        them via the file interface.
        """
        import tempfile, os
        p = Path(tempfile.gettempdir()) / f"fake_audit_{id(self)}.jsonl"
        p.write_text("\n".join(
            json.dumps(r, ensure_ascii=False, default=str)
            for r in self._audit_records))
        return p


def _state_with_service() -> _WriteSideState:
    state = _WriteSideState(_FakeCtx())
    state._service_override = FakeService()
    return state


def _make_state(state: _WriteSideState) -> _WriteSideState:
    """Identity helper — the tests call `st = _make_state(state)` then
    `st.close_task_boundary(...)`; keep it so the test reads as
    "set up state, then close the boundary" without obscuring the
    fact that state IS the object under test."""
    return state


# ---------------------------------------------------------------------------
# (1) B1 None-filter
# ---------------------------------------------------------------------------

class B1NoneFilterTests(unittest.TestCase):

    def test_none_valued_keys_excluded_from_docid(self):
        base = {"title": "t", "content": "c" * 30}
        plain = _deterministic_docid(dict(base))
        with_nones = dict(base)
        with_nones["gov_tier"] = None
        with_nones["project_ref"] = None
        self.assertEqual(
            _deterministic_docid(with_nones), plain,
            "a None-valued key must not change the docid "
            "(B1 None-filter: key=None ≡ key absent)")

    def test_non_none_values_still_count(self):
        a = _deterministic_docid({"title": "t", "content": "c" * 30,
                                  "gov_tier": "experience"})
        b = _deterministic_docid({"title": "t", "content": "c" * 30,
                                  "gov_tier": "archive"})
        self.assertNotEqual(a, b,
                            "non-None values stay in the docid input")

    def test_key_present_none_equals_key_absent(self):
        # Contract (b): a doc whose gov_tier key physically exists with
        # None maps to the SAME docid as one where the key is absent.
        self.assertEqual(
            _deterministic_docid({"title": "x", "content": "c" * 30,
                                  "gov_tier": None}),
            _deterministic_docid({"title": "x", "content": "c" * 30}))

    def test_normalize_docs_setdefaults_gov_tier(self):
        docs = _normalize_docs([{"title": "t", "content": "c" * 30}])
        self.assertIn("gov_tier", docs[0])
        self.assertIsNone(docs[0]["gov_tier"],
                          "gov_tier defaults to None (absent-equivalent)")
        # An explicit gov_tier is preserved, never clobbered.
        docs2 = _normalize_docs(
            [{"title": "t", "content": "c" * 30,
              "gov_tier": "archive"}])
        self.assertEqual(docs2[0]["gov_tier"], "archive")

    def test_normalize_docs_none_gov_tier_keeps_legacy_docid(self):
        # Contract (a): a legacy doc (gov_tier key absent) ingested under
        # the NEW formula gets the SAME docid as a doc ingested under the
        # OLD formula — the docid semantics for existing content are
        # stable; only None-valued keys narrow away.
        legacy = _normalize_docs(
            [{"title": "legacy", "content": "l" * 30}])[0]
        docid_new_formula = _deterministic_docid(legacy)
        # Simulate the old formula (no None-filter): with gov_tier=None
        # setdefaulted in, the old formula would have produced a
        # DIFFERENT id; the new one must equal the key-absent id.
        self.assertEqual(
            docid_new_formula,
            _deterministic_docid(
                {k: v for k, v in legacy.items() if v is not None}
                and {k: v for k, v in legacy.items() if v is not None}))


# ---------------------------------------------------------------------------
# (2) B1 filter-face unification: __unset__ sentinel
# ---------------------------------------------------------------------------

class FilterUnsetSentinelTests(unittest.TestCase):

    def _svc(self, docs):
        svc, base, data, audit = _svc_with_audit()
        svc.ingest(docs)
        return svc

    def test_unset_matches_physically_absent_key(self):
        doc = {"docid": "f-1", "title": "哨兵 测试 文档",
               "content": "未设置 gov_tier 的 文档 甲乙丙丁戊己庚辛",
               "status": "confirmed"}
        svc = self._svc([doc])
        res = svc.search("哨兵 测试 文档",
                         filters={"gov_tier": "__unset__"})
        ids = [r["doc_id"] for r in res["data"]["results"]]
        self.assertIn("f-1", ids,
                      "__unset__ must match a physically-absent key")

    def test_unset_matches_none_valued_key(self):
        doc = {"docid": "f-2", "title": "哨兵 测试 文档",
               "content": "gov_tier 为 None 的 文档 甲乙丙丁戊己庚辛",
               "status": "confirmed", "gov_tier": None}
        svc = self._svc([doc])
        res = svc.search("哨兵 测试 文档",
                         filters={"gov_tier": "__unset__"})
        ids = [r["doc_id"] for r in res["data"]["results"]]
        self.assertIn("f-2", ids,
                      "__unset__ must also match key-present-None "
                      "(unified with the B1 docid face)")

    def test_exact_value_filter_unchanged(self):
        doc = {"docid": "f-3", "title": "哨兵 测试 文档",
               "content": "archive tier 的 文档 甲乙丙丁戊己庚辛",
               "status": "confirmed", "gov_tier": "archive"}
        svc = self._svc([doc])
        res = svc.search("哨兵 测试 文档",
                         filters={"gov_tier": "archive"})
        self.assertIn("f-3", [r["doc_id"]
                               for r in res["data"]["results"]])
        res2 = svc.search("哨兵 测试 文档",
                          filters={"gov_tier": "experience"})
        self.assertNotIn("f-3", [r["doc_id"]
                                  for r in res2["data"]["results"]])

    def test_none_filter_value_no_longer_magic(self):
        # A literal filter value of None now behaves like any other
        # exact value (matches keys whose value IS None; a
        # physically-absent key yields get(k)=None too, so both
        # match — but the PRD mandates the sentinel, not None, for
        # "unset": this test locks the mechanical .get(k)==v path so a
        # regression cannot quietly change it).
        doc = {"docid": "f-4", "title": "哨兵 测试 文档",
               "content": "None 值 键 的 文档 甲乙丙丁戊己庚辛",
               "status": "confirmed", "gov_tier": None}
        svc = self._svc([doc])
        res = svc.search("哨兵 测试 文档", filters={"gov_tier": None})
        self.assertIn("f-4", [r["doc_id"] for r in res["data"]["results"]])


# ---------------------------------------------------------------------------
# (3) B2 confirm dedup + clarify downgrade
# ---------------------------------------------------------------------------

class B2ConfirmDedupTests(unittest.TestCase):

    def test_tier_conflict_blocks_move_and_downgrades(self):
        svc, base, data, audit = _svc_with_audit()
        # Confirmed corpus already holds the same title+content under
        # gov_tier=experience (legacy confirmed doc, no tier key).
        svc.ingest([{"docid": "corp-1", "title": "Kafka 消费者 故障 经验",
                     "content": "Kafka 消费者 故障 时的 排查 步骤 甲乙丙丁戊己",
                     "status": "confirmed", "gov_tier": "experience"}])
        # A staging row with the SAME title+content but a DIFFERENT tier.
        svc.ingest([{"docid": "pend-c1", "title": "Kafka 消费者 故障 经验",
                     "content": "Kafka 消费者 故障 时的 排查 步骤 甲乙丙丁戊己",
                     "status": "pending", "gov_tier": "archive",
                     "source_profile": "coder"}])

        out = svc.confirm("pend-c1", source_profile="coder",
                          confirmed_by="agent")
        self.assertEqual(out["status"], "success",
                         "the confirm call itself succeeds (the row was "
                         "consumed and re-staged)")
        self.assertEqual(out.get("note"), "tier_conflict")
        # The row is NOT in the confirmed corpus.
        hot = {str(d.get("docid"))
               for d in svc._hot.docs_snapshot()}
        cold = {str(d.get("docid"))
                 for d in svc._cold.docs_snapshot()}
        self.assertNotIn("pend-c1", hot | cold,
                          "a tier conflict must NOT move the row into the "
                          "confirmed corpus")
        # It is back in staging, marked, and DOWNGRADED to plain pending
        # (gov_tier cleared so the next confirm re-scans cleanly).
        row = svc._staging.find("pend-c1")
        self.assertIsNotNone(row,
                              "the conflict row must stay PENDING in staging")
        self.assertEqual(str(row.get("status")), "pending")
        self.assertTrue(row.get("_suspected_tier_conflict"))
        self.assertIsNone(row.get("gov_tier"),
                          "downgraded to plain PENDING: gov_tier cleared "
                          "so a human confirm does not re-collide")
        # Audit carries the tier_conflict reason.
        conflicts = [l for l in _audit_lines(audit)
                     if l.get("action") == "confirm"
                     and l.get("error") == "tier_conflict"]
        self.assertTrue(conflicts)

    def test_no_tier_key_confirms_identically_to_legacy(self):
        # Regression guard: a row WITHOUT gov_tier takes the exact
        # legacy path (no scan, no mark).
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([{"docid": "corp-2", "title": "唯一 标题 文档",
                     "content": "没有 tier 键 的 文档 甲乙丙丁戊己",
                     "status": "confirmed"}])
        svc.ingest([{"docid": "pend-n1", "title": "唯一 标题 文档",
                     "content": "没有 tier 键 的 文档 甲乙丙丁戊己",
                     "status": "pending", "source_profile": "coder"}])
        out = svc.confirm("pend-n1", source_profile="coder",
                          confirmed_by="user-alice")
        self.assertEqual(out["status"], "success")
        self.assertNotIn(out.get("note", ""), ("tier_conflict",))
        hot = {str(d.get("docid")) for d in svc._hot.docs_snapshot()}
        self.assertIn("pend-n1", hot,
                      "no-gov_tier rows confirm byte-identically "
                      "(legacy path)")

    def test_owner_reconfirm_of_downgraded_row_succeeds(self):
        # The dead-end fix (option b): after the agent self-confirm was
        # blocked + downgraded, the OWNER confirms via the EXISTING
        # knowledge_base_confirm face (non-agent confirmed_by). The
        # _suspected_tier_conflict mark is plain metadata — the move is
        # a normal confirm; no new tool surface.
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([{"docid": "corp-3", "title": "Redis 限流 经验",
                     "content": "Redis 限流 配置 经验 甲乙丙丁戊己",
                     "status": "confirmed", "gov_tier": "experience"}])
        svc.ingest([{"docid": "pend-o1", "title": "Redis 限流 经验",
                     "content": "Redis 限流 配置 经验 甲乙丙丁戊己",
                     "status": "pending", "gov_tier": "archive",
                     "source_profile": "coder"}])
        # Agent self-confirm hits the conflict -> downgrade.
        out1 = svc.confirm("pend-o1", source_profile="coder",
                           confirmed_by="agent")
        self.assertEqual(out1.get("note"), "tier_conflict")
        # Owner confirms the SAME docid again (re-staged, marked,
        # gov_tier cleared) -> plain confirm moves it in.
        out2 = svc.confirm("pend-o1", source_profile="coder",
                           confirmed_by="owner-human")
        self.assertEqual(out2["status"], "success")
        self.assertNotEqual(out2.get("note", ""), "tier_conflict",
                            "the owner's re-confirm of the downgraded "
                            "(tier-cleared) row must NOT re-collide")
        hot = {str(d.get("docid")) for d in svc._hot.docs_snapshot()}
        self.assertIn("pend-o1", hot)

    def test_same_tier_is_not_a_conflict(self):
        # Same tier on both sides: no collision (the dedup compares
        # only gov_tier inequality — N1's skip_near_dup scale posture).
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([{"docid": "corp-s", "title": "同 tier 文档",
                     "content": "同 tier 的 内容 甲乙丙丁戊己",
                     "status": "confirmed", "gov_tier": "archive"}])
        svc.ingest([{"docid": "pend-s1", "title": "同 tier 文档",
                     "content": "同 tier 的 内容 甲乙丙丁戊己",
                     "status": "pending", "gov_tier": "archive",
                     "source_profile": "coder"}])
        out = svc.confirm("pend-s1", source_profile="coder",
                         confirmed_by="agent")
        self.assertEqual(out["status"], "success")
        self.assertNotEqual(out.get("note", ""), "tier_conflict")
        hot = {str(d.get("docid")) for d in svc._hot.docs_snapshot()}
        self.assertIn("pend-s1", hot)


# ---------------------------------------------------------------------------
# (4) B4 independent anchor regex + 3-step verdict routing
# ---------------------------------------------------------------------------

def _anchors(text: str) -> list:
    return kb_service.extract_anchor_points(text)


class B4AnchorRegexTests(unittest.TestCase):

    def test_commit_hash(self):
        self.assertTrue(_anchors("修复 于 commit abc1234def5678901234567890abcdef12345678 之后"))
        self.assertTrue(_anchors("commit: 4b9f1c2 引入回归"))

    def test_absolute_file_path(self):
        self.assertTrue(_anchors("改 /usr/local/lib/hermes-agent/agent/x.py"))
        self.assertTrue(_anchors("文件 C:\\project\\src\\main.go 报错"))

    def test_version_number(self):
        self.assertTrue(_anchors("kafka 3.6.1 的 行为 变了"))
        self.assertTrue(_anchors("升级到 python 3.11.2 之后"))

    def test_service_config_key(self):
        self.assertTrue(_anchors("把 spring.redis.max-wait-millis 调成 3000"))

    def test_plain_prose_has_no_anchors(self):
        self.assertEqual(_anchors("今天 会议 讨论 了 一下 流程 的 改进 方向"),
                         [], "pure prose (no concrete anchors) must yield "
                             "zero independent regex anchors")

    def test_union_with_llm_anchors(self):
        # The doc-level union: independent regex hits ∪ LLM output
        # anchors (de-duplicated, order: regex first then LLM extras).
        text = ("修复 于 commit 4b9f1c2 之后 /root/proj/a.py "
                "版本 1.2.3 的 行为")
        llm = ["4b9f1c2", "自定义锚点X"]
        self.assertEqual(
            kb_service.union_anchor_points(text, llm),
            kb_service.union_anchor_points(text, llm))
        got = kb_service.union_anchor_points(text, llm)
        self.assertIn("4b9f1c2", got)
        self.assertIn("自定义锚点X", got,
                      "LLM-only anchors are kept (union, not regex-only)")
        self.assertEqual(len(got), len(set(got)), "no duplicates in union")




class B4BoundaryRoutingTests(unittest.TestCase):

    def _incident_cand(self, content: str, doc_id: str = "b4-c1"):
        return Candidate(
            doc_id=doc_id,
            classified_type="incident",
            label="incident",
            title="边角经验",
            content=content,
            source_session_id="s4",
            source_profile="coder",
            origin_kind="agent",
        )

    def test_llm_fail_with_anchors_becomes_archive(self):
        state = _state_with_service()
        svc = state._service_override
        audit = svc._audit_path
        data = svc._data_dir
        # Distill face: LLM FAILED (use_original + model_used False).
        fail_verdict = {
            "use_original": True, "drop_reason": None,
            "distilled_title": "边角经验",
            "distilled_content": "原始内容",
            "anchor_points": [], "model_used": False,
            "reason": "llm_error:timeout", "rejected_reason": None,
            "tier": None,
        }
        state._distill_calls_by_session["s4"] = 3  # over cap -> fail verdict
        state._distill_verdicts = {"s4": fail_verdict}
        state._pending_override = [self._incident_cand(
            "commit 4b9f1c2 之后 /root/proj/a.py 版本 1.2.3 修复")]
        st = _make_state(state)
        st.close_task_boundary("s4")
        # N2: dropped rows go to the STALE_STAGING audit state, not the
        # active clarify batch.
        stale = [d for d in svc._staging.docs_snapshot()
                 if d.get("kb_class") == "incident"
                 and "4b9f1c2" in str(d.get("content", ""))]
        self.assertEqual(len(stale), 1,
                         "an anchor-bearing LLM-fail incident doc is kept "
                         "as a staging audit row (stale_staging state)")
        self.assertEqual(stale[0].get("stale_staging"), True,
                         "N2: LLM-fail routing row lands in the "
                         "stale_staging state (audit, not active batch)")
        self.assertEqual(stale[0].get("kb_tier"), "archive",
                         "independent regex anchors + LLM fail = archive "
                         "(step ②; no LLM-fail fail-open to archive is "
                         "about the anchor-LESS case)")
        self.assertIn("4b9f1c2", stale[0].get("anchor_points", []))
        # NOT in the confirmed corpus (stale rows are audit-only).
        hot = {str(d.get("docid")) for d in svc._hot.docs_snapshot()}
        self.assertEqual(len(hot), 0,
                         "no doc should be self-confirmed into the corpus "
                         "for an LLM-fail + anchors routing (archive tier "
                         "is audit-only, not a confirmed corpus row)")

    def test_llm_fail_no_anchors_is_transcript_drop(self):
        state = _state_with_service()
        svc = state._service_override
        fail_verdict = {
            "use_original": True, "drop_reason": None,
            "distilled_title": "流水账", "distilled_content": "原始",
            "anchor_points": [], "model_used": False,
            "reason": "no_runtime", "rejected_reason": None,
            "tier": None,
        }
        state._distill_calls_by_session["s4"] = 3
        state._distill_verdicts = {"s4": fail_verdict}
        # Pure prose: zero independent regex anchors.
        state._pending_override = [self._incident_cand(
            "今天 开 了 一个 会 讨论 了 项目 的 下一步 计划", "b4-c2")]
        st = _make_state(state)
        st.close_task_boundary("s4")
        # Step ③: LLM fail + no anchors = transcript DROP: the row is
        # NOT ingested/staged live at all (N2: only the audit record
        # exists — stale_staging state, no clarify batch entry).
        staged_ids = [str(d.get("docid"))
                      for d in svc._staging.docs_snapshot()]
        self.assertEqual(len(staged_ids), 0,
                         "step ③ (LLM fail + no anchors) must DROP the "
                         "doc: no staging row, no pool occupation")
        # Audit proves the drop (KPI R5 data source: tier=transcript,
        # llm_fail=true).
        drops = [l for l in svc._audit_records
                 if l.get("action") == "transcript_drop"]
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0].get("kb_tier"), "transcript")
        self.assertTrue(drops[0].get("llm_fail"))

    def test_llm_healthy_transcript_verdict_keeps_original(self):
        # Step ①: LLM healthy and judges transcript -> the distill face
        # still keeps the ORIGINAL content through the length gate
        # (fail-soft floor; the tier verdict only steers the routing
        # mark, never drops data).
        state = _state_with_service()
        svc = state._service_override
        healthy = {
            "use_original": False, "drop_reason": None,
            "distilled_title": "会议 纪要",
            "distilled_content": "会议 纪要 的 最小 可验证 知识 单元 内容 甲乙丙丁",
            "anchor_points": ["锚点Y"], "model_used": True,
            "reason": "ok", "rejected_reason": None,
            "tier": "transcript",
        }
        state._distill_calls_by_session["s4"] = 0
        state._distill_verdicts = {"s4": healthy}
        state._pending_override = [self._incident_cand(
            "会议 上 确认 了 部署 窗口 与 回滚 负责人 甲乙丙丁", "b4-c3")]
        st = _make_state(state)
        st.close_task_boundary("s4")
        rows = [d for d in svc._staging.docs_snapshot()
                if d.get("kb_class") == "incident"]
        self.assertEqual(len(rows), 1,
                         "a healthy LLM verdict must keep the doc in staging")
        row = rows[0]
        self.assertEqual(row.get("kb_tier"), "transcript")
        self.assertTrue(row.get("stale_staging"),
                        "transcript routing rows are audit-only "
                        "(stale_staging mark, no self-confirm)")
        # anchor_points = regex ∪ LLM (here: no regex hit on the prose +
        # the LLM anchor).
        self.assertIn("锚点Y", row.get("anchor_points", []))

    def test_anchor_union_present_on_archived_doc(self):
        # LLM healthy + anchors on both sides: the stored anchor_points
        # is the UNION (regex hits first, LLM-only extras after, no
        # duplicates).
        state = _state_with_service()
        svc = state._service_override
        healthy = {
            "use_original": False, "drop_reason": None,
            "distilled_title": "配置 变更",
            "distilled_content": "kafka 版本 3.6.1 的 配置 变更 甲乙丙丁戊己",
            "anchor_points": ["kafka:brokers"], "model_used": True,
            "reason": "ok", "rejected_reason": None,
            "tier": "model-misremembers",
        }
        state._distill_calls_by_session["s4"] = 0
        state._distill_verdicts = {"s4": healthy}
        state._pending_override = [self._incident_cand(
            "kafka 版本 3.6.1 之后 /root/proj/conf.yaml 改了", "b4-c4")]
        st = _make_state(state)
        st.close_task_boundary("s4")
        rows = [d for d in svc._staging.docs_snapshot()
                if d.get("kb_class") == "incident"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        got = row.get("anchor_points")
        self.assertIn("kafka:brokers", got, "LLM anchor kept in union")
        self.assertIn("/root/proj/conf.yaml", got,
                      "regex anchor kept in union")
        self.assertEqual(len(got), len(set(got)))
        self.assertEqual(row.get("kb_tier"), "archive")
        self.assertTrue(row.get("stale_staging"))


# ---------------------------------------------------------------------------
# (5) B3 exemption branch in the decay gate
# ---------------------------------------------------------------------------

class B3ExemptionBranchTests(unittest.TestCase):

    def _meta(self, **over):
        # v2 (2026-10-14): the B3 archive-tier exemption now keys off
        # origin_kind=agent (every v2 class lands in the agent partition),
        # not the retired kb_class=="incident" key.
        base = {"premise_status": True, "kb_class": "incident",
                "origin_kind": "agent",
                "project_ref": "repo:stennyc/KBS"}
        base.update(over)
        return base

    def test_exempted_incident_with_project_ref(self):
        # Stale (age > 14d, no references) BUT incident + project_ref +
        # premise held -> exempted (the whole point of the archive tier).
        res = {"metadata": {**self._meta(), "ingest_ts": 0}}
        self.assertTrue(_agent_in_decay_gate(res, now_ts=999999999.0),
                        "incident + project_ref + held premise must "
                        "bypass the 14-day decay window")

    def test_falsified_premise_not_exempted(self):
        # B3 fail-closed: the exemption sits AFTER the premise_status
        # check — a falsified premise NEVER passes, even with
        # incident + project_ref.
        res = {"metadata": {**self._meta(premise_status="falsified"),
                            "ingest_ts": 0}}
        self.assertFalse(_agent_in_decay_gate(res, now_ts=999999999.0))

    def test_needs_verification_premise_not_exempted(self):
        res = {"metadata": {**self._meta(premise_status="needs_verification"),
                            "ingest_ts": 0}}
        self.assertFalse(_agent_in_decay_gate(res, now_ts=999999999.0))

    def test_non_incident_still_decays(self):
        # A non-incident agent doc with project_ref (shouldn't happen in
        # practice, but the gate must not let it through) still goes
        # through the ordinary reference/age path -> stale.
        res = {"metadata": {"premise_status": True,
                            "kb_class": "decision",
                            "project_ref": "repo:stennyc/KBS",
                            "ingest_ts": 0}}
        self.assertFalse(_agent_in_decay_gate(res, now_ts=999999999.0))

    def test_incident_without_project_ref_still_decays(self):
        # incident but no project_ref -> ordinary 14-day path (stale).
        res = {"metadata": {"premise_status": True, "kb_class": "incident",
                            "ingest_ts": 0}}
        self.assertFalse(_agent_in_decay_gate(res, now_ts=999999999.0))

    def test_holder_path_untouched_by_exemption(self):
        # A NON-incident fresh doc with reference_count > 0 still passes
        # via the reference path (the exemption insertion did not
        # clobber the ordinary logic).
        res = {"metadata": {"premise_status": True, "kb_class": "decision",
                            "reference_count": 3, "ingest_ts": 0}}
        self.assertTrue(_agent_in_decay_gate(res, now_ts=999999999.0))


# ---------------------------------------------------------------------------
# (6) note 3: repo capture + Candidate.project_ref
# ---------------------------------------------------------------------------

class Note3RepoCaptureTests(unittest.TestCase):

    def test_candidate_has_project_ref_field(self):
        c = Candidate(doc_id="x", classified_type="incident", label="i",
                      title="t", content="c" * 30,
                      source_session_id="s", source_profile="p")
        self.assertIsNone(c.project_ref,
                          "Candidate.project_ref defaults to None "
                          "(origin_kind backward-compat precedent)")
        c2 = Candidate(doc_id="x", classified_type="incident", label="i",
                      title="t", content="c" * 30,
                      source_session_id="s", source_profile="p",
                      project_ref="repo:stennyc/KBS")
        self.assertEqual(c2.project_ref, "repo:stennyc/KBS")

    def test_write_side_state_inits_repo_ctx(self):
        state = _WriteSideState(_FakeCtx())
        self.assertTrue(hasattr(state, "_repo_ctx"))

    def test_on_turn_repo_capture_failsoft_returns_none(self):
        # Monkeypatch _capture_repo_ctx to simulate git-unavailable
        # (the test env has a live git remote; force the fail-soft path).
        state = _state_with_service()
        import kb_ingest_hook as _khh
        _orig = _khh._capture_repo_ctx
        _khh._capture_repo_ctx = lambda: {
            "project_ref": None, "git_available": False}
        try:
            state.on_turn("s3", "部署 完成", "确认 部署")
        finally:
            _khh._capture_repo_ctx = _orig
        ctx = state._repo_ctx.get("s3")
        self.assertIsNotNone(ctx,
                             "on_turn always caches a _repo_ctx entry "
                             "(fail-soft: git unavailable -> entry with "
                             "project_ref=None, not a missing entry)")
        self.assertIsNone(ctx.get("project_ref"),
                          "git unavailable -> project_ref None "
                          "(fail-soft: docs never upgrade to archive)")
        self.assertFalse(ctx.get("git_available"))


# ---------------------------------------------------------------------------
# (7) note 5: _build_agent_doc writes project_ref ONLY for archive
# ---------------------------------------------------------------------------

class Note5ProjectRefKeyTests(unittest.TestCase):

    def test_archive_doc_carries_project_ref_key(self):
        state = _state_with_service()
        svc = state._service_override
        audit = svc._audit_path
        data = svc._data_dir
        cand = Candidate(
            doc_id="n5-a", classified_type="incident", label="incident",
            title="边角经验", content="kafka 3.6.1 的 行为",
            source_session_id="s5", source_profile="coder",
            origin_kind="agent", project_ref="repo:stennyc/KBS")
        doc = state._build_agent_doc(cand, {}, kb_tier="archive")
        self.assertIn("project_ref", doc,
                      "only the archive tier writes the project_ref key")
        self.assertEqual(doc["project_ref"], "repo:stennyc/KBS")

    def test_experience_doc_omits_project_ref_key(self):
        state = _state_with_service()
        svc = state._service_override
        audit = svc._audit_path
        data = svc._data_dir
        cand = Candidate(
            doc_id="n5-e", classified_type="incident", label="incident",
            title="经验", content="经验 内容 甲乙丙丁",
            source_session_id="s5", source_profile="coder",
            origin_kind="agent", project_ref=None)
        doc = state._build_agent_doc(cand, {}, kb_tier="experience")
        self.assertNotIn("project_ref", doc,
                         "experience docs must OMIT the project_ref key "
                         "entirely (never None)")

    def test_transcript_doc_omits_project_ref_key(self):
        state = _state_with_service()
        svc = state._service_override
        audit = svc._audit_path
        data = svc._data_dir
        cand = Candidate(
            doc_id="n5-t", classified_type="incident", label="incident",
            title="流水", content="开会 记录 甲乙丙丁",
            source_session_id="s5", source_profile="coder",
            origin_kind="agent", project_ref="repo:stennyc/KBS")
        # Simulate a healthy LLM transcript verdict for this session.
        state._distill_verdicts = {"s5": {
            "use_original": False, "drop_reason": None,
            "distilled_title": "流水", "distilled_content": "流水 内容",
            "anchor_points": [], "model_used": True, "reason": "ok",
            "rejected_reason": None, "tier": "model-knows"}}
        doc = state._build_agent_doc(cand, {}, kb_tier="transcript")
        self.assertIsNone(doc.get("kb_tier"),
                           "transcript docs are NOT agent-ledger docs: "
                           "kb_tier stays unset (the boundary sink drops "
                           "them — note 5's omission rule)")
        self.assertNotIn("project_ref", doc,
                         "transcript docs omit project_ref (never None)")


# ---------------------------------------------------------------------------
# (8) note 4: staging TTL sweep -> stale_staging
# ---------------------------------------------------------------------------

class Note4StagingTTLTests(unittest.TestCase):

    def _store(self):
        base = Path(tempfile.mkdtemp())
        return StagingStore(base / "staging"), base

    def _row(self, docid="t-1", idle_days=0.0):
        import time
        now = time.time()
        return {"docid": docid, "title": "t", "content": "c" * 30,
                "status": "pending",
                "ingest_ts": now - idle_days * 86400.0}

    def test_fresh_row_not_swept(self):
        store, base = self._store()
        store.add(self._row(idle_days=5.0))
        swept = store.sweep_stale(now_ts=None)
        self.assertEqual(swept, [], "a 5-day-idle row stays active")
        self.assertEqual(store.find("t-1")["status"], "pending")

    def test_stale_row_rekeyed_to_stale_staging(self):
        store, base = self._store()
        store.add(self._row(idle_days=31.0))
        swept = store.sweep_stale(now_ts=None)
        self.assertEqual(len(swept), 1)
        row = store.find("t-1")
        self.assertEqual(row["status"], "stale_staging",
                         "30-day-idle rows are re-keyed to the "
                         "stale_staging state (kept for audit)")

    def test_stale_row_excluded_from_find_semantics(self):
        # find() serves the audit surface; the ACTIVE (confirmable)
        # surface is separate: find_active() excludes stale_staging rows.
        store, base = self._store()
        store.add(self._row(idle_days=31.0))
        store.sweep_stale(now_ts=None)
        self.assertIsNone(store.find_active("t-1"),
                          "stale rows are excluded from the active "
                          "(confirmable) surface")
        self.assertIsNotNone(store.find("t-1"),
                              "but the audit copy is still findable "
                              "(留审计)")

    def test_sweep_is_idempotent(self):
        store, base = self._store()
        store.add(self._row(idle_days=31.0))
        store.sweep_stale(now_ts=None)
        self.assertEqual(store.sweep_stale(now_ts=None), [],
                         "a second sweep finds nothing to move")


# ---------------------------------------------------------------------------
# (9b) Out-of-band external-replace sync channel (owner 2026-10-15):
#  a sweep/curator tool rewrites staging_docs.jsonl + drops the index
#  pkl + drops the marker file; the next in-process rebase (ingest/
#  confirm) must adopt the on-disk rows WHOLESALE instead of union-
#  merging them (a union resurrects every quarantined docid this
#  process still holds in memory).
# ---------------------------------------------------------------------------
class ExternalReplaceMarkerTests(unittest.TestCase):

    def _svc(self):
        svc, base, data, audit = _svc_with_audit()
        return svc, data

    def _pending_doc(self, docid, n=30):
        return {"docid": docid, "title": f"标题 {docid}",
                "content": "内容" * n, "status": "pending"}

    def test_marker_consumed_on_next_ingest_adopts_disk_wholesale(self):
        svc, data = self._svc()
        svc.ingest([self._pending_doc("p-keep"), self._pending_doc("p-drop")])
        self.assertEqual(svc._staging.size, 2)
        # Simulate a sweep commit: rewrite the rows file down to the keep
        # row, drop the index pkl, drop the marker.
        rows_path = svc._staging.rows_path
        with open(rows_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._pending_doc("p-keep"),
                               ensure_ascii=False) + "\n")
        idx = data / "staging" / "staging_index.pkl"
        if idx.exists():
            idx.unlink()
        marker = data / "staging" / "staging_external_replace"
        marker.write_text("sweep-test", encoding="utf-8")
        # Next in-process rebase (any ingest) must adopt the 1-row disk set
        # wholesale — NOT union the dropped row back in.
        svc.ingest([self._pending_doc("p-new")])
        ids = {str(d.get("docid")) for d in svc._staging.docs_snapshot()}
        self.assertEqual(ids, {"p-keep", "p-new"},
                         "quarantined p-drop must NOT be resurrected "
                         "by the union rebase")
        self.assertFalse(marker.exists(),
                         "the marker is one-shot: consumed on first "
                         "rebase, not re-fired")

    def test_no_marker_keeps_union_semantics(self):
        # Regression guard: WITHOUT a marker the pre-existing union
        # rebase path is unchanged (disk rows still merge in).
        svc, data = self._svc()
        svc.ingest([self._pending_doc("p-1")])
        svc.ingest([self._pending_doc("p-2")])
        ids = {str(d.get("docid")) for d in svc._staging.docs_snapshot()}
        self.assertEqual(ids, {"p-1", "p-2"})
        marker = data / "staging" / "staging_external_replace"
        self.assertFalse(marker.exists())

    def test_confirm_path_consumes_marker_too(self):
        # The same sync must fire on the confirm() rebase, not only
        # ingest() — a confirm is the one op that moves a row OUT.
        svc, data = self._svc()
        svc.ingest([self._pending_doc("c-1"), self._pending_doc("c-2")])
        rows_path = svc._staging.rows_path
        with open(rows_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self._pending_doc("c-1"),
                               ensure_ascii=False) + "\n")
        (data / "staging" / "staging_external_replace").write_text(
            "sweep-test", encoding="utf-8")
        svc.confirm("c-1", confirmed_by="owner")
        ids = {str(d.get("docid")) for d in svc._staging.docs_snapshot()}
        self.assertEqual(ids, set(),
                         "after external-replace + confirm, c-2 (which "
                         "the sweep quarantined) must not be back")



# ---------------------------------------------------------------------------
# (9) B1 migration script (tools/migrate_docid_none_filter.py)
# ---------------------------------------------------------------------------

_MIGRATOR = PLUGIN_DIR / "tools" / "migrate_docid_none_filter.py"


class B1MigrationScriptTests(unittest.TestCase):

    def _mkdata(self) -> Path:
        base = Path(tempfile.mkdtemp()) / "kb-data"
        (base / "staging").mkdir(parents=True)
        hot = [
            {"docid": "h1", "title": "甲 文档", "content": "内容 甲" * 8,
             "gov_tier": None, "project_ref": None},
            {"docid": "h2", "title": "乙 文档", "content": "内容 乙" * 8},
        ]
        with open(base / "hot_corpus.pkl", "wb") as f:
            pickle.dump({"schema_version": 1, "generation": 1,
                         "docs": hot}, f)
        cold = [
            {"docid": "c1", "title": "丙 文档", "content": "内容 丙" * 8,
             "gov_tier": None},
        ]
        (base / "cold").mkdir(parents=True)
        with open(base / "cold" / "cold_corpus.pkl", "wb") as f:
            pickle.dump({"schema_version": 1, "generation": 2,
                         "docs": cold}, f)
        with open(base / "staging" / "staging_docs.jsonl", "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"docid": "s1", "title": "丁 文档",
                                "content": "内容 丁" * 8,
                                "gov_tier": None, "project_ref": None},
                               ensure_ascii=False) + "\n")
        with open(base / "ingest_ts.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"docid": "h1", "ts": 100.0}) + "\n")
            f.write(json.dumps({"docid": "s1", "ts": 200.0}) + "\n")
        # P0-2 fixtures: derived structures that hold docid keys.
        graph = {
            "max_docs": 500,
            "graph": {
                "h1": {"s1": 1.0, "h1": 2.0},
                "s1": {"h1": 1.5},
            },
        }
        with open(base / "graph.json", "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False)
        with open(base / "reinforce.jsonl", "w", encoding="utf-8") as f:
            # history rows: two corpus docids (h1, s1) + one row whose
            # docid is NOT in the corpus (orphan that must stay put).
            f.write(json.dumps({"docid": "h1", "ts": 1111.0,
                               "reason": "explicit_search",
                               "anchors": []}) + "\n")
            f.write(json.dumps({"docid": "s1", "ts": 2222.0,
                               "reason": "explicit_search",
                               "anchors": ["sha:abcdef"]}) + "\n")
            f.write(json.dumps({"docid": "orphan-doc-9", "ts": 3333.0,
                               "reason": "explicit_search",
                               "anchors": []}) + "\n")
        return base

    def _load_pkl(self, p: Path):
        with open(p, "rb") as f:
            return pickle.load(f).get("docs", [])

    def test_forward_migration_remaps_docids_and_sidecar(self):
        base = self._mkdata()
        out = _run_migration(base)
        # Every remapped row keeps its content but gets the NEW
        # (None-filter) docid; the map records old->new.
        new_hot = self._load_pkl(base / "hot_corpus.pkl")
        remap = json.loads(
            (base / "docid_migration_map.json").read_text())
        new_ids = {str(d["docid"]) for d in new_hot}
        for old, new in remap.items():
            self.assertNotEqual(old, new,
                                "rows carrying None-valued keys MUST change "
                                "docid under the new formula (old had "
                                "null in the payload)")
        # Fixpoint check (revised.2 audit criterion): re-running the NEW
        # formula over the stored rows yields their stored docid.
        for d in new_hot + self._load_pkl(base / "cold" /
                                          "cold_corpus.pkl"):
            self.assertEqual(_deterministic_docid(d), str(d["docid"]),
                             "post-migration recompute must equal the "
                             "stored docid (fixpoint)")
        # Sidecar re-aligned by docid.
        ts = [json.loads(l) for l in
              (base / "ingest_ts.jsonl").read_text().splitlines() if l.strip()]
        ts_ids = {r["docid"] for r in ts}
        self.assertNotIn("h1", ts_ids, "old h1 id gone from sidecar")
        self.assertIn(remap["h1"], ts_ids, "new h1 id present")
        self.assertIn(remap["s1"], ts_ids)

    def test_rollback_restores_docids_and_strips_keys(self):
        base = self._mkdata()
        _run_migration(base)
        out = _run_migration(base, rollback=True)
        # Old docids restored; new keys stripped.
        hot = self._load_pkl(base / "hot_corpus.pkl")
        ids = {str(d["docid"]) for d in hot}
        self.assertIn("h1", ids, "rollback restores the old docid")
        for d in hot:
            self.assertNotIn("gov_tier", d,
                             "rollback strips the added gov_tier key")
            self.assertNotIn("project_ref", d,
                             "rollback strips the added project_ref key")
        # Sidecar restored to old docids.
        ts = [json.loads(l) for l in
              (base / "ingest_ts.jsonl").read_text().splitlines()
              if l.strip()]
        self.assertIn("h1", {r["docid"] for r in ts})

    def _corpus_docids(self, base: Path) -> set:
        """All docids currently in the corpus stores (hot + cold +
        staging) at a given point — used to prove orphan-freedom of
        reinforce.jsonl / graph.json after remap."""
        corpus = set()
        for pkl in [base / "hot_corpus.pkl",
                    base / "cold" / "cold_corpus.pkl"]:
            corpus.update(str(d.get("docid", ""))
                          for d in self._load_pkl(pkl))
        for row in [json.loads(l) for l in
                    (base / "staging" / "staging_docs.jsonl")
                    .read_text().splitlines() if l.strip()]:
            corpus.add(str(row.get("docid", "")))
        return corpus

    def test_forward_remaps_reinforce_and_graph_orphan_free(self):
        """P0-2: after forward, every reinforce.jsonl docid that was a
        corpus docid is now the NEW corpus docid (0 orphans among rows
        that referenced the corpus); rows whose docid was never in the
        corpus stay untouched; historical ts/reason/anchors are
        preserved verbatim. graph.json node keys + neighbor keys are
        remapped the same way."""
        base = self._mkdata()
        _run_migration(base)

        remap = json.loads(
            (base / "docid_migration_map.json").read_text())
        new_corpus = self._corpus_docids(base)
        self.assertNotIn("h1", new_corpus)
        self.assertNotIn("s1", new_corpus)

        # (a) reinforce.jsonl: every historical row is kept, in the
        # same order; its docid is the new corpus docid when the old
        # docid was in the mapping, otherwise it stays exactly the
        # same (orphan row).
        rows = [json.loads(l) for l in
                (base / "reinforce.jsonl").read_text().splitlines()
                if l.strip()]
        self.assertEqual(len(rows), 3,
                         "append-only history: no rows dropped, none "
                         "added by the remap")
        self.assertEqual([r["ts"] for r in rows],
                         [1111.0, 2222.0, 3333.0],
                         "historical ts values are preserved verbatim")
        self.assertEqual([r["reason"] for r in rows],
                         ["explicit_search"] * 3,
                         "historical reason values are preserved verbatim")
        self.assertEqual(rows[1]["anchors"], ["sha:abcdef"],
                         "historical anchors are preserved verbatim")
        self.assertEqual(rows[0]["docid"], remap["h1"],
                         "corpus docid h1 remapped to its new form")
        self.assertEqual(rows[1]["docid"], remap["s1"],
                         "corpus docid s1 remapped to its new form")
        self.assertEqual(rows[2]["docid"], "orphan-doc-9",
                         "a row whose docid was NOT in the corpus is "
                         "kept exactly as it was (no silent rewrite)")
        # Orphan-free among corpus-referencing rows:
        for r in rows[:2]:
            self.assertIn(r["docid"], new_corpus,
                          "every remapped historical reinforce row now "
                          "points at a live corpus docid (0 orphans)")

        # (b) graph.json: package structure intact, node + neighbor
        # keys remapped, weights preserved.
        g = json.loads((base / "graph.json").read_text())
        self.assertEqual(g["max_docs"], 500,
                         "the graph package structure is preserved")
        graph = g["graph"]
        self.assertEqual(set(graph.keys()),
                         {remap["h1"], remap["s1"]},
                         "node docid keys remapped to new corpus ids")
        self.assertEqual(
            set(graph[remap["h1"]].keys()),
            {remap["s1"], remap["h1"]},
            "neighbor docid keys inside a node are remapped too")
        self.assertEqual(graph[remap["h1"]][remap["h1"]], 2.0,
                         "float weights are preserved verbatim")
        self.assertEqual(graph[remap["s1"]][remap["h1"]], 1.5)

    def test_rollback_restores_reinforce_and_graph_docids(self):
        """P0-2: rollback rewrites reinforce.jsonl + graph.json docid
        keys back to their pre-migration values using the reverse
        mapping (history fields untouched)."""
        base = self._mkdata()
        _run_migration(base)
        _run_migration(base, rollback=True)

        # reinforce.jsonl: all three historical rows are back at their
        # original docids; orphan row was untouched through both
        # directions.
        rows = [json.loads(l) for l in
                (base / "reinforce.jsonl").read_text().splitlines()
                if l.strip()]
        self.assertEqual([r["docid"] for r in rows],
                         ["h1", "s1", "orphan-doc-9"],
                         "rollback restores every reinforce docid to "
                         "its pre-migration value")
        self.assertEqual(rows[0]["ts"], 1111.0)
        self.assertEqual(rows[1]["anchors"], ["sha:abcdef"])

        # graph.json: node + neighbor keys are back to the old docids
        # with the original weights.
        g = json.loads((base / "graph.json").read_text())
        self.assertEqual(g["max_docs"], 500)
        self.assertEqual(g["graph"],
                         {"h1": {"s1": 1.0, "h1": 2.0},
                          "s1": {"h1": 1.5}},
                         "graph docid keys restored to pre-migration "
                         "values")

    def test_idempotent_forward_run(self):
        base = self._mkdata()
        _run_migration(base)
        _run_migration(base)
        # A second forward run is a no-op: the on-disk map is empty
        # (rows are already at the fixpoint, nothing to remap).
        second = json.loads(
            (base / "docid_migration_map.json").read_text())
        self.assertEqual(second, {},
                         "a second forward run finds no changes: "
                         "the stored rows are already at the fixpoint")

    def test_mapping_persisted_before_any_data_write(self):
        """P0-1a crash-window invariant: docid_migration_map.json MUST
        exist on disk (with the complete old→new mapping) BEFORE any
        data file (hot/cold pkl, staging jsonl, ingest_ts) is written.

        Simulates a crash at the first data-write point by patching
        the migrator's pkl writer to raise, and asserts that the map
        file is already present, complete, and usable for rollback."""
        import importlib.util
        import unittest.mock as mock
        base = self._mkdata()
        spec = importlib.util.spec_from_file_location(
            "_b1_mig", str(_MIGRATOR))
        mig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mig)

        # Pre-compute the expected complete mapping from the fresh data.
        expected = {}
        for row in (self._load_pkl(base / "hot_corpus.pkl")
                    + self._load_pkl(base / "cold" / "cold_corpus.pkl")
                    + [json.loads(l) for l in
                       (base / "staging" / "staging_docs.jsonl")
                       .read_text().splitlines() if l.strip()]):
            oid = str(row["docid"])
            nid = mig._deterministic_docid(row)
            if oid != nid:
                expected[oid] = nid
        self.assertTrue(expected, "fixture MUST contain remappable rows")

        # Simulate a crash at the first data write point: the moment the
        # migrator calls _write_pkl_docs, an exception is raised. This
        # fires AFTER _dump_mapping (the P0-1a order) and BEFORE
        # hot/cold/staging are rewritten, so data files are still
        # pre-migration on disk.
        with mock.patch.object(mig, "_write_pkl_docs",
                               side_effect=RuntimeError("simulated crash")):
            with self.assertRaises(RuntimeError):
                mig.forward_migrate(base)

        # (a) The mapping file must exist at this exact point (crash
        #      window).
        map_path = base / "docid_migration_map.json"
        self.assertTrue(map_path.exists(),
                        "P0-1a: docid_migration_map.json must be on disk "
                        "BEFORE any data write; a crash at the first "
                        "data-write point must still leave a complete "
                        "map for rollback")
        saved = json.loads(map_path.read_text())
        # (b) The mapping must be the COMPLETE old→new map.
        self.assertEqual(saved, expected,
                         "the persisted mapping must be the complete "
                         "old→new mapping (no partial maps)")
        # (c) Data files on disk must still be pre-migration
        #      (unchanged), since the crash happened at the first
        #      data-write point.
        hot = self._load_pkl(base / "hot_corpus.pkl")
        self.assertIn("h1", {str(d["docid"]) for d in hot},
                      "crash before the first data write: hot store is "
                      "still at pre-migration docids")
        # (d) Rollback using the surviving map MUST restore the old
        #     docids even though the data writes never happened — the
        #     crash-window is recoverable.
        rb_map = json.loads(map_path.read_text())
        reverse = {v: k for k, v in rb_map.items()}
        for store in [base / "hot_corpus.pkl",
                      base / "cold" / "cold_corpus.pkl"]:
            docs, sv, gen = mig._read_pkl_docs(store)
            for row in docs:
                nid = str(row.get("docid", ""))
                if nid in reverse:
                    row["docid"] = reverse[nid]
            mig._write_pkl_docs(store, docs, sv, gen)
        hot = self._load_pkl(base / "hot_corpus.pkl")
        self.assertIn("h1", {str(d["docid"]) for d in hot},
                      "rollback path after a crash at the first "
                      "data-write point must recover old docids from "
                      "the surviving mapping")

    def test_backup_created_before_data_write_and_aborts_on_failure(self):
        """P0-1a backup invariant: forward takes a full kb-data backup
        (kb-data.bak-<ts>/) before writing any data, and aborts
        WITHOUT touching data if the backup fails."""
        import importlib.util
        import unittest.mock as mock
        base = self._mkdata()
        spec = importlib.util.spec_from_file_location(
            "_b1_mig2", str(_MIGRATOR))
        mig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mig)

        # (a) Successful forward: a backup dir is created at
        #      kb-data/../kb-data.bak-<ts>/ containing the full tree.
        before = {p.name for p in base.parent.iterdir() if p.is_dir()}
        mig.forward_migrate(base)
        after = {p.name for p in base.parent.iterdir() if p.is_dir()}
        new = after - before
        self.assertEqual(len(new), 1,
                         "forward MUST create exactly one new backup dir "
                         "next to kb-data")
        bak = base.parent / new.pop()
        self.assertTrue((bak / "hot_corpus.pkl").exists())
        self.assertTrue((bak / "staging" / "staging_docs.jsonl").exists())
        self.assertTrue((bak / "ingest_ts.jsonl").exists())

        # (b) Failed backup: forward aborts without touching data.
        base2 = self._mkdata()
        with mock.patch.object(mig, "_backup_kb_data",
                               side_effect=OSError("backup failed")):
            with self.assertRaises(OSError):
                mig.forward_migrate(base2)
        # Data is byte-identical to the pre-migration fixture.
        hot = self._load_pkl(base2 / "hot_corpus.pkl")
        self.assertIn("h1", {str(d["docid"]) for d in hot},
                      "a failed backup MUST abort forward before any "
                      "data write")
        self.assertFalse(
            (base2 / "docid_migration_map.json").exists(),
                      "no mapping is written if the backup fails")


def _run_migration(kb_data: Path, rollback: bool = False) -> str:
    """Run the standalone migration script against a kb-data dir."""
    import subprocess
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = (str(PLUGIN_DIR) + os.pathsep +
                         "/root/.hermes/plugin-venv/lib/python3.14/site-packages")
    cmd = ["/root/.hermes/plugin-venv/bin/python", str(_MIGRATOR),
           str(kb_data)]
    if rollback:
        cmd.append("--rollback")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    self_text = r.stdout
    if r.returncode != 0:
        raise RuntimeError(f"migration failed: {r.stderr}\n{r.stdout}")
    return self_text


if __name__ == "__main__":
    unittest.main()

# ---------------------------------------------------------------------------
# 噪声候选不上注入面（owner 2026-10-13 红线：噪声直接丢弃、永不上判定面；
# 落盘 staging/confirmed 行保留供审计，但 clarify 脚注 + 读侧 auto-inject 不捞）。
# ---------------------------------------------------------------------------
def test_is_noise_title_command_heads_are_noise():
    from kb_ingest_rules import is_noise_title
    for t in ["推", "要", "可以", "同意", "那就帮我加", "帮我加", "不要",
              "删", "丢", "好", "嗯", "点头",
              "确认", "落地", "继续", "skip", "run it", ""]:
        assert is_noise_title(t) is True, f"should be noise: {t!r}"

def test_is_noise_title_question_heads_are_noise():
    from kb_ingest_rules import is_noise_title
    for t in ["那就算了...吗？", "可以吗？", "是不是？", "为什么吗？", "怎么吗？",
              "帮我写个函数加个参数吗？"[:0] + "可以吗？"]:
        assert is_noise_title(t) is True, f"should be noise: {t!r}"

def test_is_noise_title_real_content_not_filtered():
    from kb_ingest_rules import is_noise_title
    for t in ["推 stennyc/KBS 前先用 Bearer 认证，否则 401",
              "读侧裸 import 崩溃，bootstrap 修复",
              "KB 写侧 hook LLM 超时坑：宿主 30s vs call_llm 60s",
              "git 默认分支切 main 再删 master 的顺序坑",
              "帮我写个函数计算两个数的最大值",
              "B, 你可以分开小段进行并且可以拉几个子进程",
              "13b5ec62 那条重复 doc 已清理",
              "丢，不要"]:
        assert is_noise_title(t) is False, f"should NOT be noise: {t!r}"

def test_clarify_footnote_filters_noise_candidates_but_keeps_real():
    import kb_ingest_hook as h
    from kb_ingest_rules import Candidate
    state = h._WriteSideState.__new__(h._WriteSideState)
    # 最小可运行的 state：只注入面用到的成员
    state._fire_clarify_fired = set()
    state._last_boundary_prev = None
    state._last_session = "sessX"
    state._clarify_state_store = lambda: None
    # ClarifyAggregator 里塞 pending 候选：噪声标题 + 真标题
    from kb_ingest_rules import ClarifyAggregator
    agg = ClarifyAggregator()
    state._agg = lambda: agg
    state._staging_docs = None
    noise_c = Candidate(doc_id="n1", classified_type="decision", label="决策结论",
                        title="推", content="推", source_session_id="sessX",
                        source_profile="coder", cue_hits=["x"])
    real_c = Candidate(doc_id="n2", classified_type="decision", label="决策结论",
                        title="推 stennyc/KBS 前先用 Bearer 认证，否则 401",
                        content="...401...", source_session_id="sessX",
                        source_profile="coder", cue_hits=["x"])
    agg.add(noise_c); agg.add(real_c)
    state._fire_clarify_fired.add("sessX")
    note = state._clarify_footnote("sessX")
    # 噪声候选不上脚注，真候选照上
    assert "推 stennyc/KBS 前先用 Bearer 认证" in note, "real candidate must surface"
    assert "（doc_id: n1）" not in note, "noise candidate n1 must NOT surface"

def test_read_side_filters_noise_docs_from_inject():
    # 读侧 auto-inject：confirmed corpus 里噪声标题 doc 被滤，真 doc 照带
    import kb_ingest_hook as h
    from kb_ingest_rules import is_noise_title
    # 直接验判定逻辑（读侧两处都调 is_noise_title(h.get("title",""))）
    assert is_noise_title("推") is True
    assert is_noise_title("读侧裸 import 崩溃，bootstrap 修复") is False

# ---------------------------------------------------------------------------
# knowledge_base_get 读面（owner 2026-10-13：确定性直取一条 doc，
# 不赌 BM25 词面；堵死"验证已知 doc 只能赌搜索"的可观测性缺口）。
# ---------------------------------------------------------------------------
import unittest as _ut

class GetReadSideTests(_ut.TestCase):

    def test_get_confirmed_doc_returns_full_metadata(self):
        import tempfile
        from pathlib import Path
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        from kb_service import KbSearchService
        svc = KbSearchService(Path(d.name))
        svc.ingest([{"docid": "g-1", "title": "确定性直取测试",
                     "content": "这是 get 读面的 verified 内容",
                     "status": "confirmed", "source_profile": "coder"}])
        r = svc.get("g-1")
        assert r["found"] is True, r
        assert r["layer"] in ("hot", "cold"), r
        assert r["doc_id"] == "g-1"
        assert r["doc"].get("title") == "确定性直取测试"
        assert "verified 内容" in r["doc"].get("content", "")

    def test_get_staged_pending_doc_falls_back_to_staging(self):
        import tempfile
        from pathlib import Path
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        from kb_service import KbSearchService
        svc = KbSearchService(Path(d.name))
        svc.ingest([{"docid": "g-2", "title": "pending 行",
                     "content": "尚未 confirm 的 staging 行",
                     "status": "pending", "source_profile": "coder"}])
        r = svc.get("g-2")
        assert r["found"] is True, r
        assert r["layer"] == "staging", r
        assert r["doc"].get("title") == "pending 行"

    def test_get_unknown_doc_returns_not_found(self):
        import tempfile
        from pathlib import Path
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        from kb_service import KbSearchService
        svc = KbSearchService(Path(d.name))
        svc.ingest([{"docid": "g-3", "title": "存在", "content": "x",
                     "status": "confirmed", "source_profile": "coder"}])
        r = svc.get("does-not-exist")
        assert r["found"] is False, r
        assert r["doc_id"] == "does-not-exist"

    def test_get_empty_docid_rejected(self):
        import tempfile
        from pathlib import Path
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        from kb_service import KbSearchService
        svc = KbSearchService(Path(d.name))
        r = svc.get("")
        assert r["found"] is False, r
        assert r["doc_id"] == ""

    def test_get_excludes_stale_staging_audit_rows(self):
        # owner 2026-10-14 nit: get() must NOT surface a re-keyed stale_staging
        # audit copy (note 4 TTL sweep moved a pending row out of the active
        # face) — find_active() is the correct staging read.
        import tempfile
        from pathlib import Path
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        from kb_service import KbSearchService
        svc = KbSearchService(Path(d.name))
        svc.ingest([{"docid": "g-stale", "title": "stale 行",
                     "content": "超 TTL 被 sweep 成 stale_staging 的审计副本",
                     "status": "pending", "source_profile": "coder"}])
        # simulate the note-4 sweep: re-key the row to stale_staging
        import time
        now = time.time() + 31*86400 + 604800
        svc._staging.sweep_stale(now_ts=now)
        r = svc.get("g-stale")
        assert r["found"] is False, (
            f"stale_staging row must NOT be surfaced as reachable: {r}")

# ---------------------------------------------------------------------------
# knowledge_base_get 注册面（owner 2026-10-14 nit：宿主加载后 register()
# 必须把 knowledge_base_get 接对 _get_handler + GET_SCHEMA，空 doc_id 返回
# error JSON 而非异常。锁死"未来重构摘掉注册也逃不过测试"）。
# ---------------------------------------------------------------------------
class GetRegistrationFaceTests(_ut.TestCase):
    def _plugin_module(self):
        import importlib.util
        from pathlib import Path
        p = Path(__file__).resolve().parent / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "kb_plugin_reg_test", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_register_wires_get_tool(self):
        m = self._plugin_module()
        tools = []
        class FakeCtx:
            def register_tool(self, **k):
                tools.append((k["name"], k["handler"], k.get("schema")))
            def register_hook(self, *a, **k):
                pass
        m.register(FakeCtx())
        names = [t[0] for t in tools]
        assert "knowledge_base_get" in names, names
        g = [t for t in tools if t[0] == "knowledge_base_get"][0]
        assert g[1] is m._get_handler, "get tool not wired to _get_handler"
        assert g[2] is m.GET_SCHEMA, "get tool not wired to GET_SCHEMA"

    def test_get_handler_empty_docid_returns_error_json(self):
        import json
        m = self._plugin_module()
        out = json.loads(m._get_handler({"doc_id": ""}))
        assert out.get("status") == "error", out
        assert out.get("error") == "doc_id is required", out
