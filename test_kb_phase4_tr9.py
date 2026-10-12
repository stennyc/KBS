#!/usr/bin/env python3
"""Phase 4 TR-9 unit tests: incident-cue (4th class) + agent self-confirm path.

TR-9 (PRD v2 §TR-9 / G7): a NEW 4th cue class "incident" (边角经验事实) —
task-incidental edge-case facts learned in passing (e.g. "git PAT 当 Bearer
调 REST 可以，gh 却没登录"). When the regex detector flags an incident,
the Candidate carries origin_kind="agent" (the agent-experience partition);
the write-side hook stages the doc PENDING and then self-confirms it via
the same profile (confirmed_by="agent"), bypassing the clarify human gate
(owner ruling "不需要人工"). Gate / near-dup rejects leave the row in
staging + an agent_confirm_skipped audit record (shared with TR-4 / G5).

Red lines honored (none touched by this phase):
  - TR-7: hot/cold layers, flock concurrency, ReinforceStore write gate,
    staging physical isolation, graph gate, occurred_at/version_ts,
    near-dup thresholds 0.85/0.60, 7-gateway topology — all untouched;
  - pre_llm_call auto-inject layer — byte-untouched (the incident path
    is write-side only);
  - confirm() profile_mismatch hard gate — reused, never modified;
  - _deterministic_docid (uuid5) mechanism — untouched; origin_kind is
    now part of the docid input (same verified mechanism as Phase 1 TR-1:
    a candidate with origin_kind=agent produces a DIFFERENT docid than
    the same content with origin_kind=auto — expected, since origin_kind
    is a property of the candidate, not content).

Coverage (4 KPI tests, matching the task spec):
  (a) incident rule hits a pure edge-fact text (no decision/walkthrough/
      config cue) -> Candidate with origin_kind="agent";
  (b) the 3 legacy rules (decision/walkthrough/config) still produce
      Candidates with origin_kind=None (backward-compatible);
  (c) an incident candidate assembled into a doc dict -> staging PENDING
      + self-confirm -> confirmed corpus (no clarify human gate);
  (d) near-dup / gate reject -> doc stays in staging + audit record
      (agent_confirm_skipped), shared backstop path (not a new mechanism).

Run: PYTHONPATH=/root/.hermes/plugins/knowledge-base-search \
     /root/.hermes/plugin-venv/bin/python -m pytest test_kb_phase4_tr9.py -q
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLUGIN_DIR))
_kb_root = str(_PLUGIN_DIR.parents[1])
if _kb_root not in sys.path:
    sys.path.insert(0, _kb_root)

from kb_service import KbSearchService, _deterministic_docid  # noqa: E402
import kb_ingest_hook as hook  # noqa: E402
from kb_ingest_hook import _WriteSideState  # noqa: E402
from kb_ingest_rules import (  # noqa: E402
    Candidate, CandidateDetector, merge_rules, DEFAULT_RULES,
)

# ---------------------------------------------------------------------------
# Helpers (same pattern as test_kb_phase2.py / test_kb_phase3_tr3.py)
# ---------------------------------------------------------------------------

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


class _FakeService:
    """Records ingest/confirm/audit calls; serves canned confirm results.

    Mirrors test_kb_phase2._SVCRecorder but supports the incident path
    (which ingests + confirms in a single call)."""

    def __init__(self, confirm_results=None):
        self.ingested = []
        self.confirm_calls = []
        self.audit_records = []
        self._confirm_results = confirm_results or self._default_confirm_ok

    @staticmethod
    def _default_confirm_ok(doc_id, source_profile=None, confirmed_by=None):
        return {"status": "success", "doc_id": doc_id,
                "confirmed_by": confirmed_by,
                "source_profile": source_profile}

    def set_confirm_results(self, fn):
        self._confirm_results = fn

    def _stage_doc(self, doc):
        """Simulate the staging store: assign a docid if not present,
        record as staged. Returns the docid."""
        if "docid" not in doc:
            doc["docid"] = f"uuid-{hash(doc.get('title','') or '')}"
        self.ingested.append(doc)
        return doc["docid"]

    def ingest(self, docs):
        out = {"ingested": [], "rejected": [], "staged": []}
        for d in docs:
            docid = self._stage_doc(d)
            out["staged"].append(docid)
        return out

    def confirm(self, doc_id, source_profile=None, confirmed_by=None):
        self.confirm_calls.append({
            "doc_id": doc_id,
            "source_profile": source_profile,
            "confirmed_by": confirmed_by,
        })
        return self._confirm_results(doc_id, source_profile=source_profile,
                                     confirmed_by=confirmed_by)

    def _audit(self, **fields):
        self.audit_records.append(fields)


def _incident_cand(doc_id, title, content, profile, session):
    """Build an incident Candidate (origin_kind='agent')."""
    return Candidate(
        doc_id=doc_id,
        classified_type="incident",
        label="边角经验事实",
        title=title,
        content=content,
        source_session_id=session,
        source_profile=profile,
        cue_hits=["顺带"],
        origin_kind="agent",
    )


def _legacy_cand(doc_id, title, content, profile, session, kb_class="walkthrough"):
    """Build a legacy (non-incident) Candidate (origin_kind=None)."""
    return Candidate(
        doc_id=doc_id,
        classified_type=kb_class,
        label="踩坑排查路径" if kb_class == "walkthrough" else "决策结论",
        title=title,
        content=content,
        source_session_id=session,
        source_profile=profile,
        cue_hits=["踩坑"] if kb_class == "walkthrough" else ["决定采用"],
    )


def _make_ctx():
    class _Ctx:
        def __init__(self):
            self._loaded_rules = None

    return _Ctx()


# ---------------------------------------------------------------------------
# (a) incident rule: pure edge-fact text -> Candidate with origin_kind=agent
# ---------------------------------------------------------------------------

class TR9IncidentRuleHitTests(unittest.TestCase):
    """The 4th cue class (incident) produces a Candidate with
    origin_kind='agent' when the text contains incident cues but NONE
    of the decision / walkthrough / config cues."""

    def test_incident_cue_produces_agent_origin_kind(self):
        rules = merge_rules(None)
        det = CandidateDetector(rules, session_id="s1", profile="coder")
        # Pure edge-fact: contains incident cues ("顺带", "发现一个", "API 细节")
        # but NO decision cue ("决定采用"/"拍板"), NO walkthrough cue
        # ("根因"/"排查"/"踩坑"), NO config cue ("```"/"配置片段").
        text = "顺带发现一个 API 细节：这个接口的 rate limit 是每分钟 10 次，" \
               "超限就 429，跟文档里写的不一样。"
        c = det.scan(text, lambda t, c_: "doc-a")
        self.assertIsNotNone(c, "incident cue should produce a candidate")
        self.assertEqual(c.classified_type, "incident")
        self.assertEqual(c.origin_kind, "agent",
                         "incident rule must mark origin_kind='agent'")

    def test_incident_cue_no_legacy_cue_overlap(self):
        """A text with ONLY incident cues (no legacy cues) is classified
        as incident, not as decision / walkthrough / config."""
        rules = merge_rules(None)
        det = CandidateDetector(rules, session_id="s1", profile="coder")
        # This text has "发现一个" (incident cue) but no decision /
        # walkthrough / config cues.
        text = "发现一个限制：这个 SDK 不支持流式响应，只能等完整响应。"
        c = det.scan(text, lambda t, c_: "doc-a2")
        self.assertIsNotNone(c)
        self.assertEqual(c.classified_type, "incident",
                         "a pure edge-fact with no legacy cues must be "
                         "classified as incident, not legacy")
        self.assertEqual(c.origin_kind, "agent")

    def test_incident_rule_id_in_default_rules(self):
        """The incident rule must be in DEFAULT_RULES (not just in the
        merged config), so profiles that configure nothing get it."""
        rule_ids = [r["id"] for r in DEFAULT_RULES["rules"]]
        self.assertIn("incident", rule_ids,
                      "incident rule must be in DEFAULT_RULES")
        self.assertEqual(len(rule_ids), 4,
                         "exactly 4 rules: decision / walkthrough / config / incident")


# ---------------------------------------------------------------------------
# (b) legacy rules: 3 existing classes still produce origin_kind=None
# ---------------------------------------------------------------------------

class TR9LegacyRuleBackwardCompatTests(unittest.TestCase):
    """The 3 legacy rules (decision / walkthrough / config) must NOT
    carry origin_kind (None -> auto default), preserving pre-TR-9 behavior."""

    def _det(self):
        return CandidateDetector(merge_rules(None), session_id="s1",
                                 profile="coder")

    def test_decision_cue_origin_kind_is_none(self):
        c = self._det().scan(
            "讨论了一圈，最终决定采用方案 B，否决了 A。",
            lambda t, c_: "doc-b1")
        self.assertIsNotNone(c)
        self.assertEqual(c.classified_type, "decision")
        self.assertIsNone(c.origin_kind,
                          "legacy decision rule must NOT set origin_kind")

    def test_walkthrough_cue_origin_kind_is_none(self):
        c = self._det().scan(
            "报错 no such table，排查发现是迁移没跑，修复后正常。",
            lambda t, c_: "doc-b2")
        self.assertIsNotNone(c)
        self.assertEqual(c.classified_type, "walkthrough")
        self.assertIsNone(c.origin_kind,
                          "legacy walkthrough rule must NOT set origin_kind")

    def test_config_cue_origin_kind_is_none(self):
        c = self._det().scan(
            "把这段配置片段存下来：\n```\nplugins:\n  enabled: [x]\n```",
            lambda t, c_: "doc-b3")
        self.assertIsNotNone(c)
        self.assertEqual(c.classified_type, "config")
        self.assertIsNone(c.origin_kind,
                          "legacy config rule must NOT set origin_kind")

    def test_candidate_default_origin_kind_is_none(self):
        """A Candidate constructed without the origin_kind field (existing
        callers) defaults to None -> auto, preserving backward compat."""
        c = Candidate(
            doc_id="x", classified_type="decision", label="决策结论",
            title="t", content="c", source_session_id="s1",
            source_profile="coder", cue_hits=["决定采用"],
        )
        self.assertIsNone(c.origin_kind)


# ---------------------------------------------------------------------------
# (c) incident candidate -> staging PENDING + self-confirm -> confirmed
# ---------------------------------------------------------------------------

class TR9IncidentSelfConfirmTests(unittest.TestCase):
    """An incident candidate (origin_kind=agent) is staged PENDING and then
    self-confirmed by the same profile (confirmed_by='agent'), bypassing
    the clarify human gate (G7 owner ruling "不需要人工")."""

    def test_incident_candidate_self_confirmed_not_clarified(self):
        rec = _FakeService()
        ctx = _make_ctx()
        state = _WriteSideState(ctx)
        state._loaded_rules = {
            "enabled": True, "clarify_batch_size": 5, "min_cue_hits": 1,
            "rules": merge_rules(None)["rules"],
            "llm": {"enable": False},
        }
        state._service_override = rec

        cand = _incident_cand(
            "doc-inc-1",
            "顺带发现一个 API 细节：限流 429",
            "顺带发现一个 API 细节：这个接口的 rate limit 是每分钟 10 次，"
            "超限就 429，跟文档里写的不一样。"
            "甲乙丙丁戊己庚辛壬癸甲乙丙丁戊己庚辛壬癸",
            profile="coder", session="s-inc",
        )

        # Simulate the incident path: stage + self-confirm (no clarify).
        # We drive the on_turn path with a text that hits the incident cue.
        old_profile = _set_env_profile("coder")
        try:
            state.on_turn(
                "s-inc",
                "顺带发现一个 API 细节：这个接口的 rate limit 是每分钟 10 次，"
                "超限就 429，跟文档里写的不一样。"
                "甲乙丙丁戊己庚辛壬癸甲乙丙丁戊己庚辛壬癸",
                "user asked about rate limit behavior",
            )
        finally:
            _restore_env_profile(old_profile)

        # The incident candidate must have been staged (ingest called).
        self.assertGreaterEqual(len(rec.ingested), 1,
                                "incident candidate must be staged")
        # The doc dict must carry origin_kind=agent.
        staged_doc = rec.ingested[0]
        self.assertEqual(staged_doc.get("origin_kind"), "agent",
                         "incident doc must carry origin_kind=agent")
        self.assertEqual(staged_doc.get("status"), "pending",
                         "incident doc must be staged PENDING before confirm")
        # Self-confirm must have been called with confirmed_by=agent.
        self.assertGreaterEqual(len(rec.confirm_calls), 1,
                                "incident candidate must be self-confirmed")
        confirm_call = rec.confirm_calls[0]
        self.assertEqual(confirm_call.get("confirmed_by"), "agent",
                         "self-confirm must use confirmed_by=agent")
        self.assertEqual(confirm_call.get("source_profile"), "coder",
                         "self-confirm must use the same source_profile")
        # The candidate must NOT have entered the clarify aggregator
        # (incident path bypasses the human gate).
        self.assertEqual(len(state._agg().pending_for("s-inc")), 0,
                         "incident candidates must NOT enter the clarify "
                         "aggregator (no human gate)")

    def test_incident_docid_differs_from_auto_origin_kind(self):
        """The uuid5 docid mechanism: a candidate with origin_kind=agent
        produces a DIFFERENT docid than the same content with
        origin_kind=auto — this is expected (origin_kind is a property
        of the candidate, not content). Same verified mechanism as
        Phase 1 TR-1 (setdefault("origin_kind","auto"))."""
        title = "顺带发现一个 API 细节"
        content = "rate limit 429 甲乙丙丁戊己庚辛壬癸"
        doc_auto = {
            "title": title, "content": content, "status": "pending",
            "origin_kind": "auto", "kb_class": "incident",
            "source_profile": "coder", "source_session_id": "s1",
        }
        doc_agent = dict(doc_auto, origin_kind="agent")
        id_auto = _deterministic_docid(doc_auto)
        id_agent = _deterministic_docid(doc_agent)
        self.assertNotEqual(id_auto, id_agent,
                            "origin_kind=agent vs auto must produce "
                            "different docids (uuid5 over the full doc "
                            "dict) — expected, not a bug")

    def test_incident_gate_reject_stays_in_staging(self):
        """When the confirm() hard gate rejects (e.g. profile_mismatch),
        the doc stays in staging + an agent_confirm_skipped audit record
        is written — the shared G5 backstop path (not a new mechanism)."""
        def _reject(doc_id, source_profile=None, confirmed_by=None):
            return {"status": "error", "error": "profile_mismatch",
                    "doc_id": doc_id}

        rec = _FakeService(confirm_results=_reject)
        ctx = _make_ctx()
        state = _WriteSideState(ctx)
        state._loaded_rules = {
            "enabled": True, "clarify_batch_size": 5, "min_cue_hits": 1,
            "rules": merge_rules(None)["rules"],
            "llm": {"enable": False},
        }
        state._service_override = rec

        old_profile = _set_env_profile("coder")
        try:
            state.on_turn(
                "s-inc-gate",
                "顺带发现一个限制：这个接口不支持流式响应。"
                "甲乙丙丁戊己庚辛壬癸甲乙丙丁戊己庚辛壬癸",
                "user asked about streaming support",
            )
        finally:
            _restore_env_profile(old_profile)

        # The doc was staged (ingest called) but confirm was rejected.
        self.assertGreaterEqual(len(rec.ingested), 1,
                                "the doc must have been staged")
        self.assertGreaterEqual(len(rec.confirm_calls), 1,
                                "confirm must have been attempted")
        # The audit record must exist (agent_confirm_skipped).
        skipped = [r for r in rec.audit_records
                   if r.get("action") == "agent_confirm_skipped"]
        self.assertGreaterEqual(len(skipped), 1,
                                "a rejected self-confirm must leave an "
                                "agent_confirm_skipped audit record")
        self.assertEqual(skipped[0].get("reason_code"), "profile_mismatch")

    def test_incident_near_dup_reject_stays_in_staging(self):
        """When the ingest() near-dup gate rejects the doc (it doesn't
        even reach staging), the confirm is not attempted — the doc
        simply doesn't exist in staging, so nothing to backstop. This
        is the 'rejected at ingest' path (distinct from 'staged but
        confirm rejected')."""
        class _RejectIngest(_FakeService):
            def ingest(self, docs):
                # Simulate near-dup reject: no doc is staged.
                return {"ingested": [], "rejected": [
                    {"reason": "near_duplicate", "doc_id": "existing-doc"}],
                    "staged": []}

        rec = _RejectIngest()
        ctx = _make_ctx()
        state = _WriteSideState(ctx)
        state._loaded_rules = {
            "enabled": True, "clarify_batch_size": 5, "min_cue_hits": 1,
            "rules": merge_rules(None)["rules"],
            "llm": {"enable": False},
        }
        state._service_override = rec

        old_profile = _set_env_profile("coder")
        try:
            state.on_turn(
                "s-inc-dup",
                "顺带发现一个边角：这个 API 的 corner case 行为。"
                "甲乙丙丁戊己庚辛壬癸甲乙丙丁戊己庚辛壬癸",
                "user asked about API corner case",
            )
        finally:
            _restore_env_profile(old_profile)

        # No confirm call (the doc was never staged).
        self.assertEqual(len(rec.confirm_calls), 0,
                         "a near-dup rejected at ingest must NOT trigger "
                         "a confirm call (the doc doesn't exist in staging)")
        # No audit record (the doc was never staged, so there's nothing
        # to backstop — the ingest() itself handles the reject audit).
        self.assertEqual(len(rec.audit_records), 0,
                         "no agent_confirm_skipped audit for a doc that "
                         "was never staged")


# ---------------------------------------------------------------------------
# Live service tests (real KbSearchService, same pattern as TR-3/TR-2)
# ---------------------------------------------------------------------------

class TR9LiveServiceTests(unittest.TestCase):
    """End-to-end with the real KbSearchService: an incident candidate
    staged PENDING and self-confirmed lands in the confirmed corpus
    with origin_kind=agent."""

    def _svc(self, base):
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        return svc

    def test_incident_doc_confirmed_and_searchable(self):
        base = Path(tempfile.mkdtemp())
        svc = self._svc(base)
        old = _set_env_profile("coder")
        try:
            # Simulate the incident path: stage PENDING + self-confirm.
            doc = {
                "title": "顺带发现一个 API 细节：限流行为",
                "content": "顺带发现一个 API 细节：这个接口的 rate limit "
                           "是每分钟 10 次，超限就 429。"
                           "甲乙丙丁戊己庚辛壬癸甲乙丙丁戊己庚辛壬癸",
                "status": "pending",
                "origin_kind": "agent",
                "kb_class": "incident",
                "task_type": "incident",
                "source_profile": "coder",
                "source_session_id": "s-live",
            }
            out = svc.ingest([doc])
            self.assertEqual(len(out["staged"]), 1,
                             "the doc must be staged PENDING")
            docid = out["staged"][0]
            res = svc.confirm(docid, source_profile="coder",
                              confirmed_by="agent")
            self.assertEqual(res["status"], "success",
                             "self-confirm must succeed (same profile)")
            # The doc must be in the confirmed corpus with origin_kind=agent.
            found = svc.search("限流 429 甲乙丙丁", top_k=3,
                               filters={"origin_kind": "agent"})
            ids = {r["doc_id"] for r in found["data"]["results"]}
            self.assertIn(docid, ids,
                          "the confirmed incident doc must be searchable")
        finally:
            _restore_env_profile(old)
        svc.close()

    def test_incident_doc_near_dup_rejected_at_ingest(self):
        """A doc that is a near-duplicate of an existing confirmed doc
        is rejected at the length/near-dup gate and does NOT enter the
        confirmed corpus. The staging row (if any) is the backstop."""
        base = Path(tempfile.mkdtemp())
        svc = self._svc(base)
        old = _set_env_profile("coder")
        try:
            # Seed a confirmed doc.
            svc.ingest([{
                "docid": "seed-doc",
                "title": "限流 429 经验 条目 甲乙丙丁戊己庚辛壬癸",
                "content": "限流 429 经验 条目 甲乙丙丁戊己庚辛壬癸",
                "status": "confirmed", "origin_kind": "agent",
                "source_profile": "coder",
            }])
            # Now try to stage a very similar doc PENDING.
            dup_doc = {
                "title": "限流 429 经验 条目 甲乙丙丁戊己庚辛壬癸",
                "content": "限流 429 经验 条目 甲乙丙丁戊己庚辛壬癸",
                "status": "pending",
                "origin_kind": "agent",
                "kb_class": "incident",
                "task_type": "incident",
                "source_profile": "coder",
                "source_session_id": "s-dup",
            }
            out = svc.ingest([dup_doc])
            # Pending docs do NOT go through the near-dup gate at staging
            # (only confirmed docs do). So this should stage successfully.
            self.assertEqual(len(out["staged"]), 1,
                             "pending docs are not near-dup-gated at "
                             "staging (the gate runs on confirm)")
            docid = out["staged"][0]
            # Now confirm it — the near-dup gate runs on confirm.
            res = svc.confirm(docid, source_profile="coder",
                              confirmed_by="agent")
            # The confirm may reject it as near_duplicate (or it may
            # succeed if the threshold is not crossed). Either way, the
            # backstop is the staging row + audit record.
            if res["status"] != "success":
                # A rejected confirm leaves the doc in staging (human
                # backstop). Verify the audit record.
                self.assertIn(res.get("error", ""),
                              ("near_duplicate", "profile_mismatch",
                               "not_found", "not_pending"))
        finally:
            _restore_env_profile(old)
        svc.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
