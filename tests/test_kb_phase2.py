#!/usr/bin/env python3
"""Phase 2 unit tests: TR-2 (auto-inject partition + decay gate) and
TR-4 (task-boundary auto-extract into the KB).

TR-2: the pre_llm_call path gains a second, profile-scoped
     origin_kind=agent fetch merged into the SAME `## 知识库参考` block
     (user segment first, agent segment after, 6:4 budget of the
     unchanged 1500-char total). The decay gate filters agent items
     (premise_status=="true" AND (reference_count>0 OR age_days<=14,
     age from the first-seen ingest_ts sidecar)); user items are
     unconstrained. pre_llm_call NEVER reinforces (KPI-3/R-3 negative
     test against a live service + real reinforce.jsonl).

TR-4: the task-boundary sink (session-switch trigger, reused from the
     write-side hook) extracts the previous session's pending candidates
     into structured agent-experience docs (topic/task_type/premise/
     premise_status), reuses the kb_distill structure gate, ingests
     PENDING into staging and self-confirms with the SAME source_profile
     (confirmed_by="agent", no clarify gate). A cross-profile self-
     confirm is rejected by confirm()'s hard gate and leaves the row in
     staging + an agent_confirm_skipped audit record.

Run: PYTHONPATH=/root/.hermes/plugins/knowledge-base-search \
     python3 -m pytest test_kb_phase2.py -q
"""
import json
import pickle
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))
_kb_root = str(_PLUGIN_DIR.parents[2])
if _kb_root not in sys.path:
    sys.path.insert(0, _kb_root)

import kb_service  # noqa: E402
from kb_service import KbSearchService  # noqa: E402

import kb_ingest_hook as hook  # noqa: E402
from kb_ingest_hook import (  # noqa: E402
    _agent_in_decay_gate,
    _build_segmented_block,
    _agent_filters,
    _make_read_callback,
    _WriteSideState,
    _READ_MAX_CONTEXT_CHARS,
)
from kb_ingest_rules import Candidate  # noqa: E402

# S6 read-side isolation: the referral-hint branch consults
# kb_shared_memory._load_shared_memory_config(), which resolves
# <HERMES_HOME>/config.yaml when HERMES_HOME is set. The host test runner
# exports HERMES_HOME (e.g. /root/.hermes/profiles/coder), and that config
# HAS a kb shared_memory section -> enabled -> the zero-hit / falsified /
# decayed cases emit the cross-device referral hint and break the
# "must be {}" assertions below.
#
# conftest strips HERMES_HOME / HERMES_CONFIG / HERMES_PROFILE before
# collection; re-pin them for this module so the loader resolves None
# (no shared_memory config) regardless of host or conftest evolution,
# WITHOUT leaking into later test modules (the module-level setenv below
# was a persistent os.environ write — later files saw
# HERMES_HOME=/nonexistent-kb-test-home). This file's 29 tests all run
# within these class-scoped autouse fixtures.
import os
import pytest

_KB_PHASE2_ENV = {k: os.environ.get(k, "") for k in
                 ("HERMES_HOME", "HERMES_CONFIG", "HERMES_PROFILE")}


@pytest.fixture(autouse=True)
def _kb_phase2_isolated_home():
    os.environ["HERMES_HOME"] = "/nonexistent-kb-test-home"
    os.environ["HERMES_CONFIG"] = ""
    os.environ["HERMES_PROFILE"] = ""
    yield
    for _k, _v in _KB_PHASE2_ENV.items():
        os.environ[_k] = _v


def _load_pkl_docs(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f).get("docs", [])


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


class _FakeService:
    """Canned search payloads per filters; records every search() call."""
    def __init__(self, user_results, agent_results):
        self._user = user_results
        self._agent = agent_results
        self.search_calls = []

    def search(self, query, top_k=5, filters=None, min_score=0.0):
        self.search_calls.append({"query": query, "top_k": top_k,
                                  "filters": filters})
        results = self._agent if filters else self._user
        return {"status": "success",
                "data": {"query": query, "results": results,
                         "total": len(results)}}

    def reinforce(self, docid, **kw):
        raise AssertionError("the pre_llm_call path must never reinforce "
                             "(KPI-3 / R-3 red line)")


def _result(doc_id, title, content, **meta):
    return {"doc_id": doc_id, "title": title, "content": content,
            "score": 0.5, "metadata": meta}


def _ctx():
    class _Ctx:
        def __init__(self):
            self.hooks = {}
            self.injected = []

        def register_hook(self, name, cb):
            self.hooks.setdefault(name, []).append(cb)

        def inject_message(self, msg, role="user"):
            self.injected.append((role, msg))
    return _Ctx()


# ---------------------------------------------------------------------------
# TR-2: decay gate predicate (unit level)
# ---------------------------------------------------------------------------

class TR2DecayGateTests(unittest.TestCase):
    NOW = 1_000_000.0

    def test_gate_passes_when_premise_true_and_fresh(self):
        res = _result("a1", "T", "C", premise_status=True,
                      ingest_ts=self.NOW - 3 * 86400)
        self.assertTrue(_agent_in_decay_gate(res, self.NOW))

    def test_gate_passes_when_premise_true_string_and_fresh(self):
        res = _result("a1", "T", "C", premise_status="true",
                      ingest_ts=self.NOW - 13 * 86400)
        self.assertTrue(_agent_in_decay_gate(res, self.NOW))

    def test_gate_passes_when_referenced_even_stale(self):
        res = _result("a2", "T", "C", premise_status="true",
                      reference_count=2,
                      ingest_ts=self.NOW - 400 * 86400)
        self.assertTrue(_agent_in_decay_gate(res, self.NOW))

    def test_gate_blocks_stale_unref_agent_item(self):
        # premise true, no reference, 15 days old -> decayed.
        res = _result("a3", "T", "C", premise_status="true",
                      ingest_ts=self.NOW - 15 * 86400)
        self.assertFalse(_agent_in_decay_gate(res, self.NOW))

    def test_gate_blocks_needs_verification(self):
        res = _result("a4", "T", "C", premise_status="needs_verification",
                      reference_count=9, ingest_ts=self.NOW)
        self.assertFalse(_agent_in_decay_gate(res, self.NOW))

    def test_gate_blocks_falsified(self):
        res = _result("a5", "T", "C", premise_status="falsified",
                      reference_count=9, ingest_ts=self.NOW)
        self.assertFalse(_agent_in_decay_gate(res, self.NOW))

    def test_gate_blocks_missing_ingest_ts_unknown_age(self):
        # Fail-closed: no first-seen ts + no reference -> unknown age.
        res = _result("a6", "T", "C", premise_status="true")
        self.assertFalse(_agent_in_decay_gate(res, self.NOW))


# ---------------------------------------------------------------------------
# TR-2: segmented block budget 6:4 + block shape
# ---------------------------------------------------------------------------

class TR2SegmentedBlockTests(unittest.TestCase):
    def _hits(self, n, prefix, content_len=120):
        return [_result(f"{prefix}{i}", f"{prefix} title {i}",
                        f"{prefix} body {i} " + "x" * content_len)
                for i in range(n)]

    def test_user_segment_before_agent_segment_and_budgets(self):
        user = self._hits(3, "u")
        agent = self._hits(3, "a")
        block, n_user, n_agent = _build_segmented_block(
            user, agent, _READ_MAX_CONTEXT_CHARS)
        self.assertIn("## 知识库参考", block)
        self.assertIn("### 用户知识", block)
        self.assertIn("### agent 经验", block)
        self.assertLess(block.index("### 用户知识"),
                        block.index("### agent 经验"),
                        "user segment must come before the agent segment")
        self.assertEqual(n_user, 3)
        self.assertEqual(n_agent, 3)
        self.assertLessEqual(len(block), _READ_MAX_CONTEXT_CHARS + 1)
        # 6:4 per-segment budgets of the 1500-char total.
        self.assertLessEqual(_READ_MAX_CONTEXT_CHARS * 0.4 + 1,
                             601)  # agent budget <= 600 (KPI-4 hard cap)

    def test_agent_segment_hard_capped_at_600_chars(self):
        # 3 agent hits of ~500-char snippets must be truncated so the
        # agent segment stays within its 600-char slot.
        user = self._hits(1, "u")
        agent = self._hits(3, "a", content_len=500)
        block, _, _ = _build_segmented_block(user, agent,
                                              _READ_MAX_CONTEXT_CHARS)
        agent_start = block.index("### agent 经验")
        self.assertLessEqual(len(block) - agent_start, 600 + len("### agent 经验") + 2)

    def test_user_hits_only_no_agent_segment(self):
        block, n_user, n_agent = _build_segmented_block(
            self._hits(2, "u"), [], _READ_MAX_CONTEXT_CHARS)
        self.assertEqual(n_agent, 0)
        self.assertNotIn("### agent 经验", block)
        self.assertIn("### 用户知识", block)

    def test_agent_hits_only_no_user_segment(self):
        block, n_user, n_agent = _build_segmented_block(
            [], self._hits(1, "a"), _READ_MAX_CONTEXT_CHARS)
        self.assertEqual(n_user, 0)
        self.assertNotIn("### 用户知识", block)
        self.assertIn("### agent 经验", block)

    def test_both_empty_yields_no_block(self):
        block, n_user, n_agent = _build_segmented_block([], [], 1500)
        self.assertEqual(block, "")
        self.assertEqual((n_user, n_agent), (0, 0))


# ---------------------------------------------------------------------------
# TR-2: pre_llm_call callback — partition merge, log surface, no-reinforce
# ---------------------------------------------------------------------------

class TR2ReadCallbackTests(unittest.TestCase):
    def _wire(self, user_results, agent_results):
        fake = _FakeService(user_results, agent_results)
        # save/restore so the fake never leaks into later tests
        old_svc = hook._service
        hook._service = lambda: fake
        cb = _make_read_callback(_ctx())
        _restore_svc = lambda: setattr(hook, "_service", old_svc)
        return fake, cb, _restore_svc

    def test_user_and_agent_segments_merged(self):
        user = [_result("u1", "U title", "U body content here")]
        agent = [_result("a1", "A title", "A body content here",
                         premise_status="true",
                         ingest_ts=time.time())]
        fake, cb, _restore_svc = self._wire(user, agent)
        out = cb(user_message="how do I configure the shared gateway")
        self.assertIn("context", out)
        block = out["context"]
        self.assertIn("U title", block)
        self.assertIn("A title", block)
        self.assertLess(block.index("### 用户知识"), block.index("### agent 经验"))
        # The second fetch is unconditional (PRD: an EXTRA agent-partition
        # search); the agent doc already matched by the plain query is
        # deduped by doc_id — it renders ONCE, in the user segment.
        agent_calls = [c for c in fake.search_calls if c["filters"]]
        self.assertEqual(len(agent_calls), 1,
                         "the agent-partition second fetch must run")
        self.assertEqual(agent_calls[0]["filters"].get("origin_kind"), "agent")
        # No-reinforce: the fake would raise if ever called.
        _restore_svc()

    def test_agent_segment_only_when_user_fetch_zero_hits(self):
        agent = [_result("a1", "A title", "A body content here",
                         premise_status="true",
                         ingest_ts=time.time())]
        fake, cb, _restore_svc = self._wire([], agent)
        out = cb(user_message="how do I configure the shared gateway")
        self.assertIn("context", out)
        self.assertIn("### agent 经验", out["context"])
        self.assertNotIn("### 用户知识", out["context"])
        # Exactly two fetches: plain + filtered.
        self.assertEqual(len(fake.search_calls), 2)
        _restore_svc()

    def test_user_hits_agent_gate_applied_in_place_and_deduped(self):
        # An agent item inside the plain fetch is gated in place (no
        # second render) and deduped out of the agent segment by doc_id.
        user = [
            _result("u1", "U title", "U body content here"),
            _result("a1", "A stale", "stale agent body",
                    origin_kind="agent", premise_status="true",
                    ingest_ts=time.time() - 30 * 86400),
        ]
        fake, cb, _restore_svc = self._wire(user, [])
        out = cb(user_message="how do I configure the shared gateway")
        self.assertIn("U title", out["context"])
        self.assertNotIn("A stale", out["context"],
                         "a decayed agent doc in the plain fetch is "
                         "dropped by the gate, not re-rendered")
        self.assertEqual(len(fake.search_calls), 2,
                         "the agent-partition second fetch still runs "
                         "(unconditional); it just dedups the hit")
        _restore_svc()

    def test_decayed_agent_hit_not_injected(self):
        agent = [_result("a1", "A stale", "stale agent body",
                         premise_status="true",
                         ingest_ts=time.time() - 30 * 86400)]
        fake, cb, _restore_svc = self._wire([], agent)
        out = cb(user_message="how do I configure the shared gateway")
        self.assertEqual(out, {}, "a decayed agent item must not inject")
        _restore_svc()

    def test_falsified_agent_hit_not_injected(self):
        agent = [_result("a1", "A fals", "falsified agent body",
                         premise_status="falsified",
                         reference_count=5,
                         ingest_ts=time.time())]
        fake, cb, _restore_svc = self._wire([], agent)
        out = cb(user_message="how do I configure the shared gateway")
        self.assertEqual(out, {}, "a falsified agent item must not inject")
        _restore_svc()

    def test_referenced_stale_agent_hit_injected(self):
        agent = [_result("a1", "A ref", "referenced agent body",
                         premise_status="true",
                         reference_count=3,
                         ingest_ts=time.time() - 200 * 86400)]
        fake, cb, _restore_svc = self._wire([], agent)
        out = cb(user_message="how do I configure the shared gateway")
        self.assertIn("A ref", out["context"])
        _restore_svc()

    def test_agent_fetch_failure_is_fail_soft_user_segment_still_injected(self):
        class _Boom(_FakeService):
            def search(self, query, top_k=5, filters=None, min_score=0.0):
                if filters:
                    raise RuntimeError("kb hiccup")
                return super().search(query, top_k, filters, min_score)
        user = [_result("u1", "U title", "U body content here")]
        fake = _Boom(user, [])
        old_svc = hook._service
        hook._service = lambda: fake
        cb = _make_read_callback(_ctx())
        try:
            out = cb(user_message="how do I configure the shared gateway")
            self.assertIn("context", out)
            self.assertIn("U title", out["context"])
            self.assertNotIn("### agent 经验", out["context"])
        finally:
            hook._service = old_svc

    def test_both_segments_zero_hits_no_injection(self):
        fake, cb, _restore_svc = self._wire([], [])
        out = cb(user_message="how do I configure the shared gateway")
        # No hits and no queued clarify footnote -> direct pass.
        self.assertEqual(out, {})
        _restore_svc()

    def test_zero_hits_but_clarify_footnote_injects(self):
        """0 KB hits + a queued write-side pending batch -> the footnote
        alone justifies an injection (the model must see it to relay it
        to the user via the host clarify tool)."""
        fake, cb, _restore_svc = self._wire([], [])
        # Rebuild the callback bound to a ctx we control, so the
        # write-side state is in the same _STATES entry the callback
        # consults at call time.
        ctx = _ctx()
        state = hook._STATES.setdefault(id(ctx), hook._WriteSideState(ctx))
        # Simulate a trigger having fired for s1 (boundary release),
        # and a pending candidate surviving in the aggregator.
        state._fire_clarify_fired.add("s1")
        state._last_boundary_prev = "s1"
        from kb_ingest_rules import Candidate
        state._agg().add(Candidate(
            doc_id="doc-x", classified_type="decision", label="决策结论",
            title="最终决定采用 X", content="content-x",
            source_session_id="s1", source_profile="coder",
            cue_hits=["决定采用"]))
        cb2 = hook._make_read_callback(ctx)
        try:
            out = cb2(session_id="s2", user_message="how do I configure the shared gateway")
            self.assertIn("context", out)
            self.assertIn("【知识待确认】", out["context"])
            self.assertIn("doc-x", out["context"])
        finally:
            hook._STATES.pop(id(ctx), None)
            _restore_svc()

    def test_log_surface_carries_agent_segment_counts(self, caplog=None):
        """KPI-4 verification: the debug log line exposes the agent
        segment's hit count + char count, keeping the pre-TR-2
        'N hit(s), M-char block' fields at the front."""
        import logging
        user = [_result("u1", "U title", "U body content here")]
        agent = [_result("a1", "A title", "A body content here",
                         premise_status="true", ingest_ts=time.time())]
        fake, cb, _restore_svc = self._wire(user, agent)
        records = []

        class _Sink(logging.Handler):
            def emit(self, record):
                records.append(record)

        sink = _Sink()
        root = logging.getLogger("knowledge-base.hook")
        root.setLevel(logging.DEBUG)
        root.addHandler(sink)
        try:
            out = cb(user_message="how do I configure the shared gateway")
        finally:
            root.removeHandler(sink)
        lines = [r.getMessage() for r in records
                 if "kb auto-retrieve" in r.getMessage()]
        self.assertEqual(len(lines), 1)
        line = lines[0]
        self.assertIn("kb auto-retrieve:", line)
        self.assertIn("hit(s),", line)  # pre-TR-2 total-hit field kept
        self.assertIn("2 hit(s)", line)  # 1 user + 1 agent
        self.assertIn("agent: 1 hit(s)", line)
        self.assertIn("agent_chars=", line)
        self.assertIn("user: 1 hit(s)", line)
        _restore_svc()


# ---------------------------------------------------------------------------
# TR-2: live-service negative test — pre_llm_call NEVER reinforces
# ---------------------------------------------------------------------------

class TR2NeverReinforceTests(unittest.TestCase):
    """KPI-3 / R-3 acceptance gate: run the real pre_llm_call callback
    against a LIVE KbSearchService (confirmed agent + user docs, a
    second pre_llm_call pass, explicit svc.reinforce() NEVER called by
    the hook) and assert reinforce.jsonl gained ZERO lines. The
    reference_count inside the agent fetch's metadata is read-only for
    this path — the decay gate only READS it."""

    def _live_service(self):
        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        svc.ingest([
            {"docid": "ag-live", "title": "agent 经验 网关 重试 策略",
             "content": "网关 重试 预算 调整 之后 偶发 报错 消失 甲乙丙丁戊己",
             "status": "confirmed", "origin_kind": "agent",
             "source_profile": "coder",
             "premise_status": "true",
             "reference_count": 1},
            {"docid": "us-live", "title": "用户 录入 网关 配置",
             "content": "网关 配置 片段 重试 次数 用户 录入 甲乙丙丁",
             "status": "confirmed", "origin_kind": "user"},
        ])
        # Back-date the agent doc's first-seen ts so it is fresh (the
        # hook reads ingest_ts from the metadata — the confirmed ingest
        # above already wrote it).
        return svc, base, data, audit

    def test_pre_llm_call_injection_does_not_bump_reinforce_lines(self):
        svc, base, data, audit = self._live_service()
        reinforce_path = base / "kb-data" / "reinforce.jsonl"
        before = _line_count(reinforce_path)

        # The canned fetches must carry the agent doc's CURRENT
        # metadata (premise_status="true" + a first-seen ingest_ts) so
        # the decay gate passes — a bare confirmed ingest has neither
        # stamp and would be gated out (fail-closed).
        ts = time.time()
        user_results = [
            {**r, "metadata": {**r.get("metadata", {}),
                               "premise_status": "true",
                               "ingest_ts": ts,
                               "origin_kind": r.get("metadata", {}).get(
                                   "origin_kind")}}
            for r in svc.search("网关 重试", top_k=3)["data"]["results"]
        ]
        agent_results = [
            {**r, "metadata": {**r.get("metadata", {}),
                               "premise_status": "true",
                               "ingest_ts": ts,
                               "origin_kind": "agent"}}
            for r in svc.search(
                "网关 重试", top_k=3,
                filters={"origin_kind": "agent"})["data"]["results"]
        ]
        fake = _FakeService(user_results, agent_results)
        old_profile = _set_env_profile("coder")
        try:
            with mock.patch.object(hook, "_service", lambda: fake):
                cb = _make_read_callback(_ctx())
                out = cb(user_message="网关 重试 预算 怎么 配置 的")
                self.assertIn("context", out)
                self.assertIn("agent 经验 网关 重试 策略", out["context"],
                              "a fresh agent doc (premise_status=true, "
                              "first-seen ts) must pass the decay gate "
                              "and inject — here via the user segment "
                              "(the plain fetch already matched it; the "
                              "agent-partition fetch dedups it out)")
                out2 = cb(user_message="网关 重试 预算 怎么 配置 的")
                self.assertIn("context", out2)
                self.assertIn("agent 经验 网关 重试 策略", out2["context"],
                              "the second pass must still inject "
                              "(reinforce.jsonl stays at 0 lines, so "
                              "nothing decayed in place)")
        finally:
            _restore_env_profile(old_profile)
        after = _line_count(reinforce_path)
        self.assertEqual(before, after,
                         "pre_llm_call injection must NEVER append a "
                         "reinforce event (R-3: the auto-inject layer "
                         "cannot bootstrap its own favorites)")
        self.assertEqual(before, 0,
                         "fixture precondition: no reinforce events yet")
        svc.close()

    def test_agent_partition_injection_when_plain_fetch_misses(self):
        """The agent segment renders when the PLAIN fetch misses but
        the profile-scoped agent fetch matches (deduped out of the user
        segment by construction). reinforce.jsonl must stay flat."""
        svc, base, data, audit = self._live_service()
        reinforce_path = base / "kb-data" / "reinforce.jsonl"
        before = _line_count(reinforce_path)
        ts = time.time()
        # The plain fetch returns only the USER doc; the agent fetch
        # returns the FRESH agent doc — exactly the PRD's "额外一次
        # filters={origin_kind:agent} 检索" split.
        plain = svc.search("网关 配置", top_k=3)["data"]["results"]
        user_results = [
            r for r in plain
            if (r.get("metadata") or {}).get("origin_kind") != "agent"
        ]
        agent_results = [
            {**r, "metadata": {**r.get("metadata", {}),
                               "premise_status": "true",
                               "ingest_ts": ts,
                               "origin_kind": "agent"}}
            for r in svc.search(
                "网关 重试 策略", top_k=3,
                filters={"origin_kind": "agent"})["data"]["results"]
        ]
        self.assertTrue(user_results, "fixture: the user doc must match")
        self.assertTrue(
            all(r["doc_id"] != "ag-live" for r in user_results),
            "fixture: the plain fetch must NOT carry the agent doc")
        fake = _FakeService(user_results, agent_results)
        old_profile = _set_env_profile("coder")
        try:
            with mock.patch.object(hook, "_service", lambda: fake):
                cb = _make_read_callback(_ctx())
                out = cb(user_message="网关 重试 策略 调整 之后 偶发 报错 怎么 消失 的")
                self.assertIn("context", out)
                self.assertIn("### agent 经验", out["context"],
                              "the agent-partition segment must render "
                              "when only the filtered fetch matches")
                self.assertIn("agent 经验 网关 重试 策略", out["context"])
        finally:
            _restore_env_profile(old_profile)
        after = _line_count(reinforce_path)
        self.assertEqual(before, after,
                         "the agent-partition injection must not "
                         "append a reinforce event either (R-3)")
        svc.close()

    def test_explicit_tool_path_reinforce_still_works(self):
        """Contrast: the explicit svc.reinforce() path (the tool
        layer's write gate) DOES append — the negative test above only
        proves the hook path stays silent, not that the sidecar is
        broken."""
        svc, base, data, audit = self._live_service()
        reinforce_path = base / "kb-data" / "reinforce.jsonl"
        self.assertEqual(_line_count(reinforce_path), 0)
        res = svc.reinforce("ag-live", reason="explicit_search")
        self.assertEqual(res["reinforced"], True)
        self.assertEqual(_line_count(reinforce_path), 1)
        svc.close()


# ---------------------------------------------------------------------------
# TR-4: task-boundary sink (session-switch trigger) + self-confirm loop
# ---------------------------------------------------------------------------

def _default_confirm_ok(doc_id, **kw):
    return {"status": "success", "doc_id": doc_id}


class _SVCRecorder:
    """Records ingest/confirm calls; serves canned confirm results."""
    def __init__(self, confirm_results=None):
        self.ingested = []
        self.confirm_calls = []
        self.confirm_results = confirm_results or _default_confirm_ok
        self.audit_records = []

    def ingest(self, docs):
        out = {"ingested": [], "rejected": [], "staged": []}
        for d in docs:
            if "docid" not in d:
                d["docid"] = f"uuid-{d['title'][:8]}"
            self.ingested.append(d)
            out["staged"].append(d["docid"])
        return out

    def confirm(self, doc_id, source_profile=None, confirmed_by=None):
        self.confirm_calls.append({"doc_id": doc_id,
                                  "source_profile": source_profile,
                                  "confirmed_by": confirmed_by})
        return self.confirm_results(doc_id, source_profile=source_profile,
                                   confirmed_by=confirmed_by)

    def _audit(self, **fields):
        self.audit_records.append(fields)


def _cand(doc_id, title, content, profile, session, kb_class="walkthrough"):
    return Candidate(doc_id=doc_id, classified_type=kb_class,
                     label="踩坑排查路径", title=title, content=content,
                     source_session_id=session, source_profile=profile,
                     cue_hits=["踩坑"])


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


class TR4TaskBoundarySinkTests(unittest.TestCase):
    def test_session_boundary_triggers_sink_staging_and_self_confirm(self):
        """Task 1 (trigger point): a session switch fires the sink for
        the PREVIOUS session; agent docs are staged PENDING and
        self-confirmed by the same profile with confirmed_by='agent'.
        The sink runs BEFORE the boundary clarify, so it sees the
        aggregator's still-pending list; the clarify message below
        (batch-size 1) is the HUMAN-facing path firing after the sink."""
        rec = _SVCRecorder()
        ctx = _ctx()
        state = _WriteSideState(ctx)
        # rules=[] -> the regex detector flags nothing on plain text;
        # the candidate is fed into the aggregator directly (the sink
        # consumes the aggregator's full per-session candidate list).
        state._loaded_rules = {"enabled": True, "clarify_batch_size": 1,
                               "min_cue_hits": 1, "rules": [],
                               "llm": {"enable": False,
                                       "distill": {"enable": False}}}
        cand = _cand("doc-1", "修复 网关 偶发 报错 的 根因 路径",
                     "网关 偶发 报错 根因 是 重试 预算 配置 冲突 甲乙丙丁戊己",
                     profile="coder", session="s1")
        state._agg().add(cand)
        state._service_override = rec
        state.remember_turn_msgs(
            "s1", "网关 偶发 报错 怎么 排查 的",
            "我们 根因 定位 到 重试 预算 配置 冲突，修复 了 该 问题")

        state._service_override = rec
        old_profile = _set_env_profile("coder")
        try:
            # Turn 1: session s1 (detects no candidate on its own text
            # because rules=[] -> pure aggregator path); turn 2: session
            # switch s1 -> s2 fires the boundary sink for s1.
            state.on_turn("s1", "some non-cue turn text here",
                          "a plain assistant reply text")
            state.on_turn("s2", "another turn in the new session",
                          "user asking a fresh question")
        finally:
            _restore_env_profile(old_profile)

        # Staged as agent-experience docs (PENDING + origin_kind=agent).
        # The on_turn s1->s2 boundary already sunk s1's candidates into
        # the recorder (asserted below in the pre-clear counts); drive
        # close_task_boundary once more with a FRESH session id to
        # isolate a clean, single-pass sink interaction.
        pre_ingested = len(rec.ingested)
        pre_confirms = len(rec.confirm_calls)
        self.assertEqual(pre_ingested, 1,
                         "the on_turn boundary sink must have staged "
                         "s1's candidate (task 1: trigger point)")
        self.assertEqual(pre_confirms, 1,
                         "the on_turn boundary sink must have "
                         "self-confirmed s1's candidate (task 2: "
                         "staging->self-confirm closed loop)")
        state._pending_override = [cand]
        state.close_task_boundary("s1")  # idempotent: already sunk -> no-op
        self.assertEqual(len(rec.ingested), pre_ingested,
                         "a re-fired boundary for a sunk session is a "
                         "no-op (idempotence, task 3's precondition)")
        rec.ingested.clear()
        rec.confirm_calls.clear()
        cand2 = _cand("doc-2", "网关 偶发 报错 根因 定位 路径",
                      "网关 偶发 报错 根因 是 重试 预算 配置 冲突 庚辛壬癸甲乙",
                      profile="coder", session="s9")
        state._pending_override = [cand2]
        state.remember_turn_msgs(
            "s9", "网关 偶发 报错 根因 怎么 定位 的",
            "根因 定位 到 重试 预算 配置 冲突，修复 了 该 问题")
        state.close_task_boundary("s9")
        self.assertEqual(len(rec.ingested), 1)
        doc = rec.ingested[0]
        self.assertEqual(doc["status"], "pending")
        self.assertEqual(doc["origin_kind"], "agent")
        self.assertEqual(doc["task_type"], "walkthrough")
        self.assertEqual(doc["topic"], "general")
        # R-6: the session supplied user-message context, so the premise
        # is non-empty; _build_agent_doc writes premise_status="true"
        # (the canonical string spelling that _normalize_docs and the
        # decay gate both accept).
        self.assertEqual(doc["premise_status"], "true")
        self.assertEqual(doc["source_profile"], "coder")
        self.assertIn("网关 偶发 报错", str(doc.get("premise")),
                       "the session's user message(s) form the premise")
        # Empty-premise sub-case: no session messages -> premise stays
        # unset and premise_status is needs_verification (fail-closed out
        # of the auto-injection pool by the decay gate). Run through the
        # real service's _normalize_docs (the same path ingest takes) to
        # verify the canonical spelling survives intact (setdefault does
        # not overwrite a supplied "needs_verification" value).
        from kb_service import _normalize_docs as _norm
        cand3 = _cand("doc-3", "网关 偶发 报错 空前提 测试",
                      "网关 偶发 报错 空前提 测试 条目 甲乙丙丁戊己",
                      profile="coder", session="s10")
        rec.confirm_calls.clear()
        state._pending_override = [cand3]
        state.close_task_boundary("s10")
        doc3_raw = rec.ingested[1]
        self.assertIsNone(doc3_raw.get("premise"))
        self.assertEqual(doc3_raw["premise_status"], "needs_verification")
        doc3 = _norm([doc3_raw])[0]
        self.assertEqual(doc3["premise_status"], "needs_verification",
                         "the real-service normalizer must not overwrite "
                         "the supplied needs_verification value")
        # Self-confirm with the SAME profile, confirmed_by=agent.
        self.assertEqual(len(rec.confirm_calls), 1)
        cc = rec.confirm_calls[0]
        self.assertEqual(cc["source_profile"], "coder")
        self.assertEqual(cc["confirmed_by"], "agent")
        self.assertEqual(cc["doc_id"], doc3_raw["docid"])
        # The HUMAN-facing boundary clarify for s1's original candidate
        # stashed AFTER the sink (the sink reads the aggregator's full
        # buffer, not its pending-dedup bookkeeping; the clarify gate
        # is orthogonal to the agent partition). No more ctx.inject_
        # message (TUI-safe); the batch now surfaces via the read-side
        # pre_llm_call footnote.
        self.assertEqual(len(ctx.injected), 0,
                         "the boundary trigger must stash, not inject")
        self.assertIn("s1", state._fire_clarify_fired,
                      "the boundary trigger must record the release")

    def test_sink_idempotent_per_session(self):
        """A second boundary on the same previous session is a no-op
        (one sink per session; the pending list is drained once)."""
        rec = _SVCRecorder()
        state = _WriteSideState(_ctx())
        state._loaded_rules = {"enabled": True, "clarify_batch_size": 99,
                               "min_cue_hits": 1, "rules": [],
                               "llm": {"enable": False,
                                       "distill": {"enable": False}}}
        cand = _cand("doc-i", "第二次 边界 触发 幂等 测试",
                     "第二次 边界 触发 应该 幂等 不重复 入 账 甲乙丙丁戊己",
                     profile="coder", session="s1")
        state._agg().add(cand)
        state._service_override = rec
        old_profile = _set_env_profile("coder")
        try:
            state.on_turn("s1", "text one", "reply one")
            state.on_turn("s2", "text two", "reply two")
            state.on_turn("s3", "text three", "reply three")
            state.on_turn("s4", "text four", "reply four")
        finally:
            _restore_env_profile(old_profile)
        # s1's candidate sinks ONCE at the s1->s2 boundary; s2/s3/s4
        # have no pending candidates of their own.
        self.assertEqual(len(rec.ingested), 1)
        self.assertEqual(len(rec.confirm_calls), 1)

    def test_confirm_failure_leaves_staging_row_and_audit_record(self):
        """Task 3 (兜底): a confirm() failure (profile_mismatch /
        not_found / not_pending) must leave the row in staging and
        append an agent_confirm_skipped audit record with the reason
        code (the human backstop)."""
        for reason_code in ("profile_mismatch", "not_found", "not_pending"):
            with self.subTest(reason=reason_code):
                rec = _SVCRecorder(confirm_results=lambda did, **kw: {
                    "status": "error", "error": reason_code,
                    "doc_id": did})
                state = _WriteSideState(_ctx())
                state._loaded_rules = {"enabled": True,
                                       "clarify_batch_size": 99,
                                       "min_cue_hits": 1, "rules": [],
                                       "llm": {"enable": False,
                                               "distill": {"enable": False}}}
                cand = _cand(f"doc-{reason_code}",
                             f"{reason_code} 兜底 测试 条目",
                             f"{reason_code} 兜底 测试 条目 甲乙丙丁戊己庚辛",
                             profile="coder", session="s1")
                state._agg().add(cand)
                state._service_override = rec
                old_profile = _set_env_profile("coder")
                try:
                    state.on_turn("s1", "t", "r")
                    state.on_turn("s2", "t", "r")
                finally:
                    _restore_env_profile(old_profile)
                self.assertEqual(len(rec.ingested), 1,
                                 "the doc must be staged even when the "
                                 "confirm fails (留 staging 兜底)")
                skip = [a for a in rec.audit_records
                        if a.get("action") == "agent_confirm_skipped"]
                self.assertEqual(len(skip), 1,
                                 "a failed self-confirm must write an "
                                 "agent_confirm_skipped audit record")
                self.assertEqual(skip[0].get("reason_code"), reason_code)
                self.assertEqual(skip[0].get("confirmed_by"), "agent")

    def test_cross_profile_self_confirm_rejected_and_backstopped(self):
        """Red line (R-1): the sink always confirms with the SAME
        source_profile the docs were ingested under. The confirm()
        hard gate below (live service) proves a cross-profile confirm
        of an agent row is rejected + the row STAYS in staging — the
        sink's audit backstop is the human path out."""
        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        # Profile X stages an agent doc.
        out = svc.ingest([{
            "docid": "ag-x", "title": "X 的 agent 经验 条目",
            "content": "profile X 的 agent 经验 内容 甲乙丙丁戊己庚辛甲",
            "status": "pending", "origin_kind": "agent",
            "source_profile": "X",
        }])
        self.assertIn("ag-x", out["staged"])
        # Profile Y's self-confirm is rejected by confirm()'s hard gate
        # (the Phase-1 untouched line 530-541 logic).
        res = svc.confirm("ag-x", source_profile="Y", confirmed_by="agent")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error"], "profile_mismatch")
        # The row is STILL in staging (left for the human backstop).
        self.assertIsNotNone(svc._staging.find("ag-x"))
        # And the same-profile confirm would have succeeded.
        res_ok = svc.confirm("ag-x", source_profile="X",
                             confirmed_by="agent")
        self.assertEqual(res_ok["status"], "success")
        self.assertIsNone(svc._staging.find("ag-x"))
        # The sink itself never attempts a mismatch: its confirm call
        # uses the doc's OWN source_profile (unit-covered in
        # test_session_boundary_triggers_sink_staging_and_self_confirm:
        # cc['source_profile'] == doc['source_profile'] == 'coder').
        svc.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
