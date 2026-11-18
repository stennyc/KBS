#!/usr/bin/env python3
"""Behavior tests for the knowledge-base write-side hook layer.

Run:  python3 -m pytest test_kb_ingest_hook.py -v

Proves the wiring end-to-end WITHOUT a live LLM:
  - a post_llm_call turn that matches a rule lands the candidate PENDING
  - the SAME session continuing below batch -> no clarify footnote yet
  - hitting the batch threshold -> the pending batch is queued and the
    NEXT pre_llm_call footnotes it (filemem-style; no ctx.inject_message)
  - a session boundary with pending -> previous session's batch footnoted
  - a no-match turn stages nothing, footnotes nothing
"""

import os
import sys
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = os.environ.get("KB_HOME", str(_PLUGIN_ROOT))
KB_DIR = os.environ.get("KB_HOME", str(_PLUGIN_ROOT))
for p in (PLUGIN_DIR, KB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import kb_ingest_hook as hook  # noqa: E402
from kb_ingest_rules import Candidate, merge_rules  # noqa: E402


class FakeCtx:
    """Minimal PluginContext stand-in: record hook + inject_message calls."""

    def __init__(self, batch_size=5):
        self.registered_hooks: list = []
        self.injects: list = []
        # Point the service at a temp data dir + inject a no-op config.
        self._batch_size = batch_size

    def register_hook(self, name, cb):
        self.registered_hooks.append(name)

    def register_tool(self, **kw):
        pass

    def inject_message(self, content, role=None, session_key=None, **kw):
        self.injects.append({"content": content, "role": role})
        return True

    def load_config(self):
        return {"plugins": {"entries": {"knowledge-base-search": {
            "kb_ingest": {"rules": {"clarify_batch_size": self._batch_size}}
        }}}}


@pytest.fixture(autouse=True)
def _temp_service(tmp_path, monkeypatch):
    """Route the hook's service + KB package into a fresh temp data dir."""
    monkeypatch.setenv("KB_DATA_DIR", str(tmp_path / "kbdata"))
    monkeypatch.setenv("KB_LOG_PATH", str(tmp_path / "kb.log"))
    # Drop any cached module-level service singleton so each test starts
    # clean (the plugin package is a package, not the stdlib __init__).
    import importlib
    try:
        kb_pkg = importlib.import_module("knowledge-base-search")
    except ImportError:
        kb_pkg = None
    if kb_pkg is not None and hasattr(kb_pkg, "_service"):
        monkeypatch.setattr(kb_pkg, "_service", None)
    # The hook module caches a per-ctx state; reset between tests.
    hook._STATES.clear()
    yield


def _fire(ctx, cb, session_id, assistant, user="q"):
    cb(session_id=session_id, user_message=user,
       assistant_response=assistant, model="m", platform="test")


def _note(ctx, session_id, user="q"):
    """Drive the read-side pre_llm_call for session_id and return its
    out dict (the host would splice out["context"] into the turn)."""
    cb = hook._make_read_callback(ctx)
    return cb(session_id=session_id, user_message=user,
              assistant_response="", model="m", platform="test")


def _service_stub():
    """A fake KB service that returns 0 hits and no-ops writes: the footnote
    path must be testable without a live corpus.  The ``ingest`` no-op keeps
    _stage's svc.ingest call from raising AttributeError when the batch
    threshold path exercises the full write → clarify cycle in unit tests."""
    return type("S", (), {
        "search": staticmethod(
            lambda query, top_k=5, filters=None, min_score=0.0: {
                "status": "success",
                "data": {"results": [], "total": 0}}),
        "ingest": staticmethod(
            lambda docs: {"staged": 0}),
    })()


def _queue_pending(ctx, session_id, doc_ids, *, fired=True, last_session=None):
    """Test hook: plant pending candidates into the ctx's write-side
    aggregator directly (no service/LLM involved)."""
    state = hook._STATES.setdefault(id(ctx), hook._WriteSideState(ctx))
    if fired:
        state._fire_clarify_fired.add(session_id)
    if last_session is not None:
        state._last_session = last_session
    for d in doc_ids:
        state._agg().add(Candidate(
            doc_id=d, classified_type="decision", label="决策结论",
            title=f"title-{d}", content=f"content-{d}",
            source_session_id=session_id, source_profile="coder",
            cue_hits=["决定采用"]))


def _queue_legacy_pending(ctx, session_id, doc_ids, *, fired=True, last_session=None):
    """Like _queue_pending but for the LEGACY rule path (v2, 2026-10-14):
    origin_kind=None candidates stay on the clarify human gate - they
    must still build a footnote + ask, proving the legacy backstop
    (a stale pre-v2 profile override re-introducing decision /
    walkthrough / config cues) survives v2 unchanged."""
    state = hook._STATES.setdefault(id(ctx), hook._WriteSideState(ctx))
    if fired:
        state._fire_clarify_fired.add(session_id)
    if last_session is not None:
        state._last_session = last_session
    for d in doc_ids:
        state._agg().add(Candidate(
            doc_id=d, classified_type="decision", label="决策结论",
            title=f"title-{d}", content=f"content-{d}",
            source_session_id=session_id, source_profile="coder",
            cue_hits=["决定采用"], origin_kind=None))


class TestHookBehavior:
    def test_matching_turn_stages_pending(self, tmp_path):
        ctx = FakeCtx()
        cb = hook._make_callback(ctx)
        # The regex front gate is the v2 agent-experience rule set: a
        # candidate needs a TASK SUBJECT + PREMISE/CONDITION structure,
        # not an incidental-marker word. "最终决定采用方案 B" is a bare
        # verdict (filemem L2 territory, deliberately GONE from the v2
        # rules — see kb_ingest_rules.py L270-274). The premise-class
        # compound cue ("前提：...只在/当/满足") is what the v2 set
        # captures, so feed a premise-structured turn.
        _fire(ctx, cb, "s1",
              "前提：只在 7-gateway 场景下才需要串行化 sweep，"
              "否则并行会丢更新。")
        # A pending row now exists in the isolated staging store.
        svc = hook._service()
        assert svc._staging.size >= 1

    def test_no_match_no_footnote(self):
        ctx = FakeCtx()
        cb = hook._make_callback(ctx)
        _fire(ctx, cb, "s1", "今天天气不错。")
        old = hook._service
        hook._service = _service_stub
        try:
            out = _note(ctx, "s1")
            assert "context" not in out or "【知识待确认】" not in out.get("context", "")
        finally:
            hook._service = old

    def test_batch_threshold_queues_footnote(self):
        """v2 (2026-10-14): the legacy clarify batch path (origin_kind=None,
        i.e. a stale pre-v2 profile override re-introducing decision /
        walkthrough / config cues) still STASHES at the batch threshold and
        surfaces as a footnote on the next pre_llm_call, without
        re-surfacing. The regex front gate can no longer produce a legacy
        candidate by default (v2 retired those cues), so the legacy
        behavior is driven via _queue_legacy_pending - the same code path
        the write side takes for ANY origin_kind=None candidate.

        The V2 path (all four v2 classes carry origin_kind=agent and
        bypass the human gate entirely) is covered by
        test_kb_ingest_rules_v2.py."""
        ctx = FakeCtx(batch_size=3)
        _queue_legacy_pending(ctx, "s1", ["doc-0", "doc-1", "doc-2"],
                              fired=True, last_session="s1")
        out = _note(ctx, "s1")
        assert "context" in out, "at batch threshold the next pre_llm_call footnotes"
        assert "【知识待确认】" in out["context"]
        assert "决策结论" in out["context"]
        # Anti re-surface: the surfaced batch is marked asked -> the next
        # note is clean.
        out2 = _note(ctx, "s1")
        assert "【知识待确认】" not in out2.get("context", ""), \
            "a surfaced batch must not re-surface next turn"

    def test_below_batch_does_not_footnote(self):
        """Sub-batch pending candidates stay stashed until a TRIGGER
        (batch threshold / session boundary) — a triggerless note call
        must be silent."""
        ctx = FakeCtx(batch_size=3)
        cb = hook._make_callback(ctx)
        old = hook._service
        hook._service = _service_stub
        try:
            # Plant 2 candidates WITHOUT ever hitting a trigger
            # (batch_size=3, no boundary fired, _fire_clarify_fired
            # not set): they stay below the release threshold and must
            # not footnote.
            _queue_pending(ctx, "s1", ["p1", "p2"], fired=False,
                          last_session="s1")
            out = _note(ctx, "s1")
            assert "context" not in out or "【知识待确认】" not in out.get("context", ""), \
                "below batch must not footnote"
        finally:
            hook._service = old

    def test_session_boundary_footnotes_previous(self):
        """Boundary s1->s2 releases BOTH sessions' stashed batches
        (full-replication: current session_id + state._last_session):
        the previous session's below-threshold candidates surface
        together with the current one on s2's next pre_llm_call."""
        ctx = FakeCtx(batch_size=5)
        cb = hook._make_callback(ctx)
        _queue_pending(ctx, "s1", ["doc-x"], last_session="s1")
        _queue_pending(ctx, "s2", ["doc-y"], fired=False, last_session="s2")
        old = hook._service
        hook._service = _service_stub
        try:
            # Simulate the real on_turn boundary flow: the boundary
            # fires _fire_clarify(prev_session, prev_session=prev),
            # recording s1 in _fire_clarify_fired + _last_boundary_prev.
            st = hook._STATES[id(ctx)]
            st._fire_clarify_fired.add("s1")
            st._last_boundary_prev = "s1"
            _fire(ctx, cb, "s2", "最终决定采用 Y")  # boundary s1->s2
            assert ctx.injects == [], "boundary trigger must not inject_message"
            out = _note(ctx, "s2")
            assert "context" in out and "【知识待确认】" in out["context"]
            assert "doc-x" in out["context"], \
                "the previous session's pending batch must be footnoted"
            assert "doc-y" in out["context"], \
                "the current session's pending batch must be footnoted"
        finally:
            hook._service = old

    def test_footnote_surfaces_pending_without_hits(self):
        """0 KB hits + queued pending batch -> the footnote alone
        justifies an injection (owner decision 2: open the 0-hit path)."""
        ctx = FakeCtx()
        _queue_pending(ctx, "s1", ["doc-a"])
        old = hook._service
        hook._service = _service_stub
        try:
            out = _note(ctx, "s1")
            assert out == {"context": out["context"]}
            assert "【知识待确认】" in out["context"]
        finally:
            hook._service = old
        # No pending, no hits -> still a clean pass.
        ctx2 = FakeCtx()
        out2 = _note(ctx2, "s9")
        assert out2 == {}

    def test_register_wires_post_llm_call(self):
        ctx = FakeCtx()
        hook.register_write_side(ctx)
        assert "post_llm_call" in ctx.registered_hooks


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
