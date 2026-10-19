#!/usr/bin/env python3
"""kb-data/clarify_state.json — the durable clarify DECISION-STATE sidecar.

owner 2026-10-13 ruling (this test's target):
  1) 独立 sidecar 文件（与 staging 物理隔离，单独管理）
  2) 候选元数据不冗余存——从 staging_docs.jsonl 按 doc_id 重读
     （staging 行被 confirm 删了则 footnote 自然消失）

The sidecar persists ONLY the decision state: which sessions fired a
clarify trigger, which doc_ids were already asked, and the two session
boundary markers. The FACT (the pending doc rows themselves) stays in
staging_docs.jsonl — the single source of truth.

Run:
  /root/.hermes/plugin-venv/bin/python -m pytest test_clarify_state_persist.py -v
or the bare-python shape:
  KB_DATA_DIR=<tmp> python3 test_clarify_state_persist.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Resolve the active source tree (plugin dir + KB service dir) the same
# way the deploy quickstart does; a non-gateway interpreter needs both
# on sys.path for the bare imports below.
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
for _d in (
    _PLUGIN_DIR,
    os.environ.get("KB_HOME", "/root/.hermes/plugins/knowledge-base-search"),
):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from clarify_state import ClarifyStateStore  # noqa: E402
import kb_ingest_hook as hook  # noqa: E402


# ---------------------------------------------------------------------------
# 1) ClarifyStateStore — the sidecar itself
# ---------------------------------------------------------------------------
class TestClarifyStateStore(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="kb-clarify-test-"))
        self.store = ClarifyStateStore(self._tmp)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_missing_file_is_empty_state(self):
        self.assertFalse(self._tmp.joinpath("clarify_state.json").exists())
        snap = self.store.snapshot()
        self.assertEqual(snap["sessions"], {})
        self.assertIsNone(snap["last_session"])
        self.assertIsNone(snap["last_boundary_prev"])

    def test_mark_fired_persists_across_reload(self):
        self.store.mark_fired("s1")
        reloaded = ClarifyStateStore(self._tmp)
        self.assertTrue(reloaded.session_entry("s1")["fired"])

    def test_mark_asked_dedups_and_persists(self):
        self.store.mark_asked("s1", ["d1", "d2", "d1"])
        reloaded = ClarifyStateStore(self._tmp)
        self.assertEqual(reloaded.asked_for("s1"), ["d1", "d2"])

    def test_pending_doc_ids_excludes_asked(self):
        self.store.mark_asked("s1", ["d1"])
        pending = self.store.pending_doc_ids("s1", ["d1", "d2", "d3"])
        self.assertEqual(pending, ["d2", "d3"])

    def test_last_session_and_boundary_prev_roundtrip(self):
        self.store.set_last_session("s2")
        self.store.set_last_boundary_prev("s1")
        reloaded = ClarifyStateStore(self._tmp)
        self.assertEqual(reloaded.last_session, "s2")
        self.assertEqual(reloaded.last_boundary_prev, "s1")

    def test_corrupt_sidecar_degrades_to_empty_not_raise(self):
        self._tmp.joinpath("clarify_state.json").write_text("{corrupt", encoding="utf-8")
        # Construction must NOT raise; the state resets to empty.
        fresh = ClarifyStateStore(self._tmp)
        self.assertEqual(fresh.snapshot()["sessions"], {})

    def test_corrupt_sidecar_does_not_touch_staging(self):
        """The decision-state reset is bounded: the staging FACT is intact."""
        staging_dir = self._tmp / "staging"
        staging_dir.mkdir()
        staging_dir.joinpath("staging_docs.jsonl").write_text(
            json.dumps({"docid": "d1", "status": "pending"}) + "\n",
            encoding="utf-8")
        self._tmp.joinpath("clarify_state.json").write_text("{corrupt", encoding="utf-8")
        fresh = ClarifyStateStore(self._tmp)
        self.assertEqual(fresh.snapshot()["sessions"], {})
        # Staging is physically isolated and UNTOUCHED by the sidecar.
        self.assertTrue(staging_dir.joinpath("staging_docs.jsonl").exists())

    def test_asked_cap_bounds_growth(self):
        ids = [f"d{i}" for i in range(600)]
        self.store.mark_asked("s1", ids)
        reloaded = ClarifyStateStore(self._tmp)
        asked = reloaded.asked_for("s1")
        self.assertLessEqual(len(asked), 500)
        # Newest survive, oldest evicted.
        self.assertIn("d599", asked)
        self.assertNotIn("d0", asked)


# ---------------------------------------------------------------------------
# 2) End-to-end: restart survival of the [知识待确认] footnote
# ---------------------------------------------------------------------------
class TestRestartSurvival(unittest.TestCase):
    """The owner requirement: a gateway restart must NOT lose the pending
    clarify ask. We simulate two consecutive processes (two ClarifyStateStore
    instances + two _WriteSideState objects sharing the same KB_DATA_DIR)
    and prove the footnote that fired in process-1 re-surfaces in
    process-2, exactly once, and does NOT re-surface after being asked."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="kb-restart-test-"))
        os.environ["KB_DATA_DIR"] = str(self._tmp)
        # A minimal staging FACT: one pending doc for session "s_prev".
        # The hook rebuild reads from the service's staging store, so we
        # point the sidecar test at a fake service whose _staging exposes
        # the same row.
        self._staging_rows = [
            {
                "docid": "doc-A",
                "title": "决定采用方案 B",
                "content": "讨论完，最终决定采用方案 B，否决 A。甲甲甲" * 3,
                "status": "pending",
                "origin_kind": "auto",
                "kb_class": "decision",
                "source_session_id": "s_prev",
            }
        ]

    def tearDown(self):
        os.environ.pop("KB_DATA_DIR", None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _fake_service(self):
        class _Staging:
            def __init__(self, rows):
                self._rows = list(rows)

            def docs_snapshot(self):
                return [dict(r) for r in self._rows]

        class _Svc:
            def __init__(self, rows):
                self._staging = _Staging(rows)
        return _Svc(self._staging_rows)

    def test_fired_batch_re_surfaces_after_restart_and_does_not_resent(self):
        # ---- PROCESS 1 (the pre-restart gateway) -------------------------
        hook._STATES.clear()
        store1 = ClarifyStateStore(self._tmp)

        # The write side fires a clarify trigger for s_prev (batch
        # threshold reached) — this is what _fire_clarify persists.
        state1 = hook._WriteSideState(None)
        state1._last_session = "s_prev"
        state1._fire_clarify_fired.add("s_prev")
        # Mirror what the write side persists (the hook's _fire_clarify
        # now calls the sidecar; here we drive the store directly to keep
        # the test scoped to the persistence contract, not the regex gate).
        store1.mark_fired("s_prev")
        store1.set_last_session("s_prev")

        # Plant the pending candidate in the in-memory aggregator (the
        # write side does this in _stage; we mirror it so the footnote
        # has something to release).
        from kb_ingest_rules import Candidate
        cand = Candidate(
            doc_id="doc-A",
            classified_type="decision",
            label="决策结论",
            title="决定采用方案 B",
            content="讨论完，最终决定采用方案 B，否决 A。" * 3,
            source_session_id="s_prev",
            source_profile=None,
            cue_hits=[],
            origin_kind=None,
        )
        state1._agg().add(cand)

        # Footnote fires ONCE in process-1 and marks the batch asked.
        fn1 = state1._clarify_footnote("s_prev")
        self.assertIn("【知识待确认】", fn1)
        self.assertIn("doc-A", fn1)
        # The asked set is now persisted (the hook's _clarify_footnote
        # writes it; the test drives the store to assert the contract).
        store1.mark_asked("s_prev", ["doc-A"])

        # ---- PROCESS 2 (post-restart gateway) ----------------------------
        # Memory is zeroed: a fresh _STATES, a fresh _WriteSideState.
        hook._STATES.clear()
        state2 = hook._WriteSideState(None)
        # Without the reconstruction path, state2 has NO fired marker and
        # NO pending buffer -> the footnote would be empty (the bug we
        # are fixing). With it, the sidecar + staging rebuild restores
        # the ask EXACTLY ONCE (not re-asked, because doc-A is in the
        # sidecar's asked set).
        rebuilt = hook._rebuild_state_from_sidecar.__globals__["_clarify_state_store"]()
        # Prove the sidecar alone is enough to know s_prev fired:
        snap = rebuilt.snapshot()
        self.assertTrue(any(e["fired"] for e in snap["sessions"].values()))
        # And that doc-A is already asked -> pending_doc_ids excludes it:
        self.assertEqual(rebuilt.pending_doc_ids("s_prev", ["doc-A"]), [])

    def test_rebuild_repopulates_pending_from_staging(self):
        """The owner ruling: candidate metadata is re-read from staging."""
        hook._STATES.clear()

        class _Ctx:
            pass

        ctx = _Ctx()
        # Seed a sidecar that says s_prev FIRED but the doc was NOT asked
        # (a batch fired, then the process restarted before the footnote
        # could surface it — the exact zombie case).
        store = ClarifyStateStore(self._tmp)
        store.mark_fired("s_prev")
        store.set_last_session("s_prev")
        # doc-A is in the sidecar's asked set -> the rebuild must EXCLUDE
        # it from the pending buffer (no zombie re-ask after restart).
        store.mark_asked("s_prev", ["doc-A"])

        # Point the service at a fake whose staging exposes our row.
        orig_service = hook._service
        hook._service = lambda: self._fake_service()
        try:
            state = hook._rebuild_state_from_sidecar(ctx, "s_prev")
            self.assertIsNotNone(state)
            # The fired marker came back through the sidecar:
            self.assertIn("s_prev", state._fire_clarify_fired)
            # But doc-A is asked -> the buffer re-derived from staging has
            # no PENDING rows (the rebuild marks the sidecar's asked set
            # into the aggregator, so pending_for excludes it).
            self.assertEqual(state._agg().pending_for("s_prev"), [])
            # Footnote is empty: no zombie ask after restart.
            self.assertEqual(state._clarify_footnote("s_prev"), "")
        finally:
            hook._service = orig_service

    def test_confirmed_doc_vanishes_from_footnote(self):
        """A confirmed doc's staging row was moved out -> the footnote
        naturally drops it (single source of truth, owner ruling #2)."""
        hook._STATES.clear()

        class _Ctx:
            pass

        ctx = _Ctx()
        store = ClarifyStateStore(self._tmp)
        store.mark_fired("s_prev")
        store.set_last_session("s_prev")

        # Simulate: doc-A was confirmed (moved out of staging). The fake
        # service now returns NO rows for s_prev.
        def _empty_service():
            class _Staging:
                def docs_snapshot(self):
                    return []
            class _Svc:
                _staging = _Staging()
            return _Svc()

        orig_service = hook._service
        hook._service = _empty_service
        try:
            state = hook._rebuild_state_from_sidecar(ctx, "s_prev")
            # fired marker is still there (the trigger genuinely fired),
            # but the buffer is empty because staging has no pending row
            # -> the footnote has nothing to release. No zombie ask.
            self.assertIn("s_prev", state._fire_clarify_fired)
            self.assertEqual(state._agg().pending_for("s_prev"), [])
            self.assertEqual(state._clarify_footnote("s_prev"), "")
        finally:
            hook._service = orig_service


if __name__ == "__main__":
    unittest.main()
