#!/usr/bin/env python3
"""Unit tests for the knowledge-base-search write side.

Run:  python3 -m pytest test_kb_ingest_rules.py -v
      python3 -m pytest test_kb_ingest_hook.py -v

Covers the room's acceptance criteria:
  - rule engine: 3 narrow classes, min_cue_hits, per-rule classification
  - config merge: incremental override (merge not replace)
  - aggregator:  BOTH trigger paths (session boundary AND batch=5) with
    behavior assertions (should_clarify() is a pure function of state)
  - staging isolation: a pending doc is NOT surfaced by search() — the
    architecture-axis proof reviewer demanded (search only touches
    self._backend, never self._staging)
  - idempotency: same content -> same docid -> no duplicate row
"""

import sys
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import pytest

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = os.environ.get("KB_HOME", str(_PLUGIN_ROOT))
KB_DIR = os.environ.get("KB_HOME", str(_PLUGIN_ROOT))
for p in (PLUGIN_DIR, KB_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from kb_ingest_rules import (  # noqa: E402
    Candidate, DEFAULT_RULES, CandidateDetector, ClarifyAggregator,
    merge_rules)
from kb_service import KbSearchService, _deterministic_docid  # noqa: E402


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------
class TestDetector:
    def _det(self, rules=None):
        rules = rules or merge_rules(None)
        return CandidateDetector(rules, session_id="s1", profile="coder")

    # v2 (2026-10-14): the 3 legacy content-form classes
    # (decision / walkthrough / config) are RETIRED — they collided
    # word-for-word with filemem's L2/L1/L0 trigger words. These three
    # tests are flipped to REGRESSION assertions: the exact same inputs
    # must now MISS the v2 rule set (a bare verdict / a fix narrative /
    # a raw config dump carry no task-subject+premise structure). The
    # v2 positive cases live in test_kb_ingest_rules_v2.py.
    def test_decision_verdict_no_longer_recognized(self):
        d = self._det()
        text = "讨论了一圈，最终决定采用方案 B，否决了 A。"
        c = d.scan(text, lambda t, c_: "doc-x")
        assert c is None, "a bare verdict (no premise/condition) is filemem L2 territory"

    def test_walkthrough_narrative_no_longer_recognized(self):
        d = self._det()
        text = "报错 no such table: activations，排查发现是迁移没跑，修复后正常。"
        c = d.scan(text, lambda t, c_: "doc-y")
        assert c is None, "a root-cause->fix narrative is filemem L1 territory"

    def test_config_dump_no_longer_recognized(self):
        d = self._det()
        text = "把这段配置片段存下来：\n```\nplugins:\n  enabled: [x]\n```"
        c = d.scan(text, lambda t, c_: "doc-z")
        assert c is None, "a raw config dump carries no task-subject+premise structure"

    def test_no_cue_no_candidate(self):
        d = self._det()
        assert d.scan("今天天气不错。", lambda t, c_: "id") is None

    def test_disabled_rules_emit_nothing(self):
        d = self._det(rules=merge_rules({"enabled": False}))
        assert d.scan("最终决定采用 X。", lambda t, c_: "id") is None

    def test_faq_excluded_by_default(self):
        # "先窄后宽" — FAQ/config-docs are NOT in the default 3 classes.
        d = self._det()
        text = "什么是容器？容器是一种轻量级虚拟化技术，常用于应用隔离。"
        assert d.scan(text, lambda t, c_: "id") is None


class TestMergeRules:
    def test_no_override_returns_defaults(self):
        out = merge_rules(None)
        assert out["clarify_batch_size"] == DEFAULT_RULES["clarify_batch_size"]
        assert out["rules"] == DEFAULT_RULES["rules"]
        # deepcopy: mutating the result must not touch the source dict.
        out["rules"][0]["cues"].append("MUTATED")
        assert "MUTATED" not in DEFAULT_RULES["rules"][0]["cues"]

    def test_scalar_override_merges_not_replaces(self):
        # A config override is the WHOLE kb_ingest.rules block (devops
        # contract): scalars and the cue-list field merge independently.
        out = merge_rules({"clarify_batch_size": 2})
        assert out["clarify_batch_size"] == 2
        # untouched keys keep defaults
        assert out["min_cue_hits"] == DEFAULT_RULES["min_cue_hits"]
        assert out["enabled"] is True

    def test_rules_list_wholesale_replacement(self):
        custom = [{"id": "only", "label": "唯一", "cues": [r"x"]}]
        out = merge_rules({"rules": custom})
        assert len(out["rules"]) == 1
        assert out["rules"][0]["id"] == "only"
        # scalars untouched by a rules-only override
        assert out["clarify_batch_size"] == DEFAULT_RULES["clarify_batch_size"]

    def test_nested_override_shape_matches_profile_config(self):
        # The real profile config path: entry.kb_ingest.rules carries the
        # whole override block. _effective_rules passes entry["kb_ingest"]
        # down, so merge_rules must accept clarify_batch_size at the TOP
        # level of that block (the test's config shape).
        out = merge_rules({"rules": {"clarify_batch_size": 3}})
        assert out["clarify_batch_size"] == 3


# ---------------------------------------------------------------------------
# Clarify aggregation — BOTH trigger paths, behavior-asserted
# ---------------------------------------------------------------------------
def _fake_cand(i, session="s1"):
    return Candidate(
        doc_id=f"doc-{i}", classified_type="decision", label="决策结论",
        title=f"t{i}", content=f"c{i}", source_session_id=session,
        source_profile="coder", cue_hits=["决定采用"])


class TestAggregatorBoundary:
    def test_boundary_fires_with_pending(self):
        agg = ClarifyAggregator(batch_size=5)
        agg.add(_fake_cand(1))
        assert agg.should_clarify("s1", at_boundary=True) is True

    def test_boundary_no_pending_no_fire(self):
        agg = ClarifyAggregator(batch_size=5)
        assert agg.should_clarify("s1", at_boundary=True) is False

    def test_session_switch_is_a_boundary(self):
        # Turn 1 in s1 stages a candidate; turn 2 lands in s2 -> boundary.
        agg = ClarifyAggregator(batch_size=5)
        agg.add(_fake_cand(1, session="s1"))
        # same-session continuation must NOT fire (below batch, no boundary)
        assert agg.should_clarify("s1", at_boundary=False) is False
        # boundary for s2 with its own pending candidate -> fires
        agg.add(_fake_cand(2, session="s2"))
        assert agg.should_clarify("s2", at_boundary=True) is True


class TestAggregatorBatch:
    def test_batch_fires_at_threshold(self):
        agg = ClarifyAggregator(batch_size=3)
        for i in range(2):
            agg.add(_fake_cand(i))
            assert agg.should_clarify("s1", at_boundary=False) is False
        agg.add(_fake_cand(2))
        assert agg.should_clarify("s1", at_boundary=False) is True

    def test_mark_asked_prevents_refire(self):
        agg = ClarifyAggregator(batch_size=3)
        for i in range(3):
            agg.add(_fake_cand(i))
        assert agg.should_clarify("s1") is True
        agg.mark_asked("s1", [f"doc-{i}" for i in range(3)])
        assert agg.should_clarify("s1") is False

    def test_asked_set_is_bounded(self):
        # reviewer nit: a week-long session must not grow _asked unbounded.
        # With a small cap the oldest asked ids evict (FIFO LRU).
        agg = ClarifyAggregator(batch_size=5, asked_cap=3)
        agg.mark_asked("s1", ["a", "b", "c", "d"])
        # cap=3 -> the oldest ("a") must have been evicted, "d" retained
        assert "a" not in agg._asked
        assert "d" in agg._asked
        assert len(agg._asked) == 3

    def test_clarify_message_lists_all_pending(self):
        agg = ClarifyAggregator(batch_size=5)
        for i in range(3):
            agg.add(_fake_cand(i))
        msg = agg.build_clarify_message("s1")
        assert "doc-0" in msg and "doc-1" in msg and "doc-2" in msg


# ---------------------------------------------------------------------------
# Staging isolation + idempotency (architecture-axis proof)
# ---------------------------------------------------------------------------
class TestStagingIsolation:
    def _svc(self, tmp):
        return KbSearchService(Path(tmp))

    def test_pending_doc_not_searchable(self, tmp_path):
        svc = self._svc(str(tmp_path))
        doc = {"title": "待确认文档", "content": "这是一段足够长的待确认内容甲甲甲甲甲甲甲" * 3,
               "status": "pending"}
        out = svc.ingest([doc])
        assert out["staged"], "pending doc must land in staging"
        assert not out["ingested"], "pending must NOT enter the corpus"
        # The proof reviewer demanded: search() returns nothing for it.
        res = svc.search(query="待确认内容", top_k=5)
        assert res["data"]["results"] == [], \
            "a pending doc must never surface in the search path"

    def test_confirm_then_searchable(self, tmp_path):
        svc = self._svc(str(tmp_path))
        doc = {"title": "待确认文档", "content": "这是一段足够长的待确认内容甲甲甲甲甲甲甲" * 3,
               "status": "pending"}
        staged = svc.ingest([doc])["staged"]
        assert len(staged) == 1
        assert svc.confirm(staged[0])["status"] == "success"
        res = svc.search(query="待确认内容", top_k=5)
        assert len(res["data"]["results"]) == 1

    def test_idempotent_reingest_same_content(self, tmp_path):
        svc = self._svc(str(tmp_path))
        doc = {"title": "t", "content": "同一段内容", "status": "pending"}
        a = svc.ingest([doc])["staged"]
        b = svc.ingest([doc])["staged"]
        # deterministic docid -> the second ingest is a no-op on the store
        assert a == b
        assert svc._staging.size == 1

    def test_docid_is_content_addressed(self):
        x = _deterministic_docid({"title": "a", "content": "b"})
        y = _deterministic_docid({"title": "a", "content": "b"})
        z = _deterministic_docid({"title": "a", "content": "c"})
        assert x == y and x != z


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
