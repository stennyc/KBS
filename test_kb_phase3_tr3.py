#!/usr/bin/env python3
"""Phase 3 TR-3 unit tests: task-topic coarse-filter thin tool face.

TR-3 (PRD v2 §TR-3): a NEW thin tool face over the agent partition —
knowledge_base_search(topic=..., origin_kind=agent) — whose underlying
mechanism is the EXISTING svc.search() metadata exact-match filter gate
(filters={'topic': <value>, 'origin_kind': 'agent'} + implicit
source_profile scoping, the TR-8/_search_handler precedent built at the
plugin layer). The topic vocabulary (first-phase enum, PRD L39
verbatim: messaging / deployment / data-pipeline / general) is
maintained in kb_ingest_rules.TOPIC_VOCAB (config section).

Chosen clarity semantics for out-of-vocabulary topics: a CLIENT-SIDE
REJECT ({status: error, error: 'invalid_topic'} listing the valid
values), NOT a silent zero-hit search — a curated vocab means a typo
or a new value is a caller bug to fix, not 'no experience under this
topic'. topic=None / omitted = no topic gate (whole agent partition).

Red lines honored (none touched by this phase):
  - confirm() profile_mismatch hard gate, _audit face, ReinforceStore
    write gate, staging isolation, graph gate, near-dup threshold,
    hot/cold + flock, 7-gateway topology (TR-7);
  - the pre_llm_call auto-inject layer (TR-2's callback is
    byte-untouched — tested here via the real service: the face's
    svc.search path never appends reinforce events the way pre_llm_call
    must not, and the face itself records explicit-tool reinforce
    parity with _search_handler, asserted separately);
  - kb_rerank.py rrf kernel (TR-6's line — zero diff here).

Run: PYTHONPATH=/root/.hermes/plugins/knowledge-base-search \
     /root/.hermes/plugin-venv/bin/python -m pytest test_kb_phase3_tr3.py -q
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

from kb_service import KbSearchService  # noqa: E402

import kb_ingest_hook as hook  # noqa: E402
from kb_ingest_hook import (  # noqa: E402
    _topic_search_filters,
    topic_search_service_args,
    _current_profile,
)
from kb_ingest_rules import TOPIC_VOCAB  # noqa: E402

# The plugin package entry (__init__.py) imports as itself when the
# plugin dir is on sys.path: the deploy shape is 'knowledge-base-search'
# which is not a valid module name, so import the file directly.
import importlib.util  # noqa: E402

def _load_plugin_pkg() -> object:
    path = _PLUGIN_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "kb_plugin_pkg_tr3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeService:
    """Records the exact svc.search() kwargs the face issues; canned
    results per (topic filter, profile scoping) so the origin_kind=agent
    partition isolation + profile scoping are asserted on the kwargs,
    not on a live corpus."""

    def __init__(self, results_by_topic=None):
        self._by_topic = results_by_topic or {}
        self.search_calls = []
        self.reinforce_calls = []

    def search(self, query, top_k=5, filters=None, min_score=0.0,
               rerank=False):
        self.search_calls.append({
            "query": query, "top_k": top_k, "filters": dict(filters or {}),
            "min_score": min_score, "rerank": rerank,
        })
        # Key the canned table on the RESOLVED topic (None when the face
        # omits the topic key): the fake mirrors the filters exact-match
        # gate — an entry under topic=messaging only surfaces when the
        # face's filters carry topic=messaging.
        results = []
        if filters:
            t = filters.get("topic")
            results = self._by_topic.get(t, [])
        else:
            results = self._by_topic.get(None, [])
        return {"status": "success",
                "data": {"query": query, "results": results,
                         "total": len(results)}}

    def reinforce(self, docid, **kw):
        self.reinforce_calls.append((docid, kw))


def _res(doc_id, title, content, **meta):
    return {"doc_id": doc_id, "title": title, "content": content,
            "score": 0.5, "metadata": meta}


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


# ---------------------------------------------------------------------------
# Vocabulary config section (kb_ingest_rules.TOPIC_VOCAB)
# ---------------------------------------------------------------------------

class TR3TopicVocabConfigTests(unittest.TestCase):
    """The first-phase enum is the PRD L39 verbatim set — no invented
    values, no drift between the config section and the face's
    validation."""

    def test_vocab_is_the_prd_first_phase_enum(self):
        # owner 2026-10-19: frontend added (owner-reviewed extension).
        self.assertEqual(set(TOPIC_VOCAB),
                         {"messaging", "deployment",
                          "data-pipeline", "frontend", "general"})

    def test_face_rejects_out_of_vocab_and_accepts_all_enum_values(self):
        for t in TOPIC_VOCAB:
            old = _set_env_profile("coder")
            try:
                args = topic_search_service_args("query", t)
                self.assertEqual(args["filters"]["topic"], t)
            finally:
                _restore_env_profile(old)
        for bad in ("infra", "M", "GENERAL"):
            with self.assertRaises(ValueError):
                topic_search_service_args("query", bad)
        # Edge rule pinned by the chosen clarity semantics: the EMPTY
        # STRING degrades to "no topic filter" (omits the topic key —
        # the plain agent-partition search), unlike any other
        # undefined value which hard-rejects.
        # Whitespace-padded "general " degrades to its stripped
        # in-vocab form "general" (strip is normalization, not
        # rejection — an undefined value like "data pipeline" still
        # hard-rejects).
        old = _set_env_profile("coder")
        try:
            args = topic_search_service_args("query", "")
            self.assertNotIn("topic", args["filters"])
            self.assertEqual(args["filters"],
                             {"origin_kind": "agent",
                              "source_profile": "coder"})
            args2 = topic_search_service_args("query", None)
            self.assertNotIn("topic", args2["filters"])
            args3 = topic_search_service_args("query", "general ")
            self.assertEqual(args3["filters"]["topic"], "general")
        finally:
            _restore_env_profile(old)


# ---------------------------------------------------------------------------
# Filters dict construction (the thin face's contract with svc.search)
# ---------------------------------------------------------------------------

class TR3TopicFiltersTests(unittest.TestCase):
    def test_agent_partition_plus_topic_plus_profile_scoping(self):
        f = _topic_search_filters("messaging", "coder")
        self.assertEqual(f, {"origin_kind": "agent",
                             "topic": "messaging",
                             "source_profile": "coder"})

    def test_cli_global_profile_none_has_no_scoping_key(self):
        f = _topic_search_filters("deployment", None)
        self.assertEqual(f, {"origin_kind": "agent",
                             "topic": "deployment"})
        self.assertNotIn("source_profile", f)

    def test_topic_none_omits_topic_key_whole_agent_partition(self):
        f = _topic_search_filters(None, "coder")
        self.assertEqual(f, {"origin_kind": "agent",
                             "source_profile": "coder"})


# ---------------------------------------------------------------------------
# Handler level (the registered tool face: knowledge_base_topic_search)
# ---------------------------------------------------------------------------

class TR3TopicSearchHandlerTests(unittest.TestCase):
    def setUp(self):
        self._pkg = _load_plugin_pkg()

    def _patch(self, fake):
        return mock.patch.object(self._pkg, "_svc", lambda: fake)

    def test_topic_exact_match_hits_agent_partition(self):
        """KPI-1 of this phase: topic=messaging surfaces the agent
        entries whose metadata topic equals 'messaging' (exact match
        through the existing svc.search filters gate)."""
        msg_doc = _res("m-1", "messaging 网关 经验", "body m",
                       topic="messaging", origin_kind="agent",
                       source_profile="coder")
        dep_doc = _res("d-1", "deployment 网关 经验", "body d",
                       topic="deployment", origin_kind="agent",
                       source_profile="coder")
        fake = _FakeService({"messaging": [msg_doc], "deployment": [dep_doc]})
        old = _set_env_profile("coder")
        try:
            with self._patch(fake):
                out = self._pkg._topic_search_handler(
                    {"query": "网关", "topic": "messaging", "top_k": 5})
        finally:
            _restore_env_profile(old)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "success")
        ids = [r["doc_id"] for r in payload["data"]["results"]]
        self.assertEqual(ids, ["m-1"],
                         "topic exact match: only the messaging entry "
                         "surfaces; the deployment entry does not leak "
                         "in under a messaging query")
        call = fake.search_calls[-1]
        self.assertEqual(
            call["filters"],
            {"origin_kind": "agent", "topic": "messaging",
             "source_profile": "coder"},
            "the face must issue EXACTLY {topic, origin_kind: agent} "
            "+ implicit source_profile scoping (TR-8 precedent)")

    def test_topic_mismatch_does_not_hit_other_topic_entries(self):
        """Negative: query for messaging must NOT surface the
        deployment entry (exact-match semantics, no fuzzy topic
        cross-hit at this layer)."""
        dep_doc = _res("d-1", "deployment 网关 经验", "body d",
                       topic="deployment", origin_kind="agent",
                       source_profile="coder")
        fake = _FakeService({"messaging": [], "deployment": [dep_doc]})
        old = _set_env_profile("coder")
        try:
            with self._patch(fake):
                out = self._pkg._topic_search_handler(
                    {"query": "网关", "topic": "messaging", "top_k": 5})
        finally:
            _restore_env_profile(old)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["data"]["results"], [])
        self.assertEqual(payload["data"]["total"], 0)
        self.assertEqual(fake.search_calls[-1]["filters"]["topic"],
                         "messaging",
                         "the topic gate travels as the filter, not as "
                         "query relaxation: mismatch = zero hits by "
                         "construction")

    def test_out_of_vocab_topic_is_a_clarity_reject_not_silent_zero(self):
        """Chosen semantics: an undefined topic value rejects with an
        error payload listing the valid values — the caller fixes the
        input; a silent zero-hit search would read as 'no experience
        under this topic', which a curated vocab must not fake."""
        fake = _FakeService()
        old = _set_env_profile("coder")
        try:
            with self._patch(fake):
                out = self._pkg._topic_search_handler(
                    {"query": "网关", "topic": "infra", "top_k": 5})
        finally:
            _restore_env_profile(old)
        payload = json.loads(out)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "invalid_topic")
        for t in TOPIC_VOCAB:
            self.assertIn(t, payload["message"],
                          "the error payload must list the valid "
                          "values so the caller can correct")
        self.assertEqual(fake.search_calls, [],
                         "a rejected topic must not reach svc.search "
                         "(no silent zero-hit search issued)")
        self.assertEqual(fake.reinforce_calls, [],
                         "a rejected topic must not record any "
                         "reinforce event")

    def test_origin_kind_agent_partition_isolation(self):
        """origin_kind=agent is a HARD filter in every call this face
        issues: user entries (even with a matching topic) are
        structurally out of this face's candidate pool — the fake's
        per-topic table is keyed ONLY on the topic the face passes,
        so a user-origin entry stored under the same topic key would
        be the only way it could leak, and the filters dict proves
        the svc.search gate itself excludes it."""
        user_doc = _res("u-1", "user 网关 知识", "body u",
                        topic="messaging", origin_kind="user",
                        source_profile="coder")
        fake = _FakeService({"messaging": []})  # agent table empty
        old = _set_env_profile("coder")
        try:
            with self._patch(fake):
                out = self._pkg._topic_search_handler(
                    {"query": "网关", "topic": "messaging", "top_k": 5})
        finally:
            _restore_env_profile(old)
        payload = json.loads(out)
        self.assertEqual(payload["data"]["results"], [])
        call = fake.search_calls[-1]
        self.assertEqual(call["filters"].get("origin_kind"), "agent",
                         "the agent partition is gated by the filters "
                         "dict itself: a user-origin entry with a "
                         "matching topic value cannot surface through "
                         "this face")
        # Document the isolation as an invariants assertion on the
        # filter shape (the corpus side applies it; the face never
        # widens it):
        self.assertNotEqual(
            (call["filters"].get("source_profile") or "").lower(), "user",
            "the face must never scope to a 'user' source_profile "
            "(origin_kind=agent is the partition gate, not source)")

    def test_profile_scoping_user_partition_not_leaked(self):
        """TR-8 carry-over on this face: profile Y calling the face
        gets ONLY Y's agent entries — a profile X entry that matches
        the topic filter is excluded by the implicit source_profile
        scoping key the face injects (same precedent as
        _search_handler; the cross-profile global door stays the
        plain knowledge_base_search explicit filters opt-in, NOT
        this face)."""
        x_doc = _res("x-1", "X 的 agent 经验", "body x",
                     topic="messaging", origin_kind="agent",
                     source_profile="X")
        y_doc = _res("y-1", "Y 的 agent 经验", "body y",
                     topic="messaging", origin_kind="agent",
                     source_profile="coder")
        # Fake resolves per (topic, profile-scoping): the face's
        # filters carry BOTH keys; only the Y entry survives the gate.
        fake = _FakeService({"messaging": [x_doc, y_doc]})

        class _ScopingService(_FakeService):
            def search(self, query, top_k=5, filters=None, min_score=0.0,
                       rerank=False):
                res = super().search(query, top_k, filters, min_score,
                                      rerank)
                f = dict(filters or {})
                sp = f.get("source_profile")
                res["data"]["results"] = [
                    r for r in res["data"]["results"]
                    if (r.get("metadata") or {}).get("source_profile")
                    == sp
                ]
                res["data"]["total"] = len(res["data"]["results"])
                return res

        scoping = _ScopingService({"messaging": [x_doc, y_doc]})
        old = _set_env_profile("coder")  # profile Y == 'coder' here
        try:
            with self._patch(scoping):
                out = self._pkg._topic_search_handler(
                    {"query": "网关", "topic": "messaging", "top_k": 5})
        finally:
            _restore_env_profile(old)
        payload = json.loads(out)
        ids = [r["doc_id"] for r in payload["data"]["results"]]
        self.assertEqual(ids, ["y-1"],
                         "the implicit source_profile scoping must keep "
                         "profile X's agent entry out of profile Y's "
                         "topic-search results (R-7 / TR-8)")
        self.assertEqual(scoping.search_calls[-1]["filters"],
                         {"origin_kind": "agent", "topic": "messaging",
                          "source_profile": "coder"})

    def test_no_topic_arg_searches_whole_agent_partition(self):
        """topic omitted -> no topic gate key at all (the face is the
        coarse FILTER, not a gate-by-default: the whole agent
        partition is searchable when the caller wants it)."""
        fake = _FakeService({None: [_res("a-1", "A 经验", "body",
                                          origin_kind="agent")]})
        old = _set_env_profile("coder")
        try:
            with self._patch(fake):
                out = self._pkg._topic_search_handler(
                    {"query": "网关", "top_k": 5})
        finally:
            _restore_env_profile(old)
        payload = json.loads(out)
        self.assertEqual(payload["data"]["total"], 1)
        call = fake.search_calls[-1]
        self.assertNotIn("topic", call["filters"])
        self.assertEqual(call["filters"],
                         {"origin_kind": "agent",
                          "source_profile": "coder"})

    def test_explicit_tool_face_reinforces_like_plain_search(self):
        """Reinforce write-gate parity: the topic face is an explicit
        tool call — it records top-hit reinforce events exactly like
        _search_handler does (R-3's negative constraint stays on the
        pre_llm_call layer only)."""
        fake = _FakeService({"messaging": [_res("m-1", "m 经验", "body",
                                                 topic="messaging",
                                                 origin_kind="agent")]})
        old = _set_env_profile("coder")
        try:
            with self._patch(fake):
                out = self._pkg._topic_search_handler(
                    {"query": "网关", "topic": "messaging", "top_k": 5})
        finally:
            _restore_env_profile(old)
        self.assertEqual(len(fake.reinforce_calls), 1)
        docid, kw = fake.reinforce_calls[0]
        self.assertEqual(docid, "m-1")
        self.assertEqual(kw.get("reason"), "explicit_search")


# ---------------------------------------------------------------------------
# Live-service integration (topic metadata survives ingest + exact
# match + partition isolation end-to-end, same e2 style as Phase 1/2)
# ---------------------------------------------------------------------------

class TR3LiveServiceTests(unittest.TestCase):
    def _svc(self, base):
        data = base / "kb-data"
        data.mkdir()
        audit = base / "audit.jsonl"
        svc = KbSearchService(data, audit_path=audit)
        svc.ingest([
            {"docid": "ag-msg", "title": "messaging 网关 经验 条目",
             "content": "网关 messaging 重试 预算 经验 甲乙丙丁戊己甲甲",
             "status": "confirmed", "origin_kind": "agent",
             "topic": "messaging", "source_profile": "coder"},
            {"docid": "ag-dep", "title": "deployment 网关 经验 条目",
             "content": "网关 deployment 灰度 发布 经验 庚辛壬癸甲乙甲",
             "status": "confirmed", "origin_kind": "agent",
             "topic": "deployment", "source_profile": "coder"},
            {"docid": "us-msg", "title": "user 录入 网关 知识",
             "content": "网关 知识 用户 录入 甲乙丙丁甲甲甲甲甲甲甲",
             "status": "confirmed", "origin_kind": "user",
             "topic": "messaging", "source_profile": "coder"},
        ])
        return svc

    def test_live_topic_exact_match_and_partition_isolation(self):
        base = Path(tempfile.mkdtemp())
        svc = self._svc(base)
        old = _set_env_profile("coder")
        try:
            res = svc.search("网关 经验", top_k=10,
                             filters={"topic": "messaging",
                                      "origin_kind": "agent"})
            ids = {r["doc_id"] for r in res["data"]["results"]}
            self.assertEqual(ids, {"ag-msg"},
                             "live corpus: only the agent entry with "
                             "topic=messaging surfaces (the user-origin "
                             "entry shares the topic value but is gated "
                             "out by origin_kind; the deployment entry "
                             "is gated out by topic)")
            res2 = svc.search("网关 经验", top_k=10,
                              filters={"topic": "deployment",
                                       "origin_kind": "agent"})
            self.assertEqual(
                {r["doc_id"] for r in res2["data"]["results"]},
                {"ag-dep"})
            # topic=None-style: no topic key -> whole agent partition
            res3 = svc.search("网关 经验", top_k=10,
                              filters={"origin_kind": "agent"})
            self.assertEqual(
                {r["doc_id"] for r in res3["data"]["results"]},
                {"ag-msg", "ag-dep"})
        finally:
            _restore_env_profile(old)
        svc.close()

    def test_pre_llm_call_layer_untouched_by_tr3(self):
        """Red line check: the auto-inject layer (pre_llm_call) is
        byte-unchanged by TR-3 — its agent-partition fetch still uses
        the TR-2 filters ({origin_kind: agent, source_profile}) with
        NO topic key, and the TR-2 decay gate still governs the agent
        segment. Run the REAL callback against a live service and
        assert the agent fetch's filters dict carries no 'topic'.
        """
        base = Path(tempfile.mkdtemp())
        svc = self._svc(base)
        ts = time.time()
        # Seed the live fetches the way Phase 2's live test does (the
        # agent doc is fresh + premise-true so it passes the decay
        # gate and renders in the block).
        plain = svc.search("网关 经验", top_k=3)["data"]["results"]
        user_results = [
            {**r, "metadata": {**r.get("metadata", {}),
                               "premise_status": "true",
                               "ingest_ts": ts,
                               "origin_kind":
                               r.get("metadata", {}).get("origin_kind")}}
            for r in plain
        ]
        agent_results = [
            {**r, "metadata": {**r.get("metadata", {}),
                               "premise_status": "true",
                               "ingest_ts": ts,
                               "origin_kind": "agent"}}
            for r in svc.search("网关 经验", top_k=3,
                                filters={"origin_kind": "agent"})
            ["data"]["results"]
        ]

        class _RecService:
            def __init__(self):
                self.calls = []

            def search(self, query, top_k=5, filters=None,
                       min_score=0.0, rerank=False):
                self.calls.append(dict(filters or {}))
                if filters:
                    return {"status": "success",
                            "data": {"query": query,
                                     "results": agent_results,
                                     "total": len(agent_results)}}
                return {"status": "success",
                        "data": {"query": query,
                                 "results": user_results,
                                 "total": len(user_results)}}

        rec = _RecService()
        old = _set_env_profile("coder")
        try:
            with mock.patch.object(hook, "_service", lambda: rec):
                cb = hook._make_read_callback(object())
                out = cb(user_message="网关 经验 怎么 积累 的")
            self.assertIn("context", out,
                          "the TR-2 auto-inject path still renders the "
                          "agent segment (TR-3 must not have broken it)")
            agent_calls = [f for f in rec.calls if f]
            self.assertTrue(agent_calls,
                            "the pre_llm_call agent fetch must still "
                            "run (TR-2 behavior preserved)")
            for f in agent_calls:
                self.assertEqual(f.get("origin_kind"), "agent")
                self.assertNotIn("topic", f,
                                 "the auto-inject layer stays TR-2: "
                                 "its agent-partition fetch carries NO "
                                 "topic key (TR-3's topic gate is the "
                                 "thin TOOL face only)")
        finally:
            _restore_env_profile(old)
        svc.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
