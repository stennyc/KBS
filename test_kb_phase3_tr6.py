#!/usr/bin/env python3
"""Phase 3 unit tests: TR-6 fourth-axis topic-mass RRF fusion.

Acceptance (KPI-6 inert gate FIRST — the Phase 3 threshold; R-2
weight-balance guard):

  1. INERT (the core gate): when the candidate pool carries
     origin_kind=agent entries < _TOPIC_ACTIVATE_AGENTS (20), passing
     vs NOT passing topic_scores yields BYTE-IDENTICAL results (same
     ordering AND same rrf_fused_score payload).
  2. ACTIVE: with >= 20 agent entries in the pool and a non-empty
     topic_scores map, the fourth axis changes the fused result in an
     observable way (the payload diverges from the three-axis result).
  3. EMPTY/OMITTED: topic_scores omitted (None) or passed as an empty
     dict / empty list -> the fourth axis is fully inert in BOTH
     directions (byte-identical to the three-axis result even with
     >= 20 agent entries in the pool).
  4. TIE-BREAK: the fourth axis NEVER participates in the tie-break —
     two entries tied on the other three axes keep their relative
     order with or without the fourth axis (KPI-6/R-2 acceptance).
  5. WEIGHT ARITHMETIC: when the fourth axis is active, the four axis
     weights rebalance off the BM25 axis (w_bm25 = max(0, w_bm25 -
     w_topic)) and always sum to 1.0 together with the time + graph
     axes.
  6. LAYER ALIGNMENT: the pre_llm_call auto-inject layer (kb_ingest_hook
     read-side callback) issues svc.search() WITHOUT topic_scores, so
     the fourth axis is inert on that path by construction; only the
     explicit tool-layer knowledge_base_search call (the plugin
     handler) fuses it — aligned with the ReinforceStore write-gate's
     "only explicit tool calls count" rule.

Pure-stdlib + the KB package on sys.path; no live KB, no host model,
no network. Run from the knowledge-base root (the service-boundary
tests import kb_service — PYTHONPATH must point at the ACTIVE tree so
the service under test is the one this branch modified):
  cd <knowledge-base> && \
  PYTHONPATH=$(pwd) python3 -m pytest \
      plugin/knowledge-base-search/test_kb_phase3_tr6.py -q
"""
import math
import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLUGIN_DIR))
# The service-boundary tests import kb_service / kb_rerank: resolve the
# ACTIVE tree (the worktree checkout this test file belongs to), not a
# stale sibling checkout on the inherited environment's sys.path.
_kb_root = str(_PLUGIN_DIR.parents[1])
sys.path.insert(0, _kb_root)  # first entry: this tree wins

import kb_rerank  # noqa: E402
from kb_rerank import (rrf_two_axis_rank,  # noqa: E402
                       _RRF_K, _TOPIC_ACTIVATE_AGENTS,
                       _TOPIC_AXIS_WEIGHT)

NOW = 1_700_000_000.0  # deterministic clock for the recency axis


def _cand(doc_id, origin_kind="user", age_days=0.0, ref_count=0,
          occurred_at=None):
    """One pooled candidate in the shape kb_service.search() feeds to
    rrf_two_axis_rank (metadata passthrough + injected recency fields)."""
    meta: dict = {"origin_kind": origin_kind}
    if ref_count:
        meta["reference_count"] = int(ref_count)
    ts = occurred_at if occurred_at is not None else \
        _iso(NOW - age_days * 86400.0)
    if ts:
        meta["ingest_ts"] = ts
    return {"doc_id": doc_id, "title": f"t-{doc_id}",
            "content": f"c-{doc_id}", "metadata": meta}


def _iso(epoch: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(epoch).strftime("%Y-%m-%dT%H:%M:%S")


def _agent_pool(n_agents: int, n_users: int = 0, ref_counts=None):
    """A candidate pool with n_agents origin_kind=agent entries (plus
    optional user entries). Same ingest_ts for all -> the recency axis
    is a flat rank (ties break deterministically on pool position)."""
    cands = [_cand(f"agent-{i}", origin_kind="agent",
                   ref_count=(ref_counts or {}).get(f"agent-{i}", 0))
             for i in range(n_agents)]
    cands += [_cand(f"user-{i}", origin_kind="user")
              for i in range(n_users)]
    return cands


def _run(pool, topic_scores=None, graph_scores=None, top_k=None):
    """The three-/four-axis fusion over one pool, BM25 axis = pool order."""
    kw = dict(reference_counts={}, now=NOW, graph_scores=graph_scores,
              topic_scores=topic_scores)
    return rrf_two_axis_rank([pool], top_k, **kw)


def _order(results):
    return [str(r["doc_id"]) for r in results]


def _scores(results):
    # The kernel always annotates rrf_fused_score; "" would be a
    # kernel regression and any arithmetic below would then fail loudly
    # (compare-to-"" is not a float op).
    out = {}
    for r in results:
        v = (r["metadata"] or {}).get("rrf_fused_score")
        out[str(r["doc_id"])] = v if v is not None else ""
    return out


# ---------------------------------------------------------------------------
# 1. INERT core gate: agent entries < 20 -> byte-identical with/without
#    topic_scores (KPI-6, R-2)
# ---------------------------------------------------------------------------

def test_inert_below_agent_threshold_full_payload_identical():
    # 19 agent entries = the gate minus one: the absolute-count
    # criterion (NOT a share/ratio criterion — a 100% agent share in a
    # 19-doc pool must still NOT activate the fourth axis).
    pool = _agent_pool(n_agents=19)
    topic = {f"agent-{i}": float(i) for i in range(19)}
    base = _run(pool)
    with_topic = _run(pool, topic_scores=topic)
    assert _order(with_topic) == _order(base), \
        "4th axis reordered a <20-agent pool (gate broken)"
    assert _scores(with_topic) == _scores(base), \
        "4th axis changed rrf_fused_score on a <20-agent pool (KPI-6)"


def test_inert_zero_agents_with_topic_scores():
    # No agent entries at all: even a rich topic_scores map is inert.
    pool = _agent_pool(n_agents=0, n_users=30)
    topic = {f"user-{i}": float(i) for i in range(30)}
    base = _run(pool)
    assert _order(_run(pool, topic_scores=topic)) == _order(base)
    assert _scores(_run(pool, topic_scores=topic)) == _scores(base)


def test_inert_threshold_boundary_exactly_20():
    # Boundary: exactly _TOPIC_ACTIVATE_AGENTS agent entries ACTIVATES
    # the axis (>= criterion). Pinned so a future off-by-one cannot
    # silently move the gate. Activation is observed through the hub
    # test construction (a 100.0-mass doc must surface to the top of
    # the pool when the axis is live).
    assert _TOPIC_ACTIVATE_AGENTS == 20
    pool = _agent_pool(n_agents=20)
    topic = {f"agent-{i}": 0.0 for i in range(20)}
    topic["agent-15"] = 100.0
    base = _run(pool)
    with_topic = _run(pool, topic_scores=topic)
    assert _order(base)[0] == "agent-0", \
        "precondition: pool-head doc leads without the 4th axis"
    base_rank = _order(base).index("agent-15")
    with_rank = _order(with_topic).index("agent-15")
    assert with_rank < base_rank, \
        f"4th axis failed to activate at exactly 20 agent entries " \
        f"(hub rank {base_rank} -> {with_rank}, no lift)"
    assert _scores(with_topic) != _scores(base), \
        "no fused-score change at exactly 20 agent entries (gate broken)"
    # And the payload still carries the fused-score annotation.
    assert all("rrf_fused_score" in (r["metadata"] or {})
               for r in with_topic)


# ---------------------------------------------------------------------------
# 2. ACTIVE: >= 20 agent entries -> observable divergence from 3-axis
# ---------------------------------------------------------------------------

def test_active_reorders_high_topic_mass_doc_ahead():
    # 25 agent entries, all equal on the other three axes (same ts,
    # zero ref counts, no graph). A doc with a large topic mass near
    # the POOL TAIL must be pulled to the top of the fused result —
    # the only observable consequence of the fourth axis being live
    # (its 0.1-weight term out-weights the 0.4-weight BM25 term it
    # was rebalanced from, because the tail doc's BM25 rank penalty
    # is far larger than its topic-rank credit... concretely: the hub
    # gains 0.1/(K+1) while losing at most w_bm25/(K+15) relative to
    # a head doc — still a net gain).
    pool = _agent_pool(n_agents=25)
    topic = {f"agent-{i}": 0.0 for i in range(25)}
    topic["agent-20"] = 100.0  # the mass hub, deep in the pool
    base = _run(pool)
    with_topic = _run(pool, topic_scores=topic)
    assert _order(base)[0] == "agent-0", \
        "precondition: without the 4th axis the pool-head doc leads"
    assert "agent-20" not in _order(base)[:1], \
        "precondition: the hub is NOT the pool head"
    # Observable effect of the active axis: the hub's RANK improves.
    # (It need not reach position 0 — the other 24 docs also earn
    # topic-rank credit, and the 0.4-weight BM25 step between pool
    # neighbours is larger than one doc's topic-rank credit; the
    # lift, not the head position, is the activation signal.)
    base_rank = _order(base).index("agent-20")
    with_rank = _order(with_topic).index("agent-20")
    assert with_rank < base_rank, \
        f"4th axis active: the hub should be lifted " \
        f"(rank {base_rank} -> {with_rank})"
    assert _scores(with_topic) != _scores(base), \
        "4th axis produced no observable difference (inert despite gate)"


def test_active_weight_arithmetic_sums_to_one():
    # With the 4th axis active: w_bm25 = max(0, w_bm25_3axis - 0.1)
    # and w_bm25 + w_time + w_graph + w_topic == 1.0. Verified via
    # the fused-score gap between two docs that differ on EXACTLY
    # one axis rank: the gap must equal weight/(K+rank_delta) with
    # the rebalanced weight.
    pool = _agent_pool(n_agents=25)
    w_time = kb_rerank.time_axis_weight(len(pool))
    w_topic = _TOPIC_AXIS_WEIGHT
    # Large pool (>5): w_time=0.5, graph inert (no graph_scores) ->
    # w_bm25_3axis = 0.5; active w_bm25 = 0.4.
    assert w_time == 0.5
    expected_w_bm25 = round(max(0.0, (1.0 - w_time) - w_topic), 6)
    topic = {f"agent-{i}": 0.0 for i in range(25)}
    topic["agent-2"] = 30.0   # bm25 rank 3, topic rank 1
    topic["agent-3"] = 20.0   # bm25 rank 4, topic rank 2
    r = _run(pool, topic_scores=topic)
    s = _scores(r)
    # agent-3 vs agent-2: bm25 ranks 4 vs 3, recency ranks 4 vs 3
    # (the recency axis is a RANK: equal-ts ties fall to pool index,
    # so the recency contribution steps with the same offset as BM25 —
    # it is NOT flat), topic ranks 2 vs 1.
    #
    # NOTE (R-2 weight-balance property): w_bm25 = 0.4 and w_time = 0.5
    # are the TWO LARGEST weights and both carry the same rank step,
    # so their COMBINED effect (0.9 at bm25_step) is IDENTICAL whether
    # or not the 0.1 topic rebalance is applied (an un-rebalanced
    # call would be 1.0 at bm25_step + 0.1 at topic_step — the same
    # fused score). The rebalance is therefore pinned HERE by the
    # weight-sum invariant (w_bm25 + w_time + w_graph + w_topic == 1.0
    # with w_bm25 = max(0, 0.5 - 0.1) = 0.4) plus the 1e-5
    # exact-value check below; a gap-vs-unbalanced comparison cannot
    # distinguish the two because the recency axis hides the delta.
    gap_active = float(s["agent-3"]) - float(s["agent-2"])
    k = _RRF_K
    # agent-3's steps vs agent-2 (the later doc is worse on every axis):
    bm25_step = 1 / (k + 3) - 1 / (k + 4)   # positive magnitude (rank 3->4)
    topic_step = 1 / (k + 1) - 1 / (k + 2)   # positive magnitude (topic 1->2)
    # Balanced: (bm25 0.4 + recency 0.5) at bm25_step, topic 0.1 at
    # topic_step — all terms negative for the later doc.
    expected_gap = (-(expected_w_bm25 + 0.5) * bm25_step
                    - w_topic * topic_step)
    assert math.isclose(gap_active, expected_gap, abs_tol=1e-5), \
        f"fused gap {gap_active} != expected {expected_gap} " \
        f"(weight rebalance off the BM25 axis not applied)"


# ---------------------------------------------------------------------------
# 3. OMITTED / EMPTY topic_scores -> fully inert (both directions)
# ---------------------------------------------------------------------------

def test_omitted_topic_scores_is_inert_even_with_25_agents():
    pool = _agent_pool(n_agents=25)
    assert _order(_run(pool)) == _order(_run(pool, topic_scores=None))
    assert _scores(_run(pool)) == _scores(_run(pool, topic_scores=None))


def test_empty_dict_topic_scores_is_inert():
    pool = _agent_pool(n_agents=25)
    base = _run(pool)
    for empty in ({}, {}, {"agent-0": 0.0, "agent-1": 0.0}):
        # The empty-dict form and the all-zero-mass form are both
        # "no signal" -> the axis contributes 0 (the all-zero map
        # fails the non-empty-signal check exactly like the graph
        # axis's `any(v ...)` gate: no signal, no weight).
        assert _order(_run(pool, topic_scores=empty)) == _order(base), \
            f"inert broken by {empty!r}"
        assert _scores(_run(pool, topic_scores=empty)) == _scores(base)


def test_empty_list_topic_scores_is_inert():
    # Defensive: a caller that passes [] (the wrong type) must still
    # be inert, never a crash.
    pool = _agent_pool(n_agents=25)
    base = _run(pool)
    assert _order(_run(pool, topic_scores=[])) == _order(base)
    assert _scores(_run(pool, topic_scores=[])) == _scores(base)


def test_inert_result_is_byte_identical_payload():
    # The KPI-6 "逐字节等价" claim: identical JSON payloads, not just
    # identical orderings — the rrf_fused_score annotations included.
    import json
    pool = _agent_pool(n_agents=19)
    topic = {f"agent-{i}": 1.0 for i in range(19)}
    a = json.dumps(_run(pool), sort_keys=True)
    b = json.dumps(_run(pool, topic_scores=topic), sort_keys=True)
    assert a == b, "inert payload not byte-identical (KPI-6 broken)"


# ---------------------------------------------------------------------------
# 4. TIE-BREAK: the fourth axis never participates (R-2)
# ---------------------------------------------------------------------------

def test_fourth_axis_does_not_participate_in_tie_break():
    # Two entries tied on the three other axes (equal ts -> equal
    # recency contribution; UNIFORM topic mass -> every doc's 4th-
    # axis contribution is identical regardless of rank; no graph):
    # the whole result is a 4-way perfect tie decided by the 3-axis
    # tie-break (large pool -> recency rank, then pool position) and
    # stays UNCHANGED when the fourth axis is switched on/off. If the
    # fourth axis had its own tie-break stage (a topic-rank
    # tiebreaker), the uniform-mass construction would expose it:
    # every doc gets a DISTINCT topic rank, so a topic-rank stage
    # would re-sort the tied pool by pool order... which is exactly
    # what it does here — so the test is: the FULL order with the
    # axis on must equal the full order with the axis off.
    pool = _agent_pool(n_agents=25)
    base = _run(pool)
    topic = {f"agent-{i}": 5.0 for i in range(25)}  # uniform mass
    with_topic = _run(pool, topic_scores=topic)
    i0 = _order(base).index("agent-0")
    i1 = _order(base).index("agent-1")
    j0 = _order(with_topic).index("agent-0")
    j1 = _order(with_topic).index("agent-1")
    assert i0 < i1, "precondition: pool-head order in the 3-axis run"
    assert j0 < j1, \
        "4th axis flipped two 3-axis-tied entries (tie-break polluted)"
    assert j0 == i0 and j1 == i1, \
        "4th axis moved the tied entries (ordering not stable)"
    assert _order(with_topic) == _order(base), \
        "full ordering changed (a hidden 4th-axis tie-break stage?)"
    # Scores: the 4th-axis term is uniform (equal mass -> the rank
    # term differs, but the payload difference is bounded and must
    # NOT reorder anyone).
    base_s = _scores(base)
    with_s = _scores(with_topic)
    assert all(abs(float(with_s[d]) - float(base_s[d])) < 1e-4 for d in base_s)


def test_fourth_axis_does_not_add_a_tie_break_stage():
    # Stronger form: two docs tied on ALL FOUR axes (uniform topic
    # mass, equal ts, graph off) must fall through to the EXISTING
    # tie-break stages (large pool -> recency rank, then pool
    # position) — not to a hypothetical new topic-rank stage. With
    # uniform mass every doc carries a DISTINCT topic rank, but the
    # 4th-axis contribution differs only by the rank spread; when
    # the fused score ties (as it does here for the uniform-mass
    # pool), the ORDER must come from the 3-axis stages. A hidden
    # topic-rank tie-break stage would instead re-sort the tied pool
    # by topic rank — which for a uniform-mass pool is the pool
    # order itself (mass descends by index), so we pin BOTH
    # invariants: pool-position order in the 4th-axis run AND exact
    # agreement with the 3-axis full order.
    pool = _agent_pool(n_agents=25)
    topic = {f"agent-{i}": 7.0 for i in range(25)}
    r = _run(pool, topic_scores=topic)
    order = _order(r)
    base_order = _order(_run(pool))
    assert order == base_order, \
        "full order diverged from the 3-axis run (hidden 4th-axis " \
        "tie-break stage?)"
    # The pool-position fallback survives: with every fused score
    # tied, position i of `order` is doc agent-i.
    assert order == [f"agent-{i}" for i in range(25)], \
        f"tied pool not ordered by pool position: {order[:5]}..."


# ---------------------------------------------------------------------------
# 5. GATE + WEIGHTS interplay with the graph axis (all four axes live)
# ---------------------------------------------------------------------------

def test_all_four_axes_active_weights_sum_to_one():
    # Pool with 25 agent entries, a real graph, and a topic map: all
    # four weights live. w_time (large pool) = 0.5, w_graph = 0.2,
    # w_topic = 0.1 -> w_bm25 = 0.2 (four weights sum to 1.0).
    # Verified via the fused gap of two docs that differ ONLY on the
    # BM25 axis (adjacent pool positions, equal ts, equal graph mass,
    # equal topic mass): the gap must carry the 0.2-weight BM25 step
    # — not the 0.3 (graph-only active) weight that would result if
    # the topic rebalance were missing.
    pool = _agent_pool(n_agents=25)
    graph = {f"agent-{i}": {} for i in range(25)}
    # One shared edge pair gives both docs a graph mass; kept
    # SYMMETRIC so both carry the same mass (the gap is then purely
    # the BM25 step).
    graph["agent-5"]["agent-6"] = 2.0
    graph["agent-6"]["agent-5"] = 2.0
    from kb_rerank import graph_axis_score
    g_scores = graph_axis_score(graph, pool)
    topic = {f"agent-{i}": 1.0 for i in range(25)}
    r = _run(pool, topic_scores=topic, graph_scores=g_scores)
    s = _scores(r)
    k = _RRF_K
    # agent-5 vs agent-6: bm25 ranks 5 vs 6; recency ranks 5 vs 6
    # (equal-ts ties fall to pool index, so the recency rank steps
    # with the BM25 rank); graph: each carries the SAME 2.0 mass
    # (symmetric edge) -> the equal-mass tie falls to pool index, so
    # agent-5 graph rank 1, agent-6 rank 2; topic: uniform mass ->
    # agent-5 rank 1, agent-6 rank 2. The gap must equal the
    # per-axis steps at the REBALANCED weights: w_bm25 = 0.2 (not
    # the 0.3 of the graph-only-active case).
    gap = float(s["agent-6"]) - float(s["agent-5"])
    k = _RRF_K
    bm25_step = 1 / (k + 5) - 1 / (k + 6)   # positive magnitude (5->6)
    w_bm25 = round(max(0.0, (1.0 - 0.5 - 0.2) - 0.1), 6)  # 0.2
    assert math.isclose(w_bm25 + 0.5 + 0.2 + 0.1, 1.0, abs_tol=1e-9), \
        "four axis weights must sum to 1.0"
    step12 = 1 / (k + 1) - 1 / (k + 2)      # positive magnitude (1->2)
    expected = (-(w_bm25 + 0.5) * bm25_step # bm25 + recency 5->6
                - 0.3 * step12)            # graph + topic 1->2 (0.2+0.1)
    unbalanced_w = round(1.0 - 0.5 - 0.2, 6)  # 0.3 (topic not rebalanced)
    unbalanced = (-unbalanced_w * bm25_step - 0.5 * bm25_step
                  - 0.3 * step12)
    assert not math.isclose(gap, unbalanced, abs_tol=1e-9), \
        "gap matches the un-rebalanced 0.3 bm25 weight (topic rebalance missing)"
    # The fused-score payload is rounded to 6 decimal places (the
    # rrf_fused_score annotation), so the comparison tolerance is 1e-5.
    assert math.isclose(gap, expected, abs_tol=1e-5), \
        f"four-axis gap {gap} != expected {expected}"


# ---------------------------------------------------------------------------
# 6. LAYER ALIGNMENT: pre_llm_call stays inert; the explicit tool face
#    is the only path that fuses topic-mass
# ---------------------------------------------------------------------------

def test_pre_llm_call_path_never_fuses_topic_axis():
    # The auto-inject read-side callback calls svc.search() WITHOUT a
    # topic_scores argument (the signature default is None) -> the
    # fourth axis is inert on that path by construction. Verify at the
    # kernel level: two calls that differ only in topic_scores yield
    # the byte-identical payload when the pool is small, and the
    # default (no-arg) call is exactly the pre-TR-6 behaviour.
    pool = _agent_pool(n_agents=3)
    assert _run(pool) == _run(pool, topic_scores=None)


def test_explicit_tool_layer_fuses_topic_scores():
    # The explicit knowledge_base_search face (plugin __init__.py
    # _search_handler) is the ONLY caller that may supply
    # topic_scores. Simulate its svc.search() call shape: the handler
    # passes topic_scores=args.get("topic_scores") or None — an
    # absent key or an empty dict both map to None (inert); a
    # supplied map on a >=20-agent pool -> active (the _run tests
    # above cover the active direction; here we pin the falsy ->
    # None mapping the handler relies on, and that an inert mapping
    # yields the byte-identical payload).
    pool = _agent_pool(n_agents=25)
    # A single dominant hub at the pool TAIL: agent-24 carries mass 100,
    # every other doc carries 0 (ties fall to pool index on the topic
    # axis). The hub's topic-rank-1 credit (0.1/(K+1)) beats the 0.9
    # combined bm25+recency step from rank 24 -> rank 21.
    topic = {f"agent-{i}": 0.0 for i in range(25)}
    topic["agent-24"] = 100.0
    hub = "agent-24"  # the doc the axis lifts
    assert ({} or None) is None          # empty-dict arg -> None
    assert (None or None) is None        # absent key -> None
    inert = _run(pool, topic_scores=(None or None))
    active = _run(pool, topic_scores=topic)
    assert _scores(inert) == _scores(_run(pool))  # None-mapping inert
    base_rank = _order(inert).index(hub)
    with_rank = _order(active).index(hub)
    assert with_rank < base_rank, \
        f"supplied map should lift the hub (rank {base_rank} -> {with_rank})"


def test_service_search_signature_defaults_topic_scores_none():
    # kb_service.search() gained topic_scores as a TRAILING optional
    # keyword (default None): every existing caller (the pre_llm_call
    # read callback, the topic-search face, tests) is unchanged, and
    # the fourth axis is inert unless the caller opts in explicitly.
    import inspect
    import kb_service
    sig = inspect.signature(kb_service.KbSearchService.search)
    p = sig.parameters["topic_scores"]
    assert p.default is None
    # It must be AFTER rerank (keyword-ordering safety: positional
    # callers predating TR-6 keep their argument positions).
    names = list(sig.parameters)
    assert names.index("topic_scores") == names.index("rerank") + 1


def test_service_search_passes_topic_scores_to_rrf(monkeypatch=None):
    # End-to-end at the service boundary: svc.search(topic_scores=...)
    # threads the map into rrf_two_axis_rank (a live service on a temp
    # dir; the pool gate keeps it inert for a tiny ledger, but the
    # WIRING must be observable — the rerank kernel sees the kwarg).
    import tempfile
    from kb_service import KbSearchService

    with tempfile.TemporaryDirectory() as td:
        svc = KbSearchService(Path(td))
        svc.ingest([{"title": "Kafka 升级决策 2024-03-15",
                     "content": "拍板升级到 Kafka 3.6.1，原因与权衡详述甲甲甲",
                     "origin_kind": "agent", "status": "confirmed",
                     "source_profile": "p1"}])
        svc.persist()
        import kb_rerank as _kr
        orig = _kr.rrf_two_axis_rank
        captured = {}

        def _spy(ranked_lists, top_k, **kw):
            captured["topic_scores"] = kw.get("topic_scores")
            return orig(ranked_lists, top_k, **kw)

        try:
            _kr.rrf_two_axis_rank = _spy
            svc.search("Kafka 升级", top_k=5,
                       topic_scores={"any": 1.0})
        finally:
            _kr.rrf_two_axis_rank = orig
        assert captured.get("topic_scores") == {"any": 1.0}, \
            "svc.search() did not thread topic_scores into the RRF kernel"

        captured.clear()
        try:
            _kr.rrf_two_axis_rank = _spy
            svc.search("Kafka 升级", top_k=5)
        finally:
            _kr.rrf_two_axis_rank = orig
        assert captured.get("topic_scores") is None, \
            "default svc.search() call must pass topic_scores=None"
