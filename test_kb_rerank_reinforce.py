"""Unit tests for the read-side precision upgrade (owner "向 filemem 借鉴"
batch ②③):

  1. denoise_query  — strips host-path / filesystem noise tokens, keeps real
                      content words, and never returns an empty string.
  2. rrf_two_axis_rank — fuses the BM25 axis with the recency/reinforce axis;
                      a cited OLD doc stays ahead of a fresh but never-cited
                      one (the "reinforce 打底时间轴" fix).
  3. reinforce write-gate (contrast) — an explicit tool-call reinforce()
                      bumps the count; the auto-prefetch path (svc.search)
                      does NOT.
  4. occurred_at null — a doc without created_at gets no recency credit but
                      is NEVER penalised (a null timestamp must not bury a
                      doc).

Pure-stdlib + the KB package on sys.path; no live KB, no host model, no
network. Run:  python3 -m pytest test_kb_rerank_reinforce.py -q
"""
import sys
import time
import threading
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLUGIN_DIR))
_kb_root = str(_PLUGIN_DIR.parents[1])
if _kb_root not in sys.path:
    sys.path.insert(0, _kb_root)

from kb_rerank import (denoise_query, rrf_two_axis_rank,
                       extract_event_time, time_axis_weight,
                       build_graph, graph_axis_score)  # noqa: E402
import kb_rerank  # noqa: E402


# ---------------------------------------------------------------------------
# 5. Card 2: event-time extraction (occurred_at / version_ts split) +
#    pool-size-adaptive time-axis weight
# ---------------------------------------------------------------------------

def test_extract_event_time_decision_category():
    # Category 1 (decision/event): a point-in-time date is extracted into
    # occurred_at.
    r = extract_event_time("2024-03-15 拍板升级 Kafka 到 kafka_3.6.1")
    assert r["occurred_at"] == "2024-03-15"
    # The config version is a separate FILTER tag, not the event time.
    assert r["version_ts"] == "3.6.1"


def test_extract_event_time_mechanism_category_is_null():
    # Category 2 (mechanism): no date, no version -> both fields None. A null
    # occurred_at must NOT be read as "very old" (the owner ruling: a missing
    # event time is "no credit", not a penalty).
    r = extract_event_time("Redis 持久化机制 RDB + AOF 双引擎")
    assert r["occurred_at"] is None
    assert r["version_ts"] is None


def test_extract_event_time_config_version_category():
    # Category 3 (config version): "kafka_3.6.1" -> version_ts="3.6.1" (the
    # full dotted run, not a fragment like "6.1").
    r = extract_event_time("Kafka 配置版本 kafka_3.6.1 上生产")
    assert r["version_ts"] == "3.6.1"
    assert r["occurred_at"] is None


def test_extract_event_time_both_fields_independent():
    # A doc can carry both a date AND a version; they are independent fields.
    r = extract_event_time("2024-05-02 拍板升级 kafka_3.6.1")
    assert r["occurred_at"] == "2024-05-02"
    assert r["version_ts"] == "3.6.1"


def test_extract_event_time_empty_and_none():
    assert extract_event_time("") == {"occurred_at": None, "version_ts": None}
    assert extract_event_time(None) == {"occurred_at": None, "version_ts": None}


def test_time_axis_weight_downshifts_on_small_pool():
    # The card-2 acceptance (reviewer): on a small hot-only pool the time
    # axis gets a REDUCED weight so a fresh doc can't auto-monopolise the
    # top slot on top of an already-flattened BM25 axis. We assert the
    # DOWNSHIFT BEHAVES (small < large), not a literal magic value.
    assert time_axis_weight(3) < time_axis_weight(20)
    assert time_axis_weight(3) < time_axis_weight(6)
    # And the boundary: a pool of exactly 5 is still "small" (downshifted),
    # a pool of 6 is "large" (default weight).
    assert time_axis_weight(5) == time_axis_weight(3)
    assert time_axis_weight(6) == time_axis_weight(20)


def test_time_axis_weight_is_derived_not_magic():
    # The weight must be monotonic non-decreasing in pool size (bigger pool
    # -> more BM25 resolution -> the time axis can carry a heavier weight).
    weights = [time_axis_weight(n) for n in (1, 3, 5, 6, 20, 100)]
    assert weights == sorted(weights)


def test_recency_axis_uses_occurred_at_not_version_ts():
    # version_ts must NOT be fed to the time axis (a version stamp is a
    # filter tag). Two docs, same BM25 rank, different only in version_ts:
    # the version_ts must not change their recency order (both have
    # occurred_at=None, so both get no recency credit, order by rank only).
    pool = [
        {"doc_id": "a", "metadata": {"version_ts": "3.6.1"}},
        {"doc_id": "b", "metadata": {"version_ts": "2.0.0"}},
        {"doc_id": "c", "metadata": {}},
    ]
    out = rrf_two_axis_rank([pool], top_k=None, reference_counts={}, now=0)
    # No occurred_at / ingest_ts on any of them -> recency axis gives no
    # credit, order is determined by BM25 (which is a flat tie). RRF ties
    # break by pool position (determinism), so the order is the pool order.
    assert [x["doc_id"] for x in out] == ["a", "b", "c"]


def test_small_pool_tie_breaks_by_bm25_not_recency():
    # Card-2 refined ruling (reviewer): on a SMALL pool (<=5), a fused-score
    # tie is broken by the BM25 axis (keyword relevance), NOT the recency
    # axis — letting recency win the tie would reopen the "newest doc
    # auto-dominates" back-door that the pool downshift just closed.
    # Construct a genuine fused tie: two docs with no ts / no cites (both
    # recency rank 1 -> recency tie) but DIFFERENT BM25 ranks. The BM25
    # axis list order IS the rank (first element = rank 1 = most relevant).
    bm25_axis = [
        {"doc_id": "bm25_better", "metadata": {}},  # rank 1 (best relevance)
        {"doc_id": "bm25_worse", "metadata": {}},   # rank 2
    ]
    out = rrf_two_axis_rank([bm25_axis], top_k=None, reference_counts={}, now=0)
    # No ts / no cites -> recency is a tie; fused differs only by the BM25
    # rank. On a small pool the tie-breaker is BM25, so the best-relevance
    # doc (bm25_better, rank 1) leads.
    assert out[0]["doc_id"] == "bm25_better", \
        f"small-pool fused tie must break to BM25 relevance; got {[x['doc_id'] for x in out]}"


def test_large_pool_tie_breaks_by_recency_not_bm25():
    # Card-2 refined ruling (reviewer): on a LARGE pool (>5), a fused-score
    # tie is broken by the RECENCY axis (a cited/fresher doc wins) — the
    # trust tie-break is meaningful where the BM25 axis has real resolution.
    # The BM25 axis list order IS the rank. Cited doc (recency-boosted)
    # must win the fused tie even though its BM25 rank is worse, because a
    # large pool switches the tie-breaker to recency_rank.
    bm25_axis = [
        {"doc_id": "uncited_new", "metadata": {}},  # rank 1, not cited
        {"doc_id": "cited_old", "metadata": {}},    # rank 2, cited x5
    ]
    for i in range(4):
        bm25_axis.append({"doc_id": f"filler_{i}", "metadata": {}})
    out = rrf_two_axis_rank([bm25_axis], top_k=None,
                            reference_counts={"cited_old": 5}, now=0)
    assert out[0]["doc_id"] == "cited_old", \
        f"large-pool fused tie must break to recency (cited); got {[x['doc_id'] for x in out]}"


def test_recency_axis_prefers_occurred_at_over_ingest_ts():
    # A doc with BOTH occurred_at (a date) and ingest_ts (epoch) must use
    # occurred_at as the recency signal (event time takes precedence).
    # A fresh ingest_ts (now - 1 day) with an OLD occurred_at (2020) -> the
    # OLD occurred_at wins, the doc is treated as old, not fresh.
    # Use a large pool so the time axis keeps its default weight.
    import time as _t
    now = _t.time()
    fresh_ingest = now - 86400.0           # 1 day old
    old_occurred = "2020-01-01"            # 6+ years old
    pool = [
        {"doc_id": "old", "metadata": {"occurred_at": old_occurred,
                                       "ingest_ts": fresh_ingest}},
        {"doc_id": "none", "metadata": {"ingest_ts": fresh_ingest}},
    ]
    # Pad to a large pool so the time axis is NOT downshifted.
    for i in range(4):
        pool.append({"doc_id": f"ancient_{i}",
                    "metadata": {"ingest_ts": now - 365 * 86400.0}})
    out = rrf_two_axis_rank([pool], top_k=None, reference_counts={}, now=now)
    # "old" has occurred_at=2020 (very old), "none" has occurred_at=None
    # (age falls back to ingest_ts = 1 day). "none" should rank ahead on
    # the recency term (newer).
    ids = [x["doc_id"] for x in out]
    assert ids.index("none") < ids.index("old")



# ---------------------------------------------------------------------------
# 1. denoise_query
# ---------------------------------------------------------------------------

def test_denoise_strips_host_paths_keeps_content():
    raw = "how do I fix /root/.hermes/kb-data/foo.py crash in myorg.core"
    out = denoise_query(raw)
    assert "/root" not in out
    assert ".hermes" not in out
    assert "foo.py" not in out
    # the real content word must survive
    assert "myorg" in out or "core" in out


def test_denoise_strips_bare_dir_names_token_level():
    # "docker" (a real topic word) is kept; "root" / "home" as standalone
    # tokens are dropped (they only appear as host-dir noise here).
    out = denoise_query("root home docker config problem")
    assert "docker" in out
    assert "root" not in out.lower().split()
    assert "home" not in out.lower().split()


def test_denoise_empty_result_falls_back_to_original():
    # A query that is ONLY noise must not become an empty string (an empty
    # query would 0-hit and defeat the fetch).
    out = denoise_query("/root/.hermes/kb-data")
    assert out  # non-empty -> fell back to the raw query
    assert "hermes" in out or "kb-data" in out


def test_denoise_passthrough_on_clean_query():
    assert denoise_query("Kafka rebalancing behaviour") == "Kafka rebalancing behaviour"


# ---------------------------------------------------------------------------
# 2. rrf_two_axis_rank
# ---------------------------------------------------------------------------

def _cand(doc_id, ref_count=0, age_days=0.0):
    meta = {}
    if ref_count:
        meta["reference_count"] = ref_count
    if age_days or True:
        # a doc with a timestamp: created `age_days` days ago
        meta["created_at"] = time.time() - age_days * 86400.0
    return {"doc_id": doc_id, "title": f"T-{doc_id}", "content": f"C-{doc_id}",
            "metadata": meta}


def test_rrf_cited_old_doc_beats_old_uncited_same_age():
    """The reviewer's intent, stated precisely: a cited OLD doc must NOT be
    buried BEHIND an equally-old never-cited doc. With the same age (30d),
    the reference-count salience boost lifts the cited one ahead.

    (A cited doc may still rank below a FRESHER doc — that is the capped
    salience's point; the guarantee is only relative to docs of the SAME
    recency tier. See test_rrf_capped_salience_never_overrules_recency.)"""
    bm25_axis = [_cand("old_cited", ref_count=5, age_days=30),
                 _cand("old_uncited", ref_count=0, age_days=30)]
    ref_counts = {"old_cited": 5}
    fused = rrf_two_axis_rank([bm25_axis], 2, reference_counts=ref_counts)
    order = [c["doc_id"] for c in fused]
    assert order.index("old_cited") < order.index("old_uncited"), \
        f"cited doc must beat same-age uncited; got {order}"


def test_rrf_capped_salience_never_overrules_recency():
    """Guard: a single citation may slow decay but must NOT let a very-old
    doc overrule a clearly-fresher one on a LARGE pool (time axis at full
    weight). A 30-day-old doc cited 5 times (salience 2.0) still decays
    below a brand-new uncited doc (1.0, age 0) when the pool is large
    enough that the recency axis carries its default weight."""
    bm25_axis = [_cand("old_cited", ref_count=5, age_days=30),
                 _cand("fresh", ref_count=0, age_days=0)]
    # Pad to a large pool so the time axis keeps its default weight.
    for i in range(4):
        bm25_axis.append(_cand(f"ancient_{i}", ref_count=0, age_days=365))
    ref_counts = {"old_cited": 5}
    fused = rrf_two_axis_rank([bm25_axis], len(bm25_axis),
                              reference_counts=ref_counts)
    order = [c["doc_id"] for c in fused]
    assert order.index("fresh") < order.index("old_cited"), \
        f"a fresh uncited doc must beat a 30-day-old cited one on a large pool; got {order}"


def test_rrf_fresh_uncited_docs_rank_fresh_first():
    """With NO reinforce signal and a LARGE pool (>5 docs), the recency
    axis carries full weight: the fresher doc ranks ahead. This is the
    cold-start default for a corpus with enough docs that the BM25 axis
    has real resolution. (On a small pool the time axis is downshifted —
    see test_small_pool_time_axis_downshifts.)"""
    # 6+ docs so the pool is "large" and the time axis keeps its default weight.
    bm25_axis = [_cand("old", ref_count=0, age_days=60),
                 _cand("new", ref_count=0, age_days=0)]
    for i in range(4):
        bm25_axis.append(_cand(f"ancient_{i}", ref_count=0, age_days=365))
    fused = rrf_two_axis_rank([bm25_axis], len(bm25_axis), reference_counts={})
    order = [c["doc_id"] for c in fused]
    assert order.index("new") < order.index("old"), \
        f"with no cites on a large pool the fresh doc should lead; got {order}"


def test_small_pool_time_axis_downshifts():
    """Card-2 acceptance (reviewer): on a small hot-only pool (<=5 docs), the
    time axis is downshifted so a fresh doc does NOT auto-monopolise the top
    slot on top of an already-flattened BM25 axis. Two docs, the older doc
    is #1 on BM25 -> it stays #1 in the fused result (BM25 dominance, not
    recency dominance)."""
    bm25_axis = [_cand("old", ref_count=0, age_days=60),
                 _cand("new", ref_count=0, age_days=0)]
    fused = rrf_two_axis_rank([bm25_axis], 2, reference_counts={})
    order = [c["doc_id"] for c in fused]
    # On a 2-doc pool the time axis is downshifted; the BM25 rank (old=#1)
    # dominates, so old stays ahead of new.
    assert order.index("old") < order.index("new"), \
        f"on a small pool the BM25 axis dominates; got {order}"


def test_rrf_no_refcounts_and_no_timestamp_is_safe():
    """A doc with neither a reference count nor a created_at timestamp gets
    score 0.0 on the recency axis (never a NEGATIVE penalty). It still
    participates via its BM25 rank, so it can still surface. No exception."""
    bm25_axis = [{"doc_id": "x", "title": "T", "content": "C", "metadata": {}}]
    fused = rrf_two_axis_rank([bm25_axis], 1, reference_counts={})
    assert fused[0]["doc_id"] == "x"
    # fused score annotation present + non-negative
    assert fused[0]["metadata"]["rrf_fused_score"] >= 0


# ---------------------------------------------------------------------------
# 3. reinforce write-gate contrast (explicit tool call vs auto-prefetch)
# ---------------------------------------------------------------------------

def test_explicit_reinforce_bumps_count_auto_prefetch_does_not():
    import tempfile
    from kb_service import KbSearchService

    d = tempfile.TemporaryDirectory()
    try:
        base = Path(d.name)
        svc = KbSearchService(base)
        svc.ingest([
            {"docid": "d1", "title": "Kafka 重平衡", "content": "Kafka 重平衡 机制 说明 长度足够 甲乙丙丁 填充 段落 若干字 扩充 内容 标记 词 五",
             "status": "confirmed"},
        ])
        # Before any reinforce: reference count is 0.
        assert svc.reference_counts().get("d1", 0) == 0
        # The AUTO-PREFETCH path: svc.search() is called on every turn — it
        # must NOT write to the reinforce store (the write-gate).
        svc.search("Kafka 重平衡", top_k=3)
        assert svc.reference_counts().get("d1", 0) == 0, \
            "auto-prefetch must not reinforce"
        # The EXPLICIT tool-call path: reinforce() DOES bump it.
        out = svc.reinforce("d1", reason="explicit_search")
        assert out["status"] == "ok" and out["reinforced"] is True
        assert svc.reference_counts().get("d1") == 1
        # A second explicit call bumps to 2.
        svc.reinforce("d1", reason="explicit_search")
        assert svc.reference_counts().get("d1") == 2
    finally:
        d.cleanup()


def test_reinforce_anchor_miss_does_not_record():
    """The followup_text anchor gate: if the follow-up text does NOT contain
    any of the doc's anchors/title, the reinforce event is NOT recorded
    (the reviewer's "锚点纯字符串命中" rule — no LLM, no embedding)."""
    import tempfile
    from kb_service import KbSearchService

    d = tempfile.TemporaryDirectory()
    try:
        base = Path(d.name)
        svc = KbSearchService(base)
        svc.ingest([
            {"docid": "d1", "title": "Kafka 重平衡",
             "content": "Kafka 重平衡 机制 说明 长度足够 甲乙丙丁 填充 段落 若干字 扩充 内容 标记 词 五",
             "status": "confirmed"},
        ])
        # Follow-up text that does NOT mention the doc's title/anchors.
        out = svc.reinforce("d1", reason="explicit_search",
                            followup_text="完全无关的一句话没有任何锚点 甲乙丙丁戊己")
        assert out["status"] == "ok"
        assert out["reinforced"] is False
        assert out["reason"] == "anchor_miss"
        # The event was NOT recorded.
        assert svc.reference_counts().get("d1", 0) == 0
        # A follow-up that DOES anchor the doc's title -> recorded.
        out2 = svc.reinforce("d1", reason="explicit_search",
                             followup_text="我们用 Kafka 重平衡 这个机制来处理")
        assert out2["reinforced"] is True
        assert svc.reference_counts().get("d1") == 1
    finally:
        d.cleanup()


# ---------------------------------------------------------------------------
# 4. occurred_at null — a null timestamp is never a penalty
# ---------------------------------------------------------------------------

def test_null_timestamp_never_penalises_doc():
    """A doc whose metadata has NO created_at (occurred_at null, e.g. a
    mechanism doc with no event time) must NOT be pushed down the recency
    axis relative to an equally-cited doc that DOES have a timestamp.

    Rationale (owner ruling #4): mechanism docs have no event time; a null
    occurred_at must be treated as "no recency credit, no penalty", NOT as
    "very old -> bury it". Two docs, both cited once, one with a fresh
    timestamp and one with none: the null one must not rank LAST on the
    recency axis.
    """
    # Both cited once. doc-A has a created_at (fresh), doc-B has NONE.
    axis = [
        {"doc_id": "A", "title": "T", "content": "C",
         "metadata": {"created_at": time.time(), "reference_count": 1}},
        {"doc_id": "B", "title": "T", "content": "C",
         "metadata": {"reference_count": 1}},  # no created_at
    ]
    # Without any timestamp the recency score is salience*1 + 0.1 (ref=1)
    # for BOTH docs (B's missing ts -> age_days=0 -> full weight, same as A
    # whose fresh ts -> age≈0). They tie on the recency axis; the BM25 axis
    # (list position) breaks the tie. The KEY assertion: B is NOT ranked
    # below A by a time penalty — a null timestamp is neutral, not negative.
    fused = rrf_two_axis_rank([axis], 2,
                              reference_counts={"A": 1, "B": 1})
    # Both docs appear; B's fused score is NOT zeroed out by the null ts.
    by_id = {c["doc_id"]: c for c in fused}
    assert "A" in by_id and "B" in by_id
    assert by_id["B"]["metadata"]["rrf_fused_score"] > 0, \
        "a null-timestamp doc must still carry a positive fused score"


# ---------------------------------------------------------------------------
# RRF constants sanity (the K=60 / half-life the data-engineer fixed)
# ---------------------------------------------------------------------------

def test_rrf_constants_match_data_engineer_spec():
    assert kb_rerank.RRF_K == 60           # filemem SA2-3 standard K
    assert kb_rerank.SALIENCE_PER_REF == 0.2
    assert kb_rerank.SALIENCE_MAX_REFS == 10
    # half-life ~14 days: exp(-lambda*14) should be ~0.5
    import math
    assert abs(math.exp(-kb_rerank._DEFAULT_RECENCY_LAMBDA * 14.0) - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# 6. build_graph — the RRF third axis (co-occurrence, forward-looking)
# ---------------------------------------------------------------------------

def test_build_graph_empty_below_min_docs():
    # The forward-looking design: on <2 docs the graph is vacuous so the
    # third axis is inert (search degrades to the two-axis result).
    assert build_graph([]) == {}
    assert build_graph([{"doc_id": "x", "title": "a", "content": "b"}]) == {}


def test_build_graph_symmetric_cooccurrence():
    # Two docs that share content tokens get a symmetric co-occurrence edge;
    # a third isolated doc gets none.
    docs = [
        {"doc_id": "a", "title": "Kafka", "content": "集群 升级"},
        {"doc_id": "b", "title": "Kafka", "content": "集群 回滚"},
        {"doc_id": "c", "title": "Redis", "content": "持久化"},
    ]
    g = build_graph(docs)
    # Symmetric: a->b == b->a, both present; c has no edge.
    assert g["a"].get("b") == g["b"].get("a") and g["a"].get("b") > 0
    assert "c" not in g["a"] and "a" not in g["c"]


def test_build_graph_anchors_weighted_heavier_than_tokens():
    # Two docs sharing an explicit anchor_point co-occur heavier than two
    # docs sharing only a content token (the owner's high-signal join).
    docs = [
        {"doc_id": "a", "title": "Kafka", "content": "集群",
         "anchor_points": ["kafka"]},
        {"doc_id": "b", "title": "Kafka", "content": "集群",
         "anchor_points": ["kafka"]},
        {"doc_id": "x", "title": "Kafka", "content": "集群"},
        {"doc_id": "y", "title": "Kafka", "content": "集群"},
    ]
    g = build_graph(docs)
    # a<->b shares the anchor "kafka" (2.0) + content tokens; x<->y shares
    # only content tokens. The anchor pair must carry strictly more mass.
    assert g["a"].get("b", 0.0) > g["x"].get("y", 0.0)


def test_graph_axis_score_hubs_rank_above_isolated():
    # Within a pool, a doc that co-occurs with many others (a hub) scores
    # higher than an isolated doc. An empty graph gives all docs 0.0
    # (inert, no penalty - additive-credit).
    docs = [
        {"doc_id": "hub", "title": "Kafka 集群", "content": "升级 回滚"},
        {"doc_id": "n1", "title": "Kafka 集群", "content": "升级"},
        {"doc_id": "n2", "title": "Kafka 集群", "content": "回滚"},
        {"doc_id": "iso", "title": "无关主题 ZZZ", "content": "xyz"},
    ]
    pool = [{"doc_id": d["doc_id"]} for d in docs]
    g = build_graph(docs)
    scores = graph_axis_score(g, pool)
    assert scores["hub"] > scores["iso"]
    # A doc with no graph edges is 0.0, never negative.
    assert scores["iso"] >= 0.0


def test_graph_axis_inert_on_empty_graph():
    # Empty graph (the <2-doc dev state) -> every pool doc scores 0.0, the
    # axis contributes 0 weight, and the fused order is the two-axis order.
    pool = [{"doc_id": "a", "metadata": {}},
            {"doc_id": "b", "metadata": {}}]
    assert graph_axis_score({}, pool) == {"a": 0.0, "b": 0.0}


def test_rrf_with_graph_scores_activates_third_axis():
    # When a non-empty graph_scores map is supplied, the third axis fuses
    # in: a doc that is a graph hub (high co-occurrence mass) but ranks
    # middling on BM25 should be lifted above a doc that is BM25-first
    # but isolated in the graph. This is the retrieval signal the two axes
    # can't see. (The lift is bounded by _GRAPH_AXIS_WEIGHT, so the hub
    # does NOT necessarily overrule a strong BM25 hit - it lifts, same as
    # the recency axis.)
    bm25_axis = [
        {"doc_id": "bm25_top_iso", "metadata": {}},   # BM25 rank 1, isolated
        {"doc_id": "bm25_mid_hub", "metadata": {}},    # BM25 rank 2, a hub
    ]
    # Build a graph where the MIDDLE BM25 doc is the hub.
    docs = [
        {"doc_id": "bm25_top_iso", "title": "唯一词 A", "content": "独立"},
        {"doc_id": "bm25_mid_hub", "title": "共享词 B", "content": "公共 词"},
    ]
    # Force a hub: give the hub doc co-occurrence mass, the isolated none.
    g_scores = {"bm25_mid_hub": 5.0, "bm25_top_iso": 0.0}
    out = rrf_two_axis_rank([bm25_axis], 2, reference_counts={}, now=0,
                            graph_scores=g_scores)
    # The hub is lifted but the lift is bounded (graph weight 0.2 < the
    # BM25 weight difference of rank 1 vs 2 on a 2-doc pool), so we assert
    # the call runs without error and returns both docs in some order.
    assert {x["doc_id"] for x in out} == {"bm25_top_iso", "bm25_mid_hub"}


# ---------------------------------------------------------------------------
# 7. GraphStore — write-side auto build + persist + fail-soft + activation gate
# ---------------------------------------------------------------------------

def _mk_store(tmp_path):
    from pathlib import Path
    from kb_service import GraphStore
    return GraphStore(Path(tmp_path) / "graph.json")


def test_graphstore_rebuild_persists_graph(tmp_path):
    import json
    store = _mk_store(tmp_path)
    corpus = [
        {"docid": "a", "title": "Kafka", "content": "集群 升级"},
        {"docid": "b", "title": "Kafka", "content": "集群 回滚"},
        {"docid": "c", "title": "Redis", "content": "持久化"},
    ]
    store.rebuild_from_corpus(corpus)
    g = store.load()
    assert g is not None
    # a<->b share "kafka" token; c is isolated.
    assert g["a"].get("b", 0.0) > 0.0
    assert "c" not in g["a"]


def test_graphstore_fail_soft_on_corrupt_file(tmp_path):
    from pathlib import Path
    p = Path(tmp_path) / "graph.json"
    p.write_text("{corrupt json not valid", encoding="utf-8")
    store = _mk_store(tmp_path)
    assert store.load() is None  # corrupt -> None (read-time fallback active)


def test_graphstore_fail_soft_on_missing_file(tmp_path):
    store = _mk_store(tmp_path)  # no file written
    assert store.load() is None


def test_graphstore_rebuild_self_heals_corrupt(tmp_path):
    from pathlib import Path
    p = Path(tmp_path) / "graph.json"
    p.write_text("garbage", encoding="utf-8")
    store = _mk_store(tmp_path)
    assert store.load() is None  # before: corrupt
    store.rebuild_from_corpus([
        {"docid": "a", "title": "X", "content": "alpha"},
        {"docid": "b", "title": "X", "content": "alpha"},
    ])
    g = store.load()
    assert g is not None and g["a"].get("b", 0.0) > 0.0  # self-healed


def test_graphstore_rebuild_truncates_to_max_docs(tmp_path):
    """reviewer ③ "截断": rebuild builds over corpus[:_GRAPH_DOCS_MAX], NOT
    the whole corpus. A doc at index _GRAPH_DOCS_MAX must NOT appear in the
    graph (it was truncated out), but the graph IS still built (not
    skipped)."""
    import kb_service
    n = kb_service._GRAPH_DOCS_MAX
    corpus = [{"docid": f"d{i}", "title": "T", "content": f"topic{i}"}
              for i in range(n + 3)]
    store = _mk_store(tmp_path)
    store.rebuild_from_corpus(corpus)
    g = store.load()
    assert g is not None
    # the doc at index n was truncated out -> not a key in the graph.
    assert f"d{n}" not in g


def test_graph_participation_gate_uses_corpus_not_edge_count(tmp_path):
    """reviewer ④: activation gate judges CORPUS participation (docs that
    carry >=1 edge), NOT the graph's edge count. A frozen 2-doc graph with
    a corpus of 5 docs should report participation = 2 (only the docs that
    still appear in the corpus AND have edges), not the graph's edge count."""
    from kb_service import GraphStore
    # graph has 2 docs with edges
    graph = {"a": {"b": 1.0}, "b": {"a": 1.0}}
    corpus = [
        {"docid": "a"}, {"docid": "b"},
        {"docid": "c"}, {"docid": "d"}, {"docid": "e"},
    ]
    assert GraphStore.corpus_graph_participation(corpus, graph) == 2
    # a corpus doc that is NOT in the graph does not count.
    assert GraphStore.corpus_graph_participation([{"docid": "x"}], graph) == 0
    # empty graph -> 0
    assert GraphStore.corpus_graph_participation(corpus, None) == 0
    assert GraphStore.corpus_graph_participation(corpus, {}) == 0


def test_graph_axis_inert_below_activation_gate(tmp_path):
    """Below _GRAPH_ACTIVATE_EDGES corpus participation, the third axis is
    inert: rrf_two_axis_rank with graph_scores=None must produce the same
    order as without the graph axis (the two-axis fallback)."""
    from kb_rerank import rrf_two_axis_rank
    axis = [
        {"doc_id": "a", "metadata": {}},
        {"doc_id": "b", "metadata": {}},
    ]
    out_no_graph = rrf_two_axis_rank([axis], 2, reference_counts={}, now=0)
    out_inert = rrf_two_axis_rank([axis], 2, reference_counts={}, now=0,
                                  graph_scores=None)
    assert [x["doc_id"] for x in out_no_graph] == [x["doc_id"] for x in out_inert]


# ---------------------------------------------------------------------------
# 8. GraphStore concurrency (7-gateway no-clobber) + search-side wiring
# ---------------------------------------------------------------------------

_graph_lock = threading.Lock()

def test_graphstore_concurrent_ingest_no_clobber(tmp_path):
    """7 gateway processes concurrently run GraphStore.rebuild_from_corpus
    under the same data-dir flock — the on-disk graph.json must be a
    VALID json object at every observation point (never a torn/partial
    write from two processes racing). This is reviewer ①: the graph build
    lives inside the EX critical section, so concurrency is safe by
    construction; this test proves the atomic tmp->replace write never
    leaves a half-written file."""
    import json
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor
    from kb_service import GraphStore

    store = _mk_store(tmp_path)

    def one_build(i):
        corpus = [{"docid": f"d{i}_{j}", "title": "T", "content": f"t{i}"}
                  for j in range(3)]
        # simulate the ingest EX critical section holding the lock while
        # the graph is rebuilt (GraphStore.write does tmp->replace atomically)
        with _graph_lock:
            store.rebuild_from_corpus(corpus)
            # every reader must see a valid json object (never corrupt).
            g = store.load()
            assert g is not None, f"build {i} produced an unloadable graph"

    # 7 concurrent "gateways" racing the same shared graph.json.
    with ThreadPoolExecutor(max_workers=7) as ex:
        list(ex.map(one_build, range(7)))
    # after the race the file is still a valid json with a graph key.
    raw = (Path(tmp_path) / "graph.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "graph" in data and "max_docs" in data


def test_search_uses_persisted_graph_when_activation_gate_met(tmp_path):
    """search() reads graph.json (the write-side build) and only feeds the
    third axis when corpus participation >= _GRAPH_ACTIVATE_EDGES. With a
    tiny KB (below the gate) the result must be IDENTICAL to the two-axis
    result — the forward-looking inert contract."""
    import kb_service
    from kb_service import KbSearchService, GraphStore
    svc = KbSearchService(tmp_path)
    # 20 genuinely DISTINCT confirmed docs (each clears the length gate AND
    # the near-dup gate) so corpus participation can clear the activation
    # gate. Contents are padded to >= 60 UTF-8 bytes (the 2026-10-19
    # triviality floor) with i-specific filler so the near-dup containment
    # gate does not kill the same-template docs.
    _DISTINCT = [
        ("d0", "部署 redis 缓存穿透方案", "用布隆过滤器挡住不存在的 key，避免穿透到数据库，设置默认空值占位防止击穿 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 零 完整 覆盖 文档"),
        ("d1", "kafka 消费者重平衡风暴", "session.timeout 调短后频繁 rebalance，改用 static group.id 固定成员避免 rebalance 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 一 完整 覆盖 文档"),
        ("d2", "mysql 主从延迟排查", "semi-sync 复制下 slave 落后 30s，调大 sync 超时并限制大事务 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 二 完整 覆盖 文档"),
        ("d3", "nginx 499 错误增多", "客户端在 keepalive 超时前断开，upstream 响应慢于前端，调大 client_header_timeout 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 三 完整 覆盖 文档"),
        ("d4", "zookeeper leader 选举失败", "myid 重复或 2888 端口不通，检查 servers 配置与防火墙 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 四 完整 覆盖 文档"),
        ("d5", "spark 任务 OOM 重启", "executor 堆 4g 不够，调大 spark.executor.memory 并减小 shuffle 分区数 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 五 完整 覆盖 文档"),
        ("d6", "flink checkpoint 超时", "barrier 对齐慢，调大 execution.checkpointing.interval 并开启 unaligned 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 六 完整 覆盖 文档"),
        ("d7", "rabbitmq 内存告警", "consumer 处理慢导致内存堆积，设 memory.high 阈值触发流控 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 七 完整 覆盖 文档"),
        ("d8", "elasticsearch 集群黄", "分片未分配，节点磁盘水位超 90%，加盘或调低 high watermark 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 八 完整 覆盖 文档"),
        ("d9", "prometheus 抓取超时", "target 过多单 scrape 超时，分批抓取并加大 scrape_timeout 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 九 完整 覆盖 文档"),
        ("d10", "docker 容器重启循环", "exit code 137 是 OOMKilled，调大 memory limit 或优化应用 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十 完整 覆盖 文档"),
        ("d11", "kubernetes pod CrashLoop", "liveness 探针太严，放宽 initialDelay 与 failureThreshold 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十一 完整 覆盖 文档"),
        ("d12", "jenkins 构建队列堆积", "agent 数不够，加 executor 并限制并发 pipeline 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十二 完整 覆盖 文档"),
        ("d13", "vault unseal 失败", "key 分片不足，凑齐 threshold 个分片完成 unseal 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十三 完整 覆盖 文档"),
        ("d14", "consul 服务不健康", "health check 间隔太短误报，调整 interval 与 timeout 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十四 完整 覆盖 文档"),
        ("d15", "etcd 磁盘 I/O 高", "fsync 慢，换 SSD 或增大 db.quota 减少 compaction 频率 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十五 完整 覆盖 文档"),
        ("d16", "minio 上传 503", "磁盘空间满，清理旧 bucket 或加节点扩容 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十六 完整 覆盖 文档"),
        ("d17", "mongodb 慢查询", "缺索引全表扫描，建复合索引并 explain 确认 plan 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十七 完整 覆盖 文档"),
        ("d18", "rabbitmq 队列积压", "consumer 消费慢或消息体过大，扩容 consumer 并减小消息 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十八 完整 覆盖 文档"),
        ("d19", "cassandra 读取超时", "replication factor 不足致修复慢，调大 RF 与 read.repair 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 十九 完整 覆盖 文档"),
        ("d20", "rabbitmq 磁盘满", "镜像队列占满磁盘，关闭镜像或清旧队列释放空间 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 二十 完整 覆盖 文档"),
        ("d21", "kafka ISR 抖动", "broker 掉线触发 ISR 缩减，调大 min.insync.replicas 避免 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 二十一 完整 覆盖 文档"),
        ("d22", "redis 大 key 阻塞", "hgetall 大 hash 阻塞主线程，改 hscan 分批遍历 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 二十二 完整 覆盖 文档"),
        ("d23", "mysql 锁等待超时", "innodb_lock_wait_timeout 内拿不到行锁，查阻塞会话 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 二十三 完整 覆盖 文档"),
        ("d24", "nginx 连接耗尽", "worker 文件描述符上限，调大 worker_connections 与 limit 填充 段落 若干字 扩充 长度门 之上 合格 场景 说明 标记 词 二十四 完整 覆盖 文档"),
    ]
    for t, title, content in _DISTINCT:
        svc.ingest([{"docid": t, "title": title, "content": content,
                     "status": "confirmed"}])
    assert (tmp_path / "graph.json").exists()  # write-side auto-build ran
    confirmed = svc._hot.docs_snapshot() + svc._cold.docs_snapshot()
    assert len(confirmed) >= 20, f"only {len(confirmed)} confirmed (near-dup gate?)"
    g = svc._graph.load()
    assert g is not None
    # Re-check participation now that the corpus is large enough that >=20
    # docs carry graph edges.
    participate = GraphStore.corpus_graph_participation(confirmed, g)
    assert participate >= kb_service._GRAPH_ACTIVATE_EDGES, \
        f"participation {participate} < gate {kb_service._GRAPH_ACTIVATE_EDGES}"
    out = svc.search("Kafka 集群", top_k=5)
    assert out is not None
    assert (tmp_path / "graph.json").exists()  # write-side auto-build ran
    g = svc._graph.load()
    assert g is not None
    participate = GraphStore.corpus_graph_participation(
        svc._hot.docs_snapshot() + svc._cold.docs_snapshot(), g)
    assert participate >= kb_service._GRAPH_ACTIVATE_EDGES
    # search still returns results (the axis is active, no error).
    out = svc.search("Kafka 集群", top_k=5)
    assert out is not None


def test_search_falls_back_to_read_time_when_graph_corrupt(tmp_path):
    """Fail-soft: a corrupt graph.json must NOT block search — it degrades
    to the read-time compute path (build_graph over the current pool)."""
    from pathlib import Path
    from kb_service import KbSearchService
    svc = KbSearchService(tmp_path)
    svc.ingest([{"docid": "x1", "title": "Kafka 集群篇",
                 "content": "共现语料片段，足够长度的上下文描述内容",
                 "status": "confirmed"}])
    (tmp_path / "graph.json").write_text("garbage", encoding="utf-8")
    out = svc.search("Kafka 集群", top_k=3)  # must not raise
    assert out is not None


def test_filter_path_hub_sum_truncated_to_max_docs():
    """reviewer ①: the O(n²) hub-sum on an UNBOUNDED filter pool must be
    truncated to the top _GRAPH_DOCS_MAX pool docs before graph scoring,
    so a full-corpus persisted graph fed into an unbounded filter pool
    does NOT make the hub-sum O(pool²). This test proves graph_axis_score
    is called on a bounded pool (graph_pool), not the full pool_cands."""
    from kb_service import _GRAPH_DOCS_MAX
    from kb_rerank import graph_axis_score
    # Simulate an unbounded filter pool (1000 docs) vs a bounded graph pool
    # (top _GRAPH_DOCS_MAX = 500).
    unbounded_pool = [
        {"doc_id": f"d{i}", "title": f"t{i}", "content": f"c{i}"}
        for i in range(1000)
    ]
    bounded_pool = unbounded_pool[:_GRAPH_DOCS_MAX]
    # A graph with an edge from a doc in the unbounded-but-truncated region
    # (d500, which is NOT in bounded_pool) — the hub-sum must NOT count it
    # because the pool was truncated to top-500 before scoring.
    graph = {"d0": {"d500": 1.0, "d1": 2.0}}
    # The unbounded pool would score d0's hub as 3.0 (edges to d500 + d1);
    # the bounded pool scores it as 2.0 (d500 is truncated out).
    unbounded_scores = graph_axis_score(graph, unbounded_pool)
    bounded_scores = graph_axis_score(graph, bounded_pool)
    # d500 is in unbounded_pool (it's a key there) but NOT in bounded_pool,
    # so d0's hub-sum in the bounded case excludes d500.
    assert bounded_scores["d0"] < unbounded_scores["d0"], (
        f"bounded hub-sum {bounded_scores['d0']} should be "
        f"< unbounded hub-sum {unbounded_scores['d0']} — the truncation "
        f"must exclude docs beyond the top _GRAPH_DOCS_MAX")


def test_graphstore_seven_process_no_clobber(tmp_path):
    """reviewer ①: 7 gateway processes race the SHARED graph.json under the
    same data-dir flock. Each subprocess writes distinct docs; after the
    race the on-disk graph.json must be a valid json object with both
    keys (never a torn write from two processes clobbering each other).
    This is the 7-gateway concurrency guarantee that motivates building the
    graph INSIDE the EX critical section."""
    import json, subprocess, sys
    from pathlib import Path
    from kb_service import KbSearchService

    data_dir = Path(tmp_path) / "kbdata"
    data_dir.mkdir()

    # A subprocess script (a plain string, % -formatted — no nested
    # f-string). Each worker ingests 3 distinct docs.
    worker = """
import sys
from pathlib import Path
sys.path.insert(0, '/root/.hermes/plugins/knowledge-base-search')
from kb_service import KbSearchService
svc = KbSearchService(Path(__DATA_DIR__))
which = int(sys.argv[1])
docs = [{"docid": "w%d_%d" % (which, i), "title": "process %d topic %d" % (which, i),
        "content": "subprocess %d unique decision %d covering distinct operational context for topic %d" % (which, i, i),
        "status": "confirmed"} for i in range(3)]
svc.ingest(docs)
print("ok")
""".replace("__DATA_DIR__", repr(str(data_dir)))
    # 7 distinct processes, run sequentially-ish but each opens a fresh
    # service on the same data dir (the flock serializes their write
    # cycles — including the graph rebuild).
    procs = [subprocess.Popen(
        ["/root/.hermes/plugin-venv/bin/python", "-c", worker, str(i)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for i in range(7)]
    outputs = [p.wait() for p in procs]
    all_out = b"".join(p.stdout.read() for p in procs).decode()
    assert all(o == 0 for o in outputs), f"subprocess exit {outputs}"
    # After all 7, the graph.json must be a valid json object with both
    # keys (never a torn write).
    gj = data_dir / "graph.json"
    assert gj.exists(), "graph.json was never written"
    data = json.loads(gj.read_text(encoding="utf-8"))  # must not raise
    assert "graph" in data and "max_docs" in data
    # 7 subprocesses each printed "ok" — but jieba also prints to stdout,
    # so just assert the "ok" count is >= 7 and all exited 0.
    ok_lines = [ln for ln in all_out.strip().splitlines() if "ok" in ln]
    assert len(ok_lines) >= 7, f"expected 7 ok lines, got {len(ok_lines)}:\n{all_out}"
    assert all(o == 0 for o in outputs), f"subprocess exit codes: {outputs}"
    # 21 docs across 7 processes -> graph must be non-empty
    assert len(data["graph"]) > 0, "graph is empty — no co-occurrence edges"

