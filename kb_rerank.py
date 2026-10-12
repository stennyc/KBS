"""Read-side precision upgrade (owner "向 filemem 借鉴" batch ②③).

Two pure-code (ZERO LLM, ZERO host-API) mechanisms that lift retrieval
precision on top of the existing two-layer BM25, mirroring filemem's
multi-axis search but adapted to a knowledge-doc corpus:

  1. denoise_query — strip host-path / high-frequency noise tokens from a
     user query BEFORE it reaches BM25. A query that pasted a file path
     ("/root/.hermes/.../foo.py") makes every doc that mentions "/root" or
     ".hermes" co-occur; those tokens are noise, not signal. Pure text
     processing, no LLM. (filemem's _normalize_search_text, same idea.)

  2. rrf_two_axis_rank — fuse TWO independent ranking axes by reciprocal
     rank fusion (RRF, K=60, the filemem constant):
       axis A: BM25 relevance rank (the layer's native per-doc score)
       axis B: a freshness/recency axis seeded by the reinforce write-gate
               (kb_service.ReinforceStore) + the doc's ingest timestamp.
     RRF is rank-based (not score-based), so the two axes need no shared
     scale and a recency-only hit can never drown a real BM25 hit — each
     axis contributes 1/(K + rank) and the sums decide order. This is the
     "时间轴初始值用 reinforce 计数打底" fix the reviewer demanded: a doc
     that has been genuinely cited (high reference_count) keeps its
     recency rank high even if it is old, so a fresh-but-never-cited doc
     does NOT auto-monopolise the top slot.

Both are pure stdlib. No model, no external API, no host import (PRD §1).
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional

# RRF constant (filemem's SA2-3 value; rank-based fusion, no config surface —
# YAGNI). K large enough that top-rank hits dominate but rank differences
# still matter: contribution of rank r is 1/(K + r).
_RRF_K = 60

# Recency axis: exp(-lambda_days * age_days). default half-life ~14 days.
_DEFAULT_RECENCY_LAMBDA = math.log(2.0) / 14.0

# Recency salience boost (multiplicative, the filemem reference_count>=3 idea,
# generalised + capped): salience = 1.0 + SALIENCE_PER_REF * min(ref_count,
# SALIENCE_MAX_REFS). A citation slows a doc's recency decay (effective
# half-life stretches by the salience factor) so a frequently-cited OLD doc
# stays afloat; the cap keeps one citation from ever overuling recency.
_SALIENCE_PER_REF = 0.2
_SALIENCE_MAX_REFS = 10

# Time-axis weight (reviewer card-2 fix): the recency/occurred_at axis must
# NOT auto-dominate when the candidate pool is small (hot-only, <=5 docs) —
# with only a handful of docs the BM25 keyword axis has already been
# min-max-flattened to full [0,1] range (no lower-bound exit), so a
# recency-weighted top slot would just become a "newest doc wins" monotone
# degeneracy. Pool-size-adaptive: the time axis gets a HIGHER weight only
# when the pool is large enough that the BM25 axis has real resolution.
# Not a hardcoded magic constant at call sites — it is derived from the
# candidate pool size the caller already knows, so a test asserts the
# DOWNSHIFT BEHAVES (weight drops when pool <= _SMALL_POOL), not a value.
_SMALL_POOL_THRESHOLD = 5
_TIME_AXIS_WEIGHT_LARGE = 0.5
_TIME_AXIS_WEIGHT_SMALL = 0.3
# Third-axis (build_graph co-occurrence) weight, active ONLY when
# graph_scores is provided and non-empty; the weight is rebalanced off the
# BM25 axis so w_bm25 + w_time + w_graph == 1.0 always.
_GRAPH_AXIS_WEIGHT = 0.2

# Fourth-axis (TR-6 topic-mass) weight: the SMALLEST of the four axes so a
# topic-mass signal can lift, never overrule, the keyword + recency + graph
# signal (R-2 weight-balance guard; the activation gate below is the second
# of the triple hedge).
_TOPIC_AXIS_WEIGHT = 0.1
# Fourth-axis activation gate: the absolute-count criterion (PRD L42, v2),
# isomorphic to the graph axis's _GRAPH_ACTIVATE_EDGES gate — the fourth
# axis gets weight only when the candidate pool carries >= this many
# origin_kind=agent entries AND the caller actually supplied topic_scores.
# The v1 "agent share >= 50%" ratio criterion is retired: a tiny pool with
# a high share must NOT activate an axis built for a mature ledger.
_TOPIC_ACTIVATE_AGENTS = 20


def time_axis_weight(pool_size: int) -> float:
    """The recency/occurred_at axis weight for a candidate pool of
    ``pool_size`` docs. Small pools (hot-only, <= _SMALL_POOL_THRESHOLD)
    get the reduced weight so a fresh doc can't auto-monopolise the top
    slot on top of an already-flattened BM25 axis (reviewer card-2
    acceptance). Large pools keep the default weight. Pure function — no
    state, no LLM, testable by asserting the downshift, not the literal."""
    return (_TIME_AXIS_WEIGHT_SMALL
            if pool_size <= _SMALL_POOL_THRESHOLD
            else _TIME_AXIS_WEIGHT_LARGE)


# ---------------------------------------------------------------------------
# 3. Event-time extraction (occurred_at / version_ts split, card 2)
# ---------------------------------------------------------------------------
# Three KB doc categories (owner ruling): decision/event docs carry a
# point-in-time ("2024-03-15 拍板 Kafka 升级"); mechanism docs have NO event
# time ("Redis 持久化机制" -> null, never penalised on the time axis); config
# docs carry a VERSION timestamp ("kafka_3.6.1" -> version_ts, used only as a
# FILTER tag, NOT fed to the time axis). The split prevents a config version
# stamp from polluting the event-time axis (a mechanism vs. config-version
# are different dimensions — mixing them into one axis would let "kafka_3.6.1"
# decay like "3 周前的决策").

# ISO-ish / CN date: 2024-03-15, 2024/03/15, 2024.3.15, 2024年3月15日.
_OCCURRED_DATE_RE = re.compile(
    r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*日?",
)
# A bare 4-digit year (decision docs often just say "2024 拍板").
_OCCURRED_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
# Version stamp: kafka_3.6.1 / v3.6.1 / 3.6.1 (dotted numeric, >=2 parts).
# Greedy on the dotted run so "kafka_3.6.1" -> "3.6.1", not "6.1".
_VERSION_TS_RE = re.compile(
    r"(?:v|_)(\d+(?:\.\d+){1,3})\b",
)


def _month_day_safe(month: int, day: int, year: int) -> tuple:
    """Clamp a parsed (y,m,d) to a valid calendar date (regex can over-capture
    e.g. 2024-13-45 from noise); returns (year, month, day) with the
    out-of-range parts clamped to a safe in-month value so a malformed stamp
    still yields a parseable date instead of a crash."""
    import calendar
    month = max(1, min(12, month))
    day = max(1, min(calendar.monthrange(year, month)[1], day))
    return year, month, day


def extract_event_time(text: str) -> Dict[str, Optional[str]]:
    """Extract the doc's time fields from raw text (pure regex, zero LLM).

    Returns {"occurred_at": ISO-date-or-None, "version_ts": dotted-version-
    or-None}. `occurred_at` is the event/decision point-in-time (category 1);
    a doc with no date match (mechanism, category 2) gets occurred_at=None
    and is NEVER penalised on the time axis (the owner ruling: a missing
    event time is "no credit", not "very old"). `version_ts` is a config
    version stamp (category 3) — it is a FILTER TAG, not fed to the time
    axis. A doc can carry both (a config doc that states when it was
    adopted); they are independent fields, not mutually exclusive. When
    both are present the time axis uses `occurred_at` (event time takes
    precedence over version time — a config's adoption date is the real
    recency signal, the version number is just a tag)."""
    if not text:
        return {"occurred_at": None, "version_ts": None}
    occurred: Optional[str] = None
    m = _OCCURRED_DATE_RE.search(text)
    if m:
        y, mo, d = _month_day_safe(int(m.group(2)), int(m.group(3)), int(m.group(1)))
        occurred = f"{y:04d}-{mo:02d}-{d:02d}"
    else:
        ym = _OCCURRED_YEAR_RE.search(text)
        if ym:
            occurred = f"{ym.group(1)}-01-01"
    version: Optional[str] = None
    vm = _VERSION_TS_RE.search(text)
    if vm:
        version = vm.group(1)
    return {"occurred_at": occurred, "version_ts": version}


# ---------------------------------------------------------------------------
# 1. Query denoising (pure text, no LLM)
# ---------------------------------------------------------------------------

# Host-path / filesystem noise tokens that co-occur with nearly every doc and
# carry no retrieval signal. Matched against the QUERY only (docs are not
# touched). Anchored so a real identifier "foo" is kept when embedded in a
# path token: we drop the WHOLE path-token, not the inner word.
_NOISE_TOKEN_RE = re.compile(
    r"""
    (?:^|\s)
    (
        /\S+                          # absolute path  (/root/.hermes/...)
      | \.\.?\S+                      # relative path (../foo, ./bar)
      | ~\S+                          # home-anchored  (~/.hermes)
      | \b(?:\d{1,3}\.){3}\d{1,3}\S*  # IP-ish tokens (usually host noise)
      | \b[A-Z]:\\[^\s]+              # windows paths
      | \b(?:tmp|tmpdir|scratch|__pycache__)\\?/?\S*   # ephemeral dir names
    )
    """,
    re.VERBOSE,
)

# Standalone common filesystem / host-dir names that appear as bare query
# words when a user pastes context. Lowercased membership (token-level, not
# substring) so a legitimate "docker" topic word is NOT dropped — only the
# exact noisy container names listed here are removed when they stand alone.
_NOISE_DIR_NAMES = frozenset({
    "root", "home", "var", "etc", "usr", "bin", "sbin", "opt",
    "dev", "proc", "sys", "boot", "lib", "lib64", "run", "srv",
    "tmpdir", "scratch", "cache",
})


def denoise_query(query: str) -> str:
    """Strip host-path noise tokens from a user query.

    Returns the cleaned query string. On an empty/noisy-only query it returns
    the ORIGINAL (never returns an empty string — an empty query would
    trivially 0-hit and defeat the fetch; if denoising removes everything,
    the caller falls back to the raw query so a real retrieval still happens).
    """
    if not query:
        return query
    # 1) drop whole path tokens (regex keeps the leading whitespace so the
    #    join stays clean).
    cleaned = _NOISE_TOKEN_RE.sub(" ", query)
    # 2) drop standalone noisy dir-name words (token-level, case-insensitive).
    tokens = [
        t for t in cleaned.split(" ")
        if t.strip() and t.strip().lower() not in _NOISE_DIR_NAMES
    ]
    out = " ".join(t for t in tokens if t).strip()
    return out or query  # never empty -> fall back to the original


# ---------------------------------------------------------------------------
# 2. RRF two-axis fusion (BM25 relevance + recency/reinforce)
# ---------------------------------------------------------------------------

def _age_days_from_ts(ts: Any, now: float) -> float:
    """Age in days for a timestamp that may be an ISO date string ("2024-03-15")
    or a unix epoch float. A malformed / missing value yields 0.0 (no recency
    credit, never a penalty — a null occurred_at must not bury a doc, owner
    card-2 ruling)."""
    try:
        if isinstance(ts, str):
            import datetime as _dt
            # ISO date (occurred_at) or ISO datetime; epoch-string fallback.
            s = ts.strip()
            if "T" in s or " " in s or len(s) > 10:
                # datetime string
                naive = s.replace("Z", "+00:00")
                try:
                    return max(0.0, (now - _dt.datetime.fromisoformat(naive).timestamp()) / 86400.0)
                except ValueError:
                    return max(0.0, (now - float(s)) / 86400.0)
            # pure date "YYYY-MM-DD"
            return max(0.0, (now - _dt.datetime.strptime(s, "%Y-%m-%d").timestamp()) / 86400.0)
        return max(0.0, (now - float(ts)) / 86400.0)
    except (TypeError, ValueError):
        return 0.0


def _recency_axis(rank_items: List[Dict[str, Any]],
                  reference_counts: Dict[str, int],
                  now: Optional[float] = None,
                  lambda_days: float = _DEFAULT_RECENCY_LAMBDA) -> List[int]:
    """The recency axis, seeded by the reinforce write-gate.

    For each candidate doc, freshness =
        reference_count-salience  *  exp(-lambda_days * age_days)

    - ``age_days`` = days since the doc's ``created_at`` (or, for legacy docs
      without a timestamp, since now -> age 0 -> full weight; a missing
      timestamp must NOT punish a doc, it just gets no recency credit).
    - ``reference_count`` comes from the reinforce store: a doc the model has
      explicitly cited gets salience >= threshold and its decay slows
      (the reviewer fix: cited OLD docs stay afloat, uncited NEW docs do not
      auto-dominate).

    Returns a 1-based deterministic total order (list of candidate indices
    sorted by (-score, index)) — the shape RRF expects. A doc with NO
    timestamp AND NO reference count scores 0.0 on this axis and simply
    ranks last on the recency term (RRF then decides by the BM25 term).
    """
    now = now if now is not None else time.time()
    scored: List[float] = []
    for cand in rank_items:
        meta = cand.get("metadata") or {}
        # Event time takes precedence (card 2): an explicit occurred_at (a
        # decision/event point-in-time, or a config's adoption date) is the
        # true recency signal; fall back to the deterministic ingest_ts
        # sidecar for docs with no extracted event time. version_ts is
        # deliberately NOT consulted here (a version stamp is a filter tag,
        # not a recency source).
        ts = (meta.get("occurred_at")
              or meta.get("ingest_ts")
              or meta.get("created_at")
              or meta.get("ingested_at"))
        age_days = 0.0
        if ts:
            age_days = _age_days_from_ts(ts, now)
        ref_count = int(reference_counts.get(cand.get("doc_id", ""), 0))
        # Multiplicative salience: the reference-count boost MULTIPLIES the
        # recency score. A cited doc decays slower (its effective half-life
        # stretches by the boost factor), so a frequently-cited OLD doc stays
        # afloat relative to a never-cited one — but the boost is capped so a
        # single citation can NEVER fully outrank a fresh doc that is many
        # days newer (the reviewer's guard: a citation lifts, it does not
        # overrule recency).
        salience = 1.0 + _SALIENCE_PER_REF * min(ref_count, _SALIENCE_MAX_REFS)
        score = salience * math.exp(-lambda_days * age_days)
        scored.append(score)
    order = sorted(range(len(scored)), key=lambda i: (-scored[i], i))
    return order


def rrf_two_axis_rank(
    ranked_lists: List[List[Dict[str, Any]]],
    top_k: Optional[int],
    reference_counts: Optional[Dict[str, int]] = None,
    now: Optional[float] = None,
    graph_scores: Optional[Dict[str, float]] = None,
    topic_scores: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Fuse multiple per-axis ranked lists into one RRF-ordered result set.

    ``ranked_lists[i]`` is axis i's candidate list, BEST FIRST (each list
    already deduped by the two-layer merge). Every doc across all axes is
    pooled; for each axis a doc's rank (1-based, position+1) contributes
    ``1/(K + rank)``; docs absent from an axis contribute 0 for that axis.
    The final order is by total fused score (ties -> earlier pooled position
    -> doc_id, for determinism).

    The recency axis is DERIVED here from the pooled docs + the reinforce
    store, so the caller passes only the BM25-ordered axis(es) plus
    ``reference_counts``; the fresh-vs-cited ordering is computed, not
    given.

    Optional THIRD axis (build_graph, forward-looking): ``graph_scores`` is
    a {doc_id: co-occurrence-mass} map (from kb_rerank.graph_axis_score).
    When provided AND non-empty, a third RRF term is fused with
    _GRAPH_AXIS_WEIGHT (rebalanced off the BM25 axis, so the three
    weights still sum to 1.0). When OMITTED or the map is empty (a
    <2-doc corpus - the current dev state), the fusion is IDENTICAL
    to the two-axis result: the third axis contributes 0 weight, so a
    tiny KB degrades exactly to today's behaviour and the axis
    auto-activates with no code change once the corpus grows.

    Optional FOURTH axis (TR-6 topic-mass, the agent experience-ledger
    task-topic signal): ``topic_scores`` is a {doc_id: topic-mass} map
    the CALLER supplies (only an explicit tool-layer
    knowledge_base_search call computes it — the pre_llm_call
    auto-inject layer never does, mirroring the ReinforceStore
    write-gate's "only explicit tool calls count" rule). The axis is
    ACTIVE only when BOTH hold (the activation gate is isomorphic to
    the graph axis's corpus-participation gate, but judged on the
    candidate POOL instead of the corpus — the topic-mass signal is
    pool-scoped):
      1. the caller passed a non-empty ``topic_scores`` map, AND
      2. the pooled candidates carry >= _TOPIC_ACTIVATE_AGENTS
         origin_kind=agent entries (the ABSOLUTE-count criterion;
         the v1 "agent share >= 50%" ratio gate is retired).
    When active, the fourth RRF term fuses with _TOPIC_AXIS_WEIGHT
    (rebalanced off the BM25 axis, so all four weights still sum to
    1.0). When inert (topic_scores omitted/empty, or a pool with
    <20 agent entries), the fused scores are BYTE-IDENTICAL to the
    three-axis result (KPI-6 inert acceptance gate, R-2) — the axis
    contributes 0 weight and the ordering / rrf_fused_score payload
    changes not at all. The fourth axis NEVER participates in the
    tie-break: the tie-breaker stays the pool-size-adaptive
    bm25/recency stage + pool position; a fourth-axis rank is never
    consulted when fused scores tie.
    """
    reference_counts = reference_counts or {}
    # Pool every candidate once, in first-seen order (stable, deterministic).
    pool: Dict[str, Dict[str, Any]] = {}
    pool_order: List[str] = []
    for axis in ranked_lists:
        for cand in axis:
            key = str(cand.get("doc_id", ""))
            if key not in pool:
                pool[key] = cand
                pool_order.append(key)
    if not pool_order:
        return []

    # BM25 axis ranks: each ranked_list gives a 1-based rank for the docs it
    # returned (they are pre-sorted best-first).
    bm25_rank: Dict[str, int] = {}
    for axis in ranked_lists:
        for pos, cand in enumerate(axis, start=1):
            key = str(cand.get("doc_id", ""))
            # keep the BEST (lowest) rank a doc got across the BM25 axes.
            if key not in bm25_rank or pos < bm25_rank[key]:
                bm25_rank[key] = pos

    # Recency axis over the POOL (1-based order returned as a rank map).
    pool_items = [pool[k] for k in pool_order]
    recency_order = _recency_axis(pool_items, reference_counts, now=now)
    recency_rank: Dict[str, int] = {}
    for pos, idx in enumerate(recency_order, start=1):
        recency_rank[pool_order[idx]] = pos

    # Pool-size-adaptive time-axis weight (card 2): a small hot-only pool has
    # an already-flattened BM25 axis, so the recency/occurred_at axis gets a
    # reduced weight there; large pools keep the default. Derived from the
    # pool size (the caller's actual candidate count), not hardcoded at the
    # call site. The two axis weights sum to 1.0 (keeps RRF bounded).
    w_time = time_axis_weight(len(pool_order))
    w_bm25 = 1.0 - w_time

    # Third axis (build_graph co-occurrence), forward-looking. INERT when
    # graph_scores is omitted/empty (the <2-doc dev state): the two-axis
    # weights are used as-is so a tiny KB degrades exactly to today's
    # behaviour. When a non-empty graph_scores map is supplied, the graph
    # axis contributes _GRAPH_AXIS_WEIGHT and the BM25 weight is reduced so
    # the three axis weights sum to 1.0 (the recency axis keeps its
    # pool-size-adaptive weight; the graph axis is the additive "semantic
    # neighbourhood" signal the other two can't see).
    has_graph = bool(graph_scores and any(v for v in graph_scores.values()))
    if has_graph:
        w_graph = _GRAPH_AXIS_WEIGHT
        w_bm25 = max(0.0, w_bm25 - w_graph)
        # 1-based rank over the pool by descending co-occurrence mass
        # (additive-credit like the recency axis: no penalty for a doc
        # that has no graph edges, it just ranks last on this axis).
        graph_rank: Dict[str, int] = {}
        by_mass = sorted(pool_order, key=lambda k: -graph_scores.get(k, 0.0))
        for pos, k in enumerate(by_mass, start=1):
            graph_rank[k] = pos
    else:
        w_graph = 0.0
        graph_rank = {}

    # Fourth axis (TR-6 topic-mass), forward-looking for the agent
    # experience ledger. INERT unless BOTH the caller supplied a
    # non-empty topic_scores map AND the candidate pool carries >=
    # _TOPIC_ACTIVATE_AGENTS origin_kind=agent entries (the
    # absolute-count criterion, isomorphic to the graph axis's
    # activation gate — same "weight only when the signal is mature"
    # shape, judged here on the POOL the fusion actually sees). Below
    # the gate the axis contributes 0 weight, so the fused scores —
    # and the rrf_fused_score payload — are byte-identical to the
    # three-axis result (KPI-6 inert gate: a ledger that has not yet
    # accumulated 20 agent entries must not be reordered by a topic
    # signal that is too thin to be meaningful).
    agent_pool_count = sum(
        1 for k in pool_order
        if ((pool[k].get("metadata") or {}).get("origin_kind") == "agent"))
    has_topic = bool(topic_scores
                     and any(v for v in topic_scores.values())
                     and agent_pool_count >= _TOPIC_ACTIVATE_AGENTS)
    if has_topic:
        w_topic = _TOPIC_AXIS_WEIGHT
        w_bm25 = max(0.0, w_bm25 - w_topic)
        # 1-based rank over the pool by descending topic mass
        # (additive-credit like the graph/recency axes: a doc with
        # no topic mass scores 0.0 and simply ranks last on this
        # axis, never penalised).
        topic_rank: Dict[str, int] = {}
        by_topic = sorted(pool_order,
                          key=lambda k: -float(topic_scores.get(k, 0.0)))
        for pos, k in enumerate(by_topic, start=1):
            topic_rank[k] = pos
    else:
        w_topic = 0.0
        topic_rank = {}

    fused: Dict[str, float] = {}
    for key in pool_order:
        s = 0.0
        if key in bm25_rank:
            s += w_bm25 / (_RRF_K + bm25_rank[key])
        if key in recency_rank:
            s += w_time / (_RRF_K + recency_rank[key])
        if has_graph and key in graph_rank:
            s += w_graph / (_RRF_K + graph_rank[key])
        if has_topic and key in topic_rank:
            s += w_topic / (_RRF_K + topic_rank[key])
        fused[key] = s

    # RRF ties (two docs each #1 on a different axis) are broken by a
    # POOL-SIZE-ADAPTIVE tie-breaker (owner ruling, card 2):
    #   - SMALL pool (<= _SMALL_POOL_THRESHOLD): tie-breaker = bm25_rank
    #     (keyword relevance wins the tie). On a hot-only small pool the
    #     BM25 axis has already been min-max-flattened to full range, so
    #     letting the recency/trust axis win the tie would reopen the
    #     "newest doc auto-dominates" back-door that the pool-size
    #     downshift (time_axis_weight) just closed. BM25-first keeps the
    #     small-pool behaviour coherent with "small pool defends against
    #     recency dominance".
    #   - LARGE pool (> threshold): tie-breaker = recency_rank (a cited /
    #     fresher doc wins the tie). On a large pool the BM25 axis has real
    #     resolution, so the trust/recency tie-break adds signal instead of
    #     bias (the owner's "a citation lifts, it does not overrule recency"
    #     ruling, scoped to where it's meaningful).
    # The pool position is the final deterministic tie-breaker in BOTH
    # branches (two docs at the same fused score, same axis rank — cannot
    # happen for distinct pool positions, but the key is total).
    pool_size = len(pool_order)
    tie_second = (bm25_rank.get
                  if pool_size <= _SMALL_POOL_THRESHOLD
                  else recency_rank.get)
    ordered = sorted(
        pool_order,
        key=lambda k: (
            -fused[k],
            tie_second(k, pool_size + 1),
            pool_order.index(k),
        ),
    )
    # top_k=None means "unbounded" (the filter path fuses the full pool).
    take = ordered if top_k is None else ordered[:top_k]
    top: List[Dict[str, Any]] = []
    for k in take:
        cand = dict(pool[k])
        cand["metadata"] = {**(cand.get("metadata") or {}),
                            "rrf_fused_score": round(fused[k], 6)}
        top.append(cand)
    return top


__all__ = ["denoise_query", "rrf_two_axis_rank",
           "build_graph", "graph_axis_score", "time_axis_weight",
           "extract_event_time", "RRF_K", "SALIENCE_PER_REF",
           "SALIENCE_MAX_REFS", "SMALL_POOL_THRESHOLD",
           "GRAPH_AXIS_WEIGHT", "TOPIC_AXIS_WEIGHT",
           "TOPIC_ACTIVATE_AGENTS"]
RRF_K = _RRF_K
SALIENCE_PER_REF = _SALIENCE_PER_REF
SALIENCE_MAX_REFS = _SALIENCE_MAX_REFS
SMALL_POOL_THRESHOLD = _SMALL_POOL_THRESHOLD
GRAPH_AXIS_WEIGHT = _GRAPH_AXIS_WEIGHT
TOPIC_AXIS_WEIGHT = _TOPIC_AXIS_WEIGHT
TOPIC_ACTIVATE_AGENTS = _TOPIC_ACTIVATE_AGENTS


# ---------------------------------------------------------------------------
# 4. build_graph — the RRF third axis (co-occurrence, forward-looking)
# ---------------------------------------------------------------------------
# The owner ruling: "开发就是面向未来" — do NOT hold this axis hostage to a
# data gate ("wait until confirmed docs >= 20 + a measured miss case"). Build
# it now, designed to be INERT on a tiny corpus (an empty or single-doc graph
# contributes 0 weight, so search degrades exactly to today's two-axis
# result) and to AUTO-ACTIVATE as the corpus accumulates, with zero code
# change at the threshold. The axis answers: "which docs in the current
# candidate pool co-occur most with the others?" — a doc that shares
# anchor_points / content tokens with many of the pool's docs is a hub, and
# hubs surface ahead of isolated docs. That is the retrieval signal the two
# axes (keyword + recency) can't see: semantic neighbourhood in the corpus.
#
# build_graph is PURE (no LLM, no I/O) so it is fully testable now on a
# synthetic corpus and costs nothing to run per-query on a 2-doc KB.

# Anchor weight > content-token weight: an explicit anchor_point match is a
# stronger co-occurrence signal than a shared content token (the owner's
# distilled anchors are the high-signal join, content is the weak fallback).
_ANCHOR_CO_WEIGHT = 2.0
_TOKEN_CO_WEIGHT = 1.0
_MIN_GRAPH_DOCS = 2  # below this the graph is vacuous -> weight 0 (inert)


def _anchor_set(doc: Dict[str, Any]) -> set:
    """A doc's explicit anchor_points (the strong co-occurrence join)."""
    anchors = (doc.get("anchors") or doc.get("anchor_points")
               or (doc.get("metadata") or {}).get("anchors")
               or (doc.get("metadata") or {}).get("anchor_points") or [])
    if isinstance(anchors, str):
        anchors = [anchors]
    return {str(a).lower() for a in anchors if a}


def build_graph(docs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Build a symmetric co-occurrence graph over ``docs``.

    Returns ``{doc_id: {other_doc_id: co_score}}``. ``co_score`` is the
    weighted overlap of two docs' signal sets: shared ANCHOR points count
    ``_ANCHOR_CO_WEIGHT`` each, shared content/title TOKENS count
    ``_TOKEN_CO_WEIGHT`` each (anchors are the high-signal join, content is
    the weak fallback). The graph is symmetric (a<b and b<a carry the same
    score) and the diagonal is omitted (a doc does not co-occur with
    itself). Zero-overlap pairs are absent (the dict stays dense).

    PURE + deterministic: no LLM, no I/O, no ordering dependence beyond the
    input list. On < _MIN_GRAPH_DOCS docs the graph is vacuous (``{}``) -
    the caller treats that as "no graph signal" and the third axis
    contributes 0 weight, so search degrades exactly to the two-axis result
    (the forward-looking design: build it now, it activates as the corpus
    grows with zero code change at the threshold).

    Token co-occurrence reuses hot_layer.tokenize (the same jieba +
    number-token pipeline the BM25 axis uses), so the graph and the keyword
    axis share one tokenisation - no divergent token sets."""
    if len(docs) < _MIN_GRAPH_DOCS:
        return {}
    from hot_layer import tokenize
    # Per-doc signal: (doc_id, anchor_set, token_set).
    def _did(d, i):
        # The service corpus keys docs by ``docid`` (kb_service._normalize_docs)
        # but the search-side cand dicts key by ``doc_id``. Read whichever
        # is present (docid first, the persisted form), fall back to the
        # list index so an empty-key doc still gets a stable identity.
        return str(d.get("docid") or d.get("doc_id") or i)
    sigs = [(_did(d, i), _anchor_set(d),
             set(tokenize(f"{d.get('title','')} {d.get('content','')}")))
            for i, d in enumerate(docs)]
    graph: Dict[str, Dict[str, float]] = {did: {} for did, _, _ in sigs}
    for i in range(len(sigs)):
        did_a, anchors_a, tokens_a = sigs[i]
        for j in range(i + 1, len(sigs)):
            did_b, anchors_b, tokens_b = sigs[j]
            anchor_ov = len(anchors_a & anchors_b)
            token_ov = len(tokens_a & tokens_b)
            if not (anchor_ov or token_ov):
                continue
            score = (anchor_ov * _ANCHOR_CO_WEIGHT
                     + token_ov * _TOKEN_CO_WEIGHT)
            graph[did_a][did_b] = score
            graph[did_b][did_a] = score
    return graph


def graph_axis_score(graph: Dict[str, Dict[str, float]],
                     pool: List[Dict[str, Any]]) -> Dict[str, float]:
    """The third-axis score for each POOL doc: the total co-occurrence mass
    it carries to the OTHER pool docs (its hub-ness within the candidate
    pool, not the whole corpus). A doc that co-occurs with many of the
    pool's other docs ranks ahead of an isolated one; a doc with no graph
    edges (or an empty graph) scores 0.0 - no credit, never a penalty. The
    axis is additive-credit like the recency axis, so an empty graph leaves
    the two-axis order intact (the fail-soft the owner wants: build it now,
    it is inert on a 2-doc KB and activates as docs accumulate).
    """
    def _did(c, i):
        return str(c.get("docid") or c.get("doc_id") or i)
    pool_ids = {_did(c, i) for i, c in enumerate(pool)}
    out: Dict[str, float] = {}
    for i, c in enumerate(pool):
        did = _did(c, i)
        adj = graph.get(did, {})
        out[did] = sum(s for n, s in adj.items()
                       if n in pool_ids and n != did)
    return out
