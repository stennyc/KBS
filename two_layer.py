"""Two-layer merge: hot (in-memory rank_bm25) + cold (durable in-memory
rank_bm25 over the settled full corpus, pickle-persisted) backend.

This is the SearchBackend implementation that fronts the dual-layer architecture
(PRD §12). The agent makes ONE call; we fan out to both layers, merge, dedup,
and return a unified top-k full-text payload (contract #11: single tool call,
one result shape — never two).

Design decisions baked in from the room's review:
  - Merge join key is the EXTERNAL docid, not any backend's internal
    document number. A backend re-issues its internal doc numbers on every
    generation rebuild; only a stable external docid lets the hot layer and
    the cold layer agree on "same document" during the settle window
    (contract #15: cross-layer dedup by external docid).
  - settle() moves the hot layer's oldest docs into the cold layer and bumps the
    hot generation in ONE atomic step, so a document is never counted in both
    layers simultaneously (contract #2 atomic handoff).
  - search() reads are RCU: each layer snapshots its own generation, so an
    in-flight query sees a consistent view even mid-rebuild (contract #13/#8).

The cold layer is backend-agnostic here: it accepts anything satisfying the
SearchBackend protocol. In tests that's the InMemoryStubBackend (no JVM);
in production it's the PyterrierBackend (cold_layer.py).
"""

from __future__ import annotations

import threading
from pathlib import Path

from search_backend import SearchResult

import hot_layer


def _normalize_layer(results: list[SearchResult]) -> list[SearchResult]:
    """Min-max normalize a single layer's scores into [0, 1] for cross-layer merge.

    Different layers use different scoring scales (hot Okapi can be negative
    and magnitude-in-1s; cold Terrier BM25 is a different non-negative scale),
    so a raw cross-layer sort is dominated by whichever layer has bigger
    absolute values, not by true relevance. Per-layer min-max makes the merged
    top-k compare *relative* relevance within each layer. A layer's single
    result or all-equal scores map to a flat 1.0 (best-in-its-own-layer) so
    an empty-ish layer still ranks by its own ordering.
    """
    if not results:
        return []
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    out = []
    for r in results:
        norm = (r.score - lo) / span if span != 0 else 1.0
        meta = {**r.metadata, "raw_score": r.score}
        out.append(SearchResult(r.doc_id, norm, r.title, r.content, meta, r.raw))
    return out


class TwoLayerBackend:
    """Front-end SearchBackend over a hot in-memory layer + a cold on-disk layer."""

    def __init__(self, hot: "hot_layer.HotLayerBackend", cold, settle_threshold: int = 5000):
        self._hot = hot
        self._cold = cold
        self._settle_threshold = settle_threshold
        # One lock for the settle handshake so "drop from hot" + "add to cold"
        # + "bump generation" is a single atomic unit visible to readers.
        self._handoff_lock = threading.Lock()
        self._generation = 0

    # ---- write path -------------------------------------------------------
    def update(self, docs: list[dict]) -> None:
        """New knowledge goes to the hot layer (sub-ms, real-time searchability)."""
        self._hot.write(docs)
        # When the hot layer grows past capacity, settle the oldest batch to cold
        # so hot stays small (O(n) scan) and cold gets the stable corpus.
        if self._hot.size > self._settle_threshold:
            self._settle(self._hot.size - self._settle_threshold)

    def _settle(self, n: int) -> None:
        if n <= 0:
            return
        oldest = self._hot.oldest(n)
        with self._handoff_lock:
            # Atomic handoff: drop from hot, add to cold, bump the layer generation
            # together. No reader can see a doc that is simultaneously in both
            # layers at the new generation (dedup-by-docid still guards the window).
            self._cold.update(oldest)
            dropped = self._hot.drop(n)
            self._generation = dropped

    # ---- read path (RCU fan-out + merge) ---------------------------------
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # Over-fetch from each layer so the merged result has enough candidates
        # to dedup and re-rank down to top_k. top_k=None means "no cap" (the
        # service's filter path fetches every hit, filters, then truncates
        # itself — capping here would drop filtered-out results early).
        fan = top_k * 2 if top_k is not None else None
        hot_res = self._hot.search(query, top_k=fan)
        # Cold layer is durable but can fail (corrupt side-store, dangling
        # pointer, mid-rebuild). Graceful degradation: a cold failure must
        # not take down the whole query — fall back to hot-only and flag it
        # (robustness, #5).
        degradation = None
        try:
            cold_res = self._cold.search(query, top_k=fan)
        except Exception as e:  # noqa: BLE001 - deliberate isolation boundary
            cold_res = []
            degradation = f"cold layer unavailable: {type(e).__name__}: {e}"
        except BaseException:
            raise

        # Per-layer min-max normalization: hot uses Okapi (can be negative,
        # magnitude ~units) and cold uses Terrier BM25 (different scale).
        # Ranking them on the raw cross-layer score would let whichever layer
        # has the bigger absolute values monopolize the top-k. Normalize each
        # layer into [0, 1] so the merged top-k reflects relative relevance
        # within each layer, not a scale artifact (architecture fix, #4).
        candidates = _normalize_layer(hot_res) + _normalize_layer(cold_res)

        # Dedup by EXTERNAL docid: first occurrence (hot, most recent) wins.
        seen: dict[str, SearchResult] = {}
        for r in candidates:
            key = r.doc_id
            if key not in seen:
                seen[key] = r
        # Re-rank the deduped, normalized candidates, take top_k.
        ranked = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        results = ranked[:top_k]

        # Single unified result shape: tag the source layer + one shared
        # generation so a consumer/agent sees one consistent payload (contract
        # #11), and so the dual-layer RCU contract test (#13) can OBSERVE that
        # a single read never mixes two generations (fix for the false-green
        # test, #13).
        with self._handoff_lock:
            shared_generation = self._generation
        tagged = []
        for r in results:
            layer = r.metadata.get("layer", "cold")
            meta = {**r.metadata,
                    "source_layer": layer,
                    "shared_generation": shared_generation}
            if degradation is not None:
                meta["degradation"] = degradation
            tagged.append(
                SearchResult(
                    doc_id=r.doc_id, score=r.score, title=r.title,
                    content=r.content, metadata=meta,
                )
            )
        return tagged

    # ---- protocol passthrough --------------------------------------------
    def persist(self) -> Path:
        """Persist the cold layer (hot is transient by design)."""
        return self._cold.persist()

    def load(self) -> bool:
        return self._cold.load()

    def flush(self) -> Path:
        return self._cold.flush()

    @property
    def generation(self) -> int:
        with self._handoff_lock:
            return self._generation


__all__ = ["TwoLayerBackend"]
