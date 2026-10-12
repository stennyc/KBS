"""Cold layer: durable full-corpus search backend.

Design (settled after the JVM-gate dispute, 2026-10-05):

  The cold layer's CORRECTNESS is an always-on in-memory rank_bm25 over the
  full corpus — no JVM, reloadable from a pickle side-store, rebuildable on a
  daily/weekly cycle (matches the user's "minute-level tolerated" constraint).
  This is what search() uses, so the single-shot tool call is never broken by
  a missing JVM and there is no "just-settled doc not yet searchable" gap
  (settled docs enter the rank_bm25 corpus immediately).

  Pyterrier on-disk (DFIndexer) was the intended million-scale accelerator,
  but it is NOT installed on this VPS (ModuleNotFoundError, confirmed by
  arb/probe_env.py) and its cross-process reload path was never proven here.
  Per the room's YAGNI ruling the dead build path is deleted outright — it was
  the source of B1 (an unconsumed retriever: search() read only rank_bm25,
  nothing ever queried the Terrier index) and B2 (generation-dir vs gen-0
  write mismatch), and it mis-created an empty knowledge-base/pyterrier/
  index dir. Re-introduce it only when a JVM-capable host is explicitly
  requested (@user); until then the in-memory corpus IS the cold layer.

Tokenisation is shared with the hot layer: jieba words, space-joined, lower —
so index terms and query terms always match (the 0-result root cause from
review is closed on both sides).

SearchBackend protocol unchanged; two_layer.py and the existing contract
suite are untouched except for the 3 new real-cold tests.
"""

from __future__ import annotations

import heapq
import pickle
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

from search_backend import SearchResult, SearchBackendError, SCHEMA_VERSION

import hot_layer  # shared jieba tokeniser + segmentation pipeline


class PyterrierBackend:
    """Durable cold layer: always-on in-memory rank_bm25 + pickle side-store.

    JVM-free, so a single-shot query is never broken on any VPS. The class
    name is retained for API/OID continuity across the room; see the module
    docstring for why it no longer touches Pyterrier."""

    def __init__(self, base_dir: Path):
        self._base = Path(base_dir)
        self._corpus_pkl = self._base / "cold_corpus.pkl"
        self._lock = threading.RLock()
        self._docs: list[dict] = []
        self._doc_tokens: list[list[str]] = []  # per-doc tokens (overlap gate)
        self._bm25 = None          # in-memory rank_bm25 over full corpus
        self._generation = 0

    # ---- SearchBackend protocol ------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        with self._lock:
            docs = list(self._docs)
            doc_tokens = list(self._doc_tokens)
            bm25 = self._bm25
            generation = self._generation
        if bm25 is None or not docs:
            return []  # empty corpus -> no placeholders (contract #4/#12)
        tokens = _query_tokens(query)
        if not tokens:
            return []
        q_terms = set(tokens)
        scores = bm25.get_scores(tokens)
        # Term-overlap gate (same as the hot layer): keep a doc only if it
        # shares a content word with the query. A settled small corpus can
        # produce a 0.0/low Okapi score for a real match, so we gate on word
        # overlap, not score sign — a true no-match (no shared word, contract
        # #10) is dropped, a real match that happens to score 0 is kept.
        hit = [i for i in range(len(doc_tokens)) if q_terms & set(doc_tokens[i])]
        if top_k is None:
            top = sorted(hit, key=lambda i: scores[i], reverse=True)
        else:
            top = heapq.nlargest(top_k, hit, key=lambda i: scores[i])
        return [
            SearchResult(
                doc_id=str(docs[i].get("docid", i)),
                score=float(scores[i]),
                title=str(docs[i].get("title", "")),
                content=str(docs[i].get("content", "")),
                metadata={"layer": "cold", "generation": generation,
                          **{k: v for k, v in docs[i].items()
                              if k not in ("docid", "title", "content")}},
            )
            for i in top
        ]

    def update(self, docs: list[dict]) -> None:
        """Settled docs land in the in-memory cold corpus immediately, so
        they are searchable with no rebuild gap (fixes the settle-window
        miss the old build-then-reload design had)."""
        if not docs:
            return
        with self._lock:
            self._docs.extend(docs)
            self._rebuild_bm25()
            self._generation += 1

    def persist(self) -> Path:
        """Durability: pickle the cold corpus atomically (reloadable,
        JVM-free). Atomic via tmp + replace so no torn file survives."""
        with self._lock:
            self._base.mkdir(parents=True, exist_ok=True)
            tmp = self._corpus_pkl.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump({
                    "schema_version": SCHEMA_VERSION,
                    "generation": self._generation,
                    "docs": self._docs,
                }, f)
            tmp.replace(self._corpus_pkl)
            return self._corpus_pkl

    def load(self) -> bool:
        """Reload the pickle side-store. Raises SearchBackendError on
        corrupt or schema-drifted state — never silently mis-parses."""
        with self._lock:
            if not self._corpus_pkl.exists():
                return True  # fresh layer, nothing to load
            try:
                with open(self._corpus_pkl, "rb") as f:
                    data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                raise SearchBackendError(
                    f"Corrupt cold corpus {self._corpus_pkl}: {e}") from e
            if data.get("schema_version") != SCHEMA_VERSION:
                raise SearchBackendError(
                    f"Cold corpus {self._corpus_pkl} schema_version "
                    f"{data.get('schema_version')!r} != expected "
                    f"{SCHEMA_VERSION}")
            self._docs = data.get("docs", [])
            self._generation = data.get("generation", 0)
            self._rebuild_bm25()
        return True

    def flush(self) -> Path:
        return self.persist()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def store_path(self) -> Path:
        """The pickle side-store (used by the service's cross-process rebase)."""
        return self._corpus_pkl

    def docs_snapshot(self) -> list[dict]:
        """Co-copy of the in-memory docs (for the service's cross-process
        rebase). Read under the layer lock."""
        with self._lock:
            return [dict(d) for d in self._docs]

    def replace(self, docs: list[dict], generation: int) -> None:
        """Atomically replace the in-memory corpus (cross-process rebase,
        called by the service under the data-dir flock)."""
        with self._lock:
            self._docs = list(docs)
            self._generation = generation
            self._rebuild_bm25()

    # ---- internal ---------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        """Must be called under self._lock."""
        self._doc_tokens = [_doc_tokens(d) for d in self._docs]
        # Same guard as the hot layer: BM25Okapi divides by avgdl, so an
        # all-empty-token corpus must not be built (ZeroDivisionError);
        # search() then returns [] via the bm25-is-None guard.
        self._bm25 = (
            BM25Okapi(self._doc_tokens)
            if self._docs and any(self._doc_tokens) else None
        )


def _query_tokens(query: str) -> list[str]:
    """jieba words (lower), space-stripped, len>1 — mirrors the index side."""
    return [w.lower() for w in hot_layer.tokenize(query) if len(w) > 1]


def _doc_tokens(doc: dict) -> list[str]:
    text = f"{doc.get('title','')} {doc.get('content','')}"
    return [w.lower() for w in hot_layer.tokenize(text) if len(w) > 1]


__all__ = ["PyterrierBackend"]
