"""Hot layer: in-memory rank_bm25 index for recently-written knowledge.

Design (PRD §12, dual-layer hot/cold):
  - write path takes the lock and rebuilds the BM25Okapi over the full in-memory
    doc list, then bumps the generation.
  - read path is RCU: inside ONE lock acquisition we snapshot the (docs, bm25,
    generation) triple as a single consistent unit, release the lock, then score.
    This is NOT the three separate unlocked attribute reads @reviewer flagged —
    the snapshot is taken atomically so a reader never sees a docs/generation
    pair from two different writes.
  - NO score>0 filter: Okapi IDF on a small (hot) corpus goes negative/zero,
    and filtering by sign would silently empty small-corpus results (the B1
    lesson, now a contract test #12). We keep the top-k by relative rank.
  - Results carry the external docid (join key) + a generation tag, so the
    merge layer and the RCU contract test can both verify consistency.

No model, no API — pure rank_bm25 over jieba tokens (PRD §1 core constraint).
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

import jieba

from search_backend import SCHEMA_VERSION, SearchBackendError, SearchResult

# Single shared jieba tokeniser. Module-level, so no per-call prefix-dict cost.
# (jieba caches its dict after first load; the hot path just calls cut.)


import re

# Independent number tokens: the review of the conc "同模板不同编号" mis-kills
# found jieba drops single-char digits ("编号0" -> the 0 in "编号0" is a
# len-1 token, filtered out by the len>1 guard), so 编号0/1/2 all collapse to
# the SAME word-set and the near-dup self-check can't distinguish them.
# Extracting \d+ as independent tokens puts "编号0" -> [编号, 0], "编号1" ->
# [编号, 1] -> word sets differ -> Jaccard ~0.83 -> _DEDUP_BLOCK=0.85 passes
# naturally. Runs on the raw text before jieba so a digit glued to a CJK word
# is still captured; a standalone integer in the text is one token each.
_NUM_TOKEN_RE = re.compile(r"\d+")


def tokenize(text: str) -> list[str]:
    """jieba word segmentation; keep content words only (len>1, no pure punct).

    Number runs (\\d+) are also emitted as independent tokens (the "编号0/1/2"
    fix): jieba drops single-digit CJK-adjacent numbers under the len>1
    guard, so they never reach the index. Without this, same-template
    docs differing only in a trailing number collapse to identical word
    sets and the near-duplicate gate mis-kills them (reviewer acceptance
    #1 for the conc mis-kill card)."""
    base = [w for w in jieba.lcut(text) if len(w.strip()) > 1 and not w.isspace()]
    nums = _NUM_TOKEN_RE.findall(text or "")
    # Preserve insertion order without dupes within the doc (a repeated "0"
    # in two places stays a single token for set-based overlap).
    seen: set[str] = set()
    ordered = [w for w in base if not (w in seen or seen.add(w))]
    for n in nums:
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered


class HotLayerBackend:
    """In-memory BM25 hot layer over the most-recently written documents."""

    def __init__(self, capacity: int = 10000):
        self._lock = threading.RLock()
        self._docs: list[dict] = []
        self._doc_tokens: list[list[str]] = []   # per-doc tokens (overlap gate)
        self._bm25 = None
        self._generation = 0
        # Hot layer is bounded: when it exceeds capacity the oldest batch is
        # "settled" to the cold layer (see two_layer.settle). We track insertion
        # order so settle() can drop the head.
        self._capacity = capacity
        # Durability: the hot layer holds recent, not-yet-settled docs. Without
        # its own side-store those docs vanish on a process restart (the gap
        # the E2E round-trip test caught). Set via set_persist_path().
        self._persist_path: Optional[Path] = None

    def set_persist_path(self, path: Path) -> None:
        self._persist_path = Path(path)

    def persist(self) -> Optional[Path]:
        """Atomically pickle the not-yet-settled hot docs. No-op (returns None)
        if no persist path was configured."""
        if self._persist_path is None:
            return None
        with self._lock:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persist_path.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump({"schema_version": SCHEMA_VERSION,
                             "generation": self._generation,
                             "docs": self._docs}, f)
            tmp.replace(self._persist_path)
            return self._persist_path

    def load(self) -> bool:
        """Restore not-yet-settled hot docs. A missing side-store means a fresh
        layer (returns True). Corrupt or schema-drifted state raises
        SearchBackendError — it never silently mis-parses."""
        if self._persist_path is None or not self._persist_path.exists():
            return True
        with self._lock:
            try:
                with open(self._persist_path, "rb") as f:
                    data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                raise SearchBackendError(
                    f"Corrupt hot corpus {self._persist_path}: {e}") from e
            if data.get("schema_version") != SCHEMA_VERSION:
                raise SearchBackendError(
                    f"Hot corpus {self._persist_path} schema_version "
                    f"{data.get('schema_version')!r} != expected {SCHEMA_VERSION}")
            self._docs = data.get("docs", [])
            self._generation = data.get("generation", 0)
            self._rebuild_index()
        return True

    # ---- write path -------------------------------------------------------
    def write(self, docs: list[dict]) -> int:
        """Append docs; rebuild the index and bump the generation.

        Each doc must carry an external ``docid`` (the cross-layer join key).
        Returns the new generation.
        """
        if not docs:
            with self._lock:
                return self._generation
        with self._lock:
            self._docs.extend(docs)
            self._rebuild_index()
            self._generation += 1
            return self._generation

    def _rebuild_index(self) -> None:
        """Rebuild BM25 + per-doc token lists together. Must be under lock."""
        self._doc_tokens = [tokenize(self._doc_text(d)) for d in self._docs]
        # BM25Okapi divides by avgdl — a corpus with NO tokens anywhere
        # (all docs segment to zero content words) is a ZeroDivisionError,
        # not an empty index. Build only when at least one doc has tokens;
        # otherwise search() returns [] via the bm25-is-None guard.
        self._bm25 = (
            BM25Okapi(self._doc_tokens)
            if self._docs and any(self._doc_tokens) else None
        )

    # ---- read path (RCU snapshot) ---------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """RCU read: snapshot (docs, doc_tokens, bm25, generation) atomically,
        then score.

        A term-overlap gate keeps a doc only if it shares at least one
        content word with the query. This is score-sign-agnostic: a
        small-corpus match whose Okapi IDF is 0 or negative (the B1 lesson)
        still shares a word, so it is kept — while a true no-match (no shared
        word, contract #10) is dropped. This supersedes the old "keep all by
        rank" approach, which let no-match docs surface as placeholders."""
        with self._lock:
            docs = list(self._docs)
            doc_tokens = list(self._doc_tokens)
            bm25 = self._bm25
            generation = self._generation
        if bm25 is None or not docs:
            return []  # hot layer empty -> no placeholders (contract #14)
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_terms = set(q_tokens)
        scores = bm25.get_scores(q_tokens)
        candidates = [
            i for i in range(len(doc_tokens))
            if q_terms & set(doc_tokens[i])
        ]
        if top_k is None:
            top = sorted(candidates, key=lambda i: scores[i], reverse=True)
        else:
            top = sorted(candidates, key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                doc_id=str(docs[i].get("docid", i)),
                score=float(scores[i]),
                title=str(docs[i].get("title", "")),
                content=str(docs[i].get("content", "")),
                metadata={"layer": "hot", "generation": generation,
                          **{k: v for k, v in docs[i].items()
                              if k not in ("docid", "title", "content")}},
            )
            for i in top
        ]

    # ---- settle path ------------------------------------------------------
    def oldest(self, n: int) -> list[dict]:
        """The n oldest docs, for settling them into the cold layer.

        Returns them without dropping here — two_layer.settle() coordinates the
        generation handoff (contract #2: settle + generation bump must be atomic
        so a doc is not double-counted across layers during the window).
        """
        with self._lock:
            return list(self._docs[:n])

    def drop(self, n: int) -> int:
        """Remove the n oldest docs after they've settled to cold; new generation."""
        with self._lock:
            if n <= 0:
                return self._generation
            self._docs = self._docs[n:]
            self._rebuild_index()
            self._generation += 1
            return self._generation

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._docs)

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def store_path(self) -> Optional[Path]:
        """The hot side-store path (None if persist wasn't configured)."""
        return self._persist_path

    def docs_snapshot(self) -> list[dict]:
        """Co-copy of the in-memory docs (for the service's cross-process
        rebase). Read under the layer lock."""
        with self._lock:
            return [dict(d) for d in self._docs]

    def replace(self, docs: list[dict], generation: int) -> None:
        """Atomically replace the in-memory docs (cross-process rebase, called
        by the service under the data-dir flock)."""
        with self._lock:
            self._docs = list(docs)
            self._generation = generation
            self._rebuild_index()

    @staticmethod
    def _doc_text(doc: dict) -> str:
        return f"{doc.get('title', '')} {doc.get('content', '')}"


__all__ = ["HotLayerBackend", "tokenize"]
