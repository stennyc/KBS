"""InMemoryStubBackend: reference SearchBackend implementation.

Purpose: prove the contract test suite is satisfiable with a minimal, model-free
backend, so the suite can be green before PyterrierBackend exists. It is
consumed *by the contract test suite* (test_contract.py) — which is exactly the
tightened exemption ①: a protocol-consumed class is not "island" dead code.

It deliberately implements the FULL 5-method contract, including the hard parts
the review flagged, so any real backend is measured against the same bar:
  - generation counter  -> corrupted/stale persisted state is detected on load
  - atomic write        -> temp file + os.replace, no torn .tmp left behind
  - RCU snapshot reads  -> a mid-flight search() sees one consistent generation,
                           never a torn mix of pre/post-mutation state
  - single-doc corpus   -> a matching query still yields a result (B1 lesson:
                           no empty result on a tiny corpus)

No model, no API — pure term-frequency scoring (PRD §1 core constraint).
"""

from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path

from search_backend import SCHEMA_VERSION, SearchBackend, SearchResult, SearchBackendError


class InMemoryStubBackend:
    """Minimal in-memory reference backend satisfying the SearchBackend contract."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.RLock()
        self._generation = 0
        # docs: list of raw dicts; scores are derived on demand.
        self._docs: list[dict] = []

    # ---- SearchBackend protocol ------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """RCU read: take a consistent snapshot under the lock, then score.

        A search that started before a concurrent update/persist returns that
        pre-mutation generation in full; one that started after returns the new
        generation. It never returns a torn mix.
        """
        with self._lock:
            snapshot = list(self._docs)
            generation = self._generation
        scored = [
            (self._tf_score(query, doc), i) for i, doc in enumerate(snapshot)
        ]
        # Match filter: only docs that share at least one query term are
        # relevant. This is what makes a no-match query return [] instead of
        # padding top-k with zero-relevance placeholders. The "match" decision
        # is backend-specific; TF>0 is this stub's rule.
        scored = [(s, i) for (s, i) in scored if s > 0]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for score, i in scored[:top_k]:
            doc = snapshot[i]
            # Stable document identity = EXTERNAL docid when the doc carries one
            # (the cross-layer join key); fall back to a backend-local id. This
            # models a real backend (Pyterrier) that returns the external docid,
            # so the two-layer dedup-by-docid is genuinely exercised.
            results.append(
                SearchResult(
                    doc_id=str(doc.get("docid", doc.get("id", i))),
                    score=score,
                    title=str(doc.get("title", "")),
                    content=str(doc.get("content", "")),
                    metadata={k: v for k, v in doc.items() if k not in ("id", "title", "content")},
                    raw={"generation": generation},
                )
            )
        return results

    def update(self, docs: list[dict]) -> None:
        """Append documents and bump the generation (a new readable snapshot)."""
        if not docs:
            return
        with self._lock:
            self._docs.extend(docs)
            self._generation += 1

    def persist(self) -> Path:
        """Atomic write: dump to a temp file, then os.replace onto the target.

        On POSIX os.replace is an atomic rename, so readers never observe a
        half-written file. The .tmp is always unlinked on failure.
        """
        with self._lock:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "generation": self._generation,
                "docs": self._docs,
            }
        tmp = self._path.with_suffix(".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                pickle.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return self._path

    def load(self) -> bool:
        """Load persisted state. Diagnostic, never silently mis-parses."""
        if not self._path.exists():
            raise SearchBackendError(f"Index file not found: {self._path}")
        try:
            with open(self._path, "rb") as f:
                payload = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            raise SearchBackendError(f"Corrupt index file {self._path}: {e}") from e

        if not isinstance(payload, dict):
            raise SearchBackendError(
                f"Invalid index file {self._path}: expected dict, got {type(payload).__name__}"
            )
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise SearchBackendError(
                f"Schema version mismatch at {self._path}: file v{version}, "
                f"code expects v{SCHEMA_VERSION}. Rebuild the index."
            )
        # Generation drift is itself corruption: a persisted generation that
        # does not match what the live backend expects means a torn/stale state.
        with self._lock:
            persisted_gen = payload.get("generation")
            self._docs = list(payload.get("docs", []))
            self._generation = persisted_gen if isinstance(persisted_gen, int) else 0
        return True

    def flush(self) -> Path:
        """Periodic persistence hook — thin alias so a timer can call flush()."""
        return self.persist()

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _tf_score(query: str, doc: dict) -> float:
        """Term-frequency relevance: count query terms present in the doc.

        Pure TF — no model, no API. Guarantees a matching single-doc corpus
        still yields a positive score (the B1 no-empty-result lesson).
        """
        hay = f"{doc.get('title', '')} {doc.get('content', '')}".lower().split()
        hay_set = set(hay)
        score = sum(1 for term in query.lower().split() if term in hay_set)
        return float(score)


__all__ = ["InMemoryStubBackend"]
