"""Staging layer: physically-isolated store for PENDING knowledge.

P0 governance (room ruling 2026-10-10): a knowledge doc only enters the
SEARCHABLE corpus (hot/cold two-layer) AFTER an explicit user confirm.
Model/agent auto-detection lands the doc in this staging store as
``status=pending``; nothing in staging is ever indexed by the hot/cold
BM25 layers, so a default ``search()`` can never surface a pending doc.

Physical isolation (the reviewer's acceptance criterion #1/#3):
  - staging lives under ``<data_dir>/staging/`` with its OWN row store
    (``staging_docs.jsonl``) and is NOT part of the two-layer backend.
  - ``confirm()`` performs an ATOMIC move: the row is removed from staging
    and appended to the confirmed corpus (hot layer) under one data-dir
    flock, so a reader can never see a doc in both places or in neither.

Cross-process safety reuses the service's flock sidecar (the caller holds
EX around a confirm/staging-write cycle); this module only does the
row-store bookkeeping. No model, no external API — pure local I/O (PRD §1).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from hot_layer import HotLayerBackend
from search_backend import SCHEMA_VERSION, SearchBackendError

# Row-store + index side-store live in the deploy-created staging dir, never
# in the repo (a symlinked repo update must not touch live pending data).
_STAGING_DOCS = "staging_docs.jsonl"
_STAGING_INDEX = "staging_index.pkl"


class StagingStore:
    """Pending-doc row store + an independent HotLayerBackend index.

    The index is a *separate* HotLayerBackend instance (its own generation,
    its own persist path inside ``staging/``) so pending docs get their own
    in-memory BM25 — but that index is NEVER wired into the two-layer
    search path. It exists only so a future confirm/sweep can rank within
    staging. Default search() ignores it entirely (physical isolation).
    """

    def __init__(self, staging_dir: Path, capacity: int = 10000):
        self._dir = Path(staging_dir)
        self._rows_path = self._dir / _STAGING_DOCS
        self._index = HotLayerBackend(capacity=capacity)
        self._index.set_persist_path(self._dir / _STAGING_INDEX)
        self._docs: list[dict] = []
        self._generation = 0
        self._load()

    # ---- load / persist ---------------------------------------------------

    def _load(self) -> None:
        """Restore the row store + the independent index. A missing row file
        means an empty staging area (fresh, returns nothing to load)."""
        if self._rows_path.exists():
            self._docs = self._read_rows()
            self._index.replace(self._docs, self._generation)
        elif self._index.store_path and self._index.store_path.exists():
            # Index survived but rows were wiped out-of-band: rebuild rows
            # from the index snapshot so the two stay consistent.
            self._docs = self._index.docs_snapshot()
            self._persist_rows()
        self._index.load()

    def _read_rows(self) -> list[dict]:
        out: list[dict] = []
        with open(self._rows_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _persist_rows(self) -> None:
        """Atomically rewrite the row file (tmp + replace, same pattern as the
        hot/cold pickle stores so no torn row file survives a crash)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for d in self._docs:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
            os.replace(tmp, str(self._rows_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def persist(self) -> None:
        """Durably write both the row store and the independent index."""
        self._persist_rows()
        self._index.persist()

    # ---- read -------------------------------------------------------------

    def find(self, doc_id: str) -> Optional[dict]:
        """The pending row with this doc_id, or None (under the caller's lock)."""
        for d in self._docs:
            if str(d.get("docid", "")) == str(doc_id):
                return d
        return None

    def docs_snapshot(self) -> list[dict]:
        return [dict(d) for d in self._docs]

    @property
    def size(self) -> int:
        return len(self._docs)

    @property
    def generation(self) -> int:
        return self._index.generation

    @property
    def rows_path(self) -> Path:
        return self._rows_path

    @property
    def store_path(self) -> Optional[Path]:
        """The independent index's pickle side-store (for rebase symmetry)."""
        return self._index.store_path

    # ---- write / remove (caller holds the data-dir EX lock) ---------------

    def add(self, doc: dict) -> int:
        """Register a pending doc. Caller must have stamped ``docid`` + the
        governance fields (status/origin_kind/source_profile) before calling.
        Idempotent on docid: re-adding the same docid replaces the row in
        place (content-addressed identity, same as the hot layer)."""
        docid = str(doc.get("docid", ""))
        if not docid:
            raise SearchBackendError("staging add: doc is missing a docid")
        self._docs = [d for d in self._docs if str(d.get("docid", "")) != docid]
        self._docs.append(dict(doc))
        self._index.replace(self._docs, self._index.generation + 1)
        self._generation = self._index.generation
        self.persist()
        return self._generation

    def remove(self, doc_id: str) -> Optional[dict]:
        """Atomically drop a pending row by docid (the confirm move-out half).

        Returns the removed row (with its full metadata) so the caller can
        re-route it into the confirmed corpus, or None if it wasn't pending.
        """
        kept, removed = [], None
        for d in self._docs:
            if str(d.get("docid", "")) == str(doc_id):
                removed = dict(d)
            else:
                kept.append(d)
        if removed is None:
            return None
        self._docs = kept
        self._index.replace(self._docs, self._index.generation + 1)
        self._generation = self._index.generation
        self.persist()
        return removed

    def _rebase(self, disk_docs: list[dict], disk_gen: int) -> None:
        """Cross-process merge of the row store (union by docid, disk wins the
        base, memory rows layered on top) + reload the index from merged rows."""
        merged: dict[str, dict] = {}
        for d in disk_docs:
            merged[str(d.get("docid", ""))] = dict(d)
        for d in self._docs:
            merged[str(d.get("docid", ""))] = dict(d)
        self._docs = list(merged.values())
        self._index.replace(self._docs, max(disk_gen, self._index.generation))
        self._generation = self._index.generation
        self.persist()

    def replace(self, docs: list[dict], generation: int) -> None:
        """Atomic in-memory replace (cross-process rebase entry point)."""
        self._docs = [dict(d) for d in docs]
        self._index.replace(self._docs, generation)
        self._generation = generation


__all__ = ["StagingStore"]
