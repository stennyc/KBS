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
import time
from pathlib import Path
from typing import Optional

from hot_layer import HotLayerBackend
from search_backend import SCHEMA_VERSION, SearchBackendError

# Row-store + index side-store live in the deploy-created staging dir, never
# in the repo (a symlinked repo update must not touch live pending data).
_STAGING_DOCS = "staging_docs.jsonl"
_STAGING_INDEX = "staging_index.pkl"

# note 4 (PRD v2-final): TTL sweep threshold — a pending staging row
# with no user interaction for more than 30 days is re-keyed to the
# stale_staging state (audit-only, excluded from the active clarify
# batch + confirm surface). The 30-day clock is anchored on the row's
# first-seen ingest_ts (the deterministic first-seen ts the ReinforceStore
# sidecar tracks for confirmed docs; a pending row without an ingest_ts
# falls back to the row's last-modified mtime — a row that is never
# interacted with keeps its original ts, the fail-soft posture).
_STALE_STAGING_TTL_DAYS = 30


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

    # ---- note 4: TTL sweep to stale_staging -------------------------------

    def find_active(self, doc_id: str) -> Optional[dict]:
        """The ACTIVE (confirmable / clarify-eligible) row with this doc_id.

        ``find`` returns any row (including a re-keyed stale_staging
        audit copy); ``find_active`` excludes rows whose status is
        ``stale_staging`` — the active surface is what the clarify
        batch + confirm() move-out half operate on, so a stale row is
        invisible to both (留审计: the row still exists in the store,
        just not on the active face).
        """
        for d in self._docs:
            if str(d.get("docid", "")) == str(doc_id):
                if str(d.get("status", "pending")) == "stale_staging":
                    return None
                return d
        return None

    def sweep_stale(self, now_ts: Optional[float] = None) -> list[dict]:
        """note 4 (PRD v2-final): re-key idle PENDING rows to
        ``stale_staging``.

        A row that has been PENDING with no user interaction for more
        than ``_STALE_STAGING_TTL_DAYS`` (30) days is moved to the
        ``stale_staging`` state: it is EXCLUDED from the active clarify
        batch + confirm surface (see ``find_active``) but KEPT in the
        row store for audit (the B2 tier-conflict blocked rows and the
        B4 transcript-drop rows are the two producers of long-idle
        pending rows; this sweep is the terminal backstop against
        unbounded growth — the owner's Phase D ruling).

        The idle clock is anchored on the row's first-seen
        ``ingest_ts`` when present, else the row's ``_staged_at``
        (stamped here on first add), else now (unknown age -> not
        swept, the fail-soft posture).

        Idempotent: a second sweep finds no PENDING rows to move.

        Returns the list of swept (re-keyed) rows, or [] when nothing
        was idle long enough.
        """
        if now_ts is None:
            now_ts = time.time()
        cutoff = now_ts - _STALE_STAGING_TTL_DAYS * 86400.0
        swept: list[dict] = []
        changed = False
        for d in self._docs:
            if str(d.get("status", "pending")) != "pending":
                continue
            ts = d.get("ingest_ts")
            if ts is None:
                ts = d.get("_staged_at")
            if ts is None:
                # Unknown age: not swept (fail-soft — a row without any
                # interaction ts stays PENDING; the clarify backstop
                # still surfaces it).
                continue
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                continue
            if ts < cutoff:
                d["status"] = "stale_staging"
                swept.append(dict(d))
                changed = True
        if changed:
            self._index.replace(self._docs, self._index.generation + 1)
            self._generation = self._index.generation
            self.persist()
        return swept

    def add(self, doc: dict) -> int:
        """Register a pending doc. Caller must have stamped ``docid`` + the
        governance fields (status/origin_kind/source_profile) before calling.
        Idempotent on docid: re-adding the same docid replaces the row in
        place (content-addressed identity, same as the hot layer).

        note 4: stamps ``_staged_at`` (first-seen wall ts) so the TTL
        sweep has an idle anchor even when the row never carries an
        ``ingest_ts`` (a pending row that is re-added later KEEPS its
        original ``_staged_at`` — the clock does not reset on re-stage)."""
        docid = str(doc.get("docid", ""))
        if not docid:
            raise SearchBackendError("staging add: doc is missing a docid")
        # Keep the original _staged_at across re-adds (idempotent clock).
        existing = self.find(docid)
        new_doc = dict(doc)
        if existing and existing.get("_staged_at") is not None:
            new_doc["_staged_at"] = existing["_staged_at"]
        elif new_doc.get("_staged_at") is None:
            new_doc["_staged_at"] = time.time()
        self._docs = [d for d in self._docs if str(d.get("docid", "")) != docid]
        self._docs.append(new_doc)
        self._index.replace(self._docs, self._index.generation + 1)
        self._generation = self._index.generation
        self.persist()
        return self._generation

__all__ = ["StagingStore"]
