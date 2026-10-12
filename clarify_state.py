#!/usr/bin/env python3
"""kb clarify state sidecar (owner 2026-10-13 ruling).

The write-side hook (kb_ingest_hook) keeps the clarify-trigger state in
MEMORY per process (_fire_clarify_fired / ClarifyAggregator buffer+asked /
_last_session / _last_boundary_prev, all under _STATES keyed by id(ctx)).
A gateway restart zeroes that memory: the PENDING docs survive (they are
already durable in staging_docs.jsonl — the single source of truth for the
FACT, owner ruling "从 staging_docs.jsonl 按 doc_id 重读"), but the DECISION
STATE ("which session fired a trigger", "which docs were already asked")
is lost, so the read-side pre_llm_call [知识待确认] footnote silently stops
surfacing and the pending docs rot in staging as zombies.

This module is the persistence backstop for the DECISION STATE ONLY:
  - path: <data_dir>/clarify_state.json (independent sidecar, physically
    isolated from the staging/ dir — owner ruling "独立 sidecar 文件")
  - shape: {"sessions": {"<sid>": {"fired": bool, "asked": [doc_id, ...]}},
            "last_session": "...", "last_boundary_prev": "..."}
  - the sidecar stores doc_ids + session markers; it NEVER stores the
    candidate's title/label/content — those are re-read from the staging
    store by the read-side face at footnote time, so a confirmed doc (its
    staging row moved out) naturally vanishes from the footnote.

Concurrency: the service's data-dir flock (kb_service._open_lock) already
serialises all governance writers (ingest/confirm) against this data dir,
and the service is the process-wide singleton that owns one StagingStore,
so each process takes the SH/EX lock around every sidecar write and
re-reads on the read path. A gateway restart is the only expected reader
of the post-restart state, and it acquires the lock on first use.

Ownership: this file is written by the knowledge-base service layer
(kb_service / this module), NOT by the host gateway. The host code
(/usr/local/lib/hermes-agent) must never import or touch it.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from search_backend import SearchBackendError


class ClarifyStateStore:
    """Durable decision-state sidecar for the KB clarify face.

    Per-session:  {"fired": bool, "asked": [doc_id, ...]}
    Global:       "last_session", "last_boundary_prev"

    All mutations are atomic (tmp + os.replace under the caller's lock).
    A missing / corrupt sidecar degrades to an EMPTY state (fail-soft:
    the read face re-derives from the staging FACT; the decision-state
    loss is bounded to "re-ask a batch once after a restart").
    """

    def __init__(self, data_dir: Path):
        self._dir = Path(data_dir)
        self._path = self._dir / "clarify_state.json"
        self._state: Dict[str, Any] = self._load()

    # ---- load / persist --------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("sessions", {})
                    data.setdefault("last_session", None)
                    data.setdefault("last_boundary_prev", None)
                    return data
            except Exception:
                # Corrupt sidecar: start empty; the caller logs the reason.
                # The FACT (staging) is unaffected — only the decision-state
                # resets, which is the documented fail-soft bound.
                pass
        return {
            "sessions": {},
            "last_session": None,
            "last_boundary_prev": None,
        }

    def _persist(self) -> None:
        """Atomically rewrite the sidecar (tmp + replace, same pattern as
        StagingStore._persist_rows — no torn file survives a crash)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self._dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._state, f, ensure_ascii=False)
            os.replace(tmp, str(self._path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ---- session-level decision state ------------------------------------

    def session_entry(self, session_id: str) -> Dict[str, Any]:
        """This session's fired/asked entry (created empty on first touch)."""
        e = self._state["sessions"].setdefault(
            str(session_id), {"fired": False, "asked": []})
        if "fired" not in e:
            e["fired"] = False
        if "asked" not in e:
            e["asked"] = []
        return e

    def mark_fired(self, session_id: str) -> None:
        self.session_entry(session_id)["fired"] = True
        self._persist()

    def mark_asked(self, session_id: str, doc_ids: list) -> None:
        e = self.session_entry(session_id)
        asked = e["asked"]
        for d in doc_ids:
            d = str(d)
            if d not in asked:
                asked.append(d)
        # Bounded LRU mirror of ClarifyAggregator._asked_cap (500): the
        # sidecar must not grow without bound on a long-lived gateway.
        cap = 500
        if len(asked) > cap:
            del asked[: len(asked) - cap]
        self._persist()

    def asked_for(self, session_id: str) -> list:
        return list(self.session_entry(session_id)["asked"])

    def pending_doc_ids(self, session_id: str, all_doc_ids: list) -> list:
        """The session's pending doc_ids = all staging rows for it minus
        the asked set (mirrors ClarifyAggregator.pending_for, which
        re-derives from the buffer; here the buffer is re-built from the
        staging FACT by the caller, so we only need the asked-exclusion)."""
        asked = set(self.asked_for(session_id))
        return [d for d in all_doc_ids if str(d) not in asked]

    # ---- global boundary markers -----------------------------------------

    def set_last_session(self, session_id: Optional[str]) -> None:
        self._state["last_session"] = session_id
        self._persist()

    def set_last_boundary_prev(self, session_id: Optional[str]) -> None:
        self._state["last_boundary_prev"] = session_id
        self._persist()

    @property
    def last_session(self) -> Optional[str]:
        return self._state.get("last_session")

    @property
    def last_boundary_prev(self) -> Optional[str]:
        return self._state.get("last_boundary_prev")

    # ---- read-side reconstruction (post-restart) -------------------------

    def snapshot(self) -> Dict[str, Any]:
        """A deep-ish copy of the current decision state (the read face
        consumes this to re-build the in-memory ClarifyAggregator /
        _fire_clarify_fired / boundary markers after a restart)."""
        out: Dict[str, Any] = {
            "last_session": self._state.get("last_session"),
            "last_boundary_prev": self._state.get("last_boundary_prev"),
            "sessions": {},
        }
        for sid, e in (self._state.get("sessions") or {}).items():
            out["sessions"][sid] = {
                "fired": bool(e.get("fired")),
                "asked": list(e.get("asked") or []),
            }
        return out

    @property
    def path(self) -> Path:
        return self._path


__all__ = ["ClarifyStateStore"]
