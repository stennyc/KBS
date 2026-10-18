"""Knowledge-base search service: the single-shot entry point.

This is the object the Hermes plugin (knowledge-base-search) and the
end-to-end tests both call — it is deliberately NOT coupled to Hermes, so the
retrieval logic stays unit-testable. The plugin's handler is a thin adapter
that builds a KbSearchService and forwards args.

Single-shot contract (PRD §13.2): one search() call returns a structured JSON
payload with full title + content + score + metadata per hit, so a consuming
agent never needs a second fetch.

Layers (architecture settled in 57d2ee6):
  hot  = rank_bm25 over recently-ingested docs (real-time, write)
  cold = rank_bm25 over the settled full corpus, pickle-persisted (durable)
Both are JVM-free — no model, no external API, local only (PRD §1).

Cross-process locking (B2-(b), the 7-gateway deployment shape):
  7 profile gateways can hold this tool in 7 processes against ONE shared
  data dir. Atomic tmp+replace writes alone only guard a single process; the
  fcntl flock sidecar (<data_dir>/lock) guards the whole cycle:
    - read (search): LOCK_SH
    - write (ingest / persist / the settle drop+add+persist unit): LOCK_EX
  and every EX write REBASES first — the on-disk corpus is merged into this
  process's memory before persisting, so a later writer can never silently
  clobber an earlier writer's docs (the last-writer-wins hole a bare flock
  leaves open). A stale long-lived reader detects the seq marker and
  upgrades its lock in place to rebase before answering.

  The lock file and seq marker live in the DATA dir, never in the repo: a
  symlinked repo update must not touch the lock.

Startup guard: a missing data dir raises SearchBackendError (loud fail).
The deploy step creates the dir; the runtime guard never auto-creates it —
two mutually exclusive responsibilities so 7 shared profiles can't each
fall back to a quiet empty index.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from hot_layer import HotLayerBackend
from cold_layer import PyterrierBackend
from two_layer import TwoLayerBackend
from search_backend import SearchBackendError, SearchResult
from staging import StagingStore

try:
    import fcntl
except ImportError:  # non-POSIX dev fallback: no-op locking, single process
    fcntl = None

# Lock + write-epoch marker live in the DATA dir (deploy-created), not the repo.
_LOCK_NAME = "lock"
_SEQ_NAME = "seq"
# Staging sub-dir (deploy-created, 770): pending knowledge, physically isolated
# from the two-layer confirmed corpus.
_STAGING_DIR_NAME = "staging"
# Out-of-band external-replace signal (owner 2026-10-15): a sweep/curator
# tool (tools/kb_stage_v2_sweep.py --commit) rewrites the staging rows +
# drops the index pkl on disk; this marker file (written by the same tool,
# same transaction) tells the NEXT in-process rebase to adopt the on-disk
# row set WHOLESALE instead of union-merging it (a union can only add
# docids back, never drop the quarantined ones). One-shot: consumed on
# first sight under the EX lock.
_STAGING_EXTERNAL_REPLACE = "staging_external_replace"
# Audit trail (a single jsonl, devops-owned path; the service only emits).
_AUDIT_NAME = "knowledge-base-ingest.jsonl"
# Governance gates (P0 room ruling):
#   length gate — title/content bounds (reject out-of-range, log the reason).
_TITLE_MIN, _TITLE_MAX = 5, 200
_CONTENT_MIN, _CONTENT_MAX = 20, 10000
#   near-duplicate self-check (BM25 over the confirmed corpus, no model/API):
#     top-1 normalized score >= _DEDUP_BLOCK  -> reject/merge (near-exact dup)
#     _DEDUP_AUDIT <= score < _DEDUP_BLOCK     -> log "suspected dup", don't block
#     score < _DEDUP_AUDIT                     -> accept
_DEDUP_BLOCK = 0.85
_DEDUP_AUDIT = 0.60
# TR-1 (agent 经验账本): the four experience-ledger fields travel through the
# EXISTING metadata channel (hot/cold doc-dict passthrough: every doc key but
# docid/title/content lands in search metadata). They are defaulted at the
# ingest boundary ONLY when absent (setdefault, the occurred_at/version_ts
# precedent) so every confirmed experience doc carries all four keys.
#   topic           — controlled vocabulary string, default "general"
#   task_type       — free-form task label, None = unset (absent in metadata)
#   premise         — premise-condition text, None = unset
#   premise_status  — True | "needs_verification" | "falsified" (default True,
#                     i.e. "premise held at ingest"); search() filters key.
# TR-5 (premise falsification state bit): the only legal premise_status
# transitions (set_premise_status enforces these; anything else is rejected +
# audited): True -> needs_verification -> falsified, falsified ->
# needs_verification (re-check), falsified -> True (re-verified).
_PREMISE_STATUSES = (True, "needs_verification", "falsified")
_PREMISE_STATUS_MIGRATIONS: Dict[str, tuple] = {
    "true": ("needs_verification",),
    "needs_verification": ("falsified", "true"),
    "falsified": ("needs_verification", "true"),
}


def _premise_status_canonical(value: Any) -> Optional[str]:
    """The canonical spelling of a valid premise_status value, else None.

    Accepts True/"true" (the default), "needs_verification", "falsified" —
    case-insensitive for the two string values; anything else is invalid.
    The input value is stored VERBATIM in the doc dict (bool True stays
    bool True — JSON/pickle roundtrips are byte-identical), so this helper
    is used ONLY for the migration-table lookup and for audit records.
    """
    if value is True or value == "true":
        return "true"
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("needs_verification", "falsified"):
            return low
    return None


# Containment near-dup gate (path A, replaces the min-max-BM25 second gate):
# the candidate's own token set is the QUERY; for each existing doc,
# containment = |query_tokens ∩ doc_tokens| / |query_tokens|. This is the
# "short query contained in a doc" ratio, scale-stable across corpus size
# (unlike min-max-normalized BM25, which flattens to 1.0 in a small corpus
# and mis-kills same-template-different-number docs). >= _DEDUP_BLOCK ->
# reject (near-exact dup); _DEDUP_AUDIT.. -> flag suspected; < -> accept.
# Number tokens (\d+) ARE in the set (hot_layer.tokenize), so 编号0/1/2
# differ -> containment < 0.85 -> passes.
_DEDUP_CONTAINMENT_BLOCK = 0.85
_DEDUP_CONTAINMENT_AUDIT = 0.60

# Observability (上线后排查靠猜→有日志): structured records on the hot paths.
# The service only *emits* records; wiring the handler (a log file) is the
# adapter's job via configure_logging(), so the layer stays Hermes-free.
# Single file, no rotation — KB is small; grep once = full trace.
logger = logging.getLogger("knowledge-base")


# ---- B3: deterministic content-addressed docid ---------------------------

def _deterministic_docid(doc: dict) -> str:
    """Content-addressed docid for docs that arrive without one.

    uuid5 over the sorted JSON of the doc (minus any docid key): the same
    content maps to the same id in every process, so cross-process dedup
    keeps working, re-ingest is idempotent, and a layer position number
    (the old ``str(d.get("docid", i))`` fallback) never leaks into
    identity — that leak broke the external-docid dedup contract #15.

    B1 None-filter (PRD v2-final, owner ruling): the payload also drops
    every None-VALUED key — ``k != "docid" and v is not None``. A
    setdefault-stamped ``gov_tier=None`` / ``project_ref=None`` key must
    not pollute the docid: ``null`` in the JSON payload used to change
    the uuid5 for docs that only differ by None-valued keys, so a
    re-ingest of the SAME content minted a NEW docid and left the old
    row behind (the double-row bug the set_premise_status comment
    guards against). After the filter, ``key present with None`` and
    ``key physically absent`` map to the same docid — the unified
    "None ≡ key absent" semantics (the filter face's ``__unset__``
    sentinel, see search(), is the same ruling on the read side)."""
    payload = json.dumps(
        {k: v for k, v in sorted(doc.items())
         if k != "docid" and v is not None},
        ensure_ascii=False, default=str,
    )
    return str(uuid.uuid5(uuid.NAMESPACE_OID, payload))


def _normalize_docs(docs: list[dict]) -> list[dict]:
    """Ingest boundary: every doc carries a docid — supplied or deterministic.

    Also extracts the card-2 event-time fields from the raw doc text
    (kb_rerank.extract_event_time, pure regex, zero LLM):

      - ``occurred_at`` — a decision/event point-in-time (a decision doc's
        "2024-03-15 拍板" or a config's adoption date). Fed to the read-side
        RRF time axis. A mechanism doc with no date -> None (never a
        penalty on the time axis, owner card-2 ruling).
      - ``version_ts`` — a config version stamp ("kafka_3.6.1"). Stored as a
        FILTER TAG only; NOT fed to the time axis (a version number and an
        event time are different dimensions, mixing them would let a config
        version decay like a dated decision).

    Does NOT stamp a process-local ``created_at`` onto the doc dict: that
    field would differ across the 7 gateway processes for the SAME docid and
    break the cross-process rebase merge. Recency for the RRF axis falls back
    to the deterministic ``ingest_ts`` sidecar when occurred_at is None."""
    from kb_rerank import extract_event_time
    out = []
    for d in docs:
        d = dict(d)
        if not d.get("docid"):
            d["docid"] = _deterministic_docid(d)
        # Card 2: extract event-time fields from the raw text. Only stamp
        # when the caller didn't supply an explicit value (a doc that states
        # its own occurred_at/version_ts wins over the regex extraction).
        raw_text = f"{d.get('title','')} {d.get('content','')}"
        ev = extract_event_time(raw_text)
        d.setdefault("occurred_at", ev["occurred_at"])
        d.setdefault("version_ts", ev["version_ts"])
        # TR-1: the four experience-ledger fields, defaulted here ONLY when
        # absent (the occurred_at/version_ts setdefault precedent). The
        # values are part of the doc dict, so they (a) flow into the
        # deterministic docid (uuid5 over the whole dict) and (b) pass
        # through into search metadata via the hot/cold doc-dict passthrough
        # (every doc key except docid/title/content). Absent keys keep their
        # setdefault defaults; supplied keys are never overwritten.
        #   task_type/premise: None (absent) means "not set" — deliberately
        #   left out of the doc dict so they stay out of the docid input and
        #   of the metadata payload.
        d.setdefault("topic", "general")
        d.setdefault("premise_status", True)
        # PRD v2-final gov_tier (three-tier governance, B1 setdefault
        # extension point): a NEW controlled enum field, defaulted to
        # None here ONLY when absent (the premise_status setdefault
        # precedent). Because the B1 None-filter keeps None-valued keys
        # OUT of the docid payload, the setdefault is docid-stable for
        # legacy docs — gov_tier=None (absent) and gov_tier absent both
        # hash to the same payload. Supplied non-None values stay in
        # the payload (experience/archive are docid-meaningful).
        d.setdefault("gov_tier", None)
        # doc_kind (orthogonal document-SHAP axis, PRD 2026-10-18 owner-
        # agreed): NOT back-filled — a doc that omits doc_kind stays
        # doc_kind=None ("unlabeled", NOT implicitly "experience"). The
        # default MUST be None (never "experience"): the B1 None-filter
        # keeps a None-valued key out of the docid payload, so
        # doc_kind=None (absent) and doc_kind key-absent hash byte-
        # identically for all legacy docs (the gov_tier precedent, one
        # line up — a non-None default would re-mint every existing
        # docid and sever the reinforce/ingest_ts/graph sidecar links).
        d.setdefault("doc_kind", None)
        out.append(d)
    return out


def _merge_docs(disk: list[dict], mem: list[dict]) -> list[dict]:
    """Union keyed by docid; memory wins a conflict (the writer's fresher
    view of that doc). Insertion order: disk docs first, memory after."""
    out: dict[str, dict] = {}
    for d in disk:
        out[str(d.get("docid", ""))] = dict(d)
    for d in mem:
        out[str(d.get("docid", ""))] = dict(d)
    return list(out.values())


class KbSearchService:
    """Two-layer backend + cross-process data-dir lock + deterministic docid.

    Owns the hot/cold layers and the flock sidecar. Construction loads both
    durable side-stores; every read/write cycle takes the dir lock."""

    def __init__(self, data_dir: Path, hot_capacity: int = 10000,
                 settle_threshold: int = 5000,
                 audit_path: Optional[Path] = None):
        self._data_dir = Path(data_dir)
        # Startup guard (loud fail, never a silent empty index). The deploy
        # step owns creation; this guard refuses to mask its absence.
        if not self._data_dir.is_dir():
            raise SearchBackendError(
                f"KB data dir {self._data_dir} does not exist; create it "
                f"first (deploy step: mkdir -p {self._data_dir}). "
                f"Refusing to build a silent empty index.")
        self._lock_path = self._data_dir / _LOCK_NAME
        self._seq_path = self._data_dir / _SEQ_NAME
        self._cold = PyterrierBackend(self._data_dir / "cold")
        self._hot = HotLayerBackend(capacity=hot_capacity)
        # The hot layer holds not-yet-settled recent docs; its own side-store
        # so a restart doesn't lose them (the E2E round-trip gap).
        self._hot.set_persist_path(self._data_dir / "hot_corpus.pkl")
        self._backend = TwoLayerBackend(
            self._hot, self._cold, settle_threshold=settle_threshold)
        self._backend.load()
        self._hot.load()
        self._seq = self._read_seq()
        # P0 governance: physically-isolated pending store + audit trail.
        # Staging lives under <data_dir>/staging (deploy-created, 770); it is
        # NEVER part of the two-layer backend, so default search() cannot
        # surface a pending doc (reviewer acceptance #1/#3).
        self._staging = StagingStore(self._data_dir / _STAGING_DIR_NAME)
        # Audit log: devops-owned jsonl path (default next to the log dir,
        # but the service accepts the real deploy path via audit_path).
        self._audit_path = Path(audit_path) if audit_path else None
        # Reinforce write-gate sidecar (explicit-tool-call-only counter that
        # seeds the read-side recency/reinforce ranking axis; the read-side
        # auto-prefetch NEVER writes this, so it can't bootstrap its own
        # favorites — owner/reviewer rule ③).
        self._reinforce = ReinforceStore(self._data_dir / _REINFORCE_NAME)
        # Co-occurrence graph sidecar (the RRF third axis, auto-built on the
        # write path under the EX lock; read-time fallback when corrupt).
        self._graph = GraphStore(self._data_dir / _GRAPH_NAME)
        # TR-5b (owner 2026-10-14): the write-side CONFLICT probe's own
        # budget pool (supersedes detection, independent of the judge /
        # distill pools — a probe budget exhaustion must never starve the
        # judge/distill faces and vice versa). Per-process counter, not
        # persisted: the probe is a fail-soft enhancement, a restart
        # resetting it is harmless (worst case a re-probe on next ingest).
        self._conflict_budgets: Dict[str, int] = {}

    # ---- reinforce write-gate (explicit tool call only) ------------------

    def reference_counts(self) -> Dict[str, int]:
        """SH-lock: {docid: reinforcement-count} from the sidecar log.

        The read-side RRF recency axis (kb_rerank._recency_axis) consumes
        this; a missing/corrupt log yields an empty map (the axis then
        scores purely on recency, no crash)."""
        fd = self._open_lock(exclusive=False)
        try:
            return self._reinforce.load()
        finally:
            self._close_lock(fd)

    def reinforce(self, docid: str, reason: str = "explicit_search",
                  anchors: Optional[List[str]] = None,
                  followup_text: str = "") -> dict:
        """Write-gate reinforcement: record that the model EXPLICITLY
        searched for ``docid`` (knowledge_base_search tool) and then anchored
        it in a later turn.

        This is the ONLY path that bumps a doc's reference count. The
        read-side auto-prefetch calls ``search()`` but NOT this, so a doc
        the prefetch keeps surfacing does not reinforce itself (the
        reviewer's cold-start-poisoning guard).

        ``followup_text`` (optional): when given, the anchor-hit check
        (_anchor_hit) must PASS for the doc before the event is recorded —
        a pure-string match on the doc's anchor_points[] / title (no LLM, no
        embedding). When omitted the caller is asserting the anchor already
        hit (the plugin handler passes it after checking the tool return's
        doc against the next turn's text).
        """
        docid = str(docid or "").strip()
        if not docid:
            return {"status": "error", "error": "docid required"}
        # Anchor gate: if followup text is supplied, require a pure-string
        # anchor hit on the doc's anchors/title before recording.
        if followup_text:
            # Resolve the doc's anchors/title by doc_id (the external join key),
            # NOT by searching the doc_id string as a query (that 0-hits).
            doc = self._find_doc_by_id(docid)
            if doc is None or not _anchor_hit(doc, followup_text):
                return {"status": "ok", "reinforced": False,
                        "reason": "anchor_miss"}
        # Record (EX lock so the 7-gateway JSONL append is atomic; the
        # reinforce log is a separate sidecar, not the corpus write-epoch,
        # so no _seq bump — readers reload the log fresh under SH).
        fd = self._open_lock(exclusive=True)
        try:
            self._reinforce.append(docid, reason=reason, anchors=anchors)
        finally:
            self._close_lock(fd)
        return {"status": "ok", "reinforced": True, "docid": docid}

    def _find_doc_by_id(self, docid: str) -> Optional[Dict[str, Any]]:
        """SH-lock: locate a doc by its external doc_id across the two-layer
        corpus. Returns a plain dict {docid, title, content, anchors/anchor_points,
        ...} or None (the doc is not in the confirmed corpus). Used by the
        reinforce anchor gate to resolve the doc's anchors/title for a
        pure-string match — NOT by searching the id string (which 0-hits)."""
        fd = self._open_lock(exclusive=False)
        try:
            # Both layers expose docs_snapshot(); scan for the doc_id. The
            # hot layer holds recent docs, the cold the settled corpus — a
            # doc is in exactly one at a time, so first hit wins.
            for layer in (self._hot, self._cold):
                for d in layer.docs_snapshot():
                    if str(d.get("docid")) == docid:
                        out = {k: v for k, v in d.items()}
                        out["anchor_points"] = (d.get("anchor_points")
                                                or d.get("anchors") or [])
                        return out
            return None
        finally:
            self._close_lock(fd)

    def _locate_doc_by_id_with_layer(self, docid: str) -> tuple:
        """Single-pass locator returning (doc, layer) or (None, None).

        The O(n) companion to ``_find_doc_by_id`` for the ``get()`` read
        face: one hot->cold sweep yields BOTH the doc and the layer it was
        found in, so callers never need a second docs_snapshot() scan to
        re-derive the layer label. Same SH-lock + first-hit-wins invariant
        as ``_find_doc_by_id`` (a doc lives in exactly one layer at a time).
        ``_find_doc_by_id`` itself is left untouched (reinforce() depends
        on its doc-only return shape).
        """
        fd = self._open_lock(exclusive=False)
        try:
            for layer_name, layer in (("hot", self._hot),
                                      ("cold", self._cold)):
                for d in layer.docs_snapshot():
                    if str(d.get("docid")) == docid:
                        out = {k: v for k, v in d.items()}
                        out["anchor_points"] = (d.get("anchor_points")
                                                or d.get("anchors") or [])
                        return out, layer_name
            return None, None
        finally:
            self._close_lock(fd)

    # ---- audit -------------------------------------------------------------

    def _audit(self, **fields) -> None:
        """Append one JSONL record to the audit trail (ingest/confirm/search).

        Best-effort: a missing/undriven audit path is a no-op (the layer
        stays Hermes-free; the plugin adapter passes the real path via
        audit_path or configure_audit())."""
        if self._audit_path is None:
            return
        import time
        rec = {"ts": time.time(), **fields}
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError as e:  # audit must never block the main write path
            logger.warning("audit write failed: %s", e)

    def configure_audit(self, path: Path) -> None:
        """Point the audit trail at a real file (adapter wiring, like
        configure_logging). Idempotent per-path."""
        self._audit_path = Path(path)

    # ---- ingest (write path) ---------------------------------------------

    def ingest(self, docs: list[dict]) -> dict:
        """Write new knowledge with P0 governance routing.

        Each doc carries optional governance fields:
          ``status``         — "pending" (default) or "confirmed".
          ``origin_kind``   — "user" | "auto" | "factory" | "agent"
                              (default "auto"; TR-1: "agent" is the
                              agent-experience-ledger partition).
          ``source_profile`` — owning profile (default None).
          TR-1 ledger fields (defaulted in ``_normalize_docs`` when absent,
          all travel through the metadata channel):
          ``topic`` (default "general"), ``task_type`` (optional),
          ``premise`` (optional), ``premise_status`` (default True, a valid
          search() filters key).

        Routing:
          - ``status=confirmed`` → length gate + near-dup self-check → hot/cold
            two-layer corpus (the searchable path). Rejected docs are logged,
            never silently dropped.
          - ``status=pending``   → staging store (physically isolated, NOT
            searchable by default). No length/dup gate — pending is a
            candidate, not yet authoritative; the user confirm is the gate.

        Returns a summary dict: {"ingested": [...], "rejected": [...],
        "staged": [...], "generation": int}.
        """
        norm = _normalize_docs(docs)
        if not norm:
            logger.info("ingest no-op (0 docs)")
            return {"ingested": [], "rejected": [], "staged": [],
                   "generation": self._backend.generation}

        confirmed: list[dict] = []
        rejected: list[dict] = []
        pending: list[dict] = []

        for d in norm:
            status = d.get("status", "pending")
            # Stamp governance fields so downstream layers can see them.
            # TR-1: origin_kind is a passthrough enum — any string is stored
            # as-is (including "agent"); the setdefault keeps the legacy
            # "auto" default for docs that omit it (byte-identical to the
            # pre-TR-1 behaviour when no origin_kind is supplied).
            d.setdefault("status", status)
            d.setdefault("origin_kind", "auto")
            if status == "confirmed":
                gate = self._governance_gate(d)
                if gate is None:
                    confirmed.append(d)
                else:
                    rejected.append({**gate, "doc_id": d.get("docid")})
            else:  # pending (or any non-confirmed) -> staging
                pending.append(d)

        fd = self._open_lock(exclusive=True)
        try:
            self._rebase()
            # Cross-process staging rebase: merge on-disk rows from other
            # writers into this process's staging memory BEFORE writing, so
            # a later writer cannot clobber an earlier writer's pending rows
            # (the same last-writer-wins hole the hot/cold rebase closes, now
            # applied to the physically-isolated staging store).
            staging_disk = self._read_staging_rows()
            mem_staging = self._staging.docs_snapshot()
            if self._consume_external_replace_marker():
                # Out-of-band sweep/curator commit: the on-disk rows are the
                # authoritative pending set — adopt them WHOLESALE (no union,
                # which would resurrect quarantined docids) before adding.
                self._staging.apply_external_replace(staging_disk)
                mem_staging = staging_disk
            merged_staging = _merge_docs(staging_disk, mem_staging)
            if merged_staging != mem_staging:
                self._staging.replace(merged_staging,
                                      max(self._staging.generation, 0))
            # confirmed -> two-layer corpus (the hot path)
            if confirmed:
                self._backend.update(confirmed)
                # Deterministic first-seen recency ts (cross-process-stable;
                # NOT a per-doc created_at field). Recorded under the EX
                # lock already held here; only NEW docids are written.
                self._reinforce.record_ingest_ts(
                    [d.get("docid") for d in confirmed], time.time())
                # TR-5b conflict probe: before the just-confirmed docs can
                # silently coexist with an older confirmed entry they
                # logically replace, probe the existing corpus for a
                # semantically-adjacent doc and, on a supersedes verdict,
                # stamp superseded_by IN PLACE on the older row. Runs AFTER
                # _backend.update so the new doc is searchable; fail-soft:
                # any probe hiccup (budget exhausted / no LLM runtime /
                # search failure) leaves the older row untouched and the
                # new doc lands anyway — a missed supersession is always
                # the safe direction, never a wrongly suppressed live doc.
                self._probe_conflicts(confirmed)
            # pending -> staging store (physically isolated, independent gen)
            for d in pending:
                self._staging.add(d)
            self._flush()
            self._staging.persist()
            # build_graph write-side (owner 面向未来 ruling): the co-occurrence
            # graph is a system-maintained retrieval index — rebuilt
            # automatically on ingest, NOT by the caller. The FULL confirmed
            # corpus (hot + cold) is the build input; rebuild_from_corpus
            # truncates it to corpus[:_GRAPH_DOCS_MAX] (reviewer ③ "截断": the
            # graph is still built over a bounded prefix, NOT skipped; skip is
            # reserved for the fail-soft corrupt/disk-fail path). Runs inside
            # the EX critical section (reviewer ①), so 7-gateway concurrent
            # ingests never clobber each other. A write-disk failure is
            # swallowed inside GraphStore (fail-soft: last-good kept, the
            # read-time fallback stays active on the next search).
            confirmed_corpus = (self._hot.docs_snapshot()
                                + self._cold.docs_snapshot())
            self._graph.rebuild_from_corpus(confirmed_corpus)
        finally:
            self._close_lock(fd)

        # Audit: one record per doc outcome (devops 660 jsonl).
        for d in confirmed:
            self._audit(action="ingest_confirmed", doc_id=d.get("docid"),
                        source_profile=d.get("source_profile"),
                        origin_kind=d.get("origin_kind", "auto"),
                        status="confirmed")
        for r in rejected:
            self._audit(action="ingest_rejected", doc_id=r.get("doc_id"),
                        reason=r.get("reason"),
                        source_profile=r.get("source_profile"),
                        near_dup_score=r.get("near_dup_score"))
        for d in pending:
            self._audit(action="ingest_staged", doc_id=d.get("docid"),
                        source_profile=d.get("source_profile"),
                        origin_kind=d.get("origin_kind", "auto"),
                        status="pending")

        logger.info("ingest docs=%d confirmed=%d rejected=%d staged=%d "
                    "hot_gen=%d", len(norm), len(confirmed), len(rejected),
                    len(pending), self._hot.generation)
        return {"ingested": [d.get("docid") for d in confirmed],
                "rejected": rejected,
                "staged": [d.get("docid") for d in pending],
                "generation": self._hot.generation}

    def _governance_gate(self, doc: dict, *, skip_near_dup: bool = False,
                        ) -> Optional[dict]:
        """P0 gates for a CONFIRMED doc. Returns None (accept) or a reject
        record {reason, ...}. Runs under the caller's EX lock.

        1. Length gate — title/content bounds (reviewer thresholds).
        2. Near-dup self-check — BM25 the candidate against the current
           confirmed corpus (no model/API); top-1 normalized score decides:
             >= _DEDUP_BLOCK  -> reject (near-exact dup, merge candidate)
             _DEDUP_AUDIT..   -> accept but flag suspected (audit only)
             <  _DEDUP_AUDIT  -> accept clean
        ``skip_near_dup`` (TR-10): an explicit caller-owned write (e.g.
        share_to_global's global copy of an already-gated doc) may skip
        the self-check; the length gate always runs.
        """
        title = str(doc.get("title", ""))
        content = str(doc.get("content", ""))
        if not (_TITLE_MIN <= len(title) <= _TITLE_MAX):
            return {"reason": "length_gate", "field": "title",
                    "got": len(title),
                    "bounds": [_TITLE_MIN, _TITLE_MAX],
                    "source_profile": doc.get("source_profile")}
        if not (_CONTENT_MIN <= len(content) <= _CONTENT_MAX):
            return {"reason": "length_gate", "field": "content",
                    "got": len(content),
                    "bounds": [_CONTENT_MIN, _CONTENT_MAX],
                    "source_profile": doc.get("source_profile")}
        if skip_near_dup:
            return None
        # Near-dup: token-containment self-check (path A, scale-stable).
        # The candidate's OWN token set is the "query"; for each confirmed
        # doc, containment = |cand ∩ doc| / |cand| (cand = the shorter/new
        # side). This is independent of min-max BM25 normalization (which
        # flattens to 1.0 in a small corpus and mis-kills same-template
        # docs), and it is number-aware (hot_layer.tokenize emits \d+ as
        # independent tokens, so 编号0 vs 编号1 differ -> containment <
        # the block threshold -> passes).
        import hot_layer as _hl
        q = f"{title} {content}"
        cand_tokens = set(_hl.tokenize(q))
        if cand_tokens:
            # Scan the confirmed corpus for the doc with the highest
            # containment against the candidate. (Not a BM25 search: that
            # path's min-max-normalized score is corpus-size dependent.)
            worst: tuple[float, str] = (0.0, "")
            for layer in (self._hot, self._cold):
                for d in layer.docs_snapshot():
                    if str(d.get("docid")) == doc.get("docid"):
                        continue  # never compare a doc against itself
                    doc_tokens = set(_hl.tokenize(
                        f"{d.get('title','')} {d.get('content','')}"))
                    if not doc_tokens:
                        continue
                    inter = len(cand_tokens & doc_tokens)
                    cont = inter / len(cand_tokens)  # containment (short/long)
                    if cont > worst[0]:
                        worst = (cont, str(d.get("docid")))
            if worst[0] >= _DEDUP_CONTAINMENT_BLOCK:
                return {"reason": "near_duplicate",
                        "near_dup_score": round(worst[0], 4),
                        "against_doc": worst[1],
                        "source_profile": doc.get("source_profile")}
            if worst[0] >= _DEDUP_CONTAINMENT_AUDIT:
                doc["_suspected_dup_of"] = worst[1]
        return None

    # ---- confirm (governance move: staging -> confirmed corpus) ----------

    def confirm(self, doc_id: str, source_profile: Optional[str] = None,
                confirmed_by: Optional[str] = None) -> dict:
        """Atomically move a PENDING doc from staging into the confirmed
        corpus. The code-side backstop for the room's "must user-confirm to
        ingest" rule (the clarify step itself is a model-side policy gate;
        this is the hard gate that enforces profile ownership + audit).

        Validation chain (devops 2):
          1. doc exists in staging
          2. its status == "pending"
          3. source_profile match: if the caller's source_profile is given it
             must equal the stored row's; a mismatch -> reject + audit
             ``error: profile_mismatch``. (A None caller is allowed only when
             the stored row has no source_profile — legacy/global row.)
        On success: EX-lock, remove the row from staging, append it to the
        two-layer corpus as status=confirmed, audit the move, return the doc.
        """
        # Pre-check (read side) under SH so a cross-process concurrent confirm
        # of the same doc doesn't both pass the find() then double-append.
        fd = self._open_lock(exclusive=True)
        try:
            self._rebase()
            # Cross-process staging rebase (same as ingest): merge on-disk
            # pending rows from other writers before we inspect/move, so a
            # doc another process staged is visible to this confirm.
            staging_disk = self._read_staging_rows()
            mem_staging = self._staging.docs_snapshot()
            if self._consume_external_replace_marker():
                # Out-of-band sweep/curator commit: on-disk rows are the
                # authoritative pending set — adopt wholesale, no union.
                self._staging.apply_external_replace(staging_disk)
                mem_staging = staging_disk
            merged_staging = _merge_docs(staging_disk, mem_staging)
            if merged_staging != mem_staging:
                self._staging.replace(merged_staging,
                                      max(self._staging.generation, 0))
            row = self._staging.find(doc_id)
            if row is None:
                self._audit(action="confirm", doc_id=doc_id,
                            source_profile=source_profile,
                            confirmed_by=confirmed_by,
                            error="not_found")
                return {"status": "error", "error": "not_found",
                        "doc_id": doc_id}
            if str(row.get("status", "pending")) != "pending":
                self._audit(action="confirm", doc_id=doc_id,
                            source_profile=source_profile,
                            confirmed_by=confirmed_by,
                            error="not_pending")
                return {"status": "error", "error": "not_pending",
                        "doc_id": doc_id}
            stored_profile = row.get("source_profile")
            if source_profile is not None and stored_profile is not None and \
                    source_profile != stored_profile:
                # Cross-profile confirm: this profile cannot confirm another
                # profile's pending doc.
                self._audit(action="confirm", doc_id=doc_id,
                            source_profile=source_profile,
                            confirmed_by=confirmed_by,
                            error="profile_mismatch",
                            stored_profile=stored_profile)
                return {"status": "error", "error": "profile_mismatch",
                        "doc_id": doc_id}
            # Atomic move-out + move-in under the same EX lock.
            moved = self._staging.remove(doc_id)  # returns the removed row
            if moved is None:
                # find() passed but remove() missed: a concurrent EX writer
                # already consumed this docid between the two calls. Treat
                # as a no-op success (the doc is already confirmed somewhere
                # else) rather than a double-confirm error.
                self._audit(action="confirm", doc_id=doc_id,
                            source_profile=stored_profile,
                            confirmed_by=confirmed_by,
                            note="already_moved_concurrent")
                return {"status": "success", "doc_id": doc_id,
                        "confirmed_by": confirmed_by,
                        "source_profile": stored_profile,
                        "note": "already_moved_concurrent"}
            moved["status"] = "confirmed"
            moved["confirmed_by"] = confirmed_by
            moved.setdefault("origin_kind", "auto")
            # B2 metadata-aware dedup (PRD v2-final, revised.2 option b):
            # BEFORE the corpus move, when the moving row carries a
            # gov_tier, scan the CONFIRMED corpus (hot + cold snapshots)
            # for a doc with the SAME title+content but a DIFFERENT
            # gov_tier (N1 scale posture: compare the gov_tier value
            # only — the skip_near_dup "don't re-run the full
            # token-containment scan" idea applied to the tier axis;
            # title+content identity is exact-match, no tokenize).
            # On a hit the tier is in CONFLICT: the doc is NOT moved
            # into the confirmed corpus (no auto-confirm of a tier the
            # owner already ruled differently). Instead the row is
            # DOWNGRADED to a plain PENDING candidate — gov_tier is
            # cleared (so an owner re-confirm does not re-collide) and
            # the _suspected_tier_conflict mark rides along — re-added
            # to staging, audited (tier_conflict), and returned with
            # note=tier_conflict. The caller (write-side hook) re-fires
            # the session's clarify channel for the moved-out doc id so
            # the footnote surfaces it to the owner; the owner confirms
            # through the EXISTING knowledge_base_confirm face (a
            # non-agent confirmed_by is the explicit re-confirm — no new
            # tool surface, no confirm signature change; skipping leaves
            # the row PENDING for audit). A stale-staging TTL sweep
            # (note 4) is the terminal backstop against unbounded
            # growth of blocked rows.
            tier = moved.get("gov_tier")
            if tier:
                conflict_doc = None
                for layer in (self._hot, self._cold):
                    for d in layer.docs_snapshot():
                        if (str(d.get("title") or "") == str(moved.get("title") or "")
                                and str(d.get("content") or "") == str(moved.get("content") or "")
                                and d.get("gov_tier") not in (None, tier)):
                            conflict_doc = d
                            break
                    if conflict_doc:
                        break
                if conflict_doc is not None:
                    # Downgrade: re-stage the row as a plain PENDING
                    # candidate (status back to pending, gov_tier
                    # cleared, conflict mark set), keep it in the
                    # staging store (the move-out already happened via
                    # remove(); re-adding restores it), audit + return
                    # the conflict note to the caller.
                    downgraded = {k: v for k, v in moved.items()
                                  if k not in ("status", "confirmed_by",
                                                "gov_tier")}
                    downgraded["status"] = "pending"
                    downgraded["gov_tier"] = None
                    downgraded["_suspected_tier_conflict"] = True
                    downgraded["_tier_conflict_against"] = (
                        str(conflict_doc.get("docid")))
                    self._staging.add(downgraded)
                    self._staging.persist()
                    self._flush()
                    self._audit(action="confirm", doc_id=doc_id,
                                source_profile=stored_profile,
                                confirmed_by=confirmed_by,
                                error="tier_conflict",
                                against_doc=str(conflict_doc.get("docid")),
                                gov_tier=tier)
                    logger.info("confirm doc_id=%s tier_conflict "
                                "against=%s; downgraded to plain "
                                "PENDING + clarify", doc_id,
                                conflict_doc.get("docid"))
                    return {"status": "success", "doc_id": doc_id,
                            "confirmed_by": confirmed_by,
                            "source_profile": stored_profile,
                            "note": "tier_conflict"}
            self._backend.update([moved])
            self._flush()
            self._staging.persist()
            self._audit(action="confirm", doc_id=doc_id,
                        source_profile=stored_profile,
                        confirmed_by=confirmed_by,
                        status="confirmed")
            logger.info("confirm doc_id=%s profile=%s by=%s",
                        doc_id, stored_profile, confirmed_by)
            # TR-5b: same conflict-probe hook as ingest() — a confirmed
            # doc that logically replaces an older confirmed entry is
            # stamped superseded_by on that older row (fail-soft, own
            # budget pool). Runs inside the EX lock already held here.
            self._probe_conflicts([moved])
            return {"status": "success", "doc_id": doc_id,
                    "confirmed_by": confirmed_by,
                    "source_profile": stored_profile}
        finally:
            self._close_lock(fd)

    # ---- share_to_global (TR-10: explicit escalation to the global layer)

    def share_to_global(self, doc_id: str, source_profile: Optional[str] = None) -> dict:
        """Escalate a CONFIRMED agent-experience doc to the global shared
        layer (PRD v2 TR-10; owner red line: agent experience does NOT
        walk unannounced across profiles).

        Semantics: the doc is COPIED into the confirmed corpus as a second
        row with ``source_profile=None`` (the global layer that TR-8's
        ``source_profile=None`` opt-in full search hits). The ORIGINAL
        row (``source_profile=<owning profile>``) is left intact — the
        escalation is additive, not a move. The copy is a NEW docid
        (uuid5 input differs on source_profile, the same mechanism as the
        Phase 1 origin_kind-into-uuid5 precedent). After one successful
        escalation the corpus physically holds TWO rows for this doc —
        the original profile row (1) plus the source_profile=None global
        copy (1); a repeat escalation of the same doc is IDEMPOTENT:
        it detects the existing copy, writes no new physical row, and
        still records an audit line (already_shared=True) + returns
        success (a recorded no-op, not a silent skip).

        Hard gate (the red line, enforced by two conditions, mirroring
        confirm()'s source_profile matching — no bypass invented):
          (a) source_profile match: the doc row's stored source_profile
              must equal the caller's ``source_profile`` (a None caller
              is only admitted when the stored row is also None — the
              legacy/global-row rule confirm() uses verbatim);
          (b) origin_kind == "agent": only agent-produced experience may
              be escalated (user/auto entries stay in their profile).
        Any failure (doc not in the confirmed corpus, either condition
        unmet) is REJECTED and written to the EXISTING _audit trail as
        action=share_to_global with an error code:
          - not_found            doc_id not in the confirmed two-layer corpus
          - profile_mismatch     condition (a) failed (cross-profile doc)
          - not_agent_origin    condition (b) failed (origin_kind != agent)
          - premise_falsified   doc's premise_status is the falsified state
                                 (False; "needs_verification"/True/"true"
                                 still admissible — a three-state allow,
                                 which is deliberately BROADER than the
                                 KPI-3 decay gate's two-state allow: the
                                 decay gate (kb_ingest_hook
                                 _agent_in_decay_gate) retires
                                 needs_verification from the
                                 auto-injection pool, while share_to_global
                                 still admits it. The difference is
                                 intentional: an explicit, audited
                                 escalation action is less restrictive
                                 than passive auto-injection —
                                 needs_verification (neither falsified
                                 nor verified) is not barred from being
                                 made visible to other profiles, since
                                 escalation only writes a global copy
                                 that TR-8 opt-in search can find; it
                                 does NOT auto-inject the doc into any
                                 profile's context. A falsified premise
                                 is the one state that is barred in
                                 BOTH places.) — a falsified
                                 premise must not walk into the global
                                 pool
        On success the global copy is written under the EX lock via the
        existing ingest() write path with status=confirmed and
        source_profile=None (length gate re-runs — the copy is a
        re-stamped row; the near-dup self-check is skipped for THIS
        copy: it is the explicit, caller-authorized twin of a doc that
        already passed the gate at its original ingest, and blocking it
        would make the PRD's one-line escalation self-defeating).
        Audit: action=share_to_global, doc_id + global_doc_id + status.

        Returns {status: success|error, ...} — the same result shape
        confirm() uses, so the plugin handler stays one-liner.
        """
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            return {"status": "error", "error": "not_found", "doc_id": ""}
        fd = self._open_lock(exclusive=True)
        try:
            self._rebase()
            current = None
            for layer in (self._hot, self._cold):
                for d in layer.docs_snapshot():
                    if str(d.get("docid")) == doc_id:
                        current = dict(d)
                        break
                if current is not None:
                    break
            if current is None:
                self._audit(action="share_to_global", doc_id=doc_id,
                             source_profile=source_profile,
                             error="not_found")
                return {"status": "error", "error": "not_found",
                        "doc_id": doc_id}
            stored_profile = current.get("source_profile")
            if (source_profile is not None and stored_profile is not None
                    and source_profile != stored_profile):
                # Red line, condition (a): a profile cannot escalate
                # another profile's experience to the global layer.
                self._audit(action="share_to_global", doc_id=doc_id,
                             source_profile=source_profile,
                             error="profile_mismatch",
                             stored_profile=stored_profile)
                return {"status": "error", "error": "profile_mismatch",
                        "doc_id": doc_id}
            origin_kind = current.get("origin_kind", "auto")
            if origin_kind != "agent":
                # Red line, condition (b): only agent-produced experience
                # may walk to the global layer; user/auto entries are
                # profile-private by governance (the PRD's "agent 经验
                # 不走私" half: NOT every profile doc may be escalated,
                # only the agent's own).
                self._audit(action="share_to_global", doc_id=doc_id,
                             source_profile=source_profile,
                             error="not_agent_origin",
                             origin_kind=origin_kind)
                return {"status": "error", "error": "not_agent_origin",
                        "doc_id": doc_id}
            # Semantic gate: a doc whose premise has been FALSIFIED
            # (premise_status is the falsified state — the literal False
            # value set_premise_status stores, OR its "falsified" string
            # spelling; NOT "needs_verification" and NOT True/"true",
            # both of which stay admissible here — a three-state allow,
            # deliberately BROADER than the KPI-3 decay gate
            # (kb_ingest_hook _agent_in_decay_gate), which is two-state
            # (True/"true" only) and retires needs_verification from the
            # auto-injection pool. The difference is intentional: an
            # explicit, audited escalation action is less restrictive
            # than passive auto-injection — needs_verification is not
            # barred from being made visible to other profiles, since
            # escalation only writes a global copy that TR-8 opt-in
            # search can find and does NOT auto-inject the doc into any
            # profile's context; a falsified premise is the one state
            # that is barred in both places) must not walk to the global
            # layer: a falsified premise should come out of the shared
            # pool, not into it. Only the explicitly falsified state
            # blocks; re-verification (falsified -> needs_verification/
            # true via set_premise_status) lifts the block.
            _ps = current.get("premise_status", True)
            if _ps is False or _premise_status_canonical(_ps) == "falsified":
                self._audit(action="share_to_global", doc_id=doc_id,
                             source_profile=source_profile,
                             error="premise_falsified")
                return {"status": "error", "error": "premise_falsified",
                        "doc_id": doc_id}
            # Idempotency: the global copy has a deterministic docid
            # (uuid5 over the copy dict, source_profile=None). After a
            # successful first escalation the corpus physically holds
            # TWO rows for this doc — the original profile row (1) plus
            # the source_profile=None global copy (1). A second call
            # would re-run update + record_ingest_ts + _flush on the
            # same row (write+persist churn that grows unboundedly
            # with repeat calls even though the row count stays flat);
            # instead, detect the existing copy and skip all three. The
            # idempotent hit is still AUDITED (already_shared=True) and
            # returns success — it is a recorded no-op, not a silent
            # skip. No new physical row is added by a repeat call.
            copy = {k: v for k, v in current.items()
                    if k not in ("docid", "source_profile")}
            copy["source_profile"] = None
            copy["status"] = "confirmed"
            norm = _normalize_docs([copy])
            global_doc_id = str(norm[0].get("docid"))
            existing_global = [
                d for layer in (self._hot, self._cold)
                for d in layer.docs_snapshot()
                if str(d.get("docid")) == global_doc_id
                and d.get("source_profile") is None]
            if existing_global:
                self._audit(action="share_to_global", doc_id=doc_id,
                             source_profile=stored_profile,
                             global_doc_id=global_doc_id,
                             already_shared=True,
                             status="shared")
                logger.info(
                    "share_to_global doc_id=%s profile=%s -> %s "
                    "(already shared, idempotent no-op write)",
                    doc_id, stored_profile, global_doc_id)
                return {"status": "success", "doc_id": doc_id,
                        "global_doc_id": global_doc_id,
                        "source_profile": stored_profile,
                        "already_shared": True}
            gate = self._governance_gate(norm[0], skip_near_dup=True)
            if gate is not None:
                # Only the length gate can fire here (near-dup is skipped):
                # the original row already passed it. Defensive path.
                self._audit(action="share_to_global", doc_id=doc_id,
                             source_profile=source_profile,
                             error="gate_rejected",
                             reason=gate.get("reason"))
                return {"status": "error", "error": "gate_rejected",
                        "doc_id": doc_id, "reason": gate.get("reason")}
            self._backend.update(norm)
            self._reinforce.record_ingest_ts(
                [str(norm[0].get("docid"))], time.time())
            self._flush()
            self._audit(action="share_to_global", doc_id=doc_id,
                         source_profile=stored_profile,
                         global_doc_id=norm[0].get("docid"),
                         status="shared")
            logger.info("share_to_global doc_id=%s profile=%s -> %s",
                        doc_id, stored_profile, norm[0].get("docid"))
            return {"status": "success", "doc_id": doc_id,
                    "global_doc_id": norm[0].get("docid"),
                    "source_profile": stored_profile}
        finally:
            self._close_lock(fd)

    # ---- premise falsification state bit (TR-5) -------------------------

    def set_premise_status(self, doc_id: str, new_status,
                           premise_text: Optional[str] = None) -> dict:
        """Migrate a CONFIRMED corpus doc's ``premise_status`` (TR-5).

        A confirmed experience doc carries a premise (``premise`` text +
        ``premise_status`` bit: True = held, "needs_verification", or
        "falsified"). When the premise is falsified the doc is DOWNGRADED,
        never deleted: it stays in the confirmed corpus under the same
        docid and becomes unreachable by the default filters until an
        explicit migration restores it.

        Legal transitions (anything else is rejected + audited):
          True -> needs_verification -> falsified
          falsified -> needs_verification   (re-check)
          falsified -> True                 (manually re-verified)

        INVARIANT (the invariant that makes this feasible, per the data
        engineer's feasibility call): premise_status is part of the doc
        dict, and the doc dict is the input to ``_deterministic_docid``
        (uuid5 over the sorted dict). The status therefore can ONLY be
        changed IN PLACE on the same docid — it must be re-persisted as
        the SAME row (the docid key is what _merge_docs keys on, so
        replace() overwrites the row in place). NEVER recompute
        ``_deterministic_docid`` on the mutated dict — that would mint a
        NEW docid and leave a stale row behind (the double-row bug).

        Also: only the two keys ``premise_status`` and (optionally)
        ``premise`` are ever touched. No other key of the doc is
        re-stamped or re-extracted (re-running ``_normalize_docs`` would
        setdefault the other card-2 fields and change the uuid5 input).

        Skeleton: EX lock + rebase + in-place replace + flush + seq bump.
        No staging persist (the doc is confirmed corpus, not staging).
        """
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            return {"status": "error", "error": "doc_id required"}
        target = _premise_status_canonical(new_status)
        if target is None:
            return {"status": "error", "error": "invalid_status",
                    "doc_id": doc_id,
                    "value": new_status,
                    "allowed": [True, "needs_verification", "falsified"]}
        fd = self._open_lock(exclusive=True)
        try:
            self._rebase()
            # Locate the confirmed doc in-memory (both layers).
            current = None
            for layer in (self._hot, self._cold):
                for d in layer.docs_snapshot():
                    if str(d.get("docid")) == doc_id:
                        current = d
                        break
                if current is not None:
                    break
            if current is None:
                self._audit(action="premise_status_change",
                            doc_id=doc_id,
                            to=target,
                            premise_text=premise_text,
                            error="not_found")
                return {"status": "error", "error": "not_found",
                        "doc_id": doc_id}
            # If not explicitly supplied, keep the doc's current premise
            # text (the caller may want to migrate the bit without touching
            # the premise prose). A None premise_text means "leave as-is".
            from_status = _premise_status_canonical(
                current.get("premise_status", True)) or "true"
            allowed = _PREMISE_STATUS_MIGRATIONS.get(from_status, ())
            if target not in allowed:
                self._audit(action="premise_status_change_rejected",
                            doc_id=doc_id,
                            from_status=from_status,
                            to=target,
                            premise_text=premise_text,
                            reason=f"illegal transition {from_status} -> {target}; "
                                   f"allowed from '{from_status}': {list(allowed)}")
                return {"status": "error",
                        "error": "illegal_transition",
                        "doc_id": doc_id,
                        "from_status": from_status,
                        "to": target,
                        "allowed": list(allowed)}
            # Build the mutated doc in place: same docid, same every-other-key.
            # Only premise_status is re-stamped (and premise if supplied).
            # This MUST NOT re-run _normalize_docs (it would setdefault the
            # other card-2 fields and shift the uuid5 input).
            modified = dict(current)  # shallow copy: keys + values unchanged
            # The canonical value goes into the dict: bool True stays bool
            # True (the default the setdefault stamps), strings stay their
            # canonical strings — so the uuid5 input is deterministic
            # regardless of which valid spelling the caller supplied.
            modified["premise_status"] = target
            if premise_text is not None:
                modified["premise"] = premise_text
            # In-place row replace keyed by docid (memory-wins): each layer's
            # docs list is swapped to the mutated dict (same docid) — never
            # appended. replace() is a wholesale list swap, so a doc in the
            # OTHER layer simply keeps its (unmatched) row. _merge_docs (via
            # _rebase on the next cycle) keys on docid, so the on-disk row
            # is overwritten in place and no double-row can form.
            hot_docs = self._hot.docs_snapshot()
            if any(str(d.get("docid")) == doc_id for d in hot_docs):
                hot_docs = [modified if str(d.get("docid")) == doc_id else d
                            for d in hot_docs]
                self._hot.replace(hot_docs, self._hot.generation)
            cold_docs = self._cold.docs_snapshot()
            if any(str(d.get("docid")) == doc_id for d in cold_docs):
                cold_docs = [modified if str(d.get("docid")) == doc_id else d
                             for d in cold_docs]
                self._cold.replace(cold_docs, self._cold.generation)
            self._flush()
            self._audit(action="premise_status_change",
                        doc_id=doc_id,
                        from_status=from_status,
                        to=target,
                        premise_text=premise_text)
            logger.info("set_premise_status doc_id=%s %s -> %s premise=%r",
                        doc_id, from_status, target,
                        premise_text if premise_text is not None else "<unchanged>")
            return {"status": "success", "doc_id": doc_id,
                    "from_status": from_status,
                    "to": target}
        finally:
            self._close_lock(fd)

    # ---- superseded_by knowledge-override bit (owner 2026-10-14) --------
    # A CONFIRMED doc that a NEWER confirmed doc logically supersedes is
    # STAMPED IN PLACE with ``superseded_by=<newer docid>`` (never a new
    # row, the set_premise_status double-row bug class) and DROPPED from
    # the auto-inject pool + the default tool-layer search, but stays
    # physically present in the corpus: an explicit knowledge_base_get
    # by docid still reaches it (留盘可查). reinstate() clears the stamp
    # and puts the doc back in the pools — reversible, unlike a premise
    # falsification which needs its own migration path. The bit is
    # ORTHOGONAL to premise_status: a doc may carry both (its premise
    # broke AND a better doc superseded it); each is tracked/lifted
    # independently.

    def _find_confirmed_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """EX-lock caller only: the confirmed-corpus row with this docid,
        or None. No nested lock — the caller already holds EX."""
        for layer in (self._hot, self._cold):
            for d in layer.docs_snapshot():
                if str(d.get("docid")) == doc_id:
                    return dict(d)
        return None

    def _replace_confirmed_in_place(self, modified: Dict[str, Any]) -> None:
        """EX-lock caller only: swap the docid-matched row in each layer
        for ``modified`` (same docid, every-other-key unchanged) without
        appending — the set_premise_status in-place pattern that keeps
        the docid set (and hence the corpus row count) byte-stable."""
        doc_id = str(modified["docid"])
        hot_docs = self._hot.docs_snapshot()
        if any(str(d.get("docid")) == doc_id for d in hot_docs):
            hot_docs = [modified if str(d.get("docid")) == doc_id else d
                        for d in hot_docs]
            self._hot.replace(hot_docs, self._hot.generation)
        cold_docs = self._cold.docs_snapshot()
        if any(str(d.get("docid")) == doc_id for d in cold_docs):
            cold_docs = [modified if str(d.get("docid")) == doc_id else d
                         for d in cold_docs]
            self._cold.replace(cold_docs, self._cold.generation)

    def supersede(self, doc_id: str, superseded_by: str,
                  confirmed_by: Optional[str] = None,
                  reason: Optional[str] = None) -> dict:
        """Stamp a CONFIRMED doc as superseded by a newer confirmed doc.

        The older row is mutated IN PLACE (``superseded_by=<newer docid>``
        + an audit stamp), never re-minted. Effects:
          - dropped from the auto-inject agent pool
            (kb_ingest_hook._agent_in_decay_gate) and from the default
            tool-layer search() result (the service-side drop, below);
          - still reachable via knowledge_base_get / explicit docid
            lookup (留盘可查);
          - orthogonal to premise_status: a falsified doc may be
            superseded too, and supersede() never touches that bit.
        ``superseded_by`` (the newer docid) is AUDITED but not existence-
        checked: the caller (the write-side conflict probe, or a manual
        invocation) is responsible for pointing at a real confirmed doc;
        a dangling pointer is a caller error, not a store error — the
        older row's drop-from-pool effect does not depend on the newer
        row existing.
        """
        fd = self._open_lock(exclusive=True)
        try:
            self._rebase()  # merge on-disk confirmed rows first (EX held)
            current = self._find_confirmed_doc(doc_id)
            if current is None:
                self._audit(action="supersede_rejected", doc_id=doc_id,
                            superseded_by=superseded_by,
                            confirmed_by=confirmed_by,
                            error="not_found")
                return {"status": "error", "error": "not_found",
                        "doc_id": doc_id}
            modified = dict(current)
            modified["superseded_by"] = superseded_by
            modified["superseded_at"] = time.time()
            if confirmed_by:
                modified["superseded_by_confirmed_by"] = confirmed_by
            if reason:
                modified["supersede_reason"] = reason
            self._replace_confirmed_in_place(modified)
            self._flush()
            self._audit(action="supersede", doc_id=doc_id,
                        superseded_by=superseded_by,
                        confirmed_by=confirmed_by, reason=reason)
            logger.info("supersede doc_id=%s superseded_by=%s by=%r",
                        doc_id, superseded_by, confirmed_by)
            return {"status": "success", "doc_id": doc_id,
                    "superseded_by": superseded_by}
        finally:
            self._close_lock(fd)

    def reinstate(self, doc_id: str,
                  confirmed_by: Optional[str] = None) -> dict:
        """Lift the superseded_by stamp: the doc re-enters the auto-inject
        + default search pools. The row itself is untouched otherwise
        (nothing new is appended, nothing removed). A doc that was never
        superseded is a recorded no-op success (idempotent)."""
        fd = self._open_lock(exclusive=True)
        try:
            self._rebase()  # merge on-disk confirmed rows first (EX held)
            current = self._find_confirmed_doc(doc_id)
            if current is None:
                self._audit(action="reinstate_rejected", doc_id=doc_id,
                            confirmed_by=confirmed_by,
                            error="not_found")
                return {"status": "error", "error": "not_found",
                        "doc_id": doc_id}
            if "superseded_by" not in current:
                # Idempotent: already active — recorded no-op, still a
                # success (a caller must not treat a clean doc as an
                # error when re-asserting it active).
                self._audit(action="reinstate", doc_id=doc_id,
                             confirmed_by=confirmed_by, already_active=True)
                return {"status": "success", "doc_id": doc_id,
                        "already_active": True}
            modified = dict(current)
            del modified["superseded_by"]
            for key in ("superseded_at", "superseded_by_confirmed_by",
                        "supersede_reason"):
                modified.pop(key, None)
            self._replace_confirmed_in_place(modified)
            self._flush()
            self._audit(action="reinstate", doc_id=doc_id,
                        confirmed_by=confirmed_by)
            logger.info("reinstate doc_id=%s by=%r", doc_id, confirmed_by)
            return {"status": "success", "doc_id": doc_id}
        finally:
            self._close_lock(fd)

    def set_doc_kind(self, doc_id: str,
                     new_doc_kind: Optional[str]) -> dict:
        """Re-label a CONFIRMED doc's ``doc_kind`` IN PLACE (same docid,
        every-other-key untouched) — the one supported path for moving
        a legacy unlabeled doc into a new doc_kind door without re-
        minting its docid (re-minting would trip the B2 containment
        near-dup gate; this one doesn't, because B2 only runs on
        ingest()'s fresh-content path, never on an in-place replace).

        ``new_doc_kind`` must be a ``kb_ingest_rules.DOC_KIND_VOCAB``
        value or None (None = clear the label, back to "unlabeled").
        Out-of-vocab values are rejected + audited, NOT silently
        accepted (the same clarity semantics as topic_search's
        invalid_topic — the caller must fix its input rather than read
        a misleading result).

        INVARIANT: the doc's ``docid`` key is NEVER recomputed here —
        the row is mutated in place and re-persisted under the same
        docid (the set_premise_status in-place pattern,
        _replace_confirmed_in_place, docid-keyed list swap, never
        append). No B2 gate, no _normalize_docs re-run (which would
        setdefault the other keys and shift the uuid5 input).

        Skeleton: EX lock + rebase + in-place replace + flush + audit.
        No staging persist (the doc is confirmed corpus, not staging).
        """
        from kb_ingest_rules import DOC_KIND_VOCAB
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            return {"status": "error", "error": "doc_id required"}
        # Validate the target value up front (cheap, no lock needed).
        if new_doc_kind is not None and new_doc_kind not in DOC_KIND_VOCAB:
            self._audit(action="doc_kind_change_rejected", doc_id=doc_id,
                        to=new_doc_kind,
                        error="invalid_doc_kind",
                        reason=(f"doc_kind must be one of "
                                f"{list(DOC_KIND_VOCAB)} or null"))
            return {"status": "error", "error": "invalid_doc_kind",
                    "doc_id": doc_id,
                    "value": new_doc_kind,
                    "allowed": list(DOC_KIND_VOCAB)}
        fd = self._open_lock(exclusive=True)
        try:
            self._rebase()
            current = self._find_confirmed_doc(doc_id)
            if current is None:
                self._audit(action="doc_kind_change_rejected",
                            doc_id=doc_id, to=new_doc_kind,
                            error="not_found")
                return {"status": "error", "error": "not_found",
                        "doc_id": doc_id}
            from_kind = current.get("doc_kind")
            modified = dict(current)
            if new_doc_kind is None:
                # Clear the label: drop the key entirely (absent == None
                # under the B1 None-filter, so this is byte-identical to
                # leaving a None-valued key in place — but physically
                # absent is the canonical "unlabeled" spelling).
                modified.pop("doc_kind", None)
            else:
                modified["doc_kind"] = new_doc_kind
            self._replace_confirmed_in_place(modified)
            self._flush()
            self._audit(action="doc_kind_change", doc_id=doc_id,
                        from_kind=from_kind, to=new_doc_kind)
            logger.info("set_doc_kind doc_id=%s %r -> %r",
                        doc_id, from_kind, new_doc_kind)
            return {"status": "success", "doc_id": doc_id,
                    "from_kind": from_kind,
                    "to": new_doc_kind}
        finally:
            self._close_lock(fd)

    def _is_superseded(self, doc: Dict[str, Any]) -> bool:
        """True iff the confirmed row carries an active superseded_by
        stamp (the drop-from-pool predicate, shared by the search()
        default path and the auto-inject gate)."""
        return str(doc.get("superseded_by") or "") != ""

    # ---- TR-5b write-side conflict probe ---------------------------------
    # Supersedes detection: when a doc lands CONFIRMED, probe the existing
    # confirmed corpus for an entry it logically replaces and, on a
    # "supersedes" verdict, stamp superseded_by on the older row IN PLACE
    # (no new docid, no deletion — 留盘可查 via knowledge_base_get).
    #
    # Budget model: an INDEPENDENT per-service-instance pool
    # (self._conflict_budgets), SEPARATE from the judge / distill pools —
    # exhausting the conflict pool never starves judge or distill and
    # vice-versa. Per-process only (not persisted): the probe is a
    # fail-soft enhancement, a restart resetting the counter is harmless.
    #
    # Fail-open everywhere: no LLM runtime available / over-budget / probe
    # search hiccup -> the older doc is left untouched and the new doc has
    # already landed via the normal confirmed path above. A missed
    # supersession is always the safe direction.

    _CONFLICT_BUDGET_DEFAULT = 3
    _CONFLICT_PROBE_TOP_K = 3

    def _conflict_probe_enabled(self) -> bool:
        """The probe is on iff the host LLM surface is importable; a
        standalone / no-runtime process simply never runs it (fail-soft
        floor: the structural field/stamp machinery still works via the
        explicit supersede() tool face, the auto-probe is the only thing
        that degrades to inert)."""
        try:
            from kb_ingest_hook import _HOST_CALL_LLM  # noqa: F401
            return _HOST_CALL_LLM is not None
        except Exception:
            return False

    def _probe_conflicts(self, new_confirmed: List[Dict[str, Any]]) -> None:
        """EX-lock caller only (ingest()/confirm()): probe for supersedes
        conflicts between the just-confirmed docs and the rest of the
        confirmed corpus. Takes NO lock of its own - the caller's EX
        section is reused end to end (reviewer-3 deadlock fix: a second
        flock descriptor on this thread would self-deadlock, and
        search()/supersede() each open their own descriptor).

        For each new doc, run the lock-free search body (_search_body)
        over the EXISTING corpus (excluding the new doc itself - the
        two-layer backend already includes it after _backend.update),
        capped at _CONFLICT_PROBE_TOP_K. A hit with score >= the block
        threshold is a candidate older doc; the LLM (budget-gated,
        conflict pool) decides "supersedes" vs "coexist". Supersedes ->
        stamp the older row in place via _supersede_inside_lock().
        Coexist / no verdict / over-budget / no runtime -> no-op.
        """
        if not new_confirmed or not self._conflict_probe_enabled():
            return
        probe_key = "conflict_probe"
        limit = self._CONFLICT_BUDGET_DEFAULT
        used = self._conflict_budgets.get(probe_key, 0)
        if used >= limit:
            return  # over-cap: no new LLM call, older rows untouched
        for new_doc in new_confirmed:
            query = f"{new_doc.get('title', '')} {new_doc.get('content', '')}"
            new_docid = str(new_doc.get("docid"))
            try:
                # Lock-free search: the caller (ingest()/confirm()) already
                # holds the EX flock; search() would open a SECOND descriptor
                # (flock is per-descriptor, non-reentrant) and block on the
                # caller's own EX lock -> self-deadlock. _search_body skips
                # the lock entirely and reads the in-memory layers directly.
                hits = self._search_body(query,
                                         top_k=self._CONFLICT_PROBE_TOP_K,
                                         rerank=False)
            except Exception:
                continue  # probe hiccup: fail-open, older doc untouched
            for hit in hits.get("data", {}).get("results", []):
                old_docid = str(hit.get("doc_id", ""))
                if not old_docid or old_docid == new_docid:
                    continue
                if hit.get("score", 0.0) < _DEDUP_AUDIT:
                    continue  # not semantically adjacent enough to probe
                verdict = self._llm_conflict_verdict(
                    new_doc, hit, used, limit)
                if verdict is None:
                    continue  # over-budget / no LLM / call failed: coexist
                if verdict.get("supersedes"):
                    # Inline stamp (NOT self.supersede(), which opens its
                    # own EX lock -> deadlock on the caller's held lock):
                    # same in-place mutation + audit as supersede(), but
                    # executed inside the EX section the caller already holds.
                    self._supersede_inside_lock(old_docid, new_docid,
                                                 reason=(new_doc.get("title") or ""))
                    used += 1
                    self._conflict_budgets[probe_key] = used
                    if used >= limit:
                        return
                    break  # one supersedes verdict per new doc is enough

    def _supersede_inside_lock(self, doc_id: str, superseded_by: str,
                                reason: Optional[str] = None) -> bool:
        """Lock-free body of supersede() for callers already inside an EX
        section (the conflict probe). Same in-place mutation + audit trail
        as supersede(), but takes no lock of its own (the caller's EX
        flock is held). Returns True if the older row was found + stamped."""
        current = self._find_confirmed_doc(doc_id)
        if current is None:
            self._audit(action="supersede_rejected", doc_id=doc_id,
                        superseded_by=superseded_by,
                        confirmed_by="conflict_probe",
                        error="not_found")
            return False
        modified = dict(current)
        modified["superseded_by"] = superseded_by
        modified["superseded_at"] = time.time()
        modified["superseded_by_confirmed_by"] = "conflict_probe"
        if reason:
            modified["supersede_reason"] = reason
        self._replace_confirmed_in_place(modified)
        self._flush()
        self._audit(action="supersede", doc_id=doc_id,
                    superseded_by=superseded_by,
                    confirmed_by="conflict_probe", reason=reason)
        logger.info("supersede doc_id=%s superseded_by=%s by=%r",
                    doc_id, superseded_by, "conflict_probe")
        return True

    def _llm_conflict_verdict(self, new_doc: Dict[str, Any],
                             old_hit: Dict[str, Any],
                             used: int, limit: int) -> Optional[dict]:
        """Budget-gated LLM supersedes/coexist verdict for one new-vs-old
        pair. Returns None when the pool is exhausted or the LLM call
        failed (fail-open: no verdict applied), else the verdict dict
        {"supersedes": bool, ...}. The verdict call is counted against
        the pool ONLY after a real successful call (a failed call must
        not burn the slot, same fail-soft rule as the judge face)."""
        if used >= limit:
            return None
        try:
            from kb_ingest_hook import _HOST_CALL_LLM, _norm_main_runtime
        except Exception:
            return None
        if _HOST_CALL_LLM is None:
            return None
        rt = None
        try:
            rt = _norm_main_runtime(None)
        except Exception:
            rt = None
        if rt is None:
            return None  # no runtime: coexist stands, slot NOT consumed
        prompt = (
            "You are deciding whether a newly confirmed knowledge entry "
            "SUPERSEDES (replaces, makes obsolete) an older confirmed "
            "entry, or whether they COEXIST (both stay active).\n\n"
            "New entry title: %s\nNew entry content: %s\n\n"
            "Older entry title: %s\nOlder entry content: %s\n\n"
            "Reply with JSON: {\"supersedes\": true|false, "
            "\"reason\": \"<one line>\"}\n"
            "Supersedes ONLY when the new entry explicitly corrects / "
            "replaces / updates the older one (e.g. a newer version, a "
            "revised procedure, a newer tool/environment state). "
            "Coexist when they cover different conditions or are simply "
            "independent." % (
                new_doc.get("title", ""), new_doc.get("content", "")[:500],
                old_hit.get("title", ""), old_hit.get("content", "")[:500]))
        try:
            resp = _HOST_CALL_LLM(
                main_runtime=rt,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=120, timeout=15)
            import json as _json
            content = (resp or {}).get("content", "") or ""
            # Reviewer issue #3 fix: parse the raw content directly first;
            # the old lstrip("{")+rsplit("}",1)+"}" recipe truncated any
            # verdict whose reason text legitimately contained a "}"
            # (e.g. 'v2 replaces {v1}'), silently dropping a supersedes
            # verdict into the fail-soft None branch.
            text = content.strip()
            verdict = None
            if text:
                try:
                    verdict = _json.loads(text)
                except Exception:
                    verdict = None
            if verdict is None:
                # Last resort: strip markdown code fences (```lang ...
                # ```) and keep the FULL JSON body intact - never a
                # truncated brace reassembly.
                stripped = text
                if "```" in stripped:
                    # keep only the JSON body between the fences
                    import re as _re
                    m = _re.search(r"```[a-zA-Z]*\n?(.*?)\n?```",
                                   stripped, _re.DOTALL)
                    if m:
                        stripped = m.group(1).strip()
                try:
                    verdict = _json.loads(stripped) if stripped else None
                except Exception:
                    verdict = None
            if verdict is None:
                return None  # unparseable: fail-open, slot not consumed
            # Count the slot only after a real successful verdict.
            return verdict
        except Exception:
            return None  # call failed: fail-open, slot not consumed

    # ---- get (deterministic single-doc read path) ------------------------
    # Backs the knowledge_base_get tool face. Fills the observability gap
    # left by search(): verifying "is this known doc reachable" must NOT
    # require a BM25 word-surface gamble. Returns {found, doc_id, layer,
    # doc|staging_row} without scanning the whole corpus text index.

    def refresh(self) -> dict:
        """EX-lock rebase of BOTH the confirmed two-layer corpus and the
        pending staging row store into this process's memory, WITHOUT
        flushing either to disk (no persist, no seq bump, no graph
        rebuild — a pure read-side re-sync, distinct from persist()
        which rebase + flush).

        Closes the 4be2560 read-path visibility gap: after the
        in-place SH→EX seq-upgrade rebase was removed from search()
        (the _probe_conflicts self-deadlock fix), a long-lived
        READ-ONLY instance (a gateway whose KB face only ever calls
        search()/get() and never ingest()/confirm()) stays stale —
        it never sees another process's new docs until one of its OWN
        write-path calls (ingest/confirm/persist) happens to trigger
        _rebase(). refresh() gives such a read-only face an explicit,
        on-demand re-sync door: call it (e.g. at the top of a
        pre_llm_call auto-inject cycle, or before a deliberate
        read-heavy batch) and the next search()/get() in the same
        process sees every other gateway's writes.

        Takes EX (not SH): _rebase() mutates this instance's hot/cold
        + staging in-memory state, which is a write to the instance's
        OWN memory even though it reads DISK state written by others —
        the EX lock is what makes two concurrent refresh() calls on
        the same data dir serialize cleanly (the same argument as
        ingest()/confirm() taking EX around their own _rebase() call).
        COST (callers must not assume refresh() is cheap/SH-gradable):
        while refresh() holds EX, every other gateway's concurrent
        ingest()/confirm() on the same data dir blocks on the flock
        for the whole rebase duration. A read-only gateway calling
        refresh() on a tight loop will therefore starve peer writers
        the same way any EX-holding operation does — size refresh
        cadence accordingly (a pre_llm_call cycle calling it once per
        LLM turn is fine; calling it in a hot inner loop is not).
        It does NOT bump self._seq or write any sidecar file, so a
        read-only process calling refresh() in a tight loop still
        never becomes a writer of the seq marker (the read-side
        reinforce write-gate "only explicit tool calls count" rule is
        untouched — refresh() is not a reinforce event, not an
        ingest_ts event, not a graph rebuild trigger).

        Out-of-band sweep marker side effect (reviewer nit, 2026-10-18
        data-plane pass): refresh() runs the SAME staging-row
        cross-process merge block ingest()/confirm() run, including
        _consume_external_replace_marker() — if a tools/kb_stage_v2_sweep
        --commit sweep left its marker file behind before this refresh()
        ran, refresh() CONSUMES (unlinks) that marker AND applies the
        wholesale-adopt exactly as ingest()/confirm() would (no sweep
        signal is lost; the adopted set is now this instance's in-memory
        staging truth, so a later ingest()/confirm() simply finds no
        marker and unions over an already-correct set — no quarantined
        docid resurrection). If callers care about "who" consumed the
        marker (e.g. for audit), this is the one observable side effect
        refresh() has beyond re-syncing memory: it can be the first
        process to notice + apply a sweep commit, not the last one to
        write.
        """
        fd = self._open_lock(exclusive=True)
        try:
            # confirmed two-layer corpus rebase (hot/cold), same call
            # ingest()/confirm()/persist() all make first.
            self._rebase()
            # staging row store rebase: the exact same cross-process
            # merge block ingest()/confirm() run (union-merge by
            # docid, external-replace marker consumption, wholesale
            # adopt on a sweep commit) — reused verbatim so a
            # read-only process's staging view stays consistent with
            # what a writer's rebase would produce, without ever
            # becoming a writer itself.
            staging_disk = self._read_staging_rows()
            mem_staging = self._staging.docs_snapshot()
            if self._consume_external_replace_marker():
                self._staging.apply_external_replace(staging_disk)
                mem_staging = staging_disk
            merged_staging = _merge_docs(staging_disk, mem_staging)
            if merged_staging != mem_staging:
                self._staging.replace(merged_staging,
                                      max(self._staging.generation, 0))
            out = {
                "status": "success",
                "hot_gen": self._hot.generation,
                "cold_gen": self._cold.generation,
                "staging_gen": self._staging.generation,
                "note": "rebase only — no flush/persist, no seq bump",
            }
            logger.debug("refresh rebase-only hot_gen=%d staging_gen=%d",
                         self._hot.generation, self._staging.generation)
            return out
        finally:
            self._close_lock(fd)

    def get(self, doc_id: str) -> dict:
        """Deterministically fetch one doc by its external doc_id.

        Reads BOTH faces, in priority order:
          1. confirmed two-layer corpus (hot then cold) via
             ``_find_doc_by_id`` — the searchable path.
          2. staging store (pending rows) — a not-yet-confirmed doc is
             still addressable by id, returned with its ``status`` so the
             caller can tell "reachable pending" from "gone".

        Returns ``{"found": True, "layer": "hot"|"cold"|"staging",
        "doc_id": ..., "doc": {...full metadata incl. content...}}`` on a
        hit, or ``{"found": False, "doc_id": ...}`` when the id is not in
        either face. No BM25, no word-surface dependency: this is the
        direct path the search() tool cannot offer.
        """
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            return {"found": False, "doc_id": ""}
        # Single pass over hot->cold returns BOTH the doc and its layer, so
        # get() is O(n) not O(2n) (no second docs_snapshot() scan to re-derive
        # the layer label). _find_doc_by_id is left untouched for reinforce().
        found, layer = self._locate_doc_by_id_with_layer(doc_id)
        if found is not None:
            return {"found": True, "layer": layer,
                    "doc_id": doc_id, "doc": found}
        # Confirmed corpus miss: check the physically-isolated staging face.
        # Use find_active() — NOT find() — so a re-keyed stale_staging audit
        # copy (note 4 TTL sweep) is not surfaced as a reachable pending row;
        # that matches the active clarify/confirm face exactly (owner 2026-10-14
        # review nit: a get() that hands back a stale_staging row would let an
        # agent mistake an audit-only row for a confirmable pending one).
        row = self._staging.find_active(doc_id)
        if row is not None:
            return {"found": True, "layer": "staging",
                    "doc_id": doc_id, "doc": dict(row)}
        return {"found": False, "doc_id": doc_id}

    # ---- search (single-shot read path) ----------------------------------

    def search(self, query: str, top_k: int = 5,
               filters: Optional[dict] = None,
               min_score: float = 0.0,
               rerank: bool = True,
               topic_scores: Optional[dict] = None) -> dict:
        fd = self._open_lock(exclusive=False)
        try:
            return self._search_body(query, top_k=top_k, rerank=rerank,
                                     filters=filters, min_score=min_score,
                                     topic_scores=topic_scores)
        finally:
            self._close_lock(fd)

    def _search_body(self, query: str, top_k: int = 5,
                    filters: Optional[dict] = None,
                    min_score: float = 0.0,
                    rerank: bool = True,
                    topic_scores: Optional[dict] = None) -> dict:
        """The lock-free search pipeline (reviewer-3 deadlock fix):
        callers must hold their own lock - search() takes the SH flock,
        _probe_conflicts reuses the caller's EX section so it never
        re-enters a flock on this thread. Return the PRD 13.2
        structured payload.

        ``filters`` (optional) is a dict of metadata keys that a result must
        match exactly (e.g. ``{"doc_type": "guide"}`). TR-4 sentinel (PRD
        v2-final note 2, semantic unification with the B1 docid face): the
        value ``"__unset__"`` means "key physically absent OR key present
        with None" (the unified "None ≡ key absent" ruling — the same
        semantics the B1 None-filter gives the docid face). Every OTHER
        filter value keeps the exact-equality ``metadata.get(k) == v``
        semantics, including a literal None value (mechanical equality;
        the PRD mandates the sentinel string, not None, for "unset").

        ``min_score`` (optional, default 0.0) is a lower-bound threshold on
        the relevance score; results with ``score < min_score`` are excluded
        before the top_k truncation.  A default of 0.0 preserves the existing
        behaviour exactly (all scores are >= 0), so existing callers see no
        change unless they explicitly pass a positive value.

        B1: when filters are active the fetch is UNCAPPED — the layers return
        every term-overlap hit, the filter is applied, then top_k truncates.
        Capping the fetch (the old top_k*2 over-fetch) silently starves the
        filter: if the top candidates are mostly non-matching, fewer than
        top_k filtered results come back.

        Read-side precision (owner "向 filemem 借鉴" batch ②③): when
        ``rerank`` is True (default), the two-layer BM25 result is fused
        with the recency/reinforce axis via RRF (kb_rerank.rrf_two_axis_rank)
        before the min_score / filter / top_k truncation. The recency axis
        is seeded by the explicit-tool-call reinforce store, so a doc the
        model has genuinely cited decays slower and a fresh-but-never-cited
        doc does not auto-dominate. min_score / filters / top_k semantics
        are unchanged — RRF only re-orders the candidate pool it is given.

        ``topic_scores`` (optional, TR-6 fourth-axis topic-mass) is a
        {doc_id: topic-mass} map that ONLY the explicit tool-layer
        knowledge_base_search call passes (the pre_llm_call auto-inject
        layer never does — aligned with the ReinforceStore write-gate's
        "only explicit tool calls count" rule). It is inert unless the
        candidate pool also carries >= 20 origin_kind=agent entries
        (kb_rerank._TOPIC_ACTIVATE_AGENTS absolute-count gate, isomorphic
        to the graph axis activation gate); below it, the payload is
        byte-identical to the three-axis result (KPI-6).
        """
        from kb_rerank import rrf_two_axis_rank
        # Lock-free by contract: the caller already holds a flock on the
        # lock sidecar (search() SH; _probe_conflicts reuses the caller's
        # EX section). Re-entry on a second descriptor would self-deadlock
        # (flock is per-descriptor, non-reentrant). A writer that persisted
        # since this instance last rebased is picked up on the NEXT caller
        # (the SH/EX lock holder runs _rebase()); in-probe reads see the
        # just-confirmed doc via the in-memory backend, which is all the
        # probe's own new docs ever need.
        cap = None if filters else top_k
        results = self._backend.search(query, top_k=cap)
        logger.debug("search query=%r top_k=%d filters=%r min_score=%r",
                     query, top_k, filters, min_score)
        # RRF two-axis fusion (BM25 + recency/reinforce), BEFORE the
        # min_score / filter / top_k gates so the fused order drives the
        # truncation. reference_counts come from the explicit-tool-call
        # write-gate sidecar (the read-side prefetch never writes it).
        # Lock-free read (reviewer-3 deadlock fix): reference_counts() opens
        # its own SH flock descriptor - calling it from here (already inside
        # a held lock) would self-deadlock on a second descriptor. The
        # underlying sidecar load is a plain file read, safe to call
        # directly under the caller's held lock.
        ref_counts = self._reinforce.load()
        # Deterministic first-seen recency ts (cross-process-stable; NOT a
        # per-doc created_at field that would differ across gateways). The
        # recency axis reads this alongside reference_count.
        ingest_ts = self._reinforce.ingest_ts()
        if rerank and results:
            # SearchResult is a frozen dataclass -> build fresh instances
            # with the injected reference_count + ingest_ts, not in-place
            # mutation.
            def _with_ref(r):
                meta = {**(r.metadata or {})}
                rc = ref_counts.get(r.doc_id, 0)
                if rc:
                    meta["reference_count"] = rc
                ts = ingest_ts.get(r.doc_id)
                if ts is not None:
                    meta["ingest_ts"] = ts
                return SearchResult(r.doc_id, r.score, r.title, r.content,
                                    meta, r.raw)
            results = [_with_ref(r) for r in results]
            # BM25 axis = the candidate pool. When filters are active the
            # fetch was UNCAPPED (cap=None above) so `results` is the full
            # term-overlap set — fusing that whole set is correct and lets the
            # filter step (below) see every matching doc, so we must NOT
            # truncate the pool here (that was the filter-starvation bug the
            # B1 lesson closed). With no filter, over-fetch top_k*2 gives the
            # fusion a pool bigger than the truncation it will apply.
            if filters:
                pool = results
            else:
                over = top_k * 2 if top_k is not None else None
                pool = results[:over] if over else results
            pool_cands = [{"doc_id": r.doc_id, "title": r.title,
                           "content": r.content, "metadata": r.metadata}
                          for r in pool]
            # Under a FILTER the RRF result set must be UNBOUNDED (top_k=None
            # pool -> fuse everything -> the filter step below sees every
            # matching doc, preserving the B1 "filter not starved" contract).
            # Without a filter, cap to top_k so the payload truncation is
            # still top_k.
            fused_topk = None if filters else top_k
            # Third axis (build_graph, forward-looking): build the
            # co-occurrence graph over the SAME candidate pool that the RRF
            # fusion sees (pool_cands), so the graph signal is scoped to
            # THIS result set (a hub doc is one that co-occurs with the
            # other docs competing right now, not the whole corpus). This
            # is INERT on a <2-doc corpus: build_graph returns {} when the
            # pool has <2 docs / no co-occurrence, so g_scores is None and
            # the fusion degrades exactly to the two-axis result (the
            # owner's "开发面向未来" ruling - build it now, it activates as
            # docs accumulate, no data gate).
            from kb_rerank import build_graph, graph_axis_score
            # PERSISTED graph first (the write-side build); fall back to the
            # read-time compute path when graph.json is absent/corrupt
            # (fail-soft — never block search).
            graph = self._graph.load()
            if graph is None:
                # Fail-soft fallback: compute the graph over the CURRENT
                # candidate pool (the old read-time behaviour). This is the
                # inert-by-default path on a tiny KB (<2 docs -> empty graph
                # -> 0 weight).
                graph = build_graph(pool_cands)
            # O(n²) hub-sum truncation (reviewer ①, shipped in this batch or
            # else the persisted full-corpus graph fed into an UNBOUNDED
            # filter pool makes the hub-sum O(pool²) and the upgrade
            # regresses): when the pool exceeds _GRAPH_DOCS_MAX, cap the
            # graph axis to the top _GRAPH_DOCS_MAX pool docs (the ones
            # that will actually surface) so the filter path stays bounded.
            graph_pool = (pool_cands[:_GRAPH_DOCS_MAX]
                          if len(pool_cands) > _GRAPH_DOCS_MAX else pool_cands)
            # Activation gate (reviewer ④): only feed the graph into the RRF
            # third axis when the CORPUS has >= _GRAPH_ACTIVATE_EDGES docs
            # that carry at least one graph edge — NOT when the graph itself
            # has that many edges (a frozen 500-doc graph with a 5000-doc
            # corpus must not activate; the gate judges live corpus
            # participation, not the stale edge count). Below the gate the
            # axis is inert (g_scores=None), so a tiny KB degrades exactly to
            # the two-axis result. The corpus for the gate is the full
            # confirmed two-layer snapshot (hot + cold).
            corpus_docs = (self._hot.docs_snapshot()
                           + self._cold.docs_snapshot())
            participate = GraphStore.corpus_graph_participation(corpus_docs,
                                                               graph)
            if participate >= _GRAPH_ACTIVATE_EDGES:
                g_scores = (graph_axis_score(graph, graph_pool)
                            if graph else None)
            else:
                g_scores = None
            fused = rrf_two_axis_rank([pool_cands], fused_topk,
                                      reference_counts=ref_counts,
                                      graph_scores=g_scores,
                                      topic_scores=topic_scores)
            orig = {r.doc_id: r for r in results}
            # Re-emit fused candidates as SearchResult (carry the original
            # layer metadata + the rrf_fused_score annotation; rrf only ADDS,
            # it never clobbers the two-layer source_layer / generation).
            results = [
                SearchResult(f["doc_id"], orig[f["doc_id"]].score,
                             orig[f["doc_id"]].title,
                             orig[f["doc_id"]].content,
                             {**orig[f["doc_id"]].metadata,
                              **(f.get("metadata") or {})},
                             orig[f["doc_id"]].raw)
                for f in fused if f.get("doc_id") in orig
            ]
        payload = []
        for r in results:
            if r.score < min_score:
                continue
            meta = r.metadata or {}
            # superseded_by drop (owner 2026-10-14): a confirmed doc that
            # a newer confirmed doc supersedes is DROPPED from the default
            # tool-layer result, like premise_status=falsified — but only
            # when the caller did NOT explicitly filter ON superseded_by
            # (an explicit filters={"superseded_by": ...} query is the
            # 留盘可查 door, not the default pool).
            if ("superseded_by" not in (filters or {})
                    and meta.get("superseded_by")):
                continue
            if filters and not all(
                    (k not in meta or meta.get(k) is None
                     if v == "__unset__"
                     else meta.get(k) == v)
                    for k, v in filters.items()):
                continue
            payload.append({
                "doc_id": r.doc_id,
                "title": r.title,
                "content": r.content,
                "score": round(r.score, 4),
                "metadata": r.metadata,
            })
            if len(payload) >= top_k:
                break
        return {
            "status": "success",
            "data": {
                "query": query,
                "results": payload,
                "total": len(payload),
            },
        }

    # ---- persistence ------------------------------------------------------

    def persist(self) -> Path:
        """EX-lock: rebase, flush both stores + seq marker. Returns the cold
        corpus path."""
        fd = self._open_lock(exclusive=True)
        try:
            self._rebase()
            self._flush()
        finally:
            self._close_lock(fd)
        return self._cold.store_path

    def close(self) -> None:
        self.persist()

    # ---- cross-process machinery ------------------------------------------

    def _open_lock(self, exclusive: bool) -> Optional[int]:
        if fcntl is None:
            return None
        fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        return fd

    def _close_lock(self, fd: Optional[int]) -> None:
        if fd is None:
            return
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def _rebase(self) -> None:
        """Merge the on-disk corpora into this process's memory so a write
        cycle never clobbers another process's docs (the last-writer-wins
        hole). Caller holds EX.

        A doc another process settled while this one still holds it in hot
        coexists in hot + cold here; the two-layer cross-layer docid dedup
        (contract #15) keeps results correct, and the next settle drops the
        hot copy."""
        hot_disk, hot_gen = self._read_store(self._hot.store_path)
        cold_disk, cold_gen = self._read_store(self._cold.store_path)
        mem_hot = self._hot.docs_snapshot()
        mem_cold = self._cold.docs_snapshot()
        merged_hot = _merge_docs(hot_disk, mem_hot)
        merged_cold = _merge_docs(cold_disk, mem_cold)
        if merged_hot != mem_hot or merged_cold != mem_cold:
            self._hot.replace(merged_hot, max(self._hot.generation, hot_gen))
            self._cold.replace(merged_cold, max(self._cold.generation, cold_gen))
        self._seq = self._read_seq()

    def _flush(self) -> None:
        """Under EX: both layer stores + the write-epoch marker."""
        self._hot.persist()
        self._cold.persist()
        seq = self._read_seq() + 1
        self._seq = seq
        tmp = self._seq_path.with_suffix(".seq.tmp")
        tmp.write_text(str(seq))
        tmp.replace(self._seq_path)

    def _read_store(self, path: Path) -> tuple[list, int]:
        if not path.exists():
            return [], 0
        with open(path, "rb") as f:
            data = pickle.load(f)  # corruption propagates loudly, never silent
        return data.get("docs", []), data.get("generation", 0)

    def _read_staging_rows(self) -> list:
        """Read the on-disk staging row store (JSONL) for a cross-process
        rebase. A missing file means no other process has staged anything
        yet (empty, not an error)."""
        rows_path = self._staging.rows_path
        if not rows_path.exists():
            return []
        out = []
        with open(rows_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _consume_external_replace_marker(self) -> bool:
        """One-shot out-of-band replace signal (owner 2026-10-15).

        tools/kb_stage_v2_sweep.py --commit writes staging_docs.jsonl +
        drops staging_index.pkl + drops this marker file, all atomically
        enough for the next in-process rebase to adopt the swept row set
        WHOLESALE (apply_external_replace) instead of union-merging it
        (a union-merge would resurrect every quarantined docid from this
        process's memory).  The marker is consumed (deleted) on first
        sight under the EX lock, so it fires exactly once per sweep.
        A missing/empty marker is a no-op (the normal rebase path).
        """
        marker = self._data_dir / _STAGING_DIR_NAME / _STAGING_EXTERNAL_REPLACE
        if not marker.exists():
            return False
        try:
            marker.unlink()
            return True
        except OSError:
            # Two concurrent EX writers racing for the same marker: the
            # loser just falls through to the normal union path — safe,
            # idempotent, the winner already applied the wholesale set.
            return False

    def _read_seq(self) -> int:
        if not self._seq_path.exists():
            return 0
        try:
            return int(self._seq_path.read_text().strip() or "0")
        except ValueError:
            # The marker is ours and atomically written; a non-int means
            # external tamper → treat as fresh (worst case one extra rebase,
            # never data loss).
            return 0


_REINFORCE_NAME = "reinforce.jsonl"  # one JSON line per reinforcement event
_INGEST_TS_NAME = "ingest_ts.jsonl"  # deterministic first-seen-ts sidecar


class ReinforceStore:
    """Explicit-tool-call-only reference counter (the "reinforce write-gate").

    The owner/reviewer rule: ONLY a doc that the model explicitly searched
    for (knowledge_base_search tool) AND then anchored in a later turn
    (one of its anchor_points or title tokens appears in the follow-up
    text) counts as a reference. Auto-prefetch hits do NOT reinforce
    (would let the read-side prefetch bootstrap its own favorites).

    Persistent: JSONL append at <data_dir>/reinforce.jsonl (one object per
    line: {docid, ts, reason, anchors}). The write-gate method
    (KbSearchService.reinforce) appends under the EX lock so 7-gateway
    processes all see every event; the reader (search) loads the event
    log under SH and computes per-doc counts.

    Also owns the deterministic ``ingest_ts`` sidecar (ingest_ts.jsonl):
    the FIRST-seen unix-ts per docid, written by ingest() under EX. This
    is the cross-process-stable recency source for the read-side RRF axis
    — NOT a per-doc created_at field (which would differ across the 7
    gateway processes for the same docid and break the rebase merge).
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # The ingest_ts sidecar sits next to the reinforce log.
        self._ingest_ts_path = self._path.with_name(_INGEST_TS_NAME)

    def append(self, docid: str, reason: str = "explicit_search",
               anchors: Optional[List[str]] = None) -> None:
        """Append one reinforcement event. Caller already holds the EX lock."""
        rec = {"docid": str(docid), "ts": time.time(), "reason": reason,
               "anchors": list(anchors or [])}
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load(self) -> Dict[str, int]:
        """Read the whole event log; return {docid: count} (0 if missing)."""
        if not self._path.exists():
            return {}
        counts: Dict[str, int] = {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn trailing line mid-write: skip it
                    docid = str(rec.get("docid", ""))
                    if docid:
                        counts[docid] = counts.get(docid, 0) + 1
        except OSError:
            pass  # unreadable log: recency axis just sees no signal (safe)
        return counts

    # ---- deterministic ingest_ts sidecar (cross-process recency source) ---

    def record_ingest_ts(self, docids: List[str], ts: float) -> None:
        """Append first-seen-ts rows for freshly-ingested docids. Caller
        holds the EX lock. Only docids NOT already in the sidecar are
        written, so a cross-process replay keeps the FIRST process's
        timestamp for a given docid (deterministic, no last-writer-wins
        recency flip)."""
        existing = self._known_ts_docids()
        with open(self._ingest_ts_path, "a", encoding="utf-8") as f:
            for docid in docids:
                docid = str(docid)
                if not docid or docid in existing:
                    continue
                f.write(json.dumps({"docid": docid, "ts": ts},
                                   ensure_ascii=False) + "\n")

    def _known_ts_docids(self) -> set:
        known: set = set()
        if not self._ingest_ts_path.exists():
            return known
        try:
            with open(self._ingest_ts_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    docid = str(rec.get("docid", ""))
                    if docid:
                        known.add(docid)
        except OSError:
            pass
        return known

    def ingest_ts(self) -> Dict[str, float]:
        """{docid: first-seen ts} for the read-side RRF recency axis.
        Caller holds the SH lock (the sidecar is a separate file, so the
        read is safe even while a writer appends to it)."""
        if not self._ingest_ts_path.exists():
            return {}
        out: Dict[str, float] = {}
        try:
            with open(self._ingest_ts_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    docid = str(rec.get("docid", ""))
                    ts = rec.get("ts")
                    if not docid or ts is None:
                        continue
                    # first-seen wins: keep the earliest ts per docid.
                    if docid not in out or float(ts) < out[docid]:
                        out[docid] = float(ts)
        except OSError:
            pass
        return out


_GRAPH_NAME = "graph.json"  # co-occurrence graph sidecar (shared, flock-guarded)
# Write-side co-occurrence graph limits (owner ruling 2026-10-11):
#  - _GRAPH_DOCS_MAX: the graph is built over at most this many docs
#    (corpus[:N]); beyond that the incremental build is SKIPPED and the
#    graph stays a frozen snapshot from the last build (reviewer ③:
#    "截断" not "skip" — the graph is still BUILT on corpus[:500], not
#    absent; skip is reserved for graph.json corrupt / write-disk fail).
#  - _GRAPH_ACTIVATE_EDGES: the read-side third axis only gets weight when
#    the corpus has >= this many docs that carry a graph edge (reviewer
#    ④: judge "corpus 有图边参与数", NOT "图里有几条边" — a frozen
#    500-doc graph with a 5000-doc corpus must NOT activate the axis,
#    because the edge count is stale data, not a live signal).
_GRAPH_DOCS_MAX = 500
_GRAPH_ACTIVATE_EDGES = 20


class GraphStore:
    """Persistent co-occurrence graph sidecar (the RRF third-axis source).

    The owner's "面向未来" ruling: the graph is BUILT AUTOMATICALLY on the
    write path (ingest), not by the caller — a retrieval index is
    system-maintained, like the BM25 inverted index, not something the
    model must remember to call. The graph is a single SHARED file
    (graph.json) guarded by the data-dir flock (same as hot_corpus.pkl),
    so 7-gateway concurrent ingests never clobber each other (reviewer
    ①: the edge computation happens INSIDE the EX critical section, not
    async/background).

    The graph is a symmetric adjacency map {doc_id: {other_id: co_score}},
    co_score = anchor_overlaps * _ANCHOR_CO_WEIGHT + token_overlaps *
    _TOKEN_CO_WEIGHT (reuses kb_rerank.build_graph / hot_layer.tokenize —
    one tokenisation shared with the BM25 axis, no divergent token sets).

    Truncation (reviewer ③, owner max_graph_docs=500): the graph is built
    over corpus[:_GRAPH_DOCS_MAX]. Above that the incremental build is
    SKIPPED (the on-disk graph becomes a frozen snapshot from the last
    build) — this is "截断", not "skip 图构建": the graph is still there,
    just not refreshed. Skip is reserved for the fail-soft path
    (graph.json corrupt / write-disk OSError), never for normal n>=500.

    Read-side activation (reviewer ④): search() only feeds the graph into
    the RRF third axis when the CORPUS has >= _GRAPH_ACTIVATE_EDGES docs
    that carry at least one graph edge — NOT when the graph itself has
    that many edges (a frozen 500-doc graph with 5000 docs must not
    activate; the activation gate judges the live corpus, not the stale
    edge count). Below the gate the axis is inert (weight 0), so a tiny
    KB degrades exactly to the two-axis result (zero regression, the
    forward-looking design).

    Fail-soft (the owner's standing rule, same as the recency axis): a
    corrupt/unreadable graph.json degrades to the read-time
    compute-fallback (kb_rerank.build_graph over the current pool) and
    NEVER blocks search.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ---- write path (caller holds the EX lock) ---------------------------

    def rebuild_from_corpus(self, corpus: List[dict]) -> None:
        """Build the graph over corpus[:_GRAPH_DOCS_MAX] and persist it.

        The corpus is the FULL confirmed two-layer snapshot (hot + cold)
        taken under the EX lock; only the first _GRAPH_DOCS_MAX docs are
        fed to build_graph (reviewer ③: "截断" — the graph is still built,
        just over a bounded prefix, not skipped). A write-disk failure is
        swallowed (fail-soft): the on-disk graph keeps its last-good
        content, search degrades to the read-time fallback, nothing blocks
        ingest. A corrupt prior file is simply overwritten on the next
        successful build (self-heal, no crash)."""
        from kb_rerank import build_graph
        head = corpus[:_GRAPH_DOCS_MAX]
        graph = build_graph(head)
        try:
            tmp = self._path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"max_docs": _GRAPH_DOCS_MAX, "graph": graph},
                          f, ensure_ascii=False)
            tmp.replace(self._path)
        except OSError as e:
            import logging
            logging.getLogger(__name__).warning(
                "graph.json write failed (%s); keeping last-good graph "
                "(read-time fallback active on next search)", e)

    # ---- read path (caller holds the SH lock) ---------------------------

    def load(self) -> Optional[Dict[str, Dict[str, float]]]:
        """The persisted graph, or None when missing/corrupt (the
        fail-soft signal: a None graph means the caller must fall back to
        the read-time compute path, never raise)."""
        if not self._path.exists():
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        graph = data.get("graph")
        if not isinstance(graph, dict):
            return None
        return graph

    # ---- activation gate (reviewer ④) -----------------------------------

    @staticmethod
    def corpus_graph_participation(corpus: List[dict],
                                   graph: Optional[Dict[str, Dict[str, float]]]) -> int:
        """The number of CORPUS docs that carry at least one graph edge.

        This is the activation-gate signal (reviewer ④): it judges how many
        of the LIVE corpus docs participate in the graph, NOT how many
        edges the graph holds (which would be stale once the corpus
        outgrows a frozen snapshot)."""
        if not graph:
            return 0
        corpus_ids = {str(d.get("docid", "")) for d in corpus}
        participated = 0
        for did in corpus_ids:
            if graph.get(did):
                participated += 1
        return participated


def _anchor_hit(doc: Dict[str, Any], followup_text: str) -> bool:
    """Pure-string anchor match (no LLM, no embedding — the reviewer's
    "别做全文相似度匹配，做锚点匹配" rule).

    A doc is "cited" when ANY of its anchor_points[] or title tokens
    (lower-cased, non-trivial, length > 2) appears verbatim in the follow-up
    text. Anchor points come from the distill verdict (or a manual
    metadata.anchors set); title tokens are a last-resort fallback so a
    doc with no distilled anchors still gets a (weaker) signal.
    """
    if not followup_text:
        return False
    text = followup_text.lower()
    anchors = (doc.get("anchors") or doc.get("anchor_points") or [])
    if isinstance(anchors, str):
        anchors = [anchors]
    anchors = [str(a).lower() for a in anchors if a]
    for a in anchors:
        a = a.strip()
        if len(a) > 2 and a in text:
            return True
    # Fallback: jieba word overlap on the title (the service reuses the
    # shared hot_layer tokeniser so index/query terms always match). A
    # follow-up rarely quotes a CJK title verbatim, but it shares its
    # words. Whole-title substring is the last-resort, tokeniser-failure
    # path. Still pure-string, zero LLM.
    title = str(doc.get("title") or "").strip().lower()
    if len(title) > 2:
        try:
            import hot_layer
            title_words = {w.lower() for w in hot_layer.tokenize(title)
                           if len(w) > 1}
            follow_words = {w.lower() for w in hot_layer.tokenize(followup_text)
                            if len(w) > 1}
            if title_words & follow_words:
                return True
        except Exception:
            if title in text:
                return True
    return False


# ---------------------------------------------------------------------------
# B4 independent anchor-point extraction (PRD v2-final TR-5 / revised.3
# N2): four regex classes run over the RAW content, INDEPENDENT of any
# LLM face — this is the anchor source that stays live when the LLM
# distill fails (the v1 gap: LLM-only anchors meant an LLM fail dropped
# every anchor and fail-open-archived a transcript).
#
#  Class                     pattern (case-insensitive, word-bounded)
#  commit hash              \b[0-9a-f]{7,40}\b  (7-40 hex chars)
#  abs file path            /abs POSIX or Windows drive-letter paths
#                            with at least one more segment
#  version number           semantic x.y[.z] (x.y also matches)
#  service+config key       word.word / word:word dotted/coloned keys
#                            (2+ segments, >= 3 chars total)
#
# extract_anchor_points() returns the deduped hit list in a fixed
# class order (commit, path, version, service-key); union_anchor_points()
# adds the LLM-output anchors AFTER the regex hits (independent regex
# anchors > LLM anchors; union, not replacement — "LLM fail 时锚点全丢→
# 该归档的没归档" is exactly what the union prevents).
# ---------------------------------------------------------------------------

_ANCHOR_RE = {
    "commit": re.compile(r"\b[0-9a-f]{7,40}\b"),
    "version": re.compile(r"\b\d+\.\d+(?:\.\d+)?\b"),
    "service_key": re.compile(
        r"\b[A-Za-z][A-Za-z0-9_-]*(?:[.:][A-Za-z0-9_-]+){1,}\b"),
}
# The POSIX/Windows abs-path pattern (built separately so the {7,40}
# hex-class regex above stays a clean commit-hash shape): a leading
# "/" or "C:\" followed by at least two path segments.
_ANCHOR_ABS_PATH_RE = re.compile(
    r"(?:/(?:[\w.-]+/)*[\w.-]+|[A-Za-z]:\\(?:[\w.-]+\\)*[\w.-]+)")


def extract_anchor_points(raw_text: str) -> List[str]:
    """B4 step ①: independent 4-class anchor extraction (zero LLM).

    Returns the de-duplicated list of anchor strings (class order:
    commit, abs_path, version, service_key; within a class, first-hit
    order). Empty for pure prose — that empty result is exactly the
    step ③ signal (LLM fail + no anchors = transcript drop, N2).
    """
    if not raw_text:
        return []
    out: List[str] = []
    seen: set = set()

    def _add(m):
        v = m.group(0)
        if v not in seen:
            seen.add(v)
            out.append(v)

    for m in _ANCHOR_RE["commit"].finditer(raw_text):
        _add(m)
    for m in _ANCHOR_ABS_PATH_RE.finditer(raw_text):
        _add(m)
    for m in _ANCHOR_RE["version"].finditer(raw_text):
        _add(m)
    for m in _ANCHOR_RE["service_key"].finditer(raw_text):
        _add(m)
    return out


def union_anchor_points(raw_text: str,
                        llm_anchors: Optional[List[str]]) -> List[str]:
    """anchor_points = independent regex hits ∪ LLM-output anchors.

    Regex hits come FIRST (they are the source of truth — LLM fail
    leaves only these); LLM-only extras are appended after, deduped
    case-insensitively. `llm_anchors` may be None (LLM never ran) —
    the union then equals the regex hit list.
    """
    base = list(extract_anchor_points(raw_text))
    seen = {a.lower() for a in base}
    for a in (llm_anchors or []):
        a = str(a or "").strip()
        if a and a.lower() not in seen:
            seen.add(a.lower())
            base.append(a)
    return base


def render(payload: dict) -> str:
    """Serialize the service payload to the JSON string a tool returns."""
    return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_path: str, level: int = logging.INFO) -> None:
    """Point the "knowledge-base" logger at a single log file (no rotation).

    Adapter wiring (the plugin / CLI calls this once at startup); the
    service layer only *emits* records, so it stays Hermes-free. Calling it
    more than once with the same path must NOT duplicate handlers — a
    7-gateway process re-entering the path would otherwise fan one event out
    to N identical lines.
    """
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("knowledge-base")
    lg.setLevel(level)
    # Idempotent: only add the file handler if we don't already own one for
    # this exact path (dedup by formatter marker + path).
    handler = logging.FileHandler(str(p), encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"))
    for h in list(getattr(lg, "handlers", [])):
        if isinstance(h, logging.FileHandler):
            try:
                if h.baseFilename == str(p):
                    lg.removeHandler(h)  # replace stale re-point, no dup
            except AttributeError:
                pass
    lg.addHandler(handler)


__all__ = ["KbSearchService", "render", "configure_logging",
           "extract_anchor_points", "union_anchor_points"]
