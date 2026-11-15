#!/usr/bin/env python3
"""knowledge-base-search Hermes plugin.

Registers the single-shot ``knowledge_base_search`` tool (PRD §13.2) plus a
companion ``knowledge_base_ingest`` write path, both backed by the
repo's KbSearchService (knowledge-base/kb_service.py). The plugin adds the
knowledge-base package dir to sys.path so the service imports resolve
regardless of where the gateway spawns the process.

Single-shot: one search() call returns a structured JSON payload with full
title + content + score + metadata, so an agent needs no follow-up fetch.
No model, no external API — local rank_bm25 only (PRD §1).
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path

# --- cross-platform HERMES_HOME normalizer -----------------------------------
# On Windows, a Linux-style absolute path (e.g. "/root/.hermes" from a
# migrated config) produces a *relative* WindowsPath anchored at the drive
# root path when passed verbatim. Detect and normalise so all downstream Path()
# calls resolve correctly regardless of how HERMES_HOME was seeded.
def _norm_hermes_home(raw: str) -> str:
    if sys.platform == "win32":
        # Windows: paths starting with "/" are Linux absolute → treat as relative
        if raw.startswith("/") and not raw.startswith("//"):
            return str(Path.home() / ".hermes")
    return raw

# --- resolve the knowledge-base package (repo asset, not a site package) ----
# The service lives in the project repo; make it importable from anywhere the
# gateway spawns this plugin. Env override wins for non-default deployments.
_hermes_home = _norm_hermes_home(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
_KB_DIR = os.environ.get(
    "KB_HOME",
    os.path.join(_hermes_home, "plugins", "knowledge-base-search"),
)
if _KB_DIR not in sys.path:
    sys.path.insert(0, _KB_DIR)

# --- plugin-local dir resolution (bootstrap for sibling / lazy imports) ---
# The plugin dir itself must be on sys.path so that kb_ingest_hook.py's
# bare sibling import (from kb_ingest_rules import ...) and kb_service's
# lazy function-level imports (from kb_ingest_hook import _current_profile)
# both resolve regardless of whether the host loads the plugin as a
# package (hermes_plugins.knowledge-base-search) or via a direct
# module-file load. No-op when the host already placed the plugin dir
# on sys.path. Rollback-safe: deleting this two-line block restores the
# prior host-only behaviour.
# CRITICAL: this block MUST run before any sibling import below — the host
# loads __init__.py via spec_from_file_location WITHOUT putting the plugin
# dir on sys.path, so `from handler_args import ...` raises
# "No module named 'handler_args'" if it executes before this bootstrap.
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from handler_args import _normalize_handler_args


# --- KB deps resolve from the shared plugin venv (owner 10-08 directive) ------
# The KB service is loaded IN-PROCESS by each gateway (no child process), so it
# resolves rank_bm25 + jieba against the *host* interpreter's site-packages.
# Those two packages live in the shared dev venv (/root/.hermes/plugin-venv),
# NOT in the host tools-python. When the host lacks them, splice the
# plugin-venv site-packages dir onto sys.path so the in-process import succeeds.
# No-op when the host already resolves both; never blocks the import.
def _ensure_kb_dep_paths() -> None:
    """Make rank_bm25 + jieba importable from the running interpreter.

    Superseded by dep_bootstrap.ensure_kb_dep_paths (shared single source of
    truth, 2026-11-11) — kept as a thin re-export wrapper so any existing
    caller/referencer of this exact name still resolves; the actual splice
    logic now lives in dep_bootstrap."""
    import dep_bootstrap
    dep_bootstrap.ensure_kb_dep_paths()


_ensure_kb_dep_paths()

# doc_file URL fetch guards (http/https door, 11-08 P0 fix): a URL passed
# as doc_file is fetched via urllib with these ceilings. 100KB matches the
# content length gate upper bound _CONTENT_MAX (kb_service) — fetching
# more would be wasted since the ingest gate would reject it anyway.
_DOC_FILE_URL_MAX_BYTES = 100_000
_DOC_FILE_URL_TIMEOUT_S = 15

# Data dir: the shared host location (created by the deploy step; the
# service guards loud on absence). Override via KB_DATA_DIR.
_DATA_DIR = Path(os.environ.get("KB_DATA_DIR", os.path.join(_hermes_home, "kb-data")))

# Structured log: one shared file for all 7 gateway processes (single file,
# no rotation — KB is small; grep once = full trace). Override via
# KB_LOG_PATH. The service only *emits* records; this wiring points the
# file handler at the shared path so ingest (INFO) + search (DEBUG) land.
_LOG_PATH = os.environ.get(
    "KB_LOG_PATH", os.path.join(_hermes_home, "logs", "knowledge-base.log"),
)

# P0 audit trail: one shared JSONL for the governance actions (ingest /
# confirm / search) across all gateways. devops owns the 660 file + the
# 770 archive dir; override via KB_AUDIT_PATH.
_AUDIT_PATH = os.environ.get(
    "KB_AUDIT_PATH", os.path.join(_hermes_home, "logs", "knowledge-base-ingest.jsonl"),
)

# Module-level service so the hot layer persists across calls in one process;
# the cold layer reloads its pickle side-store on first use (fresh process).
_service = None


def _svc():
    global _service
    if _service is None:
        from kb_service import KbSearchService, configure_logging
        # Point the shared log file at the deploy path once (idempotent —
        # a 7-gateway process re-entering must not duplicate handlers).
        configure_logging(_LOG_PATH)
        _service = KbSearchService(_DATA_DIR, audit_path=Path(_AUDIT_PATH))
        # Q1 启动对账腿 (owner 定稿 2026-10-23, 选 A): svc 首次构造后做一次
        # "本地 hot+cold 全集 vs 中央 kb_docs 差集补推" (幂等, 只推差集,
        # 模块级哨兵守 host 生命周期一次, 防 _svc() per-session 重入再跑).
        # fail-soft: 中央不可达/token 拒静默跳过, 绝不阻塞 svc 构造.
        try:
            import kb_shared_memory as _kb_sm
            _kb_sm._svc_reconcile_once(_service)
        except Exception as e:
            logging.getLogger("knowledge-base").warning(
                f"kb: startup reconcile failed (fail-soft; retry next "
                f"host start): {e}")
    return _service


# ---------------------------------------------------------------------------
# Nested-field type coercion (reviewer M-1, 2026-10-19).
#
# _normalize_handler_args (handler_args.py) only validates that the OUTER
# payload is a dict. A model can still send a JSON-valid dict whose NESTED
# value is the wrong type — e.g. {"top_k": "abc"} or {"min_score": null} —
# and the handler's bare int()/float() coercion then raises an uncaught
# ValueError/TypeError (the same crash class handler_args.py was created
# to fix, one level down the field hierarchy). These two helpers wrap each
# nested coercion and fail CLOSED with a structured invalid_args payload
# instead of a traceback; a correctly-typed value passes through
# unchanged (byte-identical for every existing well-formed caller).
# ---------------------------------------------------------------------------

def _coerce_int_arg(args: dict, key: str, default: int) -> tuple:
    """Coerce args[key] to int with a default.

    Returns (value, None) on success, or (None, error_payload_dict) when
    the value is not a valid integer (string like 'abc', null, etc.).
    Missing key -> default (no error).
    """
    if key not in args:
        return default, None
    raw = args[key]
    try:
        return int(raw), None
    except (ValueError, TypeError):
        return None, {"status": "error", "error": "invalid_args",
                      "message": f"{key} must be an integer (got {raw!r})"}


def _coerce_float_arg(args: dict, key: str, default: float) -> tuple:
    """Coerce args[key] to float with a default.

    Returns (value, None) on success, or (None, error_payload_dict) when
    the value is not a valid number.
    """
    if key not in args:
        return default, None
    raw = args[key]
    try:
        return float(raw), None
    except (ValueError, TypeError):
        return None, {"status": "error", "error": "invalid_args",
                      "message": f"{key} must be a number (got {raw!r})"}


SEARCH_SCHEMA = {
    "name": "knowledge_base_search",
    "description": (
        "Single-shot knowledge-base search over the local two-layer "
        "retrieval service. Returns a JSON payload with top-k hits, each "
        "carrying full title + content + score + metadata — no follow-up "
        "fetch needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query (Chinese or English).",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5).",
            },
            "filters": {
                "type": "object",
                "description": (
                    "Optional metadata key-equality filters. Special key "
                    "'source_profile' controls profile scoping (TR-8): "
                    "when OMITTED, the caller's own profile is auto-injected "
                    "(a profile never sees other profiles' docs). "
                    "Pass source_profile=null to opt into the cross-profile "
                    "GLOBAL search door (the only path out of profile-"
                    "scoping, R-7). Pass source_profile='__unset__' to "
                    "match rows whose source_profile key is physically "
                    "absent OR null-valued (B1 None-filter sentinel). "
                    "Other filter keys (origin_kind, topic, premise_status, "
                    "kb_class, doc_kind) use exact-equality or the "
                    "'__unset__' sentinel (matches key-absent OR "
                    "value-None). doc_kind is the document-shape axis "
                    "(experience/tutorial/reference, orthogonal to "
                    "topic's task-domain axis — AND-filtered, see "
                    "kb_ingest_rules.DOC_KIND_VOCAB)."),
            },
            "min_score": {
                "type": "number",
                "description": (
                    "Optional lower-bound threshold on relevance score. "
                    "Results with score < min_score are excluded before "
                    "top_k truncation. Default 0.0 (no filtering, "
                    "preserves existing behaviour)."
                ),
            },
            "topic_scores": {
                "type": "object",
                "description": (
                    "TR-6 fourth-axis topic-mass map {doc_id: mass}. "
                    "Optional; ONLY this explicit tool face may supply "
                    "it (the pre_llm_call auto-inject layer never fuses "
                    "topic-mass — aligned with the ReinforceStore "
                    "write-gate 'only explicit tool calls count' rule). "
                    "The axis is inert unless the candidate pool "
                    "carries >= 20 origin_kind=agent entries, so on a "
                    "small ledger the payload is byte-identical to the "
                    "default three-axis search."
                ),
            },
        },
        "required": ["query"],
    },
}

INGEST_SCHEMA = {
    "name": "knowledge_base_ingest",
    "description": (
        "Ingest documents into the knowledge base. Each document is a "
        "dict {docid?, title, content, ...metadata}. docid is optional: "
        "when omitted a deterministic content-addressed id (uuid5 over "
        "the doc's fields) is assigned, so re-ingesting the same content "
        "is idempotent. "
        "Governance (P0): each doc may carry status ('pending' default | "
        "'confirmed'), origin_kind ('auto' default | 'user' | 'factory'), "
        "and source_profile (owning profile). pending docs go to a "
        "physically-isolated staging store and are NOT searchable until "
        "confirmed; confirmed docs enter the two-layer corpus and are "
        "searchable immediately. Confirmed docs pass a length gate "
        "(title 5-200 chars, content 60-10000 UTF-8 bytes — 60 bytes ≈ "
        "20 CJK chars; a short CJK string below ~20 chars is rejected as "
        "trivial; the gate measures UTF-8 BYTES, not characters, "
        "kb_service._CONTENT_MIN, 2026-10-19 owner decision) and a BM25 "
        "near-duplicate self-check (no model, no external API). "
        "Ceiling guidance: a doc whose content exceeds the 10000-byte "
        "UTF-8 cap is REJECTED by the length gate — before ingesting, "
        "measure content byte length and compress to under 10000 bytes "
        "(keep the phenomenon/root-cause/fix/verification chain, trim "
        "redundant phrasing), or run knowledge_base_distill on it "
        "first. Retrying an over-cap doc without compressing or "
        "distilling always re-rejects. "
        "Payload discipline (2026-11-04, hardened 2026-11-07): the host "
        "tool_call envelope stringifies the WHOLE call as JSON; every char "
        "of an inline content body lands in a nested JSON string (quotes, "
        "newlines, code fences, CJK all re-escaped), and in practice the "
        "envelope breaks mid-string at ~3KB, making the entire tool_call "
        "call fail host-side JSON parsing — the doc is then lost and the "
        "whole call retries. HARD RULE: a doc whose body exceeds ~1KB, or "
        "contains quotes/newlines/code blocks/escape-heavy text, MUST use "
        "doc_file — write the body with write_file under the KB data dir, "
        "then pass doc_file pointing at it; the inline envelope stays tiny "
        "and the call cannot break. Only short plain-text bodies (< ~1KB, "
        "no quotes or newlines or code blocks) may be inlined. Do NOT "
        "inline long bodies and hope; if a tool_call fails with \"not "
        "valid JSON\", the fix is switching that doc to doc_file, not "
        "re-typing the same payload."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "docs": {
                "type": "array",
                "description": "List of document dicts to ingest.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {
                            "type": "string",
                            "description": (
                                "The document body. OMIT when 'doc_file' is "
                                "set — content is read from that file instead."
                            ),
                        },
                        "doc_file": {
                            "type": "string",
                            "description": (
                                "Optional: an absolute path, a path relative "
                                "to the KB data dir, OR an http(s) URL, to "
                                "read 'content' from instead of inlining it. "
                                "A URL (http:// or https://) is fetched via "
                                "urllib with a 15s timeout and a 100KB ceiling "
                                "(UTF-8); a non-UTF-8 or over-cap response is "
                                "refused structurally. REQUIRED for any doc "
                                "body > ~1KB or containing quotes/newlines/code "
                                "blocks (see the tool's Payload discipline "
                                "note) — it kills multi-layer JSON-escape "
                                "payloads at the root (write the doc to a file "
                                "with write_file, then point doc_file at it; "
                                "the envelope stays tiny and cannot break). "
                                "When set, any 'content' "
                                "field on the doc is ignored and overwritten "
                                "by the file's text, and the doc_file key is "
                                "removed before ingest so the "
                                "content-addressed docid is identical to the "
                                "inline-content equivalent (idempotent). "
                                "Refused (whole call fails structurally) if "
                                "the resolved path escapes the KB data dir or "
                                "the file is missing/unreadable."
                            ),
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "confirmed"],
                            "description": (
                                "Governance status. 'pending' (default) "
                                "lands the doc in staging (not searchable "
                                "until confirmed). 'confirmed' enters the "
                                "searchable corpus directly."
                            ),
                        },
                        "origin_kind": {
                            "type": "string",
                            "enum": ["auto", "user", "factory", "agent"],
                            "description": (
                                "Provenance: 'auto' (model-detected, "
                                "default), 'user' (human-authored), "
                                "'factory' (seed/built-in), "
                                "'agent' (agent-experience-ledger, TR-1)."
                            ),
                        },
                        "source_profile": {
                            "type": "string",
                            "description": (
                                "The owning Hermes profile (e.g. "
                                "'coder', 'devops'). Optional; when "
                                "omitted the doc is a shared/global "
                                "entry."
                            ),
                        },
                        "doc_kind": {
                            "type": ["string", "null"],
                            "enum": ["experience", "tutorial",
                                     "manual", "reference", "material",
                                     None],
                            "description": (
                                "Document-SHAP axis (orthogonal to "
                                "topic, which is a task-domain axis — a "
                                "doc may carry topic=deployment AND "
                                "doc_kind=tutorial, AND-filtered in "
                                "search()). 'experience' = retrospective "
                                "pitfall/post-mortem; 'tutorial' = "
                                "forward-looking followable procedure "
                                "(SOP/runbook, 'unused until needed'); "
                                "'manual' = reference manual / runbook "
                                "(a system's how-to, not a step-by-step "
                                "lesson); 'reference' = pure reference "
                                "fact (no judgement, no steps); "
                                "'material' = teaching material / book "
                                "excerpt (long-form study content). "
                                "Optional — omit for an unlabeled doc "
                                "(doc_kind=None, NOT implicitly "
                                "'experience'; no back-fill of legacy "
                                "docs). Valid values: "
                                "kb_ingest_rules.DOC_KIND_VOCAB. "
                                "Governance (owner 2026-10-19): a "
                                "user-specified doc_kind (this face) "
                                "lands as-is — no review gate; the "
                                "owner-review constraint applies to "
                                "AGENT auto-captured docs (which "
                                "default to doc_kind=None and are "
                                "labeled only via the explicit "
                                "knowledge_base_set_doc_kind face)."
                            ),
                        },
                    },
                    "required": ["title"],
                    "anyOf": [
                        {"required": ["content"]},
                        {"required": ["doc_file"]},
                    ],
                },
            },
        },
        "required": ["docs"],
    },
}

CONFIRM_SCHEMA = {
    "name": "knowledge_base_confirm",
    "description": (
        "Confirm a PENDING knowledge document into the searchable corpus. "
        "Moves the doc atomically from the physically-isolated staging "
        "store into the confirmed two-layer corpus (hot + cold rank_bm25). "
        "Requires that the doc exists in staging with status=pending and "
        "that the caller's source_profile matches the stored row's "
        "source_profile (cross-profile confirm is rejected + audited as "
        "profile_mismatch). The caller must have obtained explicit user "
        "consent (via clarify or equivalent) before invoking this tool — "
        "this is the code-side backstop; the policy gate lives in the "
        "model's prompt."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "The content-addressed docid to confirm.",
            },
            "source_profile": {
                "type": "string",
                "description": (
                    "The caller's profile identity. Must match the "
                    "stored row's source_profile (or both be null for "
                    "global/legacy rows)."
                ),
            },
            "confirmed_by": {
                "type": "string",
                "description": (
                    "The user identifier who approved this confirm "
                    "(e.g. a clawchat user id). Recorded in the audit "
                    "trail."
                ),
            },
        },
        "required": ["doc_id"],
    },
}

DISTILL_SCHEMA = {
    "name": "knowledge_base_distill",
    "description": (
        "Ingest-side distillation (owner '蒸馏' requirement): reduce raw "
        "text to a MINIMAL VERIFIABLE knowledge unit before it enters the "
        "KB corpus. The LLM judges which tier applies — (1) the model "
        "already knows it cold (drop_reason set, staged PENDING with a "
        "`dropped` mark so the user can object, NOT silently discarded), "
        "(2) the model misremembers it (keep only the part it gets "
        "wrong), (3) the model does not know it at all (distill to a "
        "minimal self-contained unit) — and outputs "
        "{tier, drop_reason?, distilled_title, distilled_content, "
        "anchor_points[]}. Fail-soft (any failure / over-budget / no "
        "host runtime): the ORIGINAL text is stored through the existing "
        "length gate; distillation is an enhancement, never the only "
        "judge. The host main-model channel is the ONLY model route — "
        "no custom model/API/provider config (owner rule ⑤). This is a "
        "STANDALONE tool: it returns the verdict JSON; it does NOT "
        "ingest by itself. To persist the distilled doc, call "
        "knowledge_base_ingest with distilled_title/distilled_content "
        "(the caller is expected to swap them in)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "raw_title": {"type": "string"},
            "raw_content": {"type": "string"},
        },
        "required": ["raw_title", "raw_content"],
    },
}


def _search_handler(args: dict, **kwargs) -> str:
    args = _normalize_handler_args(args)
    if args.get("status") == "error":
        return json.dumps(args, ensure_ascii=False)
    # Required-field guard (m-3): schema declares query required — an
    # empty/missing query returns a structured error, not a zero-hit search.
    if not str(args.get("query") or "").strip():
        return json.dumps({"status": "error",
                           "error": "query is required"},
                          ensure_ascii=False)
    svc = _svc()
    # Nested-field coercion (M-1): fail closed on a malformed min_score.
    min_score, min_err = _coerce_float_arg(args, "min_score", 0.0)
    if min_err is not None:
        return json.dumps(min_err, ensure_ascii=False)
    top_k, top_err = _coerce_int_arg(args, "top_k", 5)
    if top_err is not None:
        return json.dumps(top_err, ensure_ascii=False)
    filters = args.get("filters") or None
    # TR-8: read-side default profile-scoping. If the caller did NOT pass an
    # explicit source_profile filter, inject the current caller's
    # source_profile as an implicit filter (mirrors confirm()'s source_profile
    # matching; TR-8 of the agent-experience-ledger PRD).
    #   - filters absent (None)                     -> inject {source_profile}
    #   - filters given WITHOUT source_profile      -> inject source_profile
    #   - filters given WITH source_profile=<value> -> respect that value
    #   - filters given WITH source_profile=None    -> the intentional
    #     cross-profile GLOBAL opt-in door (R-7): the ONLY path out of
    #     profile-scoping. Kept open on purpose; document, don't seal.
    # The confirm() side's source_profile hard gate (profile_mismatch reject
    # + audit) is a separate write-path rule and is untouched by this.
    from kb_ingest_hook import _current_profile
    caller_profile = _current_profile()
    if caller_profile is not None and (
            filters is None or "source_profile" not in filters):
        if filters is None:
            filters = {"source_profile": caller_profile}
        else:
            filters = {**filters, "source_profile": caller_profile}
    # TR-6 fourth-axis topic-mass: only THIS explicit tool face may
    # supply a topic_scores map; the pre_llm_call auto-inject layer
    # (kb_ingest_hook read-side callback) calls svc.search() without it
    # and stays inert (aligns with the ReinforceStore write-gate: only
    # explicit tool calls count). Below the 20-agent-entry pool gate
    # the map is inert and the payload is byte-identical to the
    # default three-axis result (KPI-6).
    topic_scores = args.get("topic_scores") or None
    payload = svc.search(
        query=args.get("query", ""),
        top_k=top_k,
        filters=filters,
        min_score=min_score,
        # RRF two-axis rerank ON by default (owner batch ②): the explicit
        # tool path fuses BM25 + the recency/reinforce axis so a genuinely
        # cited doc decays slower. The read-side prefetch calls the same
        # svc.search() but NEVER records a reinforce event, so the two
        # faces have distinct write-gates (owner/reviewer rule ③).
        rerank=True,
        topic_scores=topic_scores,
    )
    # ③ Reinforce write-gate: an explicit knowledge_base_search tool call is
    # a model citation signal. Record the top hit(s) so the read-side recency
    # axis can later rank them. (The tool return itself is not the follow-up
    # turn; anchor-matching against the next turn happens in a separate
    # reinforce(followup_text=...) call the caller makes later.)
    data = payload.get("data", {}).get("results", [])
    for hit in data[: top_k]:
        docid = str(hit.get("doc_id", ""))
        if docid:
            svc.reinforce(docid, reason="explicit_search")
    return json.dumps(payload, ensure_ascii=False)


CONFIRM_TOPIC_SCHEMA = {
    "name": "knowledge_base_topic_search",
    "description": (
        "Task-topic coarse-filter over the agent experience ledger "
        "(origin_kind=agent partition). Exact-match filter search: "
        "returns top-k agent-experience docs whose metadata topic "
        "equals the given value. topic must be one of the curated "
        "vocabulary (kb_ingest_rules.TOPIC_VOCAB, first phase: "
        "messaging / deployment / data-pipeline / frontend / "
        "general); an "
        "out-of-vocabulary topic is rejected with an error payload "
        "listing the valid values (NOT a silent zero-hit search). "
        "Omit topic (or pass null) to search the whole agent "
        "partition without the topic gate. Implicit source_profile "
        "scoping applies (TR-8, same as knowledge_base_search): "
        "profile Y never sees profile X's agent experience unless "
        "the caller opts into the global cross-profile door via the "
        "plain knowledge_base_search filters={source_profile: null}. "
        "This face ONLY reads the existing svc.search() filter "
        "gate — it does not change confirm(), the audit trail, the "
        "ReinforceStore write gate, or the pre_llm_call auto-inject "
        "layer. The fourth topic-mass RRF ranking axis (TR-6) is a "
        "separate change and is NOT part of this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query (Chinese or English).",
            },
            "topic": {
                "type": ["string", "null"],
                "description": (
                    "Task-topic coarse filter (exact metadata match). "
                    "A built-in TOPIC_VOCAB value (messaging / deployment "
                    "/ data-pipeline / frontend / general) OR an owner "
                    "project tag listed in kb-data/custom_topics.jsonl "
                    "(extensible without a code deploy — 11-08). "
                    "null / omitted = no topic gate, whole agent "
                    "partition. An unknown value is rejected with an "
                    "error payload listing the valid values (NOT a "
                    "silent zero-hit search)."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5).",
            },
        },
        "required": ["query"],
    },
}


def _topic_search_handler(args: dict, **kwargs) -> str:
    """TR-3 thin tool face: topic coarse-filter over the agent
    partition. Validates topic against kb_ingest_rules.TOPIC_VOCAB
    (out-of-vocab -> error payload, the chosen CLARITY semantics:
    the caller must fix its input rather than read a misleading
    'no experience under this topic'), then issues the EXISTING
    svc.search() with filters={topic, origin_kind: agent} + the
    implicit source_profile scoping (TR-8 precedent, plugin layer).
    No rrf kernel / confirm / audit / reinforce / pre_llm_call
    touch: the payload is the bare search result (the agent
    partition here is the coarse FILTER, the fourth topic-mass
    ranking axis is TR-6's separate change)."""
    import json as _json
    from kb_ingest_hook import topic_search_service_args
    args = _normalize_handler_args(args)
    if args.get("status") == "error":
        return _json.dumps(args, ensure_ascii=False)
    # Required-field guard (m-4): schema declares query required — an
    # empty/missing query returns a structured error, not a zero-hit search.
    if not str(args.get("query") or "").strip():
        return _json.dumps({"status": "error",
                            "error": "query is required"},
                           ensure_ascii=False)
    # Nested-field coercion (M-1): fail closed on a malformed top_k.
    top_k, top_err = _coerce_int_arg(args, "top_k", 5)
    if top_err is not None:
        return _json.dumps(top_err, ensure_ascii=False)
    try:
        kwargs_s = topic_search_service_args(
            args.get("query", ""),
            args.get("topic"),
            top_k,
        )
    except ValueError as exc:
        # The chosen clarity semantics: out-of-vocab topic rejects
        # HERE with the valid values listed, instead of silently
        # searching zero.
        return _json.dumps(
            {"status": "error", "error": "invalid_topic",
             "message": str(exc)}, ensure_ascii=False)
    svc = _svc()
    payload = svc.search(**kwargs_s)
    # ③ Reinforce write-gate parity with _search_handler: this is an
    # explicit tool call (a model citation signal). Record the top
    # hit(s) so the recency axis can rank them later — the pre_llm_call
    # path remains the zero-reinforce one (R-3).
    data = payload.get("data", {}).get("results", [])
    for hit in data[: top_k]:
        docid = str(hit.get("doc_id", ""))
        if docid:
            svc.reinforce(docid, reason="explicit_search")
    return _json.dumps(payload, ensure_ascii=False)


def _heal_escaped_content(docs: list) -> tuple[list, list]:
    """A-guard (2026-11-04): peel EXACTLY ONE layer of accidental
    JSON-serialization off each doc's string fields (title/content).

    The host tool_call wrapper stringifies the WHOLE args payload; when the
    MODEL ALSO json.dumps'd a doc's content field (or the payload was
    double-serialized end-to-end), the content value arrives as a
    JSON-encoded string literal instead of the raw text:
    ``"content": "\"real text\\\\n\""`` — a string that ITSELF parses back
    to a string. That shape used to trip repeated rejections on the tool
    face; this guard detects it and peels one layer so the ingest lands.

    Detection (conservative, per field): the value is a non-empty str,
    its stripped form is a valid JSON string LITERAL (starts/ends with
    double-quote and json.loads returns another str — NOT a dict/list,
    which would mean the caller MEANT a JSON blob as content), and is
    therefore peeled. One layer only: no recursion (a double-dumped
    payload still has one accidental layer left and the next tool face
    retry is fine; recursive unrolling risks mis-peeling legitimate
    JSON-blob content). Non-string fields, empty strings, non-dict docs,
    and invalid-JSON strings pass through UNCHANGED.

    Pure function, zero I/O — returns ``(healed_docs, notes)`` where
    ``notes[i]`` is None (doc i untouched) or a short human-readable note
    describing the peel. The original list is not mutated.
    """
    import json as _json
    healed: list = []
    notes: list = []
    for i, doc in enumerate(docs):
        note: str | None = None
        if isinstance(doc, dict):
            fixed = dict(doc)
            for field in ("title", "content"):
                val = fixed.get(field)
                if not isinstance(val, str) or not val:
                    continue
                s = val.strip()
                if len(s) < 2 or not (s[0] == '"' and s[-1] == '"'):
                    continue
                try:
                    inner = _json.loads(s)
                except (ValueError, _json.JSONDecodeError):
                    continue
                if isinstance(inner, str):
                    fixed[field] = inner
                    note = (note + "; " if note else "") + (
                        f"peeled one accidental JSON-serialization layer off "
                        f"'{field}' (was {len(val)} -> {len(inner)} chars)"
                    )
            healed.append(fixed)
        else:
            healed.append(doc)
        notes.append(note)
    return healed, notes


def _read_doc_file(doc_file: str, data_dir: Path) -> tuple[str | None, str | None]:
    """B: read a doc's content from a file on disk — or from an http(s) URL.

    ``doc_file`` is one of:
      - an http:// or https:// URL -> fetched via urllib (15s timeout,
        100KB ceiling); on any failure returns (None, error), never
        silently falls back to a local path. Recognizing the scheme up
        front is REQUIRED: Path('https://x') is a relative path on POSIX
        and would be joined under data_dir then refused by the
        relative_to guard (the 11-08 ingest bug, doc_file
        '/root/.hermes/https://kb-font-offline.md' resolves outside the
        KB data dir).
      - an absolute path, or a path relative to ``data_dir`` -> read from
        disk. The RESOLVED path must stay inside ``data_dir`` — anything
        escaping the root (``../../etc/passwd``, an absolute path under a
        sibling dir, a symlink out of the root) is refused.

    Pure I/O — no service state; callers delete the ``doc_file`` key from
    the doc dict BEFORE ingest so the content-addressed docid
    (uuid5 over the whole dict) is identical for the inline-content and
    doc_file-read versions of the same document.
    """
    stripped = doc_file.strip()
    low = stripped.lower()
    if low.startswith("http://") or low.startswith("https://"):
        import urllib.request
        req = urllib.request.Request(
            stripped, headers={"User-Agent": "hermes-kb-ingest/1.0"})
        try:
            with urllib.request.urlopen(req,
                                        timeout=_DOC_FILE_URL_TIMEOUT_S) as resp:
                body = resp.read(_DOC_FILE_URL_MAX_BYTES + 1)
        except Exception as exc:  # URLError, HTTPError, timeout, ...
            return None, f"doc_file URL '{doc_file}' fetch failed: {exc}"
        if len(body) > _DOC_FILE_URL_MAX_BYTES:
            return None, (f"doc_file URL '{doc_file}' exceeds "
                          f"{_DOC_FILE_URL_MAX_BYTES} bytes; refused.")
        try:
            return body.decode("utf-8"), None
        except UnicodeDecodeError:
            return None, (f"doc_file URL '{doc_file}' is not valid UTF-8; "
                          f"refused.")

    p = Path(doc_file).expanduser()
    if not p.is_absolute():
        p = data_dir / p
    try:
        resolved = p.resolve()
    except OSError as exc:
        return None, f"doc_file '{doc_file}': path resolution failed ({exc})"
    try:
        resolved.relative_to(data_dir)
    except ValueError:
        return None, (f"doc_file '{doc_file}' resolves outside the KB data "
                      f"dir ({data_dir}); refused.")
    if not resolved.is_file():
        return None, f"doc_file '{doc_file}' is not a readable file."
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"doc_file '{doc_file}' read failed: {exc}"
    return text, None


def _ingest_handler(args: dict, **kwargs) -> str:
    args = _normalize_handler_args(args)
    if args.get("status") == "error":
        return json.dumps(args, ensure_ascii=False)
    svc = _svc()
    docs = args.get("docs") or []
    if not docs:
        return json.dumps({"status": "error",
                           "error": "no docs provided"}, ensure_ascii=False)

    # A-guard: peel one accidental JSON-serialization layer off string
    # fields (title/content) so a double-dumped model payload still lands
    # instead of getting re-rejected. Per-doc notes surface in out below.
    docs, escape_notes = _heal_escaped_content(docs)

    # B: doc_file override — a doc whose 'content' is absent or is pointed
    # at a file on disk via 'doc_file' has its content read from that file
    # (root-constrained to the KB data dir). The doc_file key is then
    # DELETED so the content-addressed docid matches the inline-content
    # equivalent. A read failure for ANY doc fails the WHOLE call
    # structurally (no partial ingest of a mis-specified batch).
    #
    # Required-field guard (m-5, same discipline as m-1..m-4): the schema
    # says each doc needs 'content' OR 'doc_file'; a dict doc carrying
    # NEITHER (or a non-dict doc) is a malformed batch — fail the whole
    # call structurally, never ingest a doc whose body is empty.
    resolved_docs: list = []
    for doc in docs:
        if not isinstance(doc, dict):
            return json.dumps({"status": "error",
                               "error": (
                                   f"malformed doc #{len(resolved_docs)}: "
                                   f"a doc must be an object with 'title' "
                                   f"and 'content' (or 'doc_file')")},
                              ensure_ascii=False)
        has_content = bool(str(doc.get("content") or "").strip())
        has_doc_file = bool(str(doc.get("doc_file") or "").strip())
        if not has_content and not has_doc_file:
            return json.dumps({"status": "error",
                               "error": (
                                   f"malformed doc #{len(resolved_docs)}: "
                                   f"'content' or 'doc_file' is required "
                                   f"(title={doc.get('title')!r})")},
                              ensure_ascii=False)
        if has_doc_file:
            text, err = _read_doc_file(str(doc["doc_file"]), _DATA_DIR)
            if err:
                return json.dumps({"status": "error",
                                   "error": f"doc_file: {err}"},
                                  ensure_ascii=False)
            merged = dict(doc)
            merged.pop("doc_file", None)
            merged["content"] = text
            resolved_docs.append(merged)
        else:
            resolved_docs.append(doc)

    # 11-08 manual-ingest LLM topic auto-classify (owner 方案②, write-side
    # LLM face #2): after doc_file resolution (so the LLM sees the REAL
    # content, not just the title), for each doc that does NOT explicitly
    # carry a topic, ask the host main-model to pick the best topic tag
    # from the valid vocab (built-in TOPIC_VOCAB + custom_topics.jsonl
    # project tags). Budget: 3 LLM calls per tool_call so one large batch
    # cannot burn the whole session. Any failure (no host runtime, host
    # down, timeout, unparseable / out-of-vocab reply, over-budget) leaves
    # the doc's topic UNTOUCHED — the floor "general" (stamped later by
    # svc.ingest's _normalize_docs when the key is absent) stands. A doc
    # that DID carry an explicit topic is NEVER overridden (explicit
    # user/agent input wins; the LLM face only fills the gap).
    topic_notes: list = []
    from kb_ingest_rules import TOPIC_VOCAB, DOC_KIND_VOCAB
    from kb_ingest_hook import _llm_topic_classify, _custom_topics, \
        _llm_doc_kind_classify
    _TOPIC_AUTO_LIMIT = 3
    _DOC_KIND_AUTO_LIMIT = 3
    valid_topics = list(TOPIC_VOCAB) + list(_custom_topics())
    # A main_runtime snapshot is only available in a host-driven process;
    # a standalone / no-host call gets None -> every auto-classify is a
    # no-op floor (fail-soft: the manual face must never break on the
    # LLM face's absence).
    main_runtime = None
    try:
        from kb_ingest_hook import _norm_main_runtime
        if _norm_main_runtime is not None:
            main_runtime = _norm_main_runtime(None)
    except Exception:
        main_runtime = None
    topic_calls = 0
    doc_kind_calls = 0
    for doc in resolved_docs:
        if not isinstance(doc, dict):
            continue
        title = str(doc.get("title") or "").strip()
        body = str(doc.get("content") or "")[:1500]
        # 11-08 topic LLM auto-classify: for each doc that does NOT
        # explicitly carry a topic, ask the host main-model to pick the
        # best topic tag (the LLM face only fills the gap; a doc that
        # DID carry an explicit topic is NEVER overridden).
        explicit = str(doc.get("topic") or "").strip()
        if not explicit and main_runtime is not None \
                and topic_calls < _TOPIC_AUTO_LIMIT:
            picked = _llm_topic_classify(
                main_runtime, f"{title}\n{body}", valid_topics)
            topic_calls += 1
            if picked:
                doc["topic"] = picked
                topic_notes.append(
                    f"topic auto-classified to '{picked}' for "
                    f"'{title[:50]}' (LLM write-side face)")
        # 11-08 doc_kind LLM auto-classify (owner ruling: ingest 自动打标):
        # for each doc that does NOT explicitly carry a doc_kind, ask the
        # host main-model to pick the best document-shape tag from the
        # controlled DOC_KIND_VOCAB (experience/tutorial/manual/
        # reference/material). Budget: _DOC_KIND_AUTO_LIMIT LLM calls per
        # tool_call (separate from the topic pool so one large batch
        # cannot burn both faces' budgets). Any failure (no host runtime,
        # host down, timeout, unparseable / out-of-vocab reply,
        # over-budget) leaves the doc's doc_kind UNTOUCHED — the floor
        # None ("unlabeled") that svc.ingest's _normalize_docs stamps when
        # the key is absent stands. A doc that DID carry an explicit
        # doc_kind is NEVER overridden (explicit user/agent input wins;
        # the LLM face only fills the gap).
        explicit_kind = str(doc.get("doc_kind") or "").strip()
        if not explicit_kind and main_runtime is not None \
                and doc_kind_calls < _DOC_KIND_AUTO_LIMIT:
            picked_kind = _llm_doc_kind_classify(
                main_runtime, f"{title}\n{body}", list(DOC_KIND_VOCAB))
            doc_kind_calls += 1
            if picked_kind:
                doc["doc_kind"] = picked_kind
                topic_notes.append(
                    f"doc_kind auto-classified to '{picked_kind}' for "
                    f"'{title[:50]}' (LLM write-side face)")

    result = svc.ingest(resolved_docs)
    svc.persist()  # make ingested docs durable across process restarts
    # P0 governance summary: which docs confirmed (searchable), which were
    # rejected (length/dup gate), which are staged pending user confirm.
    out = {
        "status": "success",
        "ingested": result.get("ingested", []),
        "rejected": result.get("rejected", []),
        "staged": result.get("staged", []),
        "generation": result.get("generation"),
    }
    # Payload normalization notes (A-guard peels + B doc_file reads) so the
    # caller sees the payload was healed rather than silently mis-read.
    # `docs` here is the post-heal list; a doc that still carries the
    # doc_file key used the file path — report it explicitly (the key is
    # already popped from resolved_docs, so it can't be seen there).
    payload_notes = [n for n in escape_notes if n] + topic_notes
    for doc in docs:
        if isinstance(doc, dict) and "doc_file" in doc:
            payload_notes.append(
                f"content read from doc_file '{doc['doc_file']}'")
    if payload_notes:
        out["payload_notes"] = payload_notes
    if result.get("staged"):
        out["hint"] = (
            "Staged docs are NOT searchable until confirmed. Call "
            "knowledge_base_confirm(doc_id, source_profile, confirmed_by) "
            "after obtaining user consent to move them into the corpus."
        )
    if result.get("rejected"):
        # Context-aware rejection guidance (10-28 lesson: over-cap docs got
        # retried 3x with no hint about the byte limit). Carry the exact
        # gate bounds + actionable fix for length_gate, per-reason hint
        # otherwise — the agent should be able to act on 'rejected' without
        # reading kb_service.py.
        gate = next((r for r in result["rejected"]
                     if r.get("reason") == "length_gate"), None)
        if gate is not None:
            field = str(gate.get("field", "content"))
            bounds = gate.get("bounds") or [None, None]
            unit = str(gate.get("unit", "chars"))
            lo, hi = bounds[0], bounds[1]
            if field == "content":
                reject_hint = (
                    f"Length gate rejected '{gate.get('doc_id')}' on content: "
                    f"got={gate.get('got')} UTF-8 bytes, allowed {lo}-{hi} UTF-8 bytes. "
                    f"Compress the body under {hi} UTF-8 bytes (keep the "
                    f"phenomenon/root-cause/fix/verification chain; trim redundant "
                    f"phrasing) or run knowledge_base_distill on it first, then "
                    f"re-ingest. Retrying the same over-cap content re-rejects."
                )
            else:
                reject_hint = (
                    f"Length gate rejected '{gate.get('doc_id')}' on {field}: "
                    f"got={gate.get('got')} {unit}, allowed {lo}-{hi} {unit}. "
                    f"Adjust the {field} into that range and re-ingest."
                )
            out["hint"] = (reject_hint
                           if "hint" not in out else
                           out["hint"] + " " + reject_hint)
        else:
            out["note"] = (
                "Rejected docs failed the length gate or the BM25 near-duplicate "
                "self-check; see 'rejected' for the per-doc reason."
            )
    return json.dumps(out, ensure_ascii=False)


def _confirm_handler(args: dict, **kwargs) -> str:
    args = _normalize_handler_args(args)
    if args.get("status") == "error":
        return json.dumps(args, ensure_ascii=False)
    svc = _svc()
    doc_id = str(args.get("doc_id", "")).strip()
    if not doc_id:
        return json.dumps({"status": "error",
                           "error": "doc_id is required"},
                          ensure_ascii=False)
    source_profile = args.get("source_profile")
    confirmed_by = args.get("confirmed_by")
    out = svc.confirm(doc_id, source_profile=source_profile,
                      confirmed_by=confirmed_by)
    return json.dumps(out, ensure_ascii=False)


def _distill_handler(args: dict, **kwargs) -> str:
    """Standalone distill tool: returns the verdict JSON. It does NOT
    ingest — the caller (the write-side hook, or a human agent) decides
    whether to persist via knowledge_base_ingest, swapping in
    distilled_title / distilled_content when the verdict passed the
    structure gate."""
    args = _normalize_handler_args(args)
    if args.get("status") == "error":
        return json.dumps(args, ensure_ascii=False)
    from kb_ingest_hook import _WriteSideState, _STATES, _current_profile
    raw_title = str(args.get("raw_title") or "").strip()
    raw_content = str(args.get("raw_content") or "").strip()
    # Required-field guard (m-2): schema declares BOTH raw_title and
    # raw_content required — enforce both (parity with _search/_topic
    # query guards), not the weaker one-of-two.
    if not raw_title:
        return json.dumps({"status": "error",
                           "error": "raw_title is required"},
                          ensure_ascii=False)
    if not raw_content:
        return json.dumps({"status": "error",
                           "error": "raw_content is required"},
                          ensure_ascii=False)
    # Reuse the process-local host main_runtime snapshot (the same cache
    # the post_llm_call hook face fills — a standalone tool call from a
    # live agent turn has the snapshot available; a CLI / standalone call
    # degrades to the fail-soft original-stands path).
    state = _WriteSideState(None)
    rt = state._main_runtime_live()
    from kb_distill import distill_content
    verdict = distill_content(rt, raw_title, raw_content)
    return json.dumps({"status": "success", "verdict": verdict},
                      ensure_ascii=False)


SHARE_TO_GLOBAL_SCHEMA = {
    "name": "knowledge_base_share_to_global",
    "description": (
        "Escalate ONE confirmed agent-experience document (origin_kind="
        "agent) to the global shared layer (PRD v2 TR-10). The doc is "
        "COPIED into the confirmed corpus with source_profile=None — the "
        "original profile row is left intact (additive, not a move) — so "
        "other profiles reach it via the TR-8 source_profile=None opt-in "
        "full search. HARD GATE: the document's stored source_profile must "
        "equal the caller's current profile (a profile cannot escalate "
        "another profile's experience), and its origin_kind must be "
        "'agent' (user/auto entries never walk to the global layer). Any "
        "violation is rejected and written to the audit trail as "
        "action=share_to_global with the error code (profile_mismatch / "
        "not_agent_origin / not_found). This is the explicit cross-profile "
        "boundary of the owner red line 'agent 经验不走私': same-profile "
        "self-confirm stays gateless, escalating to the global layer is "
        "this one audited action. Only one doc_id per call — no batch, no "
        "other knobs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": (
                    "The content-addressed docid of the confirmed agent "
                    "experience doc to escalate to the global layer."
                ),
            },
        },
        "required": ["doc_id"],
    },
}


def _share_to_global_handler(args: dict, **kwargs) -> str:
    """TR-10 narrow tool face: escalate one confirmed agent-experience
    doc to the global shared layer. Exactly one action, no other knobs.
    The caller's profile comes from the same _current_profile() source
    as _search_handler's implicit TR-8 scoping (one identity face, no
    new plumbing); the service method enforces both hard-gate
    conditions and writes every outcome to the existing _audit trail.
    """
    args = _normalize_handler_args(args)
    if args.get("status") == "error":
        return json.dumps(args, ensure_ascii=False)
    from kb_ingest_hook import _current_profile
    svc = _svc()
    doc_id = str(args.get("doc_id", "")).strip()
    # Required-field guard (m-1): schema declares doc_id required — an
    # empty/missing doc_id returns a structured error (parity with
    # _confirm_handler / _get_handler), not a not_found service error.
    if not doc_id:
        return json.dumps({"status": "error",
                           "error": "doc_id is required"},
                          ensure_ascii=False)
    caller_profile = _current_profile()
    out = svc.share_to_global(doc_id, source_profile=caller_profile)
    svc.persist()  # make the global copy durable across process restarts
    return json.dumps(out, ensure_ascii=False)


SET_DOC_KIND_SCHEMA = {
    "name": "knowledge_base_set_doc_kind",
    "description": (
        "Re-label a CONFIRMED knowledge-base doc's doc_kind IN PLACE "
        "(same docid, every other key untouched) — the supported path "
        "for moving a legacy unlabeled doc into a new doc_kind door "
        "without re-minting its docid (re-minting the same content "
        "would trip the B2 near-duplicate gate; this one does not, "
        "because it never goes through ingest()'s content path). "
        "doc_kind must be one of kb_ingest_rules.DOC_KIND_VOCAB "
        "(experience / tutorial / manual / reference / material) or "
        "null to CLEAR the label (back to unlabeled). Out-of-"
        "vocabulary values are rejected and audited, not silently "
        "accepted. The doc's docid is never recomputed — the row is "
        "mutated in place and re-persisted under the same docid (the "
        "proven set_premise_status in-place pattern). Governance "
        "(owner 2026-10-19): this face IS the owner-review action for "
        "labeling — agent auto-captured docs default to doc_kind=None "
        "and get a label only through this explicit call (or a user "
        "ingest with doc_kind set, which is user-specified and lands "
        "as-is). Returns {status, doc_id, from_kind, to} on success; "
        "{status:'error', error} on not_found / invalid_doc_kind."),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": (
                    "The content-addressed docid of the confirmed doc "
                    "to re-label."
                ),
            },
            "doc_kind": {
                "type": ["string", "null"],
                "enum": ["experience", "tutorial", "manual", "reference",
                         "material", None],
                "description": (
                    "The new document-SHAP label. 'experience' = "
                    "retrospective pitfall/post-mortem; 'tutorial' = "
                    "forward-looking followable procedure (SOP/"
                    "runbook); 'manual' = reference manual / runbook; "
                    "'reference' = pure reference fact; 'material' = "
                    "teaching material / book excerpt. Pass null to "
                    "clear the label (back to unlabeled). Must be a "
                    "DOC_KIND_VOCAB value or null; anything else is "
                    "rejected."
                ),
            },
        },
        "required": ["doc_id", "doc_kind"],
    },
}


def _set_doc_kind_handler(args: dict, **kwargs) -> str:
    """Narrow tool face: re-label one confirmed doc's doc_kind in
    place. Exactly two knobs (doc_id + new value), no batch, no other
    fields — the service method enforces vocab validation + the
    in-place invariant and writes every outcome to the existing audit
    trail."""
    args = _normalize_handler_args(args)
    if args.get("status") == "error":
        return json.dumps(args, ensure_ascii=False)
    svc = _svc()
    doc_id = str(args.get("doc_id", "")).strip()
    new_doc_kind = args.get("doc_kind")  # None means "clear the label"
    out = svc.set_doc_kind(doc_id, new_doc_kind)
    svc.persist()  # make the in-place mutation durable across restarts
    return json.dumps(out, ensure_ascii=False)


GET_SCHEMA = {
    "name": "knowledge_base_get",
    "description": (
        "Deterministic single-doc read by doc_id. Unlike "
        "knowledge_base_search (BM25 word-surface, top-k ranked), get() "
        "fetches EXACTLY one doc by its content-addressed id with no "
        "query and no ranking — it cannot 0-hit on a word-surface "
        "mismatch. Use it to verify 'is this specific doc reachable' "
        "(e.g. after confirm / share_to_global, to confirm the global "
        "copy landed) without gambling on a search query. Reads the "
        "confirmed two-layer corpus (hot then cold) and falls back to "
        "the staging store for pending rows. Returns "
        "{found, layer, doc_id, doc} where doc carries full title + "
        "content + metadata; layer is 'hot' | 'cold' | 'staging'. "
        "A not-found id returns {found: false, doc_id} — no error."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": (
                    "The content-addressed docid (uuid5) to fetch."
                ),
            },
        },
        "required": ["doc_id"],
    },
}


def _get_handler(args: dict, **kwargs) -> str:
    args = _normalize_handler_args(args)
    if args.get("status") == "error":
        return json.dumps(args, ensure_ascii=False)
    svc = _svc()
    doc_id = str(args.get("doc_id", "")).strip()
    if not doc_id:
        return json.dumps({"status": "error",
                           "error": "doc_id is required"},
                          ensure_ascii=False)
    out = svc.get(doc_id)
    out["status"] = "success"
    return json.dumps(out, ensure_ascii=False)


# ── S6 read-side: kb_search_central (read-only central kb_docs query) ──────
KB_SEARCH_CENTRAL_SCHEMA = {
    "name": "kb_search_central",
    "description": (
        "S6 读侧 (owner 定稿 2026-10-22, 三扇门门2共识): 查【中央】kb_docs "
        "(其他设备已确认的知识库文档) — 别人的参考知识. 纯只读 SQL: 不写本地 "
        "jsonl / 不重建 pkl / 不自动注入 / 不回写本地. 结果只活在本轮工具返回, "
        "下一轮即失效. 本地知识库召回 < 2 条、或需参考其他设备共享知识时调用; "
        "confirmed_only=true 只回 confirmed 共识, false 回全量沉淀. "
        "【路由: 何时用我】查【长知识文档】的中央层: 某主题的教程/排查方案/参考材料/agent 经验沉淀 "
        "(带 title+content+doc_kind 的长文). 与 filemem 中央层 (shared_memory_search, 查 memories 表, "
        "存短事实/偏好/决定/跨 session 延续) 是两套不同的表. 一句话: 查'某主题怎么做的教程/经验' -> 用我; "
        "查'用户偏好/做过什么决定/环境事实' -> 用 shared_memory_search. 拿不准或怕漏 -> 两边都查. "
        "注意: 共识面 (confirmed_only=true) 只放行 doc_kind IN (reference/material/tutorial), "
        "doc_kind=NULL 的料进不了共识面, 需用 confirmed_only=false 翻全量."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "关键词 (ILIKE 匹配 title/content/source_profile).",
            },
            "confirmed_only": {
                "type": "boolean",
                "description": "true=门2共识面(只 confirmed), false=门1全量. 默认 true.",
            },
            "limit": {
                "type": "integer",
                "description": "最多返回条数, 默认 50.",
            },
        },
        "required": ["query"],
    },
}


def _kb_search_central_handler(args: dict, **kwargs) -> str:
    """S6 读侧 handler: 只读查中央 kb_docs, 不碰本地数据源.
    中央 = 别人的参考记忆: 结果只进本轮工具返回, 绝不落本地 / 不进 auto-inject.
    config 缺段 / enabled=false -> 干净错误 (不炸). search_central 内 PG hiccup
    -> 空结果 (fail-soft). `sm_module` kwarg 供测试注入 kb_shared_memory 替身."""
    args = _normalize_handler_args(args)
    if args.get("status") == "error":
        return json.dumps(args, ensure_ascii=False)
    sm_module = kwargs.get("sm_module")
    if sm_module is None:
        import kb_shared_memory as sm_module
    cfg = sm_module._load_shared_memory_config()
    if cfg is None:
        return json.dumps(
            {"success": False,
             "error": "shared_memory 未启用 (config 缺 knowledge-base-search.shared_memory 段或 enabled=false)"},
            ensure_ascii=False)
    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"success": False, "error": "query 不能为空"},
                          ensure_ascii=False)
    confirmed_only = bool(args.get("confirmed_only", True))
    try:
        limit = int(args.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    try:
        rows = sm_module.search_central(cfg, query, confirmed_only=confirmed_only,
                                        limit=limit)
    except Exception as e:
        return json.dumps({"success": False, "error": f"kb_search_central 查询失败: {e}"},
                          ensure_ascii=False)
    # datetime 兜底: 宿主进程持有的 sm_module 可能是旧模块实例 (没 reload 含 _ts 修复
    # 的版本), 或某些 PG 驱动路径下 occurred_at/version_ts 原样返回 datetime 对象。
    # 出口统一默认 json 序列化策略, 任何 timestamptz 字段都过得了 json.dumps。
    def _json_default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    out_rows = []
    for r in rows:
        out_rows.append({
            "id": r.get("docid"),
            "title": r.get("title"),
            "content": r.get("content"),
            "status": r.get("status"),
            "source_profile": r.get("source_profile"),
            "agent_id": r.get("agent_id"),
            "version": r.get("version"),
            "version_ts": r.get("version_ts"),
            "occurred_at": r.get("occurred_at"),
            "doc_kind": r.get("doc_kind"),
            "meta": r.get("meta"),
        })
    return json.dumps({"success": True, "query": query,
                       "confirmed_only": confirmed_only,
                       "count": len(out_rows), "results": out_rows},
                      ensure_ascii=False, default=_json_default)


def register(ctx):
    """Register all tools with the Hermes framework (see lucky-calc shape)."""
    ctx.register_tool(
        name="knowledge_base_search",
        toolset="knowledge-base-search",
        schema=SEARCH_SCHEMA,
        handler=_search_handler,
        emoji="📚",
    )
    ctx.register_tool(
        name="knowledge_base_topic_search",
        toolset="knowledge-base-search",
        schema=CONFIRM_TOPIC_SCHEMA,
        handler=_topic_search_handler,
        emoji="🧭",
    )
    ctx.register_tool(
        name="knowledge_base_ingest",
        toolset="knowledge-base-search",
        schema=INGEST_SCHEMA,
        handler=_ingest_handler,
        emoji="📥",
    )
    ctx.register_tool(
        name="knowledge_base_confirm",
        toolset="knowledge-base-search",
        schema=CONFIRM_SCHEMA,
        handler=_confirm_handler,
        emoji="✅",
    )
    ctx.register_tool(
        name="knowledge_base_distill",
        toolset="knowledge-base-search",
        schema=DISTILL_SCHEMA,
        handler=_distill_handler,
        emoji="🧪",
    )
    ctx.register_tool(
        name="knowledge_base_share_to_global",
        toolset="knowledge-base-search",
        schema=SHARE_TO_GLOBAL_SCHEMA,
        handler=_share_to_global_handler,
        emoji="🌐",
    )
    ctx.register_tool(
        name="knowledge_base_set_doc_kind",
        toolset="knowledge-base-search",
        schema=SET_DOC_KIND_SCHEMA,
        handler=_set_doc_kind_handler,
        emoji="🏷️",
    )
    ctx.register_tool(
        name="knowledge_base_get",
        toolset="knowledge-base-search",
        schema=GET_SCHEMA,
        handler=_get_handler,
        emoji="🔍",
    )
    ctx.register_tool(
        name="kb_search_central",
        toolset="knowledge-base-search",
        schema=KB_SEARCH_CENTRAL_SCHEMA,
        handler=_kb_search_central_handler,
        emoji="🌍",
    )
    # Write side: post_llm_call candidate detection + session-aggregated
    # clarify (pure rules, no new LLM call). The sibling-module import is
    # relative to the plugin PACKAGE (hermes_plugins.knowledge-base-search)
    # when the gateway loads it as a package, or a bare name when the
    # plugin dir is on sys.path (the in-process service path). Both resolve
    # to the same file; the try/except guards the name-mismatch case.
    try:
        from .kb_ingest_hook import register_write_side, register_read_side
    except ImportError:
        try:
            from kb_ingest_hook import register_write_side, register_read_side
        except ImportError as exc:
            # Both package-relative and bare-name import failed: the
            # write-side hook is silently absent — leave a trace so a load
            # failure is diagnosable from the log (reviewer/data-engineer nit).
            import logging
            logging.getLogger("knowledge-base").warning(
                "kb write-side hook not registered — import failed both "
                "paths: %s", exc)
            raise
    register_write_side(ctx)
    # Read-side auto-retrieve (user "每 turn 捞"): a pre_llm_call code-side
    # fetch that injects a knowledge-base context block into the user turn
    # when the local two-layer BM25 returns hits (zero LLM, zero host-API
    # change — the host splices the returned context the same way it does
    # for filemem's filemem_auto_retrieve).
    register_read_side(ctx)
