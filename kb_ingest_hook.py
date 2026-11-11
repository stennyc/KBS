#!/usr/bin/env python3
"""knowledge-base-search write-side + read-side hook wiring.

Wiring contract (current — 2026-10-11 update):
  Write side (post_llm_call):
    - regex cue front-gate (kb_ingest_rules.CandidateDetector) stays as-is;
      LLM refine runs ONLY on a regex-flagged candidate, via the host
      main-model channel (agent.auxiliary_client.call_llm, shared with the
      running agent — no custom model/API/provider config, owner rule ⑤).
      Budget-gated per session; any failure -> fail-soft back to the regex
      verdict (the LLM face is an enhancement, never the only judge).
    - A candidate lands PENDING in the physically-isolated staging store
      (ingest status="pending"); the search path never indexes staging.
    - Clarify: at a session boundary or every clarify_batch_size candidates,
      the pending batch stays queued in the write-side ClarifyAggregator
      and is piggybacked into the next pre_llm_call context as a
      [知识待确认] footnote (filemem-style nudge, works in
      TUI/desktop/gateway where ctx.inject_message has no session_key/
      origin to post into); the model relays it to the user via the host
      clarify tool and consent flows through knowledge_base_confirm.
  Read side (pre_llm_call):
    - Code-side, ZERO-LLM per-turn auto-retrieve: the host's pre_llm_call
      hook collects the plugin's returned {"context": ...} and splices it
      into the user message (never the system prompt — see host
      turn_context.py L793-847 _collect_pre_llm_call_context).
    - Trivial query gate (is_trivial_query, independent in kb_ingest_rules)
      short-circuits; 0-hit -> return {} (no injection, direct pass);
      hits -> return {"context": "## 知识库参考\n..."} bounded by
      max_context_chars (default 1500). No score gating: raw_score=0 /
      negative are valid hits (min-max flattening under small corpus).
Config: defaults live in kb_ingest_rules.DEFAULT_RULES; a profile's
  plugins.entries.knowledge-base-search.kb_ingest block
merges incrementally over them (including the llm sub-block).
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import subprocess
import time as _time
from typing import Any, Dict, Optional


def _time_utcnow() -> str:
    """UTC ISO-8601 stamp (same shape as kb_ingest_rules._utcnow)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Sibling-module import: relative when the plugin loads as a package
# (hermes_plugins.knowledge-base-search.kb_ingest_hook), bare when the
# plugin dir is directly on sys.path (in-process service path).
try:
    from .kb_ingest_rules import (
        Candidate, CandidateDetector, ClarifyAggregator, dry_run_log,
        is_trivial_query, merge_rules, is_noise_title,
    )
    from .kb_distill import distill_content as _distill_content
    from .kb_distill import LLM_CALL_TIMEOUT as _LLM_JUDGE_TIMEOUT
except ImportError:
    from kb_ingest_rules import (
        Candidate, CandidateDetector, ClarifyAggregator, dry_run_log,
        is_trivial_query, merge_rules, is_noise_title,
    )
    from kb_distill import distill_content as _distill_content
    from kb_distill import LLM_CALL_TIMEOUT as _LLM_JUDGE_TIMEOUT

# kb_class (staging row) -> user-facing clarify label (the same mapping
# the rule engine assigns at scan time; see DEFAULT_RULES in
# kb_ingest_rules). Used ONLY to rebuild a Candidate from a durable
# staging row on the post-restart read path (the owner ruling "从
# staging_docs.jsonl 按 doc_id 重读").
# v2 (2026-10-14): the 4 v2 agent-knowledge classes; the 4 legacy
# (decision/walkthrough/config/incident) labels stay as keys ONLY for
# rebuilding pre-v2 staging rows that may still be on disk — they are
# no longer produced by the default rule set.
_KBCLASS_LABEL = {
    "pattern": "可迁移套路",
    "anti-pattern": "反模式陷阱",
    "tool-quirk": "工具环境脾气",
    "premise": "前提条件经验",
    "decision": "决策结论",
    "walkthrough": "踩坑排查路径",
    "config": "可复用配置片段",
    "incident": "边角经验事实",
}

# Host shared-model channel (user rule ⑤: filemem-style, NO custom model /
# API / provider config — reuse the host main-model route). Plugin-side
# import only; a standalone / older host degrades to None -> every LLM face
# keeps its fail-soft regex verdict (never a crash).
try:
    from agent.auxiliary_client import call_llm as _HOST_CALL_LLM
    from agent.auxiliary_client import _normalize_main_runtime as _norm_main_runtime
except Exception:  # pragma: no cover - standalone / older-host fallback
    _HOST_CALL_LLM = None
    _norm_main_runtime = None

logger = logging.getLogger("knowledge-base.hook")

# Profile identity for the running gateway process. The KB service stamps
# source_profile on docs; confirm() enforces cross-profile rejection. When
# unset (CLI runs), docs are global (source_profile=None).
_PROFILE_ENV = "HERMES_PROFILE"


def _current_profile() -> Optional[str]:
    """The owning profile of the CURRENT task (multiplex-aware).

    Ladder: host context-local hermes home (per-profile spawner binds the
    owning profile's home into a ContextVar — the env var alone is frozen
    at the launch profile under a multiplexed gateway) -> derive
    ``.../profiles/<name>`` -> ``<name>``; else the process env
    HERMES_PROFILE; else None (= global/CLI docs, confirm() treats as
    no cross-profile restriction). A missing host module (standalone)
    degrades to the env-only path — never a crash, never raises.
    """
    import os
    try:
        from hermes_constants import get_hermes_home
        home = str(get_hermes_home() or "")
    except Exception:  # pragma: no cover - standalone / older-host fallback
        home = os.environ.get("HERMES_HOME", "")
    if home:
        from pathlib import Path
        parts = [p for p in Path(home).parts if p]
        if "profiles" in parts:
            idx = parts.index("profiles")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    val = os.environ.get(_PROFILE_ENV, "").strip()
    return val or None


def _clarify_state_store() -> Optional[Any]:
    """The durable clarify DECISION-STATE sidecar (kb-data/clarify_state.json).

    owner 2026-10-13 ruling: the write-side clarify state (fired / pending /
    asked / session boundary) must survive a gateway restart — the PENDING
    docs already do (staging_docs.jsonl is the single source of truth for
    the FACT), but the DECISION STATE was memory-only (_STATES) and was
    zeroed on restart, so the read-side [知识待确认] footnote silently
    stopped surfacing and the pending docs rotted as zombies. This sidecar
    persists the decision state; the FACT stays in staging (owner ruling
    "从 staging_docs.jsonl 按 doc_id 重读").

    Fail-soft: a missing dir / import hiccup returns None and every caller
    degrades to the pure in-memory path (byte-identical to pre-change
    behaviour). Never raises into the turn loop.
    """
    try:
        from clarify_state import ClarifyStateStore
        import os
        from pathlib import Path
        data_dir = os.environ.get("KB_DATA_DIR", "/root/.hermes/kb-data")
        return ClarifyStateStore(Path(data_dir))
    except Exception as exc:
        logger.debug("kb clarify-state sidecar unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Write-side LLM candidate-refine face (user "由 LLM 判断" requirement,
# frozen plan): the regex cue pre-filter (kb_ingest_rules.CandidateDetector)
# stays the FRONT gate — LLM only ever runs on a turn the regex already
# flagged as a candidate (never every turn, 避免烧预算). The LLM pass
# refines/validates the candidate through the host main-model channel
# (shared with the running agent — NO custom model/API/provider config,
# user rule ⑤). Budget-gated per session; over-cap / no runtime / any host
# failure -> fail-soft back to the regex verdict (the candidate still lands
# PENDING, the regex path is byte-identical when the LLM face is absent).
# ---------------------------------------------------------------------------
_LLM_BUDGET_DEFAULT = 3
_LLM_JUDGE_INSTRUCTIONS = (
    "You are a knowledge-base curator. Given a turn's text, decide whether it "
    "is genuinely SINKING-WORTHY AGENT KNOWLEDGE — a reusable strategy, a "
    "known trap/anti-pattern, a tool behavior boundary, or an experience "
    "whose validity carries an explicit premise. "
    "If it is mostly conversational filler, acknowledgement, a bare verdict "
    "with no stated premise (\"我们拍板用 X\" is NOT this), or a question "
    "with no durable answer, judge it not worth saving. "
    'Return a single JSON object: {"worth": boolean, "title": string, '
    '"kb_class": "pattern"|"anti-pattern"|"tool-quirk"|"premise"|null}'
)
_LLM_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "worth": {"type": "boolean"},
        "title": {"type": "string"},
        "kb_class": {"type": ["string", "null"],
                     "enum": ["pattern", "anti-pattern", "tool-quirk",
                              "premise",
                              "decision", "walkthrough", "config",
                              "incident", None]},
    },
    "required": ["worth"],
}


def _llm_topic_classify(main_runtime, text: str,
                        valid_topics: list) -> Optional[str]:
    """Budget-gated host-LLM project-topic classification of one
    auto-captured knowledge unit (11-08, owner 方案②).

    Asks the host main-model to pick the best topic tag for `text`
    from `valid_topics` (built-in TOPIC_VOCAB + custom_topics.jsonl
    project tags). Returns the chosen tag (a valid list entry), or
    None on ANY failure — no main runtime / host call surface missing /
    host down / timeout / unparseable reply / a value not in
    valid_topics. None is the caller's "keep the floor value"
    signal (fail-soft: the LLM face is an ENHANCEMENT, never the only
    classifier; the regex/capture floor "general" stands when this
    returns None).

    Same host-channel discipline as _llm_judge_candidate: explicit
    main_runtime, no task key, no custom provider (user rule ⑤);
    temperature 0.0, tiny max_tokens (the reply is a single tag).
    """
    import json as _json
    import re as _re
    if _HOST_CALL_LLM is None or not isinstance(main_runtime, dict) \
            or not main_runtime:
        return None
    if not valid_topics:
        return None
    try:
        user_msg = (
            "你是知识库项目标签分类器。给定一段知识经验文本，"
            "从下列标签中选出最贴切的一个项目/主题标签。\n"
            "标签列表（必须从其中选，不得发明新标签）：\n"
            + "\n".join(f"- {t}" for t in valid_topics)
            + "\n\n若都不贴切，选 general。\n\n"
            "只输出标签本身（一个单词/短语），不要解释，不要 JSON，"
            "不要引号。\n\n知识文本：\n" + text[:2000]
        )
        route_info: dict = {}
        response = _HOST_CALL_LLM(
            main_runtime=main_runtime,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=24,
            timeout=_LLM_JUDGE_TIMEOUT,
            route_info=route_info,
        )
    except Exception as exc:
        logger.debug("kb LLM topic classify failed (%s); "
                     "keeping floor topic", exc)
        return None
    # Extract assistant text (dict- or namespace-shaped response).
    try:
        msg = response["choices"][0]["message"] if isinstance(response, dict) \
            else response.choices[0].message
        content = msg.get("content") if isinstance(msg, dict) \
            else getattr(msg, "content", None)
        if isinstance(content, list):
            content = "".join(str(p.get("text") if isinstance(p, dict)
                                else getattr(p, "text", "")) for p in content)
        text_out = (content or "").strip()
    except Exception:
        return None
    # Strip optional fence + surrounding quotes; take the first token-ish
    # value (the reply is meant to be a bare tag).
    m = _re.search(r"```(?:json)?\s*(.+?)```", text_out,
                   _re.DOTALL | _re.IGNORECASE)
    raw = m.group(1).strip() if m else text_out
    raw = raw.strip('"').strip("'").strip()
    # The LLM may echo with a little prose; accept only an EXACT match
    # against the valid list (case-sensitive, the same discipline as
    # TOPIC_VOCAB) — anything else is None (floor stands).
    for t in valid_topics:
        if raw == t:
            return t
    return None


# ---------------------------------------------------------------------------
# LLM doc_kind classification (11-08, owner ruling: ingest 自动打标)
# ---------------------------------------------------------------------------

_DOC_KIND_CLASSIFY_INSTRUCTIONS = (
    "你是知识库文档形态（doc_kind）分类器。给定一段知识/经验文档的标题与正文，"
    "判断它最贴切的文档形态，从下列 5 个值中选一个：\n"
    "- experience：回顾性踩坑/事后复盘（retrospective pitfall / post-mortem）\n"
    "- tutorial：前瞻性可跟随的操作教程（SOP / runbook / 排查修复指南）\n"
    "- manual：既有系统的操作手册/参考手册（how-to，非分步教程）\n"
    "- reference：纯参考事实（无判断、无步骤，如 API 字段约定、契约说明）\n"
    "- material：教材/书籍类长文（学习材料，非实操）\n"
    "只输出标签本身（一个单词），不要解释，不要 JSON，不要引号。若都不贴切，输出 none。\n\n"
)


def _llm_doc_kind_classify(main_runtime, text: str,
                           valid_kinds: list) -> Optional[str]:
    """Budget-gated host-LLM doc_kind (document-shape) classification of one
    auto-captured / manually-ingested knowledge unit (11-08, owner ruling:
    ingest 自动打标, replacing the 2026-10-18 'never pre-assign on the agent
    capture path' governance note — the label is now an LLM ENHANCEMENT with
    the SAME fail-soft floor: ANY failure (no main runtime / host down /
    timeout / unparseable / out-of-vocab reply) returns None, so the doc's
    doc_kind stays None ("unlabeled") exactly as before. A valid reply OVERRIDES
    the floor (the only path by which a non-None doc_kind is ever stamped at
    ingest time; explicit caller-supplied doc_kind keys always win — the
    caller is responsible for NOT calling this face when the key is set).

    Same host-channel discipline as _llm_topic_classify: explicit
    main_runtime, no task key, no custom provider; temperature 0.0, tiny
    max_tokens (the reply is a single tag).
    """
    import json as _json
    import re as _re
    if _HOST_CALL_LLM is None or not isinstance(main_runtime, dict) \
            or not main_runtime:
        return None
    if not valid_kinds:
        return None
    try:
        user_msg = _DOC_KIND_CLASSIFY_INSTRUCTIONS + \
                   "知识文本：\n" + text[:2000]
        route_info: dict = {}
        response = _HOST_CALL_LLM(
            main_runtime=main_runtime,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=24,
            timeout=_LLM_JUDGE_TIMEOUT,
            route_info=route_info,
        )
    except Exception as exc:
        logger.debug("kb LLM doc_kind classify failed (%s); "
                     "keeping floor doc_kind=None", exc)
        return None
    # Extract assistant text (dict- or namespace-shaped response).
    try:
        msg = response["choices"][0]["message"] if isinstance(response, dict) \
            else response.choices[0].message
        content = msg.get("content") if isinstance(msg, dict) \
            else getattr(msg, "content", None)
        if isinstance(content, list):
            content = "".join(str(p.get("text") if isinstance(p, dict)
                               else getattr(p, "text", "")) for p in content)
        text_out = (content or "").strip()
    except Exception:
        return None
    # Strip optional fence + surrounding quotes; take the first token-ish
    # value (the reply is meant to be a bare tag).
    m = _re.search(r"```(?:json)?\s*(.+?)```", text_out,
                   _re.DOTALL | _re.IGNORECASE)
    raw = m.group(1).strip() if m else text_out
    raw = raw.strip('"').strip("'").strip()
    # "none" / out-of-vocab -> None (floor stands). Only an EXACT match
    # against the valid list (case-insensitive — the LLM may reply
    # "Tutorial" — matched against the controlled vocab) is accepted.
    lowered = raw.lower()
    for k in valid_kinds:
        if lowered == str(k).lower():
            return k
    return None


def _llm_judge_candidate(main_runtime, text: str,
                         fallback: "Candidate") -> "Optional[Candidate]":
    """Budget-gated host-LLM refine of a regex-flagged candidate.

    Returns the (possibly LLM-upgraded) Candidate. ANY failure — no main
    runtime, host call surface missing, host down, unparseable reply, an
    enum-invalid kb_class, or a "worth=false" that we cannot act on
    deterministically — degrades to the UNTOUCHED regex `fallback` (fail-soft;
    the regex verdict is the floor, never regressed by the LLM face). A
    `worth=false` from a healthy LLM is treated as "skip this candidate"
    (return None) so a clearly-not-worth turn does not burn a staging slot.
    """
    import json as _json
    import re as _re
    if _HOST_CALL_LLM is None or not isinstance(main_runtime, dict) or not main_runtime:
        return fallback  # standalone / non-host / no snapshot: regex verdict stands
    try:
        schema_text = _json.dumps(_LLM_JUDGE_SCHEMA, ensure_ascii=False, sort_keys=True)
        user_msg = (
            _LLM_JUDGE_INSTRUCTIONS
            + "\n\nJSON schema:\n" + schema_text
            + "\nRespond with a single JSON object. No prose, no fences.\n\n"
            + "Input:\n" + text
        )
        route_info: dict = {}
        response = _HOST_CALL_LLM(
            main_runtime=main_runtime,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=96,
            timeout=_LLM_JUDGE_TIMEOUT,
            route_info=route_info,
        )
    except Exception as exc:  # host down / timeout / shape: regex verdict stands
        logger.debug("kb LLM judge failed (%s); keeping regex verdict", exc)
        return fallback
    # Extract assistant text (dict- or namespace-shaped response).
    try:
        msg = response["choices"][0]["message"] if isinstance(response, dict) \
            else response.choices[0].message
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            content = "".join(str(p.get("text") if isinstance(p, dict)
                                 else getattr(p, "text", "")) for p in content)
        text_out = (content or "").strip()
    except Exception:
        return fallback
    # Strip optional code fence + parse.
    m = _re.search(r"```(?:json)?\s*(.+?)```", text_out, _re.DOTALL | _re.IGNORECASE)
    raw = m.group(1).strip() if m else text_out
    try:
        parsed = _json.loads(raw)
    except Exception:
        return fallback
    if not isinstance(parsed, dict):
        return fallback
    worth = bool(parsed.get("worth"))
    if not worth:
        return None  # healthy LLM veto: skip the candidate (do not stage)
    # Apply LLM refinements onto the fallback (floor stays regex).
    _kb_class = parsed.get("kb_class")
    upgraded = Candidate(
        doc_id=fallback.doc_id,
        classified_type=(
            _kb_class if _kb_class in (
                "pattern", "anti-pattern", "tool-quirk", "premise",
                "decision", "walkthrough", "config", "incident",
            )
            else fallback.classified_type
        ),
        label=fallback.label,
        title=(str(parsed.get("title")).strip()[:80] or fallback.title),
        content=fallback.content,
        source_session_id=fallback.source_session_id,
        source_profile=fallback.source_profile,
        cue_hits=list(fallback.cue_hits) + ["llm:confirmed"],
        # TR-9: carry the regex-detected origin_kind forward (incident ->
        # "agent"; the 3 legacy classes -> None). The LLM verdict may
        # refine the kb_class but must not override the origin_kind
        # marker that the regex front gate established.
        origin_kind=fallback.origin_kind,
    )
    return upgraded


def _capture_repo_ctx() -> dict:
    """PRD v2-final TR-1 (note 3): capture the git repo context for the
    write-side turn loop.

    Runs ``git toplevel`` + ``git remote get-url origin`` in the CURRENT
    working directory (fail-soft end to end): git unavailable / not in a
    work tree / no origin remote -> ``{"project_ref": None,
    "git_available": False}`` -> the candidate routes to experience or
    transcript, NEVER to archive (owner ruling: git 不可用 → 不升 archive).
    The project_ref value is the controlled vocabulary form
    ``repo:<remote-url>`` — the remote URL is the stable repo identity.

    Cost posture: called ONCE per session (the _repo_ctx cache in
    _WriteSideState), not per turn; a 2-second subprocess ceiling so a
    hung git can never stall the post_llm_call turn loop.
    """
    import os
    out = {"project_ref": None, "git_available": False}
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=2.0, cwd=os.getcwd())
        if r.returncode == 0:
            url = r.stdout.strip()
            if url:
                out["project_ref"] = f"repo:{url}"
                out["git_available"] = True
    except Exception:
        pass  # fail-soft: git missing / not a repo / timeout -> None
    return out


def _service():
    """Lazy service handle shared with __init__ (same KB package on path)."""
    import os
    import sys
    from pathlib import Path

    # Bootstrap: make sure the knowledge-base service package dir and the
    # plugin-local dir are both on sys.path so the bare sibling imports in
    # this module and the lazy kb_service imports resolve regardless of how
    # the host loaded this file (package, direct file, or test import).
    # No-op when the host already placed them. Rollback-safe: delete this
    # two-line block to restore the prior host-only behaviour.
    _plugin_dir = os.path.dirname(os.path.abspath(__file__))
    if _plugin_dir not in sys.path:
        sys.path.insert(0, _plugin_dir)
    _kb_dir = os.environ.get("KB_HOME", "/root/.hermes/plugins/knowledge-base-search")
    if _kb_dir not in sys.path:
        sys.path.insert(0, _kb_dir)

    from kb_service import KbSearchService, configure_logging

    # The hook layer and the tool layer share one KB service instance per
    # process (same lock, same hot layer). The canonical owner of the
    # singleton is the plugin package module (__init__.py); this module is
    # its sibling, so reuse it when importable, else build our own.
    pkg = globals().get("__package__") or ""
    for modname in (pkg, "__init__"):
        if not modname:
            continue
        init_mod = sys.modules.get(modname)
        if init_mod is not None and hasattr(init_mod, "_svc"):
            return init_mod._svc()
    data_dir = Path(os.environ.get("KB_DATA_DIR", "/root/.hermes/kb-data"))
    log_path = os.environ.get(
        "KB_LOG_PATH", "/root/.hermes/logs/knowledge-base.log")
    configure_logging(log_path)
    return KbSearchService(data_dir)


def _effective_rules(ctx) -> dict:
    """Defaults merged with this profile's config override, fail-open.

    The config block lives at
      plugins.entries.knowledge-base-search.kb_ingest
    Real gateway path: the profile-scoped config loader (same source the
    capability gate uses — a multi-profile process reads THIS profile's
    home, never the ambient one). Test/direct callers may instead expose
    `load_config()` on the ctx. Any failure falls back to pure defaults —
    the write side must never break the turn loop.
    """
    cfg = None
    try:
        with _plugin_home_scope(ctx):
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly() or {}
    except Exception:
        cfg = None
    if cfg is None:
        try:
            loader = getattr(ctx, "load_config", None)
            cfg = loader() if callable(loader) else None
        except Exception:
            cfg = None
    if cfg is None:
        return merge_rules(None)
    if not isinstance(cfg, dict):
        return merge_rules(None)
    entry = ((cfg.get("plugins") or {}).get("entries") or {})
    entry = entry.get("knowledge-base-search") or {}
    return merge_rules(entry.get("kb_ingest") or None)


def _plugin_home_scope(ctx):
    """This plugin context's manager home scope (no-op fallback in tests)."""
    manager = getattr(ctx, "_manager", None) or getattr(ctx, "manager", None)
    if manager is None:
        return contextlib.nullcontext()
    from hermes_cli.plugins_loader import _plugin_home_scope as _pds
    return _pds(manager.home_path)


class _WriteSideState:
    """Process-local aggregator; session boundaries detected by session_id."""

    def __init__(self, ctx):
        self._ctx = ctx
        self._loaded_rules: Optional[dict] = None
        self._rules_by_session: dict[str, CandidateDetector] = {}
        self._last_session: Optional[str] = None
        self._aggregator: Optional[ClarifyAggregator] = None
        # PRD v2-final TR-1 (note 3): per-session repo-context capture cache
        # — session_id -> {"project_ref": str|None, "git_available": bool}.
        # Populated by on_turn's _capture_repo_ctx (fail-soft: git
        # unavailable / no remote -> project_ref None -> the doc routes to
        # experience/transcript, NEVER to archive). The capture is cached
        # per session because the turn loop is the only place the plugin
        # touches the working directory; a session that never fires a git
        # probe keeps the empty dict entry (key absent -> None lookup).
        self._repo_ctx: dict[str, dict] = {}
        # LLM write-side face state (budget-gated per session; the LLM pass
        # only ever runs on a regex-flagged candidate — 避免烧预算).
        self._llm_calls_by_session: dict[str, int] = {}
        # Distill face: INDEPENDENT per-session budget pool (owner decision
        # via reviewer — judge + distill must not share one 3-call pool,
        # else one candidate turn burns 2 of 3 and a session only gets
        # ~1.5 candidates judged + distilled).
        self._distill_calls_by_session: dict[str, int] = {}
        # LLM topic-classify face (11-08, owner 方案②): INDEPENDENT
        # per-session budget pool (same owner/reviewer decision as
        # judge/distill — a classify call must not burn the judge or
        # distill slots). Over-budget / no-runtime / any LLM failure ->
        # the doc keeps its floor topic "general" (fail-soft: the LLM
        # face is an enhancement, never the only classifier).
        self._topic_calls_by_session: dict[str, int] = {}
        # LLM doc_kind-classify face (11-08, owner ruling: ingest 自动打标):
        # INDEPENDENT per-session budget pool (same owner/reviewer decision
        # as judge/distill/topic — a classify call must not burn the other
        # pools). Over-budget / no-runtime / any LLM failure -> the doc's
        # doc_kind stays None ("unlabeled", the pre-change floor; fail-soft:
        # the LLM face is an enhancement, never the only labeler).
        self._doc_kind_calls_by_session: dict[str, int] = {}
        self._main_runtime: Optional[dict] = None
        # TR-4 task-boundary sink state: bounded per-session raw message
        # history (the "premise" source for the extracted agent docs) +
        # per-session agent-sink idempotence (one sink per session
        # boundary; a re-fired boundary in the same process is a no-op).
        self._session_msgs: dict[str, dict] = {}
        self._agent_sunk_sessions: set = set()
        # TR-4 test hook: pins the candidate list the boundary sink
        # stages (production path uses the aggregator's full per-session
        # candidate list; this keeps close_task_boundary directly
        # drivable in tests).
        self._pending_override: Optional[list] = None
        # TR-4 test hook: pins the service the sink talks to (None ->
        # the module's real _service() resolution, production path).
        self._service_override = None
        # Filemem-style clarify footnote: per-session record of a fired
        # _fire_clarify trigger (batch threshold or session boundary).
        # Only released batches surface via _clarify_footnote;
        # sub-threshold candidates that never triggered stay stashed.
        self._fire_clarify_fired: set = set()
        # The previous session of the last boundary-triggered
        # _fire_clarify (the remainder the boundary released).
        self._last_boundary_prev: Optional[str] = None

    def rules(self) -> dict:
        """Profile's effective rules (defaults + config merge), cached once.

        Load failures fall back to pure defaults (fail-open: the write side
        never blocks the turn loop).
        """
        if self._loaded_rules is None:
            try:
                self._loaded_rules = _effective_rules(self._ctx)
            except Exception as exc:
                logger.debug("kb effective-rules load failed, using "
                             "defaults: %s", exc)
                from kb_ingest_rules import merge_rules
                self._loaded_rules = merge_rules(None)
        return self._loaded_rules

    def _agg(self) -> ClarifyAggregator:
        if self._aggregator is None:
            self._aggregator = ClarifyAggregator(
                int(self.rules().get("clarify_batch_size", 5)))
        return self._aggregator

    def _detector_for(self, session_id: str) -> CandidateDetector:
        det = self._rules_by_session.get(session_id)
        if det is None:
            det = CandidateDetector(self.rules(), session_id,
                                    _current_profile())
            self._rules_by_session[session_id] = det
        return det

    def on_turn(self, session_id: str, assistant_response: str,
                user_message: str) -> None:
        """One post_llm_call turn. Regex front gate; LLM refine (budget-gated,
        host main-model channel, fail-soft to regex verdict) on the candidate."""
        if not session_id or not assistant_response:
            return
        at_boundary = self._last_session is not None and \
            self._last_session != session_id
        prev_session = self._last_session
        self._last_session = session_id

        # TR-4: remember this turn's raw messages (bounded, per session) —
        # the task-boundary sink uses them as the extracted docs' premise
        # context. Pure in-memory bookkeeping, never blocks the turn.
        self.remember_turn_msgs(session_id, user_message, assistant_response)

        # PRD v2-final TR-1 (note 3): capture the git repo context
        # (toplevel/remote) for this session, ONCE per session
        # (fail-soft: git unavailable -> project_ref None -> no archive
        # upgrade). Cache in _repo_ctx; the capture itself is the module
        # helper _capture_repo_ctx below.
        if session_id not in self._repo_ctx:
            self._repo_ctx[session_id] = _capture_repo_ctx()

        det = self._detector_for(session_id)
        text = f"{user_message or ''}\n{assistant_response}"
        cand = det.scan(text, docid_fn=self._docid_fn())
        clarify_any = False

        # Write-side LLM judge (user "由 LLM 判断" requirement): the regex
        # detector is the FRONT gate (LLM never runs on non-candidate turns —
        # 避免烧预算). When the regex flagged a candidate, a budget-gated host
        # LLM pass refines/validates it; any failure or over-budget keeps the
        # regex verdict byte-identical. A healthy `worth=false` veto skips the
        # staging slot (return None), an LLM upgrade carries refined
        # kb_class/title + the "llm:confirmed" cue stamp.
        if cand is not None and self._llm_enabled():
            cand = self._maybe_llm_judge(session_id, cand, text)

        # Ingest-side distill (owner "蒸馏" requirement): after a candidate
        # survives the judge (cand is a Candidate, not None — a judge veto
        # already skipped staging), a budget-gated LLM pass distills it to a
        # minimal verifiable knowledge unit. The distill pool is INDEPENDENT
        # of the judge pool (owner/reviewer decision). fail-soft: any
        # failure / over-budget -> store the ORIGINAL content through the
        # existing length gate (never lost). `drop_reason` non-empty
        # (LLM judged the model already knows it) -> the doc is staged
        # PENDING with a `dropped` mark so the clarify user message can list
        # it and the user can object; it is NOT silently discarded.
        distill_v: Optional[dict] = None
        if cand is not None:
            distill_v = self._maybe_distill(session_id, cand)

        if at_boundary and prev_session:
            # TR-4 task-boundary sink (G5, runs BEFORE the clarify below and
            # is idempotent per session): stage the previous session's
            # pending candidates as agent-experience docs + same-profile
            # self-confirm (no clarify human gate). Then the existing
            # boundary-triggered clarify for the human-facing partition
            # (decision/walkthrough/config-snippet auto docs).
            self.close_task_boundary(prev_session)
            # A session switch surfaces the PREVIOUS session's still-pending
            # candidates (batching may not have reached its threshold). Only
            # the previous session clarifies on the boundary; the new
            # session's own pending waits for its own batch threshold.
            if self._agg().should_clarify(prev_session, at_boundary=True):
                self._fire_clarify(prev_session, prev_session=prev_session)
                clarify_any = True

        if cand:
            # v2 (2026-10-14): ANY candidate carrying origin_kind=agent
            # (all four v2 classes — pattern / anti-pattern / tool-quirk
            # / premise) bypasses the clarify human gate — staged PENDING
            # and immediately self-confirmed by the same profile (the
            # TR-9 "不需要人工" ruling extended from incident-only to the
            # whole v2 rule set). Gate / near-dup backstop leaves the row
            # in staging + audit (shared with TR-4 / G5). Legacy rule ids
            # from a stale profile override (origin_kind=None) stay on
            # the existing _stage path (staging only; clarify aggregation
            # + human confirm remains the backstop).
            if cand.origin_kind == "agent":
                self._stage_and_self_confirm(cand,
                                             distill_verdict=distill_v)
            else:
                self._stage(cand, distill_verdict=distill_v)
            # Legacy candidates enter the clarify aggregator (human gate);
            # incident candidates are self-confirmed immediately and do NOT
            # surface on the clarify footnote / batch path (no human gate).
            if cand.origin_kind != "agent":
                self._agg().add(cand)

        # Batch trigger for the CURRENT session (boundary is a no-op here —
        # a brand-new session has not yet accumulated its own batch).
        if self._agg().should_clarify(session_id, at_boundary=False):
            self._fire_clarify(session_id)
            clarify_any = True

        reason = ("boundary" if at_boundary and clarify_any else
                  "batch" if clarify_any else
                  "no_match" if cand is None else "below_batch")
        dry_run_log(cand, clarify_triggered=clarify_any, reason=reason)

    def _docid_fn(self):
        import os
        import sys
        _plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if _plugin_dir not in sys.path:
            sys.path.insert(0, _plugin_dir)
        _kb_dir = os.environ.get("KB_HOME", "/root/.hermes/plugins/knowledge-base-search")
        if _kb_dir not in sys.path:
            sys.path.insert(0, _kb_dir)
        from kb_service import _deterministic_docid
        return lambda title, content: _deterministic_docid(
            {"title": title, "content": content})

    def _llm_enabled(self) -> bool:
        """The write-side LLM face is on iff the profile config did not turn
        it off AND the host call surface is importable AND a main_runtime
        snapshot is available. Standalone / no-runtime / enable=false -> the
        regex verdict stands (the frozen plan's fail-soft floor)."""
        block = (self.rules().get("llm") or {})
        if block.get("enable", True) is False:
            return False
        return _HOST_CALL_LLM is not None

    def _main_runtime_live(self) -> Optional[dict]:
        """Lazily capture + cache the host main_runtime snapshot (explicit
        main_runtime, no task key -> host main-model route; zero custom
        provider config, user rule ⑤). None in standalone / no-context."""
        if self._main_runtime is not None:
            return self._main_runtime
        if _norm_main_runtime is None:
            return None
        try:
            self._main_runtime = _norm_main_runtime(None)
        except Exception:
            self._main_runtime = None
        return self._main_runtime or None

    def _maybe_llm_judge(self, session_id: str, cand: "Candidate",
                         text: str) -> "Optional[Candidate]":
        """Budget-gated LLM refine of one regex-flagged candidate.

        Budget: ``llm.per_session`` (default 3) per session; over-cap -> the
        regex verdict stands (no LLM call this turn, the candidate still
        lands PENDING via the regex path — the LLM face is an ENHANCEMENT,
        never the only judge). Any LLM-face failure inside
        ``_llm_judge_candidate`` degrades to the regex ``cand``; a healthy
        ``worth=false`` veto returns None (skip the staging slot).
        Fail-soft: the LLM face must never break the turn loop."""
        limit = int((self.rules().get("llm") or {}).get("per_session", _LLM_BUDGET_DEFAULT))
        used = self._llm_calls_by_session.get(session_id, 0)
        if limit <= 0 or used >= limit:
            return cand  # over-cap / disabled: regex verdict stands (no call)
        rt = self._main_runtime_live()
        # Count the attempt against the budget only when we actually have a
        # runtime to call (a standalone no-runtime turn never burns budget).
        if rt is None:
            return cand
        # budget: count only AFTER a successful LLM call (fail-soft: a host
        # failure must NOT burn the budget slot, so the next turn can retry).
        # The attempt is NOT counted when the call raised.
        self._llm_calls_by_session[session_id] = used  # reset to pre-call value
        try:
            verdict = _llm_judge_candidate(rt, text, cand)
        except Exception:
            return cand  # LLM-face crash (shouldn't happen, but fail-soft)
        if verdict is not cand:  # a real LLM verdict (upgraded or veto) happened
            self._llm_calls_by_session[session_id] = used + 1
        return verdict

    def _distill_enabled(self) -> bool:
        """The distill face is on iff the profile config did not turn it off
        AND the host call surface is importable. Standalone / enable=false
        -> original-content path stands (fail-soft floor)."""
        distill_cfg = (self.rules().get("llm") or {}).get("distill") or {}
        if distill_cfg.get("enable", True) is False:
            return False
        return _HOST_CALL_LLM is not None

    def _maybe_distill(self, session_id: str, cand: "Candidate") -> Optional[dict]:
        """Budget-gated LLM distill of one judge-surviving candidate.

        Independent per-session budget pool (``llm.distill.per_session``,
        default 3, SEPARATE from the judge pool): over-cap / no runtime /
        any failure / a structurally-invalid LLM output -> the caller
        stores the ORIGINAL content through the existing length gate
        (fail-soft: distillation is an enhancement, never the only judge).
        A `drop_reason` non-empty verdict (LLM judged the model already
        knows this) is a HEALTHY outcome — it is returned to the caller
        which stages the doc PENDING with the `dropped` mark, NOT
        silently discarded.
        Returns the verdict dict from kb_distill.distill_content, or None
        when the face is disabled / over-budget / no-runtime (caller treats
        None as "store original, no distill metadata")."""
        if not self._distill_enabled():
            return None
        distill_cfg = (self.rules().get("llm") or {}).get("distill") or {}
        limit = int(distill_cfg.get("per_session", 3))
        used = self._distill_calls_by_session.get(session_id, 0)
        if limit <= 0 or used >= limit:
            return None  # over-cap: original content stands, no LLM call
        rt = self._main_runtime_live()
        if rt is None:
            return None  # standalone / no runtime: original stands
        # Count the attempt against the budget only after a REAL LLM verdict
        # (model_used=True). A host failure / no-runtime turn does NOT consume
        # the slot (consistent with the judge face's fail-soft: a transient
        # host hiccup must not permanently kill the face within the session).
        verdict = _distill_content(rt, cand.title, cand.content)
        if verdict.get("model_used"):
            self._distill_calls_by_session[session_id] = used + 1
        return verdict

    def _topic_enabled(self) -> bool:
        """The LLM topic-classify face is on iff the profile config did
        not turn it off AND the host call surface is importable
        (11-08, owner 方案②). Standalone / enable=false -> the floor
        topic "general" stands (fail-soft floor)."""
        tcfg = (self.rules().get("llm") or {}).get("topic") or {}
        if tcfg.get("enable", True) is False:
            return False
        return _HOST_CALL_LLM is not None

    def _maybe_llm_topic(self, session_id: str,
                         doc: dict) -> Optional[str]:
        """Budget-gated LLM project-topic classification of one built
        agent-experience doc (11-08, owner 方案②).

        Independent per-session budget pool (``llm.topic.per_session``,
        default 3, SEPARATE from the judge + distill pools): over-cap /
        no-runtime / any LLM failure / a reply not in the valid vocab
        -> returns None (the caller keeps the floor topic "general" —
        fail-soft: classification is an enhancement, never the only
        tagger, and must never break the capture/staging path).
        Returns the chosen valid topic tag, or None.
        """
        # Test hook (mirror of _distill_verdicts): pre-resolved topic
        # verdicts keyed by session skip the LLM face entirely (the
        # B4 routing tests' precedent). Checked BEFORE the _topic_enabled()
        # gate: a pinned verdict is a pre-resolved, host-independent
        # classification — it must not be swallowed just because this
        # process happens to lack a live _HOST_CALL_LLM surface (the 3
        # 2026-11-11 pre-existing red tests: pinned verdicts used to be
        # unreachable whenever the standalone/test interpreter had no
        # host LLM import). A falsy value = floor "general".
        pre = getattr(self, "_topic_verdicts", None)
        if pre is not None:
            v = pre.get(session_id)
            return v if v else None
        if not self._topic_enabled():
            return None
        tcfg = (self.rules().get("llm") or {}).get("topic") or {}
        limit = int(tcfg.get("per_session", 3))
        used = self._topic_calls_by_session.get(session_id, 0)
        if limit <= 0 or used >= limit:
            return None  # over-cap / disabled: floor topic stands, no call
        rt = self._main_runtime_live()
        if rt is None:
            return None  # standalone / no runtime: floor stands
        from kb_ingest_rules import TOPIC_VOCAB
        valid = list(TOPIC_VOCAB) + list(_custom_topics())
        text = f"{doc.get('title', '')}\n{str(doc.get('content', ''))[:1500]}"
        topic = _llm_topic_classify(rt, text, valid)
        if topic is not None:
            self._topic_calls_by_session[session_id] = used + 1
        return topic

    def _doc_kind_enabled(self) -> bool:
        """The LLM doc_kind-classify face is on iff the profile config did
        not turn it off AND the host call surface is importable
        (11-08, owner ruling: ingest 自动打标). Standalone / enable=false
        -> doc_kind=None (the "unlabeled" floor) stands (fail-soft)."""
        dcfg = (self.rules().get("llm") or {}).get("doc_kind") or {}
        if dcfg.get("enable", True) is False:
            return False
        return _HOST_CALL_LLM is not None

    def _maybe_llm_doc_kind(self, session_id: str,
                            doc: dict) -> Optional[str]:
        """Budget-gated LLM doc_kind (document-shape) classification of one
        built agent-experience doc (11-08, owner ruling: ingest 自动打标,
        replacing the 2026-10-18 'never pre-assign on the agent capture
        path' governance note).

        Independent per-session budget pool (``llm.doc_kind.per_session``,
        default 3, SEPARATE from the judge + distill + topic pools):
        over-cap / no-runtime / any LLM failure / a reply not in the valid
        vocab -> returns None (the doc's doc_kind stays None, "unlabeled"
        — exactly the pre-change behavior; fail-soft: classification is an
        enhancement, never the only labeler, and must never break the
        capture/staging path). Returns the chosen valid doc_kind, or None.
        """
        # Test hook (mirror of _topic_verdicts): pre-resolved doc_kind
        # verdicts keyed by session skip the LLM face entirely (the
        # B4 routing tests' precedent). Checked BEFORE the
        # _doc_kind_enabled() gate: a pinned verdict is a pre-resolved,
        # host-independent classification — it must not be swallowed
        # just because this process happens to lack a live
        # _HOST_CALL_LLM surface (same fix as _maybe_llm_topic, 2026-11-11).
        # A falsy value = floor None.
        pre = getattr(self, "_doc_kind_verdicts", None)
        if pre is not None:
            v = pre.get(session_id)
            return v if v else None
        if not self._doc_kind_enabled():
            return None
        dcfg = (self.rules().get("llm") or {}).get("doc_kind") or {}
        limit = int(dcfg.get("per_session", 3))
        used = self._doc_kind_calls_by_session.get(session_id, 0)
        if limit <= 0 or used >= limit:
            return None  # over-cap / disabled: floor doc_kind=None stands, no call
        rt = self._main_runtime_live()
        if rt is None:
            return None  # standalone / no runtime: floor stands
        from kb_ingest_rules import DOC_KIND_VOCAB
        valid = list(DOC_KIND_VOCAB)
        text = f"{doc.get('title', '')}\n{str(doc.get('content', ''))[:1500]}"
        kind = _llm_doc_kind_classify(rt, text, valid)
        if kind is not None:
            self._doc_kind_calls_by_session[session_id] = used + 1
        return kind

    def _stage(self, cand: Candidate, distill_verdict: Optional[dict] = None) -> None:
        """Land the candidate PENDING in the isolated staging store.

        Idempotent: same content -> same docid -> StagingStore dedups on
        docid (uuid5 over content, room decision).

        `distill_verdict` (kb_distill.distill_content's return): when a
        distilled body passed the structure gate, that body + its
        anchor_points replace the raw turn text as the doc content (the
        ORIGINAL text is preserved in `original_content` metadata for
        audit). When the verdict carries a `drop_reason` (LLM judged the
        model already knows this), the doc is staged PENDING with a
        `dropped` mark + the reason so the clarify user message can list
        it and the user can object — NOT silently discarded. A None /
        fail-soft verdict stores the ORIGINAL content (no distill
        metadata)."""
        try:
            svc = _service()
            # TR-9: propagate Candidate.origin_kind into the doc dict.
            # incident candidates carry origin_kind="agent" (agent
            # partition, same-profile self-confirm, no clarify gate);
            # the 3 legacy rules (decision/walkthrough/config) carry
            # origin_kind=None, which maps to "auto" (byte-identical to
            # pre-TR-9 behavior). origin_kind is part of the uuid5
            # docid input (same mechanism as Phase 1 TR-1's
            # setdefault("origin_kind","auto")), so a candidate with
            # origin_kind=agent produces a DIFFERENT docid than the
            # same content with origin_kind=auto — expected, since
            # origin_kind is a property of this candidate, not content.
            origin_kind = cand.origin_kind if cand.origin_kind else "auto"
            doc: dict = {
                "title": cand.title,
                "content": cand.content,
                "status": "pending",
                "origin_kind": origin_kind,
                "kb_class": cand.classified_type,
                "source_profile": cand.source_profile,
                "source_session_id": cand.source_session_id,
            }
            if distill_verdict and not distill_verdict.get("use_original"):
                # Healthy distilled body that passed the structure gate.
                doc["content"] = distill_verdict["distilled_content"]
                doc["title"] = distill_verdict["distilled_title"] or cand.title
                doc["original_content"] = cand.content  # audit
                doc["anchor_points"] = distill_verdict.get("anchor_points") or []
            if distill_verdict and distill_verdict.get("drop_reason"):
                # LLM judged the model already knows this: stage it PENDING
                # with the dropped mark so clarify can list it + the user
                # can object. NOT silently discarded (owner decision).
                doc["dropped"] = True
                doc["drop_reason"] = distill_verdict["drop_reason"]
            svc.ingest([doc])
        except Exception as exc:
            # Never let a staging write break the turn loop.
            logger.debug("kb stage failed: %s", exc)

    def _stage_and_self_confirm(self, cand: Candidate,
                                distill_verdict: Optional[dict] = None) -> None:
        """TR-9: land an incident candidate PENDING in staging + immediately
        self-confirm via the same profile (no clarify human gate).

        The incident rule (id=`incident`, 边角经验事实) maps to
        origin_kind="agent" — the agent-experience partition. Per the
        PRD G7/TR-9 owner ruling ("不需要人工"), the agent partition
        bypasses the clarify user gate: the candidate is staged PENDING
        and then confirmed by the same source_profile (confirmed_by=
        "agent"), mirroring the TR-4 task-boundary sink's self-confirm
        path (reused, not reinvented).

        Gate / near-dup backstop (shared with TR-4 / G5): if the
        confirm() hard gate rejects (profile_mismatch, not_found,
        not_pending, near_duplicate, length_gate), the row stays in
        staging + an audit record (agent_confirm_skipped) is written —
        the same human-backstop path as TR-4, no new mechanism.

        Idempotent: the staging store dedups on docid (uuid5 over the
        full doc dict including origin_kind); re-staging the same
        content is a no-op.
        """
        staged_docid: Optional[str] = None
        try:
            # Test hook: allow a fake service (same pattern as TR-4's
            # close_task_boundary _service_override).
            if self._service_override is not None:
                svc = self._service_override
            else:
                svc = _service()
            # Build the doc dict the same way _stage does, then
            # self-confirm immediately after a successful staging.
            origin_kind = cand.origin_kind if cand.origin_kind else "auto"
            doc: dict = {
                "title": cand.title,
                "content": cand.content,
                "status": "pending",
                "origin_kind": origin_kind,
                "kb_class": cand.classified_type,
                "task_type": cand.classified_type,
                "source_profile": cand.source_profile,
                "source_session_id": cand.source_session_id,
            }
            if distill_verdict and not distill_verdict.get("use_original"):
                doc["content"] = distill_verdict["distilled_content"]
                doc["title"] = distill_verdict["distilled_title"] or cand.title
                doc["original_content"] = cand.content
                doc["anchor_points"] = distill_verdict.get("anchor_points") or []
            if distill_verdict and distill_verdict.get("drop_reason"):
                doc["dropped"] = True
                doc["drop_reason"] = distill_verdict["drop_reason"]

            out = svc.ingest([doc])
            staged_ids = out.get("staged") or []
            if not staged_ids:
                # Staging rejected (near-dup / length gate on a pending
                # doc — normally not gated, but the dedup check still
                # applies). The doc stays nowhere; nothing to backstop.
                return
            staged_docid = staged_ids[0]
            # Self-confirm: same profile as the ingest above (the
            # confirm() hard gate enforces source_profile match;
            # mismatch -> reject + audit, staging row persists as the
            # human backstop — exactly the TR-4 / G5 backstop path).
            res = svc.confirm(staged_docid,
                              source_profile=cand.source_profile,
                              confirmed_by="agent")
            if res.get("status") == "success":
                logger.info("kb incident self-confirm: doc_id=%s "
                            "profile=%s", staged_docid,
                            cand.source_profile)
            else:
                try:
                    svc._audit(action="agent_confirm_skipped",
                               doc_id=staged_docid,
                               source_profile=cand.source_profile,
                               confirmed_by="agent",
                               reason_code=res.get("error", "unknown"),
                               session_id=cand.source_session_id)
                except Exception:
                    logger.debug("kb incident self-confirm audit failed: %s",
                                 res.get("error"))
        except Exception as exc:
            # Never let the incident path break the turn loop.
            logger.debug("kb incident stage+self-confirm failed: %s", exc)

    def _fire_clarify(self, session_id: str, prev_session: Optional[str] = None) -> None:
        """Filemem-style clarify notice: STASH + mark the trigger, do not
        inject.

        TUI/desktop/gateway have no CLI queue for ctx.inject_message()
        (no session_key/origin -> the host drops the message with a
        warning), so the write side no longer posts a user message at
        trigger time. Instead the session's pending batch stays queued
        in the ClarifyAggregator and the trigger is recorded in
        _fire_clarify_fired; the read-side pre_llm_call footnotes the
        released batch into the next LLM context (the model then relays
        it to the user via the host clarify tool, consent closing
        through knowledge_base_confirm). Nothing is marked asked here —
        the footnote path marks the batch when it actually surfaces it.

        `prev_session` (boundary path): the PREVIOUS session whose
        remainder this trigger releases; recorded in _last_boundary_prev
        so the footnote — which runs on a later pre_llm_call when
        _last_session has already advanced — can still reach it."""
        self._fire_clarify_fired.add(session_id)
        if prev_session is not None:
            self._last_boundary_prev = prev_session
        # Durable decision-state backstop (owner 2026-10-13): persist the
        # trigger + boundary marker so a gateway restart does NOT silently
        # stop the [知识待确认] footnote (memory _STATES is zeroed on
        # restart; the FACT stays in staging, this keeps the ASK alive).
        try:
            store = _clarify_state_store()
            if store is not None:
                store.mark_fired(session_id)
                store.set_last_boundary_prev(
                    self._last_boundary_prev)
                if self._last_session:
                    store.set_last_session(self._last_session)
        except Exception as exc:
            logger.debug("kb clarify-sidecar fire persist failed: %s", exc)

    def _clarify_footnote(self, session_id: str) -> str:
        """Build the [知识待确认] footnote for this pre_llm_call turn.

        Only TRIGGERED batches surface (filemem-style nudge): a
        session's pending candidates are released when a trigger fired
        for them — the batch threshold (_fire_clarify batch path) or a
        session boundary (_fire_clarify boundary path — the PREVIOUS
        session's below-threshold remainder). Sub-threshold candidates
        that never hit a trigger stay stashed. The merge covers the
        CURRENT session (its batch-threshold release) + state.
        _last_session (the boundary's release of the previous batch),
        deduped by doc_id.

        Anti re-surface: the moment the footnote actually hands the
        batch to the model, the candidates are marked asked — the
        same anti re-ask semantic as the old inject path (a batch is
        asked ONCE until the user's knowledge_base_confirm / veto
        removes the doc from the pending queue; a NEW batch of
        triggers re-askes a NEW batch). Never re-asks the same
        batch on consecutive turns."""
        agg = self._agg()
        pending: list[Candidate] = []
        # (a) the CURRENT session's released batch (its own trigger fired
        # at the batch threshold in the previous post_llm_call).
        if session_id in self._fire_clarify_fired:
            pending = list(agg.pending_for(session_id))
        # (b) the PREVIOUS session's released batch: at the boundary,
        # on_turn fires _fire_clarify(prev_session) when the previous
        # session still had pending candidates (even below its own
        # batch threshold). The current session's own remainder rides
        # out in the same footnote (owner decision 1: 全量复刻). The
        # previous session is taken from _last_boundary_prev (the
        # boundary trigger's prev, captured at on_turn time) — NOT
        # from _last_session, which advances to the CURRENT session the
        # moment on_turn runs.
        prev = self._last_boundary_prev
        if prev is not None and prev in self._fire_clarify_fired:
            for c in agg.pending_for(prev):
                if not any(c.doc_id == p.doc_id for p in pending):
                    pending.append(c)
            # (b) ... the current session's sub-threshold remainder is
            # also included when the boundary has NOT yet released it
            # via its own batch trigger (i.e., it is not in
            # _fire_clarify_fired — the boundary batch covers it).
            if session_id not in self._fire_clarify_fired:
                for c in agg.pending_for(session_id):
                    if not any(c.doc_id == p.doc_id for p in pending):
                        pending.append(c)
        if not pending:
            return ""
        # 噪声候选（纯指令/疑问型短标题，如「推」「那就帮我加」）不上注入面：
        # 红线「噪声永不上判定面」——staging 行留盘供审计，但 clarify 脚注不列。
        # 宁漏勿进：只滤高置信噪声标题，真结论/配置片段照弹。
        pending = [c for c in pending if not is_noise_title(c.title)]
        if not pending:
            return ""
        lines = ["【知识待确认】以下条目是本轮自动识别、尚未确认的知识候选（pending，暂存于 KB staging）："]
        for i, c in enumerate(pending, 1):
            lines.append(f"{i}. [{c.label}] {c.title}（doc_id: {c.doc_id}）")
        lines.append(
            "请在合适时机调用 clarify 工具把上述候选转达给用户，让用户选择"
            "确认存入共享知识库或跳过；用户确认后调 knowledge_base_confirm "
            "工具闭环，用户否决则不调（staging 行保留待审计）。")
        # Anti re-surface: only now, at the moment the batch is actually
        # handed to the model, do we mark it asked.
        agg.mark_asked(session_id, [c.doc_id for c in pending])
        # Durable decision-state backstop (owner 2026-10-13): persist the
        # asked set so a gateway restart does NOT re-ask the same batch.
        try:
            store = _clarify_state_store()
            if store is not None:
                store.mark_asked(session_id, [c.doc_id for c in pending])
                if self._last_session:
                    store.set_last_session(self._last_session)
        except Exception as exc:
            logger.debug("kb clarify-sidecar asked persist failed: %s", exc)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # TR-4: task-boundary experience auto-sinking (G5: staging pending ->
    # same-profile agent self-confirm, NO clarify human gate; gate
    # rejects / confirm failures stay in staging + an
    # agent_confirm_skipped audit record for the human backstop).
    #
    # Trigger: the SESSION BOUNDARY the write-side hook already detects
    # (a new session_id at on_turn closing the previous session — the
    # lowest-cost signal available, reusing the existing detection; no
    # new task-close signal is invented).
    #
    # Extraction source: the PREVIOUS session's FULL candidate list
    # (ClarifyAggregator._buffer — NOT pending_for(), which is drained
    # by the boundary clarify that fires in the SAME turn, AFTER the
    # sink: _fire_clarify marks every candidate asked the moment its
    # batch threshold or the boundary triggers. The agent partition has
    # its OWN sink path and does not share the clarify's dedup
    # bookkeeping — a candidate already surfaced to the user via
    # clarify is still a valid experience record (the user may have
    # refused to keep it; the agent lesson learned from the turn stands
    # on its own).
    #
    # Each extracted doc is structured "带前提条件" (topic / task_type /
    # premise / premise_status) and its body passes the kb_distill
    # structure gate (LLM distillation, fail-soft — a verdict that
    # fails the structure gate is DISCARDED exactly like the existing
    # distill face; original content only re-enters when the gate
    # itself is unreachable). The docs are ingested PENDING into
    # staging (source_profile = the current profile, origin_kind=agent)
    # and then self-confirmed by the SAME profile
    # (confirmed_by="agent"): the confirm() hard gate (source_profile
    # match) is enforced by kb_service.confirm and is untouched — any
    # mismatch / not_found / not_pending result leaves the row in
    # staging + an agent_confirm_skipped audit record.
    # ------------------------------------------------------------------

    def _build_agent_doc(self, cand: "Candidate",
                         session_msgs: Optional[dict],
                         kb_tier: Optional[str] = None) -> Optional[dict]:
        """TR-4: one pending candidate -> one agent-experience doc.

        The "带前提条件" structuring (G3):
          - topic / task_type: the candidate's class + the session's task
            words (topic vocab stays the kb_ingest_rules defaults until
            TR-3's config section lands; task_type = the cue class).
          - premise: the session's own user message(s) — the conditions
            under which this lesson was learned (what the user asked /
            hit); empty -> premise stays unset, premise_status is
            auto-defaulted to True by _normalize_docs and R-6 treats a
            missing premise as needs_verification on next contact.

        note 5 (PRD v2-final): the ``project_ref`` key is written ONLY
        for the archive tier (kb_tier=archive AND cand.project_ref set).
        Experience/transcript docs OMIT the key entirely (never None) —
        a None-valued key would be a no-op under the B1 None-filter but
        is semantically dirty; omitting is the cleaner contract.
        """
        title = str(cand.title or cand.label or "agent 经验")[:200]
        content = str(cand.content or "").strip()
        msgs = session_msgs or {}
        premise = "；".join(
            str(m).strip() for m in (msgs.get("user") or []) if str(m).strip()
        )[:2000] or None
        doc = {
            "title": title,
            "content": content,
            "status": "pending",
            "origin_kind": "agent",
            "kb_class": cand.classified_type,
            "task_type": cand.classified_type,
            "topic": "general",
            "premise": premise,
            # R-6: an empty premise is not a verified premise — mark it
            # needs_verification so the decay gate fail-closes it out of
            # the auto-injection pool (premise_status=True only when the
            # session actually supplied user-message context).
            "premise_status": "true" if premise else "needs_verification",
            "source_profile": cand.source_profile,
            "source_session_id": cand.source_session_id,
        }
        # note 5: the project_ref key is ARCHIVE-ONLY. A non-None
        # project_ref + archive kb_tier -> stamp the key; everything
        # else (experience, transcript, or a missing project_ref)
        # omits the key entirely.
        if kb_tier == "archive" and cand.project_ref:
            doc["project_ref"] = cand.project_ref
        # owner 2026-10-19 (superseded by owner ruling 11-08: ingest 自动打标):
        # agent auto-captured docs NEVER carry a doc_kind label at _build
        # time (the key is deliberately absent here — _normalize_docs then
        # setdefaults doc_kind=None, so B1-None-filter stable). The doc_kind
        # label is now stamped by the LLM write-side face in
        # close_task_boundary (self._maybe_llm_doc_kind) AFTER the doc is
        # built (11-08, owner ruling: ingest 自动打标, replacing the
        # 2026-10-18 'never pre-assign on the agent capture path' governance
        # note — the label is now an LLM ENHANCEMENT with the SAME fail-soft
        # floor: doc_kind stays None on any LLM face failure).
        # An explicit caller-supplied doc_kind (a user ingest with
        # doc_kind set) still lands as-is — no review gate.
        if not (5 <= len(doc["title"]) <= 200):
            doc["title"] = (doc["title"][:5] + "…") if doc["title"] \
                else "agent 经验条目"
        return doc

    def close_task_boundary(self, prev_session_id: str) -> None:
        """TR-4 task-boundary sink: stage + self-confirm the PREVIOUS
        session's pending candidates as agent-experience docs.

        Called by on_turn at a detected session boundary, BEFORE the
        boundary-triggered clarify (the agent partition has NO clarify
        gate — G5/owner ruling). Idempotent per session (a re-fired
        boundary in the same process is a no-op — the aggregator's
        pending list is already drained by the first sink). Fail-soft
        end to end: any exception leaves staging untouched; per-doc
        confirm failures land in the audit trail
        (agent_confirm_skipped + the confirm error code)."""
        if not prev_session_id or prev_session_id in self._agent_sunk_sessions:
            return
        self._agent_sunk_sessions.add(prev_session_id)
        try:
            if self._pending_override is not None:
                pending = list(self._pending_override)
            else:
                # Full candidate list for the session (NOT pending_for —
                # the boundary clarify that follows in the same turn
                # marks the candidates asked; the agent sink does not
                # share the clarify's dedup bookkeeping, see the block
                # comment above).
                pending = [c for c in self._agg()._buffer.get(
                    prev_session_id, [])
                    if c.source_session_id == prev_session_id]
            if not pending:
                return
            msgs = self._session_msgs.get(prev_session_id) or {}
            try:
                if self._service_override is not None:
                    svc = self._service_override
                else:
                    svc = _service()
            except Exception as exc:
                logger.debug("kb task-boundary sink: service unavailable "
                             "(%s); staging stays as the backstop", exc)
                return
            now = _time_utcnow()
            docs: list[dict] = []
            for cand in pending:
                # B4 step ①: independent 4-class anchor regex over the
                # RAW content (zero LLM — the anchor source that stays
                # live when the distill face fails; N2: an empty result
                # is the step ③ transcript-drop signal).
                from kb_service import extract_anchor_points as _ea
                regex_anchors = _ea(cand.content)
                # note 3: carry the captured git remote into the
                # candidate so the doc can route to archive (a None
                # project_ref keeps the doc in experience/transcript).
                if cand.project_ref is None:
                    repo = self._repo_ctx.get(cand.source_session_id)
                    if repo:
                        cand.project_ref = repo.get("project_ref")

                doc = self._build_agent_doc(cand, msgs)
                if doc is None:
                    continue
                # 11-08 LLM project-topic classify (owner 方案②): after the
                # doc is built (title + content present), a budget-gated host
                # LLM pass picks the best topic tag from the valid vocab
                # (built-in TOPIC_VOCAB + custom_topics.jsonl project tags).
                # Fail-soft: None (disabled / over-budget / no-runtime / any
                # LLM failure / out-of-vocab reply) leaves the floor topic
                # "general" that _build_agent_doc stamped — never a broken
                # or missing topic. Only a REAL valid tag overrides.
                llm_topic = self._maybe_llm_topic(prev_session_id, doc)
                if llm_topic is not None:
                    doc["topic"] = llm_topic
                # 11-08 LLM doc_kind classify (owner ruling: ingest 自动打标,
                # replacing the 2026-10-18 'never pre-assign on the agent
                # capture path' governance note): after the doc is built
                # (title + content present), a budget-gated host LLM pass
                # picks the best doc_kind tag from the controlled vocab
                # (DOC_KIND_VOCAB: experience/tutorial/manual/reference/
                # material). Fail-soft: None (disabled / over-budget /
                # no-runtime / any LLM failure / out-of-vocab reply) leaves
                # the floor doc_kind=None that _normalize_docs stamps —
                # never a broken or missing label; the doc lands confirmed
                # with doc_kind=None (the pre-change behavior) when the LLM
                # face does not produce a valid tag. Only a REAL valid tag
                # overrides the floor.
                llm_doc_kind = self._maybe_llm_doc_kind(prev_session_id, doc)
                if llm_doc_kind is not None:
                    doc["doc_kind"] = llm_doc_kind
                # B4 step ②: LLM distill judges experience/transcript
                # (the existing verdict path, budget-gated, fail-soft).
                verdict = None
                # Test hook: pre-resolved verdicts keyed by session
                # (skip the LLM face entirely; used by B4 routing tests).
                pre = getattr(self, "_distill_verdicts", None)
                if pre is not None and prev_session_id in pre:
                    verdict = pre[prev_session_id]
                elif self._distill_enabled():
                    dcfg = (self.rules().get("llm") or {}).get("distill") or {}
                    limit = int(dcfg.get("per_session", 3))
                    used = self._distill_calls_by_session.get(
                        prev_session_id, 0)
                    rt = None
                    if limit > 0 and used < limit:
                        rt = self._main_runtime_live()
                    # truthy guard (NOT `is not None`): the host's
                    # _norm_main_runtime can cache a FALSY empty dict
                    # (no-context snapshot) — that is "no usable runtime"
                    # for the distill LLM face and must NOT trigger a
                    # no_runtime verdict (a no_runtime verdict + empty
                    # regex anchors = a false transcript_drop of a
                    # perfectly good doc, the 11-08 _main_runtime_live
                    # falsy-cache interaction the owner flagged). An
                    # empty dict therefore stands as the fail-soft
                    # original-content path, same as a None runtime.
                    if rt:
                        verdict = _distill_content(rt, doc["title"],
                                                   doc["content"])
                        if verdict.get("model_used"):
                            self._distill_calls_by_session[prev_session_id] = \
                                used + 1
                # B4 step ③: LLM fail + no independent regex anchors
                # = transcript DROP (N2: not staged live, not a pool
                # occupier — only the audit record carries it; stale
                # routing goes to the stale_staging audit state).
                llm_fail = (verdict is not None
                            and verdict.get("use_original")
                            and not verdict.get("model_used"))
                if llm_fail and not regex_anchors:
                    svc._audit(action="transcript_drop",
                               doc_id=cand.doc_id,
                               source_profile=cand.source_profile,
                               kb_tier="transcript",
                               llm_fail=True,
                               reason=verdict.get("reason"),
                               session_id=prev_session_id)
                    logger.debug("kb task-boundary: transcript drop "
                                 "(LLM fail + no anchors) doc_id=%s",
                                 cand.doc_id)
                    continue
                # Healthy verdict body + anchor UNION (regex ∪ LLM; the
                # regex anchors are the source of truth, LLM extras are
                # appended — "LLM fail 时锚点全丢→该归档的没归档" is
                # exactly what the union prevents).
                if verdict and not verdict.get("use_original"):
                    doc["content"] = verdict["distilled_content"]
                    doc["title"] = verdict["distilled_title"] or doc["title"]
                    doc["original_content"] = cand.content
                    from kb_service import union_anchor_points as _ua
                    doc["anchor_points"] = _ua(
                        cand.content,
                        verdict.get("anchor_points") or [])
                else:
                    # LLM fail-soft (original content stands) OR no
                    # distill verdict: stamp the regex-only anchors.
                    if regex_anchors:
                        doc["anchor_points"] = list(regex_anchors)
                # B4 step ② (LLM-healthy path) or step ① routing mark:
                # the kb_tier field is the routing outcome. When the LLM
                # verdict is healthy it carries a tier ("model-misremembers"
                # / "model-knows" / "transcript" — the distill face's own
                # three-tier judgment); when the LLM FAILED (use_original +
                # model_used=False) the tier is derived from the regex
                # anchors: anchors + incident = archive, else experience
                # (the LLM-fail fail-open to archive is FORBIDDEN — an
                # anchor-less LLM-fail doc can only be experience or
                # transcript, never archive).
                if verdict and verdict.get("model_used"):
                    # LLM healthy: its tier judgment is authoritative.
                    _llm_tier = verdict.get("tier")
                    if _llm_tier == "model-misremembers":
                        # The model misremembers — a concrete knowledge
                        # unit worth persisting (the archive tier when
                        # the doc also carries independent anchors).
                        # v2 (2026-10-14): archive eligibility is the
                        # agent partition itself (origin_kind=agent),
                        # not the retired `incident` kb_class key.
                        doc["kb_tier"] = ("archive"
                                          if regex_anchors and
                                          cand.origin_kind == "agent"
                                          else "experience")
                    elif _llm_tier == "transcript":
                        # The LLM judged this a flow log / meeting note:
                        # NOT knowledge, stays in staging as audit-only.
                        doc["kb_tier"] = "transcript"
                    else:
                        # model-knows or unknown tier: experience floor.
                        doc["kb_tier"] = "experience"
                else:
                    # LLM fail (or no verdict): derive from regex anchors.
                    # NO fail-open to archive — an anchor-less LLM-fail
                    # doc can only be experience (the tier the agent
                    # partition self-confirms by default).
                    doc["kb_tier"] = ("archive"
                                      if regex_anchors and
                                      cand.origin_kind == "agent"
                                      else "experience")
                # N2: non-experience routing docs (archive / transcript)
                # are audit rows — they stay in staging as PENDING and
                # are NEVER self-confirmed into the confirmed corpus.
                # The stale_staging mark is the signal that skips the
                # agent self-confirm (the terminal backstop is the note 4
                # TTL sweep, not the clarify batch).
                if doc.get("kb_tier") != "experience":
                    doc["stale_staging"] = True
                doc["agent_extracted_at"] = now
                docs.append(doc)
            if not docs:
                return
            out = svc.ingest(docs)
            # Self-confirm (agent 分区不走 clarify 人工门): same profile as
            # the ingest above, confirmed_by="agent". confirm() enforces
            # the source_profile hard gate; any failure leaves the row in
            # staging + an agent_confirm_skipped audit record (human
            # backstop — 留 staging 等人工兜底). Note: the confirm only
            # touches docs that ACTUALLY staged (ingest is atomic: a
            # near-dup reject would skip staging, and confirming a
            # non-staged row would just log not_found noise) — the
            # staged list from the ingest result is authoritative.
            # N2 routing: docs marked stale_staging=True are audit rows
            # — they stay PENDING in staging and are NOT self-confirmed.
            staged_ids = set(out.get("staged") or [])
            if staged_ids:
                # Map the stamped docids back onto our local doc copies
                # (1:1 order with `docs`: all are pending and
                # status-pending docs never get a reject/reject path, so
                # every local doc corresponds to one staged id in order).
                for d, sid in zip(docs, out["staged"]):
                    d["docid"] = sid
            confirmed: list[str] = []
            for d in docs:
                if d.get("docid") not in staged_ids:
                    continue
                if d.get("stale_staging"):
                    # N2: audit-only row — stay PENDING, no self-confirm.
                    continue
                res = svc.confirm(d["docid"],
                                  source_profile=d.get("source_profile"),
                                  confirmed_by="agent")
                if res.get("status") == "success":
                    confirmed.append(d["docid"])
                else:
                    try:
                        svc._audit(
                            action="agent_confirm_skipped",
                            doc_id=d["docid"],
                            source_profile=d.get("source_profile"),
                            confirmed_by="agent",
                            reason_code=res.get("error", "unknown"),
                            session_id=prev_session_id,
                        )
                    except Exception:
                        logger.debug(
                            "kb agent-confirm-skip audit failed: %s",
                            res.get("error"))
            logger.info("kb task-boundary sink: session=%s staged=%d "
                        "agent-confirmed=%d", prev_session_id, len(docs),
                        len(confirmed))
        except Exception as exc:
            # The sink must never break the turn loop; staging stays as
            # the human backstop.
            logger.debug("kb task-boundary sink failed: %s", exc)

    def remember_turn_msgs(self, session_id: str, user_message: str,
                            assistant_response: str) -> None:
        """TR-4: keep the session's raw messages (bounded) so the boundary
        sink can pull the premise context for its extracted docs."""
        if not session_id or not (user_message or assistant_response):
            return
        rec = self._session_msgs.setdefault(
            session_id, {"user": [], "assistant": []})
        if user_message:
            rec["user"].append(str(user_message)[:2000])
        if assistant_response:
            rec["assistant"].append(str(assistant_response)[:2000])
        for key in ("user", "assistant"):
            while len(rec[key]) > 20:
                rec[key].pop(0)


# ---------------------------------------------------------------------------
# Read-side auto-retrieve (user "每 turn 捞" requirement, frozen plan):
# a CODE-SIDE, ZERO-LLM per-turn fetch. pre_llm_call fires before the host
# assembles the turn; the plugin's local two-layer BM25 (no model, no
# external API) searches the confirmed corpus over the user query.
#   - trivial query (is_trivial_query) -> skip (return {} -> no injection)
#   - 0 hits                            -> skip (return {} -> no injection)
#   - hits present                      -> return {"context": "## 知识库参考\n..."}
#     and the HOST splices that context into the user turn (the host's
#     pre_llm_call context channel — same shape as filemem's
#     filemem_auto_retrieve; the plugin never writes to the system prompt).
# Reviewer acceptance: NO score gating (raw_score=0 / negative are valid
# hits under min-max flattening — relevance is controlled only by max_hits
# truncation + char cap + the service's built-in term-overlap gate).
#
# TR-2 (agent experience-ledger, Phase 2): ON TOP OF the full-corpus
# inject above, a SECOND profile-scoped search with
# filters={origin_kind:agent} (+ implicit source_profile, the same
# precedent as TR-8's _search_handler — kb_service.search() has no notion
# of an implicit filter, the filter is built here at the plugin handler
# layer) is merged into the SAME `## 知识库参考` block, user segment
# first, agent segment after, under a 6:4 per-segment char budget of the
# UNCHANGED _READ_MAX_CONTEXT_CHARS total. The agent segment is gated by
# the decay rule (agent items only): premise_status == "true" AND
# (reference_count > 0 OR age_days <= 14, age from the deterministic
# first-seen ingest_ts sidecar). The gate runs on the search results'
# metadata ONLY (plugin side) — it never touches search()/rrf internals.
# The second fetch is fail-soft: any hiccup drops the agent segment and
# keeps the user segment, the turn loop never breaks.
# This path NEVER reinforces (ReinforceStore write-gate, R-3): a prefetch
# hit is not a model citation, and the agent segment inherits the same
# guarantee — both searches are svc.search() only, no svc.reinforce().
# ---------------------------------------------------------------------------
_READ_MAX_HITS = 3
_READ_MAX_CONTEXT_CHARS = 1500
_READ_SNIPPET_CHARS = 500

# S6 读侧 (owner 定稿 2026-10-22, D1=b / D2=只进本轮不落本地): 本地知识库
# auto-retrieve 召回 < 2 时, 在注入块尾追加一行【提示】(只指路, 绝不灌中央数据),
# 由模型/人工再调 kb_search_central 主动查中央. 结果只活本轮工具返回, 不落本地.


def kb_referral_hint(n: int, enabled: bool = True) -> str:
    """kb auto-retrieve 面的跨设备参考提示行 (对称 filemem _shared_referral_hint).
    enabled=False (shared_memory 未配置) 或 n >= 2 -> 返回 "" (字节级不变)."""
    if not enabled or n >= 2:
        return ""
    return (
        f"\n[提示] 本地知识库相关条目仅 {n} 条; 需参考其他设备的共享知识库, "
        "可调用 kb_search_central (门2共识 / confirmed_only=false 查全量)。"
    )

# TR-2 agent-partition constants (KPI-4): the 6:4 split of the unchanged
# 1500-char total, and the decay gate's 14-day fresh-window threshold.
_AGENT_BUDGET_RATIO = 0.4  # agent segment <= 40% of the total budget
_AGENT_DECAY_MAX_AGE_DAYS = 14


def _build_context_block(results: list, max_chars: int = _READ_MAX_CONTEXT_CHARS) -> str:
    """Compress top-k search hits into a bounded `## 知识库参考` block.

    Each hit contributes its full title + a content snippet (capped at
    _READ_SNIPPET_CHARS per hit); the whole block is hard-capped at
    `max_chars` so a per-turn injection can't bloat the prompt. No score
    field is emitted — relevance ordering is already top_k-sorted by the
    service and the block carries no per-hit score (reviewer rule ③:
    do NOT gate or drop on score)."""
    if not results:
        return ""
    lines: list[str] = []
    budget = max_chars
    for r in results:
        title = str((r.get("title") or "")).strip() or "(untitled)"
        content = str(r.get("content") or "")[:_READ_SNIPPET_CHARS]
        entry = f"【{title}】\n{content}\n"
        lines.append(entry)
    block = "## 知识库参考\n" + "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars].rstrip() + "…"
    return block


def _segment_entries(results: list, max_chars: int) -> list:
    """TR-2 per-segment builder: the entries of one block segment
    (user or agent), each a full title + _READ_SNIPPET_CHARS snippet,
    truncated so the segment's own header + entries stay within
    `max_chars`. Returns [] when the segment would be empty."""
    if not results:
        return []
    entries: list[str] = []
    for r in results:
        title = str((r.get("title") or "")).strip() or "(untitled)"
        content = str(r.get("content") or "")[:_READ_SNIPPET_CHARS]
        entry = f"【{title}】\n{content}\n"
        if len("\n".join(entries + [entry])) > max_chars:
            break
        entries.append(entry)
    return entries


def _segment_header(user: bool) -> str:
    return ("### 用户知识" if user else "### agent 经验")


def _build_segmented_block(user_hits: list, agent_hits: list,
                           max_chars: int = _READ_MAX_CONTEXT_CHARS) -> tuple:
    """TR-2: merge the full-corpus (user) hits and the agent-partition
    hits into ONE `## 知识库参考` block, user segment first, agent
    segment after, under the 6:4 per-segment budget of `max_chars`.

    Budget: user segment <= round(max_chars * 0.6), agent segment
    <= round(max_chars * 0.4), header lines aside — the agent segment
    can never squeeze the user knowledge slot (KPI-4's 600-char hard
    cap on the agent side of the default 1500-char block).

    Returns (block, user_entry_count, agent_entry_count)."""
    agent_budget = int(max_chars * _AGENT_BUDGET_RATIO)
    user_budget = max_chars - agent_budget
    user_entries = _segment_entries(user_hits, user_budget)
    agent_entries = _segment_entries(agent_hits, agent_budget)
    if not user_entries and not agent_entries:
        return "", 0, 0
    parts: list[str] = []
    if user_entries:
        parts.append(_segment_header(True))
        parts.extend(user_entries)
    if agent_entries:
        parts.append(_segment_header(False))
        parts.extend(agent_entries)
    block = "## 知识库参考\n" + "\n".join(parts)
    if len(block) > max_chars:
        block = block[:max_chars].rstrip() + "…"
    return block, len(user_entries), len(agent_entries)


def _agent_in_decay_gate(result: dict, now_ts: float) -> bool:
    """TR-2 decay gate for origin_kind=agent results (user items are
    unconstrained by this gate and never pass through it).

    Passes iff premise_status is the truthy default/spelling ("true")
    AND the item is still fresh or proven referenced:
      reference_count > 0 (an explicit tool-call citation) OR
      age_days <= _AGENT_DECAY_MAX_AGE_DAYS (from the first-seen
      ingest_ts; a missing ts means unknown age -> fail-closed, the
      item waits for a tool-layer citation instead of auto-injecting
      a stale entry).
    premise_status: the Phase-1 setdefault stamps the bool True on
    default-state docs, set_premise_status stores the "true" string;
    both spellings are accepted (canonical truthiness), anything else
    (needs_verification / falsified / absent) decays the item.
    """
    meta = result.get("metadata") or {}
    # superseded_by drop (owner 2026-10-14, PRD §5): a confirmed doc a
    # newer doc superseded is OUT of the auto-inject pool — it stays on
    # disk (explicit get still reaches it) but is no longer auto-recalled.
    # Checked BEFORE the premise_status gate: the two failure reasons are
    # independent (a falsified AND superseded doc fails both, either one
    # alone is enough to drop it).
    if meta.get("superseded_by"):
        return False
    if meta.get("premise_status") not in (True, "true"):
        return False
    # B3 exemption branch (PRD v2-final): project-anchored agent docs
    # (kb_class carries an independent anchor + project_ref + premise
    # held) are PROJECT ARCHIVES and are EXEMPT from the 14-day decay
    # window — they stay in the auto-injection pool long-term (the
    # archive tier's retrieval guarantee). v2 (2026-10-14): the
    # exemption keys off ANY origin_kind=agent doc carrying a
    # project_ref (the old `kb_class == "incident"` key is retired —
    # every v2 class lands in the agent partition now; legacy
    # pre-v2 rows keep matching via origin_kind). The branch sits
    # DELIBERATELY AFTER the premise_status check above (B3's
    # fail-closed ruling: a FALSIFIED premise is never exempted —
    # falsified archives exit the pool via set_premise_status's
    # downgrade path, not via this gate) and BEFORE the
    # reference_count / age checks (the exemption short-circuits both:
    # an archive must not decay just because nobody has cited it yet).
    if (meta.get("origin_kind") == "agent"
            and meta.get("project_ref")
            and meta.get("premise_status") in (True, "true")):
        return True
    ref_count = meta.get("reference_count")
    if ref_count is not None and int(ref_count) > 0:
        return True
    ts = meta.get("ingest_ts")
    if ts is None:
        return False  # unknown first-seen ts -> unknown age -> fail-closed
    age_days = (now_ts - float(ts)) / 86400.0
    return age_days <= _AGENT_DECAY_MAX_AGE_DAYS


def _agent_filters(profile: Optional[str]) -> dict:
    """TR-2 agent search filters: the origin_kind=agent partition plus
    the implicit source_profile scoping (TR-8 precedent: kb_service
    search() is profile-agnostic — the scoping filter is built at the
    plugin handler layer, exactly as _search_handler in __init__.py
    does; a None profile means CLI/global, no scoping filter)."""
    f = {"origin_kind": "agent"}
    if profile is not None:
        f["source_profile"] = profile
    return f


def _is_agent_result(result: dict) -> bool:
    """TR-2: True when a search hit belongs to the origin_kind=agent
    partition (metadata passthrough from the hot/cold doc-dict)."""
    meta = result.get("metadata") or {}
    return meta.get("origin_kind") == "agent"


# ---------------------------------------------------------------------------
# TR-3 (agent experience-ledger, Phase 3): task-topic coarse-filter thin
# TOOL face. NOT part of the pre_llm_call auto-inject layer (the agent
# partition there stays the existing TR-2 behavior; the fourth topic-mass
# RRF axis is TR-6, not wired here):
#   - topic: validated against kb_ingest_rules.TOPIC_VOCAB (first-phase
#     enum, PRD L39 verbatim). Chosen semantics: an out-of-vocabulary
#     topic is a client-side reject (ValueError -> handler's
#     {status: error} payload listing the valid values) — NOT a silent
#     zero-hit search: topic is a curated vocab, so a typo / unknown
#     value means the caller must fix its input, not read "no
#     experience under this topic". Edge rule, pinned by tests: the
#     EMPTY STRING degrades to "no topic filter" (the caller
#     explicitly supplied nothing meaningful — plain agent-partition
#     search), while any other undefined value (including the
#     case-mismatch "M" or whitespace-padded "general ") is a hard
#     reject. topic=None / omitted means "no topic filter" on the
#     tool face (the plain agent-partition search, unchanged
#     TR-2/agent behavior).
#   - origin_kind: this face is the agent partition — the filters dict
#     carries {topic, origin_kind: agent} (+ implicit source_profile
#     scoping, the TR-8/_search_handler precedent, built at the plugin
#     layer; svc.search has no implicit filters) and passes straight to
#     the EXISTING svc.search() filters exact-match gate. No rrf
#     kernel change (TR-6 coder's line), no confirm/_audit/Reinforce
#     store touch, no pre_llm_call path change.
# ---------------------------------------------------------------------------

def _custom_topics() -> tuple:
    """Owner-extensible topic vocab (11-08): one line per project tag in
    kb-data/custom_topics.jsonl. Loaded fresh per call so the owner can
    add a new project tag WITHOUT a code deploy. Fail-soft: missing /
    unreadable file -> (). Case-sensitive (same as TOPIC_VOCAB)."""
    import os as _os
    from pathlib import Path as _Path
    p = _Path(_os.environ.get("KB_DATA_DIR",
                              "/root/.hermes/kb-data")) / "custom_topics.jsonl"
    if not p.exists():
        return ()
    try:
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(line)
        return tuple(out)
    except OSError:
        return ()


def _all_topics() -> tuple:
    """Built-in TOPIC_VOCAB + owner-defined custom_topics.jsonl entries."""
    from kb_ingest_rules import TOPIC_VOCAB
    return tuple(TOPIC_VOCAB) + _custom_topics()


def _topic_search_filters(topic, profile: Optional[str]) -> dict:
    """TR-3 filters dict: the topic exact-match key + the agent
    partition + the implicit source_profile scoping (None profile ->
    CLI/global, no scoping key, same as _agent_filters above).
    topic=None omits the topic key entirely (whole agent partition)."""
    f = {"origin_kind": "agent"}
    if topic:
        f["topic"] = topic
    if profile is not None:
        f["source_profile"] = profile
    return f


def topic_search_service_args(query: str, topic, top_k: int = 5) -> dict:
    """TR-3 tool face: validate `topic` against TOPIC_VOCAB and return
    the EXACT svc.search() kwargs this face issues.

    Chosen semantics, pinned by tests:
      - an out-of-vocabulary topic (typo / new value / case-mismatch /
        whitespace-padded spelling) raises ValueError (the handler maps
        it to a {status: error} payload listing the valid values — the
        caller fixes its input; a silent zero-hit search is explicitly
        NOT the behavior);
      - topic=None or the EMPTY STRING both mean "no topic filter"
        (the plain agent-partition search; the topic key is omitted
        from the filters dict entirely).

    Raises ValueError for a non-empty topic not in the union of
    TOPIC_VOCAB + custom_topics.jsonl (11-08 owner-extensible project
    tags — no deploy needed to add a new project)."""
    t: Optional[str] = None
    if topic is not None:
        topic = str(topic).strip()
        if topic:
            valid = _all_topics()
            if topic not in valid:
                raise ValueError(
                    f"invalid topic {topic!r}: expected one of "
                    f"{list(valid)} "
                    f"(built-in kb_ingest_rules.TOPIC_VOCAB + "
                    f"kb-data/custom_topics.jsonl owner project tags; "
                    f"add a new line to custom_topics.jsonl to extend "
                    f"without a deploy)")
            t = topic
    return {
        "query": str(query or ""),
        "top_k": int(top_k),
        "filters": _topic_search_filters(t, _current_profile()),
        "min_score": 0.0,
        "rerank": True,
    }


def register_read_side(ctx) -> None:
    """Wire the pre_llm_call auto-retrieve hook. Idempotent per ctx."""
    ctx.register_hook("pre_llm_call", _make_read_callback(ctx))


def _rebuild_candidates_from_staging(svc, session_ids: list) -> list:
    """Rebuild Candidate objects for *session_ids* from the Durable staging
    FACT (the owner ruling "从 staging_docs.jsonl 按 doc_id 重读").

    A confirmed doc's staging row was MOVED OUT by confirm() and so is
    naturally absent here — the footnote re-derives only from what is
    still pending, so a settled doc drops out of the ASK automatically.
    Fail-soft: any service hiccup -> [] (the footnote degrades to the
    in-memory-only path; a restart that also lost staging still surfaces
    whatever in-memory state the write side re-accumulated this process).
    """
    out: list = []
    if not session_ids:
        return out
    try:
        rows = svc._staging.docs_snapshot()
    except Exception:
        return out
    want = set(str(s) for s in session_ids)
    for row in rows:
        sid = str(row.get("source_session_id") or "")
        if sid not in want:
            continue
        doc_id = str(row.get("docid") or "")
        if not doc_id:
            continue
        kb_class = row.get("kb_class") or "decision"
        out.append(Candidate(
            doc_id=doc_id,
            classified_type=str(kb_class),
            label=_KBCLASS_LABEL.get(str(kb_class), str(kb_class)),
            title=str(row.get("title") or ""),
            content=str(row.get("content") or ""),
            source_session_id=sid,
            source_profile=row.get("source_profile"),
            cue_hits=[],
            origin_kind=row.get("origin_kind") or None,
        ))
    return out


def _rebuild_state_from_sidecar(ctx, session_id: str) -> Optional[Any]:
    """Post-restart reconstruction of the write-side clarify DECISION state.

    A gateway restart zeroes _STATES (memory). This rebuilds one
    _WriteSideState for *ctx* from (a) the durable sidecar
    (fired / asked / last_session / last_boundary_prev) and (b) the
    Durable staging FACT (the candidate buffer). It registers the rebuilt
    state into _STATES so the read-side footnote path resumes exactly
    where the previous process left off.

    Fail-soft: a missing sidecar / staging service -> None (no footnote;
    the caller's existing behaviour — a stateless read is a clean pass).
    """
    store = _clarify_state_store()
    if store is None:
        return None
    snap = store.snapshot()
    last_session = snap.get("last_session") or session_id or None
    last_boundary_prev = snap.get("last_boundary_prev")

    state = _WriteSideState(ctx)
    # (a) boundary markers
    state._last_session = last_session
    state._last_boundary_prev = last_boundary_prev
    # (b) per-session fired markers -> _fire_clarify_fired
    fired_sessions = [sid for sid, e in (snap.get("sessions") or {}).items()
                      if e.get("fired")]
    for sid in fired_sessions:
        state._fire_clarify_fired.add(sid)
    # (c) candidate buffer from the staging FACT (the owner ruling),
    # scoped to the sessions that fired a trigger (the footnote only
    # releases triggered batches — sub-threshold sessions never asked).
    rebuild_sids = sorted(
        {s for s in ([session_id, last_session, last_boundary_prev]
                     + fired_sessions) if s}
    )
    try:
        svc = _service()
    except Exception:
        svc = None
    if svc is not None:
        cands = _rebuild_candidates_from_staging(svc, rebuild_sids)
        agg = state._agg()
        for c in cands:
            agg.add(c)
        # (d) asked sets (LRU mirror) so a restart does NOT re-ask a batch.
        asked_by_session: Dict[str, list] = {}
        for sid, e in (snap.get("sessions") or {}).items():
            asked = e.get("asked") or []
            if asked:
                asked_by_session[sid] = asked
        for sid, doc_ids in asked_by_session.items():
            state._agg().mark_asked(sid, doc_ids)
    _STATES[id(ctx)] = state
    return state


def _make_read_callback(ctx):
    def _pre_llm_call(**kwargs: Any) -> Optional[dict]:
        """Host pre_llm_call hook: return {"context": ...} to inject the
        knowledge-base block into the user turn, or {} / None to pass.

        The host's pre_llm_call surface passes session_id / user_message as
        kwargs (same shape the write-side post_llm_call callback reads); the
        context returned is spliced into the assembled user turn by the host
        (it is NOT written to the system prompt — the plugin has no write
        path to the host system block by design).

        TR-2: after the existing full-corpus fetch, a second, profile-
        scoped origin_kind=agent search is merged into the same block
        (user segment first, agent after, 6:4 budget) with the agent
        decay gate applied to the agent hits. The agent fetch is
        fail-soft (any hiccup -> user segment only) and NEVER
        reinforces (the two svc.search() calls are read-only)."""
        user_query = str(kwargs.get("user_message") or kwargs.get("query") or "")
        session_id = str(kwargs.get("session_id") or "")
        # Filemem-style clarify footnote (works in TUI/desktop/gateway):
        # the write side queues pending batches in the per-ctx
        # ClarifyAggregator; surface them here so the model can relay
        # them to the user via the host clarify tool. Stateless read
        # calls that never ran the write side have no state -> no
        # footnote (a clean pass). Computed at the TOP of the callback:
        # a trivial query still carries the footnote to the model (the
        # nudge must not be lost on a short ack turn), and the merge
        # over state._last_session stays stable for the whole call.
        footnote = ""
        state = _STATES.get(id(ctx))
        # Post-restart reconstruction (owner 2026-10-13): a gateway restart
        # zeroes _STATES; if this ctx has no in-memory write-side state yet,
        # rebuild it from the durable sidecar + the staging FACT so the
        # [知识待确认] footnote resumes across the restart. Fail-soft: any
        # hiccup -> the original "stateless read = clean pass" behaviour.
        if state is None:
            state = _rebuild_state_from_sidecar(ctx, session_id)
        if state is not None:
            footnote = state._clarify_footnote(session_id)
        # Trivial-query gate (independent, in kb_ingest_rules): a pure ping /
        # < 12 chars / no real content -> no fetch (the clarify footnote,
        # if any, still rides out via the block assembly below).
        if is_trivial_query(user_query):
            return {"context": footnote} if footnote else {}
        # ① Denoise: strip host-path / filesystem noise tokens from the query
        # BEFORE it reaches BM25 (kb_rerank.denoise_query, pure text). A query
        # that pasted a file path makes every "/root"/".hermes"-mentioning doc
        # co-occur; those are noise. Never returns an empty string (falls back
        # to the raw query). THIS PATH NEVER REINFORCES — a prefetch hit is
        # not a model citation, so it must not bump any doc's reference count.
        from kb_rerank import denoise_query
        import time as _time
        search_query = denoise_query(user_query)
        try:
            svc = _service()
            results = svc.search(query=search_query, top_k=_READ_MAX_HITS)
        except Exception as exc:  # a KB hiccup must never break the turn
            logger.debug("kb auto-retrieve search failed: %s", exc)
            return {}
        user_hits = (results or {}).get("data", {}).get("results") or []
        # 噪声 doc（纯指令/疑问型短标题，如「推」「那就帮我加」）不上自动注入面：
        # 红线「噪声永不上判定面」——confirmed corpus 行保留供审计，但 auto-inject 不捞。
        # 宁漏勿进：只滤高置信噪声标题，真经验照带。
        user_hits = [h for h in user_hits if not is_noise_title(h.get("title", ""))]
        # TR-2 agent partition: SECOND, profile-scoped fetch over the
        # origin_kind=agent slice (TR-8 implicit source_profile scoping,
        # built here at the plugin layer — svc.search has no implicit
        # filters). The second fetch is UNCONDITIONAL (the PRD: "额外做
        # filters={origin_kind:agent} 的第二次检索") — an agent doc the
        # plain query already hit is deduped by doc_id (rendered once,
        # in the user segment where it matched; the agent segment only
        # carries docs the plain query MISSED). Fail-soft: any hiccup ->
        # user segment stands alone, the turn loop never breaks.
        agent_hits: list = []
        now_ts = _time.time()
        try:
            agent_results = svc.search(
                query=search_query, top_k=_READ_MAX_HITS,
                filters=_agent_filters(_current_profile()))
            seen = {str(h.get("doc_id")) for h in user_hits}
            agent_hits = [
                h for h in ((agent_results or {}).get("data", {}).get("results") or [])
                if str(h.get("doc_id")) not in seen
                and _agent_in_decay_gate(h, now_ts)
                and not is_noise_title(h.get("title", ""))
            ]
        except Exception as exc:
            logger.debug("kb auto-retrieve agent-partition search failed: %s", exc)
        # Decay gate applied to agent docs that surfaced via the plain
        # fetch too (agent items are gated even in the user segment; user
        # items are unconstrained and never pass through the gate).
        user_hits = [h for h in user_hits if not _is_agent_result(h)
                     or _agent_in_decay_gate(h, now_ts)]
        _sm_enabled = False
        try:
            import kb_shared_memory as _kb_sm
            _sm_enabled = _kb_sm._load_shared_memory_config() is not None
        except Exception:
            _sm_enabled = False
        if not user_hits and not agent_hits and not footnote:
            # 0 local kb hit (S6 D1=b): 仍提示可查中央知识库 (共享=可参考),
            # 提示只进本轮注入块, 不自动灌中央数据. 未启用 shared_memory -> 不提示.
            _hint = kb_referral_hint(0, enabled=_sm_enabled)
            if _hint:
                return {"context": _hint}
            return {}  # no result in either segment -> direct pass
        block, n_user, n_agent = _build_segmented_block(
            user_hits, agent_hits)
        if not block and not footnote:
            return {}
        if footnote:
            block = (block + "\n\n" + footnote) if block else footnote
        if not block:
            return {}
        # S6 读侧 (owner D1=b): 本地知识库 auto-retrieve 总命中 < 2 且
        # shared_memory 已启用 -> 注入块尾追加一行提示 (只指路, 不灌中央数据).
        # 未启用 shared_memory 的机器上 enabled=False -> 恒 "" (字节级不变).
        _total_hits = len(user_hits) + len(agent_hits)
        _sm_enabled = False
        try:
            import kb_shared_memory as _kb_sm
            _sm_enabled = _kb_sm._load_shared_memory_config() is not None
        except Exception:
            _sm_enabled = False
        _hint = kb_referral_hint(_total_hits, enabled=_sm_enabled)
        if _hint:
            block = block + _hint
        # Structured log line: the KPI-4 verification reads the agent-
        # segment hit/char counts out of this line (n_agent_hit /
        # agent_chars fields), alongside the pre-TR-2 total-hit +
        # total-char format (kept: the "N hit(s), M-char block" fields
        # are still first).
        # agent_chars: length of the agent segment (from its header to
        # end of block), not just the offset+header — KPI-4 reads this
        # field to verify the agent section's actual char budget usage.
        agent_chars = (
            (len(block) - block.index("### agent 经验"))
            if "### agent 经验" in block else 0
        )
        logger.debug(
            "kb auto-retrieve: %d hit(s), %d-char block (user: %d hit(s), "
            "agent: %d hit(s), agent_chars=%d)",
            n_user + n_agent, len(block), n_user, n_agent, agent_chars)
        return {"context": block}
    return _pre_llm_call


_STATES: dict[Any, _WriteSideState] = {}


def register_write_side(ctx) -> None:
    """Wire the post_llm_call hook. Idempotent per plugin context."""
    ctx.register_hook("post_llm_call", _make_callback(ctx))


def _make_callback(ctx):
    def _post_llm_call(**kwargs: Any) -> None:
        state = _STATES.setdefault(id(ctx), _WriteSideState(ctx))
        try:
            state.on_turn(
                session_id=str(kwargs.get("session_id", "")),
                assistant_response=str(kwargs.get("assistant_response") or ""),
                user_message=str(kwargs.get("user_message") or ""),
            )
        except Exception as exc:
            # Fail-open: the plugin must never break the agent turn loop.
            logger.debug("kb write-side hook error: %s", exc, exc_info=True)
    return _post_llm_call
