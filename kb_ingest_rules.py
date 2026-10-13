#!/usr/bin/env python3
"""knowledge-base-search write-side: candidate detection + session aggregation.

Room design (2026-10-06, coder room):
  - Write side: a pure rule dict classifies "sinking-worthy" content in
    post_llm_call (3 classes, narrow start: decision / walkthrough /
    config-snippet). NO new LLM call (reviewer red line) — rules only.
  - Candidates land as PENDING docs in the physically-isolated staging
    store via kb_service.ingest(status="pending") + structured dry-run log.
  - Clarify aggregation: at a session boundary OR every 5 candidates, the
    plugin injects ONE user message ("这几条要存吗") listing the pending
    docs — the model/agent then asks the user; user consent flows through
    knowledge_base_confirm(doc_id, source_profile, confirmed_by).
  - Read side: no auto-injection; the knowledge_base_search tool
    description carries the "when to call" guidance (in __init__.py).

Split: this module is the pure logic (rules + aggregation + dry-run).
The hook wiring + tool registration live in __init__.py (single entry).

Config: defaults are embedded here (single source of truth). A profile's
  config.yaml `plugins.entries.knowledge-base-search.kb_ingest.rules`
merges INCREMENTALLY over the defaults (override/add, never replace the
whole dict), so a profile that configures nothing gets the defaults.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("knowledge-base.search")

# ---------------------------------------------------------------------------
# TR-3 topic vocabulary (PRD v2 §TR-3, L39 first-phase enum, verbatim —
# do not invent new values; extend via owner review only). The task-topic
# coarse-filter thin tool face (knowledge_base_search(topic=...))
# validates against THIS config section; an out-of-vocabulary topic is a
# client-side reject, not a silent zero-hit search.
# ---------------------------------------------------------------------------
TOPIC_VOCAB: tuple = (
    "messaging",
    "deployment",
    "data-pipeline",
    "general",
)


# ---------------------------------------------------------------------------
# Read-side trivial-query gate (data-engineer contract, reviewer issue 3):
# an INDEPENDENT 3-line predicate living HERE in the plugin package (NOT an
# import of the host `agent.memory_provider.is_trivial_prompt` — cross-package
# host-internal import would break the plugin's "only touch plugin source"
# red line when the host renames the symbol). A trivial user message (pure
# ping / < 12 chars / no real content token) short-circuits the read-side
# auto-retrieve so a per-turn code-side fetch never burns a search on noise.
# Pure stdlib; no LLM, no network, no host import.
# ---------------------------------------------------------------------------
_TRIVIAL_KEEP_RE = re.compile(r"[^\w\u4e00-\u9fff]")

# ---------------------------------------------------------------------------
# Write-side noise-title filter (owner 2026-10-13: noise candidates MUST NOT
# reach the injection faces — clarify footnote / read-side auto-inject — while
# their staging rows stay on disk for audit). A candidate whose TITLE is a
# bare user command / question head (the "推 / 要 / A / 那就帮我加 / 不要，直接丢"
# shape the detector packages together with an assistant 流水账 tail) is
# noise: the command itself is not knowledge. Conservative — only obvious
# command/question heads are filtered; any real task/verdict/config snippet
# slips through (宁漏勿进).
# ---------------------------------------------------------------------------
_NOISE_CMD_WORDS = {
    # 带语义的指令 / 选项 / 确认短词（宁漏勿进：只列高置信指令词，
    # 孤立单字母 a/b/c/A/B 不入表——太易误伤占位/真内容）。
    "推", "要", "删", "丢", "修", "加", "删掉", "推完了", "落地",
    "继续", "确认", "同意", "可以", "好的", "好", "行", "对", "嗯", "收到",
    "明白", "否", "不", "别", "算了", "不要", "帮我加", "那就帮我加",
    "可以落地", "点头", "批准", "skip", "skip it", "skip that", "run it",
}
_NOISE_Q_MARKERS = ("吗", "呢", "?", "？", "吧")


def is_noise_title(title: str) -> bool:
    """True iff `title` is a bare user command / question head (noise).

    Two legs, both conservative (fail-open: only filter when clearly noise):
      1. command head: after strip, the title (first line) is a single
         instruction word from _NOISE_CMD_WORDS, OR a 1-6 char imperative
         ending in a command verb (帮我/把/别/要 + short verb) with no
         substantive noun payload;
      2. question head: ends in a question marker (吗/呢/？/…) AND the
         content token count is small (< 8) — a short question, not a
         settled answer.
    A real knowledge snippet ("推 stennyc/KBS 前先用 Bearer 认证" / "读侧
    裸 import 崩溃，bootstrap 修复") has substantive payload and returns
    False. Pure function, zero I/O."""
    s = (title or "").strip().splitlines()[0].strip() if title else ""
    if not s:
        return True
    low = s.lower()
    # Leg 1: bare command word
    if s in _NOISE_CMD_WORDS or low in _NOISE_CMD_WORDS:
        return True
    # Leg 2: 1-6 char imperative with no CJK noun payload beyond the verb
    #         (a real task always carries a substantive object / verb chain
    #         that pushes length up; "推"/"那就帮我加" stay short).
    if 1 <= len(s) <= 6 and low in _NOISE_CMD_WORDS | {"那就帮我加", "帮我加", "不要", "别", "算了"}:
        return True
    # Leg 3: short question head
    if s.endswith(_NOISE_Q_MARKERS) and len(s) <= 16:
        content = _TRIVIAL_KEEP_RE.sub("", s)
        if len(content) < 8:
            return True
    return False


def is_trivial_query(text: str) -> bool:
    """True iff `text` is too thin to be worth a knowledge-base search.

    Two legs, both conservative (fail-open: only skip when clearly empty):
      1. < 12 stripped characters — pure pings / acks / one-word nudges;
      2. stripping all non-content chars (latin/digit word chars + CJK)
         leaves < 3 content characters — punctuation / filler only.
    A real question or query (>= 12 chars with >= 3 content tokens) returns
    False so the read side fetches. Pure function, zero I/O."""
    s = (text or "").strip()
    if len(s) < 12:
        return True
    content = _TRIVIAL_KEEP_RE.sub("", s)
    return len(content) < 3

# ---------------------------------------------------------------------------
# Embedded default rules (reviewer: narrow start, 3 highest-value classes).
# A profile's `plugins.entries.knowledge-base-search.kb_ingest.rules` merges
# incrementally over these; defaults still apply when a profile configures
# nothing. FAQ / "config docs" deliberately excluded until noise is measured.
# ---------------------------------------------------------------------------
DEFAULT_RULES: dict = {
    "enabled": True,
    # Trigger: at a session boundary OR when this many candidates have
    # accumulated for the session, whichever comes first.
    "clarify_batch_size": 5,
    # The 3 narrow classes (reviewer "先窄后宽"). Each rule: a label plus
    # the keyword/regex cues that mark a turn as carrying that knowledge.
    "rules": [
        {
            "id": "decision",
            "label": "决策结论",
            # "拍板 / 决定 / 最终采用 / 不做 X" style close-out lines.
            "cues": [
                r"决定采用", r"拍板", r"最终方案", r"不做", r"否决",
                r"我们决定", r"结论是", r"approved", r"decided",
            ],
        },
        {
            "id": "walkthrough",
            "label": "踩坑排查路径",
            # error -> root-cause -> fix narratives.
            "cues": [
                r"根因", r"排查", r"报错.*解法", r"修复了", r"踩坑",
                r"workaround", r"root cause", r"fixed by",
            ],
        },
        {
            "id": "config",
            "label": "可复用配置片段",
            # reusable snippets / exact config blocks worth re-using.
            "cues": [
                r"```", r"配置片段", r"config\b", r"environment",
                r"settings\.yaml", r"plugin\.yaml", r"requirements\.txt",
            ],
        },
        {
            "id": "incident",
            "label": "边角经验事实",
            # TR-9 (G7): task-incidental "边角事实" — a small fact or
            # gotcha learned in passing during a task, NOT a settled
            # decision / a multi-step walkthrough / a reusable config
            # snippet. Examples: "git PAT 当 Bearer 调 REST 可以，gh
            # 却没登录" (an API corner-case fact), "这个接口的
            # 顺带发现一个限制" (a boundary fact discovered as a
            # by-product). The cues are phrases that mark such
            # incidental observations, distinct from the 3 legacy
            # classes' close-out / narrative / snippet patterns.
            #
            # Why these specific cues:
            #   - "顺带" / "顺便" / "incidental" / "by the way":
            #     direct markers of "learned in passing", not a focused
            #     decision or config.
            #   - "发现一个" / "发现了一个" / "discovered that":
            #     a small factual observation, not a root-cause
            #     narrative (walkthrough) or a verdict (decision).
            #   - "边角" / "corner case" / "gotcha":
            #     explicit "edge fact" vocabulary.
            #   - "API 细节" / "API 边角":
            #     API-specific micro-facts (e.g. auth quirks, rate
            #     limits) that don't fit the 3 legacy classes.
            #   - "限制" / "限制是" / "limitation":
            #     a learned boundary/limit fact, not a fix narrative.
            "cues": [
                r"顺带", r"顺便", r"incidental", r"by the way",
                r"发现一个", r"发现了一个", r"discovered that",
                r"边角", r"corner case", r"gotcha",
                r"API 细节", r"API 边角", r"limitation",
            ],
        },
    ],
    # A turn must contain at least one cue from a rule to be a candidate.
    "min_cue_hits": 1,
    # Write-side LLM face (user "由 LLM 判断" requirement): a budget-gated
    # host-LLM refine that runs ONLY on a regex-flagged candidate (the regex
    # detector is the front gate — LLM never runs on a non-candidate turn,
    # 避免烧预算). enable=False (or a missing block) -> pure-regex legacy.
    # per_session: how many LLM calls per session before the regex verdict
    # stands (0 disables the face entirely); the host main-model channel is
    # the ONLY model route — no custom model/API/provider config (rule ⑤).
    "llm": {
        "enable": True,
        "per_session": 3,
        # Independent distill pool (owner decision via reviewer): judge +
        # distill must not share one 3-call pool. enable/per_session here
        # control the LLM JUDGE face; the nested `distill` block controls
        # the LLM DISTILL face (same host channel, separate budget).
        "distill": {
            "enable": True,
            "per_session": 3,
        },
    },
}


def merge_rules(overrides: Optional[dict]) -> dict:
    """Incrementally merge a profile's `kb_ingest` config block over DEFAULT_RULES.

    The config shape (devops contract):
      plugins.entries.knowledge-base-search.kb_ingest:
        rules:          # the override block — everything here merges over defaults
          enabled: ...
          clarify_batch_size: ...
          min_cue_hits: ...
          rules: [...]  # optional: wholesale replace the cue-rule list
    So `overrides` is the `kb_ingest` dict, and the actual overrides live
    under `overrides["rules"]` (or, for direct callers, at the top level —
    both shapes are honored). Merge-not-replace: only the keys the override
    supplies change; un-mentioned defaults survive. The nested `rules`
    (cue list) is replaced wholesale when supplied.
    """
    if not overrides:
        return copy.deepcopy(DEFAULT_RULES)
    # Accept both `kb_ingest: {rules: {...}}` (config path) and the flat
    # `{...}` (direct-call path).
    raw = overrides.get("rules")
    block = raw if isinstance(raw, dict) else overrides
    assert block is not None
    out = copy.deepcopy(DEFAULT_RULES)
    for key in ("enabled", "clarify_batch_size", "min_cue_hits"):
        if key in block:
            out[key] = block[key]
    if "rules" in block:
        out["rules"] = copy.deepcopy(block["rules"])
    # LLM face config (write-side LLM judge + distill): a nested `llm`
    # sub-block in the profile override merges INCREMENTALLY over the
    # default llm block (a profile that sets only llm.per_session keeps the
    # default enable=True, and vice versa — one-sided overrides survive).
    # The nested `llm.distill` block merges the same way (independent
    # budget pool — owner decision via reviewer).
    if "llm" in block:
        llm_override = block.get("llm")
        base_llm = out.get("llm") if isinstance(out.get("llm"), dict) else {}
        if isinstance(llm_override, dict):
            merged_llm = dict(base_llm)
            merged_llm.update(llm_override)
            # Nested distill sub-block: merge key-wise (not wholesale
            # replace) so a profile setting only llm.distill.per_session
            # keeps the default llm.distill.enable=True.
            distill_ovr = merged_llm.get("distill")
            distill_base = base_llm.get("distill")
            if isinstance(distill_ovr, dict) or isinstance(distill_base, dict):
                merged_distill = dict(distill_base or {})
                if isinstance(distill_ovr, dict):
                    merged_distill.update(distill_ovr)
                merged_llm["distill"] = merged_distill
            out["llm"] = merged_llm
        else:
            out["llm"] = copy.deepcopy(llm_override)
    return out


@dataclass
class Candidate:
    """One sinking-worthy turn classified by the rule engine."""
    doc_id: str                 # deterministic content-addressed id (from kb_service)
    classified_type: str        # "decision" | "walkthrough" | "config" | "incident"
    label: str
    title: str
    content: str
    source_session_id: str
    source_profile: Optional[str] = None
    cue_hits: list = field(default_factory=list)
    # TR-9: origin_kind marker — "agent" when the rule hit is a task-incidental
    # "边角经验事实" (incident), None for the 3 legacy rules (decision /
    # walkthrough / config) which stay the auto-partition default.
    # Backward-compatible: existing callers that don't pass origin_kind get
    # None, which the write-side hook treats as "legacy auto" (no change).
    origin_kind: Optional[str] = None
    # PRD v2-final TR-1 (note 3): project repo reference — a controlled
    # vocabulary of git remote names ("repo:stennyc/KBS" etc.). Set by the
    # write-side repo-context capture (kb_ingest_hook._WriteSideState._repo_ctx,
    # on_turn fail-soft) and carried through the Candidate so
    # _build_agent_doc can stamp it on the doc dict. None (default) keeps
    # every existing caller byte-identical (the origin_kind backward-compat
    # precedent); a None project_ref means the doc never upgrades to the
    # archive tier (git unavailable -> experience/transcript routing).
    project_ref: Optional[str] = None


class CandidateDetector:
    """Pure rules engine: scan a turn's text, emit candidates.

    Stateless per-instance; thread-safe (no shared mutable state). The
    docid is derived by the caller via kb_service._deterministic_docid so
    re-ingesting the same content is idempotent (room decision: uuid5 over
    content, no schema migration needed).
    """

    # TR-9: the rule id that maps to the agent origin_kind partition.
    # Only `incident` (边角经验事实) carries origin_kind=agent; the 3
    # legacy rules (decision / walkthrough / config) stay origin_kind=None
    # (auto-partition default) — backward-compatible, no behavior change.
    _AGENT_RULE_IDS = frozenset({"incident"})

    def __init__(self, rules: dict, session_id: str, profile: Optional[str]):
        self._rules = rules
        self._session_id = session_id
        self._profile = profile
        import re
        self._re = re
        # Pre-compile the cue patterns for the active rules.
        self._compiled = [
            (r["id"], r.get("label", r["id"]),
             [re.compile(c, re.IGNORECASE) for c in r.get("cues", [])])
            for r in rules.get("rules", [])
        ]
        self._min_hits = int(rules.get("min_cue_hits", 1))
        self._enabled = bool(rules.get("enabled", True))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def scan(self, text: str, docid_fn) -> Optional[Candidate]:
        """Return a Candidate if *text* matches an active rule, else None.

        `docid_fn` maps the (title, content) pair to a deterministic id so
        the caller can make ingest idempotent.
        """
        if not self._enabled or not text:
            return None
        for rule_id, label, pats in self._compiled:
            hits = [p.pattern for p in pats if p.search(text)]
            if len(hits) >= self._min_hits:
                # title: first line / first 60 chars; content: the turn text.
                first_line = text.strip().splitlines()[0][:80] if text.strip() else ""
                title = first_line or f"kb:{rule_id}"
                # TR-9: incident rule -> origin_kind="agent" (agent partition,
                # same-profile self-confirm, no clarify human gate). The 3
                # legacy rules stay origin_kind=None (auto default, unchanged).
                ok = "agent" if rule_id in self._AGENT_RULE_IDS else None
                return Candidate(
                    doc_id=docid_fn(title, text),
                    classified_type=rule_id,
                    label=label,
                    title=title,
                    content=text,
                    source_session_id=self._session_id,
                    source_profile=self._profile,
                    cue_hits=hits,
                    origin_kind=ok,
                )
        return None


class ClarifyAggregator:
    """Session-scoped candidate buffer + clarify trigger policy.

    Trigger (room decision, reviewer line): clarify at a SESSION BOUNDARY
    OR when `clarify_batch_size` candidates have accumulated, whichever
    comes first. Behavior is asserted in tests (not just mock-calls) —
    `should_clarify()` is a pure function of state so the two paths are
    unit-testable without a live LLM.
    """

    def __init__(self, batch_size: int = 5, asked_cap: int = 500):
        self._batch_size = max(1, int(batch_size))
        self._buffer: dict[str, list[Candidate]] = {}
        # Bounded LRU of already-clarified doc_ids: a week-long session must
        # not grow this without bound (reviewer nit). The OrderedDict keeps
        # the most-recently-asked ids; oldest evict first. An evicted id can
        # in theory re-clarity — acceptable, a doc re-asked after weeks is
        # not a correctness hazard, and the staging store dedups on docid.
        self._asked: "OrderedDict[str, None]" = OrderedDict()
        self._asked_cap = max(0, int(asked_cap))

    def add(self, cand: Candidate) -> None:
        self._buffer.setdefault(cand.source_session_id, []).append(cand)

    def pending_for(self, session_id: str) -> list[Candidate]:
        """Candidates not yet covered by a clarify ask for this session."""
        return [c for c in self._buffer.get(session_id, [])
                if c.doc_id not in self._asked]

    def should_clarify(self, session_id: str, at_boundary: bool = False) -> bool:
        """True when a clarify should fire for *session_id*.

        Fires when (a) at a session boundary and there are pending
        candidates, or (b) the pending count reaches batch_size. The two
        trigger paths are independent and both unit-tested.
        """
        pending = self.pending_for(session_id)
        if at_boundary and pending:
            return True
        if len(pending) >= self._batch_size:
            return True
        return False

    def mark_asked(self, session_id: str, doc_ids: list[str]) -> None:
        """Record that a clarify ask covered these docs (idempotent, bounded).

        Most-recently-asked stay; the oldest evict once the cap is hit.
        """
        for doc_id in doc_ids:
            self._asked[doc_id] = None
            self._asked.move_to_end(doc_id)
        while len(self._asked) > self._asked_cap:
            self._asked.popitem(last=False)

    def build_clarify_message(self, session_id: str) -> str:
        """The single user-facing message listing the pending candidates.

        One batched clarify per trigger (room: '攒到边界一次性问'), not one
        ask per candidate — avoids the 'ask 8x a day' annoyance.
        """
        pending = self.pending_for(session_id)
        if not pending:
            return ""
        lines = ["以下这几条本轮识别为可沉淀的知识，要存入共享知识库吗？(回 '存' / '跳过')"]
        for i, c in enumerate(pending, 1):
            lines.append(
                f"{i}. [{c.label}] {c.title}\n   doc_id: {c.doc_id}"
            )
        return "\n".join(lines)


def dry_run_log(candidate: Optional[Candidate], clarify_triggered: bool,
                reason: str) -> None:
    """Structured dry-run record (room: threshold tuning needs the data).

    One JSON line: candidate_count, classified_type, clarify_triggered,
    reason. Emitted even when no candidate matched, so silence is
    measurable. No model, no network — local logging only.
    """
    rec = {
        "ts": _utcnow(),
        "candidate_count": 1 if candidate else 0,
        "classified_type": candidate.classified_type if candidate else None,
        "clarify_triggered": bool(clarify_triggered),
        "reason": reason,
    }
    if candidate:
        rec["doc_id"] = candidate.doc_id
        rec["source_session_id"] = candidate.source_session_id
    logger.debug("kb_dry_run %s", json.dumps(rec, ensure_ascii=False))


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
