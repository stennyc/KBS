#!/usr/bin/env python3
"""knowledge-base-search distillation face (owner "蒸馏" requirement).

Ingest-side distillation: before a candidate enters the KB corpus, an
LLM (host main-model channel — NO custom model/API/provider config, owner
rule ⑤, same import surface as filemem's consolidation/llm_caller.distill)
reduces raw content to a minimal, verifiable knowledge unit:

  - drop the parts a frontier model already knows cold (public docs,
    ubiquitous API signatures) → the "模型本来就知道" tier sets drop_reason
    and the doc is NOT silently discarded: it is staged PENDING with a
    `dropped` mark so the clarify user message lists it and the user can
    say "no, keep this one".
  - keep the parts the model gets wrong or does not know → distilled to
    "minimal verifiable knowledge units" (version-specific API behaviour,
    internal framework import paths + call examples + known gotchas,
    breaking-change deltas the user actually hit, 禁止 X / 必须 Y
    checklists from a project's coding rules) with concrete, checkable
    detail (version numbers, exact API signatures, specific behaviour
    deltas — NOT vague "added some improvements" phrasing).

Three-tier judgment baked into the prompt (owner 2026-10-11 spec):
  1. 模型大概率知道吗？    → 知道 → drop_reason 非空（不存）
  2. 模型知道但记不准吗？  → 记不准 → 存，且只存它"记不准"的那部分
  3. 模型完全不知道吗？    → 不知道 → 存，蒸馏成最小可验证知识单元

Fail-soft (mirrors filemem llm_caller.distill's "宁可漏注, 不吞脏"):
ANY failure — no main_runtime / host call surface missing / host down /
timeout / unparseable reply / self-contradictory output — coerces to
`use_original=True` (the caller stores the ORIGINAL content through the
existing length gate; the raw text is never lost because distillation
failed). Over-budget (the independent per-session pool, default 3,
separate from the write-side judge pool) is a distinct trigger with the
same fallback (regex verdict + original content), asserted separately in
unit tests.
"""
import json
import logging
import re
from typing import Any, Dict, Optional

try:
    from agent.auxiliary_client import call_llm as _HOST_CALL_LLM
except Exception:  # pragma: no cover - standalone / older-host fallback
    _HOST_CALL_LLM = None

logger = logging.getLogger("knowledge-base.distill")

# KB corpus length gate (must mirror kb_service.py's confirmed-doc gate:
# title 5-200, content 20-10000). A distill output that lands OUTSIDE the
# gate (LLM wrote MORE than it cut, or the raw text was already over the
# content ceiling) is rejected in favour of the ORIGINAL text's gate
# verdict — distillation must never push a doc across a gate the raw
# version would have passed, or hide one it would have failed silently.
_DISTILL_CONTENT_MIN = 20
_DISTILL_CONTENT_MAX = 10000
_DISTILL_TITLE_MIN = 5
_DISTILL_TITLE_MAX = 200

DISTILL_INSTRUCTIONS = (
    "You are a knowledge-base curator. Given raw text that someone wants to "
    "persist into a shared knowledge base, distill it into the MINIMAL "
    "VERIFIABLE knowledge unit worth storing. The model that will later "
    "retrieve it already knows most of the world's public documentation "
    "cold; the value of this entry is the parts it does NOT know or gets "
    "wrong. Decide which tier applies:\n"
    "  1. A frontier model almost certainly already knows this cold "
    "(public, stable, widely-documented API/behaviour with no project-"
    "specific twist): set \"drop_reason\" (one short line) and leave "
    "\"distilled_content\" empty.\n"
    "  2. The model knows it but often gets it wrong or does not know the "
    "specific version/project twist: distill to ONLY the part it gets wrong "
    "or does not know — concrete, checkable detail (exact version, exact "
    "API signature, exact behaviour delta, project-specific import path, "
    "known gotcha, 禁止 X / 必须 Y rule). Vague phrasing like "
    "\"added some improvements\" is FORBIDDEN — it is not retrievable or "
    "verifiable.\n"
    "  3. The model does not know it at all (project-internal fact, "
    "local decision, config snippet that only applies here): distill to a "
    "minimal self-contained unit (2-6 lines, or a short checklist).\n"
    "Never hallucinate — only restate what is in the input. Prefer fewer, "
    "sharper lines over coverage. Output JSON:"
)

DISTILL_SCHEMA = {
    "type": "object",
    "properties": {
        "tier": {"type": "string",
                 "enum": ["model-knows", "model-misremembers", "model-unknown"]},
        "drop_reason": {"type": ["string", "null"],
                        "description": "Non-empty ONLY for tier=model-knows"},
        "distilled_title": {"type": "string"},
        "distilled_content": {"type": "string"},
        "anchor_points": {"type": "array",
                          "items": {"type": "string"},
                          "description": "the 2-5 concrete, checkable "
                                         "anchors (version / API signature / "
                                         "behaviour delta / rule) the entry "
                                         "hinges on; empty for tier=model-knows"},
    },
    "required": ["tier"],
}


def distill_content(main_runtime: Optional[Dict[str, Any]],
                    raw_title: str, raw_content: str) -> Dict[str, Any]:
    """LLM distillation of one raw KB candidate.

    Returns a verdict dict:
      {
        "use_original": bool,       # True -> caller stores the ORIGINAL
                                     # (length-gate check on original) instead
                                     # of the distilled output
        "drop_reason": Optional[str],  # non-empty = LLM says the model
                                        # already knows this; the doc is
                                        # staged `dropped`, NOT silently
                                        # discarded (owner decision)
        "distilled_title": str,
        "distilled_content": str,
        "anchor_points": list,
        "model_used": bool,         # did the host LLM actually run and
                                     # produce a parseable verdict
        "reason": str,              # "ok" | "over_budget:<n>" | "no_runtime"
                                    # | "llm_error:<...>" | "parse_failed"
        "rejected_reason": Optional[str],  # gate rejection, if the LLM
                                           # output was structurally bad
        "tier": Optional[str],
      }

    ANY failure path (no runtime / no host call surface / host down /
    timeout / unparseable / self-contradictory output) returns
    use_original=True + the original content untouched — distillation is
    an ENHANCEMENT, never the only judge; a failed distill must not lose
    the data (owner fail-soft rule). This runs OUTSIDE the caller's
    per-session budget bookkeeping (the caller counts a successful or
    attempted call against `llm.distill.per_session` before invoking here
    — see kb_ingest_hook._maybe_distill).
    """
    verdict: Dict[str, Any] = {
        "use_original": True,
        "drop_reason": None,
        "distilled_title": raw_title,
        "distilled_content": raw_content,
        "anchor_points": [],
        "model_used": False,
        "reason": "ok",
        "rejected_reason": None,
        "tier": None,
    }
    if _HOST_CALL_LLM is None:
        verdict["reason"] = "no_runtime"
        return verdict
    if not isinstance(main_runtime, dict) or not main_runtime:
        verdict["reason"] = "no_runtime"
        return verdict
    try:
        schema_text = json.dumps(DISTILL_SCHEMA, ensure_ascii=False, sort_keys=True)
        user_msg = (
            DISTILL_INSTRUCTIONS
            + "\n\nJSON schema:\n" + schema_text
            + "\nRespond with a single JSON object. No prose, no fences.\n\n"
            + f"Original title:\n{raw_title}\n\nOriginal content:\n{raw_content}"
        )
        route_info: Dict[str, str] = {}
        response = _HOST_CALL_LLM(
            main_runtime=main_runtime,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0.0,
            max_tokens=1024,
            route_info=route_info,
        )
    except Exception as exc:  # host down / timeout / shape: original stands
        verdict["reason"] = f"llm_error:{type(exc).__name__}"
        return verdict
    verdict["model_used"] = True
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
        verdict["reason"] = "llm_error:response_shape"
        return verdict
    m = re.search(r"```(?:json)?\s*(.+?)```", text_out, re.DOTALL | re.IGNORECASE)
    raw = m.group(1).strip() if m else text_out
    try:
        parsed = json.loads(raw)
    except Exception:
        verdict["reason"] = "parse_failed"
        return verdict
    if not isinstance(parsed, dict):
        verdict["reason"] = "parse_failed"
        return verdict
    tier = parsed.get("tier")
    if tier not in ("model-knows", "model-misremembers", "model-unknown"):
        verdict["reason"] = "parse_failed:invalid_tier"
        return verdict
    verdict["tier"] = tier
    anchor_points = parsed.get("anchor_points")
    verdict["anchor_points"] = [str(a) for a in anchor_points] \
        if isinstance(anchor_points, list) else []
    drop_reason = parsed.get("drop_reason")
    if isinstance(drop_reason, str):
        drop_reason = drop_reason.strip() or None
    verdict["drop_reason"] = drop_reason
    d_title = str(parsed.get("distilled_title") or "").strip()
    d_content = str(parsed.get("distilled_content") or "").strip()

    if tier == "model-knows":
        # The model already knows this cold: it is NOT silently discarded —
        # it is staged PENDING with the `dropped` mark so the clarify
        # user message can list it and the user can object. There is no
        # distilled body to check against a length gate (nothing is being
        # stored as a live doc; only the drop_reason is recorded on the
        # staging row).
        verdict["use_original"] = True  # store nothing live; original kept
                                        # for audit only
        verdict["distilled_title"] = d_title or raw_title
        verdict["distilled_content"] = ""
        verdict["reason"] = "ok"
        return verdict

    # tier=model-misremembers / model-unknown: a distilled body MUST exist.
    if not d_content:
        verdict["reason"] = "parse_failed:empty_content"
        return verdict  # use_original stays True
    # Structure gate (mirrors kb_service's confirmed-doc length gate): the
    # distilled body must itself pass the gate to be stored — LLM wrote
    # MORE than it cut, or shrank it below the content floor, or blew the
    # title window -> reject this output, fall back to the ORIGINAL text's
    # gate verdict (the caller re-checks the original against the same
    # gate; a raw-text gate failure is a pre-existing condition, not a
    # regression this face introduced).
    if not (_DISTILL_CONTENT_MIN <= len(d_content) <= _DISTILL_CONTENT_MAX):
        verdict["rejected_reason"] = "content_length_gate"
        return verdict
    title_final = d_title or raw_title
    if not (_DISTILL_TITLE_MIN <= len(title_final) <= _DISTILL_TITLE_MAX):
        verdict["rejected_reason"] = "title_length_gate"
        return verdict
    verdict["use_original"] = False
    verdict["distilled_title"] = title_final
    verdict["distilled_content"] = d_content
    verdict["reason"] = "ok"
    return verdict
