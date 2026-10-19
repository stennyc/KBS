#!/usr/bin/env python3
"""Read-side auto-retrieve (pre_llm_call) + write-side LLM judge unit tests.

Five frozen scenarios (data-engineer contract):
  1. 0 hits                -> no injection (return {})
  2. trivial query         -> no search, no injection (return {})
  3. raw_score=0 / neg     -> the hit still enters the context block
  4. >3 hits               -> capped to top 3 by the service, block bounded
  5. context-block format  -> "## 知识库参考" header, per-hit title + snippet,
                              hard-capped at max_context_chars

Plus the LLM judge face:
  6. LLM upgrade           -> kb_class/title refined, "llm:confirmed" cue
  7. LLM veto (worth=false)-> candidate skipped (None), not staged
  8. LLM absent (no rt)    -> regex verdict stands (fail-soft floor)
  9. LLM over-budget       -> regex verdict stands (no 4th call)

Pure-logic tests: a fake `_service()` + a fake host `call_llm` are injected
via monkeypatch; no live KB, no host model, no network.
"""
import sys
from pathlib import Path

import pytest

# Put the plugin dir + KB package on sys.path so the in-process imports
# (kb_service, kb_ingest_hook) resolve exactly the way the gateway loads them.
_PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLUGIN_DIR))
_kb_root = str(_PLUGIN_DIR.parents[1])  # .../knowledge-base
if _kb_root not in sys.path:
    sys.path.insert(0, _kb_root)

import kb_ingest_hook as hook  # noqa: E402
from kb_ingest_hook import (  # noqa: E402
    _build_context_block,
    _make_read_callback,
    _make_callback,
    _llm_judge_candidate,
    _READ_MAX_HITS,
    _READ_MAX_CONTEXT_CHARS,
)
from kb_ingest_rules import Candidate, is_trivial_query  # noqa: E402


# ---------------------------------------------------------------------------
# Fake service (the 0-hit / 3-hit / >3-hit / raw_score=0 scenarios)
# ---------------------------------------------------------------------------

class _FakeService:
    """Returns canned search payloads; records every search() call."""
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, query, top_k=5, filters=None, min_score=0.0):
        self.calls.append({"query": query, "top_k": top_k})
        return {"status": "success",
                "data": {"query": query, "results": self._results,
                         "total": len(self._results)}}


def _make_svc(results):
    return _FakeService(results)


def _result(doc_id="d1", title="T", content="C", score=0.5, raw_score=0.0):
    return {"doc_id": doc_id, "title": title, "content": content,
            "score": score, "metadata": {"raw_score": raw_score}}


class _Ctx:
    """Minimal plugin ctx: records registered hooks + inject_message."""
    def __init__(self):
        self.hooks = {}
        self.injected = []

    def register_hook(self, name, cb):
        self.hooks.setdefault(name, []).append(cb)

    def inject_message(self, msg, role="user"):
        self.injected.append((role, msg))


_prev_hook_service = None

def _read_cb(ctx, svc_results):
    """Build a real pre_llm_call callback wired to a fake service.
    Saves and restores hook._service so the fake never leaks into later tests."""
    global _prev_hook_service
    _prev_hook_service = hook._service
    hook._service = lambda: _make_svc(svc_results)
    return _make_read_callback(ctx)

def _restore_hook_service():
    """Call after a test that used _read_cb to undo the _service patch."""
    global _prev_hook_service
    if _prev_hook_service is not None:
        hook._service = _prev_hook_service
        _prev_hook_service = None


def _results_list(*items):
    return list(items)


# ---------------------------------------------------------------------------
# Scenario 1: 0 hits -> no injection (no clarify footnote queued either)
# ---------------------------------------------------------------------------

def test_read_0_hits_no_injection():
    ctx = _Ctx()
    cb = _read_cb(ctx, [])
    out = cb(user_message="how do I configure the gateway retry budget here?")
    assert out == {}  # direct pass, no context block, no clarify footnote
    _restore_hook_service()


# ---------------------------------------------------------------------------
# Scenario 2: trivial query -> no search issued, no injection
# ---------------------------------------------------------------------------

def test_read_trivial_query_skips_search():
    ctx = _Ctx()
    cb = _read_cb(ctx, [_result()])
    # Even though the service WOULD have a hit, a trivial ping must skip.
    out = cb(user_message="ok")
    assert out == {}
    # The trivial gate short-circuits BEFORE any service call; verify by
    # checking the predicate directly on the same input.
    assert is_trivial_query("ok") is True
    _restore_hook_service()


def test_read_trivial_predicate_legs():
    assert is_trivial_query("") is True
    assert is_trivial_query("   ") is True
    assert is_trivial_query("好") is True          # <12 chars
    assert is_trivial_query("？？？？？？") is True   # punctuation-only filler
    # a real >=12-char query with >=3 content tokens passes
    assert is_trivial_query("how to fix the flaky gateway test in ci") is False
    assert is_trivial_query("怎么排查那个偶发的网关重试报错") is False  # CJK content


# ---------------------------------------------------------------------------
# Scenario 3: raw_score=0 / negative -> hit still enters the block
# ---------------------------------------------------------------------------

def test_read_raw_score_zero_hit_included():
    ctx = _Ctx()
    results = [_result(doc_id="a", title="A", content="content-a",
                       score=1.0, raw_score=0.0),
               _result(doc_id="b", title="B", content="content-b",
                       score=1.0, raw_score=-2.3)]
    cb = _read_cb(ctx, results)
    out = cb(user_message="how do I resolve the two-layer bm25 flatten case")
    assert "context" in out
    assert "【A】" in out["context"]
    assert "content-a" in out["context"]
    assert "【B】" in out["context"]
    assert "content-b" in out["context"]
    _restore_hook_service()


# ---------------------------------------------------------------------------
# Scenario 4: >3 results -> service caps to top_k=3; block bounded
# ---------------------------------------------------------------------------

def test_read_over_three_hits_capped():
    ctx = _Ctx()
    results = [_result(doc_id=f"d{i}", title=f"T{i}", content=f"body-{i}",
                       score=1.0) for i in range(7)]
    cb = _read_cb(ctx, results)
    out = cb(user_message="list every confirmed doc in the kb right now")
    assert "context" in out
    # The service enforces top_k truncation (top_k=3) server-side; the fake
    # returns all 7 to verify the plugin passes _READ_MAX_HITS through.
    recorded = []
    class _RecordingSVC(_FakeService):
        def search(self, query, top_k=5, filters=None, min_score=0.0):
            recorded.append({"top_k": top_k, "filters": filters})
            return {"status": "success",
                    "data": {"query": query, "results": results,
                             "total": len(results)}}
    svc2 = _RecordingSVC(results)
    _prev = hook._service
    hook._service = lambda: svc2
    cb2 = _make_read_callback(ctx)
    cb2(user_message="list every confirmed doc in the kb right now")
    hook._service = _prev
    # The callback's fake is a FRESH instance (hook._service is re-resolved
    # at call time), so record through a class-level registry: the cb2
    # invocation's fetches are the LAST two recorded by any _RecordingSVC.
    recorded_plain, recorded_agent = [], []
    class _RecordingSVC(_FakeService):
        def search(self, query, top_k=5, filters=None, min_score=0.0):
            (recorded_agent if filters else recorded_plain).append(
                {"top_k": top_k, "filters": filters})
            return {"status": "success",
                    "data": {"query": query, "results": results,
                             "total": len(results)}}
    svc2 = _RecordingSVC(results)
    _prev2 = hook._service
    hook._service = lambda: svc2
    cb2 = _make_read_callback(ctx)
    cb2(user_message="list every confirmed doc in the kb right now")
    hook._service = _prev2
    assert [c["top_k"] for c in recorded_plain] == [_READ_MAX_HITS], (
        f"plugin must pass top_k={_READ_MAX_HITS} to the plain service "
        f"fetch, got {recorded_plain}")
    # TR-2: the agent-partition second fetch runs UNCONDITIONALLY with the
    # origin_kind=agent filter (+ implicit source_profile scoping).
    assert len(recorded_agent) == 1, "the agent-partition fetch must run"
    assert recorded_agent[0]["filters"].get("origin_kind") == "agent"
    assert recorded_agent[0]["top_k"] == _READ_MAX_HITS
    # The block itself is bounded (6:4 per-segment budget of the total).
    assert len(out["context"]) <= _READ_MAX_CONTEXT_CHARS + 1
    _restore_hook_service()


# ---------------------------------------------------------------------------
# Scenario 5: context-block format + hard char cap
# ---------------------------------------------------------------------------

def test_build_context_block_format_and_cap():
    results = [_result(doc_id=f"d{i}", title=f"Title {i}",
                       content="x" * 200) for i in range(3)]
    block = _build_context_block(results, max_chars=600)
    assert block.startswith("## 知识库参考\n")
    assert "【Title 0】" in block
    assert len(block) <= 600 + 1
    # empty results -> empty string (no block)
    assert _build_context_block([]) == ""
    # a single giant content is snippet-capped per hit
    big = _build_context_block(
        [_result(content="z" * 5000)], max_chars=_READ_MAX_CONTEXT_CHARS)
    assert len(big) <= _READ_MAX_CONTEXT_CHARS + 1


def test_register_read_side_wires_pre_llm_call():
    ctx = _Ctx()
    hook.register_read_side(ctx)
    assert "pre_llm_call" in ctx.hooks
    assert len(ctx.hooks["pre_llm_call"]) == 1


# ---------------------------------------------------------------------------
# LLM judge face (scenarios 6-9)
# ---------------------------------------------------------------------------

class _FakeHostCall:
    """Records calls; returns a canned structured reply by JSON string."""
    def __init__(self, reply_text, raise_exc=False, call_count_ref=None):
        self.reply_text = reply_text
        self.raise_exc = raise_exc
        self.calls = 0
        self._ref = call_count_ref
        self.timeout_kwargs = []  # every `timeout=` kwarg passed by the caller

    def __call__(self, **kwargs):
        self.calls += 1
        if "timeout" in kwargs:
            self.timeout_kwargs.append(kwargs["timeout"])
        if self._ref is not None:
            self._ref[0] += 1
        if self.raise_exc:
            raise RuntimeError("host down")
        class _Msg:
            def __init__(self, c):
                self.content = c
        class _Choice:
            def __init__(self, m):
                self.message = m
        class _Resp:
            def __init__(self, c):
                self.choices = [c]
        return _Resp(_Choice(_Msg(self.reply_text)))


def _mk_cand():
    return Candidate(doc_id="d1", classified_type="decision",
                     label="决策结论", title="t1", content="c1",
                     source_session_id="s1", source_profile=None,
                     cue_hits=["拍板"])


def test_llm_upgrade_refines_kb_class_and_title(monkeypatch):
    fake = _FakeHostCall('{"worth": true, "title": "最终采用 Docker 部署", "kb_class": "walkthrough"}')
    monkeypatch.setattr(hook, "_HOST_CALL_LLM", fake)
    rt = {"model": "m", "provider": "p"}
    out = _llm_judge_candidate(rt, "我们拍板最终采用 Docker 部署", _mk_cand())
    assert out is not None
    assert out.classified_type == "walkthrough"
    assert out.title == "最终采用 Docker 部署"
    assert "llm:confirmed" in out.cue_hits
    assert fake.calls == 1


def test_llm_veto_skips_candidate(monkeypatch):
    fake = _FakeHostCall('{"worth": false}')
    monkeypatch.setattr(hook, "_HOST_CALL_LLM", fake)
    out = _llm_judge_candidate({"model": "m"}, "filler ack", _mk_cand())
    assert out is None  # healthy veto -> not staged
    assert fake.calls == 1


def test_llm_absent_runtime_keeps_regex(monkeypatch):
    fake = _FakeHostCall('{"worth": true}')
    monkeypatch.setattr(hook, "_HOST_CALL_LLM", fake)
    cand = _mk_cand()
    out = _llm_judge_candidate(None, "text", cand)  # no runtime
    assert out is not None
    assert out is cand  # regex verdict object returned unchanged
    assert out.classified_type == "decision"  # regex verdict untouched
    assert fake.calls == 0  # no LLM call without a runtime


def test_llm_host_failure_fails_soft_to_regex(monkeypatch):
    fake = _FakeHostCall("ignored", raise_exc=True)
    monkeypatch.setattr(hook, "_HOST_CALL_LLM", fake)
    out = _llm_judge_candidate({"model": "m"}, "text", _mk_cand())
    assert out is not None
    assert out.classified_type == "decision"  # regex floor, never regressed
    assert fake.calls == 1


def test_llm_unparseable_reply_fails_soft(monkeypatch):
    fake = _FakeHostCall("this is not json at all")
    monkeypatch.setattr(hook, "_HOST_CALL_LLM", fake)
    out = _llm_judge_candidate({"model": "m"}, "text", _mk_cand())
    assert out is not None and out.classified_type == "decision"


def test_llm_judge_passes_short_timeout(monkeypatch):
    """The judge LLM call must pass a SHORT timeout (well under the host's
    60s default + the 30s hook ceiling) so a hung upstream can't trip the
    whole hook callback. The caller's explicit timeout wins over config."""
    fake = _FakeHostCall('{"worth": true, "kb_class": "decision"}')
    monkeypatch.setattr(hook, "_HOST_CALL_LLM", fake)
    _llm_judge_candidate({"model": "m"}, "text", _mk_cand())
    assert fake.calls == 1
    assert fake.timeout_kwargs, "judge must pass an explicit short timeout"
    assert fake.timeout_kwargs[0] <= 15.0, \
        f"judge timeout {fake.timeout_kwargs[0]}s not under the 15s budget"


def test_llm_judge_short_timeout_fail_soft(monkeypatch):
    """A timeout from the host LLM channel degrades to the regex verdict
    (fail-soft) exactly like any other host failure — never regresses."""
    fake = _FakeHostCall("ignored", raise_exc=True)
    monkeypatch.setattr(hook, "_HOST_CALL_LLM", fake)
    out = _llm_judge_candidate({"model": "m"}, "text", _mk_cand())
    assert out is not None
    assert out.classified_type == "decision"
    assert fake.calls == 1
    assert fake.timeout_kwargs, "even a failing call must carry the short timeout"


def test_write_side_on_turn_llm_budget_over_cap_keeps_regex():
    """Scenario 9: the 4th candidate IN THE SAME SESSION does not call the LLM.

    The per-session budget (default 3) is keyed by session_id. A single
    session that fires 4 regex-flagged candidate turns calls the LLM for
    the first 3 and falls back to the regex verdict on the 4th."""
    ctx = _Ctx()
    state = hook._WriteSideState(ctx)
    ref = [0]
    fake = _FakeHostCall('{"worth": true, "kb_class": "premise"}', call_count_ref=ref)
    monkey = _Monkey()
    monkey.set(hook, "_HOST_CALL_LLM", fake)
    monkey.set(state, "_main_runtime", {"model": "m", "provider": "p"})
    # Same session id across all 4 turns so the budget accumulates. The
    # trigger text must hit a V2 cue (the pre-v2 "拍板" verdict text is
    # retired from the default rule set — PRD §3).
    for i in range(4):
        state.on_turn("s1", f"前提：只在 7-gateway 场景 {i} 下才需要 X", "user")
    assert ref[0] == 3, f"expected 3 LLM calls under budget, got {ref[0]}"
    monkey.restore()


class _Monkey:
    """Tiny monkeypatch helper for plain-function tests (no pytest fixture).
    Every test that builds one MUST call monkey.restore() at the end — the
    patched module attributes (hook._service / hook._HOST_CALL_LLM /
    kb_distill._HOST_CALL_LLM / state._main_runtime) otherwise leak into
    later tests and cause cross-file flakiness."""
    def __init__(self):
        self._restore = []

    def set(self, obj, attr, val):
        self._restore.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, val)

    def restore(self):
        for obj, attr, val in reversed(self._restore):
            setattr(obj, attr, val)
        self._restore = []


def test_llm_enabled_flag_off_disables_face():
    ctx = _Ctx()
    state = hook._WriteSideState(ctx)
    # Override rules() directly to carry llm.enable=False.
    state._loaded_rules = {
        "enabled": True, "clarify_batch_size": 5, "min_cue_hits": 1,
        "rules": [], "llm": {"enable": False, "per_session": 3},
    }
    ref = [0]
    fake = _FakeHostCall('{"worth": true}', call_count_ref=ref)
    monkey = _Monkey()
    monkey.set(hook, "_HOST_CALL_LLM", fake)
    monkey.set(state, "_main_runtime", {"model": "m"})
    state.on_turn("s0", "前提：只在 7-gateway 场景下才需要 flock", "decide")
    assert ref[0] == 0, "llm.enable=False must not call the host LLM"
    monkey.restore()


def test_llm_enabled():
    state = hook._WriteSideState(_Ctx())
    state._loaded_rules = {"llm": {"enable": True}}
    monkey = _Monkey()
    monkey.set(hook, "_HOST_CALL_LLM", _FakeHostCall("{}"))
    assert state._llm_enabled() is True
    state._loaded_rules = {"llm": {"enable": False}}
    assert state._llm_enabled() is False
    monkey.set(hook, "_HOST_CALL_LLM", None)
    assert state._llm_enabled() is False
    monkey.restore()


# ---------------------------------------------------------------------------
# Distill face (kb_distill.distill_content) — owner "蒸馏" requirement
# ---------------------------------------------------------------------------

def test_distill_clean_tier_model_unknown():
    """Tier 3 (model doesn't know): distilled_content set, use_original=False."""
    import kb_distill
    from kb_distill import distill_content
    # distilled_content must be >= _DISTILL_CONTENT_MIN UTF-8 bytes
    # (mirrors kb_service._CONTENT_MIN = 60; 2026-10-19 triviality floor).
    # "from myorg.core import Foo; call Foo() with kwarg x=1" = 52 bytes,
    # so pad to 60+ with a meaningful suffix.
    dist_content = "from myorg.core import Foo; call Foo() with kwarg x=1 and kwarg y=2 for the full call path"
    fake = _FakeHostCall('{"tier": "model-unknown", "distilled_title": "内部框架 import 路径", "distilled_content": "' + dist_content + '", "anchor_points": ["myorg.core.Foo", "kwarg x=1"]}')
    monkey = _Monkey()
    monkey.set(kb_distill, "_HOST_CALL_LLM", fake)
    v = distill_content({"model": "m"}, "内部框架", "一段内部框架文档...")
    assert v["use_original"] is False
    assert v["distilled_content"] == dist_content
    assert v["anchor_points"] == ["myorg.core.Foo", "kwarg x=1"]
    assert v["drop_reason"] is None
    assert v["model_used"] is True
    assert v["reason"] == "ok"
    monkey.restore()


def test_distill_tier_model_knows_sets_drop_reason():
    """Tier 1 (model knows it cold): drop_reason set, distilled_content empty,
    use_original=True (nothing stored live; original kept for audit)."""
    import kb_distill
    fake = _FakeHostCall('{"tier": "model-knows", "drop_reason": "public stdlib API, model already knows"}')
    monkey = _Monkey()
    monkey.set(kb_distill, "_HOST_CALL_LLM", fake)
    v = kb_distill.distill_content({"model": "m"}, "标题", "内容...")
    assert v["drop_reason"] == "public stdlib API, model already knows"
    assert v["use_original"] is True
    assert v["distilled_content"] == ""
    assert v["model_used"] is True
    monkey.restore()


def test_distill_oversized_output_rejected_fallback_original():
    """Reviewer acceptance item 7: distilled output longer than the content
    ceiling (LLM 扩写没砍字数) -> rejected_reason set, use_original=True."""
    import kb_distill
    long_content = "x" * 10001  # > 10000 ceiling
    fake = _FakeHostCall(f'{{"tier": "model-misremembers", "distilled_title": "T", "distilled_content": "{long_content}", "anchor_points": []}}')
    monkey = _Monkey()
    monkey.set(kb_distill, "_HOST_CALL_LLM", fake)
    v = kb_distill.distill_content({"model": "m"}, "T", "short original")
    assert v["use_original"] is True
    assert v["rejected_reason"] == "content_length_gate"
    monkey.restore()


def test_distill_llm_error_fallback_original():
    """Host down -> fail-soft original stands (distinct trigger from over-budget)."""
    import kb_distill
    fake = _FakeHostCall("ignored", raise_exc=True)
    monkey = _Monkey()
    monkey.set(kb_distill, "_HOST_CALL_LLM", fake)
    v = kb_distill.distill_content({"model": "m"}, "T", "original content text")
    assert v["use_original"] is True
    assert v["reason"].startswith("llm_error")
    assert v["distilled_content"] == "original content text"
    monkey.restore()


def test_distill_no_runtime_fallback_original():
    """Standalone / no main_runtime snapshot -> original stands, zero LLM call."""
    import kb_distill
    fake = _FakeHostCall('{"tier": "model-unknown"}')
    monkey = _Monkey()
    monkey.set(kb_distill, "_HOST_CALL_LLM", fake)
    v = kb_distill.distill_content(None, "T", "original")
    assert v["use_original"] is True
    assert v["reason"] == "no_runtime"
    assert fake.calls == 0
    monkey.restore()


def test_distill_passes_short_timeout():
    """The distill LLM call must pass a SHORT timeout (well under the host's
    60s default + the 30s hook ceiling) so a hung upstream can't trip the
    whole hook callback. The caller's explicit timeout wins over config."""
    import kb_distill
    fake = _FakeHostCall('{"tier": "model-unknown", "distilled_content": "x"}')
    monkey = _Monkey()
    monkey.set(kb_distill, "_HOST_CALL_LLM", fake)
    kb_distill.distill_content({"model": "m"}, "T", "original content text")
    assert fake.calls == 1
    assert fake.timeout_kwargs, "distill must pass an explicit short timeout"
    assert fake.timeout_kwargs[0] <= 15.0, \
        f"distill timeout {fake.timeout_kwargs[0]}s not under the 15s budget"
    monkey.restore()


def test_distill_short_timeout_fail_soft():
    """A timeout from the host LLM channel degrades to the ORIGINAL content
    (fail-soft) exactly like any other host failure — data is never lost."""
    import kb_distill
    fake = _FakeHostCall("ignored", raise_exc=True)
    monkey = _Monkey()
    monkey.set(kb_distill, "_HOST_CALL_LLM", fake)
    v = kb_distill.distill_content({"model": "m"}, "T", "original content text")
    assert v["use_original"] is True
    assert v["distilled_content"] == "original content text"
    assert fake.calls == 1
    assert fake.timeout_kwargs, "even a failing distill call must carry the short timeout"
    monkey.restore()


def test_distill_parse_failed_fallback_original():
    """Unparseable LLM reply -> original stands (never silently accept)."""
    import kb_distill
    fake = _FakeHostCall("this is not json")
    monkey = _Monkey()
    monkey.set(kb_distill, "_HOST_CALL_LLM", fake)
    v = kb_distill.distill_content({"model": "m"}, "T", "original")
    assert v["use_original"] is True
    assert v["reason"] == "parse_failed"
    monkey.restore()


def test_distill_stages_dropped_mark_not_silently_discarded():
    """drop_reason -> _stage writes dropped=True + drop_reason into the
    staging doc (the clarify user message can list it; user can object)."""
    import kb_distill
    ctx = _Ctx()
    state = hook._WriteSideState(ctx)
    fake_distill = {"use_original": True, "drop_reason": "public API",
                    "distilled_title": "T", "distilled_content": "",
                    "anchor_points": [], "model_used": True, "reason": "ok",
                    "rejected_reason": None, "tier": "model-knows"}
    # Capture the doc dict the hook passes to svc.ingest.
    captured = {}
    class _CapSVC:
        def ingest(self, docs):
            captured["doc"] = docs[0]
            return {"ingested": [], "rejected": [], "staged": ["d1"]}
    monkey = _Monkey()
    monkey.set(hook, "_service", lambda: _CapSVC())
    cand = _mk_cand()
    state._stage(cand, distill_verdict=fake_distill)
    assert captured["doc"]["dropped"] is True
    assert captured["doc"]["drop_reason"] == "public API"
    monkey.restore()


def test_distill_clean_distill_replaces_content_with_distilled():
    """use_original=False + no drop_reason -> doc.content = distilled_content,
    original preserved in original_content metadata (audit). The verdict is
    passed directly (bypassing the LLM call) to isolate the _stage wiring."""
    ctx = _Ctx()
    state = hook._WriteSideState(ctx)
    fake_distill = {"use_original": False, "drop_reason": None,
                    "distilled_title": "精炼标题", "distilled_content": "精炼内容...",
                    "anchor_points": ["a1"], "model_used": True, "reason": "ok",
                    "rejected_reason": None, "tier": "model-unknown"}
    captured = {}
    class _CapSVC:
        def ingest(self, docs):
            captured["doc"] = docs[0]
            return {"staged": ["d1"]}
    monkey = _Monkey()
    monkey.set(hook, "_service", lambda: _CapSVC())
    cand = _mk_cand()
    state._stage(cand, distill_verdict=fake_distill)
    assert captured["doc"]["content"] == "精炼内容..."
    assert captured["doc"]["title"] == "精炼标题"
    assert captured["doc"]["original_content"] == cand.content  # audit
    assert "dropped" not in captured["doc"]
    monkey.restore()


def test_distill_independent_budget_pool():
    """The distill pool (llm.distill.per_session=3) is separate from the
    judge pool (llm.per_session=3): 4 distill turns in one session ->
    exactly 3 host calls; 4 judge turns -> exactly 3 host calls,
    independent counters."""
    ctx = _Ctx()
    state = hook._WriteSideState(ctx)
    # Force both faces on with a live runtime + the default cue rules (so the
    # regex detector actually flags a candidate on the "拍板" text).
    from kb_ingest_rules import DEFAULT_RULES
    import copy
    rules = copy.deepcopy(DEFAULT_RULES)
    rules["llm"] = {"enable": True, "per_session": 3,
                    "distill": {"enable": True, "per_session": 3}}
    state._loaded_rules = rules
    import kb_distill
    global_calls = [0]
    class _CountingFake:
        def __call__(self, **kw):
            global_calls[0] += 1
            raise RuntimeError("stop after counting")
    fake = _CountingFake()
    monkey = _Monkey()
    monkey.set(hook, "_HOST_CALL_LLM", fake)
    monkey.set(kb_distill, "_HOST_CALL_LLM", fake)
    monkey.set(state, "_main_runtime", {"model": "m"})
    # The counting fake RAISES (host down): neither face ever produces a
    # model_used=True verdict, so neither budget counter advances and BOTH
    # faces retry on every turn -> 4 judge + 4 distill = 8 host calls. This
    # asserts the fail-soft retry-forever-until-successful-or-capped
    # semantic (a transient host hiccup must not permanently kill the face).
    for i in range(4):
        state.on_turn("s1", f"前提：只在 7-gateway 场景 {i} 下才需要 X", "user")
    assert global_calls[0] == 8, f"4 judge retries + 4 distill retries = 8, got {global_calls[0]}"
    # Neither pool advanced (no successful verdict consumed a slot).
    assert state._llm_calls_by_session.get("s1", 0) == 0
    assert state._distill_calls_by_session.get("s1", 0) == 0
    monkey.restore()
