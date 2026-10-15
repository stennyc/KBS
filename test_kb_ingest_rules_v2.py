#!/usr/bin/env python3
"""v2 rule-set tests (2026-10-14, PRD §3): agent personal-knowledge classes.

Covers:
  - each of the 4 v2 classes (pattern / anti-pattern / tool-quirk /
    premise) HITS its own structural cue (>= 2 examples each, zh + en)
  - the 4 v2 classes carry origin_kind=agent (agent partition,
    same-profile self-confirm, no clarify human gate)
  - PRD §3.2 explicit-exclusion samples MISS every v2 cue:
      * a bare verdict ("我们拍板用 X 方案") — filemem L2 territory
      * short command/question heads (A / 删 / 1. 丢 2. 同意 / …吗？)
      * a raw config dump (no task-subject+premise structure)
      * a root-cause->fix single-incident narrative (filemem L1)
  - legacy rule ids (decision/walkthrough/config/incident) are absent
    from DEFAULT_RULES (the retirement itself is asserted)
  - a stale pre-v2 profile override that re-introduces a legacy rule
    id degrades safely (origin_kind=None backstop, not a crash)

Run: PYTHONPATH=/usr/local/lib/hermes-agent pytest test_kb_ingest_rules_v2.py -q
"""

import sys

PLUGIN_DIR = "/root/.hermes/plugins/knowledge-base-search"
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

from kb_ingest_rules import (  # noqa: E402
    DEFAULT_RULES, CandidateDetector, merge_rules, is_meta_instruction,
)

V2_IDS = ("pattern", "anti-pattern", "tool-quirk", "premise")
LEGACY_IDS = ("decision", "walkthrough", "config", "incident")


def _det(rules=None):
    return CandidateDetector(rules or merge_rules(None),
                             session_id="s1", profile="coder")


def _docid_fn():
    return lambda t, c: "doc-v2"


# ---------------------------------------------------------------------------
# 1. Retirement of the legacy classes (the collision itself is gone)
# ---------------------------------------------------------------------------
class TestLegacyRetirement:
    def test_default_rules_carry_no_legacy_class(self):
        ids = [r["id"] for r in DEFAULT_RULES["rules"]]
        for legacy in LEGACY_IDS:
            assert legacy not in ids, (
                f"legacy class {legacy!r} must be retired from "
                f"DEFAULT_RULES (PRD §3.1); found in {ids}")

    def test_all_v2_classes_carry_agent_origin_kind(self):
        det = _det()
        for cand in self._each_class_hit(det):
            assert cand.origin_kind == "agent"

    def _each_class_hit(self, det):
        """One known-good v2 example per class (shared by tests 1 & 2)."""
        return [det.scan(t, _docid_fn()) for t in (
            "凡是部署类任务，一律先在预发环境跑回归再上生产。",
            "踩坑：在 7-gateway 下用单进程锁会因跨进程失效，flock 才对。",
            "宿主 call_llm 的限制：首发超时 60s，在 30s hook 预算内必超时。",
            "前提：只在 7-gateway 并发下才需要 flock 跨进程 rebase，否则单进程足够。",
        )]

    def test_all_v2_classes_hit_and_classified(self):
        det = _det()
        hits = self._each_class_hit(det)
        assert all(h is not None for h in hits), (
            f"every v2 example must hit its class; misses: "
            f"{[(i, t[:20]) for i, (h, t) in enumerate(zip(hits, [
                '凡是部署类任务...', '踩坑：...', '宿主 call_llm...',
                '在 7-gateway...'])) if h is None]}")
        assert [h.classified_type for h in hits] == list(V2_IDS), (
            f"v2 classes must classify in order; got "
            f"{[h.classified_type for h in hits]}")


# ---------------------------------------------------------------------------
# 2. Per-class positive examples (zh + en, >= 2 each)
# ---------------------------------------------------------------------------
class TestV2ClassHits:
    def test_pattern_hits(self):
        d = _det()
        assert d.scan("凡是 git 调 REST 的场景，优先用 Bearer 认证，别用 gh 登录。",
                      _docid_fn()).classified_type == "pattern"
        assert d.scan("通用做法：排查部署类问题先看 systemd journal，再翻应用日志。",
                      _docid_fn()).classified_type == "pattern"

    def test_anti_pattern_hits(self):
        d = _det()
        assert d.scan("不要直接 rm 宿主插件目录，因为 hardlink 会让其他 profile 的物理文件一起没。",
                      _docid_fn()).classified_type == "anti-pattern"
        assert d.scan("踩坑：在 7-gateway 下用单进程锁会因为跨进程失效，flock 才是对的。",
                      _docid_fn()).classified_type == "anti-pattern"

    def test_tool_quirk_hits(self):
        d = _det()
        assert d.scan("宿主 call_llm 的限制：首发超时 60s，在 30s hook 预算内必超时，要 catch 后重试。",
                      _docid_fn()).classified_type == "tool-quirk"
        assert d.scan("Pyterrier 在 v0.3 版本下 pickle 冷层路径必须绝对路径，相对路径会报怪错。",
                      _docid_fn()).classified_type == "tool-quirk"

    def test_premise_hits(self):
        d = _det()
        assert d.scan("前提：只在 7-gateway 并发下才需要 flock 跨进程 rebase；单进程场景默认即可。",
                      _docid_fn()).classified_type == "premise"
        assert d.scan("适用范围：agent 经验条目 14 天内未引用即衰减折出自动注入池，带 project_ref 的除外。",
                      _docid_fn()).classified_type == "premise"


# ---------------------------------------------------------------------------
# 3. PRD §3.2 explicit-exclusion samples — all must MISS every v2 cue
# ---------------------------------------------------------------------------
class TestExplicitExclusions:
    # A bare verdict: filemem's L2-decision territory, no premise attached.
    VERDICT = "我们拍板用方案 X，否决 Y，最终定为 X。"
    # Short command/question heads: noise-title shape, no knowledge.
    COMMANDS = ("A", "删", "1. 丢 2. 同意", "那就帮我加")
    # A bare question with no settled answer.
    QUESTION = "既然会影响缓存命中，那就算了，所以现在只是影响落盘文件的大小而已吗？"
    # A raw config dump: no task subject, no premise.
    CONFIG_DUMP = ("把这段存下来：\n```\nplugins:\n  kb_ingest:\n"
                   "    clarify_batch_size: 5\n```")
    # A single-incident root-cause->fix narrative: filemem's L1 territory.
    WALKTHROUGH = "报错 no such table，排查发现迁移没跑，补跑后恢复正常。"
    # A plain user fact: neither v2 class nor filemem trigger applies.
    PLAIN_FACT = "这台机器的 gateway 走 user 侧 systemd service。"

    def _must_miss(self, d, text, why):
        c = d.scan(text, _docid_fn())
        assert c is None, f"must MISS (v2 is {why}): {text[:40]!r}"

    def test_bare_verdict_misses(self):
        self._must_miss(_det(), self.VERDICT, "a one-off verdict, no premise")

    def test_command_heads_miss(self):
        d = _det()
        for cmd in self.COMMANDS:
            self._must_miss(d, cmd, "a bare command, not knowledge")

    def test_question_misses(self):
        self._must_miss(_det(), self.QUESTION, "an open question, no settled answer")

    def test_config_dump_misses(self):
        self._must_miss(_det(), self.CONFIG_DUMP, "raw dump, no task+premise structure")

    def test_walkthrough_narrative_misses(self):
        self._must_miss(_det(), self.WALKTHROUGH,
                        "a single-incident fix narrative = filemem L1")

    def test_plain_fact_misses(self):
        self._must_miss(_det(), self.PLAIN_FACT,
                        "a plain environment fact = filemem L0")

    def test_filemem_trigger_words_alone_miss(self):
        """The exact words that caused the original collision must NOT
        trigger a v2 candidate on their own (they need the v2
        task-subject+premise structure around them)."""
        d = _det()
        for text in (
            "决定采用 B。",
            "修复了这个 bug。",
            "配置改成了 X。",
            "拍板：用 Docker。",
        ):
            self._must_miss(d, text, "a filemem trigger word without v2 structure")


# ---------------------------------------------------------------------------
# 4b. Meta-instruction / task-scoped-question noise gate
#      (owner 2026-10-14: 16/31 "keep" rows were user commands / questions
#       about the task itself, NOT knowledge — the v2 anti-pattern cue's
#       踩坑/坑 substring false-hits them. Conservative: only high-
#       confidence meta shapes block; genuine knowledge slips through.)
# ---------------------------------------------------------------------------
class TestMetaInstructionGate:
    META_SHAPES = (
        "落盘计划prd, 免得开发过程遗漏任何事项",
        "开始吧，全绿才进入下阶段",
        "总结并出具计划或prd",
        "你给我一份提示词，我在新Session试试看",
        "这个有没有对应的filemem 工具模型可以直接调？",
        "需要修复吗？ 还有，为何每次调用 kb 工具后面都要使用 terminal",
        "好，先将这个 clarify 流程坑的经验存到你的知识库作为经验",
        "我认为应该进行区分，我们先来讨论什么应该是知识库应该识别的",
        "为什么不用 Docker 而用 K8s？",
        "\u201c \u2022 KB \u566a\u58f0\u5019\u9009\uff08\u201c\u63a8\u201d/\u201c\u8981\u201d/\u201cA\u201d\uff09\u5168\u90e8\u6ca1 confirm\u201d <- \u8fd9\u4e0d\u662f\u5e94\u8be5\u76f4\u63a5",
        "\u4e0d\u662f\uff0c\u662f\u8fd9\u4e2a\u201c",
    )
    GENUINE_SHAPES = (
        "拍板：用 Docker 部署，否决 K8s",
        "GitHub API Topics Patch vs Put Quirk",
        "code_map 8bd0401: clarify Other text whitelist evidence & gate behavior",
        "KB 插件复刻 filemem 脚注机制可行性调研结论",
        "Filemem relay-block purge: A/B leg logic and self-reference",
        "Knowledge Base Clarify Implementation: Filemem Paradigm Alignment",
        "Diagnosis and Fixture Fix for Superseded Memory Gate Failure",
        "KB Write-Side V2 Refactor: Rule Class Migration and Exemption Keys",
        "KB 噪声候选全部没 confirm，staging 留着待审计",
    )

    def test_meta_shapes_blocked(self):
        for t in self.META_SHAPES:
            assert is_meta_instruction(t), f"must block meta shape: {t!r}"

    def test_genuine_shapes_pass(self):
        for t in self.GENUINE_SHAPES:
            assert not is_meta_instruction(t), f"must NOT block genuine: {t!r}"

    def test_empty_title_blocked(self):
        assert is_meta_instruction("")
        assert is_meta_instruction("   ")


# ---------------------------------------------------------------------------
# 5. Backward-compat: a stale pre-v2 profile override re-introducing a
#    legacy rule id degrades safely (origin_kind=None backstop)
# ---------------------------------------------------------------------------
class TestStaleProfileOverrideBackstop:
    def test_stale_legacy_rule_id_degrades_to_auto_partition(self):
        stale_rules = merge_rules({
            "rules": [
                {"id": "decision", "label": "决策结论",
                 "cues": [r"决定采用", r"拍板"]},
            ],
        })
        d = _det(stale_rules)
        c = d.scan("最终拍板用 X。", _docid_fn())
        # a stale legacy id still matches (its cues survive the merge),
        # but it is NOT a v2 id -> origin_kind stays None ("auto"
        # partition), never silently re-entering the agent partition.
        assert c is not None
        assert c.classified_type == "decision"
        assert c.origin_kind is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
