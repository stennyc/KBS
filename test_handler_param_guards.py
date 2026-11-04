"""Regression tests: reviewer + data-manager 复盘 fixes (2026-10-19).

Reviewer findings addressed:
  M-1  Nested-field int()/float() coercion crashes (top_k / min_score)
       -> guarded by _coerce_int_arg / _coerce_float_arg, fail closed
          with a structured invalid_args payload.
  m-1  _share_to_global_handler accepted an empty doc_id
       -> now returns "doc_id is required".
  m-2  _distill_handler accepted one-of-two (schema declares both
       required) -> now enforces both raw_title AND raw_content.
  m-3  _search_handler ran an empty query when query was missing
       -> now returns "query is required".
  m-4  _topic_search_handler ran an empty query when query was missing
       -> now returns "query is required".

Data-manager finding noted (no code change required, current state is
safe): a future manual clear must truncate BOTH staging_docs.jsonl AND
staging_index.pkl, otherwise StagingStore._load branch-2 resurrects the
cleared rows from the stale index snapshot.

All handlers are called directly with a DICT payload (the normal host
path); the normalize layer is exercised by test_handler_args.py.
"""

import json
import sys
from pathlib import Path

import unittest

_PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLUGIN_DIR))

import __init__ as kb  # noqa: E402
from test_kb_isolation import install_isolated_service  # noqa: E402


class NestedCoercionTests(unittest.TestCase):
    """M-1: malformed nested numeric fields must fail closed, not crash."""

    def setUp(self):
        install_isolated_service()

    def test_search_top_k_string_fails_closed(self):
        out = json.loads(kb._search_handler({"query": "x", "top_k": "abc"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")
        self.assertIn("top_k", out["message"])

    def test_search_top_k_none_fails_closed(self):
        out = json.loads(kb._search_handler({"query": "x", "top_k": None}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")

    def test_search_min_score_string_fails_closed(self):
        out = json.loads(kb._search_handler(
            {"query": "x", "min_score": "high"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")
        self.assertIn("min_score", out["message"])

    def test_topic_search_top_k_string_fails_closed(self):
        out = json.loads(kb._topic_search_handler(
            {"query": "x", "topic": "general", "top_k": "abc"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")

    def test_search_valid_top_k_and_min_score_pass_through(self):
        out = json.loads(kb._search_handler(
            {"query": "react hooks", "top_k": 2, "min_score": 0.1}))
        self.assertEqual(out["status"], "success")

    def test_topic_search_valid_top_k_pass_through(self):
        out = json.loads(kb._topic_search_handler(
            {"query": "kafka", "topic": "frontend", "top_k": 1}))
        self.assertEqual(out["status"], "success")


class RequiredFieldGuardTests(unittest.TestCase):
    """m-1..m-4: schema-required fields must be enforced with a
    structured error, not a silent empty/zero-hit operation."""

    def setUp(self):
        install_isolated_service()

    def test_search_missing_query_rejected(self):
        out = json.loads(kb._search_handler({"top_k": 3}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "query is required")

    def test_search_blank_query_rejected(self):
        out = json.loads(kb._search_handler({"query": "   "}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "query is required")

    def test_topic_search_missing_query_rejected(self):
        out = json.loads(kb._topic_search_handler({"topic": "frontend"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "query is required")

    def test_share_to_global_empty_doc_id_rejected(self):
        out = json.loads(kb._share_to_global_handler({"doc_id": ""}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "doc_id is required")

    def test_share_to_global_missing_doc_id_rejected(self):
        out = json.loads(kb._share_to_global_handler({}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "doc_id is required")

    def test_distill_title_only_rejected(self):
        # m-2: schema declares BOTH raw_title and raw_content required.
        out = json.loads(kb._distill_handler({"raw_title": "my title"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "raw_content is required")

    def test_distill_content_only_rejected(self):
        out = json.loads(kb._distill_handler({"raw_content": "some body"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "raw_title is required")

    def test_distill_both_present_reaches_distill_path(self):
        # Both fields present -> the handler proceeds to the distill
        # verdict (success with a verdict; no required-field error).
        out = json.loads(kb._distill_handler(
            {"raw_title": "T", "raw_content": "original content body"}))
        self.assertEqual(out["status"], "success")
        self.assertIn("verdict", out)


class IngestSchemaDescriptionTests(unittest.TestCase):
    """M-2: INGEST_SCHEMA description must match the code's actual
    length gate (60-10000 UTF-8 BYTES, not '20-10000 chars')."""

    def test_ingest_schema_documents_byte_unit_and_60_floor(self):
        desc = kb.INGEST_SCHEMA["description"]
        # The stale '20-10000 chars' wording must be gone...
        self.assertNotIn("content 20-10000 chars", desc)
        # ...and replaced with the byte-based wording.
        self.assertIn("60-10000 UTF-8 bytes", desc)
        self.assertIn("UTF-8 BYTES", desc)

    def test_ingest_schema_advertises_10000_byte_ceiling_guidance(self):
        """指导面: agent 必须在调用前就知道 content 上限是 10000 UTF-8 字节,
        超了要压缩/蒸馏而不是反复重试(10-28 教训: 11k 教程被拒 3 次)。"""
        desc = kb.INGEST_SCHEMA["description"]
        self.assertIn("10000", desc)
        # 必须给出超限的处置指引(压缩或 distill), 不是只丢一个数字
        self.assertIn("distill", desc)
        # 拒绝记录必须带上限边界(got/bounds), 让 agent 一次看清差距
        svc = kb._svc()
        big = "占" * 4000  # 12000 UTF-8 bytes > 10000 上限
        out = json.loads(kb._ingest_handler({
            "docs": [{"title": "长文档测试", "content": big, "status": "confirmed"}]
        }))
        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["rejected"]), 1)
        rej = out["rejected"][0]
        self.assertEqual(rej["reason"], "length_gate")
        self.assertEqual(rej["field"], "content")
        self.assertGreater(rej["got"], 10000)
        self.assertEqual(rej["bounds"], [60, 10000])
        self.assertEqual(rej["unit"], "bytes_utf8")


    def test_ingest_schema_advertises_doc_file_payload_hygiene(self):
        """2026-11-04 (选项②, 绝不改 hermes 侧): 宿主 tool_call 外层信封
        json.loads 在长 content 时炸 (tool_search_validation.py:160),
        kb 插件侧的对策 = 工具面引导模型走 doc_file 缩小信封, 不动宿主。
        schema description 必须带这条引导, 否则 LLM 不知道该改走 doc_file。"""
        desc = kb.INGEST_SCHEMA["description"]
        self.assertIn("doc_file", desc)
        self.assertIn("Payload hygiene", desc)
        self.assertIn("bloat the tool_call envelope", desc)
        self.assertIn("escape-heavy", desc)


class IngestRejectGuidanceTests(unittest.TestCase):
    """_ingest_handler 的 reject 响应必须自带可执行的处置指引,
    让 agent 看到 rejected 就知道往 60-10000 字节之间压, 不用读源码。"""

    def setUp(self):
        install_isolated_service()

    def test_rejected_doc_carries_actionable_hint(self):
        big = "超" * 3500  # 10500 bytes, 刚好过上限
        out = json.loads(kb._ingest_handler({
            "docs": [{"title": "超限文档标题", "content": big, "status": "confirmed"}]
        }))
        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["rejected"]), 1)
        # 拒绝记录必须带 content length_gate + bounds
        rej = out["rejected"][0]
        self.assertEqual(rej["reason"], "length_gate")
        self.assertEqual(rej["field"], "content")
        # 顶层 must-carry hint, 带 10000 上限 + distill 指引
        self.assertIn("hint", out)
        hint = out["hint"]
        self.assertIn("10000", hint)
        self.assertIn("UTF-8 bytes", hint)
        self.assertIn("distill", hint)

    def test_title_gate_rejection_hint_points_at_title(self):
        # 标题过短(4 chars < 5 min)走 title 分支, hint 必须指向 title 而非 content
        out = json.loads(kb._ingest_handler({
            "docs": [{"title": "短", "content": "正文内容" * 20, "status": "confirmed"}]
        }))
        rej = out["rejected"][0]
        self.assertEqual(rej["field"], "title")
        self.assertIn("hint", out)
        self.assertIn("title", out["hint"])

    def test_staged_plus_rejected_hints_both_present(self):
        # 同一次调用既有 staged(pending)又有 rejected(over-cap content),
        # hint 必须保留 staged 指引 + 追加 length-gate 指引, 不能互相覆盖
        out = json.loads(kb._ingest_handler({"docs": [
            {"title": "待确认文档标题", "content": "这是待确认正文" * 5,
             "status": "pending"},
            {"title": "超限文档标题", "content": "超" * 3500,
             "status": "confirmed"},
        ]}))
        self.assertIn("hint", out)
        hint = out["hint"]
        # staged 侧(需 confirm)
        self.assertIn("confirm", hint)
        # rejected 侧(length gate + distill 指引)
        self.assertIn("Length gate", hint)
        self.assertIn("distill", hint)


if __name__ == "__main__":
    unittest.main()
