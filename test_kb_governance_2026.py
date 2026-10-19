"""Owner 2026-10-19 governance: KB is a knowledge corpus (经验/手册/教程/教材/资料), not a fragment vault.

Step 1: triviality floor (content >= 60 chars on BOTH confirmed entry
paths — ingest(status=confirmed) AND confirm()).
Step 2: DOC_KIND_VOCAB expanded to 5 values (experience / tutorial /
manual / reference / material); all-ways alignment.
Step 4: owner-review scope — user-specified doc_kind lands as-is (no
review gate); agent auto-captured docs default to doc_kind=None and get
labeled only via the explicit set_doc_kind face.

Run: /root/.hermes/plugin-venv/bin/python -m pytest test_kb_governance_2026.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kb_service import KbSearchService  # noqa: E402
from kb_ingest_rules import DOC_KIND_VOCAB  # noqa: E402
from kb_ingest_hook import _WriteSideState  # noqa: E402
from kb_ingest_rules import Candidate  # noqa: E402


class TrivialityFloorTests(unittest.TestCase):
    """content < 60 bytes must be rejected on both confirmed entry paths."""

    def _svc(self):
        import tempfile
        self._base = Path(tempfile.mkdtemp())
        data = self._base / "kb-data"
        data.mkdir()
        return KbSearchService(data)

    def test_ingest_confirmed_rejects_59(self):
        svc = self._svc()
        out = svc.ingest([{"docid": "t59", "title": "占位 测试 条目",
                            "content": "x" * 59, "status": "confirmed"}])
        self.assertIn("t59", [r.get("doc_id") for r in out["rejected"]])
        rej = next(r for r in out["rejected"] if r.get("doc_id") == "t59")
        self.assertEqual(rej["reason"], "length_gate")
        self.assertEqual(rej["field"], "content")

    def test_ingest_confirmed_accepts_60(self):
        svc = self._svc()
        out = svc.ingest([{"docid": "t60", "title": "恰好 六十字",
                           "content": "x" * 60, "status": "confirmed"}])
        self.assertEqual(out["ingested"], ["t60"])

    def test_confirm_rejects_59(self):
        svc = self._svc()
        svc.ingest([{"docid": "c59", "title": "staged 过短",
                     "content": "y" * 59, "status": "pending",
                     "source_profile": "coder"}])
        res = svc.confirm("c59", source_profile="coder", confirmed_by="agent")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error"], "length_gate")
        self.assertEqual(res["field"], "content")

    def test_confirm_accepts_80(self):
        svc = self._svc()
        svc.ingest([{"docid": "c80x", "title": "staged 合格",
                     "content": "y" * 80, "status": "pending",
                     "source_profile": "coder"}])
        res = svc.confirm("c80x", source_profile="coder", confirmed_by="agent")
        self.assertEqual(res["status"], "success")


class DocKindVocabTests(unittest.TestCase):
    """Step 2: the 5-value doc_kind axis (owner 2026-10-19 expansion)."""

    def test_vocab_is_five_values(self):
        self.assertEqual(DOC_KIND_VOCAB,
                         ("experience", "tutorial", "manual",
                          "reference", "material"))

    def test_set_doc_kind_accepts_manual(self):
        import tempfile
        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        svc = KbSearchService(data)
        svc.ingest([{"docid": "m-1", "title": "手册 形态 测试 文档",
                     "content": "这是一篇 系统 手册 文档 内容 若干字 扩充 至 长度门 之上 合格 内容 填充 段落 完整 覆盖 场景 M1FILL甲甲甲甲",
                     "status": "confirmed"}])
        out = svc.set_doc_kind("m-1", "manual")
        self.assertEqual(out.get("status"), "success", out)
        doc = next(x for x in svc._hot.docs_snapshot()
                   + svc._cold.docs_snapshot()
                   if x.get("docid") == "m-1")
        self.assertEqual(doc.get("doc_kind"), "manual")

    def test_set_doc_kind_accepts_material(self):
        import tempfile
        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        svc = KbSearchService(data)
        svc.ingest([{"docid": "mt-1", "title": "教材 形态 测试 文档",
                     "content": "这是一篇 教材 书籍 类 长文 内容 若干字 扩充 至 长度门 之上 合格 内容 填充 段落 完整 覆盖 场景 MT1FILL甲",
                     "status": "confirmed"}])
        out = svc.set_doc_kind("mt-1", "material")
        self.assertEqual(out.get("status"), "success", out)

    def test_set_doc_kind_rejects_out_of_vocab(self):
        import tempfile
        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        svc = KbSearchService(data)
        svc.ingest([{"docid": "x-1", "title": "越界 门类 测试 文档",
                     "content": "这是一篇 测试 文档 内容 若干字 扩充 至 长度门 之上 合格",
                     "status": "confirmed"}])
        out = svc.set_doc_kind("x-1", "cookbook")
        self.assertEqual(out.get("status"), "error")
        self.assertEqual(out.get("error"), "invalid_doc_kind")
        self.assertEqual(out.get("allowed"), list(DOC_KIND_VOCAB))

    def test_ingest_with_doc_kind_manual_lands_as_is(self):
        """Step 4: a user-specified doc_kind at ingest lands as-is —
        no review gate blocks it."""
        import tempfile
        base = Path(tempfile.mkdtemp())
        data = base / "kb-data"
        data.mkdir()
        svc = KbSearchService(data)
        out = svc.ingest([{"docid": "uk-m1", "title": "用户 直指定 门类",
                           "content": "用户 直接 ingest 指定 的 doc_kind 应直通 无 review 门 若干字 扩充 UK1FILL甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲",
                           "status": "confirmed",
                           "doc_kind": "manual"}])
        self.assertEqual(out["ingested"], ["uk-m1"])
        doc = next(x for x in svc._hot.docs_snapshot()
                   + svc._cold.docs_snapshot()
                   if x.get("docid") == "uk-m1")
        self.assertEqual(doc.get("doc_kind"), "manual")


class AgentCaptureDocKindTests(unittest.TestCase):
    """Step 4: agent auto-captured docs default to doc_kind=None —
    labeling is ONLY via the explicit set_doc_kind face (the owner
    review action)."""

    def _agent_candidate(self):
        return Candidate(
            doc_id="agent-cand-1",
            classified_type="pattern",
            label="可迁移套路",
            title="agent 捕获 的 可迁移 套路 文档 标题 若干字",
            content=("遇到 部署 回滚 场景 时 一律 先 检查 数据库 迁移 状态 "
                     "再 执行 回滚 操作 避免 数据 不一致 若干字 扩充"),
            source_session_id="sess-x",
            source_profile="coder",
            cue_hits=["凡是"],
            origin_kind="agent",
        )

    def test_agent_captured_doc_lands_with_doc_kind_none(self):
        """_build_agent_doc must NOT auto-assign a doc_kind (the owner
        review constraint: agent-captured docs default unlabeled)."""
        svc_calls = []

        class _FakeSVC:
            def ingest(self, docs):
                svc_calls.append(docs)
                return {"ingested": [], "rejected": [],
                        "staged": [d["docid"] for d in docs],
                        "generation": 1}
            def confirm(self, doc_id, source_profile=None,
                        confirmed_by=None):
                return {"status": "success", "doc_id": doc_id}
            def _audit(self, **kw):
                pass

        state = _WriteSideState(_FakeSVC())
        doc = state._build_agent_doc(self._agent_candidate(),
                                      session_msgs={"user": ["部署 回滚"]})
        self.assertIsNotNone(doc, "_build_agent_doc must return a doc")
        self.assertNotIn("doc_kind", doc,
                         "agent-captured docs must NOT carry a doc_kind "
                         "label at capture time (default None = unlabeled; "
                         "labeling is the owner's explicit set_doc_kind "
                         "action, not the agent's)")


if __name__ == "__main__":
    unittest.main()
