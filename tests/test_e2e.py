"""End-to-end test for the knowledge-base search service (PRD §13.1/13.2).

Exercises the REAL KbSearchService (hot + cold rank_bm25, pickle persist) —
not stubs — across the single-shot contract:
  ingest -> persist -> reload in a fresh service instance -> search hits.
This is the vertical slice that proves the plugin handler's payload shape and
the durable round-trip without any JVM.

Run:  python3 -m unittest test_e2e -v
"""

import kb_service
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb_service import KbSearchService, render  # noqa: E402

DOCS = [
    {"docid": "doc-1", "title": "数据管道架构",
     "content": "数据管道 批处理 流处理 Lambda 架构 抽取 转换 加载 三段式 架构 说明 若干字",
     "doc_type": "guide", "status": "confirmed"},
    {"docid": "doc-2", "title": "Docker 容器",
     "content": "Docker 开源 容器引擎 轻量 可移植 打包 隔离 镜像 构建 部署 说明 若干字",
     "doc_type": "reference", "status": "confirmed"},
    {"docid": "doc-3", "title": "Kafka 消息队列",
     "content": "Kafka 分布式 事件流 高吞吐 低延迟 发布 订阅 分区 复制 消费组 机制 说明 若干字",
     "doc_type": "guide", "status": "confirmed"},
]


class E2ETests(unittest.TestCase):

    def _svc(self, name: str) -> KbSearchService:
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        return KbSearchService(Path(d.name))

    # PRD §13.2 payload shape: one call, structured, full text, no follow-up
    def test_single_shot_payload_shape(self):
        svc = self._svc("e2e1")
        svc.ingest(DOCS)
        payload = svc.search("数据管道 架构", top_k=2)
        self.assertEqual(payload["status"], "success")
        data = payload["data"]
        self.assertEqual(data["query"], "数据管道 架构")
        # The term-overlap gate makes this precise: only doc-1 shares
        # "数据/管道/架构" — Docker & Kafka share no word, so total is 1.
        self.assertEqual(data["total"], 1)
        top = data["results"][0]
        for field in ("doc_id", "title", "content", "score", "metadata"):
            self.assertIn(field, top, f"payload missing field {field}")
        self.assertEqual(top["doc_id"], "doc-1")
        self.assertIn("数据管道", top["content"], "full text must be present")

    # Durable round-trip: ingest -> persist -> FRESH instance reloads -> hits
    def test_durable_roundtrip_across_instances(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        base = Path(d.name)
        a = KbSearchService(base)
        a.ingest(DOCS)          # ingest persists under the write lock
        # A brand-new service (like a process restart) must reload and find
        # the docs via the pickle side-stores.
        b = KbSearchService(base)
        res = b.search("Kafka 消息队列", top_k=1)
        self.assertEqual(res["data"]["results"][0]["doc_id"], "doc-3")

    # B1: a filtered search must not starve — when the top candidates are
    # mostly non-matching, top_k filtered results still come back.
    def test_filter_not_starved_by_non_matching_top_candidates(self):
        svc = self._svc("e2e6")
        guides = [{"docid": f"g{i}", "title": f"Guide {i}",
                   "content": f"gated guide alpha doc {i} 内容填充足够长度 甲乙丙丁戊己庚辛",
                   "doc_type": "guide", "status": "confirmed"} for i in range(3)]
        others = [{"docid": f"o{i}", "title": f"Other {i}",
                   # more occurrences of the shared term -> higher BM25, so
                   # these outrank the guides in the unfiltered ranking
                   "content": f"gated gated gated gated other doc {i} 内容填充 甲乙丙丁戊己庚辛",
                   "doc_type": "other", "status": "confirmed"} for i in range(10)]
        svc.ingest(guides + others)
        res = svc.search("gated", top_k=3, filters={"doc_type": "guide"})
        results = res["data"]["results"]
        # Old code capped the fetch (top_k*2) and truncated AFTER filtering:
        # the 6 top candidates were all "other", so 0 guides came back.
        self.assertEqual(len(results), 3, "filter starved by non-matching top-k")
        self.assertTrue(all(r["metadata"]["doc_type"] == "guide" for r in results))

    # B3: docs without a docid get a deterministic content-addressed id —
    # never a layer position number.
    def test_missing_docid_gets_deterministic_id(self):
        import json as _json
        import uuid as _uuid
        svc = self._svc("e2e7")
        doc = {"title": "Redis 缓存",
               "content": "唯一内容 甲 乙 丙 丁 戊 己 庚 辛 壬 癸 子 丑 寅 卯 扩充 长度门 之上",
               "status": "confirmed"}
        svc.ingest([doc])
        res = svc.search("Redis 缓存", top_k=1)
        got = res["data"]["results"][0]["doc_id"]
        # Mirror the service's normalization exactly: sorted-key JSON of the
        # doc (minus docid), uuid5 over NAMESPACE_OID.
        expected = str(_uuid.uuid5(
            _uuid.NAMESPACE_OID,
            _json.dumps({k: v for k, v in sorted(doc.items()) if k != "docid"},
                       ensure_ascii=False, default=str)))
        self.assertEqual(got, expected, "docid must be content-addressed uuid5")
        # It is a real UUID, not a layer position number leaking into identity.
        _uuid.UUID(got)  # raises if not a valid UUID
        self.assertNotIn(got, ("0", "1"), "a position number leaked into docid")
        # Re-ingest identical content in a FRESH instance -> same id
        d2 = tempfile.TemporaryDirectory(); self.addCleanup(d2.cleanup)
        b = KbSearchService(Path(d2.name))
        b.ingest([doc])
        self.assertEqual(b.search("Redis 缓存", top_k=1)["data"]["results"][0]["doc_id"],
                         got, "content-addressed id must be stable across processes")

    # Startup guard: a missing data dir is a loud SearchBackendError, not a
    # silent empty index.
    def test_missing_data_dir_loud_fail(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        from search_backend import SearchBackendError
        with self.assertRaises(SearchBackendError):
            KbSearchService(Path(d.name) / "no-such-dir")

    # Closes @reviewer's "Generation: 0 after ingest" flag: a successful
    # write MUST bump the hot layer's write generation, and ingest() returns
    # it (the two-layer settle generation only bumps on settle).
    def test_ingest_bumps_hot_generation(self):
        svc = self._svc("e2e8")
        gen_before = svc._hot.generation
        returned = svc.ingest(DOCS)
        self.assertGreater(svc._hot.generation, gen_before,
                           "a write must bump the hot layer generation")
        self.assertEqual(returned["generation"], svc._hot.generation,
                         "ingest() returns the hot write generation, "
                         "never a stale 0")

    # Observability (@researcher's 下一版 gap): structured logging — ingest at
    # INFO, search at DEBUG, single file, no rotation. configure_logging() is
    # the wiring point the plugin adapter calls; the service only *emits*
    # records, so the layer stays Hermes-free (architecture rule).
    def test_ingest_search_write_structured_log(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(base), ignore_errors=True)
        data = base / "kb-data"
        data.mkdir()
        log_path = base / "kb.log"
        kb_service.configure_logging(log_path, level=logging.DEBUG)
        kb_service.configure_logging(log_path, level=logging.DEBUG)  # idempotent
        svc = KbSearchService(data)
        svc.ingest(DOCS)
        svc.search("数据管道 架构", top_k=2)
        text = log_path.read_text()
        self.assertIn("ingest", text)
        self.assertIn("docs=3", text)
        self.assertIn("search", text)
        self.assertEqual(text.count("docs=3"), 1,
                         "re-pointing the log must not duplicate handlers")

    # Optional metadata filters narrow results
    def test_metadata_filters(self):
        svc = self._svc("e2e3")
        svc.ingest(DOCS)
        res = svc.search("架构 容器 消息队列", top_k=5,
                         filters={"doc_type": "guide"})
        ids = [r["doc_id"] for r in res["data"]["results"]]
        self.assertIn("doc-1", ids)
        self.assertNotIn("doc-2", ids, "reference doc must be filtered out")

    # No-match query -> empty result list, still a success envelope
    def test_no_match_empty(self):
        svc = self._svc("e2e4")
        svc.ingest(DOCS)
        res = svc.search("量子 物理 相对论", top_k=5)
        self.assertEqual(res["data"]["results"], [])
        self.assertEqual(res["status"], "success")

    # render() produces the exact JSON string a tool returns
    def test_render_json(self):
        svc = self._svc("e2e5")
        svc.ingest(DOCS)
        text = render(svc.search("Docker 容器", top_k=1))
        self.assertIn('"doc_id"', text)
        import json
        self.assertEqual(json.loads(text)["data"]["results"][0]["doc_id"], "doc-2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
