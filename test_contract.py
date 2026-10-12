"""Backend-agnostic contract test suite (PRD §12).

The 9 contract tests are the acceptance gate for ANY SearchBackend
implementation. They run against InMemoryStubBackend today to prove the suite
is satisfiable; when PyterrierBackend lands, only `make_backend` is re-pointed
and the suite re-runs unchanged against the new backend.

Coding-convention exemption ① applies: this suite and the protocol class are
consumed by the tests themselves, so they are not "island" dead code.

Run:  python3 -m unittest test_contract -v
      or  pytest knowledge-base/test_contract.py
"""

import pickle
import tempfile
import threading
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from search_backend import SCHEMA_VERSION, SearchResult  # noqa: E402
from stub_backend import InMemoryStubBackend  # noqa: E402
from hot_layer import HotLayerBackend  # noqa: E402
from two_layer import TwoLayerBackend  # noqa: E402
from cold_layer import PyterrierBackend  # noqa: E402  (JVM-free in-memory cold layer)


DOCS = [
    {"id": "1", "title": "数据管道架构", "content": "数据管道 批处理 流处理 Lambda 架构 抽取 转换 加载"},
    {"id": "2", "title": "Docker 容器", "content": "Docker 开源 容器引擎 轻量 可移植 打包"},
    {"id": "3", "title": "Kafka 消息队列", "content": "Apache Kafka 分布式 事件流 高吞吐 低延迟"},
]

# Docs for the two-layer suite carry an EXTERNAL docid (the cross-layer join key).
DOCS_EXT = [
    {"docid": "doc-a", "title": "数据管道架构", "content": "数据管道 批处理 流处理 Lambda 架构"},
    {"docid": "doc-b", "title": "Docker 容器", "content": "Docker 开源 容器引擎 轻量 可移植"},
    {"docid": "doc-c", "title": "Kafka 消息队列", "content": "Kafka 分布式 事件流 高吞吐 低延迟"},
]


def make_backend(path: Path) -> InMemoryStubBackend:
    """Re-point this single factory to build PyterrierBackend once it lands.
    (TDD red-phase exemption ②: referencing a not-yet-implemented class here
    is permitted.)"""
    be = InMemoryStubBackend(path)
    be.update(DOCS)
    return be


def make_two_layer(settle_threshold: int = 2) -> TwoLayerBackend:
    """Two-layer front-end over a real hot layer + a stub cold fake (no JVM).

    The stub cold backend satisfies the same SearchBackend protocol as
    PyterrierBackend, so the dual-layer contracts run deterministically here and
    re-run unchanged against the real Pyterrier cold layer.
    """
    hot = HotLayerBackend()
    cold = InMemoryStubBackend(Path(tempfile.gettempdir()) / f"cold-{id(hot)}.pkl")
    cold.update([{"docid": d["docid"], "title": d["title"], "content": d["content"]}
                 for d in DOCS_EXT])
    return TwoLayerBackend(hot, cold, settle_threshold=settle_threshold)


# ---------------------------------------------------------------------------
# Contract tests #1-#9 (PRD §12)
# ---------------------------------------------------------------------------
class ContractTests(unittest.TestCase):

    def _mk(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        return make_backend(Path(d.name) / "index.terrier")

    # 1
    def test_search_returns_top_k_results(self):
        be = self._mk()
        res = be.search("数据管道 架构", top_k=2)
        self.assertLessEqual(len(res), 2)
        self.assertEqual(res[0].doc_id, "1")
        self.assertIsInstance(res[0], SearchResult)

    # 2
    def test_update_adds_new_docs(self):
        be = self._mk()
        be.update([{"id": "4", "title": "Redis 缓存", "content": "键值 缓存 过期 穿透"}])
        self.assertEqual(be.search("Redis 缓存", top_k=1)[0].doc_id, "4")

    # 3
    def test_persist_and_load(self):
        be = self._mk()
        path = be._path
        be.persist()
        be2 = InMemoryStubBackend(path)
        self.assertTrue(be2.load())
        self.assertEqual(be2.search("Kafka 消息队列", top_k=1)[0].doc_id, "3")

    # 4  (B1 lesson: a tiny corpus must not return an empty result.
    #      Assert non-empty + correct ordering, NOT score sign — sign is
    #      backend-specific (TF>0 here, BM25 single-doc IDF<0 on Pyterrier).
    #      Score-sign checks belong in each backend's own unit tests.)
    def test_single_doc_corpus(self):
        be = InMemoryStubBackend(Path(tempfile.gettempdir()) / "single")
        be.update([{"id": "9", "title": "Hermes", "content": "Python AI agent 框架"}])
        res = be.search("Python 框架", top_k=1)
        self.assertEqual(len(res), 1, "single-doc corpus returned empty (B1 regression)")
        self.assertEqual(res[0].doc_id, "9")
        self.assertEqual(res[0].title, "Hermes")

    # 5  (B1 protection: load() -> update() -> search() on old AND new docs)
    def test_load_then_update_finds_old_and_new_docs(self):
        be = self._mk()
        path = be._path
        be.persist()
        be2 = InMemoryStubBackend(path)
        be2.load()
        be2.update([{"id": "5", "title": "Spark", "content": "大数据 分析 引擎"}])
        old = be2.search("Docker 容器", top_k=1)
        new = be2.search("Spark 大数据", top_k=1)
        self.assertEqual(old[0].doc_id, "2")
        self.assertEqual(new[0].doc_id, "5")

    # 6  (corruption detection: schema drift is loud, not silent)
    def test_generation_corruption_detection(self):
        from search_backend import SearchBackendError
        be = self._mk()
        path = be._path
        be.persist()
        with open(path, "rb") as f:
            payload = pickle.load(f)
        payload["schema_version"] = 999  # version drift
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        be2 = InMemoryStubBackend(path)
        with self.assertRaises(SearchBackendError):
            be2.load()

    # 7  (atomic write: no torn temp file survives a completed persist)
    def test_atomic_write(self):
        be = self._mk()
        path = be._path
        be.persist()
        self.assertTrue(path.exists())
        self.assertFalse(path.with_suffix(".tmp").exists(),
                         "a completed persist must not leave a .tmp behind")

    # 8  (RCU: an in-flight search sees one consistent generation)
    def test_rebuild_during_inflight_query(self):
        be = self._mk()
        results_holder: dict = {}
        def worker():
            results_holder["res"] = be.search("Docker 容器", top_k=2)
        t = threading.Thread(target=worker)
        t.start()
        # Mutate concurrently while a read may be in flight.
        for _ in range(200):
            be.update([{"id": f"x{_}", "title": "x", "content": "数据管道"}])
        t.join()
        res = results_holder["res"]
        # Every result must come from a single consistent snapshot: all share
        # one generation tag, and ids are all valid members of some snapshot.
        gens = {r.raw["generation"] for r in res if r.raw}
        self.assertLessEqual(len(gens), 1, "in-flight query saw a torn multi-generation mix")

    # 9  (single-shot: one call returns full text, no follow-up fetch)
    def test_single_shot_completeness(self):
        be = self._mk()
        res = be.search("Kafka 消息队列", top_k=1)
        top = res[0]
        self.assertIn("分布式", top.content)          # full content present
        self.assertGreater(len(top.content), 0)      # not an empty stub
        self.assertEqual(top.title, "Kafka 消息队列")
        # The consuming agent gets everything it needs from this single call.
        self.assertIsInstance(top.metadata, dict)

    # 10 (no-match query returns [] — a backend must not pad top-k with
    #     zero-relevance placeholder docs)
    def test_no_match_returns_empty(self):
        be = self._mk()
        res = be.search("量子 物理 相对论", top_k=5)   # no term overlaps the corpus
        self.assertEqual(res, [], "no-match query must return [], not placeholders")


# ---------------------------------------------------------------------------
# Dual-layer contract tests #11-#15 (PRD §12 v3.0)
# ---------------------------------------------------------------------------
class TwoLayerContractTests(unittest.TestCase):
    """The 4 dual-layer contracts + cross-layer dedup. Run against TwoLayerBackend
    (hot=rank_bm25, cold=InMemoryStubBackend as a JVM-free fake). Re-point the
    cold fake to PyterrierBackend to re-run the same contracts against prod."""

    # 11 (single tool call returns ONE unified full-text top-k payload)
    def test_single_shot_unified_payload(self):
        be = make_two_layer()
        be.update([{"docid": "doc-new", "title": "新记忆", "content": "独特 关键词 记忆 沉淀"}])
        res = be.search("记忆 沉淀 独特", top_k=5)
        # Every result is a SearchResult with full content + a source_layer tag.
        for r in res:
            self.assertIsInstance(r, SearchResult)
            self.assertGreater(len(r.content), 0, "single-shot must carry full text")
            self.assertIn("source_layer", r.metadata)

    # 12 (hot layer empty -> merged search still works, no crash/placeholder)
    def test_hot_layer_empty_no_crash(self):
        hot = HotLayerBackend()          # nothing written to hot yet
        cold = InMemoryStubBackend(Path(tempfile.gettempdir()) / "c12.pkl")
        cold.update([{"docid": "d1", "title": "Kafka", "content": "Kafka 分布式 事件流"}])
        be = TwoLayerBackend(hot, cold)
        res = be.search("Kafka 事件流", top_k=3)
        # Cold answers; hot contributes nothing but must not break the merge.
        self.assertTrue(any(r.metadata.get("source_layer") == "cold" for r in res))

    # 13 (in-flight query during a rebuild sees ONE consistent generation)
    def test_rebuild_inflight_consistent(self):
        be = make_two_layer(settle_threshold=1)   # settle on every update
        results_holder: dict = {}
        def worker():
            results_holder["res"] = be.search("数据管道 架构", top_k=3)
        t = threading.Thread(target=worker)
        t.start()
        for i in range(50):
            be.update([{"docid": f"mem{i}", "title": "x",
                        "content": "数据管道 架构 沉淀"}])
        t.join()
        # Every result — hot OR cold — carries a shared_generation tag (the
        # dual-layer RCU marker). A torn read would surface two different
        # shared_generation values in one result set.
        res = results_holder["res"]
        shared = {r.metadata.get("shared_generation") for r in res
                  if "shared_generation" in r.metadata}
        self.assertTrue(all("shared_generation" in r.metadata for r in res),
                        "results must carry the shared_generation RCU tag")
        self.assertLessEqual(len(shared), 1,
                             "in-flight query saw a torn shared_generation mix")

    # 14 (generation handoff: a settled doc is not double-counted in both layers)
    def test_settle_generation_atomic_handoff(self):
        be = make_two_layer(settle_threshold=1)   # settle fires on the 2nd write
        be.update([{"docid": "doc-1", "title": "A", "content": "记忆 沉淀 唯一 标记"}])
        be.update([{"docid": "doc-2", "title": "B", "content": "记忆 沉淀 唯一 标记 二"}])
        # threshold=1: after the 2nd write hot has 2 > 1 -> oldest (doc-1) is
        # settled to cold and dropped from hot. Both docs must still be findable.
        res = be.search("唯一 标记", top_k=10)
        ids = [r.doc_id for r in res]
        # No external docid may appear twice (hot + cold double-count).
        self.assertEqual(len(ids), len(set(ids)), f"double-counted docids: {ids}")
        # The settled doc (doc-1) must still be reachable via the cold layer.
        self.assertIn("doc-1", ids, "settled doc vanished after handoff")

    # 15 (cross-layer dedup is by EXTERNAL docid, not Terrier internal docno)
    def test_cross_layer_dedup_by_external_docid(self):
        be = make_two_layer(settle_threshold=2)
        # Write doc-a into hot, then settle it so a copy also lives in cold.
        be.update([{"docid": "doc-a", "title": "数据管道架构",
                    "content": "数据管道 批处理 流处理 Lambda 架构"}])
        be.update([{"docid": "doc-z", "title": "filler", "content": "无关 填充 词"}])
        res = be.search("数据管道 架构", top_k=10)
        a_hits = [r for r in res if r.doc_id == "doc-a"]
        # doc-a exists in both layers during the window; dedup must collapse it.
        self.assertLessEqual(len(a_hits), 1,
                             "external docid dedup failed: doc-a returned multiple times")


# ---------------------------------------------------------------------------
# Real cold-layer coverage (@reviewer gap: the 15/15 ran only on stub fakes,
# so not a single test exercised the REAL rank_bm25 cold path). These run
# PyterrierBackend (JVM-free in-memory + pickle) directly.
# ---------------------------------------------------------------------------
class ColdLayerRealTests(unittest.TestCase):
    """Real cold layer, no JVM, no stub — the in-memory rank_bm25 path."""

    def _cold(self, name: str) -> PyterrierBackend:
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        return PyterrierBackend(Path(d.name) / "idx")

    def _seed(self, be: PyterrierBackend) -> None:
        be.update([
            {"docid": "d1", "title": "数据管道架构",
             "content": "数据管道 批处理 流处理 Lambda 架构"},
            {"docid": "d2", "title": "Docker 容器",
             "content": "Docker 开源 容器引擎 轻量 可移植"},
            {"docid": "d3", "title": "Kafka 消息队列",
             "content": "Kafka 分布式 事件流 高吞吐 低延迟"},
        ])

    # ① pickle round-trip: persist -> new instance load -> old docs searchable
    def test_cold_persist_load_roundtrip(self):
        be = self._cold("c1")
        self._seed(be)
        path = be.persist()          # .../idx/cold_corpus.pkl
        be2 = PyterrierBackend(Path(path).parent)  # parent IS the base dir
        self.assertTrue(be2.load())
        # Every seeded doc is reachable after a fresh instance reloads.
        self.assertEqual(be2.search("数据管道 架构", top_k=1)[0].doc_id, "d1")
        self.assertEqual(be2.search("Kafka 消息队列", top_k=1)[0].doc_id, "d3")
        self.assertEqual(be2.generation, be.generation)

    # ② the cold layer is JVM-free by construction: full search/persist/load
    #    must work with NO pyterrier import (B1 deletion). Proven behaviorally
    #    — after a full round of ops, no pyterrier module may be loaded.
    def test_cold_layer_requires_no_pyterrier(self):
        import sys
        be = self._cold("c2")
        self._seed(be)
        self.assertEqual(be.search("Docker 容器", top_k=1)[0].doc_id, "d2")
        be.persist()
        be.load()
        self.assertNotIn("pyterrier", sys.modules,
                         "cold layer must not import pyterrier")

    # ③ settle window: a doc just settled into the REAL cold layer is hit
    #    immediately (no rebuild/reload gap — the old design's bug).
    def test_settled_doc_immediately_searchable(self):
        hot = HotLayerBackend()
        cold = self._cold("c3")
        be = TwoLayerBackend(hot, cold, settle_threshold=1)  # settle every write
        be.update([{"docid": "hot-only", "title": "新记忆",
                    "content": "即时 命中 标记 甲"}])
        be.update([{"docid": "settled", "title": "沉淀记忆",
                    "content": "即时 命中 标记 乙"}])
        # 'settled' was pushed into cold by the handoff; it must be findable now.
        res = be.search("即时 命中 标记", top_k=10)
        ids = [r.doc_id for r in res]
        self.assertIn("settled", ids, "settled doc not searchable after handoff")
        self.assertEqual(len(ids), len(set(ids)), "settled doc double-counted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
