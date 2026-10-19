"""B2-(b) cross-process re-entry tests: 7 gateways, ONE shared data dir.

@reviewer's spec, verbatim requirements:
  - a REAL second process (multiprocessing) against the same directory,
    doing read + write concurrency
  - assert the LATER writer does not clobber the EARLIER writer's docs
    (the last-writer-wins hole that a bare flock leaves open — the
    rebase-before-persist design is what closes it)

The child runs through a subprocess so the test proves nothing about the
parent's in-memory state: all cross-process visibility must come from the
lock + rebase + seq-marker machinery.

Run:  python3 -m unittest test_multiproc -v
"""

import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _child_ingest(data_dir: str, docs: list) -> int:
    """Real second process: fresh service, ingest a different batch.

    Must rebase-merge the on-disk corpus (the earlier writer's docs) before
    persisting its own — otherwise the later writer clobbers the earlier
    one (the B2 silent-corpus-loss bug)."""
    from kb_service import KbSearchService
    svc = KbSearchService(Path(data_dir))
    svc.ingest(docs)
    return 0


class CrossProcessTests(unittest.TestCase):

    def _dir(self):
        d = tempfile.TemporaryDirectory(); self.addCleanup(d.cleanup)
        return Path(d.name)

    # ① later writer must NOT clobber the earlier writer's docs
    def test_later_writer_preserves_earlier_writer_docs(self):
        base = self._dir()
        # Parent (process A): first writer — these docs must survive B's write.
        from kb_service import KbSearchService
        a = KbSearchService(base)
        a.ingest([{"docid": "waveA-1", "title": "Alpha 管道",
                   "content": "alpha 管道 架构 批处理 流处理 抽取 填充 段落 若干字 扩充 内容 标记 词 一",
                   "status": "confirmed"}])

        # Real second process (B): unrelated batch, same dir.
        ctx = multiprocessing.get_context("fork")
        p = ctx.Process(target=_child_ingest,
                        args=(str(base),
                              [{"docid": "waveB-1", "title": "Beta 容器",
                                "content": "beta 容器 引擎 轻量 可移植 打包 填充 段落 若干字 扩充 内容 标记 词 二",
                                "status": "confirmed"}]))
        p.start()
        p.join(30)
        self.assertEqual(p.exitcode, 0, "child process crashed")

        # A third fresh instance must see BOTH waves.
        c = KbSearchService(base)
        self.assertEqual(
            c.search("alpha 管道", top_k=1)["data"]["results"][0]["doc_id"],
            "waveA-1", "earlier writer's doc lost to the later writer")
        self.assertEqual(
            c.search("beta 容器", top_k=1)["data"]["results"][0]["doc_id"],
            "waveB-1")

    # ② cross-process read visibility: a long-lived instance learns about
    #    another process's write via the seq marker.
    #
    #    Post-4be2560 (reviewer issue #1 deadlock fix) semantics: the
    #    in-place SH→EX seq-upgrade rebase on the READ path was removed
    #    because _probe_conflicts runs search() inside the caller's EX
    #    section — re-acquiring the flock there self-deadlocks. Read-path
    #    search() is now fully lock-free (no rebase); the rebase is owned
    #    by the WRITE path only. A long-lived instance therefore sees a
    #    peer process's new docs after ANY write-path operation (the
    #    minimal one is persist(), which takes EX and calls _rebase()
    #    before flushing) — NOT after a bare search() call. This test
    #    pins that current behaviour.
    def test_stale_reader_rebases_before_answering(self):
        base = self._dir()
        from kb_service import KbSearchService
        reader = KbSearchService(base)          # process A (long-lived)
        self.assertEqual(reader.search("gamma 标记", top_k=5)["data"]["results"],
                         [])                     # nothing yet
        # Process B writes while A's instance is alive:
        ctx = multiprocessing.get_context("fork")
        p = ctx.Process(target=_child_ingest,
                        args=(str(base),
                              [{"docid": "waveC-1", "title": "Gamma 标记",
                                "content": "gamma 标记 跨进程 可见性 测试 用例 填充 段落 若干字 扩充 内容 标记 词 三 独立甲 独立乙 独立丙 独立丁 独立戊 独立己 独立庚 独立辛 独立壬 独立癸",
                                "status": "confirmed"}]))
        p.start()
        p.join(30)
        self.assertEqual(p.exitcode, 0, "child process crashed")
        # A's stale instance re-syncs via a WRITE-path rebase (persist()
        # takes EX and merges the on-disk corpora), not via a bare
        # read-path search() (the 4be2560 deadlock fix made search()
        # lock-free / non-rebasing by design).
        reader.persist()
        self.assertEqual(
            reader.search("gamma 标记", top_k=1)["data"]["results"][0]["doc_id"],
            "waveC-1", "stale reader did not rebase on the seq marker")

    # ③ concurrent read+write does not crash and the writer's docs land
    def test_concurrent_read_write_no_crash(self):
        base = self._dir()
        from kb_service import KbSearchService
        reader = KbSearchService(base)
        reader.ingest([{"docid": "seed-1", "title": "Seed 词",
                        "content": "seed 词汇 种子 词表 初始化 填充 段落 若干字 扩充 内容 标记 词 四",
                        "status": "confirmed"}])
        ctx = multiprocessing.get_context("fork")
        procs = [
            ctx.Process(target=_child_ingest,
                        args=(str(base),
                             [{"docid": f"conc-{i}", "title": f"并发 文档 {i}",
                               "content": f"并发 文档 批次 编号 {i} 场景 填充 段落 若干字 扩充 长度 门 之上 合格 内容 标记 {extra}",
                               "status": "confirmed"}]))
            for i, extra in [
                (0, "苹果 香蕉 橙子 葡萄 核桃 栗子 松子 榛子"),
                (1, "玫瑰 百合 菊花 兰花 梅花 桃花 杏花 梨花"),
                (2, "狮子 老虎 大象 长颈鹿 斑马 河马 犀牛 水牛"),
            ]
        ]
        # Reader churns while writers run:
        for _ in range(20):
            reader.search("并发 文档", top_k=3)
        for p in procs:
            p.start()
        for p in procs:
            p.join(30)
            self.assertEqual(p.exitcode, 0, "concurrent writer crashed")
        c = KbSearchService(base)
        for i in range(3):
            self.assertIn(f"conc-{i}",
                          [r["doc_id"] for r in c.search("并发 文档",
                                                         top_k=10)["data"]["results"]])

    # ④ the flock sidecar lives in the data dir, never in the repo
    def test_lock_sidecar_in_data_dir(self):
        base = self._dir()
        from kb_service import KbSearchService
        KbSearchService(base).ingest([{"docid": "x", "title": "t",
                                        "content": "x y z"}])
        self.assertTrue((base / "lock").exists(), "flock sidecar missing")
        repo = Path(__file__).resolve().parent
        self.assertFalse((repo / "lock").exists(),
                         "lock file leaked into the repo")


if __name__ == "__main__":
    unittest.main(verbosity=2)
