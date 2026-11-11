"""refresh() 显式 rebase 入口验收测试（PRD: KB 读路径可见性缺口）。

4be2560（reviewer issue #1 死锁修复）把 search() 的原位 SH→EX rebase 删掉后，
长活纯读实例（只调 search()/get()、从不 ingest/confirm）会一直 stale，直到
某次写操作触发 _rebase()。refresh() 是 owner 拍板的显式 re-sync 门：
  - EX-lock，rebase confirmed 两层语料 + staging 行两个面
  - 不 flush、不 bump seq、不触发 graph rebuild —— 纯读侧重同步，
    一个纯读进程可以反复调 refresh() 而永不变成写者
  - 返回 {status, hot_gen, cold_gen, staging_gen}（可观测面，无数据体）

Run:
  python3 -m pytest test_kb_refresh.py -v
"""

import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kb_service import KbSearchService  # noqa: E402


def _fresh_data_dir() -> Path:
    base = Path(tempfile.mkdtemp())
    data = base / "kb-data"
    data.mkdir()
    return data


class RefreshVisibilityTests(unittest.TestCase):
    """纯读实例 + 子进程写入 + refresh() -> 立刻可见（无其他写操作介入）。"""

    def test_read_only_instance_sees_peer_write_after_refresh(self):
        data_dir = _fresh_data_dir()
        # 长活"纯读"实例：创建后只做 search()，不做任何 ingest/confirm/persist
        reader = KbSearchService(data_dir)
        self.assertEqual(
            reader.search("缺口 标记", top_k=5)["data"]["results"], [],
            "fresh reader sees nothing before the peer writes")

        # 子进程写入（和 test_multiproc.py 同一个 _child_ingest 形态）
        import multiprocessing

        def _child_ingest():
            peer = KbSearchService(data_dir)
            peer.ingest([{"docid": "gap-1", "title": "缺口 标记 文档",
                          "content": "这是跨进程 可见性缺口 测试 用的 填充 内容 若干字",
                          "status": "confirmed"}])
            peer.persist()

        ctx = multiprocessing.get_context("fork")
        p = ctx.Process(target=_child_ingest)
        p.start()
        p.join(30)
        self.assertEqual(p.exitcode, 0, "child writer process crashed")

        # 关键断言：refresh() 之前，纯读实例仍看不到（读路径不 rebase 是 4be2560 的既定行为）
        self.assertEqual(
            reader.search("缺口 标记", top_k=5)["data"]["results"], [],
            "before refresh(), the read-only instance must still be stale "
            "(4be2560: read path does not rebase by design)")

        out = reader.refresh()
        self.assertEqual(out["status"], "success")
        self.assertIn("hot_gen", out)
        self.assertIn("cold_gen", out)
        self.assertIn("staging_gen", out)

        hit = reader.search("缺口 标记", top_k=1)
        self.assertEqual(
            hit["data"]["results"][0]["doc_id"], "gap-1",
            "after refresh(), the read-only instance must see the peer's "
            "confirmed doc without any write-path call of its own")

    def test_refresh_also_rebases_staging_face(self):
        data_dir = _fresh_data_dir()
        reader = KbSearchService(data_dir)
        reader.ingest([{"docid": "seed-1", "title": "种子 文档",
                        "content": "种子 词表 初始化 填充 内容 若干甲甲甲甲甲甲",
                        "status": "confirmed"}])

        def _child_stage():
            peer = KbSearchService(data_dir)
            peer.ingest([{"docid": "gap-pend-1", "title": "待确认 缺口 文档",
                          "content": "这条 是 待确认 的 缺口 候选 填充 若干字 扩充至长度门下限之上",
                          "status": "pending"}])
            peer.persist()

        ctx = multiprocessing.get_context("fork")
        p = ctx.Process(target=_child_stage)
        p.start()
        p.join(30)
        self.assertEqual(p.exitcode, 0)

        # refresh() 之前：纯读实例的 staging 面看不到别人刚 staged 的行
        self.assertIsNone(reader._staging.find_active("gap-pend-1"))

        reader.refresh()
        row = reader._staging.find_active("gap-pend-1")
        self.assertIsNotNone(row,
                             "after refresh(), the peer's pending staging "
                             "row must be visible in this process's staging "
                             "view (staging rebase runs alongside the "
                             "confirmed-corpus rebase in refresh())")
        self.assertEqual(row["docid"], "gap-pend-1")


class RefreshNoWriteSideEffectTests(unittest.TestCase):
    """refresh() 必须是一个"纯读重同步"门：反复调不能产生任何写侧
    副作用（seq 不涨、语料文件不重写、reinforce/graph 侧带不动）。"""

    def test_refresh_does_not_bump_seq(self):
        data_dir = _fresh_data_dir()
        svc = KbSearchService(data_dir)
        svc.ingest([{"docid": "seed-1", "title": "种子 文档",
                     "content": "种子 词表 初始化 填充 内容 若干甲甲甲甲甲甲",
                     "status": "confirmed"}])
        seq_before = svc._seq
        seq_file = data_dir / "seq"
        seq_file_val = int(seq_file.read_text().strip()) if seq_file.exists() else 0

        for _ in range(3):
            out = svc.refresh()
            self.assertEqual(out["status"], "success")

        self.assertEqual(svc._seq, seq_before,
                         "refresh() must not bump this instance's in-memory "
                         "seq (it is a read-side re-sync, not a write)")
        seq_file_val_after = int(seq_file.read_text().strip()) if seq_file.exists() else 0
        self.assertEqual(seq_file_val_after, seq_file_val,
                         "refresh() must not write to the on-disk seq marker "
                         "file (no _flush() call)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
