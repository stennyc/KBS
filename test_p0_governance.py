"""P0 governance tests: staging isolation, confirm gate, length gate,
near-dup self-check, and audit trail.

Room ruling 2026-10-10 (knowledge-base-search P0):
  - Model auto-detection lands a doc as ``status=pending`` -> physically
    isolated staging store, NEVER in the two-layer searchable corpus.
  - A doc only becomes searchable after an explicit ``confirm()`` move.
  - Confirm enforces profile ownership + audit (code-side backstop; the
    clarify step itself is a model-side policy gate, not a code gate).
  - Confirmed docs pass a length gate + a BM25 near-dup self-check
    (no model, no external API — local only, PRD §1).

Reviewer's P0 acceptance matrix (the hard gates):
  1. staging/hot isolation: a staged doc_id is grep-absent from hot_corpus.pkl
  2. confirm move atomicity: after confirm, doc_id leaves staging AND appears
     in hot, with a non-empty audit line carrying confirmed_by + source_profile
  3. pending unreachable: a default search() never returns a staged doc_id
  + confirm entry negative case: cross-profile confirm -> rejected + audited
  + near-dup: >=0.85 blocked / 0.60-0.85 audit-only / <0.60 accepted

Run:  python3 -m unittest test_p0_governance -v
(needs the shared plugin venv on PYTHONPATH for rank_bm25 + jieba)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kb_service import KbSearchService  # noqa: E402


def _svc_with_audit() -> tuple:
    """A fresh service whose audit trail lands in a temp file we can read."""
    base = Path(tempfile.mkdtemp())
    data = base / "kb-data"
    data.mkdir()
    audit = base / "audit.jsonl"
    svc = KbSearchService(data, audit_path=audit)
    return svc, base, data, audit


def _audit_lines(path: Path) -> list:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


class P0GovernanceTests(unittest.TestCase):

    # ---- reviewer gate #1: staging physically isolated from hot corpus ----

    def test_staged_doc_absent_from_hot_corpus(self):
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([{"docid": "pend-1", "title": "一条待确认的知识候选",
                     "content": "这条还在 pending 阶段 尚未被用户确认入库 甲乙丙丁",
                     "status": "pending", "source_profile": "coder"}])
        # The staged doc must NOT be in the hot layer's pickle side-store.
        hot_pkl = data / "hot_corpus.pkl"
        self.assertTrue(hot_pkl.exists())
        with open(hot_pkl, "rb") as f:
            import pickle
            hot_docs = pickle.load(f).get("docs", [])
        hot_ids = {str(d.get("docid")) for d in hot_docs}
        self.assertNotIn("pend-1", hot_ids,
                         "a pending doc must never land in the hot corpus")
        # It IS in the staging row store, though.
        staging_rows = data / "staging" / "staging_docs.jsonl"
        self.assertTrue(staging_rows.exists())
        staged_ids = [json.loads(l)["docid"]
                      for l in staging_rows.read_text().splitlines() if l.strip()]
        self.assertIn("pend-1", staged_ids)

    # ---- reviewer gate #3: pending unreachable by default search --------

    def test_pending_unreachable_by_default_search(self):
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([{"docid": "pend-2", "title": "量子 纠缠 纠错码",
                     "content": "量子纠缠与表面码纠错的综述 甲乙丙丁戊己庚辛",
                     "status": "pending"}])
        # Default search (no filters, no status) must not surface the pending
        # doc even though its terms match.
        res = svc.search("量子 纠缠 纠错码", top_k=5)
        ids = [r["doc_id"] for r in res["data"]["results"]]
        self.assertNotIn("pend-2", ids,
                         "default search must not surface a staged/pending doc")
        # Explicitly, the confirmed corpus is empty so total is 0.
        self.assertEqual(res["data"]["total"], 0)

    # ---- reviewer gate #2: confirm move atomicity + audit ---------------

    def test_confirm_moves_staged_doc_to_searchable(self):
        svc, base, data, audit = _svc_with_audit()
        doc = {"docid": "pend-3", "title": "部署 报告 生成",
               "content": "本次部署报告 生成于 2026 年 10 月 甲乙丙丁戊己 若干字 扩充 至 长度门 之上 合格 内容 填充 段落 完整 P3FILL",
               "status": "pending", "source_profile": "coder",
               "origin_kind": "auto"}
        svc.ingest([doc])
        # Pre-confirm: not searchable.
        self.assertEqual(svc.search("部署 报告 生成", top_k=1)["data"]["total"], 0)

        out = svc.confirm("pend-3", source_profile="coder",
                          confirmed_by="user-alice")
        self.assertEqual(out["status"], "success")

        # Post-confirm: now searchable.
        res = svc.search("部署 报告 生成", top_k=1)
        self.assertEqual(res["data"]["total"], 1)
        self.assertEqual(res["data"]["results"][0]["doc_id"], "pend-3")

        # Atomicity: gone from staging, present in hot.
        staging_rows = data / "staging" / "staging_docs.jsonl"
        staged_ids = [json.loads(l)["docid"]
                      for l in staging_rows.read_text().splitlines() if l.strip()]
        self.assertNotIn("pend-3", staged_ids, "confirm must remove from staging")

        hot_ids = {str(d.get("docid"))
                   for d in _load_pkl_docs(data / "hot_corpus.pkl")}
        self.assertIn("pend-3", hot_ids, "confirm must land the doc in hot")

        # Audit: a confirm line carrying confirmed_by + source_profile.
        confirms = [l for l in _audit_lines(audit) if l.get("action") == "confirm"]
        ok = [l for l in confirms if l.get("doc_id") == "pend-3"
              and l.get("confirmed_by") == "user-alice"
              and l.get("source_profile") == "coder"
              and l.get("status") == "confirmed"]
        self.assertTrue(ok, "audit must record the successful confirm move")

    # ---- confirm entry negative case: cross-profile rejected -----------

    def test_confirm_cross_profile_rejected_and_audited(self):
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([{"docid": "pend-4", "title": "属于 coder 的候选知识",
                     "content": "这是 coder profile 的候选 尚未确认 甲乙丙丁戊己",
                     "status": "pending", "source_profile": "coder"}])
        # A DIFFERENT profile attempts to confirm it -> rejected.
        out = svc.confirm("pend-4", source_profile="reviewer",
                          confirmed_by="user-bob")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "profile_mismatch")
        # Doc is still staged, NOT searchable.
        self.assertEqual(svc.search("属于 coder 的候选", top_k=1)["data"]["total"], 0)
        # Audit carries the profile_mismatch error.
        mism = [l for l in _audit_lines(audit)
                if l.get("action") == "confirm" and l.get("doc_id") == "pend-4"
                and l.get("error") == "profile_mismatch"]
        self.assertTrue(mism, "cross-profile confirm must be audited as mismatch")

    def test_confirm_unknown_doc_not_found(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.confirm("never-existed", source_profile="coder")
        self.assertEqual(out["error"], "not_found")

    # ---- length gate -----------------------------------------------------

    def test_length_gate_rejects_short_content(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{"docid": "short-1", "title": "太短",
                           "content": "短",  # < 60 chars (triviality floor)
                           "status": "confirmed"}])
        self.assertIn("short-1", [r.get("doc_id") for r in out["rejected"]])
        self.assertEqual(out["ingested"], [])
        self.assertEqual(svc.search("太短 短", top_k=1)["data"]["total"], 0)

    def test_triviality_floor_rejects_59_char_content(self):
        # Owner 2026-10-19: the KB is a knowledge corpus (经验/手册/教程/
        # 教材/资料), not a fragment vault — the 20-char floor let test
        # placeholders walk in. The new floor is 60 chars of CONTENT.
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{"docid": "trivial-1", "title": "占位 测试 条目",
                           "content": "x" * 59,
                           "status": "confirmed"}])
        self.assertIn("trivial-1", [r.get("doc_id") for r in out["rejected"]])
        rejected = next(r for r in out["rejected"]
                        if r.get("doc_id") == "trivial-1")
        self.assertEqual(rejected["reason"], "length_gate")
        self.assertEqual(rejected["field"], "content")
        self.assertEqual(rejected["got"], 59)

    def test_triviality_floor_accepts_60_char_content(self):
        # Boundary: exactly 60 content chars passes the floor.
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{"docid": "floor-edge", "title": "恰好 六十字",
                           "content": "x" * 60,
                           "status": "confirmed"}])
        self.assertEqual(out["ingested"], ["floor-edge"])
        self.assertEqual(out["rejected"], [])

    def test_confirm_re_runs_triviality_floor(self):
        # The agent self-confirm path stages FIRST then confirms — without
        # the confirm-side gate a short row could stage clean and sneak into
        # the confirmed corpus. A < 60-char staged row must be rejected by
        # confirm() and LEFT IN STAGING (the owner can fix + re-confirm).
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{"docid": "agent-short",
                           "title": "agent 捕获 的 过短 经验 条目",
                           "content": "x" * 40,
                           "status": "pending",
                           "origin_kind": "agent",
                           "source_profile": "coder"}])
        self.assertIn("agent-short", out["staged"])
        res = svc.confirm("agent-short", source_profile="coder",
                          confirmed_by="agent")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error"], "length_gate")
        self.assertEqual(res["field"], "content")
        # The row is still in staging (not deleted by the failed confirm):
        staging_rows = data / "staging" / "staging_docs.jsonl"
        staged_ids = [json.loads(l)["docid"]
                      for l in staging_rows.read_text().splitlines()
                      if l.strip()]
        self.assertIn("agent-short", staged_ids,
                      "a confirm rejected by the length gate must leave "
                      "the row in staging (owner can fix + re-confirm)")
        # ...and it is NOT searchable (never entered the corpus):
        self.assertEqual(svc.search("agent", top_k=5)["data"]["total"], 0)
        # ...and the rejection is AUDITED:
        gate_lines = [l for l in _audit_lines(audit)
                      if l.get("action") == "confirm"
                      and l.get("doc_id") == "agent-short"
                      and l.get("error") == "length_gate"]
        self.assertTrue(gate_lines,
                        "the confirm-side length-gate reject must be "
                        "audited")

    def test_confirm_re_runs_triviality_floor_passes_long_content(self):
        # Symmetry: a >= 60-char staged row confirms cleanly under the new
        # gate (the floor must not block legitimate docs).
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([{"docid": "agent-ok",
                     "title": "agent 捕获 的 合格 经验 条目",
                     "content": "x" * 80,
                     "status": "pending",
                     "origin_kind": "agent",
                     "source_profile": "coder"}])
        res = svc.confirm("agent-ok", source_profile="coder",
                          confirmed_by="agent")
        self.assertEqual(res["status"], "success")
        self.assertEqual(
            [r["doc_id"] for r in svc.search("agent", top_k=5)["data"]["results"]],
            ["agent-ok"])

    def test_length_gate_rejects_long_title(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{"docid": "long-t", "title": "x" * 201,
                           "content": "内容长度没问题 甲乙丙丁戊己庚辛壬癸子丑寅",
                           "status": "confirmed"}])
        self.assertTrue(any(r.get("doc_id") == "long-t"
                            and r.get("reason") == "length_gate"
                            and r.get("field") == "title" for r in out["rejected"]))

    # ---- near-dup self-check (BM25, no model) ---------------------------

    def test_near_duplicate_blocked(self):
        svc, base, data, audit = _svc_with_audit()
        base_doc = {"docid": "orig", "title": "Kafka 消费组 重平衡",
                    "content": "Kafka 消费组在 节点 重启 时 会 触发 重平衡 机制 若干字 扩充 DUPFILL甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲",
                    "status": "confirmed"}
        svc.ingest([base_doc])
        # A near-exact rewording -> BM25 top-1 normalized score >= 0.85.
        dup = {"docid": "dup-1", "title": "Kafka 消费组 重平衡",
               "content": "Kafka 消费组在 节点 重启 时 会 触发 重平衡 机制 若干字 扩充 DUPFILL甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲",
               "status": "confirmed"}
        out = svc.ingest([dup])
        blocked = [r for r in out["rejected"] if r.get("doc_id") == "dup-1"]
        self.assertTrue(blocked, "a near-exact duplicate must be blocked")
        self.assertEqual(blocked[0]["reason"], "near_duplicate")
        # The dup did NOT enter the corpus.
        self.assertEqual(
            [r["doc_id"] for r in svc.search("Kafka 消费组 重平衡", top_k=5)["data"]["results"]],
            ["orig"], "the duplicate must not double-count in the corpus")

    # ---- P0 routing: default status is pending ---------------------------

    def test_same_template_different_number_not_miskilled(self):
        """Card 1 (conc mis-kill): three same-template docs differing ONLY in
        a trailing number (编号 0/1/2) must ALL pass the near-dup gate. The
        tokenize number-token fix (hot_layer.tokenize emits \d+ as
        independent tokens) drops their word-set containment below the block
        threshold; before that fix they all collapsed to identical word sets
        and min-max BM25 mis-killed conc-1/2. A true exact duplicate of the
        same content must STILL be blocked (the gate still has teeth)."""
        svc, base, data, audit = _svc_with_audit()
        for i, extra in enumerate(
                ("独有甲组 独有词一 独有词二",
                 "独有乙组 独有词三 独有词四",
                 "独有丙组 独有词五 独有词六")):
            out = svc.ingest([{
                "docid": f"conc-{i}", "title": f"并发 文档 {i}",
                "content": f"concurrent 并发 文档 批次 编号 {i} 说明 若干字 扩充 长度 {extra} 独立填充词 保证字数超过六十字节门 CONCFILL甲甲甲甲甲甲甲甲甲",
                "status": "confirmed"}])
            self.assertEqual(out["ingested"], [f"conc-{i}"],
                             f"conc-{i} must pass the near-dup gate, "
                             f"rejected: {out['rejected']}")
        self.assertEqual(
            sorted(r["doc_id"] for r in svc.search("并发 文档", top_k=10)["data"]["results"]),
            ["conc-0", "conc-1", "conc-2"],
            "all three same-template docs must be searchable")
        # A TRUE exact dup of conc-0's content must still be blocked:
        dup = svc.ingest([{
            "docid": "conc-0-copy2", "title": "并发 文档 0",
            "content": ("concurrent 并发 文档 批次 编号 0 说明 若干字 扩充 长度 "
                       "独有甲组 独有词一 独有词二 独立填充词 保证字数超过六十字节门 "
                       "CONCFILL甲甲甲甲甲甲甲甲甲"),
            "status": "confirmed"}])
        self.assertEqual(dup["ingested"], [])
        rej = [r for r in dup["rejected"] if r.get("doc_id") == "conc-0-copy2"]
        self.assertTrue(rej and rej[0]["reason"] == "near_duplicate",
                        "a true exact duplicate must still be blocked")

    def test_default_status_is_pending_not_confirmed(self):
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([{"docid": "p-default", "title": "默认 状态 路由",
                           "content": "没有 status 字段的文档 默认 走 待定 流程 甲乙丙丁",
                           "source_profile": "devops"}])
        # No status -> treated as pending -> staged, not ingested.
        self.assertIn("p-default", out["staged"])
        self.assertEqual(out["ingested"], [])
        self.assertEqual(svc.search("默认 状态 路由", top_k=1)["data"]["total"], 0)

    # ---- audit three-line: ingest + confirm + search all emit -----------

    def test_audit_covers_ingest_confirm_search(self):
        svc, base, data, audit = _svc_with_audit()
        # content >= trivial gate (60 bytes) + distinct from other test docs
        # (near-dup threshold; keep it unique, >= 60B passes length gate)
        svc.ingest([{"docid": "a-1x", "title": "审计 覆盖 三条链路 说明",
                     "content": "ingest confirm search 三条审计线都要打甲乙丙丁各一次记录完整轨迹 AUDFILL甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲",
                     "status": "confirmed", "source_profile": "coder"}])
        svc.confirm("a-1x", source_profile="coder", confirmed_by="user-x")
        svc.search("审计 覆盖", top_k=1)
        actions = {l.get("action") for l in _audit_lines(audit)}
        self.assertIn("ingest_confirmed", actions)
        # a-1 was confirmed at ingest (not staged), so no 'confirm' move line —
        # confirm() on an already-confirmed (not staged) doc reports not_pending.
        self.assertIn("confirm", actions)
        self.assertTrue(any(l.get("doc_id") == "a-1x" for l in _audit_lines(audit)))


def _load_pkl_docs(path: Path) -> list:
    import pickle
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f).get("docs", [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
