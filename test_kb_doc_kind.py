"""doc_kind 形态轴验收测试（PRD: KB 文档形态轴 experience/tutorial/reference）。

PRD 不变式（每条测试锚定一条）：
  1. tutorial filter：ingest doc_kind='tutorial' confirmed 行 -> search
     filters={doc_kind:'tutorial'} 命中；filters={doc_kind:'reference'} 不命中
  2. __unset__ sentinel：不传 doc_kind 的行 -> 该行 doc_kind=None，
     filters={doc_kind:'__unset__'} 命中（None ≡ key absent 统一语义）
  3. docid 字节稳定：同一内容，doc_kind=None 显式传入 vs key 完全缺席，
     _deterministic_docid 输出字节相同（锚 gov_tier B1 None-filter 先例）
  4. doc_kind 增量零漂移（修正版验收 d）：对任意一篇 doc dict，
     "加 doc_kind=None key" 前后重算结果字节相同。注意——这条 NOT 等于
     "存量全量重算 docid 等于存盘 docid"：真实语料里存在一次历史
     docid 迁移（docid_migration_map.json，443 条 old->new 映射，
     约 512/696 篇存盘 docid 是按旧公式/手工命名生成、未按当前 B1
     公式重算的），这是迁移遗留状态，与 doc_kind 改动无关。
     本测试只钉"本次改动自身引入的增量漂移 = 0"。
  5. topic_search 面回归：knowledge_base_topic_search 行为不变
     （无 doc_kind 参数，agent 分区查询与改动前字节等价）

运行方式（同目录其他 test_*.py 的约定）：
  /root/.hermes/plugin-venv/bin/python -m pytest test_kb_doc_kind.py -v
"""

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kb_service import KbSearchService, _deterministic_docid  # noqa: E402


def _svc() -> tuple:
    base = Path(tempfile.mkdtemp())
    data = base / "kb-data"
    data.mkdir()
    audit = base / "audit.jsonl"
    svc = KbSearchService(data, audit_path=audit)
    return svc, data


class DocKindAxisTests(unittest.TestCase):
    """PRD 不变式 1-2：doc_kind 作为 filter key 的命中/不命中 + __unset__ 语义。"""

    def test_tutorial_filter_hits_and_reference_does_not(self):
        svc, data = _svc()
        out = svc.ingest([{
            "docid": "dk-tutorial",
            "title": "教程形态 部署手册 步骤一二三",
            "content": "这篇是可照做的操作手册 第一步 第二步 第三步 部署 若干字 填充 段落 覆盖 场景 DKFILL甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲",
            "status": "confirmed",
            "doc_kind": "tutorial",
        }])
        self.assertIn("dk-tutorial", out["ingested"],
                      "a confirmed doc_kind=tutorial doc must enter the corpus")
        hit = svc.search("部署 手册", filters={"doc_kind": "tutorial"})
        self.assertTrue(
            any(r["doc_id"] == "dk-tutorial" for r in hit["data"]["results"]),
            "filters={doc_kind:'tutorial'} must hit the tutorial doc")
        miss = svc.search("部署 手册", filters={"doc_kind": "reference"})
        self.assertFalse(
            any(r["doc_id"] == "dk-tutorial" for r in miss["data"]["results"]),
            "filters={doc_kind:'reference'} must NOT hit the tutorial doc")

    def test_unset_sentinel_matches_unlabeled_row(self):
        svc, data = _svc()
        out = svc.ingest([{
            "docid": "dk-unlabeled",
            "title": "未标注形态 的经验文档 测试用",
            "content": "这篇没有标注 doc_kind 只有 标题 和 正文 内容 若干字 扩充 长度门 覆盖 场景 说明 DKU1FILL甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲",
            "status": "confirmed",
        }])
        self.assertIn("dk-unlabeled", out["ingested"])
        hit = svc.search("标注 形态", filters={"doc_kind": "__unset__"})
        self.assertTrue(
            any(r["doc_id"] == "dk-unlabeled" for r in hit["data"]["results"]),
            "filters={doc_kind:'__unset__'} must match a row whose doc_kind "
            "is None (None ≡ key absent, B1 unified semantics)")


class DocKindDocidStabilityTests(unittest.TestCase):
    """PRD 不变式 3：docid 字节稳定（B1 None-filter，锚 gov_tier 先例）。"""

    def test_doc_kind_none_explicit_vs_absent_same_docid(self):
        base = {"title": "稳定 性 测试 文档",
                "content": "同一份内容 用来验证 docid 不因 缺省值 而变 若干字"}
        id_with_none = _deterministic_docid({**base, "doc_kind": None})
        id_absent = _deterministic_docid(base)
        self.assertEqual(id_with_none, id_absent,
                         "doc_kind=None explicit must hash byte-identical "
                         "to doc_kind key absent (B1 None-filter, the "
                         "gov_tier precedent)")

    def test_doc_kind_non_none_changes_docid(self):
        # 反向钉：缺省值若误选成 "experience" 而非 None，docid 会变
        # （这正是 PRD 红线要防的写法，这里留一个断言证明该选择是
        # load-bearing 的——换成任何非 None 缺省值测试 3 都会变红）。
        base = {"title": "稳定 性 测试 文档",
                "content": "同一份内容 用来验证 docid 不因 缺省值 而变 若干字"}
        id_exp = _deterministic_docid({**base, "doc_kind": "experience"})
        id_absent = _deterministic_docid(base)
        self.assertNotEqual(id_exp, id_absent,
                            "a non-None default (e.g. 'experience') would "
                            "change the docid — the PRD red line this test "
                            "anchors")


class DocKindStagingRoundtripTests(unittest.TestCase):
    """PRD 不变式 4：staging jsonl roundtrip + confirm() 跨层路径的 docid 一致性。

    data-manager 独有洞：pending 行落盘时 doc_kind=None 被 jsonl 写成 null，
    roundtrip 后 key 变成"存在且值为 None"；confirm() 把该行从 staging 移入
    confirmed corpus 时，若 B1 过滤失效，这行 dict 会和"key 完全缺席"的等价
    文档算出不同 docid —— 这里钉死跨层路径。
    """

    def test_confirm_preserves_docid_after_jsonl_roundtrip(self):
        svc, data = _svc()
        # 先算出"key 缺席"等价文档的确定性 docid（roundtrip 前）
        base_doc = {
            "title": "跨层 一致性 测试 文档",
            "content": "这篇会走 pending -> jsonl roundtrip -> confirm 的 路径 若干字 扩充 至长度门 覆盖 场景 说明 DOKIDFILL",
            "status": "pending",
        }
        docid_expected = _deterministic_docid(base_doc)

        out = svc.ingest([{**base_doc, "docid": docid_expected,
                           "doc_kind": None,
                           "source_profile": "coder"}])
        self.assertIn(docid_expected, out["staged"],
                      "an unlabeled pending doc must stage with its "
                      "deterministic docid")
        # jsonl roundtrip：读回落盘行，确认 doc_kind 存成了 null 值
        staging_rows = data / "staging" / "staging_docs.jsonl"
        rows = [json.loads(l) for l in
                staging_rows.read_text().splitlines() if l.strip()]
        staged_row = next(r for r in rows if r.get("docid") == docid_expected)
        self.assertIn("doc_kind", staged_row,
                      "jsonl roundtrip materializes doc_kind=None as a "
                      "present key (null) — the premise of this test")
        self.assertIsNone(staged_row["doc_kind"])

        move = svc.confirm(docid_expected, source_profile="coder")
        self.assertEqual(move.get("status"), "success",
                         "the pending row must confirm cleanly")
        # confirm 后该行进 confirmed corpus，docid 必须与 roundtrip 前一致
        hit = svc.search("跨层 一致性", filters={"doc_kind": "__unset__"})
        self.assertTrue(
            any(r["doc_id"] == docid_expected for r in hit["data"]["results"]),
            "after confirm, the doc must be reachable under its ORIGINAL "
            "deterministic docid (no re-mint across the staging->confirmed "
            "cross-layer move)")


class DocKindTopicSearchRegressionTests(unittest.TestCase):
    """PRD 不变式 5：knowledge_base_topic_search 面行为不变回归。

    该面只读 origin_kind=agent 分区，doc_kind 不是它的参数（PRD 明确
    topic_search 面不动）。这里直接调它背后的 service 调用路径
    （topic_search_service_args + svc.search），确认 doc_kind 的存在
    不改变 agent 分区检索结果。
    """

    def test_agent_partition_search_unaffected_by_doc_kind(self):
        import os
        from unittest import mock
        from kb_ingest_hook import topic_search_service_args
        # Multiplex fix: topic_search_service_args stamps the caller's
        # profile into the filters (_current_profile); this corpus is
        # intentionally GLOBAL (no source_profile) and pins the doc_kind
        # axis only, so run it as a no-profile caller — the same no-signal
        # posture as the CLI leg (host helper stubbed, HERMES_HOME blanked).
        old_home = os.environ.get("HERMES_HOME")
        old_profile = os.environ.get("HERMES_PROFILE")
        os.environ.pop("HERMES_HOME", None)
        os.environ.pop("HERMES_PROFILE", None)
        svc, data = _svc()
        try:
            with mock.patch.dict(sys.modules, {"hermes_constants": None}):
                out = svc.ingest([{
                    "docid": "dk-agent-1",
                    "title": "agent 经验 部署 踩坑 记录",
                    "content": "这是一条 agent 经验 部署 时踩的坑 若干字 内容 DKA1FILL甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲",
                    "status": "confirmed",
                    "origin_kind": "agent",
                    "topic": "deployment",
                    "doc_kind": "tutorial",
                }, {
                    "docid": "dk-agent-2x",
                    "title": "agent 经验 部署 另一条",
                    "content": "第二条 agent 经验 部署 相关 内容 若干字 不同 DKA2FILL甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲甲",
                    "status": "confirmed",
                    "origin_kind": "agent",
                    "topic": "deployment",
                }])
            self.assertEqual(sorted(out["ingested"]), ["dk-agent-1", "dk-agent-2x"])
            kwargs = topic_search_service_args("部署", "deployment", 5)
            self.assertEqual(kwargs["filters"].get("origin_kind"), "agent")
            self.assertEqual(kwargs["filters"].get("topic"), "deployment")
            res = svc.search(**kwargs)
            ids = {r["doc_id"] for r in res["data"]["results"]}
            self.assertEqual(ids, {"dk-agent-1", "dk-agent-2x"},
                             "the agent-partition topic search must be "
                             "independent of doc_kind (both docs hit regardless "
                             "of whether doc_kind is set)")
            # 显式钉回归语义：不带任何 filter 的同一查询也必须命中同样的两条
            # （agent 分区检索结果不随 doc_kind 轴出现/消失而变化）
            res_bare = svc.search("部署", top_k=5)
            self.assertEqual(
                {r["doc_id"] for r in res_bare["data"]["results"]},
                {"dk-agent-1", "dk-agent-2x"})
        finally:
            if old_home is not None:
                os.environ["HERMES_HOME"] = old_home
            if old_profile is not None:
                os.environ["HERMES_PROFILE"] = old_profile


# ---------------------------------------------------------------------------
# 存量回归（PRD 验收测试 d，2026-10-18 owner-locked 修正版）：
# doc_kind 改动零新增漂移断言 —— 不是"全量重算 0 差异"（那个前提错误：
# 语料里 510/693 篇 docid 是 docid_migration_map.json 迁移历史里的旧值，
# 重算当前公式本来就不相等，与本次改动无关），而是：
#   对任意一篇存量 doc dict，"显式加 doc_kind=None key" 前后，用当前
#   公式重算的 docid 必须字节相同（B1 None-filter 兜底）。
# 这是本次改动能保证的唯一正确不变式；用它做全量扫描。
#
#   /root/.hermes/plugin-venv/bin/python test_kb_doc_kind.py --regression \
#       --data-dir /root/.hermes/kb-data
# ---------------------------------------------------------------------------

def regression_no_new_drift(data_dir: Path) -> int:
    """扫描 data_dir/hot_corpus.pkl 全量 confirmed 文档：
    对每篇，重算 [现状 dict] 的 docid 与 [加 doc_kind=None key 后 dict]
    的 docid，两者必须字节相同（B1 None-filter 保证）。任何一篇不等
    都是 doc_kind 改动引入的新漂移，直接 assert 抛错。"""
    pkl = data_dir / "hot_corpus.pkl"
    with open(pkl, "rb") as f:
        store = pickle.load(f)
    docs = store.get("docs", [])
    drifted = []
    for d in docs:
        stripped = {k: v for k, v in d.items() if k != "docid"}
        id_before = _deterministic_docid(stripped)
        # 模拟 _normalize_docs 的 setdefault 语义（NOT 无条件覆盖）：
        # 仅当 doc_kind key 物理缺失时补 None（None 被 B1 filter 排除，
        # id 不变）；已有非 None doc_kind（如 'tutorial'/'experience'/
        # 'reference'）的行 key 物理存在，setdefault 不触发 → id 同样
        # 不变。旧版 bug：`sim_after['doc_kind'] = None` 会无条件覆盖
        # 已有非 None doc_kind 值 → 带 doc_kind 的行必然 drift，误报。
        sim_after = dict(stripped)
        sim_after.setdefault("doc_kind", None)
        id_after = _deterministic_docid(sim_after)
        if id_before != id_after:
            drifted.append(str(d.get("docid")))
    assert not drifted, (
        f"DOCID NEW-DRIFT (doc_kind regression): {len(drifted)} docs "
        f"rehash differently AFTER adding doc_kind=None than before "
        f"(first 5: {drifted[:5]}) — the doc_kind change introduced a "
        f"docid drift this setdefault was supposed to prevent")
    return len(docs)


class DocKindCorpusRegressionTests(unittest.TestCase):
    """验收测试 d（pytest 形态，reviewer nit 3：从脚本提升为 CI 用例）：
    真实语料上 doc_kind=None 增量零漂移。

    需要 KB_REGRESSION_DATA_DIR 环境变量指向真实数据目录（默认
    /root/.hermes/kb-data）；该目录不存在则 skip（CI 沙箱走上面 4 条
    temp 语料不变式测试，真实语料回归是 owner 本地增强项）。
    """

    def test_regression_no_new_drift(self):
        import os
        data_dir = Path(os.environ.get("KB_REGRESSION_DATA_DIR",
                                       "/root/.hermes/kb-data"))
        if not (data_dir / "hot_corpus.pkl").exists():
            self.skipTest(f"no real KB corpus at {data_dir}")
        n = regression_no_new_drift(data_dir)
        # 2026-10-19: the corpus was owner-cleared after the taxonomy
        # restructure (737 -> 0 docs). A zero-doc corpus legitimately
        # runs the regression over 0 docs (assert no drift = no-op
        # PASS), so n == 0 is not a failure — only a missing/corrupt
        # pkl is (regression_no_new_drift raises on that path).
        self.assertGreaterEqual(n, 0)


class LlmDocKindAutoLabelTests(unittest.TestCase):
    """11-08 owner ruling: ingest 自动打标 — agent-capture path
    (close_task_boundary) LLM doc_kind auto-classify face.

    通过 `_doc_kind_verdicts` 测试钩子（mirror of `_topic_verdicts`）
    钉死行为，不碰真实 host LLM：
      1. pinned verdict "tutorial" -> staged doc carries doc_kind="tutorial"
      2. pinned verdict None      -> doc stays unlabeled (doc_kind key
         absent, the pre-change floor; B1-None-filter stable)
      3. the falsy-runtime guard: a cached empty-dict `_main_runtime`
         must NOT route through the distill no_runtime path (the
         11-08 transcript_drop regression this face exposed).
    """
    def _state(self, fake_svc, cand, doc_kind_verdicts=None):
        from test_kb_llm_topic import _make_state
        st = _make_state()
        st._service_override = fake_svc
        st._pending_override = [cand]
        if doc_kind_verdicts is not None:
            st._doc_kind_verdicts = doc_kind_verdicts
        # distill face pinned OFF so the no_runtime guard branch is the
        # ONLY variable under test (mirrors the _topic_verdicts pin in
        # test_kb_llm_topic's TaskBoundaryTopicIntegrationTests).
        st._distill_verdicts = None
        return st

    def _cand(self):
        from test_kb_phase4_tr9 import _incident_cand
        return _incident_cand(
            "uuid-dk-1", "9200 修复", "hermes-ui-repo 9200 attach 修复经验",
            "coder", "s-dk")

    def _fake_svc(self):
        from test_kb_phase4_tr9 import _FakeService
        return _FakeService(
            confirm_results=lambda *a, **k: {"status": "success"})

    def test_pinned_tutorial_lands_on_doc(self):
        cand = self._cand()
        fake = self._fake_svc()
        st = self._state(fake, cand, {"s-dk": "tutorial"})
        st.close_task_boundary("s-dk")
        self.assertEqual(len(fake.ingested), 1)
        self.assertEqual(fake.ingested[0].get("doc_kind"), "tutorial")

    def test_pinned_none_stays_unlabeled_floor(self):
        cand = self._cand()
        fake = self._fake_svc()
        st = self._state(fake, cand, {"s-dk": None})
        st.close_task_boundary("s-dk")
        self.assertEqual(len(fake.ingested), 1)
        # floor: the key stays ABSENT (doc_kind=None via _normalize_docs
        # setdefault, B1-None-filter stable — never implicitly "experience")
        self.assertNotIn("doc_kind", fake.ingested[0])

    def test_no_doc_kind_verdicts_key_is_a_noop_floor(self):
        # No _doc_kind_verdicts hook at all (production path, no host
        # runtime in this sandbox) -> the face returns None -> floor
        # stands; the doc is STILL ingested (no transcript_drop, the
        # falsy-runtime guard holds).
        cand = self._cand()
        fake = self._fake_svc()
        st = self._state(fake, cand)  # no _doc_kind_verdicts set
        st.close_task_boundary("s-dk")
        self.assertEqual(len(fake.ingested), 1,
                         "the falsy-runtime guard must keep the doc on the "
                         "fail-soft original-content path (no false "
                         "transcript_drop)")
        self.assertNotIn("doc_kind", fake.ingested[0])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression", action="store_true",
                        help="run the corpus docid regression (needs a real "
                             "KB data dir, not a temp one)")
    parser.add_argument("--data-dir", default="/root/.hermes/kb-data",
                        type=Path, help="KB data dir for --regression")
    args, remaining = parser.parse_known_args()
    if args.regression:
        n = regression_no_new_drift(args.data_dir)
        print(f"REGRESSION PASS: {n} confirmed docs, 0 doc_kind-induced docid drift")
    else:
        unittest.main(argv=[sys.argv[0]] + remaining)
