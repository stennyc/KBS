"""TR-x: physical ``drop`` — TDD tests (RED before implementation).

``drop`` is the ONLY deletion path in the kb service face. Every other
lifecycle op (supersede / reinstate / set_*) mutates a row in place and
keeps the docid on disk (留盘可查); ``drop`` physically removes the docid
from BOTH the hot and cold corpora, durably, so a later cross-process
rebase (union-merge) cannot resurrect it.

Tests pin:
  1. drop removes the docid from hot + cold (both layer snapshots empty of
     it) AND the on-disk hot_corpus.pkl no longer carries it (durability —
     a fresh service instance re-reading the store must not see it back).
  2. drop is idempotent: dropping an already-absent docid is a recorded
     success (already_absent), not an error.
  3. drop does NOT touch sibling docs (only the targeted docid is gone).
  4. drop is the inverse of "留盘可查": after a drop, knowledge_base_get
     (explicit fetch) returns not_found — the row is physically gone, not
     just dropped-from-pool like a superseded row.
"""
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from kb_service import KbSearchService  # noqa: E402


def _svc_with_audit():
    base = Path(tempfile.mkdtemp())
    data = base / "kb-data"
    data.mkdir()
    audit = base / "audit.jsonl"
    svc = KbSearchService(data, audit_path=audit)
    return svc, base, data, audit


def _confirmed_doc(docid, title="缓存 穿透 治理 经验条目 甲乙丙", content=None,
                   **extra):
    d = {"docid": docid, "title": title,
         "content": content or ("缓存 穿透 治理 方案 乙丙丁戊 填充 段落 若干字 "
                                "扩充 内容 标记 词" * 3),
         "status": "confirmed"}
    d.update(extra)
    return d


def _all_docs(svc):
    return [d for d in (svc._hot.docs_snapshot() + svc._cold.docs_snapshot())]


class DropTests(unittest.TestCase):

    def test_drop_removes_from_both_layers_and_disk(self):
        svc, base, data, _ = _svc_with_audit()
        svc.ingest([_confirmed_doc("d-1", title="条目 一 缓存 穿透"),
                    _confirmed_doc("d-2", title="条目 二 缓存 治理")])
        out = svc.drop("d-1", confirmed_by="owner",
                       reason="duplicate copy, body kept")
        self.assertEqual(out.get("status"), "success", out)

        ids = {d.get("docid") for d in _all_docs(svc)}
        self.assertNotIn("d-1", ids, "d-1 must be gone from hot+cold memory")
        self.assertIn("d-2", ids, "sibling must be untouched")

        # Durability: a FRESH instance re-reading the store must not see it.
        svc2 = KbSearchService(data)
        ids2 = {d.get("docid") for d in _all_docs(svc2)}
        self.assertNotIn("d-1", ids2,
                         "a cross-process rebase must NOT resurrect d-1")
        self.assertIn("d-2", ids2)

    def test_drop_idempotent_when_absent(self):
        svc, *_ = _svc_with_audit()
        out = svc.drop("never-confirmed")
        self.assertEqual(out.get("status"), "success", out)
        self.assertTrue(out.get("already_absent"), out)
        # second drop of the same absent id is also a clean no-op success
        out2 = svc.drop("never-confirmed")
        self.assertEqual(out2.get("status"), "success", out2)
        self.assertTrue(out2.get("already_absent"), out2)

    def test_drop_only_removes_target_docid(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("a"), _confirmed_doc("b"),
                    _confirmed_doc("c")])
        svc.drop("b")
        ids = {d.get("docid") for d in _all_docs(svc)}
        self.assertEqual(ids, {"a", "c"},
                         "only the targeted docid is removed")

    def test_drop_makes_explicit_get_not_found(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("d-1")])
        self.assertTrue(svc.get("d-1").get("found"),
                         "pre-drop the doc is retrievable")
        svc.drop("d-1")
        # Explicit get after a physical drop must NOT find the row anywhere
        # (confirmed corpus AND staging) — unlike a superseded row, which
        # stays 留盘可查 via get().
        res = svc.get("d-1")
        self.assertFalse(res.get("found"), res)


if __name__ == "__main__":
    unittest.main()
