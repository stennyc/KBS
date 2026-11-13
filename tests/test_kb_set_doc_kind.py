"""knowledge_base_set_doc_kind service-method tests (TDD).

New narrow tool face: migrate / re-label a CONFIRMED doc's doc_kind
in place (same docid, every-other-key untouched) — the one supported
path for moving a legacy unlabeled doc into a new doc_kind door
without re-minting its docid (which would trip the B2 near-dup gate).

Design (mirrors set_premise_status's proven in-place pattern):
  - EX lock + rebase + _replace_confirmed_in_place + flush + audit.
  - Only the doc_kind key is ever touched. A non-None value goes into
    the doc dict; None clears the label (absent == None, B1 unified
    semantics — re-labeling to None leaves the key physically absent
    OR None-valued, both hash identically for a future re-ingest).
  - doc_kind value is validated against kb_ingest_rules.DOC_KIND_VOCAB
    (out-of-vocab -> rejected + audited, NOT a silent accept, same
    clarity semantics as topic_search's invalid_topic).
  - NOT a re-mint: the doc's docid key is never recomputed, so the B2
    containment near-dup gate (which runs only on ingest()'s content
    path) is never consulted — this is the whole point.

Run: python3 -m pytest test_kb_set_doc_kind.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb_service import KbSearchService  # noqa: E402
from kb_ingest_rules import DOC_KIND_VOCAB  # noqa: E402


def _svc_with_audit():
    base = Path(tempfile.mkdtemp())
    data = base / "kb-data"
    data.mkdir()
    audit = base / "audit.jsonl"
    svc = KbSearchService(data, audit_path=audit)
    return svc, base, data, audit


def _confirmed_doc(docid, title="未标注 存量 教程 文档", content=None, **extra):
    d = {"docid": docid, "title": title,
         "content": content or ("这是 一篇 存量 未标注 的 教程 文档 内容 若干字" * 2),
         "status": "confirmed"}
    d.update(extra)
    return d


class SetDocKindCoreTests(unittest.TestCase):

    def test_set_doc_kind_stamps_in_place_no_new_docid(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("legacy-1")])
        out = svc.set_doc_kind("legacy-1", "tutorial")
        self.assertEqual(out.get("status"), "success", out)
        corpus = svc._hot.docs_snapshot() + svc._cold.docs_snapshot()
        ids = [d.get("docid") for d in corpus]
        self.assertEqual(ids, ["legacy-1"],
                         "set_doc_kind must not mint a new docid")
        doc = next(d for d in corpus if d.get("docid") == "legacy-1")
        self.assertEqual(doc.get("doc_kind"), "tutorial")

    def test_set_doc_kind_from_unlabeled_none_to_tutorial(self):
        # The real migration case: c0f9cec3 has NO doc_kind key at all
        # (absent == None); setting it to "tutorial" must land the value.
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("legacy-2")])
        out = svc.set_doc_kind("legacy-2", "tutorial")
        self.assertEqual(out.get("status"), "success", out)
        doc = next(d for d in svc._hot.docs_snapshot()
                   + svc._cold.docs_snapshot()
                   if d.get("docid") == "legacy-2")
        self.assertEqual(doc.get("doc_kind"), "tutorial")

    def test_set_doc_kind_switch_between_two_kinds(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("legacy-3", doc_kind="reference")])
        out = svc.set_doc_kind("legacy-3", "tutorial")
        self.assertEqual(out.get("status"), "success", out)
        doc = next(d for d in svc._hot.docs_snapshot()
                   + svc._cold.docs_snapshot()
                   if d.get("docid") == "legacy-3")
        self.assertEqual(doc.get("doc_kind"), "tutorial")

    def test_set_doc_kind_clears_back_to_unlabeled(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("legacy-4", doc_kind="tutorial")])
        out = svc.set_doc_kind("legacy-4", None)
        self.assertEqual(out.get("status"), "success", out)
        doc = next(d for d in svc._hot.docs_snapshot()
                   + svc._cold.docs_snapshot()
                   if d.get("docid") == "legacy-4")
        # None clears the label: key absent OR None-valued, both == "no
        # label" under the B1 None-filter (search __unset__ sentinel).
        self.assertTrue(doc.get("doc_kind") is None,
                        "setting doc_kind=None must clear the label "
                        "(absent or None-valued, B1 unified semantics)")

    def test_every_vocab_value_is_settable(self):
        for kind in DOC_KIND_VOCAB:
            svc, *_ = _svc_with_audit()
            svc.ingest([_confirmed_doc(f"vk-{kind}")])
            out = svc.set_doc_kind(f"vk-{kind}", kind)
            self.assertEqual(out.get("status"), "success",
                             f"out-of-vocab check: {kind} must be "
                             f"settable: {out}")
            doc = next(d for d in svc._hot.docs_snapshot()
                       + svc._cold.docs_snapshot()
                       if d.get("docid") == f"vk-{kind}")
            self.assertEqual(doc.get("doc_kind"), kind)


class SetDocKindValidationTests(unittest.TestCase):

    def test_out_of_vocab_doc_kind_is_rejected_and_audited(self):
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([_confirmed_doc("v-1")])
        out = svc.set_doc_kind("v-1", "bogus-kind")
        self.assertEqual(out.get("status"), "error")
        self.assertEqual(out.get("error"), "invalid_doc_kind")
        self.assertIn("allowed", out)
        self.assertEqual(out["allowed"], list(DOC_KIND_VOCAB))
        # Doc must be untouched: a rejected value must not change the
        # doc's doc_kind (it stays None/absent — B1 unified semantics,
        # either spelling means "unlabeled").
        doc = next(d for d in svc._hot.docs_snapshot()
                   + svc._cold.docs_snapshot()
                   if d.get("docid") == "v-1")
        self.assertTrue(doc.get("doc_kind") is None,
                        "a rejected doc_kind value must leave the doc "
                        "unlabeled (doc_kind stays None)")
        # And it must be audited.
        lines = []
        if audit.exists():
            for line in audit.read_text().splitlines():
                if line.strip():
                    lines.append(json.loads(line))
        self.assertTrue(any(l.get("action") == "doc_kind_change_rejected"
                             and l.get("doc_id") == "v-1"
                             for l in lines),
                        "a rejected doc_kind value must be audited")

    def test_unknown_docid_is_rejected_and_audited(self):
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([_confirmed_doc("k-1")])
        out = svc.set_doc_kind("nope", "tutorial")
        self.assertEqual(out.get("status"), "error")
        self.assertEqual(out.get("error"), "not_found")
        lines = []
        if audit.exists():
            for line in audit.read_text().splitlines():
                if line.strip():
                    lines.append(json.loads(line))
        self.assertTrue(any(l.get("action") == "doc_kind_change_rejected"
                             and l.get("doc_id") == "nope"
                             for l in lines),
                        "a not-found doc_kind change must be audited")


class SetDocKindSearchEffectTests(unittest.TestCase):

    def test_set_doc_kind_makes_doc_searchable_under_the_new_filter(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("s-1")])
        # Before the label: the doc is NOT reachable under a tutorial
        # filter (unlabeled).
        miss = svc.search("存量 教程 文档",
                          filters={"doc_kind": "tutorial"})
        self.assertFalse(any(r["doc_id"] == "s-1"
                             for r in miss["data"]["results"]),
                         "an unlabeled doc must not match a tutorial filter")
        out = svc.set_doc_kind("s-1", "tutorial")
        self.assertEqual(out.get("status"), "success", out)
        hit = svc.search("存量 教程 文档",
                        filters={"doc_kind": "tutorial"})
        self.assertTrue(any(r["doc_id"] == "s-1"
                            for r in hit["data"]["results"]),
                        "after set_doc_kind('tutorial') the doc must "
                        "surface under the tutorial filter")

    def test_set_doc_kind_does_not_clobber_other_metadata_keys(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("s-2", topic="deployment",
                                   origin_kind="agent")])
        out = svc.set_doc_kind("s-2", "tutorial")
        self.assertEqual(out.get("status"), "success", out)
        doc = next(d for d in svc._hot.docs_snapshot()
                   + svc._cold.docs_snapshot()
                   if d.get("docid") == "s-2")
        self.assertEqual(doc.get("doc_kind"), "tutorial")
        self.assertEqual(doc.get("topic"), "deployment",
                         "set_doc_kind must not touch topic")
        self.assertEqual(doc.get("origin_kind"), "agent",
                         "set_doc_kind must not touch origin_kind")


class SetDocKindNoB2TriggerTests(unittest.TestCase):
    """The load-bearing invariant: an in-place doc_kind relabel of
    byte-identical content NEVER trips the B2 containment near-dup
    gate (that gate only runs on ingest()'s fresh-content path)."""

    def test_relabel_same_content_bypasses_dedup_gate(self):
        svc, *_ = _svc_with_audit()
        # Seed an unlabeled doc (content X).
        content = "CPU 本地 TTS 安装 配置 教程 步骤 一二三 若干字"
        svc.ingest([_confirmed_doc("b2-1", title="TTS 教程",
                                   content=content)])
        # Now a second doc with byte-identical content but pre-labeled
        # tutorial: this WOULD be a B2 near-dup (containment 1.0) if it
        # went through ingest(). Proving the gate is ingest-only by
        # showing set_doc_kind on the FIRST doc (same content, just a
        # relabel) succeeds with no rejection anywhere.
        out = svc.set_doc_kind("b2-1", "tutorial")
        self.assertEqual(out.get("status"), "success", out)
        corpus = svc._hot.docs_snapshot() + svc._cold.docs_snapshot()
        ids = [d.get("docid") for d in corpus]
        self.assertEqual(ids, ["b2-1"],
                         "relabeling the same-content doc in place must "
                         "not spawn a second row or invoke B2")


if __name__ == "__main__":
    unittest.main()
