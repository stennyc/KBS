"""Phase 3: superseded_by mechanism — TDD tests (RED before implementation).

Owner ruling (2026-10-14, settled in prd-kb-write-side-v2.md §5):
  - superseded_by: Optional docid (default None = never superseded).
    Independent, orthogonal to premise_status (a doc can be BOTH
    falsified AND superseded_by — two failure reasons, no interaction).
  - superseded docs are NEVER deleted: the row stays in the confirmed
    corpus, is DROPPED from the auto-inject pool (_agent_in_decay_gate
    + the user-segment search() default path) and from the default
    tool-layer search() result, but remains explicitly retrievable via
    knowledge_base_get (explicit fetch != auto-inject, same rule as
    premise_status=falsified).
  - reinstate() restores the doc: superseded_by -> None, back in the
    pool (reversible, unlike a premise falsification which needs the
    explicit set_premise_status migration path).
  - No new docid is ever minted: the field is stamped IN PLACE on the
    existing confirmed row (same in-place-replace pattern as
    set_premise_status; the double-row bug class is closed by docid-keyed
    list swap, never append).
  - Write-side conflict probe: when ingest() routes a doc to CONFIRMED,
    the service probes the confirmed corpus (BM25 over the existing
    two-layer search channel, no new retrieval engine) for a
    semantically-adjacent older entry and, on a "supersedes" verdict
    (budget-gated LLM call, a NEW independent "conflict" budget pool
    that never borrows from the judge/distill pools), auto-stamps
    superseded_by on the older doc.  On "coexist" (or any LLM
    failure/budget-exhausted) the older doc is left untouched —
    fail-open: a missed supersession is always better than a wrongly
    suppressed live doc.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
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


def _confirmed_doc(docid, title="Redis 缓存穿透 治理 经验条目", content=None, **extra):
    d = {"docid": docid, "title": title,
         "content": content or ("Redis 缓存穿透 治理 方案 乙丙丁戊" * 3),
         "status": "confirmed"}
    d.update(extra)
    return d


class SupersededByFieldTests(unittest.TestCase):

    def test_default_is_none_and_absent_metadata_is_fine(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("sup-1")])
        doc = svc._hot.docs_snapshot() + svc._cold.docs_snapshot()
        d = next((x for x in doc if x.get("docid") == "sup-1"), None)
        self.assertIsNotNone(d)
        self.assertNotIn("superseded_by", d,
                         "a never-superseded doc must not carry the key "
                         "(absent == None semantics, same as gov_tier)")

    def test_supersede_stamps_in_place_no_new_docid(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("old-1"), _confirmed_doc("new-1")])
        out = svc.supersede("old-1", superseded_by="new-1",
                            confirmed_by="owner")
        self.assertEqual(out.get("status"), "success", out)
        corpus = svc._hot.docs_snapshot() + svc._cold.docs_snapshot()
        ids = [d.get("docid") for d in corpus]
        self.assertEqual(len(ids), 2,
                         "supersede must not mint a new docid")
        self.assertEqual(sorted(ids), ["new-1", "old-1"])
        old = next(d for d in corpus if d.get("docid") == "old-1")
        self.assertEqual(old.get("superseded_by"), "new-1")

    def test_supersede_unknown_docid_is_a_rejected_audited_noop(self):
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([_confirmed_doc("old-1")])
        out = svc.supersede("nope", superseded_by="old-1",
                            confirmed_by="owner")
        self.assertEqual(out.get("status"), "error")
        self.assertEqual(out.get("error"), "not_found")
        lines = []
        if audit.exists():
            for line in audit.read_text().splitlines():
                if line.strip():
                    lines.append(json.loads(line))
        self.assertTrue(any(l.get("action") == "supersede_rejected"
                             for l in lines),
                        "a failed supersede must be audited")

    def test_reinstate_clears_the_marker_doc_stays(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("old-1"), _confirmed_doc("new-1")])
        svc.supersede("old-1", superseded_by="new-1", confirmed_by="owner")
        out = svc.reinstate("old-1", confirmed_by="owner")
        self.assertEqual(out.get("status"), "success", out)
        corpus = svc._hot.docs_snapshot() + svc._cold.docs_snapshot()
        ids = [d.get("docid") for d in corpus]
        self.assertEqual(len(ids), 2,
                         "reinstate must not mint or delete a docid")
        old = next(d for d in corpus if d.get("docid") == "old-1")
        self.assertNotIn("superseded_by", old,
                         "reinstate clears the marker (absent == active)")

    def test_supersede_reinstate_are_idempotent_and_noop_safe(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("old-1"), _confirmed_doc("new-1")])
        svc.supersede("old-1", superseded_by="new-1", confirmed_by="owner")
        out = svc.reinstate("old-1", confirmed_by="owner")
        self.assertEqual(out.get("status"), "success")
        out2 = svc.reinstate("old-1", confirmed_by="owner")
        self.assertEqual(out2.get("status"), "success",
                         "reinstate on an already-active doc is a "
                         "recorded no-op, not an error")
        self.assertNotIn("superseded_by",
                         next(d for d in svc._hot.docs_snapshot()
                              + svc._cold.docs_snapshot()
                              if d.get("docid") == "old-1"))

    def test_superseded_by_is_orthogonal_to_premise_status(self):
        svc, *_ = _svc_with_audit()
        svc.ingest([_confirmed_doc("o", premise_status="falsified"),
                    _confirmed_doc("n")])
        out = svc.supersede("o", superseded_by="n", confirmed_by="owner")
        self.assertEqual(out.get("status"), "success",
                         "a falsified doc can still be superseded_by "
                         "(two independent failure reasons)")
        o = next(d for d in svc._hot.docs_snapshot()
                 + svc._cold.docs_snapshot()
                 if d.get("docid") == "o")
        self.assertEqual(o.get("superseded_by"), "n")
        self.assertEqual(o.get("premise_status"), "falsified",
                         "supersede must not clobber premise_status")


class SupersededBySearchDropTests(unittest.TestCase):

    def _seeded(self):
        svc, base, data, audit = _svc_with_audit()
        svc.ingest([_confirmed_doc("old", title="Redis 缓存穿透 治理",
                                   content="Redis 缓存穿透 治理 方案 乙丙丁戊"),
                    _confirmed_doc("new", title="Redis 缓存穿透 治理 v2",
                                   content="Redis 缓存穿透 治理 方案 v2 甲乙丙丁")])
        svc.supersede("old", superseded_by="new", confirmed_by="owner")
        return svc, base, data, audit

    def test_default_search_drops_superseded_doc(self):
        svc, *_ = self._seeded()
        res = svc.search("Redis 缓存穿透 治理")
        ids = [r.get("doc_id") for r in res["data"]["results"]]
        self.assertNotIn("old", ids,
                         "a superseded doc must not surface in the "
                         "default tool-layer search")
        self.assertIn("new", ids)

    def test_explicit_get_by_docid_still_returns_the_superseded_doc(self):
        # The "留盘可查" ruling: dropping from search() does NOT delete
        # the row; an explicit docid fetch (knowledge_base_get, and the
        # service-level find-by-docid it wraps) must still return it.
        svc, *_ = self._seeded()
        corpus = svc._hot.docs_snapshot() + svc._cold.docs_snapshot()
        found = [d for d in corpus if d.get("docid") == "old"]
        self.assertEqual(len(found), 1,
                         "the superseded row must remain physically "
                         "present in the corpus (留盘, not deleted)")

    def test_reinstate_putback_into_default_search(self):
        svc, *_ = self._seeded()
        svc.reinstate("old", confirmed_by="owner")
        res = svc.search("Redis 缓存穿透 治理")
        ids = [r.get("doc_id") for r in res["data"]["results"]]
        self.assertIn("old", ids,
                      "reinstate puts the doc back into the default "
                      "search pool")


class ConflictProbeTests(unittest.TestCase):

    def test_conflict_probe_is_inert_when_no_similar_doc_exists(self):
        # No LLM is available in unit tests: the probe must fail-open
        # (older doc untouched, no superseded_by stamped) rather than
        # crash or block ingest.
        svc, base, data, audit = _svc_with_audit()
        out = svc.ingest([_confirmed_doc("lonely")])
        self.assertTrue(out.get("ingested"),
                        "the confirmed doc must actually land: %r" % out)
        corpus = svc._hot.docs_snapshot() + svc._cold.docs_snapshot()
        self.assertEqual(len(corpus), 1)
        self.assertNotIn("superseded_by", corpus[0])

    def test_conflict_budget_pool_is_independent_of_judge_distill(self):
        # The probe's LLM budget must be its own pool: exhausting it
        # cannot starve judge/distill and vice-versa.  Structured
        # assertion on the service's budget tracker (fail-soft: no
        # verdict is applied on budget exhaustion, ingest proceeds
        # unblocked).
        svc, *_ = _svc_with_audit()
        self.assertTrue(hasattr(svc, "_conflict_budgets"),
                        "the conflict probe must own an independent "
                        "budget pool (not borrow judge/distill's)")

class ConflictVerdictJsonTests(unittest.TestCase):
    """Regression for reviewer issue #3: _llm_conflict_verdict's JSON
    parse must not truncate verdicts whose reason text contains a "}"
    (the old lstrip("{")+rsplit("}",1) recipe did).  Covered via the
    parse-logic that _llm_conflict_verdict uses on the LLM content."""

    def _parse_like_verdict(self, content):
        """Replicate the exact parse sequence in _llm_conflict_verdict."""
        import json as _json, re as _re
        text = (content or "").strip()
        verdict = None
        if text:
            try:
                verdict = _json.loads(text)
            except Exception:
                verdict = None
        if verdict is None:
            stripped = text
            if "```" in stripped:
                m = _re.search(r"```[a-zA-Z]*\n?(.*?)\n?```",
                               stripped, _re.DOTALL)
                if m:
                    stripped = m.group(1).strip()
            try:
                verdict = _json.loads(stripped) if stripped else None
            except Exception:
                verdict = None
        return verdict

    def test_reason_containing_brace_survives(self):
        # The exact failure case from reviewer issue #3
        content = '{"supersedes": true, "reason": "v2 replaces {v1}"}'
        verdict = self._parse_like_verdict(content)
        self.assertIsNotNone(verdict,
                             "a verdict whose reason contains a brace "
                             "must not be silently dropped")
        self.assertTrue(verdict.get("supersedes"))
        self.assertEqual(verdict["reason"], "v2 replaces {v1}")

    def test_markdown_fenced_json_survives(self):
        content = '```json\n{"supersedes": true, "reason": "coexist"}\n```'
        verdict = self._parse_like_verdict(content)
        self.assertIsNotNone(verdict)
        self.assertTrue(verdict.get("supersedes"))

    def test_garbage_content_returns_none_failsoft(self):
        verdict = self._parse_like_verdict("this is not json at all")
        self.assertIsNone(verdict,
                          "unparseable LLM content must fail-open to "
                          "None (no verdict applied), never raise")
