"""11-08: manual knowledge_base_ingest LLM topic auto-classify
(owner 方案② write-side LLM face #2 — the manual face, on top of the
auto-capture face #1 in the task-boundary sink).

Covers:
  - a doc that carries NO explicit topic gets one auto-classified by the
    (stubbed) LLM face when a main_runtime is available;
  - a doc with an EXPLICIT topic is never overridden (the LLM face only
    fills the gap);
  - no host runtime (standalone) -> the floor "general" stands (the topic
    key stays absent and _normalize_docs stamps it later);
  - over-budget (a batch whose no-topic docs exceed the 3-call limit) ->
    only the first 3 are classified, the rest keep the floor.

Isolation: install_isolated_service() rebinds the real service + data
dir; a seeded custom_topics.jsonl makes the project tag the LLM picks a
valid vocab entry; hook._llm_topic_classify + hook._norm_main_runtime
are stubbed so NO real host call happens.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))

import kb_ingest_hook as hook  # noqa: E402
import __init__ as kb  # noqa: E402
from test_kb_isolation import install_isolated_service  # noqa: E402


def _seed_custom_topics(data_dir: Path, tag: str = "hermes-ui-repo") -> None:
    (data_dir / "custom_topics.jsonl").write_text(tag + "\n", encoding="utf-8")


def _stub_llm_pick(tag: str):
    """A _llm_topic_classify stand-in that always returns `tag`."""
    return lambda main_runtime, text, valid: tag


class ManualIngestLlmTopicTests(unittest.TestCase):
    """The manual knowledge_base_ingest face auto-classifies topic."""

    def setUp(self):
        self._data_dir = install_isolated_service()
        _seed_custom_topics(self._data_dir, "hermes-ui-repo")

    def _run_ingest(self, docs, *, llm_pick=None, main_runtime={"p": 1}):
        """Drive the real _ingest_handler with a stubbed LLM face.

        llm_pick=None -> the real _llm_topic_classify (returns None for
        a fake runtime -> floor stands). Otherwise the stub that returns
        the given tag. main_runtime=None -> no host runtime (floor).
        """
        if main_runtime is None:
            with mock.patch.object(hook, "_norm_main_runtime", return_value=None), \
                 (mock.patch.object(hook, "_llm_topic_classify",
                                    side_effect=AssertionError(
                                        "no runtime: LLM face must not run"))):
                return json.loads(kb._ingest_handler({"docs": docs}))
        if llm_pick is None:
            # Real classify (returns None for the fake runtime -> floor).
            with mock.patch.object(hook, "_norm_main_runtime",
                                   return_value=main_runtime):
                return json.loads(kb._ingest_handler({"docs": docs}))
        with mock.patch.object(hook, "_norm_main_runtime",
                               return_value=main_runtime), \
             mock.patch.object(hook, "_llm_topic_classify",
                               _stub_llm_pick(llm_pick)):
            return json.loads(kb._ingest_handler({"docs": docs}))

    def test_no_topic_doc_gets_llm_tag(self):
        docs = [{"title": "9200 attach 修复", "content": "x" * 80,
                 "status": "confirmed", "origin_kind": "agent",
                 "source_profile": "coder"}]
        out = self._run_ingest(docs, llm_pick="hermes-ui-repo")
        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["ingested"]), 1)
        # The ingested doc carried the LLM-picked tag (not the floor).
        # _ingest_handler writes doc["topic"] before svc.ingest, so the
        # staged/confirmed row has it.
        docid = out["ingested"][0]
        self.assertIn("hermes-ui-repo", out["payload_notes"][0])

    def test_explicit_topic_never_overridden(self):
        docs = [{"title": "9200 attach 修复", "content": "x" * 80,
                 "topic": "frontend",  # explicit — must survive
                 "status": "confirmed", "origin_kind": "agent",
                 "source_profile": "coder"}]
        out = self._run_ingest(docs, llm_pick="hermes-ui-repo")
        self.assertEqual(out["status"], "success")
        # No auto-classify note (the LLM face skipped the explicit topic).
        self.assertFalse(any("auto-classified" in n for n in
                              out.get("payload_notes", [])))

    def test_no_host_runtime_floor_general(self):
        docs = [{"title": "9200 attach 修复", "content": "x" * 80,
                 "status": "pending", "origin_kind": "agent",
                 "source_profile": "coder"}]
        out = self._run_ingest(docs, main_runtime=None)
        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["staged"]), 1)
        # Floor: no auto-classify note (the LLM face never ran).
        self.assertFalse(any("auto-classified" in n for n in
                              out.get("payload_notes", [])))

    def test_over_budget_only_first_three(self):
        # 5 docs without an explicit topic; the 3-call limit means docs
        # 1-3 get the tag, docs 4-5 keep the floor.
        docs = [{"title": f"t{i}", "content": "x" * 80, "status": "pending",
                 "origin_kind": "agent", "source_profile": "coder"}
                for i in range(5)]
        out = self._run_ingest(docs, llm_pick="hermes-ui-repo")
        self.assertEqual(out["status"], "success")
        notes = [n for n in out.get("payload_notes", [])
                 if "auto-classified" in n]
        self.assertEqual(len(notes), 3)


if __name__ == "__main__":
    unittest.main()
