"""11-08: LLM project-topic classification on the auto-capture path
(owner 方案② — the write-side LLM face tags each agent-experience doc's
topic from the valid vocab; fail-soft to the floor "general").

Covers:
  - _llm_topic_classify: a healthy reply picking a CUSTOM project tag
    (from custom_topics.jsonl) is returned; a reply NOT in the valid
    vocab returns None; no main runtime / no host surface -> None.
  - _WriteSideState._maybe_llm_topic: budget-gated (independent pool),
    over-budget -> None (floor stands, no LLM call), a valid LLM tag
    overrides the doc's floor "general", None keeps the floor.
  - close_task_boundary integration: a healthy LLM topic lands on the
    staged/confirmed doc (doc["topic"] != "general" when the LLM picks
    a project tag).

Isolation: install_isolated_service() rebinds the real service; the
write-side state object is driven directly with a fake main_runtime +
a monkeypatched _HOST_CALL_LLM stub so NO real host call happens.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLUGIN_DIR))

import kb_ingest_hook as hook  # noqa: E402
from kb_ingest_rules import Candidate, merge_rules, DEFAULT_RULES  # noqa: E402
from test_kb_isolation import install_isolated_service  # noqa: E402


def _fake_ctx(rules=None):
    class _Ctx:
        def __init__(self):
            self._loaded_rules = None

    ctx = _Ctx()
    return ctx


def _make_state(rules=None):
    state = hook._WriteSideState(_fake_ctx())
    if rules is not None:
        state._loaded_rules = rules
    return state


def _incident_cand(doc_id, title, content, session, profile="coder"):
    return Candidate(
        doc_id=doc_id,
        classified_type="incident",
        label="边角经验事实",
        title=title,
        content=content,
        source_session_id=session,
        source_profile=profile,
        cue_hits=["顺带"],
        origin_kind="agent",
    )


# ---------------------------------------------------------------------------
# _llm_topic_classify (the pure host-LLM call, no state)
# ---------------------------------------------------------------------------

class TopicClassifyPureTests(unittest.TestCase):
    """The _llm_topic_classify host-call: healthy custom tag returned,
    out-of-vocab / no-runtime -> None (floor)."""

    def test_custom_project_tag_returned(self):
        rt = {"provider": "fake", "model": "fake"}

        class _Msg:
            content = "hermes-ui-repo"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        with mock.patch.object(hook, "_HOST_CALL_LLM",
                               return_value=_Resp()) as stub:
            got = hook._llm_topic_classify(
                rt, "9200 dashboard attach 修复经验",
                ["general", "hermes-ui-repo", "filemem"])
        self.assertEqual(got, "hermes-ui-repo")
        self.assertTrue(stub.called)

    def test_out_of_vocab_reply_returns_none(self):
        rt = {"provider": "fake", "model": "fake"}

        class _Msg:
            content = "some-invented-tag"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        with mock.patch.object(hook, "_HOST_CALL_LLM",
                               return_value=_Resp()):
            got = hook._llm_topic_classify(
                rt, "text", ["general", "filemem"])
        self.assertIsNone(got)  # not in the valid list -> floor

    def test_no_main_runtime_returns_none(self):
        with mock.patch.object(hook, "_HOST_CALL_LLM",
                               side_effect=AssertionError("must not be "
                                                          "called")):
            got = hook._llm_topic_classify(None, "text", ["general"])
        self.assertIsNone(got)

    def test_empty_valid_list_returns_none(self):
        rt = {"provider": "fake"}
        got = hook._llm_topic_classify(rt, "text", [])
        self.assertIsNone(got)


# ---------------------------------------------------------------------------
# _maybe_llm_topic (budget-gated state method)
# ---------------------------------------------------------------------------

class MaybeLlmTopicBudgetTests(unittest.TestCase):
    """The per-session independent budget pool: over-cap -> no call,
    a healthy tag -> counted, the floor stands on None."""

    def setUp(self):
        self._data_dir = install_isolated_service()
        # The valid-topic list for the LLM face is TOPIC_VOCAB +
        # custom_topics.jsonl under the (isolated) KB_DATA_DIR. Seed the
        # project tag the tests assert on so the LLM reply "filemem"
        # is a valid vocab entry here too (mirrors the production
        # kb-data/custom_topics.jsonl).
        (Path(self._data_dir) / "custom_topics.jsonl").write_text(
            "hermes-ui-repo\nhermes-core\nkb-plugin\nfilemem\n",
            encoding="utf-8")

    def test_healthy_tag_counts_and_returns(self):
        st = _make_state()
        # Test hook: pin the LLM verdict (mirror of the _distill_verdicts
        # precedent) so no real host call is needed.
        st._topic_verdicts = {"s1": "filemem"}
        got = st._maybe_llm_topic("s1",
                                  {"title": "t", "content": "filemem 踩坑",
                                   "topic": "general"})
        self.assertEqual(got, "filemem")

    def test_healthy_tag_via_real_llm_path(self):
        """Exercise the real _llm_topic_classify call path with a stubbed
        host surface (no pre-resolved verdict) + a fake main_runtime that
        short-circuits the normalizer."""
        st = _make_state()
        st._main_runtime = {"provider": "stub"}
        # Patch _norm_main_runtime to return the stub dict directly so
        # _main_runtime_live() yields it without touching the real
        # auxiliary_client.
        with mock.patch.object(hook, "_norm_main_runtime",
                               return_value={"provider": "stub"}):
            def _good_call(main_runtime, messages, **kw):
                class _M:
                    content = "filemem"

                class _C:
                    message = _M()

                class _R:
                    choices = [_C()]

                return _R()

            with mock.patch.object(hook, "_HOST_CALL_LLM", _good_call):
                got = st._maybe_llm_topic("s2",
                                          {"title": "t",
                                           "content": "filemem 踩坑",
                                           "topic": "general"})
        self.assertEqual(got, "filemem")
        self.assertEqual(st._topic_calls_by_session.get("s2"), 1)

    def test_over_budget_no_call_floor(self):
        st = _make_state(merge_rules({"llm": {"topic": {"per_session": 1}}}))
        st._main_runtime = {"provider": "fake"}
        st._topic_calls_by_session["s1"] = 1  # already at the cap
        with mock.patch.object(hook, "_HOST_CALL_LLM",
                               side_effect=AssertionError("over-cap: no "
                                                          "LLM call")):
            got = st._maybe_llm_topic("s1",
                                      {"title": "t", "content": "x",
                                       "topic": "general"})
        self.assertIsNone(got)

    def test_no_runtime_floor(self):
        st = _make_state()
        st._main_runtime = None
        # _norm_main_runtime may yield None in the test env -> floor.
        # Force the no-runtime path deterministically (patch both the
        # cached snapshot and the normalizer) so this test is
        # environment-independent.
        with mock.patch.object(hook, "_norm_main_runtime", None):
            got = st._maybe_llm_topic("s1",
                                      {"title": "t", "content": "x",
                                       "topic": "general"})
        self.assertIsNone(got)

    def test_disabled_face_floor(self):
        st = _make_state(merge_rules({"llm": {"topic": {"enable": False}}}))
        st._main_runtime = {"provider": "fake"}
        with mock.patch.object(hook, "_HOST_CALL_LLM",
                               side_effect=AssertionError("disabled: no "
                                                          "LLM call")):
            got = st._maybe_llm_topic("s1",
                                      {"title": "t", "content": "x",
                                       "topic": "general"})
        self.assertIsNone(got)


# ---------------------------------------------------------------------------
# close_task_boundary integration: a healthy LLM tag lands on the doc
# ---------------------------------------------------------------------------

class TaskBoundaryTopicIntegrationTests(unittest.TestCase):
    """End-to-end: the LLM topic face (stubbed host) overrides the doc's
    floor "general" on the auto-captured agent doc at the task boundary."""

    def setUp(self):
        self._data_dir = install_isolated_service()
        (Path(self._data_dir) / "custom_topics.jsonl").write_text(
            "hermes-ui-repo\nhermes-agent-mobile\nhermes-core\nkb-plugin\n"
            "filemem\n",
            encoding="utf-8")

    def test_llm_topic_lands_on_doc(self):
        from test_kb_phase4_tr9 import _FakeService

        cand = _incident_cand("uuid-topic-1", "9200 修复",
                              "hermes-ui-repo 9200 attach 修复经验",
                              session="s-topic")
        fake_svc = _FakeService(confirm_results=lambda *a, **k: {"status": "success"})
        st = _make_state()
        st._service_override = fake_svc
        st._pending_override = [cand]
        # Test hook: pin the LLM topic verdict (no real host call).
        st._topic_verdicts = {"s-topic": "hermes-ui-repo"}
        st.close_task_boundary("s-topic")

        # Exactly one agent doc staged + confirmed by the fake service.
        self.assertEqual(len(fake_svc.ingested), 1)
        doc = fake_svc.ingested[0]
        self.assertEqual(doc.get("topic"), "hermes-ui-repo")
        # And it was confirmed under that doc (agent self-confirm).
        self.assertEqual(len(fake_svc.confirm_calls), 1)

    def test_floor_general_when_no_topic_verdict(self):
        from test_kb_phase4_tr9 import _FakeService

        cand = _incident_cand("uuid-topic-3", "移动端布局",
                              "hermes-agent-mobile 侧边栏布局修复",
                              session="s-topic3")
        fake_svc = _FakeService(confirm_results=lambda *a, **k: {"status": "success"})
        st = _make_state()
        st._service_override = fake_svc
        st._pending_override = [cand]
        # No _topic_verdicts hook + no main runtime -> the LLM face
        # degrades to the floor; the doc keeps "general".
        with mock.patch.object(hook, "_norm_main_runtime", None):
            st.close_task_boundary("s-topic3")

        self.assertEqual(len(fake_svc.ingested), 1)
        doc = fake_svc.ingested[0]
        self.assertEqual(doc.get("topic"), "general")


if __name__ == "__main__":
    unittest.main()
