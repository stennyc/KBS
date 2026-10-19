"""Regression tests: a plugin tool handler that receives its `args` as a
JSON-serialized STRING (the host tool_call wrapper stringifies arguments;
a JSON-string payload not re-parsed before the handler runs reaches the
handler as `str`) must be handled gracefully, not crash with
`AttributeError: 'str' object has no attribute 'get'`.

Defect fixed 2026-10-19: `_ingest_handler` (and every sibling handler in
__init__.py) crashed on stringified args. The fix wires
`handler_args._normalize_handler_args` into each handler's first line:
dict args pass through byte-identical; a JSON-object string is decoded;
any other shape fails CLOSED with a structured
`{"status": "error", "error": "invalid_args", ...}` payload.
"""

import json
import sys
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLUGIN_DIR))

from handler_args import _normalize_handler_args  # noqa: E402
import __init__ as kb  # noqa: E402
from test_kb_isolation import install_isolated_service  # noqa: E402


class NormalizeHandlerArgsTests(unittest.TestCase):
    """Unit tests for the pure normalization function."""

    def test_dict_args_pass_through_unchanged(self):
        args = {"docs": [{"title": "t", "content": "c" * 30}]}
        self.assertIs(_normalize_handler_args(args), args)

    def test_json_object_string_decoded_to_dict(self):
        raw = json.dumps({"docs": [{"title": "t", "content": "c" * 30}]})
        out = _normalize_handler_args(raw)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["docs"][0]["title"], "t")

    def test_json_string_literal_fails_closed(self):
        # A JSON string that parses to a STRING (not an object).
        out = _normalize_handler_args(json.dumps("hello"))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")

    def test_non_json_string_fails_closed(self):
        out = _normalize_handler_args("not json at all")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")

    def test_empty_string_fails_closed(self):
        out = _normalize_handler_args("")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")

    def test_none_fails_closed(self):
        out = _normalize_handler_args(None)
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")

    def test_json_number_fails_closed(self):
        out = _normalize_handler_args("42")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")


class HandlerStringArgsTests(unittest.TestCase):
    """Each handler must survive string/None/garbage args and return a
    structured JSON payload (never raise AttributeError)."""

    def setUp(self):
        # Isolate the plugin's cached service to a temp data dir so no test
        # ever writes into the production /root/.hermes/kb-data.
        install_isolated_service()

    def _assert_no_crash(self, handler, arg):
        result = handler(arg)
        self.assertIsInstance(result, str)
        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)
        return parsed

    def test_ingest_string_args_no_crash(self):
        raw = json.dumps({"docs": [{"title": "defect test",
                                     "content": "x" * 30}]})
        parsed = self._assert_no_crash(kb._ingest_handler, raw)
        # The string payload decodes to a real ingest — the doc lands.
        self.assertEqual(parsed["status"], "success")

    def test_ingest_none_args_graceful_error(self):
        parsed = self._assert_no_crash(kb._ingest_handler, None)
        self.assertEqual(parsed["status"], "error")
        self.assertEqual(parsed["error"], "invalid_args")

    def test_ingest_garbage_args_graceful_error(self):
        parsed = self._assert_no_crash(kb._ingest_handler, "not json")
        self.assertEqual(parsed["status"], "error")
        self.assertEqual(parsed["error"], "invalid_args")

    def test_confirm_garbage_args_graceful_error(self):
        parsed = self._assert_no_crash(kb._confirm_handler, "not json")
        self.assertEqual(parsed["error"], "invalid_args")

    def test_get_none_args_graceful_error(self):
        parsed = self._assert_no_crash(kb._get_handler, None)
        self.assertEqual(parsed["error"], "invalid_args")

    def test_get_empty_dict_args_still_requires_doc_id(self):
        # A dict {} passes through the normal path — the handler's own
        # validation (doc_id is required) still applies.
        parsed = self._assert_no_crash(kb._get_handler, {})
        self.assertEqual(parsed["status"], "error")
        self.assertEqual(parsed["error"], "doc_id is required")

    def test_topic_search_string_args_no_crash(self):
        raw = json.dumps({"query": "x", "topic": "general"})
        parsed = self._assert_no_crash(kb._topic_search_handler, raw)
        # A decodable JSON object is handled by the real handler path;
        # either a success payload or a controlled error — never a crash.
        self.assertIn(parsed["status"], ("success", "error"))

    def test_distill_garbage_args_graceful_error(self):
        parsed = self._assert_no_crash(kb._distill_handler, "not json")
        self.assertEqual(parsed["error"], "invalid_args")

    def test_share_to_global_garbage_args_graceful_error(self):
        parsed = self._assert_no_crash(kb._share_to_global_handler,
                                       "not json")
        self.assertEqual(parsed["error"], "invalid_args")

    def test_set_doc_kind_garbage_args_graceful_error(self):
        parsed = self._assert_no_crash(kb._set_doc_kind_handler,
                                       "not json")
        self.assertEqual(parsed["error"], "invalid_args")


if __name__ == "__main__":
    unittest.main()
