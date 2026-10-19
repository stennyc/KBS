"""Regression tests: reviewer + data-manager 复盘 fixes (2026-10-19).

Reviewer findings addressed:
  M-1  Nested-field int()/float() coercion crashes (top_k / min_score)
       -> guarded by _coerce_int_arg / _coerce_float_arg, fail closed
          with a structured invalid_args payload.
  m-1  _share_to_global_handler accepted an empty doc_id
       -> now returns "doc_id is required".
  m-2  _distill_handler accepted one-of-two (schema declares both
       required) -> now enforces both raw_title AND raw_content.
  m-3  _search_handler ran an empty query when query was missing
       -> now returns "query is required".
  m-4  _topic_search_handler ran an empty query when query was missing
       -> now returns "query is required".

Data-manager finding noted (no code change required, current state is
safe): a future manual clear must truncate BOTH staging_docs.jsonl AND
staging_index.pkl, otherwise StagingStore._load branch-2 resurrects the
cleared rows from the stale index snapshot.

All handlers are called directly with a DICT payload (the normal host
path); the normalize layer is exercised by test_handler_args.py.
"""

import json
import sys
from pathlib import Path

import unittest

_PLUGIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLUGIN_DIR))

import __init__ as kb  # noqa: E402


class NestedCoercionTests(unittest.TestCase):
    """M-1: malformed nested numeric fields must fail closed, not crash."""

    def test_search_top_k_string_fails_closed(self):
        out = json.loads(kb._search_handler({"query": "x", "top_k": "abc"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")
        self.assertIn("top_k", out["message"])

    def test_search_top_k_none_fails_closed(self):
        out = json.loads(kb._search_handler({"query": "x", "top_k": None}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")

    def test_search_min_score_string_fails_closed(self):
        out = json.loads(kb._search_handler(
            {"query": "x", "min_score": "high"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")
        self.assertIn("min_score", out["message"])

    def test_topic_search_top_k_string_fails_closed(self):
        out = json.loads(kb._topic_search_handler(
            {"query": "x", "topic": "general", "top_k": "abc"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "invalid_args")

    def test_search_valid_top_k_and_min_score_pass_through(self):
        out = json.loads(kb._search_handler(
            {"query": "react hooks", "top_k": 2, "min_score": 0.1}))
        self.assertEqual(out["status"], "success")

    def test_topic_search_valid_top_k_pass_through(self):
        out = json.loads(kb._topic_search_handler(
            {"query": "kafka", "topic": "frontend", "top_k": 1}))
        self.assertEqual(out["status"], "success")


class RequiredFieldGuardTests(unittest.TestCase):
    """m-1..m-4: schema-required fields must be enforced with a
    structured error, not a silent empty/zero-hit operation."""

    def test_search_missing_query_rejected(self):
        out = json.loads(kb._search_handler({"top_k": 3}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "query is required")

    def test_search_blank_query_rejected(self):
        out = json.loads(kb._search_handler({"query": "   "}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "query is required")

    def test_topic_search_missing_query_rejected(self):
        out = json.loads(kb._topic_search_handler({"topic": "frontend"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "query is required")

    def test_share_to_global_empty_doc_id_rejected(self):
        out = json.loads(kb._share_to_global_handler({"doc_id": ""}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "doc_id is required")

    def test_share_to_global_missing_doc_id_rejected(self):
        out = json.loads(kb._share_to_global_handler({}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "doc_id is required")

    def test_distill_title_only_rejected(self):
        # m-2: schema declares BOTH raw_title and raw_content required.
        out = json.loads(kb._distill_handler({"raw_title": "my title"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "raw_content is required")

    def test_distill_content_only_rejected(self):
        out = json.loads(kb._distill_handler({"raw_content": "some body"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"], "raw_title is required")

    def test_distill_both_present_reaches_distill_path(self):
        # Both fields present -> the handler proceeds to the distill
        # verdict (success with a verdict; no required-field error).
        out = json.loads(kb._distill_handler(
            {"raw_title": "T", "raw_content": "original content body"}))
        self.assertEqual(out["status"], "success")
        self.assertIn("verdict", out)


class IngestSchemaDescriptionTests(unittest.TestCase):
    """M-2: INGEST_SCHEMA description must match the code's actual
    length gate (60-10000 UTF-8 BYTES, not '20-10000 chars')."""

    def test_ingest_schema_documents_byte_unit_and_60_floor(self):
        desc = kb.INGEST_SCHEMA["description"]
        # The stale '20-10000 chars' wording must be gone...
        self.assertNotIn("content 20-10000 chars", desc)
        # ...and replaced with the byte-based wording.
        self.assertIn("60-10000 UTF-8 bytes", desc)
        self.assertIn("UTF-8 BYTES", desc)


if __name__ == "__main__":
    unittest.main()
