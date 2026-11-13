"""A (escape peel) + B (doc_file) behavior tests for _ingest_handler.

A: model-output guard — a doc whose title/content arrives as a JSON-encoded
string (the model accidentally serialized the payload one layer too many)
is peeled by EXACTLY one layer, with a per-doc note in the handler output.
Non-string / non-JSON / invalid-JSON values pass through untouched.

B: doc_file — a doc may point 'content' at a file on disk instead of
inlining it, to kill multi-layer escape payloads at the root. The path is
root-constrained to the KB data dir (relative = under the data dir;
absolute = must resolve under it). The doc_file key is DELETED from the
doc dict before ingest so the content-addressed docid (uuid5 over the
whole doc dict) stays identical to the inline-content equivalent.

Runs against the REAL _ingest_handler (not a mock), same contract as
test_handler_args.py / test_handler_param_guards.py. Isolation:
install_isolated_service() rebinds kb._service + kb._DATA_DIR so no test
touches the production /root/.hermes/kb-data.
"""

import json
import sys
import unittest
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGIN_DIR))

import __init__ as kb  # noqa: E402
from test_kb_isolation import install_isolated_service  # noqa: E402


class EscapePeelTests(unittest.TestCase):
    """A: _heal_escaped_content — pure function, no service involved."""

    def test_single_serialized_content_is_peeled(self):
        real = "hello 你好\nline2"
        doc = {"title": "t", "content": json.dumps(real)}
        healed, notes = kb._heal_escaped_content([doc])
        self.assertEqual(healed[0]["content"], real)
        self.assertIsNotNone(notes[0])

    def test_double_serialized_content_peels_exactly_one_layer(self):
        real = "hello 你好"
        doc = {"title": "t", "content": json.dumps(json.dumps(real))}
        healed, notes = kb._heal_escaped_content([doc])
        # Only ONE layer peeled — the remaining accidental layer stays
        # (no recursion, by design).
        self.assertEqual(healed[0]["content"], json.dumps(real))
        self.assertIsNotNone(notes[0])

    def test_title_also_peeled(self):
        real = "一个标题"
        doc = {"title": json.dumps(real), "content": "plain"}
        healed, notes = kb._heal_escaped_content([doc])
        self.assertEqual(healed[0]["title"], real)
        self.assertIn("title", notes[0])

    def test_plain_content_untouched(self):
        doc = {"title": "t", "content": "hello 你好"}
        healed, notes = kb._heal_escaped_content([doc])
        self.assertEqual(healed[0]["content"], "hello 你好")
        self.assertIsNone(notes[0])

    def test_jsonish_but_invalid_string_untouched(self):
        doc = {"title": "t", "content": '"unbalanced'}
        healed, notes = kb._heal_escaped_content([doc])
        self.assertEqual(healed[0]["content"], '"unbalanced')
        self.assertIsNone(notes[0])

    def test_non_string_fields_untouched(self):
        doc = {"title": "t", "content": 42, "source_profile": 7}
        healed, notes = kb._heal_escaped_content([doc])
        self.assertEqual(healed[0]["content"], 42)
        self.assertEqual(healed[0]["source_profile"], 7)
        self.assertIsNone(notes[0])

    def test_non_dict_doc_untouched(self):
        healed, notes = kb._heal_escaped_content(["garbage"])
        self.assertEqual(healed, ["garbage"])
        self.assertIsNone(notes[0])

    def test_empty_string_untouched(self):
        doc = {"title": "t", "content": ""}
        healed, notes = kb._heal_escaped_content([doc])
        self.assertEqual(healed[0]["content"], "")
        self.assertIsNone(notes[0])

    def test_legit_json_doc_value_kept(self):
        """A content that legitimately starts with a quote but is not a
        JSON-encoded string literal (raw markdown fence, JSON blob the
        user MEANT as content) must not be mis-peeled."""
        # A raw multi-line JSON doc as content: json.loads returns a dict,
        # not a str -> not a JSON-encoded string literal -> untouched.
        raw = '{"title": "x"}'
        doc = {"title": "t", "content": raw}
        healed, notes = kb._heal_escaped_content([doc])
        self.assertEqual(healed[0]["content"], raw)
        self.assertIsNone(notes[0])


class DocFileTests(unittest.TestCase):
    """B: _read_doc_file — pure function, no service involved."""

    def setUp(self):
        self._data_dir = install_isolated_service()

    def test_read_file_under_data_dir(self):
        p = self._data_dir / "doc.txt"
        p.write_text("disk content 你好", encoding="utf-8")
        text, err = kb._read_doc_file(str(p), self._data_dir)
        self.assertIsNone(err)
        self.assertEqual(text, "disk content 你好")

    def test_relative_path_resolves_under_data_dir(self):
        p = self._data_dir / "doc.txt"
        p.write_text("rel ok", encoding="utf-8")
        text, err = kb._read_doc_file("doc.txt", self._data_dir)
        self.assertIsNone(err)
        self.assertEqual(text, "rel ok")

    def test_escape_outside_root_refused(self):
        text, err = kb._read_doc_file("../../etc/hostname", self._data_dir)
        self.assertIsNone(text)
        self.assertIn("outside", err)

    def test_absolute_outside_refused(self):
        text, err = kb._read_doc_file("/etc/hostname", self._data_dir)
        self.assertIsNone(text)
        self.assertIsNotNone(err)

    def test_missing_file_refused(self):
        text, err = kb._read_doc_file("nope.txt", self._data_dir)
        self.assertIsNone(text)
        self.assertIsNotNone(err)

    # --- P0: URL scheme door (11-08 fix) ---
    # A doc_file value starting with http:// or https:// must be treated as
    # a URL to fetch, NOT a local path joined under data_dir (the 11-08 bug:
    # 'https://kb-font-offline.md' was resolved to a path under the KB data
    # dir and refused).
    def test_url_is_not_treated_as_local_path(self):
        """https://... must never be joined under data_dir and refused as
        'resolves outside the KB data dir' — that was the 11-08 bug."""
        text, err = kb._read_doc_file(
            "https://example.com/kb-font-offline.md", self._data_dir)
        # Network call: example.com may be unreachable in the test env, so
        # accept either a fetch failure OR success. The key assertion is that
        # the error (if any) is a fetch error, NOT a path-resolution error.
        if err is not None:
            self.assertIn("fetch failed", err,
                          msg=f"expected a URL fetch error, got: {err}")
        # If fetch succeeded, text must be non-None.
        if text is not None:
            self.assertIsInstance(text, str)

    def test_http_url_reaches_url_door_not_path_door(self):
        """http:// (not https) is also routed to the URL door, not the path
        door. A non-UTF-8 or unreachable URL yields a fetch error; the
        critical regression check is that we do NOT get a 'resolves outside
        the KB data dir' path error (that was the pre-fix behaviour)."""
        text, err = kb._read_doc_file(
            "http://127.0.0.1:1/never-listening.md", self._data_dir)
        if err is not None:
            self.assertNotIn("resolves outside", err,
                             msg="URL was mis-routed to the local-path door")
            self.assertIn("fetch failed", err)

    def test_malformed_url_is_still_routed_to_url_door(self):
        """A syntactically odd URL string that begins with https:// but is
        not fetchable must still be caught by the URL door (fetch error),
        not silently turned into a path."""
        text, err = kb._read_doc_file(
            "https://", self._data_dir)
        if err is not None:
            self.assertIn("fetch failed", err)


class IngestHandlerDocFileTests(unittest.TestCase):
    """B: _ingest_handler end-to-end with doc_file (isolated service)."""

    def setUp(self):
        self._data_dir = install_isolated_service()

    def test_doc_file_content_is_ingested(self):
        p = self._data_dir / "doc.txt"
        p.write_text("long content 你好 " * 40, encoding="utf-8")
        args = {"docs": [{"title": "t", "doc_file": "doc.txt",
                           "status": "pending"}]}
        out = json.loads(kb._ingest_handler(args))
        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["staged"]), 1)
        self.assertTrue(any("doc_file" in n for n in out.get("payload_notes", [])))

    def test_doc_file_does_not_pollute_docid(self):
        """Same content via inline vs via doc_file must map to the SAME
        deterministic docid (the doc_file key is deleted before ingest)."""
        text = "shared content 你好 " * 40
        p = self._data_dir / "shared.txt"
        p.write_text(text, encoding="utf-8")

        install_isolated_service()  # fresh dir for the inline run
        d1 = install_isolated_service()
        out1 = json.loads(kb._ingest_handler(json.dumps(
            {"docs": [{"title": "t", "content": text, "status": "pending"}]})))
        self.assertEqual(out1["status"], "success")
        docid_inline = out1["staged"][0]["doc_id"] if "doc_id" in out1["staged"][0] \
            else out1["staged"][0]

        d2 = install_isolated_service()
        (d2 / "shared.txt").write_text(text, encoding="utf-8")
        out2 = json.loads(kb._ingest_handler(json.dumps(
            {"docs": [{"title": "t", "doc_file": "shared.txt",
                       "status": "pending"}]})))
        self.assertEqual(out2["status"], "success")
        docid_file = out2["staged"][0]["doc_id"] if "doc_id" in out2["staged"][0] \
            else out2["staged"][0]
        self.assertEqual(docid_inline, docid_file)

    def test_doc_file_read_failure_is_structured_error(self):
        args = {"docs": [{"title": "t", "doc_file": "missing.txt",
                           "status": "pending"}]}
        out = json.loads(kb._ingest_handler(args))
        self.assertEqual(out["status"], "error")
        self.assertIn("doc_file", out.get("error", ""))

    def test_doc_without_content_or_doc_file_rejected(self):
        """m-5 guard: schema says 'content' OR 'doc_file' required; the
        handler enforces it structurally, not via the host schema layer."""
        out = json.loads(kb._ingest_handler(
            {"docs": [{"title": "t", "status": "pending"}]}))
        self.assertEqual(out["status"], "error")
        self.assertIn("'content' or 'doc_file' is required",
                      out.get("error", ""))

    def test_doc_with_blank_doc_file_and_blank_content_rejected(self):
        out = json.loads(kb._ingest_handler(
            {"docs": [{"title": "t", "content": "   ",
                       "doc_file": "", "status": "pending"}]}))
        self.assertEqual(out["status"], "error")
        self.assertIn("'content' or 'doc_file' is required",
                      out.get("error", ""))

    def test_mixed_inline_and_doc_file_batch_ok(self):
        """A batch with one inline doc and one doc_file doc both land."""
        p = self._data_dir / "b.txt"
        p.write_text("file body 你好", encoding="utf-8")
        out = json.loads(kb._ingest_handler(
            {"docs": [
                {"title": "inline", "content": "inline body 你好 ",
                 "status": "pending"},
                {"title": "from-file", "doc_file": "b.txt",
                 "status": "pending"},
            ]}))
        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["staged"]), 2)

    def test_stringified_args_with_doc_file(self):
        """Host path: args arrives as a JSON string; doc_file still works."""
        p = self._data_dir / "doc.txt"
        p.write_text("host-stringified 你好", encoding="utf-8")
        out = json.loads(kb._ingest_handler(json.dumps(
            {"docs": [{"title": "t", "doc_file": "doc.txt"}]})))
        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["staged"]), 1)


class EscapePeelHandlerTests(unittest.TestCase):
    """A: _ingest_handler end-to-end with an accidentally double-serialized
    payload (the exact shape that used to get repeatedly rejected)."""

    def setUp(self):
        self._data_dir = install_isolated_service()

    def test_double_serialized_doc_is_healed_not_rejected(self):
        real = "healed 你好 content " * 20
        # The model accidentally double-serialized: the WHOLE args payload
        # was json.dumps'd, and each doc's content was json.dumps'd again.
        args_payload = {
            "docs": [{"title": "t", "content": json.dumps(json.dumps(real)),
                       "status": "pending"}],
        }
        out = json.loads(kb._ingest_handler(json.dumps(args_payload)))
        self.assertEqual(out["status"], "success")
        self.assertEqual(len(out["staged"]), 1)
        notes = out.get("payload_notes") or []
        self.assertTrue(any("peeled" in n for n in notes))

    def test_payload_note_reports_peel(self):
        real = "note 你好 " * 20
        out = json.loads(kb._ingest_handler(json.dumps(
            {"docs": [{"title": "t", "content": json.dumps(real),
                       "status": "pending"}]})))
        self.assertEqual(out["status"], "success")
        self.assertTrue(any("peeled" in n for n in (out.get("payload_notes") or [])))


if __name__ == "__main__":
    unittest.main()
