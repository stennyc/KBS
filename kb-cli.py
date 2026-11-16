#!/usr/bin/env python3
"""kb-cli.py — terminal escape hatch for knowledge-base ingest/search.

WHY THIS EXISTS
----------------
The host ``tool_call`` bridge (tools/tool_search_validation.py) stringifies the
WHOLE call as a nested JSON envelope. Long document bodies (quotes, newlines,
code blocks, CJK) bloat that envelope and in practice break it mid-string at
~3KB, so the entire tool_call fails host-side JSON parsing and the doc is lost.
This script lets the model bypass that envelope entirely: the body stays a plain
file on disk (written by write_file), and this CLI feeds it straight into the
same ingest core that the tool face uses. No hermes-native change.

USAGE
-----
    # Ingest one doc from a file on disk (the common case — long body):
    python3 kb-cli.py ingest --doc-file /abs/path/to/body.md \
        --title "Some title" --status confirmed --origin-kind agent \
        --profile coder [--doc-kind experience]

    # Ingest with an inline (short) body instead — only for < ~1KB plain text:
    python3 kb-cli.py ingest --content "short body" --title "t"

    # Read back:
    python3 kb-cli.py search "query text" [--top-k 5] [--min-score 0.3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

from kbs_dep_bootstrap import resolve_hermes_home  # noqa: E402

# Keep the service / data-dir resolution identical to the tool face so a CLI
# ingest lands in the same corpus (same lock, same seq, same hot/cold split).
_hermes_home = resolve_hermes_home()
_DATA_DIR = Path(os.environ.get("KB_DATA_DIR", os.path.join(_hermes_home, "kb-data")))
_AUDIT_PATH = os.environ.get(
    "KB_AUDIT_PATH",
    os.path.join(_hermes_home,
                 "logs", "knowledge-base-ingest.jsonl"))


def _svc():
    from kb_service import KbSearchService, configure_logging
    _log_path = os.environ.get(
        "KB_LOG_PATH", os.path.join(_hermes_home, "logs", "knowledge-base.log"))
    configure_logging(_log_path)
    return KbSearchService(_DATA_DIR, audit_path=Path(_AUDIT_PATH))


def _read_body(args) -> tuple[str | None, str | None]:
    """Return (content, error). --doc-file wins; --content is the inline path."""
    if args.doc_file:
        p = Path(args.doc_file).expanduser()
        if not p.is_file():
            return None, f"--doc-file '{args.doc_file}' is not a readable file."
        try:
            return p.read_text(encoding="utf-8"), None
        except OSError as exc:
            return None, f"--doc-file read failed: {exc}"
    if args.content:
        return args.content, None
    return None, "provide --doc-file (preferred for long bodies) or --content"


def cmd_ingest(args) -> int:
    content, err = _read_body(args)
    if err:
        print(json.dumps({"status": "error", "error": err}, ensure_ascii=False))
        return 2
    doc: dict = {
        "title": args.title,
        "content": content,
        "status": args.status,
        "origin_kind": args.origin_kind,
        "source_profile": args.profile,
    }
    if args.doc_kind:
        doc["doc_kind"] = args.doc_kind
    svc = _svc()
    result = svc.ingest([doc])
    svc.persist()  # durable across process restarts (mirrors the tool face)
    out = {
        "status": "success",
        "ingested": result.get("ingested", []),
        "rejected": result.get("rejected", []),
        "staged": result.get("staged", []),
        "generation": result.get("generation"),
    }
    if args.doc_file:
        out["payload_notes"] = [f"content read from --doc-file '{args.doc_file}'"]
    print(json.dumps(out, ensure_ascii=False))
    return 0


def cmd_search(args) -> int:
    svc = _svc()
    payload = svc.search(
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score,
        rerank=True,
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kb-cli.py",
                                 description="Terminal escape hatch for kb ingest/search")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Ingest a doc (from a file or inline body)")
    ing.add_argument("--doc-file",
                     help="Absolute path to the doc body file (preferred; bypasses "
                          "the tool_call JSON envelope that breaks on long text)")
    ing.add_argument("--content",
                     help="Inline body text (only for short plain-text bodies)")
    ing.add_argument("--title", required=True, help="Doc title")
    ing.add_argument("--status", choices=["pending", "confirmed"],
                     default="confirmed")
    ing.add_argument("--origin-kind", choices=["auto", "user", "factory", "agent"],
                     default="agent")
    ing.add_argument("--profile", default=None,
                     help="source_profile (defaults to the running user's profile)")
    ing.add_argument("--doc-kind", default=None,
                     help="doc_kind label (5-value expansion; optional)")
    ing.set_defaults(func=cmd_ingest)

    sr = sub.add_parser("search", help="Search the local two-layer corpus")
    sr.add_argument("query", help="Search query (latin words and/or CJK terms)")
    sr.add_argument("--top-k", type=int, default=5)
    sr.add_argument("--min-score", type=float, default=0.3)
    sr.set_defaults(func=cmd_search)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
