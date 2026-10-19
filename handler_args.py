"""KB handler args normalization: a plugin tool handler may receive its
`args` as a JSON-SERIALIZED STRING instead of a dict (the host tool_call
wrapper stringifies arguments; a JSON-string payload that is not
re-parsed before the handler runs reaches the handler as `str`).

Defect (reproduced 2026-10-19): `_ingest_handler` crashed with
`AttributeError: 'str' object has no attribute 'get'` when args was a
stringified JSON — a bare traceback, not a graceful error payload, and
the same crash shape exists in every sibling handler in this module
(_search_handler, _topic_search_handler, _confirm_handler,
_distill_handler, _share_to_global_handler, _set_doc_kind_handler,
_get_handler).

Fix: `_normalize_handler_args` decodes a JSON-string `args` back into a
dict (the host `args` field is a JSON document) and fails CLOSED with
a structured `{"status":"error","error":"invalid_args",...}` payload
when the payload is not a JSON object — NOT a crash and NOT a silent
fallback to `{}` (an empty-dict fallback would make `doc_id`-required
handlers return a misleading "doc_id is required" for a call that
actually had a well-formed payload the caller mis-serialized).
A dict `args` passes through untouched (byte-identical for every
existing caller — the host's normal path hands handlers a dict).

Pure function, zero I/O, no service / plugin import (the caller wires
it into each handler's first line; see __init__.py).
"""

from __future__ import annotations

import json
from typing import Any


def _normalize_handler_args(args: Any) -> Any:
    """Return `args` as a dict, or a graceful error-payload dict.

    Returns:
      - `args` itself when it is a dict (the normal host path).
      - the decoded dict when `args` is a JSON string holding a JSON
        object (`'{"docs": [...]}'` -> `{"docs": [...]}`).
      - `{"status": "error", "error": "invalid_args", "message": ...}`
        for any other shape: a non-object JSON string (a JSON string
        literal / number / bare null), a non-JSON string, or None.
    The error payload is itself a dict the handler can pass straight to
    `json.dumps(..., ensure_ascii=False)` — callers that follow the
    module's existing pattern (`return json.dumps(<dict>)`) need no
    other change; only the first-line normalization is new.
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return {"status": "error", "error": "invalid_args",
                    "message": "empty arguments payload"}
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError) as exc:
            return {"status": "error", "error": "invalid_args",
                    "message": f"arguments payload is not valid JSON: {exc}"}
        if isinstance(parsed, dict):
            return parsed
        return {"status": "error", "error": "invalid_args",
                "message": "arguments payload must be a JSON object"}
    # None / other (a handler receiving a non-dict, non-str payload is
    # a host dispatch bug — fail closed with the same structured shape).
    return {"status": "error", "error": "invalid_args",
            "message": "arguments payload must be an object or a JSON string"}
