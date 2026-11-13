"""Shared fixture: point the plugin module-level service at a disposable
data dir so tests that call handlers DIRECTLY (kb._ingest_handler etc.)
never write into the production /root/.hermes/kb-data.

Root cause (2026-10-19): test_handler_args.py imports `__init__ as kb` and
calls `kb._ingest_handler(...)` without any KB_DATA_DIR isolation — the
handler resolves its service lazily via `_svc()`, which uses the module-level
`_DATA_DIR` (production `/root/.hermes/kb-data` by default) and the cached
module-level `_service`. A pending ingest there appends rows to the LIVE
staging store. This fixture rebinds BOTH the env vars (for any fresh
import) and the cached `kb._service` singleton (for the already-imported
module instance) to a per-test temp dir.

Usage (call it at the top of any test that invokes a kb handler directly):

    from test_kb_isolation import install_isolated_service
    install_isolated_service()

The temp dir is created on demand under the kernel scratch area and is NOT
cleaned up here (leaving it is harmless; the kernel prunes scratch after
24h). If a test file needs its OWN already-seeded corpus, pass
data_dir= to control the location.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))


def _make_data_dir() -> Path:
    """Create the minimal directory shape KbSearchService's constructor
    requires (data dir + cold/ + staging/) plus the lock file."""
    base = Path(tempfile.mkdtemp(prefix="kb_iso_"))
    data = base / "kb-data"
    data.mkdir()
    (data / "cold").mkdir()
    (data / "staging").mkdir()
    (data / "lock").touch()
    return data


def install_isolated_service(data_dir: Path | None = None) -> Path:
    """Rebind the plugin's cached service singleton to a temp data dir.

    Returns the data dir the service now uses. Idempotent per call: each
    invocation creates a FRESH temp dir so tests don't share write state.
    Sets KB_DATA_DIR / KB_AUDIT_PATH env (fresh-import path) AND overwrites
    `kb._service` (already-imported path) so `_svc()` returns the isolated
    instance on every handler call.
    """
    import __init__ as kb  # the plugin module
    from kb_service import KbSearchService, configure_logging

    data_dir = data_dir or _make_data_dir()
    audit_path = data_dir.parent / "audit.jsonl"

    # Env rebind: any LATER `import __init__` in another test module sees
    # the isolated paths too (module-level _DATA_DIR/_AUDIT_PATH are read
    # from env at import time).
    os.environ["KB_DATA_DIR"] = str(data_dir)
    os.environ["KB_AUDIT_PATH"] = str(audit_path)

    # Force the cached module constants to match (covers the common case
    # where `kb` was imported before this fixture ran).
    kb._DATA_DIR = Path(data_dir)
    kb._AUDIT_PATH = str(audit_path)

    # Rebuild the cached service singleton against the isolated dir so
    # `_svc()` (the handler's lazy accessor) returns this instance and
    # never touches production.
    configure_logging(kb._LOG_PATH)
    kb._service = KbSearchService(Path(data_dir), audit_path=Path(audit_path))

    return data_dir


if __name__ == "__main__":
    d = install_isolated_service()
    print("isolated data dir:", d)
