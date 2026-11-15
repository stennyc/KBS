"""Single source of truth for splicing the shared dev-venv site-packages
dir onto sys.path, so kb_service's transitive third-party imports
(rank_bm25, jieba, psycopg2) resolve under ANY calling face:

  - host in-process (gateway loads the plugin package) -> __init__.py already
    calls this on import;
  - a bare `from kb_service import ...` (e.g. the test subprocess workers in
    test_kb_rerank_reinforce.py) -> this module is what kb_service now imports
    FIRST, so the splice happens before hot_layer's `from rank_bm25 import ...`;
  - a future new calling face -> just import dep_bootstrap first (or import
    kb_service, which already does).

Before this module, "whoever calls us must remember to splice the venv
site-packages dir" was an implicit convention: the in-process face honored it
(via __init__.py's _ensure_kb_dep_paths) but the subprocess/worker face never
did, and any new face inherited the same silent break. Making the splice a
guaranteed side effect of importing kb_service itself removes the convention.

Env override: a single KB_DEP_SITES variable ('='-separated candidate dirs)
supersedes the old KB_PY_SITE / KB_PG_SITE pair — one knob for every third-
party dep this plugin needs, instead of two half-overlapping names.

Rollback-safe: every target dir is appended to sys.path only if it isn't
already there, and every dir is only appended if it actually exists
(os.path.isdir) — a host that already resolves rank_bm25/jieba/psycopg2 from
its own site-packages gets a no-op.
"""
from __future__ import annotations

import importlib.util
import glob
import os
import sys
from pathlib import Path


# --- cross-platform HERMES_HOME normalizer -----------------------------------
def norm_hermes_home(raw: str) -> str:
    if sys.platform == 'win32':
        if raw.startswith('/') and not raw.startswith('//'):
            return str(Path.home() / '.hermes')
    return raw


def _find_hermes_managed_venv(hermes_home: str) -> str | None:
    """Return the venv path stored in Hermes' ``facts.json``."""
    installs_dir = os.path.join(hermes_home, "installs")
    if not os.path.isdir(installs_dir):
        return None
    for key_dir in os.listdir(installs_dir):
        facts_path = os.path.join(installs_dir, key_dir, "facts.json")
        if not os.path.isfile(facts_path):
            continue
        try:
            import json
            with open(facts_path, encoding="utf-8-sig") as f:
                facts = json.load(f)
            venv = facts.get("packages", {}).get("venv", {}).get("environment")
            if venv and os.path.isdir(venv):
                return venv
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return None


def _find_dep_dirs() -> list[str]:
    """Every candidate site-packages dir to splice, highest priority first:
    1. the KB_DEP_SITES env override ('=' separated, if set);
    2. Hermes-managed venv from ``facts.json`` — canonical, auto-detected;
    3. the CURRENT interpreter's own site-packages dir (sysconfig
       "purelib") — derived live, never hardcoded, as last resort.
    """
    dirs: list[str] = []
    override = os.environ.get("KB_DEP_SITES")
    if override:
        dirs.extend(d for d in override.split("=") if d)

    hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
    managed = _find_hermes_managed_venv(hermes_home)
    if managed:
        for pattern in [os.path.join(managed, "lib", "python*", "site-packages"),
                        os.path.join(managed, "Lib", "site-packages")]:
            found = glob.glob(pattern)
            dirs.extend(d for d in found if os.path.isdir(d))

    try:
        import sysconfig
        dirs.append(sysconfig.get_paths()["purelib"])
    except Exception:
        pass
    return dirs


def ensure_kb_dep_paths() -> None:
    """Make rank_bm25 + jieba importable from the running interpreter.
    No-op when the host already resolves both; never blocks the import.
    Supersedes __init__.py::_ensure_kb_dep_paths (same logic, now shared)."""
    if importlib.util.find_spec("rank_bm25") and importlib.util.find_spec("jieba"):
        return
    for site in _find_dep_dirs():
        if os.path.isdir(site) and site not in sys.path:
            sys.path.append(site)


def ensure_pg_dep_paths() -> None:
    """Make psycopg2 importable from the running interpreter. No-op when the
    host already resolves it; never blocks. Supersedes
    kb_shared_memory.py::_ensure_pg_dep_paths (same logic, now shared)."""
    if importlib.util.find_spec("psycopg2"):
        return
    for site in _find_dep_dirs():
        if os.path.isdir(site) and site not in sys.path:
            sys.path.append(site)
