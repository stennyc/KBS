"""Single source of truth for splicing the shared dev-venv site-packages
dir onto sys.path, so kb_service's transitive third-party imports
(rank_bm25, jieba, psycopg2) resolve under ANY calling face:

  - host in-process (gateway loads the plugin package) -> __init__.py already
    calls this on import;
  - a bare `from kb_service import ...` (e.g. the test subprocess workers in
    test_kb_rerank_reinforce.py) -> this module is what kb_service now imports
    FIRST, so the splice happens before hot_layer's `from rank_bm25 import ...`;
  - a future new calling face -> just import kbs_dep_bootstrap first (or import
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
        # Only normalize known Linux defaults; arbitrary paths are left as-is.
        if raw in ('/root/.hermes', '/.hermes'):
            return str(Path.home() / '.hermes')
    return raw


def resolve_hermes_home() -> str:
    """Resolve HERMES_HOME with cross-platform auto-detection.

    Priority:
      1. ``HERMES_HOME`` env var (set by multiplex gateway per-profile)
      2. Windows: ``%LOCALAPPDATA%\\hermes`` (desktop app default on Win)
      3. Fallback: ``~/.hermes`` (POSIX / legacy)
    """
    hh = os.environ.get("HERMES_HOME", "").strip()
    if hh:
        return norm_hermes_home(hh)
    if sys.platform == "win32":
        localapp = os.environ.get("LOCALAPPDATA", "").strip()
        if localapp:
            candidate = os.path.join(localapp, "hermes")
            if os.path.isdir(candidate):
                return candidate
    return os.path.expanduser("~/.hermes")


def detect_native_hermes_home() -> str | None:
    """The platform-native canonical hermes home, AUTO-DETECTED — no
    hard-coded absolute paths, only the platform's own env vars + home:

      - Windows: ``%LOCALAPPDATA%\\hermes`` (the installer's data home;
        the Windows-native docs' data layout makes it the canonical tree)
      - POSIX: ``~/.hermes``

    Returns the path only when it EXISTS on disk; None on a fresh or
    relocated host so callers fall back to the per-process HERMES_HOME.
    """
    if sys.platform == "win32":
        localapp = os.environ.get("LOCALAPPDATA", "").strip()
        if localapp:
            candidate = os.path.join(localapp, "hermes")
            if os.path.isdir(candidate):
                return candidate
        return None
    candidate = os.path.expanduser("~/.hermes")
    if os.path.isdir(candidate):
        return candidate
    return None


def resolve_kb_data_dir() -> str:
    """The shared KB corpus dir: ONE per host, auto-detected.

    Split-brain fix (2026-11-17): a relay-side process (HERMES_HOME=
    ``~/.hermes``) and the gateway side (HERMES_HOME=``%LOCALAPPDATA%\\hermes``)
    each resolved kb-data into their OWN tree — the same doc was ingested
    into both. Owner ruling: converge via auto-detection, no hard-coded
    paths. Rule:

      1. ``KB_DATA_DIR`` env var = an explicit pin, supersedes everything;
      2. the platform-native canonical home (``detect_native_hermes_home``)
        owns the shared KB — every process that loads this plugin lands on
        the SAME tree, regardless of its own HERMES_HOME;
      3. the ``HERMES_HOME`` env var only when no native tree exists on
        disk (fresh host / relocated install) — reuses
        ``resolve_hermes_home`` so the last-resort default stays in ONE
        place.
    """
    explicit = os.environ.get("KB_DATA_DIR", "").strip()
    if explicit:
        return explicit
    native = detect_native_hermes_home()
    if native:
        return os.path.join(native, "kb-data")
    return os.path.join(resolve_hermes_home(), "kb-data")


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

    hermes_home = resolve_hermes_home()
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
    """Import-time face: FAST. Make rank_bm25 + jieba importable from the
    running interpreter WITHOUT blocking the host.

    Supersedes __init__.py::_ensure_kb_dep_paths (same logic, now shared).
    Behaviour:
      1. both deps already importable -> no-op;
      2. otherwise splice the candidate site-packages dirs (fast path);
      3. if a hard dep is STILL missing after splicing, kick the
         marker-guarded, fail-soft pip auto-install in a BACKGROUND thread
         (the code_map "auto-supplement" pattern, owner 2026-11-17) — the
         host import NEVER stalls; the install persists into this
         interpreter's site-packages so the next boot resolves it.
    """
    if importlib.util.find_spec("rank_bm25") and importlib.util.find_spec("jieba"):
        return
    for site in _find_dep_dirs():
        if os.path.isdir(site) and site not in sys.path:
            sys.path.append(site)
    if importlib.util.find_spec("rank_bm25") and importlib.util.find_spec("jieba"):
        return
    # Still missing -> background auto-install (never block the host import).
    if _marker_fresh():
        return
    import threading
    threading.Thread(target=ensure_kb_deps_installed, daemon=True,
                     name="kb-dep-autoinstall").start()


def ensure_pg_dep_paths() -> None:
    """Make psycopg2 importable from the running interpreter. No-op when the
    host already resolves it; never blocks. Supersedes
    kb_shared_memory.py::_ensure_pg_dep_paths (same logic, now shared)."""
    if importlib.util.find_spec("psycopg2"):
        return
    for site in _find_dep_dirs():
        if os.path.isdir(site) and site not in sys.path:
            sys.path.append(site)


# ---------------------------------------------------------------------------
# Auto-install (the code_map "auto-supplement" pattern, owner 2026-11-17):
# when the hard core deps are missing from the running interpreter, install
# them ONCE into that interpreter's own site-packages (pip) and splice the
# dir back onto sys.path, so the import works immediately (in-process) AND
# persists across restarts. A cooldown marker keeps an uninstallable host
# (no network) from thrashing pip on every boot. Fail-soft throughout: any
# failure degrades to the pre-change behaviour — this module's import never
# raises, the KB face just stays off until the deps land.
# ---------------------------------------------------------------------------

KB_DEP_REQUIRED = ("jieba", "rank_bm25")   # hard: the two-layer search core
KB_DEP_OPTIONAL = ("psycopg2-binary",)     # soft: imports as 'psycopg2' (the
                                           # shared-memory face; -binary =
                                           # a wheel, no pg_config/C build)
INSTALL_COOLDOWN_S = 6 * 3600               # re-attempt pip at most every 6h


def _pip_available() -> bool:
    try:
        return importlib.util.find_spec("pip") is not None
    except Exception:
        return False


def _marker_path() -> str:
    """Cooldown marker: next to the plugin (always writable from the running
    interpreter, survives restarts) — never under a read-only install root."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       ".kb_deps_last_attempt")


def _marker_fresh() -> bool:
    try:
        import time
        return (time.time() - os.path.getmtime(_marker_path())) < INSTALL_COOLDOWN_S
    except OSError:
        return False


def _touch_marker() -> None:
    try:
        with open(_marker_path(), "a", encoding="utf-8") as f:
            f.write("")
    except OSError:
        pass


def ensure_kb_deps_installed() -> bool:
    """SYNCHRONOUS auto-install worker (call it directly from a CLI / the
    runbook to install-and-verify NOW; the import face fires it in a thread).

    Installs any missing KB_DEP_REQUIRED (and best-effort KB_DEP_OPTIONAL)
    into the RUNNING interpreter via pip, then splices its purelib for an
    immediate in-process import. Returns True only when every hard dep is
    importable afterwards. Fail-soft: no pip / no network / timeout -> False
    (and the marker is set, so we don't retry within the cooldown)."""
    missing = [m for m in KB_DEP_REQUIRED
               if importlib.util.find_spec(m) is None]
    if not missing:
        # Already present -> splice for a foreign-interpreter case, done.
        for site in _find_dep_dirs():
            if os.path.isdir(site) and site not in sys.path:
                sys.path.append(site)
        return True
    if _marker_fresh():
        return False
    _touch_marker()
    if not _pip_available():
        return False
    import subprocess

    def _pip(targets: list[str]) -> None:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 *targets],
                timeout=240, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            pass  # fail-soft: the hard-import recheck below decides

    # HARD set first — its result GATES the return value.
    _pip(missing)
    # OPTIONAL set in a SEPARATE call: a failure building an optional dep
    # (the first design shipped psycopg2 source in ONE pip call and its
    # build failure sank the whole install, including the hard deps) must
    # never gate the hard set. Checked by module name ('psycopg2'),
    # installed by pip name ('psycopg2-binary' — a wheel, no C build).
    _OPT_MODULE = {"psycopg2-binary": "psycopg2"}
    opt_missing = [p for p in KB_DEP_OPTIONAL
                   if importlib.util.find_spec(
                       _OPT_MODULE.get(p, p)) is None]
    if opt_missing:
        _pip(opt_missing)
    still_missing = [m for m in KB_DEP_REQUIRED
                     if importlib.util.find_spec(m) is None]
    if still_missing:
        return False
    try:
        import sysconfig
        purelib = sysconfig.get_paths()["purelib"]
        if os.path.isdir(purelib) and purelib not in sys.path:
            sys.path.append(purelib)
    except Exception:
        pass
    return True
