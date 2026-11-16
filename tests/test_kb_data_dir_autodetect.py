"""kb-data directory auto-detection: ONE shared corpus per host.

Split-brain incident (2026-11-17): a relay-side process (HERMES_HOME=
~/.hermes) and the gateway side (HERMES_HOME=%LOCALAPPDATA%\\hermes)
each resolved kb-data into their OWN tree — the same doc (5ca192d2)
was ingested into both on 11-15. Owner ruling: fix via auto-detect,
NO hard-coded paths.

Resolution rule under test:
  1. KB_DATA_DIR env var = an explicit pin — supersedes everything;
  2. the platform-NATIVE canonical home (auto-detected from the
     platform's own env vars + disk, see detect_native_hermes_home)
     owns the shared KB, beating a per-process HERMES_HOME;
  3. HERMES_HOME env var only when no native tree exists on disk
     (fresh host / relocated install) — reuses resolve_hermes_home's
     fallback chain so the last-resort default stays in ONE place.

Testing strategy: this suite runs on whatever host pytest runs on, so
we do NOT fake the whole OS by flipping sys.platform (the real os.path
module doesn't follow that string, and its join()/isdir() keep the
host's native separators). Instead:
  - the PRIORITY ladder of resolve_kb_data_dir is tested by pinning the
    two inputs it delegates to (detect_native_hermes_home /
    resolve_hermes_home) — pure control-flow, host-independent;
  - the PLATFORM-BRANCH logic of detect_native_hermes_home is tested
    with a REAL temp dir (win32 branch) and controlled expanduser+isdir
    (posix branch), so it never depends on this host's disk.
"""
from __future__ import annotations

import os
import sys

import kbs_dep_bootstrap as b
from kbs_dep_bootstrap import (
    detect_native_hermes_home,
    resolve_kb_data_dir,
)


def _clear_kb_env(monkeypatch):
    for var in ("KB_DATA_DIR", "HERMES_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(var, raising=False)


def _norm(p: str) -> str:
    """Normalize a path for cross-separator/case comparison. os.path.join
    uses the HOST's native separator (backslash on Windows) regardless of
    the synthetic forward-slash inputs the tests feed in, so joined-path
    assertions must compare in normalized form."""
    return os.path.normcase(os.path.normpath(p))


# ── resolve_kb_data_dir: the priority ladder (control-flow) ────────────────

def test_explicit_kb_data_dir_pin_wins(monkeypatch):
    """KB_DATA_DIR is an explicit pin — it supersedes BOTH the native
    tree and HERMES_HOME."""
    _clear_kb_env(monkeypatch)
    monkeypatch.setenv("KB_DATA_DIR", "D:/pinned/kb")
    monkeypatch.setattr(b, "detect_native_hermes_home", lambda: "C:/native/hermes")
    assert resolve_kb_data_dir() == "D:/pinned/kb"


def test_native_home_beats_hermes_home(monkeypatch):
    """THE split-brain fix: a native tree exists on disk AND a process
    HERMES_HOME points elsewhere -> the native tree owns the corpus."""
    _clear_kb_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", "C:/Users/user/.hermes")
    monkeypatch.setattr(b, "detect_native_hermes_home", lambda: "C:/native/hermes")
    assert _norm(resolve_kb_data_dir()) == _norm("C:/native/hermes/kb-data")


def test_no_native_tree_falls_back_to_hermes_home(monkeypatch):
    """Fresh host / relocated install: no native tree -> the per-process
    HERMES_HOME decides (the real resolve_hermes_home reads it)."""
    _clear_kb_env(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", "C:/custom/relocated")
    monkeypatch.setattr(b, "detect_native_hermes_home", lambda: None)
    assert _norm(resolve_kb_data_dir()) == _norm("C:/custom/relocated/kb-data")


def test_nothing_at_all_uses_last_resort_default(monkeypatch):
    """No native tree, no HERMES_HOME -> resolve_hermes_home's own
    last-resort default owns the dir (kept in ONE place)."""
    _clear_kb_env(monkeypatch)
    monkeypatch.setattr(b, "detect_native_hermes_home", lambda: None)
    monkeypatch.setattr(b, "resolve_hermes_home", lambda: "C:/fakehome/.hermes")
    assert _norm(resolve_kb_data_dir()) == _norm("C:/fakehome/.hermes/kb-data")


# ── detect_native_hermes_home: the platform branch ─────────────────────────

def test_win32_native_home_from_localappdata(tmp_path, monkeypatch):
    """win32 branch: %LOCALAPPDATA%\\hermes, taken from the platform's own
    env var (NOT a hard-coded absolute path), only when it exists."""
    _clear_kb_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")
    native = tmp_path / "hermes"
    native.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    # this branch uses the REAL os.path.isdir against the real tmp dir
    got = detect_native_hermes_home()
    assert got is not None
    assert os.path.normcase(os.path.normpath(got)) == \
        os.path.normcase(os.path.normpath(str(native)))


def test_win32_no_localappdata_returns_none(monkeypatch):
    """win32 branch with no LOCALAPPDATA env -> nothing to auto-detect."""
    _clear_kb_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "win32")
    assert detect_native_hermes_home() is None


def test_posix_native_home_from_expanduser(monkeypatch):
    """posix branch: ~/.hermes via expanduser, controlled so the test is
    host-independent."""
    _clear_kb_env(monkeypatch)
    monkeypatch.setattr(sys, "platform", "posix")
    # emulate real expanduser semantics: '~'-prefixed input has its leading
    # '~' replaced by the home dir, the rest appended
    monkeypatch.setattr(
        "os.path.expanduser",
        lambda p: "/home/fakeuser" + str(p)[1:] if str(p).startswith("~")
        else str(p))
    monkeypatch.setattr(
        "os.path.isdir", lambda p: str(p) == "/home/fakeuser/.hermes")
    assert detect_native_hermes_home() == "/home/fakeuser/.hermes"
