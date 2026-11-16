"""kbs dependency AUTO-INSTALL (the code_map "auto-supplement" pattern).

Owner ruling 2026-11-17: kbs deps must install themselves the way code_map
does — not a manual one-shot. The mechanism under test:
  * fast path — every hard dep already importable -> no pip, splice only;
  * cooldown  — a fresh attempt marker suppresses a repeat pip (a host that
                cannot install does not thrash on every boot);
  * success   — missing deps -> one `pip install` of the hard set (+ the
                best-effort optional PG dep), then the interpreter's purelib
                is spliced so the import works in-process WITHOUT a restart;
  * fail-soft — a failed pip returns False (and the marker is set), never
                raises into the host import;
  * import face — ensure_kb_dep_paths() NEVER blocks: it splices, and if a
                hard dep is still missing it kicks the install in a daemon
                thread (captured here, not actually run).

All tests use a controllable importlib.util.find_spec + a fake
subprocess.run — no network, no real pip.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading

import kbs_dep_bootstrap as b


def _dynamic_find_spec(state, pip_present=True):
    """A find_spec stand-in driven by state['present'] (a set of module
    names). 'pip' is reportable-present unless pip_present=False."""
    def fake(name, *a, **k):
        if name == "pip":
            return type("S", (), {"name": name})() if pip_present else None
        present = state.get("present", set())
        return type("S", (), {"name": name})() if name in present else None
    return fake


def _clear_kb_env(monkeypatch):
    monkeypatch.delenv("KB_DEP_SITES", raising=False)


def test_present_fastpath_no_pip(monkeypatch, tmp_path):
    """Every hard dep importable -> True, and pip is NEVER invoked."""
    _clear_kb_env(monkeypatch)
    state = {"present": {"jieba", "rank_bm25"}}
    monkeypatch.setattr("importlib.util.find_spec",
                        _dynamic_find_spec(state))
    monkeypatch.setattr(b, "_marker_path", lambda: str(tmp_path / "m"))
    import subprocess as sp
    def no_pip(*a, **k):
        raise AssertionError("pip must not run when deps are present")
    monkeypatch.setattr(sp, "run", no_pip)
    assert b.ensure_kb_deps_installed() is True


def test_fresh_marker_blocks_pip(monkeypatch, tmp_path):
    """Deps missing but a fresh attempt marker exists -> False, no pip
    (the cooldown guard: an uninstallable host does not thrash)."""
    _clear_kb_env(monkeypatch)
    state = {"present": set()}
    monkeypatch.setattr("importlib.util.find_spec", _dynamic_find_spec(state))
    marker = tmp_path / "m"
    marker.write_text("attempt")          # mtime = now -> fresh
    monkeypatch.setattr(b, "_marker_path", lambda: str(marker))
    import subprocess as sp
    def no_pip(*a, **k):
        raise AssertionError("pip must not run under a fresh marker")
    monkeypatch.setattr(sp, "run", no_pip)
    assert b.ensure_kb_deps_installed() is False


def test_missing_deps_trigger_pip_and_succeed(monkeypatch, tmp_path):
    """Missing hard deps + no fresh marker -> pip install of the hard set
    (which GATES the result), then a SEPARATE best-effort pip for the
    optional PG dep — so a build failure of the optional can never sink
    the hard ones."""
    _clear_kb_env(monkeypatch)
    state = {"present": set()}
    monkeypatch.setattr("importlib.util.find_spec", _dynamic_find_spec(state))
    monkeypatch.setattr(b, "_marker_path", lambda: str(tmp_path / "m"))
    import subprocess as sp
    cmds = []
    def fake_run(cmd, **kw):
        cmds.append(list(cmd))
        # simulate installs landing per-call: the hard-set call lands the
        # hard deps only; the optional call lands the optional — so the
        # opt_missing recheck (between the two calls) still sees it absent
        # and the split is genuinely exercised
        joined = " ".join(cmd)
        if "rank_bm25" in joined:
            state["present"] |= {"jieba", "rank_bm25"}
        elif "psycopg2" in joined:
            state["present"] |= {"psycopg2"}
        return type("R", (), {"returncode": 0})()
    monkeypatch.setattr(sp, "run", fake_run)
    assert b.ensure_kb_deps_installed() is True
    assert len(cmds) == 2, cmds
    hard_cmd = " ".join(cmds[0])
    assert "pip" in hard_cmd and "install" in hard_cmd
    for dep in ("jieba", "rank_bm25"):
        assert dep in hard_cmd, (dep, hard_cmd)
    # the optional dep is its OWN separate call (best-effort, non-gating)
    if len(cmds) > 1:
        opt_cmd = " ".join(cmds[1])
        assert "psycopg2" in opt_cmd
    # marker now exists (so a 2nd call is cooldown-suppressed)
    assert os.path.exists(str(tmp_path / "m"))


def test_pip_failure_is_failsoft(monkeypatch, tmp_path):
    """A failed pip -> False (never raises), and the marker is set so the
    host does not retry within the cooldown window."""
    _clear_kb_env(monkeypatch)
    state = {"present": set()}
    monkeypatch.setattr("importlib.util.find_spec", _dynamic_find_spec(state))
    marker = tmp_path / "m"
    monkeypatch.setattr(b, "_marker_path", lambda: str(marker))
    import subprocess as sp
    def failing_run(cmd, **kw):
        state["present"] = set()          # nothing landed
        return type("R", (), {"returncode": 1})()
    monkeypatch.setattr(sp, "run", failing_run)
    assert b.ensure_kb_deps_installed() is False
    assert os.path.exists(str(marker))


def test_no_pip_means_failsoft(monkeypatch, tmp_path):
    """An interpreter with no pip cannot self-install -> False, no crash."""
    _clear_kb_env(monkeypatch)
    state = {"present": set()}
    monkeypatch.setattr("importlib.util.find_spec",
                       _dynamic_find_spec(state, pip_present=False))
    monkeypatch.setattr(b, "_marker_path", lambda: str(tmp_path / "m"))
    import subprocess as sp
    def no_pip(*a, **k):
        raise AssertionError("pip must not be invoked when unavailable")
    monkeypatch.setattr(sp, "run", no_pip)
    assert b.ensure_kb_deps_installed() is False


def test_import_face_spawns_background_install(monkeypatch, tmp_path):
    """ensure_kb_dep_paths() never blocks: with a hard dep still missing it
    kicks the install in a DAEMON thread (captured, not actually run)."""
    _clear_kb_env(monkeypatch)
    state = {"present": set()}
    monkeypatch.setattr("importlib.util.find_spec", _dynamic_find_spec(state))
    monkeypatch.setattr(b, "_marker_path", lambda: str(tmp_path / "m"))
    spawned = []
    class _FakeThread:
        def __init__(self, target=None, daemon=None, name=None):
            spawned.append(target)
        def start(self):            # do NOT actually run the install
            pass
    monkeypatch.setattr(threading, "Thread", _FakeThread)
    b.ensure_kb_dep_paths()          # must return immediately
    assert len(spawned) == 1 and spawned[0] is b.ensure_kb_deps_installed


def test_import_face_noop_when_present(monkeypatch, tmp_path):
    """With all hard deps present, the import face is a pure no-op (no
    thread, no pip)."""
    _clear_kb_env(monkeypatch)
    state = {"present": {"jieba", "rank_bm25"}}
    monkeypatch.setattr("importlib.util.find_spec", _dynamic_find_spec(state))
    spawned = []
    class _FakeThread:
        def __init__(self, *a, **k):
            spawned.append(1)
        def start(self):
            pass
    monkeypatch.setattr(threading, "Thread", _FakeThread)
    b.ensure_kb_dep_paths()
    assert spawned == []
