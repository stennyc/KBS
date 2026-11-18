"""kbs P0-F5: sm-config mtime-gate cache — zero-recall-diff canary.

The kbs pre_llm_call face pays a full config read + YAML parse (up to 2x
per turn) to decide one referral-hint line. The mtime gate must be
byte-invisible: cached result == fresh parse for the same file bytes,
and ANY file rewrite (panel save / hand edit lands a new mtime) is
reflected on the NEXT call. None results (no sm section / disabled) are
cached too — they are pure functions of the file bytes as well.

KBS_PERF_CACHE=off is the A/B reference leg + ops kill-switch.
"""
import os
import pathlib

import pytest

sys_path_fix = pathlib.Path(__file__).resolve().parent.parent
import sys
if str(sys_path_fix) not in sys.path:
    sys.path.insert(0, str(sys_path_fix))

import kb_shared_memory as ksm  # noqa: E402


CFG_A = (
    "plugins:\n"
    "  entries:\n"
    "    knowledge-base-search:\n"
    "      settings:\n"
    "        sm_enabled: true\n"
    "        sm_agent_id: agent-a\n"
    "        sm_token: tok-a\n"
    "        sm_pg_host: 10.0.0.5\n"
    "        sm_pg_db: sharedmem\n"
)

CFG_B = (
    "plugins:\n"
    "  entries:\n"
    "    knowledge-base-search:\n"
    "      settings:\n"
    "        sm_enabled: true\n"
    "        sm_agent_id: agent-b\n"
    "        sm_token: tok-b\n"
    "        sm_pg_host: 10.0.0.6\n"
    "        sm_pg_db: sharedmem2\n"
)

CFG_NO_SM = "plugins:\n  entries:\n    other-plugin:\n      settings: {}\n"


def _write(tmp_path: pathlib.Path, name: str, text: str) -> pathlib.Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _force_new_mtime(p: pathlib.Path) -> None:
    """Windows mtime resolution can coalesce rapid rewrites — bump it
    explicitly so the gate key ALWAYS moves (test determinism)."""
    st = p.stat()
    os.utime(p, (st.st_atime + 5, st.st_mtime + 5))


def _setup(tmp_path, monkeypatch, text: str) -> pathlib.Path:
    cfg = _write(tmp_path, "config.yaml", text)
    monkeypatch.setenv("HERMES_CONFIG", str(cfg))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("KBS_PERF_CACHE", raising=False)  # default: on
    return cfg


def test_cache_hit_steady_file(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch, CFG_A)
    cfg1 = ksm._load_shared_memory_config()
    assert cfg1 is not None and cfg1["agent_id"] == "agent-a"
    # steady file: the parse body must run exactly ONCE across many loads
    parses = []
    orig = ksm._parse_shared_memory_config

    def counting(raw):
        parses.append(raw)
        return orig(raw)

    monkeypatch.setattr(ksm, "_parse_shared_memory_config", counting)
    ksm._SM_CFG_CACHE.clear()  # force the first counted load to be a MISS
    got = ksm._load_shared_memory_config()
    assert got == cfg1 and len(parses) == 1, "cold cache must parse once"
    for _ in range(4):
        got = ksm._load_shared_memory_config()
        assert got == cfg1
    assert len(parses) == 1, f"steady file must be a cache hit, parses={len(parses)}"


def test_freshness_after_rewrite(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch, CFG_A)
    a = ksm._load_shared_memory_config()
    assert a["agent_id"] == "agent-a"
    _write(tmp_path, "config.yaml", CFG_B)
    _force_new_mtime(cfg)
    b = ksm._load_shared_memory_config()
    assert b["agent_id"] == "agent-b", "rewrite must invalidate the gate"
    assert b["pg"]["host"] == "10.0.0.6"


def test_none_result_is_cached_too(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch, CFG_NO_SM)
    assert ksm._load_shared_memory_config() is None
    parses = []
    orig = ksm._parse_shared_memory_config

    def counting(raw):
        parses.append(raw)
        return orig(raw)

    monkeypatch.setattr(ksm, "_parse_shared_memory_config", counting)
    ksm._SM_CFG_CACHE.clear()
    assert ksm._load_shared_memory_config() is None
    assert len(parses) == 1, "cold cache must parse the None result"
    assert ksm._load_shared_memory_config() is None
    assert len(parses) == 1, "None results are pure file functions too"


def test_env_killswitch_bypasses_gate(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch, CFG_A)
    monkeypatch.setenv("KBS_PERF_CACHE", "off")
    parses = []
    orig = ksm._parse_shared_memory_config

    def counting(raw):
        parses.append(raw)
        return orig(raw)

    monkeypatch.setattr(ksm, "_parse_shared_memory_config", counting)
    for _ in range(3):
        assert ksm._load_shared_memory_config() is not None
    assert len(parses) == 3, "env-off leg must parse every call (legacy cost)"


def test_on_off_legs_same_result(tmp_path, monkeypatch):
    """The canary itself: cache-on result == cache-off result, same file."""
    cfg = _setup(tmp_path, monkeypatch, CFG_A)
    monkeypatch.setenv("KBS_PERF_CACHE", "off")
    res_off = ksm._load_shared_memory_config()
    monkeypatch.setenv("KBS_PERF_CACHE", "on")
    res_on = ksm._load_shared_memory_config()
    assert res_off == res_on
