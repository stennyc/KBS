"""kbs template placeholders (owner ruling 2026-11-18, item 4; mirrors
filemem's ${SM_TOKEN} / ${SM_PG_PASSWORD} support). Public-repo manifest
ships placeholders; live config real values pass through untouched; unset
env -> "" (central door closed, local face untouched).
"""
import pathlib
import sys
from contextlib import contextmanager

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent.parent


def _import_kb_shared_memory():
    try:
        import kb_shared_memory
        return kb_shared_memory
    except Exception as e:  # pragma: no cover - interpreter-dependent
        import pytest
        pytest.skip(f"kb_shared_memory not importable here: {e}")


@contextmanager
def _with_fake_config_yaml(sm, yaml_text):
    """Pretend config.yaml resolves to `yaml_text`. The P0-F5 mtime cache
    keys on the SOURCE path's stat — the fake path ('fake-path') has no
    stat, so key=None and the cache is bypassed (legacy parse, per-call)."""
    real = sm._read_config_yaml_source
    sm._read_config_yaml_source = lambda: (yaml_text, "fake-path")
    try:
        yield
    finally:
        sm._read_config_yaml_source = real


_PANEL_CFG = (
    "plugins:\n"
    "  entries:\n"
    "    knowledge-base-search:\n"
    "      settings:\n"
    "        sm_enabled: true\n"
    "        sm_agent_id: agent-x\n"
    "        sm_token: tok-x\n"
    "        sm_pg_host: 10.0.0.5\n"
    "        sm_pg_port: 6432\n"
    "        sm_pg_db: sharedmem\n"
    "        sm_pg_user: sharedmem\n"
    "        sm_pg_password: pw-x\n"
)

_PANEL_CFG_PLACEHOLDERS = _PANEL_CFG.replace(
    "sm_token: tok-x\n", "sm_token: ${SM_TOKEN}\n", 1,
).replace(
    "sm_pg_password: pw-x\n", "sm_pg_password: ${SM_PG_PASSWORD}\n", 1,
)


def test_env_placeholder_token_password_resolved(monkeypatch):
    sm = _import_kb_shared_memory()
    monkeypatch.setenv("SM_TOKEN", "tok-live")
    monkeypatch.setenv("SM_PG_PASSWORD", "pw-live")
    with _with_fake_config_yaml(sm, _PANEL_CFG_PLACEHOLDERS):
        cfg = sm._load_shared_memory_config()
    assert cfg is not None
    assert cfg["token"] == "tok-live"
    assert cfg["pg"]["password"] == "pw-live"


def test_env_placeholder_unset_yields_empty(monkeypatch):
    sm = _import_kb_shared_memory()
    monkeypatch.delenv("SM_TOKEN", raising=False)
    monkeypatch.delenv("SM_PG_PASSWORD", raising=False)
    with _with_fake_config_yaml(sm, _PANEL_CFG_PLACEHOLDERS):
        cfg = sm._load_shared_memory_config()
    assert cfg is not None
    assert cfg["token"] == ""
    assert cfg["pg"]["password"] == ""


def test_env_placeholder_partial_pattern_passthrough():
    sm = _import_kb_shared_memory()
    text = _PANEL_CFG.replace("sm_token: tok-x\n", "sm_token: abc${SM_TOKEN}def\n", 1)
    with _with_fake_config_yaml(sm, text):
        cfg = sm._load_shared_memory_config()
    assert cfg is not None
    assert cfg["token"] == "abc${SM_TOKEN}def"


def test_live_value_passthrough_unchanged():
    sm = _import_kb_shared_memory()
    with _with_fake_config_yaml(sm, _PANEL_CFG):
        cfg = sm._load_shared_memory_config()
    assert cfg["token"] == "tok-x"
    assert cfg["pg"]["password"] == "pw-x"


def test_manifest_ships_placeholders_not_live_credentials():
    import yaml
    data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
    cs = data["config_schema"]
    assert cs["sm_token"]["default"] == "${SM_TOKEN}"
    assert cs["sm_pg_password"]["default"] == "${SM_PG_PASSWORD}"
