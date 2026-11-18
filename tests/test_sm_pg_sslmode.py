"""kbs F4 client TLS (2026-11-18): sm_pg_sslmode flows panel -> loader ->
legacy pg dict -> psycopg2.connect. Default "prefer" keeps every existing
device byte-identical (libpq default); a live config sets "require".

Mirrors filemem's tests/test_plugin_sm_manifest.py sslmode block (kbs parity
manifest: same 9-field config_schema, same types + defaults).
"""
import pathlib
import sys
from contextlib import contextmanager
from unittest import mock

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
    real = sm._read_config_yaml_text
    sm._read_config_yaml_text = lambda: yaml_text
    try:
        yield
    finally:
        sm._read_config_yaml_text = real


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


def test_reader_panel_sm_sslmode_require():
    sm = _import_kb_shared_memory()
    text = _PANEL_CFG.replace("sm_pg_password: pw-x\n",
                              "sm_pg_password: pw-x\n        sm_pg_sslmode: require\n")
    with _with_fake_config_yaml(sm, text):
        cfg = sm._load_shared_memory_config()
    assert cfg is not None
    assert cfg["pg"]["sslmode"] == "require"


def test_reader_sslmode_defaults_to_prefer():
    sm = _import_kb_shared_memory()
    text = (
        "plugins:\n  entries:\n    knowledge-base-search:\n"
        "      settings:\n"
        "        sm_enabled: true\n"
        "        sm_agent_id: agent-x\n"
        "        sm_pg_host: 10.0.0.5\n"
        "        sm_pg_db: sharedmem\n"
    )
    with _with_fake_config_yaml(sm, text):
        cfg = sm._load_shared_memory_config()
    assert cfg is not None
    assert cfg["pg"]["sslmode"] == "prefer"


def test_reader_legacy_nested_pg_sslmode():
    sm = _import_kb_shared_memory()
    text = (
        "plugins:\n  entries:\n    knowledge-base-search:\n"
        "      shared_memory:\n"
        "        enabled: true\n"
        "        agent_id: legacy-agent\n"
        "        pg:\n"
        "          host: 127.0.0.1\n"
        "          db: sharedmem\n"
        "          user: sharedmem\n"
        "          sslmode: verify-full\n"
    )
    with _with_fake_config_yaml(sm, text):
        cfg = sm._load_shared_memory_config()
    assert cfg is not None
    assert cfg["pg"]["sslmode"] == "verify-full"


def test_minimal_parser_pg_sslmode_key():
    """The minimal parser's legacy pg sub-block picks up sslmode (it has no
    panel-sm shape — a separate kbs 三缺口 item, out of scope here)."""
    sm = _import_kb_shared_memory()
    text = (
        "plugins:\n  entries:\n    knowledge-base-search:\n"
        "      shared_memory:\n"
        "        enabled: true\n"
        "        agent_id: legacy-agent\n"
        "        pg:\n"
        "          host: 127.0.0.1\n"
        "          db: sharedmem\n"
        "          sslmode: verify-ca\n"
    )
    out = sm._parse_kb_shared_memory_minimal(text)
    sm_block = out["plugins"]["entries"]["knowledge-base-search"]["shared_memory"]
    assert sm_block["pg"]["sslmode"] == "verify-ca"


def test_pg_conn_passes_sslmode_kwarg():
    """_PgConn must forward sslmode to psycopg2.connect; a pg dict without
    it (older config shape) must default to 'prefer'."""
    sm = _import_kb_shared_memory()
    calls = []

    class _FakeConn:
        autocommit = False

    def _fake_connect(**kw):
        calls.append(kw)
        return _FakeConn()

    fake_mod = mock.Mock()
    fake_mod.connect = _fake_connect
    with mock.patch.dict(sys.modules, {"psycopg2": fake_mod}):
        sm._PgConn({"host": "h", "port": 5432, "db": "d",
                    "user": "u", "password": "p", "sslmode": "require"})
        sm._PgConn({"host": "h", "port": 5432, "db": "d",
                    "user": "u", "password": "p"})
    assert calls[0]["sslmode"] == "require"
    assert calls[1]["sslmode"] == "prefer"
