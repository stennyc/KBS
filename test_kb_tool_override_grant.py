"""Regression: knowledge-base-search plugin must be granted `tools.override`
in EVERY home that loads it (global /root/.hermes/config.yaml + every profile
config). Without the grant, the host's capability gate denies the plugin's
tool registration on that home -> kb tools absent from the tool surface for
that manager.

Root cause (2026-10-28 owner report): agent.log showed
`capability_check plugin=knowledge-base-search capability=tools.override
decision=deny evidence=not granted` from the global-home manager (gateway /
cron / desktop app) — that manager reads /root/.hermes/config.yaml whose kb
entry lacked granted_capabilities. The coder profile's own home had the
grant; every other home did not.

This test only READS config.yaml files — it never writes. Failure = a home
is missing the grant; the fix is to add
    plugins.entries.knowledge-base-search.granted_capabilities: [tools.override]
to that home's config.yaml.
"""

from __future__ import annotations

import glob
import unittest
from pathlib import Path

import yaml

_GLOBAL_CONFIG = Path("/root/.hermes/config.yaml")
_PROFILE_CONFIGS = sorted(glob.glob("/root/.hermes/profiles/*/config.yaml"))

PLUGIN_ID = "knowledge-base-search"
REQUIRED_CAP = "tools.override"


def _kb_entry(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    entries = (cfg.get("plugins") or {}).get("entries") or {}
    entry = entries.get(PLUGIN_ID)
    return entry if isinstance(entry, dict) else {}


def _granted_caps(config_path: Path) -> list:
    entry = _kb_entry(config_path)
    caps = entry.get("granted_capabilities")
    if not isinstance(caps, list):
        return []
    return [c for c in caps if isinstance(c, str)]


class KbToolOverrideGrantTests(unittest.TestCase):
    """Every home that enables the kb plugin must grant tools.override."""

    def test_global_home_has_kb_entry(self):
        entry = _kb_entry(_GLOBAL_CONFIG)
        self.assertTrue(entry, f"global config has no plugins.entries.{PLUGIN_ID}")

    def test_global_home_grants_tools_override(self):
        caps = _granted_caps(_GLOBAL_CONFIG)
        self.assertIn(
            REQUIRED_CAP, caps,
            f"{_GLOBAL_CONFIG}: plugins.entries.{PLUGIN_ID}.granted_capabilities "
            f"= {caps!r}; missing '{REQUIRED_CAP}' — the global-home manager "
            f"(gateway/cron/desktop) will deny kb tool registration. "
            f"Add granted_capabilities: [tools.override] to that entry.",
        )

    def test_profile_configs_grant_tools_override(self):
        offenders = []
        for path in _PROFILE_CONFIGS:
            caps = _granted_caps(Path(path))
            if REQUIRED_CAP not in caps:
                offenders.append((path, caps))
        self.assertEqual(
            offenders, [],
            "profiles missing tools.override grant for kb: "
            + "; ".join(f"{p} caps={c}" for p, c in offenders),
        )

    def test_no_profile_config_read_error(self):
        # A config that can't be parsed YAML must not silently count as granted.
        for path in _PROFILE_CONFIGS:
            try:
                yaml.safe_load(open(path, encoding="utf-8"))
            except yaml.YAMLError as exc:
                self.fail(f"{path} is not parseable YAML: {exc}")


if __name__ == "__main__":
    unittest.main()
