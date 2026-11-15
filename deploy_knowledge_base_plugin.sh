#!/usr/bin/env bash
# Deploy / verify the knowledge-base-search plugin (PRD §13, deploy step).
#
# Single source of truth: the plugin code lives ONLY in
# $HERMES_HOME/plugins/knowledge-base-search/ (self-contained: the Hermes
# tool/hook handlers AND the KB service layer co-located in one dir).
# The host pool entry IS that directory — no symlink, no copy. Edit it in
# place; the change takes effect on the next gateway load.
#
# Usage:
#   deploy_knowledge_base_plugin.sh --check   # read-only pre-flight
#   deploy_knowledge_base_plugin.sh --apply   # fix host entry / data dir / profiles
#
# Idempotent: re-running --apply is a no-op where things are already right.

set -euo pipefail

MODE="${1:---check}"
[[ "$MODE" == "--check" || "$MODE" == "--apply" ]] || {
  echo "usage: $0 [--check|--apply]"; exit 2; }

# 跨平台：优先 HERMES_HOME，fallback 到 $HOME/.hermes
HERMES_HOME="${HERMES_HOME:-${HOME}/.hermes}"
PLUG="$HERMES_HOME/plugins/knowledge-base-search"
HOST_POOL="$HERMES_HOME/plugins"
HOST_ENTRY="$HOST_POOL/knowledge-base-search"
KB_DATA="$HERMES_HOME/kb-data"
PROFILES=(coder orchestrator reviewer researcher product-manager data-engineer default)

fail() { echo "  [FAIL] $*"; exit 1; }
ok()   { echo "  [ ok ] $*"; }

echo "== source of truth =="
[[ -f "$PLUG/__init__.py" ]]      || fail "plugin __init__.py missing: $PLUG"
[[ -f "$PLUG/plugin.yaml" ]]      || fail "plugin.yaml missing: $PLUG"
[[ -f "$PLUG/kb_service.py" ]]    || fail "kb_service.py missing: $PLUG"
[[ -f "$PLUG/clarify_state.py" ]] || fail "clarify_state.py missing: $PLUG"
ok "plugin self-contained: $PLUG"

echo "== host entry classification =="
if [[ ! -e "$HOST_ENTRY" && ! -L "$HOST_ENTRY" ]]; then
  echo "  [info] no host entry yet (will point at $PLUG on --apply)"
elif [[ -L "$HOST_ENTRY" ]]; then
  target="$(readlink -f "$HOST_ENTRY")"
  if [[ "$target" == "$PLUG" ]]; then
    ok "symlink already points at the plugin dir (acceptable: same inode)"
  else
    echo "  [warn] symlink points elsewhere: $target (will re-link on --apply)"
  fi
elif [[ -d "$HOST_ENTRY" ]]; then
  # A real directory is the intended end-state (self-contained). If it is
  # NOT $PLUG itself, it is a stray copy — replace it with a symlink to $PLUG
  # so there is exactly one physical source of truth.
  if [[ "$(readlink -f "$HOST_ENTRY")" == "$PLUG" ]]; then
    ok "host entry is the plugin dir itself"
  else
    echo "  [warn] host entry is a stray copy (real dir, not $PLUG) — --apply will remove it and link"
  fi
else
  fail "unexpected host entry type: $HOST_ENTRY"
fi

echo "== data dir =="
if [[ -d "$KB_DATA" ]]; then
  ok "data dir exists"; ls "$KB_DATA"
else
  echo "  [info] $KB_DATA does not exist yet (mkdir -p on --apply; the runtime guard then passes)"
fi

echo "== profile injection (plugins.enabled append) =="
PYBIN="$(command -v python3)"
for p in "${PROFILES[@]}"; do
  cfg="$HERMES_HOME/profiles/$p/config.yaml"
  if [[ ! -f "$cfg" ]]; then
    fail "missing profile config: $cfg"
  fi
  state="$(PYBIN - "$cfg" <<'PYEOF'
import sys
from ruamel.yaml import YAML
cfg = sys.argv[1]
y = YAML(); data = y.load(open(cfg))
enabled = (data.get("plugins") or {}).get("enabled") or []
if "knowledge-base-search" in enabled:
    print("already-injected")
elif enabled:
    print("needs-append")
else:
    print("no-plugins-block")
PYEOF
)"
  echo "  $p: $state"
done

if [[ "$MODE" == "--check" ]]; then
  echo "== pre-flight complete (read-only; run with --apply to deploy) =="
  exit 0
fi

# ---- apply ----
echo "== applying =="
mkdir -p "$KB_DATA"
ok "data dir ready: $KB_DATA"

# Host entry: end-state is a real dir (self-contained). If a stale symlink or
# a stray copy is present, remove it and install $PLUG as the physical dir.
# Because $HOST_ENTRY is literally $PLUG's parent + name, in the self-contained
# layout $HOST_ENTRY and $PLUG are the SAME path — so the "install" is a
# no-op verify, not a move.
if [[ -L "$HOST_ENTRY" ]]; then
  target="$(readlink -f "$HOST_ENTRY" 2>/dev/null || true)"
  [[ "$target" == "$PLUG" ]] || { rm -f "$HOST_ENTRY"; ok "removed stale symlink"; }
fi
if [[ ! -d "$HOST_ENTRY" ]]; then
  fail "plugin dir missing at $HOST_ENTRY after symlink handling: $HOST_ENTRY"
fi
ok "host entry resolved: $(readlink -f "$HOST_ENTRY")"

# 7 profiles: append into existing plugins.enabled, preserving disabled/entries.
PYBIN - <<'PYEOF'
import os
from pathlib import Path
from ruamel.yaml import YAML

PROFILES = ["coder", "orchestrator", "reviewer", "researcher",
            "product-manager", "data-engineer", "default"]
y = YAML()
for p in PROFILES:
    cfg = Path(f"{os.environ.get('HERMES_HOME', os.path.expanduser('~/.hermes'))}/profiles/{p}/config.yaml")
    data = y.load(open(cfg))
    plugins = data.setdefault("plugins", {})
    enabled = plugins.setdefault("enabled", [])
    if "knowledge-base-search" not in enabled:
        enabled.append("knowledge-base-search")
        y.dump(data, open(cfg, "w"))
        print(f"  {p}: appended")
    else:
        print(f"  {p}: already present")
PYEOF

echo "== post-verify =="
ok "readlink -f → $(readlink -f "$HOST_ENTRY")"
ls "$KB_DATA"
echo "== apply complete. Next: gateway restart + new-session tool visibility check."
