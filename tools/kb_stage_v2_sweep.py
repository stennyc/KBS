#!/usr/bin/env python3
"""Stage-2 dry-run sweep: re-classify every PENDING staging row with the
v2 rule set (kb_ingest_rules merge_rules(None)).

PRD §4.1 step 1 (dry-run): output three groups
  keep   : hits a v2 structural cue (re-stamp kb_class to the v2 id)
  iso    : does NOT hit any v2 cue (legacy decision/walkthrough/config
           collision rows + command/question noise + config dumps)

NOISE GUARD (PRD §3.2 carry-over, owner ruling "噪声永不上判定面"):
  a row whose TITLE is a bare command/question head, or whose CONTENT is
  an async-delegation system block, is quarantined even if the v2 cue
  regex happens to hit the noise text around it — a cue hit inside a
  noise row is NOT knowledge. These guard-leg rows are marked
  `quarantine_reason` so the commit step can route them with the iso
  group (same destination, different audit tag).

Quarantine rows are never rm'd — this dry-run only PRINTS; the commit
step (separate invocation) moves iso + noise-guard rows to
staging/quarantine/ (append-only jsonl, PRD R4).

Read-only: does not write anything except its own --out jsonl report.

Run: PYTHONPATH=/usr/local/lib/hermes-agent python3 tools/kb_stage_v2_sweep.py --dry-run
     PYTHONPATH=/usr/local/lib/hermes-agent python3 tools/kb_stage_v2_sweep.py --commit
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kb_ingest_rules import (  # noqa: E402
    merge_rules, CandidateDetector, is_noise_title, is_meta_instruction,
)

_DEFAULT_PY = os.environ.get("KB_PY_BIN", sys.executable)
DEFAULT_DATA_DIR = os.environ.get("KB_DATA_DIR", os.path.join(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")), "kb-data"))
_ASYNC_DELEGATION_RE = re.compile(r"\[ASYNC DELEGATION")


def _is_noise_guard(d: dict) -> str:
    """Return a quarantine-reason tag if the row is NOISE regardless of
    any v2 cue hit; empty string otherwise. PRD §3.2 + owner ruling
    "噪声行直接丢弃，留着审计". v2.1 (2026-10-14): adds the meta-
    instruction leg — a row whose TITLE is a user command/question about
    the task itself ("落盘计划prd", "开始吧", "这个怎么用？") is NOT
    knowledge even if the anti-pattern cue's 踩坑/坑 substring fired on
    the surrounding content."""
    title = str(d.get("title", "")).strip()
    content = str(d.get("content", ""))
    if is_noise_title(title):
        return "noise_guard: bare command/question head title"
    if is_meta_instruction(title):
        return "noise_guard: meta-instruction / task-scoped question title"
    if _ASYNC_DELEGATION_RE.search(title) or content.lstrip().startswith("[ASYNC DELEGATION"):
        return "noise_guard: async-delegation system block"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--out", default=None, help="jsonl report path")
    args = ap.parse_args()
    assert args.dry_run != args.commit, "pick exactly one of --dry-run/--commit"

    data_dir = Path(args.data_dir)
    staging = data_dir / "staging" / "staging_docs.jsonl"
    det = CandidateDetector(merge_rules(None), "s0", None)

    rows = []
    with open(staging) as f:
        for line in f:
            rows.append(json.loads(line))

    # Dedup: the staging store has historical duplicate rows (same
    # title+content, different docid — a pre-uuid5 double-row bug).
    # Keep the FIRST row per content-hash; mark later dupes iso with a
    # "dedupe" quarantine reason so the commit step routes them to
    # staging/quarantine/ (never rm'd, PRD R4).
    import hashlib
    seen_hash: dict = {}
    for d in rows:
        h = hashlib.md5(
            f"{str(d.get('title','')).strip()}\n{str(d.get('content',''))}".encode()
        ).hexdigest()
        d["_content_hash"] = h
        if h in seen_hash:
            d["_dedupe_of"] = seen_hash[h]
        else:
            seen_hash[h] = d.get("docid")

    keep, iso = [], []
    for d in rows:
        if d.get("_dedupe_of"):
            d["_sweep_verdict"] = "iso"
            d["_quarantine_reason"] = f"dedupe: duplicate of {d['_dedupe_of']}"
            d["_sweep_new_kb_class"] = None
            iso.append(d)
            continue
        text = f"{d.get('title', '')}\n{d.get('content', '')}"
        cand = det.scan(text, lambda a, b: "x")
        guard = _is_noise_guard(d)
        if cand and not guard:
            d["_sweep_verdict"] = "keep"
            d["_quarantine_reason"] = ""
        elif cand and guard:
            # v2 cue fired ON the noise text itself — route to quarantine
            # (a cue hit inside a noise row is NOT knowledge, PRD §3.2).
            d["_sweep_verdict"] = "iso"
            d["_quarantine_reason"] = guard + " (cue hit suppressed)"
        else:
            d["_sweep_verdict"] = "iso"
            d["_quarantine_reason"] = "no_v2_cue" if not cand else guard
        d["_sweep_new_kb_class"] = cand.classified_type if cand and not guard else None
        (keep if d["_sweep_verdict"] == "keep" else iso).append(d)

    report = {
        "total": len(rows),
        "keep": len(keep),
        "iso": len(iso),
        "iso_by_reason": dict(Counter(
            (d.get("_quarantine_reason") or "no_v2_cue").split(":")[0]
            for d in iso)),
        "keep_by_old_kb_class": dict(Counter(d.get("kb_class") for d in keep)),
        "iso_by_old_kb_class": dict(Counter(d.get("kb_class") for d in iso)),
        "short_title_rows_total": sum(1 for d in rows if len(str(d.get("title", ""))) <= 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_path = Path(args.out or data_dir / "staging" / "sweep_report.jsonl")
    with open(out_path, "w") as f:
        for d in rows:
            f.write(json.dumps({
                "docid": d.get("docid"),
                "old_kb_class": d.get("kb_class"),
                "new_kb_class": d.get("_sweep_new_kb_class"),
                "verdict": d.get("_sweep_verdict"),
                "quarantine_reason": d.get("_quarantine_reason", ""),
                "title": str(d.get("title", ""))[:80],
            }, ensure_ascii=False) + "\n")
    print(f"report -> {out_path}")

    if not args.commit:
        print("\n--- iso sample (12, with quarantine reason) ---")
        for d in iso[:12]:
            print(f"  iso [{d.get('kb_class')}] {d.get('_quarantine_reason','')[:36]:36s} "
                  f"{str(d.get('title',''))[:36]!r}")
        print("\n--- keep sample (8, old -> new) ---")
        for d in keep[:8]:
            print(f"  keep [{d.get('kb_class')} -> {d.get('_sweep_new_kb_class')}] "
                  f"{str(d.get('title',''))[:44]!r}")
        print("\nDRY-RUN ONLY. Commit step (move iso rows to staging/quarantine/, "
              "re-stamp keep rows' kb_class) is a separate invocation after owner "
              "sign-off on this report.")
        return

    qdir = data_dir / "staging" / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    qpath = qdir / f"v2_sweep_{ts}.jsonl"
    with open(qpath, "a") as qf:
        for d in iso:
            rec = dict(d)
            rec["quarantined_at"] = ts
            rec["quarantine_reason"] = d.get("_quarantine_reason") or "v2_sweep: no structural cue hit"
            qf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp = data_dir / "staging" / "staging_docs.jsonl.tmp"
    with open(tmp, "w") as f:
        for d in keep:
            d["kb_class"] = d.get("_sweep_new_kb_class")
            d["task_type"] = d.get("_sweep_new_kb_class")
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    import os
    os.replace(tmp, staging)
    # CRITICAL (owner-confirmed 2026-10-15 rollback incident): the staging
    # store's out-of-band recovery branch (staging.py StagingStore._load)
    # rewrites the ROWS file from a surviving index pickle snapshot when
    # the rows file is out-of-band changed.  Our commit just rewrote the
    # rows file out-of-band, so the stale index pickle (still holding the
    # pre-sweep doc set) would silently resurrect every iso row on the
    # next service start.  Removing the stale index pickle forces a
    # clean index rebuild from the new 7-row file on next load.
    idx_path = data_dir / "staging" / "staging_index.pkl"
    if idx_path.exists():
        idx_path.unlink()
        print(f"COMMIT: removed stale index {idx_path.name} (forces clean rebuild)")
    # CRITICAL 2nd leg (owner-confirmed 2026-10-15 recurring rollback):
    # deleting the index pkl only covers the cold-start _load recovery
    # branch.  A LIVE gateway process still holds the pre-sweep doc set in
    # its in-memory StagingStore; any subsequent persist() (ingest/confirm
    # rebase path) union-merges disk rows back into that memory and writes
    # the quarantined rows straight back to disk.  The marker file below
    # tells the next in-process rebase (kb_service._consume_external_replace_
    # marker) to ADOPT the on-disk rows wholesale instead of union-merging,
    # so the live memory converges to the swept set on its very next
    # write cycle.  This is the only cross-process sync channel that
    # exists between this tool and the gateway's _service singleton.
    marker = data_dir / "staging" / "staging_external_replace"
    marker.write_text(
        f"v2_sweep commit {ts}: {len(keep)} keep / {len(iso)} iso "
        f"-> quarantine/v2_sweep_{ts}.jsonl\n", encoding="utf-8")
    print(f"COMMIT: marker {marker.name} written (live-service sync signal)")
    print(f"COMMIT: {len(iso)} iso rows -> {qpath}")
    print(f"COMMIT: {len(keep)} keep rows re-stamped in staging_docs.jsonl")
    print("NOTE: live gateway service will adopt the swept set on its next "
          "ingest()/confirm() rebase (marker consumed then).  Re-verify row "
          "count AFTER that cycle — do not treat the commit moment as the "
          "durable state.")


if __name__ == "__main__":
    main()
