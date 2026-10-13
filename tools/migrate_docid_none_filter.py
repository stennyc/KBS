#!/usr/bin/env python3
"""B1 None-filter docid migration script (PRD v2-final Phase A).

Standalone, runs against a kb-data directory. Does NOT import the live
KB service or touch any running process — pure file I/O.

Forward migration (P0-1a invariant: the mapping is persisted BEFORE any
data mutation, so a crash at any data-write point leaves a complete map
from which rollback can restore the old docids):
  1. Takes a full kb-data backup (cp -a to kb-data/../kb-data.bak-<ts>/);
     aborts without touching data if the backup fails.
  2. Reads hot/cold pickle side-stores + staging JSONL.
  3. For each row: recomputes docid under the NEW None-filter formula
     (key=None treated as key-absent, matching _deterministic_docid
     after the B1 patch), collecting the old_docid → new_docid mapping
     WITHOUT touching the stores yet.
  4. Dumps the complete mapping to kb-data/docid_migration_map.json
     FIRST — before any data file is rewritten.
  5. Only then rewrites hot/cold pickle stores + staging JSONL with
     the new docids, and re-aligns the ingest_ts.jsonl sidecar.
  6. P0-2: also remaps every docid key inside graph.json (node keys
     + neighbor keys) and every per-row docid inside reinforce.jsonl
     (append-only ledger: the historical ts/reason/anchors are kept
     verbatim, ONLY the docid field is rewritten old→new). Without
     this, the 58 historical explicit_search reinforce rows and all
     cross-doc adjacency in graph.json would become dead references
     after the corpus docids were remapped.
  7. Idempotent: a second forward run finds no changes.

Rollback:
  1. Reads docid_migration_map.json.
  2. Restores the old docids in hot/cold + staging.
  3. Strips the new keys (project_ref / gov_tier) from those rows.
  4. Re-aligns ingest_ts.jsonl back to the old docids.
  5. P0-2: remaps graph.json node keys + neighbor keys, and every
     reinforce.jsonl row's docid, back to their pre-migration values
     (using the reverse mapping; the rest of each row is untouched).
  6. Removes the map file.

Usage:
  python tools/migrate_docid_none_filter.py <kb-data-dir> [--rollback]
"""
import argparse
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def _deterministic_docid(doc: dict) -> str:
    """The NEW None-filter formula (matches kb_service._deterministic_docid
    after the B1 patch).

    Excludes the docid key itself AND every None-valued key from the
    uuid5 payload — the semantic ruling "key=None ≡ key absent".
    """
    import uuid
    payload = json.dumps(
        {k: v for k, v in sorted(doc.items())
         if k != "docid" and v is not None},
        ensure_ascii=False, default=str,
    )
    return str(uuid.uuid5(uuid.NAMESPACE_OID, payload))


# ---------------------------------------------------------------------------
# Store readers / writers
# ---------------------------------------------------------------------------

def _read_pkl_docs(path: Path) -> tuple[list, int, int]:
    """Return (docs, schema_version, generation) from a pkl side-store."""
    if not path.exists():
        return [], 0, 0
    with open(path, "rb") as f:
        data = pickle.load(f)
    return (data.get("docs", []),
            data.get("schema_version", 0),
            data.get("generation", 0))


def _write_pkl_docs(path: Path, docs: list, schema_version: int,
                    generation: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".pkl.tmp")
    try:
        with open(fd, "wb") as f:
            pickle.dump({"schema_version": schema_version,
                         "generation": generation,
                         "docs": docs}, f)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_jsonl_rows(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl_rows(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".jsonl.tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _realign_ingest_ts(path: Path, mapping: Dict[str, str]) -> None:
    """Rewrite ingest_ts.jsonl: replace every old docid with its new docid
    from the mapping. Rows whose docid is not in the mapping are kept
    unchanged."""
    if not path.exists():
        return
    rows = _read_jsonl_rows(path)
    new_rows = []
    for row in rows:
        old_id = str(row.get("docid", ""))
        if old_id in mapping:
            row = dict(row)
            row["docid"] = mapping[old_id]
        new_rows.append(row)
    _write_jsonl_rows(path, new_rows)


def _remap_graph(path: Path, mapping: Dict[str, str]) -> None:
    """P0-2: rewrite every docid key inside graph.json (both the node
    keys and the per-node neighbor keys) using the old→new mapping.
    Values (float weights) and the {"max_docs", "graph"} package
    structure are preserved. docid keys not present in the mapping are
    kept unchanged (e.g. already-remapped or externally-issued ids).
    No-op if the file does not exist."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        pkg = json.load(f)
    graph = pkg.get("graph")
    if not isinstance(graph, dict):
        return
    new_graph: Dict[str, Dict[str, float]] = {}
    for node_id, neighbors in graph.items():
        new_node = mapping.get(str(node_id), str(node_id))
        new_neighbors = {}
        for nbr_id, weight in (neighbors or {}).items():
            new_nbr = mapping.get(str(nbr_id), str(nbr_id))
            new_neighbors[new_nbr] = weight
        new_graph[new_node] = new_neighbors
    pkg["graph"] = new_graph
    _atomic_write_json(path, pkg)


def _remap_reinforce(path: Path, mapping: Dict[str, str]) -> None:
    """P0-2: rewrite ONLY the per-row docid field of reinforce.jsonl
    (append-only ledger) using the old→new mapping. The historical
    ts / reason / anchors of every row are preserved verbatim — only
    the docid is remapped so historical explicit_search references do
    not become dead links after the corpus docids change. No-op if the
    file does not exist."""
    if not path.exists():
        return
    rows = _read_jsonl_rows(path)
    new_rows = []
    for row in rows:
        old_id = str(row.get("docid", ""))
        if old_id in mapping:
            row = dict(row)
            row["docid"] = mapping[old_id]
        new_rows.append(row)
    _write_jsonl_rows(path, new_rows)


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Atomically write a JSON document (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Forward migration
# ---------------------------------------------------------------------------

def _backup_kb_data(kb_data: Path) -> Path:
    """P0-1a backup step: copy the entire kb-data tree (cp -a semantics:
    preserved mtime/permissions/ownership) to kb-data/../kb-data.bak-<ts>/
    BEFORE any forward data mutation. The caller MUST treat a raised
    exception as fatal: the forward migration aborts without touching
    any data file. Returns the backup directory."""
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = kb_data.parent / f"kb-data.bak-{ts}"
    def _copy_tree(src: Path, dst: Path) -> None:
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            stat = src.stat()
            # cp -a semantics: preserve mtime / mode / ownership.
            dst.chmod(stat.st_mode)
            try:
                os.chown(dst, stat.st_uid, stat.st_gid)
            except (PermissionError, OSError):
                pass  # best-effort; not running as root on dev boxes
            for entry in src.iterdir():
                _copy_tree(entry, dst / entry.name)
        elif src.is_file():
            shutil.copy2(src, dst)
            stat = src.stat()
            dst.chmod(stat.st_mode)
            try:
                os.chown(dst, stat.st_uid, stat.st_gid)
            except (PermissionError, OSError):
                pass
    _copy_tree(kb_data, bak)
    return bak


def _remap_rows(rows: list, mapping: Dict[str, str],
                strip_keys: set = frozenset()) -> tuple[list, bool]:
    """Recompute docids for rows in-place (mutates the row dicts' docid).
    Returns (rows, changed_anything)."""
    changed = False
    for row in rows:
        old_id = str(row.get("docid", ""))
        if not old_id:
            continue
        # Recompute under the new formula.
        new_id = _deterministic_docid(row)
        if old_id != new_id:
            row["docid"] = new_id
            mapping[old_id] = new_id
            changed = True
        if strip_keys:
            for k in strip_keys:
                row.pop(k, None)
    return rows, changed


def _dump_mapping(map_path: Path, mapping: Dict[str, str]) -> None:
    """Persist the complete old→new docid mapping to disk atomically
    (P0-1a: this MUST happen before any data mutation so that a crash
    at any data-write point still leaves a complete map for rollback)."""
    kb_data = map_path.parent
    kb_data.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(kb_data), suffix=".json.tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        Path(tmp).replace(map_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def forward_migrate(kb_data: Path) -> int:
    """Run the forward B1 None-filter docid migration. Returns the
    number of remapped docids.

    P0-1a crash-window invariant: the mapping is computed and persisted
    to disk BEFORE any data file is rewritten. A crash at any data-write
    point below leaves a complete docid_migration_map.json from which
    rollback can restore every old docid.
    """
    hot_pkl = kb_data / "hot_corpus.pkl"
    cold_pkl = kb_data / "cold" / "cold_corpus.pkl"
    staging_jsonl = kb_data / "staging" / "staging_docs.jsonl"
    ingest_ts = kb_data / "ingest_ts.jsonl"
    map_path = kb_data / "docid_migration_map.json"

    # 0. Full kb-data backup BEFORE touching any data. A failed backup
    #    raises and aborts the forward migration without writing anything.
    backup_dir = _backup_kb_data(kb_data)

    mapping: Dict[str, str] = {}

    # 1-3. READ + recompute all stores in memory; collect the complete
    # mapping WITHOUT mutating the stores on disk yet.
    hot_docs, hot_sv, hot_gen = _read_pkl_docs(hot_pkl)
    _remap_rows(hot_docs, mapping)

    cold_docs, cold_sv, cold_gen = _read_pkl_docs(cold_pkl)
    _remap_rows(cold_docs, mapping)

    staging_rows = _read_jsonl_rows(staging_jsonl)
    _remap_rows(staging_rows, mapping)

    # 4. Dump the complete mapping to disk FIRST (P0-1a: mapping MUST
    # exist before any data mutation).
    _dump_mapping(map_path, mapping)

    # P0-2: remap the derived structures that hold docid keys:
    #   - graph.json (cross-doc adjacency: both node keys and
    #     per-node neighbor keys are docids)
    #   - reinforce.jsonl (append-only ledger: rewrite ONLY the
    #     per-row docid; historical ts/reason/anchors stay verbatim)
    # These must use the complete mapping collected above, and are
    # written AFTER the map is persisted but as part of the same
    # forward data-mutation window (any crash here is still fully
    # recoverable from docid_migration_map.json).
    graph_json = kb_data / "graph.json"
    reinforce_jsonl = kb_data / "reinforce.jsonl"
    _remap_graph(graph_json, mapping)
    _remap_reinforce(reinforce_jsonl, mapping)

    # 5. Only now that the mapping is safely on disk, rewrite the data
    # files with the new docids.
    _write_pkl_docs(hot_pkl, hot_docs, hot_sv, hot_gen)
    _write_pkl_docs(cold_pkl, cold_docs, cold_sv, cold_gen)
    _write_jsonl_rows(staging_jsonl, staging_rows)

    # 6. Re-align the ingest_ts sidecar.
    _realign_ingest_ts(ingest_ts, mapping)

    print(f"B1 forward migration: {len(mapping)} docid(s) remapped, "
          f"map saved to {map_path.name}, backup at {backup_dir.name}")
    return len(mapping)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def rollback(kb_data: Path) -> int:
    """Roll back the B1 forward migration: restore old docids from the
    map, strip the new keys (project_ref / gov_tier), and re-align the
    ingest_ts sidecar. Returns the number of restored docids."""
    map_path = kb_data / "docid_migration_map.json"
    if not map_path.exists():
        print("B1 rollback: no docid_migration_map.json found; nothing to do")
        return 0

    with open(map_path, "r", encoding="utf-8") as f:
        mapping: Dict[str, str] = json.load(f)

    # Reverse the mapping: new_id -> old_id.
    reverse: Dict[str, str] = {v: k for k, v in mapping.items()}

    # Strip the new Phase A keys so the rolled-back rows are byte-identical
    # to their pre-migration form.
    strip_keys = {"project_ref", "gov_tier"}

    def _restore_rows(rows: list) -> None:
        for row in rows:
            new_id = str(row.get("docid", ""))
            if new_id in reverse:
                row["docid"] = reverse[new_id]
            for k in strip_keys:
                row.pop(k, None)

    hot_pkl = kb_data / "hot_corpus.pkl"
    cold_pkl = kb_data / "cold" / "cold_corpus.pkl"
    staging_jsonl = kb_data / "staging" / "staging_docs.jsonl"
    ingest_ts = kb_data / "ingest_ts.jsonl"
    graph_json = kb_data / "graph.json"
    reinforce_jsonl = kb_data / "reinforce.jsonl"

    hot_docs, hot_sv, hot_gen = _read_pkl_docs(hot_pkl)
    _restore_rows(hot_docs)
    _write_pkl_docs(hot_pkl, hot_docs, hot_sv, hot_gen)

    cold_docs, cold_sv, cold_gen = _read_pkl_docs(cold_pkl)
    _restore_rows(cold_docs)
    _write_pkl_docs(cold_pkl, cold_docs, cold_sv, cold_gen)

    staging_rows = _read_jsonl_rows(staging_jsonl)
    _restore_rows(staging_rows)
    _write_jsonl_rows(staging_jsonl, staging_rows)

    # P0-2: restore graph.json (node + neighbor docid keys) and
    # reinforce.jsonl (per-row docid) back to their pre-migration
    # docids using the reverse mapping. No strip_keys logic here:
    # these files do not carry the new Phase A row-level keys.
    _remap_graph(graph_json, reverse)
    _remap_reinforce(reinforce_jsonl, reverse)

    # Re-align ingest_ts back to old docids (use the reverse mapping:
    # current on-disk docids are the new ones; map them back).
    _realign_ingest_ts(ingest_ts, reverse)

    # Remove the map file (the migration is rolled back).
    map_path.unlink(missing_ok=True)

    print(f"B1 rollback: {len(reverse)} docid(s) restored to pre-migration "
          f"form; {len(strip_keys)} key(s) stripped; map removed")
    return len(reverse)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B1 None-filter docid migration (forward + rollback)")
    parser.add_argument("kb_data", type=Path,
                        help="Path to the kb-data directory")
    parser.add_argument("--rollback", action="store_true",
                        help="Roll back a previously-run forward migration")
    args = parser.parse_args()

    kb_data: Path = args.kb_data
    if not kb_data.is_dir():
        print(f"error: {kb_data} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.rollback:
        rollback(kb_data)
    else:
        forward_migrate(kb_data)


if __name__ == "__main__":
    main()
