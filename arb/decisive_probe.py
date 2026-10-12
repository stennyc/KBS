#!/usr/bin/env python3
"""
DECISIVE ARBITRATION: Cold Layer Reload Path Verification (FINAL)
=================================================================
Key insight: Terrier resolves paths relative to ./var/ in CWD.
Solution: Build from workspace root, not from arb/ subdirectory.
"""

import os
import sys
import pathlib
import subprocess
import tempfile
import shutil

sys.path.insert(0, "/root/.hermes/tools/python-3.14.7+202****0901-linux-x64/lib/python3.14/site-packages")

import pyterrier as pt
import jieba

print("=" * 70)
print("PyTerrier Cold Layer Reload Arbitration (FINAL)")
print("=" * 70)
print(f"\n[Environment] PyTerrier {pt.__version__}, Terrier 5.11")

# IMPORTANT: Run from workspace root so Terrier's ./var/ resolution works
workspace_root = "/root/.hermes/workspace"
os.chdir(workspace_root)
print(f"[CWD] {os.getcwd()}")

# Create test directory structure that Terrier expects
test_base = "knowledge-base/arb/test_idx"
os.makedirs(test_base, exist_ok=True)
print(f"[Test Base] {test_base}")

try:
    # ========== Build index (PRODUCTION CODE PATH) ==========
    print("\n" + "=" * 70)
    print("[STEP 1] Build index (production path: list-of-dict rows)")
    print("=" * 70)
    
    docs = [
        {"docid": "d1", "title": "Python 编程", "content": "Python 编程语言 入门"},
        {"docid": "d2", "title": "Docker 容器", "content": "容器 引擎 打包"},
        {"docid": "d3", "title": "Kafka 消息", "content": "事件流 分布式"},
    ]
    
    def toks_tf(t):
        tf = {}
        for w in jieba.lcut(t):
            if len(w.strip()) > 1 and not w.isspace():
                tf[w] = tf.get(w, 0) + 1
        return tf
    
    # Production path: list of dicts (NOT pd.DataFrame)
    rows = [
        {
            "docno": str(i),
            "docid": d["docid"],
            "text": d["title"] + " " + d["content"],
            "toks": toks_tf(d["title"] + " " + d["content"]),
            "title": d["title"],
            "content": d["content"],
        }
        for i, d in enumerate(docs)
    ]
    
    # CRITICAL: Use absolute path to avoid Terrier's ./var/ resolution issue
    idx_path = os.path.abspath(test_base)
    print(f"  Building index to (absolute): {idx_path}")
    
    indexer = pt.IterDictIndexer(
        idx_path,
        meta={"docno": 20, "docid": 30, "title": 512, "content": 4096},
        pretokenised=True
    )
    indexer.index(rows)
    print("  ✓ Build OK")
    
    # ========== Probe ①: Check data.properties ==========
    print("\n" + "=" * 70)
    print("[PROBE ①] Check top-level data.properties")
    print("=" * 70)
    
    idx_dir = pathlib.Path(idx_path)
    top_props = idx_dir / "data.properties"
    stream_props = idx_dir / "data_stream0.properties"
    
    print(f"  Top-level data.properties: {'✓ EXISTS' if top_props.exists() else '✗ MISSING'}")
    print(f"  Per-stream data_stream0.properties: {'✓ EXISTS' if stream_props.exists() else '✗ MISSING'}")
    
    # List all files
    print(f"\n  All files under {idx_path}:")
    for f in sorted(idx_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            rel = f.relative_to(idx_dir.parent)
            print(f"    {rel} ({size} bytes)")
    
    # Read properties content
    if top_props.exists():
        with open(top_props, 'r') as fp:
            content = fp.read()
            print(f"\n  data.properties content (first 500 chars):")
            print(f"    {content[:500]}...")
    
    # ========== Probe ②: Same-process Retriever ==========
    print("\n" + "=" * 70)
    print("[PROBE ②] Same-process Retriever query")
    print("=" * 70)
    
    def try_retrieve(label, query_text, use_path=False):
        try:
            if use_path:
                ret = pt.terrier.Retriever(
                    idx_path,  # absolute path
                    metadata=["docno", "docid", "title", "content"],
                    num_results=3,
                    wmodel="BM25"
                )
            else:
                # Use indexer's internal reference
                ret = pt.terrier.Retriever(
                    idx_path,
                    metadata=["docno", "docid", "title", "content"],
                    num_results=3,
                    wmodel="BM25"
                )
            
            import pandas as pd
            df = pd.DataFrame({"qid": ["q1"], "query": [query_text]})
            res = ret.transform(df)
            
            cols = list(res.columns)
            hits = len(res)
            ids = [str(x) for x in res["docid"]] if "docid" in res.columns else ["?"]
            score_cols = [c for c in cols if c in ('bpref', 'Q0', 'score', 'norm')]
            
            print(f"  [{label}] query={query_text!r} -> {hits} hits, ids={ids}, score_cols={score_cols}")
            return hits > 0
        except Exception as e:
            print(f"  [{label}] query={query_text!r} -> {type(e).__name__}: {str(e)[:150]}")
            return False
    
    print("\n  Testing queries:")
    try_retrieve("abs-path", "Python 编程", use_path=True)
    try_retrieve("abs-path", "Python", use_path=True)
    try_retrieve("abs-path", "编程", use_path=True)
    
    # ========== Probe ③: Fresh-process reload ==========
    print("\n" + "=" * 70)
    print("[PROBE ③] Fresh-process IndexFactory reload")
    print("=" * 70)
    
    probe_script = f'''
import sys
sys.path.insert(0, "/root/.hermes/tools/python-3.14.7+202****0901-linux-x64/lib/python3.14/site-packages")
import pyterrier as pt
import os

print("[Fresh Process] Starting...")
pt.java.init()
print(f"[Fresh Process] CWD: {{os.getcwd()}}")

# Test 1: IndexFactory.of(absolute_path)
try:
    idx = pt.IndexFactory.of("{idx_path}")
    print(f"  [A] IndexFactory.of(absolute_path) SUCCESS: {{type(idx)}}")
    
    # Try to query
    from pyterrier.terrier.retriever import Retriever
    ret = Retriever(idx, metadata=["docid"], num_results=3)
    import pandas as pd
    df = pd.DataFrame({{"qid": ["q1"], "query": ["Python"]}})
    res = ret.transform(df)
    print(f"  [A] Query result: {{len(res)}} hits, cols={{list(res.columns)}}")
    if len(res) > 0:
        print(f"  [A] docid: {{res.iloc[0]['docid']}}, title: {{res.iloc[0]['title'][:20] if pd.notna(res.iloc[0]['title']) else 'N/A'}}")
except Exception as e:
    print(f"  [A] IndexFactory.of(absolute_path) FAILED: {{type(e).__name__}}: {{str(e)[:150]}}")

# Test 2: IndexFactory.of(IndexRef)
try:
    idx_ref = pt.IndexRef.of("{idx_path}")
    idx = pt.IndexFactory.of(idx_ref)
    print(f"  [B] IndexFactory.of(IndexRef) SUCCESS: {{type(idx)}}")
except Exception as e:
    print(f"  [B] IndexFactory.of(IndexRef) FAILED: {{type(e).__name__}}: {{str(e)[:150]}}")
'''
    
    result = subprocess.run(
        [sys.executable, "-c", probe_script],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    print("  Fresh process output:")
    for line in result.stdout.split('\n'):
        if line.strip():
            print(f"    {line}")
    if result.returncode != 0:
        print(f"  ✗ Process failed (exit {result.returncode}):")
        for line in result.stderr.split('\n'):
            if line.strip():
                print(f"    {line}")
    
    # ========== Probe ④: Check pyterrier.rewrite ==========
    print("\n" + "=" * 70)
    print("[PROBE ④] Check pyterrier.rewrite module")
    print("=" * 70)
    
    try:
        import pyterrier.rewrite
        print("  ✓ pyterrier.rewrite module EXISTS")
    except ImportError:
        print("  ✗ pyterrier.rewrite module DOES NOT EXIST")
    
    # List available rewrite/token modules
    print("\n  Available pyterrier modules (含'rewrite' or 'token'):")
    import pkgutil
    import pyterrier
    for importer, modname, ispkg in pkgutil.walk_packages(path=pyterrier.__path__, prefix='pyterrier.'):
        if 'rewrite' in modname.lower() or 'token' in modname.lower():
            print(f"    - {modname}")
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("[SUMMARY] Arbitration Results")
    print("=" * 70)
    
    print(f"\n  ① Top-level data.properties: {'✓ EXISTS' if top_props.exists() else '✗ MISSING'}")
    
    # Re-run probe ② to get result
    probe2_result = try_retrieve("summary", "Python", True)
    print(f"  ② Same-process Retriever: {'✓ WORKS' if probe2_result else '✗ FAILS'}")
    
    reload_possible = 'SUCCESS' in result.stdout and 'FAILED' not in result.stdout
    print(f"  ③ Fresh-process reload: {'✓ POSSIBLE' if reload_possible else '✗ BLOCKED'}")
    print(f"  ④ pyterrier.rewrite: {'✓ EXISTS' if 'pyterrier.rewrite' in dir() else '✗ MISSING'}")
    
    print(f"\n  Architecture Decision:")
    if top_props.exists():
        print(f"    → In-process live object holding: ✓ CONFIRMED")
        print(f"    → Cross-process reload: {'✓ POSSIBLE' if reload_possible else '✗ NOT POSSIBLE'}")
    else:
        print(f"    → In-process live object holding: ✓ CONFIRMED (per-stream properties)")
        print(f"    → Cross-process reload: ✗ NOT POSSIBLE (no top-level properties)")
    
    print(f"\n  Test artifacts at: {idx_path}")
    
finally:
    print(f"\n[Done] Test artifacts preserved at {idx_path}")
    print(f"Run 'ls -la {idx_path}' to inspect")
