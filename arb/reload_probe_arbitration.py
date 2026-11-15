"""裁决脚本 v3: 与 cold_layer.py 逐字一致的构建方式，4 条探针输出."""
"""Commit this OID so both @coder and @researcher sign the same machine output."""
import subprocess, sys, tempfile, os
from pathlib import Path

_hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
_WORKDIR = _hermes_home  # Use HERMES_HOME as the working directory root

print("=" * 60)
print("PYTERRIER 1.1.2 RELOAD PROBE (仲裁脚本)")
print("=" * 60)

with tempfile.TemporaryDirectory() as d:
    idx_dir = Path(d) / "idx.terrier"
    # Build: exact same pattern as cold_layer.py
    build_py = Path(d) / "build.py"
    build_py.write_text(f'''
import pyterrier as pt
from pathlib import Path
pt.java.init()
idx_dir = Path("{idx_dir}")
indexer = pt.IterDictIndexer(
    str(idx_dir), pretokenised=True,
    meta={{"docno": 20, "docid": 20, "title": 512, "content": 4096}},
)
indexer.index(iter([{{
    "docno": "d1", "docid": "external-001",
    "title": "Python编程", "content": "Python是一种编程语言",
    "toks": {{"Python": 1, "编程": 1, "语言": 1}},
}}]))
print("BUILD_OK")
''')
    r_build = subprocess.run([sys.executable, str(build_py)], capture_output=True, text=True, cwd=_WORKDIR)
    print(f"\n[Build] {r_build.stdout.strip()}")
    if r_build.stderr:
        print(f"  (stderr: {r_build.stderr[:200]})")

    # Probe 1: top-level data.properties exists?
    top_props = list(idx_dir.glob("data*.properties")) + list(idx_dir.glob("index*.properties"))
    stream_props = list(idx_dir.glob("data_stream*.properties"))
    print(f"\n[Probe 1] Top-level properties: {[p.name for p in top_props]}")
    print(f"           Stream-level:        {[p.name for p in stream_props]}")
    P1 = "HAS_TOP_LEVEL_PROPERTIES" if top_props else "NO_TOP_LEVEL_PROPERTIES"

    # Probe 2: in-process Retriever with path string
    probe2_py = Path(d) / "p2.py"
    probe2_py.write_text(f'''
import os; os.chdir("{_WORKDIR}")
import pyterrier as pt
pt.java.init()
from pyterrier.terrier import Retriever
import pandas as pd
try:
    ret = Retriever("{idx_dir}", metadata=["docid","title","content"], wmodel="BM25")
    r = ret.transform(pd.DataFrame([{{"qid":"q1","query":"Python 编程"}}]))
    print(f"OK {{len(r)}} results")
except Exception as e:
    print(f"FAIL: {{type(e).__name__}}: {{str(e)[:80]}}")
''')
    r2 = subprocess.run([sys.executable, str(probe2_py)], capture_output=True, text=True, cwd=_WORKDIR)
    print(f"\n[Probe 2] In-process Retriever(path): {r2.stdout.strip()}")
    P2 = "IN_PROCESS_OK" if "OK" in r2.stdout else "IN_PROCESS_FAIL"
    
    # Probe 3: cross-process IndexFactory.of()
    probe3_py = Path(d) / "p3.py"
    probe3_py.write_text(f'''
    import os; os.chdir("{_WORKDIR}")
    import pyterrier as pt
    pt.java.init()
    try:
        idx = pt.IndexFactory.of("{idx_dir}")
        print(f"OK docs={{idx.getCollectionStatistics().getNumberOfDocuments()}}")
    except Exception as e:
        print(f"FAIL: {{type(e).__name__}}: {{str(e)[:80]}}")
    ''')
    r3 = subprocess.run([sys.executable, str(probe3_py)], capture_output=True, text=True, cwd=_WORKDIR)
    print(f"\n[Probe 3] Cross-process IndexFactory: {r3.stdout.strip()}")
    P3 = "CROSS_PROCESS_OK" if "OK" in r3.stdout else "CROSS_PROCESS_FAIL"
    
    # Probe 4: pyterrier.rewrite vs pyterrier.terrier.rewrite
    probe4_py = Path(d) / "p4.py"
    probe4_py.write_text('''
import sys
for mod in ("pyterrier.rewrite", "pyterrier.terrier.rewrite"):
    try:
        __import__(mod)
        print(f"OK: {mod}")
    except ImportError:
        print(f"MISS: {mod}")
''')
    r4 = subprocess.run([sys.executable, str(probe4_py)], capture_output=True, text=True)
    print(f"\n[Probe 4] Module existence:\n{r4.stdout.strip()}")
    P4 = "REWRITE_EXISTS" if "OK" in r4.stdout else "REWRITE_MISSING"
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Probe 1 (top-level props):  {P1}")
    print(f"  Probe 2 (in-process query): {P2}")
    print(f"  Probe 3 (cross-process):    {P3}")
    print(f"  Probe 4 (rewrite module):   {P4}")
    print("=" * 60)
