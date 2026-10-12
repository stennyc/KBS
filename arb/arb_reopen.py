"""ARBITRATION STEP ③+④: in a FRESH Python process, re-open the on-disk index
from its path (no imported objects from the build process) and run one jieba
query. Prints the result column names. If ③+④ succeed, cross-process reload
HOLDS and the in-process limitation disappears; if ③ fails, researcher's plan
A (in-process) is the only realistic path.

CWD must be the arb base dir (same as arb_build.py).
"""
import os
import pyterrier as pt
import jieba

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Run from the dir that contains ./var/gen-arb/idx
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parent)

# ③ try every plausible way to re-open a durable Terrier index from disk
candidates = [
    "gen-arb/idx",
    "var/gen-arb/idx",
    "./var/gen-arb/idx",
    str(pathlib.Path("var/gen-arb/idx").resolve()),
]
opened = None
used = None
for c in candidates:
    try:
        ret = pt.terrier.Retriever(c, metadata=["docno", "docid", "title", "content"],
                                    num_results=3, wmodel="BM25")
        # force a query so any lazy Manager error surfaces now
        ret.query(" ".join(jieba.lcut("数据管道 架构")))
        opened = ret
        used = c
        break
    except Exception as e:
        print(f"STEP3 Retriever({c!r}) FAILED -> {type(e).__name__}: {str(e)[:140]}")

if opened is None:
    # also try the IndexFactory path (researcher's cross-process route)
    try:
        idx = pt.IndexFactory.of("var/gen-arb/idx")
        print("STEP3 IndexFactory.of OK ->", idx)
        opened = pt.terrier.Retriever(idx, metadata=["docno", "docid", "title", "content"],
                                       num_results=3, wmodel="BM25")
        used = "IndexFactory"
    except Exception as e:
        print(f"STEP3 IndexFactory.of FAILED -> {type(e).__name__}: {str(e)[:160]}")

if opened is not None:
    res = opened.query(" ".join(jieba.lcut("数据管道 架构")))
    print(f"\nSTEP3 RELOAD OK via: {used}")
    print("STEP4 RESULT COLUMNS:", list(res.columns))
    print(res.to_string())
    verdict = ("CROSS-PROCESS RELOAD HOLDS -> 'restart-safe' holds, "
               "in-process limitation does NOT apply")
else:
    verdict = ("CROSS-PROCESS RELOAD BLOCKED -> researcher plan A (in-process "
               "cold layer, rebuild-on-startup) is the only realistic path")
print("\n=== VERDICT ===\n" + verdict)
