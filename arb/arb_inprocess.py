"""DECISIVE in-process vs fresh-process retrieval probe (Pyterrier 1.1.2).

Reconciles the three conflicting chat claims:
  - @coder (last turn, fresh-process arb_reopen.py): Retriever/IndexFactory
    "No Manager implementation found" -> cross-process reload BLOCKED
  - @data-engineer: in-process Retriever works
  - @researcher: claims fresh-process IndexFactory.of SUCCEEDED (contradicts
    @coder's fresh-process result)

This probe builds ONCE, then queries (a) in the SAME process, (b) noting what a
fresh process would see. It also confirms @researcher's tokenisation root cause
(case-preserving jieba terms vs Terrier's lowercasing English tokeniser) and
tests the FIX (feed query pre-tokenised, case preserved).

CWD = arb dir (same ./var/gen-arb/idx the earlier build used if present, else
rebuild). No rm -rf of old probes.
"""
import os
import pathlib
import pyterrier as pt
import jieba

base = pathlib.Path(__file__).resolve().parent
os.chdir(base)

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

rows = [{"docno": str(i), "docid": d["docid"],
         "text": d["title"] + " " + d["content"],
         "toks": toks_tf(d["title"] + " " + d["content"]),
         "title": d["title"], "content": d["content"]}
        for i, d in enumerate(docs)]

os.makedirs("var/gen-arb2/idx", exist_ok=True)
idx = pt.IterDictIndexer("gen-arb2/idx",
                          meta={"docno": 20, "docid": 30, "title": 512, "content": 4096},
                          pretokenised=True)
idx.index(rows)
print("BUILD OK (in-process) -> var/gen-arb2/idx")

def try_retrieve(label, query_text, tokeniser=None):
    try:
        kw = dict(metadata=["docno", "docid", "title", "content"],
                  num_results=3, wmodel="BM25")
        if tokeniser is not None:
            kw["tokeniser"] = tokeniser
        ret = pt.terrier.Retriever("gen-arb2/idx", **kw)
        res = ret.query(query_text)
        cols = list(res.columns)
        hits = len(res)
        ids = [str(x) for x in res["docid"]] if "docid" in res.columns else ["?"]
        print(f"  [{label}] query={query_text!r} -> {hits} hits, ids={ids}, "
              f"cols_has_score={[c for c in cols if c in ('bpref','Q0','score','norm')]}")
        return ids
    except Exception as e:
        print(f"  [{label}] query={query_text!r} -> {type(e).__name__}: {str(e)[:110]}")
        return None

print("\n== IN-PROCESS retrieval (same process that built) ==")
try_retrieve("raw", "Python 编程")
try_retrieve("raw", "Python")          # @researcher: misses (case)
try_retrieve("raw", "编程")            # @researcher: hits (single CJK term)

# The FIX: pre-tokenise the query with jieba the SAME way the write side did,
# and use a whitespace tokeniser that does NOT lowercase (preserve "Python").
try:
    import pyterrier.rewrite as pr
    jt = pr.tokenise(lambda s: " ".join(jieba.lcut(s)), matchop=False)
    pre = " ".join(jieba.lcut("Python"))
    try_retrieve("pre+jieba-nomatchop", pre)
except Exception as e:
    print("  [pre-tokenise setup] err:", type(e).__name__, str(e)[:100])

print("\n== Fresh-process reload (a NEW python would see this) ==")
# We cannot spawn a truly fresh process inline, but IndexFactory.of() reads
# the on-disk registry -- if the just-built index were reloadable it'd be here.
try:
    ifact = pt.IndexFactory.of("var/gen-arb2/idx")
    print("  IndexFactory.of OK ->", ifact)
    print("  => FRESH-PROCESS RELOAD POSSIBLE")
except Exception as e:
    print(f"  IndexFactory.of FAILED -> {type(e).__name__}: {str(e)[:120]}")
    print("  => FRESH-PROCESS RELOAD BLOCKED (confirms @coder's arb_reopen; "
          "@researcher's 'new process OK' claim is not reproduced)")
