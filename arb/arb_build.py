"""ARBITRATION STEP ①+②: build a clean on-disk Terrier index (all-str meta,
mkdir BEFORE index), then print whether the top-level properties file exists.

CWD must be the arb base dir. Terrier resolves index paths under ./var/, so we
build into var/gen-arb/idx relative to CWD. This isolates the RELOAD question
from any cold_layer build-order Nits: if data.properties is written, reload is
*possible*; if not, no in-process trick creates it after the fact.
"""
import os
import pathlib
import pyterrier as pt
import jieba

os.makedirs("var/gen-arb/idx", exist_ok=True)   # correct order: pre-create

docs = [
    {"docid": "alpha", "title": "数据管道架构", "content": "数据管道 批处理 流处理 Lambda"},
    {"docid": "beta",  "title": "Docker 容器",   "content": "Docker 开源 容器引擎 轻量"},
    {"docid": "gamma", "title": "Kafka 消息队列", "content": "Kafka 分布式 事件流 高吞吐"},
]

def toks_tf(t):
    tf = {}
    for w in jieba.lcut(t):
        if len(w.strip()) > 1 and not w.isspace():
            tf[w] = tf.get(w, 0) + 1
    return tf

# all-str meta values (Terrier len-validates each; an int docno raises)
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

indexer = pt.IterDictIndexer("gen-arb/idx",
                             meta={"docno": 20, "docid": 30, "title": 512, "content": 4096},
                             pretokenised=True)
indexer.index(rows)
print("STEP1 BUILD OK -> var/gen-arb/idx")

# STEP ②: does the reload-critical properties file exist?
idx_dir = pathlib.Path("var/gen-arb/idx")
print("\nSTEP2 files under var/gen-arb/idx:")
for f in sorted(idx_dir.rglob("*")):
    print("   ", f, "(file)" if f.is_file() else "(dir)")
data_props = list(idx_dir.rglob("data.properties")) + list(idx_dir.rglob("index.properties"))
print("\n   data/index.properties present:", [str(p) for p in data_props] or "NONE")
