"""Locate a Python that can import rank_bm25 + jieba, or prove none exists.

Earlier this session `python3 -m unittest test_contract` ran 15/15 green, so
the deps existed in SOME interpreter. The default `python3` no longer has
them. Enumerate every Hermes-managed interpreter + any rank_bm25 on disk and
report, so we don't fabricate a green run."""
import glob
import importlib.util
import os
import subprocess
import sys
import shutil

print("== which python3 on PATH ==")
print(" ", shutil.which("python3"))
print(" __version__:", sys.version.split()[0])

# A: check the CURRENT interpreter
def has(mod):
    return importlib.util.find_spec(mod) is not None

print("\n== current interpreter has? ==")
for m in ("rank_bm25", "jieba", "pyterrier"):
    print(f"  {m}: {'YES' if has(m) else 'NO'}")

# B: find all python binaries under the hermes tools/installs trees
cands = set()
for pat in ("/root/.hermes/tools/*/bin/python*",
           "/root/.hermes/installs/*/environments/*/venv/bin/python*",
           "/usr/bin/python3*", "/usr/local/bin/python3*"):
    cands.update(glob.glob(pat))

print("\n== rank_bm25 / jieba importable per interpreter ==")
checked = 0
for py in sorted(cands):
    out = subprocess.run(
        [py, "-c",
         "import importlib.util as u;"
         "print((u.find_spec('rank_bm25') is not None),"
         "(u.find_spec('jieba') is not None))"],
        capture_output=True, text=True, timeout=30,
    )
    line = out.stdout.strip()
    if line:
        print(f"  {py} -> rank_bm25/jieba = {line}")
        checked += 1
print("  (interpreters with a parsable answer:", checked, ")")

# C: any rank_bm25 package directory anywhere on disk (bounded, common roots)
print("\n== rank_bm25 on disk ==")
found = 0
site_roots = []
for py in sorted(cands):
    out = subprocess.run(
        [py, "-c", "import site,sys; print('\n'.join(site.getsitepackages()+[site.getusersitepackages()])); print('---SEP---'.join([p for p in sys.path if 'site-packages' in p] or ['NONE']))"],
        capture_output=True, text=True, timeout=30)
    site_roots.extend(l for l in out.stdout.splitlines() if "site-packages" in l and l != "NONE")
for root in site_roots:
    if not os.path.isdir(root):
        continue
    sp = os.path.join(root, "rank_bm25")
    if os.path.isdir(sp):
        print("  FOUND:", sp)
        found += 1
    jp = os.path.join(root, "jieba")
    if os.path.isdir(jp):
        print("  jieba at:", jp)
if found == 0:
    print(f"  no rank_bm25 under {len(site_roots)} site-packages roots")
