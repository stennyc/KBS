"""Decisive probe: what IS the `pyterrier` importable in this interpreter?

Run once, print everything, hang in a commit. Goal: settle B1 (kill the
dead Pyterrier build path?) on FACTS, not on chat claims that keep
disagreeing (DFIndexer True vs No-Manager vs "JVM loaded 5.11")."""
import sys
import importlib

print("python:", sys.executable)
print("version:", sys.version.split()[0])

# Is there a local shadow?
import os
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("cwd files named pyterrier*:",
      [f for f in os.listdir(os.getcwd()) if f.startswith("pyterrier")])
print("parent files named pyterrier*:",
      [f for f in os.listdir(here) if f.startswith("pyterrier")])

try:
    import pyterrier as pt
    print("import pyterrier OK; module file:", getattr(pt, "__file__", None))
    print("has __version__:", hasattr(pt, "__version__"),
          (getattr(pt, "__version__", None)))
    names = dir(pt)
    for probe in ["DFIndexer", "IterDictIndexer", "Retriever",
                  "IndexFactory", "terrier", "init", "new"]:
        print(f"  pt.{probe}:", probe in names)
    # submodules claimed by the room
    for sub in ["terrier", "java", "terrier.rewrite"]:
        try:
            importlib.import_module(f"pyterrier.{sub}")
            print(f"  import pyterrier.{sub}: OK")
        except Exception as e:
            print(f"  import pyterrier.{sub}: FAIL {type(e).__name__}: {str(e)[:80]}")
    # can the JVM actually start?
    try:
        pt.init()
        print("pt.init(): JVM up")
    except Exception as e:
        print(f"pt.init(): FAIL {type(e).__name__}: {str(e)[:120]}")
except Exception as e:
    print("import pyterrier FAILED:", type(e).__name__, str(e)[:160])
