"""conftest.py for the tests/ directory.

Keeps the root-import face intact after the test scripts moved from the repo
root into tests/. Before the move, each test script lived at the repo root and
its own top-of-file sys.path hack (`Path(__file__).resolve().parent`) pointed
straight at the directory holding kb_service / hot_layer / two_layer / staging
/ etc. After the move into tests/, `__file__`'s parent is tests/ (which has no
importable modules), so that one-level reference would break.

This conftest is loaded by pytest (and, when run bare, by the root-level
`python -m pytest tests/...` invocations) and splices the REPO ROOT — the
directory that actually contains the importable modules — onto sys.path. It is
the canonical fix: any test that does a bare `from kb_service import ...` now
resolves regardless of its location.

The individual top-of-file sys.path hacks inside each test script are still
patched to point one level up (parent -> parent.parent), so the scripts also
work when executed directly (python tests/test_x.py) rather than only under
pytest. conftest + patched hacks = belt and suspenders; neither is redundant.
"""
import os
import sys

# repo root = the directory that holds kb_service.py and the rest of the
# importable modules (one level above this tests/ directory).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
