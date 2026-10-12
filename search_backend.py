"""SearchBackend protocol layer for the knowledge-base retrieval service.

Backend-agnostic contract. The upper service and the Hermes native tool
(`knowledge_base_search`) depend ONLY on this protocol. Concrete backends
(InMemoryStubBackend now, PyterrierBackend later) must satisfy every method
below; the contract test suite (test_contract.py) is the acceptance gate for
any backend, including a future Pyterrier/Elasticsearch swap.

Core constraints (PRD §1): no deployed models, no external APIs, local only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Persistence container format version. Bump on every persisted-dict layout
# change so load() can detect drift instead of silently mis-parsing.
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1


class SearchBackendError(Exception):
    """Raised on load/persist/flush failure with a diagnostic message.

    Distinguishing these diagnostic messages from debug logging is the line
    the coding conventions drew (a load() that fails with a clear reason is
    required behavior, not a violation of the no-debug-logs rule).
    """


@dataclass(frozen=True)
class SearchResult:
    """Fixed-shape single result. Single-shot payload: content is full text,
    so a consuming agent never needs a second fetch to read the document."""
    doc_id: str
    score: float
    title: str
    content: str
    metadata: dict = field(default_factory=dict, compare=False)
    raw: Any = field(default=None, compare=False)


@runtime_checkable
class SearchBackend(Protocol):
    """The 5-method contract from PRD §7. Concrete backends delegate here."""

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return up to top_k results, best first. Single-shot: each result
        carries full content so no follow-up fetch is required."""
        ...

    def update(self, docs: list[dict]) -> None:
        """Append documents. Append semantics only — a 'modify' is expressed
        by appending a new version and soft-deleting the old one (FR-18)."""
        ...

    def persist(self) -> Path:
        """Write index state to disk atomically (temp + rename)."""
        ...

    def load(self) -> bool:
        """Load persisted state. Returns True on success; raises
        SearchBackendError with a diagnostic on missing/corrupt/mismatched
        state. Never silently mis-parses."""
        ...

    def flush(self) -> Path:
        """Periodic persistence hook (FR-12). Default backends make this a
        thin persist() so a timer can call flush() on an interval."""
        ...
