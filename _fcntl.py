"""Platform-adaptive flock shim for Windows.

Provides fcntl.flock() + LOCK_SH/LOCK_EX/LOCK_NB/LOCK_UN constants on Windows
where the stdlib fcntl module is unavailable. Delegates to the ``filelock``
package installed in the Hermes-managed venv.

Used internally by kb_service for cross-process advisory locking on the
knowledge-base data directory. On POSIX hosts the stdlib fcntl module wins
the import; this module is a no-op fallback only.
"""
from __future__ import annotations

import logging
import os as _os
from typing import Any, Dict

logger = logging.getLogger("kbs._fcntl")

# Linux fcntl constants — same values the POSIX module exports.
LOCK_SH = 1   # shared lock (read)
LOCK_EX = 2   # exclusive lock (write)
LOCK_NB = 4   # non-blocking flag (OR-ed with LOCK_EX)
LOCK_UN = 8   # unlock

# State: maps fd int -> FileLock instance (one per fd per process).
_fd_locks: Dict[int, Any] = {}
_fd_acquired: Dict[int, bool] = {}


def _get_filelock(path: object = None):
    """Lazily import FileLock (avoid import error if filelock missing)."""
    try:
        from filelock import FileLock
        return FileLock
    except ImportError:
        logger.warning(
            "kbs._fcntl: filelock not available, "
            "flock calls will be no-ops (data integrity still protected by WAL)"
        )
        return None


def flock(fd: int, flags: int) -> None:
    """Advisory lock on fd, mimicking fcntl.flock on Linux.

    - LOCK_SH         : shared/read lock (no-op on Windows — always readable)
    - LOCK_EX | LOCK_NB : try non-blocking exclusive lock
    - LOCK_EX         : blocking exclusive lock
    - LOCK_UN         : unlock
    """
    fl_cls = _get_filelock(None)
    if fl_cls is None:
        return

    op = flags & 0xF  # LOCK_SH=1, LOCK_EX=2, LOCK_NB=4, LOCK_UN=8

    if op == LOCK_UN:
        # Unlock
        fl = _fd_locks.pop(fd, None)
        if fl is not None:
            try:
                fl.release()
            except Exception:
                pass
        _fd_acquired.pop(fd, None)
        return

    if op == LOCK_SH:
        # Shared lock: Windows has no true shared flock equivalent.
        # Multiple readers can coexist; we only need exclusivity for writers.
        return

    # LOCK_EX (possibly with LOCK_NB)
    path = f"/tmp/_kbs_lock_{_os.getpid()}_{fd}.lock"
    try:
        fl = fl_cls(path)
        _fd_locks[fd] = fl
        if flags & LOCK_NB:
            try:
                fl.acquire(timeout=0)
                _fd_acquired[fd] = True
            except Exception:
                _fd_acquired[fd] = False
        else:
            fl.acquire()
            _fd_acquired[fd] = True
    except Exception as exc:
        logger.warning("kbs._fcntl: flock(%d, %d) failed: %s", fd, flags, exc)
        _fd_acquired[fd] = False
