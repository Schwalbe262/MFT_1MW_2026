"""Cross-process singleton locks for the long-running pipeline roles.

The lock file is deliberately kept after release.  Removing a lock file can
split contenders across two inodes and therefore defeat mutual exclusion.
The operating system releases the byte-range/advisory lock when a process
exits, including an unclean exit.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
from typing import Mapping


class AlreadyRunningError(RuntimeError):
    """Raised when another process owns a pipeline role lock."""

    def __init__(self, role: str, path: Path, owner: str | None = None):
        self.role = role
        self.path = path
        self.owner = owner
        detail = f"; owner={owner}" if owner else ""
        super().__init__(f"pipeline role {role!r} is already running{detail}")


class RoleInstanceLock(AbstractContextManager["RoleInstanceLock"]):
    """Hold one OS lock for one role under a durable pipeline root."""

    VALID_ROLES = frozenset({"controller", "supervisor"})
    # Windows byte-range locks make the locked bytes unreadable.  Locking well
    # beyond EOF keeps the JSON owner record readable without weakening the
    # exclusion contract; LockFile supports ranges beyond the current EOF.
    WINDOWS_LOCK_OFFSET = 1024 * 1024

    def __init__(
        self,
        pipeline_root: str | os.PathLike[str],
        role: str,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        normalized = str(role).strip().lower()
        if normalized not in self.VALID_ROLES:
            raise ValueError(f"unsupported singleton role: {role!r}")
        self.role = normalized
        self.path = Path(pipeline_root).resolve() / "locks" / f"{normalized}.lock"
        self.metadata = dict(metadata or {})
        self._file = None

    @staticmethod
    def _try_lock(handle) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(RoleInstanceLock.WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(RoleInstanceLock.WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read_owner(self) -> str | None:
        try:
            value = self.path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        return value[:1000] or None

    def acquire(self) -> "RoleInstanceLock":
        if self._file is not None:
            raise RuntimeError("role lock is already acquired by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            self._try_lock(handle)
        except (OSError, BlockingIOError):
            handle.close()
            raise AlreadyRunningError(
                self.role, self.path, self._read_owner()
            ) from None

        payload = {
            **self.metadata,
            "schema_version": 1,
            "role": self.role,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        try:
            handle.seek(0)
            handle.write(encoded)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            try:
                self._unlock(handle)
            finally:
                handle.close()
            raise
        self._file = handle
        return self

    def release(self) -> None:
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            self._unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> "RoleInstanceLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
