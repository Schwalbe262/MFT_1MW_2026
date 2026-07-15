"""SQLite-backed idempotent job queue with expiring worker leases."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import json
import os
import sqlite3
import time
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
ACTIVE_STATES = frozenset({"queued", "retry_wait", "running"})
ALL_STATES = TERMINAL_STATES | ACTIVE_STATES


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


@dataclass(frozen=True)
class Job:
    id: int
    job_type: str
    idempotency_key: str
    input_generation: str | None
    coalesce_key: str | None
    payload: dict
    state: str
    priority: int
    owner_lease: str | None
    heartbeat_at: float | None
    lease_until: float | None
    attempt: int
    max_attempts: int
    next_retry_at: float
    terminal_reason: str | None
    output_generation: str | None
    created_at: float
    updated_at: float


class DurableJobQueue:
    """A small durable queue designed for several independent worker lanes.

    A claim and its attempt increment happen in one ``BEGIN IMMEDIATE``
    transaction.  A worker may finish a job only while it still owns the
    lease.  Expired work becomes retryable, capped by ``max_attempts``.
    """

    def __init__(self, path: str | os.PathLike[str], busy_timeout_ms: int = 30000):
        self.path = os.path.abspath(os.fspath(path))
        self.busy_timeout_ms = int(busy_timeout_ms)
        if self.busy_timeout_ms < 0:
            raise ValueError("busy timeout must be non-negative")
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS queue_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_type TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    input_generation TEXT,
                    coalesce_key TEXT,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    owner_lease TEXT,
                    heartbeat_at REAL,
                    lease_until REAL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    next_retry_at REAL NOT NULL,
                    terminal_reason TEXT,
                    output_generation TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(job_type, idempotency_key),
                    CHECK (state IN (
                        'queued', 'retry_wait', 'running',
                        'succeeded', 'failed', 'cancelled'
                    )),
                    CHECK (attempt >= 0),
                    CHECK (max_attempts >= 1)
                );
                CREATE TABLE IF NOT EXISTS job_dependencies (
                    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    depends_on_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE RESTRICT,
                    PRIMARY KEY(job_id, depends_on_id),
                    CHECK(job_id != depends_on_id)
                );
                CREATE INDEX IF NOT EXISTS ix_jobs_claim
                    ON jobs(state, next_retry_at, priority DESC, created_at, id);
                CREATE INDEX IF NOT EXISTS ix_jobs_lease
                    ON jobs(state, lease_until);
                CREATE INDEX IF NOT EXISTS ix_job_dependencies_parent
                    ON job_dependencies(depends_on_id, job_id);
                """
            )
            columns = {
                item["name"]
                for item in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "coalesce_key" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN coalesce_key TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_jobs_coalesce "
                "ON jobs(job_type, coalesce_key, state, created_at, id)"
            )
            row = connection.execute(
                "SELECT value FROM queue_meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO queue_meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) == 1:
                connection.execute(
                    "UPDATE queue_meta SET value=? WHERE key='schema_version'",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) != SCHEMA_VERSION:
                raise RuntimeError("unsupported durable queue schema")

    @staticmethod
    def _row_to_job(row: sqlite3.Row | None) -> Job | None:
        if row is None:
            return None
        return Job(
            id=int(row["id"]),
            job_type=row["job_type"],
            idempotency_key=row["idempotency_key"],
            input_generation=row["input_generation"],
            coalesce_key=row["coalesce_key"],
            payload=json.loads(row["payload_json"]),
            state=row["state"],
            priority=int(row["priority"]),
            owner_lease=row["owner_lease"],
            heartbeat_at=row["heartbeat_at"],
            lease_until=row["lease_until"],
            attempt=int(row["attempt"]),
            max_attempts=int(row["max_attempts"]),
            next_retry_at=float(row["next_retry_at"]),
            terminal_reason=row["terminal_reason"],
            output_generation=row["output_generation"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _validate_label(value: object, label: str) -> str:
        text = str(value or "").strip()
        if not text or len(text) > 512 or "\x00" in text:
            raise ValueError(f"invalid {label}")
        return text

    def enqueue(
        self,
        job_type: str,
        idempotency_key: str,
        payload: Mapping[str, object],
        *,
        input_generation: str | None = None,
        coalesce_key: str | None = None,
        coalesce_pending: bool = False,
        dependencies: Iterable[int] = (),
        priority: int = 0,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> Job:
        job_type = self._validate_label(job_type, "job type")
        idempotency_key = self._validate_label(
            idempotency_key, "idempotency key"
        )
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        payload_json = _canonical(dict(payload))
        if coalesce_key is not None:
            coalesce_key = self._validate_label(coalesce_key, "coalesce key")
        if coalesce_pending and not coalesce_key:
            raise ValueError("pending coalescing requires a coalesce key")
        dependency_ids = sorted({int(value) for value in dependencies})
        timestamp = time.time() if now is None else float(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM jobs WHERE job_type=? AND idempotency_key=?",
                (job_type, idempotency_key),
            ).fetchone()
            if existing is None:
                cursor = connection.execute(
                    """
                    INSERT INTO jobs(
                        job_type, idempotency_key, input_generation, coalesce_key,
                        payload_json, state, priority, max_attempts,
                        next_retry_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                    """,
                    (
                        job_type,
                        idempotency_key,
                        input_generation,
                        coalesce_key,
                        payload_json,
                        int(priority),
                        int(max_attempts),
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                job_id = int(cursor.lastrowid)
                if job_id in dependency_ids:
                    raise ValueError("a job cannot depend on itself")
                for dependency_id in dependency_ids:
                    if connection.execute(
                        "SELECT 1 FROM jobs WHERE id=?", (dependency_id,)
                    ).fetchone() is None:
                        raise ValueError(
                            f"unknown dependency job: {dependency_id}"
                        )
                    connection.execute(
                        "INSERT INTO job_dependencies(job_id, depends_on_id) "
                        "VALUES(?, ?)",
                        (job_id, dependency_id),
                    )
                row = connection.execute(
                    "SELECT * FROM jobs WHERE id=?", (job_id,)
                ).fetchone()
            else:
                job_id = int(existing["id"])
                actual_dependencies = [
                    int(row["depends_on_id"])
                    for row in connection.execute(
                        "SELECT depends_on_id FROM job_dependencies "
                        "WHERE job_id=? ORDER BY depends_on_id",
                        (job_id,),
                    )
                ]
                invariant = (
                    existing["payload_json"] == payload_json
                    and existing["input_generation"] == input_generation
                    and existing["coalesce_key"] == coalesce_key
                    and int(existing["priority"]) == int(priority)
                    and int(existing["max_attempts"]) == int(max_attempts)
                    and actual_dependencies == dependency_ids
                )
                if not invariant:
                    raise RuntimeError(
                        "idempotency key was reused with different job inputs"
                    )
                row = existing
            if (
                coalesce_pending
                and coalesce_key
                and row["state"] in ("queued", "retry_wait", "running")
            ):
                connection.execute(
                    """
                    UPDATE jobs SET state='cancelled', terminal_reason=?,
                        owner_lease=NULL, heartbeat_at=NULL, lease_until=NULL,
                        updated_at=?
                    WHERE job_type=? AND coalesce_key=? AND id!=?
                      AND state IN ('queued', 'retry_wait')
                    """,
                    (
                        f"superseded_by:{job_id}",
                        timestamp,
                        job_type,
                        coalesce_key,
                        job_id,
                    ),
                )
            connection.commit()
            return self._row_to_job(row)  # type: ignore[return-value]
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _reconcile_locked(connection: sqlite3.Connection, now: float) -> None:
        expired = connection.execute(
            "SELECT id, attempt, max_attempts FROM jobs "
            "WHERE state='running' AND lease_until < ?",
            (now,),
        ).fetchall()
        for row in expired:
            terminal = int(row["attempt"]) >= int(row["max_attempts"])
            connection.execute(
                """
                UPDATE jobs SET state=?, owner_lease=NULL, heartbeat_at=NULL,
                    lease_until=NULL, next_retry_at=?, terminal_reason=?,
                    updated_at=? WHERE id=? AND state='running'
                """,
                (
                    "failed" if terminal else "retry_wait",
                    now,
                    "worker_lease_expired" if terminal else None,
                    now,
                    int(row["id"]),
                ),
            )
        connection.execute(
            "UPDATE jobs SET state='queued', updated_at=? "
            "WHERE state='retry_wait' AND next_retry_at <= ?",
            (now, now),
        )
        # A dependent job can never run once any prerequisite is terminal
        # without success.  Marking it terminal makes the reason observable.
        blocked = connection.execute(
            """
            SELECT DISTINCT child.id, parent.id AS parent_id
            FROM jobs child
            JOIN job_dependencies dep ON dep.job_id=child.id
            JOIN jobs parent ON parent.id=dep.depends_on_id
            WHERE child.state IN ('queued', 'retry_wait')
              AND parent.state IN ('failed', 'cancelled')
            ORDER BY child.id, parent.id
            """
        ).fetchall()
        for row in blocked:
            connection.execute(
                "UPDATE jobs SET state='failed', terminal_reason=?, "
                "updated_at=? WHERE id=? AND state IN ('queued','retry_wait')",
                (f"dependency_failed:{int(row['parent_id'])}", now, int(row["id"])),
            )

    def reconcile(self, now: float | None = None) -> None:
        timestamp = time.time() if now is None else float(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._reconcile_locked(connection, timestamp)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def claim(
        self,
        owner_lease: str,
        *,
        job_types: Sequence[str] | None = None,
        lease_seconds: float = 120.0,
        now: float | None = None,
    ) -> Job | None:
        owner_lease = self._validate_label(owner_lease, "owner lease")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = time.time() if now is None else float(now)
        types = [self._validate_label(value, "job type") for value in job_types or []]
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._reconcile_locked(connection, timestamp)
            type_clause = ""
            parameters: list[object] = [timestamp]
            if types:
                type_clause = " AND child.job_type IN ({})".format(
                    ",".join("?" for _ in types)
                )
                parameters.extend(types)
            row = connection.execute(
                f"""
                SELECT child.* FROM jobs child
                WHERE child.state='queued'
                  AND child.next_retry_at <= ?
                  {type_clause}
                  AND NOT EXISTS (
                    SELECT 1 FROM job_dependencies dep
                    JOIN jobs parent ON parent.id=dep.depends_on_id
                    WHERE dep.job_id=child.id AND parent.state!='succeeded'
                  )
                ORDER BY child.priority DESC, child.created_at, child.id
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            job_id = int(row["id"])
            changed = connection.execute(
                """
                UPDATE jobs SET state='running', owner_lease=?, heartbeat_at=?,
                    lease_until=?, attempt=attempt+1, terminal_reason=NULL,
                    updated_at=? WHERE id=? AND state='queued'
                """,
                (
                    owner_lease,
                    timestamp,
                    timestamp + float(lease_seconds),
                    timestamp,
                    job_id,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("atomic queue claim was lost")
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            connection.commit()
            return self._row_to_job(claimed)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def heartbeat(
        self,
        job_id: int,
        owner_lease: str,
        *,
        lease_seconds: float = 120.0,
        now: float | None = None,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        timestamp = time.time() if now is None else float(now)
        with closing(self._connect()) as connection:
            changed = connection.execute(
                """
                UPDATE jobs SET heartbeat_at=?, lease_until=?, updated_at=?
                WHERE id=? AND state='running' AND owner_lease=?
                  AND lease_until >= ?
                """,
                (
                    timestamp,
                    timestamp + float(lease_seconds),
                    timestamp,
                    int(job_id),
                    owner_lease,
                    timestamp,
                ),
            ).rowcount
        return changed == 1

    def succeed(
        self,
        job_id: int,
        owner_lease: str,
        *,
        output_generation: str | None = None,
        now: float | None = None,
    ) -> Job:
        return self._finish(
            job_id,
            owner_lease,
            state="succeeded",
            terminal_reason=None,
            output_generation=output_generation,
            now=now,
        )

    def fail(
        self,
        job_id: int,
        owner_lease: str,
        reason: str,
        *,
        retry: bool = True,
        base_backoff_seconds: float = 30.0,
        max_backoff_seconds: float = 3600.0,
        now: float | None = None,
    ) -> Job:
        reason = str(reason or "job_failed")[:4000]
        timestamp = time.time() if now is None else float(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=?", (int(job_id),)
            ).fetchone()
            if (
                row is None
                or row["state"] != "running"
                or row["owner_lease"] != owner_lease
                or row["lease_until"] is None
                or float(row["lease_until"]) < timestamp
            ):
                raise RuntimeError("worker no longer owns the running job")
            can_retry = retry and int(row["attempt"]) < int(row["max_attempts"])
            delay = min(
                float(max_backoff_seconds),
                float(base_backoff_seconds) * (2 ** max(0, int(row["attempt"]) - 1)),
            )
            connection.execute(
                """
                UPDATE jobs SET state=?, owner_lease=NULL, heartbeat_at=NULL,
                    lease_until=NULL, next_retry_at=?, terminal_reason=?,
                    updated_at=? WHERE id=?
                """,
                (
                    "retry_wait" if can_retry else "failed",
                    timestamp + delay if can_retry else timestamp,
                    reason,
                    timestamp,
                    int(job_id),
                ),
            )
            result = connection.execute(
                "SELECT * FROM jobs WHERE id=?", (int(job_id),)
            ).fetchone()
            connection.commit()
            return self._row_to_job(result)  # type: ignore[return-value]
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _finish(
        self,
        job_id: int,
        owner_lease: str,
        *,
        state: str,
        terminal_reason: str | None,
        output_generation: str | None,
        now: float | None,
    ) -> Job:
        if state not in TERMINAL_STATES:
            raise ValueError("finish state must be terminal")
        timestamp = time.time() if now is None else float(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE jobs SET state=?, owner_lease=NULL, heartbeat_at=NULL,
                    lease_until=NULL, terminal_reason=?, output_generation=?,
                    updated_at=?
                WHERE id=? AND state='running' AND owner_lease=?
                  AND lease_until >= ?
                """,
                (
                    state,
                    terminal_reason,
                    output_generation,
                    timestamp,
                    int(job_id),
                    owner_lease,
                    timestamp,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("worker no longer owns the running job")
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=?", (int(job_id),)
            ).fetchone()
            connection.commit()
            return self._row_to_job(row)  # type: ignore[return-value]
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def cancel(self, job_id: int, reason: str = "operator_cancelled") -> Job:
        timestamp = time.time()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE jobs SET state='cancelled', terminal_reason=?,
                    owner_lease=NULL, heartbeat_at=NULL, lease_until=NULL,
                    updated_at=?
                WHERE id=? AND state IN ('queued', 'retry_wait')
                """,
                (str(reason)[:4000], timestamp, int(job_id)),
            ).rowcount
            if changed != 1:
                raise RuntimeError("only queued jobs can be cancelled safely")
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=?", (int(job_id),)
            ).fetchone()
            connection.commit()
            return self._row_to_job(row)  # type: ignore[return-value]
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def cancel_coalesced_pending(
        self,
        job_type: str,
        coalesce_key: str,
        reason: str,
        now: float | None = None,
    ) -> int:
        """Cancel obsolete queued work without disturbing an active owner."""
        job_type = self._validate_label(job_type, "job type")
        coalesce_key = self._validate_label(coalesce_key, "coalesce key")
        timestamp = time.time() if now is None else float(now)
        with closing(self._connect()) as connection:
            changed = connection.execute(
                """
                UPDATE jobs SET state='cancelled', terminal_reason=?,
                    owner_lease=NULL, heartbeat_at=NULL, lease_until=NULL,
                    updated_at=?
                WHERE job_type=? AND coalesce_key=?
                  AND state IN ('queued', 'retry_wait')
                """,
                (
                    str(reason or "coalesced_work_no_longer_due")[:4000],
                    timestamp,
                    job_type,
                    coalesce_key,
                ),
            ).rowcount
        return int(changed)

    def get(self, job_id: int) -> Job | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id=?", (int(job_id),)
            ).fetchone()
        return self._row_to_job(row)

    def dependencies(self, job_id: int) -> list[Job]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT parent.* FROM jobs parent
                JOIN job_dependencies dep ON dep.depends_on_id=parent.id
                WHERE dep.job_id=? ORDER BY parent.id
                """,
                (int(job_id),),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]  # type: ignore[misc]

    def list(
        self,
        *,
        states: Sequence[str] | None = None,
        job_types: Sequence[str] | None = None,
        limit: int = 1000,
    ) -> list[Job]:
        clauses = []
        parameters: list[object] = []
        if states:
            invalid = set(states) - ALL_STATES
            if invalid:
                raise ValueError(f"invalid queue states: {sorted(invalid)}")
            clauses.append("state IN ({})".format(",".join("?" for _ in states)))
            parameters.extend(states)
        if job_types:
            clauses.append(
                "job_type IN ({})".format(",".join("?" for _ in job_types))
            )
            parameters.extend(job_types)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(max(1, int(limit)))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at, id LIMIT ?",
                parameters,
            ).fetchall()
        return [self._row_to_job(row) for row in rows]  # type: ignore[misc]

    def stats(self) -> dict[str, int]:
        self.reconcile()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM jobs GROUP BY state"
            ).fetchall()
        output = {state: 0 for state in sorted(ALL_STATES)}
        output.update({row["state"]: int(row["count"]) for row in rows})
        return output
