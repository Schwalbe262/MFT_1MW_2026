"""Read a small, explicit Codex work-status artifact for the 8010 UI.

The monitor does not infer Codex activity from processes, branches, Scheduler
tasks, or free-form logs. A producer must deliberately publish the JSON file
named by ``MFT_CODEX_WORK_STATUS``. This keeps the MFT UI read-only and avoids
coupling it to the separate Scheduler project.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mft-codex-work-status-v1"
STATUS_PATH_ENV = "MFT_CODEX_WORK_STATUS"
STALE_SECONDS_ENV = "MFT_CODEX_WORK_STALE_SECONDS"
DEFAULT_STALE_SECONDS = 1800
MAX_ARTIFACT_BYTES = 256 * 1024
MAX_ITEMS_PER_GROUP = 50
ALLOWED_ITEM_STATES = {
    "in_progress",
    "completed",
    "attention",
    "blocked",
}
GROUP_STATE = {
    "current": {"in_progress"},
    "completed": {"completed"},
    "attention": {"attention", "blocked"},
}
DEADLINE_CAMPAIGN_ITEM_ID = "fea-handoff"
DEADLINE_CAMPAIGN_ALLOCATION_JOBS_PATTERN = re.compile(
    r"\bALLOC(?:ATION)?\s+JOBS?\s+(\d+)(?!\d)",
    re.IGNORECASE,
)
DEADLINE_CAMPAIGN_SUBMITTED_PATTERN = re.compile(
    r"\bSUBMITTED\s+(\d+)(?!\d)",
    re.IGNORECASE,
)
DEADLINE_CAMPAIGN_RUNNING_PATTERN = re.compile(
    r"\bRUNNING\s+(\d+)(?!\d)",
    re.IGNORECASE,
)
DEADLINE_CAMPAIGN_EXACT_RUNNING_PATTERN = re.compile(
    r"\bEXACT\s+RUNNING\s+(\d+)(?!\d)",
    re.IGNORECASE,
)
DEADLINE_CAMPAIGN_SUPPLEMENTAL_RUNNING_PATTERN = re.compile(
    r"\bSUPPLEMENTAL\s+RUNNING\s+(\d+)(?!\d)",
    re.IGNORECASE,
)
DEADLINE_CAMPAIGN_COLLECTIONS_PATTERN = re.compile(
    r"\bCOLLECTIONS?\s+(\d+)(?!\d)",
    re.IGNORECASE,
)


class CodexStatusError(ValueError):
    """Raised when the explicit work-status artifact fails validation."""


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise CodexStatusError(f"{field} must be a non-empty ISO-8601 timestamp")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CodexStatusError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CodexStatusError(f"{field} must include a UTC offset")
    return parsed


def _text(value: Any, field: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodexStatusError(f"{field} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise CodexStatusError(f"{field} exceeds {maximum} characters")
    return normalized


def _item(payload: Any, group: str, index: int) -> dict[str, Any]:
    prefix = f"{group}[{index}]"
    if not isinstance(payload, dict):
        raise CodexStatusError(f"{prefix} must be an object")
    state = _text(payload.get("state"), f"{prefix}.state", maximum=32)
    if state not in ALLOWED_ITEM_STATES or state not in GROUP_STATE[group]:
        raise CodexStatusError(f"{prefix}.state is invalid for {group}")
    updated_at = _timestamp(payload.get("updated_at"), f"{prefix}.updated_at")
    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) > 12:
        raise CodexStatusError(
            f"{prefix}.evidence must be a list with at most 12 entries"
        )
    normalized_evidence = [
        _text(value, f"{prefix}.evidence[{evidence_index}]", maximum=500)
        for evidence_index, value in enumerate(evidence)
    ]
    progress = payload.get("progress_pct")
    if progress is not None and (
        isinstance(progress, bool)
        or not isinstance(progress, (int, float))
        or not 0 <= float(progress) <= 100
    ):
        raise CodexStatusError(f"{prefix}.progress_pct must be between 0 and 100")
    return {
        "id": _text(payload.get("id"), f"{prefix}.id", maximum=80),
        "title": _text(payload.get("title"), f"{prefix}.title", maximum=160),
        "detail": _text(payload.get("detail"), f"{prefix}.detail", maximum=1200),
        "state": state,
        "updated_at": updated_at.isoformat(timespec="seconds"),
        "evidence": normalized_evidence,
        "progress_pct": float(progress) if progress is not None else None,
    }


def _unique_count(pattern: re.Pattern[str], text: str) -> int | None:
    values = {int(value) for value in pattern.findall(text)}
    return next(iter(values)) if len(values) == 1 else None


def _deadline_campaign(
    current: list[dict[str, Any]],
    summary: str = "",
) -> dict[str, Any]:
    """Project explicit campaign counters without consulting Scheduler state."""

    source = next(
        (
            item
            for item in current
            if item.get("id") == DEADLINE_CAMPAIGN_ITEM_ID
        ),
        None,
    )
    unavailable = {
        "available": False,
        "integrity_verified": False,
        "scope": "codex_deadline_campaign",
        "source_item_id": DEADLINE_CAMPAIGN_ITEM_ID,
        "allocation_jobs_active": None,
        "submitted_total": None,
        "running": None,
        "collections": None,
        "collection_zero_means_submission_zero": False,
    }
    if source is None:
        return {
            **unavailable,
            "error": "explicit deadline campaign status item is unavailable",
        }
    title = source["title"]
    allocation_jobs_active = _unique_count(
        DEADLINE_CAMPAIGN_ALLOCATION_JOBS_PATTERN,
        title,
    )
    submitted_total = _unique_count(
        DEADLINE_CAMPAIGN_SUBMITTED_PATTERN,
        title,
    )
    exact_running = _unique_count(
        DEADLINE_CAMPAIGN_EXACT_RUNNING_PATTERN,
        title,
    )
    if exact_running is None:
        running = _unique_count(DEADLINE_CAMPAIGN_RUNNING_PATTERN, title)
    else:
        supplemental_running = _unique_count(
            DEADLINE_CAMPAIGN_SUPPLEMENTAL_RUNNING_PATTERN,
            title,
        )
        running = exact_running + (supplemental_running or 0)
    collections = _unique_count(
        DEADLINE_CAMPAIGN_COLLECTIONS_PATTERN,
        title,
    )
    if collections is None:
        collections = _unique_count(
            DEADLINE_CAMPAIGN_COLLECTIONS_PATTERN,
            summary,
        )
    if (
        allocation_jobs_active is None
        or submitted_total is None
        or running is None
        or collections is None
    ):
        return {
            **unavailable,
            "observed_at": source["updated_at"],
            "error": "deadline campaign counters are unavailable in the status item",
        }
    if (
        running > submitted_total
        or collections > submitted_total
        or (running > 0 and allocation_jobs_active == 0)
    ):
        return {
            **unavailable,
            "observed_at": source["updated_at"],
            "error": "deadline campaign counters are internally inconsistent",
        }
    return {
        "available": True,
        "integrity_verified": True,
        "scope": "codex_deadline_campaign",
        "source_item_id": DEADLINE_CAMPAIGN_ITEM_ID,
        "observed_at": source["updated_at"],
        "allocation_jobs_active": allocation_jobs_active,
        "submitted_total": submitted_total,
        "running": running,
        "collections": collections,
        "collection_zero_means_submission_zero": False,
    }


class CodexWorkStatusReader:
    """Load and validate the configured Codex work-status artifact per request."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        stale_after_seconds: int | None = None,
    ) -> None:
        configured = str(path or os.environ.get(STATUS_PATH_ENV, "")).strip()
        self.path = Path(configured).resolve() if configured else None
        if stale_after_seconds is None:
            raw_stale = os.environ.get(
                STALE_SECONDS_ENV,
                str(DEFAULT_STALE_SECONDS),
            )
            try:
                stale_after_seconds = int(raw_stale)
            except ValueError as exc:
                raise CodexStatusError(
                    f"{STALE_SECONDS_ENV} must be an integer"
                ) from exc
        if stale_after_seconds < 60 or stale_after_seconds > 86400:
            raise CodexStatusError(
                f"{STALE_SECONDS_ENV} must be between 60 and 86400"
            )
        self.stale_after_seconds = stale_after_seconds

    def _unavailable(self, error: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "configured": self.path is not None,
            "available": False,
            "integrity_verified": False,
            "generated_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "status_file": str(self.path) if self.path is not None else None,
            "error": error,
            "current": [],
            "completed": [],
            "attention": [],
            "counts": {"current": 0, "completed": 0, "attention": 0},
            "deadline_campaign": _deadline_campaign([]),
            "stale": True,
        }

    def snapshot(self) -> dict[str, Any]:
        if self.path is None:
            return self._unavailable(
                f"{STATUS_PATH_ENV} is not configured"
            )
        try:
            stat = self.path.stat()
            if not self.path.is_file():
                raise CodexStatusError("configured status path is not a file")
            if stat.st_size > MAX_ARTIFACT_BYTES:
                raise CodexStatusError(
                    f"status artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
                )
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise CodexStatusError("status artifact root must be an object")
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise CodexStatusError(
                    f"schema_version must be {SCHEMA_VERSION}"
                )
            generated_at = _timestamp(
                payload.get("generated_at"),
                "generated_at",
            )
            deadline_at = _timestamp(
                payload.get("deadline_at"),
                "deadline_at",
            )
            now = datetime.now(timezone.utc)
            if generated_at.astimezone(timezone.utc) > now + timedelta(minutes=5):
                raise CodexStatusError(
                    "generated_at is more than five minutes in the future"
                )
            groups: dict[str, list[dict[str, Any]]] = {}
            for group in GROUP_STATE:
                values = payload.get(group)
                if not isinstance(values, list):
                    raise CodexStatusError(f"{group} must be a list")
                if len(values) > MAX_ITEMS_PER_GROUP:
                    raise CodexStatusError(
                        f"{group} exceeds {MAX_ITEMS_PER_GROUP} entries"
                    )
                groups[group] = [
                    _item(value, group, index)
                    for index, value in enumerate(values)
                ]
            identifiers = [
                item["id"]
                for group in groups.values()
                for item in group
            ]
            if len(identifiers) != len(set(identifiers)):
                raise CodexStatusError(
                    "item ids must be unique across all groups"
                )
            age_seconds = max(
                0,
                int(
                    (
                        now
                        - generated_at.astimezone(timezone.utc)
                    ).total_seconds()
                ),
            )
            summary = _text(
                payload.get("summary"),
                "summary",
                maximum=1000,
            )
            return {
                "schema_version": SCHEMA_VERSION,
                "configured": True,
                "available": True,
                "integrity_verified": True,
                "status_file": str(self.path),
                "goal_id": _text(
                    payload.get("goal_id"),
                    "goal_id",
                    maximum=100,
                ),
                "goal": _text(payload.get("goal"), "goal", maximum=600),
                "generated_at": generated_at.isoformat(timespec="seconds"),
                "deadline_at": deadline_at.isoformat(timespec="seconds"),
                "owner": _text(payload.get("owner"), "owner", maximum=100),
                "summary": summary,
                **groups,
                "counts": {
                    group: len(values)
                    for group, values in groups.items()
                },
                "deadline_campaign": _deadline_campaign(
                    groups["current"],
                    summary,
                ),
                "age_seconds": age_seconds,
                "stale_after_seconds": self.stale_after_seconds,
                "stale": age_seconds > self.stale_after_seconds,
            }
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            CodexStatusError,
        ) as exc:
            return self._unavailable(f"{type(exc).__name__}: {exc}")
