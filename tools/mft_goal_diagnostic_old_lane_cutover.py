#!/usr/bin/env python3
"""Fail-closed cutover for the exact legacy fixed-gap continuation lane.

The default and ``audit-old`` paths are GET-only.  Cancellation is unavailable
unless an explicitly supplied, self-sealed NEW exact100 submission receipt is
bound to a separate, self-sealed live readback attestation.  Even then,
``cutover`` remains audit-only unless ``--apply`` is present.

The only mutable Scheduler endpoint in this module is the conditional,
one-task-at-a-time endpoint::

    POST /api/tasks/{id}/cancel?expected_statuses=<fresh observed status>

No task outside 97297..97378 can enter that path.  Terminal tasks and any task
whose API/HTML submission identity does not match the immutable legacy contract
are never cancelled.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import threading
from typing import Any, Mapping, Protocol
import urllib.error
import urllib.parse
import urllib.request


SCHEDULER_URL = "http://127.0.0.1:8002"

OLD_TASK_IDS = tuple(range(97297, 97379))
OLD_TASK_ID_START = OLD_TASK_IDS[0]
OLD_TASK_ID_END = OLD_TASK_IDS[-1]
OLD_SEED_START = 2607264200
OLD_SEED_END = 2607264281
OLD_AUTHORIZED_SEED_END = 2607264299
OLD_BUNDLE_NAME = "mft-goal-diag-compact-fddc1b9f03aabb158d829441"
OLD_PROFILE_SHA256 = "76f69bb6588879e5a726953bccc8f1638a1c9bcab96dca78ac0c202754575bc8"
OLD_PRIORITY = 10
OLD_FIXED_SECONDARY_GAP_MM = 0.35
OLD_DEDUPE_PREFIX = "mft-goal-20260726-diag-compact:feeder:"

NEW_TASK_COUNT = 100
NEW_SEED_START = 2607264300
NEW_SEED_END = 2607264399
NEW_PRIORITY = 100
NEW_GAP_MINIMUM_MM = 0.35
NEW_GAP_MAXIMUM_MM = 2.0
NEW_GAP_STEP_MM = 0.001
NEW_CW2_MINIMUM_MM = 0.3
NEW_CW2_MAXIMUM_MM = 1.0

CANCELLABLE_STATUSES = frozenset({"queued", "attaching", "running"})
TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "timeout", "timed_out"}
)
KNOWN_STATUSES = CANCELLABLE_STATUSES | TERMINAL_STATUSES
REPLACEMENT_READY_STATUSES = CANCELLABLE_STATUSES | {"completed"}

OLD_AUDIT_SCHEMA = "mft-goal-diagnostic-old-fixed-gap-audit-v1"
NEW_RECEIPT_SCHEMA = "mft-goal-diagnostic-compact-submission-receipt-v1"
NEW_READBACK_SCHEMA = "mft-goal-diagnostic-new-exact100-readback-v1"
CUTOVER_AUDIT_SCHEMA = "mft-goal-diagnostic-old-lane-cutover-audit-v1"
PRE_CUTOVER_SCHEMA = "mft-goal-diagnostic-old-lane-pre-cutover-v1"
POST_CUTOVER_SCHEMA = "mft-goal-diagnostic-old-lane-post-cutover-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PAYLOAD_CELL = re.compile(
    r"<tr>\s*<th>\s*Payload JSON\s*</th>\s*"
    r"<td[^>]*>\s*<pre>(.*?)</pre>\s*</td>\s*</tr>",
    re.IGNORECASE | re.DOTALL,
)
_TITLE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
_REMOTE_CWD_CELL = re.compile(
    r"<tr>\s*<th>\s*Remote CWD\s*</th>\s*"
    r"<td[^>]*>(.*?)</td>\s*</tr>",
    re.IGNORECASE | re.DOTALL,
)


class CutoverError(RuntimeError):
    """A cutover invariant failed; no further mutation is allowed."""


class ConditionalCancelRejected(CutoverError):
    """Scheduler rejected a compare-and-swap cancellation."""


class Scheduler(Protocol):
    get_count: int
    html_get_count: int
    cancel_count: int

    def get_task(self, task_id: int) -> dict[str, Any]: ...

    def get_task_html(self, task_id: int) -> str: ...

    def cancel_task(self, task_id: int, *, expected_status: str) -> dict[str, Any]: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise CutoverError("value is already sealed")
    result["payload_sha256"] = _sha256_value(result)
    return result


def _validate_seal(value: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("payload_sha256", None)
    if result.get("schema_version") != schema or observed != _sha256_value(unsigned):
        raise CutoverError(f"{schema} seal mismatch")
    return result


def _validate_offload_receipt_seal(
    value: Mapping[str, Any], *, schema: str
) -> dict[str, Any]:
    """Validate the diagnostic offload module's historical ``sha256`` seal."""

    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("sha256", None)
    if (
        "payload_sha256" in result
        or result.get("schema_version") != schema
        or observed != _sha256_value(unsigned)
    ):
        raise CutoverError(f"{schema} seal mismatch")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CutoverError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CutoverError(f"JSON object required: {path}")
    return value


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create a JSON artifact without replacing an existing different value."""

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _read_json(target) != dict(value):
            raise CutoverError(f"immutable cutover artifact changed: {target}")
        return
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, target)
        except FileExistsError:
            if _read_json(target) != dict(value):
                raise CutoverError(f"concurrent cutover artifact differs: {target}")
    finally:
        if os.path.exists(staged):
            os.remove(staged)


class SchedulerClient:
    """Small HTTP client exposing only the reads and conditional cancel used here."""

    def __init__(self, base_url: str = SCHEDULER_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.get_count = 0
        self.html_get_count = 0
        self.cancel_count = 0
        self._lock = threading.Lock()

    def _request(self, endpoint: str, *, method: str) -> bytes:
        request = urllib.request.Request(
            self.base_url + endpoint,
            headers={"Accept": "application/json, text/html"},
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if method == "POST" and exc.code in {409, 412, 422}:
                raise ConditionalCancelRejected(
                    f"conditional cancel rejected: HTTP {exc.code}"
                ) from exc
            raise CutoverError(f"{method} {endpoint} failed: HTTP {exc.code}") from exc
        except OSError as exc:
            raise CutoverError(f"{method} {endpoint} failed: {exc}") from exc

    def get_task(self, task_id: int) -> dict[str, Any]:
        raw = self._request(f"/api/tasks/{int(task_id)}", method="GET")
        with self._lock:
            self.get_count += 1
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CutoverError("Scheduler task GET returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise CutoverError("Scheduler task GET returned non-object JSON")
        return value

    def get_task_html(self, task_id: int) -> str:
        raw = self._request(f"/tasks/{int(task_id)}", method="GET")
        with self._lock:
            self.html_get_count += 1
        return raw.decode("utf-8", errors="strict")

    def cancel_task(self, task_id: int, *, expected_status: str) -> dict[str, Any]:
        if task_id not in OLD_TASK_IDS or expected_status not in CANCELLABLE_STATUSES:
            raise CutoverError("conditional cancel target is outside authority")
        query = urllib.parse.urlencode({"expected_statuses": expected_status})
        endpoint = f"/api/tasks/{task_id}/cancel?{query}"
        raw = self._request(endpoint, method="POST")
        with self._lock:
            self.cancel_count += 1
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CutoverError("conditional cancel returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise CutoverError("conditional cancel returned non-object JSON")
        return value


def _strip_tags(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _html_submission(
    page: str, *, task_id: int, expected_name: str
) -> tuple[dict[str, Any], str]:
    title_match = _TITLE.search(page)
    payload_match = _PAYLOAD_CELL.search(page)
    remote_match = _REMOTE_CWD_CELL.search(page)
    if title_match is None or payload_match is None or remote_match is None:
        raise CutoverError("task HTML is missing identity/payload fields")
    if _strip_tags(title_match.group(1)) != f"Task {task_id}: {expected_name}":
        raise CutoverError("task HTML title identity drifted")
    try:
        payload = json.loads(html.unescape(payload_match.group(1)))
    except json.JSONDecodeError as exc:
        raise CutoverError("task HTML submission payload is invalid") from exc
    if not isinstance(payload, dict):
        raise CutoverError("task HTML submission payload must be an object")
    unsigned = dict(payload)
    observed_sha = unsigned.pop("payload_sha256", None)
    if observed_sha != _sha256_value(unsigned):
        raise CutoverError("task HTML submission payload seal mismatch")
    return payload, _strip_tags(remote_match.group(1))


def _task_id(value: Mapping[str, Any]) -> int:
    observed = value.get("task_id", value.get("id"))
    if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
        raise CutoverError("Scheduler task ID is invalid")
    return observed


def _status(value: Mapping[str, Any]) -> str:
    observed = str(value.get("status") or value.get("state") or "")
    if observed not in KNOWN_STATUSES:
        raise CutoverError(f"Scheduler task status is not recognized: {observed!r}")
    return observed


def _old_seed(task_id: int) -> int:
    if task_id not in OLD_TASK_IDS:
        raise CutoverError("legacy task ID is outside the exact authority")
    return OLD_SEED_START + task_id - OLD_TASK_ID_START


def _authenticate_old_task(scheduler: Scheduler, task_id: int) -> dict[str, Any]:
    seed = _old_seed(task_id)
    expected_name = f"{OLD_BUNDLE_NAME}-s{seed}-n1-6"
    api = scheduler.get_task(task_id)
    page = scheduler.get_task_html(task_id)
    payload, html_remote_cwd = _html_submission(
        page, task_id=task_id, expected_name=expected_name
    )
    api_remote_cwd = str(api.get("remote_cwd") or "")
    profile = payload.get("manufacturing_search_profile")
    proof = (payload.get("activation") or {}).get("aligned_compact_bank_proof")
    dedupe = str(api.get("dedupe_key") or "")
    if (
        _task_id(api) != task_id
        or api.get("name") != expected_name
        or api.get("priority") != OLD_PRIORITY
        or api_remote_cwd != html_remote_cwd
        or PurePosixPath(api_remote_cwd).name != OLD_BUNDLE_NAME
        or not dedupe.startswith(OLD_DEDUPE_PREFIX)
        or _SHA256.fullmatch(dedupe.removeprefix(OLD_DEDUPE_PREFIX)) is None
        or payload.get("seed") != seed
        or payload.get("fixed_primary_turns") != 6
        or payload.get("manufacturing_search_profile_payload_sha256")
        != OLD_PROFILE_SHA256
        or not isinstance(profile, dict)
        or profile.get("payload_sha256") != OLD_PROFILE_SHA256
        or profile.get("fixed_secondary_interturn_gap_mm") != OLD_FIXED_SECONDARY_GAP_MM
        or profile.get("fixed_primary_turns") != 6
        or profile.get("fixed_secondary_turns") != 60
        or profile.get("authorized_seed_start") != OLD_SEED_START
        or profile.get("authorized_seed_end_inclusive") != OLD_AUTHORIZED_SEED_END
        or not isinstance(proof, dict)
        or proof.get("search_profile_payload_sha256") != OLD_PROFILE_SHA256
        or proof.get("fixed_secondary_interturn_gap_mm") != OLD_FIXED_SECONDARY_GAP_MM
        or proof.get("all_rows_gap2_exactly_fixed") is not True
    ):
        raise CutoverError(f"legacy task {task_id} identity drifted")
    profile_unsigned = dict(profile)
    profile_observed_sha = profile_unsigned.pop("payload_sha256", None)
    if profile_observed_sha != _sha256_value(profile_unsigned):
        raise CutoverError(f"legacy task {task_id} profile seal drifted")
    observed_status = _status(api)
    return {
        "task_id": task_id,
        "seed": seed,
        "name": expected_name,
        "status": observed_status,
        "priority": OLD_PRIORITY,
        "dedupe_key": dedupe,
        "remote_cwd": api_remote_cwd,
        "bundle_name": OLD_BUNDLE_NAME,
        "manufacturing_search_profile_payload_sha256": OLD_PROFILE_SHA256,
        "fixed_secondary_interturn_gap_mm": OLD_FIXED_SECONDARY_GAP_MM,
        "api_observation_sha256": _sha256_value(api),
        "html_page_sha256": _sha256_bytes(page.encode("utf-8")),
        "html_submission_payload_sha256": payload["payload_sha256"],
        "identity_authenticated": True,
        "cancellable": observed_status in CANCELLABLE_STATUSES,
        "terminal_preserved": observed_status in TERMINAL_STATUSES,
    }


def _parallel_authenticate(
    scheduler: Scheduler,
    task_ids: list[int],
    authenticator: Any,
) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=min(16, len(task_ids))) as executor:
        rows = list(
            executor.map(lambda task_id: authenticator(scheduler, task_id), task_ids)
        )
    return sorted(rows, key=lambda row: int(row["task_id"]))


def audit_old(scheduler: Scheduler | None = None) -> dict[str, Any]:
    """GET-authenticate all and only the immutable 82-task legacy lane."""

    client = scheduler or SchedulerClient()
    rows = _parallel_authenticate(client, list(OLD_TASK_IDS), _authenticate_old_task)
    if (
        [row["task_id"] for row in rows] != list(OLD_TASK_IDS)
        or [row["seed"] for row in rows]
        != list(range(OLD_SEED_START, OLD_SEED_END + 1))
        or not all(row["identity_authenticated"] for row in rows)
    ):
        raise CutoverError("legacy exact82 audit coverage mismatch")
    return _seal(
        {
            "schema_version": OLD_AUDIT_SCHEMA,
            "observed_at": _now(),
            "apply": False,
            "scheduler_mutation_performed": False,
            "authorized_task_id_start": OLD_TASK_ID_START,
            "authorized_task_id_end": OLD_TASK_ID_END,
            "authorized_task_count": len(OLD_TASK_IDS),
            "authorized_seed_start": OLD_SEED_START,
            "authorized_seed_end_inclusive": OLD_SEED_END,
            "old_bundle_name": OLD_BUNDLE_NAME,
            "old_profile_payload_sha256": OLD_PROFILE_SHA256,
            "old_priority": OLD_PRIORITY,
            "fixed_secondary_interturn_gap_mm": (OLD_FIXED_SECONDARY_GAP_MM),
            "rows": rows,
            "cancellable_task_ids": [
                row["task_id"] for row in rows if row["cancellable"]
            ],
            "terminal_preserved_task_ids": [
                row["task_id"] for row in rows if row["terminal_preserved"]
            ],
            "scheduler_get_count": client.get_count,
            "scheduler_html_get_count": client.html_get_count,
            "scheduler_cancel_count": client.cancel_count,
        }
    )


def _validate_new_receipt(path: Path) -> dict[str, Any]:
    receipt = _validate_offload_receipt_seal(
        _read_json(path), schema=NEW_RECEIPT_SCHEMA
    )
    rows = receipt.get("tasks")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise CutoverError("NEW receipt task inventory is invalid")
    seeds = [row.get("seed") for row in rows]
    task_ids = [row.get("task_id") for row in rows]
    names = [row.get("name") for row in rows]
    dedupes = [row.get("dedupe_key") for row in rows]
    bundle_id = receipt.get("bundle_id")
    if (
        receipt.get("apply") is not True
        or receipt.get("task_count") != NEW_TASK_COUNT
        or receipt.get("absent_count") != 0
        or receipt.get("scheduler_cancel_count") != 0
        or receipt.get("scheduler_preempt_count") != 0
        or not isinstance(bundle_id, str)
        or not bundle_id
        or bundle_id == OLD_BUNDLE_NAME
        or len(rows) != NEW_TASK_COUNT
        or sorted(seeds) != list(range(NEW_SEED_START, NEW_SEED_END + 1))
        or any(
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or task_id in OLD_TASK_IDS
            for task_id in task_ids
        )
        or len(set(task_ids)) != NEW_TASK_COUNT
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != NEW_TASK_COUNT
        or any(not isinstance(dedupe, str) or not dedupe for dedupe in dedupes)
        or len(set(dedupes)) != NEW_TASK_COUNT
    ):
        raise CutoverError("NEW exact100 receipt contract mismatch")
    return receipt


def _validate_variable_profile(
    payload: Mapping[str, Any], *, profile_sha256: str, seed: int
) -> None:
    profile = payload.get("manufacturing_search_profile")
    proof = (payload.get("activation") or {}).get("aligned_compact_bank_proof")
    if not isinstance(profile, dict):
        raise CutoverError("NEW task manufacturing profile is absent")
    unsigned = dict(profile)
    observed = unsigned.pop("payload_sha256", None)
    gap = profile.get("secondary_interturn_gap_search_mm") or {}
    cw2 = profile.get("secondary_conductor_thickness_search_mm") or {}
    if (
        payload.get("seed") != seed
        or payload.get("manufacturing_search_profile_payload_sha256") != profile_sha256
        or observed != profile_sha256
        or observed != _sha256_value(unsigned)
        or profile_sha256 == OLD_PROFILE_SHA256
        or "fixed_secondary_interturn_gap_mm" in profile
        or gap.get("minimum") != NEW_GAP_MINIMUM_MM
        or gap.get("maximum") != NEW_GAP_MAXIMUM_MM
        or gap.get("step") != NEW_GAP_STEP_MM
        or cw2.get("minimum") != NEW_CW2_MINIMUM_MM
        or cw2.get("maximum") != NEW_CW2_MAXIMUM_MM
        or profile.get("scheduler_priority") != NEW_PRIORITY
        or profile.get("authorized_seed_start") != NEW_SEED_START
        or profile.get("authorized_seed_end_inclusive") != NEW_SEED_END
        or profile.get("fixed_primary_turns") != 6
        or profile.get("fixed_secondary_turns") != 60
        or not isinstance(proof, dict)
        or proof.get("search_profile_payload_sha256") != profile_sha256
        or proof.get("secondary_gap_mode") != "bounded_gap2_cw2"
        or proof.get("all_rows_gap2_within_allowed_band") is not True
        or proof.get("all_rows_cw2_within_allowed_band") is not True
    ):
        raise CutoverError("NEW bounded gap/cw2 profile contract mismatch")


def _authenticate_new_row(
    scheduler: Scheduler,
    *,
    receipt_row: Mapping[str, Any],
    bundle_id: str,
    profile_sha256: str,
    require_ready_status: bool,
) -> dict[str, Any]:
    task_id = receipt_row.get("task_id")
    seed = receipt_row.get("seed")
    name = receipt_row.get("name")
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not isinstance(name, str)
    ):
        raise CutoverError("NEW receipt row identity is invalid")
    api = scheduler.get_task(task_id)
    page = scheduler.get_task_html(task_id)
    payload, html_remote_cwd = _html_submission(
        page, task_id=task_id, expected_name=name
    )
    api_remote_cwd = str(api.get("remote_cwd") or "")
    observed_status = _status(api)
    if (
        _task_id(api) != task_id
        or api.get("name") != name
        or api.get("dedupe_key") != receipt_row.get("dedupe_key")
        or api.get("priority") != NEW_PRIORITY
        or api_remote_cwd != html_remote_cwd
        or PurePosixPath(api_remote_cwd).name != bundle_id
        or (require_ready_status and observed_status not in REPLACEMENT_READY_STATUSES)
    ):
        raise CutoverError(f"NEW exact100 task {task_id} readback drifted")
    _validate_variable_profile(payload, profile_sha256=profile_sha256, seed=seed)
    return {
        "task_id": task_id,
        "seed": seed,
        "name": name,
        "dedupe_key": receipt_row["dedupe_key"],
        "status": observed_status,
        "priority": NEW_PRIORITY,
        "remote_cwd": api_remote_cwd,
        "bundle_id": bundle_id,
        "manufacturing_search_profile_payload_sha256": profile_sha256,
        "api_observation_sha256": _sha256_value(api),
        "html_page_sha256": _sha256_bytes(page.encode("utf-8")),
        "html_submission_payload_sha256": payload["payload_sha256"],
        "identity_authenticated": True,
        "replacement_ready_status": (observed_status in REPLACEMENT_READY_STATUSES),
    }


def _fresh_new_rows(
    scheduler: Scheduler,
    *,
    receipt: Mapping[str, Any],
    profile_sha256: str,
    require_ready_status: bool = True,
) -> list[dict[str, Any]]:
    rows_by_id = {int(row["task_id"]): row for row in receipt["tasks"]}
    task_ids = sorted(rows_by_id)
    bundle_id = str(receipt["bundle_id"])
    with ThreadPoolExecutor(max_workers=16) as executor:
        rows = list(
            executor.map(
                lambda task_id: _authenticate_new_row(
                    scheduler,
                    receipt_row=rows_by_id[task_id],
                    bundle_id=bundle_id,
                    profile_sha256=profile_sha256,
                    require_ready_status=require_ready_status,
                ),
                task_ids,
            )
        )
    rows.sort(key=lambda row: int(row["seed"]))
    if [row["seed"] for row in rows] != list(range(NEW_SEED_START, NEW_SEED_END + 1)):
        raise CutoverError("NEW live exact100 seed coverage mismatch")
    return rows


def attest_new_readback(
    *,
    new_receipt_path: Path,
    new_profile_sha256: str,
    evidence_out: Path,
    scheduler: Scheduler | None = None,
) -> dict[str, Any]:
    """Create the separately sealed live readback evidence used by cutover."""

    if _SHA256.fullmatch(new_profile_sha256) is None:
        raise CutoverError("NEW profile SHA-256 is invalid")
    receipt = _validate_new_receipt(new_receipt_path)
    client = scheduler or SchedulerClient()
    rows = _fresh_new_rows(client, receipt=receipt, profile_sha256=new_profile_sha256)
    evidence = _seal(
        {
            "schema_version": NEW_READBACK_SCHEMA,
            "observed_at": _now(),
            "new_receipt_path": str(new_receipt_path.resolve(strict=True)),
            "new_receipt_file_sha256": _sha256_file(new_receipt_path),
            "new_receipt_payload_sha256": receipt["sha256"],
            "new_bundle_id": receipt["bundle_id"],
            "new_profile_payload_sha256": new_profile_sha256,
            "authorized_seed_start": NEW_SEED_START,
            "authorized_seed_end_inclusive": NEW_SEED_END,
            "task_count": NEW_TASK_COUNT,
            "tasks": rows,
            "all_api_and_html_readbacks_valid": True,
            "all_replacement_statuses_ready": True,
            "scheduler_get_count": client.get_count,
            "scheduler_html_get_count": client.html_get_count,
            "scheduler_cancel_count": client.cancel_count,
            "scheduler_mutation_performed": False,
        }
    )
    _immutable_json(evidence_out, evidence)
    return evidence


def _validate_new_evidence(
    path: Path,
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _validate_seal(_read_json(path), schema=NEW_READBACK_SCHEMA)
    rows = evidence.get("tasks")
    receipt_identity = {
        (
            row["task_id"],
            row["seed"],
            row["name"],
            row["dedupe_key"],
        )
        for row in receipt["tasks"]
    }
    evidence_identity = {
        (
            row.get("task_id"),
            row.get("seed"),
            row.get("name"),
            row.get("dedupe_key"),
        )
        for row in rows or []
        if isinstance(row, dict)
    }
    profile_sha = evidence.get("new_profile_payload_sha256")
    if (
        evidence.get("new_receipt_path") != str(receipt_path.resolve(strict=True))
        or evidence.get("new_receipt_file_sha256") != _sha256_file(receipt_path)
        or evidence.get("new_receipt_payload_sha256") != receipt["sha256"]
        or evidence.get("new_bundle_id") != receipt["bundle_id"]
        or _SHA256.fullmatch(str(profile_sha or "")) is None
        or profile_sha == OLD_PROFILE_SHA256
        or evidence.get("authorized_seed_start") != NEW_SEED_START
        or evidence.get("authorized_seed_end_inclusive") != NEW_SEED_END
        or evidence.get("task_count") != NEW_TASK_COUNT
        or not isinstance(rows, list)
        or len(rows) != NEW_TASK_COUNT
        or receipt_identity != evidence_identity
        or evidence.get("all_api_and_html_readbacks_valid") is not True
        or evidence.get("all_replacement_statuses_ready") is not True
        or evidence.get("scheduler_cancel_count") != 0
        or evidence.get("scheduler_mutation_performed") is not False
        or any(
            row.get("identity_authenticated") is not True
            or row.get("replacement_ready_status") is not True
            or row.get("priority") != NEW_PRIORITY
            or row.get("manufacturing_search_profile_payload_sha256") != profile_sha
            for row in rows
        )
    ):
        raise CutoverError("NEW exact100 readback evidence mismatch")
    return evidence


def _cutover_binding(
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    evidence_path: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "new_receipt_path": str(receipt_path.resolve(strict=True)),
        "new_receipt_file_sha256": _sha256_file(receipt_path),
        "new_receipt_payload_sha256": receipt["sha256"],
        "new_readback_evidence_path": str(evidence_path.resolve(strict=True)),
        "new_readback_evidence_file_sha256": _sha256_file(evidence_path),
        "new_readback_evidence_payload_sha256": evidence["payload_sha256"],
        "new_profile_payload_sha256": evidence["new_profile_payload_sha256"],
        "new_bundle_id": receipt["bundle_id"],
        "new_exact100_seed_start": NEW_SEED_START,
        "new_exact100_seed_end_inclusive": NEW_SEED_END,
        "new_exact100_task_count": NEW_TASK_COUNT,
    }


def run_cutover(
    *,
    new_receipt_path: Path,
    new_readback_evidence_path: Path,
    output_root: Path,
    apply: bool = False,
    scheduler: Scheduler | None = None,
) -> dict[str, Any]:
    """Audit or conditionally cancel the authenticated legacy nonterminal rows."""

    receipt = _validate_new_receipt(new_receipt_path)
    evidence = _validate_new_evidence(
        new_readback_evidence_path,
        receipt_path=new_receipt_path,
        receipt=receipt,
    )
    client = scheduler or SchedulerClient()

    # Re-authenticate the replacement immediately; the evidence is necessary
    # authority, not a substitute for a fresh live readback.
    fresh_new = _fresh_new_rows(
        client,
        receipt=receipt,
        profile_sha256=evidence["new_profile_payload_sha256"],
        require_ready_status=False,
    )
    evidence_identity = {
        (row["task_id"], row["seed"], row["name"], row["dedupe_key"])
        for row in evidence["tasks"]
    }
    if {
        (row["task_id"], row["seed"], row["name"], row["dedupe_key"])
        for row in fresh_new
    } != evidence_identity:
        raise CutoverError("fresh NEW readback no longer matches evidence")

    # This is deliberately the last bulk operation before an apply decision.
    # All 82 rows are read through both API and task HTML in parallel.
    fresh_old_audit = audit_old(client)
    old_rows = fresh_old_audit["rows"]
    binding = _cutover_binding(
        receipt_path=new_receipt_path,
        receipt=receipt,
        evidence_path=new_readback_evidence_path,
        evidence=evidence,
    )
    selected = [row["task_id"] for row in old_rows if row["cancellable"]]
    preserved = [row["task_id"] for row in old_rows if row["terminal_preserved"]]

    audit_value = _seal(
        {
            "schema_version": CUTOVER_AUDIT_SCHEMA,
            "observed_at": _now(),
            "apply": bool(apply),
            "new_authority": binding,
            "fresh_new_exact100_rows": fresh_new,
            "fresh_old_audit_payload_sha256": fresh_old_audit["payload_sha256"],
            "fresh_old_rows": old_rows,
            "selected_conditional_cancel_task_ids": selected,
            "terminal_preserved_task_ids": preserved,
            "conditional_cancel_endpoint": (
                "POST /api/tasks/{id}/cancel?expected_statuses=<fresh_observed_status>"
            ),
            "out_of_range_cancel_allowed": False,
            "mismatched_cancel_allowed": False,
            "terminal_cancel_allowed": False,
            "scheduler_mutation_performed": False,
            "scheduler_get_count": client.get_count,
            "scheduler_html_get_count": client.html_get_count,
            "scheduler_cancel_count": client.cancel_count,
        }
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not apply:
        _immutable_json(output_root / "cutover_audit.json", audit_value)
        return audit_value

    pre_path = output_root / "pre_cutover_receipt.json"
    post_path = output_root / "post_cutover_receipt.json"
    if post_path.exists():
        post = _validate_seal(_read_json(post_path), schema=POST_CUTOVER_SCHEMA)
        if post.get("new_authority") != binding:
            raise CutoverError("completed post-cutover authority changed")
        return post

    if pre_path.exists():
        pre = _validate_seal(_read_json(pre_path), schema=PRE_CUTOVER_SCHEMA)
        if pre.get("new_authority") != binding or set(
            pre.get("selected_conditional_cancel_task_ids") or []
        ) - set(OLD_TASK_IDS):
            raise CutoverError("pre-cutover authority changed")
    else:
        pre = _seal(
            {
                "schema_version": PRE_CUTOVER_SCHEMA,
                "observed_at": _now(),
                "apply": True,
                "new_authority": binding,
                "fresh_new_exact100_identity_sha256": _sha256_value(fresh_new),
                "fresh_old_audit_payload_sha256": fresh_old_audit["payload_sha256"],
                "fresh_old_rows": old_rows,
                "selected_conditional_cancel_task_ids": selected,
                "terminal_preserved_task_ids": preserved,
                "conditional_cancel_endpoint": (
                    "POST /api/tasks/{id}/cancel?"
                    "expected_statuses=<fresh_observed_status>"
                ),
                "authorized_old_task_ids": list(OLD_TASK_IDS),
                "maximum_cancel_post_count": len(selected),
                "out_of_range_cancel_allowed": False,
                "mismatched_cancel_allowed": False,
                "terminal_cancel_allowed": False,
                "scheduler_cancel_count_before": client.cancel_count,
            }
        )
        _immutable_json(pre_path, pre)

    selected_from_pre = list(pre["selected_conditional_cancel_task_ids"])
    if len(selected_from_pre) != len(set(selected_from_pre)) or any(
        task_id not in OLD_TASK_IDS for task_id in selected_from_pre
    ):
        raise CutoverError("pre-cutover target set is invalid")

    actions: list[dict[str, Any]] = []
    for task_id in selected_from_pre:
        # A fresh API+HTML identity read occurs immediately before each CAS POST.
        before = _authenticate_old_task(client, task_id)
        observed_status = before["status"]
        if observed_status in TERMINAL_STATUSES:
            actions.append(
                {
                    "task_id": task_id,
                    "seed": before["seed"],
                    "observed_status": observed_status,
                    "result": "terminal_preserved_before_post",
                    "cancel_post_performed": False,
                }
            )
            continue
        if observed_status not in CANCELLABLE_STATUSES:
            raise CutoverError("non-cancellable legacy status reached apply")
        try:
            response = client.cancel_task(task_id, expected_status=observed_status)
        except ConditionalCancelRejected:
            after_rejection = _authenticate_old_task(client, task_id)
            if after_rejection["status"] not in TERMINAL_STATUSES:
                raise
            actions.append(
                {
                    "task_id": task_id,
                    "seed": before["seed"],
                    "observed_status": observed_status,
                    "result": "terminal_preserved_after_cas_rejection",
                    "final_status": after_rejection["status"],
                    "cancel_post_performed": False,
                }
            )
            continue
        after = _authenticate_old_task(client, task_id)
        if after["status"] not in TERMINAL_STATUSES:
            raise CutoverError(
                f"conditional cancel {task_id} did not reach terminal state"
            )
        actions.append(
            {
                "task_id": task_id,
                "seed": before["seed"],
                "observed_status": observed_status,
                "endpoint": (
                    f"/api/tasks/{task_id}/cancel?"
                    f"expected_statuses={urllib.parse.quote(observed_status)}"
                ),
                "response_sha256": _sha256_value(response),
                "result": (
                    "conditionally_cancelled"
                    if after["status"] == "cancelled"
                    else "terminal_preserved_after_status_race"
                ),
                "final_status": after["status"],
                "cancel_post_performed": True,
            }
        )

    final_old_audit = audit_old(client)
    final_by_id = {row["task_id"]: row for row in final_old_audit["rows"]}
    if any(
        final_by_id[task_id]["status"] not in TERMINAL_STATUSES
        for task_id in selected_from_pre
    ):
        raise CutoverError("selected legacy row remained nonterminal")
    post = _seal(
        {
            "schema_version": POST_CUTOVER_SCHEMA,
            "observed_at": _now(),
            "apply": True,
            "new_authority": binding,
            "pre_cutover_receipt_path": str(pre_path),
            "pre_cutover_receipt_file_sha256": _sha256_file(pre_path),
            "pre_cutover_receipt_payload_sha256": pre["payload_sha256"],
            "selected_conditional_cancel_task_ids": selected_from_pre,
            "actions": actions,
            "final_old_audit_payload_sha256": final_old_audit["payload_sha256"],
            "final_old_rows": final_old_audit["rows"],
            "conditional_cancelled_task_ids": [
                row["task_id"]
                for row in actions
                if row["result"] == "conditionally_cancelled"
            ],
            "terminal_preserved_task_ids": [
                row["task_id"]
                for row in final_old_audit["rows"]
                if row["status"] != "cancelled"
            ],
            "out_of_range_cancel_count": 0,
            "mismatched_cancel_count": 0,
            "terminal_cancel_count": 0,
            "scheduler_get_count": client.get_count,
            "scheduler_html_get_count": client.html_get_count,
            "scheduler_cancel_count": client.cancel_count,
        }
    )
    _immutable_json(post_path, post)
    return post


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-url", default=SCHEDULER_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    old = subparsers.add_parser("audit-old")
    old.add_argument("--out", type=Path, required=True)

    attest = subparsers.add_parser("attest-new")
    attest.add_argument("--new-receipt", type=Path, required=True)
    attest.add_argument("--new-profile-sha256", required=True)
    attest.add_argument("--out", type=Path, required=True)

    cutover = subparsers.add_parser("cutover")
    cutover.add_argument("--new-receipt", type=Path, required=True)
    cutover.add_argument("--new-readback-evidence", type=Path, required=True)
    cutover.add_argument("--output-root", type=Path, required=True)
    cutover.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    client = SchedulerClient(args.scheduler_url)
    if args.command == "audit-old":
        value = audit_old(client)
        _immutable_json(args.out, value)
    elif args.command == "attest-new":
        value = attest_new_readback(
            new_receipt_path=args.new_receipt,
            new_profile_sha256=args.new_profile_sha256,
            evidence_out=args.out,
            scheduler=client,
        )
    else:
        value = run_cutover(
            new_receipt_path=args.new_receipt,
            new_readback_evidence_path=args.new_readback_evidence,
            output_root=args.output_root,
            apply=args.apply,
            scheduler=client,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
