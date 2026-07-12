"""Fail-closed 3 -> 10 -> 50 -> 300 campaign promotion controller.

The default command is read-only.  ``--execute`` is required before this tool
may submit a pinned pilot stage or ask :mod:`feeder` to refill production work.
It never cancels tasks.  The controller deliberately delegates candidate
generation, submission, capacity limits, ledgers, and strict result validation
to the existing campaign modules instead of introducing a second submit path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import requests
from filelock import FileLock


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
sys.path.insert(0, str(REGRESSION_ROOT))
sys.path.insert(0, str(REGRESSION_ROOT / "verify"))

import feeder
import deployment_gate
import pinned_pilot
import provisional_wave
import scheduler_client


SCHEMA_VERSION = 1
DEFAULT_SEED = 260710
# Operationally unlimited while retaining feeder.step's integer hard-cap API.
# The campaign must continue past 10k until an operator stops the controller.
DEFAULT_MAX_SAMPLES = 2_000_000_000
DEFAULT_LOOP_SECONDS = 60
CANDIDATE_AUDIT_COUNT = 300
PILOT_EARLY_VALID = 5
FLEET_GATE_TERMINAL = 20
FLEET_GATE_VALID_RATE = 0.90
RECENT_WINDOW = 30
RECENT_MIN_VALID_RATE = 0.70
REPEATED_ERROR_LIMIT = 3
# Lifetime failures remain invalid/accounted, but only a current concentration
# or current trailing streak is evidence of an ongoing runtime outage.
REPEATED_ERROR_MIN_RECENT_RATIO = 0.20
REPEATED_ERROR_TRAILING_LIMIT = 3
DATA_STALL_SECONDS = 90 * 60
MAX_STANDARD_TIMEOUT_SECONDS = 2 * 60 * 60
ACTIVE_STATUSES = frozenset(("queued", "attaching", "running"))
TERMINAL_STATUSES = frozenset(("completed", "failed", "cancelled"))
STATE_DIR_ENV = "MFT_RAPID_CAMPAIGN_STATE_DIR"
TEMPERATURE_PATTERN = re.compile(r"^(?:T_(?:max|mean)_|Tprobe_)")
FAILURE_STDERR_TAIL_LINES = 200
FAILURE_STDERR_MAX_BYTES = 65_536

STAGE_LOCAL3 = "local3"
STAGE_PILOT10 = "pilot10"
STAGE_FLEET50 = "fleet50"
STAGE_PRODUCTION300 = "production300"
STAGE_TARGETS = {
    STAGE_LOCAL3: 3,
    STAGE_PILOT10: 10,
    STAGE_FLEET50: 50,
    STAGE_PRODUCTION300: 300,
}


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_time(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def default_state_path(solver_revision, library_revision, seed=DEFAULT_SEED):
    override = os.environ.get(STATE_DIR_ENV, "").strip()
    root = (
        Path(override).expanduser().resolve()
        if override else pinned_pilot.campaign_manifest_dir())
    name = (
        f"rapid-s{solver_revision[:12]}-l{library_revision[:12]}-"
        f"seed{int(seed)}.json")
    return root / name


def new_state(solver_revision, library_revision, seed=DEFAULT_SEED):
    return {
        "schema_version": SCHEMA_VERSION,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "seed": int(seed),
        "stage": STAGE_LOCAL3,
        "target_active": 0,
        "paused": False,
        "pause_reasons": [],
        "candidate_audit": None,
        "last_dataset_rows": None,
        "last_dataset_growth_at": None,
        "first_production_valid_at": None,
        "last_production_valid_at": None,
        "task_outcomes": {},
        "updated_at": None,
    }


def load_state(path, solver_revision, library_revision, seed=DEFAULT_SEED):
    path = Path(path)
    if not path.is_file():
        return new_state(solver_revision, library_revision, seed)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"rapid campaign state is unreadable: {path}: {exc}") from exc
    expected = {
        "schema_version": SCHEMA_VERSION,
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "seed": int(seed),
    }
    mismatches = {
        key: (state.get(key), value)
        for key, value in expected.items() if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"rapid campaign state identity mismatch: {mismatches}")
    if state.get("stage") not in STAGE_TARGETS:
        raise RuntimeError(f"rapid campaign state has an invalid stage: {state.get('stage')!r}")
    if not isinstance(state.get("pause_reasons", []), list):
        raise RuntimeError("rapid campaign pause_reasons must be a list")
    if not isinstance(state.get("task_outcomes", {}), dict):
        raise RuntimeError("rapid campaign task_outcomes must be an object")
    return state


def save_state(state, path):
    state = dict(state)
    state["updated_at"] = _iso(_now())
    pinned_pilot._atomic_manifest(state, Path(path))


def candidate_supply_audit(
        solver_revision, library_revision, seed=DEFAULT_SEED,
        count=CANDIDATE_AUDIT_COUNT):
    """Pre-generate and authenticate the first production candidate tranche."""
    profile = provisional_wave._load_profile()
    try:
        timeout_seconds = int(profile["timeout_seconds"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("standard profile has no valid timeout_seconds") from exc
    if not 0 < timeout_seconds <= MAX_STANDARD_TIMEOUT_SECONDS:
        raise RuntimeError(
            "standard profile timeout must be at most 7200 seconds before rapid "
            f"promotion, got {timeout_seconds}")
    records = provisional_wave.build_plan(
        solver_revision, library_revision, profile, seed=seed, count=count)
    expected = provisional_wave.new_manifest(
        solver_revision, library_revision, profile, records, seed=seed)
    provisional_wave.validate_manifest(expected, expected)
    raw_indices = [record["candidate_raw_index"] for record in records]
    digests = [record["params_sha256"] for record in records]
    names = [record["name"] for record in records]
    if (len(set(raw_indices)) != count or len(set(digests)) != count
            or len(set(names)) != count):
        raise RuntimeError("prevalidated production candidates are not unique")
    payload = [provisional_wave._plan_identity(record) for record in records]
    plan_sha256 = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "count": count,
        "first_raw_index": raw_indices[0],
        "last_raw_index": raw_indices[-1],
        "plan_sha256": plan_sha256,
    }


def production_prefix(solver_revision, library_revision):
    return f"mft-camp-s{solver_revision[:7]}-l{library_revision[:7]}-"


def is_feeder_task(task, solver_revision, library_revision):
    prefix = re.escape(production_prefix(solver_revision, library_revision))
    return bool(re.fullmatch(prefix + r"\d{5,}", str(task.get("name") or "")))


def thermal_saturation_columns(result):
    """Return temperature fields at or above the Icepak trust ceiling."""
    if not isinstance(result, dict):
        return []
    saturated = []
    for key, value in result.items():
        if not TEMPERATURE_PATTERN.match(str(key)):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(numeric) and numeric >= scheduler_client.MAX_TRUSTED_TEMPERATURE_C:
            saturated.append(str(key))
    return sorted(saturated)


def invalid_result_reason(result, solver_revision, library_revision, fetch_state=None):
    saturated = thermal_saturation_columns(result)
    if saturated:
        return "thermal_saturation:" + ",".join(saturated)
    if not isinstance(result, dict):
        return f"result_{fetch_state or 'missing'}"
    solver_hash = str(result.get("git_hash") or "").strip().lower()
    if solver_hash and solver_hash != solver_revision:
        return "solver_revision_mismatch"
    library_hash = str(result.get("pyaedt_library_git_hash") or "").strip().lower()
    if library_hash and library_hash != library_revision:
        return "library_revision_mismatch"
    return f"result_{fetch_state or 'strict_invalid'}"


_LEGACY_FAILURE_FIELDS = (
    "failure_reason", "error_message", "error", "status_reason", "message",
    "failure_message", "exit_code", "status",
)
_LEGACY_FAILURE_FIELD_PATTERN = "|".join(
    re.escape(field) for field in _LEGACY_FAILURE_FIELDS
)
_AEDT_SESSION_CLEANUP_MESSAGE = re.compile(
    rf"^\s*(?:(?:{_LEGACY_FAILURE_FIELD_PATTERN}|stderr_[a-z0-9_]+)\s*=\s*)?"
    r"(?:(?:PyAEDT\s+)?ERROR:(?:Global:|root:)?\s*)?"
    r"(?:A\(n\)\s+<class\s+['\"]TypeError['\"]>\s+)?"
    r"error occurred while retrieving information for the active AEDT sessions:\s*"
    r"(?:argument of type\s+['\"]NoneType['\"]\s+is not iterable|"
    r"['\"]NoneType['\"]\s+object is not iterable)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _is_aedt_session_cleanup_message(message):
    """Identify the known post-close PyAEDT session-enumeration error."""
    return _AEDT_SESSION_CLEANUP_MESSAGE.fullmatch(
        str(message or "").strip()
    ) is not None


def _without_aedt_session_cleanup_lines(message):
    """Drop post-close cleanup noise without hiding adjacent solve failures."""
    text = str(message or "").strip()
    if not text or _is_aedt_session_cleanup_message(text):
        return ""
    return "\n".join(
        line for line in text.splitlines()
        if not _is_aedt_session_cleanup_message(line)
    ).strip()


def _legacy_failure_payloads(text):
    """Return payloads from the controller's historical ``key=value`` format."""
    parts = re.split(
        rf"\s+\|\s+(?=(?:{_LEGACY_FAILURE_FIELD_PATTERN})\s*=)",
        text,
        flags=re.IGNORECASE,
    )
    payloads = []
    for part in parts:
        match = re.fullmatch(
            rf"\s*({_LEGACY_FAILURE_FIELD_PATTERN})\s*=\s*(.*?)\s*",
            part,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            payloads.append(part.strip())
            continue
        field, value = match.groups()
        if field.lower() not in ("exit_code", "status"):
            payloads.append(value.strip())
    return [payload for payload in payloads if payload]


def _is_informative_runtime_payload(payload):
    text = _without_aedt_session_cleanup_lines(payload)
    if not text:
        return False
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    if lowered in {
            "error", "failed", "failure", "task failed", "unknown",
            "cancelled", "canceled"}:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and all(re.match(
            r"^(?:PyAEDT\s+)?(?:INFO|DEBUG)(?::|\s)",
            line,
            flags=re.IGNORECASE,
    ) for line in lines):
        return False
    return True


def _is_informative_runtime_message(message):
    """Reject status/preamble-only text while retaining genuine error payloads."""
    text = str(message or "").strip()
    if not text:
        return False
    return any(
        _is_informative_runtime_payload(payload)
        for payload in _legacy_failure_payloads(text)
    )


def _is_expected_sample_nonconvergence(message):
    """True for a completed thermal solve that only missed residual limits.

    This remains an invalid training sample and still lowers fleet/recent valid
    rates.  It is not an infrastructure/runtime fingerprint that should stop
    the entire campaign merely because several difficult geometries reach the
    same numerical rejection.
    """
    return re.search(
        r"\[thermal\]\s+solve rejected before extraction:.*"
        r"\banalyze-call-ok\s*=\s*true\b.*"
        r"\bconverged\s*=\s*0\b.*"
        r"\breason\s*=\s*residual_threshold\b",
        str(message or ""),
        re.IGNORECASE,
    ) is not None


def _structured_failure_message(task):
    parts = []
    for key in (
            "failure_reason", "error_message", "error", "status_reason",
            "message", "failure_message"):
        value = task.get(key)
        if value in (None, "", []):
            continue
        value_text = str(value).strip()
        if _is_informative_runtime_message(value_text):
            parts.append(f"{key}={value_text}")
    return " | ".join(parts)


def _stderr_failure_message(stderr):
    """Extract one stable, high-signal cause from a bounded stderr tail."""
    lines = [line.strip() for line in str(stderr or "").splitlines() if line.strip()]
    if not lines:
        return ""
    # Thermal rejection is the primary solve failure.  PyAEDT can emit a later
    # active-session TypeError while closing Desktop; choosing the last generic
    # ERROR line would then split identical monitor_missing failures across two
    # fingerprints.  Preserve the existing stderr_pyaedt message form so the
    # already-cached fingerprint remains stable.
    thermal_rejection = re.compile(
        r"^(?:PyAEDT\s+)?ERROR:(?:Global:|root:)?\s*"
        r"(\[thermal\]\s+solve rejected before extraction:\s*.+)$",
        re.IGNORECASE,
    )
    for line in reversed(lines):
        match = thermal_rejection.search(line)
        if not match:
            continue
        message = f"stderr_pyaedt={match.group(1).strip()}"
        if _is_informative_runtime_message(message):
            return message[:4000]
    patterns = (
        ("run_one_loop", re.compile(
            r"(?:ERROR:root:)?run_one_loop failed:\s*(.+)", re.IGNORECASE)),
        ("exception", re.compile(
            r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.+)$")),
        ("pyaedt", re.compile(
            r"^(?:PyAEDT\s+)?ERROR:(?:Global:|root:)?\s*(.+)$",
            re.IGNORECASE)),
        ("fatal", re.compile(r"^fatal:\s*(.+)$", re.IGNORECASE)),
        ("srun", re.compile(
            r"^srun:\s*error:\s*(.+(?:exited|killed|timeout|timed out).*)$",
            re.IGNORECASE)),
    )
    for label, pattern in patterns:
        for line in reversed(lines):
            match = pattern.search(line)
            if not match:
                continue
            detail = " ".join(match.groups()).strip()
            message = f"stderr_{label}={detail}"
            if _is_informative_runtime_message(message):
                return message[:4000]
    return ""


def _fetch_task_stderr(task_id):
    response = requests.get(
        f"{scheduler_client.SCHEDULER}/api/tasks/{int(task_id)}/stderr",
        params={
            "tail_lines": FAILURE_STDERR_TAIL_LINES,
            "max_bytes": FAILURE_STDERR_MAX_BYTES,
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.text


def _failure_message(task, stderr_fetcher=None, allow_stderr=True):
    message = _structured_failure_message(task)
    cleanup_message = bool(message) and _is_aedt_session_cleanup_message(message)
    if message and not cleanup_message:
        return message
    task_id = task.get("id", task.get("task_id"))
    if (allow_stderr and not isinstance(task_id, bool) and isinstance(task_id, int)
            and task_id > 0):
        fetcher = stderr_fetcher or _fetch_task_stderr
        try:
            stderr_message = _stderr_failure_message(fetcher(task_id))
        except Exception:
            stderr_message = ""
        if stderr_message and (
                not message
                or "[thermal] solve rejected before extraction:" in stderr_message.lower()
        ):
            return stderr_message
    if message:
        return message
    exit_code = task.get("exit_code")
    if exit_code not in (None, "", []):
        return f"exit_code={exit_code}"
    return f"status={task.get('status', 'failed')}"


def error_fingerprint(message):
    normalized = str(message or "unknown").strip().lower()
    normalized = re.sub(r"[0-9a-f]{12,}", "<hex>", normalized)
    normalized = re.sub(r"\b\d+\b", "<n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _runtime_error_fingerprint(message):
    message = _without_aedt_session_cleanup_lines(message)
    if not _is_informative_runtime_message(message):
        return None
    return error_fingerprint(message)


_CANDIDATE_JSON_COMMAND = re.compile(
    r"printf\s+['\"]%s['\"]\s+['\"](?P<params>\{.*?\})['\"]"
    r"\s*>\s*cand\.json",
    re.DOTALL,
)
_SINGLETON_RX_VOLUME_MISSING = re.compile(
    r"\[thermal\]\s+validation failed:\s*"
    r"field-summary-data\s*=\s*true\s*,\s*"
    r"required-missing\s*=\s*1\s*,\s*"
    r"missing-total\s*=\s*2\s*,\s*"
    r"analyze-call-ok\s*=\s*true\b",
    re.IGNORECASE,
)
_SEALED_B171_SINGLETON_RX_FAILURE_IDS = frozenset((28_522, 28_743, 28_780))
_SEALED_B171_PREFIX = "mft-camp-sb171c7c-le6b9b9d-"
_SCHEDULER_CPU_POLICY_CUTOVER_UTC = "2026-07-12 11:00:54"
_OPERATOR_CANCELLED_STALE_PREPOLICY_LAUNCHES = {
    28_746: {
        "name": "mft-camp-sb171c7c-le6b9b9d-18221",
        "created_at": "2026-07-12 10:59:54",
        "attached_at": "2026-07-12 11:00:46",
        "launch_started_at": "2026-07-12 11:00:47",
        "finished_at": "2026-07-12 14:24:22",
    },
    28_747: {
        "name": "mft-camp-sb171c7c-le6b9b9d-18222",
        "created_at": "2026-07-12 10:59:58",
        "attached_at": "2026-07-12 11:00:47",
        "launch_started_at": "2026-07-12 11:00:48",
        "finished_at": "2026-07-12 14:24:21",
    },
    28_748: {
        "name": "mft-camp-sb171c7c-le6b9b9d-18223",
        "created_at": "2026-07-12 11:00:04",
        "attached_at": "2026-07-12 11:00:48",
        "launch_started_at": "2026-07-12 11:00:49",
        "finished_at": "2026-07-12 14:24:21",
    },
}
_SEALED_B171_SOLVER_REVISION = "b171c7ce5f7a018be6a575a32b1a1f5b7caa980c"
_SEALED_B171_LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
# Exact MFT tasks whose four parent allocations were cancelled together at
# 2026-07-12 14:39:33Z. A future exit-143 or a different immutable identity
# must remain an ordinary fail-closed runtime failure.
_RESOLVED_SCHEDULER_PARENT_CANCEL_ROWS = (
    (29026, 18395, "53f702e3c9af07e3", 8347, "732197", "n003", "2026-07-12 13:00:59", "2026-07-12 14:43:45"),
    (29037, 18403, "4f16ea5397bc114e", 8349, "732200", "n002", "2026-07-12 13:01:07", "2026-07-12 14:44:21"),
    (29038, 18404, "529610606a46c297", 8345, "732195", "n014", "2026-07-12 13:01:06", "2026-07-12 14:44:21"),
    (29047, 18413, "2ceff7844b9c1a5a", 8346, "732196", "n007", "2026-07-12 13:02:03", "2026-07-12 14:44:21"),
    (29054, 18419, "99f24eeb3095fc02", 8349, "732200", "n002", "2026-07-12 13:02:08", "2026-07-12 14:44:22"),
    (29055, 18420, "3a00ff710661030c", 8345, "732195", "n014", "2026-07-12 13:02:09", "2026-07-12 14:44:22"),
    (29061, 18426, "17502fafe1a5cf20", 8346, "732196", "n007", "2026-07-12 13:03:27", "2026-07-12 14:44:22"),
    (29069, 18433, "7be075e2aa04ffe4", 8345, "732195", "n014", "2026-07-12 13:03:31", "2026-07-12 14:44:22"),
    (29075, 18439, "68e5d02c8a759786", 8346, "732196", "n007", "2026-07-12 13:04:24", "2026-07-12 14:44:22"),
    (29095, 18458, "3f4f3615c476cc35", 8345, "732195", "n014", "2026-07-12 13:05:36", "2026-07-12 14:45:06"),
    (29096, 18459, "99a65186e107158b", 8346, "732196", "n007", "2026-07-12 13:05:37", "2026-07-12 14:45:06"),
    (29100, 18463, "471eef1a9e310db9", 8347, "732197", "n003", "2026-07-12 13:07:55", "2026-07-12 14:45:07"),
    (29106, 18469, "1bb67807e77ed70a", 8349, "732200", "n002", "2026-07-12 13:08:03", "2026-07-12 14:45:07"),
    (29117, 18480, "a257a621f785c38f", 8349, "732200", "n002", "2026-07-12 13:08:22", "2026-07-12 14:45:07"),
    (29187, 18500, "fa36454d9cd507c4", 8345, "732195", "n014", "2026-07-12 13:18:16", "2026-07-12 14:46:27"),
    (29188, 18501, "c2ceb39f453840c7", 8345, "732195", "n014", "2026-07-12 13:18:20", "2026-07-12 14:46:27"),
    (29191, 18504, "22de86b53098e296", 8346, "732196", "n007", "2026-07-12 13:18:26", "2026-07-12 14:46:27"),
    (29193, 18506, "afd3a36c48013faf", 8349, "732200", "n002", "2026-07-12 13:18:31", "2026-07-12 14:46:28"),
    (29194, 18507, "1fdea70e98a4f930", 8349, "732200", "n002", "2026-07-12 13:18:35", "2026-07-12 14:46:28"),
    (29197, 18510, "4bd495ddb787737c", 8345, "732195", "n014", "2026-07-12 13:20:45", "2026-07-12 14:46:28"),
    (29264, 18538, "137ff85168b8e059", 8349, "732200", "n002", "2026-07-12 14:05:09", "2026-07-12 14:40:19"),
    (29266, 18540, "4f1ea95bee191e48", 8349, "732200", "n002", "2026-07-12 14:05:10", "2026-07-12 14:40:19"),
    (29270, 18544, "f1438518f870fd21", 8349, "732200", "n002", "2026-07-12 14:06:06", "2026-07-12 14:40:20"),
    (29273, 18547, "129e4e4392e120b9", 8349, "732200", "n002", "2026-07-12 14:06:11", "2026-07-12 14:40:20"),
    (29290, 18563, "5e6a1ca5623f708b", 8349, "732200", "n002", "2026-07-12 14:15:15", "2026-07-12 14:42:28"),
    (29295, 18568, "0e664af6168a3e17", 8349, "732200", "n002", "2026-07-12 14:16:41", "2026-07-12 14:42:28"),
)
_RESOLVED_SCHEDULER_PARENT_CANCEL_TASKS = {}
for (_task_id, _serial, _dedupe_suffix, _allocation_id, _slurm_job_id,
     _node_name, _started_at, _finished_at) in (
        _RESOLVED_SCHEDULER_PARENT_CANCEL_ROWS):
    _name = f"{_SEALED_B171_PREFIX}{_serial}"
    _RESOLVED_SCHEDULER_PARENT_CANCEL_TASKS[_task_id] = {
        "name": _name,
        "dedupe_key": (
            f"mft-al:{_name}:{_SEALED_B171_SOLVER_REVISION}:"
            f"{_SEALED_B171_LIBRARY_REVISION}:{_dedupe_suffix}"),
        "allocation_id": _allocation_id,
        "slurm_job_id": _slurm_job_id,
        "allocation_node_name": _node_name,
        "started_at": _started_at,
        "finished_at": _finished_at,
    }
del (_task_id, _serial, _dedupe_suffix, _allocation_id, _slurm_job_id,
     _node_name, _started_at, _finished_at, _name)
_SEALED_OLD_TIMEOUT_CONTRACT_ROWS = (
    (28749, 18224, "e16235f02fdc0064", 8034, "731400", "n046", "2026-07-12 11:04:04", "2026-07-12 15:05:23", "jji0930"),
    (28773, 18248, "6df3ddc87b54d96f", 8293, "731975", "n116", "2026-07-12 11:05:49", "2026-07-12 15:09:22", "dhj02"),
    (28789, 18263, "03f494eb3e1a6bae", 8004, "731339", "n041", "2026-07-12 11:05:59", "2026-07-12 15:09:27", "jji0930"),
    (28819, 18292, "dec009e7dd4152db", 8212, "731735", "n090", "2026-07-12 11:25:25", "2026-07-12 15:28:32", "jji0930"),
    (28830, 18303, "b18919fa3a1e70b7", 8311, "732022", "n111", "2026-07-12 11:26:20", "2026-07-12 15:29:20", "jji0930"),
    (28842, 18315, "aa38ecbc17d90435", 8289, "731970", "n082", "2026-07-12 11:27:19", "2026-07-12 15:29:29", "jji0930"),
)
_SEALED_OLD_TIMEOUT_CONTRACT_TASKS = {}
for (_task_id, _serial, _dedupe_suffix, _allocation_id, _slurm_job_id,
     _node_name, _started_at, _finished_at, _account_name) in (
        _SEALED_OLD_TIMEOUT_CONTRACT_ROWS):
    _name = f"{_SEALED_B171_PREFIX}{_serial}"
    _SEALED_OLD_TIMEOUT_CONTRACT_TASKS[_task_id] = {
        "name": _name,
        "dedupe_key": (
            f"mft-al:{_name}:{_SEALED_B171_SOLVER_REVISION}:"
            f"{_SEALED_B171_LIBRARY_REVISION}:{_dedupe_suffix}"),
        "allocation_id": _allocation_id,
        "slurm_job_id": _slurm_job_id,
        "allocation_node_name": _node_name,
        "started_at": _started_at,
        "finished_at": _finished_at,
        "account_name": _account_name,
    }
del (_task_id, _serial, _dedupe_suffix, _allocation_id, _slurm_job_id,
     _node_name, _started_at, _finished_at, _account_name, _name)

_FOUR_HOUR_TIMEOUT_MESSAGE = re.compile(
    r"^(?:failure_message=)?task timed out after 14400s$", re.IGNORECASE,
)


def _scheduler_timestamp(value):
    text = str(value or "").strip().replace("T", " ")
    text = text.removesuffix("Z")
    if not re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text):
        return None
    return text[:19]


def _is_operator_cancelled_stale_prepolicy_launch(task):
    """Match only the three operator-reviewed stale pre-policy launches."""
    task_id = task.get("id", task.get("task_id"))
    expected = _OPERATOR_CANCELLED_STALE_PREPOLICY_LAUNCHES.get(task_id)
    if expected is None:
        return False
    launch_started_at = _scheduler_timestamp(task.get("launch_started_at"))
    return (
        all(task.get(field) == value for field, value in expected.items())
        and task.get("status") == "cancelled"
        and task.get("started_at") in (None, "")
        and task.get("exit_code") in (None, "")
        and task.get("failure_message") in (None, "")
        and task.get("project") == "MFT_1MW_2026v1"
        and task.get("account_name") == "r1jae262"
        and task.get("requested_account_name") == ""
        and task.get("allocation_id") == 8_019
        and str(task.get("slurm_job_id") or "") == "731354"
        and task.get("allocation_node_name") == "n045"
        and launch_started_at is not None
        and launch_started_at < _SCHEDULER_CPU_POLICY_CUTOVER_UTC
    )


def _is_resolved_scheduler_parent_cancel_incident(task):
    """Match only the exact diagnosed four-parent Slurm cancellation burst."""
    task_id = task.get("id", task.get("task_id"))
    expected = _RESOLVED_SCHEDULER_PARENT_CANCEL_TASKS.get(task_id)
    return bool(
        expected is not None
        and all(task.get(field) == value for field, value in expected.items())
        and task.get("status") == "failed"
        and task.get("exit_code") in (143, "143")
        and task.get("project") == "MFT_1MW_2026v1"
        and task.get("account_name") == "dw16"
        and task.get("requested_account_name") == ""
        and task.get("scheduling_profile") == "fea_bursty"
        and task.get("timeout_seconds") in (14_400, "14400")
    )


def _is_sealed_old_timeout_contract_incident(task, message):
    """Match only six reviewed tasks on allocations born before cutover."""
    task_id = task.get("id", task.get("task_id"))
    expected = _SEALED_OLD_TIMEOUT_CONTRACT_TASKS.get(task_id)
    return bool(
        expected is not None
        and all(task.get(field) == value for field, value in expected.items())
        and task.get("status") == "failed"
        and task.get("exit_code") in (124, "124")
        and task.get("project") == "MFT_1MW_2026v1"
        and task.get("requested_account_name") == ""
        and task.get("scheduling_profile") == "fea_bursty"
        and task.get("timeout_seconds") in (14_400, "14400")
        and _FOUR_HOUR_TIMEOUT_MESSAGE.fullmatch(
            str(message or "").strip()) is not None
    )


def _expected_failed_sample_reason(task, message):
    """Classify only sealed, diagnosed b171 failures that remain invalid.

    The invalid result remains fail-closed.  This merely prevents three or
    more copies of the already diagnosed ``N2_side == 1`` mesh/extraction
    limitation from being mistaken for a campaign-wide runtime outage.
    """
    if _is_operator_cancelled_stale_prepolicy_launch(task):
        return "operator_cancelled_stale_prepolicy_launch"
    if _is_resolved_scheduler_parent_cancel_incident(task):
        return "resolved_scheduler_parent_cancel_incident"
    if _is_sealed_old_timeout_contract_incident(task, message):
        return "sealed_old_timeout_contract_incident"

    # These jobs started before the scheduler's allocation-local affinity and
    # adaptive admission policy was activated.  Their late four-hour timeout
    # arrivals are invalid data, but cannot diagnose the post-cutover runtime
    # cohort.  Future or timestamp-less timeouts remain stopping failures.
    started_at = _scheduler_timestamp(task.get("started_at"))
    if (str(task.get("name") or "").startswith(_SEALED_B171_PREFIX)
            and task.get("exit_code") in (124, "124")
            and _FOUR_HOUR_TIMEOUT_MESSAGE.fullmatch(str(message or "").strip())
            and started_at is not None
            and started_at < _SCHEDULER_CPU_POLICY_CUTOVER_UTC):
        return "scheduler_prepolicy_timeout"

    if _SINGLETON_RX_VOLUME_MISSING.search(str(message or "")) is None:
        return None
    # The scheduler list API deliberately omits command payloads.  These exact
    # three identities were independently audited against their stored
    # ``cand.json`` commands and all have N2_side == 1.  The allow-list is
    # incident-specific; any later extraction omission still stops refill.
    task_id = task.get("id", task.get("task_id"))
    if task_id in _SEALED_B171_SINGLETON_RX_FAILURE_IDS:
        return "singleton_rx_side_volume_missing"
    match = _CANDIDATE_JSON_COMMAND.search(str(task.get("command") or ""))
    if match is None:
        return None
    try:
        params = json.loads(match.group("params"))
        n2_side = int(params.get("N2_side"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return "singleton_rx_side_volume_missing" if n2_side == 1 else None


def _refresh_failure_outcome(
        outcome, task, *, allow_remote_stderr=True):
    """Classify a terminal-invalid task from metadata by default at call sites.

    A scheduler ``failed`` or ``cancelled`` status can never become a valid
    training result.  Fleet reconciliation must therefore not serialize an
    unbounded historical stderr scan merely to enrich its diagnostic
    fingerprint.  Explicit incident-diagnostic callers may retain the legacy
    bounded stderr lookup by leaving ``allow_remote_stderr`` enabled.
    """
    status = str(task.get("status") or outcome.get("status") or "")
    cached_message = outcome.get("error_message")
    if _is_informative_runtime_message(cached_message):
        message = str(cached_message)
    else:
        message = _failure_message(
            task,
            allow_stderr=(status == "failed" and allow_remote_stderr),
        )
    outcome["error_message"] = message
    outcome["expected_failure_reason"] = _expected_failed_sample_reason(
        task, message) if status in ("failed", "cancelled") else None
    outcome["error_fingerprint"] = (
        None if outcome["expected_failure_reason"] else _runtime_error_fingerprint(message)
        if status == "failed" else None
    )
    return outcome


def _terminal_time(task, result=None):
    for value in (
            task.get("finished_at"), task.get("completed_at"),
            task.get("updated_at"),
            result.get("saved_at") if isinstance(result, dict) else None):
        parsed = _parse_time(value)
        if parsed is not None:
            return parsed
    return None


def inspect_production_tasks(
        tasks, solver_revision, library_revision, cached_outcomes=None):
    """Judge terminal feeder tasks for only the exact pinned generation."""
    cached_outcomes = dict(cached_outcomes or {})
    selected = [
        task for task in tasks
        if is_feeder_task(task, solver_revision, library_revision)
    ]
    outcomes = []
    seen_ids = set()
    for task in selected:
        task_id = task.get("id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id <= 0:
            raise RuntimeError("production inventory contains a task without a valid ID")
        status = str(task.get("status") or "")
        if status in ACTIVE_STATUSES:
            continue
        if status not in TERMINAL_STATUSES:
            raise RuntimeError(
                f"production task {task_id} has an unknown status: {status!r}")
        seen_ids.add(task_id)
        cached = cached_outcomes.get(str(task_id))
        if (isinstance(cached, dict)
                and cached.get("task_id") == task_id
                and cached.get("name") == task.get("name")
                and cached.get("status") == status):
            cached = dict(cached)
            if status != "completed":
                _refresh_failure_outcome(
                    cached, task, allow_remote_stderr=False)
                cached_outcomes[str(task_id)] = dict(cached)
            outcomes.append(cached)
            continue
        outcome = {
            "task_id": task_id,
            "name": task.get("name"),
            "status": status,
            "state": "invalid",
            "reason": None,
            "error_fingerprint": None,
            "error_message": None,
            "terminal_at": None,
            "saturation_columns": [],
        }
        result = None
        if status == "completed":
            try:
                fetched = scheduler_client.fetch_result(
                    task_id,
                    expected_revision=solver_revision,
                    expected_library_revision=library_revision,
                )
            except scheduler_client.ResultFetchError as exc:
                raise RuntimeError(
                    f"production task {task_id} result is unavailable: {exc}") from exc
            result = fetched.result
            saturated = thermal_saturation_columns(result)
            outcome["saturation_columns"] = saturated
            if (fetched.state == scheduler_client.RESULT_VALID
                    and scheduler_client.is_valid_result(
                        result,
                        expected_revision=solver_revision,
                        expected_library_revision=library_revision)):
                outcome["state"] = "valid"
            else:
                outcome["reason"] = invalid_result_reason(
                    result, solver_revision, library_revision, fetched.state)
        else:
            outcome["reason"] = f"task_{status}"
            _refresh_failure_outcome(
                outcome, task, allow_remote_stderr=False)
        terminal_at = _terminal_time(task, result)
        outcome["terminal_at"] = _iso(terminal_at) if terminal_at else None
        outcomes.append(outcome)
        cached_outcomes[str(task_id)] = dict(outcome)
    for key, cached in cached_outcomes.items():
        if not isinstance(cached, dict):
            raise RuntimeError(f"cached production outcome {key!r} is invalid")
        task_id = cached.get("task_id")
        if (isinstance(task_id, bool) or not isinstance(task_id, int)
                or task_id <= 0 or str(task_id) != str(key)
                or cached.get("status") not in TERMINAL_STATUSES
                or cached.get("state") not in ("valid", "invalid")):
            raise RuntimeError(f"cached production outcome {key!r} has invalid identity")
        if task_id in seen_ids:
            continue
        if not is_feeder_task(cached, solver_revision, library_revision):
            raise RuntimeError(
                f"cached production outcome {key!r} has the wrong generation")
        outcomes.append(dict(cached))
    outcomes.sort(key=lambda item: (
        _parse_time(item["terminal_at"]) or datetime.min.replace(tzinfo=timezone.utc),
        item["task_id"],
    ))
    active = sum(
        str(task.get("status") or "") in ACTIVE_STATUSES for task in selected)
    return {
        "tasks": selected,
        "active": active,
        "outcomes": outcomes,
        "cache": cached_outcomes,
    }


def _pilot_path(solver_revision, library_revision, stage, seed, manifest_dir=None):
    contract = pinned_pilot.PILOT_STAGE_CONTRACT[stage]
    tag = pinned_pilot.pilot_tag(
        solver_revision, library_revision, stage, seed, contract["offset"])
    root = (
        Path(manifest_dir) if manifest_dir is not None
        else pinned_pilot.campaign_manifest_dir())
    return root / f"{tag}.json"


def inspect_pilots(solver_revision, library_revision, seed, manifest_dir=None):
    stages = {}
    for stage in ("p02", "p08"):
        path = _pilot_path(
            solver_revision, library_revision, stage, seed, manifest_dir)
        if not path.is_file():
            stages[stage] = {"exists": False, "path": str(path), "outcomes": []}
            continue
        inspected = pinned_pilot.inspect_pilot_stage(
            solver_revision, library_revision, stage, seed,
            manifest_dir=manifest_dir)
        stages[stage] = {
            "exists": True,
            "path": str(inspected["path"]),
            "outcomes": inspected["outcomes"],
        }
    return stages


def _pilot_counts(pilots):
    outcomes = [
        outcome
        for stage in ("p02", "p08")
        for outcome in pilots[stage]["outcomes"]
    ]
    return {
        "valid": sum(item["state"] == "valid" for item in outcomes),
        "invalid": sum(item["state"] == "invalid" for item in outcomes),
        "pending": sum(item["state"] == "pending" for item in outcomes),
        "outcomes": outcomes,
    }


def _production_gate_reasons(production):
    outcomes = production["outcomes"]
    reasons = []
    saturated = [
        outcome for outcome in outcomes if outcome["saturation_columns"]]
    if saturated:
        reasons.append(
            "thermal_saturation_detected:" + ",".join(
                str(item["task_id"]) for item in saturated))
    revision_mismatches = [
        item for item in outcomes
        if item["reason"] in (
            "solver_revision_mismatch", "library_revision_mismatch")]
    if revision_mismatches:
        reasons.append(
            "revision_mismatch_detected:" + ",".join(
                str(item["task_id"]) for item in revision_mismatches))

    # A pre-policy task can finish hours after the scheduler cutover and land
    # inside the newest terminal window even though it says nothing about the
    # health of the current runtime policy.  It remains an invalid sample for
    # dataset/accounting purposes, but must not displace a post-cutover outcome
    # in this current-runtime health gate.  No other expected or failed sample
    # is excluded here.
    current_runtime_outcomes = [
        item for item in outcomes
        if item.get("expected_failure_reason") not in {
            "scheduler_prepolicy_timeout",
            "operator_cancelled_stale_prepolicy_launch",
            "resolved_scheduler_parent_cancel_incident",
            "sealed_old_timeout_contract_incident",
        }
    ]
    recent = current_runtime_outcomes[-RECENT_WINDOW:]

    def informative_fingerprint(item):
        fingerprint = item["error_fingerprint"]
        message = item.get("error_message")
        if (fingerprint
                and _is_informative_runtime_message(message)
                and not _is_expected_sample_nonconvergence(message)):
            return fingerprint
        return None

    fingerprints = Counter(filter(None, (
        informative_fingerprint(item) for item in recent)))
    trailing_fingerprint = None
    trailing_count = 0
    for item in reversed(recent):
        fingerprint = informative_fingerprint(item)
        if fingerprint is None:
            break
        if trailing_fingerprint is None:
            trailing_fingerprint = fingerprint
        elif fingerprint != trailing_fingerprint:
            break
        trailing_count += 1
    repeated = sorted(
        (fingerprint, count)
        for fingerprint, count in fingerprints.items()
        if (count >= REPEATED_ERROR_LIMIT
            and (
                count / len(recent) >= REPEATED_ERROR_MIN_RECENT_RATIO
                or (fingerprint == trailing_fingerprint
                    and trailing_count >= REPEATED_ERROR_TRAILING_LIMIT)
            )))
    for fingerprint, count in repeated:
        reasons.append(f"repeated_runtime_error:{fingerprint}:{count}")

    if len(current_runtime_outcomes) >= RECENT_WINDOW:
        valid_rate = sum(item["state"] == "valid" for item in recent) / RECENT_WINDOW
        if valid_rate < RECENT_MIN_VALID_RATE:
            reasons.append(
                f"recent_valid_rate_below_70pct:{valid_rate:.3f}")
    return reasons


def _update_progress(state, dataset_rows, production, now):
    previous_rows = state.get("last_dataset_rows")
    if previous_rows is not None and dataset_rows < int(previous_rows):
        return [f"dataset_row_count_regressed:{previous_rows}->{dataset_rows}"]
    if previous_rows is None or dataset_rows > int(previous_rows):
        state["last_dataset_rows"] = int(dataset_rows)
        state["last_dataset_growth_at"] = _iso(now)

    valid = [
        item for item in production["outcomes"] if item["state"] == "valid"]
    if valid:
        valid_times = [
            parsed for parsed in (_parse_time(item["terminal_at"]) for item in valid)
            if parsed is not None]
        earliest = min(valid_times) if valid_times else now
        latest = max(valid_times) if valid_times else now
        if not state.get("first_production_valid_at"):
            state["first_production_valid_at"] = _iso(earliest)
        state["last_production_valid_at"] = _iso(latest)
    if not state.get("first_production_valid_at"):
        return []

    first_valid = _parse_time(state.get("first_production_valid_at"))
    last_growth = _parse_time(state.get("last_dataset_growth_at"))
    anchors = [anchor for anchor in (first_valid, last_growth) if anchor is not None]
    if anchors and (now - max(anchors)).total_seconds() >= DATA_STALL_SECONDS:
        return ["valid_dataset_growth_stalled_90m"]
    return []


def decide_campaign(state, local_passed, pilots, production, dataset_rows, now=None):
    """Apply monotonic promotion and all fail-closed health gates."""
    now = now or _now()
    pilot = _pilot_counts(pilots)
    reasons = []
    if state.get("target_active", 0) >= 50 and not local_passed:
        reasons.append("local3_evidence_missing_after_promotion")
    if state.get("target_active", 0) >= 50 and (
            not pilots["p02"]["exists"] or not pilots["p08"]["exists"]):
        reasons.append("pilot_evidence_missing_after_promotion")
    for outcome in pilot["outcomes"]:
        saturated = thermal_saturation_columns(outcome.get("result"))
        if saturated:
            reasons.append(f"pilot_thermal_saturation:{outcome['task_id']}")
        if outcome["state"] == "invalid":
            reasons.append(
                f"pilot_invalid:{outcome['task_id']}:{outcome.get('reason')}")

    reasons.extend(_production_gate_reasons(production))
    reasons.extend(_update_progress(state, dataset_rows, production, now))

    stage = STAGE_LOCAL3
    target_active = 0
    action = "run_local_gate"
    if local_passed:
        stage = STAGE_PILOT10
        action = "submit_p02" if not pilots["p02"]["exists"] else "wait_p02"
        p02 = pilots["p02"]["outcomes"]
        p02_all_valid = (
            len(p02) == pinned_pilot.PILOT_STAGE_CONTRACT["p02"]["tasks"]
            and all(item["state"] == "valid" for item in p02))
        if p02_all_valid:
            action = "submit_p08" if not pilots["p08"]["exists"] else "wait_pilot10"
        if (p02_all_valid and pilot["valid"] >= PILOT_EARLY_VALID
                and pilot["invalid"] == 0):
            stage = STAGE_FLEET50
            target_active = 50
            action = "refill_50"

    terminal = len(production["outcomes"])
    valid = sum(item["state"] == "valid" for item in production["outcomes"])
    production_valid_rate = valid / terminal if terminal else None
    previous_target = int(state.get("target_active") or 0)
    if (previous_target < 300 and target_active >= 50
            and terminal >= FLEET_GATE_TERMINAL):
        if production_valid_rate >= FLEET_GATE_VALID_RATE:
            stage = STAGE_PRODUCTION300
            target_active = 300
            action = "refill_300"
        else:
            reasons.append(
                f"fleet20_valid_rate_below_90pct:{production_valid_rate:.3f}")

    target_active = max(target_active, previous_target)
    if target_active >= 300:
        stage = STAGE_PRODUCTION300
        action = "refill_300"
    elif target_active >= 50:
        stage = STAGE_FLEET50
        action = "refill_50"

    state["stage"] = stage
    state["target_active"] = target_active
    if reasons:
        state["paused"] = True
        state["pause_reasons"] = sorted(set(
            [*state.get("pause_reasons", []), *reasons]))
    if state.get("paused"):
        action = "manual_intervention"
    return {
        "stage": stage,
        "stage_target": STAGE_TARGETS[stage],
        "target_active": target_active,
        "action": action,
        "paused": bool(state.get("paused")),
        "pause_reasons": list(state.get("pause_reasons", [])),
        "dataset_rows": int(dataset_rows),
        "pilot": {
            "valid": pilot["valid"],
            "invalid": pilot["invalid"],
            "pending": pilot["pending"],
        },
        "production": {
            "active": production["active"],
            "terminal": terminal,
            "valid": valid,
            "invalid": terminal - valid,
            "valid_rate": production_valid_rate,
        },
    }


def _validate_pinned_local_revisions(solver_revision, library_revision, library_root=None):
    solver_revision = provisional_wave._validate_sha(
        solver_revision, "solver revision")
    library_revision = provisional_wave._validate_sha(
        library_revision, "library revision")
    provisional_wave._validate_local_revisions(
        solver_revision, library_revision, library_root=library_root)
    return solver_revision, library_revision


def run_once(
        solver_revision, library_revision, seed=DEFAULT_SEED,
        max_samples=DEFAULT_MAX_SAMPLES, execute=False, clear_pause=False,
        library_root=None, state_path=None, manifest_dir=None, now=None):
    if execute:
        with pinned_pilot.campaign_mutation_lock():
            return _run_once_locked(
                solver_revision, library_revision, seed=seed,
                max_samples=max_samples, execute=execute,
                clear_pause=clear_pause, library_root=library_root,
                state_path=state_path, manifest_dir=manifest_dir, now=now,
            )
    return _run_once_locked(
        solver_revision, library_revision, seed=seed,
        max_samples=max_samples, execute=execute,
        clear_pause=clear_pause, library_root=library_root,
        state_path=state_path, manifest_dir=manifest_dir, now=now,
    )


def _run_once_locked(
        solver_revision, library_revision, seed=DEFAULT_SEED,
        max_samples=DEFAULT_MAX_SAMPLES, execute=False, clear_pause=False,
        library_root=None, state_path=None, manifest_dir=None, now=None):
    """Observe once and optionally perform exactly one safe promotion/refill step."""
    now = now or _now()
    solver_revision, library_revision = _validate_pinned_local_revisions(
        solver_revision, library_revision, library_root=library_root)
    if execute:
        if not library_root:
            raise RuntimeError("execute requires a library root for deployment validation")
        # Recheck on every controller cycle. A long-running loop must stop
        # before mutation if either pinned commit ceases to be advertised.
        deployment_gate.validate_deployment(
            REPO_ROOT, solver_revision, library_root, library_revision
        )
    path = Path(state_path) if state_path else default_state_path(
        solver_revision, library_revision, seed)
    if execute:
        path.parent.mkdir(parents=True, exist_ok=True)
    state_lock = (
        FileLock(str(path) + ".lock", timeout=5) if execute else nullcontext())
    with state_lock:
        state = load_state(path, solver_revision, library_revision, seed)
        if clear_pause:
            if not execute:
                raise RuntimeError("--clear-pause requires --execute")
            state["paused"] = False
            state["pause_reasons"] = []

        if execute and not state.get("candidate_audit"):
            state["candidate_audit"] = candidate_supply_audit(
                solver_revision, library_revision, seed)

        local_manifest_dir = manifest_dir
        local_path = (
            Path(local_manifest_dir)
            if local_manifest_dir is not None
            else pinned_pilot.campaign_manifest_dir()) / (
                f"{pinned_pilot.local_gate_tag(solver_revision, library_revision)}.json")
        local_passed = False
        if local_path.is_file():
            pinned_pilot.validate_local_gate(
                solver_revision, library_revision,
                manifest_dir=local_path.parent)
            local_passed = True

        pilots = inspect_pilots(
            solver_revision, library_revision, seed,
            manifest_dir=manifest_dir)
        inventory = feeder.campaign_inventory()
        production = inspect_production_tasks(
            inventory, solver_revision, library_revision,
            cached_outcomes=state.get("task_outcomes"))
        state["task_outcomes"] = production["cache"]
        dataset_rows, judged_ids = feeder.dataset_collection_snapshot()
        decision = decide_campaign(
            state, local_passed, pilots, production, dataset_rows, now=now)

        mutation = None
        if execute and not decision["paused"]:
            if decision["action"] == "submit_p02":
                result = pinned_pilot.submit_pilot_stage(
                    solver_revision, library_revision, "p02", seed=seed,
                    execute=True, manifest_dir=manifest_dir,
                    library_root=library_root)
                mutation = {
                    "submitted": "p02",
                    "task_ids": [
                        record["task_id"] for record in result["manifest"]["tasks"]],
                }
            elif decision["action"] == "submit_p08":
                result = pinned_pilot.submit_pilot_stage(
                    solver_revision, library_revision, "p08", seed=seed,
                    execute=True, manifest_dir=manifest_dir,
                    library_root=library_root)
                mutation = {
                    "submitted": "p08",
                    "task_ids": [
                        record["task_id"] for record in result["manifest"]["tasks"]],
                }
            elif decision["target_active"] >= 50:
                authorization = feeder._authorize_rapid_refill(
                    decision,
                    max_samples=max_samples,
                    solver_revision=solver_revision,
                    library_revision=library_revision,
                    candidate_seed=seed,
                    local_passed=local_passed,
                    pilots_complete=bool(
                        pilots["p02"]["exists"] and pilots["p08"]["exists"]),
                )
                feeder._step_from_rapid_controller(
                    max_samples,
                    authorization=authorization,
                    target=decision["target_active"],
                    buffer=0,
                    solver_revision=solver_revision,
                    library_revision=library_revision,
                    candidate_seed=seed,
                )
                mutation = {
                    "refill_target": decision["target_active"],
                    "max_samples": int(max_samples),
                }
            save_state(state, path)

        return {
            "schema_version": SCHEMA_VERSION,
            "mode": "execute" if execute else "read_only",
            "solver_revision": solver_revision,
            "library_revision": library_revision,
            "seed": int(seed),
            "state_path": str(path),
            "local3_passed": local_passed,
            "candidate_audit": state.get("candidate_audit"),
            "collector_judged_tasks": len(judged_ids),
            "mutation": mutation,
            **decision,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fail-closed 3 -> 10 -> 50 -> 300 MFT campaign controller")
    parser.add_argument("--solver-revision", required=True)
    parser.add_argument("--library-revision", required=True)
    parser.add_argument("--library-root")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--state-path")
    parser.add_argument("--manifest-dir")
    parser.add_argument(
        "--execute", action="store_true",
        help="allow pinned pilot submission or feeder refill; never cancels")
    parser.add_argument(
        "--clear-pause", action="store_true",
        help="explicitly clear a latched pause before re-evaluating all gates")
    parser.add_argument(
        "--loop", type=int, default=None,
        help="repeat every N seconds for continuous refill (requires --execute)")
    args = parser.parse_args(argv)
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.loop is not None and args.loop <= 0:
        parser.error("--loop must be positive")
    if args.loop is not None and not args.execute:
        parser.error("--loop requires --execute")
    if args.execute:
        if not args.library_root:
            parser.error("--execute requires --library-root for remote deployment validation")

    while True:
        try:
            result = run_once(
                args.solver_revision,
                args.library_revision,
                seed=args.seed,
                max_samples=args.max_samples,
                execute=args.execute,
                clear_pause=args.clear_pause,
                library_root=args.library_root,
                state_path=args.state_path,
                manifest_dir=args.manifest_dir,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception as exc:
            # No feeder or submission call can occur after an observation error.
            print(json.dumps({
                "mode": "execute" if args.execute else "read_only",
                "paused": True,
                "action": "manual_intervention",
                "pause_reasons": [f"controller_error:{exc}"],
            }, ensure_ascii=False, sort_keys=True), flush=True)
            if args.loop is None:
                raise
        if args.loop is None:
            return
        args.clear_pause = False
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
