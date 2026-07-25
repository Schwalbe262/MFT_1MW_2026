"""Fail-closed, restart-safe refill of exact 4fac timeout12h slots.

Dry-run is the default.  A live cycle can issue at most one Scheduler POST and
only through the existing timeout12h exact-once atomic-claim submitter.  It
never calls a Scheduler cancel endpoint.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import paramiko


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_terminal_success_watcher as terminal_watcher  # noqa: E402
from tools import mft_goal_timeout12h_retry as timeout12h  # noqa: E402


PLAN_SCHEMA = "mft-goal-safe-refill-plan-v1"
EXTENSION_AUTHORITY_SCHEMA = "mft-goal-watcher-extension-authority-v1"
EXTENSION_RECEIPT_SCHEMA = "mft-goal-watcher-authorized-extension-v1"
INTENT_SCHEMA = "mft-goal-safe-refill-submit-intent-v1"
ACTION_SCHEMA = "mft-goal-safe-refill-action-receipt-v1"
STATE_SCHEMA = "mft-goal-safe-refill-state-v1"
SCHEDULER_URL = "http://127.0.0.1:8002"
PROJECT = "MFT_1MW_2026v1"
ACTIVE_STATUSES = frozenset({"queued", "attaching", "running"})
TARGET_ACCOUNT = "r1jae262"
TARGET_NODE = "n114"
TARGET_SCHEDULER_REVISION = "4facdfe36f74ff0679a477f6c25e1dddf26ce791"
TARGET_CUTOVER_SHA256 = (
    "45cf323de69a9a9c926c528d905ee4d09715b1a3d40c68349b9fa511646486bc"
)
FIXED_IDENTITY_SHA256 = (
    "08a6e426d261803660e933ad85985e17af37fea8e2b4a29a9ee2b5eaaf4483c3"
)
SAFETY_FLOOR_GB = 10.0
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
EXACT_ALLOWLIST = {
    96223: {
        "retry_generation": "timeout12h-r3",
        "candidate_physics_sha256": (
            "2a1bb6f2be79d5a9443538702b833e2b1918877a1137923c1a1ea0f606b46660"
        ),
        "plan_payload_sha256": (
            "08748c42a0306c6cace6aebd91ff83e186ac0c190ef9d2a212447b26aae9d2d6"
        ),
    },
    96224: {
        "retry_generation": "timeout12h-r2",
        "candidate_physics_sha256": (
            "436565e3f360d79583ac86eb095c5e25b68bb56a90e2e7dba8385c6f1bb00333"
        ),
        "plan_payload_sha256": (
            "c82088d5de3f669267fbcf9bf09fb3396e15827884513bb75442406b3b5d35a9"
        ),
    },
    96230: {
        "retry_generation": "timeout12h-r4",
        "candidate_physics_sha256": (
            "b7c30cb70b95611821d233306688586985f75f026c246e4fcf87b4bb2ea63412"
        ),
        "plan_payload_sha256": (
            "d5eccb7c52ccb6642000a8adc9bd0f791abdb1692bb514686534c8b8639c32f6"
        ),
    },
}


class SafeRefillError(RuntimeError):
    """Fail-closed refill authority or live gate failure."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise SafeRefillError("payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SafeRefillError(f"JSON artifact is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise SafeRefillError(f"JSON artifact is not an object: {path}")
    return value


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != canonical_sha256(
        unsigned
    ):
        raise SafeRefillError(f"{schema} payload seal mismatch")
    return dict(value)


def _write_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(production._json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def _write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    try:
        return production._write_immutable_json(path.resolve(), value)
    except Exception as exc:
        raise SafeRefillError(f"immutable output exists or failed: {path}") from exc


def _candidate_from_path(path: Path) -> dict[str, Any]:
    plan, _params, selected, _parent = timeout12h._load_plan(path)
    record = plan["retry_of_timeout12h"]
    logical_id = record["logical_authority_task_id"]
    allowed = EXACT_ALLOWLIST.get(logical_id)
    profile = production._read_json(
        path.resolve(strict=True).parent / plan["profile"]["path"]
    )
    strict = plan["scheduler_strict_node_contract"]
    stream = record["stream_evidence"]
    storage = record["storage_audit"]
    fixed = selected["row_contract"]["fixed_identity_attestation"]
    if (
        allowed is None
        or record["retry_generation"] != allowed["retry_generation"]
        or plan["candidate_physics_sha256"]
        != allowed["candidate_physics_sha256"]
        or plan["payload_sha256"] != allowed["plan_payload_sha256"]
        or strict["scheduler_revision"] != TARGET_SCHEDULER_REVISION
        or strict["scheduler_cutover_receipt_sha256"]
        != TARGET_CUTOVER_SHA256
        or strict["requested_node_name"] != TARGET_NODE
        or strict["node_name_policy"] != "strict"
        or profile.get("mem_mb") != 32768
        or profile.get("cpus") != 8
        or profile.get("timeout_seconds") != 43200
        or storage.get("account_name") != TARGET_ACCOUNT
        or storage.get("fresh_pre_submit_storage_reauthentication_required")
        is not True
        or storage.get("parallel_submit_without_reaudit_allowed") is not False
        or stream.get("fresh_grid_output_bytes", 0) <= 0
        or not math.isclose(
            storage["prospective_grid_gb"],
            stream["fresh_grid_output_bytes"] / (1024**3),
            rel_tol=0,
            abs_tol=1e-12,
        )
        or fixed.get("attested") is not True
        or fixed.get("sha256") != FIXED_IDENTITY_SHA256
        or fixed["expected"].get("fan_velocity") != 1.5
        or fixed["expected"].get("thermal_pad_conductivity_W_mK") != 0.2
        or fixed["expected"].get("core_plate_pad_t") != 2.0
        or fixed["expected"].get("wcp_pad_t") != 2.0
    ):
        raise SafeRefillError(
            f"source is not an exact approved 4fac timeout12h slot: {path}"
        )
    reference = timeout12h._validate_claim_reference(plan)
    claim_root = Path(reference["claim_root_authority"]["resolved_root"])
    claim_directory = claim_root / reference["relative_claim_directory"]
    return {
        "logical_authority_task_id": logical_id,
        "retry_generation": record["retry_generation"],
        "candidate_physics_sha256": plan["candidate_physics_sha256"],
        "candidate_stem": plan["candidate_physics_sha256"][:12],
        "plan": production._file_record(path),
        "plan_payload_sha256": plan["payload_sha256"],
        "task_name": plan["stage"]["task_name"],
        "dedupe_key": plan["stage"]["retained_aedt_bundle"]["dedupe_key"],
        "account_name": storage["account_name"],
        "prospective_grid_gb": storage["prospective_grid_gb"],
        "fresh_grid_output_bytes": stream["fresh_grid_output_bytes"],
        "claim_key": reference["claim_key"],
        "claim_directory": str(claim_directory.resolve()),
        "fixed_identity_attestation_sha256": fixed["sha256"],
    }


def initialize_plan(
    *,
    output_root: Path,
    source_plans: Sequence[Path],
    cutover_receipt: Path,
    watcher_plan: Path,
    ssh_host: str,
    ssh_port: int,
    ssh_private_key: Path,
    ssh_known_hosts: Path,
    ssh_storage_path: str,
    code_revision: str,
) -> Path:
    if len(source_plans) != len(EXACT_ALLOWLIST):
        raise SafeRefillError("all and only three exact refill slots are required")
    candidates = [_candidate_from_path(path) for path in source_plans]
    if {item["logical_authority_task_id"] for item in candidates} != set(
        EXACT_ALLOWLIST
    ):
        raise SafeRefillError("exact refill logical allowlist is incomplete")
    cutover_record = production._file_record(cutover_receipt)
    if cutover_record["sha256"] != TARGET_CUTOVER_SHA256:
        raise SafeRefillError("active 4fac cutover receipt bytes drifted")
    base_watch = terminal_watcher._load_watch_plan(watcher_plan)
    if base_watch["scheduler_url"] != SCHEDULER_URL:
        raise SafeRefillError("watcher Scheduler authority drifted")
    revision = code_revision.strip().lower()
    if (
        ssh_host.strip() == ""
        or ssh_port <= 0
        or ssh_storage_path.strip() == ""
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise SafeRefillError("SSH/code authority is incomplete")
    root = output_root.resolve()
    value = _sealed(
        {
            "schema_version": PLAN_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "scheduler_url": SCHEDULER_URL,
            "scheduler_project": PROJECT,
            "scheduler_get_endpoints": [
                "/api/health",
                "/api/licenses",
                "/api/task-capacity",
                "/api/tasks",
            ],
            "scheduler_post_endpoint": "/api/tasks",
            "scheduler_cancel_allowed": False,
            "maximum_post_calls_per_cycle": 1,
            "live_submission_default": False,
            "output_root": str(root),
            "code_revision": revision,
            "cutover_receipt": cutover_record,
            "watcher_plan": production._file_record(watcher_plan),
            "watcher_extension_directory": str(
                (root / "watcher_extensions").resolve()
            ),
            "ssh_storage_authority": {
                "host": ssh_host.strip(),
                "port": ssh_port,
                "username": TARGET_ACCOUNT,
                "storage_path": ssh_storage_path.strip(),
                "private_key": production._file_record(ssh_private_key),
                "known_hosts": production._file_record(ssh_known_hosts),
            },
            "fixed_boundary_policy": {
                "fixed_identity_attestation_sha256": FIXED_IDENTITY_SHA256,
                "fan_velocity_m_s": 1.5,
                "thermal_pad_conductivity_W_mK": 0.2,
                "pad_thickness_mm": 2.0,
                "mutable": False,
            },
            "selection_order": (
                "smallest_sealed_prospective_grid_gb_then_logical_task_id"
            ),
            "candidates": sorted(
                candidates,
                key=lambda item: (
                    item["prospective_grid_gb"],
                    item["logical_authority_task_id"],
                ),
            ),
        }
    )
    path = root / "safe_refill_plan.json"
    if path.exists():
        if _read_json(path) != value:
            raise SafeRefillError("existing immutable refill plan differs")
        return path
    return _write_immutable(path, value)


def load_plan(path: Path) -> dict[str, Any]:
    value = _validate_seal(_read_json(path), PLAN_SCHEMA)
    root = Path(value["output_root"]).resolve()
    if path.resolve() != root / "safe_refill_plan.json":
        raise SafeRefillError("refill output root drifted")
    if (
        value.get("scheduler_url") != SCHEDULER_URL
        or value.get("scheduler_project") != PROJECT
        or value.get("scheduler_cancel_allowed") is not False
        or value.get("maximum_post_calls_per_cycle") != 1
        or value.get("live_submission_default") is not False
        or value.get("fixed_boundary_policy", {}).get("mutable") is not False
    ):
        raise SafeRefillError("refill plan policy drifted")
    refreshed = [
        _candidate_from_path(Path(item["plan"]["path"]))
        for item in value["candidates"]
    ]
    if refreshed != value["candidates"]:
        raise SafeRefillError("refill source plan bytes drifted")
    for name in ("private_key", "known_hosts"):
        record = value["ssh_storage_authority"][name]
        if production._file_record(Path(record["path"])) != record:
            raise SafeRefillError(f"SSH {name} bytes drifted")
    if production._file_record(
        Path(value["cutover_receipt"]["path"])
    ) != value["cutover_receipt"]:
        raise SafeRefillError("4fac cutover receipt drifted")
    if production._file_record(
        Path(value["watcher_plan"]["path"])
    ) != value["watcher_plan"]:
        raise SafeRefillError("base watcher plan drifted")
    return value


def create_extension_authority(
    *, refill_plan_path: Path, output: Path
) -> Path:
    plan = load_plan(refill_plan_path)
    value = _sealed(
        {
            "schema_version": EXTENSION_AUTHORITY_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "base_watcher_plan": copy.deepcopy(plan["watcher_plan"]),
            "safe_refill_plan": production._file_record(refill_plan_path),
            "extension_directory": plan["watcher_extension_directory"],
            "allowed_logical_authority_task_ids": sorted(EXACT_ALLOWLIST),
            "allowed_source_plan_payload_sha256": sorted(
                item["plan_payload_sha256"] for item in plan["candidates"]
            ),
            "extension_receipt_schema": EXTENSION_RECEIPT_SCHEMA,
            "watcher_scheduler_methods": ["GET"],
            "watcher_scheduler_mutation_performed": False,
        }
    )
    return _write_immutable(output, value)


def authenticate_extension_authority(
    path: Path, *, watch_plan_path: Path | None = None
) -> dict[str, Any]:
    value = _validate_seal(_read_json(path), EXTENSION_AUTHORITY_SCHEMA)
    refill_record = value.get("safe_refill_plan")
    watch_record = value.get("base_watcher_plan")
    if not isinstance(refill_record, Mapping) or not isinstance(
        watch_record, Mapping
    ):
        raise SafeRefillError("extension authority records are absent")
    refill_path = Path(refill_record["path"])
    plan = load_plan(refill_path)
    expected = {
        "base_watcher_plan": plan["watcher_plan"],
        "safe_refill_plan": production._file_record(refill_path),
        "extension_directory": plan["watcher_extension_directory"],
        "allowed_logical_authority_task_ids": sorted(EXACT_ALLOWLIST),
        "allowed_source_plan_payload_sha256": sorted(
            item["plan_payload_sha256"] for item in plan["candidates"]
        ),
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise SafeRefillError("extension authority drifted")
    if watch_plan_path is not None and production._file_record(
        watch_plan_path
    ) != watch_record:
        raise SafeRefillError("extension authority targets another watcher")
    return value


def _get_json(url: str) -> Any:
    request = production.urllib.request.Request(
        url, headers={"Accept": "application/json"}, method="GET"
    )
    try:
        with production.urllib.request.urlopen(
            request, timeout=30
        ) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except Exception as exc:
        raise SafeRefillError(f"Scheduler GET failed: {url}") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SafeRefillError("Scheduler GET response exceeds byte bound")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SafeRefillError("Scheduler GET response is invalid JSON") from exc


def _task_inventory(*, scheduler_url: str) -> list[dict[str, Any]]:
    query = production.urllib.parse.urlencode(
        [
            ("project", PROJECT),
            ("limit", 10000),
            *[("status", status) for status in sorted(ACTIVE_STATUSES)],
        ]
    )
    value = _get_json(f"{scheduler_url.rstrip('/')}/api/tasks?{query}")
    if not isinstance(value, list) or len(value) >= 10000:
        raise SafeRefillError("active task inventory is malformed/truncated")
    rows = [dict(row) for row in value if isinstance(row, Mapping)]
    if len(rows) != len(value):
        raise SafeRefillError("active task inventory contains malformed rows")
    return rows


def _capacity(*, scheduler_url: str, account_name: str) -> dict[str, Any]:
    query = production.urllib.parse.urlencode(
        {
            "cpus": 8,
            "memory_mb": 32768,
            "scheduling_profile": "fea_bursty",
            "aedt_backend": "standalone",
            "required_capability": "conda:pyaedt2026v1",
            "env_profile": "pyaedt2026v1",
            "project": PROJECT,
            "account_name": account_name,
            "node_name": TARGET_NODE,
            "node_name_policy": "strict",
        }
    )
    value = _get_json(
        f"{scheduler_url.rstrip('/')}/api/task-capacity?{query}"
    )
    ready = value.get("ready_fit_slots") if isinstance(value, Mapping) else None
    if (
        isinstance(ready, bool)
        or not isinstance(ready, int)
        or ready < 0
        or value.get("memory_pressure_state") != "ok"
    ):
        raise SafeRefillError("Scheduler task-capacity response is unsafe")
    return copy.deepcopy(dict(value))


def _parse_mmlsquota(output: str) -> list[dict[str, Any]]:
    headers: dict[tuple[str, str], dict[str, int]] = {}
    rows: list[list[str]] = []
    for line in output.splitlines():
        fields = line.rstrip().split(":")
        if len(fields) < 4 or fields[0] != "mmlsquota":
            continue
        key = (fields[0], fields[1])
        if fields[2] == "HEADER":
            headers[key] = {
                name: index for index, name in enumerate(fields) if name
            }
        else:
            rows.append(fields)
    parsed = []
    for fields in rows:
        indices = headers.get((fields[0], fields[1]))
        required = {
            "filesystemName",
            "quotaType",
            "blockUsage",
            "blockQuota",
            "blockLimit",
            "blockInDoubt",
        }
        if not indices or not required.issubset(indices):
            continue
        quota_type = fields[indices["quotaType"]].strip().upper()
        if quota_type not in {"USR", "FILESET"}:
            continue
        try:
            used = float(fields[indices["blockUsage"]]) / 1024 / 1024
            soft = float(fields[indices["blockQuota"]]) / 1024 / 1024
            hard = float(fields[indices["blockLimit"]]) / 1024 / 1024
            in_doubt = (
                float(fields[indices["blockInDoubt"]]) / 1024 / 1024
            )
        except (ValueError, IndexError):
            continue
        limit = hard if hard > 0 else soft
        parsed.append(
            {
                "filesystem_name": fields[
                    indices["filesystemName"]
                ].strip(),
                "quota_type": quota_type,
                "used_gb": used,
                "in_doubt_gb": max(0.0, in_doubt),
                "limit_gb": limit,
                "raw_free_gb": limit - used,
                "effective_free_gb": limit - used - max(0.0, in_doubt),
                "fileset_name": (
                    fields[indices["filesetname"]].strip()
                    if "filesetname" in indices
                    and indices["filesetname"] < len(fields)
                    else ""
                ),
            }
        )
    return parsed


def _quota_command(storage_path: str) -> str:
    quoted = shlex.quote(storage_path)
    return (
        f"storage_path={quoted}; export LC_ALL=C; "
        'test -e "$storage_path" || exit 20; '
        'fs_type=$(stat -f -c %T -- "$storage_path") || exit 21; '
        'printf "__FS__:%s\\n" "$fs_type"; '
        'case "$fs_type" in gpfs*) ;; *) exit 22;; esac; '
        "quota_cmd=$(command -v mmlsquota 2>/dev/null || "
        "printf /usr/lpp/mmfs/bin/mmlsquota); "
        "attr_cmd=$(command -v mmlsattr 2>/dev/null || "
        "printf /usr/lpp/mmfs/bin/mmlsattr); "
        'test -x "$quota_cmd" && test -x "$attr_cmd" || exit 23; '
        'device=$(df -P -- "$storage_path" | awk \'END {print $1}\'); '
        'device=${device#/dev/}; test -n "$device" || exit 24; '
        'attr_output=$("$attr_cmd" -L "$storage_path") || exit 25; '
        "fileset=$(printf \"%s\\n\" \"$attr_output\" | "
        "sed -n 's/^[[:space:]]*[Ff]ileset name[[:space:]]*:"
        "[[:space:]]*//p' | sed 's/[[:space:]]*$//' | head -n 1); "
        'test -n "$fileset" || exit 26; '
        'printf "__FILESET__:%s\\n" "$fileset"; '
        'quota_user=${USER:-$(id -un)}; '
        'user_output=$("$quota_cmd" -u "$quota_user" -v -Y '
        '"${device}:${fileset}" 2>/dev/null); user_status=$?; '
        'if [ "$user_status" -ne 0 ]; then '
        'user_output=$("$quota_cmd" -u "$quota_user" -v -Y "$device"); '
        "user_status=$?; fi; "
        'printf "%s\\n" "$user_output"; '
        '"$quota_cmd" -j "$fileset" -v -Y "$device"; '
        'fileset_status=$?; test "$user_status" -eq 0 '
        '&& test "$fileset_status" -eq 0'
    )


def _fresh_storage(plan: Mapping[str, Any]) -> dict[str, Any]:
    authority = plan["ssh_storage_authority"]
    client = paramiko.SSHClient()
    client.load_host_keys(authority["known_hosts"]["path"])
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(
            hostname=authority["host"],
            port=authority["port"],
            username=authority["username"],
            key_filename=authority["private_key"]["path"],
            allow_agent=False,
            look_for_keys=False,
            timeout=20,
            auth_timeout=20,
            banner_timeout=20,
        )
        _stdin, stdout, stderr = client.exec_command(
            _quota_command(authority["storage_path"]), timeout=60
        )
        stdout_text = stdout.read().decode("utf-8", "strict")
        stderr_text = stderr.read().decode("utf-8", "replace")
        exit_code = stdout.channel.recv_exit_status()
    except Exception as exc:
        raise SafeRefillError("fresh read-only GPFS quota probe failed") from exc
    finally:
        client.close()
    if exit_code != 0:
        raise SafeRefillError(
            f"fresh read-only GPFS quota probe exit={exit_code}: "
            f"{stderr_text[:200]}"
        )
    quotas = _parse_mmlsquota(stdout_text)
    types = {item["quota_type"] for item in quotas}
    limited = [item for item in quotas if item["limit_gb"] > 0]
    if not {"USR", "FILESET"}.issubset(types) or not limited:
        raise SafeRefillError("fresh GPFS quota response lacks USR/FILESET limits")
    limiting = min(limited, key=lambda item: item["effective_free_gb"])
    return {
        "schema_version": "mft-goal-safe-refill-gpfs-probe-v1",
        "observed_at_utc": _now(),
        "account_name": authority["username"],
        "storage_path": authority["storage_path"],
        "filesystem_type": next(
            (
                line.split(":", 1)[1]
                for line in stdout_text.splitlines()
                if line.startswith("__FS__:")
            ),
            "",
        ),
        "limiting_quota": limiting,
        "quota_types_observed": sorted(types),
        "read_only": True,
    }


def _is_fea(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("aedt_backend") or "") in {"standalone", "pooled"}
        or str(row.get("required_capability") or "").startswith("conda:pyaedt")
    )


def _is_sibling(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    name = str(row.get("name") or "")
    dedupe = str(row.get("dedupe_key") or "")
    logical = candidate["logical_authority_task_id"]
    stem = candidate["candidate_stem"]
    return (
        name == candidate["task_name"]
        or dedupe == candidate["dedupe_key"]
        or stem in name
        or f"-l{logical}-" in name
        or f"-t{logical}-" in name
    )


def _claim_state(candidate: Mapping[str, Any]) -> str:
    path = Path(candidate["claim_directory"])
    return "unsubmitted" if not path.exists() else "claimed"


def _evaluate(
    plan: Mapping[str, Any],
    *,
    task_reader: Callable[..., list[dict[str, Any]]] = _task_inventory,
    capacity_reader: Callable[..., dict[str, Any]] = _capacity,
    storage_reader: Callable[[Mapping[str, Any]], dict[str, Any]] = _fresh_storage,
    validate_cutover: bool = True,
    ignore_claim_for_logical_id: int | None = None,
) -> dict[str, Any]:
    if validate_cutover:
        strict = timeout12h._load_plan(
            Path(plan["candidates"][0]["plan"]["path"])
        )[0]["scheduler_strict_node_contract"]
        diagnostic._validate_scheduler_cutover_receipt(
            Path(plan["cutover_receipt"]["path"]),
            verify_live_launcher=True,
            require_strict_node=True,
            strict_node_contract=strict,
            require_active_strict=True,
        )
    tasks = task_reader(scheduler_url=plan["scheduler_url"])
    capacity = capacity_reader(
        scheduler_url=plan["scheduler_url"], account_name=TARGET_ACCOUNT
    )
    storage = storage_reader(plan)
    account_active = [
        row
        for row in tasks
        if row.get("account_name") == TARGET_ACCOUNT and _is_fea(row)
    ]
    rows = []
    for candidate in plan["candidates"]:
        prospective = candidate["prospective_grid_gb"]
        limiting = storage["limiting_quota"]
        remaining = (
            limiting["raw_free_gb"]
            - limiting["in_doubt_gb"]
            - prospective
            - SAFETY_FLOOR_GB
        )
        siblings = [
            row for row in tasks if _is_sibling(row, candidate)
        ]
        claim = _claim_state(candidate)
        gates = {
            "account_active_fea_zero": len(account_active) == 0,
            "scheduler_ready_fit_slots_positive": (
                capacity["ready_fit_slots"] > 0
            ),
            "fresh_gpfs_envelope_passed": remaining >= 0,
            "candidate_logical_nonterminal_siblings_zero": (
                len(siblings) == 0
            ),
            "atomic_claim_unsubmitted": (
                claim == "unsubmitted"
                or candidate["logical_authority_task_id"]
                == ignore_claim_for_logical_id
            ),
            "fixed_physics_fan_tim_pad_reauthenticated": True,
            "active_4fac_cutover_reauthenticated": True,
        }
        rows.append(
            {
                **copy.deepcopy(candidate),
                "eligible": all(gates.values()),
                "gates": gates,
                "claim_state": claim,
                "account_active_fea_count": len(account_active),
                "account_active_fea_task_ids": [
                    row.get("task_id", row.get("id"))
                    for row in account_active
                ],
                "nonterminal_sibling_count": len(siblings),
                "nonterminal_sibling_task_ids": [
                    row.get("task_id", row.get("id")) for row in siblings
                ],
                "fresh_gpfs_remaining_after_in_doubt_prospective_floor_gb": (
                    remaining
                ),
            }
        )
    selected = next((row for row in rows if row["eligible"]), None)
    return {
        "schema_version": "mft-goal-safe-refill-gate-evaluation-v1",
        "observed_at_utc": _now(),
        "task_inventory_sha256": canonical_sha256(tasks),
        "active_task_count": len(tasks),
        "capacity": capacity,
        "capacity_sha256": canonical_sha256(capacity),
        "storage": storage,
        "storage_sha256": canonical_sha256(storage),
        "candidates": rows,
        "selected_logical_authority_task_id": (
            selected["logical_authority_task_id"] if selected else None
        ),
        "all_scheduler_reads_get_only": True,
    }


def _slot_paths(
    plan: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Path]:
    root = Path(plan["output_root"])
    logical = candidate["logical_authority_task_id"]
    directory = root / "slots" / f"l{logical}-{candidate['candidate_stem']}"
    return {
        "directory": directory,
        "intent": directory / "submit_intent.json",
        "submission": directory / "submission.json",
        "action": directory / "action_receipt.json",
        "extension": (
            Path(plan["watcher_extension_directory"]) / f"l{logical}.json"
        ),
    }


def _load_intent(
    path: Path,
    *,
    refill_plan_path: Path,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_seal(_read_json(path), INTENT_SCHEMA)
    if (
        value.get("safe_refill_plan")
        != production._file_record(refill_plan_path)
        or value.get("logical_authority_task_id")
        != candidate["logical_authority_task_id"]
        or value.get("candidate_physics_sha256")
        != candidate["candidate_physics_sha256"]
        or value.get("maximum_post_calls") != 1
        or value.get("scheduler_cancel_allowed") is not False
    ):
        raise SafeRefillError("durable refill submit intent drifted")
    return value


def _extension_receipt(
    *,
    refill_plan_path: Path,
    authority_path: Path,
    candidate: Mapping[str, Any],
    submission_path: Path,
) -> dict[str, Any]:
    load_plan(refill_plan_path)
    source_plan, _params, _selected, _parent = timeout12h._load_plan(
        Path(candidate["plan"]["path"])
    )
    submission = timeout12h.load_submission_for_probe(
        submission_path, plan=source_plan
    )
    slot = terminal_watcher._load_slot_source(
        candidate["logical_authority_task_id"],
        submission_path,
        scheduler_url=SCHEDULER_URL,
        project=PROJECT,
    )
    if (
        submission.get("scheduler_submission_performed") is not True
        or slot["candidate_physics_sha256"]
        != candidate["candidate_physics_sha256"]
    ):
        raise SafeRefillError("submitted extension identity drifted")
    return _sealed(
        {
            "schema_version": EXTENSION_RECEIPT_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "extension_authority": production._file_record(authority_path),
            "safe_refill_plan": production._file_record(refill_plan_path),
            "source_plan": copy.deepcopy(candidate["plan"]),
            "source_plan_payload_sha256": candidate["plan_payload_sha256"],
            "submission": production._file_record(submission_path),
            "submission_payload_sha256": submission["payload_sha256"],
            "logical_authority_task_id": slot[
                "logical_authority_task_id"
            ],
            "execution_task_id": slot["execution_task_id"],
            "task_name": slot["task_name"],
            "dedupe_key": slot["dedupe_key"],
            "candidate_physics_sha256": slot[
                "candidate_physics_sha256"
            ],
            "fixed_identity_attestation_sha256": slot[
                "fixed_identity_attestation_sha256"
            ],
            "watcher_scheduler_methods": ["GET"],
            "watcher_scheduler_mutation_performed": False,
            "producer_scheduler_submission_performed": True,
            "authorized_at_utc": _now(),
        }
    )


def authenticate_watcher_extension_receipt(
    path: Path, *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    raw = _read_json(path)
    if raw.get("schema_version") == (
        "mft-goal-watcher-authorized-replacement-v1"
    ):
        from tools import mft_goal_startup_retry

        return (
            mft_goal_startup_retry
            .authenticate_watcher_replacement_receipt(
                path, authority=authority
            )
        )
    value = _validate_seal(raw, EXTENSION_RECEIPT_SCHEMA)
    authority_path = Path(value["extension_authority"]["path"])
    if (
        production._file_record(authority_path)
        != value["extension_authority"]
        or _validate_seal(
            _read_json(authority_path), EXTENSION_AUTHORITY_SCHEMA
        )
        != authority
    ):
        raise SafeRefillError("extension receipt authority drifted")
    refill_path = Path(value["safe_refill_plan"]["path"])
    plan = load_plan(refill_path)
    logical = value["logical_authority_task_id"]
    candidate = next(
        (
            item
            for item in plan["candidates"]
            if item["logical_authority_task_id"] == logical
        ),
        None,
    )
    if candidate is None:
        raise SafeRefillError("extension logical slot is not allowlisted")
    submission_path = Path(value["submission"]["path"])
    expected = _extension_receipt(
        refill_plan_path=refill_path,
        authority_path=authority_path,
        candidate=candidate,
        submission_path=submission_path,
    )
    for key in (
        "extension_authority",
        "safe_refill_plan",
        "source_plan",
        "source_plan_payload_sha256",
        "submission",
        "submission_payload_sha256",
        "logical_authority_task_id",
        "execution_task_id",
        "task_name",
        "dedupe_key",
        "candidate_physics_sha256",
        "fixed_identity_attestation_sha256",
        "watcher_scheduler_methods",
        "watcher_scheduler_mutation_performed",
        "producer_scheduler_submission_performed",
    ):
        if value.get(key) != expected.get(key):
            raise SafeRefillError("extension receipt reauthentication drifted")
    return terminal_watcher._load_slot_source(
        logical,
        submission_path,
        scheduler_url=SCHEDULER_URL,
        project=PROJECT,
    )


def _ensure_extension(
    *,
    refill_plan_path: Path,
    authority_path: Path,
    candidate: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> Path:
    if not paths["submission"].exists():
        raise SafeRefillError("cannot extend watcher without submission receipt")
    receipt = _extension_receipt(
        refill_plan_path=refill_plan_path,
        authority_path=authority_path,
        candidate=candidate,
        submission_path=paths["submission"],
    )
    if paths["extension"].exists():
        authenticate_watcher_extension_receipt(
            paths["extension"],
            authority=authenticate_extension_authority(authority_path),
        )
        return paths["extension"]
    return _write_immutable(paths["extension"], receipt)


def _cycle_locked(
    *,
    refill_plan_path: Path,
    extension_authority_path: Path,
    authorize_submit: bool = False,
    task_reader: Callable[..., list[dict[str, Any]]] = _task_inventory,
    capacity_reader: Callable[..., dict[str, Any]] = _capacity,
    storage_reader: Callable[[Mapping[str, Any]], dict[str, Any]] = _fresh_storage,
    submitter: Callable[..., Path] = timeout12h.submit,
    validate_cutover: bool = True,
    gate_evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plan = load_plan(refill_plan_path)
    authenticate_extension_authority(
        extension_authority_path,
        watch_plan_path=Path(plan["watcher_plan"]["path"]),
    )
    recovered_extensions = []
    pending_recovery = []
    for candidate in plan["candidates"]:
        paths = _slot_paths(plan, candidate)
        if paths["submission"].exists():
            recovered_extensions.append(
                str(
                    _ensure_extension(
                        refill_plan_path=refill_plan_path,
                        authority_path=extension_authority_path,
                        candidate=candidate,
                        paths=paths,
                    )
                )
            )
        elif paths["intent"].exists():
            _load_intent(
                paths["intent"],
                refill_plan_path=refill_plan_path,
                candidate=candidate,
            )
            if _claim_state(candidate) == "claimed":
                pending_recovery.append((candidate, paths))
    if len(pending_recovery) > 1:
        raise SafeRefillError("more than one refill recovery is pending")
    evaluator = gate_evaluator or _evaluate
    evaluation = evaluator(
        plan,
        task_reader=task_reader,
        capacity_reader=capacity_reader,
        storage_reader=storage_reader,
        validate_cutover=validate_cutover,
    )
    selected_id = evaluation["selected_logical_authority_task_id"]
    action = "blocked"
    post_calls = 0
    submission_path: Path | None = None
    if pending_recovery:
        action = "recovery_pending_authorization"
    elif selected_id is not None:
        action = "dry_run_would_submit"
    if authorize_submit and pending_recovery:
        candidate, paths = pending_recovery[0]

        def forbid_recovery_post() -> None:
            raise SafeRefillError(
                "claim recovery attempted a new Scheduler POST"
            )

        submission_path = submitter(
            plan_path=Path(candidate["plan"]["path"]),
            scheduler_cutover_receipt_path=Path(
                plan["cutover_receipt"]["path"]
            ),
            output=paths["submission"],
            additional_pre_submit_guard=forbid_recovery_post,
        )
        receipt_plan = timeout12h._load_plan(
            Path(candidate["plan"]["path"])
        )[0]
        submission = timeout12h.load_submission_for_probe(
            submission_path, plan=receipt_plan
        )
        if (
            submission["scheduler_strict_node_contract"][
                "scheduler_mutation_performed"
            ]
            is not False
        ):
            raise SafeRefillError("claim recovery unexpectedly mutated Scheduler")
        extension = _ensure_extension(
            refill_plan_path=refill_plan_path,
            authority_path=extension_authority_path,
            candidate=candidate,
            paths=paths,
        )
        if not paths["action"].exists():
            _write_immutable(
                paths["action"],
                _sealed(
                    {
                        "schema_version": ACTION_SCHEMA,
                        "safe_refill_plan": production._file_record(
                            refill_plan_path
                        ),
                        "submit_intent": production._file_record(
                            paths["intent"]
                        ),
                        "submission": production._file_record(
                            submission_path
                        ),
                        "watcher_extension": production._file_record(
                            extension
                        ),
                        "logical_authority_task_id": candidate[
                            "logical_authority_task_id"
                        ],
                        "scheduler_post_calls": 0,
                        "scheduler_cancel_calls": 0,
                        "completed_at_utc": _now(),
                    }
                ),
            )
        action = "claim_recovered_and_watcher_extended"
    elif authorize_submit and selected_id is not None:
        fresh = evaluator(
            plan,
            task_reader=task_reader,
            capacity_reader=capacity_reader,
            storage_reader=storage_reader,
            validate_cutover=validate_cutover,
        )
        if fresh["selected_logical_authority_task_id"] != selected_id:
            raise SafeRefillError("selected refill slot changed on fresh revalidation")
        candidate = next(
            item
            for item in plan["candidates"]
            if item["logical_authority_task_id"] == selected_id
        )
        paths = _slot_paths(plan, candidate)
        intent = _sealed(
            {
                "schema_version": INTENT_SCHEMA,
                "safe_refill_plan": production._file_record(refill_plan_path),
                "logical_authority_task_id": selected_id,
                "candidate_physics_sha256": candidate[
                    "candidate_physics_sha256"
                ],
                "initial_gate_evaluation_sha256": canonical_sha256(evaluation),
                "fresh_gate_evaluation_sha256": canonical_sha256(fresh),
                "maximum_post_calls": 1,
                "scheduler_cancel_allowed": False,
                "authorized_at_utc": _now(),
            }
        )
        if not paths["intent"].exists():
            _write_immutable(paths["intent"], intent)

        def final_guard() -> None:
            guarded = evaluator(
                plan,
                task_reader=task_reader,
                capacity_reader=capacity_reader,
                storage_reader=storage_reader,
                validate_cutover=validate_cutover,
                ignore_claim_for_logical_id=selected_id,
            )
            selected = next(
                row
                for row in guarded["candidates"]
                if row["logical_authority_task_id"] == selected_id
            )
            if not selected["eligible"]:
                raise SafeRefillError("immediate pre-POST refill gates failed")

        submission_path = submitter(
            plan_path=Path(candidate["plan"]["path"]),
            scheduler_cutover_receipt_path=Path(
                plan["cutover_receipt"]["path"]
            ),
            output=paths["submission"],
            additional_pre_submit_guard=final_guard,
        )
        receipt_plan = timeout12h._load_plan(
            Path(candidate["plan"]["path"])
        )[0]
        submission = timeout12h.load_submission_for_probe(
            submission_path, plan=receipt_plan
        )
        post_calls = int(
            submission["scheduler_strict_node_contract"][
                "scheduler_mutation_performed"
            ]
            is True
        )
        if post_calls > 1:
            raise SafeRefillError("more than one POST occurred in a refill cycle")
        extension = _ensure_extension(
            refill_plan_path=refill_plan_path,
            authority_path=extension_authority_path,
            candidate=candidate,
            paths=paths,
        )
        if not paths["action"].exists():
            _write_immutable(
                paths["action"],
                _sealed(
                    {
                        "schema_version": ACTION_SCHEMA,
                        "safe_refill_plan": production._file_record(
                            refill_plan_path
                        ),
                        "submit_intent": production._file_record(
                            paths["intent"]
                        ),
                        "submission": production._file_record(
                            submission_path
                        ),
                        "watcher_extension": production._file_record(
                            extension
                        ),
                        "logical_authority_task_id": selected_id,
                        "scheduler_post_calls": post_calls,
                        "scheduler_cancel_calls": 0,
                        "completed_at_utc": _now(),
                    }
                ),
            )
        action = "submitted_and_watcher_extended"
    state = _sealed(
        {
            "schema_version": STATE_SCHEMA,
            "campaign_id": "mft-goal-20260726",
            "safe_refill_plan_payload_sha256": plan["payload_sha256"],
            "observed_at_utc": _now(),
            "mode": "live_authorized" if authorize_submit else "dry_run",
            "action": action,
            "scheduler_post_calls_this_cycle": post_calls,
            "scheduler_cancel_calls_this_cycle": 0,
            "evaluation": evaluation,
            "recovered_watcher_extensions": recovered_extensions,
            "submission": (
                production._file_record(submission_path)
                if submission_path is not None
                else None
            ),
        }
    )
    _write_atomic(Path(plan["output_root"]) / "state.json", state)
    return state


def cycle(
    *,
    refill_plan_path: Path,
    extension_authority_path: Path,
    authorize_submit: bool = False,
    task_reader: Callable[..., list[dict[str, Any]]] = _task_inventory,
    capacity_reader: Callable[..., dict[str, Any]] = _capacity,
    storage_reader: Callable[[Mapping[str, Any]], dict[str, Any]] = _fresh_storage,
    submitter: Callable[..., Path] = timeout12h.submit,
    validate_cutover: bool = True,
    gate_evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plan = load_plan(refill_plan_path)
    lock_path = Path(plan["output_root"]) / "safe_refill.lock"
    with terminal_watcher.SingleInstanceLock(lock_path):
        return _cycle_locked(
            refill_plan_path=refill_plan_path,
            extension_authority_path=extension_authority_path,
            authorize_submit=authorize_submit,
            task_reader=task_reader,
            capacity_reader=capacity_reader,
            storage_reader=storage_reader,
            submitter=submitter,
            validate_cutover=validate_cutover,
            gate_evaluator=gate_evaluator,
        )


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={REPOSITORY_ROOT}", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed exact 4fac timeout12h SAFE REFILL"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output-root", type=Path, required=True)
    init.add_argument("--source-plan", type=Path, action="append", required=True)
    init.add_argument("--cutover-receipt", type=Path, required=True)
    init.add_argument("--watcher-plan", type=Path, required=True)
    init.add_argument("--ssh-host", required=True)
    init.add_argument("--ssh-port", type=int, default=22)
    init.add_argument("--ssh-private-key", type=Path, required=True)
    init.add_argument("--ssh-known-hosts", type=Path, required=True)
    init.add_argument("--ssh-storage-path", default="slurm_scheduler")
    init.add_argument("--code-revision", default="")
    authority = commands.add_parser("create-extension-authority")
    authority.add_argument("--refill-plan", type=Path, required=True)
    authority.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("cycle")
    run.add_argument("--refill-plan", type=Path, required=True)
    run.add_argument("--extension-authority", type=Path, required=True)
    run.add_argument(
        "--authorize-submit",
        action="store_true",
        help="enable at most one guarded POST; absent means GET-only dry-run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        result = initialize_plan(
            output_root=args.output_root,
            source_plans=args.source_plan,
            cutover_receipt=args.cutover_receipt,
            watcher_plan=args.watcher_plan,
            ssh_host=args.ssh_host,
            ssh_port=args.ssh_port,
            ssh_private_key=args.ssh_private_key,
            ssh_known_hosts=args.ssh_known_hosts,
            ssh_storage_path=args.ssh_storage_path,
            code_revision=args.code_revision or _git_revision(),
        )
    elif args.command == "create-extension-authority":
        result = create_extension_authority(
            refill_plan_path=args.refill_plan, output=args.output
        )
    else:
        result = cycle(
            refill_plan_path=args.refill_plan,
            extension_authority_path=args.extension_authority,
            authorize_submit=args.authorize_submit,
        )
        print(
            json.dumps(result, sort_keys=True, separators=(",", ":"))
        )
        return 0
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
