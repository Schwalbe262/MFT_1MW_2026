"""Restart-safe target100 feeder for compact diagnostic continuation seeds.

The first-clean exact100 receipt is immutable and is never reused as POST
authority.  This controller consumes a separately prepared/staged exact100
continuation plan (4200..4299, then aligned 100-seed cohorts), observes the
original receipt tasks with GET, and fills only the active-count deficit.

Scheduler mutation is limited to ``POST /api/tasks``.  Every continuation
payload is pinned round-robin to the five explicitly healthy accounts.  No
cancel, preempt, Scheduler repository edit, FEA promotion, or final-design
claim is available from this module.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_diagnostic_compact_scout as scout  # noqa: E402
from tools import mft_goal_diagnostic_compact_retry as retry  # noqa: E402
from tools import mft_goal_diagnostic_compact_slurm as offload  # noqa: E402
from tools.tier1_final1000_multiseed_contract import (  # noqa: E402
    scheduler_task_observation,
)


AUTHORITY_SCHEMA = "mft-goal-diagnostic-compact-target100-feeder-v1"
PENDING_SCHEMA = "mft-goal-diagnostic-compact-target100-pending-v1"
FINALIZED_SCHEMA = "mft-goal-diagnostic-compact-target100-finalized-v1"
CYCLE_SCHEMA = "mft-goal-diagnostic-compact-target100-cycle-v1"
TARGET_ACTIVE_COUNT = 100
ACTIVE_STATUSES = frozenset({"queued", "attaching", "running"})
TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "timeout", "timed_out"}
)
ALLOWED_ACCOUNTS = (
    "dhj02",
    "jji0930",
    "r1jae262",
    "dw16",
    "wjddn5916",
)
DEDUPE_PREFIX = offload.DEDUPE_PREFIX + "feeder:"
MAXIMUM_POSTS_PER_CYCLE = TARGET_ACTIVE_COUNT
EXPECTED_RECOVERY_ATTEMPT_COUNT = 47
_CYCLE_NAME = re.compile(r"cycle-([0-9]{8})\.json")


class Scheduler(Protocol):
    post_count: int
    get_count: int

    def list_namespace_tasks(
        self, name_prefix: str
    ) -> list[dict[str, Any]]: ...

    def get_task(self, task_id: int) -> dict[str, Any]: ...

    def submit_task(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"feeder JSON is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"feeder JSON must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise RuntimeError("feeder value is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def _validate_seal(
    value: Mapping[str, Any], *, schema: str
) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    unsigned = dict(result)
    observed = unsigned.pop("payload_sha256", None)
    if (
        result.get("schema_version") != schema
        or observed != canonical_sha256(unsigned)
    ):
        raise RuntimeError(f"{schema} seal mismatch")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
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
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _read_json(target) != dict(value):
            raise RuntimeError(f"immutable feeder artifact changed: {target}")
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
                raise RuntimeError(
                    f"concurrent feeder artifact differs: {target}"
                )
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _validate_plan_and_tasks(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    plan, _deployment, tasks, authentication = offload.authenticate_plan(path)
    seeds = tuple(sorted(int(task["seed"]) for task in tasks))
    profiles = [
        scout._validate_search_profile(
            task.get("manufacturing_search_profile") or {}
        )
        for task in tasks
    ]
    profile_hashes = {
        profile["payload_sha256"] for profile in profiles
    }
    clearances = {
        float(
            profile["geometry_constraint_profile"][
                "primary_axial_clearance"
            ]["minimum_mm"]
        )
        for profile in profiles
    }
    if (
        len(seeds) != offload.EXACT_TASK_COUNT
        or seeds
        != tuple(range(seeds[0], seeds[0] + offload.EXACT_TASK_COUNT))
        or seeds[0] < scout.CONTINUATION_SEED_START
        or (
            seeds[0] - scout.CONTINUATION_SEED_START
        )
        % scout.CONTINUATION_SEED_STRIDE
        != 0
        or plan.get("authorized_seeds") != list(seeds)
        or len(profile_hashes) != 1
        or clearances != {20.0}
        or any(
            profile["authorized_seed_start"] != seeds[0]
            or profile["authorized_seed_end_inclusive"] != seeds[-1]
            or profile["fixed_secondary_interturn_gap_mm"] != 0.35
            or profile["fixed_core_plate_thickness_mm"] != 20.0
            or profile["fixed_winding_cold_plate_thickness_mm"] != 20.0
            for profile in profiles
        )
        or any(
            task.get("screening_only") is not True
            or task.get("production_eligible") is not False
            or task.get("final_design_claim_allowed") is not False
            for task in tasks
        )
    ):
        raise RuntimeError(
            "feeder extension must be one aligned screening-only exact100 cohort"
        )
    return plan, tasks, authentication


def _validate_base_receipt(
    *,
    plan_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, _deployment, tasks, authentication = offload.authenticate_plan(
        plan_path
    )
    receipt = _read_json(receipt_path)
    expected = {
        payload["dedupe_key"]: payload
        for payload in (
            offload.scheduler_payload(plan=plan, task=task)
            for task in tasks
        )
    }
    validated = offload._validate_receipt(
        receipt,
        plan=plan,
        expected=expected,
        authentication=authentication,
        ready=receipt.get("remote_ready") or {},
        scheduler_url=str(receipt.get("scheduler_url") or ""),
    )
    return plan, validated


def _validate_recovery_receipt(
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = retry._validate_seal(
        retry._read_json(receipt_path), retry.RECEIPT_SCHEMA
    )
    plan_path = Path(str(receipt.get("retry_plan_path") or ""))
    if (
        receipt.get("apply") is not True
        or receipt.get("retry_plan_file_sha256")
        != retry._sha256_file(plan_path.resolve(strict=True))
        or receipt.get("absent_count") != 0
        or receipt.get("retry_task_count")
        != EXPECTED_RECOVERY_ATTEMPT_COUNT
        or receipt.get("allowed_accounts") != list(ALLOWED_ACCOUNTS)
        or receipt.get("failed_account_excluded") != retry.FAILED_ACCOUNT
        or receipt.get("warning_policy") != "python_-W_ignore"
        or receipt.get("equal_three_leg_air_gap_FEA_required") is not True
        or receipt.get("final_promotion_allowed") is not False
    ):
        raise RuntimeError("feeder recovery receipt contract mismatch")
    plan = retry._validate_plan(retry._read_json(plan_path))
    if (
        receipt.get("retry_plan_payload_sha256")
        != plan["payload_sha256"]
        or receipt.get("retry_task_count") != len(plan["tasks"])
    ):
        raise RuntimeError("feeder recovery plan binding mismatch")
    expected = {
        row["scheduler_payload"]["dedupe_key"]: row
        for row in plan["tasks"]
    }
    observed: set[str] = set()
    for row in receipt.get("tasks") or []:
        dedupe = str(row.get("dedupe_key") or "")
        source = expected.get(dedupe)
        task_id = row.get("task_id")
        if (
            source is None
            or dedupe in observed
            or row.get("seed") != source["seed"]
            or row.get("name") != source["scheduler_payload"]["name"]
            or row.get("requested_account")
            != source["requested_account"]
            or isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or row.get("status") not in ACTIVE_STATUSES | TERMINAL_STATUSES
        ):
            raise RuntimeError("feeder recovery task receipt mismatch")
        observed.add(dedupe)
    if observed != set(expected):
        raise RuntimeError("feeder recovery receipt inventory is incomplete")
    return plan, receipt


def initialize(
    *,
    base_plan_path: Path,
    base_receipt_path: Path,
    recovery_receipt_path: Path,
    extension_plan_path: Path,
    output_root: Path,
) -> Path:
    base_plan, base_receipt = _validate_base_receipt(
        plan_path=base_plan_path,
        receipt_path=base_receipt_path,
    )
    recovery_plan, recovery_receipt = _validate_recovery_receipt(
        recovery_receipt_path
    )
    extension_plan, tasks, _authentication = _validate_plan_and_tasks(
        extension_plan_path
    )
    base_seeds = {int(row["seed"]) for row in base_receipt["tasks"]}
    continuation_seeds = sorted(int(task["seed"]) for task in tasks)
    if base_seeds.intersection(continuation_seeds):
        raise RuntimeError("base and continuation feeder seeds overlap")
    recovery_seeds = {
        int(row["seed"]) for row in recovery_receipt["tasks"]
    }
    if not recovery_seeds.issubset(base_seeds):
        raise RuntimeError("recovery attempts escaped base logical seeds")
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "claims").mkdir(exist_ok=True)
    authority = _sealed(
        {
            "schema_version": AUTHORITY_SCHEMA,
            "created_at": _now(),
            "output_root": str(root),
            "scheduler_url": base_receipt["scheduler_url"],
            "base_plan": _file_record(base_plan_path),
            "base_receipt": _file_record(base_receipt_path),
            "base_bundle_id": base_plan["bundle_id"],
            "base_tasks": [
                {
                    "seed": int(row["seed"]),
                    "task_id": int(row["task_id"]),
                    "name": row["name"],
                    "dedupe_key": row["dedupe_key"],
                }
                for row in sorted(
                    base_receipt["tasks"], key=lambda item: int(item["seed"])
                )
            ],
            "recovery_plan": _file_record(
                Path(recovery_receipt["retry_plan_path"])
            ),
            "recovery_receipt": _file_record(recovery_receipt_path),
            "recovery_plan_payload_sha256": recovery_plan[
                "payload_sha256"
            ],
            "recovery_tasks": [
                {
                    "seed": int(row["seed"]),
                    "task_id": int(row["task_id"]),
                    "name": row["name"],
                    "dedupe_key": row["dedupe_key"],
                    "requested_account": row["requested_account"],
                }
                for row in sorted(
                    recovery_receipt["tasks"],
                    key=lambda item: int(item["seed"]),
                )
            ],
            "baseline_logical_seed_count": 100,
            "baseline_attempt_count": (
                len(base_receipt["tasks"])
                + len(recovery_receipt["tasks"])
            ),
            "extension_plan": _file_record(extension_plan_path),
            "extension_bundle_id": extension_plan["bundle_id"],
            "extension_plan_contract_sha256": extension_plan[
                "contract_sha256"
            ],
            "extension_diagnostic_plan_sha256": extension_plan[
                "diagnostic_plan_sha256"
            ],
            "continuation_seed_start": continuation_seeds[0],
            "continuation_seed_end_inclusive": continuation_seeds[-1],
            "continuation_seed_count": len(continuation_seeds),
            "target_active_count": TARGET_ACTIVE_COUNT,
            "active_statuses": sorted(ACTIVE_STATUSES),
            "allowed_accounts": list(ALLOWED_ACCOUNTS),
            "account_assignment": "seed_offset_modulo_allowed_accounts",
            "maximum_posts_per_cycle": MAXIMUM_POSTS_PER_CYCLE,
            "scheduler_methods": ["GET", "POST /api/tasks"],
            "scheduler_cancel_allowed": False,
            "scheduler_preempt_allowed": False,
            "scheduler_repository_modified": False,
            "scheduler_and_mft_functionality_mixed": False,
            "screening_only": True,
            "production_eligible": False,
            "final_design_claim_allowed": False,
            "automatic_promotion_allowed": False,
            "final_promotion_allowed": False,
            "equal_three_leg_air_gap_FEA_required": True,
            "symmetric_FEA_validation_still_required": True,
        }
    )
    path = root / "feeder_authority.json"
    _immutable_json(path, authority)
    return path


def load_authority(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    authority = _validate_seal(
        _read_json(path), schema=AUTHORITY_SCHEMA
    )
    root = Path(authority["output_root"]).resolve(strict=True)
    if path.resolve(strict=True) != root / "feeder_authority.json":
        raise RuntimeError("feeder authority escaped its output root")
    if (
        authority.get("target_active_count") != TARGET_ACTIVE_COUNT
        or authority.get("allowed_accounts") != list(ALLOWED_ACCOUNTS)
        or authority.get("scheduler_methods") != ["GET", "POST /api/tasks"]
        or authority.get("scheduler_cancel_allowed") is not False
        or authority.get("scheduler_preempt_allowed") is not False
        or authority.get("final_promotion_allowed") is not False
        or authority.get("equal_three_leg_air_gap_FEA_required") is not True
    ):
        raise RuntimeError("feeder safety authority drifted")
    for key in (
        "base_plan",
        "base_receipt",
        "recovery_plan",
        "recovery_receipt",
        "extension_plan",
    ):
        record = authority.get(key) or {}
        source = Path(str(record.get("path") or ""))
        if _file_record(source) != record:
            raise RuntimeError(f"feeder {key} bytes changed")
    base_plan, base_receipt = _validate_base_receipt(
        plan_path=Path(authority["base_plan"]["path"]),
        receipt_path=Path(authority["base_receipt"]["path"]),
    )
    if (
        base_plan["bundle_id"] != authority["base_bundle_id"]
        or authority["base_tasks"]
        != [
            {
                "seed": int(row["seed"]),
                "task_id": int(row["task_id"]),
                "name": row["name"],
                "dedupe_key": row["dedupe_key"],
            }
            for row in sorted(
                base_receipt["tasks"], key=lambda item: int(item["seed"])
            )
        ]
    ):
        raise RuntimeError("feeder base receipt binding drifted")
    recovery_plan, recovery_receipt = _validate_recovery_receipt(
        Path(authority["recovery_receipt"]["path"])
    )
    recovery_tasks = [
        {
            "seed": int(row["seed"]),
            "task_id": int(row["task_id"]),
            "name": row["name"],
            "dedupe_key": row["dedupe_key"],
            "requested_account": row["requested_account"],
        }
        for row in sorted(
            recovery_receipt["tasks"], key=lambda item: int(item["seed"])
        )
    ]
    if (
        recovery_plan["payload_sha256"]
        != authority["recovery_plan_payload_sha256"]
        or recovery_tasks != authority["recovery_tasks"]
        or authority.get("baseline_logical_seed_count") != 100
        or authority.get("baseline_attempt_count")
        != len(authority["base_tasks"]) + len(recovery_tasks)
        or authority["baseline_attempt_count"] != 147
    ):
        raise RuntimeError("feeder recovery receipt binding drifted")
    extension_plan, tasks, _authentication = _validate_plan_and_tasks(
        Path(authority["extension_plan"]["path"])
    )
    seeds = sorted(int(task["seed"]) for task in tasks)
    if (
        extension_plan["bundle_id"] != authority["extension_bundle_id"]
        or extension_plan["contract_sha256"]
        != authority["extension_plan_contract_sha256"]
        or extension_plan["diagnostic_plan_sha256"]
        != authority["extension_diagnostic_plan_sha256"]
        or seeds[0] != authority["continuation_seed_start"]
        or seeds[-1] != authority["continuation_seed_end_inclusive"]
        or len(seeds) != authority["continuation_seed_count"]
    ):
        raise RuntimeError("feeder continuation plan binding drifted")
    return authority, extension_plan, tasks


def _assigned_account(authority: Mapping[str, Any], seed: int) -> str:
    start = int(authority["continuation_seed_start"])
    accounts = tuple(authority["allowed_accounts"])
    if (
        seed < start
        or seed > int(authority["continuation_seed_end_inclusive"])
        or accounts != ALLOWED_ACCOUNTS
    ):
        raise RuntimeError("feeder seed/account authority mismatch")
    return accounts[(seed - start) % len(accounts)]


def pinned_scheduler_payload(
    *,
    authority: Mapping[str, Any],
    plan: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    seed = int(task["seed"])
    account = _assigned_account(authority, seed)
    base = offload.scheduler_payload(plan=plan, task=task)
    envelope = {
        key: copy.deepcopy(value)
        for key, value in base.items()
        if key != "dedupe_key"
    }
    command = str(envelope.get("command") or "")
    if command.count("exec python ") != 1 or "exec python -W ignore " in command:
        raise RuntimeError("feeder worker command identity is unexpected")
    envelope["command"] = command.replace(
        "exec python ", "exec python -W ignore ", 1
    )
    envelope["account_name"] = account
    dedupe = canonical_sha256(
        {
            "schema_version": (
                "mft-goal-diagnostic-compact-target100-dedupe-v1"
            ),
            "feeder_authority_payload_sha256": authority["payload_sha256"],
            "base_scheduler_dedupe_key": base["dedupe_key"],
            "pinned_scheduler_envelope": envelope,
        }
    )
    return {**envelope, "dedupe_key": DEDUPE_PREFIX + dedupe}


def _expected_payloads(
    authority: Mapping[str, Any],
    plan: Mapping[str, Any],
    tasks: Iterable[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    result = {
        int(task["seed"]): pinned_scheduler_payload(
            authority=authority, plan=plan, task=task
        )
        for task in tasks
    }
    if (
        len(result) != authority["continuation_seed_count"]
        or sorted(result)
        != list(
            range(
                authority["continuation_seed_start"],
                authority["continuation_seed_end_inclusive"] + 1,
            )
        )
        or any(
            payload["account_name"] not in ALLOWED_ACCOUNTS
            or not payload["dedupe_key"].startswith(DEDUPE_PREFIX)
            for payload in result.values()
        )
    ):
        raise RuntimeError("feeder scheduler payload inventory mismatch")
    return result


def _inventory(
    scheduler: Scheduler,
    *,
    plan: Mapping[str, Any],
    expected: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    prefix = offload._task_name_prefix(plan)
    expected_by_dedupe = {
        payload["dedupe_key"]: (seed, payload)
        for seed, payload in expected.items()
    }
    result: dict[int, dict[str, Any]] = {}
    for row in scheduler.list_namespace_tasks(prefix):
        dedupe = str(row.get("dedupe_key") or "")
        binding = expected_by_dedupe.get(dedupe)
        if binding is None:
            raise RuntimeError("foreign task occupies feeder namespace")
        seed, payload = binding
        observation = scheduler_task_observation(row)
        if (
            observation is None
            or row.get("name") != payload["name"]
            or seed in result
        ):
            raise RuntimeError("feeder inventory identity mismatch")
        result[seed] = dict(row)
    return result


def _authenticated_detail(
    scheduler: Scheduler,
    *,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    observation = scheduler_task_observation(row)
    if observation is None:
        raise RuntimeError("feeder task observation is invalid")
    detail = scheduler.get_task(observation[0])
    status = str(detail.get("status") or "")
    if (
        detail.get("name") != payload["name"]
        or detail.get("dedupe_key") != payload["dedupe_key"]
        or detail.get("requested_account_name") != payload["account_name"]
        or status not in ACTIVE_STATUSES | TERMINAL_STATUSES
    ):
        raise RuntimeError("feeder Scheduler readback drifted")
    return detail


def _claim_paths(root: Path, seed: int) -> tuple[Path, Path]:
    directory = root / "claims" / f"seed-{seed}"
    return directory / "pending.json", directory / "finalized.json"


def _pending(
    authority: Mapping[str, Any], payload: Mapping[str, Any]
) -> dict[str, Any]:
    return _sealed(
        {
            "schema_version": PENDING_SCHEMA,
            "feeder_authority_payload_sha256": authority["payload_sha256"],
            "seed": int(payload["payload_json"]["seed"]),
            "name": payload["name"],
            "dedupe_key": payload["dedupe_key"],
            "account_name": payload["account_name"],
            "scheduler_payload_sha256": canonical_sha256(payload),
            "idempotent_replay_with_same_dedupe_allowed": True,
            "maximum_logical_tasks_for_seed": 1,
        }
    )


def _acquire_pending(
    *,
    authority: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    root = Path(authority["output_root"]).resolve(strict=True)
    pending_path, finalized_path = _claim_paths(
        root, int(payload["payload_json"]["seed"])
    )
    pending_path.parent.mkdir(parents=False, exist_ok=True)
    expected = _pending(authority, payload)
    _immutable_json(pending_path, expected)
    if _validate_seal(_read_json(pending_path), schema=PENDING_SCHEMA) != expected:
        raise RuntimeError("feeder pending claim changed")
    return expected, finalized_path


def _finalize(
    *,
    pending: Mapping[str, Any],
    finalized_path: Path,
    task_id: int,
) -> dict[str, Any]:
    value = _sealed(
        {
            "schema_version": FINALIZED_SCHEMA,
            "pending_payload_sha256": pending["payload_sha256"],
            "seed": pending["seed"],
            "task_id": int(task_id),
            "dedupe_key": pending["dedupe_key"],
            "account_name": pending["account_name"],
            "scheduler_identity_verified_by_GET": True,
            "maximum_logical_tasks_for_seed": 1,
        }
    )
    _immutable_json(finalized_path, value)
    observed = _validate_seal(
        _read_json(finalized_path), schema=FINALIZED_SCHEMA
    )
    if observed != value:
        raise RuntimeError("feeder finalized claim changed")
    return value


def _load_claimed_seeds(
    authority: Mapping[str, Any],
    expected: Mapping[int, Mapping[str, Any]],
) -> tuple[set[int], set[int]]:
    root = Path(authority["output_root"]).resolve(strict=True)
    pending: set[int] = set()
    finalized: set[int] = set()
    claims = root / "claims"
    for directory in claims.iterdir():
        if not directory.is_dir() or not directory.name.startswith("seed-"):
            raise RuntimeError("unexpected feeder claim entry")
        try:
            seed = int(directory.name.removeprefix("seed-"))
        except ValueError as exc:
            raise RuntimeError("invalid feeder claim seed") from exc
        payload = expected.get(seed)
        if payload is None:
            raise RuntimeError("feeder claim is outside continuation authority")
        pending_path, finalized_path = _claim_paths(root, seed)
        observed_pending = _validate_seal(
            _read_json(pending_path), schema=PENDING_SCHEMA
        )
        if observed_pending != _pending(authority, payload):
            raise RuntimeError("feeder pending claim authority mismatch")
        pending.add(seed)
        if finalized_path.exists():
            value = _validate_seal(
                _read_json(finalized_path), schema=FINALIZED_SCHEMA
            )
            if (
                value.get("pending_payload_sha256")
                != observed_pending["payload_sha256"]
                or value.get("seed") != seed
                or value.get("dedupe_key") != payload["dedupe_key"]
                or value.get("account_name") != payload["account_name"]
                or value.get("scheduler_identity_verified_by_GET") is not True
            ):
                raise RuntimeError("feeder finalized claim authority mismatch")
            finalized.add(seed)
    return pending, finalized


@contextmanager
def _controller_lock(root: Path) -> Iterator[None]:
    path = root / "controller.lock"
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("feeder controller already active") from exc
        else:  # pragma: no cover - deployed controller is Windows
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("feeder controller already active") from exc
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def run_cycle(
    *,
    authority_path: Path,
    receipt_out: Path,
    apply: bool = False,
    scheduler: Scheduler | None = None,
) -> dict[str, Any]:
    authority, plan, tasks = load_authority(authority_path)
    root = Path(authority["output_root"]).resolve(strict=True)
    client = scheduler or offload.SchedulerClient(authority["scheduler_url"])
    with _controller_lock(root):
        expected = _expected_payloads(authority, plan, tasks)
        inventory = _inventory(client, plan=plan, expected=expected)
        pending, finalized = _load_claimed_seeds(authority, expected)
        if set(inventory) - pending:
            raise RuntimeError(
                "feeder namespace contains a task without a durable claim"
            )
        if finalized - set(inventory):
            raise RuntimeError(
                "feeder finalized claim lost its Scheduler task"
            )
        recovered_actions: list[dict[str, Any]] = []
        if apply:
            for seed in sorted((pending - finalized) & set(inventory)):
                payload = expected[seed]
                detail = _authenticated_detail(
                    client, row=inventory[seed], payload=payload
                )
                pending_path, finalized_path = _claim_paths(root, seed)
                pending_claim = _validate_seal(
                    _read_json(pending_path), schema=PENDING_SCHEMA
                )
                finalized_claim = _finalize(
                    pending=pending_claim,
                    finalized_path=finalized_path,
                    task_id=int(
                        detail.get("task_id", detail.get("id"))
                    ),
                )
                finalized.add(seed)
                recovered_actions.append(
                    {
                        "seed": seed,
                        "task_id": int(
                            detail.get("task_id", detail.get("id"))
                        ),
                        "account_name": payload["account_name"],
                        "status": detail["status"],
                        "source": "recovered_pending_inventory",
                        "finalized_claim_payload_sha256": finalized_claim[
                            "payload_sha256"
                        ],
                    }
                )

        base_rows: list[dict[str, Any]] = []
        recovery_rows: list[dict[str, Any]] = []
        base_active = 0
        recovery_active = 0
        for lane, bindings in (
            ("base", authority["base_tasks"]),
            ("recovery", authority["recovery_tasks"]),
        ):
            for binding in bindings:
                detail = client.get_task(int(binding["task_id"]))
                status = str(detail.get("status") or "")
                if (
                    detail.get("name") != binding["name"]
                    or detail.get("dedupe_key") != binding["dedupe_key"]
                    or status not in ACTIVE_STATUSES | TERMINAL_STATUSES
                    or (
                        lane == "recovery"
                        and detail.get("requested_account_name")
                        != binding["requested_account"]
                    )
                ):
                    raise RuntimeError(
                        f"{lane} Scheduler binding drifted"
                    )
                row = {
                    "seed": binding["seed"],
                    "task_id": binding["task_id"],
                    "status": status,
                }
                if lane == "base":
                    base_active += status in ACTIVE_STATUSES
                    base_rows.append(row)
                else:
                    recovery_active += status in ACTIVE_STATUSES
                    recovery_rows.append(row)

        extension_rows: dict[int, dict[str, Any]] = {}
        extension_active = 0
        for seed, row in sorted(inventory.items()):
            detail = _authenticated_detail(
                client, row=row, payload=expected[seed]
            )
            status = str(detail["status"])
            extension_active += status in ACTIVE_STATUSES
            extension_rows[seed] = {
                "seed": seed,
                "task_id": int(detail.get("task_id", detail.get("id"))),
                "status": status,
                "account_name": expected[seed]["account_name"],
            }

        active_before = base_active + recovery_active + extension_active
        deficit = max(0, TARGET_ACTIVE_COUNT - active_before)
        candidates = [
            seed
            for seed in sorted(expected)
            if seed not in finalized and seed not in inventory
        ][:deficit]
        actions: list[dict[str, Any]] = list(recovered_actions)
        if apply:
            for seed in candidates:
                payload = expected[seed]
                pending_claim, finalized_path = _acquire_pending(
                    authority=authority, payload=payload
                )
                refreshed = _inventory(client, plan=plan, expected=expected)
                row = refreshed.get(seed)
                post_source = "recovered_inventory"
                if row is None:
                    response = client.submit_task(payload)
                    task_id = response.get("task_id", response.get("id"))
                    if (
                        isinstance(task_id, bool)
                        or not isinstance(task_id, int)
                        or task_id <= 0
                    ):
                        raise RuntimeError(
                            "feeder Scheduler POST returned no task ID"
                        )
                    row = client.get_task(task_id)
                    post_source = (
                        "dedupe_replay"
                        if response.get("deduped") is True
                        else "submitted"
                    )
                detail = _authenticated_detail(
                    client, row=row, payload=payload
                )
                task_id = int(
                    detail.get("task_id", detail.get("id"))
                )
                finalized_claim = _finalize(
                    pending=pending_claim,
                    finalized_path=finalized_path,
                    task_id=task_id,
                )
                actions.append(
                    {
                        "seed": seed,
                        "task_id": task_id,
                        "account_name": payload["account_name"],
                        "status": detail["status"],
                        "source": post_source,
                        "finalized_claim_payload_sha256": finalized_claim[
                            "payload_sha256"
                        ],
                    }
                )
        else:
            actions = [
                {
                    "seed": seed,
                    "account_name": expected[seed]["account_name"],
                    "status": "would_submit",
                }
                for seed in candidates
            ]

        value = _sealed(
            {
                "schema_version": CYCLE_SCHEMA,
                "observed_at": _now(),
                "apply": bool(apply),
                "feeder_authority_payload_sha256": authority[
                    "payload_sha256"
                ],
                "target_active_count": TARGET_ACTIVE_COUNT,
                "active_before": active_before,
                "base_active_count": base_active,
                "recovery_active_count": recovery_active,
                "baseline_attempt_count": authority[
                    "baseline_attempt_count"
                ],
                "extension_active_count": extension_active,
                "deficit_before": deficit,
                "selected_seed_count": len(candidates),
                "selected_seeds": candidates,
                "actions": actions,
                "base_status_counts": {
                    status: sum(
                        row["status"] == status for row in base_rows
                    )
                    for status in sorted(
                        {row["status"] for row in base_rows}
                    )
                },
                "extension_status_counts": {
                    status: sum(
                        row["status"] == status
                        for row in extension_rows.values()
                    )
                    for status in sorted(
                        {
                            row["status"]
                            for row in extension_rows.values()
                        }
                    )
                },
                "recovery_status_counts": {
                    status: sum(
                        row["status"] == status for row in recovery_rows
                    )
                    for status in sorted(
                        {row["status"] for row in recovery_rows}
                    )
                },
                "remaining_unclaimed_seed_count": (
                    len(expected)
                    - len(finalized)
                    - len(candidates)
                ),
                "scheduler_get_count": client.get_count,
                "scheduler_post_count": client.post_count,
                "scheduler_cancel_count": 0,
                "scheduler_preempt_count": 0,
                "allowed_accounts": list(ALLOWED_ACCOUNTS),
                "screening_only": True,
                "production_eligible": False,
                "final_design_claim_allowed": False,
                "automatic_promotion_allowed": False,
                "final_promotion_allowed": False,
                "equal_three_leg_air_gap_FEA_required": True,
            }
        )
        if client.post_count > MAXIMUM_POSTS_PER_CYCLE:
            raise RuntimeError("feeder per-cycle POST ceiling exceeded")
        if apply:
            _immutable_json(receipt_out, value)
        else:
            _atomic_json(receipt_out, value)
        return value


def _next_cycle_index(
    cycles_root: Path,
    *,
    authority_payload_sha256: str,
) -> int:
    cycles_root.mkdir(parents=True, exist_ok=True)
    indices: list[int] = []
    for path in cycles_root.iterdir():
        match = _CYCLE_NAME.fullmatch(path.name)
        if match is None or not path.is_file():
            raise RuntimeError("unexpected target100 cycle artifact")
        value = _validate_seal(_read_json(path), schema=CYCLE_SCHEMA)
        if (
            value.get("feeder_authority_payload_sha256")
            != authority_payload_sha256
        ):
            raise RuntimeError("target100 cycle authority changed")
        indices.append(int(match.group(1)))
    indices.sort()
    if indices != list(range(1, len(indices) + 1)):
        raise RuntimeError("target100 cycle sequence is not contiguous")
    return len(indices) + 1


def watch(
    *,
    authority_path: Path,
    apply: bool = False,
    poll_seconds: float = 30.0,
    max_cycles: int = 0,
) -> dict[str, Any]:
    if (
        isinstance(max_cycles, bool)
        or not isinstance(max_cycles, int)
        or max_cycles < 0
        or not 1.0 <= float(poll_seconds) <= 3600.0
    ):
        raise RuntimeError("target100 watch cadence is invalid")
    authority, _plan, _tasks = load_authority(authority_path)
    cycles_root = (
        Path(authority["output_root"]).resolve(strict=True) / "cycles"
    )
    completed = 0
    last: dict[str, Any] | None = None
    while max_cycles == 0 or completed < max_cycles:
        index = _next_cycle_index(
            cycles_root,
            authority_payload_sha256=authority["payload_sha256"],
        )
        receipt = cycles_root / f"cycle-{index:08d}.json"
        last = run_cycle(
            authority_path=authority_path,
            receipt_out=receipt,
            apply=apply,
        )
        print(
            json.dumps(
                {
                    "cycle_index": index,
                    "receipt": str(receipt),
                    "active_before": last["active_before"],
                    "selected_seeds": last["selected_seeds"],
                    "scheduler_post_count": last[
                        "scheduler_post_count"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        completed += 1
        if max_cycles == 0 or completed < max_cycles:
            time.sleep(float(poll_seconds))
    assert last is not None
    return last


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize")
    init.add_argument("--base-plan", type=Path, required=True)
    init.add_argument("--base-receipt", type=Path, required=True)
    init.add_argument("--recovery-receipt", type=Path, required=True)
    init.add_argument("--extension-plan", type=Path, required=True)
    init.add_argument("--output-root", type=Path, required=True)
    cycle = commands.add_parser("cycle")
    cycle.add_argument("--authority", type=Path, required=True)
    cycle.add_argument("--receipt-out", type=Path, required=True)
    cycle.add_argument("--apply", action="store_true")
    watch_parser = commands.add_parser("watch")
    watch_parser.add_argument("--authority", type=Path, required=True)
    watch_parser.add_argument("--poll-seconds", type=float, default=30.0)
    watch_parser.add_argument("--max-cycles", type=int, default=0)
    watch_parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "initialize":
        print(
            initialize(
                base_plan_path=args.base_plan,
                base_receipt_path=args.base_receipt,
                recovery_receipt_path=args.recovery_receipt,
                extension_plan_path=args.extension_plan,
                output_root=args.output_root,
            )
        )
    elif args.command == "cycle":
        value = run_cycle(
            authority_path=args.authority,
            receipt_out=args.receipt_out,
            apply=args.apply,
        )
        print(json.dumps(value, sort_keys=True))
    else:
        watch(
            authority_path=args.authority,
            apply=args.apply,
            poll_seconds=args.poll_seconds,
            max_cycles=args.max_cycles,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
