"""Continuously harvest final1000 surrogate tasks into four isolated indexes.

This process is read-only with respect to the scheduler and every remote
account.  Its only writes are beneath ``--runtime``.  Each final-goal stage is
published as an independent, fail-closed Current7-v1 condition index so that
different size/temperature contracts can never be aggregated together.

The existing Current7 primary index is neither read as authority nor written.
The default protected path is used solely for a local path-containment guard.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_harvest import (
        AccountSftpReader,
        CACHE_RECORD_SCHEMA,
        CURRENT7_INDEX_SCHEMA,
        MAX_ARTIFACT_BYTES,
        MAX_RESULT_BYTES,
        MAX_STATUS_BYTES,
        ReadOnlySchedulerApi,
        RemoteFileStat,
        RemoteReader,
        _atomic_replace,
        _json_bytes,
        _sha_bytes,
        _strict_json,
        build_current7_snapshot,
        cache_seed_records,
        deduplicate_observations,
        harvest_terminal_task,
        publish_current7_snapshot,
    )
    from tier1_corrected_current7_slurm_seed_runner import (
        validate_result as validate_current7_result,
    )
    from tier1_final1000_slurm_controller import (
        STATE_SCHEMA as CONTROLLER_STATE_SCHEMA,
        TASK_NAME_PREFIX,
        DEDUPE_PREFIX,
        _task_for_entry,
        _task_index,
        _task_templates,
        _validate_state,
    )
    from tier1_final1000_slurm_launch import (
        load_stage_bindings,
        validate_launch_plan,
        validate_stage_result,
        validate_task,
    )
    from tier1_final1000_stage_profiles import BY_ID, STAGES, stage_profile
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_harvest import (
        AccountSftpReader,
        CACHE_RECORD_SCHEMA,
        CURRENT7_INDEX_SCHEMA,
        MAX_ARTIFACT_BYTES,
        MAX_RESULT_BYTES,
        MAX_STATUS_BYTES,
        ReadOnlySchedulerApi,
        RemoteFileStat,
        RemoteReader,
        _atomic_replace,
        _json_bytes,
        _sha_bytes,
        _strict_json,
        build_current7_snapshot,
        cache_seed_records,
        deduplicate_observations,
        harvest_terminal_task,
        publish_current7_snapshot,
    )
    from tools.tier1_corrected_current7_slurm_seed_runner import (
        validate_result as validate_current7_result,
    )
    from tools.tier1_final1000_slurm_controller import (
        STATE_SCHEMA as CONTROLLER_STATE_SCHEMA,
        TASK_NAME_PREFIX,
        DEDUPE_PREFIX,
        _task_for_entry,
        _task_index,
        _task_templates,
        _validate_state,
    )
    from tools.tier1_final1000_slurm_launch import (
        load_stage_bindings,
        validate_launch_plan,
        validate_stage_result,
        validate_task,
    )
    from tools.tier1_final1000_stage_profiles import (
        BY_ID,
        STAGES,
        stage_profile,
    )


HARVEST_SCHEMA = "mft-tier1-final1000-slurm-harvest-result-v1"
INDEX_INVENTORY_SCHEMA = "mft-tier1-final1000-condition-index-inventory-v1"
SCHEDULER_CACHE_SCHEMA = "mft-tier1-final1000-terminal-scheduler-cache-v1"

DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_RUNTIME = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_tier1_final_goal_1000_t100_resmax20_260721"
)
DEFAULT_CURRENT7_INDEX = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_tier1_nsga_slurm_rolling\canonical\current7-index.json"
)
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"Y:\git\slurm_scheduler")

VISIBLE_SCHEDULER_STATES = frozenset(
    {"queued", "attaching", "running", "completed", "failed", "cancelled", "timeout"}
)
COMPLETED_STATE = "completed"
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "timeout"})


class SchedulerReader(Protocol):
    get_count: int

    def get_task(self, task_id: int) -> Mapping[str, Any] | None: ...


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"JSON evidence is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence must be an object: {path}")
    return value


def _observed_at() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(tzinfo=None)
        .isoformat(sep=" ", timespec="seconds")
    )


def _assert_runtime_isolated(runtime: Path, protected_index: Path) -> None:
    runtime = runtime.resolve()
    protected = protected_index.resolve()
    protected_root = protected.parent.parent
    if (
        protected.is_relative_to(runtime)
        or runtime.is_relative_to(protected_root)
        or runtime == protected_root
    ):
        raise RuntimeError(
            "final1000 runtime overlaps the protected Current7 primary runtime"
        )


def _validate_inputs(
    launch_plan_path: Path,
    bindings_path: Path,
    controller_state_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    plan = validate_launch_plan(
        _read_json(launch_plan_path.resolve(strict=True))
    )
    bindings = load_stage_bindings(bindings_path)
    state = _validate_state(
        _read_json(controller_state_path.resolve(strict=True)), plan
    )
    if state.get("schema_version") != CONTROLLER_STATE_SCHEMA:
        raise RuntimeError("final1000 controller state schema drifted")
    for stage in STAGES:
        summary = plan["stage_bindings"][stage.stage_id]
        binding = bindings[stage.stage_id]
        bundle_plan = binding["plan"]
        if (
            summary.get("bundle_id") != bundle_plan.get("bundle_id")
            or summary.get("bundle_manifest_sha256")
            != bundle_plan.get("bundle_manifest_sha256")
            or summary.get("remote_bundle") != bundle_plan.get("remote_bundle")
            or summary.get("stage_spec_sha256")
            != stage_profile(stage)["stage_spec_sha256"]
        ):
            raise RuntimeError(
                f"{stage.stage_id} launch/bundle binding identity mismatch"
            )
    return plan, bindings, state


def _scheduler_cache_path(stage_runtime: Path, task_id: int) -> Path:
    return stage_runtime / "scheduler-cache" / f"task-{int(task_id)}.json"


def _sealed_scheduler_cache(task: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": SCHEDULER_CACHE_SCHEMA,
        "task": copy.deepcopy(dict(task)),
    }
    return {**value, "sha256": canonical_sha256(value)}


def _load_scheduler_cache(
    path: Path, *, task_id: int, dedupe_key: str
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _read_json(path)
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    task = value.get("task")
    if (
        value.get("schema_version") != SCHEDULER_CACHE_SCHEMA
        or value.get("sha256") != canonical_sha256(unsigned)
        or not isinstance(task, dict)
        or int(task.get("id") or task.get("task_id") or 0) != int(task_id)
        or task.get("dedupe_key") != dedupe_key
        or str(task.get("status") or "").lower() not in TERMINAL_STATES
    ):
        raise RuntimeError("terminal scheduler cache identity mismatch")
    return copy.deepcopy(task)


def _inventory_tasks(
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    scheduler: SchedulerReader,
    runtime: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[Path, bytes]], int]:
    plan_index = _task_index(plan)
    templates = _task_templates(plan)
    by_stage = {stage.stage_id: [] for stage in STAGES}
    pending_cache: list[tuple[Path, bytes]] = []
    cache_hits = 0
    seen_task_ids: set[int] = set()
    for entry in state["entries"]:
        task_id = entry.get("task_id")
        if task_id is None:
            continue
        task_id = int(task_id)
        if task_id in seen_task_ids:
            raise RuntimeError("final1000 controller task id is duplicated")
        seen_task_ids.add(task_id)
        expected = _task_for_entry(
            entry, plan_index=plan_index, templates=templates
        )
        validate_task(expected, expected_stage=BY_ID[entry["stage_id"]])
        stage_runtime = runtime / "conditions" / str(entry["stage_id"])
        cache_path = _scheduler_cache_path(stage_runtime, task_id)
        observed = _load_scheduler_cache(
            cache_path,
            task_id=task_id,
            dedupe_key=str(entry["dedupe_key"]),
        )
        if observed is not None:
            cache_hits += 1
        else:
            value = scheduler.get_task(task_id)
            if value is None:
                raise RuntimeError(f"scheduler task disappeared: {task_id}")
            observed = copy.deepcopy(dict(value))
        observed_id = int(observed.get("id") or observed.get("task_id") or 0)
        scheduler_state = str(observed.get("status") or "").lower()
        if (
            observed_id != task_id
            or observed.get("name") != expected["name"]
            or observed.get("dedupe_key") != expected["dedupe_key"]
            or not str(observed.get("name") or "").startswith(TASK_NAME_PREFIX)
            or not str(observed.get("dedupe_key") or "").startswith(DEDUPE_PREFIX)
            or scheduler_state not in VISIBLE_SCHEDULER_STATES
        ):
            raise RuntimeError(f"scheduler task identity changed: {task_id}")
        if scheduler_state in TERMINAL_STATES and not cache_path.is_file():
            pending_cache.append(
                (cache_path, _json_bytes(_sealed_scheduler_cache(observed)))
            )
        payload = copy.deepcopy(dict(expected["payload_json"]))
        lane = payload["lane"]
        by_stage[str(entry["stage_id"])].append(
            {
                "task_id": task_id,
                "name": expected["name"],
                "dedupe_key": expected["dedupe_key"],
                "bundle_id": payload["bundle_id"],
                "seed": int(payload["seed"]),
                "island_id": lane["island_id"],
                "wave": lane["wave"],
                "status": scheduler_state,
                "account_name": str(observed.get("account_name") or ""),
                "slurm_job_id": str(observed.get("slurm_job_id") or ""),
                "exit_code": observed.get("exit_code"),
                "created_at": observed.get("created_at"),
                "started_at": observed.get("started_at"),
                "finished_at": observed.get("finished_at"),
                "payload": payload,
                "payload_sha256": canonical_sha256(payload),
                "_expected_task": expected,
            }
        )
    for items in by_stage.values():
        items.sort(key=lambda item: int(item["task_id"]))
    return by_stage, pending_cache, cache_hits


class _LocalCacheReader:
    def __init__(self, files: Mapping[str, Path]):
        self.files = {str(key): Path(value) for key, value in files.items()}

    def _path(self, remote_path: str) -> Path:
        value = self.files.get(str(remote_path))
        if value is None:
            raise RuntimeError(f"cached remote identity is unknown: {remote_path}")
        return value

    def stat(self, _account_name: str, path: str) -> RemoteFileStat:
        value = self._path(path).stat()
        return RemoteFileStat(
            size=int(value.st_size),
            mtime=int(value.st_mtime_ns),
            mode=int(value.st_mode),
        )

    def read_bytes(
        self, _account_name: str, path: str, *, maximum_bytes: int
    ) -> bytes:
        value = self._path(path).read_bytes()
        if len(value) > int(maximum_bytes):
            raise RuntimeError(f"cached artifact exceeds byte limit: {path}")
        return value


def _cache_object_path(
    cache: Path,
    identity: Mapping[str, Any],
    *,
    suffix: str,
    maximum_bytes: int,
) -> Path:
    digest = str(identity.get("sha256") or "").lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError("cached object SHA is malformed")
    expected = (
        cache / "objects" / "sha256" / digest[:2] / f"{digest}{suffix}"
    ).resolve()
    if Path(str(identity.get("local_cache_path") or "")).resolve() != expected:
        raise RuntimeError("cached object path identity mismatch")
    if not expected.is_file():
        raise RuntimeError("cached object is missing")
    payload = expected.read_bytes()
    if (
        len(payload) <= 0
        or len(payload) > maximum_bytes
        or _sha_bytes(payload) != digest
    ):
        raise RuntimeError("cached object bytes failed authentication")
    return expected


def _load_cached_completed(
    item: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    runtime: Path,
    result_validator: Callable[..., None],
) -> dict[str, Any] | None:
    cache = runtime / "current7-cache"
    ref = cache / "refs" / str(plan["bundle_id"]) / f"seed-{item['seed']}.json"
    if not ref.is_file():
        return None
    record = _strict_json(ref.read_bytes(), str(ref))
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    task_id = int(item["task_id"])
    if (
        record.get("schema_version") != CACHE_RECORD_SCHEMA
        or record.get("record_sha256") != canonical_sha256(unsigned)
        or record.get("bundle_id") != plan["bundle_id"]
        or int(record.get("seed", -1)) != int(item["seed"])
        or record.get("island_id") != item["island_id"]
        or record.get("terminal_state") != COMPLETED_STATE
        or task_id not in {int(value) for value in record.get("task_ids") or []}
        or record.get("authenticated") is not True
    ):
        raise RuntimeError("cached completed seed record identity mismatch")
    status_objects = [
        value
        for value in record.get("status_objects") or []
        if isinstance(value, dict) and int(value.get("task_id") or 0) == task_id
    ]
    if len(status_objects) != 1:
        raise RuntimeError("cached terminal status identity is not unique")
    files: dict[str, Path] = {}
    status = status_objects[0]
    files[str(status["remote_path"])] = _cache_object_path(
        cache, status, suffix=".json", maximum_bytes=MAX_STATUS_BYTES
    )
    result = record.get("result_object")
    if not isinstance(result, dict):
        raise RuntimeError("cached completed seed has no result")
    files[str(result["remote_path"])] = _cache_object_path(
        cache, result, suffix=".json", maximum_bytes=MAX_RESULT_BYTES
    )
    artifacts = record.get("artifact_objects")
    if not isinstance(artifacts, dict):
        raise RuntimeError("cached artifact inventory is invalid")
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            raise RuntimeError("cached artifact identity is invalid")
        files[str(artifact["remote_path"])] = _cache_object_path(
            cache,
            artifact,
            suffix=".artifact",
            maximum_bytes=MAX_ARTIFACT_BYTES,
        )
    return harvest_terminal_task(
        item,
        plan=plan,
        manifest=manifest,
        remote=_LocalCacheReader(files),
        result_validator=result_validator,
    )


def _result_validator(expected_task: Mapping[str, Any]) -> Callable[..., None]:
    def validate(
        result: Mapping[str, Any],
        *,
        payload: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> None:
        validate_current7_result(result, payload=payload, manifest=manifest)
        validate_stage_result(result, expected_task)

    return validate


def _project_stage(
    *,
    stage_id: str,
    binding: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    remote: RemoteReader,
    runtime: Path,
    observed_at: str,
    fallback_event_at: str,
) -> dict[str, Any]:
    plan = binding["plan"]
    manifest = binding["manifest"]
    observations: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    cached_reuse_count = 0
    terminal_failures = [
        item for item in inventory if item["status"] in TERMINAL_STATES - {COMPLETED_STATE}
    ]
    for item in inventory:
        if item["status"] != COMPLETED_STATE:
            continue
        validator = _result_validator(item["_expected_task"])
        public_item = {
            key: copy.deepcopy(value)
            for key, value in item.items()
            if not str(key).startswith("_")
        }
        try:
            observation = _load_cached_completed(
                public_item,
                plan=plan,
                manifest=manifest,
                runtime=runtime,
                result_validator=validator,
            )
            if observation is None:
                observation = harvest_terminal_task(
                    public_item,
                    plan=plan,
                    manifest=manifest,
                    remote=remote,
                    result_validator=validator,
                )
            else:
                cached_reuse_count += 1
            observations.append(observation)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            refusals.append(
                {
                    "task_id": int(item["task_id"]),
                    "bundle_id": plan["bundle_id"],
                    "seed": int(item["seed"]),
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    deduplicated = deduplicate_observations(observations)
    cached, _ = cache_seed_records(runtime, deduplicated, apply=False)
    by_key = {
        (str(item["bundle_id"]), int(item["seed"])): item
        for item in deduplicated
    }
    aggregate_records = []
    for item in cached:
        source = by_key.get((str(item["bundle_id"]), int(item["seed"])))
        aggregate_records.append(
            {
                **item,
                "_result": (source or {}).get("_result"),
                "_candidates": (source or {}).get("_candidates"),
            }
        )
    public_inventory = [
        {
            key: copy.deepcopy(value)
            for key, value in item.items()
            if not str(key).startswith("_")
        }
        for item in inventory
    ]
    status, compatibility, index = build_current7_snapshot(
        plan=plan,
        manifest=manifest,
        inventory=public_inventory,
        records=aggregate_records,
        refusals=refusals,
        runtime=runtime,
    )
    if not status.get("updated_at"):
        # Keep an empty/pre-submit cohort immutable across heartbeat cycles.
        # ``observed_at`` belongs only to the mutable index pointer.
        status["updated_at"] = fallback_event_at
    index.update(
        {
            "updated_at": status["updated_at"],
            "status_event_at": status["updated_at"],
            "harvest_observed_at": observed_at,
            "final_goal_stage_id": stage_id,
            "final_goal_stage_profile_sha256": stage_profile(BY_ID[stage_id])["sha256"],
            "condition_display_only": True,
        }
    )
    index["status"]["sha256"] = _sha_bytes(_json_bytes(status))
    return {
        "stage_id": stage_id,
        "runtime": runtime,
        "status": status,
        "compatibility": compatibility,
        "index": index,
        "deduplicated": deduplicated,
        "cached_projection": cached,
        "inventory_count": len(inventory),
        "completed_count": sum(item["status"] == COMPLETED_STATE for item in inventory),
        "terminal_failure_count": len(terminal_failures),
        "cached_terminal_reuse_count": cached_reuse_count,
        "refusals": refusals,
    }


def _condition_inventory(
    *,
    runtime: Path,
    launch_plan: Mapping[str, Any],
    controller_state: Mapping[str, Any],
    projections: Sequence[Mapping[str, Any]],
    observed_at: str,
) -> dict[str, Any]:
    indexes = []
    for projection in projections:
        index = projection["index"]
        path = projection["runtime"] / "canonical" / "current7-index.json"
        indexes.append(
            {
                "stage_id": projection["stage_id"],
                "path": str(path.resolve()),
                "schema_version": CURRENT7_INDEX_SCHEMA,
                "projected_file_sha256": _sha_bytes(_json_bytes(index)),
                "hard_spec_sha256": index["hard_spec_sha256"],
                "stage_profile_sha256": index["final_goal_stage_profile_sha256"],
            }
        )
    unsigned = {
        "schema_version": INDEX_INVENTORY_SCHEMA,
        "launch_plan_sha256": launch_plan["launch_plan_sha256"],
        "controller_state_sha256": controller_state["state_sha256"],
        "controller_namespace": {
            "state_schema": CONTROLLER_STATE_SCHEMA,
            "task_name_prefix": TASK_NAME_PREFIX,
            "dedupe_prefix": DEDUPE_PREFIX,
        },
        "updated_at": observed_at,
        "indexes": indexes,
        "ui_environment": {
            "name": "MFT_TIER1_CURRENT7_CONDITION_INDEXES",
            "value": os.pathsep.join(item["path"] for item in indexes),
            "primary_authority_unchanged": True,
        },
        "scheduler_mutation_count": 0,
        "remote_write_count": 0,
        "aedt_used": False,
        "fea_submission_performed": False,
        "production_eligible": False,
        "automatic_promotion_allowed": False,
        "path_containment_root": str(runtime.resolve()),
    }
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def harvest_once(
    launch_plan_path: Path,
    bindings_path: Path,
    controller_state_path: Path,
    *,
    scheduler: SchedulerReader,
    remote: RemoteReader,
    runtime: Path = DEFAULT_RUNTIME,
    protected_current7_index: Path = DEFAULT_CURRENT7_INDEX,
    apply: bool = False,
    observed_at: str | None = None,
) -> dict[str, Any]:
    runtime = runtime.resolve()
    _assert_runtime_isolated(runtime, protected_current7_index)
    state_path = controller_state_path.resolve(strict=True)
    if not state_path.is_relative_to(runtime):
        raise RuntimeError("final1000 controller state is outside its runtime")
    plan, bindings, state = _validate_inputs(
        launch_plan_path, bindings_path, state_path
    )
    scheduler_get_before = int(getattr(scheduler, "get_count", 0))
    by_stage, pending_scheduler_cache, scheduler_cache_hits = _inventory_tasks(
        plan, state, scheduler=scheduler, runtime=runtime
    )
    heartbeat = observed_at or _observed_at()
    projections = []
    for stage in STAGES:
        stage_runtime = runtime / "conditions" / stage.stage_id
        projections.append(
            _project_stage(
                stage_id=stage.stage_id,
                binding=bindings[stage.stage_id],
                inventory=by_stage[stage.stage_id],
                remote=remote,
                runtime=stage_runtime,
                observed_at=heartbeat,
                fallback_event_at=str(plan["created_at"]),
            )
        )
    condition_inventory = _condition_inventory(
        runtime=runtime,
        launch_plan=plan,
        controller_state=state,
        projections=projections,
        observed_at=heartbeat,
    )
    local_writes = 0
    publications = []
    if apply:
        for path, payload in pending_scheduler_cache:
            local_writes += _atomic_replace(path, payload)
        for projection in projections:
            applied_cached, writes = cache_seed_records(
                projection["runtime"], projection["deduplicated"], apply=True
            )
            if applied_cached != projection["cached_projection"]:
                raise RuntimeError("final1000 cache projection changed before apply")
            local_writes += writes
            publication = publish_current7_snapshot(
                projection["runtime"],
                projection["status"],
                projection["compatibility"],
                projection["index"],
                apply=True,
            )
            local_writes += int(publication["write_count"])
            publications.append(publication)
        inventory_path = runtime / "canonical" / "condition-indexes.json"
        local_writes += _atomic_replace(
            inventory_path, _json_bytes(condition_inventory)
        )
    else:
        publications = [
            publish_current7_snapshot(
                projection["runtime"],
                projection["status"],
                projection["compatibility"],
                projection["index"],
                apply=False,
            )
            for projection in projections
        ]
    return {
        "schema_version": HARVEST_SCHEMA,
        "apply": bool(apply),
        "launch_plan_sha256": plan["launch_plan_sha256"],
        "controller_state_sha256": state["state_sha256"],
        "observed_at": heartbeat,
        "scheduler_get_count": int(getattr(scheduler, "get_count", 0))
        - scheduler_get_before,
        "scheduler_terminal_cache_hit_count": scheduler_cache_hits,
        "scheduler_mutation_count": 0,
        "remote_write_count": 0,
        "local_write_count": local_writes,
        "aedt_used": False,
        "fea_submission_performed": False,
        "condition_inventory": condition_inventory,
        "stages": [
            {
                "stage_id": projection["stage_id"],
                "inventory_count": projection["inventory_count"],
                "completed_count": projection["completed_count"],
                "terminal_failure_count": projection["terminal_failure_count"],
                "cached_terminal_reuse_count": projection[
                    "cached_terminal_reuse_count"
                ],
                "authenticated_seed_count": len(projection["cached_projection"]),
                "refused_terminal_count": len(projection["refusals"]),
                "publication": publication,
            }
            for projection, publication in zip(projections, publications, strict=True)
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-plan", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--controller-state", type=Path, required=True)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--protected-current7-index", type=Path, default=DEFAULT_CURRENT7_INDEX
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stop-file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.watch and not args.apply:
        raise RuntimeError("--watch requires --apply")
    if args.poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    scheduler = ReadOnlySchedulerApi(args.scheduler_url)
    remote = AccountSftpReader(args.accounts, args.scheduler_source)
    try:
        while True:
            if args.stop_file is not None and args.stop_file.is_file():
                break
            try:
                result = harvest_once(
                    args.launch_plan,
                    args.bindings,
                    args.controller_state,
                    scheduler=scheduler,
                    remote=remote,
                    runtime=args.runtime,
                    protected_current7_index=args.protected_current7_index,
                    apply=args.apply,
                )
                print(
                    json.dumps(result, ensure_ascii=False, sort_keys=True),
                    flush=True,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if not args.watch:
                    raise
                print(
                    json.dumps(
                        {
                            "schema_version": HARVEST_SCHEMA,
                            "apply": True,
                            "error": f"{type(exc).__name__}:{exc}",
                            "scheduler_mutation_count": 0,
                            "remote_write_count": 0,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if not args.watch:
                break
            time.sleep(args.poll_seconds)
    finally:
        remote.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
