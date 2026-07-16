"""
상시 포화 피더: scheduler의 durable simulation-policy desired 수를 유지한다.

Pooled 장기 루프는 매 주기 versioned policy를 읽고, 완료되는 만큼만 새 태스크를
채운다. ``--target``은 일회성 실행 또는 policy가 없는 구형 scheduler의 명시적
호환 fallback일 뿐 장기 운전의 source of truth가 아니다.
- 태스크: 작업당 샘플 1개
- 이름: mft-camp-s<solver>-l<library>-<일련번호> (serial은 feeder_state.json에 영속)
- 총량 상한: --max-samples 도달 시 중단 (기본 12000)

사용: python feeder.py --once --target 1  # 1회 보충 (수동)
      python feeder.py --loop 600 --aedt-pooled ...  # durable policy 운전
"""
import argparse
import copy
import glob
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass

import requests
import pyarrow.parquet as pq
from filelock import FileLock, Timeout as FileLockTimeout

from pinned_pilot import (
    LEGACY_MFT_NAME_PREFIX,
    MFT_PROJECT,
    MFT_PROJECT_MAX_ACTIVE_TASKS,
    PILOT_RESERVED_VALID_CANDIDATES,
    al_driver,
    campaign_mutation_lock,
    cursor_after_valid_candidates,
    next_valid_candidate,
    project_submission_snapshot,
    queue_allows_demand_submission,
    validate_p08_completion,
)

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY_DIR = os.path.abspath(os.path.join(HERE, "..", "verify"))
if VERIFY_DIR not in sys.path:
    sys.path.insert(0, VERIFY_DIR)
import scheduler_client
import deployment_gate

LOGGER = logging.getLogger(__name__)

_STATE_DIR = os.environ.get("MFT_FEEDER_STATE_DIR")
if _STATE_DIR:
    os.makedirs(_STATE_DIR, exist_ok=True)
else:
    _STATE_DIR = HERE
STATE = os.path.join(_STATE_DIR, "feeder_state.json")
CONTROLLER_LOCK = os.path.join(_STATE_DIR, "feeder-controller.lock")
DEFAULT_SCHEDULER = "http://127.0.0.1:8000"
LOCAL_SCHEDULER_FALLBACK = "http://127.0.0.1:8001"


def _configured_scheduler_url():
    """Resolve the scheduler endpoint once for this feeder process."""

    return (
        os.environ.get("MFT_SCHEDULER_URL", DEFAULT_SCHEDULER).strip().rstrip("/")
        or DEFAULT_SCHEDULER
    )


SCHEDULER = _configured_scheduler_url()
# Submission/reconciliation helpers live in scheduler_client.  Keep their
# endpoint aligned with the policy/inventory reads performed in this module.
scheduler_client.SCHEDULER = SCHEDULER
CAMPAIGN_PREFIX = "mft-camp-"

TARGET_ACTIVE = 50    # standalone 실행+대기 목표 (--target으로 오버라이드)
BUFFER = 0            # production 300 promotion is owned by rapid_campaign
MAX_STANDALONE_ACTIVE = 50
MAX_POOLED_ACTIVE = 500
MAX_POOLED_PROJECT_ACTIVE_TASKS = (
    scheduler_client.MFT_PROJECT_MAX_ACTIVE_TASKS_CEILING)
COUNT_PER_TASK = 1
CPUS_PER_TASK = 4
DEFAULT_AEDT_POOL_PKG_ROOT = "$HOME/slurm_scheduler/aedt_pool_pkg"
DEFAULT_AEDT_POOL_CLIENT_TOKEN_FILE = "$HOME/slurm_scheduler/aedt_pool_client"
DEFAULT_AEDT_SESSION_VERSION = "2025.2"
DEFAULT_AEDT_ISOLATION_POLICY = "family"
AEDT_ISOLATION_POLICIES = ("family", "shared_if_compatible")
AEDT_SESSION_PROFILE = json.dumps(
    {
        "profile_version": 2,
        "aedt_version": "2025.2",
        "python_environment": "pyaedt2026v1",
        "pyaedt_version": "0.22.0",
        "filesystem": "gpfs-shared-v1",
        "desktop_dso": {
            "config_name": "pyaedt_config",
            "designs": {
                "Icepak": {
                    "cores": 4,
                    "tasks": 1,
                    "gpus": 0,
                    "use_auto_settings": False,
                },
                "Maxwell 2D": {
                    "cores": 4,
                    "tasks": 1,
                    "gpus": 0,
                    "use_auto_settings": True,
                },
                "Maxwell 3D": {
                    "cores": 4,
                    "tasks": 1,
                    "gpus": 0,
                    "use_auto_settings": True,
                },
            },
        },
    },
    sort_keys=True,
    separators=(",", ":"),
)
DEFAULT_POOLED_CPUS = 1
DEFAULT_POOLED_MEMORY_MB = 6144
CPU_HEADROOM = 0.85
SCHEDULER_ATTEMPTS = 3
ACTIVE_TASK_STATUSES = ("queued", "attaching", "running")
CAMPAIGN_INVENTORY_PAGE_SIZE = 2000
PROFILE_PATH = os.path.join(HERE, "..", "verify", "profiles", "standard.json")
TRAIN_PARQUET = os.path.join(HERE, "..", "data", "dataset", "train.parquet")
COLLECT_CACHE = os.path.join(HERE, "..", "data", "dataset", "collect_cache.json")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _require_deployed_revisions(solver_revision, library_revision):
    library_root = os.environ.get("MFT_PYAEDT_LIBRARY_ROOT", "").strip()
    if not library_root:
        library_root = os.path.abspath(
            os.path.join(REPO_ROOT, "..", "pyaedt_library")
        )
    return deployment_gate.validate_deployment(
        REPO_ROOT, solver_revision, library_root, library_revision
    )


class SchedulerError(RuntimeError):
    pass


class SimulationPolicyUnavailable(SchedulerError):
    """An older scheduler does not advertise durable simulation policy."""


def _pooled_submission_kwargs(args):
    if not args.aedt_pooled:
        return None
    if not isinstance(args.aedt_pool_url, str) or not args.aedt_pool_url.strip():
        raise SchedulerError("--aedt-pool-url is required with --aedt-pooled")
    for value, flag in (
            (args.pooled_cpus, "--pooled-cpus"),
            (args.pooled_memory_mb, "--pooled-memory-mb")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SchedulerError(f"{flag} must be a positive integer")
    for value, flag in (
            (args.aedt_pool_pkg_root, "--aedt-pool-pkg-root"),
            (
                args.aedt_pool_client_token_file,
                "--aedt-pool-client-token-file",
            ),
            (args.aedt_session_version, "--aedt-session-version")):
        if not isinstance(value, str) or not value.strip():
            raise SchedulerError(f"{flag} must be non-empty")
    if args.aedt_isolation_policy not in AEDT_ISOLATION_POLICIES:
        raise SchedulerError(
            "--aedt-isolation-policy must be family or shared_if_compatible")
    return {
        "cpus": args.pooled_cpus,
        "memory_mb": args.pooled_memory_mb,
        "aedt_backend": "pooled",
        "submission_env": {
            "MFT_AEDT_BACKEND": "pooled",
            "MFT_AEDT_SHARED_CANARY": "1",
            "MFT_AEDT_SCHEDULER_URL": args.aedt_pool_url,
            "MFT_SLURM_SCHEDULER_ROOT": args.aedt_pool_pkg_root,
            "SLURM_AEDT_POOL_CLIENT_TOKEN_FILE": (
                args.aedt_pool_client_token_file
            ),
            "MFT_AEDT_POOL_WORKSPACE": (
                "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}"
            ),
            "MFT_AEDT_WORKSPACE_PATH": (
                "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}"
            ),
            "MFT_AEDT_SESSION_VERSION": args.aedt_session_version,
            "MFT_AEDT_SESSION_PROFILE": AEDT_SESSION_PROFILE,
            "MFT_AEDT_ISOLATION_POLICY": args.aedt_isolation_policy,
            "AEDT_POOL_AUTOMATION_LOCK_TIMEOUT_SECONDS": "7200",
            # Three serialized first-model builds can legitimately exceed the
            # old 120-second underfilled seal. Keep the model-ready barrier open
            # for the scheduler client's supported maximum.
            "MFT_AEDT_POOL_FILL_TIMEOUT_SECONDS": "900",
        },
    }


def _is_pooled_submission(submission):
    return (
        isinstance(submission, dict)
        and submission.get("aedt_backend") == "pooled"
    )


_RAPID_REFILL_SEAL = object()
_ADOPTED_REFILL_SEAL = object()


@dataclass(frozen=True)
class _RapidRefillAuthorization:
    target: int
    max_samples: int
    solver_revision: str
    library_revision: str
    candidate_seed: int
    seal: object


@dataclass(frozen=True)
class _AdoptedRefillAuthorization:
    """One-cycle authorization for an externally preloaded production fleet.

    The adopted controller must authenticate its original cohort on every
    cycle and prove the same local3 plus fleet20/90% evidence used by the
    normal rapid controller.  Keeping this as a distinct sealed type avoids
    weakening or pretending to satisfy the p02/p08 pilot contract.
    """

    target: int
    max_samples: int
    solver_revision: str
    library_revision: str
    candidate_seed: int
    adoption_sha256: str
    initial_count: int
    cpus: int
    memory_mb: int
    timeout_seconds: int
    evidence_mode: str
    strict_rows: int
    target_strict_rows: int
    seal: object


def _authorize_rapid_refill(
        decision, *, max_samples, solver_revision, library_revision,
        candidate_seed, local_passed, pilots_complete):
    """Seal one refill decision only after the rapid promotion evidence passes."""
    if not scheduler_client.campaign_mutation_lock_is_held():
        raise SchedulerError("rapid refill authorization requires the mutation lock")
    if not isinstance(decision, dict) or decision.get("paused"):
        raise SchedulerError("rapid refill decision is absent or paused")
    target = decision.get("target_active")
    if isinstance(target, bool) or target not in (50, 300):
        raise SchedulerError("rapid refill target must be exactly 50 or 300")
    expected_action = f"refill_{target}"
    if decision.get("action") != expected_action:
        raise SchedulerError("rapid refill action does not match its target")
    pilot = decision.get("pilot") or {}
    if (not local_passed or not pilots_complete
            or int(pilot.get("valid") or 0) < 5
            or int(pilot.get("invalid") or 0) != 0):
        raise SchedulerError("rapid refill lacks local3/pilot promotion evidence")
    if target == 300:
        production = decision.get("production") or {}
        terminal = int(production.get("terminal") or 0)
        valid_rate = production.get("valid_rate")
        if (terminal < 20 or isinstance(valid_rate, bool)
                or not isinstance(valid_rate, (int, float))
                or not math.isfinite(float(valid_rate))
                or float(valid_rate) < 0.90):
            raise SchedulerError("rapid refill lacks fleet20/90% promotion evidence")
    return _RapidRefillAuthorization(
        target=int(target),
        max_samples=int(max_samples),
        solver_revision=str(solver_revision),
        library_revision=str(library_revision),
        candidate_seed=int(candidate_seed),
        seal=_RAPID_REFILL_SEAL,
    )


def _authorize_adopted_refill(
        decision, *, max_samples, solver_revision, library_revision,
        candidate_seed, local_passed, adoption_sha256, initial_count,
        cpus, memory_mb, timeout_seconds, evidence_mode, strict_rows,
        target_strict_rows):
    """Seal one refill for an authenticated adopted or concurrent fleet.

    This is deliberately separate from :func:`_authorize_rapid_refill`:
    adopted production evidence is not represented as fictitious p02/p08
    pilot evidence.  ``preloaded250_v1`` preserves the reviewed historical
    250 -> 300 adoption path.  ``concurrent250_v1`` preserves the former
    controller contract, while ``concurrent300_v1`` is the current continuous
    mode: it may only maintain exactly 300 logical MFT active tasks and starts
    from a sealed, empty controller ledger while other MFT revisions already
    occupy part of that project-level pool.
    """
    if not scheduler_client.campaign_mutation_lock_is_held():
        raise SchedulerError("adopted refill authorization requires the mutation lock")
    if not isinstance(decision, dict) or decision.get("paused"):
        raise SchedulerError("adopted refill decision is absent or paused")
    target = decision.get("target_active")
    dynamic_contract = (
        type(target) is int
        and 1 <= target <= MFT_PROJECT_MAX_ACTIVE_TASKS
        and initial_count == 0
        and evidence_mode == "dynamic_project_cap_v1"
    )
    legacy_target = type(target) is int and target in (250, 300, 400)
    if ((not dynamic_contract and not legacy_target)
            or decision.get("action") != f"refill_{target}"):
        raise SchedulerError(
            "adopted refill decision must authorize its exact bounded target")
    if local_passed is not True:
        raise SchedulerError("adopted refill lacks local3 evidence")
    for revision, label in (
            (solver_revision, "solver"), (library_revision, "library")):
        revision = str(revision or "")
        if (len(revision) != 40 or revision != revision.lower()
                or any(char not in "0123456789abcdef" for char in revision)):
            raise SchedulerError(f"adopted refill {label} revision is invalid")
    adopted_contract = (initial_count, evidence_mode)
    if adopted_contract not in (
            (250, "preloaded250_v1"),
            (0, "concurrent250_v1"),
            (0, "concurrent300_v1"),
            (0, "concurrent400_v1"),
            (0, "dynamic_project_cap_v1")):
        raise SchedulerError(
            "adopted refill requires exact preloaded-250 or concurrent-0 "
            "cohort/evidence contract")
    if (not dynamic_contract and (
            (target == 300 and adopted_contract not in {
                (250, "preloaded250_v1"), (0, "concurrent300_v1")})
            or (target == 250 and adopted_contract != (0, "concurrent250_v1"))
            or (target == 400 and adopted_contract != (0, "concurrent400_v1")))):
        raise SchedulerError("adopted refill target/evidence contract is invalid")
    if (type(max_samples) is not int
            or max_samples != 12_000 or type(candidate_seed) is not int
            or candidate_seed != 260710):
        raise SchedulerError("adopted refill campaign contract is invalid")
    if (type(strict_rows) is not int or type(target_strict_rows) is not int
            or target_strict_rows != 3_000
            or strict_rows < 0 or strict_rows >= target_strict_rows):
        raise SchedulerError("adopted refill strict-row evidence is invalid")
    adoption_sha256 = str(adoption_sha256 or "").lower()
    if (len(adoption_sha256) != 64
            or any(char not in "0123456789abcdef" for char in adoption_sha256)):
        raise SchedulerError("adopted refill cohort seal is invalid")
    resources = (cpus, memory_mb, timeout_seconds)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in resources):
        raise SchedulerError("adopted refill resources must be integers")
    if resources != (4, 65_536, 14_400):
        raise SchedulerError("adopted refill resources must be 4 CPU/64 GiB/4 hours")
    production = decision.get("production") or {}
    terminal = production.get("terminal")
    valid = production.get("valid")
    valid_rate = production.get("valid_rate")
    if (isinstance(terminal, bool) or not isinstance(terminal, int)
            or isinstance(valid, bool) or not isinstance(valid, int)
            or valid < 0 or valid > terminal
            or (terminal == 0 and valid_rate is not None)
            or (terminal > 0 and (
                isinstance(valid_rate, bool)
                or not isinstance(valid_rate, (int, float))
                or not math.isfinite(float(valid_rate))
                or abs(float(valid_rate) - valid / terminal) > 1e-12))):
        raise SchedulerError("adopted refill production evidence is inconsistent")
    if ((target in (300, 400) or dynamic_contract)
            and (terminal < 20 or float(valid_rate) < 0.90)):
        raise SchedulerError("adopted refill lacks fleet20/90% promotion evidence")
    return _AdoptedRefillAuthorization(
        target=int(target),
        max_samples=int(max_samples),
        solver_revision=str(solver_revision),
        library_revision=str(library_revision),
        candidate_seed=int(candidate_seed),
        adoption_sha256=adoption_sha256,
        initial_count=int(initial_count),
        cpus=int(cpus),
        memory_mb=int(memory_mb),
        timeout_seconds=int(timeout_seconds),
        evidence_mode=evidence_mode,
        strict_rows=strict_rows,
        target_strict_rows=target_strict_rows,
        seal=_ADOPTED_REFILL_SEAL,
    )


def _step_from_rapid_controller(
        max_samples, *, authorization, target, buffer=0,
        solver_revision=None, library_revision=None, candidate_seed=260710):
    """Execute a production refill using one evidence-bound authorization."""
    if not scheduler_client.campaign_mutation_lock_is_held():
        raise SchedulerError("rapid production refill requires the mutation lock")
    expected = (
        int(target), int(max_samples), str(solver_revision),
        str(library_revision), int(candidate_seed),
    )
    actual = (
        getattr(authorization, "target", None),
        getattr(authorization, "max_samples", None),
        getattr(authorization, "solver_revision", None),
        getattr(authorization, "library_revision", None),
        getattr(authorization, "candidate_seed", None),
    )
    if (not isinstance(authorization, _RapidRefillAuthorization)
            or authorization.seal is not _RAPID_REFILL_SEAL
            or actual != expected or int(buffer) != 0):
        raise SchedulerError("rapid production refill authorization is invalid")
    return _step_locked(
        max_samples, target=target, buffer=buffer,
        solver_revision=solver_revision, library_revision=library_revision,
        candidate_seed=candidate_seed, _rapid_authorization=authorization,
    )


def _step_from_adopted_controller(
        max_samples, *, authorization, target, buffer=0,
        solver_revision=None, library_revision=None, candidate_seed=260710,
        adoption_sha256=None, initial_count=None, cpus=None, memory_mb=None,
        timeout_seconds=None, evidence_mode=None, strict_rows=None,
        target_strict_rows=None, journal=None):
    """Execute one evidence-bound refill for an authenticated adopted fleet."""
    if not scheduler_client.campaign_mutation_lock_is_held():
        raise SchedulerError("adopted production refill requires the mutation lock")
    expected = (
        int(target), int(max_samples), str(solver_revision),
        str(library_revision), int(candidate_seed),
        str(adoption_sha256 or "").lower(), int(initial_count),
        int(cpus), int(memory_mb), int(timeout_seconds),
        evidence_mode, int(strict_rows), int(target_strict_rows),
    )
    actual = (
        getattr(authorization, "target", None),
        getattr(authorization, "max_samples", None),
        getattr(authorization, "solver_revision", None),
        getattr(authorization, "library_revision", None),
        getattr(authorization, "candidate_seed", None),
        getattr(authorization, "adoption_sha256", None),
        getattr(authorization, "initial_count", None),
        getattr(authorization, "cpus", None),
        getattr(authorization, "memory_mb", None),
        getattr(authorization, "timeout_seconds", None),
        getattr(authorization, "evidence_mode", None),
        getattr(authorization, "strict_rows", None),
        getattr(authorization, "target_strict_rows", None),
    )
    if (not isinstance(authorization, _AdoptedRefillAuthorization)
            or authorization.seal is not _ADOPTED_REFILL_SEAL
            or actual != expected or int(buffer) != 0):
        raise SchedulerError("adopted production refill authorization is invalid")
    return _step_locked(
        max_samples, target=target, buffer=buffer,
        solver_revision=solver_revision, library_revision=library_revision,
        candidate_seed=candidate_seed, _adopted_authorization=authorization,
        _submit_resources={
            "cpus": int(cpus),
            "memory_mb": int(memory_mb),
            "timeout_seconds": int(timeout_seconds),
        },
        _refill_journal=journal,
    )


def submit(
        name, workdir, params, solver_revision, library_revision, *,
        cpus=CPUS_PER_TASK, memory_mb=32768, timeout_seconds=None,
        required_project_cap=None, aedt_backend=None, submission_env=None,
        required_hard_cap=None, max_project_active_tasks=None):
    with open(PROFILE_PATH, encoding="utf-8") as stream:
        profile = json.load(stream)
    if timeout_seconds is not None:
        profile["timeout_seconds"] = int(timeout_seconds)
    submission_options = {}
    if aedt_backend is not None:
        submission_options["aedt_backend"] = aedt_backend
    if submission_env is not None:
        submission_options["submission_env"] = submission_env
    if required_hard_cap is not None:
        submission_options["required_hard_cap"] = required_hard_cap
    if max_project_active_tasks is not None:
        submission_options["max_project_active_tasks"] = (
            max_project_active_tasks)
    return scheduler_client.submit_verification(
        name=name,
        workdir=workdir,
        params=params,
        profile=profile,
        mem_mb=int(memory_mb),
        cpus=int(cpus),
        solver_revision=solver_revision,
        library_revision=library_revision,
        required_project_cap=required_project_cap,
        **submission_options,
    )


def load_state():
    if os.path.isfile(STATE):
        try:
            with open(STATE, encoding="utf-8") as stream:
                return json.load(stream)
        except json.JSONDecodeError as exc:
            LOGGER.warning(
                "state file %s is empty or corrupt (%s); starting fresh",
                STATE,
                exc,
            )
    return {"serial": 0, "submitted_samples": 0}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(st, stream)

    last_error = None
    for attempt in range(3):
        try:
            os.replace(tmp, STATE)
            return
        except OSError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5)

    LOGGER.warning(
        "atomic state update failed after 3 attempts (%s); "
        "writing %s directly",
        last_error,
        STATE,
    )
    with open(STATE, "w", encoding="utf-8") as stream:
        json.dump(st, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass
    except OSError as exc:
        LOGGER.warning("could not remove temporary state file %s: %s", tmp, exc)


def dataset_collection_snapshot():
    """Read master row count and collector judgements in one lock epoch."""
    with FileLock(TRAIN_PARQUET + ".lock", timeout=30):
        try:
            # Direct open is authoritative on mounted drives; directory
            # metadata may transiently hide an otherwise readable parquet.
            rows = int(pq.ParquetFile(TRAIN_PARQUET).metadata.num_rows)
        except FileNotFoundError:
            rows = 0
        except (OSError, ValueError, TypeError) as exc:
            raise SchedulerError(f"collector dataset is unreadable: {exc}") from exc
        canonical_missing = False
        directory = os.path.dirname(COLLECT_CACHE) or "."
        basename = os.path.basename(COLLECT_CACHE)
        # RaiDrive directory metadata can briefly report a false negative even
        # when the canonical path is directly readable.
        candidates = [COLLECT_CACHE]
        recovery = set(glob.glob(COLLECT_CACHE + ".tmp*"))
        recovery.update(glob.glob(os.path.join(
            directory, f".{basename}.*.tmp")))
        recovery.discard(COLLECT_CACHE)
        def safe_mtime(path):
            try:
                return os.path.getmtime(path)
            except OSError:
                return -1.0

        recovery = sorted(
            recovery, key=lambda path: (safe_mtime(path), path), reverse=True)
        candidates.extend(recovery)
        errors = []
        for path in candidates:
            try:
                with open(path, encoding="utf-8") as stream:
                    cache = json.load(stream)
                if not isinstance(cache, dict):
                    raise ValueError("cache root must be an object")
                judged = set()
                for key in ("harvested", "nodata"):
                    task_ids = cache.get(key, [])
                    if not isinstance(task_ids, list):
                        raise ValueError(f"cache {key} must be a list")
                    for task_id in task_ids:
                        if (isinstance(task_id, bool) or not isinstance(task_id, int)
                                or task_id <= 0):
                            raise ValueError(f"cache {key} has an invalid task ID")
                        judged.add(task_id)
                return rows, judged
            except FileNotFoundError as exc:
                if path == COLLECT_CACHE:
                    canonical_missing = True
                else:
                    errors.append(f"{os.path.basename(path)}: {exc}")
            except (OSError, UnicodeError, ValueError, TypeError) as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        if rows == 0 and canonical_missing and not recovery:
            return rows, set()
        detail = "; ".join(errors) if errors else "canonical cache is missing"
        raise SchedulerError(f"collector cache is unavailable: {detail}")


def dataset_row_count():
    return dataset_collection_snapshot()[0]


def campaign_inventory():
    inventory = []
    seen_ids = set()
    before_id = 0
    while True:
        params = {
            "compact": True,
            "limit": CAMPAIGN_INVENTORY_PAGE_SIZE,
            "name_prefix": CAMPAIGN_PREFIX,
        }
        if before_id:
            params["before_id"] = before_id
        payload = _scheduler_json("/api/tasks", params=params)
        tasks = payload if isinstance(payload, list) else (
            payload.get("tasks") if isinstance(payload, dict) else None)
        if not isinstance(tasks, list) or len(tasks) > CAMPAIGN_INVENTORY_PAGE_SIZE:
            raise SchedulerError("scheduler returned an invalid campaign task inventory")
        page_ids = []
        for task in tasks:
            if not isinstance(task, dict):
                raise SchedulerError("scheduler returned an invalid campaign task")
            if not str(task.get("name") or "").startswith(CAMPAIGN_PREFIX):
                raise SchedulerError(
                    "scheduler campaign prefix filter returned an unrelated task")
            project = str(task.get("project") or "").strip()
            if project not in ("", MFT_PROJECT):
                raise SchedulerError(
                    f"campaign task belongs to unexpected project {project!r}")
            task_id = task.get("id", task.get("task_id"))
            if (isinstance(task_id, bool) or not isinstance(task_id, int)
                    or task_id <= 0 or task_id in seen_ids
                    or (before_id and task_id >= before_id)):
                raise SchedulerError(
                    "scheduler returned an invalid/duplicate campaign task ID")
            seen_ids.add(task_id)
            page_ids.append(task_id)
            inventory.append(task)
        if len(tasks) < CAMPAIGN_INVENTORY_PAGE_SIZE:
            return inventory
        if not page_ids:
            raise SchedulerError("scheduler campaign inventory cursor did not advance")
        next_before_id = min(page_ids)
        if before_id and next_before_id >= before_id:
            raise SchedulerError("scheduler campaign inventory cursor did not advance")
        before_id = next_before_id


def expected_task_rows(task):
    name = str((task or {}).get("name") or "")
    return 1 if name.startswith("mft-camp-s") else 8


def reserved_unjudged_rows(state, tasks, judged_ids):
    """Reserve outputs until the collector durably classifies every task ID."""
    by_id = {
        int(task["id"]): task for task in tasks
        if task.get("id") is not None
    }
    ledger_ids = {
        int(task_id) for task_id in state.get("outstanding", [])
        if task_id is not None
    }
    active_ids = {
        task_id for task_id, task in by_id.items()
        if task.get("status") in ACTIVE_TASK_STATUSES
    }
    inventory_unjudged = {
        task_id for task_id in by_id if task_id not in judged_ids
    }
    reserved_ids = (ledger_ids | active_ids | inventory_unjudged) - judged_ids
    recorded_rows = state.get("task_expected_rows") or {}
    return sum(
        int(recorded_rows.get(str(task_id), expected_task_rows(by_id.get(task_id))))
        for task_id in reserved_ids
    )


def _scheduler_json(path, params=None):
    global SCHEDULER
    last_error = None
    for attempt in range(SCHEDULER_ATTEMPTS):
        response = None
        request_bases = [SCHEDULER]
        # Fail over only from the legacy loopback listener.  An explicitly
        # configured remote scheduler must fail closed instead of silently
        # sending project traffic to a different service.
        if SCHEDULER == DEFAULT_SCHEDULER:
            request_bases.append(LOCAL_SCHEDULER_FALLBACK)
        for scheduler_base in request_bases:
            try:
                response = requests.get(
                    f"{scheduler_base}{path}", params=params, timeout=30
                )
            except requests.RequestException as exc:
                last_error = exc
                continue
            if scheduler_base != SCHEDULER:
                SCHEDULER = scheduler_base
                scheduler_client.SCHEDULER = scheduler_base
            break
        try:
            if response is None:
                raise last_error or SchedulerError(
                    "scheduler connection failed without an error"
                )
            status_code = int(response.status_code)
            if status_code >= 400:
                raise SchedulerError(f"HTTP {status_code} from {path}")
            return response.json()
        except (requests.RequestException, ValueError, SchedulerError) as exc:
            last_error = exc
            if attempt + 1 < SCHEDULER_ATTEMPTS:
                time.sleep(0.5 * (2 ** attempt))
    raise SchedulerError(f"scheduler request failed for {path}: {last_error}")


def simulation_policy_snapshot():
    """Read and validate the scheduler-owned MFT desired concurrency.

    The project safety cap is intentionally not accepted as demand.  A pooled
    controller may refill only from a versioned policy whose desired value is
    within both the scheduler rollout gate and this solver release's hard cap.
    """
    project = _scheduler_json(f"/api/projects/{MFT_PROJECT}")
    if not isinstance(project, dict):
        raise SchedulerError("scheduler returned an invalid MFT project policy")
    if str(project.get("name") or project.get("project") or "").strip() != MFT_PROJECT:
        raise SchedulerError("scheduler returned a different project policy")
    embedded = project.get("simulation_policy")
    if isinstance(embedded, dict):
        policy = {**project, **embedded}
    elif (
            "desired_simulations" in project
            or "policy_revision" in project
            or "validated_concurrency_limit" in project):
        policy = project
    else:
        raise SimulationPolicyUnavailable(
            "scheduler project has no durable simulation-policy capability")

    desired = policy.get("desired_simulations")
    validated = policy.get("validated_concurrency_limit")
    revision = policy.get("policy_revision")
    scale_down_mode = str(policy.get("scale_down_mode") or "").strip().lower()
    if (type(desired) is not int
            or not 0 <= desired <= MAX_POOLED_ACTIVE):
        raise SchedulerError(
            f"scheduler desired_simulations must be between 0 and "
            f"{MAX_POOLED_ACTIVE}")
    if (type(validated) is not int
            or not 0 <= validated <= MAX_POOLED_PROJECT_ACTIVE_TASKS):
        raise SchedulerError(
            "scheduler validated_concurrency_limit is invalid")
    if desired > validated:
        raise SchedulerError(
            "scheduler desired_simulations exceeds the validated concurrency limit")
    if (isinstance(revision, bool)
            or not isinstance(revision, (int, str))
            or not str(revision).strip()
            or (isinstance(revision, int) and revision < 0)):
        raise SchedulerError("scheduler simulation-policy revision is invalid")
    if scale_down_mode != "drain":
        raise SchedulerError("scheduler simulation-policy must use drain scale-down")
    return {
        "desired_simulations": desired,
        "effective_simulations": policy.get("effective_simulations"),
        "validated_concurrency_limit": validated,
        "policy_revision": revision,
        "scale_down_mode": scale_down_mode,
        "resource_constraint": policy.get("resource_constraint"),
    }


def _cycle_target(args):
    """Resolve one cycle's target, preferring durable policy for pooled loops."""
    policy_driven = bool(args.aedt_pooled and (args.loop or args.target is None))
    if policy_driven:
        try:
            policy = simulation_policy_snapshot()
            if args.buffer:
                raise SchedulerError(
                    "--buffer must be zero when simulation-policy drives the feeder")
            return policy["desired_simulations"], policy
        except SimulationPolicyUnavailable:
            # A supplied target is an explicit compatibility fallback for an
            # older scheduler.  New deployments omit it and therefore fail
            # closed until durable policy is available.
            if args.target is None:
                raise
            LOGGER.warning(
                "durable simulation-policy unavailable; using explicit "
                "compatibility target %s for this cycle",
                args.target,
            )
    return (TARGET_ACTIVE if args.target is None else args.target), None


def _validate_cycle_target(args, target):
    if type(target) is not int or type(args.buffer) is not int:
        raise SchedulerError("target and buffer must be integers")
    requested_active = target + args.buffer
    if requested_active < 0:
        raise SchedulerError("target plus buffer must be non-negative")
    if args.aedt_pooled and requested_active > MAX_POOLED_ACTIVE:
        raise SchedulerError(f"pooled feeder hard cap is {MAX_POOLED_ACTIVE}")
    if not args.aedt_pooled and requested_active > MAX_STANDALONE_ACTIVE:
        raise SchedulerError(
            f"standalone feeder hard cap is {MAX_STANDALONE_ACTIVE}; "
            "use rapid_campaign.py for 300-task production promotion")
    return requested_active


def scheduler_snapshot(
        required_hard_cap, *, require_exact_project_cap=False,
        require_full_project=False,
        max_project_active_tasks=MFT_PROJECT_MAX_ACTIVE_TASKS):
    global_summary = _scheduler_json("/api/tasks/summary")
    allocations = _scheduler_json("/api/allocations")
    projects = _scheduler_json("/api/projects")
    project_tasks = _scheduler_json("/api/tasks", params={
        "limit": 10000,
        "project": MFT_PROJECT,
        "status": ",".join(ACTIVE_TASK_STATUSES),
    })
    legacy_tasks = _scheduler_json("/api/tasks", params={
        "limit": 10000,
        "name_prefix": LEGACY_MFT_NAME_PREFIX,
        "status": ",".join(ACTIVE_TASK_STATUSES),
    })
    capacity = _scheduler_json("/api/task-capacity", params={
        "cpus": CPUS_PER_TASK,
        "memory_mb": 32768,
        "scheduling_profile": "fea_bursty",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        # License admission is project-scoped and fail-closed.  Omitting this
        # field makes a known MFT request look like an unknown FEA project and
        # falsely blocks refill before priority/admission ordering is reached.
        "project": MFT_PROJECT,
    })
    global_statuses = (
        global_summary.get("statuses") if isinstance(global_summary, dict) else None)
    if (not isinstance(global_statuses, dict)
            or not isinstance(allocations, list)
            or not isinstance(capacity, dict)):
        raise SchedulerError("scheduler returned an invalid snapshot")
    global_counts = {
        status: int(global_statuses.get(status, 0) or 0)
        for status in ACTIVE_TASK_STATUSES}
    try:
        queue_submission_allowed = queue_allows_demand_submission(
            capacity.get("queue_state"))
        project_options = {}
        if max_project_active_tasks != MFT_PROJECT_MAX_ACTIVE_TASKS:
            project_options["max_project_active_tasks"] = (
                max_project_active_tasks)
        project_gate = scheduler_client.project_submission_snapshot(
            projects, project_tasks, required_hard_cap,
            legacy_tasks=legacy_tasks,
            require_exact_project_cap=require_exact_project_cap,
            require_full_project=require_full_project,
            **project_options)
    except RuntimeError as exc:
        raise SchedulerError(str(exc)) from exc
    capacity_gate = {
        "ready_fit_slots": int(capacity.get("ready_fit_slots") or 0),
        "queue_state": str(capacity.get("queue_state") or "").strip().lower(),
        "queue_reason": str(capacity.get("queue_reason") or "").strip(),
        "queue_submission_allowed": queue_submission_allowed,
    }
    capacity_gate.update(project_gate)
    capacity_gate["submission_allowed"] = bool(
        queue_submission_allowed and project_gate["project_submission_slots"] > 0)
    return project_gate["project_counts"], global_counts, allocations, capacity_gate


def cpu_submission_headroom(status_counts, allocations, ready_fit_slots):
    """Report immediate owned-pool room; queued demand is not capped by it."""
    usable_allocations = [
        a for a in allocations
        if a.get("state") in ("active", "warm")
        and a.get("resource_pool", "cpu") == "cpu"]
    total_cpus = sum(max(0, int(a.get("total_cpus") or 0)) for a in usable_allocations)
    free_cpus = sum(max(0, int(a.get("free_cpus") or 0)) for a in usable_allocations)
    total_slots = math.floor(total_cpus / CPUS_PER_TASK * CPU_HEADROOM)
    free_slots = math.floor(free_cpus / CPUS_PER_TASK * CPU_HEADROOM)
    global_active = sum(status_counts.values())
    headroom = max(0, min(
        free_slots,
        total_slots - global_active,
        max(0, int(ready_fit_slots or 0)),
    ))
    return headroom, total_cpus, free_cpus, global_active


def step(max_samples, target=TARGET_ACTIVE, buffer=BUFFER,
         solver_revision=None, library_revision=None, candidate_seed=260710,
         pooled_submission=None):
    requested_active = int(target) + int(buffer)
    pooled_mode = _is_pooled_submission(pooled_submission)
    if pooled_mode and requested_active > MAX_POOLED_ACTIVE:
        raise SchedulerError(
            f"pooled feeder hard cap is {MAX_POOLED_ACTIVE}")
    if not pooled_mode and requested_active > MAX_STANDALONE_ACTIVE:
        raise SchedulerError(
            f"direct feeder hard cap is {MAX_STANDALONE_ACTIVE}; "
            "only rapid_campaign may authorize production promotion")
    pooled_options = {}
    if pooled_submission is not None:
        pooled_options["_pooled_submission"] = pooled_submission
    if (requested_active > 0
            and not scheduler_client.campaign_mutation_lock_is_held()):
        with campaign_mutation_lock():
            return _step_locked(
                max_samples, target=target, buffer=buffer,
                solver_revision=solver_revision,
                library_revision=library_revision,
                candidate_seed=candidate_seed,
                **pooled_options,
            )
    return _step_locked(
        max_samples, target=target, buffer=buffer,
        solver_revision=solver_revision,
        library_revision=library_revision,
        candidate_seed=candidate_seed,
        **pooled_options,
    )


def _step_locked(max_samples, target=TARGET_ACTIVE, buffer=BUFFER,
                 solver_revision=None, library_revision=None,
                 candidate_seed=260710, _rapid_authorization=None,
                 _adopted_authorization=None, _submit_resources=None,
                 _refill_journal=None, _pooled_submission=None):
    requested_active = int(target) + int(buffer)
    if requested_active > 0 and not scheduler_client.campaign_mutation_lock_is_held():
        raise SchedulerError("campaign refill requires the project mutation lock")
    rapid_authorized = False
    adopted_authorized = False
    pooled_mode = _is_pooled_submission(_pooled_submission)
    pooled_authorized = bool(
        pooled_mode and requested_active <= MAX_POOLED_ACTIVE)
    if pooled_mode and not pooled_authorized:
        raise SchedulerError(
            f"pooled feeder hard cap is {MAX_POOLED_ACTIVE}")
    if requested_active > MAX_STANDALONE_ACTIVE:
        rapid_authorized = (
            isinstance(_rapid_authorization, _RapidRefillAuthorization)
            and _rapid_authorization.seal is _RAPID_REFILL_SEAL
            and _rapid_authorization.target == requested_active
            and _rapid_authorization.max_samples == int(max_samples)
            and _rapid_authorization.solver_revision == str(solver_revision)
            and _rapid_authorization.library_revision == str(library_revision)
            and _rapid_authorization.candidate_seed == int(candidate_seed)
        )
        adopted_authorized = (
            isinstance(_adopted_authorization, _AdoptedRefillAuthorization)
            and _adopted_authorization.seal is _ADOPTED_REFILL_SEAL
            and _adopted_authorization.target == requested_active
            and _adopted_authorization.max_samples == int(max_samples)
            and _adopted_authorization.solver_revision == str(solver_revision)
            and _adopted_authorization.library_revision == str(library_revision)
            and _adopted_authorization.candidate_seed == int(candidate_seed)
            and _submit_resources == {
                "cpus": _adopted_authorization.cpus,
                "memory_mb": _adopted_authorization.memory_mb,
                "timeout_seconds": _adopted_authorization.timeout_seconds,
            }
        )
        if not (rapid_authorized or adopted_authorized or pooled_authorized):
            raise SchedulerError("production refill requires rapid promotion authorization")
    if _refill_journal is not None:
        if (not isinstance(_refill_journal, dict)
                or not isinstance(_refill_journal.get("events"), list)
                or _refill_journal["events"]):
            raise SchedulerError("refill journal must be a fresh events list")
        _refill_journal.update({
            "entered": True,
            "submitted_count": 0,
            "completed": False,
            "batch_commit": bool(
                adopted_authorized
                and _adopted_authorization.evidence_mode
                in {
                    "concurrent300_v1", "concurrent400_v1",
                    "dynamic_project_cap_v1",
                }),
        })
    st = load_state()
    committed_state = copy.deepcopy(st)
    # 로컬 장부나 다른 project가 아니라 scheduler의 MFT logical project가 source of truth다.
    hard_cap = max(1, int(target) + int(buffer))
    if (not pooled_authorized
            and hard_cap > MFT_PROJECT_MAX_ACTIVE_TASKS):
        raise SchedulerError(
            f"campaign hard cap {hard_cap} exceeds project maximum "
            f"{MFT_PROJECT_MAX_ACTIVE_TASKS}")
    dynamic_project_cap = bool(
        adopted_authorized
        and _adopted_authorization.evidence_mode == "dynamic_project_cap_v1")
    snapshot_options = {
        "require_exact_project_cap": dynamic_project_cap,
        "require_full_project": dynamic_project_cap,
    }
    if pooled_authorized:
        snapshot_options["max_project_active_tasks"] = (
            MAX_POOLED_PROJECT_ACTIVE_TASKS)
    campaign_counts, global_counts, allocations, capacity_gate = (
        scheduler_snapshot(hard_cap, **snapshot_options))
    ready_fit_slots = capacity_gate["ready_fit_slots"]
    campaign_active = sum(campaign_counts.values())
    data_rows, judged_ids = dataset_collection_snapshot()
    if target + buffer > 0:
        tasks = campaign_inventory()
        reserved_rows = reserved_unjudged_rows(st, tasks, judged_ids)
    else:
        reserved_rows = 0
    projected_rows = data_rows + reserved_rows
    immediate_headroom, total_cpus, free_cpus, global_active = cpu_submission_headroom(
        global_counts, allocations, ready_fit_slots)
    campaign_deficit = max(0, target + buffer - campaign_active)
    deficit = (
        min(campaign_deficit, capacity_gate["project_submission_slots"])
        if capacity_gate["submission_allowed"] else 0
    )
    if _refill_journal is not None:
        _refill_journal.update({
            "dataset_rows": int(data_rows),
            "reserved_rows": int(reserved_rows),
            "projected_rows": int(projected_rows),
            "campaign_deficit": int(campaign_deficit),
            "submission_deficit": int(deficit),
            "planned_count": 0,
            "stop_reason": None,
        })
    print(f"[feeder] campaign active {campaign_active} "
          f"(queued={campaign_counts['queued']}, attaching={campaign_counts['attaching']}, "
          f"running={campaign_counts['running']}), global active={global_active}, "
          f"CPU total/free={total_cpus}/{free_cpus}, ready_fit={ready_fit_slots}, "
          f"immediate_headroom={immediate_headroom}, "
          f"queue_state={capacity_gate['queue_state']}, "
          f"project={MFT_PROJECT} active/open="
          f"{capacity_gate['project_active']}/"
          f"{capacity_gate['project_submission_slots']}, "
          "dataset/reserved/projected="
          f"{data_rows}/{reserved_rows}/{projected_rows}")
    if data_rows >= max_samples:
        print(f"[feeder] dataset target reached ({data_rows}/{max_samples}) - no refill")
        if _refill_journal is not None:
            _refill_journal["stop_reason"] = "dataset_ceiling_reached"
            _refill_journal["completed"] = True
        return False
    if projected_rows >= max_samples:
        print(f"[feeder] no refill: projected dataset rows {projected_rows}/{max_samples}")
        if _refill_journal is not None:
            _refill_journal["stop_reason"] = "projected_ceiling_reached"
            _refill_journal["completed"] = True
        return True
    if deficit <= 0:
        print(
            f"[feeder] no refill: campaign_deficit={campaign_deficit}, "
            f"queue_state={capacity_gate['queue_state']}, "
            f"queue_reason={capacity_gate['queue_reason'] or '-'}"
        )
        if _refill_journal is not None:
            _refill_journal["stop_reason"] = "no_submission_deficit"
            _refill_journal["completed"] = True
        return True
    if not isinstance(solver_revision, str) or len(solver_revision) != 40:
        raise SchedulerError("a full pinned solver revision is required before refill")
    if not isinstance(library_revision, str) or len(library_revision) != 40:
        raise SchedulerError("a full pinned pyaedt_library revision is required before refill")
    candidate_generation = (
        f"{solver_revision}:{library_revision}:seed{int(candidate_seed)}")
    generation_changed = st.get("candidate_generation") != candidate_generation
    candidate_cursors = dict(st.get("candidate_cursors") or {})
    # Migrate the legacy single cursor once, then retain an independent cursor
    # for every revision/seed generation. Switching controllers cannot replay
    # candidates that an earlier generation already consumed.
    previous_generation = st.get("candidate_generation")
    if (previous_generation and previous_generation not in candidate_cursors
            and st.get("candidate_cursor") is not None):
        candidate_cursors[previous_generation] = int(st["candidate_cursor"])
    if candidate_generation in candidate_cursors:
        candidate_cursor = int(candidate_cursors[candidate_generation])
    else:
        candidate_cursor = cursor_after_valid_candidates(
            PILOT_RESERVED_VALID_CANDIDATES, seed=candidate_seed)
    if generation_changed:
        st["candidate_generation"] = candidate_generation
        st["candidate_cursor"] = candidate_cursor
        st.pop("candidate_raw_index", None)
    st["candidate_cursors"] = candidate_cursors
    n_new = min(deficit, (max_samples - projected_rows) // COUNT_PER_TASK)
    if _refill_journal is not None:
        _refill_journal["planned_count"] = int(n_new)
    if n_new <= 0:
        if _refill_journal is not None:
            _refill_journal["stop_reason"] = "no_planned_tasks"
            _refill_journal["completed"] = True
        return False
    batch_commit = bool(
        _refill_journal is not None
        and _refill_journal.get("batch_commit") is True)
    event_profile = None
    if _refill_journal is not None:
        with open(PROFILE_PATH, encoding="utf-8") as stream:
            event_profile = json.load(stream)
        if _submit_resources and _submit_resources.get("timeout_seconds") is not None:
            event_profile["timeout_seconds"] = int(
                _submit_resources["timeout_seconds"])
    planned = []
    ok = 0

    def submit_planned(item):
        nonlocal ok, committed_state
        event = item["event"]
        try:
            submit_kwargs = dict(_submit_resources or {})
            if _pooled_submission is not None:
                submit_kwargs.update(_pooled_submission)
            if pooled_authorized:
                submit_kwargs.update({
                    "required_hard_cap": hard_cap,
                    "max_project_active_tasks": (
                        MAX_POOLED_PROJECT_ACTIVE_TASKS),
                })
            if dynamic_project_cap:
                submit_kwargs["required_project_cap"] = hard_cap
            tid = submit(
                item["name"], item["workdir"], item["params"],
                solver_revision, library_revision, **submit_kwargs,
            )
        except Exception as exc:
            if event is not None:
                event["uncertain"] = isinstance(
                    exc, scheduler_client.TaskSubmissionUncertain)
                event["exception_type"] = type(exc).__name__
            st.clear()
            st.update(copy.deepcopy(committed_state))
            raise
        if tid is None:
            st.clear()
            st.update(copy.deepcopy(committed_state))
            raise SchedulerError(
                f"scheduler did not return a task ID for {item['name']}; "
                "candidate state was not advanced")
        if event is not None:
            event["task_id"] = int(tid)
            event["accepted_or_reconciled"] = True
        ok += 1
        st["submitted_samples"] += COUNT_PER_TASK
        st.setdefault("outstanding", []).append(tid)
        st.setdefault("task_expected_rows", {})[str(tid)] = COUNT_PER_TASK
        if not batch_commit:
            # Legacy modes retain their per-task ledger commit.  The pool400
            # mode pre-seals the whole batch and commits once below.
            save_state(st)
            if event is not None:
                event["ledger_committed"] = True
                _refill_journal["submitted_count"] += 1
            committed_state = copy.deepcopy(st)
        time.sleep(0.3)

    for _ in range(n_new):
        st["serial"] += 1
        generation = f"s{solver_revision[:7]}-l{library_revision[:7]}"
        name = f"mft-camp-{generation}-{st['serial']:05d}"
        wd = f"mft_c_t{st['serial'] % 500:03d}"  # 500개 디렉토리 풀 재사용 (클론 재활용)
        next_cursor, raw_index, params = next_valid_candidate(
            int(st.get("candidate_cursor", 0)), seed=candidate_seed)
        st["candidate_cursor"] = next_cursor
        st["candidate_cursors"][candidate_generation] = next_cursor
        st["candidate_raw_index"] = raw_index
        event = None
        if _refill_journal is not None:
            event_identity = scheduler_client.verification_submission_identity(
                name, params, event_profile, solver_revision, library_revision)
            event = {
                "name": name,
                "candidate_raw_index": int(raw_index),
                "dedupe_key": event_identity["dedupe_key"],
                "task_id": None,
                "accepted_or_reconciled": False,
                "ledger_committed": False,
                "uncertain": False,
            }
            _refill_journal["events"].append(event)
        item = {"name": name, "workdir": wd, "params": params, "event": event}
        if batch_commit:
            planned.append(item)
        else:
            submit_planned(item)
    if batch_commit:
        # All names/cursors/dedupe identities now exist in one in-memory plan.
        # The controller wrapper persists that full plan before the first POST,
        # then this loop performs only idempotent scheduler submissions.
        for item in planned:
            submit_planned(item)
        # One immutable feeder generation and one ledger journal generation
        # commit the complete accepted batch at its boundary.
        save_state(st)
        for item in planned:
            if item["event"] is not None:
                item["event"]["ledger_committed"] = True
        _refill_journal["submitted_count"] = ok
        committed_state = copy.deepcopy(st)
    print(f"[feeder] campaign active {campaign_active} -> +{ok}/{n_new} tasks "
          f"(누적 제출 샘플 {st['submitted_samples']})")
    if _refill_journal is not None:
        _refill_journal["stop_reason"] = "submitted"
        _refill_journal["completed"] = True
    return True


def publish_ready_marker(path, solver_revision, library_revision):
    """Atomically prove that this feeder completed one guarded cycle."""
    if not path:
        return
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    staged = f"{target}.{os.getpid()}.tmp"
    payload = {
        "pid": os.getpid(),
        "ready_at": time.time(),
        "solver_revision": solver_revision,
        "library_revision": library_revision,
    }
    try:
        with open(staged, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
        os.replace(staged, target)
    finally:
        if os.path.exists(staged):
            os.remove(staged)


def _argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, default=None, help="반복 주기 [s]")
    ap.add_argument("--max-samples", type=int, default=12000)
    ap.add_argument(
        "--target",
        type=int,
        default=None,
        help=(
            "compatibility/one-shot target; pooled loops read the durable "
            "scheduler simulation-policy every cycle"
        ),
    )
    ap.add_argument("--buffer", type=int, default=BUFFER,
                    help="목표 초과 대기 버퍼")
    ap.add_argument("--solver-revision")
    ap.add_argument("--library-revision")
    ap.add_argument("--trust-pinned-revisions", action="store_true")
    ap.add_argument("--candidate-seed", type=int, default=260710)
    ap.add_argument(
        "--aedt-pooled",
        action="store_true",
        help="attach MFT tasks to shared AEDT pool Desktops",
    )
    ap.add_argument("--aedt-pool-url", metavar="URL")
    ap.add_argument(
        "--aedt-pool-pkg-root",
        default=DEFAULT_AEDT_POOL_PKG_ROOT,
        metavar="PATH",
    )
    ap.add_argument(
        "--aedt-pool-client-token-file",
        "--aedt-pool-token-file",
        dest="aedt_pool_client_token_file",
        default=DEFAULT_AEDT_POOL_CLIENT_TOKEN_FILE,
        metavar="PATH",
        help=(
            "path to the lease-create-only client credential; "
            "--aedt-pool-token-file is a deprecated alias"
        ),
    )
    ap.add_argument(
        "--aedt-session-version",
        default=DEFAULT_AEDT_SESSION_VERSION,
        metavar="VERSION",
    )
    ap.add_argument(
        "--aedt-isolation-policy",
        choices=AEDT_ISOLATION_POLICIES,
        default=DEFAULT_AEDT_ISOLATION_POLICY,
        help=(
            "start family-isolated; switch to shared_if_compatible only "
            "after the mixed MFT/motor canary passes"
        ),
    )
    ap.add_argument(
        "--pooled-cpus", type=int, default=DEFAULT_POOLED_CPUS, metavar="N")
    ap.add_argument(
        "--pooled-memory-mb",
        type=int,
        default=DEFAULT_POOLED_MEMORY_MB,
        metavar="N",
    )
    ap.add_argument(
        "--ready-file",
        help="atomically written after the first successful guarded cycle",
    )
    return ap


def main():
    ap = _argument_parser()
    args = ap.parse_args()
    pooled_submission = _pooled_submission_kwargs(args)
    if args.target is not None:
        _validate_cycle_target(args, args.target)

    if args.trust_pinned_revisions:
        for revision, flag in (
                (args.solver_revision, "--solver-revision"),
                (args.library_revision, "--library-revision")):
            if (not isinstance(revision, str) or len(revision) != 40
                    or any(char not in "0123456789abcdefABCDEF"
                           for char in revision)):
                raise SchedulerError(
                    f"{flag} must be a full 40-character hex string when "
                    "--trust-pinned-revisions is set"
                )
        print(
            "[feeder] WARNING: local revision vetting and the p08 completion "
            "gate were bypassed; "
            f"using pinned solver SHA {args.solver_revision} and "
            f"library SHA {args.library_revision}"
        )

    may_submit = bool(
        args.target is None
        or args.target + args.buffer > 0
        or (args.aedt_pooled and args.loop)
    )
    if may_submit and not args.trust_pinned_revisions:
        if args.solver_revision != al_driver._current_solver_revision():
            raise SchedulerError("feeder solver revision is not the current vetted local solver")
        if args.library_revision != al_driver._current_library_revision():
            raise SchedulerError("feeder library revision is not the current clean local library")
        validate_p08_completion(
            args.solver_revision, args.library_revision,
            seed=args.candidate_seed)

    def guarded_step():
        def run_cycle(target, policy, requested_active):
            if requested_active > 0:
                _require_deployed_revisions(
                    args.solver_revision, args.library_revision
                )
            if policy is not None:
                print(
                    "[feeder] scheduler policy "
                    f"revision={policy['policy_revision']} "
                    f"desired={policy['desired_simulations']} "
                    f"effective={policy.get('effective_simulations')} "
                    f"validated={policy['validated_concurrency_limit']}"
                )
            step_options = {}
            if pooled_submission is not None:
                step_options["pooled_submission"] = pooled_submission
            return step(
                args.max_samples, target=target, buffer=args.buffer,
                solver_revision=args.solver_revision,
                library_revision=args.library_revision,
                candidate_seed=args.candidate_seed,
                **step_options,
            )

        # Serialize policy observation with every submission in that cycle.
        # The WEB UI uses the same host-wide lock, so a concurrent target
        # reduction cannot race an already-authorized refill batch.
        if args.aedt_pooled and (args.loop or args.target is None):
            with campaign_mutation_lock():
                target, policy = _cycle_target(args)
                requested_active = _validate_cycle_target(args, target)
                return run_cycle(target, policy, requested_active)

        target, policy = _cycle_target(args)
        requested_active = _validate_cycle_target(args, target)
        if requested_active > 0:
            with campaign_mutation_lock():
                return run_cycle(target, policy, requested_active)
        return run_cycle(target, policy, requested_active)

    if args.once or not args.loop:
        guarded_step()
        publish_ready_marker(
            args.ready_file, args.solver_revision, args.library_revision)
        return
    controller_lock = FileLock(CONTROLLER_LOCK)
    try:
        with controller_lock.acquire(timeout=0):
            ready_published = False
            while True:
                try:
                    keep_running = guarded_step()
                    if not ready_published:
                        publish_ready_marker(
                            args.ready_file,
                            args.solver_revision,
                            args.library_revision,
                        )
                        ready_published = True
                    if not keep_running:
                        break
                except Exception as e:
                    print(f"[feeder] step error: {e}")
                time.sleep(args.loop)
    except FileLockTimeout as exc:
        raise SchedulerError(
            f"another feeder controller owns {CONTROLLER_LOCK}"
        ) from exc


if __name__ == "__main__":
    main()
