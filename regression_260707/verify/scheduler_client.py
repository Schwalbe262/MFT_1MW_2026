"""
AL 검증용 스케줄러 클라이언트: 후보 params JSON -> 태스크 제출 -> RESULT_JSON 회수.
"""
import hashlib
import json
import math
import os
import re
import shlex
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import requests
from filelock import FileLock

try:
    from ..model_targets import CORE_REGION_TEMPERATURE_TARGETS
except ImportError:
    from model_targets import CORE_REGION_TEMPERATURE_TARGETS

SCHEDULER = "http://127.0.0.1:8000"
MFT_PROJECT = "MFT_1MW_2026v1"
MFT_PROJECT_MAX_ACTIVE_TASKS = 300
MFT_ACTIVE_STATUSES = ("queued", "attaching", "running")
LEGACY_MFT_NAME_PREFIX = "mft-"
MFT_PROJECT_REPOS = [
    {
        "url": "https://github.com/Schwalbe262/MFT_1MW_2026.git",
        "ref": "main",
        "subdir": "MFT_1MW_2026",
    },
    {
        "url": "https://github.com/Schwalbe262/pyaedt_library.git",
        "ref": "main",
    },
]
MFT_PROJECT_SETUP = (
    "source /etc/profile.d/lmod.sh 2>/dev/null || true\n"
    "module load ansys-electronics/v252 2>/dev/null || "
    "export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64\n"
    "export FLEXLM_TIMEOUT=3000000"
)
MFT_PROJECT_ENTRYPOINTS = [
    {
        "path": "run_simulation_260706.py",
        "conda_env": "pyaedt2026v1",
        "workdir": "MFT_1MW_2026",
    },
    {
        "path": "run_campaign.py",
        "conda_env": "pyaedt2026v1",
        "workdir": "MFT_1MW_2026",
    },
]
MFT_PROJECT_CLEANUP_GLOBS = "*.aedtresults"
MFT_PROJECT_OUTPUT_GLOBS = (
    "simulation_results_*.csv,failed_samples_260706.jsonl,"
    "results_parts_260706/*.parquet"
)
MFT_PROJECT_SIM_SUBDIR = "simulation"
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "").strip()
if not _LOCALAPPDATA:
    _LOCALAPPDATA = str(Path.home() / "AppData" / "Local")
CAMPAIGN_MUTATION_LOCK_PATH = (
    Path(_LOCALAPPDATA) / "MFT_1MW_2026" / "campaign-mutation.lock")
CAMPAIGN_MUTATION_LOCK_TIMEOUT = 15 * 60
_CAMPAIGN_LOCK_STATE = threading.local()
# MFT solver tasks are single-node. Fluent drops the generic Intel-MPI
# bootstrap variable and otherwise sees SLURM_JOB_ID and launches nested srun;
# its launcher consumes FLUENT_MPIRUN_FLAGS for the explicit local bootstrap.
BASE = ("source /etc/profile.d/lmod.sh 2>/dev/null || true; "
        "module load ansys-electronics/v252 || export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64; "
        "export FLEXLM_TIMEOUT=3000000; "
        "export I_MPI_HYDRA_BOOTSTRAP=fork; "
        "export FLUENT_MPIRUN_FLAGS='-bootstrap fork'; "
        "sleep $((RANDOM % 60)); ")

RESULT_VALID = "valid"
RESULT_INVALID = "invalid"
RESULT_MISSING = "missing"
MAX_STDOUT_BYTES = 1_048_576
_STANDARD_PROFILE = json.loads(
    (Path(__file__).resolve().parent / "profiles" / "standard.json").read_text(
        encoding="utf-8"))
STANDARD_PROFILE_CONTRACT = dict(_STANDARD_PROFILE["param_overrides"])
DEFAULT_TASK_TIMEOUT_SECONDS = int(_STANDARD_PROFILE["timeout_seconds"])
LOCAL_SCRATCH_ROOT = "/enroot"
LOCAL_SCRATCH_MIN_FREE_KB = 200 * 1024 * 1024
LOCAL_SCRATCH_STALE_MINUTES = 8 * 60
GPFS_RUNS_REMOTE_CWD = "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs"
GPFS_SCRATCH_STALE_MINUTES = 8 * 60
SCRATCH_LEAF_PREFIX = "mft_campaign-"
SCRATCH_LEAF_PATTERN = f"{SCRATCH_LEAF_PREFIX}*-t????????????????"
MAX_SCRATCH_LEAF_LENGTH = 198
MAX_TRUSTED_TEMPERATURE_C = 4700.0
MIN_TRUSTED_TEMPERATURE_C = -273.15

MANDATORY_TEMPERATURE_COLUMNS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_core",
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_core_center_max",
    *CORE_REGION_TEMPERATURE_TARGETS,
)
SIDE_TEMPERATURE_COLUMNS = (
    "T_max_Rx_side",
    "Tprobe_Rx_side_leeward_max",
)


class ResultFetchError(RuntimeError):
    """The scheduler stdout could not be read reliably."""


class TaskLookupError(RuntimeError):
    """The scheduler task inventory could not be reconciled reliably."""


class TaskSubmissionUncertain(RuntimeError):
    """A POST may have succeeded, but its durable task ID is not known yet."""


class ProjectContractError(RuntimeError):
    """The live logical-project mutation contract is absent or unsafe."""


class ProjectCapacityError(RuntimeError):
    """No logical MFT project slot remains for a new task."""


@dataclass(frozen=True)
class ResultFetch:
    state: str
    result: dict | None = None


@contextmanager
def campaign_mutation_lock(path=None, timeout=None):
    """Serialize every MFT task mutation across processes on this host."""
    lock_path = Path(path or CAMPAIGN_MUTATION_LOCK_PATH)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(
        str(lock_path),
        timeout=(CAMPAIGN_MUTATION_LOCK_TIMEOUT if timeout is None else timeout),
    )
    with lock:
        previous_depth = int(getattr(_CAMPAIGN_LOCK_STATE, "depth", 0))
        _CAMPAIGN_LOCK_STATE.depth = previous_depth + 1
        try:
            yield lock
        finally:
            _CAMPAIGN_LOCK_STATE.depth = previous_depth


def campaign_mutation_lock_is_held():
    """Return whether this thread owns the common mutation lock context."""
    return int(getattr(_CAMPAIGN_LOCK_STATE, "depth", 0)) > 0


def validate_project_mutation_contract(
        project, *, expected_cap=None, require_full=False):
    """Validate the bounded target and optionally every mutation field."""
    if not isinstance(project, dict):
        raise ProjectContractError("scheduler returned an invalid MFT project")
    if str(project.get("name") or "").strip() != MFT_PROJECT:
        raise ProjectContractError(
            f"scheduler project identity is not {MFT_PROJECT!r}")
    raw_cap = project.get("max_active_tasks")
    if type(raw_cap) is not int:
        raise ProjectContractError(
            "scheduler MFT project max_active_tasks is invalid")
    cap = raw_cap
    if not 1 <= cap <= MFT_PROJECT_MAX_ACTIVE_TASKS:
        raise ProjectContractError(
            "scheduler MFT project max_active_tasks must be an integer "
            f"between 1 and {MFT_PROJECT_MAX_ACTIVE_TASKS}, got {cap}")
    if expected_cap is not None:
        if (isinstance(expected_cap, bool)
                or not isinstance(expected_cap, int)
                or not 1 <= expected_cap <= MFT_PROJECT_MAX_ACTIVE_TASKS):
            raise ProjectContractError("expected MFT project cap is invalid")
        if cap != expected_cap:
            raise ProjectContractError(
                f"scheduler MFT project cap changed: expected "
                f"{expected_cap}, got {cap}")
    if project.get("auto_pull") is not False:
        raise ProjectContractError(
            "scheduler MFT project auto_pull must be exactly false")
    if require_full:
        full_checks = {
            "repos": project.get("repos") == MFT_PROJECT_REPOS,
            "setup": project.get("setup") == MFT_PROJECT_SETUP,
            "entrypoints": project.get("entrypoints") == MFT_PROJECT_ENTRYPOINTS,
            "cleanup_globs": project.get("cleanup_globs")
                == MFT_PROJECT_CLEANUP_GLOBS,
            "output_globs": project.get("output_globs")
                == MFT_PROJECT_OUTPUT_GLOBS,
            "sim_subdir": project.get("sim_subdir")
                == MFT_PROJECT_SIM_SUBDIR,
        }
        if not all(full_checks.values()):
            raise ProjectContractError(
                f"scheduler MFT project mutation fields drifted: {full_checks}")
    contract = {
        "name": MFT_PROJECT,
        "max_active_tasks": cap,
        "auto_pull": False,
    }
    if project.get("updated_at") is not None:
        contract["updated_at"] = project.get("updated_at")
    return contract


def require_live_project_mutation_contract(
        *, expected_cap=None, require_full=False):
    """Read and validate the live project immediately before a task POST."""
    try:
        response = requests.get(
            f"{SCHEDULER}/api/projects/{MFT_PROJECT}", timeout=30)
        response.raise_for_status()
        project = response.json()
    except Exception as exc:
        raise ProjectContractError(
            f"failed to read scheduler project {MFT_PROJECT!r}: {exc}") from exc
    return validate_project_mutation_contract(
        project, expected_cap=expected_cap, require_full=require_full)


def project_submission_snapshot(
        projects, project_tasks, required_hard_cap, legacy_tasks=None, *,
        require_exact_project_cap=False, require_full_project=False):
    """Count tagged and projectless legacy MFT demand without double counting."""
    if isinstance(required_hard_cap, bool) or not isinstance(required_hard_cap, int):
        raise ProjectCapacityError("MFT project hard cap must be a positive integer")
    if required_hard_cap < 1 or required_hard_cap > MFT_PROJECT_MAX_ACTIVE_TASKS:
        raise ProjectCapacityError(
            f"MFT project hard cap must be between 1 and "
            f"{MFT_PROJECT_MAX_ACTIVE_TASKS}")
    if not isinstance(projects, list):
        raise ProjectCapacityError("scheduler returned an invalid project inventory")
    matches = [
        project for project in projects
        if isinstance(project, dict)
        and str(project.get("name") or "").strip() == MFT_PROJECT
    ]
    if len(matches) != 1:
        raise ProjectCapacityError(
            f"scheduler project {MFT_PROJECT!r} is missing or ambiguous")
    project_contract = validate_project_mutation_contract(
        matches[0],
        expected_cap=(required_hard_cap if require_exact_project_cap else None),
        require_full=require_full_project,
    )
    if not isinstance(project_tasks, list):
        raise ProjectCapacityError(
            "scheduler returned an invalid MFT project task inventory")
    if legacy_tasks is None:
        legacy_tasks = []
    if not isinstance(legacy_tasks, list):
        raise ProjectCapacityError(
            "scheduler returned an invalid legacy MFT task inventory")

    def indexed(rows, source, allowed_projects, require_prefix=False):
        inventory = {}
        for task in rows:
            if not isinstance(task, dict):
                raise ProjectCapacityError(
                    f"scheduler returned an invalid {source} task")
            task_id = task.get("id", task.get("task_id"))
            if (isinstance(task_id, bool) or not isinstance(task_id, int)
                    or task_id <= 0):
                raise ProjectCapacityError(
                    f"scheduler returned an invalid {source} task ID")
            if task_id in inventory:
                raise ProjectCapacityError(
                    f"scheduler returned duplicate {source} task ID {task_id}")
            project_name = str(task.get("project") or "").strip()
            if project_name not in allowed_projects:
                raise ProjectCapacityError(
                    f"scheduler returned {source} task {task_id} from "
                    f"unexpected project {project_name!r}")
            status = str(task.get("status") or "").strip().lower()
            if status not in MFT_ACTIVE_STATUSES:
                raise ProjectCapacityError(
                    f"scheduler returned unexpected {source} active task status: "
                    f"{status!r}")
            if (require_prefix
                    and not str(task.get("name") or "").startswith(
                        LEGACY_MFT_NAME_PREFIX)):
                raise ProjectCapacityError(
                    f"scheduler legacy MFT filter returned task {task_id} "
                    "outside the mft- namespace")
            inventory[task_id] = dict(task)
        return inventory

    tagged = indexed(project_tasks, "MFT project", {MFT_PROJECT})
    legacy_scan = indexed(
        legacy_tasks, "legacy MFT", {"", MFT_PROJECT}, require_prefix=True)
    combined = dict(tagged)
    for task_id, task in legacy_scan.items():
        if task_id in combined:
            if str(task.get("project") or "").strip() != MFT_PROJECT:
                raise ProjectCapacityError(
                    f"scheduler task {task_id} is both project-tagged and legacy")
            continue
        combined[task_id] = task

    counts = {status: 0 for status in MFT_ACTIVE_STATUSES}
    tagged_counts = {status: 0 for status in MFT_ACTIVE_STATUSES}
    legacy_counts = {status: 0 for status in MFT_ACTIVE_STATUSES}
    for task in combined.values():
        status = str(task.get("status") or "").strip().lower()
        counts[status] += 1
        bucket = (
            tagged_counts
            if str(task.get("project") or "").strip() == MFT_PROJECT
            else legacy_counts
        )
        bucket[status] += 1
    active = sum(counts.values())
    server_cap = project_contract["max_active_tasks"]
    server_open_slots = max(0, server_cap - active)
    stage_open_slots = max(0, required_hard_cap - active)
    return {
        "project": MFT_PROJECT,
        "project_max_active_tasks": server_cap,
        "project_required_hard_cap": required_hard_cap,
        "project_counts": counts,
        "project_tagged_counts": tagged_counts,
        "legacy_counts": legacy_counts,
        "project_active": active,
        "project_tagged_active": sum(tagged_counts.values()),
        "legacy_active": sum(legacy_counts.values()),
        "project_server_open_slots": server_open_slots,
        "project_stage_open_slots": stage_open_slots,
        "project_submission_slots": min(server_open_slots, stage_open_slots),
    }


def _task_rows(response, source):
    response.raise_for_status()
    payload = response.json()
    rows = payload if isinstance(payload, list) else (
        payload.get("tasks") if isinstance(payload, dict) else None)
    if not isinstance(rows, list):
        raise ProjectCapacityError(
            f"scheduler returned an invalid {source} task inventory")
    return rows


def live_project_submission_snapshot(
        required_hard_cap=MFT_PROJECT_MAX_ACTIVE_TASKS, *,
        require_exact_project_cap=False, require_full_project=False):
    """Read the absolute logical-project budget while the mutation lock is held."""
    if not campaign_mutation_lock_is_held():
        raise ProjectCapacityError(
            "MFT project capacity must be checked under the campaign mutation lock")
    project = require_live_project_mutation_contract(
        expected_cap=(required_hard_cap if require_exact_project_cap else None),
        require_full=require_full_project,
    )
    statuses = ",".join(MFT_ACTIVE_STATUSES)
    try:
        project_tasks = _task_rows(requests.get(
            f"{SCHEDULER}/api/tasks",
            params={
                "limit": 10000,
                "project": MFT_PROJECT,
                "status": statuses,
            },
            timeout=30,
        ), "MFT project")
        legacy_tasks = _task_rows(requests.get(
            f"{SCHEDULER}/api/tasks",
            params={
                "limit": 10000,
                "name_prefix": LEGACY_MFT_NAME_PREFIX,
                "status": statuses,
            },
            timeout=30,
        ), "legacy MFT")
    except ProjectCapacityError:
        raise
    except Exception as exc:
        raise ProjectCapacityError(
            f"failed to read MFT project task capacity: {exc}") from exc
    return project_submission_snapshot(
        [project], project_tasks, required_hard_cap, legacy_tasks=legacy_tasks,
        require_exact_project_cap=require_exact_project_cap,
        # Full raw record was authenticated by the live contract read above.
        require_full_project=False,
    )


def reconcile_task_id(name, dedupe_key, attempts=3, retry_delay=1):
    """Find the newest exact task identity, including projectless legacy rows."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                f"{SCHEDULER}/api/tasks",
                params={
                    "limit": 10000,
                    "name_prefix": name,
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
            if not isinstance(tasks, list):
                raise ValueError("task inventory is not a list")
            exact = [
                task for task in tasks
                if isinstance(task, dict)
                and task.get("name") == name
                and task.get("dedupe_key") == dedupe_key
            ]
            foreign = [
                task for task in exact
                if str(task.get("project") or "").strip()
                not in ("", MFT_PROJECT)
            ]
            if foreign:
                projects = sorted({
                    str(task.get("project") or "").strip()
                    for task in foreign
                })
                raise ValueError(
                    f"exact task identity exists outside {MFT_PROJECT!r}: "
                    f"{projects}")
            matches = [
                task for task in exact
                if str(task.get("project") or "").strip()
                in ("", MFT_PROJECT)
            ]
            if not matches:
                return None
            return max(int(task["id"]) for task in matches)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(retry_delay)
    raise TaskLookupError(
        f"failed to reconcile task {name!r} after {attempts} attempts: {last_error}"
    ) from last_error


def effective_verification_params(params, profile):
    """Return the exact parameter payload written for a verification task."""
    if not isinstance(params, dict):
        raise TypeError("verification params must be a dict")
    if not isinstance(profile, dict):
        raise TypeError("verification profile must be a dict")
    overrides = profile.get("param_overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError("verification profile param_overrides must be a dict")
    merged = dict(params)
    merged.update(overrides)
    return merged


def verification_submission_identity(
        name, params, profile, solver_revision, library_revision,
        dedupe_scope=None):
    """Return the one canonical payload/dedupe identity used for reconciliation."""
    if (not isinstance(solver_revision, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", solver_revision)):
        raise ValueError("solver_revision must be a full 40-character git SHA")
    if (not isinstance(library_revision, str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", library_revision)):
        raise ValueError(
            "library_revision must be a full 40-character git SHA")
    solver_revision = solver_revision.lower()
    library_revision = library_revision.lower()
    merged = effective_verification_params(params, profile)
    pjson = json.dumps(merged, separators=(",", ":"))
    parameter_digest = hashlib.sha256(pjson.encode("utf-8")).hexdigest()[:16]
    normalized_scope = ""
    if dedupe_scope is not None:
        normalized_scope = str(dedupe_scope).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized_scope):
            raise ValueError("dedupe_scope must be a 64-character SHA256")
    dedupe_key = (
        f"mft-al:{name}:{solver_revision}:{library_revision}:{parameter_digest}"
        + (f":scope-{normalized_scope}" if normalized_scope else "")
    )
    return {
        "solver_revision": solver_revision,
        "library_revision": library_revision,
        "merged": merged,
        "parameter_json": pjson,
        "parameter_digest": parameter_digest,
        "dedupe_key": dedupe_key,
        "dedupe_scope": normalized_scope or None,
    }


def verification_dedupe_key(
        name, params, profile, solver_revision, library_revision,
        dedupe_scope=None):
    return verification_submission_identity(
        name, params, profile, solver_revision, library_revision,
        dedupe_scope=dedupe_scope)["dedupe_key"]


def _normalized_submission_env(submission_env):
    if submission_env is None:
        return {}
    if not isinstance(submission_env, dict):
        raise TypeError("submission_env must be a dict")
    normalized = {}
    for key, value in submission_env.items():
        key = str(key)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", key):
            raise ValueError(f"invalid submission environment key: {key!r}")
        value = str(value)
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"invalid submission environment value for {key}")
        normalized[key] = value
    return normalized


def submit_verification(
        name, workdir, params: dict, profile: dict, mem_mb=32768, cpus=4,
        solver_revision=None, library_revision=None,
        required_project_cap=None, *, aedt_backend="standalone",
        scheduling_profile="fea_bursty", submission_env=None,
        dedupe_scope=None):
    """Submit one MFT task under the shared cross-process mutation lock."""
    if campaign_mutation_lock_is_held():
        return _submit_verification_locked(
            name, workdir, params, profile, mem_mb=mem_mb, cpus=cpus,
            solver_revision=solver_revision,
            library_revision=library_revision,
            required_project_cap=required_project_cap,
            aedt_backend=aedt_backend,
            scheduling_profile=scheduling_profile,
            submission_env=submission_env,
            dedupe_scope=dedupe_scope,
        )
    with campaign_mutation_lock():
        return _submit_verification_locked(
            name, workdir, params, profile, mem_mb=mem_mb, cpus=cpus,
            solver_revision=solver_revision,
            library_revision=library_revision,
            required_project_cap=required_project_cap,
            aedt_backend=aedt_backend,
            scheduling_profile=scheduling_profile,
            submission_env=submission_env,
            dedupe_scope=dedupe_scope,
        )


def _submit_verification_locked(
        name, workdir, params: dict, profile: dict, mem_mb=32768, cpus=4,
        solver_revision=None, library_revision=None,
        required_project_cap=None, *, aedt_backend="standalone",
        scheduling_profile="fea_bursty", submission_env=None,
        dedupe_scope=None):
    """후보 파라미터를 인라인 JSON으로 실어 fixed 모드 검증 태스크 제출. 반환: task_id 또는 None"""
    if not campaign_mutation_lock_is_held():
        raise RuntimeError("MFT task mutation requires the campaign mutation lock")
    identity = verification_submission_identity(
        name, params, profile, solver_revision, library_revision,
        dedupe_scope=dedupe_scope)
    solver_revision = identity["solver_revision"]
    library_revision = identity["library_revision"]
    merged = identity["merged"]
    timeout_seconds = int(profile.get(
        "timeout_seconds", DEFAULT_TASK_TIMEOUT_SECONDS))
    if timeout_seconds <= 0:
        raise ValueError("verification timeout_seconds must be positive")
    pjson = identity["parameter_json"]
    parameter_digest = identity["parameter_digest"]
    dedupe_key = identity["dedupe_key"]
    if aedt_backend not in {"standalone", "pooled"}:
        raise ValueError("aedt_backend must be standalone or pooled")
    if scheduling_profile != "fea_bursty":
        raise ValueError("MFT FEA task scheduling_profile must be fea_bursty")
    normalized_env = _normalized_submission_env(submission_env)
    submission_provenance = {
        "aedt_backend": aedt_backend,
        "scheduling_profile": scheduling_profile,
        "dedupe_scope": identity["dedupe_scope"],
        "submission_env": normalized_env,
    }
    env_exports = "".join(
        f"export {key}={shlex.quote(value)}; "
        for key, value in sorted(normalized_env.items())
    )
    provenance_line = json.dumps(
        submission_provenance, sort_keys=True, separators=(",", ":"))
    extra = profile.get("cli_flags", "")
    run_identity = (
        f"s{solver_revision[:12]}-l{library_revision[:12]}-p{parameter_digest}")
    isolated_workdir = f"{workdir}-{run_identity}"
    safe_workdir = re.sub(
        r"[^A-Za-z0-9_-]+", "_", isolated_workdir).strip("_-")
    if not safe_workdir:
        safe_workdir = f"mft-{parameter_digest}"
    # The task name is part of the durable dedupe identity. Include its digest
    # so a retry with the same candidate cannot race terminal cleanup from the
    # previous task. Keep the basename below common filesystem NAME_MAX limits.
    task_identity = hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()[:16]
    scratch_suffix = f"-t{task_identity}"
    safe_workdir_limit = (
        MAX_SCRATCH_LEAF_LENGTH - len(SCRATCH_LEAF_PREFIX) - len(scratch_suffix))
    scratch_leaf = (
        f"{SCRATCH_LEAF_PREFIX}{safe_workdir[:safe_workdir_limit]}{scratch_suffix}")
    scratch_workdir = f"{LOCAL_SCRATCH_ROOT}/{scratch_leaf}"
    quoted_workdir = '"${MFT_WORKDIR}"'
    quoted_repo = '"${MFT_WORKDIR}/repo"'
    quoted_library = '"${MFT_WORKDIR}/pyaedt_library"'
    cleanup_workdirs = '"${MFT_NVME_WORKDIR}" "${MFT_GPFS_WORKDIR}"'
    select_workdir = (
        "MFT_GPFS_ROOT=$PWD; "
        f'MFT_GPFS_WORKDIR="$MFT_GPFS_ROOT/{scratch_leaf}"; '
        f"MFT_NVME_WORKDIR={shlex.quote(scratch_workdir)}; "
        'find "$MFT_GPFS_ROOT" -mindepth 1 -maxdepth 1 -type d '
        '-user "$USER" '
        f"-name {shlex.quote(SCRATCH_LEAF_PATTERN)} "
        f"-mmin +{GPFS_SCRATCH_STALE_MINUTES} -exec rm -rf -- {{}} + "
        "2>/dev/null || true; "
        f"MFT_ENROOT_FREE_KB=$(df -Pk {LOCAL_SCRATCH_ROOT} 2>/dev/null "
        "| awk 'NR==2 {print $4}'); "
        f"if [ \"$(findmnt -n -o FSTYPE -T {LOCAL_SCRATCH_ROOT} 2>/dev/null)\" = xfs ] "
        f"&& [ \"${{MFT_ENROOT_FREE_KB:-0}}\" -ge {LOCAL_SCRATCH_MIN_FREE_KB} ]; then "
        "MFT_WORKDIR=$MFT_NVME_WORKDIR; "
        f"find {LOCAL_SCRATCH_ROOT} -mindepth 1 -maxdepth 1 -type d "
        "-user \"$USER\" -name 'mft_*' "
        f"-mmin +{LOCAL_SCRATCH_STALE_MINUTES} -exec rm -rf -- {{}} + "
        "2>/dev/null || true; "
        "else MFT_WORKDIR=$MFT_GPFS_WORKDIR; fi; "
        "printf 'MFT_WORKDIR %s\\n' \"$MFT_WORKDIR\"; "
    )
    lib_clone = (f"([ -d {quoted_library}/.git ] || {{ [ ! -e {quoted_library} ] && "
                 "git clone -q --depth 1 "
                 f"https://github.com/Schwalbe262/pyaedt_library.git {quoted_library}.tmp.$$ "
                  f"&& {{ mv -T {quoted_library}.tmp.$$ {quoted_library} 2>/dev/null "
                  f"|| rm -rf {quoted_library}.tmp.$$; }}; }}) && "
                  f"git -C {quoted_library} fetch -q origin {library_revision} && "
                  f"git -C {quoted_library} checkout -q --detach {library_revision} && "
                  f"git -C {quoted_library} diff --quiet HEAD -- && "
                  f"git -C {quoted_library} clean -q -ffd && "
                  f"git -C {quoted_library} clean -q -ffdX && "
                  f"test -z \"$(git -C {quoted_library} status --porcelain --untracked-files=all)\" && "
                  f"test \"$(git -C {quoted_library} rev-parse HEAD)\" = \"{library_revision}\" && "
                  f"[ -d {quoted_library}/src ] && "
                  f"printf 'MFT_LIBRARY_GIT_HASH {library_revision}\\n' && ")
    run_group = (
        env_exports
        + f"printf 'MFT_SUBMISSION_PROVENANCE %s\\n' "
          f"{shlex.quote(provenance_line)}; "
        + f"mkdir -p {quoted_workdir} && "
        + lib_clone
        + f"([ -d {quoted_repo}/.git ] || git clone -q --depth 1 "
          f"https://github.com/Schwalbe262/MFT_1MW_2026.git {quoted_repo}) && "
        + f"cd {quoted_repo} && git fetch -q origin {solver_revision} && "
        + f"git checkout -q --detach {solver_revision} && "
        + "git diff --quiet HEAD -- && git clean -q -ffd && git clean -q -ffdX && "
        + "test -z \"$(git status --porcelain --untracked-files=all)\" && "
        + f"test \"$(git rev-parse HEAD)\" = \"{solver_revision}\" && "
        + f"printf '%s' {shlex.quote(pjson)} > cand.json && "
        + f"python run_simulation_260706.py --fixed {extra} --params cand.json; "
        + "simulation_rc=$?; "
        + f"printf 'MFT_LIBRARY_GIT_HASH {library_revision}\\n'; "
        + "exit $simulation_rc"
    )
    # Setup and simulation are one fail-fast subshell. The parent remains in the
    # scheduler workspace, so unconditional cleanup can target only this clone.
    cmd = (
        BASE
        + f"( {select_workdir}cleanup() {{ rm -rf -- {cleanup_workdirs} 2>/dev/null; }}; "
        + "trap cleanup EXIT; trap 'exit 143' TERM INT; "
        + run_group
        + " )"
    )
    payload = {
        "name": name, "project": MFT_PROJECT,
        "remote_cwd": GPFS_RUNS_REMOTE_CWD,
        "command": cmd, "required_capability": "conda:pyaedt2026v1", "env_profile": "pyaedt2026v1",
        "scheduling_profile": scheduling_profile,
        "aedt_backend": aedt_backend,
        "cpus": cpus, "memory_mb": mem_mb, "gpus": 0,
        "timeout_seconds": timeout_seconds,
        "dedupe_key": dedupe_key,
        # Exact per-task basename only. The scheduler applies this cleanup on
        # every terminal path, including cancellation and allocation loss.
        "cleanup_globs": scratch_leaf,
    }
    existing = reconcile_task_id(name, dedupe_key)
    if existing is not None:
        return existing
    if required_project_cap is None:
        required_hard_cap = MFT_PROJECT_MAX_ACTIVE_TASKS
        require_exact_project_cap = False
    else:
        if (isinstance(required_project_cap, bool)
                or not isinstance(required_project_cap, int)
                or not 1 <= required_project_cap
                <= MFT_PROJECT_MAX_ACTIVE_TASKS):
            raise ProjectContractError("required project cap is invalid")
        required_hard_cap = required_project_cap
        require_exact_project_cap = True
    if require_exact_project_cap:
        capacity = live_project_submission_snapshot(
            required_hard_cap,
            require_exact_project_cap=True,
            require_full_project=True,
        )
    else:
        capacity = live_project_submission_snapshot(required_hard_cap)
    if capacity["project_submission_slots"] < 1:
        raise ProjectCapacityError(
            f"MFT project has no submission slots under cap "
            f"{required_hard_cap}: {capacity}")
    try:
        r = requests.post(f"{SCHEDULER}/api/tasks", json=payload, timeout=20)
    except Exception as post_error:
        try:
            recovered = reconcile_task_id(name, dedupe_key)
        except TaskLookupError as lookup_error:
            raise TaskSubmissionUncertain(
                f"POST response and reconciliation were both unavailable for {name!r}"
            ) from lookup_error
        if recovered is not None:
            return recovered
        raise TaskSubmissionUncertain(
            f"POST response was lost and no durable task is visible yet for {name!r}"
        ) from post_error
    if r.status_code not in (200, 201):
        recovered = reconcile_task_id(name, dedupe_key)
        if recovered is not None:
            return recovered
        return None
    try:
        response_payload = r.json()
        task_id = response_payload.get("task_id") or response_payload.get("id")
        if task_id is not None:
            return int(task_id)
    except Exception:
        pass
    recovered = reconcile_task_id(name, dedupe_key)
    if recovered is not None:
        return recovered
    raise TaskSubmissionUncertain(
        f"scheduler accepted {name!r} without returning or exposing its task ID"
    )


def get_status(task_id):
    try:
        return requests.get(f"{SCHEDULER}/api/tasks/{task_id}", timeout=15).json().get("status")
    except Exception:
        return None


def cancel(task_id):
    try:
        requests.post(f"{SCHEDULER}/tasks/{task_id}/cancel", timeout=10)
    except Exception:
        pass


def cancel_queued_tasks_cas(task_ids):
    """Cancel only exact task IDs that are still queued."""
    if not campaign_mutation_lock_is_held():
        raise RuntimeError("MFT queued cancellation requires the mutation lock")
    if not isinstance(task_ids, (list, tuple)) or not task_ids:
        raise ValueError("queued cancellation requires at least one task ID")
    normalized = []
    for task_id in task_ids:
        if (isinstance(task_id, bool) or not isinstance(task_id, int)
                or task_id <= 0 or task_id in normalized):
            raise ValueError("queued cancellation task IDs are invalid/duplicate")
        normalized.append(task_id)
    try:
        response = requests.post(
            f"{SCHEDULER}/api/tasks/cancel",
            params={
                "task_ids": ",".join(map(str, normalized)),
                "statuses": "queued",
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise TaskSubmissionUncertain(
            f"queued-only cancellation response is uncertain: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProjectContractError(
            "scheduler queued cancellation acknowledgement is invalid")
    cancelled = payload.get("cancelled")
    count = payload.get("count")
    if (not isinstance(cancelled, list)
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in cancelled)
            or len(cancelled) != len(set(cancelled))
            or not set(cancelled).issubset(normalized)
            or isinstance(count, bool) or not isinstance(count, int)
            or count != len(cancelled)):
        raise ProjectContractError(
            "scheduler queued cancellation acknowledgement is inconsistent")
    return {"cancelled": sorted(cancelled), "count": count}


def _finite(result, key):
    try:
        return math.isfinite(float(result[key]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def required_temperature_columns(result):
    """Return all physical/probe temperatures required by this candidate."""
    columns = list(MANDATORY_TEMPERATURE_COLUMNS)
    try:
        if float(result["N2_side"]) > 0:
            columns.extend(SIDE_TEMPERATURE_COLUMNS)
    except (KeyError, TypeError, ValueError, OverflowError):
        return ()
    return tuple(columns)


def result_matches_params(result, params, required_keys=None):
    """Require every submitted candidate input to be echoed by the result."""
    if not isinstance(result, dict) or not isinstance(params, dict) or not params:
        return False
    if required_keys is not None and set(params) != set(required_keys):
        return False
    for key, expected in params.items():
        if key not in result:
            return False
        actual = result[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                if not math.isclose(
                        float(actual), float(expected),
                        rel_tol=1e-9, abs_tol=1e-9):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
        elif str(actual) != str(expected):
            return False
    return True


def is_valid_result(
        result, expected_revision=None, expected_library_revision=None,
        expected_profile=None):
    """Return whether a row is complete enough for AL verification ingestion."""
    if not isinstance(result, dict):
        return False
    if result.get("result_valid_em") != 1 or result.get("result_valid_thermal") != 1:
        return False
    if result.get("thermal_solved") != 1 or result.get("thermal_extraction_complete") != 1:
        return False
    if result.get("thermal_convergence_available") != 1 or result.get("thermal_converged") != 1:
        return False
    if result.get("thermal_required_missing_count") != 0:
        return False
    profile_contract = dict(
        STANDARD_PROFILE_CONTRACT if expected_profile is None else expected_profile)
    numeric_fields = (
        "Llt", "B_max_core", "full_model", "N2_side",
        "git_dirty", "pyaedt_library_git_dirty",
        "matrix_solve_attempts", "loss_solve_attempts",
        "conv_passes_matrix", "conv_consecutive_matrix",
        "conv_error_pct_matrix", "conv_delta_pct_matrix",
        "conv_passes_loss", "conv_consecutive_loss",
        "conv_error_pct_loss", "conv_delta_pct_loss",
        "matrix_winding_stranded_count",
        "matrix_conductor_mesh_operation_count",
        "matrix_plate_eddy_off_readback_count",
        "loss_winding_solid_update_count",
        "loss_winding_mesh_operation_count",
        "loss_conductor_mesh_operation_count",
        "loss_plate_eddy_on_readback_count",
        "P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total",
        "thermal_residual_flow_limit", "thermal_residual_energy_limit",
        "thermal_residual_continuity", "thermal_residual_x_velocity",
        "thermal_residual_y_velocity", "thermal_residual_z_velocity",
        "thermal_residual_energy", "thermal_iterations",
        "thermal_rx_power_balance_ok", "thermal_rx_power_balance_group_count",
        "thermal_rx_power_balance_max_abs_w", "thermal_rx_expected_power_w",
        "thermal_rx_assigned_power_w",
    ) + tuple(
        key for key, value in profile_contract.items()
        if not isinstance(value, str)
    )
    if not all(_finite(result, key) for key in numeric_fields):
        return False
    expected_full_model = float(profile_contract.get("full_model", 0))
    if (float(result["Llt"]) <= 0
            or float(result["B_max_core"]) < 0
            or float(result["N2_side"]) < 0
            or float(result["git_dirty"]) != 0.0
            or float(result["pyaedt_library_git_dirty"]) != 0.0
            or float(result["matrix_solve_attempts"]) != 1.0
            or float(result["loss_solve_attempts"]) != 1.0
            or float(result["full_model"]) != expected_full_model):
        return False
    if result.get("matrix_extraction_backend") != "export_rl_matrix":
        return False
    matrix_plate_count = float(result["matrix_plate_eddy_off_readback_count"])
    loss_plate_count = float(result["loss_plate_eddy_on_readback_count"])
    matrix_skin_mesh = float(profile_contract.get("matrix_skin_mesh", 0))
    if matrix_skin_mesh == 0.0:
        if (result.get("matrix_conductor_policy") != "stranded_no_eddy_no_skin"
                or float(result["matrix_winding_stranded_count"]) != 2.0
                or float(result["matrix_conductor_mesh_operation_count"]) != 0.0
                or float(result["loss_winding_solid_update_count"]) != 2.0
                or float(result["loss_winding_mesh_operation_count"]) != 2.0
                or matrix_plate_count < 0.0
                or loss_plate_count != matrix_plate_count
                or float(result["loss_conductor_mesh_operation_count"]) != (
                    2.0 + float(loss_plate_count > 0.0)
                )):
            return False
    elif matrix_skin_mesh == 1.0:
        # The fine path creates solid windings and skin operations in the
        # matrix design.  Its copied loss design inherits them, represented by
        # the deliberate -1 "not reconfigured" readbacks.
        if (result.get("matrix_conductor_policy") != "solid_skin"
                or float(result["matrix_winding_stranded_count"]) != 0.0
                or float(result["matrix_conductor_mesh_operation_count"]) < 2.0
                or matrix_plate_count != 0.0
                or any(float(result[key]) != -1.0 for key in (
                    "loss_winding_solid_update_count",
                    "loss_winding_mesh_operation_count",
                    "loss_conductor_mesh_operation_count",
                    "loss_plate_eddy_on_readback_count",
                ))):
            return False
    else:
        return False
    for key, expected_value in profile_contract.items():
        actual_value = result.get(key)
        if isinstance(expected_value, str):
            if str(actual_value or "").strip().lower() != expected_value.lower():
                return False
        else:
            try:
                if not math.isclose(
                        float(actual_value), float(expected_value),
                        rel_tol=1e-12, abs_tol=1e-12):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
    for label, tolerance_key, minimum_key in (
            ("matrix", "matrix_percent_error", "matrix_min_converged"),
            ("loss", "percent_error", "min_converged")):
        tolerance = float(result[tolerance_key])
        minimum_passes = float(profile_contract[minimum_key])
        total_passes = float(result[f"conv_passes_{label}"])
        consecutive = float(result[f"conv_consecutive_{label}"])
        if (not 0 < tolerance <= 1.5
                or total_passes < 1
                or consecutive < minimum_passes
                or consecutive > total_passes
                or consecutive != math.floor(consecutive)
                or not 0 <= float(result[f"conv_error_pct_{label}"]) <= tolerance
                or not 0 <= float(result[f"conv_delta_pct_{label}"]) <= tolerance):
            return False
    if not all(float(result[key]) >= 0 for key in (
            "P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total")):
        return False
    flow_limit = float(result["thermal_residual_flow_limit"])
    energy_limit = float(result["thermal_residual_energy_limit"])
    if (not 0 < flow_limit <= 1e-3
            or not 0 < energy_limit <= 1e-7
            or float(result["thermal_iterations"]) <= 0):
        return False
    if (float(result["thermal_rx_power_balance_ok"]) != 1.0
            or float(result["thermal_rx_power_balance_group_count"]) < 1.0
            or float(result["thermal_rx_expected_power_w"]) < 0.0
            or float(result["thermal_rx_assigned_power_w"]) < 0.0
            or not math.isclose(
                float(result["thermal_rx_assigned_power_w"]),
                float(result["thermal_rx_expected_power_w"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            or not 0.0 <= float(result["thermal_rx_power_balance_max_abs_w"]) <= 1e-9):
        return False
    explicit_turns = float(result.get("n_explicit_turns", -1))
    expected_rx_model = "homogenized_blocks" if explicit_turns == 0.0 else "hybrid_explicit"
    if result.get("thermal_rx_model") != expected_rx_model:
        return False
    if not all(0 <= float(result[key]) <= flow_limit for key in (
            "thermal_residual_continuity", "thermal_residual_x_velocity",
            "thermal_residual_y_velocity", "thermal_residual_z_velocity")):
        return False
    if not 0 <= float(result["thermal_residual_energy"]) <= energy_limit:
        return False
    git_hash = str(result.get("git_hash") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", git_hash):
        return False
    if expected_revision is not None:
        expected = str(expected_revision).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", expected) or expected != git_hash:
            return False
    library_hash = str(result.get("pyaedt_library_git_hash") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", library_hash):
        return False
    if expected_library_revision is not None:
        expected_library = str(expected_library_revision).strip().lower()
        if (not re.fullmatch(r"[0-9a-f]{40}", expected_library)
                or library_hash != expected_library):
            return False
    if not str(result.get("project_name") or "").strip() or not str(result.get("saved_at") or "").strip():
        return False
    try:
        expected_mask = 15 if float(result["N2_side"]) > 0 else 11
        if int(float(result["thermal_required_group_mask"])) != expected_mask:
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    temperatures = required_temperature_columns(result)
    return bool(temperatures) and all(
        _finite(result, key)
        and MIN_TRUSTED_TEMPERATURE_C < float(result[key]) < MAX_TRUSTED_TEMPERATURE_C
        for key in temperatures
    )


def fetch_result(
        task_id, attempts=3, retry_delay=2, expected_revision=None,
        expected_library_revision=None, expected_profile=None):
    """Return the latest well-formed RESULT_JSON and its validity state.

    Transport failures raise ResultFetchError. A successful stdout read with no
    JSON row is RESULT_MISSING, while a well-formed but incomplete row is
    RESULT_INVALID. Only the latest well-formed row is authoritative.
    """
    out = None
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                f"{SCHEDULER}/api/tasks/{task_id}/stdout",
                params={"max_bytes": MAX_STDOUT_BYTES}, timeout=30)
            response.raise_for_status()
            out = response.text
            break
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(retry_delay)
    if out is None:
        raise ResultFetchError(
            f"failed to fetch stdout for task {task_id} after {attempts} attempts: {last_error}"
        ) from last_error

    library_hash = None
    for line in reversed(out.splitlines()):
        if line.startswith("MFT_LIBRARY_GIT_HASH "):
            candidate = line[len("MFT_LIBRARY_GIT_HASH "):].strip().lower()
            if re.fullmatch(r"[0-9a-f]{40}", candidate):
                library_hash = candidate
                break
    if expected_library_revision is not None:
        expected_library = str(expected_library_revision).strip().lower()
        if (not re.fullmatch(r"[0-9a-f]{40}", expected_library)
                or library_hash != expected_library):
            return ResultFetch(RESULT_INVALID)

    for line in reversed(out.splitlines()):
        if not line.startswith("RESULT_JSON "):
            continue
        try:
            result = json.loads(line[len("RESULT_JSON "):])
        except Exception:
            continue
        if isinstance(result, dict) and library_hash:
            result_library_hash = str(
                result.get("pyaedt_library_git_hash") or "").strip().lower()
            if result_library_hash != library_hash:
                return ResultFetch(RESULT_INVALID, result)
        if is_valid_result(
                result, expected_revision=expected_revision,
                expected_library_revision=expected_library_revision,
                expected_profile=expected_profile):
            return ResultFetch(RESULT_VALID, result)
        return ResultFetch(RESULT_INVALID, result if isinstance(result, dict) else None)
    return ResultFetch(RESULT_MISSING)


def fetch_result_json(task_id):
    """Compatibility wrapper returning only a valid verification row."""
    try:
        fetched = fetch_result(task_id)
    except ResultFetchError:
        return None
    return fetched.result if fetched.state == RESULT_VALID else None


def wait_all(task_ids, poll_s=120, timeout_s=6 * 3600, on_progress=None):
    """태스크 집합 완료 대기. 반환: {task_id: status}"""
    t0 = time.time()
    status = {tid: None for tid in task_ids}
    while time.time() - t0 < timeout_s:
        pending = [tid for tid, s in status.items()
                   if s not in ("completed", "failed", "cancelled")]
        if not pending:
            break
        for tid in pending:
            s = get_status(tid)
            if s:
                status[tid] = s
        if on_progress:
            on_progress(status)
        if all(s in ("completed", "failed", "cancelled") for s in status.values()):
            break
        time.sleep(poll_s)
    return status
