"""
AL 검증용 스케줄러 클라이언트: 후보 params JSON -> 태스크 제출 -> RESULT_JSON 회수.
"""
import hashlib
import json
import math
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

import requests

SCHEDULER = "http://127.0.0.1:8000"
BASE = ("source /etc/profile.d/lmod.sh 2>/dev/null || true; "
        "module load ansys-electronics/v252 || export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64; "
        "export FLEXLM_TIMEOUT=3000000; "
        "export I_MPI_HYDRA_BOOTSTRAP=fork; sleep $((RANDOM % 60)); ")

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


@dataclass(frozen=True)
class ResultFetch:
    state: str
    result: dict | None = None


def reconcile_task_id(name, dedupe_key, attempts=3, retry_delay=1):
    """Find the newest exact task identity, including terminal tasks."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                f"{SCHEDULER}/api/tasks",
                params={"limit": 10000, "name_prefix": name},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
            if not isinstance(tasks, list):
                raise ValueError("task inventory is not a list")
            matches = [
                task for task in tasks
                if task.get("name") == name and task.get("dedupe_key") == dedupe_key
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


def submit_verification(
        name, workdir, params: dict, profile: dict, mem_mb=32768, cpus=4,
        solver_revision=None, library_revision=None):
    """후보 파라미터를 인라인 JSON으로 실어 fixed 모드 검증 태스크 제출. 반환: task_id 또는 None"""
    if not isinstance(solver_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", solver_revision):
        raise ValueError("solver_revision must be a full 40-character git SHA")
    solver_revision = solver_revision.lower()
    if not isinstance(library_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", library_revision):
        raise ValueError("library_revision must be a full 40-character git SHA")
    library_revision = library_revision.lower()
    merged = dict(params)
    merged.update(profile.get("param_overrides", {}))
    timeout_seconds = int(profile.get(
        "timeout_seconds", DEFAULT_TASK_TIMEOUT_SECONDS))
    if timeout_seconds <= 0:
        raise ValueError("verification timeout_seconds must be positive")
    pjson = json.dumps(merged, separators=(",", ":"))
    parameter_digest = hashlib.sha256(pjson.encode("utf-8")).hexdigest()[:16]
    dedupe_key = (
        f"mft-al:{name}:{solver_revision}:{library_revision}:{parameter_digest}")
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
        f"mkdir -p {quoted_workdir} && "
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
        "name": name, "remote_cwd": GPFS_RUNS_REMOTE_CWD,
        "command": cmd, "required_capability": "conda:pyaedt2026v1", "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty", "cpus": cpus, "memory_mb": mem_mb, "gpus": 0,
        "timeout_seconds": timeout_seconds,
        "dedupe_key": dedupe_key,
        # Exact per-task basename only. The scheduler applies this cleanup on
        # every terminal path, including cancellation and allocation loss.
        "cleanup_globs": scratch_leaf,
    }
    existing = reconcile_task_id(name, dedupe_key)
    if existing is not None:
        return existing
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


def result_matches_params(result, params):
    """Require every submitted candidate input to be echoed by the result."""
    if not isinstance(result, dict) or not isinstance(params, dict) or not params:
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
