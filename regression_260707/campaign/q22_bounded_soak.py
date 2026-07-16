"""Policy-driven, crash-safe q22 pooled production controller.

This controller deliberately separates two operator controls:

* campaign demand is the total number of accepted q22 submissions (500 by
  default, but versioned and adjustable in the Web UI); and
* simulation policy is the number of logical clients allowed to be active at
  once (never more than 30 for the validated 10 AEDT x 3 project pool).

The launch identity (solver, library, package, profile, candidate seed and
baseline serial) is immutable.  Demand is not: lowering it stops refill
without cancelling work, while raising it extends the same deterministic
candidate stream.  ``feeder.max_new_tasks`` is the final per-cycle guard, so a
restart or an uncertain scheduler POST cannot exceed the observed demand.

The default mode is a write-free plan. Execution requires the deliberately
verbose ``--execute-mft-family-production`` switch. MFT remains family-isolated
from IPMSM, so this campaign does not wait for or mutate mixed-canary state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import requests
from filelock import FileLock, Timeout as FileLockTimeout


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
REPO_ROOT = REGRESSION_ROOT.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for path in (str(HERE), str(REGRESSION_ROOT), str(VERIFY_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import deployment_gate
import feeder
from module.core_material_contract import PHYSICS_DATA_REVISION
from regression_260707 import quality_contract


SCHEMA = "q22-bounded-soak-controller-v1"
ACCOUNT_EXPANSION_SCHEMA = "q22-bounded-soak-controller-account-expansion-v2"
CAMPAIGN_ID = "q22-bounded-soak500-260716"
PROJECT = "MFT_1MW_2026v1"
DEFAULT_TOTAL_DEMAND = 500
ADOPTED_BASELINE_SERIAL = 17113
ADOPTED_BASELINE_DATASET_ROWS = 5233
MAX_TOTAL_DEMAND = 100_000
MAX_LOGICAL_ACTIVE = 30
EXPECTED_POOL_SESSIONS = 10
EXPECTED_PROJECTS_PER_AEDT = 3
EXPECTED_POOL_PROJECTS = 30
TASK_TIMEOUT_SECONDS = 86_400
RELEASE_TIMEOUT_SECONDS = 7_200
AUTOMATION_TIMEOUT_SECONDS = 7_200
NATIVE_BARRIER_TIMEOUT_SECONDS = 7_200
CANDIDATE_SEED = 260710

EXISTING_COHORT_SOLVER = "26afff8de2936f605783395fbff19d5f1d26b354"
PROVEN_RUNTIME_SOLVER = "c7a0c792e2babc74ad1596a6b95b45379a6f903d"
CAMPAIGN_SOLVER = "092a35bb6e9552fa9c0ef7388c6059606844f2cd"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
SCHEDULER_PACKAGE_REVISION = "9150e7fa7f72fdf00fb8113e157398b410833c40"
PHYSICS_REVISION = "mft1mw-1k101-native-lamination-kf0p85-v3"
Q21B_TASK_IDS = (41796, 41797, 41798)
Q21B_SESSION_ID = 536

DEFAULT_STATE_DIR = Path(
    r"Y:\git\MFT_1MW_2026\regression_260707\campaign"
)
DEFAULT_DATASET_DIR = Path(
    r"Y:\git\MFT_solver_pooled_260714\regression_260707\data\dataset"
)
DEFAULT_LIBRARY_ROOT = Path(r"Y:\git\pyaedt_library_release_e6b9_260715")
DEFAULT_DEPLOYMENT_SOLVER_ROOT = Path(r"Y:\git\MFT_1MW_2026")
DEFAULT_SCHEDULER_DB = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\data\slurm_scheduler.db"
)
DEFAULT_ACCOUNTS_CONFIG = Path(
    r"Y:\runtime\slurm_scheduler\config\accounts.yaml"
)
DEFAULT_SSH_AUDIT_PYTHON = Path(r"C:\Python314\python.exe")
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8001"
DEFAULT_POOL_URL = "http://172.16.10.37:18790"
DEFAULT_ELIGIBLE_ACCOUNTS = ("dhj02", "harry261", "jji0930")
PROFILE_PATH = VERIFY_ROOT / "profiles" / "q22_bounded_full.json"
COMPATIBILITY_PATH = HERE / "q22_physics_compatibility.json"

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
ACTIVE_TASK_STATES = ("queued", "attaching", "running")
LIVE_LEASE_STATES = ("offered", "leased", "attaching", "active", "releasing")
LIVE_MIXED_ADMISSION_STATES = ("open", "filled", "aborting")


class GateError(RuntimeError):
    """A production prerequisite is absent or has drifted."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON object required: {path}")
    return value


def _state_serial(path: Path) -> int:
    state = _read_json(path)
    serial = state.get("serial")
    if isinstance(serial, bool) or not isinstance(serial, int) or serial < 0:
        raise GateError(f"invalid feeder serial in {path}")
    return serial


def _git(repo: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
        check=False,
    )
    if check and result.returncode:
        raise GateError(
            f"git {' '.join(arguments)} failed in {repo}: {result.stdout.strip()}"
        )
    return result.stdout.strip()


def verify_profile(path: Path = PROFILE_PATH) -> dict[str, Any]:
    profile = _read_json(path)
    overrides = profile.get("param_overrides")
    if not isinstance(overrides, dict):
        raise GateError("q22 profile has no param_overrides object")
    required = {
        "full_model": 0,
        "matrix_on": 1,
        "cap_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
    }
    drift = {key: overrides.get(key) for key, expected in required.items()
             if overrides.get(key) != expected}
    if drift:
        raise GateError(f"q22 full extraction profile drifted: {drift}")
    if profile.get("timeout_seconds") != TASK_TIMEOUT_SECONDS:
        raise GateError("q22 task timeout must be exactly 86400 seconds")
    if profile.get("cpus") != 1 or profile.get("mem_mb") != 6144:
        raise GateError("q22 pooled profile must request 1 CPU and 6144 MiB")
    return profile


def verify_compatibility(
    repo_root: Path = REPO_ROOT,
    manifest_path: Path = COMPATIBILITY_PATH,
) -> dict[str, Any]:
    evidence = _read_json(manifest_path)
    pins = evidence.get("pins")
    expected_pins = {
        "existing_training_cohort_solver_revision": EXISTING_COHORT_SOLVER,
        "proven_runtime_solver_revision": PROVEN_RUNTIME_SOLVER,
        "campaign_solver_revision": CAMPAIGN_SOLVER,
        "pyaedt_library_revision": LIBRARY_REVISION,
        "scheduler_package_commit": SCHEDULER_PACKAGE_REVISION,
        "physics_data_revision": PHYSICS_REVISION,
    }
    if pins != expected_pins:
        raise GateError("q22 physics compatibility pins drifted")
    if PHYSICS_DATA_REVISION != PHYSICS_REVISION:
        raise GateError("runtime PHYSICS_DATA_REVISION drifted")
    approved = quality_contract.PHYSICS_EQUIVALENT_SOLVER_REVISIONS.get(
        EXISTING_COHORT_SOLVER, frozenset()
    )
    if CAMPAIGN_SOLVER not in approved or PROVEN_RUNTIME_SOLVER not in approved:
        raise GateError("exact q22 solver revisions are not quality-contract approved")
    for ancestor, descendant in (
        (EXISTING_COHORT_SOLVER, PROVEN_RUNTIME_SOLVER),
        (PROVEN_RUNTIME_SOLVER, CAMPAIGN_SOLVER),
    ):
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root.as_posix()}", "-C",
             str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
            check=False,
        )
        if result.returncode:
            raise GateError(f"required ancestry failed: {ancestor} -> {descendant}")
    attestation = evidence.get("runtime_surface_attestation") or {}
    objects = attestation.get("required_identical_objects") or []
    if not objects:
        raise GateError("compatibility manifest has no runtime object attestation")
    for item in objects:
        path = str(item.get("path") or "")
        base = _git(repo_root, "rev-parse", f"{PROVEN_RUNTIME_SOLVER}:{path}")
        candidate = _git(repo_root, "rev-parse", f"{CAMPAIGN_SOLVER}:{path}")
        if base != candidate:
            raise GateError(f"runtime surface changed between proven and campaign SHA: {path}")
        if base != item.get("base_object_sha1") or candidate != item.get(
            "candidate_object_sha1"
        ):
            raise GateError(f"runtime object attestation drifted: {path}")
    return evidence


def verify_local_library(path: Path) -> None:
    if _git(path, "rev-parse", "HEAD") != LIBRARY_REVISION:
        raise GateError("local PyAEDT library checkout is not the pinned revision")
    if _git(path, "status", "--porcelain", "--untracked-files=all"):
        raise GateError("local PyAEDT library checkout is dirty")


def _http_json(base_url: str, path: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=30)
        response.raise_for_status()
        value = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise GateError(f"scheduler GET {path} failed: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"scheduler GET {path} returned a non-object")
    return value


def read_campaign_demand(base_url: str) -> dict[str, Any]:
    demand = _http_json(base_url, f"/api/projects/{PROJECT}/campaign-demand")
    total = demand.get("total_simulations")
    revision = demand.get("demand_revision")
    if (isinstance(total, bool) or not isinstance(total, int)
            or not 0 <= total <= MAX_TOTAL_DEMAND):
        raise GateError("campaign total_simulations is invalid")
    if (isinstance(revision, bool) or not isinstance(revision, (int, str))
            or not str(revision).strip()):
        raise GateError("campaign demand_revision is invalid")
    return demand


def verify_pool_and_policy(base_url: str) -> tuple[dict[str, Any], int]:
    summary = _http_json(base_url, "/api/aedt-pool")
    config = summary.get("config")
    if not isinstance(config, dict):
        raise GateError("AEDT pool summary has no config")
    required = {
        "max_aedt_sessions": EXPECTED_POOL_SESSIONS,
        "projects_per_aedt": EXPECTED_PROJECTS_PER_AEDT,
        "target_project_concurrency": EXPECTED_POOL_PROJECTS,
        "enabled": True,
        "adapter_ready": True,
        "validation_passed": True,
        "operational": True,
    }
    drift = {key: config.get(key) for key, value in required.items()
             if config.get(key) != value}
    if drift:
        raise GateError(f"AEDT pool must remain exact 10x3/30 and operational: {drift}")
    project = _http_json(base_url, f"/api/projects/{PROJECT}")
    embedded = project.get("simulation_policy")
    policy = {**project, **embedded} if isinstance(embedded, dict) else project
    if project.get("max_active_tasks") != 500:
        raise GateError("MFT scheduler project max_active_tasks must remain 500")
    desired = policy.get("desired_simulations")
    validated = policy.get("validated_concurrency_limit")
    effective = policy.get("effective_simulations")
    if (type(desired) is not int or not 0 <= desired <= MAX_LOGICAL_ACTIVE
            or validated != MAX_LOGICAL_ACTIVE):
        raise GateError("active policy must be within the validated 0..30 range")
    if effective is None:
        effective = desired
    if type(effective) is not int or not 0 <= effective <= desired:
        raise GateError("effective simulation policy is invalid")
    if str(policy.get("scale_down_mode") or "").lower() != "drain":
        raise GateError("simulation policy scale-down must be drain")
    return summary, min(desired, effective, MAX_LOGICAL_ACTIVE)


def _connect_readonly(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise GateError(f"cannot open scheduler DB read-only: {exc}") from exc
    connection.row_factory = sqlite3.Row
    return connection


def verify_scheduler_evidence(db_path: Path) -> dict[str, Any]:
    placeholders = ",".join("?" for _ in Q21B_TASK_IDS)
    try:
        with _connect_readonly(db_path) as connection:
            tasks = connection.execute(
                f"SELECT id, status, exit_code, command, timeout_seconds, aedt_backend "
                f"FROM tasks WHERE id IN ({placeholders}) ORDER BY id",
                Q21B_TASK_IDS,
            ).fetchall()
            leases = connection.execute(
                f"SELECT task_id, session_id, state, native_pipeline_completed_at, "
                f"finished_at FROM aedt_project_leases "
                f"WHERE task_id IN ({placeholders}) ORDER BY task_id",
                Q21B_TASK_IDS,
            ).fetchall()
    except sqlite3.Error as exc:
        raise GateError(f"scheduler evidence query failed: {exc}") from exc
    if [int(row["id"]) for row in tasks] != list(Q21B_TASK_IDS):
        raise GateError("q21b task evidence is incomplete")
    for row in tasks:
        command = str(row["command"] or "")
        if (row["status"] != "completed" or row["exit_code"] != 0
                or row["timeout_seconds"] != TASK_TIMEOUT_SECONDS
                or row["aedt_backend"] != "pooled"
                or PROVEN_RUNTIME_SOLVER not in command
                or LIBRARY_REVISION not in command):
            raise GateError(f"q21b task {row['id']} no longer proves the pinned 1x3 run")
    if len(leases) != 3:
        raise GateError("q21b lease evidence is incomplete")
    for row in leases:
        if (row["session_id"] != Q21B_SESSION_ID or row["state"] != "released"
                or not str(row["native_pipeline_completed_at"] or "").strip()
                or not str(row["finished_at"] or "").strip()):
            raise GateError(f"q21b lease for task {row['task_id']} is not settled")
    return {
        "q21b_tasks": list(Q21B_TASK_IDS),
        "q21b_session": Q21B_SESSION_ID,
        "mixed_canary_gate": "not-required-family-isolation",
    }


def _load_accounts_config(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import yaml
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError) as exc:
        raise GateError(f"cannot load scheduler accounts config: {exc}") from exc
    rows = value.get("accounts") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise GateError("scheduler accounts config has no accounts list")
    return {str(row.get("name") or ""): row for row in rows if isinstance(row, dict)}


def audit_remote_packages(
    config_path: Path,
    eligible_accounts: Sequence[str],
    audit_python: Path = DEFAULT_SSH_AUDIT_PYTHON,
) -> list[dict[str, Any]]:
    try:
        import paramiko
    except ImportError as exc:
        if not audit_python.is_file():
            raise GateError(
                f"paramiko is unavailable and audit Python is missing: {audit_python}"
            ) from exc
        command = [
            str(audit_python),
            str(HERE / "q22_remote_package_audit.py"),
            "--accounts-config", str(config_path),
            "--expected", SCHEDULER_PACKAGE_REVISION,
        ]
        for account in eligible_accounts:
            command.extend(["--account", account])
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(90, 60 * len(eligible_accounts)),
            check=False,
        )
        if result.returncode:
            raise GateError(
                f"remote AEDT package audit helper failed: {result.stdout.strip()}"
            )
        try:
            payload = json.loads(result.stdout)
        except ValueError as error:
            raise GateError("remote package audit helper returned invalid JSON") from error
        if not isinstance(payload, list) or len(payload) != len(eligible_accounts):
            raise GateError("remote package audit helper returned incomplete evidence")
        return payload
    configured = _load_accounts_config(config_path)
    results = []
    remote_command_prefix = (
        "set -eu; root=\"$HOME/slurm_scheduler/aedt_pool_pkg\"; "
        f"test \"$(git -C \"$root\" rev-parse HEAD)\" = \"{SCHEDULER_PACKAGE_REVISION}\"; "
        "git -C \"$root\" diff --quiet HEAD --; "
        "status=$(git -C \"$root\" status --porcelain --untracked-files=all "
        "| grep -Ev '^\\?\\? batch\\.log$' || true); test -z \"$status\"; "
    )
    remote_command_suffix = (
        "; PYTHONPATH=\"$root\" python -c \"from slurm_scheduler.aedt_attach_client "
        "import AedtProjectLease; assert hasattr(AedtProjectLease, "
        "'wait_for_native_pipeline_barrier')\"; "
        f"printf '%s\\n' {SCHEDULER_PACKAGE_REVISION}"
    )
    for account_name in eligible_accounts:
        account = configured.get(account_name)
        if not account:
            raise GateError(f"eligible account is absent from accounts.yaml: {account_name}")
        capabilities = account.get("capabilities") or []
        if "conda:pyaedt2026v1" not in capabilities:
            raise GateError(f"eligible account lacks pyaedt2026v1: {account_name}")
        profiles = account.get("env_profiles") or {}
        env_setup = str(profiles.get("pyaedt2026v1") or "").strip()
        if not env_setup:
            raise GateError(
                f"eligible account has no pyaedt2026v1 setup: {account_name}"
            )
        remote_command = remote_command_prefix + env_setup + remote_command_suffix
        key = Path(str(account.get("private_key_path") or ""))
        if not key.is_file():
            raise GateError(f"SSH key is missing for {account_name}: {key}")
        host = str(account.get("host") or "").strip()
        username = str(account.get("username") or "").strip()
        port = int(account.get("port") or 22)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=host,
                port=port,
                username=username,
                key_filename=str(key),
                look_for_keys=False,
                allow_agent=False,
                timeout=15,
                banner_timeout=15,
                auth_timeout=15,
            )
            _stdin, stdout, stderr = client.exec_command(remote_command, timeout=45)
            output = stdout.read().decode("utf-8", errors="replace").strip()
            error = stderr.read().decode("utf-8", errors="replace").strip()
            return_code = stdout.channel.recv_exit_status()
        except Exception as exc:
            raise GateError(
                f"remote AEDT package audit connection failed for "
                f"{account_name}: {exc}"
            ) from exc
        finally:
            client.close()
        if return_code or output.splitlines()[-1:] != [SCHEDULER_PACKAGE_REVISION]:
            raise GateError(
                f"remote AEDT package audit failed for {account_name}: "
                f"{error or output}"
            )
        results.append({"account": account_name, "package": SCHEDULER_PACKAGE_REVISION})
    return results


def manifest_identity(
    baseline_serial: int,
    state_path: Path,
    eligible_accounts: Sequence[str],
) -> dict[str, Any]:
    profile = verify_profile()
    return {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "baseline_serial": int(baseline_serial),
        "state_path": str(state_path.resolve()),
        "candidate_seed": CANDIDATE_SEED,
        "solver_revision": CAMPAIGN_SOLVER,
        "proven_runtime_solver_revision": PROVEN_RUNTIME_SOLVER,
        "library_revision": LIBRARY_REVISION,
        "scheduler_package_revision": SCHEDULER_PACKAGE_REVISION,
        "physics_data_revision": PHYSICS_REVISION,
        "profile_sha256": _digest(profile),
        "eligible_accounts": list(eligible_accounts),
        "max_logical_active": MAX_LOGICAL_ACTIVE,
        "pool_topology": {"sessions": 10, "projects_per_aedt": 3, "projects": 30},
        "timeouts": {
            "task": TASK_TIMEOUT_SECONDS,
            "release": RELEASE_TIMEOUT_SECONDS,
            "automation": AUTOMATION_TIMEOUT_SECONDS,
            "native_barrier": NATIVE_BARRIER_TIMEOUT_SECONDS,
        },
        "adoption": {
            "prelaunch_serial": ADOPTED_BASELINE_SERIAL,
            "prelaunch_dataset_rows": ADOPTED_BASELINE_DATASET_ROWS,
            "legacy_feeder_max_samples": 5733,
            "semantics": "adopt-existing-q22-submissions-no-second-plus500",
        },
    }


def build_manifest(identity: Mapping[str, Any]) -> dict[str, Any]:
    immutable = dict(identity)
    return {
        **immutable,
        "identity_sha256": _digest(immutable),
        "created_at_epoch": time.time(),
        "demand_control": {
            "endpoint": f"/api/projects/{PROJECT}/campaign-demand",
            "default_total_simulations": DEFAULT_TOTAL_DEMAND,
            "mutable_with_cas": True,
            "decrease_semantics": "stop-refill-no-cancel",
            "increase_semantics": "extend-deterministic-stream",
        },
    }


def manifest_path_for_version(state_dir: Path, version: int) -> Path:
    if version == 1:
        return state_dir / f"{CAMPAIGN_ID}.manifest.json"
    if version == 2:
        return state_dir / f"{CAMPAIGN_ID}.manifest.v2.json"
    raise GateError("controller manifest version must be 1 or 2")


def _ordered_strict_superset(
    previous: Sequence[str], expanded: Sequence[str]
) -> bool:
    """Require account expansion to append accounts without reordering old pins."""
    return (
        len(expanded) > len(previous)
        and list(expanded[:len(previous)]) == list(previous)
        and len(set(expanded)) == len(expanded)
    )


def account_expansion_identity(
    previous_manifest: Mapping[str, Any],
    state_path: Path,
    eligible_accounts: Sequence[str],
    transition_serial: int,
) -> dict[str, Any]:
    previous_accounts = previous_manifest.get("eligible_accounts")
    if not isinstance(previous_accounts, list) or not all(
        isinstance(item, str) and item for item in previous_accounts
    ):
        raise GateError("predecessor manifest account set is invalid")
    if not _ordered_strict_superset(previous_accounts, eligible_accounts):
        raise GateError(
            "v2 eligible accounts must append a strict superset of v1 accounts"
        )
    baseline = int(previous_manifest["baseline_serial"])
    if transition_serial < baseline:
        raise GateError("account expansion transition serial precedes campaign baseline")
    identity = manifest_identity(baseline, state_path, eligible_accounts)
    identity.update({
        "schema": ACCOUNT_EXPANSION_SCHEMA,
        "manifest_version": 2,
        "transition": {
            "kind": "append-only-account-superset",
            "predecessor_schema": previous_manifest["schema"],
            "predecessor_identity_sha256": previous_manifest["identity_sha256"],
            "predecessor_eligible_accounts": list(previous_accounts),
            "transition_serial": int(transition_serial),
            "baseline_and_demand_semantics": (
                "same-baseline-and-campaign-demand-no-resubmission"
            ),
        },
    })
    return identity


def validate_manifest(
    manifest: Mapping[str, Any],
    state_path: Path,
    eligible_accounts: Sequence[str],
) -> dict[str, Any]:
    baseline = manifest.get("baseline_serial")
    if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < 0:
        raise GateError("controller manifest baseline_serial is invalid")
    expected = manifest_identity(baseline, state_path, eligible_accounts)
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected or manifest.get("identity_sha256") != _digest(expected):
        raise GateError("controller manifest immutable identity drifted")
    return dict(manifest)


def validate_account_expansion_manifest(
    manifest: Mapping[str, Any],
    previous_manifest: Mapping[str, Any],
    state_path: Path,
    eligible_accounts: Sequence[str],
) -> dict[str, Any]:
    transition = manifest.get("transition")
    transition_serial = transition.get("transition_serial") if isinstance(
        transition, dict
    ) else None
    if (isinstance(transition_serial, bool)
            or not isinstance(transition_serial, int)):
        raise GateError("v2 manifest transition_serial is invalid")
    expected = account_expansion_identity(
        previous_manifest,
        state_path,
        eligible_accounts,
        transition_serial,
    )
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected or manifest.get("identity_sha256") != _digest(expected):
        raise GateError("v2 controller manifest immutable identity drifted")
    return dict(manifest)


def load_or_create_account_expansion_manifest(
    path: Path,
    predecessor_path: Path,
    state_path: Path,
    eligible_accounts: Sequence[str],
    *,
    execute: bool,
    baseline_serial: int | None = None,
) -> dict[str, Any]:
    if not predecessor_path.is_file():
        raise GateError(f"v2 predecessor manifest is missing: {predecessor_path}")
    raw_previous = _read_json(predecessor_path)
    previous_accounts = raw_previous.get("eligible_accounts")
    if not isinstance(previous_accounts, list):
        raise GateError("v2 predecessor eligible_accounts is invalid")
    previous = validate_manifest(raw_previous, state_path, previous_accounts)
    if (baseline_serial is not None
            and int(previous["baseline_serial"]) != int(baseline_serial)):
        raise GateError("v2 predecessor does not match adopted baseline serial")
    if path.exists():
        return validate_account_expansion_manifest(
            _read_json(path), previous, state_path, eligible_accounts
        )
    transition_serial = _state_serial(state_path)
    identity = account_expansion_identity(
        previous, state_path, eligible_accounts, transition_serial
    )
    manifest = build_manifest(identity)
    if not execute:
        return manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return validate_account_expansion_manifest(
            _read_json(path), previous, state_path, eligible_accounts
        )
    return manifest


def load_manifest_version(
    path: Path,
    predecessor_path: Path,
    state_path: Path,
    eligible_accounts: Sequence[str],
    *,
    version: int,
    execute: bool,
    baseline_serial: int | None = None,
) -> dict[str, Any]:
    if version == 1:
        return load_or_create_manifest(
            path,
            state_path,
            eligible_accounts,
            execute=execute,
            baseline_serial=baseline_serial,
        )
    if version == 2:
        return load_or_create_account_expansion_manifest(
            path,
            predecessor_path,
            state_path,
            eligible_accounts,
            execute=execute,
            baseline_serial=baseline_serial,
        )
    raise GateError("controller manifest version must be 1 or 2")


def load_or_create_manifest(
    path: Path,
    state_path: Path,
    eligible_accounts: Sequence[str],
    *,
    execute: bool,
    baseline_serial: int | None = None,
) -> dict[str, Any]:
    if path.exists():
        manifest = validate_manifest(_read_json(path), state_path, eligible_accounts)
        if (baseline_serial is not None
                and int(manifest["baseline_serial"]) != int(baseline_serial)):
            raise GateError("existing manifest does not match adopted baseline serial")
        return manifest
    current_serial = _state_serial(state_path)
    baseline = current_serial if baseline_serial is None else int(baseline_serial)
    if baseline < 0 or baseline > current_serial:
        raise GateError(
            f"adopted baseline serial {baseline} is outside 0..{current_serial}"
        )
    manifest = build_manifest(
        manifest_identity(baseline, state_path, eligible_accounts)
    )
    if not execute:
        return manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        manifest = validate_manifest(_read_json(path), state_path, eligible_accounts)
        if (baseline_serial is not None
                and int(manifest["baseline_serial"]) != int(baseline_serial)):
            raise GateError("racing manifest does not match adopted baseline serial")
        return manifest
    return manifest


def campaign_progress(
    manifest: Mapping[str, Any],
    current_serial: int,
    total_demand: int,
) -> dict[str, int]:
    baseline = int(manifest["baseline_serial"])
    if current_serial < baseline:
        raise GateError("feeder serial regressed below the campaign baseline")
    accepted = current_serial - baseline
    remaining = max(0, int(total_demand) - accepted)
    return {
        "baseline_serial": baseline,
        "current_serial": current_serial,
        "accepted_simulations": accepted,
        "total_simulations": int(total_demand),
        "remaining_simulations": remaining,
        "oversupplied_by": max(0, accepted - int(total_demand)),
    }


def verify_owned_serials(
    db_path: Path,
    manifest: Mapping[str, Any],
    current_serial: int,
) -> None:
    baseline = int(manifest["baseline_serial"])
    if current_serial == baseline:
        return
    prefix = f"mft-camp-s{CAMPAIGN_SOLVER[:7]}-l{LIBRARY_REVISION[:7]}-"
    try:
        with _connect_readonly(db_path) as connection:
            rows = connection.execute(
                "SELECT name, dedupe_key FROM tasks WHERE name LIKE ?",
                (prefix + "%",),
            ).fetchall()
    except sqlite3.Error as exc:
        raise GateError(f"campaign ownership query failed: {exc}") from exc
    by_serial = {}
    for row in rows:
        name = str(row["name"] or "")
        suffix = name[len(prefix):]
        if suffix.isdecimal():
            by_serial[int(suffix)] = row
    for serial in range(baseline + 1, current_serial + 1):
        row = by_serial.get(serial)
        dedupe = str(row["dedupe_key"] or "") if row else ""
        if (row is None or CAMPAIGN_SOLVER not in dedupe
                or LIBRARY_REVISION not in dedupe):
            raise GateError(
                f"feeder serial {serial} is not owned by the exact q22 pins"
            )


def configure_feeder(args: argparse.Namespace) -> Path:
    state_path = args.state_dir / "feeder_state.json"
    feeder.STATE = str(state_path)
    feeder.CONTROLLER_LOCK = str(args.state_dir / "feeder-controller.lock")
    feeder.TRAIN_PARQUET = str(args.dataset_dir / "train.parquet")
    feeder.COLLECT_CACHE = str(args.dataset_dir / "collect_cache.json")
    feeder.PROFILE_PATH = str(PROFILE_PATH)
    feeder.SCHEDULER = args.scheduler_url.rstrip("/")
    feeder.scheduler_client.SCHEDULER = feeder.SCHEDULER
    return state_path


def pooled_submission(args: argparse.Namespace) -> dict[str, Any]:
    environment = {
        "MFT_AEDT_BACKEND": "pooled",
        "MFT_AEDT_SHARED_CANARY": "1",
        "MFT_AEDT_SCHEDULER_URL": args.pool_url,
        "MFT_SLURM_SCHEDULER_ROOT": "$HOME/slurm_scheduler/aedt_pool_pkg",
        "SLURM_AEDT_POOL_CLIENT_TOKEN_FILE": "$HOME/slurm_scheduler/aedt_pool_client",
        "MFT_AEDT_POOL_WORKSPACE": "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}",
        "MFT_AEDT_WORKSPACE_PATH": "/gpfs/tmp_cpu2/mft_pool/mft-${SLURM_SCHED_TASK_ID}",
        "MFT_AEDT_SESSION_VERSION": "2025.2",
        "MFT_AEDT_SESSION_PROFILE": feeder.AEDT_SESSION_PROFILE,
        "MFT_AEDT_ISOLATION_POLICY": "family",
        "AEDT_POOL_AUTOMATION_LOCK_TIMEOUT_SECONDS": str(AUTOMATION_TIMEOUT_SECONDS),
        "AEDT_POOL_NATIVE_PIPELINE_BARRIER_TIMEOUT_SECONDS": str(
            NATIVE_BARRIER_TIMEOUT_SECONDS
        ),
        "MFT_AEDT_RELEASE_WAIT_SECONDS": str(RELEASE_TIMEOUT_SECONDS),
        "MFT_AEDT_POOLED_SOLVE_TIMEOUT_SECONDS": "7200",
        "MFT_AEDT_POOL_FILL_TIMEOUT_SECONDS": "900",
        "MFT_CAMPAIGN_ID": CAMPAIGN_ID,
        "MFT_CAMPAIGN_PHYSICS_DATA_REVISION": PHYSICS_REVISION,
        "MFT_CAMPAIGN_SCHEDULER_PACKAGE_REVISION": SCHEDULER_PACKAGE_REVISION,
    }
    return {
        "cpus": 1,
        "memory_mb": 6144,
        "timeout_seconds": TASK_TIMEOUT_SECONDS,
        "profile_path": str(PROFILE_PATH),
        "aedt_backend": "pooled",
        "submission_env": environment,
        "account_names": tuple(args.eligible_accounts),
    }


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with staged.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, path)


def static_plan(args: argparse.Namespace, manifest: Mapping[str, Any]) -> dict[str, Any]:
    progress = campaign_progress(
        manifest,
        _state_serial(args.state_dir / "feeder_state.json"),
        args.planned_total_demand,
    )
    submission = pooled_submission(args)
    return {
        "mode": "write-free-dry-run",
        "campaign": CAMPAIGN_ID,
        "demand": progress,
        "active_control": {
            "source": "versioned scheduler simulation-policy/Web UI",
            "range": [0, MAX_LOGICAL_ACTIVE],
            "pool": "10 AEDT x 3 projects = 30",
        },
        "pins": {
            "solver": CAMPAIGN_SOLVER,
            "library": LIBRARY_REVISION,
            "package": SCHEDULER_PACKAGE_REVISION,
            "physics": PHYSICS_REVISION,
        },
        "resources": {key: submission[key] for key in (
            "cpus", "memory_mb", "timeout_seconds", "account_names"
        )},
        "environment": submission["submission_env"],
        "execution_requires": [
            "campaign-demand API deployed with CAS default 500",
            "q21b tasks 41796-41798 and releases remain valid",
            "pool remains exact 10x3/30",
            "active policy remains 0..30 with validated limit 30",
            "every eligible account has clean exact scheduler package",
            "solver/library revisions remain advertised remote branch heads",
        ],
        "mixed_canary_dependency": False,
        "workload_isolation": "family",
        "writes_performed": False,
        "tasks_submitted": 0,
    }


def run_live_gates(args: argparse.Namespace) -> dict[str, Any]:
    verify_compatibility(REPO_ROOT, COMPATIBILITY_PATH)
    verify_profile(PROFILE_PATH)
    verify_local_library(args.library_root)
    try:
        deployment = deployment_gate.validate_deployment(
            args.deployment_solver_root,
            CAMPAIGN_SOLVER,
            args.library_root,
            LIBRARY_REVISION,
        )
    except Exception as exc:
        raise GateError(f"remote deployment revision gate failed: {exc}") from exc
    pool, logical_target = verify_pool_and_policy(args.scheduler_url)
    evidence = verify_scheduler_evidence(args.scheduler_db)
    packages = audit_remote_packages(
        args.accounts_config, args.eligible_accounts, args.ssh_audit_python
    )
    return {
        "deployment": deployment,
        "logical_target": logical_target,
        "pool_validation_passed": bool((pool.get("config") or {}).get(
            "validation_passed"
        )),
        "scheduler_evidence": evidence,
        "packages": packages,
    }


def execute_cycle(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    # Remote/package evidence is checked immediately before entering the one
    # host-wide mutation epoch.  Demand and active policy are then observed
    # under the same lock used by both Web UI PATCH routes.
    gates = run_live_gates(args)
    with feeder.campaign_mutation_lock():
        demand = read_campaign_demand(args.scheduler_url)
        _pool, logical_target = verify_pool_and_policy(args.scheduler_url)
        current_serial = _state_serial(args.state_dir / "feeder_state.json")
        verify_owned_serials(args.scheduler_db, manifest, current_serial)
        progress = campaign_progress(
            manifest, current_serial, int(demand["total_simulations"])
        )
        remaining = progress["remaining_simulations"]
        if remaining and logical_target:
            feeder.step(
                2_000_000_000,
                target=logical_target,
                buffer=0,
                solver_revision=CAMPAIGN_SOLVER,
                library_revision=LIBRARY_REVISION,
                candidate_seed=CANDIDATE_SEED,
                pooled_submission=pooled_submission(args),
                max_new_tasks=remaining,
            )
            current_serial = _state_serial(args.state_dir / "feeder_state.json")
            verify_owned_serials(args.scheduler_db, manifest, current_serial)
            progress = campaign_progress(
                manifest, current_serial, int(demand["total_simulations"])
            )
        return {
            "schema": SCHEMA,
            "campaign": CAMPAIGN_ID,
            "manifest": {
                "version": int(manifest.get("manifest_version") or 1),
                "identity_sha256": manifest["identity_sha256"],
                "eligible_accounts": list(manifest["eligible_accounts"]),
            },
            "phase": "demand-satisfied" if not progress["remaining_simulations"] else (
                "paused-by-active-policy" if not logical_target else "rolling-refill"
            ),
            "updated_at_epoch": time.time(),
            "demand_revision": demand["demand_revision"],
            "progress": progress,
            "logical_target": logical_target,
            "gates": gates,
            "no_cancellation_performed": True,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-mft-family-production",
        "--execute-approved-after-mixed",
        action="store_true",
        dest="execute_mft_family_production",
        help=(
            "execute the MFT-only family-isolated campaign; the old alias is "
            "retained only for supervisor compatibility"
        ),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--live-readonly-gates", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument(
        "--manifest-version",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "1 uses the original immutable manifest; 2 creates/loads an "
            "append-only account-superset manifest referencing v1"
        ),
    )
    parser.add_argument("--planned-total-demand", type=int, default=DEFAULT_TOTAL_DEMAND)
    parser.add_argument(
        "--adopt-baseline-serial",
        type=int,
        default=ADOPTED_BASELINE_SERIAL,
        help=(
            "prelaunch canonical feeder serial; already accepted later "
            "serials count toward total demand"
        ),
    )
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--pool-url", default=DEFAULT_POOL_URL)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--library-root", type=Path, default=DEFAULT_LIBRARY_ROOT)
    parser.add_argument(
        "--deployment-solver-root",
        type=Path,
        default=DEFAULT_DEPLOYMENT_SOLVER_ROOT,
    )
    parser.add_argument("--scheduler-db", type=Path, default=DEFAULT_SCHEDULER_DB)
    parser.add_argument("--accounts-config", type=Path, default=DEFAULT_ACCOUNTS_CONFIG)
    parser.add_argument(
        "--ssh-audit-python", type=Path, default=DEFAULT_SSH_AUDIT_PYTHON
    )
    parser.add_argument(
        "--eligible-account",
        action="append",
        dest="eligible_accounts",
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.eligible_accounts = tuple(args.eligible_accounts or DEFAULT_ELIGIBLE_ACCOUNTS)
    if (not args.eligible_accounts or len(set(args.eligible_accounts))
            != len(args.eligible_accounts)):
        raise GateError("eligible accounts must be a non-empty unique list")
    if (isinstance(args.planned_total_demand, bool)
            or not 0 <= args.planned_total_demand <= MAX_TOTAL_DEMAND):
        raise GateError(f"planned total demand must be 0..{MAX_TOTAL_DEMAND}")
    if args.interval_seconds < 5:
        raise GateError("interval-seconds must be at least 5")

    state_path = configure_feeder(args)
    if not state_path.is_file():
        raise GateError(f"canonical feeder state does not exist: {state_path}")
    verify_compatibility(REPO_ROOT, COMPATIBILITY_PATH)
    verify_profile(PROFILE_PATH)
    manifest_path = manifest_path_for_version(args.state_dir, args.manifest_version)
    predecessor_path = manifest_path_for_version(args.state_dir, 1)
    status_path = args.state_dir / f"{CAMPAIGN_ID}.status.json"

    if not args.execute_mft_family_production:
        manifest = load_manifest_version(
            manifest_path,
            predecessor_path,
            state_path,
            args.eligible_accounts,
            version=args.manifest_version,
            execute=False,
            baseline_serial=args.adopt_baseline_serial,
        )
        plan = static_plan(args, manifest)
        if args.live_readonly_gates:
            try:
                plan["live_gates"] = run_live_gates(args)
                plan["live_demand"] = read_campaign_demand(args.scheduler_url)
            except GateError as exc:
                plan["live_gate_blocker"] = str(exc)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    try:
        with FileLock(feeder.CONTROLLER_LOCK, timeout=0):
            if args.manifest_version == 2:
                # Never persist an expanded eligible-account identity until
                # every appended account passes the exact remote package and
                # environment audit. execute_cycle repeats these gates before
                # every possible submission.
                run_live_gates(args)
            with feeder.campaign_mutation_lock():
                # Refuse to create the immutable baseline until the mutable
                # demand API exists.  The initial UI value is expected to be
                # 500, but it may subsequently change by CAS.
                read_campaign_demand(args.scheduler_url)
                manifest = load_manifest_version(
                    manifest_path,
                    predecessor_path,
                    state_path,
                    args.eligible_accounts,
                    version=args.manifest_version,
                    execute=True,
                    baseline_serial=args.adopt_baseline_serial,
                )
            while True:
                try:
                    status = execute_cycle(args, manifest)
                except GateError as exc:
                    status = {
                        "schema": SCHEMA,
                        "campaign": CAMPAIGN_ID,
                        "phase": "blocked-fail-closed",
                        "updated_at_epoch": time.time(),
                        "blocker": str(exc),
                        "no_submission_attempted": True,
                    }
                _write_status(status_path, status)
                print(json.dumps(status, sort_keys=True), flush=True)
                if args.once:
                    return 0 if status.get("phase") != "blocked-fail-closed" else 2
                time.sleep(args.interval_seconds)
    except FileLockTimeout as exc:
        raise GateError("another feeder/controller already owns the canonical lock") from exc


if __name__ == "__main__":
    raise SystemExit(main())
