"""Plan, stage, submit, and harvest immutable CPU-only NSGA-II Slurm lanes.

The default action is non-mutating.  Cluster staging and scheduler submission
both require an explicit ``--apply`` flag.  This keeps the local Windows lanes
running while providing a reproducible offload path that does not use AEDT or
modify/restart the scheduler service.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import posixpath
import shlex
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(r"C:\Users\peets\slurm_scheduler_runtime")
DEFAULT_DEPLOYMENT = (
    RUNTIME
    / "mft_nsga_fastlane_deployments"
    / ("core4-cw1-5t-res20k-box1000x1000x750-20260718")
)
DEFAULT_DATASET = (
    RUNTIME
    / "mft_pipeline"
    / "experimental_rejected_quarantine"
    / "runs"
    / "20260717T204739-e7d59cb3"
    / "dataset"
    / "strict_7c5e2eb29733ed89_776851043385.parquet"
)
DEFAULT_GENERATION = (
    RUNTIME
    / "mft_nsga_transition_audit"
    / "canonical-e186-wave-048048bfaefdf1de"
    / "isolated_registry"
    / "generations"
    / "20260717T204739-e7d59cb3"
)
DEFAULT_QUALITY = (
    RUNTIME
    / "mft_pipeline"
    / "experimental_rejected_quarantine"
    / "runs"
    / "20260717T204739-e7d59cb3"
    / "evidence"
    / "quality_status.json"
)
DEFAULT_LOCAL_ROOT = RUNTIME / "mft_nsga_slurm_offload"
DEFAULT_REMOTE_ROOT = "/gpfs/tmp_cpu2/mft_nsga_offload"
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"

PLAN_SCHEMA = "mft-slurm-nsga-offload-plan-v1"
BUNDLE_SCHEMA = "mft-slurm-nsga-bundle-v1"
LANE_SCHEMA = "mft-slurm-nsga-lane-v1"

# The scheduler's text endpoints reject bursts above four readers.  Artifact
# bytes do not go through those endpoints at all; they are copied directly over
# SFTP using at most one persistent SSH connection per concurrently processed
# account.
MAX_HARVEST_API_WORKERS = 4
MAX_HARVEST_SFTP_CONNECTIONS = 4
DEFAULT_HARVEST_RETRIES = 3
LANE_STATUS_MAX_BYTES = 1024 * 1024

CRITICAL_PACKAGES = {
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "scikit-learn": "1.9.0",
    "scipy": "1.17.0",
    "pymoo": "0.6.2",
    "joblib": "1.5.3",
    "pyarrow": "24.0.0",
    "lightgbm": "4.6.0",
    "xgboost": "3.2.0",
    "catboost": "1.2.10",
}
REQUIREMENTS_LOCK = "".join(
    f"{name}=={version}\n" for name, version in CRITICAL_PACKAGES.items()
)

CODE_EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}
CODE_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normal_relative(value: Path) -> str:
    return value.as_posix().lstrip("/")


def iter_code_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in CODE_EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in CODE_EXCLUDED_SUFFIXES:
            continue
        yield path, relative


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def _revision_from_quality(quality: dict, preferred: str, fallback: str) -> str:
    value = str(quality.get(preferred) or quality.get(fallback) or "").lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"quality evidence has no exact {preferred}")
    return value


def build_plan(
    deployment: Path,
    dataset: Path,
    generation: Path,
    quality_status: Path,
    local_root: Path,
    remote_root: str,
    runner: Path | None = None,
    warm_artifact: Path | None = None,
    warm_contract: Path | None = None,
    warm_kind: str = "standard_fea",
) -> tuple[dict, dict]:
    deployment = deployment.resolve()
    dataset = dataset.resolve()
    generation = generation.resolve()
    quality_status = quality_status.resolve()
    runner = (runner or (REPO_ROOT / "tools" / "slurm_nsga_lane_runner.py")).resolve()
    for path in (deployment, dataset, generation, quality_status, runner):
        if not path.exists():
            raise FileNotFoundError(path)
    if bool(warm_artifact) != bool(warm_contract):
        raise ValueError("warm artifact and contract must be supplied together")
    if warm_kind not in {"standard_fea", "cross_run"}:
        raise ValueError("warm_kind must be standard_fea or cross_run")

    report_path = generation / "train_report.json"
    report = _load_json(report_path)
    quality = _load_json(quality_status)
    model_artifacts = report.get("artifacts")
    if not isinstance(model_artifacts, dict) or not model_artifacts:
        raise RuntimeError("generation report has no immutable artifact inventory")

    files: dict[str, dict] = {}
    local_sources: dict[str, str] = {}

    def add_file(remote_relative: str, local_path: Path, kind: str, expected=None):
        local_path = local_path.resolve()
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        actual = sha256_file(local_path) if expected is None else str(expected)
        if len(actual) != 64:
            raise RuntimeError(f"invalid expected SHA-256 for {local_path}")
        files[remote_relative] = {
            "sha256": actual,
            "size": local_path.stat().st_size,
            "kind": kind,
        }
        local_sources[remote_relative] = str(local_path)

    for path, relative in iter_code_files(deployment):
        add_file(
            "artifacts/code/" + _normal_relative(relative),
            path,
            "optimizer_code",
        )
    add_file("artifacts/dataset/train.parquet", dataset, "dataset")
    add_file("artifacts/quality/quality_status.json", quality_status, "quality")
    add_file("artifacts/runner/slurm_nsga_lane_runner.py", runner, "runner")

    generation_remote = f"artifacts/registry/generations/{generation.name}"
    add_file(f"{generation_remote}/train_report.json", report_path, "generation_report")
    inventory_paths = set()
    for relative, expected in sorted(model_artifacts.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe model artifact path: {relative}")
        source = generation / relative_path
        add_file(
            f"{generation_remote}/{_normal_relative(relative_path)}",
            source,
            "model_artifact",
            expected=expected,
        )
        inventory_paths.add(relative_path.as_posix())
    actual_generation_files = {
        path.relative_to(generation).as_posix()
        for path in generation.rglob("*")
        if path.is_file() and path.name != "train_report.json"
    }
    if actual_generation_files != inventory_paths:
        missing = sorted(inventory_paths - actual_generation_files)
        extra = sorted(actual_generation_files - inventory_paths)
        raise RuntimeError(
            f"generation inventory mismatch; missing={missing[:5]} extra={extra[:5]}"
        )

    requirements_bytes = REQUIREMENTS_LOCK.encode("utf-8")
    files["artifacts/runtime/requirements.lock"] = {
        "sha256": sha256_bytes(requirements_bytes),
        "size": len(requirements_bytes),
        "kind": "runtime_lock",
    }

    warm = None
    if warm_artifact and warm_contract:
        warm_artifact = warm_artifact.resolve()
        warm_contract = warm_contract.resolve()
        artifact_rel = "artifacts/warm/" + warm_artifact.name
        contract_rel = "artifacts/warm/" + warm_contract.name
        add_file(artifact_rel, warm_artifact, "warm_start")
        add_file(contract_rel, warm_contract, "warm_start_contract")
        warm = {
            "kind": warm_kind,
            "artifact": artifact_rel,
            "contract": contract_rel,
        }

    source_paths = {
        "code_root": "artifacts/code",
        "dataset": "artifacts/dataset/train.parquet",
        "registry": "artifacts/registry",
        "registry_generation": generation_remote,
        "quality_status": "artifacts/quality/quality_status.json",
    }
    high_value_relatives = [
        "artifacts/code/regression_260707/optimization/run_nsga2.py",
        "artifacts/code/regression_260707/optimization/nsga2_problem.py",
        "artifacts/code/regression_260707/optimization/resonance.py",
        "artifacts/code/regression_260707/training/predictor.py",
        "artifacts/code/regression_260707/uncertainty_contract.py",
        "artifacts/dataset/train.parquet",
        "artifacts/quality/quality_status.json",
        f"{generation_remote}/train_report.json",
        "artifacts/runner/slurm_nsga_lane_runner.py",
        "artifacts/runtime/requirements.lock",
    ]
    absent = [relative for relative in high_value_relatives if relative not in files]
    if absent:
        raise RuntimeError(f"deployment is missing high-value files: {absent}")

    stable_contract = {
        "schema_version": BUNDLE_SCHEMA,
        "source_paths": source_paths,
        "files": files,
        "model_artifacts_sha256": model_artifacts,
        "high_value_sha256": {
            relative: files[relative]["sha256"] for relative in high_value_relatives
        },
        "runtime": {
            "python_major_minor": "3.11",
            "critical_packages": CRITICAL_PACKAGES,
        },
        "execution_contract": {
            "restart_executor_on_linux": "process_pool",
            "slurm_safe_workers_per_task": 1,
            "parallelism_axis": "one_seed_per_scheduler_task",
            "reason": (
                "run_nsga2 passes the multi-gigabyte fitted-model graph as a "
                "ProcessPool task argument on Linux; one worker avoids repeated "
                "serialization while many independent Slurm tasks consume CPUs"
            ),
        },
        "training_run_id": report.get("training_run_id"),
        "strict_full_rows": report.get("strict_full_rows"),
        "dataset_sha256": files["artifacts/dataset/train.parquet"]["sha256"],
        "generation_report_sha256": files[f"{generation_remote}/train_report.json"][
            "sha256"
        ],
        "quality_status_sha256": files["artifacts/quality/quality_status.json"][
            "sha256"
        ],
        "fea_solver_revision": _revision_from_quality(
            quality, "solver_revision_pin", "solver_revision"
        ),
        "fea_library_revision": _revision_from_quality(
            quality, "library_revision_pin", "library_revision"
        ),
        "warm_start": warm,
    }
    contract_sha = sha256_bytes(canonical_bytes(stable_contract))
    bundle_id = f"core4-{contract_sha[:20]}"
    manifest = dict(stable_contract)
    manifest.update(
        {
            "bundle_id": bundle_id,
            "contract_sha256": contract_sha,
        }
    )

    plan_dir = local_root.resolve() / bundle_id
    requirements_path = plan_dir / "requirements.lock"
    manifest_path = plan_dir / "bundle_manifest.json"
    source_map_path = plan_dir / "local_sources.json"
    plan_path = plan_dir / "offload_plan.json"
    requirements_path.parent.mkdir(parents=True, exist_ok=True)
    # ``Path.write_text`` performs ``\n`` -> ``\r\n`` translation on Windows,
    # which would make the staged file differ from the LF-only bytes hashed in
    # the immutable manifest above.
    requirements_path.write_bytes(requirements_bytes)
    local_sources["artifacts/runtime/requirements.lock"] = str(requirements_path)
    atomic_json(manifest_path, manifest)
    atomic_json(source_map_path, local_sources)
    manifest_sha = sha256_file(manifest_path)
    remote_bundle = remote_root.rstrip("/") + "/" + bundle_id
    plan = {
        "schema_version": PLAN_SCHEMA,
        "created_at": _utc_now(),
        "bundle_id": bundle_id,
        "contract_sha256": contract_sha,
        "bundle_manifest_sha256": manifest_sha,
        "bundle_manifest": str(manifest_path),
        "local_sources": str(source_map_path),
        "local_plan_dir": str(plan_dir),
        "remote_root": remote_root.rstrip("/"),
        "remote_bundle": remote_bundle,
        "runtime_requirements": str(requirements_path),
        "warm_start": warm,
        "recommended_resources": {
            "scheduling_profile": "standard",
            "aedt_used": False,
            "gpus": 0,
            "cpus_per_seed_task": 1,
            "memory_mb_per_seed_task": 32768,
            "workers_per_seed_task": 1,
            "recommended_seed_tasks": 32,
            "max_workers_per_node": 4,
            "timeout_seconds": 86400,
        },
    }
    atomic_json(plan_path, plan)
    return plan, manifest


def load_plan(path: Path) -> tuple[dict, dict, dict]:
    plan = _load_json(path.resolve())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise RuntimeError("unsupported offload plan schema")
    manifest_path = Path(plan["bundle_manifest"])
    manifest = _load_json(manifest_path)
    if manifest.get("bundle_id") != plan.get("bundle_id") or sha256_file(
        manifest_path
    ) != plan.get("bundle_manifest_sha256"):
        raise RuntimeError("local offload plan fingerprint mismatch")
    source_map = _load_json(Path(plan["local_sources"]))
    return plan, manifest, source_map


def _scheduler_imports(source: Path):
    source = source.resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from slurm_scheduler.config import load_accounts
    from slurm_scheduler.slurm import SSHSession

    return load_accounts, SSHSession


def _account(accounts_path: Path, scheduler_source: Path, name: str):
    load_accounts, ssh_session = _scheduler_imports(scheduler_source)
    accounts = {account.name: account for account in load_accounts(accounts_path)}
    if name not in accounts:
        raise RuntimeError(f"unknown scheduler account: {name}")
    return accounts[name], ssh_session


def _remote_ready(session, remote_bundle: str, manifest_sha: str) -> bool:
    command = (
        f"test -f {shlex.quote(remote_bundle + '/READY.json')} && "
        f"cat {shlex.quote(remote_bundle + '/READY.json')}"
    )
    result = session.run(command, timeout=60)
    if result.exit_code != 0:
        return False
    try:
        ready = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    return (
        ready.get("bundle_manifest_sha256") == manifest_sha
        and ready.get("runtime_verified") is True
    )


def stage_bundle(
    plan_path: Path,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    staging_account: str = "harry261",
    resume_incoming: str | None = None,
    apply: bool = False,
) -> dict:
    plan, manifest, source_map = load_plan(plan_path)
    summary = {
        "action": "stage",
        "apply": bool(apply),
        "bundle_id": plan["bundle_id"],
        "remote_bundle": plan["remote_bundle"],
        "file_count": len(manifest["files"]),
        "byte_count": sum(int(item["size"]) for item in manifest["files"].values()),
        "staging_account": staging_account,
    }
    if not apply:
        return summary

    account, ssh_session = _account(accounts_path, scheduler_source, staging_account)
    remote_bundle = plan["remote_bundle"]
    incoming_prefix = plan["remote_root"] + "/.incoming-" + plan["bundle_id"] + "-"
    if resume_incoming is not None:
        incoming = str(resume_incoming).rstrip("/")
        suffix = incoming.removeprefix(incoming_prefix)
        if (
            not incoming.startswith(incoming_prefix)
            or not suffix
            or "/" in suffix
            or any(char not in "0123456789abcdef" for char in suffix.lower())
        ):
            raise ValueError(
                "resume incoming must be the exact incomplete directory for this bundle"
            )
        summary["resumed_from"] = incoming
    else:
        incoming = incoming_prefix + uuid.uuid4().hex[:10]
    with ssh_session(account, default_timeout=300) as session:
        if _remote_ready(session, remote_bundle, plan["bundle_manifest_sha256"]):
            summary["already_ready"] = True
            return summary
        check = session.run(f"test ! -e {shlex.quote(remote_bundle)}", timeout=60)
        if check.exit_code != 0:
            raise RuntimeError(
                "remote bundle exists without matching READY.json; refusing overwrite"
            )
        if resume_incoming is not None:
            resumed = session.run(
                f"test -d {shlex.quote(incoming)} && "
                f"test ! -e {shlex.quote(incoming + '/READY.json')}",
                timeout=60,
            )
            if resumed.exit_code != 0:
                raise RuntimeError(
                    "requested incomplete staging directory is absent or already ready"
                )
        create = session.run(
            "set -e; "
            f"mkdir -p {shlex.quote(incoming + '/runs')}; "
            f"chmod 0755 {shlex.quote(incoming)}; "
            f"chmod 1777 {shlex.quote(incoming + '/runs')}",
            timeout=60,
        )
        if create.exit_code != 0:
            raise RuntimeError(create.stderr or "failed to create remote staging root")

        local_manifest = Path(plan["bundle_manifest"])
        resume_verified: set[str] = set()
        resume_manifest = incoming + "/.resume-manifest.json"
        if resume_incoming is not None:
            session.upload_file(str(local_manifest), resume_manifest + ".part")
            promoted = session.run(
                f"mv {shlex.quote(resume_manifest + '.part')} "
                f"{shlex.quote(resume_manifest)}",
                timeout=60,
            )
            if promoted.exit_code != 0:
                raise RuntimeError(promoted.stderr or "resume manifest upload failed")
            resume_script = """
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
verified = []
for rel, record in manifest['files'].items():
    path = root / rel
    if not path.is_file() or path.stat().st_size != int(record['size']):
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() == record['sha256']:
        verified.append(rel)
print(json.dumps(verified, sort_keys=True))
""".strip()
            resume_command = (
                "python - "
                + shlex.quote(incoming)
                + " "
                + shlex.quote(resume_manifest)
                + " <<'PY'\n"
                + resume_script
                + "\nPY"
            )
            checked = session.run(
                "bash -lc " + shlex.quote(resume_command), timeout=1800
            )
            if checked.exit_code != 0:
                raise RuntimeError(
                    checked.stderr or checked.stdout or "resume verification failed"
                )
            resume_verified = set(json.loads(checked.stdout.strip().splitlines()[-1]))
            summary["resume_verified_files"] = len(resume_verified)

        try:
            for index, (relative, record) in enumerate(
                sorted(manifest["files"].items()), 1
            ):
                local = Path(source_map[relative])
                if local.stat().st_size != int(record["size"]):
                    raise RuntimeError(f"local size changed after plan: {local}")
                if sha256_file(local) != record["sha256"]:
                    raise RuntimeError(f"local fingerprint changed after plan: {local}")
                remote = incoming + "/" + relative
                parent = remote.rsplit("/", 1)[0]
                mkdir = session.run(f"mkdir -p {shlex.quote(parent)}", timeout=60)
                if mkdir.exit_code != 0:
                    raise RuntimeError(mkdir.stderr or f"mkdir failed: {parent}")
                if relative in resume_verified:
                    print(
                        json.dumps(
                            {
                                "event": "resume-skip",
                                "index": index,
                                "count": len(manifest["files"]),
                                "path": relative,
                                "bytes": record["size"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                part = remote + ".part"
                session.upload_file(str(local), part)
                moved = session.run(
                    "set -e; "
                    f"test $(stat -c %s {shlex.quote(part)}) -eq {int(record['size'])}; "
                    f"mv {shlex.quote(part)} {shlex.quote(remote)}",
                    timeout=60,
                )
                if moved.exit_code != 0:
                    raise RuntimeError(
                        moved.stderr or f"upload finalize failed: {relative}"
                    )
                print(
                    json.dumps(
                        {
                            "event": "uploaded",
                            "index": index,
                            "count": len(manifest["files"]),
                            "path": relative,
                            "bytes": record["size"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            session.run(f"rm -f {shlex.quote(resume_manifest)}", timeout=60)
            session.upload_file(
                str(local_manifest), incoming + "/bundle_manifest.json.part"
            )
            finalize_manifest = session.run(
                "set -e; "
                f"echo {shlex.quote(plan['bundle_manifest_sha256'] + '  ' + incoming + '/bundle_manifest.json.part')} | sha256sum -c -; "
                f"mv {shlex.quote(incoming + '/bundle_manifest.json.part')} "
                f"{shlex.quote(incoming + '/bundle_manifest.json')}",
                timeout=120,
            )
            if finalize_manifest.exit_code != 0:
                raise RuntimeError(
                    finalize_manifest.stderr or "bundle manifest upload mismatch"
                )

            setup = (account.env_profiles or {}).get("pyaedt2026v1", "")
            site = incoming + "/artifacts/python-site"
            requirements = incoming + "/artifacts/runtime/requirements.lock"
            install_command = "\n".join(
                [
                    "set -euo pipefail",
                    setup,
                    f"mkdir -p {shlex.quote(site)}",
                    "python -m pip install --disable-pip-version-check --no-input "
                    f"--target {shlex.quote(site)} -r {shlex.quote(requirements)}",
                ]
            )
            installed = session.run(
                "bash -lc " + shlex.quote(install_command), timeout=1800
            )
            if installed.exit_code != 0:
                raise RuntimeError(installed.stderr or "runtime package staging failed")

            verify_script = """
import hashlib, importlib.metadata, json, pathlib, sys
root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'bundle_manifest.json').read_text())
for rel,record in manifest['files'].items():
    path=root/rel
    if not path.is_file() or path.stat().st_size != int(record['size']):
        raise SystemExit('missing or wrong size: '+rel)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    if h.hexdigest() != record['sha256']:
        raise SystemExit('sha256 mismatch: '+rel)
actual={name:importlib.metadata.version(name) for name in manifest['runtime']['critical_packages']}
if actual != manifest['runtime']['critical_packages']:
    raise SystemExit('runtime package mismatch: '+repr(actual))
print(json.dumps(actual,sort_keys=True))
""".strip()
            verify_command = "\n".join(
                [
                    "set -euo pipefail",
                    setup,
                    f"export PYTHONPATH={shlex.quote(site)}",
                    "python - " + shlex.quote(incoming) + " <<'PY'",
                    verify_script,
                    "PY",
                ]
            )
            verified = session.run(
                "bash -lc " + shlex.quote(verify_command), timeout=1800
            )
            if verified.exit_code != 0:
                raise RuntimeError(
                    verified.stderr or verified.stdout or "remote verification failed"
                )
            actual_packages = json.loads(verified.stdout.strip().splitlines()[-1])
            ready = {
                "schema_version": BUNDLE_SCHEMA,
                "bundle_id": plan["bundle_id"],
                "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
                "runtime_verified": True,
                "runtime_packages": actual_packages,
                "verified_at": _utc_now(),
                "verified_by_account": staging_account,
            }
            session.write_text_file(
                incoming + "/READY.json",
                json.dumps(ready, indent=2, sort_keys=True) + "\n",
            )
            freeze = session.run(
                "set -e; "
                f"chmod -R a-w {shlex.quote(incoming + '/artifacts')}; "
                f"chmod -R a+rX {shlex.quote(incoming + '/artifacts')}; "
                f"chmod 1777 {shlex.quote(incoming + '/runs')}; "
                f"mv {shlex.quote(incoming)} {shlex.quote(remote_bundle)}",
                timeout=300,
            )
            if freeze.exit_code != 0:
                raise RuntimeError(freeze.stderr or "atomic bundle publication failed")
        except Exception:
            # Preserve the incomplete root for resumable forensic inspection;
            # it is never considered runnable without READY.json at final path.
            summary["incomplete_remote"] = incoming
            raise
    summary["ready"] = True
    return summary


def _api_json(url: str, method: str = "GET", payload: dict | None = None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = canonical_bytes(payload)
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def validate_seed_ranges(seed_bases: list[int], restarts: int) -> None:
    if restarts < 1:
        raise ValueError("restarts must be positive")
    occupied = set()
    for seed in seed_bases:
        if seed < 0:
            raise ValueError("seed bases must be non-negative")
        lane = set(range(seed, seed + restarts))
        if occupied & lane:
            raise ValueError("lane seed ranges overlap")
        occupied.update(lane)


def build_task_payload(
    plan: dict,
    manifest: dict,
    seed_base: int,
    restarts: int,
    population: int,
    max_generations: int,
    workers: int,
    cpus: int = 1,
    memory_mb: int = 32768,
    timeout_seconds: int = 86400,
    priority: int = 0,
    max_workers_per_node: int = 4,
    lane_prefix: str = "lane",
) -> dict:
    if not (1 <= workers <= 4 and workers <= restarts):
        raise ValueError("workers must be 1..4 and no greater than restarts")
    safe_workers = int(
        (manifest.get("execution_contract") or {}).get("slurm_safe_workers_per_task", 1)
    )
    if workers != safe_workers or restarts != 1:
        raise ValueError(
            "this immutable optimizer bundle must run as one seed/one worker per "
            "Slurm task; scale CPU use by submitting more seed tasks"
        )
    if population < 2 or max_generations < 1 or cpus < workers:
        raise ValueError("invalid population/generation/CPU shape")
    lane_id = (
        f"{lane_prefix}_seed{seed_base}_r{restarts}_p{population}_g{max_generations}"
    )
    lane_contract = {
        "schema_version": LANE_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "lane_id": lane_id,
        "seed_base": int(seed_base),
        "restarts": int(restarts),
        "population": int(population),
        "max_generations": int(max_generations),
        "workers": int(workers),
        "round": 0,
        "fea_solver_revision": manifest["fea_solver_revision"],
        "fea_library_revision": manifest["fea_library_revision"],
        "warm_start": manifest.get("warm_start"),
        "production_eligible": False,
        "experimental_active_learning": True,
    }
    lane_contract_sha256 = sha256_bytes(canonical_bytes(lane_contract))
    command = "\n".join(
        [
            "set -euo pipefail",
            'export PYTHONPATH="$PWD/artifacts/python-site${PYTHONPATH:+:$PYTHONPATH}"',
            'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:?scheduler payload path is missing}"',
            'case "$payload_path" in /*) ;; *) payload_path="$HOME/$payload_path" ;; esac',
            'payload_path=$(realpath -e -- "$payload_path")',
            'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
            'case "$payload_path" in "$payload_root"/*/payload.json) ;; '
            '*) echo "unsafe scheduler payload path: $payload_path" >&2; exit 66 ;; esac',
            "exec python artifacts/runner/slurm_nsga_lane_runner.py "
            '--bundle-root "$PWD" --payload "$payload_path" '
            '--payload-root "$payload_root" '
            f"--payload-sha256 {lane_contract_sha256}",
        ]
    )
    dedupe_contract = {
        "bundle": plan["bundle_id"],
        "lane": lane_contract,
        "cpus": cpus,
        "memory_mb": memory_mb,
    }
    dedupe_sha = sha256_bytes(canonical_bytes(dedupe_contract))
    return {
        "name": f"mft-nsga2-cpu-{plan['bundle_id'][-8:]}-{seed_base}",
        "remote_cwd": plan["remote_bundle"],
        "command": command,
        "payload_json": lane_contract,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "cpus": int(cpus),
        "memory_mb": int(memory_mb),
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": int(priority),
        "timeout_seconds": int(timeout_seconds),
        "dedupe_key": "mft-nsga2-cpu:" + dedupe_sha,
        "max_workers_per_node": int(max_workers_per_node),
    }


def submit_lanes(
    plan_path: Path,
    seed_bases: list[int],
    restarts: int,
    population: int,
    max_generations: int,
    workers: int,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    cpus: int = 1,
    memory_mb: int = 32768,
    timeout_seconds: int = 86400,
    priority: int = 0,
    max_workers_per_node: int = 4,
    lane_prefix: str = "lane",
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    staging_account: str = "harry261",
    apply: bool = False,
) -> dict:
    validate_seed_ranges(seed_bases, restarts)
    plan, manifest, _ = load_plan(plan_path)
    payloads = [
        build_task_payload(
            plan,
            manifest,
            seed,
            restarts,
            population,
            max_generations,
            workers,
            cpus=cpus,
            memory_mb=memory_mb,
            timeout_seconds=timeout_seconds,
            priority=priority,
            max_workers_per_node=max_workers_per_node,
            lane_prefix=lane_prefix,
        )
        for seed in seed_bases
    ]
    result = {
        "action": "submit",
        "apply": bool(apply),
        "bundle_id": plan["bundle_id"],
        "scheduler_url": scheduler_url.rstrip("/"),
        "payloads": payloads,
        "submissions": [],
    }
    if not apply:
        return result

    account, ssh_session = _account(accounts_path, scheduler_source, staging_account)
    with ssh_session(account, default_timeout=60) as session:
        if not _remote_ready(
            session, plan["remote_bundle"], plan["bundle_manifest_sha256"]
        ):
            raise RuntimeError("remote immutable bundle is not ready")
    for payload in payloads:
        response = _api_json(
            scheduler_url.rstrip("/") + "/api/tasks",
            method="POST",
            payload=payload,
            timeout=30,
        )
        result["submissions"].append(
            {
                "task_id": int(response["task_id"]),
                "deduped": bool(response.get("deduped")),
                "name": payload["name"],
                "lane": payload["payload_json"],
                "submitted_at": _utc_now(),
            }
        )
    ledger_path = Path(plan["local_plan_dir"]) / "submissions.json"
    existing = []
    if ledger_path.is_file():
        loaded = _load_json(ledger_path)
        existing = list(loaded.get("submissions") or [])
    atomic_json(
        ledger_path,
        {
            "schema_version": PLAN_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "scheduler_url": scheduler_url.rstrip("/"),
            "submissions": existing + result["submissions"],
            "updated_at": _utc_now(),
        },
    )
    result["ledger"] = str(ledger_path)
    return result


HARVEST_FILENAMES = (
    "COMPLETED",
    "optimization_manifest.json",
    "pareto_front.csv",
    "pareto_X.npy",
    "pareto_F.npy",
    "infeasibility_report.json",
    "least_violation_manifest.json",
    "least_violation_front.csv",
    "least_violation_X.npy",
    "least_violation_F.npy",
    "least_violation_G.npy",
)


TERMINAL_REQUIRED_ARTIFACTS = {
    "completed": frozenset(
        {
            "COMPLETED",
            "optimization_manifest.json",
            "pareto_front.csv",
            "pareto_X.npy",
            "pareto_F.npy",
        }
    ),
    "infeasible": frozenset({"infeasibility_report.json"}),
}


class RemoteArtifactMissing(FileNotFoundError):
    """The requested artifact does not exist on its owning scheduler account."""


class RemoteArtifactChanged(RuntimeError):
    """A supposedly immutable artifact changed during an SFTP transaction."""


class _PersistentAccountConnection:
    """One reconnectable, persistent SSH transport owned by one worker thread."""

    def __init__(self, account, ssh_session, timeout: int = 300):
        self.account = account
        self.ssh_session = ssh_session
        self.timeout = timeout
        self._session = None
        self.connection_count = 0

    def get(self):
        if self._session is None:
            session = self.ssh_session(self.account, default_timeout=self.timeout)
            session.__enter__()
            self._session = session
            self.connection_count += 1
        return self._session

    def reset(self) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        try:
            session.__exit__(None, None, None)
        except Exception:
            pass

    def close(self) -> None:
        self.reset()


def _remote_file_identity(session, remote_path: str) -> dict:
    """Hash a remote regular file without routing any bytes through the web API."""
    script = "; ".join(
        [
            "set -e",
            "p=" + shlex.quote(remote_path),
            'if [ ! -f "$p" ]; then exit 44; fi',
            'size=$(stat -c %s -- "$p")',
            'sha=$(sha256sum -- "$p")',
            "sha=${sha%% *}",
            'printf "%s\\t%s\\n" "$size" "$sha"',
        ]
    )
    result = session.run("bash -lc " + shlex.quote(script), timeout=300)
    if result.exit_code == 44:
        raise RemoteArtifactMissing(remote_path)
    if result.exit_code != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or "remote identity command failed").strip()
        )
    line = next(
        (item.strip() for item in reversed(result.stdout.splitlines()) if item.strip()),
        "",
    )
    size_text, separator, digest = line.partition("\t")
    digest = digest.strip().lower()
    if (
        not separator
        or not size_text.isdigit()
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise RuntimeError("invalid remote sha256/stat response")
    return {"bytes": int(size_text), "sha256": digest}


def _retry_remote_identity(
    connection: _PersistentAccountConnection,
    remote_path: str,
    retries: int,
) -> tuple[dict, int]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return _remote_file_identity(connection.get(), remote_path), attempt
        except RemoteArtifactMissing:
            raise
        except Exception as exc:
            last_error = exc
            connection.reset()
            if attempt < retries:
                time.sleep(min(0.5, 0.05 * (2 ** (attempt - 1))))
    assert last_error is not None
    raise RuntimeError(
        f"remote_identity_failed_after_{retries}_attempts:"
        f"{type(last_error).__name__}:{last_error}"
    ) from last_error


def _download_verified_sftp(
    connection: _PersistentAccountConnection,
    remote_path: str,
    destination: Path,
    retries: int,
    expected_identity: dict | None = None,
    max_bytes: int | None = None,
) -> dict:
    """Atomically install bytes only after remote pre/post and local SHA agree."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    initial_identity = expected_identity
    for attempt in range(1, retries + 1):
        part = destination.with_name(
            destination.name
            + f".part.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}"
        )
        try:
            session = connection.get()
            before = (
                dict(initial_identity)
                if initial_identity is not None
                else _remote_file_identity(session, remote_path)
            )
            initial_identity = None
            if max_bytes is not None and int(before["bytes"]) > int(max_bytes):
                raise RuntimeError(
                    f"remote artifact exceeds {int(max_bytes)} byte safety limit"
                )
            session.download_file(remote_path, str(part))
            after = _remote_file_identity(session, remote_path)
            local_size = part.stat().st_size
            local_sha = sha256_file(part)
            if before != after:
                raise RemoteArtifactChanged("remote identity changed during download")
            if local_size != int(after["bytes"]) or local_sha != after["sha256"]:
                raise RemoteArtifactChanged("SFTP bytes do not match remote identity")
            os.replace(part, destination)
            return {
                "bytes": local_size,
                "sha256": local_sha,
                "local_sha256": local_sha,
                "remote_sha256": after["sha256"],
                "remote_bytes": int(after["bytes"]),
                "verified": True,
                "transport": "direct_sftp",
                "attempts": attempt,
            }
        except RemoteArtifactMissing:
            part.unlink(missing_ok=True)
            raise
        except Exception as exc:
            part.unlink(missing_ok=True)
            last_error = exc
            connection.reset()
            if attempt < retries:
                time.sleep(min(0.5, 0.05 * (2 ** (attempt - 1))))
    assert last_error is not None
    raise RuntimeError(
        f"verified_sftp_download_failed_after_{retries}_attempts:"
        f"{type(last_error).__name__}:{last_error}"
    ) from last_error


def _quarantine_stale_local_artifact(task_dir: Path, filename: str) -> dict | None:
    """Move a local byte stream aside when the authoritative remote file is absent.

    Older text-API harvests could leave zero-length or UTF-8-corrupted ``.npy``
    files.  A terminal remote inventory that says the artifact is absent must
    not leave those bytes discoverable under the canonical filename.
    """
    destination = task_dir / filename
    if not destination.exists():
        return None
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError(f"unsafe stale local artifact: {filename}")
    byte_count = destination.stat().st_size
    digest = sha256_file(destination)
    quarantine = task_dir / ".stale"
    quarantine.mkdir(parents=True, exist_ok=True)
    quarantined = quarantine / f"{filename}.{digest}.stale"
    os.replace(destination, quarantined)
    return {
        "file": filename,
        "bytes": byte_count,
        "sha256": digest,
        "quarantined_path": str(quarantined),
        "reason": "authoritative_remote_artifact_absent",
    }


def _safe_lane_output_relative(value: object, task_id: int, lane_id: str) -> str:
    raw = str(value or "")
    if not raw or "\\" in raw:
        raise ValueError("optimizer_output must be a POSIX relative path")
    path = PurePosixPath(raw)
    parts = path.parts
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or parts != ("runs", f"task-{task_id}", lane_id)
    ):
        raise ValueError("optimizer_output escapes the exact task/lane directory")
    return path.as_posix()


def _validate_lane_status(
    lane_status: dict,
    submission: dict,
    plan: dict,
) -> str:
    task_id = int(submission["task_id"])
    if lane_status.get("schema_version") != LANE_SCHEMA:
        raise ValueError("lane status schema mismatch")
    if str(lane_status.get("task_id")) != str(task_id):
        raise ValueError("lane status task ID mismatch")
    if lane_status.get("bundle_id") != plan["bundle_id"]:
        raise ValueError("lane status bundle mismatch")
    if lane_status.get("bundle_manifest_sha256") != plan["bundle_manifest_sha256"]:
        raise ValueError("lane status bundle fingerprint mismatch")
    lane_contract = submission.get("lane") or {}
    lane_id = str(lane_status.get("lane_id") or "")
    if not lane_id or (
        lane_contract.get("lane_id") is not None
        and lane_id != str(lane_contract["lane_id"])
    ):
        raise ValueError("lane status lane ID mismatch")
    for field in ("seed_base", "workers"):
        if lane_contract.get(field) is not None and int(
            lane_status.get(field, -1)
        ) != int(lane_contract[field]):
            raise ValueError(f"lane status {field} mismatch")
    return _safe_lane_output_relative(
        lane_status.get("optimizer_output"), task_id, lane_id
    )


def _prior_verified_cache(
    local_results: Path,
    task_id: int,
    prior: dict | None,
    lane_status: dict,
) -> tuple[dict[str, dict], set[str]]:
    if not prior or prior.get("lane_status") != lane_status:
        return {}, set()
    records: dict[str, dict] = {}
    for record in prior.get("downloaded") or []:
        if not isinstance(record, dict):
            continue
        filename = str(record.get("file") or "")
        if filename not in HARVEST_FILENAMES or filename in records:
            continue
        destination = local_results / f"task-{task_id}" / filename
        try:
            byte_count = int(record.get("bytes"))
        except (TypeError, ValueError):
            continue
        remote_sha = str(record.get("remote_sha256") or "")
        local_sha = str(record.get("local_sha256") or record.get("sha256") or "")
        if (
            record.get("verified") is not True
            or not remote_sha
            or remote_sha != local_sha
            or not destination.is_file()
            or destination.stat().st_size != byte_count
            or sha256_file(destination) != local_sha
        ):
            continue
        records[filename] = copy.deepcopy(record)
    missing = {
        str(item)
        for item in prior.get("missing") or []
        if str(item) in HARVEST_FILENAMES
    }
    return records, missing


def _sealed_prior_inventory(
    prior: dict | None,
    lane_status: dict,
) -> tuple[dict[str, dict], set[str]]:
    """Return the prior remote inventory only when it is complete and sealed."""
    if (
        not prior
        or prior.get("lane_status") != lane_status
        or prior.get("artifacts_complete") is not True
        or prior.get("transport") != "direct_sftp"
    ):
        return {}, set()
    records = {}
    for record in prior.get("downloaded") or []:
        if not isinstance(record, dict):
            return {}, set()
        filename = str(record.get("file") or "")
        remote_sha = str(record.get("remote_sha256") or "")
        try:
            remote_bytes = int(record.get("remote_bytes"))
        except (TypeError, ValueError):
            return {}, set()
        if (
            filename not in HARVEST_FILENAMES
            or filename in records
            or record.get("verified") is not True
            or len(remote_sha) != 64
            or remote_bytes < 0
        ):
            return {}, set()
        records[filename] = {
            "bytes": remote_bytes,
            "sha256": remote_sha,
        }
    missing = {
        str(item)
        for item in prior.get("missing") or []
        if str(item) in HARVEST_FILENAMES
    }
    if set(records) | missing != set(HARVEST_FILENAMES):
        return {}, set()
    return records, missing


def _cached_downloads_valid(
    local_results: Path,
    task_id: int,
    prior: dict | None,
    lane_status: dict,
) -> list[dict] | None:
    """Return a complete locally sound cache; SFTP still rechecks remote SHA."""
    if not prior or prior.get("artifacts_complete") is not True:
        return None
    records, missing = _prior_verified_cache(local_results, task_id, prior, lane_status)
    if set(records) | missing != set(HARVEST_FILENAMES):
        return None
    return [records[name] for name in HARVEST_FILENAMES if name in records]


def _api_json_with_retry(url: str, retries: int) -> dict:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return _api_json(url, timeout=30)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 and not 500 <= exc.code < 600:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(0.5, 0.05 * (2 ** (attempt - 1))))
    assert last_error is not None
    raise last_error


def _empty_harvest_row(task_id: int) -> dict:
    return {
        "task_id": int(task_id),
        "scheduler_status": None,
        "account_name": None,
        "node_name": None,
        "remote_cwd": None,
        "lane_status": None,
        "lane_status_file": None,
        "downloaded": [],
        "missing": [],
        "artifacts_complete": False,
        "cache_reused": False,
        "transport": "direct_sftp",
        "error": None,
    }


def _query_submission(
    submission: dict,
    scheduler_url: str,
    plan: dict,
    retries: int,
) -> dict:
    task_id = int(submission["task_id"])
    row = _empty_harvest_row(task_id)
    try:
        task = _api_json_with_retry(
            scheduler_url.rstrip("/") + f"/api/tasks/{task_id}", retries
        )
        if int(task.get("task_id", task.get("id", -1))) != task_id:
            raise ValueError("scheduler returned a different task ID")
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:
        row["error"] = f"scheduler_query_failed:{type(exc).__name__}:{exc}"
        return row
    row.update(
        {
            "scheduler_status": task.get("status"),
            "account_name": task.get("account_name") or None,
            "node_name": task.get("actual_node_name")
            or task.get("allocation_node_name"),
            "remote_cwd": task.get("remote_cwd") or None,
        }
    )
    expected_cwd = str(plan["remote_bundle"]).rstrip("/")
    actual_cwd = str(task.get("remote_cwd") or "").rstrip("/")
    if actual_cwd != expected_cwd:
        row["error"] = "remote_cwd_fingerprint_mismatch"
    elif not row["account_name"] and str(row["scheduler_status"] or "") not in {
        "queued",
        "attaching",
    }:
        row["error"] = "task_has_no_configured_account"
    return row


def _artifact_evidence_errors(row: dict) -> list[str]:
    status = row.get("lane_status") or {}
    records = {
        str(item.get("file")): item
        for item in row.get("downloaded") or []
        if isinstance(item, dict)
    }
    errors = []
    evidence = status.get("evidence") or {}
    expected_report = str(evidence.get("infeasibility_report_sha256") or "")
    if expected_report:
        actual = str(
            (records.get("infeasibility_report.json") or {}).get("sha256") or ""
        )
        if actual != expected_report:
            errors.append("infeasibility_report_sha256_mismatch")
    return errors


def _harvest_one_submission(
    submission: dict,
    plan: dict,
    local_results: Path,
    prior: dict | None,
    row: dict,
    connection: _PersistentAccountConnection,
    retries: int,
    publish,
) -> dict:
    task_id = int(submission["task_id"])
    task_dir = local_results / f"task-{task_id}"
    status_relative = f"runs/task-{task_id}/lane_status.json"
    status_remote = posixpath.join(plan["remote_bundle"].rstrip("/"), status_relative)
    status_destination = task_dir / "lane_status.json"
    try:
        status_record = _download_verified_sftp(
            connection,
            status_remote,
            status_destination,
            retries,
            max_bytes=LANE_STATUS_MAX_BYTES,
        )
        status_record.update({"file": "lane_status.json", "remote": status_relative})
        status_bytes = status_destination.read_bytes()
        lane_status = json.loads(status_bytes.decode("utf-8"))
        if not isinstance(lane_status, dict):
            raise ValueError("lane status must be a JSON object")
        output = _validate_lane_status(lane_status, submission, plan)
        row["lane_status"] = lane_status
        row["lane_status_file"] = status_record
        row["error"] = None
        publish(row)
    except RemoteArtifactMissing as exc:
        row["error"] = f"lane_status_unavailable:RemoteArtifactMissing:{exc}"
        return row
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        row["error"] = f"lane_status_unavailable:{type(exc).__name__}:{exc}"
        return row

    if lane_status.get("state") not in TERMINAL_REQUIRED_ARTIFACTS:
        return row

    prior_records, prior_missing = _prior_verified_cache(
        local_results, task_id, prior, lane_status
    )
    sealed_records, sealed_missing = _sealed_prior_inventory(prior, lane_status)
    inventory_is_sealed = bool(sealed_records or sealed_missing)
    row["downloaded"] = []
    row["missing"] = []
    row["quarantined_stale_local"] = []
    row.pop("download_warnings", None)
    row.pop("integrity_errors", None)
    all_reused = True
    operational_failures = []
    terminal_mutations = []
    output_round = output.rstrip("/") + "/round_00"
    for index, filename in enumerate(HARVEST_FILENAMES, 1):
        relative = output_round + "/" + filename
        remote = posixpath.join(plan["remote_bundle"].rstrip("/"), relative)
        try:
            identity, identity_attempts = _retry_remote_identity(
                connection, remote, retries
            )
        except RemoteArtifactMissing:
            row["missing"].append(filename)
            quarantined = _quarantine_stale_local_artifact(task_dir, filename)
            if quarantined is not None:
                row["quarantined_stale_local"].append(quarantined)
            if inventory_is_sealed and filename in sealed_records:
                terminal_mutations.append(f"terminal_artifact_disappeared:{filename}")
            if filename not in prior_missing:
                all_reused = False
        except Exception as exc:
            operational_failures.append(f"{filename}:{type(exc).__name__}:{exc}")
            all_reused = False
        else:
            if inventory_is_sealed and filename in sealed_missing:
                terminal_mutations.append(f"terminal_artifact_appeared:{filename}")
                all_reused = False
                row["artifact_progress"] = {
                    "checked": index,
                    "total": len(HARVEST_FILENAMES),
                }
                publish(row)
                continue
            if (
                inventory_is_sealed
                and filename in sealed_records
                and sealed_records[filename] != identity
            ):
                terminal_mutations.append(f"terminal_artifact_changed:{filename}")
                all_reused = False
                row["artifact_progress"] = {
                    "checked": index,
                    "total": len(HARVEST_FILENAMES),
                }
                publish(row)
                continue
            cached = prior_records.get(filename)
            if (
                cached is not None
                and int(cached.get("bytes", -1)) == int(identity["bytes"])
                and str(cached.get("remote_sha256") or "") == identity["sha256"]
            ):
                record = copy.deepcopy(cached)
                record["remote_reverified_at"] = _utc_now()
                record["remote_identity_attempts"] = identity_attempts
            else:
                all_reused = False
                try:
                    record = _download_verified_sftp(
                        connection,
                        remote,
                        task_dir / filename,
                        retries,
                        expected_identity=identity,
                    )
                except Exception as exc:
                    operational_failures.append(
                        f"{filename}:{type(exc).__name__}:{exc}"
                    )
                    row["artifact_progress"] = {
                        "checked": index,
                        "total": len(HARVEST_FILENAMES),
                    }
                    publish(row)
                    continue
                record.update({"file": filename, "remote": relative})
                record["remote_identity_attempts"] = identity_attempts
            row["downloaded"].append(record)
        row["artifact_progress"] = {
            "checked": index,
            "total": len(HARVEST_FILENAMES),
        }
        publish(row)

    required_missing = sorted(
        TERMINAL_REQUIRED_ARTIFACTS[str(lane_status["state"])] & set(row["missing"])
    )
    integrity_errors = _artifact_evidence_errors(row)
    integrity_errors.extend(terminal_mutations)
    if required_missing:
        integrity_errors.append(
            "required_artifacts_missing:" + ",".join(required_missing)
        )
    if operational_failures:
        row["download_warnings"] = operational_failures
    if integrity_errors:
        row["integrity_errors"] = integrity_errors
    accounted = len(row["downloaded"]) + len(row["missing"])
    row["artifacts_complete"] = bool(
        accounted == len(HARVEST_FILENAMES)
        and not operational_failures
        and not integrity_errors
    )
    row["cache_reused"] = bool(row["artifacts_complete"] and all_reused)
    if not row["artifacts_complete"] and row.get("error") is None:
        row["error"] = "artifact_harvest_incomplete"
    return row


def harvest_status(
    plan_path: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    max_workers: int = 8,
    accounts_path: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    retries: int = DEFAULT_HARVEST_RETRIES,
) -> dict:
    plan, _, _ = load_plan(plan_path)
    ledger_path = Path(plan["local_plan_dir"]) / "submissions.json"
    if not ledger_path.is_file():
        raise RuntimeError("no submitted Slurm lanes are recorded for this plan")
    ledger = _load_json(ledger_path)
    local_results = Path(plan["local_plan_dir"]) / "harvest"
    submissions = list(ledger.get("submissions") or [])
    task_ids = [int(item["task_id"]) for item in submissions]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("submission ledger contains duplicate task IDs")
    if not 1 <= int(max_workers) <= 32:
        raise ValueError("max_workers must be 1..32")
    if not 1 <= int(retries) <= 10:
        raise ValueError("retries must be 1..10")
    status_path = local_results / "status.json"
    prior_by_task = {}
    if status_path.is_file():
        prior_summary = _load_json(status_path)
        prior_by_task = {
            int(item["task_id"]): item
            for item in prior_summary.get("lanes") or []
            if isinstance(item, dict) and item.get("task_id") is not None
        }
    rows_by_task = {
        task_id: copy.deepcopy(prior_by_task[task_id])
        for task_id in task_ids
        if task_id in prior_by_task
    }
    checkpoint_lock = threading.Lock()
    checkpoint_sequence = 0

    def checkpoint(complete: bool) -> dict:
        nonlocal checkpoint_sequence
        checkpoint_sequence += 1
        summary = {
            "schema_version": PLAN_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "updated_at": _utc_now(),
            "complete": bool(complete),
            "transport": "direct_sftp",
            "api_concurrency_limit": min(int(max_workers), MAX_HARVEST_API_WORKERS),
            "sftp_connection_limit": min(
                int(max_workers), MAX_HARVEST_SFTP_CONNECTIONS
            ),
            "checkpoint_sequence": checkpoint_sequence,
            "expected_lane_count": len(submissions),
            "observed_lane_count": len(rows_by_task),
            "artifact_complete_count": sum(
                1
                for row in rows_by_task.values()
                if row.get("artifacts_complete") is True
            ),
            "lanes": [
                rows_by_task[task_id] for task_id in task_ids if task_id in rows_by_task
            ],
        }
        atomic_json(status_path, summary)
        return summary

    def publish(row: dict) -> None:
        with checkpoint_lock:
            rows_by_task[int(row["task_id"])] = copy.deepcopy(row)
            checkpoint(False)

    if not submissions:
        with checkpoint_lock:
            return checkpoint(True)

    # A killed harvester must not erase verified lanes merely because their
    # network turn had not arrived yet.
    with checkpoint_lock:
        checkpoint(False)

    api_workers = min(int(max_workers), MAX_HARVEST_API_WORKERS, len(submissions))
    metadata_by_task = {}
    with ThreadPoolExecutor(max_workers=api_workers) as executor:
        futures = {
            executor.submit(
                _query_submission,
                submission,
                scheduler_url,
                plan,
                int(retries),
            ): int(submission["task_id"])
            for submission in submissions
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = _empty_harvest_row(task_id)
                row["error"] = (
                    f"unexpected_scheduler_query_failure:{type(exc).__name__}:{exc}"
                )
            metadata_by_task[task_id] = row
            publish(row)

    by_account = defaultdict(list)
    for submission in submissions:
        task_id = int(submission["task_id"])
        row = metadata_by_task[task_id]
        if row.get("error") is None and row.get("account_name"):
            by_account[str(row["account_name"])].append((submission, row))

    def harvest_account(account_name: str, items: list[tuple[dict, dict]]) -> None:
        try:
            account, ssh_session = _account(
                accounts_path, scheduler_source, account_name
            )
        except Exception as exc:
            for _, row in items:
                row["error"] = f"account_lookup_failed:{type(exc).__name__}:{exc}"
                publish(row)
            return
        connection = _PersistentAccountConnection(account, ssh_session)
        try:
            for submission, row in items:
                task_id = int(submission["task_id"])
                try:
                    result = _harvest_one_submission(
                        submission,
                        plan,
                        local_results,
                        prior_by_task.get(task_id),
                        row,
                        connection,
                        int(retries),
                        publish,
                    )
                except Exception as exc:
                    result = copy.deepcopy(row)
                    result["error"] = (
                        f"unexpected_harvest_failure:{type(exc).__name__}:{exc}"
                    )
                    connection.reset()
                result["ssh_connection_count"] = connection.connection_count
                publish(result)
        finally:
            connection.close()

    if by_account:
        sftp_workers = min(
            int(max_workers), MAX_HARVEST_SFTP_CONNECTIONS, len(by_account)
        )
        with ThreadPoolExecutor(max_workers=sftp_workers) as executor:
            futures = {
                executor.submit(harvest_account, account_name, items): account_name
                for account_name, items in sorted(by_account.items())
            }
            for future in as_completed(futures):
                account_name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    for _, row in by_account[account_name]:
                        row["error"] = (
                            f"account_harvest_failed:{type(exc).__name__}:{exc}"
                        )
                        publish(row)
    with checkpoint_lock:
        return checkpoint(True)


def _comma_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--deployment", type=Path, default=DEFAULT_DEPLOYMENT)
    plan_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    plan_parser.add_argument("--generation", type=Path, default=DEFAULT_GENERATION)
    plan_parser.add_argument("--quality-status", type=Path, default=DEFAULT_QUALITY)
    plan_parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    plan_parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    plan_parser.add_argument("--warm-artifact", type=Path)
    plan_parser.add_argument("--warm-contract", type=Path)
    plan_parser.add_argument(
        "--warm-kind", choices=("standard_fea", "cross_run"), default="standard_fea"
    )

    stage_parser = sub.add_parser("stage")
    stage_parser.add_argument("--plan", type=Path, required=True)
    stage_parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    stage_parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    stage_parser.add_argument("--staging-account", default="harry261")
    stage_parser.add_argument(
        "--resume-incoming",
        help="exact preserved .incoming-<bundle>-<hex> directory to resume",
    )
    stage_parser.add_argument("--apply", action="store_true")

    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--plan", type=Path, required=True)
    submit_parser.add_argument("--seed-bases", type=_comma_ints, required=True)
    submit_parser.add_argument("--restarts", type=int, default=1)
    submit_parser.add_argument("--population", type=int, default=320)
    submit_parser.add_argument("--max-generations", type=int, default=600)
    submit_parser.add_argument("--workers", type=int, default=1)
    submit_parser.add_argument("--cpus", type=int, default=1)
    submit_parser.add_argument("--memory-mb", type=int, default=32768)
    submit_parser.add_argument("--timeout-seconds", type=int, default=86400)
    submit_parser.add_argument("--priority", type=int, default=0)
    submit_parser.add_argument("--max-workers-per-node", type=int, default=4)
    submit_parser.add_argument("--lane-prefix", default="lane")
    submit_parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    submit_parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    submit_parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    submit_parser.add_argument("--staging-account", default="harry261")
    submit_parser.add_argument("--apply", action="store_true")

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--plan", type=Path, required=True)
    status_parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    status_parser.add_argument("--max-workers", type=int, default=8)
    status_parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    status_parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    status_parser.add_argument("--retries", type=int, default=DEFAULT_HARVEST_RETRIES)

    args = parser.parse_args()
    if args.command == "plan":
        plan, manifest = build_plan(
            args.deployment,
            args.dataset,
            args.generation,
            args.quality_status,
            args.local_root,
            args.remote_root,
            warm_artifact=args.warm_artifact,
            warm_contract=args.warm_contract,
            warm_kind=args.warm_kind,
        )
        output = {
            "plan": plan,
            "manifest_summary": {
                "file_count": len(manifest["files"]),
                "byte_count": sum(
                    int(item["size"]) for item in manifest["files"].values()
                ),
                "model_artifact_count": len(manifest["model_artifacts_sha256"]),
            },
        }
    elif args.command == "stage":
        output = stage_bundle(
            args.plan,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            staging_account=args.staging_account,
            resume_incoming=args.resume_incoming,
            apply=args.apply,
        )
    elif args.command == "submit":
        output = submit_lanes(
            args.plan,
            args.seed_bases,
            args.restarts,
            args.population,
            args.max_generations,
            args.workers,
            scheduler_url=args.scheduler_url,
            cpus=args.cpus,
            memory_mb=args.memory_mb,
            timeout_seconds=args.timeout_seconds,
            priority=args.priority,
            max_workers_per_node=args.max_workers_per_node,
            lane_prefix=args.lane_prefix,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            staging_account=args.staging_account,
            apply=args.apply,
        )
    else:
        output = harvest_status(
            args.plan,
            scheduler_url=args.scheduler_url,
            max_workers=args.max_workers,
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            retries=args.retries,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
