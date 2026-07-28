"""Seal the current 8010 runtime and verify a completed monitor transition.

Both commands are read-only with respect to the service and campaign.  They
record enough process identity to execute an exact rollback without guessing.
No process is stopped or started by this module.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from .release_canary import (
    CanaryFailure,
    _authenticate_blocker_hpo_v2_status,
    _authenticate_current7_condition_indexes,
    _authenticate_current7_index,
    _canonical_json,
    _get_json,
    _get_text,
    _git,
    _pyvenv_configuration,
    _sha256_bytes,
    _sha256_file,
    _source_identity,
    _verify_data_api,
    _verify_blocker_hpo_v2_api,
    _verify_current7_api,
    _verify_current7_condition_searches,
    _verify_deadline_design_api,
    _verify_local_gui_launches,
    _wait_for_stable_nsga_generation,
    _write_new_json,
)


FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
RUNTIME_ENVIRONMENT_KEYS = frozenset({
    "LOCALAPPDATA",
    "PYTHONPATH",
    "MFT_MONITOR_OPERATOR_HOSTS",
    "MFT_MONITOR_OPERATOR_NETWORKS",
    "MFT_MONITOR_ROOT",
    "MFT_MONITOR_DISABLE_HISTORY",
    "MFT_PIPELINE_ROOT",
    "MFT_TIER1_ROLLING_INDEX",
    "MFT_TIER1_CURRENT7_INDEX",
    "MFT_TIER1_CURRENT7_CONDITION_INDEXES",
    "MFT_BLOCKER_HPO_V2_STATUS",
    "MFT_NSGA_SLURM_OFFLOAD_STATUS",
    "MFT_NSGA_UI_STATUS_ROOTS",
    "MFT_SCHEDULER_URL",
    "MFT_SCHEDULER_PROJECT",
    "MFT_MONITOR_TASK_PREFIX",
    "MFT_SCHEDULER_TIMEOUT",
    "MFT_SCHEDULER_OPTIONAL_TIMEOUT",
    "MFT_SCHEDULER_POOL_TIMEOUT",
    "MFT_LOCAL_AEDT_GUI_RUNTIME",
    "MFT_LOCAL_AEDT_PYTHON",
    "MFT_LOCAL_AEDT_SOLVER_ROOT",
    "MFT_LOCAL_AEDT_SOLVER_REVISION",
    "MFT_LOCAL_AEDT_RUNNER_SHA256",
    "MFT_LOCAL_AEDT_GUI_MAX_ACTIVE",
    "MFT_DEADLINE_DESIGN_PUBLICATION",
    "MFT_DEADLINE_DESIGN_PUBLICATION_SHA256",
})


def _verify_optional_runtime_artifacts(
    environment: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Bind optional mutable monitor inputs to their sealed runtime paths."""

    verified: dict[str, dict[str, Any] | None] = {}
    for variable, manifest_key, authenticator in (
        (
            "MFT_TIER1_CURRENT7_INDEX",
            "current7_index",
            _authenticate_current7_index,
        ),
        (
            "MFT_BLOCKER_HPO_V2_STATUS",
            "blocker_hpo_v2_status",
            _authenticate_blocker_hpo_v2_status,
        ),
    ):
        declared = manifest.get(manifest_key)
        observed_text = str(environment.get(variable) or "").strip()
        if declared is None:
            if observed_text:
                raise CanaryFailure(
                    f"deployed {variable} was not declared by the release manifest"
                )
            verified[manifest_key] = None
            continue
        if not isinstance(declared, dict):
            raise CanaryFailure(
                f"release manifest {manifest_key} identity is malformed"
            )
        expected_text = str(declared.get("path") or "").strip()
        if not observed_text or not expected_text:
            raise CanaryFailure(f"deployed {variable} path is missing")
        try:
            observed = Path(observed_text).resolve(strict=True)
            expected = Path(expected_text).resolve(strict=True)
        except OSError as exc:
            raise CanaryFailure(
                f"deployed {variable} path is unavailable"
            ) from exc
        if observed != expected:
            raise CanaryFailure(
                f"deployed {variable} differs from the release manifest"
            )
        identity = authenticator(observed)
        if identity.get("path") != str(expected):
            raise CanaryFailure(f"deployed {variable} resolved identity drifted")
        verified[manifest_key] = identity
    declared_conditions = manifest.get("current7_condition_indexes") or []
    if not isinstance(declared_conditions, list) or not all(
        isinstance(item, dict) and str(item.get("path") or "").strip()
        for item in declared_conditions
    ):
        raise CanaryFailure(
            "release manifest current7 condition identities are malformed"
        )
    observed_condition_text = str(
        environment.get("MFT_TIER1_CURRENT7_CONDITION_INDEXES") or ""
    ).strip()
    observed_condition_paths = [
        Path(item.strip())
        for item in observed_condition_text.split(";")
        if item.strip()
    ]
    if len(observed_condition_paths) != len(declared_conditions):
        raise CanaryFailure("deployed current7 condition index count diverged")
    for position, (observed_path, declared) in enumerate(
        zip(observed_condition_paths, declared_conditions, strict=True)
    ):
        try:
            observed = observed_path.resolve(strict=True)
            expected = Path(str(declared["path"])).resolve(strict=True)
        except OSError as exc:
            raise CanaryFailure(
                f"deployed current7 condition index {position} is unavailable"
            ) from exc
        if observed != expected:
            raise CanaryFailure(
                f"deployed current7 condition index {position} path diverged"
            )
    verified["current7_condition_indexes"] = (
        _authenticate_current7_condition_indexes(observed_condition_paths)
    )
    declared_deadline = manifest.get("deadline_design")
    observed_deadline_path = str(
        environment.get("MFT_DEADLINE_DESIGN_PUBLICATION") or ""
    ).strip()
    observed_deadline_sha = str(
        environment.get("MFT_DEADLINE_DESIGN_PUBLICATION_SHA256") or ""
    ).strip().lower()
    if declared_deadline is None:
        if observed_deadline_path or observed_deadline_sha:
            raise CanaryFailure(
                "deployed deadline publication was not declared by the "
                "release manifest"
            )
        verified["deadline_design"] = None
    else:
        if not isinstance(declared_deadline, dict):
            raise CanaryFailure(
                "release manifest deadline design identity is malformed"
            )
        expected_path_text = str(
            declared_deadline.get("publication_path") or ""
        ).strip()
        expected_sha = str(
            declared_deadline.get("publication_sha256") or ""
        ).strip().lower()
        if not (
            observed_deadline_path
            and observed_deadline_sha
            and expected_path_text
            and expected_sha
        ):
            raise CanaryFailure(
                "deployed deadline publication path/hash is missing"
            )
        try:
            observed_path = Path(observed_deadline_path).resolve(strict=True)
            expected_path = Path(expected_path_text).resolve(strict=True)
        except OSError as exc:
            raise CanaryFailure(
                "deployed deadline publication path is unavailable"
            ) from exc
        if observed_path != expected_path:
            raise CanaryFailure(
                "deployed deadline publication path differs from the "
                "release manifest"
            )
        if observed_deadline_sha != expected_sha:
            raise CanaryFailure(
                "deployed deadline publication hash differs from the "
                "release manifest"
            )
        from .deadline_design import load_publication

        authenticated = load_publication(observed_path, observed_deadline_sha)
        generation = authenticated["generation"]
        candidate = authenticated["candidate"]
        deadline_identity = {
            "verified": True,
            "publication_path": str(observed_path),
            "publication_sha256": observed_deadline_sha,
            "generation_id": generation["id"],
            "candidate_id": candidate["id"],
            "cooling_variant": candidate.get("cooling_variant"),
            "fan_config": candidate.get("fan_config"),
            "fan_velocity_m_s": candidate.get("fan_velocity_m_s"),
            "solver_revision": candidate.get("solver_revision"),
            "thermal_pad_material_policy": candidate.get(
                "thermal_pad_material_policy"
            ),
            "thermal_pad_native_readback_contract_version": candidate.get(
                "thermal_pad_native_readback_contract_version"
            ),
            "thermal_pad_native_readback_attested": candidate.get(
                "thermal_pad_native_readback_attested"
            ),
            "thermal_pad_native_thermal_conductivity_W_mK": candidate.get(
                "thermal_pad_native_thermal_conductivity_W_mK"
            ),
            "thermal_pad_native_electrical_conductivity_S_m": candidate.get(
                "thermal_pad_native_electrical_conductivity_S_m"
            ),
            "half_magnetizing_resonance_Hz": candidate.get(
                "pred_f_res_min_screen_Hz"
            ),
            "half_magnetizing_resonance_minimum_Hz": candidate.get(
                "resonance_minimum_required_Hz"
            ),
            "half_magnetizing_resonance_margin_Hz": candidate.get(
                "resonance_margin_Hz"
            ),
            "full_actual_hard_pass": authenticated["publication"].get(
                "full_actual_hard_pass"
            ) is True,
            "gui_build_eligible": candidate.get("gui_build_eligible") is True,
            "gui_solve_eligible": candidate.get("gui_solve_eligible") is True,
        }
        if candidate.get("validation_state") is not None:
            deadline_identity.update(
                {
                    "complete_full_pending": (
                        candidate.get("complete_full_pending") is True
                    ),
                    "standard_corroboration_pass": (
                        candidate.get("standard_corroboration_pass") is True
                    ),
                    "validation_state": candidate.get("validation_state"),
                    "validation_badge": candidate.get("validation_badge"),
                    "hard_spec_authority": candidate.get(
                        "hard_spec_authority"
                    ),
                    "final_design_approved": (
                        candidate.get("final_design_approved") is True
                    ),
                }
            )
        for key, value in deadline_identity.items():
            if declared_deadline.get(key) != value:
                raise CanaryFailure(
                    "deployed deadline publication identity differs from "
                    f"the release manifest: {key}"
                )
        verified["deadline_design"] = deadline_identity
    return verified


def _wait_for_runtime_nsga_generation(
    base_url: str,
    optional_artifacts: dict[str, dict[str, Any] | None],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Verify the authority lane authenticated from the live environment."""
    return _wait_for_stable_nsga_generation(
        base_url,
        expected_current7_index=optional_artifacts.get("current7_index"),
        expected_current7_condition_indexes=optional_artifacts.get(
            "current7_condition_indexes"
        ),
    )


def _read_sealed_json(path: Path, expected_kind: str) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise CanaryFailure(f"sealed JSON is not a regular file: {resolved}")
    sha = _sha256_file(resolved, 4 * 1024 * 1024)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    try:
        tokens = sidecar.read_text(encoding="ascii").strip().split()
    except OSError as exc:
        raise CanaryFailure(f"sealed JSON sidecar is unavailable: {sidecar}") from exc
    if len(tokens) != 2 or tokens[0].lower() != sha or tokens[1] != resolved.name:
        raise CanaryFailure(f"sealed JSON sidecar mismatch: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise CanaryFailure(f"sealed JSON is invalid: {resolved}") from exc
    if not isinstance(value, dict) or value.get("kind") != expected_kind:
        raise CanaryFailure(f"sealed JSON kind mismatch: {resolved}")
    return value, sha


def _write_sidecar(path: Path, sha: str) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        with sidecar.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(f"{sha}  {path.name}\n")
    except FileExistsError as exc:
        raise CanaryFailure(f"refusing to overwrite seal sidecar: {sidecar}") from exc
    return sidecar


def _listener_process(port: int) -> dict[str, Any]:
    import psutil

    pids = {
        connection.pid
        for connection in psutil.net_connections(kind="tcp")
        if connection.pid is not None
        and connection.status == psutil.CONN_LISTEN
        and connection.laddr
        and int(connection.laddr.port) == port
    }
    if len(pids) != 1:
        raise CanaryFailure(
            f"expected exactly one TCP listener PID on {port}; found {sorted(pids)}"
        )
    pid = int(next(iter(pids)))
    try:
        process = psutil.Process(pid)
        executable = Path(process.exe()).resolve(strict=True)
        command = [str(item) for item in process.cmdline()]
        cwd = Path(process.cwd()).resolve(strict=True)
        create_time = float(process.create_time())
        name = process.name()
        raw_environment = process.environ()
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError) as exc:
        raise CanaryFailure(f"cannot inspect listener PID {pid}: {exc}") from exc
    if not command:
        raise CanaryFailure(f"listener PID {pid} has an empty command line")
    return {
        "pid": pid,
        "name": name,
        "create_time": create_time,
        "executable": str(executable),
        "executable_sha256": _sha256_file(executable, 64 * 1024 * 1024),
        "cwd": str(cwd),
        "command": command,
        "command_sha256": _sha256_bytes(_canonical_json(command)),
        "environment": {
            key: raw_environment[key]
            for key in sorted(RUNTIME_ENVIRONMENT_KEYS)
            if key in raw_environment
        },
        "port": port,
    }


def _require_process_location(
    process: dict[str, Any], source_root: Path, venv_root: Path,
    runtime_process_identity: dict[str, Any] | None = None,
) -> None:
    source = source_root.resolve(strict=True)
    venv = venv_root.resolve(strict=True)
    raw_actual_python = Path(str(process["executable"]))
    if raw_actual_python.is_symlink() or not raw_actual_python.is_file():
        raise CanaryFailure("listener executable is not a regular file")
    actual_python = raw_actual_python.resolve(strict=True)
    python_candidates = [
        candidate.resolve(strict=True)
        for candidate in (
            venv / "Scripts" / "python.exe",
            venv / "python.exe",
        )
        if candidate.is_file()
    ]
    allowed_executables = {
        candidate: _sha256_file(candidate, 64 * 1024 * 1024)
        for candidate in python_candidates
    }
    if (
        runtime_process_identity is None
        and actual_python not in allowed_executables
    ):
        # On Windows a venv launcher starts the base interpreter image, so
        # psutil reports ``pyvenv.cfg: executable`` rather than
        # ``venv\Scripts\python.exe``.  The rollback seal must authenticate
        # that declared pair instead of rejecting the healthy live process.
        raw_pyvenv_path = venv / "pyvenv.cfg"
        if raw_pyvenv_path.is_symlink() or not raw_pyvenv_path.is_file():
            raise CanaryFailure("rollback pyvenv.cfg is not a regular file")
        pyvenv = _pyvenv_configuration(raw_pyvenv_path)
        raw_base_interpreter = Path(pyvenv["executable"])
        if (
            raw_base_interpreter.is_symlink()
            or not raw_base_interpreter.is_file()
        ):
            raise CanaryFailure(
                "rollback pyvenv base interpreter is not a regular file"
            )
        base_interpreter = raw_base_interpreter.resolve(strict=True)
        allowed_executables[base_interpreter] = _sha256_file(
            base_interpreter, 64 * 1024 * 1024
        )
    if runtime_process_identity is not None:
        if not isinstance(runtime_process_identity, dict):
            raise CanaryFailure("sealed runtime process identity is malformed")
        pyvenv_identity = runtime_process_identity.get("pyvenv")
        launcher_identity = runtime_process_identity.get("venv_launcher")
        base_identity = runtime_process_identity.get("base_interpreter")
        if not all(isinstance(item, dict) for item in (
            pyvenv_identity, launcher_identity, base_identity,
        )):
            raise CanaryFailure("sealed pyvenv executable pair is incomplete")
        raw_pyvenv_path = Path(str(pyvenv_identity.get("path") or ""))
        if raw_pyvenv_path.is_symlink() or not raw_pyvenv_path.is_file():
            raise CanaryFailure("sealed pyvenv.cfg is not a regular file")
        pyvenv_path = raw_pyvenv_path.resolve(strict=True)
        expected_pyvenv_path = (venv / "pyvenv.cfg").resolve(strict=True)
        if pyvenv_path != expected_pyvenv_path:
            raise CanaryFailure("sealed pyvenv.cfg is outside candidate venv")
        if _sha256_file(pyvenv_path, 256 * 1024) != str(
            pyvenv_identity.get("sha256") or ""
        ).lower():
            raise CanaryFailure("sealed pyvenv.cfg hash mismatch")
        pyvenv = _pyvenv_configuration(pyvenv_path)
        sealed_home = Path(
            str(pyvenv_identity.get("home") or "")
        ).resolve(strict=True)
        sealed_base_executable = Path(
            str(pyvenv_identity.get("executable") or "")
        ).resolve(strict=True)
        if (
            Path(pyvenv["home"]).resolve(strict=True) != sealed_home
            or Path(pyvenv["executable"]).resolve(strict=True)
            != sealed_base_executable
        ):
            raise CanaryFailure("sealed pyvenv.cfg home/executable mismatch")
        raw_launcher = Path(str(launcher_identity.get("path") or ""))
        raw_base_interpreter = Path(str(base_identity.get("path") or ""))
        if (
            raw_launcher.is_symlink() or not raw_launcher.is_file()
            or raw_base_interpreter.is_symlink()
            or not raw_base_interpreter.is_file()
        ):
            raise CanaryFailure("sealed pyvenv executable pair is not regular")
        launcher = raw_launcher.resolve(strict=True)
        base_interpreter = raw_base_interpreter.resolve(strict=True)
        if launcher not in python_candidates:
            raise CanaryFailure("sealed venv launcher is outside candidate venv")
        if base_interpreter != sealed_base_executable:
            raise CanaryFailure("sealed base interpreter differs from pyvenv.cfg")
        observed_process_executable = Path(str(
            runtime_process_identity.get("observed_process_executable") or ""
        )).resolve(strict=True)
        if observed_process_executable not in {launcher, base_interpreter}:
            raise CanaryFailure(
                "canary-observed process image is outside sealed pyvenv pair"
            )
        if actual_python != observed_process_executable:
            raise CanaryFailure(
                "listener process image differs from canary-observed image"
            )
        for candidate, identity, label in (
            (launcher, launcher_identity, "venv launcher"),
            (base_interpreter, base_identity, "base interpreter"),
        ):
            if candidate.is_symlink() or not candidate.is_file():
                raise CanaryFailure(f"sealed {label} is not a regular file")
            expected_sha = str(identity.get("sha256") or "").lower()
            if _sha256_file(candidate, 64 * 1024 * 1024) != expected_sha:
                raise CanaryFailure(f"sealed {label} hash mismatch")
            allowed_executables[candidate] = expected_sha
    if actual_python not in allowed_executables:
        raise CanaryFailure(
            "listener executable does not belong to the rollback/candidate "
            f"environment: {actual_python} not in {list(allowed_executables)}"
        )
    if process.get("executable_sha256") != allowed_executables[actual_python]:
        raise CanaryFailure("listener executable hash differs from sealed runtime")
    cwd = Path(str(process["cwd"])).resolve(strict=True)
    command_text = "\n".join(str(item) for item in process["command"]).casefold()
    if cwd != source and str(source).casefold() not in command_text:
        raise CanaryFailure("listener command/cwd does not identify the expected source root")


def _verify_browser_bundle(
    base_url: str, *, require_new_polling_contract: bool = True
) -> dict[str, Any]:
    html = _get_text(base_url, "/", "text/html", 30)
    javascript = _get_text(base_url, "/static/app.js", "text/javascript", 30)
    if "/static/app.js" not in html:
        raise CanaryFailure("served dashboard does not load /static/app.js")
    endpoints = (
        "/api/dashboard",
        "/api/nsga2/progress",
        "/api/local-aedt-gui/launches",
    )
    missing = [endpoint for endpoint in endpoints if endpoint not in javascript]
    if missing and require_new_polling_contract:
        raise CanaryFailure(
            "served browser bundle does not poll: " + ", ".join(missing)
        )
    return {
        "dashboard_html_sha256": _sha256_bytes(html.encode("utf-8")),
        "app_js_sha256": _sha256_bytes(javascript.encode("utf-8")),
        "polled_endpoints": list(endpoints),
        "missing_polled_endpoints": missing,
        "new_polling_contract_verified": not missing,
    }


def _python_identity(executable: Path) -> dict[str, Any]:
    script = """
import importlib.metadata as metadata
import json, platform, sys
def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
print(json.dumps({
    "executable": sys.executable,
    "python_version": platform.python_version(),
    "packages": {
        name: version(name) for name in ("pyarrow", "fastapi", "uvicorn")
    },
}, separators=(",", ":")))
"""
    completed = subprocess.run(
        [str(executable), "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise CanaryFailure(
            "rollback Python identity probe failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[:500]
        )
    try:
        value = json.loads(completed.stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        raise CanaryFailure("rollback Python identity probe returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise CanaryFailure("rollback Python identity probe returned no object")
    return value


def _source_revision(source_root: Path) -> str:
    revision = _git(source_root, "rev-parse", "HEAD").lower()
    if not FULL_SHA.fullmatch(revision):
        raise CanaryFailure("rollback source revision is malformed")
    dirty = _git(
        source_root, "status", "--porcelain", "--untracked-files=all"
    )
    if dirty.strip():
        raise CanaryFailure("rollback source checkout is dirty")
    return revision


def _seal(args: argparse.Namespace) -> int:
    manifest, manifest_sha = _read_sealed_json(
        args.deployment_manifest,
        "mft-monitor-production-deployment-manifest",
    )
    if manifest.get("deploy_authorized") is not True:
        raise CanaryFailure("candidate deployment manifest is not authorized")
    rollback_source = args.rollback_source_root.resolve(strict=True)
    rollback_venv = args.rollback_venv_root.resolve(strict=True)
    process = _listener_process(args.port)
    _require_process_location(process, rollback_source, rollback_venv)
    health = _get_json(
        args.base_url, "/healthz", 30, require_utf8_charset=False
    )
    if health.get("status") != "ok":
        raise CanaryFailure("current runtime health is not ok")
    bundle = _verify_browser_bundle(
        args.base_url, require_new_polling_contract=False
    )
    now = datetime.now(timezone.utc)
    seal_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    value = {
        "schema_version": 1,
        "kind": "mft-monitor-runtime-transition-pre-seal",
        "seal_id": seal_id,
        "sealed_at": now.isoformat(timespec="microseconds"),
        "base_url": args.base_url,
        "candidate_deployment_manifest": str(
            args.deployment_manifest.resolve(strict=True)
        ),
        "candidate_deployment_manifest_sha256": manifest_sha,
        "candidate_revision": manifest.get("candidate_revision"),
        "old_runtime": process,
        "rollback": {
            "source_root": str(rollback_source),
            "source_revision": _source_revision(rollback_source),
            "venv_root": str(rollback_venv),
            "python_executable": process["executable"],
            "command": process["command"],
            "cwd": process["cwd"],
            "environment": process["environment"],
            "runtime": _python_identity(Path(str(process["executable"]))),
        },
        "browser_bundle": bundle,
    }
    path = args.evidence_dir / f"runtime_pre_seal_{seal_id}.json"
    sha = _write_new_json(path, value)
    _write_sidecar(path, sha)
    print(json.dumps({
        "sealed": True,
        "path": str(path),
        "sha256": sha,
        "old_pid": process["pid"],
        "rollback_command": process["command"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    preseal, preseal_sha = _read_sealed_json(
        args.pre_seal, "mft-monitor-runtime-transition-pre-seal"
    )
    manifest, manifest_sha = _read_sealed_json(
        args.deployment_manifest,
        "mft-monitor-production-deployment-manifest",
    )
    if preseal.get("candidate_deployment_manifest_sha256") != manifest_sha:
        raise CanaryFailure("pre-seal and deployment manifest identities diverge")
    process = _listener_process(args.port)
    if process["pid"] == preseal.get("old_runtime", {}).get("pid"):
        raise CanaryFailure("8010 listener PID did not change during deployment")
    source_root = Path(str(manifest.get("candidate_source_root") or ""))
    venv_root = Path(str(manifest.get("venv_root") or ""))
    runtime_process_identity = manifest.get("runtime_process_identity")
    runtime_process_identity = (
        runtime_process_identity
        if isinstance(runtime_process_identity, dict) else {}
    )
    launcher_identity = runtime_process_identity.get("venv_launcher")
    launcher_identity = (
        launcher_identity if isinstance(launcher_identity, dict) else {}
    )
    if Path(str(manifest.get("python_executable") or "")).resolve(
        strict=True
    ) != Path(str(launcher_identity.get("path") or "")).resolve(strict=True):
        raise CanaryFailure(
            "manifest Python executable differs from sealed venv launcher"
        )
    _require_process_location(
        process, source_root, venv_root,
        runtime_process_identity,
    )
    source_identity = _source_identity(
        source_root, str(manifest.get("candidate_revision") or "")
    )
    if source_identity.get("monitoring_tree_sha256") != manifest.get(
        "monitoring_tree_sha256"
    ):
        raise CanaryFailure("deployed monitoring tree differs from manifest")
    environment = process.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    for variable, manifest_key in (
        ("MFT_MONITOR_ROOT", "regression_root"),
        ("MFT_PIPELINE_ROOT", "pipeline_root"),
        ("MFT_TIER1_ROLLING_INDEX", "rolling_index"),
    ):
        try:
            observed = Path(str(environment.get(variable) or "")).resolve(
                strict=True
            )
            expected = Path(str(manifest.get(manifest_key) or "")).resolve(
                strict=True
            )
        except OSError as exc:
            raise CanaryFailure(
                f"deployed {variable} path is unavailable"
            ) from exc
        if observed != expected:
            raise CanaryFailure(
                f"deployed {variable} differs from the release manifest"
            )
    optional_artifacts = _verify_optional_runtime_artifacts(
        environment, manifest
    )

    bundle = _verify_browser_bundle(args.base_url)
    health = _get_json(args.base_url, "/healthz", 30)
    if health.get("status") != "ok":
        raise CanaryFailure("new runtime health is not ok")
    data_payload = _get_json(args.base_url, "/api/data", 240)
    dashboard_payload = (
        _get_json(args.base_url, "/api/dashboard", 240)
        if optional_artifacts["blocker_hpo_v2_status"] is not None
        else None
    )
    (
        nsga_payload, progress_payload, nsga, progress,
        nsga_stability_gate,
    ) = _wait_for_runtime_nsga_generation(
        args.base_url, optional_artifacts
    )
    launches_payload = _get_json(
        args.base_url, "/api/local-aedt-gui/launches", 60
    )
    cohort, parquet_read = _verify_data_api(
        data_payload, Path(str(manifest.get("pipeline_root") or ""))
    )
    local_gui = _verify_local_gui_launches(launches_payload)
    current7 = (
        _verify_current7_api(
            nsga_payload, optional_artifacts["current7_index"]
        )
        if optional_artifacts["current7_index"] is not None
        else None
    )
    current7_conditions = (
        _verify_current7_condition_searches(
            nsga_payload,
            nsga_stability_gate.get("current7_condition_indexes")
            or optional_artifacts["current7_condition_indexes"],
        )
        if optional_artifacts["current7_condition_indexes"]
        else []
    )
    blocker_hpo_v2 = (
        _verify_blocker_hpo_v2_api(
            dashboard_payload,
            optional_artifacts["blocker_hpo_v2_status"],
        )
        if dashboard_payload is not None
        and optional_artifacts["blocker_hpo_v2_status"] is not None
        else None
    )
    deadline_design = None
    deadline_identity = optional_artifacts["deadline_design"]
    if deadline_identity is not None:
        generation_id = str(deadline_identity.get("generation_id") or "")
        deadline_detail_payload = _get_json(
            args.base_url,
            f"/api/nsga2/generations/{generation_id}",
            120,
        )
        deadline_design = _verify_deadline_design_api(
            nsga_payload,
            deadline_detail_payload,
            publication_path=Path(
                str(deadline_identity["publication_path"])
            ),
            publication_sha256=str(
                deadline_identity["publication_sha256"]
            ),
        )
        if deadline_design != deadline_identity:
            raise CanaryFailure(
                "deployed deadline design API differs from the authenticated "
                "publication"
            )

    now = datetime.now(timezone.utc)
    seal_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    value = {
        "schema_version": 1,
        "kind": "mft-monitor-runtime-transition-post-verification",
        "seal_id": seal_id,
        "verified_at": now.isoformat(timespec="microseconds"),
        "passed": True,
        "base_url": args.base_url,
        "pre_seal_path": str(args.pre_seal.resolve(strict=True)),
        "pre_seal_sha256": preseal_sha,
        "deployment_manifest_path": str(
            args.deployment_manifest.resolve(strict=True)
        ),
        "deployment_manifest_sha256": manifest_sha,
        "new_runtime": process,
        "browser_bundle": bundle,
        "strict_cohort": cohort,
        "parquet_read_canary": parquet_read,
        "nsga2": nsga,
        "nsga2_progress": progress,
        "nsga_stability_gate": nsga_stability_gate,
        "optional_runtime_artifacts": optional_artifacts,
        "tier1_current7": current7,
        "tier1_current7_conditions": current7_conditions,
        "blocker_hpo_v2": blocker_hpo_v2,
        "deadline_design": deadline_design,
        "local_aedt_gui_launches": local_gui,
    }
    path = args.evidence_dir / f"runtime_post_verify_{seal_id}.json"
    sha = _write_new_json(path, value)
    _write_sidecar(path, sha)
    print(json.dumps({
        "passed": True,
        "path": str(path),
        "sha256": sha,
        "new_pid": process["pid"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--base-url", default="http://127.0.0.1:8010")
    seal.add_argument("--port", type=int, default=8010)
    seal.add_argument("--rollback-source-root", type=Path, required=True)
    seal.add_argument("--rollback-venv-root", type=Path, required=True)
    seal.add_argument("--deployment-manifest", type=Path, required=True)
    seal.add_argument("--evidence-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--base-url", default="http://127.0.0.1:8010")
    verify.add_argument("--port", type=int, default=8010)
    verify.add_argument("--pre-seal", type=Path, required=True)
    verify.add_argument("--deployment-manifest", type=Path, required=True)
    verify.add_argument("--evidence-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        return _seal(args) if args.command == "seal" else _verify(args)
    except Exception as exc:
        print(json.dumps({
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
