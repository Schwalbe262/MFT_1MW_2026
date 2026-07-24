#!/usr/bin/env python3
"""Prepare an offline, immutable fixed-boundary Standard/Full replay handoff.

The helper never contacts the Scheduler, uploads files, submits work, or
promotes UI state.  It authenticates the task-95067 candidate artifacts,
re-decodes the candidate with the authoritative fan/pad/TIM contract, and
fails closed before building a bundle or payload when the authoritative
geometric prescreen is not satisfied.  Eligible payloads still require their
remote release/output roots to be staged and ACL-probed independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module.fixed_boundary_contract import (  # noqa: E402
    FIXED_BOUNDARY_CONTRACT_SCHEMA,
    FIXED_BOUNDARY_CONTRACT_SHA256,
    FIXED_CORE_PLATE_PAD_THICKNESS_MM,
    FIXED_FAN_VELOCITY_M_S,
    FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK,
    FIXED_WCP_PAD_THICKNESS_MM,
    attest_fixed_boundary,
    evaluate_fixed_boundary,
)
from module.input_parameter_260706 import (  # noqa: E402
    KEYS,
    create_input_parameter,
    get_drawing_default_params,
    validation_check,
)
from regression_260707.optimization.design_summary import (  # noqa: E402
    B_AREA_BASIS_GROSS_WITH_LAMINATION,
    design_analytical_b_field_t,
)
from regression_260707.optimization.geometry_metrics import (  # noqa: E402
    bounding_box_lit,
)


SCHEMA = "mft-fixed-thermal-replay-handoff-v1"
PROFILE_SCHEMA = "mft-fixed-thermal-replay-profile-v1"
PAYLOAD_SCHEMA = "mft-fixed-thermal-replay-task-v1"
SOURCE_FULL_EM_TASK_ID = 95067
SOURCE_PHYSICAL_CANDIDATE_DIGEST = (
    "504ee1789afbcf517891a4f94043f8d20077ba7d5617e47e9ff0ce686a9ce6a9"
)
SOURCE_ARTIFACTS = {
    "standard_payload": {
        "sha256": "84d6a97b8053b702418fd41c05471d0292db5f1de9e01b985b90bdfa222c4bd7",
        "size_bytes": 7469,
    },
    "followup_profile": {
        "sha256": "b574f0d214c9b8f0a54bd35003dffac0a72c5fea37b68b5ec713391c7efb1d4b",
        "size_bytes": 1161,
    },
    "sweep_result": {
        "sha256": "7805dc73ae8aaf4c48e81777cd7121b0f9c1a1acac7da9e43b2e6ae735754cd6",
        "size_bytes": 13188,
    },
}
SOURCE_PROVISIONAL_VALUES = {
    "fan_velocity": 6.0,
    "wcp_pad_t": 1.0,
    "core_plate_pad_t": 1.0,
}
SOURCE_PROVISIONAL_THERMAL_PAD_CONDUCTIVITY_W_MK = 3.0
HARD_SPEC = {
    "size_W_max_mm": 1200.0,
    "size_L_max_mm": 1000.0,
    "size_H_max_mm": 750.0,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "n_core_group_max": 4.0,
    "primary_conductor_thickness_mm": 5.0,
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "resonance_min_Hz": 15_000.0,
    "temperature_max_C": 100.0,
}
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
LIBRARY_BUNDLE_SHA256 = (
    "ec60dedbb4c57a9c847201c68f3621526529336f24eaa710d4ab7a76df24b5b7"
)
LIBRARY_BUNDLE_SIZE = 618_631
MESH_LINEAGE_REVISION = "1641bff5fde3901debab2f267332cb60de28d8ad"
SEALED_K3_SOURCE_REVISION = "8a8d90f68e8728669282f586f24304c7cc807029"
PRE_DEADLINE_K0P2_REVISION = "b171c7c"
STORAGE_ACCOUNT = "wjddn5916"
LANES = {
    "standard": {
        "account_name": "r1jae262",
        "node_name": "n107",
        "memory_mb": 65_536,
        "timeout_seconds": 18_000,
    },
    "full": {
        "account_name": "jji0930",
        "node_name": "n115",
        "memory_mb": 98_304,
        "timeout_seconds": 28_800,
    },
}
PRODUCTION_BLOBS = (
    "run_simulation_260706.py",
    "module/input_parameter_260706.py",
    "module/fixed_boundary_contract.py",
    "module/thermal_260706.py",
)
FIXED_PARAMETER_VALUES = {
    "fan_velocity": FIXED_FAN_VELOCITY_M_S,
    "wcp_pad_t": FIXED_WCP_PAD_THICKNESS_MM,
    "core_plate_pad_t": FIXED_CORE_PLATE_PAD_THICKNESS_MM,
}
FIXED_THERMAL_PAD_ELECTRICAL_CONDUCTIVITY_S_M = 0.0
FIXED_THERMAL_PAD_MATERIAL_POLICY = (
    "fixed_boundary_tim_k0p2_native_attested_0p2WmK_"
    "electrically_insulating_v1"
)


class ReplayPreparationError(RuntimeError):
    """A source identity, physical replay, or immutable artifact drifted."""


def attest_fixed_thermal_parameters(
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the shared fixed-boundary contract to a parameter mapping."""
    return attest_fixed_boundary(
        params,
        thermal_pad_conductivity_w_mk=(
            FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK
        ),
    )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    os.replace(temporary, path)


def _git(root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode:
        raise ReplayPreparationError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _read_authenticated_json(
    path: Path, label: str, expected: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ReplayPreparationError(f"{label} is not a regular file")
    if (
        resolved.stat().st_size != int(expected["size_bytes"])
        or file_sha256(resolved) != expected["sha256"]
    ):
        raise ReplayPreparationError(f"{label} immutable identity mismatch")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayPreparationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ReplayPreparationError(f"{label} JSON root is not an object")
    return resolved, value


def extract_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    command = str(payload.get("command") or "")
    match = re.search(
        r"""printf '%s' '(\{.*?\})' > cand\.json""",
        command,
        flags=re.DOTALL,
    )
    if match is None:
        raise ReplayPreparationError("Standard payload has no sealed cand.json")
    try:
        params = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ReplayPreparationError("sealed cand.json is invalid") from exc
    if not isinstance(params, dict) or set(params) != set(KEYS):
        raise ReplayPreparationError("sealed candidate schema is not exact KEYS")
    for name, expected in SOURCE_PROVISIONAL_VALUES.items():
        if not math.isclose(
            float(params[name]), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ReplayPreparationError(
                f"source provisional value drifted: {name}"
            )
    return params


def fixed_execution_params(
    source_params: Mapping[str, Any], fidelity: str
) -> dict[str, Any]:
    if fidelity not in {"standard", "full"}:
        raise ReplayPreparationError(f"unsupported fidelity: {fidelity}")
    params = dict(source_params)
    params.update(FIXED_PARAMETER_VALUES)
    if fidelity == "standard":
        params.update(
            full_model=0,
            loss_sym_on=1,
            thermal_symmetry="eighth",
            n_explicit_turns=0,
            keep_project=0,
        )
    else:
        params.update(
            full_model=1,
            loss_sym_on=0,
            thermal_symmetry="full",
            n_explicit_turns=2,
            keep_project=1,
        )
    attest_fixed_thermal_parameters(params)
    if set(params) != set(KEYS):
        raise ReplayPreparationError("fixed execution params escaped KEYS")
    return params


def replay_geometry(params: Mapping[str, Any]) -> dict[str, Any]:
    input_df = create_input_parameter(dict(params))
    valid, frame, errors = validation_check(
        input_df, strict=False, return_errors=True
    )
    if not valid or errors or len(frame) != 1:
        raise ReplayPreparationError(
            "fixed candidate validation failed: " + "; ".join(errors)
        )
    row = frame.iloc[0]
    volume_l, dimensions = bounding_box_lit(row)
    width_mm, length_mm, height_mm = map(float, dimensions)
    design_b = design_analytical_b_field_t(
        row,
        core_lamination_factor=0.85,
        area_basis=B_AREA_BASIS_GROSS_WITH_LAMINATION,
    )
    insulation = {
        name: float(row[name])
        for name in (
            "cc_w2c_space_x",
            "cc_w2c_space_y",
            "w2c_w1c_space_x",
            "w2c_w1c_space_y",
            "w1c_w2s_gap_x_actual",
            "w1s_cs_space_x",
            "cs_w1s_space_y",
            "h_gap2",
        )
    }
    checks = {
        "width": width_mm <= HARD_SPEC["size_W_max_mm"],
        "length": length_mm <= HARD_SPEC["size_L_max_mm"],
        "height": height_mm <= HARD_SPEC["size_H_max_mm"],
        "design_flux_density": design_b <= HARD_SPEC["B_limit_T"],
        "minimum_insulation": min(insulation.values())
        >= HARD_SPEC["insulation_min_mm"],
        "core_group_count": float(row["n_core_group"])
        <= HARD_SPEC["n_core_group_max"],
        "primary_conductor_thickness": math.isclose(
            float(row["cw1"]),
            HARD_SPEC["primary_conductor_thickness_mm"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    return {
        "validation_pass": True,
        "validation_errors": [],
        "dimensions_mm": {
            "W": width_mm,
            "L": length_mm,
            "H": height_mm,
        },
        "volume_L": float(volume_l),
        "B_design_square_material_analytic_T": float(design_b),
        "Ae_gross_m2": float(row["Ae_m2"]),
        "Ae_effective_m2": float(row["Ae_m2"]) * 0.85,
        "core_depth_each_mm": float(row["core_depth_each"]),
        "tx_winding_y_pack_mm": float(row["nwb1_main_y"]),
        "minimum_insulation_mm": min(insulation.values()),
        "realized_insulation_mm": insulation,
        "hard_checks": checks,
        "hard_prescreen_pass": all(checks.values()),
        "superseded_size_contract": {
            "size_L_max_mm": 1200.0,
            "authoritative": False,
            "pass": length_mm <= 1200.0,
            "reason": "superseded_by_final_1200x1000x750_envelope",
        },
    }


def _source_electrical(result: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "Llt",
        "Lmt",
        "Lmr",
        "C_tx_tx_F",
        "C_rx_rx_F",
        "C_tx_rx_F",
        "f_res_tx_self_Hz",
        "f_res_rx_self_Hz",
        "f_res_interwinding_Hz",
    )
    values: dict[str, float] = {}
    for name in required:
        try:
            value = float(result[name])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ReplayPreparationError(
                f"source Full-EM result lacks {name}"
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise ReplayPreparationError(
                f"source Full-EM result has invalid {name}"
            )
        values[name] = value
    half_lm_tx = values["f_res_tx_self_Hz"] * math.sqrt(2.0)
    half_lm_rx = values["f_res_rx_self_Hz"] * math.sqrt(2.0)
    return {
        **values,
        "half_magnetizing_resonance_Hz": min(half_lm_tx, half_lm_rx),
        "measurement_authority": "actual Full-EM task 95067",
        "reusable_for_fixed_geometry": False,
        "invalidation_reason": (
            "pad1_to_pad2_changes_core_depth_and_Tx_y_pack; "
            "fresh Maxwell matrix/capacitance solve required"
        ),
    }


def build_candidate_report(
    source_params: Mapping[str, Any],
    source_result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    standard_params = fixed_execution_params(source_params, "standard")
    full_params = fixed_execution_params(source_params, "full")
    source_geometry = replay_geometry(source_params)
    fixed_geometry = replay_geometry(standard_params)
    defaults = get_drawing_default_params()
    fixed_boundary_evidence = attest_fixed_thermal_parameters(defaults)
    source_boundary_evidence = evaluate_fixed_boundary(
        source_params,
        thermal_pad_conductivity_w_mk=(
            SOURCE_PROVISIONAL_THERMAL_PAD_CONDUCTIVITY_W_MK
        ),
    )
    if (
        source_boundary_evidence["authority_class"]
        != "diagnostic_override_only"
        or not source_boundary_evidence["promotion_forbidden_by_fixed_boundary"]
    ):
        raise ReplayPreparationError(
            "provisional source unexpectedly crossed fixed-boundary authority"
        )
    fixed_identity = {
        "schema_version": "mft-fixed-thermal-physical-candidate-v1",
        "source_full_em_task_id": SOURCE_FULL_EM_TASK_ID,
        "source_physical_candidate_digest": SOURCE_PHYSICAL_CANDIDATE_DIGEST,
        "fixed_boundary_contract_schema": FIXED_BOUNDARY_CONTRACT_SCHEMA,
        "fixed_boundary_contract_sha256": FIXED_BOUNDARY_CONTRACT_SHA256,
        "fixed_parameter_values": FIXED_PARAMETER_VALUES,
        "thermal_pad_conductivity_W_mK": (
            FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK
        ),
        "standard_params_sha256": canonical_sha256(standard_params),
        "full_params_sha256": canonical_sha256(full_params),
    }
    candidate_digest = canonical_sha256(fixed_identity)
    prescreen_pass = bool(fixed_geometry["hard_prescreen_pass"])
    failed_prescreens = sorted(
        name
        for name, passed in fixed_geometry["hard_checks"].items()
        if not passed
    )
    report = {
        "schema_version": "mft-fixed-thermal-candidate-replay-v1",
        "status": (
            "eligible_for_fresh_standard_and_full_replay"
            if prescreen_pass
            else "diagnostic_nonselectable_hard_prescreen_failed"
        ),
        "candidate_digest": candidate_digest,
        "candidate_identity": fixed_identity,
        "source_candidate": {
            "full_em_task_id": SOURCE_FULL_EM_TASK_ID,
            "physical_candidate_digest": SOURCE_PHYSICAL_CANDIDATE_DIGEST,
            "provisional_values": SOURCE_PROVISIONAL_VALUES,
            "provisional_thermal_pad_conductivity_W_mK": (
                SOURCE_PROVISIONAL_THERMAL_PAD_CONDUCTIVITY_W_MK
            ),
            "fixed_boundary_evaluation": source_boundary_evidence,
            "geometry": source_geometry,
            "electrical": _source_electrical(source_result),
        },
        "fixed_boundary": {
            "contract_schema": FIXED_BOUNDARY_CONTRACT_SCHEMA,
            "contract_sha256": FIXED_BOUNDARY_CONTRACT_SHA256,
            "fan_velocity_m_s": FIXED_FAN_VELOCITY_M_S,
            "wcp_pad_thickness_mm": FIXED_WCP_PAD_THICKNESS_MM,
            "core_plate_pad_thickness_mm": (
                FIXED_CORE_PLATE_PAD_THICKNESS_MM
            ),
            "thermal_pad_conductivity_W_mK": (
                FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK
            ),
            "thermal_pad_electrical_conductivity_S_m": (
                FIXED_THERMAL_PAD_ELECTRICAL_CONDUCTIVITY_S_M
            ),
            "thermal_pad_material_policy": (
                FIXED_THERMAL_PAD_MATERIAL_POLICY
            ),
            "attestation": fixed_boundary_evidence,
        },
        "fixed_geometry_and_analytic_electrical_prescreen": fixed_geometry,
        "fresh_fea_required": {
            "Llt": True,
            "capacitance": True,
            "resonance": True,
            "loss": True,
            "temperature": True,
            "reason": "physical pad stack and TIM boundary changed",
        },
        "final_hard_gate_status": {
            "geometry_and_analytic_prescreen": (
                "pass" if prescreen_pass else "fail"
            ),
            "failed_prescreens": failed_prescreens,
            "Llt": "fresh_fea_required",
            "half_magnetizing_resonance": "fresh_fea_required",
            "loss": "fresh_fea_required",
            "temperature": "fresh_fea_required",
            "final_design_selectable": False,
        },
        "hard_spec": HARD_SPEC,
        "submission_eligible": prescreen_pass,
        "design_approved": False,
        "automatic_promotion_allowed": False,
        "canonical_dataset_mutation_allowed": False,
        "ui_mutation_allowed": False,
    }
    return report, {"standard": standard_params, "full": full_params}


def standalone_8_core_auth(revision: str) -> str:
    return canonical_sha256(
        {
            "backend": "standalone",
            "contract_version": "mft-standalone-core-optin-v1",
            "requested_num_cores": 8,
            "required_slurm_cpus_per_task": 8,
            "solver_revision": revision,
        }
    )


def _execution_command(
    *,
    fidelity: str,
    params: Mapping[str, Any],
    revision: str,
    bundle_sha256: str,
    bundle_size: int,
    profile_sha256: str,
    release_root: str,
    output_parent: str,
) -> str:
    params_json = canonical_bytes(params).decode("ascii")
    full_flag = " --full" if fidelity == "full" else ""
    preserve_project = (
        'if [ -d simulation ]; then cp -a simulation "$RESULT_DIR/"; fi;'
        if fidelity == "full"
        else ""
    )
    run_slug = f"mft-fixed-replay-{fidelity}-{profile_sha256[:16]}"
    lines = [
        "set -euo pipefail",
        "umask 077",
        f"RELEASE_ROOT={shlex.quote(release_root)}",
        f"OUTPUT_PARENT={shlex.quote(output_parent)}",
        f'RUN_ROOT="/enroot/{run_slug}-${{SLURM_SCHED_TASK_ID}}"',
        'test "$(findmnt -n -o FSTYPE -T /enroot)" = xfs',
        'test "$(df -Pk /enroot | awk \'NR==2 {print $4}\')" -ge 209715200',
        f'test "$(stat -c %s "$RELEASE_ROOT/source.bundle")" = {bundle_size}',
        (
            'test "$(sha256sum "$RELEASE_ROOT/source.bundle" '
            f'| awk \'{{print $1}}\')" = {bundle_sha256}'
        ),
        (
            'test "$(stat -c %s "$RELEASE_ROOT/pyaedt-library.bundle")" '
            f"= {LIBRARY_BUNDLE_SIZE}"
        ),
        (
            'test "$(sha256sum "$RELEASE_ROOT/pyaedt-library.bundle" '
            f'| awk \'{{print $1}}\')" = {LIBRARY_BUNDLE_SHA256}'
        ),
        'test -r "$RELEASE_ROOT/release-manifest.json"',
        'cleanup() { rm -rf -- "$RUN_ROOT"; }',
        "trap cleanup EXIT",
        'mkdir -p "$RUN_ROOT"',
        (
            'git clone -q "$RELEASE_ROOT/source.bundle" "$RUN_ROOT/repo" '
            f"&& git -C \"$RUN_ROOT/repo\" checkout -q --detach {revision}"
        ),
        (
            'test "$(git -C "$RUN_ROOT/repo" rev-parse HEAD)" = '
            f"{revision}"
        ),
        (
            'test -z "$(git -C "$RUN_ROOT/repo" status '
            '--porcelain --untracked-files=all)"'
        ),
        (
            'git clone -q "$RELEASE_ROOT/pyaedt-library.bundle" '
            '"$RUN_ROOT/pyaedt_library" '
            f"&& git -C \"$RUN_ROOT/pyaedt_library\" "
            f"checkout -q --detach {LIBRARY_REVISION}"
        ),
        (
            'test "$(git -C "$RUN_ROOT/pyaedt_library" rev-parse HEAD)" = '
            f"{LIBRARY_REVISION}"
        ),
        (
            'test -z "$(git -C "$RUN_ROOT/pyaedt_library" status '
            '--porcelain --untracked-files=all)"'
        ),
        (
            'export MFT_PYAEDT_LIBRARY_ROOT="$RUN_ROOT/pyaedt_library/src"'
        ),
        'export MFT_AEDT_BACKEND="standalone"',
        'export MFT_STANDALONE_CORE_CONTRACT="mft-standalone-core-optin-v1"',
        'export MFT_STANDALONE_CORE_COUNT="8"',
        (
            'export MFT_STANDALONE_CORE_AUTH_SHA256='
            f'"{standalone_8_core_auth(revision)}"'
        ),
        (
            'export MFT_FIXED_THERMAL_PROFILE_SHA256='
            f'"{profile_sha256}"'
        ),
        'source /etc/profile.d/lmod.sh 2>/dev/null || true',
        (
            "module load ansys-electronics/v252 2>/dev/null || "
            "export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64"
        ),
        "export FLEXLM_TIMEOUT=3000000",
        "export I_MPI_HYDRA_BOOTSTRAP=fork",
        "export FLUENT_MPIRUN_FLAGS='-bootstrap fork'",
        'cd "$RUN_ROOT/repo"',
        f"printf '%s' {shlex.quote(params_json)} > cand.json",
        (
            "simulation_rc=0; "
            "python run_simulation_260706.py --fixed --thermal --headless"
            f"{full_flag} --params cand.json || simulation_rc=$?"
        ),
        (
            'RESULT_DIR="$OUTPUT_PARENT/'
            f'{run_slug}-${{SLURM_SCHED_TASK_ID}}"'
        ),
        'mkdir "$RESULT_DIR"',
        (
            'if [ -f simulation_results_260706.csv ]; then '
            'cp simulation_results_260706.csv "$RESULT_DIR/"; fi;'
        ),
        preserve_project,
        (
            "printf '%s\\n' "
            f"{shlex.quote(profile_sha256)} "
            '> "$RESULT_DIR/profile.sha256"'
        ),
        "exit $simulation_rc",
    ]
    return "; ".join(line for line in lines if line)


def build_payload(
    *,
    fidelity: str,
    params: Mapping[str, Any],
    candidate_digest: str,
    profile_sha256: str,
    revision: str,
    bundle_sha256: str,
    bundle_size: int,
    release_root: str,
    output_parent: str,
) -> dict[str, Any]:
    attest_fixed_thermal_parameters(params)
    if not replay_geometry(params)["hard_prescreen_pass"]:
        raise ReplayPreparationError(
            "refusing payload for a fixed candidate that failed hard prescreen"
        )
    lane = LANES[fidelity]
    params_sha = canonical_sha256(params)
    payload = {
        "name": f"mft-fixed-{fidelity}-8c-{candidate_digest[:12]}",
        "command": _execution_command(
            fidelity=fidelity,
            params=params,
            revision=revision,
            bundle_sha256=bundle_sha256,
            bundle_size=bundle_size,
            profile_sha256=profile_sha256,
            release_root=release_root,
            output_parent=output_parent,
        ),
        "remote_cwd": "__SLURM_SCHEDULER_ACCOUNT_WORKSPACE__/runs",
        "cpus": 8,
        "memory_mb": lane["memory_mb"],
        "gpus": 0,
        "partition": "cpu2",
        "node_name": lane["node_name"],
        "account_name": lane["account_name"],
        "exclusive_node": False,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "priority": 5,
        "timeout_seconds": lane["timeout_seconds"],
        "dedupe_key": (
            f"mft-fixed-thermal-replay:{fidelity}:{candidate_digest}:"
            f"{revision}:{profile_sha256}"
        ),
        "max_workers_per_node": 0,
        "payload_json": {
            "schema_version": PAYLOAD_SCHEMA,
            "fidelity": fidelity,
            "candidate_digest": candidate_digest,
            "params_sha256": params_sha,
            "profile_sha256": profile_sha256,
            "source_revision": revision,
            "source_bundle_sha256": bundle_sha256,
            "fixed_boundary_contract_schema": (
                FIXED_BOUNDARY_CONTRACT_SCHEMA
            ),
            "fixed_boundary_contract_sha256": (
                FIXED_BOUNDARY_CONTRACT_SHA256
            ),
            "storage_account": STORAGE_ACCOUNT,
            "remote_stage_precondition": True,
            "automatic_promotion_allowed": False,
            "canonical_dataset_mutation_allowed": False,
            "ui_mutation_allowed": False,
        },
        "cleanup_globs": "",
        "project": "MFT_1MW_2026v1",
        "entrypoint": "",
        "requested_account_name": lane["account_name"],
        "requested_allocation_id": 0,
        "same_node_as_task_id": 0,
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
    }
    return payload


def _lineage(revision: str) -> dict[str, Any]:
    for ancestor in (
        MESH_LINEAGE_REVISION,
        SEALED_K3_SOURCE_REVISION,
        PRE_DEADLINE_K0P2_REVISION,
    ):
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "-C",
                str(REPO_ROOT),
                "merge-base",
                "--is-ancestor",
                ancestor,
                revision,
            ],
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            raise ReplayPreparationError(
                f"required thermal lineage is absent: {ancestor}"
            )
    return {
        "pre_deadline_authoritative_k0p2_revision": PRE_DEADLINE_K0P2_REVISION,
        "superseded_isolated_k3_revision": SEALED_K3_SOURCE_REVISION,
        "latest_mesh_and_fluent_mapping_revision": MESH_LINEAGE_REVISION,
        "fresh_fixed_boundary_revision": revision,
        "source_git_blobs": {
            path: _git(REPO_ROOT, "rev-parse", f"{revision}:{path}")
            for path in PRODUCTION_BLOBS
        },
        "sealed_source_k3_is_historical_only": True,
        "fresh_replay_uses_k0p2": True,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    standard_path, standard_payload = _read_authenticated_json(
        args.standard_payload,
        "task95067 Standard payload",
        SOURCE_ARTIFACTS["standard_payload"],
    )
    followup_path, followup = _read_authenticated_json(
        args.followup_profile,
        "task95067 followup profile",
        SOURCE_ARTIFACTS["followup_profile"],
    )
    result_path, sweep_result = _read_authenticated_json(
        args.sweep_result,
        "task95067 sweep result",
        SOURCE_ARTIFACTS["sweep_result"],
    )
    if (
        int(followup.get("sweep_task_id") or 0) != SOURCE_FULL_EM_TASK_ID
        or followup.get("physical_candidate_digest")
        != SOURCE_PHYSICAL_CANDIDATE_DIGEST
        or int(sweep_result.get("result_valid_em") or 0) != 1
        or int(sweep_result.get("full_model") or 0) != 1
    ):
        raise ReplayPreparationError("task95067 source identity drifted")
    source_params = extract_candidate(standard_payload)
    report, params_by_fidelity = build_candidate_report(
        source_params, sweep_result
    )

    revision = _git(REPO_ROOT, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ReplayPreparationError("source revision is not exact 40-hex")
    if _git(REPO_ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise ReplayPreparationError("source worktree must be clean")
    lineage = _lineage(revision)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ReplayPreparationError(
            f"output directory already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    report["source_artifacts"] = {
        "standard_payload": artifact_ref(standard_path),
        "followup_profile": artifact_ref(followup_path),
        "sweep_result": artifact_ref(result_path),
    }
    report["lineage"] = lineage
    profiles: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    bundle_ref: dict[str, Any] | None = None
    release_root: str | None = None
    output_parent: str | None = None
    if report["submission_eligible"]:
        if args.library_bundle is None:
            raise ReplayPreparationError(
                "--library-bundle is required for an eligible replay"
            )
        bundle = output_dir / "source.bundle"
        subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "-C",
                str(REPO_ROOT),
                "bundle",
                "create",
                str(bundle),
                "HEAD",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "-C",
                str(REPO_ROOT),
                "bundle",
                "verify",
                str(bundle),
            ],
            check=True,
        )
        bundle_ref = artifact_ref(bundle)
        library = args.library_bundle.resolve(strict=True)
        if (
            file_sha256(library) != LIBRARY_BUNDLE_SHA256
            or library.stat().st_size != LIBRARY_BUNDLE_SIZE
        ):
            raise ReplayPreparationError(
                "PyAEDT library bundle identity mismatch"
            )
        library_ref = artifact_ref(library)
        release_root = (
            f"/gpfs/home1/{STORAGE_ACCOUNT}/slurm_scheduler/runs/"
            f"mft_fixed_boundary_replay_{revision[:12]}_"
            f"{bundle_ref['sha256'][:16]}"
        )
        output_parent = (
            f"/gpfs/home1/{STORAGE_ACCOUNT}/slurm_scheduler/runs/"
            f"mft_fixed_boundary_replay_{revision[:12]}_outputs"
        )
        release_manifest = {
            "schema_version": "mft-fixed-thermal-replay-release-v1",
            "source_revision": revision,
            "source_bundle_sha256": bundle_ref["sha256"],
            "source_bundle_size_bytes": bundle_ref["size_bytes"],
            "library_revision": LIBRARY_REVISION,
            "library_bundle_sha256": LIBRARY_BUNDLE_SHA256,
            "library_bundle_size_bytes": LIBRARY_BUNDLE_SIZE,
            "fixed_boundary_contract_schema": (
                FIXED_BOUNDARY_CONTRACT_SCHEMA
            ),
            "fixed_boundary_contract_sha256": (
                FIXED_BOUNDARY_CONTRACT_SHA256
            ),
            "candidate_digest": report["candidate_digest"],
            "lineage": lineage,
            "storage_account": STORAGE_ACCOUNT,
            "compute_acl_read_required": ["r1jae262", "jji0930"],
            "compute_acl_write_output_required": ["r1jae262", "jji0930"],
            "remote_stage_performed": False,
            "scheduler_submission_performed": False,
            "automatic_promotion_allowed": False,
            "ui_mutation_allowed": False,
        }
        manifest_path = output_dir / "release-manifest.json"
        atomic_json(manifest_path, release_manifest)
        manifest_ref = artifact_ref(manifest_path)

        for fidelity, params in params_by_fidelity.items():
            profile = {
                "schema_version": PROFILE_SCHEMA,
                "fidelity": fidelity,
                "candidate_digest": report["candidate_digest"],
                "params": params,
                "params_sha256": canonical_sha256(params),
                "source_revision": revision,
                "source_bundle_sha256": bundle_ref["sha256"],
                "release_manifest_sha256": manifest_ref["sha256"],
                "fixed_boundary": report["fixed_boundary"],
                "geometry_and_analytic_electrical_prescreen": report[
                    "fixed_geometry_and_analytic_electrical_prescreen"
                ],
                "fresh_fea_required": report["fresh_fea_required"],
                "automatic_promotion_allowed": False,
                "canonical_dataset_mutation_allowed": False,
                "ui_mutation_allowed": False,
            }
            profile_sha = canonical_sha256(profile)
            profile["profile_sha256"] = profile_sha
            profile_path = output_dir / f"{fidelity}-profile.json"
            atomic_json(profile_path, profile)
            payload = build_payload(
                fidelity=fidelity,
                params=params,
                candidate_digest=report["candidate_digest"],
                profile_sha256=profile_sha,
                revision=revision,
                bundle_sha256=bundle_ref["sha256"],
                bundle_size=bundle_ref["size_bytes"],
                release_root=release_root,
                output_parent=output_parent,
            )
            payload_path = output_dir / f"{fidelity}-payload.json"
            atomic_json(payload_path, payload)
            profiles[fidelity] = {
                **artifact_ref(profile_path),
                "canonical_payload_sha256": profile_sha,
                "params_sha256": profile["params_sha256"],
            }
            payloads[fidelity] = {
                **artifact_ref(payload_path),
                "canonical_payload_sha256": canonical_sha256(payload),
                "requested_account_name": payload["account_name"],
                "requested_node_name": payload["node_name"],
                "cpus": payload["cpus"],
            }
        report["source_release"] = {
            "source_revision": revision,
            "bundle": bundle_ref,
            "release_manifest": manifest_ref,
            "library_bundle": library_ref,
            "remote_release_root": release_root,
            "remote_output_parent": output_parent,
            "remote_stage_performed": False,
        }
    else:
        report["source_release"] = {
            "source_revision": revision,
            "bundle": None,
            "release_manifest": None,
            "library_bundle": None,
            "remote_release_root": None,
            "remote_output_parent": None,
            "remote_stage_performed": False,
            "release_skipped": True,
            "reason": "fixed candidate failed authoritative hard prescreen",
        }
    candidate_report_path = output_dir / "candidate-replay.json"
    atomic_json(candidate_report_path, report)
    handoff = {
        "schema_version": SCHEMA,
        "status": (
            "immutable_local_handoff_remote_stage_required"
            if report["submission_eligible"]
            else "diagnostic_nonselectable_no_submission_payloads"
        ),
        "candidate_digest": report["candidate_digest"],
        "candidate_replay": artifact_ref(candidate_report_path),
        "source_release": report["source_release"],
        "profiles": profiles,
        "payloads": payloads,
        "desired_compute_lanes": LANES,
        "durable_storage_account": STORAGE_ACCOUNT,
        "stage_preconditions": {
            "release_owner": STORAGE_ACCOUNT,
            "release_read_acl_accounts": ["r1jae262", "jji0930"],
            "output_write_acl_accounts": ["r1jae262", "jji0930"],
            "exact_read_write_probe_required": True,
            "n107_n115_enroot_xfs_probe_required": True,
        },
        "submission_eligible": report["submission_eligible"],
        "release_materialized": bool(report["submission_eligible"]),
        "scheduler_submission_performed": False,
        "automatic_promotion_allowed": False,
        "canonical_dataset_mutation_allowed": False,
        "ui_mutation_allowed": False,
    }
    handoff["handoff_payload_sha256"] = canonical_sha256(handoff)
    handoff_path = output_dir / "handoff.json"
    atomic_json(handoff_path, handoff)
    return {
        "output_dir": str(output_dir),
        "handoff": artifact_ref(handoff_path),
        "handoff_payload_sha256": handoff["handoff_payload_sha256"],
        "candidate_digest": report["candidate_digest"],
        "bundle": bundle_ref,
        "profiles": profiles,
        "payloads": payloads,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard-payload", type=Path, required=True)
    parser.add_argument("--followup-profile", type=Path, required=True)
    parser.add_argument("--sweep-result", type=Path, required=True)
    parser.add_argument(
        "--library-bundle",
        type=Path,
        help="required only when the candidate passes the hard prescreen",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    result = prepare(_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
