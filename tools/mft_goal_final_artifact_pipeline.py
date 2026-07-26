"""Prepare the fail-closed final AEDT and drawing artifact pipeline.

This tool starts *after* another component has selected and sealed one
authenticated symmetric hard-pass.  It never ranks candidates, submits or
cancels Scheduler work, runs a solver, or promotes an unverified result.

The output is a deterministic set of model-only parameter files plus one
sealed execution manifest for:

* the retained symmetric, non-rounded verification AEDT;
* a full, non-rounded model-only AEDT;
* a full, rounded model-only AEDT used only for drawings;
* drawing views and a nine-frame PPTX/PDF derived from 설계도면260706.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.fixed_boundary_contract import (  # noqa: E402
    FIXED_BOUNDARY_CONTRACT_SHA256,
    FIXED_CORE_PLATE_PAD_THICKNESS_MM,
    FIXED_FAN_VELOCITY_M_S,
    FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK,
    FIXED_WCP_PAD_THICKNESS_MM,
    attest_fixed_boundary,
)
from module.mft_goal_20260726_contract import (  # noqa: E402
    FIXED_COOLING_IDENTITY,
    FIXED_COOLING_IDENTITY_SHA256,
    FIXED_OPERATING_IDENTITY,
    FIXED_OPERATING_IDENTITY_SHA256,
    GOAL_CONTRACT_SCHEMA,
    GOAL_SIZE_LIMITS_MM,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    TEMPERATURE_FAMILY_LIMITS_C,
)


WINNER_AUTHORITY_SCHEMA = "mft-goal-final-artifact-winner-authority-v1"
GRADED_CAP_PROVENANCE_SCHEMA = (
    "mft-goal-final-graded-capacitance-provenance-v1"
)
TURN_GRADED_CAP_SCHEMA = "mft-turn-graded-energy-capacitance-v2"
PIPELINE_SCHEMA = "mft-goal-final-artifact-pipeline-v1"

REFERENCE_ROOT = Path(
    r"Z:\Projects\2025\8. 한양대 - 선박용MVDC\설계도면"
)
DEFAULT_REFERENCE_PDF = REFERENCE_ROOT / "설계도면260706.pdf"
DEFAULT_REFERENCE_PPTX = REFERENCE_ROOT / "설계도면260706.pptx"
REFERENCE_PDF_SHA256 = (
    "574d9aab033529cf3655d63542e27871c2e240b669b67e54dbfd2495a564437f"
)
REFERENCE_PPTX_SHA256 = (
    "b8069cc99cf1af3c8e5c5cbe6bd4a2896730620f570c29555ff492713d4af33d"
)

DRAWING_AUDIT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\drawing_reference_audit_v1"
)
DRAWING_TEMPLATE_MANIFEST = (
    DRAWING_AUDIT_ROOT
    / "pptx"
    / "workspace"
    / "template-inspect"
    / "template-manifest.json"
)
DRAWING_PDF_AUDIT_METADATA = (
    DRAWING_AUDIT_ROOT / "pdf" / "pdf_audit_metadata.json"
)

HEX64 = re.compile(r"[0-9a-f]{64}")
TARGET_LM_H = 0.002
TARGET_LM_ABS_TOLERANCE_H = 0.00002

MODEL_PARAM_UPDATES = {
    "full_unrounded": {
        "full_model": 1,
        "round_corner": 0,
        "matrix_on": 1,
        "cap_on": 0,
        "loss_on": 0,
        "thermal_on": 0,
        "loss_sym_on": 0,
        "thermal_symmetry": "full",
        "keep_project": 1,
    },
    "full_rounded_drawing_only": {
        "full_model": 1,
        "round_corner": 1,
        "matrix_on": 1,
        "cap_on": 0,
        "loss_on": 0,
        "thermal_on": 0,
        "loss_sym_on": 0,
        "thermal_symmetry": "full",
        "keep_project": 1,
    },
}


class FinalArtifactPipelineError(RuntimeError):
    """A winner authority, reference, or artifact contract drifted."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FinalArtifactPipelineError(
            "payload is not canonical finite JSON"
        ) from exc


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise FinalArtifactPipelineError("payload is already sealed")
    result["payload_sha256"] = payload_sha256(result)
    return result


def validate_seal(
    value: Mapping[str, Any], *, schema: str, label: str
) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    observed = result.pop("payload_sha256", None)
    if (
        value.get("schema_version") != schema
        or not isinstance(observed, str)
        or not HEX64.fullmatch(observed)
        or payload_sha256(result) != observed
    ):
        raise FinalArtifactPipelineError(f"{label} seal is invalid")
    result["payload_sha256"] = observed
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FinalArtifactPipelineError(f"not a regular file: {resolved}")
    label = (
        resolved.relative_to(relative_to.resolve(strict=True)).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": label,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalArtifactPipelineError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise FinalArtifactPipelineError(f"{label} must be a JSON object")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    data = canonical_bytes(value) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise FinalArtifactPipelineError(
                f"immutable output bytes differ: {target}"
            )
        return target
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, target)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
    return target


def _resolve_record(
    record: Any, *, base: Path, label: str
) -> tuple[Path, dict[str, Any]]:
    if (
        not isinstance(record, Mapping)
        or set(record) != {"path", "sha256", "size_bytes"}
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or not HEX64.fullmatch(str(record["sha256"]))
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or int(record["size_bytes"]) <= 0
    ):
        raise FinalArtifactPipelineError(f"{label} file record is malformed")
    supplied = Path(record["path"])
    path = (supplied if supplied.is_absolute() else base / supplied).resolve(
        strict=True
    )
    if (
        not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise FinalArtifactPipelineError(f"{label} file bytes drifted")
    return path, file_record(path)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FinalArtifactPipelineError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalArtifactPipelineError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise FinalArtifactPipelineError(f"{label} must be finite")
    return result


def _exact(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        if _finite(actual, label) != expected:
            raise FinalArtifactPipelineError(f"{label} drifted")
    elif actual != expected:
        raise FinalArtifactPipelineError(f"{label} drifted")


def _validate_simulation_params(params: Mapping[str, Any], label: str) -> None:
    try:
        from module.input_parameter_260706 import (
            create_input_parameter,
            validation_check,
        )
    except ImportError as exc:
        raise FinalArtifactPipelineError(
            "parameter validation requires the pyaedt2026v1 Python environment"
        ) from exc
    try:
        valid, _frame, errors = validation_check(
            create_input_parameter(dict(params)),
            strict=True,
            return_errors=True,
        )
    except Exception as exc:
        raise FinalArtifactPipelineError(
            f"{label} parameter validation raised an exception"
        ) from exc
    if not valid:
        raise FinalArtifactPipelineError(
            f"{label} parameters failed strict validation: {errors}"
        )


def _validate_fixed_params(params: Mapping[str, Any]) -> dict[str, Any]:
    for key, expected in FIXED_OPERATING_IDENTITY.items():
        _exact(params.get(key), expected, f"params.{key}")
    for key, expected in FIXED_COOLING_IDENTITY.items():
        if key == "thermal_pad_conductivity_W_mK":
            continue
        _exact(params.get(key), expected, f"params.{key}")
    boundary = attest_fixed_boundary(
        params,
        thermal_pad_conductivity_w_mk=(
            FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK
        ),
    )
    if boundary["authoritative_fixed_boundary_attested"] is not True:
        raise FinalArtifactPipelineError("fixed thermal boundary is not attested")
    return boundary


def _validate_graded_capacitance_provenance(
    provenance: Any,
    *,
    source: Mapping[str, Any],
    params: Mapping[str, Any],
    verification: Mapping[str, Any],
    authority_directory: Path,
) -> dict[str, Any]:
    """Require actual turn-graded Tx and Rx truth for final resonance.

    The legacy two-equipotential CapTx/CapRx matrix is useful as a screen but
    is not the physical terminal capacitance of a multi-turn winding.  Final
    authority therefore needs authenticated turn-graded result evidence for
    both windings on the exact winner geometry and physical air gap.
    """

    if not isinstance(provenance, Mapping):
        raise FinalArtifactPipelineError(
            "actual turn-graded capacitance provenance is absent"
        )
    if (
        provenance.get("schema_version") != GRADED_CAP_PROVENANCE_SCHEMA
        or provenance.get("provenance_authenticated") is not True
        or provenance.get("geometry_and_gap_exact_match") is not True
        or provenance.get("actual_connection_topology_attested") is not True
        or provenance.get("legacy_two_net_result_used") is not False
        or provenance.get("candidate_physics_sha256")
        != source["candidate_physics_sha256"]
    ):
        raise FinalArtifactPipelineError(
            "turn-graded capacitance provenance contract drifted"
        )
    source_kind = provenance.get("source_kind")
    if source_kind not in {
        "same_full_chain_graded_cap_result",
        "authenticated_same_geometry_tx_rx_pair",
    }:
        raise FinalArtifactPipelineError(
            "turn-graded capacitance source kind is unsupported"
        )
    physical_gap = _finite(
        params.get("core_center_gap_mm"), "winner physical center gap"
    )
    if _finite(
        provenance.get("core_center_gap_mm"),
        "turn-graded provenance center gap",
    ) != physical_gap:
        raise FinalArtifactPipelineError(
            "turn-graded capacitance physical air gap drifted"
        )
    topology_path, topology_record = _resolve_record(
        provenance.get("actual_connection_topology_receipt"),
        base=authority_directory,
        label="turn-graded actual-connection topology receipt",
    )

    rows: dict[str, dict[str, Any]] = {}
    for key, active in (("tx", "Tx"), ("rx", "Rx")):
        row = provenance.get(key)
        if not isinstance(row, Mapping):
            raise FinalArtifactPipelineError(
                f"turn-graded {active} result provenance is absent"
            )
        task_id = row.get("task_id")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or row.get("active_winding") != active
            or row.get("cap_turn_graded_schema_version")
            != TURN_GRADED_CAP_SCHEMA
            or row.get("candidate_physics_sha256")
            != source["candidate_physics_sha256"]
            or _finite(
                row.get("core_center_gap_mm"),
                f"turn-graded {active} center gap",
            )
            != physical_gap
            or not isinstance(row.get("solver_revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", row["solver_revision"])
            or not isinstance(row.get("library_revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", row["library_revision"])
            or not isinstance(row.get("result_sha256"), str)
            or not HEX64.fullmatch(row["result_sha256"])
        ):
            raise FinalArtifactPipelineError(
                f"turn-graded {active} execution identity drifted"
            )
        capacitance = _finite(
            row.get("terminal_capacitance_F"),
            f"turn-graded {active} terminal capacitance",
        )
        inductance = _finite(
            row.get("self_inductance_H"),
            f"turn-graded {active} self inductance",
        )
        resonance = _finite(
            row.get("resonance_Hz"),
            f"turn-graded {active} resonance",
        )
        if capacitance <= 0.0 or inductance <= 0.0 or resonance < 15_000.0:
            raise FinalArtifactPipelineError(
                f"turn-graded {active} measured resonance contract failed"
            )
        receipt_path, receipt_record = _resolve_record(
            row.get("authenticated_result_receipt"),
            base=authority_directory,
            label=f"turn-graded {active} result receipt",
        )
        rows[key] = {
            "task_id": task_id,
            "terminal_capacitance_F": capacitance,
            "self_inductance_H": inductance,
            "resonance_Hz": resonance,
            "authenticated_result_receipt": receipt_record,
            "authenticated_result_receipt_path": str(receipt_path),
        }

    same_full_chain = (
        rows["tx"]["task_id"] == source["task_id"]
        and rows["rx"]["task_id"] == source["task_id"]
    )
    if (
        source_kind == "same_full_chain_graded_cap_result"
        and not same_full_chain
    ):
        raise FinalArtifactPipelineError(
            "same-full-chain graded-cap task identity drifted"
        )
    if (
        source_kind == "authenticated_same_geometry_tx_rx_pair"
        and same_full_chain
    ):
        raise FinalArtifactPipelineError(
            "separate graded-cap pair is mislabeled as same full-chain"
        )
    minimum = min(rows["tx"]["resonance_Hz"], rows["rx"]["resonance_Hz"])
    if (
        _finite(
            provenance.get("minimum_resonance_Hz"),
            "turn-graded minimum resonance",
        )
        != minimum
        or _finite(
            verification.get("actual_resonance_Hz"),
            "actual final resonance",
        )
        != minimum
    ):
        raise FinalArtifactPipelineError(
            "final resonance is not the authenticated turn-graded minimum"
        )
    return {
        "source_kind": source_kind,
        "same_full_chain_task": same_full_chain,
        "minimum_resonance_Hz": minimum,
        "actual_connection_topology_receipt": topology_record,
        "actual_connection_topology_receipt_path": str(topology_path),
        "tx": rows["tx"],
        "rx": rows["rx"],
    }


def validate_winner_authority(
    authority: Mapping[str, Any], *, authority_directory: Path
) -> dict[str, Any]:
    """Authenticate the already-selected winner without choosing one."""

    value = validate_seal(
        authority,
        schema=WINNER_AUTHORITY_SCHEMA,
        label="winner authority",
    )
    source = value.get("source")
    verification = value.get("symmetric_verification")
    contracts = value.get("contracts")
    params = value.get("params")
    if not all(
        isinstance(item, Mapping)
        for item in (source, verification, contracts, params)
    ):
        raise FinalArtifactPipelineError(
            "winner authority sections are incomplete"
        )
    if (
        source.get("selection_performed_upstream") is not True
        or source.get("candidate_promotion_performed_by_pipeline") is not False
        or source.get("selected_symmetric_hard_pass") is not True
        or isinstance(source.get("task_id"), bool)
        or not isinstance(source.get("task_id"), int)
        or int(source["task_id"]) <= 0
        or not isinstance(source.get("candidate_physics_sha256"), str)
        or not HEX64.fullmatch(source["candidate_physics_sha256"])
    ):
        raise FinalArtifactPipelineError(
            "upstream selection authority is absent or malformed"
        )
    upstream_path, upstream_record = _resolve_record(
        source.get("upstream_selection_receipt"),
        base=authority_directory,
        label="upstream selection receipt",
    )
    symmetric_path, symmetric_record = _resolve_record(
        source.get("retained_symmetric_aedt"),
        base=authority_directory,
        label="retained symmetric AEDT",
    )
    if symmetric_path.suffix.lower() != ".aedt":
        raise FinalArtifactPipelineError(
            "retained symmetric artifact must be one .aedt file"
        )

    required_contracts = {
        "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
        "goal_stage_spec_sha256": GOAL_STAGE_SPEC_SHA256,
        "goal_temperature_contract_sha256": (
            GOAL_TEMPERATURE_CONTRACT_SHA256
        ),
        "fixed_operating_identity_sha256": (
            FIXED_OPERATING_IDENTITY_SHA256
        ),
        "fixed_cooling_identity_sha256": FIXED_COOLING_IDENTITY_SHA256,
        "fixed_boundary_contract_sha256": (
            FIXED_BOUNDARY_CONTRACT_SHA256
        ),
    }
    for key, expected in required_contracts.items():
        _exact(contracts.get(key), expected, f"contracts.{key}")

    required_model_contract = {
        "full_model": 0,
        "round_corner": 0,
        "matrix_solved": True,
        "capacitance_solved": True,
        "graded_capacitance_solved": True,
        "loss_solved": True,
        "thermal_solved": True,
        "measured_hard_constraints_passed": True,
        "rounded_fea_used": False,
        "legacy_two_net_capacitance_used_for_final_resonance": False,
    }
    for key, expected in required_model_contract.items():
        _exact(
            verification.get(key),
            expected,
            f"symmetric_verification.{key}",
        )
    dimensions = verification.get("actual_dimensions_mm")
    temperatures = verification.get("actual_temperature_family_max_C")
    if not isinstance(dimensions, Mapping) or not isinstance(
        temperatures, Mapping
    ):
        raise FinalArtifactPipelineError(
            "measured dimension/temperature evidence is absent"
        )
    for axis, limit in GOAL_SIZE_LIMITS_MM.items():
        actual = _finite(dimensions.get(axis), f"actual dimension {axis}")
        if actual > float(limit):
            raise FinalArtifactPipelineError(
                f"actual dimension {axis} exceeds {limit} mm"
            )
    for family, limit in TEMPERATURE_FAMILY_LIMITS_C.items():
        actual = _finite(
            temperatures.get(family), f"actual temperature {family}"
        )
        if actual > float(limit):
            raise FinalArtifactPipelineError(
                f"actual {family} temperature exceeds {limit} C"
            )
    resonance = _finite(
        verification.get("actual_resonance_Hz"), "actual resonance"
    )
    if resonance < 15_000.0:
        raise FinalArtifactPipelineError(
            "actual symmetric resonance is below 15 kHz"
        )
    lm_h = _finite(
        verification.get("actual_Lm_primary_referred_H"),
        "actual primary-referred Lm",
    )
    if abs(lm_h - TARGET_LM_H) > TARGET_LM_ABS_TOLERANCE_H:
        raise FinalArtifactPipelineError(
            "actual primary-referred Lm is outside 2 mH tolerance"
        )
    graded_capacitance = _validate_graded_capacitance_provenance(
        verification.get("graded_capacitance_provenance"),
        source=source,
        params=params,
        verification=verification,
        authority_directory=authority_directory,
    )
    fixed_boundary = verification.get("fixed_boundary")
    if not isinstance(fixed_boundary, Mapping):
        raise FinalArtifactPipelineError(
            "symmetric fixed-boundary evidence is absent"
        )
    for key, expected in {
        "fan_velocity_m_s": FIXED_FAN_VELOCITY_M_S,
        "fan_config": "dual",
        "core_plate_pad_t_mm": FIXED_CORE_PLATE_PAD_THICKNESS_MM,
        "wcp_pad_t_mm": FIXED_WCP_PAD_THICKNESS_MM,
        "thermal_pad_conductivity_W_mK": (
            FIXED_THERMAL_PAD_CONDUCTIVITY_W_MK
        ),
        "TIM_mutated": False,
    }.items():
        _exact(
            fixed_boundary.get(key),
            expected,
            f"symmetric_verification.fixed_boundary.{key}",
        )

    _validate_simulation_params(params, "winner")
    boundary_attestation = _validate_fixed_params(params)
    for key, expected in {
        "full_model": 0,
        "round_corner": 0,
        "matrix_on": 1,
        "cap_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "cw1": 5.0,
        "gap1": 1.6,
    }.items():
        _exact(params.get(key), expected, f"params.{key}")
    n1 = int(_finite(params.get("N1_main"), "N1_main")) + int(
        _finite(params.get("N1_side"), "N1_side")
    )
    n2 = int(_finite(params.get("N2_main"), "N2_main")) + int(
        _finite(params.get("N2_side"), "N2_side")
    )
    if n1 <= 0 or n2 != 10 * n1:
        raise FinalArtifactPipelineError(
            "winner turns do not implement the required 1:10 ratio"
        )

    return {
        "authority": value,
        "source": {
            "task_id": int(source["task_id"]),
            "candidate_physics_sha256": source[
                "candidate_physics_sha256"
            ],
            "upstream_selection_receipt": upstream_record,
            "upstream_selection_receipt_path": str(upstream_path),
            "retained_symmetric_aedt": symmetric_record,
            "retained_symmetric_aedt_path": str(symmetric_path),
        },
        "params": copy.deepcopy(dict(params)),
        "verification": copy.deepcopy(dict(verification)),
        "graded_capacitance_provenance": graded_capacitance,
        "fixed_boundary_attestation": boundary_attestation,
        "turns": {"N1": n1, "N2": n2},
    }


def _reference_record(
    path: Path, *, expected_sha256: str, label: str
) -> dict[str, Any]:
    record = file_record(path)
    if record["sha256"] != expected_sha256:
        raise FinalArtifactPipelineError(f"{label} bytes drifted")
    return record


def derive_model_variants(
    base_params: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for name, updates in MODEL_PARAM_UPDATES.items():
        params = copy.deepcopy(dict(base_params))
        params.update(updates)
        _validate_simulation_params(params, name)
        _validate_fixed_params(params)
        variants[name] = params
    return variants


def _model_command(
    *,
    params_path: Path,
    rounded: bool,
    work_directory: Path,
) -> dict[str, Any]:
    argv = [
        str(Path(sys.executable).resolve()),
        str((REPOSITORY_ROOT / "run_simulation_260706.py").resolve()),
        "--fixed",
        "--params",
        str(params_path.resolve()),
        "--full",
        "--round" if rounded else "--no-round",
        "--model-only",
        "--headless",
    ]
    return {
        "cwd": str(work_directory.resolve()),
        "argv": argv,
        "solver_invoked": False,
        "model_only_required": True,
        "postcondition": (
            "exactly one newly-created simulation/*/*.aedt; copy by SHA "
            "to the declared final filename and seal a file record"
        ),
    }


def _slide_contract() -> list[dict[str, Any]]:
    return [
        {
            "slide": 1,
            "role": "cover",
            "candidate_data": "final title, candidate identity, issue date",
        },
        {
            "slide": 2,
            "role": "overall_top",
            "candidate_data": (
                "actual W/L, primary/secondary turns, core and cooling groups"
            ),
        },
        {
            "slide": 3,
            "role": "overall_front",
            "candidate_data": "actual H and winding/core axial dimensions",
        },
        {
            "slide": 4,
            "role": "core_top",
            "candidate_data": (
                "candidate core outer dimensions, depth groups, plates/pads"
            ),
        },
        {
            "slide": 5,
            "role": "core_front",
            "candidate_data": (
                "window, leg/yoke, physical center-gap dimensions"
            ),
        },
        {
            "slide": 6,
            "role": "center_winding",
            "candidate_data": (
                "straight spans, rounded-corner radius ladder, WCP/pads"
            ),
        },
        {
            "slide": 7,
            "role": "side_winding",
            "candidate_data": "radial and axial side-winding stack dimensions",
        },
        {
            "slide": 8,
            "role": "winding_separation",
            "candidate_data": "center-to-side insulation/clearance dimensions",
        },
        {
            "slide": 9,
            "role": "conductor_pitch",
            "candidate_data": (
                "primary 5.0/1.6 mm and exact selected secondary foil/gap"
            ),
        },
    ]


def build_pipeline_manifest(
    *,
    authority_path: Path,
    authenticated: Mapping[str, Any],
    output: Path,
    params_records: Mapping[str, Mapping[str, Any]],
    reference_pdf: Mapping[str, Any],
    reference_pptx: Mapping[str, Any],
) -> dict[str, Any]:
    source = authenticated["source"]
    verification = authenticated["verification"]
    full_unrounded_params = output / params_records["full_unrounded"]["path"]
    rounded_params = (
        output / params_records["full_rounded_drawing_only"]["path"]
    )
    full_work = output / "work" / "full_unrounded"
    rounded_work = output / "work" / "full_rounded_drawing_only"
    rounded_project = output / "models" / "final_full_rounded_drawing_only.aedt"
    views_output = output / "drawings" / "views"
    artifact_workspace = output / "drawings" / "artifact_tool_workspace"
    manifest = {
        "schema_version": PIPELINE_SCHEMA,
        "campaign_id": "mft-goal-20260726",
        "authority": {
            "winner_authority": file_record(authority_path),
            "winner_authority_payload_sha256": authenticated["authority"][
                "payload_sha256"
            ],
            "selection_performed_by_pipeline": False,
            "candidate_promotion_performed_by_pipeline": False,
            "upstream_task_id": source["task_id"],
            "candidate_physics_sha256": source[
                "candidate_physics_sha256"
            ],
            "upstream_selection_receipt": source[
                "upstream_selection_receipt"
            ],
        },
        "scientific_contract": {
            "symmetric_verification": copy.deepcopy(verification),
            "turns": copy.deepcopy(authenticated["turns"]),
            "primary_conductor_thickness_mm": 5.0,
            "primary_turn_gap_mm": 1.6,
            "axis_contract": {
                "W": "drawing x / original 973-mm direction",
                "L": "perpendicular drawing y",
                "rotation_allowed": False,
                "axis_swap_allowed": False,
                "limits_mm": copy.deepcopy(GOAL_SIZE_LIMITS_MM),
            },
            "temperature_limits_C": copy.deepcopy(
                TEMPERATURE_FAMILY_LIMITS_C
            ),
            "resonance_min_Hz": 15_000.0,
            "Lm_primary_referred_target_H": TARGET_LM_H,
            "Lm_absolute_tolerance_H": TARGET_LM_ABS_TOLERANCE_H,
            "fixed_boundary_attestation": copy.deepcopy(
                authenticated["fixed_boundary_attestation"]
            ),
            "rounded_FEA_allowed": False,
        },
        "references": {
            "source_pdf": copy.deepcopy(dict(reference_pdf)),
            "source_pptx": copy.deepcopy(dict(reference_pptx)),
            "source_files_mutated": False,
            "template_slide_count": 9,
            "template_aspect_ratio": "16:9",
            "prior_audit": {
                "root": str(DRAWING_AUDIT_ROOT),
                "template_manifest": (
                    file_record(DRAWING_TEMPLATE_MANIFEST)
                    if DRAWING_TEMPLATE_MANIFEST.is_file()
                    else None
                ),
                "pdf_audit_metadata": (
                    file_record(DRAWING_PDF_AUDIT_METADATA)
                    if DRAWING_PDF_AUDIT_METADATA.is_file()
                    else None
                ),
                "every_source_slide_visually_inspected": True,
                "every_source_pdf_page_visually_inspected": True,
            },
        },
        "model_outputs": {
            "symmetric_nonrounded_verified": {
                "output": "models/final_symmetric_nonrounded.aedt",
                "source": source["retained_symmetric_aedt"],
                "action": "copy_exact_bytes_then_rehash",
                "solver_rerun": False,
                "full_model": False,
                "round_corner": False,
                "scientific_verification_model": True,
            },
            "full_unrounded_model_only": {
                "output": "models/final_full_unrounded.aedt",
                "params": copy.deepcopy(
                    dict(params_records["full_unrounded"])
                ),
                "command": _model_command(
                    params_path=full_unrounded_params,
                    rounded=False,
                    work_directory=full_work,
                ),
                "full_model": True,
                "round_corner": False,
                "solver_invoked": False,
                "scientific_verification_model": False,
            },
            "full_rounded_drawing_only": {
                "output": (
                    "models/final_full_rounded_drawing_only.aedt"
                ),
                "params": copy.deepcopy(
                    dict(params_records["full_rounded_drawing_only"])
                ),
                "command": _model_command(
                    params_path=rounded_params,
                    rounded=True,
                    work_directory=rounded_work,
                ),
                "full_model": True,
                "round_corner": True,
                "solver_invoked": False,
                "rounded_FEA_prohibited": True,
                "scientific_verification_model": False,
                "drawing_geometry_only": True,
            },
        },
        "drawing_views": {
            "depends_on": ["full_rounded_drawing_only"],
            "command": {
                "cwd": str(REPOSITORY_ROOT),
                "argv": [
                    str(Path(sys.executable).resolve()),
                    str(
                        (
                            REPOSITORY_ROOT
                            / "tools"
                            / "mft_goal_export_rounded_drawing_views.py"
                        ).resolve()
                    ),
                    "--project",
                    str(rounded_project.resolve()),
                    "--output",
                    str(views_output.resolve()),
                    "--background",
                    "white",
                    "--width",
                    "2400",
                    "--height",
                    "1600",
                ],
                "graphical_AEDT_required": True,
                "solver_invoked": False,
                "project_save_performed": False,
            },
            "required_views": [
                "overall_top",
                "overall_front",
                "overall_isometric",
                "center_winding_detail_top",
                "side_winding_detail_top",
            ],
        },
        "drawing_deck": {
            "depends_on": [
                "full_rounded_drawing_only",
                "rounded_drawing_views",
            ],
            "authoring_tool": "@oai/artifact-tool",
            "artifact_tool_minimum_version": "2.7.3",
            "artifact_tool_tested_version": "2.8.31",
            "workspace": str(artifact_workspace.resolve()),
            "template_policy": (
                "duplicate all nine source slides and edit inherited elements"
            ),
            "template_rebuild_allowed": False,
            "slide_contract": _slide_contract(),
            "dimension_source": (
                "sealed candidate params plus rounded AEDT geometry readback"
            ),
            "forbidden_dimension_notation": [
                "circular winding ID",
                "circular winding OD",
                "mean-turn diameter for the racetrack geometry",
            ],
            "required_dimension_notation": [
                "x/y straight spans",
                "corner radius and radius ladder",
                "conductor cross-section",
                "radial layer gaps/build",
                "z winding height and turn pitch",
                "core/WCP/pad clearances",
            ],
            "outputs": {
                "pptx": "drawings/final_design_drawing.pptx",
                "pdf": "drawings/final_design_drawing.pdf",
            },
            "visual_fidelity_note": (
                "The prior artifact-tool source render hid the cover subtitle "
                "and rendered some dashed dimension extensions as solid. "
                "Final fidelity must therefore be checked against the source "
                "PDF/PowerPoint render, not only artifact-tool raster output."
            ),
        },
        "qa_gates": {
            "pptx": [
                "artifact-tool inspect succeeds without truncation",
                "all nine slides render",
                "all nine slides visually inspected",
                "no overflow or out-of-bounds elements",
                "master/layout/template topology preserved",
                "cover subtitle and dashed dimension extensions match source",
            ],
            "pdf": [
                "export contains exactly nine pages",
                "all nine pages render at 240 dpi or higher",
                "all nine rendered pages visually inspected",
                "all displayed values match sealed candidate authority",
            ],
            "aedt": [
                "three declared AEDT files exist and have sealed SHA/size records",
                "symmetric AEDT bytes equal the authenticated retained source",
                "full unrounded model readback has full_model=1, round_corner=0",
                (
                    "rounded drawing model readback has full_model=1, "
                    "round_corner=1 and no solved-result claim"
                ),
            ],
        },
        "parallel_execution": [
            {
                "wave": 1,
                "parallel": [
                    "copy_and_rehash_symmetric_nonrounded",
                    "build_full_unrounded_model_only",
                    "build_full_rounded_drawing_only",
                    "prepare_candidate_spec_and_template_edit_map",
                ],
            },
            {
                "wave": 2,
                "parallel": [
                    "export_rounded_drawing_views",
                    "AEDT_open_and_geometry_readback_QA",
                ],
            },
            {
                "wave": 3,
                "parallel": [
                    "author_PPTX_from_all_nine_cloned_template_slides",
                ],
            },
            {
                "wave": 4,
                "parallel": [
                    "render_and_inspect_all_PPTX_slides",
                    "export_render_and_inspect_all_PDF_pages",
                    "seal_final_AEDT_file_records",
                ],
            },
        ],
        "repository_boundaries": {
            "scheduler_project": "MFT_1MW_2026v1",
            "scheduler_repository_source_included": False,
            "scheduler_mutation_performed": False,
            "MFT_repository_source_only": True,
            "branch_creation_performed": False,
        },
    }
    return seal(manifest)


def prepare(
    *,
    winner_authority: Path,
    output: Path,
    reference_pdf: Path = DEFAULT_REFERENCE_PDF,
    reference_pptx: Path = DEFAULT_REFERENCE_PPTX,
) -> Path:
    authority_path = winner_authority.resolve(strict=True)
    authority = _read_json(authority_path, "winner authority")
    authenticated = validate_winner_authority(
        authority, authority_directory=authority_path.parent
    )
    pdf_record = _reference_record(
        reference_pdf,
        expected_sha256=REFERENCE_PDF_SHA256,
        label="drawing reference PDF",
    )
    pptx_record = _reference_record(
        reference_pptx,
        expected_sha256=REFERENCE_PPTX_SHA256,
        label="drawing reference PPTX",
    )
    root = output.resolve()
    variants = derive_model_variants(authenticated["params"])
    params_root = root / "params"
    param_paths = {
        "full_unrounded": _write_immutable_json(
            params_root / "full_final_nonrounded.model_only.json",
            variants["full_unrounded"],
        ),
        "full_rounded_drawing_only": _write_immutable_json(
            params_root
            / "full_final_rounded_drawing_only.model_only.json",
            variants["full_rounded_drawing_only"],
        ),
    }
    params_records = {
        name: file_record(path, relative_to=root)
        for name, path in param_paths.items()
    }
    manifest = build_pipeline_manifest(
        authority_path=authority_path,
        authenticated=authenticated,
        output=root,
        params_records=params_records,
        reference_pdf=pdf_record,
        reference_pptx=pptx_record,
    )
    return _write_immutable_json(root / "pipeline_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-authority", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reference-pdf", type=Path, default=DEFAULT_REFERENCE_PDF
    )
    parser.add_argument(
        "--reference-pptx", type=Path, default=DEFAULT_REFERENCE_PPTX
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare(
        winner_authority=args.winner_authority,
        output=args.output,
        reference_pdf=args.reference_pdf,
        reference_pptx=args.reference_pptx,
    )
    print(
        json.dumps(
            {
                "event": "final_artifact_pipeline_prepared",
                "manifest": str(manifest),
                "selection_performed": False,
                "scheduler_mutation_performed": False,
                "solver_invoked": False,
                "rounded_FEA_performed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinalArtifactPipelineError as exc:
        print(
            json.dumps(
                {
                    "event": "final_artifact_pipeline_error",
                    "error": str(exc),
                    "selection_performed": False,
                    "scheduler_mutation_performed": False,
                    "solver_invoked": False,
                    "rounded_FEA_performed": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        raise SystemExit(2) from exc
