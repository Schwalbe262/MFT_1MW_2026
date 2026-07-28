"""Authenticate and project the deadline Full-EM HIT publication.

The deadline search produces two deliberately different validation states:

* ``split``: a completed Full-EM matrix/capacitance HIT is authoritative for
  exterior geometry, Llt and half-Lm resonance while a completed Standard
  pipeline corroborates B, loss, temperature and the fixed design controls.
  Complete-Full loss/thermal validation is still pending.
* ``complete``: a completed Full pipeline is authoritative for every hard
  specification and every displayed measurement; Standard remains an
  independent corroboration lane.

Both states are read-only UI publications.  Neither is engineering approval.
The configured file is a hash-pinned selector envelope.  This reader opens
and authenticates its sealed candidate-set BOM and every referenced result on
each coherent read so that a stale or replaced nested artifact fails closed.
"""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from regression_260707.optimization.geometry_metrics import bounding_box_lit


SELECTION_SCHEMA_V1 = "mft-deadline-8010-first-complete-pass-selection-v1"
SELECTION_SCHEMA_V2 = "mft-deadline-8010-hit-validation-selection-v2"
COMPLETE_PUBLICATION_SCHEMA = (
    "mft-deadline-8010-actual-pair-publication-v1"
)
SPLIT_PUBLICATION_SCHEMA = (
    "mft-deadline-8010-split-validated-publication-v1"
)
BOM_SCHEMA = "mft-deadline-full-em-hit-candidate-set-bom-v1"

COMPLETE_STATUS = "first_joined_actual_complete_pass_ready"
SPLIT_STATUS = "provisional_full_em_standard_pass"

STANDARD_SOLVER_REVISION = "8a8d90f68e8728669282f586f24304c7cc807029"
FULL_SOLVER_REVISION = "146142be579e2f3e45f12961214018a4e28445c5"
LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"

MAX_BOM_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 32 * 1024 * 1024
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")

SAFE_AUTHORITY = {
    "offline_only": True,
    "live_8010_mutated": False,
    "scheduler_8002_mutated": False,
    "canonical_dataset_mutated": False,
    "final_design_approved": False,
    "engineering_signoff_required": True,
}

HARD_CHECK_NAMES = (
    "width",
    "length",
    "height",
    "leakage_inductance",
    "half_magnetizing_resonance",
    "design_flux_density",
    "maximum_temperature",
    "minimum_insulation",
    "core_group_count",
    "primary_conductor_thickness",
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _same(actual: Any, expected: Any, label: str, *, tolerance: float = 1e-8) -> None:
    if isinstance(expected, bool):
        if actual is not expected:
            raise ValueError(f"{label} boolean identity drifted")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        observed = _finite(actual, label)
        target = _finite(expected, f"{label} expected")
        if not math.isclose(
            observed,
            target,
            rel_tol=0.0,
            abs_tol=max(tolerance, abs(target) * 1e-10),
        ):
            raise ValueError(f"{label} numeric identity drifted")
        return
    if actual != expected:
        raise ValueError(f"{label} identity drifted")


def _clean_tree(value: Any, label: str = "publication", depth: int = 0) -> None:
    if depth > 32:
        raise ValueError(f"{label} nesting exceeds the bounded contract")
    if isinstance(value, dict):
        if len(value) > 2_000:
            raise ValueError(f"{label} object is unbounded")
        for key, nested in value.items():
            if (
                not isinstance(key, str)
                or not key
                or len(key) > 240
                or any(ord(character) < 32 for character in key)
            ):
                raise ValueError(f"{label} has an invalid key")
            _clean_tree(nested, f"{label}.{key}", depth + 1)
        return
    if isinstance(value, list):
        if len(value) > 2_000:
            raise ValueError(f"{label} list is unbounded")
        for index, nested in enumerate(value):
            _clean_tree(nested, f"{label}[{index}]", depth + 1)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, str):
        if len(value) > 1_000_000 or "\x00" in value:
            raise ValueError(f"{label} contains an invalid string")


def _verify_payload_digest(value: dict[str, Any], label: str) -> None:
    body = dict(value)
    declared = str(body.pop("payload_sha256", "")).lower()
    if HEX64.fullmatch(declared) is None:
        raise ValueError(f"{label} payload SHA-256 is malformed")
    if declared != _canonical_sha256(body):
        raise ValueError(f"{label} payload SHA-256 mismatch")


def _verify_authority(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} authority is missing")
    for key, expected in SAFE_AUTHORITY.items():
        if value.get(key) is not expected:
            raise ValueError(f"{label} authority drifted: {key}")
    for mutation_flag in (
        "scheduler_task_mutation_performed",
        "task_mutation_performed",
        "automatic_publish_performed",
        "automatic_promotion_allowed",
    ):
        if mutation_flag in value and value.get(mutation_flag) is not False:
            raise ValueError(f"{label} authority drifted: {mutation_flag}")
    if (
        "strict_full_equivalent" in value
        and value.get("strict_full_equivalent") is not False
    ):
        raise ValueError(f"{label} strict-Full authority was overstated")
    return value


def _verify_reference(
    reference: Any,
    label: str,
    legacy: Any,
    *,
    max_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    try:
        path, payload = legacy._verify_reference(reference, label, max_bytes)
    except legacy.DeadlineDesignError:
        raise
    _clean_tree(payload, label)
    return path, payload


def _hard_spec(value: Any, legacy: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(
        legacy.EXPECTED_HARD_SPEC
    ):
        raise ValueError(f"{label} hard-spec identity drifted")
    for key, expected in legacy.EXPECTED_HARD_SPEC.items():
        _same(value.get(key), expected, f"{label}.hard_spec.{key}")
    return dict(value)


def _load_bom(
    selector: dict[str, Any],
    legacy: Any,
) -> tuple[Path, dict[str, Any]]:
    source = selector.get("source_candidate_bom")
    path, bom = _verify_reference(
        source,
        "deadline HIT candidate-set BOM",
        legacy,
        max_bytes=MAX_BOM_BYTES,
    )
    if bom.get("schema_version") != BOM_SCHEMA:
        raise ValueError("deadline HIT candidate-set BOM schema drifted")
    _verify_payload_digest(bom, "deadline HIT candidate-set BOM")
    if not isinstance(source, dict) or (
        source.get("payload_sha256") != bom.get("payload_sha256")
    ):
        raise ValueError("selector/BOM payload identity drifted")
    _hard_spec(bom.get("active_hard_spec"), legacy, "candidate-set BOM")
    _verify_authority(bom.get("authority"), "candidate-set BOM")
    candidates = bom.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 32:
        raise ValueError("candidate-set BOM candidate inventory is invalid")
    keys: set[str] = set()
    task_ids: set[int] = set()
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate-set BOM candidate {position} is invalid")
        key = candidate.get("key")
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 200
            or key in keys
        ):
            raise ValueError("candidate-set BOM candidate key drifted")
        keys.add(key)
        for digest_key in (
            "candidate_identity_sha256",
            "physical_candidate_digest",
            "followup_profile_sha256",
            "sweep_profile_sha256",
        ):
            if HEX64.fullmatch(str(candidate.get(digest_key) or "").lower()) is None:
                raise ValueError(
                    f"candidate-set BOM {key} {digest_key} is malformed"
                )
        variant_name = str(candidate.get("cooling_variant") or "")
        variant = legacy.ALLOWED_TIM_COOLING_VARIANTS.get(variant_name)
        if variant is None:
            raise ValueError(
                f"candidate-set BOM {key} cooling variant is not allowlisted"
            )
        _same(
            candidate.get("fan_velocity_m_s"),
            variant["fan_velocity_m_s"],
            f"candidate-set BOM {key} fan velocity",
        )
        geometry = candidate.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"candidate-set BOM {key} geometry is missing")
        for geometry_key in (
            "width_mm",
            "length_mm",
            "height_mm",
            "volume_L",
            "footprint_cm2",
        ):
            _finite(geometry.get(geometry_key), f"{key}.{geometry_key}")
        task_map = candidate.get("validation_task_ids")
        if not isinstance(task_map, dict):
            raise ValueError(f"candidate-set BOM {key} task map is missing")
        for lane in ("full_em_hit", "standard", "complete_full"):
            task_id = task_map.get(lane)
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id <= 0
                or task_id in task_ids
            ):
                # A Full-EM HIT is intentionally shared by fan variants.  Its
                # task ID may repeat; the complete lanes may not.
                if lane != "full_em_hit" or task_id not in task_ids:
                    raise ValueError(
                        f"candidate-set BOM {key} task map drifted: {lane}"
                    )
            task_ids.add(task_id)
        evidence = candidate.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"candidate-set BOM {key} evidence is missing")
        for required in (
            "sweep_result",
            "standard_payload",
            "complete_full_payload",
        ):
            if not isinstance(evidence.get(required), dict):
                raise ValueError(
                    f"candidate-set BOM {key} evidence is missing: {required}"
                )
    return path, bom


def _candidate_from_bom(
    bom: dict[str, Any],
    candidate_key: str,
) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in bom["candidates"]
        if candidate.get("key") == candidate_key
    ]
    if len(matches) != 1:
        raise ValueError("selected candidate does not uniquely join to the BOM")
    return matches[0]


def _status_result(
    reference: Any,
    task_id: int,
    lane: str,
    bom_candidate: dict[str, Any],
    legacy: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _, wrapper = _verify_reference(
        reference,
        f"{lane} task {task_id} result",
        legacy,
        max_bytes=MAX_RESULT_BYTES,
    )
    task = wrapper.get("task")
    if isinstance(task, dict):
        observed_id = task.get("id", task.get("task_id"))
        task_name = task.get("name", task.get("task_name"))
        state = task.get("status", task.get("state"))
        exit_code = task.get("exit_code")
    else:
        observed_id = wrapper.get("task_id")
        task_name = wrapper.get("task_name")
        state = wrapper.get("task_status", wrapper.get("result_state"))
        exit_code = wrapper.get("exit_code", 0 if state == "completed" else None)
    if (
        observed_id != task_id
        or state != "completed"
        or exit_code not in (0, None)
        or not isinstance(task_name, str)
        or not task_name
    ):
        raise ValueError(f"{lane} task terminal identity drifted")
    contract = wrapper.get("terminal_result_contract")
    if isinstance(contract, dict):
        result = contract.get("result")
        contract_pass = contract.get("pass")
    else:
        result = wrapper.get("result")
        contract_pass = wrapper.get(
            "result_contract_valid",
            wrapper.get("result_contract_pass"),
        )
    if contract_pass is not True or not isinstance(result, dict):
        raise ValueError(f"{lane} task result contract failed closed")
    _clean_tree(result, f"{lane} task result")
    for key in (
        "physical_candidate_digest",
        "followup_profile_sha256",
        "sweep_hit_profile_sha256",
    ):
        expected_key = (
            "sweep_profile_sha256"
            if key == "sweep_hit_profile_sha256"
            else key
        )
        observed = wrapper.get(key)
        if observed is not None and observed != bom_candidate.get(expected_key):
            raise ValueError(f"{lane} task candidate identity drifted: {key}")

    payload_key = (
        "standard_payload" if lane == "standard" else "complete_full_payload"
    )
    _, payload = _verify_reference(
        bom_candidate["evidence"][payload_key],
        f"{lane} task payload",
        legacy,
        max_bytes=MAX_PAYLOAD_BYTES,
    )
    if payload.get("name") != task_name:
        raise ValueError(f"{lane} task name/payload identity drifted")
    raw_parameters = payload.get("parameters")
    if not isinstance(raw_parameters, dict):
        raw_parameters = payload.get("payload")
        raw_parameters = (
            raw_parameters.get("parameters")
            if isinstance(raw_parameters, dict)
            else None
        )
    if not isinstance(raw_parameters, dict):
        raw_parameters = payload.get("cand")
    if not isinstance(raw_parameters, dict):
        snapshots = bom_candidate.get("sealed_parameter_snapshots")
        snapshots = snapshots if isinstance(snapshots, dict) else {}
        raw_parameters = snapshots.get(
            "standard" if lane == "standard" else "complete_full"
        )
    if not isinstance(raw_parameters, dict) or not raw_parameters:
        raise ValueError(f"{lane} task payload parameters are missing")
    return wrapper, result, raw_parameters


def _raw_sweep_result(
    bom_candidate: dict[str, Any],
    legacy: Any,
) -> dict[str, Any]:
    _, result = _verify_reference(
        bom_candidate["evidence"]["sweep_result"],
        "Full-EM HIT result",
        legacy,
        max_bytes=MAX_RESULT_BYTES,
    )
    return result


def _physical_llt_uH(result: dict[str, Any], label: str) -> float:
    if "cap_L_leakage_H" in result:
        return _finite(
            result.get("cap_L_leakage_H"), f"{label}.cap_L_leakage_H"
        ) * 1e6
    return _finite(result.get("Llt"), f"{label}.Llt") * _finite(
        result.get("cap_inductance_restoration_factor"),
        f"{label}.cap_inductance_restoration_factor",
    )


def _half_lm_resonance(
    result: dict[str, Any],
    factor: float,
    label: str,
) -> float:
    tx = _finite(result.get("f_res_tx_self_Hz"), f"{label}.f_res_tx_self_Hz")
    rx = _finite(result.get("f_res_rx_self_Hz"), f"{label}.f_res_rx_self_Hz")
    return min(tx, rx) / math.sqrt(factor)


def _max_temperature(result: dict[str, Any], label: str) -> float:
    temperatures = [
        _finite(value, f"{label}.{key}")
        for key, value in result.items()
        if key.startswith("T_max_")
        or (key.startswith("Tprobe_") and key.endswith("_max"))
    ]
    if not temperatures:
        raise ValueError(f"{label} has no thermal maxima")
    return max(temperatures)


def _losses(result: dict[str, Any], label: str) -> dict[str, float]:
    mapping = {
        "rated_output_W": "P_target",
        "core_W": "P_core_total",
        "primary_winding_W": "P_Tx_main_group",
        "secondary_center_winding_W": "P_Rx_main_group",
        "secondary_side_winding_W": "P_Rx_side_total",
        "winding_total_W": "P_winding_total",
        "core_cold_plate_W": "P_core_plate_total",
        "winding_cold_plate_W": "P_wcp_total",
    }
    values = {
        output: _finite(result.get(source), f"{label}.{source}")
        for output, source in mapping.items()
    }
    if values["rated_output_W"] <= 0 or any(
        value < 0
        for key, value in values.items()
        if key != "rated_output_W"
    ):
        raise ValueError(f"{label} has non-physical loss values")
    component = (
        values["primary_winding_W"]
        + values["secondary_center_winding_W"]
        + values["secondary_side_winding_W"]
    )
    if not math.isclose(
        component,
        values["winding_total_W"],
        rel_tol=2e-3,
        abs_tol=1e-3,
    ):
        raise ValueError(f"{label} winding loss balance drifted")
    values["modeled_total_W"] = (
        values["core_W"]
        + values["winding_total_W"]
        + values["core_cold_plate_W"]
        + values["winding_cold_plate_W"]
    )
    values["efficiency_pct"] = (
        values["rated_output_W"]
        / (values["rated_output_W"] + values["modeled_total_W"])
        * 100.0
    )
    return values


def _verify_result_contract(
    result: dict[str, Any],
    *,
    lane: str,
    fan_velocity: float,
    pad_thickness: float,
    full_pipeline: bool,
    expected_parameters: dict[str, Any] | None,
    legacy: Any,
) -> None:
    full_model = lane in {"full_em_hit", "complete_full"}
    solver_revision = (
        FULL_SOLVER_REVISION if full_model else STANDARD_SOLVER_REVISION
    )
    expected = {
        "full_model": 1 if full_model else 0,
        "matrix_on": 1,
        "cap_on": 1,
        "git_hash": solver_revision,
        "pyaedt_library_git_hash": LIBRARY_REVISION,
    }
    if full_model:
        expected["n_explicit_turns"] = 2
    if full_pipeline:
        expected.update({
            "loss_on": 1,
            "thermal_on": 1,
            "result_valid_em": 1,
            "result_valid_thermal": 1,
        })
    for key, expected_value in expected.items():
        _same(result.get(key), expected_value, f"{lane}.{key}")
    if full_pipeline and result.get("thermal_extraction_complete") not in {
        True,
        1,
    }:
        raise ValueError(f"{lane} thermal extraction is incomplete")
    if full_pipeline:
        _same(result.get("fan_config"), "dual", f"{lane}.fan_config")
        _same(result.get("fan_velocity"), fan_velocity, f"{lane}.fan_velocity")
        _same(result.get("plate_temp"), 50.0, f"{lane}.plate_temp")
        _same(result.get("air_temp"), 50.0, f"{lane}.air_temp")
        _same(
            result.get("core_plate_pad_t"),
            pad_thickness,
            f"{lane}.core_plate_pad_t",
        )
        _same(
            result.get("wcp_pad_t"),
            pad_thickness,
            f"{lane}.wcp_pad_t",
        )
        for key, expected_value in (
            (
                "thermal_pad_material_policy",
                legacy.EXPECTED_TIM["material_policy"],
            ),
            (
                "thermal_pad_native_readback_contract_version",
                legacy.EXPECTED_TIM["native_readback_contract_version"],
            ),
            ("thermal_pad_native_readback_attested", 1),
            (
                "thermal_pad_native_thermal_conductivity_W_mK",
                legacy.EXPECTED_TIM["native_thermal_conductivity_W_mK"],
            ),
            (
                "thermal_pad_native_electrical_conductivity_S_m",
                legacy.EXPECTED_TIM["native_electrical_conductivity_S_m"],
            ),
        ):
            _same(result.get(key), expected_value, f"{lane}.{key}")
    if expected_parameters is not None:
        for key, expected_value in expected_parameters.items():
            if key not in result:
                raise ValueError(f"{lane} result omitted parameter {key}")
            _same(result.get(key), expected_value, f"{lane}.{key}")


def _geometry_from_result(
    result: dict[str, Any],
    expected: dict[str, Any],
    label: str,
) -> dict[str, float]:
    try:
        volume, dimensions = bounding_box_lit(result)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} geometry cannot be replayed") from exc
    observed = {
        "width_mm": float(dimensions[0]),
        "length_mm": float(dimensions[1]),
        "height_mm": float(dimensions[2]),
        "volume_L": float(volume),
        "footprint_cm2": float(dimensions[0] * dimensions[1] / 100.0),
    }
    for key, value in observed.items():
        _same(value, expected.get(key), f"{label}.{key}", tolerance=1e-5)
    return observed


def _provenance_by_lane(
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    values = candidate.get("validation_provenance")
    if not isinstance(values, list):
        raise ValueError("candidate validation provenance is missing")
    lanes: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("candidate validation provenance is malformed")
        lane = value.get("lane")
        if not isinstance(lane, str) or lane in lanes:
            raise ValueError("candidate validation provenance lane drifted")
        lanes[lane] = value
    return lanes


def _verify_measurements(
    candidate: dict[str, Any],
    bom_candidate: dict[str, Any],
    *,
    final: bool,
    legacy: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task_map = bom_candidate["validation_task_ids"]
    lanes = _provenance_by_lane(candidate)
    required_lanes = {"full_em_hit", "standard"}
    if final:
        required_lanes.add("complete_full")
    if not required_lanes.issubset(lanes):
        raise ValueError("candidate validation provenance is incomplete")

    full_em = _raw_sweep_result(bom_candidate, legacy)
    _verify_result_contract(
        full_em,
        lane="full_em_hit",
        fan_velocity=float(bom_candidate["fan_velocity_m_s"]),
        pad_thickness=1.0,
        full_pipeline=False,
        expected_parameters=None,
        legacy=legacy,
    )
    _geometry_from_result(
        full_em,
        bom_candidate["geometry"],
        "Full-EM HIT",
    )
    if lanes["full_em_hit"].get("task_id") != task_map["full_em_hit"]:
        raise ValueError("Full-EM HIT provenance task drifted")

    standard_lane = lanes["standard"]
    if (
        standard_lane.get("task_id") != task_map["standard"]
        or standard_lane.get("state") != "actual_corroboration_pass"
    ):
        raise ValueError("Standard corroboration provenance drifted")
    _, standard, standard_parameters = _status_result(
        standard_lane.get("result_reference"),
        task_map["standard"],
        "standard",
        bom_candidate,
        legacy,
    )
    variant = legacy.ALLOWED_TIM_COOLING_VARIANTS[
        bom_candidate["cooling_variant"]
    ]
    _verify_result_contract(
        standard,
        lane="standard",
        fan_velocity=variant["fan_velocity_m_s"],
        pad_thickness=variant["pad_thickness_mm"],
        full_pipeline=True,
        expected_parameters=standard_parameters,
        legacy=legacy,
    )
    _geometry_from_result(
        standard,
        bom_candidate["geometry"],
        "Standard corroboration",
    )

    full = full_em
    if final:
        complete_lane = lanes["complete_full"]
        if (
            complete_lane.get("task_id") != task_map["complete_full"]
            or complete_lane.get("state") != "actual_hard_spec_pass"
            or complete_lane.get("strict_full_equivalent") is not False
        ):
            raise ValueError("complete-Full provenance authority drifted")
        _, full, full_parameters = _status_result(
            complete_lane.get("result_reference"),
            task_map["complete_full"],
            "complete_full",
            bom_candidate,
            legacy,
        )
        _verify_result_contract(
            full,
            lane="complete_full",
            fan_velocity=variant["fan_velocity_m_s"],
            pad_thickness=variant["pad_thickness_mm"],
            full_pipeline=True,
            expected_parameters=full_parameters,
            legacy=legacy,
        )
        _geometry_from_result(
            full,
            bom_candidate["geometry"],
            "complete-Full",
        )

    authoritative_em = full if final else full_em
    thermal_loss = full if final else standard
    factor = legacy.EXPECTED_HARD_SPEC["magnetizing_inductance_factor"]
    measurements = {
        "geometry": dict(bom_candidate["geometry"]),
        "Llt_uH": _physical_llt_uH(authoritative_em, "authoritative EM"),
        "half_lm_resonance_Hz": _half_lm_resonance(
            authoritative_em,
            factor,
            "authoritative EM",
        ),
        "f_res_tx_self_Hz": _finite(
            authoritative_em.get("f_res_tx_self_Hz"),
            "authoritative EM.f_res_tx_self_Hz",
        ),
        "f_res_rx_self_Hz": _finite(
            authoritative_em.get("f_res_rx_self_Hz"),
            "authoritative EM.f_res_rx_self_Hz",
        ),
        "f_res_interwinding_Hz": _finite(
            authoritative_em.get("f_res_interwinding_Hz"),
            "authoritative EM.f_res_interwinding_Hz",
        ),
        "C_tx_tx_F": _finite(
            authoritative_em.get("C_tx_tx_F"),
            "authoritative EM.C_tx_tx_F",
        ),
        "C_rx_rx_F": _finite(
            authoritative_em.get("C_rx_rx_F"),
            "authoritative EM.C_rx_rx_F",
        ),
        "C_tx_rx_F": _finite(
            authoritative_em.get("C_tx_rx_F"),
            "authoritative EM.C_tx_rx_F",
        ),
        "B_design_T": _finite(
            thermal_loss.get("B_design_square_material_analytic"),
            "authoritative B",
        ),
        "B_mean_T": _finite(
            thermal_loss.get(
                "B_mean_core_material",
                thermal_loss.get("B_mean_core"),
            ),
            "authoritative B mean",
        ),
        "B_max_T": _finite(
            thermal_loss.get(
                "B_max_core_material",
                thermal_loss.get("B_max_core"),
            ),
            "authoritative B max",
        ),
        "maximum_temperature_C": _max_temperature(
            thermal_loss,
            "authoritative thermal",
        ),
        "losses": _losses(thermal_loss, "authoritative loss"),
        "parameters": {
            key: value
            for key, value in thermal_loss.items()
            if key in bom_candidate.get(
                "sealed_parameter_snapshots", {}
            ).get("standard", {})
        },
        "raw_standard": standard,
        "raw_authoritative": thermal_loss,
    }
    return measurements, standard, full


def _candidate_metric(
    candidate: dict[str, Any],
    key: str,
    *,
    report_key: str | None = None,
) -> Any:
    if key in candidate:
        return candidate[key]
    report = candidate.get("report")
    if isinstance(report, dict):
        return report.get(report_key or key)
    return None


def _verify_candidate_values(
    candidate: dict[str, Any],
    bom_candidate: dict[str, Any],
    measurements: dict[str, Any],
    *,
    final: bool,
    legacy: Any,
) -> None:
    for key, expected in (
        ("candidate_identity_sha256", bom_candidate["candidate_identity_sha256"]),
        ("candidate_digest", bom_candidate["physical_candidate_digest"]),
        ("cooling_variant", bom_candidate["cooling_variant"]),
    ):
        _same(candidate.get(key), expected, f"candidate.{key}")
    expected_flags = {
        "spec_status": (
            "pass" if final else "provisional_pass_complete_full_pending"
        ),
        "artifact_hydrated": True,
        "embedded_authenticated_result": True,
        "fea_verified": True,
        "standard_corroboration_pass": True,
        "full_actual_hard_pass": final,
        "strict_full_equivalent": False,
        "final_design_approved": False,
        "engineering_review_required": True,
    }
    if final:
        expected_flags["paired_actual_hard_pass"] = True
    else:
        expected_flags.update({
            "paired_actual_hard_pass": False,
            "full_em_actual_pass": True,
            "complete_full_pending": True,
        })
    for key, expected in expected_flags.items():
        _same(candidate.get(key), expected, f"candidate.{key}")

    geometry = measurements["geometry"]
    for key, report_key, expected in (
        ("size_W_mm", "size_W_mm", geometry["width_mm"]),
        ("size_L_mm", "size_L_mm", geometry["length_mm"]),
        ("size_H_mm", "size_H_mm", geometry["height_mm"]),
        ("volume_L", "volume_L", geometry["volume_L"]),
        ("footprint_cm2", "footprint_cm2", geometry["footprint_cm2"]),
        ("pred_Llt_phys", "pred_leakage_inductance_uH", measurements["Llt_uH"]),
        (
            "pred_f_res_min_screen_Hz",
            "pred_f_res_min_screen_Hz",
            measurements["half_lm_resonance_Hz"],
        ),
        ("C_tx_tx_F", "C_tx_tx_F", measurements["C_tx_tx_F"]),
        ("C_rx_rx_F", "C_rx_rx_F", measurements["C_rx_rx_F"]),
        ("C_tx_rx_F", "C_tx_rx_F", measurements["C_tx_rx_F"]),
        ("B_design_analytic_T", "B_design_analytic_T", measurements["B_design_T"]),
        ("pred_B_mean_core", "pred_B_mean_core", measurements["B_mean_T"]),
        (
            "diagnostic_pred_B_max_core",
            "diagnostic_pred_B_max_core",
            measurements["B_max_T"],
        ),
        (
            "pred_max_temperature_C",
            "pred_max_temperature_C",
            measurements["maximum_temperature_C"],
        ),
        (
            "total_loss_W",
            "pred_total_loss_W",
            measurements["losses"]["modeled_total_W"],
        ),
    ):
        _same(
            _candidate_metric(candidate, key, report_key=report_key),
            expected,
            f"candidate.{key}",
            tolerance=1e-5,
        )
    _same(
        candidate.get("resonance_minimum_required_Hz"),
        legacy.EXPECTED_HARD_SPEC["resonance_min_Hz"],
        "candidate.resonance_minimum_required_Hz",
    )
    _same(
        candidate.get("resonance_margin_Hz"),
        measurements["half_lm_resonance_Hz"]
        - legacy.EXPECTED_HARD_SPEC["resonance_min_Hz"],
        "candidate.resonance_margin_Hz",
        tolerance=1e-5,
    )
    variant = legacy.ALLOWED_TIM_COOLING_VARIANTS[
        bom_candidate["cooling_variant"]
    ]
    for key, expected in (
        ("fan_config", "dual"),
        ("fan_velocity_m_s", variant["fan_velocity_m_s"]),
        ("plate_temp_C", 50.0),
        ("air_temp_C", 50.0),
        ("core_thermal_pad_thickness_mm", variant["pad_thickness_mm"]),
        ("winding_thermal_pad_thickness_mm", variant["pad_thickness_mm"]),
    ):
        _same(candidate.get(key), expected, f"candidate.{key}")
    minimum_insulation = _finite(
        bom_candidate.get("known_constraints", {})
        .get("minimum_insulation", {})
        .get("observed"),
        "candidate minimum insulation",
    )
    _same(
        candidate.get("min_insulation_mm"),
        minimum_insulation,
        "candidate.min_insulation_mm",
    )
    parameters = candidate.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("candidate local-GUI parameters are missing")
    for key in legacy.REQUIRED_PARAMETERS:
        _finite(parameters.get(key), f"candidate.parameters.{key}")
    for key in legacy.REQUIRED_STRING_PARAMETERS:
        value = parameters.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"candidate.parameters.{key} is invalid")
    for key, value in measurements["parameters"].items():
        if key in parameters:
            _same(parameters[key], value, f"candidate.parameters.{key}")


def _hard_checks(
    measurements: dict[str, Any],
    bom_candidate: dict[str, Any],
    legacy: Any,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    spec = legacy.EXPECTED_HARD_SPEC
    geometry = measurements["geometry"]
    parameters = measurements["parameters"]
    minimum_insulation = _finite(
        bom_candidate.get("known_constraints", {})
        .get("minimum_insulation", {})
        .get("observed"),
        "minimum insulation",
    )
    n_core = _finite(parameters.get("n_core_group"), "n_core_group")
    cw1 = _finite(parameters.get("cw1"), "cw1")
    llt_low = spec["Llt_target_uH"] - spec["Llt_tol_uH"]
    llt_high = spec["Llt_target_uH"] + spec["Llt_tol_uH"]
    values = {
        "width": geometry["width_mm"],
        "length": geometry["length_mm"],
        "height": geometry["height_mm"],
        "leakage_inductance": measurements["Llt_uH"],
        "half_magnetizing_resonance": measurements[
            "half_lm_resonance_Hz"
        ],
        "design_flux_density": measurements["B_design_T"],
        "maximum_temperature": measurements["maximum_temperature_C"],
        "minimum_insulation": minimum_insulation,
        "core_group_count": n_core,
        "primary_conductor_thickness": cw1,
    }
    checks = {
        "width": values["width"] <= spec["size_W_max_mm"],
        "length": values["length"] <= spec["size_L_max_mm"],
        "height": values["height"] <= spec["size_H_max_mm"],
        "leakage_inductance": llt_low
        <= values["leakage_inductance"]
        <= llt_high,
        "half_magnetizing_resonance": (
            values["half_magnetizing_resonance"] >= spec["resonance_min_Hz"]
        ),
        "design_flux_density": (
            values["design_flux_density"] <= spec["B_limit_T"]
        ),
        "maximum_temperature": (
            values["maximum_temperature"] <= spec["T_limit_C"]
        ),
        "minimum_insulation": (
            values["minimum_insulation"] >= spec["insulation_min_mm"]
        ),
        "core_group_count": (
            values["core_group_count"] <= spec["n_core_group_max"]
        ),
        "primary_conductor_thickness": math.isclose(
            values["primary_conductor_thickness"],
            spec["primary_conductor_thickness_mm"],
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
    }
    ui = {
        "size_W": {
            "value": values["width"],
            "limit": spec["size_W_max_mm"],
            "operator": "<=",
            "pass": checks["width"],
        },
        "size_L": {
            "value": values["length"],
            "limit": spec["size_L_max_mm"],
            "operator": "<=",
            "pass": checks["length"],
        },
        "size_H": {
            "value": values["height"],
            "limit": spec["size_H_max_mm"],
            "operator": "<=",
            "pass": checks["height"],
        },
        "llt": {
            "value": values["leakage_inductance"],
            "limit": [llt_low, llt_high],
            "target_uH": spec["Llt_target_uH"],
            "tolerance_uH": spec["Llt_tol_uH"],
            "pass": checks["leakage_inductance"],
        },
        "resonance": {
            "value": values["half_magnetizing_resonance"],
            "limit": spec["resonance_min_Hz"],
            "minimum_Hz": spec["resonance_min_Hz"],
            "direction": "minimum",
            "operator": ">=",
            "pass": checks["half_magnetizing_resonance"],
        },
        "temperature": {
            "value": values["maximum_temperature"],
            "limit": spec["T_limit_C"],
            "operator": "<=",
            "pass": checks["maximum_temperature"],
        },
        "bfield": {
            "value": values["design_flux_density"],
            "limit": spec["B_limit_T"],
            "operator": "<=",
            "pass": checks["design_flux_density"],
        },
        "insulation": {
            "value": values["minimum_insulation"],
            "limit": spec["insulation_min_mm"],
            "operator": ">=",
            "pass": checks["minimum_insulation"],
        },
        "core_group": {
            "value": values["core_group_count"],
            "limit": spec["n_core_group_max"],
            "operator": "<=",
            "pass": checks["core_group_count"],
        },
        "primary_thickness": {
            "value": values["primary_conductor_thickness"],
            "limit": spec["primary_conductor_thickness_mm"],
            "operator": "=",
            "pass": checks["primary_conductor_thickness"],
        },
    }
    return checks, ui


def _validate_lane_measurement_echoes(
    candidate: dict[str, Any],
    checks: dict[str, bool],
    *,
    final: bool,
) -> None:
    paired = candidate.get("paired_measurements")
    if not isinstance(paired, dict):
        raise ValueError("candidate paired measurements are missing")
    standard = paired.get("standard")
    if not isinstance(standard, dict) or (
        standard.get("pass") is not True
        or standard.get("status") != "actual_corroboration_pass"
    ):
        raise ValueError("candidate Standard corroboration gate drifted")
    standard_policy = standard.get("policy_checks")
    if not isinstance(standard_policy, dict) or not standard_policy or any(
        value is not True for value in standard_policy.values()
    ):
        raise ValueError("candidate Standard policy checks are not PASS")
    if final:
        complete = paired.get("complete_full")
        if not isinstance(complete, dict) or (
            complete.get("pass") is not True
            or complete.get("hard_spec_pass") is not True
            or complete.get("status") != "actual_hard_spec_pass"
            or complete.get("checks") != checks
        ):
            raise ValueError("candidate complete-Full hard-spec gate drifted")
    else:
        complete = paired.get("complete_full")
        source_constraints = candidate.get("constraints")
        full_em_checks = (
            source_constraints.get("full_em_hit")
            if isinstance(source_constraints, dict)
            else None
        )
        standard_checks = (
            source_constraints.get("standard")
            if isinstance(source_constraints, dict)
            else None
        )
        complete_constraint = (
            source_constraints.get("complete_full")
            if isinstance(source_constraints, dict)
            else None
        )
        if not (
            isinstance(complete, dict)
            and complete.get("status") == "pending"
            and isinstance(full_em_checks, dict)
            and full_em_checks
            and all(value is True for value in full_em_checks.values())
            and isinstance(standard_checks, dict)
            and standard_checks == standard.get("checks")
            and isinstance(complete_constraint, dict)
            and complete_constraint.get("status") == "pending"
        ):
            raise ValueError("candidate split hard-spec gate drifted")


def _normalized_candidate(
    source: dict[str, Any],
    bom_candidate: dict[str, Any],
    measurements: dict[str, Any],
    ui_checks: dict[str, dict[str, Any]],
    *,
    final: bool,
    legacy: Any,
) -> dict[str, Any]:
    candidate = copy.deepcopy(source)
    report = candidate.get("report")
    report = copy.deepcopy(report) if isinstance(report, dict) else {}
    geometry = measurements["geometry"]
    losses = measurements["losses"]
    report.update({
        "size_W_mm": geometry["width_mm"],
        "size_L_mm": geometry["length_mm"],
        "size_H_mm": geometry["height_mm"],
        "size_WxLxH_mm": (
            f"{geometry['width_mm']:.3f} × {geometry['length_mm']:.3f} × "
            f"{geometry['height_mm']:.3f}"
        ),
        "volume_L": geometry["volume_L"],
        "footprint_cm2": geometry["footprint_cm2"],
        "pred_leakage_inductance_uH": measurements["Llt_uH"],
        "pred_total_loss_W": losses["modeled_total_W"],
        "rated_power_W": losses["rated_output_W"],
        "pred_efficiency_pct": losses["efficiency_pct"],
        "pred_core_loss_W": losses["core_W"],
        "pred_primary_winding_loss_W": losses["primary_winding_W"],
        "pred_secondary_center_winding_loss_W": losses[
            "secondary_center_winding_W"
        ],
        "pred_secondary_side_winding_loss_W": losses[
            "secondary_side_winding_W"
        ],
        "pred_secondary_winding_loss_W": (
            losses["secondary_center_winding_W"]
            + losses["secondary_side_winding_W"]
        ),
        "pred_component_winding_loss_sum_W": (
            losses["primary_winding_W"]
            + losses["secondary_center_winding_W"]
            + losses["secondary_side_winding_W"]
        ),
        "pred_total_winding_loss_W": losses["winding_total_W"],
        "pred_core_cold_plate_loss_W": losses["core_cold_plate_W"],
        "pred_winding_cold_plate_loss_W": losses[
            "winding_cold_plate_W"
        ],
        "surrogate_output_basis": (
            "actual_complete_full_fea_corroborated_by_actual_standard"
            if final
            else "actual_full_em_plus_actual_standard_split_validation"
        ),
    })
    validation_badge = (
        "complete-Full hard-spec PASS + Standard corroboration"
        if final
        else "Full-EM + Standard PASS / complete Full pending"
    )
    task_map = bom_candidate["validation_task_ids"]
    raw = measurements["raw_authoritative"]
    candidate.update({
        "report": report,
        "size_W_mm": geometry["width_mm"],
        "size_L_mm": geometry["length_mm"],
        "size_H_mm": geometry["height_mm"],
        "footprint_cm2": geometry["footprint_cm2"],
        "volume_L": geometry["volume_L"],
        "total_loss_W": losses["modeled_total_W"],
        "rated_power_W": losses["rated_output_W"],
        "pred_efficiency_pct": losses["efficiency_pct"],
        "pred_Llt_phys": measurements["Llt_uH"],
        "B_design_analytic_T": measurements["B_design_T"],
        "pred_B_mean_core": measurements["B_mean_T"],
        "diagnostic_pred_B_max_core": measurements["B_max_T"],
        "pred_max_temperature_C": measurements["maximum_temperature_C"],
        "C_tx_tx_F": measurements["C_tx_tx_F"],
        "C_rx_rx_F": measurements["C_rx_rx_F"],
        "C_tx_rx_F": measurements["C_tx_rx_F"],
        "f_res_tx_self_Hz": measurements["f_res_tx_self_Hz"],
        "f_res_rx_self_Hz": measurements["f_res_rx_self_Hz"],
        "f_res_interwinding_Hz": measurements["f_res_interwinding_Hz"],
        "pred_f_res_tx_screen_Hz": (
            measurements["f_res_tx_self_Hz"]
            / math.sqrt(
                legacy.EXPECTED_HARD_SPEC[
                    "magnetizing_inductance_factor"
                ]
            )
        ),
        "pred_f_res_rx_screen_Hz": (
            measurements["f_res_rx_self_Hz"]
            / math.sqrt(
                legacy.EXPECTED_HARD_SPEC[
                    "magnetizing_inductance_factor"
                ]
            )
        ),
        "pred_f_res_interwinding_screen_Hz": measurements[
            "f_res_interwinding_Hz"
        ],
        "pred_f_res_min_screen_Hz": measurements[
            "half_lm_resonance_Hz"
        ],
        "resonance_minimum_required_Hz": legacy.EXPECTED_HARD_SPEC[
            "resonance_min_Hz"
        ],
        "resonance_margin_Hz": (
            measurements["half_lm_resonance_Hz"]
            - legacy.EXPECTED_HARD_SPEC["resonance_min_Hz"]
        ),
        "constraints": ui_checks,
        "parameters": copy.deepcopy(measurements["parameters"]),
        "spec_status": (
            "pass" if final else "provisional_pass_complete_full_pending"
        ),
        "artifact_hydrated": True,
        "embedded_authenticated_result": True,
        "fea_verified": True,
        "gui_launch_eligible": True,
        "gui_build_eligible": True,
        "gui_solve_eligible": final,
        "gui_launch_limit_reason": (
            ""
            if final
            else (
                "Complete-Full loss/thermal validation is pending; geometry "
                "build is available but solve is fail-closed."
            )
        ),
        "local_gui_solver_contract": {
            **copy.deepcopy(legacy.EXPECTED_LOCAL_GUI_SOLVER),
            "required_result_echo_keys": list(
                legacy.REQUIRED_COOLING_ECHO_KEYS
            ),
        },
        "solver_revision": legacy.EXPECTED_TIM["solver_revision"],
        "validation_solver_revision": (
            FULL_SOLVER_REVISION if final else STANDARD_SOLVER_REVISION
        ),
        "thermal_pad_material_policy": legacy.EXPECTED_TIM[
            "material_policy"
        ],
        "thermal_pad_native_readback_contract_version": (
            legacy.EXPECTED_TIM["native_readback_contract_version"]
        ),
        "thermal_pad_native_readback_attested": _finite(
            raw.get("thermal_pad_native_readback_attested"),
            "thermal native readback attestation",
        ),
        "thermal_pad_native_thermal_conductivity_W_mK": _finite(
            raw.get("thermal_pad_native_thermal_conductivity_W_mK"),
            "thermal native conductivity",
        ),
        "thermal_pad_native_electrical_conductivity_S_m": _finite(
            raw.get("thermal_pad_native_electrical_conductivity_S_m"),
            "thermal native electrical conductivity",
        ),
        "validation_badge": validation_badge,
        "validation_state": (
            "complete_full_hard_spec_pass_standard_corroborated"
            if final
            else "provisional_split_validated_complete_full_pending"
        ),
        "hard_spec_authority": (
            "complete_full"
            if final
            else "full_em_geometry_llt_resonance_plus_standard_B_loss_thermal"
        ),
        "measurement_authority": {
            "geometry_Llt_resonance": (
                "complete_full" if final else "full_em_hit"
            ),
            "B_loss_temperature": (
                "complete_full" if final else "standard"
            ),
            "corroboration": "standard",
        },
        "validation_task_ids": dict(task_map),
        "validation_task_summary": (
            f"Full-EM {task_map['full_em_hit']} · Standard "
            f"{task_map['standard']} · complete-Full "
            f"{task_map['complete_full']}"
        ),
        "complete_full_pending": not final,
        "full_actual_hard_pass": final,
        "standard_corroboration_pass": True,
        "strict_full_equivalent": False,
        "final_design_approved": False,
        "engineering_review_required": True,
    })
    return candidate


def load_hit_publication(
    path: Path,
    selector: dict[str, Any],
    expected_sha256: str,
    legacy: Any,
) -> dict[str, Any]:
    """Validate a sealed HIT selector and return the monitor adapter shape."""

    try:
        _clean_tree(selector)
        if selector.get("schema_version") not in {
            SELECTION_SCHEMA_V1,
            SELECTION_SCHEMA_V2,
        }:
            raise ValueError("deadline HIT selector schema drifted")
        _verify_payload_digest(selector, "deadline HIT selector")
        _verify_authority(selector.get("authority"), "deadline HIT selector")
        _, bom = _load_bom(selector, legacy)
        publication = selector.get("publication")
        if not isinstance(publication, dict):
            raise ValueError("deadline HIT selector publication is missing")
        _verify_payload_digest(publication, "deadline HIT publication")
        schema = publication.get("schema_version")
        status = publication.get("status")
        if (schema, status) == (
            COMPLETE_PUBLICATION_SCHEMA,
            COMPLETE_STATUS,
        ):
            final = True
        elif status == SPLIT_STATUS and schema in {
            COMPLETE_PUBLICATION_SCHEMA,
            SPLIT_PUBLICATION_SCHEMA,
        }:
            final = False
        else:
            raise ValueError("deadline HIT publication state/schema drifted")
        if selector.get("status") != status:
            raise ValueError("selector/publication state drifted")
        _hard_spec(
            publication.get("active_hard_spec"),
            legacy,
            "deadline HIT publication",
        )
        _verify_authority(
            publication.get("authority"),
            "deadline HIT publication",
        )
        candidates = publication.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError("deadline HIT publication must contain one candidate")
        source_candidate = candidates[0]
        if not isinstance(source_candidate, dict):
            raise ValueError("deadline HIT publication candidate is invalid")
        selected = publication.get("selected")
        top_selected = selector.get("selected")
        if not isinstance(selected, dict) or not isinstance(top_selected, dict):
            raise ValueError("deadline HIT selector selection is missing")
        candidate_key = selected.get("candidate_key")
        if (
            not isinstance(candidate_key, str)
            or top_selected.get("candidate_key") != candidate_key
            or selected.get("physical_candidate_digest")
            != top_selected.get("physical_candidate_digest")
        ):
            raise ValueError("deadline HIT selected identity drifted")
        bom_candidate = _candidate_from_bom(bom, candidate_key)
        if (
            selected.get("physical_candidate_digest")
            != bom_candidate["physical_candidate_digest"]
        ):
            raise ValueError("deadline HIT selected BOM digest drifted")
        expected_tasks = bom_candidate["validation_task_ids"]
        for field, lane in (
            ("standard_task_id", "standard"),
            ("complete_full_task_id", "complete_full"),
            ("full_em_hit_task_id", "full_em_hit"),
        ):
            if field in selected and selected.get(field) != expected_tasks[lane]:
                raise ValueError(f"deadline HIT selected {field} drifted")
        measurements, _, _ = _verify_measurements(
            source_candidate,
            bom_candidate,
            final=final,
            legacy=legacy,
        )
        _verify_candidate_values(
            source_candidate,
            bom_candidate,
            measurements,
            final=final,
            legacy=legacy,
        )
        checks, ui_checks = _hard_checks(measurements, bom_candidate, legacy)
        if set(checks) != set(HARD_CHECK_NAMES) or not all(checks.values()):
            raise ValueError("deadline HIT actual hard-spec checks are not PASS")
        _validate_lane_measurement_echoes(
            source_candidate,
            checks,
            final=final,
        )
        candidate = _normalized_candidate(
            source_candidate,
            bom_candidate,
            measurements,
            ui_checks,
            final=final,
            legacy=legacy,
        )
        candidate_id = candidate.get("id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or len(candidate_id) > 200
            or any(ord(character) < 32 for character in candidate_id)
        ):
            raise ValueError("deadline HIT candidate ID is invalid")
    except legacy.DeadlineDesignError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise legacy.DeadlineDesignError(
            f"deadline HIT publication failed closed: {exc}"
        ) from exc

    generation_identity = (
        f"{candidate['candidate_identity_sha256']}\n{expected_sha256}"
    ).encode("utf-8")
    generation_id = (
        f"tier1-hit-{hashlib.sha256(generation_identity).hexdigest()[:20]}"
    )
    badge = candidate["validation_badge"]
    generation = {
        "id": generation_id,
        "label": (
            f"마감 설계 · {badge} · 1200×1000×750 mm · "
            "공진 ≥15 kHz · 100°C"
        ),
        "active": False,
        "read_only": True,
        "authority_eligible": False,
        "selectable": True,
        "state": candidate["validation_state"],
        "full_pending": not final,
        "source_healthy": True,
        "source_kind": "deadline_validated_design",
        "constraint_version": legacy.EXPECTED_CONSTRAINT_VERSION,
        "hard_spec": dict(legacy.EXPECTED_HARD_SPEC),
        "hard_spec_sha256": legacy._canonical_sha256(
            legacy.EXPECTED_HARD_SPEC
        ),
        "candidate_count": 1,
        "display_candidate_count": 1,
        "near_feasible_count": 0,
        "search_count": 0,
        "completed_count": 3 if final else 2,
        "metadata_verified": True,
        "detail_integrity_pending": False,
        "gui_launch_eligible": True,
        "gui_build_eligible": True,
        "gui_solve_eligible": final,
        "validation_badge": badge,
        "final_design_approved": False,
        "candidate_endpoint": f"/api/nsga2/generations/{generation_id}",
        "updated_at": selector.get("created_at"),
    }
    candidate = {
        **candidate,
        "generation_id": generation_id,
        "artifact_source": str(path),
        "publication_sha256": expected_sha256,
    }
    summary = {
        "candidate_count": 1,
        "display_candidate_count": 1,
        "valid_candidate_count": 1,
        "historical_diagnostic_count": 0,
        "min_volume_L": candidate["volume_L"],
        "min_loss_W": candidate["total_loss_W"],
        "min_volume_candidate_id": candidate["id"],
        "min_loss_candidate_id": candidate["id"],
    }
    warnings = (
        [
            "Complete-Full loss/thermal validation is pending; this split-"
            "validated candidate is not final-approved."
        ]
        if not final
        else [
            "Actual complete-Full hard-spec validation passed; engineering "
            "sign-off remains required and final_design_approved=false."
        ]
    )
    normalized_publication = {
        **copy.deepcopy(publication),
        "constraint_version": legacy.EXPECTED_CONSTRAINT_VERSION,
        "hard_spec": dict(legacy.EXPECTED_HARD_SPEC),
        "hard_spec_sha256": legacy._canonical_sha256(
            legacy.EXPECTED_HARD_SPEC
        ),
        "ui_publication_ready": True,
        "engineering_handoff_ready": final,
        "standard_actual_hard_pass": True,
        "full_actual_hard_pass": final,
        "full_pending": not final,
        "final_design_approved": False,
        "engineering_review_required": True,
    }
    payload = {
        "schema_version": 1,
        "available": True,
        "integrity_verified": True,
        "status": "completed" if final else "provisional",
        "source_kind": "deadline_validated_design",
        "selected_generation_id": generation_id,
        "generation": generation,
        "candidate_count": 1,
        "display_candidate_count": 1,
        "valid_candidate_count": 1,
        "candidates": [candidate],
        "near_feasible_preview": [],
        "summary": summary,
        "constraint_version": legacy.EXPECTED_CONSTRAINT_VERSION,
        "constraints": dict(legacy.EXPECTED_HARD_SPEC),
        "source": str(path),
        "updated_at": selector.get("created_at"),
        "note": badge,
        "warnings": warnings,
    }
    return {
        "path": path,
        "sha256": expected_sha256,
        "publication": normalized_publication,
        "selector": selector,
        "generation": generation,
        "candidate": candidate,
        "payload": payload,
    }
