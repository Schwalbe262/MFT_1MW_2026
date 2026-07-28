"""Hash-authenticated projection of one deadline FEA design package.

This reader has no Scheduler client and no write path.  Configuration is an
absolute publication path plus the exact file SHA-256.  Every referenced FEA
artifact is checked again on each coherent read, so replacing either the
publication or underlying evidence fails closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any


PUBLICATION_SCHEMA = "mft-deadline-validated-design-publication-v1"
RESULT_SCHEMA = "mft-deadline-design-fea-result-v1"
PATH_ENV = "MFT_DEADLINE_DESIGN_PUBLICATION"
SHA256_ENV = "MFT_DEADLINE_DESIGN_PUBLICATION_SHA256"
MAX_PUBLICATION_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 32 * 1024 * 1024
EXPECTED_HARD_SPEC = {
    "size_W_max_mm": 1200.0,
    "size_L_max_mm": 1000.0,
    "size_H_max_mm": 750.0,
    "T_limit_C": 100.0,
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "n_core_group_max": 4.0,
    "primary_conductor_thickness_mm": 5.0,
    "magnetizing_inductance_factor": 0.5,
    "resonance_min_Hz": 15_000.0,
}
EXPECTED_CONSTRAINT_VERSION = (
    "deadline-20260724-0900-kst-resonance-min15-v2"
)
EXPECTED_TIM = {
    "thermal_conductivity_W_mK": 3.0,
    "material_policy": (
        "deadline_tim_k3_native_attested_3WmK_electrically_insulating_v2"
    ),
    "electrically_insulating_required": True,
    "native_readback_contract_version": (
        "thermal-pad-native-material-readback-v1"
    ),
    "native_readback_required": True,
    "native_readback_attested": 1.0,
    "native_thermal_conductivity_W_mK": 3.0,
    "native_electrical_conductivity_S_m": 0.0,
    "solver_variant": "deadline-tim-k3",
    "solver_revision": "8a8d90f68e8728669282f586f24304c7cc807029",
    "solver_thermal_module_sha256": (
        "58b217e35ad1a6d4c1ffa6aafd88462872ee5c0da600c287c37b9a08d2bd3224"
    ),
}
ALLOWED_TIM_COOLING_VARIANTS = {
    "timk3": {
        "pad_thickness_mm": 1.0,
        "fan_velocity_m_s": 1.5,
    },
    "tim05k3": {
        "pad_thickness_mm": 0.5,
        "fan_velocity_m_s": 1.5,
    },
    "timk3fan3": {
        "pad_thickness_mm": 1.0,
        "fan_velocity_m_s": 3.0,
    },
    "tim05k3fan3": {
        "pad_thickness_mm": 0.5,
        "fan_velocity_m_s": 3.0,
    },
    "tim1k3fan6": {
        "pad_thickness_mm": 1.0,
        "fan_velocity_m_s": 6.0,
    },
    "tim1k3fan7": {
        "pad_thickness_mm": 1.0,
        "fan_velocity_m_s": 7.0,
    },
    "tim1k3fan8": {
        "pad_thickness_mm": 1.0,
        "fan_velocity_m_s": 8.0,
    },
    "tim1k3fan9": {
        "pad_thickness_mm": 1.0,
        "fan_velocity_m_s": 9.0,
    },
}
DEADLINE_LOCAL_GUI_RUNNER_SHA256 = (
    "4b533c4c945ac63bfd695c02629d7a25f92a7d51dc5737f3616ca200d76c86a0"
)
LOCAL_GUI_EXECUTION_PROFILE_VERSION = (
    "deadline-tim-k3-local-gui-exact-fea-profile-v1"
)
LOCAL_GUI_COMMON_SOLVER_PARAMETERS = {
    "freq": 1000.0,
    "V1_rms": 1000.0,
    "I1_rated": 1000.0,
    "I2_rated": 100.0,
    "I2_phase_deg": 0.0,
    "P_target": 1_000_000.0,
    "V2_rms": 10_000.0,
    "core_cm": 1.377,
    "core_x": 1.51,
    "core_y": 1.74,
    "core_k_thermal": 2.0,
    "k_ins": 0.2,
    "conductor_temp_C": 80.0,
    "core_plate_on": 1,
    "wcp_on": 1,
    "round_corner": 0,
    "loss_from_copy": 1,
    "loss_on": 1,
    "matrix_on": 1,
    "thermal_on": 1,
    "cap_on": 1,
    "thermal_max_iterations": 250,
    "rx_mesh_mode": "skin",
    "keep_project": 1,
}
LOCAL_GUI_MODE_SOLVER_PARAMETERS = {
    "symmetry": {
        "full_model": 0,
        "loss_sym_on": 1,
        "thermal_symmetry": "eighth",
        "n_explicit_turns": 0,
        "matrix_percent_error": 1.5,
        "matrix_max_passes": 20,
        "matrix_min_converged": 1,
        "matrix_skin_mesh": 0,
        "percent_error": 1.5,
        "max_passes": 10,
        "min_converged": 2,
    },
    "full": {
        "full_model": 1,
        "loss_sym_on": 0,
        "thermal_symmetry": "full",
        "n_explicit_turns": 2,
        "matrix_percent_error": 0.5,
        "matrix_max_passes": 24,
        "matrix_min_converged": 2,
        "matrix_skin_mesh": 1,
        "percent_error": 0.5,
        "max_passes": 18,
        "min_converged": 2,
    },
}
LOCAL_GUI_PROFILE_ECHO_KEYS = tuple(dict.fromkeys((
    *LOCAL_GUI_COMMON_SOLVER_PARAMETERS,
    *LOCAL_GUI_MODE_SOLVER_PARAMETERS["symmetry"],
    *LOCAL_GUI_MODE_SOLVER_PARAMETERS["full"],
)))
EXPECTED_LOCAL_GUI_SOLVER = {
    "solver_variant": "deadline-tim-k3",
    "solver_revision": EXPECTED_TIM["solver_revision"],
    "solver_branch": "deadline-timk3-k3-20260723",
    "solver_source_runner_sha256": (
        "c3e2b8a2dce2dbf6f87e93723ffabd864702c579db6d2fe9b2f78494b227a14c"
    ),
    "solver_thermal_module_sha256": EXPECTED_TIM[
        "solver_thermal_module_sha256"
    ],
    "gui_runner_sha256": DEADLINE_LOCAL_GUI_RUNNER_SHA256,
    "library_revision": "e6b9b9d20a832ff5c3f7ca97218737a0b8650781",
    "backend": "standalone",
    "keep_project": 1,
    "thermal_on": 1,
    "half_magnetizing_resonance_minimum_Hz": 15_000.0,
    "thermal_pad_native_readback_contract_version": EXPECTED_TIM[
        "native_readback_contract_version"
    ],
    "thermal_pad_native_readback_required": True,
    "thermal_pad_native_thermal_conductivity_W_mK": 3.0,
    "thermal_pad_native_electrical_conductivity_S_m": 0.0,
    "execution_profile_contract_version": (
        LOCAL_GUI_EXECUTION_PROFILE_VERSION
    ),
    "common_solver_parameters": LOCAL_GUI_COMMON_SOLVER_PARAMETERS,
    "mode_solver_parameters": LOCAL_GUI_MODE_SOLVER_PARAMETERS,
    "result_persistence_required": True,
    "retained_gui_required": True,
}
REQUIRED_COOLING_ECHO_KEYS = (
    "fan_config",
    "fan_velocity",
    "plate_temp",
    "air_temp",
    "core_plate_pad_t",
    "wcp_pad_t",
    "thermal_pad_conductivity_W_mK",
    "thermal_pad_material_policy",
    "thermal_pad_native_readback_contract_version",
    "thermal_pad_native_readback_attested",
    "thermal_pad_native_thermal_conductivity_W_mK",
    "thermal_pad_native_electrical_conductivity_S_m",
    *LOCAL_GUI_PROFILE_ECHO_KEYS,
)
REQUIRED_CANDIDATE_NUMBERS = (
    "size_W_mm",
    "size_L_mm",
    "size_H_mm",
    "footprint_cm2",
    "volume_L",
    "total_loss_W",
    "rated_power_W",
    "pred_efficiency_pct",
    "pred_core_loss_W",
    "pred_total_winding_loss_W",
    "pred_primary_winding_loss_W",
    "pred_secondary_center_winding_loss_W",
    "pred_secondary_side_winding_loss_W",
    "pred_core_cold_plate_loss_W",
    "pred_winding_cold_plate_loss_W",
    "pred_Llt_phys",
    "B_design_analytic_T",
    "pred_B_mean_core",
    "diagnostic_pred_B_max_core",
    "pred_max_temperature_C",
    "min_insulation_mm",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_F",
    "f_res_tx_self_Hz",
    "f_res_rx_self_Hz",
    "f_res_interwinding_Hz",
    "pred_f_res_tx_screen_Hz",
    "pred_f_res_rx_screen_Hz",
    "pred_f_res_interwinding_screen_Hz",
    "pred_f_res_min_screen_Hz",
    "resonance_minimum_required_Hz",
    "resonance_margin_Hz",
    "core_thermal_pad_thickness_mm",
    "winding_thermal_pad_thickness_mm",
    "thermal_pad_conductivity_W_mK",
    "thermal_pad_native_readback_attested",
    "thermal_pad_native_thermal_conductivity_W_mK",
    "thermal_pad_native_electrical_conductivity_S_m",
    "fan_velocity_m_s",
    "plate_temp_C",
    "air_temp_C",
)
REQUIRED_PARAMETERS = (
    "N1_main", "N1_side", "N2_main", "N2_side",
    "l1", "l2", "h1", "w1",
    "n_core_group", "core_plate_t", "core_plate_pad_t",
    "cw1", "gap1", "cw2", "gap2", "nwh1", "nwh2",
    "cc_w2c_space_x", "cc_w2c_space_y",
    "w2c_w1c_space_x", "w2c_w1c_space_y",
    "w1c_w2s_space_x", "w2s_w1s_space_x", "w1s_w2s_space_y",
    "w1s_cs_space_x", "cs_w1s_space_y",
    "wcp_t", "wcp_pad_t", "wcp_len_x",
    "fan_velocity", "plate_temp", "air_temp",
)
REQUIRED_STRING_PARAMETERS = ("fan_config",)


class DeadlineDesignError(RuntimeError):
    """The configured package or one of its evidence files failed closed."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str, max_bytes: int) -> Path:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    reparse = getattr(info, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
    )
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or reparse
        or info.st_size <= 0
        or info.st_size > max_bytes
    ):
        raise DeadlineDesignError(
            f"{label} is not a bounded regular file: {resolved}"
        )
    return resolved


def _read_json(path: Path, label: str, max_bytes: int) -> tuple[Path, dict]:
    resolved = _regular_file(path, label, max_bytes)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeadlineDesignError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise DeadlineDesignError(f"{label} root is not an object")
    return resolved, value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeadlineDesignError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DeadlineDesignError(f"{label} is not finite")
    return number


def _same_number(actual: Any, expected: float) -> bool:
    try:
        return math.isclose(
            _finite(actual, "numeric value"),
            expected,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    except DeadlineDesignError:
        return False


def _verify_reference(reference: Any, label: str, max_bytes: int) -> tuple[Path, dict]:
    if not isinstance(reference, dict):
        raise DeadlineDesignError(f"{label} reference is missing")
    raw_path = reference.get("path")
    sha256 = str(reference.get("sha256") or "").lower()
    size = reference.get("size_bytes")
    if not isinstance(raw_path, str) or not raw_path:
        raise DeadlineDesignError(f"{label} path is missing")
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise DeadlineDesignError(f"{label} SHA-256 is invalid")
    path, value = _read_json(Path(raw_path), label, max_bytes)
    if _file_sha256(path) != sha256:
        raise DeadlineDesignError(f"{label} SHA-256 mismatch")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size != path.stat().st_size
    ):
        raise DeadlineDesignError(f"{label} size mismatch")
    return path, value


def _verify_evidence(
    publication: dict[str, Any],
    fidelity: str,
    required: bool,
) -> None:
    evidence = publication.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    item = evidence.get(fidelity)
    if item is None and not required:
        return
    if not isinstance(item, dict):
        raise DeadlineDesignError(f"{fidelity} evidence is missing")
    _, result = _verify_reference(
        item.get("result"), f"{fidelity} result", MAX_RESULT_BYTES
    )
    _, plan = _verify_reference(
        item.get("plan"), f"{fidelity} plan", MAX_PUBLICATION_BYTES
    )
    _verify_reference(
        item.get("submission_receipt"),
        f"{fidelity} submission",
        MAX_PUBLICATION_BYTES,
    )
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("fidelity") != fidelity
        or result.get("actual_hard_spec_pass") is not True
        or result.get("candidate_identity_matches") is not True
        or result.get("result_contract_valid") is not True
        or result.get("result_state") != "valid"
        or result.get("task_status") != "completed"
        or result.get("solver_variant_contract_valid") is not True
        or result.get("final_design_approved") is not False
        or result.get("canonical_dataset_mutated") is not False
        or result.get("scheduler_configuration_mutated") is not False
        or result.get("task_cancellation_performed") is not False
        or result.get("candidate_identity_sha256")
        != publication.get("candidate_identity_sha256")
        or result.get("candidate_digest") != publication.get("candidate_digest")
    ):
        raise DeadlineDesignError(f"{fidelity} actual PASS identity drifted")
    plan_candidate = plan.get("candidate")
    solver = plan.get("solver_contract")
    decoded = (
        plan_candidate.get("decoded_params")
        if isinstance(plan_candidate, dict) else None
    )
    native = result.get("result")
    candidate = publication.get("candidate")
    target = plan.get("target")
    gate = result.get("actual_hard_spec_gate")
    parameters = (
        candidate.get("parameters") if isinstance(candidate, dict) else None
    )
    if not all(
        isinstance(value, dict)
        for value in (
            solver, decoded, native, candidate, parameters, target, gate
        )
    ):
        raise DeadlineDesignError(
            f"{fidelity} cooling/result echo evidence is incomplete"
        )
    if (
        set(target) != set(EXPECTED_HARD_SPEC)
        or any(
            not _same_number(target.get(key), expected)
            for key, expected in EXPECTED_HARD_SPEC.items()
        )
        or gate.get("pass") is not True
        or _finite(
            gate.get("half_magnetizing_resonance_Hz"),
            f"{fidelity} actual half-Lm resonance",
        ) + 1e-9
        < EXPECTED_HARD_SPEC["resonance_min_Hz"]
        or (
            fidelity
            == (
                "full"
                if publication.get("full_actual_hard_pass") is True
                else "standard"
            )
            and not _same_number(
                candidate.get("pred_f_res_min_screen_Hz"),
                _finite(
                    gate.get("half_magnetizing_resonance_Hz"),
                    f"{fidelity} actual half-Lm resonance",
                ),
            )
        )
    ):
        raise DeadlineDesignError(
            f"{fidelity} evidence is not the >=15 kHz hard-spec contract"
        )
    cooling = publication.get("cooling_boundary_contract")
    if (
        not isinstance(cooling, dict)
        or plan_candidate.get("candidate_id") != cooling.get("variant")
    ):
        raise DeadlineDesignError(
            f"{fidelity} cooling variant identity drifted"
        )
    if (
        solver.get("solver_variant") != EXPECTED_TIM["solver_variant"]
        or solver.get("solver_revision") != EXPECTED_TIM["solver_revision"]
        or solver.get("library_revision")
        != EXPECTED_LOCAL_GUI_SOLVER["library_revision"]
        or solver.get("thermal_pad_native_readback_contract_version")
        != EXPECTED_TIM["native_readback_contract_version"]
        or solver.get("thermal_pad_native_readback_required") is not True
        or native.get("git_hash") != EXPECTED_TIM["solver_revision"]
        or native.get("pyaedt_library_git_hash")
        != EXPECTED_LOCAL_GUI_SOLVER["library_revision"]
        or not _same_number(
            native.get("thermal_pad_conductivity_W_mK"),
            EXPECTED_TIM["thermal_conductivity_W_mK"],
        )
        or native.get("thermal_pad_material_policy")
        != EXPECTED_TIM["material_policy"]
        or native.get("thermal_pad_native_readback_contract_version")
        != EXPECTED_TIM["native_readback_contract_version"]
        or not _same_number(
            native.get("thermal_pad_native_readback_attested"), 1.0
        )
        or not _same_number(
            native.get("thermal_pad_native_thermal_conductivity_W_mK"),
            EXPECTED_TIM["native_thermal_conductivity_W_mK"],
        )
        or not _same_number(
            native.get("thermal_pad_native_electrical_conductivity_S_m"),
            EXPECTED_TIM["native_electrical_conductivity_S_m"],
        )
    ):
        raise DeadlineDesignError(
            f"{fidelity} solver/TIM result echo drifted"
        )
    for key in (
        "fan_velocity", "plate_temp", "air_temp",
        "core_plate_pad_t", "wcp_pad_t",
    ):
        expected = decoded.get(key)
        if (
            not _same_number(native.get(key), _finite(expected, key))
            or not _same_number(parameters.get(key), _finite(expected, key))
        ):
            raise DeadlineDesignError(
                f"{fidelity} cooling result echo drifted for {key}"
            )
    fan_config = decoded.get("fan_config")
    if (
        not isinstance(fan_config, str)
        or not fan_config
        or native.get("fan_config") != fan_config
        or parameters.get("fan_config") != fan_config
    ):
        raise DeadlineDesignError(
            f"{fidelity} cooling result echo drifted for fan_config"
        )
    expected_full = 1 if fidelity == "full" else 0
    if not _same_number(native.get("full_model"), float(expected_full)):
        raise DeadlineDesignError(f"{fidelity} model fidelity echo drifted")


def load_publication(
    raw_path: str | Path,
    expected_sha256: str,
) -> dict[str, Any]:
    expected_sha256 = str(expected_sha256 or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise DeadlineDesignError("configured publication SHA-256 is invalid")
    path, publication = _read_json(
        Path(raw_path), "deadline design publication", MAX_PUBLICATION_BYTES
    )
    if _file_sha256(path) != expected_sha256:
        raise DeadlineDesignError("deadline design publication SHA-256 mismatch")
    if publication.get("schema_version") in {
        "mft-deadline-8010-first-complete-pass-selection-v1",
        "mft-deadline-8010-hit-validation-selection-v2",
    }:
        from .deadline_hit_design import load_hit_publication

        return load_hit_publication(
            path,
            publication,
            expected_sha256,
            legacy=__import__(__name__, fromlist=[""]),
        )
    if publication.get("schema_version") != PUBLICATION_SCHEMA:
        raise DeadlineDesignError("deadline design publication schema mismatch")
    if (
        publication.get("ui_publication_ready") is not True
        or publication.get("standard_actual_hard_pass") is not True
        or publication.get("final_design_approved") is not False
        or publication.get("engineering_review_required") is not True
        or publication.get("live_8010_mutated") is not False
        or publication.get("scheduler_8002_mutated") is not False
        or publication.get("task_mutation_performed") is not False
        or publication.get("canonical_dataset_mutated") is not False
        or publication.get("constraint_version")
        != EXPECTED_CONSTRAINT_VERSION
    ):
        raise DeadlineDesignError("deadline publication authority boundary drifted")
    hard_spec = publication.get("hard_spec")
    if (
        not isinstance(hard_spec, dict)
        or set(hard_spec) != set(EXPECTED_HARD_SPEC)
        or any(
            not _same_number(hard_spec.get(key), expected)
            for key, expected in EXPECTED_HARD_SPEC.items()
        )
        or publication.get("hard_spec_sha256")
        != _canonical_sha256(hard_spec)
    ):
        raise DeadlineDesignError("deadline hard-spec identity drifted")
    tim = publication.get("tim_contract")
    if not isinstance(tim, dict) or any(
        (
            not _same_number(tim.get(key), expected)
            if isinstance(expected, float)
            else tim.get(key) != expected
        )
        for key, expected in EXPECTED_TIM.items()
    ):
        raise DeadlineDesignError("deadline TIM contract drifted")
    cooling = publication.get("cooling_boundary_contract")
    if not isinstance(cooling, dict):
        raise DeadlineDesignError("deadline cooling boundary contract is missing")
    cooling_variant = ALLOWED_TIM_COOLING_VARIANTS.get(
        str(cooling.get("variant") or "")
    )
    if cooling_variant is None:
        raise DeadlineDesignError(
            "deadline cooling boundary variant is not allowlisted"
        )
    selected_pad_thickness = cooling_variant["pad_thickness_mm"]
    selected_fan_velocity = cooling_variant["fan_velocity_m_s"]
    if (
        cooling.get("fan_config") != "dual"
        or not _same_number(
            cooling.get("fan_velocity_m_s"), selected_fan_velocity
        )
        or not _same_number(cooling.get("plate_temp_C"), 50.0)
        or not _same_number(cooling.get("air_temp_C"), 50.0)
        or not _same_number(
            cooling.get("core_pad_thickness_mm"),
            selected_pad_thickness,
        )
        or not _same_number(
            cooling.get("winding_pad_thickness_mm"),
            selected_pad_thickness,
        )
        or not _same_number(
            cooling.get("core_pad_conductivity_W_mK"),
            EXPECTED_TIM["thermal_conductivity_W_mK"],
        )
        or not _same_number(
            cooling.get("winding_pad_conductivity_W_mK"),
            EXPECTED_TIM["thermal_conductivity_W_mK"],
        )
        or cooling.get("core_pad_material_policy")
        != EXPECTED_TIM["material_policy"]
        or cooling.get("winding_pad_material_policy")
        != EXPECTED_TIM["material_policy"]
        or cooling.get("native_readback_contract_version")
        != EXPECTED_TIM["native_readback_contract_version"]
        or cooling.get("native_readback_required") is not True
        or not _same_number(
            cooling.get("native_readback_attested"), 1.0
        )
        or not _same_number(
            cooling.get("native_thermal_conductivity_W_mK"),
            EXPECTED_TIM["native_thermal_conductivity_W_mK"],
        )
        or not _same_number(
            cooling.get("native_electrical_conductivity_S_m"),
            EXPECTED_TIM["native_electrical_conductivity_S_m"],
        )
    ):
        raise DeadlineDesignError("deadline cooling boundary contract drifted")
    if (
        not _same_number(
            tim.get("core_pad_thickness_mm"), selected_pad_thickness
        )
        or not _same_number(
            tim.get("winding_pad_thickness_mm"), selected_pad_thickness
        )
    ):
        raise DeadlineDesignError(
            "deadline TIM thickness/cooling variant drifted"
        )

    _verify_evidence(publication, "standard", required=True)
    full_required = publication.get("full_actual_hard_pass") is True
    if (
        publication.get("engineering_handoff_ready") is not full_required
        or publication.get("full_pending") is not (not full_required)
    ):
        raise DeadlineDesignError("full-FEA handoff readiness drifted")
    _verify_evidence(publication, "full", required=full_required)
    authority = publication.get("authority")
    if (
        not isinstance(authority, dict)
        or authority.get("read_only_ui_publication") is not True
        or authority.get("local_gui_geometry_build") is not True
        or authority.get("local_gui_solve") is not full_required
        or authority.get("automatic_production_promotion") is not False
        or authority.get("automatic_scheduler_submission") is not False
        or authority.get("engineering_signoff") is not False
    ):
        raise DeadlineDesignError(
            "deadline publication local-GUI authority drifted"
        )

    candidate = publication.get("candidate")
    if not isinstance(candidate, dict):
        raise DeadlineDesignError("deadline UI candidate is missing")
    if publication.get("candidate_sha256") != _canonical_sha256(candidate):
        raise DeadlineDesignError("deadline UI candidate digest mismatch")
    full_pass = publication.get("full_actual_hard_pass") is True
    local_contract = candidate.get("local_gui_solver_contract")
    if (
        not isinstance(local_contract, dict)
        or any(
            local_contract.get(key) != value
            for key, value in EXPECTED_LOCAL_GUI_SOLVER.items()
        )
        or {
            str(value)
            for value in local_contract.get("required_result_echo_keys", [])
        }
        != set(REQUIRED_COOLING_ECHO_KEYS)
    ):
        raise DeadlineDesignError(
            "deadline local GUI solver contract drifted"
        )
    if (
        candidate.get("spec_status") != "pass"
        or candidate.get("artifact_hydrated") is not True
        or candidate.get("embedded_authenticated_result") is not True
        or candidate.get("fea_verified") is not True
        or candidate.get("gui_launch_eligible") is not True
        or candidate.get("gui_build_eligible") is not True
        or candidate.get("gui_solve_eligible") is not full_pass
        or candidate.get("thermal_pad_material_policy")
        != EXPECTED_TIM["material_policy"]
        or candidate.get("thermal_pad_native_readback_contract_version")
        != EXPECTED_TIM["native_readback_contract_version"]
    ):
        raise DeadlineDesignError("deadline UI candidate authority drifted")
    for key in REQUIRED_CANDIDATE_NUMBERS:
        _finite(candidate.get(key), f"candidate {key}")
    if (
        candidate.get("fan_config") != cooling["fan_config"]
        or not _same_number(
            candidate.get("fan_velocity_m_s"),
            cooling["fan_velocity_m_s"],
        )
        or not _same_number(
            candidate.get("plate_temp_C"), cooling["plate_temp_C"]
        )
        or not _same_number(
            candidate.get("air_temp_C"), cooling["air_temp_C"]
        )
    ):
        raise DeadlineDesignError(
            "deadline candidate cooling display fields drifted"
        )
    parameters = candidate.get("parameters")
    if not isinstance(parameters, dict):
        raise DeadlineDesignError("deadline GUI parameters are missing")
    for key in REQUIRED_PARAMETERS:
        _finite(parameters.get(key), f"candidate parameter {key}")
    for key in REQUIRED_STRING_PARAMETERS:
        if not isinstance(parameters.get(key), str) or not parameters[key]:
            raise DeadlineDesignError(
                f"candidate parameter {key} is not a non-empty string"
            )
    if (
        parameters["fan_config"] != cooling["fan_config"]
        or not _same_number(
            parameters["fan_velocity"], cooling["fan_velocity_m_s"]
        )
        or not _same_number(
            parameters["plate_temp"], cooling["plate_temp_C"]
        )
        or not _same_number(
            parameters["air_temp"], cooling["air_temp_C"]
        )
    ):
        raise DeadlineDesignError(
            "deadline candidate cooling boundary projection drifted"
        )
    constraints = candidate.get("constraints")
    if (
        not isinstance(constraints, dict)
        or not constraints
        or any(
            not isinstance(item, dict) or item.get("pass") is not True
            for item in constraints.values()
        )
    ):
        raise DeadlineDesignError("deadline candidate constraints are not PASS")
    resonance_value = _finite(
        candidate.get("pred_f_res_min_screen_Hz"),
        "candidate half-Lm resonance",
    )
    resonance_minimum = EXPECTED_HARD_SPEC["resonance_min_Hz"]
    resonance_constraint = constraints.get("resonance")
    if (
        resonance_value + 1e-9 < resonance_minimum
        or not _same_number(
            candidate.get("resonance_minimum_required_Hz"),
            resonance_minimum,
        )
        or not _same_number(
            candidate.get("resonance_margin_Hz"),
            resonance_value - resonance_minimum,
        )
        or not isinstance(resonance_constraint, dict)
        or resonance_constraint.get("direction") != "minimum"
        or resonance_constraint.get("operator") != ">="
        or not _same_number(
            resonance_constraint.get("minimum_Hz"), resonance_minimum
        )
        or not _same_number(
            resonance_constraint.get("limit"), resonance_minimum
        )
        or not _same_number(
            resonance_constraint.get("value"), resonance_value
        )
    ):
        raise DeadlineDesignError(
            "deadline candidate resonance is not the >=15 kHz contract"
        )

    generation_identity = (
        f"{publication['candidate_identity_sha256']}\n{expected_sha256}"
    ).encode("utf-8")
    generation_id = (
        f"tier1-{hashlib.sha256(generation_identity).hexdigest()[:20]}"
    )
    generation = {
        "id": generation_id,
        "label": (
            "마감 설계 · actual "
            f"{'Standard+Full' if full_pass else 'Standard'} PASS · "
            "1200×1000×750 mm · 공진 ≥ 15 kHz · 100°C"
        ),
        "active": False,
        "read_only": True,
        "authority_eligible": False,
        "selectable": True,
        "state": (
            "actual_standard_and_full_pass"
            if full_pass else "actual_standard_pass_full_pending"
        ),
        "full_pending": not full_pass,
        "source_healthy": True,
        "source_kind": "deadline_validated_design",
        "constraint_version": publication.get("constraint_version"),
        "hard_spec": dict(hard_spec),
        "hard_spec_sha256": publication.get("hard_spec_sha256"),
        "candidate_count": 1,
        "display_candidate_count": 1,
        "near_feasible_count": 0,
        "search_count": 0,
        "completed_count": 1 + int(full_pass),
        "metadata_verified": True,
        "detail_integrity_pending": False,
        "gui_launch_eligible": True,
        "gui_build_eligible": True,
        "gui_solve_eligible": full_pass,
        "candidate_endpoint": f"/api/nsga2/generations/{generation_id}",
        "updated_at": publication.get("created_at"),
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
    payload = {
        "schema_version": 1,
        "available": True,
        "integrity_verified": True,
        "status": "completed",
        "source_kind": "deadline_validated_design",
        "selected_generation_id": generation_id,
        "generation": generation,
        "candidate_count": 1,
        "display_candidate_count": 1,
        "valid_candidate_count": 1,
        "candidates": [candidate],
        "near_feasible_preview": [],
        "summary": summary,
        "constraint_version": publication.get("constraint_version"),
        "constraints": dict(hard_spec),
        "source": str(path),
        "updated_at": publication.get("created_at"),
        "note": (
            "해시 인증된 actual FEA 마감 설계입니다. 제조 승인 전 "
            "엔지니어 검토가 필요합니다."
        ),
        "warnings": (
            []
            if full_pass
            else ["Full/fine actual FEA validation is still pending."]
        ),
    }
    return {
        "path": path,
        "sha256": expected_sha256,
        "publication": publication,
        "generation": generation,
        "candidate": candidate,
        "payload": payload,
    }


def load_configured_publication() -> dict[str, Any] | None:
    raw_path = os.environ.get(PATH_ENV, "").strip()
    raw_sha = os.environ.get(SHA256_ENV, "").strip()
    if not raw_path and not raw_sha:
        return None
    if not raw_path or not raw_sha:
        raise DeadlineDesignError(
            f"{PATH_ENV} and {SHA256_ENV} must be configured together"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        raise DeadlineDesignError("deadline publication path must be absolute")
    return load_publication(path, raw_sha)
