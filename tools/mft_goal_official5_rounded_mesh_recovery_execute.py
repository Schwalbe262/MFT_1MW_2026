"""Execute the one-shot B6 rounded WCP mesh-only recovery.

This launcher is intentionally narrower than the normal runner.  It accepts
only the authenticated candidate-5 rounded Standard parameter file, installs
one process-local mesh policy override, and then delegates to the unchanged
``run_simulation_260706.py`` entry point.

The only numerical change is the nonzero tangential WCP MeshRegion padding:
the B5 eighth-symmetry vector ``[0, 2, 0, 0, 2, 0]`` becomes
``[0, 1, 0, 0, 1, 0]`` millimetres.  Physical TIMs, geometry, winding turns,
level-5 controls, contact/symmetry clipping, setup, and cooling boundaries are
not modified.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

OPT_IN_ENV = "MFT_ROUNDED_WCP_TANGENTIAL_PADDING_RECOVERY"
OPT_IN_TOKEN = "official5-rounded-b6-eighth-wcp-tangential-1mm-v1"
PADDING_ENV = "MFT_ROUNDED_WCP_TANGENTIAL_PADDING_MM"
EXPECTED_PARAMS_SHA256 = (
    "b7ed829d4d71ed01f59d2d98bea389628bc40385d053acaef11363e96f7c094d"
)
SOURCE_POLICY = (
    "b5-rxmain-l5-wcp-pad-symmetry-contact-clipped-regions-v1"
)
TARGET_POLICY = (
    "b6-rxmain-l5-wcp-pad-symmetry-contact-clipped-"
    "tangential-1mm-regions-v1"
)
SOURCE_REGION_CONTRACT = (
    "wcp-pad-per-object-symmetry-contact-clipped-region-v3"
)
TARGET_REGION_CONTRACT = (
    "wcp-pad-per-object-symmetry-contact-clipped-region-b6-v1"
)
TARGET_TANGENTIAL_PADDING_MM = 1.0
EXPECTED_EIGHTH_PADDING_MM = (0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
EXPECTED_WCP_REGION_COUNT = 4
MARKER_SCHEMA = "mft-goal-rounded-wcp-mesh-recovery-executor-b6-v1"
thermal: ModuleType | Any | None = None


class RecoveryExecutionError(RuntimeError):
    """The one-shot B6 execution contract was not exact."""


def _thermal_module() -> ModuleType | Any:
    global thermal
    if thermal is None:
        thermal = importlib.import_module("module.thermal_260706")
    return thermal


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _params_path(argv: list[str]) -> Path:
    if "--full" in argv or "--symmetry-thermal-direct-analyze" in argv:
        raise RecoveryExecutionError(
            "B6 is eighth Standard with explicit native mesh preflight"
        )
    if argv.count("--params") != 1:
        raise RecoveryExecutionError("B6 requires exactly one --params")
    index = argv.index("--params")
    if index + 1 >= len(argv):
        raise RecoveryExecutionError("B6 --params has no value")
    return Path(argv[index + 1]).resolve(strict=True)


def authenticate_invocation(
    argv: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Authenticate opt-in, exact rounded params, and Standard-only CLI."""

    values = os.environ if environ is None else environ
    args = list(sys.argv[1:] if argv is None else argv)
    if values.get(OPT_IN_ENV) != OPT_IN_TOKEN:
        raise RecoveryExecutionError("B6 explicit opt-in token is absent")
    try:
        padding = float(values.get(PADDING_ENV, ""))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RecoveryExecutionError("B6 padding opt-in is invalid") from exc
    if not math.isclose(
        padding,
        TARGET_TANGENTIAL_PADDING_MM,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RecoveryExecutionError("B6 tangential padding is not 1.0mm")
    path = _params_path(args)
    try:
        params = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryExecutionError("B6 parameter file is unreadable") from exc
    if not isinstance(params, dict) or _canonical_sha256(params) != (
        EXPECTED_PARAMS_SHA256
    ):
        raise RecoveryExecutionError(
            "B6 parameter bytes are not exact rounded candidate 5"
        )
    exact = {
        "full_model": 0,
        "thermal_symmetry": "eighth",
        "round_corner": 1,
        "corner_radius": 10.0,
        "corner_segments": 4,
        "fan_velocity": 1.5,
        "fan_config": "dual",
        "k_ins": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "N1_main": 6,
        "N2_main": 37,
        "N2_side": 23,
    }
    drift = {
        key: {"expected": value, "actual": params.get(key)}
        for key, value in exact.items()
        if params.get(key) != value
    }
    if drift:
        raise RecoveryExecutionError(f"B6 fixed candidate drifted: {drift}")
    return {
        "schema_version": MARKER_SCHEMA,
        "candidate_physics_sha256": (
            "909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42"
        ),
        "params_sha256": EXPECTED_PARAMS_SHA256,
        "full_model": 0,
        "thermal_symmetry": "eighth",
        "round_corner": 1,
        "corner_radius_mm": 10.0,
        "corner_segments": 4,
        "fan_velocity_m_s": 1.5,
        "thermal_pad_conductivity_W_mK": 0.2,
        "core_plate_pad_t_mm": 2.0,
        "wcp_pad_t_mm": 2.0,
        "source_policy": SOURCE_POLICY,
        "target_policy": TARGET_POLICY,
        "target_eighth_padding_mm": list(EXPECTED_EIGHTH_PADDING_MM),
        "expected_wcp_mesh_region_count": EXPECTED_WCP_REGION_COUNT,
        "explicit_native_mesh_preflight": True,
        "direct_analyze": False,
    }


def _validate_mesh_plan(plan: Mapping[str, Any]) -> None:
    wcp = [
        item
        for item in plan.get("operations", [])
        if item.get("category") == "wcp_pad_region"
    ]
    expected = list(EXPECTED_EIGHTH_PADDING_MM)
    if (
        plan.get("policy") != TARGET_POLICY
        or plan.get("symmetry_mode") != "eighth"
        or plan.get("required_objects_missing") != []
        or plan.get("wcp_pad_mesh_region_count")
        != EXPECTED_WCP_REGION_COUNT
        or len(wcp) != EXPECTED_WCP_REGION_COUNT
        or any(item.get("level") != 5 for item in wcp)
        or any(len(item.get("objects", [])) != 1 for item in wcp)
        or any(item.get("padding_values_mm") != expected for item in wcp)
        or any(
            list(item.get("padding_by_direction_mm", {}).values())
            != expected
            for item in wcp
        )
    ):
        raise RecoveryExecutionError(
            "B6 WCP native padding/readback mesh plan failed"
        )


def _validate_native_preflight(evidence: Mapping[str, Any]) -> None:
    mapping = evidence.get("mesh_mapping_coverage", {})
    native = evidence.get("native_operation_readback", {})
    if (
        evidence.get("passed") is not True
        or evidence.get("generate_mesh_returned") is not True
        or evidence.get("mesh_mapping_coverage_passed") is not True
        or evidence.get("required_objects_missing") != []
        or evidence.get("unmeshed_objects") != []
        or evidence.get("wcp_pad_mesh_region_count")
        != EXPECTED_WCP_REGION_COUNT
        or mapping.get("passed") is not True
        or mapping.get("expected_local_region_count")
        != EXPECTED_WCP_REGION_COUNT
        or mapping.get("missing_local_regions") != []
        or mapping.get("local_regions_without_mesh") != []
        or mapping.get("uncoupled_local_regions") != []
        or mapping.get("local_region_objects_missing") != []
        or mapping.get("required_objects_missing") != []
        or native.get("required_thin_objects_missing") != []
        or native.get("mesh_region_operation_count")
        != EXPECTED_WCP_REGION_COUNT
        or native.get("mesh_region_part_readback_passed") is not True
    ):
        raise RecoveryExecutionError(
            "B6 native WCP domain/overlap or thin-solid coverage failed"
        )


def install_recovery_policy() -> None:
    """Install one process-local override and strict evidence wrappers."""

    module = _thermal_module()
    if (
        module.THERMAL_MESH_POLICY != SOURCE_POLICY
        or module.WCP_PAD_MESH_REGION_CONTRACT_VERSION
        != SOURCE_REGION_CONTRACT
        or not math.isclose(
            module.WCP_PAD_MESH_REGION_PADDING_MM,
            2.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RecoveryExecutionError("B5 source mesh policy drifted")
    original_assign = module._assign_thermal_mesh
    original_preflight = module._generate_and_attest_thermal_mesh
    module.WCP_PAD_MESH_REGION_PADDING_MM = TARGET_TANGENTIAL_PADDING_MM
    module.THERMAL_MESH_POLICY = TARGET_POLICY
    module.WCP_PAD_MESH_REGION_CONTRACT_VERSION = TARGET_REGION_CONTRACT
    padding = tuple(
        module._wcp_pad_mesh_region_padding_mm("eighth").values()
    )
    if padding != EXPECTED_EIGHTH_PADDING_MM:
        raise RecoveryExecutionError("B6 eighth padding vector drifted")

    def checked_assign(*args: Any, **kwargs: Any) -> dict[str, Any]:
        plan = original_assign(*args, **kwargs)
        _validate_mesh_plan(plan)
        return plan

    def checked_preflight(*args: Any, **kwargs: Any) -> dict[str, Any]:
        evidence = original_preflight(*args, **kwargs)
        _validate_native_preflight(evidence)
        return evidence

    module._assign_thermal_mesh = checked_assign
    module._generate_and_attest_thermal_mesh = checked_preflight


def main() -> None:
    marker = authenticate_invocation()
    install_recovery_policy()
    print(
        "MFT_ROUNDED_B6_MESH_RECOVERY "
        + json.dumps(marker, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    from run_simulation_260706 import main as runner_main

    runner_main()


if __name__ == "__main__":
    main()
