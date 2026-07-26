"""Prepare, but never submit, the rounded final Standard lane for candidate 5.

The source authority is the exact official candidate-5 package used by the
task96338 direct-Analyze lane.  This utility changes only the winding corner
representation:

* ``round_corner = 1``
* ``corner_radius = 10.0 mm``
* ``corner_segments = 4``

The 1.5 m/s dual-fan, 0.2 W/(m K) TIM/insulation, and both 2 mm pad
conditions are immutable.  The generated Scheduler payload retains one
``symmetric.aedt`` bundle and uses the already reviewed direct-Analyze path,
but this module intentionally contains no POST or submit command.

The package also contains exact drawing dimensions and a post-retention view
export contract.  Native AEDT screenshots are graphical-mode only, so view
export is kept out of the FEA critical path.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools import (  # noqa: E402
    mft_goal_official5_direct_analyze_fastlane as direct,
)
from tools import (  # noqa: E402
    mft_goal_export_rounded_drawing_views as view_export,
)


reviewed = direct.reviewed
PostdeadlineContractError = direct.PostdeadlineContractError

CAMPAIGN_ID = direct.CAMPAIGN_ID
PROJECT = direct.PROJECT
SCHEDULER_URL = direct.SCHEDULER_URL
SOURCE_CANDIDATE_SHA256 = direct.CANDIDATE_SHA256
SOURCE_TASK_ID = 96338
SOURCE_SELECTION_ORDER = 5
# Keep the rounded lane parallel to the n115 candidate-5 verification.  The
# still-active n113 allocation is independently pinned by its terminal source
# task and explicit allocation id so Scheduler cannot silently relax it.
ACCOUNT_NAME = "dw16"
NODE_NAME = "n113"
SAME_NODE_AS_TASK_ID = 96328
SOURCE_ALLOCATION_ID = 14620
SOURCE_SLURM_JOB_ID = "829579"
CPUS = direct.CPUS
MEMORY_MB = direct.MEMORY_MB
SOLVER_SECONDS = 3 * 60 * 60
KILL_GRACE_SECONDS = direct.KILL_GRACE_SECONDS
RETENTION_SECONDS = direct.RETENTION_SECONDS
SCHEDULER_SECONDS = (
    SOLVER_SECONDS + KILL_GRACE_SECONDS + RETENTION_SECONDS
)
MAX_WORKERS_PER_NODE = direct.MAX_WORKERS_PER_NODE
PRIORITY = direct.PRIORITY
LIBRARY_REVISION = direct.LIBRARY_REVISION

ROUND_CORNER = 1
CORNER_RADIUS_MM = 10.0
CORNER_SEGMENTS = 4
ROUNDING_POLICY = {
    "round_corner": ROUND_CORNER,
    "corner_radius_mm": CORNER_RADIUS_MM,
    "corner_segments_per_corner": CORNER_SEGMENTS,
    "path_kind": "uniform_four-chord-per-quarter rounded rectangle",
}
FIXED_BOUNDARY = {
    "fan_config": "dual",
    "fan_velocity_m_s": 1.5,
    "thermal_pad_conductivity_W_mK": 0.2,
    "core_plate_pad_t_mm": 2.0,
    "wcp_pad_t_mm": 2.0,
}

TASK_NAME = (
    "mft-goal-final-standard-official5-rounded-r10-s4-v2-"
    f"{SOURCE_CANDIDATE_SHA256[:12]}-{NODE_NAME}"
)
WORKDIR = (
    "mft_goal_final_standard_official5_rounded_r10_s4_v2_"
    f"{SOURCE_CANDIDATE_SHA256[:12]}_{NODE_NAME}"
)
OUTPUT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\final_standard_official5_rounded_r10_s4_prepare_v2"
)

PLAN_NAME = "rounded_final_prepare_plan.json"
SOURCE_AUTHORITY_NAME = "source_authority.json"
SOURCE_SELECTED_NAME = "source_selected_candidate.json"
PARAMS_NAME = "rounded_fea_params.json"
PROFILE_NAME = "rounded_execution_profile.json"
DIMENSIONS_NAME = "drawing_dimensions.json"
RECEIPT_NAME = "prepare_receipt.json"

PLAN_SCHEMA = "mft-goal-official5-rounded-final-prepare-plan-v2"
RECEIPT_SCHEMA = "mft-goal-official5-rounded-final-prepare-receipt-v2"
REVISION_SCHEMA = "mft-goal-rounded-final-solver-attestation-v1"
DIMENSIONS_SCHEMA = "mft-goal-rounded-final-drawing-dimensions-v1"
PROFILE_SCHEMA = "mft-goal-diagnostic-standard-rounded-final-profile-v1"
PROFILE_PATH = (
    REPOSITORY
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_diagnostic_standard_rounded_final.json"
)

canonical_bytes = direct.canonical_bytes
payload_sha256 = direct.payload_sha256
sealed = direct.sealed
validate_seal = direct.validate_seal
sha256_file = direct.sha256_file
read_json = direct.read_json
write_immutable_json = direct.write_immutable_json


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PostdeadlineContractError(f"{label} is not finite") from exc
    if not math.isfinite(number):
        raise PostdeadlineContractError(f"{label} is not finite")
    return number


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise PostdeadlineContractError(f"{label} is not an integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PostdeadlineContractError(
            f"{label} is not an integer"
        ) from exc
    if number != value or number < (0 if allow_zero else 1):
        raise PostdeadlineContractError(f"{label} is outside its domain")
    return number


def _relative_record(root: Path, path: Path) -> dict[str, Any]:
    root = root.resolve()
    path = path.resolve(strict=True)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PostdeadlineContractError(
            "prepared artifact escapes output root"
        ) from exc
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _contained(root: Path, record: Any, label: str) -> Path:
    if (
        not isinstance(record, Mapping)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or type(record.get("size_bytes")) is not int
    ):
        raise PostdeadlineContractError(f"{label} record is malformed")
    root = root.resolve(strict=True)
    path = (root / str(record["path"])).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PostdeadlineContractError(f"{label} escapes plan root") from exc
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record["size_bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise PostdeadlineContractError(f"{label} bytes drifted")
    return path


def attest_rounded_solver_revision(
    revision: str,
    *,
    source_reader: Callable[[str, str], bytes] = reviewed._git_show,
) -> dict[str, Any]:
    """Require committed direct-Analyze and rounded-racetrack sources."""

    base = direct.attest_solver_revision(
        revision, source_reader=source_reader
    )
    try:
        modeling = source_reader("module/modeling_260706.py", revision)
        input_source = source_reader(
            "module/input_parameter_260706.py", revision
        )
        runner = source_reader("run_simulation_260706.py", revision)
    except Exception as exc:
        raise PostdeadlineContractError(
            "rounded solver source cannot be read"
        ) from exc
    modeling_markers = (
        b"def _rounded_turn_points(",
        b"segments_per_corner=4",
        b"r_i = corner_radius + (x - x_pos[0])",
        b'polyline_kwargs["segment_type"] = segments',
    )
    input_markers = (
        b'"round_corner": 0, "corner_radius": 10.0, "corner_segments": 4',
        b"wcp_len_ref_x = sl1_main_x - (2.0 * corner_radius",
    )
    runner_markers = (
        b'round_corner = int(self.df_plus["round_corner"].iloc[0]) != 0',
        b'corner_segments = int(self.df_plus["corner_segments"].iloc[0])',
    )
    if (
        any(marker not in modeling for marker in modeling_markers)
        or any(marker not in input_source for marker in input_markers)
        or any(marker not in runner for marker in runner_markers)
    ):
        raise PostdeadlineContractError(
            "solver revision lacks the reviewed rounded-turn implementation"
        )
    return sealed(
        {
            "schema_version": REVISION_SCHEMA,
            "solver_revision": revision,
            "direct_analyze_attestation": base,
            "rounded_turn_source": {
                "path": "module/modeling_260706.py",
                "sha256": hashlib.sha256(modeling).hexdigest(),
                "size_bytes": len(modeling),
            },
            "input_contract_source": {
                "path": "module/input_parameter_260706.py",
                "sha256": hashlib.sha256(input_source).hexdigest(),
                "size_bytes": len(input_source),
            },
            "runner_source": {
                "path": "run_simulation_260706.py",
                "sha256": hashlib.sha256(runner).hexdigest(),
                "size_bytes": len(runner),
            },
            "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
            "committed_source_readback": True,
        }
    )


def derive_rounded_params_and_profile(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply only the reviewed rounded-turn transformation."""

    params = copy.deepcopy(source.get("params"))
    source_profile = source.get("profile")
    if not isinstance(params, dict) or not isinstance(source_profile, dict):
        raise PostdeadlineContractError(
            "candidate-5 source params/profile are incomplete"
        )
    source_overrides = source_profile.get("param_overrides")
    if not isinstance(source_overrides, dict):
        raise PostdeadlineContractError(
            "candidate-5 source execution profile is malformed"
        )
    original = {
        "params_round_corner": params.get("round_corner"),
        "params_corner_radius": params.get("corner_radius"),
        "params_corner_segments": params.get("corner_segments"),
        "profile_round_corner": source_overrides.get("round_corner"),
    }
    if original != {
        "params_round_corner": 0,
        "params_corner_radius": 10.0,
        "params_corner_segments": 4,
        "profile_round_corner": 0,
    }:
        raise PostdeadlineContractError(
            "source candidate rounding baseline drifted"
        )
    params.update(
        {
            "round_corner": ROUND_CORNER,
            "corner_radius": CORNER_RADIUS_MM,
            "corner_segments": CORNER_SEGMENTS,
        }
    )
    try:
        profile = read_json(PROFILE_PATH.resolve(strict=True))
    except OSError as exc:
        raise PostdeadlineContractError(
            "reviewed rounded final profile is unavailable"
        ) from exc
    overrides = profile.get("param_overrides")
    if (
        profile.get("schema_version") != PROFILE_SCHEMA
        or profile.get("stage") != "standard"
        or profile.get("cpus") != CPUS
        or profile.get("mem_mb") != MEMORY_MB
        or profile.get("timeout_seconds") != SOLVER_SECONDS
        or not isinstance(overrides, dict)
        or profile.get("fixed_boundary_contract")
        != source_profile.get("fixed_boundary_contract")
        or profile.get("artifact_retention")
        != source_profile.get("artifact_retention")
    ):
        raise PostdeadlineContractError(
            "reviewed rounded final profile drifted"
        )
    effective = copy.deepcopy(params)
    effective.update(overrides)
    _validate_effective_contract(effective)
    return params, profile


def _validate_effective_contract(effective: Mapping[str, Any]) -> None:
    observed = {
        "round_corner": effective.get("round_corner"),
        "corner_radius": effective.get("corner_radius"),
        "corner_segments": effective.get("corner_segments"),
        "full_model": effective.get("full_model"),
        "thermal_symmetry": effective.get("thermal_symmetry"),
        "fan_config": effective.get("fan_config"),
        "fan_velocity": effective.get("fan_velocity"),
        "k_ins": effective.get("k_ins"),
        "core_plate_pad_t": effective.get("core_plate_pad_t"),
        "wcp_pad_t": effective.get("wcp_pad_t"),
        "keep_project": effective.get("keep_project"),
        "matrix_on": effective.get("matrix_on"),
        "loss_on": effective.get("loss_on"),
        "thermal_on": effective.get("thermal_on"),
    }
    expected = {
        "round_corner": 1,
        "corner_radius": 10.0,
        "corner_segments": 4,
        "full_model": 0,
        "thermal_symmetry": "eighth",
        "fan_config": "dual",
        "fan_velocity": 1.5,
        "k_ins": 0.2,
        "core_plate_pad_t": 2.0,
        "wcp_pad_t": 2.0,
        "keep_project": 1,
        "matrix_on": 1,
        "loss_on": 1,
        "thermal_on": 1,
    }
    if observed != expected:
        raise PostdeadlineContractError(
            f"rounded final effective contract drifted: {observed}"
        )


@contextmanager
def _reviewed_payload_patch(solver_revision: str) -> Any:
    replacements = {
        "SCHEDULER_URL": SCHEDULER_URL,
        "PROJECT": PROJECT,
        "ACCOUNT_NAME": ACCOUNT_NAME,
        "NODE_NAME": NODE_NAME,
        "TASK_NAME": TASK_NAME,
        "WORKDIR": WORKDIR,
        "CPUS": CPUS,
        "MEMORY_MB": MEMORY_MB,
        "SOLVER_SECONDS": SOLVER_SECONDS,
        "KILL_GRACE_SECONDS": KILL_GRACE_SECONDS,
        "RETENTION_SECONDS": RETENTION_SECONDS,
        "SCHEDULER_SECONDS": SCHEDULER_SECONDS,
        "MAX_WORKERS_PER_NODE": MAX_WORKERS_PER_NODE,
        "PRIORITY": PRIORITY,
        "SOLVER_REVISION": solver_revision,
        "LIBRARY_REVISION": LIBRARY_REVISION,
        "CORE_AUTH_SHA256": direct.standalone_core_auth_sha256(
            solver_revision
        ),
    }
    previous = {name: getattr(reviewed, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(reviewed, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(reviewed, name, value)


def derive_scheduler_payload(
    params: dict[str, Any],
    profile: dict[str, Any],
    solver_revision: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derive with an in-process fake POST; perform no network mutation."""

    with _reviewed_payload_patch(solver_revision):
        payload, environment, retained = (
            reviewed._capture_scheduler_payload(params, profile)
        )
        reviewed.validate_scheduler_payload(payload, retained)
    payload = copy.deepcopy(payload)
    environment = copy.deepcopy(environment)
    retained = copy.deepcopy(retained)
    payload["same_node_as_task_id"] = SAME_NODE_AS_TASK_ID
    payload["command"] = direct._direct_command(
        str(payload.get("command") or "")
    )
    environment[direct.DIRECT_ENV_NAME] = direct.DIRECT_ENV_TOKEN
    validate_rounded_payload(
        payload,
        environment,
        retained,
        params=params,
        profile=profile,
        solver_revision=solver_revision,
    )
    return payload, environment, retained


def validate_rounded_payload(
    payload: Mapping[str, Any],
    environment: Mapping[str, Any],
    retained: Mapping[str, Any],
    *,
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
    solver_revision: str,
) -> None:
    effective = copy.deepcopy(dict(params))
    overrides = profile.get("param_overrides")
    if not isinstance(overrides, dict):
        raise PostdeadlineContractError("rounded profile overrides are absent")
    effective.update(overrides)
    _validate_effective_contract(effective)
    expected = {
        "name": TASK_NAME,
        "project": PROJECT,
        "account_name": ACCOUNT_NAME,
        "node_name": NODE_NAME,
        "node_name_policy": "strict",
        "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
        "cpus": CPUS,
        "memory_mb": MEMORY_MB,
        "max_workers_per_node": MAX_WORKERS_PER_NODE,
        "timeout_seconds": SCHEDULER_SECONDS,
        "aedt_backend": "standalone",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    command = str(payload.get("command") or "")
    core_auth = direct.standalone_core_auth_sha256(solver_revision)
    artifact = str(retained.get("artifact_path") or "")
    results = str(retained.get("results_path") or "")
    chunks = str(
        retained.get("transport", {}).get("chunk_directory") or ""
    )
    if (
        drift
        or environment.get(direct.DIRECT_ENV_NAME)
        != direct.DIRECT_ENV_TOKEN
        or environment.get("MFT_STANDALONE_CORE_CONTRACT")
        != direct.STANDALONE_CORE_CONTRACT
        or environment.get("MFT_STANDALONE_CORE_COUNT") != str(CPUS)
        or environment.get("MFT_STANDALONE_CORE_AUTH_SHA256")
        != core_auth
        or command.count("--symmetry-thermal-direct-analyze") != 1
        or "--full" in command
        or solver_revision not in command
        or str(payload.get("dedupe_key") or "") == ""
        or payload.get("dedupe_key") != retained.get("dedupe_key")
        or retained.get("solver_revision") != solver_revision
        or retained.get("stage") != "standard"
        or not artifact.endswith("/symmetric.aedt")
        or not results.endswith("/symmetric.aedtresults")
        or not chunks.endswith("/symmetric.aedt.chunks")
        or retained.get("retention_required") is not True
        or retained.get("prune_protection_required") is not True
    ):
        raise PostdeadlineContractError(
            f"rounded Scheduler payload/retention drifted: {drift}"
        )


def _build_width(turns: int, conductor: float, gap: float) -> float:
    if turns <= 0:
        return 0.0
    return turns * conductor + (turns - 1) * gap


def _turn_group(
    *,
    name: str,
    turns: int,
    conductor_thickness: float,
    conductor_height: float,
    gap_x: float,
    gaps_y: Sequence[float],
    clear_span_x: float,
    clear_span_y: float,
    offset_x: float,
) -> dict[str, Any]:
    if len(gaps_y) != max(turns - 1, 0):
        raise PostdeadlineContractError(f"{name} y-gap count drifted")
    x = clear_span_x / 2.0 + conductor_thickness / 2.0
    y = clear_span_y / 2.0 + conductor_thickness / 2.0
    records: list[dict[str, Any]] = []
    radius = CORNER_RADIUS_MM
    for index in range(turns):
        if index:
            x += conductor_thickness + gap_x
            y += conductor_thickness + gaps_y[index - 1]
            radius += conductor_thickness + gap_x
        if radius <= 0 or radius >= min(x, y):
            raise PostdeadlineContractError(
                f"{name} turn {index} rounded radius is invalid"
            )
        chord = 2.0 * radius * math.sin(
            math.pi / (4.0 * CORNER_SEGMENTS)
        )
        polygon_corner_length = 4.0 * CORNER_SEGMENTS * chord
        straight_length = 4.0 * (x + y - 2.0 * radius)
        records.append(
            {
                "turn_index_zero_based": index,
                "center_offset_x_mm": offset_x,
                "center_offset_y_mm": 0.0,
                "center_offset_z_mm": 0.0,
                "centerline_half_extent_x_mm": x,
                "centerline_half_extent_y_mm": y,
                "corner_radius_centerline_mm": radius,
                "horizontal_straight_span_centerline_mm": 2.0 * (x - radius),
                "vertical_straight_span_centerline_mm": 2.0 * (y - radius),
                "polygon_corner_chord_length_mm": chord,
                "polygon_centerline_perimeter_mm": (
                    straight_length + polygon_corner_length
                ),
                "exact_arc_reference_perimeter_mm": (
                    straight_length + 2.0 * math.pi * radius
                ),
                "polygon_radial_sagitta_mm": (
                    radius
                    * (
                        1.0
                        - math.cos(
                            math.pi / (4.0 * CORNER_SEGMENTS)
                        )
                    )
                ),
            }
        )
    return {
        "name": name,
        "turn_count": turns,
        "conductor_cross_section": {
            "radial_thickness_mm": conductor_thickness,
            "axial_height_mm": conductor_height,
        },
        "turn_center_z_mm": 0.0,
        "z_pitch_mm": 0.0,
        "x_gap_between_conductors_mm": gap_x,
        "y_gaps_between_conductors_mm": list(gaps_y),
        "x_center_pitch_mm": conductor_thickness + gap_x,
        "y_center_pitches_mm": [
            conductor_thickness + gap for gap in gaps_y
        ],
        "radial_build_x_mm": _build_width(
            turns, conductor_thickness, gap_x
        ),
        "clear_window_span_x_mm": clear_span_x,
        "clear_window_span_y_mm": clear_span_y,
        "innermost_clear_straight_span_x_mm": (
            clear_span_x - 2.0 * CORNER_RADIUS_MM
        ),
        "innermost_clear_straight_span_y_mm": (
            clear_span_y - 2.0 * CORNER_RADIUS_MM
        ),
        "radius_ladder_mm": [
            item["corner_radius_centerline_mm"] for item in records
        ],
        "turns": records,
    }


def derive_drawing_dimensions(
    params: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Build exact dimensions from the same effective solver payload."""

    effective = copy.deepcopy(dict(params))
    overrides = profile.get("param_overrides")
    if not isinstance(overrides, dict):
        raise PostdeadlineContractError("rounded profile overrides are absent")
    effective.update(overrides)
    _validate_effective_contract(effective)

    n1m = _positive_int(effective.get("N1_main"), "N1_main")
    n1s = _positive_int(
        effective.get("N1_side"), "N1_side", allow_zero=True
    )
    n2m = _positive_int(effective.get("N2_main"), "N2_main")
    n2s = _positive_int(effective.get("N2_side"), "N2_side")
    if n1s != 0:
        raise PostdeadlineContractError(
            "candidate-5 drawing contract expects no primary side turns"
        )
    l1 = _finite(effective.get("l1"), "l1")
    l2 = _finite(effective.get("l2"), "l2")
    h1 = _finite(effective.get("h1"), "h1")
    w1 = _finite(effective.get("w1"), "w1")
    cw1 = _finite(effective.get("cw1"), "cw1")
    cw2 = _finite(effective.get("cw2"), "cw2")
    gap1 = _finite(effective.get("gap1"), "gap1")
    gap2 = _finite(effective.get("gap2"), "gap2")
    nwh1 = _finite(effective.get("nwh1"), "nwh1")
    nwh2 = _finite(effective.get("nwh2"), "nwh2")

    nwl1m = _build_width(n1m, cw1, gap1)
    nwl2m = _build_width(n2m, cw2, gap2)
    nwl2s = _build_width(n2s, cw2, gap2)
    wcp_t = _finite(effective.get("wcp_t"), "wcp_t")
    wcp_pad = _finite(effective.get("wcp_pad_t"), "wcp_pad_t")
    wcp_slot = wcp_t + 2.0 * wcp_pad
    tx_y_gaps = [gap1] * (n1m - 1)
    tx_y_gaps[0] = wcp_slot
    tx_y_gaps[-1] = wcp_slot
    nwb1m_y = n1m * cw1 + sum(tx_y_gaps)

    sl2m_x = 2.0 * l1 + 2.0 * _finite(
        effective.get("cc_w2c_space_x"), "cc_w2c_space_x"
    )
    sl2m_y = w1 + 2.0 * _finite(
        effective.get("cc_w2c_space_y"), "cc_w2c_space_y"
    )
    sl1m_x = (
        sl2m_x
        + 2.0 * nwl2m
        + 2.0
        * _finite(
            effective.get("w2c_w1c_space_x"),
            "w2c_w1c_space_x",
        )
    )
    sl1m_y = (
        sl2m_y
        + 2.0 * nwl2m
        + 2.0
        * _finite(
            effective.get("w2c_w1c_space_y"),
            "w2c_w1c_space_y",
        )
    )
    sl2s_x = l1 + 2.0 * _finite(
        effective.get("w1s_cs_space_x"), "w1s_cs_space_x"
    )
    sl2s_y = w1 + 2.0 * _finite(
        effective.get("cs_w1s_space_y"), "cs_w1s_space_y"
    )
    side_offset = l1 + l2 + l1 / 2.0

    core_x = 4.0 * l1 + 2.0 * l2
    center_x = sl1m_x + 2.0 * nwl1m
    center_y = sl1m_y + 2.0 * nwb1m_y
    side_x = 2.0 * (side_offset + sl2s_x / 2.0 + nwl2s)
    side_y = 2.0 * (sl2s_y / 2.0 + nwl2s)
    exterior = {
        "width_x_mm": max(core_x, center_x, side_x),
        "length_y_mm": max(w1, center_y, side_y),
        "height_z_mm": h1 + 2.0 * l1,
    }
    exterior["volume_L"] = (
        exterior["width_x_mm"]
        * exterior["length_y_mm"]
        * exterior["height_z_mm"]
        * 1e-6
    )

    core_groups = _positive_int(
        effective.get("n_core_group"), "n_core_group"
    )
    core_plate_t = _finite(
        effective.get("core_plate_t"), "core_plate_t"
    )
    core_pad_t = _finite(
        effective.get("core_plate_pad_t"), "core_plate_pad_t"
    )
    core_plate_stack = core_plate_t + 2.0 * core_pad_t
    core_depth = (
        w1 - (core_groups + 1) * core_plate_stack
    ) / core_groups

    tx = _turn_group(
        name="Tx_main",
        turns=n1m,
        conductor_thickness=cw1,
        conductor_height=nwh1,
        gap_x=gap1,
        gaps_y=tx_y_gaps,
        clear_span_x=sl1m_x,
        clear_span_y=sl1m_y,
        offset_x=0.0,
    )
    rx_main = _turn_group(
        name="Rx_main",
        turns=n2m,
        conductor_thickness=cw2,
        conductor_height=nwh2,
        gap_x=gap2,
        gaps_y=[gap2] * (n2m - 1),
        clear_span_x=sl2m_x,
        clear_span_y=sl2m_y,
        offset_x=0.0,
    )
    rx_side = _turn_group(
        name="Rx_side",
        turns=n2s,
        conductor_thickness=cw2,
        conductor_height=nwh2,
        gap_x=gap2,
        gaps_y=[gap2] * (n2s - 1),
        clear_span_x=sl2s_x,
        clear_span_y=sl2s_y,
        offset_x=-side_offset,
    )
    wcp_len = _finite(effective.get("wcp_len_x"), "wcp_len_x")
    wcp_ref = sl1m_x - 2.0 * CORNER_RADIUS_MM
    actual_tx_side_gap = (
        l2
        - (
            _finite(
                effective.get("cc_w2c_space_x"),
                "cc_w2c_space_x",
            )
            + nwl2m
            + _finite(
                effective.get("w2c_w1c_space_x"),
                "w2c_w1c_space_x",
            )
            + nwl1m
        )
        - (
            _finite(
                effective.get("w1s_cs_space_x"),
                "w1s_cs_space_x",
            )
            + nwl2s
        )
    )

    value = {
        "schema_version": DIMENSIONS_SCHEMA,
        "source_candidate_physics_sha256": SOURCE_CANDIDATE_SHA256,
        "source_standard_task_id": SOURCE_TASK_ID,
        "units": "mm unless explicitly stated",
        "coordinate_system": {
            "width_axis": "x",
            "length_or_depth_axis": "y",
            "height_axis": "z",
            "winding_center_plane_z_mm": 0.0,
        },
        "rounding": {
            **copy.deepcopy(ROUNDING_POLICY),
            "radius_rule": (
                "R_i=10mm+(x_center_i-x_center_0); "
                "same x-pitch ladder for each winding group"
            ),
            "nominal_arc_reference_only": True,
            "actual_aedt_path": (
                "four equal straight chords per 90-degree corner"
            ),
            "maximum_polygon_sagitta_mm": max(
                max(tx["turns"], key=lambda row: row[
                    "polygon_radial_sagitta_mm"
                ])["polygon_radial_sagitta_mm"],
                max(rx_main["turns"], key=lambda row: row[
                    "polygon_radial_sagitta_mm"
                ])["polygon_radial_sagitta_mm"],
                max(rx_side["turns"], key=lambda row: row[
                    "polygon_radial_sagitta_mm"
                ])["polygon_radial_sagitta_mm"],
            ),
        },
        "exterior_envelope": exterior,
        "core": {
            "outer_width_x_mm": core_x,
            "outer_height_z_mm": h1 + 2.0 * l1,
            "total_depth_y_mm": w1,
            "center_leg_full_width_x_mm": 2.0 * l1,
            "side_leg_width_x_mm": l1,
            "window_width_x_mm": l2,
            "window_height_z_mm": h1,
            "group_count": core_groups,
            "magnetic_depth_each_group_y_mm": core_depth,
            "cooling_assembly_count_y": core_groups + 1,
            "cooling_assembly_stack_y_mm": core_plate_stack,
            "aluminum_plate_t_mm": core_plate_t,
            "thermal_pad_each_side_t_mm": core_pad_t,
        },
        "winding_groups": [tx, rx_main, rx_side],
        "full_model_side_offsets_x_mm": [-side_offset, side_offset],
        "standard_model_retained_octant": {
            "x": "negative",
            "y": "positive",
            "z": "positive",
            "drawing_use": (
                "audit only; final overall drawing views must use the "
                "rounded Full AEDT"
            ),
        },
        "winding_cooling_plates": {
            "slot_gap_total_y_mm": wcp_slot,
            "slot_indices_zero_based": [0, n1m - 2],
            "aluminum_t_y_mm": wcp_t,
            "thermal_pad_each_side_t_y_mm": wcp_pad,
            "length_x_mm": wcp_len,
            "height_z_mm": nwh1,
            "rounded_inner_clear_reference_x_mm": wcp_ref,
            "length_fraction_percent": 100.0 * wcp_len / wcp_ref,
            "end_clearance_each_side_x_mm": (wcp_ref - wcp_len) / 2.0,
            "mirror_sides_y": ["negative", "positive"],
        },
        "clearances": {
            "center_core_to_Rx_main_x_mm": _finite(
                effective.get("cc_w2c_space_x"), "cc_w2c_space_x"
            ),
            "center_core_to_Rx_main_y_mm": _finite(
                effective.get("cc_w2c_space_y"), "cc_w2c_space_y"
            ),
            "Rx_main_to_Tx_main_x_mm": _finite(
                effective.get("w2c_w1c_space_x"),
                "w2c_w1c_space_x",
            ),
            "Rx_main_to_Tx_main_y_mm": _finite(
                effective.get("w2c_w1c_space_y"),
                "w2c_w1c_space_y",
            ),
            "side_core_to_Rx_side_x_mm": _finite(
                effective.get("w1s_cs_space_x"), "w1s_cs_space_x"
            ),
            "side_core_to_Rx_side_y_mm": _finite(
                effective.get("cs_w1s_space_y"), "cs_w1s_space_y"
            ),
            "Tx_main_to_Rx_side_actual_x_mm": actual_tx_side_gap,
            "Tx_main_to_Rx_side_required_x_mm": _finite(
                effective.get("w1c_w2s_space_x"),
                "w1c_w2s_space_x",
            ),
            "Tx_axial_core_gap_each_side_z_mm": (h1 - nwh1) / 2.0,
            "Rx_axial_core_gap_each_side_z_mm": (h1 - nwh2) / 2.0,
        },
        "fixed_thermal_boundary": copy.deepcopy(FIXED_BOUNDARY),
        "thermal_geometry_disclosure": {
            "em_windings_use_rounded_path": True,
            "icepak_Tx_path_uses_round_corner": False,
            "icepak_Rx_representation": (
                "homogenized rectangular blocks when n_explicit_turns=0"
            ),
            "interpretation": (
                "rounded Standard updates EM/capacitance/loss geometry; "
                "thermal shape remains the reviewed homogenized model and "
                "consumes the rounded EM loss allocation"
            ),
        },
        "view_export_contract": {
            "tool": "tools/mft_goal_export_rounded_drawing_views.py",
            "schema_version": view_export.VIEW_EXPORT_SCHEMA,
            "automatic_in_solver_task": False,
            "requires_graphical_aedt": True,
            "preferred_background": "white",
            "required_overall_views": ["top", "front", "isometric"],
            "required_detail_views": [
                "center_winding_detail_top",
                "side_winding_detail_top",
            ],
            "drawing_color_convention_rgb": {
                "primary": list(view_export.TX_COLOR),
                "secondary": list(view_export.RX_COLOR),
                "plates": list(view_export.PLATE_COLOR),
                "pads": list(view_export.PAD_COLOR),
                "core": list(view_export.CORE_COLOR),
            },
        },
    }
    return sealed(value)


SourceAuthenticator = Callable[[], dict[str, Any]]
RevisionAttester = Callable[[str], dict[str, Any]]
PayloadBuilder = Callable[
    [dict[str, Any], dict[str, Any], str],
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
]


def prepare(
    *,
    solver_revision: str,
    output: Path = OUTPUT_ROOT,
    source_authenticator: SourceAuthenticator = (
        direct.authenticate_source_authority
    ),
    revision_attester: RevisionAttester = attest_rounded_solver_revision,
    payload_builder: PayloadBuilder = derive_scheduler_payload,
    observed_at: datetime | None = None,
) -> Path:
    """Atomically seal one no-POST rounded final lane package."""

    target = output.resolve()
    if target.exists():
        raise PostdeadlineContractError(
            f"immutable rounded prepare output already exists: {target}"
        )
    now = observed_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise PostdeadlineContractError(
            "rounded prepare time must be timezone-aware"
        )
    revision = revision_attester(solver_revision)
    if (
        validate_seal(revision, REVISION_SCHEMA) is not revision
        or revision.get("solver_revision") != solver_revision
    ):
        raise PostdeadlineContractError(
            "rounded solver revision attestation drifted"
        )
    source = source_authenticator()
    authority = source.get("authority")
    selected = source.get("selected")
    if (
        not isinstance(authority, dict)
        or not isinstance(selected, dict)
        or authority.get("candidate_physics_sha256")
        != SOURCE_CANDIDATE_SHA256
        or authority.get("official_standard_selection_order")
        != SOURCE_SELECTION_ORDER
    ):
        raise PostdeadlineContractError(
            "official candidate-5 source authority is incomplete"
        )
    params, profile = derive_rounded_params_and_profile(source)
    payload, environment, retained = payload_builder(
        params, profile, solver_revision
    )
    validate_rounded_payload(
        payload,
        environment,
        retained,
        params=params,
        profile=profile,
        solver_revision=solver_revision,
    )
    dimensions = derive_drawing_dimensions(params, profile)
    exporter_path = (
        REPOSITORY / "tools" / "mft_goal_export_rounded_drawing_views.py"
    ).resolve(strict=True)

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent)
    )
    try:
        authority_path = write_immutable_json(
            staging / SOURCE_AUTHORITY_NAME, authority
        )
        selected_path = write_immutable_json(
            staging / SOURCE_SELECTED_NAME, selected
        )
        params_path = write_immutable_json(staging / PARAMS_NAME, params)
        profile_path = write_immutable_json(staging / PROFILE_NAME, profile)
        dimensions_path = write_immutable_json(
            staging / DIMENSIONS_NAME, dimensions
        )
        plan = sealed(
            {
                "schema_version": PLAN_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "created_at_utc": now.astimezone(timezone.utc).isoformat(),
                "prepare_only": True,
                "scheduler_get_calls": 0,
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
                "submission_capability_present": False,
                "source_candidate_physics_sha256": (
                    SOURCE_CANDIDATE_SHA256
                ),
                "source_standard_task_id": SOURCE_TASK_ID,
                "placement": {
                    "account_name": ACCOUNT_NAME,
                    "node_name": NODE_NAME,
                    "node_name_policy": "strict",
                    "preferred_node_relaxed_allowed": False,
                    "same_node_as_task_id": SAME_NODE_AS_TASK_ID,
                    "same_node_as_allocation_id": SOURCE_ALLOCATION_ID,
                    "same_node_as_slurm_job_id": SOURCE_SLURM_JOB_ID,
                    "requested_allocation_id_posted": False,
                    "allocation_selected_by_same_node_lineage": True,
                    "parallel_to_source_task96338": True,
                },
                "source_authority": _relative_record(
                    staging, authority_path
                ),
                "source_selected_candidate": _relative_record(
                    staging, selected_path
                ),
                "rounded_fea_params": _relative_record(
                    staging, params_path
                ),
                "rounded_fea_params_sha256": payload_sha256(params),
                "rounded_execution_profile": _relative_record(
                    staging, profile_path
                ),
                "effective_rounded_params_sha256": payload_sha256(
                    {
                        **params,
                        **profile["param_overrides"],
                    }
                ),
                "drawing_dimensions": _relative_record(
                    staging, dimensions_path
                ),
                "drawing_dimensions_payload_sha256": dimensions[
                    "payload_sha256"
                ],
                "solver_revision_attestation": revision,
                "solver_revision": solver_revision,
                "library_revision": LIBRARY_REVISION,
                "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
                "fixed_boundary": copy.deepcopy(FIXED_BOUNDARY),
                "scheduler_payload": payload,
                "scheduler_payload_sha256": payload_sha256(payload),
                "submission_environment": environment,
                "submission_environment_sha256": payload_sha256(
                    environment
                ),
                "retained_aedt_bundle": retained,
                "retained_aedt_bundle_sha256": payload_sha256(retained),
                "resource_contract": {
                    "cpus": CPUS,
                    "memory_mb": MEMORY_MB,
                    "max_workers_per_node": MAX_WORKERS_PER_NODE,
                    "solver_timeout_seconds": SOLVER_SECONDS,
                    "scheduler_timeout_seconds": SCHEDULER_SECONDS,
                    "aedt_backend": "standalone",
                },
                "selection_prerequisite": {
                    "task96338_terminal_success_required": False,
                    "parallel_to_task96338": True,
                    "small_miss_policy": (
                        "same-candidate local correction; do not restart "
                        "global candidate rejection loops"
                    ),
                    "automatic_full_trigger": False,
                    "full_model_limit": (
                        "one selected rounded candidate only"
                    ),
                },
                "view_export": {
                    "automatic_in_solver_task": False,
                    "requires_graphical_aedt": True,
                    "exporter": {
                        "path": str(
                            exporter_path.relative_to(REPOSITORY).as_posix()
                        ),
                        "sha256": sha256_file(exporter_path),
                        "size_bytes": exporter_path.stat().st_size,
                    },
                    "post_collection_command_argv": [
                        "conda",
                        "run",
                        "-n",
                        "pyaedt2026v1",
                        "python",
                        "tools/mft_goal_export_rounded_drawing_views.py",
                        "--project",
                        "<collected-symmetric.aedt>",
                        "--output",
                        "<drawing-view-output>",
                        "--background",
                        "white",
                    ],
                    "final_drawing_requires_rounded_full_views": True,
                },
                "thermal_geometry_disclosure": dimensions[
                    "thermal_geometry_disclosure"
                ],
            }
        )
        plan_path = write_immutable_json(staging / PLAN_NAME, plan)
        receipt = sealed(
            {
                "schema_version": RECEIPT_SCHEMA,
                "created_at_utc": now.astimezone(timezone.utc).isoformat(),
                "prepare_only": True,
                "plan": _relative_record(staging, plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "source_candidate_physics_sha256": (
                    SOURCE_CANDIDATE_SHA256
                ),
                "solver_revision": solver_revision,
                "rounding_policy": copy.deepcopy(ROUNDING_POLICY),
                "scheduler_post_calls": 0,
                "scheduler_submission_performed": False,
                "ready_for_submit_tool_implementation": True,
                "automatic_full_trigger": False,
            }
        )
        write_immutable_json(staging / RECEIPT_NAME, receipt)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target / PLAN_NAME


def load_plan(plan_path: Path | None = None) -> dict[str, Any]:
    """Reauthenticate every immutable local artifact; perform no network I/O."""

    path = (plan_path or (OUTPUT_ROOT / PLAN_NAME)).resolve(strict=True)
    plan = validate_seal(read_json(path), PLAN_SCHEMA)
    root = path.parent
    authority_path = _contained(
        root, plan.get("source_authority"), "source authority"
    )
    selected_path = _contained(
        root,
        plan.get("source_selected_candidate"),
        "source selected candidate",
    )
    params_path = _contained(
        root, plan.get("rounded_fea_params"), "rounded params"
    )
    profile_path = _contained(
        root, plan.get("rounded_execution_profile"), "rounded profile"
    )
    dimensions_path = _contained(
        root, plan.get("drawing_dimensions"), "drawing dimensions"
    )
    source = direct.authenticate_source_authority()
    if (
        read_json(authority_path) != source.get("authority")
        or read_json(selected_path) != source.get("selected")
    ):
        raise PostdeadlineContractError(
            "rounded plan source authority drifted"
        )
    params = read_json(params_path)
    profile = read_json(profile_path)
    expected_params, expected_profile = derive_rounded_params_and_profile(
        source
    )
    dimensions = validate_seal(
        read_json(dimensions_path), DIMENSIONS_SCHEMA
    )
    if (
        params != expected_params
        or profile != expected_profile
        or dimensions != derive_drawing_dimensions(params, profile)
        or plan.get("rounded_fea_params_sha256")
        != payload_sha256(params)
        or plan.get("drawing_dimensions_payload_sha256")
        != dimensions["payload_sha256"]
    ):
        raise PostdeadlineContractError(
            "rounded plan artifact derivation drifted"
        )
    payload, environment, retained = derive_scheduler_payload(
        params, profile, str(plan["solver_revision"])
    )
    if (
        plan.get("scheduler_payload") != payload
        or plan.get("scheduler_payload_sha256")
        != payload_sha256(payload)
        or plan.get("submission_environment") != environment
        or plan.get("submission_environment_sha256")
        != payload_sha256(environment)
        or plan.get("retained_aedt_bundle") != retained
        or plan.get("retained_aedt_bundle_sha256")
        != payload_sha256(retained)
        or plan.get("scheduler_post_calls") != 0
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("submission_capability_present") is not False
    ):
        raise PostdeadlineContractError(
            "rounded plan Scheduler derivation drifted"
        )
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="seal one local no-POST rounded lane"
    )
    prepare_parser.add_argument("--solver-revision", required=True)
    prepare_parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    inspect_parser = commands.add_parser(
        "inspect", help="reauthenticate one existing local plan"
    )
    inspect_parser.add_argument(
        "--plan", type=Path, default=OUTPUT_ROOT / PLAN_NAME
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare(
            solver_revision=args.solver_revision,
            output=args.output,
        )
        plan = validate_seal(read_json(path), PLAN_SCHEMA)
        result = {
            "event": "rounded_final_lane_prepared",
            "plan": str(path),
            "plan_payload_sha256": plan["payload_sha256"],
            "source_candidate_physics_sha256": (
                SOURCE_CANDIDATE_SHA256
            ),
            "solver_revision": plan["solver_revision"],
            "scheduler_get_calls": 0,
            "scheduler_post_calls": 0,
            "scheduler_submission_performed": False,
        }
    else:
        plan = load_plan(args.plan)
        result = {
            "event": "rounded_final_lane_plan_authenticated",
            "plan": str(args.plan.resolve()),
            "plan_payload_sha256": plan["payload_sha256"],
            "scheduler_get_calls": 0,
            "scheduler_post_calls": 0,
            "scheduler_submission_performed": False,
        }
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PostdeadlineContractError as exc:
        print(
            json.dumps(
                {
                    "event": "rounded_final_lane_prepare_error",
                    "error": str(exc),
                    "scheduler_get_calls": 0,
                    "scheduler_post_calls": 0,
                    "scheduler_submission_performed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
