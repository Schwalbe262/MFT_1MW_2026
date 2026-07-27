"""Prepare and collect a bounded symmetric FEA neighborhood batch.

The input authority is the *final* cross-seed collector for the targeted
``cw1=5 mm / gap1=1.6 mm / fixed-Lm=2 mH`` NSGA-II campaign.  Preparation is
strictly local: it selects 8--16 diverse candidates and writes immutable
Scheduler-ready lane records.  No Scheduler POST is possible unless the
operator invokes ``submit --apply`` against that sealed plan.

The current AEDT parameter schema has no physical core-air-gap variable.
Consequently these runs are acquisition runs: their measured leakage,
capacitance, loss, and temperature are re-scored with ``Lm=2 mH`` while the
actual air-gap geometry remains a required later tuning/attestation step.  The
collector never calls such a run a final design PASS.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.parse
import urllib.request


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.input_parameter_260706 import ALL_INPUT_KEYS  # noqa: E402
from module import thermal_truth_contract as thermal_truth  # noqa: E402
from regression_260707.optimization import geometry_metrics  # noqa: E402
from regression_260707.verify import scheduler_client  # noqa: E402
from tools import tier1_corrected_generation_preflight as preflight  # noqa: E402


TARGETED_STATUS_SCHEMA = "mft-goal-fixed-lm2mh-targeted-global-nds-v1"
TARGETED_PARETO_SCHEMA = (
    "mft-goal-fixed-lm2mh-targeted-pareto-manifest-v1"
)
BATCH_PLAN_SCHEMA = "mft-goal-targeted-symmetric-fea-batch-plan-v1"
SUBMISSION_SCHEMA = "mft-goal-targeted-symmetric-fea-submission-v1"
COLLECTION_SCHEMA = "mft-goal-targeted-symmetric-fea-collection-v1"
DRY_RUN_SCHEMA = "mft-goal-targeted-symmetric-fea-dry-run-v1"
TARGETED_CAMPAIGN_ID = (
    "mft-goal-fixed-primary-5t-lm2mh-axis-w1200-l1000-targeted-v1"
)
BATCH_CAMPAIGN_ID = "mft-goal-lm2mh-symmetric-neighborhood-v1"
EXPECTED_HARD_SPEC_SHA256 = (
    "227cdc0db3b8dae490275d549e8aea93b295a75ee97ea98cdaa601bb96591298"
)
EXPECTED_RESONANCE_CONTRACT_SHA256 = (
    "9858b78f3084d344077fe0d90f7e390f7485d18248d1d711d2fde22c36c01ed1"
)
PROFILE_PATH = (
    REPOSITORY_ROOT
    / "regression_260707"
    / "verify"
    / "profiles"
    / "goal_standard.json"
)
PROFILE_CANONICAL_SHA256 = (
    "767f78f2b269bb1196bbfa5fba4f178cc13ee442fa77e004d8674fed0a04e049"
)
DEFAULT_AUTHORITY_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_lm2mh_targeted_w1200_l1000_v1_global_nds"
)
DEFAULT_OUTPUT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\targeted_symmetric_fea_neighborhood_v1"
)
TURN_GRADED_CAP_CANARY_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rank1_turn_graded_cap_canary_v3"
)
TURN_GRADED_CAP_CALIBRATION_PLAN = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\targeted_symmetric_fea_neighborhood_v2\batch_plan.json"
)
TURN_GRADED_CAP_COLLECTION_SCHEMA = (
    "mft-goal-turn-graded-cap-collection-v1"
)
TURN_GRADED_CAP_PLAN_SCHEMA = "mft-goal-turn-graded-cap-batch-plan-v1"
EXPECTED_TURN_GRADED_CAP_COLLECTION_PAYLOAD_SHA256 = (
    "b56e6381489d051be9fde63e141b7ff8e35d2557f49421f0485645bcecb8deda"
)
EXPECTED_TURN_GRADED_CAP_PLAN_PAYLOAD_SHA256 = (
    "adf169f90a0842be540363ec1f4ccfc62db94fed49f690829a1f1e4ef9eb9bdd"
)
EXPECTED_TURN_GRADED_CAP_CALIBRATION_PLAN_PAYLOAD_SHA256 = (
    "87541e4d0fbcebed6451682ee03e4bce2dab358824d218833c756d9e21613bf9"
)
TURN_GRADED_CAP_CALIBRATION_GEOMETRY_SHA256 = (
    "2b2138a99445c4ed7d50db5b07f7617789a6ff7af735a85fd7038fd1ab266d60"
)
SCHEDULER_URL = "http://127.0.0.1:8002"
LM_TARGET_H = 0.002
RESONANCE_MIN_HZ = 15_000.0
SIZE_LIMITS_MM = {"W": 1200.0, "L": 1000.0, "H": 750.0}
PRIMARY_CONDUCTOR_MM = 5.0
PRIMARY_GAP_MM = 1.6
MIN_BATCH = 8
MAX_BATCH = 16
DEFAULT_BATCH = 12
CPUS = 8
MEMORY_MB = 65_536
TIMEOUT_SECONDS = 14_400
MAX_WORKERS_PER_NODE = 1
PRIORITY = 95
PROJECT_ACTIVE_TASK_CAP = 500
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
PRIMARY_WINDING_TEMPERATURES = (
    "T_max_Tx",
    "Tprobe_Tx_leeward_max",
)
SECONDARY_WINDING_TEMPERATURES = (
    "T_max_Rx_main",
    "T_max_Rx_side",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
)
CORE_TEMPERATURES = (
    "T_max_core",
    "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)
THERMAL_RX_INTERFACE_CONTRACT_FIELDS = (
    thermal_truth.THERMAL_RX_INTERFACE_CONTRACT_FIELDS
)
THERMAL_RX_INTERFACE_CONTRACT_VERSIONS = (
    thermal_truth.THERMAL_RX_INTERFACE_CONTRACT_VERSIONS
)
THERMAL_INVALID_LOG_MARKERS = thermal_truth.THERMAL_INVALID_LOG_MARKERS
THERMAL_LIMITER_FAIL_CLOSE_K = thermal_truth.THERMAL_LIMITER_FAIL_CLOSE_K
THERMAL_LIMITER_FAIL_CLOSE_C = thermal_truth.THERMAL_LIMITER_FAIL_CLOSE_C
N1_6_COOLER_ANCHOR_PREFIXES = (
    "7fae822212dfea8a",
    "45e20fbd810f9d24",
    "a12e33b8d064b244",
)
N1_6_COOLER_MODEL_TARGETS = (
    "Llt_phys",
    "C_tx_tx_F",
    "C_rx_rx_F",
    "C_tx_rx_F",
    "P_winding_total",
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_total",
    "P_core_total",
    "T_max_Tx",
    "Tprobe_Tx_leeward_max",
    "T_max_Rx_main",
    "T_max_Rx_side",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
    "T_max_core",
    "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)


class BatchContractError(RuntimeError):
    """The targeted search, candidate, or FEA batch contract drifted."""


def _builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _builtin(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise BatchContractError("payload is already sealed")
    result["payload_sha256"] = _sha(result)
    return result


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BatchContractError(f"{schema} payload is not an object")
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != _sha(unsigned):
        raise BatchContractError(f"{schema} seal mismatch")
    return dict(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchContractError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BatchContractError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Any, *, replace: bool = False) -> Path:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise BatchContractError(f"output already exists: {destination}")
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(
                    _builtin(value),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, destination)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    return destination


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise BatchContractError(f"regular file required: {resolved}")
    display = (
        str(resolved.relative_to(relative_to.resolve(strict=True)))
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": display,
        "sha256": _sha_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchContractError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise BatchContractError(f"{label} is not finite")
    return number


def _truth(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise BatchContractError(f"{label} is not a canonical boolean")
    return normalized == "true"


def _json_cell(value: Any, label: str, expected: type) -> Any:
    if not isinstance(value, str):
        raise BatchContractError(f"{label} is not a JSON cell")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise BatchContractError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, expected):
        raise BatchContractError(f"{label} has the wrong JSON type")
    return parsed


def _base_profile() -> dict[str, Any]:
    profile = _read_json(PROFILE_PATH)
    overrides = profile.get("param_overrides")
    boundary = profile.get("fixed_boundary_contract")
    if (
        _sha(profile) != PROFILE_CANONICAL_SHA256
        or profile.get("stage") != "standard"
        or profile.get("cpus") != CPUS
        or profile.get("timeout_seconds") != TIMEOUT_SECONDS
        or not isinstance(overrides, dict)
        or overrides.get("full_model") != 0
        or overrides.get("round_corner") != 0
        or overrides.get("loss_sym_on") != 1
        or overrides.get("thermal_symmetry") != "eighth"
        or overrides.get("fan_velocity") != 1.5
        or overrides.get("fan_config") != "dual"
        or overrides.get("core_plate_pad_t") != 2.0
        or overrides.get("wcp_pad_t") != 2.0
        or boundary
        != {
            "thermal_pad_conductivity_W_mK": 0.2,
            "core_plate_pad_t_mm": 2.0,
            "wcp_pad_t_mm": 2.0,
            "fan_velocity_m_s": 1.5,
            "fan_config": "dual",
            "core_plate_on": 1,
            "wcp_on": 1,
        }
    ):
        raise BatchContractError("goal Standard profile drifted")
    return profile


def _profile() -> dict[str, Any]:
    """Return the reviewed direct-analyze pre-gap acquisition profile."""

    profile = copy.deepcopy(_base_profile())
    profile["comment"] = (
        "Pre-gap eighth-symmetry acquisition: Matrix+Cap+Loss+Thermal, "
        "direct Icepak analyze, retained AEDT"
    )
    profile["reviewed_solver_path"] = (
        "run_simulation_260706.py --fixed --thermal --headless "
        "--symmetry-thermal-direct-analyze"
    )
    profile["cli_flags"] = (
        "--thermal --headless --symmetry-thermal-direct-analyze"
    )
    profile["param_overrides"]["loss_from_copy"] = 0
    profile["param_overrides"]["thermal_on"] = 1
    # A modified retained-AEDT profile is intentionally rejected by the
    # Scheduler client.  Acquisition tasks therefore keep only RESULT_JSON;
    # final symmetric/full AEDT retention remains a later gapped-design stage.
    profile.pop("artifact_retention", None)
    if (
        profile["param_overrides"].get("full_model") != 0
        or profile["param_overrides"].get("round_corner") != 0
        or profile["param_overrides"].get("thermal_symmetry") != "eighth"
        or profile["param_overrides"].get("loss_from_copy") != 0
        or "--symmetry-thermal-direct-analyze" not in profile["cli_flags"]
    ):
        raise BatchContractError("pre-gap direct acquisition profile drifted")
    return profile


def _authority(
    authority_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = authority_root.resolve(strict=True)
    status_path = root / "collector_status.json"
    pareto_path = root / "global_pareto_manifest.json"
    status = _validate_seal(
        _read_json(status_path), TARGETED_STATUS_SCHEMA
    )
    pareto = _validate_seal(
        _read_json(pareto_path), TARGETED_PARETO_SCHEMA
    )
    if (
        status.get("campaign_id") != TARGETED_CAMPAIGN_ID
        or status.get("hard_spec_sha256") != EXPECTED_HARD_SPEC_SHA256
        or status.get("fixed_lm2mh_resonance_contract_sha256")
        != EXPECTED_RESONANCE_CONTRACT_SHA256
        or status.get("expected_seed_count") != 16
        or status.get("successful_terminal_seed_count") != 16
        or status.get("raw_terminal_row_count") != 5120
        or status.get("expected_raw_terminal_row_count") != 5120
        or status.get("global_nds_final") is not True
        or status.get("final_files_written") is not True
        or status.get("classification") != "screening-only"
        or status.get("production_eligible") is not False
        or status.get("pareto_manifest_payload_sha256")
        != pareto["payload_sha256"]
    ):
        raise BatchContractError("targeted collector is not final/authenticated")
    if (
        pareto.get("campaign_id") != TARGETED_CAMPAIGN_ID
        or pareto.get("source_seed_count") != 16
        or pareto.get("source_raw_terminal_row_count") != 5120
        or pareto.get("global_non_dominated_sorting_complete") is not True
        or pareto.get("hard_spec_sha256") != EXPECTED_HARD_SPEC_SHA256
        or pareto.get("fixed_lm2mh_resonance_contract_sha256")
        != EXPECTED_RESONANCE_CONTRACT_SHA256
        or pareto.get("screening_only") is not True
        or pareto.get("production_eligible") is not False
    ):
        raise BatchContractError("targeted Pareto authority drifted")
    record = (pareto.get("files") or {}).get("fea_acquisition_candidates.csv")
    if not isinstance(record, dict):
        raise BatchContractError("FEA acquisition candidate record is absent")
    csv_path = Path(str(record.get("path") or "")).resolve(strict=True)
    try:
        csv_path.relative_to(root)
    except ValueError as exc:
        raise BatchContractError("candidate CSV escapes authority root") from exc
    if (
        _sha_file(csv_path) != record.get("sha256")
        or csv_path.stat().st_size <= 0
        or int(record.get("row_count") or -1)
        != int(pareto.get("fea_acquisition_candidate_count") or -2)
    ):
        raise BatchContractError("FEA acquisition candidate CSV identity drifted")
    return status, pareto, csv_path


def _geometry_sha(decoded: Mapping[str, Any]) -> str:
    geometry = {
        name: _builtin(decoded[name])
        for name in preflight.DECODED_GEOMETRY_IDENTITY_COLUMNS
    }
    return _sha(geometry)


def _fixed_lm_resonance(
    *,
    llt_uH: float,
    c_tx_F: float,
    c_rx_F: float,
    c_inter_F: float,
    n1: int,
    n2: int,
) -> dict[str, float]:
    if n1 <= 0 or n2 <= 0:
        raise BatchContractError("turns must be positive")
    values = (llt_uH, c_tx_F, c_rx_F, c_inter_F)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise BatchContractError("fixed-Lm resonance inputs must be positive")
    llt_h = llt_uH * 1e-6
    ltx_h = LM_TARGET_H + llt_h
    ratio = n2 / n1
    lrx_h = ltx_h * ratio * ratio
    f_tx = 1.0 / (2.0 * math.pi * math.sqrt(ltx_h * c_tx_F))
    f_rx = 1.0 / (2.0 * math.pi * math.sqrt(lrx_h * c_rx_F))
    f_inter = 1.0 / (2.0 * math.pi * math.sqrt(llt_h * c_inter_F))
    return {
        "Lm_primary_referred_H": LM_TARGET_H,
        "Llt_phys_uH": llt_uH,
        "Ltx_H": ltx_h,
        "turns_ratio_N2_over_N1": ratio,
        "Lrx_H": lrx_h,
        "fTx_Hz": f_tx,
        "fRx_Hz": f_rx,
        "fInter_Hz_diagnostic": f_inter,
        "fmin_Hz": min(f_tx, f_rx),
    }


def _candidate(
    row: Mapping[str, Any],
    row_number: int,
    *,
    require_final_track_gate: bool = True,
) -> dict[str, Any] | None:
    label = f"candidate row {row_number}"
    required = {
        "physical_geometry_sha256",
        "coordinate_unit_json",
        "decoded_physical_params_json",
        "physical_G_fixed_lm2mh_json",
        "fixed_lm2mh_resonance_contract_sha256",
        "pred_Llt_phys_uH_fixed_lm2mh",
        "pred_C_tx_tx_F_fixed_lm2mh",
        "pred_C_rx_rx_F_fixed_lm2mh",
        "pred_C_tx_rx_F_fixed_lm2mh",
        "fTx_Hz_fixed_lm2mh",
        "fRx_Hz_fixed_lm2mh",
        "resonance_min_Hz_fixed_lm2mh",
        "exterior_W_drawing_x_mm",
        "exterior_L_perpendicular_y_mm",
        "exterior_H_mm",
        "normalized_constraint_violation_l2",
        "objective_volume_L",
        "objective_total_loss_W",
        "conditional_nonthermal_feasible_diagnostic",
        "scheduler_task_id",
        "source_seed",
        "fixed_primary_turns_stratum",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise BatchContractError(f"{label} columns missing: {missing}")
    identity = str(row["physical_geometry_sha256"])
    if not HEX64.fullmatch(identity):
        raise BatchContractError(f"{label} geometry SHA is invalid")
    decoded = _json_cell(
        row["decoded_physical_params_json"],
        f"{label}.decoded params",
        dict,
    )
    coordinate = [
        _finite(value, f"{label}.coordinate")
        for value in _json_cell(
            row["coordinate_unit_json"], f"{label}.coordinate", list
        )
    ]
    if (
        len(coordinate) != 25
        or any(value < 0.0 or value > 1.0 for value in coordinate)
        or any(
            key not in decoded
            for key in preflight.DECODED_GEOMETRY_IDENTITY_COLUMNS
        )
        or _geometry_sha(decoded) != identity
    ):
        raise BatchContractError(f"{label} decoded geometry identity drifted")
    n1 = int(_finite(decoded.get("N1"), f"{label}.N1"))
    n2 = int(_finite(decoded.get("N2"), f"{label}.N2"))
    if (
        n1 != int(_finite(row["fixed_primary_turns_stratum"], f"{label}.stratum"))
        or not math.isclose(
            _finite(decoded.get("cw1"), f"{label}.cw1"),
            PRIMARY_CONDUCTOR_MM,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(decoded.get("gap1"), f"{label}.gap1"),
            PRIMARY_GAP_MM,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(_finite(decoded.get("round_corner"), f"{label}.round_corner"))
        != 0
        or row["fixed_lm2mh_resonance_contract_sha256"]
        != EXPECTED_RESONANCE_CONTRACT_SHA256
    ):
        raise BatchContractError(f"{label} fixed manufacturing contract drifted")
    _volume, observed = geometry_metrics.bounding_box_lit(decoded)
    reported = (
        _finite(row["exterior_W_drawing_x_mm"], f"{label}.W"),
        _finite(row["exterior_L_perpendicular_y_mm"], f"{label}.L"),
        _finite(row["exterior_H_mm"], f"{label}.H"),
    )
    if any(
        not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-6)
        for actual, expected in zip(observed, reported)
    ):
        raise BatchContractError(f"{label} axis-specific dimensions drifted")
    # No swap is attempted: W is always drawing-x and L is perpendicular-y.
    if any(
        value > SIZE_LIMITS_MM[axis] + 1e-9
        for axis, value in zip(("W", "L", "H"), reported)
    ):
        return None
    replay = _fixed_lm_resonance(
        llt_uH=_finite(
            row["pred_Llt_phys_uH_fixed_lm2mh"], f"{label}.Llt"
        ),
        c_tx_F=_finite(
            row["pred_C_tx_tx_F_fixed_lm2mh"], f"{label}.C_tx"
        ),
        c_rx_F=_finite(
            row["pred_C_rx_rx_F_fixed_lm2mh"], f"{label}.C_rx"
        ),
        c_inter_F=_finite(
            row["pred_C_tx_rx_F_fixed_lm2mh"], f"{label}.C_inter"
        ),
        n1=n1,
        n2=n2,
    )
    for source, replay_key in (
        ("fTx_Hz_fixed_lm2mh", "fTx_Hz"),
        ("fRx_Hz_fixed_lm2mh", "fRx_Hz"),
        ("resonance_min_Hz_fixed_lm2mh", "fmin_Hz"),
    ):
        if not math.isclose(
            _finite(row[source], f"{label}.{source}"),
            replay[replay_key],
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise BatchContractError(f"{label} fixed-Lm replay drifted: {source}")
    physical_g = _json_cell(
        row["physical_G_fixed_lm2mh_json"],
        f"{label}.physical G",
        dict,
    )
    if not math.isclose(
        _finite(
            physical_g.get("fixed_Lm2mH_self_resonance_minimum"),
            f"{label}.resonance G",
        ),
        RESONANCE_MIN_HZ - replay["fmin_Hz"],
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise BatchContractError(f"{label} physical resonance G drifted")
    conditional = _truth(
        row["conditional_nonthermal_feasible_diagnostic"],
        f"{label}.conditional nonthermal",
    )
    if require_final_track_gate and (
        replay["fmin_Hz"] < RESONANCE_MIN_HZ or not conditional
    ):
        return None
    # The authenticated search predates additive fixed-input controls such as
    # ``core_center_gap_mm`` and the opt-in turn-graded electrostatic switches.
    # They are non-geometry defaults.  Normalize through the current input
    # schema rather than rejecting an otherwise identical historical geometry.
    from module.input_parameter_260706 import create_input_parameter

    normalized = create_input_parameter(
        {
            key: _builtin(decoded[key])
            for key in ALL_INPUT_KEYS
            if key in decoded
        }
    ).iloc[0].to_dict()
    params = {key: _builtin(normalized[key]) for key in sorted(ALL_INPUT_KEYS)}
    profile = _profile()
    effective = scheduler_client.effective_verification_params(params, profile)
    if (
        effective.get("full_model") != 0
        or effective.get("round_corner") != 0
        or effective.get("thermal_symmetry") != "eighth"
        or effective.get("fan_velocity") != 1.5
        or effective.get("fan_config") != "dual"
        or effective.get("core_plate_pad_t") != 2.0
        or effective.get("wcp_pad_t") != 2.0
    ):
        raise BatchContractError(f"{label} effective Standard profile drifted")
    return {
        "physical_geometry_sha256": identity,
        "coordinate": coordinate,
        "params": params,
        "params_sha256": _sha(params),
        "effective_params_sha256": _sha(effective),
        "source": {
            "scheduler_task_id": int(
                _finite(row["scheduler_task_id"], f"{label}.task")
            ),
            "seed": int(_finite(row["source_seed"], f"{label}.seed")),
            "fixed_primary_turns_stratum": n1,
        },
        "dimensions_mm": {
            "W_drawing_x": reported[0],
            "L_perpendicular_y": reported[1],
            "H": reported[2],
        },
        "objective_volume_L": _finite(
            row["objective_volume_L"], f"{label}.volume"
        ),
        "objective_total_loss_W": _finite(
            row["objective_total_loss_W"], f"{label}.loss"
        ),
        "normalized_constraint_violation_l2": _finite(
            row["normalized_constraint_violation_l2"],
            f"{label}.violation",
        ),
        "fixed_lm2mh_screen": replay,
        "conditional_nonthermal_feasible": conditional,
        "screening_winding_max_C": _finite(
            row.get("winding_robust_max_C_screening"),
            f"{label}.winding temperature",
        ),
        "screening_core_max_C": _finite(
            row.get("core_robust_max_C_screening"),
            f"{label}.core temperature",
        ),
        "raw_row_sha256": _sha(dict(row)),
        "acquisition_rank": int(
            _finite(row.get("acquisition_rank"), f"{label}.acquisition rank")
        ),
    }


def _generate_neighborhood(
    anchors: Sequence[dict[str, Any]], count: int
) -> list[tuple[str, dict[str, Any]]]:
    """Create a deterministic capacitance-reduction neighborhood.

    Secondary pitch, main/side split, axial height, and plate length are moved
    together.  Every point is sent through the production decoder/geometry
    checks before the authenticated capacitance models are evaluated.
    """

    import numpy as np
    import pandas as pd

    from module.input_parameter_260706 import (
        create_input_parameter,
        validation_check,
    )
    from tools import mft_goal_fixed_lm2mh_rescore as rescore

    if not MIN_BATCH <= count <= MAX_BATCH:
        raise BatchContractError("batch count must be within 8..16")
    unique_anchors: list[dict[str, Any]] = []
    seen_anchor_params = set()
    for anchor in sorted(
        anchors,
        key=lambda item: (
            -item["fixed_lm2mh_screen"]["fmin_Hz"],
            item["normalized_constraint_violation_l2"],
        ),
    ):
        identity = anchor["params_sha256"]
        if identity not in seen_anchor_params:
            unique_anchors.append(anchor)
            seen_anchor_params.add(identity)
    if not unique_anchors:
        raise BatchContractError("no authenticated acquisition anchors")

    rng = random.Random(2_707_270_427)
    raw: list[dict[str, Any]] = []
    seen_params: set[str] = set()
    for anchor_index, anchor in enumerate(unique_anchors[:6]):
        attempts = 1800 if anchor_index == 0 else 180
        base = anchor["params"]
        for _ in range(attempts):
            params = dict(base)
            h_delta = rng.choice(
                [0, -2, -4, -6, -8, -10, -14, -18, -22, -30]
            )
            params["h1"] = round(float(base["h1"]) + h_delta, 1)
            params["l1"] = round(float(base["l1"]) - h_delta / 2.0, 1)
            n2 = int(base["N2_main"]) + int(base["N2_side"])
            split_delta = rng.choice([-6, -4, -2, 0, 2, 4, 6])
            params["N2_main"] = int(base["N2_main"]) + split_delta
            params["N2_side"] = n2 - int(params["N2_main"])
            factor = rng.choice(
                [0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88,
                 0.90, 0.92, 0.94, 0.96, 0.98, 1.0]
            )
            params["cw2"] = round(
                max(0.5, float(base["cw2"]) * factor), 3
            )
            params["gap2"] = round(
                float(base["gap2"])
                + (float(base["cw2"]) - float(params["cw2"]))
                + rng.choice([-0.02, -0.01, 0.0, 0.005, 0.01,
                              0.02, 0.03, 0.05, 0.08]),
                3,
            )
            if not 0.3 <= float(params["gap2"]) <= 2.0:
                continue
            params["nwh2"] = round(
                max(
                    float(params["h1"]) * 0.5,
                    min(
                        float(params["h1"]) * 0.9,
                        float(base["nwh2"])
                        * rng.choice([0.92, 0.96, 1.0, 1.04]),
                    ),
                ),
                1,
            )
            params["wcp_len_x"] = round(
                float(base["wcp_len_x"])
                * rng.choice([0.90, 0.95, 1.0, 1.05, 1.10]),
                1,
            )
            for key in (
                "cc_w2c_space_x",
                "w2c_w1c_space_x",
                "w1c_w2s_space_x",
                "w1s_cs_space_x",
            ):
                params[key] = round(
                    float(base[key]) + rng.choice([-2.0, 0.0, 2.0]), 1
                )
            try:
                valid, decoded_frame = validation_check(
                    create_input_parameter(params),
                    strict=False,
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not valid:
                continue
            decoded = {
                key: _builtin(value)
                for key, value in decoded_frame.iloc[0].to_dict().items()
            }
            try:
                volume, dimensions = geometry_metrics.bounding_box_lit(decoded)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if any(
                float(value) > SIZE_LIMITS_MM[axis] + 1e-9
                for axis, value in zip(("W", "L", "H"), dimensions)
            ):
                continue
            projected = {
                key: _builtin(decoded[key]) for key in sorted(ALL_INPUT_KEYS)
            }
            params_sha = _sha(projected)
            if params_sha in seen_params:
                continue
            seen_params.add(params_sha)
            raw.append(
                {
                    "decoded": decoded,
                    "params": projected,
                    "params_sha256": params_sha,
                    "geometry_sha256": _geometry_sha(decoded),
                    "dimensions": [float(value) for value in dimensions],
                    "volume_L": float(volume),
                    "anchor": anchor,
                    "mutations": {
                        "h1_delta_mm": h_delta,
                        "N2_main_delta_turns": split_delta,
                        "cw2_factor": factor,
                        "gap2_delta_from_pitch_preserving_mm": (
                            float(params["gap2"])
                            - float(base["gap2"])
                            - (
                                float(base["cw2"])
                                - float(params["cw2"])
                            )
                        ),
                        "nwh2_mm": float(params["nwh2"]),
                        "wcp_len_x_mm": float(params["wcp_len_x"]),
                    },
                }
            )
    if len(raw) < count:
        raise BatchContractError("decoded neighborhood is too small")
    models, model_identity = rescore._authenticate_and_load_models(
        rescore.GENERATION
    )
    frame = pd.DataFrame([item["decoded"] for item in raw])
    predictions = {}
    for target, model in models.items():
        mean, _half_width = model.predict_mu_sigma(frame, conformal=True)
        predictions[target] = np.asarray(mean, dtype=float).reshape(-1)
    eligible: list[dict[str, Any]] = []
    profile = _profile()
    for index, item in enumerate(raw):
        decoded = item["decoded"]
        replay = _fixed_lm_resonance(
            llt_uH=float(predictions["Llt_phys"][index]),
            c_tx_F=float(predictions["C_tx_tx_F"][index]),
            c_rx_F=float(predictions["C_rx_rx_F"][index]),
            c_inter_F=float(predictions["C_tx_rx_F"][index]),
            n1=int(decoded["N1"]),
            n2=int(decoded["N2"]),
        )
        if replay["fmin_Hz"] < RESONANCE_MIN_HZ:
            continue
        anchor = item["anchor"]
        effective = scheduler_client.effective_verification_params(
            item["params"], profile
        )
        if (
            effective.get("full_model") != 0
            or effective.get("round_corner") != 0
            or effective.get("thermal_symmetry") != "eighth"
            or effective.get("loss_from_copy") != 0
            or effective.get("fan_velocity") != 1.5
            or effective.get("core_plate_pad_t") != 2.0
            or effective.get("wcp_pad_t") != 2.0
        ):
            raise BatchContractError("generated effective profile drifted")
        vector = [
            float(decoded[name])
            for name in (
                "cw2",
                "gap2",
                "l1",
                "h1",
                "w1",
                "nwh2",
                "wcp_len_x",
                "N2_main",
                "N2_side",
            )
        ] + item["dimensions"] + [replay["fmin_Hz"]]
        eligible.append(
            {
                "physical_geometry_sha256": item["geometry_sha256"],
                "params": item["params"],
                "params_sha256": item["params_sha256"],
                "effective_params_sha256": _sha(effective),
                "source": {
                    **anchor["source"],
                    "anchor_physical_geometry_sha256": anchor[
                        "physical_geometry_sha256"
                    ],
                    "anchor_acquisition_rank": anchor["acquisition_rank"],
                },
                "dimensions_mm": {
                    "W_drawing_x": item["dimensions"][0],
                    "L_perpendicular_y": item["dimensions"][1],
                    "H": item["dimensions"][2],
                },
                "objective_volume_L": item["volume_L"],
                "objective_total_loss_W": anchor[
                    "objective_total_loss_W"
                ],
                "objective_total_loss_source": (
                    "anchor-surrogate-diagnostic-only; fresh Loss FEA required"
                ),
                "normalized_constraint_violation_l2": anchor[
                    "normalized_constraint_violation_l2"
                ],
                "fixed_lm2mh_screen": replay,
                "screening_winding_max_C": anchor[
                    "screening_winding_max_C"
                ],
                "screening_core_max_C": anchor[
                    "screening_core_max_C"
                ],
                "neighborhood_mutations": item["mutations"],
                "raw_row_sha256": _sha(
                    {
                        "anchor": anchor["raw_row_sha256"],
                        "params": item["params_sha256"],
                        "replay": replay,
                    }
                ),
                "_vector": vector,
            }
        )
    if len(eligible) < count:
        raise BatchContractError(
            f"only {len(eligible)} generated neighbors pass fixed-Lm fmin"
        )
    vector_width = len(eligible[0]["_vector"])
    minima = [
        min(row["_vector"][index] for row in eligible)
        for index in range(vector_width)
    ]
    spans = [
        max(row["_vector"][index] for row in eligible) - minima[index]
        for index in range(vector_width)
    ]
    for row in eligible:
        row["coordinate"] = [
            0.0
            if spans[index] <= 0.0
            else (row["_vector"][index] - minima[index]) / spans[index]
            for index in range(vector_width)
        ]
    selected: list[tuple[str, dict[str, Any]]] = []
    seen = set()

    def add(role: str, row: dict[str, Any]) -> None:
        identity = row["physical_geometry_sha256"]
        if identity not in seen and len(selected) < count:
            selected.append((role, row))
            seen.add(identity)

    add(
        "maximum_predicted_fixed_Lm_resonance_margin",
        max(eligible, key=lambda row: row["fixed_lm2mh_screen"]["fmin_Hz"]),
    )
    add(
        "minimum_secondary_conductor_thickness",
        min(eligible, key=lambda row: row["params"]["cw2"]),
    )
    add(
        "maximum_secondary_conductor_thickness",
        max(eligible, key=lambda row: row["params"]["cw2"]),
    )
    add(
        "minimum_volume_generated_neighbor",
        min(eligible, key=lambda row: row["objective_volume_L"]),
    )
    while len(selected) < count:
        remaining = [
            row
            for row in eligible
            if row["physical_geometry_sha256"] not in seen
        ]
        choice = max(
            remaining,
            key=lambda row: min(
                math.sqrt(
                    sum(
                        (left - right) ** 2
                        for left, right in zip(
                            row["coordinate"], prior["coordinate"]
                        )
                    )
                )
                for _, prior in selected
            ),
        )
        add("generated_neighborhood_max_min_diversity", choice)
    for _role, row in selected:
        row.pop("_vector", None)
        row["surrogate_model_identity"] = model_identity
    return selected


def _n1_6_cooler_predictions(
    frame: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Predict sequentially from the authenticated surrogate generation.

    Each ensemble is released before the next target is loaded.  The complete
    generation is large enough that retaining all target ensembles at once can
    consume tens of gigabytes without improving inference correctness.
    """

    import gc
    import numpy as np
    from tools import mft_goal_fixed_lm2mh_rescore as rescore
    from predictor import EnsemblePredictor

    generation = rescore.GENERATION.resolve(strict=True)
    report = rescore._read_json(generation / "train_report.json")
    artifact_hashes = report.get("artifacts")
    if (
        report.get("dataset_sha256") != rescore.EXPECTED_DATASET_SHA256
        or not isinstance(artifact_hashes, dict)
        or not set(N1_6_COOLER_MODEL_TARGETS).issubset(
            set(report.get("targets") or [])
        )
    ):
        raise BatchContractError("N1=6 cooler surrogate generation drifted")
    active = {"generation": str(generation), "report": report}
    means: dict[str, Any] = {}
    upper: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for target in N1_6_COOLER_MODEL_TARGETS:
        records = {}
        for name in ("models.pkl", "meta.json"):
            relative = f"{target}/{name}"
            path = (generation / relative).resolve(strict=True)
            observed = _sha_file(path)
            if artifact_hashes.get(relative) != observed:
                raise BatchContractError(
                    f"N1=6 cooler model artifact drifted: {relative}"
                )
            records[name] = {"path": str(path), "sha256": observed}
        model = EnsemblePredictor._load_record(target, active)
        binding = model.configure_inference_threads(threads=8)
        mean, half_width = model.predict_mu_sigma(frame, conformal=True)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        half_width = np.asarray(half_width, dtype=float).reshape(-1)
        if (
            mean.shape != (len(frame),)
            or half_width.shape != (len(frame),)
            or not np.isfinite(mean).all()
            or not np.isfinite(half_width).all()
            or np.any(half_width < 0.0)
        ):
            raise BatchContractError(
                f"N1=6 cooler surrogate output is invalid: {target}"
            )
        means[target] = mean
        upper[target] = mean + half_width
        evidence[target] = {
            "artifacts": records,
            "features_sha256": _sha(list(model.features)),
            "inference_binding": binding,
        }
        del model
        gc.collect()
    return means, upper, {
        "generation": str(generation),
        "train_report_sha256": rescore.EXPECTED_TRAIN_REPORT_SHA256,
        "dataset_sha256": rescore.EXPECTED_DATASET_SHA256,
        "evaluation_model_sha256": rescore.EXPECTED_EVALUATION_MODEL_SHA256,
        "targets": evidence,
    }


def _turn_graded_crx_calibration() -> tuple[float, dict[str, Any]]:
    """Authenticate the first corrected turn-graded Rx truth calibration.

    The legacy surrogate was trained against a lumped terminal-voltage
    electrostatic setup.  The reviewed production setup grades each turn by
    its physical series voltage.  Until enough corrected samples exist for a
    retrain, the single measured ratio is used only to rank acquisition FEA
    candidates.  Every submitted lane still runs the corrected Cap stage and
    cannot become a final PASS from this transfer estimate.
    """

    source_plan = _validate_seal(
        _read_json(TURN_GRADED_CAP_CALIBRATION_PLAN),
        BATCH_PLAN_SCHEMA,
    )
    if (
        source_plan.get("payload_sha256")
        != EXPECTED_TURN_GRADED_CAP_CALIBRATION_PLAN_PAYLOAD_SHA256
    ):
        raise BatchContractError("turn-graded calibration source plan drifted")
    source_candidates = [
        lane["candidate"]
        for lane in source_plan.get("lanes") or []
        if (lane.get("candidate") or {}).get("physical_geometry_sha256")
        == TURN_GRADED_CAP_CALIBRATION_GEOMETRY_SHA256
    ]
    if len(source_candidates) != 1:
        raise BatchContractError(
            "turn-graded calibration source geometry is not unique"
        )
    legacy_replay = source_candidates[0].get("fixed_lm2mh_screen") or {}
    legacy_lrx_H = _finite(
        legacy_replay.get("Lrx_H"), "turn_graded_calibration.Lrx_H"
    )
    legacy_frx_Hz = _finite(
        legacy_replay.get("fRx_Hz"), "turn_graded_calibration.fRx_Hz"
    )
    legacy_crx_F = 1.0 / (
        4.0 * math.pi * math.pi * legacy_lrx_H * legacy_frx_Hz**2
    )

    canary_plan_path = TURN_GRADED_CAP_CANARY_ROOT / "batch_plan.json"
    canary_plan = _validate_seal(
        _read_json(canary_plan_path),
        TURN_GRADED_CAP_PLAN_SCHEMA,
    )
    if (
        canary_plan.get("payload_sha256")
        != EXPECTED_TURN_GRADED_CAP_PLAN_PAYLOAD_SHA256
        or (canary_plan.get("source") or {}).get("physical_geometry_sha256")
        != TURN_GRADED_CAP_CALIBRATION_GEOMETRY_SHA256
        or canary_plan.get("selected_variant_ids") != ["rx-main-side-mid"]
        or canary_plan.get("symmetric_nonrounded") is not True
    ):
        raise BatchContractError("turn-graded calibration canary plan drifted")

    collection_path = (
        TURN_GRADED_CAP_CANARY_ROOT / "collection_completed_v2.json"
    )
    collection = _validate_seal(
        _read_json(collection_path),
        TURN_GRADED_CAP_COLLECTION_SCHEMA,
    )
    rows = collection.get("rows") or []
    if (
        collection.get("payload_sha256")
        != EXPECTED_TURN_GRADED_CAP_COLLECTION_PAYLOAD_SHA256
        or collection.get("plan_payload_sha256")
        != canary_plan["payload_sha256"]
        or collection.get("all_terminal") is not True
        or collection.get("valid_result_count") != 1
        or len(rows) != 1
        or rows[0].get("contract_valid") is not True
        or (rows[0].get("variant") or {}).get("id")
        != "rx-main-side-mid"
    ):
        raise BatchContractError(
            "turn-graded calibration collection is not valid"
        )
    corrected_crx_F = _finite(
        rows[0].get("C_terminal_F"),
        "turn_graded_calibration.C_terminal_F",
    )
    ratio = corrected_crx_F / legacy_crx_F
    if (
        not 0.70 <= ratio <= 0.85
        or not math.isclose(corrected_crx_F, 4.14885e-10, abs_tol=5e-16)
    ):
        raise BatchContractError("turn-graded Crx calibration ratio drifted")
    return ratio, {
        "status": "single_truth_acquisition_transfer_not_final_validation",
        "application": (
            "multiply legacy C_rx_rx_F mean only; preserve legacy Ctx/Cinter; "
            "fresh corrected Cap FEA required on every selected lane"
        ),
        "calibration_geometry_sha256": (
            TURN_GRADED_CAP_CALIBRATION_GEOMETRY_SHA256
        ),
        "legacy_Crx_F": legacy_crx_F,
        "corrected_turn_graded_Crx_F": corrected_crx_F,
        "corrected_to_legacy_ratio": ratio,
        "source_plan": _file_record(TURN_GRADED_CAP_CALIBRATION_PLAN),
        "canary_plan": _file_record(canary_plan_path),
        "canary_collection": _file_record(collection_path),
        "canary_task_id": int(rows[0]["task_id"]),
    }


def _generate_n1_6_cooler_neighborhood(
    anchors: Sequence[dict[str, Any]], count: int
) -> list[tuple[str, dict[str, Any]]]:
    """Build a deterministic N1=6/N2=60 capacitance/thermal hedge.

    The secondary foil pitch stays close to each authenticated anchor while
    copper thickness is exchanged for insulation gap and shorter axial foil
    height.  The available Rx temperature margin is therefore spent directly
    on reducing terminal capacitance.  Primary foil height and the unchanged
    production cold plates are enlarged together to target the 100 C Tx gate.
    """

    import pandas as pd

    from module.input_parameter_260706 import (
        create_input_parameter,
        validation_check,
    )

    if not MIN_BATCH <= count <= MAX_BATCH:
        raise BatchContractError("N1=6 cooler batch count must be within 8..16")
    exact = []
    by_prefix = {
        item["physical_geometry_sha256"][:16]: item for item in anchors
    }
    for prefix in N1_6_COOLER_ANCHOR_PREFIXES:
        anchor = by_prefix.get(prefix)
        if anchor is None:
            raise BatchContractError(
                f"N1=6 cooler anchor is absent: {prefix}"
            )
        if (
            int(anchor["source"]["fixed_primary_turns_stratum"]) != 6
            or (
                int(anchor["params"]["N1_main"])
                + 2 * int(anchor["params"]["N1_side"])
            )
            != 6
            or (
                int(anchor["params"]["N2_main"])
                + int(anchor["params"]["N2_side"])
            )
            != 60
        ):
            raise BatchContractError("N1=6 cooler anchor topology drifted")
        exact.append(anchor)

    raw: list[dict[str, Any]] = []
    seen_params: set[str] = set()
    secondary_cw_mm = (0.74, 0.78, 0.82, 0.86, 0.90)
    secondary_height_mm = (225.0, 245.0, 265.0)
    vertical_extension_mm = (25.0, 50.0, 60.0)
    l2_reduction_mm = (0.0, 4.0, 8.0)
    cooling_level_specs = (
        (0.0, 0.78, 1.0),
        (4.0, 0.79, 2.0),
        (8.0, 0.795, 3.0),
        (12.0, 0.80, 4.0),
    )
    for anchor in exact:
        base = anchor["params"]
        base_pitch = float(base["cw2"]) + float(base["gap2"])
        for cw2 in secondary_cw_mm:
            gap2 = round(min(2.0, base_pitch - cw2), 3)
            if gap2 < 1.75:
                continue
            nwl2_main = (
                int(base["N2_main"]) * cw2
                + (int(base["N2_main"]) - 1) * gap2
            )
            wcp_reference_x = (
                2.0 * float(base["l1"])
                + 2.0 * float(base["cc_w2c_space_x"])
                + 2.0 * nwl2_main
                + 2.0 * float(base["w2c_w1c_space_x"])
            )
            cooling_levels = tuple(
                {
                    "wcp_t": round(
                        float(base["wcp_t"]) + wcp_t_delta, 1
                    ),
                    "wcp_len_x": round(
                        wcp_reference_x * wcp_length_fraction, 1
                    ),
                    "core_plate_t": round(
                        float(base["core_plate_t"]) + core_plate_delta,
                        1,
                    ),
                }
                for (
                    wcp_t_delta,
                    wcp_length_fraction,
                    core_plate_delta,
                ) in cooling_level_specs
            )
            for nwh2 in secondary_height_mm:
                for vertical_delta in vertical_extension_mm:
                    for l2_delta in l2_reduction_mm:
                        for cooling in cooling_levels:
                            params = dict(base)
                            params.update(cooling)
                            params.update(
                                {
                                    "cw1": PRIMARY_CONDUCTOR_MM,
                                    "gap1": PRIMARY_GAP_MM,
                                    "cw2": cw2,
                                    "gap2": gap2,
                                    "nwh2": nwh2,
                                    "h1": round(
                                        float(base["h1"]) + vertical_delta, 1
                                    ),
                                    "nwh1": round(
                                        float(base["nwh1"]) + vertical_delta, 1
                                    ),
                                    "l2": round(
                                        float(base["l2"]) - l2_delta, 1
                                    ),
                                }
                            )
                            try:
                                valid, decoded_frame = validation_check(
                                    create_input_parameter(params),
                                    strict=False,
                                )
                            except (
                                KeyError,
                                TypeError,
                                ValueError,
                                OverflowError,
                            ):
                                continue
                            if not valid:
                                continue
                            decoded = {
                                key: _builtin(value)
                                for key, value in decoded_frame.iloc[
                                    0
                                ].to_dict().items()
                            }
                            if (
                                int(decoded["N1"]) != 6
                                or int(decoded["N2"]) != 60
                                or float(decoded["cw1"])
                                != PRIMARY_CONDUCTOR_MM
                                or float(decoded["gap1"]) != PRIMARY_GAP_MM
                            ):
                                raise BatchContractError(
                                    "N1=6 cooler topology/fixed controls escaped"
                                )
                            volume, dimensions = (
                                geometry_metrics.bounding_box_lit(decoded)
                            )
                            if any(
                                float(value)
                                > SIZE_LIMITS_MM[axis] + 1e-9
                                for axis, value in zip(
                                    ("W", "L", "H"), dimensions
                                )
                            ):
                                continue
                            projected = {
                                key: _builtin(decoded[key])
                                for key in sorted(ALL_INPUT_KEYS)
                            }
                            params_sha = _sha(projected)
                            if params_sha in seen_params:
                                continue
                            seen_params.add(params_sha)
                            raw.append(
                                {
                                    "decoded": decoded,
                                    "params": projected,
                                    "params_sha256": params_sha,
                                    "geometry_sha256": _geometry_sha(decoded),
                                    "dimensions": [
                                        float(value) for value in dimensions
                                    ],
                                    "volume_L": float(volume),
                                    "anchor": anchor,
                                    "mutations": {
                                        "cw2_mm": cw2,
                                        "gap2_mm": gap2,
                                        "secondary_pitch_mm": cw2 + gap2,
                                        "nwh2_mm": nwh2,
                                        "h1_and_nwh1_extension_mm": (
                                            vertical_delta
                                        ),
                                        "l2_reduction_mm": l2_delta,
                                        **cooling,
                                    },
                                }
                            )
    if len(raw) < count:
        raise BatchContractError("N1=6 cooler decoded pool is too small")

    frame = pd.DataFrame([item["decoded"] for item in raw])
    means, upper, model_identity = _n1_6_cooler_predictions(frame)
    crx_calibration_ratio, crx_calibration = (
        _turn_graded_crx_calibration()
    )

    profile = _profile()
    eligible: list[dict[str, Any]] = []
    screened: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        decoded = item["decoded"]
        legacy_c_rx_F = float(means["C_rx_rx_F"][index])
        corrected_crx_estimate_F = (
            legacy_c_rx_F * crx_calibration_ratio
        )
        legacy_replay = _fixed_lm_resonance(
            llt_uH=float(means["Llt_phys"][index]),
            c_tx_F=float(means["C_tx_tx_F"][index]),
            c_rx_F=legacy_c_rx_F,
            c_inter_F=float(means["C_tx_rx_F"][index]),
            n1=6,
            n2=60,
        )
        replay = _fixed_lm_resonance(
            llt_uH=float(means["Llt_phys"][index]),
            c_tx_F=float(means["C_tx_tx_F"][index]),
            c_rx_F=corrected_crx_estimate_F,
            c_inter_F=float(means["C_tx_rx_F"][index]),
            n1=6,
            n2=60,
        )
        primary_C = max(
            float(upper[target][index])
            for target in PRIMARY_WINDING_TEMPERATURES
        )
        secondary_C = max(
            float(upper[target][index])
            for target in SECONDARY_WINDING_TEMPERATURES
        )
        core_C = max(
            float(upper[target][index]) for target in CORE_TEMPERATURES
        )
        screened.append(
            {
                "fmin_Hz": float(replay["fmin_Hz"]),
                "corrected_Crx_estimate_F": corrected_crx_estimate_F,
                "legacy_Crx_surrogate_F": legacy_c_rx_F,
                "legacy_fmin_Hz": float(legacy_replay["fmin_Hz"]),
                "mutations": item["mutations"],
                "dimensions": item["dimensions"],
            }
        )
        if (
            replay["fmin_Hz"] < RESONANCE_MIN_HZ
            or corrected_crx_estimate_F > 0.55e-9
        ):
            continue
        effective = scheduler_client.effective_verification_params(
            item["params"], profile
        )
        if (
            effective.get("full_model") != 0
            or effective.get("round_corner") != 0
            or effective.get("thermal_symmetry") != "eighth"
            or effective.get("loss_from_copy") != 0
            or effective.get("fan_velocity") != 1.5
            or effective.get("fan_config") != "dual"
            or effective.get("core_plate_pad_t") != 2.0
            or effective.get("wcp_pad_t") != 2.0
        ):
            raise BatchContractError("N1=6 cooler effective profile drifted")
        loss = (
            float(means["P_winding_total"][index])
            + float(means["P_core_total"][index])
        )
        violation = math.sqrt(
            max(0.0, (primary_C - 100.0) / 10.0) ** 2
            + max(0.0, (secondary_C - 120.0) / 10.0) ** 2
            + max(0.0, (core_C - 120.0) / 10.0) ** 2
            + max(
                0.0,
                (corrected_crx_estimate_F - 0.55e-9) / 0.05e-9,
            )
            ** 2
        )
        screen = {
            **replay,
            "C_tx_tx_F": float(means["C_tx_tx_F"][index]),
            "C_rx_rx_F": corrected_crx_estimate_F,
            "C_rx_rx_corrected_turn_graded_transfer_estimate_F": (
                corrected_crx_estimate_F
            ),
            "C_rx_rx_legacy_surrogate_F": legacy_c_rx_F,
            "C_tx_rx_F": float(means["C_tx_rx_F"][index]),
            "legacy_fixed_lm2mh_screen": legacy_replay,
            "primary_winding_robust_max_C": primary_C,
            "secondary_winding_robust_max_C": secondary_C,
            "core_robust_max_C": core_C,
            "P_Tx_main_group_W": float(
                means["P_Tx_main_group"][index]
            ),
            "P_Rx_main_group_W": float(
                means["P_Rx_main_group"][index]
            ),
            "P_Rx_side_total_W": float(
                means["P_Rx_side_total"][index]
            ),
            "P_winding_total_W": float(
                means["P_winding_total"][index]
            ),
            "P_core_total_W": float(means["P_core_total"][index]),
            "P_total_screen_W": loss,
            "Crx_target_F": 0.55e-9,
            "Crx_selection_ceiling_F": 0.55e-9,
            "Crx_selection_basis": (
                "authenticated_single_truth_corrected_turn_graded_transfer"
            ),
        }
        vector = [
            float(decoded[name])
            for name in (
                "cw2",
                "gap2",
                "nwh2",
                "h1",
                "nwh1",
                "l2",
                "wcp_t",
                "wcp_len_x",
                "core_plate_t",
            )
        ]
        eligible.append(
            {
                "physical_geometry_sha256": item["geometry_sha256"],
                "params": item["params"],
                "params_sha256": item["params_sha256"],
                "effective_params_sha256": _sha(effective),
                "source": {
                    **item["anchor"]["source"],
                    "anchor_physical_geometry_sha256": item["anchor"][
                        "physical_geometry_sha256"
                    ],
                    "anchor_acquisition_rank": item["anchor"][
                        "acquisition_rank"
                    ],
                },
                "dimensions_mm": {
                    "W_drawing_x": item["dimensions"][0],
                    "L_perpendicular_y": item["dimensions"][1],
                    "H": item["dimensions"][2],
                },
                "objective_volume_L": item["volume_L"],
                "objective_total_loss_W": loss,
                "normalized_constraint_violation_l2": violation,
                "fixed_lm2mh_screen": replay,
                "surrogate_screen": screen,
                "turn_graded_Crx_calibration": crx_calibration,
                "screening_primary_winding_max_C": primary_C,
                "screening_secondary_winding_max_C": secondary_C,
                "screening_winding_max_C": max(primary_C, secondary_C),
                "screening_core_max_C": core_C,
                "neighborhood_mutations": item["mutations"],
                "raw_row_sha256": _sha(
                    {
                        "anchor": item["anchor"]["raw_row_sha256"],
                        "params": item["params_sha256"],
                        "screen": screen,
                    }
                ),
                "_vector": vector,
            }
        )
    if len(eligible) < count:
        minimum_crx = min(
            screened,
            key=lambda row: (
                row["corrected_Crx_estimate_F"],
                -row["fmin_Hz"],
            ),
        )
        maximum_fmin = max(
            screened,
            key=lambda row: (
                row["fmin_Hz"],
                -row["C_rx_rx_F"],
            ),
        )
        raise BatchContractError(
            "only "
            f"{len(eligible)} N1=6 cooler candidates pass "
            "calibrated corrected-turn-graded Crx<=0.55nF and "
            "fixed-Lm fmin>=15kHz; "
            f"minimum_Crx={minimum_crx!r}; maximum_fmin={maximum_fmin!r}"
        )

    vector_width = len(eligible[0]["_vector"])
    minima = [
        min(row["_vector"][index] for row in eligible)
        for index in range(vector_width)
    ]
    spans = [
        max(row["_vector"][index] for row in eligible) - minima[index]
        for index in range(vector_width)
    ]
    for row in eligible:
        row["coordinate"] = [
            0.0
            if spans[index] <= 0.0
            else (row["_vector"][index] - minima[index]) / spans[index]
            for index in range(vector_width)
        ]

    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    def add(role: str, row: dict[str, Any]) -> None:
        identity = row["physical_geometry_sha256"]
        if identity not in seen and len(selected) < count:
            selected.append((role, row))
            seen.add(identity)

    ordered = sorted(
        eligible,
        key=lambda row: (
            row["normalized_constraint_violation_l2"],
            row["screening_primary_winding_max_C"],
            row["objective_total_loss_W"],
            row["physical_geometry_sha256"],
        ),
    )
    add("minimum_split_temperature_violation", ordered[0])
    add(
        "minimum_predicted_Tx",
        min(eligible, key=lambda row: row["screening_primary_winding_max_C"]),
    )
    add(
        "minimum_predicted_Crx",
        min(
            eligible,
            key=lambda row: row["surrogate_screen"][
                "C_rx_rx_corrected_turn_graded_transfer_estimate_F"
            ],
        ),
    )
    add(
        "minimum_predicted_total_loss",
        min(eligible, key=lambda row: row["objective_total_loss_W"]),
    )
    add(
        "minimum_volume",
        min(eligible, key=lambda row: row["objective_volume_L"]),
    )
    high_wcp = [
        row
        for row in eligible
        if float(row["neighborhood_mutations"]["wcp_t"]) >= 24.0
    ]
    if high_wcp:
        add(
            "minimum_violation_high_wcp_ge24mm",
            min(
                high_wcp,
                key=lambda row: (
                    row["normalized_constraint_violation_l2"],
                    row["screening_primary_winding_max_C"],
                    row["physical_geometry_sha256"],
                ),
            ),
        )
    maximum_wcp_t = max(
        float(row["neighborhood_mutations"]["wcp_t"])
        for row in eligible
    )
    add(
        "maximum_wcp_thickness_minimum_violation",
        min(
            (
                row
                for row in eligible
                if float(row["neighborhood_mutations"]["wcp_t"])
                == maximum_wcp_t
            ),
            key=lambda row: (
                row["normalized_constraint_violation_l2"],
                row["screening_primary_winding_max_C"],
                row["physical_geometry_sha256"],
            ),
        ),
    )
    top = ordered[: max(96, count * 8)]
    while len(selected) < count:
        remaining = [
            row
            for row in top
            if row["physical_geometry_sha256"] not in seen
        ]
        if not remaining:
            remaining = [
                row
                for row in eligible
                if row["physical_geometry_sha256"] not in seen
            ]
        choice = max(
            remaining,
            key=lambda row: (
                min(
                    math.dist(row["coordinate"], prior["coordinate"])
                    for _, prior in selected
                ),
                -row["normalized_constraint_violation_l2"],
            ),
        )
        add("N1_6_cooler_max_min_diversity", choice)
    for _role, row in selected:
        row.pop("_vector", None)
        row["surrogate_model_identity"] = model_identity
    return selected


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("objective_volume_L", "objective_total_loss_W")
    pairs = [(float(left[key]), float(right[key])) for key in keys]
    return all(a <= b + 1e-12 for a, b in pairs) and any(
        a < b - 1e-12 for a, b in pairs
    )


def _front(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            item["objective_volume_L"],
            item["objective_total_loss_W"],
            item["physical_geometry_sha256"],
        ),
    ):
        if any(_dominates(other, row) for other in result):
            continue
        result = [other for other in result if not _dominates(row, other)]
        result.append(row)
    return result


def _select(
    rows: Sequence[dict[str, Any]], count: int
) -> list[tuple[str, dict[str, Any]]]:
    if not MIN_BATCH <= count <= MAX_BATCH:
        raise BatchContractError("batch count must be within 8..16")
    if len(rows) < count:
        raise BatchContractError(
            f"only {len(rows)} candidates pass geometry/nonthermal/fmin gates"
        )
    pareto = _front(rows)
    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    def add(role: str, row: dict[str, Any]) -> None:
        identity = row["physical_geometry_sha256"]
        if identity not in seen and len(selected) < count:
            selected.append((role, row))
            seen.add(identity)

    add(
        "least_constraint_violation",
        min(
            rows,
            key=lambda row: (
                row["normalized_constraint_violation_l2"],
                row["objective_volume_L"],
            ),
        ),
    )
    add(
        "near_pareto_minimum_volume",
        min(pareto, key=lambda row: row["objective_volume_L"]),
    )
    add(
        "near_pareto_minimum_loss",
        min(pareto, key=lambda row: row["objective_total_loss_W"]),
    )
    volume_min = min(row["objective_volume_L"] for row in pareto)
    volume_span = max(row["objective_volume_L"] for row in pareto) - volume_min
    loss_min = min(row["objective_total_loss_W"] for row in pareto)
    loss_span = max(row["objective_total_loss_W"] for row in pareto) - loss_min
    add(
        "near_pareto_balanced_knee",
        min(
            pareto,
            key=lambda row: math.hypot(
                0.0
                if volume_span <= 0
                else (row["objective_volume_L"] - volume_min) / volume_span,
                0.0
                if loss_span <= 0
                else (row["objective_total_loss_W"] - loss_min) / loss_span,
            ),
        ),
    )
    add(
        "maximum_fixed_lm_resonance_margin",
        max(rows, key=lambda row: row["fixed_lm2mh_screen"]["fmin_Hz"]),
    )
    for turns in (5, 6, 7, 8):
        stratum = [
            row
            for row in rows
            if row["source"]["fixed_primary_turns_stratum"] == turns
        ]
        if stratum:
            add(
                f"N1_{turns}_least_violation",
                min(
                    stratum,
                    key=lambda row: (
                        row["normalized_constraint_violation_l2"],
                        row["objective_volume_L"],
                    ),
                ),
            )
    while len(selected) < count:
        remaining = [
            row
            for row in rows
            if row["physical_geometry_sha256"] not in seen
        ]
        choice = max(
            remaining,
            key=lambda row: (
                min(
                    math.sqrt(
                        sum(
                            (a - b) ** 2
                            for a, b in zip(
                                row["coordinate"], prior["coordinate"]
                            )
                        )
                    )
                    for _, prior in selected
                ),
                -row["normalized_constraint_violation_l2"],
            ),
        )
        add("coordinate_farthest_point_fill", choice)
    return selected


def _core_environment(solver_revision: str) -> dict[str, str]:
    contract = "mft-standalone-core-optin-v1"
    auth = _sha(
        {
            "backend": "standalone",
            "contract_version": contract,
            "requested_num_cores": CPUS,
            "required_slurm_cpus_per_task": CPUS,
            "solver_revision": solver_revision,
        }
    )
    return {
        "MFT_STANDALONE_CORE_CONTRACT": contract,
        "MFT_STANDALONE_CORE_COUNT": str(CPUS),
        "MFT_STANDALONE_CORE_AUTH_SHA256": auth,
    }


def prepare(
    *,
    authority_root: Path,
    output: Path,
    count: int,
    solver_revision: str,
    library_revision: str,
    n1_6_cooler_hedge: bool = False,
) -> Path:
    solver = str(solver_revision).lower()
    library = str(library_revision).lower()
    if not HEX40.fullmatch(solver) or not HEX40.fullmatch(library):
        raise BatchContractError("full 40-character solver/library revisions required")
    status, pareto, candidate_path = _authority(authority_root)
    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()
    with candidate_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise BatchContractError("candidate CSV header is absent")
        for row_number, row in enumerate(reader, start=2):
            candidate = _candidate(
                row, row_number, require_final_track_gate=False
            )
            if candidate is None:
                continue
            identity = candidate["physical_geometry_sha256"]
            if identity in seen:
                raise BatchContractError("global candidate CSV is not deduplicated")
            anchors.append(candidate)
            seen.add(identity)
    selected = (
        _generate_n1_6_cooler_neighborhood(anchors, count)
        if n1_6_cooler_hedge
        else _generate_neighborhood(anchors, count)
    )
    profile = _profile()
    destination = output.resolve()
    if destination.exists():
        raise BatchContractError(f"batch output already exists: {destination}")
    destination.mkdir(parents=True)
    try:
        lanes = []
        for rank, (role, candidate) in enumerate(selected, start=1):
            stem = candidate["physical_geometry_sha256"][:12]
            name = f"mft-goal-lm2-neigh-r{rank:02d}-{stem}"
            workdir = f"mft_goal_lm2_neigh_r{rank:02d}_{stem}"
            params_path = _atomic_json(
                destination / "params" / f"rank-{rank:02d}-{stem}.json",
                candidate["params"],
            )
            identity = scheduler_client.verification_submission_identity(
                name,
                candidate["params"],
                profile,
                solver,
                library,
            )
            retained = scheduler_client.retained_aedt_identity(
                name,
                candidate["params"],
                profile,
                solver,
                library,
            )
            lanes.append(
                {
                    "rank": rank,
                    "selection_role": role,
                    "candidate": {
                        key: copy.deepcopy(value)
                        for key, value in candidate.items()
                        if key not in {"params", "coordinate"}
                    },
                    "neighborhood_vector_normalized": candidate["coordinate"],
                    "params": _file_record(params_path, relative_to=destination),
                    "params_sha256": candidate["params_sha256"],
                    "scheduler": {
                        "project": scheduler_client.MFT_PROJECT,
                        "name": name,
                        "workdir": workdir,
                        "dedupe_key": identity["dedupe_key"],
                        "parameter_digest": identity["parameter_digest"],
                        "effective_params_sha256": _sha(identity["merged"]),
                        "cpus": CPUS,
                        "memory_mb": MEMORY_MB,
                        "timeout_seconds": TIMEOUT_SECONDS,
                        "max_workers_per_node": MAX_WORKERS_PER_NODE,
                        "priority": PRIORITY,
                        "aedt_backend": "standalone",
                        "environment": _core_environment(solver),
                        "retained_aedt": retained,
                        "retained_aedt_required": False,
                    },
                }
            )
        profile_copy = _atomic_json(destination / "goal_standard.json", profile)
        authority = {
            "collector_status": _file_record(
                authority_root / "collector_status.json"
            ),
            "collector_status_payload_sha256": status["payload_sha256"],
            "pareto_manifest": _file_record(
                authority_root / "global_pareto_manifest.json"
            ),
            "pareto_manifest_payload_sha256": pareto["payload_sha256"],
            "global_terminal_candidates": _file_record(candidate_path),
            "source_seed_count": 16,
            "source_raw_terminal_row_count": 5120,
            "global_non_dominated_sorting_complete": True,
        }
        plan = _seal(
            {
                "schema_version": BATCH_PLAN_SCHEMA,
                "campaign_id": BATCH_CAMPAIGN_ID,
                "created_at_utc": _now(),
                "authority": authority,
                "hard_spec": {
                    "size_limits_mm": copy.deepcopy(SIZE_LIMITS_MM),
                    "axis_contract": {
                        "W": "drawing_x_original_973mm_direction",
                        "L": "drawing_y_perpendicular_direction",
                        "rotation_or_axis_swap_allowed": False,
                    },
                    "primary_conductor_thickness_mm": PRIMARY_CONDUCTOR_MM,
                    "primary_interturn_gap_mm": PRIMARY_GAP_MM,
                    "Lm_primary_referred_H": LM_TARGET_H,
                    "Lm_tuning": "physical_core_air_gap",
                    "core_center_gap_mm_in_this_batch": 0.0,
                    "self_resonance_min_Hz": RESONANCE_MIN_HZ,
                    "primary_winding_temperature_max_C": 100.0,
                    "secondary_winding_temperature_max_C": 120.0,
                    "core_temperature_max_C": 120.0,
                    "fan_velocity_m_s": 1.5,
                    "TIM_conductivity_W_mK": 0.2,
                    "TIM_thickness_mm": 2.0,
                    "topology": "eighth_symmetric",
                    "winding_geometry": "unrounded",
                    "rounded_FEA_allowed": False,
                },
                "air_gap_attestation": {
                    "target_Lm_primary_referred_H": LM_TARGET_H,
                    "fixed_Lm_resonance_screen_applied_before_selection": True,
                    "core_center_gap_mm": 0.0,
                    "current_solver_input_has_physical_core_gap_variable": False,
                    "this_batch_models_air_gap_geometry": False,
                    "measured_matrix_Lm_is_ungapped_baseline": True,
                    "physical_gap_synthesis_and_gapped_Lm_FEA_required_later": True,
                    "final_design_PASS_allowed_from_this_batch": False,
                },
                "selection": {
                    "authenticated_anchor_count": len(anchors),
                    "selected_candidate_count": len(lanes),
                    "minimum_count": MIN_BATCH,
                    "maximum_count": MAX_BATCH,
                    "fixed_Lm_fmin_gate_applied": True,
                    "nonthermal_gate_applied": True,
                    "N1_6_cooler_hedge": bool(n1_6_cooler_hedge),
                    "N1_6_cooler_anchor_prefixes": (
                        list(N1_6_COOLER_ANCHOR_PREFIXES)
                        if n1_6_cooler_hedge
                        else []
                    ),
                    "N1_6_N2_60_topology_hard_fixed": bool(
                        n1_6_cooler_hedge
                    ),
                    "Crx_target_F": (
                        0.55e-9 if n1_6_cooler_hedge else None
                    ),
                    "N1_6_Crx_selection_basis": (
                        "authenticated single corrected-turn-graded truth "
                        "transfer for acquisition ranking only; fresh Cap FEA "
                        "is authoritative"
                        if n1_6_cooler_hedge
                        else None
                    ),
                    "N1_6_final_classification_requires_raw_FEA": bool(
                        n1_6_cooler_hedge
                    ),
                    "N1_6_required_raw_FEA_evidence": (
                        [
                            "corrected turn-graded terminal capacitance",
                            "fixed-Lm=2mH resonance replay",
                            "primary split temperature <=100C",
                            "secondary split temperature <=120C",
                            "core split temperature <=120C",
                        ]
                        if n1_6_cooler_hedge
                        else []
                    ),
                    "split_temperature_limits_C": {
                        "primary_winding": 100.0,
                        "secondary_winding": 120.0,
                        "core": 120.0,
                    },
                    "diversity_method": (
                        (
                            "exact authenticated N1=6 cooler anchors -> "
                            "deterministic secondary copper-for-gap and "
                            "cold-plate/primary-height stencil -> authenticated "
                            "cap/loss/split-temperature surrogate rescore -> "
                            "single-truth corrected-turn-graded Crx transfer "
                            "for acquisition ranking -> estimated Crx<=0.55nF "
                            "and fixed-Lm fmin>=15kHz -> "
                            "minimum-violation plus max-min diversity"
                        )
                        if n1_6_cooler_hedge
                        else (
                            "authenticated NSGA acquisition anchors -> "
                            "deterministic secondary-capacitance-reduction "
                            "stencil -> fixed-Lm fmin>=15kHz -> greedy "
                            "max-min physical diversity"
                        )
                    ),
                },
                "solver_revision": solver,
                "library_revision": library,
                "profile": _file_record(profile_copy, relative_to=destination),
                "profile_canonical_sha256": _sha(profile),
                "scheduler_priority": PRIORITY,
                "lanes": lanes,
                "submission_ready": True,
                "explicit_apply_required": True,
                "automatic_submission_enabled": False,
                "scheduler_repository_modified": False,
                "scheduler_project_configuration_modified": False,
                "scheduler_submission_performed": False,
                "classification": "symmetric-unrounded-FEA-acquisition-only",
                "hard_label": (
                    "core_center_gap_mm=0; pre-gap acquisition only; "
                    "Lm=2mH is not physically attested"
                ),
                "production_eligible": False,
            }
        )
        return _atomic_json(destination / "batch_plan.json", plan)
    except BaseException:
        # Preserve a partially written directory for forensics; preparation is
        # local and immutable outputs make a second run choose another path.
        raise


def _load_plan(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    plan_path = path.resolve(strict=True)
    root = plan_path.parent
    plan = _validate_seal(_read_json(plan_path), BATCH_PLAN_SCHEMA)
    scheduler_priority = plan.get("scheduler_priority", PRIORITY)
    if (
        plan.get("campaign_id") != BATCH_CAMPAIGN_ID
        or plan.get("submission_ready") is not True
        or plan.get("explicit_apply_required") is not True
        or plan.get("automatic_submission_enabled") is not False
        or plan.get("scheduler_repository_modified") is not False
        or plan.get("scheduler_project_configuration_modified") is not False
        or plan.get("scheduler_submission_performed") is not False
        or plan.get("classification")
        != "symmetric-unrounded-FEA-acquisition-only"
        or plan.get("hard_label")
        != (
            "core_center_gap_mm=0; pre-gap acquisition only; "
            "Lm=2mH is not physically attested"
        )
        or plan.get("production_eligible") is not False
        or not MIN_BATCH <= len(plan.get("lanes") or []) <= MAX_BATCH
        or isinstance(scheduler_priority, bool)
        or not isinstance(scheduler_priority, int)
        or not 0 <= scheduler_priority <= 100
    ):
        raise BatchContractError("batch plan contract drifted")
    profile_record = plan.get("profile") or {}
    profile_path = (root / str(profile_record.get("path") or "")).resolve(
        strict=True
    )
    try:
        profile_path.relative_to(root)
    except ValueError as exc:
        raise BatchContractError("profile escapes batch root") from exc
    profile = _read_json(profile_path)
    if (
        _file_record(profile_path, relative_to=root) != profile_record
        or _sha(profile) != plan.get("profile_canonical_sha256")
        or profile != _profile()
    ):
        raise BatchContractError("batch profile bytes drifted")
    authority_root = Path(
        plan["authority"]["collector_status"]["path"]
    ).resolve(strict=True).parent
    status, pareto, candidate_path = _authority(authority_root)
    if (
        status["payload_sha256"]
        != plan["authority"]["collector_status_payload_sha256"]
        or pareto["payload_sha256"]
        != plan["authority"]["pareto_manifest_payload_sha256"]
        or _file_record(candidate_path)
        != plan["authority"]["global_terminal_candidates"]
    ):
        raise BatchContractError("fresh targeted authority differs from plan")
    identities = set()
    for lane in plan["lanes"]:
        params_path = (root / lane["params"]["path"]).resolve(strict=True)
        params = _read_json(params_path)
        scheduler = lane["scheduler"]
        replay = scheduler_client.verification_submission_identity(
            scheduler["name"],
            params,
            profile,
            plan["solver_revision"],
            plan["library_revision"],
        )
        if (
            _file_record(params_path, relative_to=root) != lane["params"]
            or _sha(params) != lane["params_sha256"]
            or replay["dedupe_key"] != scheduler["dedupe_key"]
            or replay["parameter_digest"] != scheduler["parameter_digest"]
            or _sha(replay["merged"]) != scheduler["effective_params_sha256"]
            or scheduler["project"] != scheduler_client.MFT_PROJECT
            or scheduler["cpus"] != CPUS
            or scheduler["memory_mb"] != MEMORY_MB
            or scheduler["timeout_seconds"] != TIMEOUT_SECONDS
            or scheduler["max_workers_per_node"] != MAX_WORKERS_PER_NODE
            or scheduler["priority"] != scheduler_priority
            or scheduler["environment"]
            != _core_environment(plan["solver_revision"])
            or scheduler["name"] in identities
            or scheduler["dedupe_key"] in identities
        ):
            raise BatchContractError("batch lane execution identity drifted")
        identities.update({scheduler["name"], scheduler["dedupe_key"]})
    return plan, root, profile


def dry_run(*, plan_path: Path, output: Path) -> Path:
    plan, _root, _profile_value = _load_plan(plan_path)
    value = _seal(
        {
            "schema_version": DRY_RUN_SCHEMA,
            "created_at_utc": _now(),
            "plan_payload_sha256": plan["payload_sha256"],
            "candidate_count": len(plan["lanes"]),
            "all_source_contracts_replayed": True,
            "all_scheduler_identities_replayed": True,
            "fixed_Lm_fmin_gate_replayed": True,
            "axis_swap_used": False,
            "symmetric_model": True,
            "rounded_winding_used": False,
            "fan_velocity_m_s": 1.5,
            "TIM_mutated": False,
            "scheduler_POST_performed": False,
            "ready_for_explicit_apply": True,
            "classification": "dry-run",
        }
    )
    return _atomic_json(output, value)


def submit(
    *,
    plan_path: Path,
    output: Path,
    scheduler_url: str,
    apply: bool,
    submitter: Callable[..., Any] = scheduler_client.submit_verification,
) -> Path:
    plan, root, profile = _load_plan(plan_path)
    if not apply:
        return dry_run(plan_path=plan_path, output=output)
    if output.exists():
        raise BatchContractError(f"submission output already exists: {output}")
    submitted = []
    try:
        for lane in plan["lanes"]:
            params = _read_json(root / lane["params"]["path"])
            scheduler = lane["scheduler"]
            evidence = submitter(
                scheduler["name"],
                scheduler["workdir"],
                params,
                profile,
                mem_mb=scheduler["memory_mb"],
                cpus=scheduler["cpus"],
                solver_revision=plan["solver_revision"],
                library_revision=plan["library_revision"],
                priority=scheduler["priority"],
                max_workers_per_node=scheduler["max_workers_per_node"],
                aedt_backend="standalone",
                submission_env=scheduler["environment"],
                required_hard_cap=PROJECT_ACTIVE_TASK_CAP,
                return_submission_evidence=True,
                max_project_active_tasks=PROJECT_ACTIVE_TASK_CAP,
                scheduler_url=scheduler_url,
            )
            if not isinstance(evidence, dict):
                raise BatchContractError("Scheduler submission evidence is absent")
            readback = (
                evidence.get("api_pre_submission_readback")
                or evidence.get("api_post_submission_response")
                or evidence
            )
            if isinstance(readback, dict) and isinstance(
                readback.get("task"), dict
            ):
                readback = readback["task"]
            if not isinstance(readback, dict):
                raise BatchContractError(
                    "Scheduler submission readback is absent"
                )
            task_id = int(
                evidence.get(
                    "task_id",
                    evidence.get(
                        "id", readback.get("id", readback.get("task_id", 0))
                    ),
                )
            )
            if (
                task_id <= 0
                or readback.get("name") != scheduler["name"]
                or readback.get("dedupe_key") != scheduler["dedupe_key"]
                or str(readback.get("project") or "")
                != scheduler_client.MFT_PROJECT
                or int(readback.get("cpus") or 0) != scheduler["cpus"]
                or int(readback.get("memory_mb") or 0)
                != scheduler["memory_mb"]
                or int(readback.get("timeout_seconds") or 0)
                != scheduler["timeout_seconds"]
            ):
                raise BatchContractError("Scheduler submission readback drifted")
            submitted.append(
                {
                    "rank": lane["rank"],
                    "physical_geometry_sha256": lane["candidate"][
                        "physical_geometry_sha256"
                    ],
                    "task_id": task_id,
                    "name": scheduler["name"],
                    "dedupe_key": scheduler["dedupe_key"],
                    "readback": readback,
                    "submission_evidence": evidence,
                }
            )
    except BaseException as exc:
        failure = _seal(
            {
                "schema_version": SUBMISSION_SCHEMA,
                "created_at_utc": _now(),
                "plan": _file_record(plan_path),
                "plan_payload_sha256": plan["payload_sha256"],
                "scheduler_url": scheduler_url.rstrip("/"),
                "requested_candidate_count": len(plan["lanes"]),
                "submitted_candidate_count": len(submitted),
                "submissions": submitted,
                "complete": False,
                "failure": f"{type(exc).__name__}: {exc}",
                "scheduler_repository_modified": False,
            }
        )
        _atomic_json(output, failure)
        raise
    receipt = _seal(
        {
            "schema_version": SUBMISSION_SCHEMA,
            "created_at_utc": _now(),
            "plan": _file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "scheduler_url": scheduler_url.rstrip("/"),
            "requested_candidate_count": len(plan["lanes"]),
            "submitted_candidate_count": len(submitted),
            "submissions": submitted,
            "complete": True,
            "all_tasks_are_independent": True,
            "parallel_execution_requested": True,
            "scheduler_repository_modified": False,
        }
    )
    return _atomic_json(output, receipt)


def _api_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        value = json.loads(response.read().decode("utf-8"))
    # The dedicated stdout endpoint returns a JSON string, while task
    # metadata endpoints return an object.  Normalize both so collection can
    # parse the final RESULT_JSON without requesting a truncated task body.
    if isinstance(value, str):
        return {"stdout": value}
    if not isinstance(value, dict):
        raise BatchContractError("Scheduler returned a non-object")
    return value


def _result_json(task: Mapping[str, Any]) -> dict[str, Any] | None:
    result = task.get("result_json")
    if isinstance(result, dict):
        return dict(result)
    output = str(task.get("stdout") or task.get("output") or "")
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT_JSON=") or line.startswith("RESULT_JSON "):
            try:
                if line.startswith("RESULT_JSON="):
                    encoded = line.split("=", 1)[1]
                else:
                    encoded = line.split(" ", 1)[1]
                parsed = json.loads(encoded)
            except (IndexError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _optional_finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _available_numeric(
    result: Mapping[str, Any],
    names: Iterable[str],
) -> dict[str, float]:
    values = {}
    for name in names:
        value = _optional_finite(result.get(name))
        if value is not None:
            values[name] = value
    return values


def _thermal_scientific_truth_contract(
    result: Mapping[str, Any],
    *,
    observed_temperatures_C: Mapping[str, float],
    task_log_text: str,
) -> dict[str, Any]:
    """Fail-close legacy or interface-isolated thermal acquisition results."""
    return thermal_truth.thermal_scientific_truth_contract(
        result,
        observed_temperatures_C=observed_temperatures_C,
        task_log_text=task_log_text,
    )


def _measured_result(
    result: Mapping[str, Any],
    lane: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    profile: Mapping[str, Any],
    task_log_text: str = "",
) -> dict[str, Any]:
    identity_reasons = []

    def equal(name: str, expected: Any) -> None:
        if result.get(name) != expected:
            identity_reasons.append(f"result_identity_mismatch:{name}")

    equal("full_model", 0)
    equal("round_corner", 0)
    equal("thermal_symmetry", "eighth")
    equal("loss_sym_on", 1)
    if not math.isclose(
        _finite(result.get("fan_velocity"), "result fan velocity"),
        1.5,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        identity_reasons.append("fixed_boundary_mismatch:fan_velocity")
    for name in ("core_plate_pad_t", "wcp_pad_t"):
        if not math.isclose(
            _finite(result.get(name), f"result {name}"),
            2.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            identity_reasons.append(f"fixed_boundary_mismatch:{name}")
    if result.get("fan_config") != "dual":
        identity_reasons.append("fixed_boundary_mismatch:fan_config")

    params = _read_json(Path(lane["_plan_root"]) / lane["params"]["path"])
    effective_params = scheduler_client.effective_verification_params(
        params, dict(profile)
    )
    strict_solver_result_valid = scheduler_client.is_valid_result(
        dict(result),
        expected_revision=plan["solver_revision"],
        expected_library_revision=plan["library_revision"],
        expected_profile=dict(profile.get("param_overrides") or {}),
    )
    exact_params_echo_valid = scheduler_client.result_matches_params(
        dict(result), effective_params
    )
    if not strict_solver_result_valid:
        identity_reasons.append("strict_solver_result_contract_invalid")
    if not exact_params_echo_valid:
        identity_reasons.append("effective_params_echo_mismatch")

    llt = result.get("Llt_phys")
    if llt is None:
        llt = 2.0 * _finite(result.get("Llt"), "result symmetric Llt")
    n1 = int(effective_params["N1_main"]) + int(
        effective_params.get("N1_side") or 0
    )
    n2 = int(effective_params["N2_main"]) + int(
        effective_params.get("N2_side") or 0
    )
    legacy_capacitance = {
        "C_tx_tx_F": _finite(
            result.get("C_tx_tx_F"), "result C_tx_tx_F"
        ),
        "C_rx_rx_F": _finite(
            result.get("C_rx_rx_F"), "result C_rx_rx_F"
        ),
        "C_tx_rx_F": _finite(
            result.get("C_tx_rx_F"), "result C_tx_rx_F"
        ),
    }
    legacy_replay = _fixed_lm_resonance(
        llt_uH=_finite(llt, "result Llt_phys"),
        c_tx_F=legacy_capacitance["C_tx_tx_F"],
        c_rx_F=legacy_capacitance["C_rx_rx_F"],
        c_inter_F=legacy_capacitance["C_tx_rx_F"],
        n1=n1,
        n2=n2,
    )
    calibration = lane["candidate"].get("turn_graded_Crx_calibration") or {}
    transfer_ratio = _optional_finite(
        calibration.get("corrected_to_legacy_ratio")
    )
    corrected_transfer_replay = None
    corrected_transfer_capacitance = None
    if transfer_ratio is not None and 0.0 < transfer_ratio < 1.0:
        corrected_transfer_capacitance = {
            "C_tx_tx_F": legacy_capacitance["C_tx_tx_F"],
            "C_rx_rx_F": (
                legacy_capacitance["C_rx_rx_F"] * transfer_ratio
            ),
            "C_tx_rx_F": legacy_capacitance["C_tx_rx_F"],
            "C_rx_rx_corrected_to_legacy_ratio": transfer_ratio,
            "classification": (
                "single-canary transfer estimate; not actual corrected "
                "turn-graded FEA"
            ),
        }
        corrected_transfer_replay = _fixed_lm_resonance(
            llt_uH=_finite(llt, "result Llt_phys"),
            c_tx_F=corrected_transfer_capacitance["C_tx_tx_F"],
            c_rx_F=corrected_transfer_capacitance["C_rx_rx_F"],
            c_inter_F=corrected_transfer_capacitance["C_tx_rx_F"],
            n1=n1,
            n2=n2,
        )

    temperature_reasons = []
    primary_winding = {}
    for name in PRIMARY_WINDING_TEMPERATURES:
        try:
            primary_winding[name] = _finite(
                result.get(name), f"result {name}"
            )
        except BatchContractError:
            temperature_reasons.append(f"temperature_missing:{name}")
    secondary_winding = {}
    for name in SECONDARY_WINDING_TEMPERATURES:
        if name in {"T_max_Rx_side", "Tprobe_Rx_side_leeward_max"} and int(
            result.get("N2_side") or 0
        ) <= 0:
            continue
        try:
            secondary_winding[name] = _finite(
                result.get(name), f"result {name}"
            )
        except BatchContractError:
            temperature_reasons.append(f"temperature_missing:{name}")
    core = {}
    for name in CORE_TEMPERATURES:
        try:
            core[name] = _finite(result.get(name), f"result {name}")
        except BatchContractError:
            temperature_reasons.append(f"temperature_missing:{name}")
    primary_winding_max = (
        max(primary_winding.values()) if primary_winding else math.nan
    )
    secondary_winding_max = (
        max(secondary_winding.values()) if secondary_winding else math.nan
    )
    core_max = max(core.values()) if core else math.nan
    scientific_truth = _thermal_scientific_truth_contract(
        result,
        observed_temperatures_C={
            **primary_winding,
            **secondary_winding,
            **core,
        },
        task_log_text=task_log_text,
    )
    identity_reasons.extend(scientific_truth["reasons"])
    if primary_winding and primary_winding_max > 100.0:
        temperature_reasons.append(
            "primary_winding_temperature_above_100C"
        )
    if secondary_winding and secondary_winding_max > 120.0:
        temperature_reasons.append(
            "secondary_winding_temperature_above_120C"
        )
    if core and core_max > 120.0:
        temperature_reasons.append("core_temperature_above_120C")

    volume_litres, recomputed_dimensions = geometry_metrics.bounding_box_lit(
        effective_params
    )
    dimensions = {
        "W_drawing_x": float(recomputed_dimensions[0]),
        "L_perpendicular_y": float(recomputed_dimensions[1]),
        "H": float(recomputed_dimensions[2]),
    }
    size_reasons = [
        f"size_above_limit:{axis}"
        for axis, limit in SIZE_LIMITS_MM.items()
        if dimensions[
            {
                "W": "W_drawing_x",
                "L": "L_perpendicular_y",
                "H": "H",
            }[axis]
        ]
        > limit + 1e-9
    ]

    symmetric_inductance_uH = _available_numeric(
        result, ("Ltx", "Lrx", "M", "k", "Lmt", "Lmr", "Llt", "Llr")
    )
    full_physical_inductance_uH = {
        name: 2.0 * value
        for name, value in symmetric_inductance_uH.items()
        if name != "k"
    }
    if "k" in symmetric_inductance_uH:
        full_physical_inductance_uH["k"] = symmetric_inductance_uH["k"]
    measured_ungapped_primary_self_H = (
        2.0 * symmetric_inductance_uH["Ltx"] * 1e-6
        if "Ltx" in symmetric_inductance_uH
        else None
    )
    measured_ungapped_primary_magnetizing_H = (
        2.0 * symmetric_inductance_uH["Lmt"] * 1e-6
        if "Lmt" in symmetric_inductance_uH
        else None
    )
    losses_W = _available_numeric(
        result,
        (
            "P_Tx_main_group",
            "P_Rx_main_group",
            "P_Rx_side_total",
            "P_winding_total",
            "P_core_total",
            "P_core_plate_total",
            "P_wcp_total",
        ),
    )
    losses_W["P_total_thermal_input"] = sum(
        losses_W.get(name, 0.0)
        for name in (
            "P_winding_total",
            "P_core_total",
            "P_core_plate_total",
            "P_wcp_total",
        )
    )
    convergence = _available_numeric(
        result,
        (
            "conv_passes_matrix",
            "conv_consecutive_matrix",
            "conv_error_pct_matrix",
            "conv_delta_pct_matrix",
            "conv_passes_loss",
            "conv_consecutive_loss",
            "conv_error_pct_loss",
            "conv_delta_pct_loss",
            "thermal_iterations",
            "thermal_residual_continuity",
            "thermal_residual_x_velocity",
            "thermal_residual_y_velocity",
            "thermal_residual_z_velocity",
            "thermal_residual_energy",
        ),
    )
    timing = {}
    for key, raw_value in result.items():
        normalized_key = str(key).lower()
        if (
            "elapsed" not in normalized_key
            and not normalized_key.endswith("_seconds")
        ):
            continue
        parsed_value = _optional_finite(raw_value)
        if parsed_value is not None:
            timing[str(key)] = parsed_value

    result_contract_valid = not identity_reasons
    thermal_pass = not temperature_reasons
    size_pass = not size_reasons
    legacy_resonance_pass = (
        legacy_replay["fmin_Hz"] >= RESONANCE_MIN_HZ
    )
    provisional_corrected_transfer_pass = (
        corrected_transfer_replay is not None
        and corrected_transfer_replay["fmin_Hz"] >= RESONANCE_MIN_HZ
    )
    acquisition_reasons = (
        identity_reasons
        + temperature_reasons
        + size_reasons
        + (
            []
            if legacy_resonance_pass
            else ["legacy_cap_fixed_Lm_resonance_below_15kHz"]
        )
    )
    return {
        "contract_valid": not acquisition_reasons,
        "reasons": acquisition_reasons,
        "result_contract_valid": result_contract_valid,
        "result_contract_reasons": identity_reasons,
        "thermal_result_scientific_valid": scientific_truth["valid"],
        "thermal_result_scientific_reasons": scientific_truth["reasons"],
        "thermal_rx_interface_contract": scientific_truth,
        "strict_solver_result_valid": strict_solver_result_valid,
        "exact_effective_params_echo_valid": exact_params_echo_valid,
        "thermal_pass": thermal_pass,
        "thermal_reasons": temperature_reasons,
        "size_pass": size_pass,
        "size_reasons": size_reasons,
        "dimensions_mm": dimensions,
        "volume_L": float(volume_litres),
        "measured_fixed_lm2mh": legacy_replay,
        "legacy_equipotential_capacitance_F": legacy_capacitance,
        "legacy_cap_fixed_lm2mh_resonance_pass": legacy_resonance_pass,
        "corrected_turn_graded_capacitance_available": False,
        "corrected_turn_graded_capacitance_blocker": (
            "current goal_standard run uses legacy equipotential Cap; "
            "separate turn-graded terminal solves are required"
        ),
        "corrected_turn_graded_transfer_estimate_F": (
            corrected_transfer_capacitance
        ),
        "corrected_transfer_fixed_lm2mh_replay": (
            corrected_transfer_replay
        ),
        "provisional_corrected_transfer_resonance_pass": (
            provisional_corrected_transfer_pass
        ),
        "inductance_basis": {
            "physical_core_center_gap_mm": 0.0,
            "classification": (
                "measured ungapped symmetric one-eighth matrix with "
                "full-physical restoration factor 2"
            ),
            "symmetric_native_uH": symmetric_inductance_uH,
            "full_physical_restored_uH": full_physical_inductance_uH,
            "measured_ungapped_primary_self_H": (
                measured_ungapped_primary_self_H
            ),
            "measured_ungapped_primary_magnetizing_H": (
                measured_ungapped_primary_magnetizing_H
            ),
            "fixed_2mH_replay_is_not_measured_gapped_Lm": True,
        },
        "losses_W": losses_W,
        "temperatures_C": {
            "primary_winding": primary_winding,
            "secondary_winding": secondary_winding,
            "core": core,
        },
        "convergence": convergence,
        "timing_seconds": timing,
        "measured_primary_winding_max_C": primary_winding_max,
        "measured_secondary_winding_max_C": secondary_winding_max,
        "measured_core_max_C": core_max,
        "temperature_acceptance_contract": {
            "primary_winding_max_C": 100.0,
            "secondary_winding_max_C": 120.0,
            "core_max_C": 120.0,
        },
        "artifact_contract": {
            "retention_requested": (
                lane["scheduler"].get("retained_aedt") is not None
            ),
            "retained_aedt_required": bool(
                lane["scheduler"].get("retained_aedt_required")
            ),
            "retained_aedt_expected": False,
            "status": (
                "not_requested_by_goal_standard_profile; RESULT_JSON is "
                "the required acquisition artifact"
            ),
        },
        "final_design_pass": False,
        "final_design_pass_blocker": (
            "physical air-gap/gapped Lm and actual corrected turn-graded "
            "capacitance are not attested"
        ),
    }


def collect(
    *,
    plan_path: Path,
    submission_path: Path,
    output: Path,
    scheduler_url: str | None = None,
    getter: Callable[[str], dict[str, Any]] = _api_json,
) -> dict[str, Any]:
    plan, root, profile_value = _load_plan(plan_path)
    submission = _validate_seal(
        _read_json(submission_path.resolve(strict=True)), SUBMISSION_SCHEMA
    )
    if (
        submission.get("complete") is not True
        or submission.get("plan_payload_sha256") != plan["payload_sha256"]
        or submission.get("submitted_candidate_count") != len(plan["lanes"])
        or len(submission.get("submissions") or []) != len(plan["lanes"])
    ):
        raise BatchContractError("complete batch submission receipt required")
    origin = (scheduler_url or submission["scheduler_url"]).rstrip("/")
    receipt_by_rank = {
        int(row["rank"]): row for row in submission["submissions"]
    }
    rows = []
    terminal = 0
    valid = 0
    thermal_pass_count = 0
    legacy_replay_pass_count = 0
    provisional_transfer_pass_count = 0
    for lane_source in plan["lanes"]:
        lane = copy.deepcopy(lane_source)
        lane["_plan_root"] = str(root)
        receipt = receipt_by_rank[lane["rank"]]
        task_id = int(receipt["task_id"])
        task = getter(f"{origin}/api/tasks/{task_id}")
        if isinstance(task.get("task"), dict):
            task = dict(task["task"])
        if (
            int(task.get("id", task.get("task_id", -1))) != task_id
            or task.get("name") != lane["scheduler"]["name"]
            or task.get("dedupe_key") != lane["scheduler"]["dedupe_key"]
            or str(task.get("project") or "") != scheduler_client.MFT_PROJECT
        ):
            raise BatchContractError(f"Scheduler task {task_id} identity drifted")
        status = str(task.get("status") or "unknown")
        item: dict[str, Any] = {
            "rank": lane["rank"],
            "physical_geometry_sha256": lane["candidate"][
                "physical_geometry_sha256"
            ],
            "task_id": task_id,
            "status": status,
            "node": task.get("actual_node_name"),
            "result_available": False,
            "contract_valid": False,
            "result_contract_valid": False,
            "thermal_result_scientific_valid": False,
            "thermal_pass": False,
            "legacy_cap_fixed_lm2mh_resonance_pass": False,
            "provisional_corrected_transfer_resonance_pass": False,
            "final_design_pass": False,
            "dimensions_mm": copy.deepcopy(
                lane["candidate"]["dimensions_mm"]
            ),
            "volume_L": lane["candidate"]["objective_volume_L"],
            "artifact_contract": {
                "retention_requested": False,
                "retained_aedt_required": False,
                "retained_aedt_expected": False,
                "status": (
                    "not_requested_by_goal_standard_profile; no AEDT "
                    "artifact is missing from this acquisition lane"
                ),
            },
        }
        if status in {"completed", "failed", "cancelled"}:
            terminal += 1
        if status == "completed" and int(task.get("exit_code") or 0) == 0:
            result = _result_json(task)
            task_log_parts: list[str] = []
            log_query = urllib.parse.urlencode(
                {"tail_lines": "100000", "max_bytes": "5000000"}
            )
            for stream in ("stdout", "stderr"):
                try:
                    log_payload = getter(
                        f"{origin}/api/tasks/{task_id}/{stream}?{log_query}"
                    )
                except (BatchContractError, OSError, ValueError):
                    continue
                task_log_parts.append(
                    str(
                        log_payload.get(stream)
                        or log_payload.get("stdout")
                        or log_payload.get("output")
                        or ""
                    )
                )
            if result is None:
                result = _result_json(
                    {"stdout": "\n".join(task_log_parts)}
                )
            if result is not None:
                measured = _measured_result(
                    result,
                    lane,
                    plan=plan,
                    profile=profile_value,
                    task_log_text="\n".join(task_log_parts),
                )
                item.update(measured)
                item["result_available"] = True
                item["result_sha256"] = _sha(result)
                if measured["result_contract_valid"]:
                    valid += 1
                if (
                    measured["result_contract_valid"]
                    and measured["thermal_pass"]
                ):
                    thermal_pass_count += 1
                if (
                    measured["result_contract_valid"]
                    and measured[
                        "legacy_cap_fixed_lm2mh_resonance_pass"
                    ]
                ):
                    legacy_replay_pass_count += 1
                if (
                    measured["result_contract_valid"]
                    and measured["thermal_pass"]
                    and measured[
                        "provisional_corrected_transfer_resonance_pass"
                    ]
                ):
                    provisional_transfer_pass_count += 1
        rows.append(item)

    def violation(item: Mapping[str, Any]) -> float:
        if not item.get("result_contract_valid"):
            return math.inf
        primary = _optional_finite(
            item.get("measured_primary_winding_max_C")
        )
        secondary = _optional_finite(
            item.get("measured_secondary_winding_max_C")
        )
        core = _optional_finite(item.get("measured_core_max_C"))
        replay = item.get("corrected_transfer_fixed_lm2mh_replay") or {}
        fmin = _optional_finite(replay.get("fmin_Hz"))
        values = (
            (primary, 100.0),
            (secondary, 120.0),
            (core, 120.0),
        )
        score = sum(
            max(0.0, (value - limit) / limit)
            if value is not None
            else 1.0
            for value, limit in values
        )
        score += (
            max(0.0, (RESONANCE_MIN_HZ - fmin) / RESONANCE_MIN_HZ)
            if fmin is not None
            else 1.0
        )
        return score

    ranked_rows = sorted(
        (item for item in rows if item.get("result_available")),
        key=lambda item: (
            violation(item),
            _optional_finite(
                item.get("measured_primary_winding_max_C")
            )
            or math.inf,
            _optional_finite(item.get("measured_core_max_C")) or math.inf,
            _optional_finite(
                (item.get("losses_W") or {}).get(
                    "P_total_thermal_input"
                )
            )
            or math.inf,
            _optional_finite(item.get("volume_L")) or math.inf,
            int(item["rank"]),
        ),
    )
    for acquisition_rank, item in enumerate(ranked_rows, start=1):
        item["authenticated_acquisition_rank"] = acquisition_rank
        item["normalized_constraint_violation_sum"] = violation(item)

    def ranked_identity(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "authenticated_acquisition_rank": item.get(
                "authenticated_acquisition_rank"
            ),
            "source_rank": item["rank"],
            "task_id": item["task_id"],
            "physical_geometry_sha256": item[
                "physical_geometry_sha256"
            ],
            "normalized_constraint_violation_sum": item.get(
                "normalized_constraint_violation_sum"
            ),
            "primary_winding_max_C": item.get(
                "measured_primary_winding_max_C"
            ),
            "secondary_winding_max_C": item.get(
                "measured_secondary_winding_max_C"
            ),
            "core_max_C": item.get("measured_core_max_C"),
            "corrected_transfer_fmin_Hz": (
                item.get("corrected_transfer_fixed_lm2mh_replay") or {}
            ).get("fmin_Hz"),
            "classification": (
                "authenticated legacy FEA plus single-canary corrected-Crx "
                "transfer estimate; not final corrected-cap validation"
            ),
        }

    rankings = {
        "authenticated_result_contract": [
            ranked_identity(item)
            for item in ranked_rows
            if item.get("result_contract_valid")
        ],
        "authenticated_split_temperature_pass": [
            ranked_identity(item)
            for item in ranked_rows
            if item.get("result_contract_valid")
            and item.get("thermal_pass")
        ],
        "provisional_corrected_transfer_plus_thermal_pass": [
            ranked_identity(item)
            for item in ranked_rows
            if item.get("result_contract_valid")
            and item.get("thermal_pass")
            and item.get(
                "provisional_corrected_transfer_resonance_pass"
            )
        ],
        "actual_corrected_turn_graded_final_pass": [],
    }
    value = _seal(
        {
            "schema_version": COLLECTION_SCHEMA,
            "created_at_utc": _now(),
            "plan": _file_record(plan_path),
            "plan_payload_sha256": plan["payload_sha256"],
            "submission": _file_record(submission_path),
            "submission_payload_sha256": submission["payload_sha256"],
            "scheduler_url": origin,
            "candidate_count": len(rows),
            "terminal_count": terminal,
            "valid_symmetric_acquisition_count": valid,
            "authenticated_split_temperature_pass_count": (
                thermal_pass_count
            ),
            "legacy_cap_resonance_pass_count": legacy_replay_pass_count,
            "provisional_corrected_transfer_plus_thermal_pass_count": (
                provisional_transfer_pass_count
            ),
            "all_terminal": terminal == len(rows),
            "rows": rows,
            "rankings": rankings,
            "final_design_pass_count": 0,
            "physical_air_gap_attestation_pending": True,
            "actual_corrected_turn_graded_capacitance_pending": True,
            "legacy_capacitance_is_not_final_truth": True,
            "corrected_transfer_estimate_is_not_final_truth": True,
            "rounded_FEA_used": False,
            "scheduler_mutation_performed": False,
            "classification": "GET-only-symmetric-acquisition-collection",
            "production_eligible": False,
        }
    )
    _atomic_json(output, value, replace=True)
    return value


def watch_prepare(
    *,
    authority_root: Path,
    output: Path,
    count: int,
    solver_revision: str,
    library_revision: str,
    poll_seconds: float,
    n1_6_cooler_hedge: bool = False,
) -> Path:
    if poll_seconds < 10.0 or poll_seconds > 300.0:
        raise BatchContractError("poll interval must be within 10..300 seconds")
    while True:
        try:
            return prepare(
                authority_root=authority_root,
                output=output,
                count=count,
                solver_revision=solver_revision,
                library_revision=library_revision,
                n1_6_cooler_hedge=n1_6_cooler_hedge,
            )
        except (FileNotFoundError, BatchContractError) as exc:
            status_path = authority_root / "collector_status.json"
            if status_path.is_file():
                try:
                    status = _read_json(status_path)
                except BatchContractError:
                    status = {}
                if status.get("global_nds_final") is True:
                    raise
            print(
                json.dumps(
                    {
                        "state": "awaiting_final_targeted_authority",
                        "observed_at_utc": _now(),
                        "reason": str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(poll_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "watch-prepare"):
        command = commands.add_parser(name)
        command.add_argument(
            "--authority-root", type=Path, default=DEFAULT_AUTHORITY_ROOT
        )
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--count", type=int, default=DEFAULT_BATCH)
        command.add_argument("--solver-revision", required=True)
        command.add_argument("--library-revision", required=True)
        command.add_argument(
            "--n1-6-cooler-hedge",
            action="store_true",
            help=(
                "Use only the three authenticated N1=6 cooler anchors and "
                "target Crx/Tx with split 100/120/120 C limits."
            ),
        )
        if name == "watch-prepare":
            command.add_argument("--poll-seconds", type=float, default=30.0)
    submit_command = commands.add_parser("submit")
    submit_command.add_argument("--plan", type=Path, required=True)
    submit_command.add_argument("--output", type=Path, required=True)
    submit_command.add_argument("--scheduler-url", default=SCHEDULER_URL)
    submit_command.add_argument("--apply", action="store_true")
    collect_command = commands.add_parser("collect")
    collect_command.add_argument("--plan", type=Path, required=True)
    collect_command.add_argument("--submission", type=Path, required=True)
    collect_command.add_argument("--output", type=Path, required=True)
    collect_command.add_argument("--scheduler-url")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "prepare":
        result = prepare(
            authority_root=args.authority_root,
            output=args.output,
            count=args.count,
            solver_revision=args.solver_revision,
            library_revision=args.library_revision,
            n1_6_cooler_hedge=args.n1_6_cooler_hedge,
        )
    elif args.command == "watch-prepare":
        result = watch_prepare(
            authority_root=args.authority_root,
            output=args.output,
            count=args.count,
            solver_revision=args.solver_revision,
            library_revision=args.library_revision,
            poll_seconds=args.poll_seconds,
            n1_6_cooler_hedge=args.n1_6_cooler_hedge,
        )
    elif args.command == "submit":
        result = submit(
            plan_path=args.plan,
            output=args.output,
            scheduler_url=args.scheduler_url,
            apply=args.apply,
        )
    else:
        collect(
            plan_path=args.plan,
            submission_path=args.submission,
            output=args.output,
            scheduler_url=args.scheduler_url,
        )
        result = args.output.resolve()
    print(json.dumps({"status": "ok", "path": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
