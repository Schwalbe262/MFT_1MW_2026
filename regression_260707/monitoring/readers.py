"""Safe readers and view-model builders for MFT campaign artifacts.

Most readers intentionally depend only on the Python standard library.  The
campaign reader loads the lossless Parquet audit dataset lazily so the
electrostatic fields that are not present in ``train_io.csv`` remain visible.
A missing Parquet dependency, partially written file, or corrupt artifact is
still isolated from the rest of the dashboard.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

try:
    from ..model_targets import (
        SURROGATE_TEMPERATURE_TARGETS,
    )
except ImportError:  # Direct execution with regression_260707 on sys.path.
    from model_targets import (
        SURROGATE_TEMPERATURE_TARGETS,
    )

try:
    from module.core_material_contract import (
        PHYSICS_DATA_REVISION as CURRENT_PHYSICS_DATA_REVISION,
        solver_revision_matches_physics_cohort,
    )
except Exception as exc:  # The dashboard must remain available if repo imports fail.
    CURRENT_PHYSICS_DATA_REVISION: str | None = None
    solver_revision_matches_physics_cohort = None
    PHYSICS_DATA_REVISION_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    PHYSICS_DATA_REVISION_IMPORT_ERROR = None

try:
    try:
        from ..training.checkpoint_contract import (
            checkpoint_status_revision_identity_matches,
        )
    except ImportError:
        from training.checkpoint_contract import (
            checkpoint_status_revision_identity_matches,
        )
except Exception:  # Keep the dashboard fail-soft when repo imports are unavailable.
    checkpoint_status_revision_identity_matches = None

from .pipeline_status import ContinuousPipelineReader
from .deadline_design import (
    DeadlineDesignError,
    PATH_ENV as DEADLINE_DESIGN_PATH_ENV,
    SHA256_ENV as DEADLINE_DESIGN_SHA256_ENV,
    load_publication as load_deadline_design_publication,
)

try:
    from tools.tier1_final1000_multiseed_contract import (
        COMPACT_INDEX_SCHEMA as CURRENT7_COMPACT_INDEX_SCHEMA,
    )
    from tools.tier1_final1000_multiseed_monitor import (
        adapt_condition_index as adapt_compact_condition_index,
        load_compact_condition_index,
    )
except Exception as exc:  # Existing v1 pages must survive an optional adapter failure.
    CURRENT7_COMPACT_INDEX_SCHEMA = (
        "mft-tier1-final1000-compact-condition-index-v2"
    )
    adapt_compact_condition_index = None
    load_compact_condition_index = None
    COMPACT_V2_ADAPTER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    COMPACT_V2_ADAPTER_IMPORT_ERROR = None


SCHEMA_VERSION = 1
DATA_GOAL = 3_000
STRETCH_GOAL = 10_000
# A browser that loaded the pre-repair bundle cached an HTTP-200 archive
# integrity failure under the generation's immutable status timestamp.  Apply
# one stable presentation revision while preserving chronological ordering so
# those already-open clients observe a different legacy key and refetch once.
TIER1_ARCHIVED_DETAIL_CACHE_REVISION_US = 1
TIER1_ARCHIVED_DETAIL_CACHE_REPAIR_AT = "2026-07-19T10:30:00+09:00"
LEGACY_PHYSICS_DATA_REVISION = "legacy_unspecified"
THERMAL_MODEL_TAGS = (
    "isotropic_legacy",
    "anisotropic_wound_rule_of_mixtures_v1",
)
CAPACITANCE_FIELDS = {
    "tx_tx": "C_tx_tx_F",
    "rx_rx": "C_rx_rx_F",
    "tx_rx": "C_tx_rx_F",
}
CAPACITANCE_PARITY_TARGETS = frozenset(CAPACITANCE_FIELDS.values())
CAPACITANCE_RECOVERY_CONTRACT = "mft-capacitance-lc-inverse-v1"
CAPACITANCE_PARITY_ARTIFACT_ENV = "MFT_CAPACITANCE_PARITY_ARTIFACT"
CAPACITANCE_PARITY_SHA256_ENV = "MFT_CAPACITANCE_PARITY_SHA256"
CAPACITANCE_PARITY_INVALID_STATUS = "legacy_quantized_labels"
CAPACITANCE_PARITY_INVALID_MESSAGE = (
    "legacy quantized labels / 교정 재학습 대기"
)
RESONANCE_FIELDS = {
    "tx_self": "f_res_tx_self_Hz",
    "rx_self": "f_res_rx_self_Hz",
    "interwinding": "f_res_interwinding_Hz",
}
# The scheduler owns the authoritative bounds.  These values are used only
# while an older scheduler has no simulation-policy capability to advertise.
# Keep the fallback in one place so the UI never grows its own competing cap.
DEFAULT_PARALLEL_TARGET_MIN = 0
DEFAULT_PARALLEL_TARGET_MAX = 500
PARALLEL_TARGET_SAFETY_CEILING = 600
NODE_LOCAL_AEDT_PROJECT = "_aedt_pool_hosts"
NODE_LOCAL_AEDT_ENTRYPOINT = "aedt_node_canary_host"
NODE_LOCAL_AEDT_ACTIVE_STATES = ("queued", "attaching", "running")
NODE_LOCAL_AEDT_TASK_LIMIT = 1_000
DEFAULT_CONTROLLER_STATE_PATH = (
    "C:/Users/peets/slurm_scheduler_runtime/mft_controller/"
    "restart_v3_7_controller_state.json"
)
DEFAULT_CONTROLLER_LOG_PATH = (
    "C:/Users/peets/slurm_scheduler_runtime/mft_controller/"
    "controller_restart_v3_7.log"
)
CONTROLLER_LOG_TAIL_BYTES = 128 * 1024
CONTROLLER_STATE_MAX_BYTES = 8 * 1024 * 1024
SIMULATION_TIMING_WINDOW_ROWS = 100
MAPE_ZERO_ABS_TOLERANCE = 1e-9
SIMULATION_TIMING_FIELDS = (
    ("matrix", "time_matrix"),
    ("loss", "time_loss"),
    ("icepak", "time_thermal"),
    ("total", "time"),
)
CAP_TIMING_FIELDS = ("cap_solve_time_s", "cap_extraction_time_s")
CHECKPOINT_STATE_SCHEMA_VERSION = 2
CHECKPOINT_METRICS_SCHEMA_VERSION = 1
CHECKPOINT_PARITY_SCHEMA_VERSION = 1
CHECKPOINT_PARITY_ARTIFACT_TYPE = "checkpoint_cv_oof_parity"
CHECKPOINT_PARITY_PAIR_LIMIT = 2_000
ROLLING_INDEX_MAX_BYTES = 256 * 1024
ROLLING_MODEL_POINTER_MAX_BYTES = 512 * 1024
ROLLING_STATUS_MAX_BYTES = 16 * 1024 * 1024
ROLLING_SNAPSHOT_MAX_ATTEMPTS = 8
CURRENT7_INDEX_SCHEMA = "mft-tier1-current7-slurm-rolling-index-v1"
CURRENT7_STATUS_SCHEMA = "mft-tier1-current7-slurm-rolling-status-v1"
CURRENT7_AGGREGATE_SCHEMA = "mft-tier1-current7-slurm-aggregate-v1"
CURRENT7_COMPATIBILITY_SCHEMA = (
    "mft-tier1-current7-monitor-compatibility-v1"
)
CURRENT7_SNAPSHOT_IDENTITY_SCHEMA = (
    "mft-tier1-current7-snapshot-identity-v1"
)
CURRENT7_SCALE_SUMMARY_SCHEMA = "mft-tier1-current7-scale-summary-v1"
CURRENT7_INDEX_ENV = "MFT_TIER1_CURRENT7_INDEX"
CURRENT7_CONDITION_INDEXES_ENV = "MFT_TIER1_CURRENT7_CONDITION_INDEXES"
CURRENT7_CONDITION_INDEX_LIMIT = 16
# The cumulative legacy condition projection has crossed 32 MiB in production.
# Keep the same max+1 bounded read and a single, temporary 64 MiB bridge while
# the already-supported compact-v2 condition format replaces this payload.
CURRENT7_STATUS_MAX_BYTES = 64 * 1024 * 1024
CURRENT7_CANDIDATE_PREVIEW_LIMIT = 256
CURRENT7_COMPACT_DETAIL_PAGE_SIZE = 256
CURRENT7_COMPACT_DETAIL_MAX_RECORDS = 65_536
CURRENT7_COMPACT_TASK_STATES = frozenset({
    "queued", "attaching", "running", "completed",
    "completed_with_failures", "failed", "cancelled", "canceled",
    "timeout", "timed_out", "stopped", "deadline",
})
CURRENT7_ACTIVE_FRESHNESS_MAX_AGE_SECONDS = 300.0
CURRENT7_FUTURE_SKEW_TOLERANCE_SECONDS = 60.0
CURRENT7_TERMINAL_RECORD_SCHEMA = (
    "mft-tier1-current7-content-addressed-seed-record-v1"
)
CURRENT7_TASK_STATES = frozenset({
    "queued", "attaching", "running", "completed", "failed", "cancelled",
    "timeout",
})
CURRENT7_FAIL_CLOSED_FLAGS = (
    "production_eligible",
    "fea_submission_approved",
    "fea_submission_performed",
    "aedt_used",
    "automatic_promotion_allowed",
)
CURRENT7_IDENTITY_FIELDS = (
    "constraint_version",
    "hard_spec",
    "hard_spec_sha256",
    "hard_constraint_contract_sha256",
    "temperature_contract_sha256",
    "constraint_names",
    "temperature_targets",
)
BLOCKER_HPO_V2_STATUS_ENV = "MFT_BLOCKER_HPO_V2_STATUS"
BLOCKER_HPO_V2_STATUS_SCHEMA = "mft-blocker-hpo-live-status-v2"
BLOCKER_HPO_V2_STATUS_MAX_BYTES = 8 * 1024 * 1024
BLOCKER_HPO_V2_HEARTBEAT_MAX_AGE_SECONDS = 45.0


def _tier1_controller_mode_verified(
        controller: dict[str, Any], *, historical_snapshot: bool) -> bool:
    """Validate a live worker or a sealed final historical snapshot."""
    mode = controller.get("mode")
    return bool(
        mode == "rolling-worker"
        or (historical_snapshot and mode == "once")
    )


def _tier1_staleness_warning(
        *, freshness_ok: bool, historical_snapshot: bool) -> str | None:
    """Only live rolling pointers require a fresh heartbeat."""
    if freshness_ok or historical_snapshot:
        return None
    return "Tier-1 rolling status is stale"
ROLLING_SNAPSHOT_RETRY_SECONDS = 0.05
ROLLING_NEAR_FEASIBLE_PREVIEW_LIMIT = 64
ROLLING_NEAR_FEASIBLE_PER_TERMINAL_LIMIT = 3
ROLLING_CANDIDATE_PREVIEW_LIMIT = 128
TIER1_SUPPLEMENTAL_ROOTS_ENV = "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOTS"
TIER1_SUPPLEMENTAL_ROOT_LIMIT_ENV = (
    "MFT_TIER1_NSGA_SUPPLEMENTAL_ROOT_LIMIT"
)
# A deployment normally needs fewer than this many independent search roots,
# but the staged Tier-1 search now intentionally runs more than eight lanes.
# Keep a useful default while retaining a hard ceiling so an environment typo
# cannot turn one dashboard request into an unbounded artifact scan.
TIER1_SUPPLEMENTAL_ROOT_LIMIT = 16
TIER1_SUPPLEMENTAL_ROOT_LIMIT_CEILING = 32


def _configured_tier1_supplemental_root_limit() -> tuple[int, tuple[str, ...]]:
    """Return a bounded operator override without weakening the hard cap."""
    raw_limit = os.environ.get(
        TIER1_SUPPLEMENTAL_ROOT_LIMIT_ENV, ""
    ).strip()
    if not raw_limit:
        return TIER1_SUPPLEMENTAL_ROOT_LIMIT, ()
    if not re.fullmatch(r"[0-9]+", raw_limit):
        return TIER1_SUPPLEMENTAL_ROOT_LIMIT, (
            "Tier-1 supplemental root limit override is invalid; default "
            f"{TIER1_SUPPLEMENTAL_ROOT_LIMIT} was retained",
        )
    requested_limit = int(raw_limit)
    if not 1 <= requested_limit <= TIER1_SUPPLEMENTAL_ROOT_LIMIT_CEILING:
        return TIER1_SUPPLEMENTAL_ROOT_LIMIT, (
            "Tier-1 supplemental root limit override is outside the bounded "
            f"range 1..{TIER1_SUPPLEMENTAL_ROOT_LIMIT_CEILING}; default "
            f"{TIER1_SUPPLEMENTAL_ROOT_LIMIT} was retained",
        )
    return requested_limit, ()


SEALED_SUCCESSOR_HANDOFF_ENV = "MFT_T120_SEALED_SUCCESSOR_HANDOFF"
SEALED_SUCCESSOR_HANDOFF_SHA_ENV = "MFT_T120_SEALED_SUCCESSOR_HANDOFF_SHA256"
SEALED_SUCCESSOR_HANDOFF_SHA256 = (
    "be5164eb62be97175ab39a6e0e17b72f7c2a9e09f0e5a0da71c46984e81c165f"
)
SEALED_SUCCESSOR_HANDOFF_SCHEMA = "mft-t120-exact-adapter-handoff-v3"
SEALED_SUCCESSOR_COHORT_ID = "res10k-41497047d1-23821df6fa-ba31d09c3d"
SEALED_SUCCESSOR_CONSTRAINT_VERSION = (
    "mft-tier1-envelope-1200x1200x750-res10k-t120all11-core4-5t-lmhalf-v3"
)
SEALED_SUCCESSOR_REPLAY_COMMIT = "20cdb89a8002ad9ef44e59a70bfb72e9caa7197b"
DUAL_SUCCESSOR_PLAN_ENV = "MFT_T120_DUAL_SUCCESSOR_PLAN"
DUAL_SUCCESSOR_PLAN_SHA_ENV = "MFT_T120_DUAL_SUCCESSOR_PLAN_SHA256"
DUAL_SUCCESSOR_UI_ENV = "MFT_T120_DUAL_SUCCESSOR_UI"
DUAL_SUCCESSOR_UI_SHA_ENV = "MFT_T120_DUAL_SUCCESSOR_UI_SHA256"
DUAL_SUCCESSOR_RECEIPT_ENV = "MFT_T120_DUAL_SUCCESSOR_RECEIPT"
DUAL_SUCCESSOR_RECEIPT_SHA_ENV = "MFT_T120_DUAL_SUCCESSOR_RECEIPT_SHA256"
DUAL_SUCCESSOR_PLAN_SHA256 = (
    "f558322633aa4651b770ebf538970d834f2dbb2b90fb17cdaed75bec281ed20f"
)
DUAL_SUCCESSOR_UI_SHA256 = (
    "efebefe8bf97b57cd1c9ce7fe49f0cdc3edc4c158e1d330cfbda679873b329eb"
)
DUAL_SUCCESSOR_RECEIPT_SHA256 = (
    "7fecae134ec02f9fa8a423afed3d2336ccc266b37c75bef4a2136c7468b7e897"
)
DUAL_SUCCESSOR_RELEASE_REVISION = "d3ba15de3b22c9b90d528255fdc6c55bd29370f6"
DUAL_SUCCESSOR_RELEASE_SOURCE_SHA256 = (
    "2f31399991c656ed1f06fa10bd6cc7915e04ce25f56ce12f705a3f94054c9660"
)
DUAL_SUCCESSOR_EXACT_SHA256 = (
    "9375b77f0797c3c6dcd55c87594e6b35c59b6629aa295a807af4d69afed97d1b"
)
DUAL_SUCCESSOR_INTERIOR_SHA256 = (
    "be52ebce726446fc5c36f75339b03405be2a7cd009719cf9690019dd375d4498"
)
DUAL_SUCCESSOR_HARD_SPEC_SHA256 = (
    "23821df6fa595a4d9e44ba8cd05336ba524bddb8b896eb7dde389643ef833836"
)
DUAL_SUCCESSOR_TEMPERATURE_CONTRACT_SHA256 = (
    "9edde0eb85267f8cf48423b9dfdd428e09052e0c961af67792df16e23ad406e5"
)
DUAL_VALIDATION_ROOT_ENV = "MFT_T120_DUAL_FEA_RUNTIME_ROOT"
DUAL_VALIDATION_ALLOWED_ROOT_ENV = "MFT_T120_DUAL_FEA_ALLOWED_ROOT"
DUAL_VALIDATION_MANIFEST_SHA_ENV = "MFT_T120_DUAL_FEA_MANIFEST_SHA256"
DUAL_VALIDATION_PLAN_SHA_ENV = "MFT_T120_DUAL_FEA_PLAN_SHA256"
DUAL_VALIDATION_SOURCE_SHA_ENV = "MFT_T120_DUAL_FEA_SOURCE_SHA256"
DUAL_VALIDATION_DEFAULT_ALLOWED_ROOT = (
    "C:/Users/peets/slurm_scheduler_runtime"
)
DUAL_VALIDATION_MANIFEST_SHA256 = (
    "10d8785e06058361a7bfb22c29ae3cd067223942ad87aa4cefd44bd1bff6ef1e"
)
DUAL_VALIDATION_PLAN_SHA256 = (
    "b78eb6da19e3074c86e3c19693f8739bd296c01f1a9ca8a138da063cc9fe0d75"
)
DUAL_VALIDATION_SOURCE_SHA256 = (
    "bcbd70356721d47ca1c27a78d1a8f20ed7f939cd5bd517772174880477b2ee6f"
)
DUAL_VALIDATION_HANDOFF_SHA256 = (
    "15ee96fb76ccddad96b911565298fae28ddb47c3ca26d704838b6dbc4cc6639f"
)
DUAL_VALIDATION_SOLVER_REVISION = "7768510433858c9056f04320e66819d5fcc90f1a"
DUAL_VALIDATION_LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
DUAL_VALIDATION_CANDIDATE_DIGESTS = {
    DUAL_SUCCESSOR_EXACT_SHA256: (
        "d65e4cbbdb1925765c293494c2d4b7efcf490db3e00a2f3b07e45d867cef3df6"
    ),
    DUAL_SUCCESSOR_INTERIOR_SHA256: (
        "bb63894ee450a2d1cf062b2d31ef9b3f1a83982b49b1b650b5ee1669ac711bd0"
    ),
}
DUAL_VALIDATION_ACTIVE_TASK_STATUSES = frozenset(
    {"queued", "attaching", "running"}
)
DUAL_VALIDATION_TASK_STATUSES = frozenset(
    {
        "queued", "attaching", "running", "completed", "failed",
        "cancelled", "timeout",
    }
)
DUAL_VALIDATION_STATE_MAX_BYTES = 2 * 1024 * 1024
DUAL_VALIDATION_STATUS_MAX_BYTES = 512 * 1024
DUAL_VALIDATION_RESULT_MAX_BYTES = 16 * 1024 * 1024
SEALED_SUCCESSOR_TEMPERATURE_TARGETS = (
    "T_max_Tx",
    "T_max_Rx_main",
    "T_max_Rx_side",
    "T_max_core",
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
    "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)
DUAL_VALIDATION_LLT_TARGET_UH = 27.5
DUAL_VALIDATION_LLT_TOLERANCE_UH = 0.55
DUAL_VALIDATION_TEMPERATURE_LIMIT_C = 120.0
DUAL_VALIDATION_SIDE_TEMPERATURE_TARGETS = frozenset(
    {"T_max_Rx_side", "Tprobe_Rx_side_leeward_max"}
)
DUAL_VALIDATION_TEMPERATURE_CONTRACT = {
    "constraint_name_format": "temperature_robust_limit:{target}",
    "formula": "surrogate_mu_plus_q90_conformal_half_width_le_limit",
    "hard_spec_temperature_key": "T_limit_C",
    "robust_upper_bound_C": DUAL_VALIDATION_TEMPERATURE_LIMIT_C,
    "schema_version": "mft-tier1-temperature-constraint-contract-v3",
    "semantic_version": "all11-robust-q90-half-width-t120-v3",
    "side_winding_absent_behavior": (
        "finite_N2_side_eq_0_disables_with_negative_BIG"
    ),
    "side_winding_activation": "finite_N2_side_gt_0",
    "side_winding_conditional_target_count": 2,
    "side_winding_conditional_targets": [
        "T_max_Rx_side", "Tprobe_Rx_side_leeward_max",
    ],
    "side_winding_missing_behavior": (
        "missing_or_nonfinite_N2_side_fails_with_positive_BIG"
    ),
    "source": "model_targets.SURROGATE_TEMPERATURE_TARGETS",
    "target_count": len(SEALED_SUCCESSOR_TEMPERATURE_TARGETS),
    "targets": list(SEALED_SUCCESSOR_TEMPERATURE_TARGETS),
    "uncertainty_contract": "q90_conformal_half_width_physical_v1",
    "unconditional_target_count": 9,
    "unconditional_targets": [
        name for name in SEALED_SUCCESSOR_TEMPERATURE_TARGETS
        if name not in DUAL_VALIDATION_SIDE_TEMPERATURE_TARGETS
    ],
}
SEALED_SUCCESSOR_CONSTRAINTS = (
    "Llt_robust_band",
    *(f"temperature_robust_limit:{name}" for name in SEALED_SUCCESSOR_TEMPERATURE_TARGETS),
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "minimum_physical_insulation",
    "strict_full_density_support",
    "Llt_ensemble_disagreement",
    "core_group_manufacturability_limit",
    "half_magnetizing_resonance_minimum",
    "exterior_width_limit",
    "exterior_length_limit",
    "exterior_height_limit",
)
TARGETS: tuple[dict[str, str], ...] = (
    {"name": "Llt_phys", "label": "누설 인덕턴스 (Llt)", "unit": "µH"},
    {"name": "P_winding_total", "label": "권선 손실", "unit": "W"},
    {"name": "P_core_total", "label": "코어 손실", "unit": "W"},
    {"name": "P_core_plate_total", "label": "코어 플레이트 손실", "unit": "W"},
    {"name": "P_wcp_total", "label": "권선 냉각판 손실", "unit": "W"},
    {"name": "P_Tx_main_group", "label": "1차 권선 손실", "unit": "W"},
    {"name": "P_Rx_main_group", "label": "2차 중앙 권선 손실", "unit": "W"},
    {"name": "P_Rx_side_total", "label": "2차 측면 권선 손실", "unit": "W"},
    {"name": "Tprobe_Tx_leeward_max", "label": "Tx 최대 온도", "unit": "°C"},
    {"name": "Tprobe_Rx_main_leeward_max", "label": "Rx main 최대 온도", "unit": "°C"},
    {"name": "Tprobe_Rx_side_leeward_max", "label": "Rx side 최대 온도", "unit": "°C"},
    {"name": "Tprobe_core_center_max", "label": "코어 최대 온도(3영역 최대)", "unit": "°C"},
    {"name": "Tprobe_core_center_leg_max", "label": "코어 중앙 레그 최대 온도", "unit": "°C"},
    {"name": "Tprobe_core_side_leg_max", "label": "코어 사이드 레그 최대 온도", "unit": "°C"},
    {"name": "Tprobe_core_top_yoke_max", "label": "코어 상부 요크 최대 온도", "unit": "°C"},
    {"name": "k", "label": "결합계수 (k)", "unit": ""},
    {"name": "B_mean_core", "label": "코어 평균 자속밀도", "unit": "T"},
    {"name": "B_max_core", "label": "코어 최대 자속밀도", "unit": "T"},
    {"name": "C_tx_tx_F", "label": "Tx 자기 정전용량", "unit": "F"},
    {"name": "C_rx_rx_F", "label": "Rx 자기 정전용량", "unit": "F"},
    {"name": "C_tx_rx_F", "label": "Tx-Rx 상호 정전용량", "unit": "F"},
)
TARGET_META = {item["name"]: item for item in TARGETS}
TEMPERATURE_TARGETS = tuple(SURROGATE_TEMPERATURE_TARGETS)
CURRENT7_TEMPERATURE_TARGETS = TEMPERATURE_TARGETS
CURRENT7_CONSTRAINT_NAMES = (
    "Llt_robust_band",
    *(f"temperature_robust_limit:{target}" for target in CURRENT7_TEMPERATURE_TARGETS),
    "analytical_flux_density_limit",
    "decoded_space_shrink",
    "secondary_vertical_insulation",
    "strict_full_density_support",
    "Llt_ensemble_disagreement",
    "minimum_physical_insulation",
    "core_group_manufacturability_limit",
    "half_magnetizing_resonance_minimum",
    "exterior_width_limit",
    "exterior_length_limit",
    "exterior_height_limit",
)
CURRENT7_RESONANCE_MINIMUM_CONSTRAINT = (
    "half_magnetizing_resonance_minimum"
)
CURRENT7_RESONANCE_MAXIMUM_CONSTRAINT = (
    "half_magnetizing_resonance_maximum"
)
CURRENT7_CONDITION_INVARIANT_HARD_SPEC = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "q_sigma": 1.0,
    "n_core_group_max": 4,
    "primary_conductor_thickness_mm": 5.0,
    "magnetizing_inductance_factor": 0.5,
}
CURRENT7_HARD_SPEC = {
    "Llt_target_uH": 27.5,
    "Llt_tol_uH": 0.55,
    "T_limit_C": 110.0,
    "B_limit_T": 1.2,
    "insulation_min_mm": 40.0,
    "q_sigma": 1.0,
    "n_core_group_max": 4,
    "primary_conductor_thickness_mm": 5.0,
    "resonance_min_Hz": 15_000.0,
    "magnetizing_inductance_factor": 0.5,
    "size_W_max_mm": 1_200.0,
    "size_L_max_mm": 1_200.0,
    "size_H_max_mm": 750.0,
}
CURRENT7_HARD_SPEC_SHA256 = hashlib.sha256(json.dumps(
    CURRENT7_HARD_SPEC,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()
CANDIDATE_TEMPERATURE_TARGETS = (
    "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
    *TEMPERATURE_TARGETS,
)
DESIGN_PARAMETER_KEYS = (
    "N1_main", "N1_side", "N2_main", "N2_side", "l1", "l2", "h1", "w1",
    "n_core_group", "core_plate_t", "core_plate_pad_t",
    "cw1", "gap1", "cw2", "gap2",
    "nwh1", "nwh2", "cc_w2c_space_x", "cc_w2c_space_y",
    "w2c_w1c_space_x", "w2c_w1c_space_y", "w1c_w2s_gap_x_actual",
    "w1c_w2s_space_x", "w2s_w1s_space_x", "w1s_w2s_space_y",
    "w1s_cs_space_x", "cs_w1s_space_y", "h_gap2", "wcp_t", "wcp_pad_t",
    "wcp_len_pct", "wcp_len_x",
)
CANDIDATE_REPORT_FIELDS = (
    "size_W_mm", "size_L_mm", "size_H_mm", "size_WxLxH_mm",
    "volume_L", "footprint_cm2",
    "turns_primary", "turns_secondary_center", "turns_secondary_side",
    "cw1_conductor_thickness_mm", "cw2_conductor_thickness_mm",
    "gap1_mm", "gap2_mm",
    "nwl1_main_pack_width_mm", "nwl1_side_pack_width_mm",
    "nwl2_main_pack_width_mm", "nwl2_side_pack_width_mm",
    "nwh1_winding_height_mm", "nwh2_winding_height_mm",
    "core_depth_each_mm", "n_core_group",
    "core_cold_plate_thickness_mm", "core_thermal_pad_thickness_mm",
    "winding_cold_plate_thickness_mm", "winding_thermal_pad_thickness_mm",
    "wcp_len_pct", "wcp_len_x_mm",
    "leakage_target_uH", "pred_leakage_inductance_uH",
    "B_design_analytic_T", "B_legacy_0p7_T", "B_design_waveform",
    "B_denominator_coefficient", "Ae_m2", "Ae_gross_m2",
    "Ae_effective_m2", "core_lamination_factor", "B_area_basis",
    "pred_core_loss_W", "pred_core_cold_plate_loss_W",
    "pred_winding_cold_plate_loss_W", "pred_primary_winding_loss_W",
    "pred_secondary_center_winding_loss_W",
    "pred_secondary_side_winding_loss_W", "pred_secondary_winding_loss_W",
    "pred_component_winding_loss_sum_W", "pred_total_winding_loss_W",
    "pred_total_loss_W", "rated_power_W", "pred_efficiency_pct",
    "pred_k", "pred_C_tx_tx_F", "pred_C_rx_rx_F", "pred_C_tx_rx_F",
    "pred_f_res_tx_self_Hz", "pred_f_res_rx_self_Hz",
    "pred_f_res_interwinding_Hz",
    "pred_f_res_tx_screen_Hz", "pred_f_res_rx_screen_Hz",
    "pred_f_res_interwinding_screen_Hz", "pred_f_res_min_screen_Hz",
    "pred_Lm_tx_inferred_H", "pred_Lm_rx_inferred_H",
    "pred_Lm_tx_screen_H", "pred_Lm_rx_screen_H",
    "pred_magnetizing_inductance_factor",
    "core_vol_m3", "core_vol_gross_m3", "core_vol_effective_m3",
    "core_mass_kg", "core_mass_gross_kg", "core_mass_effective_kg",
    "MLT_Tx_mm", "MLT_Rx_main_mm", "MLT_Rx_side_mm",
    "cu_mass_Tx_kg", "cu_mass_Rx_main_kg", "cu_mass_Rx_side_kg",
    "cu_mass_total_kg", "window_fill_x", "window_fill_z1",
    "aspect_h1_l2", "aspect_w1_l2",
    "surrogate_output_basis",
)
_CANDIDATE_DIMENSION_ALIASES = {
    "size_W_mm": (
        "size_W_mm", "exterior_width_mm", "overall_width_mm",
        "bounding_box_width_mm", "bbox_width_mm",
    ),
    "size_L_mm": (
        "size_L_mm", "exterior_length_mm", "overall_length_mm",
        "bounding_box_length_mm", "bbox_length_mm",
    ),
    "size_H_mm": (
        "size_H_mm", "exterior_height_mm", "overall_height_mm",
        "bounding_box_height_mm", "bbox_height_mm",
    ),
    "footprint_cm2": (
        "footprint_cm2", "exterior_footprint_cm2",
        "overall_footprint_cm2", "bounding_box_footprint_cm2",
    ),
}
_CANDIDATE_SIZE_TRIPLET_ALIASES = (
    "size_WxLxH_mm", "exterior_size_WxLxH_mm",
    "overall_size_WxLxH_mm", "bounding_box_mm", "bbox_mm",
)
_CANDIDATE_REPLAY_VOLUME_REL_TOLERANCE = 1e-9
_CANDIDATE_REPLAY_VOLUME_ABS_TOLERANCE_L = 1e-6
INSULATION_KEYS = (
    "cc_w2c_space_x", "cc_w2c_space_y", "w2c_w1c_space_x",
    "w2c_w1c_space_y", "w1c_w2s_gap_x_actual", "w1s_cs_space_x",
    "cs_w1s_space_y", "h_gap2",
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _candidate_dimension_sources(*values: Any) -> list[dict[str, Any]]:
    """Return candidate geometry containers in deterministic precedence order."""

    sources: list[dict[str, Any]] = []
    seen: set[int] = set()
    pending = list(values)
    nested_keys = (
        "report", "decoded_params", "parameters", "design", "row", "full_row",
    )
    while pending:
        value = pending.pop(0)
        if not isinstance(value, dict) or id(value) in seen:
            continue
        seen.add(id(value))
        sources.append(value)
        pending.extend(value.get(key) for key in nested_keys)
    return sources


def _candidate_dimension_value(
    sources: list[dict[str, Any]], keys: tuple[str, ...],
) -> float | None:
    # Canonical fields outrank aliases even if an alias happens to occur in an
    # earlier wrapper object.
    for key in keys:
        for source in sources:
            value = _finite_number(source.get(key))
            if value is not None and value > 0.0:
                return value
    return None


def _candidate_size_triplet(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        raw = [
            value.get("W", value.get("width")),
            value.get("L", value.get("length")),
            value.get("H", value.get("height")),
        ]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    elif isinstance(value, str):
        raw = re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value,
        )
    else:
        return None
    if len(raw) != 3:
        return None
    parsed = tuple(_finite_number(item) for item in raw)
    if any(item is None or item <= 0.0 for item in parsed):
        return None
    return parsed  # type: ignore[return-value]


def _candidate_bounding_box(source: dict[str, Any]) -> dict[str, float]:
    """Replay the production exterior-box formula without inventing geometry."""

    try:
        try:
            from ..optimization.geometry_metrics import bounding_box_lit
        except ImportError:  # Direct execution with regression_260707 on sys.path.
            from optimization.geometry_metrics import bounding_box_lit
        volume_l, dimensions = bounding_box_lit(source)
        width, length, height = (
            _finite_number(item) for item in dimensions
        )
        volume = _finite_number(volume_l)
    except (ImportError, KeyError, TypeError, ValueError, OverflowError):
        return {}
    if (
        volume is None or volume <= 0.0
        or width is None or width <= 0.0
        or length is None or length <= 0.0
        or height is None or height <= 0.0
    ):
        return {}
    return {
        "size_W_mm": width,
        "size_L_mm": length,
        "size_H_mm": height,
        "volume_L": volume,
        "footprint_cm2": width * length / 100.0,
    }


def _candidate_dimension_report(*values: Any) -> dict[str, Any]:
    """Normalize explicit aliases, then fill gaps from authenticated geometry."""

    sources = _candidate_dimension_sources(*values)
    source_volume = _candidate_dimension_value(
        sources, ("volume_L", "objective_volume_L", "pred_volume_L"),
    )
    dimensions = {
        canonical: _candidate_dimension_value(sources, aliases)
        for canonical, aliases in _CANDIDATE_DIMENSION_ALIASES.items()
    }
    triplet = None
    supplied_triplet = None
    for key in _CANDIDATE_SIZE_TRIPLET_ALIASES:
        for source in sources:
            if key not in source:
                continue
            candidate_triplet = _candidate_size_triplet(source.get(key))
            if candidate_triplet is not None:
                triplet = candidate_triplet
                supplied_triplet = source.get(key)
                break
        if triplet is not None:
            break
    if triplet is not None:
        for canonical, value in zip(
            ("size_W_mm", "size_L_mm", "size_H_mm"), triplet,
        ):
            if dimensions[canonical] is None:
                dimensions[canonical] = value

    calculated: dict[str, float] = {}
    if any(dimensions[key] is None for key in (
        "size_W_mm", "size_L_mm", "size_H_mm",
    )):
        for source in sources:
            replay = _candidate_bounding_box(source)
            replay_volume = _finite_number(replay.get("volume_L"))
            if (
                replay_volume is not None
                and source_volume is not None
                and math.isclose(
                    replay_volume,
                    source_volume,
                    rel_tol=_CANDIDATE_REPLAY_VOLUME_REL_TOLERANCE,
                    abs_tol=_CANDIDATE_REPLAY_VOLUME_ABS_TOLERANCE_L,
                )
            ):
                calculated = replay
                break
        for key in ("size_W_mm", "size_L_mm", "size_H_mm"):
            if dimensions[key] is None:
                dimensions[key] = calculated.get(key)

    width = dimensions["size_W_mm"]
    length = dimensions["size_L_mm"]
    height = dimensions["size_H_mm"]
    if dimensions["footprint_cm2"] is None and None not in (width, length):
        dimensions["footprint_cm2"] = width * length / 100.0

    report = {
        key: value for key, value in dimensions.items() if value is not None
    }
    if None not in (width, length, height):
        report["size_WxLxH_mm"] = (
            supplied_triplet
            if isinstance(supplied_triplet, str) and supplied_triplet.strip()
            else f"{width:g} × {length:g} × {height:g}"
        )
    return report


def _duration_seconds(value: Any) -> float | None:
    """Return a non-negative finite duration without inventing missing timing."""
    if isinstance(value, bool):
        return None
    number = _finite_number(value)
    return number if number is not None and number >= 0 else None


def _simulation_timing_summary(
    frame: Any,
    local_tz=None,
    limit: int = SIMULATION_TIMING_WINDOW_ROWS,
    active_cohort: tuple[str, str] | None = None,
    current_physics_revision: str | None = CURRENT_PHYSICS_DATA_REVISION,
) -> dict[str, Any]:
    """Summarize recent timing fields for the dynamically active cohort."""
    columns = (
        "git_hash", "physics_data_revision", "saved_at",
        *(source_field for _, source_field in SIMULATION_TIMING_FIELDS),
        "cap_on", *CAP_TIMING_FIELDS,
    )
    rows = _frame_records(frame, columns)
    if active_cohort is None:
        active_cohort = _active_cohort_identity(
            rows, local_tz, current_physics_revision
        )
    active = _active_cohort_view(
        active_cohort, current_physics_revision
    )
    cohort_rows = [
        row for row in rows
        if _is_active_cohort(_cohort_identity(row), active_cohort)
    ]
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for index, row in enumerate(cohort_rows):
        stamp = _parse_time(row.get("saved_at"), local_tz)
        ranked.append((stamp.timestamp() if stamp else float("-inf"), index, row))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    recent = [item[2] for item in ranked[:max(0, int(limit))]]
    stages = {}
    for name, source_field in SIMULATION_TIMING_FIELDS:
        values = [
            duration for row in recent
            if (duration := _duration_seconds(row.get(source_field))) is not None
        ]
        stages[name] = {
            "source_field": source_field,
            "sample_count": len(values),
            "mean_seconds": sum(values) / len(values) if values else None,
            "median_seconds": statistics.median(values) if values else None,
        }
    electrostatic_values = []
    for row in recent:
        if _optional_flag(row.get("cap_on")) is not True:
            continue
        cap_parts = [
            _duration_seconds(row.get(source_field))
            for source_field in CAP_TIMING_FIELDS
        ]
        if all(value is not None for value in cap_parts):
            electrostatic_values.append(sum(cap_parts))
    stages["electrostatic"] = {
        "source_fields": list(CAP_TIMING_FIELDS),
        "sample_count": len(electrostatic_values),
        "mean_seconds": (
            sum(electrostatic_values) / len(electrostatic_values)
            if electrostatic_values else None
        ),
        "median_seconds": (
            statistics.median(electrostatic_values)
            if electrostatic_values else None
        ),
    }
    return {
        "available": any(stage["sample_count"] for stage in stages.values()),
        "cohort_basis": "active_identity",
        "cohort_label": active["label"],
        "cohort_filter": {
            "git_hash": active["git_hash"],
            "physics_data_revision": (
                active["physics_data_revision"]
                or active["expected_physics_data_revision"]
            ),
        },
        "active_cohort": active,
        "cohort_rows": len(cohort_rows),
        "unit": "seconds",
        "window_limit_rows": max(0, int(limit)),
        "window_rows": len(recent),
        "stages": stages,
    }


def _zero_aware_percentage_metrics(
    actual_values: list[Any],
    predicted_values: list[Any],
    zero_abs_tolerance: float = MAPE_ZERO_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Calculate APE metrics without dividing by structural zero targets."""
    tolerance = _finite_number(zero_abs_tolerance)
    if tolerance is None or tolerance < 0:
        raise ValueError("MAPE zero tolerance must be finite and non-negative")
    if len(actual_values) != len(predicted_values):
        raise ValueError("MAPE actual/predicted lengths differ")
    relative_errors: list[float] = []
    valid_pair_count = 0
    excluded_zero_count = 0
    for raw_actual, raw_predicted in zip(actual_values, predicted_values):
        actual = _finite_number(raw_actual)
        predicted = _finite_number(raw_predicted)
        if actual is None or predicted is None:
            continue
        valid_pair_count += 1
        if abs(actual) <= tolerance:
            excluded_zero_count += 1
            continue
        relative_errors.append(abs(predicted - actual) / abs(actual))
    return {
        "mape_pct": (
            statistics.mean(relative_errors) * 100
            if relative_errors else None
        ),
        "p90_ape_pct": (
            statistics.quantiles(relative_errors, n=10, method="inclusive")[8] * 100
            if len(relative_errors) >= 2
            else relative_errors[0] * 100 if relative_errors else None
        ),
        "mape_n": len(relative_errors),
        "mape_excluded_zero_count": excluded_zero_count,
        "mape_valid_pair_count": valid_pair_count,
        "mape_zero_abs_tolerance": tolerance,
    }


def _integer(value: Any, default: int = 0) -> int:
    number = _finite_number(value)
    return int(number) if number is not None else default


def _current7_identity_integer(value: Any) -> int:
    """Return an exact JSON integer, never a bool/coerced numeric identity."""
    return value if type(value) is int else -1


def _rolling_candidate_preview_contract(
    aggregate: dict[str, Any], candidates: list[Any],
) -> dict[str, Any]:
    """Validate the bounded aggregate Pareto preview without losing totals.

    New writers declare all three preview fields.  Legacy writers are accepted
    only while the complete Pareto set fits within the fixed preview limit;
    omission above that boundary is ambiguous and therefore fails closed.
    """

    raw_total = aggregate.get("pareto_count")
    declared_total = (
        raw_total
        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
        else -1
    )
    field_names = (
        "candidate_preview_count",
        "candidate_preview_limit",
        "candidate_preview_truncated",
    )
    present = tuple(name in aggregate for name in field_names)
    explicit = all(present)
    partial = any(present) and not explicit
    preview_count = len(candidates)
    if explicit:
        declared_preview_count = aggregate.get("candidate_preview_count")
        declared_preview_limit = aggregate.get("candidate_preview_limit")
        declared_truncated = aggregate.get("candidate_preview_truncated")
        exact_counts = bool(
            isinstance(declared_preview_count, int)
            and not isinstance(declared_preview_count, bool)
            and isinstance(declared_preview_limit, int)
            and not isinstance(declared_preview_limit, bool)
        )
        safe = bool(
            exact_counts
            and declared_total >= 0
            and declared_preview_limit == ROLLING_CANDIDATE_PREVIEW_LIMIT
            and declared_preview_count == preview_count
            and preview_count == min(
                declared_total, ROLLING_CANDIDATE_PREVIEW_LIMIT
            )
            and isinstance(declared_truncated, bool)
            and declared_truncated == (declared_total > preview_count)
        )
        return {
            "safe": safe,
            "explicit": True,
            "total_count": max(0, declared_total),
            "preview_count": preview_count,
            "preview_limit": ROLLING_CANDIDATE_PREVIEW_LIMIT,
            "preview_truncated": (
                declared_truncated if isinstance(declared_truncated, bool)
                else None
            ),
        }
    safe = bool(
        not partial
        and 0 <= declared_total <= ROLLING_CANDIDATE_PREVIEW_LIMIT
        and declared_total == preview_count
    )
    return {
        "safe": safe,
        "explicit": False,
        "total_count": max(0, declared_total),
        "preview_count": preview_count,
        "preview_limit": ROLLING_CANDIDATE_PREVIEW_LIMIT,
        "preview_truncated": False if safe else None,
    }


def _flag(value: Any) -> bool:
    number = _finite_number(value)
    if number is not None:
        return number == 1.0
    return str(value).strip().lower() in {"true", "yes", "pass", "passed"}


def _safe_text(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _display_text_has_mojibake(value: Any) -> bool:
    """Recognize common UTF-8-as-legacy-codepage display corruption.

    Korean UI sentences are Hangul plus ASCII punctuation.  CJK ideographs
    and compatibility ideographs in these two bounded status fields are a
    reliable corruption signal (for example ``理쒖...``), as are replacement
    characters and the usual Latin-1 UTF-8 fragments.
    """
    text = _safe_text(value, 4_000)
    if text is None:
        return True
    suspicious_fragments = ("\ufffd", "Ã", "Â", "â", "ì", "ë", "í")
    if any(fragment in text for fragment in suspicious_fragments):
        return True
    return any(
        "\u3400" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in text
    )


def _normalized_display_text(value: Any, fallback: str, limit: int = 1_000) -> str:
    text = _safe_text(value, limit)
    return fallback if text is None or _display_text_has_mojibake(text) else text


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded_json_bytes(path: Path, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    """Read one regular JSON file without accepting a partial/oversize value."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} is not a regular file")
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"{path.name} exceeds {max_bytes} byte safety limit")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value, raw


def _parse_time(value: Any, local_tz=None) -> datetime | None:
    text = _safe_text(value, 80)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%y%m%d_%H%M%S_%f", "%y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                parsed = None
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_tz or _now().tzinfo)
    return parsed


def _tier1_archived_detail_updated_at(value: Any) -> str:
    """Return a stable cache-busting timestamp for repaired archive details."""
    source_text = _safe_text(value, 100)
    source_time = _parse_time(source_text)
    if source_time is None:
        return TIER1_ARCHIVED_DETAIL_CACHE_REPAIR_AT
    return (source_time + timedelta(
        microseconds=TIER1_ARCHIVED_DETAIL_CACHE_REVISION_US
    )).isoformat()


def _coerce(value: Any) -> Any:
    number = _finite_number(value)
    if number is not None:
        return int(number) if number.is_integer() else number
    return _safe_text(value)


def _optional_text(value: Any, limit: int = 500) -> str | None:
    """Normalize schema-union nulls without turning NaN into a cohort tag."""
    if value is None:
        return None
    number = _finite_number(value)
    if number is None and isinstance(value, (float, int)):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "<na>", "none", "null"}:
        return None
    return text[:limit]


def _cohort_identity(row: dict[str, Any]) -> tuple[str, str]:
    """Return the normalized identity used by every campaign cohort reader."""
    revision = (_optional_text(row.get("git_hash"), 40) or "").lower()
    physics_revision = (
        _optional_text(row.get("physics_data_revision"), 160)
        or LEGACY_PHYSICS_DATA_REVISION
    )
    return revision, physics_revision


def _active_cohort_identity(
    frame: Any,
    local_tz=None,
    current_physics_revision: str | None = CURRENT_PHYSICS_DATA_REVISION,
) -> tuple[str, str] | None:
    """Select the newest solver identity for the deployed physics revision.

    Rows without a comparable ``saved_at`` cannot establish recency and are
    ignored.  If the newest comparable row lacks a solver hash, no active pair
    is claimed instead of silently selecting an older solver cohort.
    """
    expected = _optional_text(current_physics_revision, 160)
    if expected is None:
        return None
    ranked: list[tuple[float, int, tuple[str, str]]] = []
    for index, row in enumerate(_frame_records(
        frame, ("git_hash", "physics_data_revision", "saved_at")
    )):
        identity = _cohort_identity(row)
        if identity[1] != expected:
            continue
        stamp = _parse_time(row.get("saved_at"), local_tz)
        if stamp is None:
            continue
        ranked.append((
            stamp.timestamp(),
            index,
            identity,
        ))
    if not ranked:
        return None
    identity = max(ranked, key=lambda item: (item[0], item[1]))[2]
    return identity if identity[0] else None


def _is_active_cohort(
    identity: tuple[str, str],
    active_cohort: tuple[str, str] | None,
) -> bool:
    if active_cohort is None or identity[1] != active_cohort[1]:
        return False
    if solver_revision_matches_physics_cohort is None:
        return identity == active_cohort
    return solver_revision_matches_physics_cohort(
        identity[0], active_cohort[0], identity[1]
    )


def _active_cohort_view(
    active_cohort: tuple[str, str] | None,
    current_physics_revision: str | None = CURRENT_PHYSICS_DATA_REVISION,
) -> dict[str, Any]:
    expected = _optional_text(current_physics_revision, 160)
    if active_cohort is not None:
        git_hash, physics_revision = active_cohort
        fallback = f"Active cohort {git_hash[:10]}"
        return {
            "available": True,
            "status": "active",
            "git_hash": git_hash,
            "git_hash_short": git_hash[:10],
            "physics_data_revision": physics_revision,
            "expected_physics_data_revision": expected,
            "label": _normalized_display_text(
                f"활성 코호트 {git_hash[:10]}", fallback
            ),
        }
    if expected is not None:
        label = f"현재 revision 데이터 없음 · 기대 revision: {expected}"
        status = "no_current_revision_rows"
    else:
        label = "현재 revision 확인 불가 · PHYSICS_DATA_REVISION import 실패"
        status = "physics_revision_unavailable"
    return {
        "available": False,
        "status": status,
        "git_hash": None,
        "git_hash_short": None,
        "physics_data_revision": None,
        "expected_physics_data_revision": expected,
        "label": _normalized_display_text(
            label, "Current physics-revision cohort is unavailable"
        ),
    }


def _optional_flag(value: Any) -> bool | None:
    if value is None:
        return None
    number = _finite_number(value)
    if number is not None:
        if number == 1.0:
            return True
        if number == 0.0:
            return False
        return None
    text = str(value).strip().casefold()
    if text in {"true", "yes", "pass", "passed"}:
        return True
    if text in {"false", "no", "fail", "failed"}:
        return False
    return None


def _frame_records(frame: Any, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Project a pandas-like frame or record iterable onto bounded columns."""
    if frame is None:
        return []
    if isinstance(frame, dict):
        return [{key: frame.get(key) for key in columns if key in frame}]
    if isinstance(frame, (list, tuple)):
        return [
            {key: row.get(key) for key in columns if key in row}
            for row in frame if isinstance(row, dict)
        ]
    available = getattr(frame, "columns", ())
    try:
        selected = [key for key in columns if key in available]
        if not selected:
            return [{} for _ in range(len(frame))]
        projected = frame.loc[:, selected]
        records = projected.to_dict(orient="records")
    except (AttributeError, KeyError, TypeError, ValueError):
        return []
    return [dict(row) for row in records if isinstance(row, dict)]


def _scaled_stats(
    rows: list[dict[str, Any]],
    column: str,
    scale: float,
    suffix: str,
) -> dict[str, Any]:
    values = [
        value * scale
        for row in rows
        if (value := _finite_number(row.get(column))) is not None
    ]
    return {
        "source_column": column,
        "sample_count": len(values),
        f"min_{suffix}": min(values) if values else None,
        f"median_{suffix}": statistics.median(values) if values else None,
        f"max_{suffix}": max(values) if values else None,
    }


def _plain_stats(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    values = [
        value for row in rows
        if (value := _finite_number(row.get(column))) is not None
    ]
    return {
        "source_column": column,
        "sample_count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def _invalid_reasons(row: dict[str, Any]) -> list[str]:
    raw = _optional_text(row.get("_strict_invalid_reasons"), 20_000)
    if not raw:
        raw = _optional_text(row.get("em_validity_reason"), 20_000)
    if not raw:
        return []
    return list(dict.fromkeys(
        reason.strip() for reason in raw.split(";") if reason.strip()
    ))


def _campaign_frame_summary(
    frame: Any,
    now: datetime,
    active_cohort: tuple[str, str] | None = None,
    current_physics_revision: str | None = CURRENT_PHYSICS_DATA_REVISION,
) -> dict[str, Any]:
    """Summarize the campaign audit frame while tolerating missing columns.

    Canonically audited frames carry ``_strict_*`` columns.  Their per-row
    classifications are aggregated by physics revision without imposing an
    additional active-SHA gate.  Synthetic and CSV fallback rows may only
    carry stored result flags; those remain fail-closed outside the active
    pair because they were not recomputed by the quality contract.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    columns = (
        "git_hash", "physics_data_revision", "saved_at",
        "result_valid_em", "result_valid_thermal",
        "_strict_valid_em", "_strict_valid_full",
        "_strict_invalid_reasons", "em_validity_reason",
        "cap_on", *CAPACITANCE_FIELDS.values(), *RESONANCE_FIELDS.values(),
        "thermal_core_conductivity_model", "thermal_core_k_inplane",
        "thermal_core_k_throughstack", "core_lamination_factor",
        "winding_flux_linkage_readback_status",
        "winding_flux_linkage_readback_applicable",
        "winding_flux_linkage_readback_available",
        "winding_flux_linkage_readback_passed",
        "winding_flux_linkage_readback_reason",
    )
    records = _frame_records(frame, columns)
    if active_cohort is None:
        active_cohort = _active_cohort_identity(
            records, now.tzinfo, current_physics_revision
        )
    active = _active_cohort_view(
        active_cohort, current_physics_revision
    )
    prepared: list[dict[str, Any]] = []
    for row in records:
        revision, physics_revision = _cohort_identity(row)
        current = _is_active_cohort(
            (revision, physics_revision), active_cohort
        )
        strict_em_flag = _optional_flag(row.get("_strict_valid_em"))
        strict_full_flag = _optional_flag(row.get("_strict_valid_full"))
        strict_flags_recomputed = (
            strict_em_flag is not None and strict_full_flag is not None
        )
        if not strict_flags_recomputed:
            if strict_em_flag is None:
                strict_em_flag = (
                    _optional_flag(row.get("result_valid_em")) is True
                )
        if strict_full_flag is None:
            strict_full_flag = (
                bool(strict_em_flag)
                and _optional_flag(row.get("result_valid_thermal")) is True
            )
        prepared.append({
            **row,
            "_monitor_git_hash": revision,
            "_monitor_physics_revision": physics_revision,
            "_monitor_current": current,
            "_monitor_strict_recomputed": strict_flags_recomputed,
            "_monitor_strict_em": bool(
                strict_em_flag and (strict_flags_recomputed or current)
            ),
            "_monitor_strict_full": bool(
                strict_em_flag and strict_full_flag
                and (strict_flags_recomputed or current)
            ),
        })

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        grouped[(
            row["_monitor_git_hash"], row["_monitor_physics_revision"]
        )].append(row)

    cutoff = now - timedelta(hours=1)
    cohorts: list[dict[str, Any]] = []
    for (revision, physics_revision), cohort_rows in grouped.items():
        saved_stamps = [
            stamp
            for row in cohort_rows
            if (
                stamp := _parse_time(row.get("saved_at"), now.tzinfo)
            ) is not None
        ]
        latest_saved_at = max(
            saved_stamps, key=lambda stamp: stamp.timestamp(), default=None
        )
        strict_em_rows = sum(row["_monitor_strict_em"] for row in cohort_rows)
        strict_full_rows = sum(row["_monitor_strict_full"] for row in cohort_rows)
        recent_growth = sum(
            1 for row in cohort_rows
            if row["_monitor_strict_full"]
            and (
                stamp := _parse_time(row.get("saved_at"), now.tzinfo)
            ) is not None
            and cutoff <= stamp <= now
        )
        current = _is_active_cohort(
            (revision, physics_revision), active_cohort
        )
        cohorts.append({
            "git_hash": revision or None,
            "git_hash_short": revision[:10] if revision else "unknown",
            "physics_data_revision": physics_revision,
            "latest_saved_at": (
                latest_saved_at.isoformat() if latest_saved_at else None
            ),
            "active": current,
            "current": current,
            "raw_rows": len(cohort_rows),
            "strict_em_rows": strict_em_rows,
            "strict_full_rows": strict_full_rows,
            "growth_rate_per_hour": float(recent_growth),
        })

    def _cohort_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        latest = _parse_time(item.get("latest_saved_at"), now.tzinfo)
        latest_rank = latest.timestamp() if latest is not None else float("-inf")
        return (
            not item["active"], -latest_rank, -item["raw_rows"],
            item["git_hash_short"], item["physics_data_revision"],
        )

    cohorts.sort(key=_cohort_sort_key)

    expected_physics_revision = _optional_text(
        current_physics_revision, 160
    )
    revision_rows = [
        row for row in prepared
        if expected_physics_revision is not None
        and row["_monitor_physics_revision"] == expected_physics_revision
    ]
    revision_strict_em = [
        row for row in revision_rows if row["_monitor_strict_em"]
    ]
    revision_strict_full = [
        row for row in revision_rows if row["_monitor_strict_full"]
    ]
    revision_strict_stamps = sorted(
        stamp
        for row in revision_strict_full
        if (
            stamp := _parse_time(row.get("saved_at"), now.tzinfo)
        ) is not None
    )
    revision_history: list[dict[str, Any]] = []
    cumulative = 0
    for stamp, count in Counter(revision_strict_stamps).items():
        cumulative += count
        revision_history.append({
            "time": stamp.isoformat(),
            "added": count,
            "total": cumulative,
        })
    if len(revision_history) > 240:
        stride = math.ceil(len(revision_history) / 240)
        sampled_history = revision_history[::stride]
        if sampled_history[-1] != revision_history[-1]:
            sampled_history.append(revision_history[-1])
        revision_history = sampled_history
    member_cohorts = [
        cohort for cohort in cohorts
        if cohort["physics_data_revision"] == expected_physics_revision
        and cohort.get("git_hash")
    ]
    physics_revision_aggregate = {
        "available": expected_physics_revision is not None,
        "physics_data_revision": expected_physics_revision,
        "has_rows": bool(revision_rows),
        "raw_rows": len(revision_rows),
        "strict_em_rows": len(revision_strict_em),
        "strict_full_rows": len(revision_strict_full),
        "growth_rate_per_hour": float(sum(
            cutoff <= stamp <= now for stamp in revision_strict_stamps
        )),
        "added_24h": sum(
            now - timedelta(hours=24) <= stamp <= now
            for stamp in revision_strict_stamps
        ),
        "member_git_hashes": [
            cohort["git_hash"] for cohort in member_cohorts
        ],
        "member_git_hash_shorts": [
            str(cohort["git_hash"])[:7] for cohort in member_cohorts
        ],
        "first_strict_saved_at": (
            revision_strict_stamps[0].isoformat()
            if revision_strict_stamps else None
        ),
        "latest_strict_saved_at": (
            revision_strict_stamps[-1].isoformat()
            if revision_strict_stamps else None
        ),
        "history": revision_history,
    }

    current_rows = [row for row in prepared if row["_monitor_current"]]
    current_strict = [
        row for row in current_rows if row["_monitor_strict_full"]
    ]
    present_rows: list[dict[str, Any]] = []
    absent_rows = 0
    unknown_cap_rows = 0
    for row in current_strict:
        cap_flag = _optional_flag(row.get("cap_on"))
        if cap_flag is True:
            present_rows.append(row)
        elif cap_flag is False:
            absent_rows += 1
        else:
            unknown_cap_rows += 1
    electrostatic = {
        "available": bool(current_strict and present_rows),
        "cohort_basis": "active_strict_full",
        "cohort_label": active["label"],
        "cohort_filter": {
            "git_hash": active["git_hash"],
            "physics_data_revision": (
                active["physics_data_revision"]
                or active["expected_physics_data_revision"]
            ),
        },
        "active_cohort": active,
        "cohort_rows": len(current_strict),
        "cap_stage_present_rows": len(present_rows),
        "cap_stage_absent_rows": absent_rows,
        "cap_stage_unknown_rows": unknown_cap_rows,
        "capacitance": {
            key: _scaled_stats(present_rows, column, 1e9, "nF")
            for key, column in CAPACITANCE_FIELDS.items()
        },
        "resonance": {
            key: _scaled_stats(present_rows, column, 1e-3, "kHz")
            for key, column in RESONANCE_FIELDS.items()
        },
    }

    model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in current_rows:
        model = _optional_text(row.get("thermal_core_conductivity_model"), 160)
        if model:
            model_rows[model].append(row)
    ordered_models = [
        model for model in THERMAL_MODEL_TAGS if model in model_rows
    ] + sorted(model for model in model_rows if model not in THERMAL_MODEL_TAGS)
    thermal_models = {
        "available": bool(model_rows),
        "cohort_basis": "active_identity",
        "cohort_label": active["label"],
        "cohort_filter": {
            "git_hash": active["git_hash"],
            "physics_data_revision": (
                active["physics_data_revision"]
                or active["expected_physics_data_revision"]
            ),
        },
        "active_cohort": active,
        "total_rows": len(current_rows),
        "tagged_rows": sum(len(rows) for rows in model_rows.values()),
        "missing_rows": len(current_rows) - sum(
            len(rows) for rows in model_rows.values()
        ),
        "models": [
            {
                "model": model,
                "count": len(model_rows[model]),
                "percent": (
                    len(model_rows[model]) / len(current_rows) * 100.0
                    if current_rows else 0.0
                ),
                "thermal_core_k_inplane": _plain_stats(
                    model_rows[model], "thermal_core_k_inplane"
                ),
                "thermal_core_k_throughstack": _plain_stats(
                    model_rows[model], "thermal_core_k_throughstack"
                ),
            }
            for model in ordered_models
        ],
    }

    current_reason_counts: Counter[str] = Counter()
    legacy_reason_counts: Counter[str] = Counter()
    current_quarantined = 0
    legacy_quarantined = 0
    for row in prepared:
        reasons = _invalid_reasons(row)
        if row["_monitor_current"]:
            if row["_monitor_strict_full"]:
                continue
            current_quarantined += 1
            if not reasons:
                if not row["_monitor_strict_em"]:
                    reasons.append("stored_flag:result_valid_em")
                else:
                    reasons.append("stored_flag:result_valid_thermal")
            current_reason_counts.update(reasons)
        else:
            if row["_monitor_strict_full"]:
                continue
            legacy_quarantined += 1
            if (
                active_cohort is None
                or row["_monitor_git_hash"] != active_cohort[0]
            ):
                reasons.append(
                    "untrusted_provenance:solver_revision_mismatch"
                )
            if (
                current_physics_revision is not None
                and
                row["_monitor_physics_revision"]
                != current_physics_revision
            ):
                reasons.append("cohort:physics_data_revision_mismatch")
            legacy_reason_counts.update(dict.fromkeys(reasons, 1))

    def _reason_items(counts: Counter[str]) -> list[dict[str, Any]]:
        return [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    statuses = Counter(
        _optional_text(
            row.get("winding_flux_linkage_readback_status"), 80
        ) or "missing"
        for row in current_rows
    )
    readback_available = 0
    readback_unavailable = 0
    readback_missing = 0
    for row in current_rows:
        available = _optional_flag(
            row.get("winding_flux_linkage_readback_available")
        )
        status = _optional_text(
            row.get("winding_flux_linkage_readback_status"), 80
        )
        if available is True or status == "available":
            readback_available += 1
        elif available is False or status == "unavailable":
            readback_unavailable += 1
        else:
            readback_missing += 1

    return {
        "active_cohort": active,
        "cohorts": cohorts,
        "physics_revision_aggregate": physics_revision_aggregate,
        "electrostatic": electrostatic,
        "thermal_models": thermal_models,
        "quarantine": {
            "current": {
                "label": (
                    f"활성 코호트 {active['git_hash_short']}"
                    if active["available"] else active["label"]
                ),
                "rows": current_quarantined,
                "reasons": _reason_items(current_reason_counts),
            },
            "legacy": {
                "label": "레거시 cohort 잡음",
                "rows": legacy_quarantined,
                "reasons": _reason_items(legacy_reason_counts),
            },
        },
        "current_cohort_metadata": {
            "core_lamination_factor": _plain_stats(
                current_rows, "core_lamination_factor"
            ),
            "winding_flux_linkage_readback": {
                "cohort_rows": len(current_rows),
                "available_rows": readback_available,
                "unavailable_rows": readback_unavailable,
                "missing_rows": readback_missing,
                "statuses": [
                    {"status": status, "count": count}
                    for status, count in sorted(statuses.items())
                ],
            },
        },
    }


@dataclass(frozen=True)
class ReadResult:
    value: Any
    path: str
    exists: bool
    mtime: datetime | None = None
    warning: str | None = None


@dataclass
class _CacheEntry:
    signature: tuple[int, int]
    value: Any
    mtime: datetime


class SafeArtifactCache:
    """Caches the last good parse of each artifact by size and mtime."""

    def __init__(self) -> None:
        self._good: dict[tuple[str, str], _CacheEntry] = {}
        self._failed: dict[tuple[str, str], tuple[int, int]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _signature(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size

    def _read(
        self,
        path: Path,
        kind: str,
        parser: Callable[[Path], Any],
        default: Any,
        *,
        reuse_last_good_on_error: bool = True,
    ) -> ReadResult:
        key = (str(path.resolve()), kind)
        try:
            signature = self._signature(path)
        except FileNotFoundError:
            return ReadResult(default, str(path), False)
        except OSError as exc:
            return ReadResult(default, str(path), False, warning=f"{path.name} 상태 확인 실패: {exc}")

        with self._lock:
            cached = self._good.get(key)
            if cached and cached.signature == signature:
                return ReadResult(cached.value, str(path), True, cached.mtime)
            if self._failed.get(key) == signature:
                previous = (
                    cached.value
                    if reuse_last_good_on_error and cached else default
                )
                previous_time = (
                    cached.mtime
                    if reuse_last_good_on_error and cached else None
                )
                return ReadResult(
                    previous,
                    str(path),
                    True,
                    previous_time,
                    f"{path.name} 손상/작성 중: 마지막 정상 데이터를 표시합니다.",
                )

        try:
            value = parser(path)
            mtime = datetime.fromtimestamp(signature[0] / 1_000_000_000, tz=_now().tzinfo)
        except (
            OSError, UnicodeError, ValueError, TypeError, ImportError,
            csv.Error, json.JSONDecodeError,
        ) as exc:
            with self._lock:
                self._failed[key] = signature
                cached = self._good.get(key)
            previous = (
                cached.value
                if reuse_last_good_on_error and cached else default
            )
            previous_time = (
                cached.mtime
                if reuse_last_good_on_error and cached else None
            )
            return ReadResult(
                previous,
                str(path),
                True,
                previous_time,
                f"{path.name} 읽기 실패: {type(exc).__name__}: {exc}",
            )

        with self._lock:
            self._good[key] = _CacheEntry(signature, value, mtime)
            self._failed.pop(key, None)
        return ReadResult(value, str(path), True, mtime)

    def json(
        self,
        path: Path,
        default: Any = None,
        max_bytes: int = 16 * 1024 * 1024,
        *,
        fail_closed: bool = False,
    ) -> ReadResult:
        def parser(source: Path) -> Any:
            if source.stat().st_size > max_bytes:
                raise ValueError(f"file exceeds {max_bytes} byte safety limit")
            with source.open("r", encoding="utf-8-sig") as handle:
                return json.load(handle)

        return self._read(
            path,
            "json",
            parser,
            default,
            reuse_last_good_on_error=not fail_closed,
        )

    def csv(self, path: Path, max_rows: int = 100_000) -> ReadResult:
        def parser(source: Path) -> list[dict[str, str]]:
            rows: list[dict[str, str]] = []
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError("CSV header is missing")
                for index, row in enumerate(reader):
                    if index >= max_rows:
                        raise ValueError(f"CSV exceeds {max_rows} row safety limit")
                    rows.append(dict(row))
            return rows

        return self._read(path, f"csv:{max_rows}", parser, [],)

    def parquet(
        self,
        path: Path,
        max_rows: int = 100_000,
        max_columns: int = 2_000,
    ) -> ReadResult:
        """Read a bounded Parquet frame lazily and retain the last good frame."""
        def parser(source: Path) -> Any:
            try:
                import pandas as pd
                import pyarrow.parquet as parquet

                metadata = parquet.ParquetFile(source).metadata
                if metadata.num_rows > max_rows:
                    raise ValueError(
                        f"Parquet exceeds {max_rows} row safety limit"
                    )
                if metadata.num_columns > max_columns:
                    raise ValueError(
                        f"Parquet exceeds {max_columns} column safety limit"
                    )
                return pd.read_parquet(source)
            except (ImportError, OSError, TypeError, ValueError):
                raise
            except Exception as exc:
                # Arrow exception classes are intentionally not imported at
                # module load time; normalize parser failures for _read().
                raise ValueError(
                    f"Parquet parse failed: {type(exc).__name__}: {exc}"
                ) from exc

        return self._read(
            path,
            f"parquet:{max_rows}:{max_columns}",
            parser,
            None,
        )


class RefillControllerReader:
    """Read the external refill controller's last durable status tick."""

    def __init__(
        self,
        state_path: str | Path | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        configured_state_path = state_path or os.environ.get(
            "MFT_CONTROLLER_STATE_PATH", DEFAULT_CONTROLLER_STATE_PATH
        )
        configured_log_path = log_path or os.environ.get(
            "MFT_CONTROLLER_LOG_PATH", DEFAULT_CONTROLLER_LOG_PATH
        )
        self.state_path = Path(configured_state_path)
        self.log_path = Path(configured_log_path)

    @staticmethod
    def _count(value: Any) -> int | None:
        if type(value) is int and value >= 0:
            return value
        if isinstance(value, float) and math.isfinite(value) and value >= 0:
            integer = int(value)
            return integer if integer == value else None
        return None

    @classmethod
    def _concurrency_target(cls, state: dict[str, Any]) -> int | None:
        paths = (
            ("policy", "target"),
            ("policy", "concurrency_target"),
            ("policy", "project_concurrency_target"),
            ("target",),
            ("concurrency_target",),
            ("project_concurrency_target",),
            ("generation", "policy", "target"),
            ("generation", "identity", "project_concurrency_target"),
            ("generation", "identity", "concurrency_target"),
        )
        for path in paths:
            value: Any = state
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            target = cls._count(value)
            if target is not None:
                return target
        return cls._count(state.get("policy/target"))

    def _read_state(self) -> dict[str, Any] | None:
        stat = self.state_path.stat()
        if stat.st_size <= 0 or stat.st_size > CONTROLLER_STATE_MAX_BYTES:
            return None
        with self.state_path.open("rb") as handle:
            raw = handle.read(CONTROLLER_STATE_MAX_BYTES + 1)
        if len(raw) > CONTROLLER_STATE_MAX_BYTES:
            return None
        value = json.loads(raw)
        return value if isinstance(value, dict) else None

    def _read_last_tick(self) -> tuple[dict[str, Any] | None, datetime | None]:
        stat = self.log_path.stat()
        if stat.st_size <= 0:
            return None, None
        with self.log_path.open("rb") as handle:
            handle.seek(max(0, stat.st_size - CONTROLLER_LOG_TAIL_BYTES))
            tail = handle.read(CONTROLLER_LOG_TAIL_BYTES)
        last_line = next((line for line in reversed(tail.splitlines()) if line.strip()), None)
        if last_line is None:
            return None, None
        value = json.loads(last_line)
        if not isinstance(value, dict):
            return None, None
        tick_at = datetime.fromtimestamp(stat.st_mtime, tz=_now().tzinfo)
        return value, tick_at

    def snapshot(self) -> dict[str, Any]:
        try:
            state = self._read_state()
            tick, tick_at = self._read_last_tick()
            if state is None or tick is None or tick_at is None:
                return {"available": False}

            action = tick.get("action")
            if not isinstance(action, str) or not action.strip():
                return {"available": False}

            result: dict[str, Any] = {
                "available": True,
                "last_tick_at": _iso(tick_at),
                "action": action.strip(),
            }
            for key in (
                "active_project_tasks_before",
                "accepted_or_reconciled_count",
            ):
                count = self._count(tick.get(key))
                if count is not None:
                    result[key] = count
            generation = tick.get("generation")
            generation_id = generation.get("id") if isinstance(generation, dict) else None
            if isinstance(generation_id, str) and generation_id.strip():
                result["generation_id"] = generation_id.strip()
            concurrency_target = self._concurrency_target(state)
            if concurrency_target is not None:
                result["concurrency_target"] = concurrency_target
            return result
        except Exception:
            return {"available": False}


class SimulationPolicyConflict(RuntimeError):
    """The scheduler rejected a stale simulation-policy revision."""


class SchedulerReader:
    """Adapter for MFT scheduler status and durable simulation policy."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        task_prefix: str = "mft",
        project_name: str = "MFT_1MW_2026v1",
        timeout: float = 2.0,
        optional_timeout: float | None = None,
        ttl: float = 10.0,
        opener: Callable[..., Any] = urlopen,
        pool_timeout: float | None = None,
        optional_stale_ttl: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.task_prefix = task_prefix
        self.project_name = project_name.strip()
        self.timeout = timeout
        self.optional_timeout = (
            timeout if optional_timeout is None else max(0.1, optional_timeout)
        )
        self.pool_timeout = (
            self.optional_timeout
            if pool_timeout is None
            else max(0.1, pool_timeout)
        )
        self.optional_stale_ttl = max(0.0, optional_stale_ttl)
        self.ttl = ttl
        self._opener = opener
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._cached_at = 0.0
        self._cached: dict[str, Any] | None = None
        self._aedt_last_good: dict[str, tuple[float, dict[str, Any]]] = {}

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        body = None
        headers = {
            "Accept": "application/json",
            "User-Agent": "mft-monitor/1",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        with self._opener(
            request,
            timeout=self.timeout if timeout is None else timeout,
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _nonnegative_integer(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        number = _finite_number(value)
        if number is None or number < 0 or not number.is_integer():
            return None
        return int(number)

    @staticmethod
    def _boolean(value: Any) -> bool | None:
        return value if type(value) is bool else None

    @classmethod
    def _pool_snapshot(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("AEDT pool response is invalid")
        config = payload.get("config")
        plan = payload.get("plan")
        sessions = payload.get("sessions")
        leases = payload.get("leases")
        if (
            not isinstance(config, dict)
            or not isinstance(plan, dict)
            or not isinstance(sessions, list)
            or not isinstance(leases, list)
        ):
            raise ValueError("AEDT pool response is missing config/plan/sessions/leases")
        session_states = {
            str(state): max(0, _integer(count, 0))
            for state, count in (
                plan.get("state_counts")
                if isinstance(plan.get("state_counts"), dict) else {}
            ).items()
        }
        observed_session_states = Counter(
            state
            for item in sessions if isinstance(item, dict)
            if (state := _optional_text(item.get("state"), 80))
        )
        if not session_states:
            session_states = dict(observed_session_states)
        lease_states = {
            str(state): max(0, _integer(count, 0))
            for state, count in (
                plan.get("lease_counts")
                if isinstance(plan.get("lease_counts"), dict) else {}
            ).items()
        }
        observed_lease_states = Counter(
            state
            for item in leases if isinstance(item, dict)
            if (state := _optional_text(item.get("state"), 80))
        )
        if not lease_states:
            lease_states = dict(observed_lease_states)
        live_leases = cls._nonnegative_integer(plan.get("live_projects"))
        if live_leases is None:
            # Match the scheduler's LEASE_LIVE_STATES contract.  Queued is
            # also reported separately below so the UI can expose pressure.
            live_leases = sum(
                lease_states.get(state, 0)
                for state in ("queued", "leased", "active", "releasing")
            )
        return {
            "available": True,
            "enabled": cls._boolean(config.get("enabled")),
            "adapter_ready": cls._boolean(config.get("adapter_ready")),
            "validation_passed": cls._boolean(
                config.get("validation_passed")
            ),
            "operational": cls._boolean(config.get("operational")),
            "max_sessions": cls._nonnegative_integer(
                config.get("max_aedt_sessions")
            ),
            "min_idle_sessions": cls._nonnegative_integer(
                config.get("min_idle_aedt_sessions")
            ),
            "idle_sessions": cls._nonnegative_integer(
                plan.get("idle_session_count")
            ),
            "hard_sessions": cls._nonnegative_integer(
                plan.get("hard_session_count")
            ),
            "warm_spare_deficit": cls._nonnegative_integer(
                plan.get("warm_spare_deficit")
            ),
            "warm_spare_start_needed": cls._nonnegative_integer(
                plan.get("warm_spare_start_needed")
            ),
            # These arrays are capped history/diagnostic records, not live
            # capacity.  Keep their sizes explicitly named and never use
            # them as the denominator for pool utilization.
            "session_record_count": len(sessions),
            "lease_record_count": len(leases),
            "live_leases": live_leases,
            "queued_leases": lease_states.get("queued", 0),
            "ready_sessions": session_states.get("ready", 0),
            "busy_sessions": session_states.get("busy", 0),
            "session_states": dict(sorted(session_states.items())),
            "lease_states": dict(sorted(lease_states.items())),
            "warm_spare_reason": _safe_text(
                plan.get("warm_spare_status_reason"), 500
            ),
            "error": None,
        }

    @classmethod
    def _license_snapshot(cls, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("license response is invalid")
        candidates: list[dict[str, Any]] = []
        for key in ("display", "features", "in_use"):
            values = payload.get(key)
            if isinstance(values, list):
                candidates.extend(
                    item for item in values if isinstance(item, dict)
                )
        selected = next((
            item for item in candidates
            if str(item.get("feature") or "").strip().casefold()
            == "electronics_desktop"
        ), None)
        admission = payload.get("admission")
        admission = admission if isinstance(admission, dict) else {}
        admission_features = (
            admission.get("features")
            if isinstance(admission.get("features"), dict) else {}
        )
        if selected is None:
            fallback = admission_features.get("electronics_desktop")
            selected = fallback if isinstance(fallback, dict) else None
        if selected is None:
            raise ValueError("electronics_desktop license feature is unavailable")
        used = cls._nonnegative_integer(selected.get("used"))
        total = cls._nonnegative_integer(selected.get("total"))
        if used is None or total is None or used > total:
            raise ValueError("electronics_desktop license counts are invalid")
        error = _safe_text(payload.get("error"), 500)
        snapshot_valid = admission.get("snapshot_valid")
        if type(snapshot_valid) is not bool:
            snapshot_valid = payload.get("server_up") is True and not error
        return {
            "available": True,
            "feature": "electronics_desktop",
            "label": _safe_text(selected.get("label"), 100)
            or "AnsysElectronicsDesktop",
            "used": used,
            "total": total,
            "snapshot_valid": snapshot_valid,
            "checked_at": _safe_text(payload.get("checked_at"), 80),
            "error": error,
        }

    @classmethod
    def _node_local_snapshot(cls, payload: Any) -> dict[str, Any]:
        """Normalize active node-local AEDT host tasks and bundle identities."""
        if not isinstance(payload, list):
            raise ValueError("node-local AEDT host response is invalid")
        hosts: list[dict[str, Any]] = []
        seen_task_ids: set[int] = set()
        statuses: Counter[str] = Counter()
        expected_projects_by_bundle: dict[str, int] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            project = _optional_text(item.get("project"), 160)
            if project not in (None, NODE_LOCAL_AEDT_PROJECT):
                continue
            name = _optional_text(item.get("name"), 500) or ""
            entrypoint = _optional_text(item.get("entrypoint"), 500)
            if entrypoint is not None:
                entrypoint_leaf = entrypoint.replace("\\", "/").rsplit(
                    "/", 1
                )[-1]
                entrypoint_leaf = entrypoint_leaf.removesuffix(".py").rsplit(
                    ".", 1
                )[-1]
                if entrypoint_leaf != NODE_LOCAL_AEDT_ENTRYPOINT:
                    continue
            elif not name.endswith("-host"):
                # Compatibility for older task-list responses that omitted
                # entrypoint.  Canary host names are ``{bundle_id}-host``.
                continue
            status = (
                _optional_text(item.get("status"), 40) or ""
            ).lower()
            if status not in NODE_LOCAL_AEDT_ACTIVE_STATES:
                continue
            task_id = cls._nonnegative_integer(
                item.get("task_id", item.get("id"))
            )
            if task_id is None or task_id in seen_task_ids:
                continue
            seen_task_ids.add(task_id)
            raw_payload = item.get("payload_json")
            if isinstance(raw_payload, str):
                try:
                    raw_payload = json.loads(raw_payload)
                except (TypeError, ValueError):
                    raw_payload = None
            task_payload = raw_payload if isinstance(raw_payload, dict) else {}
            bundle_id = _optional_text(
                task_payload.get("aedt_canary_bundle_id"), 500
            )
            if bundle_id is None and name.endswith("-host"):
                bundle_id = name[:-5] or None
            expected_projects = cls._nonnegative_integer(
                task_payload.get("aedt_canary_expected_projects")
            )
            if bundle_id and expected_projects is not None:
                expected_projects_by_bundle[bundle_id] = expected_projects
            statuses[status] += 1
            hosts.append({
                "task_id": task_id,
                "name": name or None,
                "status": status,
                "bundle_id": bundle_id,
            })
        bundle_ids = list(dict.fromkeys(
            host["bundle_id"] for host in hosts if host["bundle_id"]
        ))
        return {
            "available": True,
            "project": NODE_LOCAL_AEDT_PROJECT,
            "active_host_tasks": len(hosts),
            "statuses": dict(sorted(statuses.items())),
            "bundle_count": len(bundle_ids),
            "bundle_ids": bundle_ids,
            "expected_projects": (
                sum(expected_projects_by_bundle[bundle_id]
                    for bundle_id in bundle_ids)
                if bundle_ids and all(
                    bundle_id in expected_projects_by_bundle
                    for bundle_id in bundle_ids
                ) else None
            ),
            "hosts": hosts,
            "error": None,
        }

    @staticmethod
    def _unavailable_pool(error: str | None = None) -> dict[str, Any]:
        return {
            "available": False,
            "enabled": None,
            "adapter_ready": None,
            "validation_passed": None,
            "operational": None,
            "max_sessions": None,
            "min_idle_sessions": None,
            "idle_sessions": None,
            "hard_sessions": None,
            "warm_spare_deficit": None,
            "warm_spare_start_needed": None,
            "session_record_count": None,
            "lease_record_count": None,
            "live_leases": None,
            "queued_leases": None,
            "ready_sessions": None,
            "busy_sessions": None,
            "session_states": {},
            "lease_states": {},
            "warm_spare_reason": None,
            "error": error,
        }

    @staticmethod
    def _unavailable_license(error: str | None = None) -> dict[str, Any]:
        return {
            "available": False,
            "feature": "electronics_desktop",
            "label": "AnsysElectronicsDesktop",
            "used": None,
            "total": None,
            "snapshot_valid": None,
            "checked_at": None,
            "error": error,
        }

    def _remember_aedt_snapshot(
        self, name: str, snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        self._aedt_last_good[name] = (time.monotonic(), dict(snapshot))
        return snapshot

    def _stale_aedt_snapshot(
        self,
        name: str,
        error: str,
        unavailable: Callable[[str | None], dict[str, Any]],
    ) -> dict[str, Any]:
        cached = self._aedt_last_good.get(name)
        if cached is not None:
            cached_at, snapshot = cached
            age = max(0.0, time.monotonic() - cached_at)
            if age <= self.optional_stale_ttl:
                return {
                    **snapshot,
                    "stale": True,
                    "stale_age_seconds": round(age, 3),
                    "error": error,
                }
        return unavailable(error)

    @staticmethod
    def _unavailable_node_local(error: str | None = None) -> dict[str, Any]:
        return {
            "available": False,
            "project": NODE_LOCAL_AEDT_PROJECT,
            "active_host_tasks": None,
            "statuses": {},
            "bundle_count": None,
            "bundle_ids": [],
            "expected_projects": None,
            "hosts": [],
            "error": error,
        }

    def _aedt_attach_snapshot(self) -> dict[str, Any]:
        """Fetch bounded attach diagnostics without self-contention on 8002."""
        pool = self._unavailable_pool()
        license_status = self._unavailable_license()
        node_local = self._unavailable_node_local()
        node_local_query = urlencode({
            "project": NODE_LOCAL_AEDT_PROJECT,
            "status": ",".join(NODE_LOCAL_AEDT_ACTIVE_STATES),
            "limit": NODE_LOCAL_AEDT_TASK_LIMIT,
        })
        requests = {
            # Ask the two light endpoints first.  Some legacy scheduler builds
            # serialize the 300+ KiB pool projection on the event loop.
            "license": ("/api/licenses", self.optional_timeout),
            "node_local": (
                f"/api/tasks?{node_local_query}", self.optional_timeout
            ),
            "pool": ("/api/aedt-pool", self.pool_timeout),
        }
        # The legacy 8002 process can serialize the large pool response on its
        # event loop.  Concurrent monitor requests therefore made the tiny
        # license request time out behind our own pool request.  Keep each call
        # independently bounded, but issue the lightweight views first.
        for name, (path, request_timeout) in requests.items():
            try:
                payload = self._request_json(path, timeout=request_timeout)
                if name == "pool":
                    pool = self._remember_aedt_snapshot(
                        "pool", self._pool_snapshot(payload)
                    )
                elif name == "license":
                    license_status = self._remember_aedt_snapshot(
                        "license", self._license_snapshot(payload)
                    )
                else:
                    node_local = self._node_local_snapshot(payload)
            except Exception as exc:
                message = (
                    f"{path} 조회 실패: {type(exc).__name__}: {exc}"
                )
                if name == "pool":
                    pool = self._stale_aedt_snapshot(
                        "pool", message, self._unavailable_pool
                    )
                elif name == "license":
                    license_status = self._stale_aedt_snapshot(
                        "license", message, self._unavailable_license
                    )
                else:
                    # Older scheduler deployments do not expose the filtered
                    # host-task view.  Keep that optional absence isolated from
                    # central pool/license health.
                    node_local = self._unavailable_node_local(message)
        errors = [
            error for error in (pool.get("error"), license_status.get("error"))
            if error
        ]
        if pool["available"]:
            if pool["enabled"] is False:
                state = "disabled"
            elif pool["operational"] is not True:
                state = "gated"
            else:
                idle = pool.get("idle_sessions")
                minimum_idle = pool.get("min_idle_sessions")
                if pool.get("stale") or bool(pool.get("error")):
                    state = "degraded"
                elif idle is None or minimum_idle is None:
                    state = "partial"
                elif idle < minimum_idle:
                    state = (
                        "warming"
                        if (pool.get("warm_spare_start_needed") or 0) > 0
                        else "shortfall"
                    )
                elif not license_status["available"]:
                    state = "partial"
                elif (
                    license_status.get("snapshot_valid") is not True
                    or bool(license_status.get("error"))
                ):
                    state = "degraded"
                else:
                    state = "operational"
        elif license_status["available"]:
            state = "pool_unavailable"
        else:
            state = "unavailable"
        return {
            "available": bool(pool["available"] or license_status["available"]),
            "state": state,
            "license": license_status,
            "pool": pool,
            "node_local": node_local,
            "errors": errors,
        }

    @classmethod
    def _first_count(cls, *values: Any) -> int | None:
        for value in values:
            parsed = cls._nonnegative_integer(value)
            if parsed is not None:
                return parsed
        return None

    def _validate_project_identity(self, project: Any) -> dict[str, Any]:
        if not isinstance(project, dict):
            raise ValueError("scheduler project response is invalid")
        name = str(project.get("project") or project.get("name") or "").strip()
        if name != self.project_name:
            raise ValueError("scheduler returned a different project")
        return project

    def _project_status(self, project: Any) -> dict[str, Any]:
        """Normalize project counters without treating its safety cap as demand."""
        project = self._validate_project_identity(project)
        queued = self._first_count(project.get("queued_count"))
        attaching = self._first_count(project.get("attaching_count"))
        active = self._first_count(
            project.get("active_count"), project.get("executing_count")
        )
        solving = self._first_count(project.get("solving_count"), active)
        if any(value is None for value in (queued, attaching, active, solving)):
            raise ValueError("scheduler project live counts are unavailable or invalid")
        return {
            "project": self.project_name,
            "live_queued": queued,
            "live_attaching": attaching,
            "live_active": active,
            "live_solving": solving,
            # Compatibility for older dashboard consumers.
            "live_running": active,
            # Desired concurrency excludes work that is merely queued.
            "logical_active": attaching + active,
            "legacy_project_cap": self._first_count(project.get("max_active_tasks")),
            "project_updated_at": project.get("updated_at"),
        }

    def _simulation_policy_control(
        self,
        policy: Any,
        *,
        project: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Normalize the scheduler's versioned, durable desired-concurrency policy."""
        if not isinstance(policy, dict):
            raise ValueError("scheduler simulation policy response is invalid")
        combined = {**(project or {}), **policy}
        self._validate_project_identity(combined)
        limits = combined.get("limits")
        limits = limits if isinstance(limits, dict) else {}
        desired_min = self._first_count(
            combined.get("min_desired_simulations"),
            combined.get("desired_simulations_min"),
            limits.get("min_desired_simulations"),
            limits.get("minimum"),
            DEFAULT_PARALLEL_TARGET_MIN,
        )
        desired_max = self._first_count(
            combined.get("max_desired_simulations"),
            combined.get("desired_simulations_max"),
            limits.get("max_desired_simulations"),
            limits.get("maximum"),
            DEFAULT_PARALLEL_TARGET_MAX,
        )
        validated = self._first_count(
            combined.get("validated_concurrency_limit"),
            limits.get("validated_concurrency_limit"),
        )
        desired = self._first_count(combined.get("desired_simulations"))
        effective = self._first_count(combined.get("effective_simulations"))
        if desired_min is None or desired_max is None or desired_min > desired_max:
            raise ValueError("scheduler simulation policy bounds are invalid")
        if desired_max > PARALLEL_TARGET_SAFETY_CEILING:
            raise ValueError("scheduler simulation policy exceeds the MFT safety ceiling")
        if (
            validated is None
            or not desired_min <= validated <= PARALLEL_TARGET_SAFETY_CEILING
        ):
            raise ValueError("scheduler validated concurrency limit is invalid")
        # Accept a legacy desired value above a newly lowered capability so an
        # operator can repair it through the bounded UI.  The next value is
        # still limited to parallel_target_max below.
        if (
            desired is None
            or not desired_min <= desired <= PARALLEL_TARGET_SAFETY_CEILING
        ):
            raise ValueError("scheduler desired simulation count is invalid")
        if effective is not None and effective > desired_max:
            raise ValueError("scheduler effective simulation count is invalid")
        revision = combined.get("policy_revision")
        if isinstance(revision, bool) or not isinstance(revision, (int, str)):
            raise ValueError("scheduler simulation policy revision is invalid")
        if (isinstance(revision, int) and revision < 0) or not str(revision).strip():
            raise ValueError("scheduler simulation policy revision is invalid")

        counts = combined.get("counts")
        counts = counts if isinstance(counts, dict) else {}
        queued = self._first_count(
            combined.get("queued_simulations"), combined.get("queued_count"),
            counts.get("queued"),
        )
        attaching = self._first_count(
            combined.get("attaching_simulations"), combined.get("attaching_count"),
            counts.get("attaching"),
        )
        active = self._first_count(
            combined.get("active_simulations"), combined.get("active_count"),
            combined.get("executing_count"), counts.get("active"),
        )
        solving = self._first_count(
            combined.get("solving_simulations"), combined.get("solving_count"),
            counts.get("solving"), active,
        )
        reported_logical = self._first_count(
            combined.get("logical_active_simulations"),
            combined.get("logical_active_count"),
            counts.get("logical_active"),
        )
        if project is not None:
            status = self._project_status(project)
            queued = status["live_queued"] if queued is None else queued
            attaching = status["live_attaching"] if attaching is None else attaching
            active = status["live_active"] if active is None else active
            solving = status["live_solving"] if solving is None else solving
        if any(value is None for value in (queued, attaching, active, solving)):
            raise ValueError("scheduler simulation policy live counts are unavailable")
        # Compatibility with the first policy API revision, where active_count
        # was accidentally emitted as attaching+solving.  The explicit logical
        # total makes that wire shape unambiguous and keeps UI semantics stable.
        if reported_logical == active and attaching > 0 and solving <= active:
            active = max(solving, active - attaching)
        if solving > active:
            raise ValueError("scheduler solving count exceeds active count")

        raw_constraint = combined.get("resource_constraint")
        if isinstance(raw_constraint, dict):
            constraint = {
                str(key): value for key, value in raw_constraint.items()
                if isinstance(key, str) and value is not None
            }
        elif raw_constraint is None:
            constraint = None
        else:
            constraint = {"reason": _safe_text(raw_constraint, 500)}
        gate_reason = (
            _optional_text(combined.get("control_gate_reason"), 500)
            or _optional_text(combined.get("gate_reason"), 500)
            or (
                _optional_text(constraint.get("reason"), 500)
                if isinstance(constraint, dict) else None
            )
        )
        scheduler_control_enabled = combined.get("control_enabled")
        control_enabled = (
            scheduler_control_enabled
            if type(scheduler_control_enabled) is bool else True
        )
        if not control_enabled and not gate_reason:
            gate_reason = "scheduler가 simulation-policy 변경을 잠갔습니다"
        return {
            "project": self.project_name,
            "policy_supported": True,
            "control_enabled": control_enabled,
            "read_only": not control_enabled,
            "parallel_target": desired,
            "desired_simulations": desired,
            "effective_simulations": effective,
            "validated_concurrency_limit": validated,
            "parallel_target_min": desired_min,
            # An operator may only select a concurrency level that has passed
            # the scheduler rollout gate, even if the configured safety cap is higher.
            "parallel_target_max": min(desired_max, validated),
            "configured_target_max": desired_max,
            "policy_revision": revision,
            "scale_down_mode": str(
                combined.get("scale_down_mode") or "drain"
            ).strip().lower(),
            "live_queued": queued,
            "live_attaching": attaching,
            "live_active": active,
            "live_solving": solving,
            "live_running": active,
            "logical_active": attaching + active,
            "resource_constraint": constraint,
            "control_gate_reason": gate_reason,
            "project_updated_at": combined.get("updated_at"),
        }

    def set_simulation_policy(
        self,
        desired_simulations: int,
        *,
        expected_revision: int | str,
    ) -> dict[str, Any]:
        """CAS-update durable MFT demand; lowering always uses graceful drain."""
        if (
            type(desired_simulations) is not int
            or not DEFAULT_PARALLEL_TARGET_MIN
            <= desired_simulations
            <= PARALLEL_TARGET_SAFETY_CEILING
        ):
            raise ValueError(
                "desired simulations must be an integer between "
                f"{DEFAULT_PARALLEL_TARGET_MIN} and "
                f"{PARALLEL_TARGET_SAFETY_CEILING}"
            )
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, (int, str))
            or not str(expected_revision).strip()
        ):
            raise ValueError("expected policy revision is required")
        if not self.project_name:
            raise ValueError("scheduler project control is not configured")
        project_path = quote(self.project_name, safe="")
        try:
            policy = self._request_json(
                f"/api/projects/{project_path}/simulation-policy",
                method="PATCH",
                payload={
                    "desired_simulations": desired_simulations,
                    "expected_revision": expected_revision,
                    "scale_down_mode": "drain",
                },
            )
        except HTTPError as exc:
            if exc.code == 409:
                raise SimulationPolicyConflict(
                    "simulation policy changed; refresh and retry"
                ) from exc
            raise
        control = self._simulation_policy_control(policy)
        if control["desired_simulations"] != desired_simulations:
            raise ValueError("scheduler simulation policy readback mismatch")
        with self._lock:
            self._cached = None
            self._cached_at = 0.0
        return control

    def set_parallel_target(
        self, target: int, *, expected_revision: int | str | None = None
    ) -> dict[str, Any]:
        """Compatibility name for callers migrated to the versioned policy API."""
        if expected_revision is None:
            raise ValueError("expected policy revision is required")
        return self.set_simulation_policy(
            target, expected_revision=expected_revision
        )

    def mft_pipeline_status(self) -> dict[str, Any]:
        """Return the scheduler's authoritative MFT pipeline projection.

        The standalone monitor predates the durable multi-lane NSGA-II
        controllers.  Their authenticated result discovery lives in the
        scheduler, so keep this optional adapter fail-soft for older
        scheduler deployments while allowing the monitor to consume the
        current result contract.
        """
        try:
            payload = self._request_json(
                "/api/mft-pipeline/status",
                timeout=self.optional_timeout,
            )
            if not isinstance(payload, dict):
                raise ValueError("MFT pipeline response is invalid")
            if not isinstance(payload.get("nsga"), dict):
                raise ValueError("MFT pipeline response has no NSGA section")
            return payload
        except Exception as exc:
            return {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def snapshot(self) -> dict[str, Any]:
        """Return one shared refresh to concurrent dashboard callers."""
        now_monotonic = time.monotonic()
        with self._lock:
            if (
                self._cached is not None
                and now_monotonic - self._cached_at < self.ttl
            ):
                return self._cached
        # FastAPI exposes several read-only views that a browser can request at
        # once.  Without single-flight protection each request expanded the
        # scheduler's full AEDT pool independently and caused false timeouts.
        with self._refresh_lock:
            return self._snapshot_uncached()

    def _snapshot_uncached(self) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        with self._lock:
            if self._cached is not None and now_monotonic - self._cached_at < self.ttl:
                return self._cached
        # The campaign has more than ten thousand historical tasks.  Reading
        # /api/tasks would make a dashboard refresh take many seconds, so use
        # the aggregate summary plus the single MFT project record.
        query = urlencode({"name_prefix": self.task_prefix})
        try:
            payload = self._request_json(f"/api/tasks/summary?{query}")
            if not isinstance(payload, dict) or not isinstance(payload.get("statuses"), dict):
                raise ValueError("scheduler summary response is invalid")
            statuses = Counter({
                str(name).strip().lower(): _integer(count, 0)
                for name, count in payload["statuses"].items()
            })
            running = sum(statuses.get(name, 0) for name in ("running", "executing"))
            pending = sum(statuses.get(name, 0) for name in ("pending", "queued", "submitted"))
            completed = sum(statuses.get(name, 0) for name in ("completed", "complete", "succeeded", "success"))
            failed = sum(statuses.get(name, 0) for name in ("failed", "error", "timed_out", "timeout"))
            cancelled = sum(statuses.get(name, 0) for name in ("cancelled", "canceled"))
            result = {
                "connected": True,
                "url": self.base_url,
                "read_only": True,
                "control_enabled": False,
                "policy_supported": False,
                "task_prefix": self.task_prefix,
                "total": _integer(payload.get("total"), sum(statuses.values())),
                "running": running,
                "pending": pending,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "other": max(0, _integer(payload.get("total"), sum(statuses.values())) - running - pending - completed - failed - cancelled),
                "statuses": dict(sorted(statuses.items())),
                "error": None,
                "updated_at": _iso(_now()),
            }
            if self.project_name:
                try:
                    project_path = quote(self.project_name, safe="")
                    project = self._request_json(f"/api/projects/{project_path}")
                    result.update(self._project_status(project))
                    embedded = project.get("simulation_policy")
                    if isinstance(embedded, dict):
                        policy = {**project, **embedded}
                    elif (
                        "desired_simulations" in project
                        or "policy_revision" in project
                        or "validated_concurrency_limit" in project
                    ):
                        # Transitional schedulers advertise the core policy on
                        # the project record and expose effective/gate/count
                        # fields at the dedicated GET endpoint.
                        policy = self._request_json(
                            f"/api/projects/{project_path}/simulation-policy"
                        )
                    else:
                        policy = None
                    if policy is not None:
                        result.update(
                            self._simulation_policy_control(policy, project=project)
                        )
                    else:
                        result.update({
                            "parallel_target": None,
                            "desired_simulations": None,
                            "effective_simulations": None,
                            "validated_concurrency_limit": None,
                            "parallel_target_min": DEFAULT_PARALLEL_TARGET_MIN,
                            "parallel_target_max": DEFAULT_PARALLEL_TARGET_MAX,
                            "policy_revision": None,
                            "resource_constraint": None,
                            "control_gate_reason": (
                                "scheduler가 durable simulation-policy "
                                "capability를 아직 제공하지 않습니다"
                            ),
                        })
                    result["project_error"] = None
                except (
                    HTTPError,
                    URLError,
                    OSError,
                    ValueError,
                    UnicodeError,
                    json.JSONDecodeError,
                ) as exc:
                    result["project"] = self.project_name
                    result["parallel_target"] = None
                    result["desired_simulations"] = None
                    result["effective_simulations"] = None
                    result["validated_concurrency_limit"] = None
                    result["parallel_target_min"] = DEFAULT_PARALLEL_TARGET_MIN
                    result["parallel_target_max"] = DEFAULT_PARALLEL_TARGET_MAX
                    result["policy_revision"] = None
                    result["live_queued"] = max(0, statuses.get("queued", 0))
                    result["live_attaching"] = max(0, statuses.get("attaching", 0))
                    result["live_active"] = max(0, statuses.get("running", 0))
                    result["live_solving"] = result["live_active"]
                    result["live_running"] = result["live_active"]
                    result["logical_active"] = (
                        result["live_attaching"]
                        + result["live_running"]
                    )
                    result["resource_constraint"] = None
                    result["control_gate_reason"] = (
                        "scheduler project 상태를 검증할 수 없습니다"
                    )
                    result["project_error"] = (
                        f"scheduler project 조회 실패: {type(exc).__name__}: {exc}"
                    )
            result["aedt_attach"] = self._aedt_attach_snapshot()
        except (HTTPError, URLError, OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            result = {
                "connected": False,
                "url": self.base_url,
                "read_only": True,
                "control_enabled": False,
                "policy_supported": False,
                "task_prefix": self.task_prefix,
                "project": self.project_name,
                "parallel_target": None,
                "desired_simulations": None,
                "effective_simulations": None,
                "validated_concurrency_limit": None,
                "parallel_target_min": DEFAULT_PARALLEL_TARGET_MIN,
                "parallel_target_max": DEFAULT_PARALLEL_TARGET_MAX,
                "policy_revision": None,
                "live_queued": 0,
                "live_attaching": 0,
                "live_active": 0,
                "live_solving": 0,
                "live_running": 0,
                "logical_active": 0,
                "project_error": None,
                "resource_constraint": None,
                "control_gate_reason": "scheduler에 연결할 수 없습니다",
                "total": 0,
                "running": 0,
                "pending": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
                "other": 0,
                "statuses": {},
                "error": f"scheduler 조회 실패: {type(exc).__name__}: {exc}",
                "updated_at": _iso(_now()),
                "aedt_attach": {
                    "available": False,
                    "state": "unavailable",
                    "license": self._unavailable_license(),
                    "pool": self._unavailable_pool(),
                    "node_local": self._unavailable_node_local(),
                    "errors": [],
                },
            }
        with self._lock:
            self._cached = result
            self._cached_at = now_monotonic
        return result


class RuntimeRecorder:
    """Persists a compact current snapshot and low-frequency history."""

    def __init__(self, directory: Path, min_interval_seconds: int = 60) -> None:
        self.directory = directory
        self.snapshot_path = directory / "monitor_snapshot.json"
        self.history_path = directory / "monitor_history.jsonl"
        self.min_interval_seconds = min_interval_seconds
        self._lock = threading.Lock()
        self._last_signature: tuple[Any, ...] | None = None
        self._last_write = 0.0
        self._snapshot_error: str | None = None

    def _write_snapshot(self, payload: dict[str, Any]) -> None:
        """Write one durable snapshot with a RaiDrive-safe bounded fallback."""
        serialized = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.snapshot_path.name}.",
            suffix=".tmp",
            dir=self.directory,
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())

            replace_error: OSError | None = None
            for attempt in range(5):
                try:
                    os.replace(temp, self.snapshot_path)
                    return
                except OSError as exc:
                    replace_error = exc
                    if attempt < 4:
                        time.sleep(0.05 * (2 ** attempt))

            # Some Windows network filesystems deny rename/replace even when
            # creating or overwriting the target directly is allowed.  The
            # recorder lock serializes writers while this fallback writes,
            # fsyncs, and verifies the complete serialized payload.
            try:
                with self.snapshot_path.open("wb") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                if self.snapshot_path.read_bytes() != serialized:
                    raise OSError("snapshot direct-write readback mismatch")
                return
            except OSError as fallback_error:
                raise OSError(
                    f"snapshot replace failed ({replace_error}); "
                    f"direct-write fallback failed ({fallback_error})"
                ) from fallback_error
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def _append_history(self, summary: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _summary(dashboard: dict[str, Any]) -> dict[str, Any]:
        data = dashboard.get("data", {})
        models = dashboard.get("models", {})
        nsga = dashboard.get("nsga2", {})
        verification = dashboard.get("verification", {})
        scheduler = dashboard.get("scheduler", {})
        return {
            "schema_version": SCHEMA_VERSION,
            "time": dashboard.get("generated_at"),
            "overall": dashboard.get("status", {}).get("overall"),
            "data": {
                "total_rows": data.get("total_rows"),
                "complete_rows": data.get("complete_rows"),
                "throughput_1h": data.get("throughput_1h"),
                "count_basis": data.get("count_basis"),
                "physics_data_revision": data.get(
                    "current_physics_data_revision"
                ),
                "revision_raw_rows": data.get("revision_raw_rows"),
                "member_git_hashes": data.get("member_git_hashes"),
                "pinned_solver_revision": data.get("pinned_revision"),
                "pinned_library_revision": data.get("pinned_library_revision"),
                "cohorts": [
                    {
                        key: cohort.get(key)
                        for key in (
                            "git_hash", "physics_data_revision", "raw_rows",
                            "strict_em_rows", "strict_full_rows",
                        )
                    }
                    for cohort in data.get("cohorts", [])
                    if isinstance(cohort, dict)
                ],
            },
            "models": {
                "trained": models.get("trained_count"),
                "planned": models.get("target_count"),
            },
            "nsga2": {
                "round": nsga.get("round"),
                "candidate_count": nsga.get("candidate_count"),
                "min_volume_L": (nsga.get("summary") or {}).get("min_volume_L"),
            },
            "verification": {
                "stage": verification.get("stage"),
                "valid": (verification.get("counts") or {}).get("valid"),
                "total": (verification.get("counts") or {}).get("total"),
                "final_status": (verification.get("final") or {}).get("status"),
            },
            "scheduler": {
                "connected": scheduler.get("connected"),
                "running": scheduler.get("running"),
                "pending": scheduler.get("pending"),
                "failed": scheduler.get("failed"),
            },
        }

    @staticmethod
    def _snapshot(dashboard: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        """Keep the durable snapshot small; detailed points remain in source artifacts."""
        return {
            **summary,
            "project": dashboard.get("project"),
            "status": dashboard.get("status"),
            "data": {
                **summary["data"],
                "em_valid_rows": dashboard.get("data", {}).get("em_valid_rows"),
                "thermal_valid_rows": dashboard.get("data", {}).get("thermal_valid_rows"),
                "eta_3000": dashboard.get("data", {}).get("eta_3000"),
                "latest_data_at": dashboard.get("data", {}).get("latest_data_at"),
            },
            "models": {
                **summary["models"],
                "items": [
                    {
                        key: model.get(key)
                        for key in ("target", "status", "n_used", "r2", "rmse", "mape_pct", "p90_ape_pct")
                    }
                    for model in dashboard.get("models", {}).get("models", [])
                ],
            },
            "nsga2": {**summary["nsga2"], "summary": dashboard.get("nsga2", {}).get("summary")},
            "verification": {
                **summary["verification"],
                "counts": dashboard.get("verification", {}).get("counts"),
                "agreement": dashboard.get("verification", {}).get("agreement"),
            },
            "scheduler": {**summary["scheduler"], "total": dashboard.get("scheduler", {}).get("total")},
        }

    def record(self, dashboard: dict[str, Any]) -> None:
        summary = self._summary(dashboard)
        signature = (
            summary["overall"],
            summary["data"]["total_rows"],
            summary["data"]["complete_rows"],
            summary["data"]["physics_data_revision"],
            tuple(summary["data"].get("member_git_hashes") or []),
            tuple(
                (
                    cohort.get("git_hash"),
                    cohort.get("physics_data_revision"),
                    cohort.get("strict_full_rows"),
                )
                for cohort in summary["data"]["cohorts"]
            ),
            summary["models"]["trained"],
            summary["nsga2"]["round"],
            summary["nsga2"]["candidate_count"],
            summary["verification"]["stage"],
            summary["verification"]["final_status"],
            summary["scheduler"]["running"],
            summary["scheduler"]["pending"],
        )
        now_monotonic = time.monotonic()
        with self._lock:
            if signature == self._last_signature and now_monotonic - self._last_write < self.min_interval_seconds:
                if self._snapshot_error:
                    raise OSError(self._snapshot_error)
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            errors = []
            try:
                self._write_snapshot(self._snapshot(dashboard, summary))
                self._snapshot_error = None
            except (OSError, TypeError, ValueError) as exc:
                self._snapshot_error = f"snapshot write failed: {type(exc).__name__}: {exc}"
                errors.append(self._snapshot_error)

            history_written = False
            try:
                self._append_history(summary)
                history_written = True
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"history append failed: {type(exc).__name__}: {exc}")

            if history_written:
                self._last_signature = signature
                self._last_write = now_monotonic
            if errors:
                raise OSError("; ".join(errors))

    def history(self, limit: int = 2_000) -> dict[str, Any]:
        if not self.history_path.exists():
            return {"entries": [], "warning": None}
        entries: list[dict[str, Any]] = []
        bad_lines = 0
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict):
                            entries.append(value)
                    except json.JSONDecodeError:
                        bad_lines += 1
        except OSError as exc:
            return {"entries": [], "warning": f"모니터링 이력 읽기 실패: {exc}"}
        entries = entries[-limit:]
        warning = f"손상된 이력 {bad_lines}줄을 건너뛰었습니다." if bad_lines else None
        return {"entries": entries, "warning": warning}


class ArtifactService:
    """Builds stable JSON view models from live campaign files."""

    def __init__(
        self,
        regression_root: Path,
        scheduler: SchedulerReader | None = None,
        clock: Callable[[], datetime] = _now,
        record_runtime: bool = True,
        refill_controller: RefillControllerReader | None = None,
        continuous_pipeline: ContinuousPipelineReader | None = None,
    ) -> None:
        self.root = Path(regression_root).resolve()
        self.cache = SafeArtifactCache()
        self.scheduler = scheduler or SchedulerReader()
        self.refill_controller = refill_controller or RefillControllerReader()
        self.continuous_pipeline = (
            continuous_pipeline or ContinuousPipelineReader()
        )
        self.clock = clock
        self.recorder = RuntimeRecorder(self.root / "monitoring" / "runtime") if record_runtime else None
        self._campaign_audit_lock = threading.RLock()
        self._campaign_audit_key: tuple[Any, ...] | None = None
        self._campaign_audit_frame: Any = None
        self._campaign_audit_warning: str | None = None
        configured_capacitance_parity = os.environ.get(
            CAPACITANCE_PARITY_ARTIFACT_ENV, ""
        ).strip()
        self._capacitance_parity_artifact = (
            Path(configured_capacitance_parity).absolute()
            if configured_capacitance_parity else None
        )
        self._capacitance_parity_sha256 = os.environ.get(
            CAPACITANCE_PARITY_SHA256_ENV, ""
        ).strip().lower()
        configured_pareto_roots = os.environ.get("MFT_NSGA_ARTIFACT_ROOTS", "")
        if configured_pareto_roots.strip():
            self._pareto_artifact_roots = tuple(
                Path(item).resolve()
                for item in configured_pareto_roots.split(os.pathsep)
                if item.strip()
            )
        else:
            self._pareto_artifact_roots = (
                (self.root / "al_rounds").resolve(),
                Path("C:/Users/peets/slurm_scheduler_runtime").resolve(),
            )
        self._pareto_artifact_lock = threading.RLock()
        self._pareto_artifacts_by_sha: dict[str, Path] = {}
        self._pareto_artifact_scan_at = 0.0
        configured_nsga_status_roots = os.environ.get(
            "MFT_NSGA_UI_STATUS_ROOTS", ""
        )
        if configured_nsga_status_roots.strip():
            self._nsga_ui_status_roots = tuple(
                Path(item).resolve()
                for item in configured_nsga_status_roots.split(os.pathsep)
                if item.strip()
            )
        else:
            # Local process discovery is an explicit workstation capability;
            # shared/remote monitor deployments must remain scheduler-only.
            self._nsga_ui_status_roots = ()
        configured_slurm_offload_status = os.environ.get(
            "MFT_NSGA_SLURM_OFFLOAD_STATUS", ""
        ).strip()
        self._nsga_slurm_offload_status = (
            Path(configured_slurm_offload_status).resolve()
            if configured_slurm_offload_status else None
        )
        configured_tier1_status = os.environ.get(
            "MFT_TIER1_NSGA_STATUS", ""
        ).strip()
        self._tier1_nsga_status = (
            Path(configured_tier1_status).resolve()
            if configured_tier1_status else None
        )
        configured_tier1_pointer = os.environ.get(
            "MFT_TIER1_NSGA_POINTER", ""
        ).strip()
        configured_tier1_rolling_index = os.environ.get(
            "MFT_TIER1_ROLLING_INDEX", ""
        ).strip()
        self._tier1_rolling_index_configured = bool(
            configured_tier1_rolling_index
        )
        configured_current7_index = os.environ.get(
            CURRENT7_INDEX_ENV, ""
        ).strip()
        self._tier1_current7_index = (
            Path(configured_current7_index).resolve()
            if configured_current7_index else None
        )
        configured_condition_indexes = [
            item.strip() for item in os.environ.get(
                CURRENT7_CONDITION_INDEXES_ENV, ""
            ).split(os.pathsep)
            if item.strip()
        ]
        self._tier1_current7_condition_configuration_intent = bool(
            configured_condition_indexes
        )
        self._current7_condition_configuration_warnings: list[str] = []
        condition_indexes: list[Path] = []
        if len(configured_condition_indexes) > CURRENT7_CONDITION_INDEX_LIMIT:
            self._current7_condition_configuration_warnings.append(
                "current7 condition index configuration exceeds the bounded "
                f"limit of {CURRENT7_CONDITION_INDEX_LIMIT}; all condition "
                "indexes were rejected"
            )
        else:
            seen_condition_indexes: set[Path] = set()
            for raw_path in configured_condition_indexes:
                try:
                    condition_path = Path(raw_path).resolve()
                except (OSError, RuntimeError, ValueError) as exc:
                    self._current7_condition_configuration_warnings.append(
                        "current7 condition index path was rejected: "
                        f"{raw_path}: {exc}"
                    )
                    continue
                if condition_path == self._tier1_current7_index:
                    self._current7_condition_configuration_warnings.append(
                        "current7 primary index was ignored in the read-only "
                        "condition index list"
                    )
                    continue
                if condition_path in seen_condition_indexes:
                    continue
                seen_condition_indexes.add(condition_path)
                condition_indexes.append(condition_path)
        self._tier1_current7_condition_indexes = tuple(condition_indexes)
        self._deadline_design_publication_path = os.environ.get(
            DEADLINE_DESIGN_PATH_ENV, ""
        ).strip()
        self._deadline_design_publication_sha256 = os.environ.get(
            DEADLINE_DESIGN_SHA256_ENV, ""
        ).strip().lower()
        self._current7_diagnostics_lock = threading.RLock()
        self._current7_reference_cache: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        self._current7_diagnostics_cache: dict[str, dict[str, Any]] = {}
        configured_blocker_hpo_v2_status = os.environ.get(
            BLOCKER_HPO_V2_STATUS_ENV, ""
        ).strip()
        self._blocker_hpo_v2_status_path = (
            Path(configured_blocker_hpo_v2_status).resolve()
            if configured_blocker_hpo_v2_status else None
        )
        default_pointer = None
        continuous_root = getattr(self.continuous_pipeline, "root", None)
        if continuous_root is not None:
            try:
                default_pointer = (
                    Path(continuous_root).resolve() / "nsga" / "latest.json"
                )
            except OSError:
                default_pointer = None
        self._tier1_nsga_pointer = (
            Path(
                configured_tier1_rolling_index or configured_tier1_pointer
            ).resolve()
            if configured_tier1_rolling_index or configured_tier1_pointer
            else default_pointer
        )
        configured_tier1_cohort_root = os.environ.get(
            "MFT_TIER1_NSGA_COHORT_ROOT", ""
        ).strip()
        self._tier1_nsga_cohort_root = (
            Path(configured_tier1_cohort_root).resolve()
            if configured_tier1_cohort_root else None
        )
        configured_supplemental_roots = os.environ.get(
            TIER1_SUPPLEMENTAL_ROOTS_ENV, ""
        ).strip()
        supplemental_root_limit, limit_warnings = (
            _configured_tier1_supplemental_root_limit()
        )
        supplemental_roots: list[Path] = []
        supplemental_root_warnings = list(limit_warnings)
        seen_supplemental_roots: set[str] = set()
        for raw_root in (
            item.strip() for item in configured_supplemental_roots.split(
                os.pathsep
            )
        ):
            if not raw_root:
                continue
            root = Path(raw_root).absolute()
            identity = os.path.normcase(str(root))
            if identity in seen_supplemental_roots:
                continue
            seen_supplemental_roots.add(identity)
            if len(supplemental_roots) >= supplemental_root_limit:
                supplemental_root_warnings.append(
                    "Tier-1 supplemental root limit exceeded; extra roots "
                    f"were rejected (limit={supplemental_root_limit})"
                )
                break
            supplemental_roots.append(root)
        self._tier1_nsga_supplemental_root_limit = supplemental_root_limit
        self._tier1_nsga_supplemental_roots = tuple(supplemental_roots)
        self._tier1_nsga_supplemental_root_warnings = tuple(
            supplemental_root_warnings
        )
        self._tier1_archive_cache_lock = threading.RLock()
        self._tier1_archive_cache: dict[str, Any] | None = None
        self._tier1_history_lock = threading.RLock()
        self._tier1_history_diagnostics: dict[
            tuple[str, int, int], dict[str, Any]
        ] = {}
        configured_successor_handoff = os.environ.get(
            SEALED_SUCCESSOR_HANDOFF_ENV, ""
        ).strip()
        self._sealed_successor_handoff = (
            Path(configured_successor_handoff).absolute()
            if configured_successor_handoff else None
        )
        configured_successor_sha = os.environ.get(
            SEALED_SUCCESSOR_HANDOFF_SHA_ENV, ""
        ).strip().lower()
        self._sealed_successor_handoff_sha256 = (
            configured_successor_sha or SEALED_SUCCESSOR_HANDOFF_SHA256
        )
        configured_dual_plan = os.environ.get(DUAL_SUCCESSOR_PLAN_ENV, "").strip()
        configured_dual_ui = os.environ.get(DUAL_SUCCESSOR_UI_ENV, "").strip()
        configured_dual_receipt = os.environ.get(
            DUAL_SUCCESSOR_RECEIPT_ENV, ""
        ).strip()
        self._dual_successor_plan = (
            Path(configured_dual_plan).absolute() if configured_dual_plan else None
        )
        self._dual_successor_ui = (
            Path(configured_dual_ui).absolute() if configured_dual_ui else None
        )
        self._dual_successor_receipt = (
            Path(configured_dual_receipt).absolute()
            if configured_dual_receipt else None
        )
        self._dual_successor_plan_sha256 = (
            os.environ.get(DUAL_SUCCESSOR_PLAN_SHA_ENV, "").strip().lower()
            or DUAL_SUCCESSOR_PLAN_SHA256
        )
        self._dual_successor_ui_sha256 = (
            os.environ.get(DUAL_SUCCESSOR_UI_SHA_ENV, "").strip().lower()
            or DUAL_SUCCESSOR_UI_SHA256
        )
        self._dual_successor_receipt_sha256 = (
            os.environ.get(DUAL_SUCCESSOR_RECEIPT_SHA_ENV, "").strip().lower()
            or DUAL_SUCCESSOR_RECEIPT_SHA256
        )
        configured_validation_root = os.environ.get(
            DUAL_VALIDATION_ROOT_ENV, ""
        ).strip()
        self._dual_validation_root = (
            Path(configured_validation_root)
            if configured_validation_root else None
        )
        self._dual_validation_allowed_root = Path(
            os.environ.get(
                DUAL_VALIDATION_ALLOWED_ROOT_ENV,
                DUAL_VALIDATION_DEFAULT_ALLOWED_ROOT,
            ).strip()
            or DUAL_VALIDATION_DEFAULT_ALLOWED_ROOT
        )
        self._dual_validation_manifest_sha256 = (
            os.environ.get(
                DUAL_VALIDATION_MANIFEST_SHA_ENV, ""
            ).strip().lower()
            or DUAL_VALIDATION_MANIFEST_SHA256
        )
        self._dual_validation_plan_sha256 = (
            os.environ.get(
                DUAL_VALIDATION_PLAN_SHA_ENV, ""
            ).strip().lower()
            or DUAL_VALIDATION_PLAN_SHA256
        )
        self._dual_validation_source_sha256 = (
            os.environ.get(
                DUAL_VALIDATION_SOURCE_SHA_ENV, ""
            ).strip().lower()
            or DUAL_VALIDATION_SOURCE_SHA256
        )

    @staticmethod
    def _warnings(*results: ReadResult) -> list[str]:
        return [result.warning for result in results if result.warning]

    def _sealed_successor_rejected(
        self, warning: str | None = None,
    ) -> dict[str, Any]:
        configured = self._sealed_successor_handoff is not None
        return {
            "schema_version": "mft-monitor-sealed-successor-v1",
            "configured": configured,
            "available": False,
            "status": "rejected" if configured else "not_configured",
            "lifecycle_state": "sealed_successor",
            "source_kind": "sealed_successor_evidence",
            "display_only": True,
            "integrity_verified": False,
            "canonical_candidate": False,
            "terminal_result": False,
            "production": False,
            "production_eligible": False,
            "production_submission_enabled": False,
            "fea": False,
            "fea_submission_enabled": False,
            "fea_submission_performed": False,
            "fea_status": "blocked" if configured else "not_configured",
            "scheduler_task_id": None,
            "source": (
                str(self._sealed_successor_handoff)
                if self._sealed_successor_handoff else None
            ),
            "warning": warning,
        }

    @staticmethod
    def _hash_bound_json(
        reference: Any, *, label: str, max_bytes: int,
    ) -> tuple[dict[str, Any], Path, str]:
        if not isinstance(reference, dict):
            raise ValueError(f"{label} reference is not an object")
        path_text = reference.get("path")
        expected_sha = str(reference.get("sha256") or "").lower()
        if not isinstance(path_text, str) or not path_text.strip():
            raise ValueError(f"{label} path is missing")
        raw_path = Path(path_text)
        if not raw_path.is_absolute():
            raise ValueError(f"{label} path is not absolute")
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
            raise ValueError(f"{label} SHA-256 is malformed")
        if raw_path.is_symlink():
            raise ValueError(f"{label} is a symbolic link")
        path = raw_path.resolve()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} is not a regular file")
        size = path.stat().st_size
        if size <= 0 or size > max_bytes:
            raise ValueError(f"{label} size is outside the read-only limit")
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"{label} SHA-256 mismatch")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} JSON is unreadable: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{label} JSON is not an object")
        return payload, path, actual_sha

    def _sealed_successor_evidence(self) -> dict[str, Any]:
        """Authenticate one immutable successor without promoting it to Pareto.

        The configured handoff SHA is the trust anchor.  Every referenced
        artifact is then hash checked and cross-bound semantically.  A failure
        returns metadata only; no candidate, geometry, or prediction survives
        the fail-closed boundary.
        """
        path = self._sealed_successor_handoff
        if path is None:
            return self._sealed_successor_rejected()
        expected_handoff_sha = self._sealed_successor_handoff_sha256
        try:
            if re.fullmatch(r"[0-9a-f]{64}", expected_handoff_sha) is None:
                raise ValueError("configured handoff SHA-256 is malformed")
            if path.is_symlink() or not path.is_file():
                raise ValueError("sealed successor handoff is not a regular file")
            size = path.stat().st_size
            if size <= 0 or size > 128 * 1024:
                raise ValueError("sealed successor handoff exceeds the read-only limit")
            actual_handoff_sha = _sha256_file(path)
            if actual_handoff_sha != expected_handoff_sha:
                raise ValueError("sealed successor handoff SHA-256 mismatch")
            try:
                handoff = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"sealed successor handoff JSON is unreadable: {exc}") from exc
            if not isinstance(handoff, dict):
                raise ValueError("sealed successor handoff is not an object")

            def require(condition: bool, message: str) -> None:
                if not condition:
                    raise ValueError(message)

            require(
                handoff.get("schema_version") == SEALED_SUCCESSOR_HANDOFF_SCHEMA,
                "sealed successor handoff schema mismatch",
            )
            require(
                handoff.get("cohort_id") == SEALED_SUCCESSOR_COHORT_ID,
                "sealed successor cohort identity mismatch",
            )
            evidence_id = _safe_text(handoff.get("evidence_id"), 160)
            require(bool(evidence_id), "sealed successor evidence ID is missing")
            require(
                handoff.get("all_hard_pass") is True
                and _finite_number(handoff.get("total_positive_violation")) == 0.0,
                "sealed successor handoff is not all-hard feasible",
            )
            require(
                handoff.get("production_submission_enabled") is False
                and handoff.get("fea_submission_enabled") is False
                and handoff.get("scheduler_task_id") is None
                and handoff.get("live_pointer_switch_performed") is False,
                "sealed successor handoff attempted live authority",
            )

            deployment, deployment_path, deployment_sha = self._hash_bound_json(
                handoff.get("deployment_plan"),
                label="sealed successor deployment plan",
                max_bytes=512 * 1024,
            )
            manifest, manifest_path, manifest_sha = self._hash_bound_json(
                handoff.get("bundle_manifest"),
                label="sealed successor bundle manifest",
                max_bytes=4 * 1024 * 1024,
            )
            replay, replay_path, replay_sha = self._hash_bound_json(
                handoff.get("replay_attestation"),
                label="sealed successor replay attestation",
                max_bytes=512 * 1024,
            )
            candidate, candidate_path, candidate_sha = self._hash_bound_json(
                handoff.get("candidate"),
                label="sealed successor candidate",
                max_bytes=512 * 1024,
            )

            hard_spec = deployment.get("hard_spec")
            expected_hard_spec = {
                "B_limit_T": 1.2,
                "Llt_target_uH": 27.5,
                "Llt_tol_uH": 0.55,
                "T_limit_C": 120.0,
                "insulation_min_mm": 40.0,
                "magnetizing_inductance_factor": 0.5,
                "n_core_group_max": 4,
                "primary_conductor_thickness_mm": 5.0,
                "q_sigma": 1.0,
                "resonance_min_Hz": 10000.0,
                "size_H_max_mm": 750.0,
                "size_L_max_mm": 1200.0,
                "size_W_max_mm": 1200.0,
                "uncertainty_contract": "q90_conformal_half_width_physical_v1",
            }
            hard_spec_sha = str(deployment.get("hard_spec_sha256") or "").lower()
            require(
                deployment.get("schema_version") == "mft-tier1-slurm-deployment-v1"
                and deployment.get("attestation_schema_version")
                == "mft-tier1-slurm-temperature-attestation-v3"
                and deployment.get("task_schema_version")
                == "mft-tier1-slurm-seed-task-v3",
                "sealed successor deployment schema mismatch",
            )
            require(
                deployment.get("cohort_id") == handoff.get("cohort_id")
                and deployment.get("constraint_version")
                == SEALED_SUCCESSOR_CONSTRAINT_VERSION,
                "sealed successor deployment identity mismatch",
            )
            require(
                hard_spec == expected_hard_spec
                and _canonical_json_sha256(hard_spec) == hard_spec_sha,
                "sealed successor hard-spec identity mismatch",
            )
            require(
                deployment.get("bundle_manifest_sha256") == manifest_sha
                and Path(str(deployment.get("bundle_manifest"))).resolve()
                == manifest_path,
                "sealed successor deployment/manifest binding mismatch",
            )
            require(
                deployment.get("production_eligible") is False
                and deployment.get("fea_submission_approved") is False,
                "sealed successor deployment attempted production or FEA authority",
            )
            code_revision = str(deployment.get("nsga_code_revision") or "").lower()
            require(
                re.fullmatch(r"[0-9a-f]{40}", code_revision) is not None,
                "sealed successor NSGA code revision is malformed",
            )

            bundle_root = manifest_path.parent.resolve()
            local_bundle = Path(str(deployment.get("local_bundle"))).resolve()
            require(
                local_bundle == bundle_root
                and candidate_path.is_relative_to(bundle_root),
                "sealed successor candidate escapes the immutable bundle",
            )
            candidate_relative = candidate_path.relative_to(bundle_root).as_posix()
            manifest_files = manifest.get("files")
            manifest_files = manifest_files if isinstance(manifest_files, dict) else {}
            manifest_entry = manifest_files.get(candidate_relative)
            manifest_entry = manifest_entry if isinstance(manifest_entry, dict) else {}
            high_value = manifest.get("high_value_sha256")
            high_value = high_value if isinstance(high_value, dict) else {}
            require(
                manifest.get("schema_version") == "mft-tier1-slurm-deployment-v1"
                and manifest.get("attestation_schema_version")
                == "mft-tier1-slurm-temperature-attestation-v3"
                and manifest.get("task_schema_version")
                == "mft-tier1-slurm-seed-task-v3",
                "sealed successor bundle schema mismatch",
            )
            require(
                manifest.get("cohort_id") == handoff.get("cohort_id")
                and manifest.get("constraint_version")
                == SEALED_SUCCESSOR_CONSTRAINT_VERSION
                and manifest.get("hard_spec") == hard_spec
                and manifest.get("hard_spec_sha256") == hard_spec_sha
                and manifest.get("nsga_code_revision") == code_revision,
                "sealed successor bundle identity mismatch",
            )
            require(
                manifest.get("production_eligible") is False
                and manifest.get("fea_submission_approved") is False
                and manifest.get("automatic_promotion_allowed") is False,
                "sealed successor bundle attempted automatic authority",
            )
            require(
                manifest_entry.get("sha256") == candidate_sha
                and high_value.get(candidate_relative) == candidate_sha,
                "sealed successor candidate is not bound by bundle inventory",
            )
            for identity_key in (
                "source_model_manifest_sha256",
                "deployment_model_manifest_sha256",
                "stable_identity_sha256",
                "temperature_constraint_contract_sha256",
                "warm_start_sha256",
            ):
                require(
                    deployment.get(identity_key) == manifest.get(identity_key),
                    f"sealed successor {identity_key} mismatch",
                )

            replay_reference = handoff.get("replay_attestation")
            replay_reference = (
                replay_reference if isinstance(replay_reference, dict) else {}
            )
            replay_commit = str(replay_reference.get("commit") or "").lower()
            require(
                replay_commit == SEALED_SUCCESSOR_REPLAY_COMMIT,
                "sealed successor replay commit mismatch",
            )
            require(
                replay.get("schema_version")
                == "mft-nsga2-t120-independent-bundle-replay-v3"
                and replay.get("production_eligible") is False
                and replay.get("live_switch_performed") is False
                and replay.get("fea_submission_performed") is False,
                "sealed successor replay schema or authority mismatch",
            )
            replay_bundle = replay.get("successor_bundle")
            replay_bundle = replay_bundle if isinstance(replay_bundle, dict) else {}
            require(
                replay_bundle.get("cohort_id") == handoff.get("cohort_id")
                and replay_bundle.get("bundle_manifest_sha256") == manifest_sha
                and replay_bundle.get("nsga_code_revision") == code_revision
                and replay_bundle.get("all_manifest_file_hashes_passed") is True
                and replay_bundle.get("all_high_value_hashes_passed") is True,
                "sealed successor independent replay identity mismatch",
            )
            for identity_key in (
                "source_model_manifest_sha256",
                "deployment_model_manifest_sha256",
                "stable_identity_sha256",
                "warm_start_sha256",
            ):
                require(
                    replay_bundle.get(identity_key) == deployment.get(identity_key),
                    f"sealed successor replay {identity_key} mismatch",
                )
            discovery_seal = replay.get("append_only_discovery_seal")
            discovery_seal = (
                discovery_seal if isinstance(discovery_seal, dict) else {}
            )
            require(
                discovery_seal.get("sha256") == candidate_sha
                and discovery_seal.get("sha256_before_replay") == candidate_sha
                and discovery_seal.get("sha256_after_replay") == candidate_sha
                and discovery_seal.get("size_before_equals_after") is True
                and discovery_seal.get("mtime_before_equals_after") is True
                and discovery_seal.get("sha256_before_equals_after") is True,
                "sealed successor append-only discovery seal mismatch",
            )
            replay_result = replay.get("independent_replay")
            replay_result = replay_result if isinstance(replay_result, dict) else {}
            require(
                replay_result.get("repair_bit_exact") is True
                and replay_result.get("decoder_valid") is True
                and replay_result.get("hard_geometry_joint_pass") is True
                and replay_result.get("feature_parity_pass") is True
                and replay_result.get("all_hard_pass") is True
                and _finite_number(replay_result.get("total_positive_violation"))
                == 0.0,
                "sealed successor independent replay did not pass",
            )
            release_decision = replay.get("release_decision")
            release_decision = (
                release_decision if isinstance(release_decision, dict) else {}
            )
            require(
                release_decision.get("immutable_successor_bundle_ready") is True
                and release_decision.get("independent_replay_passed") is True
                and release_decision.get("automatic_live_switch_allowed") is False
                and release_decision.get("automatic_fea_submission_allowed") is False
                and release_decision.get("root_approval_required") is True,
                "sealed successor replay release decision mismatch",
            )

            require(
                candidate.get("schema_version")
                == "mft-nsga2-t120-exact-candidate-discovery-v3"
                and candidate.get("production_eligible") is False
                and candidate.get("live_switch_performed") is False
                and candidate.get("fea_submission_performed") is False
                and _finite_number(candidate.get("total_positive_violation"))
                == 0.0,
                "sealed successor candidate schema or authority mismatch",
            )
            provenance = candidate.get("provenance")
            provenance = provenance if isinstance(provenance, dict) else {}
            require(
                provenance.get("composite_model_manifest_sha256")
                == deployment.get("deployment_model_manifest_sha256")
                and provenance.get("source_model_manifest_sha256")
                == deployment.get("source_model_manifest_sha256"),
                "sealed successor candidate model identity mismatch",
            )

            decoded = candidate.get("decoded_geometry")
            require(isinstance(decoded, dict), "sealed successor geometry is missing")
            geometry_keys = {
                "N1_main", "N2_side", "l1_mm", "l2_mm", "h1_mm", "w1_mm",
                "n_core_group", "cw1_mm", "gap1_mm", "cw2_mm", "gap2_mm",
                "core_plate_t_mm", "core_plate_pad_t_mm", "wcp_t_mm",
                "wcp_pad_t_mm", "wcp_len_pct",
            }
            require(
                geometry_keys.issubset(decoded)
                and all(_finite_number(decoded.get(key)) is not None for key in geometry_keys),
                "sealed successor geometry is incomplete",
            )
            require(
                _finite_number(decoded.get("cw1_mm")) == 5.0
                and _finite_number(decoded.get("n_core_group")) == 4.0
                and (_finite_number(decoded.get("N2_side")) or 0.0) > 0.0,
                "sealed successor geometry violates the pinned manufacturing contract",
            )

            def finite_map(
                value: Any, expected_keys: tuple[str, ...], label: str,
            ) -> dict[str, float]:
                require(isinstance(value, dict), f"{label} is missing")
                require(
                    set(value) == set(expected_keys),
                    f"{label} key set mismatch",
                )
                result: dict[str, float] = {}
                for key in expected_keys:
                    number = _finite_number(value.get(key))
                    require(number is not None, f"{label} contains non-finite values")
                    result[key] = number
                return result

            constraints = finite_map(
                candidate.get("constraints_G"),
                SEALED_SUCCESSOR_CONSTRAINTS,
                "sealed successor constraints",
            )
            replay_constraints = finite_map(
                replay_result.get("constraint_G"),
                SEALED_SUCCESSOR_CONSTRAINTS,
                "sealed successor replay constraints",
            )
            require(
                all(value <= 0.0 for value in constraints.values()),
                "sealed successor has a positive hard-constraint violation",
            )
            require(
                all(
                    math.isclose(
                        constraints[key], replay_constraints[key],
                        rel_tol=0.0, abs_tol=1e-12,
                    )
                    for key in SEALED_SUCCESSOR_CONSTRAINTS
                ),
                "sealed successor constraint replay mismatch",
            )

            temperatures = finite_map(
                candidate.get("robust_temperature_C"),
                SEALED_SUCCESSOR_TEMPERATURE_TARGETS,
                "sealed successor robust temperatures",
            )
            replay_temperatures = finite_map(
                replay_result.get("robust_temperature_C"),
                SEALED_SUCCESSOR_TEMPERATURE_TARGETS,
                "sealed successor replay temperatures",
            )
            require(
                all(
                    math.isclose(
                        temperatures[key], replay_temperatures[key],
                        rel_tol=0.0, abs_tol=1e-12,
                    )
                    for key in SEALED_SUCCESSOR_TEMPERATURE_TARGETS
                ),
                "sealed successor temperature replay mismatch",
            )
            worst_temperature = max(temperatures.values())
            require(
                worst_temperature <= expected_hard_spec["T_limit_C"]
                and math.isclose(
                    worst_temperature,
                    _finite_number(handoff.get("worst_temperature_C")) or math.nan,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "sealed successor worst-temperature evidence mismatch",
            )

            exterior_raw = candidate.get("exterior_box_mm")
            replay_exterior_raw = replay_result.get("exterior_box_mm")
            require(
                isinstance(exterior_raw, list) and len(exterior_raw) == 3
                and isinstance(replay_exterior_raw, list)
                and len(replay_exterior_raw) == 3,
                "sealed successor exterior box is malformed",
            )
            exterior = [_finite_number(value) for value in exterior_raw]
            replay_exterior = [_finite_number(value) for value in replay_exterior_raw]
            require(
                all(value is not None for value in exterior)
                and all(value is not None for value in replay_exterior),
                "sealed successor exterior box contains non-finite values",
            )
            exterior_values = [float(value) for value in exterior if value is not None]
            replay_exterior_values = [
                float(value) for value in replay_exterior if value is not None
            ]
            require(
                all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
                    for left, right in zip(exterior_values, replay_exterior_values)
                )
                and exterior_values[0] <= expected_hard_spec["size_W_max_mm"]
                and exterior_values[1] <= expected_hard_spec["size_L_max_mm"]
                and exterior_values[2] <= expected_hard_spec["size_H_max_mm"],
                "sealed successor exterior-box replay or limit mismatch",
            )
            volume_liters = _finite_number(
                replay_result.get("objective_volume_liters")
            )
            total_loss_w = _finite_number(replay_result.get("objective_total_loss_W"))
            require(
                volume_liters is not None and total_loss_w is not None
                and math.isclose(
                    math.prod(exterior_values) / 1_000_000.0,
                    volume_liters,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                "sealed successor objective evidence mismatch",
            )

            resonance_hz = _finite_number(replay_result.get("resonance_screen_Hz"))
            require(
                resonance_hz is not None
                and resonance_hz >= expected_hard_spec["resonance_min_Hz"]
                and math.isclose(
                    expected_hard_spec["resonance_min_Hz"] - resonance_hz,
                    constraints["half_magnetizing_resonance_minimum"],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                "sealed successor resonance evidence mismatch",
            )
            llt_mu = _finite_number(replay_result.get("Llt_mu_uH"))
            llt_half_width = _finite_number(
                replay_result.get("Llt_q90_half_width_uH")
            )
            llt_g = _finite_number(replay_result.get("Llt_robust_G_uH"))
            handoff_llt_g = _finite_number(handoff.get("Llt_robust_G_uH"))
            require(
                llt_mu is not None and llt_half_width is not None and llt_g is not None
                and handoff_llt_g is not None
                and math.isclose(
                    abs(llt_mu - expected_hard_spec["Llt_target_uH"])
                    + llt_half_width - expected_hard_spec["Llt_tol_uH"],
                    llt_g,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    llt_g, constraints["Llt_robust_band"],
                    rel_tol=0.0, abs_tol=1e-12,
                )
                and math.isclose(
                    llt_g, handoff_llt_g, rel_tol=0.0, abs_tol=1e-12,
                )
                and llt_g <= 0.0,
                "sealed successor robust Llt evidence mismatch",
            )

            return {
                "schema_version": "mft-monitor-sealed-successor-v1",
                "configured": True,
                "available": True,
                "status": "fea_pending",
                "lifecycle_state": "sealed_successor",
                "source_kind": "sealed_successor_evidence",
                "display_only": True,
                "integrity_verified": True,
                "canonical_candidate": False,
                "terminal_result": False,
                "production": False,
                "production_eligible": False,
                "production_submission_enabled": False,
                "fea": False,
                "fea_submission_enabled": False,
                "fea_submission_performed": False,
                "fea_status": "pending",
                "scheduler_task_id": None,
                "live_pointer_switch_performed": False,
                "evidence_id": evidence_id,
                "cohort_id": handoff.get("cohort_id"),
                "constraint_version": SEALED_SUCCESSOR_CONSTRAINT_VERSION,
                "hard_spec": hard_spec,
                "model_identity": {
                    "source_model_manifest_sha256": deployment.get(
                        "source_model_manifest_sha256"
                    ),
                    "deployment_model_manifest_sha256": deployment.get(
                        "deployment_model_manifest_sha256"
                    ),
                    "stable_identity_sha256": deployment.get(
                        "stable_identity_sha256"
                    ),
                    "nsga_code_revision": code_revision,
                },
                "candidate": {
                    "candidate_sha256": candidate_sha,
                    "created_at": _safe_text(candidate.get("created_at"), 100),
                    "decoded_geometry": decoded,
                    "exterior_box_mm": exterior_values,
                    "objective_volume_L": volume_liters,
                    "objective_total_loss_W": total_loss_w,
                    "resonance": {
                        "screen_Hz": resonance_hz,
                        "minimum_Hz": expected_hard_spec["resonance_min_Hz"],
                        "pass": True,
                    },
                    "Llt_robust": {
                        "mu_uH": llt_mu,
                        "q90_half_width_uH": llt_half_width,
                        "G_uH": llt_g,
                        "target_uH": expected_hard_spec["Llt_target_uH"],
                        "tolerance_uH": expected_hard_spec["Llt_tol_uH"],
                        "pass": True,
                    },
                    "robust_temperatures_C": temperatures,
                    "worst_temperature_C": worst_temperature,
                    "temperature_limit_C": expected_hard_spec["T_limit_C"],
                    "constraints_G": constraints,
                    "all_constraints_pass": True,
                    "total_positive_violation": 0.0,
                },
                "integrity": {
                    "handoff_sha256": actual_handoff_sha,
                    "deployment_plan_sha256": deployment_sha,
                    "bundle_manifest_sha256": manifest_sha,
                    "candidate_sha256": candidate_sha,
                    "replay_attestation_sha256": replay_sha,
                    "replay_attestation_commit": replay_commit,
                    "all_referenced_hashes_verified": True,
                    "bundle_inventory_binding_verified": True,
                    "independent_replay_verified": True,
                },
                "source": str(path),
                "updated_at": _safe_text(
                    replay.get("completed_at") or candidate.get("created_at"), 100
                ),
                "warning": None,
            }
        except (OSError, TypeError, ValueError) as exc:
            return self._sealed_successor_rejected(
                f"sealed successor evidence rejected: {exc}"
            )

    def _dual_successors_rejected(
        self, warning: str | None = None,
    ) -> dict[str, Any]:
        paths = (
            self._dual_successor_plan,
            self._dual_successor_ui,
            self._dual_successor_receipt,
        )
        configured = any(path is not None for path in paths)
        return {
            "schema_version": "mft-monitor-dual-sealed-successors-v1",
            "configured": configured,
            "available": False,
            "status": "rejected" if configured else "not_configured",
            "source_kind": "sealed_successor_evidence",
            "display_only": True,
            "read_only": True,
            "integrity_verified": False,
            "candidate_count": 0,
            "candidates": [],
            "canonical_candidate": False,
            "terminal_result": False,
            "pareto": False,
            "production": False,
            "production_eligible": False,
            "fea": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "scheduler_mutation_performed": False,
            "standard_fea_validation": {
                "schema_version": (
                    "mft-monitor-dual-standard-fea-runtime-v1"
                ),
                "configured": self._dual_validation_root is not None,
                "available": False,
                "integrity_verified": False,
                "read_only": True,
                "candidate_count": 0,
                "candidates": {},
                "warning": (
                    "sealed successor evidence is unavailable; validation "
                    "overlay is fail-closed"
                    if self._dual_validation_root is not None else None
                ),
            },
            "warning": warning,
        }

    @staticmethod
    def _dual_successor_candidate(
        raw: dict[str, Any],
        *,
        hard_spec: dict[str, Any],
        integrity: dict[str, Any],
        model_identity: dict[str, Any],
    ) -> dict[str, Any]:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(message)

        params = raw.get("decoded_fea_params")
        constraints = raw.get("constraints_G")
        temperatures = raw.get("robust_temperature_C")
        geometry = raw.get("decoded_geometry")
        exterior = raw.get("exterior_box_mm")
        require(isinstance(params, dict) and len(params) == 71, "dual candidate 71-parameter inventory mismatch")
        require(
            _canonical_json_sha256(params)
            == str(raw.get("decoded_fea_params_sha256") or "").lower(),
            "dual candidate parameter SHA-256 mismatch",
        )
        require(
            isinstance(constraints, dict)
            and set(constraints) == set(SEALED_SUCCESSOR_CONSTRAINTS),
            "dual candidate all22 constraint inventory mismatch",
        )
        finite_constraints = {
            key: _finite_number(constraints.get(key))
            for key in SEALED_SUCCESSOR_CONSTRAINTS
        }
        require(
            all(value is not None and value <= 0.0 for value in finite_constraints.values()),
            "dual candidate no longer passes all22 constraints",
        )
        require(
            isinstance(temperatures, dict)
            and set(temperatures) == set(SEALED_SUCCESSOR_TEMPERATURE_TARGETS),
            "dual candidate all11 temperature inventory mismatch",
        )
        finite_temperatures = {
            key: _finite_number(temperatures.get(key))
            for key in SEALED_SUCCESSOR_TEMPERATURE_TARGETS
        }
        temperature_limit = _finite_number(hard_spec.get("T_limit_C"))
        require(
            temperature_limit is not None
            and all(
                value is not None and value <= temperature_limit
                for value in finite_temperatures.values()
            ),
            "dual candidate no longer passes all11 temperatures",
        )
        require(isinstance(geometry, dict), "dual candidate geometry is missing")
        require(
            _finite_number(geometry.get("cw1_mm")) == 5.0
            and (_finite_number(geometry.get("n_core_group")) or math.inf) <= 4.0,
            "dual candidate manufacturing geometry drifted",
        )
        require(
            isinstance(exterior, list)
            and len(exterior) == 3
            and all(_finite_number(value) is not None for value in exterior),
            "dual candidate exterior box is malformed",
        )
        exterior_values = [float(value) for value in exterior]
        require(
            exterior_values[0] <= float(hard_spec["size_W_max_mm"])
            and exterior_values[1] <= float(hard_spec["size_L_max_mm"])
            and exterior_values[2] <= float(hard_spec["size_H_max_mm"]),
            "dual candidate exterior box exceeds the hard specification",
        )
        artifact_sha = str(raw.get("artifact_sha256") or "").lower()
        params_sha = str(raw.get("decoded_fea_params_sha256") or "").lower()
        dedupe_sha = str(raw.get("dedupe_identity_sha256") or "").lower()
        dedupe_identity = {
            "schema_version": "mft-t120-successor-fea-candidate-dedupe-v1",
            "cohort_id": SEALED_SUCCESSOR_COHORT_ID,
            "constraint_version": SEALED_SUCCESSOR_CONSTRAINT_VERSION,
            "candidate_artifact_sha256": artifact_sha,
            "decoded_fea_params_sha256": params_sha,
        }
        require(
            _canonical_json_sha256(dedupe_identity) == dedupe_sha,
            "dual candidate dedupe identity mismatch",
        )
        worst_temperature = max(float(value) for value in finite_temperatures.values())
        require(
            math.isclose(
                worst_temperature,
                float(raw.get("worst_robust_temperature_C")),
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "dual candidate worst-temperature summary drifted",
        )
        llt_g = _finite_number(raw.get("Llt_robust_G_uH"))
        resonance_hz = _finite_number(raw.get("resonance_min_screen_Hz"))
        require(
            llt_g is not None
            and math.isclose(
                llt_g,
                float(finite_constraints["Llt_robust_band"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and resonance_hz is not None
            and resonance_hz >= float(hard_spec["resonance_min_Hz"]),
            "dual candidate robust Llt/resonance summary drifted",
        )
        require(
            raw.get("source_kind") == "sealed_successor_evidence"
            and raw.get("production_eligible") is False
            and raw.get("fea_submission_approved") is False
            and raw.get("total_positive_violation") == 0.0,
            "dual candidate attempted production or FEA authority",
        )
        return {
            "schema_version": "mft-monitor-sealed-successor-v1",
            "configured": True,
            "available": True,
            "status": "validate_only",
            "lifecycle_state": "sealed_successor",
            "source_kind": "sealed_successor_evidence",
            "display_only": True,
            "read_only": True,
            "integrity_verified": True,
            "canonical_candidate": False,
            "terminal_result": False,
            "pareto": False,
            "production": False,
            "production_eligible": False,
            "production_submission_enabled": False,
            "fea": False,
            "fea_submission_enabled": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "fea_status": "not_submitted",
            "scheduler_task_id": None,
            "evidence_id": _safe_text(raw.get("display_id"), 160),
            "display_label": _safe_text(raw.get("display_label"), 160),
            "append_only_order": _integer(raw.get("append_only_order"), 0),
            "cohort_id": SEALED_SUCCESSOR_COHORT_ID,
            "constraint_version": SEALED_SUCCESSOR_CONSTRAINT_VERSION,
            "hard_spec": hard_spec,
            "model_identity": model_identity,
            "candidate": {
                "candidate_sha256": artifact_sha,
                "decoded_fea_param_count": 71,
                "decoded_fea_params_sha256": params_sha,
                "dedupe_identity_sha256": dedupe_sha,
                "decoded_geometry": geometry,
                "exterior_box_mm": exterior_values,
                "objective_volume_L": None,
                "objective_total_loss_W": None,
                "resonance": {
                    "screen_Hz": resonance_hz,
                    "minimum_Hz": hard_spec["resonance_min_Hz"],
                    "pass": True,
                },
                "Llt_robust": {
                    "mu_uH": None,
                    "q90_half_width_uH": None,
                    "G_uH": llt_g,
                    "target_uH": hard_spec["Llt_target_uH"],
                    "tolerance_uH": hard_spec["Llt_tol_uH"],
                    "pass": True,
                },
                "robust_temperatures_C": finite_temperatures,
                "worst_temperature_C": worst_temperature,
                "temperature_limit_C": temperature_limit,
                "constraints_G": finite_constraints,
                "all_constraints_pass": True,
                "total_positive_violation": 0.0,
            },
            "integrity": {
                **integrity,
                "candidate_sha256": artifact_sha,
                "decoded_fea_params_sha256": params_sha,
                "dedupe_identity_sha256": dedupe_sha,
                "all22_all11_and_71_params_verified": True,
            },
            "warning": None,
        }

    @staticmethod
    def _dual_validation_candidate_unavailable(
        *, configured: bool, warning: str | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "mft-monitor-standard-fea-validation-v1",
            "configured": configured,
            "available": False,
            "integrity_verified": False,
            "read_only": True,
            "runtime_state": None,
            "submission_state": None,
            "task_status": "rejected" if configured else "not_configured",
            "collection_state": None,
            "task": {"id": None, "name": None, "status": None},
            "result_identity": {
                "available": False,
                "path": None,
                "sha256": None,
                "candidate_digest": None,
                "task_id": None,
                "contract_valid": False,
                "candidate_identity_matches": False,
            },
            "collected": False,
            "valid": False,
            "actual_gates": None,
            "actual_hard_gates_pass": None,
            "full_model_validation_candidate_eligible": False,
            "pareto_eligible": False,
            "production_eligible": False,
            "warning": warning,
        }

    @staticmethod
    def _dual_validation_actual_gate(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("standard-FEA actual gate is not an object")
        if (
            raw.get("contract")
            != "mft-tier1-actual-Llt-and-conditional-all11-T120-v1"
            or raw.get("Llt_target_uH")
            != DUAL_VALIDATION_LLT_TARGET_UH
            or raw.get("Llt_tolerance_uH")
            != DUAL_VALIDATION_LLT_TOLERANCE_UH
            or raw.get("temperature_limit_C")
            != DUAL_VALIDATION_TEMPERATURE_LIMIT_C
            or raw.get("temperature_contract")
            != DUAL_VALIDATION_TEMPERATURE_CONTRACT
            or not isinstance(raw.get("pass"), bool)
            or not isinstance(raw.get("reasons"), list)
            or any(not isinstance(item, str) for item in raw["reasons"])
        ):
            raise ValueError("standard-FEA actual gate contract drifted")
        actual = raw.get("actual")
        if not isinstance(actual, dict):
            raise ValueError("standard-FEA actual gate payload is missing")
        llt_full = _finite_number(actual.get("Llt_full_uH"))
        raw_side_turns = actual.get("N2_side")
        side_turns = (
            raw_side_turns
            if isinstance(raw_side_turns, int)
            and not isinstance(raw_side_turns, bool)
            else -1
        )
        temperatures = actual.get("temperatures")
        if (
            llt_full is None
            or side_turns < 0
            or not isinstance(temperatures, dict)
            or set(temperatures) != set(SEALED_SUCCESSOR_TEMPERATURE_TARGETS)
        ):
            raise ValueError("standard-FEA actual all11 inventory drifted")
        reasons: list[str] = []
        if abs(llt_full - DUAL_VALIDATION_LLT_TARGET_UH) > (
            DUAL_VALIDATION_LLT_TOLERANCE_UH + 1e-9
        ):
            reasons.append("Llt_actual_band")
        normalized_temperatures: dict[str, dict[str, Any]] = {}
        for name in SEALED_SUCCESSOR_TEMPERATURE_TARGETS:
            item = temperatures.get(name)
            if not isinstance(item, dict) or not isinstance(
                item.get("applicable"), bool
            ):
                raise ValueError(
                    f"standard-FEA actual temperature gate drifted: {name}"
                )
            expected_applicable = not (
                name in DUAL_VALIDATION_SIDE_TEMPERATURE_TARGETS
                and side_turns == 0
            )
            if item["applicable"] is not expected_applicable:
                raise ValueError(
                    "standard-FEA actual temperature applicability "
                    f"drifted: {name}"
                )
            value = _finite_number(item.get("value_C"))
            if item["applicable"] and value is None:
                raise ValueError(
                    f"standard-FEA actual temperature is missing: {name}"
                )
            if not item["applicable"] and value is not None:
                raise ValueError(
                    f"disabled standard-FEA temperature has a value: {name}"
                )
            temperature_pass = (
                value <= DUAL_VALIDATION_TEMPERATURE_LIMIT_C + 1e-9
                if item["applicable"] and value is not None else None
            )
            if temperature_pass is False:
                reasons.append(f"temperature_actual_limit:{name}")
            normalized_temperatures[name] = {
                "applicable": item["applicable"],
                "value_C": value,
                "pass": temperature_pass,
            }
        recomputed_pass = not reasons
        if raw["pass"] is not recomputed_pass or raw["reasons"] != reasons:
            raise ValueError(
                "standard-FEA declared actual gate disagrees with canonical "
                "T120 recomputation"
            )
        return {
            "contract": raw["contract"],
            "pass": recomputed_pass,
            "reasons": reasons,
            "Llt_target_uH": DUAL_VALIDATION_LLT_TARGET_UH,
            "Llt_tolerance_uH": DUAL_VALIDATION_LLT_TOLERANCE_UH,
            "temperature_limit_C": DUAL_VALIDATION_TEMPERATURE_LIMIT_C,
            "temperature_contract": DUAL_VALIDATION_TEMPERATURE_CONTRACT,
            "actual": {
                "Llt_full_uH": llt_full,
                "N2_side": side_turns,
                "temperatures": normalized_temperatures,
            },
        }

    def _dual_standard_fea_validation(
        self, sealed_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Read one exact two-candidate validation runtime without authority.

        The mutable state/status/result files are accepted only below a
        hash-bound runtime manifest and are cross-checked against both sealed
        candidates.  Any missing or inconsistent identity removes the entire
        overlay while leaving the immutable sealed evidence untouched.
        """

        configured = self._dual_validation_root is not None
        unavailable = {
            item["candidate"]["candidate_sha256"]:
            self._dual_validation_candidate_unavailable(
                configured=configured, warning=None,
            )
            for item in sealed_candidates
        }
        base = {
            "schema_version": "mft-monitor-dual-standard-fea-runtime-v1",
            "configured": configured,
            "available": False,
            "integrity_verified": False,
            "read_only": True,
            "candidate_count": 0,
            "candidates": unavailable,
            "runtime_root": (
                str(self._dual_validation_root)
                if self._dual_validation_root is not None else None
            ),
            "runtime_state": None,
            "updated_at": None,
            "integrity": None,
            "warning": None,
        }
        if not configured:
            return base
        try:
            def require(condition: bool, message: str) -> None:
                if not condition:
                    raise ValueError(message)

            def is_reparse_point(path: Path) -> bool:
                attributes = getattr(
                    path.lstat(), "st_file_attributes", 0
                )
                return bool(path.is_symlink() or attributes & 0x400)

            for value, label in (
                (self._dual_validation_manifest_sha256, "manifest"),
                (self._dual_validation_plan_sha256, "execution plan"),
                (self._dual_validation_source_sha256, "plan source"),
            ):
                require(
                    re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                    f"dual standard-FEA {label} SHA-256 is malformed",
                )
            raw_root = self._dual_validation_root
            raw_allowed_root = self._dual_validation_allowed_root
            require(
                raw_root is not None
                and raw_root.is_absolute()
                and raw_allowed_root.is_absolute(),
                "dual standard-FEA runtime/allowed root must be absolute",
            )
            require(
                not is_reparse_point(raw_root)
                and not is_reparse_point(raw_allowed_root),
                "dual standard-FEA runtime/allowed root cannot be a reparse "
                "point",
            )
            allowed_root = raw_allowed_root.resolve(strict=True)
            root = raw_root.resolve(strict=True)
            require(root.is_dir(), "dual standard-FEA runtime is not a directory")
            require(
                root != allowed_root and root.is_relative_to(allowed_root),
                "dual standard-FEA runtime escapes the allowed root",
            )

            def direct_runtime_path(
                path: Path, *, label: str, directory: bool,
            ) -> Path:
                require(
                    path.is_absolute() and path.is_relative_to(root),
                    f"dual standard-FEA {label} escapes the runtime",
                )
                relative = path.relative_to(root)
                current = root
                for part in relative.parts:
                    current /= part
                    require(
                        not is_reparse_point(current),
                        f"dual standard-FEA {label} uses a reparse point",
                    )
                resolved = path.resolve(strict=True)
                require(
                    resolved == path and resolved.is_relative_to(root),
                    f"dual standard-FEA {label} redirects outside its "
                    "declared path",
                )
                require(
                    resolved.is_dir() if directory else resolved.is_file(),
                    f"dual standard-FEA {label} has the wrong file type",
                )
                return resolved

            manifest_path = direct_runtime_path(
                root / "manifest.json",
                label="runtime manifest",
                directory=False,
            )
            manifest, _, manifest_sha = self._hash_bound_json(
                {
                    "path": str(manifest_path),
                    "sha256": self._dual_validation_manifest_sha256,
                },
                label="dual standard-FEA runtime manifest",
                max_bytes=128 * 1024,
            )
            source_path = Path(
                str(manifest.get("experimental_source") or "")
            )
            require(
                source_path.is_absolute(),
                "dual standard-FEA plan source path is not absolute",
            )
            require(
                source_path
                == root / "sealed-plan-source" / "experimental_source.json",
                "dual standard-FEA plan source escapes the runtime",
            )
            source_path = direct_runtime_path(
                source_path,
                label="sealed plan source",
                directory=False,
            )
            source, _, source_sha = self._hash_bound_json(
                {
                    "path": str(source_path),
                    "sha256": self._dual_validation_source_sha256,
                },
                label="dual standard-FEA sealed plan source",
                max_bytes=32 * 1024,
            )
            require(
                manifest.get("schema_version")
                == "mft-continuous-fea-validation-v1"
                and manifest.get("backend") == "standalone"
                and manifest.get("task_prefix") == "mft-nsgafea"
                and manifest.get("controller_role")
                == "NSGA candidate standard-FEA validation"
                and manifest.get("lane_cap") == 28
                and manifest.get("priority") == 5
                and manifest.get("submission_enabled") is True
                and manifest.get("production_gate_bypass") is False
                and manifest.get("scheduler_orphan_adoption_enabled") is False
                and manifest.get("solver_revision")
                == DUAL_VALIDATION_SOLVER_REVISION
                and manifest.get("library_revision")
                == DUAL_VALIDATION_LIBRARY_REVISION,
                "dual standard-FEA runtime manifest contract drifted",
            )
            require(
                source.get("schema_version")
                == "mft-tier1-t120-active-learning-source-snapshot-v2"
                and source.get("plan_sha256")
                == self._dual_validation_plan_sha256
                and source.get("production_eligible") is False,
                "dual standard-FEA execution plan identity drifted",
            )

            state_path = direct_runtime_path(
                root / "state.json", label="state", directory=False
            )
            status_path = direct_runtime_path(
                root / "status.json", label="status", directory=False
            )
            stable = None
            for _ in range(4):
                state, state_raw = _bounded_json_bytes(
                    state_path, DUAL_VALIDATION_STATE_MAX_BYTES
                )
                status, status_raw = _bounded_json_bytes(
                    status_path, DUAL_VALIDATION_STATUS_MAX_BYTES
                )
                state_after, state_after_raw = _bounded_json_bytes(
                    state_path, DUAL_VALIDATION_STATE_MAX_BYTES
                )
                status_after, status_after_raw = _bounded_json_bytes(
                    status_path, DUAL_VALIDATION_STATUS_MAX_BYTES
                )
                if (
                    state_raw == state_after_raw
                    and status_raw == status_after_raw
                ):
                    stable = (
                        state_after, state_after_raw,
                        status_after, status_after_raw,
                    )
                    break
            require(
                stable is not None,
                "dual standard-FEA state/status changed during read",
            )
            state, state_raw, status, status_raw = stable
            require(
                state.get("schema_version")
                == "mft-continuous-fea-validation-v1"
                and status.get("schema_version")
                == "mft-continuous-fea-validation-v1",
                "dual standard-FEA state/status schema drifted",
            )
            status_paths = status.get("paths")
            status_paths = status_paths if isinstance(status_paths, dict) else {}
            require(
                status_paths.get("manifest") == str(manifest_path)
                and status_paths.get("state") == str(state_path)
                and status_paths.get("status") == str(status_path)
                and status_paths.get("results") == str(root / "results")
                and status.get("production_gate_bypass") is False
                and status.get("scheduler_orphan_adoption_enabled") is False,
                "dual standard-FEA status path/authority drifted",
            )
            state_candidates = state.get("candidates")
            require(
                isinstance(state_candidates, dict)
                and set(state_candidates)
                == set(DUAL_VALIDATION_CANDIDATE_DIGESTS.values()),
                "dual standard-FEA candidate inventory drifted",
            )
            counts = status.get("counts")
            counts = counts if isinstance(counts, dict) else {}
            require(
                counts.get("discovered") == 2
                and counts.get("experimental") == 2
                and counts.get("production_model") == 0,
                "dual standard-FEA status counts drifted",
            )
            active_tasks = status.get("active_tasks")
            require(
                isinstance(active_tasks, list),
                "dual standard-FEA active-task inventory is missing",
            )
            active_by_digest: dict[str, dict[str, Any]] = {}
            for item in active_tasks:
                require(
                    isinstance(item, dict)
                    and item.get("candidate_digest")
                    in DUAL_VALIDATION_CANDIDATE_DIGESTS.values()
                    and item.get("candidate_digest") not in active_by_digest,
                    "dual standard-FEA active-task identity drifted",
                )
                active_by_digest[item["candidate_digest"]] = item

            overlays: dict[str, dict[str, Any]] = {}
            candidate_root = direct_runtime_path(
                root / "candidates",
                label="candidate directory",
                directory=True,
            )
            result_root = direct_runtime_path(
                root / "results",
                label="result directory",
                directory=True,
            )
            for sealed in sealed_candidates:
                sealed_design = sealed.get("candidate")
                sealed_design = (
                    sealed_design if isinstance(sealed_design, dict) else {}
                )
                artifact_sha = str(
                    sealed_design.get("candidate_sha256") or ""
                ).lower()
                digest = DUAL_VALIDATION_CANDIDATE_DIGESTS.get(artifact_sha)
                require(
                    digest is not None,
                    "dual standard-FEA sealed candidate identity is unknown",
                )
                record = state_candidates.get(digest)
                require(
                    isinstance(record, dict),
                    "dual standard-FEA candidate state is missing",
                )
                candidate_path = direct_runtime_path(
                    candidate_root / f"{digest}.json",
                    label=f"candidate record {digest}",
                    directory=False,
                )
                candidate_file, candidate_raw = _bounded_json_bytes(
                    candidate_path, 512 * 1024
                )
                require(
                    all(
                        candidate_file.get(key) == record.get(key)
                        for key in (
                            "candidate_digest",
                            "adapter_rank",
                            "append_only_order",
                            "decoded_params_sha256",
                            "source_candidate_sha256",
                            "dual_dedupe_identity_sha256",
                            "sealed_plan_sha256",
                            "source_artifact_sha256",
                            "source_kind",
                            "active_learning_only",
                            "production_eligible",
                            "pareto_eligible",
                            "fea_submission_approved",
                            "solver_revision",
                            "library_revision",
                            "params",
                        )
                    ),
                    "dual standard-FEA candidate/state identity drifted",
                )
                expected_params_sha = str(
                    sealed_design.get("decoded_fea_params_sha256") or ""
                ).lower()
                expected_dedupe_sha = str(
                    sealed_design.get("dedupe_identity_sha256") or ""
                ).lower()
                source_result = record.get("source_result")
                source_result = (
                    source_result if isinstance(source_result, dict) else {}
                )
                source_record = record.get("source_record")
                source_record = (
                    source_record if isinstance(source_record, dict) else {}
                )
                require(
                    record.get("candidate_digest") == digest
                    and record.get("adapter_rank")
                    == sealed.get("append_only_order") - 1
                    and record.get("append_only_order")
                    == sealed.get("append_only_order")
                    and record.get("decoded_params_sha256")
                    == expected_params_sha
                    and record.get("source_candidate_sha256") == artifact_sha
                    and record.get("dual_dedupe_identity_sha256")
                    == expected_dedupe_sha
                    and record.get("sealed_plan_sha256")
                    == self._dual_validation_plan_sha256
                    and record.get("source_artifact_sha256")
                    == DUAL_SUCCESSOR_PLAN_SHA256
                    and source_result.get("sha256")
                    == DUAL_SUCCESSOR_PLAN_SHA256
                    and source_record.get("sha256")
                    == DUAL_VALIDATION_HANDOFF_SHA256
                    and record.get("source_kind")
                    == "sealed_successor_evidence"
                    and record.get("active_learning_only") is True
                    and record.get("approval_label")
                    == "EXPERIMENTAL / NOT-APPROVED"
                    and record.get("model_lane") == "experimental"
                    and record.get("production_eligible") is False
                    and record.get("production_eligible_model") is False
                    and record.get("pareto_eligible") is False
                    and record.get("fea_submission_approved") is False
                    and record.get("solver_revision")
                    == DUAL_VALIDATION_SOLVER_REVISION
                    and record.get("library_revision")
                    == DUAL_VALIDATION_LIBRARY_REVISION
                    and _canonical_json_sha256(record.get("params"))
                    == expected_params_sha,
                    "dual standard-FEA candidate provenance drifted",
                )
                task_id = record.get("task_id")
                task_name = record.get("task_name")
                task_status = record.get("task_status")
                if task_id is None:
                    require(
                        task_name is None and task_status is None,
                        "dual standard-FEA unsubmitted task identity drifted",
                    )
                else:
                    require(
                        isinstance(task_id, int)
                        and not isinstance(task_id, bool)
                        and task_id > 0
                        and task_name
                        == f"mft-nsgafea-x-{digest[:16]}"
                        and task_status in DUAL_VALIDATION_TASK_STATUSES,
                        "dual standard-FEA task identity/status drifted",
                    )
                active = active_by_digest.get(digest)
                if task_status in DUAL_VALIDATION_ACTIVE_TASK_STATUSES:
                    require(
                        active is not None
                        and active.get("task_id") == task_id
                        and active.get("task_name") == task_name
                        and active.get("status") == task_status,
                        "dual standard-FEA active task cross-check drifted",
                    )
                else:
                    require(
                        active is None,
                        "dual standard-FEA terminal task remains active",
                    )

                result_identity = {
                    "available": False,
                    "path": None,
                    "sha256": None,
                    "candidate_digest": digest,
                    "task_id": task_id,
                    "contract_valid": False,
                    "candidate_identity_matches": False,
                }
                actual_gates = None
                measurements = None
                result_path_text = record.get("result_path")
                if result_path_text:
                    result_path = Path(str(result_path_text))
                    require(
                        result_path.is_absolute(),
                        "dual standard-FEA result path is not absolute",
                    )
                    require(
                        result_path == result_root / f"task-{task_id}.json",
                        "dual standard-FEA result escapes the runtime",
                    )
                    result_path = direct_runtime_path(
                        result_path,
                        label=f"task {task_id} result",
                        directory=False,
                    )
                    result_document, result_raw = _bounded_json_bytes(
                        result_path, DUAL_VALIDATION_RESULT_MAX_BYTES
                    )
                    contract_valid = record.get("result_contract_valid")
                    identity_matches = record.get(
                        "candidate_identity_matches"
                    )
                    require(
                        isinstance(contract_valid, bool)
                        and identity_matches is True
                        and result_document.get("schema_version")
                        == "mft-continuous-fea-validation-v1"
                        and result_document.get("candidate_digest") == digest
                        and result_document.get("task_id") == task_id
                        and result_document.get("result_contract_valid")
                        is contract_valid
                        and result_document.get("candidate_identity_matches")
                        is True
                        and result_document.get("approval_label")
                        == "EXPERIMENTAL / NOT-APPROVED"
                        and result_document.get("final_design_approved")
                        is False
                        and result_document.get("pareto_eligible") is False
                        and result_document.get("production_eligible") is False,
                        "dual standard-FEA result identity/classification drifted",
                    )
                    record_gate = record.get("tier1_t120_actual_gate")
                    document_gate = result_document.get(
                        "tier1_t120_actual_gate"
                    )
                    require(
                        record_gate is not None
                        and record_gate == document_gate,
                        "dual standard-FEA actual gate snapshot drifted",
                    )
                    actual_gates = self._dual_validation_actual_gate(
                        record_gate
                    )
                    raw_result = result_document.get("result")
                    raw_result = (
                        raw_result if isinstance(raw_result, dict) else {}
                    )
                    params = record.get("params")
                    params = params if isinstance(params, dict) else {}
                    expected_side_turns = params.get("N2_side")
                    require(
                        isinstance(expected_side_turns, int)
                        and not isinstance(expected_side_turns, bool)
                        and expected_side_turns >= 0
                        and actual_gates["actual"]["N2_side"]
                        == expected_side_turns,
                        "dual standard-FEA actual side-turn identity drifted",
                    )
                    raw_side_turns = _finite_number(
                        raw_result.get("N2_side")
                    )
                    raw_llt = _finite_number(raw_result.get("Llt"))
                    raw_full_model = _finite_number(
                        raw_result.get("full_model")
                    )
                    require(
                        raw_side_turns is not None
                        and raw_side_turns.is_integer()
                        and int(raw_side_turns) == expected_side_turns
                        and raw_llt is not None
                        and raw_full_model in {0.0, 1.0},
                        "dual standard-FEA actual result identity drifted",
                    )
                    expected_full_llt = raw_llt * (
                        2.0 if raw_full_model == 0.0 else 1.0
                    )
                    require(
                        math.isclose(
                            actual_gates["actual"]["Llt_full_uH"],
                            expected_full_llt,
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        ),
                        "dual standard-FEA actual Llt snapshot drifted",
                    )
                    for name, gate in actual_gates["actual"][
                        "temperatures"
                    ].items():
                        if gate["applicable"]:
                            result_temperature = _finite_number(
                                raw_result.get(name)
                            )
                            require(
                                result_temperature is not None
                                and math.isclose(
                                    gate["value_C"],
                                    result_temperature,
                                    rel_tol=0.0,
                                    abs_tol=1e-9,
                                ),
                                "dual standard-FEA actual temperature "
                                f"snapshot drifted: {name}",
                            )
                    finite_temperatures = [
                        value for value in (
                            _finite_number(raw_result.get(name))
                            for name in SEALED_SUCCESSOR_TEMPERATURE_TARGETS
                        ) if value is not None
                    ]
                    measurements = {
                        "Llt_raw_uH": raw_llt,
                        "Llt_full_uH": (
                            actual_gates["actual"]["Llt_full_uH"]
                            if actual_gates is not None else None
                        ),
                        "full_model": int(raw_full_model),
                        "maximum_temperature_C": (
                            max(finite_temperatures)
                            if finite_temperatures else None
                        ),
                        "B_design_T": _finite_number(
                            raw_result.get(
                                "B_design_square_material_analytic"
                            )
                        ),
                        "P_core_total_W": _finite_number(
                            raw_result.get("P_core_total")
                        ),
                        "P_winding_total_W": _finite_number(
                            raw_result.get("P_winding_total")
                        ),
                    }
                    result_identity = {
                        "available": True,
                        "path": str(result_path),
                        "sha256": hashlib.sha256(result_raw).hexdigest(),
                        "candidate_digest": digest,
                        "task_id": task_id,
                        "contract_valid": contract_valid,
                        "candidate_identity_matches": True,
                    }
                else:
                    require(
                        record.get("result_contract_valid") is not True
                        and record.get("candidate_identity_matches") is not True
                        and record.get("tier1_t120_actual_gate") is None,
                        "dual standard-FEA result flags exist without a result",
                    )
                collection_state = _safe_text(
                    record.get("collection_state"), 100
                )
                collected = collection_state == "collector_succeeded"
                valid = bool(
                    task_status == "completed"
                    and collected
                    and isinstance(task_id, int)
                    and not isinstance(task_id, bool)
                    and task_id > 0
                    and result_identity["available"]
                    and result_identity["contract_valid"]
                    and result_identity["candidate_identity_matches"]
                )
                actual_pass = (
                    actual_gates["pass"]
                    if actual_gates is not None else None
                )
                overlays[artifact_sha] = {
                    "schema_version": (
                        "mft-monitor-standard-fea-validation-v1"
                    ),
                    "configured": True,
                    "available": True,
                    "integrity_verified": True,
                    "read_only": True,
                    "runtime_state": _safe_text(status.get("state"), 100),
                    "submission_state": _safe_text(
                        record.get("submission_state"), 100
                    ),
                    "task_status": task_status or "not_submitted",
                    "collection_state": collection_state,
                    "task": {
                        "id": task_id,
                        "name": task_name,
                        "status": task_status,
                    },
                    "candidate_digest": digest,
                    "result_identity": result_identity,
                    "measurements": measurements,
                    "collected": collected,
                    "valid": valid,
                    "actual_gates": actual_gates,
                    "actual_hard_gates_pass": actual_pass,
                    "full_model_validation_candidate_eligible": bool(
                        valid
                        and actual_pass is True
                        and record.get(
                            "full_model_validation_candidate_eligible"
                        ) is True
                    ),
                    "pareto_eligible": False,
                    "production_eligible": False,
                    "updated_at": _safe_text(
                        record.get("terminal_at")
                        or record.get("submitted_at")
                        or state.get("updated_at"),
                        100,
                    ),
                    "integrity": {
                        "manifest_sha256": manifest_sha,
                        "execution_plan_sha256": (
                            self._dual_validation_plan_sha256
                        ),
                        "state_sha256": hashlib.sha256(
                            state_raw
                        ).hexdigest(),
                        "status_sha256": hashlib.sha256(
                            status_raw
                        ).hexdigest(),
                        "candidate_record_sha256": hashlib.sha256(
                            candidate_raw
                        ).hexdigest(),
                    },
                    "warning": _safe_text(status.get("last_error"), 500),
                }
            require(
                set(overlays)
                == set(DUAL_VALIDATION_CANDIDATE_DIGESTS),
                "dual standard-FEA overlay candidate mapping drifted",
            )
            terminal_statuses = DUAL_VALIDATION_TASK_STATUSES.difference(
                DUAL_VALIDATION_ACTIVE_TASK_STATUSES
            )
            require(
                counts.get("active") == len(active_by_digest)
                and counts.get("terminal") == sum(
                    item["task_status"] in terminal_statuses
                    for item in overlays.values()
                )
                and counts.get("valid_result") == sum(
                    item["valid"] for item in overlays.values()
                ),
                "dual standard-FEA collector validity counts drifted",
            )
            return {
                **base,
                "available": True,
                "integrity_verified": True,
                "candidate_count": 2,
                "candidates": overlays,
                "runtime_root": str(root),
                "runtime_state": _safe_text(status.get("state"), 100),
                "updated_at": _safe_text(status.get("updated_at"), 100),
                "integrity": {
                    "manifest_sha256": manifest_sha,
                    "sealed_plan_source_sha256": source_sha,
                    "execution_plan_sha256": (
                        self._dual_validation_plan_sha256
                    ),
                    "state_sha256": hashlib.sha256(state_raw).hexdigest(),
                    "status_sha256": hashlib.sha256(status_raw).hexdigest(),
                    "allowed_root": str(allowed_root),
                },
                "warning": _safe_text(status.get("last_error"), 500),
            }
        except (OSError, TypeError, ValueError) as exc:
            warning = f"dual standard-FEA overlay rejected: {exc}"
            return {
                **base,
                "candidates": {
                    key: self._dual_validation_candidate_unavailable(
                        configured=True, warning=warning,
                    )
                    for key in unavailable
                },
                "warning": warning,
            }

    def _dual_sealed_successors(self) -> dict[str, Any]:
        paths = (
            self._dual_successor_plan,
            self._dual_successor_ui,
            self._dual_successor_receipt,
        )
        if all(path is None for path in paths):
            return self._dual_successors_rejected()
        if any(path is None for path in paths):
            return self._dual_successors_rejected(
                "dual sealed successor requires plan, UI, and validation receipt"
            )
        try:
            plan, plan_path, plan_sha = self._hash_bound_json(
                {"path": str(self._dual_successor_plan), "sha256": self._dual_successor_plan_sha256},
                label="dual successor ValidateOnly plan",
                max_bytes=512 * 1024,
            )
            ui, ui_path, ui_sha = self._hash_bound_json(
                {"path": str(self._dual_successor_ui), "sha256": self._dual_successor_ui_sha256},
                label="dual successor read-only UI handoff",
                max_bytes=128 * 1024,
            )
            receipt, _, receipt_sha = self._hash_bound_json(
                {"path": str(self._dual_successor_receipt), "sha256": self._dual_successor_receipt_sha256},
                label="dual successor validation receipt",
                max_bytes=128 * 1024,
            )

            def require(condition: bool, message: str) -> None:
                if not condition:
                    raise ValueError(message)

            false_plan_fields = (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
                "scheduler_mutation_performed",
                "live_pointer_mutation_performed",
                "monitor_8010_mutation_performed",
                "scheduler_8002_mutation_performed",
            )
            policy = plan.get("submission_policy")
            policy = policy if isinstance(policy, dict) else {}
            release = plan.get("release")
            release = release if isinstance(release, dict) else {}
            release_source = release.get("source")
            release_source = release_source if isinstance(release_source, dict) else {}
            require(
                plan.get("schema_version")
                == "mft-tier1-t120-dual-successor-validateonly-plan-v1"
                and plan.get("mode") == "ValidateOnly"
                and plan.get("cohort_id") == SEALED_SUCCESSOR_COHORT_ID
                and plan.get("constraint_version")
                == SEALED_SUCCESSOR_CONSTRAINT_VERSION
                and plan.get("candidate_count") == 2
                and plan.get("maximum_submission_count") == 2,
                "dual successor plan identity/count mismatch",
            )
            require(
                release.get("revision") == DUAL_SUCCESSOR_RELEASE_REVISION
                and release.get("git_dirty") is False
                and release_source.get("sha256")
                == DUAL_SUCCESSOR_RELEASE_SOURCE_SHA256,
                "dual successor validator release drifted",
            )
            hard_spec = plan.get("hard_spec")
            require(
                isinstance(hard_spec, dict)
                and _canonical_json_sha256(hard_spec)
                == DUAL_SUCCESSOR_HARD_SPEC_SHA256
                and plan.get("hard_spec_sha256")
                == DUAL_SUCCESSOR_HARD_SPEC_SHA256
                and plan.get("temperature_constraint_contract_sha256")
                == DUAL_SUCCESSOR_TEMPERATURE_CONTRACT_SHA256,
                "dual successor hard-spec/temperature contract drifted",
            )
            temperature_contract = plan.get("temperature_constraint_contract")
            require(
                isinstance(temperature_contract, dict)
                and temperature_contract.get("target_count") == 11
                and tuple(temperature_contract.get("targets") or ())
                == SEALED_SUCCESSOR_TEMPERATURE_TARGETS,
                "dual successor all11 temperature contract drifted",
            )
            require(
                policy.get("validate_only") is True
                and policy.get("pointer_cutover_verified") is True
                and policy.get("maximum_submission_count") == 2
                and policy.get("execution_ready") is False
                and policy.get("operator_authorization_required") is True
                and policy.get("fresh_cap28_inventory_recheck_required") is True
                and all(plan.get(key) is False for key in false_plan_fields),
                "dual successor plan attempted live authority",
            )
            pointer = plan.get("pointer_cutover")
            pointer = pointer if isinstance(pointer, dict) else {}
            capacity = plan.get("capacity_contract")
            capacity = capacity if isinstance(capacity, dict) else {}
            require(
                pointer.get("pointer_switch_verified") is True
                and pointer.get("cohort_id") == SEALED_SUCCESSOR_COHORT_ID
                and pointer.get("constraint_version")
                == SEALED_SUCCESSOR_CONSTRAINT_VERSION
                and capacity.get("global_lane_cap") == 28
                and capacity.get("observe_only_passed") is True
                and capacity.get("fresh_execution_time_recheck_required") is True,
                "dual successor pointer/cap28 ValidateOnly evidence drifted",
            )
            require(
                ui.get("schema_version")
                == "mft-tier1-t120-dual-successor-readonly-ui-v1"
                and ui.get("source_plan")
                == {"path": str(plan_path), "bytes": plan_path.stat().st_size, "sha256": plan_sha}
                and ui.get("section") == "sealed_successor_evidence"
                and ui.get("read_only") is True
                and ui.get("display_as_scheduler_terminal") is False
                and ui.get("display_as_near_or_pareto") is False
                and ui.get("candidate_count") == 2
                and ui.get("production_eligible") is False
                and ui.get("fea_submission_approved") is False
                and ui.get("fea_submission_performed") is False,
                "dual successor UI handoff attempted canonical authority",
            )
            require(
                receipt.get("schema_version")
                == "mft-tier1-t120-dual-successor-validation-receipt-v1"
                and receipt.get("plan")
                == {"path": str(plan_path), "bytes": plan_path.stat().st_size, "sha256": plan_sha}
                and receipt.get("read_only_ui")
                == {"path": str(ui_path), "bytes": ui_path.stat().st_size, "sha256": ui_sha}
                and receipt.get("validation_passed") is True
                and receipt.get("pointer_cutover_verified") is True
                and receipt.get("candidate_count") == 2
                and receipt.get("maximum_submission_count") == 2
                and receipt.get("all22_all11_and_71_params_verified") is True
                and receipt.get("distinct_dedupe_verified") is True
                and receipt.get("live_mutation_performed") is False
                and receipt.get("fea_submission_performed") is False,
                "dual successor validation receipt drifted",
            )
            raw_candidates = plan.get("candidates")
            ui_candidates = ui.get("candidates")
            require(
                isinstance(raw_candidates, list)
                and isinstance(ui_candidates, list)
                and len(raw_candidates) == len(ui_candidates) == 2
                and [item.get("append_only_order") for item in raw_candidates]
                == [1, 2],
                "dual successor append-only candidate inventory drifted",
            )
            artifacts = [
                str(item.get("artifact_sha256") or "").lower()
                for item in raw_candidates
            ]
            require(
                artifacts
                == [DUAL_SUCCESSOR_EXACT_SHA256, DUAL_SUCCESSOR_INTERIOR_SHA256],
                "dual successor exact/interior artifact order drifted",
            )
            ui_identity = [
                (
                    item.get("display_id"),
                    item.get("artifact_sha256"),
                    item.get("decoded_fea_params_sha256"),
                    item.get("dedupe_identity_sha256"),
                )
                for item in ui_candidates
            ]
            plan_identity = [
                (
                    item.get("display_id"),
                    item.get("artifact_sha256"),
                    item.get("decoded_fea_params_sha256"),
                    item.get("dedupe_identity_sha256"),
                )
                for item in raw_candidates
            ]
            require(
                ui_identity == plan_identity and len(set(plan_identity)) == 2,
                "dual successor UI/plan candidate identity drifted",
            )
            model_identity = {
                "source_model_manifest_sha256": plan.get(
                    "source_model_manifest_sha256"
                ),
                "deployment_model_manifest_sha256": plan.get(
                    "deployment_model_manifest_sha256"
                ),
                "validator_release_revision": release.get("revision"),
            }
            integrity = {
                "validateonly_plan_sha256": plan_sha,
                "read_only_ui_sha256": ui_sha,
                "validation_receipt_sha256": receipt_sha,
                "validation_receipt_verified": True,
                "pointer_cutover_verified": True,
                "cap28_observe_only_verified": True,
            }
            candidates = [
                self._dual_successor_candidate(
                    raw,
                    hard_spec=hard_spec,
                    integrity=integrity,
                    model_identity=model_identity,
                )
                for raw in raw_candidates
            ]
            require(
                len({item["candidate"]["decoded_fea_params_sha256"] for item in candidates}) == 2
                and len({item["candidate"]["dedupe_identity_sha256"] for item in candidates}) == 2,
                "dual successor candidates are not distinct",
            )
            standard_fea_validation = self._dual_standard_fea_validation(
                candidates
            )
            validation_by_artifact = standard_fea_validation["candidates"]
            candidates = [
                {
                    **item,
                    "standard_fea_validation": validation_by_artifact.get(
                        item["candidate"]["candidate_sha256"],
                        self._dual_validation_candidate_unavailable(
                            configured=standard_fea_validation["configured"],
                            warning=(
                                "dual standard-FEA candidate mapping is missing"
                            ),
                        ),
                    ),
                }
                for item in candidates
            ]
            return {
                "schema_version": "mft-monitor-dual-sealed-successors-v1",
                "configured": True,
                "available": True,
                "status": "validate_only",
                "source_kind": "sealed_successor_evidence",
                "display_only": True,
                "read_only": True,
                "integrity_verified": True,
                "candidate_count": 2,
                "candidates": candidates,
                "canonical_candidate": False,
                "terminal_result": False,
                "pareto": False,
                "production": False,
                "production_eligible": False,
                "fea": False,
                "fea_submission_approved": False,
                "fea_submission_performed": False,
                "scheduler_mutation_performed": False,
                "integrity": integrity,
                "standard_fea_validation": standard_fea_validation,
                "source": str(plan_path),
                "updated_at": _safe_text(receipt.get("validated_at"), 100),
                "warning": None,
            }
        except (OSError, TypeError, ValueError) as exc:
            return self._dual_successors_rejected(
                f"dual sealed successor evidence rejected: {exc}"
            )

    def _scan_pareto_artifacts(self, *, force: bool = False) -> None:
        """Index local Pareto fronts by content hash for scheduler projections.

        The scheduler intentionally exposes only compact Pareto points.  The
        monitor runs on the same workstation as the immutable NSGA artifacts,
        so the scheduler-provided SHA-256 is the safe join key for recovering
        the complete 267-column row without trusting a mutable path.
        """
        now = time.monotonic()
        with self._pareto_artifact_lock:
            if (
                not force
                and self._pareto_artifact_scan_at
                and now - self._pareto_artifact_scan_at < 30.0
            ):
                return
            discovered = dict(self._pareto_artifacts_by_sha)
            seen_paths: set[Path] = set()
            for root in self._pareto_artifact_roots:
                try:
                    paths = [root] if root.is_file() else root.rglob("pareto_front.csv")
                    for path in paths:
                        try:
                            resolved = path.resolve()
                        except OSError:
                            continue
                        if resolved in seen_paths or not resolved.is_file():
                            continue
                        seen_paths.add(resolved)
                        digest = _sha256_file(resolved)
                        if digest:
                            discovered[digest.lower()] = resolved
                except OSError:
                    continue
            self._pareto_artifacts_by_sha = discovered
            self._pareto_artifact_scan_at = now

    def _pareto_row_by_sha(
        self, digest: Any, row_number: Any
    ) -> tuple[dict[str, Any] | None, Path | None, str | None]:
        sha = (_safe_text(digest, 64) or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            return None, None, "Pareto result set has no valid SHA-256"
        ordinal = _integer(row_number, 0)
        if ordinal <= 0:
            return None, None, "Pareto point has no valid one-based row_number"

        self._scan_pareto_artifacts()
        with self._pareto_artifact_lock:
            path = self._pareto_artifacts_by_sha.get(sha)
        if path is None:
            # A worker may have committed a new run after the last periodic
            # scan.  One forced rescan keeps the UI current without polling
            # the filesystem continuously.
            self._scan_pareto_artifacts(force=True)
            with self._pareto_artifact_lock:
                path = self._pareto_artifacts_by_sha.get(sha)
        if path is None:
            return None, None, f"Hash-verified Pareto artifact not found: {sha[:12]}"
        if _sha256_file(path) != sha:
            return None, path, f"Pareto artifact changed after indexing: {path.name}"

        result = self.cache.csv(path, max_rows=20_000)
        rows = result.value if isinstance(result.value, list) else []
        if result.warning:
            return None, path, result.warning
        if ordinal > len(rows):
            return None, path, (
                f"Pareto row_number {ordinal} exceeds artifact row count {len(rows)}"
            )
        row = rows[ordinal - 1]
        return (dict(row) if isinstance(row, dict) else None), path, None

    def _audited_campaign_frame(
        self,
        result: ReadResult,
        expected_solver_revision: str | None,
        expected_library_revision: str | None,
    ) -> tuple[Any, str | None]:
        """Cache canonical per-SHA strict annotations for one Parquet read.

        Every well-formed solver SHA is passed back into the quality contract
        as the exact expected revision for that subgroup.  This keeps strict
        evidence recomputed per row while allowing one physics revision to
        span multiple clean solver rolls.  Missing or malformed identities
        are still audited, without an expected SHA, and fail provenance.
        """
        if not result.exists or result.value is None:
            return None, result.warning
        key = (
            result.path,
            result.mtime,
            id(result.value),
            expected_solver_revision,
            expected_library_revision,
        )
        with self._campaign_audit_lock:
            if key == self._campaign_audit_key:
                return self._campaign_audit_frame, self._campaign_audit_warning
            warning = None
            audit_fields = (
                "_strict_valid_em",
                "_strict_valid_thermal",
                "_strict_valid_full",
                "_strict_invalid_reasons",
            )
            try:
                from ..quality_contract import annotate_validity

                source = result.value
                if "git_hash" not in source.columns:
                    audited = annotate_validity(
                        source,
                        expected_solver_revision=None,
                        expected_library_revision=None,
                    )
                else:
                    audited = source.copy()
                    audited["_strict_valid_em"] = False
                    audited["_strict_valid_thermal"] = False
                    audited["_strict_valid_full"] = False
                    audited["_strict_invalid_reasons"] = ""
                    positions_by_revision: dict[str, list[int]] = defaultdict(list)
                    for position, value in enumerate(source["git_hash"].tolist()):
                        revision = (_optional_text(value, 160) or "").lower()
                        positions_by_revision[revision].append(position)
                    active_revision = (
                        _optional_text(expected_solver_revision, 160) or ""
                    ).lower()
                    for revision, positions in positions_by_revision.items():
                        exact_revision = (
                            revision
                            if re.fullmatch(r"[0-9a-f]{40}", revision)
                            else None
                        )
                        cohort = source.iloc[positions].copy()
                        cohort_audited = annotate_validity(
                            cohort,
                            expected_solver_revision=exact_revision,
                            expected_library_revision=(
                                expected_library_revision
                                if exact_revision == active_revision
                                else None
                            ),
                        )
                        for field in audit_fields:
                            column_position = audited.columns.get_loc(field)
                            audited.iloc[positions, column_position] = (
                                cohort_audited[field].to_numpy()
                            )
            except Exception as exc:
                # Stored validity flags remain a fail-soft operational view;
                # _campaign_frame_summary still pins them to the dynamically
                # selected active identity instead of trusting other cohorts.
                audited = result.value
                if hasattr(audited, "drop"):
                    # Never mistake persisted/stale underscore columns for a
                    # successful recomputation after the canonical audit
                    # itself failed.
                    audited = audited.drop(
                        columns=list(audit_fields), errors="ignore"
                    )
                warning = (
                    "train.parquet strict audit failed; stored validity flags "
                    "are shown for the active cohort: "
                    f"{type(exc).__name__}: {exc}"
                )
            self._campaign_audit_key = key
            self._campaign_audit_frame = audited
            self._campaign_audit_warning = warning
            return audited, warning

    def data(self) -> dict[str, Any]:
        now = self.clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        dataset_dir = self.root / "data" / "dataset"
        manifest_result = self.cache.json(dataset_dir / "manifest.json", {})
        rows_result = self.cache.csv(dataset_dir / "train_io.csv")
        parquet_result = self.cache.parquet(dataset_dir / "train.parquet")
        cache_result = self.cache.json(dataset_dir / "collect_cache.json", {})
        pipeline_root = getattr(self.continuous_pipeline, "root", None)
        canonical_training_root: Path | None = None
        if pipeline_root is not None:
            try:
                candidate_root = Path(pipeline_root).resolve() / "canonical_checkpoint"
                if candidate_root.is_dir():
                    canonical_training_root = candidate_root
            except OSError:
                canonical_training_root = None
        strict_path = (
            canonical_training_root / "strict_data_status.json"
            if canonical_training_root is not None
            else self.root / "training" / "strict_data_status.json"
        )
        strict_result = self.cache.json(
            strict_path, {}, fail_closed=True
        )
        manifest = manifest_result.value if isinstance(manifest_result.value, dict) else {}
        rows = rows_result.value if isinstance(rows_result.value, list) else []
        collector_cache = cache_result.value if isinstance(cache_result.value, dict) else {}
        strict_status = (
            strict_result.value if isinstance(strict_result.value, dict) else {}
        )
        warnings = self._warnings(
            manifest_result, rows_result, parquet_result, cache_result,
            strict_result,
        )
        if PHYSICS_DATA_REVISION_IMPORT_ERROR:
            warnings.append(
                "PHYSICS_DATA_REVISION import failed; active cohort panels "
                f"are unavailable: {PHYSICS_DATA_REVISION_IMPORT_ERROR}"
            )

        manifest_total = _integer(manifest.get("total_rows"), -1)
        parquet_rows = (
            len(parquet_result.value)
            if parquet_result.value is not None
            and hasattr(parquet_result.value, "__len__") else -1
        )
        observed_rows = parquet_rows if parquet_rows >= 0 else len(rows)
        raw_total = manifest_total if manifest_total >= 0 else observed_rows
        if rows and manifest_total >= 0 and len(rows) != manifest_total:
            warnings.append(f"manifest({manifest_total})와 train_io.csv({len(rows)}) 행 수가 다릅니다.")
        if parquet_rows >= 0 and manifest_total >= 0 and parquet_rows != manifest_total:
            warnings.append(
                f"manifest({manifest_total})와 train.parquet({parquet_rows}) 행 수가 다릅니다."
            )

        identity = strict_status.get("state_identity")
        identity = identity if isinstance(identity, dict) else {}
        status_solver_revision = (
            strict_status.get("expected_solver_revision")
            or identity.get("solver_revision")
        )
        status_library_revision = (
            strict_status.get("expected_library_revision")
            or identity.get("library_revision")
        )
        pinned_revision = str(status_solver_revision or "").strip().lower()
        pinned_library_revision = str(
            status_library_revision or ""
        ).strip().lower()
        full_pins = all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in (
            pinned_revision, pinned_library_revision
        ))
        pins_valid = bool(
            full_pins
            and checkpoint_status_revision_identity_matches is not None
            and checkpoint_status_revision_identity_matches(
                strict_status, pinned_revision, pinned_library_revision
            )
        )
        strict_available = bool(strict_result.exists and pins_valid)
        if not strict_available:
            pinned_revision = None
            pinned_library_revision = None
        training_raw_rows = self._exact_integer(strict_status.get("raw_rows"))
        training_strict_em_rows = self._exact_integer(
            strict_status.get("strict_em_rows")
        )
        training_strict_full_rows = self._exact_integer(
            strict_status.get("strict_full_rows")
        )
        training_counts_valid = bool(
            strict_available
            and training_raw_rows is not None
            and training_strict_em_rows is not None
            and training_strict_full_rows is not None
            and 0 <= training_strict_full_rows
            <= training_strict_em_rows
            <= training_raw_rows
        )
        if strict_available and not training_counts_valid:
            warnings.append(
                "pinned strict-data counts are missing or inconsistent; "
                "the training cohort is unavailable."
            )
            strict_available = False
            pinned_revision = None
            pinned_library_revision = None
        training_cohort = {
            "available": training_counts_valid,
            "count_basis": "exact_solver_library_strict_full",
            "source_kind": (
                "pipeline_canonical_checkpoint"
                if canonical_training_root is not None
                else "campaign_training_fallback"
            ),
            "raw_rows": training_raw_rows if training_counts_valid else None,
            "strict_em_rows": (
                training_strict_em_rows if training_counts_valid else None
            ),
            "strict_full_rows": (
                training_strict_full_rows if training_counts_valid else None
            ),
            "solver_revision": pinned_revision,
            "library_revision": pinned_library_revision,
            "source_dataset_generation": (
                _safe_text(strict_status.get("source_dataset_generation"), 160)
                if training_counts_valid else None
            ),
            "updated_at": (
                _safe_text(strict_status.get("time"), 80)
                if training_counts_valid else None
            ),
        }
        try:
            pipeline_truth = self.continuous_pipeline.snapshot()
        except Exception as exc:
            pipeline_truth = {
                "warnings": [
                    "continuous pipeline truth unavailable: "
                    f"{type(exc).__name__}: {exc}"
                ]
            }
        pipeline_training = (
            pipeline_truth.get("training")
            if isinstance(pipeline_truth, dict)
            and isinstance(pipeline_truth.get("training"), dict)
            else {}
        )
        latest_eligible_cohort = pipeline_training.get(
            "latest_eligible_cohort"
        )
        completed_training_snapshot = pipeline_training.get(
            "latest_completed_training_snapshot"
        )
        active_model_state = (
            pipeline_truth.get("active_model")
            if isinstance(pipeline_truth, dict)
            and isinstance(pipeline_truth.get("active_model"), dict)
            else None
        )
        raw_campaign_frame = (
            parquet_result.value
            if parquet_result.value is not None else rows
        )
        active_cohort = _active_cohort_identity(
            raw_campaign_frame, now.tzinfo, CURRENT_PHYSICS_DATA_REVISION
        )
        active_solver_revision = (
            active_cohort[0] if active_cohort is not None else None
        )
        audit_library_revision = (
            pinned_library_revision
            if (
                pinned_revision is not None
                and active_cohort is not None
                and solver_revision_matches_physics_cohort is not None
                and solver_revision_matches_physics_cohort(
                    active_solver_revision,
                    pinned_revision,
                    active_cohort[1],
                )
            ) else None
        )
        campaign_frame, audit_warning = self._audited_campaign_frame(
            parquet_result,
            active_solver_revision,
            audit_library_revision,
        )
        if campaign_frame is None:
            campaign_frame = rows
        if audit_warning and audit_warning not in warnings:
            warnings.append(audit_warning)
        if not strict_available:
            warnings.append(
                "pinned strict-data status is unavailable; the physics-revision "
                "aggregate uses row-level validity evidence instead."
            )

        simulation_timing = _simulation_timing_summary(
            campaign_frame,
            now.tzinfo,
            active_cohort=active_cohort,
            current_physics_revision=CURRENT_PHYSICS_DATA_REVISION,
        )
        campaign_summary = _campaign_frame_summary(
            campaign_frame,
            now,
            active_cohort=active_cohort,
            current_physics_revision=CURRENT_PHYSICS_DATA_REVISION,
        )
        physics_aggregate = campaign_summary["physics_revision_aggregate"]
        revision_raw_rows = _integer(physics_aggregate.get("raw_rows"), 0)
        total = _integer(physics_aggregate.get("strict_full_rows"), 0)
        em_valid = _integer(physics_aggregate.get("strict_em_rows"), 0)
        thermal_valid = complete = total
        throughput_1h = _integer(
            physics_aggregate.get("growth_rate_per_hour"), 0
        )
        added_24h = _integer(physics_aggregate.get("added_24h"), 0)
        latest_data = _parse_time(
            physics_aggregate.get("latest_strict_saved_at"), now.tzinfo
        )
        first_data = _parse_time(
            physics_aggregate.get("first_strict_saved_at"), now.tzinfo
        )
        stalled_minutes = (
            max(0.0, (now - latest_data).total_seconds() / 60.0)
            if latest_data else None
        )
        hourly_rate = float(throughput_1h)
        if hourly_rate <= 0 and added_24h:
            hourly_rate = added_24h / 24.0
        remaining = max(0, DATA_GOAL - total)
        eta_hours = remaining / hourly_rate if hourly_rate > 0 else None
        eta = now + timedelta(hours=eta_hours) if eta_hours is not None else None
        history = list(physics_aggregate.get("history") or [])

        campaign_identity_rows = _frame_records(
            campaign_frame, ("git_hash", "saved_at")
        )
        revision_rows = campaign_identity_rows or rows
        revisions = Counter(
            revision.lower()
            for row in revision_rows
            if (revision := _optional_text(row.get("git_hash"), 40))
        )
        latest_revision = None
        timed_revisions = [
            (stamp, revision)
            for row in revision_rows
            if (stamp := _parse_time(row.get("saved_at"), now.tzinfo)) is not None
            if (revision := _optional_text(row.get("git_hash"), 40))
        ]
        if timed_revisions:
            latest_revision = max(timed_revisions, key=lambda item: item[0])[1].lower()
        if not latest_revision and revisions:
            latest_revision = revisions.most_common(1)[0][0]
        revision_mismatch = (
            sum(count for revision, count in revisions.items() if revision != latest_revision)
            if latest_revision else 0
        )
        hashes = manifest.get("git_hashes") if isinstance(manifest.get("git_hashes"), list) else []
        harvested = collector_cache.get("harvested")
        nodata = collector_cache.get("nodata")
        local_parts = collector_cache.get("local_parts")

        return {
            "schema_version": SCHEMA_VERSION,
            "available": (
                manifest_result.exists or rows_result.exists
                or parquet_result.exists
            ),
            "count_basis": "physics_revision_strict_full",
            "count_bases": {
                "raw_total": "campaign_manifest_or_observed_rows",
                "training_strict_full": training_cohort["count_basis"],
                "active_physics_strict_em": "physics_revision_strict_em",
                "active_physics_strict_full": "physics_revision_strict_full",
            },
            "strict_status_available": strict_available,
            "raw_total_rows": raw_total,
            "revision_raw_rows": revision_raw_rows,
            "total_rows": total,
            "em_valid_rows": em_valid,
            "thermal_valid_rows": thermal_valid,
            "complete_rows": complete,
            "em_only_rows": max(0, em_valid - complete),
            "invalid_em_rows": max(0, revision_raw_rows - em_valid),
            "manifest_new_rows": _integer(manifest.get("new_rows"), 0),
            "manifest_new_unique_rows": _integer(manifest.get("new_unique_rows"), 0),
            "goal": DATA_GOAL,
            "stretch_goal": STRETCH_GOAL,
            "goal_progress_pct": min(100.0, total / DATA_GOAL * 100.0),
            "stretch_progress_pct": min(100.0, total / STRETCH_GOAL * 100.0),
            "remaining_to_goal": remaining,
            "throughput_1h": throughput_1h,
            "added_24h": added_24h,
            "effective_hourly_rate": hourly_rate,
            "eta_3000": _iso(eta),
            "eta_hours": eta_hours,
            "first_data_at": _iso(first_data),
            "latest_data_at": _iso(latest_data),
            "stalled_minutes": stalled_minutes,
            "stalled": bool(stalled_minutes is not None and stalled_minutes >= 90 and total < DATA_GOAL),
            "latest_revision": latest_revision,
            "pinned_revision": pinned_revision,
            "pinned_library_revision": pinned_library_revision,
            "training_cohort": training_cohort,
            "latest_eligible_cohort": latest_eligible_cohort,
            "latest_completed_training": {
                "snapshot": completed_training_snapshot,
                "job": pipeline_training.get("latest_completed_job"),
            },
            "active_training_job": pipeline_training.get("active_job"),
            "active_model": active_model_state,
            "current_solver_revision": active_solver_revision,
            "current_physics_data_revision": CURRENT_PHYSICS_DATA_REVISION,
            "member_git_hashes": physics_aggregate["member_git_hashes"],
            "member_git_hash_shorts": (
                physics_aggregate["member_git_hash_shorts"]
            ),
            "revision_count": len(revisions) or len(hashes),
            "rows_not_latest_revision": revision_mismatch,
            "rows_not_current_physics_revision": max(
                0, raw_total - revision_raw_rows
            ),
            **campaign_summary,
            "collector": {
                "harvested_tasks": len(harvested) if isinstance(harvested, list) else None,
                "no_data_tasks": len(nodata) if isinstance(nodata, list) else None,
                "local_parts": len(local_parts) if isinstance(local_parts, list) else None,
            },
            "simulation_timing": simulation_timing,
            "history": history,
            "source": {
                "manifest": str(manifest_result.path),
                "rows": str(rows_result.path),
                "parquet": str(parquet_result.path),
                "campaign_rows": (
                    str(parquet_result.path)
                    if parquet_result.exists
                    and parquet_result.value is not None
                    else str(rows_result.path)
                ),
                "strict_status": str(strict_result.path),
                "updated_at": _iso(
                    parquet_result.mtime or manifest_result.mtime
                    or rows_result.mtime or strict_result.mtime
                ),
            },
            "warnings": warnings,
        }

    @staticmethod
    def _exact_integer(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        number = _finite_number(value)
        if number is None or not number.is_integer():
            return None
        return int(number)

    @classmethod
    def _capacitance_parity_provenance(
            cls, recovery: Any = None, *, n_rows: int | None = None,
            pairs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Validate the LC-recovery evidence required by C_* parity plots.

        Older checkpoint sidecars contain finite-looking capacitance labels
        that were already quantized in solver JSON.  A chart must therefore
        fail closed unless the sidecar proves that those labels were rebuilt
        from the lossless LC evidence under the current recovery contract.
        """
        recovery_payload = recovery if isinstance(recovery, dict) else {}
        contract = _safe_text(recovery_payload.get("contract"), 160)
        recovery_status = _safe_text(recovery_payload.get("status"), 80)
        recovered_row_count = cls._exact_integer(
            recovery_payload.get("recovered_row_count")
        )
        unique_actual_count = (
            len({pair["actual"] for pair in pairs})
            if isinstance(pairs, list) else None
        )
        valid = bool(
            contract == CAPACITANCE_RECOVERY_CONTRACT
            and recovery_status == "applied"
            and recovered_row_count is not None
            and recovered_row_count > 0
            and (n_rows is None or recovered_row_count >= n_rows)
        )
        if valid:
            return {
                "required": True,
                "valid": True,
                "status": "corrected",
                "message": "LC 역산 교정 labels",
                "contract": contract,
                "expected_contract": CAPACITANCE_RECOVERY_CONTRACT,
                "recovery_status": recovery_status,
                "recovered_row_count": recovered_row_count,
                "unique_actual_count": unique_actual_count,
            }
        return {
            "required": True,
            "valid": False,
            "status": CAPACITANCE_PARITY_INVALID_STATUS,
            "message": CAPACITANCE_PARITY_INVALID_MESSAGE,
            "contract": contract,
            "expected_contract": CAPACITANCE_RECOVERY_CONTRACT,
            "recovery_status": recovery_status,
            "recovered_row_count": recovered_row_count,
            "unique_actual_count": unique_actual_count,
        }

    def _capacitance_parity_overlay(self) -> dict[str, Any]:
        """Load the optional SHA-authenticated corrected C_* OOF sidecar."""
        path = self._capacitance_parity_artifact
        expected_sha = self._capacitance_parity_sha256
        empty = {
            "configured": bool(path or expected_sha),
            "valid": False,
            "targets": {},
            "provenance": {},
            "metadata": {},
            "source": None,
            "warnings": [],
        }
        if path is None and not expected_sha:
            return empty
        if path is None or not expected_sha:
            empty["warnings"] = [
                "capacitance parity overlay configuration is incomplete"
            ]
            return empty
        if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
            empty["warnings"] = [
                "capacitance parity overlay SHA-256 is malformed"
            ]
            return empty

        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            empty["warnings"] = [
                f"capacitance parity overlay SHA-256 mismatch: {path}"
            ]
            return empty
        result = self.cache.json(
            path, {}, max_bytes=64 * 1024 * 1024, fail_closed=True
        )
        if result.warning or not result.exists:
            empty["warnings"] = self._warnings(result) or [
                f"capacitance parity overlay is unavailable: {path}"
            ]
            return empty
        # Do not authenticate one byte sequence and then expose another if an
        # external writer replaces the file during parsing.
        if _sha256_file(path) != expected_sha:
            empty["warnings"] = [
                f"capacitance parity overlay changed while reading: {path}"
            ]
            return empty

        payload = result.value if isinstance(result.value, dict) else {}
        checkpoint = self._exact_integer(payload.get("checkpoint"))
        recovery = (
            payload.get("capacitance_recovery")
            if isinstance(payload.get("capacitance_recovery"), dict)
            else {}
        )
        if (
            payload.get("schema_version") != CHECKPOINT_PARITY_SCHEMA_VERSION
            or payload.get("artifact_type") != CHECKPOINT_PARITY_ARTIFACT_TYPE
            or payload.get("prediction_kind") != "out_of_fold"
            or checkpoint is None
            or checkpoint <= 0
            or recovery.get("contract") != CAPACITANCE_RECOVERY_CONTRACT
            or recovery.get("status") != "applied"
        ):
            empty["warnings"] = [
                f"capacitance parity overlay contract mismatch: {path}"
            ]
            return empty
        recovered_row_count = self._exact_integer(
            recovery.get("recovered_row_count")
        )
        raw_targets = payload.get("targets")
        if recovered_row_count is None or recovered_row_count <= 0 or not isinstance(
            raw_targets, dict
        ):
            empty["warnings"] = [
                f"capacitance parity overlay recovery evidence is malformed: {path}"
            ]
            return empty

        targets: dict[str, dict[str, Any]] = {}
        provenance_by_target: dict[str, dict[str, Any]] = {}
        for target in sorted(CAPACITANCE_PARITY_TARGETS):
            target_payload = raw_targets.get(target)
            if not isinstance(target_payload, dict):
                empty["warnings"] = [
                    f"capacitance parity overlay target is missing: {target}"
                ]
                return empty
            pairs = target_payload.get("pairs")
            sample_count = self._exact_integer(target_payload.get("sample_count"))
            n_rows = self._exact_integer(target_payload.get("n"))
            if (
                not isinstance(pairs, list)
                or sample_count is None
                or sample_count <= 0
                or sample_count != len(pairs)
                or sample_count > CHECKPOINT_PARITY_PAIR_LIMIT
                or n_rows is None
                or n_rows <= 0
                or sample_count > n_rows
                or recovered_row_count < n_rows
            ):
                empty["warnings"] = [
                    f"capacitance parity overlay target metadata is malformed: {target}"
                ]
                return empty
            clean_pairs: list[dict[str, Any]] = []
            for pair in pairs:
                if not isinstance(pair, dict):
                    empty["warnings"] = [
                        f"capacitance parity overlay pair is malformed: {target}"
                    ]
                    return empty
                actual = _finite_number(pair.get("actual"))
                predicted = _finite_number(pair.get("predicted"))
                if actual is None or predicted is None:
                    empty["warnings"] = [
                        f"capacitance parity overlay pair is non-finite: {target}"
                    ]
                    return empty
                clean_pairs.append({
                    "row_position": self._exact_integer(
                        pair.get("row_position")
                    ),
                    "row_index": _coerce(pair.get("row_index")),
                    "actual": actual,
                    "predicted": predicted,
                })
            provenance = self._capacitance_parity_provenance(
                recovery, n_rows=n_rows, pairs=clean_pairs
            )
            if provenance["valid"] is not True:
                empty["warnings"] = [
                    f"capacitance parity overlay provenance is invalid: {target}"
                ]
                return empty
            provenance_by_target[target] = {
                **provenance,
                "source_kind": "authenticated_overlay",
                "overlay_path": str(path),
                "overlay_sha256": expected_sha,
                "checkpoint": checkpoint,
                "completed_at": _safe_text(payload.get("completed_at"), 80),
            }
            targets[target] = {
                "n": n_rows,
                "sample_count": sample_count,
                "sampling": (
                    target_payload.get("sampling")
                    if isinstance(target_payload.get("sampling"), dict)
                    else {}
                ),
                "pairs": clean_pairs,
            }

        metadata = {
            "artifact_type": payload.get("artifact_type"),
            "prediction_kind": payload.get("prediction_kind"),
            "cv": payload.get("cv") if isinstance(payload.get("cv"), dict) else {},
            "max_pairs_per_target": self._exact_integer(
                payload.get("max_pairs_per_target")
            ),
            "capacitance_recovery": {
                "contract": recovery.get("contract"),
                "status": recovery.get("status"),
                "recovered_row_count": recovered_row_count,
            },
            "authenticated_overlay": {
                "path": str(path),
                "sha256": expected_sha,
                "checkpoint": checkpoint,
                "completed_at": _safe_text(payload.get("completed_at"), 80),
            },
        }
        return {
            **empty,
            "valid": True,
            "targets": targets,
            "provenance": provenance_by_target,
            "metadata": metadata,
            "source": str(path),
        }

    def _model_training_root(self) -> Path:
        """Prefer the live canonical registry without pinning a generation."""
        pipeline_root = getattr(self.continuous_pipeline, "root", None)
        if pipeline_root is not None:
            try:
                canonical = Path(pipeline_root).resolve() / "canonical_checkpoint"
                if canonical.is_dir():
                    return canonical
            except OSError:
                pass
        return self.root / "training"

    def _latest_checkpoint_evidence(self) -> dict[str, Any]:
        """Return the newest SHA-authorized metrics-only checkpoint.

        Checkpoint CV is evaluation evidence, not a deployed model.  The state
        item authorizes the immutable metrics result by SHA-256; learning-curve
        rows are deliberately not considered here.
        """
        training_root = self._model_training_root()
        strict_result = self.cache.json(
            training_root / "strict_data_status.json", {}, fail_closed=True
        )
        warnings = self._warnings(strict_result)
        if strict_result.exists and strict_result.warning:
            return {"candidate": None, "warnings": warnings}
        strict_payload = strict_result.value if isinstance(strict_result.value, dict) else {}
        strict_identity = (
            strict_payload.get("state_identity")
            if isinstance(strict_payload.get("state_identity"), dict)
            else {}
        )
        identity_keys = (
            (
                "solver_revision_cohort",
                "physics_data_revision",
                "library_revision",
            )
            if strict_identity.get("solver_revision_cohort")
            else ("solver_revision", "library_revision")
        )
        required_identity = {
            key: str(strict_identity[key]).strip().lower()
            for key in identity_keys
            if _safe_text(strict_identity.get(key), 160)
        }

        candidates: list[tuple[int, float, dict[str, Any]]] = []
        run_root_text = _safe_text(
            strict_payload.get("checkpoint_run_root"), 4_096
        )
        try:
            if run_root_text:
                configured_run = Path(run_root_text).resolve()
                state_paths = [configured_run / "checkpoint_state.json"]
            else:
                runs_root = training_root / "checkpoint_runs"
                state_paths = sorted(runs_root.glob("*/checkpoint_state.json"))
        except OSError as exc:
            return {
                "candidate": None,
                "warnings": warnings + [f"checkpoint state scan failed: {exc}"],
            }

        for state_path in state_paths:
            state_result = self.cache.json(state_path, {})
            if state_result.warning:
                warnings.append(state_result.warning)
                continue
            state = state_result.value if isinstance(state_result.value, dict) else {}
            if state.get("schema_version") != CHECKPOINT_STATE_SCHEMA_VERSION:
                continue
            identity = state.get("identity") if isinstance(state.get("identity"), dict) else {}
            if any(
                str(identity.get(key, "")).strip().lower() != expected
                for key, expected in required_identity.items()
            ):
                # Other solver/library runs are retained as archives and are
                # expected to coexist with the currently pinned campaign.
                continue
            completed = state.get("completed")
            if not isinstance(completed, list):
                continue
            try:
                run_root = state_path.parent.resolve()
            except OSError as exc:
                warnings.append(f"checkpoint run path resolution failed: {exc}")
                continue

            for item in completed:
                if not isinstance(item, dict) or item.get("kind") != "metrics_only":
                    continue
                threshold = self._exact_integer(item.get("threshold"))
                strict_rows = self._exact_integer(item.get("actual_strict_full_rows"))
                completed_text = _safe_text(item.get("completed_at"), 80)
                completed_at = _parse_time(completed_text, self.clock().tzinfo)
                if threshold is None or threshold < 0 or strict_rows is None or completed_at is None:
                    warnings.append(f"checkpoint completion is malformed: {state_path}")
                    continue

                metrics_text = _safe_text(item.get("metrics_result"), 4_096)
                if not metrics_text:
                    warnings.append(f"checkpoint metrics path is missing: {state_path}")
                    continue
                metrics_path = Path(metrics_text)
                if not metrics_path.is_absolute():
                    metrics_path = run_root / metrics_path
                try:
                    metrics_path = metrics_path.resolve()
                    if not metrics_path.is_relative_to(run_root):
                        raise ValueError("metrics result escapes checkpoint run root")
                except (OSError, ValueError) as exc:
                    warnings.append(f"checkpoint metrics path rejected: {exc}")
                    continue

                expected_hash = str(item.get("metrics_result_sha256", "")).strip().lower()
                actual_hash = _sha256_file(metrics_path)
                if not expected_hash or actual_hash != expected_hash:
                    warnings.append(f"checkpoint metrics hash mismatch: {metrics_path}")
                    continue
                metrics_result = self.cache.json(metrics_path, {})
                if metrics_result.warning or not metrics_result.exists:
                    warnings.extend(self._warnings(metrics_result))
                    continue
                metrics_payload = (
                    metrics_result.value if isinstance(metrics_result.value, dict) else {}
                )
                snapshot_sha = str(item.get("snapshot_sha256", "")).strip().lower()
                profile_sha = str(item.get("profile_sha256", "")).strip().lower()
                payload_checkpoint = self._exact_integer(metrics_payload.get("checkpoint"))
                payload_rows = self._exact_integer(metrics_payload.get("strict_full_rows"))
                evidence_matches = (
                    metrics_payload.get("schema_version") == CHECKPOINT_METRICS_SCHEMA_VERSION
                    and payload_checkpoint == threshold
                    and payload_rows == strict_rows
                    and str(metrics_payload.get("dataset_sha256", "")).strip().lower() == snapshot_sha
                    and str(metrics_payload.get("profile_sha256", "")).strip().lower() == profile_sha
                    and bool(snapshot_sha)
                    and bool(profile_sha)
                )
                if not evidence_matches:
                    warnings.append(f"checkpoint metrics identity mismatch: {metrics_path}")
                    continue
                state_profile_sha = str(identity.get("profile_sha256", "")).strip().lower()
                if state_profile_sha and state_profile_sha != profile_sha:
                    warnings.append(f"checkpoint state/profile identity mismatch: {state_path}")
                    continue

                metrics_rows = metrics_payload.get("metrics")
                if not isinstance(metrics_rows, list):
                    warnings.append(f"checkpoint metrics rows are malformed: {metrics_path}")
                    continue
                global_metrics: dict[str, dict[str, Any]] = {}
                duplicate_target = False
                for row in metrics_rows:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("slice", "global")).strip().lower() != "global":
                        continue
                    target = _safe_text(row.get("target"), 120)
                    if not target:
                        continue
                    if target in global_metrics:
                        duplicate_target = True
                        break
                    global_metrics[target] = row
                if duplicate_target or not global_metrics:
                    warnings.append(f"checkpoint global metrics are ambiguous or empty: {metrics_path}")
                    continue

                evaluated_at = (
                    _safe_text(metrics_payload.get("completed_at"), 80)
                    or completed_text
                )
                activation_minimum = self._exact_integer(
                    identity.get("activation_minimum_strict_full_rows")
                )
                if activation_minimum is None:
                    activation_minimum = self._exact_integer(
                        item.get("activation_minimum_strict_full_rows")
                    )
                candidate = {
                    "checkpoint": threshold,
                    "completed_at": completed_text,
                    "evaluated_at": evaluated_at,
                    "strict_full_rows": strict_rows,
                    "activation_minimum_strict_full_rows": activation_minimum,
                    "metrics": global_metrics,
                    "metrics_payload": metrics_payload,
                    "metrics_path": metrics_path,
                    "state_path": state_path,
                    "parity_path": metrics_path.with_suffix(".parity.json"),
                    "parity_targets": {},
                    "parity_metadata": {},
                    "parity_sources": {},
                    "parity_metadata_by_target": {},
                    "parity_provenance": {
                        target: self._capacitance_parity_provenance()
                        for target in global_metrics
                        if target in CAPACITANCE_PARITY_TARGETS
                    },
                }
                candidates.append((threshold, completed_at.timestamp(), candidate))

        if not candidates:
            return {"candidate": None, "warnings": list(dict.fromkeys(warnings))}
        _, _, selected = max(candidates, key=lambda entry: (entry[0], entry[1]))
        parity_path = selected["parity_path"]
        parity_result = self.cache.json(parity_path, {}, max_bytes=64 * 1024 * 1024)
        if parity_result.warning:
            warnings.append(parity_result.warning)
        elif parity_result.exists:
            parity = parity_result.value if isinstance(parity_result.value, dict) else {}
            metrics_payload = selected["metrics_payload"]
            parity_matches = (
                parity.get("schema_version") == CHECKPOINT_PARITY_SCHEMA_VERSION
                and parity.get("artifact_type") == CHECKPOINT_PARITY_ARTIFACT_TYPE
                and self._exact_integer(parity.get("checkpoint")) == selected["checkpoint"]
                and self._exact_integer(parity.get("strict_full_rows")) == selected["strict_full_rows"]
                and str(parity.get("dataset_sha256", "")).strip().lower()
                == str(metrics_payload.get("dataset_sha256", "")).strip().lower()
                and str(parity.get("profile_sha256", "")).strip().lower()
                == str(metrics_payload.get("profile_sha256", "")).strip().lower()
            )
            raw_targets = parity.get("targets")
            if not parity_matches or not isinstance(raw_targets, dict):
                warnings.append(f"checkpoint parity identity mismatch: {parity_path}")
            else:
                parity_targets: dict[str, dict[str, Any]] = {}
                parity_error: str | None = None
                capacitance_recovery = parity.get("capacitance_recovery")
                for target, target_payload in raw_targets.items():
                    if not isinstance(target, str) or not isinstance(target_payload, dict):
                        parity_error = "target payload is malformed"
                        break
                    pairs = target_payload.get("pairs")
                    sample_count = self._exact_integer(target_payload.get("sample_count"))
                    n_rows = self._exact_integer(target_payload.get("n"))
                    metric = selected["metrics"].get(target)
                    metric_n = self._exact_integer(metric.get("n")) if metric else None
                    if (
                        not isinstance(pairs, list)
                        or sample_count != len(pairs)
                        or sample_count is None
                        or sample_count > CHECKPOINT_PARITY_PAIR_LIMIT
                        or n_rows is None
                        or sample_count > n_rows
                        or (metric_n is not None and metric_n != n_rows)
                    ):
                        parity_error = f"{target} sample metadata is malformed"
                        break
                    clean_pairs = []
                    for pair in pairs:
                        if not isinstance(pair, dict):
                            parity_error = f"{target} pair is malformed"
                            break
                        actual = _finite_number(pair.get("actual"))
                        predicted = _finite_number(pair.get("predicted"))
                        if actual is None or predicted is None:
                            parity_error = f"{target} pair is non-finite"
                            break
                        clean_pairs.append({
                            "row_position": self._exact_integer(pair.get("row_position")),
                            "row_index": _coerce(pair.get("row_index")),
                            "actual": actual,
                            "predicted": predicted,
                        })
                    if parity_error:
                        break
                    if target in CAPACITANCE_PARITY_TARGETS:
                        provenance = self._capacitance_parity_provenance(
                            capacitance_recovery,
                            n_rows=n_rows,
                            pairs=clean_pairs,
                        )
                        selected["parity_provenance"][target] = provenance
                        if provenance["valid"] is not True:
                            # Keep every unrelated target available.  Only the
                            # capacitance parity whose label lineage is not
                            # proven is suppressed.
                            continue
                    parity_targets[target] = {
                        "n": n_rows,
                        "sample_count": sample_count,
                        "sampling": (
                            target_payload.get("sampling")
                            if isinstance(target_payload.get("sampling"), dict)
                            else {}
                        ),
                        "pairs": clean_pairs,
                    }
                if parity_error:
                    warnings.append(f"checkpoint parity malformed ({parity_error}): {parity_path}")
                else:
                    selected["parity_targets"] = parity_targets
                    selected["parity_metadata"] = {
                        "artifact_type": parity.get("artifact_type"),
                        "prediction_kind": _safe_text(parity.get("prediction_kind"), 80),
                        "cv": parity.get("cv") if isinstance(parity.get("cv"), dict) else {},
                        "max_pairs_per_target": self._exact_integer(
                            parity.get("max_pairs_per_target")
                        ),
                        "capacitance_recovery": {
                            "contract": _safe_text(
                                capacitance_recovery.get("contract"), 160
                            ),
                            "status": _safe_text(
                                capacitance_recovery.get("status"), 80
                            ),
                            "recovered_row_count": self._exact_integer(
                                capacitance_recovery.get("recovered_row_count")
                            ),
                        } if isinstance(capacitance_recovery, dict) else {},
                    }

        overlay = self._capacitance_parity_overlay()
        warnings.extend(overlay["warnings"])
        if overlay["valid"] is True:
            for target in CAPACITANCE_PARITY_TARGETS:
                # The overlay is parity evidence only.  It cannot invent a
                # model row that has no SHA-authorized checkpoint metrics.
                if target not in selected["metrics"]:
                    continue
                selected["parity_targets"][target] = overlay["targets"][target]
                selected["parity_provenance"][target] = overlay[
                    "provenance"
                ][target]
                selected["parity_sources"][target] = overlay["source"]
                selected["parity_metadata_by_target"][target] = overlay[
                    "metadata"
                ]

        for target, provenance in selected["parity_provenance"].items():
            if provenance.get("valid") is not True:
                warnings.append(
                    f"{target} parity provenance invalid: "
                    f"{CAPACITANCE_PARITY_INVALID_MESSAGE}"
                )

        return {"candidate": selected, "warnings": list(dict.fromkeys(warnings))}

    def models(self, current_data_count: int | None = None) -> dict[str, Any]:
        training_root = self._model_training_root()
        registry = training_root / "registry"
        pointer_result = self.cache.json(
            registry / "current.json", {}, fail_closed=True
        )
        pointer = (
            pointer_result.value
            if isinstance(pointer_result.value, dict)
            else {}
        )
        generation = registry / "generations" / "__unavailable__"
        pointer_warning = pointer_result.warning
        relative = pointer.get("generation")
        try:
            if pointer_warning:
                raise ValueError("active model pointer is unreadable")
            candidate = (registry / relative).resolve() if relative else None
            generations_root = (registry / "generations").resolve()
            if pointer.get("schema_version") != 2:
                raise ValueError("accepted schema-v2 model pointer is unavailable")
            if candidate is None or not candidate.is_relative_to(generations_root):
                raise ValueError("model pointer escapes generations root")
            generation = candidate
        except (OSError, TypeError, ValueError) as exc:
            pointer_warning = str(exc)
        report_result = self.cache.json(generation / "train_report.json", {})
        gate_result = self.cache.json(generation / "quality_gate.json", {})
        curve_result = self.cache.csv(
            training_root / "learning_curve.csv", max_rows=200_000
        )
        report_payload = report_result.value if isinstance(report_result.value, dict) else {}
        gate_payload = gate_result.value if isinstance(gate_result.value, dict) else {}
        report = report_payload.get("report") if isinstance(report_payload.get("report"), dict) else {}
        curve_rows = curve_result.value if isinstance(curve_result.value, list) else []
        warnings = self._warnings(pointer_result, report_result, gate_result, curve_result)
        evidence_error = pointer_warning or report_result.warning or gate_result.warning
        if not evidence_error and (
            _sha256_file(Path(report_result.path))
            != pointer.get("generation_report_sha256")
        ):
            evidence_error = "active generation report fingerprint mismatch"
        if not evidence_error and (
            _sha256_file(Path(gate_result.path))
            != pointer.get("quality_gate_sha256")
        ):
            evidence_error = "active generation gate fingerprint mismatch"
        if pointer_warning:
            warnings.append(pointer_warning)
            report = {}
        elif evidence_error:
            warnings.append(str(evidence_error))
            report = {}
        elif (
            gate_payload.get("passed") is not True
            or gate_payload.get("training_run_id") != pointer.get("training_run_id")
            or report_payload.get("training_run_id") != pointer.get("training_run_id")
            or gate_payload.get("generation") != pointer.get("generation")
            or gate_payload.get("generation_report_sha256")
            != pointer.get("generation_report_sha256")
            or gate_payload.get("dataset_sha256") != pointer.get("dataset_sha256")
            or gate_payload.get("profile_sha256") != pointer.get("profile_sha256")
            or report_payload.get("dataset_sha256") != pointer.get("dataset_sha256")
            or report_payload.get("profile_sha256") != pointer.get("profile_sha256")
            or report_payload.get("strict_full_rows")
            != pointer.get("strict_full_rows")
        ):
            warnings.append("active model generation has no matching passing gate")
            report = {}
        if current_data_count is None:
            data = self.data()
            training_cohort = data.get("training_cohort")
            current_data_count = (
                training_cohort.get("strict_full_rows")
                if isinstance(training_cohort, dict)
                and training_cohort.get("available") is True
                else data["total_rows"]
            )

        checkpoint_result = self._latest_checkpoint_evidence()
        warnings.extend(checkpoint_result["warnings"])
        checkpoint = checkpoint_result["candidate"]
        checkpoint_metrics = checkpoint["metrics"] if checkpoint else {}
        activation_minimum = (
            checkpoint["activation_minimum_strict_full_rows"] if checkpoint else None
        )
        preactivation_checkpoint = bool(
            checkpoint
            and not pointer_result.exists
            and activation_minimum is not None
            and current_data_count < activation_minimum
        )
        if preactivation_checkpoint and pointer_warning:
            # Before the activation floor, an absent pointer is the expected
            # state: checkpoint CV has run, but no deployable generation may be
            # promoted yet.  Existing/corrupt pointers and post-floor absence
            # remain warnings.
            warnings = [warning for warning in warnings if warning != pointer_warning]

        histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in curve_rows:
            target = _safe_text(row.get("target"), 120)
            if not target or str(row.get("slice", "global")).strip().lower() != "global":
                continue
            item = {
                "time": _safe_text(row.get("time"), 80),
                "n": _integer(row.get("n"), 0),
                "r2": _finite_number(row.get("r2")),
                "rmse": _finite_number(row.get("rmse")),
                "mape_pct": _finite_number(row.get("mape_pct")),
                "p90_ape_pct": _finite_number(row.get("p90_ape_pct")),
            }
            histories[target].append(item)

        ordered_targets = [item["name"] for item in TARGETS]
        ordered_targets.extend(
            sorted((set(report) | set(checkpoint_metrics)) - set(ordered_targets))
        )
        models: list[dict[str, Any]] = []
        latest_times: list[datetime] = []
        for target in ordered_targets:
            active_metrics = report.get(target) if isinstance(report.get(target), dict) else None
            meta_result = self.cache.json(generation / target / "meta.json", {})
            if meta_result.warning:
                warnings.append(meta_result.warning)
            meta = meta_result.value if isinstance(meta_result.value, dict) else {}
            if active_metrics is None and isinstance(meta.get("metrics"), dict):
                active_metrics = meta["metrics"]
            active_metrics = active_metrics or {}
            trained = bool(active_metrics)
            checkpoint_metric = (
                checkpoint_metrics.get(target)
                if not trained and isinstance(checkpoint_metrics.get(target), dict)
                else None
            )
            evaluated = trained or checkpoint_metric is not None
            metrics = active_metrics if trained else (checkpoint_metric or {})
            trained_at = (
                _safe_text(meta.get("trained_at") or report_payload.get("time"), 80)
                if trained else None
            )
            parsed_trained_at = _parse_time(trained_at, self.clock().tzinfo)
            if parsed_trained_at:
                latest_times.append(parsed_trained_at)
            n_train = _integer(metrics.get("n_train"), 0) if trained else 0
            n_holdout = _integer(metrics.get("n_holdout"), 0) if trained else 0
            n_used = (
                n_train + n_holdout
                if trained else max(0, self._exact_integer(metrics.get("n")) or 0)
            )
            stale = bool(trained and current_data_count and n_used and current_data_count >= n_used + max(100, int(n_used * 0.25)))
            history = [dict(item) for item in histories.get(target, [])[-100:]]
            r2 = _finite_number(metrics.get("r2"))
            mape = _finite_number(metrics.get("mape_pct"))
            p90_ape = _finite_number(metrics.get("p90_ape_pct"))
            mape_n = self._exact_integer(metrics.get("mape_n"))
            mape_excluded_zero_count = self._exact_integer(
                metrics.get("mape_excluded_zero_count")
            )
            mape_zero_abs_tolerance = _finite_number(
                metrics.get("mape_zero_abs_tolerance")
            )
            percentage_metric_source = (
                "artifact_zero_aware" if mape_n is not None
                else "legacy_all_rows"
            )
            # Legacy checkpoints used max(|actual|, 1e-9) as the denominator,
            # so a structural zero could turn one ordinary prediction error
            # into a multi-million-percent MAPE.  The parity artifact contains
            # every OOF pair while n <= the artifact limit, allowing an exact,
            # non-mutating display correction until the next zero-aware
            # checkpoint is produced.
            parity_metric = (
                checkpoint["parity_targets"].get(target, {})
                if checkpoint_metric is not None else {}
            )
            parity_provenance = (
                checkpoint["parity_provenance"].get(target)
                if checkpoint_metric is not None
                and target in CAPACITANCE_PARITY_TARGETS
                else None
            )
            parity_pairs = parity_metric.get("pairs", [])
            if (
                mape_n is None
                and isinstance(parity_pairs, list)
                and parity_metric.get("sample_count") == parity_metric.get("n")
                and parity_metric.get("n") == n_used
            ):
                corrected = _zero_aware_percentage_metrics(
                    [pair.get("actual") for pair in parity_pairs],
                    [pair.get("predicted") for pair in parity_pairs],
                )
                if corrected["mape_n"] > 0:
                    mape = corrected["mape_pct"]
                    p90_ape = corrected["p90_ape_pct"]
                    mape_n = corrected["mape_n"]
                    mape_excluded_zero_count = corrected[
                        "mape_excluded_zero_count"
                    ]
                    mape_zero_abs_tolerance = corrected[
                        "mape_zero_abs_tolerance"
                    ]
                    percentage_metric_source = "parity_recomputed_zero_aware"
                    for item in reversed(history):
                        if item.get("n") == n_used:
                            item.update({
                                "mape_pct": mape,
                                "p90_ape_pct": p90_ape,
                                "mape_n": mape_n,
                                "mape_excluded_zero_count": (
                                    mape_excluded_zero_count
                                ),
                                "mape_zero_abs_tolerance": (
                                    mape_zero_abs_tolerance
                                ),
                            })
                            break
            # learning_curve.csv remains historical display data only.  It is
            # never used as the numeric source for checkpoint fallback rows.
            previous = history[-1] if trained and history else None
            attention = bool(trained and ((r2 is not None and r2 < 0.8) or (mape is not None and mape > 20.0)))
            if (
                checkpoint_metric is not None
                and isinstance(parity_provenance, dict)
                and parity_provenance.get("source_kind")
                == "authenticated_overlay"
            ):
                status = "parity_overlay"
            elif (
                checkpoint_metric is not None
                and isinstance(parity_provenance, dict)
                and parity_provenance.get("valid") is not True
            ):
                status = "invalid_provenance"
            elif checkpoint_metric is not None:
                status = "checkpoint"
            elif not trained:
                status = "not_trained"
            elif stale:
                status = "stale"
            elif attention:
                status = "attention"
            else:
                status = "trained"
            models.append({
                "target": target,
                "label": TARGET_META.get(target, {}).get("label", target),
                "unit": TARGET_META.get(target, {}).get("unit", ""),
                "status": status,
                "trained": trained,
                "evaluated": evaluated,
                "deployable": trained,
                "evaluation_kind": (
                    "active_registry" if trained
                    else ("checkpoint_cv" if checkpoint_metric is not None else None)
                ),
                "checkpoint": checkpoint["checkpoint"] if checkpoint_metric is not None else None,
                "stale": stale,
                "n_train": n_train,
                "n_holdout": n_holdout,
                "n_used": n_used,
                "r2": r2,
                "rmse": _finite_number(metrics.get("rmse")),
                "mape_pct": mape,
                "p90_ape_pct": p90_ape,
                "mape_n": mape_n,
                "mape_excluded_zero_count": mape_excluded_zero_count,
                "mape_zero_abs_tolerance": mape_zero_abs_tolerance,
                "percentage_metric_source": percentage_metric_source,
                "q90_conformal": _finite_number(metrics.get("q90_conformal") or meta.get("q90")),
                "trained_at": trained_at,
                "evaluated_at": (
                    trained_at
                    if trained else (checkpoint["evaluated_at"] if checkpoint_metric is not None else None)
                ),
                "delta_r2": (r2 - previous["r2"] if r2 is not None and previous and previous["r2"] is not None else None),
                "delta_mape_pct": (mape - previous["mape_pct"] if mape is not None and previous and previous["mape_pct"] is not None else None),
                "parity_available": bool(
                    checkpoint_metric is not None
                    and checkpoint["parity_targets"].get(target, {}).get("pairs")
                ),
                "parity_sample_count": (
                    checkpoint["parity_targets"].get(target, {}).get("sample_count", 0)
                    if checkpoint_metric is not None else 0
                ),
                "parity_checkpoint": (
                    parity_provenance.get("checkpoint")
                    if isinstance(parity_provenance, dict)
                    and parity_provenance.get("source_kind")
                    == "authenticated_overlay"
                    else (
                        checkpoint["checkpoint"]
                        if checkpoint_metric is not None
                        and target in checkpoint["parity_targets"] else None
                    )
                ),
                "parity_source_kind": (
                    parity_provenance.get("source_kind")
                    if isinstance(parity_provenance, dict) else None
                ),
                "parity_source": (
                    checkpoint["parity_sources"].get(
                        target, str(checkpoint["parity_path"])
                    )
                    if checkpoint_metric is not None
                    and target in checkpoint["parity_targets"] else None
                ),
                "parity_provenance": parity_provenance,
                "source_kind": (
                    "active_registry" if trained
                    else ("checkpoint_cv" if checkpoint_metric is not None else None)
                ),
                "source": (
                    str(report_result.path) if trained
                    else (str(checkpoint["metrics_path"]) if checkpoint_metric is not None else None)
                ),
                "history": history,
            })

        trained_count = sum(model["trained"] for model in models)
        evaluated_count = sum(model["evaluated"] for model in models)
        primary_source = (
            str(report_result.path) if trained_count
            else (str(checkpoint["metrics_path"]) if checkpoint else str(report_result.path))
        )
        quality_note = (
            f"현재 strict 데이터 {current_data_count}개는 활성화 기준 {activation_minimum}개 미만이므로 "
            "검증된 checkpoint CV 평가만 표시하며 배포 모델로 취급하지 않습니다."
            if preactivation_checkpoint else
            "주의 표시는 탐색용 기준(R² < 0.8 또는 MAPE > 20%)이며, 최종 합격은 독립 FEA 검증으로 판정합니다."
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "available": report_result.exists or curve_result.exists or checkpoint is not None,
            "target_count": len(models),
            "trained_count": trained_count,
            "evaluated_count": evaluated_count,
            "missing_count": len(models) - trained_count,
            "current_data_count": current_data_count,
            "latest_trained_at": _iso(max(latest_times)) if latest_times else _safe_text(report_payload.get("time"), 80),
            "latest_checkpoint": checkpoint["checkpoint"] if checkpoint else None,
            "checkpoint_evaluated_at": checkpoint["evaluated_at"] if checkpoint else None,
            "activation_minimum_strict_full_rows": activation_minimum,
            "activation_state": (
                "active_registry" if trained_count
                else ("preactivation_checkpoint" if preactivation_checkpoint
                      else ("activation_due" if checkpoint and activation_minimum is not None
                            and current_data_count >= activation_minimum else "unavailable"))
            ),
            "quality_note": quality_note,
            "models": models,
            "warnings": list(dict.fromkeys(warnings)),
            "source": primary_source,
            "source_kind": (
                "active_registry" if trained_count
                else ("checkpoint_cv" if checkpoint else "unavailable")
            ),
            "active_source": str(report_result.path),
            "checkpoint_source": str(checkpoint["metrics_path"]) if checkpoint else None,
            "checkpoint_state_source": str(checkpoint["state_path"]) if checkpoint else None,
        }

    def model_history(self, target: str) -> dict[str, Any] | None:
        if target not in TARGET_META:
            return None
        models = self.models()
        model = next((item for item in models["models"] if item["target"] == target), None)
        return {"target": target, "label": TARGET_META[target]["label"], "history": model["history"] if model else []}

    def model_parity(self, target: str) -> dict[str, Any] | None:
        if target not in TARGET_META:
            return None
        models = self.models()
        model = next((item for item in models["models"] if item["target"] == target), None)
        base = {
            "schema_version": SCHEMA_VERSION,
            "target": target,
            "label": TARGET_META[target]["label"],
            "unit": TARGET_META[target]["unit"],
            "available": False,
            "evaluation_kind": model.get("evaluation_kind") if model else None,
            "checkpoint": model.get("checkpoint") if model else None,
            "source_kind": model.get("parity_source_kind") if model else None,
            "evaluated_at": model.get("evaluated_at") if model else None,
            "n": model.get("n_used", 0) if model else 0,
            "sample_count": 0,
            "sampling": {},
            "pairs": [],
            "metadata": {},
            "parity_provenance": (
                model.get("parity_provenance") if model else None
            ),
            "source": None,
            "warnings": models.get("warnings", []),
        }
        # A passing active registry has priority.  A checkpoint sidecar is not
        # presented as parity evidence for a different, deployed generation.
        if not model or model.get("evaluation_kind") != "checkpoint_cv":
            return base

        checkpoint_result = self._latest_checkpoint_evidence()
        candidate = checkpoint_result["candidate"]
        warnings = list(dict.fromkeys(
            [*base["warnings"], *checkpoint_result["warnings"]]
        ))
        if not candidate or candidate["checkpoint"] != model.get("checkpoint"):
            base["warnings"] = warnings
            return base
        parity = candidate["parity_targets"].get(target)
        if not isinstance(parity, dict) or not parity.get("pairs"):
            if target in CAPACITANCE_PARITY_TARGETS:
                base["parity_provenance"] = candidate[
                    "parity_provenance"
                ].get(target, base["parity_provenance"])
            base["warnings"] = warnings
            return base
        parity_provenance = candidate["parity_provenance"].get(
            target, base["parity_provenance"]
        )
        is_overlay = bool(
            isinstance(parity_provenance, dict)
            and parity_provenance.get("source_kind")
            == "authenticated_overlay"
        )
        return {
            **base,
            "available": True,
            "checkpoint": (
                parity_provenance.get("checkpoint")
                if is_overlay else candidate["checkpoint"]
            ),
            "source_kind": (
                "authenticated_overlay" if is_overlay
                else "checkpoint_sidecar"
            ),
            "evaluated_at": (
                parity_provenance.get("completed_at")
                if is_overlay else candidate["evaluated_at"]
            ),
            "n": parity["n"],
            "sample_count": parity["sample_count"],
            "sampling": parity["sampling"],
            "pairs": parity["pairs"],
            "metadata": candidate["parity_metadata_by_target"].get(
                target, candidate["parity_metadata"]
            ),
            "parity_provenance": parity_provenance,
            "source": candidate["parity_sources"].get(
                target, str(candidate["parity_path"])
            ),
            "warnings": warnings,
        }

    @staticmethod
    def _constraint(value: float | None, limit: float | tuple[float, float], mode: str) -> dict[str, Any]:
        if value is None:
            return {"value": None, "limit": limit, "margin": None, "pass": None}
        if mode == "max":
            limit_value = float(limit)
            return {"value": value, "limit": limit_value, "margin": limit_value - value, "pass": value <= limit_value}
        if mode == "min":
            limit_value = float(limit)
            return {"value": value, "limit": limit_value, "margin": value - limit_value, "pass": value >= limit_value}
        low, high = limit
        return {
            "value": value,
            "limit": [low, high],
            "margin": min(value - low, high - value),
            "pass": low <= value <= high,
        }

    def _candidate(self, row: dict[str, Any], round_number: int, index: int) -> dict[str, Any]:
        volume = _finite_number(row.get("volume_L"))
        loss = _finite_number(row.get("total_loss_W"))
        llt = _finite_number(row.get("pred_Llt_phys"))
        b_design = _finite_number(row.get("B_design_analytic_T"))
        b_mean = _finite_number(row.get("pred_B_mean_core"))
        bmax_diagnostic = _finite_number(row.get("pred_B_max_core"))
        temperatures = {
            target: _finite_number(row.get(f"pred_{target}"))
            for target in CANDIDATE_TEMPERATURE_TARGETS
        }
        available_temperatures = [value for value in temperatures.values() if value is not None]
        max_temperature = max(available_temperatures) if available_temperatures else None
        insulation_values = [
            value for key in INSULATION_KEYS if (value := _finite_number(row.get(key))) is not None
        ]
        min_insulation = min(insulation_values) if insulation_values else None
        core_groups = _finite_number(row.get("n_core_group"))
        dimension_report = _candidate_dimension_report(row)
        size_width = _finite_number(dimension_report.get("size_W_mm"))
        size_length = _finite_number(dimension_report.get("size_L_mm"))
        size_height = _finite_number(dimension_report.get("size_H_mm"))
        resonance_screen: dict[str, float] = {}
        try:
            coupling = float(row["pred_k"])
            c_tx = float(row["pred_C_tx_tx_F"])
            c_rx = float(row["pred_C_rx_rx_F"])
            c_cross = float(row["pred_C_tx_rx_F"])
            n1 = float(row.get("N1", 0)) or (
                float(row["N1_main"]) + float(row["N1_side"])
            )
            n2 = float(row.get("N2", 0)) or (
                float(row["N2_main"]) + float(row["N2_side"])
            )
            if not (
                llt is not None and llt > 0 and 0 < coupling < 1
                and c_tx > 0 and c_rx > 0 and c_cross > 0
                and n1 > 0 and n2 > 0
            ):
                raise ValueError("non-physical resonance input")
            leakage_h = llt * 1e-6
            lm_tx = leakage_h * coupling * coupling / (1.0 - coupling * coupling)
            lm_rx = lm_tx * (n2 / n1) ** 2
            factor = 0.5
            tx_f = 1.0 / (2.0 * math.pi * math.sqrt(factor * lm_tx * c_tx))
            rx_f = 1.0 / (2.0 * math.pi * math.sqrt(factor * lm_rx * c_rx))
            cross_f = 1.0 / (2.0 * math.pi * math.sqrt(leakage_h * c_cross))
            resonance_screen = {
                "pred_f_res_tx_screen_Hz": tx_f,
                "pred_f_res_rx_screen_Hz": rx_f,
                "pred_f_res_interwinding_screen_Hz": cross_f,
                "pred_f_res_min_screen_Hz": min(tx_f, rx_f, cross_f),
                "pred_Lm_tx_inferred_H": lm_tx,
                "pred_Lm_rx_inferred_H": lm_rx,
                "pred_Lm_tx_screen_H": factor * lm_tx,
                "pred_Lm_rx_screen_H": factor * lm_rx,
                "pred_magnetizing_inductance_factor": factor,
            }
        except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
            pass
        constraints = {
            "llt": self._constraint(llt, (26.95, 28.05), "band"),
            "temperature": self._constraint(max_temperature, 97.0, "max"),
            "bfield": self._constraint(b_design, 1.2, "max"),
            "insulation": self._constraint(min_insulation, 40.0, "min"),
            "core_group": self._constraint(core_groups, 4.0, "max"),
            "resonance": self._constraint(
                resonance_screen.get("pred_f_res_min_screen_Hz"), 20_000.0, "min"
            ),
            "size_width": self._constraint(size_width, 1_000.0, "max"),
            "size_length": self._constraint(size_length, 1_000.0, "max"),
            "size_height": self._constraint(size_height, 750.0, "max"),
        }
        passes = [item["pass"] for item in constraints.values()]
        spec_status = "fail" if False in passes else ("pass" if passes and all(value is True for value in passes) else "unknown")
        sigmas = {
            key.removeprefix("sigma_"): value
            for key, raw in row.items()
            if key.startswith("sigma_") and (value := _finite_number(raw)) is not None
        }
        parameters = {
            key: _coerce(row.get(key)) for key in DESIGN_PARAMETER_KEYS if row.get(key) not in (None, "")
        }
        report = {
            key: _coerce(row.get(key))
            for key in CANDIDATE_REPORT_FIELDS
            if row.get(key) not in (None, "")
        }
        report.update(dimension_report)
        report.update(resonance_screen)
        return {
            "id": f"r{round_number:02d}-{index:04d}",
            "index": index,
            "round": round_number,
            "volume_L": volume,
            "total_loss_W": loss,
            "pred_Llt_phys": llt,
            "B_design_analytic_T": b_design,
            "pred_B_mean_core": b_mean,
            "diagnostic_pred_B_max_core": bmax_diagnostic,
            "pred_max_temperature_C": max_temperature,
            "pred_temperatures_C": temperatures,
            "min_insulation_mm": min_insulation,
            "constraints": constraints,
            "spec_status": spec_status,
            "uncertainty": sigmas,
            "report": report,
            "parameters": parameters,
        }

    def _authenticated_tier1_candidate(
        self,
        row: dict[str, Any],
        round_number: int,
        index: int,
        *,
        source_candidate: dict[str, Any],
        hard_spec: dict[str, Any],
        constraint_version: str | None,
        temperature_targets: tuple[str, ...] = CANDIDATE_TEMPERATURE_TARGETS,
    ) -> dict[str, Any]:
        """Project an authenticated Tier-1 Pareto point with its active spec.

        ``_candidate`` intentionally retains the historical dashboard limits for
        legacy/local result sets.  Rolling Tier-1 points, however, have already
        been byte-for-byte authenticated against a terminal result and the
        cohort hard spec.  Re-running them through the legacy projection used to
        relabel T120/10 kHz/1200 mm candidates as 97 C/20 kHz/1000 mm failures.

        Values that the bounded Pareto preview omits (notably the analytical B
        value and exterior box for older terminal schemas) remain explicitly
        unavailable, while their pass state is backed by authenticated Pareto
        membership rather than invented numeric data.
        """

        candidate = self._candidate(row, round_number, index)
        decoded = source_candidate.get("decoded_params")
        predictions = source_candidate.get("predictions")
        if not isinstance(predictions, dict):
            predictions = source_candidate.get("surrogate_mean_predictions")
        half_width = source_candidate.get("conformal_half_width")
        if not isinstance(half_width, dict):
            half_width = source_candidate.get(
                "surrogate_q90_conformal_half_widths"
            )
        derived = source_candidate.get("derived_resonance")
        decoded = decoded if isinstance(decoded, dict) else {}
        predictions = predictions if isinstance(predictions, dict) else {}
        half_width = half_width if isinstance(half_width, dict) else {}
        derived = derived if isinstance(derived, dict) else {}
        if not derived:
            derived = {
                key: source_candidate.get(key)
                for key in (
                    "f_res_tx_half_magnetizing_Hz",
                    "f_res_rx_half_magnetizing_Hz",
                    "f_res_min_tx_rx_only_Hz",
                    "pred_f_res_tx_screen_Hz",
                    "pred_f_res_rx_screen_Hz",
                    "pred_f_res_min_screen_Hz",
                    "pred_f_res_interwinding_screen_Hz",
                )
                if source_candidate.get(key) is not None
            }
        dimension_report = _candidate_dimension_report(
            candidate.get("report"), row, decoded, source_candidate,
        )
        candidate["report"].update(dimension_report)

        def hard_number(name: str) -> float | None:
            return _finite_number(hard_spec.get(name))

        def authenticated_constraint(
            value: float | None,
            limit: float | list[float] | None,
            *,
            margin: float | None = None,
            passed: bool | None = None,
            evidence_only: bool = False,
        ) -> dict[str, Any]:
            if passed is None and evidence_only and limit is not None:
                passed = True
            return {
                "value": value,
                "limit": limit,
                "margin": margin,
                "pass": passed,
                "evidence": (
                    "authenticated_tier1_feasible_pareto"
                    if evidence_only else "independent_display_replay"
                ),
            }

        llt_target = hard_number("Llt_target_uH")
        llt_tolerance = hard_number("Llt_tol_uH")
        llt_mu = _finite_number(predictions.get("Llt_phys"))
        llt_half = _finite_number(half_width.get("Llt_phys"))
        llt_limit = (
            [llt_target - llt_tolerance, llt_target + llt_tolerance]
            if llt_target is not None and llt_tolerance is not None
            else None
        )
        llt_margin = (
            llt_tolerance - abs(llt_mu - llt_target) - llt_half
            if None not in (llt_target, llt_tolerance, llt_mu, llt_half)
            else None
        )
        llt_constraint = authenticated_constraint(
            llt_mu,
            llt_limit,
            margin=llt_margin,
            passed=llt_margin >= -1e-9 if llt_margin is not None else None,
            evidence_only=llt_margin is None,
        )
        llt_constraint["q90_conformal_half_width_uH"] = llt_half

        temperature_limit = hard_number("T_limit_C")
        n2_side = _finite_number(decoded.get("N2_side"))
        conditional = {"T_max_Rx_side", "Tprobe_Rx_side_leeward_max"}
        robust_temperatures: dict[str, float] = {}
        temperature_complete = temperature_limit is not None
        for target in temperature_targets:
            if target in conditional and n2_side == 0.0:
                continue
            mu = _finite_number(predictions.get(target))
            width = _finite_number(half_width.get(target))
            if mu is None or width is None:
                temperature_complete = False
                continue
            robust_temperatures[target] = mu + width
        robust_max_temperature = (
            max(robust_temperatures.values()) if robust_temperatures else None
        )
        temperature_margin = (
            temperature_limit - robust_max_temperature
            if temperature_limit is not None and robust_max_temperature is not None
            else None
        )
        temperature_constraint = authenticated_constraint(
            robust_max_temperature,
            temperature_limit,
            margin=temperature_margin,
            passed=(
                temperature_complete and temperature_margin >= -1e-9
                if temperature_margin is not None else None
            ),
            evidence_only=not temperature_complete,
        )
        temperature_constraint["value_basis"] = (
            "worst_surrogate_mu_plus_q90_conformal_half_width"
        )

        b_limit = hard_number("B_limit_T")
        b_value = _finite_number(decoded.get("B_design_analytic_T"))
        if b_value is None:
            b_value = _finite_number(decoded.get("B_design_square_material_analytic"))
        if b_value is None:
            b_value = _finite_number(source_candidate.get("B_design_analytic_T"))
        b_margin = b_limit - b_value if None not in (b_limit, b_value) else None

        insulation_limit = hard_number("insulation_min_mm")
        insulation_values = [
            value for key in INSULATION_KEYS
            if (value := _finite_number(decoded.get(key))) is not None
        ]
        insulation_value = min(insulation_values) if insulation_values else None
        if insulation_value is None:
            insulation_value = _finite_number(
                source_candidate.get("minimum_realized_insulation_mm")
            )
        insulation_margin = (
            insulation_value - insulation_limit
            if None not in (insulation_value, insulation_limit) else None
        )

        group_limit = hard_number("n_core_group_max")
        group_value = _finite_number(decoded.get("n_core_group"))
        group_margin = (
            group_limit - group_value
            if None not in (group_limit, group_value) else None
        )

        primary_limit = hard_number("primary_conductor_thickness_mm")
        primary_value = _finite_number(decoded.get("cw1"))
        primary_margin = (
            -abs(primary_value - primary_limit)
            if None not in (primary_value, primary_limit) else None
        )

        resonance_minimum = hard_number("resonance_min_Hz")
        resonance_maximum = hard_number("resonance_max_Hz")
        if resonance_maximum is not None and resonance_minimum is not None:
            resonance_limit: float | list[float] | None = [
                resonance_minimum,
                resonance_maximum,
            ]
            resonance_direction = "band"
            resonance_operator = "band"
        elif resonance_maximum is not None:
            resonance_limit = resonance_maximum
            resonance_direction = "maximum"
            resonance_operator = "<"
        elif resonance_minimum is not None:
            resonance_limit = resonance_minimum
            resonance_direction = "minimum"
            resonance_operator = ">="
        else:
            resonance_limit = None
            resonance_direction = None
            resonance_operator = None
        resonance_value = _finite_number(derived.get("f_res_min_screen_Hz"))
        if resonance_value is None:
            resonance_value = _finite_number(
                derived.get("pred_f_res_min_screen_Hz")
            )
        if resonance_value is None:
            resonance_value = _finite_number(
                derived.get("f_res_min_tx_rx_only_Hz")
            )
        if resonance_value is None or resonance_limit is None:
            resonance_margin = None
            resonance_passed = None
        elif resonance_direction == "band":
            resonance_margin = min(
                resonance_value - resonance_minimum,
                resonance_maximum - resonance_value,
            )
            resonance_passed = bool(
                resonance_value >= resonance_minimum
                and resonance_value < resonance_maximum
            )
        elif resonance_direction == "maximum":
            resonance_margin = resonance_limit - resonance_value
            resonance_passed = resonance_value < resonance_limit
        else:
            resonance_margin = resonance_value - resonance_limit
            resonance_passed = resonance_value >= resonance_limit

        size_constraints: dict[str, dict[str, Any]] = {}
        for suffix, row_key in (
            ("width", "size_W_mm"),
            ("length", "size_L_mm"),
            ("height", "size_H_mm"),
        ):
            limit = hard_number(f"size_{suffix[0].upper()}_max_mm")
            value = _finite_number(dimension_report.get(row_key))
            margin = limit - value if None not in (limit, value) else None
            size_constraints[f"size_{suffix}"] = authenticated_constraint(
                value,
                limit,
                margin=margin,
                passed=margin >= -1e-9 if margin is not None else None,
                evidence_only=value is None,
            )

        resonance_constraint = authenticated_constraint(
            resonance_value,
            resonance_limit,
            margin=resonance_margin,
            passed=resonance_passed,
            evidence_only=resonance_value is None,
        )
        resonance_constraint.update({
            "direction": resonance_direction,
            "operator": resonance_operator,
            "limit_key": (
                "resonance_max_Hz"
                if resonance_direction == "maximum"
                else (
                    "resonance_min_Hz"
                    if resonance_direction == "minimum" else None
                )
            ),
            "minimum_Hz": resonance_minimum,
            "maximum_Hz": resonance_maximum,
            "minimum_operator": (
                ">=" if resonance_minimum is not None else None
            ),
            "maximum_operator": (
                "<" if resonance_maximum is not None else None
            ),
        })

        constraints = {
            "llt": llt_constraint,
            "temperature": temperature_constraint,
            "bfield": authenticated_constraint(
                b_value,
                b_limit,
                margin=b_margin,
                passed=b_margin >= -1e-9 if b_margin is not None else None,
                evidence_only=b_value is None,
            ),
            "insulation": authenticated_constraint(
                insulation_value,
                insulation_limit,
                margin=insulation_margin,
                passed=(
                    insulation_margin >= -1e-9
                    if insulation_margin is not None else None
                ),
                evidence_only=insulation_value is None,
            ),
            "core_group": authenticated_constraint(
                group_value,
                group_limit,
                margin=group_margin,
                passed=group_margin >= -1e-9 if group_margin is not None else None,
                evidence_only=group_value is None,
            ),
            "primary_thickness": authenticated_constraint(
                primary_value,
                primary_limit,
                margin=primary_margin,
                passed=(
                    abs(primary_value - primary_limit) <= 1e-9
                    if None not in (primary_value, primary_limit) else None
                ),
                evidence_only=primary_value is None,
            ),
            "resonance": resonance_constraint,
            **size_constraints,
        }
        passes = [item["pass"] for item in constraints.values()]
        spec_status = (
            "fail" if False in passes
            else ("pass" if passes and all(value is True for value in passes)
                  else "unknown")
        )
        candidate.update({
            "constraints": constraints,
            "spec_status": spec_status,
            "constraint_version": constraint_version,
            "constraint_source": "authenticated_tier1_robust_hard_spec",
            "robust_temperatures_C": robust_temperatures,
            "robust_max_temperature_C": robust_max_temperature,
            "hard_spec": dict(hard_spec),
        })
        return candidate

    def _round_directories(self) -> list[tuple[int, Path]]:
        base = self.root / "al_rounds"
        if not base.is_dir():
            return []
        found = []
        for child in base.iterdir():
            match = re.fullmatch(r"round_(\d+)", child.name)
            if match and child.is_dir() and (child / "pareto_front.csv").exists():
                found.append((int(match.group(1)), child))
        return sorted(found)

    def _local_nsga_ui_status(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Read workstation-local isolated NSGA lane status fail-soft.

        The scheduler projection can lag newly launched immutable lanes because
        those lanes deliberately do not register with or mutate the scheduler.
        The 8010 monitor runs on the workstation that owns their authenticated
        ``ui_status.json`` files, so it can merge process observability without
        changing 8002 or treating a running search as an FEA-approved result.
        """
        paths: set[Path] = set()
        warnings: list[str] = []
        for root in self._nsga_ui_status_roots:
            try:
                if root.is_file() and root.name == "ui_status.json":
                    paths.add(root)
                    continue
                direct = root / "ui_status.json"
                if direct.is_file():
                    paths.add(direct)
                # Runtime waves use either <wave>/isolated_nsga/<lane> or a
                # root that points directly at isolated_nsga.  Deliberately
                # avoid an unrestricted rglob through multi-GB model trees.
                for pattern in (
                    "isolated_nsga/*/ui_status.json",
                    "*/isolated_nsga/*/ui_status.json",
                    "*/ui_status.json",
                ):
                    paths.update(path for path in root.glob(pattern) if path.is_file())
            except OSError as exc:
                warnings.append(
                    f"local NSGA status scan failed for {root}: "
                    f"{type(exc).__name__}: {exc}"
                )

        lanes: list[dict[str, Any]] = []
        for path in sorted(paths, key=lambda item: str(item).lower()):
            result = self.cache.json(path, {})
            if result.warning:
                warnings.append(result.warning)
            raw = result.value
            if not isinstance(raw, dict):
                continue
            if raw.get("schema_version") != "mft-isolated-nsga-live-v1":
                warnings.append(f"unsupported local NSGA status schema: {path}")
                continue
            model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
            config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
            terminal = (
                raw.get("terminal")
                if isinstance(raw.get("terminal"), dict)
                else {}
            )
            pareto = (
                terminal.get("pareto")
                if isinstance(terminal.get("pareto"), dict)
                else {}
            )
            pid = max(0, _integer(raw.get("pid"), 0))
            alive = raw.get("alive") is True
            state = (_safe_text(raw.get("state"), 40) or "unknown").lower()
            # A stale JSON file must never keep consuming the displayed worker
            # budget after its OS process has exited.
            if state == "running" and not alive:
                state = "stale"
            lane_name = _safe_text(raw.get("lane"), 100) or path.parent.name
            model_id = _safe_text(model.get("model_id"), 500) or ""
            lanes.append({
                "name": lane_name,
                "lane": lane_name,
                "state": state,
                "pid": pid or None,
                "alive": alive,
                "started_at": _safe_text(raw.get("started_at"), 100),
                "elapsed_seconds": _finite_number(raw.get("elapsed_seconds")),
                "updated_at": _safe_text(raw.get("updated_at"), 100),
                "workers": max(0, _integer(config.get("workers"), 0)),
                "restarts": max(0, _integer(config.get("restarts"), 0)),
                "population": max(0, _integer(config.get("population"), 0)),
                "max_generations": max(
                    0, _integer(config.get("max_generations"), 0)
                ),
                "seed_base": max(0, _integer(config.get("seed_base"), 0)),
                "model_id": model_id,
                "training_run_id": _safe_text(
                    model.get("training_run_id"), 160
                ),
                "dataset_rows": max(
                    0, _integer(model.get("strict_full_rows"), 0)
                ),
                "dataset_sha256": (
                    _safe_text(model.get("dataset_sha256"), 64) or ""
                ).lower(),
                "generation_report_sha256": (
                    _safe_text(model.get("generation_report_sha256"), 64) or ""
                ).lower(),
                "model_lane": _safe_text(
                    (model.get("eligibility") or {}).get("lane")
                    if isinstance(model.get("eligibility"), dict)
                    else None,
                    80,
                ),
                "terminal_outcome": _safe_text(terminal.get("outcome"), 80),
                "pareto": pareto if pareto.get("available") is True else None,
                "source_kind": "local_isolated_nsga_ui_status",
                "source": str(path),
            })
        return lanes, warnings

    def _slurm_nsga_offload_diagnostics(self) -> dict[str, Any]:
        """Read completed Slurm search evidence as diagnostics, never designs."""
        path = self._nsga_slurm_offload_status
        if path is None:
            return {"available": False, "configured": False}
        result = self.cache.json(path, {}, max_bytes=4 * 1024 * 1024)
        status = result.value if isinstance(result.value, dict) else {}
        base = {
            "available": bool(result.exists and status),
            "configured": True,
            "source": str(path),
            "warning": result.warning,
            "display_only": True,
            "production_eligible": False,
            "fea_submission_approved": False,
        }
        if not status:
            return base
        lanes = status.get("lanes")
        lanes = lanes if isinstance(lanes, list) else []
        state_counts: Counter[str] = Counter()
        best: list[dict[str, Any]] = []
        common_zero_pass: set[str] | None = None
        production_lane_count = 0
        total_candidates = 0
        detail_warnings: list[str] = []
        integrity_rows = []
        for lane in lanes[:100]:
            if not isinstance(lane, dict):
                continue
            lane_status = (
                lane.get("lane_status")
                if isinstance(lane.get("lane_status"), dict) else {}
            )
            state = _safe_text(lane_status.get("state"), 80) or "unknown"
            state_counts[state] += 1
            lane_id = _safe_text(lane_status.get("lane_id"), 200) or ""
            task_id = _integer(lane.get("task_id"), -1)
            is_production_lane = not lane_id.startswith("smoke-")
            if is_production_lane:
                production_lane_count += 1
            downloaded = lane.get("downloaded")
            downloaded = downloaded if isinstance(downloaded, list) else []
            downloads_verified = all(
                isinstance(item, dict)
                and item.get("verified") is True
                and item.get("remote_sha256") == item.get("local_sha256")
                for item in downloaded
            )
            integrity_rows.append(
                lane.get("artifacts_complete") is True
                and not lane.get("error")
                and lane.get("transport") == "direct_sftp"
                and downloads_verified
            )
            if task_id < 0:
                continue
            task_dir = path.parent / f"task-{task_id}"
            manifest_result = self.cache.json(
                task_dir / "least_violation_manifest.json", {},
                max_bytes=256 * 1024,
            )
            manifest = (
                manifest_result.value
                if isinstance(manifest_result.value, dict) else {}
            )
            if manifest_result.warning:
                detail_warnings.append(manifest_result.warning)
            candidate_count = max(0, _integer(manifest.get("candidate_count"), 0))
            total_candidates += candidate_count
            violation = _finite_number(
                manifest.get("best_total_positive_violation")
            )
            if violation is not None:
                best.append({
                    "task_id": task_id,
                    "seed": _integer(lane_status.get("seed_base"), -1),
                    "total_positive_violation": violation,
                    "candidate_count": candidate_count,
                })
            if not is_production_lane:
                continue
            report_result = self.cache.json(
                task_dir / "infeasibility_report.json", {},
                max_bytes=2 * 1024 * 1024,
            )
            report = (
                report_result.value
                if isinstance(report_result.value, dict) else {}
            )
            if report_result.warning:
                detail_warnings.append(report_result.warning)
            zero_pass = {
                _safe_text(item, 200)
                for item in report.get("seed_invariant_zero_pass_constraints", [])
                if _safe_text(item, 200)
            }
            common_zero_pass = (
                zero_pass if common_zero_pass is None
                else common_zero_pass & zero_pass
            )
        best.sort(key=lambda item: item["total_positive_violation"])
        expected = max(0, _integer(status.get("expected_lane_count"), 0))
        observed = max(0, _integer(status.get("observed_lane_count"), 0))
        complete_count = max(0, _integer(status.get("artifact_complete_count"), 0))
        integrity_complete = bool(
            status.get("complete") is True
            and expected > 0
            and expected == observed == complete_count == len(lanes)
            and len(integrity_rows) == len(lanes)
            and all(integrity_rows)
        )
        return {
            **base,
            "schema_version": _safe_text(status.get("schema_version"), 100),
            "bundle_id": _safe_text(status.get("bundle_id"), 160),
            "updated_at": _safe_text(status.get("updated_at"), 100),
            "complete": status.get("complete") is True,
            "integrity_complete": integrity_complete,
            "expected_lane_count": expected,
            "observed_lane_count": observed,
            "artifact_complete_count": complete_count,
            "production_lane_count": production_lane_count,
            "state_counts": dict(sorted(state_counts.items())),
            "feasible_lane_count": state_counts.get("completed", 0),
            "infeasible_lane_count": state_counts.get("infeasible", 0),
            "total_least_violation_candidates": total_candidates,
            "best": best[:5],
            "best_total_positive_violation": (
                best[0]["total_positive_violation"] if best else None
            ),
            "common_zero_pass_constraints": sorted(common_zero_pass or ()),
            "warnings": list(dict.fromkeys(detail_warnings)),
        }

    @staticmethod
    def _coherent_rolling_snapshot(pointer_path: Path) -> dict[str, Any]:
        """Read one stable rolling index/status/model generation.

        The writer replaces the canonical index after publishing a new status.
        A reader can therefore observe a legitimate short-lived hash mismatch.
        The index is read on both sides of the referenced artifacts; only an
        unchanged index with exact content hashes is accepted.
        """
        last_error = "rolling snapshot was not read"
        for attempt in range(1, ROLLING_SNAPSHOT_MAX_ATTEMPTS + 1):
            try:
                index, index_before = _bounded_json_bytes(
                    pointer_path, ROLLING_INDEX_MAX_BYTES
                )
                if index.get("schema_version") != (
                    "mft-tier1-slurm-rolling-index-v1"
                ):
                    raise ValueError("Tier-1 rolling index schema is invalid")
                containment_text = _safe_text(
                    index.get("path_containment_root"), 4_096
                )
                if not containment_text:
                    raise ValueError(
                        "Tier-1 rolling containment root is missing"
                    )
                containment_root = Path(containment_text).resolve()
                if (
                    not containment_root.is_dir()
                    or not pointer_path.resolve().is_relative_to(
                        containment_root
                    )
                ):
                    raise ValueError(
                        "Tier-1 rolling index escapes containment root"
                    )
                status_identity = index.get("status")
                model_identity = index.get("model_pointer")
                if not isinstance(status_identity, dict) or not isinstance(
                    model_identity, dict
                ):
                    raise ValueError(
                        "Tier-1 rolling artifact identities are missing"
                    )
                if status_identity.get("schema_version") != (
                    "mft-tier1-slurm-rolling-status-v1"
                ) or model_identity.get("schema_version") != (
                    "mft-tier1-slurm-model-pointer-v1"
                ):
                    raise ValueError(
                        "Tier-1 rolling artifact schemas are invalid"
                    )
                status_sha = str(status_identity.get("sha256") or "").lower()
                model_sha = str(model_identity.get("sha256") or "").lower()
                if not re.fullmatch(r"[0-9a-f]{64}", status_sha) or not (
                    re.fullmatch(r"[0-9a-f]{64}", model_sha)
                ):
                    raise ValueError(
                        "Tier-1 rolling artifact hashes are malformed"
                    )
                status_path = Path(
                    str(status_identity.get("path") or "")
                ).resolve()
                model_path = Path(
                    str(model_identity.get("path") or "")
                ).resolve()
                for artifact_path in (status_path, model_path):
                    if not artifact_path.is_relative_to(containment_root):
                        raise ValueError(
                            "Tier-1 rolling artifact escapes its sealed root"
                        )

                status, status_bytes = _bounded_json_bytes(
                    status_path, ROLLING_STATUS_MAX_BYTES
                )
                status_actual_sha = hashlib.sha256(status_bytes).hexdigest()
                model_pointer, model_bytes = _bounded_json_bytes(
                    model_path, ROLLING_MODEL_POINTER_MAX_BYTES
                )
                model_actual_sha = hashlib.sha256(model_bytes).hexdigest()
                _, index_after = _bounded_json_bytes(
                    pointer_path, ROLLING_INDEX_MAX_BYTES
                )
                if index_before != index_after:
                    raise ValueError(
                        "Tier-1 rolling index changed during snapshot read"
                    )
                if status_actual_sha != status_sha:
                    raise ValueError(
                        "Tier-1 rolling index/status hash mismatch"
                    )
                if model_actual_sha != model_sha:
                    raise ValueError(
                        "Tier-1 rolling model pointer hash mismatch"
                    )
                return {
                    "index": index,
                    "status": status,
                    "status_path": status_path,
                    "model_pointer": model_pointer,
                    "model_path": model_path,
                    "attempts": attempt,
                }
            except (OSError, TypeError, UnicodeError, ValueError) as exc:
                last_error = str(exc)
                if attempt < ROLLING_SNAPSHOT_MAX_ATTEMPTS:
                    time.sleep(ROLLING_SNAPSHOT_RETRY_SECONDS)
        raise ValueError(
            "Tier-1 rolling coherent snapshot unavailable after "
            f"{ROLLING_SNAPSHOT_MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _coherent_current7_snapshot(
        self, index_path: Path
    ) -> dict[str, Any]:
        """Read the additive current7 index without trusting a partial publish."""
        last_error = "current7 snapshot was not read"
        for attempt in range(1, ROLLING_SNAPSHOT_MAX_ATTEMPTS + 1):
            try:
                index, index_before = _bounded_json_bytes(
                    index_path, ROLLING_INDEX_MAX_BYTES
                )
                if index.get("schema_version") != CURRENT7_INDEX_SCHEMA:
                    raise ValueError("current7 secondary index schema is invalid")
                root_text = _safe_text(index.get("path_containment_root"), 4_096)
                if not root_text:
                    raise ValueError("current7 containment root is missing")
                containment_root = Path(root_text).resolve()
                resolved_index = index_path.resolve()
                if (
                    not containment_root.is_dir()
                    or not resolved_index.is_relative_to(containment_root)
                ):
                    raise ValueError("current7 index escapes its containment root")
                references = {}
                next_reference_cache: dict[
                    tuple[str, str], dict[str, Any]
                ] = {}
                for key, schema, maximum in (
                    ("status", CURRENT7_STATUS_SCHEMA, CURRENT7_STATUS_MAX_BYTES),
                    (
                        "compatibility",
                        CURRENT7_COMPATIBILITY_SCHEMA,
                        ROLLING_MODEL_POINTER_MAX_BYTES,
                    ),
                ):
                    identity = index.get(key)
                    if not isinstance(identity, dict):
                        raise ValueError(f"current7 {key} identity is missing")
                    if identity.get("schema_version") != schema:
                        raise ValueError(f"current7 {key} schema is invalid")
                    expected_sha = str(identity.get("sha256") or "").lower()
                    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                        raise ValueError(f"current7 {key} SHA is malformed")
                    raw_path = str(identity.get("path") or "")
                    raw_artifact_path = Path(raw_path)
                    if not raw_artifact_path.is_absolute():
                        raise ValueError(f"current7 {key} path is not absolute")
                    if (
                        raw_artifact_path.is_symlink()
                        or not raw_artifact_path.is_file()
                    ):
                        raise ValueError(
                            f"current7 {key} is not a regular file"
                        )
                    artifact_path = raw_artifact_path.resolve(strict=True)
                    if not artifact_path.is_relative_to(containment_root):
                        raise ValueError(
                            f"current7 {key} escapes its containment root"
                        )
                    artifact_stat = artifact_path.stat()
                    cache_key = (str(artifact_path), expected_sha)
                    cached = self._current7_reference_cache.get(cache_key)
                    if (
                        isinstance(cached, dict)
                        and cached.get("size") == artifact_stat.st_size
                        and cached.get("mtime_ns") == artifact_stat.st_mtime_ns
                    ):
                        value = cached["value"]
                    else:
                        value, raw = _bounded_json_bytes(
                            artifact_path, maximum
                        )
                        if hashlib.sha256(raw).hexdigest() != expected_sha:
                            raise ValueError(f"current7 {key} hash mismatch")
                    next_reference_cache[cache_key] = {
                        "value": value,
                        "size": artifact_stat.st_size,
                        "mtime_ns": artifact_stat.st_mtime_ns,
                    }
                    references[key] = {
                        "value": value,
                        "path": artifact_path,
                    }
                _, index_after = _bounded_json_bytes(
                    index_path, ROLLING_INDEX_MAX_BYTES
                )
                if index_before != index_after:
                    raise ValueError("current7 index changed during snapshot read")
                if len(self._current7_reference_cache) > 64:
                    self._current7_reference_cache.clear()
                self._current7_reference_cache.update(next_reference_cache)
                return {
                    "index": index,
                    "index_file_sha256": hashlib.sha256(
                        index_before
                    ).hexdigest(),
                    "status": references["status"]["value"],
                    "status_path": references["status"]["path"],
                    "status_file_sha256": str(
                        index["status"]["sha256"]
                    ).lower(),
                    "compatibility": references["compatibility"]["value"],
                    "compatibility_path": references["compatibility"]["path"],
                    "attempts": attempt,
                }
            except (OSError, TypeError, UnicodeError, ValueError) as exc:
                last_error = str(exc)
                if attempt < ROLLING_SNAPSHOT_MAX_ATTEMPTS:
                    time.sleep(ROLLING_SNAPSHOT_RETRY_SECONDS)
        raise ValueError(
            "current7 coherent snapshot unavailable after "
            f"{ROLLING_SNAPSHOT_MAX_ATTEMPTS} attempts: {last_error}"
        )

    @staticmethod
    def _condition_index_schema(index_path: Path) -> str:
        """Read only the bounded condition pointer before choosing a parser."""

        index, _raw = _bounded_json_bytes(index_path, ROLLING_INDEX_MAX_BYTES)
        schema = _safe_text(index.get("schema_version"), 160)
        if schema not in {CURRENT7_INDEX_SCHEMA, CURRENT7_COMPACT_INDEX_SCHEMA}:
            raise ValueError("configured condition index schema is unsupported")
        return schema

    @staticmethod
    def _compact_terminal_record_identity(
        record: Any,
        *,
        expected_identity: dict[str, Any],
        position: int,
    ) -> tuple[str, int]:
        """Validate one sealed logical seed without inventing a task ID."""

        if not isinstance(record, dict):
            raise ValueError(
                f"compact-v2 terminal record {position} is not an object"
            )
        claimed = str(record.get("record_sha256") or "").lower()
        unsigned = {
            key: value for key, value in record.items()
            if key != "record_sha256"
        }
        bundle_id = _safe_text(record.get("bundle_id"), 240) or ""
        seed = record.get("seed")
        if (
            record.get("schema_version") != CURRENT7_TERMINAL_RECORD_SCHEMA
            or claimed != _canonical_json_sha256(unsigned)
            or not bundle_id
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or record.get("authenticated") is not True
            or any(
                record.get(flag) is not False
                for flag in CURRENT7_FAIL_CLOSED_FLAGS
            )
        ):
            raise ValueError(
                f"compact-v2 terminal record {position} identity is invalid"
            )
        for field in (
            "constraint_version",
            "hard_spec_sha256",
            "hard_constraint_contract_sha256",
            "temperature_contract_sha256",
        ):
            if record.get(field) != expected_identity.get(field):
                raise ValueError(
                    "compact-v2 terminal constraint identity mismatch: "
                    f"{field} at record {position}"
                )
        task_ids = record.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or any(
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id <= 0
                for task_id in task_ids
            )
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError(
                f"compact-v2 terminal record {position} physical task IDs are invalid"
            )
        parent_id = record.get("physical_parent_task_id")
        if parent_id is not None and (
            isinstance(parent_id, bool)
            or not isinstance(parent_id, int)
            or parent_id <= 0
        ):
            raise ValueError(
                f"compact-v2 terminal record {position} parent task ID is invalid"
            )
        return bundle_id, int(seed)

    def _tier1_compact_condition_diagnostics_locked(
        self,
        *,
        index_path: Path,
        hydrate_candidates: bool,
    ) -> dict[str, Any]:
        """Authenticate a compact-v2 condition and optionally hydrate shards.

        Inventory calls stop after the compact index/status/manifest seals.
        The operator-selected detail call walks terminal records in bounded
        pages and retains only provenance identities plus the 256-candidate
        aggregate preview.  Physical Scheduler parents remain physical; no
        child seed is projected as a Scheduler task.
        """

        base = {
            "configured": True,
            "available": False,
            "integrity_verified": False,
            "healthy": False,
            "constraint_contract_verified": False,
            "authority_eligible": False,
            "display_only": True,
            "read_only": True,
            "condition_index_schema_version": CURRENT7_COMPACT_INDEX_SCHEMA,
            "detail_hydrated": False,
            "resonance_contract": None,
            "freshness_required": False,
            "freshness_ok": False,
            "freshness_age_seconds": None,
            "harvest_observed_at": None,
            "status_event_at": None,
            "source": str(index_path),
            "warnings": [],
            "lanes": [],
            "candidate_rows": [],
            "near_candidate_rows": [],
        }
        try:
            if (
                load_compact_condition_index is None
                or adapt_compact_condition_index is None
            ):
                raise ValueError(
                    "compact-v2 adapter is unavailable: "
                    f"{COMPACT_V2_ADAPTER_IMPORT_ERROR or 'unknown import error'}"
                )
            snapshot = load_compact_condition_index(index_path)
            index = snapshot["index"]
            status = snapshot["status"]
            static = status.get("frontend_static")
            if not isinstance(static, dict):
                raise ValueError("compact-v2 frontend static contract is missing")
            if any(
                flag in static and static.get(flag) is not False
                for flag in CURRENT7_FAIL_CLOSED_FLAGS
            ):
                raise ValueError(
                    "compact-v2 frontend static attempted execution authority"
                )
            aggregate = static.get("aggregate")
            if (
                not isinstance(aggregate, dict)
                or aggregate.get("schema_version")
                != CURRENT7_AGGREGATE_SCHEMA
                or any(
                    aggregate.get(flag) is not False
                    for flag in (
                        "production_eligible",
                        "fea_submission_approved",
                        "fea_submission_performed",
                    )
                )
            ):
                raise ValueError("compact-v2 aggregate contract is invalid")

            expected_identity = {
                field: copy.deepcopy(aggregate.get(field))
                for field in CURRENT7_IDENTITY_FIELDS
            }
            hard_spec = expected_identity["hard_spec"]
            hard_spec_sha = str(
                expected_identity.get("hard_spec_sha256") or ""
            ).lower()
            constraint_version = _safe_text(
                expected_identity.get("constraint_version"), 200
            ) or ""
            constraint_names = expected_identity.get("constraint_names")
            temperatures = expected_identity.get("temperature_targets")
            if (
                not constraint_version
                or not isinstance(hard_spec, dict)
                or not hard_spec
                or not re.fullmatch(r"[0-9a-f]{64}", hard_spec_sha)
                or _canonical_json_sha256(hard_spec) != hard_spec_sha
                or temperatures != list(CURRENT7_TEMPERATURE_TARGETS)
                or any(
                    not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(expected_identity.get(field) or "").lower(),
                    )
                    for field in (
                        "hard_constraint_contract_sha256",
                        "temperature_contract_sha256",
                    )
                )
            ):
                raise ValueError(
                    "compact-v2 hard-constraint identity is invalid"
                )
            for field, value in expected_identity.items():
                if field in static and static.get(field) != value:
                    raise ValueError(
                        f"compact-v2 frontend/aggregate identity drifted: {field}"
                    )
            resonance_contract = self._validated_current7_condition_contract(
                hard_spec,
                constraint_names,
            )
            stage_id = _safe_text(index.get("stage_id"), 160) or ""
            declared_stage = _safe_text(
                static.get("final_goal_stage_id"), 160
            )
            if declared_stage and declared_stage != stage_id:
                raise ValueError("compact-v2 stage identity is inconsistent")

            physical_lanes = status.get("latest_tasks")
            physical_lanes = (
                physical_lanes if isinstance(physical_lanes, list) else []
            )
            lanes: list[dict[str, Any]] = []
            physical_task_ids: list[int] = []
            for position, lane in enumerate(physical_lanes):
                if not isinstance(lane, dict):
                    raise ValueError(
                        f"compact-v2 physical lane {position} is invalid"
                    )
                task_id = lane.get("task_id")
                state = (_safe_text(lane.get("state"), 40) or "unknown").lower()
                if (
                    isinstance(task_id, bool)
                    or not isinstance(task_id, int)
                    or task_id <= 0
                    or task_id in physical_task_ids
                    or state not in CURRENT7_COMPACT_TASK_STATES
                ):
                    raise ValueError(
                        "compact-v2 physical Scheduler task identity is invalid"
                    )
                physical_task_ids.append(task_id)
                active = state in {"queued", "attaching", "running"}
                current_seed = lane.get("current_seed")
                lanes.append({
                    "lane": f"tier1-compact-parent-{task_id}",
                    "name": f"final1000-physical-parent-{task_id}",
                    "task_id": task_id,
                    "seed": (
                        current_seed
                        if isinstance(current_seed, int)
                        and not isinstance(current_seed, bool)
                        and current_seed >= 0
                        else None
                    ),
                    "state": state,
                    "alive": active,
                    "workers": 1 if active else 0,
                    "model_id": f"compact-v2:{stage_id}",
                    "model_lane": "final1000_compact_v2",
                    "updated_at": lane.get("updated_at") or status.get("updated_at"),
                    "source_kind": "tier1_current7_compact_parent",
                })

            refusals = static.get("refusals")
            refusals = refusals if isinstance(refusals, list) else []
            refused_count = static.get("refused_terminal_count", len(refusals))
            if (
                isinstance(refused_count, bool)
                or not isinstance(refused_count, int)
                or refused_count != len(refusals)
            ):
                raise ValueError("compact-v2 refusal count is inconsistent")
            refusal_identities: set[tuple[int, int]] = set()
            for position, refusal in enumerate(refusals):
                task_id = refusal.get("task_id") if isinstance(refusal, dict) else None
                seed = refusal.get("seed") if isinstance(refusal, dict) else None
                reason = refusal.get("reason") if isinstance(refusal, dict) else None
                identity = (task_id, seed)
                if (
                    not isinstance(refusal, dict)
                    or isinstance(task_id, bool)
                    or not isinstance(task_id, int)
                    or task_id <= 0
                    or isinstance(seed, bool)
                    or not isinstance(seed, int)
                    or seed < 0
                    or not isinstance(reason, str)
                    or not reason.strip()
                    or identity in refusal_identities
                ):
                    raise ValueError(
                        f"compact-v2 refusal {position} identity is invalid"
                    )
                refusal_identities.add(identity)

            aggregate_counts = {
                name: aggregate.get(name)
                for name in (
                    "authenticated_seed_count",
                    "candidate_count",
                    "feasible_candidate_count",
                    "pareto_count",
                )
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in aggregate_counts.values()
            ):
                raise ValueError("compact-v2 aggregate counts are invalid")
            total_records = int(status["authenticated_terminal_seed_count"])
            if (
                aggregate_counts["authenticated_seed_count"] != total_records
                or aggregate_counts["candidate_count"]
                < aggregate_counts["feasible_candidate_count"]
                or aggregate_counts["feasible_candidate_count"]
                < aggregate_counts["pareto_count"]
            ):
                raise ValueError("compact-v2 aggregate counts drifted")
            pareto = aggregate.get("pareto_candidates")
            if not isinstance(pareto, list):
                raise ValueError("compact-v2 Pareto preview is invalid")
            pareto_count = int(aggregate_counts["pareto_count"])
            if (
                len(pareto) != min(pareto_count, CURRENT7_CANDIDATE_PREVIEW_LIMIT)
                or aggregate.get("pareto_preview_truncated")
                is not (pareto_count > len(pareto))
            ):
                raise ValueError("compact-v2 Pareto preview count drifted")

            candidate_rows: list[dict[str, Any]] = []
            near_rows: list[dict[str, Any]] = []
            if hydrate_candidates:
                if total_records > CURRENT7_COMPACT_DETAIL_MAX_RECORDS:
                    raise ValueError(
                        "compact-v2 terminal inventory exceeds the bounded "
                        f"detail limit of {CURRENT7_COMPACT_DETAIL_MAX_RECORDS}"
                    )
                record_identities: set[tuple[str, int]] = set()
                offset = 0
                while offset < total_records:
                    page = adapt_compact_condition_index(
                        index_path,
                        offset=offset,
                        limit=CURRENT7_COMPACT_DETAIL_PAGE_SIZE,
                    )
                    records = page.get("terminal_results")
                    returned = page.get("terminal_results_returned_count")
                    if (
                        page.get("schema_version") != CURRENT7_STATUS_SCHEMA
                        or page.get("terminal_results_offset") != offset
                        or page.get("terminal_results_total_count") != total_records
                        or not isinstance(records, list)
                        or isinstance(returned, bool)
                        or not isinstance(returned, int)
                        or returned != len(records)
                        or returned <= 0
                    ):
                        raise ValueError(
                            "compact-v2 terminal page contract is invalid"
                        )
                    for record in records:
                        identity = self._compact_terminal_record_identity(
                            record,
                            expected_identity=expected_identity,
                            position=offset,
                        )
                        if identity in record_identities:
                            raise ValueError(
                                "compact-v2 terminal record identity is duplicated"
                            )
                        record_identities.add(identity)
                        offset += 1
                if offset != total_records or len(record_identities) != total_records:
                    raise ValueError("compact-v2 terminal pagination has a gap")
                _final_index, final_index_bytes = _bounded_json_bytes(
                    index_path, ROLLING_INDEX_MAX_BYTES
                )
                if final_index_bytes != snapshot["index_bytes"]:
                    raise ValueError(
                        "compact-v2 condition index advanced during detail pagination"
                    )

                expected_resonance = {
                    name for name in (
                        resonance_contract.get("minimum_constraint"),
                        resonance_contract.get("maximum_constraint"),
                    )
                    if isinstance(name, str) and name
                }

                def validate_candidate(
                    candidate: Any,
                    label: str,
                    *,
                    require_feasible: bool,
                ) -> tuple[str, int]:
                    if not isinstance(candidate, dict):
                        raise ValueError(f"compact-v2 {label} is invalid")
                    bundle_id = _safe_text(candidate.get("source_bundle_id"), 240) or ""
                    seed = candidate.get("source_seed")
                    decoded = candidate.get("decoded_params")
                    constraints = candidate.get("physical_constraint_G")
                    constraints = constraints if isinstance(constraints, dict) else {}
                    volume = _finite_number(candidate.get("volume_L"))
                    loss = _finite_number(candidate.get("total_loss_W"))
                    violation = _finite_number(
                        candidate.get("total_positive_violation")
                    )
                    observed_resonance = {
                        name for name in (
                            CURRENT7_RESONANCE_MINIMUM_CONSTRAINT,
                            CURRENT7_RESONANCE_MAXIMUM_CONSTRAINT,
                        )
                        if name in constraints
                    }
                    if (
                        not bundle_id
                        or isinstance(seed, bool)
                        or not isinstance(seed, int)
                        or (bundle_id, seed) not in record_identities
                        or not isinstance(decoded, dict)
                        or not decoded
                        or candidate.get("candidate_identity_sha256")
                        != _canonical_json_sha256(decoded)
                        or any(
                            candidate.get(flag) is not False
                            for flag in CURRENT7_FAIL_CLOSED_FLAGS
                        )
                        or volume is None
                        or volume < 0.0
                        or loss is None
                        or loss < 0.0
                        or violation is None
                        or violation < 0.0
                        or observed_resonance != expected_resonance
                        or any(
                            _finite_number(constraints.get(name)) is None
                            for name in expected_resonance
                        )
                        or (
                            require_feasible
                            and (
                                candidate.get("feasible") is not True
                                or violation > 1e-12
                                or any(
                                    float(constraints[name]) > 1e-12
                                    for name in expected_resonance
                                )
                            )
                        )
                    ):
                        raise ValueError(
                            f"compact-v2 {label} provenance/physics is invalid"
                        )
                    return bundle_id, int(seed)

                prior_objectives: tuple[float, float] | None = None
                for position, candidate in enumerate(pareto):
                    _bundle, seed = validate_candidate(
                        candidate,
                        f"Pareto candidate {position}",
                        require_feasible=True,
                    )
                    objectives = (
                        float(candidate["volume_L"]),
                        float(candidate["total_loss_W"]),
                    )
                    if prior_objectives is not None and objectives < prior_objectives:
                        raise ValueError("compact-v2 Pareto order is invalid")
                    prior_objectives = objectives
                    candidate_rows.append({
                        "seed": seed,
                        "candidate": candidate,
                        "source_role": "current7_condition",
                        "source_label": "read-only-current7-compact-v2",
                        "source_cohort_id": stage_id,
                    })
                least = aggregate.get("least_violation_candidate")
                if least is not None:
                    _bundle, seed = validate_candidate(
                        least,
                        "least-violation candidate",
                        require_feasible=False,
                    )
                    if least.get("feasible") is not True:
                        near_rows.append({
                            "seed": seed,
                            "candidate": least,
                            "source_role": "current7_condition",
                            "source_label": "read-only-current7-compact-v2",
                            "source_cohort_id": stage_id,
                        })

            state_counts = dict(status.get("state_counts") or {})
            active_count = sum(
                _integer(state_counts.get(name), 0)
                for name in ("queued", "attaching", "running")
            )
            updated_at = _safe_text(status.get("updated_at"), 100)
            updated_time = _parse_time(updated_at, timezone.utc)
            now = self.clock()
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            freshness_age = (
                (
                    now.astimezone(timezone.utc)
                    - updated_time.astimezone(timezone.utc)
                ).total_seconds()
                if updated_time is not None else None
            )
            freshness_required = active_count > 0
            freshness_ok = bool(
                not freshness_required
                or (
                    freshness_age is not None
                    and -CURRENT7_FUTURE_SKEW_TOLERANCE_SECONDS
                    <= freshness_age
                    <= CURRENT7_ACTIVE_FRESHNESS_MAX_AGE_SECONDS
                )
            )
            warnings: list[str] = []
            if refusals:
                warnings.append(
                    f"compact-v2 refused {len(refusals)} physical task artifact(s)"
                )
            if freshness_required and not freshness_ok:
                warnings.append(
                    "compact-v2 condition heartbeat is stale or future-dated; "
                    "the authenticated snapshot remains read-only"
                )
            return {
                **base,
                "available": True,
                "integrity_verified": True,
                "healthy": not refusals,
                "constraint_contract_verified": True,
                "detail_hydrated": hydrate_candidates,
                "resonance_contract": resonance_contract,
                "freshness_required": freshness_required,
                "freshness_ok": freshness_ok,
                "freshness_age_seconds": (
                    round(freshness_age, 3)
                    if freshness_age is not None else None
                ),
                "harvest_observed_at": updated_at,
                "status_event_at": updated_at,
                "updated_at": updated_at,
                "stage_id": stage_id,
                "cohort_id": stage_id,
                "bundle_id": _safe_text(static.get("bundle_id"), 200)
                or stage_id,
                "constraint_version": constraint_version,
                "hard_spec_sha256": hard_spec_sha,
                "hard_constraint_contract_sha256": expected_identity[
                    "hard_constraint_contract_sha256"
                ],
                "temperature_contract_sha256": expected_identity[
                    "temperature_contract_sha256"
                ],
                "constraint_identity_sha256": _canonical_json_sha256(
                    expected_identity
                ),
                "constraints": copy.deepcopy(hard_spec),
                "constraint_names": list(constraint_names),
                "temperature_targets": list(temperatures),
                "search_count": int(status["physical_lane_count"]),
                "physical_lane_count": int(status["physical_lane_count"]),
                "physical_task_ids": physical_task_ids,
                "logical_seed_count": int(status["logical_seed_count"]),
                "logical_sealed_seed_count": int(
                    status["logical_sealed_seed_count"]
                ),
                "logical_unsealed_seed_count": int(
                    status["logical_unsealed_seed_count"]
                ),
                "running_count": _integer(state_counts.get("running"), 0),
                "queued_count": _integer(state_counts.get("queued"), 0),
                "attaching_count": _integer(state_counts.get("attaching"), 0),
                "completed_count": _integer(state_counts.get("completed"), 0),
                "failed_count": _integer(state_counts.get("failed"), 0),
                "cancelled_count": sum(
                    _integer(state_counts.get(name), 0)
                    for name in ("cancelled", "canceled")
                ),
                "timeout_count": sum(
                    _integer(state_counts.get(name), 0)
                    for name in ("timeout", "timed_out", "deadline")
                ),
                "active_plus_queued": active_count,
                "authenticated_terminal_seed_count": total_records,
                "refused_terminal_count": len(refusals),
                "feasible_pareto_count": pareto_count,
                "candidate_preview_count": len(pareto),
                "candidate_preview_limit": CURRENT7_CANDIDATE_PREVIEW_LIMIT,
                "candidate_preview_truncated": aggregate.get(
                    "pareto_preview_truncated"
                ) is True,
                "near_feasible_count": (
                    1
                    if isinstance(
                        aggregate.get("least_violation_candidate"), dict
                    )
                    and aggregate["least_violation_candidate"].get("feasible")
                    is not True
                    else 0
                ),
                "candidate_rows": candidate_rows,
                "near_candidate_rows": near_rows,
                "state_counts": state_counts,
                "lanes": lanes,
                "snapshot_sha256": status.get("status_sha256"),
                "snapshot_file_sha256": hashlib.sha256(
                    snapshot["status_bytes"]
                ).hexdigest(),
                "index_file_sha256": hashlib.sha256(
                    snapshot["index_bytes"]
                ).hexdigest(),
                "snapshot_attempts": 1,
                "pointer_verified": True,
                "gui_launch_eligible": False,
                "virtual_scheduler_task_ids_created": False,
                "warnings": warnings,
            }
        except (OSError, RuntimeError, TypeError, UnicodeError, ValueError) as exc:
            return {
                **base,
                "available": True,
                "warnings": [f"compact-v2 condition index rejected: {exc}"],
            }

    @staticmethod
    def _current7_index_cache_contract(index: dict[str, Any]) -> str:
        """Bind every pointer field except the mutable harvest heartbeat."""
        stable_index = {
            key: value for key, value in index.items()
            if key != "harvest_observed_at"
        }
        return _canonical_json_sha256(stable_index)

    def _current7_cached_diagnostics(
        self,
        snapshot: dict[str, Any],
        *,
        index_path: Path,
        condition_display_only: bool,
    ) -> dict[str, Any] | None:
        cache_key = (
            f"{'condition' if condition_display_only else 'primary'}:"
            f"{index_path}"
        )
        cached = self._current7_diagnostics_cache.get(cache_key)
        if not isinstance(cached, dict):
            return None
        index = snapshot["index"]
        if (
            cached.get("status_file_sha256")
            != snapshot["status_file_sha256"]
            or cached.get("index_cache_contract")
            != self._current7_index_cache_contract(index)
        ):
            return None

        status = snapshot["status"]
        status_event_at_text = _safe_text(index.get("status_event_at"), 100)
        if (
            not status_event_at_text
            or status_event_at_text
            != _safe_text(status.get("updated_at"), 100)
        ):
            raise ValueError("current7 status event identity is inconsistent")
        harvest_observed_at_text = _safe_text(
            index.get("harvest_observed_at"), 100
        )
        harvest_observed_at = _parse_time(
            harvest_observed_at_text, timezone.utc
        )
        if harvest_observed_at is None:
            raise ValueError("current7 harvest_observed_at is invalid")
        now = self.clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        freshness_age_seconds = (
            now.astimezone(timezone.utc)
            - harvest_observed_at.astimezone(timezone.utc)
        ).total_seconds()

        result = copy.deepcopy(cached["diagnostics"])
        active_task_count = sum(
            _integer(result.get(name), 0)
            for name in (
                "running_count", "queued_count", "attaching_count"
            )
        )
        freshness_required = active_task_count > 0
        freshness_ok = bool(
            not freshness_required
            or (
                -CURRENT7_FUTURE_SKEW_TOLERANCE_SECONDS
                <= freshness_age_seconds
                <= CURRENT7_ACTIVE_FRESHNESS_MAX_AGE_SECONDS
            )
        )
        warnings = [
            warning for warning in result.get("warnings", [])
            if not str(warning).startswith((
                "current7 active harvester heartbeat",
                "current7 condition harvester heartbeat",
            ))
        ]
        if freshness_required and not freshness_ok:
            if condition_display_only:
                warnings.append(
                    "current7 condition harvester heartbeat is stale or "
                    "future-dated; the authenticated snapshot remains "
                    "read-only"
                )
            else:
                warnings.append(
                    "current7 active harvester heartbeat is stale or future-dated "
                    f"({freshness_age_seconds:.1f}s); authority was rejected"
                )
        result.update({
            "authority_eligible": bool(
                not condition_display_only
                and result.get("healthy") is True
                and result.get("constraint_contract_verified") is True
                and _integer(result.get("refused_terminal_count"), 0) == 0
                and freshness_ok
            ),
            "freshness_required": freshness_required,
            "freshness_ok": freshness_ok,
            "freshness_age_seconds": round(freshness_age_seconds, 3),
            "harvest_observed_at": harvest_observed_at_text,
            "status_event_at": status_event_at_text,
            "updated_at": status_event_at_text,
            "index_file_sha256": snapshot["index_file_sha256"],
            "snapshot_attempts": snapshot["attempts"],
            "warnings": warnings,
        })
        return result

    @staticmethod
    def _validated_current7_condition_contract(
        hard_spec: Any,
        constraint_names: Any,
    ) -> dict[str, Any]:
        """Validate one self-identifying read-only Current7 condition.

        Condition indexes may vary the staged size, temperature, and resonance
        limit, but may not alter the electrical/manufacturing invariants.  The
        resonance key and constraint name are validated as a pair so a maximum
        contract can never be displayed with the legacy minimum direction.
        """

        if not isinstance(hard_spec, dict) or not isinstance(
            constraint_names, list
        ):
            raise ValueError("current7 condition contract is missing")
        has_minimum = "resonance_min_Hz" in hard_spec
        has_maximum = "resonance_max_Hz" in hard_spec
        if not has_minimum and not has_maximum:
            raise ValueError(
                "current7 condition must declare a resonance bound"
            )
        resonance_keys = {
            key for key, present in (
                ("resonance_min_Hz", has_minimum),
                ("resonance_max_Hz", has_maximum),
            )
            if present
        }
        expected_keys = (
            set(CURRENT7_CONDITION_INVARIANT_HARD_SPEC)
            | {
                "T_limit_C",
                "size_W_max_mm",
                "size_L_max_mm",
                "size_H_max_mm",
            }
            | resonance_keys
        )
        if set(hard_spec) != expected_keys:
            raise ValueError("current7 condition hard-spec keys drifted")
        for key, expected in CURRENT7_CONDITION_INVARIANT_HARD_SPEC.items():
            observed = hard_spec.get(key)
            value = None if isinstance(observed, bool) else _finite_number(observed)
            if value is None or not math.isclose(
                value, float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"current7 condition invariant drifted: {key}"
                )
        for key in (
            "T_limit_C",
            *sorted(resonance_keys),
            "size_W_max_mm",
            "size_L_max_mm",
            "size_H_max_mm",
        ):
            observed = hard_spec.get(key)
            value = None if isinstance(observed, bool) else _finite_number(observed)
            if value is None or value <= 0.0:
                raise ValueError(
                    f"current7 condition limit is invalid: {key}"
                )
        if has_minimum and has_maximum and not (
            float(hard_spec["resonance_min_Hz"])
            < float(hard_spec["resonance_max_Hz"])
        ):
            raise ValueError(
                "current7 condition resonance band is empty or reversed"
            )
        expected_constraints = list(CURRENT7_CONSTRAINT_NAMES)
        minimum_position = expected_constraints.index(
            CURRENT7_RESONANCE_MINIMUM_CONSTRAINT
        )
        if not has_minimum:
            expected_constraints.pop(minimum_position)
        if has_maximum:
            expected_constraints.insert(
                minimum_position + (1 if has_minimum else 0),
                CURRENT7_RESONANCE_MAXIMUM_CONSTRAINT,
            )
        if constraint_names != expected_constraints:
            raise ValueError(
                "current7 condition constraint names/resonance direction drifted"
            )
        direction = (
            "band" if has_minimum and has_maximum
            else ("maximum" if has_maximum else "minimum")
        )
        return {
            "direction": direction,
            "minimum_key": "resonance_min_Hz" if has_minimum else None,
            "maximum_key": "resonance_max_Hz" if has_maximum else None,
            "minimum_constraint": (
                CURRENT7_RESONANCE_MINIMUM_CONSTRAINT
                if has_minimum else None
            ),
            "maximum_constraint": (
                CURRENT7_RESONANCE_MAXIMUM_CONSTRAINT
                if has_maximum else None
            ),
            "minimum_operator": ">=" if has_minimum else None,
            "maximum_operator": "<" if has_maximum else None,
        }

    def _tier1_current7_diagnostics(self) -> dict[str, Any]:
        """Serialize authentication so concurrent UI polls share one snapshot."""
        with self._current7_diagnostics_lock:
            return self._tier1_current7_diagnostics_locked(
                index_path=self._tier1_current7_index,
                condition_display_only=False,
            )

    def _tier1_current7_diagnostics_locked(
        self,
        *,
        index_path: Path | None,
        condition_display_only: bool,
    ) -> dict[str, Any]:
        """Authenticate and project the corrected seven-temperature search."""
        diagnostic_name = (
            "current7 condition index"
            if condition_display_only else "current7 secondary index"
        )
        base = {
            "configured": index_path is not None,
            "available": False,
            "integrity_verified": False,
            "healthy": False,
            "constraint_contract_verified": False,
            "authority_eligible": False,
            "display_only": condition_display_only,
            "read_only": condition_display_only,
            "resonance_contract": None,
            "freshness_required": False,
            "freshness_ok": False,
            "freshness_age_seconds": None,
            "harvest_observed_at": None,
            "status_event_at": None,
            "source": (
                str(index_path) if index_path is not None else None
            ),
            "warnings": [],
            "lanes": [],
            "candidate_rows": [],
            "near_candidate_rows": [],
        }
        if index_path is None:
            return base
        try:
            snapshot = self._coherent_current7_snapshot(
                index_path
            )
            cached_diagnostics = self._current7_cached_diagnostics(
                snapshot,
                index_path=index_path,
                condition_display_only=condition_display_only,
            )
            if cached_diagnostics is not None:
                return cached_diagnostics
            index = snapshot["index"]
            status = snapshot["status"]
            compatibility = snapshot["compatibility"]
            if status.get("schema_version") != CURRENT7_STATUS_SCHEMA:
                raise ValueError("current7 cohort status schema is invalid")
            if compatibility.get("schema_version") != (
                CURRENT7_COMPATIBILITY_SCHEMA
            ):
                raise ValueError("current7 compatibility schema is invalid")
            aggregate = status.get("aggregate")
            if not isinstance(aggregate, dict) or aggregate.get(
                "schema_version"
            ) != CURRENT7_AGGREGATE_SCHEMA:
                raise ValueError("current7 aggregate schema is invalid")

            for document_name, document in (
                ("index", index),
                ("status", status),
            ):
                if any(
                    document.get(flag) is not False
                    for flag in CURRENT7_FAIL_CLOSED_FLAGS
                ):
                    raise ValueError(
                        f"current7 {document_name} attempted execution authority"
                    )
            for flag in (
                "production_eligible",
                "fea_submission_approved",
                "fea_submission_performed",
            ):
                if aggregate.get(flag) is not False:
                    raise ValueError("current7 aggregate attempted execution authority")
            if (
                compatibility.get("compatible") is not False
                or compatibility.get("activation_allowed") is not False
                or compatibility.get("legacy_canonical_index_write_allowed")
                is not False
                or index.get("activation_allowed") is not False
                or index.get("legacy_canonical_index_touched") is not False
                or (index.get("compatibility") or {}).get(
                    "legacy_84ff69f_compatible"
                ) is not False
            ):
                raise ValueError("current7 legacy-monitor isolation is invalid")
            if status.get("legacy_8010_compatibility") != compatibility:
                raise ValueError("current7 compatibility copy is inconsistent")
            if status.get("cohort_id") != index.get("active_cohort_id"):
                raise ValueError("current7 active cohort identity is inconsistent")

            identity = status.get("snapshot_identity")
            if not isinstance(identity, dict) or identity.get(
                "schema_version"
            ) != CURRENT7_SNAPSHOT_IDENTITY_SCHEMA:
                raise ValueError("current7 snapshot identity schema is invalid")
            tasks = status.get("latest_tasks")
            records = status.get("terminal_results")
            refusals = status.get("refusals")
            tasks = tasks if isinstance(tasks, list) else None
            records = records if isinstance(records, list) else None
            refusals = refusals if isinstance(refusals, list) else None
            if tasks is None or records is None or refusals is None:
                raise ValueError("current7 status inventories are invalid")
            anchor_bundle_id = _safe_text(status.get("bundle_id"), 500)
            if not anchor_bundle_id:
                raise ValueError("current7 anchor bundle identity is invalid")
            status_mixed_projection = status.get("mixed_bundle_projection")
            index_mixed_projection = index.get("mixed_bundle_projection")
            if (
                status_mixed_projection is not None
                and type(status_mixed_projection) is not bool
            ) or (
                index_mixed_projection is not None
                and type(index_mixed_projection) is not bool
            ):
                raise ValueError("current7 mixed-bundle flag is invalid")
            if (status_mixed_projection is True) is not (
                index_mixed_projection is True
            ):
                raise ValueError(
                    "current7 mixed-bundle projection copy is inconsistent"
                )
            mixed_bundle_projection = status_mixed_projection is True
            allowed_bundle_manifests = {
                anchor_bundle_id: status.get("bundle_manifest_sha256")
            }
            if mixed_bundle_projection:
                if (
                    not condition_display_only
                    or index.get("condition_display_only") is not True
                ):
                    raise ValueError(
                        "current7 mixed-bundle projection is not display-only"
                    )
                status_cohorts = status.get("source_bundle_cohorts")
                index_cohorts = index.get("source_bundle_cohorts")
                if (
                    not isinstance(status_cohorts, list)
                    or len(status_cohorts) < 2
                    or index_cohorts != status_cohorts
                ):
                    raise ValueError(
                        "current7 source bundle cohort copy is inconsistent"
                    )
                cohorts_sha256 = str(
                    status.get("source_bundle_cohorts_sha256") or ""
                ).lower()
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", cohorts_sha256)
                    or index.get("source_bundle_cohorts_sha256")
                    != cohorts_sha256
                    or _canonical_json_sha256(status_cohorts)
                    != cohorts_sha256
                ):
                    raise ValueError(
                        "current7 source bundle cohort SHA is invalid"
                    )
                if (
                    status.get("projection_anchor_bundle_id")
                    != anchor_bundle_id
                    or index.get("projection_anchor_bundle_id")
                    != anchor_bundle_id
                ):
                    raise ValueError(
                        "current7 projection anchor bundle is inconsistent"
                    )
                allowed_bundle_manifests = {}
                source_roles: dict[str, str] = {}
                resource_policy_ids: set[str] = set()
                cohort_hash_fields = (
                    "bundle_manifest_sha256",
                    "launch_plan_sha256",
                    "manifest_contract_sha256",
                    "publication_receipt_sha256",
                    "ready_sha256",
                    "stage_spec_sha256",
                )
                for position, cohort in enumerate(status_cohorts):
                    if not isinstance(cohort, dict):
                        raise ValueError(
                            "current7 source bundle cohort is invalid at "
                            f"position {position}"
                        )
                    bundle_id = _safe_text(cohort.get("bundle_id"), 500)
                    role = _safe_text(cohort.get("role"), 40)
                    policies = cohort.get("resource_policy_ids")
                    if (
                        not bundle_id
                        or bundle_id in allowed_bundle_manifests
                        or role not in {"predecessor", "successor"}
                        or not isinstance(policies, list)
                        or not policies
                        or len(policies) != len(set(policies))
                        or any(
                            not isinstance(policy, str) or not policy.strip()
                            for policy in policies
                        )
                        or not _safe_text(cohort.get("remote_bundle"), 4_096)
                        or (
                            cohort.get("cohort_id") is not None
                            and not _safe_text(cohort.get("cohort_id"), 500)
                        )
                        or any(
                            not re.fullmatch(
                                r"[0-9a-f]{64}",
                                str(cohort.get(field) or "").lower(),
                            )
                            for field in cohort_hash_fields
                        )
                        or cohort.get("stage_spec_sha256")
                        != status.get("hard_spec_sha256")
                    ):
                        raise ValueError(
                            "current7 source bundle cohort identity is invalid "
                            f"at position {position}"
                        )
                    allowed_bundle_manifests[bundle_id] = cohort[
                        "bundle_manifest_sha256"
                    ]
                    source_roles[bundle_id] = role
                    resource_policy_ids.update(policies)
                if (
                    len(allowed_bundle_manifests) < 2
                    or source_roles.get(anchor_bundle_id) != "successor"
                    or sum(
                        role == "successor" for role in source_roles.values()
                    ) != 1
                    or allowed_bundle_manifests.get(anchor_bundle_id)
                    != status.get("bundle_manifest_sha256")
                    or index.get("bundle_manifest_sha256")
                    != status.get("bundle_manifest_sha256")
                ):
                    raise ValueError(
                        "current7 sealed source cohorts do not contain the "
                        "exact projection anchor"
                    )
                expected_mixed_resources = len(resource_policy_ids) > 1
                if (
                    status.get("mixed_resource_policy_projection")
                    is not expected_mixed_resources
                    or index.get("mixed_resource_policy_projection")
                    is not expected_mixed_resources
                ):
                    raise ValueError(
                        "current7 mixed resource-policy projection is invalid"
                    )
            record_hashes = []
            record_seeds: set[int] = set()
            record_identities: set[tuple[str, int]] = set()
            record_task_ids: dict[tuple[str, int], tuple[int, ...] | None] = {}
            for position, record in enumerate(records):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"current7 terminal record {position} is invalid"
                    )
                claimed = str(record.get("record_sha256") or "").lower()
                unsigned = {
                    key: value for key, value in record.items()
                    if key != "record_sha256"
                }
                if claimed != _canonical_json_sha256(unsigned):
                    raise ValueError(
                        f"current7 terminal record {position} hash mismatch"
                    )
                seed = _current7_identity_integer(record.get("seed"))
                record_bundle_id = _safe_text(record.get("bundle_id"), 500)
                record_identity = (record_bundle_id or "", seed)
                if (
                    record.get("schema_version")
                    != CURRENT7_TERMINAL_RECORD_SCHEMA
                    or record_bundle_id != anchor_bundle_id
                    or record.get("authenticated") is not True
                    or seed < 0
                    or record_identity in record_identities
                    or any(
                        record.get(flag) is not False
                        for flag in CURRENT7_FAIL_CLOSED_FLAGS
                    )
                ):
                    raise ValueError(
                        "current7 terminal record provenance is invalid at "
                        f"position {position}"
                    )
                record_seeds.add(seed)
                record_identities.add(record_identity)
                raw_task_ids = record.get("task_ids")
                if raw_task_ids is None:
                    record_task_ids[record_identity] = None
                elif (
                    not isinstance(raw_task_ids, list)
                    or not raw_task_ids
                    or any(
                        isinstance(task_id, bool)
                        or not isinstance(task_id, int)
                        or task_id < 0
                        for task_id in raw_task_ids
                    )
                    or len(raw_task_ids) != len(set(raw_task_ids))
                ):
                    raise ValueError(
                        "current7 terminal record task identity is invalid at "
                        f"position {position}"
                    )
                else:
                    record_task_ids[record_identity] = tuple(raw_task_ids)
                record_hashes.append(claimed)
                for field in (
                    "constraint_version",
                    "hard_spec_sha256",
                    "hard_constraint_contract_sha256",
                    "temperature_contract_sha256",
                ):
                    if record.get(field) != status.get(field):
                        raise ValueError(
                            "current7 terminal compact constraint identity "
                            f"mismatch at record {position}: {field}"
                        )
            refusal_identities: set[tuple[str, int, int]] = set()
            for position, refusal in enumerate(refusals):
                if not isinstance(refusal, dict):
                    raise ValueError(
                        f"current7 refusal {position} is invalid"
                    )
                task_id = _current7_identity_integer(refusal.get("task_id"))
                seed = _current7_identity_integer(refusal.get("seed"))
                bundle_id = _safe_text(refusal.get("bundle_id"), 500)
                refusal_identity = (bundle_id or "", task_id, seed)
                reason = refusal.get("reason")
                if bundle_id not in allowed_bundle_manifests:
                    raise ValueError(
                        "current7 refusal bundle is not in sealed source "
                        f"cohorts at position {position}"
                    )
                if (
                    task_id < 0
                    or seed < 0
                    or refusal_identity in refusal_identities
                    or not isinstance(reason, str)
                    or not reason.strip()
                ):
                    raise ValueError(
                        "current7 refusal identity is invalid or duplicate at "
                        f"position {position}"
                    )
                refusal_identities.add(refusal_identity)
            scale_policy_sha256: str | None = None
            scale_summary: dict[str, Any] | None = None
            scale_fields_present = any(
                field in document
                for document, field in (
                    (index, "scale_policy_sha256"),
                    (index, "scale_summary"),
                    (status, "scale_policy_sha256"),
                    (status, "scale_summary"),
                    (identity, "scale_policy_sha256"),
                    (identity, "scale_summary_sha256"),
                )
            )
            scale_identity: dict[str, str] = {}
            if scale_fields_present:
                scale_policy_sha256 = str(
                    status.get("scale_policy_sha256") or ""
                ).lower()
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", scale_policy_sha256)
                    or index.get("scale_policy_sha256")
                    != scale_policy_sha256
                    or identity.get("scale_policy_sha256")
                    != scale_policy_sha256
                ):
                    raise ValueError(
                        "current7 scale policy identity is inconsistent"
                    )
                status_scale_summary = status.get("scale_summary")
                index_scale_summary = index.get("scale_summary")
                if (
                    not isinstance(status_scale_summary, dict)
                    or index_scale_summary != status_scale_summary
                ):
                    raise ValueError(
                        "current7 scale summary copy is inconsistent"
                    )
                scale_summary = status_scale_summary
                summary_sha256 = str(
                    scale_summary.get("summary_sha256") or ""
                ).lower()
                unsigned_summary = {
                    key: value for key, value in scale_summary.items()
                    if key != "summary_sha256"
                }
                scaled_resources = scale_summary.get(
                    "scaled_resource_contract"
                )
                scaled_resources = (
                    scaled_resources
                    if isinstance(scaled_resources, dict) else None
                )
                scaled_resource_sha256 = str(
                    scale_summary.get("scaled_resource_contract_sha256")
                    or ""
                ).lower()
                base_resource_sha256 = str(
                    scale_summary.get("base_resource_contract_sha256")
                    or ""
                ).lower()
                effective_active_total = scale_summary.get(
                    "effective_active_total"
                )
                max_workers_per_node = scale_summary.get(
                    "max_workers_per_node"
                )
                island_quotas = scale_summary.get("island_active_quotas")
                quota_values = (
                    list(island_quotas.values())
                    if isinstance(island_quotas, dict) else []
                )
                if (
                    scale_summary.get("schema_version")
                    != CURRENT7_SCALE_SUMMARY_SCHEMA
                    or scale_summary.get("bundle_id")
                    != status.get("bundle_id")
                    or scale_summary.get("policy_sha256")
                    != scale_policy_sha256
                    or not re.fullmatch(r"[0-9a-f]{64}", summary_sha256)
                    or summary_sha256
                    != _canonical_json_sha256(unsigned_summary)
                    or identity.get("scale_summary_sha256")
                    != summary_sha256
                    or scaled_resources is None
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", scaled_resource_sha256
                    )
                    or scaled_resource_sha256
                    != _canonical_json_sha256(scaled_resources)
                    or not re.fullmatch(
                        r"[0-9a-f]{64}", base_resource_sha256
                    )
                    or isinstance(effective_active_total, bool)
                    or not isinstance(effective_active_total, int)
                    or effective_active_total <= 0
                    or isinstance(max_workers_per_node, bool)
                    or not isinstance(max_workers_per_node, int)
                    or max_workers_per_node <= 0
                    or scaled_resources.get("max_workers_per_node")
                    != max_workers_per_node
                    or not isinstance(island_quotas, dict)
                    or not island_quotas
                    or any(
                        not isinstance(name, str)
                        or not name.strip()
                        or isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for name, value in island_quotas.items()
                    )
                    or sum(quota_values) != effective_active_total
                    or any(
                        not isinstance(task, dict)
                        or task.get("scale_policy_sha256")
                        != scale_policy_sha256
                        for task in tasks
                    )
                ):
                    raise ValueError(
                        "current7 scale summary identity/SHA is invalid"
                    )
                scale_identity = {
                    "scale_policy_sha256": scale_policy_sha256,
                    "scale_summary_sha256": summary_sha256,
                }
            expected_identity = {
                "schema_version": CURRENT7_SNAPSHOT_IDENTITY_SCHEMA,
                "bundle_id": status.get("bundle_id"),
                "bundle_manifest_sha256": status.get(
                    "bundle_manifest_sha256"
                ),
                "task_schema_version": status.get("task_schema_version"),
                "result_schema_version": status.get("result_schema_version"),
                "inventory_sha256": _canonical_json_sha256(tasks),
                "terminal_record_sha256": sorted(record_hashes),
                "refusals_sha256": _canonical_json_sha256(refusals),
                "aggregate_sha256": _canonical_json_sha256(aggregate),
                **scale_identity,
            }
            if identity != expected_identity:
                raise ValueError("current7 snapshot identity replay failed")
            snapshot_sha = _canonical_json_sha256(identity)
            if (
                status.get("snapshot_sha256") != snapshot_sha
                or index.get("snapshot_sha256") != snapshot_sha
            ):
                raise ValueError("current7 snapshot SHA mismatch")

            if (
                status.get("bundle_id") != index.get("bundle_id")
                or status.get("bundle_manifest_sha256")
                != index.get("bundle_manifest_sha256")
            ):
                raise ValueError("current7 bundle identity is inconsistent")
            for field in CURRENT7_IDENTITY_FIELDS:
                if not (
                    index.get(field) == status.get(field) == aggregate.get(field)
                ):
                    raise ValueError(
                        f"current7 {field} identity is inconsistent"
                    )
            temperatures = status.get("temperature_targets")
            if temperatures != list(CURRENT7_TEMPERATURE_TARGETS):
                raise ValueError("current7 temperature targets are not exact seven")
            constraint_names = status.get("constraint_names")
            terminal_count = len(records)
            hard_spec = status.get("hard_spec")
            hard_spec_sha256 = str(
                status.get("hard_spec_sha256") or ""
            ).lower()
            contract_identity_valid = bool(
                isinstance(hard_spec, dict)
                and re.fullmatch(r"[0-9a-f]{64}", hard_spec_sha256)
                and _canonical_json_sha256(hard_spec) == hard_spec_sha256
                and str(status.get("constraint_version") or "").strip()
                and all(
                    re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(status.get(field) or "").lower(),
                    )
                    for field in (
                        "hard_constraint_contract_sha256",
                        "temperature_contract_sha256",
                    )
                )
            )
            resonance_contract = None
            if condition_display_only:
                resonance_contract = self._validated_current7_condition_contract(
                    hard_spec,
                    constraint_names,
                )
                exact_constraint_contract = contract_identity_valid
            else:
                exact_constraint_contract = bool(
                    contract_identity_valid
                    and hard_spec == CURRENT7_HARD_SPEC
                    and hard_spec_sha256 == CURRENT7_HARD_SPEC_SHA256
                    and constraint_names == list(CURRENT7_CONSTRAINT_NAMES)
                )
            legacy_empty_contract = bool(
                not condition_display_only
                and not terminal_count
                and status.get("constraint_version") is None
                and status.get("hard_spec") is None
                and status.get("hard_spec_sha256") is None
                and status.get("hard_constraint_contract_sha256") is None
                and status.get("temperature_contract_sha256") is None
                and constraint_names == []
            )
            if terminal_count and not exact_constraint_contract:
                raise ValueError("current7 hard-constraint contract drifted")
            if not terminal_count and not (
                exact_constraint_contract or legacy_empty_contract
            ):
                raise ValueError("empty current7 cohort constraint contract drifted")
            constraint_identity_sha256 = (
                _canonical_json_sha256({
                    "constraint_version": status.get("constraint_version"),
                    "hard_spec": status.get("hard_spec"),
                    "hard_spec_sha256": status.get("hard_spec_sha256"),
                    "hard_constraint_contract_sha256": status.get(
                        "hard_constraint_contract_sha256"
                    ),
                    "temperature_contract_sha256": status.get(
                        "temperature_contract_sha256"
                    ),
                    "constraint_names": constraint_names,
                    "temperature_targets": temperatures,
                })
                if exact_constraint_contract else None
            )

            state_counts = Counter()
            lanes = []
            seen_task_identities: set[tuple[str, int, int]] = set()
            completed_task_identities: set[tuple[str, int, int]] = set()
            completed_tasks_by_logical_identity: dict[
                tuple[str, int], set[tuple[str, int, int]]
            ] = defaultdict(set)
            for position, task in enumerate(tasks):
                if not isinstance(task, dict):
                    raise ValueError(f"current7 task {position} is invalid")
                raw_id = task.get("id")
                raw_task_id = task.get("task_id")
                if (
                    raw_id is not None
                    and raw_task_id is not None
                    and (
                        _current7_identity_integer(raw_id) < 0
                        or _current7_identity_integer(raw_task_id) < 0
                        or raw_id != raw_task_id
                    )
                ):
                    raise ValueError(
                        "current7 task id/task_id aliases are inconsistent"
                    )
                task_id = _current7_identity_integer(
                    raw_id if raw_id is not None else raw_task_id
                )
                seed = _current7_identity_integer(task.get("seed"))
                bundle_id = _safe_text(task.get("bundle_id"), 500)
                task_identity = (bundle_id or "", task_id, seed)
                state = (_safe_text(task.get("status"), 40) or "unknown").lower()
                if (
                    task_id < 0
                    or seed < 0
                    or bundle_id not in allowed_bundle_manifests
                    or task_identity in seen_task_identities
                    or (
                        mixed_bundle_projection
                        and task.get("bundle_manifest_sha256")
                        != allowed_bundle_manifests[bundle_id]
                    )
                    or state not in CURRENT7_TASK_STATES
                ):
                    raise ValueError("current7 task identity is invalid or duplicate")
                seen_task_identities.add(task_identity)
                if state == "completed":
                    completed_task_identities.add(task_identity)
                    completed_tasks_by_logical_identity[
                        (bundle_id, seed)
                    ].add(task_identity)
                state_counts[state] += 1
                active = state in {"queued", "attaching", "running"}
                lanes.append({
                    "lane": (
                        f"tier1-current7-task-{bundle_id}-{task_id}-{seed}"
                        if mixed_bundle_projection
                        else f"tier1-current7-task-{task_id}"
                    ),
                    "name": _safe_text(task.get("name"), 160)
                    or f"current7-{task_id}",
                    "task_id": task_id,
                    "seed": seed,
                    "state": state,
                    "alive": active,
                    "workers": 1 if active else 0,
                    "model_id": f"current7:{status.get('bundle_id')}",
                    "model_lane": "corrected_current7",
                    "updated_at": status.get("updated_at"),
                    "source_kind": "tier1_current7_secondary_index",
                })
            if dict(sorted(state_counts.items())) != status.get("state_counts"):
                raise ValueError("current7 task state counts are inconsistent")
            record_terminal_identities: set[tuple[str, int, int]] = set()
            for logical_identity, declared_task_ids in record_task_ids.items():
                matching_completed = completed_tasks_by_logical_identity.get(
                    logical_identity, set()
                )
                if declared_task_ids is None:
                    if len(matching_completed) != 1:
                        raise ValueError(
                            "current7 terminal record references an unknown or "
                            "ambiguous completed task identity"
                        )
                    record_terminal_identities.update(matching_completed)
                    continue
                bundle_id, seed = logical_identity
                declared_identities = {
                    (bundle_id, task_id, seed)
                    for task_id in declared_task_ids
                }
                if (
                    not declared_identities
                    or not declared_identities.issubset(matching_completed)
                ):
                    raise ValueError(
                        "current7 terminal record references an unknown or "
                        "non-completed task identity"
                    )
                record_terminal_identities.update(declared_identities)
            overlapping_terminal_identities = (
                record_terminal_identities & refusal_identities
            )
            if overlapping_terminal_identities:
                raise ValueError(
                    "current7 terminal outcome partition overlaps for seed(s): "
                    + ", ".join(
                        str(seed)
                        for seed in sorted({
                            identity[2]
                            for identity in overlapping_terminal_identities
                        })
                    )
                )
            for bundle_id, refusal_task_id, seed in refusal_identities:
                if (bundle_id, refusal_task_id, seed) not in (
                    completed_task_identities
                ):
                    raise ValueError(
                        "current7 refusal references an unknown or non-completed "
                        "task identity: "
                        f"bundle_id={bundle_id}, task_id={refusal_task_id}, "
                        f"seed={seed}"
                    )
            terminal_outcome_identities = (
                record_terminal_identities | refusal_identities
            )
            if completed_task_identities != terminal_outcome_identities:
                raise ValueError(
                    "current7 terminal outcome partition does not match "
                    "completed tasks"
                )
            if (
                status.get("scheduler_task_count") != len(tasks)
                or status.get("authenticated_terminal_seed_count") != len(records)
                or status.get("refused_terminal_count") != len(refusals)
                or aggregate.get("authenticated_seed_count") != len(records)
                or status.get("healthy") is not (not refusals)
            ):
                raise ValueError("current7 inventory counts are inconsistent")

            status_event_at_text = _safe_text(
                index.get("status_event_at"), 100
            )
            status_updated_at_text = _safe_text(
                status.get("updated_at"), 100
            )
            if (
                not status_event_at_text
                or status_event_at_text != status_updated_at_text
            ):
                raise ValueError("current7 status event identity is inconsistent")
            status_event_at = _parse_time(
                status_event_at_text, timezone.utc
            )
            harvest_observed_at_text = _safe_text(
                index.get("harvest_observed_at"), 100
            )
            harvest_observed_at = _parse_time(
                harvest_observed_at_text, timezone.utc
            )
            if status_event_at is None:
                raise ValueError("current7 status_event_at is invalid")
            if harvest_observed_at is None:
                raise ValueError("current7 harvest_observed_at is invalid")
            now = self.clock()
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            freshness_age_seconds = (
                now.astimezone(timezone.utc)
                - harvest_observed_at.astimezone(timezone.utc)
            ).total_seconds()
            active_task_count = sum(
                state_counts.get(name, 0)
                for name in ("queued", "attaching", "running")
            )
            freshness_required = active_task_count > 0
            freshness_ok = bool(
                not freshness_required
                or (
                    -CURRENT7_FUTURE_SKEW_TOLERANCE_SECONDS
                    <= freshness_age_seconds
                    <= CURRENT7_ACTIVE_FRESHNESS_MAX_AGE_SECONDS
                )
            )

            candidates = aggregate.get("pareto_candidates")
            candidates = candidates if isinstance(candidates, list) else None
            if candidates is None:
                raise ValueError("current7 Pareto preview is invalid")
            pareto_count = _integer(aggregate.get("pareto_count"), -1)
            if (
                pareto_count < 0
                or len(candidates) != min(
                    pareto_count, CURRENT7_CANDIDATE_PREVIEW_LIMIT
                )
                or aggregate.get("pareto_preview_truncated")
                is not (pareto_count > len(candidates))
            ):
                raise ValueError("current7 Pareto preview count is inconsistent")
            candidate_rows = []
            prior_objectives = None
            candidate_source_role = (
                "current7_condition" if condition_display_only else "current7"
            )
            candidate_source_label = (
                "read-only-current7-condition"
                if condition_display_only else "corrected-current7"
            )

            def validate_candidate_provenance(
                candidate: dict[str, Any], position_label: str,
            ) -> None:
                seed = _current7_identity_integer(candidate.get("source_seed"))
                if (
                    candidate.get("source_bundle_id")
                    != status.get("bundle_id")
                    or seed not in record_seeds
                    or any(
                        candidate.get(flag) is not False
                        for flag in CURRENT7_FAIL_CLOSED_FLAGS
                    )
                ):
                    raise ValueError(
                        "current7 candidate provenance is invalid at "
                        f"{position_label}"
                    )

            for position, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    raise ValueError(f"current7 candidate {position} is invalid")
                validate_candidate_provenance(candidate, str(position))
                decoded = candidate.get("decoded_params")
                if not isinstance(decoded, dict) or not decoded:
                    raise ValueError("current7 candidate decoded identity is missing")
                if candidate.get("candidate_identity_sha256") != (
                    _canonical_json_sha256(decoded)
                ):
                    raise ValueError("current7 candidate identity hash mismatch")
                if condition_display_only:
                    physical_g = candidate.get("physical_constraint_G")
                    physical_g = (
                        physical_g if isinstance(physical_g, dict) else {}
                    )
                    expected_resonance_constraints = {
                        name for name in (
                            resonance_contract.get("minimum_constraint"),
                            resonance_contract.get("maximum_constraint"),
                        )
                        if isinstance(name, str) and name
                    }
                    observed_resonance_constraints = {
                        name for name in (
                            CURRENT7_RESONANCE_MINIMUM_CONSTRAINT,
                            CURRENT7_RESONANCE_MAXIMUM_CONSTRAINT,
                        )
                        if name in physical_g
                    }
                    if (
                        observed_resonance_constraints
                        != expected_resonance_constraints
                        or any(
                            (value := _finite_number(physical_g.get(name)))
                            is None or value > 1e-12
                            for name in expected_resonance_constraints
                        )
                    ):
                        raise ValueError(
                            "current7 condition candidate resonance direction "
                            "is inconsistent"
                        )
                volume = _finite_number(candidate.get("volume_L"))
                loss = _finite_number(candidate.get("total_loss_W"))
                violation = _finite_number(
                    candidate.get("total_positive_violation")
                )
                if (
                    volume is None or loss is None or violation is None
                    or volume < 0 or loss < 0 or violation > 1e-12
                    or candidate.get("feasible") is not True
                ):
                    raise ValueError("current7 Pareto candidate is not feasible")
                objectives = (volume, loss)
                if prior_objectives is not None and objectives < prior_objectives:
                    raise ValueError("current7 Pareto preview order is invalid")
                prior_objectives = objectives
                candidate_rows.append({
                    "seed": _integer(candidate.get("source_seed"), -1),
                    "candidate": candidate,
                    "source_role": candidate_source_role,
                    "source_label": candidate_source_label,
                    "source_cohort_id": status.get("cohort_id"),
                })
            least = aggregate.get("least_violation_candidate")
            near_rows = []
            if least is not None:
                if not isinstance(least, dict):
                    raise ValueError(
                        "current7 least-violation candidate is invalid"
                    )
                validate_candidate_provenance(least, "least-violation")
                least_decoded = least.get("decoded_params")
                least_violation = _finite_number(
                    least.get("total_positive_violation")
                )
                if (
                    not isinstance(least_decoded, dict)
                    or not least_decoded
                    or least.get("candidate_identity_sha256")
                    != _canonical_json_sha256(least_decoded)
                    or least_violation is None
                    or least_violation < 0.0
                    or _finite_number(least.get("volume_L")) is None
                    or _finite_number(least.get("total_loss_W")) is None
                ):
                    raise ValueError(
                        "current7 least-violation identity is invalid"
                    )
                if least.get("feasible") is not True:
                    near_rows.append({
                        "seed": _integer(least.get("source_seed"), -1),
                        "candidate": least,
                        "source_role": candidate_source_role,
                        "source_label": candidate_source_label,
                        "source_cohort_id": status.get("cohort_id"),
                    })

            warnings = []
            if refusals:
                warnings.append(
                    f"current7 refused {len(refusals)} terminal artifact(s)"
                )
            if freshness_required and not freshness_ok:
                if condition_display_only:
                    warnings.append(
                        "current7 condition harvester heartbeat is stale or "
                        "future-dated; the authenticated snapshot remains "
                        "read-only"
                    )
                else:
                    warnings.append(
                        "current7 active harvester heartbeat is stale or "
                        "future-dated "
                        f"({freshness_age_seconds:.1f}s); authority was rejected"
                    )
            result = {
                **base,
                "available": True,
                "integrity_verified": True,
                "healthy": status.get("healthy") is True,
                "constraint_contract_verified": exact_constraint_contract,
                "authority_eligible": bool(
                    not condition_display_only
                    and status.get("healthy") is True
                    and exact_constraint_contract
                    and not refusals
                    and freshness_ok
                ),
                "freshness_required": freshness_required,
                "freshness_ok": freshness_ok,
                "freshness_age_seconds": round(
                    freshness_age_seconds, 3
                ),
                "harvest_observed_at": harvest_observed_at_text,
                "status_event_at": status_event_at_text,
                "updated_at": status_event_at_text,
                "cohort_id": _safe_text(status.get("cohort_id"), 200),
                "bundle_id": _safe_text(status.get("bundle_id"), 200),
                "bundle_manifest_sha256": _safe_text(
                    status.get("bundle_manifest_sha256"), 64
                ),
                "mixed_bundle_projection": mixed_bundle_projection,
                "source_bundle_count": len(allowed_bundle_manifests),
                "source_bundle_cohorts_sha256": (
                    _safe_text(
                        status.get("source_bundle_cohorts_sha256"), 64
                    )
                    if mixed_bundle_projection else None
                ),
                "scale_policy_sha256": scale_policy_sha256,
                "scale_summary": (
                    copy.deepcopy(scale_summary)
                    if scale_summary is not None else None
                ),
                "constraint_version": _safe_text(
                    status.get("constraint_version"), 200
                ),
                "hard_spec_sha256": _safe_text(
                    status.get("hard_spec_sha256"), 64
                ),
                "hard_constraint_contract_sha256": _safe_text(
                    status.get("hard_constraint_contract_sha256"), 64
                ),
                "temperature_contract_sha256": _safe_text(
                    status.get("temperature_contract_sha256"), 64
                ),
                "constraint_identity_sha256": constraint_identity_sha256,
                "resonance_contract": resonance_contract,
                "constraints": (
                    dict(status["hard_spec"])
                    if isinstance(status.get("hard_spec"), dict) else {}
                ),
                "constraint_names": list(constraint_names),
                "temperature_targets": list(temperatures),
                "search_count": len(tasks),
                "running_count": state_counts.get("running", 0),
                "queued_count": state_counts.get("queued", 0),
                "attaching_count": state_counts.get("attaching", 0),
                "completed_count": state_counts.get("completed", 0),
                "failed_count": state_counts.get("failed", 0),
                "cancelled_count": state_counts.get("cancelled", 0),
                "timeout_count": state_counts.get("timeout", 0),
                "active_plus_queued": sum(
                    state_counts.get(name, 0)
                    for name in ("queued", "attaching", "running")
                ),
                "authenticated_terminal_seed_count": len(records),
                "refused_terminal_count": len(refusals),
                "feasible_pareto_count": _integer(
                    aggregate.get("pareto_count"), 0
                ),
                "candidate_preview_count": len(candidate_rows),
                "candidate_preview_limit": CURRENT7_CANDIDATE_PREVIEW_LIMIT,
                "candidate_preview_truncated": aggregate.get(
                    "pareto_preview_truncated"
                ) is True,
                "near_feasible_count": len(near_rows),
                "candidate_rows": candidate_rows,
                "near_candidate_rows": near_rows,
                "state_counts": dict(sorted(state_counts.items())),
                "lanes": [] if condition_display_only else lanes,
                "snapshot_sha256": snapshot_sha,
                "snapshot_file_sha256": snapshot["status_file_sha256"],
                "index_file_sha256": snapshot["index_file_sha256"],
                "snapshot_attempts": snapshot["attempts"],
                "pointer_verified": True,
                "gui_launch_eligible": False,
                "warnings": warnings,
            }
            cache_key = (
                f"{'condition' if condition_display_only else 'primary'}:"
                f"{index_path}"
            )
            self._current7_diagnostics_cache[cache_key] = {
                "status_file_sha256": snapshot["status_file_sha256"],
                "index_cache_contract": (
                    self._current7_index_cache_contract(index)
                ),
                "diagnostics": copy.deepcopy(result),
            }
            return result
        except (OSError, TypeError, UnicodeError, ValueError) as exc:
            return {
                **base,
                "available": True,
                "warnings": [f"{diagnostic_name} rejected: {exc}"],
            }

    def _tier1_current7_condition_diagnostics(
        self,
    ) -> list[dict[str, Any]]:
        """Authenticate configured condition indexes as read-only evidence."""

        with self._current7_diagnostics_lock:
            diagnostics = []
            for index_path in self._tier1_current7_condition_indexes:
                try:
                    schema = self._condition_index_schema(index_path)
                except (OSError, TypeError, UnicodeError, ValueError) as exc:
                    diagnostics.append({
                        "configured": True,
                        "available": True,
                        "integrity_verified": False,
                        "healthy": False,
                        "constraint_contract_verified": False,
                        "authority_eligible": False,
                        "display_only": True,
                        "read_only": True,
                        "source": str(index_path),
                        "warnings": [
                            f"configured condition index rejected: {exc}"
                        ],
                        "lanes": [],
                        "candidate_rows": [],
                        "near_candidate_rows": [],
                    })
                    continue
                if schema == CURRENT7_COMPACT_INDEX_SCHEMA:
                    diagnostics.append(
                        self._tier1_compact_condition_diagnostics_locked(
                            index_path=index_path,
                            hydrate_candidates=False,
                        )
                    )
                else:
                    diagnostics.append(
                        self._tier1_current7_diagnostics_locked(
                            index_path=index_path,
                            condition_display_only=True,
                        )
                    )
            return diagnostics

    @staticmethod
    def _tier1_current7_near_feasible_preview(
        diagnostics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Expose one authenticated least-violation point without authority."""
        if diagnostics.get("integrity_verified") is not True:
            return []
        preview = []
        for position, record in enumerate(
            (diagnostics.get("near_candidate_rows") or [])[:1]
        ):
            if not isinstance(record, dict):
                continue
            candidate = record.get("candidate")
            if not isinstance(candidate, dict):
                continue
            constraint_g = candidate.get("physical_constraint_G")
            constraint_g = constraint_g if isinstance(constraint_g, dict) else {}
            violations = [
                {"constraint": str(name), "amount": amount}
                for name, raw in constraint_g.items()
                if (amount := _finite_number(raw)) is not None
                and amount > 1e-9
            ]
            violations.sort(
                key=lambda item: (-item["amount"], item["constraint"])
            )
            decoded = candidate.get("decoded_params")
            decoded = decoded if isinstance(decoded, dict) else {}
            condition_display_only = diagnostics.get("display_only") is True
            resonance_contract = diagnostics.get("resonance_contract")
            resonance_contract = (
                resonance_contract
                if isinstance(resonance_contract, dict) else {}
            )
            preview.append({
                "id": (
                    "near-current7-seed-"
                    f"{_integer(record.get('seed'), -1)}-{position:04d}"
                ),
                "source_role": (
                    "current7_condition"
                    if condition_display_only else "current7"
                ),
                "source_label": (
                    "read-only-current7-condition"
                    if condition_display_only else "corrected-current7"
                ),
                "source_cohort_id": diagnostics.get("cohort_id"),
                "task_id": None,
                "seed": _integer(record.get("seed"), -1),
                "terminal_population_index": _integer(
                    candidate.get("terminal_population_index"), position
                ),
                "target_acquisition_score": _finite_number(
                    candidate.get("total_positive_violation")
                ),
                "size_W_mm": _finite_number(candidate.get("size_W_mm")),
                "size_L_mm": _finite_number(candidate.get("size_L_mm")),
                "size_H_mm": _finite_number(candidate.get("size_H_mm")),
                "volume_L": _finite_number(candidate.get("volume_L")),
                "total_loss_W": _finite_number(candidate.get("total_loss_W")),
                "pred_Llt_phys_uH": _finite_number(
                    candidate.get("pred_Llt_phys")
                ),
                "Llt_q90_half_width_uH": _finite_number(
                    candidate.get("q90_Llt_phys_half_width_uH")
                ),
                "f_res_min_screen_Hz": _finite_number(
                    candidate.get("pred_f_res_min_screen_Hz")
                ),
                "resonance_minimum_G_Hz": _finite_number(
                    constraint_g.get(
                        resonance_contract.get("minimum_constraint")
                    )
                ),
                "resonance_maximum_G_Hz": _finite_number(
                    constraint_g.get(
                        resonance_contract.get("maximum_constraint")
                    )
                ),
                "resonance_direction": resonance_contract.get("direction"),
                "resonance_minimum_operator": resonance_contract.get(
                    "minimum_operator"
                ),
                "resonance_maximum_operator": resonance_contract.get(
                    "maximum_operator"
                ),
                "robust_max_temperature_C": _finite_number(
                    candidate.get("robust_max_temperature_C")
                ),
                "B_design_analytic_T": _finite_number(
                    candidate.get("B_design_analytic_T")
                ),
                "n_core_group": _finite_number(decoded.get("n_core_group")),
                "cw1_mm": _finite_number(decoded.get("cw1")),
                "violation_count": len(violations),
                "violations": violations,
                "spec_status": "near_feasible",
                "valid_pareto": False,
                "production_eligible": False,
                "fea_submission_approved": False,
                "gui_launch_eligible": False,
            })
        return preview

    @staticmethod
    def _tier1_rolling_model_identity_matches(
        rolling_index: dict[str, Any], model_pointer: dict[str, Any]
    ) -> bool:
        """Bind a hash-verified model pointer to its rolling generation."""
        current_model = model_pointer.get("current")
        hard_spec = rolling_index.get("hard_spec")
        hard_spec = hard_spec if isinstance(hard_spec, dict) else {}
        hard_spec_sha = str(
            rolling_index.get("hard_spec_sha256") or ""
        ).lower()
        return bool(
            model_pointer.get("schema_version")
            == "mft-tier1-slurm-model-pointer-v1"
            and isinstance(current_model, dict)
            and model_pointer.get("production_eligible") is False
            and model_pointer.get("fea_submission_approved") is False
            and model_pointer.get("automatic_promotion_allowed") is False
            and current_model.get("cohort_id")
            == rolling_index.get("active_cohort_id")
            and current_model.get("constraint_version")
            == rolling_index.get("constraint_version")
            and current_model.get("hard_spec") == hard_spec
            and current_model.get("hard_spec_sha256") == hard_spec_sha
            and current_model.get("source_model_manifest_sha256")
            == rolling_index.get("source_model_manifest_sha256")
            and current_model.get("deployment_model_manifest_sha256")
            == rolling_index.get("deployment_model_manifest_sha256")
            and current_model.get("production_eligible") is False
            and current_model.get("fea_submission_approved") is False
            and _canonical_json_sha256(hard_spec) == hard_spec_sha
        )

    def _tier1_supplemental_nsga_diagnostics(
        self,
    ) -> list[dict[str, Any]]:
        """Authenticate optional rolling runtimes without granting authority.

        A supplemental runtime is rooted explicitly by the monitor launcher.
        It may add progress and display-only candidates, but it can never
        select the active condition generation or replace the primary index.
        """
        diagnostics: list[dict[str, Any]] = []
        for root in self._tier1_nsga_supplemental_roots:
            source_label = root.name
            pointer_path = root / "canonical" / "index.json"
            try:
                resolved_root = root.resolve(strict=True)
                resolved_pointer = pointer_path.resolve(strict=True)
                if (
                    root.is_symlink()
                    or resolved_root != root
                    or pointer_path.is_symlink()
                    or not resolved_pointer.is_file()
                    or not resolved_pointer.is_relative_to(resolved_root)
                ):
                    raise ValueError(
                        "configured supplemental root is not a direct sealed "
                        "runtime"
                    )
                snapshot = self._coherent_rolling_snapshot(resolved_pointer)
                rolling_index = snapshot["index"]
                containment_root = Path(
                    str(rolling_index.get("path_containment_root") or "")
                ).resolve(strict=True)
                if containment_root != resolved_root:
                    raise ValueError(
                        "Tier-1 supplemental index containment root mismatch"
                    )
                model_pointer = snapshot["model_pointer"]
                if not self._tier1_rolling_model_identity_matches(
                    rolling_index, model_pointer
                ):
                    raise ValueError(
                        "Tier-1 supplemental model/constraint identity mismatch"
                    )
                if not all(
                    rolling_index.get(key) is False for key in (
                        "production_eligible", "fea_submission_approved",
                        "fea_submission_performed", "aedt_used",
                    )
                ):
                    raise ValueError(
                        "Tier-1 supplemental index attempted production/FEA "
                        "authority"
                    )
                status = snapshot["status"]
                cohort_id = _safe_text(status.get("cohort_id"), 200) or ""
                manifest_path = (
                    resolved_root / "deployments" / cohort_id / "bundle"
                    / "bundle_manifest.json"
                )
                resolved_manifest_path = manifest_path.resolve(strict=True)
                if (
                    not cohort_id
                    or manifest_path.is_symlink()
                    or not resolved_manifest_path.is_file()
                    or not resolved_manifest_path.is_relative_to(resolved_root)
                ):
                    raise ValueError(
                        "Tier-1 supplemental bundle manifest escapes its "
                        "sealed runtime"
                    )
                manifest, manifest_bytes = _bounded_json_bytes(
                    resolved_manifest_path, 2 * 1024 * 1024
                )
                manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
                status_manifest_sha = str(
                    status.get("bundle_manifest_sha256") or ""
                ).lower()
                manifest_temperature_contract = manifest.get(
                    "temperature_constraint_contract"
                )
                manifest_temperature_contract = (
                    manifest_temperature_contract
                    if isinstance(manifest_temperature_contract, dict)
                    else {}
                )
                manifest_identity_ok = bool(
                    re.fullmatch(r"[0-9a-f]{64}", status_manifest_sha)
                    and manifest_sha == status_manifest_sha
                    and manifest.get("schema_version")
                    == "mft-tier1-slurm-deployment-v1"
                    and manifest.get("cohort_id") == cohort_id
                    and manifest.get("constraint_version")
                    == status.get("constraint_version")
                    and manifest.get("hard_spec") == status.get("hard_spec")
                    and manifest.get("hard_spec_sha256")
                    == status.get("hard_spec_sha256")
                    and manifest.get("source_model_manifest_sha256")
                    == status.get("source_model_manifest_sha256")
                    and manifest.get("deployment_model_manifest_sha256")
                    == status.get("deployment_model_manifest_sha256")
                    and manifest.get("nsga_code_revision")
                    == status.get("nsga_code_revision")
                    and manifest.get("attestation_schema_version")
                    == status.get("attestation_schema_version")
                    == rolling_index.get("attestation_schema_version")
                    and manifest.get("task_schema_version")
                    == status.get("task_schema_version")
                    == rolling_index.get("task_schema_version")
                    and manifest_temperature_contract
                    == status.get("temperature_constraint_contract")
                    and manifest.get(
                        "temperature_constraint_contract_sha256"
                    ) == status.get(
                        "temperature_constraint_contract_sha256"
                    ) == rolling_index.get(
                        "temperature_constraint_contract_sha256"
                    )
                    and _canonical_json_sha256(
                        manifest_temperature_contract
                    ) == manifest.get(
                        "temperature_constraint_contract_sha256"
                    )
                    and manifest.get("production_eligible") is False
                    and manifest.get("fea_submission_approved") is False
                    and manifest.get("automatic_promotion_allowed") is False
                )
                if not manifest_identity_ok:
                    raise ValueError(
                        "Tier-1 supplemental bundle manifest identity mismatch"
                    )
                search_profile = status.get("search_profile")
                search_profile = (
                    search_profile if isinstance(search_profile, dict) else {}
                )
                source_label = (
                    _safe_text(search_profile.get("namespace"), 100)
                    or source_label
                )
                result = self._tier1_rolling_nsga_diagnostics(
                    path=snapshot["status_path"],
                    status=status,
                    rolling_index=rolling_index,
                    base={
                        "configured": True,
                        "source": str(snapshot["status_path"]),
                        "pointer": str(resolved_pointer),
                        "pointer_verified": True,
                        "constraint_version": _safe_text(
                            rolling_index.get("constraint_version"), 160
                        ),
                        "constraints": (
                            dict(rolling_index["hard_spec"])
                            if isinstance(
                                rolling_index.get("hard_spec"), dict
                            ) else {}
                        ),
                        "display_only": True,
                        "production_eligible": False,
                        "fea_submission_approved": False,
                        "warnings": [],
                        "lanes": [],
                        "candidate_rows": [],
                        "coherent_snapshot_verified": True,
                        "coherent_snapshot_attempts": snapshot.get("attempts"),
                    },
                    warnings=[],
                )
                result.update({
                    "source_role": "supplemental",
                    "source_label": source_label,
                    "source_root": str(resolved_root),
                    "bundle_manifest_sha256": manifest_sha,
                    "bundle_manifest_verified": True,
                })
                diagnostics.append(result)
            except (OSError, TypeError, UnicodeError, ValueError) as exc:
                diagnostics.append({
                    "available": False,
                    "configured": True,
                    "integrity_verified": False,
                    "pointer_verified": False,
                    "source_role": "supplemental",
                    "source_label": source_label,
                    "source_root": str(root),
                    "source": str(pointer_path),
                    "lanes": [],
                    "candidate_rows": [],
                    "warnings": [
                        f"Tier-1 supplemental source rejected: {exc}"
                    ],
                })
        return diagnostics

    def _tier1_rolling_nsga_diagnostics(
        self,
        *,
        path: Path,
        status: dict[str, Any],
        rolling_index: dict[str, Any] | None,
        base: dict[str, Any],
        warnings: list[str],
        historical_snapshot: bool = False,
    ) -> dict[str, Any]:
        """Project an authenticated continuously-refilled Slurm search.

        The live pointer requires a fresh heartbeat.  A condition generation
        that has been superseded is intentionally stale, so an explicitly
        configured historical cohort may skip only that liveness gate.  All
        identity, containment, hash, terminal-result, aggregate-Pareto, and
        non-production checks remain identical to the live projection.
        """
        if rolling_index is None or base.get("pointer_verified") is not True:
            warnings.append("Tier-1 rolling status has no verified stable index")
            return {
                **base, "available": False, "integrity_verified": False,
                "warnings": list(dict.fromkeys(warnings)),
            }
        hard_spec = status.get("hard_spec")
        hard_spec = hard_spec if isinstance(hard_spec, dict) else {}
        hard_spec_sha = str(status.get("hard_spec_sha256") or "").lower()
        source_model_sha = str(
            status.get("source_model_manifest_sha256") or ""
        ).lower()
        deployment_model_sha = str(
            status.get("deployment_model_manifest_sha256") or ""
        ).lower()
        code_revision = str(status.get("nsga_code_revision") or "").lower()
        controller = status.get("controller")
        controller = controller if isinstance(controller, dict) else {}
        temperature_contract = status.get("temperature_constraint_contract")
        temperature_contract = (
            temperature_contract
            if isinstance(temperature_contract, dict) else {}
        )
        index_temperature_contract = rolling_index.get(
            "temperature_constraint_contract"
        )
        index_temperature_contract = (
            index_temperature_contract
            if isinstance(index_temperature_contract, dict) else {}
        )
        temperature_targets = temperature_contract.get("targets")
        temperature_targets = (
            temperature_targets if isinstance(temperature_targets, list)
            else []
        )
        controller_mode_ok = _tier1_controller_mode_verified(
            controller, historical_snapshot=historical_snapshot,
        )
        identity_ok = bool(
            status.get("cohort_id") == rolling_index.get("active_cohort_id")
            and status.get("constraint_version")
            == rolling_index.get("constraint_version")
            and hard_spec == rolling_index.get("hard_spec")
            and hard_spec_sha == rolling_index.get("hard_spec_sha256")
            and _canonical_json_sha256(hard_spec) == hard_spec_sha
            and source_model_sha
            == rolling_index.get("source_model_manifest_sha256")
            and deployment_model_sha
            == rolling_index.get("deployment_model_manifest_sha256")
            and re.fullmatch(r"[0-9a-f]{64}", source_model_sha)
            and re.fullmatch(r"[0-9a-f]{64}", deployment_model_sha)
            and re.fullmatch(r"[0-9a-f]{40}", code_revision)
            and status.get("healthy") is True
            and status.get("error") is None
            and _integer(controller.get("pid"), 0) > 0
            and controller_mode_ok
            and temperature_contract == index_temperature_contract
            and temperature_targets == list(CANDIDATE_TEMPERATURE_TARGETS)
            and _integer(temperature_contract.get("target_count"), -1)
            == len(CANDIDATE_TEMPERATURE_TARGETS)
            and _finite_number(
                temperature_contract.get("robust_upper_bound_C")
            ) == _finite_number(hard_spec.get("T_limit_C"))
        )
        production_safe = bool(
            all(status.get(key) is False for key in (
                "production_eligible", "fea_submission_approved",
                "fea_submission_performed", "aedt_used",
            ))
        )
        if not identity_ok:
            warnings.append("Tier-1 rolling status/index identity mismatch")
        if not production_safe:
            warnings.append(
                "Tier-1 rolling status attempted production/FEA/AEDT authority"
            )

        updated_at = _safe_text(status.get("updated_at"), 100)
        updated_time = _parse_time(updated_at, self.clock().tzinfo)
        age_seconds = (
            max(0.0, (self.clock() - updated_time).total_seconds())
            if updated_time is not None else None
        )
        freshness_deadline = _finite_number(
            status.get("freshness_deadline_seconds")
        )
        freshness_ok = bool(
            freshness_deadline is not None
            and 1 <= freshness_deadline <= 300
            and age_seconds is not None
            and age_seconds <= freshness_deadline
        )
        freshness_warning = _tier1_staleness_warning(
            freshness_ok=freshness_ok,
            historical_snapshot=historical_snapshot,
        )
        if freshness_warning:
            warnings.append(freshness_warning)

        raw_tasks = status.get("latest_tasks")
        raw_tasks = raw_tasks if isinstance(raw_tasks, list) else []
        tasks_ok = bool(raw_tasks)
        seen_task_ids: set[int] = set()
        seen_seeds: set[int] = set()
        state_counts: Counter[str] = Counter()
        lanes: list[dict[str, Any]] = []
        allowed_states = {
            "queued", "attaching", "running", "completed", "failed",
            "cancelled", "canceled",
        }
        for position, task in enumerate(raw_tasks[:10_000]):
            if not isinstance(task, dict):
                tasks_ok = False
                continue
            task_id = _integer(task.get("task_id"), -1)
            seed = _integer(task.get("seed"), -1)
            task_state = (_safe_text(task.get("status"), 40) or "unknown").lower()
            task_ok = bool(
                task_id > 0
                and task_id not in seen_task_ids
                and seed >= 0
                and seed not in seen_seeds
                and task_state in allowed_states
                and _safe_text(task.get("name"), 300)
                and _integer(task.get("cpus"), 0) > 0
                and _integer(task.get("memory_mb"), 0) > 0
            )
            if not task_ok:
                warnings.append(
                    f"Tier-1 rolling task identity invalid at row {position}"
                )
                tasks_ok = False
            seen_task_ids.add(task_id)
            seen_seeds.add(seed)
            state_counts[task_state] += 1
            lanes.append({
                "name": _safe_text(task.get("name"), 300),
                "lane": f"tier1-rolling-seed-{seed}",
                "seed": seed if seed >= 0 else None,
                "state": task_state,
                "task_id": task_id if task_id > 0 else None,
                "slurm_job_id": _safe_text(task.get("slurm_job_id"), 100),
                "account_name": _safe_text(task.get("account_name"), 100),
                "workers": 1 if task_state == "running" else 0,
                "cpus": max(0, _integer(task.get("cpus"), 0)),
                "memory_mb": max(0, _integer(task.get("memory_mb"), 0)),
                "model_id": f"tier1-feedback:{source_model_sha}",
                "model_lane": "experimental_tier1_feedback",
                "integrity_verified": task_ok,
                "source_kind": "tier1_slurm_rolling_status",
                "source": str(path),
            })

        declared_state_counts = status.get("state_counts")
        declared_state_counts = (
            declared_state_counts
            if isinstance(declared_state_counts, dict) else {}
        )
        normalized_declared_counts = {
            str(key): max(0, _integer(value, 0))
            for key, value in declared_state_counts.items()
        }
        scheduler_task_count = max(
            0, _integer(status.get("scheduler_task_count"), 0)
        )
        exact_counts_ok = bool(
            scheduler_task_count == len(raw_tasks) == len(lanes)
            and dict(sorted(state_counts.items()))
            == dict(sorted(normalized_declared_counts.items()))
        )
        rolling = status.get("rolling")
        rolling = rolling if isinstance(rolling, dict) else {}
        active_plus_queued = sum(
            state_counts.get(key, 0) for key in ("queued", "attaching", "running")
        )
        rolling_ok = bool(
            rolling.get("enabled") is True
            and rolling.get("stop_condition") == "explicit_operator_stop_only"
            and rolling.get("terminal_completion_triggers_refill") is True
            and _integer(rolling.get("seed_identity_count"), -1)
            == len(seen_seeds)
            and _integer(rolling.get("active_plus_queued"), -1)
            == active_plus_queued
            and _integer(rolling.get("duplicate_seed_count"), -1) == 0
        )
        if not exact_counts_ok:
            warnings.append("Tier-1 rolling task/state counts disagree")
        if not rolling_ok:
            warnings.append("Tier-1 rolling refill identity is inconsistent")

        raw_terminal = status.get("terminal_results")
        terminal_results = raw_terminal if isinstance(raw_terminal, list) else []
        terminal_ok = raw_terminal is None or isinstance(raw_terminal, list)
        if not terminal_ok or len(terminal_results) > 4_096:
            warnings.append("Tier-1 rolling terminal result inventory is invalid")
            terminal_ok = False
        containment_root = Path(
            str(rolling_index.get("path_containment_root") or "")
        ).resolve()
        terminal_results_verified = 0
        failed_terminal_results_verified = 0
        verified_terminal_pairs: set[tuple[int, int]] = set()
        terminal_candidates: dict[tuple[int, int], list[dict[str, Any]]] = {}
        terminal_near_candidates: dict[
            tuple[int, int], list[dict[str, Any]]
        ] = {}
        terminal_result_hashes: dict[tuple[int, int], str] = {}
        for position, terminal in enumerate(terminal_results[:4_096]):
            terminal_row_ok = isinstance(terminal, dict)
            terminal = terminal if isinstance(terminal, dict) else {}
            task_id = _integer(terminal.get("task_id"), -1)
            seed = _integer(terminal.get("seed"), -1)
            pair = (task_id, seed)
            remote_identity = terminal.get("remote_status")
            result_identity = terminal.get("result")
            remote_identity = (
                remote_identity if isinstance(remote_identity, dict) else {}
            )
            result_identity = (
                result_identity if isinstance(result_identity, dict) else {}
            )
            scheduler_state = (
                _safe_text(terminal.get("scheduler_state"), 40) or ""
            ).lower()
            terminal_state = (
                _safe_text(terminal.get("terminal_state"), 40) or ""
            ).lower()
            scheduler_exit_code = _integer(
                terminal.get("scheduler_exit_code"), -1
            )
            explicit_exit_code = bool(
                isinstance(terminal.get("scheduler_exit_code"), int)
                and not isinstance(
                    terminal.get("scheduler_exit_code"), bool
                )
            )
            completed_terminal = bool(
                explicit_exit_code
                and scheduler_state == "completed"
                and terminal_state == "completed"
                and scheduler_exit_code == 0
            )
            failure_text = terminal.get("failure")
            failed_terminal = bool(
                explicit_exit_code
                and scheduler_state == "failed"
                and terminal_state == "failed"
                and scheduler_exit_code != 0
                and isinstance(failure_text, str)
                and 0 < len(failure_text) <= 2_000
            )
            terminal_row_ok = bool(
                terminal_row_ok
                and terminal.get("schema_version")
                == "mft-tier1-slurm-terminal-result-v1"
                and terminal.get("authenticated") is True
                and pair not in verified_terminal_pairs
                and task_id > 0 and seed >= 0
                and terminal.get("cohort_id") == status.get("cohort_id")
                and (completed_terminal or failed_terminal)
                and terminal.get("constraint_version")
                == status.get("constraint_version")
                and terminal.get("hard_spec_sha256") == hard_spec_sha
                and terminal.get("source_model_manifest_sha256")
                == source_model_sha
                and all(terminal.get(key) is False for key in (
                    "production_eligible", "fea_submission_approved",
                    "fea_submission_performed",
                ))
            )
            expected_remote_sha = str(
                remote_identity.get("sha256") or ""
            ).lower()
            expected_result_sha = str(
                result_identity.get("sha256") or ""
            ).lower()
            try:
                remote_path = Path(
                    str(remote_identity.get("local_path") or "")
                ).resolve()
                if (
                    not remote_path.is_relative_to(containment_root)
                    or remote_path.is_symlink()
                    or not remote_path.is_file()
                ):
                    raise ValueError("terminal artifact escapes sealed root")
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", expected_remote_sha)
                    or _sha256_file(remote_path) != expected_remote_sha
                ):
                    raise ValueError("terminal artifact hash/size mismatch")
                result_path = containment_root / "unavailable"
                if completed_terminal:
                    result_path = Path(
                        str(result_identity.get("local_path") or "")
                    ).resolve()
                    if (
                        not result_path.is_relative_to(containment_root)
                        or result_path.is_symlink()
                        or not result_path.is_file()
                    ):
                        raise ValueError(
                            "terminal artifact escapes sealed root"
                        )
                    if (
                        not re.fullmatch(
                            r"[0-9a-f]{64}", expected_result_sha
                        )
                        or _sha256_file(result_path) != expected_result_sha
                        or result_path.stat().st_size
                        != _integer(result_identity.get("bytes"), -1)
                    ):
                        raise ValueError(
                            "terminal artifact hash/size mismatch"
                        )
                elif terminal.get("result") is not None or result_identity:
                    raise ValueError(
                        "failed terminal unexpectedly declares a result"
                    )
            except (OSError, TypeError, ValueError) as exc:
                warnings.append(
                    f"Tier-1 rolling terminal {position} rejected: {exc}"
                )
                terminal_row_ok = False
                remote_path = containment_root / "unavailable"
                result_path = containment_root / "unavailable"

            remote_payload: dict[str, Any] = {}
            result_payload: dict[str, Any] = {}
            if terminal_row_ok:
                remote_result = self.cache.json(
                    remote_path, {}, max_bytes=2 * 1024 * 1024,
                    fail_closed=True,
                )
                remote_payload = (
                    remote_result.value
                    if isinstance(remote_result.value, dict) else {}
                )
                result_warning = None
                if completed_terminal:
                    result_result = self.cache.json(
                        result_path, {}, max_bytes=16 * 1024 * 1024,
                        fail_closed=True,
                    )
                    result_payload = (
                        result_result.value
                        if isinstance(result_result.value, dict) else {}
                    )
                    result_warning = result_result.warning
                if remote_result.warning or result_warning:
                    warnings.extend(
                        warning for warning in (
                            remote_result.warning, result_warning
                        ) if warning
                    )
                    terminal_row_ok = False
            common_remote_ok = bool(
                terminal_row_ok
                and remote_payload.get("schema_version")
                == "mft-tier1-slurm-seed-status-v1"
                and _integer(remote_payload.get("task_id"), -1) == task_id
                and _integer(remote_payload.get("seed"), -1) == seed
                and remote_payload.get("cohort_id")
                == status.get("cohort_id")
                and remote_payload.get("nsga_code_revision") == code_revision
                and remote_payload.get("source_model_manifest_sha256")
                == source_model_sha
                and remote_payload.get("deployment_model_manifest_sha256")
                == deployment_model_sha
                and all(remote_payload.get(key) is False for key in (
                    "production_eligible", "fea_submission_approved",
                    "fea_submission_performed", "aedt_used",
                ))
            )
            if failed_terminal:
                terminal_row_ok = bool(
                    common_remote_ok
                    and all(terminal.get(key) is None for key in (
                        "result", "completed_generations",
                        "feasible_pareto_count", "candidate_count",
                    ))
                    and remote_identity.get("state") == "failed"
                    and remote_identity.get("failure") == failure_text
                    and remote_payload.get("state") == "failed"
                    and _integer(remote_payload.get("exit_code"), 0)
                    == scheduler_exit_code
                    and remote_payload.get("failure") == failure_text
                    and remote_payload.get("result_sha256") in (None, "")
                )
                if not terminal_row_ok:
                    terminal_ok = False
                    warnings.append(
                        "Tier-1 rolling terminal identity mismatch at row "
                        f"{position}"
                    )
                    continue
                verified_terminal_pairs.add(pair)
                terminal_candidates[pair] = []
                terminal_near_candidates[pair] = []
                terminal_results_verified += 1
                failed_terminal_results_verified += 1
                continue
            candidates = result_payload.get("candidates")
            candidates = candidates if isinstance(candidates, list) else []
            target_plan = result_payload.get("next_target_fea_batch_plan")
            target_plan = target_plan if isinstance(target_plan, dict) else {}
            raw_target_candidates = target_plan.get("candidates")
            target_candidates = (
                raw_target_candidates
                if isinstance(raw_target_candidates, list) else []
            )
            terminal_row_ok = bool(
                common_remote_ok
                and remote_identity.get("state") == "completed"
                and remote_payload.get("state") == "completed"
                and _integer(remote_payload.get("exit_code"), -1) == 0
                and remote_payload.get("result_sha256") == expected_result_sha
                and result_payload.get("schema_version")
                == "mft-tier1-corrected-search-seed-v1"
                and _integer(result_payload.get("seed"), -1) == seed
                and result_payload.get("constraint_version")
                == status.get("constraint_version")
                and result_payload.get("hard_spec") == hard_spec
                and result_payload.get("hard_spec_sha256") == hard_spec_sha
                and result_payload.get("nsga_code_revision") == code_revision
                and result_payload.get("model_manifest_sha256")
                == deployment_model_sha
                and all(result_payload.get(key) is False for key in (
                    "production_eligible", "fea_submission_approved",
                    "automatic_promotion_allowed",
                ))
                and all(target_plan.get(key) is False for key in (
                    "production_eligible", "fea_submission_approved",
                    "submission_performed",
                    "current_candidates_eligible_for_submission",
                ))
                and target_plan.get("schema_version")
                == "mft-tier1-next-target-fea-batch-plan-v1"
                and target_plan.get("scheduler_cap_verified_before_submission")
                is False
                and _integer(target_plan.get("candidate_count"), -1)
                == len(target_candidates)
                and all(
                    isinstance(item, dict)
                    and item.get("production_eligible") is False
                    and item.get("fea_submission_approved") is False
                    and item.get("eligible_for_submission") is False
                    for item in target_candidates
                )
                and _integer(terminal.get("completed_generations"), -1)
                == _integer(result_payload.get("completed_generations"), -2)
                and _integer(terminal.get("feasible_pareto_count"), -1)
                == _integer(result_payload.get("feasible_pareto_count"), -2)
                == len(candidates)
                and _integer(terminal.get("candidate_count"), -1)
                == len(candidates)
            )
            if not terminal_row_ok:
                terminal_ok = False
                warnings.append(
                    f"Tier-1 rolling terminal identity mismatch at row {position}"
                )
                continue
            verified_candidates = [
                item for item in candidates if isinstance(item, dict)
            ]
            verified_target_candidates = [
                item for item in target_candidates if isinstance(item, dict)
            ]
            if len(verified_candidates) != len(candidates):
                terminal_ok = False
                warnings.append(
                    f"Tier-1 rolling terminal candidate row invalid at {position}"
                )
                continue
            if len(verified_target_candidates) != len(target_candidates):
                terminal_ok = False
                warnings.append(
                    "Tier-1 rolling terminal target candidate row invalid at "
                    f"{position}"
                )
                continue
            verified_terminal_pairs.add(pair)
            terminal_candidates[pair] = verified_candidates
            terminal_near_candidates[pair] = verified_target_candidates[
                :ROLLING_NEAR_FEASIBLE_PER_TERMINAL_LIMIT
            ]
            terminal_result_hashes[pair] = expected_result_sha
            terminal_results_verified += 1

        aggregate = status.get("aggregate_pareto")
        aggregate = aggregate if isinstance(aggregate, dict) else {}
        raw_aggregate_candidates = aggregate.get("candidates")
        aggregate_candidate_inventory_valid = isinstance(
            raw_aggregate_candidates, list
        )
        aggregate_candidates = (
            raw_aggregate_candidates
            if aggregate_candidate_inventory_valid else []
        )
        if not aggregate_candidate_inventory_valid:
            warnings.append(
                "Tier-1 rolling aggregate candidate inventory is not a list"
            )
        candidate_preview = _rolling_candidate_preview_contract(
            aggregate, aggregate_candidates
        )
        if not candidate_preview["safe"]:
            warnings.append(
                "Tier-1 rolling aggregate candidate preview contract mismatch"
            )
        raw_aggregate_near = aggregate.get("near_feasible")
        aggregate_near_inventory_valid = isinstance(
            raw_aggregate_near, list
        )
        aggregate_near = (
            raw_aggregate_near if aggregate_near_inventory_valid else []
        )
        if not aggregate_near_inventory_valid:
            warnings.append(
                "Tier-1 rolling aggregate near-feasible inventory is not a "
                "list"
            )
        declared_near_count = _integer(
            aggregate.get("near_feasible_count"), -1
        )
        authenticated_near_count = sum(
            len(items) for items in terminal_near_candidates.values()
        )
        expected_near_preview_count = min(
            max(0, declared_near_count),
            ROLLING_NEAR_FEASIBLE_PREVIEW_LIMIT,
        )
        aggregate_safe = bool(
            aggregate.get("schema_version")
            == "mft-tier1-slurm-aggregate-pareto-v1"
            and aggregate_candidate_inventory_valid
            and aggregate_near_inventory_valid
            and all(aggregate.get(key) is False for key in (
                "production_eligible", "fea_submission_approved",
                "fea_submission_performed",
            ))
            and _integer(aggregate.get("authenticated_terminal_count"), -1)
            == terminal_results_verified == len(terminal_results)
            and _integer(aggregate.get("source_candidate_count"), -1)
            == sum(len(items) for items in terminal_candidates.values())
            and _integer(aggregate.get("source_candidate_count"), -1)
            >= candidate_preview["total_count"]
            and candidate_preview["safe"]
            and declared_near_count == authenticated_near_count
            and declared_near_count >= len(aggregate_near)
            and len(aggregate_near) == expected_near_preview_count
            and _integer(aggregate.get("unique_candidate_count"), -1)
            >= candidate_preview["total_count"]
        )
        near_scores: list[float] = []
        near_candidate_rows: list[dict[str, Any]] = []
        seen_near_identities: set[tuple[int, int, str]] = set()
        for position, near_candidate in enumerate(aggregate_near):
            if not isinstance(near_candidate, dict):
                aggregate_safe = False
                warnings.append(
                    f"Tier-1 rolling near candidate {position} is invalid"
                )
                continue
            pair = (
                _integer(near_candidate.get("source_task_id"), -1),
                _integer(near_candidate.get("source_seed"), -1),
            )
            normalized_near = {
                key: value for key, value in near_candidate.items()
                if key not in {"source_task_id", "source_seed"}
            }
            normalized_sha = _canonical_json_sha256(normalized_near)
            source_match = any(
                _canonical_json_sha256(source_candidate) == normalized_sha
                for source_candidate in terminal_near_candidates.get(pair, [])
            )
            score = _finite_number(
                near_candidate.get("target_acquisition_score")
            )
            near_identity = (pair[0], pair[1], normalized_sha)
            candidate_safe = bool(
                pair in verified_terminal_pairs
                and source_match
                and near_identity not in seen_near_identities
                and score is not None
                and near_candidate.get("production_eligible") is False
                and near_candidate.get("fea_submission_approved") is False
                and near_candidate.get("eligible_for_submission") is False
            )
            if not candidate_safe:
                aggregate_safe = False
                warnings.append(
                    f"Tier-1 rolling near candidate {position} is unauthenticated"
                )
                continue
            seen_near_identities.add(near_identity)
            near_scores.append(score)
            near_candidate_rows.append({
                "seed": pair[1],
                "task_id": pair[0],
                "result_sha256": terminal_result_hashes[pair],
                "candidate": near_candidate,
            })
        if any(
            current < previous
            for previous, current in zip(near_scores, near_scores[1:])
        ):
            aggregate_safe = False
            warnings.append("Tier-1 rolling near candidate order is invalid")
        candidate_rows: list[dict[str, Any]] = []
        for position, candidate in enumerate(aggregate_candidates[:10_000]):
            if not isinstance(candidate, dict):
                aggregate_safe = False
                continue
            pair = (
                _integer(candidate.get("source_task_id"), -1),
                _integer(candidate.get("source_seed"), -1),
            )
            normalized_candidate = {
                key: value for key, value in candidate.items()
                if key not in {"source_task_id", "source_seed"}
            }
            source_match = any(
                _canonical_json_sha256(source_candidate)
                == _canonical_json_sha256(normalized_candidate)
                for source_candidate in terminal_candidates.get(pair, [])
            )
            candidate_safe = bool(
                pair in verified_terminal_pairs
                and source_match
                and candidate.get("production_eligible") is False
                and candidate.get("fea_submission_approved") is not True
            )
            if not candidate_safe:
                warnings.append(
                    f"Tier-1 rolling aggregate candidate {position} is unauthenticated"
                )
                aggregate_safe = False
                continue
            candidate_rows.append({
                "seed": pair[1],
                "task_id": pair[0],
                "result_sha256": terminal_result_hashes[pair],
                "candidate": candidate,
            })
        if len(candidate_rows) != len(aggregate_candidates):
            aggregate_safe = False
        if not aggregate_safe:
            warnings.append("Tier-1 rolling aggregate Pareto identity mismatch")
        terminal_ok = bool(terminal_ok and aggregate_safe)

        integrity_ok = bool(
            identity_ok and production_safe
            and (freshness_ok or historical_snapshot)
            and tasks_ok and exact_counts_ok and rolling_ok and terminal_ok
        )
        return {
            **base,
            "available": integrity_ok,
            "updated_at": updated_at,
            "age_seconds": age_seconds,
            "stale": not freshness_ok,
            "historical_snapshot": historical_snapshot,
            "cohort_id": _safe_text(status.get("cohort_id"), 200),
            "constraint_version": _safe_text(
                status.get("constraint_version"), 160
            ),
            "constraints": hard_spec,
            "temperature_constraint_contract": temperature_contract,
            "hard_spec_sha256": hard_spec_sha or None,
            "model_manifest_sha256": source_model_sha or None,
            "deployment_model_manifest_sha256": deployment_model_sha or None,
            "nsga_code_revision": code_revision or None,
            "search_count": scheduler_task_count,
            "running_count": state_counts.get("running", 0),
            "queued_count": state_counts.get("queued", 0),
            "attaching_count": state_counts.get("attaching", 0),
            "completed_count": terminal_results_verified,
            "failed_count": state_counts.get("failed", 0),
            "feasible_pareto_count": (
                candidate_preview["total_count"] if aggregate_safe else 0
            ),
            "candidate_preview_count": (
                candidate_preview["preview_count"] if aggregate_safe else 0
            ),
            "candidate_preview_limit": candidate_preview["preview_limit"],
            "candidate_preview_truncated": (
                candidate_preview["preview_truncated"]
                if aggregate_safe else None
            ),
            "candidate_preview_contract_explicit": (
                candidate_preview["explicit"] if aggregate_safe else False
            ),
            "terminal_results_verified": terminal_results_verified,
            "failed_terminal_results_verified": (
                failed_terminal_results_verified
            ),
            "terminal_result_contract_pending": False,
            "active_plus_queued": active_plus_queued,
            "near_feasible_count": max(
                0, _integer(aggregate.get("near_feasible_count"), 0)
            ),
            "near_feasible_preview_count": len(aggregate_near),
            "near_feasible_preview_limit": (
                ROLLING_NEAR_FEASIBLE_PREVIEW_LIMIT
            ),
            "near_feasible_preview_truncated": (
                declared_near_count > len(aggregate_near)
            ),
            "rolling": rolling,
            "state_counts": dict(sorted(state_counts.items())),
            "integrity_verified": integrity_ok,
            "lanes": lanes if integrity_ok and not historical_snapshot else [],
            "candidate_rows": candidate_rows if integrity_ok else [],
            "near_candidate_rows": (
                near_candidate_rows if integrity_ok else []
            ),
            "warnings": list(dict.fromkeys(warnings)),
        }

    @staticmethod
    def _tier1_generation_id(
        constraint_version: str, hard_spec_sha256: str
    ) -> str:
        identity = f"{constraint_version}\n{hard_spec_sha256}".encode("utf-8")
        return f"tier1-{hashlib.sha256(identity).hexdigest()[:20]}"

    @staticmethod
    def _tier1_generation_label(
        hard_spec: dict[str, Any],
        constraint_version: str,
        *,
        active: bool,
        read_only: bool = False,
    ) -> str:
        def compact_limit(value: Any) -> str:
            number = _finite_number(value)
            if number is None:
                return "—"
            return str(int(number)) if number.is_integer() else f"{number:g}"

        dimensions = "×".join(
            compact_limit(hard_spec.get(name))
            for name in ("size_W_max_mm", "size_L_max_mm", "size_H_max_mm")
        )
        resonance_minimum = _finite_number(hard_spec.get("resonance_min_Hz"))
        resonance_maximum = _finite_number(hard_spec.get("resonance_max_Hz"))
        if resonance_maximum is not None and resonance_minimum is not None:
            resonance_label = (
                f"공진 {resonance_minimum / 1000:g} kHz ≤ f < "
                f"{resonance_maximum / 1000:g} kHz"
            )
        elif resonance_maximum is not None:
            resonance_label = f"공진 < {resonance_maximum / 1000:g} kHz"
        elif resonance_minimum is not None:
            resonance_label = f"공진 ≥ {resonance_minimum / 1000:g} kHz"
        else:
            resonance_label = "공진 미계약"
        temperature = _finite_number(hard_spec.get("T_limit_C"))
        temperature_label = (
            f"{temperature:g}°C" if temperature is not None else "온도 미계약"
        )
        version_match = re.search(r"-(v\d+)$", constraint_version)
        version_label = version_match.group(1) if version_match else constraint_version
        prefix = "읽기전용" if read_only else ("현재" if active else "이전")
        return (
            f"{prefix} · {dimensions} mm · {resonance_label} · "
            f"{temperature_label} · {version_label}"
        )

    def _current7_condition_generation_records(
        self,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build selectable tabs from authenticated read-only indexes."""

        diagnostics = (
            diagnostics
            if diagnostics is not None
            else self._tier1_current7_condition_diagnostics()
        )
        grouped: dict[str, dict[str, Any]] = {}
        for item in diagnostics:
            if not isinstance(item, dict):
                continue
            version = _safe_text(item.get("constraint_version"), 200) or ""
            hard_spec = item.get("constraints")
            hard_spec = hard_spec if isinstance(hard_spec, dict) else {}
            hard_sha = str(item.get("hard_spec_sha256") or "").lower()
            verified = bool(
                item.get("integrity_verified") is True
                and item.get("constraint_contract_verified") is True
                and version
                and re.fullmatch(r"[0-9a-f]{64}", hard_sha)
                and _canonical_json_sha256(hard_spec) == hard_sha
                and isinstance(item.get("resonance_contract"), dict)
            )
            if not verified:
                continue
            source_healthy = bool(
                item.get("healthy") is True
                and _integer(item.get("refused_terminal_count"), 0) == 0
            )
            refused_terminal_count = max(
                0, _integer(item.get("refused_terminal_count"), 0)
            )
            generation_id = self._tier1_generation_id(version, hard_sha)
            warnings = [
                str(warning) for warning in item.get("warnings", [])
                if isinstance(warning, str) and warning.strip()
            ]
            record = {
                "id": generation_id,
                "label": self._tier1_generation_label(
                    hard_spec,
                    version,
                    active=False,
                    read_only=True,
                ),
                "active": False,
                "read_only": True,
                "authority_eligible": False,
                # A sealed refusal is an authenticated terminal outcome.  It
                # makes the search source unhealthy, but must not hide or
                # disable the read-only evidence tab.
                "selectable": True,
                "state": (
                    "read_only"
                    if source_healthy else "read_only_with_refusals"
                ),
                "source_healthy": source_healthy,
                "source_kind": "current7_condition_index",
                "condition_index_schema_version": item.get(
                    "condition_index_schema_version", CURRENT7_INDEX_SCHEMA
                ),
                "constraint_version": version,
                "hard_spec": dict(hard_spec),
                "hard_spec_sha256": hard_sha,
                "resonance_contract": dict(item["resonance_contract"]),
                "cohort_id": item.get("cohort_id"),
                "cohort_ids": [item.get("cohort_id")]
                if item.get("cohort_id") else [],
                "source_count": 1,
                "candidate_count": max(
                    0, _integer(item.get("feasible_pareto_count"), 0)
                ),
                "display_candidate_count": max(
                    0, _integer(item.get("candidate_preview_count"), 0)
                ),
                "near_feasible_count": max(
                    0, _integer(item.get("near_feasible_count"), 0)
                ),
                "search_count": max(0, _integer(item.get("search_count"), 0)),
                "completed_count": max(
                    0, _integer(item.get("completed_count"), 0)
                ),
                "authenticated_terminal_seed_count": max(
                    0,
                    _integer(
                        item.get("authenticated_terminal_seed_count"), 0
                    ),
                ),
                "refused_terminal_count": refused_terminal_count,
                "state_counts": dict(item.get("state_counts") or {}),
                "updated_at": _safe_text(item.get("updated_at"), 100),
                "metadata_verified": True,
                "detail_integrity_pending": bool(
                    item.get("condition_index_schema_version")
                    == CURRENT7_COMPACT_INDEX_SCHEMA
                    and item.get("detail_hydrated") is not True
                ),
                "gui_launch_eligible": False,
                "warning": " ".join(warnings) or None,
                "candidate_endpoint": (
                    f"/api/nsga2/generations/{generation_id}"
                ),
                "_diagnostics": item,
                "_path": Path(str(item.get("source"))).resolve(),
            }
            previous = grouped.get(generation_id)
            if previous is None or (
                str(record.get("updated_at") or ""),
                record["completed_count"],
            ) > (
                str(previous.get("updated_at") or ""),
                previous["completed_count"],
            ):
                grouped[generation_id] = record
        return sorted(
            grouped.values(),
            key=lambda record: str(record.get("updated_at") or ""),
            reverse=True,
        )

    def _tier1_generation_records(
        self,
        active: dict[str, Any],
        *,
        active_candidate_count: int = 0,
        active_display_count: int = 0,
    ) -> list[dict[str, Any]]:
        """Inventory condition generations without embedding their candidates.

        Only a configured, direct ``cohorts/<cohort>/status.json`` hierarchy is
        inspected.  This inexpensive inventory is returned on every dashboard
        refresh; full terminal/result authentication is deferred to the
        generation-detail endpoint selected by the operator.
        """
        root = self._tier1_nsga_cohort_root
        if (
            self._tier1_rolling_index_configured
            and active.get("pointer_verified") is not True
        ):
            # Without the configured canonical pointer, the inventory cannot
            # distinguish the live cohort from retired history.  Suppress the
            # optional history instead of letting a broken live cohort re-enter
            # through the archived freshness bypass.
            return []
        active_version = _safe_text(active.get("constraint_version"), 160) or ""
        active_hard_spec = (
            dict(active["constraints"])
            if isinstance(active.get("constraints"), dict) else {}
        )
        raw_active_hard_sha = active.get("hard_spec_sha256")
        active_hard_sha = (
            raw_active_hard_sha.strip().lower()
            if isinstance(raw_active_hard_sha, str) else ""
        )
        active_cohort = _safe_text(active.get("cohort_id"), 200) or ""
        active_key = (active_version, active_hard_sha)
        grouped: dict[tuple[str, str], dict[str, Any]] = {}

        if root is not None:
            try:
                resolved_root = root.resolve()
                if (
                    resolved_root != root
                    or root.is_symlink()
                    or not root.is_dir()
                ):
                    raise ValueError("configured cohort root is not a direct directory")
                children = sorted(
                    (item for item in root.iterdir() if item.is_dir()),
                    key=lambda item: item.name,
                )
                if len(children) > 64:
                    raise ValueError("configured cohort root exceeds 64 generations")
                for child in children:
                    raw_status_path = child / "status.json"
                    status_path = raw_status_path.resolve()
                    if (
                        not status_path.is_relative_to(root)
                        or child.is_symlink()
                        or raw_status_path.is_symlink()
                        or not status_path.is_file()
                    ):
                        continue
                    result = self.cache.json(
                        status_path,
                        {},
                        max_bytes=16 * 1024 * 1024,
                        fail_closed=True,
                    )
                    status = result.value if isinstance(result.value, dict) else {}
                    hard_spec = status.get("hard_spec")
                    hard_spec = hard_spec if isinstance(hard_spec, dict) else {}
                    raw_hard_sha = status.get("hard_spec_sha256")
                    hard_sha = (
                        raw_hard_sha.strip().lower()
                        if isinstance(raw_hard_sha, str) else ""
                    )
                    version = _safe_text(status.get("constraint_version"), 160) or ""
                    raw_bundle_sha = status.get("bundle_manifest_sha256")
                    bundle_sha = (
                        raw_bundle_sha.strip().lower()
                        if isinstance(raw_bundle_sha, str) else ""
                    )
                    metadata_verified = bool(
                        not result.warning
                        and status.get("schema_version")
                        == "mft-tier1-slurm-rolling-status-v1"
                        and status.get("cohort_id") == child.name
                        and version
                        and re.fullmatch(r"[0-9a-f]{64}", hard_sha)
                        and _canonical_json_sha256(hard_spec) == hard_sha
                        and re.fullmatch(r"[0-9a-f]{64}", bundle_sha)
                        and all(status.get(key) is False for key in (
                            "production_eligible",
                            "fea_submission_approved",
                            "fea_submission_performed",
                            "aedt_used",
                        ))
                    )
                    if not metadata_verified:
                        continue
                    aggregate = status.get("aggregate_pareto")
                    aggregate = aggregate if isinstance(aggregate, dict) else {}
                    raw_candidates = aggregate.get("candidates")
                    raw_candidates = (
                        raw_candidates if isinstance(raw_candidates, list) else []
                    )
                    pareto_count = max(
                        len(raw_candidates),
                        max(0, _integer(aggregate.get("pareto_count"), 0)),
                        max(0, _integer(aggregate.get("candidate_preview_total"), 0)),
                    )
                    near_feasible_count = max(
                        0, _integer(aggregate.get("near_feasible_count"), 0)
                    )
                    terminal_results = status.get("terminal_results")
                    terminal_results = (
                        terminal_results if isinstance(terminal_results, list) else []
                    )
                    state_counts = status.get("state_counts")
                    state_counts = state_counts if isinstance(state_counts, dict) else {}
                    is_active = bool(
                        active.get("integrity_verified") is True
                        and (version, hard_sha) == active_key
                        and status.get("cohort_id") == active_cohort
                    )
                    if child.name == active_cohort and not is_active:
                        # A canonical cohort whose live heartbeat/integrity gate
                        # failed is not historical evidence.  It stays hidden
                        # until the canonical path verifies again.
                        continue
                    warning = None
                    if not is_active and "T_limit_C" not in hard_spec:
                        warning = (
                            "온도 제약 계약 전에 중단된 세대이며 유효 Pareto가 없습니다."
                        )
                    elif not is_active and status.get("healthy") is not True:
                        warning = "완료되지 않은 이전 세대이며 유효 Pareto가 없습니다."
                    elif not is_active and pareto_count == 0:
                        warning = "이 조건 세대에서는 robust-feasible Pareto가 없었습니다."
                    record = {
                        "id": self._tier1_generation_id(version, hard_sha),
                        "label": self._tier1_generation_label(
                            hard_spec, version, active=is_active
                        ),
                        "active": is_active,
                        "selectable": bool(
                            is_active or status.get("healthy") is True
                        ),
                        "state": "active" if is_active else (
                            "archived" if status.get("healthy") is True else "incomplete"
                        ),
                        "constraint_version": version,
                        "hard_spec": dict(hard_spec),
                        "hard_spec_sha256": hard_sha,
                        "cohort_id": child.name,
                        "cohort_ids": [child.name],
                        "source_count": 1,
                        "candidate_count": (
                            max(0, active_candidate_count) if is_active else pareto_count
                        ),
                        "display_candidate_count": (
                            max(0, active_display_count)
                            if is_active else len(raw_candidates)
                        ),
                        "near_feasible_count": (
                            max(
                                0,
                                _integer(
                                    active.get("near_feasible_count"), 0
                                ),
                            )
                            if is_active else near_feasible_count
                        ),
                        "search_count": max(
                            0, _integer(status.get("scheduler_task_count"), 0)
                        ),
                        "completed_count": len(terminal_results),
                        "state_counts": {
                            str(key): max(0, _integer(value, 0))
                            for key, value in state_counts.items()
                        },
                        "updated_at": (
                            _safe_text(status.get("updated_at"), 100)
                            if is_active or status.get("healthy") is not True
                            else _tier1_archived_detail_updated_at(
                                status.get("updated_at")
                            )
                        ),
                        "metadata_verified": True,
                        "detail_integrity_pending": not is_active,
                        "gui_launch_eligible": is_active,
                        "warning": warning,
                        "candidate_endpoint": None,
                        "_path": status_path,
                        "_status": status,
                        "_root": child.resolve(),
                    }
                    key = (version, hard_sha)
                    previous = grouped.get(key)
                    score = (
                        1 if is_active else 0,
                        1 if status.get("healthy") is True else 0,
                        len(terminal_results),
                        pareto_count,
                        str(status.get("updated_at") or ""),
                    )
                    previous_score = previous.get("_score") if previous else None
                    if previous is None or score > previous_score:
                        record["_score"] = score
                        grouped[key] = record
            except (OSError, TypeError, ValueError):
                # The current pointer remains authoritative even if optional
                # historical discovery is unavailable.
                pass

        if (
            active.get("integrity_verified") is True
            and active_version
            and re.fullmatch(r"[0-9a-f]{64}", active_hard_sha)
            and active_hard_spec
            and _canonical_json_sha256(active_hard_spec) == active_hard_sha
        ):
            current = grouped.get(active_key)
            if current is None or current.get("active") is not True:
                current = {
                    "id": self._tier1_generation_id(
                        active_version, active_hard_sha
                    ),
                    "label": self._tier1_generation_label(
                        active_hard_spec, active_version, active=True
                    ),
                    "active": True,
                    "selectable": True,
                    "state": "active",
                    "constraint_version": active_version,
                    "hard_spec": active_hard_spec,
                    "hard_spec_sha256": active_hard_sha,
                    "cohort_id": active_cohort or None,
                    "cohort_ids": list(active.get("cohort_ids") or (
                        [active_cohort] if active_cohort else []
                    )),
                    "source_count": max(
                        1, _integer(active.get("source_count"), 1)
                    ),
                    "candidate_count": max(0, active_candidate_count),
                    "display_candidate_count": max(0, active_display_count),
                    "near_feasible_count": max(
                        0, _integer(active.get("near_feasible_count"), 0)
                    ),
                    "search_count": max(0, _integer(active.get("search_count"), 0)),
                    "completed_count": max(
                        0, _integer(active.get("completed_count"), 0)
                    ),
                    "state_counts": dict(active.get("state_counts") or {}),
                    "updated_at": _safe_text(active.get("updated_at"), 100),
                    "metadata_verified": True,
                    "detail_integrity_pending": False,
                    "gui_launch_eligible": (
                        active.get("gui_launch_eligible") is not False
                    ),
                    "warning": None,
                    "sources": [
                        dict(item) for item in active.get("sources", [])
                        if isinstance(item, dict)
                        and item.get("accepted") is True
                    ],
                    "candidate_endpoint": None,
                    "_path": None,
                    "_status": None,
                    "_root": root,
                    "_score": (2, 1, 0, 0, ""),
                }
                grouped[active_key] = current
            else:
                current.update({
                    "label": self._tier1_generation_label(
                        active_hard_spec, active_version, active=True
                    ),
                    "active": True,
                    "state": "active",
                    "candidate_count": max(0, active_candidate_count),
                    "display_candidate_count": max(0, active_display_count),
                    "near_feasible_count": max(
                        0, _integer(active.get("near_feasible_count"), 0)
                    ),
                    "search_count": max(
                        0, _integer(active.get("search_count"), 0)
                    ),
                    "completed_count": max(
                        0, _integer(active.get("completed_count"), 0)
                    ),
                    "state_counts": dict(active.get("state_counts") or {}),
                    "updated_at": _safe_text(active.get("updated_at"), 100),
                    "detail_integrity_pending": False,
                    "gui_launch_eligible": (
                        active.get("gui_launch_eligible") is not False
                    ),
                    "warning": None,
                    "cohort_ids": list(active.get("cohort_ids") or (
                        [active_cohort] if active_cohort else []
                    )),
                    "source_count": max(
                        1, _integer(active.get("source_count"), 1)
                    ),
                    "sources": [
                        dict(item) for item in active.get("sources", [])
                        if isinstance(item, dict)
                        and item.get("accepted") is True
                    ],
                })

        records = sorted(
            grouped.values(),
            key=lambda item: str(item.get("updated_at") or ""),
            reverse=True,
        )
        records.sort(key=lambda item: item.get("active") is not True)
        for record in records:
            record["candidate_endpoint"] = (
                f"/api/nsga2/generations/{record['id']}"
            )
        return records

    @staticmethod
    def _public_tier1_generation(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in record.items()
            if not key.startswith("_")
        }

    def _historical_tier1_diagnostics(
        self, record: dict[str, Any]
    ) -> dict[str, Any]:
        path = record.get("_path")
        status = record.get("_status")
        root = record.get("_root")
        if not isinstance(path, Path) or not isinstance(root, Path):
            return {
                "available": False,
                "integrity_verified": False,
                "warnings": ["이전 조건 세대의 인증된 status가 없습니다."],
                "candidate_rows": [],
            }
        status = status if isinstance(status, dict) else {}
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
        except OSError as exc:
            return {
                "available": False,
                "integrity_verified": False,
                "warnings": [f"이전 조건 세대 status를 읽을 수 없습니다: {exc}"],
                "candidate_rows": [],
            }
        with self._tier1_history_lock:
            cached = self._tier1_history_diagnostics.get(key)
            if cached is not None:
                return cached
            synthetic_index = {
                "active_cohort_id": status.get("cohort_id"),
                "constraint_version": status.get("constraint_version"),
                "hard_spec": status.get("hard_spec"),
                "hard_spec_sha256": status.get("hard_spec_sha256"),
                "source_model_manifest_sha256": status.get(
                    "source_model_manifest_sha256"
                ),
                "deployment_model_manifest_sha256": status.get(
                    "deployment_model_manifest_sha256"
                ),
                "temperature_constraint_contract": status.get(
                    "temperature_constraint_contract"
                ),
                "path_containment_root": str(root),
            }
            result = self._tier1_rolling_nsga_diagnostics(
                path=path,
                status=status,
                rolling_index=synthetic_index,
                base={
                    "configured": True,
                    "source": str(path),
                    "pointer_verified": True,
                },
                warnings=[],
                historical_snapshot=True,
            )
            if len(self._tier1_history_diagnostics) >= 16:
                self._tier1_history_diagnostics.clear()
            self._tier1_history_diagnostics[key] = result
            return result

    @staticmethod
    def _tier1_near_feasible_preview(
        diagnostics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Project authenticated acquisition candidates as compact diagnostics.

        Near-feasible rows are deliberately kept separate from Pareto candidates:
        they have failed at least one hard constraint and never carry GUI,
        production, or FEA submission authority.  Only bounded rows already
        authenticated against terminal result hashes reach this projection.
        """

        if diagnostics.get("integrity_verified") is not True:
            return []
        hard_spec = diagnostics.get("constraints")
        hard_spec = hard_spec if isinstance(hard_spec, dict) else {}

        def hard_number(name: str) -> float | None:
            return _finite_number(hard_spec.get(name))

        def upper_bound_value(
            constraints: dict[str, Any], limit_name: str, constraint_name: str,
        ) -> float | None:
            limit = hard_number(limit_name)
            violation = _finite_number(constraints.get(constraint_name))
            return (
                limit + violation
                if limit is not None and violation is not None else None
            )

        preview: list[dict[str, Any]] = []
        for position, record in enumerate(
            (diagnostics.get("near_candidate_rows") or [])[
                :ROLLING_NEAR_FEASIBLE_PREVIEW_LIMIT
            ]
        ):
            if not isinstance(record, dict):
                continue
            raw = record.get("candidate")
            if not isinstance(raw, dict) or not all(
                raw.get(name) is False for name in (
                    "production_eligible",
                    "fea_submission_approved",
                    "eligible_for_submission",
                )
            ):
                continue
            score = _finite_number(raw.get("target_acquisition_score"))
            seed = _integer(record.get("seed"), -1)
            task_id = _integer(record.get("task_id"), -1)
            if score is None or seed < 0 or task_id <= 0:
                continue

            decoded = raw.get("decoded_params")
            decoded = decoded if isinstance(decoded, dict) else {}
            predictions = raw.get("target_predictions")
            if not isinstance(predictions, dict):
                predictions = raw.get("predictions")
            predictions = predictions if isinstance(predictions, dict) else {}
            half_width = raw.get("target_conformal_half_width")
            if not isinstance(half_width, dict):
                half_width = raw.get("conformal_half_width")
            half_width = half_width if isinstance(half_width, dict) else {}
            derived = raw.get("derived_resonance")
            derived = derived if isinstance(derived, dict) else {}
            constraint_g = raw.get("constraint_G")
            constraint_g = constraint_g if isinstance(constraint_g, dict) else {}

            replay_box = _candidate_bounding_box(decoded)
            width = upper_bound_value(
                constraint_g, "size_W_max_mm", "exterior_width_limit"
            )
            length = upper_bound_value(
                constraint_g, "size_L_max_mm", "exterior_length_limit"
            )
            height = upper_bound_value(
                constraint_g, "size_H_max_mm", "exterior_height_limit"
            )
            if width is None:
                width = _finite_number(replay_box.get("size_W_mm"))
            if length is None:
                length = _finite_number(replay_box.get("size_L_mm"))
            if height is None:
                height = _finite_number(replay_box.get("size_H_mm"))
            volume = (
                width * length * height / 1_000_000.0
                if (
                    width is not None and width > 0.0
                    and length is not None and length > 0.0
                    and height is not None and height > 0.0
                ) else _finite_number(replay_box.get("volume_L"))
            )

            llt_mu = _finite_number(predictions.get("Llt_phys"))
            llt_half = _finite_number(half_width.get("Llt_phys"))
            resonance = _finite_number(derived.get("f_res_min_screen_Hz"))
            if resonance is None:
                resonance = _finite_number(
                    derived.get("pred_f_res_min_screen_Hz")
                )
            temperature_violations = [
                value
                for key, raw_value in constraint_g.items()
                if key.startswith("temperature_robust_limit:")
                and (value := _finite_number(raw_value)) is not None
            ]
            temperature_limit = hard_number("T_limit_C")
            worst_temperature = (
                temperature_limit + max(temperature_violations)
                if temperature_limit is not None and temperature_violations
                else None
            )
            b_value = upper_bound_value(
                constraint_g,
                "B_limit_T",
                "analytical_flux_density_limit",
            )

            violations = [
                {"constraint": str(name), "amount": value}
                for name, raw_value in constraint_g.items()
                if (value := _finite_number(raw_value)) is not None
                and value > 1e-9
            ]
            violations.sort(
                key=lambda item: (-item["amount"], item["constraint"])
            )
            source_role = _safe_text(record.get("source_role"), 40) or "primary"
            source_label = (
                _safe_text(record.get("source_label"), 100) or source_role
            )
            source_slug = re.sub(
                r"[^A-Za-z0-9_-]+", "-", source_label
            ).strip("-")[:40] or source_role
            population_index = _integer(
                raw.get("terminal_population_index"), position
            )
            decoded_sha = (
                _safe_text(raw.get("decoded_params_sha256"), 64) or ""
            ).lower()
            preview.append({
                "id": (
                    f"near-{source_slug}-task-{task_id}-seed-{seed}-"
                    f"{population_index:04d}"
                ),
                "source_role": source_role,
                "source_label": source_label,
                "source_cohort_id": _safe_text(
                    record.get("source_cohort_id"), 200
                ),
                "task_id": task_id,
                "seed": seed,
                "terminal_population_index": population_index,
                "result_sha256": _safe_text(
                    record.get("result_sha256"), 64
                ),
                "decoded_params_sha256": (
                    decoded_sha
                    if re.fullmatch(r"[0-9a-f]{64}", decoded_sha) else None
                ),
                "target_acquisition_score": score,
                "size_W_mm": width,
                "size_L_mm": length,
                "size_H_mm": height,
                "volume_L": volume,
                "pred_Llt_phys_uH": llt_mu,
                "Llt_q90_half_width_uH": llt_half,
                "Llt_robust_lower_uH": (
                    llt_mu - llt_half
                    if llt_mu is not None and llt_half is not None else None
                ),
                "Llt_robust_upper_uH": (
                    llt_mu + llt_half
                    if llt_mu is not None and llt_half is not None else None
                ),
                "Llt_robust_G_uH": _finite_number(
                    constraint_g.get("Llt_robust_band")
                ),
                "f_res_min_screen_Hz": resonance,
                "resonance_G_Hz": _finite_number(
                    constraint_g.get("half_magnetizing_resonance_minimum")
                ),
                "robust_max_temperature_C": worst_temperature,
                "temperature_G_C": (
                    max(temperature_violations)
                    if temperature_violations else None
                ),
                "B_design_analytic_T": b_value,
                "n_core_group": _finite_number(decoded.get("n_core_group")),
                "cw1_mm": _finite_number(decoded.get("cw1")),
                "violation_count": len(violations),
                "violations": violations,
                "spec_status": "near_feasible",
                "valid_pareto": False,
                "production_eligible": False,
                "fea_submission_approved": False,
                "gui_launch_eligible": False,
            })
        return preview

    def _current7_condition_generation_payload(
        self,
        diagnostics: dict[str, Any],
        generation: dict[str, Any],
    ) -> dict[str, Any]:
        """Render one authenticated condition index without granting authority."""

        verified = bool(
            diagnostics.get("integrity_verified") is True
            and diagnostics.get("constraint_contract_verified") is True
            and diagnostics.get("display_only") is True
            and diagnostics.get("authority_eligible") is False
        )
        hard_spec = diagnostics.get("constraints")
        hard_spec = hard_spec if isinstance(hard_spec, dict) else {}
        constraint_version = _safe_text(
            diagnostics.get("constraint_version"), 200
        )
        candidates: list[dict[str, Any]] = []
        if verified:
            for position, record in enumerate(
                diagnostics.get("candidate_rows", [])
            ):
                if not isinstance(record, dict):
                    continue
                seed = _integer(record.get("seed"), -1)
                raw_candidate = record.get("candidate")
                if seed < 0 or not isinstance(raw_candidate, dict):
                    continue
                decoded = raw_candidate.get("decoded_params")
                decoded = decoded if isinstance(decoded, dict) else {}
                predictions = raw_candidate.get("surrogate_mean_predictions")
                predictions = predictions if isinstance(predictions, dict) else {}
                row = {
                    key: value for key, value in raw_candidate.items()
                    if not isinstance(value, (dict, list))
                }
                row.update(decoded)
                row.update({
                    f"pred_{name}": value
                    for name, value in predictions.items()
                })
                loss_aliases = {
                    "pred_total_loss_W": "predicted_total_loss_W",
                    "pred_total_winding_loss_W": "predicted_winding_loss_W",
                    "pred_component_winding_loss_sum_W": (
                        "predicted_winding_loss_W"
                    ),
                    "pred_core_loss_W": "predicted_core_loss_W",
                    "pred_core_cold_plate_loss_W": (
                        "predicted_core_plate_loss_W"
                    ),
                    "pred_winding_cold_plate_loss_W": (
                        "predicted_winding_cold_plate_loss_W"
                    ),
                    "pred_primary_winding_loss_W": (
                        "predicted_Tx_main_winding_loss_W"
                    ),
                    "pred_secondary_center_winding_loss_W": (
                        "predicted_Rx_main_winding_loss_W"
                    ),
                    "pred_secondary_side_winding_loss_W": (
                        "predicted_Rx_side_winding_loss_W"
                    ),
                }
                for alias, source_name in loss_aliases.items():
                    if source_name in raw_candidate:
                        row[alias] = raw_candidate[source_name]
                candidate = self._authenticated_tier1_candidate(
                    row,
                    0,
                    position,
                    source_candidate=raw_candidate,
                    hard_spec=hard_spec,
                    constraint_version=constraint_version,
                    temperature_targets=tuple(
                        diagnostics.get("temperature_targets")
                        or CURRENT7_TEMPERATURE_TARGETS
                    ),
                )
                source_index = _integer(
                    raw_candidate.get("terminal_population_index"), position
                )
                candidate.update({
                    "id": (
                        f"{generation.get('id')}-seed-{seed}-"
                        f"{source_index:04d}"
                    ),
                    "round": None,
                    "result_lane": f"tier1-current7-condition-seed-{seed}",
                    "run_id": f"seed-{seed}",
                    "model_id": (
                        f"tier1-current7-condition:"
                        f"{diagnostics.get('bundle_id')}"
                    ),
                    "model_lane": "corrected_current7_condition",
                    "result_scope": "read_only_condition_generation",
                    "production_eligible": False,
                    "fea_submission_approved": False,
                    "fea_verified": False,
                    "artifact_hydrated": True,
                    "artifact_source": diagnostics.get("source"),
                    "embedded_authenticated_result": True,
                    "tier1_source_role": "current7_condition",
                    "tier1_source_label": "read-only-current7-condition",
                    "tier1_source_cohort_id": diagnostics.get("cohort_id"),
                    "gui_launch_eligible": False,
                    "generation_id": generation.get("id"),
                })
                candidates.append(candidate)
        valid = [
            candidate for candidate in candidates
            if candidate.get("spec_status") == "pass"
        ]
        minimum_volume = min(
            (
                candidate for candidate in valid
                if candidate.get("volume_L") is not None
            ),
            key=lambda candidate: candidate["volume_L"],
            default=None,
        )
        minimum_loss = min(
            (
                candidate for candidate in valid
                if candidate.get("total_loss_W") is not None
            ),
            key=lambda candidate: candidate["total_loss_W"],
            default=None,
        )
        for candidate in candidates:
            candidate["is_min_volume"] = bool(
                minimum_volume and candidate["id"] == minimum_volume["id"]
            )
            candidate["is_min_loss"] = bool(
                minimum_loss and candidate["id"] == minimum_loss["id"]
            )
        candidates.sort(key=lambda candidate: (
            candidate.get("volume_L") is None,
            candidate.get("volume_L") or 0.0,
        ))
        summary = {
            "candidate_count": len(candidates),
            "display_candidate_count": len(candidates),
            "valid_candidate_count": len(valid),
            "historical_diagnostic_count": 0,
            "min_volume_L": (
                minimum_volume["volume_L"] if minimum_volume else None
            ),
            "min_loss_W": (
                minimum_loss["total_loss_W"] if minimum_loss else None
            ),
            "min_volume_candidate_id": (
                minimum_volume["id"] if minimum_volume else None
            ),
            "min_loss_candidate_id": (
                minimum_loss["id"] if minimum_loss else None
            ),
            "known_spec_pass_count": len(valid),
            "known_spec_fail_count": sum(
                candidate.get("spec_status") == "fail"
                for candidate in candidates
            ),
            "unknown_spec_count": sum(
                candidate.get("spec_status") == "unknown"
                for candidate in candidates
            ),
            "excluded_other_model_or_history_count": 0,
        }
        warnings = list(dict.fromkeys(
            str(warning) for warning in diagnostics.get("warnings", [])
            if isinstance(warning, str) and warning.strip()
        ))
        public_generation = self._public_tier1_generation(generation)
        public_generation.update({
            "detail_integrity_pending": not verified,
            "detail_integrity_verified": verified,
        })
        near_feasible_preview = (
            self._tier1_current7_near_feasible_preview(diagnostics)
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "available": verified,
            "integrity_verified": verified,
            "status": "completed" if verified else "unavailable",
            "source_kind": "current7_condition_index",
            "result_scope": "read_only_condition_generation",
            "round": None,
            "al_round": None,
            "al_stage": "READ-ONLY CONDITION",
            "candidate_count": len(candidates),
            "display_candidate_count": len(candidates),
            "valid_candidate_count": len(valid),
            "candidates": candidates,
            "near_feasible_preview": near_feasible_preview,
            "near_feasible_preview_count": len(near_feasible_preview),
            "summary": summary,
            "comparison": None,
            "constraint_version": constraint_version,
            "constraints": hard_spec,
            "resonance_contract": diagnostics.get("resonance_contract"),
            "selected_generation_id": generation.get("id"),
            "generation": public_generation,
            "tier1_current7_search": {
                key: value for key, value in diagnostics.items()
                if key not in {"candidate_rows", "near_candidate_rows", "lanes"}
            },
            "source": diagnostics.get("source"),
            "updated_at": diagnostics.get("updated_at"),
            "note": (
                "인증된 Current7 조건 인덱스의 읽기 전용 Pareto입니다. "
                "이 결과에는 production, FEA, AEDT, GUI 실행 권한이 없습니다."
            ),
            "warning": " ".join(warnings) or generation.get("warning"),
            "warnings": warnings,
        }

    def _tier1_generation_payload(
        self,
        diagnostics: dict[str, Any],
        generation: dict[str, Any],
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        near_feasible_preview = self._tier1_near_feasible_preview(
            diagnostics
        )
        hard_spec = diagnostics.get("constraints")
        hard_spec = hard_spec if isinstance(hard_spec, dict) else {}
        constraint_version = _safe_text(
            diagnostics.get("constraint_version"), 160
        )
        model_sha = _safe_text(
            diagnostics.get("model_manifest_sha256"), 64
        ) or ""
        if diagnostics.get("integrity_verified") is True:
            for position, record in enumerate(diagnostics.get("candidate_rows", [])):
                if not isinstance(record, dict):
                    continue
                seed = _integer(record.get("seed"), -1)
                task_id = _integer(record.get("task_id"), -1)
                raw_candidate = record.get("candidate")
                if seed < 0 or not isinstance(raw_candidate, dict):
                    continue
                decoded = raw_candidate.get("decoded_params")
                decoded = decoded if isinstance(decoded, dict) else {}
                predictions = raw_candidate.get("predictions")
                predictions = predictions if isinstance(predictions, dict) else {}
                derived = raw_candidate.get("derived_resonance")
                derived = derived if isinstance(derived, dict) else {}
                row = dict(decoded)
                row.update({f"pred_{name}": value for name, value in predictions.items()})
                row.update(derived)
                row["volume_L"] = raw_candidate.get("volume_L")
                row["total_loss_W"] = raw_candidate.get("total_loss_W")
                source_index = _integer(raw_candidate.get("index"), position)
                candidate = self._authenticated_tier1_candidate(
                    row,
                    0,
                    position,
                    source_candidate=raw_candidate,
                    hard_spec=hard_spec,
                    constraint_version=constraint_version,
                )
                candidate.update({
                    "id": (
                        f"{generation.get('id')}-seed-{seed}-"
                        f"{source_index:04d}"
                    ),
                    "round": None,
                    "result_lane": f"tier1-feedback-seed-{seed}",
                    "run_id": f"seed-{seed}",
                    "model_id": f"tier1-feedback:{model_sha}",
                    "model_lane": "experimental_tier1_feedback",
                    "result_scope": "archived_condition_generation",
                    "production_eligible": False,
                    "fea_submission_approved": False,
                    "fea_verified": False,
                    "tier1_task_id": task_id if task_id > 0 else None,
                    "artifact_hydrated": True,
                    "artifact_source": diagnostics.get("source"),
                    "embedded_authenticated_result": True,
                    "gui_launch_eligible": False,
                    "generation_id": generation.get("id"),
                })
                candidates.append(candidate)
        valid = [item for item in candidates if item.get("spec_status") == "pass"]
        volume_candidates = [
            item for item in valid if item.get("volume_L") is not None
        ]
        loss_candidates = [
            item for item in valid if item.get("total_loss_W") is not None
        ]
        minimum_volume = (
            min(volume_candidates, key=lambda item: item["volume_L"])
            if volume_candidates else None
        )
        minimum_loss = (
            min(loss_candidates, key=lambda item: item["total_loss_W"])
            if loss_candidates else None
        )
        for candidate in candidates:
            candidate["is_min_volume"] = bool(
                minimum_volume and candidate["id"] == minimum_volume["id"]
            )
            candidate["is_min_loss"] = bool(
                minimum_loss and candidate["id"] == minimum_loss["id"]
            )
        candidates.sort(
            key=lambda item: (
                item.get("volume_L") is None,
                item.get("volume_L") or 0.0,
            )
        )
        summary = {
            "candidate_count": len(candidates),
            "display_candidate_count": len(candidates),
            "valid_candidate_count": len(valid),
            "historical_diagnostic_count": 0,
            "min_volume_L": (
                minimum_volume["volume_L"] if minimum_volume else None
            ),
            "min_loss_W": minimum_loss["total_loss_W"] if minimum_loss else None,
            "min_volume_candidate_id": (
                minimum_volume["id"] if minimum_volume else None
            ),
            "min_loss_candidate_id": minimum_loss["id"] if minimum_loss else None,
            "known_spec_pass_count": len(valid),
            "known_spec_fail_count": sum(
                item.get("spec_status") == "fail" for item in candidates
            ),
            "unknown_spec_count": sum(
                item.get("spec_status") == "unknown" for item in candidates
            ),
            "excluded_other_model_or_history_count": 0,
        }
        verified = diagnostics.get("integrity_verified") is True
        warning = " ".join(diagnostics.get("warnings", []))
        public_generation = self._public_tier1_generation(generation)
        public_generation.update({
            "detail_integrity_pending": not verified,
            "detail_integrity_verified": verified,
        })
        return {
            "schema_version": SCHEMA_VERSION,
            "available": verified,
            "integrity_verified": verified,
            "status": "completed" if verified else "unavailable",
            "source_kind": "archived_tier1_condition_generation",
            "result_scope": "archived_condition_generation",
            "round": None,
            "al_round": None,
            "al_stage": "CONDITION HISTORY",
            "candidate_count": len(candidates),
            "display_candidate_count": len(candidates),
            "valid_candidate_count": len(valid),
            "candidates": candidates,
            "near_feasible_preview": near_feasible_preview,
            "near_feasible_preview_count": len(near_feasible_preview),
            "summary": summary,
            "comparison": None,
            "constraint_version": constraint_version,
            "constraints": hard_spec,
            "selected_generation_id": generation.get("id"),
            "generation": public_generation,
            "tier1_feedback_search": {
                key: value for key, value in diagnostics.items()
                if key not in {
                    "candidate_rows", "near_candidate_rows", "lanes",
                }
            },
            "source": diagnostics.get("source") or generation.get("candidate_endpoint"),
            "updated_at": diagnostics.get("updated_at") or generation.get("updated_at"),
            "note": (
                generation.get("warning") or (
                    "인증된 이전 조건 세대의 Pareto 결과입니다. "
                    "과거 후보의 GUI/FEA 실행은 차단됩니다."
                )
                if verified else
                "이전 조건 세대의 terminal/result/Pareto 무결성 인증에 "
                "실패하여 결과 표시를 차단했습니다."
            ),
            "warnings": list(dict.fromkeys(
                diagnostics.get("warnings", [])
            )),
            "warning": warning or generation.get("warning"),
        }

    def _tier1_feedback_nsga_diagnostics(self) -> dict[str, Any]:
        """Read the authenticated Tier-1 feedback search without promoting it.

        A completed search may contribute display candidates only after its
        result hash, seed, model identity, code revision, and non-production
        flags agree with the monitored status.  Target-acquisition plans are
        diagnostics, not Pareto designs and are never projected as valid.
        """
        path = self._tier1_nsga_status
        pointer_path = self._tier1_nsga_pointer
        pointer_warning: str | None = None
        pointer_verified = False
        pointer_constraint_version: str | None = None
        pointer_constraints: dict[str, Any] = {}
        rolling_index: dict[str, Any] | None = None
        coherent_snapshot: dict[str, Any] | None = None
        if pointer_path is not None and pointer_path.exists():
            try:
                if self._tier1_rolling_index_configured:
                    coherent_snapshot = self._coherent_rolling_snapshot(
                        pointer_path
                    )
                    pointer = coherent_snapshot["index"]
                else:
                    pointer_result = self.cache.json(
                        pointer_path, {}, max_bytes=ROLLING_INDEX_MAX_BYTES,
                        fail_closed=True,
                    )
                    pointer = (
                        pointer_result.value
                        if isinstance(pointer_result.value, dict) else {}
                    )
                    if pointer_result.warning:
                        raise ValueError(pointer_result.warning)
                pointer_schema = pointer.get("schema_version")
                if pointer_schema == "mft-tier1-slurm-rolling-index-v1":
                    if coherent_snapshot is None:
                        coherent_snapshot = self._coherent_rolling_snapshot(
                            pointer_path
                        )
                        pointer = coherent_snapshot["index"]
                    candidate = coherent_snapshot["status_path"]
                    model_pointer = coherent_snapshot["model_pointer"]
                    hard_spec = pointer.get("hard_spec")
                    hard_spec = hard_spec if isinstance(hard_spec, dict) else {}
                    if not self._tier1_rolling_model_identity_matches(
                        pointer, model_pointer
                    ):
                        raise ValueError(
                            "Tier-1 rolling model/constraint identity mismatch"
                        )
                    if not all(
                        pointer.get(key) is False for key in (
                            "production_eligible", "fea_submission_approved",
                            "fea_submission_performed", "aedt_used",
                        )
                    ):
                        raise ValueError(
                            "Tier-1 rolling index attempted production/FEA authority"
                        )
                    pointer_constraint_version = _safe_text(
                        pointer.get("constraint_version"), 160
                    )
                    pointer_constraints = hard_spec
                    path = candidate
                    pointer_verified = True
                    rolling_index = pointer
                elif pointer_schema not in {
                    "mft-tier1-nsga-pointer-v1",
                    "mft-nsga-status-pointer-v1",
                }:
                    raise ValueError("unsupported Tier-1 NSGA pointer schema")
                else:
                    relative = _safe_text(pointer.get("status"), 4_096)
                    expected_sha = str(pointer.get("status_sha256") or "").lower()
                    if not relative or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                        raise ValueError("Tier-1 NSGA pointer identity is malformed")
                    pointer_root = pointer_path.parent.resolve()
                    candidate = (pointer_root / relative).resolve()
                    if not candidate.is_relative_to(pointer_root):
                        raise ValueError("Tier-1 NSGA status escapes pointer root")
                    if candidate.is_symlink() or not candidate.is_file():
                        raise ValueError("Tier-1 NSGA status is not a regular file")
                    if _sha256_file(candidate) != expected_sha:
                        raise ValueError("Tier-1 NSGA pointer/status hash mismatch")
                    pointer_constraint_version = _safe_text(
                        pointer.get("constraint_version"), 160
                    )
                    raw_pointer_constraints = pointer.get("constraints")
                    pointer_constraints = (
                        raw_pointer_constraints
                        if isinstance(raw_pointer_constraints, dict) else {}
                    )
                    path = candidate
                    pointer_verified = True
            except (OSError, TypeError, ValueError) as exc:
                pointer_warning = str(exc)
                path = None
        elif pointer_path is not None and self._tier1_nsga_status is None:
            pointer_warning = "stable Tier-1 NSGA pointer is unavailable"
        if path is None:
            return {
                "available": False,
                "configured": bool(pointer_path or self._tier1_nsga_status),
                "pointer_verified": False,
                "pointer": str(pointer_path) if pointer_path else None,
                "warnings": [pointer_warning] if pointer_warning else [],
                "lanes": [],
            }
        if coherent_snapshot is not None:
            status = coherent_snapshot["status"]
            warnings: list[str] = []
            result_exists = True
        else:
            result = self.cache.json(
                path, {}, max_bytes=ROLLING_STATUS_MAX_BYTES,
                fail_closed=True,
            )
            status = result.value if isinstance(result.value, dict) else {}
            warnings = [result.warning] if result.warning else []
            result_exists = result.exists
        base = {
            "available": bool(result_exists and status),
            "configured": True,
            "source": str(path),
            "pointer": str(pointer_path) if pointer_path else None,
            "pointer_verified": pointer_verified,
            "constraint_version": pointer_constraint_version,
            "constraints": pointer_constraints,
            "display_only": True,
            "production_eligible": False,
            "fea_submission_approved": False,
            "warnings": warnings,
            "lanes": [],
            "candidate_rows": [],
            "near_candidate_rows": [],
            "coherent_snapshot_verified": coherent_snapshot is not None,
            "coherent_snapshot_attempts": (
                coherent_snapshot.get("attempts")
                if coherent_snapshot is not None else None
            ),
        }
        if not status:
            return base
        if status.get("schema_version") == "mft-tier1-slurm-rolling-status-v1":
            return self._tier1_rolling_nsga_diagnostics(
                path=path,
                status=status,
                rolling_index=rolling_index,
                base=base,
                warnings=warnings,
            )
        if status.get("schema_version") != "mft-tier1-corrected-search-monitor-v1":
            warnings.append(f"unsupported Tier-1 NSGA status schema: {path}")
            return base
        status_constraint_version = _safe_text(
            status.get("constraint_version"), 160
        )
        status_constraints = (
            status.get("constraints")
            if isinstance(status.get("constraints"), dict) else {}
        )
        if pointer_verified and (
            not pointer_constraint_version
            or status_constraint_version != pointer_constraint_version
            or status_constraints != pointer_constraints
        ):
            warnings.append(
                "Tier-1 NSGA pointer/status constraint identity mismatch"
            )
            return {
                **base,
                "available": False,
                "integrity_verified": False,
            }

        updated_at = _safe_text(status.get("updated_at"), 100)
        updated_time = _parse_time(updated_at, self.clock().tzinfo)
        status_age_seconds = (
            max(0.0, (self.clock() - updated_time).total_seconds())
            if updated_time is not None else None
        )
        has_running_jobs = any(
            isinstance(item, dict)
            and str(item.get("state", "")).strip().lower() == "running"
            for item in (status.get("jobs") or [])
        )
        status_stale = bool(
            has_running_jobs
            and (
                status_age_seconds is None
                or status_age_seconds > 180
            )
        )
        if status_stale:
            warnings.append(
                "Tier-1 NSGA running status is stale; previous candidates are unavailable"
            )
            return {
                **base,
                "available": False,
                "stale": True,
                "updated_at": updated_at,
                "age_seconds": status_age_seconds,
                "integrity_verified": False,
            }

        model_sha = (
            _safe_text(status.get("model_manifest_sha256"), 64) or ""
        ).lower()
        code_revision = (
            _safe_text(status.get("nsga_code_revision"), 40) or ""
        ).lower()
        launch_sha = (_safe_text(status.get("launch_sha256"), 64) or "").lower()
        identity_shape_ok = bool(
            re.fullmatch(r"[0-9a-f]{64}", model_sha)
            and re.fullmatch(r"[0-9a-f]{40}", code_revision)
            and re.fullmatch(r"[0-9a-f]{64}", launch_sha)
        )
        if not identity_shape_ok:
            warnings.append("Tier-1 NSGA status has malformed identity digests")

        try:
            status_root = path.parent.resolve()
        except OSError:
            status_root = path.parent
        raw_jobs = status.get("jobs")
        raw_jobs = raw_jobs if isinstance(raw_jobs, list) else []
        state_counts: Counter[str] = Counter()
        lanes: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        terminal_results_verified = 0
        next_target_plan_count = 0
        all_jobs_integrity_ok = bool(raw_jobs)
        seen_seeds: set[int] = set()
        for position, raw_job in enumerate(raw_jobs[:128]):
            if not isinstance(raw_job, dict):
                warnings.append(f"Tier-1 NSGA job {position} is not an object")
                all_jobs_integrity_ok = False
                continue
            seed = _integer(raw_job.get("seed"), -1)
            state = (_safe_text(raw_job.get("state"), 40) or "unknown").lower()
            pid = max(0, _integer(raw_job.get("pid"), 0)) or None
            alive = raw_job.get("process_alive") is True
            launch_identity_ok = (
                raw_job.get("launch_command_identity_verified") is True
                or raw_job.get("command_identity_verified") is True
            )
            runtime_identity = raw_job.get("runtime_command_identity_verified")
            terminal_identity = raw_job.get("terminal_result_verified")
            stderr_size = max(0, _integer(raw_job.get("stderr_size"), 0))
            job_ok = bool(
                seed >= 0
                and seed not in seen_seeds
                and state in {"running", "completed", "failed"}
                and launch_identity_ok
                and stderr_size == 0
            )
            if seed >= 0:
                seen_seeds.add(seed)
            if state == "running":
                job_ok = bool(job_ok and alive and runtime_identity is True)
            elif state == "completed":
                job_ok = bool(
                    job_ok
                    and terminal_identity is True
                    and (not alive or runtime_identity is True)
                )

            public_job = {
                "name": f"tier1-feedback-seed-{seed}",
                "lane": f"tier1-feedback-seed-{seed}",
                "seed": seed if seed >= 0 else None,
                "state": state,
                "pid": pid,
                "alive": alive,
                "workers": 1 if state == "running" and alive else 0,
                "population": max(0, _integer(status.get("population"), 0)),
                "max_generations": max(
                    0, _integer(status.get("max_generations"), 0)
                ),
                "completed_generations": self._exact_integer(
                    raw_job.get("completed_generations")
                ),
                "feasible_pareto_count": self._exact_integer(
                    raw_job.get("feasible_pareto_count")
                ),
                "model_id": f"tier1-feedback:{model_sha}",
                "model_lane": "experimental_tier1_feedback",
                "launch_identity_verified": launch_identity_ok,
                "runtime_identity_verified": runtime_identity,
                "terminal_result_verified": terminal_identity,
                "process_reap_pending": state == "completed" and alive,
                "stderr_size": stderr_size,
                "source_kind": "tier1_feedback_nsga_status",
                "source": str(path),
            }

            if state == "completed" and job_ok:
                output_text = _safe_text(raw_job.get("output"), 4_096)
                expected_result_sha = (
                    _safe_text(raw_job.get("result_sha256"), 64) or ""
                ).lower()
                result_path: Path | None = None
                if output_text and re.fullmatch(r"[0-9a-f]{64}", expected_result_sha):
                    try:
                        output_path = Path(output_text).resolve()
                        if not output_path.is_relative_to(status_root):
                            raise ValueError("result directory escapes status root")
                        result_path = output_path / "result.json"
                    except (OSError, ValueError) as exc:
                        warnings.append(
                            f"Tier-1 seed {seed} result path rejected: {exc}"
                        )
                if result_path is None or _sha256_file(result_path) != expected_result_sha:
                    job_ok = False
                    warnings.append(f"Tier-1 seed {seed} result hash mismatch")
                else:
                    seed_result = self.cache.json(
                        result_path, {}, max_bytes=8 * 1024 * 1024
                    )
                    payload = (
                        seed_result.value
                        if isinstance(seed_result.value, dict) else {}
                    )
                    if seed_result.warning:
                        warnings.append(seed_result.warning)
                    payload_ok = bool(
                        payload.get("schema_version")
                        == "mft-tier1-corrected-search-seed-v1"
                        and _integer(payload.get("seed"), -2) == seed
                        and str(payload.get("model_manifest_sha256", "")).lower()
                        == model_sha
                        and str(payload.get("nsga_code_revision", "")).lower()
                        == code_revision
                        and payload.get("production_eligible") is False
                        and payload.get("fea_submission_approved") is False
                        and payload.get("automatic_promotion_allowed") is False
                        and (
                            not pointer_verified
                            or (
                                _safe_text(
                                    payload.get("constraint_version"), 160
                                ) == pointer_constraint_version
                                and payload.get("constraints")
                                == pointer_constraints
                            )
                        )
                    )
                    if not payload_ok:
                        job_ok = False
                        warnings.append(
                            f"Tier-1 seed {seed} terminal identity mismatch"
                        )
                    else:
                        terminal_results_verified += 1
                        raw_candidates = payload.get("candidates")
                        raw_candidates = (
                            raw_candidates if isinstance(raw_candidates, list)
                            else []
                        )
                        for raw_candidate in raw_candidates[:10_000]:
                            if not isinstance(raw_candidate, dict):
                                continue
                            candidate_rows.append({
                                "seed": seed,
                                "result_sha256": expected_result_sha,
                                "candidate": raw_candidate,
                            })
                        target_plan = payload.get("next_target_fea_batch_plan")
                        if isinstance(target_plan, dict):
                            next_target_plan_count += max(
                                0, _integer(target_plan.get("candidate_count"), 0)
                            )
                        public_job["result_sha256"] = expected_result_sha
                        public_job["result_source"] = str(result_path)
            if not job_ok:
                all_jobs_integrity_ok = False
            public_job["integrity_verified"] = job_ok
            state_counts[state] += 1
            lanes.append(public_job)

        production_safe = bool(
            status.get("production_eligible") is False
            and status.get("fea_submission_approved") is False
            and status.get("submission_performed") is False
        )
        if not production_safe:
            warnings.append(
                "Tier-1 NSGA status attempted production or FEA authority"
            )
        integrity_ok = bool(
            identity_shape_ok and all_jobs_integrity_ok and production_safe
        )
        return {
            **base,
            "updated_at": _safe_text(status.get("updated_at"), 100),
            "constraint_version": status_constraint_version,
            "constraints": status_constraints,
            "model_manifest_sha256": model_sha or None,
            "nsga_code_revision": code_revision or None,
            "launch_sha256": launch_sha or None,
            "population": max(0, _integer(status.get("population"), 0)),
            "max_generations": max(
                0, _integer(status.get("max_generations"), 0)
            ),
            "search_count": len(lanes),
            "running_count": state_counts.get("running", 0),
            "completed_count": state_counts.get("completed", 0),
            "failed_count": state_counts.get("failed", 0),
            "feasible_pareto_count": sum(
                max(0, item.get("feasible_pareto_count") or 0)
                for item in lanes
            ),
            "terminal_results_verified": terminal_results_verified,
            "next_target_plan_count": next_target_plan_count,
            "integrity_verified": integrity_ok,
            "lanes": lanes,
            "candidate_rows": candidate_rows,
            "warnings": list(dict.fromkeys(warnings)),
        }

    @staticmethod
    def _tier1_source_generation_identity(
        diagnostics: dict[str, Any],
    ) -> tuple[str, str, str, str, str, str]:
        temperature_contract = diagnostics.get(
            "temperature_constraint_contract"
        )
        temperature_contract = (
            temperature_contract
            if isinstance(temperature_contract, dict) else {}
        )
        return (
            _safe_text(diagnostics.get("constraint_version"), 160) or "",
            (_safe_text(diagnostics.get("hard_spec_sha256"), 64) or "").lower(),
            (
                _safe_text(diagnostics.get("model_manifest_sha256"), 64)
                or ""
            ).lower(),
            (
                _safe_text(
                    diagnostics.get("deployment_model_manifest_sha256"), 64
                ) or ""
            ).lower(),
            (
                _safe_text(diagnostics.get("nsga_code_revision"), 40) or ""
            ).lower(),
            _canonical_json_sha256(temperature_contract),
        )

    @staticmethod
    def _tier1_source_summary(
        diagnostics: dict[str, Any], *, accepted: bool,
        rejection_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "role": diagnostics.get("source_role") or "supplemental",
            "label": diagnostics.get("source_label") or "supplemental",
            "cohort_id": _safe_text(diagnostics.get("cohort_id"), 200),
            "constraint_version": _safe_text(
                diagnostics.get("constraint_version"), 160
            ),
            "hard_spec_sha256": _safe_text(
                diagnostics.get("hard_spec_sha256"), 64
            ),
            "model_manifest_sha256": _safe_text(
                diagnostics.get("model_manifest_sha256"), 64
            ),
            "deployment_model_manifest_sha256": _safe_text(
                diagnostics.get("deployment_model_manifest_sha256"), 64
            ),
            "nsga_code_revision": _safe_text(
                diagnostics.get("nsga_code_revision"), 40
            ),
            "available": diagnostics.get("available") is True,
            "integrity_verified": (
                diagnostics.get("integrity_verified") is True
            ),
            "pointer_verified": diagnostics.get("pointer_verified") is True,
            "bundle_manifest_verified": (
                diagnostics.get("bundle_manifest_verified") is True
                if "bundle_manifest_verified" in diagnostics else None
            ),
            "bundle_manifest_sha256": _safe_text(
                diagnostics.get("bundle_manifest_sha256"), 64
            ),
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "search_count": max(
                0, _integer(diagnostics.get("search_count"), 0)
            ),
            "completed_count": max(
                0, _integer(diagnostics.get("completed_count"), 0)
            ),
            "running_count": max(
                0, _integer(diagnostics.get("running_count"), 0)
            ),
            "queued_count": max(
                0, _integer(diagnostics.get("queued_count"), 0)
            ),
            "active_plus_queued": max(
                0, _integer(diagnostics.get("active_plus_queued"), 0)
            ),
            "feasible_pareto_count": max(
                0, _integer(diagnostics.get("feasible_pareto_count"), 0)
            ),
            "candidate_preview_count": max(
                0, _integer(diagnostics.get("candidate_preview_count"), 0)
            ),
            "near_feasible_count": max(
                0, _integer(diagnostics.get("near_feasible_count"), 0)
            ),
            "updated_at": _safe_text(diagnostics.get("updated_at"), 100),
        }

    def _tier1_archive_pointer_identity(self) -> str:
        """Fingerprint archived pointers without re-reading their large status."""
        pointers = []
        if self._tier1_nsga_pointer is not None:
            pointers.append(self._tier1_nsga_pointer)
        pointers.extend(
            root / "canonical" / "index.json"
            for root in self._tier1_nsga_supplemental_roots
        )
        identities: list[dict[str, Any]] = []
        for pointer in pointers:
            entry: dict[str, Any] = {"path": str(pointer)}
            try:
                if pointer.is_symlink() or not pointer.is_file():
                    raise ValueError("pointer is not a regular file")
                pointer_stat = pointer.stat()
                index, raw = _bounded_json_bytes(
                    pointer, ROLLING_INDEX_MAX_BYTES
                )
                entry.update({
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": pointer_stat.st_size,
                    "mtime_ns": pointer_stat.st_mtime_ns,
                })
                references = []
                for name in ("status", "model_pointer"):
                    declared = index.get(name)
                    declared = declared if isinstance(declared, dict) else {}
                    raw_reference_path = Path(
                        str(declared.get("path") or "")
                    )
                    reference = {
                        "name": name,
                        "path": str(raw_reference_path),
                        "sha256": str(declared.get("sha256") or "").lower(),
                    }
                    if (
                        raw_reference_path.is_absolute()
                        and not raw_reference_path.is_symlink()
                        and raw_reference_path.is_file()
                    ):
                        reference_stat = raw_reference_path.stat()
                        reference.update({
                            "size": reference_stat.st_size,
                            "mtime_ns": reference_stat.st_mtime_ns,
                        })
                    references.append(reference)
                entry["references"] = references
            except (OSError, TypeError, UnicodeError, ValueError) as exc:
                entry["unavailable"] = f"{type(exc).__name__}: {exc}"
            identities.append(entry)
        return _canonical_json_sha256(identities)

    def _archived_tier1_feedback_nsga_diagnostics(
        self,
    ) -> dict[str, Any]:
        """Reuse authenticated legacy evidence only while it is archive-only."""
        with self._tier1_archive_cache_lock:
            identity = self._tier1_archive_pointer_identity()
            cached = self._tier1_archive_cache
            if (
                isinstance(cached, dict)
                and cached.get("identity") == identity
            ):
                return copy.deepcopy(cached["diagnostics"])
            diagnostics = self._combined_tier1_feedback_nsga_diagnostics()
            self._tier1_archive_cache = {
                "identity": identity,
                "diagnostics": copy.deepcopy(diagnostics),
            }
            return diagnostics

    def _combined_tier1_feedback_nsga_diagnostics(
        self,
    ) -> dict[str, Any]:
        """Merge authenticated sidecars into, but never over, the primary.

        The configured primary rolling index remains the generation authority.
        Supplemental sources must independently pass the complete rolling
        artifact authentication and match the primary hard-spec, model, code,
        and temperature-contract identity.  A rejected source contributes no
        task, counter, or candidate data and cannot make the primary vanish.
        """
        primary = dict(self._tier1_feedback_nsga_diagnostics())
        primary.update({
            "source_role": "primary",
            "source_label": "main",
        })
        supplemental = self._tier1_supplemental_nsga_diagnostics()
        warnings = list(primary.get("warnings") or [])
        warnings.extend(self._tier1_nsga_supplemental_root_warnings)
        configured_source_count = 1 + len(
            self._tier1_nsga_supplemental_roots
        )
        primary_verified = bool(
            primary.get("available") is True
            and primary.get("integrity_verified") is True
            and (
                primary.get("pointer_verified") is True
                or not self._tier1_nsga_supplemental_roots
            )
        )
        accepted_sources: list[dict[str, Any]] = []
        source_summaries: list[dict[str, Any]] = []
        if primary_verified:
            accepted_sources.append(primary)
        source_summaries.append(self._tier1_source_summary(
            primary,
            accepted=primary_verified,
            rejection_reason=None if primary_verified else (
                "primary rolling generation is unavailable"
            ),
        ))

        primary_identity = self._tier1_source_generation_identity(primary)
        seen_cohorts = {
            _safe_text(primary.get("cohort_id"), 200) or ""
        }
        seen_task_ids = {
            _integer(item.get("task_id"), -1)
            for item in [
                *(primary.get("lanes") or []),
                *(primary.get("candidate_rows") or []),
                *(primary.get("near_candidate_rows") or []),
            ] if isinstance(item, dict) and _integer(item.get("task_id"), -1) > 0
        }
        seen_seeds = {
            _integer(item.get("seed"), -1)
            for item in [
                *(primary.get("lanes") or []),
                *(primary.get("candidate_rows") or []),
                *(primary.get("near_candidate_rows") or []),
            ] if isinstance(item, dict) and _integer(item.get("seed"), -1) >= 0
        }
        rejected_source_count = 0
        for source in supplemental:
            rejection_reason: str | None = None
            source_verified = bool(
                source.get("available") is True
                and source.get("integrity_verified") is True
                and source.get("pointer_verified") is True
            )
            if not primary_verified:
                rejection_reason = (
                    "primary rolling generation is unavailable; supplemental "
                    "cannot become authoritative"
                )
            elif not source_verified:
                rejection_reason = "supplemental rolling integrity is unavailable"
            elif self._tier1_source_generation_identity(source) != primary_identity:
                rejection_reason = (
                    "supplemental generation/model identity does not match primary"
                )
            elif source.get("constraints") != primary.get("constraints"):
                rejection_reason = (
                    "supplemental hard spec does not match primary"
                )
            source_cohort = _safe_text(source.get("cohort_id"), 200) or ""
            source_rows = [
                *(source.get("lanes") or []),
                *(source.get("candidate_rows") or []),
                *(source.get("near_candidate_rows") or []),
            ]
            source_task_ids = {
                _integer(item.get("task_id"), -1)
                for item in source_rows
                if isinstance(item, dict)
                and _integer(item.get("task_id"), -1) > 0
            }
            source_seeds = {
                _integer(item.get("seed"), -1)
                for item in source_rows
                if isinstance(item, dict)
                and _integer(item.get("seed"), -1) >= 0
            }
            if rejection_reason is None and (
                not source_cohort or source_cohort in seen_cohorts
            ):
                rejection_reason = "supplemental cohort identity is duplicated"
            if rejection_reason is None and (
                seen_task_ids & source_task_ids or seen_seeds & source_seeds
            ):
                rejection_reason = (
                    "supplemental task or seed identity overlaps another source"
                )
            accepted = rejection_reason is None
            source_summaries.append(self._tier1_source_summary(
                source, accepted=accepted, rejection_reason=rejection_reason,
            ))
            if not accepted:
                rejected_source_count += 1
                warnings.extend(source.get("warnings") or [])
                warnings.append(
                    f"Tier-1 supplemental source "
                    f"{source.get('source_label') or source_cohort or 'unknown'} "
                    f"rejected: {rejection_reason}"
                )
                continue
            accepted_sources.append(source)
            seen_cohorts.add(source_cohort)
            seen_task_ids.update(source_task_ids)
            seen_seeds.update(source_seeds)

        if not primary_verified:
            return {
                **primary,
                "source_count": 0,
                "supplemental_source_count": 0,
                "configured_source_count": configured_source_count,
                "rejected_source_count": rejected_source_count,
                "all_configured_sources_verified": False,
                "source_aggregation": "primary_authority_unavailable",
                "sources": source_summaries,
                "warnings": list(dict.fromkeys(warnings)),
            }

        def source_identity(source: dict[str, Any]) -> tuple[str, str, str]:
            return (
                source.get("source_role") or "supplemental",
                source.get("source_label") or "supplemental",
                _safe_text(source.get("cohort_id"), 200) or "unknown",
            )

        decorated_lanes: list[dict[str, Any]] = []
        candidate_queues: list[list[dict[str, Any]]] = []
        near_candidate_queues: list[list[dict[str, Any]]] = []
        state_counts: Counter[str] = Counter()
        for source in accepted_sources:
            role, label, cohort_id = source_identity(source)
            source_slug = re.sub(
                r"[^A-Za-z0-9_-]+", "-", label
            ).strip("-")[:48] or role
            for lane in source.get("lanes") or []:
                if not isinstance(lane, dict):
                    continue
                decorated = dict(lane)
                decorated.update({
                    "original_lane": lane.get("lane"),
                    "lane": (
                        f"tier1-{source_slug}-seed-"
                        f"{_integer(lane.get('seed'), -1)}"
                    ),
                    "source_role": role,
                    "source_label": label,
                    "source_cohort_id": cohort_id,
                    "alive": lane.get("state") == "running",
                })
                decorated_lanes.append(decorated)
            source_candidates: list[dict[str, Any]] = []
            for row in source.get("candidate_rows") or []:
                if not isinstance(row, dict):
                    continue
                decorated = dict(row)
                decorated.update({
                    "source_role": role,
                    "source_label": label,
                    "source_cohort_id": cohort_id,
                })
                source_candidates.append(decorated)
            candidate_queues.append(source_candidates)
            source_near_candidates: list[dict[str, Any]] = []
            for row in source.get("near_candidate_rows") or []:
                if not isinstance(row, dict):
                    continue
                decorated = dict(row)
                decorated.update({
                    "source_role": role,
                    "source_label": label,
                    "source_cohort_id": cohort_id,
                })
                source_near_candidates.append(decorated)
            near_candidate_queues.append(source_near_candidates)
            raw_counts = source.get("state_counts")
            if isinstance(raw_counts, dict):
                for state, count in raw_counts.items():
                    state_counts[str(state)] += max(0, _integer(count, 0))

        candidate_rows: list[dict[str, Any]] = []
        queue_offsets = [0] * len(candidate_queues)
        while len(candidate_rows) < ROLLING_CANDIDATE_PREVIEW_LIMIT:
            appended = False
            for position, queue in enumerate(candidate_queues):
                offset = queue_offsets[position]
                if offset >= len(queue):
                    continue
                candidate_rows.append(queue[offset])
                queue_offsets[position] += 1
                appended = True
                if len(candidate_rows) >= ROLLING_CANDIDATE_PREVIEW_LIMIT:
                    break
            if not appended:
                break

        near_candidate_rows: list[dict[str, Any]] = []
        near_queue_offsets = [0] * len(near_candidate_queues)
        while len(near_candidate_rows) < ROLLING_NEAR_FEASIBLE_PREVIEW_LIMIT:
            appended = False
            for position, queue in enumerate(near_candidate_queues):
                offset = near_queue_offsets[position]
                if offset >= len(queue):
                    continue
                near_candidate_rows.append(queue[offset])
                near_queue_offsets[position] += 1
                appended = True
                if (
                    len(near_candidate_rows)
                    >= ROLLING_NEAR_FEASIBLE_PREVIEW_LIMIT
                ):
                    break
            if not appended:
                break

        def total(name: str) -> int:
            return sum(
                max(0, _integer(source.get(name), 0))
                for source in accepted_sources
            )

        feasible_pareto_count = total("feasible_pareto_count")
        near_feasible_count = total("near_feasible_count")
        combined_rolling = dict(primary.get("rolling") or {})
        combined_rolling.update({
            "source_count": len(accepted_sources),
            "target_active_plus_queued": sum(
                max(
                    0,
                    _integer(
                        (source.get("rolling") or {}).get(
                            "target_active_plus_queued"
                        ),
                        _integer(source.get("active_plus_queued"), 0),
                    ),
                )
                for source in accepted_sources
            ),
            "active_plus_queued": total("active_plus_queued"),
            "seed_identity_count": sum(
                max(
                    0,
                    _integer(
                        (source.get("rolling") or {}).get(
                            "seed_identity_count"
                        ),
                        0,
                    ),
                )
                for source in accepted_sources
            ),
            "duplicate_seed_count": 0,
        })
        combined = {
            **primary,
            "updated_at": max(
                (
                    _safe_text(source.get("updated_at"), 100) or ""
                    for source in accepted_sources
                ),
                default="",
            ) or primary.get("updated_at"),
            "search_count": total("search_count"),
            "running_count": total("running_count"),
            "queued_count": total("queued_count"),
            "attaching_count": total("attaching_count"),
            "completed_count": total("completed_count"),
            "failed_count": total("failed_count"),
            "terminal_results_verified": total(
                "terminal_results_verified"
            ),
            "failed_terminal_results_verified": total(
                "failed_terminal_results_verified"
            ),
            "active_plus_queued": total("active_plus_queued"),
            "feasible_pareto_count": feasible_pareto_count,
            "candidate_preview_count": len(candidate_rows),
            "candidate_preview_limit": ROLLING_CANDIDATE_PREVIEW_LIMIT,
            "candidate_preview_truncated": (
                feasible_pareto_count > len(candidate_rows)
            ),
            "candidate_preview_contract_explicit": True,
            "near_feasible_count": near_feasible_count,
            "near_feasible_preview_count": len(near_candidate_rows),
            "near_feasible_preview_limit": (
                ROLLING_NEAR_FEASIBLE_PREVIEW_LIMIT
            ),
            "near_feasible_preview_truncated": (
                near_feasible_count > ROLLING_NEAR_FEASIBLE_PREVIEW_LIMIT
            ),
            "state_counts": dict(sorted(state_counts.items())),
            "rolling": combined_rolling,
            "lanes": decorated_lanes,
            "candidate_rows": candidate_rows,
            "near_candidate_rows": near_candidate_rows,
            "source_count": len(accepted_sources),
            "supplemental_source_count": len(accepted_sources) - 1,
            "configured_source_count": configured_source_count,
            "rejected_source_count": rejected_source_count,
            "all_configured_sources_verified": bool(
                rejected_source_count == 0
                and not self._tier1_nsga_supplemental_root_warnings
            ),
            "source_aggregation": "authenticated_source_pareto_union",
            "cohort_ids": [
                source_identity(source)[2] for source in accepted_sources
            ],
            "sources": source_summaries,
            "warnings": list(dict.fromkeys(warnings)),
        }
        return combined

    def _deadline_design_snapshot(
        self,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Read the optional actual-FEA deadline design without side effects."""

        path = self._deadline_design_publication_path
        sha256 = self._deadline_design_publication_sha256
        if not path and not sha256:
            return None, None
        if not path or not sha256:
            return None, (
                f"{DEADLINE_DESIGN_PATH_ENV} and "
                f"{DEADLINE_DESIGN_SHA256_ENV} must be configured together"
            )
        try:
            configured = Path(path)
            if not configured.is_absolute():
                raise DeadlineDesignError(
                    "deadline publication path must be absolute"
                )
            return load_deadline_design_publication(configured, sha256), None
        except (DeadlineDesignError, OSError, ValueError) as exc:
            return None, (
                "deadline design publication failed closed: "
                f"{type(exc).__name__}: {exc}"
            )

    def _continuous_nsga2(self) -> dict[str, Any] | None:
        """Project scheduler-discovered continuous NSGA-II results for the UI.

        A Pareto front is meaningful only within one surrogate identity.  If
        several result sets are available, prefer approved current-model
        evidence, then the newest current experimental model, and finally a
        historical model.  Other identities remain visible in ``result_sets``
        but are not mixed into the scatter plot.
        """
        local_lanes, local_warnings = self._local_nsga_ui_status()
        slurm_offload = self._slurm_nsga_offload_diagnostics()
        current7 = self._tier1_current7_diagnostics()
        current7_conditions = self._tier1_current7_condition_diagnostics()
        current7_condition_records = (
            self._current7_condition_generation_records(current7_conditions)
        )
        deadline_design, deadline_design_warning = (
            self._deadline_design_snapshot()
        )
        current7_configured = current7.get("configured") is True
        condition_archive_mode = bool(
            not current7_configured
            and self._tier1_current7_condition_configuration_intent
        )
        condition_archive_integrity = bool(
            condition_archive_mode
            and self._tier1_current7_condition_indexes
            and not self._current7_condition_configuration_warnings
            and len(current7_conditions)
            == len(self._tier1_current7_condition_indexes)
            and len(current7_condition_records) == len(current7_conditions)
            and all(
                condition.get("configured") is True
                and condition.get("available") is True
                and condition.get("integrity_verified") is True
                and condition.get("constraint_contract_verified") is True
                and condition.get("display_only") is True
                and condition.get("read_only") is True
                and condition.get("authority_eligible") is False
                for condition in current7_conditions
            )
        )
        tier1_feedback = (
            self._archived_tier1_feedback_nsga_diagnostics()
            if current7_configured
            else self._combined_tier1_feedback_nsga_diagnostics()
        )
        current7_authoritative = bool(
            current7_configured
            and current7.get("available") is True
            and current7.get("integrity_verified") is True
            and current7.get("healthy") is True
            and current7.get("constraint_contract_verified") is True
            and current7.get("authority_eligible") is True
        )
        tier1_feedback = {
            **tier1_feedback,
            "authority_role": (
                "archive"
                if current7_configured or condition_archive_mode
                else "primary"
            ),
            "archived": current7_configured or condition_archive_mode,
            "archive_state": (
                "stale"
                if (current7_configured or condition_archive_mode)
                and (
                    tier1_feedback.get("stale") is True
                    or tier1_feedback.get("integrity_verified") is not True
                )
                else (
                    "archived"
                    if current7_configured or condition_archive_mode
                    else None
                )
            ),
        }
        current7 = {
            **current7,
            "authority_role": "primary" if current7_configured else "secondary",
        }
        search_authority = {
            "configured": current7_configured,
            "kind": (
                "current7"
                if current7_configured
                else (
                    "condition_archive"
                    if condition_archive_mode else "legacy"
                )
            ),
            "available": (
                current7_authoritative
                if current7_configured
                else (
                    False
                    if condition_archive_mode
                    else tier1_feedback.get("available") is True
                )
            ),
            "integrity_verified": (
                current7_authoritative
                if current7_configured
                else (
                    False
                    if condition_archive_mode
                    else tier1_feedback.get("integrity_verified") is True
                )
            ),
            "source": (
                current7.get("source") if current7_configured else (
                    None
                    if condition_archive_mode else tier1_feedback.get("source")
                )
            ),
            "legacy_role": (
                "archive"
                if current7_configured or condition_archive_mode
                else "primary"
            ),
            "warning": (
                "read-only condition archive; no execution authority"
                if condition_archive_mode else (
                    None
                    if current7_authoritative or not current7_configured
                    else "configured current7 authority failed closed"
                )
            ),
        }

        def condition_restriction_key(
            item: dict[str, Any],
        ) -> tuple[float, float, float, float, float, str]:
            hard_spec = item.get("hard_spec")
            if not isinstance(hard_spec, dict):
                hard_spec = item.get("constraints")
            hard_spec = hard_spec if isinstance(hard_spec, dict) else {}

            def limit(name: str) -> float:
                value = hard_spec.get(name)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0
                ):
                    return float(value)
                return math.inf

            width = limit("size_W_max_mm")
            length = limit("size_L_max_mm")
            height = limit("size_H_max_mm")
            return (
                width * length * height,
                limit("T_limit_C"),
                width,
                length,
                height,
                str(item.get("constraint_version") or ""),
            )

        legacy_near_feasible_preview = self._tier1_near_feasible_preview(
            tier1_feedback
        )
        current7_near_feasible_preview = (
            self._tier1_current7_near_feasible_preview(current7)
            if current7_authoritative else []
        )
        accepted_condition_diagnostics = [
            condition for condition in current7_conditions
            if condition.get("integrity_verified") is True
            and condition.get("display_only") is True
            and condition.get("read_only") is True
            and condition.get("authority_eligible") is False
        ]
        selected_condition_diagnostics = (
            min(
                accepted_condition_diagnostics,
                key=condition_restriction_key,
            )
            if accepted_condition_diagnostics else None
        )
        condition_near_feasible_preview = (
            self._tier1_current7_near_feasible_preview(
                selected_condition_diagnostics
            )
            if selected_condition_diagnostics is not None else []
        )
        tier1_near_feasible_preview = (
            condition_near_feasible_preview
            if condition_archive_mode else (
                current7_near_feasible_preview
                if current7_configured else legacy_near_feasible_preview
            )
        )[:ROLLING_NEAR_FEASIBLE_PREVIEW_LIMIT]
        tier1_lanes = (
            []
            if condition_archive_mode else (
                tier1_feedback.get("lanes", [])
                if tier1_feedback.get("integrity_verified") is True else []
            )
        )
        current7_lanes = (
            current7.get("lanes", [])
            if current7_authoritative else []
        )
        reader = getattr(self.scheduler, "mft_pipeline_status", None)
        # The configured current7 index is the authenticated primary.  The
        # optional scheduler pipeline endpoint is legacy-only and commonly
        # absent on :8002; waiting for its timeout cannot add current7 truth.
        payload = (
            {}
            if current7_authoritative or condition_archive_mode
            else (reader() if callable(reader) else {})
        )
        if not isinstance(payload, dict) or payload.get("available") is False:
            payload = {}
        nsga = payload.get("nsga") if isinstance(payload.get("nsga"), dict) else {}
        if (
            not nsga
            and not local_lanes
            and not slurm_offload.get("available")
            and not tier1_feedback.get("available")
            and not current7.get("available")
            and not condition_archive_mode
            and deadline_design is None
        ):
            return None

        raw_sets = nsga.get("pareto_results")
        raw_sets = raw_sets if isinstance(raw_sets, list) else []
        raw_sets = [] if condition_archive_mode else list(raw_sets)
        for lane in ([] if condition_archive_mode else local_lanes):
            pareto = lane.get("pareto")
            if not isinstance(pareto, dict):
                continue
            raw_sets.append({
                **pareto,
                "lane": lane["lane"],
                "model_id": pareto.get("model_id") or lane["model_id"],
                "model_lane": pareto.get("model_lane") or lane["model_lane"],
                "scope": pareto.get("scope") or "current_model",
                "stale": current7_configured,
            })
        tier1_points_by_source_seed: dict[
            tuple[str, str, str, int], list[dict[str, Any]]
        ] = defaultdict(list)
        for record in (
            []
            if condition_archive_mode
            else tier1_feedback.get("candidate_rows", [])
        ):
            if not isinstance(record, dict):
                continue
            seed = _integer(record.get("seed"), -1)
            raw_candidate = record.get("candidate")
            if seed < 0 or not isinstance(raw_candidate, dict):
                continue
            source_role = _safe_text(record.get("source_role"), 40) or "primary"
            source_label = _safe_text(
                record.get("source_label"), 100
            ) or source_role
            source_cohort = _safe_text(
                record.get("source_cohort_id"), 200
            ) or (_safe_text(tier1_feedback.get("cohort_id"), 200) or "unknown")
            source_key = (
                source_role, source_label, source_cohort, seed
            )
            decoded = raw_candidate.get("decoded_params")
            decoded = decoded if isinstance(decoded, dict) else {}
            predictions = raw_candidate.get("predictions")
            predictions = predictions if isinstance(predictions, dict) else {}
            derived = raw_candidate.get("derived_resonance")
            derived = derived if isinstance(derived, dict) else {}
            embedded_row = dict(decoded)
            embedded_row.update({
                f"pred_{name}": value for name, value in predictions.items()
            })
            embedded_row.update(derived)
            embedded_row["volume_L"] = raw_candidate.get("volume_L")
            embedded_row["total_loss_W"] = raw_candidate.get("total_loss_W")
            tier1_points_by_source_seed[source_key].append({
                "candidate_index": _integer(
                    raw_candidate.get("index"),
                    len(tier1_points_by_source_seed[source_key]),
                ),
                "volume_L": raw_candidate.get("volume_L"),
                "total_loss_W": raw_candidate.get("total_loss_W"),
                "row": embedded_row,
                "authenticated_candidate": raw_candidate,
            })
        tier1_model_id = (
            f"tier1-feedback:{tier1_feedback.get('model_manifest_sha256')}"
            if tier1_feedback.get("integrity_verified") is True else ""
        )
        for source_key, points in sorted(
            tier1_points_by_source_seed.items()
        ):
            source_role, source_label, source_cohort, seed = source_key
            source_slug = re.sub(
                r"[^A-Za-z0-9_-]+", "-", source_label
            ).strip("-")[:48] or source_role
            raw_sets.append({
                "lane": f"tier1-{source_slug}-seed-{seed}",
                "scope": "current_model",
                "run_id": f"seed-{seed}",
                "model_id": tier1_model_id,
                "model_lane": "experimental_tier1_feedback",
                "finished_at": tier1_feedback.get("updated_at"),
                "stale": False,
                "production_eligible": False,
                "fea_submission_approved": False,
                "fea_verified": False,
                "pareto_front_sha256": "",
                "embedded_points": True,
                "constraint_version": tier1_feedback.get("constraint_version"),
                "hard_spec": tier1_feedback.get("constraints"),
                "tier1_source_role": source_role,
                "tier1_source_label": source_label,
                "tier1_source_cohort_id": source_cohort,
                "points": points,
            })
        current7_model_id = (
            f"tier1-current7:{current7.get('bundle_id')}"
            if current7_authoritative else ""
        )
        current7_points_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(
            list
        )
        for record in current7.get("candidate_rows", []):
            if not isinstance(record, dict):
                continue
            seed = _integer(record.get("seed"), -1)
            raw_candidate = record.get("candidate")
            if seed < 0 or not isinstance(raw_candidate, dict):
                continue
            decoded = raw_candidate.get("decoded_params")
            decoded = decoded if isinstance(decoded, dict) else {}
            predictions = raw_candidate.get("surrogate_mean_predictions")
            predictions = predictions if isinstance(predictions, dict) else {}
            embedded_row = {
                key: value for key, value in raw_candidate.items()
                if not isinstance(value, (dict, list))
            }
            embedded_row.update(decoded)
            embedded_row.update({
                f"pred_{name}": value for name, value in predictions.items()
            })
            loss_aliases = {
                "pred_total_loss_W": "predicted_total_loss_W",
                "pred_total_winding_loss_W": "predicted_winding_loss_W",
                "pred_component_winding_loss_sum_W": (
                    "predicted_winding_loss_W"
                ),
                "pred_core_loss_W": "predicted_core_loss_W",
                "pred_core_cold_plate_loss_W": (
                    "predicted_core_plate_loss_W"
                ),
                "pred_winding_cold_plate_loss_W": (
                    "predicted_winding_cold_plate_loss_W"
                ),
                "pred_primary_winding_loss_W": (
                    "predicted_Tx_main_winding_loss_W"
                ),
                "pred_secondary_center_winding_loss_W": (
                    "predicted_Rx_main_winding_loss_W"
                ),
                "pred_secondary_side_winding_loss_W": (
                    "predicted_Rx_side_winding_loss_W"
                ),
            }
            for alias, source_name in loss_aliases.items():
                if source_name in raw_candidate:
                    embedded_row[alias] = raw_candidate[source_name]
            secondary_center = _finite_number(
                embedded_row.get("pred_secondary_center_winding_loss_W")
            )
            secondary_side = _finite_number(
                embedded_row.get("pred_secondary_side_winding_loss_W")
            )
            if secondary_center is not None and secondary_side is not None:
                embedded_row["pred_secondary_winding_loss_W"] = (
                    secondary_center + secondary_side
                )
            current7_points_by_seed[seed].append({
                "candidate_index": _integer(
                    raw_candidate.get("terminal_population_index"),
                    len(current7_points_by_seed[seed]),
                ),
                "volume_L": raw_candidate.get("volume_L"),
                "total_loss_W": raw_candidate.get("total_loss_W"),
                "row": embedded_row,
                "authenticated_candidate": raw_candidate,
            })
        for seed, points in sorted(current7_points_by_seed.items()):
            raw_sets.append({
                "lane": f"tier1-current7-seed-{seed}",
                "scope": "current_model",
                "run_id": f"seed-{seed}",
                "model_id": current7_model_id,
                "model_lane": "corrected_current7",
                "finished_at": current7.get("updated_at"),
                "stale": not current7_authoritative,
                "production_eligible": False,
                "fea_submission_approved": False,
                "fea_verified": False,
                "gui_launch_eligible": False,
                "pareto_front_sha256": "",
                "embedded_points": True,
                "constraint_version": current7.get("constraint_version"),
                "hard_spec": current7.get("constraints"),
                "temperature_targets": current7.get("temperature_targets"),
                "tier1_source_role": "current7",
                "tier1_source_label": "corrected-current7",
                "tier1_source_cohort_id": current7.get("cohort_id"),
                "points": points,
            })
        result_sets: list[dict[str, Any]] = []
        warnings: list[str] = [
            *local_warnings,
            *self._current7_condition_configuration_warnings,
            *([deadline_design_warning] if deadline_design_warning else []),
            *(
                []
                if condition_archive_mode else (
                    current7.get("warnings", [])
                    if current7_configured
                    else tier1_feedback.get("warnings", [])
                )
            ),
            *(
                warning
                for condition in current7_conditions
                for warning in condition.get("warnings", [])
                if isinstance(warning, str) and warning.strip()
            ),
        ]
        if current7_configured:
            if not current7_authoritative:
                warnings.append(
                    "Configured current7 authority is unavailable or failed "
                    "integrity validation; no legacy fallback was allowed."
                )
        seen_result_sets: set[tuple[str, str, str, str]] = set()
        for index, raw in enumerate(raw_sets):
            if not isinstance(raw, dict):
                warnings.append(f"NSGA result set {index} is invalid")
                continue
            points = raw.get("points")
            points = points if isinstance(points, list) else []
            valid_points = [point for point in points if isinstance(point, dict)]
            item = {
                "lane": _safe_text(raw.get("lane"), 100) or f"lane-{index}",
                "scope": _safe_text(raw.get("scope"), 80) or "unknown",
                "run_id": _safe_text(raw.get("run_id"), 160) or "",
                "model_id": _safe_text(raw.get("model_id"), 500) or "",
                "model_lane": _safe_text(raw.get("model_lane"), 80),
                "finished_at": _safe_text(raw.get("finished_at"), 100),
                "stale": bool(
                    raw.get("stale") is True
                    or (
                        current7_configured
                        and raw.get("tier1_source_role") != "current7"
                    )
                ),
                "production_eligible": raw.get("production_eligible") is True,
                "fea_submission_approved": raw.get("fea_submission_approved") is True,
                "fea_verified": raw.get("fea_verified") is True,
                "pareto_front_sha256": (
                    _safe_text(raw.get("pareto_front_sha256"), 64) or ""
                ).lower(),
                "point_count": len(valid_points),
                "embedded_points": raw.get("embedded_points") is True,
                "gui_launch_eligible": raw.get("gui_launch_eligible") is not False,
                "constraint_version": _safe_text(
                    raw.get("constraint_version"), 160
                ),
                "hard_spec": (
                    dict(raw["hard_spec"])
                    if isinstance(raw.get("hard_spec"), dict) else None
                ),
                "temperature_targets": (
                    tuple(raw["temperature_targets"])
                    if isinstance(raw.get("temperature_targets"), list)
                    and all(
                        isinstance(value, str)
                        for value in raw["temperature_targets"]
                    )
                    else CANDIDATE_TEMPERATURE_TARGETS
                ),
                "tier1_source_role": _safe_text(
                    raw.get("tier1_source_role"), 40
                ),
                "tier1_source_label": _safe_text(
                    raw.get("tier1_source_label"), 100
                ),
                "tier1_source_cohort_id": _safe_text(
                    raw.get("tier1_source_cohort_id"), 200
                ),
                "points": valid_points,
            }
            identity = (
                item["lane"], item["run_id"], item["model_id"],
                item["pareto_front_sha256"],
            )
            if identity in seen_result_sets:
                continue
            seen_result_sets.add(identity)
            result_sets.append(item)

        active_local_lanes = [
            lane for lane in local_lanes
            if lane["state"] == "running" and lane["alive"] and lane["model_id"]
        ]
        preferred_model_id = (
            current7_model_id if current7_configured else tier1_model_id
        )
        if (
            not current7_configured
            and not preferred_model_id
            and active_local_lanes
        ):
            newest_lane = max(
                active_local_lanes,
                key=lambda lane: (
                    lane["dataset_rows"],
                    str(lane.get("training_run_id") or ""),
                    str(lane.get("updated_at") or ""),
                ),
            )
            preferred_model_id = newest_lane["model_id"]

        with_points = [item for item in result_sets if item["points"]]
        approved = [
            item for item in with_points
            if item["scope"] == "current_model"
            and not item["stale"]
            and item["production_eligible"]
            and item["fea_submission_approved"]
        ]
        current = [
            item for item in with_points
            if item["scope"] == "current_model" and not item["stale"]
        ]
        preferred = [
            item for item in with_points
            if preferred_model_id and item["model_id"] == preferred_model_id
        ]
        if current7_configured:
            # Explicit current7 configuration is a fail-closed authority
            # boundary.  Legacy/local rows remain inspectable as archived
            # result-set diagnostics but can never populate the active Pareto.
            selection_pool = preferred if current7_authoritative else []
        else:
            selection_pool = preferred or (
                [] if preferred_model_id else (
                    approved or current or [
                        item for item in with_points if not item["stale"]
                    ] or with_points
                )
            )

        selected_sets: list[dict[str, Any]] = []
        selected_model_id = preferred_model_id
        result_scope = "waiting_current" if preferred_model_id else "waiting"
        if selection_pool:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in selection_pool:
                groups[item["model_id"] or f"unknown:{item['lane']}"].append(item)
            selected_model_id, selected_sets = max(
                groups.items(),
                key=lambda pair: max(
                    str(item.get("finished_at") or "") for item in pair[1]
                ),
            )
            if preferred_model_id:
                result_scope = "current_model"
            elif approved:
                result_scope = "approved_current"
            elif current:
                result_scope = "experimental_current"
            else:
                result_scope = "historical"

        candidates: list[dict[str, Any]] = []
        for result_set in selected_sets:
            lane = result_set["lane"]
            run_id = result_set["run_id"] or "run"
            prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", f"{lane}-{run_id}").strip("-")
            for position, point in enumerate(result_set["points"]):
                volume = _finite_number(point.get("volume_L"))
                loss = _finite_number(point.get("total_loss_W"))
                if volume is None or loss is None:
                    warnings.append(
                        f"{lane}/{run_id} Pareto point {position} has invalid objectives"
                    )
                    continue
                design = point.get("design")
                source_index = _integer(point.get("candidate_index"), position)
                row_number = _integer(point.get("row_number"), source_index + 1)
                if result_set.get("embedded_points") is True:
                    embedded = point.get("row")
                    full_row = embedded if isinstance(embedded, dict) else None
                    artifact_path = None
                    artifact_warning = None
                else:
                    full_row, artifact_path, artifact_warning = self._pareto_row_by_sha(
                        result_set["pareto_front_sha256"], row_number
                    )
                    if artifact_warning:
                        warnings.append(f"{lane}/{run_id}: {artifact_warning}")
                row = dict(full_row) if isinstance(full_row, dict) else {}
                if isinstance(design, dict):
                    for key, value in design.items():
                        row.setdefault(key, value)
                row.update({"volume_L": volume, "total_loss_W": loss})
                authenticated_candidate = point.get("authenticated_candidate")
                if (
                    result_set.get("embedded_points") is True
                    and isinstance(authenticated_candidate, dict)
                    and isinstance(result_set.get("hard_spec"), dict)
                ):
                    candidate = self._authenticated_tier1_candidate(
                        row,
                        0,
                        position,
                        source_candidate=authenticated_candidate,
                        hard_spec=result_set["hard_spec"],
                        constraint_version=result_set.get("constraint_version"),
                        temperature_targets=tuple(
                            result_set.get("temperature_targets")
                            or CANDIDATE_TEMPERATURE_TARGETS
                        ),
                    )
                else:
                    candidate = self._candidate(row, 0, position)
                candidate.update({
                    "id": f"{prefix}-{source_index:04d}",
                    "round": None,
                    "result_lane": lane,
                    "run_id": run_id,
                    "model_id": result_set["model_id"],
                    "model_lane": result_set["model_lane"],
                    "result_scope": result_set["scope"],
                    "production_eligible": result_set["production_eligible"],
                    "fea_submission_approved": result_set["fea_submission_approved"],
                    "fea_verified": result_set["fea_verified"],
                    "pareto_row_number": row_number,
                    "pareto_front_sha256": result_set["pareto_front_sha256"] or None,
                    "artifact_hydrated": full_row is not None,
                    "artifact_source": str(artifact_path) if artifact_path else None,
                    "embedded_authenticated_result": (
                        result_set.get("embedded_points") is True
                    ),
                    "tier1_source_role": result_set.get(
                        "tier1_source_role"
                    ),
                    "tier1_source_label": result_set.get(
                        "tier1_source_label"
                    ),
                    "tier1_source_cohort_id": result_set.get(
                        "tier1_source_cohort_id"
                    ),
                    "gui_launch_eligible": result_set[
                        "gui_launch_eligible"
                    ],
                })
                candidates.append(candidate)

        current_compatible_scope = result_scope in {
            "current_model", "approved_current", "experimental_current",
        }
        constraint_valid_candidates = [
            item for item in candidates if item["spec_status"] == "pass"
        ]
        current_valid_candidates = (
            constraint_valid_candidates if current_compatible_scope else []
        )
        volume_candidates = [
            item for item in current_valid_candidates
            if item["volume_L"] is not None
        ]
        loss_candidates = [
            item for item in current_valid_candidates
            if item["total_loss_W"] is not None
        ]
        minimum_volume = min(volume_candidates, key=lambda item: item["volume_L"]) if volume_candidates else None
        minimum_loss = min(loss_candidates, key=lambda item: item["total_loss_W"]) if loss_candidates else None
        for candidate in candidates:
            candidate["is_min_volume"] = bool(
                minimum_volume and candidate["id"] == minimum_volume["id"]
            )
            candidate["is_min_loss"] = bool(
                minimum_loss and candidate["id"] == minimum_loss["id"]
            )
        candidates.sort(key=lambda item: (item["volume_L"] is None, item["volume_L"] or 0.0))
        candidate_summary = {
            "candidate_count": len(candidates),
            "display_candidate_count": len(candidates),
            "valid_candidate_count": len(current_valid_candidates),
            "historical_diagnostic_count": (
                len(candidates) if result_scope == "historical" else 0
            ),
            "min_volume_L": minimum_volume["volume_L"] if minimum_volume else None,
            "min_loss_W": minimum_loss["total_loss_W"] if minimum_loss else None,
            "min_volume_candidate_id": minimum_volume["id"] if minimum_volume else None,
            "min_loss_candidate_id": minimum_loss["id"] if minimum_loss else None,
            "known_spec_pass_count": sum(
                item["spec_status"] == "pass" for item in candidates
            ),
            "known_spec_fail_count": sum(
                item["spec_status"] == "fail" for item in candidates
            ),
            "unknown_spec_count": sum(
                item["spec_status"] == "unknown" for item in candidates
            ),
            "excluded_other_model_or_history_count": 0,
        }

        selected_keys = {
            (item["lane"], item["run_id"], item["model_id"])
            for item in selected_sets
        }
        result_summaries = []
        for item in result_sets:
            key = (item["lane"], item["run_id"], item["model_id"])
            result_summaries.append({
                name: value for name, value in item.items() if name != "points"
            } | {"selected": key in selected_keys})

        pareto_errors = nsga.get("pareto_errors")
        if isinstance(pareto_errors, list):
            for error in pareto_errors:
                if isinstance(error, str) and error.strip():
                    warnings.append(error.strip())
                elif isinstance(error, dict):
                    warnings.append(json.dumps(error, ensure_ascii=False, sort_keys=True))

        scheduler_active_workers = max(
            0, _integer(nsga.get("active_seed_workers"), 0)
        )
        scheduler_lanes = nsga.get("lanes")
        scheduler_lanes = scheduler_lanes if isinstance(scheduler_lanes, list) else []
        local_pids = {
            lane["pid"] for lane in local_lanes if lane.get("pid") is not None
        }
        local_identities = {
            (lane["lane"], lane["model_id"])
            for lane in local_lanes if lane["model_id"]
        }
        lanes = [
            lane for lane in scheduler_lanes
            if isinstance(lane, dict)
            and _integer(lane.get("pid"), 0) not in local_pids
            and (
                _safe_text(lane.get("name") or lane.get("lane"), 100) or "",
                _safe_text(lane.get("model_id"), 500) or "",
            ) not in local_identities
        ]
        lanes.extend(local_lanes)
        lanes.extend(tier1_lanes)
        lanes.extend(current7_lanes)
        local_active_workers = sum(
            lane["workers"] for lane in local_lanes
            if lane["state"] == "running" and lane["alive"]
        )
        tier1_active_workers = sum(
            max(0, _integer(lane.get("workers"), 0))
            for lane in tier1_lanes
            if lane.get("state") == "running" and lane.get("alive") is True
        )
        current7_active_workers = sum(
            max(0, _integer(lane.get("workers"), 0))
            for lane in current7_lanes
            if lane.get("state") == "running" and lane.get("alive") is True
        )
        # Scheduler and local projections may describe the same lanes.  Taking
        # the larger authenticated count avoids both double counting and the
        # current scheduler-only zero during local immutable launches.
        active_workers = (
            max(scheduler_active_workers, local_active_workers)
            + tier1_active_workers
            + current7_active_workers
        )
        condition_running = any(
            max(0, _integer(condition.get("running_count"), 0)) > 0
            for condition in accepted_condition_diagnostics
        )
        if condition_archive_mode:
            lanes = []
            scheduler_active_workers = 0
            local_active_workers = 0
            tier1_active_workers = 0
            current7_active_workers = 0
            active_workers = 0
            running = condition_running
        else:
            running = active_workers > 0 or any(
                isinstance(lane, dict)
                and str(lane.get("state", "")).lower() == "running"
                for lane in lanes
            )
        current_completed = max(0, _integer(nsga.get("current_model_completed_runs"), 0))
        current_feasible = max(0, _integer(nsga.get("current_model_feasible_runs"), 0))
        current_pareto = max(0, _integer(nsga.get("current_model_pareto_runs"), 0))
        historical_count = sum(
            item["point_count"] for item in result_sets
            if (item["lane"], item["run_id"], item["model_id"]) not in selected_keys
        )
        candidate_summary["excluded_other_model_or_history_count"] = historical_count
        selected_model_lanes = [
            lane for lane in local_lanes
            if selected_model_id and lane["model_id"] == selected_model_id
        ]
        selected_model_rows = max(
            (lane["dataset_rows"] for lane in selected_model_lanes),
            default=0,
        ) or None
        selected_training_run_id = next(
            (
                lane["training_run_id"] for lane in selected_model_lanes
                if lane.get("training_run_id")
            ),
            None,
        )
        if result_scope == "current_model":
            note = "현재 실행 중인 최신 모델의 NSGA-II Pareto 결과입니다."
        elif result_scope == "waiting_current":
            note = (
                "최신 모델의 병렬 NSGA-II lane은 실행 중이며 해당 모델의 "
                "Pareto 결과를 기다리고 있습니다."
            )
        elif result_scope == "approved_current":
            note = "현재 승인 모델의 NSGA-II Pareto 결과입니다."
        elif result_scope == "experimental_current":
            note = (
                "현재 표시 결과는 최신 실험 모델의 Pareto 탐색 결과이며 "
                "production/FEA 제출 승인은 아직 아닙니다."
            )
        elif result_scope == "historical":
            note = "현재 모델 결과가 없어 가장 최근 과거 Pareto 이력을 표시합니다."
        else:
            note = "연속 NSGA-II worker는 연결됐지만 표시할 Pareto 결과가 아직 없습니다."
        if historical_count:
            note += f" 다른 모델/과거 이력 {historical_count}개는 혼합하지 않았습니다."
        if candidates and not current_valid_candidates:
            if result_scope == "historical":
                note += (
                    f" 현재 모델에서 유효한 설계는 0개이며 과거 진단 후보 "
                    f"{len(candidates)}개만 참고용으로 표시합니다."
                )
            else:
                note += (
                    f" 표시된 탐색 후보 {len(candidates)}개 중 현재 제약을 "
                    "모두 통과한 설계는 아직 0개입니다."
                )
        if slurm_offload.get("available"):
            note += (
                " Slurm 진단 탐색은 "
                f"{slurm_offload.get('production_lane_count', 0)}개 lane에서 "
                f"least-violation 후보 "
                f"{slurm_offload.get('total_least_violation_candidates', 0)}개를 "
                f"검증했으며 feasible lane은 "
                f"{slurm_offload.get('feasible_lane_count', 0)}개입니다."
            )
        if tier1_feedback.get("available"):
            note += (
                " Tier-1 보정 탐색은 "
                f"{tier1_feedback.get('completed_count', 0)}/"
                f"{tier1_feedback.get('search_count', 0)}개 seed 완료, "
                f"{tier1_feedback.get('running_count', 0)}개 실행 중이며 "
                f"robust-feasible Pareto는 "
                f"{tier1_feedback.get('feasible_pareto_count', 0)}개입니다. "
                "이 탐색은 production 및 FEA 자동 제출 권한이 없습니다."
            )
        if current7.get("available"):
            note += (
                " Corrected current7 search: "
                f"{current7.get('completed_count', 0)}/"
                f"{current7.get('search_count', 0)} seeds completed, "
                f"{current7.get('running_count', 0)} running, "
                f"{current7.get('feasible_pareto_count', 0)} authenticated "
                "Pareto candidates. This lane has no automatic production, "
                "FEA, AEDT, or promotion authority."
            )
        if condition_archive_mode:
            note = (
                f"인증된 조건별 탐색 결과 {len(current7_condition_records)}개를 "
                "읽기 전용으로 표시합니다. 유효 Pareto와 근접 후보는 각 조건 "
                "탭에서 확인할 수 있으며 실행, FEA, AEDT, production 또는 "
                "promotion 권한은 없습니다."
            )
        note = _normalized_display_text(
            note,
            "NSGA-II status: "
            f"{current_completed} current-model runs completed; "
            f"{len(candidates)} candidates displayed.",
            limit=4_000,
        )

        active_generation_source = (
            current7 if current7_configured else tier1_feedback
        )
        if condition_archive_mode:
            # A stopped primary producer is not execution authority.  Keep its
            # legacy snapshot out of the active selector and expose only the
            # authenticated, read-only final-goal condition generations.
            condition_order = {
                (
                    condition.get("constraint_version"),
                    condition.get("hard_spec_sha256"),
                ): position
                for position, condition in enumerate(current7_conditions)
            }
            generation_records = sorted(
                current7_condition_records,
                key=lambda item: condition_order.get(
                    (
                        item.get("constraint_version"),
                        item.get("hard_spec_sha256"),
                    ),
                    len(condition_order),
                ),
            )
        else:
            generation_records = self._tier1_generation_records(
                active_generation_source,
                active_candidate_count=max(
                    0,
                    _integer(
                        active_generation_source.get(
                            "feasible_pareto_count"
                        ),
                        0,
                    ),
                ),
                active_display_count=len(candidates),
            )
            existing_generation_ids = {
                item.get("id") for item in generation_records
                if isinstance(item, dict)
            }
            generation_records.extend(
                item for item in current7_condition_records
                if item.get("id") not in existing_generation_ids
            )
        if deadline_design is not None:
            deadline_record = dict(deadline_design["generation"])
            if all(
                item.get("id") != deadline_record.get("id")
                for item in generation_records
                if isinstance(item, dict)
            ):
                generation_records.append(deadline_record)
        pareto_generations = [
            self._public_tier1_generation(item)
            for item in generation_records
        ]

        if deadline_design is not None:
            selected_generation_id = deadline_design["generation"]["id"]
        elif condition_archive_mode and pareto_generations:
            selected_generation_id = min(
                pareto_generations, key=condition_restriction_key
            )["id"]
        else:
            selected_generation_id = next(
                (
                    item["id"] for item in pareto_generations
                    if item.get("active") is True
                ),
                None,
            )
        selected_condition = next(
            (
                item for item in pareto_generations
                if item.get("id") == selected_generation_id
                and item.get("source_kind") == "current7_condition_index"
            ),
            {},
        )
        selected_condition_spec = selected_condition.get("hard_spec")
        selected_condition_spec = (
            dict(selected_condition_spec)
            if isinstance(selected_condition_spec, dict) else {}
        )
        selected_deadline = (
            deadline_design["generation"]
            if deadline_design is not None
            and deadline_design["generation"].get("id")
            == selected_generation_id
            else {}
        )
        selected_deadline_spec = selected_deadline.get("hard_spec")
        selected_deadline_spec = (
            dict(selected_deadline_spec)
            if isinstance(selected_deadline_spec, dict) else {}
        )
        selected_candidates = candidates
        selected_valid_candidate_count = len(current_valid_candidates)
        selected_candidate_summary = candidate_summary
        selected_near_feasible_preview = tier1_near_feasible_preview
        selected_note = note
        selected_warnings = list(warnings)
        if selected_deadline:
            # The inventory and selected-generation identity already point at
            # the authenticated deadline publication.  Project that same
            # immutable candidate into the top-level view so API clients and
            # the first UI paint cannot contradict the selected generation
            # with an unrelated archive's zero-candidate summary.
            deadline_payload = deadline_design.get("payload")
            deadline_payload = (
                deadline_payload
                if isinstance(deadline_payload, dict) else {}
            )
            deadline_candidates = deadline_payload.get("candidates")
            deadline_summary = deadline_payload.get("summary")
            deadline_near = deadline_payload.get("near_feasible_preview")
            if (
                deadline_payload.get("integrity_verified") is not True
                or not isinstance(deadline_candidates, list)
                or len(deadline_candidates) != 1
                or not isinstance(deadline_candidates[0], dict)
                or deadline_candidates[0].get("id")
                != deadline_design["candidate"].get("id")
                or deadline_payload.get("candidate_count") != 1
                or deadline_payload.get("display_candidate_count") != 1
                or deadline_payload.get("valid_candidate_count") != 1
                or not isinstance(deadline_summary, dict)
                or not isinstance(deadline_near, list)
            ):
                raise DeadlineDesignError(
                    "selected deadline generation projection failed closed"
                )
            selected_candidates = copy.deepcopy(deadline_candidates)
            selected_valid_candidate_count = 1
            selected_candidate_summary = copy.deepcopy(deadline_summary)
            selected_near_feasible_preview = copy.deepcopy(deadline_near)
            selected_note = (
                _safe_text(deadline_payload.get("note"), 4_000) or note
            )
            deadline_warnings = deadline_payload.get("warnings")
            deadline_warnings = (
                deadline_warnings
                if isinstance(deadline_warnings, list) else []
            )
            selected_warnings = [
                *(
                    _safe_text(value, 4_000)
                    for value in deadline_warnings
                    if _safe_text(value, 4_000)
                ),
                *warnings,
            ]

        return {
            "schema_version": SCHEMA_VERSION,
            "available": bool(
                deadline_design is not None
                or condition_archive_integrity
                or (
                    (candidates or running or lanes)
                    and search_authority["available"]
                    and search_authority["integrity_verified"]
                )
            ),
            "search_authority": search_authority,
            "status": (
                "running"
                if running else (
                    "completed"
                    if (
                        deadline_design is not None
                        or condition_archive_integrity or candidates
                    ) else "waiting"
                )
            ),
            "source_kind": (
                "deadline_design_with_condition_archive"
                if deadline_design is not None and condition_archive_mode else (
                "deadline_validated_design"
                if deadline_design is not None else (
                "current7_condition_archive"
                if condition_archive_mode else (
                "continuous_nsga_with_current7"
                if current7_configured else
                ("continuous_nsga_with_tier1_feedback"
                if tier1_feedback.get("available") else
                ("scheduler_and_local_continuous_nsga"
                 if nsga and local_lanes else
                 ("local_continuous_nsga" if local_lanes
                  else "scheduler_continuous_nsga")))))
                )
            ),
            "result_scope": result_scope,
            "round": None,
            "al_round": None,
            "al_stage": "CONTINUOUS",
            "candidate_count": len(selected_candidates),
            "display_candidate_count": len(selected_candidates),
            "valid_candidate_count": selected_valid_candidate_count,
            "historical_diagnostic_count": (
                len(selected_candidates)
                if result_scope == "historical" and not selected_deadline
                else 0
            ),
            "configured_restarts": None,
            "completed_restarts": current_completed,
            "active_seed_workers": active_workers,
            "scheduler_active_seed_workers": scheduler_active_workers,
            "local_active_seed_workers": local_active_workers,
            "tier1_active_seed_workers": tier1_active_workers,
            "current7_active_seed_workers": current7_active_workers,
            "current_model_completed_runs": current_completed,
            "current_model_feasible_runs": current_feasible,
            "current_model_pareto_runs": current_pareto,
            "lifetime_completed_runs": max(
                0, _integer(nsga.get("lifetime_completed_runs"), 0)
            ),
            "selected_model_id": selected_model_id or None,
            "selected_training_run_id": selected_training_run_id,
            "selected_model_dataset_rows": selected_model_rows,
            "lanes": lanes,
            "candidates": selected_candidates,
            "near_feasible_preview": selected_near_feasible_preview,
            "near_feasible_preview_count": len(
                selected_near_feasible_preview
            ),
            "summary": selected_candidate_summary,
            "comparison": None,
            "rounds": [],
            "result_sets": result_summaries,
            "pareto_generations": pareto_generations,
            "selected_generation_id": selected_generation_id,
            "slurm_offload": slurm_offload,
            "tier1_feedback_search": {
                key: value for key, value in tier1_feedback.items()
                if key not in {"candidate_rows", "near_candidate_rows"}
            },
            "tier1_current7_search": {
                key: value for key, value in current7.items()
                if key not in {"candidate_rows", "near_candidate_rows"}
            },
            "tier1_current7_condition_searches": [
                {
                    key: value for key, value in condition.items()
                    if key not in {
                        "candidate_rows", "near_candidate_rows", "lanes",
                    }
                }
                for condition in current7_conditions
            ],
            "constraint_version": (
                selected_deadline.get("constraint_version")
                if selected_deadline else (
                selected_condition.get("constraint_version")
                if condition_archive_mode else (
                    current7.get("constraint_version")
                    if current7_configured
                    else tier1_feedback.get("constraint_version")
                ))
            ),
            "constraints": (
                selected_deadline_spec
                if selected_deadline else (
                selected_condition_spec
                if condition_archive_mode else (
                    (current7.get("constraints") or {})
                    if current7_configured
                    else (tier1_feedback.get("constraints") or {})
                ))
            ),
            "source": (
                selected_deadline.get("candidate_endpoint")
                if selected_deadline else (
                selected_condition.get("candidate_endpoint")
                if condition_archive_mode else (
                    f"{getattr(self.scheduler, 'base_url', '')}/api/mft-pipeline/status"
                    if nsga else (
                        current7.get("source")
                        if current7_configured else (
                            tier1_feedback.get("source")
                            if tier1_feedback.get("available")
                            else "local ui_status.json"
                        )
                    )
                ))
            ),
            "updated_at": max(
                [
                    value for value in [
                        _safe_text(payload.get("generated_at"), 100),
                        *(
                            _safe_text(lane.get("updated_at"), 100)
                            for lane in local_lanes
                        ),
                        _safe_text(tier1_feedback.get("updated_at"), 100),
                        _safe_text(current7.get("updated_at"), 100),
                        *(
                            _safe_text(condition.get("updated_at"), 100)
                            for condition in current7_conditions
                        ),
                        (
                            _safe_text(
                                deadline_design["generation"].get(
                                    "updated_at"
                                ),
                                100,
                            )
                            if deadline_design is not None else None
                        ),
                    ] if value
                ],
                default=None,
            ),
            "note": selected_note,
            "warnings": list(dict.fromkeys(selected_warnings)),
        }

    def deadline_design_candidate(
        self, candidate_id: str
    ) -> dict[str, Any] | None:
        """Resolve only the configured, hash-authenticated deadline point."""

        candidate_id = _safe_text(candidate_id, 200) or ""
        deadline_design, warning = self._deadline_design_snapshot()
        if warning:
            raise DeadlineDesignError(warning)
        if deadline_design is None:
            return None
        candidate = deadline_design.get("candidate")
        if (
            isinstance(candidate, dict)
            and candidate.get("id") == candidate_id
        ):
            return dict(candidate)
        return None

    def nsga2_generation(self, generation_id: str) -> dict[str, Any]:
        """Load one condition generation on demand.

        The dashboard inventory never embeds historical candidates.  This
        bounded endpoint performs the expensive terminal/result authentication
        only after the operator selects a generation and caches immutable
        snapshots by path, size, and mtime.
        """
        generation_id = _safe_text(generation_id, 80) or ""
        if not re.fullmatch(
            r"(?:tier1|tier1-hit)-[0-9a-f]{20}", generation_id
        ):
            return {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "status": "unavailable",
                "selected_generation_id": generation_id or None,
                "candidates": [],
                "summary": {},
                "warnings": ["알 수 없는 NSGA 조건 세대 ID입니다."],
            }
        current = self._continuous_nsga2()
        if not isinstance(current, dict):
            return {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "status": "unavailable",
                "selected_generation_id": generation_id,
                "candidates": [],
                "summary": {},
                "warnings": ["NSGA 조건 세대 목록을 읽을 수 없습니다."],
            }
        generations = current.get("pareto_generations")
        generations = generations if isinstance(generations, list) else []
        selected = next(
            (
                item for item in generations
                if isinstance(item, dict) and item.get("id") == generation_id
            ),
            None,
        )
        if selected is None:
            return {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "status": "unavailable",
                "selected_generation_id": generation_id,
                "candidates": [],
                "summary": {},
                "warnings": ["요청한 NSGA 조건 세대가 현재 목록에 없습니다."],
            }
        if selected.get("selectable") is False:
            warning = selected.get("warning") or (
                "이 조건 세대는 완료된 인증 결과가 없어 선택할 수 없습니다."
            )
            return {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "integrity_verified": False,
                "status": "unavailable",
                "selected_generation_id": generation_id,
                "generation": selected,
                "candidates": [],
                "summary": {},
                "note": warning,
                "warning": warning,
                "warnings": [warning],
            }
        if selected.get("active") is True:
            return {
                **current,
                "selected_generation_id": generation_id,
                "generation": selected,
            }

        if selected.get("source_kind") == "deadline_validated_design":
            deadline_design, deadline_warning = (
                self._deadline_design_snapshot()
            )
            if (
                deadline_design is None
                or deadline_design["generation"].get("id") != generation_id
            ):
                warning = deadline_warning or (
                    "deadline design publication identity changed"
                )
                return {
                    "schema_version": SCHEMA_VERSION,
                    "available": False,
                    "integrity_verified": False,
                    "status": "unavailable",
                    "selected_generation_id": generation_id,
                    "generation": selected,
                    "candidates": [],
                    "summary": {},
                    "warnings": [warning],
                }
            return {
                **deadline_design["payload"],
                "selected_generation_id": generation_id,
                "generation": self._public_tier1_generation(
                    deadline_design["generation"]
                ),
            }

        if selected.get("source_kind") == "current7_condition_index":
            condition_record = next(
                (
                    item
                    for item in self._current7_condition_generation_records()
                    if item.get("id") == generation_id
                ),
                None,
            )
            if condition_record is None:
                return {
                    "schema_version": SCHEMA_VERSION,
                    "available": False,
                    "integrity_verified": False,
                    "status": "unavailable",
                    "selected_generation_id": generation_id,
                    "generation": selected,
                    "candidates": [],
                    "summary": {},
                    "warnings": [
                        "Current7 condition index failed closed during detail "
                        "authentication."
                    ],
                }
            diagnostics = condition_record["_diagnostics"]
            if (
                condition_record.get("condition_index_schema_version")
                == CURRENT7_COMPACT_INDEX_SCHEMA
            ):
                with self._current7_diagnostics_lock:
                    diagnostics = (
                        self._tier1_compact_condition_diagnostics_locked(
                            index_path=condition_record["_path"],
                            hydrate_candidates=True,
                        )
                    )
            return self._current7_condition_generation_payload(
                diagnostics,
                condition_record,
            )

        tier1 = current.get("tier1_feedback_search")
        tier1 = tier1 if isinstance(tier1, dict) else {}
        records = self._tier1_generation_records(
            tier1,
            active_candidate_count=max(
                0, _integer(tier1.get("feasible_pareto_count"), 0)
            ),
            active_display_count=max(
                0, _integer(current.get("display_candidate_count"), 0)
            ),
        )
        record = next(
            (item for item in records if item.get("id") == generation_id),
            None,
        )
        if record is None:
            return {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "status": "unavailable",
                "selected_generation_id": generation_id,
                "candidates": [],
                "summary": {},
                "warnings": ["NSGA 조건 세대 status가 사라졌습니다."],
            }
        # A self-declared zero count is not proof that a generation had no
        # Pareto points.  Empty and non-empty generations therefore take the
        # same terminal/result/aggregate authentication path.  The expensive
        # result is cached by the immutable status identity after the first
        # operator click.
        diagnostics = self._historical_tier1_diagnostics(record)
        return self._tier1_generation_payload(diagnostics, record)

    def nsga2(self) -> dict[str, Any]:
        sealed_successor = self._sealed_successor_evidence()
        sealed_successors = self._dual_sealed_successors()
        continuous = self._continuous_nsga2()
        if continuous is not None:
            return {
                **continuous,
                "sealed_successor": sealed_successor,
                "sealed_successors": sealed_successors,
            }
        state_result = self.cache.json(self.root / "al_rounds" / "state.json", {})
        state = state_result.value if isinstance(state_result.value, dict) else {}
        rounds = self._round_directories()
        warnings = self._warnings(state_result)
        if not rounds:
            return {
                "schema_version": SCHEMA_VERSION,
                "available": False,
                "status": "waiting",
                "round": _integer(state.get("round"), 0) if state else None,
                "al_stage": _safe_text(state.get("stage"), 30),
                "candidate_count": 0,
                "display_candidate_count": 0,
                "valid_candidate_count": 0,
                "historical_diagnostic_count": 0,
                "configured_restarts": 16,
                "completed_restarts": None,
                "candidates": [],
                "summary": {},
                "rounds": [],
                "sealed_successor": sealed_successor,
                "sealed_successors": sealed_successors,
                "warnings": warnings,
            }

        round_summaries: list[dict[str, Any]] = []
        parsed_rounds: dict[int, tuple[ReadResult, list[dict[str, str]]]] = {}
        for number, directory in rounds:
            result = self.cache.csv(directory / "pareto_front.csv", max_rows=20_000)
            rows = result.value if isinstance(result.value, list) else []
            parsed_rounds[number] = (result, rows)
            if result.warning:
                warnings.append(result.warning)
            volumes = [value for row in rows if (value := _finite_number(row.get("volume_L"))) is not None]
            losses = [value for row in rows if (value := _finite_number(row.get("total_loss_W"))) is not None]
            round_summaries.append({
                "round": number,
                "candidate_count": len(rows),
                "min_volume_L": min(volumes) if volumes else None,
                "min_loss_W": min(losses) if losses else None,
                "updated_at": _iso(result.mtime),
            })

        latest_round, latest_dir = rounds[-1]
        latest_result, latest_rows = parsed_rounds[latest_round]
        candidates = [self._candidate(row, latest_round, index) for index, row in enumerate(latest_rows)]
        valid_candidates = [
            item for item in candidates if item["spec_status"] == "pass"
        ]
        volume_candidates = [
            item for item in valid_candidates if item["volume_L"] is not None
        ]
        loss_candidates = [
            item for item in valid_candidates if item["total_loss_W"] is not None
        ]
        minimum_volume = min(volume_candidates, key=lambda item: item["volume_L"]) if volume_candidates else None
        minimum_loss = min(loss_candidates, key=lambda item: item["total_loss_W"]) if loss_candidates else None
        for candidate in candidates:
            candidate["is_min_volume"] = bool(minimum_volume and candidate["id"] == minimum_volume["id"])
            candidate["is_min_loss"] = bool(minimum_loss and candidate["id"] == minimum_loss["id"])
        candidates.sort(key=lambda item: (item["volume_L"] is None, item["volume_L"] or 0.0))

        latest_summary = round_summaries[-1]
        previous_summary = round_summaries[-2] if len(round_summaries) > 1 else None
        comparison = None
        if previous_summary:
            comparison = {
                "previous_round": previous_summary["round"],
                "min_volume_change_L": (
                    latest_summary["min_volume_L"] - previous_summary["min_volume_L"]
                    if latest_summary["min_volume_L"] is not None and previous_summary["min_volume_L"] is not None else None
                ),
                "min_loss_change_W": (
                    latest_summary["min_loss_W"] - previous_summary["min_loss_W"]
                    if latest_summary["min_loss_W"] is not None and previous_summary["min_loss_W"] is not None else None
                ),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "available": True,
            "status": "running" if str(state.get("stage", "")).upper() == "OPTIMIZE" else "completed",
            "round": latest_round,
            "al_round": _integer(state.get("round"), latest_round) if state else latest_round,
            "al_stage": _safe_text(state.get("stage"), 30),
            "candidate_count": len(candidates),
            "display_candidate_count": len(candidates),
            "valid_candidate_count": len(valid_candidates),
            "historical_diagnostic_count": 0,
            "configured_restarts": 16,
            "completed_restarts": state.get("nsga2_restarts_completed") if state else None,
            "candidates": candidates,
            "summary": {
                **latest_summary,
                "display_candidate_count": len(candidates),
                "valid_candidate_count": len(valid_candidates),
                "historical_diagnostic_count": 0,
                "min_volume_candidate_id": minimum_volume["id"] if minimum_volume else None,
                "min_loss_candidate_id": minimum_loss["id"] if minimum_loss else None,
                "known_spec_pass_count": sum(item["spec_status"] == "pass" for item in candidates),
                "known_spec_fail_count": sum(item["spec_status"] == "fail" for item in candidates),
                "unknown_spec_count": sum(item["spec_status"] == "unknown" for item in candidates),
            },
            "comparison": comparison,
            "rounds": round_summaries,
            "sealed_successor": sealed_successor,
            "sealed_successors": sealed_successors,
            "source": str(latest_dir / "pareto_front.csv"),
            "updated_at": _iso(latest_result.mtime),
            "note": "Pareto 파일의 해는 최적화기가 feasible로 반환한 후보입니다. 아직 학습되지 않은 출력의 명목 사양 판정은 ‘확인 불가’로 표시합니다.",
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _evaluate_fea(self, result: dict[str, Any], require_full_model: bool = False) -> dict[str, Any]:
        full_model_value = _finite_number(result.get("full_model"))
        llt = _finite_number(result.get("Llt_phys"))
        if llt is None:
            raw_llt = _finite_number(result.get("Llt"))
            if raw_llt is not None:
                llt = raw_llt * (1.0 if full_model_value == 1.0 else 2.0)
        bmax = _finite_number(result.get("B_max_core"))
        n2_side = _finite_number(result.get("N2_side"))
        temperature_keys = ["T_max_Tx", "T_max_Rx_main", "T_max_core"]
        if n2_side is not None and n2_side > 0:
            temperature_keys.insert(2, "T_max_Rx_side")
        temperatures = {key: _finite_number(result.get(key)) for key in temperature_keys}
        finite_temperatures = [value for value in temperatures.values() if value is not None]
        max_temperature = max(finite_temperatures) if len(finite_temperatures) == len(temperature_keys) else None
        insulation_values = [
            value for key in INSULATION_KEYS if (value := _finite_number(result.get(key))) is not None
        ]
        min_insulation = min(insulation_values) if insulation_values else None
        matrix_error = _finite_number(result.get("conv_error_pct_matrix"))
        loss_error = _finite_number(result.get("conv_error_pct_loss"))
        convergence_value = max(matrix_error, loss_error) if matrix_error is not None and loss_error is not None else None
        loss_components = [
            _finite_number(result.get(key)) for key in
            ("P_winding_total", "P_core_total", "P_core_plate_total", "P_wcp_total")
        ]
        losses_complete = all(value is not None for value in loss_components)
        losses_nonnegative = (
            all(value >= 0 for value in loss_components) if losses_complete else None
        )
        total_loss = sum(loss_components) if losses_complete else None
        checks = {
            "llt": self._constraint(llt, (26.95, 28.05), "band"),
            "temperature": self._constraint(max_temperature, 100.0, "max"),
            "bmax": self._constraint(bmax, 1.2, "max"),
            "insulation": self._constraint(min_insulation, 40.0, "min"),
            "convergence": self._constraint(convergence_value, 1.5, "max"),
            "loss_components": {
                "value": total_loss,
                "limit": "all four finite and >= 0 W",
                "margin": None,
                "pass": losses_nonnegative,
            },
        }
        if require_full_model:
            checks["full_model"] = {
                "value": full_model_value,
                "limit": 1,
                "margin": None,
                "pass": full_model_value == 1.0 if full_model_value is not None else None,
            }
        states = [item["pass"] for item in checks.values()]
        computed_status = "fail" if False in states else ("pass" if states and all(value is True for value in states) else "unknown")
        return {
            "computed_status": computed_status,
            "checks": checks,
            "Llt_phys_uH": llt,
            "B_max_core_T": bmax,
            "max_temperature_C": max_temperature,
            "temperatures_C": temperatures,
            "min_insulation_mm": min_insulation,
            "total_loss_W": total_loss,
            "volume_L": _finite_number(result.get("volume_L")),
            "conv_error_pct_matrix": matrix_error,
            "conv_error_pct_loss": loss_error,
            "solver_revision": _safe_text(result.get("git_hash"), 40),
            "library_revision": _safe_text(result.get("pyaedt_library_git_hash"), 40),
            "timing_seconds": {
                "matrix": _duration_seconds(result.get("time_matrix")),
                "loss": _duration_seconds(result.get("time_loss")),
                "icepak": _duration_seconds(result.get("time_thermal")),
                "total": _duration_seconds(result.get("time")),
            },
            "parameters": {
                key: _coerce(result.get(key)) for key in DESIGN_PARAMETER_KEYS if result.get(key) not in (None, "")
            },
        }

    def _final_artifact(self) -> tuple[ReadResult, dict[str, Any]]:
        paths = (
            self.root / "verify" / "results" / "final_verification.json",
            self.root / "verify" / "final_verification.json",
            self.root / "monitoring" / "runtime" / "final_verification.json",
        )
        for path in paths:
            if path.exists():
                result = self.cache.json(path, {})
                value = result.value if isinstance(result.value, dict) else {}
                return result, value
        return ReadResult({}, str(paths[0]), False), {}

    def verification(self, nsga: dict[str, Any] | None = None) -> dict[str, Any]:
        nsga = nsga or self.nsga2()
        state_result = self.cache.json(self.root / "al_rounds" / "state.json", {})
        state = state_result.value if isinstance(state_result.value, dict) else {}
        warnings = self._warnings(state_result)
        records = state.get("task_records") if isinstance(state.get("task_records"), dict) else {}
        # Deadline HIT candidates are authenticated top-level publications,
        # not members of the active-learning round, so they intentionally do
        # not carry an AL ``index``.  Only indexed AL candidates participate
        # in this historical task-record join.
        candidates_by_index = {
            item["index"]: item
            for item in nsga.get("candidates", [])
            if isinstance(item, dict)
            and isinstance(item.get("index"), int)
            and not isinstance(item.get("index"), bool)
        }
        standard: list[dict[str, Any]] = []
        for index_text, record_value in records.items():
            if not isinstance(record_value, dict):
                continue
            index = _integer(index_text, -1)
            result = record_value.get("result") if isinstance(record_value.get("result"), dict) else None
            evaluation = self._evaluate_fea(result) if result else None
            predicted = candidates_by_index.get(index)
            standard.append({
                "candidate_id": predicted.get("id") if predicted else f"r{_integer(state.get('round'), 0):02d}-{max(index, 0):04d}",
                "index": index,
                "profile": "standard",
                "task_id": record_value.get("active_id") or record_value.get("original_id"),
                "task_status": _safe_text(record_value.get("last_status"), 30),
                "outcome": _safe_text(record_value.get("outcome"), 50),
                "attempt": _integer(record_value.get("attempt"), 0),
                "evaluation": evaluation,
                "predicted": predicted,
                "error": _safe_text(record_value.get("fetch_error") or record_value.get("error"), 500),
            })

        fine_records = state.get("fine_task_records") \
            if isinstance(state.get("fine_task_records"), dict) else {}
        fine_queue = state.get("final_candidates") \
            if isinstance(state.get("final_candidates"), list) else []
        fine_candidates: list[dict[str, Any]] = []
        for rank_text, record_value in fine_records.items():
            if not isinstance(record_value, dict):
                continue
            rank = _integer(rank_text, -1)
            candidate = fine_queue[rank] if 0 <= rank < len(fine_queue) \
                and isinstance(fine_queue[rank], dict) else {}
            result = record_value.get("result") \
                if isinstance(record_value.get("result"), dict) else None
            fine_candidates.append({
                "rank": rank,
                "candidate_id": _safe_text(candidate.get("candidate_digest"), 100),
                "volume_L": _finite_number(candidate.get("volume_L")),
                "profile": "fine",
                "task_id": record_value.get("active_id") or record_value.get("original_id"),
                "task_status": _safe_text(record_value.get("last_status"), 30),
                "outcome": _safe_text(record_value.get("outcome"), 50),
                "attempt": _integer(record_value.get("attempt"), 0),
                "evaluation": self._evaluate_fea(result, require_full_model=True) if result else None,
                "error": _safe_text(
                    record_value.get("unverified_reason")
                    or record_value.get("fetch_error")
                    or record_value.get("error"), 500,
                ),
            })

        verification_counts = state.get("verification_counts") if isinstance(state.get("verification_counts"), dict) else {}
        if verification_counts:
            counts = {
                "total": _integer(verification_counts.get("total"), len(records)),
                "valid": _integer(verification_counts.get("valid"), 0),
                "pending": _integer(verification_counts.get("pending"), 0),
                "exhausted": _integer(verification_counts.get("exhausted"), 0),
                "ingested": _integer(verification_counts.get("ingested"), 0),
            }
        else:
            counts = {
                "total": len(records),
                "valid": sum(item.get("outcome") == "valid" for item in records.values() if isinstance(item, dict)),
                "pending": sum(item.get("outcome") in {None, "pending", "fetch_error", "submission_unknown"} for item in records.values() if isinstance(item, dict)),
                "exhausted": sum(item.get("outcome") == "exhausted" for item in records.values() if isinstance(item, dict)),
                "ingested": 0,
            }
        counts["coverage"] = counts["valid"] / counts["total"] if counts["total"] else None

        error_files = sorted((self.root / "al_rounds").glob("round_*/verification_errors.csv"))
        errors: list[dict[str, Any]] = []
        error_source = None
        if error_files:
            error_result = self.cache.csv(error_files[-1], max_rows=10_000)
            error_source = str(error_files[-1])
            errors = [
                {key: _coerce(value) for key, value in row.items()}
                for row in (error_result.value if isinstance(error_result.value, list) else [])
            ]
            if error_result.warning:
                warnings.append(error_result.warning)

        final_result, final_payload = self._final_artifact()
        if final_result.warning:
            warnings.append(final_result.warning)
        raw_final_result = final_payload.get("result") if isinstance(final_payload.get("result"), dict) else final_payload
        final_evaluation = self._evaluate_fea(raw_final_result, require_full_model=True) if final_payload else None
        declared = _safe_text(final_payload.get("status"), 30) if final_payload else None
        declared_pass = final_payload.get("passed", final_payload.get("overall_pass")) if final_payload else None
        declared_success = declared_pass is True or (
            declared and declared.lower() in {"pass", "passed", "complete", "completed"}
        )
        declared_failure = declared_pass is False or (
            declared and declared.lower() in {"fail", "failed", "error"}
        )
        # Never let a manually declared PASS override a physical check.  A
        # partial result remains unknown until every required value is present.
        if declared_failure or (final_evaluation and final_evaluation["computed_status"] == "fail"):
            final_status = "fail"
        elif final_evaluation and final_evaluation["computed_status"] == "pass":
            final_status = "pass"
        elif declared_success or final_payload:
            final_status = "unknown"
        elif state.get("stage") == "FINE_BLOCKED":
            final_status = "blocked"
        else:
            final_status = "waiting"
        final = {
            "available": bool(final_payload),
            "status": final_status,
            "candidate_id": _safe_text(final_payload.get("candidate_id"), 100) if final_payload else None,
            "profile": _safe_text(final_payload.get("profile"), 30) or ("fine" if final_payload else None),
            "task_id": (
                final_payload.get("fine_task_id") or final_payload.get("task_id")
            ) if final_payload else None,
            "task_status": _safe_text(
                final_payload.get("fine_task_status") or final_payload.get("task_status"), 30
            ) if final_payload else None,
            "evaluation": final_evaluation,
            "declared_status": declared,
            "error": _safe_text(
                (final_payload.get("error") or final_payload.get("failure_reason"))
                if final_payload else state.get("fine_block_reason"), 1000,
            ),
            "updated_at": _safe_text(
                final_payload.get("generated_at") or final_payload.get("updated_at")
                or final_payload.get("time"), 80,
            ) if final_payload else None,
            "source": final_result.path if final_result.exists else None,
        }

        history = state.get("history") if isinstance(state.get("history"), list) else []
        agreement = history[-1] if history and isinstance(history[-1], dict) else None
        return {
            "schema_version": SCHEMA_VERSION,
            "available": bool(state or standard or errors or final_payload),
            "stage": _safe_text(state.get("stage"), 30) or "NOT_STARTED",
            "round": _integer(state.get("round"), 0) if state else None,
            "counts": counts,
            "standard_candidates": standard,
            "fine_candidates": fine_candidates,
            "verification_errors": errors,
            "agreement": agreement,
            "final": final,
            "sources": {"state": state_result.path if state_result.exists else None, "errors": error_source},
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _blocker_hpo_v2_diagnostics(self) -> dict[str, Any]:
        """Read the explicit HPO v2 heartbeat instead of scanning processes."""
        base = {
            "available": False,
            "validated_running": False,
            "state": "unavailable",
            "source": (
                str(self._blocker_hpo_v2_status_path)
                if self._blocker_hpo_v2_status_path is not None else None
            ),
            "warnings": [],
        }
        path = self._blocker_hpo_v2_status_path
        if path is None:
            return base
        try:
            value, _ = _bounded_json_bytes(
                path, BLOCKER_HPO_V2_STATUS_MAX_BYTES
            )
            if value.get("schema_version") != BLOCKER_HPO_V2_STATUS_SCHEMA:
                raise ValueError("blocker HPO v2 status schema is invalid")
            for field in (
                "config_sha256",
                "dataset_sha256",
            ):
                if not re.fullmatch(
                    r"[0-9a-f]{64}", str(value.get(field) or "").lower()
                ):
                    raise ValueError(f"blocker HPO v2 {field} is malformed")
            if any(
                value.get(flag) is not False
                for flag in (
                    "production_eligible",
                    "production_model_eligible",
                    "fea_submission_approved",
                    "promotion_approved",
                )
            ):
                raise ValueError("blocker HPO v2 status asserted promotion authority")
            phase = _safe_text(value.get("phase"), 80) or "unknown"
            if phase not in {
                "starting", "preflight_complete", "optimizing",
                "succeeded", "failed",
            }:
                raise ValueError("blocker HPO v2 phase is invalid")
            stage = _integer(
                value.get("selected_cumulative_trials_per_job"), -1
            )
            jobs = value.get("jobs")
            trials = value.get("trials")
            if stage <= 0 or not isinstance(jobs, dict) or not jobs:
                raise ValueError("blocker HPO v2 job contract is invalid")
            if not isinstance(trials, dict):
                raise ValueError("blocker HPO v2 trial counts are missing")
            sums = Counter()
            complete_jobs = 0
            for key, job in jobs.items():
                if not isinstance(key, str) or not isinstance(job, dict):
                    raise ValueError("blocker HPO v2 job row is invalid")
                if _integer(job.get("total"), -1) != stage:
                    raise ValueError("blocker HPO v2 per-job trial budget drifted")
                job_observed = 0
                for name in ("complete", "running", "failed"):
                    count = _integer(job.get(name), -1)
                    if count < 0:
                        raise ValueError("blocker HPO v2 job count is invalid")
                    sums[name] += count
                    job_observed += count
                if job_observed > stage:
                    raise ValueError(
                        "blocker HPO v2 job counts exceed the stage budget"
                    )
                if sums["complete"] + sums["running"] + sums["failed"] > (
                    len(jobs) * stage
                ):
                    raise ValueError("blocker HPO v2 counts exceed the stage budget")
                if job.get("status") == "complete":
                    complete_jobs += 1
            expected_total = len(jobs) * stage
            if _integer(trials.get("total"), -1) != expected_total or any(
                _integer(trials.get(name), -1) != sums[name]
                for name in ("complete", "running", "failed")
            ):
                raise ValueError("blocker HPO v2 aggregate trial counts disagree")
            heartbeat = _parse_time(value.get("heartbeat_at"), self.clock().tzinfo)
            if heartbeat is None:
                raise ValueError("blocker HPO v2 heartbeat is invalid")
            age = max(0.0, (self.clock() - heartbeat).total_seconds())
            # A preflight-only invocation intentionally exits after sealing a
            # ``preflight_complete`` status.  It is durable evidence, not a
            # stale worker.  Only phases that require a living process are
            # subject to the heartbeat-running gate.
            live_phase = phase in {"starting", "optimizing"}
            fresh = age <= BLOCKER_HPO_V2_HEARTBEAT_MAX_AGE_SECONDS
            warnings = []
            if live_phase and not fresh:
                warnings.append(
                    "blocker HPO v2 heartbeat is stale; running state was rejected"
                )
            preflight = value.get("preflight")
            preflight = preflight if isinstance(preflight, dict) else {}
            error = value.get("error")
            error = error if isinstance(error, dict) else {}
            return {
                **base,
                "available": True,
                "validated_running": live_phase and fresh,
                "state": phase,
                "wave_phase": "experimental_hpo",
                "selected_hpo_target_count": len(jobs),
                "completed_hpo_target_count": complete_jobs,
                "active_hpo_processes": sums["running"],
                "active_hpo_batch": stage,
                "hpo_batch_count": stage,
                "observed_strict_full_rows": _integer(
                    preflight.get("strict_full_rows"), 0
                ),
                "trial_total": expected_total,
                "trial_complete": sums["complete"],
                "trial_running": sums["running"],
                "trial_failed": sums["failed"],
                "job_count": len(jobs),
                "stage_trials_per_job": stage,
                "age_seconds": round(age, 3),
                "heartbeat_at": _safe_text(value.get("heartbeat_at"), 100),
                "started_at": _safe_text(value.get("started_at"), 100),
                "activity_at": _safe_text(value.get("activity_at"), 100),
                "config_sha256": str(value["config_sha256"]).lower(),
                "dataset_sha256": str(value["dataset_sha256"]).lower(),
                "pid": _integer(value.get("pid"), -1),
                "host": _safe_text(value.get("host"), 200),
                "result": value.get("result")
                if isinstance(value.get("result"), dict) else None,
                "error": _safe_text(error.get("message"), 1_000),
                "warnings": warnings,
            }
        except (OSError, TypeError, UnicodeError, ValueError) as exc:
            return {
                **base,
                "available": True,
                "state": "invalid",
                "warnings": [f"blocker HPO v2 status rejected: {exc}"],
            }

    def _status(
        self,
        data: dict[str, Any],
        models: dict[str, Any],
        nsga: dict[str, Any],
        verification: dict[str, Any],
        scheduler: dict[str, Any],
        continuous_pipeline: dict[str, Any],
    ) -> dict[str, Any]:
        stages = []
        simulation_active = scheduler.get("running", 0) + scheduler.get("pending", 0) > 0
        if simulation_active:
            simulation_state = "active"
            simulation_detail = f"실행 {scheduler.get('running', 0)} · 대기 {scheduler.get('pending', 0)}"
        elif not scheduler.get("connected"):
            simulation_state = "warning"
            simulation_detail = "스케줄러 상태 확인 불가"
        else:
            simulation_state = "waiting"
            simulation_detail = "실행 중 작업 없음"
        stages.append({"key": "simulation", "label": "시뮬레이션", "state": simulation_state, "detail": simulation_detail})

        if data["total_rows"] >= DATA_GOAL:
            data_state, data_detail = "complete", f"목표 달성 · {data['total_rows']:,}개"
        elif data["throughput_1h"] > 0:
            data_state, data_detail = "active", f"최근 1시간 +{data['throughput_1h']}개"
        elif data["stalled"]:
            data_state, data_detail = "warning", "90분 이상 데이터 증가 없음"
        else:
            data_state, data_detail = "waiting", f"{data['total_rows']:,}개 확보"
        stages.append({"key": "data", "label": "데이터 적재", "state": data_state, "detail": data_detail})

        experimental_training = continuous_pipeline.get(
            "experimental_shadow_training"
        )
        experimental_training = (
            experimental_training
            if isinstance(experimental_training, dict)
            else {}
        )
        experimental_training_active = bool(
            experimental_training.get("validated_running") is True
        )
        if experimental_training_active:
            rows = experimental_training.get("observed_strict_full_rows")
            selected = experimental_training.get("selected_hpo_target_count")
            completed = experimental_training.get("completed_hpo_target_count")
            phase = experimental_training.get("wave_phase")
            phase_label = {
                "experimental_hpo": "후보 HPO 학습 중",
                "candidate_training": "후보 모델 학습 중",
                "paired_evaluation": "후보 품질 평가 중",
                "aggregate_temperature_safety_rejected": "후보 보완 학습 중",
            }.get(phase, "후보 모델 작업 중")
            details = [phase_label]
            if isinstance(rows, int) and not isinstance(rows, bool):
                details.append(f"{rows:,}행")
            if (
                isinstance(selected, int)
                and not isinstance(selected, bool)
                and selected > 0
            ):
                if isinstance(completed, int) and not isinstance(completed, bool):
                    details.append(f"{completed}/{selected} targets")
                else:
                    details.append(f"{selected} targets")
            details.append("운영 미승인")
            model_state, model_detail = "active", " · ".join(details)
        elif models.get("activation_state") == "preactivation_checkpoint":
            model_state = "waiting"
            model_detail = (
                f"checkpoint {models.get('latest_checkpoint')} CV "
                f"{models.get('evaluated_count', 0)}/{models.get('target_count', 0)} · "
                f"활성화 {models.get('current_data_count', 0):,}/"
                f"{models.get('activation_minimum_strict_full_rows', 0):,}"
            )
        elif models["trained_count"] == 0:
            model_state, model_detail = "waiting", "학습 모델 없음"
        elif models["missing_count"]:
            model_state, model_detail = "warning", f"{models['trained_count']}/{models['target_count']} 모델 학습"
        else:
            model_state, model_detail = "complete", f"{models['trained_count']}개 모델 준비"
        stages.append({"key": "models", "label": "모델 학습", "state": model_state, "detail": model_detail})

        nsga_state = "active" if nsga["status"] == "running" else ("complete" if nsga["available"] else "waiting")
        if nsga["available"] and nsga.get("al_stage") == "CONTINUOUS":
            nsga_detail = (
                "continuous · "
                f"{max(0, _integer(nsga.get('active_seed_workers'), 0))} workers · "
                f"{max(0, _integer(nsga.get('current_model_completed_runs'), 0))} runs · "
                f"valid {max(0, _integer(nsga.get('valid_candidate_count'), 0))}/"
                f"{max(0, _integer(nsga.get('candidate_count'), 0))}"
            )
        else:
            nsga_detail = (
                f"round {nsga.get('round')} · {nsga['candidate_count']}개"
                if nsga["available"] else "실행 전"
            )
        stages.append({"key": "nsga2", "label": "NSGA-II", "state": nsga_state, "detail": nsga_detail})

        verification_stage = str(verification.get("stage", "NOT_STARTED")).upper()
        if verification["counts"]["total"]:
            verify_state = "active" if verification["counts"]["pending"] else "complete"
            verify_detail = f"유효 {verification['counts']['valid']}/{verification['counts']['total']}"
        elif verification_stage in {"SUBMIT", "WAIT", "INGEST", "CHECK"}:
            verify_state, verify_detail = "active", verification_stage
        else:
            verify_state, verify_detail = "waiting", "검증 전"
        stages.append({"key": "verification", "label": "후보 FEA", "state": verify_state, "detail": verify_detail})

        final_status = verification["final"]["status"]
        final_state = {"pass": "complete", "fail": "error"}.get(final_status, "waiting")
        final_detail = {"pass": "최종 설계 확정", "fail": "fine FEA 실패"}.get(final_status, "검증 전")
        stages.append({"key": "final", "label": "최종 설계", "state": final_state, "detail": final_detail})

        warnings: list[str] = []
        for payload in (data, models, nsga, verification):
            warnings.extend(payload.get("warnings", []))
        if not scheduler.get("connected") and scheduler.get("error"):
            warnings.append(scheduler["error"])
        if data["stalled"]:
            warnings.append(f"유효 데이터가 약 {data['stalled_minutes']:.0f}분 동안 증가하지 않았습니다.")
        if (
            models["missing_count"]
            and models.get("activation_state") != "preactivation_checkpoint"
        ):
            missing = [model["label"] for model in models["models"] if not model["trained"]]
            warning_label = (
                "운영 승인 모델 미등록"
                if experimental_training_active
                else "미학습 모델"
            )
            warnings.append(warning_label + ": " + ", ".join(missing))
        if final_status == "fail":
            warnings.append("최종 fine FEA가 사양을 통과하지 못했습니다.")
        warnings = list(dict.fromkeys(warnings))

        if final_status == "fail" or any(stage["state"] == "error" for stage in stages):
            overall = "error"
        elif warnings:
            overall = "warning"
        elif any(stage["state"] == "active" for stage in stages):
            overall = "active"
        else:
            overall = "idle"
        current = next((stage for stage in reversed(stages) if stage["state"] == "active"), None)
        if current is None:
            current = next((stage for stage in stages if stage["state"] in {"warning", "error"}), stages[0])
        return {
            "overall": overall,
            "current_stage": current["key"],
            "current_stage_label": current["label"],
            "stages": stages,
            "warnings": warnings,
        }

    def dashboard(self, record: bool = True) -> dict[str, Any]:
        generated_at = _iso(self.clock())
        data = self.data()
        training_cohort = data.get("training_cohort")
        model_data_count = (
            training_cohort.get("strict_full_rows")
            if isinstance(training_cohort, dict)
            and training_cohort.get("available") is True
            else data["total_rows"]
        )
        models = self.models(model_data_count)
        nsga = self.nsga2()
        verification = self.verification(nsga)
        scheduler = self.scheduler.snapshot()
        refill_controller = self.refill_controller.snapshot()
        continuous_pipeline = self.continuous_pipeline.snapshot()
        continuous_pipeline = (
            dict(continuous_pipeline)
            if isinstance(continuous_pipeline, dict) else {}
        )
        blocker_hpo_v2 = self._blocker_hpo_v2_diagnostics()
        continuous_pipeline["blocker_hpo_v2"] = blocker_hpo_v2
        existing_hpo = continuous_pipeline.get("experimental_shadow_training")
        existing_hpo = existing_hpo if isinstance(existing_hpo, dict) else {}
        if blocker_hpo_v2.get("available") is True and not (
            existing_hpo.get("validated_running") is True
            and blocker_hpo_v2.get("validated_running") is not True
        ):
            continuous_pipeline["experimental_shadow_training"] = dict(
                blocker_hpo_v2
            )
        hpo_warnings = blocker_hpo_v2.get("warnings")
        if isinstance(hpo_warnings, list) and hpo_warnings:
            continuous_pipeline["warnings"] = list(dict.fromkeys([
                *(continuous_pipeline.get("warnings") or []),
                *hpo_warnings,
            ]))
        status = self._status(
            data,
            models,
            nsga,
            verification,
            scheduler,
            continuous_pipeline,
        )
        pipeline_warnings = continuous_pipeline.get("warnings", [])
        if pipeline_warnings:
            status["warnings"] = list(dict.fromkeys([
                *status["warnings"], *pipeline_warnings,
            ]))
            if status["overall"] not in {"error"}:
                status["overall"] = "warning"
        dashboard = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "project": "MFT_1MW_2026",
            "status": status,
            "data": data,
            "models": models,
            "nsga2": nsga,
            "verification": verification,
            "scheduler": scheduler,
            "refill_controller": refill_controller,
            "continuous_pipeline": continuous_pipeline,
        }
        if record and self.recorder:
            try:
                self.recorder.record(dashboard)
            except (OSError, TypeError, ValueError) as exc:
                dashboard["status"]["warnings"].append(f"모니터 이력 기록 실패: {exc}")
                if dashboard["status"]["overall"] not in {"error"}:
                    dashboard["status"]["overall"] = "warning"
        return dashboard

    def status(self) -> dict[str, Any]:
        dashboard = self.dashboard(record=False)
        return {
            "schema_version": dashboard["schema_version"],
            "generated_at": dashboard["generated_at"],
            "project": dashboard["project"],
            **dashboard["status"],
            "scheduler": dashboard["scheduler"],
            "refill_controller": dashboard["refill_controller"],
            "continuous_pipeline": dashboard["continuous_pipeline"],
        }

    def history(self) -> dict[str, Any]:
        if not self.recorder:
            return {"entries": [], "warning": None}
        return self.recorder.history()
