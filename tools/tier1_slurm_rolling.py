"""Continuously refill a normalized, warm Tier-1 NSGA-II Slurm sidecar.

The controller maintains *concurrent* queued+running inventory, not a finite
campaign total.  It never starts AEDT, submits FEA, or promotes a model.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import shlex
import socket
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import uuid

import numpy as np

try:
    from tier1_n1_6_anchor_island_contract import (
        ALL_THERMAL_SCALE_C as ANCHOR_ALL_THERMAL_SCALE_C,
        ANCHOR_ISLANDS,
        BY_VARIANT as ANCHOR_ISLAND_BY_VARIANT,
        LLT_SCALE_UH as ANCHOR_LLT_SCALE_UH,
        RESONANCE_SCALE_HZ as ANCHOR_RESONANCE_SCALE_HZ,
        island_profile as anchor_island_profile,
        optimizer_allowance_contract,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_n1_6_anchor_island_contract import (
        ALL_THERMAL_SCALE_C as ANCHOR_ALL_THERMAL_SCALE_C,
        ANCHOR_ISLANDS,
        BY_VARIANT as ANCHOR_ISLAND_BY_VARIANT,
        LLT_SCALE_UH as ANCHOR_LLT_SCALE_UH,
        RESONANCE_SCALE_HZ as ANCHOR_RESONANCE_SCALE_HZ,
        island_profile as anchor_island_profile,
        optimizer_allowance_contract,
    )

try:
    from tier1_deep_crossover_contract import (
        BY_VARIANT as DEEP_CROSSOVER_BY_VARIANT,
        DEEP_CROSSOVER_ISLANDS,
        FIXED_GENERATIONS as DEEP_FIXED_GENERATIONS,
        INFERENCE_THREADS as DEEP_INFERENCE_THREADS,
        LEGACY_TASK_PRIORITY as DEEP_LEGACY_TASK_PRIORITY,
        POPULATION as DEEP_POPULATION,
        RESONANCE_SCALE_HZ as DEEP_RESONANCE_SCALE_HZ,
        TASK_PRIORITY as DEEP_TASK_PRIORITY,
        TERMINATION_STRATEGY as DEEP_TERMINATION_STRATEGY,
        island_profile as deep_crossover_island_profile,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_deep_crossover_contract import (
        BY_VARIANT as DEEP_CROSSOVER_BY_VARIANT,
        DEEP_CROSSOVER_ISLANDS,
        FIXED_GENERATIONS as DEEP_FIXED_GENERATIONS,
        INFERENCE_THREADS as DEEP_INFERENCE_THREADS,
        LEGACY_TASK_PRIORITY as DEEP_LEGACY_TASK_PRIORITY,
        POPULATION as DEEP_POPULATION,
        RESONANCE_SCALE_HZ as DEEP_RESONANCE_SCALE_HZ,
        TASK_PRIORITY as DEEP_TASK_PRIORITY,
        TERMINATION_STRATEGY as DEEP_TERMINATION_STRATEGY,
        island_profile as deep_crossover_island_profile,
    )


REPO = Path(__file__).resolve().parents[1]
SEARCH_VARIANT = os.environ.get(
    "MFT_TIER1_SEARCH_VARIANT", "normalized-warm-v1"
)
FIXED_PRIMARY_TURNS = None
SEARCH_FOCUS = None
OPTIMIZER_LLT_SCALE_UH = None
OPTIMIZER_ALL_THERMAL_SCALE_C = None
OPTIMIZER_RESONANCE_ALLOWANCE_HZ = None
OPTIMIZER_LLT_ALLOWANCE_UH = None
ANCHOR_ISLAND = None
DEEP_CROSSOVER_ISLAND = None
OPTIMIZER_TERMINATION_STRATEGY = "default-ftol-period30-v1"
if SEARCH_VARIANT == "normalized-warm-v1":
    RUNTIME = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_normalized_t110_res15k_260719"
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-00ca4bcedee0\followup-normalized"
        r"\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = "normalized-p320-g600-warm64-v1"
    SEED_START = 1_907_191_000
    TASK_PRIORITY = -8
    LEGACY_TASK_PRIORITY = -9
    OPTIMIZER_RESONANCE_SCALE_HZ = 1_000.0
    OPTIMIZER_CORE_THERMAL_SCALE_C = None
    TASK_NAME_STEM = "mft-t1norm"
    DEDUPE_NAMESPACE = "mft-tier1-normalized-nsga"
elif SEARCH_VARIANT == "resonance-focus-250hz-v1":
    RUNTIME = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\t1r250_260719"
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-resonance-focus-50x50-v1\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = "resonance-focus250-p320-g600-warm64-v1"
    SEED_START = 1_907_192_000
    TASK_PRIORITY = -7
    LEGACY_TASK_PRIORITY = -8
    OPTIMIZER_RESONANCE_SCALE_HZ = 250.0
    OPTIMIZER_CORE_THERMAL_SCALE_C = None
    TASK_NAME_STEM = "mft-t1res250"
    DEDUPE_NAMESPACE = "mft-tier1-resonance-focus250-nsga"
elif SEARCH_VARIANT == "thermal-crossover-core1c-v1":
    RUNTIME = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_thermal_crossover_t110_res15k_260719"
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-thermal-crossover-balanced-v1"
        r"\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = (
        "thermal-crossover-core1c-res250-p320-g600-warm64-v1"
    )
    SEED_START = 1_907_193_000
    TASK_PRIORITY = -6
    LEGACY_TASK_PRIORITY = -7
    OPTIMIZER_RESONANCE_SCALE_HZ = 250.0
    OPTIMIZER_CORE_THERMAL_SCALE_C = 1.0
    TASK_NAME_STEM = "mft-t1therm1"
    DEDUPE_NAMESPACE = "mft-tier1-thermal-crossover-core1c-nsga"
elif SEARCH_VARIANT == "thermal-crossover-balanced4c-v1":
    RUNTIME = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_thermal_balanced4c_t110_res15k_260719"
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-thermal-crossover-balanced-v1"
        r"\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = (
        "thermal-crossover-balanced4c-res250-p320-g600-warm64-v1"
    )
    SEED_START = 1_907_194_000
    TASK_PRIORITY = -5
    LEGACY_TASK_PRIORITY = -6
    OPTIMIZER_RESONANCE_SCALE_HZ = 250.0
    OPTIMIZER_CORE_THERMAL_SCALE_C = 4.0
    TASK_NAME_STEM = "mft-t1therm4"
    DEDUPE_NAMESPACE = (
        "mft-tier1-thermal-crossover-balanced4c-nsga"
    )
elif SEARCH_VARIANT == "fixed-n1-5-thermal-llt-v1":
    RUNTIME = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_n1_5_thermal_llt_t110_res15k_260719"
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-thermal-crossover-balanced-v1"
        r"\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = (
        "fixed-n1-5-thermal-llt-core1c-res250-p320-g600-warm64-v1"
    )
    SEED_START = 1_907_195_000
    TASK_PRIORITY = -4
    LEGACY_TASK_PRIORITY = -5
    OPTIMIZER_RESONANCE_SCALE_HZ = 250.0
    OPTIMIZER_CORE_THERMAL_SCALE_C = 1.0
    FIXED_PRIMARY_TURNS = 5
    SEARCH_FOCUS = "thermal_then_llt_with_resonance_preserved"
    TASK_NAME_STEM = "mft-t1n5therm"
    DEDUPE_NAMESPACE = "mft-tier1-fixed-n1-5-thermal-llt-nsga"
elif SEARCH_VARIANT == "fixed-n1-6-resonance-llt-v1":
    RUNTIME = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_n1_6_resonance_llt_t110_res15k_260719"
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-thermal-crossover-balanced-v1"
        r"\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = (
        "fixed-n1-6-resonance-llt-core4c-res100-p320-g600-warm64-v1"
    )
    SEED_START = 1_907_196_000
    TASK_PRIORITY = -3
    LEGACY_TASK_PRIORITY = -4
    OPTIMIZER_RESONANCE_SCALE_HZ = 100.0
    OPTIMIZER_CORE_THERMAL_SCALE_C = 4.0
    FIXED_PRIMARY_TURNS = 6
    SEARCH_FOCUS = "resonance_then_llt_with_thermal_preserved"
    TASK_NAME_STEM = "mft-t1n6res"
    DEDUPE_NAMESPACE = "mft-tier1-fixed-n1-6-resonance-llt-nsga"
elif SEARCH_VARIANT == "fixed-n1-6-thermal-bridge-v1":
    RUNTIME = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_n1_6_thermal_bridge_t110_res15k_260719"
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-n1-6-thermal-bridge-v1"
        r"\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = (
        "fixed-n1-6-thermal-bridge-core2c-llt0p55-res150-"
        "p320-g600-warm64-v1"
    )
    SEED_START = 1_907_197_000
    TASK_PRIORITY = -2
    LEGACY_TASK_PRIORITY = -3
    OPTIMIZER_RESONANCE_SCALE_HZ = 150.0
    OPTIMIZER_CORE_THERMAL_SCALE_C = 2.0
    FIXED_PRIMARY_TURNS = 6
    SEARCH_FOCUS = "low_thermal_resonance_pass_llt_bridge"
    TASK_NAME_STEM = "mft-t1n6bridge"
    DEDUPE_NAMESPACE = "mft-tier1-fixed-n1-6-thermal-bridge-nsga"
elif SEARCH_VARIANT == "fixed-n1-6-orthogonal-bridge-v1":
    RUNTIME = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_n1_6_orthogonal_bridge_t110_res15k_260719"
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-n1-6-orthogonal-bridge-v1"
        r"\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = (
        "fixed-n1-6-orthogonal-allthermal2c-llt0p25-res150-"
        "p320-g600-warm64-v1"
    )
    SEED_START = 1_907_198_000
    TASK_PRIORITY = -1
    LEGACY_TASK_PRIORITY = -2
    OPTIMIZER_RESONANCE_SCALE_HZ = 150.0
    OPTIMIZER_CORE_THERMAL_SCALE_C = None
    OPTIMIZER_LLT_SCALE_UH = 0.25
    OPTIMIZER_ALL_THERMAL_SCALE_C = 2.0
    FIXED_PRIMARY_TURNS = 6
    SEARCH_FOCUS = "orthogonal_thermal_llt_resonance_bridge"
    TASK_NAME_STEM = "mft-t1n6ortho"
    DEDUPE_NAMESPACE = "mft-tier1-fixed-n1-6-orthogonal-bridge-nsga"
elif SEARCH_VARIANT in ANCHOR_ISLAND_BY_VARIANT:
    ANCHOR_ISLAND = ANCHOR_ISLAND_BY_VARIANT[SEARCH_VARIANT]
    RUNTIME = (
        Path(r"C:\Users\peets\slurm_scheduler_runtime")
        / ANCHOR_ISLAND.runtime_suffix
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-n1-6-anchor-islands-v1\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = ANCHOR_ISLAND.namespace
    SEED_START = ANCHOR_ISLAND.seed_start
    TASK_PRIORITY = -1
    LEGACY_TASK_PRIORITY = -2
    OPTIMIZER_RESONANCE_SCALE_HZ = ANCHOR_RESONANCE_SCALE_HZ
    OPTIMIZER_CORE_THERMAL_SCALE_C = None
    OPTIMIZER_LLT_SCALE_UH = ANCHOR_LLT_SCALE_UH
    OPTIMIZER_ALL_THERMAL_SCALE_C = ANCHOR_ALL_THERMAL_SCALE_C
    OPTIMIZER_RESONANCE_ALLOWANCE_HZ = (
        ANCHOR_ISLAND.resonance_allowance_hz
    )
    OPTIMIZER_LLT_ALLOWANCE_UH = ANCHOR_ISLAND.llt_allowance_uh
    FIXED_PRIMARY_TURNS = 6
    SEARCH_FOCUS = ANCHOR_ISLAND.search_focus
    TASK_NAME_STEM = ANCHOR_ISLAND.task_name_stem
    DEDUPE_NAMESPACE = ANCHOR_ISLAND.dedupe_namespace
elif SEARCH_VARIANT in DEEP_CROSSOVER_BY_VARIANT:
    DEEP_CROSSOVER_ISLAND = DEEP_CROSSOVER_BY_VARIANT[SEARCH_VARIANT]
    RUNTIME = (
        Path(r"C:\Users\peets\slurm_scheduler_runtime")
        / DEEP_CROSSOVER_ISLAND.runtime_suffix
    )
    if DEEP_CROSSOVER_ISLAND.fixed_primary_turns == 5:
        DEFAULT_WARM = Path(
            r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
            r"\t110-res15k-n1-5-deep-crossover-32x32-v3"
            r"\next_warm_start.npy"
        )
    else:
        DEFAULT_WARM = Path(
            r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
            r"\t110-res15k-n1-6-anchor-islands-v1"
            r"\next_warm_start.npy"
        )
    SEARCH_NAMESPACE = DEEP_CROSSOVER_ISLAND.namespace
    SEED_START = DEEP_CROSSOVER_ISLAND.seed_start
    TASK_PRIORITY = DEEP_TASK_PRIORITY
    LEGACY_TASK_PRIORITY = DEEP_LEGACY_TASK_PRIORITY
    OPTIMIZER_RESONANCE_SCALE_HZ = DEEP_RESONANCE_SCALE_HZ
    OPTIMIZER_CORE_THERMAL_SCALE_C = None
    OPTIMIZER_LLT_SCALE_UH = (
        DEEP_CROSSOVER_ISLAND.optimizer_llt_scale_uh
    )
    OPTIMIZER_ALL_THERMAL_SCALE_C = (
        DEEP_CROSSOVER_ISLAND.optimizer_all_thermal_scale_c
    )
    OPTIMIZER_RESONANCE_ALLOWANCE_HZ = (
        DEEP_CROSSOVER_ISLAND.optimizer_resonance_allowance_hz
    )
    OPTIMIZER_LLT_ALLOWANCE_UH = (
        DEEP_CROSSOVER_ISLAND.optimizer_llt_allowance_uh
    )
    FIXED_PRIMARY_TURNS = DEEP_CROSSOVER_ISLAND.fixed_primary_turns
    SEARCH_FOCUS = DEEP_CROSSOVER_ISLAND.search_focus
    TASK_NAME_STEM = DEEP_CROSSOVER_ISLAND.task_name_stem
    DEDUPE_NAMESPACE = DEEP_CROSSOVER_ISLAND.dedupe_namespace
    OPTIMIZER_TERMINATION_STRATEGY = DEEP_TERMINATION_STRATEGY
elif SEARCH_VARIANT == "fixed-n1-5-size-rx-core-v1":
    RUNTIME = Path(
        r"C:\Users\peets\slurm_scheduler_runtime"
        r"\mft_tier1_nsga_n1_5_size_rx_core_t110_res15k_260719"
    )
    DEFAULT_WARM = Path(
        r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_warm_handoffs"
        r"\t110-res15k-n1-5-size-rx-core-island-v1"
        r"\next_warm_start.npy"
    )
    SEARCH_NAMESPACE = (
        "fixed-n1-5-size-rx-core1c-llt0p55-res150-soft1000-"
        "p320-g600-warm64-v1"
    )
    SEED_START = 1_907_199_000
    TASK_PRIORITY = -1
    LEGACY_TASK_PRIORITY = -2
    OPTIMIZER_RESONANCE_SCALE_HZ = 150.0
    OPTIMIZER_CORE_THERMAL_SCALE_C = 1.0
    FIXED_PRIMARY_TURNS = 5
    SEARCH_FOCUS = (
        "size_rx_resonance_llt_preserved_plus_core_temperature_bridge"
    )
    TASK_NAME_STEM = "mft-t1n5src"
    DEDUPE_NAMESPACE = "mft-tier1-fixed-n1-5-size-rx-core-nsga"
else:
    raise RuntimeError(f"unsupported Tier-1 search variant: {SEARCH_VARIANT}")
LIVE_RUNTIME = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_slurm_rolling"
)
CANONICAL = RUNTIME / "canonical"
POINTER = CANONICAL / "model_pointer.json"
INDEX = CANONICAL / "index.json"
CONTROLLER_STATUS = CANONICAL / "controller_status.json"
STATE = RUNTIME / "controller" / "state.json"
LOCK = RUNTIME / "controller" / "owner.lock"
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")
DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_BASE_PLAN = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_nsga_slurm_offload"
    r"\core4-8ac5c0b95b3c6783e793\offload_plan.json"
)
DEFAULT_MODEL = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_feedback_260718"
    r"\32c4c67-926d8f9-v6-64\model65-active-learning-v2"
)
DEFAULT_CODE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_code\7c832f7f78f9"
)
REMOTE_ROOT = "/gpfs/tmp_cpu2/mft_tier1_nsga_rolling"
ROLLING_TARGET = (
    ANCHOR_ISLAND.active_quota
    if ANCHOR_ISLAND is not None
    else (
        DEEP_CROSSOVER_ISLAND.active_quota
        if DEEP_CROSSOVER_ISLAND is not None else 32
    )
)
POPULATION = (
    DEEP_POPULATION if DEEP_CROSSOVER_ISLAND is not None else 320
)
MAX_GENERATIONS = (
    DEEP_FIXED_GENERATIONS if DEEP_CROSSOVER_ISLAND is not None else 600
)
INFERENCE_THREADS = (
    DEEP_INFERENCE_THREADS if DEEP_CROSSOVER_ISLAND is not None else 8
)
# This is a fresh namespace, so every task starts directly at TASK_PRIORITY.
PRIORITY_MIGRATION_SEED_CUTOFF = SEED_START
POINTER_SCHEMA = "mft-tier1-slurm-model-pointer-v1"
INDEX_SCHEMA = "mft-tier1-slurm-rolling-index-v1"
STATUS_SCHEMA = "mft-tier1-slurm-rolling-status-v1"
DEPLOYMENT_SCHEMA = "mft-tier1-slurm-deployment-v1"
LEGACY_TASK_SCHEMA = "mft-tier1-slurm-seed-task-v1"
TASK_SCHEMA = "mft-tier1-slurm-seed-task-v3"
ATTESTATION_SCHEMA = "mft-tier1-slurm-temperature-attestation-v3"
TEMPERATURE_CONTRACT_SCHEMA = "mft-tier1-temperature-constraint-contract-v3"
ACTIVE_STATES = {"queued", "running", "attaching", "starting", "launching"}
TERMINAL_STATES = {"completed", "failed", "cancelled", "canceled"}
# A healthy live loop was observed taking up to 24 seconds with a 10-second
# poll interval plus bounded harvest/refill work.  Sixty seconds is greater
# than two observed worst-case cycles, but still makes a stopped/erroring
# publisher stale promptly (and remains below the reader's 300-second ceiling).
FRESHNESS_DEADLINE_SECONDS = 60
HEALTH_CONTRACT_VERSION = "mft-tier1-controller-inventory-health-v2"
# Emergency compatibility ceiling for the strict progress reader.  The reader
# authenticates task/state/seed identity against the declared scheduler total;
# truncating a 1,001+ task cohort to the former 1,000-row window made an intact
# inventory appear inconsistent.  A bounded digest protocol will supersede
# this full-row compatibility window before an unbounded cohort reaches 10k.
CANONICAL_TASK_WINDOW_LIMIT = 10_000
TERMINAL_RESULT_WINDOW_LIMIT = 1_000
HARVEST_MAX_NEW_PER_TICK = 16
TEMPERATURE_TARGETS = (
    "T_max_Tx", "T_max_Rx_main", "T_max_Rx_side", "T_max_core",
    "Tprobe_Tx_leeward_max", "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max", "Tprobe_core_center_max",
    "Tprobe_core_center_leg_max", "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)
SIDE_TEMPERATURE_TARGETS = (
    "T_max_Rx_side", "Tprobe_Rx_side_leeward_max",
)
CORE_THERMAL_PRESSURE_CONSTRAINTS = (
    "temperature_robust_limit:T_max_core",
    "temperature_robust_limit:Tprobe_core_center_max",
    "temperature_robust_limit:Tprobe_core_top_yoke_max",
)
ALL_THERMAL_PRESSURE_CONSTRAINTS = tuple(
    f"temperature_robust_limit:{target}" for target in TEMPERATURE_TARGETS
)
THERMAL_CROSSOVER_WARM_SOURCE_SCALE_C = 1.0
EXPECTED_TEMPERATURE_LIMIT_C = 110.0
TEMPERATURE_FORMULA = (
    "surrogate_mu_plus_q90_conformal_half_width_le_limit"
)
UNCERTAINTY_CONTRACT = "q90_conformal_half_width_physical_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_io_path(path: Path) -> Path:
    """Return a Windows extended-length path for local filesystem I/O.

    Deployment identities and manifests retain their original absolute paths.
    Only filesystem calls use the extended-length spelling so a long runtime
    root can still contain the complete, authenticated source inventory.
    """

    path = Path(path)
    if os.name != "nt":
        return path
    value = os.path.abspath(os.fspath(path))
    if value.startswith("\\\\?\\"):
        return Path(value)
    if value.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{value[2:]}")
    return Path(f"\\\\?\\{value}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def search_profile() -> dict:
    """Return the immutable normalized-search identity used by every task."""

    profile = {
        "schema_version": "mft-tier1-nsga-search-profile-v1",
        "namespace": SEARCH_NAMESPACE,
        "variant": SEARCH_VARIANT,
        "population": POPULATION,
        "max_generations": MAX_GENERATIONS,
        "inference_threads": INFERENCE_THREADS,
        "rolling_target": ROLLING_TARGET,
        "seed_start": SEED_START,
        "task_priority": TASK_PRIORITY,
        "optimizer_resonance_scale_Hz": OPTIMIZER_RESONANCE_SCALE_HZ,
        "optimizer_scale_scope": "search_pressure_only_physical_G_unchanged",
        "legacy_migration_from_priority": LEGACY_TASK_PRIORITY,
        "priority_migration_seed_cutoff_exclusive": (
            PRIORITY_MIGRATION_SEED_CUTOFF
        ),
        "refill_policy": (
            "continuous-within-sealed-seed-window-v1"
            if DEEP_CROSSOVER_ISLAND is not None
            else "continuous-unbounded-seeds-v1"
        ),
    }
    if OPTIMIZER_CORE_THERMAL_SCALE_C is not None:
        acquisition = thermal_crossover_acquisition_contract({
            "size_W_max_mm": 1_200.0,
            "size_L_max_mm": 1_200.0,
        })
        profile.update({
            "optimizer_core_thermal_scale_C": (
                OPTIMIZER_CORE_THERMAL_SCALE_C
            ),
            "optimizer_core_thermal_constraints": list(
                CORE_THERMAL_PRESSURE_CONSTRAINTS
            ),
            "acquisition_ranking_contract_sha256": acquisition["sha256"],
            "soft_axis_target_W_mm": 1_000.0,
            "soft_axis_target_L_mm": 1_000.0,
            "soft_axis_pressure_scope": (
                "ranking_only_positive_excess_no_hard_G_mutation"
            ),
        })
    if OPTIMIZER_ALL_THERMAL_SCALE_C is not None:
        profile.update({
            "optimizer_all_active_thermal_scale_C": (
                OPTIMIZER_ALL_THERMAL_SCALE_C
            ),
            "optimizer_all_active_thermal_constraints": list(
                ALL_THERMAL_PRESSURE_CONSTRAINTS
            ),
            "optimizer_all_active_thermal_side_activation": (
                "finite_N2_side_gt_0_else_physical_negative_BIG"
            ),
            "soft_axis_pressure_scope": "disabled_until_stage_feasible",
        })
    if FIXED_PRIMARY_TURNS is not None:
        profile.update({
            "fixed_primary_turns": FIXED_PRIMARY_TURNS,
            "fixed_primary_turns_scope": (
                "initialization_warm_start_and_every_offspring_repair"
            ),
            "search_focus": SEARCH_FOCUS,
            "optimizer_Llt_scale_uH": (
                OPTIMIZER_LLT_SCALE_UH
                if OPTIMIZER_LLT_SCALE_UH is not None
                else 0.55
            ),
        })
    if ANCHOR_ISLAND is not None:
        profile.update({
            "anchor_island": anchor_island_profile(ANCHOR_ISLAND),
            "optimizer_resonance_allowance_Hz": (
                OPTIMIZER_RESONANCE_ALLOWANCE_HZ
            ),
            "optimizer_Llt_allowance_uH": OPTIMIZER_LLT_ALLOWANCE_UH,
            "decoded_params_sha256_duplicate_cap": 1,
            "island_active_quota": ANCHOR_ISLAND.active_quota,
            "physical_hard_spec_mutation": False,
        })
    if DEEP_CROSSOVER_ISLAND is not None:
        profile.update({
            "deep_crossover_island": deep_crossover_island_profile(
                DEEP_CROSSOVER_ISLAND
            ),
            "optimizer_termination_strategy": (
                OPTIMIZER_TERMINATION_STRATEGY
            ),
            "optimizer_resonance_allowance_Hz": (
                OPTIMIZER_RESONANCE_ALLOWANCE_HZ
            ),
            "optimizer_Llt_allowance_uH": OPTIMIZER_LLT_ALLOWANCE_UH,
            "seed_window_end_exclusive": (
                DEEP_CROSSOVER_ISLAND.seed_window_end_exclusive
            ),
            "decoded_params_sha256_duplicate_cap": 1,
            "island_active_quota": DEEP_CROSSOVER_ISLAND.active_quota,
            "physical_hard_spec_mutation": False,
        })
    return profile


def optimizer_termination_contract() -> dict:
    if OPTIMIZER_TERMINATION_STRATEGY == "default-ftol-period30-v1":
        rule = {
            "kind": "pymoo_DefaultMultiObjectiveTermination",
            "ftol": 0.0025,
            "period": 30,
            "n_max_gen": MAX_GENERATIONS,
            "early_stop_allowed": True,
        }
    elif OPTIMIZER_TERMINATION_STRATEGY == "fixed-n-gen-no-ftol-v1":
        rule = {
            "kind": "pymoo_minimize_tuple_n_gen",
            "n_max_gen": MAX_GENERATIONS,
            "early_stop_allowed": False,
            "completed_evolution_generations": MAX_GENERATIONS,
            "expected_algorithm_n_gen_counter": MAX_GENERATIONS + 1,
        }
    else:
        raise RuntimeError("unsupported optimizer termination strategy")
    contract = {
        "schema_version": "mft-tier1-optimizer-termination-v1",
        "strategy": OPTIMIZER_TERMINATION_STRATEGY,
        "rule": rule,
        "pinned_initial_population": "optimization.run_nsga2._initial_population",
        "pinned_physics_repair": "optimization.run_nsga2._physics_repair_operator",
        "physical_evaluator_mutation": False,
        "physical_constraint_mutation": False,
        "objective_mutation": False,
        "authoritative_terminal_G": "physical_unscaled_replay",
    }
    contract["sha256"] = canonical_sha(contract)
    return contract


def optimizer_core_thermal_overrides() -> dict[str, float] | None:
    if OPTIMIZER_CORE_THERMAL_SCALE_C is None:
        return None
    return {
        name: float(OPTIMIZER_CORE_THERMAL_SCALE_C)
        for name in CORE_THERMAL_PRESSURE_CONSTRAINTS
    }


def optimizer_all_thermal_overrides() -> dict[str, float] | None:
    if OPTIMIZER_ALL_THERMAL_SCALE_C is None:
        return None
    return {
        name: float(OPTIMIZER_ALL_THERMAL_SCALE_C)
        for name in ALL_THERMAL_PRESSURE_CONSTRAINTS
    }


def optimizer_explicit_overrides() -> dict[str, float] | None:
    overrides = {}
    overrides.update(optimizer_core_thermal_overrides() or {})
    overrides.update(optimizer_all_thermal_overrides() or {})
    if OPTIMIZER_LLT_SCALE_UH is not None:
        overrides.update({
            "Llt_robust_band": float(OPTIMIZER_LLT_SCALE_UH),
            "Llt_ensemble_disagreement": (
                2.0 * float(OPTIMIZER_LLT_SCALE_UH)
            ),
        })
    return overrides or None


def thermal_crossover_acquisition_contract(
    hard_spec: dict,
    *,
    core_thermal_scale_c: float | None = None,
) -> dict | None:
    scale = (
        OPTIMIZER_CORE_THERMAL_SCALE_C
        if core_thermal_scale_c is None
        else core_thermal_scale_c
    )
    if scale is None:
        return None
    if isinstance(scale, bool):
        raise RuntimeError(
            "acquisition core thermal scale must be finite and positive"
        )
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(
            "acquisition core thermal scale must be finite and positive"
        )
    contract = {
        "schema_version": (
            "mft-tier1-thermal-crossover-acquisition-ranking-v1"
        ),
        "active": True,
        "purpose": "candidate_acquisition_ranking_only",
        "authoritative_terminal_G": "physical_unscaled_unchanged",
        "hard_constraint_mutation": False,
        "core_thermal_constraints": list(
            CORE_THERMAL_PRESSURE_CONSTRAINTS
        ),
        "core_thermal_positive_G_scale_C": scale,
        "soft_axis_targets_mm": {
            "exterior_width": 1_000.0,
            "exterior_length": 1_000.0,
        },
        "soft_axis_positive_excess_scales_mm": {
            "exterior_width": 100.0,
            "exterior_length": 100.0,
        },
        "soft_axis_mode": (
            "independent_positive_excess_above_1000mm_no_new_hard_G"
        ),
        "physical_axis_recovery": {
            "exterior_width_mm": (
                "G_exterior_width_limit+hard_spec.size_W_max_mm"
            ),
            "exterior_length_mm": (
                "G_exterior_length_limit+hard_spec.size_L_max_mm"
            ),
        },
        "hard_axis_limits_mm": {
            "exterior_width": float(hard_spec["size_W_max_mm"]),
            "exterior_length": float(hard_spec["size_L_max_mm"]),
        },
        "ranking_formula": (
            "base_target_score+sum(max(core_thermal_G_C,0)/core_scale_C)"
            "+max(exterior_width_mm-1000,0)/100"
            "+max(exterior_length_mm-1000,0)/100"
        ),
    }
    contract["sha256"] = canonical_sha(contract)
    return contract


def temperature_constraint_contract(hard_spec: dict) -> dict:
    """Build the one accepted semantic contract for next-cohort thermal gates."""

    if not isinstance(hard_spec, dict) or "T_limit_C" not in hard_spec:
        raise RuntimeError("hard spec must explicitly contain T_limit_C")
    value = hard_spec["T_limit_C"]
    if isinstance(value, bool):
        raise RuntimeError("T_limit_C must be exactly 110.0 C")
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("T_limit_C must be exactly 110.0 C") from exc
    if limit != EXPECTED_TEMPERATURE_LIMIT_C:
        raise RuntimeError("next cohort requires T_limit_C exactly 110.0 C")
    unconditional = [
        target for target in TEMPERATURE_TARGETS
        if target not in SIDE_TEMPERATURE_TARGETS
    ]
    return {
        "schema_version": TEMPERATURE_CONTRACT_SCHEMA,
        "semantic_version": "all11-robust-q90-half-width-t110-v4",
        "robust_upper_bound_C": limit,
        "hard_spec_temperature_key": "T_limit_C",
        "formula": TEMPERATURE_FORMULA,
        "uncertainty_contract": UNCERTAINTY_CONTRACT,
        "targets": list(TEMPERATURE_TARGETS),
        "target_count": len(TEMPERATURE_TARGETS),
        "unconditional_targets": unconditional,
        "unconditional_target_count": len(unconditional),
        "side_winding_conditional_targets": list(SIDE_TEMPERATURE_TARGETS),
        "side_winding_conditional_target_count": len(SIDE_TEMPERATURE_TARGETS),
        "side_winding_activation": "finite_N2_side_gt_0",
        "side_winding_absent_behavior": (
            "finite_N2_side_eq_0_disables_with_negative_BIG"
        ),
        "side_winding_missing_behavior": (
            "missing_or_nonfinite_N2_side_fails_with_positive_BIG"
        ),
        "constraint_name_format": "temperature_robust_limit:{target}",
        "source": "model_targets.SURROGATE_TEMPERATURE_TARGETS",
    }


def validate_temperature_constraint_contract(
    hard_spec: dict, contract: dict, expected_sha256: str | None = None,
) -> str:
    """Fail closed on limit, formula, target order, or conditional mutation."""

    expected = temperature_constraint_contract(hard_spec)
    if contract != expected:
        raise RuntimeError("temperature constraint semantic contract mismatch")
    actual_sha = canonical_sha(contract)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RuntimeError("temperature constraint contract SHA-256 mismatch")
    return actual_sha


def _is_v3_current(current: dict) -> bool:
    return current.get("attestation_schema_version") == ATTESTATION_SCHEMA


def _validate_current_attestation(current: dict) -> None:
    """Validate a current pointer without breaking the already-live v2 cohort."""

    hard_spec = current.get("hard_spec")
    if not isinstance(hard_spec, dict):
        raise RuntimeError("model pointer has no hard-spec object")
    if canonical_sha(hard_spec) != current.get("hard_spec_sha256"):
        raise RuntimeError("model pointer hard-spec SHA-256 mismatch")
    if _is_v3_current(current):
        if current.get("task_schema_version") != TASK_SCHEMA:
            raise RuntimeError("v3 pointer task schema mismatch")
        validate_temperature_constraint_contract(
            hard_spec,
            current.get("temperature_constraint_contract"),
            current.get("temperature_constraint_contract_sha256"),
        )
        if not current.get("controller_source_sha256"):
            raise RuntimeError("v3 pointer has no controller source attestation")
        if current.get("search_profile") != search_profile():
            raise RuntimeError("v3 pointer search profile mismatch")
        if (
            current.get("optimizer_resonance_scale_Hz")
            != OPTIMIZER_RESONANCE_SCALE_HZ
        ):
            raise RuntimeError("v3 pointer optimizer resonance scale mismatch")
        if (
            current.get("optimizer_core_thermal_scale_C")
            != OPTIMIZER_CORE_THERMAL_SCALE_C
        ):
            raise RuntimeError(
                "v3 pointer optimizer core thermal scale mismatch"
            )
        if current.get("optimizer_Llt_scale_uH") != OPTIMIZER_LLT_SCALE_UH:
            raise RuntimeError("v3 pointer optimizer Llt scale mismatch")
        if (
            current.get("optimizer_all_active_thermal_scale_C")
            != OPTIMIZER_ALL_THERMAL_SCALE_C
        ):
            raise RuntimeError(
                "v3 pointer optimizer all-active thermal scale mismatch"
            )
        if (
            OPTIMIZER_CORE_THERMAL_SCALE_C is not None
            and (
                current.get("acquisition_ranking_contract")
                != thermal_crossover_acquisition_contract(hard_spec)
                or current.get("acquisition_ranking_contract_sha256")
                != (
                    current["acquisition_ranking_contract"]["sha256"]
                    if isinstance(
                        current.get("acquisition_ranking_contract"), dict
                    ) else None
                )
            )
        ):
            raise RuntimeError("v3 thermal acquisition contract mismatch")
        if (
            OPTIMIZER_ALL_THERMAL_SCALE_C is not None
            and (
                current.get("acquisition_ranking_contract") is not None
                or current.get("acquisition_ranking_contract_sha256")
                is not None
            )
        ):
            raise RuntimeError(
                "orthogonal lane must not apply soft-axis acquisition pressure"
            )
        if (
            (
                OPTIMIZER_CORE_THERMAL_SCALE_C is not None
                or OPTIMIZER_ALL_THERMAL_SCALE_C is not None
            )
            and any(
                current.get(field) is not False
                for field in (
                    "production_eligible", "fea_submission_approved",
                    "fea_submission_performed", "aedt_used",
                    "automatic_promotion_allowed",
                )
            )
        ):
            raise RuntimeError("v3 thermal pointer is not fail closed")
        return
    # Compatibility is intentionally narrow: this is the one cohort already
    # running when v3 was prepared.  It remains valid and refillable until an
    # explicit pointer transition, but no new legacy cohort can be published.
    if (
        current.get("constraint_version")
        != "mft-tier1-envelope-1200x1200x750-res10k-t120all11-core4-5t-lmhalf-v2"
        or current.get("nsga_code_revision")
        != "17539ca0adb43dc52144983ef1bf5e9e00a30d47"
        or hard_spec.get("T_limit_C") != EXPECTED_TEMPERATURE_LIMIT_C
    ):
        raise RuntimeError("unsupported legacy cohort pointer")


def _assignment_dict_keys(path: Path, assignment_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == assignment_name
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        ):
            return {
                ast.literal_eval(key)
                for key in node.value.keys
                if key is not None
            }
    raise RuntimeError(f"cannot find dict assignment {assignment_name} in {path}")


def _validate_code_temperature_semantics(code_root: Path) -> None:
    regression = code_root / "regression_260707"
    model_targets_path = regression / "model_targets.py"
    spec = importlib.util.spec_from_file_location(
        f"tier1_model_targets_{sha256(model_targets_path)[:12]}",
        model_targets_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load NSGA model target contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.SURROGATE_TEMPERATURE_TARGETS) != TEMPERATURE_TARGETS:
        raise RuntimeError("NSGA code temperature target order is not v3")
    if tuple(module.SIDE_TEMPERATURE_TARGETS) != SIDE_TEMPERATURE_TARGETS:
        raise RuntimeError("NSGA code side-temperature targets are not v3")
    default_keys = _assignment_dict_keys(
        regression / "optimization" / "nsga2_problem.py", "DEFAULT_SPEC"
    )
    if "T_limit_C" in default_keys:
        raise RuntimeError("NSGA DEFAULT_SPEC must not contain T_limit_C")


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def validate_resonance_focus_warm_contract(
    warm_start: Path,
) -> tuple[Path, str] | None:
    """Authenticate the exact 32+32 handoff before focused deployment."""

    if SEARCH_VARIANT != "resonance-focus-250hz-v1":
        return None
    warm_start = warm_start.resolve(strict=True)
    contract_path = warm_start.with_name("resonance_focus_warm_contract.json")
    contract = read_json(contract_path.resolve(strict=True))
    selected = contract.get("selected_provenance")
    sources = contract.get("source_evidence")
    warm = contract.get("warm_start")
    if (
        contract.get("schema_version")
        != "mft-tier1-resonance-focus-warm-handoff-v1"
        or not isinstance(contract.get("hard_spec"), dict)
        or canonical_sha(contract["hard_spec"])
        != contract.get("hard_spec_sha256")
        or contract.get("hard_spec_sha256")
        != "1a53578a25b96cbdb0cd0f06aa2a18574d2877e0340d860c35a17ee5496c0b59"
        or contract.get("constraint_version")
        != (
            "mft-tier1-envelope-1200x1200x750-res15k-"
            "t110all11-core4-5t-lmhalf-v4"
        )
        or contract.get("optimizer_resonance_scale_Hz") != 250.0
        or contract.get("optimizer_scale_scope")
        != "search_pressure_only_physical_G_unchanged"
        or contract.get("category_counts") != {
            "all_except_resonance": 32,
            "resonance_pass_near": 32,
        }
        or not isinstance(warm, dict)
        or warm.get("filename") != warm_start.name
        or warm.get("sha256") != sha256(warm_start)
        or warm.get("shape") != [64, 25]
        or not isinstance(selected, list)
        or len(selected) != 64
        or canonical_sha(selected)
        != contract.get("selected_provenance_sha256")
        or not isinstance(sources, dict)
        or set(sources) != {"main", "deep", "normalized"}
        or contract.get("launch_performed") is not False
        or contract.get("scheduler_write_performed") is not False
        or contract.get("fea_submission_performed") is not False
        or contract.get("aedt_used") is not False
        or contract.get("production_eligible") is not False
        or contract.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("resonance-focus warm handoff contract mismatch")
    expected_categories = [
        "all_except_resonance" if index % 2 == 0 else "resonance_pass_near"
        for index in range(64)
    ]
    try:
        normalized_rows = selected[::2]
        resonance_rows = selected[1::2]
        normalized_semantics = all(
            item.get("source_role") == "normalized"
            and not isinstance(
                item.get("other_normalized_positive_violation"), bool
            )
            and math.isfinite(float(
                item["other_normalized_positive_violation"]
            ))
            and float(item["other_normalized_positive_violation"]) <= 1e-9
            for item in normalized_rows
        )
        resonance_semantics = all(
            item.get("source_role") in {"main", "deep"}
            and not isinstance(item.get("resonance_G_Hz"), bool)
            and math.isfinite(float(item["resonance_G_Hz"]))
            and float(item["resonance_G_Hz"]) <= 1e-9
            for item in resonance_rows
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        normalized_semantics = resonance_semantics = False
    generation_identities = [
        source.get("generation_identity") for source in sources.values()
    ]
    expected_generation_identity = {
        "constraint_version": (
            "mft-tier1-envelope-1200x1200x750-res15k-"
            "t110all11-core4-5t-lmhalf-v4"
        ),
        "hard_spec_sha256": (
            "1a53578a25b96cbdb0cd0f06aa2a18574d2877e0340d860c35a17ee5496c0b59"
        ),
        "source_model_manifest_sha256": (
            "41497047d12f2ff88f0f6e067619015adea90016580501ed6a143dc79763ad91"
        ),
        "deployment_model_manifest_sha256": (
            "0408d7141c68d584b480a3aef6fa05126fc99e3ef171ace1051ee6df6925fe7f"
        ),
        "temperature_constraint_contract_sha256": (
            "976b78d647245071769fa79e28055c498f6dc569925c92861af2f4128faae87f"
        ),
        "hard_spec": contract["hard_spec"],
        "nsga_code_revision": (
            "7c832f7f78f92ee2d99b2d37e14c3131f07d9cae"
        ),
    }

    def is_sha256_text(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    source_semantics = all(
        source.get("role") == role
        and isinstance(source.get("cohort_id"), str)
        and isinstance(source.get("terminal_result_sha256"), list)
        and isinstance(source.get("terminal_seed_status_sha256"), list)
        and all(is_sha256_text(value) for value in source["terminal_result_sha256"])
        and all(
            is_sha256_text(value)
            for value in source["terminal_seed_status_sha256"]
        )
        and canonical_sha(source["terminal_result_sha256"])
        == source.get("terminal_result_set_sha256")
        and canonical_sha(source["terminal_seed_status_sha256"])
        == source.get("terminal_seed_status_set_sha256")
        for role, source in sources.items()
    )
    selected_links = all(
        item.get("source_role") in sources
        and item.get("cohort_id")
        == sources[item["source_role"]].get("cohort_id")
        and item.get("result_sha256")
        in sources[item["source_role"]].get("terminal_result_sha256", [])
        and item.get("seed_status_sha256")
        in sources[item["source_role"]].get(
            "terminal_seed_status_sha256", []
        )
        for item in selected
    )
    if (
        [item.get("warm_index") for item in selected] != list(range(64))
        or [item.get("category") for item in selected] != expected_categories
        or any(
            not isinstance(source.get("terminal_result_set_sha256"), str)
            or len(source["terminal_result_set_sha256"]) != 64
            or not isinstance(source.get("generation_identity"), dict)
            for source in sources.values()
        )
        or not normalized_semantics
        or not resonance_semantics
        or {
            item.get("source_role") for item in resonance_rows
        } != {"main", "deep"}
        or not all(
            identity == generation_identities[0]
            for identity in generation_identities[1:]
        )
        or generation_identities[0] != expected_generation_identity
        or not source_semantics
        or not selected_links
        or len({
            item.get("decoded_params_sha256") for item in selected
        }) != 64
        or any(
            not is_sha256_text(item.get(field))
            for item in selected
            for field in (
                "seed_status_sha256", "result_sha256",
                "decoded_params_sha256",
            )
        )
    ):
        raise RuntimeError("resonance-focus warm provenance mismatch")
    return contract_path, sha256(contract_path)


def validate_thermal_crossover_warm_contract(
    warm_start: Path, *, allow_external: bool = False,
) -> tuple[Path, str] | None:
    """Authenticate the exact balanced 64-point thermal crossover handoff."""

    if not allow_external and SEARCH_VARIANT not in {
        "thermal-crossover-core1c-v1",
        "thermal-crossover-balanced4c-v1",
        "fixed-n1-5-thermal-llt-v1",
        "fixed-n1-6-resonance-llt-v1",
    }:
        return None
    warm_start = warm_start.resolve(strict=True)
    contract_path = warm_start.with_name(
        "thermal_crossover_warm_contract.json"
    )
    contract = read_json(contract_path.resolve(strict=True))
    selected = contract.get("selected_provenance")
    sources = contract.get("source_evidence")
    warm = contract.get("warm_start")
    expected_categories = [
        "target_complete_branch" if index % 2 == 0
        else "cool_core_diverse_branch"
        for index in range(64)
    ]
    if (
        contract.get("schema_version")
        != "mft-tier1-thermal-crossover-warm-handoff-v1"
        or contract.get("selection_contract")
        != (
            "32_target_complete_pressure_at_least24_exact_plus_"
            "32_exact_cool_core_three_source_diverse_interleaved_v1"
        )
        or not isinstance(contract.get("hard_spec"), dict)
        or canonical_sha(contract["hard_spec"])
        != contract.get("hard_spec_sha256")
        or contract.get("hard_spec_sha256")
        != "1a53578a25b96cbdb0cd0f06aa2a18574d2877e0340d860c35a17ee5496c0b59"
        or contract.get("constraint_version")
        != (
            "mft-tier1-envelope-1200x1200x750-res15k-"
            "t110all11-core4-5t-lmhalf-v4"
        )
        or contract.get("authoritative_terminal_G") != "physical_unscaled"
        or contract.get("optimizer_scale_scope")
        != "search_pressure_only_physical_G_unchanged"
        or contract.get("optimizer_resonance_scale_Hz") != 250.0
        or contract.get("optimizer_core_thermal_scale_C")
        != THERMAL_CROSSOVER_WARM_SOURCE_SCALE_C
        or contract.get("optimizer_core_thermal_constraints")
        != list(CORE_THERMAL_PRESSURE_CONSTRAINTS)
        or contract.get("acquisition_ranking_contract")
        != thermal_crossover_acquisition_contract(
            contract["hard_spec"],
            core_thermal_scale_c=THERMAL_CROSSOVER_WARM_SOURCE_SCALE_C,
        )
        or contract.get("category_counts") != {
            "target_complete_branch": 32,
            "cool_core_diverse_branch": 32,
        }
        or contract.get("minimum_exact_target_complete_count") != 24
        or not isinstance(contract.get("exact_target_complete_count"), int)
        or not 24 <= contract["exact_target_complete_count"] <= 32
        or contract.get("exact_cool_core_count") != 32
        or contract.get("selected_source_roles")
        != ["deep", "main", "normalized", "resfocus"]
        or not isinstance(warm, dict)
        or warm.get("filename") != warm_start.name
        or warm.get("sha256") != sha256(warm_start)
        or warm.get("shape") != [64, 25]
        or not isinstance(selected, list)
        or len(selected) != 64
        or canonical_sha(selected)
        != contract.get("selected_provenance_sha256")
        or not isinstance(sources, dict)
        or set(sources) != {"main", "deep", "normalized", "resfocus"}
        or contract.get("rolling_target") != 32
        or contract.get("population") != 320
        or contract.get("max_generations") != 600
        or contract.get("inference_threads") != 8
        or contract.get("seed_start") != 1_907_193_000
        or contract.get("task_priority") != -6
        or contract.get("launch_performed") is not False
        or contract.get("scheduler_write_performed") is not False
        or contract.get("fea_submission_approved") is not False
        or contract.get("fea_submission_performed") is not False
        or contract.get("aedt_used") is not False
        or contract.get("production_eligible") is not False
        or contract.get("automatic_promotion_allowed") is not False
    ):
        raise RuntimeError("thermal-crossover warm handoff contract mismatch")

    def is_sha256_text(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    expected_generation_identity = {
        "constraint_version": (
            "mft-tier1-envelope-1200x1200x750-res15k-"
            "t110all11-core4-5t-lmhalf-v4"
        ),
        "hard_spec_sha256": (
            "1a53578a25b96cbdb0cd0f06aa2a18574d2877e0340d860c35a17ee5496c0b59"
        ),
        "source_model_manifest_sha256": (
            "41497047d12f2ff88f0f6e067619015adea90016580501ed6a143dc79763ad91"
        ),
        "deployment_model_manifest_sha256": (
            "0408d7141c68d584b480a3aef6fa05126fc99e3ef171ace1051ee6df6925fe7f"
        ),
        "temperature_constraint_contract_sha256": (
            "976b78d647245071769fa79e28055c498f6dc569925c92861af2f4128faae87f"
        ),
        "hard_spec": contract["hard_spec"],
        "nsga_code_revision": (
            "7c832f7f78f92ee2d99b2d37e14c3131f07d9cae"
        ),
    }
    source_semantics = all(
        source.get("role") == role
        and isinstance(source.get("cohort_id"), str)
        and isinstance(source.get("terminal_result_sha256"), list)
        and isinstance(source.get("terminal_seed_status_sha256"), list)
        and all(
            is_sha256_text(value)
            for value in source["terminal_result_sha256"]
        )
        and all(
            is_sha256_text(value)
            for value in source["terminal_seed_status_sha256"]
        )
        and canonical_sha(source["terminal_result_sha256"])
        == source.get("terminal_result_set_sha256")
        and canonical_sha(source["terminal_seed_status_sha256"])
        == source.get("terminal_seed_status_set_sha256")
        and source.get("generation_identity")
        == expected_generation_identity
        for role, source in sources.items()
    )
    resfocus = sources["resfocus"]
    resfocus_profile = resfocus.get("search_profile") or {}
    hotfix_semantics = (
        resfocus.get("controller_source_sha256")
        == "5097c476f983f6460861ff7a53aaef4472fc67eda0924006db3d4429fd27052a"
        and resfocus_profile.get("namespace")
        == "resonance-focus250-p320-g600-warm64-v1"
        and resfocus_profile.get("variant") == "resonance-focus-250hz-v1"
        and resfocus_profile.get("optimizer_resonance_scale_Hz") == 250.0
        and resfocus_profile.get("population") == 320
        and resfocus_profile.get("max_generations") == 600
        and resfocus_profile.get("inference_threads") == 8
        and resfocus_profile.get("rolling_target") == 32
        and resfocus_profile.get("seed_start") == 1_907_192_000
        and resfocus_profile.get("task_priority") == -7
    )
    selected_links = all(
        item.get("source_role") in sources
        and item.get("cohort_id")
        == sources[item["source_role"]].get("cohort_id")
        and item.get("result_sha256")
        in sources[item["source_role"]].get("terminal_result_sha256", [])
        and item.get("seed_status_sha256")
        in sources[item["source_role"]].get(
            "terminal_seed_status_sha256", []
        )
        for item in selected
    )
    try:
        target_rows = selected[::2]
        cool_rows = selected[1::2]
        target_semantics = all(
            isinstance(item.get("target_complete"), bool)
            and not isinstance(
                item.get("target_normalized_positive_violation"), bool
            )
            and math.isfinite(float(
                item["target_normalized_positive_violation"]
            ))
            and float(item["target_normalized_positive_violation"]) >= 0.0
            for item in target_rows
        )
        cool_semantics = all(
            item.get("cool_core_complete") is True
            and set(item.get("core_thermal_G_C") or {})
            == set(CORE_THERMAL_PRESSURE_CONSTRAINTS)
            and all(
                math.isfinite(float(value)) and float(value) <= 1e-9
                for value in item["core_thermal_G_C"].values()
            )
            and math.isfinite(float(
                item["core_thermal_max_positive_violation_C"]
            ))
            and float(item["core_thermal_max_positive_violation_C"]) <= 1e-9
            for item in cool_rows
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        target_semantics = cool_semantics = False
    if (
        [item.get("warm_index") for item in selected] != list(range(64))
        or [item.get("category") for item in selected] != expected_categories
        or sum(item.get("target_complete") is True for item in target_rows)
        != contract["exact_target_complete_count"]
        or not target_semantics
        or not cool_semantics
        or not source_semantics
        or not hotfix_semantics
        or not selected_links
        or {item.get("source_role") for item in selected}
        != {"main", "deep", "normalized", "resfocus"}
        or {item.get("source_role") for item in cool_rows}
        != {"main", "deep", "normalized"}
        or "resfocus" not in {
            item.get("source_role") for item in target_rows
        }
        or len({
            item.get("decoded_params_sha256") for item in selected
        }) != 64
        or any(
            not is_sha256_text(item.get(field))
            for item in selected
            for field in (
                "seed_status_sha256", "result_sha256",
                "decoded_params_sha256", "unit_coordinate_sha256",
            )
        )
    ):
        raise RuntimeError("thermal-crossover warm provenance mismatch")
    return contract_path, sha256(contract_path)


def _fixed_bridge_decode_coordinates(
    decoded_params: list[dict], nsga_code_root: Path,
) -> np.ndarray:
    from tools.tier1_terminal_followup import (  # noqa: PLC0415
        _load_sobol_schema,
        decoded_to_unit,
    )

    dimensions, n1_min, n1_max, defaults = _load_sobol_schema(nsga_code_root)
    return np.vstack([
        decoded_to_unit(params, dimensions, n1_min, n1_max, defaults)
        for params in decoded_params
    ])


def validate_fixed_n1_6_bridge_warm_contract(
    warm_start: Path, nsga_code_root: Path | None = None,
) -> tuple[Path, str]:
    """Authenticate the 64-point fixed-N1=6 thermal/resonance bridge."""

    warm_start = warm_start.resolve(strict=True)
    contract_path = warm_start.with_name(
        "fixed_n1_6_bridge_warm_contract.json"
    )
    contract = read_json(contract_path.resolve(strict=True))
    warm = contract.get("warm_start")
    selected = contract.get("selected_provenance")
    sources = contract.get("source_evidence")
    snapshots = contract.get("source_snapshots")
    selected_artifacts = contract.get("selected_artifact_snapshots")
    post_repair_preflight = contract.get("post_repair_preflight")
    anchors = contract.get("anchors")
    required_anchors = {
        "normalized": 57059,
        "thermal": 57919,
        "fixed6": 58486,
        "resfocus": 57505,
    }
    expected_roles = set(required_anchors)
    expected_categories = [
        "low_temperature_branch" if index % 2 == 0
        else "resonance_pass_branch"
        for index in range(64)
    ]
    expected_branch_role_counts = {
        "low_temperature_branch": {
            "normalized": 16,
            "thermal": 16,
        },
        "resonance_pass_branch": {
            "fixed6": 24,
            "resfocus": 8,
        },
    }
    expected_constraint_names = (
        "Llt_robust_band",
        "temperature_robust_limit:T_max_Tx",
        "temperature_robust_limit:T_max_Rx_main",
        "temperature_robust_limit:T_max_Rx_side",
        "temperature_robust_limit:T_max_core",
        "temperature_robust_limit:Tprobe_Tx_leeward_max",
        "temperature_robust_limit:Tprobe_Rx_main_leeward_max",
        "temperature_robust_limit:Tprobe_Rx_side_leeward_max",
        "temperature_robust_limit:Tprobe_core_center_max",
        "temperature_robust_limit:Tprobe_core_center_leg_max",
        "temperature_robust_limit:Tprobe_core_side_leg_max",
        "temperature_robust_limit:Tprobe_core_top_yoke_max",
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

    def is_sha256_text(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    warm_array_ok = False
    coordinate_links_ok = False
    try:
        coordinates = np.load(warm_start, allow_pickle=False)
        warm_array_ok = bool(
            isinstance(coordinates, np.ndarray)
            and coordinates.shape == (64, 25)
            and coordinates.dtype == np.dtype("float64")
            and np.isfinite(coordinates).all()
            and (coordinates >= 0.0).all()
            and (coordinates <= 1.0).all()
        )
        row_shas = [
            canonical_sha([float(value) for value in coordinates[index]])
            for index in range(64)
        ] if warm_array_ok else []
        coordinate_links_ok = bool(
            warm_array_ok
            and isinstance(selected, list)
            and len(selected) == 64
            and len(set(row_shas)) == 64
            and all(
                row_shas[index]
                == selected[index].get("unit_coordinate_sha256")
                for index in range(64)
            )
        )
    except (OSError, ValueError, TypeError, KeyError):
        warm_array_ok = coordinate_links_ok = False

    if (
        contract.get("schema_version")
        != "mft-tier1-fixed-n1-6-bridge-warm-handoff-v1"
        or contract.get("selection_contract")
        != (
            "32_low_temperature_n1_6_normalized16_thermal16_plus_"
            "32_resonance_pass_fixed6_24_resfocus8_interleaved_v1"
        )
        or not isinstance(contract.get("hard_spec"), dict)
        or canonical_sha(contract["hard_spec"])
        != contract.get("hard_spec_sha256")
        or contract.get("hard_spec_sha256")
        != "1a53578a25b96cbdb0cd0f06aa2a18574d2877e0340d860c35a17ee5496c0b59"
        or contract.get("constraint_version")
        != (
            "mft-tier1-envelope-1200x1200x750-res15k-"
            "t110all11-core4-5t-lmhalf-v4"
        )
        or tuple(contract.get("constraint_names") or ())
        != expected_constraint_names
        or canonical_sha(contract.get("constraint_names"))
        != contract.get("constraint_names_sha256")
        or contract.get("nsga_code_revision")
        != "7c832f7f78f92ee2d99b2d37e14c3131f07d9cae"
        or contract.get("nsga_code_root_revision")
        != "7c832f7f78f92ee2d99b2d37e14c3131f07d9cae"
        or contract.get("nsga_code_root_revision_verified") is not True
        or contract.get("nsga_code_root_clean_verified") is not True
        or contract.get("nsga_decoder_relative_path")
        != "module/input_parameter_260706.py"
        or not is_sha256_text(contract.get("nsga_decoder_source_sha256"))
        or contract.get("fixed_primary_turns") != 6
        or contract.get("fixed_primary_turns_scope")
        != "runner_repair_after_authenticated_inverse_coordinate_handoff"
        or contract.get("authoritative_terminal_G") != "physical_unscaled"
        or contract.get("optimizer_scale_scope")
        != "search_pressure_only_physical_G_unchanged"
        or contract.get("optimizer_resonance_scale_Hz") != 150.0
        or contract.get("optimizer_core_thermal_scale_C") != 2.0
        or contract.get("optimizer_Llt_scale_uH") != 0.55
        or contract.get("category_counts") != {
            "low_temperature_branch": 32,
            "resonance_pass_branch": 32,
        }
        or contract.get("source_role_counts") != {
            "normalized": 16,
            "thermal": 16,
            "fixed6": 24,
            "resfocus": 8,
        }
        or contract.get("branch_source_role_counts")
        != expected_branch_role_counts
        or contract.get("required_anchor_tasks") != required_anchors
        or not isinstance(warm, dict)
        or warm.get("filename") != warm_start.name
        or warm.get("sha256") != sha256(warm_start)
        or warm.get("shape") != [64, 25]
        or warm.get("dtype") != "float64"
        or warm.get("coordinate_contract")
        != "authenticated_decoded_to_unit_then_fixed_n1_6_repair_v1"
        or not isinstance(selected, list)
        or len(selected) != 64
        or canonical_sha(selected)
        != contract.get("selected_provenance_sha256")
        or not isinstance(sources, dict)
        or set(sources) != expected_roles
        or not isinstance(snapshots, dict)
        or set(snapshots) != expected_roles
        or not isinstance(selected_artifacts, list)
        or len(selected_artifacts) != 64
        or canonical_sha(selected_artifacts)
        != contract.get("selected_artifact_snapshots_sha256")
        or not isinstance(post_repair_preflight, dict)
        or post_repair_preflight.get("schema_version")
        != "mft-tier1-fixed-n1-6-bridge-post-repair-preflight-v1"
        or post_repair_preflight.get("required_for_deployment") is not True
        or post_repair_preflight.get("verified") is not True
        or not isinstance(anchors, dict)
        or set(anchors) != expected_roles
        or contract.get("rolling_target") != 32
        or contract.get("population") != 320
        or contract.get("max_generations") != 600
        or contract.get("inference_threads") != 8
        or contract.get("seed_start") != 1_907_197_000
        or contract.get("task_priority") != -2
        or not warm_array_ok
        or not coordinate_links_ok
        or any(
            contract.get(field) is not False
            for field in (
                "launch_performed", "scheduler_write_performed",
                "fea_submission_approved", "fea_submission_performed",
                "aedt_used", "production_eligible",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("fixed-N1=6 bridge warm handoff contract mismatch")

    expected_generation_identity = {
        "constraint_version": contract["constraint_version"],
        "hard_spec_sha256": contract["hard_spec_sha256"],
        "source_model_manifest_sha256": (
            "41497047d12f2ff88f0f6e067619015adea90016580501ed6a143dc79763ad91"
        ),
        "deployment_model_manifest_sha256": (
            "0408d7141c68d584b480a3aef6fa05126fc99e3ef171ace1051ee6df6925fe7f"
        ),
        "temperature_constraint_contract_sha256": (
            "976b78d647245071769fa79e28055c498f6dc569925c92861af2f4128faae87f"
        ),
        "hard_spec": contract["hard_spec"],
        "nsga_code_revision": contract["nsga_code_revision"],
    }
    expected_profiles = {
        "normalized": None,
        "thermal": (
            "thermal-crossover-core1c-res250-p320-g600-warm64-v1",
            "thermal-crossover-core1c-v1",
        ),
        "fixed6": (
            "fixed-n1-6-resonance-llt-core4c-res100-p320-g600-warm64-v1",
            "fixed-n1-6-resonance-llt-v1",
        ),
        "resfocus": (
            "resonance-focus250-p320-g600-warm64-v1",
            "resonance-focus-250hz-v1",
        ),
    }
    expected_controller_sha = {
        "normalized": (
            "0b48bcf01e691be0ee217700f173d6cf42f80902f5457251684bf19ac0de7e86"
        ),
        "thermal": (
            "ffc451375af92ac13910f59c9dd6888a0e1d595e6767ba41f6927cee3bccc71d"
        ),
        "fixed6": (
            "fbc1c784d06e2a7b3fc4b60fab0db665e8707fafb658451fd7a4e22a6762f2a2"
        ),
        "resfocus": (
            "5097c476f983f6460861ff7a53aaef4472fc67eda0924006db3d4429fd27052a"
        ),
    }
    expected_snapshot_schemas = {
        "index": "mft-tier1-slurm-rolling-index-v1",
        "pointer": "mft-tier1-slurm-model-pointer-v1",
        "status": "mft-tier1-slurm-rolling-status-v1",
    }

    snapshot_semantics = True
    terminal_reference_tuples = {}
    snapshot_root = contract_path.parent.resolve(strict=True)
    post_repair_preflight_semantics = False
    try:
        from tools.tier1_fixed_n1_6_bridge_warm_handoff import (  # noqa: PLC0415
            _post_repair_preflight_summary,
        )

        preflight_reference = post_repair_preflight.get("result")
        if not isinstance(preflight_reference, dict):
            raise RuntimeError("post-repair preflight reference missing")
        preflight_relative = Path(str(preflight_reference.get("path") or ""))
        if preflight_relative.is_absolute() or not preflight_relative.parts:
            raise RuntimeError("post-repair preflight path is not relative")
        preflight_path = (snapshot_root / preflight_relative).resolve(strict=True)
        if preflight_path != snapshot_root and snapshot_root not in preflight_path.parents:
            raise RuntimeError("post-repair preflight escaped handoff")
        preflight_payload = preflight_path.read_bytes()
        preflight_digest = hashlib.sha256(preflight_payload).hexdigest()
        preflight_result = json.loads(preflight_payload.decode("utf-8"))
        if (
            not isinstance(preflight_result, dict)
            or preflight_digest != preflight_reference.get("sha256")
            or len(preflight_payload) != preflight_reference.get("size_bytes")
        ):
            raise RuntimeError("post-repair preflight byte seal mismatch")
        verified_summary = _post_repair_preflight_summary(
            contract, preflight_result,
        )
        post_repair_preflight_semantics = bool(
            verified_summary == post_repair_preflight.get("verified_summary")
            and canonical_sha(verified_summary)
            == post_repair_preflight.get("verified_summary_sha256")
            and verified_summary.get("source_warm_sha256")
            == warm.get("sha256")
            and verified_summary.get("post_repair_decoded_unique_count") == 64
            and verified_summary.get("post_repair_hard_feasible_count") == 64
            and verified_summary.get("post_repair_rejected_count") == 0
        )
    except (
        AttributeError, KeyError, OSError, RuntimeError, UnicodeDecodeError,
        json.JSONDecodeError, TypeError, ValueError,
    ):
        post_repair_preflight_semantics = False
    for role in expected_roles:
        role_snapshot = snapshots.get(role)
        source = sources[role]
        if not isinstance(role_snapshot, dict) or set(role_snapshot) != set(
            expected_snapshot_schemas
        ):
            snapshot_semantics = False
            continue
        loaded = {}
        for kind, schema in expected_snapshot_schemas.items():
            reference = role_snapshot.get(kind)
            try:
                if not isinstance(reference, dict):
                    raise RuntimeError("snapshot reference missing")
                relative = Path(str(reference.get("path") or ""))
                if relative.is_absolute() or not relative.parts:
                    raise RuntimeError("snapshot path is not relative")
                path = (snapshot_root / relative).resolve(strict=True)
                if path != snapshot_root and snapshot_root not in path.parents:
                    raise RuntimeError("snapshot escaped contract directory")
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                value = json.loads(payload.decode("utf-8"))
                source_reference = source.get(kind)
                if (
                    not isinstance(value, dict)
                    or value.get("schema_version") != schema
                    or digest != reference.get("sha256")
                    or len(payload) != reference.get("size_bytes")
                    or not isinstance(source_reference, dict)
                    or digest != source_reference.get("sha256")
                    or digest != reference.get("source_sha256")
                ):
                    raise RuntimeError("snapshot seal mismatch")
                loaded[kind] = value
            except (
                OSError, RuntimeError, UnicodeDecodeError,
                json.JSONDecodeError, TypeError, ValueError,
            ):
                snapshot_semantics = False
        if set(loaded) != set(expected_snapshot_schemas):
            continue
        index = loaded["index"]
        pointer = loaded["pointer"]
        status = loaded["status"]
        current = pointer.get("current")
        generation = source.get("generation_identity")
        identity_keys = (
            "constraint_version", "hard_spec_sha256",
            "source_model_manifest_sha256",
            "deployment_model_manifest_sha256",
            "temperature_constraint_contract_sha256",
        )
        snapshot_semantics = snapshot_semantics and bool(
            isinstance(current, dict)
            and isinstance(generation, dict)
            and index.get("model_pointer", {}).get("sha256")
            == role_snapshot["pointer"]["sha256"]
            and index.get("status", {}).get("sha256")
            == role_snapshot["status"]["sha256"]
            and index.get("active_cohort_id") == current.get("cohort_id")
            and status.get("cohort_id") == current.get("cohort_id")
            and current.get("cohort_id") == source.get("cohort_id")
            and index.get("updated_at") == status.get("updated_at")
            and current.get("bundle_manifest_sha256")
            == source.get("bundle_manifest_sha256")
            and current.get("controller_source_sha256")
            == source.get("controller_source_sha256")
            and current.get("search_profile") == source.get("search_profile")
            and all(
                index.get(key) == status.get(key) == current.get(key)
                == generation.get(key)
                for key in identity_keys
            )
            and index.get("hard_spec") == contract["hard_spec"]
            and status.get("hard_spec") == contract["hard_spec"]
            and current.get("hard_spec") == contract["hard_spec"]
            and generation.get("hard_spec") == contract["hard_spec"]
            and status.get("healthy") is True
            and status.get("error") in {None, ""}
            and status.get("production_eligible") is False
            and status.get("fea_submission_approved") is False
            and status.get("fea_submission_performed") is False
            and status.get("aedt_used") is False
        )
        try:
            terminals = status.get("terminal_results")
            if not isinstance(terminals, list) or not terminals:
                raise RuntimeError("terminal snapshot missing")
            terminal_references = []
            for terminal in terminals:
                result_reference = terminal.get("result")
                status_reference = terminal.get("remote_status")
                if (
                    not isinstance(result_reference, dict)
                    or not isinstance(status_reference, dict)
                    or terminal.get("authenticated") is not True
                    or terminal.get("terminal_state") != "completed"
                    or terminal.get("scheduler_state") != "completed"
                    or int(terminal.get("scheduler_exit_code", -1)) != 0
                    or terminal.get("cohort_id") != source.get("cohort_id")
                    or not is_sha256_text(result_reference.get("sha256"))
                    or not is_sha256_text(status_reference.get("sha256"))
                ):
                    raise RuntimeError("terminal snapshot row mismatch")
                terminal_references.append({
                    "task_id": int(terminal["task_id"]),
                    "seed": int(terminal["seed"]),
                    "result_sha256": result_reference["sha256"],
                    "seed_status_sha256": status_reference["sha256"],
                })
            result_order = [
                item["result_sha256"] for item in terminal_references
            ]
            status_order = [
                item["seed_status_sha256"] for item in terminal_references
            ]
            reference_set = sorted(
                terminal_references,
                key=lambda item: (
                    item["task_id"], item["seed"], item["result_sha256"],
                    item["seed_status_sha256"],
                ),
            )
            terminal_reference_tuples[role] = {
                (
                    item["task_id"], item["seed"], item["result_sha256"],
                    item["seed_status_sha256"],
                )
                for item in terminal_references
            }
            snapshot_semantics = snapshot_semantics and bool(
                len({item["task_id"] for item in terminal_references})
                == len(terminal_references)
                and len({item["seed"] for item in terminal_references})
                == len(terminal_references)
                and len(terminal_references)
                == source.get("terminal_snapshot_count")
                and sorted(result_order)
                == source.get("terminal_result_sha256")
                and sorted(status_order)
                == source.get("terminal_seed_status_sha256")
                and canonical_sha(sorted(result_order))
                == source.get("terminal_result_set_sha256")
                and canonical_sha(sorted(status_order))
                == source.get("terminal_seed_status_set_sha256")
                and canonical_sha(result_order)
                == source.get("terminal_result_order_sha256")
                and canonical_sha(status_order)
                == source.get("terminal_seed_status_order_sha256")
                and canonical_sha(terminal_references)
                == source.get("terminal_reference_order_sha256")
                and canonical_sha(reference_set)
                == source.get("terminal_reference_set_sha256")
            )
        except (KeyError, TypeError, ValueError, OverflowError, RuntimeError):
            snapshot_semantics = False
    source_semantics = True
    for role, source in sources.items():
        profile = source.get("search_profile")
        expected_profile = expected_profiles[role]
        profile_ok = (
            profile is None if expected_profile is None
            else isinstance(profile, dict)
            and profile.get("namespace") == expected_profile[0]
            and profile.get("variant") == expected_profile[1]
        )
        source_semantics = source_semantics and bool(
            source.get("role") == role
            and isinstance(source.get("cohort_id"), str)
            and source.get("generation_identity")
            == expected_generation_identity
            and source.get("controller_source_sha256")
            == expected_controller_sha[role]
            and profile_ok
            and isinstance(source.get("terminal_result_sha256"), list)
            and isinstance(source.get("terminal_seed_status_sha256"), list)
            and all(
                is_sha256_text(value)
                for value in source["terminal_result_sha256"]
            )
            and all(
                is_sha256_text(value)
                for value in source["terminal_seed_status_sha256"]
            )
            and canonical_sha(source["terminal_result_sha256"])
            == source.get("terminal_result_set_sha256")
            and canonical_sha(source["terminal_seed_status_sha256"])
            == source.get("terminal_seed_status_set_sha256")
        )

    selected_artifact_semantics = True
    decoded_params_for_rows = []
    for index, artifact in enumerate(selected_artifacts):
        provenance = selected[index]
        try:
            if (
                not isinstance(artifact, dict)
                or artifact.get("warm_index") != index
                or artifact.get("source_role")
                != provenance.get("source_role")
                or artifact.get("cohort_id") != provenance.get("cohort_id")
                or artifact.get("task_id") != provenance.get("task_id")
                or artifact.get("seed") != provenance.get("seed")
                or (
                    int(provenance.get("task_id", -1)),
                    int(provenance.get("seed", -1)),
                    provenance.get("result_sha256"),
                    provenance.get("seed_status_sha256"),
                ) not in terminal_reference_tuples.get(
                    provenance.get("source_role"), set()
                )
            ):
                raise RuntimeError("selected artifact identity mismatch")
            parsed = {}
            for kind, expected_sha in (
                ("result", provenance.get("result_sha256")),
                ("seed_status", provenance.get("seed_status_sha256")),
            ):
                reference = artifact.get(kind)
                if not isinstance(reference, dict):
                    raise RuntimeError("selected artifact reference missing")
                relative = Path(str(reference.get("path") or ""))
                if relative.is_absolute() or not relative.parts:
                    raise RuntimeError("selected artifact path is not relative")
                path = (snapshot_root / relative).resolve(strict=True)
                if path != snapshot_root and snapshot_root not in path.parents:
                    raise RuntimeError("selected artifact escaped handoff")
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                value = json.loads(payload.decode("utf-8"))
                if (
                    not isinstance(value, dict)
                    or digest != reference.get("sha256")
                    or digest != expected_sha
                    or len(payload) != reference.get("size_bytes")
                ):
                    raise RuntimeError("selected artifact byte seal mismatch")
                parsed[kind] = value
            result = parsed["result"]
            seed_status = parsed["seed_status"]
            candidates = (
                result.get("next_target_fea_batch_plan") or {}
            ).get("candidates")
            matches = [
                candidate for candidate in candidates or []
                if isinstance(candidate, dict)
                and candidate.get("decoded_params_sha256")
                == provenance.get("decoded_params_sha256")
            ]
            if len(matches) != 1:
                raise RuntimeError("selected result candidate is not unique")
            candidate = matches[0]
            params = candidate.get("decoded_params")
            constraints = candidate.get("constraint_G")
            if not isinstance(params, dict) or not isinstance(constraints, dict):
                raise RuntimeError("selected candidate payload is incomplete")
            thermal_values = [
                float(value) for name, value in constraints.items()
                if str(name).startswith("temperature_robust_limit:")
            ]
            core_names = (
                "temperature_robust_limit:T_max_core",
                "temperature_robust_limit:Tprobe_core_center_max",
                "temperature_robust_limit:Tprobe_core_top_yoke_max",
            )
            core_positive = [
                max(float(constraints[name]), 0.0) for name in core_names
            ]
            resonance_g = float(
                constraints["half_magnetizing_resonance_minimum"]
            )
            llt_g = float(constraints["Llt_robust_band"])
            numeric_values = [
                *thermal_values, *core_positive, resonance_g, llt_g,
            ]
            source = sources[provenance["source_role"]]
            generation = source["generation_identity"]
            if (
                result.get("schema_version")
                != "mft-tier1-corrected-search-seed-v1"
                or int(result.get("seed", -1)) != int(provenance["seed"])
                or result.get("nsga_code_revision")
                != contract["nsga_code_revision"]
                or result.get("constraint_version")
                != contract["constraint_version"]
                or result.get("hard_spec") != contract["hard_spec"]
                or result.get("hard_spec_sha256")
                != contract["hard_spec_sha256"]
                or result.get("model_manifest_sha256")
                != generation["deployment_model_manifest_sha256"]
                or result.get("temperature_constraint_contract_sha256")
                != generation["temperature_constraint_contract_sha256"]
                or tuple(result.get("constraint_names") or ())
                != expected_constraint_names
                or set(constraints) != set(expected_constraint_names)
                or len(constraints) != len(expected_constraint_names)
                or result.get("production_eligible") is not False
                or result.get("fea_submission_approved") is not False
                or result.get("automatic_promotion_allowed") is not False
                or seed_status.get("schema_version")
                != "mft-tier1-slurm-seed-status-v1"
                or seed_status.get("state") != "completed"
                or int(seed_status.get("exit_code", -1)) != 0
                or int(seed_status.get("seed", -1))
                != int(provenance["seed"])
                or str(seed_status.get("task_id"))
                != str(provenance["task_id"])
                or seed_status.get("cohort_id")
                != provenance["cohort_id"]
                or seed_status.get("result_sha256")
                != provenance["result_sha256"]
                or seed_status.get("bundle_manifest_sha256")
                != source["bundle_manifest_sha256"]
                or seed_status.get("hard_spec_sha256")
                != contract["hard_spec_sha256"]
                or seed_status.get("temperature_constraint_contract_sha256")
                != generation["temperature_constraint_contract_sha256"]
                or seed_status.get("production_eligible") is not False
                or seed_status.get("fea_submission_approved") is not False
                or seed_status.get("fea_submission_performed") is not False
                or canonical_sha(params)
                != provenance["decoded_params_sha256"]
                or int(params.get("N1", -1)) != provenance["original_N1"]
                or provenance["fixed_N1_repair_required"]
                is not (int(params.get("N1", -1)) != 6)
                or not thermal_values
                or not all(math.isfinite(value) for value in numeric_values)
                or provenance["all_thermal_pass"]
                is not all(value <= 1e-9 for value in thermal_values)
                or provenance["resonance_pass"] is not (resonance_g <= 1e-9)
                or not math.isclose(
                    float(provenance["Llt_robust_G_uH"]), llt_g,
                    rel_tol=0.0, abs_tol=1e-12,
                )
                or not math.isclose(
                    float(provenance["resonance_G_Hz"]), resonance_g,
                    rel_tol=0.0, abs_tol=1e-12,
                )
                or not math.isclose(
                    float(provenance["core_positive_max_C"]),
                    max(core_positive), rel_tol=0.0, abs_tol=1e-12,
                )
                or not math.isclose(
                    float(provenance["core_positive_sum_C"]),
                    sum(core_positive), rel_tol=0.0, abs_tol=1e-12,
                )
            ):
                raise RuntimeError("selected artifact semantics mismatch")
            decoded_params_for_rows.append(params)
        except (
            KeyError, OSError, RuntimeError, UnicodeDecodeError,
            json.JSONDecodeError, TypeError, ValueError, OverflowError,
        ):
            selected_artifact_semantics = False

    decoded_coordinate_semantics = False
    code_identity_semantics = False
    if selected_artifact_semantics and nsga_code_root is not None:
        try:
            from tools.tier1_fixed_n1_6_bridge_warm_handoff import (  # noqa: PLC0415
                _verify_nsga_code_root,
            )

            code_identity = _verify_nsga_code_root(
                nsga_code_root.resolve(strict=True)
            )
            code_identity_semantics = bool(
                code_identity["revision"]
                == contract["nsga_code_root_revision"]
                and code_identity["clean"] is True
                and code_identity["decoder_relative_path"]
                == contract["nsga_decoder_relative_path"]
                and code_identity["decoder_sha256"]
                == contract["nsga_decoder_source_sha256"]
            )
            decoded_coordinates = _fixed_bridge_decode_coordinates(
                decoded_params_for_rows, nsga_code_root.resolve(strict=True),
            )
            decoded_coordinate_semantics = bool(
                decoded_coordinates.shape == (64, 25)
                and np.isfinite(decoded_coordinates).all()
                and np.allclose(
                    decoded_coordinates, coordinates, rtol=0.0, atol=1e-12,
                )
            )
        except (
            OSError, RuntimeError, subprocess.SubprocessError,
            TypeError, ValueError,
        ):
            code_identity_semantics = decoded_coordinate_semantics = False

    try:
        selected_semantics = all(
            item.get("source_role") in sources
            and item.get("cohort_id")
            == sources[item["source_role"]].get("cohort_id")
            and item.get("result_sha256")
            in sources[item["source_role"]]["terminal_result_sha256"]
            and item.get("seed_status_sha256")
            in sources[item["source_role"]]["terminal_seed_status_sha256"]
            and is_sha256_text(item.get("decoded_params_sha256"))
            and is_sha256_text(item.get("unit_coordinate_sha256"))
            and isinstance(item.get("original_N1"), int)
            and item.get("fixed_N1_repair_required")
            is (item["original_N1"] != 6)
            and all(
                math.isfinite(float(item[field]))
                for field in (
                    "Llt_robust_G_uH", "resonance_G_Hz",
                    "core_positive_max_C", "core_positive_sum_C",
                )
            )
            for item in selected
        )
        low_rows = selected[::2]
        resonance_rows = selected[1::2]
        low_semantics = all(
            item["category"] == "low_temperature_branch"
            and item["source_role"] in {"normalized", "thermal"}
            and item["original_N1"] == 6
            and item["fixed_N1_repair_required"] is False
            and item["all_thermal_pass"] is True
            for item in low_rows
        )
        resonance_semantics = all(
            item["category"] == "resonance_pass_branch"
            and item["source_role"] in {"fixed6", "resfocus"}
            and item["resonance_pass"] is True
            and float(item["resonance_G_Hz"]) <= 1e-9
            and (
                item["source_role"] != "fixed6"
                or (
                    item["original_N1"] == 6
                    and item["fixed_N1_repair_required"] is False
                )
            )
            for item in resonance_rows
        )
        actual_category_counts = {
            category: sum(
                item.get("category") == category for item in selected
            )
            for category in expected_branch_role_counts
        }
        actual_source_role_counts = {
            role: sum(item.get("source_role") == role for item in selected)
            for role in expected_roles
        }
        actual_branch_role_counts = {
            category: {
                role: sum(
                    item.get("category") == category
                    and item.get("source_role") == role
                    for item in selected
                )
                for role in role_counts
            }
            for category, role_counts in expected_branch_role_counts.items()
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        selected_semantics = low_semantics = resonance_semantics = False
        actual_category_counts = {}
        actual_source_role_counts = {}
        actual_branch_role_counts = {}

    anchors_ok = all(
        anchors[role] in selected
        and anchors[role].get("source_role") == role
        and anchors[role].get("task_id") == task_id
        for role, task_id in required_anchors.items()
    )
    if (
        [item.get("warm_index") for item in selected] != list(range(64))
        or [item.get("category") for item in selected]
        != expected_categories
        or not source_semantics
        or not snapshot_semantics
        or not post_repair_preflight_semantics
        or not selected_artifact_semantics
        or not decoded_coordinate_semantics
        or not code_identity_semantics
        or not selected_semantics
        or not low_semantics
        or not resonance_semantics
        or not anchors_ok
        or actual_category_counts != contract["category_counts"]
        or actual_source_role_counts != contract["source_role_counts"]
        or actual_branch_role_counts != expected_branch_role_counts
        or actual_branch_role_counts
        != contract["branch_source_role_counts"]
        or len({item.get("decoded_params_sha256") for item in selected}) != 64
        or {item.get("source_role") for item in low_rows}
        != {"normalized", "thermal"}
        or {item.get("source_role") for item in resonance_rows}
        != {"fixed6", "resfocus"}
    ):
        raise RuntimeError("fixed-N1=6 bridge warm provenance mismatch")
    return contract_path, sha256(contract_path)


def validate_fixed_n1_6_orthogonal_warm_contract(
    warm_start: Path, nsga_code_root: Path | None = None,
) -> tuple[Path, str]:
    """Authenticate the sealed 3-way, fixed-N1=6 orthogonal warm pool."""

    from tools.tier1_fixed_n1_6_bridge_warm_handoff import (  # noqa: PLC0415
        _orthogonal_bridge_diversity_evidence,
        _orthogonal_post_repair_preflight_summary,
        _verify_nsga_code_root,
    )

    warm_start = warm_start.resolve(strict=True)
    contract_path = warm_start.with_name(
        "fixed_n1_6_orthogonal_warm_contract.json"
    ).resolve(strict=True)
    contract = read_json(contract_path)
    root = contract_path.parent.resolve(strict=True)
    selected = contract.get("selected_provenance")
    sources = contract.get("source_evidence")
    snapshots = contract.get("source_snapshots")
    artifacts = contract.get("selected_artifact_snapshots")
    preflight = contract.get("post_repair_preflight")
    roles = ("normalized", "thermal", "bridge", "fixed6")
    role_set = set(roles)
    thermal_names = tuple(
        f"temperature_robust_limit:{target}"
        for target in TEMPERATURE_TARGETS
    )
    llt_names = ("Llt_robust_band", "Llt_ensemble_disagreement")
    resonance_name = "half_magnetizing_resonance_minimum"
    constraint_names = (
        "Llt_robust_band",
        *thermal_names,
        "analytical_flux_density_limit",
        "decoded_space_shrink",
        "minimum_physical_insulation",
        "strict_full_density_support",
        "Llt_ensemble_disagreement",
        "core_group_manufacturability_limit",
        resonance_name,
        "exterior_width_limit",
        "exterior_length_limit",
        "exterior_height_limit",
    )
    required_anchors = {
        "normalized": 57059,
        "thermal": 59279,
        "fixed6": 59035,
    }
    expected_profiles = {
        "normalized": None,
        "thermal": (
            "thermal-crossover-core1c-res250-p320-g600-warm64-v1",
            "thermal-crossover-core1c-v1",
        ),
        "bridge": (
            "fixed-n1-6-thermal-bridge-core2c-llt0p55-res150-"
            "p320-g600-warm64-v1",
            "fixed-n1-6-thermal-bridge-v1",
        ),
        "fixed6": (
            "fixed-n1-6-resonance-llt-core4c-res100-p320-g600-warm64-v1",
            "fixed-n1-6-resonance-llt-v1",
        ),
    }
    expected_controller_shas = {
        "normalized": (
            "0b48bcf01e691be0ee217700f173d6cf42f80902f5457251684bf19ac0de7e86"
        ),
        "thermal": (
            "ffc451375af92ac13910f59c9dd6888a0e1d595e6767ba41f6927cee3bccc71d"
        ),
        "bridge": (
            "313c017e4372cff09b0f7429e0b86456261f28577139215e56784f55eda9b878"
        ),
        "fixed6": (
            "fbc1c784d06e2a7b3fc4b60fab0db665e8707fafb658451fd7a4e22a6762f2a2"
        ),
    }
    expected_generation = {
        "constraint_version": (
            "mft-tier1-envelope-1200x1200x750-res15k-"
            "t110all11-core4-5t-lmhalf-v4"
        ),
        "hard_spec_sha256": (
            "1a53578a25b96cbdb0cd0f06aa2a18574d2877e0340d860c35a17ee5496c0b59"
        ),
        "source_model_manifest_sha256": (
            "41497047d12f2ff88f0f6e067619015adea90016580501ed6a143dc79763ad91"
        ),
        "deployment_model_manifest_sha256": (
            "0408d7141c68d584b480a3aef6fa05126fc99e3ef171ace1051ee6df6925fe7f"
        ),
        "temperature_constraint_contract_sha256": (
            "976b78d647245071769fa79e28055c498f6dc569925c92861af2f4128faae87f"
        ),
        "hard_spec": contract.get("hard_spec"),
        "nsga_code_revision": (
            "7c832f7f78f92ee2d99b2d37e14c3131f07d9cae"
        ),
    }
    expected_category_counts = {
        "thermal_llt_pass_branch": 24,
        "bridge_near_branch": 37,
        "resonance_llt_pass_branch": 3,
    }
    expected_role_counts = {
        "normalized": 12, "thermal": 12, "bridge": 37, "fixed6": 3,
    }
    expected_branch_role_counts = {
        "thermal_llt_pass_branch": {"normalized": 12, "thermal": 12},
        "bridge_near_branch": {"bridge": 37},
        "resonance_llt_pass_branch": {"fixed6": 3},
    }

    def is_sha(value: object) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def sealed_json(reference: object) -> dict:
        if not isinstance(reference, dict):
            raise RuntimeError("orthogonal evidence reference is missing")
        relative = Path(str(reference.get("path") or ""))
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"..", ""} for part in relative.parts)
        ):
            raise RuntimeError("orthogonal evidence path is unsafe")
        root_io = _local_io_path(root).resolve(strict=True)
        path_io = (root_io / relative).resolve(strict=True)
        if path_io != root_io and root_io not in path_io.parents:
            raise RuntimeError("orthogonal evidence escaped handoff")
        payload = path_io.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if (
            digest != reference.get("sha256")
            or len(payload) != reference.get("size_bytes")
        ):
            raise RuntimeError("orthogonal evidence byte seal mismatch")
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("orthogonal evidence is not an object")
        return value

    try:
        coordinates = np.load(_local_io_path(warm_start), allow_pickle=False)
        warm_ok = bool(
            coordinates.shape == (64, 25)
            and coordinates.dtype == np.dtype("float64")
            and np.isfinite(coordinates).all()
            and (coordinates >= 0.0).all()
            and (coordinates <= 1.0).all()
        )
        coordinate_shas = [
            canonical_sha([float(value) for value in row])
            for row in coordinates
        ] if warm_ok else []
    except (OSError, TypeError, ValueError):
        warm_ok = False
        coordinate_shas = []

    warm = contract.get("warm_start")
    minimum = contract.get("bridge_minimum_terminal_contract")
    if (
        contract.get("schema_version")
        != "mft-tier1-fixed-n1-6-orthogonal-warm-handoff-v1"
        or contract.get("selection_contract")
        != (
            "24_thermal_and_llt_pass_normalized12_thermal12_plus_"
            "37_latest_bridge_near_plus_3_resonance_and_llt_pass_"
            "fixed6_interleaved_v1"
        )
        or not isinstance(contract.get("hard_spec"), dict)
        or canonical_sha(contract["hard_spec"])
        != contract.get("hard_spec_sha256")
        or contract.get("hard_spec_sha256")
        != expected_generation["hard_spec_sha256"]
        or contract.get("constraint_version")
        != expected_generation["constraint_version"]
        or tuple(contract.get("constraint_names") or ()) != constraint_names
        or canonical_sha(contract.get("constraint_names"))
        != contract.get("constraint_names_sha256")
        or contract.get("nsga_code_revision")
        != expected_generation["nsga_code_revision"]
        or contract.get("nsga_code_root_revision")
        != expected_generation["nsga_code_revision"]
        or contract.get("nsga_code_root_revision_verified") is not True
        or contract.get("nsga_code_root_clean_verified") is not True
        or contract.get("nsga_decoder_relative_path")
        != "module/input_parameter_260706.py"
        or not is_sha(contract.get("nsga_decoder_source_sha256"))
        or contract.get("fixed_primary_turns") != 6
        or contract.get("fixed_primary_turns_scope")
        != "runner_repair_after_authenticated_inverse_coordinate_handoff"
        or contract.get("authoritative_terminal_G") != "physical_unscaled"
        or contract.get("optimizer_scale_scope")
        != "search_pressure_only_physical_G_unchanged"
        or contract.get("optimizer_resonance_scale_Hz") != 150.0
        or contract.get("optimizer_Llt_scale_uH") != 0.25
        or contract.get("optimizer_Llt_disagreement_scale_uH") != 0.5
        or contract.get("optimizer_all_active_thermal_scale_C") != 2.0
        or tuple(contract.get("optimizer_all_active_thermal_constraints") or ())
        != thermal_names
        or contract.get("optimizer_all_active_thermal_side_activation")
        != "finite_N2_side_gt_0_else_physical_negative_BIG"
        or contract.get("soft_axis_pressure_scope")
        != "disabled_until_stage_feasible"
        or contract.get("acquisition_ranking_contract") is not None
        or contract.get("hard_constraint_mutation") is not False
        or contract.get("category_counts") != expected_category_counts
        or contract.get("source_role_counts") != expected_role_counts
        or contract.get("branch_source_role_counts")
        != expected_branch_role_counts
        or not isinstance(contract.get("branch_candidate_supply_counts"), dict)
        or contract["branch_candidate_supply_counts"].get(
            "normalized_thermal_llt_pass_unique"
        ) < 12
        or contract["branch_candidate_supply_counts"].get(
            "thermal_thermal_llt_pass_unique"
        ) < 12
        or contract["branch_candidate_supply_counts"].get(
            "bridge_near_unique"
        ) < 37
        or contract["branch_candidate_supply_counts"].get(
            "fixed6_resonance_llt_pass_unique"
        ) != 3
        or contract.get("resonance_llt_supply_policy")
        != (
            "seal_all_3_authenticated_unique_fixed6_pass_points_then_fill_"
            "remaining_pool_with_latest_bridge_near_points_v1"
        )
        or contract.get("quota_revision_evidence") != {
            "initial_planned_counts": {
                "thermal_llt_pass_branch": 24,
                "bridge_near_branch": 16,
                "resonance_llt_pass_branch": 24,
            },
            "authenticated_recount_fixed6_resonance_llt_pass_unique": 3,
            "authenticated_recount_bridge_near_unique": contract[
                "branch_candidate_supply_counts"
            ]["bridge_near_unique"],
            "revised_counts": expected_category_counts,
            "reason": (
                "captured_fixed6_source_has_only_3_authenticated_unique_"
                "simultaneous_resonance_and_Llt_pass_points; preserve_all_3_"
                "and_fill_21_point_shortfall_from_mature_bridge_source"
            ),
            "decoded_sha_uniqueness_preserved_across_all_branches": True,
        }
        or contract.get("required_anchor_tasks") != required_anchors
        or not isinstance(warm, dict)
        or warm.get("filename") != warm_start.name
        or warm.get("sha256") != sha256(_local_io_path(warm_start))
        or warm.get("shape") != [64, 25]
        or warm.get("dtype") != "float64"
        or warm.get("coordinate_contract")
        != "authenticated_decoded_to_unit_then_fixed_n1_6_repair_v1"
        or not warm_ok
        or len(set(coordinate_shas)) != 64
        or not isinstance(selected, list)
        or len(selected) != 64
        or canonical_sha(selected) != contract.get("selected_provenance_sha256")
        or [item.get("unit_coordinate_sha256") for item in selected]
        != coordinate_shas
        or len({item.get("decoded_params_sha256") for item in selected}) != 64
        or not isinstance(sources, dict)
        or set(sources) != role_set
        or not isinstance(snapshots, dict)
        or set(snapshots) != role_set
        or not isinstance(artifacts, list)
        or len(artifacts) != 64
        or canonical_sha(artifacts)
        != contract.get("selected_artifact_snapshots_sha256")
        or not isinstance(minimum, dict)
        or minimum.get("minimum_authenticated_terminal_count") != 32
        or int(minimum.get("observed_authenticated_terminal_count", -1)) < 32
        or minimum.get("latest_authenticated_head_required") is not True
        or minimum.get("satisfied") is not True
        or contract.get("rolling_target") != 32
        or contract.get("population") != 320
        or contract.get("max_generations") != 600
        or contract.get("inference_threads") != 8
        or contract.get("seed_start") != 1_907_198_000
        or contract.get("task_priority") != -1
        or any(
            contract.get(field) is not False
            for field in (
                "launch_performed", "scheduler_write_performed",
                "fea_submission_approved", "fea_submission_performed",
                "aedt_used", "production_eligible",
                "automatic_promotion_allowed",
            )
        )
    ):
        raise RuntimeError("fixed-N1=6 orthogonal warm handoff mismatch")

    terminal_references: dict[str, set[tuple[int, int, str, str]]] = {}
    for role in roles:
        source = sources[role]
        profile = source.get("search_profile")
        expected_profile = expected_profiles[role]
        profile_ok = (
            profile is None if expected_profile is None
            else isinstance(profile, dict)
            and profile.get("namespace") == expected_profile[0]
            and profile.get("variant") == expected_profile[1]
        )
        role_snapshots = snapshots[role]
        if (
            source.get("role") != role
            or source.get("generation_identity") != expected_generation
            or source.get("controller_source_sha256")
            != expected_controller_shas[role]
            or not profile_ok
            or not isinstance(role_snapshots, dict)
            or set(role_snapshots) != {"index", "pointer", "status"}
        ):
            raise RuntimeError("orthogonal source identity mismatch")
        loaded = {
            kind: sealed_json(role_snapshots[kind])
            for kind in ("index", "pointer", "status")
        }
        for kind, schema in {
            "index": "mft-tier1-slurm-rolling-index-v1",
            "pointer": "mft-tier1-slurm-model-pointer-v1",
            "status": "mft-tier1-slurm-rolling-status-v1",
        }.items():
            source_reference = source.get(kind)
            snapshot_reference = role_snapshots[kind]
            if (
                loaded[kind].get("schema_version") != schema
                or not isinstance(source_reference, dict)
                or snapshot_reference.get("sha256")
                != source_reference.get("sha256")
                or snapshot_reference.get("source_sha256")
                != source_reference.get("sha256")
            ):
                raise RuntimeError("orthogonal source snapshot mismatch")
        index = loaded["index"]
        pointer = loaded["pointer"]
        status = loaded["status"]
        current = pointer.get("current")
        if (
            not isinstance(current, dict)
            or index.get("model_pointer", {}).get("sha256")
            != role_snapshots["pointer"]["sha256"]
            or index.get("status", {}).get("sha256")
            != role_snapshots["status"]["sha256"]
            or index.get("active_cohort_id") != source.get("cohort_id")
            or status.get("cohort_id") != source.get("cohort_id")
            or current.get("cohort_id") != source.get("cohort_id")
            or current.get("controller_source_sha256")
            != source.get("controller_source_sha256")
            or current.get("search_profile") != profile
            or any(
                current.get(key) != expected_generation[key]
                for key in (
                    "constraint_version", "hard_spec_sha256",
                    "source_model_manifest_sha256",
                    "deployment_model_manifest_sha256",
                    "temperature_constraint_contract_sha256",
                    "nsga_code_revision",
                )
            )
            or any(
                current.get(field) is not False
                and not (
                    role == "normalized" and field not in current
                    and field in {
                        "fea_submission_performed", "aedt_used",
                        "automatic_promotion_allowed",
                    }
                )
                for field in (
                    "production_eligible", "fea_submission_approved",
                    "fea_submission_performed", "aedt_used",
                    "automatic_promotion_allowed",
                )
            )
        ):
            raise RuntimeError("orthogonal captured canonical head mismatch")
        terminals = status.get("terminal_results")
        if not isinstance(terminals, list) or not terminals:
            raise RuntimeError("orthogonal source has no terminal results")
        references = set()
        result_shas = []
        seed_status_shas = []
        for terminal in terminals:
            result_ref = terminal.get("result") if isinstance(terminal, dict) else None
            status_ref = terminal.get("remote_status") if isinstance(terminal, dict) else None
            if (
                not isinstance(result_ref, dict)
                or not isinstance(status_ref, dict)
                or terminal.get("authenticated") is not True
                or terminal.get("terminal_state") != "completed"
                or terminal.get("scheduler_state") != "completed"
                or int(terminal.get("scheduler_exit_code", -1)) != 0
                or not is_sha(result_ref.get("sha256"))
                or not is_sha(status_ref.get("sha256"))
            ):
                raise RuntimeError("orthogonal source terminal is unauthenticated")
            reference = (
                int(terminal["task_id"]), int(terminal["seed"]),
                result_ref["sha256"], status_ref["sha256"],
            )
            if reference in references:
                raise RuntimeError("orthogonal source terminal is duplicated")
            references.add(reference)
            result_shas.append(result_ref["sha256"])
            seed_status_shas.append(status_ref["sha256"])
        if (
            sorted(result_shas) != source.get("terminal_result_sha256")
            or sorted(seed_status_shas)
            != source.get("terminal_seed_status_sha256")
            or canonical_sha(sorted(result_shas))
            != source.get("terminal_result_set_sha256")
            or canonical_sha(sorted(seed_status_shas))
            != source.get("terminal_seed_status_set_sha256")
            or int(source.get("terminal_result_count", -1)) != len(terminals)
        ):
            raise RuntimeError("orthogonal terminal evidence set mismatch")
        terminal_references[role] = references

    bridge_terminal_count = int(sources["bridge"]["terminal_result_count"])
    if (
        bridge_terminal_count
        != int(minimum["observed_authenticated_terminal_count"])
        or minimum.get("captured_status_sha256")
        != sources["bridge"]["status"]["sha256"]
    ):
        raise RuntimeError("orthogonal bridge minimum evidence mismatch")

    decoded_params = []
    bridge_rows_for_diversity = []
    for index, artifact in enumerate(artifacts):
        provenance = selected[index]
        role = provenance.get("source_role")
        if (
            not isinstance(artifact, dict)
            or artifact.get("warm_index") != index
            or artifact.get("source_role") != role
            or artifact.get("cohort_id") != provenance.get("cohort_id")
            or artifact.get("task_id") != provenance.get("task_id")
            or artifact.get("seed") != provenance.get("seed")
            or (
                int(provenance.get("task_id", -1)),
                int(provenance.get("seed", -1)),
                provenance.get("result_sha256"),
                provenance.get("seed_status_sha256"),
            ) not in terminal_references.get(str(role), set())
        ):
            raise RuntimeError("orthogonal selected artifact identity mismatch")
        result = sealed_json(artifact.get("result"))
        seed_status = sealed_json(artifact.get("seed_status"))
        if (
            artifact["result"].get("sha256")
            != provenance.get("result_sha256")
            or artifact["seed_status"].get("sha256")
            != provenance.get("seed_status_sha256")
        ):
            raise RuntimeError("orthogonal selected artifact SHA mismatch")
        candidates = (
            result.get("next_target_fea_batch_plan") or {}
        ).get("candidates")
        matches = [
            candidate for candidate in candidates or []
            if isinstance(candidate, dict)
            and candidate.get("decoded_params_sha256")
            == provenance.get("decoded_params_sha256")
        ]
        if len(matches) != 1:
            raise RuntimeError("orthogonal selected candidate is not unique")
        candidate = matches[0]
        params = candidate.get("decoded_params")
        constraints = candidate.get("constraint_G")
        generation = sources[str(role)]["generation_identity"]
        if (
            not isinstance(params, dict)
            or not isinstance(constraints, dict)
            or set(constraints) != set(constraint_names)
            or len(constraints) != len(constraint_names)
            or result.get("schema_version")
            != "mft-tier1-corrected-search-seed-v1"
            or int(result.get("seed", -1)) != int(provenance["seed"])
            or result.get("nsga_code_revision")
            != contract["nsga_code_revision"]
            or result.get("constraint_version")
            != contract["constraint_version"]
            or result.get("hard_spec") != contract["hard_spec"]
            or result.get("hard_spec_sha256")
            != contract["hard_spec_sha256"]
            or result.get("model_manifest_sha256")
            != generation["deployment_model_manifest_sha256"]
            or result.get("temperature_constraint_contract_sha256")
            != generation["temperature_constraint_contract_sha256"]
            or tuple(result.get("constraint_names") or ()) != constraint_names
            or seed_status.get("schema_version")
            != "mft-tier1-slurm-seed-status-v1"
            or seed_status.get("state") != "completed"
            or int(seed_status.get("exit_code", -1)) != 0
            or int(seed_status.get("seed", -1)) != int(provenance["seed"])
            or str(seed_status.get("task_id")) != str(provenance["task_id"])
            or seed_status.get("cohort_id") != provenance["cohort_id"]
            or seed_status.get("result_sha256")
            != provenance["result_sha256"]
            or seed_status.get("bundle_manifest_sha256")
            != sources[str(role)]["bundle_manifest_sha256"]
            or canonical_sha(params) != provenance["decoded_params_sha256"]
            or int(params.get("N1", -1)) != 6
            or provenance.get("original_N1") != 6
            or provenance.get("fixed_N1_repair_required") is not False
            or any(
                result.get(field) is not False
                for field in (
                    "production_eligible", "fea_submission_approved",
                    "automatic_promotion_allowed",
                )
            )
            or any(
                field in result and result.get(field) is not False
                for field in ("fea_submission_performed", "aedt_used")
            )
            or any(
                seed_status.get(field) is not False
                for field in (
                    "production_eligible", "fea_submission_approved",
                    "fea_submission_performed",
                )
            )
            or any(
                field in seed_status and seed_status.get(field) is not False
                for field in ("aedt_used", "automatic_promotion_allowed")
            )
        ):
            raise RuntimeError("orthogonal selected artifact semantics mismatch")
        values = {name: float(constraints[name]) for name in constraint_names}
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError("orthogonal selected constraint is not finite")
        thermal_values = [values[name] for name in thermal_names]
        thermal_positive = [max(value, 0.0) for value in thermal_values]
        llt_pass = all(values[name] <= 1e-9 for name in llt_names)
        resonance_pass = values[resonance_name] <= 1e-9
        other_pass = all(
            value <= 1e-9 for name, value in values.items()
            if name not in {*thermal_names, *llt_names, resonance_name}
        )
        if (
            provenance.get("all_thermal_pass")
            is not all(value <= 1e-9 for value in thermal_values)
            or provenance.get("Llt_pass") is not llt_pass
            or provenance.get("resonance_pass") is not resonance_pass
            or provenance.get("orthogonal_other_pass") is not other_pass
            or not math.isclose(
                float(provenance["Llt_robust_G_uH"]),
                values["Llt_robust_band"], rel_tol=0.0, abs_tol=1e-12,
            )
            or not math.isclose(
                float(provenance["Llt_disagreement_G_uH"]),
                values["Llt_ensemble_disagreement"],
                rel_tol=0.0, abs_tol=1e-12,
            )
            or not math.isclose(
                float(provenance["resonance_G_Hz"]),
                values[resonance_name], rel_tol=0.0, abs_tol=1e-12,
            )
            or not math.isclose(
                float(provenance["thermal_positive_max_C"]),
                max(thermal_positive), rel_tol=0.0, abs_tol=1e-12,
            )
            or not math.isclose(
                float(provenance["thermal_positive_sum_C"]),
                sum(thermal_positive), rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise RuntimeError("orthogonal provenance constraint mismatch")
        category = provenance.get("category")
        if (
            category == "thermal_llt_pass_branch"
            and not (provenance["all_thermal_pass"] and llt_pass and other_pass)
        ) or (
            category == "resonance_llt_pass_branch"
            and not (resonance_pass and llt_pass and other_pass)
        ) or (
            category == "bridge_near_branch" and not other_pass
        ) or category not in expected_category_counts:
            raise RuntimeError("orthogonal branch eligibility mismatch")
        if category == "bridge_near_branch":
            bridge_rows_for_diversity.append({
                "coordinate": coordinates[index],
                "constraint_G": constraints,
                "decoded_params_sha256": provenance[
                    "decoded_params_sha256"
                ],
            })
        decoded_params.append(params)

    expected_categories = []
    for index in range(37):
        if index < 24:
            expected_categories.append("thermal_llt_pass_branch")
        expected_categories.append("bridge_near_branch")
        if index < 3:
            expected_categories.append("resonance_llt_pass_branch")
    actual_category_counts = {
        name: sum(item.get("category") == name for item in selected)
        for name in expected_category_counts
    }
    actual_role_counts = {
        role: sum(item.get("source_role") == role for item in selected)
        for role in roles
    }
    actual_branch_role_counts = {
        category: {
            role: sum(
                item.get("category") == category
                and item.get("source_role") == role
                for item in selected
            )
            for role in expected
        }
        for category, expected in expected_branch_role_counts.items()
    }
    anchors = contract.get("anchors")
    bridge_anchor = contract.get("bridge_anchor")
    fixed6_selected = [
        item for item in selected
        if item.get("category") == "resonance_llt_pass_branch"
    ]
    fixed6_preservation = contract.get(
        "fixed6_resonance_llt_preservation"
    )
    fixed6_preservation_unsigned = dict(fixed6_preservation or {})
    fixed6_preservation_sha = fixed6_preservation_unsigned.pop("sha256", None)
    if (
        [item.get("warm_index") for item in selected] != list(range(64))
        or [item.get("category") for item in selected] != expected_categories
        or actual_category_counts != expected_category_counts
        or actual_role_counts != expected_role_counts
        or actual_branch_role_counts != expected_branch_role_counts
        or not isinstance(anchors, dict)
        or set(anchors) != set(required_anchors)
        or any(
            anchors[role] not in selected
            or anchors[role].get("source_role") != role
            or anchors[role].get("task_id") != task
            for role, task in required_anchors.items()
        )
        or bridge_anchor not in selected
        or bridge_anchor.get("source_role") != "bridge"
        or not isinstance(fixed6_preservation, dict)
        or fixed6_preservation.get("authenticated_unique_count") != 3
        or fixed6_preservation.get("selected_unique_count") != 3
        or fixed6_preservation.get("selected_task_ids") != sorted(
            int(item["task_id"]) for item in fixed6_selected
        )
        or fixed6_preservation.get("selected_decoded_params_sha256")
        != sorted(item["decoded_params_sha256"] for item in fixed6_selected)
        or fixed6_preservation.get("required_anchor_task_id") != 59035
        or fixed6_preservation.get("required_anchor_preserved") is not True
        or fixed6_preservation.get(
            "all_authenticated_unique_points_preserved"
        ) is not True
        or canonical_sha(fixed6_preservation_unsigned)
        != fixed6_preservation_sha
    ):
        raise RuntimeError("orthogonal branch provenance mismatch")

    recomputed_bridge_diversity = _orthogonal_bridge_diversity_evidence(
        bridge_rows_for_diversity
    )
    if recomputed_bridge_diversity != contract.get(
        "bridge_diversity_evidence"
    ):
        raise RuntimeError("orthogonal bridge diversity evidence mismatch")

    if nsga_code_root is None:
        raise RuntimeError("orthogonal handoff requires authenticated NSGA code")
    code_root = nsga_code_root.resolve(strict=True)
    code_identity = _verify_nsga_code_root(code_root)
    if (
        code_identity["revision"] != contract["nsga_code_root_revision"]
        or code_identity["clean"] is not True
        or code_identity["decoder_relative_path"]
        != contract["nsga_decoder_relative_path"]
        or code_identity["decoder_sha256"]
        != contract["nsga_decoder_source_sha256"]
    ):
        raise RuntimeError("orthogonal NSGA code identity mismatch")
    decoded_coordinates = _fixed_bridge_decode_coordinates(
        decoded_params, code_root,
    )
    if (
        decoded_coordinates.shape != (64, 25)
        or not np.isfinite(decoded_coordinates).all()
        or not np.allclose(
            decoded_coordinates, coordinates, rtol=0.0, atol=1e-12,
        )
    ):
        raise RuntimeError("orthogonal decoded coordinate mismatch")

    if (
        not isinstance(preflight, dict)
        or preflight.get("schema_version")
        != "mft-tier1-fixed-n1-6-orthogonal-post-repair-preflight-v1"
        or preflight.get("required_for_deployment") is not True
        or preflight.get("verified") is not True
    ):
        raise RuntimeError("orthogonal post-repair preflight is missing")
    preflight_result = sealed_json(preflight.get("result"))
    preflight_summary = _orthogonal_post_repair_preflight_summary(
        contract, preflight_result,
    )
    if (
        preflight_summary != preflight.get("verified_summary")
        or canonical_sha(preflight_summary)
        != preflight.get("verified_summary_sha256")
        or preflight_summary.get("source_warm_sha256") != warm["sha256"]
        or preflight_summary.get("optimizer_Llt_scale_uH") != 0.25
        or preflight_summary.get("optimizer_Llt_disagreement_scale_uH") != 0.5
        or preflight_summary.get("optimizer_all_active_thermal_scale_C") != 2.0
        or preflight_summary.get("soft_axis_pressure_scope")
        != "disabled_until_stage_feasible"
    ):
        raise RuntimeError("orthogonal post-repair preflight mismatch")
    return contract_path, sha256(contract_path)


def validate_warm_handoff_contract(
    warm_start: Path, nsga_code_root: Path | None = None,
) -> tuple[Path, str] | None:
    if SEARCH_VARIANT in ANCHOR_ISLAND_BY_VARIANT:
        try:
            from tier1_fixed_n1_6_anchor_islands_warm_handoff import (
                validate_handoff_contract,
            )
        except ImportError:  # pragma: no cover - repository import path
            from tools.tier1_fixed_n1_6_anchor_islands_warm_handoff import (
                validate_handoff_contract,
            )
        return validate_handoff_contract(warm_start, nsga_code_root)
    if SEARCH_VARIANT in DEEP_CROSSOVER_BY_VARIANT:
        if DEEP_CROSSOVER_ISLAND.fixed_primary_turns == 6:
            try:
                from tier1_fixed_n1_6_anchor_islands_warm_handoff import (
                    validate_handoff_contract,
                )
            except ImportError:  # pragma: no cover - repository import path
                from tools.tier1_fixed_n1_6_anchor_islands_warm_handoff import (
                    validate_handoff_contract,
                )
            return validate_handoff_contract(warm_start, nsga_code_root)
        from tools.tier1_fixed_n1_5_deep_crossover_warm_handoff import (  # noqa: PLC0415
            validate_contract,
        )

        return validate_contract(warm_start, nsga_code_root)
    if SEARCH_VARIANT == "resonance-focus-250hz-v1":
        return validate_resonance_focus_warm_contract(warm_start)
    if SEARCH_VARIANT in {
        "thermal-crossover-core1c-v1",
        "thermal-crossover-balanced4c-v1",
        "fixed-n1-5-thermal-llt-v1",
        "fixed-n1-6-resonance-llt-v1",
    }:
        return validate_thermal_crossover_warm_contract(warm_start)
    if SEARCH_VARIANT == "fixed-n1-6-thermal-bridge-v1":
        return validate_fixed_n1_6_bridge_warm_contract(
            warm_start, nsga_code_root,
        )
    if SEARCH_VARIANT == "fixed-n1-6-orthogonal-bridge-v1":
        return validate_fixed_n1_6_orthogonal_warm_contract(
            warm_start, nsga_code_root,
        )
    if SEARCH_VARIANT == "fixed-n1-5-size-rx-core-v1":
        from tools.tier1_fixed_n1_5_size_rx_core_warm_handoff import (  # noqa: PLC0415
            validate_contract,
        )

        return validate_contract(warm_start, nsga_code_root)
    return None


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _feedback_contract(path: Path) -> tuple[str, dict, str]:
    spec = importlib.util.spec_from_file_location(
        f"tier1_feedback_contract_{sha256(path)[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load feedback constraint contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        str(module.CONSTRAINT_VERSION),
        dict(module.HARD_SPEC),
        str(module.EXPECTED_NSGA_REVISION),
    )


def _copy_tree(source: Path, destination: Path) -> None:
    source_io = _local_io_path(source)
    destination_io = _local_io_path(destination)
    for path in sorted(source_io.rglob("*")):
        relative = path.relative_to(source_io)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        target = destination_io / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file() and path.suffix.lower() not in {".pyc", ".pyo"}:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _source_inventory(root: Path, directories: tuple[str, ...]) -> dict[str, str]:
    inventory = {}
    for directory in directories:
        source = root / directory
        if not source.is_dir():
            raise FileNotFoundError(source)
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(root)
            if (
                not path.is_file()
                or any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts)
                or path.suffix.lower() in {".pyc", ".pyo"}
            ):
                continue
            inventory[relative.as_posix()] = sha256(path)
    return inventory


def _deployment_files(bundle: Path) -> dict[str, dict]:
    files = {}
    bundle_io = _local_io_path(bundle)
    for path in sorted((bundle_io / "artifacts").rglob("*")):
        if path.is_file():
            relative = path.relative_to(bundle_io).as_posix()
            files[relative] = {"sha256": sha256(path), "size": path.stat().st_size}
    return files


def _copy_warm_handoff_evidence(
    contract_path: Path, warm_destination_root: Path,
) -> list[Path]:
    """Preserve every content-addressed bridge evidence file in the bundle."""

    contract_path = contract_path.resolve(strict=True)
    contract = read_json(contract_path)
    evidence_schemas = {
        "mft-tier1-fixed-n1-6-bridge-warm-handoff-v1",
        "mft-tier1-fixed-n1-6-orthogonal-warm-handoff-v1",
        "mft-tier1-fixed-n1-6-anchor-islands-warm-handoff-v1",
        "mft-tier1-fixed-n1-5-size-rx-core-warm-handoff-v1",
    }
    if contract.get("schema_version") not in evidence_schemas:
        return []
    references = []
    sources = contract.get("source_snapshots")
    selected = contract.get("selected_artifact_snapshots")
    if not isinstance(sources, dict) or not isinstance(selected, list):
        raise RuntimeError("bridge evidence references are missing")
    for role, role_snapshots in sources.items():
        if not isinstance(role_snapshots, dict):
            raise RuntimeError("bridge source snapshot role is missing")
        references.extend(role_snapshots.values())
    for artifact in selected:
        if not isinstance(artifact, dict):
            raise RuntimeError("bridge selected artifact is invalid")
        references.extend((artifact.get("result"), artifact.get("seed_status")))
    preflight = contract.get("post_repair_preflight")
    if contract.get("schema_version") != (
        "mft-tier1-fixed-n1-6-anchor-islands-warm-handoff-v1"
    ):
        if not isinstance(preflight, dict):
            raise RuntimeError("bridge post-repair preflight is missing")
        references.append(preflight.get("result"))

    contract_root = contract_path.parent.resolve(strict=True)
    copied = {}
    for reference in references:
        if not isinstance(reference, dict):
            raise RuntimeError("bridge evidence file reference is invalid")
        relative = Path(str(reference.get("path") or ""))
        if relative.is_absolute() or not relative.parts:
            raise RuntimeError("bridge evidence path is not relative")
        if any(part in {"..", ""} for part in relative.parts):
            raise RuntimeError("bridge evidence path is unsafe")
        contract_root_io = _local_io_path(contract_root).resolve(strict=True)
        source = (contract_root_io / relative).resolve(strict=True)
        if source != contract_root_io and contract_root_io not in source.parents:
            raise RuntimeError("bridge evidence escaped handoff directory")
        digest = sha256(source)
        if (
            digest != reference.get("sha256")
            or source.stat().st_size != reference.get("size_bytes")
        ):
            raise RuntimeError("bridge evidence source seal mismatch")
        destination = warm_destination_root / relative
        prior = copied.get(relative.as_posix())
        if prior is not None:
            if prior != digest:
                raise RuntimeError("bridge evidence relative-path collision")
            continue
        destination_io = _local_io_path(destination)
        destination_io.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_io)
        if (
            sha256(destination_io) != digest
            or destination_io.stat().st_size != source.stat().st_size
        ):
            raise RuntimeError("bridge evidence bundle copy mismatch")
        copied[relative.as_posix()] = digest
    return [warm_destination_root / Path(relative) for relative in sorted(copied)]


def _capture_validated_warm_artifacts(
    warm_start: Path, contract_reference: tuple[Path, str] | None,
) -> dict:
    """Capture validated bytes once so prepare never rereads a moving handoff."""

    warm_start = warm_start.resolve(strict=True)
    captured = {}
    contract = None
    contract_sha = None
    if contract_reference is not None:
        contract_path = contract_reference[0].resolve(strict=True)
        contract_payload = _local_io_path(contract_path).read_bytes()
        contract_sha = hashlib.sha256(contract_payload).hexdigest()
        if contract_sha != contract_reference[1]:
            raise RuntimeError("warm contract changed after validation")
        contract = json.loads(contract_payload.decode("utf-8"))
        if not isinstance(contract, dict):
            raise RuntimeError("warm contract is not a JSON object")
        captured[contract_path.name] = contract_payload

    warm_payload = _local_io_path(warm_start).read_bytes()
    warm_sha = hashlib.sha256(warm_payload).hexdigest()
    if (
        isinstance(contract, dict)
        and isinstance(contract.get("warm_start"), dict)
        and contract["warm_start"].get("sha256") != warm_sha
    ):
        raise RuntimeError("warm bytes changed after contract validation")
    if warm_start.name in captured and captured[warm_start.name] != warm_payload:
        raise RuntimeError("warm artifact filename collision")
    captured[warm_start.name] = warm_payload

    evidence_relatives = []
    if (
        isinstance(contract, dict)
        and contract.get("schema_version") in {
            "mft-tier1-fixed-n1-6-bridge-warm-handoff-v1",
            "mft-tier1-fixed-n1-6-orthogonal-warm-handoff-v1",
            "mft-tier1-fixed-n1-6-anchor-islands-warm-handoff-v1",
            "mft-tier1-fixed-n1-5-size-rx-core-warm-handoff-v1",
        }
    ):
        references = []
        for role_snapshots in contract["source_snapshots"].values():
            references.extend(role_snapshots.values())
        for artifact in contract["selected_artifact_snapshots"]:
            references.extend(
                (artifact["result"], artifact["seed_status"])
            )
        if "post_repair_preflight" in contract:
            references.append(contract["post_repair_preflight"]["result"])
        contract_root = contract_reference[0].parent.resolve(strict=True)
        for reference in references:
            relative = Path(str(reference["path"]))
            if relative.is_absolute() or not relative.parts:
                raise RuntimeError("warm evidence path is not relative")
            if any(part in {"..", ""} for part in relative.parts):
                raise RuntimeError("warm evidence path is unsafe")
            contract_root_io = _local_io_path(contract_root).resolve(strict=True)
            source = (contract_root_io / relative).resolve(strict=True)
            if source != contract_root_io and contract_root_io not in source.parents:
                raise RuntimeError("warm evidence escaped contract directory")
            payload = source.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if (
                digest != reference["sha256"]
                or len(payload) != int(reference["size_bytes"])
            ):
                raise RuntimeError("warm evidence changed after validation")
            relative_text = relative.as_posix()
            prior = captured.get(relative_text)
            if prior is not None and prior != payload:
                raise RuntimeError("warm evidence filename collision")
            captured[relative_text] = payload
            evidence_relatives.append(relative_text)
    if (
        isinstance(contract, dict)
        and contract.get("schema_version")
        == "mft-tier1-fixed-n1-5-deep-crossover-warm-handoff-v1"
    ):
        contract_root = contract_reference[0].parent.resolve(strict=True)
        for record in (contract.get("source_records") or {}).values():
            captured_root = Path(str(record.get("captured_root") or ""))
            inventory = record.get("captured_inventory")
            if (
                captured_root.is_absolute() or not captured_root.parts
                or ".." in captured_root.parts or not isinstance(inventory, dict)
                or canonical_sha(inventory)
                != record.get("captured_inventory_sha256")
            ):
                raise RuntimeError("deep crossover capture inventory is invalid")
            for child, expected_sha in inventory.items():
                relative = captured_root / Path(child)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("deep crossover evidence path is unsafe")
                source = (contract_root / relative).resolve(strict=True)
                if contract_root not in source.parents:
                    raise RuntimeError("deep crossover evidence escaped handoff")
                payload = source.read_bytes()
                if hashlib.sha256(payload).hexdigest() != expected_sha:
                    raise RuntimeError("deep crossover evidence SHA mismatch")
                relative_text = relative.as_posix()
                prior = captured.get(relative_text)
                if prior is not None and prior != payload:
                    raise RuntimeError("deep crossover evidence collision")
                captured[relative_text] = payload
                evidence_relatives.append(relative_text)
        anchor = contract.get("authenticated_anchor") or {}
        anchor_root = Path(str(anchor.get("captured_root") or ""))
        anchor_files = anchor.get("files")
        if (
            anchor_root.is_absolute() or not anchor_root.parts
            or ".." in anchor_root.parts or not isinstance(anchor_files, dict)
        ):
            raise RuntimeError("deep crossover anchor inventory is invalid")
        for filename, expected_sha in anchor_files.items():
            relative = anchor_root / Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("deep crossover anchor path is unsafe")
            source = (contract_root / relative).resolve(strict=True)
            if contract_root not in source.parents:
                raise RuntimeError("deep crossover anchor escaped handoff")
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected_sha:
                raise RuntimeError("deep crossover anchor SHA mismatch")
            relative_text = relative.as_posix()
            captured[relative_text] = payload
            evidence_relatives.append(relative_text)
    return {
        "files": captured,
        "warm_start_relative": warm_start.name,
        "warm_start_sha256": warm_sha,
        "contract_relative": (
            contract_reference[0].name
            if contract_reference is not None else None
        ),
        "contract_sha256": contract_sha,
        "evidence_relatives": sorted(set(evidence_relatives)),
    }


def _bridge_bundle_evidence_attested(
    local_bundle: Path, plan: dict, manifest: dict,
) -> bool:
    try:
        evidence_files = manifest.get("warm_handoff_evidence")
        files = manifest.get("files")
        high_value = manifest.get("high_value_sha256")
        contract_relative = manifest.get("warm_handoff_contract")
        if not isinstance(contract_relative, str):
            return False
        local_bundle_io = _local_io_path(local_bundle).resolve(strict=True)
        bundled_contract_path = (
            local_bundle_io / contract_relative
        ).resolve(strict=True)
        if (
            bundled_contract_path != local_bundle_io
            and local_bundle_io not in bundled_contract_path.parents
        ):
            return False
        bundled_contract = read_json(bundled_contract_path)
        if bundled_contract.get("schema_version") == (
            "mft-tier1-fixed-n1-5-deep-crossover-warm-handoff-v1"
        ):
            expected_records = {}
            for record in (
                bundled_contract.get("source_records") or {}
            ).values():
                root = str(record.get("captured_root") or "")
                inventory = record.get("captured_inventory")
                if (
                    not root or not isinstance(inventory, dict)
                    or canonical_sha(inventory)
                    != record.get("captured_inventory_sha256")
                ):
                    return False
                for child, digest in inventory.items():
                    relative = f"artifacts/warm/{root}/{child}"
                    expected_records[relative] = {
                        "sha256": digest,
                        "size": int(files.get(relative, {}).get("size", -1)),
                    }
            anchor = bundled_contract.get("authenticated_anchor") or {}
            anchor_root = str(anchor.get("captured_root") or "")
            anchor_files = anchor.get("files")
            if not anchor_root or not isinstance(anchor_files, dict):
                return False
            for filename, digest in anchor_files.items():
                relative = f"artifacts/warm/{anchor_root}/{filename}"
                expected_records[relative] = {
                    "sha256": digest,
                    "size": int(files.get(relative, {}).get("size", -1)),
                }
            expected_evidence_files = sorted(expected_records)
            return bool(
                sha256(bundled_contract_path)
                == plan.get("warm_handoff_contract_sha256")
                and isinstance(evidence_files, list)
                and evidence_files == expected_evidence_files
                and len(evidence_files) >= 4
                and isinstance(files, dict)
                and isinstance(high_value, dict)
                and all(
                    relative in files
                    and files[relative]["sha256"]
                    == expected_records[relative]["sha256"]
                    and high_value.get(relative)
                    == expected_records[relative]["sha256"]
                    for relative in evidence_files
                )
            )
        references = []
        for role_snapshots in (
            bundled_contract.get("source_snapshots") or {}
        ).values():
            references.extend(role_snapshots.values())
        for artifact in (
            bundled_contract.get("selected_artifact_snapshots") or []
        ):
            references.extend(
                (artifact.get("result"), artifact.get("seed_status"))
            )
        if "post_repair_preflight" in bundled_contract:
            references.append(
                bundled_contract["post_repair_preflight"].get("result")
            )
        expected_records = {}
        for reference in references:
            relative = f"artifacts/warm/{reference['path']}"
            record = {
                "sha256": reference["sha256"],
                "size": int(reference["size_bytes"]),
            }
            if relative in expected_records and expected_records[relative] != record:
                return False
            expected_records[relative] = record
        expected_evidence_files = sorted(expected_records)
        return bool(
            bundled_contract.get("schema_version") in {
                "mft-tier1-fixed-n1-6-bridge-warm-handoff-v1",
                "mft-tier1-fixed-n1-6-orthogonal-warm-handoff-v1",
                "mft-tier1-fixed-n1-6-anchor-islands-warm-handoff-v1",
                "mft-tier1-fixed-n1-5-size-rx-core-warm-handoff-v1",
            }
            and sha256(bundled_contract_path)
            == plan.get("warm_handoff_contract_sha256")
            and isinstance(evidence_files, list)
            and evidence_files == expected_evidence_files
            and len(evidence_files) >= 12
            and isinstance(files, dict)
            and isinstance(high_value, dict)
            and all(
                relative in files
                and files[relative] == expected_records[relative]
                and high_value.get(relative)
                == expected_records[relative]["sha256"]
                for relative in evidence_files
            )
        )
    except (
        AttributeError, KeyError, OSError, TypeError, ValueError,
    ):
        return False


def _anchor_bundle_preflight_attested(
    local_bundle: Path, plan: dict, manifest: dict,
) -> bool:
    variant = (manifest.get("search_profile") or {}).get("variant")
    if variant not in ANCHOR_ISLAND_BY_VARIANT:
        return True
    try:
        relative = manifest.get("anchor_island_preflight")
        expected_sha = manifest.get("anchor_island_preflight_sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(expected_sha, str)
            or plan.get("anchor_island_preflight_sha256") != expected_sha
        ):
            return False
        bundle_root = _local_io_path(local_bundle).resolve(strict=True)
        path = (bundle_root / relative).resolve(strict=True)
        if bundle_root not in path.parents or sha256(path) != expected_sha:
            return False
        try:
            from tier1_n1_6_anchor_island_preflight import validate_preflight
        except ImportError:  # pragma: no cover - repository import path
            from tools.tier1_n1_6_anchor_island_preflight import (
                validate_preflight,
            )
        preflight = validate_preflight(path)
        expected_evidence = sorted(
            "artifacts/preflight/" + record["path"]
            for record in preflight["results"]
        )
        return bool(
            manifest.get("anchor_island_preflight_evidence")
            == expected_evidence
            and all(
                item in (manifest.get("high_value_sha256") or {})
                for item in expected_evidence
            )
        )
    except (
        AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError,
    ):
        return False


def _deep_bundle_preflight_attested(
    local_bundle: Path, plan: dict, manifest: dict,
) -> bool:
    variant = (manifest.get("search_profile") or {}).get("variant")
    if variant not in DEEP_CROSSOVER_BY_VARIANT:
        return True
    try:
        relative = manifest.get("deep_crossover_preflight")
        expected_sha = manifest.get("deep_crossover_preflight_sha256")
        if (
            not isinstance(relative, str) or not isinstance(expected_sha, str)
            or plan.get("deep_crossover_preflight_sha256") != expected_sha
        ):
            return False
        bundle_root = _local_io_path(local_bundle).resolve(strict=True)
        path = (bundle_root / relative).resolve(strict=True)
        if bundle_root not in path.parents or sha256(path) != expected_sha:
            return False
        try:
            from tier1_deep_crossover_preflight import validate_preflight
        except ImportError:  # pragma: no cover - repository import path
            from tools.tier1_deep_crossover_preflight import validate_preflight
        preflight = validate_preflight(path)
        expected_evidence = sorted([
            "artifacts/preflight/" + record["path"]
            for record in preflight["results"]
        ] + [
            "artifacts/preflight/" + preflight[
                "anchor_physical_replay"
            ]["path"]
        ])
        return bool(
            manifest.get("deep_crossover_preflight_evidence")
            == expected_evidence
            and all(
                item in (manifest.get("high_value_sha256") or {})
                for item in expected_evidence
            )
        )
    except (
        AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError,
    ):
        return False


def _local_deployment_integrity(plan: dict) -> bool:
    try:
        local_bundle = Path(plan["local_bundle"]).resolve(strict=True)
        manifest_path = Path(plan["bundle_manifest"]).resolve(strict=True)
        if manifest_path != local_bundle / "bundle_manifest.json":
            return False
        if sha256(manifest_path) != plan.get("bundle_manifest_sha256"):
            return False
        manifest = read_json(manifest_path)
        actual_files = _deployment_files(local_bundle)
        files = manifest.get("files")
        high_value = manifest.get("high_value_sha256")
        warm_relative = manifest.get("warm_start")
        if (
            manifest.get("stable_identity_sha256")
            != plan.get("stable_identity_sha256")
            or files != actual_files
            or plan.get("file_count") != len(actual_files)
            or plan.get("byte_count")
            != sum(record["size"] for record in actual_files.values())
            or not isinstance(high_value, dict)
            or not isinstance(warm_relative, str)
            or warm_relative not in files
            or files[warm_relative].get("sha256")
            != plan.get("warm_start_sha256")
            or high_value.get(warm_relative)
            != plan.get("warm_start_sha256")
            or any(
                relative not in files
                or files[relative].get("sha256") != digest
                for relative, digest in high_value.items()
            )
        ):
            return False
        contract_relative = manifest.get("warm_handoff_contract")
        if contract_relative is not None and (
            not isinstance(contract_relative, str)
            or contract_relative not in files
            or files[contract_relative].get("sha256")
            != plan.get("warm_handoff_contract_sha256")
            or high_value.get(contract_relative)
            != plan.get("warm_handoff_contract_sha256")
        ):
            return False
        if (
            (manifest.get("search_profile") or {}).get("variant") in {
                "fixed-n1-6-thermal-bridge-v1",
                "fixed-n1-6-orthogonal-bridge-v1",
                *(island.variant for island in ANCHOR_ISLANDS),
                *(island.variant for island in DEEP_CROSSOVER_ISLANDS),
                "fixed-n1-5-size-rx-core-v1",
            }
            and not _bridge_bundle_evidence_attested(
                local_bundle, plan, manifest,
            )
        ):
            return False
        if not _anchor_bundle_preflight_attested(
            local_bundle, plan, manifest,
        ):
            return False
        if not _deep_bundle_preflight_attested(
            local_bundle, plan, manifest,
        ):
            return False
        return True
    except (
        AttributeError, KeyError, OSError, TypeError, ValueError,
    ):
        return False


def prepare(
    model_dir: Path, code_root: Path, warm_start: Path, base_plan_path: Path,
    runtime: Path = RUNTIME, anchor_preflight: Path | None = None,
    deep_crossover_preflight: Path | None = None,
) -> dict:
    model_dir = model_dir.resolve(strict=True)
    code_root = code_root.resolve(strict=True)
    warm_start = warm_start.resolve(strict=True)
    warm_contract_reference = validate_warm_handoff_contract(
        warm_start, code_root,
    )
    warm_capture = _capture_validated_warm_artifacts(
        warm_start, warm_contract_reference,
    )
    feedback = (REPO / "tools" / "tier1_resonance_feedback.py").resolve(strict=True)
    runner = (REPO / "tools" / "tier1_slurm_seed_runner.py").resolve(strict=True)
    allowance_contract_source = (
        REPO / "tools" / "tier1_n1_6_anchor_island_contract.py"
    ).resolve(strict=True)
    deep_crossover_contract_source = (
        REPO / "tools" / "tier1_deep_crossover_contract.py"
    ).resolve(strict=True)
    controller = Path(__file__).resolve(strict=True)
    variant_entrypoints = {
        "resonance-focus-250hz-v1": (
            REPO / "tools" / "tier1_resonance_focus_slurm_rolling.py"
        ),
        "thermal-crossover-core1c-v1": (
            REPO / "tools" / "tier1_thermal_crossover_slurm_rolling.py"
        ),
        "thermal-crossover-balanced4c-v1": (
            REPO
            / "tools"
            / "tier1_thermal_balanced_crossover_slurm_rolling.py"
        ),
        "fixed-n1-5-thermal-llt-v1": (
            REPO / "tools" / "tier1_fixed_n1_5_thermal_llt_slurm_rolling.py"
        ),
        "fixed-n1-6-resonance-llt-v1": (
            REPO / "tools" / "tier1_fixed_n1_6_resonance_llt_slurm_rolling.py"
        ),
        "fixed-n1-6-thermal-bridge-v1": (
            REPO
            / "tools"
            / "tier1_fixed_n1_6_thermal_bridge_slurm_rolling.py"
        ),
        "fixed-n1-6-orthogonal-bridge-v1": (
            REPO
            / "tools"
            / "tier1_fixed_n1_6_orthogonal_bridge_slurm_rolling.py"
        ),
        "fixed-n1-5-size-rx-core-v1": (
            REPO
            / "tools"
            / "tier1_fixed_n1_5_size_rx_core_slurm_rolling.py"
        ),
    }
    for island in ANCHOR_ISLANDS:
        variant_entrypoints[island.variant] = (
            REPO
            / "tools"
            / "tier1_fixed_n1_6_anchor_island_slurm_rolling.py"
        )
    for island in DEEP_CROSSOVER_ISLANDS:
        variant_entrypoints[island.variant] = (
            REPO / "tools" / "tier1_deep_crossover_slurm_rolling.py"
        )
    entrypoint = variant_entrypoints.get(SEARCH_VARIANT, controller).resolve(
        strict=True
    )
    constraint_version, hard_spec, revision = _feedback_contract(feedback)
    anchor_preflight_capture = None
    if ANCHOR_ISLAND is not None:
        if anchor_preflight is None:
            raise RuntimeError(
                "anchor-island prepare requires --anchor-preflight"
            )
        try:
            from tier1_n1_6_anchor_island_preflight import validate_preflight
        except ImportError:  # pragma: no cover - repository import path
            from tools.tier1_n1_6_anchor_island_preflight import (
                validate_preflight,
            )
        anchor_preflight = anchor_preflight.resolve(strict=True)
        preflight_contract = validate_preflight(anchor_preflight)
        if (
            preflight_contract.get("warm_start_sha256")
            != warm_capture["warm_start_sha256"]
            or preflight_contract.get("feedback_script_sha256")
            != sha256(feedback)
            or preflight_contract.get("model_manifest_sha256")
            != sha256(model_dir / "manifest.json")
            or preflight_contract.get("nsga_code_revision") != revision
        ):
            raise RuntimeError(
                "anchor-island preflight deployment identity mismatch"
            )
        preflight_root = anchor_preflight.parent.resolve(strict=True)
        captured_preflight = {
            anchor_preflight.name: anchor_preflight.read_bytes(),
        }
        for record in preflight_contract["results"]:
            relative = Path(record["path"])
            source = (preflight_root / relative).resolve(strict=True)
            if preflight_root not in source.parents:
                raise RuntimeError("anchor preflight artifact escaped root")
            payload = source.read_bytes()
            if (
                hashlib.sha256(payload).hexdigest() != record["sha256"]
                or len(payload) != int(record["size_bytes"])
            ):
                raise RuntimeError("anchor preflight artifact changed")
            captured_preflight[relative.as_posix()] = payload
        anchor_preflight_capture = {
            "contract": preflight_contract,
            "contract_sha256": sha256(anchor_preflight),
            "files": captured_preflight,
        }
    deep_preflight_capture = None
    if DEEP_CROSSOVER_ISLAND is not None:
        if deep_crossover_preflight is None:
            raise RuntimeError(
                "deep-crossover prepare requires --deep-crossover-preflight"
            )
        try:
            from tier1_deep_crossover_preflight import validate_preflight
        except ImportError:  # pragma: no cover - repository import path
            from tools.tier1_deep_crossover_preflight import validate_preflight
        deep_crossover_preflight = deep_crossover_preflight.resolve(
            strict=True
        )
        preflight_contract = validate_preflight(deep_crossover_preflight)
        warm_key = (
            "n1_5"
            if DEEP_CROSSOVER_ISLAND.fixed_primary_turns == 5 else "n1_6"
        )
        if (
            (preflight_contract.get("warm_sources") or {}).get(
                warm_key, {}
            ).get("sha256") != warm_capture["warm_start_sha256"]
            or preflight_contract.get("feedback_script_sha256")
            != sha256(feedback)
            or preflight_contract.get("model_manifest_sha256")
            != sha256(model_dir / "manifest.json")
            or preflight_contract.get("nsga_code_revision") != revision
        ):
            raise RuntimeError(
                "deep crossover preflight deployment identity mismatch"
            )
        preflight_root = deep_crossover_preflight.parent.resolve(strict=True)
        captured_preflight = {
            deep_crossover_preflight.name: deep_crossover_preflight.read_bytes(),
        }
        references = [record["path"] for record in preflight_contract["results"]]
        references.append(preflight_contract["anchor_physical_replay"]["path"])
        for relative_text in references:
            relative = Path(relative_text)
            source = (preflight_root / relative).resolve(strict=True)
            if preflight_root not in source.parents:
                raise RuntimeError("deep crossover preflight escaped root")
            payload = source.read_bytes()
            captured_preflight[relative.as_posix()] = payload
        deep_preflight_capture = {
            "contract": preflight_contract,
            "contract_sha256": sha256(deep_crossover_preflight),
            "files": captured_preflight,
        }
    temperature_contract = temperature_constraint_contract(hard_spec)
    temperature_contract_sha = canonical_sha(temperature_contract)
    acquisition_ranking_contract = thermal_crossover_acquisition_contract(
        hard_spec
    )
    acquisition_ranking_contract_sha = (
        acquisition_ranking_contract["sha256"]
        if acquisition_ranking_contract is not None else None
    )
    if subprocess.run(
        [
            "git", "-c", f"safe.directory={code_root.as_posix()}",
            "-C", str(code_root), "rev-parse", "HEAD",
        ],
        text=True, capture_output=True, check=True,
    ).stdout.strip() != revision:
        raise RuntimeError("exact NSGA worktree revision does not match feedback pin")
    _validate_code_temperature_semantics(code_root)
    source_manifest_sha = sha256(model_dir / "manifest.json")
    hard_spec_sha = canonical_sha(hard_spec)
    code_directories = (
        "regression_260707", "module", "material", "verification_params",
    )
    source_code_inventory = _source_inventory(code_root, code_directories)
    source_code_inventory_sha = canonical_sha(source_code_inventory)
    base_plan = read_json(base_plan_path.resolve(strict=True))
    base_manifest_path = Path(base_plan["bundle_manifest"]).resolve(strict=True)
    base_manifest = read_json(base_manifest_path)
    base_remote = str(base_plan["remote_bundle"]).rstrip("/")
    base_generation_relative = base_manifest["source_paths"]["registry_generation"]
    base_generation_remote = f"{base_remote}/{base_generation_relative}"
    base_report_remote = f"{base_generation_remote}/train_report.json"
    base_report_local = Path(
        read_json(Path(base_plan["local_sources"]))[
            f"{base_generation_relative}/train_report.json"
        ]
    )
    stable_identity = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "source_model_manifest_sha256": source_manifest_sha,
        "feedback_script_sha256": sha256(feedback),
        "runner_sha256": sha256(runner),
        "optimizer_allowance_contract_source_sha256": sha256(
            allowance_contract_source
        ),
        "deep_crossover_contract_source_sha256": sha256(
            deep_crossover_contract_source
        ),
        "controller_source_sha256": sha256(controller),
        "variant_entrypoint_sha256": sha256(entrypoint),
        "attestation_schema_version": ATTESTATION_SCHEMA,
        "task_schema_version": TASK_SCHEMA,
        "temperature_constraint_contract_sha256": temperature_contract_sha,
        "nsga_code_revision": revision,
        "source_code_inventory_sha256": source_code_inventory_sha,
        "constraint_version": constraint_version,
        "hard_spec_sha256": hard_spec_sha,
        "warm_start_sha256": warm_capture["warm_start_sha256"],
        "warm_handoff_contract_sha256": (
            warm_contract_reference[1] if warm_contract_reference else None
        ),
        "anchor_island_preflight_sha256": (
            anchor_preflight_capture["contract_sha256"]
            if anchor_preflight_capture is not None else None
        ),
        "deep_crossover_preflight_sha256": (
            deep_preflight_capture["contract_sha256"]
            if deep_preflight_capture is not None else None
        ),
        "search_profile": search_profile(),
        "optimizer_resonance_scale_Hz": OPTIMIZER_RESONANCE_SCALE_HZ,
        "optimizer_core_thermal_scale_C": (
            OPTIMIZER_CORE_THERMAL_SCALE_C
        ),
        "optimizer_Llt_scale_uH": OPTIMIZER_LLT_SCALE_UH,
        "optimizer_all_active_thermal_scale_C": (
            OPTIMIZER_ALL_THERMAL_SCALE_C
        ),
        "acquisition_ranking_contract_sha256": (
            acquisition_ranking_contract_sha
        ),
        "base_bundle_id": base_plan["bundle_id"],
        "base_bundle_manifest_sha256": base_plan["bundle_manifest_sha256"],
        "base_generation_report_sha256": sha256(base_report_local),
    }
    identity_sha = canonical_sha(stable_identity)
    cohort_id = f"res15k-{source_manifest_sha[:10]}-{hard_spec_sha[:10]}-{identity_sha[:10]}"
    bundle_id = f"tier1-{cohort_id}"
    local_bundle = runtime.resolve() / "deployments" / cohort_id / "bundle"
    plan_path = local_bundle.parent / "deployment_plan.json"
    if plan_path.is_file():
        existing = read_json(plan_path)
        if existing.get("stable_identity_sha256") == identity_sha:
            if _local_deployment_integrity(existing):
                return existing
            raise RuntimeError("existing deployment failed local integrity")
        raise RuntimeError("deployment directory collision")
    local_bundle.mkdir(parents=True, exist_ok=False)
    artifacts = local_bundle / "artifacts"
    code_destination = artifacts / "code"
    for directory in code_directories:
        _copy_tree(code_root / directory, code_destination / directory)
    (code_destination / ".source-revision").write_text(revision + "\n", encoding="ascii")
    shutil.copy2(feedback, artifacts / "feedback.py")
    shutil.copy2(runner, artifacts / "seed_runner.py")
    shutil.copy2(
        allowance_contract_source,
        artifacts / "tier1_n1_6_anchor_island_contract.py",
    )
    shutil.copy2(
        deep_crossover_contract_source,
        artifacts / "tier1_deep_crossover_contract.py",
    )
    model_destination = artifacts / "model"
    _copy_tree(model_dir, model_destination)
    deployed_report_path = model_destination / "training_report.json"
    deployed_report = read_json(deployed_report_path)
    deployed_report["base_strict_full_generation"]["path"] = base_generation_remote
    deployed_report["deployment_path_rewrite"] = {
        "source_train_report_sha256": sha256(model_dir / "training_report.json"),
        "policy": "absolute_shared_gpfs_authenticated_base_generation_v1",
        "production_eligible": False,
    }
    atomic_json(deployed_report_path, deployed_report)
    deployed_model_manifest_path = model_destination / "manifest.json"
    deployed_model_manifest = read_json(deployed_model_manifest_path)
    deployed_model_manifest["files"]["training_report.json"] = sha256(
        deployed_report_path
    )
    deployed_model_manifest["source_manifest_sha256"] = source_manifest_sha
    deployed_model_manifest["production_eligible"] = False
    deployed_model_manifest["automatic_promotion_allowed"] = False
    atomic_json(deployed_model_manifest_path, deployed_model_manifest)
    for relative, payload in warm_capture["files"].items():
        atomic_bytes(_local_io_path(artifacts / "warm" / relative), payload)
    if anchor_preflight_capture is not None:
        for relative, payload in anchor_preflight_capture["files"].items():
            atomic_bytes(
                _local_io_path(artifacts / "preflight" / relative), payload,
            )
    if deep_preflight_capture is not None:
        for relative, payload in deep_preflight_capture["files"].items():
            atomic_bytes(
                _local_io_path(artifacts / "preflight" / relative), payload,
            )
    warm_contract_destination = None
    warm_evidence_destinations = []
    if warm_contract_reference is not None:
        warm_contract_destination = (
            artifacts / "warm" / warm_contract_reference[0].name
        )
        warm_evidence_destinations = [
            artifacts / "warm" / relative
            for relative in warm_capture["evidence_relatives"]
        ]
    files = _deployment_files(local_bundle)
    for relative, payload in warm_capture["files"].items():
        deployed_relative = f"artifacts/warm/{relative}"
        expected = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
        if files.get(deployed_relative) != expected:
            raise RuntimeError("captured warm artifact deployment mismatch")
    deployment_model_sha = sha256(deployed_model_manifest_path)
    remote_bundle = f"{REMOTE_ROOT}/{bundle_id}"
    high_value = {
        relative: files[relative]["sha256"]
        for relative in (
            "artifacts/code/.source-revision",
            "artifacts/code/regression_260707/optimization/nsga2_contract.py",
            "artifacts/code/regression_260707/optimization/nsga2_problem.py",
            "artifacts/code/regression_260707/optimization/run_nsga2.py",
            "artifacts/code/regression_260707/model_targets.py",
            "artifacts/code/regression_260707/uncertainty_contract.py",
            "artifacts/code/regression_260707/training/predictor.py",
            "artifacts/code/module/input_parameter_260706.py",
            "artifacts/feedback.py",
            "artifacts/seed_runner.py",
            "artifacts/tier1_n1_6_anchor_island_contract.py",
            "artifacts/tier1_deep_crossover_contract.py",
            "artifacts/model/manifest.json",
            "artifacts/model/training_report.json",
            f"artifacts/warm/{warm_start.name}",
        )
    }
    if warm_contract_destination is not None:
        relative = warm_contract_destination.relative_to(local_bundle).as_posix()
        high_value[relative] = files[relative]["sha256"]
    for destination in warm_evidence_destinations:
        relative = destination.relative_to(local_bundle).as_posix()
        high_value[relative] = files[relative]["sha256"]
    anchor_preflight_contract_destination = None
    anchor_preflight_evidence = []
    if anchor_preflight_capture is not None:
        anchor_preflight_contract_destination = (
            "artifacts/preflight/" + anchor_preflight.name
        )
        anchor_preflight_evidence = sorted(
            "artifacts/preflight/" + relative
            for relative in anchor_preflight_capture["files"]
            if relative != anchor_preflight.name
        )
        for relative in (
            anchor_preflight_contract_destination,
            *anchor_preflight_evidence,
        ):
            high_value[relative] = files[relative]["sha256"]
    deep_preflight_contract_destination = None
    deep_preflight_evidence = []
    if deep_preflight_capture is not None:
        deep_preflight_contract_destination = (
            "artifacts/preflight/" + deep_crossover_preflight.name
        )
        deep_preflight_evidence = sorted(
            "artifacts/preflight/" + relative
            for relative in deep_preflight_capture["files"]
            if relative != deep_crossover_preflight.name
        )
        for relative in (
            deep_preflight_contract_destination,
            *deep_preflight_evidence,
        ):
            high_value[relative] = files[relative]["sha256"]
    manifest = {
        **stable_identity,
        "bundle_id": bundle_id,
        "cohort_id": cohort_id,
        "stable_identity_sha256": identity_sha,
        "created_at": now(),
        "hard_spec": hard_spec,
        "temperature_constraint_contract": temperature_contract,
        "temperature_constraint_contract_sha256": temperature_contract_sha,
        "search_profile": search_profile(),
        "optimizer_resonance_scale_Hz": OPTIMIZER_RESONANCE_SCALE_HZ,
        "optimizer_core_thermal_scale_C": (
            OPTIMIZER_CORE_THERMAL_SCALE_C
        ),
        "optimizer_Llt_scale_uH": OPTIMIZER_LLT_SCALE_UH,
        "optimizer_all_active_thermal_scale_C": (
            OPTIMIZER_ALL_THERMAL_SCALE_C
        ),
        "acquisition_ranking_contract": acquisition_ranking_contract,
        "acquisition_ranking_contract_sha256": (
            acquisition_ranking_contract_sha
        ),
        "deployment_model_manifest_sha256": deployment_model_sha,
        "base_bundle_ready_path": f"{base_remote}/READY.json",
        "base_bundle_manifest_path": f"{base_remote}/bundle_manifest.json",
        "base_generation_report_path": base_report_remote,
        "base_generation_path": base_generation_remote,
        "base_python_site": f"{base_remote}/artifacts/python-site",
        "code_root": "artifacts/code",
        "feedback_script": "artifacts/feedback.py",
        "seed_runner": "artifacts/seed_runner.py",
        "optimizer_allowance_contract_script": (
            "artifacts/tier1_n1_6_anchor_island_contract.py"
        ),
        "deep_crossover_contract_script": (
            "artifacts/tier1_deep_crossover_contract.py"
        ),
        "model_dir": "artifacts/model",
        "warm_start": f"artifacts/warm/{warm_start.name}",
        "warm_handoff_contract": (
            f"artifacts/warm/{warm_contract_destination.name}"
            if warm_contract_destination is not None else None
        ),
        "warm_handoff_evidence": [
            destination.relative_to(local_bundle).as_posix()
            for destination in warm_evidence_destinations
        ],
        "anchor_island_preflight": anchor_preflight_contract_destination,
        "anchor_island_preflight_sha256": (
            anchor_preflight_capture["contract_sha256"]
            if anchor_preflight_capture is not None else None
        ),
        "anchor_island_preflight_evidence": anchor_preflight_evidence,
        "deep_crossover_preflight": deep_preflight_contract_destination,
        "deep_crossover_preflight_sha256": (
            deep_preflight_capture["contract_sha256"]
            if deep_preflight_capture is not None else None
        ),
        "deep_crossover_preflight_evidence": deep_preflight_evidence,
        "files": files,
        "high_value_sha256": high_value,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    manifest_path = local_bundle / "bundle_manifest.json"
    atomic_json(manifest_path, manifest)
    plan = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "created_at": now(),
        "cohort_id": cohort_id,
        "bundle_id": bundle_id,
        "local_bundle": str(local_bundle),
        "remote_bundle": remote_bundle,
        "bundle_manifest": str(manifest_path),
        "bundle_manifest_sha256": sha256(manifest_path),
        "stable_identity_sha256": identity_sha,
        "source_model_manifest_sha256": source_manifest_sha,
        "deployment_model_manifest_sha256": deployment_model_sha,
        "constraint_version": constraint_version,
        "hard_spec": hard_spec,
        "hard_spec_sha256": hard_spec_sha,
        "attestation_schema_version": ATTESTATION_SCHEMA,
        "task_schema_version": TASK_SCHEMA,
        "controller_source_sha256": sha256(controller),
        "variant_entrypoint_sha256": sha256(entrypoint),
        "temperature_constraint_contract": temperature_contract,
        "temperature_constraint_contract_sha256": temperature_contract_sha,
        "search_profile": search_profile(),
        "optimizer_resonance_scale_Hz": OPTIMIZER_RESONANCE_SCALE_HZ,
        "optimizer_core_thermal_scale_C": (
            OPTIMIZER_CORE_THERMAL_SCALE_C
        ),
        "optimizer_Llt_scale_uH": OPTIMIZER_LLT_SCALE_UH,
        "optimizer_all_active_thermal_scale_C": (
            OPTIMIZER_ALL_THERMAL_SCALE_C
        ),
        "acquisition_ranking_contract": acquisition_ranking_contract,
        "acquisition_ranking_contract_sha256": (
            acquisition_ranking_contract_sha
        ),
        "nsga_code_revision": revision,
        "warm_start_sha256": warm_capture["warm_start_sha256"],
        "warm_handoff_contract_sha256": (
            warm_contract_reference[1] if warm_contract_reference else None
        ),
        "anchor_island_preflight_sha256": (
            anchor_preflight_capture["contract_sha256"]
            if anchor_preflight_capture is not None else None
        ),
        "deep_crossover_preflight_sha256": (
            deep_preflight_capture["contract_sha256"]
            if deep_preflight_capture is not None else None
        ),
        "base_python_site": manifest["base_python_site"],
        "file_count": len(files),
        "byte_count": sum(item["size"] for item in files.values()),
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    atomic_json(plan_path, plan)
    return plan


def _scheduler_imports(source: Path):
    source = source.resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from slurm_scheduler.config import load_accounts
    from slurm_scheduler.slurm import SSHSession
    return load_accounts, SSHSession


def _account(accounts: Path, source: Path, name: str):
    load_accounts, session = _scheduler_imports(source)
    records = {item.name: item for item in load_accounts(accounts)}
    if name not in records:
        raise RuntimeError(f"scheduler account not configured: {name}")
    return records[name], session


def stage(
    plan_path: Path, accounts: Path = DEFAULT_ACCOUNTS,
    scheduler_source: Path = DEFAULT_SCHEDULER_SOURCE,
    account_name: str = "harry261", apply: bool = False,
) -> dict:
    plan = read_json(plan_path.resolve(strict=True))
    local_bundle = Path(plan["local_bundle"]).resolve(strict=True)
    manifest = read_json(Path(plan["bundle_manifest"]))
    local_integrity_ok = _local_deployment_integrity(plan)
    bridge_evidence_ok = True
    anchor_preflight_ok = True
    deep_preflight_ok = True
    if SEARCH_VARIANT in {
        "fixed-n1-6-thermal-bridge-v1",
        "fixed-n1-6-orthogonal-bridge-v1",
        *(island.variant for island in ANCHOR_ISLANDS),
        *(island.variant for island in DEEP_CROSSOVER_ISLANDS),
        "fixed-n1-5-size-rx-core-v1",
    }:
        bridge_evidence_ok = _bridge_bundle_evidence_attested(
            local_bundle, plan, manifest,
        )
    if SEARCH_VARIANT in ANCHOR_ISLAND_BY_VARIANT:
        anchor_preflight_ok = _anchor_bundle_preflight_attested(
            local_bundle, plan, manifest,
        )
    if SEARCH_VARIANT in DEEP_CROSSOVER_BY_VARIANT:
        deep_preflight_ok = _deep_bundle_preflight_attested(
            local_bundle, plan, manifest,
        )
    if (
        not local_integrity_ok
        or plan.get("attestation_schema_version") != ATTESTATION_SCHEMA
        or manifest.get("attestation_schema_version") != ATTESTATION_SCHEMA
        or plan.get("task_schema_version") != TASK_SCHEMA
        or manifest.get("task_schema_version") != TASK_SCHEMA
        or manifest.get("hard_spec") != plan.get("hard_spec")
        or manifest.get("hard_spec_sha256") != plan.get("hard_spec_sha256")
        or plan.get("search_profile") != search_profile()
        or manifest.get("search_profile") != search_profile()
        or plan.get("optimizer_resonance_scale_Hz")
        != OPTIMIZER_RESONANCE_SCALE_HZ
        or manifest.get("optimizer_resonance_scale_Hz")
        != OPTIMIZER_RESONANCE_SCALE_HZ
        or plan.get("optimizer_core_thermal_scale_C")
        != OPTIMIZER_CORE_THERMAL_SCALE_C
        or manifest.get("optimizer_core_thermal_scale_C")
        != OPTIMIZER_CORE_THERMAL_SCALE_C
        or plan.get("optimizer_Llt_scale_uH")
        != OPTIMIZER_LLT_SCALE_UH
        or manifest.get("optimizer_Llt_scale_uH")
        != OPTIMIZER_LLT_SCALE_UH
        or plan.get("optimizer_all_active_thermal_scale_C")
        != OPTIMIZER_ALL_THERMAL_SCALE_C
        or manifest.get("optimizer_all_active_thermal_scale_C")
        != OPTIMIZER_ALL_THERMAL_SCALE_C
        or plan.get("acquisition_ranking_contract")
        != thermal_crossover_acquisition_contract(plan["hard_spec"])
        or manifest.get("acquisition_ranking_contract")
        != plan.get("acquisition_ranking_contract")
        or plan.get("acquisition_ranking_contract_sha256")
        != (
            (plan.get("acquisition_ranking_contract") or {}).get("sha256")
            if OPTIMIZER_CORE_THERMAL_SCALE_C is not None else None
        )
        or manifest.get("acquisition_ranking_contract_sha256")
        != plan.get("acquisition_ranking_contract_sha256")
        or (
            (
                OPTIMIZER_CORE_THERMAL_SCALE_C is not None
                or OPTIMIZER_ALL_THERMAL_SCALE_C is not None
            )
            and any(
                payload.get(field) is not False
                for payload in (plan, manifest)
                for field in (
                    "production_eligible", "fea_submission_approved",
                    "fea_submission_performed", "aedt_used",
                    "automatic_promotion_allowed",
                )
            )
        )
        or plan.get("warm_handoff_contract_sha256")
        != manifest.get("warm_handoff_contract_sha256")
        or not bridge_evidence_ok
        or not anchor_preflight_ok
        or not deep_preflight_ok
        or plan.get("anchor_island_preflight_sha256")
        != manifest.get("anchor_island_preflight_sha256")
        or (
            SEARCH_VARIANT in ANCHOR_ISLAND_BY_VARIANT
            and not plan.get("anchor_island_preflight_sha256")
        )
        or plan.get("deep_crossover_preflight_sha256")
        != manifest.get("deep_crossover_preflight_sha256")
        or (
            SEARCH_VARIANT in DEEP_CROSSOVER_BY_VARIANT
            and not plan.get("deep_crossover_preflight_sha256")
        )
        or (
            SEARCH_VARIANT in {
                "resonance-focus-250hz-v1",
                "thermal-crossover-core1c-v1",
                "thermal-crossover-balanced4c-v1",
                "fixed-n1-5-thermal-llt-v1",
                "fixed-n1-6-resonance-llt-v1",
                "fixed-n1-6-thermal-bridge-v1",
                "fixed-n1-6-orthogonal-bridge-v1",
                *(island.variant for island in ANCHOR_ISLANDS),
                *(island.variant for island in DEEP_CROSSOVER_ISLANDS),
                "fixed-n1-5-size-rx-core-v1",
            }
            and not plan.get("warm_handoff_contract_sha256")
        )
    ):
        raise RuntimeError("deployment plan/manifest v3 attestation mismatch")
    validate_temperature_constraint_contract(
        plan["hard_spec"], plan.get("temperature_constraint_contract"),
        plan.get("temperature_constraint_contract_sha256"),
    )
    if (
        manifest.get("temperature_constraint_contract")
        != plan["temperature_constraint_contract"]
        or manifest.get("temperature_constraint_contract_sha256")
        != plan["temperature_constraint_contract_sha256"]
    ):
        raise RuntimeError("deployment temperature contract mismatch")
    result = {
        "action": "stage", "apply": bool(apply),
        "cohort_id": plan["cohort_id"], "remote_bundle": plan["remote_bundle"],
        "file_count": plan["file_count"], "byte_count": plan["byte_count"],
    }
    if not apply:
        return result
    account, session_type = _account(accounts, scheduler_source, account_name)
    remote = plan["remote_bundle"].rstrip("/")
    incoming = f"{REMOTE_ROOT}/.incoming-{plan['bundle_id']}-{uuid.uuid4().hex[:10]}"
    with session_type(account, default_timeout=300) as session:
        ready_check = session.run(
            f"test -f '{remote}/READY.json' && cat '{remote}/READY.json'", timeout=60
        )
        if ready_check.exit_code == 0:
            ready = json.loads(ready_check.stdout)
            if ready.get("bundle_manifest_sha256") == plan["bundle_manifest_sha256"]:
                result["already_ready"] = True
                return result
            raise RuntimeError("remote bundle exists with another READY seal")
        absent = session.run(f"test ! -e '{remote}'", timeout=60)
        if absent.exit_code != 0:
            raise RuntimeError("remote bundle exists without a matching READY seal")
        created = session.run(
            f"mkdir -p '{incoming}/runs' && chmod 1777 '{incoming}/runs'", timeout=60
        )
        if created.exit_code != 0:
            raise RuntimeError(created.stderr or "cannot create incoming bundle")
        try:
            # One archive avoids hundreds of SSH/SFTP round trips.  The
            # immutable per-file manifest below still authenticates every
            # extracted byte before READY publication.
            archive = local_bundle.parent / "bundle-upload.tar"
            with tarfile.open(archive, "w") as stream:
                stream.add(
                    _local_io_path(local_bundle / "artifacts"),
                    arcname="artifacts",
                )
                stream.add(
                    local_bundle / "bundle_manifest.json",
                    arcname="bundle_manifest.json",
                )
            remote_archive = f"{incoming}/bundle-upload.tar"
            session.upload_file(str(archive), remote_archive + ".part")
            extracted = session.run(
                f"mv '{remote_archive}.part' '{remote_archive}' && "
                f"tar -xf '{remote_archive}' -C '{incoming}' && "
                f"rm -f '{remote_archive}'",
                timeout=300,
            )
            if extracted.exit_code != 0:
                raise RuntimeError(extracted.stderr or "cannot extract staged bundle")
            verify_script = """
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'bundle_manifest.json').read_text())
for relative,record in manifest['files'].items():
    path=root/relative
    if not path.is_file() or path.stat().st_size != int(record['size']):
        raise SystemExit('missing_or_size:'+relative)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=record['sha256']:raise SystemExit('sha:'+relative)
print('verified')
""".strip()
            command = (
                "python - " + repr(incoming) + " <<'PY'\n" + verify_script + "\nPY"
            )
            verified = session.run("bash -lc " + shlex.quote(command), timeout=600)
            if verified.exit_code != 0:
                raise RuntimeError(verified.stderr or verified.stdout)
            setup = (account.env_profiles or {}).get("pyaedt2026v1", "")
            preflight_script = """
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
code=root/'artifacts'/'code'
sys.path[:0]=[str(code),str(code/'regression_260707'),str(code/'regression_260707'/'training')]
from module.input_parameter_260706 import decode_unit_sample
from optimization.nsga2_problem import DEFAULT_SPEC,MFTProblem,SIDE_TEMPERATURE_TARGETS,T_TARGETS
from model_targets import SURROGATE_TEMPERATURE_TARGETS
from predictor import EnsemblePredictor
from uncertainty_contract import CONFORMAL_HALF_WIDTH_CONTRACT
if 'T_limit_C' in DEFAULT_SPEC:raise SystemExit('stale_default_temperature_limit')
if list(T_TARGETS)!=list(SURROGATE_TEMPERATURE_TARGETS):raise SystemExit('temperature_target_order')
print(json.dumps({'module_root':str(code),'decode':decode_unit_sample.__name__,'problem':MFTProblem.__name__,'predictor':EnsemblePredictor.__name__,'temperature_targets':list(T_TARGETS),'side_temperature_targets':list(SIDE_TEMPERATURE_TARGETS),'uncertainty_contract':CONFORMAL_HALF_WIDTH_CONTRACT,'explicit_temperature_required':True},sort_keys=True))
""".strip()
            preflight_command = "\n".join([
                "set -euo pipefail",
                setup,
                f"export PYTHONPATH={shlex.quote(manifest['base_python_site'])}",
                f"python - {shlex.quote(incoming)} <<'PY'",
                preflight_script,
                "PY",
            ])
            preflight = session.run(
                "bash -lc " + shlex.quote(preflight_command), timeout=300
            )
            if preflight.exit_code != 0:
                raise RuntimeError(
                    preflight.stderr or preflight.stdout or "remote import preflight failed"
                )
            ready = {
                "schema_version": DEPLOYMENT_SCHEMA,
                "bundle_id": plan["bundle_id"],
                "cohort_id": plan["cohort_id"],
                "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
                "attestation_schema_version": ATTESTATION_SCHEMA,
                "task_schema_version": TASK_SCHEMA,
                "hard_spec_sha256": plan["hard_spec_sha256"],
                "temperature_constraint_contract_sha256": plan[
                    "temperature_constraint_contract_sha256"
                ],
                "search_profile": search_profile(),
                "optimizer_resonance_scale_Hz": (
                    OPTIMIZER_RESONANCE_SCALE_HZ
                ),
                "optimizer_core_thermal_scale_C": (
                    OPTIMIZER_CORE_THERMAL_SCALE_C
                ),
                "optimizer_Llt_scale_uH": OPTIMIZER_LLT_SCALE_UH,
                "optimizer_all_active_thermal_scale_C": (
                    OPTIMIZER_ALL_THERMAL_SCALE_C
                ),
                "acquisition_ranking_contract_sha256": plan.get(
                    "acquisition_ranking_contract_sha256"
                ),
                "warm_handoff_contract_sha256": plan.get(
                    "warm_handoff_contract_sha256"
                ),
                "anchor_island_preflight_sha256": plan.get(
                    "anchor_island_preflight_sha256"
                ),
                "deep_crossover_preflight_sha256": plan.get(
                    "deep_crossover_preflight_sha256"
                ),
                "verified": True,
                "remote_import_preflight": json.loads(
                    preflight.stdout.strip().splitlines()[-1]
                ),
                "verified_at": now(),
                "verified_by_account": account_name,
                "production_eligible": False,
                "fea_submission_approved": False,
                "fea_submission_performed": False,
                "aedt_used": False,
                "automatic_promotion_allowed": False,
            }
            session.write_text_file(
                f"{incoming}/READY.json",
                json.dumps(ready, indent=2, sort_keys=True) + "\n",
            )
            published = session.run(
                f"chmod -R a-w '{incoming}/artifacts' && "
                f"chmod -R a+rX '{incoming}/artifacts' && "
                f"chmod 1777 '{incoming}/runs' && mv '{incoming}' '{remote}'",
                timeout=300,
            )
            if published.exit_code != 0:
                raise RuntimeError(published.stderr or "cannot publish bundle")
        except Exception:
            result["incomplete_remote"] = incoming
            raise
    result["ready"] = True
    result["ready_at"] = now()
    return result


def api_json(
    url: str, method: str = "GET", payload: dict | None = None, timeout: int = 30,
):
    data = canonical_bytes(payload) if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def api_bytes(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def publish_pointer(plan_path: Path, runtime: Path = RUNTIME) -> dict:
    plan = read_json(plan_path.resolve(strict=True))
    if (
        (
            SEARCH_VARIANT in ANCHOR_ISLAND_BY_VARIANT
            and (
                not _local_deployment_integrity(plan)
                or not plan.get("anchor_island_preflight_sha256")
            )
        )
        or (
            SEARCH_VARIANT in DEEP_CROSSOVER_BY_VARIANT
            and (
                not _local_deployment_integrity(plan)
                or not plan.get("deep_crossover_preflight_sha256")
            )
        )
        or plan.get("attestation_schema_version") != ATTESTATION_SCHEMA
        or plan.get("task_schema_version") != TASK_SCHEMA
        or plan.get("search_profile") != search_profile()
        or plan.get("optimizer_resonance_scale_Hz")
        != OPTIMIZER_RESONANCE_SCALE_HZ
        or plan.get("optimizer_core_thermal_scale_C")
        != OPTIMIZER_CORE_THERMAL_SCALE_C
        or plan.get("optimizer_Llt_scale_uH")
        != OPTIMIZER_LLT_SCALE_UH
        or plan.get("optimizer_all_active_thermal_scale_C")
        != OPTIMIZER_ALL_THERMAL_SCALE_C
        or plan.get("acquisition_ranking_contract")
        != thermal_crossover_acquisition_contract(plan["hard_spec"])
        or plan.get("acquisition_ranking_contract_sha256")
        != (
            (plan.get("acquisition_ranking_contract") or {}).get("sha256")
            if OPTIMIZER_CORE_THERMAL_SCALE_C is not None else None
        )
        or (
            (
                OPTIMIZER_CORE_THERMAL_SCALE_C is not None
                or OPTIMIZER_ALL_THERMAL_SCALE_C is not None
            )
            and any(
                plan.get(field) is not False
                for field in (
                    "production_eligible", "fea_submission_approved",
                    "fea_submission_performed", "aedt_used",
                    "automatic_promotion_allowed",
                )
            )
        )
    ):
        raise RuntimeError("refusing to publish a non-v3 cohort pointer")
    validate_temperature_constraint_contract(
        plan["hard_spec"], plan.get("temperature_constraint_contract"),
        plan.get("temperature_constraint_contract_sha256"),
    )
    pointer_path = runtime.resolve() / "canonical" / "model_pointer.json"
    previous = read_json(pointer_path) if pointer_path.is_file() else None
    legacy = list((previous or {}).get("legacy_cohorts") or [])
    if previous and previous.get("current", {}).get("cohort_id") != plan["cohort_id"]:
        old = dict(previous["current"])
        old.update({
            "legacy_at": now(),
            "refill_allowed": False,
            "legacy_reason": "atomic_constraint_or_model_pointer_switch",
        })
        legacy.append(old)
    current = {
        "cohort_id": plan["cohort_id"],
        "constraint_version": plan["constraint_version"],
        "hard_spec": plan["hard_spec"],
        "hard_spec_sha256": plan["hard_spec_sha256"],
        "attestation_schema_version": plan["attestation_schema_version"],
        "task_schema_version": plan["task_schema_version"],
        "controller_source_sha256": plan["controller_source_sha256"],
        "temperature_constraint_contract": plan[
            "temperature_constraint_contract"
        ],
        "temperature_constraint_contract_sha256": plan[
            "temperature_constraint_contract_sha256"
        ],
        "source_model_manifest_sha256": plan["source_model_manifest_sha256"],
        "deployment_model_manifest_sha256": plan[
            "deployment_model_manifest_sha256"
        ],
        "bundle_id": plan["bundle_id"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "remote_bundle": plan["remote_bundle"],
        "base_python_site": plan["base_python_site"],
        "nsga_code_revision": plan["nsga_code_revision"],
        "warm_start_sha256": plan["warm_start_sha256"],
        "search_profile": plan["search_profile"],
        "optimizer_resonance_scale_Hz": plan[
            "optimizer_resonance_scale_Hz"
        ],
        "optimizer_core_thermal_scale_C": plan.get(
            "optimizer_core_thermal_scale_C"
        ),
        "optimizer_Llt_scale_uH": plan.get("optimizer_Llt_scale_uH"),
        "optimizer_all_active_thermal_scale_C": plan.get(
            "optimizer_all_active_thermal_scale_C"
        ),
        "acquisition_ranking_contract": plan.get(
            "acquisition_ranking_contract"
        ),
        "acquisition_ranking_contract_sha256": plan.get(
            "acquisition_ranking_contract_sha256"
        ),
        "refill_allowed": True,
        "rolling_target": ROLLING_TARGET,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    payload = {
        "schema_version": POINTER_SCHEMA,
        "updated_at": now(),
        "current": current,
        "legacy_cohorts": legacy,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    atomic_json(pointer_path, payload)
    return payload


def _task_prefix(cohort_id: str) -> str:
    return f"{TASK_NAME_STEM}-{cohort_id[-10:]}-s"


def list_cohort_tasks(scheduler_url: str, cohort_id: str) -> list[dict]:
    prefix = _task_prefix(cohort_id)
    all_tasks = []
    page = 1
    while True:
        query = urllib.parse.urlencode({
            "paged": "true", "page": page, "page_size": 10000,
            "name_prefix": prefix, "sort_by": "id", "sort_order": "desc",
        })
        response = api_json(f"{scheduler_url.rstrip('/')}/api/tasks?{query}")
        if isinstance(response, list):
            return response
        items = response.get("items") or response.get("tasks") or []
        all_tasks.extend(items)
        pagination = response.get("pagination") or response.get("page") or {}
        if not pagination.get("has_next"):
            break
        page += 1
    return all_tasks


def _seed_from_name(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    token = name[len(prefix):].split("-", 1)[0]
    return int(token) if token.isdigit() else None


def build_task_contract(pointer: dict, seed: int) -> dict:
    """Single builder used by both an empty-cohort fill and every refill."""

    current = pointer["current"]
    _validate_current_attestation(current)
    if DEEP_CROSSOVER_ISLAND is not None and not (
        DEEP_CROSSOVER_ISLAND.seed_start
        <= int(seed)
        < DEEP_CROSSOVER_ISLAND.seed_window_end_exclusive
    ):
        raise RuntimeError("task seed escaped deep crossover seed window")
    task_contract = {
        "schema_version": (
            TASK_SCHEMA if _is_v3_current(current) else LEGACY_TASK_SCHEMA
        ),
        "cohort_id": current["cohort_id"],
        "bundle_id": current["bundle_id"],
        "bundle_manifest_sha256": current["bundle_manifest_sha256"],
        "constraint_version": current["constraint_version"],
        "hard_spec_sha256": current["hard_spec_sha256"],
        "temperature_constraint_contract": current[
            "temperature_constraint_contract"
        ],
        "source_model_manifest_sha256": current[
            "source_model_manifest_sha256"
        ],
        "deployment_model_manifest_sha256": current[
            "deployment_model_manifest_sha256"
        ],
        "nsga_code_revision": current["nsga_code_revision"],
        "warm_start_sha256": current["warm_start_sha256"],
        "seed": int(seed),
        "population": POPULATION,
        "max_generations": MAX_GENERATIONS,
        "inference_threads": INFERENCE_THREADS,
        "search_profile": current["search_profile"],
        "optimizer_resonance_scale_Hz": current[
            "optimizer_resonance_scale_Hz"
        ],
        "optimizer_core_thermal_scale_C": current.get(
            "optimizer_core_thermal_scale_C"
        ),
        "optimizer_Llt_scale_uH": current.get("optimizer_Llt_scale_uH"),
        "optimizer_all_active_thermal_scale_C": current.get(
            "optimizer_all_active_thermal_scale_C"
        ),
        "acquisition_ranking_contract": current.get(
            "acquisition_ranking_contract"
        ),
        "acquisition_ranking_contract_sha256": current.get(
            "acquisition_ranking_contract_sha256"
        ),
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    if _is_v3_current(current):
        task_contract.update({
            "attestation_schema_version": current[
                "attestation_schema_version"
            ],
            "task_schema_version": current["task_schema_version"],
            "controller_source_sha256": current[
                "controller_source_sha256"
            ],
            "hard_spec": current["hard_spec"],
            "temperature_constraint_contract_sha256": current[
                "temperature_constraint_contract_sha256"
            ],
        })
    return task_contract


def _is_priority_migration_seed(seed: int) -> bool:
    return SEED_START <= int(seed) < PRIORITY_MIGRATION_SEED_CUTOFF


def task_payload(
    pointer: dict, seed: int, *, priority: int | None = None,
) -> dict:
    current = pointer["current"]
    task_contract = build_task_contract(pointer, seed)
    if priority is None:
        priority = (
            LEGACY_TASK_PRIORITY
            if _is_priority_migration_seed(seed)
            else TASK_PRIORITY
        )
    payload_sha = canonical_sha(task_contract)
    command = "\n".join([
        "set -euo pipefail",
        f"export PYTHONPATH='{current['base_python_site']}'${{PYTHONPATH:+:$PYTHONPATH}}",
        'payload_path="${SLURM_SCHEDULER_PAYLOAD_PATH:?missing scheduler payload}"',
        'case "$payload_path" in /*) ;; *) payload_path="$HOME/$payload_path" ;; esac',
        'payload_path=$(realpath -e -- "$payload_path")',
        'payload_root=$(realpath -e -- "$HOME/slurm_scheduler/runs")',
        'case "$payload_path" in "$payload_root"/*/payload.json) ;; '
        '*) echo "unsafe scheduler payload" >&2; exit 66 ;; esac',
        "exec python artifacts/seed_runner.py "
        '--bundle "$PWD" --payload "$payload_path" --payload-root "$payload_root" '
        f"--payload-sha256 {payload_sha}",
    ])
    resource_contract = {
        "task": task_contract,
        "cpus": 8, "memory_mb": 32768, "priority": int(priority),
        "scheduling_profile": "standard", "aedt_backend": "standalone",
    }
    prefix = _task_prefix(current["cohort_id"])
    return {
        "name": f"{prefix}{seed}",
        "remote_cwd": current["remote_bundle"],
        "command": command,
        "payload_json": task_contract,
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "cpus": 8,
        "memory_mb": 32768,
        "scheduling_profile": "standard",
        "aedt_backend": "standalone",
        "gpus": 0,
        "priority": int(priority),
        "timeout_seconds": 86400,
        "dedupe_key": (
            DEDUPE_NAMESPACE + ":" + canonical_sha(resource_contract)
        ),
        "max_workers_per_node": 4,
    }


def assert_existing_task_attestation(
    pointer: dict, tasks: list[dict],
) -> None:
    """Halt unless each task has the current or sealed migration identity."""

    current = pointer["current"]
    if not _is_v3_current(current):
        return
    prefix = _task_prefix(current["cohort_id"])
    for task in tasks:
        name = str(task.get("name") or "")
        seed = _seed_from_name(name, prefix)
        if seed is None:
            raise RuntimeError(f"v3 cohort task has no canonical seed: {name}")
        expected = task_payload(pointer, seed)
        identity_matches = not (
            task.get("dedupe_key") != expected["dedupe_key"]
            or task.get("remote_cwd") != expected["remote_cwd"]
            or name != expected["name"]
        )
        state = _task_state(task)
        priority = int(task.get("priority") or 0)
        if identity_matches and _is_priority_migration_seed(seed):
            migrated_in_place = priority == TASK_PRIORITY
            grandfathered_after_claim = (
                priority == LEGACY_TASK_PRIORITY and state != "queued"
            )
            if migrated_in_place or grandfathered_after_claim:
                continue
        elif identity_matches and (
            "priority" not in task or priority == TASK_PRIORITY
        ):
            continue
        task_id = task.get("id") or task.get("task_id")
        raise RuntimeError(
            f"v3 task {task_id} is missing canonical payload attestation"
        )


def migrate_priority(
    scheduler_url: str = DEFAULT_SCHEDULER_URL, runtime: Path = RUNTIME,
    apply: bool = False,
) -> dict:
    """Reprioritize only queued legacy-deep tasks and seal the transition."""

    pointer = read_json(runtime.resolve() / "canonical" / "model_pointer.json")
    current = pointer["current"]
    tasks = list_cohort_tasks(scheduler_url, current["cohort_id"])
    prefix = _task_prefix(current["cohort_id"])
    before = []
    update_ids = []
    already_migrated_ids = []
    for task in tasks:
        name = str(task.get("name") or "")
        seed = _seed_from_name(name, prefix)
        if seed is None:
            raise RuntimeError(f"priority migration saw foreign task: {name}")
        legacy_payload = task_payload(
            pointer, seed, priority=LEGACY_TASK_PRIORITY
        )
        successor_payload = task_payload(pointer, seed, priority=TASK_PRIORITY)
        legacy_match = task.get("dedupe_key") == legacy_payload["dedupe_key"]
        successor_match = (
            task.get("dedupe_key") == successor_payload["dedupe_key"]
        )
        expected_match = (
            legacy_match
            if _is_priority_migration_seed(seed)
            else successor_match
        )
        if not expected_match:
            raise RuntimeError(
                f"priority migration task {task.get('id')} has unknown dedupe"
            )
        state = _task_state(task)
        priority = int(task.get("priority") or 0)
        before.append({
            "task_id": int(task.get("id") or task.get("task_id")),
            "seed": seed,
            "state": state,
            "priority": priority,
            "dedupe_contract": (
                "legacy-cutoff" if _is_priority_migration_seed(seed)
                else "successor"
            ),
        })
        if state == "queued" and priority == LEGACY_TASK_PRIORITY:
            if not legacy_match:
                raise RuntimeError("queued legacy priority has current dedupe")
            update_ids.append(int(task.get("id") or task.get("task_id")))
        elif (
            _is_priority_migration_seed(seed)
            and legacy_match
            and priority == TASK_PRIORITY
        ):
            already_migrated_ids.append(
                int(task.get("id") or task.get("task_id"))
            )
        elif priority not in {TASK_PRIORITY, LEGACY_TASK_PRIORITY}:
            raise RuntimeError(
                f"priority migration task {task.get('id')} has priority {priority}"
            )
    result = {
        "schema_version": "mft-tier1-deep-priority-migration-v1",
        "created_at": now(),
        "cohort_id": current["cohort_id"],
        "task_prefix": prefix,
        "from_priority": LEGACY_TASK_PRIORITY,
        "to_priority": TASK_PRIORITY,
        "seed_cutoff_exclusive": PRIORITY_MIGRATION_SEED_CUTOFF,
        "queued_update_ids": update_ids,
        "already_migrated_ids": already_migrated_ids,
        "before": before,
        "applied": False,
        "production_eligible": False,
        "fea_submission_approved": False,
    }
    if not apply:
        return result
    responses = []
    for task_id in update_ids:
        try:
            response = api_json(
                f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}/priority",
                method="POST", payload={"priority": TASK_PRIORITY}, timeout=30,
            )
            responses.append(response)
        except Exception:
            task = api_json(
                f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}", timeout=30
            )
            if _task_state(task) == "queued":
                raise
            responses.append({
                "id": task_id,
                "priority": int(task.get("priority") or 0),
                "transitioned_before_update": True,
            })
    deadline = time.monotonic() + 10.0
    while True:
        after_tasks = list_cohort_tasks(scheduler_url, current["cohort_id"])
        try:
            assert_existing_task_attestation(pointer, after_tasks)
            stranded = [
                int(task.get("id") or task.get("task_id"))
                for task in after_tasks
                if _task_state(task) == "queued"
                and int(task.get("priority") or 0) != TASK_PRIORITY
            ]
            if not stranded:
                break
        except RuntimeError:
            stranded = update_ids
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"queued tasks missed priority migration: {stranded}"
            )
        time.sleep(0.25)
    result.update({
        "applied": True,
        "applied_at": now(),
        "responses": responses,
        "after": [{
            "task_id": int(task.get("id") or task.get("task_id")),
            "state": _task_state(task),
            "priority": int(task.get("priority") or 0),
        } for task in after_tasks],
    })
    result["receipt_sha256"] = canonical_sha(result)
    atomic_json(
        runtime.resolve() / "controller" / "priority_migration.json", result
    )
    return result


def _remote_task_file_url(
    scheduler_url: str, task_id: int, relative: str, max_bytes: int,
) -> str:
    query = urllib.parse.urlencode({
        "base": "remote_cwd", "path": relative, "max_bytes": max_bytes,
    })
    return f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}/remote-file?{query}"


def _terminal_record_dir(runtime: Path, cohort_id: str, task_id: int) -> Path:
    return runtime.resolve() / "cohorts" / cohort_id / "results" / f"task-{task_id}"


def harvest_terminal_results(
    scheduler_url: str, pointer: dict, tasks: list[dict], runtime: Path,
    max_new: int = 4,
) -> list[dict]:
    """Harvest a bounded number of terminal JSON results with exact SHA checks."""
    current = pointer["current"]
    prefix = _task_prefix(current["cohort_id"])
    terminal_tasks = [
        task for task in tasks if _task_state(task) in TERMINAL_STATES
    ]
    # Never let a repeatedly failing remote read at a low task ID block every
    # newer terminal forever.  First-attempt terminals are harvested before
    # retry records; failed reads remain retryable after the fresh backlog.
    candidates = sorted(
        terminal_tasks,
        key=lambda item: (
            (
                _terminal_record_dir(
                    runtime,
                    current["cohort_id"],
                    int(item.get("id") or item.get("task_id") or 0),
                ) / "record.json"
            ).is_file(),
            int(item.get("id") or item.get("task_id") or 0),
        ),
    )
    harvested = 0
    for task in candidates:
        task_id = int(task.get("id") or task.get("task_id"))
        record_dir = _terminal_record_dir(
            runtime, current["cohort_id"], task_id
        )
        record_path = record_dir / "record.json"
        if record_path.is_file():
            prior = read_json(record_path)
            if prior.get("authenticated") is True:
                continue
        if harvested >= max_new:
            break
        harvested += 1
        seed = _seed_from_name(str(task.get("name") or ""), prefix)
        record = {
            "schema_version": "mft-tier1-slurm-terminal-result-v1",
            "harvested_at": now(),
            "task_id": task_id,
            "seed": seed,
            "scheduler_state": _task_state(task),
            "scheduler_exit_code": task.get("exit_code"),
            "cohort_id": current["cohort_id"],
            "constraint_version": current["constraint_version"],
            "hard_spec_sha256": current["hard_spec_sha256"],
            "temperature_constraint_contract_sha256": current.get(
                "temperature_constraint_contract_sha256"
            ),
            "optimizer_resonance_scale_Hz": current.get(
                "optimizer_resonance_scale_Hz"
            ),
            "optimizer_core_thermal_scale_C": current.get(
                "optimizer_core_thermal_scale_C"
            ),
            "optimizer_Llt_scale_uH": current.get(
                "optimizer_Llt_scale_uH"
            ),
            "optimizer_all_active_thermal_scale_C": current.get(
                "optimizer_all_active_thermal_scale_C"
            ),
            "acquisition_ranking_contract_sha256": current.get(
                "acquisition_ranking_contract_sha256"
            ),
            "source_model_manifest_sha256": current[
                "source_model_manifest_sha256"
            ],
            "production_eligible": False,
            "fea_submission_approved": False,
            "fea_submission_performed": False,
            "aedt_used": False,
            "automatic_promotion_allowed": False,
            "authenticated": False,
        }
        status_relative = f"runs/task-{task_id}/seed_status.json"
        try:
            status_bytes = api_bytes(
                _remote_task_file_url(
                    scheduler_url, task_id, status_relative, 1024 * 1024
                )
            )
            status = json.loads(status_bytes.decode("utf-8"))
            if (
                status.get("schema_version")
                != "mft-tier1-slurm-seed-status-v1"
                or str(status.get("task_id")) != str(task_id)
                or int(status.get("seed", -1)) != int(seed)
                or status.get("cohort_id") != current["cohort_id"]
                or status.get("production_eligible") is not False
                or status.get("fea_submission_approved") is not False
                or status.get("fea_submission_performed") is not False
                or (
                    (
                        OPTIMIZER_CORE_THERMAL_SCALE_C is not None
                        or OPTIMIZER_ALL_THERMAL_SCALE_C is not None
                    )
                    and (
                        status.get("aedt_used") is not False
                        or status.get("automatic_promotion_allowed") is not False
                    )
                )
            ):
                raise RuntimeError("remote seed status identity mismatch")
            if _is_v3_current(current):
                expected_payload_sha = canonical_sha(
                    build_task_contract(pointer, int(seed))
                )
                if (
                    status.get("attestation_schema_version")
                    != ATTESTATION_SCHEMA
                    or status.get("task_schema_version") != TASK_SCHEMA
                    or status.get("constraint_version")
                    != current["constraint_version"]
                    or status.get("hard_spec_sha256")
                    != current["hard_spec_sha256"]
                    or status.get("temperature_constraint_contract_sha256")
                    != current["temperature_constraint_contract_sha256"]
                    or status.get("payload_sha256") != expected_payload_sha
                    or status.get("optimizer_resonance_scale_Hz")
                    != current["optimizer_resonance_scale_Hz"]
                    or status.get("optimizer_core_thermal_scale_C")
                    != current.get("optimizer_core_thermal_scale_C")
                    or status.get("optimizer_Llt_scale_uH")
                    != current.get("optimizer_Llt_scale_uH")
                    or status.get("optimizer_all_active_thermal_scale_C")
                    != current.get(
                        "optimizer_all_active_thermal_scale_C"
                    )
                    or status.get("fixed_primary_turns")
                    != (current.get("search_profile") or {}).get(
                        "fixed_primary_turns"
                    )
                    or (
                        DEEP_CROSSOVER_ISLAND is not None
                        and status.get("optimizer_termination_strategy")
                        != (current.get("search_profile") or {}).get(
                            "optimizer_termination_strategy"
                        )
                    )
                    or status.get("acquisition_ranking_contract_sha256")
                    != current.get("acquisition_ranking_contract_sha256")
                ):
                    raise RuntimeError("remote seed status v3 seal mismatch")
            atomic_bytes(record_dir / "seed_status.json", status_bytes)
            record["remote_status"] = {
                "remote_path": status_relative,
                "local_path": str(record_dir / "seed_status.json"),
                "sha256": hashlib.sha256(status_bytes).hexdigest(),
                "state": status.get("state"),
                "failure": status.get("failure"),
            }
            if status.get("state") == "completed":
                result_relative = f"runs/task-{task_id}/seed-{seed}/result.json"
                result_bytes = api_bytes(
                    _remote_task_file_url(
                        scheduler_url, task_id, result_relative, 16 * 1024 * 1024
                    )
                )
                result_sha = hashlib.sha256(result_bytes).hexdigest()
                if result_sha != status.get("result_sha256"):
                    raise RuntimeError("remote result SHA does not match seed status")
                result = json.loads(result_bytes.decode("utf-8"))
                if (
                    int(result.get("seed", -1)) != int(seed)
                    or result.get("constraint_version")
                    != current["constraint_version"]
                    or result.get("hard_spec_sha256")
                    != current["hard_spec_sha256"]
                    or result.get("model_manifest_sha256")
                    != current["deployment_model_manifest_sha256"]
                    or result.get("production_eligible") is not False
                    or result.get("fea_submission_approved") is not False
                    or (
                        (
                            OPTIMIZER_CORE_THERMAL_SCALE_C is not None
                            or OPTIMIZER_ALL_THERMAL_SCALE_C is not None
                        )
                        and (
                            result.get("fea_submission_performed") is not False
                            or result.get("aedt_used") is not False
                            or result.get("automatic_promotion_allowed") is not False
                        )
                    )
                ):
                    raise RuntimeError("remote result contract mismatch")
                if _is_v3_current(current):
                    expected_thermal_names = [
                        f"temperature_robust_limit:{target}"
                        for target in TEMPERATURE_TARGETS
                    ]
                    result_names = result.get("constraint_names") or []
                    actual_thermal_names = [
                        name for name in result_names
                        if isinstance(name, str)
                        and name.startswith("temperature_robust_limit:")
                    ]
                    allowance_profile = current.get("search_profile") or {}
                    expected_resonance_allowance = allowance_profile.get(
                        "optimizer_resonance_allowance_Hz"
                    )
                    expected_llt_allowance = allowance_profile.get(
                        "optimizer_Llt_allowance_uH"
                    )
                    if ((expected_resonance_allowance is None)
                            != (expected_llt_allowance is None)):
                        raise RuntimeError(
                            "current pointer has incomplete optimizer allowance"
                        )
                    expected_allowance_contract = None
                    if expected_resonance_allowance is not None:
                        expected_allowance_contract = (
                            optimizer_allowance_contract(
                                result_names,
                                resonance_allowance_hz=(
                                    expected_resonance_allowance
                                ),
                                llt_allowance_uh=expected_llt_allowance,
                            )
                        )
                    if (
                        result.get("hard_spec") != current["hard_spec"]
                        or canonical_sha(result.get("hard_spec"))
                        != current["hard_spec_sha256"]
                        or result.get("temperature_constraint_contract")
                        != current["temperature_constraint_contract"]
                        or result.get(
                            "temperature_constraint_contract_sha256"
                        ) != current["temperature_constraint_contract_sha256"]
                        or actual_thermal_names != expected_thermal_names
                        or result.get(
                            "optimizer_resonance_allowance_Hz"
                        ) != expected_resonance_allowance
                        or result.get("optimizer_Llt_allowance_uH")
                        != expected_llt_allowance
                        or result.get(
                            "optimizer_constraint_allowance_contract"
                        ) != expected_allowance_contract
                        or (
                            result.get("optimizer_constraint_normalization")
                            or {}
                        ).get("scales", {}).get(
                            "half_magnetizing_resonance_minimum"
                        ) != current["optimizer_resonance_scale_Hz"]
                        or result.get("optimizer_core_thermal_scale_C")
                        != current.get("optimizer_core_thermal_scale_C")
                        or result.get("optimizer_Llt_scale_uH")
                        != current.get("optimizer_Llt_scale_uH")
                        or result.get(
                            "optimizer_all_active_thermal_scale_C"
                        ) != current.get(
                            "optimizer_all_active_thermal_scale_C"
                        )
                        or result.get("fixed_primary_turns")
                        != (current.get("search_profile") or {}).get(
                            "fixed_primary_turns"
                        )
                        or (
                            DEEP_CROSSOVER_ISLAND is not None
                            and (
                                result.get("optimizer_termination_strategy")
                                != (current.get("search_profile") or {}).get(
                                    "optimizer_termination_strategy"
                                )
                                or result.get(
                                    "optimizer_termination_contract"
                                ) != optimizer_termination_contract()
                                or result.get(
                                    "fixed_generation_gate_satisfied"
                                ) is not True
                                or int(result.get(
                                    "completed_generations", -1
                                )) != MAX_GENERATIONS + 1
                                or result.get("evaluated_generations")
                                != MAX_GENERATIONS
                            )
                        )
                        or (
                            FIXED_PRIMARY_TURNS is not None
                            and result.get(
                                "terminal_population_fixed_primary_turns_verified"
                            ) is not True
                        )
                        or (
                            result.get("optimizer_constraint_normalization")
                            or {}
                        ).get("explicit_optimizer_only_scale_overrides")
                        != optimizer_explicit_overrides()
                        or (
                            OPTIMIZER_LLT_SCALE_UH is not None
                            and (
                                (result.get(
                                    "optimizer_constraint_normalization"
                                ) or {}).get("scales", {}).get(
                                    "Llt_robust_band"
                                ) != OPTIMIZER_LLT_SCALE_UH
                                or (result.get(
                                    "optimizer_constraint_normalization"
                                ) or {}).get("scales", {}).get(
                                    "Llt_ensemble_disagreement"
                                ) != 2.0 * OPTIMIZER_LLT_SCALE_UH
                            )
                        )
                        or (
                            OPTIMIZER_ALL_THERMAL_SCALE_C is not None
                            and any(
                                (result.get(
                                    "optimizer_constraint_normalization"
                                ) or {}).get("scales", {}).get(name)
                                != OPTIMIZER_ALL_THERMAL_SCALE_C
                                for name in expected_thermal_names
                            )
                        )
                        or (
                            OPTIMIZER_CORE_THERMAL_SCALE_C is not None
                            and result.get("acquisition_ranking_contract")
                            != thermal_crossover_acquisition_contract(
                                current["hard_spec"]
                            )
                        )
                        or any(
                            name not in (result.get(field) or {})
                            for field in (
                                "constraint_minimum_G",
                                "terminal_population_best_constraint_G",
                            )
                            for name in expected_thermal_names
                        )
                    ):
                        raise RuntimeError("remote result v3 thermal seal mismatch")
                atomic_bytes(record_dir / "result.json", result_bytes)
                record.update({
                    "terminal_state": "completed",
                    "authenticated": True,
                    "result": {
                        "remote_path": result_relative,
                        "local_path": str(record_dir / "result.json"),
                        "sha256": result_sha,
                        "bytes": len(result_bytes),
                    },
                    "completed_generations": int(result["completed_generations"]),
                    "evaluated_generations": result.get(
                        "evaluated_generations"
                    ),
                    "feasible_pareto_count": int(result["feasible_pareto_count"]),
                    "candidate_count": len(result.get("candidates") or []),
                    "constraint_minimum_G": result.get("constraint_minimum_G"),
                    "terminal_best_constraint_G": result.get(
                        "terminal_population_best_constraint_G"
                    ),
                    "terminal_minimum_total_positive_violation": result.get(
                        "terminal_population_minimum_total_positive_violation"
                    ),
                })
            else:
                record.update({
                    "terminal_state": status.get("state") or _task_state(task),
                    "authenticated": True,
                    "failure": status.get("failure") or task.get("failure_message"),
                })
        except Exception as exc:
            record.update({
                "terminal_state": _task_state(task),
                "harvest_error": f"{type(exc).__name__}:{exc}",
            })
        atomic_json(record_path, record)
    return _load_terminal_records(runtime, current["cohort_id"])


def _load_terminal_records(runtime: Path, cohort_id: str) -> list[dict]:
    root = runtime.resolve() / "cohorts" / cohort_id / "results"
    if not root.is_dir():
        return []
    records = []
    for path in sorted(root.glob("task-*/record.json")):
        try:
            records.append(read_json(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda item: int(item["task_id"]), reverse=True)


def _bounded_terminal_snapshot(
    runtime: Path, records: list[dict]
) -> tuple[list[dict], dict]:
    """Return a bounded, self-authenticating terminal/Pareto snapshot.

    The aggregate preview is computed over the full history first.  When the
    history exceeds the status-document limit, every terminal record backing a
    globally selected Pareto/near-feasible row (and the simultaneous best) is
    pinned, and the remaining slots are filled with the newest records.  The
    published aggregate is then recomputed over exactly that bounded record
    set.  This keeps the strict reader's source-count and identity checks
    coherent without allowing the status document to grow without bound.
    """
    full_aggregate = _aggregate_pareto(runtime, records)
    if len(records) <= TERMINAL_RESULT_WINDOW_LIMIT:
        return records, full_aggregate

    pinned_task_ids: set[int] = set()
    for key in ("candidates", "near_feasible"):
        for candidate in full_aggregate.get(key) or []:
            try:
                pinned_task_ids.add(int(candidate["source_task_id"]))
            except (KeyError, TypeError, ValueError):
                continue
    simultaneous_best = full_aggregate.get("simultaneous_best") or {}
    try:
        pinned_task_ids.add(int(simultaneous_best["task_id"]))
    except (KeyError, TypeError, ValueError):
        pass

    keep_task_ids = set(pinned_task_ids)
    for record in records:
        if len(keep_task_ids) >= TERMINAL_RESULT_WINDOW_LIMIT:
            break
        try:
            keep_task_ids.add(int(record["task_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    bounded_records = [
        record for record in records
        if int(record.get("task_id", -1)) in keep_task_ids
    ][:TERMINAL_RESULT_WINDOW_LIMIT]
    return bounded_records, _aggregate_pareto(runtime, bounded_records)


def _two_objective_nondominated(values: list[dict]) -> list[dict]:
    """Return the exact volume/loss Pareto front in O(n log n).

    Equal objective pairs are all retained because neither strictly dominates
    the other.  Non-finite legacy rows use the previous comparison semantics as
    a conservative fallback; authenticated current rows are finite.
    """

    objective_rows = [
        (float(item["volume_L"]), float(item["total_loss_W"]), index, item)
        for index, item in enumerate(values)
    ]
    if any(
        not math.isfinite(volume) or not math.isfinite(loss)
        for volume, loss, _index, _item in objective_rows
    ):
        return [
            item
            for index, item in enumerate(values)
            if not any(
                other_index != index
                and float(other["volume_L"]) <= float(item["volume_L"])
                and float(other["total_loss_W"]) <= float(item["total_loss_W"])
                and (
                    float(other["volume_L"]) < float(item["volume_L"])
                    or float(other["total_loss_W"])
                    < float(item["total_loss_W"])
                )
                for other_index, other in enumerate(values)
            )
        ]

    objective_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    nondominated = []
    best_loss_at_smaller_volume = math.inf
    start = 0
    while start < len(objective_rows):
        volume = objective_rows[start][0]
        end = start + 1
        while end < len(objective_rows) and objective_rows[end][0] == volume:
            end += 1
        group = objective_rows[start:end]
        minimum_group_loss = group[0][1]
        if minimum_group_loss < best_loss_at_smaller_volume:
            nondominated.extend(
                item
                for _volume, loss, _index, item in group
                if loss == minimum_group_loss
            )
        best_loss_at_smaller_volume = min(
            best_loss_at_smaller_volume, minimum_group_loss
        )
        start = end
    return nondominated


def _aggregate_pareto(runtime: Path, records: list[dict]) -> dict:
    candidates = []
    near = []
    for record in records:
        result_ref = record.get("result") or {}
        local = Path(str(result_ref.get("local_path") or ""))
        if not record.get("authenticated") or not local.is_file():
            continue
        if runtime.resolve() not in local.resolve().parents:
            continue
        if sha256(local) != result_ref.get("sha256"):
            continue
        result = read_json(local)
        for candidate in result.get("candidates") or []:
            item = dict(candidate)
            item["source_task_id"] = record["task_id"]
            item["source_seed"] = record["seed"]
            candidates.append(item)
        for candidate in (
            (result.get("next_target_fea_batch_plan") or {}).get("candidates") or []
        )[:3]:
            item = dict(candidate)
            item["source_task_id"] = record["task_id"]
            item["source_seed"] = record["seed"]
            near.append(item)
    unique = {}
    for item in candidates:
        params = item.get("decoded_params") or {}
        digest = canonical_sha(params)
        unique.setdefault(digest, item)
    values = list(unique.values())
    nondominated = _two_objective_nondominated(values)
    nondominated.sort(key=lambda item: (float(item["volume_L"]), float(item["total_loss_W"])))
    near.sort(
        key=lambda item: (
            float(item["target_acquisition_score"])
            if item.get("target_acquisition_score") is not None
            else 1e300
        )
    )
    completed = [
        record for record in records
        if record.get("authenticated") is True
        and record.get("terminal_state") == "completed"
    ]
    constraint_names = sorted({
        name for record in completed
        for name in (record.get("constraint_minimum_G") or {})
    })
    independent_minima = {
        name: min(
            float(record["constraint_minimum_G"][name])
            for record in completed
            if name in (record.get("constraint_minimum_G") or {})
        )
        for name in constraint_names
    }
    simultaneous_best = min(
        completed,
        key=lambda record: (
            float(record["terminal_minimum_total_positive_violation"])
            if record.get("terminal_minimum_total_positive_violation") is not None
            else 1e300
        ),
        default=None,
    )
    tracked_groups = {
        "Llt_robust": ["Llt_robust_band", "Llt_ensemble_disagreement"],
        "density": ["strict_full_density_support"],
        "Tx_thermal": [
            "temperature_robust_limit:T_max_Tx",
            "temperature_robust_limit:Tprobe_Tx_leeward_max",
        ],
    }
    bottleneck_tracking = {
        group: {
            name: {
                "independent_minimum_G": independent_minima.get(name),
                "pass_reached": (
                    independent_minima.get(name) is not None
                    and independent_minima[name] <= 0.0
                ),
            }
            for name in names
        }
        for group, names in tracked_groups.items()
    }
    pareto_preview_limit = 128
    if len(nondominated) <= pareto_preview_limit:
        pareto_preview_indexes = list(range(len(nondominated)))
        pareto_preview_method = "all"
    elif pareto_preview_limit == 1:
        pareto_preview_indexes = [0]
        pareto_preview_method = "minimum_volume_only"
    else:
        # ``nondominated`` is ordered from minimum volume to minimum loss.
        # Sampling the whole ordered front retains both objective extremes and
        # deterministic diversity.  Taking the first N points instead silently
        # discarded the minimum-loss end whenever the front exceeded N.
        pareto_preview_indexes = [
            index * (len(nondominated) - 1) // (pareto_preview_limit - 1)
            for index in range(pareto_preview_limit)
        ]
        pareto_preview_method = (
            "evenly_spaced_volume_order_including_objective_extremes"
        )
    pareto_preview = [
        nondominated[index] for index in pareto_preview_indexes
    ]
    return {
        "schema_version": "mft-tier1-slurm-aggregate-pareto-v1",
        "updated_at": now(),
        "authenticated_terminal_count": sum(
            bool(record.get("authenticated")) for record in records
        ),
        "source_candidate_count": len(candidates),
        "unique_candidate_count": len(values),
        "pareto_count": len(nondominated),
        "candidate_preview_count": len(pareto_preview),
        "candidate_preview_limit": pareto_preview_limit,
        "candidate_preview_truncated": (
            len(nondominated) > len(pareto_preview)
        ),
        "candidate_preview_selection": {
            "method": pareto_preview_method,
            "ordered_by": ["volume_L", "total_loss_W"],
            "includes_minimum_volume": 0 in pareto_preview_indexes,
            "includes_minimum_loss": (
                bool(nondominated)
                and len(nondominated) - 1 in pareto_preview_indexes
            ),
            "selected_front_indexes": pareto_preview_indexes,
        },
        "candidates": pareto_preview,
        "near_feasible_count": len(near),
        "near_feasible": near[:64],
        "constraint_independent_minimum_G": independent_minima,
        "constraint_independent_pass_reached": {
            name: value <= 0.0 for name, value in independent_minima.items()
        },
        "simultaneous_best": (
            {
                "task_id": simultaneous_best["task_id"],
                "seed": simultaneous_best["seed"],
                "minimum_total_positive_violation": simultaneous_best[
                    "terminal_minimum_total_positive_violation"
                ],
                "constraint_G": simultaneous_best.get(
                    "terminal_best_constraint_G"
                ),
            }
            if simultaneous_best is not None else None
        ),
        "bottleneck_tracking": bottleneck_tracking,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
    }


def _task_state(task: dict) -> str:
    return str(task.get("status") or task.get("state") or "unknown").lower()


def _sealed_status_snapshot(runtime: Path) -> tuple[dict, dict, str]:
    """Read one canonical status only through its containment and SHA seal."""

    root = runtime.resolve(strict=True)
    index_path = root / "canonical" / "index.json"
    index = read_json(index_path)
    if Path(index.get("path_containment_root", "")).resolve() != root:
        raise RuntimeError(f"canonical containment mismatch: {root}")
    status_ref = index.get("status") or {}
    status_path = Path(status_ref.get("path", "")).resolve(strict=True)
    if root not in status_path.parents:
        raise RuntimeError(f"canonical status escaped runtime: {status_path}")
    status_sha = sha256(status_path)
    if status_sha != status_ref.get("sha256"):
        raise RuntimeError(f"canonical status SHA mismatch: {status_path}")
    status = read_json(status_path)
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("cohort_id") != index.get("active_cohort_id")
        or status.get("updated_at") != index.get("updated_at")
    ):
        raise RuntimeError(f"canonical status identity mismatch: {status_path}")
    return index, status, status_sha


def _read_only_preview_union(live: dict, deep: dict) -> list[dict]:
    """Merge authenticated preview rows without mutating either canonical."""

    unique: dict[str, dict] = {}
    for origin, status in (("live", live), ("deep", deep)):
        candidates = (status.get("aggregate_pareto") or {}).get("candidates") or []
        for candidate in candidates:
            try:
                loss = float(candidate["total_loss_W"])
                volume = float(candidate["volume_L"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(loss) or not math.isfinite(volume):
                continue
            identity = canonical_sha({
                "decoded_params": candidate.get("decoded_params"),
                "total_loss_W": loss,
                "volume_L": volume,
            })
            row = dict(candidate)
            row["comparison_origin"] = origin
            row["comparison_identity_sha256"] = identity
            unique.setdefault(identity, row)
    rows = list(unique.values())
    nondominated = []
    for row in rows:
        loss = float(row["total_loss_W"])
        volume = float(row["volume_L"])
        if any(
            float(other["total_loss_W"]) <= loss
            and float(other["volume_L"]) <= volume
            and (
                float(other["total_loss_W"]) < loss
                or float(other["volume_L"]) < volume
            )
            for other in rows
        ):
            continue
        nondominated.append(row)
    nondominated.sort(
        key=lambda row: (float(row["volume_L"]), float(row["total_loss_W"]))
    )
    if len(nondominated) <= 128:
        return nondominated
    indexes = {
        round(index * (len(nondominated) - 1) / 127) for index in range(128)
    }
    return [nondominated[index] for index in sorted(indexes)]


def _publish_read_only_comparison(runtime: Path) -> None:
    """Publish a sealed live+deep preview union under the deep runtime only."""

    output = runtime.resolve() / "comparison" / "live_plus_deep.json"
    try:
        live_index, live_status, live_sha = _sealed_status_snapshot(LIVE_RUNTIME)
        deep_index, deep_status, deep_sha = _sealed_status_snapshot(runtime)
        identity_fields = (
            "constraint_version", "hard_spec_sha256",
            "source_model_manifest_sha256", "deployment_model_manifest_sha256",
            "temperature_constraint_contract_sha256",
        )
        mismatches = [
            field for field in identity_fields
            if live_status.get(field) != deep_status.get(field)
        ]
        if mismatches:
            raise RuntimeError(
                "live/deep comparison identity mismatch: " + ",".join(mismatches)
            )
        candidates = _read_only_preview_union(live_status, deep_status)
        deep_authenticated = int(
            (deep_status.get("aggregate_pareto") or {}).get(
                "authenticated_terminal_count", 0
            )
        )
        deep_authenticated_completed = sum(
            record.get("authenticated") is True
            and str(record.get("terminal_state") or "").lower() == "completed"
            for record in deep_status.get("terminal_results") or []
        )
        deep_preview_count = int(
            (deep_status.get("aggregate_pareto") or {}).get(
                "candidate_preview_count", 0
            )
        )
        deep_terminal_ready = deep_authenticated_completed > 0
        atomic_json(output, {
            "schema_version": "mft-tier1-live-deep-pareto-comparison-v1",
            "updated_at": now(),
            "healthy": True,
            "mode": "read_only_authenticated_preview_union",
            "writes_live_canonical": False,
            "identity": {
                field: live_status.get(field) for field in identity_fields
            },
            "components": {
                "live": {
                    "runtime": str(LIVE_RUNTIME.resolve()),
                    "cohort_id": live_index["active_cohort_id"],
                    "status_sha256": live_sha,
                    "updated_at": live_status["updated_at"],
                    "authenticated_terminal_count": int(
                        (live_status.get("aggregate_pareto") or {}).get(
                            "authenticated_terminal_count", 0
                        )
                    ),
                },
                "deep": {
                    "runtime": str(runtime.resolve()),
                    "cohort_id": deep_index["active_cohort_id"],
                    "status_sha256": deep_sha,
                    "updated_at": deep_status["updated_at"],
                    "authenticated_terminal_count": deep_authenticated,
                    "authenticated_completed_terminal_count": (
                        deep_authenticated_completed
                    ),
                    "candidate_preview_count": deep_preview_count,
                },
            },
            "deep_terminal_ready": deep_terminal_ready,
            "ui_promotion_eligible": (
                deep_terminal_ready and deep_preview_count > 0
            ),
            "candidate_preview_count": len(candidates),
            "candidate_preview_limit": 128,
            "candidates": candidates,
            "promotion_contract": (
                "consume_this_artifact_as_a_second_read_only_inventory_after_"
                "deep_terminal_ready;never_replace_live_canonical"
            ),
            "error": None,
        })
    except Exception as exc:
        atomic_json(output, {
            "schema_version": "mft-tier1-live-deep-pareto-comparison-v1",
            "updated_at": now(),
            "healthy": False,
            "mode": "read_only_authenticated_preview_union",
            "writes_live_canonical": False,
            "deep_terminal_ready": False,
            "ui_promotion_eligible": False,
            "candidates": [],
            "error": f"{type(exc).__name__}:{exc}",
        })


def _publish_status(
    pointer: dict, tasks: list[dict], *, runtime: Path, controller: dict,
    submissions: list[dict] | None = None, error: str | None = None,
) -> dict:
    current = pointer["current"]
    prefix = _task_prefix(current["cohort_id"])
    states = Counter(_task_state(task) for task in tasks)
    active = [task for task in tasks if _task_state(task) in ACTIVE_STATES]
    seeds = [_seed_from_name(str(task.get("name") or ""), prefix) for task in tasks]
    seeds = [seed for seed in seeds if seed is not None]
    duplicate_seeds = sorted(
        seed for seed, count in Counter(seeds).items() if count > 1
    )
    if duplicate_seeds:
        error = error or f"duplicate scheduler seed identities: {duplicate_seeds[:10]}"
    target = int(current["rolling_target"])
    active_count = len(active)
    deficit = max(0, target - active_count)
    at_target = active_count == target
    over_target = active_count > target
    if over_target:
        error = error or (
            f"active scheduler inventory exceeds rolling target: "
            f"{active_count}>{target}"
        )
    cohort_dir = runtime.resolve() / "cohorts" / current["cohort_id"]
    status_path = cohort_dir / "status.json"
    compact_tasks = []
    ordered_tasks = sorted(
        tasks,
        key=lambda item: int(item.get("id") or item.get("task_id") or 0),
        reverse=True,
    )[:CANONICAL_TASK_WINDOW_LIMIT]
    for task in ordered_tasks:
        compact_tasks.append({
            "task_id": int(task.get("id") or task.get("task_id")),
            "name": task.get("name"),
            "status": _task_state(task),
            "seed": _seed_from_name(str(task.get("name") or ""), prefix),
            "cpus": int(task.get("cpus") or 0),
            "memory_mb": int(task.get("memory_mb") or 0),
            "priority": int(task.get("priority") or 0),
            "account_name": task.get("account_name") or "",
            "slurm_job_id": task.get("slurm_job_id") or "",
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
            "exit_code": task.get("exit_code"),
        })
    # A failed remote harvest is persisted with ``authenticated=False`` so a
    # later controller tick can retry it.  Keep that retry state on disk, but
    # never expose it through the sealed status: strict readers must only see
    # terminal rows whose local artifacts and identities were authenticated.
    terminal_records = [
        record
        for record in _load_terminal_records(runtime, current["cohort_id"])
        if record.get("authenticated") is True
    ]
    terminal_records, aggregate_pareto = _bounded_terminal_snapshot(
        runtime, terminal_records
    )
    # Timestamp the publish after bounded aggregation, immediately before the
    # atomic status/index writes, so freshness measures reader-visible age.
    updated_at = now()
    status = {
        "schema_version": STATUS_SCHEMA,
        "updated_at": updated_at,
        "freshness_deadline_seconds": FRESHNESS_DEADLINE_SECONDS,
        "cohort_id": current["cohort_id"],
        "constraint_version": current["constraint_version"],
        "hard_spec": current["hard_spec"],
        "hard_spec_sha256": current["hard_spec_sha256"],
        "attestation_schema_version": current.get(
            "attestation_schema_version"
        ),
        "task_schema_version": current.get("task_schema_version"),
        "temperature_constraint_contract": current[
            "temperature_constraint_contract"
        ],
        "temperature_constraint_contract_sha256": current.get(
            "temperature_constraint_contract_sha256"
        ),
        "source_model_manifest_sha256": current[
            "source_model_manifest_sha256"
        ],
        "deployment_model_manifest_sha256": current[
            "deployment_model_manifest_sha256"
        ],
        "bundle_manifest_sha256": current["bundle_manifest_sha256"],
        "nsga_code_revision": current["nsga_code_revision"],
        "search_profile": search_profile(),
        "optimizer_resonance_scale_Hz": current[
            "optimizer_resonance_scale_Hz"
        ],
        "optimizer_core_thermal_scale_C": current.get(
            "optimizer_core_thermal_scale_C"
        ),
        "optimizer_Llt_scale_uH": current.get("optimizer_Llt_scale_uH"),
        "optimizer_all_active_thermal_scale_C": current.get(
            "optimizer_all_active_thermal_scale_C"
        ),
        "acquisition_ranking_contract": current.get(
            "acquisition_ranking_contract"
        ),
        "acquisition_ranking_contract_sha256": current.get(
            "acquisition_ranking_contract_sha256"
        ),
        "rolling": {
            "enabled": current.get("refill_allowed") is True,
            "unbounded_seed_stream": DEEP_CROSSOVER_ISLAND is None,
            "seed_window_end_exclusive": (
                DEEP_CROSSOVER_ISLAND.seed_window_end_exclusive
                if DEEP_CROSSOVER_ISLAND is not None else None
            ),
            "stop_condition": "explicit_operator_stop_only",
            "target_active_plus_queued": target,
            "active_plus_queued": active_count,
            "deficit": deficit,
            "at_target": at_target,
            "refill_in_progress": deficit > 0,
            "over_target": over_target,
            "bounded_submit_burst": 32,
            "seed_identity_count": len(set(seeds)),
            "duplicate_seed_count": len(duplicate_seeds),
            "terminal_completion_triggers_refill": True,
            "model_pointer_change_starts_new_cohort": True,
            "legacy_cohorts_are_not_cancelled": True,
        },
        "state_counts": dict(sorted(states.items())),
        "scheduler_task_count": len(tasks),
        "latest_tasks": compact_tasks,
        "latest_submissions": submissions or [],
        "terminal_results": terminal_records,
        "aggregate_pareto": aggregate_pareto,
        "controller": controller,
        "error": error,
        "health_contract_version": HEALTH_CONTRACT_VERSION,
        # A normal refill deficit is progress state, not a controller or
        # inventory-integrity failure. Consumers remain strict on ``healthy``;
        # exact inventory attainment is reported truthfully by ``at_target``.
        "healthy": error is None,
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    atomic_json(status_path, status)
    pointer_path = runtime.resolve() / "canonical" / "model_pointer.json"
    index = {
        "schema_version": INDEX_SCHEMA,
        "updated_at": updated_at,
        "freshness_deadline_seconds": FRESHNESS_DEADLINE_SECONDS,
        "active_cohort_id": current["cohort_id"],
        "constraint_version": current["constraint_version"],
        "hard_spec": current["hard_spec"],
        "hard_spec_sha256": current["hard_spec_sha256"],
        "attestation_schema_version": current.get(
            "attestation_schema_version"
        ),
        "task_schema_version": current.get("task_schema_version"),
        "temperature_constraint_contract": current[
            "temperature_constraint_contract"
        ],
        "temperature_constraint_contract_sha256": current.get(
            "temperature_constraint_contract_sha256"
        ),
        "source_model_manifest_sha256": current[
            "source_model_manifest_sha256"
        ],
        "deployment_model_manifest_sha256": current[
            "deployment_model_manifest_sha256"
        ],
        "search_profile": search_profile(),
        "optimizer_resonance_scale_Hz": current[
            "optimizer_resonance_scale_Hz"
        ],
        "optimizer_core_thermal_scale_C": current.get(
            "optimizer_core_thermal_scale_C"
        ),
        "optimizer_Llt_scale_uH": current.get("optimizer_Llt_scale_uH"),
        "optimizer_all_active_thermal_scale_C": current.get(
            "optimizer_all_active_thermal_scale_C"
        ),
        "acquisition_ranking_contract": current.get(
            "acquisition_ranking_contract"
        ),
        "acquisition_ranking_contract_sha256": current.get(
            "acquisition_ranking_contract_sha256"
        ),
        "model_pointer": {
            "path": str(pointer_path),
            "sha256": sha256(pointer_path),
            "schema_version": POINTER_SCHEMA,
        },
        "status": {
            "path": str(status_path),
            "sha256": sha256(status_path),
            "schema_version": STATUS_SCHEMA,
        },
        "path_containment_root": str(runtime.resolve()),
        "production_eligible": False,
        "fea_submission_approved": False,
        "fea_submission_performed": False,
        "aedt_used": False,
        "automatic_promotion_allowed": False,
    }
    atomic_json(runtime.resolve() / "canonical" / "index.json", index)
    atomic_json(runtime.resolve() / "canonical" / "controller_status.json", {
        "schema_version": "mft-tier1-slurm-controller-heartbeat-v1",
        "updated_at": updated_at,
        "controller": controller,
        "active_cohort_id": current["cohort_id"],
        "active_plus_queued": active_count,
        "target": target,
        "deficit": deficit,
        "at_target": at_target,
        "healthy": error is None,
        "health_contract_version": HEALTH_CONTRACT_VERSION,
        "freshness_deadline_seconds": FRESHNESS_DEADLINE_SECONDS,
        "error": error,
        "production_eligible": False,
        "fea_submission_approved": False,
    })
    if runtime.resolve() == RUNTIME.resolve():
        _publish_read_only_comparison(runtime)
    return status


def _load_state(runtime: Path) -> dict:
    path = runtime.resolve() / "controller" / "state.json"
    if path.is_file():
        state = read_json(path)
    else:
        state = {
            "schema_version": "mft-tier1-slurm-controller-state-v1",
            "next_seed_by_cohort": {},
        }
    return state


def _next_seed(runtime: Path, cohort_id: str, tasks: list[dict]) -> int:
    state = _load_state(runtime)
    prefix = _task_prefix(cohort_id)
    inventory = [
        _seed_from_name(str(task.get("name") or ""), prefix) for task in tasks
    ]
    inventory = [seed for seed in inventory if seed is not None]
    stored = int(state["next_seed_by_cohort"].get(cohort_id, SEED_START))
    next_seed = max(stored, max(inventory, default=stored - 1) + 1)
    if (
        DEEP_CROSSOVER_ISLAND is not None
        and next_seed >= DEEP_CROSSOVER_ISLAND.seed_window_end_exclusive
    ):
        raise RuntimeError("deep crossover seed window is exhausted")
    return next_seed


def _save_next_seed(runtime: Path, cohort_id: str, seed: int) -> None:
    state = _load_state(runtime)
    state["next_seed_by_cohort"][cohort_id] = int(seed)
    state["updated_at"] = now()
    atomic_json(runtime.resolve() / "controller" / "state.json", state)


def reconcile_once(
    scheduler_url: str = DEFAULT_SCHEDULER_URL, runtime: Path = RUNTIME,
    apply: bool = False, controller_identity: dict | None = None,
) -> dict:
    pointer_path = runtime.resolve() / "canonical" / "model_pointer.json"
    pointer = read_json(pointer_path)
    if pointer.get("schema_version") != POINTER_SCHEMA:
        raise RuntimeError("unsupported model pointer schema")
    current = pointer.get("current") or {}
    _validate_current_attestation(current)
    if _is_v3_current(current):
        feedback = REPO / "tools" / "tier1_resonance_feedback.py"
        expected_version, expected_spec, expected_revision = _feedback_contract(
            feedback
        )
        if (
            current.get("constraint_version") != expected_version
            or current.get("hard_spec_sha256") != canonical_sha(expected_spec)
            or current.get("hard_spec") != expected_spec
            or current.get("nsga_code_revision") != expected_revision
        ):
            raise RuntimeError(
                "v3 model pointer is not the current constraint envelope"
            )
    if current.get("refill_allowed") is not True:
        raise RuntimeError("active cohort pointer explicitly refuses refill")
    controller = controller_identity or {
        "pid": os.getpid(), "host": socket.gethostname(), "mode": "once",
    }
    tasks = list_cohort_tasks(scheduler_url, current["cohort_id"])
    assert_existing_task_attestation(pointer, tasks)
    harvest_terminal_results(
        scheduler_url, pointer, tasks, runtime,
        max_new=HARVEST_MAX_NEW_PER_TICK,
    )
    active = [task for task in tasks if _task_state(task) in ACTIVE_STATES]
    status = _publish_status(
        pointer, tasks, runtime=runtime, controller=controller
    )
    deficit = int(current["rolling_target"]) - len(active)
    if not apply or deficit <= 0:
        return status
    burst = min(32, deficit)
    seed = _next_seed(runtime, current["cohort_id"], tasks)
    submissions = []
    prefix = _task_prefix(current["cohort_id"])
    existing_seeds = {
        value for task in tasks
        if (value := _seed_from_name(str(task.get("name") or ""), prefix)) is not None
    }
    while len(submissions) < burst:
        if (
            DEEP_CROSSOVER_ISLAND is not None
            and seed >= DEEP_CROSSOVER_ISLAND.seed_window_end_exclusive
        ):
            raise RuntimeError("deep crossover seed window is exhausted")
        if seed in existing_seeds:
            seed += 1
            continue
        payload = task_payload(pointer, seed)
        response = api_json(
            scheduler_url.rstrip("/") + "/api/tasks", method="POST",
            payload=payload, timeout=30,
        )
        record = {
            "task_id": int(response["task_id"]),
            "seed": seed,
            "name": payload["name"],
            "dedupe_key": payload["dedupe_key"],
            "deduped": bool(response.get("deduped")),
            "submitted_at": now(),
        }
        submissions.append(record)
        existing_seeds.add(seed)
        tasks.insert(0, {
            "id": record["task_id"], "name": record["name"], "status": "queued",
            "cpus": 8, "memory_mb": 32768,
            "priority": payload["priority"],
        })
        seed += 1
        _save_next_seed(runtime, current["cohort_id"], seed)
        if len(submissions) % 4 == 0 and len(submissions) < burst:
            _publish_status(
                pointer, tasks, runtime=runtime, controller=controller,
                submissions=submissions[-32:],
            )
    return _publish_status(
        pointer, tasks, runtime=runtime, controller=controller,
        submissions=submissions,
    )


class OwnerLock:
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        if self.stream.read(1) == b"":
            self.stream.seek(0)
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("rolling controller already has an owner") from exc
        else:
            import fcntl
            try:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("rolling controller already has an owner") from exc
        return self

    def __exit__(self, *_args):
        if self.stream is None:
            return
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def controller_worker(
    scheduler_url: str, runtime: Path, poll_seconds: float,
    supervisor_pid: int | None = None,
) -> None:
    started_at = now()
    identity = {
        "pid": os.getpid(), "supervisor_pid": supervisor_pid,
        "host": socket.gethostname(), "started_at": started_at,
        "poll_seconds": poll_seconds, "mode": "rolling-worker",
        "single_owner_lock": str(runtime.resolve() / "controller" / "owner.lock"),
        "crash_restart": supervisor_pid is not None,
    }
    while True:
        try:
            status = reconcile_once(
                scheduler_url, runtime, apply=True, controller_identity=identity
            )
            deficit = int(status["rolling"]["deficit"])
            time.sleep(0.25 if deficit > 0 else poll_seconds)
        except Exception as exc:
            atomic_json(runtime.resolve() / "canonical" / "controller_status.json", {
                "schema_version": "mft-tier1-slurm-controller-heartbeat-v1",
                "updated_at": now(), "controller": identity,
                "error": f"{type(exc).__name__}:{exc}",
                "production_eligible": False, "fea_submission_approved": False,
            })
            time.sleep(min(poll_seconds, 5.0))


def supervisor(scheduler_url: str, runtime: Path, poll_seconds: float) -> None:
    with OwnerLock(runtime.resolve() / "controller" / "owner.lock"):
        started_at = now()
        while True:
            command = [
                sys.executable, str(Path(__file__).resolve()), "worker",
                "--scheduler-url", scheduler_url, "--runtime", str(runtime),
                "--poll-seconds", str(poll_seconds),
                "--supervisor-pid", str(os.getpid()),
            ]
            process = subprocess.Popen(command)
            process.wait()
            atomic_json(runtime.resolve() / "controller" / "last_restart.json", {
                "schema_version": "mft-tier1-slurm-controller-restart-v1",
                "supervisor_pid": os.getpid(), "supervisor_started_at": started_at,
                "worker_exit_code": process.returncode, "restarted_at": now(),
            })
            time.sleep(1.0)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    prepare_parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE)
    prepare_parser.add_argument("--warm-start", type=Path, default=DEFAULT_WARM)
    prepare_parser.add_argument("--base-plan", type=Path, default=DEFAULT_BASE_PLAN)
    prepare_parser.add_argument("--runtime", type=Path, default=RUNTIME)
    prepare_parser.add_argument("--anchor-preflight", type=Path, default=None)
    prepare_parser.add_argument(
        "--deep-crossover-preflight", type=Path, default=None,
    )

    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("--plan", type=Path, required=True)
    stage_parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    stage_parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    stage_parser.add_argument("--account", default="harry261")
    stage_parser.add_argument("--apply", action="store_true")

    pointer_parser = commands.add_parser("publish-pointer")
    pointer_parser.add_argument("--plan", type=Path, required=True)
    pointer_parser.add_argument("--runtime", type=Path, default=RUNTIME)

    once_parser = commands.add_parser("reconcile")
    once_parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    once_parser.add_argument("--runtime", type=Path, default=RUNTIME)
    once_parser.add_argument("--apply", action="store_true")

    priority_parser = commands.add_parser("migrate-priority")
    priority_parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    priority_parser.add_argument("--runtime", type=Path, default=RUNTIME)
    priority_parser.add_argument("--apply", action="store_true")

    supervisor_parser = commands.add_parser("supervisor")
    supervisor_parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    supervisor_parser.add_argument("--runtime", type=Path, default=RUNTIME)
    supervisor_parser.add_argument("--poll-seconds", type=float, default=10.0)

    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    worker_parser.add_argument("--runtime", type=Path, default=RUNTIME)
    worker_parser.add_argument("--poll-seconds", type=float, default=10.0)
    worker_parser.add_argument("--supervisor-pid", type=int, default=None)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare(
            args.model_dir, args.code_root, args.warm_start, args.base_plan,
            args.runtime, args.anchor_preflight, args.deep_crossover_preflight,
        )
    elif args.command == "stage":
        result = stage(
            args.plan, args.accounts, args.scheduler_source, args.account,
            args.apply,
        )
    elif args.command == "publish-pointer":
        result = publish_pointer(args.plan, args.runtime)
    elif args.command == "reconcile":
        result = reconcile_once(
            args.scheduler_url, args.runtime, apply=args.apply
        )
    elif args.command == "migrate-priority":
        result = migrate_priority(
            args.scheduler_url, args.runtime, apply=args.apply
        )
    elif args.command == "supervisor":
        supervisor(args.scheduler_url, args.runtime, args.poll_seconds)
        return 0
    else:
        controller_worker(
            args.scheduler_url, args.runtime, args.poll_seconds,
            args.supervisor_pid,
        )
        return 0
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
