"""GET-only updater for the post-deadline MFT task cards on port 8010.

The Scheduler remains a separate service.  This module only performs exact
``GET /api/tasks/{id}`` reads and atomically merges lifecycle state into the
explicit Codex UI status artifact.  Scheduler success is never interpreted as
scientific, collection, canonical, or production success.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DEFAULT_SCHEDULER_URL = "http://127.0.0.1:8002"
DEFAULT_STATUS_FILE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_goal_20260726\ui\codex-work-status.json"
)
DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\local_symmetric_selection_watch_v1\state.json"
)
DEFAULT_REFERENCE_BASELINE_GUI_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\local_reference_drawing260706_symmetric_gui_thermal_retry_v2"
)
DEFAULT_TARGET_AXIS_COLLECTOR_STATE_FILE = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_lm2mh_old16_plus_splittemp512_global_nds_v3"
    r"\collector_status.json"
)
DEFAULT_INTERVAL_SECONDS = 60
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_LOCAL_SEALED_STATE_BYTES = 8 * 1024 * 1024
CAMPAIGN_SUBMITTED_FLOOR = 130
SYNC_KEY = "postdeadline_task_sync"
SYNC_SCHEMA = "mft-goal-postdeadline-ui-sync-v1"
PID_SCHEMA = "mft-goal-postdeadline-ui-updater-pid-v1"
LOG_SCHEMA = "mft-goal-postdeadline-ui-updater-event-v1"
STATUS_SCHEMA = "mft-codex-work-status-v1"
POSTSUCCESS_STATE_SCHEMA = "mft-goal-postdeadline-standard-postsuccess-state-v1"
THERMAL_BRIDGE_STATE_SCHEMA = "mft-corrected-thermal-terminal-transport-watch-state-v1"
STANDARD_FULL_CONTINUATION_STATE_SCHEMA = "mft-goal-standard-full-continuation-state-v1"
LOCAL_SYMMETRIC_SELECTION_STATE_SCHEMA = (
    "mft-goal-local-symmetric-selection-watch-state-v1"
)
TARGET_AXIS_COLLECTOR_STATE_SCHEMA = (
    "mft-goal-fixed-lm2mh-targeted-global-nds-v3"
)
TARGET_AXIS_PARETO_MANIFEST_SCHEMA = (
    "mft-goal-fixed-lm2mh-targeted-pareto-manifest-v3"
)
REFERENCE_THERMAL_TERMINAL_SCHEMA = (
    "mft-reference-gui-thermal-terminal-state-v1"
)
REFERENCE_THERMAL_RETRY_TERMINAL_SCHEMA = (
    "mft-reference-gui-thermal-retry-terminal-state-v1"
)
FINAL_GATE_PENDING_SCHEMA = "mft-goal-final-solver-package-pending-v1"
FINAL_GATE_SEAL_SCHEMA = "mft-goal-final-solver-package-seal-v1"
FINAL_PACKAGE_NAME = "final_solver_package_v1"
SUBMITTED_PATTERN = re.compile(r"\bSUBMITTED\s+(\d+)(?!\d)", re.IGNORECASE)
COLLECTIONS_PATTERN = re.compile(r"\bCOLLECTIONS?\s+(\d+)(?!\d)", re.IGNORECASE)


class UpdaterError(RuntimeError):
    """Raised when the GET-only updater cannot safely merge a cycle."""


class ConcurrentStatusUpdate(UpdaterError):
    """Raised when another producer changed the status during a merge."""


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    card_id: str
    task_name: str
    model_label: str
    candidate_label: str
    requested_node: str
    cpus: int
    memory_mb: int
    timeout_seconds: int
    inner_solver_seconds: int | None = None
    requested_account: str | None = None
    max_workers_per_node: int | None = None
    expected_same_node_as_task_id: int = 0
    search_only: bool = False
    submission_receipt_sha256: str | None = None
    final_seal_sha256: str | None = None
    expected_allocation_id: int | None = None
    expected_slurm_job_id: str | None = None
    allocation_force_cancel_at_kst: str | None = None
    task_timeout_at_kst: str | None = None
    force_cancel_lead_seconds: int | None = None
    terminal_failure_override: str | None = None
    selection_superseded_by_task_id: int | None = None
    selection_failover_for_task_id: int | None = None
    new_allocation_required: bool = False


@dataclass(frozen=True)
class AuxiliaryTaskSpec:
    task_id: int
    task_name: str
    role: str
    cpus: int
    memory_mb: int
    timeout_seconds: int
    max_workers_per_node: int
    seed: int | None = None
    primary_turns: int | None = None


TASK_SPECS = (
    TaskSpec(
        task_id=96324,
        card_id="postdeadline-symmetric-retry-96324",
        task_name=(
            "mft-goal-corrected-thermal-l96230-b7c30cb70b95-postdeadline-r6-n111"
        ),
        model_label="SYMMETRIC",
        candidate_label="b7c30cb70b95",
        requested_node="n111",
        cpus=8,
        memory_mb=294912,
        timeout_seconds=45000,
        inner_solver_seconds=43200,
        requested_account="r1jae262",
        terminal_failure_override=(
            "Icepak native ThermalSetup execution error after an authenticated "
            "mesh preflight; no Fluent process or temperature result was "
            "produced. The later NaN JSON serialization error only affected "
            "the failure receipt."
        ),
    ),
    TaskSpec(
        task_id=96325,
        card_id="postdeadline-standard-retry-96325",
        task_name=("mft-goal-diag-standard-postdeadline-r1-l96231-efffb6518d4e-n107"),
        model_label="STANDARD",
        candidate_label="efffb6518d4e",
        requested_node="n107",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        search_only=True,
    ),
    TaskSpec(
        task_id=96326,
        card_id="postdeadline-full-retry-96326",
        task_name="mft-goal-postdeadline-diagnostic-full-retry-t96307-v1",
        model_label="FULL",
        candidate_label="task96307 retry",
        requested_node="n116",
        cpus=16,
        memory_mb=98304,
        timeout_seconds=86400,
        inner_solver_seconds=79200,
        requested_account="dhj02",
    ),
    TaskSpec(
        task_id=96327,
        card_id="postdeadline-standard-retry-96327",
        task_name=("mft-goal-diag-standard-postdeadline-r2-l96208-b6a83bfc7212-n109"),
        model_label="STANDARD",
        candidate_label="b6a83bfc7212",
        requested_node="n109",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        search_only=True,
    ),
    TaskSpec(
        task_id=96328,
        card_id="postdeadline-standard-official6-96328",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official6-s96185-2772aed82a8c-n113"
        ),
        model_label="STANDARD OFFICIAL #6",
        candidate_label="official#6 2772aed82a8c",
        requested_node="n113",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="dw16",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "42942394a873e40181f9074f2625807224239b291bcf2f4cf2ccacde50b11edc"
        ),
        final_seal_sha256=(
            "ff034d51e8da2ce97c4cd06f44767131e6d60c01f62d65d58ac0afc9bee4abe6"
        ),
        expected_allocation_id=14620,
        expected_slurm_job_id="829579",
        allocation_force_cancel_at_kst="2026-07-27T04:07:51+09:00",
        task_timeout_at_kst="2026-07-27T08:09:47+09:00",
        force_cancel_lead_seconds=14516,
    ),
    TaskSpec(
        task_id=96329,
        card_id="postdeadline-standard-official8-96329",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official8-s96141-622097dde126-n114"
        ),
        model_label="STANDARD OFFICIAL #8",
        candidate_label="official#8 622097dde126",
        requested_node="n114",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="jji0930",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "8d7e59fe51523f22e3692ef87ee529d619b4a18b5e0a5353664bef9a73500ff5"
        ),
        final_seal_sha256=(
            "5319a8a4dceb27082b91fc6badd540221298eeb9f324e2fa30e76e529313d3ec"
        ),
        selection_superseded_by_task_id=96333,
    ),
    TaskSpec(
        task_id=96330,
        card_id="postdeadline-standard-official1-96330",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official1-s96009-896084a59793-n110"
        ),
        model_label="STANDARD OFFICIAL #1",
        candidate_label="official#1 896084a59793",
        requested_node="n110",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="dhj02",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "d71987f49b062132fdf90a358bcc819a2cf9fd2e74c76fecad04e22ea9329233"
        ),
        final_seal_sha256=(
            "1c511227687977cbf001a0e4a77c5060d41fcd397b74756062bc6e9feb0f9332"
        ),
    ),
    TaskSpec(
        task_id=96331,
        card_id="postdeadline-standard-official12-96331",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official12-s96185-828cb282cf4f-n112"
        ),
        model_label="STANDARD OFFICIAL #12",
        candidate_label="official#12 828cb282cf4f",
        requested_node="n112",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="r1jae262",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "2e207daa33165e8921816e56beb2458d1dcd8515d6cb0b8d5a25cb0cf3cde532"
        ),
        final_seal_sha256=(
            "a743046a7a878030a538988e1c5e9270de1ef56a8e541e0130c5b55cc764937a"
        ),
    ),
    TaskSpec(
        task_id=96332,
        card_id="postdeadline-standard-official5-96332",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official5-s95913-909d249ebe45-n115"
        ),
        model_label="STANDARD OFFICIAL #5",
        candidate_label="official#5 909d249ebe45",
        requested_node="n115",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="jji0930",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "4275d9e8e004f3f9f57d9483ca9e7b8a48d410778f39a80848c95c396ed2171c"
        ),
        final_seal_sha256=(
            "ed28dd4d0c4ec904d2910323bc63bee2e13afddb59cd8e4d7f1a9d3d11487477"
        ),
        selection_superseded_by_task_id=96338,
    ),
    TaskSpec(
        task_id=96333,
        card_id="postdeadline-standard-official8-failover-96333",
        task_name=(
            "mft-goal-diag-standard-postdeadline-official8-failover-"
            "s96141-622097dde126-n111"
        ),
        model_label="STANDARD OFFICIAL #8 FAILOVER",
        candidate_label="official#8 622097dde126 n111 failover",
        requested_node="n111",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="r1jae262",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "8990b3f339ce36f66d4ecf5fdbbd111889d55b25d860d00e3d98e6508101180c"
        ),
        final_seal_sha256=(
            "d70eef5add63daa0148ad05af809dd231d1400a11794995a0fb81a3a6fedd45f"
        ),
        selection_failover_for_task_id=96329,
    ),
    TaskSpec(
        task_id=96337,
        card_id="postdeadline-standard-official5-direct-analyze-96337",
        task_name=(
            "mft-goal-diag-standard-official5-direct-analyze-samenode-v1-"
            "909d249ebe45-n115"
        ),
        model_label="STANDARD OFFICIAL #5 DIRECT ANALYZE ATTEMPT",
        candidate_label="official#5 909d249ebe45 same-node direct Analyze attempt",
        requested_node="n115",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="jji0930",
        max_workers_per_node=2,
        expected_same_node_as_task_id=96332,
        search_only=True,
        submission_receipt_sha256=(
            "c14d491f558308d14c1a151a0a9e63c91997921a09960e858cc495b66652bea4"
        ),
        final_seal_sha256=(
            "fb417009a6af6ce6a734f41ffe7855f4b3d9294454d93e2066c8f06b619448a3"
        ),
        expected_allocation_id=14650,
        expected_slurm_job_id="840582",
        terminal_failure_override=(
            "Pre-EM AEDT startup failed because the standalone-core opt-in "
            "authentication digest did not match. No electromagnetic or thermal "
            "solve result was produced; this is an operational failure, not a "
            "physics infeasibility."
        ),
        selection_superseded_by_task_id=96338,
    ),
    TaskSpec(
        task_id=96338,
        card_id="postdeadline-standard-official5-direct-analyze-r1-96338",
        task_name=(
            "mft-goal-diag-standard-official5-direct-analyze-samenode-r1-v2-"
            "909d249ebe45-n115"
        ),
        model_label="STANDARD OFFICIAL #5 DIRECT ANALYZE CORRECTED",
        candidate_label="official#5 909d249ebe45 corrected same-node direct Analyze",
        requested_node="n115",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=45300,
        inner_solver_seconds=43200,
        requested_account="jji0930",
        max_workers_per_node=2,
        expected_same_node_as_task_id=96332,
        search_only=True,
        submission_receipt_sha256=(
            "0731cf22e3f78f93da943cc9102b7d96a2719bfbe1ca65e782789b2d98bc30dd"
        ),
        final_seal_sha256=(
            "4e2802cd321727114926da7af8631d8f267fab37fa801d966966fdb54b75946e"
        ),
        expected_allocation_id=14650,
        expected_slurm_job_id="840582",
        selection_failover_for_task_id=96332,
    ),
    TaskSpec(
        task_id=96340,
        card_id="final-rounded-standard-symmetric-96340",
        task_name=(
            "mft-goal-final-standard-official5-rounded-r10-s4-v3-"
            "909d249ebe45-n113"
        ),
        model_label="FINAL ROUNDED STANDARD SYMMETRIC FEA",
        candidate_label="official#5 909d249ebe45 rounded R10/S4",
        requested_node="n113",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=12900,
        inner_solver_seconds=10800,
        requested_account="dw16",
        max_workers_per_node=2,
        search_only=True,
        submission_receipt_sha256=(
            "9b3ce37c8e28258554538469087c78b542ba2585e37e54b3fbde83031167f266"
        ),
        expected_allocation_id=14620,
        expected_slurm_job_id="829579",
    ),
    TaskSpec(
        task_id=96342,
        card_id="final-rounded-standard-symmetric-timeout-hedge-96342",
        task_name=(
            "mft-goal-final-standard-official5-rounded-r10-s4-hedge1-v1-"
            "909d249ebe45-n107"
        ),
        model_label="FINAL ROUNDED SYMMETRIC TIMEOUT HEDGE",
        candidate_label="same official#5 909d249ebe45 rounded R10/S4",
        requested_node="n107",
        cpus=8,
        memory_mb=98304,
        timeout_seconds=12900,
        inner_solver_seconds=10800,
        requested_account="harry261",
        max_workers_per_node=1,
        search_only=True,
        submission_receipt_sha256=(
            "0f6f62e34ba1760aa4c223dafccb3cf42f05f619fa145f7fb61ecaa4a3d30585"
        ),
        final_seal_sha256=(
            "8e5db1f9e96b957f0c81cfc8b395e82be466ff5d962046fc7bc039ecfe2a2c0c"
        ),
        new_allocation_required=True,
    ),
)

AXIS_V6_TASK_SPECS = tuple(
    AuxiliaryTaskSpec(
        task_id=96397 + index,
        task_name=(
            f"mft-5t-g1p6-s{2707275700 + index}-n1-"
            f"{(5, 6, 7, 8)[index % 4]}"
        ),
        role="axis-v6 strict 5T/1.6 NSGA-II",
        cpus=8,
        memory_mb=65_536,
        timeout_seconds=7_200,
        max_workers_per_node=8,
        seed=2707275700 + index,
        primary_turns=(5, 6, 7, 8)[index % 4],
    )
    for index in range(16)
)
REFERENCE_BASELINE_TASK_SPEC = AuxiliaryTaskSpec(
    task_id=96396,
    task_name="mft-goal-reference-drawing260706-standard-symmetric-v1",
    role="reference drawing baseline symmetric/unrounded FEA",
    cpus=8,
    memory_mb=98_304,
    timeout_seconds=43_200,
    max_workers_per_node=1,
)
REFERENCE_THERMAL_HEDGE_TASK_SPEC = AuxiliaryTaskSpec(
    task_id=96414,
    task_name="mft-goal-reference-drawing260706-standard-symmetric-thermalfix-v3",
    role="superseded reference baseline thermal-fix hedge",
    cpus=8,
    memory_mb=98_304,
    timeout_seconds=43_200,
    max_workers_per_node=1,
)
REFERENCE_DIRECT_TASK_SPEC = AuxiliaryTaskSpec(
    task_id=96415,
    task_name="mft-goal-reference-drawing260706-standard-symmetric-direct-v4",
    role="active reference baseline exact direct-analyze replacement",
    cpus=8,
    memory_mb=98_304,
    timeout_seconds=43_200,
    max_workers_per_node=1,
)
TARGET_AXIS_TASK_SPECS = tuple(
    AuxiliaryTaskSpec(
        task_id=96416 + index,
        task_name=(
            f"mft-5t-lm2-s{2707276100 + index}-n1-"
            f"{(5, 6, 7, 8)[index % 4]}"
        ),
        role="authoritative W1200/L1000 fixed-Lm2mH targeted NSGA-II",
        cpus=8,
        memory_mb=65_536,
        timeout_seconds=7_200,
        max_workers_per_node=8,
        seed=2707276100 + index,
        primary_turns=(5, 6, 7, 8)[index % 4],
    )
    for index in range(16)
)
FRESH_SPLITTEMP_TASK_RANGES = ((96_485, 96_740), (96_756, 97_011))
FRESH_SPLITTEMP_TASK_SPECS = tuple(
    AuxiliaryTaskSpec(
        task_id=(
            96_485 + index
            if index < 256
            else 96_756 + index - 256
        ),
        task_name=(
            f"mft-5t-lm2-s{2_707_277_000 + index}-n1-"
            f"{(5, 6, 7, 8)[index % 4]}"
        ),
        role=(
            "authoritative W1200/L1000 fixed-Lm2mH split-temperature "
            "NSGA-II"
        ),
        cpus=8,
        memory_mb=65_536,
        timeout_seconds=7_200,
        max_workers_per_node=8,
        seed=2_707_277_000 + index,
        primary_turns=(5, 6, 7, 8)[index % 4],
    )
    for index in range(512)
)
FINAL_SYMMETRIC_RETRY_TASK_SPEC = AuxiliaryTaskSpec(
    task_id=96_743,
    task_name="mft-final-sym-gap-2b2138a99445-g00860423-r1",
    role=(
        "final standard/unrounded symmetric tuned-gap FEA verification retry"
    ),
    cpus=8,
    memory_mb=65_536,
    timeout_seconds=14_400,
    max_workers_per_node=1,
)
SUPERSEDED_WARM_TASK_SPEC = AuxiliaryTaskSpec(
    task_id=96395,
    task_name="mft-5t-g1p6-s2707275600-n1-5",
    role="superseded axis-v5 warm-start canary",
    cpus=8,
    memory_mb=65_536,
    timeout_seconds=7_200,
    max_workers_per_node=8,
    seed=2707275600,
    primary_turns=5,
)
LEGACY_AUXILIARY_TASK_SPECS = (
    SUPERSEDED_WARM_TASK_SPEC,
    REFERENCE_BASELINE_TASK_SPEC,
    *AXIS_V6_TASK_SPECS,
    REFERENCE_THERMAL_HEDGE_TASK_SPEC,
    REFERENCE_DIRECT_TASK_SPEC,
    *TARGET_AXIS_TASK_SPECS,
)
AUTHORITATIVE_AUXILIARY_TASK_SPECS = (
    *LEGACY_AUXILIARY_TASK_SPECS,
    *FRESH_SPLITTEMP_TASK_SPECS,
    FINAL_SYMMETRIC_RETRY_TASK_SPEC,
)

LEGACY_STANDARD_SELECTION_TASK_IDS = (
    96325,
    96327,
    96328,
    96330,
    96331,
    96332,
    96333,
)
STANDARD_SELECTION_TASK_IDS = (
    96325,
    96327,
    96328,
    96333,
    96330,
    96331,
    96338,
)
STANDARD_SELECTION_LIFECYCLE_TASK_IDS = (
    *LEGACY_STANDARD_SELECTION_TASK_IDS,
    96337,
    96338,
)
STANDARD_SELECTION_OVERLAY_TASK_IDS = (96337, 96338)
STANDARD_SELECTION_SUPERSEDED_TASK_IDS = (96332, 96337)
FULL_REFERENCE_TASK_ID = 96326
SELECTION_POLICY_CARD_ID = "codex-symmetric-primary-selection-policy"
LOCAL_SYMMETRIC_SELECTION_CARD_ID = "codex-local-symmetric-selection"
LEGACY_CONTINUATION_CARD_ID = "codex-standard-full-continuation"
FINAL_DRAWING_CARD_ID = "codex-final-drawing-readiness"
PRIMARY_5T_RECOVERY_CARD_ID = "codex-primary-5t-constraint-recovery"
AXIS_V6_CARD_ID = "codex-axis-v6-fixed-5t-nsga"
TARGET_AXIS_CARD_ID = "codex-target-axis-1200x1000-nsga"
REFERENCE_BASELINE_CARD_ID = "codex-reference-drawing-baseline"
HISTORICAL_AXIS_RAW_TERMINAL = 5_120
HISTORICAL_AXIS_UNIQUE_GEOMETRY = 4_683
NEW_AXIS_GEOMETRY_PASS_RAW = 217
NEW_AXIS_GEOMETRY_PASS_UNIQUE = 210
NEW_AXIS_THERMAL_FEASIBLE = 0
NEW_AXIS_PRODUCTION_PARETO = 0
NEW_AXIS_COMPACT_SURROGATE_MIN_WINDING_C = 302.67
TARGET_AXIS_CAMPAIGN = (
    "mft-goal-fixed-primary-5t-lm2mh-axis-w1200-l1000-"
    "old16-plus-splittemp512-global-v3"
)
TARGET_AXIS_HARD_SHA256 = (
    "486418c731af63007915c9dd2034df1c65df80a5543546915aa25334c89d164f"
)
TARGET_AXIS_LEGACY_HARD_SHA256 = (
    "227cdc0db3b8dae490275d549e8aea93b295a75ee97ea98cdaa601bb96591298"
)
TARGET_AXIS_SUBMISSION_MANIFEST = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\fixed_lm2mh_targeted_w1200_l1000_v1\submission_manifest.json"
)
TARGET_AXIS_SUBMISSION_MANIFESTS = (
    TARGET_AXIS_SUBMISSION_MANIFEST,
    *(
        Path(
            r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
        )
        / name
        / "submission_manifest.json"
        for name in (
            "fixed_lm2mh_splittemp_v2_fresh64",
            "fixed_lm2mh_splittemp_v2_seeds0064_0127",
            "fixed_lm2mh_splittemp_v2_seeds0128_0191",
            "fixed_lm2mh_splittemp_v2_seeds0192_0255",
            "fixed_lm2mh_splittemp_v3_seeds0256_0319",
            "fixed_lm2mh_splittemp_v3_seeds0320_0383",
            "fixed_lm2mh_splittemp_v3_seeds0384_0447",
            "fixed_lm2mh_splittemp_v3_seeds0448_0511",
        )
    ),
)
TARGET_AXIS_EXPECTED_SEED_COUNT = 528
TARGET_AXIS_EXPECTED_FRESH_SEED_COUNT = 512
TARGET_AXIS_EXPECTED_RAW_ROWS = 168_960
TARGET_AXIS_PRIMARY_TEMPERATURE_LIMIT_C = 100.0
TARGET_AXIS_SECONDARY_TEMPERATURE_LIMIT_C = 120.0
TARGET_AXIS_CORE_TEMPERATURE_LIMIT_C = 120.0
ROUNDED_FINAL_TASK_ID = 96340
ROUNDED_TIMEOUT_HEDGE_TASK_ID = 96342
ROUNDED_FINAL_PIPELINE_CARD_ID = "codex-rounded-final-delivery-pipeline"
REFERENCE_PRIMARY_CW1_MM = 5.0
REFERENCE_PRIMARY_GAP1_MM = 1.6
SUPERSEDED_CANDIDATE_CW1_MM = 1.13
SUPERSEDED_CANDIDATE_GAP1_MM = 4.6
PRIMARY_TURN_COUNT = 6
ROUNDED_SNAPSHOT_SHA256 = (
    "c71a94a8b23a9cf8fd4ab9a98083df45f350cf7586b2f1e449deca30380cd426"
)
ROUNDED_SNAPSHOT_SIZE_BYTES = 33_204_563
ROUNDED_SNAPSHOT_MANIFEST_SHA256 = (
    "09bb2b714843ff7bff25ec1c6ae73849f307beab8dab01d95704cfde91fdeb82"
)
ROUNDED_TASK96341_CANCELLATION_SHA256 = (
    "e3f7cd6327444cb957aacc66cb707213fd021c86b7708f9dec68dce31ea553f9"
)
ROUNDED_DRAWING_VIEWS_MANIFEST_SHA256 = (
    "80a53b2e8ad1f3e642585b37a05a7806ea4e412f33110f2aaf935402e613108e"
)
ROUNDED_DRAWING_SUPPLEMENTAL_MANIFEST_SHA256 = (
    "279a1aaf7fb1eed66f3897fc33b031491bc87ead4ae5b548379c8c16aeb636a1"
)
ROUNDED_DRAWING_VIEW_COUNT = 7
ROUNDED_DRAWING_READINESS_MANIFEST_SHA256 = (
    "7e2e37a701dcce423eecf6fb4fd86835448250772d3ba462823901875345aee7"
)
ROUNDED_DRAWING_QA_SHA256 = (
    "229133ffd88aa7a72eecc0561f2246775b7da60a83b2bebe454e5444dd02e528"
)
ROUNDED_DRAWING_QA_PASSED = 20
ROUNDED_DRAWING_DRAFT_PPTX_SHA256 = (
    "28ddf2ac301f0e0f66a428790766a6340165fdd9b70f98e7703babd6aa801287"
)
ROUNDED_DRAWING_DRAFT_PDF_SHA256 = (
    "1710a2da57392f971a0984cd98187197c4db3008369c01d46805906cd0943b85"
)
ROUNDED_DRAWING_DRAFT_PPTX_BYTES = 2_931_000
ROUNDED_DRAWING_DRAFT_PDF_BYTES = 1_988_478
ROUNDED_DRAWING_DRAFT_SLIDES = 9
ROUNDED_FULL_PREPARE_COMMIT = "b4ab0dc"
ROUNDED_PACKAGE_GATE_COMMIT = "3b43d95"
ROUNDED_BOUNDED_CORRECTION_COMMIT = "441a29d"
ROUNDED_TIMEOUT_HEDGE_CANDIDATE_SHA256 = (
    "909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42"
)
ROUNDED_TIMEOUT_HEDGE_PHYSICS_SHA256 = (
    "14c4cce44e0184e0a365d561fb6166a1a30da91c6796b6363cbed51cdbb1e9a0"
)
ROUNDED_TIMEOUT_HEDGE_POST_AT_KST = "2026-07-27 02:17:03 KST"
DRAWING_REFERENCE_PDF_SHA256 = (
    "574d9aab033529cf3655d63542e27871c2e240b669b67e54dbfd2495a564437f"
)
DRAWING_REFERENCE_PPTX_SHA256 = (
    "b8069cc99cf1af3c8e5c5cbe6bd4a2896730620f570c29555ff492713d4af33d"
)

RUNNING_STATES = {"running"}
QUEUED_STATES = {
    "queued",
    "pending",
    "attaching",
    "attached",
    "assigned",
    "launching",
    "starting",
}
SUCCESS_STATES = {"succeeded", "completed", "success"}
FAILURE_STATES = {
    "failed",
    "cancelled",
    "canceled",
    "timed_out",
    "timeout",
}

TaskReader = Callable[[str, int], Mapping[str, Any]]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "payload_sha256" in result:
        raise UpdaterError("payload is already sealed")
    result["payload_sha256"] = canonical_sha256(result)
    return result


def validate_seal(value: Any, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UpdaterError("sealed value must be an object")
    unsigned = copy.deepcopy(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema:
        raise UpdaterError(f"{schema} schema mismatch")
    if observed != canonical_sha256(unsigned):
        raise UpdaterError(f"{schema} payload seal mismatch")
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(
    path: Path,
    value: Any,
    *,
    expected_sha256: str | None = None,
) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        if expected_sha256 is not None:
            if not target.is_file() or _file_sha256(target) != expected_sha256:
                raise ConcurrentStatusUpdate(
                    "status changed while the Scheduler snapshot was merged"
                )
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


class SingleInstanceLock:
    """Process-lifetime advisory lock released automatically after a crash."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.stream: Any | None = None

    def __enter__(self) -> "SingleInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.stream.close()
            self.stream = None
            raise UpdaterError(f"another updater holds the lock: {self.path}") from exc
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        if self.stream is None:
            return
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None


def _get_scheduler_task(
    scheduler_url: str,
    task_id: int,
    *,
    timeout_seconds: float = 10.0,
) -> Mapping[str, Any]:
    url = f"{scheduler_url.rstrip('/')}/api/tasks/{task_id}"
    request = urllib.request.Request(
        url, headers={"Accept": "application/json"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_RESPONSE_BYTES:
                raise UpdaterError("Scheduler GET response exceeds size bound")
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise UpdaterError(f"Scheduler GET failed for task{task_id}") from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise UpdaterError("Scheduler GET response exceeds size bound")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError("Scheduler GET returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise UpdaterError("Scheduler task response must be an object")
    return value


def _task_state(task: Mapping[str, Any]) -> str:
    for key in ("state", "status", "queue_state"):
        value = str(task.get(key) or "").strip().lower()
        if value:
            if value in RUNNING_STATES | QUEUED_STATES:
                return value
            if value in SUCCESS_STATES | FAILURE_STATES:
                return value
    raise UpdaterError("Scheduler task has an unsupported lifecycle state")


def _positive_or_none(value: Any, label: str) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise UpdaterError(f"{label} has invalid type")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise UpdaterError(f"{label} is not an integer") from exc
    if result <= 0:
        raise UpdaterError(f"{label} must be positive")
    return result


def _validate_task(spec: TaskSpec, task: Mapping[str, Any]) -> dict[str, Any]:
    identifiers = {
        int(value)
        for key in ("id", "task_id")
        if (value := task.get(key)) not in (None, "")
    }
    if identifiers != {spec.task_id}:
        raise UpdaterError(f"task{spec.task_id} identity drifted")
    if task.get("name") != spec.task_name:
        raise UpdaterError(f"task{spec.task_id} name drifted")
    expected = {
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
        "timeout_seconds": spec.timeout_seconds,
        "same_node_as_task_id": spec.expected_same_node_as_task_id,
    }
    if spec.max_workers_per_node is not None:
        expected["max_workers_per_node"] = spec.max_workers_per_node
    for key, value in expected.items():
        if task.get(key) != value:
            raise UpdaterError(f"task{spec.task_id} {key} drifted")
    requested_node = task.get("requested_node_name") or task.get("node_name")
    if requested_node != spec.requested_node:
        raise UpdaterError(f"task{spec.task_id} requested node drifted")
    if task.get("node_name_policy") != "strict":
        raise UpdaterError(f"task{spec.task_id} is not strict-node")
    if spec.requested_account is not None:
        accounts = {
            str(task.get(key) or "")
            for key in ("requested_account_name", "account_name")
        }
        if spec.requested_account not in accounts:
            raise UpdaterError(f"task{spec.task_id} account drifted")
    if spec.new_allocation_required and task.get("requested_allocation_id") not in (
        None,
        "",
        0,
    ):
        raise UpdaterError(
            f"task{spec.task_id} new-allocation requirement drifted"
        )
    state = _task_state(task)
    rounded_superseded_identity_retired = (
        spec.task_id == ROUNDED_FINAL_TASK_ID
        and state in (QUEUED_STATES | FAILURE_STATES)
        and task.get("allocation_id") in (None, "", 0)
        and str(task.get("slurm_job_id") or "") == ""
    )
    if (
        spec.expected_allocation_id is not None
        and not rounded_superseded_identity_retired
        and task.get("allocation_id") != spec.expected_allocation_id
    ):
        raise UpdaterError(f"task{spec.task_id} corrected allocation identity drifted")
    if (
        spec.expected_slurm_job_id is not None
        and not rounded_superseded_identity_retired
        and str(task.get("slurm_job_id") or "") != spec.expected_slurm_job_id
    ):
        raise UpdaterError(f"task{spec.task_id} corrected Slurm job identity drifted")
    actual_node = str(
        task.get("actual_node_name") or task.get("allocation_node_name") or ""
    )
    if state in RUNNING_STATES and (
        actual_node != spec.requested_node
        or task.get("placement_contract_satisfied") is not True
    ):
        raise UpdaterError(f"task{spec.task_id} placement is not satisfied")
    return {
        "task_id": spec.task_id,
        "state": state,
        "allocation_id": _positive_or_none(task.get("allocation_id"), "allocation_id"),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "account_name": str(task.get("account_name") or ""),
        "actual_node_name": actual_node,
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "exit_code": task.get("exit_code"),
        "failure_message": str(task.get("failure_message") or "")[:350],
    }


def fetch_tasks(
    scheduler_url: str,
    *,
    task_reader: TaskReader | None = None,
) -> dict[int, dict[str, Any]]:
    reader = task_reader or _get_scheduler_task
    with ThreadPoolExecutor(max_workers=len(TASK_SPECS)) as executor:
        futures = {
            spec.task_id: executor.submit(reader, scheduler_url, spec.task_id)
            for spec in TASK_SPECS
        }
        raw = {task_id: future.result() for task_id, future in futures.items()}
    return {
        spec.task_id: _validate_task(spec, raw[spec.task_id]) for spec in TASK_SPECS
    }


def _validate_auxiliary_task(
    spec: AuxiliaryTaskSpec,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    identifiers = {
        int(value)
        for key in ("id", "task_id")
        if (value := task.get(key)) not in (None, "")
    }
    if identifiers != {spec.task_id}:
        raise UpdaterError(f"auxiliary task{spec.task_id} identity drifted")
    if task.get("name") != spec.task_name:
        raise UpdaterError(f"auxiliary task{spec.task_id} name drifted")
    expected = {
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
        "timeout_seconds": spec.timeout_seconds,
        "max_workers_per_node": spec.max_workers_per_node,
    }
    for key, value in expected.items():
        if task.get(key) != value:
            raise UpdaterError(f"auxiliary task{spec.task_id} {key} drifted")
    state = _task_state(task)
    actual_node = str(
        task.get("actual_node_name") or task.get("allocation_node_name") or ""
    )
    if state in RUNNING_STATES and (
        not actual_node or task.get("placement_contract_satisfied") is not True
    ):
        raise UpdaterError(
            f"auxiliary task{spec.task_id} running placement is not satisfied"
        )
    return {
        "task_id": spec.task_id,
        "name": spec.task_name,
        "role": spec.role,
        "state": state,
        "allocation_id": _positive_or_none(task.get("allocation_id"), "allocation_id"),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "account_name": str(task.get("account_name") or ""),
        "actual_node_name": actual_node,
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "exit_code": task.get("exit_code"),
        "failure_message": str(task.get("failure_message") or "")[:350],
        "seed": spec.seed,
        "primary_turns": spec.primary_turns,
        "cpus": spec.cpus,
        "memory_mb": spec.memory_mb,
    }


def fetch_authoritative_auxiliary_tasks(
    scheduler_url: str,
    *,
    task_reader: TaskReader | None = None,
) -> dict[int, dict[str, Any]]:
    reader = task_reader or _get_scheduler_task
    with ThreadPoolExecutor(
        max_workers=min(64, len(AUTHORITATIVE_AUXILIARY_TASK_SPECS))
    ) as executor:
        futures = {
            spec.task_id: executor.submit(reader, scheduler_url, spec.task_id)
            for spec in AUTHORITATIVE_AUXILIARY_TASK_SPECS
        }
        raw = {task_id: future.result() for task_id, future in futures.items()}
    return {
        spec.task_id: _validate_auxiliary_task(spec, raw[spec.task_id])
        for spec in AUTHORITATIVE_AUXILIARY_TASK_SPECS
    }


def _category(state: str) -> str:
    if state in RUNNING_STATES:
        return "running"
    if state in QUEUED_STATES:
        return "queued"
    if state in SUCCESS_STATES:
        return "succeeded"
    if state in FAILURE_STATES:
        return "failed"
    raise UpdaterError(f"unsupported normalized state: {state}")


def _aware_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise UpdaterError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UpdaterError(f"{label} must include a UTC offset")
    return parsed


def _duration_text(seconds: int) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def _force_cancel_risk(
    spec: TaskSpec,
    *,
    category: str,
    observed_at: str,
) -> dict[str, Any] | None:
    fields = (
        spec.allocation_force_cancel_at_kst,
        spec.task_timeout_at_kst,
        spec.force_cancel_lead_seconds,
    )
    if fields == (None, None, None):
        return None
    if any(value is None for value in fields):
        raise UpdaterError(
            f"task{spec.task_id} force-cancel risk contract is incomplete"
        )
    force_at = _aware_timestamp(
        str(spec.allocation_force_cancel_at_kst),
        f"task{spec.task_id} allocation force boundary",
    )
    timeout_at = _aware_timestamp(
        str(spec.task_timeout_at_kst),
        f"task{spec.task_id} timeout boundary",
    )
    observed = _aware_timestamp(observed_at, "observed_at")
    lead_seconds = int((timeout_at - force_at).total_seconds())
    if (
        lead_seconds != spec.force_cancel_lead_seconds
        or lead_seconds <= 0
        or spec.expected_allocation_id is None
    ):
        raise UpdaterError(f"task{spec.task_id} force-cancel risk timing drifted")
    remaining_seconds = int((force_at - observed).total_seconds())
    if category in {"running", "queued"}:
        lifecycle_note = (
            f"{_duration_text(remaining_seconds)} remaining"
            if remaining_seconds > 0
            else "boundary passed; live Scheduler state remains authoritative"
        )
        active = True
    else:
        lifecycle_note = "terminal lifecycle; historical operational risk"
        active = False
    return {
        "allocation_id": spec.expected_allocation_id,
        "force_at": force_at,
        "timeout_at": timeout_at,
        "lead_seconds": lead_seconds,
        "remaining_seconds": remaining_seconds,
        "lifecycle_note": lifecycle_note,
        "active": active,
    }


def _rounded_final_task_card(
    spec: TaskSpec,
    task: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    """Render task96340 without mixing submission failures with FEA truth."""
    category = _category(str(task["state"]))
    node = task["actual_node_name"] or spec.requested_node
    if category == "succeeded":
        stage = "최종 rounded Standard 대칭 FEA solver 완료 · 수집/인증 대기"
        progress = 100
    elif category == "failed":
        stage = "최종 rounded Standard 대칭 FEA 운영 실패 · 과학 판정 없음"
        progress = 100
    elif category == "running":
        stage = "최종 rounded Standard 대칭 FEA ThermalSetup 실행 중"
        progress = 70
    else:
        stage = "최종 rounded Standard 대칭 FEA 대기 중"
        progress = 5
    solver_stage = (
        "ThermalSetup RUNNING / native Analyze dispatched / "
        "terminal result not available"
        if category == "running"
        else f"no active ThermalSetup claim / lifecycle={category}"
    )
    solver_detail = (
        "현재 ThermalSetup이 실행 중입니다. "
        if category == "running"
        else ""
    )

    allocation = task["allocation_id"] or "none"
    job = task["slurm_job_id"] or "none"
    force_risk = _force_cancel_risk(
        spec,
        category=category,
        observed_at=observed_at,
    )
    if category == "succeeded":
        outcome = (
            "Scheduler terminal success만 확인된 상태이며 collector의 artifact "
            "인증 전에는 scientific/production PASS가 아닙니다."
        )
    elif category == "failed":
        outcome = (
            "운영 lifecycle 실패이며 authenticated solver artifact 검토 전에는 "
            "설계 infeasibility 또는 scientific failure로 세지 않습니다."
        )
    else:
        outcome = (
            "아직 인증된 공진·권선 온도·코어 온도 결과와 scientific PASS가 "
            "없습니다."
        )

    evidence = [
        (
            f"superseded rounded lifecycle=task{spec.task_id} "
            f"{str(task['state']).upper()} / allocation{allocation} / "
            f"Slurm{job} / node{node} / solver stage={solver_stage}"
        ),
        (
            "historical role=official#5 rounded Standard symmetric attempt / "
            "current selection/scientific eligibility=false due primary 5T mismatch"
        ),
        (
            "geometry=round_corner true / corner radius=10mm / "
            "corner segments=4 per quarter / thermal symmetry=eighth"
        ),
        (
            "fixed thermal boundary=fan 1.5m/s / TIM k=0.2W/mK / "
            "WCP pad 2mm / core pad 2mm"
        ),
        (
            f"resources=cpus{spec.cpus} / memory{spec.memory_mb}MB / "
            f"scheduler timeout{spec.timeout_seconds}s / "
            f"inner solver budget{spec.inner_solver_seconds}s"
        ),
        (
            "v1 operational submission only=HTTP422 "
            "'requested_allocation_id is not accepted for new tasks' / "
            "task not created / FEA not started / scientific failure count unchanged"
        ),
        (
            "v2 operational submission only=task96339 FAILED pre-solver because "
            "same_node_as task96328 was terminal failed / no AEDT solve / "
            "scientific failure count unchanged"
        ),
        (
            "v3 effective compute=task96340 / v1-v2 are operational history, "
            "not competing scientific candidates or invalid solutions"
        ),
        (
            "parallel baseline=task96338 direct-Analyze / automatic Full off / "
            "no duplicate scientific PASS claim"
        ),
        (
            "task timeout boundary=2026-07-27 03:55:07 KST / "
            "allocation force boundary=2026-07-27 04:07:00 KST / "
            "planned residual=0h11m53s"
        ),
        (
            "collection_authenticated=false / scientific_pass_generated=false / "
            "production_claim_generated=false / canonical_promotion=false"
        ),
        f"v3 submission receipt SHA256 {spec.submission_receipt_sha256}",
    ]
    if force_risk is not None:
        evidence.append(
            (
                f"allocation{force_risk['allocation_id']} force boundary "
                f"{force_risk['force_at']:%Y-%m-%d %H:%M:%S KST} / "
                f"task timeout boundary "
                f"{force_risk['timeout_at']:%Y-%m-%d %H:%M:%S KST} / "
                f"planned residual {_duration_text(force_risk['lead_seconds'])}"
            )
        )
    if task["failure_message"]:
        evidence[-1] += f" / task96340 failure_message={task['failure_message']}"

    return {
        "id": spec.card_id,
        "title": (
            f"CODEX | {stage} | task{spec.task_id} "
            f"{str(task['state']).upper()} | {node}"
        ),
        "detail": (
            "공식 NSGA-II candidate #5의 round-corner 최종 형상을 검증하는 "
            f"1/8 Standard 대칭 FEA lane입니다. Scheduler GET lifecycle="
            f"{task['state']}, allocation={allocation}, Slurm job={job}, "
            f"node={node}. {solver_detail}{outcome} "
            "v1/v2 제출 실패는 solver 이전 운영 이력으로 "
            "분리되어 task96340의 과학 상태를 오염시키지 않습니다."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": evidence,
    }


def _rounded_timeout_hedge_task_card(
    spec: TaskSpec,
    task: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    """Render the one operational timeout hedge without duplicating design truth."""
    category = _category(str(task["state"]))
    node = task["actual_node_name"] or spec.requested_node
    allocation = task["allocation_id"] or "none"
    job = task["slurm_job_id"] or "none"
    if category == "queued":
        stage = "TIMEOUT HEDGE QUEUED · NO ALLOCATION YET"
        progress = 5
    elif category == "running":
        stage = "TIMEOUT HEDGE RUNNING"
        progress = 20
    elif category == "succeeded":
        stage = "TIMEOUT HEDGE SOLVER DONE · AUTH PENDING"
        progress = 100
    else:
        stage = "TIMEOUT HEDGE OPERATIONAL TERMINAL"
        progress = 100
    return {
        "id": spec.card_id,
        "title": (
            f"CODEX | ROUNDED B5 {stage} | task{spec.task_id} "
            f"{str(task['state']).upper()} | {node}"
        ),
        "detail": (
            "This is the single bounded operational timeout hedge for task96340: "
            "the exact same rounded B5 candidate and physics, not a new design or "
            "an additional scientific candidate. It was submitted exactly once at "
            f"{ROUNDED_TIMEOUT_HEDGE_POST_AT_KST} to harry261 with strict n107, "
            "max_workers_per_node=1, and a new-allocation requirement. "
            f"Scheduler GET lifecycle={task['state']}, allocation={allocation}, "
            f"Slurm job={job}. Task96340 remains lifecycle-visible but is no "
            "longer authoritative for the corrected design; "
            "neither lifecycle creates a scientific PASS before authenticated "
            "solver collection."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            (
                f"Scheduler GET task{spec.task_id} "
                f"{str(task['state']).upper()} / allocation{allocation} / "
                f"Slurm{job} / requested node n107 strict"
            ),
            (
                "source task96340 lifecycle retained=true / corrected rounded "
                "validation lane=false / corrected verification uses "
                "standard/unrounded symmetric"
            ),
            (
                "same rounded B5 candidate=true / candidate physics SHA256 "
                f"{ROUNDED_TIMEOUT_HEDGE_CANDIDATE_SHA256}"
            ),
            (
                "same physics contract as task96340=true / SHA256 "
                f"{ROUNDED_TIMEOUT_HEDGE_PHYSICS_SHA256}"
            ),
            (
                f"single Scheduler POST=1 / HTTP201 / submitted at "
                f"{ROUNDED_TIMEOUT_HEDGE_POST_AT_KST} / repeat POST=false"
            ),
            (
                f"submission receipt SHA256 {spec.submission_receipt_sha256} / "
                f"final seal SHA256 {spec.final_seal_sha256}"
            ),
            (
                "account=harry261 / requested node=n107 / node policy=strict / "
                "max_workers_per_node=1 / new allocation required=true / "
                "existing allocation attach=false"
            ),
            (
                f"resources=cpus{spec.cpus} / memory{spec.memory_mb}MB / "
                f"scheduler timeout{spec.timeout_seconds}s / "
                f"inner solver budget{spec.inner_solver_seconds}s"
            ),
            (
                "authenticated GET-only collector=active / Scheduler methods=GET / "
                "collector POST calls=0"
            ),
            (
                "classification=operational timeout hedge / new design=false / "
                "scientific candidate count unchanged=true"
            ),
            (
                "actual scientific PASS=0 / collection_authenticated=false / "
                "production claim=false / canonical promotion=false"
            ),
            (
                "fixed boundary unchanged=round_corner R10/S4 / fan1.5m/s / "
                "TIM k0.2W/mK / WCP pad2mm / core pad2mm"
            ),
        ],
    }


def _task_card(
    spec: TaskSpec,
    task: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    if spec.task_id == ROUNDED_FINAL_TASK_ID:
        return _supersede_rounded_candidate_card(
            _rounded_final_task_card(spec, task, observed_at)
        )
    if spec.task_id == ROUNDED_TIMEOUT_HEDGE_TASK_ID:
        return _supersede_rounded_candidate_card(
            _rounded_timeout_hedge_task_card(spec, task, observed_at)
        )
    category = _category(str(task["state"]))
    node = task["actual_node_name"] or spec.requested_node
    if category == "succeeded":
        stage = "TERMINAL SUCCEEDED · COLLECTION/PASS PENDING"
        progress = 100
    elif category == "failed":
        stage = f"TERMINAL {str(task['state']).upper()}"
        progress = 100
    elif category == "running":
        stage = "RUNNING"
        progress = 5
    else:
        stage = "QUEUED"
        progress = 0
    force_risk = _force_cancel_risk(
        spec,
        category=category,
        observed_at=observed_at,
    )
    risk_title = ""
    if force_risk is not None and force_risk["active"]:
        risk_title = f" · FORCE-CANCEL RISK {force_risk['force_at']:%m-%d %H:%M:%S KST}"
    title = (
        f"POST-DEADLINE {spec.model_label} · task{spec.task_id} {stage} · "
        f"{node}{risk_title}"
    )
    allocation = task["allocation_id"] or "none"
    job = task["slurm_job_id"] or "none"
    lifecycle = (
        f"Scheduler GET lifecycle={task['state']}, allocation={allocation}, "
        f"Slurm job={job}, node={node}."
    )
    if category == "succeeded":
        outcome = (
            "실행 lifecycle만 terminal success입니다. 별도 collector와 "
            "artifact 인증 전에는 collection·scientific PASS가 아닙니다."
        )
    elif category == "failed":
        reason = (
            spec.terminal_failure_override
            or task["failure_message"]
            or "Scheduler failure reason unavailable"
        )
        outcome = (
            f"Terminal failure reason: {reason}. 운영 실패는 과학적 "
            "infeasibility 또는 production 판정이 아닙니다."
        )
    else:
        outcome = "아직 solver·temperature·scientific PASS 결과가 없습니다."
    detail = (
        f"{spec.candidate_label} post-deadline diagnostic/noncanonical 작업. "
        f"{lifecycle} {outcome}"
    )
    if spec.selection_superseded_by_task_id is not None:
        detail += (
            " Effective selection lane=false: "
            f"task{spec.selection_superseded_by_task_id} supersedes this queued "
            "attempt for candidate selection; this task remains visible for "
            "authenticated Scheduler lifecycle only."
        )
    if spec.selection_failover_for_task_id is not None:
        detail += (
            " Effective selection lane=true: this strict-node failover "
            f"supersedes task{spec.selection_failover_for_task_id} for "
            "candidate selection; the superseded task remains lifecycle-visible."
        )
    if force_risk is not None:
        detail += (
            f" allocation{force_risk['allocation_id']}의 source-derived "
            f"force-cancel 경계는 "
            f"{force_risk['force_at']:%Y-%m-%d %H:%M:%S KST}이며 "
            f"task timeout "
            f"{force_risk['timeout_at']:%Y-%m-%d %H:%M:%S KST}보다 "
            f"{_duration_text(force_risk['lead_seconds'])} 빠릅니다. "
            f"{force_risk['lifecycle_note']}; 이는 operational risk이며 "
            "scientific infeasibility 판정이 아닙니다."
        )
    evidence = [
        (
            f"Scheduler GET task{spec.task_id} {str(task['state']).upper()} / "
            f"allocation{allocation} / Slurm{job} / node{node}"
        ),
        (
            f"cpus{spec.cpus} / memory{spec.memory_mb}MB / "
            f"scheduler timeout{spec.timeout_seconds}s"
        ),
        (
            "strict node placement contract / same_node_as_task_id="
            f"{spec.expected_same_node_as_task_id}"
        ),
        (
            "postdeadline=true / diagnostic_only=true / noncanonical=true"
            + (" / search_only=true" if spec.search_only else "")
        ),
        (
            "scheduler_lifecycle_only=true / collection_authenticated=false / "
            "scientific_pass_generated=false / production_claim_generated=false"
        ),
    ]
    if spec.inner_solver_seconds is not None:
        evidence.insert(2, f"inner solver budget{spec.inner_solver_seconds}s")
    if spec.submission_receipt_sha256 is not None:
        evidence.append(f"submission receipt SHA256 {spec.submission_receipt_sha256}")
    if spec.final_seal_sha256 is not None:
        evidence.append(f"final seal SHA256 {spec.final_seal_sha256}")
    if spec.requested_account is not None:
        evidence.append(
            f"requested account={spec.requested_account} / "
            f"requested node={spec.requested_node} / node policy=strict"
        )
    if spec.selection_superseded_by_task_id is not None:
        evidence.append(
            "selection_lane_effective=false / "
            f"superseded_by_task{spec.selection_superseded_by_task_id}=true / "
            "lifecycle_visibility_preserved=true"
        )
    if spec.selection_failover_for_task_id is not None:
        evidence.append(
            "selection_lane_effective=true / "
            f"failover_for_task{spec.selection_failover_for_task_id}=true / "
            "unique_effective_lane=true"
        )
    if force_risk is not None:
        evidence.extend(
            [
                (
                    f"allocation{force_risk['allocation_id']} source-derived "
                    f"force-cancel boundary "
                    f"{force_risk['force_at']:%Y-%m-%d %H:%M:%S KST}"
                ),
                (
                    f"task timeout boundary "
                    f"{force_risk['timeout_at']:%Y-%m-%d %H:%M:%S KST} / "
                    f"force boundary leads by "
                    f"{_duration_text(force_risk['lead_seconds'])} / "
                    "hard residual guarantee=false / operational risk only / "
                    "scientific infeasibility=false"
                ),
            ]
        )
    if task["failure_message"]:
        evidence.append(f"failure_message={task['failure_message']}")
    if category == "failed" and spec.terminal_failure_override:
        evidence.append(
            f"authenticated terminal root cause={spec.terminal_failure_override}"
        )
    return {
        "id": spec.card_id,
        "title": title,
        "detail": detail,
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": evidence,
    }


def _single_current(payload: Mapping[str, Any], item_id: str) -> dict[str, Any]:
    current = payload.get("current")
    if not isinstance(current, list):
        raise UpdaterError("status current group is missing")
    matches = [item for item in current if item.get("id") == item_id]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise UpdaterError(f"status item {item_id} is not unique")
    return matches[0]


def _read_sealed_local_json(
    path: Path,
    *,
    schema: str,
    schema_field: str = "schema_version",
    canonical_ensure_ascii: bool = False,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if (
        resolved.is_symlink()
        or resolved.stat().st_size > MAX_LOCAL_SEALED_STATE_BYTES
    ):
        raise UpdaterError(f"automation state is unsafe: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError(f"automation state is invalid: {resolved}") from exc
    if not isinstance(value, dict):
        raise UpdaterError(f"automation state is not an object: {resolved}")
    unsigned = copy.deepcopy(value)
    observed = unsigned.pop("payload_sha256", None)
    expected = (
        hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if canonical_ensure_ascii
        else canonical_sha256(unsigned)
    )
    if value.get(schema_field) != schema or observed != expected:
        raise UpdaterError(f"automation state seal drifted: {resolved}")
    return value


def _target_axis_collector_state(
    path: Path | None,
) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = _read_sealed_local_json(
        path,
        schema=TARGET_AXIS_COLLECTOR_STATE_SCHEMA,
    )
    if (
        value.get("campaign_id") != TARGET_AXIS_CAMPAIGN
        or value.get("aggregate_hard_spec_sha256")
        != TARGET_AXIS_HARD_SHA256
        or value.get("expected_seed_count")
        != TARGET_AXIS_EXPECTED_SEED_COUNT
        or value.get("expected_legacy_seed_count") != 16
        or value.get("expected_fresh_seed_count")
        != TARGET_AXIS_EXPECTED_FRESH_SEED_COUNT
        or value.get("expected_raw_terminal_row_count")
        != TARGET_AXIS_EXPECTED_RAW_ROWS
        or value.get("classification") != "screening-only"
        or value.get("production_eligible") is not False
    ):
        raise UpdaterError("target-axis collector identity drifted")
    if value.get("global_nds_final") is True and (
        value.get("successful_terminal_seed_count")
        != TARGET_AXIS_EXPECTED_SEED_COUNT
        or value.get("raw_terminal_row_count")
        != TARGET_AXIS_EXPECTED_RAW_ROWS
        or value.get("final_files_written") is not True
        or (value.get("status_counts") or {}).get("completed")
        != TARGET_AXIS_EXPECTED_SEED_COUNT
    ):
        raise UpdaterError("target-axis final collector coverage drifted")
    if value.get("global_nds_final") is True:
        manifest_path = path.resolve().parent / "global_pareto_manifest.json"
        manifest = _read_sealed_local_json(
            manifest_path,
            schema=TARGET_AXIS_PARETO_MANIFEST_SCHEMA,
        )
        if (
            value.get("pareto_manifest_payload_sha256")
            != manifest.get("payload_sha256")
            or manifest.get("campaign_id") != TARGET_AXIS_CAMPAIGN
            or manifest.get("aggregate_hard_spec_sha256")
            != TARGET_AXIS_HARD_SHA256
            or manifest.get("source_seed_count")
            != TARGET_AXIS_EXPECTED_SEED_COUNT
            or manifest.get("source_raw_terminal_row_count")
            != TARGET_AXIS_EXPECTED_RAW_ROWS
            or manifest.get("global_non_dominated_sorting_complete")
            is not True
            or manifest.get("screening_only") is not True
            or manifest.get("production_eligible") is not False
            or manifest.get("geometry_deduplicated_candidate_count")
            != value.get("geometry_deduplicated_candidate_count")
            or manifest.get("global_screening_feasible_count")
            != value.get("global_screening_feasible_count")
            or manifest.get("global_pareto_count")
            != value.get("partial_screening_pareto_count")
            or manifest.get("conditional_nonthermal_pareto_count")
            != value.get("partial_conditional_nonthermal_pareto_count")
            or manifest.get("minimum_violation_objective_front_count")
            != value.get(
                "partial_minimum_violation_objective_front_count"
            )
            or manifest.get("fea_acquisition_candidate_count")
            != value.get("fea_acquisition_candidate_count")
        ):
            raise UpdaterError("target-axis final Pareto manifest drifted")
    return value


def _upsert_current_card(payload: dict[str, Any], card: Mapping[str, Any]) -> None:
    current = payload.get("current")
    if not isinstance(current, list):
        raise UpdaterError("status current group is missing")
    card_id = card.get("id")
    matches = [
        index
        for index, item in enumerate(current)
        if isinstance(item, dict) and item.get("id") == card_id
    ]
    if len(matches) > 1:
        raise UpdaterError(f"automation card {card_id} is duplicated")
    if matches:
        current[matches[0]] = copy.deepcopy(dict(card))
        return
    insertion = next(
        (
            index
            for index, item in enumerate(current)
            if isinstance(item, dict) and item.get("id") == "parallel-workstreams"
        ),
        len(current),
    )
    current.insert(insertion, copy.deepcopy(dict(card)))


def _upsert_priority_current_card(
    payload: dict[str, Any],
    card: Mapping[str, Any],
) -> None:
    """Upsert a truth-boundary card as the first visible current item."""
    current = payload.get("current")
    if not isinstance(current, list):
        raise UpdaterError("status current group is missing")
    card_id = card.get("id")
    matches = [
        index
        for index, item in enumerate(current)
        if isinstance(item, dict) and item.get("id") == card_id
    ]
    if len(matches) > 1:
        raise UpdaterError(f"automation card {card_id} is duplicated")
    if matches:
        current.pop(matches[0])
    current.insert(0, copy.deepcopy(dict(card)))


def _remove_current_card(payload: dict[str, Any], card_id: str) -> None:
    current = payload.get("current")
    if not isinstance(current, list):
        raise UpdaterError("status current group is missing")
    matches = [
        index
        for index, item in enumerate(current)
        if isinstance(item, dict) and item.get("id") == card_id
    ]
    if len(matches) > 1:
        raise UpdaterError(f"automation card {card_id} is duplicated")
    if matches:
        current.pop(matches[0])


def _selection_lifecycle_overlay(
    tasks: Mapping[int, Mapping[str, Any]] | None,
) -> tuple[int, int]:
    """Return uncollected-pending and operational-failure overlay counts."""
    if tasks is None:
        return 0, 0
    pending = 0
    failures = 0
    for task_id in STANDARD_SELECTION_OVERLAY_TASK_IDS:
        if task_id not in tasks:
            raise UpdaterError(f"selection lifecycle task{task_id} is missing")
        category = _category(str(tasks[task_id]["state"]))
        if category == "failed":
            failures += 1
        else:
            # The sealed source collector does not yet authenticate overlay lanes.
            pending += 1
    return pending, failures


def _symmetric_primary_policy_card(
    tasks: Mapping[int, Mapping[str, Any]],
    observed_at: str,
) -> dict[str, Any]:
    lane_categories = {
        task_id: _category(str(tasks[task_id]["state"]))
        for task_id in STANDARD_SELECTION_TASK_IDS
    }
    terminal_count = sum(
        category in {"succeeded", "failed"} for category in lane_categories.values()
    )
    full_category = _category(str(tasks[FULL_REFERENCE_TASK_ID]["state"]))
    lane_ids = ",".join(f"task{task_id}" for task_id in STANDARD_SELECTION_TASK_IDS)
    lane_lifecycle = " / ".join(
        f"task{task_id}:{lane_categories[task_id].upper()}"
        for task_id in STANDARD_SELECTION_TASK_IDS
    )
    lifecycle = " / ".join(
        f"task{task_id}:{_category(str(tasks[task_id]['state'])).upper()}"
        for task_id in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
    )
    return {
        "id": SELECTION_POLICY_CARD_ID,
        "title": (
            "DESIGN SELECTION | SYMMETRY/STANDARD PRIMARY | "
            f"TERMINAL {terminal_count}/7 | AUTO FULL OFF"
        ),
        "detail": (
            "현재 NSGA-II search solution의 후보 선택은 인증된 "
            "symmetry/Standard FEA 결과를 우선 기준으로 합니다. 후보별 "
            "Standard-to-Full 자동 연쇄는 중지되어 있습니다. 기존 Full "
            "task96326은 diagnostic reference로만 계속되며 설계 선택이나 "
            "승격 근거가 아닙니다. symmetry 결과로 한 후보를 명시적으로 "
            "선택한 뒤 최대 한 후보만 최종 Full 검증할 수 있습니다. 인증된 "
            "actual 결과 전에는 scientific/production PASS를 주장하지 않습니다."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 50 + (terminal_count * 35 // 7),
        "evidence": [
            "primary candidate-selection gate=authenticated symmetric/Standard FEA",
            f"Standard selection lanes=7 / {lane_ids}",
            f"lane lifecycle={lane_lifecycle}",
            f"selection lifecycle observations=9 / {lifecycle}",
            "automatic Standard-to-Full per candidate=false",
            (
                "effective official#8 selection lane=task96333 / "
                "task96329 superseded but lifecycle-visible"
            ),
            (
                "effective official#5 selection lane=task96338 / "
                "task96332 superseded but lifecycle-visible / "
                "task96337 pre-EM operational failure only"
            ),
            (
                f"task96326 lifecycle={full_category.upper()} / "
                "role=diagnostic reference only"
            ),
            "final explicit Full validation candidate cap=1",
            (
                "policy card is not result evidence / actual scientific PASS=0 / "
                "actual production PASS=0 / canonical promotion=false / "
                "production truth=false"
            ),
        ],
    }


def _postsuccess_card(
    path: Path,
    observed_at: str,
    tasks: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    state = _read_sealed_local_json(
        path,
        schema=POSTSUCCESS_STATE_SCHEMA,
    )
    collection_count = int(state.get("collection_count") or 0)
    pending_count = int(state.get("pending_count") or 0)
    failure_count = int(state.get("terminal_failure_count") or 0)
    authoritative_lifecycle = "lifecycle_task_ids" in state
    if authoritative_lifecycle:
        lifecycle_ids = state.get("lifecycle_task_ids")
        effective_ids = state.get("effective_task_ids")
        superseded_ids = state.get("selection_superseded_task_ids")
        lanes = state.get("lanes")
        lifecycle_collection_count = int(state.get("lifecycle_collection_count") or 0)
        watcher_pending = int(state.get("lifecycle_pending_count") or 0)
        watcher_failures = int(state.get("lifecycle_terminal_failure_count") or 0)
        if (
            not isinstance(lifecycle_ids, list)
            or not isinstance(effective_ids, list)
            or not isinstance(superseded_ids, list)
            or len(lifecycle_ids) != 9
            or len(set(lifecycle_ids)) != 9
            or set(lifecycle_ids) != set(STANDARD_SELECTION_LIFECYCLE_TASK_IDS)
            or len(effective_ids) != 7
            or len(set(effective_ids)) != 7
            or set(effective_ids) != set(STANDARD_SELECTION_TASK_IDS)
            or set(superseded_ids) != set(STANDARD_SELECTION_SUPERSEDED_TASK_IDS)
            or not isinstance(lanes, list)
            or len(lanes) != 9
            or collection_count + pending_count + failure_count != 7
            or lifecycle_collection_count + watcher_pending + watcher_failures != 9
        ):
            raise UpdaterError("post-success lifecycle contract drifted")
        lane_status: dict[int, str] = {}
        for lane in lanes:
            if not isinstance(lane, Mapping):
                raise UpdaterError("post-success lifecycle lane drifted")
            task_id = lane.get("task_id")
            status = str(lane.get("status") or "")
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id not in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
                or task_id in lane_status
                or status not in {"pending", "collection_ready", "terminal_failure"}
            ):
                raise UpdaterError("post-success lifecycle lane drifted")
            lane_status[task_id] = status
        if (
            sum(status == "collection_ready" for status in lane_status.values())
            != lifecycle_collection_count
            or sum(status == "pending" for status in lane_status.values())
            != watcher_pending
            or sum(status == "terminal_failure" for status in lane_status.values())
            != watcher_failures
        ):
            raise UpdaterError("post-success lifecycle lane counts drifted")
        current_failure_ids = {
            task_id
            for task_id, status in lane_status.items()
            if status == "terminal_failure"
        }
        if tasks is not None:
            current_failure_ids.update(
                task_id
                for task_id in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
                if _category(str(tasks[task_id]["state"])) == "failed"
            )
        current_collection_ids = {
            task_id
            for task_id, status in lane_status.items()
            if status == "collection_ready"
        } - current_failure_ids
        lifecycle_failures = len(current_failure_ids)
        lifecycle_pending = 9 - len(current_collection_ids) - lifecycle_failures
        lifecycle_mode = "v6-sealed-effective-plus-live-GET-lifecycle"
        watcher_snapshot = (
            f"watcher snapshot pending={watcher_pending} / "
            f"operational terminal failures={watcher_failures}"
        )
    else:
        overlay_pending, overlay_failures = _selection_lifecycle_overlay(tasks)
        lifecycle_pending = pending_count + overlay_pending
        lifecycle_failures = failure_count + overlay_failures
        lifecycle_mode = "v5-overlay-fallback"
        watcher_snapshot = "watcher snapshot=v5 effective-only"
    lifecycle_authenticated = 9 - lifecycle_pending - lifecycle_failures
    if (
        state.get("diagnostic_only") is not True
        or state.get("production_eligible") is not False
        or state.get("scheduler_mutation_performed") is not False
        or state.get("scientific_pass_claimed") is not False
        or state.get("production_claimed") is not False
    ):
        raise UpdaterError("post-success automation safety boundary drifted")
    card = {
        "id": "codex-standard-postsuccess-pipeline",
        "title": (
            "CODEX AUTO · STANDARD RESULT PIPELINE · "
            f"COLLECTIONS {collection_count} · PENDING {pending_count}"
        ),
        "detail": (
            "Seven symmetry/Standard collectors authenticate terminal artifacts, "
            "apply hard constraints and strict-AL admission, then prepare measured "
            "global NDS inputs. Missing measured results never create a "
            "scientific or production claim."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 70 if collection_count else 40,
        "evidence": [
            f"state={state.get('status')}",
            (
                f"collections{collection_count} / pending{pending_count} / "
                f"terminal failures{failure_count}"
            ),
            (
                "surrogate retraining="
                f"{str(bool(state.get('surrogate_retraining_performed'))).lower()}"
            ),
            (
                "production Pareto emitted="
                f"{str(bool(state.get('production_pareto_emitted'))).lower()}"
            ),
            (
                "Standard selection lanes=7 / "
                + ",".join(f"task{task_id}" for task_id in STANDARD_SELECTION_TASK_IDS)
            ),
            "Scheduler mutation=false / scientific claim=false",
            f"state SHA256 {_file_sha256(path.resolve())}",
        ],
    }
    card["title"] = (
        "CODEX AUTO | STANDARD RESULT PIPELINE | "
        f"AUTH {collection_count} | PENDING {pending_count} | ACTUAL PASS 0"
    )
    card["detail"] = (
        "Nine lifecycle attempts currently feed seven effective "
        "symmetry/Standard selection lanes. Collectors authenticate terminal "
        "artifacts, apply hard constraints and strict-AL admission, then prepare "
        "measured global NDS inputs. Missing measured results never create a "
        "scientific or production claim."
    )
    card["evidence"][1] = (
        f"selection-effective authenticated={collection_count}/7 / "
        f"pending={pending_count} / operational terminal failures={failure_count}"
    )
    card["evidence"].insert(
        2,
        f"lifecycle attempts=9 / authenticated={lifecycle_authenticated}/9 / "
        f"pending={lifecycle_pending} / "
        f"operational terminal failures={lifecycle_failures}",
    )
    card["evidence"].insert(
        -2,
        f"lifecycle mode={lifecycle_mode} / attempts=9 / selection-effective=7 / "
        "task96337 failed pre-EM operationally / task96338 current",
    )
    card["evidence"].insert(-2, watcher_snapshot)
    card["evidence"].insert(-2, "actual scientific PASS=0 / actual production PASS=0")
    return card


def _local_selection_fail_safe_card(
    path: Path,
    observed_at: str,
    *,
    condition: str,
) -> dict[str, Any]:
    label = {
        "missing": "STATE MISSING",
        "unavailable": "STATE UNAVAILABLE",
        "invalid": "STATE INVALID",
    }[condition]
    return {
        "id": LOCAL_SYMMETRIC_SELECTION_CARD_ID,
        "title": (f"CODEX · SYMMETRIC FEA SELECT · {label} · AUTO FULL OFF"),
        "detail": (
            "The optional sealed local-selection state is not trusted yet. "
            "No candidate selection, local-batch, or Full continuation claim "
            "is inferred from missing or invalid bytes."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 20,
        "evidence": [
            f"state={condition}_fail_safe",
            "authenticated=unavailable / pending=unavailable / terminal=unavailable",
            "selected symmetric result=none claimed",
            "local bounded budget=max3/round × 2 rounds / prepare-only",
            "AUTO FULL OFF / Scheduler mutation=false",
            f"optional state path={path.resolve()}",
        ],
    }


def _selection_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UpdaterError(f"local symmetric selection {label} drifted")
    return value


def _selection_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpdaterError(f"local symmetric selection {label} drifted")
    number = float(value)
    if not math.isfinite(number):
        raise UpdaterError(f"local symmetric selection {label} drifted")
    return number


def _v6_local_selection_lifecycle_counts(
    value: Mapping[str, Any],
    *,
    authenticated: int,
    pending: int,
    failures: int,
    tasks: Mapping[int, Mapping[str, Any]] | None,
) -> tuple[int, int, int, int]:
    lifecycle_ids = value.get("lifecycle_task_ids")
    superseded_ids = value.get("selection_superseded_task_ids")
    lanes = value.get("lanes")
    if (
        not isinstance(lifecycle_ids, list)
        or len(lifecycle_ids) != 9
        or len(set(lifecycle_ids)) != 9
        or set(lifecycle_ids) != set(STANDARD_SELECTION_LIFECYCLE_TASK_IDS)
        or not isinstance(superseded_ids, list)
        or set(superseded_ids) != set(STANDARD_SELECTION_SUPERSEDED_TASK_IDS)
        or value.get("effective_lane_count") != 7
        or value.get("lifecycle_lane_count") != 9
        or not isinstance(lanes, list)
        or len(lanes) != 9
    ):
        raise UpdaterError("local symmetric selection v6 lifecycle drifted")
    lifecycle_statuses: list[str] = []
    effective_statuses: list[str] = []
    observed_ids: list[int] = []
    status_by_task: dict[int, str] = {}
    for lane in lanes:
        if not isinstance(lane, Mapping):
            raise UpdaterError("local symmetric selection v6 lane drifted")
        task_id = lane.get("task_id")
        status = str(lane.get("effective_status") or "")
        selection_effective = lane.get("selection_effective")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id not in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
            or status not in {"pending", "collection_ready", "terminal_failure"}
            or not isinstance(selection_effective, bool)
            or selection_effective != (task_id in STANDARD_SELECTION_TASK_IDS)
        ):
            raise UpdaterError("local symmetric selection v6 lane drifted")
        observed_ids.append(task_id)
        status_by_task[task_id] = status
        lifecycle_statuses.append(status)
        if selection_effective:
            effective_statuses.append(status)
    if len(set(observed_ids)) != 9:
        raise UpdaterError("local symmetric selection v6 lanes are duplicated")
    effective_counts = {
        status: effective_statuses.count(status)
        for status in {"pending", "collection_ready", "terminal_failure"}
    }
    if (
        effective_counts["collection_ready"] != authenticated
        or effective_counts["pending"] != pending
        or effective_counts["terminal_failure"] != failures
    ):
        raise UpdaterError("local symmetric selection v6 counts drifted")
    watcher_pending = lifecycle_statuses.count("pending")
    watcher_failures = lifecycle_statuses.count("terminal_failure")
    current_failure_ids = {
        task_id
        for task_id, lane_status in status_by_task.items()
        if lane_status == "terminal_failure"
    }
    if tasks is not None:
        current_failure_ids.update(
            task_id
            for task_id in STANDARD_SELECTION_LIFECYCLE_TASK_IDS
            if _category(str(tasks[task_id]["state"])) == "failed"
        )
    current_collection_ids = {
        task_id
        for task_id, lane_status in status_by_task.items()
        if lane_status == "collection_ready"
    } - current_failure_ids
    return (
        9 - len(current_collection_ids) - len(current_failure_ids),
        len(current_failure_ids),
        watcher_pending,
        watcher_failures,
    )


def _strict_local_symmetric_selection_card(
    path: Path,
    observed_at: str,
    tasks: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        value = _read_sealed_local_json(
            path,
            schema=LOCAL_SYMMETRIC_SELECTION_STATE_SCHEMA,
            canonical_ensure_ascii=True,
        )
    except FileNotFoundError:
        return _local_selection_fail_safe_card(path, observed_at, condition="missing")
    except OSError:
        return _local_selection_fail_safe_card(
            path, observed_at, condition="unavailable"
        )
    except UpdaterError:
        return _local_selection_fail_safe_card(path, observed_at, condition="invalid")

    allowed_statuses = {
        "awaiting_symmetric_results",
        "partial_measured_waiting",
        "selected_symmetric_hard_pass",
        "local_prepare_only_batch_ready",
        "terminal_no_passing_or_small_local_correction",
        "blocked_fail_closed",
    }
    status = str(value.get("status") or "")
    authenticated = _selection_count(
        value.get("authenticated_observation_count"),
        "authenticated observation count",
    )
    pending = _selection_count(value.get("pending_count"), "pending count")
    failures = _selection_count(
        value.get("terminal_failure_count"),
        "terminal failure count",
    )
    task_ids = value.get("expected_task_ids")
    budget = value.get("finite_local_budget")
    task_ids_valid = (
        isinstance(task_ids, list)
        and len(task_ids) == 7
        and all(
            isinstance(task_id, int) and not isinstance(task_id, bool)
            for task_id in task_ids
        )
        and len(set(task_ids)) == 7
    )
    authoritative_layout = task_ids_valid and set(task_ids) == set(
        STANDARD_SELECTION_TASK_IDS
    )
    legacy_layout = (
        task_ids_valid
        and set(task_ids) == set(LEGACY_STANDARD_SELECTION_TASK_IDS)
        and "lifecycle_task_ids" not in value
    )
    if (
        status not in allowed_statuses
        or not (authoritative_layout or legacy_layout)
        or authenticated + pending + failures != 7
        or value.get("diagnostic_only") is not True
        or value.get("production_eligible") is not False
        or value.get("symmetric_model_primary") is not True
        or value.get("prepare_only") is not True
        or value.get("scheduler_methods_used") != []
        or value.get("scheduler_mutation_performed") is not False
        or value.get("scheduler_submission_performed") is not False
        or value.get("scheduler_cancel_performed") is not False
        or value.get("scheduler_restart_performed") is not False
        or value.get("automatic_full_trigger") is not False
        or value.get("automatic_full_continuation") is not False
        or value.get("full_model_started_by_watcher") is not False
        or value.get("terminal_failures_are_physics_observations") is not False
        or value.get("pending_allows_local_candidate_generation") is not False
        or not isinstance(value.get("watch_complete"), bool)
        or not isinstance(budget, Mapping)
    ):
        return _local_selection_fail_safe_card(path, observed_at, condition="invalid")

    if authoritative_layout:
        if status == "blocked_fail_closed" and "lifecycle_task_ids" not in value:
            lifecycle_pending = pending
            lifecycle_failures = failures
            lifecycle_count = 7
            lifecycle_mode = "blocked-fail-closed-effective-only"
            watcher_snapshot = "watcher snapshot=blocked effective-only"
        else:
            try:
                (
                    lifecycle_pending,
                    lifecycle_failures,
                    watcher_pending,
                    watcher_failures,
                ) = _v6_local_selection_lifecycle_counts(
                    value,
                    authenticated=authenticated,
                    pending=pending,
                    failures=failures,
                    tasks=tasks,
                )
            except UpdaterError:
                return _local_selection_fail_safe_card(
                    path, observed_at, condition="invalid"
                )
            lifecycle_count = 9
            lifecycle_mode = "v6-sealed-effective-plus-live-GET-lifecycle"
            watcher_snapshot = (
                f"watcher snapshot pending={watcher_pending} / "
                f"operational terminal failures={watcher_failures}"
            )
    else:
        overlay_pending, overlay_failures = _selection_lifecycle_overlay(tasks)
        lifecycle_pending = pending + overlay_pending
        lifecycle_failures = failures + overlay_failures
        lifecycle_count = 9
        lifecycle_mode = "v5-overlay-fallback"
        watcher_snapshot = "watcher snapshot=v5 effective-only"
    lifecycle_authenticated = (
        lifecycle_count - lifecycle_pending - lifecycle_failures
    )

    current_round = _selection_count(budget.get("current_round"), "current local round")
    max_rounds = _selection_count(budget.get("max_rounds"), "maximum local rounds")
    max_per_round = _selection_count(
        budget.get("max_candidates_per_round"),
        "maximum candidates per round",
    )
    max_total = _selection_count(
        budget.get("max_candidates_total"),
        "maximum total candidates",
    )
    prepared = _selection_count(
        budget.get("candidate_count_this_round"),
        "candidate count this round",
    )
    if (
        current_round not in {0, 1}
        or max_rounds != 2
        or max_per_round != 3
        or max_total != 6
        or prepared > max_per_round
        or (pending and prepared)
    ):
        return _local_selection_fail_safe_card(path, observed_at, condition="invalid")

    selected = value.get("selected_symmetric_result")
    selected_text = "none"
    if status == "selected_symmetric_hard_pass":
        if not isinstance(selected, Mapping):
            return _local_selection_fail_safe_card(
                path, observed_at, condition="invalid"
            )
        task_id = selected.get("task_id")
        candidate_sha = selected.get("candidate_physics_sha256")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id not in set(task_ids)
            or not isinstance(candidate_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", candidate_sha)
        ):
            return _local_selection_fail_safe_card(
                path, observed_at, condition="invalid"
            )
        margin = _selection_number(
            selected.get("minimum_normalized_actual_margin"),
            "selected margin",
        )
        loss = _selection_number(
            selected.get("actual_total_loss_W"),
            "selected loss",
        )
        volume = _selection_number(
            selected.get("actual_volume_L"),
            "selected volume",
        )
        selected_text = (
            f"task{task_id} {candidate_sha[:12]} / margin={margin:.6g} / "
            f"loss={loss:.3f}W / volume={volume:.3f}L"
        )
        label = f"SELECTED task{task_id}"
    elif selected is not None:
        return _local_selection_fail_safe_card(path, observed_at, condition="invalid")
    else:
        label = {
            "awaiting_symmetric_results": "WAITING",
            "partial_measured_waiting": "MEASURED, WAITING",
            "local_prepare_only_batch_ready": (f"LOCAL BATCH {prepared}/3 READY"),
            "terminal_no_passing_or_small_local_correction": ("NO ELIGIBLE LOCAL STEP"),
            "blocked_fail_closed": "FAIL-CLOSED",
        }[status]

    progress = {
        "awaiting_symmetric_results": 35,
        "partial_measured_waiting": min(85, 40 + authenticated * 6),
        "selected_symmetric_hard_pass": 100,
        "local_prepare_only_batch_ready": 90,
        "terminal_no_passing_or_small_local_correction": 100,
        "blocked_fail_closed": 25,
    }[status]
    card = {
        "id": LOCAL_SYMMETRIC_SELECTION_CARD_ID,
        "title": (
            "CODEX · SYMMETRIC FEA SELECT · "
            f"{label} · AUTH {authenticated}/7 · AUTO FULL OFF"
        ),
        "detail": (
            "Authenticated symmetric FEA is the primary selection gate. "
            "A passing result stops selection immediately; otherwise only a "
            "finite prepare-only local batch can be emitted after all current "
            "lanes are terminal."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            f"status={status}",
            (
                f"authenticated={authenticated}/7 / pending={pending} / "
                f"terminal failures={failures}"
            ),
            f"selected symmetric result={selected_text}",
            (
                f"local bounded budget=round {current_round + 1}/"
                f"{max_rounds} / prepared {prepared}/{max_per_round} / "
                f"total cap {max_total} / prepare-only"
            ),
            "AUTO FULL OFF / Scheduler mutation=false / methods=[]",
            f"state SHA256 {_file_sha256(path.resolve())}",
        ],
    }
    card["title"] = (
        f"CODEX | SYMMETRIC FEA SELECT | {label} | AUTH {authenticated}/7 | "
        f"PENDING {pending} | ACTUAL PASS 0 | AUTO FULL OFF"
    )
    card["evidence"][1] = (
        f"selection-effective authenticated={authenticated}/7 / "
        f"pending={pending} / operational terminal failures={failures}"
    )
    card["evidence"].insert(
        2,
        f"lifecycle attempts={lifecycle_count} / "
        f"authenticated={lifecycle_authenticated}/{lifecycle_count} / "
        f"pending={lifecycle_pending} / "
        f"operational terminal failures={lifecycle_failures}",
    )
    card["evidence"].insert(
        3,
        f"lifecycle mode={lifecycle_mode} / observations={lifecycle_count} / "
        "selection-effective=7 / "
        "task96332 superseded by task96338 / "
        "task96337 pre-EM failure is operational only",
    )
    card["evidence"].insert(3, watcher_snapshot)
    card["evidence"].insert(-1, "actual scientific PASS=0 / actual production PASS=0")
    return card


def _local_symmetric_selection_card(
    path: Path,
    observed_at: str,
    tasks: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        return _strict_local_symmetric_selection_card(path, observed_at, tasks)
    except FileNotFoundError:
        condition = "missing"
    except OSError:
        condition = "unavailable"
    except UpdaterError:
        condition = "invalid"
    return _local_selection_fail_safe_card(path, observed_at, condition=condition)


def _thermal_bridge_card(path: Path, observed_at: str) -> dict[str, Any]:
    value = _read_sealed_local_json(
        path,
        schema=THERMAL_BRIDGE_STATE_SCHEMA,
        schema_field="schema",
    )
    state = str(value.get("watcher_state") or "").lower()
    allowed = {"armed", "running", "collected", "failed"}
    if state not in allowed:
        raise UpdaterError("thermal bridge watcher state drifted")
    pid = _positive_or_none(value.get("watcher_pid"), "thermal bridge watcher PID")
    post_count = value.get("scheduler_post_calls_total")
    if isinstance(post_count, bool) or post_count not in {0, 1}:
        raise UpdaterError("thermal bridge POST count drifted")
    tool_sha = str(value.get("tool_sha256") or "")
    source_state = str(value.get("source_task_state") or "")
    stage = str(value.get("stage") or "")
    if (
        value.get("source_task_id") != 96324
        or value.get("source_task_name") != TASK_SPECS[0].task_name
        or value.get("heartbeat_interval_seconds") != 60
        or not re.fullmatch(r"[0-9a-f]{64}", tool_sha)
        or not source_state
        or not stage
        or value.get("diagnostic_only") is not True
        or value.get("canonical") is not False
        or value.get("production_truth_eligible") is not False
        or value.get("scheduler_mutation_performed") is not False
        or value.get("scientific_pass_claimed") is not False
        or value.get("production_claimed") is not False
        or value.get("artifact_collected") is not (state == "collected")
        or (state == "failed") != bool(value.get("failure"))
    ):
        raise UpdaterError("thermal bridge safety boundary drifted")
    updated = str(value.get("updated_at_utc") or "")
    try:
        parsed = datetime.fromisoformat(updated)
    except ValueError as exc:
        raise UpdaterError("thermal bridge heartbeat timestamp drifted") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UpdaterError("thermal bridge heartbeat timestamp lacks timezone")
    progress = {
        "armed": 25,
        "running": 55,
        "collected": 100,
        "failed": 100,
    }[state]
    return {
        "id": "codex-thermal-artifact-handoff",
        "title": (f"CODEX AUTO · THERMAL ARTIFACT HANDOFF · {state.upper()}"),
        "detail": (
            "Corrected-thermal task96324의 terminal-success artifact를 "
            "인증·전송·수집하는 자동화 상태입니다. 이 lifecycle 표시는 "
            "온도 제약이나 scientific/production PASS를 주장하지 않습니다."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            f"source task96324 state={source_state} / stage={stage}",
            (f"Scheduler POST count={post_count} / heartbeat mutation=false"),
            (
                f"artifact collected="
                f"{str(bool(value.get('artifact_collected'))).lower()}"
            ),
            "scientific claim=false / production claim=false",
            f"watcher PID={pid} / tool SHA256 {tool_sha}",
            f"heartbeat updated={updated}",
            f"state SHA256 {_file_sha256(path.resolve())}",
        ],
    }


def _standard_full_continuation_card(
    path: Path,
    observed_at: str,
) -> dict[str, Any]:
    value = _read_sealed_local_json(
        path,
        schema=STANDARD_FULL_CONTINUATION_STATE_SCHEMA,
        schema_field="schema_version",
    )
    status = str(value.get("status") or "")
    allowed = {
        "pending_standard_collections",
        "terminal_no_measured_pass",
        "measured_pass_waiting_for_strict_full_lane",
        "prepared_waiting_for_post_authorization",
        "full_submitted_pending_actual_result",
        "submission_outcome_uncertain_no_repost",
        "fail_closed_retrying_get_gates",
    }
    if status not in allowed:
        raise UpdaterError("Standard-to-Full continuation status drifted")
    attempts = value.get("scheduler_post_attempts_consumed")
    if isinstance(attempts, bool) or attempts not in {0, 1}:
        raise UpdaterError("Standard-to-Full POST counter drifted")
    full_task_id = _positive_or_none(
        value.get("full_task_id"),
        "Standard-to-Full task ID",
    )
    attempt = value.get("attempt_ledger")
    receipt = value.get("submission_receipt")
    plan = value.get("plan")
    if (
        value.get("original_deadline_missed") is not True
        or value.get("postdeadline") is not True
        or value.get("canonical") is not False
        or value.get("production_eligible") is not False
        or value.get("scientific_pass_claimed") is not False
        or value.get("full_result_available") is not False
        or value.get("full_actual_constraints_passed") is not False
        or value.get("promotion_completed") is not False
        or value.get("maximum_scheduler_posts") != 1
        or value.get("scheduler_project_mutation_performed") is not False
        or value.get("scheduler_repository_modified") is not False
        or (attempts == 0 and attempt is not None)
        or (attempts == 1 and not isinstance(attempt, Mapping))
        or (
            status == "full_submitted_pending_actual_result"
            and (
                full_task_id is None
                or not isinstance(plan, Mapping)
                or not isinstance(attempt, Mapping)
                or not isinstance(receipt, Mapping)
            )
        )
        or (status != "full_submitted_pending_actual_result" and receipt is not None)
    ):
        raise UpdaterError("Standard-to-Full safety boundary drifted")
    updated = str(value.get("observed_at_utc") or "")
    try:
        parsed = datetime.fromisoformat(updated)
    except ValueError as exc:
        raise UpdaterError("Standard-to-Full heartbeat timestamp drifted") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UpdaterError("Standard-to-Full heartbeat timestamp lacks timezone")
    progress = {
        "pending_standard_collections": 25,
        "terminal_no_measured_pass": 100,
        "measured_pass_waiting_for_strict_full_lane": 50,
        "prepared_waiting_for_post_authorization": 65,
        "full_submitted_pending_actual_result": 80,
        "submission_outcome_uncertain_no_repost": 100,
        "fail_closed_retrying_get_gates": 35,
    }[status]
    return {
        "id": LEGACY_CONTINUATION_CARD_ID,
        "title": (f"CODEX AUTO · STANDARD→FULL CONTINUATION · {status.upper()}"),
        "detail": (
            "인증된 Standard 실측 hard-feasible rank-0 후보가 생길 때만 "
            "동일 후보의 Full 계산을 별도 strict-node lane에 최대 한 번 "
            "연결합니다. Full terminal result 전에는 scientific PASS나 "
            "promotion을 주장하지 않습니다."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            f"state={status} / Full task={full_task_id or 'none'}",
            f"Scheduler POST attempts consumed={attempts}/1",
            (
                f"plan={str(isinstance(plan, Mapping)).lower()} / "
                f"receipt={str(isinstance(receipt, Mapping)).lower()}"
            ),
            "scientific PASS=false / promotion=false / canonical=false",
            "Scheduler project mutation=false / repository mixing=false",
            f"heartbeat updated={updated}",
            f"state SHA256 {_file_sha256(path.resolve())}",
        ],
    }


def _final_gate_card(root: Path, observed_at: str) -> dict[str, Any]:
    resolved_root = root.resolve()
    seal_path = resolved_root / FINAL_PACKAGE_NAME / "package_seal.json"
    pending_path = resolved_root / "pending_manifest.json"
    if seal_path.is_file():
        value = _read_sealed_local_json(
            seal_path,
            schema=FINAL_GATE_SEAL_SCHEMA,
            schema_field="schema",
        )
        if (
            value.get("all_explicit_goal_constraints_actual_pass") is not True
            or value.get("solver_result_truth_included") is not True
        ):
            raise UpdaterError("published final package truth boundary drifted")
        title = "CODEX AUTO · FINAL AEDT PACKAGE GATE · PUBLISHED"
        detail = (
            "인증된 actual solver 결과와 고정 제약을 모두 통과해 Full 및 "
            "symmetric AEDT 패키지가 봉인됐습니다."
        )
        progress = 100
        evidence = [
            "full AEDT promoted=true / symmetric AEDT promoted=true",
            "actual solver result truth=true",
            "all explicit goal constraints actual pass=true",
            f"package seal SHA256 {_file_sha256(seal_path)}",
        ]
    elif pending_path.is_file():
        value = _read_sealed_local_json(
            pending_path,
            schema=FINAL_GATE_PENDING_SCHEMA,
            schema_field="schema",
        )
        reasons = value.get("pending_reasons")
        if (
            not isinstance(reasons, list)
            or value.get("final_package_created") is not False
            or value.get("full_aedt_promoted") is not False
            or value.get("symmetric_aedt_promoted") is not False
        ):
            raise UpdaterError("pending final package truth boundary drifted")
        title = "CODEX AUTO · FINAL AEDT PACKAGE GATE · PENDING"
        detail = (
            "actual Full/thermal 결과와 solver-produced AEDT가 모두 인증될 "
            "때까지 fail-closed 상태를 유지합니다."
        )
        progress = 60
        evidence = [
            *[f"pending: {reason}" for reason in reasons[:4]],
            "full AEDT promoted=false / symmetric AEDT promoted=false",
            "open-only fallback promotion=false",
            f"pending manifest SHA256 {_file_sha256(pending_path)}",
        ]
    else:
        raise UpdaterError("final package gate state is absent")
    return {
        "id": "codex-final-aedt-package-gate",
        "title": title,
        "detail": detail,
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": evidence,
    }


def _primary_5t_recovery_card(observed_at: str) -> dict[str, Any]:
    superseded_stack = (
        PRIMARY_TURN_COUNT * SUPERSEDED_CANDIDATE_CW1_MM
        + (PRIMARY_TURN_COUNT - 1) * SUPERSEDED_CANDIDATE_GAP1_MM
    )
    reference_stack = (
        PRIMARY_TURN_COUNT * REFERENCE_PRIMARY_CW1_MM
        + (PRIMARY_TURN_COUNT - 1) * REFERENCE_PRIMARY_GAP1_MM
    )
    return {
        "id": PRIMARY_5T_RECOVERY_CARD_ID,
        "title": (
            "권선/축 계약 확정 | W=x≤1200 · L=y≤1000 · H≤750 | "
            "5T/1.6 | 회전·축교환 금지"
        ),
        "detail": (
            "현재 도면 축을 그대로 고정합니다: W는 도면 x/기존 973 mm "
            "측·2차측 방향으로 1200 mm 이하, L은 도면 y/수직 방향으로 "
            "1000 mm 이하, H는 750 mm 이하이며 회전이나 축 교환은 허용하지 "
            "않습니다. 1차 권선은 5.0 mm/간격 1.6 mm로 고정됩니다. "
            "기존 axis-v6 W1000/L1200 결과는 잘못된 축 계약의 과거 자료이고, "
            "W1200/L1000 타깃 탐색 및 reference baseline FEA와 분리됩니다. "
            "현재 인증된 scientific/production PASS는 0입니다."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 25,
        "evidence": [
            (
                "authoritative axis A: W=drawing x/current 973mm-side-secondary "
                "direction <=1200mm / L=drawing y/perpendicular direction "
                "<=1000mm / H<=750mm / rotation=false / axis swap=false"
            ),
            (
                "authoritative reference=설계도면260706 slide9 / "
                f"primary cw1={REFERENCE_PRIMARY_CW1_MM:.1f}mm / "
                f"gap1={REFERENCE_PRIMARY_GAP1_MM:.1f}mm"
            ),
            (
                "superseded candidate=official#5 / "
                f"cw1={SUPERSEDED_CANDIDATE_CW1_MM:.2f}mm / "
                f"gap1={SUPERSEDED_CANDIDATE_GAP1_MM:.1f}mm / "
                "final design eligible=false"
            ),
            (
                f"6-turn primary stack={superseded_stack:.2f}mm -> "
                f"{reference_stack:.2f}mm / geometry delta="
                f"{reference_stack - superseded_stack:.2f}mm / "
                "small numerical correction=false"
            ),
            (
                "old candidate task96340=CANCELLED / task96342=CANCELLED / "
                "invalid=true / selection eligible=false / "
                "production promotion eligible=false"
            ),
            (
                "reference baseline=task96396 + local AEDT GUI / "
                "role=baseline only, not an optimized candidate"
            ),
            (
                "superseded optimization=axis-v6 tasks96397-96412 / "
                "wrong W<=1000,L<=1200 axis / historical screening only"
            ),
            (
                "current target optimization=W<=1200,L<=1000,H<=750 / "
                "fixed 5T/1.6 / no rotation or axis swap / submission pending"
            ),
            (
                "verification lane=standard/unrounded symmetric / "
                "rounded role=drawing and Full-geometry visualization only / "
                "rounded verification=false"
            ),
            (
                "resonance screening contract=full physical primary-referred / "
                "air-gap tuned Lm=2.000mH / Ltx=Lm+Llt_phys / "
                "Lrx=Ltx*(N2/N1)^2 / min(fTx,fRx)>=15kHz"
            ),
            (
                "prior geometry-predicted resonance final=false / "
                "historical fixed-Lm2mH rescore=screening-only / "
                "thermal surrogate extrapolation invalid"
            ),
            (
                "actual scientific PASS=0 / actual production PASS=0 / "
                "Pareto result pending / canonical promotion=false"
            ),
        ],
    }


def _pid_exists(pid: int) -> bool:
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except (OSError, SystemError, ValueError):
        return False
    return True


def _windows_process_snapshot() -> dict[int, tuple[int, str]]:
    """Return PID -> (parent PID, executable name) without shelling out."""

    if os.name != "nt":
        return {}

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_void_p),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    create_snapshot.restype = ctypes.c_void_p
    process_first = kernel32.Process32FirstW
    process_first.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    process_first.restype = ctypes.c_int
    process_next = kernel32.Process32NextW
    process_next.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    process_next.restype = ctypes.c_int
    snapshot = create_snapshot(0x00000002, 0)
    if snapshot in (None, ctypes.c_void_p(-1).value):
        return {}
    result: dict[int, tuple[int, str]] = {}
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(ProcessEntry32W)
        if not process_first(snapshot, ctypes.byref(entry)):
            return {}
        while True:
            result[int(entry.th32ProcessID)] = (
                int(entry.th32ParentProcessID),
                str(entry.szExeFile),
            )
            entry.dwSize = ctypes.sizeof(ProcessEntry32W)
            if not process_next(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return result


def _descendant_processes(
    root_pid: int,
    snapshot: Mapping[int, tuple[int, str]] | None = None,
) -> list[tuple[int, str]]:
    observed = dict(snapshot or _windows_process_snapshot())
    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _name) in observed.items():
            if pid in descendants:
                continue
            if parent_pid == root_pid or parent_pid in descendants:
                descendants.add(pid)
                changed = True
    return sorted(
        ((pid, observed[pid][1]) for pid in descendants),
        key=lambda item: item[0],
    )


def _reference_thermal_terminal_marker(root: Path) -> dict[str, Any] | None:
    """Read the local thermal terminal marker before using process heuristics."""

    path = root.resolve() / "thermal_failure.json"
    try:
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size > MAX_RESPONSE_BYTES
        ):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema") != REFERENCE_THERMAL_TERMINAL_SCHEMA
        or value.get("sealed") is not True
        or value.get("terminal") is not True
        or str(value.get("status") or "").lower() != "failed"
        or str(value.get("state") or "").lower() != "failed"
        or value.get("thermal_solved") is not False
        or value.get("temperature_results_available") is not False
    ):
        return None
    value["marker_path"] = path
    return value


def _reference_thermal_retry_terminal_marker(
    root: Path,
) -> dict[str, Any] | None:
    path = root.resolve() / "thermal_mesh_l4_retry_failure.json"
    try:
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size > MAX_RESPONSE_BYTES
        ):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema") != REFERENCE_THERMAL_RETRY_TERMINAL_SCHEMA
        or value.get("sealed") is not True
        or value.get("terminal") is not True
        or value.get("task_id") != 96432
        or str(value.get("status") or "").lower() != "failed"
        or value.get("result_valid_thermal") is not False
        or value.get("temperature_results_available") is not False
    ):
        return None
    value["marker_path"] = path
    return value


def _reference_local_stage(root: Path) -> dict[str, Any]:
    resolved = root.resolve()
    run_root = resolved / "simulation" / "simulation1"
    stdout_path = resolved / "local_thermal_retry_stdout.log"
    stderr_path = resolved / "local_thermal_retry_stderr.log"
    if not stdout_path.is_file():
        stdout_path = resolved / "local_gui_stdout.log"
    if not stderr_path.is_file():
        stderr_path = resolved / "local_gui_stderr.log"
    try:
        stdout_bytes = stdout_path.read_bytes()
    except OSError:
        stdout_bytes = b""
    try:
        stderr_bytes = stderr_path.read_bytes()
    except OSError:
        stderr_bytes = b""
    stdout_tail = stdout_bytes[-1024 * 1024 :].decode("utf-8", errors="replace")
    stderr_tail = stderr_bytes[-1024 * 1024 :].decode("utf-8", errors="replace")
    project_path = run_root / "simulation1.aedt"
    result_csv = resolved / "simulation_results_260706.csv"
    result_parts = resolved / "results_parts_260706"
    loss_result_root = run_root / "simulation1.aedtresults" / "maxwell_loss"
    loss_dispatched = '"stage":"loss"' in stdout_tail
    thermal_dispatched = (
        '"stage":"thermal"' in stdout_tail
        or "Solving design setup ThermalSetup" in stdout_tail
    )
    terminal_marker = _reference_thermal_terminal_marker(resolved)
    thermal_failed = terminal_marker is not None
    retry_terminal_marker = _reference_thermal_retry_terminal_marker(resolved)
    thermal_retry_failed = retry_terminal_marker is not None
    aedt_pid_active = _pid_exists(44520)
    python_pid_active = _pid_exists(34080)
    descendants = (
        _descendant_processes(44520)
        if aedt_pid_active
        and not thermal_failed
        and "Solving design setup ThermalSetup" in stdout_tail
        else []
    )
    fluent_pids = [
        pid
        for pid, name in descendants
        if name.casefold() == "fluent.exe"
    ]
    retry = (
        terminal_marker.get("retry")
        if isinstance(terminal_marker, Mapping)
        and isinstance(terminal_marker.get("retry"), Mapping)
        else {}
    )
    return {
        "root": resolved,
        "project_path": project_path,
        "project_exists": project_path.is_file(),
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "aedt_pid": 44520,
        "aedt_pid_active": aedt_pid_active,
        "python_pid": 34080,
        "python_pid_active": python_pid_active,
        "matrix_solved": (run_root / "convergence_matrix.txt").is_file(),
        "cap_solved": (run_root / "convergence_cap.txt").is_file(),
        "loss_dispatched": loss_dispatched,
        "loss_running": (
            loss_dispatched
            and not thermal_dispatched
            and aedt_pid_active
            and python_pid_active
        ),
        "loss_result_root_present": loss_result_root.is_dir(),
        "loss_restore_failed": (
            "[loss] native Analyze completed but DSO restore failed" in stderr_tail
        ),
        "thermal_dispatched": thermal_dispatched,
        "thermal_running": (
            thermal_dispatched
            and not thermal_failed
            and aedt_pid_active
            and python_pid_active
        ),
        "fluent_running": bool(fluent_pids) and not thermal_failed,
        "fluent_pids": fluent_pids,
        "thermal_process_chain": [
            {"pid": pid, "name": name} for pid, name in descendants
        ],
        "result_csv_present": result_csv.is_file(),
        "result_parts_present": (
            result_parts.is_dir() and any(result_parts.glob("*.parquet"))
        ),
        "thermal_terminal_marker_present": thermal_failed,
        "thermal_failed": thermal_failed,
        "thermal_failure_stage": (
            str(terminal_marker.get("stage") or "")
            if terminal_marker is not None
            else ""
        ),
        "thermal_failure_class": (
            str(terminal_marker.get("failure_class") or "")
            if terminal_marker is not None
            else ""
        ),
        "thermal_failure_message": (
            str(terminal_marker.get("failure_message") or "")
            if terminal_marker is not None
            else ""
        ),
        "thermal_failure_marker_path": (
            terminal_marker.get("marker_path")
            if terminal_marker is not None
            else None
        ),
        "thermal_retry_status": str(retry.get("status") or ""),
        "thermal_retry_parameter": str(retry.get("parameter") or ""),
        "thermal_retry_target_value": retry.get("target_value"),
        "thermal_retry_failed": thermal_retry_failed,
        "thermal_retry_task_id": (
            retry_terminal_marker.get("task_id")
            if retry_terminal_marker is not None
            else None
        ),
        "thermal_retry_failure_class": (
            str(retry_terminal_marker.get("failure_class") or "")
            if retry_terminal_marker is not None
            else ""
        ),
        "thermal_retry_failure_message": (
            str(retry_terminal_marker.get("failure_message") or "")
            if retry_terminal_marker is not None
            else ""
        ),
    }


def _axis_v6_card(
    auxiliary_tasks: Mapping[int, Mapping[str, Any]],
    observed_at: str,
) -> dict[str, Any]:
    tasks = [auxiliary_tasks[spec.task_id] for spec in AXIS_V6_TASK_SPECS]
    categories = [_category(str(task["state"])) for task in tasks]
    running = categories.count("running")
    queued = categories.count("queued")
    succeeded = categories.count("succeeded")
    failed = categories.count("failed")
    warm = auxiliary_tasks[SUPERSEDED_WARM_TASK_SPEC.task_id]
    first_seed = auxiliary_tasks[AXIS_V6_TASK_SPECS[0].task_id]
    first_seed_outcome = (
        "task96397 exit0 / terminal rows=320 / fixed controls attested / "
        "raw/pre-fixedLm physical feasible=0 / single-seed provisional only"
        if _category(str(first_seed["state"])) == "succeeded"
        else f"task96397 lifecycle={str(first_seed['state']).upper()}"
    )
    task_evidence = []
    for index in range(0, len(tasks), 4):
        group = tasks[index : index + 4]
        task_evidence.append(
            " | ".join(
                (
                    f"t{task['task_id']} s{task['seed']} N1="
                    f"{task['primary_turns']} {str(task['state']).upper()} "
                    f"{task['actual_node_name'] or 'pending'}/"
                    f"j{task['slurm_job_id'] or 'none'}"
                )
                for task in group
            )
        )
    all_raw_succeeded = succeeded == 16
    return {
        "id": AXIS_V6_CARD_ID,
        "title": (
            (
                "SUPERSEDED HISTORICAL WRONG AXIS | AXIS-v6 16/16 SUCCEEDED | "
                "NEW-AXIS FEASIBLE 0 · PF 0"
            )
            if all_raw_succeeded
            else (
                "SUPERSEDED HISTORICAL WRONG AXIS | AXIS-v6 | "
                f"RUNNING {running} · QUEUED {queued} · "
                f"SUCCEEDED {succeeded} · FAILED {failed} | 16 SEEDS"
            )
        ),
        "detail": (
            "Tasks 96397-96412 used the now-superseded W≤1000/L≤1200 axis "
            "contract and are retained only as historical search evidence. The "
            "5,120 terminal rows collapse to 4,683 unique geometries. Projection "
            "onto the authoritative W≤1200/L≤1000 contract leaves 217 raw/210 "
            "unique geometry-pass rows; all pass mean fixed-Lm resonance, but "
            "none passes the combined thermal screen. The compact thermal "
            "surrogate is extrapolative here (reported minimum 302.67°C), so its "
            "temperature values cannot certify invalidity or feasibility. The "
            "fixed-Lm rescore and non-dominated sorting are screening-only and "
            "not production eligible."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 100 if all_raw_succeeded else 20 + (succeeded * 70 // 16),
        "evidence": [
            (
                "superseded historical axis=W/drawing-x<=1000mm / "
                "L/perpendicular-y<=1200mm / H<=750mm / rotation=false / "
                "axis swap=false / authoritative axis=false"
            ),
            (
                "campaign=axis-v6 strict / seeds=16 / population=320 / "
                "generations=80 / requested total=128CPU + 1TiB / "
                "per task=8CPU + 65536MB"
            ),
            (
                f"raw terminal={HISTORICAL_AXIS_RAW_TERMINAL} / "
                f"unique geometry={HISTORICAL_AXIS_UNIQUE_GEOMETRY} / "
                "raw lifecycle complete=true / combined seed NDS generated=true"
            ),
            (
                f"authoritative-axis projection geometry pass raw="
                f"{NEW_AXIS_GEOMETRY_PASS_RAW} / unique="
                f"{NEW_AXIS_GEOMETRY_PASS_UNIQUE} / all geometry-pass rows "
                "mean resonance pass=true"
            ),
            (
                f"geometry+resonance+thermal feasible={NEW_AXIS_THERMAL_FEASIBLE} / "
                f"production Pareto Front={NEW_AXIS_PRODUCTION_PARETO} / "
                "scientific PASS=0 / production PASS=0"
            ),
            (
                "compact thermal surrogate extrapolation invalid=true / "
                f"reported minimum={NEW_AXIS_COMPACT_SURROGATE_MIN_WINDING_C:.2f}C / "
                "temperature certification=false / retraining or FEA required"
            ),
            (
                "fixed-Lm2mH classification=screening-only / "
                "production eligible=false / automatic promotion=false / "
                "historical result cannot answer whether NSGA-II must be rerun"
            ),
            (
                f"prior task96395={str(warm['state']).upper()} superseded / "
                f"{first_seed_outcome}"
            ),
            *task_evidence,
        ],
    }


def _target_axis_card(
    auxiliary_tasks: Mapping[int, Mapping[str, Any]],
    observed_at: str,
    collector_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    legacy_tasks = [
        auxiliary_tasks[spec.task_id] for spec in TARGET_AXIS_TASK_SPECS
    ]
    fresh_tasks = [
        auxiliary_tasks[spec.task_id] for spec in FRESH_SPLITTEMP_TASK_SPECS
    ]
    all_seed_tasks = [*legacy_tasks, *fresh_tasks]
    fresh_categories = [
        _category(str(task["state"])) for task in fresh_tasks
    ]
    all_seed_categories = [
        _category(str(task["state"])) for task in all_seed_tasks
    ]
    fresh_running = fresh_categories.count("running")
    fresh_queued = fresh_categories.count("queued")
    fresh_succeeded = fresh_categories.count("succeeded")
    fresh_failed = fresh_categories.count("failed")
    all_succeeded = all_seed_categories.count("succeeded")
    all_failed = all_seed_categories.count("failed")
    symmetric_retry = auxiliary_tasks[
        FINAL_SYMMETRIC_RETRY_TASK_SPEC.task_id
    ]
    symmetric_retry_state = str(symmetric_retry["state"]).upper()
    collector_final = bool(
        collector_status
        and collector_status.get("global_nds_final") is True
        and collector_status.get("final_files_written") is True
    )
    raw_rows = int(
        (collector_status or {}).get("raw_terminal_row_count") or 0
    )
    unique_rows = int(
        (collector_status or {}).get(
            "geometry_deduplicated_candidate_count"
        )
        or 0
    )
    feasible = int(
        (collector_status or {}).get("global_screening_feasible_count")
        or 0
    )
    pareto_count = int(
        (collector_status or {}).get("partial_screening_pareto_count")
        or 0
    )
    conditional_count = int(
        (collector_status or {}).get(
            "partial_conditional_nonthermal_pareto_count"
        )
        or 0
    )
    least_violation_count = int(
        (collector_status or {}).get(
            "partial_minimum_violation_objective_front_count"
        )
        or 0
    )
    acquisition_count = int(
        (collector_status or {}).get("fea_acquisition_candidate_count")
        or 0
    )
    collector_success = int(
        (collector_status or {}).get("successful_terminal_seed_count") or 0
    )
    turn_evidence = []
    for turns in (5, 6, 7, 8):
        turn_tasks = [
            auxiliary_tasks[spec.task_id]
            for spec in FRESH_SPLITTEMP_TASK_SPECS
            if spec.primary_turns == turns
        ]
        turn_categories = [
            _category(str(task["state"])) for task in turn_tasks
        ]
        turn_evidence.append(
            (
                f"fresh N1={turns}: seeds=128 / "
                f"RUNNING {turn_categories.count('running')} / "
                f"QUEUED {turn_categories.count('queued')} / "
                f"SUCCEEDED {turn_categories.count('succeeded')} / "
                f"FAILED {turn_categories.count('failed')}"
            )
        )
    card = {
        "id": TARGET_AXIS_CARD_ID,
        "title": (
            "AUTHORITATIVE W1200/L1000 NSGA-II | FINAL528=OLD16+FRESH512 | "
            f"FRESH RUN {fresh_running} / QUEUE {fresh_queued} / "
            f"SUCCESS {fresh_succeeded} / FAIL {fresh_failed} | "
            f"SYM96743 {symmetric_retry_state} | GLOBAL NDS "
            f"{'COMPLETE' if collector_final else 'PENDING'}"
        ),
        "detail": (
            "The current authoritative optimization scope is the original 16 "
            "fixed-Lm2mH seeds plus 512 fresh split-temperature seeds. Each seed "
            "has 320 terminal candidates, so the final authenticated global "
            "non-dominated sort requires 528 seeds and 168,960 raw rows. The "
            "fresh base tasks occupy 96485-96740 and 96756-97011; IDs between "
            "those ranges are separate FEA/retry work. Task96743 is the final "
            "standard/unrounded symmetric tuned-gap retry. Scheduler lifecycle "
            "success alone is not scientific or production PASS."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": min(
            90,
            10
            + all_succeeded
            * 75
            // TARGET_AXIS_EXPECTED_SEED_COUNT,
        ),
        "evidence": [
            (
                "authoritative contract=W/drawing-x/original-973-direction "
                "<=1200mm / L/perpendicular-y<=1000mm / H<=750mm / "
                "rotation=false / axis swap=false"
            ),
            (
                "fixed winding controls=primary cw1 5.0mm / gap1 1.6mm / "
                "cooling boundary=1.5m/s and TIM unchanged / "
                "rounded winding excluded from verification"
            ),
            (
                "split-temperature final gate=primary winding<=100C / "
                "secondary winding<=120C / core<=120C"
            ),
            (
                f"campaign={TARGET_AXIS_CAMPAIGN} / old tasks=96416-96431 / "
                "fresh task ranges=96485-96740 + 96756-97011 / "
                "seeds=16+512=528"
            ),
            (
                "fresh requested total=4096CPU + 33554432MiB / "
                "per task=8CPU + 65536MiB / timeout=7200s / "
                "max_workers_per_node=8 / "
                f"splittemp hard contract sha256={TARGET_AXIS_HARD_SHA256} / "
                f"legacy hard contract sha256={TARGET_AXIS_LEGACY_HARD_SHA256} / "
                f"sealed base manifests={len(TARGET_AXIS_SUBMISSION_MANIFESTS)}"
            ),
            (
                "final aggregate target=528 successful logical seeds / "
                "168960 raw terminal rows / one cross-seed global NDS / "
                f"collector successful now={collector_success} / "
                f"all base lifecycle success={all_succeeded}/528 / "
                f"failed originals={all_failed} / exact-seed replacements are "
                "resolved by the authenticated collector before final NDS"
            ),
            (
                f"task96743 final symmetric tuned-gap retry="
                f"{symmetric_retry_state} / "
                f"{symmetric_retry['actual_node_name'] or 'pending'}/"
                f"j{symmetric_retry['slurm_job_id'] or 'none'} / "
                "last reported solver stage=loss"
            ),
            *turn_evidence,
            (
                "scientific PASS=0 / production PASS=0 / canonical "
                "promotion=false until symmetric FEA and final gate pass"
            ),
        ],
    }
    if collector_final:
        card.update(
            {
                "title": (
                    "AUTHORITATIVE W1200/L1000 TARGET NSGA-II | "
                    "GLOBAL NDS COMPLETE | FINAL528 | SEEDS "
                    f"{collector_success}/528 | RAW {raw_rows} | "
                    f"UNIQUE {unique_rows} | "
                    f"HARD-FEASIBLE {feasible} | "
                    f"PRODUCTION FRONT {pareto_count} | "
                    f"FEA {acquisition_count} | SYM96743 "
                    f"{symmetric_retry_state}"
                ),
                "detail": (
                    "All 16 legacy and 512 fresh split-temperature NSGA-II "
                    "logical seeds completed, and the authenticated 168,960-row "
                    "cross-seed global non-dominated sorting is final. The "
                    "production Front contains hard-feasible rows only; an "
                    "empty Front means no production candidate was found. "
                    "Minimum-violation and conditional Fronts are diagnostic "
                    "only and must never be shown as feasible or production. "
                    "Thermal values remain screening-only; production promotion "
                    "still requires the standard/unrounded symmetric FEA gate."
                ),
                "progress_pct": 100,
                "evidence": [
                    (
                        "authenticated global collector=FINAL / successful "
                        f"seeds={collector_success}/528 / raw terminal rows="
                        f"{raw_rows}/168960 / "
                        f"unique geometry={unique_rows}"
                    ),
                    (
                        f"hard-feasible screening rows={feasible} / production "
                        f"hard-feasible volume-loss Pareto={pareto_count} / "
                        f"conditional nonthermal diagnostic Front="
                        f"{conditional_count} / minimum-violation 3-objective "
                        f"diagnostic Front={least_violation_count} / "
                        "diagnostic Fronts are never feasible/production"
                    ),
                    (
                        "symmetric-unrounded FEA acquisition candidates="
                        f"{acquisition_count} / thermal classification="
                        "screening-only / production eligible=false"
                    ),
                    (
                        f"collector={DEFAULT_TARGET_AXIS_COLLECTOR_STATE_FILE} / "
                        "collector payload sha256="
                        f"{collector_status['payload_sha256']} / Pareto "
                        "manifest payload sha256="
                        f"{collector_status['pareto_manifest_payload_sha256']}"
                    ),
                    *card["evidence"][:6],
                    card["evidence"][6],
                    card["evidence"][-1],
                ],
            }
        )
    return card


def _reference_baseline_card(
    auxiliary_tasks: Mapping[int, Mapping[str, Any]],
    observed_at: str,
    gui_root: Path,
    local_stage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    original = auxiliary_tasks[REFERENCE_BASELINE_TASK_SPEC.task_id]
    hedge = auxiliary_tasks[REFERENCE_THERMAL_HEDGE_TASK_SPEC.task_id]
    task = auxiliary_tasks[REFERENCE_DIRECT_TASK_SPEC.task_id]
    local = dict(local_stage or _reference_local_stage(gui_root))
    local_gui = "PID44520 ACTIVE" if local["aedt_pid_active"] else "PID44520 EXITED"
    result_present = local["result_csv_present"] or local["result_parts_present"]
    retry_state = (
        f"retry task{local['thermal_retry_task_id']}="
        f"{local['thermal_retry_failure_class']}"
        if local["thermal_retry_failed"]
        else f"retry={local['thermal_retry_status'] or 'none'} "
        f"{local['thermal_retry_parameter'] or 'none'}->"
        f"{local['thermal_retry_target_value']}"
    )
    stage = (
        "THERMAL FAILED | L4 RETRY CONTROL-FLOW FAILED"
        if local["thermal_retry_failed"]
        else "THERMAL FAILED | RETRY PREPARED"
        if local["thermal_failed"]
        else "FLUENT THERMAL SOLVE RUNNING"
        if local["fluent_running"]
        else "THERMAL MESH RUNNING"
        if local["thermal_running"]
        else "MATRIX/CAP COMPLETE · LOSS RUNNING"
        if local["loss_running"]
        else "LOCAL LOSS RESTORE FAILED"
        if local["loss_restore_failed"]
        else "MATRIX/CAP COMPLETE · LOSS/THERMAL PENDING"
        if local["matrix_solved"] and local["cap_solved"]
        else "MATRIX/CAP BUILD OR SOLVE IN PROGRESS"
    )
    local_detail = (
        "Local retry-v2 produced no valid temperature field, and Slurm "
        "mesh-only retry task96432 terminated before thermal dispatch because "
        "the L4 mesh-quality canary is incompatible with the explicit "
        "direct-analyze control path. This is a control-flow failure, not a "
        "thermal-physics result; remote exact-L5 task96415 remains active."
        if local["thermal_retry_failed"]
        else
        "Local retry-v2 terminal marker overrides the retained AEDT/Python "
        "processes: native Fluent interrupted while reading/building the case, "
        "so no valid temperature field exists. AEDT remains open for GUI "
        "inspection only; the guarded Rx side-block mesh L4 retry is prepared."
        if local["thermal_failed"]
        else
        "Local retry-v2 has completed Matrix (10 passes), Capacitance "
        "(3 passes), and loss. AEDT PID44520 has launched the native Fluent "
        f"thermal solve (Fluent PIDs {local['fluent_pids']}) under controller "
        "PID34080."
        if local["fluent_running"]
        else "Local retry-v2 has completed Matrix (10 passes), Capacitance "
        "(3 passes), and loss, and PID44520 is executing the Icepak "
        "ThermalSetup under controller PID34080."
        if local["thermal_running"]
        else "Local retry-v2 has actually completed Matrix (10 passes) and "
        "Capacitance (3 passes), and PID44520 is solving maxwell_loss under "
        "controller PID34080. Thermal has not been dispatched."
    )
    return {
        "id": REFERENCE_BASELINE_CARD_ID,
        "title": (
            f"REFERENCE BASELINE | LOCAL {stage} · {local_gui} | "
            f"REMOTE96415 DIRECT {str(task['state']).upper()} "
            f"{task['actual_node_name'] or 'pending'}/j"
            f"{task['slurm_job_id'] or 'none'}"
        ),
        "detail": (
            "This track evaluates the literal 설계도면260706 baseline with "
            "primary 5.0/1.6 mm using standard symmetric, unrounded FEA. It is a "
            "reference baseline only and is not an optimized candidate. Local "
            f"status: {local_detail} Remote task96396 and the task96414 hedge "
            "were cancelled after stalling before thermal dispatch. Exact "
            "direct-analyze replacement task96415 is the active remote reference "
            "lane and remains separate from local result authority."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": (
            70
            if local["thermal_failed"]
            else 85
            if local["fluent_running"]
            else 75
            if local["thermal_running"]
            else 65
            if local["loss_running"]
            else 55
            if local["loss_dispatched"]
            else 30
        ),
        "evidence": [
            (
                f"active Slurm task96415={str(task['state']).upper()} / "
                f"node={task['actual_node_name'] or 'pending'} / "
                f"allocation={task['allocation_id'] or 'none'} / "
                f"job={task['slurm_job_id'] or 'none'} / "
                f"account={task['account_name'] or 'pending'} / "
                "exact direct-analyze replacement=true / "
                f"superseded task96396={str(original['state']).upper()} / "
                f"task96414={str(hedge['state']).upper()}"
            ),
            (
                "baseline geometry=symmetric eighth / winding=unrounded / "
                "round_corner=0 / cw1=5.0mm / gap1=1.6mm / "
                "not an axis-v6 candidate"
            ),
            (
                "temperature gates: primary winding<=100C / "
                "secondary winding<=120C / core<=120C / "
                "three values must be reported separately"
            ),
            (
                f"local AEDT PID44520 active="
                f"{str(local['aedt_pid_active']).lower()} / "
                f"python PID34080 active={str(local['python_pid_active']).lower()} / "
                f"local stage={stage} / Fluent active="
                f"{str(local['fluent_running']).lower()} / "
                f"Fluent PIDs={local['fluent_pids']}"
            ),
            (
                f"matrix solved={str(local['matrix_solved']).lower()} / "
                f"cap solved={str(local['cap_solved']).lower()} / "
                f"loss dispatched={str(local['loss_dispatched']).lower()} / "
                f"loss running={str(local['loss_running']).lower()} / "
                f"loss result root present="
                f"{str(local['loss_result_root_present']).lower()} / "
                f"thermal dispatched={str(local['thermal_dispatched']).lower()} / "
                f"thermal running={str(local['thermal_running']).lower()} / "
                f"thermal failed={str(local['thermal_failed']).lower()}"
            ),
            (
                "terminal marker priority="
                f"{str(local['thermal_terminal_marker_present']).lower()} / "
                f"failure stage={local['thermal_failure_stage'] or 'none'} / "
                f"class={local['thermal_failure_class'] or 'none'} / "
                f"{retry_state}"
            ),
            (
                "full-equivalent actual loss: primary=2626.206W / "
                "secondary=1089.438W / core=1659.728W / total=5375.372W / "
                "eighth thermal injection=328.276/136.180/207.466W"
            ),
            (
                "full-restored exact matrix: Lm=7.682399mH / "
                "k=0.995902386 / Llk=63.348112uH"
            ),
            (
                "full exact capacitance: Ctx=7.774728nF / Crx=769.620674pF / "
                "Ctr=276.391572pF"
            ),
            (
                "exact resonance: f_tx=20.509065kHz / "
                "f_rx=6.525010kHz LIMITING / f_inter=1.202793MHz / "
                "raw Lm=7.682399mH / raw historical only"
            ),
            (
                "final screening hypothetical: Lm=2.000mH by air gap / "
                "capacitance unchanged / limiting resonance=12.788kHz / "
                "15kHz pass=false / not an NSGA candidate"
            ),
            (
                f"result CSV present={str(local['result_csv_present']).lower()} / "
                "result parquet present="
                f"{str(local['result_parts_present']).lower()} / "
                f"Results reports pending={str(not result_present).lower()} / "
                "local EM valid=true / loss values available=true / "
                "thermal authenticated=false"
            ),
        ],
    }


def _supersede_rounded_candidate_card(card: dict[str, Any]) -> dict[str, Any]:
    card["title"] = (
        "CODEX | SUPERSEDED/NON-FINAL: PRIMARY 5T MISMATCH | HISTORICAL "
        + str(card["title"])
    )
    card["detail"] = (
        "NON-FINAL/SUPERSEDED: this lifecycle belongs to candidate #5 with "
        "primary cw1=1.13 mm and gap1=4.6 mm, while the authoritative drawing "
        "requires 5.0 mm and 1.6 mm. Any solver result remains operational "
        "history and is ineligible for corrected design selection or release. "
        + str(card["detail"])
    )
    evidence = card.get("evidence")
    if not isinstance(evidence, list):
        raise UpdaterError("rounded candidate card evidence drifted")
    if not evidence:
        raise UpdaterError("rounded candidate card evidence is empty")
    evidence[0] = (
        "superseded_by_primary_5T_constraint=true / non_final=true / "
        "selection_eligible=false / scientific_pass_eligible=false / "
        "drawing_release_eligible=false / historical_lifecycle: "
        + str(evidence[0])
    )
    card["progress_pct"] = 0
    return card


def _rounded_final_pipeline_card(
    tasks: Mapping[int, Mapping[str, Any]],
    observed_at: str,
) -> dict[str, Any]:
    task = tasks[ROUNDED_FINAL_TASK_ID]
    hedge = tasks[ROUNDED_TIMEOUT_HEDGE_TASK_ID]
    category = _category(str(task["state"]))
    hedge_category = _category(str(hedge["state"]))
    task_stage = {
        "running": "THERMAL RUNNING",
        "queued": "STANDARD QUEUED",
        "succeeded": "STANDARD SOLVER DONE · AUTH PENDING",
        "failed": "STANDARD OPERATIONAL TERMINAL",
    }[category]
    hedge_stage = {
        "running": "HEDGE RUNNING",
        "queued": "HEDGE QUEUED",
        "succeeded": "HEDGE SOLVER DONE · AUTH PENDING",
        "failed": "HEDGE OPERATIONAL TERMINAL",
    }[hedge_category]
    thermal_stage = (
        "ThermalSetup RUNNING"
        if category == "running"
        else "no active ThermalSetup claim"
    )
    return {
        "id": ROUNDED_FINAL_PIPELINE_CARD_ID,
        "title": (
            "ARCHIVED INVALID CANDIDATE #5 | cw1=1.13 | "
            "task96340/96342 CANCELLED | NOT IN AXIS-v6 | "
            f"LIFECYCLE {task_stage}/{hedge_stage}"
        ),
        "detail": (
            "Task96340 and task96342 are cancelled lifecycle history only: both use "
            "candidate #5 primary cw1=1.13 mm/gap1=4.6 mm and are superseded by "
            "the authoritative 5.0 mm/1.6 mm primary constraint. Their results "
            "cannot create a scientific PASS, select the corrected design, or "
            "release drawings. The nine-slide/nine-page DRAFT passed layout QA "
            "only and is specification-invalid. The replacement is the separate "
            "axis-v6 tasks96397-96412 optimization. Corrected verification uses "
            "standard/unrounded symmetric FEA; rounded geometry is for "
            "drawing/Full visualization only."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 10,
        "evidence": [
            (
                "primary constraint mismatch=true / reference cw1=5.0mm "
                "gap1=1.6mm / candidate cw1=1.13mm gap1=4.6mm / "
                "candidate #5 final status=superseded/non-final / "
                "task96340 selection eligible=false / "
                "task96342 selection eligible=false / "
                "corrected verification=standard/unrounded symmetric / "
                "rounded verification=false / "
                "rounded role=drawing and Full-geometry visualization only / "
                "candidate target ETA=1-2h / queue risk separate"
            ),
            (
                f"task96340 lifecycle={str(task['state']).upper()} / "
                f"allocation{task['allocation_id'] or 'none'} / "
                f"Slurm{task['slurm_job_id'] or 'none'} / "
                f"{thermal_stage}"
            ),
            (
                f"read-only GPFS snapshot={ROUNDED_SNAPSHOT_SIZE_BYTES:,}B / "
                f"SHA256 {ROUNDED_SNAPSHOT_SHA256} / "
                "source before-after identity equal=true / "
                "snapshot classification=file fallback only / "
                "solver_result_truth_included=false / scientific_pass=false / "
                "production_truth_eligible=false"
            ),
            (
                "task96341 CANCELLED / attach=false / start=false / Slurm job=none / "
                "solver_contact=false / scientific_failure=false / "
                "excluded from scientific/effective counts / "
                "task96341 reason=helper targeted /enroot while authenticated "
                "source MFT_WORKDIR was GPFS; source task96340 remained running"
            ),
            (
                f"rounded Full lane commit={ROUNDED_FULL_PREPARE_COMMIT} / "
                "one-shot gate + pre-solve geometry/setup checkpoint prepared / "
                "Scheduler GET0 POST0 / submit=false / "
                "Full checkpoint diagnostic_only=true / scientific_pass=false / "
                "thermal_pass=false / production_promotion_eligible=false"
            ),
            (
                f"rounded final package gate commit={ROUNDED_PACKAGE_GATE_COMMIT} / "
                "prepare+validate only / package publish=false / "
                "scientific package allowed=false until authenticated results"
            ),
            (
                f"drawing views exported={ROUNDED_DRAWING_VIEW_COUNT} PNG / "
                "read-only inspection copy / source project save=false / "
                "solver invoked=false / "
                f"superseded drawing draft={ROUNDED_DRAWING_DRAFT_SLIDES}-slide "
                f"PPTX {ROUNDED_DRAWING_DRAFT_PPTX_BYTES:,}B / "
                f"{ROUNDED_DRAWING_DRAFT_SLIDES}-page PDF "
                f"{ROUNDED_DRAWING_DRAFT_PDF_BYTES:,}B / "
                f"layout QA={ROUNDED_DRAWING_QA_PASSED}/"
                f"{ROUNDED_DRAWING_QA_PASSED} PASS / "
                "specification validity=false"
            ),
            (
                "drawing publication=false / final deliverable=false / "
                "blocked by primary 5T mismatch / task96340 result cannot release "
                "drawing / corrected design selection pending"
            ),
            (
                f"bounded correction contingency commit="
                f"{ROUNDED_BOUNDED_CORRECTION_COMMIT} / prepare-only / "
                "Scheduler POST0 / submit=false / "
                "ineligible for 5T recovery=true"
            ),
            (
                f"snapshot manifest SHA256 {ROUNDED_SNAPSHOT_MANIFEST_SHA256} / "
                "task96341 cancellation receipt SHA256 "
                f"{ROUNDED_TASK96341_CANCELLATION_SHA256} / drawing views "
                f"manifest SHA256 {ROUNDED_DRAWING_VIEWS_MANIFEST_SHA256} / "
                "supplemental manifest SHA256 "
                f"{ROUNDED_DRAWING_SUPPLEMENTAL_MANIFEST_SHA256}"
            ),
            (
                f"drawing readiness SHA256 "
                f"{ROUNDED_DRAWING_READINESS_MANIFEST_SHA256} / QA SHA256 "
                f"{ROUNDED_DRAWING_QA_SHA256} / "
                f"draft PPTX SHA256 {ROUNDED_DRAWING_DRAFT_PPTX_SHA256} / "
                f"draft PDF SHA256 {ROUNDED_DRAWING_DRAFT_PDF_SHA256}"
            ),
            (
                "actual scientific PASS=0 / actual production PASS=0 / "
                "automatic Full off / canonical promotion=false"
            ),
        ],
    }


def _final_drawing_card(observed_at: str) -> dict[str, Any]:
    return {
        "id": FINAL_DRAWING_CARD_ID,
        "title": (
            "CODEX | DRAWING INVALID | PRIMARY 5T MISMATCH | "
            "DRAFT SUPERSEDED | FINAL RELEASE OFF"
        ),
        "detail": (
            "The existing nine-slide DRAFT PPTX and nine-page DRAFT PDF were "
            "generated from candidate #5 with primary cw1=1.13 mm/gap1=4.6 mm. "
            "The authoritative reference requires 5.0 mm/1.6 mm, so the files "
            "are superseded and specification-invalid despite passing 20 layout "
            "checks. Publication and final release are disabled until a corrected "
            "5T design is selected, modeled, and verified."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 0,
        "evidence": [
            (
                "drawing validity=false / superseded=true / "
                "primary reference=5.0/1.6mm / draft model=1.13/4.6mm / "
                "template audit=complete / 설계도면260706.pdf pages=9 / "
                "설계도면260706.pptx slides=9 / 960x540pt / 16:9"
            ),
            (
                f"source PDF SHA256 {DRAWING_REFERENCE_PDF_SHA256} / "
                "source unchanged=true (before-after audit) / "
                f"source PPTX SHA256 {DRAWING_REFERENCE_PPTX_SHA256} / "
                "source unchanged=true (before-after audit)"
            ),
            (
                "round_corner=True=rounded-rectangle racetrack / "
                "straight spans + concentric corner arcs / circular coil=false"
            ),
            (
                f"rounded AEDT file fallback={ROUNDED_SNAPSHOT_SIZE_BYTES:,}B / "
                f"SHA256 {ROUNDED_SNAPSHOT_SHA256} / scientific model=false"
            ),
            (
                f"drawing views exported={ROUNDED_DRAWING_VIEW_COUNT} PNG / "
                f"primary manifest SHA256 {ROUNDED_DRAWING_VIEWS_MANIFEST_SHA256} / "
                "supplemental manifest SHA256 "
                f"{ROUNDED_DRAWING_SUPPLEMENTAL_MANIFEST_SHA256}"
            ),
            (
                "source project save=false / solver invoked=false / "
                f"corrected DRAFT PPTX slides={ROUNDED_DRAWING_DRAFT_SLIDES} / "
                f"corrected DRAFT PDF pages={ROUNDED_DRAWING_DRAFT_SLIDES}"
            ),
            (
                f"drawing readiness QA={ROUNDED_DRAWING_QA_PASSED}/"
                f"{ROUNDED_DRAWING_QA_PASSED} layout-only PASS / "
                "specification validity=false / "
                f"manifest SHA256 {ROUNDED_DRAWING_READINESS_MANIFEST_SHA256} / "
                f"QA SHA256 {ROUNDED_DRAWING_QA_SHA256}"
            ),
            (
                f"draft PPTX={ROUNDED_DRAWING_DRAFT_PPTX_BYTES:,}B / "
                f"SHA256 {ROUNDED_DRAWING_DRAFT_PPTX_SHA256} / "
                f"draft PDF={ROUNDED_DRAWING_DRAFT_PDF_BYTES:,}B / "
                f"SHA256 {ROUNDED_DRAWING_DRAFT_PDF_SHA256}"
            ),
            (
                "Z: publication=false / final PPTX claimed=false / "
                "final PDF claimed=false / final deliverable claimed=false"
            ),
            (
                "task96340 input superseded=true / task96340 result cannot release "
                "drawing / corrected 5T design selection and verification required"
            ),
            (
                f"bounded correction contingency commit="
                f"{ROUNDED_BOUNDED_CORRECTION_COMMIT} / prepared=true / "
                "Scheduler POST0 / submitted=false / 5T eligible=false"
            ),
            (
                "audit root=C:\\Users\\peets\\slurm_scheduler_runtime\\"
                "mft_goal_20260726\\drawing_reference_audit_v1"
            ),
        ],
    }


def _counter(pattern: re.Pattern[str], title: str, label: str) -> int:
    values = {int(value) for value in pattern.findall(title)}
    if len(values) != 1:
        raise UpdaterError(f"existing {label} counter is unavailable")
    return next(iter(values))


def _protected_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    completed = payload.get("completed")
    attention = payload.get("attention")
    if not isinstance(completed, list) or not isinstance(attention, list):
        raise UpdaterError("protected status groups are missing")
    pareto = [
        item
        for item in attention
        if isinstance(item, dict) and item.get("id") == "pareto-truth-boundary"
    ]
    if len(pareto) != 1:
        raise UpdaterError("Pareto truth boundary is not unique")
    return {
        "completed_sha256": canonical_sha256(completed),
        "attention_sha256": canonical_sha256(attention),
        "pareto_truth_sha256": canonical_sha256(pareto[0]),
    }


def _live_summary(
    *,
    observed_at: str,
    allocation_jobs: int,
    submitted: int,
    running: int,
    queued: int,
    collections: int,
    auxiliary_tasks: Mapping[int, Mapping[str, Any]] | None = None,
    reference_local: Mapping[str, Any] | None = None,
) -> str:
    try:
        observed = datetime.fromisoformat(observed_at)
    except ValueError as exc:
        raise UpdaterError("observed_at is not ISO-8601") from exc
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise UpdaterError("observed_at must include a UTC offset")
    terminal = len(TASK_SPECS) - running - queued
    if terminal < 0:
        raise UpdaterError("live task counters are inconsistent")
    authoritative = ""
    if auxiliary_tasks is not None:
        reference_original = auxiliary_tasks[REFERENCE_BASELINE_TASK_SPEC.task_id]
        reference_hedge = auxiliary_tasks[REFERENCE_THERMAL_HEDGE_TASK_SPEC.task_id]
        reference = auxiliary_tasks[REFERENCE_DIRECT_TASK_SPEC.task_id]
        warm = auxiliary_tasks[SUPERSEDED_WARM_TASK_SPEC.task_id]
        legacy_target_categories = [
            _category(str(auxiliary_tasks[spec.task_id]["state"]))
            for spec in TARGET_AXIS_TASK_SPECS
        ]
        fresh_target_categories = [
            _category(str(auxiliary_tasks[spec.task_id]["state"]))
            for spec in FRESH_SPLITTEMP_TASK_SPECS
        ]
        symmetric_retry = auxiliary_tasks[
            FINAL_SYMMETRIC_RETRY_TASK_SPEC.task_id
        ]
        reference_phase = (
            "local thermal failed; mesh-only retry96432 control-flow failed"
            if reference_local and reference_local.get("thermal_retry_failed")
            else "local thermal failed; mesh-only retry prepared"
            if reference_local and reference_local.get("thermal_failed")
            else "Fluent thermal solve running"
            if reference_local and reference_local.get("fluent_running")
            else "ThermalSetup running"
            if reference_local and reference_local.get("thermal_running")
            else "local thermal pending"
        )
        authoritative = (
            "authoritative final528=old16+fresh512; fresh tasks "
            "96485-96740 + 96756-97011 "
            f"run{fresh_target_categories.count('running')}/"
            f"queue{fresh_target_categories.count('queued')}/"
            f"success{fresh_target_categories.count('succeeded')}/"
            f"fail{fresh_target_categories.count('failed')}; old16 "
            f"success{legacy_target_categories.count('succeeded')}/"
            f"fail{legacy_target_categories.count('failed')}. "
            "Final global NDS requires 528 seeds/168960 rows; split-temp "
            "gate primary<=100C, secondary<=120C, core<=120C. "
            f"Final symmetric retry task96743="
            f"{str(symmetric_retry['state']).upper()} "
            f"{symmetric_retry['actual_node_name'] or 'pending'}/"
            f"j{symmetric_retry['slurm_job_id'] or 'none'}; "
            "Old W1000/L1200 axis-v6 is superseded. "
            f"reference local {reference_phase}; remote96415="
            f"{str(reference['state']).upper()} "
            f"{reference['actual_node_name'] or 'pending'}/"
            f"j{reference['slurm_job_id'] or 'none'}; "
            f"task96396={str(reference_original['state']).upper()} and "
            f"96414={str(reference_hedge['state']).upper()}; "
            f"warm96395={str(warm['state']).upper()}. "
        )
    summary = (
        f"{observed:%H:%M} KST · 권위 축 계약: W=도면x≤1200, L=수직y≤1000, "
        "H≤750, 회전/축교환 금지, 1차=5T/1.6mm. "
        f"{authoritative}"
        "기존 fixed-Lm rescore/NDS는 screening-only입니다. "
        f"post-deadline diagnostic 작업: running{running} · queued{queued} · "
        f"terminal{terminal}. Slurm allocation jobs{allocation_jobs} · "
        f"submitted{submitted} · collections{collections}. "
        "현재 결과 범위는 old16+fresh512 final528 global NDS이며, Scheduler "
        "success도 수집 artifact 인증 전에는 scientific/production PASS가 "
        "아닙니다. 기존 screening physical feasible0은 생산 판정이 "
        "아닙니다. "
    )
    return summary + "actual scientific PASS=0 / actual production PASS=0."


def _parallel_workstreams_card(
    tasks: Mapping[int, Mapping[str, Any]],
    categories: Mapping[int, str],
    *,
    observed_at: str,
    allocation_jobs: int,
    running: int,
    queued: int,
) -> dict[str, Any]:
    terminal = len(TASK_SPECS) - running - queued
    operational_risks = [
        (spec, risk)
        for spec in TASK_SPECS
        if (
            risk := _force_cancel_risk(
                spec,
                category=categories[spec.task_id],
                observed_at=observed_at,
            )
        )
        is not None
    ]
    active_nodes = list(
        dict.fromkeys(
            str(tasks[spec.task_id]["actual_node_name"] or spec.requested_node)
            for spec in TASK_SPECS
            if categories[spec.task_id] in {"running", "queued"}
        )
    )
    nodes = f"ACTIVE NODES {len(active_nodes)}" if active_nodes else "NO ACTIVE NODES"
    evidence = [
        (
            f"active allocations{allocation_jobs} / running{running} / "
            f"queued{queued} / terminal{terminal} / all managed tasks "
            "diagnostic/noncanonical / Scheduler GET only / "
            "scientific_pass_generated=false / actual_scientific_pass_count=0 / "
            "actual_production_pass_count=0 / canonical_promotion=false"
        )
    ]
    task_evidence = [
        (
            f"task{spec.task_id} {spec.model_label} "
            f"{tasks[spec.task_id]['state']} / "
            f"allocation{tasks[spec.task_id]['allocation_id'] or 'none'} / "
            f"job{tasks[spec.task_id]['slurm_job_id'] or 'none'} / "
            f"{tasks[spec.task_id]['actual_node_name'] or spec.requested_node}"
        )
        for spec in TASK_SPECS
    ]
    evidence.extend(
        " | ".join(task_evidence[index : index + 2])
        for index in range(0, len(task_evidence), 2)
    )
    evidence.extend(
        (
            f"RISK task{spec.task_id} allocation{risk['allocation_id']} "
            f"force-cancel {risk['force_at']:%Y-%m-%d %H:%M:%S KST} / "
            f"task timeout {risk['timeout_at']:%Y-%m-%d %H:%M:%S KST} / "
            f"lead {_duration_text(risk['lead_seconds'])} / "
            "hard guarantee=false"
        )
        for spec, risk in operational_risks
    )
    active_risk_title = "".join(
        (
            f" · RISK task{spec.task_id}/allocation{risk['allocation_id']} "
            f"FORCE {risk['force_at']:%m-%d %H:%M:%S KST}"
        )
        for spec, risk in operational_risks
        if risk["active"]
    )
    return {
        "id": "parallel-workstreams",
        "title": (
            f"PARALLEL TRACKS · RUNNING {running} · QUEUED {queued} · "
            f"ALLOCATION JOBS {allocation_jobs} · {nodes}{active_risk_title}"
        ),
        "detail": (
            f"Scheduler GET-authenticated lifecycle for {len(TASK_SPECS)} managed "
            "post-deadline tracks. Each task stays separate from collection, "
            "scientific PASS, canonical promotion, and production truth."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 95 if running or queued else 100,
        "evidence": evidence,
    }


def merge_status(
    payload: Mapping[str, Any],
    tasks: Mapping[int, Mapping[str, Any]],
    *,
    observed_at: str,
    auxiliary_tasks: Mapping[int, Mapping[str, Any]] | None = None,
    reference_gui_root: Path | None = None,
    target_axis_collector_state_file: Path | None = None,
    postsuccess_state_file: Path | None = None,
    thermal_bridge_state_file: Path | None = None,
    local_symmetric_selection_state_file: Path | None = (
        DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE
    ),
    standard_full_continuation_state_file: Path | None = None,
    final_gate_root: Path | None = None,
) -> dict[str, Any]:
    if payload.get("schema_version") != STATUS_SCHEMA:
        raise UpdaterError("Codex status schema drifted")
    result = copy.deepcopy(dict(payload))
    result.pop(SYNC_KEY, None)
    protected_before = _protected_hashes(result)
    target_axis_collector = _target_axis_collector_state(
        target_axis_collector_state_file
    )
    reference_local = (
        _reference_local_stage(
            reference_gui_root or DEFAULT_REFERENCE_BASELINE_GUI_ROOT
        )
        if auxiliary_tasks is not None
        else None
    )
    if standard_full_continuation_state_file is None:
        _remove_current_card(result, LEGACY_CONTINUATION_CARD_ID)
    if auxiliary_tasks is not None:
        expected_auxiliary_ids = {
            spec.task_id for spec in AUTHORITATIVE_AUXILIARY_TASK_SPECS
        }
        if set(auxiliary_tasks) != expected_auxiliary_ids:
            raise UpdaterError("authoritative auxiliary task set drifted")
        _upsert_priority_current_card(
            result,
            _axis_v6_card(auxiliary_tasks, observed_at),
        )
        _upsert_priority_current_card(
            result,
            _reference_baseline_card(
                auxiliary_tasks,
                observed_at,
                reference_gui_root or DEFAULT_REFERENCE_BASELINE_GUI_ROOT,
                local_stage=reference_local,
            ),
        )
        _upsert_priority_current_card(
            result,
            _target_axis_card(
                auxiliary_tasks,
                observed_at,
                collector_status=target_axis_collector,
            ),
        )
    else:
        _remove_current_card(result, AXIS_V6_CARD_ID)
        _remove_current_card(result, TARGET_AXIS_CARD_ID)
        _remove_current_card(result, REFERENCE_BASELINE_CARD_ID)
    _upsert_priority_current_card(
        result,
        _primary_5t_recovery_card(observed_at),
    )
    for spec in TASK_SPECS:
        _upsert_current_card(
            result,
            _task_card(spec, tasks[spec.task_id], observed_at),
        )
    _upsert_current_card(
        result,
        _rounded_final_pipeline_card(tasks, observed_at),
    )
    _upsert_current_card(
        result,
        _symmetric_primary_policy_card(tasks, observed_at),
    )
    if postsuccess_state_file is not None:
        _upsert_current_card(
            result,
            _postsuccess_card(postsuccess_state_file, observed_at, tasks),
        )
    if thermal_bridge_state_file is not None:
        _upsert_current_card(
            result,
            _thermal_bridge_card(thermal_bridge_state_file, observed_at),
        )
    if local_symmetric_selection_state_file is not None:
        _upsert_current_card(
            result,
            _local_symmetric_selection_card(
                local_symmetric_selection_state_file,
                observed_at,
                tasks,
            ),
        )
    else:
        _remove_current_card(result, LOCAL_SYMMETRIC_SELECTION_CARD_ID)
    if standard_full_continuation_state_file is not None:
        _upsert_current_card(
            result,
            _standard_full_continuation_card(
                standard_full_continuation_state_file,
                observed_at,
            ),
        )
    if final_gate_root is not None:
        _upsert_current_card(
            result,
            _final_gate_card(final_gate_root, observed_at),
        )
    _upsert_current_card(result, _final_drawing_card(observed_at))

    handoff = _single_current(result, "fea-handoff")
    previous_title = str(handoff.get("title") or "")
    submitted = max(
        CAMPAIGN_SUBMITTED_FLOOR,
        _counter(SUBMITTED_PATTERN, previous_title, "submitted"),
    )
    collections = _counter(COLLECTIONS_PATTERN, previous_title, "collections")
    categories = {
        task_id: _category(str(task["state"])) for task_id, task in tasks.items()
    }
    running = sum(value == "running" for value in categories.values())
    queued = sum(value == "queued" for value in categories.values())
    active_allocations = {
        task["allocation_id"]
        for task_id, task in tasks.items()
        if categories[task_id] in {"running", "queued"}
        and task["allocation_id"] is not None
    }
    allocation_jobs = len(active_allocations)
    result["summary"] = _live_summary(
        observed_at=observed_at,
        allocation_jobs=allocation_jobs,
        submitted=submitted,
        running=running,
        queued=queued,
        collections=collections,
        auxiliary_tasks=auxiliary_tasks,
        reference_local=reference_local,
    )
    handoff_task_evidence = [
        (
            f"task{spec.task_id} {tasks[spec.task_id]['state']} / "
            f"allocation{tasks[spec.task_id]['allocation_id'] or 'none'} / "
            f"job{tasks[spec.task_id]['slurm_job_id'] or 'none'} / "
            f"{tasks[spec.task_id]['actual_node_name'] or spec.requested_node}"
        )
        for spec in TASK_SPECS
    ]
    handoff.update(
        {
            "title": (
                f"SLURM · ALLOCATION JOBS {allocation_jobs} · "
                f"SUBMITTED {submitted} · RUNNING {running} · "
                f"QUEUED {queued} · COLLECTIONS {collections}"
            ),
            "detail": (
                "Scheduler exact GET lifecycle 집계입니다. 각 작업의 terminal "
                "success도 별도 collector/artifact 인증 전에는 collection 또는 "
                "scientific PASS가 아닙니다. completed/attention/Pareto truth는 "
                "이 updater가 수정하지 않습니다."
            ),
            "state": "in_progress",
            "updated_at": observed_at,
            "progress_pct": 97,
            "evidence": [
                " | ".join(handoff_task_evidence[index : index + 2])
                for index in range(0, len(handoff_task_evidence), 2)
            ]
            + [
                (
                    f"active allocation jobs{allocation_jobs} / running{running} / "
                    f"queued{queued} / submitted{submitted} / "
                    f"collections{collections} preserved"
                ),
                (
                    "Scheduler GET only / scientific_pass_generated=false / "
                    "canonical_promotion=false / Scheduler project remains separate "
                    "from MFT repository"
                ),
            ],
        }
    )
    _upsert_current_card(
        result,
        _parallel_workstreams_card(
            tasks,
            categories,
            observed_at=observed_at,
            allocation_jobs=allocation_jobs,
            running=running,
            queued=queued,
        ),
    )
    result["generated_at"] = observed_at
    if _protected_hashes(result) != protected_before:
        raise UpdaterError("protected completed/attention/Pareto truth changed")
    unsigned_status = copy.deepcopy(result)
    status_payload_sha256 = canonical_sha256(unsigned_status)
    task_snapshot = [
        {
            "task_id": spec.task_id,
            "state": tasks[spec.task_id]["state"],
            "allocation_id": tasks[spec.task_id]["allocation_id"],
            "slurm_job_id": tasks[spec.task_id]["slurm_job_id"],
            "node_name": tasks[spec.task_id]["actual_node_name"],
            "exit_code": tasks[spec.task_id]["exit_code"],
            "failure_message": tasks[spec.task_id]["failure_message"],
        }
        for spec in TASK_SPECS
    ]
    auxiliary_snapshot = (
        [
            {
                "task_id": spec.task_id,
                "role": auxiliary_tasks[spec.task_id]["role"],
                "state": auxiliary_tasks[spec.task_id]["state"],
                "allocation_id": auxiliary_tasks[spec.task_id]["allocation_id"],
                "slurm_job_id": auxiliary_tasks[spec.task_id]["slurm_job_id"],
                "node_name": auxiliary_tasks[spec.task_id]["actual_node_name"],
                "exit_code": auxiliary_tasks[spec.task_id]["exit_code"],
                "failure_message": auxiliary_tasks[spec.task_id]["failure_message"],
                "seed": auxiliary_tasks[spec.task_id]["seed"],
                "primary_turns": auxiliary_tasks[spec.task_id]["primary_turns"],
            }
            for spec in (
                *LEGACY_AUXILIARY_TASK_SPECS,
                FINAL_SYMMETRIC_RETRY_TASK_SPEC,
            )
        ]
        if auxiliary_tasks is not None
        else []
    )
    axis_categories = (
        [
            _category(str(auxiliary_tasks[spec.task_id]["state"]))
            for spec in AXIS_V6_TASK_SPECS
        ]
        if auxiliary_tasks is not None
        else []
    )
    legacy_target_axis_categories = (
        [
            _category(str(auxiliary_tasks[spec.task_id]["state"]))
            for spec in TARGET_AXIS_TASK_SPECS
        ]
        if auxiliary_tasks is not None
        else []
    )
    fresh_target_axis_categories = (
        [
            _category(str(auxiliary_tasks[spec.task_id]["state"]))
            for spec in FRESH_SPLITTEMP_TASK_SPECS
        ]
        if auxiliary_tasks is not None
        else []
    )
    target_axis_categories = [
        *legacy_target_axis_categories,
        *fresh_target_axis_categories,
    ]
    result[SYNC_KEY] = _sealed(
        {
            "schema_version": SYNC_SCHEMA,
            "observed_at": observed_at,
            "scheduler_endpoint": "GET /api/tasks/{task_id}",
            "scheduler_methods_used": ["GET"],
            "scheduler_mutation_performed": False,
            "managed_task_ids": [spec.task_id for spec in TASK_SPECS],
            "authoritative_task_ids": [
                spec.task_id for spec in AUTHORITATIVE_AUXILIARY_TASK_SPECS
            ]
            if auxiliary_tasks is not None
            else [],
            "axis_v6": {
                "task_ids": [spec.task_id for spec in AXIS_V6_TASK_SPECS],
                "superseded_historical": True,
                "production_eligible": False,
                "historical_axis_contract": {
                    "width_drawing_x_max_mm": 1_000.0,
                    "length_perpendicular_y_max_mm": 1_200.0,
                    "height_max_mm": 750.0,
                    "rotation_allowed": False,
                    "axis_swap_allowed": False,
                },
                "running": axis_categories.count("running"),
                "queued": axis_categories.count("queued"),
                "succeeded": axis_categories.count("succeeded"),
                "failed": axis_categories.count("failed"),
                "requested_total_cpus": 128,
                "requested_total_memory_mb": 1_048_576,
                "raw_lifecycle_complete": (
                    axis_categories.count("succeeded") == 16
                ),
                "raw_terminal": HISTORICAL_AXIS_RAW_TERMINAL,
                "unique_geometry": HISTORICAL_AXIS_UNIQUE_GEOMETRY,
                "authoritative_axis_projection": {
                    "geometry_pass_raw": NEW_AXIS_GEOMETRY_PASS_RAW,
                    "geometry_pass_unique": NEW_AXIS_GEOMETRY_PASS_UNIQUE,
                    "all_geometry_pass_mean_resonance_pass": True,
                    "thermal_feasible": NEW_AXIS_THERMAL_FEASIBLE,
                    "production_pareto_count": NEW_AXIS_PRODUCTION_PARETO,
                },
                "thermal_screening": {
                    "surrogate_extrapolation_invalid": True,
                    "reported_minimum_C": (
                        NEW_AXIS_COMPACT_SURROGATE_MIN_WINDING_C
                    ),
                    "classification": "screening-only",
                },
                "resonance_screening_contract": {
                    "primary_referred_full_physical": True,
                    "lm_mH": 2.0,
                    "lm_tuned_by_air_gap": True,
                    "ltx_formula": "Lm+Llt_phys",
                    "lrx_formula": "Ltx*(N2/N1)^2",
                    "minimum_resonance_Hz": 15_000.0,
                },
                "scientific_pass_generated": False,
                "global_nds_generated": True,
                "fixed_lm_global_nds_generated": True,
            }
            if auxiliary_tasks is not None
            else None,
            "target_axis": {
                "width_drawing_x_max_mm": 1_200.0,
                "length_perpendicular_y_max_mm": 1_000.0,
                "height_max_mm": 750.0,
                "rotation_allowed": False,
                "axis_swap_allowed": False,
                "primary_winding_thickness_mm": 5.0,
                "primary_winding_gap_mm": 1.6,
                "rounded_winding_verification_model": False,
                "temperature_gate_C": {
                    "primary_winding_max": (
                        TARGET_AXIS_PRIMARY_TEMPERATURE_LIMIT_C
                    ),
                    "secondary_winding_max": (
                        TARGET_AXIS_SECONDARY_TEMPERATURE_LIMIT_C
                    ),
                    "core_max": TARGET_AXIS_CORE_TEMPERATURE_LIMIT_C,
                },
                "cooling_air_speed_mps": 1.5,
                "tim_changed": False,
                "submission_state": "submitted",
                "campaign": TARGET_AXIS_CAMPAIGN,
                "hard_contract_sha256": TARGET_AXIS_HARD_SHA256,
                "legacy_hard_contract_sha256": (
                    TARGET_AXIS_LEGACY_HARD_SHA256
                ),
                "submission_manifests": [
                    str(path) for path in TARGET_AXIS_SUBMISSION_MANIFESTS
                ],
                "expected_seed_count": TARGET_AXIS_EXPECTED_SEED_COUNT,
                "expected_fresh_seed_count": (
                    TARGET_AXIS_EXPECTED_FRESH_SEED_COUNT
                ),
                "expected_raw_terminal_rows": TARGET_AXIS_EXPECTED_RAW_ROWS,
                "legacy_task_ids": [
                    spec.task_id for spec in TARGET_AXIS_TASK_SPECS
                ],
                "fresh_task_ranges": [
                    list(task_range)
                    for task_range in FRESH_SPLITTEMP_TASK_RANGES
                ],
                "task_ids": [
                    *[spec.task_id for spec in TARGET_AXIS_TASK_SPECS],
                    *[
                        spec.task_id
                        for spec in FRESH_SPLITTEMP_TASK_SPECS
                    ],
                ],
                "running": target_axis_categories.count("running"),
                "queued": target_axis_categories.count("queued"),
                "succeeded": target_axis_categories.count("succeeded"),
                "failed": target_axis_categories.count("failed"),
                "fresh_running": fresh_target_axis_categories.count("running"),
                "fresh_queued": fresh_target_axis_categories.count("queued"),
                "fresh_succeeded": fresh_target_axis_categories.count(
                    "succeeded"
                ),
                "fresh_failed": fresh_target_axis_categories.count("failed"),
                "fresh_turn_strata": {
                    str(turns): {
                        "seed_count": 128,
                        "running": sum(
                            _category(
                                str(
                                    auxiliary_tasks[spec.task_id]["state"]
                                )
                            )
                            == "running"
                            for spec in FRESH_SPLITTEMP_TASK_SPECS
                            if spec.primary_turns == turns
                        ),
                        "queued": sum(
                            _category(
                                str(
                                    auxiliary_tasks[spec.task_id]["state"]
                                )
                            )
                            == "queued"
                            for spec in FRESH_SPLITTEMP_TASK_SPECS
                            if spec.primary_turns == turns
                        ),
                        "succeeded": sum(
                            _category(
                                str(
                                    auxiliary_tasks[spec.task_id]["state"]
                                )
                            )
                            == "succeeded"
                            for spec in FRESH_SPLITTEMP_TASK_SPECS
                            if spec.primary_turns == turns
                        ),
                        "failed": sum(
                            _category(
                                str(
                                    auxiliary_tasks[spec.task_id]["state"]
                                )
                            )
                            == "failed"
                            for spec in FRESH_SPLITTEMP_TASK_SPECS
                            if spec.primary_turns == turns
                        ),
                    }
                    for turns in (5, 6, 7, 8)
                },
                "requested_total_cpus": 4_224,
                "requested_total_memory_mb": 34_603_008,
                "fresh_requested_total_cpus": 4_096,
                "fresh_requested_total_memory_mb": 33_554_432,
                "population": 320,
                "generations": 80,
                "final_symmetric_retry": {
                    "task_id": FINAL_SYMMETRIC_RETRY_TASK_SPEC.task_id,
                    "candidate": "2b2138a99445",
                    "gap_mm": 0.860423,
                    "model": "standard/unrounded symmetric",
                    "state": auxiliary_tasks[
                        FINAL_SYMMETRIC_RETRY_TASK_SPEC.task_id
                    ]["state"],
                    "node_name": auxiliary_tasks[
                        FINAL_SYMMETRIC_RETRY_TASK_SPEC.task_id
                    ]["actual_node_name"],
                    "slurm_job_id": auxiliary_tasks[
                        FINAL_SYMMETRIC_RETRY_TASK_SPEC.task_id
                    ]["slurm_job_id"],
                    "last_reported_solver_stage": "loss",
                },
                "scientific_pass_generated": False,
                "production_pareto_count": 0,
                "global_nds_final": bool(
                    target_axis_collector
                    and target_axis_collector.get("global_nds_final")
                    is True
                ),
                "collector_status_file": (
                    str(target_axis_collector_state_file)
                    if target_axis_collector_state_file is not None
                    else None
                ),
                "collector_payload_sha256": (
                    target_axis_collector.get("payload_sha256")
                    if target_axis_collector
                    else None
                ),
                "raw_terminal_rows": (
                    target_axis_collector.get("raw_terminal_row_count")
                    if target_axis_collector
                    else 0
                ),
                "collector_successful_terminal_seeds": (
                    target_axis_collector.get(
                        "successful_terminal_seed_count"
                    )
                    if target_axis_collector
                    else 0
                ),
                "unique_geometry": (
                    target_axis_collector.get(
                        "geometry_deduplicated_candidate_count"
                    )
                    if target_axis_collector
                    else 0
                ),
                "screening_feasible": (
                    target_axis_collector.get(
                        "global_screening_feasible_count"
                    )
                    if target_axis_collector
                    else 0
                ),
                "least_violation_objective_front": (
                    target_axis_collector.get(
                        "partial_minimum_violation_objective_front_count"
                    )
                    if target_axis_collector
                    else 0
                ),
                "fea_acquisition_candidates": (
                    target_axis_collector.get(
                        "fea_acquisition_candidate_count"
                    )
                    if target_axis_collector
                    else 0
                ),
            }
            if auxiliary_tasks is not None
            else None,
            "submitted_total": submitted,
            "allocation_jobs_active": allocation_jobs,
            "running": running,
            "queued": queued,
            "collections_preserved": collections,
            "scientific_pass_generated": False,
            "actual_scientific_pass_count": 0,
            "actual_production_pass_count": 0,
            "operational_history": {
                "task96341": {
                    "state": "cancelled",
                    "attached": False,
                    "started": False,
                    "slurm_job_id": "",
                    "solver_contact": False,
                    "scientific_failure": False,
                    "included_in_scientific_effective_counts": False,
                },
                "task96342": {
                    "state": tasks[ROUNDED_TIMEOUT_HEDGE_TASK_ID]["state"],
                    "allocation_id": tasks[ROUNDED_TIMEOUT_HEDGE_TASK_ID][
                        "allocation_id"
                    ],
                    "slurm_job_id": tasks[ROUNDED_TIMEOUT_HEDGE_TASK_ID][
                        "slurm_job_id"
                    ],
                    "source_task_id": ROUNDED_FINAL_TASK_ID,
                    "same_candidate_and_physics": True,
                    "scheduler_post_calls": 1,
                    "collector_methods": ["GET"],
                    "collector_scheduler_mutation": False,
                    "operational_timeout_hedge": True,
                    "new_design": False,
                    "scientific_pass": False,
                    "included_in_scientific_effective_counts": False,
                }
            },
            "collection_generated": False,
            "canonical_promotion_generated": False,
            "protected": protected_before,
            "status_without_sync_sha256": status_payload_sha256,
            "tasks": task_snapshot,
            "authoritative_tasks": auxiliary_snapshot,
        }
    )
    return result


def validate_status_sync(payload: Mapping[str, Any]) -> dict[str, Any]:
    sync = validate_seal(payload.get(SYNC_KEY), SYNC_SCHEMA)
    unsigned_status = copy.deepcopy(dict(payload))
    unsigned_status.pop(SYNC_KEY, None)
    if sync.get("status_without_sync_sha256") != canonical_sha256(unsigned_status):
        raise UpdaterError("status payload does not match its sync seal")
    if sync.get("protected") != _protected_hashes(payload):
        raise UpdaterError("protected status hashes do not match sync seal")
    return sync


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def synchronize_once(
    *,
    status_file: Path,
    scheduler_url: str = DEFAULT_SCHEDULER_URL,
    task_reader: TaskReader | None = None,
    auxiliary_task_reader: TaskReader | None = None,
    observed_at: str | None = None,
    reference_gui_root: Path | None = None,
    target_axis_collector_state_file: Path | None = (
        DEFAULT_TARGET_AXIS_COLLECTOR_STATE_FILE
    ),
    postsuccess_state_file: Path | None = None,
    thermal_bridge_state_file: Path | None = None,
    local_symmetric_selection_state_file: Path | None = (
        DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE
    ),
    standard_full_continuation_state_file: Path | None = None,
    final_gate_root: Path | None = None,
) -> dict[str, Any]:
    tasks = fetch_tasks(scheduler_url, task_reader=task_reader)
    auxiliary_tasks = (
        fetch_authoritative_auxiliary_tasks(
            scheduler_url,
            task_reader=auxiliary_task_reader,
        )
        if task_reader is None or auxiliary_task_reader is not None
        else None
    )
    source = status_file.resolve().read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    try:
        payload = json.loads(source.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError("status file is invalid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise UpdaterError("status file root must be an object")
    updated = merge_status(
        payload,
        tasks,
        observed_at=observed_at or _timestamp(),
        auxiliary_tasks=auxiliary_tasks,
        reference_gui_root=(
            reference_gui_root or DEFAULT_REFERENCE_BASELINE_GUI_ROOT
        ),
        target_axis_collector_state_file=target_axis_collector_state_file,
        postsuccess_state_file=postsuccess_state_file,
        thermal_bridge_state_file=thermal_bridge_state_file,
        local_symmetric_selection_state_file=(local_symmetric_selection_state_file),
        standard_full_continuation_state_file=(standard_full_continuation_state_file),
        final_gate_root=final_gate_root,
    )
    validate_status_sync(updated)
    if len(_json_bytes(updated)) > 256 * 1024:
        raise UpdaterError("updated status exceeds monitor size bound")
    _atomic_json(status_file, updated, expected_sha256=source_sha256)
    return updated[SYNC_KEY]


def _append_log(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _sealed(
        {
            "schema_version": LOG_SCHEMA,
            "recorded_at": _timestamp(),
            **dict(event),
        }
    )
    with path.open("ab") as stream:
        stream.write(
            json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def run_updater(
    *,
    status_file: Path,
    scheduler_url: str,
    once: bool,
    interval_seconds: int,
    pid_file: Path,
    log_file: Path,
    lock_file: Path,
    task_reader: TaskReader | None = None,
    auxiliary_task_reader: TaskReader | None = None,
    max_cycles: int | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    postsuccess_state_file: Path | None = None,
    thermal_bridge_state_file: Path | None = None,
    local_symmetric_selection_state_file: Path | None = (
        DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE
    ),
    standard_full_continuation_state_file: Path | None = None,
    final_gate_root: Path | None = None,
    reference_gui_root: Path = DEFAULT_REFERENCE_BASELINE_GUI_ROOT,
    target_axis_collector_state_file: Path | None = (
        DEFAULT_TARGET_AXIS_COLLECTOR_STATE_FILE
    ),
) -> dict[str, Any] | None:
    if interval_seconds < 1:
        raise UpdaterError("interval-seconds must be positive")
    with SingleInstanceLock(lock_file):
        pid = _sealed(
            {
                "schema_version": PID_SCHEMA,
                "pid": os.getpid(),
                "started_at": _timestamp(),
                "command": [str(Path(sys.executable).resolve()), *sys.argv],
                "status_file": str(status_file.resolve()),
                "scheduler_url": scheduler_url.rstrip("/"),
                "scheduler_methods_allowed": ["GET"],
                "scheduler_mutation_performed": False,
                "interval_seconds": interval_seconds,
                "tool_sha256": _file_sha256(Path(__file__).resolve()),
            }
        )
        _atomic_json(pid_file, pid)
        _append_log(
            log_file,
            {
                "event": "updater_started",
                "pid": os.getpid(),
                "interval_seconds": interval_seconds,
            },
        )
        cycles = 0
        while True:
            try:
                result = synchronize_once(
                    status_file=status_file,
                    scheduler_url=scheduler_url,
                    task_reader=task_reader,
                    auxiliary_task_reader=auxiliary_task_reader,
                    reference_gui_root=reference_gui_root,
                    target_axis_collector_state_file=(
                        target_axis_collector_state_file
                    ),
                    postsuccess_state_file=postsuccess_state_file,
                    thermal_bridge_state_file=thermal_bridge_state_file,
                    local_symmetric_selection_state_file=(
                        local_symmetric_selection_state_file
                    ),
                    standard_full_continuation_state_file=(
                        standard_full_continuation_state_file
                    ),
                    final_gate_root=final_gate_root,
                )
                _append_log(
                    log_file,
                    {
                        "event": "cycle_completed",
                        "sync_payload_sha256": result["payload_sha256"],
                        "running": result["running"],
                        "queued": result["queued"],
                        "allocation_jobs_active": result["allocation_jobs_active"],
                    },
                )
            except Exception as exc:
                _append_log(
                    log_file,
                    {
                        "event": "cycle_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                if once:
                    raise
                result = None
            cycles += 1
            if once or (max_cycles is not None and cycles >= max_cycles):
                return result
            sleeper(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GET-only atomic updater for MFT post-deadline UI cards"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--scheduler-url", default=DEFAULT_SCHEDULER_URL)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--postsuccess-state-file", type=Path)
    parser.add_argument("--thermal-bridge-state-file", type=Path)
    parser.add_argument(
        "--local-symmetric-selection-state-file",
        type=Path,
        default=DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE,
    )
    parser.add_argument("--standard-full-continuation-state-file", type=Path)
    parser.add_argument("--final-gate-root", type=Path)
    parser.add_argument(
        "--reference-gui-root",
        type=Path,
        default=DEFAULT_REFERENCE_BASELINE_GUI_ROOT,
    )
    parser.add_argument(
        "--target-axis-collector-state-file",
        type=Path,
        default=DEFAULT_TARGET_AXIS_COLLECTOR_STATE_FILE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ui_root = args.status_file.resolve().parent
    result = run_updater(
        status_file=args.status_file,
        scheduler_url=args.scheduler_url,
        once=args.once,
        interval_seconds=args.interval_seconds,
        pid_file=args.pid_file or ui_root / "postdeadline-ui-updater.pid.json",
        log_file=args.log_file or ui_root / "postdeadline-ui-updater.jsonl",
        lock_file=args.lock_file or ui_root / "postdeadline-ui-updater.lock",
        postsuccess_state_file=args.postsuccess_state_file,
        thermal_bridge_state_file=args.thermal_bridge_state_file,
        local_symmetric_selection_state_file=(
            args.local_symmetric_selection_state_file
        ),
        standard_full_continuation_state_file=(
            args.standard_full_continuation_state_file
        ),
        final_gate_root=args.final_gate_root,
        reference_gui_root=args.reference_gui_root,
        target_axis_collector_state_file=(
            args.target_axis_collector_state_file
        ),
    )
    if args.once:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
