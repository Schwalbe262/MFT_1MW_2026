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
DEFAULT_ACTIVE_EXACT100_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\diagnostic_compact_offload"
    r"\mft-goal-diag-compact-50320c178b89180a4383855d"
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
ACTIVE_EXACT100_RECEIPT_SCHEMA = (
    "mft-goal-diagnostic-compact-submission-receipt-v1"
)
ACTIVE_EXACT100_BUNDLE_ID = (
    "mft-goal-diag-compact-50320c178b89180a4383855d"
)
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


@dataclass(frozen=True)
class LastmileTaskSpec:
    task_id: int
    rank: int
    task_name: str
    dedupe_key: str
    physical_geometry_sha256: str


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
LASTMILE_TASK_SPECS = (
    LastmileTaskSpec(
        task_id=97_033,
        rank=1,
        task_name="mft-goal-txlast-r01-d03866b78823",
        dedupe_key=(
            "mft-al:mft-goal-txlast-r01-d03866b78823:"
            "fdd2f268fa64ce39494f8d647f4162e4009a58e1:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "fe4f474407b07bcc"
        ),
        physical_geometry_sha256=(
            "d03866b788234d060b504a94e7a5956958956636cc789ef853b1e6ebdfd75db8"
        ),
    ),
    LastmileTaskSpec(
        task_id=97_034,
        rank=2,
        task_name="mft-goal-txlast-r02-0fbcc8e73970",
        dedupe_key=(
            "mft-al:mft-goal-txlast-r02-0fbcc8e73970:"
            "fdd2f268fa64ce39494f8d647f4162e4009a58e1:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "d0692e4c171ee82b"
        ),
        physical_geometry_sha256=(
            "0fbcc8e739704a24a311ccb35a28934de56a10d3e6cac04ded2ef7fa8dba63ab"
        ),
    ),
    LastmileTaskSpec(
        task_id=97_035,
        rank=3,
        task_name="mft-goal-txlast-r03-690ef78d4c97",
        dedupe_key=(
            "mft-al:mft-goal-txlast-r03-690ef78d4c97:"
            "fdd2f268fa64ce39494f8d647f4162e4009a58e1:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "292c76c4b94d17e4"
        ),
        physical_geometry_sha256=(
            "690ef78d4c97efed43dd1214d5627055e1d45f0545a4c03eff0800ec22ebe925"
        ),
    ),
    LastmileTaskSpec(
        task_id=97_036,
        rank=4,
        task_name="mft-goal-txlast-r04-26033d7b5cf9",
        dedupe_key=(
            "mft-al:mft-goal-txlast-r04-26033d7b5cf9:"
            "fdd2f268fa64ce39494f8d647f4162e4009a58e1:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "a3a9d677359b69f3"
        ),
        physical_geometry_sha256=(
            "26033d7b5cf9743d1ff2f2bb35617c3f18df3060422a33780e2072ed859db9de"
        ),
    ),
    LastmileTaskSpec(
        task_id=97_037,
        rank=5,
        task_name="mft-goal-txlast-r05-b31c08a13f3a",
        dedupe_key=(
            "mft-al:mft-goal-txlast-r05-b31c08a13f3a:"
            "fdd2f268fa64ce39494f8d647f4162e4009a58e1:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "3eb61f32a4553e4e"
        ),
        physical_geometry_sha256=(
            "b31c08a13f3a4812af0b3839292e9363e706423cebdf888ecd228f520e570bd8"
        ),
    ),
    LastmileTaskSpec(
        task_id=97_038,
        rank=6,
        task_name="mft-goal-txlast-r06-db7c5d4e8be2",
        dedupe_key=(
            "mft-al:mft-goal-txlast-r06-db7c5d4e8be2:"
            "fdd2f268fa64ce39494f8d647f4162e4009a58e1:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "65957b93c69a92f0"
        ),
        physical_geometry_sha256=(
            "db7c5d4e8be2a3aa33b7bee4ebcdccd712e26027cc614e6172cf92a7ed679863"
        ),
    ),
    LastmileTaskSpec(
        task_id=97_039,
        rank=7,
        task_name="mft-goal-txlast-r07-936854df051b",
        dedupe_key=(
            "mft-al:mft-goal-txlast-r07-936854df051b:"
            "fdd2f268fa64ce39494f8d647f4162e4009a58e1:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "44ac16b5943efe72"
        ),
        physical_geometry_sha256=(
            "936854df051b72b6023cf63f45f83aa225fb7fb501f6bb4c3a66248d8cc93c6b"
        ),
    ),
    LastmileTaskSpec(
        task_id=97_040,
        rank=8,
        task_name="mft-goal-txlast-r08-5e64db5d7be5",
        dedupe_key=(
            "mft-al:mft-goal-txlast-r08-5e64db5d7be5:"
            "fdd2f268fa64ce39494f8d647f4162e4009a58e1:"
            "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
            "ed9c138f8eb4a09c"
        ),
        physical_geometry_sha256=(
            "5e64db5d7be5249051e733f96cf436f9e7b7685f85f0d62ef13e0b5ca2d329d5"
        ),
    ),
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
COMPACT_DESIGN_STATUS_CARD_ID = "codex-compact-design-status"
RX_MAIN_L5_NATIVE_CANARY_CARD_ID = "codex-rx-main-l5-native-canary"
PRIMARY_5T_RECOVERY_CARD_ID = "codex-primary-5t-constraint-recovery"
AXIS_V6_CARD_ID = "codex-axis-v6-fixed-5t-nsga"
TARGET_AXIS_CARD_ID = "codex-target-axis-1200x1000-nsga"
REFERENCE_BASELINE_CARD_ID = "codex-reference-drawing-baseline"
EXACT_N1_6_GUI_FEA_CARD_ID = "codex-exact-n1-6-gui-fea"
EXACT_N1_6_CORRECTED_GUI_FEA_CARD_ID = (
    "codex-exact-n1-6-corrected-gui-fea"
)
LASTMILE_ACQUISITION_CARD_ID = "codex-primary-temperature-lastmile-p90"
LASTMILE_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\n1_6_primary_temperature_lastmile_v2"
)
LASTMILE_PLAN_SCHEMA = "mft-goal-targeted-symmetric-fea-batch-plan-v1"
LASTMILE_RECEIPT_SCHEMA = "mft-goal-targeted-symmetric-fea-submission-v1"
LASTMILE_PLAN_FILE_SHA256 = (
    "b5864f7fa96a3e4c8d3b0663332c5e8e1f3bdd0c49d6399e885a91055595407b"
)
LASTMILE_PLAN_PAYLOAD_SHA256 = (
    "25a71a6f5ef49c43acd3c0031485f1a6ac61f2f7f413f7754f15789747ad943a"
)
LASTMILE_RECEIPT_FILE_SHA256 = (
    "8cdce0f5f63fb1b600d488bad3aa1b1ca0d0e42ae1484d0c9a653cc4bc90318f"
)
LASTMILE_RECEIPT_PAYLOAD_SHA256 = (
    "a3338c7b3e62a59a1e43861be3755a4d59bba3c2f4e29ad95c01baa8cb6b4d82"
)
LASTMILE_PRIORITY = 90
LASTMILE_CPUS = 8
LASTMILE_MEMORY_MB = 65_536
LASTMILE_TIMEOUT_SECONDS = 14_400
EXACT_N1_6_GUI_SEED = 2_707_277_137
EXACT_N1_6_GUI_GEOMETRY_SHA256 = (
    "e5b4c3b73869c5af75b254fb0486acae121bb27d1e905fd5a0fb3d146332c2be"
)
EXACT_N1_6_GUI_SOURCE_TASK_ID = 96_622
EXACT_N1_6_GUI_CONTROLLER_PID = 3_224
EXACT_N1_6_GUI_AEDT_PID = 48_360
EXACT_N1_6_GUI_INITIAL_GAP_MM = 0.65
EXACT_N1_6_GUI_THERMAL_MESH_SECONDS = 1_075.99
EXACT_N1_6_CORRECTED_GUI_ROOT = Path(
    r"C:\w\mft-gui-6x60-corrected-b6-g065"
)
EXACT_N1_6_CORRECTED_GUI_FORENSIC_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\local_gui_6x60_rx_interface_forensic_v1"
)
EXACT_N1_6_CORRECTED_GUI_FORENSIC_UI_SCHEMA = (
    "mft-local-gui-rx-interface-ui-manifest-v1"
)
EXACT_N1_6_CORRECTED_GUI_FORENSIC_EVIDENCE_SCHEMA = (
    "mft-local-gui-rx-interface-case-parser-evidence-v1"
)
EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID = 50_132
EXACT_N1_6_CORRECTED_GUI_AEDT_PID = 45_568
EXACT_N1_6_CORRECTED_GUI_GRPC_PORT = 64_321
EXACT_N1_6_CORRECTED_GUI_SOLVER_CORES = 4
EXACT_N1_6_CORRECTED_CANARY_TASK_ID = 97_041
EXACT_N1_6_CORRECTED_CANARY_TASK_NAME = (
    "mft-goal-rx-shared-interface-canary-v1"
)
EXACT_N1_6_CORRECTED_CANARY_DEDUPE_KEY = (
    "mft-al:mft-goal-rx-shared-interface-canary-v1:"
    "c6a016c3a880acd632b12e52b02099cfe7b90fc5:"
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781:"
    "4619b92fe62b262e"
)
EXACT_N1_6_CORRECTED_CANARY_PAYLOAD_SHA256 = (
    "4a7c2df2957558abc98811924ab20b18c6702c76070780996309ba94dcffc718"
)
EXACT_N1_6_CORRECTED_CANARY_PARAMS_SHA256 = (
    "4619b92fe62b262ef4d632d79720fcd200d845e58c58cc017dd086cc588b7b8e"
)
EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256 = (
    "f7f6f2890be76943230e82d4e50992fac602537df17b5cf3a8556b04207e75b5"
)
TURN_GRADED_CAP_CARD_ID = "codex-turn-graded-cap-current25"
TURN_GRADED_CAP_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\turn_graded_cap_6x60_current25_v1"
)
TURN_GRADED_CAP_SUBMISSION_SCHEMA = (
    "mft-goal-current25-turn-graded-cap-submission-v1"
)
TURN_GRADED_CAP_COLLECTION_SCHEMA = (
    "mft-goal-current25-turn-graded-cap-collection-v1"
)
TURN_GRADED_CAP_CANARY_RECEIPT = "canary_submission_receipt.json"
TURN_GRADED_CAP_ACQUISITION_RECEIPT = "acquisition_submission_receipt.json"
TURN_GRADED_CAP_CANARY_COLLECTION = "canary_collection.json"
TURN_GRADED_CAP_ACQUISITION_COLLECTION = "acquisition_collection.json"
TURN_GRADED_CAP_PLAN_PAYLOAD_SHA256 = (
    "094e17deca33bd6083b101ff293f78cc11a91131830e86c4c709edb9c14d7a61"
)
TURN_GRADED_CAP_TASK_IDS = tuple(range(97_066, 97_116))
CLEAN_LIBRARY_THERMAL_CARD_ID = "codex-clean-library-thermal24-replay"
CLEAN_LIBRARY_THERMAL_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\clean_library_thermal24_replay_v1"
)
CLEAN_LIBRARY_THERMAL_SUBMISSION_SCHEMA = (
    "mft-goal-clean-library-thermal24-replay-submission-v1"
)
CLEAN_LIBRARY_THERMAL_RUNTIME_PROVENANCE_SCHEMA = (
    "mft-goal-clean-library-thermal24-replay-runtime-provenance-v1"
)
CLEAN_LIBRARY_THERMAL_PLAN_PAYLOAD_SHA256 = (
    "34de1d69b800100f4b411f2b7bd765cfa167b2d61cae4f43ba59a15877a1d18a"
)
CLEAN_LIBRARY_THERMAL_SUBMISSION_PAYLOAD_SHA256 = (
    "b54f7d6e96e2e1d20478ffe0f12968b6d0837b812e6cd36a5e047498ee0fe986"
)
CLEAN_LIBRARY_THERMAL_SOLVER_REVISION = (
    "c6a016c3a880acd632b12e52b02099cfe7b90fc5"
)
CLEAN_LIBRARY_THERMAL_LIBRARY_REVISION = (
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
)
CLEAN_LIBRARY_THERMAL_TASK_IDS = tuple(range(97_116, 97_140))
CLEAN_LIBRARY_THERMAL_SOURCE_TASK_IDS = tuple(range(97_042, 97_066))
HISTORICAL_AXIS_RAW_TERMINAL = 5_120
HISTORICAL_AXIS_UNIQUE_GEOMETRY = 4_683
NEW_AXIS_GEOMETRY_PASS_RAW = 217
NEW_AXIS_GEOMETRY_PASS_UNIQUE = 210
NEW_AXIS_THERMAL_FEASIBLE = 0
NEW_AXIS_PRODUCTION_PARETO = 0
NEW_AXIS_COMPACT_SURROGATE_MIN_WINDING_C = 302.67
COMPACT_DESIGN_CONTRACT_COMMIT = "ccdaa7d"
COMPACT_SEARCH_READINESS_COMMIT = "bc63ec5"
COMPACT_SEARCH_AUTH_COMMIT = "fcbefeb"
COMPACT_SEARCH_HARDENING_COMMIT = "1faee3c"
COMPACT_SEARCH_RELATED_TEST_COUNT = 78
COMPACT_SEARCH_AUDIT_FIX_COUNT = 4
COMPACT_OLD_GENERATION_SLICE_COUNT = 1_454
COMPACT_OLD_GENERATION_HARD_FEASIBLE_COUNT = 0
COMPACT_FOCUS_W_MAX_MM = 1_170.0
COMPACT_FOCUS_L_MAX_MM = 975.0
COMPACT_REFERENCE_VOLUME_L = 830.95994977
COMPACT_FRESH_EXACT_SEED_COUNT = 512
RX_MAIN_L5_NATIVE_CANARY_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rx_main_l5_native_earliest_canary_v2"
)
RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rx_main_l5_native_earliest_canary_successor_v1"
)
RX_MAIN_L5_NATIVE_CANARY_HEDGE_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rx_main_l5_native_earliest_canary_isolated_hedge_v1"
)
RX_MAIN_L5_STRICT_N107_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rx_main_l5_strict_n107_nonexclusive_canary_v1"
)
RX_MAIN_L5_STRICT_N107_SUCCESSOR_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\rx_main_l5_strict_n107_nonexclusive_canary_v2"
)
RX_MAIN_L5_NATIVE_CANARY_TASK_ID = 97_140
RX_MAIN_L5_NATIVE_CANARY_TASK_NAME = (
    "mft-goal-rxmain-l5-f7f6-earliest-v1"
)
RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID = 97_141
RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME = (
    "mft-goal-rxmain-l5-f7f6-earliest-v2"
)
RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID = 97_142
RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_NAME = (
    "mft-goal-rxmain-l5-f7f6-isolated-n107-v3"
)
RX_MAIN_L5_STRICT_N107_TASK_ID = 97_143
RX_MAIN_L5_STRICT_N107_TASK_NAME = (
    "mft-goal-rxmain-l5-f7f6-strict-n107-nonexclusive-v5"
)
RX_MAIN_L5_STRICT_N107_JOB_ID = "845454"
RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_ID = 97_144
RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_NAME = (
    "mft-goal-rxmain-l5-f7f6-strict-n107-nonexclusive-v6"
)
RX_MAIN_L5_STRICT_N107_SUCCESSOR_JOB_ID = "845487"
RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE = "n107"
RX_MAIN_L5_NATIVE_CANARY_SIMULTANEOUS_TASK_ID = 97_121
RX_MAIN_L5_NATIVE_CANARY_SHARED_ALLOCATION_ID = 14_648
RX_MAIN_L5_NATIVE_CANARY_SHARED_JOB_ID = "840585"
RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256 = (
    "f7f6f2890be76943230e82d4e50992fac602537df17b5cf3a8556b04207e75b5"
)
RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION = (
    "6ef2b687ae39a08d1b8b3c5fe7d4e553789107b8"
)
RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION = (
    "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
)
RX_MAIN_L5_NATIVE_CANARY_RECEIPT_SHA256 = (
    "9ab0941c687c71472cbcba6161e7132bab30364d1c3263ccfae22811301fd4c4"
)
RX_MAIN_L5_NATIVE_CANARY_PRE_SUBMIT_SHA256 = (
    "a0884da3024ba67c48812dae43479d7a25483eb43ceca8062355c7bdcda26a52"
)
RX_MAIN_L5_NATIVE_CANARY_PAYLOAD_SHA256 = (
    "755e58b238990df46534bcfbd63c87d2032086cde9c71f60ed3dae9deac45562"
)
RX_MAIN_L5_NATIVE_CANARY_PARAMS_SHA256 = (
    "d8ab19c738199230dc99cb788dc0654d0ae10895c40af445325c67cb2987cd10"
)
RX_MAIN_L5_NATIVE_CANARY_TERMINAL_GATE_SHA256 = (
    "5a38e36b4c57d9c0707908705116bcae24b0270796347080bcb7ae6b6fbd1c05"
)
RX_MAIN_L5_NATIVE_CANARY_TERMINAL_GATE_PAYLOAD_SHA256 = (
    "7a605a00269364222a19f72fd9b384b7e8448aa96ec30d891914d54b8b1c1bb4"
)
RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SHA256 = (
    "9996e00bc3068b0e59c5111b44873aebac28e23771633a567d95c7bf32a88908"
)
RX_MAIN_L5_NATIVE_CANARY_OLD_CORE_AUTH_SHA256 = (
    "aac640b0a02075eec505e7640ba8ce7807ffc91f39f82aefd6dac6508d3e17fa"
)
RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256 = (
    "0b9a5dbfaca629a64f4cb46454b03c5233902c24f39c9a2e65370e73e625140d"
)
RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_RECEIPT_SHA256 = (
    "3faf0616ccf0d44651a3c9881b7e093897a00b8df44a69e00ea80a1f21cca942"
)
RX_MAIN_L5_NATIVE_CANARY_LINEAGE_SHA256 = (
    "2483d6520f8a6e592b0e1ed8b187d548bb5bbb68f836a9a177b8fd4e0b0a3381"
)
RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_PAYLOAD_SHA256 = (
    "46a423089e8e6b8d9641db7299c3bad8da1dc8acd2b9722734c948244da8f1d4"
)
RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TERMINAL_GATE_SHA256 = (
    "a438548076aaf5481142a6cfe6fb8da2fec96037645d61cc60b7f89985658649"
)
RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TERMINAL_PAYLOAD_SHA256 = (
    "9f68fde13484e8079a5e9771743d1c2bc2db4f86a23ff6b228b52d72297c258e"
)
RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_STDOUT_SHA256 = (
    "aada2f6310a1de5f42245ff346353ec3eb7b68cc128349510edbe068df29f11f"
)
RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_STDERR_SHA256 = (
    "dd6c105313829d33a99761f49498ced05ec5446e5c7292ccf76bba39d280f99a"
)
RX_MAIN_L5_NATIVE_CANARY_HEDGE_RECEIPT_SHA256 = (
    "c757e518f66f26468bc87d5ea914d218e304a3f958e9ed099d1fe533dc85fe24"
)
RX_MAIN_L5_NATIVE_CANARY_HEDGE_PRE_SUBMIT_SHA256 = (
    "9f7ee2426b41e9cf87495750a4bb3a448f6a038c66d11359ed3bd8aae2a75e74"
)
RX_MAIN_L5_NATIVE_CANARY_HEDGE_POST_ATTEMPT_SHA256 = (
    "8348b7ac195ab9d7a09bd96b08e6764e7730478e14947017c4581ce6cd130b64"
)
RX_MAIN_L5_NATIVE_CANARY_HEDGE_PAYLOAD_SHA256 = (
    "758566fe1d9a8231da14681778d50740c0f41c4bafdb01d965cb0ce28a6b324e"
)
RX_MAIN_L5_STRICT_N107_RECEIPT_SHA256 = (
    "c5d24d7b96e54e2e179e627a496f6ebe15f4917c7eda33ad62d30f329707dbb9"
)
RX_MAIN_L5_STRICT_N107_PRE_SUBMIT_SHA256 = (
    "d51780ce0c6bf419a5d36ebe30d3e092e10b759006fbfbb71c5b6f85db911d43"
)
RX_MAIN_L5_STRICT_N107_POST_ATTEMPT_SHA256 = (
    "87b77c3fe4fb3ec55f5cf9bfcbe82160515e7bf865fbc08e0cdcd5f8a672a885"
)
RX_MAIN_L5_STRICT_N107_PAYLOAD_SHA256 = (
    "76f30b833d48e0d2a51825f4ed11734012c63b6e2af40076d579464097821bdd"
)
RX_MAIN_L5_STRICT_N107_MONITOR_STATE_SHA256 = (
    "992c6ed137386ec60a4fcd928333b457da6cfec950099562be5021bc1cd4d558"
)
RX_MAIN_L5_STRICT_N107_MONITOR_PROCESS_SHA256 = (
    "9cf54001b9d264fdd7e47f900a127308a20fbd658ccb12194ce77b9632f039ff"
)
RX_MAIN_L5_STRICT_N107_TERMINAL_GATE_SHA256 = (
    "c155909c2762485d32ccc33063f94ee7685eb5d814cad7aaabbc4123332bd275"
)
RX_MAIN_L5_STRICT_N107_TERMINAL_PAYLOAD_SHA256 = (
    "8e82e594de9bc4c7b8d3f90e62c495b0063ee8fa8d55bb3f6aa3737907614cb0"
)
RX_MAIN_L5_STRICT_N107_STDOUT_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
RX_MAIN_L5_STRICT_N107_STDERR_SHA256 = (
    "16b5072f445d8de0de384eb104576568b26df44d2093aad6e0b0eb136bb676d8"
)
RX_MAIN_L5_STRICT_N107_SUCCESSOR_RECEIPT_SHA256 = (
    "4ada4c964bcfdf6202c4cf078b509460fb46a441016075eeefe312b0a0d87540"
)
RX_MAIN_L5_STRICT_N107_SUCCESSOR_PRE_SUBMIT_SHA256 = (
    "0d605cf59ed3d9d43fd43dcd24960c8400552ca7576e1855e6cb12016282cd3e"
)
RX_MAIN_L5_STRICT_N107_SUCCESSOR_POST_ATTEMPT_SHA256 = (
    "2e0a8f9353210914c0a203111dd115869bb59a771f4c3ff10af868cef3bc86df"
)
RX_MAIN_L5_STRICT_N107_SUCCESSOR_PAYLOAD_SHA256 = (
    "f7c24c5d584c9919c13abfb5a889d40951d75b5c835041a8a3c21141308e2735"
)
RX_MAIN_L5_STRICT_N107_SUCCESSOR_MONITOR_PROCESS_SHA256 = (
    "4f3f03c08e107f4be08dc2dbf32885ebf2f158bca0038cbb0fea5dc4c89f95df"
)
RX_MAIN_L5_STRICT_N107_GUARD_PROCESS_SHA256 = (
    "51cc8fc5f57d645dbea92c6444cd285564510d650ae8f44c07214b22bfe86a15"
)
RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SCHEMA = (
    "mft-goal-rx-main-l5-monitor-state-v1"
)
RX_MAIN_L5_NATIVE_CANARY_MONITOR_PROCESS_SCHEMA = (
    "mft-goal-rx-main-l5-get-only-monitor-process-v1"
)
RX_MAIN_L5_NATIVE_CANARY_PLACEMENT_STATE_SCHEMA = (
    "mft-goal-rx-main-l5-placement-lineage-state-v1"
)
RX_MAIN_L5_NATIVE_CANARY_PLACEMENT_PROCESS_SCHEMA = (
    "mft-goal-rx-main-l5-placement-lineage-monitor-v1"
)
RX_MAIN_L5_NATIVE_CANARY_CPUS = 8
RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB = 65_536
RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS = 43_200
RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING = {
    "core_plate_pad_t": 2.0,
    "fan_config": "dual",
    "fan_velocity": 1.5,
    "full_model": 0,
    "k_ins": 0.2,
    "round_corner": 0,
    "wcp_pad_t": 2.0,
}
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
CORRECTED_5T_ROUNDED_FULL_GUI_PATH = (
    r"C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726"
    r"\local_5807135t_full_curved_gui_v1\simulation"
    r"\simulation_job_5807135t_full_curved_model_v1_20260727"
    r"\simulation_job_5807135t_full_curved_model_v1_20260727.aedt"
)
CORRECTED_5T_ROUNDED_FULL_GUI_SHA256 = (
    "66d568dec29fc3f419816a828f3de52d14841c57ff59434c0f02ec06580b6101"
)
CORRECTED_5T_ROUNDED_FULL_GUI_SIZE_BYTES = 10_980_217
CORRECTED_5T_ROUNDED_FULL_GUI_GEOMETRY = (
    "58071313a32e81aeaa5bc3febdddb0bfd230a2bec633940a4c180ba8b838c508"
)
CORRECTED_5T_FULL_RAW_CAP_AEDT_PATH = (
    r"C:\w\m580cap1\project"
    r"\simulation_job_5807135t_full_curved_cap_v1"
    r"\simulation_job_5807135t_full_curved_cap_v1.aedt"
)
CORRECTED_5T_FULL_RAW_CAP_AEDT_SHA256 = (
    "49c0fbb6900bc7b23fcaf53d93c8050009df0ed3f05adaddd99e6841493c6fe7"
)
CORRECTED_5T_FULL_RAW_CAP_RESULT_PATH = r"C:\w\m580cap1\direct_cap_result.json"
CORRECTED_5T_FULL_RAW_CAP_CRX_NF = 0.635780
CORRECTED_5T_FULL_RAW_CAP_LIMIT_NF = 0.555365785
CORRECTED_5T_FULL_RAW_CAP_FMIN_KHZ = 14.0193
CORRECTED_5T_FULL_RAW_CAP_SOLVE_SECONDS = 142.5687
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


def _validate_lastmile_task(
    spec: LastmileTaskSpec,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    identifiers = {
        int(value)
        for key in ("id", "task_id")
        if (value := task.get(key)) not in (None, "")
    }
    if identifiers != {spec.task_id}:
        raise UpdaterError(f"lastmile task{spec.task_id} identity drifted")
    expected = {
        "name": spec.task_name,
        "dedupe_key": spec.dedupe_key,
        "project": "MFT_1MW_2026v1",
        "priority": LASTMILE_PRIORITY,
        "cpus": LASTMILE_CPUS,
        "memory_mb": LASTMILE_MEMORY_MB,
        "timeout_seconds": LASTMILE_TIMEOUT_SECONDS,
        "max_workers_per_node": 1,
        "aedt_backend": "standalone",
        "scheduling_profile": "fea_bursty",
    }
    for key, value in expected.items():
        if task.get(key) != value:
            raise UpdaterError(
                f"lastmile task{spec.task_id} {key} drifted"
            )
    state = _task_state(task)
    actual_node = str(
        task.get("actual_node_name") or task.get("allocation_node_name") or ""
    )
    if state in RUNNING_STATES and (
        not actual_node or task.get("placement_contract_satisfied") is not True
    ):
        raise UpdaterError(
            f"lastmile task{spec.task_id} running placement is not satisfied"
        )
    return {
        "task_id": spec.task_id,
        "rank": spec.rank,
        "name": spec.task_name,
        "dedupe_key": spec.dedupe_key,
        "physical_geometry_sha256": spec.physical_geometry_sha256,
        "state": state,
        "allocation_id": _positive_or_none(
            task.get("allocation_id"), "allocation_id"
        ),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "actual_node_name": actual_node,
        "placement_contract_satisfied": task.get(
            "placement_contract_satisfied"
        ),
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "exit_code": task.get("exit_code"),
        "failure_message": str(task.get("failure_message") or "")[:350],
        "priority": LASTMILE_PRIORITY,
        "cpus": LASTMILE_CPUS,
        "memory_mb": LASTMILE_MEMORY_MB,
        "timeout_seconds": LASTMILE_TIMEOUT_SECONDS,
    }


def fetch_lastmile_tasks(
    scheduler_url: str,
    *,
    task_reader: TaskReader | None = None,
) -> dict[int, dict[str, Any]]:
    reader = task_reader or _get_scheduler_task
    with ThreadPoolExecutor(max_workers=len(LASTMILE_TASK_SPECS)) as executor:
        futures = {
            spec.task_id: executor.submit(reader, scheduler_url, spec.task_id)
            for spec in LASTMILE_TASK_SPECS
        }
        raw = {task_id: future.result() for task_id, future in futures.items()}
    return {
        spec.task_id: _validate_lastmile_task(spec, raw[spec.task_id])
        for spec in LASTMILE_TASK_SPECS
    }


def _validate_corrected_canary_task(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    identifiers = {
        int(value)
        for key in ("id", "task_id")
        if (value := task.get(key)) not in (None, "")
    }
    if identifiers != {EXACT_N1_6_CORRECTED_CANARY_TASK_ID}:
        raise UpdaterError("corrected canary task97041 identity drifted")
    expected = {
        "name": EXACT_N1_6_CORRECTED_CANARY_TASK_NAME,
        "dedupe_key": EXACT_N1_6_CORRECTED_CANARY_DEDUPE_KEY,
        "project": "MFT_1MW_2026v1",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": 8,
        "memory_mb": 65_536,
        "gpus": 0,
        "priority": 100,
        "timeout_seconds": 43_200,
        "max_workers_per_node": 1,
    }
    for key, expected_value in expected.items():
        if task.get(key) != expected_value:
            raise UpdaterError(
                f"corrected canary task97041 {key} drifted"
            )
    state = _task_state(task)
    actual_node = str(
        task.get("actual_node_name") or task.get("allocation_node_name") or ""
    )
    if state in RUNNING_STATES and (
        not actual_node or task.get("placement_contract_satisfied") is not True
    ):
        raise UpdaterError(
            "corrected canary task97041 running placement is not satisfied"
        )
    return {
        "task_id": EXACT_N1_6_CORRECTED_CANARY_TASK_ID,
        "name": EXACT_N1_6_CORRECTED_CANARY_TASK_NAME,
        "state": state,
        "allocation_id": _positive_or_none(
            task.get("allocation_id"), "allocation_id"
        ),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "actual_node_name": actual_node,
        "placement_contract_satisfied": task.get(
            "placement_contract_satisfied"
        ),
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "exit_code": task.get("exit_code"),
        "failure_message": str(task.get("failure_message") or "")[:350],
        "cpus": 8,
        "memory_mb": 65_536,
        "priority": 100,
        "timeout_seconds": 43_200,
    }


def _read_turn_graded_cap_receipt(path: Path) -> dict[str, Any]:
    """Read one immutable, self-sealed turn-graded submission receipt."""

    resolved = path.resolve()
    try:
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or resolved.stat().st_size > MAX_RESPONSE_BYTES
        ):
            raise UpdaterError(f"turn-graded receipt unavailable: {resolved}")
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError(
            f"turn-graded receipt is unreadable: {resolved}"
        ) from exc
    if not isinstance(value, dict):
        raise UpdaterError("turn-graded receipt root must be an object")
    unsigned = copy.deepcopy(value)
    observed_sha256 = unsigned.pop("payload_sha256", None)
    if observed_sha256 != canonical_sha256(unsigned):
        raise UpdaterError("turn-graded receipt payload seal drifted")
    if value.get("schema") != TURN_GRADED_CAP_SUBMISSION_SCHEMA:
        raise UpdaterError("turn-graded submission receipt schema drifted")
    if (
        value.get("plan_payload_sha256")
        != TURN_GRADED_CAP_PLAN_PAYLOAD_SHA256
    ):
        raise UpdaterError("turn-graded submission plan identity drifted")
    if value.get("complete") is not True:
        raise UpdaterError("turn-graded submission receipt is incomplete")
    if value.get("thermal_jobs_cancelled_or_modified") is not False:
        raise UpdaterError("turn-graded submission modified thermal jobs")
    return value


def _turn_graded_canary_collection(
    path: Path,
    *,
    expected_submission_payload_sha256: str,
) -> dict[str, Any]:
    """Authenticate the exact 6/60 Tx/Rx turn-graded result pair."""

    resolved = path.resolve()
    try:
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or resolved.stat().st_size > MAX_RESPONSE_BYTES
        ):
            raise UpdaterError("turn-graded canary collection is unavailable")
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError(
            "turn-graded canary collection is unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise UpdaterError("turn-graded canary collection must be an object")
    unsigned = copy.deepcopy(value)
    observed_sha256 = unsigned.pop("payload_sha256", None)
    if observed_sha256 != canonical_sha256(unsigned):
        raise UpdaterError("turn-graded canary collection seal drifted")
    expected = {
        "schema": TURN_GRADED_CAP_COLLECTION_SCHEMA,
        "selection": "canary",
        "plan_payload_sha256": TURN_GRADED_CAP_PLAN_PAYLOAD_SHA256,
        "submission_payload_sha256": expected_submission_payload_sha256,
        "all_terminal": True,
        "valid_result_count": 2,
        "valid_pair_count": 1,
        "fixed_Lm2mh_pair_pass_count": 1,
        "legacy_two_net_capacitance_used": False,
        "scheduler_mutation_performed": False,
        "symmetric_even_potential_diagnostic": True,
        "full_model_series_interconnect_attested": False,
        "final_design_pass_allowed_from_cap_collection_alone": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise UpdaterError(
                f"turn-graded canary collection {key} drifted"
            )
    rows = value.get("rows")
    pairs = value.get("pairs")
    if not isinstance(rows, list) or len(rows) != 2:
        raise UpdaterError("turn-graded canary result rows drifted")
    if not isinstance(pairs, list) or len(pairs) != 1:
        raise UpdaterError("turn-graded canary result pair drifted")
    rows_by_winding: dict[str, Mapping[str, Any]] = {}
    expected_task_ids = {"Tx": 97_066, "Rx": 97_067}
    expected_inductances = {"Tx": 0.002, "Rx": 0.2}
    for row in rows:
        if not isinstance(row, dict):
            raise UpdaterError("turn-graded canary row is invalid")
        winding = str(row.get("active_winding") or "")
        checks = row.get("contract_checks")
        capacitance = float(row.get("C_terminal_turn_graded_F") or 0.0)
        frequency = float(row.get("fixed_Lm2mh_resonance_Hz") or 0.0)
        inductance = float(
            row.get("fixed_inductance_for_resonance_H") or 0.0
        )
        if (
            winding not in expected_task_ids
            or winding in rows_by_winding
            or row.get("task_id") != expected_task_ids[winding]
            or row.get("source_corrected_task_id") != 97_041
            or row.get("candidate_index") != 0
            or row.get("observed_geometry_sha256")
            != EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256
            or row.get("status") != "completed"
            or row.get("exit_code") != 0
            or row.get("contract_valid") is not True
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
            or not math.isfinite(capacitance)
            or capacitance <= 0.0
            or not math.isfinite(frequency)
            or frequency < 15_000.0
            or row.get("fixed_Lm2mh_resonance_pass_15kHz") is not True
            or not math.isclose(
                inductance,
                expected_inductances[winding],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise UpdaterError(
                f"turn-graded canary {winding or 'unknown'} row drifted"
            )
        rows_by_winding[winding] = row
    if set(rows_by_winding) != {"Tx", "Rx"}:
        raise UpdaterError("turn-graded canary winding pair drifted")
    pair = pairs[0]
    f_tx = float(rows_by_winding["Tx"]["fixed_Lm2mh_resonance_Hz"])
    f_rx = float(rows_by_winding["Rx"]["fixed_Lm2mh_resonance_Hz"])
    f_min = min(f_tx, f_rx)
    if (
        not isinstance(pair, dict)
        or pair.get("candidate_index") != 0
        or pair.get("tx_task_id") != 97_066
        or pair.get("rx_task_id") != 97_067
        or pair.get("source_corrected_task_id") != 97_041
        or pair.get("observed_geometry_sha256")
        != EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256
        or pair.get("pair_contract_valid") is not True
        or pair.get("fixed_Lm2mh_pair_pass_15kHz") is not True
        or not math.isclose(
            float(pair.get("fixed_Lm2mh_fmin_Hz") or 0.0),
            f_min,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise UpdaterError("turn-graded canary pair contract drifted")
    return {
        "payload_sha256": observed_sha256,
        "file_sha256": _file_sha256(resolved),
        "C_tx_F": float(
            rows_by_winding["Tx"]["C_terminal_turn_graded_F"]
        ),
        "C_rx_F": float(
            rows_by_winding["Rx"]["C_terminal_turn_graded_F"]
        ),
        "f_tx_Hz": f_tx,
        "f_rx_Hz": f_rx,
        "f_min_Hz": f_min,
        "pair_pass_15kHz": True,
        "symmetric_even_potential_diagnostic": True,
        "full_model_series_interconnect_attested": False,
        "final_design_pass_allowed": False,
    }


def _turn_graded_acquisition_collection(
    path: Path,
    *,
    expected_submission_payload_sha256: str,
    expected_lanes: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate all 24 corrected-geometry turn-graded result pairs."""

    resolved = path.resolve()
    try:
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or resolved.stat().st_size > MAX_RESPONSE_BYTES
        ):
            raise UpdaterError(
                "turn-graded acquisition collection is unavailable"
            )
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError(
            "turn-graded acquisition collection is unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise UpdaterError(
            "turn-graded acquisition collection must be an object"
        )
    unsigned = copy.deepcopy(value)
    observed_sha256 = unsigned.pop("payload_sha256", None)
    if observed_sha256 != canonical_sha256(unsigned):
        raise UpdaterError("turn-graded acquisition collection seal drifted")
    expected = {
        "schema": TURN_GRADED_CAP_COLLECTION_SCHEMA,
        "selection": "acquisition",
        "plan_payload_sha256": TURN_GRADED_CAP_PLAN_PAYLOAD_SHA256,
        "submission_payload_sha256": expected_submission_payload_sha256,
        "all_terminal": True,
        "valid_result_count": 48,
        "valid_pair_count": 24,
        "fixed_Lm2mh_pair_pass_count": 24,
        "legacy_two_net_capacitance_used": False,
        "scheduler_mutation_performed": False,
        "symmetric_even_potential_diagnostic": True,
        "full_model_series_interconnect_attested": False,
        "final_design_pass_allowed_from_cap_collection_alone": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise UpdaterError(
                f"turn-graded acquisition collection {key} drifted"
            )
    rows = value.get("rows")
    pairs = value.get("pairs")
    if not isinstance(rows, list) or len(rows) != 48:
        raise UpdaterError("turn-graded acquisition result rows drifted")
    if not isinstance(pairs, list) or len(pairs) != 24:
        raise UpdaterError("turn-graded acquisition result pairs drifted")
    expected_task_ids = set(range(97_068, 97_116))
    if set(expected_lanes) != set(TURN_GRADED_CAP_TASK_IDS):
        raise UpdaterError("turn-graded expected lane set drifted")
    observed_task_ids: set[int] = set()
    rows_by_candidate: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise UpdaterError("turn-graded acquisition row is invalid")
        task_id = int(row.get("task_id") or 0)
        lane = expected_lanes.get(task_id)
        winding = str(row.get("active_winding") or "")
        candidate_index = int(row.get("candidate_index") or 0)
        checks = row.get("contract_checks")
        capacitance = float(row.get("C_terminal_turn_graded_F") or 0.0)
        frequency = float(row.get("fixed_Lm2mh_resonance_Hz") or 0.0)
        expected_inductance = 0.002 if winding == "Tx" else 0.2
        if (
            task_id not in expected_task_ids
            or task_id in observed_task_ids
            or lane is None
            or winding != lane["active_winding"]
            or candidate_index != lane["candidate_index"]
            or row.get("source_corrected_task_id") != lane["source_task_id"]
            or row.get("observed_geometry_sha256")
            != lane["geometry_sha256"]
            or row.get("status") != "completed"
            or row.get("exit_code") != 0
            or row.get("contract_valid") is not True
            or not isinstance(checks, dict)
            or not checks
            or not all(check is True for check in checks.values())
            or not math.isfinite(capacitance)
            or capacitance <= 0.0
            or not math.isfinite(frequency)
            or frequency < 15_000.0
            or row.get("fixed_Lm2mh_resonance_pass_15kHz") is not True
            or not math.isclose(
                float(
                    row.get("fixed_inductance_for_resonance_H") or 0.0
                ),
                expected_inductance,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise UpdaterError(
                f"turn-graded acquisition task{task_id} row drifted"
            )
        observed_task_ids.add(task_id)
        rows_by_candidate.setdefault(candidate_index, {})[winding] = row
    if observed_task_ids != expected_task_ids or set(
        rows_by_candidate
    ) != set(range(1, 25)):
        raise UpdaterError("turn-graded acquisition coverage drifted")
    pair_by_candidate: dict[int, Mapping[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise UpdaterError("turn-graded acquisition pair is invalid")
        candidate_index = int(pair.get("candidate_index") or 0)
        rows_for_candidate = rows_by_candidate.get(candidate_index)
        if (
            candidate_index in pair_by_candidate
            or rows_for_candidate is None
            or set(rows_for_candidate) != {"Tx", "Rx"}
        ):
            raise UpdaterError("turn-graded acquisition pair coverage drifted")
        tx_row = rows_for_candidate["Tx"]
        rx_row = rows_for_candidate["Rx"]
        f_min = min(
            float(tx_row["fixed_Lm2mh_resonance_Hz"]),
            float(rx_row["fixed_Lm2mh_resonance_Hz"]),
        )
        if (
            pair.get("tx_task_id") != tx_row["task_id"]
            or pair.get("rx_task_id") != rx_row["task_id"]
            or pair.get("source_corrected_task_id")
            != tx_row["source_corrected_task_id"]
            or pair.get("source_corrected_task_id")
            != rx_row["source_corrected_task_id"]
            or pair.get("observed_geometry_sha256")
            != tx_row["observed_geometry_sha256"]
            or pair.get("observed_geometry_sha256")
            != rx_row["observed_geometry_sha256"]
            or pair.get("pair_contract_valid") is not True
            or pair.get("fixed_Lm2mh_pair_pass_15kHz") is not True
            or not math.isclose(
                float(pair.get("fixed_Lm2mh_fmin_Hz") or 0.0),
                f_min,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise UpdaterError(
                f"turn-graded acquisition pair{candidate_index} drifted"
            )
        pair_by_candidate[candidate_index] = pair
    if set(pair_by_candidate) != set(range(1, 25)):
        raise UpdaterError("turn-graded acquisition pair set drifted")
    ranked = sorted(
        pair_by_candidate.values(),
        key=lambda item: float(item["fixed_Lm2mh_fmin_Hz"]),
        reverse=True,
    )
    return {
        "payload_sha256": observed_sha256,
        "file_sha256": _file_sha256(resolved),
        "valid_result_count": 48,
        "valid_pair_count": 24,
        "pair_pass_15kHz_count": 24,
        "min_f_min_Hz": min(
            float(pair["fixed_Lm2mh_fmin_Hz"]) for pair in ranked
        ),
        "max_f_min_Hz": float(ranked[0]["fixed_Lm2mh_fmin_Hz"]),
        "best_source_task_id": int(ranked[0]["source_corrected_task_id"]),
        "best_geometry_sha256": str(ranked[0]["observed_geometry_sha256"]),
        "symmetric_even_potential_diagnostic": True,
        "full_model_series_interconnect_attested": False,
        "final_design_pass_allowed": False,
    }


def _turn_graded_cap_submission_state(
    root: Path = TURN_GRADED_CAP_ROOT,
) -> dict[str, Any] | None:
    """Authenticate the exact 6/60 canary pair and 24 acquisition pairs."""

    resolved = root.resolve()
    canary_path = resolved / TURN_GRADED_CAP_CANARY_RECEIPT
    acquisition_path = resolved / TURN_GRADED_CAP_ACQUISITION_RECEIPT
    if not canary_path.is_file() or not acquisition_path.is_file():
        return None
    receipts = (
        _read_turn_graded_cap_receipt(canary_path),
        _read_turn_graded_cap_receipt(acquisition_path),
    )
    expected_by_selection = {
        "canary": (2, {0}),
        "acquisition": (48, set(range(1, 25))),
    }
    lanes: dict[int, dict[str, Any]] = {}
    for receipt in receipts:
        selection = str(receipt.get("selection") or "")
        if selection not in expected_by_selection:
            raise UpdaterError("turn-graded submission selection drifted")
        expected_count, expected_candidates = expected_by_selection[selection]
        submissions = receipt.get("submissions")
        if (
            receipt.get("submitted_lane_count") != expected_count
            or receipt.get("scheduler_post_or_dedupe_call_count")
            != expected_count
            or not isinstance(submissions, list)
            or len(submissions) != expected_count
        ):
            raise UpdaterError("turn-graded submission lane count drifted")
        observed_candidates: set[int] = set()
        windings_by_candidate: dict[int, set[str]] = {}
        for lane in submissions:
            if not isinstance(lane, dict):
                raise UpdaterError("turn-graded submission lane is invalid")
            task_id = int(lane.get("task_id") or 0)
            candidate_index = int(lane.get("candidate_index") or 0)
            active_winding = str(lane.get("active_winding") or "")
            geometry_sha256 = str(
                lane.get("observed_geometry_sha256") or ""
            )
            source_task_id = int(lane.get("source_corrected_task_id") or 0)
            if (
                task_id in lanes
                or active_winding not in {"Tx", "Rx"}
                or re.fullmatch(r"[0-9a-f]{64}", geometry_sha256) is None
                or lane.get("scheduler_mutation_performed") is not True
                or lane.get("submission_source")
                not in {"post_created", "dedupe_reused"}
                or source_task_id != 97_041 + candidate_index
            ):
                raise UpdaterError("turn-graded submission lane identity drifted")
            observed_candidates.add(candidate_index)
            windings_by_candidate.setdefault(candidate_index, set()).add(
                active_winding
            )
            lanes[task_id] = {
                "task_id": task_id,
                "candidate_index": candidate_index,
                "active_winding": active_winding,
                "name": str(lane.get("name") or ""),
                "dedupe_key": str(lane.get("dedupe_key") or ""),
                "geometry_sha256": geometry_sha256,
                "source_task_id": source_task_id,
                "priority": 100 if selection == "canary" else 96,
            }
        if observed_candidates != expected_candidates or any(
            windings != {"Tx", "Rx"}
            for windings in windings_by_candidate.values()
        ):
            raise UpdaterError("turn-graded Tx/Rx candidate pairing drifted")
    if set(lanes) != set(TURN_GRADED_CAP_TASK_IDS):
        raise UpdaterError("turn-graded task ID range drifted")
    canary_geometry = {
        lane["geometry_sha256"]
        for lane in lanes.values()
        if lane["candidate_index"] == 0
    }
    if canary_geometry != {EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256}:
        raise UpdaterError("turn-graded exact 6/60 geometry drifted")
    result = {
        "root": str(resolved),
        "lanes": lanes,
        "plan_payload_sha256": TURN_GRADED_CAP_PLAN_PAYLOAD_SHA256,
        "canary_receipt_payload_sha256": receipts[0]["payload_sha256"],
        "acquisition_receipt_payload_sha256": receipts[1]["payload_sha256"],
        "canary_receipt_file_sha256": _file_sha256(canary_path),
        "acquisition_receipt_file_sha256": _file_sha256(acquisition_path),
    }
    collection_path = resolved / TURN_GRADED_CAP_CANARY_COLLECTION
    result["canary_collection"] = (
        _turn_graded_canary_collection(
            collection_path,
            expected_submission_payload_sha256=receipts[0][
                "payload_sha256"
            ],
        )
        if collection_path.is_file()
        else None
    )
    acquisition_collection_path = (
        resolved / TURN_GRADED_CAP_ACQUISITION_COLLECTION
    )
    result["acquisition_collection"] = (
        _turn_graded_acquisition_collection(
            acquisition_collection_path,
            expected_submission_payload_sha256=receipts[1][
                "payload_sha256"
            ],
            expected_lanes=lanes,
        )
        if acquisition_collection_path.is_file()
        else None
    )
    return result


def _validate_turn_graded_cap_task(
    expected: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = int(expected["task_id"])
    identifiers = {
        int(value)
        for key in ("id", "task_id")
        if (value := task.get(key)) not in (None, "")
    }
    if identifiers != {task_id}:
        raise UpdaterError(f"turn-graded task{task_id} identity drifted")
    expected_fields = {
        "name": expected["name"],
        "dedupe_key": expected["dedupe_key"],
        "project": "MFT_1MW_2026v1",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": 8,
        "memory_mb": 65_536,
        "gpus": 0,
        "priority": expected["priority"],
        "timeout_seconds": 43_200,
        "max_workers_per_node": 1,
    }
    for key, expected_value in expected_fields.items():
        if task.get(key) != expected_value:
            raise UpdaterError(
                f"turn-graded task{task_id} {key} drifted"
            )
    state = _task_state(task)
    actual_node = str(
        task.get("actual_node_name") or task.get("allocation_node_name") or ""
    )
    if state in RUNNING_STATES and (
        not actual_node or task.get("placement_contract_satisfied") is not True
    ):
        raise UpdaterError(
            f"turn-graded task{task_id} running placement is not satisfied"
        )
    return {
        **dict(expected),
        "state": state,
        "allocation_id": _positive_or_none(
            task.get("allocation_id"), "allocation_id"
        ),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "actual_node_name": actual_node,
        "placement_contract_satisfied": task.get(
            "placement_contract_satisfied"
        ),
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "exit_code": task.get("exit_code"),
        "failure_message": str(task.get("failure_message") or "")[:350],
    }


def fetch_turn_graded_cap_tasks(
    scheduler_url: str,
    submission_state: Mapping[str, Any],
    *,
    task_reader: TaskReader | None = None,
) -> dict[int, dict[str, Any]]:
    lanes = submission_state.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != set(
        TURN_GRADED_CAP_TASK_IDS
    ):
        raise UpdaterError("turn-graded submission state is incomplete")
    reader = task_reader or _get_scheduler_task
    with ThreadPoolExecutor(max_workers=min(64, len(lanes))) as executor:
        futures = {
            task_id: executor.submit(reader, scheduler_url, task_id)
            for task_id in lanes
        }
        raw = {task_id: future.result() for task_id, future in futures.items()}
    return {
        task_id: _validate_turn_graded_cap_task(lanes[task_id], raw[task_id])
        for task_id in lanes
    }


def _clean_library_thermal_submission_state(
    root: Path = CLEAN_LIBRARY_THERMAL_ROOT,
) -> dict[str, Any] | None:
    """Authenticate the immutable 24-lane clean-library thermal replay."""

    resolved = root.resolve()
    receipt_path = resolved / "submission_receipt.json"
    if not receipt_path.is_file():
        return None
    try:
        if (
            receipt_path.is_symlink()
            or receipt_path.stat().st_size > MAX_RESPONSE_BYTES
        ):
            raise UpdaterError(
                "clean-library thermal submission receipt is unavailable"
            )
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError(
            "clean-library thermal submission receipt is unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise UpdaterError(
            "clean-library thermal submission receipt must be an object"
        )
    unsigned = copy.deepcopy(value)
    observed_payload_sha256 = unsigned.pop("payload_sha256", None)
    if observed_payload_sha256 != canonical_sha256(unsigned):
        raise UpdaterError(
            "clean-library thermal submission payload seal drifted"
        )
    expected = {
        "schema": CLEAN_LIBRARY_THERMAL_SUBMISSION_SCHEMA,
        "payload_sha256": CLEAN_LIBRARY_THERMAL_SUBMISSION_PAYLOAD_SHA256,
        "plan_payload_sha256": CLEAN_LIBRARY_THERMAL_PLAN_PAYLOAD_SHA256,
        "complete": True,
        "submitted_count": 24,
        "all_generated_commands_clean_library_preflight_passed": True,
        "source_tasks_cancelled_or_modified": False,
        "mft_solver_repository_modified": False,
        "scheduler_repository_modified": False,
    }
    observed = {**value, "payload_sha256": observed_payload_sha256}
    for key, expected_value in expected.items():
        if observed.get(key) != expected_value:
            raise UpdaterError(
                f"clean-library thermal submission {key} drifted"
            )
    submissions = value.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != 24:
        raise UpdaterError(
            "clean-library thermal submission lane count drifted"
        )
    lanes: dict[int, dict[str, Any]] = {}
    geometries: set[str] = set()
    source_tasks: set[int] = set()
    for lane in submissions:
        if not isinstance(lane, dict):
            raise UpdaterError(
                "clean-library thermal submission lane is invalid"
            )
        task_id = int(lane.get("task_id") or 0)
        source_task_id = int(lane.get("source_old_task_id") or 0)
        lane_index = int(lane.get("lane_index") or 0)
        geometry_sha256 = str(
            lane.get("physical_geometry_sha256") or ""
        )
        preflight = lane.get("generated_command_preflight")
        readback = lane.get("get_readback")
        if (
            task_id not in CLEAN_LIBRARY_THERMAL_TASK_IDS
            or task_id in lanes
            or source_task_id not in CLEAN_LIBRARY_THERMAL_SOURCE_TASK_IDS
            or task_id - source_task_id != 74
            or lane_index != task_id - CLEAN_LIBRARY_THERMAL_TASK_IDS[0] + 1
            or re.fullmatch(r"[0-9a-f]{64}", geometry_sha256) is None
            or geometry_sha256 in geometries
            or source_task_id in source_tasks
            or not isinstance(preflight, dict)
            or preflight.get("passed") is not True
            or preflight.get("library_export_marker_count") != 1
            or preflight.get("library_revision")
            != CLEAN_LIBRARY_THERMAL_LIBRARY_REVISION
            or preflight.get(
                "export_precedes_clone_checkout_test_and_python"
            )
            is not True
            or lane.get("scheduler_mutation_performed") is not True
            or lane.get("submission_source")
            not in {"post_created", "dedupe_reused"}
            or not isinstance(readback, dict)
            or int(readback.get("task_id") or 0) != task_id
            or int(readback.get("id") or 0) != task_id
        ):
            raise UpdaterError(
                f"clean-library thermal task{task_id} lane drifted"
            )
        geometries.add(geometry_sha256)
        source_tasks.add(source_task_id)
        lanes[task_id] = {
            "task_id": task_id,
            "source_task_id": source_task_id,
            "lane_index": lane_index,
            "name": str(lane.get("name") or ""),
            "dedupe_key": str(lane.get("dedupe_key") or ""),
            "geometry_sha256": geometry_sha256,
        }
    if (
        set(lanes) != set(CLEAN_LIBRARY_THERMAL_TASK_IDS)
        or source_tasks != set(CLEAN_LIBRARY_THERMAL_SOURCE_TASK_IDS)
        or len(geometries) != 24
    ):
        raise UpdaterError(
            "clean-library thermal submission coverage drifted"
        )
    runtime_provenance: dict[str, Any] | None = None
    provenance_path = resolved / "runtime_provenance_24of24.json"
    if provenance_path.is_file():
        try:
            if (
                provenance_path.is_symlink()
                or provenance_path.stat().st_size > MAX_RESPONSE_BYTES
            ):
                raise UpdaterError(
                    "clean-library runtime provenance is unavailable"
                )
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise UpdaterError(
                "clean-library runtime provenance is unreadable"
            ) from exc
        if not isinstance(provenance, dict):
            raise UpdaterError(
                "clean-library runtime provenance must be an object"
            )
        unsigned_provenance = copy.deepcopy(provenance)
        provenance_payload_sha256 = unsigned_provenance.pop(
            "payload_sha256", None
        )
        task_evidence = provenance.get("task_evidence")
        if (
            provenance_payload_sha256
            != canonical_sha256(unsigned_provenance)
            or provenance.get("schema")
            != CLEAN_LIBRARY_THERMAL_RUNTIME_PROVENANCE_SCHEMA
            or provenance.get("plan_payload_sha256")
            != CLEAN_LIBRARY_THERMAL_PLAN_PAYLOAD_SHA256
            or provenance.get("submission_payload_sha256")
            != CLEAN_LIBRARY_THERMAL_SUBMISSION_PAYLOAD_SHA256
            or provenance.get("expected_library_revision")
            != CLEAN_LIBRARY_THERMAL_LIBRARY_REVISION
            or provenance.get(
                "runtime_library_root_marker_count_cumulative"
            )
            != 24
            or provenance.get(
                "runtime_library_hash_count_cumulative"
            )
            != 24
            or provenance.get("scheduler_mutation_performed") is not False
            or provenance.get("source_tasks_cancelled_or_modified") is not False
            or not isinstance(task_evidence, list)
            or len(task_evidence) != 24
        ):
            raise UpdaterError(
                "clean-library runtime provenance contract drifted"
            )
        evidence_by_task: dict[int, Mapping[str, Any]] = {}
        for row in task_evidence:
            if not isinstance(row, dict):
                raise UpdaterError(
                    "clean-library runtime provenance row is invalid"
                )
            task_id = int(row.get("task_id") or 0)
            lane = lanes.get(task_id)
            if (
                lane is None
                or task_id in evidence_by_task
                or int(row.get("source_old_task_id") or 0)
                != lane["source_task_id"]
                or row.get("physical_geometry_sha256")
                != lane["geometry_sha256"]
                or not str(row.get("runtime_library_root") or "").endswith(
                    "/pyaedt_library"
                )
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(row.get("root_marker_stdout_sha256") or ""),
                )
                is None
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(row.get("hash_marker_stdout_sha256") or ""),
                )
                is None
            ):
                raise UpdaterError(
                    f"clean-library runtime task{task_id} provenance drifted"
                )
            evidence_by_task[task_id] = row
        if set(evidence_by_task) != set(CLEAN_LIBRARY_THERMAL_TASK_IDS):
            raise UpdaterError(
                "clean-library runtime provenance coverage drifted"
            )
        runtime_provenance = {
            "payload_sha256": provenance_payload_sha256,
            "file_sha256": _file_sha256(provenance_path),
            "root_marker_count": 24,
            "library_hash_count": 24,
        }
    return {
        "root": str(resolved),
        "receipt_file_sha256": _file_sha256(receipt_path),
        "submission_payload_sha256": observed_payload_sha256,
        "plan_payload_sha256": CLEAN_LIBRARY_THERMAL_PLAN_PAYLOAD_SHA256,
        "lanes": lanes,
        "runtime_provenance": runtime_provenance,
    }


def _validate_clean_library_thermal_task(
    expected: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = int(expected["task_id"])
    identifiers = {
        int(value)
        for key in ("id", "task_id")
        if (value := task.get(key)) not in (None, "")
    }
    if identifiers != {task_id}:
        raise UpdaterError(
            f"clean-library thermal task{task_id} identity drifted"
        )
    expected_fields = {
        "name": expected["name"],
        "dedupe_key": expected["dedupe_key"],
        "project": "MFT_1MW_2026v1",
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": "standalone",
        "cpus": 8,
        "memory_mb": 65_536,
        "gpus": 0,
        "priority": 99,
        "timeout_seconds": 43_200,
        "max_workers_per_node": 1,
    }
    for key, expected_value in expected_fields.items():
        if task.get(key) != expected_value:
            raise UpdaterError(
                f"clean-library thermal task{task_id} {key} drifted"
            )
    state = _task_state(task)
    actual_node = str(
        task.get("actual_node_name") or task.get("allocation_node_name") or ""
    )
    if state in RUNNING_STATES and (
        not actual_node or task.get("placement_contract_satisfied") is not True
    ):
        raise UpdaterError(
            f"clean-library thermal task{task_id} running placement drifted"
        )
    return {
        **dict(expected),
        "state": state,
        "allocation_id": _positive_or_none(
            task.get("allocation_id"), "allocation_id"
        ),
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "actual_node_name": actual_node,
        "placement_contract_satisfied": task.get(
            "placement_contract_satisfied"
        ),
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "exit_code": task.get("exit_code"),
        "failure_message": str(task.get("failure_message") or "")[:350],
    }


def fetch_clean_library_thermal_tasks(
    scheduler_url: str,
    submission_state: Mapping[str, Any],
    *,
    task_reader: TaskReader | None = None,
) -> dict[int, dict[str, Any]]:
    lanes = submission_state.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != set(
        CLEAN_LIBRARY_THERMAL_TASK_IDS
    ):
        raise UpdaterError(
            "clean-library thermal submission state is incomplete"
        )
    reader = task_reader or _get_scheduler_task
    with ThreadPoolExecutor(max_workers=min(32, len(lanes))) as executor:
        futures = {
            task_id: executor.submit(reader, scheduler_url, task_id)
            for task_id in lanes
        }
        raw = {task_id: future.result() for task_id, future in futures.items()}
    return {
        task_id: _validate_clean_library_thermal_task(
            lanes[task_id], raw[task_id]
        )
        for task_id in lanes
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


def _lastmile_submission_state(
    root: Path = LASTMILE_ROOT,
) -> dict[str, Any] | None:
    resolved = root.resolve()
    plan_path = resolved / "batch_plan.json"
    receipt_path = resolved / "submission_receipt.json"
    if not plan_path.is_file() and not receipt_path.is_file():
        return None
    if not plan_path.is_file() or not receipt_path.is_file():
        raise UpdaterError("lastmile submission artifacts are incomplete")
    if (
        _file_sha256(plan_path) != LASTMILE_PLAN_FILE_SHA256
        or _file_sha256(receipt_path) != LASTMILE_RECEIPT_FILE_SHA256
    ):
        raise UpdaterError("lastmile submission artifact bytes drifted")
    plan = _read_sealed_local_json(plan_path, schema=LASTMILE_PLAN_SCHEMA)
    receipt = _read_sealed_local_json(
        receipt_path, schema=LASTMILE_RECEIPT_SCHEMA
    )
    if (
        plan.get("payload_sha256") != LASTMILE_PLAN_PAYLOAD_SHA256
        or plan.get("scheduler_priority") != LASTMILE_PRIORITY
        or plan.get("submission_ready") is not True
        or plan.get("explicit_apply_required") is not True
        or plan.get("automatic_submission_enabled") is not False
        or plan.get("production_eligible") is not False
        or receipt.get("payload_sha256") != LASTMILE_RECEIPT_PAYLOAD_SHA256
        or receipt.get("plan_payload_sha256")
        != LASTMILE_PLAN_PAYLOAD_SHA256
        or receipt.get("scheduler_url") != DEFAULT_SCHEDULER_URL
        or receipt.get("requested_candidate_count") != len(LASTMILE_TASK_SPECS)
        or receipt.get("submitted_candidate_count") != len(
            LASTMILE_TASK_SPECS
        )
        or receipt.get("complete") is not True
        or receipt.get("all_tasks_are_independent") is not True
        or receipt.get("parallel_execution_requested") is not True
        or receipt.get("scheduler_repository_modified") is not False
    ):
        raise UpdaterError("lastmile submission truth boundary drifted")
    lanes = plan.get("lanes")
    submissions = receipt.get("submissions")
    if (
        not isinstance(lanes, list)
        or not isinstance(submissions, list)
        or len(lanes) != len(LASTMILE_TASK_SPECS)
        or len(submissions) != len(LASTMILE_TASK_SPECS)
    ):
        raise UpdaterError("lastmile submission lane count drifted")
    for spec, lane, submission in zip(
        LASTMILE_TASK_SPECS, lanes, submissions, strict=True
    ):
        scheduler = lane.get("scheduler") if isinstance(lane, dict) else None
        candidate = lane.get("candidate") if isinstance(lane, dict) else None
        readback = (
            submission.get("readback")
            if isinstance(submission, dict)
            else None
        )
        evidence = (
            submission.get("submission_evidence")
            if isinstance(submission, dict)
            else None
        )
        if (
            not isinstance(scheduler, dict)
            or not isinstance(candidate, dict)
            or not isinstance(readback, dict)
            or not isinstance(evidence, dict)
            or lane.get("rank") != spec.rank
            or scheduler.get("name") != spec.task_name
            or scheduler.get("dedupe_key") != spec.dedupe_key
            or scheduler.get("priority") != LASTMILE_PRIORITY
            or scheduler.get("cpus") != LASTMILE_CPUS
            or scheduler.get("memory_mb") != LASTMILE_MEMORY_MB
            or scheduler.get("timeout_seconds") != LASTMILE_TIMEOUT_SECONDS
            or candidate.get("physical_geometry_sha256")
            != spec.physical_geometry_sha256
            or submission.get("rank") != spec.rank
            or submission.get("task_id") != spec.task_id
            or submission.get("name") != spec.task_name
            or submission.get("dedupe_key") != spec.dedupe_key
            or submission.get("physical_geometry_sha256")
            != spec.physical_geometry_sha256
            or readback.get("name") != spec.task_name
            or readback.get("dedupe_key") != spec.dedupe_key
            or readback.get("priority") != LASTMILE_PRIORITY
            or readback.get("cpus") != LASTMILE_CPUS
            or readback.get("memory_mb") != LASTMILE_MEMORY_MB
            or readback.get("timeout_seconds") != LASTMILE_TIMEOUT_SECONDS
            or evidence.get("task_id") != spec.task_id
            or evidence.get("submission_source") != "post_created"
            or evidence.get("scheduler_mutation_performed") is not True
        ):
            raise UpdaterError(
                f"lastmile submission rank{spec.rank} receipt drifted"
            )
    return {
        "plan": plan,
        "receipt": receipt,
        "plan_path": plan_path,
        "receipt_path": receipt_path,
    }


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


def _exact_n1_6_gui_fea_card(
    observed_at: str,
) -> dict[str, Any]:
    """Expose the old invalid local thermal attempt without rejecting the design."""

    snapshot = _windows_process_snapshot()
    controller = snapshot.get(EXACT_N1_6_GUI_CONTROLLER_PID)
    aedt = snapshot.get(EXACT_N1_6_GUI_AEDT_PID)
    controller_active = (
        _pid_exists(EXACT_N1_6_GUI_CONTROLLER_PID)
        and controller is not None
        and controller[1].lower() in {"python.exe", "pythonw.exe"}
    )
    aedt_active = (
        _pid_exists(EXACT_N1_6_GUI_AEDT_PID)
        and aedt is not None
        and aedt[0] == EXACT_N1_6_GUI_CONTROLLER_PID
        and aedt[1].lower() == "ansysedt.exe"
    )
    return {
        "id": EXACT_N1_6_GUI_FEA_CARD_ID,
        "title": (
            "DIAGNOSTIC THERMAL INVALID | OLD LOCAL EXACT 6/60 | "
            "interf153/150 WALL + 5000K | NOT DESIGN FAILURE"
        ),
        "detail": (
            "The old exact 6/60 symmetric local thermal attempt is invalid "
            "because Rx_main auto-pair failures left interf153/interf150 as "
            "walls and the affected zones reached the 5000 K emergency "
            "limiter. This is a thermal mesh/interface setup failure, not "
            "evidence that the physical design violates its temperature "
            "limits. The attempt is retained as diagnostic evidence only and "
            "is not scientific or production truth."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 100,
        "evidence": [
            (
                f"old local label=seed{EXACT_N1_6_GUI_SEED}/e5b4 / historical "
                f"geometry label={EXACT_N1_6_GUI_GEOMETRY_SHA256} / "
                f"source task={EXACT_N1_6_GUI_SOURCE_TASK_ID}"
            ),
            (
                "old local diagnostic status=DIAGNOSTIC THERMAL INVALID / "
                "production_eligible=false / scientific_valid=false"
            ),
            (
                "Rx_main unpaired interfaces=interf153,interf150 / "
                "missing fluid coupling=Rx_main_block_xn,Rx_main_block_yp"
            ),
            (
                "native auto-pair reset interfaces to wall / "
                "temperature limiter triggered=true / limiter=5000 K"
            ),
            (
                "classification=thermal mesh/interface setup invalid / "
                "not a design-temperature failure / no candidate rejection"
            ),
            (
                "topology=symmetric eighth / full_model=0 / "
                "round_corner=0 / N1/N2=6/60 / cw1=5.0mm / gap1=1.6mm"
            ),
            (
                f"old local controller PID{EXACT_N1_6_GUI_CONTROLLER_PID} "
                f"active={str(controller_active).lower()} / "
                f"AEDT PID{EXACT_N1_6_GUI_AEDT_PID} "
                f"active={str(aedt_active).lower()} / observation only"
            ),
        ],
    }


def _bounded_log_tail(path: Path, limit: int = MAX_RESPONSE_BYTES) -> str:
    """Read at most ``limit`` bytes from the end of a local solver log."""

    try:
        resolved = path.resolve()
        if not resolved.is_file() or resolved.is_symlink():
            return ""
        size = resolved.stat().st_size
        with resolved.open("rb") as stream:
            if size > limit:
                stream.seek(size - limit)
            payload = stream.read(limit + 1)
    except OSError:
        return ""
    return payload[-limit:].decode("utf-8", errors="replace")


def _corrected_n1_6_gui_launch_receipt(
    root: Path = EXACT_N1_6_CORRECTED_GUI_ROOT,
) -> dict[str, Any] | None:
    """Read and fail-close the immutable visible-GUI launch identity."""

    receipt_path = root.resolve() / "corrected_gui_launch_receipt.json"
    try:
        if (
            not receipt_path.is_file()
            or receipt_path.is_symlink()
            or receipt_path.stat().st_size > MAX_RESPONSE_BYTES
        ):
            return None
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    expected = {
        "schema": "mft-corrected-visible-gui-launch-receipt-v1",
        "controller_pid": EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID,
        "aedt_pid": EXACT_N1_6_CORRECTED_GUI_AEDT_PID,
        "grpc_port": EXACT_N1_6_CORRECTED_GUI_GRPC_PORT,
        "grpc_port_listen_owner_pid": EXACT_N1_6_CORRECTED_GUI_AEDT_PID,
        "canonical_geometry_sha256": (
            EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256
        ),
        "model": "symmetric_eighth_nonrounded",
        "turns_primary": 6,
        "turns_secondary_total": 60,
        "solver_core_contract": "default-four-core-cap-v1",
        "solver_core_affinity_readback": (
            EXACT_N1_6_CORRECTED_GUI_SOLVER_CORES
        ),
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        return None
    expected_project = (
        root.resolve()
        / "simulation"
        / "simulation1"
        / "simulation1.aedt"
    )
    try:
        receipt_project = Path(str(value.get("project_path") or "")).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if receipt_project != expected_project:
        return None
    return value


def _corrected_n1_6_gui_forensic_state(
    root: Path = EXACT_N1_6_CORRECTED_GUI_FORENSIC_ROOT,
) -> dict[str, Any] | None:
    """Authenticate the immutable read-only Rx-interface failure evidence."""

    manifest_path = root.resolve() / "ui_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        if (
            manifest_path.is_symlink()
            or manifest_path.stat().st_size > MAX_RESPONSE_BYTES
        ):
            raise UpdaterError(
                "corrected GUI forensic manifest is unavailable"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError(
            "corrected GUI forensic manifest is unreadable"
        ) from exc
    if not isinstance(manifest, dict):
        raise UpdaterError(
            "corrected GUI forensic manifest must be an object"
        )
    unsigned_manifest = copy.deepcopy(manifest)
    manifest_payload_sha256 = unsigned_manifest.pop("payload_sha256", None)
    if manifest_payload_sha256 != canonical_sha256(unsigned_manifest):
        raise UpdaterError(
            "corrected GUI forensic manifest seal drifted"
        )
    expected_manifest = {
        "schema": EXACT_N1_6_CORRECTED_GUI_FORENSIC_UI_SCHEMA,
        "status": "strict_invalid_rx_main_unpaired",
        "run_root": str(EXACT_N1_6_CORRECTED_GUI_ROOT),
        "coverage_passed": False,
        "scientific_valid": False,
        "unpaired_interfaces": ["interf158", "interf160"],
        "scheduler_mutated": False,
        "solver_or_gui_mutated": False,
    }
    for key, expected_value in expected_manifest.items():
        if manifest.get(key) != expected_value:
            raise UpdaterError(
                f"corrected GUI forensic manifest {key} drifted"
            )
    evidence_path = Path(str(manifest.get("evidence_path") or "")).resolve()
    expected_evidence_path = root.resolve() / "sealed_case_parser_evidence.json"
    if (
        evidence_path != expected_evidence_path
        or not evidence_path.is_file()
        or evidence_path.is_symlink()
        or evidence_path.stat().st_size > MAX_RESPONSE_BYTES
    ):
        raise UpdaterError(
            "corrected GUI forensic evidence path drifted"
        )
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError(
            "corrected GUI forensic evidence is unreadable"
        ) from exc
    if not isinstance(evidence, dict):
        raise UpdaterError(
            "corrected GUI forensic evidence must be an object"
        )
    unsigned_evidence = copy.deepcopy(evidence)
    evidence_payload_sha256 = unsigned_evidence.pop("payload_sha256", None)
    parser_output = evidence.get("parser_output")
    implication = evidence.get("strict_scientific_implication")
    run_identity = evidence.get("run_identity")
    mutation = evidence.get("mutation_attestation")
    if (
        evidence_payload_sha256 != canonical_sha256(unsigned_evidence)
        or evidence_payload_sha256
        != manifest.get("evidence_payload_sha256")
        or evidence.get("schema")
        != EXACT_N1_6_CORRECTED_GUI_FORENSIC_EVIDENCE_SCHEMA
        or not isinstance(parser_output, dict)
        or parser_output.get("schema")
        != "thermal-rx-block-interface-coverage-v1"
        or parser_output.get("passed") is not False
        or parser_output.get("missing_fluid_coupling")
        != ["Rx_main_block_xn", "Rx_main_block_yp"]
        or parser_output.get("unpaired_interfaces")
        != ["interf158", "interf160"]
        or not isinstance(implication, dict)
        or implication.get("temperature_outputs_promotion_eligible")
        is not False
        or implication.get("thermal_result_scientific_valid_expected")
        is not False
        or not isinstance(run_identity, dict)
        or run_identity.get("canonical_geometry_sha256")
        != EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256
        or run_identity.get("controller_pid")
        != EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID
        or run_identity.get("aedt_pid")
        != EXACT_N1_6_CORRECTED_GUI_AEDT_PID
        or not isinstance(mutation, dict)
        or any(
            mutation.get(key) is not False
            for key in (
                "gui_mutated",
                "jobs_cancelled_or_resubmitted",
                "scheduler_mutated",
                "solver_process_mutated",
            )
        )
    ):
        raise UpdaterError(
            "corrected GUI forensic evidence contract drifted"
        )
    return {
        "manifest_payload_sha256": manifest_payload_sha256,
        "manifest_file_sha256": _file_sha256(manifest_path),
        "evidence_payload_sha256": evidence_payload_sha256,
        "evidence_file_sha256": _file_sha256(evidence_path),
        "case_size_bytes": int(manifest["case_size_bytes"]),
        "case_identity_sha256_sample": str(
            manifest["case_identity_sha256_sample"]
        ),
        "unpaired_interfaces": list(parser_output["unpaired_interfaces"]),
        "missing_fluid_coupling": list(
            parser_output["missing_fluid_coupling"]
        ),
        "scientific_valid": False,
        "coverage_passed": False,
    }


def _corrected_n1_6_gui_solver_stage(
    root: Path = EXACT_N1_6_CORRECTED_GUI_ROOT,
) -> dict[str, Any]:
    """Classify only stages proven by bounded native-dispatch log markers."""

    stdout_tail = _bounded_log_tail(
        root.resolve() / "corrected_gui_stdout.log"
    )
    dispatch_matches = list(
        re.finditer(
            r'SOLVER_CORE_DISPATCH_JSON\s+\{[^\r\n]*'
            r'"stage"\s*:\s*"(matrix|cap|loss|thermal)"',
            stdout_tail,
        )
    )
    if not dispatch_matches:
        return {
            "stage": "modeling",
            "state": "preparing",
            "completed_stages": [],
            "thermal_dispatch_observed": False,
            "thermal_preflight_observed": False,
            "stdout_available": bool(stdout_tail),
        }
    latest = dispatch_matches[-1]
    stage = latest.group(1)
    after_dispatch = stdout_tail[latest.end() :]
    thermal_start = stdout_tail.rfind("Solving design setup ThermalSetup")
    direct_thermal_running = thermal_start >= latest.end()
    if direct_thermal_running:
        stage = "thermal"
        active_stage_tail = stdout_tail[thermal_start:]
        solve_started = True
        solve_completed = bool(
            re.search(
                r"(?:Design setup\s+)?ThermalSetup solved correctly",
                active_stage_tail,
                flags=re.IGNORECASE,
            )
        )
    else:
        active_stage_tail = after_dispatch
        solve_started = "Solving design setup" in active_stage_tail
        solve_completed = "solved correctly" in active_stage_tail
    completed_stages: list[str] = []
    for index, match in enumerate(dispatch_matches):
        stage_end = (
            dispatch_matches[index + 1].start()
            if index + 1 < len(dispatch_matches)
            else len(stdout_tail)
        )
        if "solved correctly" in stdout_tail[match.end() : stage_end]:
            completed_stages.append(match.group(1))
    if (
        direct_thermal_running
        and latest.group(1) not in completed_stages
    ):
        completed_stages.append(latest.group(1))
    if solve_completed:
        state = "completed"
    elif solve_started:
        state = "running"
    else:
        state = "dispatching"
    return {
        "stage": stage,
        "state": state,
        "completed_stages": completed_stages,
        "thermal_dispatch_observed": any(
            match.group(1) == "thermal" for match in dispatch_matches
        )
        or direct_thermal_running,
        "thermal_preflight_observed": (
            "THERMAL_RX_INTERFACE_PREFLIGHT_JSON=" in stdout_tail
        ),
        "stdout_available": True,
    }


def _corrected_n1_6_gui_fea_card(
    observed_at: str,
    corrected_canary_task: Mapping[str, Any] | None = None,
    *,
    root: Path = EXACT_N1_6_CORRECTED_GUI_ROOT,
) -> dict[str, Any]:
    """Expose the clean corrected visible GUI without pre-claiming thermal truth."""

    snapshot = _windows_process_snapshot()
    controller = snapshot.get(EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID)
    aedt = snapshot.get(EXACT_N1_6_CORRECTED_GUI_AEDT_PID)
    controller_active = (
        _pid_exists(EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID)
        and controller is not None
        and controller[1].lower() in {"python.exe", "pythonw.exe"}
    )
    aedt_active = (
        _pid_exists(EXACT_N1_6_CORRECTED_GUI_AEDT_PID)
        and aedt is not None
        and aedt[0] == EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID
        and aedt[1].lower() == "ansysedt.exe"
    )
    receipt = _corrected_n1_6_gui_launch_receipt(root)
    receipt_verified = receipt is not None
    forensic = (
        _corrected_n1_6_gui_forensic_state()
        if root.resolve() == EXACT_N1_6_CORRECTED_GUI_ROOT.resolve()
        else None
    )
    project_path = (
        root.resolve() / "simulation" / "simulation1" / "simulation1.aedt"
    )
    project_exists = project_path.is_file()
    stage = _corrected_n1_6_gui_solver_stage(root)
    if corrected_canary_task is None:
        canary_state = "submitted"
        canary_node = "pending"
        canary_job = "pending"
        canary_allocation = "pending"
    else:
        canary_state = str(corrected_canary_task["state"])
        canary_node = (
            str(corrected_canary_task["actual_node_name"]) or "pending"
        )
        canary_job = (
            str(corrected_canary_task["slurm_job_id"]) or "pending"
        )
        canary_allocation = (
            str(corrected_canary_task["allocation_id"]) or "pending"
        )
    stage_label = str(stage["stage"]).upper()
    stage_state = str(stage["state"]).upper()
    process_state = (
        "ACTIVE"
        if controller_active and aedt_active and receipt_verified
        else "IDENTITY CHECK"
    )
    progress_by_stage = {
        "modeling": 10,
        "matrix": 25,
        "cap": 40,
        "loss": 60,
        "thermal": 80,
    }
    progress = progress_by_stage.get(str(stage["stage"]), 10)
    if stage["state"] == "completed":
        progress += 5
    if forensic is not None:
        progress = 100
    title = (
        (
            "CORRECTED GUI STRICT INVALID | EXACT 6/60 | "
            "RX-MAIN interf158/160 WALL | NOT DESIGN FAILURE"
        )
        if forensic is not None
        else (
            "CORRECTED VISIBLE GUI | EXACT 6/60 | "
            f"{stage_label} NATIVE {stage_state} | 4-CORE | "
            f"{process_state} | CANARY97041 {canary_state.upper()} "
            f"{canary_node}/j{canary_job}"
        )
    )
    detail = (
        (
            "The corrected exact 6/60 local run has now produced a native "
            "Fluent case that proves both Rx_main block interfaces were reset "
            "to wall without fluid coupling. Any later temperatures from this "
            "run are diagnostic only. This is a thermal interface-generation "
            "failure, not evidence that the physical design exceeds its "
            "temperature limits; the live solver remains untouched."
        )
        if forensic is not None
        else (
            "A fresh corrected exact 6/60 symmetric, eighth, nonrounded GUI "
            f"full-chain run is active at the {stage_label} native "
            f"{str(stage['state']).lower()} stage with a four-core local "
            "solver contract. This is operational progress only. Thermal "
            "dispatch, native Rx interface preflight, limiter-free "
            "temperatures, and scientific validity remain pending unless "
            "their explicit markers are shown below. Corrected Slurm canary "
            f"task97041 is {canary_state.upper()} on {canary_node}."
        )
    )
    interface_evidence = (
        (
            "local thermal dispatch observed="
            f"{str(stage['thermal_dispatch_observed']).lower()} / "
            "Rx native case coverage=false / unpaired interfaces="
            f"{','.join(forensic['unpaired_interfaces'])} / case bytes="
            f"{forensic['case_size_bytes']} / sampled identity SHA256="
            f"{forensic['case_identity_sha256_sample']} / temperatures="
            "diagnostic-only"
        )
        if forensic is not None
        else (
            "local thermal dispatch observed="
            f"{str(stage['thermal_dispatch_observed']).lower()} / "
            "Rx native preflight marker observed="
            f"{str(stage['thermal_preflight_observed']).lower()} / "
            "native interface coverage=pending / temperatures=pending"
        )
    )
    scientific_evidence = (
        (
            "scientific_valid=false / production_eligible=false / "
            "classification=thermal interface-generation failure / "
            "candidate temperature rejection forbidden / forensic evidence "
            f"payload SHA256={forensic['evidence_payload_sha256']} / "
            "manifest payload SHA256="
            f"{forensic['manifest_payload_sha256']}"
        )
        if forensic is not None
        else (
            "scientific_valid=pending / production_eligible=false / "
            "paired native interfaces + no 4990K limiter + authenticated "
            "temperatures are required before any PASS"
        )
    )
    return {
        "id": EXACT_N1_6_CORRECTED_GUI_FEA_CARD_ID,
        "title": title,
        "detail": detail,
        "state": (
            "in_progress"
            if forensic is not None
            else "in_progress"
            if controller_active and aedt_active and receipt_verified
            else "attention"
        ),
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            (
                "corrected visible GUI identity: controller PID"
                f"{EXACT_N1_6_CORRECTED_GUI_CONTROLLER_PID} "
                f"active={str(controller_active).lower()} / AEDT PID"
                f"{EXACT_N1_6_CORRECTED_GUI_AEDT_PID} "
                f"active={str(aedt_active).lower()} / gRPC port "
                f"{EXACT_N1_6_CORRECTED_GUI_GRPC_PORT} launch-receipt "
                f"owner PID={EXACT_N1_6_CORRECTED_GUI_AEDT_PID} / "
                f"receipt_verified={str(receipt_verified).lower()}"
            ),
            (
                f"project={project_path} / exists="
                f"{str(project_exists).lower()}"
            ),
            (
                "topology=symmetric eighth / full_model=0 / round_corner=0 / "
                "N1/N2=6/60 / cw1=5.0mm / gap1=1.6mm / "
                "core_center_gap=0.65mm"
            ),
            (
                "canonical geometry SHA256="
                f"{EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256}"
            ),
            (
                f"native solver readback={stage_label} {stage_state} / "
                "completed stages="
                f"{','.join(stage['completed_stages']) or 'none'} / "
                f"local solver cores={EXACT_N1_6_CORRECTED_GUI_SOLVER_CORES}"
            ),
            interface_evidence,
            (
                "fixed cooling unchanged: dual fan 1.5m/s / "
                "core+winding TIM 2.0mm / k=0.2W/mK"
            ),
            (
                f"corrected canary task97041={canary_state.upper()} / "
                f"node={canary_node} / allocation={canary_allocation} / "
                f"Slurm job={canary_job}"
            ),
            (
                "canary resources=8CPU+65536MB / priority=100 / "
                "timeout=43200s / standalone effective cores=8"
            ),
            (
                "canary params SHA256="
                f"{EXACT_N1_6_CORRECTED_CANARY_PARAMS_SHA256} / "
                "canonical geometry SHA256="
                f"{EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256}"
            ),
            scientific_evidence,
            (
                "observation mode=read-only / Scheduler method=GET / "
                "solver process mutation=false"
            ),
        ],
    }


def _turn_graded_cap_card(
    tasks: Mapping[int, Mapping[str, Any]],
    submission_state: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    """Summarize the authenticated Tx/Rx graded-voltage campaign."""

    if set(tasks) != set(TURN_GRADED_CAP_TASK_IDS):
        raise UpdaterError("turn-graded live task set drifted")
    categories = {
        task_id: _category(str(task["state"]))
        for task_id, task in tasks.items()
    }
    counts = {
        category: sum(value == category for value in categories.values())
        for category in ("running", "queued", "succeeded", "failed")
    }
    paired_success = 0
    for candidate_index in range(25):
        pair = [
            task
            for task in tasks.values()
            if int(task["candidate_index"]) == candidate_index
        ]
        if len(pair) != 2 or {
            str(task["active_winding"]) for task in pair
        } != {"Tx", "Rx"}:
            raise UpdaterError("turn-graded live Tx/Rx pairing drifted")
        if all(_category(str(task["state"])) == "succeeded" for task in pair):
            paired_success += 1
    canary_tx = tasks[97_066]
    canary_rx = tasks[97_067]
    canary_collection = submission_state.get("canary_collection")
    if canary_collection is not None and not isinstance(
        canary_collection, dict
    ):
        raise UpdaterError("turn-graded canary collection state drifted")
    acquisition_collection = submission_state.get(
        "acquisition_collection"
    )
    if acquisition_collection is not None and not isinstance(
        acquisition_collection, dict
    ):
        raise UpdaterError(
            "turn-graded acquisition collection state drifted"
        )
    active_nodes = sorted(
        {
            str(task["actual_node_name"])
            for task_id, task in tasks.items()
            if categories[task_id] == "running"
            and str(task["actual_node_name"])
        }
    )
    terminal = counts["succeeded"] + counts["failed"]
    progress = min(95.0, 10.0 + 80.0 * terminal / len(tasks))
    cap_result_title = (
        f" | FMIN{canary_collection['f_min_Hz'] / 1000.0:.3f}k PASS"
        if canary_collection is not None
        else ""
    )
    acquisition_result_title = (
        " | BULK24 PASS24"
        if acquisition_collection is not None
        else ""
    )
    result_evidence = (
        [
            (
                "authenticated exact-pair result: "
                f"Ctx={canary_collection['C_tx_F'] * 1e9:.6f}nF / "
                f"fTx@Lm2mH={canary_collection['f_tx_Hz'] / 1000.0:.3f}kHz / "
                f"Crx={canary_collection['C_rx_F'] * 1e9:.6f}nF / "
                f"fRx@L2=0.2H={canary_collection['f_rx_Hz'] / 1000.0:.3f}kHz / "
                f"fmin={canary_collection['f_min_Hz'] / 1000.0:.3f}kHz PASS"
            ),
            (
                "canary collection payload SHA256="
                f"{canary_collection['payload_sha256']} / "
                "symmetric even-potential diagnostic=true / "
                "full series interconnect attested=false / "
                "final design pass from capacitance alone=false"
            ),
        ]
        if canary_collection is not None
        else [
            (
                "turn-graded terminal extraction=pending / "
                "15kHz scientific resonance gate=pending / "
                "production_eligible=false"
            )
        ]
    )
    acquisition_evidence = (
        [
            (
                "authenticated acquisition result: valid rows=48/48 / "
                "valid Tx/Rx pairs=24/24 / fixed-Lm2mH 15kHz PASS=24/24 / "
                f"fmin range={acquisition_collection['min_f_min_Hz'] / 1000.0:.3f}"
                f"-{acquisition_collection['max_f_min_Hz'] / 1000.0:.3f}kHz"
            ),
            (
                "best graded-cap source thermal task="
                f"{acquisition_collection['best_source_task_id']} / "
                "geometry="
                f"{acquisition_collection['best_geometry_sha256']} / "
                "acquisition collection payload SHA256="
                f"{acquisition_collection['payload_sha256']}"
            ),
        ]
        if acquisition_collection is not None
        else []
    )
    return {
        "id": TURN_GRADED_CAP_CARD_ID,
        "title": (
            "TURN-GRADED CAP | 25 GEOM/50 SOLVES | "
            f"RUN{counts['running']} QUEUE{counts['queued']} "
            f"OK{counts['succeeded']} FAIL{counts['failed']} | "
            f"6/60 Tx97066 {str(canary_tx['state']).upper()} "
            f"Rx97067 {str(canary_rx['state']).upper()}"
            f"{cap_result_title}"
            f"{acquisition_result_title}"
        ),
        "detail": (
            "Actual per-turn midpoint-voltage electrostatic Tx and Rx solves "
            "are running in parallel for the exact 6/60 GUI geometry plus "
            "24 corrected acquisition geometries. These are the capacitance "
            "truth solves required for the 15 kHz resonance gate; Scheduler "
            "lifecycle alone is not a scientific pass."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            (
                "exact 6/60 pair: task97066 Tx "
                f"{str(canary_tx['state']).upper()} "
                f"{canary_tx['actual_node_name'] or 'pending'}/"
                f"j{canary_tx['slurm_job_id'] or 'pending'} | "
                "task97067 Rx "
                f"{str(canary_rx['state']).upper()} "
                f"{canary_rx['actual_node_name'] or 'pending'}/"
                f"j{canary_rx['slurm_job_id'] or 'pending'}"
            ),
            (
                "exact geometry SHA256="
                f"{EXACT_N1_6_CORRECTED_CANARY_GEOMETRY_SHA256} / "
                "symmetric eighth / nonrounded / N1/N2=6/60 / "
                "core gap=0.65mm"
            ),
            (
                "turn grading=physical turns + midpoint voltage / "
                "Tx=6 main / Rx=37 main+23 side / polarity=+1/+1 / "
                "matrix_on=1 / loss_on=0 / thermal_on=0"
            ),
            (
                f"bulk task range=97068-97115 / paired geometries=24 / "
                f"paired terminal success={paired_success}/25 / "
                f"active nodes={len(active_nodes)} "
                f"({','.join(active_nodes) or 'none'})"
            ),
            (
                "resources per solve=8CPU+65536MB / timeout=43200s / "
                "canary priority=100 / acquisition priority=96 / "
                f"active requested total={counts['running'] * 8}CPU+"
                f"{counts['running'] * 65_536}MB"
            ),
            (
                "canary receipt payload SHA256="
                f"{submission_state['canary_receipt_payload_sha256']} / "
                "acquisition receipt payload SHA256="
                f"{submission_state['acquisition_receipt_payload_sha256']}"
            ),
            (
                "thermal jobs cancelled or modified=false / UI updater "
                "Scheduler method=GET only / scheduler repository modified=false"
            ),
            *result_evidence,
            *acquisition_evidence,
        ],
    }


def _clean_library_thermal_card(
    tasks: Mapping[int, Mapping[str, Any]],
    submission_state: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    """Show the provenance-correct 24-geometry thermal replay."""

    if set(tasks) != set(CLEAN_LIBRARY_THERMAL_TASK_IDS):
        raise UpdaterError("clean-library thermal live task set drifted")
    categories = {
        task_id: _category(str(task["state"]))
        for task_id, task in tasks.items()
    }
    counts = {
        category: sum(value == category for value in categories.values())
        for category in ("running", "queued", "succeeded", "failed")
    }
    active_nodes = sorted(
        {
            str(task["actual_node_name"])
            for task_id, task in tasks.items()
            if categories[task_id] == "running"
            and str(task["actual_node_name"])
        }
    )
    provenance = submission_state.get("runtime_provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise UpdaterError("clean-library runtime provenance state drifted")
    provenance_count = (
        int(provenance["library_hash_count"]) if provenance is not None else 0
    )
    terminal = counts["succeeded"] + counts["failed"]
    progress = min(95.0, 15.0 + 75.0 * terminal / len(tasks))
    return {
        "id": CLEAN_LIBRARY_THERMAL_CARD_ID,
        "title": (
            "CLEAN-LIB THERMAL24 | "
            f"RUN{counts['running']} QUEUE{counts['queued']} "
            f"OK{counts['succeeded']} FAIL{counts['failed']} | "
            f"PROVENANCE {provenance_count}/24 | SCI-VALID PENDING"
        ),
        "detail": (
            "The same 24 resonance-passing 6/60 geometries are being replayed "
            "with an explicitly exported, clean PyAEDT library checkout. "
            "Only terminal results that pass solver provenance, all six "
            "thermal interface/limiter fields, and finite 25-target checks "
            "may enter surrogate retraining or Pareto promotion."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": progress,
        "evidence": [
            (
                "tasks=97116-97139 / source geometries=97042-97065 / "
                "unique geometry=24 / symmetric eighth / nonrounded / "
                "N1/N2=6/60"
            ),
            (
                f"live states: running={counts['running']} / "
                f"queued={counts['queued']} / succeeded={counts['succeeded']} / "
                f"failed={counts['failed']} / active nodes={len(active_nodes)} "
                f"({','.join(active_nodes) or 'none'})"
            ),
            (
                "resources per task=8CPU+65536MB / timeout=43200s / "
                "priority=99 / max_workers_per_node=1 / "
                f"active requested total={counts['running'] * 8}CPU+"
                f"{counts['running'] * 65_536}MB"
            ),
            (
                "fixed physics: dual fan 1.5m/s / TIM and pads 2mm / "
                "k=0.2W/mK / primary 5.0T+1.6mm / rounded=false"
            ),
            (
                "solver revision="
                f"{CLEAN_LIBRARY_THERMAL_SOLVER_REVISION} / "
                "PyAEDT library revision="
                f"{CLEAN_LIBRARY_THERMAL_LIBRARY_REVISION}"
            ),
            (
                "runtime library-root/hash provenance="
                f"{provenance_count}/24 / "
                + (
                    "payload SHA256="
                    f"{provenance['payload_sha256']}"
                    if provenance is not None
                    else "sealed runtime evidence pending"
                )
            ),
            (
                "submission payload SHA256="
                f"{submission_state['submission_payload_sha256']} / "
                "plan payload SHA256="
                f"{submission_state['plan_payload_sha256']}"
            ),
            (
                "strict scientific gate: interface contract + coverage=true + "
                "unpaired=[] + limiter=false + limiter_max<4990K + "
                "thermal_result_scientific_valid=true"
            ),
            (
                "authenticated thermal rows=0 pending terminal collection / "
                "retraining admission=pending >=8 unique from >=4 source tasks / "
                "production PASS=0"
            ),
            (
                "Scheduler observation method=GET only / source tasks unchanged / "
                "scheduler repository modified=false"
            ),
        ],
    }


def _lastmile_acquisition_card(
    tasks: Mapping[int, Mapping[str, Any]],
    submission_state: Mapping[str, Any],
    observed_at: str,
) -> dict[str, Any]:
    if set(tasks) != {spec.task_id for spec in LASTMILE_TASK_SPECS}:
        raise UpdaterError("lastmile task set drifted")
    states = [str(tasks[spec.task_id]["state"]) for spec in LASTMILE_TASK_SPECS]
    running = states.count("running")
    attaching_states = {"attaching", "attached", "assigned", "launching", "starting"}
    attaching = sum(state in attaching_states for state in states)
    queued = sum(state in {"queued", "pending"} for state in states)
    succeeded = sum(_category(state) == "succeeded" for state in states)
    failed = sum(_category(state) == "failed" for state in states)
    active_nodes = sorted(
        {
            str(tasks[spec.task_id]["actual_node_name"])
            for spec in LASTMILE_TASK_SPECS
            if str(tasks[spec.task_id]["actual_node_name"])
        }
    )
    receipt = submission_state.get("receipt")
    if not isinstance(receipt, Mapping):
        raise UpdaterError("lastmile receipt is absent")
    task_evidence = [
        (
            f"task{spec.task_id}/r{spec.rank}/{spec.physical_geometry_sha256[:12]} "
            f"{str(tasks[spec.task_id]['state']).upper()} "
            f"{tasks[spec.task_id]['actual_node_name'] or 'pending'}/"
            f"a{tasks[spec.task_id]['allocation_id'] or 'none'}/"
            f"j{tasks[spec.task_id]['slurm_job_id'] or 'none'}"
        )
        for spec in LASTMILE_TASK_SPECS
    ]
    return {
        "id": LASTMILE_ACQUISITION_CARD_ID,
        "title": (
            "PRIMARY-TEMP LASTMILE P90 | 8/8 SUBMITTED | "
            f"RUN{running} ATTACH{attaching} QUEUE{queued} "
            f"OK{succeeded} FAIL{failed} | ACQUISITION-ONLY"
        ),
        "detail": (
            "Eight independent priority-90 symmetric/unrounded neighborhood "
            "FEA tasks were submitted from the authenticated sealed plan. "
            "They retain 5.0mm primary foil, 1.6mm primary interturn spacing, "
            "1.5m/s cooling and the fixed TIM contract. This manual parallel "
            "hedge overrides automatic_submission_recommended=false only for "
            "data acquisition; it cannot create a scientific or production "
            "PASS because physical air-gap/Lm validation is absent."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": min(
            95,
            35 + 5 * running + 7 * succeeded + 2 * attaching,
        ),
        "evidence": [
            (
                "tasks=97033-97040 / priority=90 / each=8CPU+65536MB / "
                "timeout=14400s / total=64CPU+512GiB / "
                f"active nodes={','.join(active_nodes) or 'pending'}"
            ),
            (
                f"sealed plan file SHA256={LASTMILE_PLAN_FILE_SHA256} / "
                f"payload={LASTMILE_PLAN_PAYLOAD_SHA256}"
            ),
            (
                f"sealed receipt file SHA256={LASTMILE_RECEIPT_FILE_SHA256} / "
                f"payload={LASTMILE_RECEIPT_PAYLOAD_SHA256} / "
                "readback identity/resources=8/8 verified"
            ),
            (
                "existing source batch task97014-97029 remained running "
                "16/16 at submission readback / source priority=95 / "
                "lastmile geometry intersection=0"
            ),
            (
                "surrogate primary robust range=102.100-102.852C / "
                "secondary<=120C 8/8 / core<=120C 4/8 / "
                "margin-preserving=1/8 / FEA correction required"
            ),
            (
                "topology=eighth symmetric / winding=unrounded / cw1=5.0mm / "
                "gap1=1.6mm / fan=1.5m/s / TIM thickness=2.0mm / "
                "TIM k=0.2W/mK"
            ),
            (
                "core_center_gap_mm=0 pre-gap acquisition / fixed-Lm2mH "
                "resonance replay=screening-only / physical gapped Lm "
                "validation pending"
            ),
            (
                "automatic_submission_recommended=false / manual explicit "
                "acquisition override=true / production_eligible=false / "
                "actual scientific PASS=false / actual production PASS=false"
            ),
            *[
                " | ".join(task_evidence[index : index + 2])
                for index in range(0, len(task_evidence), 2)
            ],
        ],
    }


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
            "W1200/L1000 NSGA-II | FINAL528 | "
            f"FRESH OK{fresh_succeeded} RUN{fresh_running} "
            f"Q{fresh_queued} FAIL{fresh_failed} | "
            f"NDS {'COMPLETE' if collector_final else 'PENDING'} | "
            f"SYM96743 {symmetric_retry_state}"
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
                    "AUTHORITATIVE FINAL528 | GLOBAL NDS COMPLETE | SEEDS "
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
            "CODEX | CONDITIONAL—PRIMARY Z CLEARANCE DECISION PENDING | "
            "580713 h_gap1=19.45mm | 40mm HARD GATE FAIL | FINAL RELEASE OFF"
        ),
        "detail": (
            "The existing DRAFT drawing used candidate #5 with primary "
            "cw1/gap1=1.13/4.6 mm; it is superseded because the reference "
            "requires 5.0/1.6 mm. A replacement 6/60-turn Full rounded "
            "model-only AEDT is open for inspection. It is conditional, not a "
            "selected final design. Its h_gap1=19.45 mm/side fails the restored 40 mm "
            "gate and still needs 0.55 mm per side under the diagnostic 20 mm case. "
            "No analysis was run in the original inspection project. An "
            "isolated matching full-rounded capacitance replica was solved: "
            "raw two-net Crx=0.635780 nF gives 14.0193 kHz at Lm=2 mH and "
            "1:10 turns, failing 15 kHz. Magnetic/thermal and symmetric "
            "nonrounded FEA remain unconfirmed; publication is disabled."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 20,
        "evidence": [
            (
                "conditional inspection model ready=true / selected=false / "
                "cw1=5.0mm / "
                "gap1=1.6mm / turns=6/60 / analysis_run=false / "
                "primary_core_axial_clearance_each=19.45mm / "
                "bbox=1181.404x999.352x750.0mm / "
                "surrogate temperatures "
                "primary=102.182C secondary=105.032C core=119.658C / "
                "actual raw-CapMatrix fixed-Lm fmin=14.0193kHz FAIL / "
                f"Crx={CORRECTED_5T_FULL_RAW_CAP_CRX_NF:.6f}nF / "
                f"limit={CORRECTED_5T_FULL_RAW_CAP_LIMIT_NF:.9f}nF / "
                "excess=14.48% / required reduction=12.65% / "
                "raw two-net CapMatrix / cap dispatch=1 / magnetic=0"
            ),
            (
                "corrected 260707 contract insulation_min_mm=40 / hard-gate "
                "audit: PHYSICAL_INSULATION_COLUMNS omitted h_gap1 / "
                "580713 h_gap1=19.45mm/side FAIL by 20.55mm/side / "
                "bddff h_gap1=8.95mm/side FAIL by 31.05mm/side / "
                "neither candidate is final-feasible under 40mm / "
                "conditional relaxation scenario only: if primary-core axial "
                "h_gap1 is explicitly relaxed to 20mm/side, 580713 remains "
                "0.55mm/side short and requires geometry correction / "
                "user clearance decision pending / drawing freeze=false"
            ),
            (
                f"replacement AEDT={CORRECTED_5T_ROUNDED_FULL_GUI_SIZE_BYTES:,}B / "
                f"SHA256 {CORRECTED_5T_ROUNDED_FULL_GUI_SHA256} / "
                f"source geometry={CORRECTED_5T_ROUNDED_FULL_GUI_GEOMETRY[:12]} / "
                f"path={CORRECTED_5T_ROUNDED_FULL_GUI_PATH}"
            ),
            (
                "rounded Full bounded conversion: wcp_len_x "
                "392.8->381.7mm to retain <=80% straight-contact readback / "
                "fan=1.5m/s / TIM and pads=2mm,k=0.2W/mK unchanged"
            ),
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
                f"drawing views exported={ROUNDED_DRAWING_VIEW_COUNT} PNG / "
                f"primary manifest SHA256 {ROUNDED_DRAWING_VIEWS_MANIFEST_SHA256} / "
                "supplemental manifest SHA256 "
                f"{ROUNDED_DRAWING_SUPPLEMENTAL_MANIFEST_SHA256}"
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
                "task96340 input superseded=true / task96340 result cannot release "
                "drawing / corrected 5T design selection and verification required / "
                "Z: publication=false / final PPTX claimed=false / "
                "final PDF claimed=false / final deliverable claimed=false / "
                "symmetric nonrounded FEA final-confirmed=false / "
                "thermal hard-gate PASS=false / canonical candidate=false / "
                "rounded model is geometry-inspection and drawing-only"
            ),
            (
                f"bounded correction contingency commit="
                f"{ROUNDED_BOUNDED_CORRECTION_COMMIT} / prepared=true / "
                "Scheduler POST0 / submitted=false / 5T eligible=false"
            ),
        ],
    }


def _compact_design_status_card(observed_at: str) -> dict[str, Any]:
    return {
        "id": COMPACT_DESIGN_STATUS_CARD_ID,
        "title": (
            "CODEX | COMPACT SEARCH | FIXED-LM/AUTH TEST78 PASS | "
            "AUDIT FIX4 PASS | FRESH512/SCOUT POST0"
        ),
        "detail": (
            "The fixed-primary-Lm=2mH resonance path and exact compact "
            "initialization/mutation are implemented and independently tested. "
            "All four independent audit findings are now fixed: the internal "
            "Runner requires a per-seed authenticated compact activation token, "
            "the fixed-Lm wrapper leaves legacy/nonreserved runs unchanged, "
            "N1=5 no longer claims an absent compact-C bridge, and downstream "
            "evidence gives the fixed-Lm authority precedence over compatibility "
            "half-Lm aliases. Malformed string quality states are also rejected "
            "instead of being coerced to true. The 78-test regression and Ruff "
            "checks pass, and the independent re-audit found no new blocker. "
            "Scheduler POST remains zero while the clean diagnostic source is "
            "prepared and authenticated. "
            "N1=6/7/8 replay exact A/B/C plus the independent height boundary; "
            "N1=5 replays A/B/height while its decoder-unreachable C band remains "
            "assigned to N1=6..8 without a near-band fallback. Final fresh512 "
            "execution remains fail-closed until a strict B7/v8 dataset is "
            "retrained and its unchanged quality gate passes. The old audit "
            "still has zero hard-feasible rows, but that is not proof that a "
            "compact design is impossible."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 55,
        "evidence": [
            (
                f"compact fixed-Lm readiness={COMPACT_SEARCH_READINESS_COMMIT} / "
                f"authenticated scope={COMPACT_SEARCH_AUTH_COMMIT} / strict-bool "
                f"hardening={COMPACT_SEARCH_HARDENING_COMMIT} / related tests "
                f"{COMPACT_SEARCH_RELATED_TEST_COUNT} passed / Ruff passed"
            ),
            (
                f"independent code audit blockers="
                f"{COMPACT_SEARCH_AUDIT_FIX_COUNT} fixed / per-seed Runner "
                "authority + legacy/nonreserved isolation + N1=5 active-strata "
                "truth + fixed-Lm precedence / re-audit blockers=0 / "
                "Scheduler POST0"
            ),
            (
                "resonance=Ltx(0.002H+Llt_phys), "
                "Lrx=Ltx*(N2/N1)^2, min(fTx,fRx)>=15kHz / "
                "surrogate k unused / physical air-gap synthesis still required"
            ),
            (
                "exact compact strata: A=W1160..1170,L975..1000 / "
                "B=W1170..1200,L960..975 / "
                "C=W1160..1170,L960..975 / H=740..750 independent"
            ),
            (
                "N1=6/7/8 exact A/B/C/H banks replayed / N1=5 exact A/B/H / "
                "compact_C global coverage delegated to N1=6..8"
            ),
            (
                "joint compact mutation every 5 generations / exact replay "
                "required / invalid_or_near_feasible_fallback=false"
            ),
            (
                "final fresh512 activation requires strict B7/v8 dataset + "
                "retrained PASS quality generation / current real activation "
                "closed / diagnostic scout code-ready and source-auth pending"
            ),
            (
                f"compact acquisition contract sealed at commit "
                f"{COMPACT_DESIGN_CONTRACT_COMMIT} / trigger_allowed=false"
            ),
            (
                f"old-generation audit slice={COMPACT_OLD_GENERATION_SLICE_COUNT} "
                f"geometries / W<={COMPACT_FOCUS_W_MAX_MM:.0f}mm OR "
                f"L<={COMPACT_FOCUS_L_MAX_MM:.0f}mm / "
                f"volume<{COMPACT_REFERENCE_VOLUME_L:.8f}L"
            ),
            (
                "old-generation compact hard-feasible="
                f"{COMPACT_OLD_GENERATION_HARD_FEASIBLE_COUNT} / "
                "proof_of_impossibility=false / old rows are audit-only"
            ),
            (
                "axis policy=no_axis_swap / original W/L/H, temperature, "
                "resonance, cooling, operating-point, and all hard constraints "
                "retained"
            ),
        ],
    }


def _read_rx_main_l5_canary_json(
    path: Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size > MAX_LOCAL_SEALED_STATE_BYTES
        or _file_sha256(path) != expected_file_sha256
    ):
        raise UpdaterError(f"Rx-main canary evidence unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UpdaterError(f"Rx-main canary evidence invalid: {path}") from exc
    if not isinstance(value, dict):
        raise UpdaterError(f"Rx-main canary evidence is not an object: {path}")
    return value


def _rx_main_l5_canary_promotion_gate_valid(
    value: Mapping[str, Any],
) -> bool:
    promotion = value.get("promotion_gate")
    if not isinstance(promotion, Mapping):
        return False
    required = promotion.get("interface_fix_canary_requires")
    return (
        promotion.get(
            "design_scientific_promotion_forbidden_for_one_iteration_canary"
        )
        is True
        and isinstance(required, Mapping)
        and required.get("contract")
        == "thermal-rx-block-interface-coverage-v1"
        and required.get("mesh_plan_contract") == "thermal-mesh-plan-v8"
        and required.get("mesh_policy")
        == "b7-rxmain-l5-shared-region-wcp-pad-symmetry-contact-clipped-v1"
        and required.get("missing_fluid_coupling") == []
        and required.get("missing_rx_main_solids") == []
        and required.get("rx_main_adjacency_passed") is True
        and required.get("unpaired_interfaces") == []
        and required.get("no_unpaired_log_marker") is True
    )


def _rx_main_l5_canary_receipt(
    root: Path,
) -> dict[str, Any]:
    resolved = root.resolve()
    receipt = _read_rx_main_l5_canary_json(
        resolved / "submission_receipt.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_RECEIPT_SHA256,
    )
    pre_submit = _read_rx_main_l5_canary_json(
        resolved / "pre_submit_receipt.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_PRE_SUBMIT_SHA256,
    )
    payload = _read_rx_main_l5_canary_json(
        resolved / "dry_run_payload.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_PAYLOAD_SHA256,
    )
    params = _read_rx_main_l5_canary_json(
        resolved / "params.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_PARAMS_SHA256,
    )
    payload_canonical_sha256 = canonical_sha256(payload)
    params_canonical_sha256 = canonical_sha256(params)
    command = payload.get("command")
    command_sha256 = (
        hashlib.sha256(command.encode("utf-8")).hexdigest()
        if isinstance(command, str)
        else None
    )
    expected_identity = {
        "task_name": RX_MAIN_L5_NATIVE_CANARY_TASK_NAME,
        "geometry_sha256": RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256,
        "solver_revision": RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION,
        "library_revision": RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION,
    }
    if (
        receipt.get("schema")
        != "mft-goal-rx-main-l5-native-earliest-canary-submission-v1"
        or receipt.get("task_id") != RX_MAIN_L5_NATIVE_CANARY_TASK_ID
        or receipt.get("submission_source") != "post_created"
        or receipt.get("scheduler_post_calls") != 1
        or receipt.get("scheduler_mutation_performed") is not True
        or receipt.get("existing_tasks_or_gui_mutated") is not False
        or receipt.get("thermal_max_iterations") != 1
        or receipt.get("fixed_cooling")
        != RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING
        or receipt.get("interface_contract")
        != "thermal-rx-block-interface-coverage-v1"
        or receipt.get("mesh_plan_contract") != "thermal-mesh-plan-v8"
        or receipt.get("mesh_policy")
        != "b7-rxmain-l5-shared-region-wcp-pad-symmetry-contact-clipped-v1"
        or receipt.get("pre_submit_receipt_file_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_PRE_SUBMIT_SHA256
        or receipt.get("payload_canonical_sha256")
        != payload_canonical_sha256
        or receipt.get("params_canonical_sha256")
        != params_canonical_sha256
        or receipt.get("command_sha256") != command_sha256
        or not _rx_main_l5_canary_promotion_gate_valid(receipt)
        or any(receipt.get(key) != expected for key, expected in expected_identity.items())
        or pre_submit.get("schema")
        != "mft-goal-rx-main-l5-native-earliest-canary-pre-submit-v1"
        or pre_submit.get("scheduler_post_calls") != 0
        or pre_submit.get("fixed_cooling")
        != RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING
        or pre_submit.get("payload_file_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_PAYLOAD_SHA256
        or pre_submit.get("params_file_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_PARAMS_SHA256
        or pre_submit.get("payload_canonical_sha256")
        != payload_canonical_sha256
        or pre_submit.get("params_canonical_sha256")
        != params_canonical_sha256
        or pre_submit.get("command_sha256") != command_sha256
        or not _rx_main_l5_canary_promotion_gate_valid(pre_submit)
        or any(
            pre_submit.get(key) != expected
            for key, expected in expected_identity.items()
        )
        or receipt.get("dedupe_key") != pre_submit.get("dedupe_key")
        or receipt.get("dedupe_key") != payload.get("dedupe_key")
        or payload.get("name") != RX_MAIN_L5_NATIVE_CANARY_TASK_NAME
        or payload.get("cpus") != RX_MAIN_L5_NATIVE_CANARY_CPUS
        or payload.get("memory_mb") != RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB
        or payload.get("timeout_seconds")
        != RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS
        or payload.get("max_workers_per_node") != 1
        or params.get("thermal_max_iterations") != 1
        or any(
            params.get(key) != expected
            for key, expected in RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING.items()
        )
    ):
        raise UpdaterError("Rx-main canary receipt truth boundary drifted")
    return {
        "receipt_file_sha256": RX_MAIN_L5_NATIVE_CANARY_RECEIPT_SHA256,
        "pre_submit_file_sha256": RX_MAIN_L5_NATIVE_CANARY_PRE_SUBMIT_SHA256,
        "task_readback_sha256": receipt.get("task_readback_sha256"),
    }


def _rx_main_l5_monitor_payload_sha256(value: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("payload_sha256", None)
    payload = (
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rx_main_l5_predecessor_terminal(
    root: Path,
) -> dict[str, Any]:
    resolved = root.resolve()
    monitor = _read_rx_main_l5_canary_json(
        resolved / "monitor_state.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SHA256,
    )
    gate = _read_rx_main_l5_canary_json(
        resolved / "terminal_gate.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_TERMINAL_GATE_SHA256,
    )
    task = monitor.get("task")
    checks = gate.get("checks")
    strict_fields = gate.get("strict_result_fields")
    if (
        monitor.get("schema") != RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SCHEMA
        or monitor.get("payload_sha256")
        != _rx_main_l5_monitor_payload_sha256(monitor)
        or monitor.get("scheduler_mutation_performed") is not False
        or not isinstance(task, Mapping)
        or task.get("id") != RX_MAIN_L5_NATIVE_CANARY_TASK_ID
        or task.get("name") != RX_MAIN_L5_NATIVE_CANARY_TASK_NAME
        or task.get("status") != "failed"
        or task.get("exit_code") != 1
        or "standalone core opt-in authentication digest mismatch"
        not in str(task.get("failure_message") or "")
        or gate.get("schema") != "mft-goal-rx-main-l5-terminal-gate-v1"
        or gate.get("payload_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_TERMINAL_GATE_PAYLOAD_SHA256
        or gate.get("payload_sha256")
        != _rx_main_l5_monitor_payload_sha256(gate)
        or gate.get("task_id") != RX_MAIN_L5_NATIVE_CANARY_TASK_ID
        or gate.get("task_status") != "failed"
        or gate.get("scheduler_mutation_performed") is not False
        or gate.get("physical_geometry_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
        or gate.get("solver_revision")
        != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
        or gate.get("library_revision")
        != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
        or gate.get("interface_fix_canary_passed") is not False
        or gate.get("scientific_design_promotion_passed") is not False
        or gate.get("thermal_result_scientific_valid") is not False
        or gate.get("one_iteration_canary_not_a_design_temperature_result")
        is not True
        or gate.get("result_json_sha256") != ""
        or gate.get("coverage") is not None
        or gate.get("preflight") is not None
        or not isinstance(checks, Mapping)
        or checks.get("terminal_status") is not True
        or checks.get("result_json_present") is not False
        or checks.get("coverage_json_present") is not False
        or not isinstance(strict_fields, Mapping)
        or any(value is not None for value in strict_fields.values())
    ):
        raise UpdaterError("Rx-main predecessor terminal truth boundary drifted")
    payload = _read_rx_main_l5_canary_json(
        resolved / "dry_run_payload.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_PAYLOAD_SHA256,
    )
    command = str(payload.get("command") or "")
    if (
        command.count(RX_MAIN_L5_NATIVE_CANARY_OLD_CORE_AUTH_SHA256) != 1
        or RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256 in command
    ):
        raise UpdaterError("Rx-main predecessor core auth identity drifted")
    return {
        "status": "failed",
        "exit_code": 1,
        "slurm_job_id": str(task.get("slurm_job_id") or ""),
        "node_name": str(task.get("actual_node_name") or ""),
        "terminal_gate_payload_sha256": gate["payload_sha256"],
    }


def _rx_main_l5_successor_receipt(
    root: Path,
    *,
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = root.resolve()
    receipt = _read_rx_main_l5_canary_json(
        resolved / "submission_receipt.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_RECEIPT_SHA256,
    )
    lineage = _read_rx_main_l5_canary_json(
        resolved / "lineage_pre_submit.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_LINEAGE_SHA256,
    )
    payload = _read_rx_main_l5_canary_json(
        resolved / "dry_run_payload.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_PAYLOAD_SHA256,
    )
    params = _read_rx_main_l5_canary_json(
        resolved / "params.json",
        expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_PARAMS_SHA256,
    )
    receipt_predecessor = receipt.get("predecessor")
    lineage_predecessor = lineage.get("predecessor")
    unchanged = lineage.get("unchanged_identity")
    checks = lineage.get("checks")
    command = str(payload.get("command") or "")
    if (
        receipt.get("schema")
        != "mft-goal-rx-main-l5-corrected-successor-submission-v1"
        or receipt.get("task_id")
        != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID
        or receipt.get("task_name")
        != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME
        or receipt.get("submission_source") != "post_created"
        or receipt.get("scheduler_post_calls") != 1
        or receipt.get("scheduler_mutation_performed") is not True
        or receipt.get("existing_tasks_or_gui_mutated") is not False
        or receipt.get("computed_core_auth_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256
        or receipt.get("lineage_pre_submit_file_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_LINEAGE_SHA256
        or receipt.get("geometry_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
        or receipt.get("solver_revision")
        != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
        or receipt.get("library_revision")
        != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
        or receipt.get("fixed_cooling")
        != RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING
        or receipt.get("thermal_max_iterations") != 1
        or not isinstance(receipt_predecessor, Mapping)
        or receipt_predecessor.get("id") != RX_MAIN_L5_NATIVE_CANARY_TASK_ID
        or receipt_predecessor.get("status") != "failed"
        or receipt_predecessor.get("exit_code") != 1
        or receipt_predecessor.get("classification")
        != "operational_pre_solver_core_auth_failure_no_scientific_result"
        or receipt_predecessor.get("terminal_gate_payload_sha256")
        != predecessor.get("terminal_gate_payload_sha256")
        or lineage.get("schema")
        != "mft-goal-rx-main-l5-corrected-successor-lineage-v1"
        or lineage.get("scheduler_post_calls") != 0
        or lineage.get("computed_core_auth_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256
        or lineage.get("geometry_sha256")
        != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
        or lineage.get("solver_revision")
        != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
        or lineage.get("library_revision")
        != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
        or lineage.get("successor_task_name")
        != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME
        or lineage_predecessor != receipt_predecessor
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
        or not isinstance(unchanged, Mapping)
        or unchanged.get("cpus") != RX_MAIN_L5_NATIVE_CANARY_CPUS
        or unchanged.get("memory_mb") != RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB
        or unchanged.get("timeout_seconds")
        != RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS
        or unchanged.get("thermal_max_iterations") != 1
        or unchanged.get("fixed_cooling")
        != RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING
        or unchanged.get("mesh_plan_contract") != "thermal-mesh-plan-v8"
        or unchanged.get("mesh_policy")
        != "b7-rxmain-l5-shared-region-wcp-pad-symmetry-contact-clipped-v1"
        or payload.get("name")
        != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME
        or payload.get("cpus") != RX_MAIN_L5_NATIVE_CANARY_CPUS
        or payload.get("memory_mb") != RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB
        or payload.get("timeout_seconds")
        != RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS
        or params.get("thermal_max_iterations") != 1
        or any(
            params.get(key) != expected
            for key, expected in RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING.items()
        )
        or command.count(RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256) != 1
        or RX_MAIN_L5_NATIVE_CANARY_OLD_CORE_AUTH_SHA256 in command
    ):
        raise UpdaterError("Rx-main successor lineage truth boundary drifted")
    return {
        "receipt_file_sha256": RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_RECEIPT_SHA256,
        "lineage_file_sha256": RX_MAIN_L5_NATIVE_CANARY_LINEAGE_SHA256,
    }


def _rx_main_l5_canary_monitor(
    root: Path,
    *,
    task_id: int,
    task_name: str,
) -> tuple[dict[str, Any] | None, str]:
    resolved = root.resolve()
    state_path = resolved / "monitor_state.json"
    process_path = resolved / "monitor_process.json"
    if not state_path.is_file() and not process_path.is_file():
        return None, "unavailable"
    try:
        if (
            not state_path.is_file()
            or not process_path.is_file()
            or state_path.is_symlink()
            or process_path.is_symlink()
            or state_path.stat().st_size > MAX_LOCAL_SEALED_STATE_BYTES
            or process_path.stat().st_size > MAX_LOCAL_SEALED_STATE_BYTES
        ):
            raise UpdaterError("Rx-main canary monitor is incomplete")
        state_value = json.loads(state_path.read_text(encoding="utf-8"))
        process_value = json.loads(process_path.read_text(encoding="utf-8"))
        if not isinstance(state_value, dict) or not isinstance(process_value, dict):
            raise UpdaterError("Rx-main canary monitor must be an object")
        task = state_value.get("task")
        if not isinstance(task, Mapping):
            raise UpdaterError("Rx-main canary monitor task is absent")
        state = str(task.get("status") or "").strip().lower()
        category = _category(state)
        node_name = str(task.get("actual_node_name") or "")
        slurm_job_id = str(task.get("slurm_job_id") or "")
        if (
            state_value.get("schema")
            != RX_MAIN_L5_NATIVE_CANARY_MONITOR_STATE_SCHEMA
            or state_value.get("payload_sha256")
            != _rx_main_l5_monitor_payload_sha256(state_value)
            or state_value.get("scheduler_mutation_performed") is not False
            or task.get("id") != task_id
            or task.get("name") != task_name
            or process_value.get("schema")
            != RX_MAIN_L5_NATIVE_CANARY_MONITOR_PROCESS_SCHEMA
            or process_value.get("payload_sha256")
            != _rx_main_l5_monitor_payload_sha256(process_value)
            or process_value.get("task_id") != task_id
            or process_value.get("allowed_http_method") != "GET"
            or process_value.get("scheduler_mutation_performed") is not False
            or process_value.get("post_cancel_retry_forbidden") is not True
            or not isinstance(state_value.get("updated_at_utc"), str)
            or (
                category == "running"
                and (not node_name or not slurm_job_id)
            )
        ):
            raise UpdaterError("Rx-main canary monitor truth boundary drifted")
        return {
            "state": state,
            "observed_at": state_value["updated_at_utc"],
            "node_name": node_name,
            "slurm_job_id": slurm_job_id,
            "exit_code": task.get("exit_code"),
            "failure_message": str(task.get("failure_message") or ""),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
        }, "authenticated"
    except (OSError, UnicodeError, json.JSONDecodeError, UpdaterError):
        return None, "invalid"


def _rx_main_l5_successor_terminal(
    root: Path,
) -> tuple[dict[str, Any] | None, str]:
    resolved = root.resolve()
    gate_path = resolved / "terminal_gate.json"
    if not gate_path.is_file():
        return None, "unavailable"
    try:
        gate = _read_rx_main_l5_canary_json(
            gate_path,
            expected_file_sha256=(
                RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TERMINAL_GATE_SHA256
            ),
        )
        checks = gate.get("checks")
        strict_fields = gate.get("strict_result_fields")
        monitor, monitor_condition = _rx_main_l5_canary_monitor(
            resolved,
            task_id=RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID,
            task_name=RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME,
        )
        if (
            gate.get("schema") != "mft-goal-rx-main-l5-terminal-gate-v1"
            or gate.get("payload_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TERMINAL_PAYLOAD_SHA256
            or gate.get("payload_sha256")
            != _rx_main_l5_monitor_payload_sha256(gate)
            or gate.get("task_id")
            != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID
            or gate.get("task_status") != "failed"
            or gate.get("scheduler_mutation_performed") is not False
            or gate.get("physical_geometry_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
            or gate.get("solver_revision")
            != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
            or gate.get("library_revision")
            != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
            or gate.get("slurm_job_id")
            != RX_MAIN_L5_NATIVE_CANARY_SHARED_JOB_ID
            or gate.get("stdout_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_STDOUT_SHA256
            or gate.get("stderr_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_STDERR_SHA256
            or gate.get("interface_fix_canary_passed") is not False
            or gate.get("scientific_design_promotion_passed") is not False
            or gate.get("thermal_result_scientific_valid") is not False
            or gate.get(
                "one_iteration_canary_not_a_design_temperature_result"
            )
            is not True
            or gate.get("result_json_sha256") != ""
            or gate.get("coverage") is not None
            or gate.get("preflight") is not None
            or not isinstance(checks, Mapping)
            or checks.get("terminal_status") is not True
            or checks.get("result_json_present") is not False
            or checks.get("coverage_json_present") is not False
            or checks.get("no_unpaired_log_marker") is not True
            or not isinstance(strict_fields, Mapping)
            or any(value is not None for value in strict_fields.values())
            or monitor_condition != "authenticated"
            or monitor is None
            or monitor.get("state") != "failed"
            or monitor.get("exit_code") != 1
            or monitor.get("node_name") != "n110"
            or monitor.get("slurm_job_id")
            != RX_MAIN_L5_NATIVE_CANARY_SHARED_JOB_ID
        ):
            raise UpdaterError(
                "Rx-main corrected successor terminal truth boundary drifted"
            )
        return {
            "state": "failed",
            "exit_code": 1,
            "node_name": monitor["node_name"],
            "slurm_job_id": monitor["slurm_job_id"],
            "observed_at": monitor["observed_at"],
            "terminal_gate_file_sha256": (
                RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TERMINAL_GATE_SHA256
            ),
            "terminal_gate_payload_sha256": gate["payload_sha256"],
            "stdout_sha256": gate["stdout_sha256"],
            "stderr_sha256": gate["stderr_sha256"],
            # These stage facts were audited against the exact stdout/stderr
            # byte identities above.  They are not inferred from Scheduler
            # failure text, which is intentionally too lossy for this use.
            "core_auth_passed": True,
            "matrix_solve_completed": True,
            "matrix_solve_seconds": 3,
            "session_result_extraction_collapsed": True,
            "thermal_started": False,
            "scientific_result_present": False,
        }, "authenticated"
    except (OSError, UnicodeError, json.JSONDecodeError, UpdaterError):
        return None, "invalid"


def _rx_main_l5_hedge_receipt(
    root: Path,
) -> tuple[dict[str, Any] | None, str]:
    resolved = root.resolve()
    receipt_path = resolved / "submission_receipt.json"
    if not receipt_path.is_file():
        return None, "unavailable"
    try:
        receipt = _read_rx_main_l5_canary_json(
            receipt_path,
            expected_file_sha256=(
                RX_MAIN_L5_NATIVE_CANARY_HEDGE_RECEIPT_SHA256
            ),
        )
        pre_submit = _read_rx_main_l5_canary_json(
            resolved / "pre_submit_receipt.json",
            expected_file_sha256=(
                RX_MAIN_L5_NATIVE_CANARY_HEDGE_PRE_SUBMIT_SHA256
            ),
        )
        post_attempt = _read_rx_main_l5_canary_json(
            resolved / "post_attempt_receipt.json",
            expected_file_sha256=(
                RX_MAIN_L5_NATIVE_CANARY_HEDGE_POST_ATTEMPT_SHA256
            ),
        )
        payload = _read_rx_main_l5_canary_json(
            resolved / "dry_run_payload.json",
            expected_file_sha256=(
                RX_MAIN_L5_NATIVE_CANARY_HEDGE_PAYLOAD_SHA256
            ),
        )
        params = _read_rx_main_l5_canary_json(
            resolved / "params.json",
            expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_PARAMS_SHA256,
        )
        command = str(payload.get("command") or "")
        if (
            receipt.get("schema")
            != "mft-goal-rx-main-l5-isolated-hedge-submission-v1"
            or receipt.get("task_id")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID
            or receipt.get("task_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_NAME
            or receipt.get("scheduler_post_calls") != 1
            or receipt.get("scheduler_mutation_performed") is not True
            or receipt.get("retry_cancel_forbidden") is not True
            or receipt.get("strict_node_placement") is not True
            or receipt.get("requested_node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or receipt.get("node_name_policy") != "strict"
            or receipt.get("exclusive_node_response") is not None
            or receipt.get("runtime_fail_close_marker_required")
            != "MFT_EXCLUSIVE_PLACEMENT_JSON"
            or receipt.get("geometry_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
            or receipt.get("solver_revision")
            != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
            or receipt.get("library_revision")
            != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
            or receipt.get("core_auth_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256
            or receipt.get("cpus") != RX_MAIN_L5_NATIVE_CANARY_CPUS
            or receipt.get("memory_mb")
            != RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB
            or receipt.get("max_workers_per_node") != 1
            or pre_submit.get("schema")
            != "mft-goal-rx-main-l5-isolated-hedge-pre-submit-v1"
            or pre_submit.get("scheduler_post_calls") != 0
            or pre_submit.get("source_task_id")
            != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID
            or pre_submit.get("expected_node")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or pre_submit.get("node_allocations") != []
            or pre_submit.get("node_tasks") != []
            or pre_submit.get("submission_policy")
            != "exactly_one_post_no_retry_no_cancel"
            or post_attempt.get("schema")
            != "mft-goal-rx-main-l5-isolated-hedge-post-attempt-v1"
            or post_attempt.get("scheduler_post_calls") != 1
            or post_attempt.get("retry_forbidden") is not True
            or payload.get("name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_NAME
            or payload.get("cpus") != RX_MAIN_L5_NATIVE_CANARY_CPUS
            or payload.get("memory_mb")
            != RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB
            or payload.get("timeout_seconds")
            != RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS
            or payload.get("node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or payload.get("node_name_policy") != "strict"
            or payload.get("exclusive_node") is not True
            or payload.get("max_workers_per_node") != 1
            or command.count("MFT_EXCLUSIVE_PLACEMENT_JSON") != 1
            or "MFT_EXCLUSIVE_PLACEMENT_FAIL" not in command
            or "exit 86" not in command
            or command.count(RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256) != 1
            or params.get("thermal_max_iterations") != 1
            or any(
                params.get(key) != expected
                for key, expected in (
                    RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING.items()
                )
            )
        ):
            raise UpdaterError("Rx-main isolated hedge receipt drifted")
        return {
            "receipt_file_sha256": (
                RX_MAIN_L5_NATIVE_CANARY_HEDGE_RECEIPT_SHA256
            ),
            "pre_submit_file_sha256": (
                RX_MAIN_L5_NATIVE_CANARY_HEDGE_PRE_SUBMIT_SHA256
            ),
            "post_attempt_file_sha256": (
                RX_MAIN_L5_NATIVE_CANARY_HEDGE_POST_ATTEMPT_SHA256
            ),
            "payload_file_sha256": (
                RX_MAIN_L5_NATIVE_CANARY_HEDGE_PAYLOAD_SHA256
            ),
            "scheduler_post_calls": 1,
            "exclusive_payload_requested": True,
            "exclusive_scheduler_readback_available": False,
            "runtime_exclusive_fail_close": True,
        }, "authenticated"
    except (OSError, UnicodeError, json.JSONDecodeError, UpdaterError):
        return None, "invalid"


def _rx_main_l5_strict_n107_receipt(
    root: Path,
) -> tuple[dict[str, Any] | None, str]:
    resolved = root.resolve()
    receipt_path = resolved / "submission_receipt.json"
    if not receipt_path.is_file():
        return None, "unavailable"
    try:
        receipt = _read_rx_main_l5_canary_json(
            receipt_path,
            expected_file_sha256=RX_MAIN_L5_STRICT_N107_RECEIPT_SHA256,
        )
        pre_submit = _read_rx_main_l5_canary_json(
            resolved / "pre_submit_receipt.json",
            expected_file_sha256=RX_MAIN_L5_STRICT_N107_PRE_SUBMIT_SHA256,
        )
        post_attempt = _read_rx_main_l5_canary_json(
            resolved / "post_attempt_receipt.json",
            expected_file_sha256=RX_MAIN_L5_STRICT_N107_POST_ATTEMPT_SHA256,
        )
        payload = _read_rx_main_l5_canary_json(
            resolved / "dry_run_payload.json",
            expected_file_sha256=RX_MAIN_L5_STRICT_N107_PAYLOAD_SHA256,
        )
        params = _read_rx_main_l5_canary_json(
            resolved / "params.json",
            expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_PARAMS_SHA256,
        )
        command = str(payload.get("command") or "")
        payload_json = payload.get("payload_json")
        identity = (
            payload_json.get("identity")
            if isinstance(payload_json, Mapping)
            else None
        )
        placement = (
            payload_json.get("placement")
            if isinstance(payload_json, Mapping)
            else None
        )
        task_local_temp = (
            payload_json.get("task_local_temp")
            if isinstance(payload_json, Mapping)
            else None
        )
        fixed_cooling = (
            payload_json.get("fixed_cooling")
            if isinstance(payload_json, Mapping)
            else None
        )
        command_sha256 = hashlib.sha256(command.encode("utf-8")).hexdigest()
        if (
            receipt.get("schema")
            != "mft-goal-rx-main-l5-strict-n107-submission-v1"
            or receipt.get("task_id") != RX_MAIN_L5_STRICT_N107_TASK_ID
            or receipt.get("task_name") != RX_MAIN_L5_STRICT_N107_TASK_NAME
            or receipt.get("scheduler_post_calls") != 1
            or receipt.get("scheduler_cancel_calls") != 0
            or receipt.get("scheduler_mutation_performed") is not True
            or receipt.get("retry_cancel_forbidden") is not True
            or receipt.get("preserved_hedge_task_id")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID
            or receipt.get("preserved_hedge_cancelled_or_modified") is not False
            or receipt.get("strict_node_placement") is not True
            or receipt.get("requested_node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or receipt.get("node_name_policy") != "strict"
            or receipt.get("account_name") != "dhj02"
            or receipt.get("cpus") != RX_MAIN_L5_NATIVE_CANARY_CPUS
            or receipt.get("memory_mb")
            != RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB
            or receipt.get("max_workers_per_node") != 1
            or receipt.get("geometry_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
            or receipt.get("solver_revision")
            != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
            or receipt.get("library_revision")
            != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
            or receipt.get("payload_canonical_sha256")
            != canonical_sha256(payload)
            or receipt.get("params_canonical_sha256")
            != canonical_sha256(params)
            or receipt.get("command_sha256") != command_sha256
            or pre_submit.get("schema")
            != "mft-goal-rx-main-l5-strict-n107-pre-submit-v1"
            or pre_submit.get("scheduler_post_calls") != 0
            or pre_submit.get("scheduler_cancel_calls") != 0
            or pre_submit.get("submission_policy")
            != "exactly_one_post_no_retry_no_cancel"
            or post_attempt.get("schema")
            != "mft-goal-rx-main-l5-strict-n107-post-attempt-v1"
            or post_attempt.get("scheduler_post_calls") != 1
            or post_attempt.get("scheduler_cancel_calls") != 0
            or post_attempt.get("retry_forbidden") is not True
            or payload.get("name") != RX_MAIN_L5_STRICT_N107_TASK_NAME
            or payload.get("node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or payload.get("node_name_policy") != "strict"
            or payload.get("strict_node_placement") is not True
            or payload.get("exclusive_node") is not False
            or payload.get("account_name") != "dhj02"
            or payload.get("cpus") != RX_MAIN_L5_NATIVE_CANARY_CPUS
            or payload.get("memory_mb") != RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB
            or payload.get("timeout_seconds")
            != RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS
            or payload.get("max_workers_per_node") != 1
            or not isinstance(identity, Mapping)
            or identity.get("geometry_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
            or identity.get("solver_revision")
            != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
            or identity.get("library_revision")
            != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
            or identity.get("core_auth_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256
            or identity.get("thermal_max_iterations") != 1
            or not isinstance(placement, Mapping)
            or placement.get("expected_node")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or placement.get("node_name_policy") != "strict"
            or placement.get("strict_node_placement") is not True
            or placement.get("exclusive_node") is not False
            or placement.get("shared_zero_required") is not False
            or placement.get("placement_fail_closed_exit_code") != 86
            or not isinstance(task_local_temp, Mapping)
            or task_local_temp.get("fail_closed_exit_code") != 87
            or task_local_temp.get("minimum_free_kb") != 20_971_520
            or not isinstance(fixed_cooling, Mapping)
            or fixed_cooling.get("fan_velocity_m_s") != 1.5
            or fixed_cooling.get("fan_config") != "dual"
            or fixed_cooling.get("modified") is not False
            or params.get("thermal_max_iterations") != 1
            or any(
                params.get(key) != expected
                for key, expected in RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING.items()
            )
            or command.count(RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256) != 1
            or "MFT_STRICT_N107_PLACEMENT_FAIL" not in command
            or "SLURM_JOB_NUM_NODES" not in command
            or "MFT_TASK_TEMP_JSON" not in command
            or "ANS_TEMP_PATH" not in command
            or "exit 86" not in command
        ):
            raise UpdaterError("Rx-main strict n107 receipt drifted")
        return {
            "receipt_file_sha256": RX_MAIN_L5_STRICT_N107_RECEIPT_SHA256,
            "pre_submit_file_sha256": RX_MAIN_L5_STRICT_N107_PRE_SUBMIT_SHA256,
            "post_attempt_file_sha256": (
                RX_MAIN_L5_STRICT_N107_POST_ATTEMPT_SHA256
            ),
            "payload_file_sha256": RX_MAIN_L5_STRICT_N107_PAYLOAD_SHA256,
            "scheduler_post_calls": 1,
            "scheduler_cancel_calls": 0,
            "exclusive_payload_requested": False,
            "task_local_temp_configured": True,
        }, "authenticated"
    except (OSError, UnicodeError, json.JSONDecodeError, UpdaterError):
        return None, "invalid"


def _rx_main_l5_strict_n107_terminal(
    root: Path,
) -> tuple[dict[str, Any] | None, str]:
    resolved = root.resolve()
    gate_path = resolved / "terminal_gate.json"
    if not gate_path.is_file():
        return None, "unavailable"
    try:
        if (
            _file_sha256(resolved / "monitor_state.json")
            != RX_MAIN_L5_STRICT_N107_MONITOR_STATE_SHA256
            or _file_sha256(resolved / "monitor_process.json")
            != RX_MAIN_L5_STRICT_N107_MONITOR_PROCESS_SHA256
        ):
            raise UpdaterError("Rx-main strict n107 monitor files drifted")
        monitor, monitor_condition = _rx_main_l5_canary_monitor(
            resolved,
            task_id=RX_MAIN_L5_STRICT_N107_TASK_ID,
            task_name=RX_MAIN_L5_STRICT_N107_TASK_NAME,
        )
        gate = _read_rx_main_l5_canary_json(
            gate_path,
            expected_file_sha256=RX_MAIN_L5_STRICT_N107_TERMINAL_GATE_SHA256,
        )
        checks = gate.get("checks")
        strict_fields = gate.get("strict_result_fields")
        if (
            monitor is None
            or monitor_condition != "authenticated"
            or monitor.get("state") != "failed"
            or monitor.get("exit_code") != 86
            or monitor.get("node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or monitor.get("slurm_job_id") != RX_MAIN_L5_STRICT_N107_JOB_ID
            or "exit code 86" not in monitor.get("failure_message", "")
            or gate.get("schema") != "mft-goal-rx-main-l5-terminal-gate-v1"
            or gate.get("payload_sha256")
            != RX_MAIN_L5_STRICT_N107_TERMINAL_PAYLOAD_SHA256
            or gate.get("payload_sha256")
            != _rx_main_l5_monitor_payload_sha256(gate)
            or gate.get("task_id") != RX_MAIN_L5_STRICT_N107_TASK_ID
            or gate.get("task_status") != "failed"
            or gate.get("slurm_job_id") != RX_MAIN_L5_STRICT_N107_JOB_ID
            or gate.get("scheduler_mutation_performed") is not False
            or gate.get("physical_geometry_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
            or gate.get("solver_revision")
            != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
            or gate.get("library_revision")
            != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
            or gate.get("stdout_sha256")
            != RX_MAIN_L5_STRICT_N107_STDOUT_SHA256
            or gate.get("stderr_sha256")
            != RX_MAIN_L5_STRICT_N107_STDERR_SHA256
            or gate.get("interface_fix_canary_passed") is not False
            or gate.get("scientific_design_promotion_passed") is not False
            or gate.get("thermal_result_scientific_valid") is not False
            or gate.get("one_iteration_canary_not_a_design_temperature_result")
            is not True
            or gate.get("result_json_sha256") != ""
            or gate.get("coverage") is not None
            or gate.get("preflight") is not None
            or not isinstance(checks, Mapping)
            or checks.get("terminal_status") is not True
            or checks.get("result_json_present") is not False
            or checks.get("coverage_json_present") is not False
            or not isinstance(strict_fields, Mapping)
            or any(value is not None for value in strict_fields.values())
        ):
            raise UpdaterError("Rx-main strict n107 terminal truth drifted")
        return {
            **monitor,
            "terminal_gate_file_sha256": (
                RX_MAIN_L5_STRICT_N107_TERMINAL_GATE_SHA256
            ),
            "terminal_gate_payload_sha256": gate["payload_sha256"],
            "scientific_result_present": False,
            "solver_started": False,
            "guard_fail_closed": True,
        }, "authenticated"
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        UpdaterError,
        FileNotFoundError,
    ):
        return None, "invalid"


def _rx_main_l5_strict_n107_successor_receipt(
    root: Path,
) -> tuple[dict[str, Any] | None, str]:
    resolved = root.resolve()
    receipt_path = resolved / "submission_receipt.json"
    if not receipt_path.is_file():
        return None, "unavailable"
    try:
        receipt = _read_rx_main_l5_canary_json(
            receipt_path,
            expected_file_sha256=(
                RX_MAIN_L5_STRICT_N107_SUCCESSOR_RECEIPT_SHA256
            ),
        )
        pre_submit = _read_rx_main_l5_canary_json(
            resolved / "pre_submit_receipt.json",
            expected_file_sha256=(
                RX_MAIN_L5_STRICT_N107_SUCCESSOR_PRE_SUBMIT_SHA256
            ),
        )
        post_attempt = _read_rx_main_l5_canary_json(
            resolved / "post_attempt_receipt.json",
            expected_file_sha256=(
                RX_MAIN_L5_STRICT_N107_SUCCESSOR_POST_ATTEMPT_SHA256
            ),
        )
        payload = _read_rx_main_l5_canary_json(
            resolved / "dry_run_payload.json",
            expected_file_sha256=(
                RX_MAIN_L5_STRICT_N107_SUCCESSOR_PAYLOAD_SHA256
            ),
        )
        params = _read_rx_main_l5_canary_json(
            resolved / "params.json",
            expected_file_sha256=RX_MAIN_L5_NATIVE_CANARY_PARAMS_SHA256,
        )
        predecessor = pre_submit.get("failed_guard_predecessor")
        payload_json = payload.get("payload_json")
        identity = (
            payload_json.get("identity")
            if isinstance(payload_json, Mapping)
            else None
        )
        placement = (
            payload_json.get("placement")
            if isinstance(payload_json, Mapping)
            else None
        )
        command = str(payload.get("command") or "")
        command_sha256 = hashlib.sha256(command.encode("utf-8")).hexdigest()
        if (
            receipt.get("schema")
            != "mft-goal-rx-main-l5-strict-n107-submission-v1"
            or receipt.get("task_id")
            != RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_ID
            or receipt.get("task_name")
            != RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_NAME
            or receipt.get("scheduler_post_calls") != 1
            or receipt.get("scheduler_cancel_calls") != 0
            or receipt.get("scheduler_mutation_performed") is not True
            or receipt.get("retry_cancel_forbidden") is not True
            or receipt.get("failed_guard_predecessor_task_id")
            != RX_MAIN_L5_STRICT_N107_TASK_ID
            or receipt.get("failed_guard_predecessor_design_or_solver_failure")
            is not False
            or receipt.get("preserved_hedge_task_id")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID
            or receipt.get("preserved_hedge_cancelled_or_modified") is not False
            or receipt.get("requested_node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or receipt.get("node_name_policy") != "strict"
            or receipt.get("strict_node_placement") is not True
            or receipt.get("account_name") != "dhj02"
            or receipt.get("cpus") != RX_MAIN_L5_NATIVE_CANARY_CPUS
            or receipt.get("memory_mb")
            != RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB
            or receipt.get("max_workers_per_node") != 1
            or receipt.get("geometry_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
            or receipt.get("solver_revision")
            != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
            or receipt.get("library_revision")
            != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
            or receipt.get("payload_canonical_sha256")
            != canonical_sha256(payload)
            or receipt.get("params_canonical_sha256")
            != canonical_sha256(params)
            or receipt.get("command_sha256") != command_sha256
            or pre_submit.get("schema")
            != "mft-goal-rx-main-l5-strict-n107-pre-submit-v1"
            or pre_submit.get("scheduler_post_calls") != 0
            or pre_submit.get("scheduler_cancel_calls") != 0
            or pre_submit.get("submission_policy")
            != "exactly_one_post_no_retry_no_cancel"
            or not isinstance(predecessor, Mapping)
            or predecessor.get("id") != RX_MAIN_L5_STRICT_N107_TASK_ID
            or predecessor.get("status") != "failed"
            or predecessor.get("exit_code") != 86
            or predecessor.get("actual_node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or predecessor.get("slurm_job_id") != RX_MAIN_L5_STRICT_N107_JOB_ID
            or predecessor.get("classification")
            != "pre_solver_proc_permission_guard_implementation_failure"
            or predecessor.get("classification_passed") is not True
            or predecessor.get("design_or_solver_failure") is not False
            or post_attempt.get("schema")
            != "mft-goal-rx-main-l5-strict-n107-post-attempt-v1"
            or post_attempt.get("scheduler_post_calls") != 1
            or post_attempt.get("scheduler_cancel_calls") != 0
            or post_attempt.get("retry_forbidden") is not True
            or post_attempt.get("failed_guard_predecessor_task_id")
            != RX_MAIN_L5_STRICT_N107_TASK_ID
            or payload.get("name")
            != RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_NAME
            or payload.get("node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or payload.get("node_name_policy") != "strict"
            or payload.get("strict_node_placement") is not True
            or payload.get("exclusive_node") is not False
            or payload.get("account_name") != "dhj02"
            or payload.get("cpus") != RX_MAIN_L5_NATIVE_CANARY_CPUS
            or payload.get("memory_mb") != RX_MAIN_L5_NATIVE_CANARY_MEMORY_MB
            or payload.get("timeout_seconds")
            != RX_MAIN_L5_NATIVE_CANARY_TIMEOUT_SECONDS
            or payload.get("max_workers_per_node") != 1
            or not isinstance(identity, Mapping)
            or identity.get("geometry_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256
            or identity.get("solver_revision")
            != RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION
            or identity.get("library_revision")
            != RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION
            or identity.get("core_auth_sha256")
            != RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256
            or identity.get("thermal_max_iterations") != 1
            or not isinstance(placement, Mapping)
            or placement.get("expected_node")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or placement.get("node_name_policy") != "strict"
            or placement.get("strict_node_placement") is not True
            or placement.get("exclusive_node") is not False
            or placement.get("placement_fail_closed_exit_code") != 86
            or params.get("thermal_max_iterations") != 1
            or any(
                params.get(key) != expected
                for key, expected in RX_MAIN_L5_NATIVE_CANARY_FIXED_COOLING.items()
            )
            or command.count(RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256) != 1
            or "MFT_STRICT_N107_NONEXCLUSIVE_PLACEMENT_JSON" not in command
            or "MFT_STRICT_N107_PID_ENUM_FAIL" not in command
            or "scontrol show hostnames" not in command
            or "MFT_TASK_TEMP_JSON" not in command
            or "ANS_TEMP_PATH" not in command
            or "exit 86" not in command
        ):
            raise UpdaterError("Rx-main strict n107 successor receipt drifted")
        return {
            "receipt_file_sha256": (
                RX_MAIN_L5_STRICT_N107_SUCCESSOR_RECEIPT_SHA256
            ),
            "pre_submit_file_sha256": (
                RX_MAIN_L5_STRICT_N107_SUCCESSOR_PRE_SUBMIT_SHA256
            ),
            "post_attempt_file_sha256": (
                RX_MAIN_L5_STRICT_N107_SUCCESSOR_POST_ATTEMPT_SHA256
            ),
            "payload_file_sha256": (
                RX_MAIN_L5_STRICT_N107_SUCCESSOR_PAYLOAD_SHA256
            ),
            "scheduler_post_calls": 1,
            "scheduler_cancel_calls": 0,
            "failed_guard_predecessor_design_or_solver_failure": False,
        }, "authenticated"
    except (OSError, UnicodeError, json.JSONDecodeError, UpdaterError):
        return None, "invalid"


def _rx_main_l5_strict_n107_successor_live(
    root: Path,
) -> tuple[dict[str, Any] | None, str]:
    resolved = root.resolve()
    state_path = resolved / "strict_guard_state.json"
    process_path = resolved / "strict_guard_monitor_process.json"
    if not state_path.is_file() and not process_path.is_file():
        return None, "unavailable"
    try:
        if (
            not state_path.is_file()
            or not process_path.is_file()
            or state_path.is_symlink()
            or process_path.is_symlink()
            or state_path.stat().st_size > MAX_LOCAL_SEALED_STATE_BYTES
            or process_path.stat().st_size > MAX_LOCAL_SEALED_STATE_BYTES
            or _file_sha256(process_path)
            != RX_MAIN_L5_STRICT_N107_GUARD_PROCESS_SHA256
            or _file_sha256(resolved / "monitor_process.json")
            != RX_MAIN_L5_STRICT_N107_SUCCESSOR_MONITOR_PROCESS_SHA256
        ):
            raise UpdaterError("Rx-main strict n107 live evidence is incomplete")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        process = json.loads(process_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(process, dict):
            raise UpdaterError("Rx-main strict n107 live evidence is invalid")
        unsigned_state = copy.deepcopy(state)
        state_sha256 = unsigned_state.pop("payload_sha256", None)
        unsigned_process = copy.deepcopy(process)
        process_sha256 = unsigned_process.pop("payload_sha256", None)
        task = state.get("task")
        allocation = state.get("allocation")
        placement = state.get("placement_marker")
        temp = state.get("temp_marker")
        monitor, monitor_condition = _rx_main_l5_canary_monitor(
            resolved,
            task_id=RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_ID,
            task_name=RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_NAME,
        )
        if (
            state.get("schema") != "mft-goal-rx-main-l5-strict-guard-state-v1"
            or state_sha256 != canonical_sha256(unsigned_state)
            or state.get("scheduler_mutation_performed") is not False
            or not isinstance(state.get("updated_at_utc"), str)
            or process.get("schema")
            != "mft-goal-rx-main-l5-strict-guard-monitor-v1"
            or process_sha256 != canonical_sha256(unsigned_process)
            or process.get("task_id")
            != RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_ID
            or process.get("allowed_http_method") != "GET"
            or process.get("scheduler_mutation_performed") is not False
            or monitor is None
            or monitor_condition != "authenticated"
            or not isinstance(task, Mapping)
            or task.get("id") != RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_ID
            or task.get("name")
            != RX_MAIN_L5_STRICT_N107_SUCCESSOR_TASK_NAME
            or task.get("status") != monitor.get("state")
            or task.get("actual_node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or task.get("requested_node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or task.get("node_name_policy") != "strict"
            or task.get("strict_node_placement") is not True
            or task.get("placement_contract_satisfied") is not True
            or task.get("slurm_job_id")
            != RX_MAIN_L5_STRICT_N107_SUCCESSOR_JOB_ID
            or not task.get("allocation_id")
            or not isinstance(allocation, Mapping)
            or allocation.get("id") != task.get("allocation_id")
            or allocation.get("node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or allocation.get("slurm_job_id")
            != RX_MAIN_L5_STRICT_N107_SUCCESSOR_JOB_ID
            or allocation.get("exclusive_node") is not False
            or not isinstance(placement, Mapping)
            or placement.get("schema") != "mft-strict-n107-nonexclusive-v1"
            or placement.get("expected_node")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or placement.get("actual_node")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or placement.get("slurm_job_id")
            != RX_MAIN_L5_STRICT_N107_SUCCESSOR_JOB_ID
            or placement.get("slurm_job_num_nodes") != "1"
            or placement.get("same_user_foreign_jobs") != ""
            or placement.get("same_user_foreign_scheduler_tasks") != ""
            or placement.get("passed") is not True
            or state.get("placement_failure_marker_seen") is not False
            or not isinstance(temp, Mapping)
            or temp.get("schema") != "mft-task-local-temp-v1"
            or temp.get("passed") is not True
            or not isinstance(temp.get("free_kb"), int)
            or not isinstance(temp.get("min_free_kb"), int)
            or temp["free_kb"] < temp["min_free_kb"]
            or state.get("temp_failure_marker_seen") is not False
        ):
            raise UpdaterError("Rx-main strict n107 live truth drifted")
        return {
            **monitor,
            "observed_at": state["updated_at_utc"],
            "allocation_id": task["allocation_id"],
            "placement_contract_satisfied": True,
            "placement_guard_passed": True,
            "slurm_job_num_nodes": placement["slurm_job_num_nodes"],
            "same_user_foreign_jobs": placement["same_user_foreign_jobs"],
            "same_user_foreign_scheduler_tasks": (
                placement["same_user_foreign_scheduler_tasks"]
            ),
            "temp_guard_passed": True,
            "temp_path": temp.get("path"),
            "temp_free_kb": temp["free_kb"],
            "temp_min_free_kb": temp["min_free_kb"],
            "interface_preflight_seen": bool(
                state.get("interface_preflight_seen")
            ),
            "result_json_seen": bool(state.get("result_json_seen")),
        }, "authenticated"
    except (OSError, UnicodeError, json.JSONDecodeError, UpdaterError):
        return None, "invalid"


def _rx_main_l5_placement_lineage(
    root: Path,
) -> tuple[dict[str, Any] | None, str]:
    resolved = root.resolve()
    state_path = resolved / "placement_lineage_state.json"
    process_path = resolved / "placement_lineage_monitor_process.json"
    if not state_path.is_file() and not process_path.is_file():
        return None, "unavailable"
    try:
        if (
            not state_path.is_file()
            or not process_path.is_file()
            or state_path.is_symlink()
            or process_path.is_symlink()
            or state_path.stat().st_size > MAX_LOCAL_SEALED_STATE_BYTES
            or process_path.stat().st_size > MAX_LOCAL_SEALED_STATE_BYTES
        ):
            raise UpdaterError("Rx-main placement lineage monitor is incomplete")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        process = json.loads(process_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(process, dict):
            raise UpdaterError(
                "Rx-main placement lineage monitor must be an object"
            )
        predecessor = state.get("predecessor")
        simultaneous = state.get("simultaneous")
        task = state.get("task")
        allocation = state.get("allocation")
        task_state = str(
            task.get("status") if isinstance(task, Mapping) else ""
        ).strip().lower()
        task_active = _category(task_state) == "running"
        unsigned_state = copy.deepcopy(state)
        unsigned_state.pop("payload_sha256", None)
        state_payload_sha256 = hashlib.sha256(
            json.dumps(
                unsigned_state,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        unsigned_process = copy.deepcopy(process)
        unsigned_process.pop("payload_sha256", None)
        process_payload_sha256 = hashlib.sha256(
            json.dumps(
                unsigned_process,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if (
            state.get("schema")
            != RX_MAIN_L5_NATIVE_CANARY_PLACEMENT_STATE_SCHEMA
            or state.get("payload_sha256") != state_payload_sha256
            or state.get("scheduler_mutation_performed") is not False
            or not isinstance(state.get("updated_at_utc"), str)
            or not isinstance(predecessor, Mapping)
            or predecessor.get("id")
            != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID
            or predecessor.get("status") != "failed"
            or predecessor.get("exit_code") != 1
            or predecessor.get("actual_node_name") != "n110"
            or predecessor.get("allocation_id")
            != RX_MAIN_L5_NATIVE_CANARY_SHARED_ALLOCATION_ID
            or predecessor.get("slurm_job_id")
            != RX_MAIN_L5_NATIVE_CANARY_SHARED_JOB_ID
            or not isinstance(simultaneous, Mapping)
            or simultaneous.get("id")
            != RX_MAIN_L5_NATIVE_CANARY_SIMULTANEOUS_TASK_ID
            or simultaneous.get("status") != "failed"
            or simultaneous.get("actual_node_name") != "n110"
            or simultaneous.get("allocation_id")
            != RX_MAIN_L5_NATIVE_CANARY_SHARED_ALLOCATION_ID
            or simultaneous.get("slurm_job_id")
            != RX_MAIN_L5_NATIVE_CANARY_SHARED_JOB_ID
            or simultaneous.get("started_at")
            != predecessor.get("started_at")
            or not isinstance(task, Mapping)
            or task.get("id") != RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID
            or task.get("name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_NAME
            or task.get("requested_node_name")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
            or task.get("node_name_policy") != "strict"
            or task.get("strict_node_placement") is not True
            or (
                task_active
                and (
                    task.get("actual_node_name")
                    != RX_MAIN_L5_NATIVE_CANARY_HEDGE_NODE
                    or task.get("placement_contract_satisfied") is not True
                    or not task.get("allocation_id")
                    or not task.get("slurm_job_id")
                    or not isinstance(allocation, Mapping)
                )
            )
            or (
                not task_active
                and task_state in QUEUED_STATES
                and (
                    str(task.get("actual_node_name") or "")
                    or task.get("allocation_id") not in {None, 0}
                    or task.get("placement_contract_satisfied") is not False
                    or allocation is not None
                )
            )
            or process.get("schema")
            != RX_MAIN_L5_NATIVE_CANARY_PLACEMENT_PROCESS_SCHEMA
            or process.get("payload_sha256") != process_payload_sha256
            or process.get("allowed_http_method") != "GET"
            or process.get("scheduler_mutation_performed") is not False
            or process.get("task_id")
            != RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID
            or process.get("predecessor_id")
            != RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID
            or process.get("simultaneous_id")
            != RX_MAIN_L5_NATIVE_CANARY_SIMULTANEOUS_TASK_ID
        ):
            raise UpdaterError("Rx-main placement lineage truth drifted")
        return {
            "state": task_state,
            "observed_at": state["updated_at_utc"],
            "actual_node_name": str(task.get("actual_node_name") or ""),
            "allocation_id": task.get("allocation_id") or None,
            "slurm_job_id": str(task.get("slurm_job_id") or ""),
            "placement_contract_satisfied": bool(
                task.get("placement_contract_satisfied")
            ),
            "shared_node_name": predecessor["actual_node_name"],
            "shared_allocation_id": predecessor["allocation_id"],
            "shared_slurm_job_id": predecessor["slurm_job_id"],
            "simultaneous_task_id": simultaneous["id"],
            "simultaneous_started_at": simultaneous["started_at"],
        }, "authenticated"
    except (OSError, UnicodeError, json.JSONDecodeError, UpdaterError):
        return None, "invalid"


def _rx_main_l5_native_canary_fail_safe_card(
    observed_at: str,
    *,
    condition: str,
    predecessor_root: Path,
    successor_root: Path,
) -> dict[str, Any]:
    label = "LINEAGE MISSING" if condition == "missing" else "LINEAGE INVALID"
    return {
        "id": RX_MAIN_L5_NATIVE_CANARY_CARD_ID,
        "title": (
            "CODEX | B7/v8 RX-MAIN L5 NATIVE EARLIEST CANARY LINEAGE | "
            f"{label} | LIVE STATE UNCLAIMED | PROMOTION OFF"
        ),
        "detail": (
            "The pinned predecessor/successor evidence is absent or invalid. No "
            "Scheduler lifecycle, interface pass, scientific validity, design "
            "promotion, or production claim is inferred."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 0,
        "evidence": [
            f"lineage_authentication={condition}_fail_closed",
            "Scheduler lifecycle=unclaimed / authenticated GET monitor=false",
            "interface pass=unclaimed / scientific promotion=false",
            "design promotion=false / production promotion=false",
            f"predecessor root={predecessor_root.resolve()}",
            f"successor root={successor_root.resolve()}",
        ],
    }


def _rx_main_l5_native_canary_card(
    observed_at: str,
    *,
    root: Path | None = None,
    successor_root: Path | None = None,
) -> dict[str, Any]:
    predecessor_evidence_root = root or RX_MAIN_L5_NATIVE_CANARY_ROOT
    successor_evidence_root = (
        successor_root or RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_ROOT
    )
    try:
        predecessor_receipt = _rx_main_l5_canary_receipt(
            predecessor_evidence_root
        )
        predecessor = _rx_main_l5_predecessor_terminal(
            predecessor_evidence_root
        )
        successor = _rx_main_l5_successor_receipt(
            successor_evidence_root,
            predecessor=predecessor,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, UpdaterError):
        condition = (
            "missing"
            if (
                not (
                    predecessor_evidence_root.resolve()
                    / "submission_receipt.json"
                ).is_file()
                or not (
                    successor_evidence_root.resolve()
                    / "submission_receipt.json"
                ).is_file()
            )
            else "invalid"
        )
        return _rx_main_l5_native_canary_fail_safe_card(
            observed_at,
            condition=condition,
            predecessor_root=predecessor_evidence_root,
            successor_root=successor_evidence_root,
        )
    monitor, monitor_condition = _rx_main_l5_canary_monitor(
        successor_evidence_root,
        task_id=RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_ID,
        task_name=RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_TASK_NAME,
    )
    live_label = (
        (
            f"GET {monitor['state'].upper()} "
            f"{monitor['node_name']}/j{monitor['slurm_job_id']}"
            if monitor["state"] in RUNNING_STATES
            else f"GET {monitor['state'].upper()}"
        )
        if monitor is not None
        else (
            "MONITOR INVALID · LIVE STATE UNCLAIMED"
            if monitor_condition == "invalid"
            else "LIVE STATE UNCLAIMED"
        )
    )
    monitor_evidence = (
        "GET monitor authenticated=true / "
        f"state={monitor['state']} / node={monitor['node_name'] or 'none'} / "
        f"job={monitor['slurm_job_id'] or 'none'} / "
        f"observed_at={monitor['observed_at']}"
        if monitor is not None
        else (
            "GET monitor authenticated=false / monitor condition="
            f"{monitor_condition} / live state=unclaimed"
        )
    )
    return {
        "id": RX_MAIN_L5_NATIVE_CANARY_CARD_ID,
        "title": (
            "CODEX | B7/v8 RX-MAIN L5 LINEAGE | 97140 PRE-SOLVER FAILED | "
            f"97141 SUBMITTED 1 POST | {live_label} | PROMOTION OFF"
        ),
        "detail": (
            "Sealed GET and terminal-gate evidence classifies task 97140 as an "
            "operational pre-solver core-auth failure with no mesh, FEA, or "
            "scientific result—not a design failure. A sealed lineage receipt "
            "authenticates exactly one corrected successor POST for task 97141. "
            "Its receipt-time queued value is not shown as live state."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": 15,
        "evidence": [
            (
                "lineage authenticated=true / predecessor receipt+GET terminal "
                "gate=true / successor receipt=true"
            ),
            (
                f"task97140=FAILED exit1 {predecessor['node_name']}/"
                f"j{predecessor['slurm_job_id']} / operational pre-solver "
                "core-auth failure"
            ),
            (
                f"predecessor auth={RX_MAIN_L5_NATIVE_CANARY_OLD_CORE_AUTH_SHA256} "
                f"/ expected auth={RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256}"
            ),
            (
                "task97140 mesh generated=false / FEA invoked=false / "
                "result JSON=false / scientific result=false / design failure=false"
            ),
            (
                "task97141 corrected successor / Scheduler POST count=1 / "
                "resources=8CPU/65536MiB/43200s (12h)"
            ),
            (
                f"geometry={RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256} / "
                f"solver={RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION} / "
                f"library={RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION}"
            ),
            (
                "B7 mesh policy + v8 mesh plan unchanged / fixed cooling "
                "unchanged=true / thermal_max_iterations=1"
            ),
            (
                "interface pass requires missing_rx_main_solids=[] / "
                "missing_fluid_coupling=[] / rx_main_adjacency_passed=true"
            ),
            (
                "interface pass also requires unpaired_interfaces=[] / "
                "no_unpaired_log_marker=true"
            ),
            monitor_evidence,
            (
                "interface pass claimed=false / design scientific promotion=false / "
                "production promotion=false"
            ),
            (
                f"97140 receipt SHA256="
                f"{predecessor_receipt['receipt_file_sha256']} / "
                f"97141 receipt SHA256={successor['receipt_file_sha256']} / "
                f"lineage SHA256={successor['lineage_file_sha256']}"
            ),
        ],
    }


def _rx_main_l5_operational_chain_card(
    observed_at: str,
    *,
    root: Path | None = None,
    successor_root: Path | None = None,
    hedge_root: Path | None = None,
    strict_root: Path | None = None,
    strict_successor_root: Path | None = None,
) -> dict[str, Any]:
    predecessor_evidence_root = root or RX_MAIN_L5_NATIVE_CANARY_ROOT
    successor_evidence_root = (
        successor_root or RX_MAIN_L5_NATIVE_CANARY_SUCCESSOR_ROOT
    )
    hedge_evidence_root = hedge_root or RX_MAIN_L5_NATIVE_CANARY_HEDGE_ROOT
    strict_evidence_root = strict_root or RX_MAIN_L5_STRICT_N107_ROOT
    strict_successor_evidence_root = (
        strict_successor_root or RX_MAIN_L5_STRICT_N107_SUCCESSOR_ROOT
    )
    baseline = _rx_main_l5_native_canary_card(
        observed_at,
        root=root,
        successor_root=successor_root,
    )
    terminal, _ = _rx_main_l5_successor_terminal(successor_evidence_root)
    hedge_receipt, _ = _rx_main_l5_hedge_receipt(hedge_evidence_root)
    if terminal is None or hedge_receipt is None:
        return baseline
    try:
        _rx_main_l5_canary_receipt(predecessor_evidence_root)
        predecessor = _rx_main_l5_predecessor_terminal(
            predecessor_evidence_root
        )
        _rx_main_l5_successor_receipt(
            successor_evidence_root,
            predecessor=predecessor,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, UpdaterError):
        return baseline
    hedge_monitor, hedge_monitor_condition = _rx_main_l5_canary_monitor(
        hedge_evidence_root,
        task_id=RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_ID,
        task_name=RX_MAIN_L5_NATIVE_CANARY_HEDGE_TASK_NAME,
    )
    placement, placement_condition = _rx_main_l5_placement_lineage(
        hedge_evidence_root
    )
    strict_receipt, strict_receipt_condition = (
        _rx_main_l5_strict_n107_receipt(strict_evidence_root)
    )
    strict_terminal, strict_terminal_condition = (
        _rx_main_l5_strict_n107_terminal(strict_evidence_root)
    )
    strict_successor_receipt, strict_successor_receipt_condition = (
        _rx_main_l5_strict_n107_successor_receipt(
            strict_successor_evidence_root
        )
    )
    strict_successor_live, strict_successor_live_condition = (
        _rx_main_l5_strict_n107_successor_live(
            strict_successor_evidence_root
        )
    )
    hedge_state = (
        hedge_monitor["state"].upper()
        if hedge_monitor is not None
        else "LIVE-UNCLAIMED"
    )
    strict_terminal_label = (
        "GUARD-FALSE-NEG"
        if strict_terminal is not None
        else "LIVE-UNCLAIMED"
    )
    strict_successor_label = (
        (
            "GUARD+TEMP PASS "
            f"{strict_successor_live['state'].upper()}"
        )
        if strict_successor_live is not None
        else "LIVE-UNCLAIMED"
    )
    title = (
        "CODEX | RX L5 | 97141 MATRIX-OK/SESSION-FAIL | "
        f"97142 EXCL {hedge_state} | 97143 {strict_terminal_label} | "
        f"97144 n107 {strict_successor_label} | PROMOTION OFF"
    )
    if len(title) > 160:
        raise UpdaterError("Rx-main operational chain title exceeds UI bound")
    if hedge_monitor is not None:
        placement_label = (
            "authenticated=true / "
            f"contract_satisfied={placement['placement_contract_satisfied']} / "
            f"allocation={placement['allocation_id'] or 'none'}"
            if placement is not None
            else f"authenticated=false ({placement_condition})"
        )
        hedge_live_evidence = (
            "task97142 GET monitor authenticated=true / "
            f"state={hedge_monitor['state']} / "
            f"node={hedge_monitor['node_name'] or 'none'} / "
            f"job={hedge_monitor['slurm_job_id'] or 'none'} / "
            f"observed_at={hedge_monitor['observed_at']} / "
            f"placement lineage {placement_label}"
        )
    else:
        hedge_live_evidence = (
            "task97142 GET monitor authenticated=false / "
            f"condition={hedge_monitor_condition} / placement lineage="
            f"{placement_condition} / live state=unclaimed"
        )
    strict_terminal_evidence = (
        "task97143 receipt authenticated=true / strict n107 nonexclusive / "
        f"terminal=FAILED exit{strict_terminal['exit_code']} / "
        f"node={strict_terminal['node_name']} / "
        f"job={strict_terminal['slurm_job_id']} / solver started=false / "
        "classification=operational pre-solver guard implementation "
        "false-negative / runtime guard false-negative=true / "
        "scientific result=false / design invalidated=false"
        if strict_receipt is not None and strict_terminal is not None
        else (
            "task97143 authenticated=false / receipt condition="
            f"{strict_receipt_condition} / terminal condition="
            f"{strict_terminal_condition} / live state=unclaimed"
        )
    )
    strict_successor_evidence = (
        "task97144 receipt authenticated=true / strict n107 nonexclusive / "
        f"state={strict_successor_live['state']} / "
        f"allocation={strict_successor_live['allocation_id']} / "
        f"job={strict_successor_live['slurm_job_id']} / "
        "placement contract=true / one-node guard=true / "
        "same-user foreign jobs=empty / same-user Scheduler tasks=empty / "
        f"task-local temp PASS free_kb={strict_successor_live['temp_free_kb']} "
        f">= min_kb={strict_successor_live['temp_min_free_kb']} / "
        f"observed_at={strict_successor_live['observed_at']}"
        if (
            strict_successor_receipt is not None
            and strict_successor_live is not None
        )
        else (
            "task97144 authenticated=false / receipt condition="
            f"{strict_successor_receipt_condition} / live condition="
            f"{strict_successor_live_condition} / live state=unclaimed"
        )
    )
    return {
        "id": RX_MAIN_L5_NATIVE_CANARY_CARD_ID,
        "title": title,
        "detail": (
            "Task 97140 stopped before solver authentication. Task 97141 passed "
            "the corrected auth and completed the Matrix solve, but its shared "
            "n110 AEDT session collapsed during result extraction before "
            "thermal. Task 97143 reached strict n107 but its first runtime guard "
            "false-negatively stopped before the solver. Task 97144 preserves the "
            "same design and is now past corrected one-node placement plus "
            "task-local-temp guards on n107. Task 97142 remains the untouched "
            "exclusive hedge. No temperature or scientific promotion is claimed."
        ),
        "state": "in_progress",
        "updated_at": observed_at,
        "progress_pct": (
            65
            if strict_successor_live is not None
            and strict_successor_live["state"] in RUNNING_STATES
            else 45
            if strict_successor_live is not None
            and strict_successor_live["state"] in QUEUED_STATES
            else 85
            if strict_successor_live is not None
            and strict_successor_live["state"] in SUCCESS_STATES
            else 30
            if hedge_monitor is not None
            and hedge_monitor["state"] in QUEUED_STATES
            else 15
        ),
        "evidence": [
            (
                "lineage authenticated=true / task97140 receipt+GET terminal "
                "gate=true / task97141 corrected receipt=true / task97142 "
                "isolated receipt=true / task97143 guard-failure receipt+terminal="
                f"{str(strict_receipt is not None and strict_terminal is not None).lower()} "
                "/ task97144 corrected-guard receipt+live="
                f"{str(strict_successor_receipt is not None and strict_successor_live is not None).lower()}"
            ),
            (
                f"task97140=FAILED exit1 {predecessor['node_name']}/"
                f"j{predecessor['slurm_job_id']} / operational pre-solver "
                "core-auth failure / mesh generated=false / FEA invoked=false / "
                "scientific result=false / design failure=false / "
                f"predecessor auth={RX_MAIN_L5_NATIVE_CANARY_OLD_CORE_AUTH_SHA256} "
                f"/ expected auth={RX_MAIN_L5_NATIVE_CANARY_CORE_AUTH_SHA256}"
            ),
            (
                "task97141=FAILED exit1 n110/"
                f"allocation{RX_MAIN_L5_NATIVE_CANARY_SHARED_ALLOCATION_ID}/"
                f"j{terminal['slurm_job_id']} / corrected core auth PASS / "
                "Matrix Setup1 solved 3s / Project None + RL/SolutionData "
                "result-extraction collapse"
            ),
            (
                "task97141 shared-allocation evidence: task97121 started same "
                "second on n110/allocation14648/job840585 and reached the same "
                "post-solve result-extraction collapse"
            ),
            (
                "task97141 operational pre-thermal failure / thermal "
                "started=false / result JSON=false / coverage=false / "
                "scientific result=false / candidate design invalidated=false / "
                f"terminal gate SHA256={terminal['terminal_gate_file_sha256']}"
            ),
            (
                f"unchanged identity geometry="
                f"{RX_MAIN_L5_NATIVE_CANARY_GEOMETRY_SHA256} / solver="
                f"{RX_MAIN_L5_NATIVE_CANARY_SOLVER_REVISION} / library="
                f"{RX_MAIN_L5_NATIVE_CANARY_LIBRARY_REVISION} / B7 mesh policy + "
                "v8 mesh plan + fixed cooling unchanged=true / "
                "thermal_max_iterations=1 / resources=8CPU/65536MiB/43200s"
            ),
            (
                "task97142 receipt authenticated=true / Scheduler POST count=1 / "
                "strict n107 / payload exclusive_node=true / runtime physical-"
                "exclusive fail-close marker required / no retry or cancel"
            ),
            hedge_live_evidence,
            strict_terminal_evidence,
            strict_successor_evidence,
            (
                "interface pass claimed=false / design scientific "
                "promotion=false / production promotion=false"
            ),
            (
                "evidence directory names under the sealed MFT runtime root: "
                f"97140={predecessor_evidence_root.resolve().name} ; "
                f"97141={successor_evidence_root.resolve().name} ; "
                f"97142={hedge_evidence_root.resolve().name} ; "
                f"97143={strict_evidence_root.resolve().name} ; "
                f"97144={strict_successor_evidence_root.resolve().name}"
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


def _active_exact100_post_state(
    root: Path = DEFAULT_ACTIVE_EXACT100_ROOT,
) -> dict[str, Any]:
    """Fail closed until a sealed first-clean exact-100 POST receipt exists."""
    resolved = root.resolve()
    state: dict[str, Any] = {
        "phase": "preparing",
        "submitted_count": 0,
        "expected_count": 100,
        "receipt_path": None,
        "verification_error": None,
    }
    if not resolved.is_dir():
        state["verification_error"] = "active exact100 preparation root is absent"
        return state

    receipt_candidates = sorted(
        (
            path
            for path in resolved.rglob("*.json")
            if "receipt" in path.name.lower()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    matching_schema_seen = False
    last_error: str | None = None
    for path in receipt_candidates:
        try:
            if path.is_symlink() or path.stat().st_size > MAX_LOCAL_SEALED_STATE_BYTES:
                continue
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(receipt, dict)
                or receipt.get("schema_version")
                != ACTIVE_EXACT100_RECEIPT_SCHEMA
            ):
                continue
            matching_schema_seen = True
            unsigned = copy.deepcopy(receipt)
            observed_sha256 = unsigned.pop("sha256", None)
            tasks = receipt.get("tasks")
            task_ids = [
                row.get("task_id")
                for row in tasks
                if isinstance(row, dict)
            ] if isinstance(tasks, list) else []
            valid = bool(
                observed_sha256 == canonical_sha256(unsigned)
                and receipt.get("bundle_id") == ACTIVE_EXACT100_BUNDLE_ID
                and receipt.get("apply") is True
                and receipt.get("task_count") == 100
                and receipt.get("submitted_count") == 100
                and receipt.get("existing_count") == 0
                and receipt.get("absent_count") == 0
                and receipt.get("campaign_authorized_post_count") == 100
                and receipt.get("first_clean_run_exact100_scheduler_posts")
                is True
                and receipt.get("scheduler_post_count") == 100
                and receipt.get("scheduler_endpoint") == "POST /api/tasks"
                and isinstance(tasks, list)
                and len(tasks) == 100
                and len(task_ids) == 100
                and all(
                    isinstance(task_id, int) and not isinstance(task_id, bool)
                    for task_id in task_ids
                )
                and len(set(task_ids)) == 100
            )
            if not valid:
                last_error = (
                    f"exact100 receipt failed closed validation: {path.name}"
                )
                continue
            return {
                "phase": "submitted",
                "submitted_count": 100,
                "expected_count": 100,
                "receipt_path": str(path.resolve()),
                "receipt_payload_sha256": observed_sha256,
                "task_id_min": min(task_ids),
                "task_id_max": max(task_ids),
                "verification_error": None,
            }
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    state["verification_error"] = (
        last_error
        if matching_schema_seen or last_error
        else "sealed first-clean exact100 POST receipt not found"
    )
    return state


def _active_truth_ui_cards(
    *,
    observed_at: str,
    exact100: Mapping[str, Any],
) -> list[dict[str, Any]]:
    submitted = int(exact100.get("submitted_count") or 0)
    phase = "SUBMITTED" if submitted == 100 else "PREPARING"
    receipt_path = exact100.get("receipt_path")
    post_evidence = (
        f"sealed POST receipt={receipt_path}"
        if isinstance(receipt_path, str) and receipt_path
        else (
            "sealed first-clean exact100 POST receipt=not verified / "
            "displayed submitted count is therefore fail-closed at 0"
        )
    )
    return [
        {
            "id": "codex-active-contract-20260727",
            "title": (
                "ACTIVE HARD CONTRACT | 1200×900×750 | gap2 0.350 | "
                "plates 20T/20T"
            ),
            "detail": (
                "현재 탐색에 적용되는 권위 계약입니다. 외형 W≤1200 mm, "
                "L≤900 mm, H≤750 mm, 2차 턴간격 gap2=0.350 mm hard-fixed, "
                "코어 냉각판과 권선 콜드플레이트는 각각 정확히 20 mm입니다. "
                "온도 gate는 1차≤110°C, 2차≤130°C, 코어≤130°C입니다."
            ),
            "state": "in_progress",
            "updated_at": observed_at,
            "progress_pct": 100,
            "evidence": [
                "envelope_mm: W<=1200 / L<=900 / H<=750",
                "secondary interturn gap2=0.350mm / lower=upper=0.350 / hard-fixed",
                "temperature_C: primary<=110 / secondary<=130 / core<=130",
                "core cooling plate thickness=20.0mm exactly",
                "winding cold-plate thickness=20.0mm exactly",
                "N1/N2=6/60 / primary foil=5.0mm / primary gap=1.6mm / equal winding heights",
                "cooling air=1.5m/s / TIM and pad contract unchanged",
            ],
        },
        {
            "id": "codex-active-exact100-gap2p35",
            "title": (
                f"NSGA-II EXACT100 | {phase} | SUBMITTED "
                f"{submitted}/100 | gap2=0.350"
            ),
            "detail": (
                "100개 독립 seed 캠페인을 준비·제출하는 활성 검색 lane입니다. "
                "실제 POST를 증명하는 sealed receipt가 완전 검증되기 전에는 "
                "Scheduler 제출 수를 추론하지 않고 0으로 표시합니다."
            ),
            "state": "in_progress",
            "updated_at": observed_at,
            "progress_pct": 45 if submitted == 100 else 15,
            "evidence": [
                "active code revision=5823d475b9a6bf40fd709f4efca3f069e06bd722",
                "seeds=2607264100..2607264199 / population=320 / generations=300",
                "resources each=8CPU+65536MiB / max_workers_per_node=8 / priority=10",
                post_evidence,
                (
                    "Scheduler repository/service modification=false / "
                    "MFT search bundle remains separate"
                ),
            ],
        },
        {
            "id": "codex-active-capacitance-truth-boundary",
            "title": "CAPACITANCE | RAW 2-NET METRIC IS PROVISIONAL",
            "detail": (
                "현재 surrogate의 raw C_rx_rx_F는 모든 2차 턴을 한 전위로 묶은 "
                "2-net 추출값이라 턴간 전압·턴간 정전용량을 직접 나타내지 않습니다. "
                "따라서 raw capacitance 기반 공진 판정은 provisional이며, 상위 "
                "후보는 턴별 전압을 부여한 turn-graded electrostatic FEA로 재검증해야 합니다."
            ),
            "state": "in_progress",
            "updated_at": observed_at,
            "progress_pct": 20,
            "evidence": [
                "raw metric=C_rx_rx_F with all Rx turns tied to one equipotential net",
                "interturn delta-V=0 in raw two-net solve / not an interturn metric",
                "gap2 hard-fixed for manufacturability; raw-cap improvement is not the final validity basis",
                "required next gate=turn-graded FEA on top compact candidates",
                "final 15kHz resonance claim prohibited until graded-cap verification",
            ],
        },
        {
            "id": "codex-active-gui-thermal-16core",
            "title": (
                "GUI THERMAL 16-CORE | VERIFIED LEGACY DIAGNOSTIC | "
                "MAX 136.247°C"
            ),
            "detail": (
                "1.5 m/s 조건의 16-core GUI 열해석은 native HDF5와 solver "
                "profile로 검증됐습니다. 실제 범위는 49.999–136.247°C이고 "
                "clamp 없이 잔차를 통과했습니다. 다만 이 파일은 legacy diagnostic "
                "geometry이므로 새 1200×900×750 최종 설계의 온도 PASS나 최종 "
                "설계 주장으로 사용하지 않습니다."
            ),
            "state": "in_progress",
            "updated_at": observed_at,
            "progress_pct": 100,
            "evidence": [
                "project=simulation1_thermal_1p5ms_16core.aedt",
                "solver=16 CPU cores / fan speed=1.5m/s / cells=796277",
                "solver profile=Normal Completion / residual convergence=pass",
                "native HDF5 temperature=49.999..136.247C / clamp=false",
                "classification=legacy diagnostic geometry / final-design claim=false",
                "new active NSGA candidate thermal verification remains pending",
            ],
        },
    ]


def merge_status(
    payload: Mapping[str, Any],
    tasks: Mapping[int, Mapping[str, Any]],
    *,
    observed_at: str,
    auxiliary_tasks: Mapping[int, Mapping[str, Any]] | None = None,
    lastmile_tasks: Mapping[int, Mapping[str, Any]] | None = None,
    lastmile_submission_state: Mapping[str, Any] | None = None,
    corrected_canary_task: Mapping[str, Any] | None = None,
    turn_graded_cap_tasks: Mapping[int, Mapping[str, Any]] | None = None,
    turn_graded_cap_submission_state: Mapping[str, Any] | None = None,
    clean_library_thermal_tasks: Mapping[int, Mapping[str, Any]] | None = None,
    clean_library_thermal_submission_state: Mapping[str, Any] | None = None,
    reference_gui_root: Path | None = None,
    target_axis_collector_state_file: Path | None = None,
    postsuccess_state_file: Path | None = None,
    thermal_bridge_state_file: Path | None = None,
    local_symmetric_selection_state_file: Path | None = (
        DEFAULT_LOCAL_SYMMETRIC_SELECTION_STATE_FILE
    ),
    standard_full_continuation_state_file: Path | None = None,
    final_gate_root: Path | None = None,
    compact_active_truth_ui: bool = False,
) -> dict[str, Any]:
    if payload.get("schema_version") != STATUS_SCHEMA:
        raise UpdaterError("Codex status schema drifted")
    result = copy.deepcopy(dict(payload))
    result.pop(SYNC_KEY, None)
    automation_current = result.pop("_automation_current", None)
    if automation_current is not None:
        if not isinstance(automation_current, list):
            raise UpdaterError("automation current-card cache is invalid")
        result["current"] = automation_current
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
        _exact_n1_6_gui_fea_card(observed_at),
    )
    _upsert_priority_current_card(
        result,
        _corrected_n1_6_gui_fea_card(
            observed_at,
            corrected_canary_task=corrected_canary_task,
        ),
    )
    if (
        turn_graded_cap_tasks is not None
        and turn_graded_cap_submission_state is not None
    ):
        _upsert_priority_current_card(
            result,
            _turn_graded_cap_card(
                turn_graded_cap_tasks,
                turn_graded_cap_submission_state,
                observed_at,
            ),
        )
    else:
        _remove_current_card(result, TURN_GRADED_CAP_CARD_ID)
    if (
        clean_library_thermal_tasks is not None
        and clean_library_thermal_submission_state is not None
    ):
        _upsert_priority_current_card(
            result,
            _clean_library_thermal_card(
                clean_library_thermal_tasks,
                clean_library_thermal_submission_state,
                observed_at,
            ),
        )
    else:
        _remove_current_card(result, CLEAN_LIBRARY_THERMAL_CARD_ID)
    if lastmile_tasks is not None and lastmile_submission_state is not None:
        _upsert_priority_current_card(
            result,
            _lastmile_acquisition_card(
                lastmile_tasks,
                lastmile_submission_state,
                observed_at,
            ),
        )
    else:
        _remove_current_card(result, LASTMILE_ACQUISITION_CARD_ID)
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
    _upsert_current_card(
        result,
        _rx_main_l5_operational_chain_card(observed_at),
    )
    _upsert_current_card(
        result,
        _compact_design_status_card(observed_at),
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
    if compact_active_truth_ui:
        active_exact100 = _active_exact100_post_state()
        active_truth_cards = _active_truth_ui_cards(
            observed_at=observed_at,
            exact100=active_exact100,
        )
        exact100_submitted = int(active_exact100.get("submitted_count") or 0)
        exact100_phase = (
            "submitted" if exact100_submitted == 100 else "preparing"
        )
        result["summary"] = (
            "현재 권위 계약: W≤1200 × L≤900 × H≤750 mm, "
            "gap2=0.350 mm hard-fixed, 1차≤110°C, 2차/코어≤130°C, "
            "코어·권선 냉각판=20T 고정. "
            f"NSGA-II exact100={exact100_phase}, submitted "
            f"{exact100_submitted}/100 (sealed POST receipt 기준). "
            "raw 2-net capacitance는 interturn metric이 아니므로 provisional이며 "
            "상위 후보 turn-graded FEA가 필요합니다. "
            "16-core GUI legacy diagnostic은 HDF5 기준 49.999–136.247°C로 "
            "검증됐지만 새 최종 설계 PASS는 아닙니다."
        )
        result["_automation_current"] = result["current"]
        result["current"] = active_truth_cards
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
            "turn_graded_cap_tasks": (
                [
                    {
                        "task_id": task_id,
                        "candidate_index": task["candidate_index"],
                        "active_winding": task["active_winding"],
                        "state": task["state"],
                        "allocation_id": task["allocation_id"],
                        "slurm_job_id": task["slurm_job_id"],
                        "node_name": task["actual_node_name"],
                        "exit_code": task["exit_code"],
                    }
                    for task_id, task in sorted(
                        turn_graded_cap_tasks.items()
                    )
                ]
                if turn_graded_cap_tasks is not None
                else []
            ),
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
    compact_active_truth_ui: bool = False,
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
    lastmile_submission_state = (
        _lastmile_submission_state()
        if task_reader is None
        else None
    )
    lastmile_tasks = (
        fetch_lastmile_tasks(scheduler_url)
        if lastmile_submission_state is not None
        else None
    )
    corrected_canary_task = (
        _validate_corrected_canary_task(
            _get_scheduler_task(
                scheduler_url,
                EXACT_N1_6_CORRECTED_CANARY_TASK_ID,
            )
        )
        if task_reader is None
        else None
    )
    turn_graded_cap_submission_state = (
        _turn_graded_cap_submission_state()
        if task_reader is None
        else None
    )
    turn_graded_cap_tasks = (
        fetch_turn_graded_cap_tasks(
            scheduler_url,
            turn_graded_cap_submission_state,
        )
        if turn_graded_cap_submission_state is not None
        else None
    )
    clean_library_thermal_submission_state = (
        _clean_library_thermal_submission_state()
        if task_reader is None
        else None
    )
    clean_library_thermal_tasks = (
        fetch_clean_library_thermal_tasks(
            scheduler_url,
            clean_library_thermal_submission_state,
        )
        if clean_library_thermal_submission_state is not None
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
        lastmile_tasks=lastmile_tasks,
        lastmile_submission_state=lastmile_submission_state,
        corrected_canary_task=corrected_canary_task,
        turn_graded_cap_tasks=turn_graded_cap_tasks,
        turn_graded_cap_submission_state=(
            turn_graded_cap_submission_state
        ),
        clean_library_thermal_tasks=clean_library_thermal_tasks,
        clean_library_thermal_submission_state=(
            clean_library_thermal_submission_state
        ),
        reference_gui_root=(
            reference_gui_root or DEFAULT_REFERENCE_BASELINE_GUI_ROOT
        ),
        target_axis_collector_state_file=target_axis_collector_state_file,
        postsuccess_state_file=postsuccess_state_file,
        thermal_bridge_state_file=thermal_bridge_state_file,
        local_symmetric_selection_state_file=(local_symmetric_selection_state_file),
        standard_full_continuation_state_file=(standard_full_continuation_state_file),
        final_gate_root=final_gate_root,
        compact_active_truth_ui=compact_active_truth_ui,
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
    compact_active_truth_ui: bool = False,
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
                    compact_active_truth_ui=compact_active_truth_ui,
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
        compact_active_truth_ui=True,
    )
    if args.once:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
