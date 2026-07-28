"""FastAPI entry point for the standalone MFT campaign monitor."""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime
from ipaddress import ip_address, ip_network
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from filelock import FileLock

from .readers import (
    TARGETS,
    ArtifactService,
    SchedulerReader,
    SimulationPolicyConflict,
)
from .local_aedt_gui import (
    CandidateLaunchError,
    DuplicateLaunchError,
    LaunchCapacityError,
    LocalAedtGuiError,
    LocalAedtGuiLauncher,
    RoutedLocalAedtGuiLauncher,
    ACTION_VALUES as LOCAL_AEDT_GUI_ACTIONS,
    MODE_VALUES as LOCAL_AEDT_GUI_MODES,
)
from .deadline_local_gui_runner import (
    EXACT_LIBRARY_REVISION as DEADLINE_GUI_LIBRARY_REVISION,
    EXACT_SOLVER_BRANCH as DEADLINE_GUI_SOLVER_BRANCH,
    EXACT_SOLVER_REVISION as DEADLINE_GUI_SOLVER_REVISION,
    EXACT_SOLVER_RUNNER_SHA256 as DEADLINE_GUI_SOURCE_RUNNER_SHA256,
    EXACT_THERMAL_MODULE_SHA256 as DEADLINE_GUI_THERMAL_MODULE_SHA256,
    SOLVER_ROOT_ENV as DEADLINE_GUI_SOLVER_ROOT_ENV,
)
from .deadline_design import DEADLINE_LOCAL_GUI_RUNNER_SHA256
from .compute_campaign_status import ComputeCampaignStatusReader
from .codex_status import CodexWorkStatusReader


HERE = Path(__file__).resolve().parent
DEFAULT_REGRESSION_ROOT = HERE.parent
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", "").strip()
if not _LOCALAPPDATA:
    _LOCALAPPDATA = str(Path.home() / "AppData" / "Local")
CAMPAIGN_MUTATION_LOCK_PATH = (
    Path(_LOCALAPPDATA) / "MFT_1MW_2026" / "campaign-mutation.lock")
DEFAULT_OPERATOR_HOSTS = ("localhost", "127.0.0.1", "::1")
DEFAULT_OPERATOR_NETWORKS = ("192.168.0.0/24",)
LOGGER = logging.getLogger(__name__)
DEFAULT_DEADLINE_SOLVER_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/"
    "mft_deadline_timk3_solver_260723/worktree"
)
DEFAULT_DEADLINE_LIBRARY_ROOT = Path(
    "C:/Users/peets/slurm_scheduler_runtime/"
    "mft_deadline_timk3_library_260723/worktree"
)


def _default_local_gui_launcher(
    repo_root: Path,
) -> RoutedLocalAedtGuiLauncher:
    default = LocalAedtGuiLauncher(repo_root)
    solver_root = Path(os.environ.get(
        "MFT_DEADLINE_LOCAL_AEDT_SOLVER_ROOT",
        str(DEFAULT_DEADLINE_SOLVER_ROOT),
    )).resolve()
    library_root = Path(os.environ.get(
        "MFT_DEADLINE_LOCAL_AEDT_LIBRARY_ROOT",
        str(DEFAULT_DEADLINE_LIBRARY_ROOT),
    )).resolve()
    if not solver_root.exists() and not library_root.exists():
        return RoutedLocalAedtGuiLauncher(default, None)
    if not solver_root.exists() or not library_root.exists():
        raise RuntimeError(
            "deadline local AEDT solver/library must both be present"
        )
    runtime_root = os.environ.get(
        "MFT_DEADLINE_LOCAL_AEDT_GUI_RUNTIME", ""
    ).strip() or str(
        Path(_LOCALAPPDATA)
        / "MFT_1MW_2026"
        / "local-aedt-gui-deadline-timk3"
    )
    deadline = LocalAedtGuiLauncher(
        repo_root,
        solver_root=solver_root,
        runner_path=HERE / "deadline_local_gui_runner.py",
        expected_solver_revision=DEADLINE_GUI_SOLVER_REVISION,
        expected_solver_branch=DEADLINE_GUI_SOLVER_BRANCH,
        expected_runner_sha256=DEADLINE_LOCAL_GUI_RUNNER_SHA256,
        expected_solver_source_runner_sha256=(
            DEADLINE_GUI_SOURCE_RUNNER_SHA256
        ),
        expected_solver_thermal_module_sha256=(
            DEADLINE_GUI_THERMAL_MODULE_SHA256
        ),
        library_root=library_root,
        expected_library_revision=DEADLINE_GUI_LIBRARY_REVISION,
        extra_environment={
            DEADLINE_GUI_SOLVER_ROOT_ENV: str(solver_root),
        },
        runtime_root=runtime_root,
        max_active=int(os.environ.get(
            "MFT_DEADLINE_LOCAL_AEDT_GUI_MAX_ACTIVE", "2"
        )),
    )
    return RoutedLocalAedtGuiLauncher(default, deadline)


def _current7_progress_summary(
    payload: dict[str, Any],
    search: dict[str, Any],
    legacy: dict[str, Any],
) -> dict[str, Any]:
    """Normalize the configured current7 lane without reviving legacy truth."""

    def count(name: str) -> int | None:
        value = search.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    running = count("running_count")
    queued = count("queued_count")
    attaching = count("attaching_count")
    active_plus_queued = count("active_plus_queued")
    terminal = count("authenticated_terminal_seed_count")
    completed = count("completed_count")
    failed = count("failed_count")
    cancelled = count("cancelled_count")
    timed_out = count("timeout_count")
    refused = count("refused_terminal_count")
    search_count = count("search_count")
    pareto = count("feasible_pareto_count")
    candidate_preview_count = count("candidate_preview_count")
    candidate_preview_limit = count("candidate_preview_limit")
    near = count("near_feasible_count")
    near = 0 if near is None else near
    near_preview_count = min(near, 1)
    required = (
        running,
        queued,
        attaching,
        active_plus_queued,
        terminal,
        completed,
        failed,
        cancelled,
        timed_out,
        refused,
        search_count,
        pareto,
        candidate_preview_count,
        candidate_preview_limit,
    )
    state_counts = search.get("state_counts")
    state_counts = state_counts if isinstance(state_counts, dict) else {}
    state_counts_consistent = bool(
        state_counts
        and all(
            isinstance(name, str)
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for name, value in state_counts.items()
        )
        and search_count == sum(state_counts.values())
        and running == state_counts.get("running", 0)
        and queued == state_counts.get("queued", 0)
        and attaching == state_counts.get("attaching", 0)
        and completed == state_counts.get("completed", 0)
        and failed == state_counts.get("failed", 0)
        and cancelled == state_counts.get("cancelled", 0)
        and timed_out == state_counts.get("timeout", 0)
    )
    counters_consistent = bool(
        all(value is not None for value in required)
        and active_plus_queued == running + queued + attaching
        and completed == terminal
        and refused == 0
        and state_counts_consistent
    )
    preview_consistent = bool(
        pareto is not None
        and candidate_preview_count is not None
        and candidate_preview_limit == 256
        and candidate_preview_count == min(pareto, candidate_preview_limit)
        and search.get("candidate_preview_truncated")
        is (pareto > candidate_preview_count)
    )
    authority = payload.get("search_authority")
    authority = authority if isinstance(authority, dict) else {}
    bundle_sha = str(search.get("bundle_manifest_sha256") or "").lower()
    snapshot_sha = str(search.get("snapshot_sha256") or "").lower()
    snapshot_file_sha = str(
        search.get("snapshot_file_sha256") or ""
    ).lower()
    index_file_sha = str(search.get("index_file_sha256") or "").lower()
    constraint_identity_sha = str(
        search.get("constraint_identity_sha256") or ""
    ).lower()
    sha256 = re.compile(r"[0-9a-f]{64}").fullmatch
    temperature_targets = search.get("temperature_targets")
    temperature_targets = (
        list(temperature_targets)
        if isinstance(temperature_targets, list) else []
    )
    constraints = search.get("constraints")
    constraints = dict(constraints) if isinstance(constraints, dict) else {}
    harvest_observed_at = search.get("harvest_observed_at")
    harvest_observed_at = (
        harvest_observed_at.strip()
        if isinstance(harvest_observed_at, str) else ""
    )
    status_event_at = search.get("status_event_at")
    status_event_at = (
        status_event_at.strip() if isinstance(status_event_at, str) else ""
    )
    temperature_contract = {
        "schema_version": "mft-tier1-current7-temperature-reference-v1",
        "sha256": search.get("temperature_contract_sha256"),
        "target_count": len(temperature_targets),
        "targets": temperature_targets,
        "robust_upper_bound_C": constraints.get("T_limit_C"),
    }
    integrity_verified = bool(
        payload.get("available") is True
        and search.get("configured") is True
        and search.get("available") is True
        and search.get("integrity_verified") is True
        and search.get("healthy") is True
        and search.get("pointer_verified") is True
        and search.get("constraint_contract_verified") is True
        and search.get("authority_eligible") is True
        and search.get("freshness_ok") is True
        and bool(harvest_observed_at)
        and bool(status_event_at)
        and search.get("updated_at") == status_event_at
        and authority.get("configured") is True
        and authority.get("kind") == "current7"
        and authority.get("available") is True
        and authority.get("integrity_verified") is True
        and authority.get("legacy_role") == "archive"
        and sha256(bundle_sha) is not None
        and sha256(snapshot_sha) is not None
        and sha256(snapshot_file_sha) is not None
        and sha256(index_file_sha) is not None
        and sha256(constraint_identity_sha) is not None
        and sha256(str(search.get("hard_spec_sha256") or "").lower())
        is not None
        and sha256(str(
            search.get("hard_constraint_contract_sha256") or ""
        ).lower()) is not None
        and sha256(str(
            search.get("temperature_contract_sha256") or ""
        ).lower()) is not None
        and len(temperature_targets) == 7
        and counters_consistent
        and preview_consistent
    )
    source = {
        "role": "primary",
        "label": "corrected-current7",
        "cohort_id": search.get("cohort_id"),
        "constraint_version": search.get("constraint_version"),
        "hard_spec_sha256": search.get("hard_spec_sha256"),
        "model_manifest_sha256": bundle_sha,
        "deployment_model_manifest_sha256": bundle_sha,
        "accepted": integrity_verified,
    }
    return {
        "schema_version": 1,
        "available": integrity_verified,
        "status": payload.get("status"),
        "source_kind": payload.get("source_kind"),
        "source_endpoint": "/api/nsga2",
        "authority_kind": "current7",
        "authority_identity_sha256": bundle_sha or None,
        "authority_index_file_sha256": index_file_sha or None,
        "authority_snapshot_sha256": snapshot_file_sha or None,
        "authority_snapshot_identity_sha256": snapshot_sha or None,
        "constraint_identity_sha256": constraint_identity_sha or None,
        "integrity_verified": integrity_verified,
        "counters_consistent": counters_consistent,
        "state_counts": state_counts,
        "candidate_preview_consistent": preview_consistent,
        "freshness_required": search.get("freshness_required") is True,
        "freshness_ok": search.get("freshness_ok") is True,
        "freshness_age_seconds": search.get("freshness_age_seconds"),
        "harvest_observed_at": harvest_observed_at or None,
        "status_event_at": status_event_at or None,
        "running_count": running,
        "search_count": search_count,
        "queued_count": queued,
        "attaching_count": attaching,
        "active_plus_queued": active_plus_queued,
        "authenticated_terminal_seed_count": terminal,
        "terminal_results_verified": terminal,
        "failed_count": failed,
        "cancelled_count": cancelled,
        "timeout_count": timed_out,
        "refused_terminal_count": refused,
        "failed_terminal_results_verified": refused,
        "completed_count": completed,
        "feasible_pareto_count": pareto,
        "candidate_preview_count": candidate_preview_count,
        "candidate_preview_limit": candidate_preview_limit,
        "candidate_preview_truncated": search.get(
            "candidate_preview_truncated"
        ),
        "near_feasible_count": near,
        "near_feasible_preview_count": near_preview_count,
        "near_feasible_preview_limit": 1,
        "near_feasible_preview_truncated": near > near_preview_count,
        "candidate_count": payload.get("candidate_count"),
        "valid_candidate_count": payload.get("valid_candidate_count"),
        "constraint_version": search.get("constraint_version"),
        "constraints": constraints,
        "temperature_targets": temperature_targets,
        "temperature_constraint_contract": temperature_contract,
        "coherent_snapshot": search.get("pointer_verified") is True,
        "coherent_snapshot_verified": search.get("pointer_verified") is True,
        "coherent_snapshot_attempts": search.get("snapshot_attempts"),
        "source_count": 1,
        "supplemental_source_count": 0,
        "configured_source_count": 1,
        "rejected_source_count": 0 if integrity_verified else 1,
        "all_configured_sources_verified": integrity_verified,
        "source_aggregation": "current7_authoritative_primary",
        "sources": [source],
        # Preserve the established generation-identity fields for clients while
        # binding both to the immutable current7 bundle instead of stale legacy.
        "model_manifest_sha256": bundle_sha or None,
        "deployment_model_manifest_sha256": bundle_sha or None,
        "updated_at": search.get("updated_at") or payload.get("updated_at"),
        "legacy_archive": {
            "available": legacy.get("available") is True,
            "integrity_verified": legacy.get("integrity_verified") is True,
            "stale": legacy.get("stale") is True,
            "archive_state": legacy.get("archive_state"),
            "updated_at": legacy.get("updated_at"),
        },
        "warnings": search.get("warnings") if isinstance(
            search.get("warnings"), list
        ) else [],
        "error": None if integrity_verified else (
            "configured current7 NSGA authority failed closed"
        ),
    }


def _current7_condition_archive_progress_summary(
    payload: dict[str, Any],
    searches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize authenticated read-only condition generations.

    A stopped primary producer must not make the four final-goal condition
    views disappear.  These snapshots remain display-only: this projection
    reports their counters and integrity without granting search, FEA, AEDT,
    promotion, or production authority.
    """

    configured = [
        search for search in searches
        if isinstance(search, dict) and search.get("configured") is True
    ]

    def condition_restriction_key(
        search: dict[str, Any],
    ) -> tuple[float, float, float, float, float, str]:
        constraints = search.get("constraints")
        constraints = constraints if isinstance(constraints, dict) else {}

        def limit(name: str) -> float:
            value = constraints.get(name)
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
            str(search.get("constraint_version") or ""),
        )

    selected = (
        min(configured, key=condition_restriction_key)
        if configured else {}
    )
    authority = payload.get("search_authority")
    authority = authority if isinstance(authority, dict) else {}
    legacy = payload.get("tier1_feedback_search")
    legacy = legacy if isinstance(legacy, dict) else {}
    archive_boundary_valid = bool(
        payload.get("available") is True
        and payload.get("source_kind") in {
            "current7_condition_archive",
            "deadline_design_with_condition_archive",
        }
        and authority.get("configured") is False
        and authority.get("kind") == "condition_archive"
        and authority.get("available") is False
        and authority.get("integrity_verified") is False
        and authority.get("source") is None
        and authority.get("legacy_role") == "archive"
        and legacy.get("authority_role") == "archive"
        and legacy.get("archived") is True
    )
    near_preview = payload.get("near_feasible_preview")
    near_preview = near_preview if isinstance(near_preview, list) else []
    reported_near_preview_count = payload.get(
        "near_feasible_preview_count"
    )
    near_preview_contract_valid = bool(
        isinstance(reported_near_preview_count, int)
        and not isinstance(reported_near_preview_count, bool)
        and reported_near_preview_count >= 0
        and reported_near_preview_count == len(near_preview)
        and reported_near_preview_count <= 1
    )

    def exact_count(search: dict[str, Any], name: str) -> int | None:
        value = search.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    required_names = (
        "search_count",
        "running_count",
        "queued_count",
        "attaching_count",
        "active_plus_queued",
        "completed_count",
        "failed_count",
        "cancelled_count",
        "timeout_count",
        "authenticated_terminal_seed_count",
        "refused_terminal_count",
        "feasible_pareto_count",
        "candidate_preview_count",
        "candidate_preview_limit",
        "near_feasible_count",
    )
    counts_valid = bool(configured)
    aggregate_state_counts: dict[str, int] = {}
    totals = {name: 0 for name in required_names}
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []
    for position, search in enumerate(configured):
        row_counts = {
            name: exact_count(search, name) for name in required_names
        }
        state_counts = search.get("state_counts")
        state_counts = state_counts if isinstance(state_counts, dict) else {}
        state_counts_valid = bool(
            state_counts
            and all(
                isinstance(name, str)
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for name, value in state_counts.items()
            )
        )
        if state_counts_valid:
            for name, value in state_counts.items():
                aggregate_state_counts[name] = (
                    aggregate_state_counts.get(name, 0) + value
                )
        preview_count = row_counts["candidate_preview_count"]
        preview_limit = row_counts["candidate_preview_limit"]
        pareto_count = row_counts["feasible_pareto_count"]
        row_consistent = bool(
            all(value is not None for value in row_counts.values())
            and state_counts_valid
            and row_counts["search_count"] == sum(state_counts.values())
            and row_counts["running_count"] == state_counts.get("running", 0)
            and row_counts["queued_count"] == state_counts.get("queued", 0)
            and row_counts["attaching_count"] == state_counts.get(
                "attaching", 0
            )
            and row_counts["completed_count"] == state_counts.get(
                "completed", 0
            )
            and row_counts["failed_count"] == state_counts.get("failed", 0)
            and row_counts["cancelled_count"] == state_counts.get(
                "cancelled", 0
            )
            and row_counts["timeout_count"] == state_counts.get("timeout", 0)
            and row_counts["active_plus_queued"]
            == row_counts["running_count"]
            + row_counts["queued_count"]
            + row_counts["attaching_count"]
            and row_counts["completed_count"]
            == row_counts["authenticated_terminal_seed_count"]
            + row_counts["refused_terminal_count"]
            and preview_limit == 256
            and preview_count == min(pareto_count, preview_limit)
            and search.get("candidate_preview_truncated")
            is (pareto_count > preview_count)
        )
        counts_valid = counts_valid and row_consistent
        if all(value is not None for value in row_counts.values()):
            for name, value in row_counts.items():
                totals[name] += value
        source_warnings = search.get("warnings")
        if isinstance(source_warnings, list):
            warnings.extend(
                warning for warning in source_warnings
                if isinstance(warning, str) and warning.strip()
            )
        sources.append({
            "role": "condition",
            "label": search.get("constraint_version") or f"condition-{position}",
            "cohort_id": search.get("cohort_id"),
            "constraint_version": search.get("constraint_version"),
            "hard_spec_sha256": search.get("hard_spec_sha256"),
            "model_manifest_sha256": search.get("bundle_manifest_sha256"),
            "deployment_model_manifest_sha256": search.get(
                "bundle_manifest_sha256"
            ),
            "accepted": bool(
                search.get("available") is True
                and search.get("integrity_verified") is True
                and search.get("constraint_contract_verified") is True
                and search.get("display_only") is True
                and search.get("read_only") is True
                and search.get("authority_eligible") is False
            ),
        })

    integrity_verified = bool(
        configured
        and counts_valid
        and near_preview_contract_valid
        and archive_boundary_valid
        and len(configured) == len(searches)
        and all(source["accepted"] is True for source in sources)
    )
    near_preview_count = (
        reported_near_preview_count
        if near_preview_contract_valid else 0
    )
    return {
        "schema_version": 1,
        "available": integrity_verified,
        "status": payload.get("status"),
        "source_kind": payload.get("source_kind"),
        "source_endpoint": "/api/nsga2",
        "authority_kind": "none",
        "integrity_verified": integrity_verified,
        "counters_consistent": counts_valid,
        "candidate_preview_consistent": bool(
            counts_valid and near_preview_contract_valid
        ),
        "condition_count": len(configured),
        "state_counts": dict(sorted(aggregate_state_counts.items())),
        "freshness_required": False,
        "freshness_ok": integrity_verified,
        "running_count": totals["running_count"],
        "search_count": totals["search_count"],
        "queued_count": totals["queued_count"],
        "attaching_count": totals["attaching_count"],
        "active_plus_queued": totals["active_plus_queued"],
        "authenticated_terminal_seed_count": totals[
            "authenticated_terminal_seed_count"
        ],
        "terminal_results_verified": totals[
            "authenticated_terminal_seed_count"
        ],
        "failed_count": totals["failed_count"],
        "cancelled_count": totals["cancelled_count"],
        "timeout_count": totals["timeout_count"],
        "refused_terminal_count": totals["refused_terminal_count"],
        "failed_terminal_results_verified": totals["refused_terminal_count"],
        "completed_count": totals["completed_count"],
        "feasible_pareto_count": totals["feasible_pareto_count"],
        "candidate_preview_count": totals["candidate_preview_count"],
        "candidate_preview_limit": 256 * len(configured),
        "candidate_preview_truncated": any(
            search.get("candidate_preview_truncated") is True
            for search in configured
        ),
        "near_feasible_count": totals["near_feasible_count"],
        "near_feasible_preview_count": near_preview_count,
        "near_feasible_preview_limit": 1,
        "near_feasible_preview_truncated": (
            totals["near_feasible_count"] > near_preview_count
        ),
        "candidate_count": totals["feasible_pareto_count"],
        "valid_candidate_count": totals["feasible_pareto_count"],
        "constraint_version": selected.get("constraint_version"),
        "constraints": selected.get("constraints")
        if isinstance(selected.get("constraints"), dict) else {},
        "temperature_targets": selected.get("temperature_targets")
        if isinstance(selected.get("temperature_targets"), list) else [],
        "coherent_snapshot": integrity_verified,
        "coherent_snapshot_verified": integrity_verified,
        "coherent_snapshot_attempts": max(
            (
                search.get("snapshot_attempts", 0)
                for search in configured
                if isinstance(search.get("snapshot_attempts"), int)
                and not isinstance(search.get("snapshot_attempts"), bool)
            ),
            default=0,
        ),
        "source_count": len(configured),
        "supplemental_source_count": max(0, len(configured) - 1),
        "configured_source_count": len(searches),
        "rejected_source_count": len(searches) - sum(
            source["accepted"] is True for source in sources
        ),
        "all_configured_sources_verified": integrity_verified,
        "source_aggregation": "current7_read_only_condition_archive",
        "sources": sources,
        "model_manifest_sha256": selected.get("bundle_manifest_sha256"),
        "deployment_model_manifest_sha256": selected.get(
            "bundle_manifest_sha256"
        ),
        "updated_at": max(
            (
                search.get("updated_at")
                for search in configured
                if isinstance(search.get("updated_at"), str)
            ),
            default=payload.get("updated_at"),
        ),
        "warnings": list(dict.fromkeys(warnings)),
        "error": None if integrity_verified else (
            "read-only current7 condition archive failed integrity validation"
        ),
    }


def nsga_progress_summary(payload: Any) -> dict[str, Any]:
    """Project one read-only progress view from the canonical NSGA payload.

    The summary deliberately has no reader, cache, or artifact paths of its
    own.  ``/api/nsga2`` remains the UI authority; this projection exists only
    for deploy canaries and lightweight operational polling.
    """
    if not isinstance(payload, dict):
        return {
            "schema_version": 1,
            "available": False,
            "integrity_verified": False,
            "counters_consistent": False,
            "source_endpoint": "/api/nsga2",
            "error": "canonical NSGA payload is not an object",
        }
    search = payload.get("tier1_feedback_search")
    search = search if isinstance(search, dict) else {}
    current7 = payload.get("tier1_current7_search")
    current7 = current7 if isinstance(current7, dict) else {}
    if current7.get("configured") is True or current7.get("source"):
        return _current7_progress_summary(payload, current7, search)
    condition_searches = payload.get("tier1_current7_condition_searches")
    condition_searches = (
        [item for item in condition_searches if isinstance(item, dict)]
        if isinstance(condition_searches, list) else []
    )
    if condition_searches:
        return _current7_condition_archive_progress_summary(
            payload, condition_searches
        )
    sources = search.get("sources")
    sources = [
        dict(item) for item in sources
        if isinstance(item, dict)
    ] if isinstance(sources, list) else []

    def count(name: str) -> int | None:
        value = search.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    running = count("running_count")
    queued = count("queued_count")
    attaching = count("attaching_count")
    active_plus_queued = count("active_plus_queued")
    terminal = count("terminal_results_verified")
    failed_terminal = count("failed_terminal_results_verified")
    completed = count("completed_count")
    pareto = count("feasible_pareto_count")
    candidate_preview_count = count("candidate_preview_count")
    candidate_preview_limit = count("candidate_preview_limit")
    candidate_preview_truncated = search.get(
        "candidate_preview_truncated"
    )
    near = count("near_feasible_count")
    required_counts = (
        running, queued, attaching, active_plus_queued, terminal, completed,
        pareto, candidate_preview_count, candidate_preview_limit, near,
    )
    preview_consistent = bool(
        pareto is not None
        and candidate_preview_count is not None
        and candidate_preview_limit == 128
        and isinstance(candidate_preview_truncated, bool)
        and (
            (
                candidate_preview_truncated is True
                and pareto > candidate_preview_count
                and candidate_preview_count == candidate_preview_limit
            )
            or (
                candidate_preview_truncated is False
                and pareto == candidate_preview_count
                and candidate_preview_count <= candidate_preview_limit
            )
        )
    )
    counters_consistent = bool(
        all(value is not None for value in required_counts)
        and active_plus_queued == running + queued + attaching
        and completed == terminal
        and preview_consistent
    )
    integrity_verified = bool(
        search.get("available") is True
        and search.get("integrity_verified") is True
        and counters_consistent
    )
    return {
        "schema_version": 1,
        "available": bool(payload.get("available") is True and integrity_verified),
        "status": payload.get("status"),
        "source_kind": payload.get("source_kind"),
        "source_endpoint": "/api/nsga2",
        "integrity_verified": integrity_verified,
        "counters_consistent": counters_consistent,
        "candidate_preview_consistent": preview_consistent,
        "running_count": running,
        "queued_count": queued,
        "attaching_count": attaching,
        "active_plus_queued": active_plus_queued,
        "terminal_results_verified": terminal,
        "failed_terminal_results_verified": failed_terminal,
        "completed_count": completed,
        "feasible_pareto_count": pareto,
        "candidate_preview_count": candidate_preview_count,
        "candidate_preview_limit": candidate_preview_limit,
        "candidate_preview_truncated": candidate_preview_truncated,
        "near_feasible_count": near,
        "near_feasible_preview_count": search.get(
            "near_feasible_preview_count"
        ),
        "near_feasible_preview_limit": search.get(
            "near_feasible_preview_limit"
        ),
        "near_feasible_preview_truncated": search.get(
            "near_feasible_preview_truncated"
        ),
        "candidate_count": payload.get("candidate_count"),
        "valid_candidate_count": payload.get("valid_candidate_count"),
        "constraint_version": search.get("constraint_version"),
        "constraints": search.get("constraints") if isinstance(
            search.get("constraints"), dict
        ) else {},
        "temperature_constraint_contract": search.get(
            "temperature_constraint_contract"
        ) if isinstance(search.get("temperature_constraint_contract"), dict)
        else {},
        "coherent_snapshot_verified": search.get(
            "coherent_snapshot_verified"
        ),
        "coherent_snapshot_attempts": search.get(
            "coherent_snapshot_attempts"
        ),
        "source_count": search.get("source_count"),
        "supplemental_source_count": search.get(
            "supplemental_source_count"
        ),
        "configured_source_count": search.get("configured_source_count"),
        "rejected_source_count": search.get("rejected_source_count"),
        "all_configured_sources_verified": search.get(
            "all_configured_sources_verified"
        ),
        "source_aggregation": search.get("source_aggregation"),
        "sources": sources,
        "model_manifest_sha256": search.get("model_manifest_sha256"),
        "deployment_model_manifest_sha256": search.get(
            "deployment_model_manifest_sha256"
        ),
        "updated_at": search.get("updated_at") or payload.get("updated_at"),
        "warnings": search.get("warnings") if isinstance(
            search.get("warnings"), list
        ) else [],
        "error": None if integrity_verified else (
            "authoritative rolling NSGA integrity/count contract is unavailable"
        ),
    }


def _operator_host_allowlist() -> frozenset[str]:
    configured = os.environ.get("MFT_MONITOR_OPERATOR_HOSTS", "")
    values = configured.split(",") if configured.strip() else DEFAULT_OPERATOR_HOSTS
    return frozenset(
        value.strip().lower().removeprefix("[").removesuffix("]")
        for value in values
        if value.strip()
    )


def _operator_network_allowlist() -> tuple[Any, ...]:
    configured = os.environ.get("MFT_MONITOR_OPERATOR_NETWORKS", "")
    values = configured.split(",") if configured.strip() else DEFAULT_OPERATOR_NETWORKS
    try:
        return tuple(
            ip_network(value.strip(), strict=False)
            for value in values
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError("MFT_MONITOR_OPERATOR_NETWORKS contains an invalid CIDR") from exc


def create_app(
    regression_root: str | Path | None = None,
    service: ArtifactService | None = None,
    local_gui_launcher: Any | None = None,
    codex_status_reader: Any | None = None,
) -> FastAPI:
    service_was_supplied = service is not None
    compute_campaign = ComputeCampaignStatusReader()
    launcher_was_supplied = local_gui_launcher is not None
    root = Path(
        regression_root or os.environ.get("MFT_MONITOR_ROOT") or DEFAULT_REGRESSION_ROOT
    ).resolve()
    if service is None:
        scheduler = SchedulerReader(
            base_url=os.environ.get("MFT_SCHEDULER_URL", "http://127.0.0.1:8000"),
            task_prefix=os.environ.get("MFT_MONITOR_TASK_PREFIX", "mft"),
            project_name=os.environ.get("MFT_SCHEDULER_PROJECT", "MFT_1MW_2026v1"),
            timeout=float(os.environ.get("MFT_SCHEDULER_TIMEOUT", "2")),
            optional_timeout=float(
                os.environ.get("MFT_SCHEDULER_OPTIONAL_TIMEOUT", "3")
            ),
            pool_timeout=float(
                os.environ.get("MFT_SCHEDULER_POOL_TIMEOUT", "5")
            ),
        )
        service = ArtifactService(
            root,
            scheduler=scheduler,
            record_runtime=os.environ.get("MFT_MONITOR_DISABLE_HISTORY", "0") != "1",
        )

    app = FastAPI(
        title="MFT 1MW Campaign Monitor",
        description="Project dashboard for MFT campaign status and bounded trusted-LAN operator control.",
        version="1.2.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.service = service
    app.state.regression_root = root
    app.state.operator_host_allowlist = _operator_host_allowlist()
    app.state.operator_network_allowlist = _operator_network_allowlist()
    if codex_status_reader is None:
        codex_status_reader = CodexWorkStatusReader()
    app.state.codex_status_reader = codex_status_reader
    if local_gui_launcher is None:
        local_gui_launcher = _default_local_gui_launcher(root.parent)
    app.state.local_gui_launcher = local_gui_launcher
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.middleware("http")
    async def no_cache_api(request: Request, call_next):
        response = await call_next(request)
        # The operator dashboard is a live local tool.  Do not let a browser
        # keep an older HTML/JS/CSS bundle while the service has already moved
        # to a newer response schema.
        if request.url.path in {"/", "/cohorts"} or request.url.path.startswith(
            ("/api/", "/static/")
        ):
            response.headers["Cache-Control"] = "no-store"
        content_type = response.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() == "application/json":
            # JSON is UTF-8 by specification, but the explicit charset avoids
            # legacy clients treating Korean display fields as a local ANSI
            # code page and turning them into mojibake.
            response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    async def invoke(call: Callable[[], Any], section: str) -> JSONResponse:
        try:
            payload = await run_in_threadpool(call)
            return JSONResponse(payload)
        except Exception as exc:  # A single bad live artifact must not kill the UI.
            return JSONResponse(
                {
                    "schema_version": 1,
                    "available": False,
                    "section": section,
                    "error": f"{type(exc).__name__}: {exc}",
                    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
                status_code=200,
            )

    def attach_local_validations(nsga: Any) -> Any:
        if not isinstance(nsga, dict):
            return nsga
        candidates = nsga.get("candidates")
        if not isinstance(candidates, list):
            return nsga
        eligible_candidates = [
            candidate for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("gui_launch_eligible") is not False
        ]
        if not eligible_candidates:
            # Archived condition generations are intentionally read-only.
            # Their candidate identifiers must never join workstation GUI/FEA
            # validation records belonging to the active generation.
            return {
                **nsga,
                "candidates": [
                    {
                        **candidate,
                        "local_aedt_validation": None,
                    } if isinstance(candidate, dict) else candidate
                    for candidate in candidates
                ],
                "local_aedt_validation_count": 0,
            }
        if service_was_supplied and not launcher_was_supplied:
            # Unit/embedded callers that inject only an ArtifactService must
            # not accidentally inspect the workstation's default GUI runtime.
            return nsga
        try:
            validations = local_gui_launcher.validation_index()
        except Exception as exc:
            return {
                **nsga,
                "local_aedt_validation_error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        enriched = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                enriched.append(candidate)
                continue
            candidate_id = candidate.get("id")
            enriched.append({
                **candidate,
                "local_aedt_validation": (
                    validations.get(candidate_id)
                    if candidate.get("gui_launch_eligible") is not False
                    else None
                ),
            })
        return {
            **nsga,
            "candidates": enriched,
            "local_aedt_validation_count": sum(
                item.get("local_aedt_validation") is not None
                for item in enriched if isinstance(item, dict)
            ),
        }

    def dashboard_with_local_validations() -> Any:
        payload = service.dashboard()
        if not isinstance(payload, dict):
            return payload
        return {
            **payload,
            "nsga2": attach_local_validations(payload.get("nsga2")),
            "compute_campaign": compute_campaign.snapshot(),
        }

    def require_local_operator_request(
        request: Request,
        control_token: str = "simulation-policy-v1",
    ) -> None:
        """Permit bounded trusted-LAN operation while resisting Host/CSRF abuse."""
        client_host = request.client.host if request.client else ""
        try:
            client_address = ip_address(client_host)
            mapped = getattr(client_address, "ipv4_mapped", None)
            if mapped is not None:
                client_address = mapped
            if not (
                client_address.is_loopback
                or any(
                    client_address.version == network.version
                    and client_address in network
                    for network in app.state.operator_network_allowlist
                )
            ):
                raise ValueError("outside trusted operator network")
        except ValueError as exc:
            raise HTTPException(
                status_code=403,
                detail=(
                    "operator control is available only from loopback or a "
                    "configured trusted operator network"
                ),
            ) from exc
        host_header = request.headers.get("host", "").strip()
        host_name = urlsplit(f"//{host_header}").hostname or ""
        if host_name.lower() not in app.state.operator_host_allowlist:
            raise HTTPException(
                status_code=403,
                detail="operator control Host is not allowlisted",
            )
        if request.headers.get("x-mft-operator-control") != control_token:
            raise HTTPException(status_code=403, detail="operator control header is required")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise HTTPException(status_code=415, detail="operator control requires application/json")
        if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
            raise HTTPException(status_code=403, detail="cross-site operator control is forbidden")
        origin = request.headers.get("origin", "").strip()
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != host_header.lower():
                raise HTTPException(status_code=403, detail="operator control origin mismatch")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "title": "MFT 1MW 최적설계 모니터",
                "refresh_seconds": 20,
                "project_root": str(root),
            },
        )

    @app.get("/cohorts", response_class=HTMLResponse, include_in_schema=False)
    async def cohorts_page(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="cohorts.html",
            context={
                "title": "MFT 데이터 코호트 상세",
                "refresh_seconds": 20,
                "project_root": str(root),
            },
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {
            "status": "ok",
            "service": "mft-monitor",
            "regression_root": str(root),
        }

    @app.get("/api/dashboard")
    async def api_dashboard():
        return await invoke(
            dashboard_with_local_validations, "dashboard"
        )

    @app.get("/api/codex-work")
    async def api_codex_work():
        return await invoke(codex_status_reader.snapshot, "codex_work")

    @app.get("/api/status")
    async def api_status():
        return await invoke(service.status, "status")

    @app.get("/api/data")
    async def api_data():
        return await invoke(service.data, "data")

    @app.get("/api/models")
    async def api_models():
        return await invoke(service.models, "models")

    @app.get("/api/models/{target}/history")
    async def api_model_history(target: str):
        if target not in {item["name"] for item in TARGETS}:
            raise HTTPException(status_code=404, detail="unknown model target")
        return await invoke(lambda: service.model_history(target), "model_history")

    @app.get("/api/models/{target}/parity")
    async def api_model_parity(target: str):
        if target not in {item["name"] for item in TARGETS}:
            raise HTTPException(status_code=404, detail="unknown model target")
        return await invoke(lambda: service.model_parity(target), "model_parity")

    @app.get("/api/nsga2")
    async def api_nsga2():
        return await invoke(
            lambda: attach_local_validations(service.nsga2()), "nsga2"
        )

    @app.get("/api/nsga2/generations/{generation_id}")
    async def api_nsga2_generation(generation_id: str):
        return await invoke(
            lambda: attach_local_validations(
                service.nsga2_generation(generation_id)
            ),
            "nsga2_generation",
        )

    @app.get("/api/nsga2/progress")
    async def api_nsga2_progress():
        return await invoke(
            lambda: nsga_progress_summary(service.nsga2()),
            "nsga2_progress",
        )

    async def local_aedt_gui_snapshot(candidate_id: str | None = None):
        try:
            return await run_in_threadpool(
                lambda: local_gui_launcher.snapshot(candidate_id=candidate_id)
            )
        except CandidateLaunchError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"local AEDT GUI status failed: {type(exc).__name__}: {exc}",
            ) from exc

    @app.get("/api/local-aedt-gui/status")
    async def api_local_aedt_gui_status(candidate_id: str | None = None):
        return await local_aedt_gui_snapshot(candidate_id)

    @app.get("/api/local-aedt-gui/launches")
    async def api_local_aedt_gui_launches(candidate_id: str | None = None):
        """Read-only alias used by release canaries and operator tooling."""
        return await local_aedt_gui_snapshot(candidate_id)

    @app.post("/api/operator/local-aedt-gui/launch")
    async def api_launch_local_aedt_gui(request: Request):
        require_local_operator_request(request, "local-aedt-gui-v1")
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        candidate_id = payload.get("candidate_id")
        mode = payload.get("model")
        action = payload.get("action", "build")
        if not isinstance(candidate_id, str) or not candidate_id.strip() or len(candidate_id) > 200:
            raise HTTPException(status_code=422, detail="candidate_id is invalid")
        if mode not in LOCAL_AEDT_GUI_MODES:
            raise HTTPException(status_code=422, detail="model must be 'symmetry' or 'full'")
        if action not in LOCAL_AEDT_GUI_ACTIONS:
            raise HTTPException(status_code=422, detail="action must be 'build' or 'solve'")
        try:
            nsga = await run_in_threadpool(service.nsga2)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"current Pareto candidates are unavailable: {type(exc).__name__}: {exc}",
            ) from exc
        candidates = nsga.get("candidates") if isinstance(nsga, dict) else None
        candidates = candidates if isinstance(candidates, list) else []
        matches = [
            item for item in candidates
            if isinstance(item, dict) and item.get("id") == candidate_id
        ]
        if not matches:
            try:
                deadline_candidate = await run_in_threadpool(
                    lambda: service.deadline_design_candidate(candidate_id)
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "deadline design publication failed authentication: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                ) from exc
            if isinstance(deadline_candidate, dict):
                matches = [deadline_candidate]
        if len(matches) != 1:
            raise HTTPException(
                status_code=409,
                detail="selected Pareto candidate is stale or ambiguous; refresh and select it again",
            )
        candidate = matches[0]
        if candidate.get("gui_launch_eligible") is False:
            raise HTTPException(
                status_code=409,
                detail="the selected candidate is not eligible for local GUI launch",
            )
        action_key = (
            "gui_solve_eligible" if action == "solve"
            else "gui_build_eligible"
        )
        if candidate.get(action_key) is False:
            raise HTTPException(
                status_code=409,
                detail=(
                    candidate.get("gui_launch_limit_reason")
                    or f"the selected candidate is not eligible for {action}"
                ),
            )
        candidate_sha = candidate.get("pareto_front_sha256")
        candidate_row = candidate.get("pareto_row_number")
        if candidate.get("artifact_hydrated") is not True:
            raise HTTPException(
                status_code=409,
                detail=(
                    "the selected Pareto point is not backed by a hash-verified "
                    "full artifact row; refresh after artifact hydration"
                ),
            )
        if (
            not isinstance(candidate_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", candidate_sha) is None
            or not isinstance(candidate_row, int)
            or isinstance(candidate_row, bool)
            or candidate_row <= 0
        ):
            raise HTTPException(
                status_code=409,
                detail="the selected Pareto point has no valid artifact identity",
            )
        if (
            payload.get("pareto_front_sha256") != candidate_sha
            or payload.get("pareto_row_number") != candidate_row
        ):
            raise HTTPException(
                status_code=409,
                detail="Pareto artifact identity changed; refresh and select the candidate again",
            )
        try:
            launch = await run_in_threadpool(
                lambda: local_gui_launcher.launch(candidate, mode, action)
            )
        except DuplicateLaunchError as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "launch": exc.launch},
            ) from exc
        except LaunchCapacityError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except CandidateLaunchError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LocalAedtGuiError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"local AEDT GUI launch failed: {type(exc).__name__}: {exc}",
            ) from exc
        LOGGER.info(
            "local_aedt_gui_launch source=%s candidate=%s mode=%s action=%s pid=%s launch_id=%s",
            request.client.host if request.client else "",
            candidate_id,
            mode,
            action,
            launch.get("pid"),
            launch.get("launch_id"),
        )
        return {
            "schema_version": 1,
            "launched": True,
            "backend": "standalone",
            "launch": launch,
        }

    @app.get("/api/verification")
    async def api_verification():
        return await invoke(service.verification, "verification")

    @app.get("/api/history")
    async def api_history():
        return await invoke(service.history, "history")

    @app.get("/api/pipeline")
    async def api_pipeline():
        return await invoke(service.continuous_pipeline.snapshot, "pipeline")

    @app.get("/api/compute-campaign")
    async def api_compute_campaign():
        return await invoke(compute_campaign.snapshot, "compute_campaign")

    @app.patch("/api/operator/simulation-policy")
    async def api_set_simulation_policy(request: Request):
        require_local_operator_request(request)
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        target = payload.get("desired_simulations")
        expected_revision = payload.get("expected_revision")
        if type(target) is not int:
            raise HTTPException(status_code=422, detail="desired_simulations must be an integer")
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, (int, str))
            or not str(expected_revision).strip()
        ):
            raise HTTPException(status_code=422, detail="expected_revision is required")
        if payload.get("scale_down_mode") != "drain":
            raise HTTPException(status_code=422, detail="scale_down_mode must be drain")
        scheduler = getattr(service, "scheduler", None)
        setter = getattr(scheduler, "set_simulation_policy", None)
        if not callable(setter):
            raise HTTPException(status_code=503, detail="scheduler operator control is unavailable")
        try:
            def set_under_campaign_lock():
                CAMPAIGN_MUTATION_LOCK_PATH.parent.mkdir(
                    parents=True, exist_ok=True)
                with FileLock(
                        str(CAMPAIGN_MUTATION_LOCK_PATH), timeout=15 * 60):
                    current = scheduler.snapshot()
                    if current.get("policy_supported") is not True:
                        raise RuntimeError(
                            current.get("control_gate_reason")
                            or "scheduler simulation-policy is unavailable"
                        )
                    if current.get("control_enabled") is not True:
                        raise PermissionError(
                            current.get("control_gate_reason")
                            or "scheduler simulation-policy is gated"
                        )
                    if current.get("policy_revision") != expected_revision:
                        raise SimulationPolicyConflict(
                            "simulation policy changed; refresh and retry"
                        )
                    minimum = current.get("parallel_target_min")
                    maximum = current.get("parallel_target_max")
                    if (
                        type(minimum) is not int
                        or type(maximum) is not int
                        or not minimum <= target <= maximum
                    ):
                        raise ValueError(
                            f"desired_simulations must be between {minimum} and {maximum}"
                        )
                    return setter(
                        target, expected_revision=expected_revision
                    )

            result = await run_in_threadpool(set_under_campaign_lock)
        except SimulationPolicyConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=423, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"scheduler target update failed: {type(exc).__name__}: {exc}",
            ) from exc
        LOGGER.info(
            "simulation_policy_update source=%s project=%s expected_revision=%s "
            "new_revision=%s desired_simulations=%s",
            request.client.host if request.client else "",
            result.get("project"),
            expected_revision,
            result.get("policy_revision"),
            target,
        )
        return {
            "schema_version": 2,
            "updated": True,
            **result,
        }

    return app


app = create_app()
