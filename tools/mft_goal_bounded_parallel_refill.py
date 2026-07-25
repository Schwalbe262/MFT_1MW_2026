"""Bounded parallel extension for the exact MFT timeout12h SAFE REFILL.

This module does not create a second submission path.  It reuses the sealed
SAFE REFILL plan, timeout12h atomic claims, exact-once submitter, and watcher
extension receipts.  One invocation can POST at most one task and the entire
extension can ever POST only logical slots 96224 and 96230.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import canonical_sha256  # noqa: E402
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_safe_refill as refill  # noqa: E402


EVALUATION_SCHEMA = "mft-goal-bounded-parallel-refill-evaluation-v1"
ADDITIONAL_LOGICAL_IDS = frozenset({96224, 96230})
ACTIVE_LOGICAL_IDS = frozenset({96223, *ADDITIONAL_LOGICAL_IDS})
MAXIMUM_ADDITIONAL_POSTS_TOTAL = 2
KST = timezone(timedelta(hours=9))
SUBMISSION_CUTOFF_KST = datetime(2026, 7, 26, 5, 30, tzinfo=KST)
SUBMISSION_CUTOFF_UTC = SUBMISSION_CUTOFF_KST.astimezone(timezone.utc)
EXPECTED_RECLAIM_EVIDENCE_PATH = Path(
    "C:/Users/peets/slurm_scheduler_runtime/mft_goal_20260726/"
    "r1_reclaim_archive_20260726T002432_KST/"
    "ABCE_joint_hardlink_reclaim_floor_20260726T0205KST.json"
)
EXPECTED_RECLAIM_EVIDENCE_SHA256 = (
    "ed8a061cfe6f15483c65ab4eb94f42dc3c4a14d819bd116a27f0bcc7a5d3c103"
)
RECLAIM_SCHEMA = "mft-r1-joint-hardlink-reclaim-v1"
FULL_TRANSITION_LOGICAL_ID = 96223
FULL_SYMMETRY_EXPANSION_FACTOR = 8
_EXTENSION_NAME = re.compile(r"l([1-9][0-9]*)\.json")


class BoundedParallelRefillError(refill.SafeRefillError):
    """Fail-closed bounded-parallel authority or gate failure."""


def _task_id(row: Mapping[str, Any]) -> int | None:
    value = row.get("task_id", row.get("id"))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _deadline_open() -> bool:
    return _deadline_open_at(datetime.now(timezone.utc))


def _deadline_open_at(observed: datetime) -> bool:
    if observed.tzinfo is None:
        raise BoundedParallelRefillError(
            "submission cutoff observation lacks timezone"
        )
    return observed.astimezone(timezone.utc) < SUBMISSION_CUTOFF_UTC


def _all_active_task_inventory(
    *, scheduler_url: str
) -> list[dict[str, Any]]:
    query = production.urllib.parse.urlencode(
        [
            ("limit", 10000),
            *[
                ("status", status)
                for status in sorted(refill.ACTIVE_STATUSES)
            ],
        ]
    )
    value = refill._get_json(
        f"{scheduler_url.rstrip('/')}/api/tasks?{query}"
    )
    if not isinstance(value, list) or len(value) >= 10000:
        raise BoundedParallelRefillError(
            "global active task inventory is malformed/truncated"
        )
    rows = [dict(row) for row in value if isinstance(row, Mapping)]
    if len(rows) != len(value):
        raise BoundedParallelRefillError(
            "global active task inventory contains malformed rows"
        )
    return rows


def _candidate_index(
    plan: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    candidates = [dict(row) for row in plan.get("candidates", [])]
    by_logical = {
        int(row["logical_authority_task_id"]): row for row in candidates
    }
    uniqueness = (
        len(candidates)
        == len(by_logical)
        == len({row["dedupe_key"] for row in candidates})
        == len({row["task_name"] for row in candidates})
        == len({row["candidate_physics_sha256"] for row in candidates})
        == len({row["claim_directory"] for row in candidates})
    )
    if not uniqueness or set(by_logical) != set(refill.EXACT_ALLOWLIST):
        raise BoundedParallelRefillError(
            "SAFE REFILL candidate logical/dedupe/claim identity is not unique"
        )
    if any(
        row.get("account_name") != refill.TARGET_ACCOUNT
        or row.get("fixed_identity_attestation_sha256")
        != refill.FIXED_IDENTITY_SHA256
        for row in candidates
    ):
        raise BoundedParallelRefillError(
            "bounded parallel candidates changed account or fixed physics"
        )
    return by_logical


def _authenticated_extensions(
    plan: Mapping[str, Any],
    *,
    extension_authority_path: Path,
) -> dict[int, dict[str, Any]]:
    authority = refill.authenticate_extension_authority(
        extension_authority_path,
        watch_plan_path=Path(plan["watcher_plan"]["path"]),
    )
    directory = Path(authority["extension_directory"]).resolve()
    if directory != Path(plan["watcher_extension_directory"]).resolve():
        raise BoundedParallelRefillError(
            "watcher extension directory authority drifted"
        )
    if not directory.is_dir():
        return {}
    result: dict[int, dict[str, Any]] = {}
    execution_ids: set[int] = set()
    dedupe_keys: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        match = _EXTENSION_NAME.fullmatch(path.name)
        if match is None:
            raise BoundedParallelRefillError(
                "unexpected watcher extension filename"
            )
        logical = int(match.group(1))
        if logical not in ACTIVE_LOGICAL_IDS or logical in result:
            raise BoundedParallelRefillError(
                "watcher extension logical identity is duplicate/unapproved"
            )
        slot = refill.authenticate_watcher_extension_receipt(
            path, authority=authority
        )
        execution = int(slot["execution_task_id"])
        dedupe = str(slot["dedupe_key"])
        if execution in execution_ids or dedupe in dedupe_keys:
            raise BoundedParallelRefillError(
                "watcher extension execution/dedupe identity is duplicate"
            )
        execution_ids.add(execution)
        dedupe_keys.add(dedupe)
        result[logical] = copy.deepcopy(slot)
    return result


def _phase_transition_evidence(
    plan: Mapping[str, Any], *, reclaim_evidence_path: Path
) -> dict[str, Any]:
    by_logical = _candidate_index(plan)
    candidate = by_logical[FULL_TRANSITION_LOGICAL_ID]
    full_bound_gib = (
        int(candidate["fresh_grid_output_bytes"])
        * FULL_SYMMETRY_EXPANSION_FACTOR
        / (1024**3)
    )
    full_minimum_gib = full_bound_gib + refill.SAFETY_FLOOR_GB
    resolved_evidence = reclaim_evidence_path.resolve(strict=True)
    evidence_record = production._file_record(resolved_evidence)
    if (
        resolved_evidence != EXPECTED_RECLAIM_EVIDENCE_PATH.resolve(strict=True)
        or evidence_record["sha256"] != EXPECTED_RECLAIM_EVIDENCE_SHA256
    ):
        raise BoundedParallelRefillError(
            "reclaim evidence path or immutable byte identity drifted"
        )
    evidence = refill._read_json(resolved_evidence)
    floor = evidence.get("capacity_floor")
    release_gib = (
        int(evidence.get("guaranteed_regular_allocated_release_bytes") or 0)
        / (1024**3)
    )
    if (
        evidence.get("schema_version") != RECLAIM_SCHEMA
        or evidence.get("archive_target_filesystem") != "local C:"
        or evidence.get("archive_growth_charge_to_gpfs_bytes") != 0
        or not isinstance(floor, Mapping)
        or not math.isclose(
            release_gib,
            float(
                evidence.get(
                    "guaranteed_regular_allocated_release_gib", -1
                )
            ),
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(floor.get("full_minimum_gib", -1)),
            full_minimum_gib,
            rel_tol=0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(floor.get("projected_postdelete_free_gib", -1)),
            float(floor.get("observed_free_gib", -1)) + release_gib,
            rel_tol=0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(floor.get("projected_margin_gib", -1)),
            float(floor.get("projected_postdelete_free_gib", -1))
            - full_minimum_gib,
            rel_tol=0,
            abs_tol=1e-6,
        )
        or float(floor.get("projected_margin_gib", -1)) < 0
        or floor.get("active_output_bound_gib") != 0.0
        or not isinstance(floor.get("conditions"), list)
        or len(floor["conditions"]) < 3
    ):
        raise BoundedParallelRefillError(
            "sealed reclaim evidence cannot preserve the 96223 Full phase"
        )
    return {
        "source": evidence_record,
        "full_transition_logical_authority_task_id": (
            FULL_TRANSITION_LOGICAL_ID
        ),
        "standard_grid_bound_gib": candidate["prospective_grid_gb"],
        "full_symmetry_expansion_factor": (
            FULL_SYMMETRY_EXPANSION_FACTOR
        ),
        "full_prospective_grid_bound_gib": full_bound_gib,
        "full_minimum_with_floor_gib": full_minimum_gib,
        "projected_post_reclaim_free_gib": floor[
            "projected_postdelete_free_gib"
        ],
        "projected_margin_gib": floor["projected_margin_gib"],
        "conditional_on_all_sealed_reclaim_conditions": True,
        "standard_and_full_reservations_temporally_separated": True,
        "full_submission_requires_terminal_active_count_zero": True,
        "fresh_full_gpfs_gate_still_required": True,
        "other_standard_candidates_do_not_gain_r1_full_authority": True,
    }


def _active_bound_rows(
    *,
    tasks: Sequence[Mapping[str, Any]],
    candidates: Mapping[int, Mapping[str, Any]],
    extensions: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    by_execution = {
        int(slot["execution_task_id"]): (logical, slot)
        for logical, slot in extensions.items()
    }
    bounded: list[dict[str, Any]] = []
    unapproved: list[int] = []
    seen_logical: set[int] = set()
    for task in tasks:
        actual_account = str(task.get("account_name") or "")
        requested_account = str(task.get("requested_account_name") or "")
        accounts = {actual_account, requested_account} - {""}
        if refill.TARGET_ACCOUNT not in accounts or not refill._is_fea(task):
            continue
        if not accounts.issubset({refill.TARGET_ACCOUNT}):
            raise BoundedParallelRefillError(
                "active bounded task requested/actual account drifted"
            )
        task_id = _task_id(task)
        authenticated = by_execution.get(task_id or -1)
        if authenticated is None:
            unapproved.append(task_id or -1)
            continue
        logical, slot = authenticated
        candidate = candidates[logical]
        requested_node = task.get(
            "requested_node_name", task.get("node_name")
        )
        requested_policy = task.get(
            "requested_node_name_policy", task.get("node_name_policy")
        )
        if (
            logical in seen_logical
            or task.get("name") != slot["task_name"]
            or task.get("dedupe_key") != slot["dedupe_key"]
            or task.get("project") != refill.PROJECT
            or actual_account not in {"", refill.TARGET_ACCOUNT}
            or requested_account not in {"", refill.TARGET_ACCOUNT}
            or task.get("cpus") != 8
            or task.get("memory_mb") != 32768
            or task.get("timeout_seconds") != 43200
            or requested_node != refill.TARGET_NODE
            or requested_policy != "strict"
            or slot["candidate_physics_sha256"]
            != candidate["candidate_physics_sha256"]
            or slot["fixed_identity_attestation_sha256"]
            != refill.FIXED_IDENTITY_SHA256
        ):
            raise BoundedParallelRefillError(
                "active bounded task identity drifted"
            )
        seen_logical.add(logical)
        bounded.append(
            {
                "logical_authority_task_id": logical,
                "execution_task_id": task_id,
                "prospective_grid_gb": candidate["prospective_grid_gb"],
            }
        )
    return bounded, sorted(unapproved)


def evaluate(
    plan: Mapping[str, Any],
    *,
    extension_authority_path: Path,
    reclaim_evidence_path: Path,
    task_reader: Callable[..., list[dict[str, Any]]] = (
        _all_active_task_inventory
    ),
    capacity_reader: Callable[..., dict[str, Any]] = refill._capacity,
    storage_reader: Callable[[Mapping[str, Any]], dict[str, Any]] = (
        refill._fresh_storage
    ),
    validate_cutover: bool = True,
    ignore_claim_for_logical_id: int | None = None,
    extension_reader: Callable[..., dict[int, dict[str, Any]]] = (
        _authenticated_extensions
    ),
    deadline_reader: Callable[[], bool] = _deadline_open,
) -> dict[str, Any]:
    candidates = _candidate_index(plan)
    if validate_cutover:
        strict = refill.timeout12h._load_plan(
            Path(plan["candidates"][0]["plan"]["path"])
        )[0]["scheduler_strict_node_contract"]
        diagnostic._validate_scheduler_cutover_receipt(
            Path(plan["cutover_receipt"]["path"]),
            verify_live_launcher=True,
            require_strict_node=True,
            strict_node_contract=strict,
            require_active_strict=True,
        )
    tasks = task_reader(scheduler_url=plan["scheduler_url"])
    capacity = capacity_reader(
        scheduler_url=plan["scheduler_url"],
        account_name=refill.TARGET_ACCOUNT,
    )
    storage = storage_reader(plan)
    extensions = extension_reader(
        plan, extension_authority_path=extension_authority_path
    )
    active_bounds, unapproved = _active_bound_rows(
        tasks=tasks, candidates=candidates, extensions=extensions
    )
    active_logicals = {
        row["logical_authority_task_id"] for row in active_bounds
    }
    submitted_additional = sorted(
        ADDITIONAL_LOGICAL_IDS.intersection(extensions)
    )
    if len(submitted_additional) > MAXIMUM_ADDITIONAL_POSTS_TOTAL:
        raise BoundedParallelRefillError(
            "bounded parallel lifetime POST ceiling was exceeded"
        )
    active_bound_gib = sum(
        float(row["prospective_grid_gb"]) for row in active_bounds
    )
    limiting = storage["limiting_quota"]
    effective_free_gib = (
        float(limiting["raw_free_gb"])
        - float(limiting["in_doubt_gb"])
    )
    allocations = capacity.get("allocations")
    placement_rows = allocations if isinstance(allocations, list) else []
    placement_exact = bool(placement_rows) and all(
        row.get("account_name") == refill.TARGET_ACCOUNT
        and row.get("node_name") == refill.TARGET_NODE
        and row.get("state") in {"warm", "active"}
        and int(row.get("fit_slots") or 0) > 0
        for row in placement_rows
    )
    available = capacity.get("standalone_aedt_available")
    license_ready = (
        not isinstance(available, bool)
        and isinstance(available, int)
        and available > 0
    )
    phase = _phase_transition_evidence(
        plan, reclaim_evidence_path=reclaim_evidence_path
    )
    deadline_open = deadline_reader()
    rows = []
    for logical in sorted(
        ADDITIONAL_LOGICAL_IDS,
        key=lambda item: (
            float(candidates[item]["prospective_grid_gb"]),
            item,
        ),
    ):
        candidate = candidates[logical]
        siblings = [
            row for row in tasks if refill._is_sibling(row, candidate)
        ]
        claim = refill._claim_state(candidate)
        remaining = (
            effective_free_gib
            - active_bound_gib
            - float(candidate["prospective_grid_gb"])
            - refill.SAFETY_FLOOR_GB
        )
        gates = {
            "all_r1_active_fea_have_authenticated_bounds": not unapproved,
            "candidate_not_already_active": logical not in active_logicals,
            "candidate_not_already_submitted": (
                logical not in submitted_additional
            ),
            "maximum_two_additional_posts_not_exhausted": (
                len(submitted_additional)
                < MAXIMUM_ADDITIONAL_POSTS_TOTAL
            ),
            "scheduler_ready_fit_slots_positive": (
                int(capacity.get("ready_fit_slots") or 0) > 0
            ),
            "strict_r1_n114_placement_available": placement_exact,
            "standalone_aedt_license_slot_available": license_ready,
            "cumulative_active_plus_prospective_gpfs_envelope_passed": (
                remaining >= 0
            ),
            "candidate_logical_nonterminal_siblings_zero": (
                len(siblings) == 0
            ),
            "atomic_claim_unsubmitted": (
                claim == "unsubmitted"
                or logical == ignore_claim_for_logical_id
            ),
            "fixed_physics_fan_tim_pad_reauthenticated": True,
            "active_4fac_cutover_reauthenticated": True,
            "submission_cutoff_open": deadline_open,
            "conditional_96223_full_phase_preserved": (
                phase["projected_margin_gib"] >= 0
            ),
        }
        rows.append(
            {
                **copy.deepcopy(candidate),
                "eligible": all(gates.values()),
                "gates": gates,
                "claim_state": claim,
                "nonterminal_sibling_task_ids": [
                    _task_id(row) for row in siblings
                ],
                "active_authenticated_storage_bound_gib": (
                    active_bound_gib
                ),
                "cumulative_bound_with_this_candidate_gib": (
                    active_bound_gib
                    + float(candidate["prospective_grid_gb"])
                ),
                "fresh_gpfs_remaining_after_active_prospective_floor_gib": (
                    remaining
                ),
            }
        )
    selected = next((row for row in rows if row["eligible"]), None)
    return {
        "schema_version": EVALUATION_SCHEMA,
        "observed_at_utc": refill._now(),
        "task_inventory_sha256": canonical_sha256(tasks),
        "active_task_count": len(tasks),
        "capacity": copy.deepcopy(capacity),
        "capacity_sha256": canonical_sha256(capacity),
        "storage": copy.deepcopy(storage),
        "storage_sha256": canonical_sha256(storage),
        "authenticated_active_bounds": active_bounds,
        "active_authenticated_storage_bound_gib": active_bound_gib,
        "unapproved_r1_active_fea_task_ids": unapproved,
        "submitted_additional_logical_ids": submitted_additional,
        "remaining_additional_post_authority": (
            MAXIMUM_ADDITIONAL_POSTS_TOTAL - len(submitted_additional)
        ),
        "maximum_additional_posts_total": (
            MAXIMUM_ADDITIONAL_POSTS_TOTAL
        ),
        "submission_cutoff_utc": SUBMISSION_CUTOFF_UTC.isoformat(),
        "phase_transition_readiness": phase,
        "cross_account_replanning_allowed": False,
        "required_account_name": refill.TARGET_ACCOUNT,
        "required_node_name": refill.TARGET_NODE,
        "candidates": rows,
        "selected_logical_authority_task_id": (
            selected["logical_authority_task_id"] if selected else None
        ),
        "all_scheduler_reads_get_only": True,
    }


def cycle(
    *,
    refill_plan_path: Path,
    extension_authority_path: Path,
    reclaim_evidence_path: Path,
    authorize_submit: bool = False,
    task_reader: Callable[..., list[dict[str, Any]]] = (
        _all_active_task_inventory
    ),
    capacity_reader: Callable[..., dict[str, Any]] = refill._capacity,
    storage_reader: Callable[[Mapping[str, Any]], dict[str, Any]] = (
        refill._fresh_storage
    ),
    submitter: Callable[..., Path] = refill.timeout12h.submit,
    validate_cutover: bool = True,
) -> dict[str, Any]:
    def gate(
        plan: Mapping[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        return evaluate(
            plan,
            extension_authority_path=extension_authority_path,
            reclaim_evidence_path=reclaim_evidence_path,
            **kwargs,
        )

    return refill.cycle(
        refill_plan_path=refill_plan_path,
        extension_authority_path=extension_authority_path,
        authorize_submit=authorize_submit,
        task_reader=task_reader,
        capacity_reader=capacity_reader,
        storage_reader=storage_reader,
        submitter=submitter,
        validate_cutover=validate_cutover,
        gate_evaluator=gate,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded parallel exact MFT SAFE REFILL extension"
    )
    parser.add_argument("--refill-plan", type=Path, required=True)
    parser.add_argument("--extension-authority", type=Path, required=True)
    parser.add_argument("--reclaim-evidence", type=Path, required=True)
    parser.add_argument(
        "--authorize-submit",
        action="store_true",
        help="enable at most one exact guarded POST in this invocation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state = cycle(
        refill_plan_path=args.refill_plan,
        extension_authority_path=args.extension_authority,
        reclaim_evidence_path=args.reclaim_evidence,
        authorize_submit=args.authorize_submit,
    )
    print(json.dumps(state, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
