"""Read-only bridge from terminal Standard truth to surrogate retraining.

The terminal watcher state is mutable operational status, so it is never used
as training authority by itself.  This tool reauthenticates the immutable watch
plan, any authorized watcher extensions, every terminal success receipt and
its sealed Standard collection.  When feasible truth exists it also requires
the watcher's immutable global-NDS receipt and truth Pareto manifest.

All successful strict Standard rows, including rows that miss the goal hard
constraints, are then passed to :mod:`mft_goal_strict_al_ingest` in memory.
No dataset, model, Scheduler task, or campaign is created here.  Exact dataset
build and local-only training-plan argv are emitted only when the 8-row /
8-geometry / 4-source retraining admission gate passes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import mft_goal_safe_refill as safe_refill  # noqa: E402
from tools import mft_goal_strict_al_ingest as strict_al  # noqa: E402
from tools import mft_goal_terminal_success_watcher as watcher  # noqa: E402
from tools import mft_goal_truth_promotion as promotion  # noqa: E402


REPORT_SCHEMA = "mft-goal-surrogate-retrain-readiness-v1"
DEFAULT_BASE_DATASET = Path(
    r"C:\Users\peets\slurm_scheduler_runtime\mft_cap_recovery_runs"
    r"\7d3efe22995e-8e89f6e8846d\dataset\strict_006151.parquet"
)
DEFAULT_BASE_SHA256 = "0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3"
DEFAULT_BASE_ROWS = 6_151
DEFAULT_REMOTE_TRAINING_ROOT = "/gpfs/tmp_cpu2/mft_goal_20260726/al_training"
KNOWN_NON_SUCCESS_STATES = frozenset(
    {
        "active",
        "pending",
        "queued",
        "attaching",
        "starting",
        "running",
        "unknown",
        "terminal_failure",
        "transient_or_contract_error",
    }
)


class ReadinessContractError(RuntimeError):
    """Raised when watcher/truth authority cannot be reauthenticated."""


@dataclass(frozen=True)
class TerminalInventory:
    """Reauthenticated immutable terminal evidence."""

    watch_plan: Mapping[str, Any]
    state: Mapping[str, Any]
    receipts: tuple[Mapping[str, Any], ...]
    collection_paths: tuple[Path, ...]
    feasible_receipts: tuple[Mapping[str, Any], ...]
    truth_nds: Mapping[str, Any] | None
    state_file_record: Mapping[str, Any] | None = None


def _record(path: Path) -> dict[str, Any]:
    try:
        return production._file_record(path.resolve(strict=True))
    except Exception as exc:
        raise ReadinessContractError(
            f"artifact record cannot be authenticated: {path}"
        ) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = production._read_json(path.resolve(strict=True))
    except Exception as exc:
        raise ReadinessContractError(
            f"JSON artifact cannot be authenticated: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise ReadinessContractError(f"JSON artifact is not an object: {path}")
    return value


def _effective_watch_plan(
    watch_plan_path: Path,
    *,
    extension_authority_path: Path | None,
) -> dict[str, Any]:
    """Load base + authorized extensions without creating directories."""

    try:
        plan = watcher._load_watch_plan(watch_plan_path)
    except Exception as exc:
        raise ReadinessContractError(
            "immutable terminal watch plan authentication failed"
        ) from exc
    merged = copy.deepcopy(dict(plan))
    merged["authorized_extension_count"] = 0
    if extension_authority_path is None:
        return merged
    try:
        authority = safe_refill.authenticate_extension_authority(
            extension_authority_path,
            watch_plan_path=watch_plan_path,
        )
        extension_directory = Path(
            str(authority.get("extension_directory") or "")
        ).resolve(strict=True)
    except Exception as exc:
        raise ReadinessContractError(
            "watcher extension authority authentication failed"
        ) from exc
    if not extension_directory.is_dir() or extension_directory.is_symlink():
        raise ReadinessContractError(
            "watcher extension directory is not an existing regular directory"
        )
    extensions = []
    for path in sorted(extension_directory.glob("*.json")):
        try:
            extensions.append(
                safe_refill.authenticate_watcher_extension_receipt(
                    path, authority=authority
                )
            )
        except Exception as exc:
            raise ReadinessContractError(
                f"watcher extension receipt authentication failed: {path}"
            ) from exc
    slots = [*merged["slots"], *extensions]
    logical_ids = [item.get("logical_authority_task_id") for item in slots]
    execution_ids = [item.get("execution_task_id") for item in slots]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in [*logical_ids, *execution_ids]
        )
        or len(set(logical_ids)) != len(slots)
        or len(set(execution_ids)) != len(slots)
    ):
        raise ReadinessContractError(
            "effective watch plan contains duplicate/invalid slots"
        )
    merged["slots"] = sorted(slots, key=lambda item: item["logical_authority_task_id"])
    merged["exact_logical_slot_count"] = len(slots)
    merged["authorized_extension_count"] = len(extensions)
    merged["extension_authority"] = _record(extension_authority_path)
    return merged


def _load_state(
    state_path: Path,
    *,
    watch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        resolved_state = state_path.resolve(strict=True)
        raw = resolved_state.read_bytes()
        decoded = json.loads(raw.decode("utf-8"))
        state = watcher._validate_seal(decoded, watcher.STATE_SCHEMA)
    except Exception as exc:
        raise ReadinessContractError(
            "terminal watcher state seal authentication failed"
        ) from exc
    state_file_record = {
        "path": str(resolved_state),
        "sha256": production._sha256_bytes(raw),
        "size_bytes": len(raw),
    }
    root = Path(str(watch_plan.get("output_root") or "")).resolve()
    if resolved_state != root / "state.json":
        raise ReadinessContractError(
            "terminal watcher state escaped the sealed output root"
        )
    state_slots = state.get("slots")
    if (
        state.get("campaign_id") != "mft-goal-20260726"
        or state.get("watch_plan_payload_sha256") != watch_plan.get("payload_sha256")
        or state.get("scheduler_methods_used") != ["GET"]
        or state.get("scheduler_submission_performed") is not False
        or state.get("scheduler_cancel_performed") is not False
        or state.get("scheduler_mutation_performed") is not False
        or not isinstance(state_slots, list)
    ):
        raise ReadinessContractError(
            "terminal watcher state safety/authority contract drifted"
        )
    expected = {
        (
            item["logical_authority_task_id"],
            item["execution_task_id"],
        )
        for item in watch_plan["slots"]
    }
    observed = []
    for item in state_slots:
        if not isinstance(item, Mapping):
            raise ReadinessContractError("terminal state slot is malformed")
        pair = (
            item.get("logical_authority_task_id"),
            item.get("execution_task_id"),
        )
        observed.append(pair)
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ReadinessContractError(
            "terminal watcher state/effective-plan slot inventory drifted"
        )
    recomputed_counts = dict(
        sorted(Counter(str(item.get("state") or "") for item in state_slots).items())
    )
    if state.get("counts") != recomputed_counts:
        raise ReadinessContractError("terminal watcher state counts drifted")
    authenticated = dict(state)
    authenticated["_authenticated_state_file_record"] = state_file_record
    return authenticated


def _assert_state_record_current(
    state_path: Path,
    expected: Mapping[str, Any] | None,
) -> None:
    if expected is None or _record(state_path) != dict(expected):
        raise ReadinessContractError(
            "terminal watcher state changed during readiness audit"
        )


def _failure_ledger(
    path: Path,
    *,
    slot: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value = watcher._validate_seal(_read_json(path), watcher.FAILURE_LEDGER_SCHEMA)
    except Exception as exc:
        raise ReadinessContractError(
            f"terminal failure ledger authentication failed: {path}"
        ) from exc
    if (
        value.get("logical_authority_task_id") != slot["logical_authority_task_id"]
        or value.get("execution_task_id") != slot["execution_task_id"]
        or value.get("physical_truth_claimed") is not False
        or value.get("global_truth_nds_included") is not False
        or value.get("scheduler_get_only") is not True
        or value.get("scheduler_mutation_performed") is not False
    ):
        raise ReadinessContractError("terminal failure ledger contract drifted")
    return value


def _authenticate_truth_nds(
    *,
    root: Path,
    state: Mapping[str, Any],
    feasible_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    manifest_value = state.get("latest_truth_manifest")
    if not feasible_receipts:
        if manifest_value is not None:
            raise ReadinessContractError(
                "truth manifest exists without feasible success receipts"
            )
        return None
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ReadinessContractError(
            "feasible terminal truth lacks global-NDS manifest"
        )
    try:
        manifest_path = Path(manifest_value).resolve(strict=True)
        manifest_path.relative_to((root / "truth_snapshots").resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ReadinessContractError(
            "global truth manifest escaped watcher truth snapshots"
        ) from exc
    receipt_path = (
        manifest_path.parent.parent / f"{manifest_path.parent.name}.receipt.json"
    )
    try:
        nds_receipt = watcher._validate_seal(
            _read_json(receipt_path), watcher.NDS_RECEIPT_SCHEMA
        )
        truth_manifest, ranked, _by_candidate = promotion._load_truth_manifest(
            manifest_path
        )
    except Exception as exc:
        raise ReadinessContractError(
            "global truth NDS receipt/manifest authentication failed"
        ) from exc
    ordered = sorted(
        feasible_receipts,
        key=lambda item: item["logical_authority_task_id"],
    )
    expected_logical_ids = [item["logical_authority_task_id"] for item in ordered]
    expected_collections = [dict(item["collection"]) for item in ordered]
    if (
        nds_receipt.get("campaign_id") != "mft-goal-20260726"
        or nds_receipt.get("authority_key") != watcher._nds_key(ordered)
        or nds_receipt.get("logical_authority_task_ids") != expected_logical_ids
        or nds_receipt.get("input_collections") != expected_collections
        or nds_receipt.get("truth_manifest") != _record(manifest_path)
        or nds_receipt.get("global_nondominated_sort_performed") is not True
        or nds_receipt.get("scheduler_mutation_performed") is not False
        or truth_manifest.get("input_collection_count") != len(ordered)
    ):
        raise ReadinessContractError(
            "global truth NDS receipt/manifest inventory drifted"
        )
    return {
        "receipt": _record(receipt_path),
        "receipt_payload_sha256": nds_receipt["payload_sha256"],
        "manifest": _record(manifest_path),
        "manifest_payload_sha256": truth_manifest["payload_sha256"],
        "input_collection_count": len(ordered),
        "ranked_truth_count": len(ranked),
        "authenticated": True,
    }


def authenticate_terminal_inventory(
    *,
    watch_plan_path: Path,
    state_path: Path,
    extension_authority_path: Path | None = None,
) -> TerminalInventory:
    """Reauthenticate all completed watcher evidence without Scheduler I/O."""

    watch_plan = _effective_watch_plan(
        watch_plan_path,
        extension_authority_path=extension_authority_path,
    )
    state = _load_state(state_path, watch_plan=watch_plan)
    state = dict(state)
    state_file_record = state.pop("_authenticated_state_file_record", None)
    root = Path(watch_plan["output_root"]).resolve()
    slot_by_identity = {
        (
            item["logical_authority_task_id"],
            item["execution_task_id"],
        ): item
        for item in watch_plan["slots"]
    }
    receipts = []
    collection_paths = []
    for state_slot in state["slots"]:
        pair = (
            state_slot["logical_authority_task_id"],
            state_slot["execution_task_id"],
        )
        slot = slot_by_identity[pair]
        paths = watcher._slot_paths(root, slot)
        kind = str(state_slot.get("state") or "")
        if kind in {"authenticated_feasible", "authenticated_infeasible"}:
            try:
                observed_path = Path(str(state_slot.get("receipt") or "")).resolve(
                    strict=True
                )
            except OSError as exc:
                raise ReadinessContractError(
                    "authenticated terminal state lacks its success receipt"
                ) from exc
            if observed_path != paths["receipt"].resolve():
                raise ReadinessContractError(
                    "terminal success receipt escaped its exact slot"
                )
            try:
                receipt = watcher._load_success_receipt(observed_path, slot)
            except Exception as exc:
                raise ReadinessContractError(
                    "terminal success receipt reauthentication failed"
                ) from exc
            expected_kind = (
                "authenticated_feasible"
                if receipt["eligible_for_global_truth_nds"] is True
                else "authenticated_infeasible"
            )
            if kind != expected_kind:
                raise ReadinessContractError(
                    "terminal success receipt feasibility/state drifted"
                )
            receipts.append(receipt)
            collection_paths.append(
                Path(receipt["collection"]["path"]).resolve(strict=True)
            )
        elif kind == "terminal_failure":
            try:
                ledger_path = Path(str(state_slot.get("ledger") or "")).resolve(
                    strict=True
                )
            except OSError as exc:
                raise ReadinessContractError(
                    "terminal failure state lacks its ledger"
                ) from exc
            if ledger_path != paths["failure"].resolve():
                raise ReadinessContractError(
                    "terminal failure ledger escaped its exact slot"
                )
            _failure_ledger(ledger_path, slot=slot)
        elif kind not in KNOWN_NON_SUCCESS_STATES:
            raise ReadinessContractError(
                f"unsupported terminal state classification: {kind!r}"
            )
    logical_ids = [item["logical_authority_task_id"] for item in receipts]
    collection_shas = [item["collection"]["sha256"] for item in receipts]
    if len(set(logical_ids)) != len(logical_ids) or len(set(collection_shas)) != len(
        collection_shas
    ):
        raise ReadinessContractError(
            "terminal success receipt logical/collection identity is duplicated"
        )
    feasible = tuple(
        item for item in receipts if item["eligible_for_global_truth_nds"] is True
    )
    truth_nds = _authenticate_truth_nds(
        root=root,
        state=state,
        feasible_receipts=feasible,
    )
    return TerminalInventory(
        watch_plan=watch_plan,
        state=state,
        receipts=tuple(receipts),
        collection_paths=tuple(collection_paths),
        feasible_receipts=feasible,
        truth_nds=truth_nds,
        state_file_record=state_file_record,
    )


def _strict_readiness(
    *,
    base_dataset: Path,
    expected_base_sha256: str,
    expected_base_rows: int,
    collection_paths: Sequence[Path],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if collection_paths:
        prepared = strict_al.prepare_ingest(
            base_dataset=base_dataset,
            expected_base_sha256=expected_base_sha256,
            expected_base_rows=expected_base_rows,
            collection_paths=collection_paths,
        )
        return strict_al._summary(prepared), prepared.retraining_admission
    profile = strict_al._profile_content()
    _frame, base, revision, target_rows = strict_al._base_audit(
        base_dataset,
        expected_sha256=expected_base_sha256,
        expected_rows=expected_base_rows,
        profile=profile,
    )
    admission = strict_al._admission(
        [],
        minimum_useful_rows=strict_al.DEFAULT_MINIMUM_USEFUL_ROWS,
        minimum_source_tasks=strict_al.DEFAULT_MINIMUM_SOURCE_TASKS,
    )
    return {
        "schema_version": "mft-goal-strict-al-inspection-v1",
        "base_dataset": base,
        "new_rows": 0,
        "output_rows": expected_base_rows,
        "physics_data_revision_cohort": revision,
        "all_25_targets_gain_each_new_row": False,
        "base_target_eligible_rows": target_rows,
        "retraining_admission": admission,
        "scheduler_mutation_performed": False,
    }, admission


def _command_plan(
    *,
    admitted: bool,
    python_executable: Path,
    code_root: Path,
    expected_code_revision: str,
    base_dataset: Path,
    expected_base_sha256: str,
    expected_base_rows: int,
    collection_paths: Sequence[Path],
    derived_dataset_output: Path,
    training_plan_root: Path,
    remote_training_root: str,
) -> dict[str, Any]:
    if not admitted:
        return {
            "emitted": False,
            "reason": "strict authenticated truth admission gate is closed",
            "dataset_build_argv": None,
            "training_plan_argv": None,
        }
    revision = expected_code_revision.strip().lower()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ReadinessContractError(
            "expected clean code revision is not an exact git SHA"
        )
    build = [
        str(python_executable),
        "-m",
        "tools.mft_goal_strict_al_ingest",
        "build",
        "--base-dataset",
        str(base_dataset),
        "--expected-base-sha256",
        expected_base_sha256,
        "--expected-base-rows",
        str(expected_base_rows),
    ]
    for collection in collection_paths:
        build.extend(["--collection", str(collection)])
    build.extend(
        [
            "--minimum-useful-rows",
            str(strict_al.DEFAULT_MINIMUM_USEFUL_ROWS),
            "--minimum-source-tasks",
            str(strict_al.DEFAULT_MINIMUM_SOURCE_TASKS),
            "--require-retraining-ready",
            "--output-dir",
            str(derived_dataset_output),
        ]
    )
    train_plan = [
        str(python_executable),
        "-m",
        "tools.mft_goal_al_slurm_train",
        "plan",
        "--dataset-manifest",
        str(derived_dataset_output / "manifest.json"),
        "--code-root",
        str(code_root),
        "--expected-code-revision",
        revision,
        "--local-root",
        str(training_plan_root),
        "--remote-root",
        remote_training_root,
    ]
    return {
        "emitted": True,
        "reason": "strict authenticated truth admission gate passed",
        "dataset_build_argv": build,
        "training_plan_argv": train_plan,
        "training_stage_apply_emitted": False,
        "scheduler_submit_apply_emitted": False,
        "scheduler_post_authorized": False,
        "required_after_training": [
            "authenticate the complete 25-target training collection",
            "require a new dataset/model inventory SHA lineage",
            "run unchanged model quality thresholds",
            "seal a disjoint 512-seed N1=5/6/7/8 campaign",
            "globally sort all 512 authenticated terminal populations",
        ],
        "old_generation_result_mixing_allowed": False,
    }


def build_report(
    *,
    watch_plan_path: Path,
    state_path: Path,
    extension_authority_path: Path | None,
    base_dataset: Path,
    expected_base_sha256: str,
    expected_base_rows: int,
    python_executable: Path,
    code_root: Path,
    expected_code_revision: str,
    derived_dataset_output: Path,
    training_plan_root: Path,
    remote_training_root: str,
) -> dict[str, Any]:
    inventory = authenticate_terminal_inventory(
        watch_plan_path=watch_plan_path,
        state_path=state_path,
        extension_authority_path=extension_authority_path,
    )
    strict_summary, admission = _strict_readiness(
        base_dataset=base_dataset,
        expected_base_sha256=expected_base_sha256,
        expected_base_rows=expected_base_rows,
        collection_paths=inventory.collection_paths,
    )
    commands = _command_plan(
        admitted=admission.get("allowed") is True,
        python_executable=python_executable,
        code_root=code_root,
        expected_code_revision=expected_code_revision,
        base_dataset=base_dataset,
        expected_base_sha256=expected_base_sha256,
        expected_base_rows=expected_base_rows,
        collection_paths=inventory.collection_paths,
        derived_dataset_output=derived_dataset_output,
        training_plan_root=training_plan_root,
        remote_training_root=remote_training_root,
    )
    feasible = len(inventory.feasible_receipts)
    success = len(inventory.receipts)
    _assert_state_record_current(
        state_path,
        inventory.state_file_record,
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_mode": "read_only_fail_closed",
        "watcher_authority": {
            "watch_plan": _record(watch_plan_path),
            "watch_plan_payload_sha256": inventory.watch_plan["payload_sha256"],
            "state": copy.deepcopy(dict(inventory.state_file_record)),
            "state_payload_sha256": inventory.state["payload_sha256"],
            "extension_authority": (
                _record(extension_authority_path)
                if extension_authority_path is not None
                else None
            ),
            "effective_slot_count": len(inventory.watch_plan["slots"]),
            "authorized_extension_count": inventory.watch_plan[
                "authorized_extension_count"
            ],
            "state_counts": dict(inventory.state["counts"]),
        },
        "terminal_truth_inventory": {
            "authenticated_success_count": success,
            "goal_feasible_success_count": feasible,
            "goal_infeasible_success_count": success - feasible,
            "all_successes_used_for_strict_training_validation": True,
            "only_goal_feasible_successes_used_for_truth_global_nds": True,
            "records": [
                {
                    "logical_authority_task_id": item["logical_authority_task_id"],
                    "execution_task_id": item["execution_task_id"],
                    "collection": copy.deepcopy(item["collection"]),
                    "strict_authentication": copy.deepcopy(
                        item["strict_authentication"]
                    ),
                    "goal_feasible": item["eligible_for_global_truth_nds"],
                    "fixed_boundary_policy_preserved": item[
                        "fixed_boundary_policy_preserved"
                    ],
                }
                for item in inventory.receipts
            ],
        },
        "truth_global_nds": (
            copy.deepcopy(inventory.truth_nds)
            if inventory.truth_nds is not None
            else {
                "authenticated": False,
                "required": feasible > 0,
                "reason": "no goal-feasible Standard truth exists",
            }
        ),
        "strict_data_gate": strict_summary,
        "retraining_admission": dict(admission),
        "commands": commands,
        "mutations": {
            "canonical_dataset_mutated": False,
            "derived_dataset_created": False,
            "model_training_started": False,
            "scheduler_post_performed": False,
            "scheduler_cancel_performed": False,
            "scheduler_mutation_performed": False,
            "campaign_created": False,
        },
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reauthenticate terminal Standard truth and emit retraining argv "
            "only when the strict active-learning data gate passes."
        )
    )
    parser.add_argument("--watch-plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--extension-authority", type=Path)
    parser.add_argument("--base-dataset", type=Path, default=DEFAULT_BASE_DATASET)
    parser.add_argument("--expected-base-sha256", default=DEFAULT_BASE_SHA256)
    parser.add_argument("--expected-base-rows", type=int, default=DEFAULT_BASE_ROWS)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--derived-dataset-output", type=Path, required=True)
    parser.add_argument("--training-plan-root", type=Path, required=True)
    parser.add_argument("--remote-training-root", default=DEFAULT_REMOTE_TRAINING_ROOT)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(
        watch_plan_path=args.watch_plan,
        state_path=args.state,
        extension_authority_path=args.extension_authority,
        base_dataset=args.base_dataset,
        expected_base_sha256=args.expected_base_sha256.lower(),
        expected_base_rows=args.expected_base_rows,
        python_executable=args.python_executable,
        code_root=args.code_root,
        expected_code_revision=args.expected_code_revision,
        derived_dataset_output=args.derived_dataset_output,
        training_plan_root=args.training_plan_root,
        remote_training_root=args.remote_training_root,
    )
    if args.output_json is not None:
        _write_atomic(args.output_json, report)
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReadinessContractError as exc:
        print(f"[surrogate-retrain-readiness] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
