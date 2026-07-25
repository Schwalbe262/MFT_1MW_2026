"""Audit whether runtime FEA truth may enter the goal surrogate dataset.

This command is intentionally read-only with respect to the canonical dataset
and Scheduler.  Only collection schemas supported by
``mft_goal_strict_al_ingest`` can become admitted rows.  Historical result
JSON is classified as documentary evidence and can never be promoted by this
tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_G0_MODEL_TARGETS,
    fixed_identity_mismatches,
)
from tools import mft_goal_strict_al_ingest as strict_al  # noqa: E402


REPORT_SCHEMA = "mft-goal-surrogate-retrain-audit-v1"
DEFAULT_BASE_SHA256 = (
    "0f0cb22a528cf029ce42101e7703d38200ae03038bd0aa95cc1619f34bca06a3"
)
DEFAULT_BASE_ROWS = 6_151
EXPECTED_PHYSICS_DATA_REVISION = "mft1mw-1k101-native-lamination-kf0p85-v3"
MAX_JSON_BYTES = 32 * 1024 * 1024
SUPPORTED_COLLECTION_SCHEMAS = frozenset(
    {
        strict_al.PRODUCTION_COLLECTION_SCHEMA,
        strict_al.DIAGNOSTIC_COLLECTION_SCHEMA,
    }
)
DOCUMENTARY_FILENAME_TOKENS = ("collection", "result")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def discover_json_artifacts(roots: Sequence[Path]) -> tuple[list[Path], dict]:
    """Discover bounded collection/result JSON without following symlinks."""

    paths: dict[str, Path] = {}
    skipped = Counter()
    for supplied_root in roots:
        try:
            root = supplied_root.resolve(strict=True)
        except OSError:
            skipped["missing_root"] += 1
            continue
        candidates = [root] if root.is_file() else root.rglob("*.json")
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
                if (
                    not resolved.is_file()
                    or resolved.is_symlink()
                    or candidate.is_symlink()
                ):
                    skipped["not_regular_file"] += 1
                    continue
                name = resolved.name.casefold()
                if not any(token in name for token in DOCUMENTARY_FILENAME_TOKENS):
                    continue
                if resolved.stat().st_size > MAX_JSON_BYTES:
                    skipped["larger_than_32_mib"] += 1
                    continue
            except OSError:
                skipped["unavailable"] += 1
                continue
            paths[str(resolved).casefold()] = resolved
    return sorted(paths.values(), key=lambda item: str(item).casefold()), {
        "counts": dict(sorted(skipped.items())),
        "candidate_json_files": len(paths),
    }


def _same_exact(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return actual == expected
    if isinstance(actual, bool):
        return False
    try:
        observed = float(actual)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(observed) and observed == float(expected)


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _standard_mode_mismatches(result: Mapping[str, Any]) -> list[str]:
    return sorted(
        key
        for key, expected in strict_al.STANDARD_MODE.items()
        if not _same_exact(result.get(key), expected)
    )


def classify_documentary_result(
    path: Path,
    payload: Mapping[str, Any],
    *,
    physics_data_revision: str,
) -> dict[str, Any] | None:
    """Classify raw historical result evidence without admitting it."""

    result = payload.get("result")
    if not isinstance(result, Mapping):
        return None
    schema = str(payload.get("schema_version") or "")
    if schema in SUPPORTED_COLLECTION_SCHEMAS:
        return None

    identity_mismatches = fixed_identity_mismatches(
        result, require_thermal_pad_metadata=True
    )
    mode_mismatches = _standard_mode_mismatches(result)
    missing_targets = sorted(
        target
        for target in GOAL_G0_MODEL_TARGETS
        if target not in result or not _finite(result.get(target))
    )
    rejection_codes = ["unsupported_collection_schema"]
    if identity_mismatches:
        rejection_codes.append("fixed_operating_or_cooling_identity_mismatch")
    if mode_mismatches:
        rejection_codes.append("retained_standard_mode_mismatch")
    if result.get("physics_data_revision") != physics_data_revision:
        rejection_codes.append("physics_data_revision_mismatch")
    if not _same_exact(result.get("N1"), strict_al.TARGETED_PRIMARY_TURNS):
        rejection_codes.append("targeted_N1_not_6")
    if missing_targets:
        rejection_codes.append("not_all_25_targets_finite")

    source_identity = payload.get("selected_candidate_identity")
    if not isinstance(source_identity, Mapping) or not source_identity.get(
        "source_task_payload_sha256"
    ):
        rejection_codes.append("authenticated_source_task_lineage_absent")
    task_id = payload.get("task_id", result.get("task_id"))
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": schema or None,
        "task_id": task_id,
        "N1": result.get("N1"),
        "fixed_identity_exact": not identity_mismatches,
        "fixed_identity_mismatch_fields": sorted(
            str(item.get("field")) for item in identity_mismatches
        ),
        "retained_standard_mode_exact": not mode_mismatches,
        "standard_mode_mismatch_fields": mode_mismatches,
        "physics_data_revision_exact": (
            result.get("physics_data_revision") == physics_data_revision
        ),
        "all_25_targets_finite": not missing_targets,
        "missing_or_nonfinite_targets": missing_targets,
        "authenticated_collection": False,
        "strict_collection_admissible": False,
        "rejection_codes": rejection_codes,
    }


def summarize_quality_status(
    path: Path,
    *,
    expected_dataset_sha256: str,
) -> dict[str, Any]:
    """Bind a quality result to the 25-target goal inventory."""

    payload = _read_object(path)
    targets = payload.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("quality status target inventory is absent")
    expected = set(GOAL_G0_MODEL_TARGETS)
    observed = set(map(str, targets))
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    passed_targets = sorted(
        target
        for target, value in targets.items()
        if isinstance(value, Mapping) and value.get("passed") is True
    )
    failed_targets = sorted(observed - set(passed_targets))
    dataset_exact = (
        str(payload.get("dataset_sha256") or "").lower()
        == expected_dataset_sha256.lower()
    )
    complete_inventory = not missing and not unexpected
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256_file(path.resolve(strict=True)),
        "training_run_id": payload.get("training_run_id"),
        "dataset_sha256": payload.get("dataset_sha256"),
        "dataset_identity_exact": dataset_exact,
        "strict_full_rows": payload.get("strict_full_rows"),
        "expected_target_count": len(expected),
        "observed_target_count": len(observed),
        "target_inventory_complete": complete_inventory,
        "missing_targets": missing,
        "unexpected_targets": unexpected,
        "passing_target_count": len(passed_targets),
        "failing_target_count": len(failed_targets),
        "failing_targets": failed_targets,
        "capacitance_recovery_passed": (
            isinstance(payload.get("capacitance_recovery"), Mapping)
            and payload["capacitance_recovery"].get("passed") is True
        ),
        "reported_passed": payload.get("passed") is True,
        "goal_25_target_quality_passed": (
            payload.get("passed") is True
            and dataset_exact
            and complete_inventory
        ),
        "reason_count": len(payload.get("reasons") or []),
    }


def _base_audit(
    base_dataset: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> tuple[dict[str, Any], str]:
    profile = strict_al._profile_content()
    _, record, revision, target_rows = strict_al._base_audit(
        base_dataset,
        expected_sha256=expected_sha256,
        expected_rows=expected_rows,
        profile=profile,
    )
    record = dict(record)
    record["physics_data_revision"] = revision
    record["all_25_targets_trainable"] = (
        set(target_rows) == set(GOAL_G0_MODEL_TARGETS)
        and all(count > 0 for count in target_rows.values())
    )
    record["target_trainable_rows"] = target_rows
    return record, revision


def _audit_supported_collections(
    collection_paths: Sequence[Path],
    *,
    base_dataset: Path,
    expected_base_sha256: str,
    expected_base_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    authenticated_paths = []
    for path in collection_paths:
        try:
            truth = strict_al.authenticate_collection(path)
        except Exception as exc:
            records.append(
                {
                    "path": str(path),
                    "schema_version": _read_object(path).get("schema_version"),
                    "authenticated": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        records.append(
            {
                "path": str(path),
                "schema_version": truth.collection.get("schema_version"),
                "authenticated": True,
                "adapter_kind": truth.adapter_kind,
                "source_task_payload_sha256": (
                    truth.source_task_payload_sha256
                ),
            }
        )
        authenticated_paths.append(path)

    if not authenticated_paths:
        return records, strict_al._admission(
            [],
            minimum_useful_rows=strict_al.DEFAULT_MINIMUM_USEFUL_ROWS,
            minimum_source_tasks=strict_al.DEFAULT_MINIMUM_SOURCE_TASKS,
        )
    try:
        prepared = strict_al.prepare_ingest(
            base_dataset=base_dataset,
            expected_base_sha256=expected_base_sha256,
            expected_base_rows=expected_base_rows,
            collection_paths=authenticated_paths,
        )
    except Exception as exc:
        return records, {
            "allowed": False,
            "reasons": [f"strict_prepare_ingest_failed:{type(exc).__name__}"],
            "strict_prepare_ingest_error": str(exc),
            "strict_new_rows": 0,
            "minimum_useful_rows": strict_al.DEFAULT_MINIMUM_USEFUL_ROWS,
            "unique_complete_geometries": 0,
            "minimum_unique_complete_geometries": (
                strict_al.DEFAULT_MINIMUM_UNIQUE_GEOMETRIES
            ),
            "unique_source_tasks": 0,
            "minimum_source_tasks": strict_al.DEFAULT_MINIMUM_SOURCE_TASKS,
        }
    return records, dict(prepared.retraining_admission)


def build_audit_report(
    *,
    base_dataset: Path,
    expected_base_sha256: str,
    expected_base_rows: int,
    scan_roots: Sequence[Path],
    quality_status: Path,
) -> dict[str, Any]:
    base, revision = _base_audit(
        base_dataset,
        expected_sha256=expected_base_sha256,
        expected_rows=expected_base_rows,
    )
    artifacts, discovery = discover_json_artifacts(scan_roots)
    supported_paths = []
    documentary = []
    decode_failures = []
    for path in artifacts:
        try:
            payload = _read_object(path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            decode_failures.append(
                {"path": str(path), "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue
        if payload.get("schema_version") in SUPPORTED_COLLECTION_SCHEMAS:
            supported_paths.append(path)
            continue
        record = classify_documentary_result(
            path, payload, physics_data_revision=revision
        )
        if record is not None:
            documentary.append(record)

    supported, admission = _audit_supported_collections(
        supported_paths,
        base_dataset=base_dataset,
        expected_base_sha256=expected_base_sha256,
        expected_base_rows=expected_base_rows,
    )
    rejection_counts = Counter(
        code for row in documentary for code in row["rejection_codes"]
    )
    exact_fixed_documentary = [
        row
        for row in documentary
        if row["fixed_identity_exact"]
        and row["retained_standard_mode_exact"]
        and row["physics_data_revision_exact"]
        and row["N1"] == strict_al.TARGETED_PRIMARY_TURNS
        and row["all_25_targets_finite"]
    ]
    quality = summarize_quality_status(
        quality_status, expected_dataset_sha256=expected_base_sha256
    )
    gate_open = admission.get("allowed") is True
    return {
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_mode": "read_only_fail_closed",
        "base_dataset": base,
        "scan": {
            "roots": [str(path) for path in scan_roots],
            **discovery,
            "decode_failures": decode_failures,
        },
        "supported_collection_inventory": {
            "supported_schemas": sorted(SUPPORTED_COLLECTION_SCHEMAS),
            "discovered_count": len(supported_paths),
            "authenticated_count": sum(
                record["authenticated"] is True for record in supported
            ),
            "records": supported,
        },
        "historical_documentary_inventory": {
            "result_artifact_count": len(documentary),
            "exact_contract_documentary_count": len(exact_fixed_documentary),
            "strict_collection_admissible_count": 0,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "records": documentary,
        },
        "retraining_admission": admission,
        "quality_gate": quality,
        "actions": {
            "training_gate_open": gate_open,
            "derived_dataset_created": False,
            "local_retraining_started": False,
            "slurm_training_post_performed": False,
            "standalone_fea_post_performed": False,
            "scheduler_mutation_performed": False,
            "canonical_dataset_mutated": False,
            "reason": (
                "strict authenticated rows/source lineage gate is open; "
                "use the sealed build/train tools"
                if gate_open
                else "strict authenticated rows/source lineage gate is closed"
            ),
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    admission = report["retraining_admission"]
    quality = report["quality_gate"]
    supported = report["supported_collection_inventory"]
    documentary = report["historical_documentary_inventory"]
    missing = ", ".join(quality["missing_targets"]) or "none"
    reasons = ", ".join(admission.get("reasons") or []) or "none"
    return "\n".join(
        [
            "# MFT goal surrogate retraining audit",
            "",
            f"- Generated (UTC): `{report['generated_at_utc']}`",
            f"- Canonical base rows: `{report['base_dataset']['row_count']}`",
            (
                "- Canonical base all-25-target trainability: "
                f"`{report['base_dataset']['all_25_targets_trainable']}`"
            ),
            (
                "- Supported sealed collections: "
                f"`{supported['authenticated_count']}` authenticated / "
                f"`{supported['discovered_count']}` discovered"
            ),
            (
                "- Historical documentary results: "
                f"`{documentary['result_artifact_count']}`; admitted `0`"
            ),
            (
                "- Strict new rows / source tasks: "
                f"`{admission.get('strict_new_rows', 0)}` / "
                f"`{admission.get('unique_source_tasks', 0)}`"
            ),
            f"- Retraining admission: `{admission.get('allowed')}` ({reasons})",
            (
                "- Recomputed model inventory: "
                f"`{quality['observed_target_count']}/"
                f"{quality['expected_target_count']}` targets"
            ),
            f"- Missing model targets: `{missing}`",
            (
                "- Existing generation quality: "
                f"`{quality['passing_target_count']}` pass, "
                f"`{quality['failing_target_count']}` fail; "
                f"goal gate `{quality['goal_25_target_quality_passed']}`"
            ),
            "",
            "No dataset, Scheduler, FEA, or project mutation was performed.",
            "",
        ]
    )


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", default=DEFAULT_BASE_SHA256)
    parser.add_argument("--expected-base-rows", type=int, default=DEFAULT_BASE_ROWS)
    parser.add_argument(
        "--scan-root", type=Path, action="append", required=True
    )
    parser.add_argument("--quality-status", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report = build_audit_report(
        base_dataset=args.base_dataset,
        expected_base_sha256=args.expected_base_sha256,
        expected_base_rows=args.expected_base_rows,
        scan_roots=args.scan_root,
        quality_status=args.quality_status,
    )
    encoded = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _write_atomic(args.output_json, encoded)
    _write_atomic(args.output_md, render_markdown(report).encode("utf-8"))
    print(json.dumps(report["actions"], sort_keys=True))


if __name__ == "__main__":
    main()
