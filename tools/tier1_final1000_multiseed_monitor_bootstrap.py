"""Create and seal a frozen compact-v2 probe before production writers exist.

This module is deliberately local-only.  It builds a deterministic fixture,
publishes it through the production compact snapshot functions, and emits
content-addressed probe evidence.  It has no Scheduler or remote client and
does not import any AEDT/FEA code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_final1000_multiseed_contract import (
        COMPACT_INDEX_SCHEMA,
        PROTOCOL_VERSION,
        json_bytes,
    )
    from tier1_final1000_multiseed_monitor import (
        MAX_CAPABILITY_RECEIPT_BYTES,
        adapt_condition_index,
        build_backend_capability_receipt,
        default_adapter_code_files,
        require_backend_capability,
    )
    from tier1_final1000_multiseed_status import (
        MAX_HOT_WIRE_BYTES,
        SINGLE_SEED_PROTOCOL,
        build_compact_snapshot,
        publish_compact_snapshot,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_final1000_multiseed_contract import (
        COMPACT_INDEX_SCHEMA,
        PROTOCOL_VERSION,
        json_bytes,
    )
    from tools.tier1_final1000_multiseed_monitor import (
        MAX_CAPABILITY_RECEIPT_BYTES,
        adapt_condition_index,
        build_backend_capability_receipt,
        default_adapter_code_files,
        require_backend_capability,
    )
    from tools.tier1_final1000_multiseed_status import (
        MAX_HOT_WIRE_BYTES,
        SINGLE_SEED_PROTOCOL,
        build_compact_snapshot,
        publish_compact_snapshot,
    )


BOOTSTRAP_MANIFEST_SCHEMA = (
    "mft-tier1-final1000-frozen-compact-v2-bootstrap-manifest-v1"
)
PROBE_EVIDENCE_SCHEMA = "mft-tier1-final1000-frozen-compact-v2-probe-evidence-v1"
FROZEN_STAGE_ID = "final1000-compact-v2-bootstrap-probe"
FROZEN_UPDATED_AT = "2026-07-22T00:00:00+00:00"
SINGLE_SEED_TASK_ID = 910001
BATCH4_TASK_ID = 910004
MAX_BOOTSTRAP_MANIFEST_BYTES = 4 * 1024 * 1024

EXPECTED_PROBE = {
    "physical_lane_count": 2,
    "logical_seed_count": 5,
    "logical_sealed_seed_count": 3,
    "logical_unsealed_seed_count": 2,
    "hidden_unsealed_seed_record_count": 1,
    "returned_terminal_seed_count": 3,
    "returned_seeds": [11001, 12001, 12002],
    "positive_physical_task_ids": [SINGLE_SEED_TASK_ID, BATCH4_TASK_ID],
    "virtual_scheduler_task_ids_created": False,
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: str) -> str:
    relative = PurePosixPath(str(value))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in str(value)
        or "\x00" in str(value)
        or str(relative) != str(value)
    ):
        raise RuntimeError("bootstrap output reference is unsafe")
    return str(relative)


def _output_child(root: Path, relative: str) -> Path:
    safe = _safe_relative(relative)
    candidate = (root / Path(*PurePosixPath(safe).parts)).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise RuntimeError("bootstrap output reference escaped its explicit root")
    return candidate


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    resolved = path.resolve(strict=True)
    size = resolved.stat().st_size
    if size < 0 or size >= maximum:
        raise RuntimeError(f"{label} exceeds its byte bound")
    payload = resolved.read_bytes()
    if len(payload) != size:
        raise RuntimeError(f"{label} changed during read")
    return payload


def _immutable_write(path: Path, payload: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable bootstrap artifact collision: {path}")
        return 0
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return 1


def _file_reference(root: Path, path: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise RuntimeError("bootstrap artifact is outside its explicit root")
    payload = _read_bounded(resolved, MAX_HOT_WIRE_BYTES, "bootstrap artifact")
    relative = resolved.relative_to(resolved_root).as_posix()
    return {"path": relative, "sha256": _sha256(payload), "size": len(payload)}


def _fixture_snapshot() -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]
]:
    lanes = [
        {
            "task_id": SINGLE_SEED_TASK_ID,
            "stage_id": FROZEN_STAGE_ID,
            "protocol_version": SINGLE_SEED_PROTOCOL,
            "state": "completed",
            "account_name": "bootstrap-probe",
        },
        {
            "task_id": BATCH4_TASK_ID,
            "stage_id": FROZEN_STAGE_ID,
            "protocol_version": PROTOCOL_VERSION,
            "state": "running",
            "account_name": "bootstrap-probe",
            "batch_length": 4,
            "current_seed": 12003,
            "sealed_child_count": 2,
        },
    ]
    records = [
        {
            "bundle_id": "bootstrap-single-seed",
            "seed": 11001,
            "terminal_state": "completed",
            "source_physical_task_id": SINGLE_SEED_TASK_ID,
        },
        {
            "bundle_id": "bootstrap-batch4",
            "seed": 12001,
            "terminal_state": "completed",
            "physical_parent_task_id": BATCH4_TASK_ID,
            "batch_ordinal": 0,
        },
        {
            "bundle_id": "bootstrap-batch4",
            "seed": 12002,
            "terminal_state": "failed",
            "physical_parent_task_id": BATCH4_TASK_ID,
            "batch_ordinal": 1,
        },
        # This authenticated-looking receipt intentionally sits ahead of the
        # parent cursor.  The compact builder must hide it from the probe page.
        {
            "bundle_id": "bootstrap-batch4",
            "seed": 12003,
            "terminal_state": "completed",
            "physical_parent_task_id": BATCH4_TASK_ID,
            "batch_ordinal": 2,
        },
    ]
    return build_compact_snapshot(
        stage_id=FROZEN_STAGE_ID,
        physical_lanes=lanes,
        seed_records=records,
        frontend_static={
            "cohort_id": "frozen-compact-v2-bootstrap",
            "probe_fixture": True,
        },
        shard_size=2,
        relative_root="frozen-seed-results",
        updated_at=FROZEN_UPDATED_AT,
    )


def _probe_assertions(index_path: Path) -> tuple[dict[str, Any], list[str]]:
    page = adapt_condition_index(index_path, offset=0, limit=256)
    task_ids = sorted(int(item["task_id"]) for item in page["latest_tasks"])
    returned_seeds = sorted(int(item["seed"]) for item in page["terminal_results"])
    wire_size = len(json_bytes(page))
    observed = {
        "physical_lane_count": int(page["physical_lane_count"]),
        "logical_seed_count": int(page["logical_seed_count"]),
        "logical_sealed_seed_count": int(page["logical_sealed_seed_count"]),
        "logical_unsealed_seed_count": int(page["logical_unsealed_seed_count"]),
        "hidden_unsealed_seed_record_count": int(
            page["hidden_unsealed_seed_record_count"]
        ),
        "returned_terminal_seed_count": int(page["terminal_results_returned_count"]),
        "returned_seeds": returned_seeds,
        "positive_physical_task_ids": task_ids,
        "virtual_scheduler_task_ids_created": page[
            "virtual_scheduler_task_ids_created"
        ],
    }
    if observed != EXPECTED_PROBE:
        raise RuntimeError("frozen compact-v2 probe semantics drifted")
    if any(task_id <= 0 for task_id in task_ids):
        raise RuntimeError("frozen compact-v2 probe invented a virtual task id")
    if wire_size >= MAX_HOT_WIRE_BYTES:
        raise RuntimeError("frozen compact-v2 probe exceeds the 32-MiB wire cap")
    names = [
        "physical-single-and-batch4-lanes",
        "positive-physical-task-ids-only",
        "logical-counts-separate-from-physical-counts",
        "sealed-batch-prefix-visible",
        "unsealed-batch-child-hidden",
        "no-virtual-scheduler-task-ids",
        "normalized-wire-below-32-mib",
    ]
    return {**observed, "normalized_wire_size": wire_size}, names


def _junit_payload(assertion_names: Sequence[str], probe_sha256: str) -> bytes:
    cases = "".join(
        f'<testcase classname="final1000.compact_v2.bootstrap" name="{name}"/>'
        for name in assertion_names
    )
    value = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite name="final1000-frozen-compact-v2-bootstrap" '
        f'tests="{len(assertion_names)}" failures="0" errors="0" skipped="0">'
        f'<properties><property name="probe_sha256" value="{probe_sha256}"/>'
        f"</properties>{cases}</testsuite>\n"
    )
    return value.encode("utf-8")


def publish_frozen_probe(output_root: Path) -> dict[str, Any]:
    """Build and immutably publish the deterministic local probe."""

    output_root.mkdir(parents=True, exist_ok=True)
    root = output_root.resolve(strict=True)
    status, manifest, shards, index = _fixture_snapshot()
    index_payload = json_bytes(index)
    index_name = f"frozen-compact-v2-index-{_sha256(index_payload)}.json"
    index_path = _output_child(root, index_name)
    if index_path.exists() and index_path.read_bytes() != index_payload:
        raise RuntimeError("immutable frozen compact-v2 index collision")
    writes = publish_compact_snapshot(
        root,
        status,
        manifest,
        shards,
        index,
        pointer_name=index_name,
    )
    index_path.chmod(0o444)
    observed, assertion_names = _probe_assertions(index_path)
    evidence_unsigned = {
        "schema_version": PROBE_EVIDENCE_SCHEMA,
        "stage_id": FROZEN_STAGE_ID,
        "updated_at": FROZEN_UPDATED_AT,
        "index_file_sha256": _sha256(index_payload),
        "index_sha256": index["index_sha256"],
        "assertions": assertion_names,
        "observed": observed,
        "scheduler_mutation_performed": False,
        "remote_write_performed": False,
        "aedt_used": False,
        "fea_submission_performed": False,
    }
    evidence = {
        **evidence_unsigned,
        "evidence_sha256": canonical_sha256(evidence_unsigned),
    }
    evidence_payload = json_bytes(evidence)
    evidence_path = _output_child(
        root,
        f"evidence/frozen-compact-v2-probe-{_sha256(evidence_payload)}.json",
    )
    writes += _immutable_write(evidence_path, evidence_payload)
    junit_payload = _junit_payload(assertion_names, evidence["evidence_sha256"])
    junit_path = _output_child(
        root,
        f"evidence/frozen-compact-v2-probe-{_sha256(junit_payload)}.junit.xml",
    )
    writes += _immutable_write(junit_path, junit_payload)
    manifest_unsigned = {
        "schema_version": BOOTSTRAP_MANIFEST_SCHEMA,
        "publication_root": str(root),
        "stage_id": FROZEN_STAGE_ID,
        "updated_at": FROZEN_UPDATED_AT,
        "compact_index_schema_version": COMPACT_INDEX_SCHEMA,
        "index": _file_reference(root, index_path)
        | {"index_sha256": index["index_sha256"]},
        "probe_evidence": _file_reference(root, evidence_path)
        | {"evidence_sha256": evidence["evidence_sha256"]},
        "junit_evidence": _file_reference(root, junit_path),
        "expected_probe": copy.deepcopy(EXPECTED_PROBE),
        "scheduler_mutation_performed": False,
        "remote_write_performed": False,
        "aedt_used": False,
        "fea_submission_performed": False,
    }
    bootstrap_manifest = {
        **manifest_unsigned,
        "manifest_sha256": canonical_sha256(manifest_unsigned),
    }
    bootstrap_payload = json_bytes(bootstrap_manifest)
    bootstrap_path = _output_child(
        root,
        f"frozen-compact-v2-bootstrap-{_sha256(bootstrap_payload)}.json",
    )
    writes += _immutable_write(bootstrap_path, bootstrap_payload)
    return {
        "publication_root": str(root),
        "bootstrap_manifest_path": str(bootstrap_path),
        "bootstrap_manifest_sha256": _sha256(bootstrap_payload),
        "index_path": str(index_path),
        "index_file_sha256": _sha256(index_payload),
        "probe_evidence_path": str(evidence_path),
        "junit_evidence_path": str(junit_path),
        "writes_performed": writes,
        "scheduler_mutation_performed": False,
        "remote_write_performed": False,
        "aedt_used": False,
        "fea_submission_performed": False,
    }


def _validated_reference(root: Path, reference: Mapping[str, Any], label: str) -> Path:
    if not isinstance(reference, Mapping) or set(reference) not in (
        {"path", "sha256", "size"},
        {"path", "sha256", "size", "index_sha256"},
        {"path", "sha256", "size", "evidence_sha256"},
    ):
        raise RuntimeError(f"frozen bootstrap {label} reference is invalid")
    path = _output_child(root, str(reference.get("path") or ""))
    payload = _read_bounded(path, MAX_HOT_WIRE_BYTES, label)
    if (
        isinstance(reference.get("size"), bool)
        or not isinstance(reference.get("size"), int)
        or reference["size"] != len(payload)
        or reference.get("sha256") != _sha256(payload)
    ):
        raise RuntimeError(f"frozen bootstrap {label} reference drifted")
    return path


def load_frozen_probe(bootstrap_manifest_path: Path) -> dict[str, Any]:
    """Validate the exact, non-relocatable frozen probe publication."""

    manifest_path = bootstrap_manifest_path.resolve(strict=True)
    payload = _read_bounded(
        manifest_path, MAX_BOOTSTRAP_MANIFEST_BYTES, "bootstrap manifest"
    )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("bootstrap manifest is not valid JSON") from exc
    unsigned = (
        {key: item for key, item in value.items() if key != "manifest_sha256"}
        if isinstance(value, dict)
        else {}
    )
    required = {
        "schema_version",
        "publication_root",
        "stage_id",
        "updated_at",
        "compact_index_schema_version",
        "index",
        "probe_evidence",
        "junit_evidence",
        "expected_probe",
        "scheduler_mutation_performed",
        "remote_write_performed",
        "aedt_used",
        "fea_submission_performed",
        "manifest_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != BOOTSTRAP_MANIFEST_SCHEMA
        or value.get("stage_id") != FROZEN_STAGE_ID
        or value.get("updated_at") != FROZEN_UPDATED_AT
        or value.get("compact_index_schema_version") != COMPACT_INDEX_SCHEMA
        or value.get("expected_probe") != EXPECTED_PROBE
        or value.get("manifest_sha256") != canonical_sha256(unsigned)
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_mutation_performed",
                "remote_write_performed",
                "aedt_used",
                "fea_submission_performed",
            )
        )
    ):
        raise RuntimeError("frozen compact-v2 bootstrap manifest identity is invalid")
    if manifest_path.name != (f"frozen-compact-v2-bootstrap-{_sha256(payload)}.json"):
        raise RuntimeError(
            "frozen compact-v2 bootstrap manifest is not content-addressed"
        )
    root = Path(str(value["publication_root"])).resolve(strict=True)
    if manifest_path.parent != root:
        raise RuntimeError("frozen compact-v2 bootstrap publication was relocated")
    index_path = _validated_reference(root, value["index"], "index")
    evidence_path = _validated_reference(
        root, value["probe_evidence"], "probe evidence"
    )
    junit_path = _validated_reference(root, value["junit_evidence"], "JUnit evidence")
    expected_paths = {
        "index": f"frozen-compact-v2-index-{value['index']['sha256']}.json",
        "probe_evidence": (
            f"evidence/frozen-compact-v2-probe-{value['probe_evidence']['sha256']}.json"
        ),
        "junit_evidence": (
            "evidence/frozen-compact-v2-probe-"
            f"{value['junit_evidence']['sha256']}.junit.xml"
        ),
    }
    if any(value[name].get("path") != path for name, path in expected_paths.items()):
        raise RuntimeError(
            "frozen compact-v2 bootstrap artifact is not content-addressed"
        )
    page_observed, assertion_names = _probe_assertions(index_path)
    evidence_payload = _read_bounded(
        evidence_path, MAX_HOT_WIRE_BYTES, "probe evidence"
    )
    evidence = json.loads(evidence_payload)
    evidence_unsigned = (
        {key: item for key, item in evidence.items() if key != "evidence_sha256"}
        if isinstance(evidence, dict)
        else {}
    )
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != PROBE_EVIDENCE_SCHEMA
        or evidence.get("evidence_sha256") != canonical_sha256(evidence_unsigned)
        or evidence.get("evidence_sha256")
        != value["probe_evidence"].get("evidence_sha256")
        or evidence.get("index_file_sha256") != value["index"].get("sha256")
        or evidence.get("index_sha256") != value["index"].get("index_sha256")
        or evidence.get("assertions") != assertion_names
        or evidence.get("observed") != page_observed
    ):
        raise RuntimeError("frozen compact-v2 probe evidence drifted")
    expected_junit = _junit_payload(assertion_names, evidence["evidence_sha256"])
    if junit_path.read_bytes() != expected_junit:
        raise RuntimeError("frozen compact-v2 JUnit evidence drifted")
    return {
        "manifest": copy.deepcopy(value),
        "manifest_path": manifest_path,
        "publication_root": root,
        "index_path": index_path,
        "probe_evidence_path": evidence_path,
        "junit_evidence_path": junit_path,
    }


def _named_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        logical = _safe_relative(name) if separator else ""
        if not separator or not logical or not raw_path or logical in result:
            raise RuntimeError(f"{label} must use unique LOGICAL_NAME=PATH values")
        result[logical] = Path(raw_path)
    if not result:
        raise RuntimeError(f"at least one {label} is required")
    return result


def _capability_inputs(
    *,
    bootstrap_manifest_path: Path,
    code_root: Path,
    code_revision: str,
    backend_files: Mapping[str, Path],
    backend_revision: str,
    test_revision: str,
) -> dict[str, Any]:
    frozen = load_frozen_probe(bootstrap_manifest_path)
    return {
        "index_path": frozen["index_path"],
        "code_files": default_adapter_code_files(code_root),
        "code_revision": code_revision,
        "backend_files": dict(backend_files),
        "backend_revision": backend_revision,
        "test_evidence_files": {
            "bootstrap/frozen-compact-v2-probe.json": frozen["probe_evidence_path"],
            "bootstrap/frozen-compact-v2-probe.junit.xml": frozen[
                "junit_evidence_path"
            ],
        },
        "test_revision": test_revision,
    }


def seal_frozen_probe_capability(
    *,
    output_root: Path,
    bootstrap_manifest_path: Path,
    code_root: Path,
    code_revision: str,
    backend_files: Mapping[str, Path],
    backend_revision: str,
    test_revision: str,
) -> dict[str, Any]:
    """Seal the exact adapter, backend, generated evidence, and frozen index."""

    output_root.mkdir(parents=True, exist_ok=True)
    root = output_root.resolve(strict=True)
    inputs = _capability_inputs(
        bootstrap_manifest_path=bootstrap_manifest_path,
        code_root=code_root,
        code_revision=code_revision,
        backend_files=backend_files,
        backend_revision=backend_revision,
        test_revision=test_revision,
    )
    receipt = build_backend_capability_receipt(**inputs)
    payload = json_bytes(receipt)
    if len(payload) >= MAX_CAPABILITY_RECEIPT_BYTES:
        raise RuntimeError("backend capability receipt exceeds its byte bound")
    receipt_path = _output_child(
        root, f"capabilities/backend-capability-{_sha256(payload)}.json"
    )
    writes = _immutable_write(receipt_path, payload)
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "condition_index_identity_sha256": receipt["condition_index_identity"][
            "identity_sha256"
        ],
        "writes_performed": writes,
    }


def require_frozen_probe_capability(
    receipt_path: Path,
    *,
    bootstrap_manifest_path: Path,
    code_root: Path,
    code_revision: str,
    backend_files: Mapping[str, Path],
    backend_revision: str,
    test_revision: str,
) -> dict[str, Any]:
    inputs = _capability_inputs(
        bootstrap_manifest_path=bootstrap_manifest_path,
        code_root=code_root,
        code_revision=code_revision,
        backend_files=backend_files,
        backend_revision=backend_revision,
        test_revision=test_revision,
    )
    return require_backend_capability(receipt_path, **inputs)


def _identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bootstrap-manifest", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument(
        "--backend-file", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument("--backend-revision", required=True)
    parser.add_argument("--test-revision", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--output-root", type=Path, required=True)
    seal = commands.add_parser("seal-capability")
    seal.add_argument("--output-root", type=Path, required=True)
    _identity_arguments(seal)
    validate = commands.add_parser("validate-capability")
    validate.add_argument("--receipt", type=Path, required=True)
    _identity_arguments(validate)
    return parser


def _cli_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bootstrap_manifest_path": args.bootstrap_manifest,
        "code_root": args.code_root,
        "code_revision": args.code_revision,
        "backend_files": _named_paths(args.backend_file, "backend file"),
        "backend_revision": args.backend_revision,
        "test_revision": args.test_revision,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        result = publish_frozen_probe(args.output_root)
    elif args.command == "seal-capability":
        result = seal_frozen_probe_capability(
            output_root=args.output_root, **_cli_identity(args)
        )
    else:
        receipt = require_frozen_probe_capability(args.receipt, **_cli_identity(args))
        result = {
            "valid": True,
            "receipt_sha256": receipt["receipt_sha256"],
            "condition_index_identity_sha256": receipt["condition_index_identity"][
                "identity_sha256"
            ],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
