"""Build an explicit project-only package from a provisional Full bridge.

This package format is intentionally distinct from
``mft-goal-truth-full-package-v1``.  It carries the authenticated Standard
and provisional Full AEDT project files plus the actual-truth global-NDS
evidence that authorized the pair.  A provisional Full run has no retained
``.aedtresults`` tree, so this consumer records that absence and never
synthesizes one.

The only Scheduler operation supported here is the retained-artifact GET used
to reconstruct the symmetric Standard project.  There is no submit, cancel,
claim, or remote mutation path in this module.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from module.mft_goal_20260726_contract import (  # noqa: E402
    GOAL_CONTRACT_SCHEMA,
    GOAL_STAGE_SPEC,
    GOAL_STAGE_SPEC_SHA256,
    GOAL_TEMPERATURE_CONTRACT_SHA256,
    canonical_sha256,
)
from tools import mft_goal_diagnostic_standard_probe as diagnostic  # noqa: E402
from tools import mft_goal_fea_handoff as production  # noqa: E402
from tools import (  # noqa: E402
    mft_goal_provisional_full_promotion_bridge as bridge_adapter,
)
from tools import mft_goal_terminal_success_watcher as terminal  # noqa: E402
from tools import mft_goal_truth_promotion as promotion  # noqa: E402


PACKAGE_SCHEMA = "mft-goal-provisional-full-explicit-package-v2"
CAMPAIGN_ID = "mft-goal-20260726"
MANIFEST_NAME = "package_manifest.json"
PAYLOAD_ROLES = {
    "symmetric_model.aedt": "authenticated_standard_symmetric_project",
    "full_model.aedt": "authenticated_provisional_full_project",
    "promotion_bridge.json": "explicit_project_only_promotion_bridge",
    "standard_collection.json": "actual_standard_collection_evidence",
    "provisional_full_success.json": "provisional_full_success_evidence",
    "canonical_full_plan.json": "rank0_actual_truth_full_plan",
    "actual_truth_pareto_manifest.json": "actual_truth_pareto_manifest",
    "actual_truth_nds_receipt.json": "global_nondominated_sort_receipt",
    "truth_validated_pareto_front.csv": "actual_truth_pareto_front",
}
PAYLOAD_NAMES = frozenset(PAYLOAD_ROLES)


HandoffContractError = production.HandoffContractError


def _static_flags() -> dict[str, bool]:
    return {
        "explicit_project_only_package": True,
        "canonical_truth_full_package_v1_compatible": False,
        "canonical_truth_full_package_v1_impersonated": False,
        "canonical_truth_full_collection_v1_impersonated": False,
        "full_aedtresults_available": False,
        "full_aedtresults_synthesized": False,
        "scheduler_get_only_artifact_reconstruction": True,
        "scheduler_mutation_performed": False,
        "scheduler_post_performed": False,
        "scheduler_cancel_performed": False,
        "remote_artifact_mutation_performed": False,
        "self_contained_deep_validation_supported": False,
        "external_authority_required_for_deep_reauthentication": True,
        "package_directory_relocatable_with_external_authority": True,
        "standalone_cross_machine_relocatable": False,
    }


PACKAGE_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "campaign_id",
        "package_kind",
        "canonical_package_schema",
        "goal_contract_schema",
        "hard_spec",
        "hard_spec_sha256",
        "temperature_contract_sha256",
        "candidate_physics_sha256",
        "logical_authority_task_id",
        "standard_task_id",
        "provisional_full_task_id",
        "solver_revision",
        "library_revision",
        "fea_params_sha256",
        "effective_full_params_sha256",
        "fixed_boundary",
        "fixed_identity_attestation",
        "promotion_bridge",
        "source_authority",
        "models",
        "result_trees",
        "actual_truth_global_nds",
        "inventory",
        "package_tree_sha256",
        "bridge_passed",
        "project_only_model_package_eligible",
        "production_eligible",
        "canonical_v1_promotion_allowed",
        "consumer_contract",
        "created_at_utc",
        *_static_flags(),
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffContractError(f"{label} is absent or malformed")
    return dict(value)


def _source_record(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
    record = _require_mapping(value, label)
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise HandoffContractError(f"{label} file record fields drifted")
    path = Path(str(record.get("path") or "")).resolve(strict=True)
    if production._file_record(path) != record:
        raise HandoffContractError(f"{label} source bytes drifted")
    return path, record


def _copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    resolved = source.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink() or destination.exists():
        raise HandoffContractError("unsafe immutable package copy")
    shutil.copyfile(resolved, destination)
    if (
        not destination.is_file()
        or destination.is_symlink()
        or production._sha256_file(destination) != production._sha256_file(resolved)
        or destination.stat().st_size != resolved.stat().st_size
    ):
        raise HandoffContractError("packaged evidence copy differs")
    os.chmod(destination, 0o444)
    return {
        "relative_path": destination.name,
        "sha256": production._sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "role": PAYLOAD_ROLES[destination.name],
    }


def _inventory_record(path: Path) -> dict[str, Any]:
    if path.name not in PAYLOAD_NAMES or not path.is_file() or path.is_symlink():
        raise HandoffContractError("package inventory entry is unsafe")
    return {
        "relative_path": path.name,
        "sha256": production._sha256_file(path),
        "size_bytes": path.stat().st_size,
        "role": PAYLOAD_ROLES[path.name],
    }


def _validate_bridge_gate(bridge: Mapping[str, Any]) -> None:
    required_true = (
        "authenticated_standard_constraint_pass",
        "authenticated_provisional_full_constraint_pass",
        "exact_same_candidate_parameters_revisions",
        "exact_fixed_cooling_identity",
        "exact_geometry_identity",
        "rank0_standard_truth_authority",
        "project_only_model_package_eligible",
        "bridge_passed",
        "production_eligible",
    )
    if (
        any(bridge.get(name) is not True for name in required_true)
        or bridge.get("schema_version") != bridge_adapter.BRIDGE_SCHEMA
        or bridge.get("truth_non_dominated_rank") != 0
        or bridge.get("canonical_v1_promotion_allowed") is not False
        or bridge.get("campaign_id") != CAMPAIGN_ID
        or bridge.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or bridge.get("hard_spec") != GOAL_STAGE_SPEC
        or bridge.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or bridge.get("temperature_contract_sha256") != GOAL_TEMPERATURE_CONTRACT_SHA256
    ):
        raise HandoffContractError("provisional Full bridge is not package-eligible")
    full_model = _require_mapping(bridge.get("full_model"), "Full model")
    symmetric_model = _require_mapping(bridge.get("symmetric_model"), "symmetric model")
    consumer = _require_mapping(
        bridge.get("consumer_contract"), "bridge consumer contract"
    )
    if (
        full_model.get("full_model") != 1
        or full_model.get("thermal_symmetry") != "full"
        or full_model.get("aedtresults") is not None
        or symmetric_model.get("full_model") != 0
        or symmetric_model.get("thermal_symmetry") != "eighth"
        or bridge.get("full_aedtresults_available") is not False
        or consumer.get("schema_version")
        != "mft-goal-project-only-full-package-consumer-v1"
        or consumer.get("full_aedtresults_must_not_be_synthesized") is not True
        or consumer.get("truth_full_collection_v1_must_not_be_emitted") is not True
        or consumer.get("truth_full_package_v1_must_not_be_emitted") is not True
    ):
        raise HandoffContractError("provisional Full bridge consumer boundary drifted")
    # The bridge owns these no-mutation facts.  Require every one again at the
    # package boundary rather than relying only on its top-level pass bit.
    expected_bridge_flags = bridge_adapter._static_flags()
    if any(
        bridge.get(name) is not expected
        for name, expected in expected_bridge_flags.items()
    ):
        raise HandoffContractError("provisional Full bridge safety flags drifted")


def _truth_binding(
    *,
    bridge: Mapping[str, Any],
    truth_path: Path,
    truth_loader: Any,
    predictor: Any | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    manifest, ranked, _by_candidate = truth_loader(truth_path, predictor=predictor)
    candidate = bridge["candidate_physics_sha256"]
    matches = [
        row for row in ranked if row.get("candidate_physics_sha256") == candidate
    ]
    if (
        len(matches) != 1
        or matches[0].get("truth_non_dominated_rank") != 0
        or matches[0].get("truth_row_sha256") != bridge.get("truth_row_sha256")
        or manifest.get("payload_sha256")
        != bridge.get("truth_promotion_payload_sha256")
    ):
        raise HandoffContractError(
            "bridge candidate is not an authenticated global rank-0 row"
        )
    artifact = _require_mapping(
        manifest.get("truth_validated_pareto_front"),
        "truth Pareto CSV",
    )
    relative = str(artifact.get("path") or "")
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or len(pure.parts) != 1
    ):
        raise HandoffContractError("truth Pareto CSV path is unsafe")
    csv_path = (truth_path.parent / relative).resolve(strict=True)
    csv_record = production._file_record(csv_path)
    if csv_record["sha256"] != artifact.get("sha256") or csv_record[
        "size_bytes"
    ] != artifact.get("size_bytes"):
        raise HandoffContractError("truth Pareto CSV bytes drifted")
    return manifest, matches[0], csv_path


def _validate_nds_binding(
    *,
    path: Path,
    bridge: Mapping[str, Any],
    enforce_source_name: bool,
) -> dict[str, Any]:
    receipt = production._validate_seal(
        production._read_json(path.resolve(strict=True)),
        terminal.NDS_RECEIPT_SCHEMA,
    )
    logical_ids = receipt.get("logical_authority_task_ids")
    collections = receipt.get("input_collections")
    if (
        not isinstance(logical_ids, list)
        or not logical_ids
        or logical_ids != sorted(set(logical_ids))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in logical_ids
        )
        or not isinstance(collections, list)
        or len(collections) != len(logical_ids)
        or receipt.get("truth_manifest") != bridge.get("truth_promotion_manifest")
        or receipt.get("global_nondominated_sort_performed") is not True
        or receipt.get("scheduler_mutation_performed") is not False
        or receipt.get("payload_sha256")
        != bridge.get("truth_nds_receipt_payload_sha256")
    ):
        raise HandoffContractError("global-NDS receipt contract drifted")
    authority = []
    matched = False
    for logical_id, value in zip(logical_ids, collections, strict=True):
        record = _require_mapping(value, "global-NDS input collection")
        source, authenticated = _source_record(record, "global-NDS input collection")
        del source
        authority.append(
            {
                "logical_authority_task_id": logical_id,
                "collection_payload_sha256": authenticated["sha256"],
            }
        )
        if logical_id == bridge.get(
            "logical_authority_task_id"
        ) and authenticated == bridge.get("standard_collection"):
            matched = True
    key = canonical_sha256(authority)[:16]
    if receipt.get("authority_key") != key or not matched:
        raise HandoffContractError(
            "global-NDS receipt does not bind the Standard authority"
        )
    if enforce_source_name:
        expected_snapshot_name = f"n{len(collections):02d}-{key}"
        if (
            path.name != f"{expected_snapshot_name}.receipt.json"
            or path.parent.joinpath(expected_snapshot_name).name
            != expected_snapshot_name
        ):
            raise HandoffContractError("global-NDS receipt snapshot identity drifted")
    return receipt


def _package_file_records(package_root: Path) -> list[dict[str, Any]]:
    names = []
    for child in package_root.iterdir():
        if child.is_symlink() or not child.is_file():
            raise HandoffContractError("package must contain only regular flat files")
        names.append(child.name)
    if set(names) != PAYLOAD_NAMES | {MANIFEST_NAME}:
        raise HandoffContractError("package file inventory is incomplete")
    return [_inventory_record(package_root / name) for name in sorted(PAYLOAD_NAMES)]


def _build_manifest(
    *,
    bridge_path: Path,
    bridge: Mapping[str, Any],
    inventory: list[dict[str, Any]],
    symmetric: Mapping[str, Any],
    full: Mapping[str, Any],
    truth_manifest: Mapping[str, Any],
    truth_row: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return production._seal(
        {
            "schema_version": PACKAGE_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "package_kind": "explicit_provisional_full_project_only",
            "canonical_package_schema": promotion.PACKAGE_SCHEMA,
            "goal_contract_schema": GOAL_CONTRACT_SCHEMA,
            "hard_spec": copy.deepcopy(GOAL_STAGE_SPEC),
            "hard_spec_sha256": GOAL_STAGE_SPEC_SHA256,
            "temperature_contract_sha256": (GOAL_TEMPERATURE_CONTRACT_SHA256),
            "candidate_physics_sha256": bridge["candidate_physics_sha256"],
            "logical_authority_task_id": bridge["logical_authority_task_id"],
            "standard_task_id": bridge["standard_task_id"],
            "provisional_full_task_id": bridge["provisional_full_task_id"],
            "solver_revision": bridge["solver_revision"],
            "library_revision": bridge["library_revision"],
            "fea_params_sha256": bridge["fea_params_sha256"],
            "effective_full_params_sha256": bridge["effective_full_params_sha256"],
            "fixed_boundary": copy.deepcopy(bridge["fixed_boundary"]),
            "fixed_identity_attestation": copy.deepcopy(
                bridge["fixed_identity_attestation"]
            ),
            "promotion_bridge": {
                "relative_path": "promotion_bridge.json",
                "source": production._file_record(bridge_path),
                "payload_sha256": bridge["payload_sha256"],
            },
            "source_authority": copy.deepcopy(dict(source_records)),
            "models": {
                "symmetric_model": copy.deepcopy(dict(symmetric)),
                "full_model": copy.deepcopy(dict(full)),
            },
            "result_trees": {
                "symmetric_model": {
                    "included_in_package": False,
                    "retained_remote_available": True,
                    "retained_remote_receipt": copy.deepcopy(
                        bridge["symmetric_model"]["retained_remote_receipt"]
                    ),
                },
                "full_model": {
                    "included_in_package": False,
                    "retained_remote_available": False,
                    "aedtresults": None,
                },
            },
            "actual_truth_global_nds": {
                "truth_manifest_schema": truth_manifest["schema_version"],
                "truth_manifest_payload_sha256": truth_manifest["payload_sha256"],
                "truth_row_sha256": truth_row["truth_row_sha256"],
                "truth_non_dominated_rank": 0,
                "global_nondominated_sort_performed": True,
                "standard_actual_truth": copy.deepcopy(bridge["standard_actual_truth"]),
                "provisional_full_actual_truth": copy.deepcopy(
                    bridge["provisional_full_actual_truth"]
                ),
            },
            "inventory": copy.deepcopy(inventory),
            "package_tree_sha256": canonical_sha256(inventory),
            "bridge_passed": True,
            "project_only_model_package_eligible": True,
            "production_eligible": True,
            "canonical_v1_promotion_allowed": False,
            "consumer_contract": {
                "schema_version": ("mft-goal-project-only-full-package-consumer-v2"),
                "package_root_binding": "manifest_parent",
                "bridge_reauthentication_required": True,
                "full_aedtresults_must_not_be_synthesized": True,
                "canonical_truth_full_collection_v1_must_not_be_emitted": (True),
                "canonical_truth_full_package_v1_must_not_be_emitted": True,
                "self_contained_deep_validation_supported": False,
                "external_authority_required_for_deep_reauthentication": (True),
                "package_directory_relocatable_with_external_authority": (True),
                "standalone_cross_machine_relocatable": False,
            },
            "created_at_utc": _utc_now(),
            **_static_flags(),
        }
    )


def _authenticated_bridge(
    path: Path,
    *,
    bridge_authenticator: Any,
    predictor: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    view = bridge_authenticator(path.resolve(strict=True), predictor=predictor)
    view_map = _require_mapping(view, "authenticated bridge view")
    bridge = _require_mapping(view_map.get("bridge"), "authenticated bridge")
    if view_map.get("production_eligible") is not True:
        raise HandoffContractError(
            "authenticated bridge view is not production-eligible"
        )
    _validate_bridge_gate(bridge)
    return view_map, bridge


def build_package(
    *,
    bridge_receipt_path: Path,
    output: Path,
    bridge_authenticator: Any = bridge_adapter.authenticate_bridge,
    truth_loader: Any = promotion._load_truth_manifest,
    remote_fetcher: Any = production._fetch_remote_to_path,
    predictor: Any | None = None,
) -> Path:
    """Build an immutable project-only package after full reauthentication."""
    bridge_path = bridge_receipt_path.resolve(strict=True)
    _view, bridge = _authenticated_bridge(
        bridge_path,
        bridge_authenticator=bridge_authenticator,
        predictor=predictor,
    )
    source_paths: dict[str, Path] = {}
    source_records: dict[str, dict[str, Any]] = {}
    source_fields = {
        "standard_collection": "standard_collection",
        "provisional_full_success": "provisional_full_success",
        "canonical_full_plan": "canonical_full_plan",
        "truth_promotion_manifest": "truth_promotion_manifest",
        "truth_nds_receipt": "truth_nds_receipt",
    }
    for key, field in source_fields.items():
        source_paths[key], source_records[key] = _source_record(bridge.get(field), key)
    truth_manifest, truth_row, csv_path = _truth_binding(
        bridge=bridge,
        truth_path=source_paths["truth_promotion_manifest"],
        truth_loader=truth_loader,
        predictor=predictor,
    )
    source_records["truth_validated_pareto_front"] = production._file_record(csv_path)
    _validate_nds_binding(
        path=source_paths["truth_nds_receipt"],
        bridge=bridge,
        enforce_source_name=True,
    )
    full_source, full_source_record = _source_record(
        _require_mapping(bridge["full_model"], "Full model").get("local_artifact"),
        "provisional Full AEDT",
    )
    standard_collection = production._read_json(source_paths["standard_collection"])
    scheduler_url = str(standard_collection.get("scheduler_url") or "").rstrip("/")
    if scheduler_url != diagnostic.DIAGNOSTIC_SCHEDULER_URL.rstrip("/"):
        raise HandoffContractError(
            "project-only package requires diagnostic Scheduler port 8002"
        )
    target = output.resolve()
    if target.exists():
        if target.is_dir() and not target.is_symlink():
            authenticated = authenticate_package(
                target / MANIFEST_NAME,
                bridge_authenticator=bridge_authenticator,
                truth_loader=truth_loader,
                predictor=predictor,
            )
            if authenticated["manifest"]["promotion_bridge"][
                "source"
            ] == production._file_record(bridge_path):
                return target
        raise HandoffContractError(f"immutable package output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.{os.getpid()}.",
            suffix=".tmp",
            dir=target.parent,
        )
    )
    try:
        evidence_sources = {
            "promotion_bridge.json": bridge_path,
            "standard_collection.json": source_paths["standard_collection"],
            "provisional_full_success.json": source_paths["provisional_full_success"],
            "canonical_full_plan.json": source_paths["canonical_full_plan"],
            "actual_truth_pareto_manifest.json": source_paths[
                "truth_promotion_manifest"
            ],
            "actual_truth_nds_receipt.json": source_paths["truth_nds_receipt"],
            "truth_validated_pareto_front.csv": csv_path,
        }
        for name, source in evidence_sources.items():
            _copy_exact(source, staging / name)
        symmetric_receipt = _require_mapping(
            _require_mapping(bridge["symmetric_model"], "symmetric model").get(
                "retained_remote_receipt"
            ),
            "symmetric retained receipt",
        )
        symmetric = promotion._package_artifact(
            scheduler_url=scheduler_url,
            task_id=int(bridge["standard_task_id"]),
            receipt=symmetric_receipt,
            destination=staging / "symmetric_model.aedt",
            final_path=target / "symmetric_model.aedt",
            remote_fetcher=remote_fetcher,
            full_model=0,
            thermal_symmetry="eighth",
        )
        symmetric = dict(symmetric)
        symmetric.pop("local_absolute_path")
        symmetric["relative_path"] = "symmetric_model.aedt"
        _copy_exact(full_source, staging / "full_model.aedt")
        full = {
            "relative_path": "full_model.aedt",
            "sha256": full_source_record["sha256"],
            "size_bytes": full_source_record["size_bytes"],
            "source_task_id": bridge["provisional_full_task_id"],
            "source_local_artifact": copy.deepcopy(full_source_record),
            "source_remote_receipt": copy.deepcopy(
                bridge["full_model"]["retained_remote_receipt"]
            ),
            "full_model": 1,
            "thermal_symmetry": "full",
            "aedtresults": None,
        }
        inventory = [
            _inventory_record(staging / name) for name in sorted(PAYLOAD_NAMES)
        ]
        manifest = _build_manifest(
            bridge_path=bridge_path,
            bridge=bridge,
            inventory=inventory,
            symmetric=symmetric,
            full=full,
            truth_manifest=truth_manifest,
            truth_row=truth_row,
            source_records=source_records,
        )
        production._write_immutable_json(staging / MANIFEST_NAME, manifest)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    authenticate_package(
        target / MANIFEST_NAME,
        bridge_authenticator=bridge_authenticator,
        truth_loader=truth_loader,
        predictor=predictor,
    )
    return target


def authenticate_package(
    manifest_path: Path,
    *,
    bridge_authenticator: Any = bridge_adapter.authenticate_bridge,
    truth_loader: Any = promotion._load_truth_manifest,
    predictor: Any | None = None,
) -> dict[str, Any]:
    """Deeply validate a package, its source bridge, and all packaged bytes."""
    resolved = manifest_path.resolve(strict=True)
    if resolved.name != MANIFEST_NAME or resolved.is_symlink():
        raise HandoffContractError("project-only package manifest path drifted")
    package_root = resolved.parent
    manifest = production._validate_seal(
        production._read_json(resolved), PACKAGE_SCHEMA
    )
    if (
        set(manifest) != PACKAGE_FIELDS
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("package_kind") != "explicit_provisional_full_project_only"
        or manifest.get("canonical_package_schema") != promotion.PACKAGE_SCHEMA
        or manifest.get("goal_contract_schema") != GOAL_CONTRACT_SCHEMA
        or manifest.get("hard_spec") != GOAL_STAGE_SPEC
        or manifest.get("hard_spec_sha256") != GOAL_STAGE_SPEC_SHA256
        or manifest.get("temperature_contract_sha256")
        != GOAL_TEMPERATURE_CONTRACT_SHA256
        or manifest.get("bridge_passed") is not True
        or manifest.get("project_only_model_package_eligible") is not True
        or manifest.get("production_eligible") is not True
        or manifest.get("canonical_v1_promotion_allowed") is not False
        or any(
            manifest.get(name) is not expected
            for name, expected in _static_flags().items()
        )
        or not str(manifest.get("created_at_utc") or "")
    ):
        raise HandoffContractError("project-only package manifest contract drifted")
    inventory = manifest.get("inventory")
    observed_inventory = _package_file_records(package_root)
    if (
        not isinstance(inventory, list)
        or inventory != observed_inventory
        or manifest.get("package_tree_sha256") != canonical_sha256(observed_inventory)
    ):
        raise HandoffContractError("project-only package inventory drifted")
    bridge_meta = _require_mapping(
        manifest.get("promotion_bridge"), "package promotion bridge"
    )
    if bridge_meta.get("relative_path") != "promotion_bridge.json":
        raise HandoffContractError("package promotion bridge path drifted")
    packaged_bridge = package_root / "promotion_bridge.json"
    source_bridge_path, source_bridge_record = _source_record(
        bridge_meta.get("source"), "source promotion bridge"
    )
    if (
        production._sha256_file(packaged_bridge) != source_bridge_record["sha256"]
        or packaged_bridge.stat().st_size != source_bridge_record["size_bytes"]
    ):
        raise HandoffContractError("packaged promotion bridge bytes drifted")
    _view, bridge = _authenticated_bridge(
        packaged_bridge,
        bridge_authenticator=bridge_authenticator,
        predictor=predictor,
    )
    if bridge_meta.get("payload_sha256") != bridge.get(
        "payload_sha256"
    ) or production._sha256_file(source_bridge_path) != production._sha256_file(
        packaged_bridge
    ):
        raise HandoffContractError("package bridge authority drifted")
    expected_source_records = {
        "standard_collection": bridge["standard_collection"],
        "provisional_full_success": bridge["provisional_full_success"],
        "canonical_full_plan": bridge["canonical_full_plan"],
        "truth_promotion_manifest": bridge["truth_promotion_manifest"],
        "truth_nds_receipt": bridge["truth_nds_receipt"],
    }
    source_authority = _require_mapping(
        manifest.get("source_authority"), "package source authority"
    )
    csv_source = source_authority.get("truth_validated_pareto_front")
    if set(source_authority) != set(expected_source_records) | {
        "truth_validated_pareto_front"
    } or any(
        source_authority.get(name) != record
        for name, record in expected_source_records.items()
    ):
        raise HandoffContractError("package source authority records drifted")
    evidence_names = {
        "standard_collection": "standard_collection.json",
        "provisional_full_success": "provisional_full_success.json",
        "canonical_full_plan": "canonical_full_plan.json",
        "truth_promotion_manifest": "actual_truth_pareto_manifest.json",
        "truth_nds_receipt": "actual_truth_nds_receipt.json",
        "truth_validated_pareto_front": ("truth_validated_pareto_front.csv"),
    }
    for key, name in evidence_names.items():
        _source, record = _source_record(
            source_authority.get(key), f"package source {key}"
        )
        packaged = package_root / name
        if (
            production._sha256_file(packaged) != record["sha256"]
            or packaged.stat().st_size != record["size_bytes"]
        ):
            raise HandoffContractError(f"packaged evidence differs from source: {key}")
    truth_path = package_root / "actual_truth_pareto_manifest.json"
    truth_manifest, truth_row, csv_path = _truth_binding(
        bridge=bridge,
        truth_path=truth_path,
        truth_loader=truth_loader,
        predictor=predictor,
    )
    if csv_path != package_root / "truth_validated_pareto_front.csv" or not isinstance(
        csv_source, Mapping
    ):
        raise HandoffContractError("packaged truth Pareto CSV binding drifted")
    _validate_nds_binding(
        path=package_root / "actual_truth_nds_receipt.json",
        bridge=bridge,
        enforce_source_name=False,
    )
    models = _require_mapping(manifest.get("models"), "package models")
    symmetric = _require_mapping(
        models.get("symmetric_model"), "packaged symmetric model"
    )
    full = _require_mapping(models.get("full_model"), "packaged Full model")
    symmetric_receipt = bridge["symmetric_model"]["retained_remote_receipt"]
    full_source = bridge["full_model"]["local_artifact"]
    inventory_by_name = {item["relative_path"]: item for item in observed_inventory}
    symmetric_inventory = inventory_by_name["symmetric_model.aedt"]
    full_inventory = inventory_by_name["full_model.aedt"]
    if (
        symmetric_inventory["sha256"] != symmetric_receipt["artifact_sha256"]
        or symmetric_inventory["size_bytes"] != symmetric_receipt["artifact_size_bytes"]
        or full_inventory["sha256"] != full_source["sha256"]
        or full_inventory["size_bytes"] != full_source["size_bytes"]
    ):
        raise HandoffContractError(
            "packaged AEDT bytes drifted from authenticated source"
        )
    if (
        symmetric.get("relative_path") != "symmetric_model.aedt"
        or "local_absolute_path" in symmetric
        or symmetric.get("sha256") != symmetric_receipt["artifact_sha256"]
        or symmetric.get("size_bytes") != symmetric_receipt["artifact_size_bytes"]
        or symmetric.get("source_task_id") != bridge["standard_task_id"]
        or symmetric.get("full_model") != 0
        or symmetric.get("thermal_symmetry") != "eighth"
        or full.get("relative_path") != "full_model.aedt"
        or "local_absolute_path" in full
        or full.get("sha256") != full_source["sha256"]
        or full.get("size_bytes") != full_source["size_bytes"]
        or full.get("source_task_id") != bridge["provisional_full_task_id"]
        or full.get("source_local_artifact") != full_source
        or full.get("source_remote_receipt")
        != bridge["full_model"]["retained_remote_receipt"]
        or full.get("full_model") != 1
        or full.get("thermal_symmetry") != "full"
        or full.get("aedtresults") is not None
    ):
        raise HandoffContractError("packaged AEDT model authority drifted")
    results = _require_mapping(manifest.get("result_trees"), "package result trees")
    if (
        results.get("symmetric_model", {}).get("included_in_package") is not False
        or results.get("symmetric_model", {}).get("retained_remote_available")
        is not True
        or results.get("symmetric_model", {}).get("retained_remote_receipt")
        != symmetric_receipt
        or results.get("full_model")
        != {
            "included_in_package": False,
            "retained_remote_available": False,
            "aedtresults": None,
        }
    ):
        raise HandoffContractError(
            "project-only package result-tree declaration drifted"
        )
    actual = _require_mapping(
        manifest.get("actual_truth_global_nds"),
        "package actual truth global NDS",
    )
    if (
        actual.get("truth_manifest_schema") != truth_manifest["schema_version"]
        or actual.get("truth_manifest_payload_sha256")
        != truth_manifest["payload_sha256"]
        or actual.get("truth_row_sha256") != truth_row["truth_row_sha256"]
        or actual.get("truth_non_dominated_rank") != 0
        or actual.get("global_nondominated_sort_performed") is not True
        or actual.get("standard_actual_truth") != bridge["standard_actual_truth"]
        or actual.get("provisional_full_actual_truth")
        != bridge["provisional_full_actual_truth"]
        or manifest.get("candidate_physics_sha256")
        != bridge["candidate_physics_sha256"]
        or manifest.get("logical_authority_task_id")
        != bridge["logical_authority_task_id"]
        or manifest.get("standard_task_id") != bridge["standard_task_id"]
        or manifest.get("provisional_full_task_id")
        != bridge["provisional_full_task_id"]
        or manifest.get("solver_revision") != bridge["solver_revision"]
        or manifest.get("library_revision") != bridge["library_revision"]
        or manifest.get("fea_params_sha256") != bridge["fea_params_sha256"]
        or manifest.get("effective_full_params_sha256")
        != bridge["effective_full_params_sha256"]
        or manifest.get("fixed_boundary") != bridge["fixed_boundary"]
        or manifest.get("fixed_identity_attestation")
        != bridge["fixed_identity_attestation"]
    ):
        raise HandoffContractError(
            "project-only package actual-truth authority drifted"
        )
    consumer = _require_mapping(
        manifest.get("consumer_contract"), "package consumer contract"
    )
    if (
        consumer.get("package_root_binding") != "manifest_parent"
        or consumer.get("bridge_reauthentication_required") is not True
        or consumer.get("full_aedtresults_must_not_be_synthesized") is not True
        or consumer.get("canonical_truth_full_collection_v1_must_not_be_emitted")
        is not True
        or consumer.get("canonical_truth_full_package_v1_must_not_be_emitted")
        is not True
        or consumer.get("self_contained_deep_validation_supported") is not False
        or consumer.get("external_authority_required_for_deep_reauthentication")
        is not True
        or consumer.get("package_directory_relocatable_with_external_authority")
        is not True
        or consumer.get("standalone_cross_machine_relocatable") is not False
    ):
        raise HandoffContractError("project-only package consumer contract drifted")
    return {
        "schema_version": (
            "mft-goal-provisional-full-explicit-package-authenticated-v2"
        ),
        "manifest_path": resolved,
        "package_root": package_root,
        "manifest": manifest,
        "bridge": bridge,
        "production_eligible": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/validate an explicit provisional Full package v2"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--bridge-receipt", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        path = build_package(
            bridge_receipt_path=args.bridge_receipt,
            output=args.output,
        )
        print(path / MANIFEST_NAME)
        return 0
    authenticated = authenticate_package(args.manifest)
    print(authenticated["manifest_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
