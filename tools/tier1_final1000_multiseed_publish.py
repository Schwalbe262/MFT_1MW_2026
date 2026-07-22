"""Authenticate and publish one prepared Final1000 multi-seed stage.

The generic delta publisher deliberately defaults to the original Current7
production parent.  A four-stage multi-seed release has four different,
authenticated parents, so this entry point derives the selected parent only
after validating the complete sealed preparation and its 30811ea release
gate.  Dry-run is the default and does not construct a remote transport.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import (
        atomic_json,
        read_json,
        sha256_file,
    )
    from tier1_corrected_current7_slurm_publish import (
        load_bundle_plan,
        validate_publication_receipt,
    )
    from tier1_final1000_delta_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        DELTA_PLAN_RECEIPT_SCHEMA,
        ParentIdentity,
        load_delta_plan,
        publish_delta,
        scheduler_delta_transport,
    )
    from tier1_final1000_multiseed_release import (
        MULTISTAGE_PREPARATION_SCHEMA,
        RELEASE_PREPARATION_SCHEMA,
        SCIENCE_BASE_REVISION,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import (
        atomic_json,
        read_json,
        sha256_file,
    )
    from tools.tier1_corrected_current7_slurm_publish import (
        load_bundle_plan,
        validate_publication_receipt,
    )
    from tools.tier1_final1000_delta_publish import (
        DEFAULT_ACCOUNTS,
        DEFAULT_SCHEDULER_SOURCE,
        DELTA_PLAN_RECEIPT_SCHEMA,
        ParentIdentity,
        load_delta_plan,
        publish_delta,
        scheduler_delta_transport,
    )
    from tools.tier1_final1000_multiseed_release import (
        MULTISTAGE_PREPARATION_SCHEMA,
        RELEASE_PREPARATION_SCHEMA,
        SCIENCE_BASE_REVISION,
    )


MULTISEED_PUBLICATION_SCHEMA = "mft-tier1-final1000-multiseed-publication-v1"
REMOTE_GATE_SCHEMA = "mft-tier1-semlock-successor-remote-release-v1"
REMOTE_GATE_SOURCE_REVISION = "30811ea159c6379647291e67f1961d74bcb23658"
REMOTE_GATE_STATUS = "passed_remote_publish_no_write_replay_and_two_pass_full_sha"
EXPECTED_STAGE_IDS = frozenset(
    {
        "entry-1200-t125",
        "bridge-1150-t115",
        "close-1075-t107p5",
        "final-1000-t100",
    }
)
_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class AuthenticatedStage:
    stage_id: str
    preparation_receipt_path: Path
    preparation_receipt_file_sha256: str
    preparation_receipt_sha256: str
    stage_receipt_path: Path
    stage_receipt_file_sha256: str
    stage_receipt_sha256: str
    checkout_revision: str
    release_gate_path: Path
    release_gate_file_sha256: str
    release_gate_evidence_sha256: str
    delta_plan_path: Path
    delta_plan_file_sha256: str
    delta_plan_receipt_path: Path
    delta_plan_receipt_file_sha256: str
    delta_plan_receipt_sha256: str
    parent_identity: ParentIdentity
    bundle_id: str
    contract_sha256: str
    remote_bundle: str


def _validate_seal(value: Mapping[str, Any], *, field: str, label: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(unsigned):
        raise RuntimeError(f"{label} seal mismatch")


def _contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"{label} escaped its authenticated root") from exc
    return resolved


def _parent_identity(plan_path: Path, publication_path: Path) -> ParentIdentity:
    plan, manifest, _ = load_bundle_plan(plan_path)
    publication = validate_publication_receipt(
        read_json(publication_path), plan=plan, manifest=manifest
    )
    if publication.get("publication_complete") is not True:
        raise RuntimeError("parent publication is not complete")
    source_map_path = Path(str(plan["local_sources"])).resolve(strict=True)
    return ParentIdentity(
        bundle_id=str(plan["bundle_id"]),
        contract_sha256=str(plan["contract_sha256"]),
        manifest_sha256=str(plan["bundle_manifest_sha256"]),
        plan_sha256=sha256_file(plan_path),
        source_map_sha256=sha256_file(source_map_path),
        publication_file_sha256=sha256_file(publication_path),
        remote_bundle=str(plan["remote_bundle"]),
    )


def _authenticate_gate(
    preparation: Mapping[str, Any], gate_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Mapping[str, Any]]]:
    gate_path = gate_path.resolve(strict=True)
    if (
        gate_path.name != "remote_release_gate.json"
        or gate_path.parent.name != "evidence"
    ):
        raise RuntimeError("remote release gate path is not canonical")
    release_root = gate_path.parent.parent.resolve(strict=True)
    gate = read_json(gate_path)
    gate_unsigned = {
        key: value for key, value in gate.items() if key != "evidence_sha256"
    }
    rows = gate.get("stages")
    if (
        gate.get("schema_version") != REMOTE_GATE_SCHEMA
        or gate.get("source_revision") != REMOTE_GATE_SOURCE_REVISION
        or gate.get("status") != REMOTE_GATE_STATUS
        or gate.get("evidence_sha256") != canonical_sha256(gate_unsigned)
        or not isinstance(rows, list)
        or len(rows) != 4
    ):
        raise RuntimeError(
            "30811ea four-stage remote release gate is not authenticated"
        )
    if (
        preparation.get("parent_release_gate_path") != str(gate_path)
        or preparation.get("parent_release_gate_file_sha256") != sha256_file(gate_path)
        or preparation.get("parent_release_gate_evidence_sha256")
        != gate.get("evidence_sha256")
    ):
        raise RuntimeError("preparation/remote-gate file identity mismatch")
    required_true = (
        "all_family_thread_bindings_authenticated",
        "all_manifest_files_full_sha_verified_twice_independently",
        "all_pass2_already_ready_without_remote_write",
        "all_permissions_and_runtime_packages_verified",
        "all_ready_written_last",
        "all_remote_relocations_authenticated",
        "final_warm_handoff_authenticated",
    )
    required_zero = (
        "aedt_count",
        "fea_count",
        "controller_mutation_count",
        "scheduler_post_count",
        "scheduler_cancel_count",
        "scheduler_preempt_count",
    )
    if any(gate.get(field) is not True for field in required_true) or any(
        gate.get(field) != 0 for field in required_zero
    ):
        raise RuntimeError("remote release gate safety evidence is incomplete")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("remote release gate stage row is invalid")
        stage_id = str(row.get("stage_id") or "")
        if stage_id in indexed:
            raise RuntimeError("remote release gate has duplicate stage identity")
        indexed[stage_id] = row
    if set(indexed) != EXPECTED_STAGE_IDS:
        raise RuntimeError("remote release gate stage cardinality/identity mismatch")
    return gate, release_root, indexed


def _authenticate_stage(
    *,
    stage_id: str,
    embedded: Mapping[str, Any],
    gate_row: Mapping[str, Any],
    preparation_root: Path,
    release_root: Path,
    expected_checkout_revision: str,
) -> AuthenticatedStage:
    stage_root = preparation_root / stage_id
    stage_receipt_path = _contained(
        stage_root / "release_preparation_receipt.json",
        preparation_root,
        "stage preparation receipt",
    )
    stage_receipt = read_json(stage_receipt_path)
    if stage_receipt != dict(embedded):
        raise RuntimeError("embedded/stage preparation receipt file mismatch")
    _validate_seal(stage_receipt, field="receipt_sha256", label="stage preparation")
    checkout = stage_receipt.get("checkout") or {}
    if (
        stage_receipt.get("schema_version") != RELEASE_PREPARATION_SCHEMA
        or checkout.get("revision") != expected_checkout_revision
        or checkout.get("clean") is not True
        or checkout.get("science_base_revision") != SCIENCE_BASE_REVISION
        or stage_receipt.get("ready_reused") is not False
        or stage_receipt.get("remote_write_performed") is not False
        or stage_receipt.get("scheduler_submission_performed") is not False
        or (stage_receipt.get("isolated_import_smoke") or {}).get("passed") is not True
        or (stage_receipt.get("isolated_import_smoke") or {}).get("isolated")
        is not True
    ):
        raise RuntimeError("stage preparation safety/checkout identity mismatch")

    parent_plan_path = _contained(
        Path(str(gate_row.get("plan_path") or "")),
        release_root,
        "parent plan",
    )
    publication_row = gate_row.get("publication") or {}
    pass2 = publication_row.get("pass2") or {}
    parent_publication_path = _contained(
        Path(str(pass2.get("receipt_path") or "")),
        release_root,
        "parent publication receipt",
    )
    if (
        sha256_file(parent_plan_path) != gate_row.get("plan_file_sha256")
        or sha256_file(parent_publication_path) != pass2.get("receipt_file_sha256")
        or pass2.get("already_ready") is not True
        or pass2.get("remote_write_performed") is not False
    ):
        raise RuntimeError("remote gate parent file identity drifted")
    parent_publication = read_json(parent_publication_path)
    if pass2.get("receipt_sha256") != parent_publication.get("receipt_sha256"):
        raise RuntimeError("remote gate parent publication seal mismatch")
    identity = _parent_identity(parent_plan_path, parent_publication_path)
    if stage_receipt.get("parent_identity") != asdict(identity):
        raise RuntimeError("stage preparation dynamic parent identity mismatch")
    gate_identity = {
        "bundle_id": gate_row.get("bundle_id"),
        "contract_sha256": gate_row.get("contract_sha256"),
        "manifest_sha256": gate_row.get("manifest_file_sha256"),
        "plan_sha256": gate_row.get("plan_file_sha256"),
        "source_map_sha256": gate_row.get("source_map_file_sha256"),
        "publication_file_sha256": pass2.get("receipt_file_sha256"),
        "remote_bundle": gate_row.get("remote_bundle"),
    }
    if gate_identity != asdict(identity):
        raise RuntimeError("selected stage/remote-gate parent identity mismatch")

    candidate_path = _contained(
        Path(str(stage_receipt.get("candidate_plan") or "")),
        stage_root,
        "candidate plan",
    )
    expected_candidate_path = (stage_root / "candidate" / "offload_plan.json").resolve()
    if candidate_path != expected_candidate_path:
        raise RuntimeError("candidate plan path is not canonical for its stage")
    candidate_plan, candidate_manifest, _ = load_bundle_plan(candidate_path)
    if candidate_plan.get("bundle_id") != stage_receipt.get(
        "candidate_bundle_id"
    ) or candidate_manifest.get("contract_sha256") != stage_receipt.get(
        "candidate_contract_sha256"
    ):
        raise RuntimeError("candidate plan/stage receipt identity mismatch")

    delta_plan_path = _contained(
        Path(str(stage_receipt.get("delta_plan") or "")),
        stage_root,
        "delta plan",
    )
    expected_delta_path = (
        stage_root
        / "delta"
        / str(stage_receipt.get("delta_bundle_id") or "")
        / "offload_plan.json"
    ).resolve()
    if delta_plan_path != expected_delta_path:
        raise RuntimeError("delta plan path is not canonical for its stage")
    delta_receipt_path = _contained(
        delta_plan_path.with_name("delta_plan_receipt.json"),
        stage_root,
        "delta plan receipt",
    )
    delta_receipt = read_json(delta_receipt_path)
    _validate_seal(delta_receipt, field="receipt_sha256", label="delta plan receipt")
    if (
        delta_receipt.get("schema_version") != DELTA_PLAN_RECEIPT_SCHEMA
        or delta_receipt.get("receipt_sha256")
        != stage_receipt.get("delta_plan_receipt_sha256")
        or delta_receipt.get("parent_identity") != asdict(identity)
    ):
        raise RuntimeError("stage/delta-plan receipt identity mismatch")
    delta_plan, delta_manifest, *_ = load_delta_plan(
        delta_plan_path, required_parent=identity
    )
    if (
        delta_plan.get("bundle_id") != stage_receipt.get("delta_bundle_id")
        or delta_manifest.get("contract_sha256")
        != stage_receipt.get("delta_contract_sha256")
        or delta_plan.get("remote_bundle") != delta_receipt.get("remote_bundle")
        or (delta_plan.get("delta_publication") or {}).get("candidate_plan_sha256")
        != sha256_file(candidate_path)
    ):
        raise RuntimeError("stage/delta plan identity mismatch")
    return AuthenticatedStage(
        stage_id=stage_id,
        preparation_receipt_path=preparation_root / "release_preparation_receipt.json",
        preparation_receipt_file_sha256="",
        preparation_receipt_sha256="",
        stage_receipt_path=stage_receipt_path,
        stage_receipt_file_sha256=sha256_file(stage_receipt_path),
        stage_receipt_sha256=str(stage_receipt["receipt_sha256"]),
        checkout_revision=expected_checkout_revision,
        release_gate_path=release_root / "evidence" / "remote_release_gate.json",
        release_gate_file_sha256="",
        release_gate_evidence_sha256="",
        delta_plan_path=delta_plan_path,
        delta_plan_file_sha256=sha256_file(delta_plan_path),
        delta_plan_receipt_path=delta_receipt_path,
        delta_plan_receipt_file_sha256=sha256_file(delta_receipt_path),
        delta_plan_receipt_sha256=str(delta_receipt["receipt_sha256"]),
        parent_identity=identity,
        bundle_id=str(delta_plan["bundle_id"]),
        contract_sha256=str(delta_plan["contract_sha256"]),
        remote_bundle=str(delta_plan["remote_bundle"]),
    )


def authenticate_multiseed_stage(
    preparation_receipt_path: Path,
    *,
    stage_id: str,
    expected_checkout_revision: str,
) -> AuthenticatedStage:
    """Authenticate all four prepared stages and return the selected one."""

    if _REVISION.fullmatch(expected_checkout_revision) is None:
        raise RuntimeError("an exact lowercase 40-hex checkout revision is required")
    if stage_id not in EXPECTED_STAGE_IDS:
        raise RuntimeError("unknown Final1000 multi-seed stage")
    preparation_receipt_path = preparation_receipt_path.resolve(strict=True)
    if preparation_receipt_path.name != "release_preparation_receipt.json":
        raise RuntimeError("four-stage preparation receipt path is not canonical")
    preparation_root = preparation_receipt_path.parent.resolve(strict=True)
    preparation = read_json(preparation_receipt_path)
    _validate_seal(preparation, field="receipt_sha256", label="four-stage preparation")
    stages = preparation.get("stages")
    if (
        preparation.get("schema_version") != MULTISTAGE_PREPARATION_SCHEMA
        or preparation.get("stage_count") != 4
        or not isinstance(stages, Mapping)
        or set(stages) != EXPECTED_STAGE_IDS
        or preparation.get("ready_reused") is not False
        or preparation.get("remote_write_performed") is not False
        or preparation.get("scheduler_submission_performed") is not False
    ):
        raise RuntimeError("four-stage preparation receipt is not authenticated")
    gate_path = Path(str(preparation.get("parent_release_gate_path") or ""))
    gate, release_root, gate_rows = _authenticate_gate(preparation, gate_path)
    if any(
        path.name.upper() in {"READY", "READY.JSON"}
        for path in preparation_root.rglob("*")
    ):
        raise RuntimeError("prepared multi-seed release attempted to reuse READY")

    authenticated: dict[str, AuthenticatedStage] = {}
    for current_stage in sorted(EXPECTED_STAGE_IDS):
        embedded = stages[current_stage]
        if not isinstance(embedded, Mapping):
            raise RuntimeError("stage preparation receipt is invalid")
        authenticated[current_stage] = _authenticate_stage(
            stage_id=current_stage,
            embedded=embedded,
            gate_row=gate_rows[current_stage],
            preparation_root=preparation_root,
            release_root=release_root,
            expected_checkout_revision=expected_checkout_revision,
        )
    selected = authenticated[stage_id]
    return AuthenticatedStage(
        **{
            **selected.__dict__,
            "preparation_receipt_path": preparation_receipt_path,
            "preparation_receipt_file_sha256": sha256_file(preparation_receipt_path),
            "preparation_receipt_sha256": str(preparation["receipt_sha256"]),
            "release_gate_path": Path(
                str(preparation["parent_release_gate_path"])
            ).resolve(strict=True),
            "release_gate_file_sha256": sha256_file(
                Path(str(preparation["parent_release_gate_path"])).resolve(strict=True)
            ),
            "release_gate_evidence_sha256": str(gate["evidence_sha256"]),
        }
    )


def publish_authenticated_stage(
    authenticated: AuthenticatedStage,
    *,
    apply: bool = False,
    transport: Any | None = None,
) -> dict[str, Any]:
    """Publish only after authentication, always with the dynamic parent guard."""

    publication = publish_delta(
        authenticated.delta_plan_path,
        apply=apply,
        transport=transport,
        required_parent=authenticated.parent_identity,
    )
    unsigned = {
        "schema_version": MULTISEED_PUBLICATION_SCHEMA,
        "stage_id": authenticated.stage_id,
        "checkout_revision": authenticated.checkout_revision,
        "preparation_receipt_path": str(authenticated.preparation_receipt_path),
        "preparation_receipt_file_sha256": (
            authenticated.preparation_receipt_file_sha256
        ),
        "preparation_receipt_sha256": authenticated.preparation_receipt_sha256,
        "stage_receipt_path": str(authenticated.stage_receipt_path),
        "stage_receipt_file_sha256": authenticated.stage_receipt_file_sha256,
        "stage_receipt_sha256": authenticated.stage_receipt_sha256,
        "release_gate_path": str(authenticated.release_gate_path),
        "release_gate_file_sha256": authenticated.release_gate_file_sha256,
        "release_gate_evidence_sha256": authenticated.release_gate_evidence_sha256,
        "delta_plan_path": str(authenticated.delta_plan_path),
        "delta_plan_file_sha256": authenticated.delta_plan_file_sha256,
        "delta_plan_receipt_path": str(authenticated.delta_plan_receipt_path),
        "delta_plan_receipt_file_sha256": (
            authenticated.delta_plan_receipt_file_sha256
        ),
        "delta_plan_receipt_sha256": authenticated.delta_plan_receipt_sha256,
        "parent_identity": asdict(authenticated.parent_identity),
        "bundle_id": authenticated.bundle_id,
        "contract_sha256": authenticated.contract_sha256,
        "remote_bundle": authenticated.remote_bundle,
        "apply": bool(apply),
        "ready_reused_from_parent": False,
        "scheduler_submission_performed": False,
        "publication": publication,
        "remote_write_performed": bool(publication["remote_write_performed"]),
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-receipt", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(EXPECTED_STAGE_IDS), required=True)
    parser.add_argument("--expected-checkout-revision", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    parser.add_argument("--account", default="harry261")
    parser.add_argument("--receipt-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.apply and args.receipt_out is None:
        raise RuntimeError("--apply requires an explicit --receipt-out")
    if not args.apply and args.receipt_out is not None:
        raise RuntimeError("--receipt-out is only valid with --apply")
    if args.receipt_out is not None and args.receipt_out.exists():
        raise RuntimeError("publication receipt output already exists")
    authenticated = authenticate_multiseed_stage(
        args.preparation_receipt,
        stage_id=args.stage,
        expected_checkout_revision=args.expected_checkout_revision,
    )
    if args.apply:
        with scheduler_delta_transport(
            accounts_path=args.accounts,
            scheduler_source=args.scheduler_source,
            account_name=args.account,
        ) as transport:
            output = publish_authenticated_stage(
                authenticated, apply=True, transport=transport
            )
        atomic_json(args.receipt_out, output)
    else:
        output = publish_authenticated_stage(authenticated, apply=False)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
