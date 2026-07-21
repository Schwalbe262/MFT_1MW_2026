"""Plan and publish a small final1000 delta over the immutable Current7 bundle.

The input to ``plan`` is an ordinary, content-addressed Current7 child plan and
its complete source map.  The resulting plan adds authenticated parent lineage
and classifies every child file as either inherited or delta.  Publication is
dry-run by default.  With ``--apply``, inherited files and the pinned Python
runtime are cloned inside GPFS by reflink (preferred) or hardlink; only changed
or new size-bounded files cross the network.

The publisher authenticates the live parent READY, manifest, runtime, and file
inventory before its first remote write.  It authenticates the complete child
twice before writing READY, writes READY last, and then performs only an atomic
same-filesystem directory promotion.  A deterministic incoming directory and
sealed journal make interrupted publication resumable and fail closed.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
from typing import Any, Iterator, Mapping, Protocol, Sequence

try:
    from tier1_corrected_current7_receipt import canonical_sha256
    from tier1_corrected_current7_slurm_bundle import (
        READY_SCHEMA,
        atomic_json,
        read_json,
        sha256_file,
    )
    from tier1_corrected_current7_slurm_publish import (
        PUBLICATION_RECEIPT_SCHEMA,
        SSHPublicationTransport,
        _ready_matches,
        load_bundle_plan,
        validate_publication_receipt,
    )
except ImportError:  # pragma: no cover - repository import path
    from tools.tier1_corrected_current7_receipt import canonical_sha256
    from tools.tier1_corrected_current7_slurm_bundle import (
        READY_SCHEMA,
        atomic_json,
        read_json,
        sha256_file,
    )
    from tools.tier1_corrected_current7_slurm_publish import (
        PUBLICATION_RECEIPT_SCHEMA,
        SSHPublicationTransport,
        _ready_matches,
        load_bundle_plan,
        validate_publication_receipt,
    )


DELTA_SCHEMA = "mft-tier1-final1000-delta-publication-v1"
DELTA_PLAN_RECEIPT_SCHEMA = "mft-tier1-final1000-delta-plan-receipt-v1"
DELTA_JOURNAL_SCHEMA = "mft-tier1-final1000-delta-journal-v1"
DEFAULT_MAX_DELTA_FILE_BYTES = 16 * 1024**2
DEFAULT_MAX_DELTA_TOTAL_BYTES = 64 * 1024**2
DEFAULT_ACCOUNTS = Path(r"Y:\runtime\slurm_scheduler\config\accounts.yaml")
DEFAULT_SCHEDULER_SOURCE = Path(r"C:\Users\peets\NEC\slurm_scheduler")
DEFAULT_PARENT_ROOT = Path(
    r"C:\Users\peets\slurm_scheduler_runtime"
    r"\mft_current7_safe_bridge_20260721"
)
DEFAULT_PARENT_PLAN = (
    DEFAULT_PARENT_ROOT / "current7-091003f1dbe262ed9aca" / "offload_plan.json"
)
DEFAULT_PARENT_PUBLICATION = DEFAULT_PARENT_ROOT / "publication_receipt.json"


@dataclass(frozen=True)
class ParentIdentity:
    bundle_id: str
    contract_sha256: str
    manifest_sha256: str
    plan_sha256: str
    source_map_sha256: str
    publication_file_sha256: str
    remote_bundle: str


PRODUCTION_PARENT = ParentIdentity(
    bundle_id="current7-091003f1dbe262ed9aca",
    contract_sha256=(
        "091003f1dbe262ed9aca2b8b4bc1562444f08ca0a47a9d41ebaa889c27317183"
    ),
    manifest_sha256=(
        "10b99b6c70ccb6753bb899862c5db008be41f68272a6751a0d969bd9ea6f0c74"
    ),
    plan_sha256=("7e02e8ce6d722073be7637515181e513f2cbb0554d7f5061493dae23eb924884"),
    source_map_sha256=(
        "27a9ccce52b492162bc488a850b39638bb54ccb7ec2138335d3ea2c76ea70805"
    ),
    publication_file_sha256=(
        "31e22b26181cf3245fd25374a0297cded5725826e3c66c0d3fb09fbe5e001175"
    ),
    remote_bundle=(
        "/gpfs/tmp_cpu2/mft_tier1_current7_bundles/current7-091003f1dbe262ed9aca"
    ),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parse_json(value: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid remote JSON: {label}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"remote JSON object required: {label}")
    return parsed


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or str(path) != value
    ):
        raise RuntimeError(f"unsafe bundle-relative path: {value}")
    return str(path)


def _seal(value: Mapping[str, Any], field: str = "receipt_sha256") -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_sha256(result)
    return result


def _validate_seal(value: Mapping[str, Any], field: str = "receipt_sha256") -> None:
    unsigned = {key: item for key, item in value.items() if key != field}
    if value.get(field) != canonical_sha256(unsigned):
        raise RuntimeError("sealed delta receipt SHA mismatch")


def _record_matches(
    actual: Mapping[str, Any] | None, expected: Mapping[str, Any]
) -> bool:
    return actual is not None and (
        int(actual.get("size", -1)) == int(expected["size"])
        and actual.get("sha256") == expected["sha256"]
    )


def _identity_from_mapping(value: Mapping[str, Any]) -> ParentIdentity:
    try:
        return ParentIdentity(
            bundle_id=str(value["bundle_id"]),
            contract_sha256=str(value["contract_sha256"]),
            manifest_sha256=str(value["manifest_sha256"]),
            plan_sha256=str(value["plan_sha256"]),
            source_map_sha256=str(value["source_map_sha256"]),
            publication_file_sha256=str(value["publication_file_sha256"]),
            remote_bundle=str(value["remote_bundle"]),
        )
    except KeyError as exc:
        raise RuntimeError("delta parent identity is incomplete") from exc


def _authenticate_parent(
    plan_path: Path,
    publication_path: Path,
    identity: ParentIdentity,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], dict[str, Any]]:
    plan_path = plan_path.resolve(strict=True)
    publication_path = publication_path.resolve(strict=True)
    if sha256_file(plan_path) != identity.plan_sha256:
        raise RuntimeError("parent plan file SHA mismatch")
    plan, manifest, source_map = load_bundle_plan(plan_path)
    source_map_path = Path(str(plan["local_sources"])).resolve(strict=True)
    if sha256_file(source_map_path) != identity.source_map_sha256:
        raise RuntimeError("parent source-map file SHA mismatch")
    if sha256_file(publication_path) != identity.publication_file_sha256:
        raise RuntimeError("parent publication file SHA mismatch")
    expected = {
        "bundle_id": identity.bundle_id,
        "contract_sha256": identity.contract_sha256,
        "bundle_manifest_sha256": identity.manifest_sha256,
        "remote_bundle": identity.remote_bundle,
    }
    if any(plan.get(key) != item for key, item in expected.items()):
        raise RuntimeError("parent plan identity mismatch")
    publication = validate_publication_receipt(
        read_json(publication_path), plan=plan, manifest=manifest
    )
    return plan, manifest, source_map, publication


def _classify_files(
    parent_files: Mapping[str, Any], child_files: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inherited: dict[str, Any] = {}
    delta: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    for relative, expected in sorted(child_files.items()):
        relative = _safe_relative(str(relative))
        if parent_files.get(relative) == expected:
            inherited[relative] = copy.deepcopy(expected)
        else:
            delta[relative] = copy.deepcopy(expected)
    for relative, expected in sorted(parent_files.items()):
        if relative not in child_files:
            removed[_safe_relative(str(relative))] = copy.deepcopy(expected)
    return inherited, delta, removed


def _validate_delta_limits(
    delta: Mapping[str, Any],
    source_map: Mapping[str, Path],
    *,
    max_file_bytes: int,
    max_total_bytes: int,
) -> int:
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise RuntimeError("delta byte limits must be positive")
    total = 0
    for relative, expected in sorted(delta.items()):
        source = source_map.get(relative)
        size = int(expected.get("size", -1))
        if size < 0 or size > max_file_bytes:
            raise RuntimeError(f"delta file exceeds the small-file limit: {relative}")
        if (
            source is None
            or not source.is_file()
            or source.stat().st_size != size
            or sha256_file(source) != expected.get("sha256")
        ):
            raise RuntimeError(f"delta local source authentication failed: {relative}")
        total += size
    if total > max_total_bytes:
        raise RuntimeError("delta inventory exceeds the total upload limit")
    return total


def _validate_runtime_inheritance(
    parent_manifest: Mapping[str, Any], child_manifest: Mapping[str, Any]
) -> None:
    if child_manifest.get("runtime") != parent_manifest.get("runtime"):
        raise RuntimeError("final1000 delta must inherit the exact parent runtime")
    requirements = str(
        (parent_manifest.get("runtime") or {}).get("requirements_lock", "")
    )
    if not requirements or child_manifest.get("files", {}).get(
        requirements
    ) != parent_manifest.get("files", {}).get(requirements):
        raise RuntimeError("runtime requirements lock may not be a delta file")


def _validate_code_inventory(manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files") or {}
    code = manifest.get("code_inventory") or {}
    if (
        not isinstance(code, dict)
        or any(files.get(relative) != record for relative, record in code.items())
        or manifest.get("code_inventory_sha256") != canonical_sha256(code)
    ):
        raise RuntimeError("child code inventory is not sealed to child files")


def create_delta_plan(
    candidate_plan_path: Path,
    parent_plan_path: Path,
    parent_publication_path: Path,
    output_root: Path,
    *,
    identity: ParentIdentity = PRODUCTION_PARENT,
    max_delta_file_bytes: int = DEFAULT_MAX_DELTA_FILE_BYTES,
    max_delta_total_bytes: int = DEFAULT_MAX_DELTA_TOTAL_BYTES,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Seal a standard child plan with immutable-parent delta lineage."""

    parent_plan, parent_manifest, _, parent_publication = _authenticate_parent(
        parent_plan_path, parent_publication_path, identity
    )
    candidate_plan_path = candidate_plan_path.resolve(strict=True)
    candidate_plan, candidate_manifest, candidate_sources = load_bundle_plan(
        candidate_plan_path
    )
    if candidate_manifest.get("delta_publication") is not None:
        raise RuntimeError("candidate manifest is already a delta publication")
    if candidate_plan.get("remote_root") != parent_plan.get("remote_root"):
        raise RuntimeError("child and parent must share one remote GPFS root")
    _validate_runtime_inheritance(parent_manifest, candidate_manifest)
    _validate_code_inventory(candidate_manifest)
    inherited, delta, removed = _classify_files(
        parent_manifest["files"], candidate_manifest["files"]
    )
    if not delta:
        raise RuntimeError("candidate has no changed or new delta files")
    delta_bytes = _validate_delta_limits(
        delta,
        candidate_sources,
        max_file_bytes=max_delta_file_bytes,
        max_total_bytes=max_delta_total_bytes,
    )
    candidate_stable = {
        key: copy.deepcopy(item)
        for key, item in candidate_manifest.items()
        if key not in {"bundle_id", "contract_sha256"}
    }
    candidate_id = str(candidate_manifest["bundle_id"])
    if candidate_id in json.dumps(candidate_stable, ensure_ascii=False):
        raise RuntimeError("candidate embeds its transient bundle id in its contract")
    lineage = {
        "schema_version": DELTA_SCHEMA,
        "purpose": "final1000_small_delta_over_live_current7",
        "parent_identity": asdict(identity),
        "parent_ready_sha256": canonical_sha256(parent_publication["ready"]),
        "parent_file_inventory_sha256": canonical_sha256(parent_manifest["files"]),
        "inherited_file_inventory_sha256": canonical_sha256(inherited),
        "delta_file_inventory_sha256": canonical_sha256(delta),
        "removed_file_inventory_sha256": canonical_sha256(removed),
        "inherited_file_count": len(inherited),
        "delta_file_count": len(delta),
        "removed_file_count": len(removed),
        "inherited_byte_count": sum(int(row["size"]) for row in inherited.values()),
        "delta_byte_count": delta_bytes,
        "maximum_delta_file_bytes": int(max_delta_file_bytes),
        "maximum_delta_total_bytes": int(max_delta_total_bytes),
        "unchanged_clone_methods_allowed": ["reflink", "hardlink"],
        "regular_copy_fallback_allowed": False,
        "runtime_byte_identical_to_parent": True,
        "upload_only_changed_or_new_small_files": True,
        "verify_live_parent_before_first_write": True,
        "verify_every_child_file_before_ready": True,
        "write_ready_last": True,
        "atomic_same_filesystem_promotion": True,
    }
    stable = {**candidate_stable, "delta_publication": lineage}
    contract_sha = canonical_sha256(stable)
    bundle_id = f"current7-{contract_sha[:20]}"
    manifest = {**stable, "bundle_id": bundle_id, "contract_sha256": contract_sha}

    plan_dir = output_root.resolve() / bundle_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plan_dir / "bundle_manifest.json"
    source_map_path = plan_dir / "local_sources.json"
    plan_path = plan_dir / "offload_plan.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        source_map_path,
        {relative: str(path) for relative, path in sorted(candidate_sources.items())},
    )
    manifest_sha = sha256_file(manifest_path)
    remote_root = str(parent_plan["remote_root"]).rstrip("/")
    plan = copy.deepcopy(candidate_plan)
    plan.pop("bridge_clone", None)
    plan.update(
        {
            "created_at": _now(),
            "bundle_id": bundle_id,
            "contract_sha256": contract_sha,
            "bundle_manifest": str(manifest_path),
            "bundle_manifest_sha256": manifest_sha,
            "local_sources": str(source_map_path),
            "remote_root": remote_root,
            "remote_bundle": f"{remote_root}/{bundle_id}",
            "delta_publication": {
                "schema_version": DELTA_SCHEMA,
                "parent_plan": str(parent_plan_path.resolve(strict=True)),
                "parent_publication": str(parent_publication_path.resolve(strict=True)),
                "parent_identity": asdict(identity),
                "candidate_plan_sha256": sha256_file(candidate_plan_path),
                "candidate_source_map_sha256": sha256_file(
                    Path(str(candidate_plan["local_sources"]))
                ),
                "payload_upload_paths": sorted(delta),
                "scheduler_submission_performed": False,
            },
            "stage_performed": False,
            "submission_performed": False,
        }
    )
    atomic_json(plan_path, plan)
    receipt = _seal(
        {
            "schema_version": DELTA_PLAN_RECEIPT_SCHEMA,
            "bundle_id": bundle_id,
            "contract_sha256": contract_sha,
            "bundle_manifest_sha256": manifest_sha,
            "plan_sha256": sha256_file(plan_path),
            "source_map_sha256": sha256_file(source_map_path),
            "parent_identity": asdict(identity),
            "parent_ready_sha256": lineage["parent_ready_sha256"],
            "inherited_file_inventory_sha256": lineage[
                "inherited_file_inventory_sha256"
            ],
            "delta_file_inventory_sha256": lineage["delta_file_inventory_sha256"],
            "delta_file_count": len(delta),
            "delta_byte_count": delta_bytes,
            "remote_bundle": plan["remote_bundle"],
            "scheduler_submission_performed": False,
            "remote_write_performed": False,
        }
    )
    atomic_json(plan_dir / "delta_plan_receipt.json", receipt)
    return plan, manifest, receipt


def load_delta_plan(
    plan_path: Path,
    *,
    required_parent: ParentIdentity | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Authenticate the child plan, parent evidence, and every delta source."""

    plan_path = plan_path.resolve(strict=True)
    plan, manifest, source_map = load_bundle_plan(plan_path)
    lineage = manifest.get("delta_publication") or {}
    plan_lineage = plan.get("delta_publication") or {}
    if lineage.get("schema_version") != DELTA_SCHEMA:
        raise RuntimeError("child manifest has no final1000 delta contract")
    identity = _identity_from_mapping(lineage.get("parent_identity") or {})
    if required_parent is not None and identity != required_parent:
        raise RuntimeError("delta is not pinned to the required live parent")
    if plan_lineage.get("parent_identity") != asdict(identity):
        raise RuntimeError("child plan/manifest parent identity mismatch")
    parent_plan, parent_manifest, _, parent_publication = _authenticate_parent(
        Path(str(plan_lineage.get("parent_plan") or "")),
        Path(str(plan_lineage.get("parent_publication") or "")),
        identity,
    )
    if plan.get("remote_root") != parent_plan.get("remote_root"):
        raise RuntimeError("child escaped the parent remote GPFS root")
    _validate_runtime_inheritance(parent_manifest, manifest)
    _validate_code_inventory(manifest)
    inherited, delta, removed = _classify_files(
        parent_manifest["files"], manifest["files"]
    )
    delta_bytes = _validate_delta_limits(
        delta,
        source_map,
        max_file_bytes=int(lineage.get("maximum_delta_file_bytes", 0)),
        max_total_bytes=int(lineage.get("maximum_delta_total_bytes", 0)),
    )
    expected_lineage = {
        "parent_ready_sha256": canonical_sha256(parent_publication["ready"]),
        "parent_file_inventory_sha256": canonical_sha256(parent_manifest["files"]),
        "inherited_file_inventory_sha256": canonical_sha256(inherited),
        "delta_file_inventory_sha256": canonical_sha256(delta),
        "removed_file_inventory_sha256": canonical_sha256(removed),
        "inherited_file_count": len(inherited),
        "delta_file_count": len(delta),
        "removed_file_count": len(removed),
        "inherited_byte_count": sum(int(row["size"]) for row in inherited.values()),
        "delta_byte_count": delta_bytes,
    }
    if any(lineage.get(key) != item for key, item in expected_lineage.items()):
        raise RuntimeError("delta lineage inventory identity mismatch")
    required_flags = (
        "runtime_byte_identical_to_parent",
        "upload_only_changed_or_new_small_files",
        "verify_live_parent_before_first_write",
        "verify_every_child_file_before_ready",
        "write_ready_last",
        "atomic_same_filesystem_promotion",
    )
    if (
        any(lineage.get(flag) is not True for flag in required_flags)
        or lineage.get("regular_copy_fallback_allowed") is not False
        or lineage.get("unchanged_clone_methods_allowed") != ["reflink", "hardlink"]
        or plan_lineage.get("payload_upload_paths") != sorted(delta)
    ):
        raise RuntimeError("delta publication safety contract mismatch")
    receipt_path = plan_path.with_name("delta_plan_receipt.json")
    receipt = read_json(receipt_path)
    _validate_seal(receipt)
    expected_receipt = {
        "schema_version": DELTA_PLAN_RECEIPT_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "contract_sha256": plan["contract_sha256"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "plan_sha256": sha256_file(plan_path),
        "source_map_sha256": sha256_file(Path(plan["local_sources"])),
        "parent_identity": asdict(identity),
        "parent_ready_sha256": lineage["parent_ready_sha256"],
        "inherited_file_inventory_sha256": lineage["inherited_file_inventory_sha256"],
        "delta_file_inventory_sha256": lineage["delta_file_inventory_sha256"],
        "delta_file_count": len(delta),
        "delta_byte_count": delta_bytes,
        "remote_bundle": plan["remote_bundle"],
        "scheduler_submission_performed": False,
        "remote_write_performed": False,
    }
    if any(receipt.get(key) != item for key, item in expected_receipt.items()):
        raise RuntimeError("delta plan receipt identity mismatch")
    return (
        plan,
        manifest,
        source_map,
        parent_plan,
        parent_manifest,
        parent_publication,
        {"inherited": inherited, "delta": delta, "removed": removed},
    )


class DeltaTransport(Protocol):
    """Remote surface used by the restartable mixed clone/upload publisher."""

    write_count: int

    def exists(self, path: str) -> bool: ...

    def is_dir(self, path: str) -> bool: ...

    def read_bytes(self, path: str) -> bytes: ...

    def mkdir(self, path: str, mode: int = 0o755) -> None: ...

    def upload_file(self, local: Path, remote: str) -> None: ...

    def replace(self, source: str, destination: str) -> None: ...

    def write_control(self, path: str, value: bytes) -> None: ...

    def write_ready(self, path: str, value: bytes) -> None: ...

    def file_record(self, path: str) -> Mapping[str, Any] | None: ...

    def choose_clone_method(
        self, source: str, probe_destination: str, expected: Mapping[str, Any]
    ) -> str: ...

    def clone_file(self, source: str, destination: str, method: str) -> None: ...

    def clone_runtime_tree(
        self, source: str, destination: str, method: str
    ) -> None: ...

    def verify_runtime(
        self, root: str, expected_packages: Mapping[str, str]
    ) -> Mapping[str, str]: ...

    def seal_permissions(self, root: str) -> None: ...

    def verify_permissions(self, root: str, *, promoted: bool) -> bool: ...

    def promote_directory(self, incoming: str, destination: str) -> None: ...


def _verify_inventory(
    transport: DeltaTransport,
    root: str,
    files: Mapping[str, Any],
    label: str,
) -> None:
    for relative, expected in sorted(files.items()):
        relative = _safe_relative(str(relative))
        if not _record_matches(transport.file_record(f"{root}/{relative}"), expected):
            raise RuntimeError(f"{label} file authentication failed: {relative}")


def _verify_manifest(
    transport: DeltaTransport,
    root: str,
    manifest_bytes: bytes,
    manifest_sha256: str,
) -> None:
    path = f"{root}/bundle_manifest.json"
    if transport.read_bytes(path) != manifest_bytes or not _record_matches(
        transport.file_record(path),
        {"size": len(manifest_bytes), "sha256": manifest_sha256},
    ):
        raise RuntimeError("remote child manifest authentication failed")


def _delta_ready_matches(
    ready: Mapping[str, Any],
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    parent_plan: Mapping[str, Any],
    inventories: Mapping[str, Mapping[str, Any]],
) -> bool:
    lineage = manifest["delta_publication"]
    return (
        _ready_matches(ready, plan, manifest)
        and ready.get("delta_publication") is True
        and ready.get("parent_bundle_id") == parent_plan.get("bundle_id")
        and ready.get("parent_contract_sha256") == parent_plan.get("contract_sha256")
        and ready.get("parent_ready_sha256") == lineage["parent_ready_sha256"]
        and ready.get("inherited_file_count") == len(inventories["inherited"])
        and ready.get("delta_file_count") == len(inventories["delta"])
        and ready.get("delta_byte_count") == lineage["delta_byte_count"]
        and ready.get("clone_method") in {"reflink", "hardlink"}
        and ready.get("regular_copy_fallback_used") is False
        and ready.get("network_payload_bytes_uploaded") == lineage["delta_byte_count"]
    )


def _verify_complete_child(
    transport: DeltaTransport,
    root: str,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_bytes: bytes,
    *,
    promoted: bool,
) -> dict[str, str]:
    _verify_manifest(
        transport,
        root,
        manifest_bytes,
        str(plan["bundle_manifest_sha256"]),
    )
    _verify_inventory(transport, root, manifest["files"], "child")
    packages = dict(
        transport.verify_runtime(root, manifest["runtime"]["critical_packages"])
    )
    if packages != manifest["runtime"]["critical_packages"]:
        raise RuntimeError("child runtime package identity mismatch")
    if not transport.verify_permissions(root, promoted=promoted):
        raise RuntimeError("child read-only/runs permission seal mismatch")
    return packages


def _publication_receipt(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    ready: Mapping[str, Any] | None,
    incoming: str,
    apply: bool,
    already_ready: bool,
    clone_method: str | None,
    cloned_files: int,
    uploaded_files: int,
    uploaded_bytes: int,
    resumed_files: int,
) -> dict[str, Any]:
    lineage = manifest["delta_publication"]
    value = {
        "schema_version": PUBLICATION_RECEIPT_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "contract_sha256": manifest["contract_sha256"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "remote_bundle": plan["remote_bundle"],
        "incoming_bundle": incoming,
        "apply": bool(apply),
        "already_ready": bool(already_ready),
        "publication_complete": ready is not None,
        "uploaded_files": int(uploaded_files),
        "resumed_files": int(resumed_files),
        "remote_write_performed": bool(apply and not already_ready),
        "ready": dict(ready) if ready is not None else None,
        "ready_sha256": canonical_sha256(ready) if ready is not None else None,
        "delta_publication": True,
        "clone_method": clone_method,
        "server_side_cloned_files": int(cloned_files),
        "network_payload_bytes_uploaded": (
            int(lineage["delta_byte_count"]) if ready is not None else 0
        ),
        "network_payload_bytes_uploaded_this_invocation": int(uploaded_bytes),
        "parent_bundle_id": lineage["parent_identity"]["bundle_id"],
        "parent_contract_sha256": lineage["parent_identity"]["contract_sha256"],
    }
    return _seal(value)


def publish_delta(
    plan_path: Path,
    *,
    apply: bool = False,
    transport: DeltaTransport | None = None,
    required_parent: ParentIdentity | None = None,
) -> dict[str, Any]:
    """Publish a mixed server-side clone/small-upload child bundle."""

    (
        plan,
        manifest,
        source_map,
        parent_plan,
        parent_manifest,
        parent_publication,
        inventories,
    ) = load_delta_plan(plan_path, required_parent=required_parent)
    incoming = (
        f"{plan['remote_root']}/.incoming-{plan['bundle_id']}-"
        f"{plan['bundle_manifest_sha256'][:12]}"
    )
    if not apply:
        return _publication_receipt(
            plan=plan,
            manifest=manifest,
            ready=None,
            incoming=incoming,
            apply=False,
            already_ready=False,
            clone_method=None,
            cloned_files=0,
            uploaded_files=0,
            uploaded_bytes=0,
            resumed_files=0,
        )
    if transport is None:
        raise RuntimeError("--apply requires an explicit delta transport")

    parent_root = str(parent_plan["remote_bundle"])
    parent_ready_path = f"{parent_root}/READY.json"
    parent_manifest_path = f"{parent_root}/bundle_manifest.json"
    if not transport.exists(parent_ready_path):
        raise RuntimeError("authenticated parent READY is absent")
    live_parent_ready = _parse_json(
        transport.read_bytes(parent_ready_path), parent_ready_path
    )
    if live_parent_ready != parent_publication["ready"] or not _ready_matches(
        live_parent_ready, parent_plan, parent_manifest
    ):
        raise RuntimeError("live parent READY does not match sealed publication")
    parent_manifest_bytes = Path(parent_plan["bundle_manifest"]).read_bytes()
    if transport.read_bytes(
        parent_manifest_path
    ) != parent_manifest_bytes or not _record_matches(
        transport.file_record(parent_manifest_path),
        {
            "size": len(parent_manifest_bytes),
            "sha256": parent_plan["bundle_manifest_sha256"],
        },
    ):
        raise RuntimeError("live parent manifest bytes changed")
    _verify_inventory(transport, parent_root, parent_manifest["files"], "parent")
    parent_packages = dict(
        transport.verify_runtime(
            parent_root, parent_manifest["runtime"]["critical_packages"]
        )
    )
    if parent_packages != parent_manifest["runtime"]["critical_packages"]:
        raise RuntimeError("parent runtime package identity mismatch")

    final = str(plan["remote_bundle"])
    final_ready_path = f"{final}/READY.json"
    manifest_bytes = Path(plan["bundle_manifest"]).read_bytes()
    if transport.exists(final_ready_path):
        ready = _parse_json(transport.read_bytes(final_ready_path), final_ready_path)
        if not _delta_ready_matches(ready, plan, manifest, parent_plan, inventories):
            raise RuntimeError("content-addressed child has mismatched READY")
        _verify_complete_child(
            transport, final, plan, manifest, manifest_bytes, promoted=True
        )
        return _publication_receipt(
            plan=plan,
            manifest=manifest,
            ready=ready,
            incoming=incoming,
            apply=True,
            already_ready=True,
            clone_method=str(ready["clone_method"]),
            cloned_files=0,
            uploaded_files=0,
            uploaded_bytes=0,
            resumed_files=len(manifest["files"]),
        )
    if transport.exists(final):
        raise RuntimeError("content-addressed child exists without exact READY")

    if transport.exists(incoming):
        if not transport.is_dir(incoming):
            raise RuntimeError("delta incoming path is not a directory")
    else:
        transport.mkdir(incoming, 0o755)
    if not transport.exists(f"{incoming}/runs"):
        transport.mkdir(f"{incoming}/runs", 0o1777)
    journal_path = f"{incoming}/.delta-journal.json"
    if transport.exists(journal_path):
        journal = _parse_json(transport.read_bytes(journal_path), journal_path)
        _validate_seal(journal)
        expected_journal = {
            "schema_version": DELTA_JOURNAL_SCHEMA,
            "bundle_id": plan["bundle_id"],
            "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
            "parent_bundle_id": parent_plan["bundle_id"],
            "delta_file_inventory_sha256": manifest["delta_publication"][
                "delta_file_inventory_sha256"
            ],
        }
        if any(journal.get(key) != item for key, item in expected_journal.items()):
            raise RuntimeError("incoming delta journal identity mismatch")
        clone_method = str(journal.get("clone_method"))
        if clone_method not in {"reflink", "hardlink"}:
            raise RuntimeError("incoming delta journal clone method is invalid")
    else:
        probe_relative = str(parent_manifest["runtime"]["requirements_lock"])
        clone_method = transport.choose_clone_method(
            f"{parent_root}/{probe_relative}",
            f"{incoming}/.delta-clone-probe",
            parent_manifest["files"][probe_relative],
        )
        if clone_method not in {"reflink", "hardlink"}:
            raise RuntimeError("transport selected an unsafe clone method")
        journal = _seal(
            {
                "schema_version": DELTA_JOURNAL_SCHEMA,
                "bundle_id": plan["bundle_id"],
                "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
                "parent_bundle_id": parent_plan["bundle_id"],
                "parent_manifest_sha256": parent_plan["bundle_manifest_sha256"],
                "delta_file_inventory_sha256": manifest["delta_publication"][
                    "delta_file_inventory_sha256"
                ],
                "clone_method": clone_method,
                "regular_copy_fallback_allowed": False,
            }
        )
        transport.write_control(journal_path, _json_bytes(journal))

    incoming_ready_path = f"{incoming}/READY.json"
    if transport.exists(incoming_ready_path):
        incoming_ready = _parse_json(
            transport.read_bytes(incoming_ready_path), incoming_ready_path
        )
        if not _delta_ready_matches(
            incoming_ready, plan, manifest, parent_plan, inventories
        ):
            raise RuntimeError("sealed incoming child has mismatched READY")
        _verify_complete_child(
            transport,
            incoming,
            plan,
            manifest,
            manifest_bytes,
            promoted=False,
        )
        transport.promote_directory(incoming, final)
        live_ready = _parse_json(
            transport.read_bytes(final_ready_path), final_ready_path
        )
        if live_ready != incoming_ready:
            raise RuntimeError("recovered child READY changed during promotion")
        _verify_complete_child(
            transport, final, plan, manifest, manifest_bytes, promoted=True
        )
        return _publication_receipt(
            plan=plan,
            manifest=manifest,
            ready=live_ready,
            incoming=incoming,
            apply=True,
            already_ready=False,
            clone_method=clone_method,
            cloned_files=0,
            uploaded_files=0,
            uploaded_bytes=0,
            resumed_files=len(manifest["files"]),
        )

    cloned = 0
    uploaded = 0
    uploaded_bytes = 0
    resumed = 0
    for relative, expected in sorted(inventories["inherited"].items()):
        destination = f"{incoming}/{relative}"
        record = transport.file_record(destination)
        if _record_matches(record, expected):
            resumed += 1
            continue
        if record is not None:
            raise RuntimeError(f"incoming inherited file is mismatched: {relative}")
        parent_directory = destination.rsplit("/", 1)[0]
        if not transport.exists(parent_directory):
            transport.mkdir(parent_directory, 0o755)
        transport.clone_file(f"{parent_root}/{relative}", destination, clone_method)
        if not _record_matches(transport.file_record(destination), expected):
            raise RuntimeError(f"server-side clone authentication failed: {relative}")
        cloned += 1

    suffix = str(plan["bundle_manifest_sha256"])[:12]
    for relative, expected in sorted(inventories["delta"].items()):
        destination = f"{incoming}/{relative}"
        record = transport.file_record(destination)
        if _record_matches(record, expected):
            resumed += 1
            continue
        if record is not None:
            raise RuntimeError(f"incoming delta file is mismatched: {relative}")
        parent_directory = destination.rsplit("/", 1)[0]
        if not transport.exists(parent_directory):
            transport.mkdir(parent_directory, 0o755)
        part = f"{destination}.part.{suffix}"
        part_record = transport.file_record(part)
        if part_record is not None and not _record_matches(part_record, expected):
            raise RuntimeError(f"incoming delta part is mismatched: {relative}")
        if part_record is None:
            transport.upload_file(source_map[relative], part)
            uploaded_bytes += int(expected["size"])
        if not _record_matches(transport.file_record(part), expected):
            raise RuntimeError(f"uploaded delta authentication failed: {relative}")
        transport.replace(part, destination)
        if not _record_matches(transport.file_record(destination), expected):
            raise RuntimeError(f"uploaded delta changed during promotion: {relative}")
        uploaded += 1

    child_manifest_path = f"{incoming}/bundle_manifest.json"
    if transport.exists(child_manifest_path):
        if transport.read_bytes(child_manifest_path) != manifest_bytes:
            raise RuntimeError("incoming child manifest is mismatched")
    else:
        transport.write_control(child_manifest_path, manifest_bytes)
    _verify_manifest(
        transport,
        incoming,
        manifest_bytes,
        str(plan["bundle_manifest_sha256"]),
    )
    child_runtime = f"{incoming}/artifacts/python-site"
    if transport.exists(child_runtime):
        actual_packages = dict(
            transport.verify_runtime(incoming, manifest["runtime"]["critical_packages"])
        )
        if actual_packages != manifest["runtime"]["critical_packages"]:
            raise RuntimeError("existing incoming runtime is mismatched")
    else:
        transport.clone_runtime_tree(
            f"{parent_root}/artifacts/python-site",
            child_runtime,
            clone_method,
        )
        actual_packages = dict(
            transport.verify_runtime(incoming, manifest["runtime"]["critical_packages"])
        )
        if actual_packages != manifest["runtime"]["critical_packages"]:
            raise RuntimeError("cloned runtime package identity mismatch")

    _verify_inventory(transport, incoming, manifest["files"], "staged child")
    transport.seal_permissions(incoming)
    actual_packages = _verify_complete_child(
        transport, incoming, plan, manifest, manifest_bytes, promoted=False
    )
    lineage = manifest["delta_publication"]
    ready = {
        "schema_version": READY_SCHEMA,
        "bundle_id": plan["bundle_id"],
        "contract_sha256": manifest["contract_sha256"],
        "bundle_manifest_sha256": plan["bundle_manifest_sha256"],
        "file_count": len(manifest["files"]),
        "byte_count": sum(int(row["size"]) for row in manifest["files"].values()),
        "runtime_verified": True,
        "runtime_packages": actual_packages,
        "all_file_sha256_verified": True,
        "every_file_sha256_verified": True,
        "code_inventory_sha256": manifest["code_inventory_sha256"],
        "relocation_contract_sha256": manifest["relocation"]["contract_sha256"],
        "remote_git_checkout_performed": False,
        "artifacts_read_only": True,
        "runs_mode": "1777",
        "delta_publication": True,
        "parent_bundle_id": parent_plan["bundle_id"],
        "parent_contract_sha256": parent_plan["contract_sha256"],
        "parent_ready_sha256": lineage["parent_ready_sha256"],
        "inherited_file_count": len(inventories["inherited"]),
        "delta_file_count": len(inventories["delta"]),
        "delta_byte_count": lineage["delta_byte_count"],
        "clone_method": clone_method,
        "regular_copy_fallback_used": False,
        "network_payload_bytes_uploaded": lineage["delta_byte_count"],
        "published_at": _now(),
    }
    # Deliberately the final remote file write.  Only atomic promotion follows.
    transport.write_ready(incoming_ready_path, _json_bytes(ready))
    transport.promote_directory(incoming, final)
    live_ready = _parse_json(transport.read_bytes(final_ready_path), final_ready_path)
    if live_ready != ready or not _delta_ready_matches(
        live_ready, plan, manifest, parent_plan, inventories
    ):
        raise RuntimeError("child READY changed during atomic promotion")
    _verify_complete_child(
        transport, final, plan, manifest, manifest_bytes, promoted=True
    )
    return _publication_receipt(
        plan=plan,
        manifest=manifest,
        ready=ready,
        incoming=incoming,
        apply=True,
        already_ready=False,
        clone_method=clone_method,
        cloned_files=cloned,
        uploaded_files=uploaded,
        uploaded_bytes=uploaded_bytes,
        resumed_files=resumed,
    )


class SSHDeltaTransport(SSHPublicationTransport):
    """SSH transport with no regular-copy or runtime-install fallback."""

    _MAX_CONTROL_BYTES = 1_000_000

    def write_control(self, path: str, value: bytes) -> None:
        if len(value) > self._MAX_CONTROL_BYTES:
            raise RuntimeError("delta publisher refuses payload-sized control writes")
        basename = path.rsplit("/", 1)[-1]
        if basename not in {"bundle_manifest.json", ".delta-journal.json"}:
            raise RuntimeError("delta publisher refuses unknown control-file write")
        part = path + ".part"
        self.session.write_text_file(part, value.decode("utf-8"))
        self._run(
            f"chmod 0444 {shlex.quote(part)} && "
            f"mv -f -- {shlex.quote(part)} {shlex.quote(path)}",
            60,
        )
        self.write_count += 2

    def choose_clone_method(
        self, source: str, probe_destination: str, expected: Mapping[str, Any]
    ) -> str:
        for method, command in (
            (
                "reflink",
                "cp --reflink=always -- "
                f"{shlex.quote(source)} {shlex.quote(probe_destination)}",
            ),
            (
                "hardlink",
                f"ln -- {shlex.quote(source)} {shlex.quote(probe_destination)}",
            ),
        ):
            self.session.run(f"rm -f -- {shlex.quote(probe_destination)}", timeout=60)
            result = self.session.run(command, timeout=300)
            if result.exit_code == 0 and _record_matches(
                self.file_record(probe_destination), expected
            ):
                self._run(f"rm -f -- {shlex.quote(probe_destination)}", 60)
                self.write_count += 1
                return method
        self.session.run(f"rm -f -- {shlex.quote(probe_destination)}", timeout=60)
        raise RuntimeError("remote filesystem supports neither reflink nor hardlink")

    def clone_file(self, source: str, destination: str, method: str) -> None:
        part = destination + ".delta-clone-part"
        if method == "reflink":
            command = (
                f"cp --reflink=always -- {shlex.quote(source)} {shlex.quote(part)}"
            )
        elif method == "hardlink":
            command = f"ln -- {shlex.quote(source)} {shlex.quote(part)}"
        else:
            raise RuntimeError("unsafe delta clone method")
        self._run(
            f"rm -f -- {shlex.quote(part)}; {command}; "
            f"mv -f -- {shlex.quote(part)} {shlex.quote(destination)}",
            1800,
        )
        self.write_count += 1

    def clone_runtime_tree(self, source: str, destination: str, method: str) -> None:
        part = destination + ".delta-clone-part"
        if method == "reflink":
            clone = (
                f"cp -a --reflink=always -- {shlex.quote(source)} {shlex.quote(part)}"
            )
        elif method == "hardlink":
            clone = f"cp -al -- {shlex.quote(source)} {shlex.quote(part)}"
        else:
            raise RuntimeError("unsafe delta runtime clone method")
        self._run(
            f"rm -rf -- {shlex.quote(part)}; {clone}; "
            f"test ! -e {shlex.quote(destination)}; "
            f"mv -T -- {shlex.quote(part)} {shlex.quote(destination)}",
            1800,
        )
        self.write_count += 1

    def verify_runtime(
        self, root: str, expected_packages: Mapping[str, str]
    ) -> Mapping[str, str]:
        site = f"{root}/artifacts/python-site"
        script = (
            "import importlib.metadata,json\n"
            f"names={json.dumps(sorted(expected_packages))}\n"
            "print(json.dumps({n:importlib.metadata.version(n) for n in names},sort_keys=True))\n"
        )
        command = "\n".join(
            [
                "set -euo pipefail",
                self.env_setup,
                f"test -d {shlex.quote(site)}",
                f"export PYTHONPATH={shlex.quote(site)}",
                "python - <<'PY'",
                script.rstrip(),
                "PY",
            ]
        )
        output = self._run("bash -lc " + shlex.quote(command), 600)
        return json.loads(output.strip().splitlines()[-1])

    def seal_permissions(self, root: str) -> None:
        command = "\n".join(
            [
                "set -euo pipefail",
                f"chmod -R a+rX,a-w {shlex.quote(root + '/artifacts')}",
                f"chmod a+r,a-w {shlex.quote(root + '/bundle_manifest.json')}",
                f"chmod a+r,a-w {shlex.quote(root + '/.delta-journal.json')}",
                f"chmod 1777 {shlex.quote(root + '/runs')}",
            ]
        )
        self._run("bash -lc " + shlex.quote(command), 600)
        self.write_count += 1

    def verify_permissions(self, root: str, *, promoted: bool) -> bool:
        root_check = f"test ! -w {shlex.quote(root)} && " if promoted else ""
        command = (
            f"{root_check}test \"$(stat -c '%a' {shlex.quote(root + '/runs')})\" = 1777 && "
            f'test -z "$(find {shlex.quote(root + "/artifacts")} -perm /222 -print -quit)" && '
            f"test ! -w {shlex.quote(root + '/bundle_manifest.json')} && "
            f"test ! -w {shlex.quote(root + '/.delta-journal.json')}"
        )
        return self.session.run(command, timeout=300).exit_code == 0

    def promote_directory(self, incoming: str, destination: str) -> None:
        self._run(
            f"test ! -e {shlex.quote(destination)} && "
            f"mv -T -- {shlex.quote(incoming)} {shlex.quote(destination)}",
            60,
        )
        self.write_count += 1


@contextmanager
def scheduler_delta_transport(
    *, accounts_path: Path, scheduler_source: Path, account_name: str
) -> Iterator[SSHDeltaTransport]:
    source = scheduler_source.resolve(strict=True)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from slurm_scheduler.config import load_accounts
    from slurm_scheduler.slurm import SSHSession

    accounts = {account.name: account for account in load_accounts(accounts_path)}
    if account_name not in accounts:
        raise RuntimeError(f"unknown scheduler account: {account_name}")
    account = accounts[account_name]
    env_setup = (account.env_profiles or {}).get("pyaedt2026v1", "")
    with SSHSession(account, default_timeout=300) as session:
        yield SSHDeltaTransport(session, env_setup)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--candidate-plan", type=Path, required=True)
    plan.add_argument("--parent-plan", type=Path, default=DEFAULT_PARENT_PLAN)
    plan.add_argument(
        "--parent-publication", type=Path, default=DEFAULT_PARENT_PUBLICATION
    )
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument(
        "--max-delta-file-bytes", type=int, default=DEFAULT_MAX_DELTA_FILE_BYTES
    )
    plan.add_argument(
        "--max-delta-total-bytes", type=int, default=DEFAULT_MAX_DELTA_TOTAL_BYTES
    )
    publish = commands.add_parser("publish")
    publish.add_argument("--plan", type=Path, required=True)
    publish.add_argument("--apply", action="store_true")
    publish.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    publish.add_argument(
        "--scheduler-source", type=Path, default=DEFAULT_SCHEDULER_SOURCE
    )
    publish.add_argument("--account", default="harry261")
    publish.add_argument("--receipt-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        plan, manifest, receipt = create_delta_plan(
            args.candidate_plan,
            args.parent_plan,
            args.parent_publication,
            args.output_root,
            identity=PRODUCTION_PARENT,
            max_delta_file_bytes=args.max_delta_file_bytes,
            max_delta_total_bytes=args.max_delta_total_bytes,
        )
        output = {
            "plan": plan,
            "manifest": {
                "bundle_id": manifest["bundle_id"],
                "contract_sha256": manifest["contract_sha256"],
                "file_count": len(manifest["files"]),
                "delta_file_count": manifest["delta_publication"]["delta_file_count"],
                "delta_byte_count": manifest["delta_publication"]["delta_byte_count"],
            },
            "receipt": receipt,
        }
    else:
        if args.apply:
            with scheduler_delta_transport(
                accounts_path=args.accounts,
                scheduler_source=args.scheduler_source,
                account_name=args.account,
            ) as transport:
                output = publish_delta(
                    args.plan,
                    apply=True,
                    transport=transport,
                    required_parent=PRODUCTION_PARENT,
                )
        else:
            output = publish_delta(
                args.plan,
                apply=False,
                required_parent=PRODUCTION_PARENT,
            )
        if args.receipt_out is not None:
            if not args.apply:
                raise RuntimeError("--receipt-out requires --apply")
            atomic_json(args.receipt_out, output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
