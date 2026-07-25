"""Durable, fail-closed campaign claims for exactly-once MFT submissions.

The Scheduler API intentionally remains a separate project and is not treated
as an atomic uniqueness authority here.  This module gives one frozen MFT
campaign a shared filesystem claim namespace.  A claim directory is acquired
with one atomic ``mkdir`` before any Scheduler mutation, and is never released.

The module has no Scheduler or SQLite dependency.  Callers remain responsible
for validating Scheduler-specific task evidence before finalization.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Any
import uuid

from module.mft_goal_20260726_contract import canonical_sha256


ROOT_AUTHORITY_SCHEMA = "mft-campaign-atomic-claim-root-v1"
CLAIM_REFERENCE_SCHEMA = "mft-campaign-atomic-claim-reference-v1"
PENDING_CLAIM_SCHEMA = "mft-campaign-atomic-pending-claim-v1"
FINALIZED_CLAIM_SCHEMA = "mft-campaign-atomic-finalized-claim-v1"

ROOT_MARKER_NAME = ".mft-campaign-claim-root.json"
CLAIMS_DIRECTORY_NAME = "claims"
PENDING_CLAIM_NAME = "pending.json"
FINALIZED_CLAIM_NAME = "finalized.json"
MAX_JSON_BYTES = 1024 * 1024

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_GENERATION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")

__all__ = [
    "CLAIM_REFERENCE_SCHEMA",
    "FINALIZED_CLAIM_SCHEMA",
    "PENDING_CLAIM_SCHEMA",
    "ROOT_AUTHORITY_SCHEMA",
    "ClaimContractError",
    "acquire_claim",
    "build_claim_reference",
    "finalize_claim",
    "initialize_claim_root",
    "load_claim_root",
    "recover_pending_claim",
    "validate_claim_reference",
    "validate_finalized_claim",
    "validate_pending_claim",
]


class ClaimContractError(RuntimeError):
    """Raised whenever claim authority or recovery cannot be proven exactly."""


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ClaimContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClaimContractError(f"{label} must be a positive integer")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ClaimContractError(f"{label} is unsafe or empty")
    return value


def _require_generation(value: Any) -> str:
    if not isinstance(value, str) or _GENERATION_RE.fullmatch(value) is None:
        raise ClaimContractError("claim retry generation is unsafe or empty")
    return value


def _json_clone(value: Any, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        cloned = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ClaimContractError(f"{label} is not canonical JSON") from exc
    if cloned != value:
        raise ClaimContractError(f"{label} changes during JSON normalization")
    return cloned


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = _json_clone(dict(value), "claim payload")
    if "payload_sha256" in sealed:
        raise ClaimContractError("claim payload already contains a seal")
    sealed["payload_sha256"] = canonical_sha256(sealed)
    return sealed


def _validate_seal(
    value: Any,
    *,
    schema: str,
    expected_fields: set[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        *expected_fields,
        "payload_sha256",
    }:
        raise ClaimContractError(f"{label} is malformed")
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema_version") != schema or observed != canonical_sha256(unsigned):
        raise ClaimContractError(f"{label} seal mismatch")
    return _json_clone(value, label)


def _utc_timestamp(value: str | None = None) -> str:
    if value is None:
        value = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ClaimContractError("claim timestamp must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ClaimContractError("claim timestamp is invalid") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != timezone.utc.utcoffset(parsed)
        or parsed.microsecond != 0
        or parsed.isoformat().replace("+00:00", "Z") != value
    ):
        raise ClaimContractError("claim timestamp must be canonical UTC")
    return value


def _is_reparse(path: Path) -> bool:
    try:
        status = os.lstat(path)
    except OSError as exc:
        raise ClaimContractError(
            f"claim authority path is unavailable: {path}"
        ) from exc
    if stat.S_ISLNK(status.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(status, "st_file_attributes", 0)
    return bool(attributes & reparse_flag)


def _reject_reparse_chain(path: Path) -> None:
    chain: list[Path] = []
    current = path
    while True:
        chain.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    for component in reversed(chain):
        if _is_reparse(component):
            raise ClaimContractError(
                f"claim authority path uses a reparse point: {component}"
            )


def _normalized_path_text(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _canonical_existing_directory(path: Path, label: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ClaimContractError(f"{label} must be an absolute canonical path")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ClaimContractError(f"{label} is unavailable") from exc
    if _normalized_path_text(raw) != _normalized_path_text(resolved):
        raise ClaimContractError(
            f"{label} must not use aliases, symlinks, or reparse points"
        )
    _reject_reparse_chain(resolved)
    if not resolved.is_dir():
        raise ClaimContractError(f"{label} is not a directory")
    return resolved


def _canonical_new_child(path: Path, label: str) -> tuple[Path, Path]:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts or raw.name in {"", ".", ".."}:
        raise ClaimContractError(f"{label} must be an absolute canonical path")
    parent = _canonical_existing_directory(raw.parent, f"{label} parent")
    expected = parent / raw.name
    if _normalized_path_text(raw) != _normalized_path_text(expected):
        raise ClaimContractError(f"{label} must be an absolute canonical path")
    return expected, parent


def _regular_file(path: Path, label: str) -> Path:
    try:
        status = os.lstat(path)
    except OSError as exc:
        raise ClaimContractError(f"{label} is unavailable") from exc
    if _is_reparse(path) or not stat.S_ISREG(status.st_mode):
        raise ClaimContractError(f"{label} is not a regular file")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
    except OSError as exc:
        raise ClaimContractError(f"{label} is unavailable") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise ClaimContractError(f"{label} exceeds the byte bound")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ClaimContractError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ClaimContractError(f"{label} must be a JSON object")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _fsync_directory(path: Path) -> None:
    # Windows CreateDirectory/CreateFile CREATE_NEW provide the exclusivity
    # primitive used here.  Python cannot portably open a Windows directory for
    # FlushFileBuffers, while POSIX can and should persist the directory entry.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    label: str,
) -> None:
    payload = _json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ClaimContractError(f"{label} cannot be created") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Never unlink a partial authority/claim file.  Its presence forces a
        # fail-closed manual audit instead of permitting a second submission.
        raise
    _fsync_directory(path.parent)


def _storage_identity(path: Path) -> dict[str, int]:
    status = path.stat()
    return {
        "device": int(status.st_dev),
        "inode": int(status.st_ino),
    }


def _validate_root_authority_payload(value: Any) -> dict[str, Any]:
    authority = _validate_seal(
        value,
        schema=ROOT_AUTHORITY_SCHEMA,
        expected_fields={
            "schema_version",
            "root_id",
            "campaign_id",
            "campaign_authority_sha256",
            "resolved_root",
            "storage_identity",
            "claims_directory",
            "created_at_utc",
        },
        label="campaign claim-root authority",
    )
    if (
        not isinstance(authority.get("root_id"), str)
        or _ROOT_ID_RE.fullmatch(authority["root_id"]) is None
    ):
        raise ClaimContractError("campaign claim-root ID is invalid")
    _require_identifier(authority.get("campaign_id"), "campaign ID")
    _require_sha(
        authority.get("campaign_authority_sha256"),
        "campaign authority SHA",
    )
    resolved_root_value = authority.get("resolved_root")
    resolved_root_path = (
        Path(resolved_root_value) if isinstance(resolved_root_value, str) else Path()
    )
    if (
        not isinstance(authority.get("resolved_root"), str)
        or not resolved_root_path.is_absolute()
        or ".." in resolved_root_path.parts
        or _normalized_path_text(resolved_root_path)
        != _normalized_path_text(Path(os.path.abspath(str(resolved_root_path))))
        or authority.get("claims_directory") != CLAIMS_DIRECTORY_NAME
        or not isinstance(authority.get("storage_identity"), dict)
        or set(authority["storage_identity"]) != {"device", "inode"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in authority["storage_identity"].values()
        )
    ):
        raise ClaimContractError("campaign claim-root identity is malformed")
    _utc_timestamp(authority.get("created_at_utc"))
    return authority


def initialize_claim_root(
    root: Path,
    *,
    campaign_id: str,
    campaign_authority_sha256: str,
    root_id: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Create or reauthenticate one immutable campaign claim-root authority.

    A pre-existing valid root is returned idempotently.  A pre-existing empty,
    partial, copied, or differently bound root fails closed.
    """

    campaign_id = _require_identifier(campaign_id, "campaign ID")
    campaign_authority_sha256 = _require_sha(
        campaign_authority_sha256, "campaign authority SHA"
    )
    requested_root_id = root_id
    if root_id is not None and _ROOT_ID_RE.fullmatch(str(root_id)) is None:
        raise ClaimContractError("campaign claim-root ID is invalid")
    target, parent = _canonical_new_child(Path(root), "campaign claim root")
    if target.exists():
        authority = load_claim_root(target)
        if (
            authority["campaign_id"] != campaign_id
            or authority["campaign_authority_sha256"] != campaign_authority_sha256
            or (
                requested_root_id is not None
                and authority["root_id"] != requested_root_id
            )
        ):
            raise ClaimContractError(
                "existing campaign claim root has different authority"
            )
        return authority
    root_id = requested_root_id or uuid.uuid4().hex
    try:
        os.mkdir(target, 0o700)
    except FileExistsError:
        # Another initializer may be between mkdir and marker fsync.  Do not
        # guess or repair that partial state.
        raise ClaimContractError(
            "campaign claim root appeared during initialization"
        ) from None
    except OSError as exc:
        raise ClaimContractError("campaign claim root cannot be created") from exc
    _fsync_directory(parent)
    resolved = _canonical_existing_directory(target, "campaign claim root")
    claims_directory = resolved / CLAIMS_DIRECTORY_NAME
    try:
        os.mkdir(claims_directory, 0o700)
    except OSError as exc:
        raise ClaimContractError("campaign claims directory cannot be created") from exc
    _fsync_directory(resolved)
    authority = _seal(
        {
            "schema_version": ROOT_AUTHORITY_SCHEMA,
            "root_id": root_id,
            "campaign_id": campaign_id,
            "campaign_authority_sha256": campaign_authority_sha256,
            "resolved_root": str(resolved),
            "storage_identity": _storage_identity(resolved),
            "claims_directory": CLAIMS_DIRECTORY_NAME,
            "created_at_utc": _utc_timestamp(now),
        }
    )
    _write_exclusive_json(
        resolved / ROOT_MARKER_NAME,
        authority,
        label="campaign claim-root marker",
    )
    return load_claim_root(resolved, expected_authority=authority)


def load_claim_root(
    root: Path,
    *,
    expected_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the live root marker and its physical path identity."""

    resolved = _canonical_existing_directory(Path(root), "campaign claim root")
    authority = _validate_root_authority_payload(
        _read_json(
            resolved / ROOT_MARKER_NAME,
            "campaign claim-root marker",
        )
    )
    claims_directory = _canonical_existing_directory(
        resolved / CLAIMS_DIRECTORY_NAME,
        "campaign claims directory",
    )
    if (
        _normalized_path_text(Path(authority["resolved_root"]))
        != _normalized_path_text(resolved)
        or authority["storage_identity"] != _storage_identity(resolved)
        or claims_directory.parent != resolved
    ):
        raise ClaimContractError("campaign claim-root physical identity drifted")
    if expected_authority is not None:
        expected = _validate_root_authority_payload(
            _json_clone(
                dict(expected_authority),
                "expected campaign claim-root authority",
            )
        )
        if authority != expected:
            raise ClaimContractError(
                "campaign claim-root authority differs from the frozen "
                "campaign contract"
            )
    return authority


def _claim_context(
    authority: Mapping[str, Any],
    *,
    candidate_physics_sha256: str,
    logical_authority_task_id: int,
    retry_generation: str,
) -> dict[str, Any]:
    validated_authority = _validate_root_authority_payload(
        _json_clone(dict(authority), "campaign claim-root authority")
    )
    return {
        "root_id": validated_authority["root_id"],
        "campaign_id": validated_authority["campaign_id"],
        "campaign_authority_sha256": validated_authority["campaign_authority_sha256"],
        "candidate_physics_sha256": _require_sha(
            candidate_physics_sha256, "candidate physics SHA"
        ),
        "logical_authority_task_id": _require_positive_int(
            logical_authority_task_id, "logical authority task ID"
        ),
        "retry_generation": _require_generation(retry_generation),
    }


def _claim_key(context: Mapping[str, Any]) -> str:
    digest = canonical_sha256(context)
    return (
        f"{context['retry_generation']}-"
        f"t{context['logical_authority_task_id']}-"
        f"{context['candidate_physics_sha256'][:12]}-{digest[:16]}"
    )


def build_claim_reference(
    authority: Mapping[str, Any],
    *,
    candidate_physics_sha256: str,
    logical_authority_task_id: int,
    retry_generation: str,
) -> dict[str, Any]:
    """Build the portable plan reference to one logical retry claim slot."""

    validated_authority = _validate_root_authority_payload(
        _json_clone(dict(authority), "campaign claim-root authority")
    )
    context = _claim_context(
        validated_authority,
        candidate_physics_sha256=candidate_physics_sha256,
        logical_authority_task_id=logical_authority_task_id,
        retry_generation=retry_generation,
    )
    key = _claim_key(context)
    return _seal(
        {
            "schema_version": CLAIM_REFERENCE_SCHEMA,
            "claim_root_authority": validated_authority,
            "candidate_physics_sha256": context["candidate_physics_sha256"],
            "logical_authority_task_id": context["logical_authority_task_id"],
            "retry_generation": context["retry_generation"],
            "claim_key": key,
            "relative_claim_directory": (f"{CLAIMS_DIRECTORY_NAME}/{key}"),
        }
    )


def validate_claim_reference(
    reference: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a plan reference against the frozen live-root authority."""

    validated = _validate_seal(
        _json_clone(dict(reference), "campaign claim reference"),
        schema=CLAIM_REFERENCE_SCHEMA,
        expected_fields={
            "schema_version",
            "claim_root_authority",
            "candidate_physics_sha256",
            "logical_authority_task_id",
            "retry_generation",
            "claim_key",
            "relative_claim_directory",
        },
        label="campaign claim reference",
    )
    expected = build_claim_reference(
        authority,
        candidate_physics_sha256=_require_sha(
            validated.get("candidate_physics_sha256"),
            "candidate physics SHA",
        ),
        logical_authority_task_id=_require_positive_int(
            validated.get("logical_authority_task_id"),
            "logical authority task ID",
        ),
        retry_generation=_require_generation(validated.get("retry_generation")),
    )
    if validated != expected:
        raise ClaimContractError("campaign claim reference differs from root authority")
    return validated


def _claim_paths(
    root: Path,
    reference: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    authority = load_claim_root(
        root,
        expected_authority=(
            reference.get("claim_root_authority")
            if isinstance(reference, Mapping)
            else None
        ),
    )
    validated = validate_claim_reference(reference, authority)
    root_path = Path(authority["resolved_root"])
    claims_directory = _canonical_existing_directory(
        root_path / CLAIMS_DIRECTORY_NAME,
        "campaign claims directory",
    )
    claim_directory = claims_directory / validated["claim_key"]
    if claim_directory.parent != claims_directory:
        raise ClaimContractError("campaign claim path escaped its root")
    return (
        claim_directory,
        claim_directory / PENDING_CLAIM_NAME,
        claim_directory / FINALIZED_CLAIM_NAME,
    )


def _validate_winner(
    winner: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(winner, Mapping):
        raise ClaimContractError("campaign claim winner is absent")
    expected_fields = {
        "immediate_task_id",
        "immediate_retry_kind",
        "plan_payload_sha256",
        "plan_file_sha256",
        "profile_sha256",
        "resources",
        "task_name",
        "dedupe_key",
    }
    if set(winner) != expected_fields:
        raise ClaimContractError("campaign claim winner is malformed")
    normalized = _json_clone(dict(winner), "campaign claim winner")
    immediate_task_id = _require_positive_int(
        normalized.get("immediate_task_id"), "immediate task ID"
    )
    immediate_kind = normalized.get("immediate_retry_kind")
    if immediate_kind not in {"none", "timeout"}:
        raise ClaimContractError("immediate retry kind is unsupported")
    logical_task_id = _require_positive_int(
        reference.get("logical_authority_task_id"),
        "logical authority task ID",
    )
    if (immediate_kind == "none" and immediate_task_id != logical_task_id) or (
        immediate_kind == "timeout" and immediate_task_id == logical_task_id
    ):
        raise ClaimContractError("campaign claim ancestry is inconsistent")
    for field, label in (
        ("plan_payload_sha256", "plan payload SHA"),
        ("plan_file_sha256", "plan file SHA"),
        ("profile_sha256", "profile SHA"),
    ):
        _require_sha(normalized.get(field), label)
    resources = normalized.get("resources")
    if not isinstance(resources, dict) or not {"cpus", "timeout_seconds"}.issubset(
        resources
    ):
        raise ClaimContractError("campaign claim resources are malformed")
    for field in ("cpus", "timeout_seconds"):
        _require_positive_int(resources.get(field), f"resource {field}")
    for field in ("memory_mb",):
        if field in resources:
            _require_positive_int(resources.get(field), f"resource {field}")
    for field, label, maximum in (
        ("task_name", "task name", 256),
        ("dedupe_key", "dedupe key", 2048),
    ):
        value = normalized.get(field)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > maximum
            or any(ord(char) < 32 for char in value)
            or "/" in value
            or "\\" in value
        ):
            raise ClaimContractError(f"campaign claim {label} is unsafe")
    return normalized


def _pending_claim(
    reference: Mapping[str, Any],
    winner: Mapping[str, Any],
    *,
    nonce: str,
    now: str | None,
) -> dict[str, Any]:
    validated_reference = _json_clone(dict(reference), "campaign claim reference")
    validated_winner = _validate_winner(winner, reference=validated_reference)
    if _NONCE_RE.fullmatch(nonce) is None:
        raise ClaimContractError("campaign claim nonce is invalid")
    return _seal(
        {
            "schema_version": PENDING_CLAIM_SCHEMA,
            "state": "pending",
            "claim_reference": validated_reference,
            "winner": validated_winner,
            "winner_sha256": canonical_sha256(validated_winner),
            "nonce": nonce,
            "created_at_utc": _utc_timestamp(now),
        }
    )


def _validate_pending_payload(
    value: Any,
    *,
    reference: Mapping[str, Any],
    expected_winner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pending = _validate_seal(
        value,
        schema=PENDING_CLAIM_SCHEMA,
        expected_fields={
            "schema_version",
            "state",
            "claim_reference",
            "winner",
            "winner_sha256",
            "nonce",
            "created_at_utc",
        },
        label="pending campaign claim",
    )
    if pending.get("state") != "pending":
        raise ClaimContractError("pending campaign claim state drifted")
    if pending.get("claim_reference") != reference:
        raise ClaimContractError("pending campaign claim reference drifted")
    winner = _validate_winner(pending.get("winner") or {}, reference=reference)
    if pending.get("winner_sha256") != canonical_sha256(winner):
        raise ClaimContractError("pending campaign claim winner drifted")
    if (
        not isinstance(pending.get("nonce"), str)
        or _NONCE_RE.fullmatch(pending["nonce"]) is None
    ):
        raise ClaimContractError("pending campaign claim nonce drifted")
    _utc_timestamp(pending.get("created_at_utc"))
    if expected_winner is not None:
        normalized_expected = _validate_winner(expected_winner, reference=reference)
        if winner != normalized_expected:
            raise ClaimContractError(
                "campaign claim already belongs to different ancestry"
            )
    return pending


def _wait_for_claim_file(
    path: Path,
    *,
    wait_seconds: float,
) -> None:
    if (
        isinstance(wait_seconds, bool)
        or not isinstance(wait_seconds, (int, float))
        or wait_seconds < 0
        or wait_seconds > 5
    ):
        raise ClaimContractError("claim-file wait bound is invalid")
    deadline = time.monotonic() + float(wait_seconds)
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)


def acquire_claim(
    root: Path,
    reference: Mapping[str, Any],
    winner: Mapping[str, Any],
    *,
    nonce: str | None = None,
    now: str | None = None,
    wait_seconds: float = 1.0,
) -> dict[str, Any]:
    """Atomically acquire one logical retry slot before Scheduler mutation.

    The returned status is ``fresh_pending``, ``existing_pending``, or
    ``existing_finalized``.  An existing claim with different immediate
    ancestry or plan identity is always rejected.  Only ``fresh_pending``
    authorizes the caller's first Scheduler POST.  ``existing_pending`` must
    use :func:`recover_pending_claim` and must never POST or re-POST.
    """

    claim_directory, pending_path, finalized_path = _claim_paths(Path(root), reference)
    normalized_winner = _validate_winner(winner, reference=reference)
    fresh = False
    try:
        os.mkdir(claim_directory, 0o700)
        fresh = True
        _fsync_directory(claim_directory.parent)
    except FileExistsError:
        _canonical_existing_directory(claim_directory, "campaign claim directory")
    except OSError as exc:
        raise ClaimContractError("campaign claim directory cannot be acquired") from exc

    if fresh:
        pending = _pending_claim(
            reference,
            normalized_winner,
            nonce=nonce or secrets.token_hex(16),
            now=now,
        )
        _write_exclusive_json(
            pending_path,
            pending,
            label="pending campaign claim",
        )
        durable = validate_pending_claim(
            root,
            reference,
            claim=pending,
            expected_winner=normalized_winner,
        )
        return {"status": "fresh_pending", "claim": durable}

    if finalized_path.exists():
        finalized = validate_finalized_claim(
            root,
            reference,
            expected_winner=normalized_winner,
        )
        return {"status": "existing_finalized", "claim": finalized}

    _wait_for_claim_file(pending_path, wait_seconds=wait_seconds)
    if not pending_path.exists():
        raise ClaimContractError(
            "campaign claim directory has no durable pending claim; "
            "manual recovery is required"
        )
    pending = validate_pending_claim(
        root,
        reference,
        expected_winner=normalized_winner,
    )
    return {"status": "existing_pending", "claim": pending}


def validate_pending_claim(
    root: Path,
    reference: Mapping[str, Any],
    *,
    claim: Mapping[str, Any] | None = None,
    expected_winner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the durable pending record and an optional embedded copy."""

    claim_directory, pending_path, _finalized_path = _claim_paths(Path(root), reference)
    _canonical_existing_directory(claim_directory, "campaign claim directory")
    durable = _validate_pending_payload(
        _read_json(pending_path, "pending campaign claim"),
        reference=reference,
        expected_winner=expected_winner,
    )
    if claim is not None:
        embedded = _validate_pending_payload(
            _json_clone(dict(claim), "embedded pending campaign claim"),
            reference=reference,
            expected_winner=expected_winner,
        )
        if durable != embedded:
            raise ClaimContractError(
                "embedded pending claim differs from durable authority"
            )
    return durable


def _task_id_from_readback(value: Mapping[str, Any]) -> int:
    observed = [
        value[field]
        for field in ("task_id", "id")
        if field in value and value[field] is not None
    ]
    if not observed:
        raise ClaimContractError("claim task readback has no task ID")
    task_id = _require_positive_int(observed[0], "claim task ID")
    if any(item != task_id for item in observed):
        raise ClaimContractError("claim task readback IDs disagree")
    return task_id


def _validated_task_readback(
    task_readback: Mapping[str, Any],
    *,
    task_id: int,
    pending: Mapping[str, Any],
    evidence_validator: (
        Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None
    ),
) -> dict[str, Any]:
    if not isinstance(task_readback, Mapping):
        raise ClaimContractError("claim task readback is absent")
    raw = _json_clone(dict(task_readback), "claim task readback")
    winner = pending["winner"]
    if (
        _task_id_from_readback(raw) != task_id
        or raw.get("name") != winner["task_name"]
        or raw.get("dedupe_key") != winner["dedupe_key"]
        or any(
            raw.get(field) != expected
            for field, expected in winner["resources"].items()
        )
    ):
        raise ClaimContractError(
            "claim task readback differs from canonical submission identity"
        )
    if evidence_validator is None:
        return raw
    try:
        normalized = evidence_validator(copy.deepcopy(raw), pending)
    except ClaimContractError:
        raise
    except Exception as exc:
        raise ClaimContractError(
            "claim task evidence validator rejected readback"
        ) from exc
    if not isinstance(normalized, Mapping):
        raise ClaimContractError(
            "claim task evidence validator returned no normalized evidence"
        )
    normalized_copy = _json_clone(dict(normalized), "normalized claim task readback")
    if (
        _task_id_from_readback(normalized_copy) != task_id
        or normalized_copy.get("name") != winner["task_name"]
        or normalized_copy.get("dedupe_key") != winner["dedupe_key"]
        or any(
            normalized_copy.get(field) != expected
            for field, expected in winner["resources"].items()
        )
    ):
        raise ClaimContractError(
            "normalized claim task readback changed canonical identity"
        )
    return normalized_copy


def _validated_sibling_snapshot(
    value: Mapping[str, Any],
    *,
    task_id: int,
    winner: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ClaimContractError("claim sibling snapshot is absent")
    snapshot = _json_clone(dict(value), "claim sibling snapshot")
    rows = snapshot.get("matching_tasks")
    if (
        snapshot.get("matching_task_count") != 1
        or not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], dict)
        or _task_id_from_readback(rows[0]) != task_id
        or rows[0].get("name") != winner["task_name"]
        or rows[0].get("dedupe_key") != winner["dedupe_key"]
    ):
        raise ClaimContractError(
            "claim finalization requires exactly one matching sibling"
        )
    return snapshot


def _validate_finalized_payload(
    value: Any,
    *,
    reference: Mapping[str, Any],
    pending: Mapping[str, Any],
    expected_winner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    finalized = _validate_seal(
        value,
        schema=FINALIZED_CLAIM_SCHEMA,
        expected_fields={
            "schema_version",
            "state",
            "claim_reference",
            "pending_claim",
            "pending_claim_payload_sha256",
            "winner_sha256",
            "task_id",
            "task_readback",
            "task_readback_sha256",
            "sibling_snapshot",
            "sibling_snapshot_sha256",
            "finalized_at_utc",
        },
        label="finalized campaign claim",
    )
    task_id = _require_positive_int(finalized.get("task_id"), "finalized claim task ID")
    if (
        finalized.get("state") != "finalized"
        or finalized.get("claim_reference") != reference
        or finalized.get("pending_claim") != pending
        or finalized.get("pending_claim_payload_sha256") != pending["payload_sha256"]
        or finalized.get("winner_sha256") != pending["winner_sha256"]
    ):
        raise ClaimContractError("finalized campaign claim ancestry drifted")
    if expected_winner is not None:
        normalized_winner = _validate_winner(expected_winner, reference=reference)
        if pending["winner"] != normalized_winner:
            raise ClaimContractError(
                "campaign claim already belongs to different ancestry"
            )
    task_readback = finalized.get("task_readback")
    sibling_snapshot = finalized.get("sibling_snapshot")
    if (
        not isinstance(task_readback, dict)
        or finalized.get("task_readback_sha256") != canonical_sha256(task_readback)
        or not isinstance(sibling_snapshot, dict)
        or finalized.get("sibling_snapshot_sha256")
        != canonical_sha256(sibling_snapshot)
    ):
        raise ClaimContractError("finalized campaign claim evidence drifted")
    _validated_task_readback(
        task_readback,
        task_id=task_id,
        pending=pending,
        evidence_validator=None,
    )
    _validated_sibling_snapshot(
        sibling_snapshot,
        task_id=task_id,
        winner=pending["winner"],
    )
    finalized_time = _utc_timestamp(finalized.get("finalized_at_utc"))
    pending_time = _utc_timestamp(pending.get("created_at_utc"))
    if finalized_time < pending_time:
        raise ClaimContractError(
            "finalized campaign claim predates its pending authority"
        )
    return finalized


def finalize_claim(
    root: Path,
    reference: Mapping[str, Any],
    pending_claim: Mapping[str, Any],
    *,
    task_id: int,
    task_readback: Mapping[str, Any],
    sibling_snapshot: Mapping[str, Any],
    now: str | None = None,
    evidence_validator: (
        Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Finalize the winning claim with one exact Scheduler task readback."""

    task_id = _require_positive_int(task_id, "claim task ID")
    claim_directory, _pending_path, finalized_path = _claim_paths(Path(root), reference)
    _canonical_existing_directory(claim_directory, "campaign claim directory")
    pending = validate_pending_claim(root, reference, claim=pending_claim)
    normalized_task = _validated_task_readback(
        task_readback,
        task_id=task_id,
        pending=pending,
        evidence_validator=evidence_validator,
    )
    normalized_siblings = _validated_sibling_snapshot(
        sibling_snapshot,
        task_id=task_id,
        winner=pending["winner"],
    )
    finalized_at = _utc_timestamp(now)
    if finalized_at < _utc_timestamp(pending["created_at_utc"]):
        raise ClaimContractError(
            "finalized campaign claim predates its pending authority"
        )

    if finalized_path.exists():
        existing = validate_finalized_claim(
            root,
            reference,
            expected_winner=pending["winner"],
        )
        if (
            existing["task_id"] != task_id
            or existing["task_readback"] != normalized_task
            or existing["sibling_snapshot"] != normalized_siblings
        ):
            raise ClaimContractError(
                "campaign claim was finalized with different evidence"
            )
        return existing

    finalized = _seal(
        {
            "schema_version": FINALIZED_CLAIM_SCHEMA,
            "state": "finalized",
            "claim_reference": _json_clone(dict(reference), "campaign claim reference"),
            "pending_claim": pending,
            "pending_claim_payload_sha256": pending["payload_sha256"],
            "winner_sha256": pending["winner_sha256"],
            "task_id": task_id,
            "task_readback": normalized_task,
            "task_readback_sha256": canonical_sha256(normalized_task),
            "sibling_snapshot": normalized_siblings,
            "sibling_snapshot_sha256": canonical_sha256(normalized_siblings),
            "finalized_at_utc": finalized_at,
        }
    )
    try:
        _write_exclusive_json(
            finalized_path,
            finalized,
            label="finalized campaign claim",
        )
    except FileExistsError:
        existing = validate_finalized_claim(
            root,
            reference,
            expected_winner=pending["winner"],
        )
        if (
            existing["task_id"] != task_id
            or existing["task_readback"] != normalized_task
            or existing["sibling_snapshot"] != normalized_siblings
        ):
            raise ClaimContractError(
                "campaign claim was concurrently finalized with different " "evidence"
            )
        return existing
    return validate_finalized_claim(
        root,
        reference,
        claim=finalized,
        expected_winner=pending["winner"],
    )


def validate_finalized_claim(
    root: Path,
    reference: Mapping[str, Any],
    *,
    claim: Mapping[str, Any] | None = None,
    expected_winner: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate durable finalized authority and an optional receipt copy."""

    claim_directory, _pending_path, finalized_path = _claim_paths(Path(root), reference)
    _canonical_existing_directory(claim_directory, "campaign claim directory")
    pending = validate_pending_claim(
        root,
        reference,
        expected_winner=expected_winner,
    )
    durable = _validate_finalized_payload(
        _read_json(finalized_path, "finalized campaign claim"),
        reference=reference,
        pending=pending,
        expected_winner=expected_winner,
    )
    if claim is not None:
        embedded = _validate_finalized_payload(
            _json_clone(dict(claim), "embedded finalized campaign claim"),
            reference=reference,
            pending=pending,
            expected_winner=expected_winner,
        )
        if durable != embedded:
            raise ClaimContractError(
                "embedded finalized claim differs from durable authority"
            )
    return durable


def recover_pending_claim(
    root: Path,
    reference: Mapping[str, Any],
    pending_claim: Mapping[str, Any],
    *,
    matching_tasks: Sequence[Mapping[str, Any]],
    sibling_snapshot: Mapping[str, Any],
    evidence_validator: Callable[
        [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ],
    now: str | None = None,
) -> dict[str, Any]:
    """Finalize a post-POST crash only from exactly one validated API task.

    This function never submits or re-submits work.  Zero, multiple, malformed,
    or validator-rejected readbacks remain pending and require manual review.
    """

    pending = validate_pending_claim(root, reference, claim=pending_claim)
    if (
        not isinstance(matching_tasks, Sequence)
        or isinstance(matching_tasks, (str, bytes, bytearray))
        or len(matching_tasks) != 1
        or not isinstance(matching_tasks[0], Mapping)
    ):
        raise ClaimContractError(
            "pending claim recovery requires exactly one matching API task"
        )
    if not callable(evidence_validator):
        raise ClaimContractError(
            "pending claim recovery requires an evidence validator"
        )
    task = _json_clone(dict(matching_tasks[0]), "pending claim recovery task")
    task_id = _task_id_from_readback(task)
    return finalize_claim(
        root,
        reference,
        pending,
        task_id=task_id,
        task_readback=task,
        sibling_snapshot=sibling_snapshot,
        now=now,
        evidence_validator=evidence_validator,
    )
