"""Pure, fail-closed Phase-B deficit and parent-liveness planning.

The module consumes one sealed bounded-watcher snapshot and emits immutable
Scheduler submission *intents*.  It deliberately has no HTTP client, cancel
surface, sleep loop, filesystem write, or Scheduler mutation method.

Logical child identity is preserved across a failed physical parent.  A
recovery attempt therefore keeps the exact child task, seed, payload hash and
logical dedupe key.  The Phase-B v1 parent dedupe is derived only from those
children, however, so replaying an identical tail would resolve to the dead
parent.  Submission intents consequently use a sealed physical-attempt
dedupe while retaining the byte-identical, validated Phase-B payload.  The
remote Phase-B runner consumes and validates that payload; scientific child
identity is unchanged.
"""

from __future__ import annotations

from collections import defaultdict
import copy
from typing import Any, Mapping, Sequence

try:
    import tier1_final1000_multiseed_contract as phase_a
    import tier1_final1000_multiseed_phase_b_contract as phase_b
    from tier1_final1000_multiseed_phase_b_successor import (
        MUTATION_AUTHORITY_PROTOCOL,
        STAGE_LOGICAL_QUOTAS,
        TOTAL_LOGICAL_TARGET,
    )
    from tier1_final1000_slurm_controller import _refill_task
    from tier1_final1000_slurm_launch import REQUIRED_SCHEDULER_FIELDS, validate_task
    from tier1_final1000_stage_profiles import BY_ID, STAGES
except ImportError:  # pragma: no cover - repository import path
    from tools import tier1_final1000_multiseed_contract as phase_a
    from tools import tier1_final1000_multiseed_phase_b_contract as phase_b
    from tools.tier1_final1000_multiseed_phase_b_successor import (
        MUTATION_AUTHORITY_PROTOCOL,
        STAGE_LOGICAL_QUOTAS,
        TOTAL_LOGICAL_TARGET,
    )
    from tools.tier1_final1000_slurm_controller import _refill_task
    from tools.tier1_final1000_slurm_launch import (
        REQUIRED_SCHEDULER_FIELDS,
        validate_task,
    )
    from tools.tier1_final1000_stage_profiles import BY_ID, STAGES


AUTHORITY_SCHEMA = "mft-tier1-final1000-phase-b-deficit-authority-v1"
CONTROL_FENCE_SCHEMA = "mft-tier1-final1000-phase-b-cas-control-fence-v1"
CAS_COMMIT_RECEIPT_SCHEMA = "mft-tier1-final1000-phase-b-cas-commit-receipt-v1"
OBSERVATION_SCHEMA = "mft-tier1-final1000-phase-b-parent-observation-v1"
INVENTORY_SCHEMA = "mft-tier1-final1000-phase-b-bounded-inventory-v1"
WATCHER_PAGINATION_SCHEMA = "mft-tier1-final1000-phase-b-watcher-pagination-receipt-v1"
INTENT_SCHEMA = "mft-tier1-final1000-phase-b-submission-intent-v1"
CHECKPOINT_SCHEMA = "mft-tier1-final1000-phase-b-deficit-checkpoint-v1"
PLAN_SCHEMA = "mft-tier1-final1000-phase-b-deficit-plan-v1"
CONTROL_FENCE_STATES = frozenset({"active", "stopped", "revoked"})
ACTIVE_SCHEDULER_STATES = frozenset({"queued", "attaching", "running"})
TERMINAL_SCHEDULER_STATES = frozenset(
    {"completed", "failed", "cancelled", "timeout", "timed_out"}
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _seal(value: Mapping[str, Any], seal_field: str) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item) for key, item in value.items() if key != seal_field
    }
    return {**unsigned, seal_field: phase_a.canonical_sha256(unsigned)}


def build_control_fence(
    *,
    owner_identity: str,
    epoch: int,
    state: str,
    phase_a_handoff_lease_sha256: str,
    control_store_capability_sha256: str,
    watcher_capability_sha256: str,
    submitter_capability_sha256: str,
    cas_commit_receipt_sha256: str,
    cas_store_revision: int,
    previous_fence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal one durable compare-and-swap control-record transition.

    This module never performs the CAS.  ``cas_commit_receipt_sha256`` is the
    immutable receipt returned by the external durable control store after a
    successful compare-and-swap and read-back.  A submitter must read that
    store again immediately before every POST; a locally cached fence is not
    mutation authority.
    """

    owner = str(owner_identity).strip()
    epoch = _integer(epoch, "control fence epoch", minimum=1)
    cas_store_revision = _integer(
        cas_store_revision, "control store revision", minimum=1
    )
    state = str(state)
    hashes = (
        phase_a_handoff_lease_sha256,
        control_store_capability_sha256,
        watcher_capability_sha256,
        submitter_capability_sha256,
        cas_commit_receipt_sha256,
    )
    if (
        not owner
        or state not in CONTROL_FENCE_STATES
        or not all(_is_sha256(value) for value in hashes)
    ):
        raise RuntimeError("Phase-B durable CAS control fence is incomplete")
    previous = (
        validate_control_fence(previous_fence) if previous_fence is not None else None
    )
    if previous is None:
        if epoch != 1 or state != "active":
            raise RuntimeError("initial Phase-B control fence must be active epoch 1")
        parent_sha = None
        expected_store_revision = cas_store_revision - 1
    else:
        if previous["state"] != "active":
            raise RuntimeError(
                "stopped/revoked Phase-B control fence cannot transition"
            )
        if epoch != int(previous["epoch"]) + 1:
            raise RuntimeError("Phase-B control fence epoch is not monotonic")
        if cas_store_revision != int(previous["cas_committed_store_revision"]) + 1:
            raise RuntimeError("Phase-B control store revision is not monotonic")
        for field, observed in (
            ("phase_a_handoff_lease_sha256", phase_a_handoff_lease_sha256),
            ("control_store_capability_sha256", control_store_capability_sha256),
            ("watcher_capability_sha256", watcher_capability_sha256),
            ("submitter_capability_sha256", submitter_capability_sha256),
        ):
            if previous[field] != observed:
                raise RuntimeError("Phase-B control fence capability lineage drifted")
        parent_sha = previous["control_fence_sha256"]
        expected_store_revision = int(previous["cas_committed_store_revision"])
    control_record = {
        "schema_version": CONTROL_FENCE_SCHEMA,
        "protocol": MUTATION_AUTHORITY_PROTOCOL,
        "record_kind": "durable-compare-and-swap-current-owner-fence",
        "state": state,
        "epoch": epoch,
        "current_owner_identity": owner,
        "parent_control_fence_sha256": parent_sha,
        "cas_expected_epoch": epoch - 1,
        "cas_committed_epoch": epoch,
        "cas_expected_store_revision": expected_store_revision,
        "cas_committed_store_revision": cas_store_revision,
        "phase_a_handoff_lease_sha256": phase_a_handoff_lease_sha256,
        "control_store_capability_sha256": control_store_capability_sha256,
        "watcher_capability_sha256": watcher_capability_sha256,
        "submitter_capability_sha256": submitter_capability_sha256,
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
    }
    control_record_payload_sha256 = phase_a.canonical_sha256(control_record)
    receipt_unsigned = {
        "schema_version": CAS_COMMIT_RECEIPT_SCHEMA,
        "control_store_capability_sha256": control_store_capability_sha256,
        "expected_parent_control_fence_sha256": parent_sha,
        "expected_store_revision": expected_store_revision,
        "committed_store_revision": cas_store_revision,
        "committed_control_epoch": epoch,
        "committed_owner_identity": owner,
        "committed_state": state,
        "committed_control_record_payload_sha256": control_record_payload_sha256,
        "external_cas_commit_object_sha256": cas_commit_receipt_sha256,
        "compare_and_swap_succeeded": True,
        "current_record_readback_verified": True,
        "scheduler_post_count": 0,
    }
    receipt = _seal(receipt_unsigned, "cas_receipt_sha256")
    unsigned = {
        **control_record,
        "control_record_payload_sha256": control_record_payload_sha256,
        "cas_commit_receipt": receipt,
        "cas_commit_receipt_sha256": receipt["cas_receipt_sha256"],
    }
    return validate_control_fence(_seal(unsigned, "control_fence_sha256"))


def validate_control_fence(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: item for key, item in value.items() if key != "control_fence_sha256"
    }
    required = {
        "schema_version",
        "protocol",
        "record_kind",
        "state",
        "epoch",
        "current_owner_identity",
        "parent_control_fence_sha256",
        "cas_expected_epoch",
        "cas_committed_epoch",
        "cas_expected_store_revision",
        "cas_committed_store_revision",
        "phase_a_handoff_lease_sha256",
        "control_store_capability_sha256",
        "watcher_capability_sha256",
        "submitter_capability_sha256",
        "cas_commit_receipt_sha256",
        "control_record_payload_sha256",
        "cas_commit_receipt",
        "scheduler_post_count",
        "scheduler_cancel_count",
        "control_fence_sha256",
    }
    epoch = _integer(value.get("epoch"), "control fence epoch", minimum=1)
    committed_store_revision = _integer(
        value.get("cas_committed_store_revision"),
        "control store committed revision",
        minimum=1,
    )
    parent = value.get("parent_control_fence_sha256")
    receipt_value = value.get("cas_commit_receipt")
    if not isinstance(receipt_value, Mapping):
        raise RuntimeError("Phase-B CAS commit receipt is missing")
    control_record = {
        key: item
        for key, item in unsigned.items()
        if key
        not in {
            "control_record_payload_sha256",
            "cas_commit_receipt",
            "cas_commit_receipt_sha256",
        }
    }
    receipt_unsigned = {
        key: item for key, item in receipt_value.items() if key != "cas_receipt_sha256"
    }
    expected_receipt = {
        "schema_version": CAS_COMMIT_RECEIPT_SCHEMA,
        "control_store_capability_sha256": value.get("control_store_capability_sha256"),
        "expected_parent_control_fence_sha256": parent,
        "expected_store_revision": value.get("cas_expected_store_revision"),
        "committed_store_revision": committed_store_revision,
        "committed_control_epoch": epoch,
        "committed_owner_identity": value.get("current_owner_identity"),
        "committed_state": value.get("state"),
        "committed_control_record_payload_sha256": value.get(
            "control_record_payload_sha256"
        ),
        "external_cas_commit_object_sha256": receipt_value.get(
            "external_cas_commit_object_sha256"
        ),
        "compare_and_swap_succeeded": True,
        "current_record_readback_verified": True,
        "scheduler_post_count": 0,
    }
    if (
        set(value) != required
        or value.get("schema_version") != CONTROL_FENCE_SCHEMA
        or value.get("protocol") != MUTATION_AUTHORITY_PROTOCOL
        or value.get("record_kind") != "durable-compare-and-swap-current-owner-fence"
        or value.get("state") not in CONTROL_FENCE_STATES
        or not str(value.get("current_owner_identity") or "").strip()
        or (epoch == 1) != (parent is None)
        or (parent is not None and not _is_sha256(parent))
        or value.get("cas_expected_epoch") != epoch - 1
        or value.get("cas_committed_epoch") != epoch
        or value.get("cas_expected_store_revision") != committed_store_revision - 1
        or any(
            not _is_sha256(value.get(field))
            for field in (
                "phase_a_handoff_lease_sha256",
                "control_store_capability_sha256",
                "watcher_capability_sha256",
                "submitter_capability_sha256",
            )
        )
        or value.get("control_record_payload_sha256")
        != phase_a.canonical_sha256(control_record)
        or set(receipt_value) != set(expected_receipt) | {"cas_receipt_sha256"}
        or receipt_unsigned != expected_receipt
        or not _is_sha256(receipt_value.get("external_cas_commit_object_sha256"))
        or receipt_value.get("cas_receipt_sha256")
        != phase_a.canonical_sha256(receipt_unsigned)
        or value.get("cas_commit_receipt_sha256")
        != receipt_value.get("cas_receipt_sha256")
        or value.get("scheduler_post_count") != 0
        or value.get("scheduler_cancel_count") != 0
        or value.get("control_fence_sha256") != phase_a.canonical_sha256(unsigned)
    ):
        raise RuntimeError("Phase-B durable CAS control fence mismatch")
    if epoch == 1 and value.get("state") != "active":
        raise RuntimeError("initial Phase-B control fence must be active")
    return copy.deepcopy(dict(value))


def build_authority_lease(*, control_fence: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the sole dry-planning authority deterministically from CAS."""

    fence = validate_control_fence(control_fence)
    if fence["state"] != "active":
        raise RuntimeError("Phase-B authority requires an active control fence")
    unsigned = {
        "schema_version": AUTHORITY_SCHEMA,
        "protocol": MUTATION_AUTHORITY_PROTOCOL,
        "owner_kind": "phase-b-deficit-liveness-controller",
        "owner_identity": fence["current_owner_identity"],
        "owner_count": 1,
        "exclusive": True,
        "simultaneous_owners_allowed": False,
        "control_fence": fence,
        "control_fence_sha256": fence["control_fence_sha256"],
        "control_epoch": fence["epoch"],
        "control_state": "active",
        "phase_a_handoff_lease_sha256": fence["phase_a_handoff_lease_sha256"],
        "phase_a_authority_relinquished": True,
        "submission_intent_authority_granted": True,
        "scheduler_write_authorized": False,
        "scheduler_client_present": False,
        "allowed_output": "sealed-scheduler-submission-intents-only",
        "cancellation_allowed": False,
        "preemption_allowed": False,
    }
    return validate_authority_lease(_seal(unsigned, "authority_lease_sha256"))


def validate_authority_lease(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: item for key, item in value.items() if key != "authority_lease_sha256"
    }
    required = {
        "schema_version",
        "protocol",
        "owner_kind",
        "owner_identity",
        "owner_count",
        "exclusive",
        "simultaneous_owners_allowed",
        "control_fence",
        "control_fence_sha256",
        "control_epoch",
        "control_state",
        "phase_a_handoff_lease_sha256",
        "phase_a_authority_relinquished",
        "submission_intent_authority_granted",
        "scheduler_write_authorized",
        "scheduler_client_present",
        "allowed_output",
        "cancellation_allowed",
        "preemption_allowed",
        "authority_lease_sha256",
    }
    fence_value = value.get("control_fence")
    if not isinstance(fence_value, Mapping):
        raise RuntimeError("Phase-B authority has no durable CAS control fence")
    fence = validate_control_fence(fence_value)
    if (
        set(value) != required
        or value.get("schema_version") != AUTHORITY_SCHEMA
        or value.get("protocol") != MUTATION_AUTHORITY_PROTOCOL
        or value.get("owner_kind") != "phase-b-deficit-liveness-controller"
        or not str(value.get("owner_identity") or "").strip()
        or value.get("owner_count") != 1
        or value.get("exclusive") is not True
        or value.get("simultaneous_owners_allowed") is not False
        or value.get("owner_identity") != fence["current_owner_identity"]
        or value.get("control_fence_sha256") != fence["control_fence_sha256"]
        or value.get("control_epoch") != fence["epoch"]
        or value.get("control_state") != "active"
        or fence["state"] != "active"
        or value.get("phase_a_handoff_lease_sha256")
        != fence["phase_a_handoff_lease_sha256"]
        or value.get("phase_a_authority_relinquished") is not True
        or value.get("submission_intent_authority_granted") is not True
        or value.get("scheduler_write_authorized") is not False
        or value.get("scheduler_client_present") is not False
        or value.get("allowed_output") != "sealed-scheduler-submission-intents-only"
        or value.get("cancellation_allowed") is not False
        or value.get("preemption_allowed") is not False
        or value.get("authority_lease_sha256") != phase_a.canonical_sha256(unsigned)
    ):
        raise RuntimeError("Phase-B single mutation authority lease mismatch")
    return copy.deepcopy(dict(value))


def require_current_authority(
    authority: Mapping[str, Any], *, current_control_fence: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fence stale, stopped, revoked, or concurrently replaced controllers."""

    sealed_authority = validate_authority_lease(authority)
    current = validate_control_fence(current_control_fence)
    if current["state"] != "active":
        raise RuntimeError(
            f"Phase-B control fence is {current['state']}; submission intents forbidden"
        )
    if current != sealed_authority["control_fence"]:
        raise RuntimeError("Phase-B authority is not the current durable CAS owner")
    return sealed_authority, current


def _intent_child_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage_id": str(item["stage_id"]),
        "seed": int(item["seed"]),
        "logical_dedupe_key": str(item["logical_dedupe_key"]),
        "child_task_sha256": str(item["child_task_sha256"]),
        "disposition": str(item["disposition"]),
        "source_parent_dedupe_keys": sorted(
            str(value) for value in item["source_parent_dedupe_keys"]
        ),
    }


def _build_submission_intent(
    base_parent: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    *,
    authority: Mapping[str, Any],
    plan_generation: int,
    intent_ordinal: int,
) -> dict[str, Any]:
    authority = validate_authority_lease(authority)
    canonical = phase_b.validate_batch_task(base_parent)
    summaries = [_intent_child_summary(item) for item in items]
    base_sha = phase_a.canonical_sha256(canonical)
    attempt_identity = {
        "schema_version": INTENT_SCHEMA,
        "authority_lease_sha256": authority["authority_lease_sha256"],
        "control_fence_sha256": authority["control_fence_sha256"],
        "control_epoch": authority["control_epoch"],
        "plan_generation": int(plan_generation),
        "intent_ordinal": int(intent_ordinal),
        "canonical_parent_task_sha256": base_sha,
        "ordered_children": summaries,
    }
    physical_dedupe = phase_b.PHYSICAL_ATTEMPT_DEDUPE_PREFIX + phase_a.canonical_sha256(
        attempt_identity
    )
    task = copy.deepcopy(canonical)
    task["dedupe_key"] = physical_dedupe
    unsigned = {
        "schema_version": INTENT_SCHEMA,
        "phase_b_protocol_version": phase_b.PROTOCOL_VERSION,
        "authority_lease_sha256": authority["authority_lease_sha256"],
        "control_fence_sha256": authority["control_fence_sha256"],
        "control_epoch": authority["control_epoch"],
        "submitter_capability_sha256": authority["control_fence"][
            "submitter_capability_sha256"
        ],
        "plan_generation": int(plan_generation),
        "intent_ordinal": int(intent_ordinal),
        "action": "submit-phase-b-parent",
        "scheduler_endpoint": "POST /api/tasks",
        "canonical_parent_task_sha256": base_sha,
        "physical_parent_dedupe_key": physical_dedupe,
        "ordered_children": summaries,
        "parent_task": task,
        "parent_task_sha256": phase_a.canonical_sha256(task),
        "logical_child_identity_preserved": True,
        "phase_b_payload_contract_validated": True,
        "new_physical_parent_identity_required": True,
        "checkpoint_before_post_required": True,
        "submit_before_checkpoint_allowed": False,
        "post_requires_current_control_fence_cas_recheck": True,
        "scheduler_write_performed": False,
        "http_request_performed": False,
        "cancellation_performed": False,
        "preemption_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return validate_submission_intent(
        _seal(unsigned, "submission_intent_sha256"), authority=authority
    )


def validate_submission_intent(
    value: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    authority = validate_authority_lease(authority)
    unsigned = {
        key: item for key, item in value.items() if key != "submission_intent_sha256"
    }
    required = {
        "schema_version",
        "phase_b_protocol_version",
        "authority_lease_sha256",
        "control_fence_sha256",
        "control_epoch",
        "submitter_capability_sha256",
        "plan_generation",
        "intent_ordinal",
        "action",
        "scheduler_endpoint",
        "canonical_parent_task_sha256",
        "physical_parent_dedupe_key",
        "ordered_children",
        "parent_task",
        "parent_task_sha256",
        "logical_child_identity_preserved",
        "phase_b_payload_contract_validated",
        "new_physical_parent_identity_required",
        "checkpoint_before_post_required",
        "submit_before_checkpoint_allowed",
        "post_requires_current_control_fence_cas_recheck",
        "scheduler_write_performed",
        "http_request_performed",
        "cancellation_performed",
        "preemption_performed",
        "fea_submission_performed",
        "aedt_used",
        "submission_intent_sha256",
    }
    task = value.get("parent_task")
    children = value.get("ordered_children")
    if not isinstance(task, Mapping) or not isinstance(children, list) or not children:
        raise RuntimeError("Phase-B submission intent task inventory is missing")
    canonical = phase_b.canonical_parent_from_runtime_task(task)
    payload_children = canonical["payload_json"]["children"]
    if len(children) != len(payload_children):
        raise RuntimeError("Phase-B submission intent child count drifted")
    normalized_children: list[dict[str, Any]] = []
    for summary, child in zip(children, payload_children):
        if not isinstance(summary, Mapping) or set(summary) != {
            "stage_id",
            "seed",
            "logical_dedupe_key",
            "child_task_sha256",
            "disposition",
            "source_parent_dedupe_keys",
        }:
            raise RuntimeError("Phase-B submission child summary drifted")
        sources = summary.get("source_parent_dedupe_keys")
        if (
            summary.get("stage_id") != canonical["payload_json"]["stage_id"]
            or summary.get("seed") != child["seed"]
            or summary.get("logical_dedupe_key") != child["logical_dedupe_key"]
            or summary.get("child_task_sha256") != child["child_task_sha256"]
            or summary.get("disposition") not in {"recover-incomplete", "fresh-refill"}
            or not isinstance(sources, list)
            or sources != sorted(set(sources))
            or (summary.get("disposition") == "recover-incomplete" and not sources)
            or (summary.get("disposition") == "fresh-refill" and sources)
        ):
            raise RuntimeError("Phase-B submission child identity drifted")
        normalized_children.append(copy.deepcopy(dict(summary)))
    generation = _integer(
        value.get("plan_generation"), "intent plan generation", minimum=1
    )
    ordinal = _integer(value.get("intent_ordinal"), "intent ordinal")
    base_sha = phase_a.canonical_sha256(canonical)
    identity = {
        "schema_version": INTENT_SCHEMA,
        "authority_lease_sha256": authority["authority_lease_sha256"],
        "control_fence_sha256": authority["control_fence_sha256"],
        "control_epoch": authority["control_epoch"],
        "plan_generation": generation,
        "intent_ordinal": ordinal,
        "canonical_parent_task_sha256": base_sha,
        "ordered_children": normalized_children,
    }
    expected_dedupe = phase_b.PHYSICAL_ATTEMPT_DEDUPE_PREFIX + phase_a.canonical_sha256(
        identity
    )
    if (
        set(value) != required
        or value.get("schema_version") != INTENT_SCHEMA
        or value.get("phase_b_protocol_version") != phase_b.PROTOCOL_VERSION
        or value.get("authority_lease_sha256") != authority["authority_lease_sha256"]
        or value.get("control_fence_sha256") != authority["control_fence_sha256"]
        or value.get("control_epoch") != authority["control_epoch"]
        or value.get("submitter_capability_sha256")
        != authority["control_fence"]["submitter_capability_sha256"]
        or value.get("action") != "submit-phase-b-parent"
        or value.get("scheduler_endpoint") != "POST /api/tasks"
        or value.get("canonical_parent_task_sha256") != base_sha
        or value.get("physical_parent_dedupe_key") != expected_dedupe
        or task.get("dedupe_key") != expected_dedupe
        or value.get("parent_task_sha256") != phase_a.canonical_sha256(task)
        or value.get("logical_child_identity_preserved") is not True
        or value.get("phase_b_payload_contract_validated") is not True
        or value.get("new_physical_parent_identity_required") is not True
        or value.get("checkpoint_before_post_required") is not True
        or value.get("submit_before_checkpoint_allowed") is not False
        or value.get("post_requires_current_control_fence_cas_recheck") is not True
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "http_request_performed",
                "cancellation_performed",
                "preemption_performed",
                "fea_submission_performed",
                "aedt_used",
            )
        )
        or value.get("submission_intent_sha256") != phase_a.canonical_sha256(unsigned)
    ):
        raise RuntimeError("Phase-B sealed Scheduler submission intent mismatch")
    return copy.deepcopy(dict(value))


def seal_parent_observation(
    *,
    task_id: int,
    scheduler_status: str,
    parent_task: Mapping[str, Any],
    batch_manifest: Mapping[str, Any] | None = None,
    task_status: Mapping[str, Any] | None = None,
    child_receipts: Sequence[Mapping[str, Any]] = (),
    submission_intent: Mapping[str, Any] | None = None,
    authority: Mapping[str, Any] | None = None,
    terminal_receipt_recovery_attestation: Mapping[str, Any] | None = None,
    artifact_inventory_complete: bool = True,
) -> dict[str, Any]:
    """Authenticate one bounded-watcher parent and its durable artifacts."""

    task_id = _integer(task_id, "Scheduler task id", minimum=1)
    scheduler_status = str(scheduler_status)
    if scheduler_status not in ACTIVE_SCHEDULER_STATES | TERMINAL_SCHEDULER_STATES:
        raise RuntimeError("Phase-B Scheduler parent status is unsupported")
    if submission_intent is None:
        task = phase_b.validate_batch_task(parent_task)
    else:
        if authority is None:
            raise RuntimeError("attempt observation requires its authority lease")
        intent = validate_submission_intent(submission_intent, authority=authority)
        if dict(parent_task) != intent["parent_task"]:
            raise RuntimeError("attempt observation task/intent binding mismatch")
        task = copy.deepcopy(dict(parent_task))
        submission_intent = intent
    expected_manifest = phase_b.batch_manifest_from_payload(task["payload_json"])
    manifest = None
    if batch_manifest is not None:
        manifest = phase_b.validate_batch_manifest(batch_manifest)
        if manifest != expected_manifest:
            raise RuntimeError("Phase-B observation manifest/payload binding mismatch")
    status = None
    if task_status is not None:
        if manifest is None:
            raise RuntimeError("Phase-B status requires an authenticated manifest")
        status = phase_b.validate_task_status(task_status)
        if (
            str(status["task_id"]) != str(task_id)
            or status["manifest_sha256"] != manifest["manifest_sha256"]
        ):
            raise RuntimeError("Phase-B observation status identity mismatch")
    receipts: list[dict[str, Any]] = []
    if child_receipts and manifest is None:
        raise RuntimeError("Phase-B receipts require an authenticated manifest")
    for receipt in child_receipts:
        sealed = phase_b.validate_child_receipt(receipt, manifest=manifest)
        if str(sealed["task_id"]) != str(task_id):
            raise RuntimeError("Phase-B observation receipt task id mismatch")
        receipts.append(sealed)
    receipts.sort(key=lambda item: int(item["ordinal"]))
    if [item["ordinal"] for item in receipts] != list(range(len(receipts))):
        raise RuntimeError("Phase-B observation receipts are not a contiguous prefix")
    if len({item["receipt_sha256"] for item in receipts}) != len(receipts):
        raise RuntimeError("Phase-B observation contains duplicate receipts")

    terminal_status = (
        status is not None and status["state"] in phase_a.TERMINAL_PARENT_STATES
    )
    if scheduler_status in {"queued", "attaching"}:
        if manifest is not None or status is not None or receipts:
            raise RuntimeError("queued Phase-B parent has ambiguous remote artifacts")
    elif scheduler_status == "running":
        if manifest is None or status is None or terminal_status:
            raise RuntimeError("running Phase-B parent lacks a live sealed journal")
    elif scheduler_status == "completed":
        if (
            manifest is None
            or status is None
            or status["state"] not in {"completed", "completed_with_failures"}
        ):
            raise RuntimeError("completed Scheduler parent lacks terminal evidence")
    elif status is not None and status["state"] in {
        "completed",
        "completed_with_failures",
    }:
        raise RuntimeError("failed Scheduler parent contradicts a completed journal")

    if status is not None:
        cursor = int(status["sealed_child_count"])
        if len(receipts) < cursor:
            raise RuntimeError("Phase-B status cursor exceeds durable receipts")
        prefix = receipts[:cursor]
        if (
            sum(item["state"] == "completed" for item in prefix)
            != status["completed_child_count"]
            or sum(item["state"] != "completed" for item in prefix)
            != status["failed_child_count"]
        ):
            raise RuntimeError("Phase-B status cursor/receipt counts contradict")
        if terminal_status and len(receipts) != cursor:
            raise RuntimeError("terminal Phase-B status has receipts beyond its cursor")
        if status["state"] in {"completed", "completed_with_failures"} and (
            cursor != task["payload_json"]["batch_length"]
        ):
            raise RuntimeError("completed Phase-B journal omitted logical children")
    receipts_beyond_terminal_cursor = (
        scheduler_status in TERMINAL_SCHEDULER_STATES
        and bool(receipts)
        and (status is None or len(receipts) > int(status["sealed_child_count"]))
    )
    recovery_attestation = None
    if receipts_beyond_terminal_cursor:
        if (
            authority is None
            or manifest is None
            or status is None
            or not isinstance(terminal_receipt_recovery_attestation, Mapping)
        ):
            raise RuntimeError(
                "terminal Phase-B receipts beyond the status cursor require "
                "a shared recovery attestation"
            )
        sealed_authority = validate_authority_lease(authority)
        watcher_revision = terminal_receipt_recovery_attestation.get(
            "watcher_revision_sha256"
        )
        recovery_attestation = phase_b.validate_terminal_receipt_recovery_attestation(
            terminal_receipt_recovery_attestation,
            parent_task=task,
            manifest=manifest,
            task_status=status,
            task_id=task_id,
            scheduler_state=scheduler_status,
            expected_watcher_capability_sha256=sealed_authority["control_fence"][
                "watcher_capability_sha256"
            ],
            expected_watcher_revision_sha256=watcher_revision,
        )
        attested_receipts = [
            record["receipt"]
            for record in recovery_attestation["receipt_scan"]
            if record["present"]
        ]
        if attested_receipts != receipts:
            raise RuntimeError(
                "terminal Phase-B recovery attestation/receipt inventory mismatch"
            )
    elif terminal_receipt_recovery_attestation is not None:
        raise RuntimeError("terminal Phase-B recovery attestation is not required")
    if (
        scheduler_status in TERMINAL_SCHEDULER_STATES
        and receipts
        and (status is None or len(receipts) > int(status["sealed_child_count"]))
        and recovery_attestation is None
    ):
        raise RuntimeError("terminal Phase-B recovery attestation was not accepted")
    if artifact_inventory_complete is not True:
        raise RuntimeError("Phase-B artifact inventory must be complete")

    unsigned = {
        "schema_version": OBSERVATION_SCHEMA,
        "task_id": task_id,
        "scheduler_status": scheduler_status,
        "scheduler_terminal": scheduler_status in TERMINAL_SCHEDULER_STATES,
        "parent_task": copy.deepcopy(task),
        "parent_task_sha256": phase_a.canonical_sha256(task),
        "submission_intent": copy.deepcopy(submission_intent),
        "terminal_receipt_recovery_attestation": copy.deepcopy(recovery_attestation),
        "batch_manifest": copy.deepcopy(manifest),
        "task_status": copy.deepcopy(status),
        "child_receipts": receipts,
        "artifact_inventory_complete": True,
        "artifact_inventory_sha256": phase_a.canonical_sha256(
            {
                "manifest": manifest,
                "task_status": status,
                "child_receipts": receipts,
                "terminal_receipt_recovery_attestation": recovery_attestation,
            }
        ),
    }
    return _seal(unsigned, "observation_sha256")


def validate_parent_observation(
    value: Mapping[str, Any], *, authority: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "observation_sha256"}
    required = {
        "schema_version",
        "task_id",
        "scheduler_status",
        "scheduler_terminal",
        "parent_task",
        "parent_task_sha256",
        "submission_intent",
        "terminal_receipt_recovery_attestation",
        "batch_manifest",
        "task_status",
        "child_receipts",
        "artifact_inventory_complete",
        "artifact_inventory_sha256",
        "observation_sha256",
    }
    if set(value) != required or value.get("schema_version") != OBSERVATION_SCHEMA:
        raise RuntimeError("Phase-B parent observation fields drifted")
    rebuilt = seal_parent_observation(
        task_id=_integer(value.get("task_id"), "Scheduler task id", minimum=1),
        scheduler_status=str(value.get("scheduler_status") or ""),
        parent_task=value.get("parent_task") or {},
        batch_manifest=value.get("batch_manifest"),
        task_status=value.get("task_status"),
        child_receipts=value.get("child_receipts") or [],
        submission_intent=value.get("submission_intent"),
        authority=authority,
        terminal_receipt_recovery_attestation=value.get(
            "terminal_receipt_recovery_attestation"
        ),
        artifact_inventory_complete=value.get("artifact_inventory_complete") is True,
    )
    if dict(value) != rebuilt or value.get(
        "observation_sha256"
    ) != phase_a.canonical_sha256(unsigned):
        raise RuntimeError("Phase-B parent observation seal mismatch")
    return copy.deepcopy(dict(value))


def build_watcher_pagination_receipt(
    *,
    authority: Mapping[str, Any],
    current_control_fence: Mapping[str, Any],
    namespace_fence_sha256: str,
    page_task_ids: Sequence[Sequence[int]],
    page_cursor_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Seal the read-only watcher's exact, fully paged Scheduler inventory."""

    authority, fence = require_current_authority(
        authority, current_control_fence=current_control_fence
    )
    if not _is_sha256(namespace_fence_sha256):
        raise RuntimeError("bounded watcher namespace fence is not authenticated")
    if (
        not isinstance(page_task_ids, Sequence)
        or isinstance(page_task_ids, (str, bytes))
        or not page_task_ids
        or len(page_task_ids) != len(page_cursor_sha256s)
    ):
        raise RuntimeError("bounded watcher page inventory is incomplete")
    pages: list[dict[str, Any]] = []
    flattened: list[int] = []
    for index, (task_ids_value, cursor_sha) in enumerate(
        zip(page_task_ids, page_cursor_sha256s)
    ):
        if (
            not isinstance(task_ids_value, Sequence)
            or isinstance(task_ids_value, (str, bytes))
            or not _is_sha256(cursor_sha)
        ):
            raise RuntimeError("bounded watcher page identity is invalid")
        task_ids = [
            _integer(task_id, "watcher Scheduler task id", minimum=1)
            for task_id in task_ids_value
        ]
        page_unsigned = {
            "page_index": index,
            "page_cursor_sha256": cursor_sha,
            "task_ids": task_ids,
            "task_count": len(task_ids),
            "scheduler_http_method": "GET",
            "scheduler_post_count": 0,
        }
        pages.append(_seal(page_unsigned, "page_sha256"))
        flattened.extend(task_ids)
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("bounded watcher pages repeat a physical task id")
    unsigned = {
        "schema_version": WATCHER_PAGINATION_SCHEMA,
        "authority_lease_sha256": authority["authority_lease_sha256"],
        "control_fence_sha256": fence["control_fence_sha256"],
        "control_epoch": fence["epoch"],
        "watcher_capability_sha256": fence["watcher_capability_sha256"],
        "namespace_fence_sha256": namespace_fence_sha256,
        "query_kind": "bounded-current-phase-b-cohort-full-pagination",
        "pagination_complete": True,
        "page_count": len(pages),
        "pages": pages,
        "exact_task_ids": flattened,
        "exact_task_count": len(flattened),
        "exact_task_ids_sha256": phase_a.canonical_sha256(flattened),
        "scheduler_get_count": len(pages),
        "scheduler_post_count": 0,
        "scheduler_cancel_count": 0,
    }
    return validate_watcher_pagination_receipt(
        _seal(unsigned, "watcher_pagination_receipt_sha256"), authority=authority
    )


def validate_watcher_pagination_receipt(
    value: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    authority = validate_authority_lease(authority)
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "watcher_pagination_receipt_sha256"
    }
    pages_value = value.get("pages")
    if not isinstance(pages_value, list) or not pages_value:
        raise RuntimeError("bounded watcher pagination receipt has no pages")
    pages: list[dict[str, Any]] = []
    flattened: list[int] = []
    for index, page in enumerate(pages_value):
        if not isinstance(page, Mapping):
            raise RuntimeError("bounded watcher page receipt is invalid")
        page_unsigned = {
            key: item for key, item in page.items() if key != "page_sha256"
        }
        task_ids_value = page.get("task_ids")
        if not isinstance(task_ids_value, list):
            raise RuntimeError("bounded watcher page task inventory is missing")
        task_ids = [
            _integer(task_id, "watcher Scheduler task id", minimum=1)
            for task_id in task_ids_value
        ]
        if (
            set(page)
            != {
                "page_index",
                "page_cursor_sha256",
                "task_ids",
                "task_count",
                "scheduler_http_method",
                "scheduler_post_count",
                "page_sha256",
            }
            or page.get("page_index") != index
            or not _is_sha256(page.get("page_cursor_sha256"))
            or page.get("task_count") != len(task_ids)
            or page.get("scheduler_http_method") != "GET"
            or page.get("scheduler_post_count") != 0
            or page.get("page_sha256") != phase_a.canonical_sha256(page_unsigned)
        ):
            raise RuntimeError("bounded watcher page receipt seal mismatch")
        pages.append(copy.deepcopy(dict(page)))
        flattened.extend(task_ids)
    required = {
        "schema_version",
        "authority_lease_sha256",
        "control_fence_sha256",
        "control_epoch",
        "watcher_capability_sha256",
        "namespace_fence_sha256",
        "query_kind",
        "pagination_complete",
        "page_count",
        "pages",
        "exact_task_ids",
        "exact_task_count",
        "exact_task_ids_sha256",
        "scheduler_get_count",
        "scheduler_post_count",
        "scheduler_cancel_count",
        "watcher_pagination_receipt_sha256",
    }
    exact_task_ids = value.get("exact_task_ids")
    if (
        set(value) != required
        or value.get("schema_version") != WATCHER_PAGINATION_SCHEMA
        or value.get("authority_lease_sha256") != authority["authority_lease_sha256"]
        or value.get("control_fence_sha256") != authority["control_fence_sha256"]
        or value.get("control_epoch") != authority["control_epoch"]
        or value.get("watcher_capability_sha256")
        != authority["control_fence"]["watcher_capability_sha256"]
        or not _is_sha256(value.get("namespace_fence_sha256"))
        or value.get("query_kind") != "bounded-current-phase-b-cohort-full-pagination"
        or value.get("pagination_complete") is not True
        or value.get("page_count") != len(pages)
        or not isinstance(exact_task_ids, list)
        or exact_task_ids != flattened
        or value.get("exact_task_count") != len(flattened)
        or value.get("exact_task_ids_sha256") != phase_a.canonical_sha256(flattened)
        or len(flattened) != len(set(flattened))
        or value.get("scheduler_get_count") != len(pages)
        or value.get("scheduler_post_count") != 0
        or value.get("scheduler_cancel_count") != 0
        or value.get("watcher_pagination_receipt_sha256")
        != phase_a.canonical_sha256(unsigned)
    ):
        raise RuntimeError("bounded watcher pagination receipt seal mismatch")
    return copy.deepcopy(dict(value))


def _require_recovery_watcher_bindings(
    rows: Sequence[Mapping[str, Any]], watcher_receipt: Mapping[str, Any]
) -> None:
    for row in rows:
        attestation = row.get("terminal_receipt_recovery_attestation")
        if attestation is None:
            continue
        if (
            attestation.get("watcher_revision_sha256")
            != watcher_receipt["watcher_pagination_receipt_sha256"]
            or attestation.get("watcher_capability_sha256")
            != watcher_receipt["watcher_capability_sha256"]
            or attestation.get("scheduler_post_count") != 0
        ):
            raise RuntimeError(
                "terminal Phase-B recovery attestation/watcher pagination mismatch"
            )


def build_bounded_inventory_snapshot(
    observations: Sequence[Mapping[str, Any]],
    *,
    authority: Mapping[str, Any],
    watcher_pagination_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind authenticated observations to one exact paged watcher receipt."""

    authority = validate_authority_lease(authority)
    receipt = validate_watcher_pagination_receipt(
        watcher_pagination_receipt, authority=authority
    )
    rows = [
        validate_parent_observation(item, authority=authority) for item in observations
    ]
    rows.sort(key=lambda item: int(item["task_id"]))
    _require_recovery_watcher_bindings(rows, receipt)
    task_ids = [int(item["task_id"]) for item in rows]
    parent_dedupes = [str(item["parent_task"]["dedupe_key"]) for item in rows]
    if (
        task_ids != sorted(int(task_id) for task_id in receipt["exact_task_ids"])
        or len(task_ids) != len(set(task_ids))
        or len(parent_dedupes) != len(set(parent_dedupes))
    ):
        raise RuntimeError("bounded watcher receipt/observation inventory mismatch")
    unsigned = {
        "schema_version": INVENTORY_SCHEMA,
        "authority_lease_sha256": authority["authority_lease_sha256"],
        "control_fence_sha256": authority["control_fence_sha256"],
        "watch_mode": "sealed-full-pagination-receipt-v1",
        "watcher_pagination_receipt": receipt,
        "watcher_pagination_receipt_sha256": receipt[
            "watcher_pagination_receipt_sha256"
        ],
        "active_namespace_complete": True,
        "observed_terminal_lineage_complete": True,
        "scheduler_write_count": 0,
        "observations": rows,
        "observation_count": len(rows),
        "observation_inventory_sha256": phase_a.canonical_sha256(
            [item["observation_sha256"] for item in rows]
        ),
    }
    return validate_bounded_inventory_snapshot(
        _seal(unsigned, "inventory_snapshot_sha256"), authority=authority
    )


def validate_bounded_inventory_snapshot(
    value: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    authority = validate_authority_lease(authority)
    unsigned = {
        key: item for key, item in value.items() if key != "inventory_snapshot_sha256"
    }
    rows_value = value.get("observations")
    receipt_value = value.get("watcher_pagination_receipt")
    if not isinstance(rows_value, list) or not isinstance(receipt_value, Mapping):
        raise RuntimeError("bounded Phase-B observation inventory is missing")
    rows = [
        validate_parent_observation(item, authority=authority) for item in rows_value
    ]
    receipt = validate_watcher_pagination_receipt(receipt_value, authority=authority)
    _require_recovery_watcher_bindings(rows, receipt)
    task_ids = [int(item["task_id"]) for item in rows]
    required = {
        "schema_version",
        "authority_lease_sha256",
        "control_fence_sha256",
        "watch_mode",
        "watcher_pagination_receipt",
        "watcher_pagination_receipt_sha256",
        "active_namespace_complete",
        "observed_terminal_lineage_complete",
        "scheduler_write_count",
        "observations",
        "observation_count",
        "observation_inventory_sha256",
        "inventory_snapshot_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != INVENTORY_SCHEMA
        or value.get("authority_lease_sha256") != authority["authority_lease_sha256"]
        or value.get("control_fence_sha256") != authority["control_fence_sha256"]
        or value.get("watch_mode") != "sealed-full-pagination-receipt-v1"
        or value.get("watcher_pagination_receipt_sha256")
        != receipt["watcher_pagination_receipt_sha256"]
        or value.get("active_namespace_complete") is not True
        or value.get("observed_terminal_lineage_complete") is not True
        or value.get("scheduler_write_count") != 0
        or value.get("observation_count") != len(rows)
        or value.get("observation_inventory_sha256")
        != phase_a.canonical_sha256([item["observation_sha256"] for item in rows])
        or value.get("inventory_snapshot_sha256") != phase_a.canonical_sha256(unsigned)
        or rows != sorted(rows, key=lambda item: int(item["task_id"]))
        or task_ids != sorted(int(task_id) for task_id in receipt["exact_task_ids"])
        or len({item["task_id"] for item in rows}) != len(rows)
        or len({item["parent_task"]["dedupe_key"] for item in rows}) != len(rows)
    ):
        raise RuntimeError("bounded Phase-B inventory snapshot seal mismatch")
    return copy.deepcopy(dict(value))


def build_checkpoint(
    *,
    authority: Mapping[str, Any],
    next_seed_by_stage: Mapping[str, Any],
    generation: int = 0,
    pending_submission_intents: Sequence[Mapping[str, Any]] = (),
    last_inventory_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    authority = validate_authority_lease(authority)
    generation = _integer(generation, "checkpoint generation")
    next_seeds: dict[str, int] = {}
    if set(next_seed_by_stage) != set(STAGE_LOGICAL_QUOTAS):
        raise RuntimeError("Phase-B checkpoint seed cursors are incomplete")
    for stage in STAGES:
        seed = _integer(
            next_seed_by_stage[stage.stage_id], f"{stage.stage_id} next seed"
        )
        if not stage.seed_start <= seed < stage.seed_window_end_exclusive:
            raise RuntimeError("Phase-B checkpoint seed cursor escaped its window")
        next_seeds[stage.stage_id] = seed
    pending = [
        validate_submission_intent(item, authority=authority)
        for item in pending_submission_intents
    ]
    pending.sort(
        key=lambda item: (int(item["plan_generation"]), int(item["intent_ordinal"]))
    )
    if any(int(item["plan_generation"]) > generation for item in pending):
        raise RuntimeError("Phase-B checkpoint contains a future-generation intent")
    if len({item["physical_parent_dedupe_key"] for item in pending}) != len(pending):
        raise RuntimeError("Phase-B checkpoint contains duplicate pending intents")
    pending_logical_children = [
        child["logical_dedupe_key"]
        for intent in pending
        for child in intent["ordered_children"]
    ]
    if len(pending_logical_children) != len(set(pending_logical_children)):
        raise RuntimeError("Phase-B checkpoint repeats a pending logical child")
    if last_inventory_snapshot_sha256 is not None and not _is_sha256(
        last_inventory_snapshot_sha256
    ):
        raise RuntimeError("Phase-B checkpoint inventory seal is invalid")
    unsigned = {
        "schema_version": CHECKPOINT_SCHEMA,
        "authority_lease_sha256": authority["authority_lease_sha256"],
        "control_fence_sha256": authority["control_fence_sha256"],
        "control_epoch": authority["control_epoch"],
        "control_state": "active",
        "stop_requested": False,
        "generation": generation,
        "next_seed_by_stage": next_seeds,
        "pending_submission_intents": pending,
        "last_inventory_snapshot_sha256": last_inventory_snapshot_sha256,
        "checkpoint_persist_before_post_required": True,
        "submit_before_checkpoint_allowed": False,
        "post_requires_current_control_fence_cas_recheck": True,
        "scheduler_write_performed": False,
        "cancellation_performed": False,
        "preemption_performed": False,
    }
    return _seal(unsigned, "checkpoint_sha256")


def validate_checkpoint(
    value: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    authority = validate_authority_lease(authority)
    unsigned = {key: item for key, item in value.items() if key != "checkpoint_sha256"}
    required = {
        "schema_version",
        "authority_lease_sha256",
        "control_fence_sha256",
        "control_epoch",
        "control_state",
        "stop_requested",
        "generation",
        "next_seed_by_stage",
        "pending_submission_intents",
        "last_inventory_snapshot_sha256",
        "checkpoint_persist_before_post_required",
        "submit_before_checkpoint_allowed",
        "post_requires_current_control_fence_cas_recheck",
        "scheduler_write_performed",
        "cancellation_performed",
        "preemption_performed",
        "checkpoint_sha256",
    }
    if set(value) != required or value.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError("Phase-B deficit checkpoint fields drifted")
    rebuilt = build_checkpoint(
        authority=authority,
        next_seed_by_stage=value.get("next_seed_by_stage") or {},
        generation=_integer(value.get("generation"), "checkpoint generation"),
        pending_submission_intents=value.get("pending_submission_intents") or [],
        last_inventory_snapshot_sha256=value.get("last_inventory_snapshot_sha256"),
    )
    if (
        value.get("authority_lease_sha256") != authority["authority_lease_sha256"]
        or value.get("control_fence_sha256") != authority["control_fence_sha256"]
        or value.get("control_epoch") != authority["control_epoch"]
        or value.get("control_state") != "active"
        or value.get("stop_requested") is not False
        or value.get("checkpoint_persist_before_post_required") is not True
        or value.get("submit_before_checkpoint_allowed") is not False
        or value.get("post_requires_current_control_fence_cas_recheck") is not True
        or value.get("scheduler_write_performed") is not False
        or value.get("cancellation_performed") is not False
        or value.get("preemption_performed") is not False
        or value.get("checkpoint_sha256") != phase_a.canonical_sha256(unsigned)
        or dict(value) != rebuilt
    ):
        raise RuntimeError("Phase-B deficit checkpoint seal mismatch")
    return copy.deepcopy(dict(value))


def _validate_templates(
    templates: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(templates) != set(STAGE_LOGICAL_QUOTAS):
        raise RuntimeError("Phase-B fresh task templates are incomplete")
    result: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        result[stage.stage_id] = validate_task(
            templates[stage.stage_id], expected_stage=stage
        )
    return result


def _candidate_group_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    task = item["legacy_task"]
    payload = task["payload_json"]
    return (
        item["stage_id"],
        payload["bundle_id"],
        payload["bundle_manifest_sha256"],
        payload["lane"]["wave"],
        tuple((field, repr(task[field])) for field in phase_a.RESOURCE_FIELDS),
    )


def _group_candidates(items: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    stage_order = {stage.stage_id: index for index, stage in enumerate(STAGES)}
    ordered = sorted(
        (copy.deepcopy(dict(item)) for item in items),
        key=lambda item: (stage_order[item["stage_id"]], int(item["seed"])),
    )
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in ordered:
        if current and (
            len(current) >= phase_b.MAX_CONCURRENT_CHILDREN
            or _candidate_group_key(current[-1]) != _candidate_group_key(item)
            or int(item["seed"]) != int(current[-1]["seed"]) + 1
        ):
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def _add_child_claim(
    records: dict[str, dict[str, Any]],
    seed_keys: dict[tuple[str, int], str],
    *,
    stage_id: str,
    child: Mapping[str, Any],
) -> dict[str, Any]:
    logical = str(child["logical_dedupe_key"])
    seed = int(child["seed"])
    seed_key = (stage_id, seed)
    if seed_key in seed_keys and seed_keys[seed_key] != logical:
        raise RuntimeError("Phase-B seed maps to multiple logical dedupe keys")
    seed_keys[seed_key] = logical
    record = records.setdefault(
        logical,
        {
            "stage_id": stage_id,
            "seed": seed,
            "task": copy.deepcopy(child["task"]),
            "child_task_sha256": child["child_task_sha256"],
            "active_claims": [],
            "success_receipts": [],
            "incomplete_claims": [],
        },
    )
    if (
        record["stage_id"] != stage_id
        or record["seed"] != seed
        or record["task"] != child["task"]
        or record["child_task_sha256"] != child["child_task_sha256"]
    ):
        raise RuntimeError("Phase-B logical dedupe maps to different child identity")
    return record


def reconcile_deficit(
    *,
    authority: Mapping[str, Any],
    current_control_fence: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    inventory_snapshot: Mapping[str, Any],
    fresh_task_templates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return an exact-500 dry reconciliation plan and durable outbox state."""

    authority, _current_fence = require_current_authority(
        authority, current_control_fence=current_control_fence
    )
    state = validate_checkpoint(checkpoint, authority=authority)
    inventory = validate_bounded_inventory_snapshot(
        inventory_snapshot, authority=authority
    )
    templates = _validate_templates(fresh_task_templates)
    observations = inventory["observations"]
    observed_by_dedupe = {
        item["parent_task"]["dedupe_key"]: item for item in observations
    }

    retained_pending: list[dict[str, Any]] = []
    for pending in state["pending_submission_intents"]:
        observed = observed_by_dedupe.get(pending["physical_parent_dedupe_key"])
        if observed is None:
            retained_pending.append(pending)
        elif observed["parent_task"] != pending["parent_task"]:
            raise RuntimeError("pending Phase-B intent reconciled to a foreign task")

    records: dict[str, dict[str, Any]] = {}
    seed_keys: dict[tuple[str, int], str] = {}
    physical_dedupes = set(observed_by_dedupe)
    bundle_identity_by_stage: dict[str, tuple[str, str]] = {}

    for observation in observations:
        task = observation["parent_task"]
        payload = task["payload_json"]
        stage_id = str(payload["stage_id"])
        identity = (str(payload["bundle_id"]), str(payload["bundle_manifest_sha256"]))
        previous_identity = bundle_identity_by_stage.setdefault(stage_id, identity)
        if previous_identity != identity:
            raise RuntimeError("Phase-B stage spans multiple immutable bundles")
        receipts = {
            int(item["ordinal"]): item for item in observation["child_receipts"]
        }
        live_parent = observation["scheduler_status"] in ACTIVE_SCHEDULER_STATES
        for child in payload["children"]:
            record = _add_child_claim(
                records, seed_keys, stage_id=stage_id, child=child
            )
            ordinal = int(child["ordinal"])
            receipt = receipts.get(ordinal)
            source = str(task["dedupe_key"])
            if receipt is None:
                target = "active_claims" if live_parent else "incomplete_claims"
                record[target].append(source)
            elif receipt["state"] == "completed":
                record["success_receipts"].append(receipt["receipt_sha256"])
            else:
                record["incomplete_claims"].append(source)

    # A sealed outbox reservation is active until the bounded watcher observes
    # its exact physical dedupe.  The next checkpoint MUST be durably CAS
    # committed before an external driver POSTs any new intent.  A POST before
    # that commit is forbidden and deliberately has no recovery claim here.
    for pending in retained_pending:
        task = pending["parent_task"]
        stage_id = str(task["payload_json"]["stage_id"])
        for child in task["payload_json"]["children"]:
            record = _add_child_claim(
                records, seed_keys, stage_id=stage_id, child=child
            )
            record["active_claims"].append(
                "pending:" + pending["physical_parent_dedupe_key"]
            )
        physical_dedupes.add(pending["physical_parent_dedupe_key"])

    active_by_stage = {stage.stage_id: 0 for stage in STAGES}
    recovery_records: list[dict[str, Any]] = []
    for logical, record in records.items():
        active_claims = record["active_claims"]
        successes = record["success_receipts"]
        if len(active_claims) > 1:
            raise RuntimeError(
                f"duplicate active Phase-B logical seed/dedupe: {logical}"
            )
        if len(successes) > 1:
            raise RuntimeError(
                f"duplicate completed Phase-B logical seed/dedupe: {logical}"
            )
        if active_claims and successes:
            raise RuntimeError(
                f"completed Phase-B child is simultaneously active: {logical}"
            )
        if active_claims:
            active_by_stage[record["stage_id"]] += 1
        elif not successes and record["incomplete_claims"]:
            recovery_records.append(record)

    for stage_id, count in active_by_stage.items():
        if count > STAGE_LOGICAL_QUOTAS[stage_id]:
            raise RuntimeError(f"Phase-B {stage_id} active quota is exceeded")

    max_seen_by_stage = {
        stage.stage_id: max(
            [
                seed
                for (candidate_stage, seed), _logical in seed_keys.items()
                if candidate_stage == stage.stage_id
            ]
            or [stage.seed_start - 1]
        )
        for stage in STAGES
    }
    next_seeds = {
        stage.stage_id: max(
            int(state["next_seed_by_stage"][stage.stage_id]),
            max_seen_by_stage[stage.stage_id] + 1,
        )
        for stage in STAGES
    }
    candidates: list[dict[str, Any]] = []
    recovery_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in recovery_records:
        recovery_by_stage[record["stage_id"]].append(record)

    for stage in STAGES:
        stage_id = stage.stage_id
        recoveries = sorted(
            recovery_by_stage[stage_id], key=lambda item: int(item["seed"])
        )
        deficit = STAGE_LOGICAL_QUOTAS[stage_id] - active_by_stage[stage_id]
        if len(recoveries) > deficit:
            raise RuntimeError(
                f"Phase-B {stage_id} incomplete children were replaced externally"
            )
        for record in recoveries:
            legacy = phase_a._legacy_child_task_from_phase_a(
                record["task"],
                expected_stage=stage,
            )
            candidates.append(
                {
                    "stage_id": stage_id,
                    "seed": int(record["seed"]),
                    "logical_dedupe_key": next(
                        logical for logical, value in records.items() if value is record
                    ),
                    "child_task_sha256": str(record["child_task_sha256"]),
                    "legacy_task": legacy,
                    "disposition": "recover-incomplete",
                    "source_parent_dedupe_keys": sorted(
                        set(record["incomplete_claims"])
                    ),
                }
            )
        fresh_count = deficit - len(recoveries)
        template = templates[stage_id]
        template_payload = template["payload_json"]
        observed_identity = bundle_identity_by_stage.get(stage_id)
        template_identity = (
            str(template_payload["bundle_id"]),
            str(template_payload["bundle_manifest_sha256"]),
        )
        if observed_identity is not None and observed_identity != template_identity:
            raise RuntimeError("fresh task template does not match observed bundle")
        for _ in range(fresh_count):
            seed = int(next_seeds[stage_id])
            if not stage.seed_start <= seed < stage.seed_window_end_exclusive:
                raise RuntimeError(f"Phase-B {stage_id} seed window is exhausted")
            legacy = _refill_task(template, stage_id=stage_id, seed=seed, wave="refill")
            transformed = phase_a._phase_a_child_task(legacy)
            logical = str(transformed["dedupe_key"])
            if logical in records or (stage_id, seed) in seed_keys:
                raise RuntimeError("fresh Phase-B child reuses a seed/dedupe identity")
            records[logical] = {
                "stage_id": stage_id,
                "seed": seed,
                "task": transformed,
                "child_task_sha256": phase_a.canonical_sha256(transformed),
                "active_claims": [],
                "success_receipts": [],
                "incomplete_claims": [],
            }
            seed_keys[(stage_id, seed)] = logical
            candidates.append(
                {
                    "stage_id": stage_id,
                    "seed": seed,
                    "logical_dedupe_key": logical,
                    "child_task_sha256": phase_a.canonical_sha256(transformed),
                    "legacy_task": legacy,
                    "disposition": "fresh-refill",
                    "source_parent_dedupe_keys": [],
                }
            )
            next_seeds[stage_id] = seed + 1

    plan_generation = int(state["generation"]) + 1
    new_intents: list[dict[str, Any]] = []
    for ordinal, group in enumerate(_group_candidates(candidates)):
        base_parent = phase_b.build_concurrent_batch_task(
            [item["legacy_task"] for item in group]
        )
        intent = _build_submission_intent(
            base_parent,
            group,
            authority=authority,
            plan_generation=plan_generation,
            intent_ordinal=ordinal,
        )
        if intent["physical_parent_dedupe_key"] in physical_dedupes:
            raise RuntimeError("Phase-B physical attempt dedupe already exists")
        physical_dedupes.add(intent["physical_parent_dedupe_key"])
        new_intents.append(intent)

    pending_after = [*retained_pending, *new_intents]
    next_checkpoint = build_checkpoint(
        authority=authority,
        next_seed_by_stage=next_seeds,
        generation=plan_generation,
        pending_submission_intents=pending_after,
        last_inventory_snapshot_sha256=inventory["inventory_snapshot_sha256"],
    )
    recovery_counts = {stage.stage_id: 0 for stage in STAGES}
    fresh_counts = {stage.stage_id: 0 for stage in STAGES}
    for item in candidates:
        target = (
            recovery_counts
            if item["disposition"] == "recover-incomplete"
            else fresh_counts
        )
        target[item["stage_id"]] += 1
    after_by_stage = {
        stage.stage_id: (
            active_by_stage[stage.stage_id]
            + recovery_counts[stage.stage_id]
            + fresh_counts[stage.stage_id]
        )
        for stage in STAGES
    }
    if (
        after_by_stage != STAGE_LOGICAL_QUOTAS
        or sum(after_by_stage.values()) != TOTAL_LOGICAL_TARGET
    ):
        raise RuntimeError("Phase-B reconciliation did not restore exact logical 500")

    unsigned = {
        "schema_version": PLAN_SCHEMA,
        "phase_b_protocol_version": phase_b.PROTOCOL_VERSION,
        "authority_lease_sha256": authority["authority_lease_sha256"],
        "control_fence_sha256": authority["control_fence_sha256"],
        "control_epoch": authority["control_epoch"],
        "checkpoint_sha256_before": state["checkpoint_sha256"],
        "inventory_snapshot_sha256": inventory["inventory_snapshot_sha256"],
        "plan_generation": plan_generation,
        "stage_logical_quotas": copy.deepcopy(STAGE_LOGICAL_QUOTAS),
        "logical_active_before_by_stage": active_by_stage,
        "recovered_incomplete_by_stage": recovery_counts,
        "fresh_refill_by_stage": fresh_counts,
        "logical_active_after_by_stage": after_by_stage,
        "logical_active_after": sum(after_by_stage.values()),
        "submission_intents": new_intents,
        "submission_intent_count": len(new_intents),
        "submission_logical_child_count": len(candidates),
        "pending_intent_count_after": len(pending_after),
        "next_checkpoint": next_checkpoint,
        "next_checkpoint_sha256": next_checkpoint["checkpoint_sha256"],
        "post_precondition_checkpoint_sha256": next_checkpoint["checkpoint_sha256"],
        "checkpoint_must_be_durable_before_post": True,
        "submit_before_checkpoint_allowed": False,
        "post_requires_current_control_fence_cas_recheck": True,
        "max_parent_lane_count": phase_b.MAX_CONCURRENT_CHILDREN,
        "deterministic_contiguous_regrouping": True,
        "duplicate_active_logical_children_allowed": False,
        "completed_seed_replay_allowed": False,
        "incomplete_seed_identity_preserved": True,
        "scheduler_write_performed": False,
        "http_request_performed": False,
        "cancellation_performed": False,
        "preemption_performed": False,
        "fea_submission_performed": False,
        "aedt_used": False,
    }
    return validate_reconciliation_plan(
        _seal(unsigned, "plan_sha256"), authority=authority
    )


def validate_reconciliation_plan(
    value: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, Any]:
    authority = validate_authority_lease(authority)
    unsigned = {key: item for key, item in value.items() if key != "plan_sha256"}
    intents_value = value.get("submission_intents")
    if not isinstance(intents_value, list):
        raise RuntimeError("Phase-B reconciliation intent inventory is missing")
    intents = [
        validate_submission_intent(item, authority=authority) for item in intents_value
    ]
    checkpoint = validate_checkpoint(
        value.get("next_checkpoint") or {}, authority=authority
    )
    stage_fields = (
        "logical_active_before_by_stage",
        "recovered_incomplete_by_stage",
        "fresh_refill_by_stage",
        "logical_active_after_by_stage",
    )
    stage_counts: dict[str, dict[str, int]] = {}
    for field in stage_fields:
        observed = value.get(field)
        if not isinstance(observed, Mapping) or set(observed) != set(
            STAGE_LOGICAL_QUOTAS
        ):
            raise RuntimeError(f"Phase-B reconciliation {field} is incomplete")
        stage_counts[field] = {
            stage_id: _integer(observed[stage_id], field)
            for stage_id in STAGE_LOGICAL_QUOTAS
        }
    plan_generation = _integer(
        value.get("plan_generation"), "plan generation", minimum=1
    )
    intent_ordinals = [int(item["intent_ordinal"]) for item in intents]
    physical_dedupes = [str(item["physical_parent_dedupe_key"]) for item in intents]
    intent_children = [
        child for intent in intents for child in intent["ordered_children"]
    ]
    logical_children = [str(child["logical_dedupe_key"]) for child in intent_children]
    stage_seed_children = [
        (str(child["stage_id"]), int(child["seed"])) for child in intent_children
    ]
    pending_current_generation = [
        item
        for item in checkpoint["pending_submission_intents"]
        if int(item["plan_generation"]) == plan_generation
    ]
    disposition_counts = {
        "recover-incomplete": {stage_id: 0 for stage_id in STAGE_LOGICAL_QUOTAS},
        "fresh-refill": {stage_id: 0 for stage_id in STAGE_LOGICAL_QUOTAS},
    }
    for child in intent_children:
        disposition_counts[str(child["disposition"])][str(child["stage_id"])] += 1
    arithmetic_valid = all(
        stage_counts["logical_active_before_by_stage"][stage_id]
        + stage_counts["recovered_incomplete_by_stage"][stage_id]
        + stage_counts["fresh_refill_by_stage"][stage_id]
        == stage_counts["logical_active_after_by_stage"][stage_id]
        == STAGE_LOGICAL_QUOTAS[stage_id]
        for stage_id in STAGE_LOGICAL_QUOTAS
    )
    required = {
        "schema_version",
        "phase_b_protocol_version",
        "authority_lease_sha256",
        "control_fence_sha256",
        "control_epoch",
        "checkpoint_sha256_before",
        "inventory_snapshot_sha256",
        "plan_generation",
        "stage_logical_quotas",
        "logical_active_before_by_stage",
        "recovered_incomplete_by_stage",
        "fresh_refill_by_stage",
        "logical_active_after_by_stage",
        "logical_active_after",
        "submission_intents",
        "submission_intent_count",
        "submission_logical_child_count",
        "pending_intent_count_after",
        "next_checkpoint",
        "next_checkpoint_sha256",
        "post_precondition_checkpoint_sha256",
        "checkpoint_must_be_durable_before_post",
        "submit_before_checkpoint_allowed",
        "post_requires_current_control_fence_cas_recheck",
        "max_parent_lane_count",
        "deterministic_contiguous_regrouping",
        "duplicate_active_logical_children_allowed",
        "completed_seed_replay_allowed",
        "incomplete_seed_identity_preserved",
        "scheduler_write_performed",
        "http_request_performed",
        "cancellation_performed",
        "preemption_performed",
        "fea_submission_performed",
        "aedt_used",
        "plan_sha256",
    }
    if (
        set(value) != required
        or value.get("schema_version") != PLAN_SCHEMA
        or value.get("phase_b_protocol_version") != phase_b.PROTOCOL_VERSION
        or value.get("authority_lease_sha256") != authority["authority_lease_sha256"]
        or value.get("control_fence_sha256") != authority["control_fence_sha256"]
        or value.get("control_epoch") != authority["control_epoch"]
        or not _is_sha256(value.get("checkpoint_sha256_before"))
        or not _is_sha256(value.get("inventory_snapshot_sha256"))
        or plan_generation != checkpoint["generation"]
        or value.get("stage_logical_quotas") != STAGE_LOGICAL_QUOTAS
        or stage_counts["logical_active_after_by_stage"] != STAGE_LOGICAL_QUOTAS
        or not arithmetic_valid
        or value.get("logical_active_after") != TOTAL_LOGICAL_TARGET
        or value.get("submission_intent_count") != len(intents)
        or value.get("submission_logical_child_count") != len(intent_children)
        or any(int(item["plan_generation"]) != plan_generation for item in intents)
        or intent_ordinals != list(range(len(intents)))
        or len(physical_dedupes) != len(set(physical_dedupes))
        or len(logical_children) != len(set(logical_children))
        or len(stage_seed_children) != len(set(stage_seed_children))
        or disposition_counts["recover-incomplete"]
        != stage_counts["recovered_incomplete_by_stage"]
        or disposition_counts["fresh-refill"] != stage_counts["fresh_refill_by_stage"]
        or any(
            int(item["plan_generation"]) > plan_generation
            for item in checkpoint["pending_submission_intents"]
        )
        or pending_current_generation != intents
        or value.get("pending_intent_count_after")
        != len(checkpoint["pending_submission_intents"])
        or value.get("next_checkpoint_sha256") != checkpoint["checkpoint_sha256"]
        or value.get("post_precondition_checkpoint_sha256")
        != checkpoint["checkpoint_sha256"]
        or checkpoint["last_inventory_snapshot_sha256"]
        != value.get("inventory_snapshot_sha256")
        or value.get("checkpoint_must_be_durable_before_post") is not True
        or value.get("submit_before_checkpoint_allowed") is not False
        or value.get("post_requires_current_control_fence_cas_recheck") is not True
        or value.get("max_parent_lane_count") != phase_b.MAX_CONCURRENT_CHILDREN
        or value.get("deterministic_contiguous_regrouping") is not True
        or value.get("duplicate_active_logical_children_allowed") is not False
        or value.get("completed_seed_replay_allowed") is not False
        or value.get("incomplete_seed_identity_preserved") is not True
        or any(
            value.get(field) is not False
            for field in (
                "scheduler_write_performed",
                "http_request_performed",
                "cancellation_performed",
                "preemption_performed",
                "fea_submission_performed",
                "aedt_used",
            )
        )
        or value.get("plan_sha256") != phase_a.canonical_sha256(unsigned)
    ):
        raise RuntimeError("Phase-B deficit reconciliation plan seal mismatch")
    return copy.deepcopy(dict(value))
