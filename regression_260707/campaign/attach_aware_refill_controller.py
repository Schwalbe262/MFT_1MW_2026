"""Attach-aware rolling refill controller for the reviewed MFT campaign.

``plan`` performs only scheduler GET requests (plus local generation-state
initialisation).  ``run`` is the sole scheduler mutation path and requires an
exact generation acknowledgement before it can submit an idempotent rolling
refill.  Run mode may also cancel only exact, state-owned pooled host tasks
after verifying their scheduler identity.

The pure policy/coordinator APIs remain usable without the live controller
shell; ``offline`` retains the original file-to-file planner interface.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from filelock import FileLock


HERE = Path(__file__).resolve().parent
REGRESSION_ROOT = HERE.parent
VERIFY_ROOT = REGRESSION_ROOT / "verify"
for _import_root in (HERE, REGRESSION_ROOT, VERIFY_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

import pinned_pilot
import scheduler_client

try:
    from .attach_refill_policy import (
        AttachRefillPolicy,
        NODE_LOCAL_POOLED_VALIDATION_ATTESTATION,
        POOLED,
        ProjectCandidate,
        RefillBundle,
        RevisionProvenance,
        STANDALONE,
        desired_aedt_sessions,
        logical_refill_deficit,
        make_refill_bundles,
        pin_candidate_params,
        reconcile_failed_bundle,
        task_submission_options,
    )
except ImportError:
    from attach_refill_policy import (  # type: ignore
        AttachRefillPolicy,
        NODE_LOCAL_POOLED_VALIDATION_ATTESTATION,
        POOLED,
        ProjectCandidate,
        RefillBundle,
        RevisionProvenance,
        STANDALONE,
        desired_aedt_sessions,
        logical_refill_deficit,
        make_refill_bundles,
        pin_candidate_params,
        reconcile_failed_bundle,
        task_submission_options,
    )


POLICY_SCHEMA = "mft-attach-aware-refill-policy-v1"
PLAN_SCHEMA = "mft-attach-aware-refill-plan-v1"
STATE_SCHEMA = "mft-restart-v3-controller-state-v2"
GENERATION_SCHEMA = "mft-restart-v3-generation-v1"
DEFAULT_STATE_PATH = HERE / "restart_v3_controller_state.json"
DEFAULT_POLICY_PATH = HERE / "attach_refill_policy_canary_n2.json"
DEFAULT_CANDIDATE_SEED = 260_710
DEFAULT_CPUS = 4
DEFAULT_MEMORY_MB = 65_536
DEFAULT_TIMEOUT_SECONDS = 14_400
DEFAULT_LOOP_SECONDS = 60
PROFILE_PATH = VERIFY_ROOT / "profiles" / "standard.json"
PRODUCTION_SOLVER_REVISION = "bffbb15fe2cdec74a72f47e7eb9bacbf0f4e95f7"
PRODUCTION_LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
PRODUCTION_PHYSICS_DATA_REVISION = "mft1mw-1k101-native-lamination-kf0p85-v3"
PRODUCTION_CORE_LAMINATION_FACTOR = 0.85
PRODUCTION_PROJECTS_PER_AEDT = 2
PRODUCTION_MAX_AEDT_SESSIONS = 250
# Ten legacy p02/p08 candidates plus the 25 fresh entries in the submitted
# restart-v3 pilot manifest occupy the first 35 valid Sobol candidates.
PRODUCTION_CANDIDATE_VALID_OFFSET = 35
NODE_CANARY_HOST_ENTRYPOINT = "aedt_node_canary_host"
NODE_CANARY_CLIENT_ENTRYPOINT = "aedt_node_canary_client"
NODE_CANARY_HOST_PROJECT = "_aedt_pool_hosts"
NODE_CANARY_SCHEDULER_REPOSITORY = (
    "https://github.com/Schwalbe262/slurm_scheduler.git"
)
NODE_CANARY_SCHEDULER_REVISION = "e4718e4b6f229175f8c3d85dbf2cf8c34c7ee93e"
NODE_CANARY_DISCOVERY_PREFIX = "NODE_CANARY_DISCOVERY "
# 30 min: the host is pinned to a live (usually full) allocation, so it must
# first wait for a task slot to free and then cold-start AEDT on a busy node.
# 10 min covered only ~20% of hosts in production on 2026-07-14.
NODE_CANARY_DISCOVERY_TIMEOUT_SECONDS = 30 * 60
NODE_CANARY_HOST_TIMEOUT_SECONDS = 14_400
NODE_CANARY_HOST_TASK_TIMEOUT_SECONDS = 6 * 3600
NODE_CANARY_HOST_CPUS = 1
NODE_CANARY_HOST_MEMORY_MB = 4_096
NODE_CANARY_HOST_PRIORITY = 100_000
NODE_CANARY_STDOUT_MAX_BYTES = 1_048_576
TASK_TERMINAL_STATES = {
    "completed",
    "failed",
    "cancelled",
    "timeout",
}
TASK_STATUS_ALIASES = {"canceled": "cancelled", "timed_out": "timeout"}
POOLED_BUNDLE_HOST_PREFIX = "mft-aedt-pooled-"
HOST_CANCEL_STATUSES = {
    "requested",
    "already_terminal",
    "failed",
    "identity_mismatch",
}


class ExactTaskSubmissionRejected(RuntimeError):
    """The scheduler definitively rejected an exact-placement host task."""


def _normalized_task_status(value: object) -> str:
    status = str(value or "").strip().lower()
    return TASK_STATUS_ALIASES.get(status, status)


def load_policy(payload: Mapping[str, object]) -> AttachRefillPolicy:
    if payload.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"policy schema must be {POLICY_SCHEMA}")
    provenance_payload = payload.get("provenance")
    if not isinstance(provenance_payload, Mapping):
        raise ValueError("policy provenance is missing")
    provenance = RevisionProvenance(**dict(provenance_payload))
    fields = {
        key: value
        for key, value in payload.items()
        if key not in {"schema", "provenance"}
    }
    return AttachRefillPolicy(provenance=provenance, **fields)


def pool_gate(
    policy: AttachRefillPolicy,
    status: Mapping[str, object] | None = None,
) -> dict:
    """Authenticate the reviewed node-local pooled-canary policy envelope."""

    if policy.primary_backend == STANDALONE:
        return {
            "eligible": False,
            "reason": "standalone_selected",
            "observed": None,
        }
    if policy.pooled_fraction <= 0:
        return {
            "eligible": False,
            "reason": "pooled_fraction_zero",
            "observed": None,
        }
    # ``status`` is retained for compatibility with the pure/offline API.  The
    # central scheduler pool is intentionally unreachable from compute nodes,
    # so live eligibility is derived only from the reviewed policy attestation.
    observed = {
        "node_local_pooled_enabled": policy.node_local_pooled_enabled,
        "node_local_pooled_validation_attestation": (
            policy.node_local_pooled_validation_attestation
        ),
        "projects_per_aedt": policy.projects_per_aedt,
        "validated_projects_per_aedt": policy.validated_projects_per_aedt,
    }
    checks = {
        "node_local_pooled_enabled": policy.node_local_pooled_enabled is True,
        "validation_attestation": (
            policy.node_local_pooled_validation_attestation
            == NODE_LOCAL_POOLED_VALIDATION_ATTESTATION
        ),
        "validated_n2": (
            policy.projects_per_aedt == PRODUCTION_PROJECTS_PER_AEDT
            and policy.validated_projects_per_aedt
            == PRODUCTION_PROJECTS_PER_AEDT
        ),
    }
    return {
        "eligible": all(checks.values()),
        "reason": "ready" if all(checks.values()) else "pool_gate_failed",
        "checks": checks,
        "observed": observed,
        "latest_validation": policy.node_local_pooled_validation_attestation,
        "pinned_validation_provenance": {
            "attach_validation_revision": (
                policy.provenance.attach_validation_revision
            ),
            "attach_validation_scheduler_revision": (
                policy.provenance.attach_validation_scheduler_revision
            ),
            "attach_timeout_validation_scheduler_revision": (
                policy.provenance.attach_timeout_validation_scheduler_revision
            ),
        },
    }


def _candidate_uses_pooled_backend(
    policy: AttachRefillPolicy, candidate: ProjectCandidate
) -> bool:
    """Deterministically admit the configured fraction across rolling cycles."""

    if policy.pooled_fraction <= 0:
        return False
    if policy.pooled_fraction >= 1:
        return True
    identity = (
        f"{policy.digest}:{candidate.name}:{candidate.params_sha256}"
    ).encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
    return bucket / float(1 << 64) < policy.pooled_fraction


@dataclass(frozen=True)
class AttachAwareRefillCoordinator:
    policy: AttachRefillPolicy

    def plan_cycle(
        self,
        *,
        active_project_tasks: int,
        candidates: Sequence[ProjectCandidate],
        pool_status: Mapping[str, object] | None,
    ) -> dict[str, object]:
        deficit = logical_refill_deficit(
            target=self.policy.project_concurrency_target,
            active_project_tasks=active_project_tasks,
        )
        if len(candidates) < deficit:
            raise ValueError(
                f"candidate manifest has {len(candidates)} entries for deficit {deficit}"
            )
        selected_candidates = tuple(candidates[:deficit])
        gate = pool_gate(self.policy, pool_status)
        pooled_candidates: tuple[ProjectCandidate, ...] = ()
        if self.policy.primary_backend == POOLED and gate["eligible"]:
            pooled_candidates = tuple(
                candidate
                for candidate in selected_candidates
                if _candidate_uses_pooled_backend(self.policy, candidate)
            )
        pooled_count = len(pooled_candidates)
        if pooled_count == deficit and deficit:
            selected_backend = POOLED
            backend_reason = "pool_gate_passed"
        elif pooled_count:
            selected_backend = "mixed"
            backend_reason = "pool_gate_passed_fractional_admission"
        elif self.policy.primary_backend == POOLED and self.policy.pooled_fraction > 0:
            # No cancellation and no waiting gap: this cycle alone reuses the
            # proven standalone path.  A later cycle may re-enter pooled mode
            # after the independently managed pool is healthy again.
            selected_backend = STANDALONE
            backend_reason = (
                "pool_unavailable_standalone_fallback"
                if not gate["eligible"]
                else "pooled_fraction_rounds_to_zero"
            )
        else:
            selected_backend = STANDALONE
            backend_reason = (
                "pooled_fraction_zero"
                if self.policy.primary_backend == POOLED
                else "standalone_selected"
            )
        pooled_names = {candidate.name for candidate in pooled_candidates}
        standalone_candidates = tuple(
            candidate
            for candidate in selected_candidates
            if candidate.name not in pooled_names
        )
        bundles = (
            *make_refill_bundles(
                pooled_candidates, self.policy, backend=POOLED
            ),
            *make_refill_bundles(
                standalone_candidates, self.policy, backend=STANDALONE
            ),
        )
        pooled_projects = sum(
            bundle.expected_rows for bundle in bundles if bundle.backend == POOLED
        )
        plan = {
            "schema": PLAN_SCHEMA,
            "policy_digest": self.policy.digest,
            "provenance": self.policy.provenance.as_dict(),
            "provenance_digest": self.policy.provenance.digest,
            "project_concurrency_target": self.policy.project_concurrency_target,
            "active_project_tasks": active_project_tasks,
            "logical_project_deficit": deficit,
            "selected_backend": selected_backend,
            "backend_reason": backend_reason,
            "scheduling_profile": self.policy.standalone_profile,
            "pooled_fraction": self.policy.pooled_fraction,
            "pooled_project_count": pooled_projects,
            "standalone_project_count": deficit - pooled_projects,
            "projects_per_aedt": self.policy.projects_per_aedt,
            "max_aedt_sessions": self.policy.max_aedt_sessions,
            "desired_aedt_sessions": desired_aedt_sessions(
                pooled_projects, self.policy.projects_per_aedt
            ),
            "pool_gate": gate,
            "bundles": [bundle.as_dict() for bundle in bundles],
            "bundle_expected_rows": [bundle.expected_rows for bundle in bundles],
            "expected_rows": sum(bundle.expected_rows for bundle in bundles),
            "scheduler_task_count": sum(len(bundle.candidates) for bundle in bundles),
            "cancel_task_ids": [],
            "mass_cancel_authorized": False,
        }
        plan["task_submission_options"] = [
            {
                "candidate_name": candidate.name,
                **task_submission_options(
                    bundle, self.policy, candidate_index=index
                ),
            }
            for bundle in bundles
            for index, candidate in enumerate(bundle.candidates)
        ]
        return plan

    def plan_failed_bundle_fallback(
        self,
        bundle: RefillBundle,
        *,
        task_ids: Sequence[int],
        task_statuses: Mapping[int, str],
        accepted_row_task_ids: Sequence[int],
    ) -> dict[str, object]:
        decision = reconcile_failed_bundle(
            bundle,
            task_ids=task_ids,
            task_statuses=task_statuses,
            accepted_row_task_ids=accepted_row_task_ids,
            policy=self.policy,
        )
        if decision["action"] != "submit_standalone_fallback":
            return decision
        fallback_candidates = tuple(
            ProjectCandidate(
                # A terminal scheduler identity cannot be reused: exact
                # reconciliation would otherwise return the failed pooled
                # task instead of creating a standalone replacement.  Keep
                # the same parameter digest but give the retry its own stable
                # task/dedupe identity.
                name=(
                    f"{bundle.candidates[index].name}-sa-retry-"
                    f"{bundle.bundle_id[-8:]}-{index}"
                ),
                params_sha256=bundle.candidates[index].params_sha256,
            )
            for index in decision["missing_candidate_indices"]
        )
        fallback_bundles = make_refill_bundles(
            fallback_candidates,
            self.policy,
            backend=STANDALONE,
            fallback_of=bundle.bundle_id,
        )
        return {
            **decision,
            "fallback_bundles": [item.as_dict() for item in fallback_bundles],
            "fallback_submission_options": [
                {
                    "candidate_name": candidate.name,
                    **task_submission_options(
                        item, self.policy, candidate_index=index
                    ),
                }
                for item in fallback_bundles
                for index, candidate in enumerate(item.candidates)
            ],
        }


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_policy(path: str | Path) -> AttachRefillPolicy:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("policy JSON must be an object")
    policy = load_policy(payload)
    production_checks = {
        "solver_revision": (
            policy.provenance.solver_revision == PRODUCTION_SOLVER_REVISION
        ),
        "library_revision": (
            policy.provenance.library_revision == PRODUCTION_LIBRARY_REVISION
        ),
        "physics_data_revision": (
            policy.provenance.physics_data_revision
            == PRODUCTION_PHYSICS_DATA_REVISION
        ),
        "data_contract_revision": (
            policy.provenance.data_contract_revision
            == PRODUCTION_PHYSICS_DATA_REVISION
        ),
        "core_lamination_factor": (
            policy.provenance.core_lamination_factor
            == PRODUCTION_CORE_LAMINATION_FACTOR
        ),
        "project_concurrency_target": policy.project_concurrency_target == 500,
        "primary_backend": policy.primary_backend == POOLED,
        "node_local_pooled_validation_attestation": (
            policy.node_local_pooled_validation_attestation
            == NODE_LOCAL_POOLED_VALIDATION_ATTESTATION
        ),
        "projects_per_aedt": (
            policy.projects_per_aedt == PRODUCTION_PROJECTS_PER_AEDT
            and policy.validated_projects_per_aedt
            == PRODUCTION_PROJECTS_PER_AEDT
        ),
        "max_aedt_sessions": (
            policy.max_aedt_sessions == PRODUCTION_MAX_AEDT_SESSIONS
        ),
    }
    if not all(production_checks.values()):
        raise ValueError(
            f"live controller production pins do not match: {production_checks}"
        )
    return policy


def _load_profile(timeout_seconds: int) -> dict[str, object]:
    payload = _load_json(PROFILE_PATH)
    if not isinstance(payload, Mapping):
        raise ValueError("standard scheduler profile must be an object")
    profile = dict(payload)
    overrides = profile.get("param_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("standard scheduler profile overrides must be an object")
    pinned_overrides = {
        "physics_data_revision": PRODUCTION_PHYSICS_DATA_REVISION,
        "core_lamination_factor": PRODUCTION_CORE_LAMINATION_FACTOR,
    }
    conflicts = {
        key: overrides[key]
        for key, expected in pinned_overrides.items()
        if key in overrides and overrides[key] != expected
    }
    if conflicts:
        raise ValueError(f"standard scheduler profile overrides physics pins: {conflicts}")
    if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    profile["timeout_seconds"] = int(timeout_seconds)
    return profile


def _normalize_scheduler_url(value: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("scheduler URL must be an HTTP(S) origin without credentials")
    return normalized


def _configure_scheduler(scheduler_url: str) -> None:
    scheduler_client.SCHEDULER = scheduler_url
    pinned_pilot.SCHEDULER = scheduler_url


def _generation_contract(
    policy: AttachRefillPolicy,
    *,
    candidate_seed: int,
    profile: Mapping[str, object],
    cpus: int,
    memory_mb: int,
) -> dict[str, object]:
    if (
        isinstance(candidate_seed, bool)
        or not isinstance(candidate_seed, int)
        or candidate_seed < 0
    ):
        raise ValueError("candidate_seed must be a non-negative integer")
    for name, value in (("cpus", cpus), ("memory_mb", memory_mb)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    generation_policy = policy.as_dict()
    # Rollout controls may be flipped after plan review without abandoning the
    # existing candidate ledger.  The attestation and every capacity/physics
    # pin remain generation identity; only enablement and fraction are neutral.
    generation_policy["pooled_fraction"] = 0.0
    generation_policy["node_local_pooled_enabled"] = False
    identity = {
        "schema": GENERATION_SCHEMA,
        "policy_digest": _canonical_sha256(generation_policy),
        "provenance_digest": policy.provenance.digest,
        "node_local_pooled_validation_attestation": (
            policy.node_local_pooled_validation_attestation
        ),
        "node_canary_scheduler_revision": NODE_CANARY_SCHEDULER_REVISION,
        "solver_revision": policy.provenance.solver_revision,
        "library_revision": policy.provenance.library_revision,
        "physics_data_revision": policy.provenance.physics_data_revision,
        "core_lamination_factor": policy.provenance.core_lamination_factor,
        "candidate_seed": candidate_seed,
        "candidate_valid_offset": PRODUCTION_CANDIDATE_VALID_OFFSET,
        "candidate_start_cursor": pinned_pilot.cursor_after_valid_candidates(
            PRODUCTION_CANDIDATE_VALID_OFFSET, seed=candidate_seed
        ),
        "project_concurrency_target": policy.project_concurrency_target,
        "profile_sha256": _canonical_sha256(profile),
        "cpus": cpus,
        "memory_mb": memory_mb,
    }
    digest = _canonical_sha256(identity)
    return {
        "id": f"restart-v3-{digest[:20]}",
        "digest": digest,
        "identity": identity,
    }


def _new_state(
    generation: Mapping[str, object], scheduler_url: str
) -> dict[str, object]:
    now = _now()
    return {
        "schema": STATE_SCHEMA,
        "state_revision": 0,
        "generation": dict(generation),
        "scheduler_url": scheduler_url,
        "candidate_cursor": int(
            generation["identity"]["candidate_start_cursor"]  # type: ignore[index]
        ),
        "next_serial": 1,
        "submissions": [],
        "reservations": [],
        "pooled_bundles": [],
        "last_host_account": "",
        "created_at": now,
        "updated_at": now,
    }


def _validate_state_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    lowered = resolved.name.lower()
    if "adopted_refill_" in lowered or "continuous_refill_b171c7c" in lowered:
        raise ValueError("restart-v3 controller cannot reuse a prior-generation state")
    return resolved


def _validate_state(
    state: object,
    generation: Mapping[str, object],
    scheduler_url: str,
) -> dict[str, object]:
    if not isinstance(state, dict):
        raise RuntimeError("controller state must be an object")
    identity_checks = {
        "schema": state.get("schema") == STATE_SCHEMA,
        "generation": state.get("generation") == dict(generation),
        "scheduler_url": state.get("scheduler_url") == scheduler_url,
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"restart-v3 state identity drifted: {identity_checks}")
    for field, minimum in (
        ("state_revision", 0),
        ("candidate_cursor", 0),
        ("next_serial", 1),
    ):
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise RuntimeError(f"controller state {field} is invalid")
    submissions = state.get("submissions")
    reservations = state.get("reservations")
    pooled_bundles = state.get("pooled_bundles")
    if not isinstance(submissions, list):
        raise RuntimeError("controller state submissions must be a list")
    if not isinstance(reservations, list):
        raise RuntimeError("controller state reservations must be a list")
    if not isinstance(pooled_bundles, list):
        raise RuntimeError("controller state pooled_bundles must be a list")
    if not isinstance(state.get("last_host_account"), str):
        raise RuntimeError("controller state last_host_account is invalid")
    names: set[str] = set()
    dedupe_keys: set[str] = set()
    task_ids: set[int] = set()
    records_by_serial: dict[int, dict[str, object]] = {}
    for item in submissions:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or isinstance(item.get("task_id"), bool)
            or not isinstance(item.get("task_id"), int)
            or int(item["task_id"]) <= 0
            or not isinstance(item.get("dedupe_key"), str)
            or not isinstance(item.get("params_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(item["params_sha256"]))
            or item.get("backend") not in {STANDALONE, POOLED}
            or isinstance(item.get("serial"), bool)
            or not isinstance(item.get("serial"), int)
            or int(item["serial"]) <= 0
            or isinstance(item.get("candidate_cursor_before"), bool)
            or not isinstance(item.get("candidate_cursor_before"), int)
            or isinstance(item.get("candidate_cursor_after"), bool)
            or not isinstance(item.get("candidate_cursor_after"), int)
            or int(item["candidate_cursor_after"])
            <= int(item["candidate_cursor_before"])
        ):
            raise RuntimeError("controller state contains an invalid submission")
        serial = int(item["serial"])
        if serial in records_by_serial:
            raise RuntimeError("controller state contains a duplicate serial")
        if (
            item["name"] in names
            or item["dedupe_key"] in dedupe_keys
            or int(item["task_id"]) in task_ids
        ):
            raise RuntimeError("controller state contains a duplicate submission")
        names.add(str(item["name"]))
        dedupe_keys.add(str(item["dedupe_key"]))
        task_ids.add(int(item["task_id"]))
        records_by_serial[serial] = item
    for item in reservations:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("dedupe_key"), str)
            or not isinstance(item.get("params_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(item["params_sha256"]))
            or item.get("backend") not in {STANDALONE, POOLED}
            or isinstance(item.get("serial"), bool)
            or not isinstance(item.get("serial"), int)
            or int(item["serial"]) <= 0
            or isinstance(item.get("candidate_cursor_before"), bool)
            or not isinstance(item.get("candidate_cursor_before"), int)
            or isinstance(item.get("candidate_cursor_after"), bool)
            or not isinstance(item.get("candidate_cursor_after"), int)
            or int(item["candidate_cursor_after"])
            <= int(item["candidate_cursor_before"])
            or not isinstance(item.get("params"), dict)
            or not isinstance(item.get("submission_env"), dict)
        ):
            raise RuntimeError("controller state contains an invalid reservation")
        serial = int(item["serial"])
        if serial in records_by_serial:
            raise RuntimeError("controller state contains a duplicate serial")
        records_by_serial[serial] = item
    committed_cursor = int(
        generation["identity"]["candidate_start_cursor"]  # type: ignore[index]
    )
    for expected_serial in range(1, len(records_by_serial) + 1):
        item = records_by_serial.get(expected_serial)
        if item is None or item.get("candidate_cursor_before") != committed_cursor:
            raise RuntimeError("controller state candidate reservation chain drifted")
        committed_cursor = int(item["candidate_cursor_after"])
    bundle_ids: set[str] = set()
    valid_phases = {
        "host_submit",
        "discovery_wait",
        "clients_submit",
        "clients_partial_tracked",
        "clients_tracked",
        "client_fallback_submit",
        "fallback_submit",
        "complete",
    }
    for bundle in pooled_bundles:
        if (
            not isinstance(bundle, dict)
            or not isinstance(bundle.get("bundle_id"), str)
            or not bundle["bundle_id"]
            or bundle.get("phase") not in valid_phases
            or not isinstance(bundle.get("action_serials"), list)
            or not bundle["action_serials"]
            or any(
                isinstance(serial, bool)
                or not isinstance(serial, int)
                or serial <= 0
                or serial not in records_by_serial
                for serial in bundle["action_serials"]
            )
            or len(set(bundle["action_serials"]))
            != len(bundle["action_serials"])
            or (
                bundle.get("host_terminal_status") is not None
                and bundle.get("host_terminal_status")
                not in TASK_TERMINAL_STATES
            )
            or (
                (bundle.get("host_terminal_status") is None)
                != (bundle.get("host_terminal_at") is None)
            )
            or (
                bundle.get("host_terminal_at") is not None
                and (
                    not isinstance(bundle.get("host_terminal_at"), str)
                    or not str(bundle.get("host_terminal_at")).strip()
                )
            )
            or (
                bundle.get("host_cancel_status") is not None
                and bundle.get("host_cancel_status") not in HOST_CANCEL_STATUSES
            )
            or (
                bundle.get("host_cancel_error") is not None
                and (
                    not isinstance(bundle.get("host_cancel_error"), str)
                    or not str(bundle.get("host_cancel_error")).strip()
                )
            )
            or (
                bundle.get("host_cancel_at") is not None
                and (
                    not isinstance(bundle.get("host_cancel_at"), str)
                    or not str(bundle.get("host_cancel_at")).strip()
                )
            )
            or (
                bundle.get("host_cancel_task_id") is not None
                and (
                    isinstance(bundle.get("host_cancel_task_id"), bool)
                    or not isinstance(bundle.get("host_cancel_task_id"), int)
                    or int(bundle["host_cancel_task_id"]) <= 0
                )
            )
        ):
            raise RuntimeError("controller state contains an invalid pooled bundle")
        bundle_id = str(bundle["bundle_id"])
        if bundle_id in bundle_ids:
            raise RuntimeError("controller state contains a duplicate pooled bundle")
        bundle_ids.add(bundle_id)
    ledger_checks = {
        "state_revision": state["state_revision"] == len(submissions),
        "next_serial": state["next_serial"] == len(records_by_serial) + 1,
        "candidate_cursor": state["candidate_cursor"] == committed_cursor,
    }
    if not all(ledger_checks.values()):
        raise RuntimeError(f"controller state ledger drifted: {ledger_checks}")
    return state


def _atomic_save_state(path: Path, state: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A verified os.replace above is the commit point. Some mounted
            # filesystems retain a stale directory entry briefly afterward.
            pass


def _load_or_initialize_state(
    path: Path,
    generation: Mapping[str, object],
    scheduler_url: str,
    *,
    create: bool,
) -> tuple[dict[str, object], bool]:
    path = _validate_state_path(path)
    if path.exists():
        state = _validate_state(_load_json(path), generation, scheduler_url)
        return state, False
    if not create:
        raise RuntimeError(
            f"controller state does not exist; run read-only plan first: {path}"
        )
    state = _new_state(generation, scheduler_url)
    _validate_state(state, generation, scheduler_url)
    _atomic_save_state(path, state)
    return state, True


def _response_json(response: Any, source: str) -> object:
    response.raise_for_status()
    payload = response.json()
    if payload is None:
        raise RuntimeError(f"scheduler returned an empty {source} response")
    return payload


def _task_rows(payload: object, source: str) -> list[dict[str, object]]:
    rows = payload if isinstance(payload, list) else (
        payload.get("tasks") if isinstance(payload, Mapping) else None
    )
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"scheduler returned an invalid {source} task inventory")
    return rows


def _response_text(response: Any, source: str) -> str:
    response.raise_for_status()
    text = response.text
    if not isinstance(text, str):
        raise RuntimeError(f"scheduler returned an invalid {source} response")
    return text


def _timestamp(value: object) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError(f"invalid persisted timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"persisted timestamp lacks timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _deadline(seconds: int) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ).isoformat().replace("+00:00", "Z")


def _loopback_scheduler_url(value: object) -> str:
    normalized = _normalize_scheduler_url(str(value or ""))
    hostname = (urlsplit(normalized).hostname or "").strip().lower()
    if hostname == "localhost":
        return normalized
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError("node canary scheduler URL must use a loopback host")
    return normalized


def parse_node_canary_discovery(
    stdout: str, *, expected_projects: int
) -> dict[str, object] | None:
    """Return the newest authenticated host discovery record, if present."""

    if not isinstance(stdout, str):
        raise TypeError("node canary stdout must be text")
    for line in reversed(stdout.splitlines()):
        if not line.startswith(NODE_CANARY_DISCOVERY_PREFIX):
            continue
        try:
            payload = json.loads(line[len(NODE_CANARY_DISCOVERY_PREFIX) :])
        except (TypeError, ValueError) as exc:
            raise ValueError("node canary discovery line is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("node canary discovery payload must be an object")
        checks = {
            "schema_version": payload.get("schema_version") == 1,
            "mode": (
                payload.get("mode")
                == "scheduler_managed_node_local_canary"
            ),
            "expected_projects": payload.get("expected_projects")
            == expected_projects,
            "node": bool(str(payload.get("node") or "").strip()),
            "rollback_file": str(payload.get("rollback_file") or "").startswith(
                "/tmp/"
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"node canary discovery contract failed: {checks}")
        result = dict(payload)
        result["scheduler_url"] = _loopback_scheduler_url(
            payload.get("scheduler_url")
        )
        return result
    return None


def select_host_allocations(
    allocations: Sequence[Mapping[str, object]],
    bundle_project_counts: Sequence[int],
    *,
    client_cpus: int,
    client_memory_mb: int,
    reserved_by_allocation: Mapping[int, Mapping[str, int]] | None = None,
    last_account: str = "",
) -> tuple[list[dict[str, object] | None], str]:
    """Select exact active allocations with account round-robin fairness."""

    for name, value in (
        ("client_cpus", client_cpus),
        ("client_memory_mb", client_memory_mb),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    counts = list(bundle_project_counts)
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= PRODUCTION_PROJECTS_PER_AEDT
        for count in counts
    ):
        raise ValueError("bundle project counts exceed the validated N=2 bound")
    persisted = dict(reserved_by_allocation or {})
    eligible: list[dict[str, object]] = []
    for raw in allocations:
        row = dict(raw)
        allocation_id = row.get("id")
        free_cpus = row.get("free_cpus")
        free_memory_mb = row.get("free_memory_mb")
        if (
            row.get("state") != "active"
            or str(row.get("resource_pool") or "cpu") != "cpu"
            or isinstance(allocation_id, bool)
            or not isinstance(allocation_id, int)
            or allocation_id <= 0
            or not str(row.get("account_name") or "").strip()
            or not str(row.get("node_name") or "").strip()
            or not str(row.get("slurm_job_id") or "").strip()
            or isinstance(free_cpus, bool)
            or not isinstance(free_cpus, int)
            or isinstance(free_memory_mb, bool)
            or not isinstance(free_memory_mb, int)
        ):
            continue
        reservation = dict(persisted.get(allocation_id) or {})
        row["_remaining_cpus"] = max(
            0, free_cpus - max(0, int(reservation.get("cpus") or 0))
        )
        row["_remaining_memory_mb"] = max(
            0,
            free_memory_mb
            - max(0, int(reservation.get("memory_mb") or 0)),
        )
        row["_reserved_hosts"] = max(
            0, int(reservation.get("hosts") or 0)
        )
        eligible.append(row)
    accounts = sorted(
        {str(row["account_name"]).strip() for row in eligible}
    )
    if not accounts:
        return [None for _ in counts], str(last_account or "")
    if last_account in accounts:
        next_account_index = (accounts.index(last_account) + 1) % len(accounts)
    else:
        next_account_index = 0
    selected: list[dict[str, object] | None] = []
    newest_account = str(last_account or "")
    for project_count in counts:
        required_cpus = NODE_CANARY_HOST_CPUS + project_count * client_cpus
        required_memory_mb = (
            NODE_CANARY_HOST_MEMORY_MB + project_count * client_memory_mb
        )
        choice: dict[str, object] | None = None
        choice_index = next_account_index
        for offset in range(len(accounts)):
            account_index = (next_account_index + offset) % len(accounts)
            account = accounts[account_index]
            candidates = sorted(
                (
                    row
                    for row in eligible
                    if str(row["account_name"]).strip() == account
                    and int(row["_remaining_cpus"]) >= required_cpus
                    and int(row["_remaining_memory_mb"])
                    >= required_memory_mb
                ),
                key=lambda row: (
                    int(row["_reserved_hosts"]),
                    int(row["id"]),
                ),
            )
            if candidates:
                choice = candidates[0]
                choice_index = account_index
                break
        if choice is None:
            selected.append(None)
            continue
        choice["_remaining_cpus"] = int(choice["_remaining_cpus"]) - required_cpus
        choice["_remaining_memory_mb"] = (
            int(choice["_remaining_memory_mb"]) - required_memory_mb
        )
        choice["_reserved_hosts"] = int(choice["_reserved_hosts"]) + 1
        newest_account = accounts[choice_index]
        next_account_index = (choice_index + 1) % len(accounts)
        selected.append(
            {
                key: value
                for key, value in choice.items()
                if not str(key).startswith("_")
            }
        )
    return selected, newest_account


def _coordination_paths(bundle_id: str) -> dict[str, str]:
    leaf = re.sub(r"[^A-Za-z0-9_.-]+", "-", bundle_id).strip("-.")
    if not leaf:
        raise ValueError("pooled bundle ID cannot form coordination paths")
    return {
        "discovery": f"/tmp/{leaf}.discovery.json",
        "evidence": f"/tmp/{leaf}.evidence.json",
        "rollback": f"/tmp/{leaf}.rollback",
    }


def build_node_canary_host_command(
    bundle_id: str, *, expected_projects: int
) -> str:
    paths = _coordination_paths(bundle_id)
    clone_leaf = f"{bundle_id}-host"
    root = f'"$PWD/{clone_leaf}"'
    return (
        "set -euo pipefail; "
        # Raw-command task: unlike MFT project entrypoints it gets no project
        # setup, so the AEDT environment must be loaded here or Desktop()
        # dies with "AEDT is not installed" (observed fleet-wide 2026-07-13).
        "source /etc/profile.d/lmod.sh 2>/dev/null || true; "
        "module load ansys-electronics/v252 2>/dev/null || "
        "export ANSYSEM_ROOT252=/opt/ohpc/pub/Electronics/v252/Linux64; "
        "export FLEXLM_TIMEOUT=3000000; "
        f"root={root}; "
        'test ! -e "$root"; '
        f"git clone --no-checkout {shlex.quote(NODE_CANARY_SCHEDULER_REPOSITORY)} "
        '"$root"; '
        f"git -C \"$root\" checkout --detach {NODE_CANARY_SCHEDULER_REVISION}; "
        f'test "$(git -C "$root" rev-parse HEAD)" = "{NODE_CANARY_SCHEDULER_REVISION}"; '
        'cd "$root"; '
        "python scripts/aedt_pool_node_canary_host.py "
        f"--discovery-file {shlex.quote(paths['discovery'])} "
        f"--evidence-file {shlex.quote(paths['evidence'])} "
        f"--rollback-file {shlex.quote(paths['rollback'])} "
        f"--expected-projects {expected_projects} "
        f"--timeout-seconds {NODE_CANARY_HOST_TIMEOUT_SECONDS}"
    )


def _node_canary_host_payload(
    bundle: Mapping[str, object], allocation: Mapping[str, object]
) -> dict[str, object]:
    bundle_id = str(bundle["bundle_id"])
    expected_projects = len(bundle["action_serials"])  # type: ignore[arg-type]
    paths = _coordination_paths(bundle_id)
    payload_json = {
        "aedt_canary_bundle_id": bundle_id,
        "aedt_canary_expected_projects": expected_projects,
        "aedt_canary_discovery_file": paths["discovery"],
        "aedt_canary_evidence_file": paths["evidence"],
        "aedt_canary_rollback_file": paths["rollback"],
        "aedt_canary_scheduler_revision": NODE_CANARY_SCHEDULER_REVISION,
    }
    return {
        "name": f"{bundle_id}-host",
        "project": NODE_CANARY_HOST_PROJECT,
        "remote_cwd": scheduler_client.GPFS_RUNS_REMOTE_CWD,
        "command": build_node_canary_host_command(
            bundle_id, expected_projects=expected_projects
        ),
        "required_capability": "conda:pyaedt2026v1",
        "env_profile": "pyaedt2026v1",
        "scheduling_profile": "fea_bursty",
        "aedt_backend": STANDALONE,
        "cpus": NODE_CANARY_HOST_CPUS,
        "memory_mb": NODE_CANARY_HOST_MEMORY_MB,
        "gpus": 0,
        "priority": NODE_CANARY_HOST_PRIORITY,
        "timeout_seconds": NODE_CANARY_HOST_TASK_TIMEOUT_SECONDS,
        "dedupe_key": str(bundle["host_dedupe_key"]),
        "entrypoint": NODE_CANARY_HOST_ENTRYPOINT,
        "requested_allocation_id": int(allocation["id"]),
        "same_node_as_task_id": 0,
        "account_name": str(allocation["account_name"]),
        "payload_json": payload_json,
    }


def _reconcile_exact_task_id(
    scheduler_url: str,
    *,
    name: str,
    dedupe_key: str,
    project: str,
) -> int | None:
    rows = _task_rows(
        _response_json(
            scheduler_client.requests.get(
                f"{scheduler_url}/api/tasks",
                params={"limit": 10_000, "name_prefix": name},
                timeout=30,
            ),
            "task reconciliation",
        ),
        "task reconciliation",
    )
    matches = [
        row
        for row in rows
        if row.get("name") == name
        and row.get("dedupe_key") == dedupe_key
        and str(row.get("project") or "") == project
    ]
    return max((int(row["id"]) for row in matches), default=None)


def _submit_exact_task(
    scheduler_url: str, payload: Mapping[str, object]
) -> int:
    name = str(payload["name"])
    dedupe_key = str(payload["dedupe_key"])
    project = str(payload["project"])
    existing = _reconcile_exact_task_id(
        scheduler_url,
        name=name,
        dedupe_key=dedupe_key,
        project=project,
    )
    if existing is not None:
        return existing
    try:
        response = scheduler_client.requests.post(
            f"{scheduler_url}/api/tasks", json=dict(payload), timeout=20
        )
    except Exception:
        recovered = _reconcile_exact_task_id(
            scheduler_url,
            name=name,
            dedupe_key=dedupe_key,
            project=project,
        )
        if recovered is not None:
            return recovered
        raise
    status_code = int(getattr(response, "status_code", 200) or 200)
    if status_code not in {200, 201}:
        recovered = _reconcile_exact_task_id(
            scheduler_url,
            name=name,
            dedupe_key=dedupe_key,
            project=project,
        )
        if recovered is not None:
            return recovered
        detail = str(getattr(response, "text", "") or "").strip()
        raise ExactTaskSubmissionRejected(
            f"scheduler rejected exact host placement for {name!r} "
            f"with HTTP {status_code}: {detail[:500]}"
        )
    response.raise_for_status()
    body = _response_json(response, "task submission")
    if isinstance(body, Mapping):
        task_id = body.get("task_id") or body.get("id")
        if isinstance(task_id, int) and not isinstance(task_id, bool) and task_id > 0:
            return task_id
    recovered = _reconcile_exact_task_id(
        scheduler_url,
        name=name,
        dedupe_key=dedupe_key,
        project=project,
    )
    if recovered is None:
        raise RuntimeError(f"scheduler accepted {name!r} without a task ID")
    return recovered


def _read_live_scheduler(
    policy: AttachRefillPolicy,
    scheduler_url: str,
) -> tuple[dict[str, object], list[dict[str, object]], int]:
    """Read one logical-capacity snapshot using GET requests only."""

    target = policy.project_concurrency_target
    statuses = ",".join(scheduler_client.MFT_ACTIVE_STATUSES)
    project_response = scheduler_client.requests.get(
        f"{scheduler_url}/api/projects/{scheduler_client.MFT_PROJECT}", timeout=30
    )
    project = _response_json(project_response, "project")
    if not isinstance(project, dict):
        raise RuntimeError("scheduler project response must be an object")
    project_tasks = _task_rows(
        _response_json(
            scheduler_client.requests.get(
                f"{scheduler_url}/api/tasks",
                params={
                    "limit": 10_000,
                    "project": scheduler_client.MFT_PROJECT,
                    "status": statuses,
                },
                timeout=30,
            ),
            "MFT project tasks",
        ),
        "MFT project",
    )
    legacy_tasks = _task_rows(
        _response_json(
            scheduler_client.requests.get(
                f"{scheduler_url}/api/tasks",
                params={
                    "limit": 10_000,
                    "name_prefix": scheduler_client.LEGACY_MFT_NAME_PREFIX,
                    "status": statuses,
                },
                timeout=30,
            ),
            "legacy MFT tasks",
        ),
        "legacy MFT",
    )
    # Pooled AEDT host tasks share the "mft-" name prefix but live in the
    # dedicated host project; they are capacity-tracked by the bundle state
    # machine, not the MFT project cap.
    legacy_tasks = [
        task
        for task in legacy_tasks
        if str((task or {}).get("project") or "").strip()
        != NODE_CANARY_HOST_PROJECT
    ]
    snapshot = scheduler_client.project_submission_snapshot(
        [project],
        project_tasks,
        target,
        legacy_tasks=legacy_tasks,
        require_exact_project_cap=True,
    )
    query_count = 3
    return snapshot, [], query_count


def _candidate_records(
    state: Mapping[str, object],
    count: int,
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: Mapping[str, object],
) -> tuple[tuple[ProjectCandidate, ...], list[dict[str, object]]]:
    cursor = int(state["candidate_cursor"])
    serial = int(state["next_serial"])
    seed = int(generation["identity"]["candidate_seed"])  # type: ignore[index]
    candidates: list[ProjectCandidate] = []
    records: list[dict[str, object]] = []
    for _ in range(count):
        cursor_before = cursor
        cursor, raw_index, raw_params = pinned_pilot.next_valid_candidate(
            cursor, seed=seed
        )
        params = pin_candidate_params(raw_params, policy.provenance)
        if (
            params.get("physics_data_revision")
            != policy.provenance.physics_data_revision
            or params.get("core_lamination_factor")
            != policy.provenance.core_lamination_factor
        ):
            raise RuntimeError("candidate physics pins were not applied")
        effective_params = scheduler_client.effective_verification_params(
            params, dict(profile)
        )
        if (
            effective_params.get("physics_data_revision")
            != policy.provenance.physics_data_revision
            or effective_params.get("core_lamination_factor")
            != policy.provenance.core_lamination_factor
        ):
            raise RuntimeError("effective candidate payload changed the physics pins")
        params_sha256 = _canonical_sha256(effective_params)
        name = (
            f"mft-camp-rv3-s{policy.provenance.solver_revision[:7]}-"
            f"g{str(generation['digest'])[:12]}-{serial:06d}"
        )
        candidate = ProjectCandidate(name=name, params_sha256=params_sha256)
        identity = scheduler_client.verification_submission_identity(
            name,
            params,
            profile,
            policy.provenance.solver_revision,
            policy.provenance.library_revision,
            dedupe_scope=str(generation["digest"]),
        )
        merged = identity["merged"]
        if (
            merged.get("physics_data_revision")
            != policy.provenance.physics_data_revision
            or merged.get("core_lamination_factor")
            != policy.provenance.core_lamination_factor
        ):
            raise RuntimeError("effective scheduler params changed the physics pins")
        candidates.append(candidate)
        records.append(
            {
                "action": "submit_rolling_replacement",
                "serial": serial,
                "name": name,
                "workdir": name.replace("-", "_"),
                "candidate_cursor_before": cursor_before,
                "candidate_cursor_after": cursor,
                "candidate_raw_index": raw_index,
                "params": params,
                "effective_params": effective_params,
                "params_sha256": params_sha256,
                "dedupe_key": identity["dedupe_key"],
                "dedupe_scope": generation["digest"],
            }
        )
        serial += 1
    return tuple(candidates), records


def _reserved_allocation_footprints(
    state: Mapping[str, object], generation: Mapping[str, object]
) -> dict[int, dict[str, int]]:
    footprints: dict[int, dict[str, int]] = {}
    client_cpus = int(generation["identity"]["cpus"])  # type: ignore[index]
    client_memory_mb = int(
        generation["identity"]["memory_mb"]  # type: ignore[index]
    )
    for raw in state["pooled_bundles"]:  # type: ignore[index]
        bundle = dict(raw)
        allocation_id = bundle.get("allocation_id")
        action_serials = bundle.get("action_serials")
        if (
            isinstance(allocation_id, bool)
            or not isinstance(allocation_id, int)
            or allocation_id <= 0
            or not isinstance(action_serials, list)
        ):
            continue
        phase = bundle.get("phase")
        host_task_id = bundle.get("host_task_id")
        host_terminal = bundle.get("host_terminal_status") in TASK_TERMINAL_STATES
        if phase in {"fallback_submit", "complete"}:
            # FEA-bursty tasks do not decrement allocation free_* counters.  A
            # discovery-fallback host can therefore keep occupying its node
            # until its own timeout.  Retain that host footprint after clients
            # fall back/complete, releasing it only after terminal observation.
            if (
                host_terminal
                or isinstance(host_task_id, bool)
                or not isinstance(host_task_id, int)
                or host_task_id <= 0
            ):
                continue
            item = footprints.setdefault(
                allocation_id, {"cpus": 0, "memory_mb": 0, "hosts": 0}
            )
            item["cpus"] += NODE_CANARY_HOST_CPUS
            item["memory_mb"] += NODE_CANARY_HOST_MEMORY_MB
            item["hosts"] += 1
            continue
        projects = len(action_serials)
        item = footprints.setdefault(
            allocation_id, {"cpus": 0, "memory_mb": 0, "hosts": 0}
        )
        item["cpus"] += NODE_CANARY_HOST_CPUS + projects * client_cpus
        item["memory_mb"] += (
            NODE_CANARY_HOST_MEMORY_MB + projects * client_memory_mb
        )
        item["hosts"] += 1
    return footprints


def _live_plan(
    policy: AttachRefillPolicy,
    state: Mapping[str, object],
    generation: Mapping[str, object],
    profile: Mapping[str, object],
    scheduler_url: str,
) -> dict[str, object]:
    snapshot, allocations, query_count = _read_live_scheduler(policy, scheduler_url)
    active = int(snapshot["project_active"])
    reserved_projects = len(state["reservations"])  # type: ignore[arg-type]
    accounted_active = active + reserved_projects
    deficit = logical_refill_deficit(
        target=policy.project_concurrency_target,
        active_project_tasks=accounted_active,
    )
    candidates, records = _candidate_records(
        state, deficit, policy, generation, profile
    )
    plan = AttachAwareRefillCoordinator(policy).plan_cycle(
        active_project_tasks=accounted_active,
        candidates=candidates,
        pool_status=None,
    )
    options_by_name: dict[str, dict[str, object]] = {}
    for item in plan["task_submission_options"]:  # type: ignore[index]
        options = dict(item)
        options["dedupe_scope"] = generation["digest"]
        options_by_name[str(options["candidate_name"])] = options
    for record in records:
        options = options_by_name[str(record["name"])]
        record["backend"] = options["aedt_backend"]
        record["scheduling_profile"] = options["scheduling_profile"]
        record["submission_env"] = options["submission_env"]
        record["bundle_id"] = options["bundle_id"]
        record["bundle_expected_rows"] = options["bundle_expected_rows"]
    pooled_bundles = [
        dict(bundle)
        for bundle in plan["bundles"]  # type: ignore[index]
        if bundle["backend"] == POOLED
    ]
    if pooled_bundles:
        allocation_payload = _response_json(
            scheduler_client.requests.get(
                f"{scheduler_url}/api/allocations", timeout=30
            ),
            "allocations",
        )
        query_count += 1
        if not isinstance(allocation_payload, list) or not all(
            isinstance(row, dict) for row in allocation_payload
        ):
            raise RuntimeError("scheduler returned an invalid allocation inventory")
        allocations = allocation_payload
    full_pooled_bundles = [
        bundle
        for bundle in pooled_bundles
        if int(bundle["expected_rows"]) == PRODUCTION_PROJECTS_PER_AEDT
    ]
    selections, next_host_account = select_host_allocations(
        allocations,
        [int(bundle["expected_rows"]) for bundle in full_pooled_bundles],
        client_cpus=int(generation["identity"]["cpus"]),  # type: ignore[index]
        client_memory_mb=int(
            generation["identity"]["memory_mb"]  # type: ignore[index]
        ),
        reserved_by_allocation=_reserved_allocation_footprints(
            state, generation
        ),
        last_account=str(state["last_host_account"]),
    )
    allocation_by_bundle = {
        str(bundle["bundle_id"]): selection
        for bundle, selection in zip(full_pooled_bundles, selections)
    }
    for bundle in pooled_bundles:
        allocation_by_bundle.setdefault(str(bundle["bundle_id"]), None)
    for record in records:
        selection = allocation_by_bundle.get(str(record["bundle_id"]))
        if selection is not None:
            record["requested_allocation_id"] = selection["id"]
            record["requested_allocation_account"] = selection["account_name"]
    plan["task_submission_options"] = list(options_by_name.values())
    plan.update(
        {
            "mode": "plan",
            "scheduler_url": scheduler_url,
            "scheduler_query_count": query_count,
            "scheduler_mutation_count": 0,
            "generation": dict(generation),
            "state_revision": state["state_revision"],
            "candidate_cursor": state["candidate_cursor"],
            "active_counts": snapshot["project_counts"],
            "active_project_tasks": active,
            "reserved_project_tasks": reserved_projects,
            "accounted_active_project_tasks": accounted_active,
            "logical_project_deficit": deficit,
            "would_submit": len(records),
            "would_replace": len(records),
            "planned_actions": records,
            "pooled_host_allocations": allocation_by_bundle,
            "next_host_account": next_host_account,
            "cancel_task_ids": [],
            "mass_cancel_authorized": False,
        }
    )
    return plan


def _json_copy(value: object) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _save_controller_state(
    path: Path,
    state: dict[str, object],
    generation: Mapping[str, object],
    scheduler_url: str,
) -> None:
    state["updated_at"] = _now()
    _validate_state(state, generation, scheduler_url)
    _atomic_save_state(path, state)


def _reserve_plan(
    state: dict[str, object],
    plan: Mapping[str, object],
    generation: Mapping[str, object],
) -> None:
    actions = [_json_copy(item) for item in plan["planned_actions"]]  # type: ignore[index]
    if not actions:
        return
    if state["reservations"]:
        raise RuntimeError("cannot reserve a new plan over pending actions")
    expected_serial = int(state["next_serial"])
    cursor = int(state["candidate_cursor"])
    for action in actions:
        if (
            int(action["serial"]) != expected_serial
            or int(action["candidate_cursor_before"]) != cursor
        ):
            raise RuntimeError("live plan does not continue the state ledger")
        expected_serial += 1
        cursor = int(action["candidate_cursor_after"])
    state["reservations"] = actions
    state["next_serial"] = expected_serial
    state["candidate_cursor"] = cursor
    allocations = dict(plan.get("pooled_host_allocations") or {})
    bundles = list(state["pooled_bundles"])  # type: ignore[arg-type]
    for raw_bundle in plan["bundles"]:  # type: ignore[index]
        if raw_bundle["backend"] != POOLED:
            continue
        bundle_id = str(raw_bundle["bundle_id"])
        bundle_actions = [
            action for action in actions if action["bundle_id"] == bundle_id
        ]
        action_serials = [int(action["serial"]) for action in bundle_actions]
        allocation = allocations.get(bundle_id)
        full_n2 = len(action_serials) == PRODUCTION_PROJECTS_PER_AEDT
        if not full_n2:
            phase = "fallback_submit"
            failure_reason = "pooled bundle is not a complete validated N=2 bundle"
        elif not isinstance(allocation, Mapping):
            phase = "fallback_submit"
            failure_reason = "no active allocation can fit the complete pooled bundle"
        else:
            phase = "host_submit"
            failure_reason = None
        paths = _coordination_paths(bundle_id)
        bundles.append(
            {
                "bundle_id": bundle_id,
                "phase": phase,
                "expected_projects": len(action_serials),
                "action_serials": action_serials,
                "actions": _json_copy(bundle_actions),
                "allocation_id": (
                    int(allocation["id"])
                    if isinstance(allocation, Mapping)
                    else None
                ),
                "allocation_account": (
                    str(allocation["account_name"])
                    if isinstance(allocation, Mapping)
                    else None
                ),
                "host_name": f"{bundle_id}-host",
                "host_dedupe_key": (
                    f"mft-node-canary-host:{bundle_id}:"
                    f"{str(generation['digest'])[:20]}"
                ),
                "host_task_id": None,
                "host_submitted_at": None,
                "discovery_deadline_at": None,
                "host_terminal_status": None,
                "host_terminal_at": None,
                "host_cancel_status": None,
                "host_cancel_error": None,
                "host_cancel_at": None,
                "host_cancel_task_id": None,
                "coordination_files": paths,
                "host_clone_root": (
                    f"~/slurm_scheduler/runs/{bundle_id}-host"
                ),
                "discovery": None,
                "client_task_ids": [None for _ in action_serials],
                "fallback_task_ids": [None for _ in action_serials],
                "client_fallback_task_ids": [None for _ in action_serials],
                "missing_candidate_indices": [],
                "failure_reason": failure_reason,
                "created_at": _now(),
                "updated_at": _now(),
            }
        )
    state["pooled_bundles"] = bundles
    if any(value is not None for value in allocations.values()):
        state["last_host_account"] = str(plan.get("next_host_account") or "")
    state["updated_at"] = _now()


def _reservation_by_serial(
    state: Mapping[str, object], serial: int
) -> dict[str, object] | None:
    for action in state["reservations"]:  # type: ignore[index]
        if int(action["serial"]) == serial:
            return action
    return None


def _submission_by_serial(
    state: Mapping[str, object], serial: int
) -> dict[str, object] | None:
    for submission in state["submissions"]:  # type: ignore[index]
        if int(submission["serial"]) == serial:
            return submission
    return None


def _commit_submission(
    state: dict[str, object], action: Mapping[str, object], task_id: int
) -> None:
    serial = int(action["serial"])
    existing = _submission_by_serial(state, serial)
    if existing is not None:
        if int(existing["task_id"]) != task_id:
            raise RuntimeError("candidate serial reconciled to conflicting task IDs")
        return
    reservations = list(state["reservations"])  # type: ignore[arg-type]
    if not any(int(item["serial"]) == serial for item in reservations):
        raise RuntimeError("cannot commit an unreserved candidate action")
    reservations = [
        item for item in reservations if int(item["serial"]) != serial
    ]
    submissions = list(state["submissions"])  # type: ignore[arg-type]
    submissions.append(
        {
            "name": action["name"],
            "task_id": task_id,
            "dedupe_key": action["dedupe_key"],
            "params_sha256": action["params_sha256"],
            "candidate_raw_index": action["candidate_raw_index"],
            "serial": action["serial"],
            "candidate_cursor_before": action["candidate_cursor_before"],
            "candidate_cursor_after": action["candidate_cursor_after"],
            "backend": action["backend"],
            "bundle_id": action.get("bundle_id"),
            "fallback_of": action.get("fallback_of"),
            "submitted_or_reconciled_at": _now(),
        }
    )
    state["reservations"] = reservations
    state["submissions"] = submissions
    state["state_revision"] = int(state["state_revision"]) + 1
    state["updated_at"] = _now()


def _standalone_fallback_action(
    action: Mapping[str, object],
    bundle_id: str,
    candidate_index: int,
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: Mapping[str, object],
) -> dict[str, object]:
    retry_name = (
        f"{action['name']}-sa-retry-{bundle_id[-8:]}-{candidate_index}"
    )
    candidate = ProjectCandidate(
        name=retry_name,
        params_sha256=str(action["params_sha256"]),
    )
    fallback_bundle = make_refill_bundles(
        (candidate,), policy, backend=STANDALONE, fallback_of=bundle_id
    )[0]
    options = task_submission_options(
        fallback_bundle, policy, candidate_index=0
    )
    identity = scheduler_client.verification_submission_identity(
        retry_name,
        dict(action["params"]),
        dict(profile),
        policy.provenance.solver_revision,
        policy.provenance.library_revision,
        dedupe_scope=str(generation["digest"]),
    )
    fallback = dict(_json_copy(action))
    fallback.update(
        {
            "name": retry_name,
            "workdir": retry_name.replace("-", "_"),
            "dedupe_key": identity["dedupe_key"],
            "dedupe_scope": generation["digest"],
            "backend": STANDALONE,
            "scheduling_profile": options["scheduling_profile"],
            "submission_env": options["submission_env"],
            "bundle_id": fallback_bundle.bundle_id,
            "bundle_expected_rows": 1,
            "fallback_of": bundle_id,
        }
    )
    return fallback


def _replace_reservation(
    state: dict[str, object], replacement: Mapping[str, object]
) -> None:
    serial = int(replacement["serial"])
    reservations = list(state["reservations"])  # type: ignore[arg-type]
    matches = [index for index, item in enumerate(reservations) if int(item["serial"]) == serial]
    if len(matches) != 1:
        raise RuntimeError("fallback replacement must target one reservation")
    reservations[matches[0]] = dict(_json_copy(replacement))
    state["reservations"] = reservations


def _prepare_bundle_fallback(
    state: dict[str, object],
    bundle: dict[str, object],
    reason: str,
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: Mapping[str, object],
    scheduler_url: str,
    lifecycle_events: list[dict[str, object]],
) -> None:
    entering_fallback = bundle.get("phase") != "fallback_submit"
    for index, serial in enumerate(bundle["action_serials"]):  # type: ignore[index]
        action = _reservation_by_serial(state, int(serial))
        if action is None:
            continue
        if action.get("backend") == STANDALONE and action.get("fallback_of") == bundle["bundle_id"]:
            continue
        _replace_reservation(
            state,
            _standalone_fallback_action(
                action,
                str(bundle["bundle_id"]),
                index,
                policy,
                generation,
                profile,
            ),
        )
    bundle["phase"] = "fallback_submit"
    bundle["failure_reason"] = reason
    bundle["updated_at"] = _now()
    if entering_fallback:
        event = _guarded_cancel_bundle_host(
            scheduler_url, bundle, trigger="bundle_fallback"
        )
        if event is not None:
            lifecycle_events.append(event)


def _interrupt_client_submission(
    state: dict[str, object],
    bundle: dict[str, object],
    reason: str,
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: Mapping[str, object],
    scheduler_url: str,
    lifecycle_events: list[dict[str, object]],
) -> str:
    """Move an interrupted N=2 admission to a resumable fallback phase."""

    task_ids = list(bundle["client_task_ids"])  # type: ignore[arg-type]
    admitted = [task_id for task_id in task_ids if task_id is not None]
    if not admitted:
        _prepare_bundle_fallback(
            state,
            bundle,
            reason,
            policy,
            generation,
            profile,
            scheduler_url,
            lifecycle_events,
        )
        return "fallback_submit"
    bundle["failure_reason"] = reason
    if all(task_id is not None for task_id in task_ids):
        # A crash can leave the final client ID durable before the phase flip.
        # Both clients are already admitted, so normal terminal reconciliation
        # remains authoritative even if the readiness switch has since closed.
        bundle["phase"] = "clients_tracked"
    else:
        bundle["phase"] = "clients_partial_tracked"
    bundle["updated_at"] = _now()
    return str(bundle["phase"])


def build_node_canary_client_submission(
    action: Mapping[str, object],
    bundle: Mapping[str, object],
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: Mapping[str, object],
) -> dict[str, object]:
    discovery = bundle.get("discovery")
    host_task_id = bundle.get("host_task_id")
    if not isinstance(discovery, Mapping):
        raise RuntimeError("pooled client submission requires persisted discovery")
    if (
        isinstance(host_task_id, bool)
        or not isinstance(host_task_id, int)
        or host_task_id <= 0
    ):
        raise RuntimeError("pooled client submission requires a host task ID")
    expected_projects = len(bundle["action_serials"])  # type: ignore[arg-type]
    submission_env = dict(action["submission_env"])
    submission_env.update(
        {
            "MFT_AEDT_BACKEND": POOLED,
            "MFT_AEDT_SHARED_CANARY": "1",
            "MFT_AEDT_SCHEDULER_URL": str(discovery["scheduler_url"]),
            "MFT_SLURM_SCHEDULER_ROOT": str(bundle["host_clone_root"]),
            "MFT_PHYSICS_DATA_REVISION": (
                policy.provenance.physics_data_revision
            ),
            "MFT_CORE_LAMINATION_FACTOR": str(
                policy.provenance.core_lamination_factor
            ),
        }
    )
    return {
        "name": str(action["name"]),
        "workdir": str(action["workdir"]),
        "params": dict(action["params"]),
        "profile": dict(profile),
        "mem_mb": int(generation["identity"]["memory_mb"]),  # type: ignore[index]
        "cpus": int(generation["identity"]["cpus"]),  # type: ignore[index]
        "solver_revision": policy.provenance.solver_revision,
        "library_revision": policy.provenance.library_revision,
        "required_project_cap": policy.project_concurrency_target,
        "aedt_backend": POOLED,
        "scheduling_profile": str(action["scheduling_profile"]),
        "submission_env": submission_env,
        "dedupe_scope": str(generation["digest"]),
        "entrypoint": NODE_CANARY_CLIENT_ENTRYPOINT,
        "same_node_as_task_id": host_task_id,
        "payload_json": {
            "aedt_canary_bundle_id": str(bundle["bundle_id"]),
            "aedt_canary_expected_projects": expected_projects,
        },
    }


def _standalone_submission(
    action: Mapping[str, object],
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: Mapping[str, object],
) -> dict[str, object]:
    submission_env = dict(action["submission_env"])
    submission_env["MFT_PHYSICS_DATA_REVISION"] = (
        policy.provenance.physics_data_revision
    )
    submission_env["MFT_CORE_LAMINATION_FACTOR"] = str(
        policy.provenance.core_lamination_factor
    )
    return {
        "name": str(action["name"]),
        "workdir": str(action["workdir"]),
        "params": dict(action["params"]),
        "profile": dict(profile),
        "mem_mb": int(generation["identity"]["memory_mb"]),  # type: ignore[index]
        "cpus": int(generation["identity"]["cpus"]),  # type: ignore[index]
        "solver_revision": policy.provenance.solver_revision,
        "library_revision": policy.provenance.library_revision,
        "required_project_cap": policy.project_concurrency_target,
        "aedt_backend": STANDALONE,
        "scheduling_profile": str(action["scheduling_profile"]),
        "submission_env": submission_env,
        "dedupe_scope": str(generation["digest"]),
    }


def _scheduler_task_record(scheduler_url: str, task_id: int) -> dict[str, object]:
    payload = _response_json(
        scheduler_client.requests.get(
            f"{scheduler_url}/api/tasks/{task_id}", timeout=30
        ),
        f"task {task_id}",
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"scheduler task {task_id} response must be an object")
    return payload


def _scheduler_task_stdout(scheduler_url: str, task_id: int) -> str:
    return _response_text(
        scheduler_client.requests.get(
            f"{scheduler_url}/api/tasks/{task_id}/stdout",
            params={"max_bytes": NODE_CANARY_STDOUT_MAX_BYTES},
            timeout=30,
        ),
        f"task {task_id} stdout",
    )


def _record_host_terminal(bundle: dict[str, object], status: str) -> None:
    if status not in TASK_TERMINAL_STATES:
        raise ValueError(f"host status is not terminal: {status}")
    now = _now()
    bundle["host_terminal_status"] = status
    bundle["host_terminal_at"] = now
    bundle["updated_at"] = now


def _record_host_cancel(
    bundle: dict[str, object],
    task_id: int,
    status: str,
    error: str | None = None,
) -> None:
    if status not in HOST_CANCEL_STATUSES:
        raise ValueError(f"invalid host cancellation status: {status}")
    now = _now()
    bundle["host_cancel_status"] = status
    bundle["host_cancel_error"] = error
    bundle["host_cancel_at"] = now
    bundle["host_cancel_task_id"] = task_id
    bundle["updated_at"] = now


def _guarded_cancel_bundle_host(
    scheduler_url: str,
    bundle: dict[str, object],
    *,
    trigger: str,
    task_id: int | None = None,
) -> dict[str, object] | None:
    """Fail-soft cancel of one exact, scheduler-verified bundle host."""

    target = bundle.get("host_task_id") if task_id is None else task_id
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        return None
    event: dict[str, object] = {
        "bundle_id": bundle.get("bundle_id"),
        "transition": "host_cancel",
        "trigger": trigger,
        "host_task_id": target,
    }
    try:
        task = _scheduler_task_record(scheduler_url, target)
    except Exception as exc:
        error = f"task verification failed: {type(exc).__name__}: {exc}"
        _record_host_cancel(bundle, target, "failed", error)
        event.update(host_cancel_status="failed", host_cancel_error=error)
        return event

    expected_name = str(bundle.get("host_name") or "")
    actual_name = str(task.get("name") or "")
    actual_project = str(task.get("project") or "")
    if actual_name != expected_name or actual_project != NODE_CANARY_HOST_PROJECT:
        error = (
            f"refused to cancel task {target}: expected name {expected_name!r} "
            f"in project {NODE_CANARY_HOST_PROJECT!r}, got name "
            f"{actual_name!r} in project {actual_project!r}"
        )
        _record_host_cancel(bundle, target, "identity_mismatch", error)
        event.update(
            host_cancel_status="identity_mismatch", host_cancel_error=error
        )
        return event

    task_status = _normalized_task_status(task.get("status") or task.get("state"))
    event["host_status"] = task_status
    if task_status in TASK_TERMINAL_STATES:
        _record_host_terminal(bundle, task_status)
        _record_host_cancel(bundle, target, "already_terminal")
        event["host_cancel_status"] = "already_terminal"
        return event

    try:
        response = scheduler_client.requests.post(
            f"{scheduler_url}/api/tasks/{target}/cancel", timeout=60
        )
        response.raise_for_status()
    except Exception as exc:
        error = f"cancel request failed: {type(exc).__name__}: {exc}"
        _record_host_cancel(bundle, target, "failed", error)
        event.update(host_cancel_status="failed", host_cancel_error=error)
        return event

    _record_host_cancel(bundle, target, "requested")
    event["host_cancel_status"] = "requested"
    return event


def _reconcile_state_owned_stale_hosts(
    state: dict[str, object],
    generation: Mapping[str, object],
    scheduler_url: str,
    state_path: Path,
    lifecycle_events: list[dict[str, object]],
) -> None:
    """Cancel active pooled hosts owned by fallback/completed state bundles."""

    stale_by_name = {
        str(bundle.get("host_name")): bundle
        for bundle in state["pooled_bundles"]  # type: ignore[index]
        if bundle.get("phase")
        in {"fallback_submit", "client_fallback_submit", "complete"}
        and str(bundle.get("host_name") or "").startswith(
            POOLED_BUNDLE_HOST_PREFIX
        )
    }
    if not stale_by_name:
        return
    try:
        rows = _task_rows(
            _response_json(
                scheduler_client.requests.get(
                    f"{scheduler_url}/api/tasks",
                    params={
                        "limit": 10_000,
                        "project": NODE_CANARY_HOST_PROJECT,
                        "name_prefix": POOLED_BUNDLE_HOST_PREFIX,
                        "status": ",".join(scheduler_client.MFT_ACTIVE_STATUSES),
                    },
                    timeout=30,
                ),
                "active pooled host reconciliation",
            ),
            "active pooled host reconciliation",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        lifecycle_events.append(
            {
                "transition": "stale_host_reconciliation_failed",
                "host_cancel_status": "failed",
                "host_cancel_error": error,
            }
        )
        changed = False
        for bundle in stale_by_name.values():
            task_id = bundle.get("host_task_id")
            if (
                isinstance(task_id, bool)
                or not isinstance(task_id, int)
                or task_id <= 0
            ):
                continue
            bundle_error = f"active host inventory failed: {error}"
            _record_host_cancel(bundle, task_id, "failed", bundle_error)
            lifecycle_events.append(
                {
                    "bundle_id": bundle.get("bundle_id"),
                    "transition": "host_cancel",
                    "trigger": "startup_reconciliation",
                    "host_task_id": task_id,
                    "host_cancel_status": "failed",
                    "host_cancel_error": bundle_error,
                }
            )
            changed = True
        if changed:
            _save_controller_state(state_path, state, generation, scheduler_url)
        return

    changed = False
    for task in rows:
        task_id = task.get("id")
        name = str(task.get("name") or "")
        status = _normalized_task_status(task.get("status") or task.get("state"))
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
            or not name.startswith(POOLED_BUNDLE_HOST_PREFIX)
            or str(task.get("project") or "") != NODE_CANARY_HOST_PROJECT
            or status not in scheduler_client.MFT_ACTIVE_STATUSES
        ):
            continue
        bundle = stale_by_name.get(name)
        if bundle is None:
            continue
        if bundle.get("host_task_id") is None:
            bundle["host_task_id"] = task_id
        event = _guarded_cancel_bundle_host(
            scheduler_url,
            bundle,
            trigger="startup_reconciliation",
            task_id=task_id,
        )
        if event is not None:
            lifecycle_events.append(event)
            changed = True
    if changed:
        _save_controller_state(state_path, state, generation, scheduler_url)


def _refresh_bundle_host_terminals(
    state: dict[str, object],
    generation: Mapping[str, object],
    scheduler_url: str,
    state_path: Path,
    lifecycle_events: list[dict[str, object]],
) -> None:
    """Persist terminal fallback/completed hosts before releasing capacity."""

    changed = False
    for bundle in state["pooled_bundles"]:  # type: ignore[index]
        if bundle.get("phase") not in {"fallback_submit", "complete"}:
            continue
        if bundle.get("host_terminal_status") in TASK_TERMINAL_STATES:
            continue
        task_id = bundle.get("host_task_id")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or task_id <= 0
        ):
            continue
        try:
            task = _scheduler_task_record(scheduler_url, task_id)
        except Exception:
            # This read only allows a conservative reservation to be released;
            # uncertain liveness must retain the host footprint.
            continue
        status = _normalized_task_status(task.get("status") or task.get("state"))
        if status not in TASK_TERMINAL_STATES:
            continue
        _record_host_terminal(bundle, status)
        lifecycle_events.append(
            {
                "bundle_id": bundle["bundle_id"],
                "transition": "host_terminal_observed",
                "host_task_id": task_id,
                "host_status": status,
            }
        )
        changed = True
    if changed:
        _save_controller_state(state_path, state, generation, scheduler_url)


def _required_task_id(task_id: object, name: object) -> int:
    if (
        isinstance(task_id, bool)
        or not isinstance(task_id, int)
        or task_id <= 0
    ):
        raise RuntimeError(f"scheduler returned no task ID for {name}")
    return task_id


def _advance_pooled_bundle(
    state: dict[str, object],
    bundle: dict[str, object],
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: Mapping[str, object],
    scheduler_url: str,
    state_path: Path,
    accepted: list[dict[str, object]],
    lifecycle_events: list[dict[str, object]],
) -> None:
    while True:
        phase = str(bundle["phase"])
        gate = pool_gate(policy)
        if phase in {"host_submit", "discovery_wait", "clients_submit"} and not gate[
            "eligible"
        ]:
            reason = f"node-local pool gate closed: {gate['reason']}"
            if phase == "host_submit" and bundle.get("host_task_id") is None:
                recovered = _reconcile_exact_task_id(
                    scheduler_url,
                    name=str(bundle["host_name"]),
                    dedupe_key=str(bundle["host_dedupe_key"]),
                    project=NODE_CANARY_HOST_PROJECT,
                )
                if recovered is not None:
                    bundle["host_task_id"] = recovered
                    bundle["host_submitted_at"] = _now()
            if phase == "clients_submit":
                next_phase = _interrupt_client_submission(
                    state,
                    bundle,
                    reason,
                    policy,
                    generation,
                    profile,
                    scheduler_url,
                    lifecycle_events,
                )
            else:
                _prepare_bundle_fallback(
                    state,
                    bundle,
                    reason,
                    policy,
                    generation,
                    profile,
                    scheduler_url,
                    lifecycle_events,
                )
                next_phase = "fallback_submit"
            lifecycle_events.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "transition": f"{phase}->{next_phase}",
                    "reason": reason,
                }
            )
            _save_controller_state(state_path, state, generation, scheduler_url)
            if next_phase == "fallback_submit":
                continue
            return
        if phase == "host_submit":
            allocation_id = bundle.get("allocation_id")
            allocation_account = bundle.get("allocation_account")
            if (
                isinstance(allocation_id, bool)
                or not isinstance(allocation_id, int)
                or allocation_id <= 0
                or not str(allocation_account or "").strip()
            ):
                _prepare_bundle_fallback(
                    state,
                    bundle,
                    "persisted pooled bundle has no valid host allocation",
                    policy,
                    generation,
                    profile,
                    scheduler_url,
                    lifecycle_events,
                )
                _save_controller_state(
                    state_path, state, generation, scheduler_url
                )
                continue
            if bundle.get("discovery_deadline_at") is None:
                # Persist the absolute bound before the POST.  A crash after
                # scheduler acceptance can then reconcile the exact host
                # without granting discovery a fresh ten-minute window.
                bundle["discovery_deadline_at"] = _deadline(
                    NODE_CANARY_DISCOVERY_TIMEOUT_SECONDS
                )
                bundle["updated_at"] = _now()
                _save_controller_state(
                    state_path, state, generation, scheduler_url
                )
            host_payload = _node_canary_host_payload(
                bundle,
                {"id": allocation_id, "account_name": allocation_account},
            )
            discovery_expired = datetime.now(timezone.utc) >= _timestamp(
                bundle["discovery_deadline_at"]
            )
            host_task_id = None
            if discovery_expired:
                host_task_id = _reconcile_exact_task_id(
                    scheduler_url,
                    name=str(host_payload["name"]),
                    dedupe_key=str(host_payload["dedupe_key"]),
                    project=str(host_payload["project"]),
                )
            if discovery_expired and host_task_id is None:
                _prepare_bundle_fallback(
                    state,
                    bundle,
                    "host discovery deadline elapsed before an exact host "
                    "submission could be reconciled",
                    policy,
                    generation,
                    profile,
                    scheduler_url,
                    lifecycle_events,
                )
                lifecycle_events.append(
                    {
                        "bundle_id": bundle["bundle_id"],
                        "transition": "host_submit->fallback_submit",
                        "reason": bundle["failure_reason"],
                    }
                )
                _save_controller_state(state_path, state, generation, scheduler_url)
                continue
            if host_task_id is None:
                try:
                    host_task_id = _submit_exact_task(
                        scheduler_url, host_payload
                    )
                except ExactTaskSubmissionRejected as exc:
                    _prepare_bundle_fallback(
                        state,
                        bundle,
                        f"host submission rejected: {exc}",
                        policy,
                        generation,
                        profile,
                        scheduler_url,
                        lifecycle_events,
                    )
                    lifecycle_events.append(
                        {
                            "bundle_id": bundle["bundle_id"],
                            "transition": "host_submit->fallback_submit",
                            "reason": bundle["failure_reason"],
                        }
                    )
                    _save_controller_state(
                        state_path, state, generation, scheduler_url
                    )
                    continue
            bundle["host_task_id"] = host_task_id
            bundle["host_submitted_at"] = _now()
            bundle["phase"] = "discovery_wait"
            bundle["updated_at"] = _now()
            lifecycle_events.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "transition": "host_submit->discovery_wait",
                    "host_task_id": host_task_id,
                    "requested_allocation_id": allocation_id,
                }
            )
            _save_controller_state(state_path, state, generation, scheduler_url)
            continue
        if phase == "discovery_wait":
            host_task_id = _required_task_id(
                bundle.get("host_task_id"), bundle.get("host_name")
            )
            discovery_expired = datetime.now(timezone.utc) >= _timestamp(
                bundle["discovery_deadline_at"]
            )
            try:
                task = _scheduler_task_record(scheduler_url, host_task_id)
            except Exception:
                if not discovery_expired:
                    raise
                _prepare_bundle_fallback(
                    state,
                    bundle,
                    f"host discovery timed out after {NODE_CANARY_DISCOVERY_TIMEOUT_SECONDS} seconds",
                    policy,
                    generation,
                    profile,
                    scheduler_url,
                    lifecycle_events,
                )
                lifecycle_events.append(
                    {
                        "bundle_id": bundle["bundle_id"],
                        "transition": "discovery_wait->fallback_submit",
                        "reason": bundle["failure_reason"],
                    }
                )
                _save_controller_state(state_path, state, generation, scheduler_url)
                continue
            status = _normalized_task_status(
                task.get("status") or task.get("state")
            )
            if status in TASK_TERMINAL_STATES:
                _record_host_terminal(bundle, status)
                _prepare_bundle_fallback(
                    state,
                    bundle,
                    f"host task {host_task_id} became terminal before discovery: {status}",
                    policy,
                    generation,
                    profile,
                    scheduler_url,
                    lifecycle_events,
                )
                lifecycle_events.append(
                    {
                        "bundle_id": bundle["bundle_id"],
                        "transition": "discovery_wait->fallback_submit",
                        "reason": bundle["failure_reason"],
                    }
                )
                _save_controller_state(state_path, state, generation, scheduler_url)
                continue
            if discovery_expired:
                _prepare_bundle_fallback(
                    state,
                    bundle,
                    f"host discovery timed out after {NODE_CANARY_DISCOVERY_TIMEOUT_SECONDS} seconds",
                    policy,
                    generation,
                    profile,
                    scheduler_url,
                    lifecycle_events,
                )
                lifecycle_events.append(
                    {
                        "bundle_id": bundle["bundle_id"],
                        "transition": "discovery_wait->fallback_submit",
                        "reason": bundle["failure_reason"],
                    }
                )
                _save_controller_state(state_path, state, generation, scheduler_url)
                continue
            stdout = _scheduler_task_stdout(scheduler_url, host_task_id)
            try:
                discovery = parse_node_canary_discovery(
                    stdout,
                    expected_projects=int(bundle["expected_projects"]),
                )
            except ValueError as exc:
                _prepare_bundle_fallback(
                    state,
                    bundle,
                    f"invalid host discovery: {exc}",
                    policy,
                    generation,
                    profile,
                    scheduler_url,
                    lifecycle_events,
                )
                lifecycle_events.append(
                    {
                        "bundle_id": bundle["bundle_id"],
                        "transition": "discovery_wait->fallback_submit",
                        "reason": bundle["failure_reason"],
                    }
                )
                _save_controller_state(state_path, state, generation, scheduler_url)
                continue
            if discovery is not None:
                bundle["discovery"] = discovery
                bundle["phase"] = "clients_submit"
                bundle["updated_at"] = _now()
                lifecycle_events.append(
                    {
                        "bundle_id": bundle["bundle_id"],
                        "transition": "discovery_wait->clients_submit",
                        "host_task_id": host_task_id,
                        "node": discovery["node"],
                    }
                )
                _save_controller_state(state_path, state, generation, scheduler_url)
                continue
            return
        if phase == "clients_submit":
            task_ids = list(bundle["client_task_ids"])  # type: ignore[arg-type]
            for index, serial in enumerate(bundle["action_serials"]):  # type: ignore[index]
                if task_ids[index] is not None:
                    continue
                submission = _submission_by_serial(state, int(serial))
                if submission is not None:
                    task_ids[index] = int(submission["task_id"])
                    bundle["client_task_ids"] = task_ids
                    _save_controller_state(
                        state_path, state, generation, scheduler_url
                    )
                    continue
                host_task_id = _required_task_id(
                    bundle.get("host_task_id"), bundle.get("host_name")
                )
                host_task = _scheduler_task_record(
                    scheduler_url, host_task_id
                )
                host_status = _normalized_task_status(
                    host_task.get("status") or host_task.get("state")
                )
                if host_status in TASK_TERMINAL_STATES:
                    _record_host_terminal(bundle, host_status)
                    reason = (
                        f"host task {host_task_id} became terminal during "
                        f"client admission: {host_status}"
                    )
                    next_phase = _interrupt_client_submission(
                        state,
                        bundle,
                        reason,
                        policy,
                        generation,
                        profile,
                        scheduler_url,
                        lifecycle_events,
                    )
                    lifecycle_events.append(
                        {
                            "bundle_id": bundle["bundle_id"],
                            "transition": f"clients_submit->{next_phase}",
                            "reason": reason,
                        }
                    )
                    _save_controller_state(
                        state_path, state, generation, scheduler_url
                    )
                    break
                action = _reservation_by_serial(state, int(serial))
                if action is None:
                    raise RuntimeError("pooled client action is neither reserved nor committed")
                kwargs = build_node_canary_client_submission(
                    action, bundle, policy, generation, profile
                )
                submitted_task_id = scheduler_client.submit_verification(**kwargs)
                if submitted_task_id is None:
                    reason = (
                        "scheduler definitively rejected a node-local client "
                        f"admission at bundle index {index}"
                    )
                    next_phase = _interrupt_client_submission(
                        state,
                        bundle,
                        reason,
                        policy,
                        generation,
                        profile,
                        scheduler_url,
                        lifecycle_events,
                    )
                    lifecycle_events.append(
                        {
                            "bundle_id": bundle["bundle_id"],
                            "transition": f"clients_submit->{next_phase}",
                            "reason": reason,
                        }
                    )
                    _save_controller_state(
                        state_path, state, generation, scheduler_url
                    )
                    break
                task_id = _required_task_id(submitted_task_id, action["name"])
                _commit_submission(state, action, task_id)
                task_ids[index] = task_id
                bundle["client_task_ids"] = task_ids
                bundle["updated_at"] = _now()
                accepted.append(
                    {
                        "name": action["name"],
                        "task_id": task_id,
                        "backend": POOLED,
                        "dedupe_key": action["dedupe_key"],
                        "bundle_id": bundle["bundle_id"],
                    }
                )
                _save_controller_state(state_path, state, generation, scheduler_url)
            if bundle["phase"] == "fallback_submit":
                continue
            if bundle["phase"] != "clients_submit":
                return
            bundle["phase"] = "clients_tracked"
            bundle["updated_at"] = _now()
            lifecycle_events.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "transition": "clients_submit->clients_tracked",
                    "client_task_ids": list(bundle["client_task_ids"]),
                }
            )
            _save_controller_state(state_path, state, generation, scheduler_url)
            return
        if phase == "fallback_submit":
            _prepare_bundle_fallback(
                state,
                bundle,
                str(bundle.get("failure_reason") or "pooled materialization failed"),
                policy,
                generation,
                profile,
                scheduler_url,
                lifecycle_events,
            )
            _save_controller_state(state_path, state, generation, scheduler_url)
            fallback_ids = list(bundle["fallback_task_ids"])  # type: ignore[arg-type]
            for index, serial in enumerate(bundle["action_serials"]):  # type: ignore[index]
                if fallback_ids[index] is not None:
                    continue
                submission = _submission_by_serial(state, int(serial))
                if submission is not None:
                    fallback_ids[index] = int(submission["task_id"])
                    continue
                action = _reservation_by_serial(state, int(serial))
                if action is None:
                    raise RuntimeError("fallback action is neither reserved nor committed")
                task_id = _required_task_id(
                    scheduler_client.submit_verification(
                        **_standalone_submission(
                            action, policy, generation, profile
                        )
                    ),
                    action["name"],
                )
                _commit_submission(state, action, task_id)
                fallback_ids[index] = task_id
                bundle["fallback_task_ids"] = fallback_ids
                bundle["updated_at"] = _now()
                accepted.append(
                    {
                        "name": action["name"],
                        "task_id": task_id,
                        "backend": STANDALONE,
                        "dedupe_key": action["dedupe_key"],
                        "fallback_of": bundle["bundle_id"],
                    }
                )
                _save_controller_state(state_path, state, generation, scheduler_url)
            bundle["fallback_task_ids"] = fallback_ids
            bundle["phase"] = "complete"
            bundle["updated_at"] = _now()
            lifecycle_events.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "transition": "fallback_submit->complete",
                    "fallback_task_ids": fallback_ids,
                }
            )
            _save_controller_state(state_path, state, generation, scheduler_url)
            return
        return


def _materialize_reservations(
    state: dict[str, object],
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: Mapping[str, object],
    scheduler_url: str,
    state_path: Path,
    accepted: list[dict[str, object]],
    lifecycle_events: list[dict[str, object]],
) -> None:
    for bundle in state["pooled_bundles"]:  # type: ignore[index]
        if bundle["phase"] in {
            "host_submit",
            "discovery_wait",
            "clients_submit",
            "fallback_submit",
        }:
            _advance_pooled_bundle(
                state,
                bundle,
                policy,
                generation,
                profile,
                scheduler_url,
                state_path,
                accepted,
                lifecycle_events,
            )
    bundled_serials = {
        int(serial)
        for bundle in state["pooled_bundles"]  # type: ignore[index]
        for serial in bundle["action_serials"]
    }
    for raw_action in list(state["reservations"]):  # type: ignore[arg-type]
        action = dict(raw_action)
        if int(action["serial"]) in bundled_serials:
            continue
        task_id = _required_task_id(
            scheduler_client.submit_verification(
                **_standalone_submission(action, policy, generation, profile)
            ),
            action["name"],
        )
        _commit_submission(state, action, task_id)
        accepted.append(
            {
                "name": action["name"],
                "task_id": task_id,
                "backend": STANDALONE,
                "dedupe_key": action["dedupe_key"],
            }
        )
        _save_controller_state(state_path, state, generation, scheduler_url)


def _tracked_refill_bundle(
    bundle: Mapping[str, object], policy: AttachRefillPolicy
) -> RefillBundle:
    actions = bundle.get("actions")
    if not isinstance(actions, list):
        raise RuntimeError("tracked pooled bundle is missing candidate actions")
    candidates = tuple(
        ProjectCandidate(
            name=str(action["name"]),
            params_sha256=str(action["params_sha256"]),
        )
        for action in actions
    )
    return RefillBundle(
        bundle_id=str(bundle["bundle_id"]),
        backend=POOLED,
        projects_per_aedt=policy.projects_per_aedt,
        expected_rows=len(candidates),
        candidates=candidates,
        scheduling_profile=policy.pooled_profile,
        policy_digest=policy.digest,
        provenance_digest=policy.provenance.digest,
    )


def _reconcile_tracked_bundles(
    state: dict[str, object],
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: Mapping[str, object],
    scheduler_url: str,
    state_path: Path,
    accepted: list[dict[str, object]],
    lifecycle_events: list[dict[str, object]],
) -> None:
    coordinator = AttachAwareRefillCoordinator(policy)
    for bundle in state["pooled_bundles"]:  # type: ignore[index]
        tracked_phase = str(bundle["phase"])
        if tracked_phase in {"clients_tracked", "clients_partial_tracked"}:
            indexed_task_ids = list(bundle["client_task_ids"])
            submitted = [
                (index, int(value))
                for index, value in enumerate(indexed_task_ids)
                if value is not None
            ]
            statuses = {
                task_id: _normalized_task_status(
                    _scheduler_task_record(scheduler_url, task_id).get("status")
                )
                for _, task_id in submitted
            }
            if not all(
                status in TASK_TERMINAL_STATES for status in statuses.values()
            ):
                continue
            host_event = _guarded_cancel_bundle_host(
                scheduler_url, bundle, trigger="all_clients_terminal"
            )
            if host_event is not None:
                lifecycle_events.append(host_event)
                _save_controller_state(
                    state_path, state, generation, scheduler_url
                )
            accepted_rows: list[int] = []
            transport_unavailable = False
            for task_id, status in statuses.items():
                if status != "completed":
                    continue
                try:
                    fetched = scheduler_client.fetch_result(
                        task_id,
                        attempts=1,
                        retry_delay=0,
                        expected_revision=policy.provenance.solver_revision,
                        expected_library_revision=(
                            policy.provenance.library_revision
                        ),
                    )
                except scheduler_client.ResultFetchError:
                    transport_unavailable = True
                    break
                if fetched.state == scheduler_client.RESULT_VALID:
                    accepted_rows.append(task_id)
            if transport_unavailable:
                continue
            if tracked_phase == "clients_tracked":
                task_ids = [int(value) for value in indexed_task_ids]
                decision = coordinator.plan_failed_bundle_fallback(
                    _tracked_refill_bundle(bundle, policy),
                    task_ids=task_ids,
                    task_statuses=statuses,
                    accepted_row_task_ids=accepted_rows,
                )
                if decision["action"] == "wait":
                    continue
                missing_indices = list(
                    decision.get("missing_candidate_indices") or []
                )
            else:
                accepted_set = set(accepted_rows)
                missing_indices = [
                    index
                    for index, value in enumerate(indexed_task_ids)
                    if value is None or int(value) not in accepted_set
                ]
            if not missing_indices:
                bundle["phase"] = "complete"
                bundle["updated_at"] = _now()
                lifecycle_events.append(
                    {
                        "bundle_id": bundle["bundle_id"],
                        "transition": f"{tracked_phase}->complete",
                        "accepted_row_task_ids": accepted_rows,
                    }
                )
                _save_controller_state(
                    state_path, state, generation, scheduler_url
                )
                continue
            bundle["missing_candidate_indices"] = missing_indices
            bundle["phase"] = "client_fallback_submit"
            bundle["failure_reason"] = (
                "admitted or missing pooled clients lack accepted result rows"
            )
            bundle["updated_at"] = _now()
            lifecycle_events.append(
                {
                    "bundle_id": bundle["bundle_id"],
                    "transition": (
                        f"{tracked_phase}->client_fallback_submit"
                    ),
                    "missing_candidate_indices": list(
                        bundle["missing_candidate_indices"]
                    ),
                }
            )
            _save_controller_state(state_path, state, generation, scheduler_url)
        if bundle["phase"] != "client_fallback_submit":
            continue
        fallback_ids = list(bundle["client_fallback_task_ids"])
        original_actions = list(bundle["actions"])
        for index in bundle["missing_candidate_indices"]:
            index = int(index)
            if fallback_ids[index] is not None:
                continue
            serial = int(bundle["action_serials"][index])
            action = _standalone_fallback_action(
                original_actions[index],
                str(bundle["bundle_id"]),
                index,
                policy,
                generation,
                profile,
            )
            committed = _submission_by_serial(state, serial)
            if committed is None:
                reservation = _reservation_by_serial(state, serial)
                if reservation is None:
                    raise RuntimeError(
                        "missing client has neither a reservation nor submission"
                    )
                if not (
                    reservation.get("backend") == STANDALONE
                    and reservation.get("fallback_of") == bundle["bundle_id"]
                ):
                    _replace_reservation(state, action)
                    _save_controller_state(
                        state_path, state, generation, scheduler_url
                    )
                    reservation = _reservation_by_serial(state, serial)
                if reservation is None:
                    raise RuntimeError("standalone fallback intent was not persisted")
                action = reservation
            submitted_task_id = scheduler_client.submit_verification(
                **_standalone_submission(
                    action, policy, generation, profile
                )
            )
            if submitted_task_id is None:
                return
            task_id = _required_task_id(submitted_task_id, action["name"])
            if committed is None:
                _commit_submission(state, action, task_id)
            fallback_ids[index] = task_id
            bundle["client_fallback_task_ids"] = fallback_ids
            bundle["updated_at"] = _now()
            accepted.append(
                {
                    "name": action["name"],
                    "task_id": task_id,
                    "backend": STANDALONE,
                    "dedupe_key": action["dedupe_key"],
                    "fallback_of": bundle["bundle_id"],
                }
            )
            _save_controller_state(state_path, state, generation, scheduler_url)
        bundle["phase"] = "complete"
        bundle["updated_at"] = _now()
        lifecycle_events.append(
            {
                "bundle_id": bundle["bundle_id"],
                "transition": "client_fallback_submit->complete",
                "fallback_task_ids": fallback_ids,
            }
        )
        _save_controller_state(state_path, state, generation, scheduler_url)


def _run_cycle(
    policy: AttachRefillPolicy,
    generation: Mapping[str, object],
    profile: dict[str, object],
    scheduler_url: str,
    state_path: Path,
) -> dict[str, object]:
    state_path = _validate_state_path(state_path)
    with FileLock(str(state_path) + ".lock", timeout=30):
        state, _ = _load_or_initialize_state(
            state_path, generation, scheduler_url, create=False
        )
        with scheduler_client.campaign_mutation_lock():
            accepted: list[dict[str, object]] = []
            lifecycle_events: list[dict[str, object]] = []
            _reconcile_state_owned_stale_hosts(
                state,
                generation,
                scheduler_url,
                state_path,
                lifecycle_events,
            )
            _refresh_bundle_host_terminals(
                state,
                generation,
                scheduler_url,
                state_path,
                lifecycle_events,
            )
            _reconcile_tracked_bundles(
                state,
                policy,
                generation,
                profile,
                scheduler_url,
                state_path,
                accepted,
                lifecycle_events,
            )
            _materialize_reservations(
                state,
                policy,
                generation,
                profile,
                scheduler_url,
                state_path,
                accepted,
                lifecycle_events,
            )
            plan = _live_plan(
                policy, state, generation, profile, scheduler_url
            )
            if (
                not state["reservations"]
                and not accepted
                and plan["planned_actions"]
            ):
                _reserve_plan(state, plan, generation)
                _save_controller_state(
                    state_path, state, generation, scheduler_url
                )
                _materialize_reservations(
                    state,
                    policy,
                    generation,
                    profile,
                    scheduler_url,
                    state_path,
                    accepted,
                    lifecycle_events,
                )
    pending_reservations = len(state["reservations"])
    if accepted and not pending_reservations:
        action = "rolling_refill_complete"
    elif accepted:
        action = "rolling_refill_progress"
    elif pending_reservations:
        action = "pooled_bundle_pending"
    elif lifecycle_events:
        action = "pooled_bundle_lifecycle_progress"
    else:
        action = "no_refill_needed"
    return {
        "mode": "run",
        "action": action,
        "generation": dict(generation),
        "active_project_tasks_before": plan["active_project_tasks"],
        "logical_project_deficit_before": plan["logical_project_deficit"],
        "accepted_or_reconciled_count": len(accepted),
        "accepted_or_reconciled": accepted,
        "bundle_lifecycle_events": lifecycle_events,
        "pending_reservation_count": pending_reservations,
        "pooled_bundles": [
            {
                "bundle_id": bundle["bundle_id"],
                "phase": bundle["phase"],
                "host_task_id": bundle.get("host_task_id"),
                "host_terminal_status": bundle.get("host_terminal_status"),
                "host_cancel_status": bundle.get("host_cancel_status"),
                "host_cancel_error": bundle.get("host_cancel_error"),
                "host_cancel_at": bundle.get("host_cancel_at"),
                "host_cancel_task_id": bundle.get("host_cancel_task_id"),
                "client_task_ids": bundle.get("client_task_ids"),
                "fallback_task_ids": bundle.get("fallback_task_ids"),
                "client_fallback_task_ids": bundle.get(
                    "client_fallback_task_ids"
                ),
                "failure_reason": bundle.get("failure_reason"),
            }
            for bundle in state["pooled_bundles"]
        ],
        "state_path": str(state_path),
        "state_revision": state["state_revision"],
        "cancel_task_ids": [],
        "mass_cancel_authorized": False,
    }


def _offline(args: argparse.Namespace) -> dict[str, object]:
    policy_payload = _load_json(args.policy)
    if not isinstance(policy_payload, Mapping):
        raise ValueError("policy JSON must be an object")
    candidate_payload = _load_json(args.candidates)
    if not isinstance(candidate_payload, list):
        raise ValueError("candidate JSON must be a list")
    status_payload = _load_json(args.pool_status) if args.pool_status else None
    if status_payload is not None and not isinstance(status_payload, Mapping):
        raise ValueError("pool status JSON must be an object")
    return AttachAwareRefillCoordinator(load_policy(policy_payload)).plan_cycle(
        active_project_tasks=args.active_project_tasks,
        candidates=tuple(ProjectCandidate(**item) for item in candidate_payload),
        pool_status=status_payload,
    )


def _add_live_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--scheduler-url", default="http://127.0.0.1:8000"
    )
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--candidate-seed", type=int, default=DEFAULT_CANDIDATE_SEED)
    parser.add_argument("--cpus", type=int, default=DEFAULT_CPUS)
    parser.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB)
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan", help="GET live capacity and print a read-only refill plan"
    )
    _add_live_arguments(plan)
    plan.add_argument("--output", type=Path)
    run = subparsers.add_parser(
        "run", help="run the explicitly authorized rolling refill controller"
    )
    _add_live_arguments(run)
    run.add_argument("--authorize-generation", required=True)
    run.add_argument("--loop", type=int)
    offline = subparsers.add_parser(
        "offline", help="run the original file-to-file pure planner"
    )
    offline.add_argument("--policy", required=True)
    offline.add_argument("--candidates", required=True)
    offline.add_argument("--pool-status")
    offline.add_argument("--active-project-tasks", required=True, type=int)
    offline.add_argument("--output", type=Path)
    return parser


def _emit(payload: object, output: Path | None = None, *, compact: bool = False) -> None:
    text = json.dumps(
        payload,
        indent=None if compact else 2,
        separators=(",", ":") if compact else None,
        sort_keys=True,
    ) + "\n"
    if output is None:
        print(text, end="", flush=True)
    else:
        output.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve the former CLI spelling as a compatibility alias for offline.
    if raw_argv and raw_argv[0].startswith("-") and raw_argv[0] not in {"-h", "--help"}:
        raw_argv.insert(0, "offline")
    args = _parser().parse_args(raw_argv)
    try:
        if args.command == "offline":
            result = _offline(args)
            _emit(result, args.output)
            return 0
        loop_seconds = getattr(args, "loop", None)
        if loop_seconds is not None and loop_seconds < DEFAULT_LOOP_SECONDS:
            raise ValueError("--loop must be at least 60 seconds")
        scheduler_url = _normalize_scheduler_url(args.scheduler_url)
        _configure_scheduler(scheduler_url)
        policy = _load_policy(args.policy)
        profile = _load_profile(args.timeout_seconds)
        generation = _generation_contract(
            policy,
            candidate_seed=args.candidate_seed,
            profile=profile,
            cpus=args.cpus,
            memory_mb=args.memory_mb,
        )
        state_path = _validate_state_path(args.state_path)
        if args.command == "plan":
            with FileLock(str(state_path) + ".lock", timeout=30):
                state, initialized = _load_or_initialize_state(
                    state_path, generation, scheduler_url, create=True
                )
                result = _live_plan(
                    policy, state, generation, profile, scheduler_url
                )
            result["state_path"] = str(state_path)
            result["state_initialized"] = initialized
            _emit(result, args.output)
            return 0
        if args.authorize_generation != generation["id"]:
            raise ValueError(
                "run requires exact --authorize-generation "
                f"{generation['id']}"
            )
        while True:
            started = time.monotonic()
            result = _run_cycle(
                policy, generation, profile, scheduler_url, state_path
            )
            _emit(result, compact=loop_seconds is not None)
            if loop_seconds is None:
                return 0
            elapsed = max(0.0, time.monotonic() - started)
            time.sleep(max(0.0, float(loop_seconds) - elapsed))
    except Exception as exc:
        failed = {
            "mode": args.command,
            "action": "failed_closed",
            "error": f"{type(exc).__name__}: {exc}",
            "scheduler_mutation_count": 0 if args.command != "run" else None,
            "mutation_may_have_occurred": args.command == "run",
            "cancel_task_ids": [],
            "mass_cancel_authorized": False,
        }
        print(json.dumps(failed, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
