"""Pure planning contracts for the attach-aware MFT refill controller.

The production controller still owns candidate generation and the scheduler
mutation lock.  This module deliberately owns only the backend decision,
bundle ledger, capacity arithmetic and immutable provenance.  Keeping those
decisions pure makes the standalone rollback path usable even when the AEDT
pool itself is unavailable.

One *logical project* is one simulation/result row.  A pooled bundle contains
up to ``projects_per_aedt`` logical projects and therefore has
``expected_rows == len(projects)``.  The scheduler still receives one task per
logical project; the AEDT pool groups their leases onto a Desktop.  This keeps
the scheduler project cap expressed in logical project concurrency, rather
than accidentally changing it to a Desktop count.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


STANDALONE = "standalone"
POOLED = "pooled"
FEA_BURSTY = "fea_bursty"
REVIEWED_PROJECT_CONCURRENCY_CAP = 500
ACTIVE_TASK_STATES = frozenset({"queued", "attaching", "running"})
TERMINAL_TASK_STATES = frozenset({"completed", "failed", "cancelled", "timeout"})
FAILURE_TASK_STATES = frozenset({"failed", "cancelled", "timeout"})
_SHA40 = re.compile(r"[0-9a-f]{40}")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _full_sha(name: str, value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA40.fullmatch(normalized):
        raise ValueError(f"{name} must be a full 40-character git SHA")
    return normalized


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class RevisionProvenance:
    """The exact code/data interpretation used by submitted simulations."""

    solver_revision: str
    library_revision: str
    data_contract_revision: str
    physics_data_revision: str
    core_lamination_factor: float
    scheduler_selector_revision: str
    scheduler_runtime_revision: str
    controller_base_revision: str
    attach_canary_revision: str
    attach_validation_revision: str
    attach_validation_scheduler_revision: str
    attach_timeout_validation_scheduler_revision: str

    def __post_init__(self) -> None:
        for field in (
            "solver_revision",
            "library_revision",
            "scheduler_selector_revision",
            "scheduler_runtime_revision",
            "controller_base_revision",
            "attach_canary_revision",
            "attach_validation_revision",
            "attach_validation_scheduler_revision",
            "attach_timeout_validation_scheduler_revision",
        ):
            object.__setattr__(self, field, _full_sha(field, getattr(self, field)))
        contract = str(self.data_contract_revision or "").strip()
        if not contract or len(contract) > 160:
            raise ValueError("data_contract_revision must be 1..160 characters")
        object.__setattr__(self, "data_contract_revision", contract)
        if not isinstance(self.physics_data_revision, str):
            raise ValueError("physics_data_revision must be a string")
        physics_revision = self.physics_data_revision.strip()
        if (
            not physics_revision
            or len(physics_revision) > 160
            or any(character.isspace() for character in physics_revision)
        ):
            raise ValueError(
                "physics_data_revision must be a 1..160 character token"
            )
        object.__setattr__(self, "physics_data_revision", physics_revision)
        if isinstance(self.core_lamination_factor, bool):
            raise ValueError(
                "core_lamination_factor must be finite and satisfy 0 < kf <= 1"
            )
        try:
            lamination_factor = float(self.core_lamination_factor)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "core_lamination_factor must be finite and satisfy 0 < kf <= 1"
            ) from exc
        if not math.isfinite(lamination_factor) or not 0 < lamination_factor <= 1:
            raise ValueError(
                "core_lamination_factor must be finite and satisfy 0 < kf <= 1"
            )
        object.__setattr__(self, "core_lamination_factor", lamination_factor)

    def as_dict(self) -> dict[str, object]:
        return {
            "solver_revision": self.solver_revision,
            "library_revision": self.library_revision,
            "data_contract_revision": self.data_contract_revision,
            "physics_data_revision": self.physics_data_revision,
            "core_lamination_factor": self.core_lamination_factor,
            "scheduler_selector_revision": self.scheduler_selector_revision,
            "scheduler_runtime_revision": self.scheduler_runtime_revision,
            "controller_base_revision": self.controller_base_revision,
            "attach_canary_revision": self.attach_canary_revision,
            "attach_validation_revision": self.attach_validation_revision,
            "attach_validation_scheduler_revision": (
                self.attach_validation_scheduler_revision
            ),
            "attach_timeout_validation_scheduler_revision": (
                self.attach_timeout_validation_scheduler_revision
            ),
        }

    @property
    def digest(self) -> str:
        return _canonical_sha256(self.as_dict())


def pin_candidate_params(
    params: Mapping[str, object], provenance: RevisionProvenance
) -> dict[str, object]:
    """Copy a candidate payload and apply the reviewed physics identity."""

    if not isinstance(params, Mapping):
        raise TypeError("candidate params must be a mapping")
    if not isinstance(provenance, RevisionProvenance):
        raise TypeError("candidate provenance must be RevisionProvenance")
    pinned = dict(params)
    pinned["core_lamination_factor"] = provenance.core_lamination_factor
    pinned["physics_data_revision"] = provenance.physics_data_revision
    return pinned


@dataclass(frozen=True)
class AttachRefillPolicy:
    """Operator-reviewed backend and logical-capacity policy.

    ``validated_projects_per_aedt`` is separate from the requested value.  A
    future N is accepted by the code only after evidence explicitly validates
    that same or a larger N; there is no hard-coded 1:2 ceiling here.
    """

    primary_backend: str
    project_concurrency_target: int
    max_aedt_sessions: int
    projects_per_aedt: int
    validated_projects_per_aedt: int
    provenance: RevisionProvenance
    pooled_fraction: float = 0.0
    standalone_profile: str = FEA_BURSTY
    pooled_profile: str = FEA_BURSTY
    failed_bundle_fallback: str = STANDALONE

    def __post_init__(self) -> None:
        if self.primary_backend not in {STANDALONE, POOLED}:
            raise ValueError("primary_backend must be standalone or pooled")
        if self.failed_bundle_fallback != STANDALONE:
            raise ValueError("failed pooled bundles may fall back only to standalone")
        if isinstance(self.pooled_fraction, bool):
            raise ValueError("pooled_fraction must be finite and between 0 and 1")
        try:
            pooled_fraction = float(self.pooled_fraction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "pooled_fraction must be finite and between 0 and 1"
            ) from exc
        if not math.isfinite(pooled_fraction) or not 0 <= pooled_fraction <= 1:
            raise ValueError("pooled_fraction must be finite and between 0 and 1")
        object.__setattr__(self, "pooled_fraction", pooled_fraction)
        for field in (
            "project_concurrency_target",
            "max_aedt_sessions",
            "projects_per_aedt",
            "validated_projects_per_aedt",
        ):
            _positive_int(field, getattr(self, field))
        if self.project_concurrency_target > REVIEWED_PROJECT_CONCURRENCY_CAP:
            raise ValueError(
                "project_concurrency_target exceeds the reviewed cap "
                f"{REVIEWED_PROJECT_CONCURRENCY_CAP}"
            )
        if self.projects_per_aedt > self.validated_projects_per_aedt:
            raise ValueError(
                "projects_per_aedt exceeds the attach validation evidence"
            )
        if self.primary_backend == POOLED:
            # Authenticate the complete pooled envelope even while a staged
            # policy has pooled_fraction=0.0 and selects standalone at runtime.
            required_aedt_sessions = math.ceil(
                self.project_concurrency_target / self.projects_per_aedt
            )
            capacity_checks = (
                required_aedt_sessions <= self.max_aedt_sessions,
                self.project_concurrency_target <= self.max_pooled_projects,
            )
            if not all(capacity_checks):
                raise ValueError(
                    "AEDT session ceiling cannot cover the logical project target"
                )
        if self.standalone_profile != FEA_BURSTY:
            raise ValueError("standalone FEA scheduling profile must be fea_bursty")
        if self.pooled_profile != FEA_BURSTY:
            raise ValueError("pooled FEA scheduling profile must be fea_bursty")

    @property
    def max_pooled_projects(self) -> int:
        return self.max_aedt_sessions * self.projects_per_aedt

    def profile_for(self, backend: str) -> str:
        if backend == STANDALONE:
            return self.standalone_profile
        if backend == POOLED:
            return self.pooled_profile
        raise ValueError(f"unknown backend: {backend!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "primary_backend": self.primary_backend,
            "project_concurrency_target": self.project_concurrency_target,
            "max_aedt_sessions": self.max_aedt_sessions,
            "projects_per_aedt": self.projects_per_aedt,
            "validated_projects_per_aedt": self.validated_projects_per_aedt,
            "pooled_fraction": self.pooled_fraction,
            "standalone_profile": self.standalone_profile,
            "pooled_profile": self.pooled_profile,
            "failed_bundle_fallback": self.failed_bundle_fallback,
            "provenance": self.provenance.as_dict(),
        }

    @property
    def digest(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class ProjectCandidate:
    name: str
    params_sha256: str

    def __post_init__(self) -> None:
        if not str(self.name or "").strip():
            raise ValueError("candidate name is empty")
        digest = str(self.params_sha256 or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("params_sha256 must be a 64-character SHA256")
        object.__setattr__(self, "params_sha256", digest)


@dataclass(frozen=True)
class RefillBundle:
    bundle_id: str
    backend: str
    projects_per_aedt: int
    expected_rows: int
    candidates: tuple[ProjectCandidate, ...]
    scheduling_profile: str
    policy_digest: str
    provenance_digest: str
    fallback_of: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {STANDALONE, POOLED}:
            raise ValueError("bundle backend is invalid")
        _positive_int("projects_per_aedt", self.projects_per_aedt)
        _positive_int("expected_rows", self.expected_rows)
        if self.expected_rows != len(self.candidates):
            raise ValueError("bundle expected_rows must equal candidate count")
        if self.backend == POOLED and len(self.candidates) > self.projects_per_aedt:
            raise ValueError("pooled bundle exceeds projects_per_aedt")
        if self.scheduling_profile != FEA_BURSTY:
            raise ValueError("every FEA backend must use fea_bursty")

    def as_dict(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "backend": self.backend,
            "projects_per_aedt": self.projects_per_aedt,
            "expected_rows": self.expected_rows,
            "candidates": [candidate.__dict__ for candidate in self.candidates],
            "scheduling_profile": self.scheduling_profile,
            "policy_digest": self.policy_digest,
            "provenance_digest": self.provenance_digest,
            "fallback_of": self.fallback_of,
        }


def desired_aedt_sessions(projects: int, projects_per_aedt: int) -> int:
    if isinstance(projects, bool) or not isinstance(projects, int) or projects < 0:
        raise ValueError("projects must be a non-negative integer")
    _positive_int("projects_per_aedt", projects_per_aedt)
    return math.ceil(projects / projects_per_aedt) if projects else 0


def logical_refill_deficit(*, target: int, active_project_tasks: int) -> int:
    _positive_int("target", target)
    if (
        isinstance(active_project_tasks, bool)
        or not isinstance(active_project_tasks, int)
        or active_project_tasks < 0
    ):
        raise ValueError("active_project_tasks must be a non-negative integer")
    return max(0, target - active_project_tasks)


def _bundle_id(
    *,
    policy: AttachRefillPolicy,
    backend: str,
    candidates: Sequence[ProjectCandidate],
    fallback_of: str | None,
) -> str:
    digest = _canonical_sha256(
        {
            "policy_digest": policy.digest,
            "backend": backend,
            "fallback_of": fallback_of,
            "candidates": [candidate.__dict__ for candidate in candidates],
        }
    )
    return f"mft-aedt-{backend}-{digest[:20]}"


def make_refill_bundles(
    candidates: Sequence[ProjectCandidate],
    policy: AttachRefillPolicy,
    *,
    backend: str | None = None,
    fallback_of: str | None = None,
) -> tuple[RefillBundle, ...]:
    """Group candidates without changing logical project accounting."""

    selected = policy.primary_backend if backend is None else backend
    if selected not in {STANDALONE, POOLED}:
        raise ValueError("backend must be standalone or pooled")
    if fallback_of is not None and selected != STANDALONE:
        raise ValueError("a failed bundle fallback must be standalone")
    group_size = policy.projects_per_aedt if selected == POOLED else 1
    bundles: list[RefillBundle] = []
    for offset in range(0, len(candidates), group_size):
        group = tuple(candidates[offset : offset + group_size])
        if not group:
            continue
        bundles.append(
            RefillBundle(
                bundle_id=_bundle_id(
                    policy=policy,
                    backend=selected,
                    candidates=group,
                    fallback_of=fallback_of,
                ),
                backend=selected,
                projects_per_aedt=policy.projects_per_aedt,
                expected_rows=len(group),
                candidates=group,
                scheduling_profile=policy.profile_for(selected),
                policy_digest=policy.digest,
                provenance_digest=policy.provenance.digest,
                fallback_of=fallback_of,
            )
        )
    return tuple(bundles)


def task_submission_options(
    bundle: RefillBundle,
    policy: AttachRefillPolicy,
    *,
    candidate_index: int,
) -> dict[str, object]:
    """Return scheduler-client options for one logical project task."""

    if not 0 <= candidate_index < len(bundle.candidates):
        raise IndexError("candidate_index is outside the bundle")
    if bundle.policy_digest != policy.digest:
        raise ValueError("bundle does not belong to this policy")
    env = {
        "MFT_CONTROLLER_POLICY_SHA256": policy.digest,
        "MFT_DATA_CONTRACT_REVISION": policy.provenance.data_contract_revision,
        "MFT_PHYSICS_DATA_REVISION": policy.provenance.physics_data_revision,
        "MFT_CORE_LAMINATION_FACTOR": str(
            policy.provenance.core_lamination_factor
        ),
        "MFT_AEDT_BUNDLE_ID": bundle.bundle_id,
        "MFT_AEDT_BUNDLE_EXPECTED_ROWS": str(bundle.expected_rows),
        "MFT_AEDT_BUNDLE_PROJECT_INDEX": str(candidate_index),
        "MFT_PROJECTS_PER_AEDT": str(policy.projects_per_aedt),
        "MFT_EXPECTED_ROWS": "1",
    }
    if bundle.backend == POOLED:
        env["MFT_AEDT_SHARED_CANARY"] = "1"
    return {
        "aedt_backend": bundle.backend,
        "scheduling_profile": bundle.scheduling_profile,
        "submission_env": env,
        "dedupe_scope": policy.provenance.digest,
        "expected_rows": 1,
        "bundle_id": bundle.bundle_id,
        "bundle_expected_rows": bundle.expected_rows,
    }


def reconcile_failed_bundle(
    bundle: RefillBundle,
    *,
    task_ids: Sequence[int],
    task_statuses: Mapping[int, str],
    accepted_row_task_ids: Iterable[int],
    policy: AttachRefillPolicy,
) -> dict[str, object]:
    """Plan a non-destructive standalone replacement for missing rows.

    Completed and collected rows are retained.  Failed/missing members are
    reissued as new standalone candidates by the caller.  This function never
    returns cancellation authority and never disables unrelated pooled
    bundles.
    """

    if len(task_ids) != bundle.expected_rows or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must map one-to-one to bundle candidates")
    if any(isinstance(task_id, bool) or not isinstance(task_id, int) for task_id in task_ids):
        raise ValueError("task_ids are invalid")
    accepted = set(accepted_row_task_ids)
    if not accepted.issubset(set(task_ids)):
        raise ValueError("accepted row task is outside this bundle")
    normalized = {task_id: str(task_statuses.get(task_id) or "") for task_id in task_ids}
    terminal = all(status in TERMINAL_TASK_STATES for status in normalized.values())
    failed = [
        task_id
        for task_id, status in normalized.items()
        if status in FAILURE_TASK_STATES
        or (status == "completed" and task_id not in accepted)
    ]
    missing_candidate_indices = [
        index for index, task_id in enumerate(task_ids) if task_id not in accepted
    ]
    missing_rows = bundle.expected_rows - len(accepted)
    if bundle.backend != POOLED or not terminal or missing_rows <= 0:
        action = "wait" if not terminal else "none"
    else:
        action = "submit_standalone_fallback"
    return {
        "bundle_id": bundle.bundle_id,
        "backend": bundle.backend,
        "terminal": terminal,
        "failed_task_ids": sorted(failed),
        "accepted_rows": len(accepted),
        "missing_rows": missing_rows,
        "missing_candidate_indices": missing_candidate_indices,
        "action": action,
        "fallback_backend": (
            policy.failed_bundle_fallback
            if action == "submit_standalone_fallback"
            else None
        ),
        "fallback_expected_rows": missing_rows if action == "submit_standalone_fallback" else 0,
        "cancel_task_ids": [],
        "affects_other_bundles": False,
    }
