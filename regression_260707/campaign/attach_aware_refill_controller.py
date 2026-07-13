"""Attach-aware rolling refill controller for the reviewed MFT campaign.

``plan`` performs only scheduler GET requests (plus local generation-state
initialisation).  ``run`` is the sole scheduler mutation path and requires an
exact generation acknowledgement before it can submit an idempotent rolling
refill.  Neither mode has scheduler cancellation authority.

The pure policy/coordinator APIs remain usable without the live controller
shell; ``offline`` retains the original file-to-file planner interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
STATE_SCHEMA = "mft-restart-v3-controller-state-v1"
GENERATION_SCHEMA = "mft-restart-v3-generation-v1"
DEFAULT_STATE_PATH = HERE / "restart_v3_controller_state.json"
DEFAULT_POLICY_PATH = HERE / "attach_refill_policy_canary_n2.json"
DEFAULT_CANDIDATE_SEED = 260_710
DEFAULT_CPUS = 4
DEFAULT_MEMORY_MB = 65_536
DEFAULT_TIMEOUT_SECONDS = 14_400
DEFAULT_LOOP_SECONDS = 60
PROFILE_PATH = VERIFY_ROOT / "profiles" / "standard.json"
PRODUCTION_SOLVER_REVISION = "dba903eb671e37642168afc5578b8e6a93e9c046"
PRODUCTION_LIBRARY_REVISION = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781"
PRODUCTION_PHYSICS_DATA_REVISION = "mft1mw-1k101-native-lamination-kf0p85-v3"
PRODUCTION_CORE_LAMINATION_FACTOR = 0.85
PRODUCTION_POOLED_FRACTION = 0.0
PRODUCTION_PROJECTS_PER_AEDT = 2
PRODUCTION_MAX_AEDT_SESSIONS = 250
# Ten legacy p02/p08 candidates plus the 25 fresh entries in the submitted
# restart-v3 pilot manifest occupy the first 35 valid Sobol candidates.
PRODUCTION_CANDIDATE_VALID_OFFSET = 35


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


def pool_gate(policy: AttachRefillPolicy, status: Mapping[str, object] | None) -> dict:
    """Authenticate the scheduler pool limits before selecting pooled mode."""

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
    summary = dict(status or {})
    config_value = summary.get("config")
    observed = (
        dict(config_value) if isinstance(config_value, Mapping) else summary
    )
    checks = {
        "operational": observed.get("operational") is True,
        "enabled": observed.get("enabled") is True,
        "validation_passed": observed.get("validation_passed") is True,
        "projects_per_aedt": (
            observed.get("projects_per_aedt") == policy.projects_per_aedt
        ),
        "max_aedt_sessions": (
            isinstance(observed.get("max_aedt_sessions"), int)
            and not isinstance(observed.get("max_aedt_sessions"), bool)
            and int(observed["max_aedt_sessions"]) >= policy.max_aedt_sessions
        ),
        "target_project_concurrency": (
            observed.get("target_project_concurrency")
            == policy.project_concurrency_target
        ),
    }
    return {
        "eligible": all(checks.values()),
        "reason": "ready" if all(checks.values()) else "pool_gate_failed",
        "checks": checks,
        "observed": observed,
        "latest_validation": summary.get("latest_validation"),
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
        "pooled_fraction": (
            policy.pooled_fraction == PRODUCTION_POOLED_FRACTION
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
    identity = {
        "schema": GENERATION_SCHEMA,
        "policy_digest": policy.digest,
        "provenance_digest": policy.provenance.digest,
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
    if not isinstance(submissions, list):
        raise RuntimeError("controller state submissions must be a list")
    names: set[str] = set()
    dedupe_keys: set[str] = set()
    task_ids: set[int] = set()
    committed_cursor = int(
        generation["identity"]["candidate_start_cursor"]  # type: ignore[index]
    )
    for expected_serial, item in enumerate(submissions, start=1):
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
            or item.get("serial") != expected_serial
            or item.get("candidate_cursor_before") != committed_cursor
            or isinstance(item.get("candidate_cursor_after"), bool)
            or not isinstance(item.get("candidate_cursor_after"), int)
            or int(item["candidate_cursor_after"]) <= committed_cursor
        ):
            raise RuntimeError("controller state contains an invalid submission")
        if (
            item["name"] in names
            or item["dedupe_key"] in dedupe_keys
            or int(item["task_id"]) in task_ids
        ):
            raise RuntimeError("controller state contains a duplicate submission")
        names.add(item["name"])
        dedupe_keys.add(item["dedupe_key"])
        task_ids.add(int(item["task_id"]))
        committed_cursor = int(item["candidate_cursor_after"])
    ledger_checks = {
        "state_revision": state["state_revision"] == len(submissions),
        "next_serial": state["next_serial"] == len(submissions) + 1,
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


def _read_live_scheduler(
    policy: AttachRefillPolicy,
    scheduler_url: str,
) -> tuple[dict[str, object], Mapping[str, object] | None, int]:
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
    snapshot = scheduler_client.project_submission_snapshot(
        [project],
        project_tasks,
        target,
        legacy_tasks=legacy_tasks,
        require_exact_project_cap=True,
    )
    query_count = 3
    pool_status: Mapping[str, object] | None = None
    if policy.primary_backend == POOLED and policy.pooled_fraction > 0:
        query_count += 1
        try:
            pool_payload = _response_json(
                scheduler_client.requests.get(
                    f"{scheduler_url}/api/aedt-pool", timeout=30
                ),
                "AEDT pool",
            )
            if isinstance(pool_payload, Mapping):
                pool_status = pool_payload
            else:
                pool_status = {
                    "operational": False,
                    "error": "scheduler AEDT pool response is not an object",
                }
        except Exception as exc:
            pool_status = {
                "operational": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return snapshot, pool_status, query_count


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


def _live_plan(
    policy: AttachRefillPolicy,
    state: Mapping[str, object],
    generation: Mapping[str, object],
    profile: Mapping[str, object],
    scheduler_url: str,
) -> dict[str, object]:
    snapshot, pool_status, query_count = _read_live_scheduler(policy, scheduler_url)
    active = int(snapshot["project_active"])
    deficit = logical_refill_deficit(
        target=policy.project_concurrency_target, active_project_tasks=active
    )
    candidates, records = _candidate_records(
        state, deficit, policy, generation, profile
    )
    plan = AttachAwareRefillCoordinator(policy).plan_cycle(
        active_project_tasks=active,
        candidates=candidates,
        pool_status=pool_status,
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
            "logical_project_deficit": deficit,
            "would_submit": len(records),
            "would_replace": len(records),
            "planned_actions": records,
            "cancel_task_ids": [],
            "mass_cancel_authorized": False,
        }
    )
    return plan


def _commit_submission(
    state: dict[str, object], action: Mapping[str, object], task_id: int
) -> None:
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
            "submitted_or_reconciled_at": _now(),
        }
    )
    state["submissions"] = submissions
    state["candidate_cursor"] = int(action["candidate_cursor_after"])
    state["next_serial"] = int(action["serial"]) + 1
    state["state_revision"] = int(state["state_revision"]) + 1
    state["updated_at"] = _now()


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
            plan = _live_plan(
                policy, state, generation, profile, scheduler_url
            )
            accepted: list[dict[str, object]] = []
            for action in plan["planned_actions"]:  # type: ignore[index]
                submission_env = dict(action["submission_env"])
                submission_env["MFT_PHYSICS_DATA_REVISION"] = (
                    policy.provenance.physics_data_revision
                )
                submission_env["MFT_CORE_LAMINATION_FACTOR"] = str(
                    policy.provenance.core_lamination_factor
                )
                task_id = scheduler_client.submit_verification(
                    name=str(action["name"]),
                    workdir=str(action["workdir"]),
                    params=dict(action["params"]),
                    profile=profile,
                    mem_mb=int(generation["identity"]["memory_mb"]),  # type: ignore[index]
                    cpus=int(generation["identity"]["cpus"]),  # type: ignore[index]
                    solver_revision=policy.provenance.solver_revision,
                    library_revision=policy.provenance.library_revision,
                    required_project_cap=policy.project_concurrency_target,
                    aedt_backend=str(action["backend"]),
                    scheduling_profile=str(action["scheduling_profile"]),
                    submission_env=submission_env,
                    dedupe_scope=str(generation["digest"]),
                )
                if task_id is None:
                    raise RuntimeError(
                        f"scheduler returned no task ID for {action['name']}"
                    )
                _commit_submission(state, action, int(task_id))
                _validate_state(state, generation, scheduler_url)
                _atomic_save_state(state_path, state)
                accepted.append(
                    {
                        "name": action["name"],
                        "task_id": int(task_id),
                        "backend": action["backend"],
                        "dedupe_key": action["dedupe_key"],
                    }
                )
    return {
        "mode": "run",
        "action": "rolling_refill_complete" if accepted else "no_refill_needed",
        "generation": dict(generation),
        "active_project_tasks_before": plan["active_project_tasks"],
        "logical_project_deficit_before": plan["logical_project_deficit"],
        "accepted_or_reconciled_count": len(accepted),
        "accepted_or_reconciled": accepted,
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
