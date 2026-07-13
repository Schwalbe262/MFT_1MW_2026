"""Attach-aware planning layer for the reviewed 300-maintenance controller.

This executable is intentionally read-only.  It produces the exact bundle and
submission-option manifest that the reviewed mature controller consumes while
holding its existing campaign mutation lock and immutable cycle journal.  The
separation lets the pool policy be reviewed/deployed independently without
creating a second writer or cancelling the live standalone fleet.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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
        reconcile_failed_bundle,
        task_submission_options,
    )


POLICY_SCHEMA = "mft-attach-aware-refill-policy-v1"
PLAN_SCHEMA = "mft-attach-aware-refill-plan-v1"


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
        if self.policy.primary_backend == POOLED and gate["eligible"]:
            selected_backend = POOLED
            backend_reason = "pool_gate_passed"
        elif self.policy.primary_backend == POOLED:
            # No cancellation and no waiting gap: this cycle alone reuses the
            # proven standalone path.  A later cycle may re-enter pooled mode
            # after the independently managed pool is healthy again.
            selected_backend = STANDALONE
            backend_reason = "pool_unavailable_standalone_fallback"
        else:
            selected_backend = STANDALONE
            backend_reason = "standalone_selected"
        bundles = make_refill_bundles(
            selected_candidates, self.policy, backend=selected_backend
        )
        pooled_projects = (
            sum(bundle.expected_rows for bundle in bundles)
            if selected_backend == POOLED
            else 0
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
            "scheduling_profile": self.policy.profile_for(selected_backend),
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


def _load_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--pool-status")
    parser.add_argument("--active-project-tasks", required=True, type=int)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    policy_payload = _load_json(args.policy)
    if not isinstance(policy_payload, Mapping):
        raise ValueError("policy JSON must be an object")
    candidate_payload = _load_json(args.candidates)
    if not isinstance(candidate_payload, list):
        raise ValueError("candidate JSON must be a list")
    status_payload = _load_json(args.pool_status) if args.pool_status else None
    if status_payload is not None and not isinstance(status_payload, Mapping):
        raise ValueError("pool status JSON must be an object")
    coordinator = AttachAwareRefillCoordinator(load_policy(policy_payload))
    plan = coordinator.plan_cycle(
        active_project_tasks=args.active_project_tasks,
        candidates=tuple(ProjectCandidate(**item) for item in candidate_payload),
        pool_status=status_payload,
    )
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
